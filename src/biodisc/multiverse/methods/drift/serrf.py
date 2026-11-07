"""SERRF — Systematic Error Removal using Random Forest (Fan 2019).

Per-feature drift correction that uses a Random Forest to predict the
intensity of a target feature from (a) its chronological position and (b)
the intensities of the N most-correlated other features at that
injection. The RF is trained ONLY on QC samples (where biological
variation is minimal — anything left is technical drift); predictions
are then made for every sample to obtain the drift-estimate-per-sample.
The corrected intensity is `original × reference / drift_estimate`.

Why use co-varying features as predictors
-----------------------------------------
A feature that drifts because of slow column degradation will co-vary
with chemically similar features (sharing column-affinity behaviour);
those make excellent predictors. By contrast, simple QC-RSC only uses
position. SERRF typically achieves 10–20% better D-ratio than QC-RSC
on long sequences (Fan 2019, *Anal. Chem.*).

Algorithm
---------
For each feature ``f``:

  1. Compute correlation of ``f`` with every other feature, using only
     the QC samples (where biological variance is minimal).
  2. Pick the top ``n_correlated_features`` most-correlated others.
  3. Build training matrix X_train from QC samples only:
       columns = [position, intensity of correlated_1, …, correlated_N]
       target = intensity of f at QC samples.
  4. Train a small RandomForestRegressor.
  5. Build prediction matrix X_pred for ALL samples (QC + biological).
  6. Predict drift estimate for every sample.
  7. Corrected = original × median(QC f intensities) / drift_estimate.

Edge cases
----------
* Feature with constant or all-zero QC intensities → pass through.
* RF prediction ≤ 0 → clip to small positive floor.
* Fewer correlated features available than ``n_correlated_features`` →
  use whatever is available.

References
----------
* Fan S et al. 2019, *Anal. Chem.* 91:3590 — original SERRF paper.
* Han W & Li L 2022, *Briefings in Bioinformatics* 23:bbab535 — TIGER
  benchmark showing SERRF consistently 10–20% better than QC-RSC.
* Sánchez-Illana A et al. 2023, *Metabolites* 13:665 — recent benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)

_MIN_DRIFT_FLOOR = 1e-6


def _auto_n_correlated(n_features: int) -> int:
    """Data-aware default for n_correlated_features.

    Fan 2019 used 10 by default but called out that this should scale
    with feature density. The scaling: more features → more potential
    co-varying neighbours; cap at 30 to keep RF tractable.
    """
    return int(max(5, min(30, n_features // 200)))


def _auto_n_estimators(n_qc: int) -> int:
    """Data-aware default for RF n_estimators.

    With few QC samples (tight training set), more trees just memorise
    the training noise. With more QCs, larger ensembles help. Scale
    linearly with sqrt(n_qc) and clamp.
    """
    return int(max(20, min(100, 5 * int(np.sqrt(n_qc)) + 20)))


def _auto_max_depth(n_qc: int) -> int:
    """Data-aware tree-depth bound — prevents perfect QC memorisation.

    Heuristic: ``max(2, floor(log2(n_qc)))``. With log2 we ensure each
    leaf still sees a few QCs on average; depth never exceeds the
    information capacity of the QC training set.

    For n_qc=8 → depth=3; n_qc=16 → 4; n_qc=64 → 6.
    """
    return int(max(2, np.log2(max(n_qc, 2))))


def _auto_min_samples_leaf(n_qc: int) -> int:
    """Data-aware leaf-size bound — pairs with max_depth to limit overfit.

    Heuristic: ``max(1, n_qc // 8)``. For n_qc=16 → 2; n_qc=64 → 8.
    """
    return int(max(1, n_qc // 8))


def correct(
    ctx: StageContext,
    *,
    n_correlated_features: int | None = None,
    n_estimators: int | None = None,
    max_depth: int | None = None,
    min_samples_leaf: int | None = None,
    correlation_pool: str = "all",
    seed: int = 42,
) -> pd.DataFrame:
    """Apply SERRF (Random-Forest based) drift correction across all samples.

    Hardened version with overfitting safeguards:

    * ``max_depth`` defaults to ``log2(n_qc)`` to prevent the per-feature
      RF from perfectly memorising the (small) QC training set.
    * ``min_samples_leaf`` defaults to ``n_qc // 8`` for the same reason.
    * Correlation ranking now uses ALL biological+QC samples by default
      (47 vs 16) — ~3× better statistical power. Set
      ``correlation_pool="qc"`` to recover the original Fan 2019 spec.

    Parameters
    ----------
    n_correlated_features : int | None, default None (auto)
        Number of co-varying features used as RF predictors per target.
        When ``None``, ``max(5, min(30, n_features // 200))``.
    n_estimators : int | None, default None (auto)
        Number of trees in each per-feature RF. When ``None``,
        scales with sqrt(n_qc) and clamped to [20, 100].
    max_depth : int | None, default None (auto)
        Maximum tree depth. When ``None``, ``max(2, log2(n_qc))`` —
        bounds tree complexity to QC training-set size to prevent
        overfitting in the typical n_qc ∈ [8, 30] regime.
    min_samples_leaf : int | None, default None (auto)
        Minimum QCs per leaf. When ``None``, ``max(1, n_qc // 8)``.
    correlation_pool : str, default "all"
        Sample subset used to compute feature-feature correlation for
        neighbour selection. ``"all"`` uses biological+QC samples (more
        statistical power); ``"qc"`` uses QCs only (Fan 2019 original).
    seed : int, default 42
        Deterministic random_state for the RF.
    """
    from sklearn.ensemble import RandomForestRegressor

    matrix = ctx.matrix
    if matrix.isna().any().any():
        raise ValueError("SERRF requires an imputed matrix (no NaN).")

    positions = ctx.sample_positions()
    qc_ids = ctx.qc_sample_ids()
    if len(qc_ids) < 5:
        raise ValueError(
            f"SERRF needs ≥ 5 QC injections to train per-feature RFs; "
            f"found {len(qc_ids)}. Use QC-RSC instead, or skip drift."
        )

    n_features = matrix.shape[0]
    chosen_n_corr = (
        _auto_n_correlated(n_features)
        if n_correlated_features is None
        else n_correlated_features
    )
    chosen_n_corr = min(chosen_n_corr, n_features - 1)
    chosen_n_estimators = (
        _auto_n_estimators(len(qc_ids)) if n_estimators is None else n_estimators
    )
    chosen_max_depth = _auto_max_depth(len(qc_ids)) if max_depth is None else max_depth
    chosen_min_leaf = (
        _auto_min_samples_leaf(len(qc_ids)) if min_samples_leaf is None else min_samples_leaf
    )

    # Pre-compute QC submatrix and column-position vectors once
    qc_matrix = matrix[qc_ids]
    all_position_vec = positions[matrix.columns].to_numpy(dtype=float).reshape(-1, 1)
    qc_position_vec = positions[qc_ids].to_numpy(dtype=float).reshape(-1, 1)

    # Per-feature correlation pool — use ALL samples by default (more power)
    if correlation_pool == "qc":
        corr_source = qc_matrix
    elif correlation_pool == "all":
        corr_source = matrix
    else:
        raise ValueError(f"correlation_pool must be 'all' or 'qc', got {correlation_pool!r}")
    corr_values = corr_source.to_numpy(dtype=np.float32)
    full_corr = np.corrcoef(corr_values)
    np.fill_diagonal(full_corr, np.nan)

    # OOB diagnostic: only meaningful when n_qc reasonable; skip otherwise.
    use_oob = len(qc_ids) >= 8
    oob_scores: list[float] = []

    out = matrix.copy()
    n_pass_through = 0
    n_corrected = 0

    feature_index = matrix.index.tolist()
    for i, fid in enumerate(feature_index):
        qc_target = qc_matrix.loc[fid].to_numpy(dtype=float)
        if float(np.var(qc_target, ddof=0)) == 0:
            n_pass_through += 1
            continue

        corr_row = full_corr[i]
        ranking = np.argsort(-np.nan_to_num(np.abs(corr_row)))[:chosen_n_corr]
        correlated_ids = [feature_index[j] for j in ranking]

        X_train_corr = qc_matrix.loc[correlated_ids].to_numpy(dtype=float).T
        X_train = np.hstack([qc_position_vec, X_train_corr])
        y_train = qc_target

        X_pred_corr = matrix.loc[correlated_ids].to_numpy(dtype=float).T
        X_pred = np.hstack([all_position_vec, X_pred_corr])

        rf = RandomForestRegressor(
            n_estimators=chosen_n_estimators,
            random_state=seed,
            n_jobs=1,
            max_depth=chosen_max_depth,
            min_samples_leaf=chosen_min_leaf,
            oob_score=use_oob,
            bootstrap=True,
        )
        rf.fit(X_train, y_train)
        if use_oob and hasattr(rf, "oob_score_") and np.isfinite(rf.oob_score_):
            oob_scores.append(float(rf.oob_score_))
        drift_estimate = rf.predict(X_pred)
        drift_estimate = np.maximum(drift_estimate, _MIN_DRIFT_FLOOR)

        reference = float(np.median(qc_target))
        out.loc[fid] = matrix.loc[fid].to_numpy(dtype=float) * reference / drift_estimate
        n_corrected += 1

    out = out.clip(lower=0.0)

    median_oob = float(np.median(oob_scores)) if oob_scores else float("nan")
    log.info(
        "serrf_drift_correction_done",
        n_features=len(matrix),
        n_samples=matrix.shape[1],
        n_qc=len(qc_ids),
        n_corrected=n_corrected,
        n_pass_through_zero_var=n_pass_through,
        n_correlated_features=chosen_n_corr,
        n_correlated_features_auto=(n_correlated_features is None),
        n_estimators=chosen_n_estimators,
        n_estimators_auto=(n_estimators is None),
        max_depth=chosen_max_depth,
        max_depth_auto=(max_depth is None),
        min_samples_leaf=chosen_min_leaf,
        min_samples_leaf_auto=(min_samples_leaf is None),
        correlation_pool=correlation_pool,
        median_oob_r2=median_oob,
        oob_enabled=use_oob,
    )
    if use_oob and median_oob < 0.3 and not np.isnan(median_oob):
        log.warning(
            "serrf_low_oob_r2",
            median_oob_r2=median_oob,
            hint="RF generalisation poor — drift correction may be unreliable",
        )
    return out


def loqo_predict(
    ctx: StageContext,
    held_qc: str,
    *,
    n_correlated_features: int | None = None,
    n_estimators: int | None = None,
    max_depth: int | None = None,
    min_samples_leaf: int | None = None,
    correlation_pool: str = "all",
    seed: int = 42,
) -> np.ndarray:
    """Predict held-out QC's per-feature intensity by re-fitting SERRF on remaining QCs.

    The honest LOQO test for SERRF: re-train each per-feature RF on the
    QCs MINUS the held one, then predict at the held QC's row of
    (position, correlated-features-at-held-QC). The residual against
    the true held-QC intensity is the out-of-sample drift-prediction
    error — the metric that doesn't lie about overfitting.
    """
    from sklearn.ensemble import RandomForestRegressor

    matrix = ctx.matrix
    positions = ctx.sample_positions()
    qc_ids = ctx.qc_sample_ids()
    kept_qcs = [q for q in qc_ids if q != held_qc]
    if len(kept_qcs) < 4:
        return matrix[kept_qcs].mean(axis=1).to_numpy(dtype=float)

    n_features = matrix.shape[0]
    chosen_n_corr = (
        _auto_n_correlated(n_features) if n_correlated_features is None else n_correlated_features
    )
    chosen_n_corr = min(chosen_n_corr, n_features - 1)
    chosen_n_estimators = (
        _auto_n_estimators(len(kept_qcs)) if n_estimators is None else n_estimators
    )
    chosen_max_depth = (
        _auto_max_depth(len(kept_qcs)) if max_depth is None else max_depth
    )
    chosen_min_leaf = (
        _auto_min_samples_leaf(len(kept_qcs))
        if min_samples_leaf is None
        else min_samples_leaf
    )

    kept_matrix = matrix[kept_qcs]
    held_pos = float(positions[held_qc])
    kept_pos_vec = positions[kept_qcs].to_numpy(dtype=float).reshape(-1, 1)
    held_pos_arr = np.array([[held_pos]], dtype=float)

    if correlation_pool == "qc":
        corr_source = kept_matrix
    elif correlation_pool == "all":
        # Use the FULL ctx matrix MINUS the held QC for correlation ranking
        corr_source = matrix.drop(columns=[held_qc])
    else:
        raise ValueError(f"correlation_pool must be 'all' or 'qc', got {correlation_pool!r}")
    corr_values = corr_source.to_numpy(dtype=np.float32)
    full_corr = np.corrcoef(corr_values)
    np.fill_diagonal(full_corr, np.nan)

    feature_index = matrix.index.tolist()
    predictions = np.zeros(n_features, dtype=float)

    for i, fid in enumerate(feature_index):
        qc_target = kept_matrix.loc[fid].to_numpy(dtype=float)
        if float(np.var(qc_target, ddof=0)) == 0:
            predictions[i] = float(np.mean(qc_target))
            continue

        corr_row = full_corr[i]
        ranking = np.argsort(-np.nan_to_num(np.abs(corr_row)))[:chosen_n_corr]
        correlated_ids = [feature_index[j] for j in ranking]

        X_train_corr = kept_matrix.loc[correlated_ids].to_numpy(dtype=float).T
        X_train = np.hstack([kept_pos_vec, X_train_corr])
        y_train = qc_target

        # At held position, take the correlated features' values from the
        # FULL ctx (the held QC has actual intensity values for those features)
        held_corr_values = matrix.loc[correlated_ids, held_qc].to_numpy(dtype=float)
        X_pred = np.hstack([held_pos_arr, held_corr_values.reshape(1, -1)])

        rf = RandomForestRegressor(
            n_estimators=chosen_n_estimators,
            random_state=seed,
            n_jobs=1,
            max_depth=chosen_max_depth,
            min_samples_leaf=chosen_min_leaf,
            bootstrap=True,
        )
        rf.fit(X_train, y_train)
        predictions[i] = float(rf.predict(X_pred)[0])

    return np.asarray(np.maximum(predictions, _MIN_DRIFT_FLOOR), dtype=float)


register(
    MethodSpec(
        name="serrf",
        callable=correct,
        description="Random-Forest drift correction using QC anchors + correlated features as predictors.",
        citation="Fan et al. 2019, Anal. Chem. 91:3590",
        category="drift",
    )
)
