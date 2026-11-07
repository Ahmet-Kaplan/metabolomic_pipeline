"""QC-RSC — Quality Control Robust Spline Correction (Kirwan 2013).

Per-feature drift correction that fits a smooth cubic spline through the
QC-pool intensities measured across the chronological run, then divides
every sample's intensity by the predicted drift at that sample's
position. The result: QC trend flattened, biological samples rescaled to
the same effective baseline.

Algorithm
---------
For each feature:

  1. Collect QC intensities at their chronological positions.
  2. Fit a smoothed cubic spline on (qc_position, qc_intensity).
     ``smoothing_factor`` × n × var(intensities) controls smoothness:
     0 → exact interpolation through every QC (no smoothing — overfit
     to single bad QC); high → near-linear trend.
  3. Evaluate the spline at every sample's position to get the predicted
     drift estimate at that sample.
  4. Compute reference = median of QC intensities (preserves overall
     scale; avoids inflating the signal post-correction).
  5. Corrected intensity = original × reference / drift_estimate.

Edge cases
----------
* Feature with all-zero or constant QC intensities → no drift to correct;
  pass through unchanged (with a log entry).
* Spline fit failure (rare, e.g. fewer than 4 unique QC positions) →
  fall back to median-scaling: divide by mean QC, multiply by reference.
* Drift estimate ≤ 0 anywhere → clip to a small positive epsilon to
  avoid division by zero / negative correction.

References
----------
* Kirwan JA et al. 2013, *Analyst* 138:5337 — original QC-RSC paper.
* Broadhurst DI et al. 2018, *Metabolomics* 14:72 — mQACC framework
  recommending QC-RSC for single-batch acquisition.
* Sánchez-Illana A et al. 2023, *Metabolites* 13:665 — recent benchmark
  confirming QC-RSC remains transparent gold standard for n_QC ≥ 6.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)

_MIN_DRIFT_FLOOR = 1e-6  # absolute floor: avoid division by zero post-spline
_DEFAULT_CORRECTION_CAP = 10.0  # max ratio of correction multiplier — see docstring


def _auto_smoothing_factor(n_qc: int) -> float:
    """Data-aware default for the spline smoothing knob.

    Higher smoothing_factor → smoother (closer to linear) trend.
    With more QCs we can afford less smoothing (the data is more
    self-regularising). With fewer QCs we need more smoothing to avoid
    overfitting individual noisy QC points.
    """
    # n_qc=6 → 1.0; n_qc=16 → 0.5; n_qc=30+ → 0.3 (clamped)
    return float(max(0.3, min(1.0, 6.0 / n_qc)))


def _evaluate_spline_clamped(
    spline: Any,  # scipy.interpolate.UnivariateSpline (no stubs)
    positions: np.ndarray,
    qc_min: float,
    qc_max: float,
) -> np.ndarray:
    """Evaluate spline at positions, clamping to [qc_min, qc_max] interval.

    Why: ``UnivariateSpline`` extrapolates polynomially outside the
    fitted-x interval, which can produce wild values for samples whose
    chronological position lies before the first QC or after the last
    (e.g. start/end blanks, dilution samples). Clamping evaluates the
    spline at the boundary instead, giving a defensible "drift = the
    nearest measured QC's predicted value" extrapolation.
    """
    clamped = np.clip(positions, qc_min, qc_max)
    return np.asarray(spline(clamped), dtype=float)


def correct(
    ctx: StageContext,
    *,
    smoothing_factor: float | None = None,
    spline_order: int = 3,
    correction_cap: float = _DEFAULT_CORRECTION_CAP,
) -> pd.DataFrame:
    """Apply QC-RSC drift correction across all samples.

    Parameters
    ----------
    smoothing_factor : float | None, default None (auto)
        Multiplier on ``n_qc * var(qc_intensities)`` for the scipy spline
        smoothing parameter ``s``. When ``None``, computed via
        ``_auto_smoothing_factor(n_qc)``.
    spline_order : int, default 3
        Cubic by default (matches the literal "QC-RSC" spec). Use 1 for
        piecewise-linear trends if your QCs are very sparse.
    correction_cap : float, default 10.0
        Maximum ratio of correction multiplier ``reference / drift_estimate``.
        Without this, when the cubic spline overshoots near zero (which
        can happen between sparse QC anchors), the correction can inflate
        feature values by 10⁶–10⁸ × — silently destroying biology in
        downstream stages (MSTUS, stats). The cap clamps drift estimates
        to ``[reference / cap, reference × cap]`` per feature, so the
        correction multiplier stays in ``[1/cap, cap]``. Drift outside
        this band is treated as method failure (likely spline overshoot,
        not real drift). The default 10.0 means no feature can be scaled
        by more than 10× in either direction — generous for genuine drift,
        prohibitive for spline pathologies.
    """
    from scipy.interpolate import UnivariateSpline

    matrix = ctx.matrix
    if matrix.isna().any().any():
        raise ValueError(
            "QC-RSC requires an imputed matrix (no NaN). Run Stage 11 "
            "imputation first."
        )

    positions = ctx.sample_positions()  # Series[sample_id → position]
    qc_ids = ctx.qc_sample_ids()
    if len(qc_ids) < 4:
        raise ValueError(
            f"QC-RSC needs ≥ 4 QC injections; found {len(qc_ids)}. "
            f"Use a different drift method or skip drift correction."
        )

    qc_positions = positions[qc_ids].to_numpy()
    all_positions = positions[matrix.columns].to_numpy()

    # Sort QC positions ascending (required by UnivariateSpline)
    qc_order = np.argsort(qc_positions)
    qc_positions_sorted = qc_positions[qc_order]
    qc_ids_sorted = [qc_ids[i] for i in qc_order]
    qc_min_pos = float(qc_positions_sorted[0])
    qc_max_pos = float(qc_positions_sorted[-1])

    # How many sample positions are outside the QC interval?
    n_extrapolated = int(((all_positions < qc_min_pos) | (all_positions > qc_max_pos)).sum())

    # Auto-default smoothing
    chosen_smooth = (
        _auto_smoothing_factor(len(qc_ids)) if smoothing_factor is None else smoothing_factor
    )

    out = matrix.copy()
    n_pass_through = 0
    n_fallback_median = 0
    n_corrected = 0
    n_features_capped = 0  # drift estimates clamped by correction_cap
    n_cells_capped = 0  # individual cells in inflated regions

    for fid in matrix.index:
        qc_intensities = matrix.loc[fid, qc_ids_sorted].to_numpy(dtype=float)
        qc_var = float(np.var(qc_intensities, ddof=0))

        # Pass through if no variation among QCs (no drift to correct)
        if qc_var == 0 or not np.isfinite(qc_var):
            n_pass_through += 1
            continue

        s_param = chosen_smooth * len(qc_intensities) * qc_var
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # scipy emits chatty FutureWarnings
                spline = UnivariateSpline(
                    qc_positions_sorted,
                    qc_intensities,
                    k=min(spline_order, len(qc_ids) - 1),
                    s=s_param,
                )
            drift_estimate = _evaluate_spline_clamped(
                spline, all_positions, qc_min_pos, qc_max_pos
            )
        except Exception as exc:  # rare — fallback to median scaling
            log.warning("qc_rsc_spline_fit_failed", feature=fid, error=str(exc))
            drift_estimate = np.full_like(all_positions, np.mean(qc_intensities), dtype=float)
            n_fallback_median += 1

        reference = float(np.median(qc_intensities))

        # CRITICAL: cap drift_estimate to [reference / cap, reference × cap].
        # This prevents runaway correction when the spline overshoots near
        # zero (or inflates above sensible). Without this, late-cycle
        # samples can get features inflated by 10⁶–10⁸×, destroying biology
        # downstream (discovered Apr-2026 via MSTUS guardrail trigger).
        drift_lower = max(reference / correction_cap, _MIN_DRIFT_FLOOR)
        drift_upper = reference * correction_cap
        capped = np.clip(drift_estimate, drift_lower, drift_upper)
        n_capped_this = int((capped != drift_estimate).sum())
        if n_capped_this > 0:
            n_features_capped += 1
            n_cells_capped += n_capped_this
        out.loc[fid] = matrix.loc[fid].to_numpy(dtype=float) * reference / capped
        n_corrected += 1

    # Clip at zero for safety (spline could under-shoot near edges)
    out = out.clip(lower=0.0)

    log.info(
        "qc_rsc_drift_correction_done",
        n_features=len(matrix),
        n_samples=matrix.shape[1],
        n_qc=len(qc_ids),
        n_corrected=n_corrected,
        n_pass_through_zero_var=n_pass_through,
        n_fallback_median=n_fallback_median,
        n_samples_extrapolated_to_boundary=n_extrapolated,
        qc_position_range=(qc_min_pos, qc_max_pos),
        smoothing_factor=chosen_smooth,
        smoothing_factor_auto=(smoothing_factor is None),
        spline_order=spline_order,
        correction_cap=correction_cap,
        n_features_capped=n_features_capped,
        n_cells_capped=n_cells_capped,
    )
    if n_features_capped > 0:
        log.warning(
            "qc_rsc_correction_capped",
            n_features_capped=n_features_capped,
            n_cells_capped=n_cells_capped,
            correction_cap=correction_cap,
            hint="spline overshoot detected — drift estimate clipped to prevent runaway scaling",
        )
    return out


def loqo_predict(
    ctx: StageContext,
    held_qc: str,
    *,
    smoothing_factor: float | None = None,
    spline_order: int = 3,
    correction_cap: float = _DEFAULT_CORRECTION_CAP,
) -> np.ndarray:
    """Predict held-out QC intensities by re-fitting the spline on remaining QCs.

    The honest LOQO test: refit the cubic spline using all QCs except
    ``held_qc``, then evaluate at ``held_qc``'s position. If the QC-RSC
    fit generalises beyond the training QCs, the predicted intensity at
    ``held_qc``'s position should be close to its true measured value.
    """
    from scipy.interpolate import UnivariateSpline

    matrix = ctx.matrix
    positions = ctx.sample_positions()
    qc_ids = ctx.qc_sample_ids()
    kept_qcs = [q for q in qc_ids if q != held_qc]
    if len(kept_qcs) < 4:
        # Can't fit cubic with < 4 points — fall back to mean of kept
        kept_intensities = matrix[kept_qcs].mean(axis=1).to_numpy(dtype=float)
        return kept_intensities

    qc_pos = positions[kept_qcs].to_numpy(dtype=float)
    held_pos = float(positions[held_qc])
    qc_min, qc_max = float(qc_pos.min()), float(qc_pos.max())
    held_pos_clamped = float(np.clip(held_pos, qc_min, qc_max))

    qc_order = np.argsort(qc_pos)
    qc_pos_sorted = qc_pos[qc_order]
    kept_qcs_sorted = [kept_qcs[i] for i in qc_order]

    chosen_smooth = (
        _auto_smoothing_factor(len(kept_qcs)) if smoothing_factor is None else smoothing_factor
    )

    predictions = np.zeros(matrix.shape[0], dtype=float)
    for i, fid in enumerate(matrix.index):
        qc_int = matrix.loc[fid, kept_qcs_sorted].to_numpy(dtype=float)
        qc_var = float(np.var(qc_int, ddof=0))
        if qc_var == 0:
            predictions[i] = float(np.mean(qc_int))
            continue
        s_param = chosen_smooth * len(qc_int) * qc_var
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spline = UnivariateSpline(
                    qc_pos_sorted,
                    qc_int,
                    k=min(spline_order, len(kept_qcs) - 1),
                    s=s_param,
                )
            raw_pred = float(spline(held_pos_clamped))
        except Exception:
            raw_pred = float(np.mean(qc_int))
        # Same cap as correct(): prediction must be in [ref/cap, ref*cap]
        reference = float(np.median(qc_int))
        lower = max(reference / correction_cap, _MIN_DRIFT_FLOOR)
        upper = reference * correction_cap
        predictions[i] = float(np.clip(raw_pred, lower, upper))
    return np.asarray(predictions, dtype=float)


register(
    MethodSpec(
        name="qc_rsc",
        callable=correct,
        description="Cubic-spline drift correction anchored on QC injections — transparent gold standard.",
        citation="Kirwan et al. 2013, Analyst 138:5337; Broadhurst 2018",
        category="drift",
    )
)
