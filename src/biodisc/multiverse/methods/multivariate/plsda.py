"""PLS-DA — Partial Least Squares Discriminant Analysis (Geladi & Kowalski 1986).

Classical workhorse multivariate-discrimination method for metabolomics.
Wraps ``sklearn.cross_decomposition.PLSRegression`` with one-hot-encoded
class labels — the standard PLS-DA construction (Wold 1973 generalised
to discriminant analysis by adding a categorical y).

What we report (NOT prediction metrics)
---------------------------------------
This method is a **feature-importance generator** for biomarker
discovery, not a classifier. We do NOT report AUC, accuracy, or
confusion matrix. We DO report:

  * **VIP** (Variable Importance in Projection) — Wold's per-feature
    importance score; the established PLS-DA biomarker-ranking metric.
  * **β regression coefficients** per class — direction + magnitude of
    each feature's contribution to discriminating that class.
  * **Permutation importance** — model-agnostic, less biased than VIP
    at small N (Strobl 2007); computed via sklearn.inspection.
  * **Bootstrap stability** — fraction of bootstrap resamples in which
    the feature ranks in the top-K (Nogueira 2018).
  * **Permutation p-value** — significance vs label-shuffle null.

Auto-derived parameters
-----------------------
* ``n_components``: ``K - 1`` where K is the number of classes (4 for
  our 5 cell lines). Theoretical max for class separation; FIXED, not
  CV-tuned, so no leakage. If n_samples - 1 < K-1, falls back to
  n_samples - 1.

References
----------
* Geladi P, Kowalski BR. 1986. *Anal. Chim. Acta* 185:1.
* Wold S, Sjöström M, Eriksson L. 2001. *Chemom. Intell. Lab. Syst.* 58:109.
* Filzmoser P, Liebmann B, Varmuza K. 2009. *J. Chemom.* 23:160 (component
  selection by repeated double CV — but we explicitly avoid CV-based
  tuning to prevent leakage; we fix n_components = K-1).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.multivariate_utils import (
    MultivariateResult,
    bootstrap_stability,
    permutation_test,
)
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)


def _build_xy(
    ctx: StageContext, biological_only: bool = True
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Extract (X, y, feature_ids, sample_ids, class_names) from context.

    X is samples × features (sklearn convention). y is integer class
    labels in 0..K-1.
    """
    if biological_only:
        bio_ids = ctx.samples.loc[
            ctx.samples["injection_type"] == "biological", "sample_id"
        ].tolist()
    else:
        bio_ids = ctx.samples["sample_id"].tolist()
    cols = [s for s in bio_ids if s in ctx.matrix.columns]
    if len(cols) < 4:
        raise ValueError(f"PLS-DA needs ≥ 4 samples; found {len(cols)} after filtering.")
    sub_matrix = ctx.matrix[cols]
    X = sub_matrix.T.to_numpy(dtype=float)  # samples × features

    sample_to_class = ctx.samples.set_index("sample_id")["cell_line"].astype(str).to_dict()
    classes = [sample_to_class[s] for s in cols]
    class_names = sorted(set(classes))
    class_to_int = {c: i for i, c in enumerate(class_names)}
    y = np.array([class_to_int[c] for c in classes])
    feature_ids = sub_matrix.index.astype(str).tolist()
    return X, y, feature_ids, cols, class_names


def _compute_vip(pls_model: Any, X: np.ndarray) -> np.ndarray:  # noqa: N803 — sklearn X convention
    """Variable Importance in Projection per feature.

    Standard formula (Wold 1995):
        VIP_j = sqrt(p * sum_a(SS_a * w_{ja}^2 / ||w_a||^2) / sum_a(SS_a))

    where:
        p = n_features
        SS_a = explained sum-of-squares of component a
        w_{ja} = weight of feature j on component a

    Returns array of shape (n_features,).
    """
    # weights ('x_weights_') shape: (n_features, n_components)
    w = pls_model.x_weights_
    # x_scores_ shape: (n_samples, n_components)
    t = pls_model.x_scores_
    # y_loadings_ shape: (n_targets, n_components)
    q = pls_model.y_loadings_
    n_features = w.shape[0]
    n_components = w.shape[1]

    # Per-component explained SS (in y): t.T @ t * q^2 (summed over targets)
    # See Chong & Jun 2005, Wold 1995
    ss_per_comp = np.zeros(n_components)
    for a in range(n_components):
        ss_per_comp[a] = float(t[:, a] @ t[:, a] * (q[:, a] ** 2).sum())
    total_ss = float(ss_per_comp.sum())
    if total_ss <= 0:
        return np.zeros(n_features)

    # Normalised squared weights per component
    w_norm_sq = (w**2) / np.maximum(((w**2).sum(axis=0))[None, :], 1e-12)
    # VIP² = p * sum_a (SS_a * w_norm_sq_ja) / total_ss
    vip_sq = n_features * (w_norm_sq * ss_per_comp[None, :]).sum(axis=1) / total_ss
    vip: np.ndarray = np.sqrt(np.maximum(vip_sq, 0.0))
    return vip


def _fit_plsda(
    X: np.ndarray,  # noqa: N803 — sklearn X convention
    y: np.ndarray,
    n_components: int,
    scale: bool = False,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Fit PLS-DA with one-hot y; return (model, vip, beta_per_class, y_onehot)."""
    from sklearn.cross_decomposition import PLSRegression

    n_classes = len(np.unique(y))
    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y] = 1.0

    pls = PLSRegression(n_components=n_components, scale=scale)
    pls.fit(X, y_onehot)
    vip = _compute_vip(pls, X)
    # β coefficients per feature per class: pls.coef_ has shape (n_targets, n_features) in sklearn ≥1.4
    beta = pls.coef_  # (n_classes, n_features)
    if beta.shape[0] != n_classes:
        beta = beta.T  # older sklearn convention
    return pls, vip, beta, y_onehot


def fit(
    ctx: StageContext,
    *,
    n_components: int | None = None,
    n_bootstrap: int = 100,
    n_permutation: int = 1000,
    seed: int = 42,
    biological_only: bool = True,
    scale: bool = False,
) -> MultivariateResult:
    """Fit PLS-DA and produce per-feature importance + stability + p-values.

    Parameters
    ----------
    n_components : int | None, default None (auto)
        When ``None``, fixed at ``min(K-1, n_samples-1)`` where K is the
        number of classes. NOT CV-tuned — no test set, no leakage.
    n_bootstrap : int, default 100
        Bootstrap resamples for stability assessment.
    n_permutation : int, default 1000
        Label permutations for p-value computation.
    seed : int, default 42
        Master RNG seed.
    biological_only : bool, default True
        Restrict to biological samples (drop QCs from feature ranking).
    scale : bool, default False
        Whether PLSRegression should auto-scale features. Default False
        because we already Pareto-scaled in Stage 16; double-scaling
        would distort.
    """
    t0 = time.time()
    X, y, feature_ids, _sample_ids, class_names = _build_xy(ctx, biological_only=biological_only)
    n_samples, n_features = X.shape
    n_classes = len(class_names)

    # Auto n_components: K-1 (theoretical max for class separation), capped at n-1
    chosen_nc = min(n_classes - 1, n_samples - 1) if n_components is None else int(n_components)
    chosen_nc = max(1, chosen_nc)

    log.info(
        "plsda_fit_start",
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        class_names=class_names,
        n_components=chosen_nc,
        n_components_auto=(n_components is None),
    )

    # Primary fit on full data
    pls, vip_obs, beta_obs, _ = _fit_plsda(X, y, chosen_nc, scale=scale)

    # Permutation importance (model-agnostic)
    from sklearn.inspection import permutation_importance

    # We need a "score" function. For an unsupervised feature-importance
    # ranking, use R² of the PLS regression on its own training data
    # — NOT a generalisation metric. Permutation importance just checks
    # how much each feature contributes to the model's TRAINING fit.
    perm_imp_result = permutation_importance(
        pls,
        X,
        y if pls.coef_.shape[0] == 1 else np.eye(n_classes)[y],
        n_repeats=10,
        random_state=seed,
        n_jobs=1,
    )
    perm_imp_obs = perm_imp_result.importances_mean

    # Bootstrap stability (using VIP as the importance metric)
    def vip_importance_fn(
        X_b: np.ndarray,  # noqa: N803 — sklearn X convention
        y_b: np.ndarray,
        _rng: np.random.Generator,
    ) -> np.ndarray:
        # Skip if fewer than 2 classes survive the resample
        if len(np.unique(y_b)) < 2:
            return np.zeros(n_features)
        try:
            _, v, _, _ = _fit_plsda(X_b, y_b, chosen_nc, scale=scale)
            return v
        except Exception:
            return np.zeros(n_features)

    stability = bootstrap_stability(
        importance_fn=vip_importance_fn,
        matrix_arr=X,
        labels=y,
        feature_ids=feature_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    # Permutation test (using VIP)
    perm_pvals = permutation_test(
        importance_fn=vip_importance_fn,
        matrix_arr=X,
        labels=y,
        observed_importance=vip_obs,
        feature_ids=feature_ids,
        n_permutation=n_permutation,
        seed=seed + 1,
    )

    # Build per-feature DataFrame
    table = pd.DataFrame({"feature_id": feature_ids})
    table["vip"] = vip_obs
    for i, cls in enumerate(class_names):
        table[f"beta_{cls}"] = beta_obs[i]
    table["abs_beta_max"] = np.abs(beta_obs).max(axis=0)
    table["perm_importance"] = perm_imp_obs
    table["stability_score"] = stability.stability_score
    table["perm_p_value"] = perm_pvals.p_value

    elapsed = time.time() - t0
    log.info(
        "plsda_fit_done",
        elapsed_s=round(elapsed, 1),
        n_features_high_vip=int((vip_obs >= 1.0).sum()),
        n_features_stable=int((stability.stability_score >= 0.8).sum()),
        n_features_p_lt_05=int((perm_pvals.p_value < 0.05).sum()),
    )

    return MultivariateResult(
        method_name="plsda",
        per_feature=table,
        params_used={
            "n_components": chosen_nc,
            "scale": scale,
            "biological_only": biological_only,
            "stability_top_k": stability.top_k,
        },
        runtime_seconds=elapsed,
        n_bootstrap=n_bootstrap,
        n_permutation=n_permutation,
    )


register(
    MethodSpec(
        name="plsda",
        callable=fit,
        description="PLS-DA feature-importance generator (VIP, β-coef, perm imp).",
        citation="Geladi & Kowalski 1986, Anal. Chim. Acta 185:1; Wold 2001",
        category="multivariate",
    )
)
