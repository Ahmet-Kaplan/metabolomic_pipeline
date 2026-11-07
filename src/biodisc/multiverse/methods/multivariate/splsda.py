"""sPLS-DA — sparse Partial Least Squares Discriminant Analysis (Lê Cao 2011).

Hard-sparsity variant of PLS-DA. At each NIPALS component the weight vector
is hard-thresholded to keep only the ``keep_x`` features with the largest
absolute weight; the rest are zeroed out. ``selected`` (boolean) per feature
is the union of those top-K sets across all components — the method's
nominated biomarker shortlist.

What we report (NOT prediction metrics)
---------------------------------------
Same biomarker-discovery contract as PLS-DA:

  * **selected** — bool, ``True`` iff feature is in the top-K of any component
  * **VIP** — Variable Importance in Projection (computed on the sparse model)
  * **β regression coefficients** per class
  * **bootstrap stability** — fraction of bootstrap resamples in which the
    feature ranks in the top-K (Nogueira 2018)
  * **permutation p-value** — significance vs label-shuffle null

We do NOT report ``perm_importance`` — sklearn's ``permutation_importance``
relies on a ``predict`` interface that's expensive to evaluate on this
custom NIPALS model and the value duplicates VIP semantics for hard-sparse
methods. The column is kept (NaN) so the multivariate result schema is
uniform across methods.

Auto-derived parameters
-----------------------
* ``n_components``: ``K - 1`` where K = number of classes (FIXED, not CV-tuned).
* ``keep_x`` per component: ``max(10, min(50, n_features // 100))``.

References
----------
* Lê Cao K-A, Boitard S, Besse P. 2011. *J. Chemom.* 25:151. — sparse PLS-DA.
* Lê Cao K-A, Rossouw D, Robert-Granié C, Besse P. 2008. *Stat. Appl. Genet.
  Mol. Biol.* 7:35. — sparse PLS theory.
* Filzmoser P, Liebmann B, Varmuza K. 2009. *J. Chemom.* 23:160. — variable
  selection in PLS-DA (motivates the hard-thresholding choice).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.methods.multivariate.plsda import _build_xy, _compute_vip
from biodisc.multiverse.multivariate_utils import (
    MultivariateResult,
    bootstrap_stability,
    permutation_test,
)
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)


def _auto_keep_x(n_features: int) -> int:
    """Sparsity per component: 10 ≤ keep_x ≤ 50, scaling with n_features.

    Floor 10 (need enough to compute meaningful loadings); ceiling 50 (a
    biomarker shortlist of more than 50 per component defeats the purpose
    of sparsity); else ``n_features // 100``.
    """
    return int(max(10, min(50, n_features // 100)))


@dataclass
class _SparsePLSModel:
    """Lightweight model duck-typed to sklearn's PLSRegression interface.

    Exposes the attributes ``_compute_vip`` reads (``x_weights_``,
    ``x_scores_``, ``y_loadings_``) plus ``coef_`` for β coefficients —
    so we can reuse the same VIP machinery as classical PLS-DA.
    """

    x_weights_: np.ndarray
    x_loadings_: np.ndarray
    x_scores_: np.ndarray
    y_loadings_: np.ndarray
    coef_: np.ndarray
    selected_per_comp: list[np.ndarray] = field(default_factory=list)


def _nipals_sparse(
    X: np.ndarray,  # noqa: N803 — sklearn X convention
    Y: np.ndarray,  # noqa: N803 — sklearn Y convention (one-hot targets)
    n_components: int,
    keep_x: int,
    tol: float = 1e-9,
    max_iter: int = 500,
    seed: int = 0,
) -> _SparsePLSModel:
    """NIPALS sPLS with hard top-K sparsification per component.

    Outer loop over components a:
      1. NIPALS inner loop until convergence: build w from X.T @ u, then
         hard-threshold to keep top-K by |w|, normalise, compute scores
         t and y-side q, update u.
      2. Compute x-loadings p, deflate residuals.

    Returns a model object with sklearn-PLS-compatible attributes.
    """
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]
    x_resid = X.astype(float).copy()
    y_resid = Y.astype(float).copy()

    weights = np.zeros((n_features, n_components))
    loadings = np.zeros((n_features, n_components))
    scores = np.zeros((n_samples, n_components))
    y_load = np.zeros((n_targets, n_components))
    selected_per_comp: list[np.ndarray] = []

    rng = np.random.default_rng(seed)

    for a in range(n_components):
        # Initialise u with column of Y_resid having highest variance
        col_var = y_resid.var(axis=0)
        u = (
            y_resid[:, int(np.argmax(col_var))].copy()
            if col_var.max() > 1e-12
            else rng.standard_normal(n_samples)
        )

        w = np.zeros(n_features)
        t = np.zeros(n_samples)
        for _ in range(max_iter):
            w_new = x_resid.T @ u
            wn = float(np.linalg.norm(w_new))
            if wn < 1e-12:
                break
            w_new = w_new / wn

            # Hard-sparsify: keep top-K by absolute value, zero rest
            k = min(keep_x, n_features)
            if k < n_features:
                cutoff = np.partition(np.abs(w_new), n_features - k)[n_features - k]
                w_new = np.where(np.abs(w_new) > cutoff, w_new, 0.0)
            wn = float(np.linalg.norm(w_new))
            if wn < 1e-12:
                break
            w_new = w_new / wn

            t_new = x_resid @ w_new
            t_norm_sq = float(t_new @ t_new)
            if t_norm_sq < 1e-12:
                break
            q = y_resid.T @ t_new / t_norm_sq
            u_new = y_resid @ q

            if np.linalg.norm(u_new - u) < tol and np.linalg.norm(w_new - w) < tol:
                w, t, u = w_new, t_new, u_new
                break
            w, t, u = w_new, t_new, u_new

        if np.linalg.norm(w) < 1e-12:
            # Component degenerated — no further useful directions
            break

        t_norm_sq = float(t @ t)
        p = x_resid.T @ t / max(t_norm_sq, 1e-12)
        q = y_resid.T @ t / max(t_norm_sq, 1e-12)

        # Deflate residuals
        x_resid = x_resid - np.outer(t, p)
        y_resid = y_resid - np.outer(t, q)

        weights[:, a] = w
        loadings[:, a] = p
        scores[:, a] = t
        y_load[:, a] = q
        selected_per_comp.append(np.where(np.abs(w) > 1e-12)[0])

    # Regression coefficients: B = W (P.T W)^-1 Q.T  →  shape (n_features, n_targets)
    # Then transpose to (n_targets, n_features) to match sklearn ≥1.4.
    inner = loadings.T @ weights
    inner_pinv = np.linalg.pinv(inner) if np.linalg.matrix_rank(inner) > 0 else np.zeros_like(inner)
    coef = (weights @ inner_pinv @ y_load.T).T

    return _SparsePLSModel(
        x_weights_=weights,
        x_loadings_=loadings,
        x_scores_=scores,
        y_loadings_=y_load,
        coef_=coef,
        selected_per_comp=selected_per_comp,
    )


def _fit_splsda(
    X: np.ndarray,  # noqa: N803 — sklearn X convention
    y: np.ndarray,
    n_components: int,
    keep_x: int,
    seed: int = 0,
) -> tuple[_SparsePLSModel, np.ndarray, np.ndarray, np.ndarray]:
    """Fit sPLS-DA with one-hot y; return (model, vip, beta_per_class, y_onehot)."""
    n_classes = len(np.unique(y))
    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y] = 1.0

    model = _nipals_sparse(X, y_onehot, n_components=n_components, keep_x=keep_x, seed=seed)
    vip = _compute_vip(model, X)
    beta = model.coef_  # (n_classes, n_features)
    if beta.shape[0] != n_classes:
        beta = beta.T
    return model, vip, beta, y_onehot


def fit(
    ctx: StageContext,
    *,
    n_components: int | None = None,
    keep_x: int | None = None,
    n_bootstrap: int = 100,
    n_permutation: int = 1000,
    seed: int = 42,
    biological_only: bool = True,
) -> MultivariateResult:
    """Fit sPLS-DA and produce per-feature importance + selection + stability.

    Parameters
    ----------
    n_components : int | None, default None (auto)
        ``min(K-1, n_samples-1)`` when ``None``.
    keep_x : int | None, default None (auto)
        ``max(10, min(50, n_features // 100))`` per component when ``None``.
    n_bootstrap : int, default 100
        Bootstrap resamples for stability assessment.
    n_permutation : int, default 1000
        Label permutations for p-value computation.
    seed : int, default 42
        Master RNG seed.
    biological_only : bool, default True
        Restrict to biological samples (drop QCs from feature ranking).
    """
    t0 = time.time()
    X, y, feature_ids, _sample_ids, class_names = _build_xy(ctx, biological_only=biological_only)
    n_samples, n_features = X.shape
    n_classes = len(class_names)

    chosen_nc = min(n_classes - 1, n_samples - 1) if n_components is None else int(n_components)
    chosen_nc = max(1, chosen_nc)
    chosen_keep_x = _auto_keep_x(n_features) if keep_x is None else int(keep_x)
    chosen_keep_x = max(1, min(chosen_keep_x, n_features))

    log.info(
        "splsda_fit_start",
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        class_names=class_names,
        n_components=chosen_nc,
        keep_x=chosen_keep_x,
        n_components_auto=(n_components is None),
        keep_x_auto=(keep_x is None),
    )

    # Primary fit on full data
    model, vip_obs, beta_obs, _ = _fit_splsda(X, y, chosen_nc, chosen_keep_x, seed=seed)

    # Selected = union of per-component top-K (any feature with non-zero weight)
    if model.selected_per_comp:
        sel_idx = np.unique(np.concatenate(model.selected_per_comp))
    else:
        sel_idx = np.array([], dtype=int)
    selected_mask = np.zeros(n_features, dtype=bool)
    selected_mask[sel_idx] = True

    # Bootstrap stability via VIP
    def vip_importance_fn(
        X_b: np.ndarray,  # noqa: N803 — sklearn X convention
        y_b: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if len(np.unique(y_b)) < 2:
            return np.zeros(n_features)
        try:
            sub_seed = int(rng.integers(0, 2**31 - 1))
            _, v, _, _ = _fit_splsda(X_b, y_b, chosen_nc, chosen_keep_x, seed=sub_seed)
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

    perm_pvals = permutation_test(
        importance_fn=vip_importance_fn,
        matrix_arr=X,
        labels=y,
        observed_importance=vip_obs,
        feature_ids=feature_ids,
        n_permutation=n_permutation,
        seed=seed + 1,
    )

    table = pd.DataFrame({"feature_id": feature_ids})
    table["selected"] = selected_mask
    table["vip"] = vip_obs
    for i, cls in enumerate(class_names):
        table[f"beta_{cls}"] = beta_obs[i]
    table["abs_beta_max"] = np.abs(beta_obs).max(axis=0)
    table["perm_importance"] = np.full(n_features, np.nan)  # see module docstring
    table["stability_score"] = stability.stability_score
    table["perm_p_value"] = perm_pvals.p_value

    elapsed = time.time() - t0
    log.info(
        "splsda_fit_done",
        elapsed_s=round(elapsed, 1),
        n_selected=int(selected_mask.sum()),
        n_features_high_vip=int((vip_obs >= 1.0).sum()),
        n_features_stable=int((stability.stability_score >= 0.8).sum()),
        n_features_p_lt_05=int((perm_pvals.p_value < 0.05).sum()),
    )

    return MultivariateResult(
        method_name="splsda",
        per_feature=table,
        params_used={
            "n_components": chosen_nc,
            "keep_x": chosen_keep_x,
            "biological_only": biological_only,
            "stability_top_k": stability.top_k,
        },
        runtime_seconds=elapsed,
        n_bootstrap=n_bootstrap,
        n_permutation=n_permutation,
    )


register(
    MethodSpec(
        name="splsda",
        callable=fit,
        description="sPLS-DA — sparse Lê Cao variant; hard top-K weights per component.",
        citation="Lê Cao 2011, J Chemom 25:151",
        category="multivariate",
    )
)
