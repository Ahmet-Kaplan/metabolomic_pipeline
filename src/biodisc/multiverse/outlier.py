"""Per-path PCA-based outlier flagging for the multiverse pipeline.

Implements the SIMCA-style outlier toolkit (Wold 1976; Eriksson 2006
*Multi- and Megavariate Data Analysis*) widely used in metabolomics:

* **Hotelling's T²** — the *in-model* distance. A sample's projection
  onto the PCA score space is compared against the cluster of all
  samples; large T² = far from the cluster centre in the PC plane.
  Critical value comes from the F-distribution.

* **DModX** — *Distance to Model in X-space* — the *out-of-model*
  distance. A sample's residuals after reconstructing it from the
  top-k PCs; large DModX = the PCA model doesn't explain this sample
  (different shape, not just extreme position).

* **Q-residuals** — squared prediction error per sample. Closely
  related to DModX² (DModX = sqrt(SSE_i / (P − k))); reported alongside
  for diagnostic completeness.

Why combine T² + DModX
----------------------
A sample can be:

  * High T², low DModX — extreme but well-explained (a real biological
    extreme; consider whether to keep or drop).
  * High DModX, normal T² — different shape, not far in PC space (often
    technical issue: contamination, instrument hiccup).
  * Both high — comprehensive outlier; almost always worth investigating.

Flagging via either statistic alone catches both kinds.

mQACC requirement: NEVER auto-remove
------------------------------------
The pipeline FLAGS outliers and produces a diagnostic. The user
inspects the flag (typically via the silhouette/PCA plots downstream)
and decides whether to add the sample to ``parse.exclude_samples`` —
which then re-runs every stage from Stage 1 with the cleaner sample
set (see the sample-exclusion mechanism in the parser).

Auto-derived parameters
-----------------------
* ``n_components``: number of PCs to retain. Default ``"auto"``: the
  smallest k such that explained variance ≥ 95% OR k = max(2,
  n_samples // 6) — whichever is smaller. Two PCs minimum so the
  PCA scatter remains 2D-interpretable.
* ``alpha``: significance level for both T² and DModX critical values
  (default 0.05).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import structlog

from biodisc.multiverse._types import StageContext

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class OutlierResult:
    """Per-sample outlier diagnostics for one PCA fit.

    Attributes
    ----------
    sample_ids
        Order matches the columns of ``ctx.matrix`` restricted to the
        analysed subset (typically biological-only).
    t_squared
        Per-sample Hotelling's T² statistic.
    dmodx
        Per-sample DModX (out-of-model distance).
    q_residuals
        Per-sample squared prediction error.
    t_squared_critical
        Critical value at α = ``alpha`` (F-distribution).
    dmodx_critical
        Critical value at α = ``alpha`` (F-distribution).
    flagged_t2
        Boolean array — sample exceeds T² critical.
    flagged_dmodx
        Boolean array — sample exceeds DModX critical.
    flagged_either
        Boolean array — flagged by either test.
    n_components
        Number of PCs retained in the PCA fit.
    explained_variance_ratio
        Per-PC variance explained.
    alpha
        Significance level used.
    """

    sample_ids: list[str]
    t_squared: np.ndarray
    dmodx: np.ndarray
    q_residuals: np.ndarray
    t_squared_critical: float
    dmodx_critical: float
    flagged_t2: np.ndarray
    flagged_dmodx: np.ndarray
    flagged_either: np.ndarray
    n_components: int
    explained_variance_ratio: np.ndarray
    alpha: float

    @property
    def n_outliers_t2(self) -> int:
        return int(self.flagged_t2.sum())

    @property
    def n_outliers_dmodx(self) -> int:
        return int(self.flagged_dmodx.sum())

    @property
    def n_outliers_either(self) -> int:
        return int(self.flagged_either.sum())

    @property
    def outlier_sample_ids(self) -> list[str]:
        """Sample IDs flagged by either T² or DModX."""
        return [s for s, f in zip(self.sample_ids, self.flagged_either, strict=True) if f]


def _auto_n_components(
    n_samples: int,
    n_features: int,
    centered: np.ndarray,
) -> int:
    """Data-aware default for PCA n_components.

    Smallest ``k`` such that cumulative explained variance ≥ 95%, capped
    at ``max(2, n_samples // 6)`` to avoid over-fitting and ensure the
    PCA scatter stays interpretable. Always at least 2.
    """
    from sklearn.decomposition import PCA

    max_k = max(2, n_samples // 6)
    max_k = min(max_k, n_samples - 1, n_features - 1)
    pca = PCA(n_components=max_k)
    pca.fit(centered)
    cum = np.cumsum(pca.explained_variance_ratio_)
    # Find smallest k achieving 95% variance
    idx = np.searchsorted(cum, 0.95) + 1
    chosen = min(int(idx), max_k)
    return max(2, chosen)


def detect_outliers(
    ctx: StageContext,
    *,
    biological_only: bool = True,
    n_components: int | str = "auto",
    alpha: float = 0.05,
    sample_ids: Iterable[str] | None = None,
) -> OutlierResult:
    """Run PCA and compute per-sample T², DModX, Q-residuals.

    Parameters
    ----------
    biological_only : bool, default True
        If True, restrict to ``injection_type == 'biological'`` samples
        when ``sample_ids`` is None. QC samples cluster tightly by design
        and would not be expected to flag; including them in the PCA
        biases the centre toward QCs.
    n_components : int | "auto", default "auto"
        Number of PCs to retain. ``"auto"`` chooses the smallest k for
        ≥95% variance, capped by ``max(2, n_samples // 6)``.
    alpha : float, default 0.05
        Significance level for T² and DModX critical values.
    sample_ids : iterable of str, optional
        Explicit sample subset to analyse. Overrides ``biological_only``.
    """
    from scipy.stats import f as f_dist
    from sklearn.decomposition import PCA

    if ctx.matrix.isna().any().any():
        raise ValueError("Outlier detection requires a NaN-free matrix.")

    # 1) Pick samples to analyse
    if sample_ids is not None:
        cols = [s for s in sample_ids if s in ctx.matrix.columns]
    elif biological_only:
        cols = ctx.samples.loc[
            ctx.samples["injection_type"] == "biological", "sample_id"
        ].tolist()
        cols = [s for s in cols if s in ctx.matrix.columns]
    else:
        cols = list(ctx.matrix.columns)

    if len(cols) < 4:
        raise ValueError(
            f"Outlier detection needs ≥ 4 samples; found {len(cols)} after filtering."
        )

    # 2) Build feature × sample → samples × features (sklearn convention)
    sub = ctx.matrix[cols]
    samples_x_features = sub.T.to_numpy(dtype=float)
    n_samples, n_features = samples_x_features.shape

    # Center features (mean=0). PCA centers by default but we'll need centred
    # data explicitly for residual computation.
    means = samples_x_features.mean(axis=0)
    centered = samples_x_features - means

    # 3) Auto-derive n_components if requested
    if isinstance(n_components, str):
        if n_components != "auto":
            raise ValueError(f"n_components must be int or 'auto'; got {n_components!r}")
        chosen_k = _auto_n_components(n_samples, n_features, centered)
    else:
        chosen_k = int(n_components)
        chosen_k = max(2, min(chosen_k, n_samples - 1, n_features - 1))

    # 4) Fit PCA
    pca = PCA(n_components=chosen_k)
    scores = pca.fit_transform(centered)  # shape (n_samples, k)
    eigvals = pca.explained_variance_  # length k

    # 5) Hotelling's T² per sample
    # T²_i = sum over k components of (score_{i,k}² / eigval_k)
    # Critical: T²_crit = k * (n-1) * F(k, n-k, α) / (n - k)
    inv_eigvals = 1.0 / np.maximum(eigvals, 1e-12)
    t_squared = (scores * scores * inv_eigvals[None, :]).sum(axis=1)
    n = n_samples
    if n - chosen_k > 0:
        f_crit_t2 = float(f_dist.ppf(1 - alpha, chosen_k, n - chosen_k))
        t_squared_critical = chosen_k * (n - 1) * f_crit_t2 / (n - chosen_k)
    else:
        t_squared_critical = float("inf")

    # 6) DModX per sample (out-of-model distance)
    # Reconstruct: X̂ = scores @ components + means
    reconstruction = scores @ pca.components_ + means
    residuals = samples_x_features - reconstruction
    # Per-sample SSE
    sse_per_sample = (residuals * residuals).sum(axis=1)
    # Total residual variance
    df_residual = max(1, n_features - chosen_k)
    s0_squared = sse_per_sample.sum() / max(1, (n - chosen_k - 1) * df_residual)
    s0 = float(np.sqrt(s0_squared)) if s0_squared > 0 else 1e-12
    dmodx = np.sqrt(sse_per_sample / df_residual) / s0
    # Critical value: F-distribution with (P-k, (N-k-1)*(P-k)) df
    if df_residual > 0 and (n - chosen_k - 1) > 0:
        f_crit_dmodx = float(f_dist.ppf(1 - alpha, df_residual, (n - chosen_k - 1) * df_residual))
        dmodx_critical = float(np.sqrt(f_crit_dmodx))
    else:
        dmodx_critical = float("inf")

    # 7) Q-residuals (= sse_per_sample for completeness)
    q_residuals = sse_per_sample

    flagged_t2 = t_squared > t_squared_critical
    flagged_dmodx = dmodx > dmodx_critical
    flagged_either = flagged_t2 | flagged_dmodx

    log.info(
        "outlier_detection_done",
        n_samples=n_samples,
        n_features=n_features,
        n_components=chosen_k,
        n_components_auto=(n_components == "auto"),
        alpha=alpha,
        n_outliers_t2=int(flagged_t2.sum()),
        n_outliers_dmodx=int(flagged_dmodx.sum()),
        n_outliers_either=int(flagged_either.sum()),
        t_squared_critical=t_squared_critical,
        dmodx_critical=dmodx_critical,
        explained_variance_total=float(pca.explained_variance_ratio_.sum()),
    )
    if flagged_either.any():
        flagged_ids = [cols[i] for i in np.where(flagged_either)[0]]
        log.warning(
            "outliers_flagged",
            sample_ids=flagged_ids,
            n=len(flagged_ids),
            hint="add to parse.exclude_samples and `dvc repro` to re-run from Stage 1",
        )

    return OutlierResult(
        sample_ids=cols,
        t_squared=t_squared,
        dmodx=dmodx,
        q_residuals=q_residuals,
        t_squared_critical=t_squared_critical,
        dmodx_critical=dmodx_critical,
        flagged_t2=flagged_t2,
        flagged_dmodx=flagged_dmodx,
        flagged_either=flagged_either,
        n_components=chosen_k,
        explained_variance_ratio=pca.explained_variance_ratio_,
        alpha=alpha,
    )
