"""MSTUS — MS Total Useful Signal normalisation (Warrack 2009).

Per-sample factor = sum of "useful" feature intensities, scaled so the
median sample factor across the cohort = 1. The "useful" feature set
comprises features detected (non-zero) in at least ``useful_min_pct`` of
samples — the assumption being that consistently-detected features form
the bulk of the analytical signal, and their summed intensity tracks
sample loading.

When MSTUS shines vs PQN
------------------------
* Strong **dilution-driven variability** (e.g., spot-urine where
  hydration varies sample-to-sample): MSTUS's per-sample sum tracks
  total volume directly.
* When biological samples have very different feature compositions
  but share a "useful" core; PQN can struggle if the median quotient
  is dominated by a few highly-variable features.

When PQN beats MSTUS (Warrack himself notes)
--------------------------------------------
* When a few highly-abundant features change strongly between groups
  — they dominate the sum and bias the per-sample factor.
* Most discovery LC-MS settings: PQN is the safer default, MSTUS the
  alternative for robustness checks (which is exactly what the
  multiverse uses both for).

Algorithm
---------
1. Identify "useful" features: detected (intensity > 0) in
   ≥ ``useful_min_pct`` of samples.
2. Per sample s, compute ``sum_s = Σ_{f ∈ useful} matrix[f, s]``.
3. Per-sample factor = ``sum_s / median(sum_s across all samples)``.
4. Normalise: ``corrected_{f, s} = matrix_{f, s} / factor_s``.

Edge cases handled
------------------
* No useful features (all features sparse) → fall back to median
  normalisation across all features (logged).
* Sample sum = 0 (all useful features zero in this sample) → factor
  = 1.0 (no-op for that sample).

Auto-derived parameters
-----------------------
* ``useful_min_pct``: defaults to ``0.80``. This matches our Stage 3
  presence threshold — features that survived Stage 3 are by
  construction detected in ≥ 80% of at least one group, which is the
  Bijlsma 2006 modified-80%-rule. Using 0.80 here keeps the
  "useful" definition consistent with the upstream filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)


def normalise(
    ctx: StageContext,
    *,
    useful_min_pct: float = 0.80,
) -> pd.DataFrame:
    """Apply MSTUS normalisation. See module docstring for algorithm details.

    Parameters
    ----------
    useful_min_pct : float, default 0.80
        Fraction of samples in which a feature must be detected
        (intensity > 0) to count as "useful". 0.80 matches the
        upstream Stage-3 presence-filter threshold (Bijlsma 2006).
        Use 1.0 for the strict definition (features in ALL samples)
        or 0.50 for a more permissive useful set.
    """
    matrix = ctx.matrix
    if matrix.isna().any().any():
        raise ValueError(
            "MSTUS requires an imputed matrix (no NaN). Run Stage 11 imputation first."
        )

    n_features, n_samples = matrix.shape

    # Per-feature presence count (across samples)
    presence_pct = (matrix > 0).mean(axis=1)
    useful_mask = presence_pct >= useful_min_pct
    n_useful = int(useful_mask.sum())

    if n_useful == 0:
        log.warning(
            "mstus_no_useful_features",
            useful_min_pct=useful_min_pct,
            hint="falling back to median normalisation",
        )
        # Fallback: median normalisation across all features
        sample_medians = matrix.replace(0, np.nan).median(axis=0).fillna(1.0)
        global_median = float(sample_medians.median())
        factors_series = sample_medians / global_median
    else:
        # Per-sample sum of useful features
        useful_matrix = matrix.loc[useful_mask]
        sums = useful_matrix.sum(axis=0)
        median_sum = float(sums.median())
        if median_sum <= 0:
            raise ValueError(
                "MSTUS median of per-sample useful-feature sums is zero. "
                "Check that the imputed matrix has non-zero data."
            )
        factors_series = sums / median_sum

    # Handle zero-sum samples (factor = 0 → div-by-zero downstream)
    n_factor_fallback = int((factors_series == 0).sum())
    factors_series = factors_series.replace(0, 1.0)

    out = matrix.div(factors_series, axis=1)
    out = out.clip(lower=0.0)

    log.info(
        "mstus_normalisation_done",
        n_features=n_features,
        n_samples=n_samples,
        n_useful_features=n_useful,
        useful_min_pct=useful_min_pct,
        n_factor_fallback=n_factor_fallback,
        factor_min=float(factors_series.min()),
        factor_max=float(factors_series.max()),
        factor_median=float(factors_series.median()),
    )
    if (factors_series < 0.1).any() or (factors_series > 10).any():
        log.warning(
            "mstus_extreme_factors",
            extreme_low=int((factors_series < 0.1).sum()),
            extreme_high=int((factors_series > 10).sum()),
            hint="extreme sample-factor values suggest sample-loading outliers",
        )
    return out


register(
    MethodSpec(
        name="mstus",
        callable=normalise,
        description="MS Total Useful Signal — sum-based normalisation, robust to dilution effects.",
        citation="Warrack et al. 2009, J. Chromatogr. B 877:547",
        category="normalisation",
    )
)
