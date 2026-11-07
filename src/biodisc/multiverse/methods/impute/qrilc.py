"""QRILC — Quantile Regression Imputation of Left-Censored data.

Per-feature imputation specifically designed for **MNAR** (missing-not-at-random)
data dominated by below-LOD values. Implements Lazar et al. 2016 (*J Proteome
Res* 15:1116) algorithm:

  For each feature:
    1. Collect observed values across all samples for this feature.
    2. Log-transform (LC-MS abundances are log-normal).
    3. Estimate the underlying normal distribution: mean and std of log values.
    4. Estimate the LOD as a low quantile (default 1st percentile) of log-obs.
    5. For each missing value, sample from a truncated normal
       N(mean, std) restricted to values < LOD. Imputed values are
       biologically plausible (positive, below the smallest observed).

This is the MNAR reference standard cited in QComics 2024 and remains the
gold-standard MNAR comparison method in the 2024 imputomics benchmark
(Krzywiec et al., *Bioinformatics*).

Edge cases:
  * Feature with < ``min_observations`` non-NaN values → fallback to half-min
    (insufficient data to estimate the truncated-normal parameters).
  * Feature with all observed values identical → fallback to half-min.

Parameters
----------
lod_quantile : float, default 0.01
    Quantile of log-observed values used as the upper truncation. Lower =
    more conservative (imputes deeper in the tail). Literature constant
    from Lazar 2016 — kept fixed.
min_observations : int | None, default None (auto)
    Minimum non-missing values required to fit the truncated normal.
    When ``None``, computed data-aware as ``max(3, n_samples // 10)`` —
    requires at least 3 observations or 10% of samples, whichever is
    larger. Below this, fall back to half-minimum (insufficient data
    to estimate distribution parameters reliably).
seed : int, default 42
    For deterministic truncated-normal sampling.

Auto-default rationale (min_observations)
-----------------------------------------
``max(3, n_samples // 10)`` ensures we have ≥ 3 observations OR
≥ 10% of samples to fit the log-normal. With 47 samples → 5; with
100 samples → 10. Floor of 3 is the absolute minimum to estimate
mean + sd.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from scipy.stats import truncnorm

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)


def _auto_min_observations(n_samples: int) -> int:
    """Data-aware default for min_observations. See module docstring rationale."""
    return max(3, n_samples // 10)


def impute(
    ctx: StageContext,
    *,
    lod_quantile: float = 0.01,
    min_observations: int | None = None,
    seed: int = 42,
    treat_zero_as_missing: bool = True,
) -> pd.DataFrame:
    """QRILC imputation per feature.

    ``treat_zero_as_missing=True`` (default) converts Progenesis QI's
    below-LOD encoding (cells = 0) to NaN before imputation, then
    proceeds as normal. Set False only if zeros in your data are
    legitimate measurements.
    """
    matrix = ctx.matrix
    if treat_zero_as_missing:
        n_zero_in = int((matrix == 0).sum().sum())
        matrix = matrix.where(matrix != 0, np.nan)
    else:
        n_zero_in = 0

    n_missing = int(matrix.isna().sum().sum())
    if n_missing == 0:
        log.info(
            "qrilc_no_missing_values",
            n_features=len(matrix),
            treat_zero_as_missing=treat_zero_as_missing,
        )
        return matrix.copy()

    n_samples = matrix.shape[1]
    chosen_min_obs = (
        _auto_min_observations(n_samples) if min_observations is None else min_observations
    )

    rng = np.random.default_rng(seed)
    out = matrix.copy()
    n_fallback_halfmin = 0
    n_fallback_zero = 0
    n_qrilc_filled = 0

    for fid in matrix.index:
        row = matrix.loc[fid]
        if not row.isna().any():
            continue
        observed = row.dropna()
        # Edge case: all-NaN feature
        if len(observed) == 0:
            out.loc[fid] = row.fillna(0.0)
            n_fallback_zero += int(row.isna().sum())
            continue

        # Edge case: too few observed → fallback to half-min
        if len(observed) < chosen_min_obs:
            out.loc[fid] = row.fillna(0.5 * observed.min())
            n_fallback_halfmin += int(row.isna().sum())
            continue

        # Edge case: all observed are identical → SD=0 → fallback
        log_obs = np.log(observed.astype(float))
        mu = float(log_obs.mean())
        sigma = float(log_obs.std(ddof=1))  # sample SD (unbiased; n >= chosen_min_obs >= 3)
        if sigma == 0 or not np.isfinite(sigma):
            out.loc[fid] = row.fillna(0.5 * observed.min())
            n_fallback_halfmin += int(row.isna().sum())
            continue

        # Truncate at the lod_quantile of log-observed values
        lod_log = float(np.quantile(log_obs, lod_quantile))
        # Convert truncation point to standard-normal units for scipy.truncnorm
        b = (lod_log - mu) / sigma
        n_to_fill = int(row.isna().sum())
        log_samples = truncnorm.rvs(
            -np.inf,
            b,
            loc=mu,
            scale=sigma,
            size=n_to_fill,
            random_state=rng,
        )
        samples = np.exp(log_samples)
        nan_mask = row.isna().to_numpy()
        new_row = row.to_numpy().copy()
        new_row[nan_mask] = samples
        out.loc[fid] = new_row
        n_qrilc_filled += n_to_fill

    log.info(
        "qrilc_imputation_done",
        n_features=len(matrix),
        n_missing_filled=n_missing,
        n_zero_treated_as_missing=n_zero_in,
        treat_zero_as_missing=treat_zero_as_missing,
        n_qrilc_filled=n_qrilc_filled,
        n_fallback_halfmin=n_fallback_halfmin,
        n_fallback_zero=n_fallback_zero,
        lod_quantile=lod_quantile,
        min_observations=chosen_min_obs,
        min_observations_auto=(min_observations is None),
    )
    return out


register(
    MethodSpec(
        name="qrilc",
        callable=impute,
        description="Quantile Regression for Left-Censored data — MNAR reference standard.",
        citation="Lazar et al. 2016, J Proteome Res 15:1116",
        category="imputation",
    )
)
