"""PQN — Probabilistic Quotient Normalization (Dieterle 2006).

The gold-standard sample-loading normaliser for clinical and discovery
metabolomics. NOREVA 2.0 (Yang 2020) ranks PQN among the top performers;
the 2024 mQACC workshop (Mosley *Metabolomics*) confirms it as the
default for biofluid + cell-extract LC-MS.

Algorithm (Dieterle 2006, *Anal Chem* 78:4281)
----------------------------------------------
1. Build a **reference spectrum** — per-feature median across the QC
   pool (preferred — minimal biological variance) or, if no QCs are
   available, across all biological samples.
2. For each sample, compute per-feature **quotients**:
       q_{f,s} = sample_{f,s} / reference_f
   (only for features where reference_f > 0).
3. Per-sample **normalisation factor** = median of that sample's
   non-zero quotients.
4. **Normalise**: corrected_{f,s} = sample_{f,s} / sample_factor_s.

Why median (not mean) of quotients
----------------------------------
Median is robust to differentially-expressed features. The whole point
of PQN is to estimate the per-sample dilution / loading factor without
being thrown off by the few features that genuinely changed between
groups (which would bias a mean-based estimator).

Edge cases handled
------------------
* Reference value = 0 for a feature → that feature is excluded from
  every sample's quotient calculation (we cannot divide by zero).
* Sample value = 0 → that feature is excluded from THIS sample's
  median quotient calculation (zero would pull the median down
  spuriously).
* Sample has fewer than ``min_features_for_factor`` valid quotients
  (typically because most features are zero in that sample) → fall
  back to factor = 1.0 (no normalisation; logged).
* No QC samples available → fall back to all-sample median reference.

Auto-derived parameters
-----------------------
* ``reference``: ``"qc_median"`` if ≥ 3 QCs, else ``"all_median"``.
  No user input needed in normal use.
* ``min_features_for_factor``: ``max(10, n_features // 100)`` —
  ensures the per-sample factor is computed from enough features to
  be statistically robust. For 4644 features → 46.
"""

from __future__ import annotations

import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)


def _auto_min_features_for_factor(n_features: int) -> int:
    """Data-aware minimum number of valid quotients to trust a sample factor.

    Heuristic: ``max(10, n_features // 100)``. Floor of 10 ensures the
    median is statistically meaningful; scale with feature count for
    larger datasets where 1% gives a tighter requirement.
    """
    return int(max(10, n_features // 100))


def normalise(
    ctx: StageContext,
    *,
    reference: str = "auto",
    min_features_for_factor: int | None = None,
    min_qc_for_reference: int = 3,
) -> pd.DataFrame:
    """Apply PQN normalisation. See module docstring for algorithm details.

    Parameters
    ----------
    reference : str, default "auto"
        How to build the per-feature reference spectrum:
          - ``"auto"``: use QC median if ≥ ``min_qc_for_reference`` QC
            samples exist, else all-sample median.
          - ``"qc_median"``: force QC-based reference; raise if too few QCs.
          - ``"all_median"``: median across all samples.
    min_features_for_factor : int | None, default None (auto)
        Minimum non-zero quotients required to trust a sample factor.
        Below this, factor = 1.0 (no-op for that sample). When ``None``,
        ``max(10, n_features // 100)``.
    min_qc_for_reference : int, default 3
        Minimum QC count to use QC-based reference. Below this, ``"auto"``
        falls back to ``"all_median"``.
    """
    matrix = ctx.matrix
    if matrix.isna().any().any():
        raise ValueError(
            "PQN requires an imputed matrix (no NaN). Run Stage 11 imputation first."
        )

    n_features = matrix.shape[0]
    chosen_min_feat = (
        _auto_min_features_for_factor(n_features)
        if min_features_for_factor is None
        else min_features_for_factor
    )

    qc_ids = ctx.qc_sample_ids()
    n_qc = len(qc_ids)

    # Choose reference
    if reference == "auto":
        if n_qc >= min_qc_for_reference:
            reference_used = "qc_median"
            ref_series = matrix[qc_ids].median(axis=1)
        else:
            reference_used = "all_median"
            ref_series = matrix.median(axis=1)
    elif reference == "qc_median":
        if n_qc < min_qc_for_reference:
            raise ValueError(
                f"reference='qc_median' requires ≥ {min_qc_for_reference} QC "
                f"samples; found {n_qc}. Use reference='auto' or "
                f"reference='all_median'."
            )
        reference_used = "qc_median"
        ref_series = matrix[qc_ids].median(axis=1)
    elif reference == "all_median":
        reference_used = "all_median"
        ref_series = matrix.median(axis=1)
    else:
        raise ValueError(
            f"reference must be 'auto', 'qc_median', or 'all_median'; got {reference!r}"
        )

    # Features with reference > 0 are eligible for the quotient calculation
    valid_features = ref_series > 0
    n_valid_features = int(valid_features.sum())
    if n_valid_features == 0:
        raise ValueError(
            "PQN reference has zero valid features (all reference values are 0). "
            "Check that the QC samples (or all samples) have non-zero data."
        )

    ref_valid = ref_series[valid_features]
    matrix_valid = matrix.loc[valid_features]

    # Per-sample factor: median of quotients (skip zero sample values)
    sample_factors: dict[str, float] = {}
    n_factor_fallback = 0
    for sid in matrix.columns:
        sample_col = matrix_valid[sid]
        # Skip zero sample values to avoid biasing the median downward
        positive_mask = sample_col > 0
        if int(positive_mask.sum()) < chosen_min_feat:
            sample_factors[sid] = 1.0
            n_factor_fallback += 1
            continue
        quotients = sample_col[positive_mask] / ref_valid[positive_mask]
        sample_factors[sid] = float(quotients.median())

    factors_series = pd.Series(sample_factors)
    out = matrix.div(factors_series, axis=1)
    out = out.clip(lower=0.0)

    # Warn when the median factor deviates substantially from 1.0 across
    # samples — this indicates that >50% of features may be differentially
    # expressed, violating PQN's core assumption (most features unchanged).
    factor_spread = float(factors_series.max() - factors_series.min())
    median_factor_deviation = float(abs(factors_series.median() - 1.0))
    if median_factor_deviation > 0.1 or factor_spread > 0.5:
        log.warning(
            "pqn_assumption_may_be_violated",
            reason="median_factor_deviation_or_spread_too_large",
            median_factor=float(factors_series.median()),
            factor_min=float(factors_series.min()),
            factor_max=float(factors_series.max()),
            factor_spread=factor_spread,
            median_deviation_from_1=median_factor_deviation,
            hint=(
                "PQN assumes most features are NOT differentially expressed. "
                "Large factor deviations suggest >50% of features are DE, "
                "which biases the normalisation factor. Consider whether "
                "another normalisation method is more appropriate."
            ),
        )

    log.info(
        "pqn_normalisation_done",
        n_features=n_features,
        n_samples=matrix.shape[1],
        reference_used=reference_used,
        n_qc_for_reference=n_qc if reference_used == "qc_median" else 0,
        n_valid_features_in_reference=n_valid_features,
        min_features_for_factor=chosen_min_feat,
        min_features_for_factor_auto=(min_features_for_factor is None),
        n_factor_fallback=n_factor_fallback,
        factor_min=float(factors_series.min()),
        factor_max=float(factors_series.max()),
        factor_median=float(factors_series.median()),
    )
    if (factors_series < 0.1).any() or (factors_series > 10).any():
        log.warning(
            "pqn_extreme_factors",
            extreme_low=int((factors_series < 0.1).sum()),
            extreme_high=int((factors_series > 10).sum()),
            hint="extreme sample factors suggest poor reference fit or sample-loading outlier",
        )
    return out


register(
    MethodSpec(
        name="pqn",
        callable=normalise,
        description="Probabilistic Quotient Normalization — gold standard for clinical metabolomics.",
        citation="Dieterle et al. 2006, Anal. Chem. 78:4281",
        category="normalisation",
    )
)
