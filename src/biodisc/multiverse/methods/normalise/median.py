"""Median normalisation — simplest baseline.

Per-sample factor = sample median / global median. Divides each sample
by its own median (relative to the overall median across samples) so
all samples end up centered around the same intensity scale.

When to use
-----------
* As the **simplest possible normalisation baseline** — useful in the
  multiverse for distinguishing "PQN/MSTUS captured something subtle"
  from "any centering would have worked". If PQN and median produce
  similar shortlists, the methodological choice didn't matter for
  this dataset.

* Sometimes used as a fast pre-PQN check. If sample medians differ by
  > 2× across the cohort, that's a sample-loading red flag worth
  investigating before applying any normalisation.

Algorithm
---------
1. Per sample s, compute ``median_s`` over non-zero values
   (zeros are imputed-LOD; including them would bias toward zero).
2. Compute ``global_median = median(median_s)``.
3. Per-sample factor = ``median_s / global_median``.
4. Normalise: ``corrected_{f, s} = matrix_{f, s} / factor_s``.

Edge cases handled
------------------
* Sample with all-zero values → median=NaN → factor = 1.0 (no-op).
* Global median = 0 (every sample is all-zero) → raises ValueError.

No tunable parameters
--------------------
Median normalisation has no hyperparameters by design. The "no
parameter" property makes it the cleanest control method in the
multiverse — any biomarker that emerges from the median path AND a
PQN/MSTUS path is robust to normalisation algorithm choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)


def normalise(ctx: StageContext) -> pd.DataFrame:
    """Apply median normalisation. See module docstring for algorithm details."""
    matrix = ctx.matrix
    if matrix.isna().any().any():
        raise ValueError(
            "Median normalisation requires an imputed matrix (no NaN). "
            "Run Stage 11 imputation first."
        )

    # Per-sample median of NON-ZERO values
    sample_medians = matrix.replace(0, np.nan).median(axis=0)
    global_median = float(sample_medians.median(skipna=True))

    if not np.isfinite(global_median) or global_median <= 0:
        raise ValueError(
            "Median normalisation cannot proceed: global median across samples is "
            f"{global_median}. Check for an all-zero or NaN matrix."
        )

    # Compute factors; samples with NaN median (all-zero) → factor = 1
    factors_series = (sample_medians / global_median).fillna(1.0)
    n_all_zero = int(sample_medians.isna().sum())

    out = matrix.div(factors_series, axis=1)
    out = out.clip(lower=0.0)

    log.info(
        "median_normalisation_done",
        n_features=matrix.shape[0],
        n_samples=matrix.shape[1],
        n_samples_all_zero_fallback=n_all_zero,
        factor_min=float(factors_series.min()),
        factor_max=float(factors_series.max()),
        factor_median=float(factors_series.median()),
        global_reference_median=global_median,
    )
    if (factors_series < 0.1).any() or (factors_series > 10).any():
        log.warning(
            "median_extreme_factors",
            extreme_low=int((factors_series < 0.1).sum()),
            extreme_high=int((factors_series > 10).sum()),
            hint="extreme sample medians suggest sample-loading outliers",
        )
    return out


register(
    MethodSpec(
        name="median",
        callable=normalise,
        description="Per-sample median scaling — simplest normalisation baseline.",
        citation="standard practice; see van den Berg 2006 review",
        category="normalisation",
    )
)
