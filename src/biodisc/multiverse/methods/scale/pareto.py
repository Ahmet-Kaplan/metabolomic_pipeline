"""Pareto scaling — ``(x - mean) / sqrt(SD)`` per feature.

The standard scaling for multivariate metabolomics analyses (PLS-DA,
OPLS-DA, PCA). Less aggressive than auto-scaling (which divides by
SD), more informative than mean-centering alone.

Why Pareto specifically (van den Berg 2006)
-------------------------------------------
* **Auto-scaling** (z-score) over-equalises feature contributions —
  noise features get the same weight as real biomarkers.
* **No scaling** lets the few highest-variance features dominate PLS-DA
  loadings — usually not biologically informative.
* **Pareto** is the middle ground: divides by sqrt(SD) so feature
  variances become proportional to SD instead of equal. High-variance
  features still get more weight than low-variance ones, but not so
  much that they swamp everything.

van den Berg 2006 (*BMC Genomics* 7:142) compared 6 scaling methods
on 3 metabolomics datasets and concluded Pareto is the best default
for multivariate analysis.

Algorithm
---------
For each feature ``f``:
::

    out_{f, s} = (matrix_{f, s} - mean_f) / sqrt(SD_f)

where ``mean_f`` and ``SD_f`` are the per-feature statistics computed
across the columns (samples).

Edge case
---------
* Feature with SD = 0 (constant across samples) → divide by 1
  instead, so the centred values are unchanged. Logged.

No tunable parameters
---------------------
Pareto is a fixed transformation by design. The only thing the user
can choose is whether to apply it at all (vs auto-scale, vs none) —
and that's a Phase 6 / Stage 17 concern, not a Pareto-internal one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)


def scale(ctx: StageContext) -> pd.DataFrame:
    """Apply per-feature Pareto scaling. See module docstring for details."""
    matrix = ctx.matrix
    if matrix.isna().any().any():
        raise ValueError("Pareto requires a NaN-free matrix.")

    means = matrix.mean(axis=1)
    sds = matrix.std(axis=1, ddof=1)

    # Edge case: constant features → SD=0 → divide by 1 (centred values unchanged)
    n_constant_features = int((sds == 0).sum())
    sqrt_sds = np.sqrt(sds.replace(0, 1.0))

    centred = matrix.subtract(means, axis=0)
    out = centred.divide(sqrt_sds, axis=0)

    log.info(
        "pareto_scaling_done",
        n_features=matrix.shape[0],
        n_samples=matrix.shape[1],
        n_constant_features=n_constant_features,
        per_feature_mean_after=float(out.mean(axis=1).abs().median()),
        per_feature_sd_after_median=float(out.std(axis=1, ddof=1).median()),
    )
    if n_constant_features > 0:
        log.warning(
            "pareto_constant_features",
            n_constant_features=n_constant_features,
            hint="features with no variation contribute nothing to PLS-DA; consider removing pre-stats",
        )
    return out


register(
    MethodSpec(
        name="pareto",
        callable=scale,
        description="(x - mean) / sqrt(SD) per feature — gold standard for multivariate metabolomics.",
        citation="van den Berg et al. 2006, BMC Genomics 7:142",
        category="scaling",
    )
)
