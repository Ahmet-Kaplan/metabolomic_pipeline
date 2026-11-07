"""log2(x + ε) transformation — the workhorse.

The textbook log transformation for LC-MS metabolomics. Stabilises
right-skewed log-normal intensity distributions, makes per-feature
variance roughly constant across the dynamic range, and renders
fold-changes additive (log2 FC = log2(A) − log2(B)).

When log2 wins vs glog
----------------------
* When zeros are absent or have been imputed with sensible LOD
  values (our pipeline guarantees this post-Stage-11).
* When the dynamic range is moderate (≤ 6 orders of magnitude).
* When interpretability matters — log2 FC is universally understood;
  glog FCs require explanation.

When glog wins
--------------
* When low-abundance features dominate the analysis and noise needs
  to be linearised at the bottom of the range.

Both belong in the multiverse — running them in parallel + extracting
consensus biomarkers is the only honest way to handle this choice.

Algorithm
---------
``out_{f, s} = log2(matrix_{f, s} + ε)``

where ``ε`` is a small offset to handle zero values without producing
``-inf``. ε must be in the same units as the data and small enough
not to compress real signals; data-aware default = half of the smallest
observed positive value.

Auto-derived parameters
-----------------------
``epsilon``: ``(smallest positive value in matrix) / 2``. This puts
the imputed-zero floor at a sensible -1 below the smallest real
observation in log2 units, and is small enough that non-zero values
are barely shifted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)


def _auto_epsilon(matrix: pd.DataFrame) -> float:
    """Data-aware default for ε. See module docstring rationale."""
    positive = matrix[matrix > 0]
    if positive.size == 0:
        return 1.0  # absolute fallback for an all-zero matrix
    smallest = float(positive.min().min())
    return smallest / 2.0


def transform(
    ctx: StageContext,
    *,
    epsilon: float | None = None,
) -> pd.DataFrame:
    """Apply log2(x + ε) transformation.

    Parameters
    ----------
    epsilon : float | None, default None (auto)
        Offset added before taking log2 to handle zeros. When ``None``,
        ``(smallest positive observation) / 2``.
    """
    matrix = ctx.matrix
    if matrix.isna().any().any():
        raise ValueError("log2 requires a NaN-free matrix.")
    if (matrix < 0).any().any():
        raise ValueError("log2 cannot accept negative input values.")

    chosen_eps = _auto_epsilon(matrix) if epsilon is None else epsilon
    if chosen_eps <= 0:
        raise ValueError(f"epsilon must be > 0; got {chosen_eps}")

    out_arr = np.log2(matrix.to_numpy(dtype=float) + chosen_eps)
    out = pd.DataFrame(out_arr, index=matrix.index, columns=matrix.columns)

    log.info(
        "log2_transformation_done",
        n_features=matrix.shape[0],
        n_samples=matrix.shape[1],
        epsilon=chosen_eps,
        epsilon_auto=(epsilon is None),
        output_min=float(out.min().min()),
        output_max=float(out.max().max()),
        output_median=float(out.median().median()),
    )
    return out


register(
    MethodSpec(
        name="log2",
        callable=transform,
        description="log2(x+ε) transformation — workhorse with data-aware ε.",
        citation="standard practice; van den Berg et al. 2006, BMC Genomics 7:142",
        category="transformation",
    )
)
