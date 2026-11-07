"""Half-minimum imputation — LOD surrogate.

For each feature, replace missing values with ``0.5 × min(observed across
samples)``. Rationale: in untargeted LC-MS, missing values are dominated by
MNAR (Missing Not At Random) — the analyte is below the limit of detection.
Half the smallest observed value is a defensible LOD surrogate (Hrydziuszko &
Viant 2012 *Metabolomics*).

Krzywiec et al. 2024 (*Bioinformatics* — imputomics 42-method benchmark)
crowned half-minimum the **MNAR winner** by NRMSE; it remains the de-facto
baseline in NOREVA and MetaboAnalyst.

Progenesis missing-value encoding
---------------------------------
Progenesis QI exports below-LOD cells as ``0``, NOT NaN. So with the
default ``treat_zero_as_missing=True``, zero cells are converted to NaN
at entry and imputed alongside any genuine NaN cells. Set this False
only when zeros in your data carry biological meaning (e.g., a different
upstream parser already converted Progenesis zeros to NaN).

Edge cases:
  * Feature with NO observed positive values → fallback to
    ``fallback_value`` (default 0.0 — gets log-transformed downstream as
    min-of-min).
  * ``factor`` parameter lets users tune ½ → ⅓ etc. without changing code.

Parameters
----------
factor : float, default 0.5
    Multiplier on the per-feature minimum. 0.5 = half-minimum (textbook).
fallback_value : float, default 0.0
    Used when a feature has zero observations across all samples (rare after
    Stage 3 presence filter, but defensive).
treat_zero_as_missing : bool, default True
    If True, treat ``x == 0`` as missing alongside ``NaN``. Required for
    Progenesis QI output (below-LOD = 0). Set False if zeros in your
    data are legitimate measurements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)


def impute(
    ctx: StageContext,
    *,
    factor: float = 0.5,
    fallback_value: float = 0.0,
    treat_zero_as_missing: bool = True,
) -> pd.DataFrame:
    """Half-minimum imputation. Returns a copy with NaN (and optionally 0) replaced."""
    matrix = ctx.matrix

    # Convert zero cells to NaN if requested. After this, the rest of the
    # imputation logic works unchanged on a "NaN = missing" matrix.
    if treat_zero_as_missing:
        n_zero_in = int((matrix == 0).sum().sum())
        matrix = matrix.where(matrix != 0, np.nan)
    else:
        n_zero_in = 0

    n_missing_in = int(matrix.isna().sum().sum())
    if n_missing_in == 0:
        log.info(
            "halfmin_no_missing_values",
            n_features=len(matrix),
            treat_zero_as_missing=treat_zero_as_missing,
        )
        return matrix.copy()

    # Per-feature minimum of observed (non-NaN, > 0) values
    feature_min = matrix.min(axis=1, skipna=True)
    fill_value = feature_min * factor

    # Features with all-NaN: feature_min is NaN; fill with fallback_value
    fill_value = fill_value.where(feature_min.notna(), fallback_value)

    # Apply imputation: only replace NaN cells, leave observed ones alone
    out = matrix.copy()
    # Broadcast fill_value across columns
    fill_broadcast = pd.DataFrame(
        np.broadcast_to(fill_value.to_numpy()[:, None], out.shape),
        index=out.index,
        columns=out.columns,
    )
    out = out.where(out.notna(), fill_broadcast)

    n_features_all_nan = int(feature_min.isna().sum())
    log.info(
        "halfmin_imputation_done",
        n_features=len(matrix),
        n_samples=matrix.shape[1],
        n_missing_filled=n_missing_in,
        n_zero_treated_as_missing=n_zero_in,
        treat_zero_as_missing=treat_zero_as_missing,
        factor=factor,
        n_features_all_nan_filled_fallback=n_features_all_nan,
    )
    return out


register(
    MethodSpec(
        name="halfmin",
        callable=impute,
        description="Half-minimum (½ × per-feature minimum) — MNAR LOD surrogate.",
        citation="Hrydziuszko & Viant 2012, Metabolomics 8:161",
        category="imputation",
    )
)
