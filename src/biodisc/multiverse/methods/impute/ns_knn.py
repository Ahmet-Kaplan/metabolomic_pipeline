"""NS-kNN — No-Skip k-Nearest-Neighbours imputation.

Implements Lee & Styczynski 2018 (*Metabolomics* 14:153). Standard kNN
imputation drops features with high missingness from the neighbour pool —
problematic in untargeted LC-MS where many features have at least some
missingness. NS-kNN computes pairwise distances using only the samples
that are observed in BOTH features (the intersection), so no features are
excluded.

Implementation notes
--------------------
We use scikit-learn's :class:`~sklearn.impute.KNNImputer`, which uses
``nan_euclidean`` distance by default — exactly the NS-kNN behaviour
(distance computed on shared non-missing entries, scaled to compensate
for variable overlap).

Orientation: KNNImputer treats rows as observations. Our matrix is
``rows = features, cols = samples``, so neighbours are FEATURES (correct
for the metabolomics interpretation: similar features in observed samples
predict missing values).

Parameters
----------
k : int | None, default None (auto)
    Number of nearest features used to impute. When ``None``, computed
    data-aware as ``max(5, min(20, n_features // 200))`` — gives k=5 for
    1,000-feature datasets, k=20 for 4,000+. Lee & Styczynski 2018
    reported k=10 robust across datasets, but auto-scaling avoids the
    arbitrary "always 10" choice. Auto-clamped to ``n_features-1`` if
    user passes a too-large value.
weights : str, default "uniform"
    Either ``"uniform"`` (mean of neighbours) or ``"distance"``
    (inverse-distance weighted). NS-kNN's published version uses uniform.

Auto-default rationale
----------------------
``max(5, min(20, n_features // 200))`` keeps k in [5, 20]. The lower
bound 5 ensures stable averaging (≥5 neighbours' mean is robust);
the upper bound 20 prevents over-smoothing (kNN with k=100 imputes
toward the global mean). The ``// 200`` slope means each additional
200 features adds 1 to k — empirically matches the "extra
collinearity needs more averaging" intuition for highly redundant
metabolomics data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)


def _auto_k(n_features: int) -> int:
    """Data-aware default for k. See module docstring rationale."""
    return max(5, min(20, n_features // 200))


def impute(
    ctx: StageContext,
    *,
    k: int | None = None,
    weights: str = "uniform",
    treat_zero_as_missing: bool = True,
) -> pd.DataFrame:
    """NS-kNN imputation.

    ``treat_zero_as_missing=True`` (default) converts Progenesis QI's
    below-LOD encoding (cells = 0) to NaN before passing to
    ``KNNImputer``. Set False only if zeros in your data are legitimate
    measurements.
    """
    from sklearn.impute import KNNImputer

    matrix = ctx.matrix
    if treat_zero_as_missing:
        n_zero_in = int((matrix == 0).sum().sum())
        matrix = matrix.where(matrix != 0, np.nan)
    else:
        n_zero_in = 0

    n_missing = int(matrix.isna().sum().sum())
    if n_missing == 0:
        log.info(
            "ns_knn_no_missing_values",
            n_features=len(matrix),
            treat_zero_as_missing=treat_zero_as_missing,
        )
        return matrix.copy()

    n_features = matrix.shape[0]
    chosen_k = _auto_k(n_features) if k is None else k
    effective_k = min(chosen_k, max(1, n_features - 1))
    if effective_k != chosen_k:
        log.warning(
            "ns_knn_k_clamped",
            requested_k=chosen_k,
            effective_k=effective_k,
            n_features=n_features,
        )

    imputer = KNNImputer(n_neighbors=effective_k, weights=weights)
    # Rows=features, cols=samples. KNNImputer finds k similar features for
    # each row with missing values, averages their values at the missing
    # column.
    imputed = imputer.fit_transform(matrix.values)
    out = pd.DataFrame(imputed, index=matrix.index, columns=matrix.columns)

    # Floor at zero (defensive — KNN averages of non-negative inputs are
    # non-negative, but floating-point edge cases possible)
    out = out.clip(lower=0.0)

    log.info(
        "ns_knn_imputation_done",
        n_features=len(matrix),
        n_samples=matrix.shape[1],
        n_missing_filled=n_missing,
        n_zero_treated_as_missing=n_zero_in,
        treat_zero_as_missing=treat_zero_as_missing,
        k=effective_k,
        k_auto=(k is None),
        weights=weights,
    )
    return out


register(
    MethodSpec(
        name="ns_knn",
        callable=impute,
        description="No-Skip k-NN — kNN that doesn't drop high-missingness features.",
        citation="Lee & Styczynski 2018, Metabolomics 14:153",
        category="imputation",
    )
)
