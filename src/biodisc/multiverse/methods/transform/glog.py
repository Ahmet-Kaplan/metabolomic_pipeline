"""glog — Generalized Logarithm (Parsons 2007 / Durbin 2002).

A variance-stabilising transformation that behaves linearly at small
intensity (where noise dominates and log would amplify it) and
logarithmically at large intensity (where signal dominates and the
log compresses the dynamic range).

Algorithm (Parsons 2007, *BMC Bioinformatics* 8:234)
----------------------------------------------------
``out = log2((x + sqrt(x² + λ²)) / 2)``

This is mathematically equivalent to ``arcsinh(x / λ) / ln(2)`` when
``λ > 0``. The transition between linear and log behaviour happens
around ``x ≈ λ``: below that, the transform is approximately linear
in x; well above, it asymptotes to ``log2(x)``.

The crucial parameter is **λ** — it sets where "small" stops and
"large" starts. Wrong λ:
- Too small → glog ≈ log everywhere → loses noise-stabilising property
- Too large → glog ≈ linear/x ≈ x → loses dynamic-range compression

Auto-derived λ
--------------
We use the **median QC standard deviation per feature** as λ. This is
data-aware: it places the linear/log knee at the **estimated noise
floor** of the dataset. Features with intensity well above the QC
noise scale are log-transformed; features near the noise floor are
linearised. This matches Parsons 2007's rationale and Durbin 2002's
"variance-stabilising-by-design" framing.

If no QC samples are available, we fall back to median of all per-
feature SDs.

When glog wins vs log2
----------------------
* Low-abundance features: glog avoids amplifying noise that would
  dominate log2 near the LOD.
* Heteroscedastic data where high-intensity features have small
  CV but low-intensity features have large CV.

When log2 wins
--------------
* Interpretability: a glog "fold change" is hard to explain.
* Most discovery LC-MS where pre-Stage-11 imputation handled the
  zero/LOD issue cleanly.

Both belong in the multiverse — running them in parallel and seeing
which biomarkers survive both is the publication-grade workflow.

References
----------
* Parsons HM et al. 2007, *BMC Bioinformatics* 8:234 — glog for
  metabolomics with parameter selection guidance.
* Durbin BP et al. 2002, *Bioinformatics* 18:S105 — original glog.
* van den Berg RA et al. 2006, *BMC Genomics* 7:142 — heteroscedasticity
  metric used to evaluate transformation quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)


def _auto_lambda(ctx: StageContext) -> tuple[float, str]:
    """Data-aware default for λ. Returns (lambda_value, derivation_source).

    Preferred: median per-feature SD across QC samples (the noise floor).
    Fallback: median per-feature SD across all samples (less noise-pure
    but still data-aware).
    """
    matrix = ctx.matrix
    qc_ids = ctx.qc_sample_ids()
    if len(qc_ids) >= 3:
        sds = matrix[qc_ids].std(axis=1, ddof=1)
        source = "qc_median_sd"
    else:
        sds = matrix.std(axis=1, ddof=1)
        source = "all_median_sd"
    sds = sds[sds > 0]
    if len(sds) == 0:
        return 1.0, "fallback_constant"
    return float(sds.median()), source


def transform(
    ctx: StageContext,
    *,
    lambda_param: float | None = None,
) -> pd.DataFrame:
    """Apply glog transformation. See module docstring for algorithm details.

    Parameters
    ----------
    lambda_param : float | None, default None (auto)
        The transition parameter — features at intensity ≈ λ are at the
        linear/log knee. When ``None``, set to the median per-feature
        QC SD (the noise floor). Must be > 0.
    """
    matrix = ctx.matrix
    if matrix.isna().any().any():
        raise ValueError("glog requires a NaN-free matrix.")
    if (matrix < 0).any().any():
        raise ValueError("glog cannot accept negative input values.")

    if lambda_param is None:
        chosen_lambda, lambda_source = _auto_lambda(ctx)
    else:
        chosen_lambda, lambda_source = float(lambda_param), "user"
    if chosen_lambda <= 0:
        raise ValueError(f"lambda_param must be > 0; got {chosen_lambda}")

    x = matrix.to_numpy(dtype=float)
    out_arr = np.log2((x + np.sqrt(x * x + chosen_lambda * chosen_lambda)) / 2.0)
    out = pd.DataFrame(out_arr, index=matrix.index, columns=matrix.columns)

    log.info(
        "glog_transformation_done",
        n_features=matrix.shape[0],
        n_samples=matrix.shape[1],
        lambda_param=chosen_lambda,
        lambda_param_auto=(lambda_param is None),
        lambda_source=lambda_source,
        output_min=float(out.min().min()),
        output_max=float(out.max().max()),
        output_median=float(out.median().median()),
    )
    return out


register(
    MethodSpec(
        name="glog",
        callable=transform,
        description="Generalized log — variance-stabilising transform with QC-derived knee.",
        citation="Parsons et al. 2007, BMC Bioinformatics 8:234; Durbin et al. 2002",
        category="transformation",
    )
)
