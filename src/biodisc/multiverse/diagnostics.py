"""mQACC validation metrics for multiverse imputation + drift correction.

Implements the canonical metabolomics QC framework (Broadhurst 2018,
*Metabolomics* 14:72; mQACC) plus complementary measures from Li 2017,
Kirwan 2013, and De Livera 2015. Each function takes a ``StageContext``
or its components and returns a single scalar (or DataFrame for
per-feature metrics) — pure measurement utilities, no side-effects, no
model fitting. They form the diagnostic stack that drift / normalisation
/ imputation methods are evaluated against.

Why these and not just CV
-------------------------
For drift correction in particular, we need to answer THREE questions
that no single metric can:

  1. **Did QC variance drop?**  → D-ratio, PMSD, QC-CV
  2. **Did biological signal survive?**  → biological silhouette
  3. **Does the model generalize beyond the QCs it trained on?**
     → LOQO-CV residual (the only out-of-sample metric here)

A method that wins all three is genuinely solid. One that wins #1 alone
may be overfitting to QCs (the classical SERRF-on-small-N failure mode).

Functions
---------
``d_ratio`` — Broadhurst 2018, primary mQACC drift metric. Lower is better.
``pmsd`` — Li 2017, pooled median SD of QCs.
``qc_cv_median`` — median per-feature CV across QCs.
``biological_silhouette`` — Rousseeuw 1987 silhouette on biological samples.
``pca_qc_ellipse_area`` — Kirwan 2013 PC1×PC2 95% confidence ellipse area.
``loqo_residuals`` — Leave-One-QC-Out CV runner for any drift method.

Implementation notes
--------------------
All metrics treat zero intensities as observed (true zeros may exist
post-imputation; we don't filter them as missing). They handle empty /
degenerate inputs by returning NaN rather than raising.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext

log = structlog.get_logger(__name__)


# ============================================================
# Per-feature spread metrics
# ============================================================
def _per_feature_sd(matrix: pd.DataFrame, sample_ids: Iterable[str]) -> pd.Series:
    """Return per-feature SD across the given sample subset.

    Uses ``ddof=1`` (sample SD) — matches mQACC convention.
    """
    cols = [s for s in sample_ids if s in matrix.columns]
    if not cols:
        return pd.Series(dtype=float)
    return matrix[cols].std(axis=1, ddof=1)


def _per_feature_cv_pct(matrix: pd.DataFrame, sample_ids: Iterable[str]) -> pd.Series:
    """Return per-feature CV% = 100 × SD / mean across the sample subset."""
    cols = [s for s in sample_ids if s in matrix.columns]
    if not cols:
        return pd.Series(dtype=float)
    sub = matrix[cols]
    means = sub.mean(axis=1)
    sds = sub.std(axis=1, ddof=1)
    cv = 100.0 * sds / means.replace(0, np.nan)
    return cv


# ============================================================
# Headline drift metrics
# ============================================================
def d_ratio(
    matrix: pd.DataFrame,
    qc_ids: Iterable[str],
    sample_ids: Iterable[str],
) -> float:
    """D-ratio (Broadhurst 2018) — primary mQACC drift-quality metric.

    .. math::
        D = \\frac{\\mathrm{median}_f\\bigl(\\sigma_{\\mathrm{QC},f}\\bigr)}
                  {\\mathrm{median}_f\\bigl(\\sigma_{\\mathrm{sample},f}\\bigr)}

    Per Broadhurst's framework: < 0.5 acceptable, < 0.4 good,
    < 0.3 very good, < 0.2 excellent, < 0.1 outstanding. Returns NaN if
    either median is non-positive.
    """
    qc_sd = _per_feature_sd(matrix, qc_ids)
    sample_sd = _per_feature_sd(matrix, sample_ids)
    if qc_sd.empty or sample_sd.empty:
        return float("nan")
    qc_med = float(qc_sd.median())
    samp_med = float(sample_sd.median())
    if samp_med <= 0:
        return float("nan")
    return qc_med / samp_med


def pmsd(
    matrix: pd.DataFrame,
    qc_ids: Iterable[str],
) -> float:
    """Pooled median SD of QCs (Li et al. 2017, *Sci Rep* 6:38881).

    The median of per-feature QC SDs. Lower = QCs are tighter. Used as
    the headline normalisation-quality metric in NOREVA.
    """
    qc_sd = _per_feature_sd(matrix, qc_ids)
    if qc_sd.empty:
        return float("nan")
    return float(qc_sd.median())


def qc_cv_median(
    matrix: pd.DataFrame,
    qc_ids: Iterable[str],
) -> float:
    """Median per-feature QC %CV (Dunn 2011 / mQACC). Standard threshold: 30%."""
    cv = _per_feature_cv_pct(matrix, qc_ids)
    if cv.empty:
        return float("nan")
    return float(cv.median())


def qc_cv_pct_passing(
    matrix: pd.DataFrame,
    qc_ids: Iterable[str],
    threshold: float = 30.0,
) -> float:
    """Fraction of features with QC %CV ≤ threshold (mQACC two-pass guideline)."""
    cv = _per_feature_cv_pct(matrix, qc_ids)
    if cv.empty:
        return float("nan")
    return float((cv <= threshold).mean())


# ============================================================
# Biological signal preservation (overcorrection guardrail)
# ============================================================
def biological_silhouette(
    matrix: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    biological_only: bool = True,
    use_log: bool = True,
) -> float:
    """Mean silhouette of biological samples by ``cell_line`` group.

    Higher is better — biological samples cluster more tightly within
    their groups. Used as the overcorrection guardrail: if drift correction
    erases biology, silhouette drops.

    Parameters
    ----------
    matrix
        Wide feature × sample matrix.
    samples
        Sample metadata with ``sample_id``, ``injection_type``, ``cell_line``.
    biological_only
        If True, restrict to ``injection_type == 'biological'`` samples.
    use_log
        If True, ``log1p`` the matrix before computing silhouette
        (intensity scale is non-Euclidean otherwise).

    Returns NaN if fewer than 2 cell lines or fewer than 2 samples per
    group are present.
    """
    from sklearn.metrics import silhouette_score

    if biological_only:
        bio_ids = samples.loc[
            samples["injection_type"] == "biological", "sample_id"
        ].tolist()
    else:
        bio_ids = samples["sample_id"].tolist()
    cols = [s for s in bio_ids if s in matrix.columns]
    if len(cols) < 4:  # need at least 2 groups × 2 samples
        return float("nan")

    sub = matrix[cols]
    sub_arr_raw = sub.to_numpy(dtype=float)
    # Auto-detect already-log-transformed data: negative values mean caller
    # already logged. Skip log1p in that case (would produce NaN).
    apply_log = use_log and (sub_arr_raw.min() >= 0)
    sub_arr = np.log1p(sub_arr_raw) if apply_log else sub_arr_raw

    # X: rows=samples, cols=features (sklearn convention)
    X = sub_arr.T
    labels = (
        samples.set_index("sample_id").loc[cols, "cell_line"].astype(str).to_numpy()
    )
    if len(set(labels)) < 2:
        return float("nan")
    try:
        return float(silhouette_score(X, labels, metric="euclidean"))
    except ValueError:
        return float("nan")


# ============================================================
# QC tightness in PCA space
# ============================================================
def pca_qc_ellipse_area(
    matrix: pd.DataFrame,
    qc_ids: Iterable[str],
    *,
    use_log: bool = True,
    confidence: float = 0.95,
) -> float:
    """Area of the QC 95% confidence ellipse in PC1×PC2 (Kirwan 2013).

    Smaller = QCs cluster more tightly = drift better corrected. PCA is
    fit on QC samples only (rows=samples, cols=features); the ellipse
    is from the standard 2-sigma chi-square approximation at the chosen
    confidence level.

    Returns NaN if fewer than 3 QC samples.
    """
    from scipy.stats import chi2
    from sklearn.decomposition import PCA

    cols = [s for s in qc_ids if s in matrix.columns]
    if len(cols) < 3:
        return float("nan")

    sub = matrix[cols]
    sub_arr_raw = sub.to_numpy(dtype=float)
    # Auto-detect already-log-transformed data (same logic as silhouette)
    apply_log = use_log and (sub_arr_raw.min() >= 0)
    sub_arr = np.log1p(sub_arr_raw) if apply_log else sub_arr_raw
    X = sub_arr.T

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)  # shape (n_qc, 2)
    cov = np.cov(scores, rowvar=False)
    # Eigenvalues are the variances along principal axes
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    chi2_val = float(chi2.ppf(confidence, df=2))
    # Ellipse axes lengths: sqrt(chi2 × eigval); area = pi × a × b
    a = float(np.sqrt(chi2_val * eigvals[0]))
    b = float(np.sqrt(chi2_val * eigvals[1]))
    return float(np.pi * a * b)


# ============================================================
# Out-of-sample diagnostic — Leave-One-QC-Out CV
# ============================================================
@dataclass(frozen=True)
class LOQOResult:
    """Per-feature LOQO-CV residual statistics for one drift method."""

    n_qc: int
    median_residual_pct: float  # median |residual| / |true| × 100, all (feature × held-QC) pairs
    p95_residual_pct: float
    fraction_features_passing: float  # features with per-feature median residual ≤ threshold
    threshold_pct: float


def loqo_residuals_from_predictor(
    ctx: StageContext,
    *,
    predict_held_qc: Callable[[StageContext, str], np.ndarray],
    threshold_pct: float = 30.0,
) -> LOQOResult:
    """Generic Leave-One-QC-Out runner. Method-specific predict callable required.

    Each drift module exposes its own ``loqo_predict`` function which,
    given a ``StageContext`` and the held-out QC's ``sample_id``, returns
    a per-feature prediction for that held QC's intensities. This module
    handles the orchestration: hold out each QC in turn, call the
    predictor, accumulate residuals as (|pred − true| / true × 100).

    Why method-specific predict
    ---------------------------
    Drift correction is inherently a fitting problem (spline / RF over QC
    anchors). To get a clean held-out residual we must re-fit the model
    on the remaining QCs and evaluate at the held position — there is
    no generic way to do this without knowing the method's structure.
    Each drift module therefore contains a ``loqo_predict(ctx, held_qc)``
    helper alongside the main ``correct(ctx, …)`` entrypoint.

    Threshold ``30%`` is the mQACC standard QC-CV threshold (Broadhurst
    2018); features with median residual above this are poorly-corrected.
    """
    qc_ids = ctx.qc_sample_ids()
    if len(qc_ids) < 5:
        log.warning("loqo_too_few_qcs", n_qc=len(qc_ids))
        return LOQOResult(
            n_qc=len(qc_ids),
            median_residual_pct=float("nan"),
            p95_residual_pct=float("nan"),
            fraction_features_passing=float("nan"),
            threshold_pct=threshold_pct,
        )

    matrix = ctx.matrix
    n_features = matrix.shape[0]
    # Per-feature accumulators of residual % across held QCs
    per_feature_residuals: list[list[float]] = [[] for _ in range(n_features)]

    for held_qc in qc_ids:
        try:
            predicted = predict_held_qc(ctx, held_qc)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("loqo_predict_failed", held_qc=held_qc, error=str(exc))
            continue
        true_vals = matrix[held_qc].to_numpy(dtype=float)
        # Residual % per feature
        with np.errstate(divide="ignore", invalid="ignore"):
            res_pct = np.where(
                true_vals > 0,
                np.abs(predicted - true_vals) / true_vals * 100.0,
                np.nan,
            )
        for i, r in enumerate(res_pct):
            if np.isfinite(r):
                per_feature_residuals[i].append(float(r))

    all_res = np.array(
        [r for sub in per_feature_residuals for r in sub], dtype=float
    )
    if all_res.size == 0:
        return LOQOResult(
            n_qc=len(qc_ids),
            median_residual_pct=float("nan"),
            p95_residual_pct=float("nan"),
            fraction_features_passing=float("nan"),
            threshold_pct=threshold_pct,
        )

    feature_medians = np.array(
        [np.median(r) if r else np.nan for r in per_feature_residuals]
    )

    return LOQOResult(
        n_qc=len(qc_ids),
        median_residual_pct=float(np.nanmedian(all_res)),
        p95_residual_pct=float(np.nanpercentile(all_res, 95)),
        fraction_features_passing=float(
            np.nanmean(feature_medians <= threshold_pct)
        ),
        threshold_pct=threshold_pct,
    )


# ============================================================
# Transformation quality (van den Berg 2006)
# ============================================================
def heteroscedasticity_slope(matrix: pd.DataFrame) -> float:
    """Slope of ``log(per-feature SD) ~ log(per-feature mean)`` (van den Berg 2006).

    Closer to 0 = better variance stabilisation across the dynamic range
    (the headline transformation-quality metric). On raw LC-MS data
    pre-log-transform, the slope is typically near 1 (variance scales
    with mean — heteroscedastic). Post-log2 it should drop substantially.
    Pos-glog with a well-tuned λ should be even closer to 0.

    Returns NaN if fewer than 5 features have positive mean and SD.
    """
    means = matrix.mean(axis=1)
    sds = matrix.std(axis=1, ddof=1)
    valid = (means > 0) & (sds > 0)
    if int(valid.sum()) < 5:
        return float("nan")
    log_means = np.log(means[valid].to_numpy())
    log_sds = np.log(sds[valid].to_numpy())
    # Linear regression slope via numpy
    slope, _ = np.polyfit(log_means, log_sds, deg=1)
    return float(slope)


def mean_abs_skewness(matrix: pd.DataFrame) -> float:
    """Mean per-feature |skewness|. Lower = more normal distribution shape.

    Useful as a "did the transformation help approach normality" check.
    Right-skewed LC-MS intensities (pre-log) typically have skewness 3–10;
    post-log they should be < 1.
    """
    from scipy.stats import skew as scipy_skew

    if matrix.shape[1] < 3:
        return float("nan")
    skews = matrix.apply(lambda row: scipy_skew(row.to_numpy(dtype=float), bias=False), axis=1)
    return float(np.abs(skews).mean())


def ks_normality_fraction(
    matrix: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> float:
    """Fraction of features passing Kolmogorov-Smirnov normality test at α.

    "Passing" means failing to reject the null of normality (p > α).
    Higher = more normal-looking distributions per feature.
    """
    from scipy.stats import kstest

    if matrix.shape[1] < 4:
        return float("nan")
    n_pass = 0
    n_total = 0
    means = matrix.mean(axis=1)
    sds = matrix.std(axis=1, ddof=1)
    for fid in matrix.index:
        row = matrix.loc[fid].to_numpy(dtype=float)
        m = float(means.loc[fid])
        s = float(sds.loc[fid])
        if s == 0 or not np.isfinite(s):
            continue
        # Standardise then KS-test against standard normal
        z = (row - m) / s
        try:
            _, p = kstest(z, "norm")
        except Exception:
            continue
        n_total += 1
        if p > alpha:
            n_pass += 1
    if n_total == 0:
        return float("nan")
    return n_pass / n_total


@dataclass(frozen=True)
class TransformationDiagnostics:
    """Full diagnostic stack for one transformation state."""

    heteroscedasticity_slope: float
    mean_abs_skewness: float
    ks_normality_fraction: float


def compute_transformation_diagnostics(
    matrix: pd.DataFrame,
) -> TransformationDiagnostics:
    """Compute all transformation-relevant diagnostics in one call."""
    return TransformationDiagnostics(
        heteroscedasticity_slope=heteroscedasticity_slope(matrix),
        mean_abs_skewness=mean_abs_skewness(matrix),
        ks_normality_fraction=ks_normality_fraction(matrix),
    )


# ============================================================
# Silhouette guardrail (overcorrection check)
# ============================================================
@dataclass(frozen=True)
class SilhouetteCheck:
    """Result of a before/after silhouette comparison."""

    silhouette_before: float
    silhouette_after: float
    relative_drop: float  # (before - after) / |before|
    threshold: float  # acceptable relative drop (e.g. 0.05 = 5%)
    overcorrected: bool


def silhouette_guardrail(
    matrix_before: pd.DataFrame,
    matrix_after: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    relative_drop_threshold: float = 0.05,
) -> SilhouetteCheck:
    """Check whether drift correction erased biological structure.

    Computes biological silhouette before and after a transformation,
    flags overcorrection if relative drop exceeds the threshold. The
    mQACC framework (Broadhurst 2018) implicitly recommends this check
    — De Livera 2015 explicitly documents the 5% rule as a typical
    overcorrection guardrail.

    Returns a ``SilhouetteCheck`` with both values + whether the drop
    crosses the threshold. Caller decides what to do (warn, block,
    keep-with-flag).
    """
    sil_before = biological_silhouette(matrix_before, samples)
    sil_after = biological_silhouette(matrix_after, samples)
    if not (np.isfinite(sil_before) and np.isfinite(sil_after)):
        return SilhouetteCheck(
            silhouette_before=sil_before,
            silhouette_after=sil_after,
            relative_drop=float("nan"),
            threshold=relative_drop_threshold,
            overcorrected=False,
        )
    rel_drop = 0.0 if sil_before == 0 else (sil_before - sil_after) / abs(sil_before)
    overcorrected = rel_drop > relative_drop_threshold
    if overcorrected:
        log.warning(
            "silhouette_guardrail_triggered",
            silhouette_before=sil_before,
            silhouette_after=sil_after,
            relative_drop=rel_drop,
            threshold=relative_drop_threshold,
            hint="drift correction may have erased biology — verify visually",
        )
    return SilhouetteCheck(
        silhouette_before=sil_before,
        silhouette_after=sil_after,
        relative_drop=rel_drop,
        threshold=relative_drop_threshold,
        overcorrected=overcorrected,
    )


# ============================================================
# Convenience: full diagnostic stack as one call
# ============================================================
@dataclass(frozen=True)
class DriftDiagnostics:
    """Full diagnostic stack for one drift-correction state (before or after)."""

    d_ratio: float
    pmsd: float
    qc_cv_median: float
    qc_cv_pct_passing_30: float
    biological_silhouette: float
    pca_qc_ellipse_area: float


def compute_drift_diagnostics(
    matrix: pd.DataFrame,
    samples: pd.DataFrame,
    qc_ids: Iterable[str],
    sample_ids: Iterable[str],
) -> DriftDiagnostics:
    """Compute all drift-relevant diagnostics in one call."""
    return DriftDiagnostics(
        d_ratio=d_ratio(matrix, qc_ids, sample_ids),
        pmsd=pmsd(matrix, qc_ids),
        qc_cv_median=qc_cv_median(matrix, qc_ids),
        qc_cv_pct_passing_30=qc_cv_pct_passing(matrix, qc_ids, threshold=30.0),
        biological_silhouette=biological_silhouette(matrix, samples),
        pca_qc_ellipse_area=pca_qc_ellipse_area(matrix, qc_ids),
    )
