"""Transformation section — variance stabilisation diagnostics.

The reviewer needs to see three things:

1. **Heteroscedasticity slope** — log(per-feature SD) regressed on
   log(per-feature mean). Pre-log2 the slope is typically ≈ 1 (variance
   scales with mean — heteroscedastic). Post-log2 it should drop towards
   0 (variance stabilised). The headline transformation-quality metric
   (van den Berg 2006).
2. **Skewness distribution** — per-feature skewness should move from
   right-skewed (values 3–10 pre-log) to roughly symmetric (≤ 1 post-log).
3. **Normality fraction** — fraction of features that pass a KS-normality
   test at α = 0.05. Should rise after a successful transformation.

References
----------
* van den Berg et al. 2006, *BMC Genomics* 7:142 — heteroscedasticity-slope
  framework, Pareto/scaling guidance.
* Parsons et al. 2007, *BMC Bioinformatics* 8:234 — glog and skewness
  framing.
* Krzywiec et al. 2024 — modern benchmarking confirms log2 + Pareto
  remains the default after dec ades.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biodisc.multiverse.diagnostics import (
    ks_normality_fraction,
    mean_abs_skewness,
)
from biodisc.multiverse.reporting.core import PathReport
from biodisc.multiverse.reporting.figures import (
    histogram_overlay,
    paired_metric_bar,
    scatter_with_fit,
    violin_paired,
)


def _per_feature_skewness(matrix: pd.DataFrame) -> np.ndarray:
    """Per-feature skewness across samples, ignoring NaN."""
    from scipy.stats import skew as scipy_skew

    out = []
    for fid in matrix.index:
        row = matrix.loc[fid].to_numpy(dtype=float)
        finite = row[np.isfinite(row)]
        if finite.size < 3:
            out.append(np.nan)
        else:
            out.append(float(scipy_skew(finite, bias=False)))
    return np.asarray(out, dtype=float)


def _log_mean_log_sd(matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (log_mean, log_sd) per feature for the heteroscedasticity scatter.

    Filters to features with positive mean AND positive SD.
    """
    means = matrix.mean(axis=1)
    sds = matrix.std(axis=1, ddof=1)
    valid = (means > 0) & (sds > 0)
    if int(valid.sum()) < 5:
        return np.array([]), np.array([])
    return np.log(means[valid].to_numpy()), np.log(sds[valid].to_numpy())


def _het_slope(matrix: pd.DataFrame) -> float:
    """Heteroscedasticity slope (van den Berg 2006). NaN if too few features."""
    log_mean, log_sd = _log_mean_log_sd(matrix)
    if log_mean.size < 5:
        return float("nan")
    slope, _ = np.polyfit(log_mean, log_sd, deg=1)
    return float(slope)


def add_transformation_section(
    report: PathReport,
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    samples: pd.DataFrame | None = None,  # accepted but not used; keeps signature uniform
    method: str,
    parameters: dict[str, Any] | None = None,
    section_name: str = "transformation",
) -> None:
    """Add the transformation section to ``report``."""
    parameters = dict(parameters or {})
    _ = samples  # unused by transformation diagnostics

    title_map = {
        "log2": "log2(x + ε) transformation",
        "glog": "Generalized logarithm (glog) transformation",
        "cube_root": "Cube-root transformation",
        "none": "No transformation (identity)",
    }
    description_map = {
        "log2": (
            "Workhorse transformation for LC-MS metabolomics. Stabilises right-skewed "
            "log-normal intensities; renders fold-changes additive (van den Berg 2006). "
            "ε offset added before log to handle zeros / imputed-LOD values."
        ),
        "glog": (
            "Generalized logarithm: ``y = log(x + sqrt(x² + λ))`` (Parsons 2007). "
            "Linearises the noise floor of low-abundance features more aggressively "
            "than log2 — useful when zero-inflated or imputed LOD values dominate."
        ),
        "cube_root": (
            "y = sign(x) × |x|^(1/3). Variance-stabilising for positive-skewed "
            "distributions; less aggressive than log."
        ),
        "none": "Identity. Used as baseline in the multiverse benchmark.",
    }

    report.add_section(
        section_name,
        title=title_map.get(method, f"Transformation — {method}"),
        description=description_map.get(method, ""),
        method=method,
        parameters=parameters,
    )

    # ── Headline metrics ──────────────────────────────────────────
    het_before = _het_slope(before)
    het_after = _het_slope(after)
    skew_before = mean_abs_skewness(before)
    skew_after = mean_abs_skewness(after)
    ksn_before = ks_normality_fraction(before)
    ksn_after = ks_normality_fraction(after)

    metric_pairs = [
        ("heteroscedasticity slope", het_before, het_after),
        ("mean |skewness|", skew_before, skew_after),
        ("KS-normal fraction", ksn_before, ksn_after),
    ]
    for label, b, a in metric_pairs:
        report.add_metric(
            section_name,
            f"{label} (before)",
            round(float(b), 4) if np.isfinite(b) else "n/a",
        )
        report.add_metric(
            section_name,
            f"{label} (after)",
            round(float(a), 4) if np.isfinite(a) else "n/a",
        )

    # ── Figure 1: paired metric bars ──────────────────────────────
    metric_fig = paired_metric_bar(
        [m[0] for m in metric_pairs],
        [float(m[1]) if np.isfinite(m[1]) else 0.0 for m in metric_pairs],
        [float(m[2]) if np.isfinite(m[2]) else 0.0 for m in metric_pairs],
        title="Variance-stabilisation metrics — before vs after",
        yaxis_title="metric value",
    )
    report.add_figure(
        section_name,
        title="Headline metric panel",
        fig=metric_fig,
        caption=(
            "Heteroscedasticity slope: closer to 0 = better variance stabilisation "
            "(van den Berg 2006). Mean |skewness|: lower = more symmetric distributions. "
            "KS-normal fraction: higher = more features look log-normal."
        ),
        method_metric="van den Berg 2006 transformation framework",
    )

    # ── Figure 2: heteroscedasticity scatter ──────────────────────
    log_mean_b, log_sd_b = _log_mean_log_sd(before)
    log_mean_a, log_sd_a = _log_mean_log_sd(after)
    if log_mean_b.size > 5:
        het_b_fig = scatter_with_fit(
            log_mean_b,
            log_sd_b,
            title="Heteroscedasticity scatter — before transformation",
            xaxis_title="log(per-feature mean)",
            yaxis_title="log(per-feature SD)",
            color="#999999",
            annotation=f"slope = {het_before:+.3f}",
        )
        report.add_figure(
            section_name,
            title="Heteroscedasticity (before)",
            fig=het_b_fig,
            caption=(
                "Each point = one feature. A slope near 1 means SD scales with mean — "
                "the classical right-skewed pattern that needs log."
            ),
            method_metric="heteroscedasticity slope (van den Berg 2006)",
        )

    if log_mean_a.size > 5:
        het_a_fig = scatter_with_fit(
            log_mean_a,
            log_sd_a,
            title="Heteroscedasticity scatter — after transformation",
            xaxis_title="log(per-feature mean)",
            yaxis_title="log(per-feature SD)",
            color="#0072B2",
            annotation=f"slope = {het_after:+.3f}",
        )
        report.add_figure(
            section_name,
            title="Heteroscedasticity (after)",
            fig=het_a_fig,
            caption=(
                "Slope should drop towards 0 — variance stabilised across the dynamic "
                "range. Residual slope &gt; 0.3 = variance still scales with mean."
            ),
            method_metric="heteroscedasticity slope (van den Berg 2006)",
        )

    # ── Figure 3: per-feature skewness violins ────────────────────
    skew_b = _per_feature_skewness(before)
    skew_a = _per_feature_skewness(after)
    if np.isfinite(skew_b).any() or np.isfinite(skew_a).any():
        skew_fig = violin_paired(
            [skew_b, skew_a],
            ["before", "after"],
            title="Per-feature skewness — before vs after",
            yaxis_title="skewness",
            colors=["#999999", "#0072B2"],
        )
        skew_fig.add_hline(
            y=0,
            line={"color": "#444", "width": 1.0, "dash": "dot"},
            annotation={"text": "perfect symmetry", "font": {"size": 10, "color": "#444"}},
            annotation_position="top right",
        )
        report.add_figure(
            section_name,
            title="Per-feature skewness violin",
            fig=skew_fig,
            caption=(
                "Symmetric distributions cluster around 0. LC-MS intensities are "
                "right-skewed pre-log (median &gt; 1, often 3–10); post-log they "
                "should fall to ≤ 1."
            ),
            method_metric="skewness distribution (Parsons 2007)",
        )

    # ── Figure 4: per-feature mean histograms (separate panels — different units) ──
    # Before and after live on incommensurable axes — pre-transform spans 4–6
    # orders of magnitude on a positive scale; post-log is signed and compressed.
    # Putting them on a single axis crushes both. Render as two separate panels
    # with appropriately scaled axes.
    means_b_raw = before.mean(axis=1).to_numpy(dtype=float)
    means_b = means_b_raw[np.isfinite(means_b_raw) & (means_b_raw > 0)]
    if means_b.size > 0:
        before_fig = histogram_overlay(
            [np.log10(means_b)],
            ["per-feature mean (before)"],
            title="Per-feature mean intensity — before transformation (log10 of raw mean)",
            xaxis_title="log10(per-feature mean)",
            colors=["#999999"],
            nbins=40,
        )
        report.add_figure(
            section_name,
            title="Per-feature mean intensity — before",
            fig=before_fig,
            caption=(
                "log10 of the raw per-feature mean. LC-MS abundances are "
                "approximately log-normal; expect a roughly bell-shaped "
                "distribution centred on the typical log-abundance, with the "
                "spread reflecting the dynamic range of the experiment."
            ),
            method_metric="distribution shape (raw scale)",
        )

    means_a_raw = after.mean(axis=1).to_numpy(dtype=float)
    means_a = means_a_raw[np.isfinite(means_a_raw)]
    if means_a.size > 0:
        after_fig = histogram_overlay(
            [means_a],
            ["per-feature mean (after)"],
            title="Per-feature mean intensity — after transformation",
            xaxis_title="per-feature mean (transformed units)",
            colors=["#0072B2"],
            nbins=40,
        )
        report.add_figure(
            section_name,
            title="Per-feature mean intensity — after",
            fig=after_fig,
            caption=(
                "Distribution after the chosen transformation. For log2 the "
                "x-axis is signed log2-units; the distribution should look "
                "more symmetric than its raw counterpart and is bounded above "
                "by ~log2(max raw intensity)."
            ),
            method_metric="distribution shape (transformed)",
        )
