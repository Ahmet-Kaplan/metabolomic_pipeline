"""Scaling section — verifies the per-feature centring + variance behaviour.

For Pareto specifically (van den Berg 2006), the post-scaling matrix
should satisfy:

* per-feature mean ≈ 0
* per-feature SD ≈ sqrt(SD_before) — the defining property: divide by
  sqrt(SD), so the new SD is the geometric mean of 1 (auto-scale) and the
  original SD (no scale). High-variance features still weigh more, but
  not by the full SD ratio.

The figures here are sanity checks — they don't make a method-selection
case (Pareto is fixed; auto / none aren't run on the multiverse). They
exist to catch silent bugs (e.g. constant-feature handling, broadcast
errors, NaN leakage) in published reports.

Figures
-------
1. **Mean-zero verification** — histogram of per-feature mean post-scale.
2. **SD distribution before vs after** — KDE on log-axis to span the
   dynamic range.
3. **Per-feature SD vs sqrt(SD_before)** — scatter that *should* fall on
   a y=1 line.
4. **Constant-feature audit** — count of features that had SD = 0
   (Pareto fallback to dividing-by-1).

Reference
---------
* van den Berg et al. 2006, *BMC Genomics* 7:142.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biodisc.multiverse.reporting.core import PathReport
from biodisc.multiverse.reporting.figures import (
    histogram_overlay,
    kde_overlay,
    scatter_with_fit,
)


def add_scaling_section(
    report: PathReport,
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    samples: pd.DataFrame | None = None,
    method: str,
    parameters: dict[str, Any] | None = None,
    section_name: str = "scaling",
) -> None:
    """Add the scaling section to ``report``."""
    parameters = dict(parameters or {})
    _ = samples  # unused

    title_map = {
        "pareto": "Pareto scaling — (x − mean) / √SD per feature",
        "auto": "Auto scaling (z-score per feature)",
        "none": "No scaling (identity)",
    }
    description_map = {
        "pareto": (
            "Per-feature centring with √SD denominator (van den Berg 2006, "
            "<em>BMC Genomics</em> 7:142). Less aggressive than auto-scale, more "
            "informative than mean-only centring. The default for multivariate "
            "metabolomics."
        ),
        "auto": (
            "Per-feature z-score: ``(x − mean) / SD``. Equalises feature contribution "
            "to PLS-DA loadings — useful when noise features must not dominate, but "
            "risks washing out genuine high-variance biomarkers."
        ),
        "none": "Identity. Used as a robustness baseline for tree-based models.",
    }

    report.add_section(
        section_name,
        title=title_map.get(method, f"Scaling — {method}"),
        description=description_map.get(method, ""),
        method=method,
        parameters=parameters,
    )

    # ── Per-feature stats ─────────────────────────────────────────
    means_after = after.mean(axis=1)
    sd_before = before.std(axis=1, ddof=1)
    sd_after = after.std(axis=1, ddof=1)
    n_constant = int((sd_before == 0).sum())

    report.add_metric(
        section_name,
        "per-feature mean (after) — median |·|",
        round(float(means_after.abs().median()), 6),
        description="should be ≈ 0 — verifies centring",
    )
    report.add_metric(
        section_name,
        "per-feature SD (after) — median",
        round(float(sd_after.median()), 4),
        description="for Pareto, ≈ sqrt(median SD_before)",
    )
    report.add_metric(
        section_name,
        "per-feature SD (before) — median",
        round(float(sd_before.median()), 4),
    )
    report.add_metric(
        section_name,
        "n_constant_features",
        n_constant,
        unit="features",
        description="features with SD = 0 in input — bypassed (no scaling applied)",
    )

    # ── Figure 1: per-feature mean histogram (mean-zero check) ────
    mean_fig = histogram_overlay(
        [means_after.to_numpy(dtype=float)],
        ["per-feature mean (after)"],
        title="Per-feature mean (post-scaling) — should be ≈ 0",
        xaxis_title="per-feature mean",
        colors=["#0072B2"],
        nbins=60,
    )
    mean_fig.add_vline(
        x=0.0,
        line={"color": "#444", "width": 1.0, "dash": "dot"},
        annotation={"text": "0", "font": {"size": 10, "color": "#444"}},
    )
    report.add_figure(
        section_name,
        title="Mean-zero verification",
        fig=mean_fig,
        caption=(
            "All mass should sit in a tight bar centred on 0. Spread away from 0 = "
            "centring bug. Constant features (SD = 0) are passed through unchanged "
            "in Pareto and may sit off-zero — these are flagged separately."
        ),
        method_metric="centring sanity check",
    )

    # ── Figure 2: SD distribution before vs after on log scale ────
    sd_b_arr = sd_before[sd_before > 0].to_numpy(dtype=float)
    sd_a_arr = sd_after[sd_after > 0].to_numpy(dtype=float)
    if sd_b_arr.size > 0 and sd_a_arr.size > 0:
        sd_fig = kde_overlay(
            [np.log10(sd_b_arr), np.log10(sd_a_arr)],
            ["before", "after"],
            title="Per-feature SD — before vs after (log10 scale)",
            xaxis_title="log10(per-feature SD)",
            colors=["#999999", "#0072B2"],
        )
        report.add_figure(
            section_name,
            title="Per-feature SD distribution",
            fig=sd_fig,
            caption=(
                "Pareto compresses the SD distribution: SD_after ≈ sqrt(SD_before). "
                "On log10 the &quot;after&quot; curve should sit at half the abscissa "
                "of the &quot;before&quot; curve and have smaller spread."
            ),
            method_metric="SD distribution (van den Berg 2006)",
        )

    # ── Figure 3: SD_after vs sqrt(SD_before) — Pareto identity ──
    if method == "pareto":
        sqrt_sd_before = np.sqrt(sd_before.replace(0, np.nan).to_numpy(dtype=float))
        sd_after_arr = sd_after.to_numpy(dtype=float)
        finite = np.isfinite(sqrt_sd_before) & np.isfinite(sd_after_arr)
        if finite.sum() > 5:
            id_fig = scatter_with_fit(
                sqrt_sd_before[finite],
                sd_after_arr[finite],
                title="Pareto identity: SD_after ≈ sqrt(SD_before)",
                xaxis_title="sqrt(SD_before)",
                yaxis_title="SD_after",
                color="#0072B2",
                show_fit=True,
                annotation="ideal: y = x · 1.0",
            )
            # Add y=x reference line
            x_max = float(np.nanmax(sqrt_sd_before[finite]))
            id_fig.add_shape(
                type="line",
                x0=0,
                x1=x_max,
                y0=0,
                y1=x_max,
                line={"color": "#444", "width": 1.0, "dash": "dot"},
            )
            report.add_figure(
                section_name,
                title="Pareto identity check",
                fig=id_fig,
                caption=(
                    "If Pareto is correctly applied, points fall on the y = x line. "
                    "Deviation from y = x = numerical bug or unexpected scaling. "
                    "Constant features (sqrt(SD_before) = 0) excluded."
                ),
                method_metric="Pareto algebraic identity",
            )
