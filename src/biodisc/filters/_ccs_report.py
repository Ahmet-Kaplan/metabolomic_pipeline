"""Stage 8 — HTML report for the CCS-error filter.

Three plots + an index:

    1. CCS error histogram — threshold line + kept/dropped colouring (only
       the filterable subset; pass-throughs aren't on this plot).
    2. Stage-8 decision breakdown — kept, dropped-by-CCS, dropped-by-
       quality-floor, pass-through-no-reference, pass-through-no-candidate.
    3. Top-20 best-validated features table-as-bar (smallest CCS error,
       most-confidently kept). Hover shows the matched compound + adduct +
       source DB so you can spot-check identifications.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from biodisc.eda._output import write_plot
from biodisc.eda.report import ReportEntry, build_index


def plot_err_histogram(
    metrics: pd.DataFrame,
    *,
    threshold: float,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Histogram of CCS-error % for the filterable subset, coloured by kept/dropped."""
    filt = metrics[metrics["s8_filterable"]].dropna(subset=["best_ccs_err_pct"]).copy()

    n_filt = len(filt)
    if n_filt == 0:
        # Empty plot fallback
        fig = go.Figure()
        fig.add_annotation(
            text="No filterable features (no candidate had a CCS reference match).",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        fig.update_layout(height=480)
        return write_plot(fig, output_dir, name="01_ccs_err_histogram", emit_png=emit_png)

    n_kept = int(filt["kept"].sum())
    n_dropped = n_filt - n_kept

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=filt.loc[~filt["kept"], "best_ccs_err_pct"],
            xbins=dict(
                start=0.0, end=max(filt["best_ccs_err_pct"].max() + 0.5, threshold + 1.0), size=0.1
            ),
            marker_color="#c7254e",
            opacity=0.85,
            name=f"Dropped (err > {threshold:g} %)",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=filt.loc[filt["kept"], "best_ccs_err_pct"],
            xbins=dict(
                start=0.0, end=max(filt["best_ccs_err_pct"].max() + 0.5, threshold + 1.0), size=0.1
            ),
            marker_color="#1e4377",
            opacity=0.85,
            name=f"Kept (err ≤ {threshold:g} %)",
        )
    )
    fig.add_vline(
        x=threshold,
        line=dict(color="#c9a94e", width=2, dash="dash"),
        annotation_text=f"Threshold {threshold:g} %",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Best-candidate CCS error distribution (filterable subset)</b><br>"
                f"<sup>{n_filt:,} filterable features · "
                f"<b>KEPT {n_kept:,} ({n_kept / n_filt * 100:.1f} %) · "
                f"DROPPED {n_dropped:,} ({n_dropped / n_filt * 100:.1f} %)</b></sup>"
            )
        ),
        xaxis_title="Best |measured − reference| / reference × 100  (%)",
        yaxis_title="Feature count",
        barmode="stack",
        bargap=0.02,
        height=600,
        legend=dict(title="Fate"),
    )
    return write_plot(fig, output_dir, name="01_ccs_err_histogram", emit_png=emit_png)


def plot_decision_breakdown(
    *,
    counts: dict[str, int],
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Stage-8 fate bar chart — every input feature lands in exactly one bucket."""
    labels = [
        "Input",
        "Kept (CCS validated)",
        "Pass-through (no candidate)",
        "Pass-through (no reference)",
        "Dropped — CCS err > thr",
        "Dropped — quality floor",
    ]
    keys = [
        "n_input",
        "n_kept_ccs",
        "n_passthrough_no_candidate",
        "n_passthrough_no_reference",
        "n_dropped_ccs_err",
        "n_dropped_quality_floor",
    ]
    values = [counts.get(k, 0) for k in keys]
    colours = ["#0b2545", "#1e4377", "#9aa6b2", "#9aa6b2", "#c7254e", "#a04060"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colours,
            text=[f"{v:,}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text="<b>Stage 8 fate breakdown — every feature in exactly one bucket</b>"),
        xaxis_title="Bucket",
        yaxis_title="Feature count",
        height=540,
        showlegend=False,
    )
    return write_plot(fig, output_dir, name="02_decision_breakdown", emit_png=emit_png)


def plot_top_validated(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    n_top: int = 20,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Top-N best-validated kept features — smallest CCS error."""
    kept_filtered = metrics[metrics["kept"] & metrics["best_ccs_err_pct"].notna()].copy()
    kept_filtered = kept_filtered.nsmallest(n_top, "best_ccs_err_pct")

    if kept_filtered.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No features were validated with CCS evidence (all kept are pass-through).",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        fig.update_layout(height=420)
        return write_plot(fig, output_dir, name="03_top_validated", emit_png=emit_png)

    # Build a short label per row
    kept_filtered["label"] = (
        kept_filtered["feature_id"].astype(str)
        + "  →  "
        + kept_filtered["best_compound_name"].fillna("(unnamed)").str[:40]
        + "  ["
        + kept_filtered["best_adduct"].fillna("?")
        + "]"
    )
    kept_filtered = kept_filtered.iloc[::-1]  # ascending bottom-up

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=kept_filtered["best_ccs_err_pct"],
            y=kept_filtered["label"],
            orientation="h",
            marker_color="#1e4377",
            text=[f"{v:.3f} %" for v in kept_filtered["best_ccs_err_pct"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "CCS error: %{x:.3f} %<br>"
                "Source: %{customdata[0]}<br>"
                "Reference CCS (corrected): %{customdata[1]:.2f} Å²<br>"
                "Measured CCS: %{customdata[2]:.2f} Å²<extra></extra>"
            ),
            customdata=kept_filtered[
                ["best_source_db", "best_corrected_reference_ccs", "measured_ccs"]
            ].values,
        )
    )
    fig.update_layout(
        title=dict(text=f"<b>Top {n_top} best-CCS-validated kept features</b>"),
        xaxis_title="Best CCS error (%)",
        yaxis=dict(title=None, automargin=True),
        height=max(400, 26 * len(kept_filtered) + 120),
        showlegend=False,
        margin=dict(l=10, r=20, t=80, b=40),
    )
    return write_plot(fig, output_dir, name="03_top_validated", emit_png=emit_png)


def build_report(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    threshold: float,
    counts: dict[str, int],
    emit_png: bool = True,
) -> dict[str, Path]:
    """Compose the three plots into an index.html."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    entries: list[ReportEntry] = []
    outputs: dict[str, Path] = {}

    p1 = plot_err_histogram(metrics, threshold=threshold, output_dir=plots_dir, emit_png=emit_png)
    outputs["01_ccs_err_histogram"] = p1["html"]
    entries.append(
        ReportEntry(
            key="01_ccs_err_histogram",
            title="1. CCS-error distribution (filterable subset)",
            description=(
                "Histogram of best-candidate CCS error % for features that had at "
                "least one Progenesis candidate with a matching adduct in the "
                "external CCS reference. Pass-through features (no candidate or no "
                "reference) are excluded — see plot 2 for those counts."
            ),
            html_href=f"plots/{p1['html'].name}",
            png_href=f"plots/{p1['png'].name}" if "png" in p1 else None,
        )
    )

    p2 = plot_decision_breakdown(counts=counts, output_dir=plots_dir, emit_png=emit_png)
    outputs["02_decision_breakdown"] = p2["html"]
    entries.append(
        ReportEntry(
            key="02_decision_breakdown",
            title="2. Stage 8 fate breakdown",
            description=(
                "Every input feature lands in exactly one bucket. Pass-through "
                "buckets contain features we cannot evaluate (no Progenesis "
                "candidate, or candidate exists but has no entry in the external "
                "CCS reference). These are kept, not dropped — Stage 8 only drops "
                "features when a check is applicable AND fails."
            ),
            html_href=f"plots/{p2['html'].name}",
            png_href=f"plots/{p2['png'].name}" if "png" in p2 else None,
        )
    )

    p3 = plot_top_validated(metrics, output_dir=plots_dir, emit_png=emit_png)
    outputs["03_top_validated"] = p3["html"]
    entries.append(
        ReportEntry(
            key="03_top_validated",
            title="3. Top-20 best-CCS-validated features",
            description=(
                "The 20 kept features with the smallest CCS error against an "
                "external reference. Hover for compound name, adduct, source DB, "
                "and corrected reference CCS — these are the most-confidently "
                "validated identifications in the dataset."
            ),
            html_href=f"plots/{p3['html'].name}",
            png_href=f"plots/{p3['png'].name}" if "png" in p3 else None,
        )
    )

    n_input = counts.get("n_input", 0)
    n_kept_total = counts.get("n_kept_total", 0)
    n_kept_ccs = counts.get("n_kept_ccs", 0)
    extra_info = {
        "Input features (Stage-7-active)": f"{n_input:,}",
        "Kept (total, incl. pass-through)": f"{n_kept_total:,} ({n_kept_total / max(n_input, 1) * 100:.2f} %)",
        "  • Kept via CCS validation": f"{n_kept_ccs:,}",
        "  • Pass-through (no candidate)": f"{counts.get('n_passthrough_no_candidate', 0):,}",
        "  • Pass-through (no reference)": f"{counts.get('n_passthrough_no_reference', 0):,}",
        "Dropped — CCS error > threshold": f"{counts.get('n_dropped_ccs_err', 0):,}",
        "Dropped — quality floor failed": f"{counts.get('n_dropped_quality_floor', 0):,}",
        "CCS error threshold": f"≤ {threshold:g} % ⇒ keep",
        "Quality floor": f"Progenesis Score ≥ {counts.get('min_score', 30)} AND |mass error| ≤ {counts.get('max_ppm', 5)} ppm",
        "Reference": "McLean Unified CCS Compendium (Picache 2019, Tier-2 experimental)",
    }
    index_path = build_index(
        output_path=output_dir / "index.html",
        entries=entries,
        stage_label="Stage 8 — CCS error filter (adduct-aware, quality-floored)",
        extra_info=extra_info,
    )
    outputs["index"] = index_path
    return outputs
