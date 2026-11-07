"""Stage 5 — HTML report (Sample-RSD / QC-RSD ratio filter).

Three plots + an index:

    1. Ratio histogram — every feature placed at its (capped) ratio, coloured
       by kept vs dropped. Threshold line at 1.0 (or whatever was configured).
    2. RSD_samples vs RSD_QC scatter — each feature plotted on the (RSD_QC,
       RSD_samples) plane. The y = x diagonal IS the ratio = 1 threshold.
    3. CV-component breakdown of dropped features — what's the dominant
       reason: high QC noise (denominator), low biological variation
       (numerator), or both?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from biodisc.eda._output import write_plot
from biodisc.eda.report import ReportEntry, build_index


# ----------------------------------------------------------
# Plot 1 — ratio histogram
# ----------------------------------------------------------
def plot_ratio_histogram(
    metrics: pd.DataFrame,
    *,
    threshold: float,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Histogram of ratio (capped at 5 for display) coloured by fate."""
    valid = metrics.dropna(subset=["ratio"]).copy()
    # Cap +inf / very large for the histogram only
    valid["ratio_capped"] = valid["ratio"].clip(upper=5.0)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=valid.loc[~valid["kept"], "ratio_capped"],
            xbins=dict(start=0.0, end=5.0, size=0.05),
            marker_color="#c7254e",
            opacity=0.85,
            name=f"Dropped (≤ {threshold:g})",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=valid.loc[valid["kept"], "ratio_capped"],
            xbins=dict(start=0.0, end=5.0, size=0.05),
            marker_color="#1e4377",
            opacity=0.85,
            name=f"Kept (> {threshold:g})",
        )
    )
    fig.add_vline(
        x=threshold,
        line=dict(color="#c9a94e", width=2, dash="dash"),
        annotation_text=f"Threshold {threshold:g}",
        annotation_position="top right",
    )

    n_total = len(metrics)
    n_kept = int(metrics["kept"].sum())
    n_dropped = n_total - n_kept
    n_inf = int(np.isinf(metrics["ratio"]).sum())
    n_nan = int(metrics["ratio"].isna().sum())

    fig.update_layout(
        title=dict(
            text=(
                "<b>Sample-RSD / QC-RSD ratio (Stage 5 before/after)</b><br>"
                f"<sup>{n_total:,} features (post-Stage-4) · "
                f"<b>KEPT {n_kept:,} ({n_kept / n_total * 100:.1f} %) · "
                f"DROPPED {n_dropped:,} ({n_dropped / n_total * 100:.1f} %)</b> · "
                f"display capped at ratio = 5 (n_∞ = {n_inf}, n_NaN = {n_nan})</sup>"
            )
        ),
        xaxis_title="ratio = RSD_samples / RSD_QC  (capped at 5 for display)",
        yaxis_title="Feature count",
        barmode="stack",
        bargap=0.02,
        height=600,
        legend=dict(title="Fate"),
    )
    return write_plot(fig, output_dir, name="01_ratio_histogram", emit_png=emit_png)


# ----------------------------------------------------------
# Plot 2 — RSD_samples vs RSD_QC scatter
# ----------------------------------------------------------
def plot_rsd_scatter(
    metrics: pd.DataFrame,
    *,
    threshold: float,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Per-feature RSD_QC vs RSD_samples; the line y = threshold·x is the cutoff."""
    valid = metrics.dropna(subset=["sample_rsd_pct", "qc_rsd_pct"]).copy()
    kept = valid[valid["kept"]]
    dropped = valid[~valid["kept"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=dropped["qc_rsd_pct"],
            y=dropped["sample_rsd_pct"],
            mode="markers",
            marker=dict(size=3, color="#c7254e", opacity=0.45),
            name=f"Dropped ({len(dropped):,})",
            text=dropped["feature_id"],
            hovertemplate=("%{text}<br>RSD_QC=%{x:.2f}%<br>RSD_samples=%{y:.2f}%<extra></extra>"),
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=kept["qc_rsd_pct"],
            y=kept["sample_rsd_pct"],
            mode="markers",
            marker=dict(size=3, color="#1e4377", opacity=0.35),
            name=f"Kept ({len(kept):,})",
            text=kept["feature_id"],
            hovertemplate=("%{text}<br>RSD_QC=%{x:.2f}%<br>RSD_samples=%{y:.2f}%<extra></extra>"),
        )
    )

    # Plot the y = threshold * x cutoff line
    if not valid.empty:
        x_max = float(min(valid["qc_rsd_pct"].quantile(0.99), 200.0))
        fig.add_trace(
            go.Scatter(
                x=[0, x_max],
                y=[0, x_max * threshold],
                mode="lines",
                line=dict(color="#c9a94e", width=2, dash="dash"),
                name=f"Cutoff: y = {threshold:g} · x",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                "<b>RSD_samples vs RSD_QC (the geometry of the filter)</b><br>"
                "<sup>Each dot is one feature. Above the dashed line: biology > "
                "technical noise → KEPT. Below: technical noise dominates → DROPPED.</sup>"
            )
        ),
        xaxis_title="RSD_QC (%) — technical noise",
        yaxis_title="RSD_samples (%) — biological variation",
        xaxis=dict(
            range=[0, min(valid["qc_rsd_pct"].quantile(0.99) if not valid.empty else 100, 200)]
        ),
        yaxis=dict(
            range=[0, min(valid["sample_rsd_pct"].quantile(0.99) if not valid.empty else 100, 200)]
        ),
        height=640,
        legend=dict(title="Fate"),
    )
    return write_plot(fig, output_dir, name="02_rsd_scatter", emit_png=emit_png)


# ----------------------------------------------------------
# Plot 3 — drop-reason breakdown (waterfall)
# ----------------------------------------------------------
def plot_drop_breakdown(
    metrics: pd.DataFrame,
    *,
    threshold: float,
    n_input: int,
    n_kept: int,
    n_dropped_ratio: int,
    n_dropped_detections: int,
    n_dropped_undefined: int,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Waterfall-style stacked bar showing where features went."""
    labels = [
        "Input",
        "Kept",
        "Dropped — ratio ≤ thr",
        "Dropped — insufficient detections",
        "Dropped — ratio undefined",
    ]
    values = [n_input, n_kept, n_dropped_ratio, n_dropped_detections, n_dropped_undefined]
    colours = ["#0b2545", "#1e4377", "#c7254e", "#a04060", "#7d3a4b"]

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
        title=dict(
            text=(
                "<b>Stage 5 fate breakdown</b><br>"
                f"<sup>Threshold ratio > {threshold:g}. Sum of last three bars = "
                f"{n_dropped_ratio + n_dropped_detections + n_dropped_undefined:,} dropped.</sup>"
            )
        ),
        xaxis_title="Bucket",
        yaxis_title="Feature count",
        height=540,
        showlegend=False,
    )
    return write_plot(fig, output_dir, name="03_drop_breakdown", emit_png=emit_png)


# ----------------------------------------------------------
# Report orchestrator
# ----------------------------------------------------------
def build_report(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    threshold: float,
    n_input: int,
    n_kept: int,
    n_dropped_ratio: int,
    n_dropped_detections: int,
    n_dropped_undefined: int,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Generate Stage 5 plots + index.html."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    entries: list[ReportEntry] = []
    outputs: dict[str, Path] = {}

    p1 = plot_ratio_histogram(metrics, threshold=threshold, output_dir=plots_dir, emit_png=emit_png)
    outputs["01_ratio_histogram"] = p1["html"]
    entries.append(
        ReportEntry(
            key="01_ratio_histogram",
            title="1. RSD_samples / RSD_QC ratio (before/after)",
            description=(
                "Histogram of per-feature ratio coloured by fate. "
                "Threshold line marks the drop cutoff."
            ),
            html_href=f"plots/{p1['html'].name}",
            png_href=f"plots/{p1['png'].name}" if "png" in p1 else None,
        )
    )

    p2 = plot_rsd_scatter(metrics, threshold=threshold, output_dir=plots_dir, emit_png=emit_png)
    outputs["02_rsd_scatter"] = p2["html"]
    entries.append(
        ReportEntry(
            key="02_rsd_scatter",
            title="2. RSD_samples vs RSD_QC scatter",
            description=(
                "The geometry of the filter. Each dot is a feature; the y = "
                f"{threshold:g}·x line separates kept from dropped."
            ),
            html_href=f"plots/{p2['html'].name}",
            png_href=f"plots/{p2['png'].name}" if "png" in p2 else None,
        )
    )

    p3 = plot_drop_breakdown(
        metrics,
        threshold=threshold,
        n_input=n_input,
        n_kept=n_kept,
        n_dropped_ratio=n_dropped_ratio,
        n_dropped_detections=n_dropped_detections,
        n_dropped_undefined=n_dropped_undefined,
        output_dir=plots_dir,
        emit_png=emit_png,
    )
    outputs["03_drop_breakdown"] = p3["html"]
    entries.append(
        ReportEntry(
            key="03_drop_breakdown",
            title="3. Stage 5 fate breakdown",
            description=(
                "Bar chart partitioning the input into kept and the three drop "
                "reasons (ratio ≤ threshold / insufficient detections / undefined)."
            ),
            html_href=f"plots/{p3['html'].name}",
            png_href=f"plots/{p3['png'].name}" if "png" in p3 else None,
        )
    )

    extra_info = {
        "Input features (Stage-4-kept)": f"{n_input:,}",
        "Kept": f"{n_kept:,} ({n_kept / n_input * 100:.2f} %)",
        "Dropped — ratio ≤ threshold": f"{n_dropped_ratio:,}",
        "Dropped — insufficient detections": f"{n_dropped_detections:,}",
        "Dropped — ratio undefined": f"{n_dropped_undefined:,}",
        "Threshold": f"{threshold} (ratio > threshold ⇒ keep)",
    }
    index_path = build_index(
        output_path=output_dir / "index.html",
        entries=entries,
        stage_label="Stage 5 — Sample-RSD / QC-RSD ratio",
        extra_info=extra_info,
    )
    outputs["index"] = index_path
    return outputs
