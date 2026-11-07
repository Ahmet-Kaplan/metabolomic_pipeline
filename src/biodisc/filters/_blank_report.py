"""Stage 6 — HTML report (blank filter before/after).

Three plots + an index:

    1. blank_pct histogram (log x-axis) — kept vs dropped + threshold line.
    2. mean_blank vs mean_sq scatter — points above the cutoff line drop.
    3. Drop fate breakdown (waterfall).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from biodisc.eda._output import write_plot
from biodisc.eda.report import ReportEntry, build_index


def plot_pct_histogram(
    metrics: pd.DataFrame,
    *,
    threshold: float,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Histogram of blank_pct (log10 x-axis, capped) coloured by fate."""
    valid = metrics.dropna(subset=["blank_pct"]).copy()
    valid = valid[~np.isinf(valid["blank_pct"])]
    # Use log10(blank_pct + 1) so 0 % maps to 0; cap top at 1e4 % for display
    valid["log_pct"] = np.log10(valid["blank_pct"].clip(upper=1e4) + 1.0)
    log_thr = float(np.log10(threshold + 1.0))

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=valid.loc[~valid["kept"], "log_pct"],
            xbins=dict(start=0.0, end=4.5, size=0.05),
            marker_color="#c7254e",
            opacity=0.85,
            name=f"Dropped (blank_pct > {threshold:g} %)",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=valid.loc[valid["kept"], "log_pct"],
            xbins=dict(start=0.0, end=4.5, size=0.05),
            marker_color="#1e4377",
            opacity=0.85,
            name=f"Kept (blank_pct ≤ {threshold:g} %)",
        )
    )
    fig.add_vline(
        x=log_thr,
        line=dict(color="#c9a94e", width=2, dash="dash"),
        annotation_text=f"log₁₀({threshold:g} + 1) = {log_thr:.2f}",
        annotation_position="top right",
    )

    n_total = len(metrics)
    n_kept = int(metrics["kept"].sum())
    n_inf = int(np.isinf(metrics["blank_pct"]).sum())
    n_nan = int(metrics["blank_pct"].isna().sum())

    fig.update_layout(
        title=dict(
            text=(
                "<b>Blank-percentage distribution (Stage 6 before/after)</b><br>"
                f"<sup>{n_total:,} features · "
                f"<b>KEPT {n_kept:,} ({n_kept / n_total * 100:.1f} %) · "
                f"DROPPED {n_total - n_kept:,} ({(n_total - n_kept) / n_total * 100:.1f} %)</b><br>"
                f"x-axis = log₁₀(blank_pct + 1); display cap 10 000 % "
                f"(n_∞ = {n_inf}, n_NaN = {n_nan})</sup>"
            )
        ),
        xaxis_title="log₁₀(blank_pct + 1)  —  i.e. log scale of blank/sample × 100",
        yaxis_title="Feature count",
        barmode="stack",
        bargap=0.02,
        height=600,
        legend=dict(title="Fate"),
    )
    return write_plot(fig, output_dir, name="01_pct_histogram", emit_png=emit_png)


def plot_blank_vs_sample(
    metrics: pd.DataFrame,
    *,
    threshold: float,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """log10 mean_blank vs log10 mean_sq scatter; cutoff = y = (threshold/100)·x."""
    valid = metrics.copy()
    valid = valid[(valid["mean_blank"] > 0) & (valid["mean_sq"] > 0)]
    valid["log_blank"] = np.log10(valid["mean_blank"])
    valid["log_sq"] = np.log10(valid["mean_sq"])
    kept = valid[valid["kept"]]
    dropped = valid[~valid["kept"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=dropped["log_sq"],
            y=dropped["log_blank"],
            mode="markers",
            marker=dict(size=3, color="#c7254e", opacity=0.45),
            name=f"Dropped ({len(dropped):,})",
            text=dropped["feature_id"],
            hovertemplate="%{text}<br>log_sq=%{x:.2f}<br>log_blank=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=kept["log_sq"],
            y=kept["log_blank"],
            mode="markers",
            marker=dict(size=3, color="#1e4377", opacity=0.35),
            name=f"Kept ({len(kept):,})",
            text=kept["feature_id"],
            hovertemplate="%{text}<br>log_sq=%{x:.2f}<br>log_blank=%{y:.2f}<extra></extra>",
        )
    )

    # Cutoff line: blank = (threshold/100) · sq, i.e. log_blank = log_sq + log10(thr/100)
    if not valid.empty:
        x_min = float(valid["log_sq"].min())
        x_max = float(valid["log_sq"].max())
        offset = float(np.log10(threshold / 100.0))
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[x_min + offset, x_max + offset],
                mode="lines",
                line=dict(color="#c9a94e", width=2, dash="dash"),
                name=f"Cutoff: blank = {threshold:g}% × sq",
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        title=dict(
            text=(
                "<b>mean_blank vs mean_sample+QC (the geometry of Stage 6)</b><br>"
                "<sup>Points ABOVE the dashed line have blank_pct > threshold → DROPPED. "
                "Points below → KEPT. Both axes log₁₀(intensity).</sup>"
            )
        ),
        xaxis_title="log₁₀(mean intensity in sample+QC)",
        yaxis_title="log₁₀(mean intensity in blanks)",
        height=640,
        legend=dict(title="Fate"),
    )
    return write_plot(fig, output_dir, name="02_blank_vs_sample", emit_png=emit_png)


def plot_drop_breakdown(
    *,
    threshold: float,
    n_input: int,
    n_kept: int,
    n_dropped_pct: int,
    n_dropped_detections: int,
    n_dropped_undefined: int,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Bar chart breaking input into the four buckets."""
    labels = [
        "Input",
        "Kept",
        "Dropped — blank_pct > thr",
        "Dropped — insufficient sample/QC detections",
        "Dropped — undefined",
    ]
    values = [n_input, n_kept, n_dropped_pct, n_dropped_detections, n_dropped_undefined]
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
                "<b>Stage 6 fate breakdown</b><br>"
                f"<sup>Threshold blank_pct > {threshold:g} %.</sup>"
            )
        ),
        xaxis_title="Bucket",
        yaxis_title="Feature count",
        height=540,
        showlegend=False,
    )
    return write_plot(fig, output_dir, name="03_drop_breakdown", emit_png=emit_png)


def build_report(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    threshold: float,
    n_input: int,
    n_kept: int,
    n_dropped_pct: int,
    n_dropped_detections: int,
    n_dropped_undefined: int,
    emit_png: bool = True,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    entries: list[ReportEntry] = []
    outputs: dict[str, Path] = {}

    p1 = plot_pct_histogram(metrics, threshold=threshold, output_dir=plots_dir, emit_png=emit_png)
    outputs["01_pct_histogram"] = p1["html"]
    entries.append(
        ReportEntry(
            key="01_pct_histogram",
            title="1. Blank-pct distribution (before/after)",
            description=(
                "Histogram of per-feature blank_pct on log scale, coloured by fate. "
                "Threshold marks the drop boundary."
            ),
            html_href=f"plots/{p1['html'].name}",
            png_href=f"plots/{p1['png'].name}" if "png" in p1 else None,
        )
    )

    p2 = plot_blank_vs_sample(metrics, threshold=threshold, output_dir=plots_dir, emit_png=emit_png)
    outputs["02_blank_vs_sample"] = p2["html"]
    entries.append(
        ReportEntry(
            key="02_blank_vs_sample",
            title="2. mean_blank vs mean_sample+QC scatter",
            description=(
                "Geometric view: each point is a feature; the cutoff line "
                f"separates blank_pct ≤ {threshold:g}% (below) from > {threshold:g}% (above)."
            ),
            html_href=f"plots/{p2['html'].name}",
            png_href=f"plots/{p2['png'].name}" if "png" in p2 else None,
        )
    )

    p3 = plot_drop_breakdown(
        threshold=threshold,
        n_input=n_input,
        n_kept=n_kept,
        n_dropped_pct=n_dropped_pct,
        n_dropped_detections=n_dropped_detections,
        n_dropped_undefined=n_dropped_undefined,
        output_dir=plots_dir,
        emit_png=emit_png,
    )
    outputs["03_drop_breakdown"] = p3["html"]
    entries.append(
        ReportEntry(
            key="03_drop_breakdown",
            title="3. Stage 6 fate breakdown",
            description="Input → kept + three drop reasons.",
            html_href=f"plots/{p3['html'].name}",
            png_href=f"plots/{p3['png'].name}" if "png" in p3 else None,
        )
    )

    extra_info = {
        "Input features (Stage-5-kept)": f"{n_input:,}",
        "Kept": f"{n_kept:,} ({n_kept / n_input * 100:.2f} %)",
        "Dropped — blank_pct > thr": f"{n_dropped_pct:,}",
        "Dropped — insufficient sample/QC detections": f"{n_dropped_detections:,}",
        "Dropped — blank_pct undefined": f"{n_dropped_undefined:,}",
        "Threshold": f"blank_pct > {threshold} % ⇒ drop",
    }
    index_path = build_index(
        output_path=output_dir / "index.html",
        entries=entries,
        stage_label="Stage 6 — Blank filter (% formula)",
        extra_info=extra_info,
    )
    outputs["index"] = index_path
    return outputs
