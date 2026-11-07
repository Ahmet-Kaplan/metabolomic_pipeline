"""Stage 7 — HTML report (dilution-linearity R² before/after).

Three plots + an index:

    1. R² histogram — threshold line + kept/dropped colouring.
    2. Stratified curve overlay — 6 sampled features (low / mid / high R²)
       drawn on a shared (factor → intensity) panel so you can see *visually*
       what passes and what fails.
    3. Drop fate breakdown bar chart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from biodisc.eda._output import write_plot
from biodisc.eda.report import ReportEntry, build_index

# Same factor order as the filter module.
_FACTORS: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)


def plot_r2_histogram(
    metrics: pd.DataFrame,
    *,
    threshold: float,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Histogram of R²_linear coloured by fate."""
    valid = metrics.dropna(subset=["r2_linear"]).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=valid.loc[~valid["kept"], "r2_linear"],
            xbins=dict(start=0.0, end=1.001, size=0.02),
            marker_color="#c7254e",
            opacity=0.85,
            name=f"Dropped (R² < {threshold:g})",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=valid.loc[valid["kept"], "r2_linear"],
            xbins=dict(start=0.0, end=1.001, size=0.02),
            marker_color="#1e4377",
            opacity=0.85,
            name=f"Kept (R² ≥ {threshold:g})",
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

    fig.update_layout(
        title=dict(
            text=(
                "<b>Dilution-linearity R² distribution (Stage 7 before/after)</b><br>"
                f"<sup>{n_total:,} features (post-Stage-6) · "
                f"<b>KEPT {n_kept:,} ({n_kept / n_total * 100:.1f} %) · "
                f"DROPPED {n_dropped:,} ({n_dropped / n_total * 100:.1f} %)</b></sup>"
            )
        ),
        xaxis_title="R²_linear",
        yaxis_title="Feature count",
        barmode="stack",
        bargap=0.02,
        height=600,
        legend=dict(title="Fate"),
    )
    return write_plot(fig, output_dir, name="01_r2_histogram", emit_png=emit_png)


def plot_curve_overlay(
    metrics: pd.DataFrame,
    *,
    threshold: float,
    output_dir: Path,
    n_sample: int = 6,
    seed: int = 42,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Overlay of sampled feature dilution curves — stratified by R²."""
    valid = metrics.dropna(subset=["r2_linear"]).copy()
    rng = np.random.default_rng(seed)

    # Stratify into three R² bands — pick n_sample/3 from each
    per_band = max(n_sample // 3, 1)
    high = valid[valid["r2_linear"] >= 0.95].sample(
        n=min(per_band, len(valid[valid["r2_linear"] >= 0.95])),
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    mid = valid[(valid["r2_linear"] >= threshold) & (valid["r2_linear"] < 0.95)].sample(
        n=min(
            per_band, len(valid[(valid["r2_linear"] >= threshold) & (valid["r2_linear"] < 0.95)])
        ),
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    low = valid[valid["r2_linear"] < threshold].sample(
        n=min(per_band, len(valid[valid["r2_linear"] < threshold])),
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    sampled = pd.concat([high, mid, low])

    y_cols = ["y_factor_1", "y_factor_0_5", "y_factor_0_25", "y_factor_0_125"]

    fig = go.Figure()
    for _, row in sampled.iterrows():
        y_vals = [row[c] for c in y_cols]
        colour = "#1e4377" if row["kept"] else "#c7254e"
        fig.add_trace(
            go.Scatter(
                x=list(_FACTORS),
                y=y_vals,
                mode="lines+markers",
                line=dict(color=colour, width=1.5),
                marker=dict(size=8),
                name=f"{row['feature_id']} (R²={row['r2_linear']:.2f})",
                hovertemplate=f"{row['feature_id']}<br>factor=%{{x}}<br>intensity=%{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Dilution response curves (stratified sample)</b><br>"
                "<sup>Each line is one feature: blue = KEPT (high R²), red = DROPPED. "
                "Healthy curves descend roughly linearly with dilution factor.</sup>"
            )
        ),
        xaxis=dict(title="Dilution factor", autorange="reversed"),
        yaxis=dict(title="Intensity (log scale)", type="log"),
        height=620,
    )
    return write_plot(fig, output_dir, name="02_curve_overlay", emit_png=emit_png)


def plot_drop_breakdown(
    *,
    threshold: float,
    n_input: int,
    n_kept: int,
    n_dropped_r2: int,
    n_dropped_detections: int,
    n_dropped_undefined: int,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Bar chart of input → kept + three drop reasons."""
    labels = [
        "Input",
        "Kept",
        "Dropped — R² < thr",
        "Dropped — insufficient detections",
        "Dropped — R² undefined",
    ]
    values = [n_input, n_kept, n_dropped_r2, n_dropped_detections, n_dropped_undefined]
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
            text=(f"<b>Stage 7 fate breakdown</b><br><sup>Threshold R² ≥ {threshold:g}.</sup>")
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
    n_dropped_r2: int,
    n_dropped_detections: int,
    n_dropped_undefined: int,
    emit_png: bool = True,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    entries: list[ReportEntry] = []
    outputs: dict[str, Path] = {}

    p1 = plot_r2_histogram(metrics, threshold=threshold, output_dir=plots_dir, emit_png=emit_png)
    outputs["01_r2_histogram"] = p1["html"]
    entries.append(
        ReportEntry(
            key="01_r2_histogram",
            title="1. R² distribution (before/after)",
            description=(
                "Histogram of per-feature R²_linear coloured by fate. "
                "The threshold line marks the drop boundary."
            ),
            html_href=f"plots/{p1['html'].name}",
            png_href=f"plots/{p1['png'].name}" if "png" in p1 else None,
        )
    )

    p2 = plot_curve_overlay(metrics, threshold=threshold, output_dir=plots_dir, emit_png=emit_png)
    outputs["02_curve_overlay"] = p2["html"]
    entries.append(
        ReportEntry(
            key="02_curve_overlay",
            title="2. Sampled dilution-response curves",
            description=(
                "Stratified sample (high / mid / low R²) of feature trajectories "
                "across the 4 dilution factors. Visually shows what the threshold separates."
            ),
            html_href=f"plots/{p2['html'].name}",
            png_href=f"plots/{p2['png'].name}" if "png" in p2 else None,
        )
    )

    p3 = plot_drop_breakdown(
        threshold=threshold,
        n_input=n_input,
        n_kept=n_kept,
        n_dropped_r2=n_dropped_r2,
        n_dropped_detections=n_dropped_detections,
        n_dropped_undefined=n_dropped_undefined,
        output_dir=plots_dir,
        emit_png=emit_png,
    )
    outputs["03_drop_breakdown"] = p3["html"]
    entries.append(
        ReportEntry(
            key="03_drop_breakdown",
            title="3. Stage 7 fate breakdown",
            description="Input → kept + three drop reasons.",
            html_href=f"plots/{p3['html'].name}",
            png_href=f"plots/{p3['png'].name}" if "png" in p3 else None,
        )
    )

    extra_info = {
        "Input features (Stage-6-kept)": f"{n_input:,}",
        "Kept": f"{n_kept:,} ({n_kept / n_input * 100:.2f} %)",
        "Dropped — R² < threshold": f"{n_dropped_r2:,}",
        "Dropped — insufficient detections": f"{n_dropped_detections:,}",
        "Dropped — R² undefined": f"{n_dropped_undefined:,}",
        "Threshold": f"R²_linear ≥ {threshold} ⇒ keep",
    }
    index_path = build_index(
        output_path=output_dir / "index.html",
        entries=entries,
        stage_label="Stage 7 — Dilution linearity (R²_linear)",
        extra_info=extra_info,
    )
    outputs["index"] = index_path
    return outputs
