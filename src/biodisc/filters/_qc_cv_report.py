"""Stage 4 — HTML report (QC-CV filter before/after).

Three plots + an index:

    1. CV% histogram — every feature placed at its qc_cv_pct (log x-axis),
       coloured by kept vs dropped, threshold line marked.
    2. Example QC traces — three feature trajectories across the 16 QC
       injections (low-CV, borderline, high-CV) showing what the threshold
       is actually separating.
    3. CV% vs log mean-intensity scatter — reveals whether high-CV features
       are concentrated at low intensity (the typical heteroscedastic pattern).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from biodisc.eda._output import write_plot
from biodisc.eda.report import ReportEntry, build_index


def plot_cv_histogram(
    metrics: pd.DataFrame,
    *,
    threshold: float,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Histogram of qc_cv_pct coloured by fate."""
    valid = metrics.dropna(subset=["qc_cv_pct"]).copy()
    cv = valid["qc_cv_pct"]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=cv[cv > threshold],
            xbins=dict(start=0.0, end=300.0, size=2.0),
            marker_color="#c7254e",
            opacity=0.85,
            name=f"Dropped (> {threshold:g} %)",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=cv[cv <= threshold],
            xbins=dict(start=0.0, end=300.0, size=2.0),
            marker_color="#1e4377",
            opacity=0.85,
            name=f"Kept (≤ {threshold:g} %)",
        )
    )
    fig.add_vline(
        x=threshold,
        line=dict(color="#c9a94e", width=2, dash="dash"),
        annotation_text=f"Threshold {threshold:g} %",
        annotation_position="top right",
    )

    n_total = len(metrics)
    n_kept = int(metrics["kept"].sum())
    n_dropped = n_total - n_kept
    med = cv.median()
    p95 = cv.quantile(0.95)

    fig.update_layout(
        title=dict(
            text=(
                "<b>QC-CV % distribution (Stage 4 before/after)</b><br>"
                f"<sup>{n_total:,} features (post-Stage-3) · "
                f"<b>KEPT {n_kept:,} ({n_kept / n_total * 100:.1f} %) · "
                f"DROPPED {n_dropped:,} ({n_dropped / n_total * 100:.1f} %)</b> · "
                f"median {med:.1f} % · 95th pct {p95:.1f} %</sup>"
            )
        ),
        xaxis_title="QC-CV % (capped at 300 for display)",
        yaxis_title="Feature count",
        barmode="stack",
        bargap=0.02,
        height=600,
        legend=dict(title="Fate"),
    )
    return write_plot(fig, output_dir, name="01_cv_histogram", emit_png=emit_png)


def plot_example_traces(
    metrics: pd.DataFrame,
    abundances_long: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    threshold: float,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Three QC-intensity traces: low-CV, borderline, high-CV."""
    qc_samples = samples[samples["injection_type"] == "qc"].sort_values("replicate")
    qc_ids = qc_samples["sample_id"].tolist()

    # Pick representative features: fully detected in all 16 QCs
    fully = metrics[metrics["qc_n_detected"] == 16].dropna(subset=["qc_cv_pct"])
    if len(fully) < 3:
        # Fallback: pick whatever we have
        fully = metrics.dropna(subset=["qc_cv_pct"])

    low = fully.nsmallest(1, "qc_cv_pct").iloc[0]
    high = fully.nlargest(1, "qc_cv_pct").iloc[0]
    border_idx = (fully["qc_cv_pct"] - threshold).abs().idxmin()
    border = fully.loc[border_idx]

    picks = [
        ("LOW CV (most stable)", low, "#1e4377"),
        (f"BORDERLINE (~{threshold:.0f} %)", border, "#c9a94e"),
        ("HIGH CV (most unstable)", high, "#c7254e"),
    ]

    fig = go.Figure()
    for label, row, colour in picks:
        fid = row["feature_id"]
        vals = (
            abundances_long[
                (abundances_long["feature_id"] == fid) & (abundances_long["sample_id"].isin(qc_ids))
            ]
            .set_index("sample_id")["intensity"]
            .reindex(qc_ids)
            .to_numpy(dtype=float)
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(qc_ids) + 1)),
                y=vals,
                mode="lines+markers",
                line=dict(color=colour, width=2),
                marker=dict(size=7),
                name=f"{label}: {fid}  (CV={row['qc_cv_pct']:.1f} %)",
                hovertemplate=(f"{fid}<br>QC #%{{x}}<br>intensity=%{{y:.0f}}<extra></extra>"),
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Example QC traces — what the CV threshold separates</b><br>"
                "<sup>Each line is ONE feature's intensity across the 16 QC injections. "
                "Flat = stable (low CV). Spiky = unstable (high CV).</sup>"
            )
        ),
        xaxis_title="QC injection number",
        yaxis_title="Intensity (log scale)",
        yaxis_type="log",
        height=620,
        legend=dict(title="Feature"),
    )
    return write_plot(fig, output_dir, name="02_example_traces", emit_png=emit_png)


def plot_cv_vs_intensity(
    metrics: pd.DataFrame,
    *,
    threshold: float,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """CV% vs log10(mean QC intensity) — diagnostic for heteroscedasticity."""
    valid = metrics.dropna(subset=["qc_cv_pct", "qc_mean"]).copy()
    valid = valid[valid["qc_mean"] > 0]
    valid["log_mean"] = np.log10(valid["qc_mean"])

    kept = valid[valid["kept"]]
    dropped = valid[~valid["kept"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=kept["log_mean"],
            y=kept["qc_cv_pct"],
            mode="markers",
            marker=dict(size=3, color="#1e4377", opacity=0.35),
            name=f"Kept ({len(kept):,})",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=dropped["log_mean"],
            y=dropped["qc_cv_pct"],
            mode="markers",
            marker=dict(size=3, color="#c7254e", opacity=0.45),
            name=f"Dropped ({len(dropped):,})",
            text=dropped["feature_id"],
            hovertemplate=("%{text}<br>log₁₀(QC mean)=%{x:.2f}<br>CV=%{y:.2f} %<extra></extra>"),
        )
    )
    fig.add_hline(
        y=threshold,
        line=dict(color="#c9a94e", width=2, dash="dash"),
        annotation_text=f"Threshold {threshold:g} %",
        annotation_position="top right",
    )
    fig.update_layout(
        title=dict(
            text=(
                "<b>CV % vs log₁₀(QC mean intensity)</b><br>"
                "<sup>Heteroscedasticity check: does CV rise as intensity falls? "
                "If yes, drift correction + normalisation will help; "
                "features not in this pattern are genuinely unreliable.</sup>"
            )
        ),
        xaxis_title="log₁₀(mean QC intensity)",
        yaxis_title="CV % (capped display 500)",
        yaxis=dict(range=[0, 500]),
        height=620,
    )
    return write_plot(fig, output_dir, name="03_cv_vs_intensity", emit_png=emit_png)


def build_report(
    metrics: pd.DataFrame,
    abundances_long: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    output_dir: Path,
    threshold: float,
    min_qc_detections: int,
    n_input: int,
    n_kept: int,
    n_dropped_cv: int,
    n_dropped_detections: int,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Generate all Stage 4 plots + index.html."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    entries: list[ReportEntry] = []
    outputs: dict[str, Path] = {}

    p1 = plot_cv_histogram(metrics, threshold=threshold, output_dir=plots_dir, emit_png=emit_png)
    outputs["01_cv_histogram"] = p1["html"]
    entries.append(
        ReportEntry(
            key="01_cv_histogram",
            title="1. QC-CV % distribution (before/after)",
            description=(
                "Histogram of per-feature CV % across the 16 QCs, coloured by kept "
                "vs dropped. Threshold line marks the drop cutoff."
            ),
            html_href=f"plots/{p1['html'].name}",
            png_href=f"plots/{p1['png'].name}" if "png" in p1 else None,
        )
    )

    p2 = plot_example_traces(
        metrics,
        abundances_long,
        samples,
        threshold=threshold,
        output_dir=plots_dir,
        emit_png=emit_png,
    )
    outputs["02_example_traces"] = p2["html"]
    entries.append(
        ReportEntry(
            key="02_example_traces",
            title="2. Example QC traces",
            description=(
                "Three representative features (low-CV / borderline / high-CV) "
                "plotted across the 16 QC injections so you can see what kind of "
                "instability the threshold is filtering."
            ),
            html_href=f"plots/{p2['html'].name}",
            png_href=f"plots/{p2['png'].name}" if "png" in p2 else None,
        )
    )

    p3 = plot_cv_vs_intensity(metrics, threshold=threshold, output_dir=plots_dir, emit_png=emit_png)
    outputs["03_cv_vs_intensity"] = p3["html"]
    entries.append(
        ReportEntry(
            key="03_cv_vs_intensity",
            title="3. CV % vs mean QC intensity",
            description=(
                "Heteroscedasticity diagnostic. Low-intensity features typically "
                "have higher CV; that pattern suggests drift correction + "
                "normalisation will improve them."
            ),
            html_href=f"plots/{p3['html'].name}",
            png_href=f"plots/{p3['png'].name}" if "png" in p3 else None,
        )
    )

    total_dropped = n_dropped_cv + n_dropped_detections
    extra_info = {
        "Input features (Stage-3-kept)": f"{n_input:,}",
        "Kept": f"{n_kept:,} ({n_kept / n_input * 100:.2f} %)",
        "Dropped — CV exceeds threshold": f"{n_dropped_cv:,}",
        "Dropped — insufficient QC detections": f"{n_dropped_detections:,}",
        "Total dropped": f"{total_dropped:,} ({total_dropped / n_input * 100:.2f} %)",
        "CV threshold": f"{threshold} %",
        "Min QC detections": f"{min_qc_detections}",
    }
    index_path = build_index(
        output_path=output_dir / "index.html",
        entries=entries,
        stage_label="Stage 4 — QC-CV Filter",
        extra_info=extra_info,
    )
    outputs["index"] = index_path
    return outputs
