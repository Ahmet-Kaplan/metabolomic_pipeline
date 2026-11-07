"""Plot 6 — blank-vs-sample ratio histogram.

Preview of Stage 6 blank filter. For each feature, compute
``log10(mean(biological + QC) / mean(blank))`` and plot the distribution.

Healthy pattern: right-skewed distribution with mass at ratio ≥ 10 (log ≥ 1);
small left tail of contamination-dominated features.

Overlay: vertical line at the default Stage 6 threshold (log₁₀(5) ≈ 0.70)
corresponding to Saeed's ``blank/mean(sample+QC) × 100 > 20 %`` drop rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from biodisc.eda._loader import Stage1Data
from biodisc.eda._output import write_plot


def plot(
    data: Stage1Data,
    *,
    output_dir: Path,
    default_blank_pct_threshold: float = 20.0,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Render the blank-vs-sample histogram.

    ``default_blank_pct_threshold`` — the Stage 6 drop threshold in %
    (``blank / mean(sample + QC) × 100 > threshold → drop``). A vertical marker
    is drawn at the equivalent log₁₀-ratio: ``log₁₀(100 / threshold)``.
    """
    # Mean blank intensity per feature (raw block)
    bk = data.blanks[data.blanks["block"] == "raw"]
    mean_blank = bk.groupby("feature_id")["intensity"].mean().rename("mean_blank")

    # Mean intensity across (biological + QC) samples per feature
    bio_qc_ids = data.samples[data.samples["injection_type"].isin(["biological", "qc"])][
        "sample_id"
    ].tolist()
    bio_qc = data.abundances_raw_long[data.abundances_raw_long["sample_id"].isin(bio_qc_ids)]
    mean_sample = bio_qc.groupby("feature_id")["intensity"].mean().rename("mean_sample")

    df = pd.concat([mean_sample, mean_blank], axis=1)
    total_features = len(df)

    # Classify every feature into one of four buckets
    both_zero = df[(df["mean_sample"] == 0) & (df["mean_blank"] == 0)]
    sample_zero = df[(df["mean_sample"] == 0) & (df["mean_blank"] > 0)]  # dead
    blank_zero = df[(df["mean_sample"] > 0) & (df["mean_blank"] == 0)]  # clean
    both_pos = df[(df["mean_sample"] > 0) & (df["mean_blank"] > 0)].copy()  # boundary
    both_pos["log_ratio"] = np.log10(both_pos["mean_sample"] / both_pos["mean_blank"])

    threshold_log = np.log10(100.0 / default_blank_pct_threshold)  # log10(5) at pct=20

    # Boundary features above/below threshold
    boundary_drop = int((both_pos["log_ratio"] < threshold_log).sum())
    boundary_keep = len(both_pos) - boundary_drop

    # True totals over ALL features
    total_drop = boundary_drop + len(sample_zero)
    total_keep = boundary_keep + len(blank_zero)
    total_ambiguous = len(both_zero)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=both_pos["log_ratio"],
            nbinsx=80,
            marker_color="#1e4377",
            opacity=0.85,
            name="Features (both sample & blank > 0)",
        )
    )
    fig.add_vline(
        x=threshold_log,
        line=dict(color="#c7254e", width=2, dash="dash"),
        annotation_text=(
            f"Stage 6 threshold: log₁₀(100 / {default_blank_pct_threshold:g}) = {threshold_log:.2f}"
        ),
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Blank-vs-sample ratio (Stage 6 preview)</b><br>"
                f"<sup>Histogram shows {len(both_pos):,} boundary features (both sample "
                "& blank non-zero). Features with blank == 0 always pass; features with "
                "sample == 0 always drop.</sup>"
            )
        ),
        xaxis_title="log₁₀(mean_sample / mean_blank)",
        yaxis_title="Feature count",
        height=640,
        bargap=0.02,
        margin=dict(l=60, r=40, t=110, b=160),
    )
    # Summary over ALL features, not just the histogram
    fig.add_annotation(
        text=(
            f"<b>Drop outcome over ALL {total_features:,} features at "
            f"blank_pct_threshold={default_blank_pct_threshold:g}%:</b><br>"
            f"  • Clean (blank=0, always pass): {len(blank_zero):,} ({len(blank_zero) / total_features * 100:.1f}%)<br>"
            f"  • Boundary kept (ratio ≥ thr):  {boundary_keep:,} ({boundary_keep / total_features * 100:.1f}%)<br>"
            f"  • Boundary dropped (ratio < thr): {boundary_drop:,} ({boundary_drop / total_features * 100:.1f}%)<br>"
            f"  • Dead (sample=0, always drop): {len(sample_zero):,} ({len(sample_zero) / total_features * 100:.1f}%)<br>"
            f"  • Both-zero (investigate): {total_ambiguous:,}<br>"
            f"  <b>Total: KEEP {total_keep:,} ({total_keep / total_features * 100:.1f}%) · "
            f"DROP {total_drop:,} ({total_drop / total_features * 100:.1f}%)</b>"
        ),
        xref="paper",
        yref="paper",
        x=0.02,
        y=-0.28,
        showarrow=False,
        font=dict(size=10, color="#222"),
        align="left",
        bordercolor="#c9a94e",
        borderwidth=1,
        borderpad=8,
        bgcolor="#fdf9ed",
    )

    return write_plot(fig, output_dir, name="06_blank_sample_ratio", emit_png=emit_png)
