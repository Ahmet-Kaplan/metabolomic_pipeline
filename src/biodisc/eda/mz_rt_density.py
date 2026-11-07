"""Plot 4 — m/z × RT density (feature-space map).

Scatter of every feature at its (retention_time, m/z) coordinate, coloured by
log₁₀(mean intensity). This is the "star map" of the chemical space covered.

Healthy: smooth diagonal-ish band for reverse-phase column behaviour; no dead
RT zones; no single m/z streaks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from biodisc.eda._loader import Stage1Data
from biodisc.eda._output import write_plot


def plot(
    data: Stage1Data,
    *,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Render the m/z × RT density scatter."""
    # Compute mean intensity per feature across BIOLOGICAL samples only
    bio_samples = data.samples[data.samples["injection_type"] == "biological"]["sample_id"]
    bio_abund = data.abundances_raw_long[data.abundances_raw_long["sample_id"].isin(bio_samples)]
    mean_int = bio_abund.groupby("feature_id")["intensity"].mean().rename("mean_intensity")

    plot_df = data.features[["feature_id", "rt_min", "m_over_z"]].merge(
        mean_int, on="feature_id", how="left"
    )
    plot_df = plot_df.dropna(subset=["mean_intensity"])
    plot_df["log_int"] = np.log10(plot_df["mean_intensity"].clip(lower=1.0))

    fig = go.Figure(
        data=go.Scattergl(
            x=plot_df["rt_min"],
            y=plot_df["m_over_z"],
            mode="markers",
            marker=dict(
                size=4,
                color=plot_df["log_int"],
                colorscale="Viridis",
                colorbar=dict(title="log₁₀(mean int)"),
                opacity=0.55,
            ),
            text=plot_df["feature_id"],
            hovertemplate=(
                "%{text}<br>RT=%{x:.2f} min<br>m/z=%{y:.4f}<br>log₁₀I=%{marker.color:.2f}"
                "<extra></extra>"
            ),
        )
    )
    # Detect m/z streaks — single m/z with many features (possible contaminants)
    mz_rounded = plot_df["m_over_z"].round(2)
    top_mz = mz_rounded.value_counts().head(5)
    streak_text = " · ".join(f"m/z={mz:.2f} ({cnt}×)" for mz, cnt in top_mz.items())
    n_void = int((plot_df["rt_min"] < 0.5).sum())

    fig.update_layout(
        title=dict(
            text=(
                f"<b>m/z × RT feature map</b><br>"
                f"<sup>{len(plot_df):,} features, coloured by log₁₀(mean intensity "
                "across biological samples)</sup>"
            )
        ),
        xaxis_title="Retention time (min)",
        yaxis_title="m/z",
        height=760,
        margin=dict(l=60, r=40, t=100, b=160),
    )
    fig.add_annotation(
        text=(
            f"<b>Diagnostics:</b><br>"
            f"  Void-volume features (RT < 0.5 min): {n_void:,} "
            f"({n_void / len(plot_df) * 100:.1f}%)<br>"
            f"  Top m/z streaks (possible contaminants / isomer clusters): {streak_text}"
        ),
        xref="paper",
        yref="paper",
        x=0.02,
        y=-0.22,
        showarrow=False,
        font=dict(size=10, color="#222"),
        align="left",
        bordercolor="#c9a94e",
        borderwidth=1,
        borderpad=8,
        bgcolor="#fdf9ed",
    )

    return write_plot(fig, output_dir, name="04_mz_rt_density", emit_png=emit_png)
