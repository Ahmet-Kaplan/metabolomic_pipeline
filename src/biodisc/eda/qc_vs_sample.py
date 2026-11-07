"""Plot 3 — QC-vs-sample intensity boxplot (per injection).

One box per injection, ordered by (injection_type, sample_id). Y = log₁₀(intensity).
Colour grouped by injection_type so QC vs sample vs blank vs dilution are
visually separated.

Healthy: biological samples similar; QCs tight clump; blanks much lower;
dilutions descending staircase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from biodisc.eda._loader import Stage1Data
from biodisc.eda._output import write_plot

_COLOURS = {
    "biological": "#1e4377",
    "qc": "#c9a94e",
    "blank": "#8b2d2d",
    "dilution": "#4a8a4a",
}


def plot(
    data: Stage1Data,
    *,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Render the QC-vs-sample intensity boxplot."""
    # Assemble a long frame: (sample_id, intensity, injection_type)
    bio_qc = data.abundances_raw_long.copy()
    bio_qc["intensity"] = pd.to_numeric(bio_qc["intensity"], errors="coerce")

    # Blanks (raw block only)
    bk = data.blanks[data.blanks["block"] == "raw"].copy()
    bk["sample_id"] = "Blank_" + bk["blank_replicate"].astype(str)
    bk = bk[["feature_id", "sample_id", "intensity"]]

    # Dilutions (raw block only, true dilutions < 1.0)
    dl = data.dilution_ladder[
        (data.dilution_ladder["block"] == "raw") & (data.dilution_ladder["dilution_factor"] < 1.0)
    ].copy()
    dl["sample_id"] = dl["dilution_level"].apply(lambda d: f"QC_1_{int(d)}")
    dl = dl[["feature_id", "sample_id", "intensity"]]

    long = pd.concat([bio_qc, bk, dl], ignore_index=True)
    long = long.dropna(subset=["intensity"])
    long = long[long["intensity"] > 0]
    long["log_intensity"] = np.log10(long["intensity"])

    # Attach injection_type + column_order for x-axis ordering
    meta = data.samples[["sample_id", "injection_type", "column_order"]]
    long = long.merge(meta, on="sample_id", how="left")

    sample_order = data.samples.sort_values(["injection_type", "column_order"])[
        "sample_id"
    ].tolist()
    long["sample_id"] = pd.Categorical(long["sample_id"], categories=sample_order, ordered=True)
    long = long.sort_values("sample_id")

    fig = go.Figure()
    for injection_type, group in long.groupby("injection_type", observed=True):
        fig.add_trace(
            go.Box(
                x=group["sample_id"].astype(str),
                y=group["log_intensity"],
                name=injection_type,
                marker_color=_COLOURS.get(str(injection_type), "#888"),
                boxpoints=False,
            )
        )
    # Compute biology-vs-blank gap honestly — flag small gaps as data-quality red flag
    bio_med = (
        long[long["injection_type"] == "biological"]
        .groupby("sample_id", observed=True)["log_intensity"]
        .median()
        .median()
    )
    blk_med = (
        long[long["injection_type"] == "blank"]
        .groupby("sample_id", observed=True)["log_intensity"]
        .median()
        .median()
    )
    gap = bio_med - blk_med

    fig.update_layout(
        title=dict(
            text=(
                "<b>Per-injection intensity distribution</b><br>"
                "<sup>log₁₀(raw intensity) per feature. One box = one injection. "
                "Ordered by injection type then acquisition order.</sup>"
            )
        ),
        xaxis=dict(title="Injection", tickangle=-60),
        yaxis=dict(title="log₁₀ intensity"),
        height=760,
        boxmode="group",
        showlegend=True,
        legend=dict(title="Injection type"),
        margin=dict(l=80, r=40, t=100, b=200),
    )
    warning = ""
    if gap < 1.0:
        warning = (
            " <b style='color:#c7254e'>⚠ Data-quality concern: "
            "healthy LC-MS typically shows gap ≥ 2 log units.</b>"
        )
    fig.add_annotation(
        text=(
            f"<b>Biology-to-blank gap:</b> {gap:.2f} log₁₀ units (~{10**gap:.1f}× ratio).{warning}"
        ),
        xref="paper",
        yref="paper",
        x=0.02,
        y=-0.35,
        showarrow=False,
        font=dict(size=10, color="#222"),
        align="left",
        bordercolor="#c9a94e",
        borderwidth=1,
        borderpad=8,
        bgcolor="#fdf9ed",
    )

    return write_plot(fig, output_dir, name="03_qc_vs_sample_box", emit_png=emit_png)
