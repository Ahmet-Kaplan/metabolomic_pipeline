"""Plot 5 — QC run-order drift.

Median (and IQR band) of log₁₀(intensity) across all features vs QC injection
number. If the QC numbering matches chronological injection order (to be
verified with the MS operator), a monotonic trend indicates detector drift.

Caveat: until the MS operator confirms, this plot is informational only; see
plan §12 item 20 (run-order metadata pending).
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
    emit_png: bool = True,
) -> dict[str, Path]:
    """Render the QC drift plot."""
    qc_samples = (
        data.samples[data.samples["injection_type"] == "qc"]
        .sort_values("replicate")
        .reset_index(drop=True)
    )
    qc_ids = qc_samples["sample_id"].tolist()

    long = data.abundances_raw_long[data.abundances_raw_long["sample_id"].isin(qc_ids)].copy()
    long["intensity"] = pd.to_numeric(long["intensity"], errors="coerce")
    long = long[long["intensity"] > 0]
    long["log_int"] = np.log10(long["intensity"])

    # For each QC, summarise log intensity across features: median + IQR
    summary = (
        long.groupby("sample_id")["log_int"]
        .agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            count="count",
        )
        .reindex(qc_ids)
    )
    summary["order"] = summary.index.map({s: i for i, s in enumerate(qc_ids, start=1)})

    fig = go.Figure()
    # IQR band
    fig.add_trace(
        go.Scatter(
            x=summary["order"].tolist() + summary["order"].tolist()[::-1],
            y=summary["q75"].tolist() + summary["q25"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(30,67,119,0.20)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="IQR (25–75 %)",
        )
    )
    # Median line
    fig.add_trace(
        go.Scatter(
            x=summary["order"],
            y=summary["median"],
            mode="lines+markers",
            line=dict(color="#0b2545", width=2),
            marker=dict(size=8),
            name="Median",
            text=summary.index,
            hovertemplate="%{text}<br>order=%{x}<br>median log₁₀ I=%{y:.3f}<extra></extra>",
        )
    )

    # Linear regression to detect trend (diagnostic only)
    x = summary["order"].values.astype(float)
    y = summary["median"].values
    slope, intercept = np.polyfit(x, y, 1)
    trend = slope * x + intercept
    fig.add_trace(
        go.Scatter(
            x=x,
            y=trend,
            mode="lines",
            line=dict(color="#c9a94e", dash="dash", width=1.5),
            name=f"Linear trend (slope={slope:.4f}/injection)",
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>QC run-order drift</b><br>"
                "<sup>Median log₁₀(intensity) across features per QC injection. "
                "Flat = stable; trend = drift (check with MS operator whether QC numbering "
                "is chronological).</sup>"
            )
        ),
        xaxis_title="QC injection number",
        yaxis_title="log₁₀ intensity (median across features)",
        height=600,
    )

    return write_plot(fig, output_dir, name="05_qc_drift", emit_png=emit_png)
