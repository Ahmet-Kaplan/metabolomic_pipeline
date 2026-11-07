"""Plot 1 — Missingness heatmap (features × samples).

What it shows: which (feature, sample) cells are missing. Rows are features
sorted descending by missingness %; columns are samples sorted by injection_type
then by column_order so groups are visually contiguous.

Healthy pattern: mostly-present matrix with group-private white bands
(features absent only in some cell lines — normal biology).

Sick pattern: whole-column white stripes (failed injection), whole-row white
bands except in QCs (possible analytical artifact).
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
    max_features: int = 5_000,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Render the missingness heatmap.

    Parameters
    ----------
    data
        Stage-1 bundle.
    output_dir
        Directory to write the plot outputs into.
    max_features
        Cap on the number of features drawn (full matrix would be 30k × 53 —
        too dense for a useful PNG). We keep the top ``max_features`` rows
        ranked by missingness %, so the interesting rows (most missing) are
        always shown.
    """
    wide = data.abundances_raw_wide.set_index("feature_id")
    # Order columns by (injection_type, column_order) via the samples table
    samples = data.samples.sort_values(["injection_type", "column_order"])
    # Only keep columns that exist in the wide matrix (= biological + QC)
    col_order = [c for c in samples["sample_id"] if c in wide.columns]
    wide = wide[col_order]

    # Boolean missingness (1 = missing, 0 = present), sorted by row missingness.
    # Progenesis encodes "below LOD" as 0 (not NaN), so count both as missing.
    miss = (wide.isna() | (wide == 0)).astype(np.int8)
    row_miss_pct = miss.mean(axis=1).sort_values(ascending=False)
    if len(row_miss_pct) > max_features:
        keep_idx = row_miss_pct.head(max_features).index
        miss = miss.loc[keep_idx]
        subtitle = (
            f"Showing top {max_features:,} features ranked by missingness % "
            f"(of {len(row_miss_pct):,} total)"
        )
    else:
        miss = miss.loc[row_miss_pct.index]
        subtitle = f"Showing all {len(row_miss_pct):,} features"

    fig = go.Figure(
        data=go.Heatmap(
            z=miss.values,
            x=miss.columns.tolist(),
            y=list(range(len(miss))),  # index; feature_id is too dense to label
            colorscale=[[0, "#0b2545"], [1, "#ffffff"]],
            showscale=False,
            hovertemplate="feature rank: %{y}<br>sample: %{x}<br>missing: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                "<b>Missingness heatmap (NaN OR zero treated as missing)</b><br>"
                f"<sup>{subtitle}. Dark = detected, white = below LOD. "
                "Rows sorted by missingness %. "
                "Note: Progenesis exports below-detection features as 0, not NaN.</sup>"
            )
        ),
        xaxis=dict(title="Sample (grouped by injection type)", tickangle=-45, side="bottom"),
        yaxis=dict(title="Feature (rank)", autorange="reversed"),
        height=700,
        margin=dict(l=80, r=40, t=100, b=140),
    )

    # Summary annotation: how many features fully missing / partially / complete
    fully_missing = (row_miss_pct == 1.0).sum()
    fully_present = (row_miss_pct == 0.0).sum()
    partial = len(row_miss_pct) - fully_missing - fully_present
    fig.add_annotation(
        text=(
            f"fully present: {fully_present:,} | partial: {partial:,} | "
            f"fully missing: {fully_missing:,}"
        ),
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.22,
        showarrow=False,
        font=dict(size=11, color="#555"),
    )

    return write_plot(fig, output_dir, name="01_missingness", emit_png=emit_png)
