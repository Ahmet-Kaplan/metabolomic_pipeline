"""Plot 2 — PCA by group.

2-component PCA on the log-transformed abundance matrix. Points are coloured by
``cell_line`` (or injection_type for QC/blank/dilution), shape by
``injection_type``, hover shows ``sample_id``.

Healthy pattern: tight QC cluster; 5 cell-line clusters; blanks far from bio;
dilutions stretching from QC centroid toward origin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from biodisc.eda._loader import Stage1Data
from biodisc.eda._output import write_plot

_MARKER_SHAPES = {
    "biological": "circle",
    "qc": "diamond",
    "blank": "x",
    "dilution": "triangle-up",
}


def plot(
    data: Stage1Data,
    *,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Render the PCA-by-group plot."""
    # Wide matrix: abundances_raw_wide already has biological + QC columns;
    # blanks and dilutions live in separate long tables. For PCA we'll build
    # a unified samples × features matrix from all three sources.
    bio_qc = data.abundances_raw_wide.set_index("feature_id")

    # Pivot blanks + dilutions (raw block only) to wide
    raw_blanks = data.blanks[data.blanks["block"] == "raw"].copy()
    raw_blanks["sample_id"] = "Blank_" + raw_blanks["blank_replicate"].astype(str)
    blanks_wide = raw_blanks.pivot_table(
        index="feature_id", columns="sample_id", values="intensity"
    )

    raw_dil = data.dilution_ladder[
        (data.dilution_ladder["block"] == "raw") & (data.dilution_ladder["dilution_factor"] < 1.0)
    ].copy()
    raw_dil["sample_id"] = raw_dil["dilution_level"].apply(lambda d: f"QC_1_{int(d)}")
    dil_wide = raw_dil.pivot_table(index="feature_id", columns="sample_id", values="intensity")

    full = pd.concat([bio_qc, blanks_wide, dil_wide], axis=1)

    # Take the transpose: rows = samples, cols = features
    X = full.T

    # log1p transform for variance stabilisation; imputation = 0 (fine for PCA exploration)
    X_log = np.log1p(X.fillna(0))
    imp = SimpleImputer(strategy="constant", fill_value=0)
    X_ready = imp.fit_transform(X_log)

    pca = PCA(n_components=2, random_state=42)  # matches global seed cfg.seed=42
    coords = pca.fit_transform(X_ready)
    var_pct = pca.explained_variance_ratio_ * 100

    # Decorate with sample metadata
    pc = pd.DataFrame(coords, columns=["PC1", "PC2"], index=X.index)
    pc.index.name = "sample_id"
    pc = pc.reset_index()
    samples = data.samples.copy()
    samples["cell_line"] = samples["cell_line"].fillna(samples["injection_type"])
    pc = pc.merge(samples[["sample_id", "injection_type", "cell_line"]], on="sample_id", how="left")

    # Build a trace per cell_line so plotly auto-legends; plus one trace each
    # for QC/blank/dilution
    fig = go.Figure()
    for (cell_line, injection_type), group in pc.groupby(["cell_line", "injection_type"]):
        fig.add_trace(
            go.Scatter(
                x=group["PC1"],
                y=group["PC2"],
                mode="markers",
                name=f"{cell_line}",
                marker=dict(
                    size=12,
                    symbol=_MARKER_SHAPES.get(injection_type, "circle"),
                    line=dict(width=1, color="#222"),
                ),
                text=group["sample_id"],
                hovertemplate="%{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
            )
        )

    # Compute cluster diagnostics to put honest numbers in the caption
    bio = pc[pc["injection_type"] == "biological"][["PC1", "PC2"]]
    qc = pc[pc["injection_type"] == "qc"][["PC1", "PC2"]]
    bio_ctr = bio.mean().values
    qc_ctr = qc.mean().values
    bio_radius = np.sqrt(((bio.values - bio_ctr) ** 2).sum(axis=1)).max()
    qc_to_bio = float(np.sqrt(((qc_ctr - bio_ctr) ** 2).sum()))

    # Between/within separation for biological groups
    centroids = {
        cl: sub[["PC1", "PC2"]].mean().values
        for cl, sub in pc[pc["injection_type"] == "biological"].groupby("cell_line")
    }
    within = [
        float(np.sqrt(((sub[["PC1", "PC2"]].values - centroids[cl]) ** 2).sum(axis=1)).mean())
        for cl, sub in pc[pc["injection_type"] == "biological"].groupby("cell_line")
    ]
    within_mean = float(np.mean(within))
    between_dists = [
        float(np.sqrt(((centroids[a] - centroids[b]) ** 2).sum()))
        for i, a in enumerate(centroids)
        for b in list(centroids)[i + 1 :]
    ]
    between_mean = float(np.mean(between_dists))
    sep_ratio = between_mean / within_mean if within_mean > 0 else float("nan")
    qc_inside = qc_to_bio < bio_radius

    fig.update_layout(
        title=dict(
            text=(
                "<b>PCA — log₁+(raw abundance)</b><br>"
                f"<sup>PC1 {var_pct[0]:.1f} % | PC2 {var_pct[1]:.1f} % | "
                "shape = injection type, colour = cell line</sup>"
            )
        ),
        xaxis_title=f"PC1 ({var_pct[0]:.1f} %)",
        yaxis_title=f"PC2 ({var_pct[1]:.1f} %)",
        height=760,
        legend=dict(title="Group"),
        margin=dict(l=60, r=40, t=100, b=160),
    )
    fig.add_annotation(
        text=(
            "<b>Cluster diagnostics:</b><br>"
            f"  Biological between/within separation: <b>{sep_ratio:.2f}</b> "
            "(>2 = well separated)<br>"
            f"  QC distance to biology centroid: {qc_to_bio:.1f} · "
            f"biology radius: {bio_radius:.1f} · "
            f"QC {'INSIDE' if qc_inside else 'OUTSIDE'} bio envelope · "
            f"QC drift along PC1 (see Plot 5)"
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

    return write_plot(fig, output_dir, name="02_pca", emit_png=emit_png)
