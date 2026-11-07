"""Plot 7 — dilution response curves.

Two sub-figures (vertically stacked):
  A. For a random sample of ``n_sample_features`` features, plot the response
     at (QC=1.0, 0.5, 0.25, 0.125) with the linear fit.
  B. Histogram of R²_linear across ALL features — previews how many pass
     Stage 7 (domain-expert rule, confirmed 2026-04-24: linear R² ≥ 0.7 only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from biodisc.eda._loader import Stage1Data
from biodisc.eda._output import write_plot


def _fit_r2(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """Return R² for a polynomial fit of given degree, or NaN if not fittable."""
    if len(x) < degree + 1 or np.any(np.isnan(y)):
        return np.nan
    coefs = np.polyfit(x, y, degree)
    y_pred = np.polyval(coefs, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _build_dilution_matrix(data: Stage1Data) -> pd.DataFrame:
    """Return a DataFrame keyed on feature_id with columns {1.0, 0.5, 0.25, 0.125}."""
    raw = data.dilution_ladder[data.dilution_ladder["block"] == "raw"].copy()
    # Aggregate across replicate injections at each factor (mean)
    agg = raw.groupby(["feature_id", "dilution_factor"])["intensity"].mean().unstack()
    # Ensure canonical column order
    ordered_cols = [1.0, 0.5, 0.25, 0.125]
    return agg.reindex(columns=ordered_cols)


def plot(
    data: Stage1Data,
    *,
    output_dir: Path,
    n_sample_features: int = 30,
    r2_threshold: float = 0.7,
    seed: int = 42,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Render dilution response + R² histograms."""
    mat = _build_dilution_matrix(data)
    factors = np.array(mat.columns.tolist(), dtype=float)

    # Compute linear R² per feature (domain-expert-confirmed rule is linear-only)
    r2_lin: list[float] = []
    for _, row in mat.iterrows():
        y = row.values.astype(float)
        if np.any(np.isnan(y)):
            r2_lin.append(np.nan)
            continue
        r2_lin.append(_fit_r2(factors, y, degree=1))
    mat = mat.assign(r2_linear=r2_lin)
    mat_valid = mat.dropna(subset=["r2_linear"])

    # A subset for drawing individual curves — stratified by R² so we see good + bad
    rng = np.random.default_rng(seed)
    pick_strata = min(n_sample_features // 3, 10)
    q_low = mat_valid["r2_linear"].quantile(0.10)
    q_mid = mat_valid["r2_linear"].quantile(0.50)
    q_hi = mat_valid["r2_linear"].quantile(0.90)

    low = mat_valid[mat_valid["r2_linear"] <= q_low]
    mid = mat_valid[(mat_valid["r2_linear"] > q_low) & (mat_valid["r2_linear"] <= q_mid)]
    hi = mat_valid[mat_valid["r2_linear"] > q_hi]
    samples = []
    for pool in (low, mid, hi):
        if len(pool) > 0:
            take = min(pick_strata, len(pool))
            samples.append(pool.sample(n=take, random_state=rng.integers(0, 2**31 - 1)))
    sampled = pd.concat(samples).head(n_sample_features)

    pass_lin = int((mat_valid["r2_linear"] >= r2_threshold).sum())
    pass_pct = pass_lin / len(mat_valid) * 100 if len(mat_valid) else 0

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"(a) Response curves for {len(sampled)} features (stratified by R²_linear)",
            f"(b) R²_linear distribution across {len(mat_valid):,} features "
            f"— threshold {r2_threshold:g} marked",
        ),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.15,
    )

    # Top: sampled curves coloured by pass/fail
    for fid, row in sampled.iterrows():
        y = row[factors].values.astype(float)
        colour = "#1e4377" if row["r2_linear"] >= r2_threshold else "#c7254e"
        fig.add_trace(
            go.Scatter(
                x=factors,
                y=y,
                mode="lines+markers",
                line=dict(color=colour, width=1),
                marker=dict(size=5),
                name=f"{fid} (R²={row['r2_linear']:.2f})",
                showlegend=False,
                hovertemplate=f"{fid}<br>factor=%{{x}}<br>intensity=%{{y:.0f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(title_text="Dilution factor", autorange="reversed", row=1, col=1)
    fig.update_yaxes(title_text="Intensity", type="log", row=1, col=1)

    # Bottom: R²_linear histogram
    fig.add_trace(
        go.Histogram(
            x=mat_valid["r2_linear"],
            name="R²_linear",
            marker_color="#1e4377",
            opacity=0.85,
            nbinsx=60,
        ),
        row=2,
        col=1,
    )
    fig.add_vline(
        x=r2_threshold,
        line=dict(color="#c7254e", width=2, dash="dash"),
        row=2,
        col=1,
        annotation_text=f"R² ≥ {r2_threshold:g}",
        annotation_position="top right",
    )
    fig.update_xaxes(title_text="R²_linear", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="Feature count", row=2, col=1)

    fig.update_layout(
        title=dict(
            text=(
                "<b>Dilution response + R²_linear distribution (Stage 7 preview)</b><br>"
                f"<sup><b>{pass_lin:,} / {len(mat_valid):,} features ({pass_pct:.1f} %) "
                f"would pass</b> the domain-expert rule: R²_linear ≥ {r2_threshold:g}</sup>"
            )
        ),
        height=900,
        legend=dict(title="Fit"),
        margin=dict(l=60, r=40, t=110, b=60),
    )

    return write_plot(fig, output_dir, name="07_dilution_response", emit_png=emit_png)
