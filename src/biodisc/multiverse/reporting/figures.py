"""Shared Plotly figure builders used across stage-section modules.

Every figure here is **publication-grade by default**: serif font,
``simple_white`` template, fixed class palette, axis titles always
present. Stage-specific modules import these and pass in stage data —
they never touch ``go.Figure`` plumbing directly.

Helpers fall into a few rough buckets:

* **Distribution overlays** — KDE / histogram / violin pairs for
  before-vs-after comparisons (the workhorse for imputation, drift,
  normalisation, transformation diagnostics).
* **Scatter family** — heteroscedasticity / mean-vs-SD / paired score
  comparisons.
* **Trajectory family** — per-sample line traces ordered by run-order
  (drift QC trajectories, sample-sum boxplots).
* **Class-coloured plots** — anything coloured by ``cell_line`` or
  ``injection_type`` routes through ``_color_for_label`` so the same
  cell line is the same colour in PCA, score plots, volcano legends,
  etc.

All builders return a ``go.Figure`` already styled by ``apply_theme``.
The caller adds it to a ``PathReport`` via ``rpt.add_figure(...)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from biodisc.multiverse.reporting.theme import (
    CLASS_COLORS,
    apply_theme,
    color_for,
    display_class,
)


# ============================================================
# Class-colour resolution
# ============================================================
def _color_for_label(label: str, fallback_index: int = 0) -> str:
    """Look up a fixed class colour, with a stable fallback for unknown labels."""
    return color_for(label, fallback_index=fallback_index)


def _legend_marker_size(n: int) -> int:
    """Smaller markers for crowded plots, larger for sparse."""
    if n > 200:
        return 4
    if n > 50:
        return 6
    return 8


# ============================================================
# Distribution helpers
# ============================================================
def _kde_xy(values: np.ndarray, *, n_grid: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) for a Gaussian KDE; (empty, empty) if input is degenerate."""
    finite = values[np.isfinite(values)]
    if finite.size < 3 or float(np.std(finite)) == 0.0:
        return np.array([]), np.array([])
    x = np.linspace(float(finite.min()), float(finite.max()), n_grid)
    try:
        y = gaussian_kde(finite)(x)
    except (np.linalg.LinAlgError, ValueError):
        return np.array([]), np.array([])
    return x, np.asarray(y, dtype=float)


def kde_overlay(
    arrays: Sequence[np.ndarray],
    names: Sequence[str],
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str = "density",
    colors: Sequence[str] | None = None,
    fill: bool = True,
) -> go.Figure:
    """Overlay multiple KDE curves on a shared axis.

    Used for ``imputed vs observed`` distributions, ``before vs after``
    skewness, etc. Empty arrays are silently skipped.
    """
    palette = colors or ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#888888")
    fig = go.Figure()
    for i, (arr, name) in enumerate(zip(arrays, names, strict=False)):
        x, y = _kde_xy(np.asarray(arr, dtype=float))
        if x.size == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line={"color": palette[i % len(palette)], "width": 2.0},
                fill="tozeroy" if fill else None,
                fillcolor=_with_alpha(palette[i % len(palette)], 0.15) if fill else None,
                hovertemplate=f"{name}<br>%{{x:.3g}}, %{{y:.3g}}<extra></extra>",
            )
        )
    apply_theme(fig, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig


def histogram_overlay(
    arrays: Sequence[np.ndarray],
    names: Sequence[str],
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str = "feature count",
    nbins: int = 60,
    colors: Sequence[str] | None = None,
    barmode: str = "overlay",
) -> go.Figure:
    """Overlay multiple histograms with shared bin edges."""
    palette = colors or ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    fig = go.Figure()
    for i, (arr, name) in enumerate(zip(arrays, names, strict=False)):
        finite = np.asarray(arr, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        fig.add_trace(
            go.Histogram(
                x=finite,
                name=name,
                nbinsx=nbins,
                marker={"color": _with_alpha(palette[i % len(palette)], 0.55)},
                opacity=0.75,
            )
        )
    fig.update_layout(barmode=barmode)
    apply_theme(fig, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig


def violin_paired(
    arrays: Sequence[np.ndarray],
    names: Sequence[str],
    *,
    title: str,
    yaxis_title: str,
    colors: Sequence[str] | None = None,
) -> go.Figure:
    """Side-by-side violins (one per array)."""
    palette = colors or ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    fig = go.Figure()
    for i, (arr, name) in enumerate(zip(arrays, names, strict=False)):
        finite = np.asarray(arr, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        fig.add_trace(
            go.Violin(
                y=finite,
                name=name,
                box_visible=True,
                meanline_visible=True,
                line_color=palette[i % len(palette)],
                fillcolor=_with_alpha(palette[i % len(palette)], 0.30),
                opacity=0.85,
            )
        )
    apply_theme(fig, title=title, xaxis_title="state", yaxis_title=yaxis_title)
    return fig


# ============================================================
# Scatter helpers
# ============================================================
def scatter_with_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    name: str = "features",
    color: str = "#0072B2",
    show_fit: bool = True,
    annotation: str | None = None,
) -> go.Figure:
    """Scatter with optional OLS line; used for log-mean × log-SD plots."""
    fig = go.Figure()
    finite = np.isfinite(x) & np.isfinite(y)
    xf = x[finite]
    yf = y[finite]
    fig.add_trace(
        go.Scattergl(
            x=xf,
            y=yf,
            mode="markers",
            name=name,
            marker={"color": color, "size": 4, "opacity": 0.4},
            hovertemplate="%{x:.3g}, %{y:.3g}<extra></extra>",
        )
    )
    if show_fit and xf.size >= 5 and float(np.std(xf)) > 0:
        slope, intercept = np.polyfit(xf, yf, deg=1)
        x_line = np.linspace(float(xf.min()), float(xf.max()), 50)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line={"color": "#D55E00", "width": 2.0, "dash": "dash"},
                name=f"OLS fit (slope = {slope:+.3f})",
            )
        )
    if annotation is not None:
        fig.add_annotation(
            text=annotation,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            align="left",
            font={"family": "ui-monospace, SF Mono, Menlo, monospace", "size": 11},
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#bbbbbb",
            borderwidth=1,
        )
    apply_theme(fig, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig


# ============================================================
# Per-sample helpers (boxplot / strip / bar over samples)
# ============================================================
def per_sample_boxplot(
    matrix: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    title: str,
    yaxis_title: str,
    use_log1p: bool = True,
    hover_extra: str | None = None,
) -> go.Figure:
    """One box per sample column, coloured by injection type or cell_line.

    Used for normalisation diagnostics (sample sums / row distributions).
    Boxes ordered by ``column_order`` if available, else by sample id.
    """
    cols = list(matrix.columns)
    samples_idx = samples.set_index("sample_id")
    # Order — column_order if available; else alphabetic
    if "column_order" in samples_idx.columns:
        ordered = samples_idx.loc[cols].sort_values("column_order")
        cols = ordered.index.tolist()

    fig = go.Figure()
    for sid in cols:
        row = samples_idx.loc[sid]
        cell_line = str(row.get("cell_line", "") or "")
        inj_type = str(row.get("injection_type", "") or "")
        # Color: cell line if biological, else use injection-type tag
        color_key = cell_line if inj_type == "biological" and cell_line else inj_type
        col_color = _color_for_label(color_key)
        values = matrix[sid].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        if use_log1p and float(finite.min()) >= 0:
            finite = np.log1p(finite)
        fig.add_trace(
            go.Box(
                y=finite,
                name=sid,
                marker={"color": col_color, "size": 3},
                line={"color": col_color, "width": 1.2},
                fillcolor=_with_alpha(col_color, 0.30),
                boxpoints=False,
                hovertemplate=f"sample: {sid}<br>{inj_type} / {cell_line}<extra></extra>",
            )
        )
    if hover_extra is None:
        hover_extra = ""
    apply_theme(
        fig,
        title=title,
        xaxis_title=f"sample (ordered by column_order){hover_extra}",
        yaxis_title=yaxis_title + (" — log1p" if use_log1p else ""),
    )
    fig.update_layout(showlegend=False, xaxis={"tickangle": -55, "tickfont": {"size": 9}})
    return fig


# ============================================================
# Run-order trajectories
# ============================================================
def run_order_trajectories(
    matrix_before: pd.DataFrame | None,
    matrix_after: pd.DataFrame,
    samples: pd.DataFrame,
    run_order: pd.DataFrame,
    *,
    feature_ids: Sequence[str],
    title: str,
    yaxis_title: str,
    n_features_show: int = 8,
    use_log_y: bool = True,
) -> go.Figure:
    """Plot per-feature intensity vs run-order position, before/after correction.

    Solid line = after, dotted line = before. Vertical hairlines mark QC
    injection positions. Used for the QC drift trajectory figure.

    The y-axis is log10 by default (``use_log_y=True``) — without it,
    high-abundance features dominate the plot and low-abundance traces
    look flat. Drift is a *relative* phenomenon; log scale preserves
    relative shape while making cross-feature comparison readable.
    """
    fids = list(feature_ids)[:n_features_show]
    if not fids:
        fig = go.Figure()
        apply_theme(fig, title=title + " (no features)", xaxis_title="run order", yaxis_title=yaxis_title)
        return fig

    pos_map = run_order.set_index("raw_sample_id")["position"].to_dict()
    sample_meta = samples.set_index("sample_id")
    # Build position per matrix column
    sid_to_pos: dict[str, int] = {}
    sid_to_inj: dict[str, str] = {}
    for sid in matrix_after.columns:
        if sid not in sample_meta.index:
            continue
        raw_id = sample_meta.loc[sid, "raw_sample_id"] if "raw_sample_id" in sample_meta.columns else sid
        if raw_id in pos_map:
            sid_to_pos[sid] = int(pos_map[raw_id])
            sid_to_inj[sid] = str(sample_meta.loc[sid, "injection_type"])

    if not sid_to_pos:
        # No run-order linkage — fall back to column order
        for i, sid in enumerate(matrix_after.columns):
            sid_to_pos[sid] = i
            inj = sample_meta.loc[sid, "injection_type"] if sid in sample_meta.index else "unknown"
            sid_to_inj[sid] = str(inj)

    palette = [
        *list(CLASS_COLORS.values())[:n_features_show],
        "#56B4E9", "#882255", "#117733", "#999933", "#AA4499", "#332288",
    ]

    fig = go.Figure()
    # Order columns by run position
    cols_sorted = sorted(sid_to_pos.keys(), key=lambda s: sid_to_pos[s])
    positions = np.array([sid_to_pos[s] for s in cols_sorted])
    inj_arr = np.array([sid_to_inj[s] for s in cols_sorted])
    qc_mask = inj_arr == "qc"

    for i, fid in enumerate(fids):
        col = palette[i % len(palette)]
        if matrix_before is not None and fid in matrix_before.index:
            yb = np.asarray(
                matrix_before.loc[fid].reindex(cols_sorted).to_numpy(),
                dtype=float,
            )
            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=yb,
                    mode="lines+markers",
                    line={"color": col, "width": 1.2, "dash": "dot"},
                    marker={"color": col, "size": 5, "opacity": 0.5},
                    name=f"{fid} (before)",
                    legendgroup=fid,
                    hovertemplate=f"{fid} (before)<br>pos: %{{x}}<br>val: %{{y:.3g}}<extra></extra>",
                )
            )
        if fid in matrix_after.index:
            ya = np.asarray(
                matrix_after.loc[fid].reindex(cols_sorted).to_numpy(),
                dtype=float,
            )
            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=ya,
                    mode="lines+markers",
                    line={"color": col, "width": 2.2},
                    marker={"color": col, "size": 6},
                    name=f"{fid} (after)",
                    legendgroup=fid,
                    hovertemplate=f"{fid} (after)<br>pos: %{{x}}<br>val: %{{y:.3g}}<extra></extra>",
                )
            )
        # Mark QC positions on x-axis once per feature with a vertical hint
    # Add QC position markers as a band / small annotations
    if qc_mask.any():
        qc_positions = positions[qc_mask]
        for qp in qc_positions:
            fig.add_vline(
                x=int(qp),
                line={"color": "#888888", "width": 0.5, "dash": "dot"},
                opacity=0.4,
            )

    final_yaxis_title = (
        f"{yaxis_title} (log10)" if use_log_y else yaxis_title
    )
    apply_theme(
        fig,
        title=title,
        xaxis_title="injection position (run order)",
        yaxis_title=final_yaxis_title,
    )
    if use_log_y:
        fig.update_yaxes(type="log")
    fig.update_layout(legend={"font": {"size": 10}})
    return fig


# ============================================================
# Heatmap helper (sub-sampled missingness)
# ============================================================
def missingness_heatmap(
    matrix: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    title: str,
    max_features: int = 600,
    treat_zero_as_missing: bool = True,
) -> go.Figure:
    """Compact missingness heatmap for embedding in a section.

    Sub-samples to ``max_features`` rows ranked by missing rate so dense
    matrices stay legible. Columns ordered by injection type then
    column_order so groups are contiguous.
    """
    # Order columns
    if {"injection_type", "column_order"}.issubset(samples.columns):
        ord_samples = samples.sort_values(["injection_type", "column_order"])
        cols = [c for c in ord_samples["sample_id"] if c in matrix.columns]
    else:
        cols = list(matrix.columns)
    sub = matrix[cols]

    if treat_zero_as_missing:
        miss = (sub.isna() | (sub == 0)).astype(np.int8)
    else:
        miss = sub.isna().astype(np.int8)

    row_miss = miss.mean(axis=1).sort_values(ascending=False)
    if len(row_miss) > max_features:
        keep = row_miss.head(max_features).index
        miss = miss.loc[keep]
        subtitle = f"top {max_features:,} of {len(row_miss):,} features (most missing first)"
    else:
        miss = miss.loc[row_miss.index]
        subtitle = f"{len(row_miss):,} features"

    fig = go.Figure(
        data=go.Heatmap(
            z=miss.values,
            x=miss.columns.tolist(),
            y=list(range(len(miss))),
            colorscale=[[0, "#0b2545"], [1, "#ffffff"]],
            showscale=False,
            hovertemplate="rank %{y}<br>%{x}<br>missing: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        height=420,
        margin={"l": 60, "r": 30, "t": 60, "b": 100},
        xaxis={"tickangle": -55, "tickfont": {"size": 9}},
        yaxis={"autorange": "reversed", "title": "feature rank"},
    )
    apply_theme(fig, title=title + f"<br><sup>{subtitle}</sup>", xaxis_title="sample", yaxis_title="feature rank")
    return fig


# ============================================================
# PCA scores helper (used by drift + outlier sections)
# ============================================================
def pca_scores_2d(
    matrix: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    title: str,
    use_log1p: bool = True,
    color_by: str = "cell_line",
    add_class_ellipses: bool = False,
    annotate_runorder: bool = False,
    run_order: pd.DataFrame | None = None,
) -> go.Figure:
    """PC1×PC2 scatter, samples coloured by ``color_by`` column.

    Optionally draws 95 % confidence ellipses per class — used for the
    drift section ("biology preserved post-correction") and the outlier
    section ("Hotelling T² ellipse boundary").
    """
    from sklearn.decomposition import PCA

    from biodisc.multiverse.reporting.helpers import confidence_ellipse_2d

    cols = [c for c in matrix.columns if c in samples["sample_id"].values]
    if len(cols) < 3:
        fig = go.Figure()
        apply_theme(fig, title=title + " (too few samples)", xaxis_title="PC1", yaxis_title="PC2")
        return fig

    sub = matrix[cols]
    arr = sub.to_numpy(dtype=float)
    if use_log1p and float(np.nanmin(arr)) >= 0:
        arr = np.log1p(arr)
    X = arr.T  # (n_samples, n_features)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_
    samples_idx = samples.set_index("sample_id")
    labels = samples_idx.loc[cols, color_by].astype(str).to_numpy()

    fig = go.Figure()
    unique_labels: list[str] = sorted(set(labels))
    for j, lab in enumerate(unique_labels):
        mask = labels == lab
        pts_x = scores[mask, 0]
        pts_y = scores[mask, 1]
        col = _color_for_label(lab, fallback_index=j)
        text_arr = np.array(cols)[mask]
        fig.add_trace(
            go.Scatter(
                x=pts_x,
                y=pts_y,
                mode="markers+text" if mask.sum() <= 6 else "markers",
                marker={"color": col, "size": 9, "line": {"width": 0.7, "color": "#222"}},
                text=text_arr if mask.sum() <= 6 else None,
                textposition="top center",
                textfont={"size": 9, "color": "#444"},
                name=str(lab),
                hovertemplate="%{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
            )
        )
        if add_class_ellipses and pts_x.size >= 3:
            rx, ry = confidence_ellipse_2d(pts_x, pts_y)
            if rx.size > 0:
                fig.add_trace(
                    go.Scatter(
                        x=rx,
                        y=ry,
                        mode="lines",
                        line={"color": col, "width": 1.5, "dash": "dot"},
                        name=f"{lab} 95 % CI",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    if annotate_runorder and run_order is not None:
        # Add a hidden trace with run_order in customdata for hover
        try:
            pos_map = run_order.set_index("raw_sample_id")["position"].to_dict()
            raw_col = "raw_sample_id" if "raw_sample_id" in samples_idx.columns else "sample_id"
            positions = [pos_map.get(samples_idx.loc[c, raw_col], None) for c in cols]
            fig.add_trace(
                go.Scatter(
                    x=scores[:, 0],
                    y=scores[:, 1],
                    mode="markers",
                    marker={"size": 0.001, "color": "rgba(0,0,0,0)"},
                    customdata=np.array(positions, dtype=object).reshape(-1, 1),
                    hovertemplate="run pos: %{customdata[0]}<extra></extra>",
                    showlegend=False,
                )
            )
        except Exception:
            pass

    apply_theme(
        fig,
        title=title,
        xaxis_title=f"PC1 ({var_ratio[0] * 100:.1f} %)",
        yaxis_title=f"PC2 ({var_ratio[1] * 100:.1f} %)",
    )
    return fig


# ============================================================
# Class + QC PCA (full sample set with 95 % ellipses)
# ============================================================
def pca_class_and_qc_ellipses(
    matrix: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    title: str,
    use_log1p: bool = True,
    flagged_sample_ids: set[str] | None = None,
    var_explained_override: tuple[float, float] | None = None,
) -> go.Figure:
    """Full-sample PCA scatter with one 95 % CI ellipse per class + QC pool.

    Used by the drift section (full-sample PCA pre / post correction) and
    the outlier section (post-Pareto PCA with class structure visible).

    Parameters
    ----------
    matrix
        Wide feature × sample matrix. Will be log1p-transformed when
        ``use_log1p=True`` (default — appropriate for raw / imputed /
        drift-corrected / normalised matrices). Set False for already-log
        or already-scaled matrices to avoid double-transformation.
    samples
        Sample metadata with ``sample_id`` / ``injection_type`` /
        ``cell_line`` columns. Real-data convention: QC / blank / dilution
        rows have ``cell_line=None``; this helper uses ``display_class``
        so they get proper labels.
    flagged_sample_ids
        Optional set of sample ids to label with text annotation. Useful
        in outlier panels.
    var_explained_override
        Override PC1/PC2 explained-variance ratios in the axis titles —
        only useful when ``matrix`` was already PC-projected upstream.
    """
    from sklearn.decomposition import PCA

    from biodisc.multiverse.reporting.helpers import confidence_ellipse_2d

    cols = [c for c in matrix.columns if c in samples["sample_id"].values]
    if len(cols) < 3:
        fig = go.Figure()
        apply_theme(fig, title=title + " (too few samples)", xaxis_title="PC1", yaxis_title="PC2")
        return fig

    sub = matrix[cols]
    arr = sub.to_numpy(dtype=float)
    if use_log1p and float(np.nanmin(arr)) >= 0:
        arr = np.log1p(arr)
    X = arr.T  # rows = samples, cols = features
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)
    var_ratio = (
        var_explained_override
        if var_explained_override is not None
        else (
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1]),
        )
    )
    samples_idx = samples.set_index("sample_id")
    labels = np.array([
        display_class(
            samples_idx.loc[sid, "cell_line"],
            samples_idx.loc[sid, "injection_type"],
        )
        for sid in cols
    ])
    inj_types = samples_idx.loc[cols, "injection_type"].astype(str).to_numpy()
    flagged = flagged_sample_ids or set()

    # Plot biological classes first, then qc / blank / dilution last so
    # neutral-grey QC ellipse sits on top of bright biological ellipses.
    _ORDER = {"qc": 1000, "blank": 1001, "dilution": 1002}
    unique_groups = sorted(set(labels), key=lambda g: (_ORDER.get(g, 0), g))

    fig = go.Figure()
    for j, group in enumerate(unique_groups):
        mask = labels == group
        if not mask.any():
            continue
        col = color_for(group, fallback_index=j)
        pts_x = scores[mask, 0]
        pts_y = scores[mask, 1]
        sids = np.array(cols)[mask]
        text = np.array([sid if sid in flagged else "" for sid in sids])
        sample_inj = inj_types[mask]
        symbol = (
            "circle" if (sample_inj == "biological").all()
            else "diamond" if (sample_inj == "qc").all()
            else "square"
        )
        fig.add_trace(
            go.Scatter(
                x=pts_x,
                y=pts_y,
                mode="markers+text",
                marker={
                    "color": col,
                    "size": 11,
                    "symbol": symbol,
                    "line": {"width": 0.7, "color": "#222"},
                },
                text=text,
                textposition="top center",
                textfont={"size": 9, "color": "#D55E00"},
                name=group,
                customdata=sids,
                hovertemplate=(
                    "%{customdata}<br>"
                    f"class: {group}<br>PC1=%{{x:.3f}}<br>PC2=%{{y:.3f}}<extra></extra>"
                ),
            )
        )
        if pts_x.size >= 3:
            rx, ry = confidence_ellipse_2d(pts_x, pts_y)
            if rx.size > 0:
                fig.add_trace(
                    go.Scatter(
                        x=rx,
                        y=ry,
                        mode="lines",
                        line={"color": col, "width": 1.5, "dash": "dot"},
                        name=f"{group} 95 % CI",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    apply_theme(
        fig,
        title=title,
        xaxis_title=f"PC1 ({var_ratio[0] * 100:.1f} %)",
        yaxis_title=f"PC2 ({var_ratio[1] * 100:.1f} %)",
    )
    return fig


# ============================================================
# Bar / paired-comparison helpers
# ============================================================
def paired_metric_bar(
    metric_names: Sequence[str],
    before: Sequence[float],
    after: Sequence[float],
    *,
    title: str,
    yaxis_title: str,
    before_color: str = "#999999",
    after_color: str = "#0072B2",
) -> go.Figure:
    """Side-by-side before/after bars per metric — used for headline metric panels."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(metric_names),
            y=list(before),
            name="before",
            marker={"color": before_color},
            hovertemplate="%{x}<br>before: %{y:.3g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=list(metric_names),
            y=list(after),
            name="after",
            marker={"color": after_color},
            hovertemplate="%{x}<br>after: %{y:.3g}<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    apply_theme(fig, title=title, xaxis_title="metric", yaxis_title=yaxis_title)
    return fig


def per_sample_bar(
    sample_ids: Sequence[str],
    values: Sequence[float],
    samples: pd.DataFrame,
    *,
    title: str,
    yaxis_title: str,
    color_by: str = "injection_type",
) -> go.Figure:
    """Bar plot, one bar per sample, coloured by metadata column."""
    samples_idx = samples.set_index("sample_id")
    fig = go.Figure()
    by_label: dict[str, list[tuple[str, float]]] = {}
    for sid, v in zip(sample_ids, values, strict=False):
        label = (
            "unknown"
            if sid not in samples_idx.index
            else str(samples_idx.loc[sid, color_by])
        )
        by_label.setdefault(label, []).append((sid, float(v)))
    for j, (label, items) in enumerate(sorted(by_label.items())):
        col = _color_for_label(label, fallback_index=j)
        xs, ys = zip(*items, strict=False)
        fig.add_trace(
            go.Bar(
                x=list(xs),
                y=list(ys),
                name=label,
                marker={"color": col},
                hovertemplate="%{x}<br>%{y:.3g}<extra></extra>",
            )
        )
    apply_theme(fig, title=title, xaxis_title="sample", yaxis_title=yaxis_title)
    fig.update_layout(barmode="stack", xaxis={"tickangle": -55, "tickfont": {"size": 9}})
    return fig


# ============================================================
# Phase 8e.7 — multiverse index figures
# ============================================================
def specification_curve_bar(
    *,
    path_names: list[str],
    n_shortlisted: list[int],
    n_per_tier_per_path: list[dict[str, int]] | None = None,
    title: str = "Specification curve — shortlist size per path",
) -> go.Figure:
    """Bar chart of n_shortlisted per path, sorted ascending.

    Inspired by Simonsohn et al. 2020's specification curve. Sorted
    ascending so the reader can read instability from left to right.
    Optional ``n_per_tier_per_path`` (per-path tier counts) renders the
    bar as a stacked composition of tier_1 / tier_2 / tier_3 to show
    *what kind* of features each path flagged.
    """
    order = np.argsort(n_shortlisted)
    sorted_names = [path_names[i] for i in order]
    sorted_n = [int(n_shortlisted[i]) for i in order]

    fig = go.Figure()
    if n_per_tier_per_path is None:
        fig.add_trace(
            go.Bar(
                x=sorted_names,
                y=sorted_n,
                marker={"color": color_for("MCF7"), "line": {"color": "#222", "width": 0.6}},
                text=[str(n) for n in sorted_n],
                textposition="outside",
                hovertemplate="%{x}<br>n_shortlisted: %{y}<extra></extra>",
                name="shortlisted",
            )
        )
    else:
        sorted_tiers = [n_per_tier_per_path[i] for i in order]
        for tier, color in (
            ("tier_1", "#0072B2"),
            ("tier_2", "#56B4E9"),
            ("tier_3", "#CCCCCC"),
        ):
            fig.add_trace(
                go.Bar(
                    x=sorted_names,
                    y=[t.get(tier, 0) for t in sorted_tiers],
                    marker={"color": color, "line": {"color": "#222", "width": 0.6}},
                    name=tier,
                    hovertemplate="%{x}<br>" + tier + ": %{y}<extra></extra>",
                )
            )
        fig.update_layout(barmode="stack")

    apply_theme(
        fig,
        title=title,
        xaxis_title="path",
        yaxis_title="n features in path's shortlist",
    )
    fig.update_layout(xaxis={"tickangle": -25, "tickfont": {"size": 11}})
    return fig


def jaccard_similarity_heatmap(
    *,
    path_names: list[str],
    feature_sets: list[set[str]],
    title: str = "Cross-path Jaccard similarity of shortlists",
) -> go.Figure:
    """Pairwise Jaccard heatmap (symmetric, diagonal = 1.0)."""
    n = len(path_names)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = feature_sets[i], feature_sets[j]
            denom = len(a | b)
            jac = len(a & b) / denom if denom else 0.0
            matrix[i, j] = jac
            matrix[j, i] = jac

    text = [[f"{matrix[i, j]:.2f}" for j in range(n)] for i in range(n)]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=path_names,
            y=path_names,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 12, "color": "#111"},
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": {"text": "Jaccard"}, "len": 0.6},
            hovertemplate="%{y} ∩ %{x}: J = %{z:.3f}<extra></extra>",
        )
    )
    apply_theme(fig, title=title, xaxis_title="path", yaxis_title="path")
    fig.update_layout(
        xaxis={"tickangle": -25, "tickfont": {"size": 10}},
        yaxis={"tickfont": {"size": 10}},
    )
    return fig


def consensus_tier_bar(
    *,
    n_per_tier: dict[str, int],
    title: str = "Consensus tier breakdown",
) -> go.Figure:
    """Horizontal bar chart of features per consensus tier.

    tier_1 (≥75% of paths) → strongest, drawn at top.
    """
    order = ["tier_1", "tier_2", "tier_3", "none"]
    labels = ["tier_1 (≥75 % paths)", "tier_2 (50–75 %)", "tier_3 (1–50 %)", "none (0 %)"]
    counts = [int(n_per_tier.get(t, 0)) for t in order]
    palette = ["#0072B2", "#56B4E9", "#CCCCCC", "#F0F0F0"]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker={"color": palette, "line": {"color": "#222", "width": 0.6}},
            text=[str(c) for c in counts],
            textposition="outside",
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")  # tier_1 on top
    apply_theme(fig, title=title, xaxis_title="n features", yaxis_title="tier")
    fig.update_layout(showlegend=False)
    return fig


def method_drilldown_heatmap(
    *,
    feature_ids: list[str],
    path_names: list[str],
    method_names: list[str],
    flag_cube: np.ndarray,
    title: str = "Method evidence per path × method",
) -> go.Figure:
    """Binary heatmap rows=features, cols=(path, method) cells.

    ``flag_cube`` shape (n_features, n_paths, n_methods). Cell value
    is 1 if that method's gate passed in that path for that feature.
    Visually reveals whether the consensus is multi-method or
    dominated by one method (flat horizontal stripes = one method).
    """
    n_feat, n_paths, n_methods = flag_cube.shape
    # Reshape to a (features × (paths × methods)) 2D matrix
    z = flag_cube.reshape(n_feat, n_paths * n_methods).astype(float)
    # Column labels: e.g. "path_a:univ"
    col_labels = []
    for p in path_names:
        for m in method_names:
            short_p = p.replace("path_", "p_")
            short_m = m[:4]
            col_labels.append(f"{short_p}:{short_m}")

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=col_labels,
            y=feature_ids,
            colorscale=[[0, "#fafafa"], [1, color_for("MCF7")]],
            showscale=False,
            hovertemplate="feature %{y}<br>%{x}: %{z:.0f}<extra></extra>",
            xgap=1,
            ygap=1,
        )
    )
    # Vertical separators between paths
    for i in range(1, n_paths):
        fig.add_vline(
            x=i * n_methods - 0.5,
            line={"color": "#222", "width": 1.0},
        )
    apply_theme(fig, title=title, xaxis_title="path × method", yaxis_title="feature")
    fig.update_layout(
        xaxis={"tickangle": -55, "tickfont": {"size": 9}},
        yaxis={"tickfont": {"size": 9}, "autorange": "reversed"},
        height=max(360, 16 * n_feat + 120),
    )
    return fig


# ============================================================
# Tiny helpers
# ============================================================
def _with_alpha(hex_color: str, alpha: float) -> str:
    """Convert ``#RRGGBB`` to ``rgba(r,g,b,a)``. Alpha clipped to [0, 1]."""
    if not hex_color.startswith("#") or len(hex_color) != 7:
        return hex_color
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    a = max(0.0, min(1.0, alpha))
    return f"rgba({r},{g},{b},{a:.3f})"


def _to_finite_array(values: Iterable[Any]) -> np.ndarray:
    """Cast to float array and drop non-finite — used by histogram helpers."""
    arr = np.asarray(list(values), dtype=float)
    return np.asarray(arr[np.isfinite(arr)], dtype=float)
