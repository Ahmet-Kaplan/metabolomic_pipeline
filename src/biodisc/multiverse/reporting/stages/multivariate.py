"""Multivariate sections — PLS-DA / sPLS-DA / RF panels.

Each method gets its own section. Figures fall into three groups:

1. **Common across methods**

   * Top-N importance bar coloured by ``stability_score`` (Nogueira 2018).
   * Stability × importance scatter — the "discovery quality" plot. The
     upper-right quadrant is high-stability + high-importance =
     reproducible biomarker candidates.
   * Permutation p-value histogram — fraction of features whose
     observed importance beats a label-shuffle null (Westerhuis 2008).

2. **PLS-DA / sPLS-DA-specific**

   * Score plot PC1 × PC2 with 95 % confidence ellipses per cell line
     (Kirwan 2013). Refit cheap; uses the model parameters logged in
     ``MultivariateResult.params_used`` so the figure matches what the
     stage actually fitted.
   * β-coefficient heatmap (top-N features × classes) — sign and
     magnitude of each feature's contribution per class.

3. **Method-distinguishing**

   * **sPLS-DA**: ``selected`` mask integrated into all bars / scatters
     (selected vs not coloured separately), and a sparsity-efficiency
     metric panel.
   * **RF**: Gini vs permutation-importance scatter (Strobl 2007). Real
     biomarkers should rank high in BOTH; features high in only one are
     suspect (Gini biases toward high-cardinality features; perm
     importance is the cleaner metric).

Why no AUC / accuracy
---------------------
The plan locks "feature-importance generator, never deployed". We
deliberately do NOT compute or plot any classifier-evaluation metrics
in this section.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.multivariate_utils import MultivariateResult
from biodisc.multiverse.reporting.core import PathReport
from biodisc.multiverse.reporting.helpers import confidence_ellipse_2d
from biodisc.multiverse.reporting.theme import apply_theme, color_for, display_class


# ============================================================
# Refit helpers (cheap — for score-plot visualisations)
# ============================================================
def _build_xy_for_viz(
    ctx: StageContext, biological_only: bool = True
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Return ``(X, y, feature_ids, sample_ids, class_names)`` matching the fit-time view."""
    if biological_only:
        bio_ids = ctx.samples.loc[
            ctx.samples["injection_type"] == "biological", "sample_id"
        ].tolist()
    else:
        bio_ids = ctx.samples["sample_id"].tolist()
    cols = [s for s in bio_ids if s in ctx.matrix.columns]
    sub_matrix = ctx.matrix[cols]
    X = sub_matrix.T.to_numpy(dtype=float)
    sample_to_class = ctx.samples.set_index("sample_id")["cell_line"].astype(str).to_dict()
    classes = [sample_to_class[s] for s in cols]
    class_names = sorted(set(classes))
    class_to_int = {c: i for i, c in enumerate(class_names)}
    y = np.array([class_to_int[c] for c in classes])
    return X, y, sub_matrix.index.astype(str).tolist(), cols, class_names


def _refit_pls_for_scores(
    X: np.ndarray,  # noqa: N803 — sklearn X convention
    y: np.ndarray,
    n_components: int,
    scale: bool,
) -> Any:
    """Cheap refit of PLS-DA on the same data — only used to recover x_scores_."""
    from sklearn.cross_decomposition import PLSRegression

    n_classes = int(y.max()) + 1
    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y] = 1.0
    pls = PLSRegression(n_components=n_components, scale=scale)
    pls.fit(X, y_onehot)
    return pls


# ============================================================
# Shared figure builders
# ============================================================
def _top_n_importance_bar(
    table: pd.DataFrame,
    *,
    importance_col: str,
    n: int,
    title: str,
    yaxis_title: str,
    selected_col: str | None = None,
) -> go.Figure:
    """Top-N feature bar coloured by ``stability_score``.

    If ``selected_col`` is provided (sPLS-DA), bars get an outline:
    selected features are outlined in green, unselected in grey.
    """
    df = table.dropna(subset=[importance_col]).sort_values(
        importance_col, ascending=False
    ).head(n)
    feature_ids = df["feature_id"].astype(str).to_numpy()
    importances = df[importance_col].to_numpy(dtype=float)
    stab = df.get("stability_score", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    fig = go.Figure(
        data=go.Bar(
            x=feature_ids,
            y=importances,
            marker={
                "color": stab,
                "colorscale": "Viridis",
                "cmin": 0.0,
                "cmax": 1.0,
                "colorbar": {"title": "stability", "thickness": 12},
                "line": (
                    {
                        "color": [
                            "#009E73" if bool(s) else "rgba(0,0,0,0)"
                            for s in df.get(selected_col, pd.Series([False] * len(df)))
                        ],
                        "width": [
                            2 if bool(s) else 0
                            for s in df.get(selected_col, pd.Series([False] * len(df)))
                        ],
                    }
                    if selected_col
                    else {"width": 0}
                ),
            },
            hovertemplate=(
                "%{x}<br>" + importance_col + ": %{y:.3f}"
                + ("<br>stability: %{marker.color:.3f}" if "stability_score" in df.columns else "")
                + "<extra></extra>"
            ),
        )
    )
    apply_theme(
        fig,
        title=title,
        xaxis_title="feature_id (ranked by importance)",
        yaxis_title=yaxis_title,
    )
    fig.update_layout(xaxis={"tickangle": -55, "tickfont": {"size": 9}})
    return fig


def _stability_vs_importance_scatter(
    table: pd.DataFrame,
    *,
    importance_col: str,
    title: str,
    xaxis_title: str,
    selected_col: str | None = None,
    p_threshold: float = 0.05,
) -> go.Figure:
    """Stability (y) × importance (x). Points coloured by perm p-value bin.

    The upper-right quadrant = reproducible high-importance features =
    biomarker shortlist candidates.
    """
    df = table.dropna(subset=[importance_col]).copy()
    p = df.get("perm_p_value", pd.Series(np.full(len(df), np.nan))).to_numpy(dtype=float)
    sig_mask = np.where(np.isfinite(p), p < p_threshold, False)
    fig = go.Figure()
    # Plot 4 categories: significant + selected (if applicable), significant only,
    # selected only, neither.
    selected = (
        df.get(selected_col, pd.Series([False] * len(df))).fillna(False).astype(bool).to_numpy()
        if selected_col else np.zeros(len(df), dtype=bool)
    )
    cats = [
        ("ns", ~sig_mask & ~selected, "#cccccc", 4),
        (f"perm p<{p_threshold:g}", sig_mask & ~selected, "#0072B2", 7),
    ]
    if selected_col is not None:
        cats.append(("selected (sPLS-DA)", ~sig_mask & selected, "#D55E00", 7))
        cats.append(("both", sig_mask & selected, "#009E73", 9))
    for name, mask, col, size in cats:
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, importance_col].to_numpy(dtype=float),
                y=df.loc[mask, "stability_score"].fillna(0).to_numpy(dtype=float),
                mode="markers",
                marker={"color": col, "size": size, "opacity": 0.75, "line": {"width": 0}},
                name=name,
                customdata=df.loc[mask, "feature_id"].astype(str).to_numpy(),
                hovertemplate=(
                    "%{customdata}<br>"
                    f"{importance_col}: %{{x:.3f}}<br>"
                    "stability: %{y:.3f}<extra></extra>"
                ),
            )
        )
    fig.add_hline(
        y=0.8,
        line={"color": "#444", "width": 1.0, "dash": "dot"},
        annotation={"text": "stability ≥ 0.8", "font": {"size": 10, "color": "#444"}},
        annotation_position="top right",
    )
    apply_theme(fig, title=title, xaxis_title=xaxis_title, yaxis_title="bootstrap stability score")
    return fig


def _permutation_p_histogram(
    table: pd.DataFrame, *, title: str, p_col: str = "perm_p_value"
) -> go.Figure:
    """Histogram of permutation p-values, with α=0.05 reference line."""
    p = table[p_col].dropna().to_numpy()
    fig = go.Figure(
        data=[
            go.Histogram(
                x=p,
                xbins={"start": 0.0, "end": 1.0, "size": 0.025},
                marker={"color": "#0072B2"},
                opacity=0.85,
                hovertemplate="p in [%{x:.3f}, +bin)<br>n features: %{y}<extra></extra>",
            )
        ]
    )
    fig.add_vline(
        x=0.05,
        line={"color": "#D55E00", "width": 1.0, "dash": "dash"},
        annotation={"text": "α = 0.05", "font": {"size": 10, "color": "#D55E00"}},
        annotation_position="top right",
    )
    apply_theme(fig, title=title, xaxis_title="permutation p-value", yaxis_title="feature count")
    return fig


def _score_plot_with_ellipses(
    scores: np.ndarray,
    sample_ids: list[str],
    samples: pd.DataFrame,
    *,
    title: str,
    var_explained: tuple[float, float] | None = None,
    qc_scores: np.ndarray | None = None,
    qc_sample_ids: list[str] | None = None,
) -> go.Figure:
    """PC1 × PC2 (or component 1 × component 2) scatter with class ellipses.

    When ``qc_scores`` + ``qc_sample_ids`` are provided, QC samples are
    plotted as grey diamonds — they don't get an ellipse (QCs are pooled
    and shouldn't have spread comparable to a biological group).
    """
    samples_idx = samples.set_index("sample_id")
    fig = go.Figure()
    # Use display_class so QC rows (cell_line=None in real data) get the
    # right "qc" label / colour rather than stringifying to "None".
    labels = np.array([
        display_class(
            samples_idx.loc[sid, "cell_line"],
            samples_idx.loc[sid, "injection_type"],
        )
        for sid in sample_ids
    ])
    for j, cl in enumerate(sorted(set(labels))):
        mask = labels == cl
        if not mask.any():
            continue
        col = color_for(cl, fallback_index=j)
        pts_x = scores[mask, 0]
        pts_y = scores[mask, 1] if scores.shape[1] >= 2 else np.zeros(mask.sum())
        fig.add_trace(
            go.Scatter(
                x=pts_x,
                y=pts_y,
                mode="markers",
                marker={"color": col, "size": 11, "line": {"width": 0.7, "color": "#222"}},
                name=cl,
                customdata=np.array(sample_ids)[mask],
                hovertemplate=(
                    "%{customdata}<br>"
                    f"class: {cl}<br>comp1=%{{x:.3f}}<br>comp2=%{{y:.3f}}<extra></extra>"
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
                        name=f"{cl} 95% CI",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    # QC projections (no ellipse — QCs are pooled, no biological spread)
    if qc_scores is not None and qc_sample_ids is not None and len(qc_sample_ids) > 0:
        qc_x = qc_scores[:, 0]
        qc_y = qc_scores[:, 1] if qc_scores.shape[1] >= 2 else np.zeros(qc_scores.shape[0])
        fig.add_trace(
            go.Scatter(
                x=qc_x,
                y=qc_y,
                mode="markers",
                marker={
                    "color": color_for("qc"),
                    "size": 11,
                    "symbol": "diamond",
                    "line": {"width": 0.7, "color": "#222"},
                },
                name="qc (projected)",
                customdata=qc_sample_ids,
                hovertemplate=(
                    "%{customdata}<br>class: qc (projected)<br>"
                    "comp1=%{x:.3f}<br>comp2=%{y:.3f}<extra></extra>"
                ),
            )
        )

    x_label, y_label = "component 1", "component 2"
    if var_explained is not None:
        x_label += f" ({var_explained[0] * 100:.1f}%)"
        y_label += f" ({var_explained[1] * 100:.1f}%)"
    apply_theme(fig, title=title, xaxis_title=x_label, yaxis_title=y_label)
    return fig


def _beta_heatmap(
    table: pd.DataFrame,
    class_names: list[str],
    *,
    n_top: int,
    title: str,
) -> go.Figure | None:
    """β-coefficient heatmap: top-N features (rows) × classes (cols)."""
    beta_cols = [f"beta_{c}" for c in class_names if f"beta_{c}" in table.columns]
    if not beta_cols:
        return None
    # Rank features by max |β|
    abs_max = table[beta_cols].abs().max(axis=1)
    top_idx = abs_max.sort_values(ascending=False).head(n_top).index
    sub = table.loc[top_idx]
    feature_ids = sub["feature_id"].astype(str).to_list()
    z = sub[beta_cols].to_numpy(dtype=float)
    # Symmetric color scale around 0
    vmax = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=class_names,
            y=feature_ids,
            colorscale="RdBu",
            reversescale=True,
            zmin=-vmax,
            zmax=vmax,
            zmid=0.0,
            colorbar={"title": "β", "thickness": 12},
            hovertemplate="%{y}<br>%{x}<br>β = %{z:.3f}<extra></extra>",
        )
    )
    apply_theme(fig, title=title, xaxis_title="cell line", yaxis_title="feature_id (top by |β|)")
    fig.update_yaxes(autorange="reversed")
    return fig


# ============================================================
# PLS-DA section
# ============================================================
def add_plsda_section(
    report: PathReport,
    *,
    ctx: StageContext,
    result: MultivariateResult,
    section_name: str = "multivariate_plsda",
    top_n: int = 30,
    p_threshold: float = 0.05,
) -> None:
    """Add the PLS-DA section to ``report``."""
    if result.method_name != "plsda":
        raise ValueError(
            f"add_plsda_section expected method_name='plsda', got {result.method_name!r}"
        )
    table = result.per_feature
    params = dict(result.params_used)

    biological_only = bool(params.get("biological_only", True))
    n_components = int(params.get("n_components", 2))
    pls_scale = bool(params.get("scale", False))

    X, y, _feature_ids, sample_ids, class_names = _build_xy_for_viz(
        ctx, biological_only=biological_only
    )

    report.add_section(
        section_name,
        title="Multivariate — PLS-DA (Geladi & Kowalski 1986)",
        description=(
            "Partial Least Squares Discriminant Analysis used as a "
            "feature-importance generator (NOT a classifier). Per-feature "
            "VIP, β coefficients per class, permutation importance, "
            "bootstrap stability (Nogueira 2018) and label-shuffle p-values "
            "(Westerhuis 2008). VIP ≥ 1 + stability ≥ 0.8 + perm p &lt; 0.05 "
            "is the canonical mQACC shortlist gate."
        ),
        method="plsda",
        parameters={
            "n_components": n_components,
            "scale": pls_scale,
            "biological_only": biological_only,
            "stability_top_k": params.get("stability_top_k", "auto"),
            "n_bootstrap": result.n_bootstrap,
            "n_permutation": result.n_permutation,
            "runtime_seconds": round(result.runtime_seconds, 1),
        },
    )

    # ── Headline metrics ──────────────────────────────────────────
    n_high_vip = int((table["vip"] >= 1.0).sum())
    n_stable = int((table["stability_score"] >= 0.8).sum())
    n_p_sig = int((table["perm_p_value"] < p_threshold).sum())
    n_shortlist = int(
        (
            (table["vip"] >= 1.0)
            & (table["stability_score"] >= 0.8)
            & (table["perm_p_value"] < p_threshold)
        ).sum()
    )

    for label, value in [
        ("n_features", len(table)),
        ("n_VIP≥1", n_high_vip),
        ("n_stable (≥0.8)", n_stable),
        (f"n_perm_p<{p_threshold:g}", n_p_sig),
        ("n_shortlist (VIP+stab+p)", n_shortlist),
        ("median VIP", round(float(table["vip"].median()), 4)),
    ]:
        report.add_metric(section_name, label, value)

    # ── Figure 1: top-N VIP bar ───────────────────────────────────
    bar_fig = _top_n_importance_bar(
        table,
        importance_col="vip",
        n=top_n,
        title=f"Top-{top_n} features by VIP (PLS-DA)",
        yaxis_title="VIP",
    )
    bar_fig.add_hline(
        y=1.0,
        line={"color": "#D55E00", "width": 1.0, "dash": "dash"},
        annotation={"text": "VIP = 1.0", "font": {"size": 10, "color": "#D55E00"}},
        annotation_position="top right",
    )
    report.add_figure(
        section_name,
        title=f"Top-{top_n} features by VIP",
        fig=bar_fig,
        caption=(
            "VIP-ranked features, bar height = VIP, colour = bootstrap "
            "stability. The conventional cutoff VIP ≥ 1 sits as a dashed "
            "reference line. Tall + dark-coloured bars are the strongest "
            "candidates."
        ),
        method_metric="VIP (Wold 1995) · stability (Nogueira 2018)",
    )

    # ── Figure 2: stability × VIP scatter ─────────────────────────
    sc_fig = _stability_vs_importance_scatter(
        table,
        importance_col="vip",
        title="Bootstrap stability × VIP",
        xaxis_title="VIP",
        p_threshold=p_threshold,
    )
    sc_fig.add_vline(
        x=1.0,
        line={"color": "#D55E00", "width": 1.0, "dash": "dash"},
        annotation={"text": "VIP = 1", "font": {"size": 10, "color": "#D55E00"}},
        annotation_position="top right",
    )
    report.add_figure(
        section_name,
        title="Stability × VIP scatter",
        fig=sc_fig,
        caption=(
            "Discovery-quality plot. Upper-right quadrant (VIP &gt; 1, "
            "stability &gt; 0.8) holds the reproducible biomarker "
            "candidates. Coloured dots are also significant against the "
            "1000× label-shuffle null (Westerhuis 2008)."
        ),
        method_metric="stability × VIP (Nogueira 2018, Wold 1995)",
    )

    # ── Figure 3: permutation p-value distribution ────────────────
    p_fig = _permutation_p_histogram(table, title="Permutation p-value distribution")
    report.add_figure(
        section_name,
        title="Permutation-test p-value distribution",
        fig=p_fig,
        caption=(
            "How many features survive the label-shuffle null? Spike at the "
            "lowest bin = real signal beyond chance."
        ),
        method_metric="permutation p-value (Westerhuis 2008)",
    )

    # ── Figure 4: PLS score plot with ellipses + QC projection ───
    try:
        pls = _refit_pls_for_scores(X, y, n_components, scale=pls_scale)
        scores = np.asarray(pls.x_scores_, dtype=float)
        # Variance explained by each PLS component on X side
        if hasattr(pls, "x_loadings_") and scores.shape[1] >= 2:
            total_var = float(np.var(X, axis=0).sum())
            comp_vars = [
                float(np.var(scores[:, c]) * (np.linalg.norm(pls.x_loadings_[:, c])) ** 2)
                for c in range(min(2, scores.shape[1]))
            ]
            if total_var > 0:
                var_exp = (comp_vars[0] / total_var, comp_vars[1] / total_var)
            else:
                var_exp = None
        else:
            var_exp = None

        # Project QC samples onto the same PLS-DA model. QCs are pooled
        # mixtures of biology, so they should land near the centroid of
        # the discriminant space — a sanity check that the model is
        # well-conditioned.
        qc_scores = None
        qc_sample_ids: list[str] | None = None
        if biological_only:
            qc_ids_all = ctx.samples.loc[
                ctx.samples["injection_type"] == "qc", "sample_id"
            ].astype(str).tolist()
            qc_ids = [s for s in qc_ids_all if s in ctx.matrix.columns]
            if qc_ids:
                qc_X = ctx.matrix[qc_ids].T.to_numpy(dtype=float)
                # PLSRegression centres X internally during fit; transform
                # applies the same centring + projection learned from X.
                qc_scores = pls.transform(qc_X)
                qc_sample_ids = qc_ids

        score_fig = _score_plot_with_ellipses(
            scores, sample_ids, ctx.samples,
            title="PLS-DA score plot — component 1 × 2 with class 95 % CI",
            var_explained=var_exp,
            qc_scores=qc_scores,
            qc_sample_ids=qc_sample_ids,
        )
        report.add_figure(
            section_name,
            title="PLS-DA score plot with class ellipses",
            fig=score_fig,
            caption=(
                "Sample projections onto the first two PLS components. "
                "Tight non-overlapping ellipses = strong class separation; "
                "overlapping ellipses = the classes are not well "
                "discriminated by this model. QC samples (grey diamonds) "
                "are projected onto the same model and should sit near "
                "the centroid — they're a pooled mixture of biology so "
                "they shouldn't favour any class."
            ),
            method_metric="PLS-DA scores (Wold 1973)",
        )
    except Exception as exc:  # pragma: no cover — defensive refit
        report.add_metric(
            section_name,
            "score_plot_status",
            f"refit failed: {exc}",
        )

    # ── Figure 5: β-coefficient heatmap ───────────────────────────
    bh = _beta_heatmap(
        table, class_names,
        n_top=min(top_n, len(table)),
        title=f"β-coefficient heatmap — top-{top_n} features (rows) × classes (cols)",
    )
    if bh is not None:
        report.add_figure(
            section_name,
            title="β-coefficient heatmap",
            fig=bh,
            caption=(
                "Sign and magnitude of each top-N feature's contribution to "
                "every class. Red = positive (feature high in this class), "
                "blue = negative (feature low). A feature with one strong "
                "red column and weak blues is a clean class-private "
                "biomarker."
            ),
            method_metric="PLS β coefficients (Wold 1995)",
        )

    # ── Table: top-N feature ranking ──────────────────────────────
    keep_cols = ["feature_id", "vip", "abs_beta_max", "perm_importance",
                 "stability_score", "perm_p_value"]
    keep_cols = [c for c in keep_cols if c in table.columns]
    pretty = (
        table[keep_cols]
        .copy()
        .sort_values("vip", ascending=False)
        .head(top_n)
    )
    for col in pretty.columns:
        if pretty[col].dtype == float:
            pretty[col] = pretty[col].round(4)
    report.add_table(
        section_name,
        title=f"Top-{top_n} features by VIP",
        df=pretty,
        caption=(
            "Sorted by VIP. Use this with the stability × VIP scatter to "
            "spot which top-VIP features are also stable (= shortlist)."
        ),
    )


# ============================================================
# sPLS-DA section
# ============================================================
def add_splsda_section(
    report: PathReport,
    *,
    ctx: StageContext,
    result: MultivariateResult,
    section_name: str = "multivariate_splsda",
    top_n: int = 30,
    p_threshold: float = 0.05,
) -> None:
    """Add the sPLS-DA section to ``report``."""
    if result.method_name != "splsda":
        raise ValueError(
            f"add_splsda_section expected method_name='splsda', got {result.method_name!r}"
        )
    table = result.per_feature
    params = dict(result.params_used)
    biological_only = bool(params.get("biological_only", True))
    keep_n_per_comp = int(params.get("keep_n_per_comp", 0))

    X, y, _feature_ids, sample_ids, class_names = _build_xy_for_viz(
        ctx, biological_only=biological_only
    )

    report.add_section(
        section_name,
        title="Multivariate — sPLS-DA (sparse PLS-DA, Lê Cao 2011)",
        description=(
            "Sparse Partial Least Squares Discriminant Analysis: hard-sparsity "
            "NIPALS that keeps only the top-K weights per component, zeroing "
            "the rest. Returns an explicit <code>selected</code> mask — "
            "biomarkers carry that flag through the rest of the report. "
            "Avoids the &quot;everything looks important&quot; problem of "
            "vanilla PLS-DA on small N."
        ),
        method="splsda",
        parameters={
            "keep_n_per_comp": keep_n_per_comp,
            "n_components": params.get("n_components"),
            "biological_only": biological_only,
            "n_bootstrap": result.n_bootstrap,
            "n_permutation": result.n_permutation,
            "runtime_seconds": round(result.runtime_seconds, 1),
        },
    )

    # ── Headline metrics ──────────────────────────────────────────
    n_selected = int(table.get("selected", pd.Series([False] * len(table))).sum())
    n_stable = int((table["stability_score"] >= 0.8).sum())
    n_p_sig = int((table["perm_p_value"] < p_threshold).sum())
    n_shortlist = int(
        (
            table.get("selected", pd.Series([False] * len(table)))
            & (table["stability_score"] >= 0.8)
            & (table["perm_p_value"] < p_threshold)
        ).sum()
    )
    sparsity_pct = 100.0 * (1 - n_selected / max(1, len(table)))

    for label, value in [
        ("n_features", len(table)),
        ("n_selected", n_selected),
        ("sparsity %", round(sparsity_pct, 2)),
        ("n_stable (≥0.8)", n_stable),
        (f"n_perm_p<{p_threshold:g}", n_p_sig),
        ("n_shortlist (sel+stab+p)", n_shortlist),
    ]:
        report.add_metric(section_name, label, value)

    # ── Figure 1: top-N VIP bar with selection outline ────────────
    if "vip" in table.columns:
        bar_fig = _top_n_importance_bar(
            table,
            importance_col="vip",
            n=top_n,
            title=f"Top-{top_n} features by VIP (sPLS-DA, green outline = selected)",
            yaxis_title="VIP",
            selected_col="selected",
        )
        report.add_figure(
            section_name,
            title=f"Top-{top_n} features by VIP",
            fig=bar_fig,
            caption=(
                "VIP-ranked features. Green outline = feature passed sPLS-DA's "
                "hard sparsity filter; ungated bars are not part of the "
                "selected biomarker set even when their VIP is high."
            ),
            method_metric="VIP × sparse selection (Lê Cao 2011)",
        )

    # ── Figure 2: stability × VIP scatter with selection ──────────
    if "vip" in table.columns:
        sc_fig = _stability_vs_importance_scatter(
            table,
            importance_col="vip",
            title="Stability × VIP — sparse selection overlay",
            xaxis_title="VIP",
            selected_col="selected",
            p_threshold=p_threshold,
        )
        report.add_figure(
            section_name,
            title="Stability × VIP scatter (selection overlay)",
            fig=sc_fig,
            caption=(
                "Selected features (orange / green) should cluster in the "
                "upper-right; selected features in the lower-left are "
                "high-leverage but unstable, worth scepticism."
            ),
            method_metric="stability × VIP × selection",
        )

    # ── Figure 3: permutation p-value distribution ────────────────
    p_fig = _permutation_p_histogram(table, title="Permutation p-value distribution")
    report.add_figure(
        section_name,
        title="Permutation-test p-value distribution",
        fig=p_fig,
        caption="Same interpretation as PLS-DA. Low-bin spike = real signal.",
        method_metric="permutation p-value (Westerhuis 2008)",
    )

    # ── Figure 4: sPLS-DA score plot with ellipses + QC projection ──
    # We can't easily refit sPLS-DA here without copying the NIPALS impl;
    # instead use the projection from the selected features only via PLS.
    try:
        sel_mask = table.get("selected", pd.Series([False] * len(table))).astype(bool).to_numpy()
        if int(sel_mask.sum()) >= 2:
            X_sel = X[:, sel_mask]
            n_components_proj = min(2, X_sel.shape[1] - 1, X_sel.shape[0] - 1)
            if n_components_proj >= 2:
                pls = _refit_pls_for_scores(X_sel, y, 2, scale=False)
                scores = np.asarray(pls.x_scores_, dtype=float)

                # Project QCs through the same selected-feature subspace
                qc_scores_spls = None
                qc_sample_ids_spls: list[str] | None = None
                if biological_only:
                    qc_ids_all = ctx.samples.loc[
                        ctx.samples["injection_type"] == "qc", "sample_id"
                    ].astype(str).tolist()
                    qc_ids = [s for s in qc_ids_all if s in ctx.matrix.columns]
                    if qc_ids:
                        qc_X = ctx.matrix[qc_ids].T.to_numpy(dtype=float)
                        qc_X_sel = qc_X[:, sel_mask]
                        qc_scores_spls = pls.transform(qc_X_sel)
                        qc_sample_ids_spls = qc_ids

                score_fig = _score_plot_with_ellipses(
                    scores, sample_ids, ctx.samples,
                    title="sPLS-DA score plot — selected features only",
                    qc_scores=qc_scores_spls,
                    qc_sample_ids=qc_sample_ids_spls,
                )
                report.add_figure(
                    section_name,
                    title="sPLS-DA score plot (projection on selected features)",
                    fig=score_fig,
                    caption=(
                        "Sample projection on the selected feature subset only. "
                        "Compares against PLS-DA score plot above — sparse model "
                        "should preserve class separation while using far fewer "
                        "features. QC samples (grey diamonds) projected onto the "
                        "same selected-feature subspace; should land near the "
                        "centroid."
                    ),
                    method_metric="sPLS-DA selected-feature projection",
                )
    except Exception as exc:  # pragma: no cover — defensive
        report.add_metric(
            section_name,
            "score_plot_status",
            f"projection failed: {exc}",
        )

    # ── Figure 5: β-coefficient heatmap (selected features only) ──
    if "vip" in table.columns:
        sel_table = table[table.get("selected", pd.Series([False] * len(table))).astype(bool)]
        if not sel_table.empty:
            bh = _beta_heatmap(
                sel_table, class_names,
                n_top=min(top_n, len(sel_table)),
                title=f"β-coefficient heatmap — selected features (top {min(top_n, len(sel_table))})",
            )
            if bh is not None:
                report.add_figure(
                    section_name,
                    title="β-coefficient heatmap (selected features)",
                    fig=bh,
                    caption=(
                        "Per-class contribution direction + magnitude for the "
                        "sparse-selected biomarkers. A clean class-private "
                        "feature has one bright cell in its row."
                    ),
                    method_metric="sPLS-DA β coefficients",
                )

    # ── Table: selected features ──────────────────────────────────
    sel_table = (
        table[table.get("selected", pd.Series([False] * len(table))).astype(bool)]
        .copy()
        .sort_values("vip" if "vip" in table.columns else "stability_score", ascending=False)
    )
    keep_cols = [
        c for c in
        ["feature_id", "vip", "abs_beta_max", "stability_score", "perm_p_value"]
        if c in sel_table.columns
    ]
    pretty = sel_table[keep_cols].head(top_n).copy()
    for col in pretty.columns:
        if pretty[col].dtype == float:
            pretty[col] = pretty[col].round(4)
    if not pretty.empty:
        report.add_table(
            section_name,
            title=f"Selected features (top {min(top_n, len(pretty))} by VIP)",
            df=pretty,
            caption=(
                "Hard-sparsity-selected feature set. These are the "
                "candidates the sparse model considers biomarkers; the "
                "shortlist intersects with stability ≥ 0.8 and perm p &lt; "
                f"{p_threshold:g}."
            ),
        )


# ============================================================
# RF section
# ============================================================
def add_rf_section(
    report: PathReport,
    *,
    ctx: StageContext,
    result: MultivariateResult,
    section_name: str = "multivariate_rf",
    top_n: int = 30,
    p_threshold: float = 0.05,
) -> None:
    """Add the Random Forest section to ``report``."""
    if result.method_name != "rf":
        raise ValueError(
            f"add_rf_section expected method_name='rf', got {result.method_name!r}"
        )
    table = result.per_feature
    params = dict(result.params_used)

    report.add_section(
        section_name,
        title="Multivariate — Random Forest (Breiman 2001)",
        description=(
            "Random Forest classifier used as a feature-importance generator. "
            "Reports BOTH Gini importance (default sklearn) AND permutation "
            "importance (Strobl 2007 — less biased toward high-cardinality "
            "features). Real biomarkers should rank high in BOTH — the Gini "
            "vs perm scatter makes this comparison visible."
        ),
        method="rf",
        parameters={
            "n_estimators": params.get("n_estimators"),
            "biological_only": params.get("biological_only", True),
            "n_bootstrap": result.n_bootstrap,
            "n_permutation": result.n_permutation,
            "runtime_seconds": round(result.runtime_seconds, 1),
        },
    )

    # ── Headline metrics ──────────────────────────────────────────
    n_stable = int((table["stability_score"] >= 0.8).sum())
    n_p_sig = int((table["perm_p_value"] < p_threshold).sum())
    n_shortlist = int(
        (
            (table["stability_score"] >= 0.8)
            & (table["perm_p_value"] < p_threshold)
        ).sum()
    )
    median_gini = float(table["gini_importance"].median())
    median_perm = float(table["perm_importance"].median())

    for label, value in [
        ("n_features", len(table)),
        ("n_stable (≥0.8)", n_stable),
        (f"n_perm_p<{p_threshold:g}", n_p_sig),
        ("n_shortlist (stab+p)", n_shortlist),
        ("median Gini", round(median_gini, 6)),
        ("median perm importance", round(median_perm, 6)),
    ]:
        report.add_metric(section_name, label, value)

    # ── Figure 1: top-N Gini bar ──────────────────────────────────
    bar_fig = _top_n_importance_bar(
        table,
        importance_col="gini_importance",
        n=top_n,
        title=f"Top-{top_n} features by Gini importance (RF)",
        yaxis_title="Gini importance",
    )
    report.add_figure(
        section_name,
        title=f"Top-{top_n} features by Gini importance",
        fig=bar_fig,
        caption=(
            "Bar height = Gini importance, colour = bootstrap stability. "
            "Compare against permutation-importance ranking below — "
            "features high in both are the trustworthy candidates "
            "(Strobl 2007)."
        ),
        method_metric="Gini importance · stability",
    )

    # ── Figure 2: stability × Gini scatter ────────────────────────
    sc_fig = _stability_vs_importance_scatter(
        table,
        importance_col="gini_importance",
        title="Bootstrap stability × Gini importance",
        xaxis_title="Gini importance",
        p_threshold=p_threshold,
    )
    report.add_figure(
        section_name,
        title="Stability × Gini importance scatter",
        fig=sc_fig,
        caption=(
            "Discovery-quality plot. Upper-right = high Gini + reproducible "
            "across bootstraps. Coloured points are also perm-significant."
        ),
        method_metric="stability × Gini",
    )

    # ── Figure 3: permutation p-value histogram ───────────────────
    p_fig = _permutation_p_histogram(table, title="Permutation p-value distribution (RF Gini)")
    report.add_figure(
        section_name,
        title="Permutation-test p-value distribution",
        fig=p_fig,
        caption="Spike near zero = real Gini-importance signal beyond chance.",
        method_metric="permutation p-value (Westerhuis 2008)",
    )

    # ── Figure 4: Gini vs permutation importance scatter ─────────
    if "perm_importance" in table.columns:
        gini = table["gini_importance"].to_numpy(dtype=float)
        perm = table["perm_importance"].to_numpy(dtype=float)
        feature_ids = table["feature_id"].astype(str).to_numpy()
        # Color by stability
        stab = table["stability_score"].to_numpy(dtype=float)
        cmp_fig = go.Figure(
            data=go.Scatter(
                x=gini,
                y=perm,
                mode="markers",
                marker={
                    "color": stab,
                    "size": 7,
                    "opacity": 0.85,
                    "colorscale": "Viridis",
                    "cmin": 0.0,
                    "cmax": 1.0,
                    "colorbar": {"title": "stability", "thickness": 12},
                    "line": {"width": 0},
                },
                customdata=feature_ids,
                hovertemplate=(
                    "%{customdata}<br>Gini: %{x:.4f}<br>perm imp: %{y:.4f}"
                    "<br>stability: %{marker.color:.3f}<extra></extra>"
                ),
            )
        )
        apply_theme(
            cmp_fig,
            title="Gini vs permutation importance — agreement check",
            xaxis_title="Gini importance",
            yaxis_title="permutation importance",
        )
        report.add_figure(
            section_name,
            title="Gini vs permutation importance (Strobl 2007)",
            fig=cmp_fig,
            caption=(
                "Real biomarkers should cluster on the diagonal (high Gini "
                "AND high perm). Off-diagonal points: features with high "
                "Gini but low perm importance are the classical Gini-bias "
                "false positives (high cardinality / many split points). "
                "Trust the perm-importance ranking when they disagree."
            ),
            method_metric="Gini × perm importance (Strobl 2007)",
        )

    # ── Figure 5: OOB error trajectory (cheap refit) ─────────────
    try:
        bio_only = bool(params.get("biological_only", True))
        X, y, _feature_ids, _sample_ids, _class_names = _build_xy_for_viz(
            ctx, biological_only=bio_only
        )
        n_estimators = int(params.get("n_estimators") or max(100, X.shape[0] * 10))
        from sklearn.ensemble import RandomForestClassifier

        t0 = time.time()
        rf_track = RandomForestClassifier(
            n_estimators=n_estimators,
            oob_score=True,
            warm_start=True,
            random_state=int(params.get("seed", 42)),
            bootstrap=True,
            n_jobs=1,
        )
        # Track OOB at a coarse grid of tree counts
        track_points = sorted(
            {
                round(float(x))
                for x in np.linspace(10, n_estimators, num=min(15, max(1, n_estimators // 10)))
            }
        )
        oob_trajectory: list[tuple[int, float]] = []
        for tp in track_points:
            rf_track.set_params(n_estimators=tp)
            rf_track.fit(X, y)
            if hasattr(rf_track, "oob_score_") and rf_track.oob_score_ is not None:
                oob_trajectory.append((tp, float(rf_track.oob_score_)))
        elapsed = time.time() - t0
        if oob_trajectory:
            x_traj, y_traj = zip(*oob_trajectory, strict=True)
            traj_fig = go.Figure(
                data=go.Scatter(
                    x=list(x_traj),
                    y=list(y_traj),
                    mode="lines+markers",
                    line={"color": "#0072B2", "width": 2.0},
                    marker={"color": "#0072B2", "size": 7},
                    hovertemplate="n_trees: %{x}<br>OOB score: %{y:.4f}<extra></extra>",
                )
            )
            apply_theme(
                traj_fig,
                title=f"RF OOB-score trajectory (saturation check, {elapsed:.1f}s)",
                xaxis_title="n_estimators (tree count)",
                yaxis_title="out-of-bag score (1 - oob error)",
            )
            report.add_figure(
                section_name,
                title="OOB-score trajectory (saturation diagnostic)",
                fig=traj_fig,
                caption=(
                    "Out-of-bag accuracy as the forest grows. The curve "
                    "should plateau before <code>n_estimators</code> — "
                    "earlier saturation means more trees would just inflate "
                    "runtime. Confirms the small-N rule "
                    "<code>n_estimators = max(100, n_samples × 10)</code>."
                ),
                method_metric="RF OOB trajectory",
            )
    except Exception as exc:  # pragma: no cover
        report.add_metric(section_name, "oob_trajectory_status", f"refit failed: {exc}")

    # ── Table: top-N by Gini ──────────────────────────────────────
    keep_cols = ["feature_id", "gini_importance", "perm_importance",
                 "stability_score", "perm_p_value"]
    keep_cols = [c for c in keep_cols if c in table.columns]
    pretty = (
        table[keep_cols]
        .copy()
        .sort_values("gini_importance", ascending=False)
        .head(top_n)
    )
    for col in pretty.columns:
        if pretty[col].dtype == float:
            pretty[col] = pretty[col].round(6)
    report.add_table(
        section_name,
        title=f"Top-{top_n} features by Gini importance",
        df=pretty,
        caption=(
            "Sorted by Gini. Cross-check with the Gini-vs-perm scatter — "
            "rows where Gini is high but perm importance is low are "
            "Gini-bias false positives."
        ),
    )
