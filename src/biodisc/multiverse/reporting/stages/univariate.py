"""Univariate section — ANOVA + BH-FDR + per-pair effect-size figures.

Wraps ``biodisc.multiverse.stats_univariate.compute_univariate`` into a
publication-grade section. The univariate stack is run on the
**log-transformed** matrix (pre-Pareto): log-scale lets
``log2 FC = mean_A − mean_B`` work directly, and ANOVA assumptions
(homoscedastic residuals across groups) hold better there. Pareto-scaled
data centres each feature at zero and z-equivalents the contrasts —
losing the biological "fold change" interpretation that reviewers
expect.

Figure inventory
----------------
1. **BH-FDR p-value histogram** — distribution of raw p-values overlaid
   on the uniform null. Mass piling near zero = real signal.
2. **η² (effect-size) distribution** — Cohen's small / medium / large
   reference lines so reviewers can read off how many features carry
   meaningful magnitude vs marginal significance.
3. **Per-pair volcano grid** — Plotly subplots, one panel per group
   pair. x = pair-specific log2 FC, y = −log10(ANOVA q). Threshold lines
   at |FC| = 1 and q = 0.05. Significant + large-effect points are
   coloured by their pair-specific class colour.
4. **Top-30 features bar** — top features by −log10(q), coloured by η².
   Reviewer's quick "which features dominate" view.
5. **Per-pair Tukey heatmap** — k×k matrix, cell = fraction of features
   with Tukey post-hoc p < α. Reveals which contrasts drive most signal.

Headline metrics
----------------
* ``n_features``, ``n_groups``, group names + per-group sizes.
* ``n_significant_at_q05`` and ``n_significant_with_large_eta`` (η² ≥ 0.14).
* ``median_eta_squared``, ``median_p_value``.

References
----------
* Benjamini & Hochberg 1995, *J R Stat Soc Ser B* 57:289 — FDR control.
* Cohen 1988, *Statistical Power Analysis* — η² thresholds 0.01 / 0.06 / 0.14.
* Tukey 1953, *Multiple comparisons* — post-hoc.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.reporting.core import PathReport
from biodisc.multiverse.reporting.theme import apply_theme, color_for
from biodisc.multiverse.stats_univariate import UnivariateResult, compute_univariate

# Cohen's η² thresholds (Cohen 1988): small / medium / large
_ETA_SMALL = 0.01
_ETA_MEDIUM = 0.06
_ETA_LARGE = 0.14


def _per_pair_volcano_grid(
    table: pd.DataFrame,
    group_names: list[str],
    *,
    q_threshold: float,
    log2_fc_threshold: float,
    title: str,
) -> go.Figure:
    """Faceted volcano: one subplot per group pair.

    All subplots share the same y-axis (-log10(q_anova)) so vertical
    position of a feature is consistent across panels. The x-axis varies
    per pair (its log2 FC).
    """
    pairs = list(combinations(group_names, 2))
    n_pairs = len(pairs)
    if n_pairs == 0:
        fig = go.Figure()
        apply_theme(fig, title=title + " (no pairs)", xaxis_title="log2 FC", yaxis_title="-log10(q)")
        return fig

    # Layout: aim for ≤ 3 columns for readability
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    subplot_titles = [f"{a} vs {b}" for a, b in pairs]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
        shared_yaxes=True,
    )

    q_values = table["q_value"].to_numpy(dtype=float)
    q_safe = np.where(q_values > 0, q_values, np.nextafter(0, 1))
    neg_log10_q = -np.log10(q_safe)
    significant = q_values < q_threshold

    for idx, (g_a, g_b) in enumerate(pairs):
        col_name = f"pair_{g_a}_{g_b}_log2_fc"
        if col_name not in table.columns:
            continue
        log2_fc = table[col_name].to_numpy(dtype=float)
        feature_ids = table["feature_id"].to_numpy()
        large_fc = np.abs(log2_fc) >= log2_fc_threshold
        # 4 categories: insignificant + small FC, significant only, large FC only, both
        cat_both = significant & large_fc
        cat_sig_only = significant & ~large_fc
        cat_fc_only = ~significant & large_fc
        cat_neither = ~significant & ~large_fc

        row, col = idx // n_cols + 1, idx % n_cols + 1
        # Plot 4 categories with different colours
        colour_a = color_for(g_a)
        colour_b = color_for(g_b)
        for mask, name, marker_color, marker_size in (
            (cat_neither, "ns", "#cccccc", 4),
            (cat_sig_only, "q<0.05", "#888888", 5),
            (cat_fc_only, f"|FC|≥{log2_fc_threshold}", "#999999", 5),
            (cat_both, "both", colour_a if abs(log2_fc[cat_both].mean() if cat_both.any() else 0) > 0 else colour_b, 7),
        ):
            if not mask.any():
                continue
            fig.add_trace(
                go.Scattergl(
                    x=log2_fc[mask],
                    y=neg_log10_q[mask],
                    mode="markers",
                    marker={
                        "color": marker_color,
                        "size": marker_size,
                        "opacity": 0.7,
                        "line": {"width": 0},
                    },
                    name=name,
                    showlegend=(idx == 0),
                    legendgroup=name,
                    hovertemplate=(
                        "%{customdata}<br>log2 FC: %{x:.3f}<br>-log10(q): %{y:.3f}"
                        "<extra></extra>"
                    ),
                    customdata=feature_ids[mask],
                ),
                row=row,
                col=col,
            )
        # Reference lines per subplot
        fig.add_hline(
            y=-np.log10(q_threshold),
            line={"color": "#D55E00", "width": 1.0, "dash": "dash"},
            row=row,
            col=col,
        )
        fig.add_vline(
            x=log2_fc_threshold,
            line={"color": "#444", "width": 1.0, "dash": "dot"},
            row=row,
            col=col,
        )
        fig.add_vline(
            x=-log2_fc_threshold,
            line={"color": "#444", "width": 1.0, "dash": "dot"},
            row=row,
            col=col,
        )
        # Per-subplot axis titles
        fig.update_xaxes(title_text=f"log2 FC ({g_a} − {g_b})", row=row, col=col)
        fig.update_yaxes(title_text="−log10(ANOVA q)", row=row, col=col)

    apply_theme(fig, title=title)
    fig.update_layout(
        height=300 * n_rows + 80,
        legend={"orientation": "h", "y": -0.10, "x": 0.0, "xanchor": "left"},
    )
    return fig


def _tukey_heatmap(
    table: pd.DataFrame,
    group_names: list[str],
    *,
    alpha: float,
    title: str,
) -> go.Figure:
    """Symmetric k×k heatmap of % features with Tukey p < alpha per pair.

    Computed only over the rows where the Tukey p column is non-NaN —
    in ``auto`` mode, that's the ANOVA-significant subset, so the
    heatmap shows "of the features ANOVA flagged, what fraction does
    each contrast drive?"
    """
    k = len(group_names)
    grid = np.full((k, k), np.nan, dtype=float)
    n_in_subset = np.zeros((k, k), dtype=int)
    for i, g_a in enumerate(group_names):
        for j, g_b in enumerate(group_names[i + 1:], start=i + 1):
            col = f"pair_{g_a}_{g_b}_tukey_p"
            if col not in table.columns:
                continue
            vals = table[col].to_numpy(dtype=float)
            valid = np.isfinite(vals)
            if int(valid.sum()) == 0:
                continue
            frac = float((vals[valid] < alpha).mean())
            grid[i, j] = frac
            grid[j, i] = frac
            n_in_subset[i, j] = int(valid.sum())
            n_in_subset[j, i] = int(valid.sum())

    text = [
        [
            f"{grid[i, j] * 100:.0f}%<br>(n={n_in_subset[i, j]})"
            if np.isfinite(grid[i, j]) else ""
            for j in range(k)
        ]
        for i in range(k)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            x=group_names,
            y=group_names,
            colorscale=[[0.0, "#ffffff"], [0.5, "#fdb863"], [1.0, "#b35806"]],
            zmin=0,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#222"},
            hovertemplate=(
                "%{y} vs %{x}<br>"
                "fraction Tukey p&lt;α: %{z:.3f}<extra></extra>"
            ),
            colorbar={"title": "frac p&lt;α", "thickness": 12},
        )
    )
    apply_theme(fig, title=title, xaxis_title="cell line", yaxis_title="cell line")
    fig.update_yaxes(autorange="reversed")
    return fig


def _top_n_anova_bar(
    table: pd.DataFrame,
    *,
    n: int,
    title: str,
) -> go.Figure:
    """Top-N features by -log10(q), coloured by η²."""
    df = table.copy()
    df = df.dropna(subset=["q_value"])
    df["neg_log10_q"] = -np.log10(np.where(df["q_value"] > 0, df["q_value"], np.nextafter(0, 1)))
    df = df.sort_values("neg_log10_q", ascending=False).head(n)

    fig = go.Figure(
        data=go.Bar(
            x=df["feature_id"].astype(str),
            y=df["neg_log10_q"],
            marker={
                "color": df["eta_squared"],
                "colorscale": "Viridis",
                "cmin": 0,
                "cmax": 1,
                "colorbar": {"title": "η²", "thickness": 12},
            },
            hovertemplate=(
                "%{x}<br>-log10(q): %{y:.3f}<br>η²: %{marker.color:.3f}"
                "<extra></extra>"
            ),
        )
    )
    apply_theme(
        fig,
        title=title,
        xaxis_title="feature_id (ranked by significance)",
        yaxis_title="-log10(ANOVA q)",
    )
    fig.update_layout(xaxis={"tickangle": -55, "tickfont": {"size": 9}})
    return fig


def add_univariate_section(
    report: PathReport,
    *,
    matrix: pd.DataFrame,
    samples: pd.DataFrame,
    section_name: str = "univariate",
    min_per_group: int = 3,
    tukey_post_hoc: str = "auto",
    tukey_q_threshold: float = 0.05,
    q_significance: float = 0.05,
    log2_fc_threshold: float = 1.0,
    top_n_features: int = 30,
) -> UnivariateResult:
    """Add the univariate section to ``report``. Returns ``UnivariateResult``.

    Pass the **log-transformed** matrix here, NOT the Pareto-scaled one.
    log2 FC = mean_A − mean_B is the natural interpretation on log
    scale; on Pareto-scaled data the contrasts are in z-units, not
    biological fold changes.
    """
    ctx = StageContext(
        matrix=matrix,
        samples=samples,
        features=pd.DataFrame({"feature_id": matrix.index}),
        run_order=None,
    )
    # The univariate section's contract is to receive the log-transformed
    # matrix (so log2 FC = mean_A − mean_B is biologically interpretable).
    # Pass is_log_scale=True so compute_univariate doesn't fall back to
    # its `log2(mean_A / mean_B)` branch when the biological subset
    # happens to be all-positive.
    res = compute_univariate(
        ctx,
        min_per_group=min_per_group,
        tukey_post_hoc=tukey_post_hoc,
        tukey_q_threshold=tukey_q_threshold,
        is_log_scale=True,
    )

    # Per-group sample counts (for the parameters section)
    bio_samples = samples[samples["injection_type"] == "biological"]
    group_counts = (
        bio_samples["cell_line"].astype(str).value_counts().to_dict()
        if len(bio_samples) > 0
        else {}
    )

    report.add_section(
        section_name,
        title=(
            "Univariate statistics — one-way ANOVA + BH-FDR + η² + per-pair "
            "log2 FC + Tukey HSD"
        ),
        description=(
            "Per-feature one-way ANOVA across the cell-line groups followed by "
            "Benjamini-Hochberg FDR correction (q-value), η² effect size "
            "(Cohen 1988), per-pair log2 fold-change + Cohen's d, and Tukey HSD "
            "post-hoc for ANOVA-significant features. Every feature's "
            "significance gets a magnitude companion (η² for the omnibus, "
            "log2 FC + Cohen's d for each contrast) so the shortlist can "
            "filter on biological meaning, not just p-value."
        ),
        method="anova_bh_eta_tukey",
        parameters={
            "min_per_group": min_per_group,
            "fdr_method": res.fdr_method,
            "tukey_post_hoc": tukey_post_hoc,
            "tukey_q_threshold": tukey_q_threshold,
            "q_significance": q_significance,
            "log2_fc_threshold": log2_fc_threshold,
            "n_groups": res.n_groups,
            "group_names": ", ".join(res.group_names),
            "group_counts": ", ".join(f"{g}={n}" for g, n in sorted(group_counts.items())),
        },
    )

    # ── Headline metrics ──────────────────────────────────────────
    table = res.table
    n_sig_q05 = int((table["q_value"] < q_significance).sum())
    n_large_eta = int((table["eta_squared"] >= _ETA_LARGE).sum())
    n_sig_and_large = int(
        ((table["q_value"] < q_significance) & (table["eta_squared"] >= _ETA_LARGE)).sum()
    )
    median_eta = float(table["eta_squared"].median(skipna=True))
    median_p = float(table["p_value"].median(skipna=True))

    report.add_metric(section_name, "n_features", res.n_features, unit="features")
    report.add_metric(section_name, "n_groups", res.n_groups)
    report.add_metric(section_name, "n_significant (q<0.05)", n_sig_q05, unit="features")
    report.add_metric(section_name, "n_large_eta_squared (η²≥0.14)", n_large_eta, unit="features")
    report.add_metric(
        section_name,
        "n_significant + large η²",
        n_sig_and_large,
        unit="features",
        description="features that are both ANOVA-significant AND large effect",
    )
    report.add_metric(
        section_name,
        "median η²",
        round(median_eta, 4) if np.isfinite(median_eta) else "n/a",
    )
    report.add_metric(
        section_name,
        "median p_value",
        f"{median_p:.3g}" if np.isfinite(median_p) else "n/a",
    )

    # ── Figure 1: BH-FDR p-value histogram ────────────────────────
    p_arr = table["p_value"].dropna().to_numpy()
    p_fig = go.Figure(
        data=[
            go.Histogram(
                x=p_arr,
                xbins={"start": 0.0, "end": 1.0, "size": 0.025},
                marker={"color": "#0072B2"},
                opacity=0.8,
                hovertemplate="p in [%{x:.3f}, +bin)<br>n features: %{y}<extra></extra>",
            )
        ]
    )
    # Uniform-null reference: expected count per bin = n / n_bins
    n_bins = 40
    expected_per_bin = len(p_arr) / n_bins
    p_fig.add_hline(
        y=expected_per_bin,
        line={"color": "#444", "width": 1.0, "dash": "dot"},
        annotation={"text": f"uniform null (n/{n_bins})", "font": {"size": 10, "color": "#444"}},
        annotation_position="top right",
    )
    apply_theme(
        p_fig,
        title="BH-FDR p-value distribution",
        xaxis_title="raw p-value",
        yaxis_title="feature count",
    )
    report.add_figure(
        section_name,
        title="P-value histogram (uniform null expected under no signal)",
        fig=p_fig,
        caption=(
            "Distribution of raw ANOVA p-values across all features. Under the "
            "global null (no real biology) p-values are uniform — every bar "
            "should sit near the dotted reference line. Excess mass near zero "
            "= real signal. The taller the spike at the lowest bin, the more "
            "discoveries BH-FDR will accept."
        ),
        method_metric="p-value distribution (Benjamini & Hochberg 1995)",
    )

    # ── Figure 2: η² distribution ─────────────────────────────────
    eta_arr = table["eta_squared"].dropna().to_numpy()
    eta_fig = go.Figure(
        data=[
            go.Histogram(
                x=eta_arr,
                xbins={"start": 0.0, "end": 1.0, "size": 0.02},
                marker={"color": "#009E73"},
                opacity=0.8,
                hovertemplate="η²: %{x:.3f}<br>n features: %{y}<extra></extra>",
            )
        ]
    )
    for x_val, label in (
        (_ETA_SMALL, "small (0.01)"),
        (_ETA_MEDIUM, "medium (0.06)"),
        (_ETA_LARGE, "large (0.14)"),
    ):
        eta_fig.add_vline(
            x=x_val,
            line={"color": "#D55E00", "width": 1.0, "dash": "dash"},
            annotation={"text": label, "font": {"size": 9, "color": "#D55E00"}},
            annotation_position="top right",
        )
    apply_theme(
        eta_fig,
        title="η² (effect-size) distribution",
        xaxis_title="η² (proportion of variance explained by group)",
        yaxis_title="feature count",
    )
    report.add_figure(
        section_name,
        title="η² distribution with Cohen's small/medium/large reference lines",
        fig=eta_fig,
        caption=(
            "Per-feature effect size from the one-way ANOVA. Cohen's "
            "convention: η² ≥ 0.01 = small, ≥ 0.06 = medium, ≥ 0.14 = large. "
            "Features in the right tail carry meaningful biological "
            "magnitude; features with significant p but low η² are real but "
            "marginal — filter them out at the shortlist stage."
        ),
        method_metric="η² (Cohen 1988)",
    )

    # ── Figure 3: per-pair volcano grid ───────────────────────────
    if res.n_groups >= 2:
        volcano_fig = _per_pair_volcano_grid(
            table,
            res.group_names,
            q_threshold=q_significance,
            log2_fc_threshold=log2_fc_threshold,
            title=(
                f"Volcano grid — {res.n_groups} groups × {len(list(combinations(res.group_names, 2)))} "
                f"pairs (y = −log10(ANOVA q))"
            ),
        )
        report.add_figure(
            section_name,
            title="Per-pair volcano plots",
            fig=volcano_fig,
            caption=(
                "One subplot per cell-line pair. x = pair-specific log2 FC "
                "(mean_A − mean_B on log scale), y = −log10(ANOVA q-value) "
                "shared across panels (the omnibus test isn't pair-specific). "
                "Threshold lines: q = 0.05 (orange dashed) and "
                f"|log2 FC| = {log2_fc_threshold:g} (grey dotted). Points in "
                "the upper-outer corners are the shortlist candidates."
            ),
            method_metric="ANOVA q + per-pair log2 FC",
        )

    # ── Figure 4: top-N bar ───────────────────────────────────────
    bar_fig = _top_n_anova_bar(
        table,
        n=top_n_features,
        title=f"Top-{top_n_features} features by ANOVA significance",
    )
    report.add_figure(
        section_name,
        title=f"Top-{top_n_features} features by −log10(q), coloured by η²",
        fig=bar_fig,
        caption=(
            "The single most useful overview when triaging discoveries. "
            "Bars descend by ANOVA −log10(q); colour saturates with η² so a "
            "tall, dark bar is both significant AND large-effect. Shortlist "
            "candidates dominate the left side of this plot."
        ),
        method_metric="top-N significance × effect-size view",
    )

    # ── Figure 5: Tukey heatmap ───────────────────────────────────
    if tukey_post_hoc != "none" and res.n_groups >= 2:
        heat_fig = _tukey_heatmap(
            table,
            res.group_names,
            alpha=q_significance,
            title=(
                f"Tukey HSD post-hoc — fraction of features with p&lt;{q_significance:g}"
                + (" (ANOVA-significant subset)" if tukey_post_hoc == "auto" else " (all features)")
            ),
        )
        report.add_figure(
            section_name,
            title="Tukey post-hoc heatmap",
            fig=heat_fig,
            caption=(
                "Per-pair fraction of features whose Tukey HSD post-hoc test "
                "is significant. In <code>auto</code> mode (default) Tukey is "
                "computed only for ANOVA-significant features, so this matrix "
                "asks: <em>among the features ANOVA flagged, which contrasts "
                "drive that signal?</em> Hotter cells = more contrasts within "
                "that pair."
            ),
            method_metric="Tukey HSD (Tukey 1953)",
        )

    # ── Table: top-N features (significant + large effect, joined) ──
    sig_table = (
        table.assign(
            neg_log10_q=lambda d: -np.log10(
                np.where(d["q_value"] > 0, d["q_value"], np.nextafter(0, 1))
            )
        )
        .sort_values("neg_log10_q", ascending=False)
        .head(top_n_features)
    )
    keep_cols = ["feature_id", "f_statistic", "p_value", "q_value", "eta_squared"]
    keep_cols = [c for c in keep_cols if c in sig_table.columns]
    pretty = sig_table[keep_cols].copy()
    for col in ("f_statistic", "eta_squared"):
        if col in pretty.columns:
            pretty[col] = pretty[col].round(4)
    for col in ("p_value", "q_value"):
        if col in pretty.columns:
            pretty[col] = pretty[col].apply(
                lambda v: f"{v:.3g}" if pd.notna(v) else "n/a"
            )
    report.add_table(
        section_name,
        title=f"Top-{top_n_features} features by ANOVA q",
        df=pretty,
        caption=(
            "Sorted by ANOVA significance. Use this with the volcano grid to "
            "trace which contrast drives each feature's signal — feature ids "
            "in this table appear as labels in the top-N bar above."
        ),
    )

    return res
