"""Path-level summary — feature attrition + cross-method overlap + runtime.

The closing section of every path report. It answers the four questions
a reviewer asks AFTER looking at the per-stage panels:

1. **How many features survived each gate?** — waterfall bar showing
   the cumulative attrition through the pipeline. The classic
   biomarker-discovery shape: a steep drop at the early presence /
   blank filters, then more gradual at the variance-aware stages, then
   a final cliff at the statistical / shortlist gates.

2. **Where do the methods agree?** — pairwise Jaccard similarity of
   per-method shortlists (k × k heatmap) plus a "n-methods-agreeing"
   bar that shows how many features are flagged by 1 / 2 / 3 / 4
   methods. Consensus across methods is the single strongest
   defensible-shortlist signal in a multiverse design (Steegen 2016).

3. **Where did time go?** — per-stage runtime breakdown. Useful when
   tuning permutation counts or deciding which method to drop in a
   speed-constrained run.

4. **What was the path?** — config table listing the exact method
   choice at every stage (imputation, drift, normalisation,
   transformation, scaling, multivariate). The same information is in
   the path-chain banner at the top of the report, but the full table
   makes parameter-level details inspectable.

Shortlist gates per method
--------------------------
* **univariate**: ``q < 0.05`` AND ``η² ≥ 0.14``
* **PLS-DA**: ``vip ≥ 1.0`` AND ``stability ≥ 0.8`` AND ``perm_p < 0.05``
* **sPLS-DA**: ``selected = True`` AND ``stability ≥ 0.8`` AND
  ``perm_p < 0.05``
* **RF**: ``stability ≥ 0.8`` AND ``perm_p < 0.05``

These are the standard defensive thresholds; the Phase 9 shortlist
stage replaces these with user-tunable rules. Here we use the textbook
defaults so the section is interpretable without configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from biodisc.multiverse.multivariate_utils import MultivariateResult
from biodisc.multiverse.reporting.core import PathReport
from biodisc.multiverse.reporting.theme import apply_theme
from biodisc.multiverse.stats_univariate import UnivariateResult


# ============================================================
# Path-summary state container
# ============================================================
@dataclass
class PathSummary:
    """Mutable accumulator for path-level diagnostics.

    Pipelines fill this as they run — append to ``attrition_history`` /
    ``runtime_history`` after each stage, and set ``method_choices``
    when a method is locked in.

    Attributes
    ----------
    attrition_history
        Ordered list of ``(stage_name, n_features_after)`` tuples. The
        waterfall preserves insertion order.
    runtime_history
        Ordered list of ``(stage_name, seconds)`` tuples.
    method_choices
        Mapping from category (``"imputation"``, ``"drift"``, ...) to the
        chosen method id (``"halfmin"``, ``"qc_rsc"``, ...).
    extra_params
        Per-category dict of method parameters; surfaces in the config
        table.
    """

    attrition_history: list[tuple[str, int]] = field(default_factory=list)
    runtime_history: list[tuple[str, float]] = field(default_factory=list)
    method_choices: dict[str, str] = field(default_factory=dict)
    extra_params: dict[str, dict[str, object]] = field(default_factory=dict)


# ============================================================
# Shortlist computation per method
# ============================================================
def _univariate_shortlist(
    table: pd.DataFrame,
    *,
    q_threshold: float,
    eta_threshold: float,
) -> set[str]:
    sig = table["q_value"] < q_threshold
    big = table["eta_squared"] >= eta_threshold
    return set(table.loc[sig & big, "feature_id"].astype(str))


def _plsda_shortlist(
    table: pd.DataFrame,
    *,
    vip_threshold: float,
    stability_threshold: float,
    p_threshold: float,
) -> set[str]:
    mask = (
        (table["vip"] >= vip_threshold)
        & (table["stability_score"] >= stability_threshold)
        & (table["perm_p_value"] < p_threshold)
    )
    return set(table.loc[mask, "feature_id"].astype(str))


def _splsda_shortlist(
    table: pd.DataFrame,
    *,
    stability_threshold: float,
    p_threshold: float,
) -> set[str]:
    sel = table.get("selected", pd.Series([False] * len(table))).astype(bool)
    mask = sel & (
        (table["stability_score"] >= stability_threshold)
        & (table["perm_p_value"] < p_threshold)
    )
    return set(table.loc[mask, "feature_id"].astype(str))


def _rf_shortlist(
    table: pd.DataFrame,
    *,
    stability_threshold: float,
    p_threshold: float,
) -> set[str]:
    mask = (
        (table["stability_score"] >= stability_threshold)
        & (table["perm_p_value"] < p_threshold)
    )
    return set(table.loc[mask, "feature_id"].astype(str))


# ============================================================
# Figure builders
# ============================================================
def _attrition_waterfall(
    history: list[tuple[str, int]],
    *,
    title: str,
) -> go.Figure:
    """Bar chart of cumulative feature attrition through the pipeline.

    Decreasing bars = filters removing features; the final bar shows
    survivors heading into multivariate analysis. Plotly's
    ``go.Waterfall`` would also work but ordinary bars with a connecting
    trend line are easier to read at glance.
    """
    if not history:
        fig = go.Figure()
        apply_theme(fig, title=title + " (no data)", xaxis_title="stage", yaxis_title="features")
        return fig

    stage_names = [h[0] for h in history]
    counts = [int(h[1]) for h in history]
    # Compute drops (negative deltas) for hover annotation
    deltas: list[int] = []
    for i, c in enumerate(counts):
        deltas.append(0 if i == 0 else c - counts[i - 1])

    bar_colors = ["#0072B2" if d == 0 or i == 0 else "#888888" for i, d in enumerate(deltas)]
    bar_colors[0] = "#222222"  # input bar in dark to anchor

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=stage_names,
            y=counts,
            marker={"color": bar_colors},
            text=[f"{c:,}" for c in counts],
            textposition="outside",
            hovertemplate=(
                "%{x}<br>features: %{y:,}<br>delta vs prev: %{customdata:+,}<extra></extra>"
            ),
            customdata=deltas,
        )
    )
    # Trend line connecting the bars
    fig.add_trace(
        go.Scatter(
            x=stage_names,
            y=counts,
            mode="lines+markers",
            line={"color": "#D55E00", "width": 1.5},
            marker={"color": "#D55E00", "size": 6, "symbol": "diamond"},
            name="trend",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    apply_theme(fig, title=title, xaxis_title="pipeline stage", yaxis_title="features remaining")
    fig.update_layout(xaxis={"tickangle": -55, "tickfont": {"size": 10}})
    return fig


def _jaccard_heatmap(
    method_shortlists: dict[str, set[str]],
    *,
    title: str,
) -> go.Figure:
    """k × k pairwise Jaccard heatmap of method shortlists (Nogueira 2018).

    Convention used here: ``Jaccard(∅, ∅) = 1`` — two empty shortlists
    are by definition identical. This keeps the diagonal at 1.0 even
    when a method's shortlist is empty under the chosen thresholds, and
    matches the standard limit-of-shrinking-sets convention used in the
    set-similarity literature (Real & Vargas 1996).
    """
    methods = list(method_shortlists.keys())
    k = len(methods)
    grid = np.full((k, k), np.nan, dtype=float)
    sizes = np.zeros((k, k), dtype=int)
    for i, m_a in enumerate(methods):
        a = method_shortlists[m_a]
        for j, m_b in enumerate(methods):
            b = method_shortlists[m_b]
            if not a and not b:
                # Both empty → identical → Jaccard = 1.0 (set-similarity
                # convention; keeps the diagonal coherent for empty
                # shortlists). Off-diagonals where both are empty also
                # collapse to 1.0 — they "agree" by both having nothing.
                grid[i, j] = 1.0
                sizes[i, j] = 0
                continue
            jac = len(a & b) / max(1, len(a | b))
            grid[i, j] = jac
            sizes[i, j] = len(a & b)

    text = [
        [
            f"{grid[i, j]:.2f}<br>(n={sizes[i, j]})" if np.isfinite(grid[i, j]) else "—"
            for j in range(k)
        ]
        for i in range(k)
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            x=methods,
            y=methods,
            colorscale=[[0.0, "#ffffff"], [0.5, "#9ecae1"], [1.0, "#08306b"]],
            zmin=0,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#222"},
            hovertemplate="%{y} ∩ %{x}<br>Jaccard: %{z:.3f}<extra></extra>",
            colorbar={"title": "Jaccard", "thickness": 12},
        )
    )
    apply_theme(fig, title=title, xaxis_title="method", yaxis_title="method")
    fig.update_yaxes(autorange="reversed")
    return fig


def _consensus_count_bar(
    method_shortlists: dict[str, set[str]],
    *,
    title: str,
) -> go.Figure:
    """Bar of how many features are flagged by exactly N methods (1..k).

    Replaces the Venn diagram with a clean ordinal view: features
    flagged by ≥ 2 methods are the consensus candidates the multiverse
    is supposed to surface (Steegen 2016).
    """
    if not method_shortlists:
        fig = go.Figure()
        apply_theme(fig, title=title + " (no shortlists)", xaxis_title="agreement", yaxis_title="features")
        return fig

    all_features: set[str] = set().union(*method_shortlists.values())
    if not all_features:
        fig = go.Figure()
        apply_theme(fig, title=title + " (empty shortlists)", xaxis_title="agreement", yaxis_title="features")
        return fig

    methods = list(method_shortlists.keys())
    k = len(methods)
    counts = np.zeros(k + 1, dtype=int)
    for fid in all_features:
        n_methods = sum(fid in method_shortlists[m] for m in methods)
        counts[n_methods] += 1
    # We never have agreement of 0 (would mean feature not in any shortlist)
    counts = counts[1:]
    labels = [f"{i}/{k}" + (" methods" if i > 1 else " method") for i in range(1, k + 1)]
    colors = ["#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
    bar_colors = [colors[min(i, len(colors) - 1)] for i in range(k)]

    fig = go.Figure(
        data=go.Bar(
            x=labels,
            y=counts.tolist(),
            marker={"color": bar_colors},
            text=[str(int(c)) for c in counts],
            textposition="outside",
            hovertemplate="agreement: %{x}<br>n features: %{y}<extra></extra>",
        )
    )
    apply_theme(
        fig,
        title=title,
        xaxis_title="agreement (methods that flagged the feature)",
        yaxis_title="features",
    )
    return fig


def _runtime_bar(
    history: list[tuple[str, float]],
    *,
    title: str,
) -> go.Figure:
    """Per-stage runtime in seconds."""
    if not history:
        fig = go.Figure()
        apply_theme(fig, title=title + " (no data)", xaxis_title="stage", yaxis_title="seconds")
        return fig
    stages = [h[0] for h in history]
    seconds = [float(h[1]) for h in history]
    fig = go.Figure(
        data=go.Bar(
            x=stages,
            y=seconds,
            marker={"color": "#0072B2"},
            text=[f"{s:.2f}s" for s in seconds],
            textposition="outside",
            hovertemplate="%{x}<br>runtime: %{y:.3f}s<extra></extra>",
        )
    )
    apply_theme(
        fig, title=title, xaxis_title="stage", yaxis_title="runtime (seconds)"
    )
    fig.update_layout(xaxis={"tickangle": -55, "tickfont": {"size": 10}})
    return fig


# ============================================================
# Section entrypoint
# ============================================================
def add_path_summary_section(
    report: PathReport,
    *,
    summary: PathSummary,
    univariate_result: UnivariateResult | None = None,
    multivariate_results: dict[str, MultivariateResult] | None = None,
    section_name: str = "path_summary",
    q_threshold: float = 0.05,
    eta_threshold: float = 0.14,
    vip_threshold: float = 1.0,
    stability_threshold: float = 0.8,
    perm_p_threshold: float = 0.05,
) -> dict[str, set[str]]:
    """Add the path-level summary section to ``report``.

    Returns the per-method shortlists dict so callers can pipe it into
    Phase 9 (consensus shortlist) without reconstructing it.
    """
    multivariate_results = multivariate_results or {}

    report.add_section(
        section_name,
        title="Path summary — attrition, runtime, cross-method consensus",
        description=(
            "End-of-path roll-up. Shows feature attrition through the "
            "pipeline, runtime breakdown, and the cross-method overlap of "
            "biomarker shortlists computed at standard thresholds. "
            "Consensus across methods (≥ 2 / 3+) is the multiverse "
            "design's strongest defensible signal (Steegen 2016)."
        ),
        method="path_summary",
        parameters={
            "q_threshold": q_threshold,
            "eta_threshold": eta_threshold,
            "vip_threshold": vip_threshold,
            "stability_threshold": stability_threshold,
            "perm_p_threshold": perm_p_threshold,
        },
    )

    # ── Compute per-method shortlists ────────────────────────────
    shortlists: dict[str, set[str]] = {}
    if univariate_result is not None:
        shortlists["univariate"] = _univariate_shortlist(
            univariate_result.table,
            q_threshold=q_threshold,
            eta_threshold=eta_threshold,
        )
    for name, res in multivariate_results.items():
        if res.method_name == "plsda":
            shortlists["plsda"] = _plsda_shortlist(
                res.per_feature,
                vip_threshold=vip_threshold,
                stability_threshold=stability_threshold,
                p_threshold=perm_p_threshold,
            )
        elif res.method_name == "splsda":
            shortlists["splsda"] = _splsda_shortlist(
                res.per_feature,
                stability_threshold=stability_threshold,
                p_threshold=perm_p_threshold,
            )
        elif res.method_name == "rf":
            shortlists["rf"] = _rf_shortlist(
                res.per_feature,
                stability_threshold=stability_threshold,
                p_threshold=perm_p_threshold,
            )
        else:
            # Unknown method — skip but note it
            shortlists[name] = set()

    # ── Headline metrics ──────────────────────────────────────────
    if summary.attrition_history:
        first_count = int(summary.attrition_history[0][1])
        last_count = int(summary.attrition_history[-1][1])
        retained_pct = 100.0 * last_count / max(1, first_count)
        report.add_metric(
            section_name,
            "n_features_input",
            first_count,
            description="features at the very first stage",
        )
        report.add_metric(
            section_name,
            "n_features_final",
            last_count,
            description="features after the final attrition step",
        )
        report.add_metric(
            section_name,
            "overall retention %",
            round(retained_pct, 2),
        )
    if summary.runtime_history:
        total_runtime = sum(float(s) for _, s in summary.runtime_history)
        report.add_metric(
            section_name,
            "total runtime (s)",
            round(total_runtime, 2),
        )
    for method_name, sl in shortlists.items():
        report.add_metric(
            section_name,
            f"shortlist size — {method_name}",
            len(sl),
            unit="features",
        )
    consensus_features = (
        set().union(*shortlists.values()) if shortlists else set()
    )
    report.add_metric(
        section_name,
        "n_features_in_any_shortlist",
        len(consensus_features),
        unit="features",
    )
    if shortlists:
        # n features in ALL shortlists (unanimity)
        unanimity = (
            set.intersection(*shortlists.values()) if shortlists else set()
        )
        report.add_metric(
            section_name,
            "n_features_in_all_shortlists",
            len(unanimity),
            unit="features",
        )

    # ── Figure 1: attrition waterfall ─────────────────────────────
    if summary.attrition_history:
        atr_fig = _attrition_waterfall(
            summary.attrition_history,
            title="Feature attrition waterfall",
        )
        report.add_figure(
            section_name,
            title="Feature attrition waterfall",
            fig=atr_fig,
            caption=(
                "Bar height = features remaining after each stage. The "
                "biomarker-discovery shape: steep early drops at presence / "
                "blank filters, gentler attrition through variance-aware "
                "stages, then a final shortlist cliff. Hover for the per-"
                "step delta."
            ),
            method_metric="cumulative attrition",
        )

    # ── Figure 2: cross-method Jaccard heatmap ────────────────────
    if len(shortlists) >= 2:
        jac_fig = _jaccard_heatmap(
            shortlists,
            title="Cross-method shortlist Jaccard similarity",
        )
        report.add_figure(
            section_name,
            title="Cross-method shortlist Jaccard heatmap",
            fig=jac_fig,
            caption=(
                "Pairwise Jaccard = |A ∩ B| / |A ∪ B| of method-specific "
                "shortlists. High values along a row = that method agrees "
                "with the others; isolated low cells = method-specific "
                "discoveries that need extra justification before going on "
                "the consensus shortlist."
            ),
            method_metric="pairwise Jaccard (Nogueira 2018)",
        )

    # ── Figure 3: agreement-count bar ─────────────────────────────
    if shortlists and any(shortlists.values()):
        cb_fig = _consensus_count_bar(
            shortlists,
            title="Cross-method consensus — features flagged by N methods",
        )
        report.add_figure(
            section_name,
            title="Consensus agreement bar",
            fig=cb_fig,
            caption=(
                "Cleaner alternative to a Venn diagram: how many features "
                "are flagged by exactly 1, 2, 3, … methods? Tier-1 "
                "consensus = features in ≥ 3 / k methods (Steegen 2016)."
            ),
            method_metric="multiverse consensus tiers",
        )

    # ── Figure 4: runtime breakdown ───────────────────────────────
    if summary.runtime_history:
        rt_fig = _runtime_bar(
            summary.runtime_history,
            title="Per-stage runtime",
        )
        report.add_figure(
            section_name,
            title="Per-stage runtime breakdown",
            fig=rt_fig,
            caption=(
                "Wall-clock seconds per pipeline stage. Bootstrap and "
                "permutation loops dominate multivariate methods; "
                "filtering and per-feature stats are typically &lt; 1 s "
                "each."
            ),
            method_metric="runtime breakdown",
        )

    # ── Table: path config ────────────────────────────────────────
    if summary.method_choices:
        config_rows: list[dict[str, str]] = []
        for cat, method in summary.method_choices.items():
            params = summary.extra_params.get(cat, {})
            params_str = ", ".join(
                f"{k}={v}" for k, v in params.items()
            ) if params else "—"
            config_rows.append({"category": cat, "method": method, "parameters": params_str})
        report.add_table(
            section_name,
            title="Path method choices (full configuration)",
            df=pd.DataFrame(config_rows),
            caption=(
                "Every method choice in the path, with the parameters "
                "actually used. Combined with the path-chain banner at the "
                "top, this is the single source of truth for reproducibility."
            ),
        )

    # ── Table: per-method shortlist sizes ─────────────────────────
    if shortlists:
        rows: list[dict[str, str]] = [
            {
                "method": method,
                "n_features": str(len(sl)),
                "thresholds": {
                    "univariate": f"q < {q_threshold:g} AND η² ≥ {eta_threshold:g}",
                    "plsda": (
                        f"VIP ≥ {vip_threshold:g} AND stability ≥ "
                        f"{stability_threshold:g} AND perm_p < {perm_p_threshold:g}"
                    ),
                    "splsda": (
                        f"selected AND stability ≥ {stability_threshold:g} "
                        f"AND perm_p < {perm_p_threshold:g}"
                    ),
                    "rf": (
                        f"stability ≥ {stability_threshold:g} AND perm_p < "
                        f"{perm_p_threshold:g}"
                    ),
                }.get(method, "—"),
            }
            for method, sl in shortlists.items()
        ]
        report.add_table(
            section_name,
            title="Per-method shortlist sizes (with applied thresholds)",
            df=pd.DataFrame(rows),
            caption=(
                "Default mQACC thresholds — the Phase 9 shortlist stage "
                "replaces these with user-tunable rules. Sizes here are "
                "diagnostic, not the final per-path shortlist."
            ),
        )

    return shortlists
