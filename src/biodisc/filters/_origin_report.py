"""Stage 10 — HTML report for the origin filter.

Three plots + an index:

    1. Outcome breakdown — endogenous_evidence / exo_or_unknown /
       passthrough_l4 / all_exogenous-dropped. The headline.
    2. Top-20 dropped features — the diagnostic. Each bar is one dropped
       feature; hover shows the candidate compound names so you can
       sanity-check the drops (e.g. "diethyl phthalate dropped → makes
       sense, that's a plasticizer").
    3. Origin × MSI tier crosstab — stacked bar showing how each MSI tier
       distributes across origin outcomes. Diagnostic for whether L4
       features are mostly endogenous-likely or exogenous-likely.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from biodisc.eda._output import write_plot
from biodisc.eda.report import ReportEntry, build_index


def plot_outcome_breakdown(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Bar chart of origin outcomes (kept vs dropped)."""
    counts = {
        "Kept — endogenous evidence": int((metrics["outcome"] == "endogenous_evidence").sum())
        if not metrics.empty
        else 0,
        "Kept — exo-or-unknown (conservative)": int((metrics["outcome"] == "exo_or_unknown").sum())
        if not metrics.empty
        else 0,
        "Kept — pass-through L4": int((metrics["outcome"] == "passthrough_l4").sum())
        if not metrics.empty
        else 0,
        "Dropped — all candidates exogenous": int(
            ((metrics["outcome"] == "all_exogenous") & (~metrics["kept"])).sum()
        )
        if not metrics.empty
        else 0,
    }
    total = sum(counts.values())
    labels = list(counts.keys())
    values = list(counts.values())
    pcts = [f"{v / max(total, 1) * 100:.1f} %" for v in values]
    text = [f"{v:,}<br>{p}" for v, p in zip(values, pcts, strict=True)]
    colours = ["#1e4377", "#9aa6b2", "#5a7bb1", "#c7254e"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colours,
            text=text,
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                "<b>Origin outcome breakdown</b><br>"
                f"<sup>{total:,} active features evaluated</sup>"
            )
        ),
        xaxis_title="Outcome",
        yaxis_title="Feature count",
        height=560,
        showlegend=False,
    )
    return write_plot(fig, output_dir, name="01_outcome_breakdown", emit_png=emit_png)


def plot_top_dropped(
    metrics: pd.DataFrame,
    ranked_identification: pd.DataFrame,
    *,
    output_dir: Path,
    n_top: int = 20,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Top-N dropped features — diagnostic for verifying the drops are sensible."""
    dropped = metrics if metrics.empty else metrics[~metrics["kept"]].copy()

    if dropped.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No features dropped (all kept by origin filter).",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        fig.update_layout(height=420)
        return write_plot(fig, output_dir, name="02_top_dropped", emit_png=emit_png)

    # Top N by count of pure-Exo candidates (most-confidently-exogenous)
    dropped = dropped.nlargest(n_top, "n_exogenous_only").copy()

    # Attach a sample compound name from ranked_identification (first candidate)
    sample = (
        ranked_identification.groupby("feature_id")
        .agg(
            sample_compound=("description", "first"),
            sample_compound_id=("compound_id", "first"),
        )
        .reset_index()
    )
    dropped = dropped.merge(sample, on="feature_id", how="left")
    dropped["label"] = (
        dropped["feature_id"].astype(str)
        + "  →  "
        + dropped["sample_compound"].fillna("(unnamed)").str[:40]
    )
    dropped = dropped.iloc[::-1]  # ascending bottom-up so largest is at top

    customdata = list(
        zip(
            dropped["sample_compound_id"].fillna("—"),
            dropped["msi_tier"],
            dropped["n_exogenous_only"],
            strict=True,
        )
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dropped["n_exogenous_only"],
            y=dropped["label"],
            orientation="h",
            marker_color="#c7254e",
            text=[f"{int(v)}" for v in dropped["n_exogenous_only"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Pure-Exo candidates: %{x}<br>"
                "Sample compound ID: %{customdata[0]}<br>"
                "MSI tier: L%{customdata[1]}<extra></extra>"
            ),
            customdata=customdata,
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Top {n_top} dropped features (most pure-Exo candidates)</b><br>"
                "<sup>Sanity-check: do these compound names make sense as "
                "contamination (drugs, plasticizers, food, FBS)?</sup>"
            )
        ),
        xaxis_title="Number of pure-exogenous candidates",
        yaxis=dict(title=None, automargin=True),
        height=max(400, 26 * len(dropped) + 140),
        showlegend=False,
        margin=dict(l=10, r=20, t=100, b=40),
    )
    return write_plot(fig, output_dir, name="02_top_dropped", emit_png=emit_png)


def plot_msi_origin_crosstab(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Stacked bar: how each MSI tier distributes across origin outcomes."""
    if metrics.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        fig.update_layout(height=420)
        return write_plot(fig, output_dir, name="03_msi_origin_crosstab", emit_png=emit_png)

    metrics = metrics.copy()
    metrics["msi_label"] = metrics["msi_tier"].map({2: "L2 unique", 3: "L3 ambiguous", 4: "L4 unannot."})

    crosstab = (
        pd.crosstab(metrics["msi_label"], metrics["outcome"])
        .reindex(columns=["endogenous_evidence", "exo_or_unknown", "passthrough_l4", "all_exogenous"])
        .fillna(0)
    )
    crosstab = crosstab.reindex(["L2 unique", "L3 ambiguous", "L4 unannot."]).fillna(0)

    colours = {
        "endogenous_evidence": "#1e4377",
        "exo_or_unknown": "#9aa6b2",
        "passthrough_l4": "#5a7bb1",
        "all_exogenous": "#c7254e",
    }
    pretty = {
        "endogenous_evidence": "Endogenous evidence",
        "exo_or_unknown": "Exo or unknown (kept)",
        "passthrough_l4": "Pass-through L4",
        "all_exogenous": "All exogenous (dropped)",
    }

    fig = go.Figure()
    for col in crosstab.columns:
        fig.add_trace(
            go.Bar(
                x=crosstab.index,
                y=crosstab[col].astype(int),
                name=pretty[col],
                marker_color=colours[col],
                text=[f"{int(v):,}" if v > 0 else "" for v in crosstab[col]],
                textposition="inside",
            )
        )
    fig.update_layout(
        title=dict(
            text=(
                "<b>Origin outcome by MSI tier</b><br>"
                "<sup>How does annotation confidence (L2/L3/L4) correlate with "
                "origin classification?</sup>"
            )
        ),
        xaxis_title="MSI tier",
        yaxis_title="Feature count",
        barmode="stack",
        height=560,
        legend=dict(title="Origin outcome"),
    )
    return write_plot(fig, output_dir, name="03_msi_origin_crosstab", emit_png=emit_png)


def build_report(
    metrics: pd.DataFrame,
    ranked_identification: pd.DataFrame,
    *,
    output_dir: Path,
    counts: dict[str, int],
    config_summary: dict[str, object],
    emit_png: bool = True,
) -> dict[str, Path]:
    """Compose the three plots into an index.html."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    entries: list[ReportEntry] = []
    outputs: dict[str, Path] = {}

    p1 = plot_outcome_breakdown(metrics, output_dir=plots_dir, emit_png=emit_png)
    outputs["01_outcome_breakdown"] = p1["html"]
    entries.append(
        ReportEntry(
            key="01_outcome_breakdown",
            title="1. Origin outcome breakdown",
            description=(
                "Every active feature lands in exactly one bucket: "
                "<b>endogenous evidence</b> (≥ 1 surviving cascade candidate "
                "is HMDB-tagged Endogenous or Both), <b>exo-or-unknown</b> "
                "(no Endo evidence but ≥ 1 candidate has no annotation — "
                "kept conservatively), <b>pass-through L4</b> (no annotation "
                "at all — cannot classify), or <b>dropped — all exogenous</b> "
                "(every surviving candidate is HMDB-tagged exogenous-only)."
            ),
            html_href=f"plots/{p1['html'].name}",
            png_href=f"plots/{p1['png'].name}" if "png" in p1 else None,
        )
    )

    p2 = plot_top_dropped(
        metrics, ranked_identification, output_dir=plots_dir, emit_png=emit_png
    )
    outputs["02_top_dropped"] = p2["html"]
    entries.append(
        ReportEntry(
            key="02_top_dropped",
            title="2. Top-20 dropped features (sanity check)",
            description=(
                "The 20 features with the most pure-exogenous candidates that "
                "got dropped. Sample compound name shown for each. Use this as "
                "a sanity check: the names should make sense as contamination "
                "(drugs, plasticizers, food components, FBS-derived lipids). "
                "If you see compounds you'd expect to be biological here, the "
                "filter may be too aggressive for your data."
            ),
            html_href=f"plots/{p2['html'].name}",
            png_href=f"plots/{p2['png'].name}" if "png" in p2 else None,
        )
    )

    p3 = plot_msi_origin_crosstab(metrics, output_dir=plots_dir, emit_png=emit_png)
    outputs["03_msi_origin_crosstab"] = p3["html"]
    entries.append(
        ReportEntry(
            key="03_msi_origin_crosstab",
            title="3. Origin outcome by MSI tier",
            description=(
                "Stacked bar showing how features in each MSI tier "
                "distribute across origin outcomes. L2 and L3 features have "
                "Progenesis annotations and so a determinate origin call. "
                "L4 features have no annotation and are all classed as "
                "pass-through. If a high fraction of L2/L3 are pure-Exo, "
                "the dataset has substantial contamination signal."
            ),
            html_href=f"plots/{p3['html'].name}",
            png_href=f"plots/{p3['png'].name}" if "png" in p3 else None,
        )
    )

    n_input = counts.get("n_input", 0)
    extra_info = {
        "Input features (Stage-9-active)": f"{n_input:,}",
        "Kept (TOTAL)": (
            f"{counts.get('n_kept', 0):,} "
            f"({counts.get('n_kept', 0) / max(n_input, 1) * 100:.2f} %)"
        ),
        "  • Endogenous evidence": f"{counts.get('n_endogenous_evidence', 0):,}",
        "  • Exo-or-unknown (conservative keep)": f"{counts.get('n_exo_or_unknown', 0):,}",
        "  • Pass-through L4 (no annotation)": f"{counts.get('n_passthrough_l4', 0):,}",
        "Dropped — all candidates pure-Exo": f"{counts.get('n_dropped', 0):,}",
        "Drop mode": (
            "HARD drop"
            if config_summary.get("drop_pure_exogenous", True)
            else "SOFT tag (no drops)"
        ),
        "Endogenous source tag": '"Endogenous" (HMDB Source ontology)',
        "Aggregation rule": (
            "Any candidate Endo/Both → keep. "
            "All candidates pure-Exo → drop. "
            "Mixed Exo+Unannotated → keep (conservative). "
            "L4 (no annotation) → pass-through keep."
        ),
    }
    index_path = build_index(
        output_path=output_dir / "index.html",
        entries=entries,
        stage_label="Stage 10 — Origin filter (annotation-driven)",
        extra_info=extra_info,
    )
    outputs["index"] = index_path
    return outputs
