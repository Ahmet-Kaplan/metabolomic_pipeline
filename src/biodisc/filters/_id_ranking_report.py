"""Stage 9 — HTML report for the identification candidate-ranking stage.

Seven plots + an index:

    1. MSI tier distribution — L2 / L3 / L4 (no floor survivor) / L4 (no
       candidate). The headline annotation-quality picture.
    2. Cascade resolution funnel — where in the cascade each compound got
       resolved (Score / Frag / Mass error / single-after-floor / still
       tied / no floor survivor / no candidate).
    3. Top-20 highest-Score L2 unique annotations — the strongest single-
       candidate IDs in the dataset, ranked by Progenesis Score.
    4. Top-20 most-ambiguous L3 features — the biologist watch-list of
       compounds with the most unresolved isomers/duplicates.
    5. Score distribution by MSI tier — boxplot of top_score for L2 vs L3.
       Tells you whether L3 ambiguity is at high or low quality.
    6. Mass-error |ppm| distribution of kept annotations — confirms mass-
       accuracy health; the 5 ppm floor line is overlaid for reference.
    7. L4-no-floor-survivor casualty breakdown — for the features whose
       candidates ALL failed the quality floor, was Score, |ppm|, or both
       to blame? Diagnostic for whether the floor is calibrated correctly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from biodisc.eda._output import write_plot
from biodisc.eda.report import ReportEntry, build_index


def plot_msi_breakdown(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Bar chart of MSI tier distribution."""
    counts = {
        "L2 (unique putative)": int((metrics["msi_tier"] == 2).sum()) if not metrics.empty else 0,
        "L3 (compound class)": int((metrics["msi_tier"] == 3).sum()) if not metrics.empty else 0,
        "L4 (no floor survivor)": int((metrics["outcome"] == "l4_no_floor_survivor").sum())
        if not metrics.empty
        else 0,
        "L4 (no candidate)": int((metrics["outcome"] == "l4_no_candidate").sum())
        if not metrics.empty
        else 0,
    }
    total = sum(counts.values())
    labels = list(counts.keys())
    values = list(counts.values())
    pcts = [f"{v / max(total, 1) * 100:.1f} %" for v in values]
    text = [f"{v:,}<br>{p}" for v, p in zip(values, pcts, strict=True)]

    colours = ["#1e4377", "#9aa6b2", "#c7254e", "#a04060"]
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
                "<b>MSI tier distribution after candidate ranking</b><br>"
                f"<sup>{total:,} active features</sup>"
            )
        ),
        xaxis_title="MSI tier",
        yaxis_title="Feature count",
        height=540,
        showlegend=False,
    )
    return write_plot(fig, output_dir, name="01_msi_breakdown", emit_png=emit_png)


def plot_resolution_funnel(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Bar chart of where each compound's ranking was resolved."""
    if metrics.empty:
        counts = dict.fromkeys(
            [
                "single_after_floor",
                "score",
                "frag",
                "ppm",
                "tied",
                "no_floor_survivor",
                "no_candidate",
            ],
            0,
        )
    else:
        counts = {
            k: int((metrics["resolved_at"] == k).sum())
            for k in [
                "single_after_floor",
                "score",
                "frag",
                "ppm",
                "tied",
                "no_floor_survivor",
                "no_candidate",
            ]
        }

    labels = [
        "Single after floor",
        "Resolved at Score",
        "Resolved at Frag",
        "Resolved at Mass error",
        "Still tied (L3)",
        "No floor survivor (L4)",
        "No candidate (L4)",
    ]
    values = [
        counts["single_after_floor"],
        counts["score"],
        counts["frag"],
        counts["ppm"],
        counts["tied"],
        counts["no_floor_survivor"],
        counts["no_candidate"],
    ]
    colours = [
        "#1e4377",
        "#1e4377",
        "#5a7bb1",
        "#9aa6b2",
        "#c9a94e",
        "#c7254e",
        "#a04060",
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colours,
            text=[f"{v:,}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text="<b>Cascade resolution breakdown</b>"),
        xaxis_title="Resolution outcome",
        yaxis_title="Feature count",
        height=540,
        showlegend=False,
    )
    return write_plot(fig, output_dir, name="02_resolution_funnel", emit_png=emit_png)


def plot_top_l2(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    n_top: int = 20,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Top-N highest-Score L2 unique annotations."""
    if metrics.empty:
        l2 = metrics
    else:
        l2 = metrics[metrics["msi_tier"] == 2].copy()
        l2 = l2.dropna(subset=["top_score"])
        l2 = l2.nlargest(n_top, "top_score")

    if l2.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No L2 unique annotations.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        fig.update_layout(height=420)
        return write_plot(fig, output_dir, name="03_top_l2", emit_png=emit_png)

    l2["label"] = (
        l2["feature_id"].astype(str)
        + "  →  "
        + l2["top_compound_name"].fillna("(unnamed)").str[:40]
    )
    l2 = l2.iloc[::-1]  # ascending bottom-up so highest is at top of bar

    frag_str = l2["top_frag_score"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    ppm_str = l2["top_ppm"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    customdata = list(zip(frag_str, ppm_str, l2["top_compound_id"].fillna("—"), strict=True))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=l2["top_score"],
            y=l2["label"],
            orientation="h",
            marker_color="#1e4377",
            text=[f"{v:.1f}" for v in l2["top_score"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Score: %{x:.2f}<br>"
                "Frag Score: %{customdata[0]}<br>"
                "Mass error (ppm): %{customdata[1]}<br>"
                "Compound ID: %{customdata[2]}<extra></extra>"
            ),
            customdata=customdata,
        )
    )
    fig.update_layout(
        title=dict(text=f"<b>Top {n_top} highest-Score L2 annotations</b>"),
        xaxis_title="Progenesis Score",
        yaxis=dict(title=None, automargin=True),
        height=max(400, 26 * len(l2) + 120),
        showlegend=False,
        margin=dict(l=10, r=20, t=80, b=40),
    )
    return write_plot(fig, output_dir, name="03_top_l2", emit_png=emit_png)


def plot_top_l3_ambiguous(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    n_top: int = 20,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Top-N most-ambiguous L3 features — the biologist watch-list."""
    if metrics.empty:
        l3 = metrics
    else:
        l3 = metrics[metrics["msi_tier"] == 3].copy()
        l3 = l3.nlargest(n_top, "n_candidates_final")

    if l3.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No L3 ambiguous features.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        fig.update_layout(height=420)
        return write_plot(fig, output_dir, name="04_top_l3_ambiguous", emit_png=emit_png)

    l3["label"] = (
        l3["feature_id"].astype(str)
        + "  →  "
        + l3["top_compound_name"].fillna("(unnamed)").str[:35]
    )
    l3 = l3.iloc[::-1]  # ascending bottom-up so worst is at top

    formula_str = l3["top_formula"].fillna("—")
    msms_str = l3["had_msms"].map({True: "yes", False: "no"})
    customdata = list(zip(formula_str, msms_str, l3["top_score"].fillna(0), strict=True))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=l3["n_candidates_final"],
            y=l3["label"],
            orientation="h",
            marker_color="#c9a94e",
            text=[f"{int(v):,}" for v in l3["n_candidates_final"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Surviving candidates: %{x:,}<br>"
                "Formula: %{customdata[0]}<br>"
                "MS/MS evidence: %{customdata[1]}<br>"
                "Score: %{customdata[2]:.1f}<extra></extra>"
            ),
            customdata=customdata,
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Top {n_top} most-ambiguous L3 features</b><br>"
                "<sup>High candidate counts often mean cross-database duplicates "
                "or genuine isomer families (e.g., lipid positional isomers).</sup>"
            )
        ),
        xaxis_title="Surviving candidates after cascade",
        yaxis=dict(title=None, automargin=True),
        height=max(400, 26 * len(l3) + 140),
        showlegend=False,
        margin=dict(l=10, r=20, t=100, b=40),
    )
    return write_plot(fig, output_dir, name="04_top_l3_ambiguous", emit_png=emit_png)


def plot_score_distribution_by_tier(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Box plot of top_score for L2 vs L3 features."""
    if metrics.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No annotated features.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        fig.update_layout(height=420)
        return write_plot(fig, output_dir, name="05_score_by_tier", emit_png=emit_png)

    l2_scores = metrics.loc[metrics["msi_tier"] == 2, "top_score"].dropna()
    l3_scores = metrics.loc[metrics["msi_tier"] == 3, "top_score"].dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=l2_scores,
            name=f"L2 unique (n={len(l2_scores):,})",
            marker_color="#1e4377",
            boxpoints="outliers",
        )
    )
    fig.add_trace(
        go.Box(
            y=l3_scores,
            name=f"L3 compound class (n={len(l3_scores):,})",
            marker_color="#9aa6b2",
            boxpoints="outliers",
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                "<b>Annotation Score by MSI tier</b><br>"
                "<sup>L3 medians near 30 (the floor) imply borderline-quality ambiguity. "
                "L3 medians near L2 imply genuine isomer ambiguity at high quality.</sup>"
            )
        ),
        yaxis_title="Top Progenesis Score",
        xaxis_title="MSI tier",
        height=540,
        showlegend=False,
    )
    return write_plot(fig, output_dir, name="05_score_by_tier", emit_png=emit_png)


def plot_ppm_distribution(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    floor_ppm: float = 5.0,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Histogram of |top_ppm| for L2 + L3 annotations, with floor line overlaid."""
    if metrics.empty:
        annotated = metrics
    else:
        annotated = metrics[metrics["msi_tier"].isin([2, 3])].copy()
        annotated = annotated.dropna(subset=["top_ppm"])

    if annotated.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No annotated features with mass error available.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        fig.update_layout(height=420)
        return write_plot(fig, output_dir, name="06_ppm_distribution", emit_png=emit_png)

    abs_ppm = annotated["top_ppm"].abs()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=abs_ppm[annotated["msi_tier"] == 2],
            xbins=dict(start=0.0, end=max(abs_ppm.max() + 0.5, floor_ppm + 1.0), size=0.25),
            marker_color="#1e4377",
            opacity=0.85,
            name=f"L2 unique (n={int((annotated['msi_tier'] == 2).sum()):,})",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=abs_ppm[annotated["msi_tier"] == 3],
            xbins=dict(start=0.0, end=max(abs_ppm.max() + 0.5, floor_ppm + 1.0), size=0.25),
            marker_color="#9aa6b2",
            opacity=0.85,
            name=f"L3 ambiguous (n={int((annotated['msi_tier'] == 3).sum()):,})",
        )
    )
    fig.add_vline(
        x=floor_ppm,
        line=dict(color="#c9a94e", width=2, dash="dash"),
        annotation_text=f"Quality floor: {floor_ppm:g} ppm",
        annotation_position="top right",
    )
    fig.update_layout(
        title=dict(
            text=(
                "<b>Mass-error |ppm| distribution of kept annotations</b><br>"
                "<sup>Healthy Q-TOF performance: the bulk should sit well below "
                f"the {floor_ppm:g} ppm floor.</sup>"
            )
        ),
        xaxis_title="|Mass error| (ppm)",
        yaxis_title="Feature count",
        barmode="stack",
        bargap=0.02,
        height=560,
        legend=dict(title="MSI tier"),
    )
    return write_plot(fig, output_dir, name="06_ppm_distribution", emit_png=emit_png)


def plot_floor_failure_breakdown(
    metrics: pd.DataFrame,
    identification: pd.DataFrame,
    *,
    output_dir: Path,
    min_score: float,
    max_mass_error_ppm: float,
    emit_png: bool = True,
) -> dict[str, Path]:
    """Diagnostic — for the L4-no-floor-survivor features, did Score, ppm, or both block them?

    Per feature, the verdict comes from its closest-to-passing candidate:
      - "Score-fail (loosen Score floor would rescue)" — best candidate had |ppm|≤floor
        but Score<floor. Loosening Score may save these.
      - "|ppm|-fail (loosen ppm floor would rescue)" — best candidate had Score≥floor
        but |ppm|>floor. Loosening ppm may save these.
      - "Both fail (no rescue path)" — no candidate met either criterion.
    """
    if metrics.empty:
        floor_fails = metrics
    else:
        floor_fails = metrics[metrics["outcome"] == "l4_no_floor_survivor"]

    if floor_fails.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No floor-failure features to diagnose.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        fig.update_layout(height=420)
        return write_plot(fig, output_dir, name="07_floor_failure_breakdown", emit_png=emit_png)

    fail_ids = set(floor_fails["feature_id"])
    fail_idents = identification[identification["feature_id"].isin(fail_ids)].copy()

    score_pass_mask = fail_idents["score"].fillna(-1) >= min_score
    ppm_pass_mask = (
        fail_idents["mass_error_ppm"].abs().fillna(float("inf")) <= max_mass_error_ppm
    )

    # Per-feature: does ANY candidate satisfy each individual criterion?
    grouped = fail_idents.groupby("feature_id")
    has_score_pass = grouped.apply(lambda g: bool(score_pass_mask.loc[g.index].any()))
    has_ppm_pass = grouped.apply(lambda g: bool(ppm_pass_mask.loc[g.index].any()))

    diag = pd.DataFrame(
        {
            "feature_id": has_score_pass.index,
            "any_score_pass": has_score_pass.values,
            "any_ppm_pass": has_ppm_pass.values,
        }
    )

    def _classify(row: pd.Series) -> str:
        if row["any_score_pass"] and not row["any_ppm_pass"]:
            return "ppm_only_blocker"  # candidate near on Score, blocked by ppm
        if row["any_ppm_pass"] and not row["any_score_pass"]:
            return "score_only_blocker"  # candidate near on ppm, blocked by Score
        if row["any_score_pass"] and row["any_ppm_pass"]:
            return "cross_fail"  # different candidates near on each axis
        return "both_fail"  # nothing close

    diag["category"] = diag.apply(_classify, axis=1)

    counts = diag["category"].value_counts().to_dict()
    labels = [
        f"Score-fail only<br>(loosen Score ≥ {min_score:g} would help)",
        f"|ppm|-fail only<br>(loosen |ppm| ≤ {max_mass_error_ppm:g} would help)",
        "Cross-fail<br>(near on each axis, different candidates)",
        "Both fail<br>(no rescue path — junk-quality)",
    ]
    keys = ["score_only_blocker", "ppm_only_blocker", "cross_fail", "both_fail"]
    values = [int(counts.get(k, 0)) for k in keys]
    colours = ["#5a7bb1", "#c9a94e", "#9aa6b2", "#c7254e"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colours,
            text=[f"{v:,}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                f"<b>L4-no-floor-survivor breakdown ({len(diag):,} features)</b><br>"
                "<sup>Where would loosening one threshold help? "
                f"Floor: Score ≥ {min_score:g} AND |ppm| ≤ {max_mass_error_ppm:g}.</sup>"
            )
        ),
        xaxis_title="Failure category",
        yaxis_title="Feature count",
        height=560,
        showlegend=False,
    )
    return write_plot(fig, output_dir, name="07_floor_failure_breakdown", emit_png=emit_png)


def build_report(
    metrics: pd.DataFrame,
    identification: pd.DataFrame,
    *,
    output_dir: Path,
    counts: dict[str, int],
    config_summary: dict[str, float],
    emit_png: bool = True,
) -> dict[str, Path]:
    """Compose the seven plots into an index.html."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    entries: list[ReportEntry] = []
    outputs: dict[str, Path] = {}

    p1 = plot_msi_breakdown(metrics, output_dir=plots_dir, emit_png=emit_png)
    outputs["01_msi_breakdown"] = p1["html"]
    entries.append(
        ReportEntry(
            key="01_msi_breakdown",
            title="1. MSI tier distribution",
            description=(
                "Each active feature is assigned to exactly one MSI tier: "
                "<b>L2</b> (single putative annotation surviving the cascade), "
                "<b>L3</b> (compound class — multiple equally-good candidates "
                "remain after Score / Frag / Mass-error tiebreaks; biologist "
                "decides), or <b>L4</b> (no annotation — either no candidate "
                "cleared the quality floor, or no Progenesis candidate exists "
                "for this feature)."
            ),
            html_href=f"plots/{p1['html'].name}",
            png_href=f"plots/{p1['png'].name}" if "png" in p1 else None,
        )
    )

    p2 = plot_resolution_funnel(metrics, output_dir=plots_dir, emit_png=emit_png)
    outputs["02_resolution_funnel"] = p2["html"]
    entries.append(
        ReportEntry(
            key="02_resolution_funnel",
            title="2. Cascade resolution breakdown",
            description=(
                "Where did the cascade do its work? Most compounds resolve "
                "trivially (single candidate after the quality floor) or at "
                "Step 1 (max Score). A small fraction need Step 2 "
                "(Fragmentation Score) or Step 3 (min |Mass error|) to break "
                "ties. The remainder are honestly reported as L3 ambiguous "
                "(compound class only) or L4 unannotated."
            ),
            html_href=f"plots/{p2['html'].name}",
            png_href=f"plots/{p2['png'].name}" if "png" in p2 else None,
        )
    )

    p3 = plot_top_l2(metrics, output_dir=plots_dir, emit_png=emit_png)
    outputs["03_top_l2"] = p3["html"]
    entries.append(
        ReportEntry(
            key="03_top_l2",
            title="3. Top-20 highest-Score L2 annotations",
            description=(
                "The 20 most-confident single-candidate annotations in the "
                "dataset, ranked by Progenesis Score. Hover for fragmentation "
                "score, mass error, and compound ID. These are your strongest "
                "putative IDs heading into stats and shortlist."
            ),
            html_href=f"plots/{p3['html'].name}",
            png_href=f"plots/{p3['png'].name}" if "png" in p3 else None,
        )
    )

    p4 = plot_top_l3_ambiguous(metrics, output_dir=plots_dir, emit_png=emit_png)
    outputs["04_top_l3_ambiguous"] = p4["html"]
    entries.append(
        ReportEntry(
            key="04_top_l3_ambiguous",
            title="4. Top-20 most-ambiguous L3 features (watch list)",
            description=(
                "The 20 features with the most surviving candidates after the "
                "cascade — your biologist watch-list. High counts typically "
                "mean cross-database duplicates of the same molecule "
                "(HMDB + ChemSpider) or genuine isomer families that LC-MS "
                "cannot physically distinguish (e.g., lipid positional isomers). "
                "Compound class is the honest annotation here."
            ),
            html_href=f"plots/{p4['html'].name}",
            png_href=f"plots/{p4['png'].name}" if "png" in p4 else None,
        )
    )

    p5 = plot_score_distribution_by_tier(metrics, output_dir=plots_dir, emit_png=emit_png)
    outputs["05_score_by_tier"] = p5["html"]
    entries.append(
        ReportEntry(
            key="05_score_by_tier",
            title="5. Score distribution by MSI tier",
            description=(
                "Boxplot of top Progenesis Score for L2 vs L3 features. "
                "If L3 medians cluster near the floor (Score ≈ 30), the floor "
                "may be too lenient — low-quality candidates are entering the "
                "cascade. If L3 medians are close to L2's, the ambiguity is "
                "high-quality (genuine isomer/database duplication) and the "
                "floor is well-calibrated."
            ),
            html_href=f"plots/{p5['html'].name}",
            png_href=f"plots/{p5['png'].name}" if "png" in p5 else None,
        )
    )

    p6 = plot_ppm_distribution(
        metrics,
        output_dir=plots_dir,
        floor_ppm=config_summary.get("max_mass_error_ppm", 5.0),
        emit_png=emit_png,
    )
    outputs["06_ppm_distribution"] = p6["html"]
    entries.append(
        ReportEntry(
            key="06_ppm_distribution",
            title="6. Mass-error |ppm| distribution of kept annotations",
            description=(
                "Distribution of |mass error| for L2 + L3 kept annotations, "
                "with the quality-floor line overlaid. A healthy Q-TOF run "
                "puts the bulk well below the floor (typically < 2 ppm). "
                "If many surviving annotations sit close to the floor, the "
                "instrument may be drifting or miscalibrated."
            ),
            html_href=f"plots/{p6['html'].name}",
            png_href=f"plots/{p6['png'].name}" if "png" in p6 else None,
        )
    )

    p7 = plot_floor_failure_breakdown(
        metrics,
        identification,
        output_dir=plots_dir,
        min_score=config_summary.get("min_score", 30.0),
        max_mass_error_ppm=config_summary.get("max_mass_error_ppm", 5.0),
        emit_png=emit_png,
    )
    outputs["07_floor_failure_breakdown"] = p7["html"]
    entries.append(
        ReportEntry(
            key="07_floor_failure_breakdown",
            title="7. L4-no-floor-survivor failure diagnostic",
            description=(
                "For features whose candidates ALL failed the quality floor, "
                "which gate did the work? Score-fail-only candidates would "
                "be rescued by lowering the Score floor; |ppm|-fail-only by "
                "loosening the mass-error floor. Cross-fails have near-passes "
                "on each axis but on different candidates. Both-fail features "
                "are genuinely junk-quality with no realistic rescue path. "
                "If one bar dominates, the corresponding floor may be too tight."
            ),
            html_href=f"plots/{p7['html'].name}",
            png_href=f"plots/{p7['png'].name}" if "png" in p7 else None,
        )
    )

    n_input = counts.get("n_input", 0)
    extra_info = {
        "Input features (Stage-8-active)": f"{n_input:,}",
        "  • With Progenesis candidate(s)": f"{counts.get('n_with_candidates', 0):,}",
        "  • No candidate (auto L4)": f"{counts.get('n_msi_l4_no_candidate', 0):,}",
        "MSI L2 (unique annotation)": f"{counts.get('n_msi_l2', 0):,}",
        "MSI L3 (compound class)": f"{counts.get('n_msi_l3', 0):,}",
        "MSI L4 (no floor survivor)": f"{counts.get('n_msi_l4_no_floor_survivor', 0):,}",
        "MSI L4 (no candidate)": f"{counts.get('n_msi_l4_no_candidate', 0):,}",
        "Single after quality floor": f"{counts.get('n_resolved_single_after_floor', 0):,}",
        "Resolved at Step 1 (Score)": f"{counts.get('n_resolved_at_score', 0):,}",
        "Resolved at Step 2 (Frag Score)": f"{counts.get('n_resolved_at_frag', 0):,}",
        "Resolved at Step 3 (Mass error)": f"{counts.get('n_resolved_at_ppm', 0):,}",
        "Still tied after all 3 steps": f"{counts.get('n_still_tied', 0):,}",
        "Quality floor": (
            f"Score ≥ {config_summary.get('min_score', 30):g} "
            f"AND |ppm| ≤ {config_summary.get('max_mass_error_ppm', 5):g}"
        ),
        "Note": (
            "Stage 9 is annotation-only — every active feature passes through. "
            "L4 features are not dropped; they proceed to downstream stages "
            "without an annotation, where stats may still flag them as biomarkers."
        ),
    }
    index_path = build_index(
        output_path=output_dir / "index.html",
        entries=entries,
        stage_label="Stage 9 — Identification candidate ranking (annotation-only)",
        extra_info=extra_info,
    )
    outputs["index"] = index_path
    return outputs
