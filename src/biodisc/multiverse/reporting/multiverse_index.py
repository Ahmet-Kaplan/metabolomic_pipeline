"""Phase 8e.7 — top-level multiverse index page.

Builds ``data/reports/multiverse/index.html``: a landing page that
summarises the whole multiverse run (path table + specification curve +
cross-path Jaccard agreement + consensus tier breakdown + top features +
per-method drilldown). Re-uses the per-path :class:`PathReport`
machinery (sections / figures / tables / metrics) under a different
Jinja template so styling stays consistent with the per-path reports.

Inputs are the in-memory ``PathRunResult`` list returned by
:func:`run_multiverse` and the ``ConsensusResult`` from
:func:`compute_consensus` — same objects the smoke already passes around,
so no extra I/O is needed.

Why this matters
----------------
The per-path reports answer "what did *one* method choice produce?".
The index answers "how stable is the answer across method choices?" — the
multiverse design's only real product (Steegen 2016, Simonsohn 2020).
A reviewer should land on this page first.

References
----------
* Steegen S et al. 2016. *Increasing transparency through a multiverse
  analysis.* **Perspect Psychol Sci** 11:702.
* Simonsohn U et al. 2020. *Specification curve analysis.* **Nat Hum
  Behav** 4:1208.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse.reporting.core import PathReport
from biodisc.multiverse.reporting.figures import (
    consensus_tier_bar,
    jaccard_similarity_heatmap,
    method_drilldown_heatmap,
    specification_curve_bar,
)

if TYPE_CHECKING:
    from biodisc.multiverse.consensus import ConsensusResult
    from biodisc.multiverse.orchestrator import PathRunResult

log = structlog.get_logger(__name__)

_METHOD_AXIS_NAMES = ("univariate", "plsda", "splsda", "rf")


# ============================================================
# Section helpers
# ============================================================
def _add_overview_section(
    rpt: PathReport,
    *,
    path_results: list[PathRunResult],
    consensus: ConsensusResult,
) -> None:
    rpt.add_section(
        name="overview",
        title="Multiverse summary",
        description=(
            "Each path through the multiverse runs the full discovery "
            "pipeline with one method choice swapped relative to the "
            "baseline. Consensus = features in ≥ threshold of paths' "
            "shortlists (per-path shortlist gate stays at the path's own "
            "k-of-N rule)."
        ),
    )
    succeeded = [r for r in path_results if r.succeeded]
    failed = [r for r in path_results if not r.succeeded]
    rpt.add_metric("overview", "n_paths_total", len(path_results))
    rpt.add_metric("overview", "n_paths_succeeded", len(succeeded))
    if failed:
        rpt.add_metric("overview", "n_paths_failed", len(failed))
    rpt.add_metric(
        "overview",
        "n_features",
        f"{int(consensus.table.shape[0]):,}",
        description="universe (union across paths)",
    )
    rpt.add_metric(
        "overview",
        "n_consensus",
        consensus.n_consensus,
        description=f"≥ {consensus.threshold:.0%} of paths agree",
    )
    rpt.add_metric("overview", "tier_1", consensus.n_per_tier.get("tier_1", 0))
    rpt.add_metric("overview", "tier_2", consensus.n_per_tier.get("tier_2", 0))
    rpt.add_metric("overview", "tier_3", consensus.n_per_tier.get("tier_3", 0))


def _add_paths_table_section(
    rpt: PathReport,
    *,
    path_results: list[PathRunResult],
    output_dir: Path,
) -> None:
    """One row per path with a hyperlink to that path's index.html."""
    rpt.add_section(
        name="paths",
        title="Paths in this multiverse",
        description=(
            "Each row is one full pipeline path. Click the link to open that "
            "path's full report (per-stage figures, multivariate panels, "
            "per-method shortlist parquet)."
        ),
    )
    rows: list[dict[str, object]] = []
    for r in path_results:
        # Hyperlink relative to the index.html we are about to write
        rel_path = Path(r.spec.name) / "index.html"
        href = f'<a href="{rel_path.as_posix()}">↗ {r.spec.name}</a>'
        rows.append(
            {
                "path": href,
                "imputation": r.spec.imputation,
                "drift": r.spec.drift,
                "normalisation": r.spec.normalisation,
                "transformation": r.spec.transformation,
                "scaling": r.spec.scaling,
                "runtime_min": (
                    f"{r.runtime_seconds / 60:.1f}" if r.succeeded else "FAIL"
                ),
                "n_shortlisted": (
                    int(r.shortlist.n_shortlisted) if (r.succeeded and r.shortlist) else "—"
                ),
                "status": "✓" if r.succeeded else "✗",
            }
        )
    df = pd.DataFrame(rows)
    rpt.add_table(
        "paths",
        title="Per-path summary",
        df=df,
        caption=(
            "Each path runs the same data through different method choices. "
            "The shortlist count varies because each method-axis swap "
            "perturbs the variance / correlation structure that the "
            "downstream selection sees."
        ),
        max_rows=50,
        escape=False,  # allow the <a href=…> hyperlink in the path column
    )


def _add_specification_curve_section(
    rpt: PathReport,
    *,
    path_results: list[PathRunResult],
) -> None:
    rpt.add_section(
        name="specification_curve",
        title="Specification curve",
        description=(
            "How many features each path called shortlisted, sorted "
            "ascending. Reading instability (Simonsohn et al. 2020): a "
            "wide spread = the answer is sensitive to method choice; a "
            "tight band = the methods agree on the size of the answer."
        ),
    )
    succeeded = [r for r in path_results if r.succeeded and r.shortlist is not None]
    if not succeeded:
        return
    names = [r.spec.name for r in succeeded]
    n_short = [int(r.shortlist.n_shortlisted) for r in succeeded]  # type: ignore[union-attr]
    fig = specification_curve_bar(path_names=names, n_shortlisted=n_short)
    rpt.add_figure(
        "specification_curve",
        title="Shortlist size per path",
        fig=fig,
        caption=(
            "Each bar = one path's per-path shortlist count (under the "
            "path's own k-of-N rule). Annotated count on top of each bar."
        ),
        method_metric="specification curve (Simonsohn 2020)",
    )


def _add_agreement_section(
    rpt: PathReport,
    *,
    path_results: list[PathRunResult],
) -> None:
    rpt.add_section(
        name="agreement",
        title="Cross-path agreement (Jaccard)",
        description=(
            "Pairwise overlap of per-path shortlists. J = |∩| / |∪|. "
            "Diagonal is always 1.0; off-diagonal close to 1.0 = paths "
            "agree on which features are biomarkers; close to 0 = each "
            "path is calling a different set."
        ),
    )
    succeeded = [r for r in path_results if r.succeeded and r.shortlist is not None]
    if not succeeded:
        return
    names = [r.spec.name for r in succeeded]
    sets: list[set[str]] = []
    for r in succeeded:
        tbl = r.shortlist.table  # type: ignore[union-attr]
        sets.append(set(tbl.loc[tbl["passed_shortlist"], "feature_id"].astype(str)))
    fig = jaccard_similarity_heatmap(path_names=names, feature_sets=sets)
    rpt.add_figure(
        "agreement",
        title="Pairwise Jaccard heatmap",
        fig=fig,
        caption=(
            "Heatmap of pairwise Jaccard similarity between path "
            "shortlists. Annotated values are J, range [0, 1]."
        ),
        method_metric="Jaccard similarity (per-path shortlists)",
    )


def _add_tier_breakdown_section(
    rpt: PathReport,
    *,
    consensus: ConsensusResult,
) -> None:
    rpt.add_section(
        name="tiers",
        title="Consensus tier breakdown",
        description=(
            "Tier_1: feature in ≥ 75 % of paths. Tier_2: 50–75 %. "
            "Tier_3: 1–50 %. Tier_1 + Tier_2 = passed multiverse "
            "consensus at threshold ≥ 50 %."
        ),
    )
    fig = consensus_tier_bar(n_per_tier=consensus.n_per_tier)
    rpt.add_figure(
        "tiers",
        title="Features per tier",
        fig=fig,
        caption="Bar lengths are absolute feature counts. Total = universe.",
        method_metric="multiverse consensus (Steegen 2016)",
    )


def _add_top_features_section(
    rpt: PathReport,
    *,
    consensus: ConsensusResult,
    top_n: int = 30,
) -> None:
    rpt.add_section(
        name="top_features",
        title=f"Top {top_n} consensus features",
        description=(
            "Sorted by paths-flagging fraction. n_paths_<axis> columns "
            "show how many paths flagged the feature via each method "
            "axis — useful to see whether the consensus is multi-method "
            "or driven by a single method."
        ),
    )
    df = consensus.table.copy()
    cols = [
        "feature_id", "n_paths_flagging", "frac_paths_flagging",
        "consensus_tier", "passed_consensus",
    ]
    drilldown_cols = [
        f"n_paths_{a}" for a in _METHOD_AXIS_NAMES
        if f"n_paths_{a}" in df.columns
    ]
    cols += drilldown_cols
    rpt.add_table(
        "top_features",
        title=f"Top {top_n} by consensus fraction",
        df=df[cols].head(top_n),
        max_rows=top_n,
        caption=(
            "Top features ranked by cross-path agreement. "
            "frac_paths_flagging ∈ [0, 1]; passed_consensus = "
            "frac ≥ threshold."
        ),
    )


def _add_method_drilldown_section(
    rpt: PathReport,
    *,
    path_results: list[PathRunResult],
    consensus: ConsensusResult,
    top_n: int = 30,
) -> None:
    rpt.add_section(
        name="method_drilldown",
        title=f"Method evidence drilldown — top {top_n} consensus features",
        description=(
            "Cell = 1 (coloured) when that method's gate passed for that "
            "feature in that path. Vertical lines separate paths. Whole "
            "horizontal bands of one method = the consensus is dominated "
            "by that method; mixed = multi-method agreement (the strong "
            "case)."
        ),
    )
    succeeded = [r for r in path_results if r.succeeded and r.shortlist is not None]
    if not succeeded:
        return
    top_features = consensus.table.head(top_n)["feature_id"].astype(str).tolist()
    if not top_features:
        return
    feature_idx = {fid: i for i, fid in enumerate(top_features)}
    n_feat = len(top_features)
    n_paths = len(succeeded)
    n_methods = len(_METHOD_AXIS_NAMES)
    cube = np.zeros((n_feat, n_paths, n_methods), dtype=int)
    for j, r in enumerate(succeeded):
        tbl = r.shortlist.table.set_index(  # type: ignore[union-attr]
            r.shortlist.table["feature_id"].astype(str)  # type: ignore[union-attr]
        )
        for k, axis in enumerate(_METHOD_AXIS_NAMES):
            col = f"{axis}_passed"
            if col not in tbl.columns:
                continue
            for fid in top_features:
                if fid in tbl.index:
                    val = bool(tbl.loc[fid, col])
                    if val:
                        cube[feature_idx[fid], j, k] = 1

    fig = method_drilldown_heatmap(
        feature_ids=top_features,
        path_names=[r.spec.name for r in succeeded],
        method_names=list(_METHOD_AXIS_NAMES),
        flag_cube=cube,
    )
    rpt.add_figure(
        "method_drilldown",
        title="Per-feature method evidence per path",
        fig=fig,
        caption=(
            "If you see horizontal stripes only on a few methods (e.g. "
            "univariate + plsda) and never on others (splsda, rf), the "
            "shortlist gate's k-of-N rule is in practice an AND of those "
            "two methods — re-document accordingly."
        ),
        method_metric="per-feature × path × method-axis evidence",
    )


def _add_methods_section(
    rpt: PathReport,
    *,
    path_results: list[PathRunResult],
    consensus: ConsensusResult,
    output_dir: Path,
) -> None:
    rpt.add_section(
        name="methods",
        title="Methods + reproducibility",
        description=(
            "Multiverse design: Steegen et al. 2016 (Perspect Psychol Sci 11:702). "
            "Specification curve: Simonsohn et al. 2020 (Nat Hum Behav 4:1208). "
            "Per-path shortlist (mQACC k-of-N): Westerhuis 2008, Saccenti 2014. "
            "All numbers on this page are reproducible from the per-path "
            "shortlist parquets and the consensus parquet (paths listed below)."
        ),
    )
    n_b: int | str = "?"
    n_p: int | str = "?"
    if path_results and path_results[0].succeeded:
        plsda_obj = path_results[0].multivariate.get("plsda")
        if plsda_obj is not None:
            n_b = plsda_obj.n_bootstrap
            n_p = plsda_obj.n_permutation
    rpt.add_metric("methods", "consensus_threshold", consensus.threshold)
    rpt.add_metric("methods", "n_bootstrap", n_b, description="per-path stability resamples")
    rpt.add_metric("methods", "n_permutation", n_p, description="per-path label-shuffle null")
    files_df = pd.DataFrame(
        {
            "file": [
                "consensus.parquet",
                "consensus.csv",
                "consensus_passed.parquet",
                "consensus_config.json",
                "<path_name>/shortlist.parquet",
                "<path_name>/index.html",
            ],
            "purpose": [
                "full per-feature consensus table (all 4644 features)",
                "same, human-readable",
                "passed_consensus=True subset only",
                "threshold, n_paths, path_names, n_per_tier, runtime",
                "Phase 9 per-path shortlist (per path subdir)",
                "full per-path HTML report",
            ],
        }
    )
    rpt.add_table(
        "methods",
        title="Output files (under " + str(output_dir) + ")",
        df=files_df,
        max_rows=20,
    )


# ============================================================
# Public entry point
# ============================================================
def build_multiverse_index(
    *,
    path_results: list[PathRunResult],
    consensus: ConsensusResult,
    output_dir: Path,
    title: str = "Multiverse biomarker discovery — overview",
    top_n_features: int = 30,
    write_png: bool = False,
) -> Path:
    """Render ``output_dir/index.html`` summarising the multiverse run.

    Parameters
    ----------
    path_results
        Output of :func:`biodisc.multiverse.orchestrator.run_multiverse`.
    consensus
        Output of :func:`biodisc.multiverse.consensus.compute_consensus`.
    output_dir
        Where to write ``index.html`` (typically
        ``data/reports/multiverse/``). Per-path subdirectories are
        expected to already exist with their own ``index.html``.
    title
        Page title (banner + browser tab).
    top_n_features
        How many top consensus features to render in the table + drilldown.
    write_png
        Forward to ``PathReport.write(export_png=...)``.

    Returns
    -------
    Path to the rendered ``index.html``.
    """
    output_dir = Path(output_dir)
    # Path chain shows the AXES (not method names) varied across paths,
    # auto-derived by checking which spec fields differ across paths.
    axes = _derive_varying_axes(path_results)

    rpt = PathReport(
        name="multiverse",
        output_dir=output_dir,
        title=title,
        path_chain=axes,
    )
    _add_overview_section(rpt, path_results=path_results, consensus=consensus)
    _add_paths_table_section(rpt, path_results=path_results, output_dir=output_dir)
    _add_specification_curve_section(rpt, path_results=path_results)
    _add_agreement_section(rpt, path_results=path_results)
    _add_tier_breakdown_section(rpt, consensus=consensus)
    _add_top_features_section(rpt, consensus=consensus, top_n=top_n_features)
    _add_method_drilldown_section(
        rpt, path_results=path_results, consensus=consensus, top_n=top_n_features,
    )
    _add_methods_section(
        rpt, path_results=path_results, consensus=consensus, output_dir=output_dir,
    )

    index_path = rpt.write(export_png=write_png, template_name="multiverse_index.html.j2")
    log.info(
        "multiverse_index_written",
        output=str(index_path),
        n_paths=len(path_results),
        n_consensus=consensus.n_consensus,
    )
    return index_path


def _derive_varying_axes(path_results: list[PathRunResult]) -> list[str]:
    """Identify which method-axis fields differ across paths (≥ 2 unique values)."""
    if not path_results:
        return []
    axes_to_check = (
        "imputation", "drift", "normalisation", "transformation", "scaling",
    )
    varying: list[str] = []
    for axis in axes_to_check:
        vals = {getattr(r.spec, axis) for r in path_results}
        if len(vals) >= 2:
            varying.append(f"{axis}∈{{{','.join(sorted(vals))}}}")
        else:
            varying.append(f"{axis}={next(iter(vals))}")
    return varying
