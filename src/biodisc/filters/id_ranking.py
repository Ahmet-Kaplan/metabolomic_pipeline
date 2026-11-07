"""Stage 9 — Identification candidate ranking (annotation-only, no drops).

Algorithm
---------
For each Stage-8-active feature with at least one Progenesis candidate:

  Step 0. Quality floor — drop candidates where Score < min_score
          OR |mass_error_ppm| > max_mass_error_ppm. These are noise-quality
          matches that shouldn't compete in the cascade (Progenesis user
          guide + Broadhurst 2018 mQACC).

  Step 1. Score tiebreak — among Step-0 survivors, find the maximum Score.
          Keep candidates whose Score == max (rounded to 2 decimals).

  Step 2. Fragmentation tiebreak — among Step-1 survivors, if at least one
          has a Fragmentation Score, find the maximum Frag Score and keep
          only those whose Frag Score == max. If none have Frag Score,
          skip this step (no MS/MS evidence available).

  Step 3. Mass-error tiebreak — among Step-2 survivors, find the minimum
          |mass_error_ppm|. Keep those whose |ppm| == min.

  Whatever survives Step 3 is the final candidate set:
    - 1 candidate → MSI L2 (single putative annotation)
    - >1 candidate → MSI L3 (compound class — biologist decides)
    - 0 candidates → MSI L4 (none passed quality floor)

Features with no candidate from the start → MSI L4 ("no_candidate").
ALL FEATURES PASS THROUGH — this stage annotates, it does not drop.

Outputs (per feature):
  identification_ranked.parquet  trimmed identification long table
  metrics.parquet                per-feature compound summary
  kept_features.parquet          == Stage 8 kept (pass-through)
  audit/09_id_ranking.parquet    one audit row per feature
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
import structlog

from biodisc.audit import AuditLogger
from biodisc.filters._output import write_outputs  # noqa: F401 — re-exported for CLI

log = structlog.get_logger(__name__)


# ============================================================
# Config + result dataclasses
# ============================================================
@dataclass(frozen=True)
class IdRankingConfig:
    """Mirrors conf/filter/default.yaml:id_ranking."""

    min_score: float = 30.0  # Progenesis composite Score floor (vendor default)
    max_mass_error_ppm: float = 5.0  # MS₁ accuracy floor (Q-TOF capability)
    emit_csv_mirror: bool = True


@dataclass(frozen=True)
class IdRankingResult:
    ranked_identification: pd.DataFrame
    metrics: pd.DataFrame
    kept_features: pd.DataFrame
    n_input: int
    n_with_candidates: int
    n_msi_l2: int
    n_msi_l3: int
    n_msi_l4_no_candidate: int
    n_msi_l4_no_floor_survivor: int
    n_resolved_single_after_floor: int
    n_resolved_at_score: int
    n_resolved_at_frag: int
    n_resolved_at_ppm: int
    n_still_tied: int


# ============================================================
# Per-compound cascade
# ============================================================
def _rank_one_compound(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Apply the Score → Frag → Mass-error cascade to one feature's candidates.

    Input is the post-quality-floor candidate set. Returns the surviving
    rows plus a string label indicating where the cascade resolved.
    """
    if len(candidates) == 0:
        return candidates, "no_floor_survivor"
    if len(candidates) == 1:
        return candidates, "single_after_floor"

    # Step 1 — Score tiebreak
    scores = candidates["score"].round(2)
    max_score = scores.max()
    step1 = candidates[scores == max_score]
    if len(step1) == 1:
        return step1, "score"

    # Step 2 — Fragmentation tiebreak (only if ≥1 candidate has Frag Score)
    frag_present = step1["fragmentation_score"].notna()
    if frag_present.any():
        with_frag = step1[frag_present]
        frag_rounded = with_frag["fragmentation_score"].round(2)
        max_frag = frag_rounded.max()
        step2 = with_frag[frag_rounded == max_frag]
        if len(step2) == 1:
            return step2, "frag"
    else:
        step2 = step1

    # Step 3 — Mass-error tiebreak
    abs_ppm = step2["mass_error_ppm"].abs().round(3)
    min_ppm = abs_ppm.min()
    step3 = step2[abs_ppm == min_ppm]
    if len(step3) == 1:
        return step3, "ppm"

    return step3, "tied"


# ============================================================
# Per-feature summary row
# ============================================================
def _build_summary_row(
    feature_id: str,
    n_initial: int,
    n_after_floor: int,
    final_candidates: pd.DataFrame,
    resolved_at: str,
    had_msms: bool,
) -> dict[str, object]:
    """Build one summary row for a feature's annotation outcome."""
    n_final = len(final_candidates)

    if n_initial == 0:
        msi_tier = 4
        outcome = "l4_no_candidate"
    elif n_after_floor == 0:
        msi_tier = 4
        outcome = "l4_no_floor_survivor"
    elif n_final == 1:
        msi_tier = 2
        outcome = "l2_unique"
    else:
        msi_tier = 3
        outcome = "l3_ambiguous"

    if n_final >= 1:
        top = final_candidates.iloc[0]
        top_compound_id = top.get("compound_id")
        top_compound_name = top.get("description")
        top_formula = top.get("formula")
        top_score = top.get("score")
        top_frag_score = top.get("fragmentation_score")
        top_ppm = top.get("mass_error_ppm")
    else:
        top_compound_id = None
        top_compound_name = None
        top_formula = None
        top_score = None
        top_frag_score = None
        top_ppm = None

    return {
        "feature_id": feature_id,
        "n_candidates_initial": n_initial,
        "n_candidates_after_floor": n_after_floor,
        "n_candidates_final": n_final,
        "resolved_at": resolved_at,
        "msi_tier": msi_tier,
        "outcome": outcome,
        "had_msms": had_msms,
        "top_compound_id": top_compound_id,
        "top_compound_name": top_compound_name,
        "top_formula": top_formula,
        "top_score": top_score,
        "top_frag_score": top_frag_score,
        "top_ppm": top_ppm,
        "kept": True,
    }


# ============================================================
# Public entry point
# ============================================================
def run(
    *,
    identification: pd.DataFrame,
    kept_features: Iterable[str],
    config: IdRankingConfig,
    audit: AuditLogger,
) -> IdRankingResult:
    """Apply Stage 9 cascade and produce ranked identification + summary + audit."""
    kf = list(kept_features)
    kf_set = set(kf)

    # 1. Restrict identification to active features
    ident_active = identification[identification["feature_id"].isin(kf_set)].copy()

    # 2. Apply quality floor (Step 0)
    score_ok = ident_active["score"].fillna(-1) >= config.min_score
    ppm_ok = ident_active["mass_error_ppm"].abs().fillna(float("inf")) <= config.max_mass_error_ppm
    quality_ok = score_ok & ppm_ok
    ident_floor = ident_active[quality_ok].copy()

    log.info(
        "id_ranking_quality_floor",
        candidates_in=len(ident_active),
        candidates_after_floor=len(ident_floor),
        dropped_floor=len(ident_active) - len(ident_floor),
    )

    # 3. Run cascade per feature
    ranked_chunks: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    grouped_active = ident_active.groupby("feature_id", sort=False)
    grouped_floor = dict(list(ident_floor.groupby("feature_id", sort=False)))

    for fid, group_active in grouped_active:
        n_initial = len(group_active)
        had_msms = bool(group_active["fragmentation_score"].notna().any())

        floor_group = grouped_floor.get(fid)
        if floor_group is None:
            n_after_floor = 0
            summaries.append(
                _build_summary_row(
                    str(fid),
                    n_initial,
                    0,
                    ident_active.iloc[0:0],
                    "no_floor_survivor",
                    had_msms,
                )
            )
            continue

        n_after_floor = len(floor_group)
        final_candidates, resolved_at = _rank_one_compound(floor_group)
        if len(final_candidates) > 0:
            ranked_chunks.append(final_candidates)
        summaries.append(
            _build_summary_row(
                str(fid),
                n_initial,
                n_after_floor,
                final_candidates,
                resolved_at,
                had_msms,
            )
        )

    # 4. Features with no candidate at all → auto-L4
    feats_with_candidates = set(ident_active["feature_id"].unique())
    feats_no_candidate = kf_set - feats_with_candidates
    for fid in feats_no_candidate:
        summaries.append(
            _build_summary_row(
                str(fid),
                0,
                0,
                ident_active.iloc[0:0],
                "no_candidate",
                False,
            )
        )

    metrics_df = pd.DataFrame(summaries)
    if ranked_chunks:
        ranked_df = pd.concat(ranked_chunks, ignore_index=True)
    else:
        ranked_df = ident_active.iloc[0:0].copy()

    # 5. Pass-through kept list (Stage 9 doesn't drop features)
    kept_df = pd.DataFrame({"feature_id": kf})

    # 6. Counts
    n_input = len(kf)
    n_with_cand = int((metrics_df["n_candidates_initial"] > 0).sum()) if not metrics_df.empty else 0
    n_l2 = int((metrics_df["msi_tier"] == 2).sum()) if not metrics_df.empty else 0
    n_l3 = int((metrics_df["msi_tier"] == 3).sum()) if not metrics_df.empty else 0
    n_l4_nocand = (
        int((metrics_df["outcome"] == "l4_no_candidate").sum()) if not metrics_df.empty else 0
    )
    n_l4_nofloor = (
        int((metrics_df["outcome"] == "l4_no_floor_survivor").sum())
        if not metrics_df.empty
        else 0
    )
    n_single = (
        int((metrics_df["resolved_at"] == "single_after_floor").sum())
        if not metrics_df.empty
        else 0
    )
    n_score = int((metrics_df["resolved_at"] == "score").sum()) if not metrics_df.empty else 0
    n_frag = int((metrics_df["resolved_at"] == "frag").sum()) if not metrics_df.empty else 0
    n_ppm = int((metrics_df["resolved_at"] == "ppm").sum()) if not metrics_df.empty else 0
    n_tied = int((metrics_df["resolved_at"] == "tied").sum()) if not metrics_df.empty else 0

    # 7. Audit rows (one per feature, all "kept" — pass-through stage)
    actions: list[str] = []
    reasons: list[str] = []
    if not metrics_df.empty:
        for _, r in metrics_df.iterrows():
            actions.append("kept")
            outcome = r["outcome"]
            n_final = r["n_candidates_final"]
            resolved = r["resolved_at"]
            if outcome == "l2_unique":
                reasons.append(
                    f"l2_unique_via_{r['top_compound_id']}_resolved_at_{resolved}"
                )
            elif outcome == "l3_ambiguous":
                reasons.append(f"l3_ambiguous_n_final={n_final}_resolved_at_{resolved}")
            elif outcome == "l4_no_floor_survivor":
                reasons.append(
                    f"l4_no_candidate_passed_floor_score{config.min_score:g}"
                    f"_ppm{config.max_mass_error_ppm:g}"
                )
            else:  # l4_no_candidate
                reasons.append("l4_no_progenesis_candidate")

        audit.record_bulk(
            feature_ids=metrics_df["feature_id"].tolist(),
            actions=actions,
            reasons=reasons,
            metrics=metrics_df["n_candidates_final"].tolist(),
            thresholds=[config.min_score] * len(metrics_df),
        )

    log.info(
        "id_ranking_result",
        n_input=n_input,
        n_with_candidates=n_with_cand,
        n_l2=n_l2,
        n_l3=n_l3,
        n_l4_no_candidate=n_l4_nocand,
        n_l4_no_floor_survivor=n_l4_nofloor,
        single_after_floor=n_single,
        resolved_score=n_score,
        resolved_frag=n_frag,
        resolved_ppm=n_ppm,
        still_tied=n_tied,
    )

    return IdRankingResult(
        ranked_identification=ranked_df,
        metrics=metrics_df,
        kept_features=kept_df,
        n_input=n_input,
        n_with_candidates=n_with_cand,
        n_msi_l2=n_l2,
        n_msi_l3=n_l3,
        n_msi_l4_no_candidate=n_l4_nocand,
        n_msi_l4_no_floor_survivor=n_l4_nofloor,
        n_resolved_single_after_floor=n_single,
        n_resolved_at_score=n_score,
        n_resolved_at_frag=n_frag,
        n_resolved_at_ppm=n_ppm,
        n_still_tied=n_tied,
    )



