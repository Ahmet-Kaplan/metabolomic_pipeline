"""Stage 10 — Origin filter (annotation-driven).

Drops features whose ALL surviving cascade candidates are pure-exogenous
(no Endogenous tag in HMDB ontology). Cell-line metabolomics-specific:
food, drug, plasticizer, FBS, microbial signals are contamination, not
biology of interest.

Algorithm
---------
For each Stage-9-active feature:

  1. If feature is L4 (no_candidate or no_floor_survivor) → KEEP
     (we cannot classify origin without an annotation; falls under the
     Bijlsma 2006 untargeted-discovery principle of not dropping based
     on a check we cannot apply).

  2. Otherwise, look up the origin classification of every surviving
     cascade candidate (from identification_ranked.parquet) in the
     HMDB origin table.

  3. Aggregate per feature:
       - If ANY surviving candidate has is_endogenous == True
         (i.e. origin_class ∈ {endogenous_only, both})
         → outcome = "endogenous_evidence", KEEP.
       - Else if EVERY surviving candidate has origin_class == "exogenous_only"
         → outcome = "all_exogenous", DROP.
       - Else (mix of "exogenous_only" + "unannotated", no Endo evidence)
         → outcome = "exo_or_unknown", KEEP (conservative — at least one
         candidate had no annotation, so the feature is not provably exo).

Outputs:
  metrics.parquet        per-feature origin classification + decision
  kept_features.parquet  feature_id list of survivors
  audit/10_origin.parquet
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
class OriginFilterConfig:
    """Mirrors conf/filter/default.yaml:origin."""

    drop_pure_exogenous: bool = True  # if False → soft-tag mode (no drops)
    emit_csv_mirror: bool = True


@dataclass(frozen=True)
class OriginFilterResult:
    metrics: pd.DataFrame
    kept_features: pd.DataFrame
    n_input: int
    n_kept: int
    n_dropped: int
    n_endogenous_evidence: int
    n_exo_or_unknown: int
    n_passthrough_l4: int
    n_all_exogenous: int

    @property
    def drop_pct(self) -> float:
        return self.n_dropped / self.n_input * 100 if self.n_input else 0.0


# ============================================================
# Per-feature aggregation
# ============================================================
def _aggregate_origin(
    candidate_origins: pd.DataFrame,
) -> tuple[str, int, int, int, int, list[str]]:
    """Aggregate origin classes across a feature's surviving candidates.

    Returns (outcome, n_endo, n_both, n_exo, n_unannotated, sample_top_compounds).
    Outcome is one of: "endogenous_evidence", "all_exogenous", "exo_or_unknown".
    """
    classes = candidate_origins["origin_class"].fillna("unannotated")
    n_endo = int((classes == "endogenous_only").sum())
    n_both = int((classes == "both").sum())
    n_exo = int((classes == "exogenous_only").sum())
    n_unann = int((classes == "unannotated").sum())

    if n_endo + n_both > 0:
        outcome = "endogenous_evidence"
    elif n_exo > 0 and n_unann == 0:
        outcome = "all_exogenous"
    else:
        outcome = "exo_or_unknown"

    return outcome, n_endo, n_both, n_exo, n_unann, []


def _build_summary_row(
    *,
    feature_id: str,
    msi_tier: int,
    is_l4: bool,
    n_cands_total: int,
    candidate_origins: pd.DataFrame,  # may be empty for L4
    drop_pure_exogenous: bool,
) -> dict[str, object]:
    """Build one summary row for a feature's origin outcome."""
    if is_l4:
        return {
            "feature_id": feature_id,
            "msi_tier": msi_tier,
            "n_candidates_evaluated": 0,
            "n_endogenous_only": 0,
            "n_both": 0,
            "n_exogenous_only": 0,
            "n_unannotated": 0,
            "outcome": "passthrough_l4",
            "kept": True,
        }

    outcome, n_endo, n_both, n_exo, n_unann, _ = _aggregate_origin(candidate_origins)
    kept = not (outcome == "all_exogenous" and drop_pure_exogenous)
    return {
        "feature_id": feature_id,
        "msi_tier": msi_tier,
        "n_candidates_evaluated": n_cands_total,
        "n_endogenous_only": n_endo,
        "n_both": n_both,
        "n_exogenous_only": n_exo,
        "n_unannotated": n_unann,
        "outcome": outcome,
        "kept": kept,
    }


# ============================================================
# Public entry point
# ============================================================
def run(
    *,
    metrics_in: pd.DataFrame,  # Stage 9 metrics.parquet
    ranked_identification: pd.DataFrame,  # Stage 9 identification_ranked.parquet
    hmdb_origin: pd.DataFrame,  # external/hmdb_origin.parquet
    kept_features: Iterable[str],  # Stage 9 kept_features
    config: OriginFilterConfig,
    audit: AuditLogger,
) -> OriginFilterResult:
    """Apply Stage 10 origin filter and produce metrics + audit + kept list."""
    kf = list(kept_features)
    kf_set = set(kf)

    # 1. Restrict Stage-9 metrics to active features
    m_active = metrics_in[metrics_in["feature_id"].isin(kf_set)].copy()
    if len(m_active) != len(kf_set):
        log.warning(
            "origin_metrics_active_mismatch",
            metrics_rows=len(m_active),
            kept_rows=len(kf_set),
        )

    # 2. Join surviving candidates with origin table
    ranked = ranked_identification[
        ranked_identification["feature_id"].isin(kf_set)
    ][["feature_id", "compound_id"]].copy()
    ranked = ranked.merge(
        hmdb_origin[["hmdb_id", "origin_class", "is_endogenous"]],
        left_on="compound_id",
        right_on="hmdb_id",
        how="left",
    )
    ranked["origin_class"] = ranked["origin_class"].fillna("unannotated")

    # 3. Per-feature aggregation
    summaries: list[dict[str, object]] = []
    grouped = dict(list(ranked.groupby("feature_id", sort=False)))

    for _, row in m_active.iterrows():
        fid = str(row["feature_id"])
        msi_tier = int(row["msi_tier"])
        outcome_in = row["outcome"]  # from Stage 9
        is_l4 = outcome_in in ("l4_no_candidate", "l4_no_floor_survivor")

        if is_l4:
            summaries.append(
                _build_summary_row(
                    feature_id=fid,
                    msi_tier=msi_tier,
                    is_l4=True,
                    n_cands_total=0,
                    candidate_origins=ranked.iloc[0:0],
                    drop_pure_exogenous=config.drop_pure_exogenous,
                )
            )
            continue

        cand_origins = grouped.get(fid)
        if cand_origins is None or cand_origins.empty:
            # Feature is non-L4 in Stage 9 but has no rows in identification_ranked —
            # shouldn't happen, but treat as passthrough_l4 for safety.
            summaries.append(
                _build_summary_row(
                    feature_id=fid,
                    msi_tier=msi_tier,
                    is_l4=True,
                    n_cands_total=0,
                    candidate_origins=ranked.iloc[0:0],
                    drop_pure_exogenous=config.drop_pure_exogenous,
                )
            )
            continue

        summaries.append(
            _build_summary_row(
                feature_id=fid,
                msi_tier=msi_tier,
                is_l4=False,
                n_cands_total=len(cand_origins),
                candidate_origins=cand_origins,
                drop_pure_exogenous=config.drop_pure_exogenous,
            )
        )

    metrics = pd.DataFrame(summaries)

    # 4. Counts
    n_input = len(metrics)
    n_kept = int(metrics["kept"].sum())
    n_dropped = n_input - n_kept
    n_endo_evidence = int((metrics["outcome"] == "endogenous_evidence").sum())
    n_exo_unknown = int((metrics["outcome"] == "exo_or_unknown").sum())
    n_pt_l4 = int((metrics["outcome"] == "passthrough_l4").sum())
    n_all_exo = int((metrics["outcome"] == "all_exogenous").sum())

    # 5. Audit
    actions: list[str] = []
    reasons: list[str] = []
    for _, r in metrics.iterrows():
        outcome = r["outcome"]
        if r["kept"]:
            actions.append("kept")
            if outcome == "passthrough_l4":
                reasons.append("passthrough_l4_unclassifiable")
            elif outcome == "endogenous_evidence":
                reasons.append(
                    f"endogenous_evidence_n_endo={r['n_endogenous_only']}_n_both={r['n_both']}"
                )
            elif outcome == "exo_or_unknown":
                reasons.append(
                    f"exo_or_unknown_n_exo={r['n_exogenous_only']}_n_unann={r['n_unannotated']}"
                )
            elif outcome == "all_exogenous":
                # only happens when drop_pure_exogenous=False (soft mode)
                reasons.append(
                    f"all_exogenous_soft_kept_n_exo={r['n_exogenous_only']}"
                )
            else:
                reasons.append(f"kept_outcome_{outcome}")
        else:
            actions.append("dropped")
            reasons.append(f"all_candidates_pure_exogenous_n_exo={r['n_exogenous_only']}")

    audit.record_bulk(
        feature_ids=metrics["feature_id"].tolist(),
        actions=actions,
        reasons=reasons,
        metrics=metrics["n_exogenous_only"].astype(float).tolist(),
        thresholds=[None] * len(metrics),
    )

    log.info(
        "origin_filter_result",
        n_input=n_input,
        n_kept=n_kept,
        n_dropped=n_dropped,
        n_endogenous_evidence=n_endo_evidence,
        n_exo_or_unknown=n_exo_unknown,
        n_passthrough_l4=n_pt_l4,
        n_all_exogenous=n_all_exo,
        drop_mode="hard" if config.drop_pure_exogenous else "soft_tag",
    )

    kept_df = pd.DataFrame({"feature_id": metrics.loc[metrics["kept"], "feature_id"].tolist()})

    return OriginFilterResult(
        metrics=metrics,
        kept_features=kept_df,
        n_input=n_input,
        n_kept=n_kept,
        n_dropped=n_dropped,
        n_endogenous_evidence=n_endo_evidence,
        n_exo_or_unknown=n_exo_unknown,
        n_passthrough_l4=n_pt_l4,
        n_all_exogenous=n_all_exo,
    )
