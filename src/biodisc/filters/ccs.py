"""Stage 8 — CCS error filter (adduct-aware, quality-floored).

Algorithm
---------
For each Stage-7-active feature with a measured CCS:

  1. Pull every Progenesis identification candidate.
  2. Each candidate may hypothesise multiple adducts (comma-separated in
     the Progenesis ``adducts`` column). Expand to one row per
     (candidate, adduct).
  3. Join the unified CCS reference on (hmdb_id, adduct) to get the
     experimental ``reference_ccs`` for that exact adduct.
  4. Apply TWIMS bias correction: when the reference instrument is
     DTIMS / TIMS, multiply ``reference_ccs`` by an instrument-specific
     factor so it's comparable to a Waters TWIMS measurement.
  5. Compute ``ccs_err_pct = |measured − corrected_reference| / corrected_reference × 100``.
  6. Apply the publication-grade quality floor: only count a candidate
     as a "rescuing" match if it ALSO satisfies
     ``Progenesis Score ≥ min_score`` and ``|mass_error_ppm| ≤ max_ppm``.
     This prevents low-confidence Progenesis matches from passing the
     filter on coincidental CCS agreement.
  7. Decision (the "any-candidate-passes" rule):
       - If ANY (candidate, adduct) row has all of:
             ccs_err_pct  ≤ max_ccs_err_pct
             score        ≥ min_score
             |ppm|        ≤ max_ppm
         → KEEP the feature.
       - Else if no (candidate, adduct) row had ANY reference at all →
         PASS-THROUGH (we cannot apply a check we can't compute).
       - Else → DROP.

Outputs (per feature):
  metrics.parquet        per-feature row with the BEST match details
                         (best_compound, best_adduct, best_ccs_err_pct,
                          n_candidates, n_evaluated, decision)
  kept_features.parquet  the kept feature_id list
  audit/08_ccs.parquet   one audit row per feature
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from biodisc.audit import AuditLogger
from biodisc.filters._output import write_outputs  # noqa: F401 — re-exported for CLI

log = structlog.get_logger(__name__)


# ============================================================
# Config + result dataclasses
# ============================================================
@dataclass(frozen=True)
class CCSFilterConfig:
    """Mirrors conf/filter/default.yaml:ccs."""

    max_ccs_err_pct: float = 2.0  # Paglia 2014 default; sweep-able
    min_progenesis_score: float = 30.0  # quality floor on rescuing candidate
    max_mass_error_ppm: float = 5.0  # MS₁ accuracy floor
    # Per-instrument bias correction for non-TWIMS reference values
    twims_bias_dtims: float = 1.015  # +1.5% (Hines 2017 typical)
    twims_bias_tims: float = 1.010  # +1.0% (Mendoza-Salcido 2022 typical)
    emit_csv_mirror: bool = True


@dataclass(frozen=True)
class CCSFilterResult:
    kept_features: pd.DataFrame
    metrics: pd.DataFrame
    n_input: int
    n_kept: int
    n_dropped: int
    n_passthrough_no_candidate: int
    n_passthrough_no_reference: int

    @property
    def n_filterable(self) -> int:
        return self.n_input - self.n_passthrough_no_candidate - self.n_passthrough_no_reference

    @property
    def drop_pct(self) -> float:
        return self.n_dropped / self.n_input * 100 if self.n_input else 0.0


# ============================================================
# Helpers
# ============================================================
def _split_progenesis_adducts(s: str | float | None) -> list[str]:
    """Progenesis emits comma-separated adducts: 'M+H, M+K, M+Na, …'."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    return [a.strip() for a in str(s).split(",") if a.strip()]


def _bias_factor(instrument_type: str, config: CCSFilterConfig) -> float:
    """Instrument-aware bias correction factor (multiply reference_ccs by this)."""
    if instrument_type == "DTIMS":
        return config.twims_bias_dtims
    if instrument_type == "TIMS":
        return config.twims_bias_tims
    # TWIMS reference → no correction
    return 1.0


# ============================================================
# Build the per-(feature, candidate, adduct) evaluation table
# ============================================================
def _build_evaluation_table(
    features: pd.DataFrame,  # feature_id, measured_ccs (active-only)
    identification: pd.DataFrame,  # feature_id, compound_id, adducts, score, mass_error_ppm
    reference: pd.DataFrame,  # hmdb_id, adduct, reference_ccs, instrument_type, …
    config: CCSFilterConfig,
) -> pd.DataFrame:
    """Returns one row per (feature, candidate, adduct, reference_match)."""
    # 1. Restrict identification to active features
    ident = identification[identification["feature_id"].isin(features["feature_id"])].copy()
    # 2. Explode adducts (comma-separated → one row per adduct)
    ident["adduct"] = ident["adducts"].apply(_split_progenesis_adducts)
    ident = ident.explode("adduct", ignore_index=True)
    ident = ident[ident["adduct"].notna() & (ident["adduct"] != "")].copy()

    # 3. Join with reference on (hmdb_id, adduct)
    #    LEFT JOIN: keep candidates that have no matching reference (so we can
    #    distinguish "no reference at all" from "reference exists but bad fit")
    ident = ident.merge(
        reference[
            [
                "hmdb_id",
                "adduct",
                "reference_ccs",
                "instrument_type",
                "compound_name",
                "source_db",
                "confidence_tier",
            ]
        ],
        left_on=["compound_id", "adduct"],
        right_on=["hmdb_id", "adduct"],
        how="left",
    )

    # 4. Attach measured_ccs from features
    ident = ident.merge(features[["feature_id", "measured_ccs"]], on="feature_id", how="left")

    # 5. Apply bias correction (vectorised)
    factors = ident["instrument_type"].map(
        lambda x: _bias_factor(x, config) if isinstance(x, str) else np.nan
    )
    ident["corrected_reference_ccs"] = ident["reference_ccs"] * factors

    # 6. Compute CCS error %
    ident["ccs_err_pct"] = (
        (ident["measured_ccs"] - ident["corrected_reference_ccs"]).abs()
        / ident["corrected_reference_ccs"]
        * 100
    )

    # 7. Apply quality floor — flag rows that are "valid candidates" for filtering
    quality_ok = (ident["score"] >= config.min_progenesis_score) & (
        ident["mass_error_ppm"].abs() <= config.max_mass_error_ppm
    )
    ident["passes_quality_floor"] = quality_ok
    ident["has_reference"] = ident["reference_ccs"].notna()
    ident["passes_ccs"] = (
        ident["has_reference"] & (ident["ccs_err_pct"] <= config.max_ccs_err_pct) & quality_ok
    )

    return ident


# ============================================================
# Per-feature decision + summary
# ============================================================
_DECISION_SCHEMA: list[str] = [
    "feature_id",
    "n_candidates",
    "n_evaluated_with_reference",
    "best_compound_id",
    "best_compound_name",
    "best_adduct",
    "best_score",
    "best_mass_error_ppm",
    "best_reference_ccs",
    "best_corrected_reference_ccs",
    "best_ccs_err_pct",
    "best_source_db",
    "best_instrument_type",
    "decision",
]


def _decide_per_feature(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-(feature, candidate, adduct) eval table to one row per feature."""
    if eval_df.empty:
        return pd.DataFrame(columns=_DECISION_SCHEMA)
    grouped = eval_df.groupby("feature_id", sort=False)

    rows = []
    for fid, group in grouped:
        n_candidates = group["compound_id"].nunique()
        n_with_ref = int(group["has_reference"].sum())

        # Best (smallest CCS error) row that ALSO passes quality floor
        scoreable = group[group["has_reference"]].copy()
        if scoreable.empty:
            decision = "passthrough_no_reference"
            best = {}
        else:
            quality_passing = scoreable[scoreable["passes_quality_floor"]]
            if quality_passing.empty:
                # No candidate cleared the quality floor; report best by CCS but mark dropped
                best_row = scoreable.loc[scoreable["ccs_err_pct"].idxmin()]
                decision = "dropped_quality_floor"
                best = best_row.to_dict()
            else:
                best_row = quality_passing.loc[quality_passing["ccs_err_pct"].idxmin()]
                decision = "kept" if best_row["passes_ccs"] else "dropped_ccs_err"
                best = best_row.to_dict()

        rows.append(
            {
                "feature_id": fid,
                "n_candidates": n_candidates,
                "n_evaluated_with_reference": n_with_ref,
                "best_compound_id": best.get("compound_id"),
                "best_compound_name": best.get("compound_name"),
                "best_adduct": best.get("adduct"),
                "best_score": best.get("score"),
                "best_mass_error_ppm": best.get("mass_error_ppm"),
                "best_reference_ccs": best.get("reference_ccs"),
                "best_corrected_reference_ccs": best.get("corrected_reference_ccs"),
                "best_ccs_err_pct": best.get("ccs_err_pct"),
                "best_source_db": best.get("source_db"),
                "best_instrument_type": best.get("instrument_type"),
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# Public entry point
# ============================================================
def run(
    *,
    features: pd.DataFrame,  # feature_id, measured_ccs
    identification: pd.DataFrame,  # Progenesis candidates
    reference: pd.DataFrame,  # unified CCS reference
    kept_features: Iterable[str],
    config: CCSFilterConfig,
    audit: AuditLogger,
) -> CCSFilterResult:
    """Apply Stage 8 filter and produce metrics + audit + kept list."""
    kf = set(kept_features)
    feats_active = features[features["feature_id"].isin(kf)].copy()
    if feats_active.empty:
        raise ValueError("No active features after restriction to kept_features")

    # Features without a measured_ccs cannot be evaluated → pass-through
    no_ccs_mask = feats_active["measured_ccs"].isna()
    if no_ccs_mask.any():
        log.warning("ccs_features_without_measured_ccs", n=int(no_ccs_mask.sum()))

    # Features without ANY Progenesis candidate cannot be evaluated → pass-through
    feats_with_candidate_ids = set(identification["feature_id"]) & kf

    # Build per-(feature, candidate, adduct) evaluation table for feats with candidates
    feats_filterable = feats_active[
        feats_active["measured_ccs"].notna()
        & feats_active["feature_id"].isin(feats_with_candidate_ids)
    ]
    eval_df = _build_evaluation_table(feats_filterable, identification, reference, config)
    per_feature = _decide_per_feature(eval_df)

    # Build the full per-feature metrics dataframe (covers all active features,
    # incl. pass-throughs)
    all_features = feats_active[["feature_id", "measured_ccs"]].copy()
    metrics = all_features.merge(per_feature, on="feature_id", how="left")

    # Tag pass-throughs that never made it into the eval table
    no_candidate = ~metrics["feature_id"].isin(feats_with_candidate_ids)
    decision_missing = metrics["decision"].isna()
    metrics.loc[no_candidate & decision_missing, "decision"] = "passthrough_no_candidate"
    # Anything still NaN (shouldn't happen) becomes pass-through-no-reference
    metrics.loc[metrics["decision"].isna(), "decision"] = "passthrough_no_reference"

    metrics["s8_filterable"] = metrics["decision"].isin(
        ["kept", "dropped_ccs_err", "dropped_quality_floor"]
    )
    metrics["kept"] = metrics["decision"].isin(
        ["kept", "passthrough_no_candidate", "passthrough_no_reference"]
    )
    # Headline metric column for the master ledger (StageSpec("08_ccs", ..., "ccs_err_pct"))
    metrics["ccs_err_pct"] = metrics["best_ccs_err_pct"]

    # Counts
    n_input = len(metrics)
    n_kept = int(metrics["kept"].sum())
    n_dropped = int((~metrics["kept"]).sum())
    n_pt_no_cand = int((metrics["decision"] == "passthrough_no_candidate").sum())
    n_pt_no_ref = int((metrics["decision"] == "passthrough_no_reference").sum())

    # Audit rows
    actions: list[str] = []
    reasons: list[str] = []
    for _, r in metrics.iterrows():
        if r["decision"] == "kept":
            actions.append("kept")
            reasons.append(
                f"ccs_err={r['best_ccs_err_pct']:.2f}%_le_{config.max_ccs_err_pct}%"
                f"_via_{r['best_compound_id']}_{r['best_adduct']}_{r['best_source_db']}"
            )
        elif r["decision"] == "passthrough_no_candidate":
            actions.append("kept")
            reasons.append("passthrough_no_progenesis_candidate")
        elif r["decision"] == "passthrough_no_reference":
            actions.append("kept")
            reasons.append("passthrough_no_external_ccs_reference")
        elif r["decision"] == "dropped_ccs_err":
            actions.append("dropped")
            reasons.append(
                f"ccs_err={r['best_ccs_err_pct']:.2f}%_gt_{config.max_ccs_err_pct}%"
                f"_via_{r['best_compound_id']}_{r['best_adduct']}"
            )
        elif r["decision"] == "dropped_quality_floor":
            actions.append("dropped")
            reasons.append("no_candidate_passed_score_or_ppm_quality_floor")
        else:
            actions.append("dropped")
            reasons.append(f"unknown_decision_{r['decision']}")

    audit.record_bulk(
        feature_ids=metrics["feature_id"].tolist(),
        actions=actions,
        reasons=reasons,
        metrics=metrics["best_ccs_err_pct"].tolist(),
        thresholds=[config.max_ccs_err_pct] * len(metrics),
    )

    log.info(
        "ccs_filter_result",
        n_input=n_input,
        n_kept=n_kept,
        n_dropped=n_dropped,
        n_passthrough_no_candidate=n_pt_no_cand,
        n_passthrough_no_reference=n_pt_no_ref,
        threshold=config.max_ccs_err_pct,
    )

    kept_df = pd.DataFrame({"feature_id": metrics.loc[metrics["kept"], "feature_id"].tolist()})

    return CCSFilterResult(
        kept_features=kept_df,
        metrics=metrics,
        n_input=n_input,
        n_kept=n_kept,
        n_dropped=n_dropped,
        n_passthrough_no_candidate=n_pt_no_cand,
        n_passthrough_no_reference=n_pt_no_ref,
    )



