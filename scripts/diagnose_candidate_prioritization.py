"""Candidate prioritization score — rank consensus features for validation.

Combines statistical robustness, annotation quality, KEGG pathway
evidence, and chemical feasibility into a single prioritization score
so the biologist knows which candidates to validate first.

Score components (each 0-1, weighted equally)
---------------------------------------------
1. **Statistical robustness**: (paths_flagging / total_paths + mean_eta_sq +
   mean_methods_passed) / 3
2. **Annotation quality**: MSI tier weight (L2=1.0, L3=0.6, L4=0.2) ×
   has_compound_name (0.5 if only tier, 1.0 if has name)
3. **KEGG evidence**: has_kegg (1 if yes, 0 if no) × normalized pathways
4. **Biological plausibility**: endogenous origin + known cancer lipid class

Output: docs/evaluation/candidate_prioritization.md
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

INTERIM = Path("data/interim")
CONSENSUS = Path("data/reports/multiverse")
REPORT = Path("docs/evaluation/candidate_prioritization.md")

# Weights for each score component (must sum to 1.0)
W_STATISTICAL = 0.30
W_ANNOTATION = 0.30
W_KEGG = 0.25
W_BIOLOGICAL = 0.15


def _load_consensus() -> pd.DataFrame:
    path = CONSENSUS / "consensus_passed.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _load_per_path_shortlists() -> dict[str, pd.DataFrame]:
    """Load per-path shortlist tables for eta-squared and method details."""
    result: dict[str, pd.DataFrame] = {}
    for name in ["path_a_classical", "path_b_robust_ml", "path_c_mnar_aware", "path_d_middle"]:
        path = CONSENSUS / name / "shortlist.parquet"
        if path.exists():
            result[name] = pd.read_parquet(path)
    return result


def _load_stage9() -> pd.DataFrame:
    path = INTERIM / "09_id_ranking" / "metrics.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    needed = {"feature_id", "top_compound_id", "top_compound_name",
              "msi_tier", "outcome", "top_formula", "n_candidates_final",
              "resolved_at"}
    keep = [c for c in needed if c in df.columns]
    return df[keep].copy()


def _load_enrichment() -> pd.DataFrame:
    path = CONSENSUS / "kegg_enrichment.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _load_features() -> pd.DataFrame:
    path = INTERIM / "01_parsed" / "features.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _compute_prioritization(
    consensus: pd.DataFrame,
    path_shortlists: dict[str, pd.DataFrame],
    stage9: pd.DataFrame,
    enrichment: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-feature prioritization score."""
    if consensus.empty:
        return pd.DataFrame()

    # Load HMDB origin for endogenous/xenobiotic classification
    origin_map: dict = {}
    origin_path = Path("data/external/hmdb_origin.parquet")
    if origin_path.exists():
        odf = pd.read_parquet(origin_path)
        for _, r in odf.iterrows():
            origin_map[r["hmdb_id"]] = {
                "origin_class": r.get("origin_class", "unannotated"),
                "is_endogenous": r.get("is_endogenous", False),
            }

    n_paths_total = max(consensus["n_paths_total"].max(), 1)
    scoring_rows: list[dict] = []

    for _, feat in consensus.iterrows():
        fid = feat["feature_id"]
        n_paths = int(feat.get("n_paths_flagging", 0))
        consensus_tier = feat.get("consensus_tier", "none")

        # 1. Statistical robustness
        eta_sq_vals: list[float] = []
        methods_passed: list[int] = []
        for _name, sl_tbl in path_shortlists.items():
            if fid in sl_tbl["feature_id"].values:
                row = sl_tbl[sl_tbl["feature_id"] == fid]
                for c in row.columns:
                    if "eta_squared" in c:
                        val = row[c].values[0]
                        if not (isinstance(val, float) and np.isnan(val)):
                            eta_sq_vals.append(float(val))
                if "n_methods_passed" in row.columns:
                    methods_passed.append(int(row["n_methods_passed"].values[0]))
        stat_score = (
            0.4 * (n_paths / n_paths_total)
            + 0.3 * (np.mean(eta_sq_vals) if eta_sq_vals else 0)
            + 0.3 * (np.mean(methods_passed) / 4 if methods_passed else 0)
        )

        # 2. Annotation quality
        ann = {}
        if not stage9.empty and fid in stage9["feature_id"].values:
            s9 = stage9[stage9["feature_id"] == fid].iloc[0]
            ann["compound_id"] = s9.get("top_compound_id")
            ann["compound_name"] = s9.get("top_compound_name")
            ann["msi_tier"] = int(s9["msi_tier"]) if not isinstance(s9["msi_tier"], str) else (
                4 if "4" in str(s9["msi_tier"]) else 3 if "3" in str(s9["msi_tier"]) else 2
            )
            ann["has_name"] = bool(ann.get("compound_name")) and str(ann["compound_name"]).strip() not in ("", "nan", "None")
            ann["formula"] = s9.get("top_formula")
        else:
            ann = {"compound_id": None, "compound_name": None, "msi_tier": 4, "has_name": False, "formula": None}

        msi_weight = {2: 1.0, 3: 0.6, 4: 0.2}.get(ann["msi_tier"], 0)
        name_bonus = 1.0 if ann.get("has_name") else 0.5
        annotation_score = msi_weight * name_bonus

        # 3. KEGG pathway evidence
        kegg_score = 0.0
        kegg_hit = False
        if not enrichment.empty and not stage9.empty:
            compound_id = ann.get("compound_id")
            if compound_id:
                kegg_rows = enrichment[enrichment["foreground_kegg_ids"].str.contains(
                    compound_id, na=False, regex=False
                )]
                if not kegg_rows.empty:
                    kegg_hit = True
                    min_q = kegg_rows["q_value"].min()
                    kegg_score = 0.5 + 0.5 * max(0, 1 - min_q / 0.05)

        # 4. Biological plausibility (with xenobiotic penalty)
        bio_score = 0.0
        compound_id = ann.get("compound_id")
        formula = ann.get("formula", "")
        compound_name_str = str(ann.get("compound_name", ""))
        compound_id_str = str(compound_id) if compound_id else ""

        # Xenobiotic detection: halogen atoms (Cl, F, Br, I) in formula
        has_halogen = bool(formula and re.search(r"\b(Cl|F|Br|I)\d?", str(formula)))
        # Also check name for drug-like patterns
        name_lower = compound_name_str.lower()
        is_drug_like = any(kw in name_lower for kw in [
            "drug", "pharma", "meloxicam", "chloro", "fluoro", "bromo",
        ])

        if has_halogen or is_drug_like:
            bio_score -= 0.5  # strong xenobiotic penalty

        if compound_id_str:
            bio_score += 0.3  # has any annotation
            # Check HMDB origin
            if compound_id_str in origin_map:
                oc = origin_map[compound_id_str]
                if oc.get("is_endogenous"):
                    bio_score += 0.4  # endogenous bonus
                elif oc.get("origin_class") == "exogenous_only":
                    bio_score -= 0.3  # exogenous penalty

        if kegg_hit:
            bio_score += 0.2

        # Lipid class bonus (gangliosides are cancer-relevant)
        lipid_keywords = ["lipid", "fatty acid", "glycero", "sphin", "sterol",
                          "ganglio", "ceramide", "phospho", "bile", "prostagland"]
        if any(kw in name_lower for kw in lipid_keywords):
            bio_score += 0.2

        bio_score = max(0.0, bio_score)

        # Composite
        composite = (
            W_STATISTICAL * stat_score
            + W_ANNOTATION * annotation_score
            + W_KEGG * kegg_score
            + W_BIOLOGICAL * bio_score
        )

        scoring_rows.append({
            "feature_id": fid,
            "consensus_tier": consensus_tier,
            "n_paths_flagging": n_paths,
            "statistical_score": round(stat_score, 3),
            "annotation_score": round(annotation_score, 3),
            "kegg_pathway_score": round(kegg_score, 3),
            "biological_score": round(bio_score, 3),
            "composite_score": round(composite, 3),
            "compound_id": ann.get("compound_id") or "—",
            "compound_name": ann.get("compound_name") or "—",
            "msi_tier": ann.get("msi_tier", 4),
            "kegg_hit": kegg_hit,
        })

    result = pd.DataFrame(scoring_rows)
    result = result.sort_values("composite_score", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


def _write_report(scores: pd.DataFrame) -> str:
    if scores.empty:
        return "# Candidate Prioritization\n\nNo consensus features found."

    n_high = int((scores["composite_score"] >= 0.6).sum())
    n_medium = int(((scores["composite_score"] >= 0.3) & (scores["composite_score"] < 0.6)).sum())
    n_low = int((scores["composite_score"] < 0.3).sum())

    lines = [
        "# Candidate Prioritization for Validation",
        "",
        "_Auto-generated by `scripts/diagnose_candidate_prioritization.py`._",
        "",
        "Ranks consensus features by a composite score combining:",
        f"- **Statistical robustness** ({W_STATISTICAL:.0%}) — path agreement, η², methods passed",
        f"- **Annotation quality** ({W_ANNOTATION:.0%}) — MSI tier, compound name availability",
        f"- **KEGG pathway evidence** ({W_KEGG:.0%}) — pathway membership, enrichment q-value",
        f"- **Biological plausibility** ({W_BIOLOGICAL:.0%}) — annotation, lipid class, KEGG hit",
        "",
        "## Summary",
        "",
        "| Priority tier | Count | Criteria |",
        "|--------------|:-----:|----------|",
        f"| **High** (score ≥ 0.6) | {n_high} | Named + high η² + KEGG pathway | Validate first — MS/MS or purchase standard",
        f"| **Medium** (score 0.3–0.6) | {n_medium} | Named or high η² but missing KEGG | Requires literature check",
        f"| **Low** (score < 0.3) | {n_low} | Unidentified (MSI L4) or low evidence | MS/MS acquisition needed first",
        "",
        "## Ranked Candidates",
        "",
        "| Rank | feature_id | Score | Statistical | Annotation | KEGG | Biological | Tier | Compound |",
        "|:----:|------------|:----:|:-----------:|:----------:|:----:|:----------:|:----:|----------|",
    ]

    for _, r in scores.iterrows():
        tier_label = {"tier_1": "T1", "tier_2": "T2", "tier_3": "T3", "none": "—"}.get(
            r["consensus_tier"], "—"
        )
        name = str(r.get("compound_name", "—"))
        if len(name) > 50:
            name = name[:47] + "…"
        rank = int(r["rank"])
        score = r["composite_score"]
        stat = r["statistical_score"]
        ann = r["annotation_score"]
        kegg = r["kegg_pathway_score"]
        bio = r["biological_score"]
        lines.append(
            f"| {rank} | {r['feature_id']} | {score:.2f} | {stat:.2f} | {ann:.2f} | {kegg:.2f} | "
            f"{bio:.2f} | {tier_label} | {name} |"
        )

    lines += [
        "",
        "## Score Breakdown by Feature",
        "",
        "| feature_id | composite | stat | annotation | KEGG | bio | msi | compound |",
        "|------------|:--------:|:----:|:----------:|:----:|:---:|:---:|----------|",
    ]
    for _, r in scores.iterrows():
        name = str(r.get("compound_name", "—"))
        lines.append(
            f"| {r['feature_id']} | {r['composite_score']:.2f} | {r['statistical_score']:.2f} | "
            f"{r['annotation_score']:.2f} | {r['kegg_pathway_score']:.2f} | "
            f"{r['biological_score']:.2f} | {r['msi_tier']} | {name} |"
        )

    lines += [
        "",
        "## Interpretation Guide",
        "",
        "| Score Range | Action |",
        "|:----------:|--------|",
        "| 0.8–1.0 | **Top priority.** Purchase standard or acquire MS/MS. Proceed to targeted assay.",
        "| 0.6–0.8 | **High priority.** Strong evidence across multiple axes. Literature review before purchasing.",
        "| 0.4–0.6 | **Medium priority.** Good statistical signal but incomplete annotation. Consider MS/MS.",
        "| 0.2–0.4 | **Low priority.** Unidentified or limited evidence. Needs more data.",
        "| < 0.2 | **Exploratory only.** MSI L4 with no additional evidence. Do not validate yet.",
        "",
        "## Method",
        "",
        f"- **Statistical robustness** ({W_STATISTICAL:.0%} weight): mean of normalized path agreement,",
        "  mean eta-squared across paths, and mean methods-passed across paths.",
        f"- **Annotation quality** ({W_ANNOTATION:.0%} weight): MSI tier (L2=1.0, L3=0.6, L4=0.2) ×",
        "  name bonus (1.0 if named, 0.5 if tier-only).",
        f"- **KEGG pathway** ({W_KEGG:.0%} weight): whether the compound maps to a KEGG pathway,",
        "  weighted by the enrichment q-value if below 0.05.",
        f"- **Biological plausibility** ({W_BIOLOGICAL:.0%} weight): has any annotation (0.5),",
        "  is in a KEGG pathway (0.3), is a lipid-class compound (0.2).",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("Candidate Prioritization Score")
    print("=" * 60)
    print()

    consensus = _load_consensus()
    if consensus.empty:
        print("No consensus features found. Run biodisc multiverse run first.")
        return

    path_sls = _load_per_path_shortlists()
    stage9 = _load_stage9()
    enrichment = _load_enrichment()
    features = _load_features()

    print(f"  Consensus features: {len(consensus)}")
    print(f"  Path shortlists:    {len(path_sls)}")
    print(f"  Stage 9 metrics:    {len(stage9)}")
    print(f"  KEGG enrichment:    {'yes' if not enrichment.empty else 'no'}")
    print(f"  Features table:     {len(features)}")

    scores = _compute_prioritization(consensus, path_sls, stage9, enrichment)
    if scores.empty:
        print("No scores computed.")
        return

    report = _write_report(scores)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report)

    n_high = int((scores["composite_score"] >= 0.6).sum())
    n_medium = int(((scores["composite_score"] >= 0.3) & (scores["composite_score"] < 0.6)).sum())
    print(f"\n  High priority:  {n_high}")
    print(f"  Medium priority: {n_medium}")
    print(f"  Low priority:   {len(scores) - n_high - n_medium}")
    print("\n  Top 3:")
    for _, r in scores.head(3).iterrows():
        print(f"    {r['rank']}. {r['feature_id']:25s} score={r['composite_score']:.2f}  "
              f"({r.get('compound_name', '—')})")
    print(f"\n[written] {REPORT}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
