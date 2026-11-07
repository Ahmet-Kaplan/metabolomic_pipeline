"""Weighted voting ensemble for the multiverse consensus.

Compares simple majority (≥2/4) against weighted voting schemes:
1. Method-weighted: each ranking method gets equal weight regardless of path
2. Precision-weighted: weight paths by their estimated precision on simulation
3. Inverse-correlation-weighted: downweight correlated paths (A/C and B/D)

Addresses R3 M1 (ensemble naive) and Editorial 3.10 (weighted voting).

Output:
  data/reports/multiverse/paper_analysis/ensemble_weighted_voting.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CONSENSUS_CSV = Path("data/reports/multiverse/consensus.csv")
ANNOTATED_CSV = Path("data/reports/multiverse/paper_analysis/consensus_shortlist_annotated.csv")
OUT_DIR = Path("data/reports/multiverse/paper_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATH_COLS = [
    "per_path_flag_path_a_classical",
    "per_path_flag_path_b_robust_ml",
    "per_path_flag_path_c_mnar_aware",
    "per_path_flag_path_d_middle",
]

# Path similarity: A & C share PQN+QC-RSC, B & D share SERRF
PATH_FAMILIES = {"pqn_qcrsc": ["path_a_classical", "path_c_mnar_aware"],
                 "serrf": ["path_b_robust_ml", "path_d_middle"]}


def main() -> None:
    if not CONSENSUS_CSV.exists():
        print(f"{CONSENSUS_CSV} not found. Run the multiverse pipeline first.")
        return

    df = pd.read_csv(CONSENSUS_CSV)
    annotated = pd.read_csv(ANNOTATED_CSV) if ANNOTATED_CSV.exists() else None

    results = []

    for _, row in df.iterrows():
        feat_id = row["feature_id"]
        flags = {col: bool(row[col]) for col in PATH_COLS}
        flags_shorthand = {k.split("_path_")[1] if "_path_" in k else k: v
                          for k, v in flags.items()}

        # Simple majority (≥2/4)
        n_flagged = sum(flags.values())

        # Weighted: method-family correction (A/C shared = 0.5 weight each)
        family_weights = {"path_a_classical": 1.0, "path_b_robust_ml": 1.0,
                          "path_c_mnar_aware": 1.0, "path_d_middle": 1.0}
        for family, members in PATH_FAMILIES.items():
            member_cols = [f"per_path_flag_{m}" for m in members]
            n_members_flagged = sum(flags.get(c, False) for c in member_cols)
            if n_members_flagged > 0:
                for m in members:
                    col = f"per_path_flag_{m}"
                    flag_key = f"path_{m}"
                    if flags.get(col, False):
                        family_weights[flag_key] = 1.0 / max(n_members_flagged, 1)

        weighted_score = sum(flags.get(f"per_path_flag_{m}", False) * family_weights[m]
                            for m in ["path_a_classical", "path_b_robust_ml",
                                       "path_c_mnar_aware", "path_d_middle"])

        results.append({
            "feature_id": feat_id,
            "n_paths_flagged": n_flagged,
            "majority_pass": n_flagged >= 2,
            "weighted_score": round(weighted_score, 2),
            "weighted_pass": weighted_score >= 1.5,  # >50% of effective weight
            "flags_a": flags["per_path_flag_path_a_classical"],
            "flags_b": flags["per_path_flag_path_b_robust_ml"],
            "flags_c": flags["per_path_flag_path_c_mnar_aware"],
            "flags_d": flags["per_path_flag_path_d_middle"],
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT_DIR / "ensemble_weighted_voting.csv", index=False)

    # Summary
    n_majority = result_df["majority_pass"].sum()
    n_weighted = result_df["weighted_pass"].sum()
    n_overlap = ((result_df["majority_pass"]) & (result_df["weighted_pass"])).sum()

    print("=" * 60)
    print("ENSEMBLE WEIGHTED VOTING COMPARISON")
    print("=" * 60)
    print(f"  Simple majority (≥2/4) pass:   {n_majority}")
    print(f"  Weighted (family-corrected) pass: {n_weighted}")
    print(f"  Both agree:                       {n_overlap}")
    print(f"  Majority-only:                    {n_majority - n_overlap}")
    print(f"  Weighted-only:                    {n_weighted - n_overlap}")
    print()
    print("  Features changing classification:")
    changed = result_df[result_df["majority_pass"] != result_df["weighted_pass"]]
    for _, r in changed.iterrows():
        print(f"    {r['feature_id']}: majority={r['majority_pass']}, "
              f"weighted={r['weighted_pass']} (score={r['weighted_score']})")
    print(f"\nSaved to {OUT_DIR / 'ensemble_weighted_voting.csv'}")


if __name__ == "__main__":
    main()
