"""Stability analysis of the consensus threshold k.

Computes the Adjusted Rand Index (ARI) between consensus outputs at
neighbouring k values, and reports the full threshold sweep.

Addresses R1 W2 + DA m9: quantifies sensitivity of the consensus to the
chosen ≥2/4 threshold.

Outputs:
  - data/reports/multiverse/paper_analysis/threshold_stability.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

CONSENSUS_CSV = Path("data/reports/multiverse/consensus.csv")
OUT_DIR = Path("data/reports/multiverse/paper_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATH_COLS = {
    "path_a_classical": "per_path_flag_path_a_classical",
    "path_b_robust_ml": "per_path_flag_path_b_robust_ml",
    "path_c_mnar_aware": "per_path_flag_path_c_mnar_aware",
    "path_d_middle": "per_path_flag_path_d_middle",
}


def compute_labels_at_threshold(df: pd.DataFrame, k: int, n_paths: int) -> np.ndarray:
    """Assign tier labels at a given consensus threshold."""
    counts = df[list(PATH_COLS.values())].sum(axis=1).values
    labels = np.zeros(len(df), dtype=int)
    labels[counts >= k] = 1  # consensus feature
    # Tier 1 for >=3/4 when k <= 3
    if k <= 3:
        labels[counts >= min(3, n_paths)] = 2
    return labels


def main() -> None:
    if not CONSENSUS_CSV.exists():
        print(f"{CONSENSUS_CSV} not found. Run the multiverse pipeline first.")
        return

    df = pd.read_csv(CONSENSUS_CSV)
    n_paths = len(PATH_COLS)

    # Threshold sweep
    rows = []
    for k in range(0, n_paths + 1):
        n_feats = int((df[list(PATH_COLS.values())].sum(axis=1) >= k).sum())
        rows.append({"threshold_k": k, "n_consensus_features": n_feats})

    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT_DIR / "threshold_stability_sweep.csv", index=False)

    # ARI between neighbouring k values
    ari_rows = []
    for k in range(1, n_paths):
        labels_k = compute_labels_at_threshold(df, k, n_paths)
        labels_k1 = compute_labels_at_threshold(df, k + 1, n_paths)
        ari = adjusted_rand_score(labels_k, labels_k1)
        ari_rows.append({
            "k_lower": k,
            "k_upper": k + 1,
            "adjusted_rand_index": round(ari, 4),
            "description": f"≥{k}/4 vs ≥{k+1}/4 consensus",
        })
    ari_df = pd.DataFrame(ari_rows)
    ari_df.to_csv(OUT_DIR / "threshold_stability_ari.csv", index=False)

    # Summary
    print("=" * 70)
    print("CONSENSUS THRESHOLD STABILITY ANALYSIS")
    print("=" * 70)
    print("\nThreshold sweep:")
    for _, row in sweep.iterrows():
        print(f"  ≥{row['threshold_k']}/4 paths: {row['n_consensus_features']} features")
    print("\nAdjusted Rand Index between neighbouring thresholds:")
    for _, row in ari_df.iterrows():
        print(f"  {row['description']}: ARI = {row['adjusted_rand_index']}")
    print(f"\nResults saved to {OUT_DIR / 'threshold_stability_sweep.csv'} and "
          f"{OUT_DIR / 'threshold_stability_ari.csv'}")


if __name__ == "__main__":
    main()
