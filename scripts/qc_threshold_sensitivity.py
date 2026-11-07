"""QC filter threshold sensitivity analysis.

Varies key QC thresholds (CV, presence, blank ratio) and reports the
cumulative effect on feature retention.

Addresses R2 M2 + DA m10: quantifies sensitivity of the filter cascade to
threshold choices.

NOTE: This script reads the existing audit ledger and simulates threshold
variation using the per-feature metrics. It does NOT rerun the DVC pipeline.

Outputs:
  - data/reports/multiverse/paper_analysis/qc_threshold_sensitivity.csv
  - data/reports/multiverse/paper_analysis/qc_threshold_sensitivity.txt (summary)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AUDIT_LEDGER = Path("data/audit/master_feature_ledger.csv")
INTERIM_DIR = Path("data/interim")
OUT_DIR = Path("data/reports/multiverse/paper_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics(stage: str, metric_col: str) -> pd.DataFrame:
    """Load per-feature metrics CSV for a given filter stage."""
    csv_path = INTERIM_DIR / f"{stage}/metrics.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # Only keep feature_id and the metric column
    if metric_col in df.columns:
        return df[["feature_id", metric_col]]
    return df


def simulate_stage3_presence(thresholds: list[float]) -> list[dict]:
    """Simulate varying the presence threshold (Stage 3)."""
    # Load presence metrics
    metrics = load_metrics("03_presence", "max_presence_pct")
    if metrics.empty:
        return []
    rows = []
    for thr in thresholds:
        n_kept = int((metrics["max_presence_pct"] >= thr).sum())
        rows.append({"stage": "3_presence", "threshold": f">={thr*100:.0f}%",
                     "n_kept": n_kept,
                     "pct_kept": round(n_kept / len(metrics) * 100, 1)})
    return rows


def simulate_stage4_qc_cv(thresholds: list[float]) -> list[dict]:
    """Simulate varying the QC-CV threshold (Stage 4).
    
    Note: qc_cv_pct column is in percentage points (0-100+ scale).
    """
    metrics = load_metrics("04_qc_cv", "qc_cv_pct")
    if metrics.empty:
        return []
    rows = []
    for thr in thresholds:
        valid = metrics["qc_cv_pct"].notna()
        n_kept = int((metrics.loc[valid, "qc_cv_pct"] <= thr).sum())
        rows.append({"stage": "4_qc_cv", "threshold": f"<={thr:.0f}%",
                     "n_kept": n_kept,
                     "pct_kept": round(n_kept / len(metrics) * 100, 1)})
    return rows


def simulate_stage6_blank(thresholds: list[float]) -> list[dict]:
    """Simulate varying the blank ratio threshold (Stage 6)."""
    metrics = load_metrics("06_blank", "blank_pct")
    if metrics.empty:
        return []
    rows = []
    for thr in thresholds:
        valid = metrics["blank_pct"].notna()
        n_kept = int((metrics.loc[valid, "blank_pct"] <= thr).sum())
        rows.append({"stage": "6_blank", "threshold": f"<={thr*100:.0f}%",
                     "n_kept": n_kept,
                     "pct_kept": round(n_kept / len(metrics) * 100, 1)})
    return rows


def main() -> None:
    # Define threshold ranges
    cv_thresholds = [20, 25, 30, 35, 40, 50]
    presence_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    blank_thresholds = [0.10, 0.15, 0.20, 0.30, 0.50]

    all_rows = []
    all_rows.extend(simulate_stage3_presence(presence_thresholds))
    all_rows.extend(simulate_stage4_qc_cv(cv_thresholds))
    all_rows.extend(simulate_stage6_blank(blank_thresholds))

    if not all_rows:
        print("No metrics data found. Run the QC filter cascade first.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "qc_threshold_sensitivity.csv", index=False)

    print("=" * 65)
    print("QC FILTER THRESHOLD SENSITIVITY")
    print("=" * 65)
    for stage in ["3_presence", "4_qc_cv", "6_blank"]:
        sub = df[df["stage"] == stage]
        print(f"\nStage {stage}:")
        for _, row in sub.iterrows():
            print(f"  Threshold {row['threshold']}: {row['n_kept']} kept "
                  f"({row['pct_kept']}%)")
    print(f"\nSaved to {OUT_DIR / 'qc_threshold_sensitivity.csv'}")


if __name__ == "__main__":
    main()
