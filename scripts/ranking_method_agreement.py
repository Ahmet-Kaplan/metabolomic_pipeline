"""Compute pairwise rank correlations between ranking methods within each path.

Addresses R3 M1 and DA m11: quantifies PLS-DA/sPLS-DA collinearity and
method independence within each preprocessing path.

Outputs:
  - data/reports/multiverse/paper_analysis/ranking_method_agreement.csv
  - data/reports/multiverse/paper_analysis/ranking_method_agreement.txt (summary)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

PATHS = {
    "path_a_classical": Path("data/reports/multiverse/path_a_classical/shortlist.csv"),
    "path_b_robust_ml": Path("data/reports/multiverse/path_b_robust_ml/shortlist.csv"),
    "path_c_mnar_aware": Path("data/reports/multiverse/path_c_mnar_aware/shortlist.csv"),
    "path_d_middle": Path("data/reports/multiverse/path_d_middle/shortlist.csv"),
}

OUT_DIR = Path("data/reports/multiverse/paper_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Score columns for each ranking method
METHOD_SCORES = {
    "ANOVA (η²)": "univariate_eta_sq",
    "PLS-DA (VIP)": "plsda_vip",
    "sPLS-DA (selected)": "splsda_selected",
    "RF (Gini)": "rf_gini",
}


def compute_agreement(path_name: str, path_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    results = []
    methods = list(METHOD_SCORES.keys())
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            col1, col2 = METHOD_SCORES[m1], METHOD_SCORES[m2]
            valid = df[col1].notna() & df[col2].notna()
            x, y = df.loc[valid, col1], df.loc[valid, col2]
            if len(x) < 10:
                continue
            # Convert boolean to numeric for sPLS-DA
            if x.dtype == bool:
                x = x.astype(float)
            if y.dtype == bool:
                y = y.astype(float)
            tau, tau_p = kendalltau(x, y)
            rho, rho_p = spearmanr(x, y)
            results.append({
                "path": path_name,
                "method_1": m1,
                "method_2": m2,
                "kendall_tau": round(tau, 4),
                "kendall_p": f"{tau_p:.2e}" if tau_p < 0.001 else f"{tau_p:.4f}",
                "spearman_rho": round(rho, 4),
                "spearman_p": f"{rho_p:.2e}" if rho_p < 0.001 else f"{rho_p:.4f}",
                "n_features": len(x),
            })
    return pd.DataFrame(results)


def main() -> None:
    all_rows = []
    for path_name, path_csv in PATHS.items():
        if not path_csv.exists():
            print(f"Skipping {path_name}: {path_csv} not found")
            continue
        result = compute_agreement(path_name, path_csv)
        all_rows.append(result)

    if not all_rows:
        print("No data found. Run the multiverse pipeline first.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT_DIR / "ranking_method_agreement.csv", index=False)

    # Summary
    print("=" * 80)
    print("RANKING METHOD AGREEMENT (Kendall's τ)")
    print("=" * 80)
    for path_name in PATHS:
        sub = combined[combined["path"] == path_name]
        print(f"\n  {path_name}:")
        for _, row in sub.iterrows():
            print(f"    {row['method_1']} vs {row['method_2']}: "
                  f"τ = {row['kendall_tau']:+.4f} (p={row['kendall_p']}), "
                  f"n = {row['n_features']}")

    # Cross-path average
    print("\n" + "=" * 80)
    print("CROSS-PATH AVERAGE AGREEMENT")
    print("=" * 80)
    for m1, m2 in [
        ("PLS-DA (VIP)", "sPLS-DA (selected)"),
        ("ANOVA (η²)", "RF (Gini)"),
        ("ANOVA (η²)", "PLS-DA (VIP)"),
        ("RF (Gini)", "PLS-DA (VIP)"),
    ]:
        sub = combined[(combined["method_1"] == m1) & (combined["method_2"] == m2)]
        if not sub.empty:
            avg_tau = sub["kendall_tau"].mean()
            print(f"  {m1} vs {m2}: mean τ = {avg_tau:+.4f} (across {len(sub)} paths)")

    # Save summary as text
    with open(OUT_DIR / "ranking_method_agreement.txt", "w") as f:
        f.write(f"n_features_evaluated (total union): {combined['n_features'].sum()}\n")
        for _, row in combined.iterrows():
            f.write(f"{row['path']},{row['method_1']},{row['method_2']},"
                    f"{row['kendall_tau']},{row['spearman_rho']}\n")

    print(f"\nResults saved to {OUT_DIR / 'ranking_method_agreement.csv'}")


if __name__ == "__main__":
    main()
