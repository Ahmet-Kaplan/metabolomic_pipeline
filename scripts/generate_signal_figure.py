"""Generate ganglioside signal figure (F6) with publication styling."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RAW_ABUNDANCE = Path("data/interim/01_parsed/abundances_raw_long.parquet")
SAMPLES = Path("data/interim/01_parsed/samples.parquet")
OUT_DIR = Path("data/reports/multiverse/paper_analysis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GANGLIOSIDE_FEATURE = "10.76_1918.0639n"
TEAL = "#2A9D8F"
CORAL = "#E76F51"


def plot_ganglioside(abd: pd.DataFrame) -> None:
    sub = abd[abd["feature_id"] == GANGLIOSIDE_FEATURE].copy()
    cell_line_order = (
        sub.groupby("cell_line")["intensity"].median().sort_values().index.tolist()
    )
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    bp = sns.boxplot(
        data=sub, x="cell_line", y="intensity", order=cell_line_order,
        palette=[TEAL, CORAL, "#E9C46A", "#F4A261", "#264653"],
        ax=ax, width=0.55, linewidth=1.2,
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )
    sns.stripplot(
        data=sub, x="cell_line", y="intensity", order=cell_line_order,
        color="#333", size=4.5, alpha=0.6, ax=ax, jitter=0.12,
    )
    ax.set_xlabel("Cell line")
    ax.set_ylabel("Raw intensity")
    ax.set_title("Ganglioside GD1a/GD1b")
    ax.grid(axis="y", alpha=0.12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "F6_ganglioside_boxplot.pdf")
    fig.savefig(OUT_DIR / "F6_ganglioside_boxplot.png", dpi=300)
    plt.close(fig)
    print(f"Saved ganglioside boxplot to {OUT_DIR}")


def build_per_path_table() -> None:
    paths = {
        "path_a_classical": Path("data/reports/multiverse/path_a_classical/shortlist.csv"),
        "path_b_robust_ml": Path("data/reports/multiverse/path_b_robust_ml/shortlist.csv"),
        "path_c_mnar_aware": Path("data/reports/multiverse/path_c_mnar_aware/shortlist.csv"),
        "path_d_middle": Path("data/reports/multiverse/path_d_middle/shortlist.csv"),
    }
    consensus = pd.read_csv(
        Path("data/reports/multiverse/paper_analysis/consensus_shortlist_annotated.csv")
    )
    consensus_features = set(consensus["feature_id"])

    rows = []
    for path_name, path_csv in paths.items():
        df = pd.read_csv(path_csv)
        df = df[df["feature_id"].isin(consensus_features)]
        for _, row in df.iterrows():
            rows.append({
                "feature_id": row["feature_id"],
                "path": path_name,
                "univariate_q": row["univariate_q"],
                "univariate_eta_sq": row["univariate_eta_sq"],
            })

    per_path = pd.DataFrame(rows)
    out_dir = OUT_DIR.parent
    pivot_q = per_path.pivot_table(
        index="feature_id", columns="path", values="univariate_q", aggfunc="first"
    )
    pivot_eta = per_path.pivot_table(
        index="feature_id", columns="path", values="univariate_eta_sq", aggfunc="first"
    )
    best = consensus[["feature_id", "best_univariate_q", "max_univariate_eta_sq"]].set_index("feature_id")
    result = pivot_q.join(pivot_eta, lsuffix="_q", rsuffix="_eta").join(best)
    result.to_csv(out_dir / "per_path_q_eta.csv")
    print(f"Saved per-path q/eta² to {out_dir / 'per_path_q_eta.csv'}")


def main() -> None:
    abd = pd.read_parquet(RAW_ABUNDANCE)
    samples = pd.read_parquet(SAMPLES)
    sample_lookup = samples[["sample_id", "cell_line"]].drop_duplicates()
    abd = abd.merge(sample_lookup, on="sample_id", how="left")
    plot_ganglioside(abd)
    build_per_path_table()


if __name__ == "__main__":
    main()
