"""Render publication-quality figures (F2-F7) for the multiverse-consensus paper.

Publication styling: Times Roman font, Ocean Dusk palette, clean grids,
proper sizing for BMC Bioinformatics / GigaScience (full width ~6.75 in).

Usage:
    python scripts/make_paper_figures.py
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE = Path("data/reports/multiverse/paper_analysis")
FIG_DIR = BASE / "figures"
PATHS = ["path_a_classical", "path_b_robust_ml", "path_c_mnar_aware", "path_d_middle"]
PATH_LABELS = ["A: Classical", "B: Robust-ML", "C: MNAR-aware", "D: Middle"]
PATH_SHORT = ["Classical", "Robust-ML", "MNAR-aware", "Middle"]

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.12,
    "grid.linestyle": "-",
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
})

# Ocean Dusk palette (professional, colorblind-safe)
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
SANDY = "#F4A261"
CORAL = "#E76F51"
DEEP_TEAL = "#264653"
BLUE = "#0072B2"
SKY = "#56B4E9"
GRAY = "#8C8C8C"
LIGHT_GRAY = "#B0BEC5"

FIG_SINGLE = (3.25, 2.5)
FIG_MED = (5.0, 3.0)
FIG_FULL = (6.75, 3.2)
FIG_FULL_TALL = (6.75, 3.8)


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    logger.info("wrote %s/%s.{png,pdf}", FIG_DIR, name)


def fig2_waterfall() -> None:
    df = pd.read_csv(BASE / "waterfall.csv")
    fig, ax = plt.subplots(figsize=FIG_FULL)
    bars = ax.bar(range(len(df)), df["n_surviving"], color=TEAL, width=0.65,
                  edgecolor="white", linewidth=0.5)
    ax.bar(range(len(df)), df["n_surviving"], color=DEEP_TEAL, width=0.65,
           edgecolor="white", linewidth=0.5, alpha=0.15)
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(i, r["n_surviving"], f"{r['n_surviving']:,}",
                ha="center", va="bottom", fontsize=7, color="#444")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["stage"], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Features surviving")
    ax.set_title("Filter cascade: features retained after each QC stage")
    ax.set_ylim(0, df["n_surviving"].max() * 1.15)
    _save(fig, "F2_waterfall")


def fig3_instability() -> None:
    d = json.loads((BASE / "instability.json").read_text())
    sizes = d["single_pipeline_instability"]["per_path_shortlist_size"]
    jac = d["single_pipeline_instability"]["pairwise_jaccard"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_FULL[0], FIG_FULL[1] * 0.9),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    colors_bar = [TEAL, GOLD, SANDY, CORAL]
    bars = ax1.bar(PATH_LABELS, [sizes[p] for p in PATHS], color=colors_bar,
                   width=0.55, edgecolor="white", linewidth=0.5)
    for i, p in enumerate(PATHS):
        ax1.text(i, sizes[p] + 0.3, str(sizes[p]), ha="center", va="bottom",
                 fontsize=8, fontweight="bold")
    ax1.set_ylabel("Shortlist size")
    ax1.set_title("(a) Single-pipeline shortlists")
    ax1.tick_params(axis="x", labelrotation=25, labelsize=7.5)
    ax1.set_ylim(0, max(sizes.values()) * 1.25)
    ax1.grid(axis="y", alpha=0.12)

    n = len(PATHS)
    mat = np.eye(n)
    for a, b in combinations(range(n), 2):
        key = f"{PATHS[a]}__{PATHS[b]}"
        mat[a, b] = mat[b, a] = jac[key]

    im = ax2.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(["A", "B", "C", "D"])
    ax2.set_yticklabels(["A", "B", "C", "D"])
    for a in range(n):
        for b in range(n):
            if a != b:
                ax2.text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center",
                         color="white" if mat[a, b] < 0.4 else "#444",
                         fontsize=8.5, fontweight="bold")
    ax2.set_title("(b) Pairwise Jaccard overlap")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    _save(fig, "F3_instability")


def fig4_leave_one_out() -> None:
    df = pd.read_csv(BASE / "leave_one_path_out.csv")
    labels = [p.replace("path_", "").replace("_", " ") for p in df["dropped_path"]]
    labels = [l.replace("a classical", "Classical").replace("b robust ml", "Robust-ML")
             .replace("c mnar aware", "MNAR-aware").replace("d middle", "Middle")
             for l in labels]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=FIG_MED)
    ax.bar(x, df["n_full_consensus_retained"], label="Retained", color=TEAL,
           width=0.5, edgecolor="white", linewidth=0.5)
    ax.bar(x, df["n_full_consensus_lost"], bottom=df["n_full_consensus_retained"],
           label="Lost", color=CORAL, width=0.5, edgecolor="white", linewidth=0.5)
    for i, (_, r) in enumerate(df.iterrows()):
        retained, lost = r["n_full_consensus_retained"], r["n_full_consensus_lost"]
        ax.text(i, retained / 2, str(retained), ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")
        if lost > 0:
            ax.text(i, retained + lost / 2, str(lost), ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
        else:
            ax.text(i, retained + 0.4, "0", ha="center", va="bottom",
                    fontsize=7, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Consensus features (of 15)")
    ax.set_title("Leave-one-path-out sensitivity")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(0, 18)
    ax.grid(axis="y", alpha=0.12)
    _save(fig, "F4_leave_one_path_out")


def fig5_ablation() -> None:
    df = pd.read_csv(BASE / "threshold_ablation.csv")
    fig, ax = plt.subplots(figsize=FIG_MED)
    ax.plot(df["min_paths"], df["n_features"], "-o", color=DEEP_TEAL,
            linewidth=2, markersize=8, markerfacecolor=TEAL,
            markeredgecolor=DEEP_TEAL, markeredgewidth=1)
    ax.fill_between(df["min_paths"], 0, df["n_features"], alpha=0.08, color=TEAL)
    for _, r in df.iterrows():
        ax.text(r["min_paths"], r["n_features"] + 1.2, f"{int(r['n_features'])}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(df["min_paths"])
    ax.set_xlabel("Consensus threshold (min. paths of 4)")
    ax.set_ylabel("Features retained")
    ax.set_title("Effect of consensus stringency")
    ax.set_ylim(0, df["n_features"].max() * 1.25)
    ax.grid(axis="y", alpha=0.12)
    _save(fig, "F5_threshold_ablation")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fig2_waterfall()
    fig3_instability()
    fig4_leave_one_out()
    fig5_ablation()
    logger.info("All figures written to %s", FIG_DIR)


if __name__ == "__main__":
    main()
