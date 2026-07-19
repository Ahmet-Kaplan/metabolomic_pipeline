"""Build the methods-paper tables and the filter-cascade waterfall.

Outputs (data/reports/multiverse/paper_analysis/):
  waterfall.csv                     F2  survivors after each filter stage
  consensus_shortlist_annotated.csv T1  the 15 consensus features + annotation

Every value comes from committed pipeline output. The two xenobiotic flags are
NOT computed here (the pipeline's origin filter did not catch them); they are
carried, with attribution, from the manual scientific evaluation in
docs/evaluation/00_executive_summary.md so the table stays honest about how they
were identified.

Usage:
    python scripts/build_paper_tables.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

LEDGER = Path("data/audit/master_feature_ledger.parquet")
CONSENSUS = Path("data/reports/multiverse/consensus.csv")
ID_METRICS = Path("data/interim/09_id_ranking/metrics.parquet")
PATH_SHORTLISTS = {
    "path_a_classical": "data/reports/multiverse/path_a_classical/shortlist.parquet",
    "path_b_robust_ml": "data/reports/multiverse/path_b_robust_ml/shortlist.parquet",
    "path_c_mnar_aware": "data/reports/multiverse/path_c_mnar_aware/shortlist.parquet",
    "path_d_middle": "data/reports/multiverse/path_d_middle/shortlist.parquet",
}
OUT_DIR = Path("data/reports/multiverse/paper_analysis")

# Filter cascade in execution order (stage-9 id-ranking drops nothing).
STAGE_ORDER = [
    "03_presence",
    "04_qc_cv",
    "05_sample_qc_ratio",
    "06_blank",
    "07_dilution",
    "08_ccs",
    "10_origin",
]

# Sourced from docs/evaluation/00_executive_summary.md (manual expert review),
# NOT from the automated pipeline. See module docstring.
XENOBIOTIC_FLAGS = {
    "2.37_343.0578n": "halogenated synthetic (Cl, F3); not endogenous",
    "2.37_703.0737m/z": "Meloxicam (NSAID) 2M+H homodimer",
}


def build_waterfall(ledger: pd.DataFrame) -> pd.DataFrame:
    dropped = ledger["dropped_at_stage"].value_counts(dropna=False).to_dict()
    n = len(ledger)
    rows = [{"stage": "00_raw", "n_dropped_here": 0, "n_surviving": n}]
    for stage in STAGE_ORDER:
        d = int(dropped.get(stage, 0))
        n -= d
        rows.append({"stage": stage, "n_dropped_here": d, "n_surviving": n})
    return pd.DataFrame(rows)


def build_shortlist_table(
    consensus: pd.DataFrame,
    id_metrics: pd.DataFrame,
    path_stats: pd.DataFrame,
) -> pd.DataFrame:
    keep = consensus[consensus["passed_consensus"]].copy()
    cols = [
        "feature_id",
        "consensus_tier",
        "n_paths_flagging",
        "frac_paths_flagging",
        "per_path_flag_path_a_classical",
        "per_path_flag_path_b_robust_ml",
        "per_path_flag_path_c_mnar_aware",
        "per_path_flag_path_d_middle",
    ]
    out = keep[cols].merge(
        id_metrics[
            ["feature_id", "msi_tier", "outcome", "top_compound_name", "top_formula",
             "top_ppm", "had_msms"]
        ],
        on="feature_id",
        how="left",
    )
    out = out.merge(path_stats, on="feature_id", how="left")
    out["xenobiotic_flag"] = out["feature_id"].map(XENOBIOTIC_FLAGS).fillna("")
    out["is_endogenous_candidate"] = out["xenobiotic_flag"] == ""
    out = out.sort_values(
        ["frac_paths_flagging", "feature_id"], ascending=[False, True]
    ).reset_index(drop=True)
    return out


def best_path_stats() -> pd.DataFrame:
    """Best univariate q / effect size / VIP a feature reaches across paths."""
    frames = []
    for p in PATH_SHORTLISTS.values():
        df = pd.read_parquet(p)[
            ["feature_id", "univariate_q", "univariate_eta_sq", "plsda_vip"]
        ].copy()
        frames.append(df)
    allp = pd.concat(frames)
    agg = allp.groupby("feature_id").agg(
        best_univariate_q=("univariate_q", "min"),
        max_univariate_eta_sq=("univariate_eta_sq", "max"),
        max_plsda_vip=("plsda_vip", "max"),
    )
    return agg.reset_index()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for f in (LEDGER, CONSENSUS, ID_METRICS):
        if not Path(f).exists():
            raise FileNotFoundError(f"{f} not found; run the pipeline first.")

    ledger = pd.read_parquet(LEDGER)
    consensus = pd.read_csv(CONSENSUS)
    id_metrics = pd.read_parquet(ID_METRICS)
    path_stats = best_path_stats()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    waterfall = build_waterfall(ledger)
    shortlist = build_shortlist_table(consensus, id_metrics, path_stats)

    waterfall.to_csv(OUT_DIR / "waterfall.csv", index=False)
    shortlist.to_csv(OUT_DIR / "consensus_shortlist_annotated.csv", index=False)

    logger.info("Waterfall:\n%s", waterfall.to_string(index=False))
    logger.info(
        "\nShortlist: %d features (%d endogenous candidates, %d xenobiotic)",
        len(shortlist),
        int(shortlist["is_endogenous_candidate"].sum()),
        int((~shortlist["is_endogenous_candidate"]).sum()),
    )
    logger.info("Wrote outputs to %s", OUT_DIR)


if __name__ == "__main__":
    main()
