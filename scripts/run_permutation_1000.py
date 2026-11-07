"""Re-run the pipeline-wide label-permutation FDR test with 1000 iterations.

Addresses R1 W1, DA M5: replaces the original 50-permutation estimate
with a properly powered 1000-permutation test.

This script runs the univariate ranking (Path A) with permuted group labels
1000 times and reports the null distribution of shortlist sizes.

Usage:
    python scripts/run_permutation_1000.py [--n-permutations 1000]

Output:
    data/reports/multiverse/paper_analysis/permutation_fdr_1000.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

logger = logging.getLogger(__name__)

RAW_ABUNDANCE = Path("data/interim/01_parsed/abundances_raw_long.parquet")
QC_FEATURES = Path("data/interim/10_origin/kept_features.parquet")
SAMPLES = Path("data/interim/01_parsed/samples.parquet")
OUT_DIR = Path("data/reports/multiverse/paper_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BH_THRESHOLD = 0.05
ETA_SQ_THRESHOLD = 0.14


def _load_data():
    """Load post-QC abundance data and group labels."""
    abd = pd.read_parquet(RAW_ABUNDANCE)
    samples = pd.read_parquet(SAMPLES)
    sample_lookup = samples[["sample_id", "cell_line"]].drop_duplicates()
    abd = abd.merge(sample_lookup, on="sample_id", how="left")
    qc_features = set(pd.read_parquet(QC_FEATURES)["feature_id"])
    abd = abd[abd["feature_id"].isin(qc_features)]
    wide = abd.pivot_table(index="feature_id", columns="sample_id",
                           values="intensity", aggfunc="first")
    intensity = wide.values.astype(np.float64)
    groups = abd[["sample_id", "cell_line"]].drop_duplicates()
    groups = groups.dropna(subset=["cell_line"])
    group_map = dict(zip(groups["sample_id"], groups["cell_line"]))
    true_labels = np.array([group_map.get(s, "UNKNOWN") for s in wide.columns])
    return wide.index.values, intensity, true_labels


def _anova_eta2(data, labels):
    """One-way ANOVA returning p-value and η²."""
    valid = ~pd.isna(labels)
    labels = labels[valid]
    data = data[valid]
    groups = np.unique(labels)
    grand_mean = np.nanmean(data)
    ss_between, ss_within, n_total = 0.0, 0.0, 0
    for g in groups:
        mask = labels == g
        g_data = data[mask]
        g_mean = np.nanmean(g_data)
        n_g = np.sum(~np.isnan(g_data))
        ss_between += n_g * (g_mean - grand_mean) ** 2
        ss_within += np.nansum((g_data - g_mean) ** 2)
        n_total += n_g
    ss_total = ss_between + ss_within
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    df_between, df_within = len(groups) - 1, n_total - len(groups)
    if df_within <= 0 or ss_within <= 0:
        return 1.0, eta2
    f_stat = (ss_between / df_between) / (ss_within / df_within)
    p_value = 1.0 - f_dist.cdf(f_stat, df_between, df_within)
    return p_value, eta2


def _run_permutation(intensity, labels, feature_ids):
    """One permutation: return number of features passing the univariate gate."""
    perm_labels = labels[np.random.permutation(len(labels))]
    p_vals = np.full(len(feature_ids), np.nan)
    eta2_vals = np.full(len(feature_ids), np.nan)
    for i in range(len(feature_ids)):
        p_vals[i], eta2_vals[i] = _anova_eta2(intensity[i], perm_labels)
    valid = ~np.isnan(p_vals)
    n_valid = valid.sum()
    if n_valid == 0:
        return 0
    pv, ev = p_vals[valid], eta2_vals[valid]
    order = np.argsort(pv)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(n_valid) + 1
    bh_q = np.minimum.accumulate((pv * n_valid / ranks)[np.argsort(order)])
    return int(((bh_q <= BH_THRESHOLD) & (ev >= ETA_SQ_THRESHOLD)).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run only 10 iterations to test")
    args = parser.parse_args()

    n_perms = 10 if args.dry_run else args.n_permutations
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    logger.info(f"Loading data for {n_perms} permutations...")
    feature_ids, intensity, true_labels = _load_data()
    logger.info(f"Loaded {len(feature_ids)} features, {intensity.shape[1]} samples")

    null_counts = np.zeros(n_perms, dtype=int)
    for perm in range(n_perms):
        null_counts[perm] = _run_permutation(intensity, true_labels, feature_ids)
        if (perm + 1) % 100 == 0 or perm == n_perms - 1:
            logger.info(f"  [{perm + 1}/{n_perms}] null features: {null_counts[perm]}")

    mean_null = null_counts.mean()
    sd_null = null_counts.std()
    p95, p99 = np.percentile(null_counts, [95, 99])
    maximum = null_counts.max()

    logger.info("=" * 60)
    logger.info(f"PIPELINE-WIDE PERMUTATION FDR ({n_perms} iterations)")
    logger.info(f"  Mean null features:        {mean_null:.2f}")
    logger.info(f"  SD:                        {sd_null:.2f}")
    logger.info(f"  95th percentile:           {p95:.0f}")
    logger.info(f"  99th percentile:           {p99:.0f}")
    logger.info(f"  Maximum observed:          {maximum}")
    logger.info(f"  Observed (Path A):         14")
    logger.info(f"  Observed (consensus):      15")
    logger.info("=" * 60)

    out = pd.DataFrame({
        "n_permutations": n_perms,
        "mean_null": round(mean_null, 2),
        "sd_null": round(sd_null, 2),
        "p95_null": int(p95),
        "p99_null": int(p99),
        "max_null": int(maximum),
        "observed_path_a": 14,
        "observed_consensus": 15,
    }, index=[0])
    out.to_csv(OUT_DIR / "permutation_fdr_1000.csv", index=False)
    logger.info(f"Saved to {OUT_DIR / 'permutation_fdr_1000.csv'}")


if __name__ == "__main__":
    main()
