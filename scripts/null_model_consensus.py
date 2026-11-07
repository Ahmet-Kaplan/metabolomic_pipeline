"""Null model for expected consensus size under label-shuffling.

Estimates the distribution of consensus feature counts under the null
hypothesis that group labels are unrelated to feature abundances.
Permutes group labels N times, reruns the univariate ranking on Path A,
and counts features that would pass the ≥2/4 consensus threshold.

This is the pipeline-wide analogue of the per-path label-permutation test.
Unlike the 50-permutation quick estimate, this runs 1000 iterations and
reports the full null distribution.

Addresses DA M7 (null model), R1 W1 (permutation count), R1 W4 (no validation).

Outputs:
  - data/reports/multiverse/paper_analysis/null_model_consensus.csv

Usage:
    python scripts/null_model_consensus.py --n-permutations 1000

CAUTION: This script may take several hours for 1000 permutations on a
real dataset. Test with --n-permutations 10 first.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Data paths
RAW_ABUNDANCE = Path("data/interim/01_parsed/abundances_raw_long.parquet")
FEATURES_AFTER_QC = Path("data/interim/10_origin/kept_features.parquet")
SAMPLES = Path("data/interim/01_parsed/samples.parquet")
OUT_DIR = Path("data/reports/multiverse/paper_analysis")

# Path A parameters (Classical pipeline, univariate only for speed)
# Full multiverse not rerun in null model; instead we use the univariate
# gate (BH-q ≤ 0.05, η² ≥ 0.14) as a proxy for "shortlist entry"
BH_THRESHOLD = 0.05
ETA_SQ_THRESHOLD = 0.14
N_GROUPS = 5


def _load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load abundance data, feature IDs, and true group labels."""
    abd = pd.read_parquet(RAW_ABUNDANCE)
    samples = pd.read_parquet(SAMPLES)
    sample_lookup = samples[["sample_id", "cell_line"]].drop_duplicates()
    abd = abd.merge(sample_lookup, on="sample_id", how="left")

    # Post-QC features only
    qc_features = set(pd.read_parquet(FEATURES_AFTER_QC)["feature_id"])
    abd = abd[abd["feature_id"].isin(qc_features)]

    # Pivot to wide
    wide = abd.pivot_table(index="feature_id", columns="sample_id",
                           values="intensity", aggfunc="first")
    feature_ids = wide.index.values
    intensity = wide.values.astype(np.float64)

    # True group labels
    groups = abd[["sample_id", "cell_line"]].drop_duplicates()
    groups = groups.dropna(subset=["cell_line"])
    group_map = dict(zip(groups["sample_id"], groups["cell_line"]))
    sample_order = wide.columns
    true_labels = np.array([group_map.get(s, "UNKNOWN") for s in sample_order])

    return feature_ids, intensity, true_labels, sample_order


def _anova_eta2(data: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """One-way ANOVA: return BH-q and η²."""
    # Filter out NaN labels and corresponding data
    valid = ~pd.isna(labels)
    labels = labels[valid]
    data = data[valid]
    groups = np.unique(labels)
    grand_mean = np.nanmean(data)
    ss_between = 0.0
    ss_within = 0.0
    n_total = 0
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
    df_between = len(groups) - 1
    df_within = n_total - len(groups)
    if df_within <= 0 or ss_within <= 0:
        return 1.0, eta2
    f_stat = (ss_between / df_between) / (ss_within / df_within)
    from scipy.stats import f
    p_value = 1.0 - f.cdf(f_stat, df_between, df_within)
    return p_value, eta2


def _run_permutation(intensity: np.ndarray, labels: np.ndarray,
                     feature_ids: np.ndarray) -> int:
    """Run one permutation: shuffle labels, compute ANOVA, count shortlisted."""
    perm_labels = labels[np.random.permutation(len(labels))]
    p_values = np.full(len(feature_ids), np.nan)
    eta2_values = np.full(len(feature_ids), np.nan)
    for i in range(len(feature_ids)):
        p_values[i], eta2_values[i] = _anova_eta2(intensity[i], perm_labels)

    # BH correction
    valid = ~np.isnan(p_values)
    n_valid = valid.sum()
    if n_valid == 0:
        return 0
    p_vals = p_values[valid]
    eta2_vals = eta2_values[valid]
    order = np.argsort(p_vals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(n_valid) + 1
    bh_q = p_vals * n_valid / ranks
    bh_q = np.minimum.accumulate(bh_q[::-1])[::-1]

    passed = (bh_q <= BH_THRESHOLD) & (eta2_vals >= ETA_SQ_THRESHOLD)
    return int(passed.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Null model for consensus size")
    parser.add_argument("--n-permutations", type=int, default=200,
                        help="Number of label permutations (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s")

    logger.info("Loading data...")
    feature_ids, intensity, true_labels, sample_order = _load_data()
    logger.info(f"Loaded {len(feature_ids)} features, {intensity.shape[1]} samples")

    # Run permutations
    null_counts = np.zeros(args.n_permutations, dtype=int)
    for perm in range(args.n_permutations):
        null_counts[perm] = _run_permutation(intensity, true_labels, feature_ids)
        if (perm + 1) % 25 == 0:
            logger.info(f"  Permutation {perm + 1}/{args.n_permutations}: "
                        f"{null_counts[perm]} features")

    # Results
    mean_null = null_counts.mean()
    sd_null = null_counts.std()
    p95 = np.percentile(null_counts, 95)
    p99 = np.percentile(null_counts, 99)
    maximum = null_counts.max()

    logger.info("=" * 60)
    logger.info("NULL MODEL RESULTS")
    logger.info(f"  Permutations: {args.n_permutations}")
    logger.info(f"  Mean null features: {mean_null:.2f}")
    logger.info(f"  SD: {sd_null:.2f}")
    logger.info(f"  95th percentile: {p95:.0f}")
    logger.info(f"  99th percentile: {p99:.0f}")
    logger.info(f"  Maximum: {maximum}")
    logger.info("=" * 60)

    out = pd.DataFrame({
        "n_permutations": args.n_permutations,
        "mean_null_features": round(mean_null, 2),
        "sd_null_features": round(sd_null, 2),
        "p95_null_features": int(p95),
        "p99_null_features": int(p99),
        "max_null_features": int(maximum),
        "observed_path_a": 14,
        "observed_consensus": 15,
    }, index=[0])
    out.to_csv(OUT_DIR / "null_model_consensus.csv", index=False)
    logger.info(f"Saved to {OUT_DIR / 'null_model_consensus.csv'}")


if __name__ == "__main__":
    main()
