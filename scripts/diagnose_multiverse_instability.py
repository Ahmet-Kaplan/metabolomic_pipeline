"""Quantify single-pipeline instability vs. multiverse consensus.

Reads the existing multiverse consensus table (no re-run) and emits the numbers
behind the methods-paper figures:

  F3  per-path shortlist sizes + pairwise Jaccard (single-pipeline instability)
  F4  leave-one-path-out consensus robustness
  F5  consensus-threshold ablation (>=1/2/3/4 paths)

All outputs are computed from real pipeline output; nothing is estimated.

Usage:
    python scripts/diagnose_multiverse_instability.py
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CONSENSUS_CSV = Path("data/reports/multiverse/consensus.csv")
OUT_DIR = Path("data/reports/multiverse/paper_analysis")
PATH_COLS = {
    "path_a_classical": "per_path_flag_path_a_classical",
    "path_b_robust_ml": "per_path_flag_path_b_robust_ml",
    "path_c_mnar_aware": "per_path_flag_path_c_mnar_aware",
    "path_d_middle": "per_path_flag_path_d_middle",
}


def _flag_sets(df: pd.DataFrame) -> dict[str, set[str]]:
    """Feature-id set each single path would report on its own."""
    return {
        path: set(df.loc[df[col].astype(bool), "feature_id"])
        for path, col in PATH_COLS.items()
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def per_path_and_jaccard(sets: dict[str, set[str]]) -> dict:
    paths = list(sets)
    sizes = {p: len(sets[p]) for p in paths}
    pairwise = {
        f"{a}__{b}": _jaccard(sets[a], sets[b]) for a, b in combinations(paths, 2)
    }
    intersect_all = set.intersection(*sets.values())
    union_all = set.union(*sets.values())
    return {
        "per_path_shortlist_size": sizes,
        "pairwise_jaccard": pairwise,
        "min_pairwise_jaccard": min(pairwise.values()),
        "features_in_all_paths": sorted(intersect_all),
        "n_in_all_paths": len(intersect_all),
        "n_in_any_path": len(union_all),
    }


def threshold_ablation(sets: dict[str, set[str]]) -> list[dict]:
    """How many features are retained at each >=k-of-4 consensus threshold."""
    counts: dict[str, int] = {}
    for s in sets.values():
        for f in s:
            counts[f] = counts.get(f, 0) + 1
    rows = []
    n_paths = len(sets)
    for k in range(1, n_paths + 1):
        kept = [f for f, c in counts.items() if c >= k]
        rows.append(
            {
                "min_paths": k,
                "frac_paths": round(k / n_paths, 3),
                "n_features": len(kept),
            }
        )
    return rows


def leave_one_path_out(
    sets: dict[str, set[str]], min_paths: int = 2
) -> tuple[list[dict], list[str]]:
    """Recompute the >=min_paths consensus after dropping each path in turn.

    Reports how many of the full-consensus features survive, i.e. how much the
    shortlist depends on any single path being present.
    """
    full_counts: dict[str, int] = {}
    for s in sets.values():
        for f in s:
            full_counts[f] = full_counts.get(f, 0) + 1
    full_consensus = {f for f, c in full_counts.items() if c >= min_paths}

    rows = []
    for dropped in sets:
        remaining = {p: s for p, s in sets.items() if p != dropped}
        counts: dict[str, int] = {}
        for s in remaining.values():
            for f in s:
                counts[f] = counts.get(f, 0) + 1
        # keep threshold constant in absolute path count
        loo_consensus = {f for f, c in counts.items() if c >= min_paths}
        rows.append(
            {
                "dropped_path": dropped,
                "n_consensus_without_path": len(loo_consensus),
                "n_full_consensus_retained": len(full_consensus & loo_consensus),
                "n_full_consensus_lost": len(full_consensus - loo_consensus),
            }
        )
    return rows, sorted(full_consensus)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not CONSENSUS_CSV.exists():
        raise FileNotFoundError(
            f"{CONSENSUS_CSV} not found; run `biodisc multiverse run` first."
        )
    df = pd.read_csv(CONSENSUS_CSV)
    sets = _flag_sets(df)

    f3 = per_path_and_jaccard(sets)
    f5 = threshold_ablation(sets)
    loo_rows, full_consensus = leave_one_path_out(sets, min_paths=2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "n_features_evaluated": int(len(df)),
        "single_pipeline_instability": f3,
        "threshold_ablation": f5,
        "leave_one_path_out": {
            "min_paths": 2,
            "n_full_consensus": len(full_consensus),
            "per_dropped_path": loo_rows,
        },
    }
    (OUT_DIR / "instability.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(f5).to_csv(OUT_DIR / "threshold_ablation.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(OUT_DIR / "leave_one_path_out.csv", index=False)

    logger.info("Per-path shortlist sizes: %s", f3["per_path_shortlist_size"])
    logger.info("Min pairwise Jaccard:     %.4f", f3["min_pairwise_jaccard"])
    logger.info("Features in ALL 4 paths:  %d", f3["n_in_all_paths"])
    logger.info("Features in ANY path:     %d", f3["n_in_any_path"])
    logger.info("Threshold ablation:       %s", f5)
    logger.info("Wrote outputs to %s", OUT_DIR)


def _self_check() -> None:
    """Tiny synthetic check that the set math is right."""
    sets = {"p1": {"x", "y"}, "p2": {"y"}, "p3": {"y", "z"}, "p4": set()}
    j = per_path_and_jaccard(sets)
    assert j["per_path_shortlist_size"]["p1"] == 2
    assert j["n_in_all_paths"] == 0  # p4 empty -> intersection empty
    assert j["n_in_any_path"] == 3  # {x,y,z}
    abl = {r["min_paths"]: r["n_features"] for r in threshold_ablation(sets)}
    assert abl[1] == 3 and abl[3] == 1  # y appears in 3 paths
    loo, full = leave_one_path_out(sets, min_paths=2)
    assert len(loo) == len(sets)  # one row per dropped path
    assert set(full) == {"y"}  # y in >=2 paths
    assert _jaccard({"a"}, {"a"}) == 1.0 and _jaccard(set(), set()) == 0.0
    print("self-check OK")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
