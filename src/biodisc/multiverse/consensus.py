"""Phase 10 — multiverse consensus aggregation.

Takes the per-path Phase 9 shortlists (one per ``PathSpec`` from the
orchestrator) and computes which features survived ≥ M-of-N paths.
This is the multiverse design's payoff: a feature flagged by many
methodological choices is robust to any single method's pathology
(Steegen 2016, Simonsohn 2020).

What "survives" means
---------------------
Within each path, Phase 9 already produced a per-feature boolean
``passed_shortlist`` based on the path's chosen rule (typically k=2 of
4 methods). The consensus aggregator counts, for each feature, how
many paths set this flag. Features pass the multiverse-consensus gate
when their cross-path agreement fraction exceeds ``threshold``
(default 0.5 — at least half the paths agree).

Output
------
``ConsensusResult.table`` is one row per feature with explicit columns:

  * ``feature_id``
  * ``n_paths_total``: total paths considered (succeeded ones)
  * ``n_paths_flagging``: how many flagged this feature
  * ``frac_paths_flagging``: ``n_paths_flagging / n_paths_total``
  * ``per_path_flag_<name>``: boolean evidence per path (one column each)
  * ``consensus_tier``: ``"tier_1"`` (≥ 75 %), ``"tier_2"`` (50–75 %),
    ``"tier_3"`` (1–50 %), ``"none"`` (0).
  * ``passed_consensus``: ``frac_paths_flagging >= threshold``

Plus per-method drill-down columns: how many paths agreed on each
method's evidence — useful when reviewers want to know "was this a
univariate-driven consensus or a multivariate one?".

References
----------
* Steegen S et al. 2016. *Increasing transparency through a multiverse
  analysis.* **Perspect Psychol Sci** 11:702.
* Simonsohn U et al. 2020. *Specification curve analysis.* **Nat Hum
  Behav** 4:1208.
* Patel CJ et al. 2015. *Assessment of vibration of effects due to
  model specification can demonstrate the instability of observational
  associations.* **J Clin Epidemiol** 68:1046.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse.orchestrator import PathRunResult

log = structlog.get_logger(__name__)


# ============================================================
# Result container
# ============================================================
@dataclass(frozen=True)
class ConsensusResult:
    """Output of :func:`compute_consensus`.

    Attributes
    ----------
    table
        Per-feature DataFrame (sorted by ``passed_consensus`` desc, then
        ``frac_paths_flagging`` desc, then ``feature_id``).
    n_paths_total
        Number of succeeded paths input.
    n_consensus
        ``int(table["passed_consensus"].sum())``.
    n_per_tier
        Map ``{"tier_1", "tier_2", "tier_3", "none"} → int``.
    threshold
        Threshold used for ``passed_consensus``.
    path_names
        Names of all paths in input order — preserved for traceability.
    runtime_seconds
        Wall time of :func:`compute_consensus`.
    """

    table: pd.DataFrame
    n_paths_total: int
    n_consensus: int
    n_per_tier: dict[str, int]
    threshold: float
    path_names: list[str]
    runtime_seconds: float


# ============================================================
# Aggregation
# ============================================================
def compute_consensus(
    path_results: list[PathRunResult],
    *,
    threshold: float = 0.5,
    include_failed_paths: bool = False,
    method_drilldown: bool = True,
) -> ConsensusResult:
    """Aggregate per-path shortlists into a multiverse consensus.

    Parameters
    ----------
    path_results
        Output of ``run_multiverse``. Failed paths
        (``succeeded=False``) are skipped by default.
    threshold
        Minimum fraction of paths that must flag a feature for
        ``passed_consensus=True``. Default ``0.5`` = simple majority.
    include_failed_paths
        If ``True``, include failed paths in the denominator (i.e.
        treat failures as "not flagged"). Default ``False`` —
        failures are excluded from both numerator and denominator,
        which is the correct behaviour when a path crashes for
        non-biology reasons (parameter incompatibility, NaN explosion
        etc.).
    method_drilldown
        If ``True`` (default), add per-method-axis drill-down columns
        showing how often each method axis (``univariate_passed``,
        ``plsda_passed``, …) was active across paths.

    Returns
    -------
    ConsensusResult
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1]; got {threshold}")

    t0 = time.time()
    if include_failed_paths:
        results = list(path_results)
    else:
        results = [r for r in path_results if r.succeeded and r.shortlist is not None]
    n_paths = len(results)
    if n_paths == 0:
        raise ValueError(
            "No succeeded paths in input — cannot compute consensus."
        )
    path_names = [r.spec.name for r in results]

    # 1. Build the master feature index = union of feature_ids across paths
    feature_id_sets: list[set[str]] = []
    for r in results:
        if r.shortlist is None:
            feature_id_sets.append(set())
            continue
        feature_id_sets.append(set(r.shortlist.table["feature_id"].astype(str)))
    all_features = sorted(set().union(*feature_id_sets))

    # 2. Per-feature × per-path flag matrix
    flag_matrix = np.zeros((len(all_features), n_paths), dtype=bool)
    feat_to_idx = {fid: i for i, fid in enumerate(all_features)}
    for j, r in enumerate(results):
        if r.shortlist is None:
            continue
        sub = r.shortlist.table.loc[r.shortlist.table["passed_shortlist"]]
        for fid in sub["feature_id"].astype(str):
            i = feat_to_idx.get(fid)
            if i is not None:
                flag_matrix[i, j] = True

    n_flagging = flag_matrix.sum(axis=1)
    frac = n_flagging / float(n_paths)
    passed = frac >= threshold

    # 3. Build the consensus table
    table = pd.DataFrame(
        {
            "feature_id": all_features,
            "n_paths_total": np.full(len(all_features), n_paths, dtype=int),
            "n_paths_flagging": n_flagging,
            "frac_paths_flagging": frac,
        }
    )

    # Per-path columns
    for j, r in enumerate(results):
        col = f"per_path_flag_{r.spec.name}"
        table[col] = flag_matrix[:, j]

    # Method drilldown — how many paths flagged this feature via each method axis?
    if method_drilldown:
        method_cols = (
            "univariate_passed", "plsda_passed", "splsda_passed", "rf_passed",
        )
        for col in method_cols:
            counts = np.zeros(len(all_features), dtype=int)
            for r in results:
                if r.shortlist is None:
                    continue
                if col not in r.shortlist.table.columns:
                    continue
                tbl = r.shortlist.table
                fids = tbl["feature_id"].astype(str).to_numpy()
                vals = tbl[col].fillna(False).astype(bool).to_numpy()
                for fid, val in zip(fids, vals, strict=True):
                    if val and fid in feat_to_idx:
                        counts[feat_to_idx[fid]] += 1
            table[f"n_paths_{col.replace('_passed', '')}"] = counts

    # Consensus tier
    table["consensus_tier"] = "none"
    table.loc[(frac > 0) & (frac < 0.5), "consensus_tier"] = "tier_3"
    table.loc[(frac >= 0.5) & (frac < 0.75), "consensus_tier"] = "tier_2"
    table.loc[frac >= 0.75, "consensus_tier"] = "tier_1"

    table["passed_consensus"] = passed

    # Sort
    table = table.sort_values(
        ["passed_consensus", "frac_paths_flagging", "n_paths_flagging", "feature_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    n_per_tier = {
        tier: int((table["consensus_tier"] == tier).sum())
        for tier in ("tier_1", "tier_2", "tier_3", "none")
    }
    n_consensus = int(passed.sum())

    elapsed = time.time() - t0
    log.info(
        "consensus_computed",
        n_paths=n_paths,
        n_features=len(all_features),
        threshold=threshold,
        n_consensus=n_consensus,
        n_per_tier=n_per_tier,
        elapsed_s=round(elapsed, 2),
    )
    return ConsensusResult(
        table=table,
        n_paths_total=n_paths,
        n_consensus=n_consensus,
        n_per_tier=n_per_tier,
        threshold=threshold,
        path_names=path_names,
        runtime_seconds=elapsed,
    )


# ============================================================
# I/O
# ============================================================
def write_consensus(
    result: ConsensusResult,
    output_dir: Any,
    *,
    name: str = "consensus",
) -> dict[str, Any]:
    """Write ``result.table`` to parquet + CSV with config sidecar.

    Output files in ``output_dir``:
      * ``{name}.parquet``
      * ``{name}.csv``
      * ``{name}_passed.parquet`` — rows with ``passed_consensus=True``
      * ``{name}_config.json`` — threshold, n_paths, path_names, n_per_tier
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Any] = {}
    parquet_path = output_dir / f"{name}.parquet"
    csv_path = output_dir / f"{name}.csv"
    result.table.to_parquet(parquet_path, index=False)
    result.table.to_csv(csv_path, index=False)
    paths["parquet"] = parquet_path
    paths["csv"] = csv_path

    passed_path = output_dir / f"{name}_passed.parquet"
    result.table[result.table["passed_consensus"]].to_parquet(passed_path, index=False)
    paths["passed_parquet"] = passed_path

    config_path = output_dir / f"{name}_config.json"
    config_path.write_text(
        json.dumps(
            {
                "threshold": result.threshold,
                "n_paths_total": result.n_paths_total,
                "n_consensus": result.n_consensus,
                "n_per_tier": result.n_per_tier,
                "path_names": result.path_names,
                "runtime_seconds": result.runtime_seconds,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    paths["config"] = config_path

    log.info(
        "consensus_written",
        output_dir=str(output_dir),
        n_features=len(result.table),
        n_consensus=result.n_consensus,
    )
    return paths
