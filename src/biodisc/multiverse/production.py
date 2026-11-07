"""Production multiverse orchestration — Phase 2 (Stages 11–17).

Single source of truth for running the full multiverse end-to-end:

    load Stage-10 matrix → run N paths → consensus → landing page

Both the CLI (``biodisc multiverse run``) and the standalone scripts in
``scripts/`` call into here, so the data loader and the four methodological
paths are defined once. The numerical behaviour is identical to the original
``scripts/run_multiverse_production.py`` — this module simply moves that glue
into the package and makes its parameters configurable.

Why this is NOT a DVC stage
---------------------------
A full run takes hours and is usually launched explicitly (often on a
dedicated server). It is deliberately kept out of the auto ``dvc repro``
DAG so a routine re-run of the cheap Stage 1–10 filters never triggers a
multi-hour job. Run it on demand via ``biodisc multiverse run``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from biodisc.multiverse.consensus import ConsensusResult, compute_consensus, write_consensus
from biodisc.multiverse.io import load_active_feature_ids
from biodisc.multiverse.orchestrator import PathRunResult, PathSpec, run_multiverse
from biodisc.multiverse.reporting import build_multiverse_index

log = structlog.get_logger(__name__)

DEFAULT_INTERIM_DIR = Path("data/interim")
DEFAULT_RUN_ORDER_PATH = Path("data/external/run_order.parquet")
DEFAULT_OUTPUT_DIR = Path("data/reports/multiverse")


# ============================================================
# Data loader (canonical — moved verbatim from scripts/_load_real_data)
# ============================================================
def load_multiverse_data(
    *,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    run_order_path: Path = DEFAULT_RUN_ORDER_PATH,
    outcome_filter: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load (matrix, samples, features, run_order) restricted to Stage-10 survivors.

    The matrix is feature-indexed × sample-columns, in raw / Progenesis units
    (Stage 1 did not transform). Columns are restricted to the sample IDs in
    ``samples.parquet``.

    ``outcome_filter`` (default ``None``) is forwarded to
    :func:`biodisc.multiverse.io.load_active_feature_ids`. ``None`` reads
    ``kept_features.parquet`` directly (all Stage-10 survivors). A subset of
    ``{'endogenous_evidence', 'exo_or_unknown', 'passthrough_l4'}`` restricts
    the input by Stage-10 origin category.
    """
    interim_dir = Path(interim_dir)
    samples = pd.read_parquet(interim_dir / "01_parsed/samples.parquet")
    features_full = pd.read_parquet(interim_dir / "01_parsed/features.parquet")
    abund = pd.read_parquet(interim_dir / "01_parsed/abundances_raw_wide.parquet")
    kept_id_list = load_active_feature_ids(
        interim_dir / "10_origin",
        outcome_filter=outcome_filter,
    )
    run_order = pd.read_parquet(run_order_path)

    # The wide matrix has feature_id as a column AND sample columns. Index it.
    if "feature_id" in abund.columns:
        abund = abund.set_index("feature_id")

    kept_ids = set(kept_id_list)
    matrix = abund.loc[abund.index.astype(str).isin(kept_ids)]
    features = features_full[
        features_full["feature_id"].astype(str).isin(kept_ids)
    ].copy()

    # Restrict columns to the sample IDs we know about
    sample_ids = set(samples["sample_id"].astype(str))
    keep_cols = [c for c in matrix.columns if str(c) in sample_ids]
    matrix = matrix[keep_cols]

    log.info(
        "multiverse_data_loaded",
        n_features=int(matrix.shape[0]),
        n_samples=int(matrix.shape[1]),
        n_run_order=len(run_order),
        outcome_filter=(None if outcome_filter is None else sorted(outcome_filter)),
    )
    return matrix, samples, features, run_order


# ============================================================
# Path specifications
# ============================================================
def build_default_paths() -> list[PathSpec]:
    """The four locked methodological-philosophy paths (production defaults).

    Each path is a complete worldview, not a single-axis swap:
      A classical   — halfmin · QC-RSC · PQN · log2 · Pareto
      B robust ML   — missForest · SERRF · MSTUS · glog · Pareto  (all auto)
      C MNAR-aware  — QRILC · QC-RSC · PQN · glog · Pareto
      D middle      — NS-kNN · SERRF · median · log2 · Pareto
    """
    return [
        PathSpec(
            name="path_a_classical",
            imputation="halfmin",
            drift="qc_rsc",
            normalisation="pqn",
            transformation="log2",
        ),
        PathSpec(
            name="path_b_robust_ml",
            imputation="missforest",
            drift="serrf",
            normalisation="mstus",
            transformation="glog",
        ),
        PathSpec(
            name="path_c_mnar_aware",
            imputation="qrilc",
            drift="qc_rsc",
            normalisation="pqn",
            transformation="glog",
        ),
        PathSpec(
            name="path_d_middle",
            imputation="ns_knn",
            drift="serrf",
            normalisation="median",
            transformation="log2",
        ),
    ]


def build_paths_from_config(paths_cfg: Iterable[Mapping[str, Any]]) -> list[PathSpec]:
    """Build ``PathSpec`` objects from a config list of plain mappings.

    Each entry must define ``name``, ``imputation``, ``drift``,
    ``normalisation``, ``transformation``; ``scaling`` defaults to ``pareto``.
    Optional ``*_params`` mappings pass method kwargs through (empty → the
    method's auto-derived defaults).
    """
    required = ("name", "imputation", "drift", "normalisation", "transformation")
    specs: list[PathSpec] = []
    for i, entry in enumerate(paths_cfg):
        missing = [k for k in required if k not in entry or entry[k] is None]
        if missing:
            raise ValueError(
                f"multiverse path #{i} is missing required field(s): {missing}. "
                f"Got: {dict(entry)}"
            )
        specs.append(
            PathSpec(
                name=str(entry["name"]),
                imputation=str(entry["imputation"]),
                drift=str(entry["drift"]),
                normalisation=str(entry["normalisation"]),
                transformation=str(entry["transformation"]),
                scaling=str(entry.get("scaling", "pareto")),
                imputation_params=dict(entry.get("imputation_params") or {}),
                drift_params=dict(entry.get("drift_params") or {}),
                normalisation_params=dict(entry.get("normalisation_params") or {}),
                transformation_params=dict(entry.get("transformation_params") or {}),
                scaling_params=dict(entry.get("scaling_params") or {}),
            )
        )
    if not specs:
        raise ValueError("multiverse config defines no paths.")
    return specs


# ============================================================
# Full run
# ============================================================
def run(
    *,
    paths: list[PathSpec] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    run_order_path: Path = DEFAULT_RUN_ORDER_PATH,
    n_bootstrap: int = 100,
    n_permutation: int = 1000,
    threshold: float = 0.5,
    outcome_filter: Iterable[str] | None = None,
    seed: int = 42,
    n_threads: int | None = None,
) -> tuple[list[PathRunResult], ConsensusResult]:
    """Run the full multiverse: load → paths → consensus → landing page.

    Parameters
    ----------
    paths
        Path specs to run. ``None`` → :func:`build_default_paths`.
    output_dir
        Root for per-path subdirectories + consensus + ``index.html``.
    n_bootstrap, n_permutation
        Stability / permutation budgets (production: 100 / 1000).
    threshold
        Consensus fraction-of-paths cutoff (0.5 = simple majority).
    outcome_filter
        Stage-10 origin categories to include (``None`` = all survivors).
    seed
        Global deterministic seed.
    n_threads
        If set, cap BLAS/OpenMP threads to this value for the duration of
        the run (prevents desktop oversubscription). ``None`` = no cap.

    Returns
    -------
    (path_results, consensus)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chosen_paths = paths if paths is not None else build_default_paths()

    matrix, samples, features, run_order = load_multiverse_data(
        interim_dir=interim_dir,
        run_order_path=run_order_path,
        outcome_filter=outcome_filter,
    )

    log.info(
        "multiverse_production_start",
        n_paths=len(chosen_paths),
        path_names=[p.name for p in chosen_paths],
        n_bootstrap=n_bootstrap,
        n_permutation=n_permutation,
        threshold=threshold,
        n_threads=n_threads,
        output_dir=str(output_dir),
    )

    def _do_run() -> list[PathRunResult]:
        return run_multiverse(
            chosen_paths,
            matrix=matrix,
            samples=samples,
            features=features,
            run_order=run_order,
            output_dir=output_dir,
            n_bootstrap=n_bootstrap,
            n_permutation=n_permutation,
            seed=seed,
        )

    if n_threads is not None:
        # Runtime BLAS/OpenMP cap — works regardless of import order, unlike
        # env vars which OpenBLAS reads only at import time.
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=int(n_threads)):
            results = _do_run()
    else:
        results = _do_run()

    consensus = compute_consensus(results, threshold=threshold)
    write_consensus(consensus, output_dir)
    build_multiverse_index(
        path_results=results,
        consensus=consensus,
        output_dir=output_dir,
    )
    log.info(
        "multiverse_production_done",
        n_succeeded=sum(r.succeeded for r in results),
        n_paths=len(results),
        n_consensus=consensus.n_consensus,
        n_per_tier=consensus.n_per_tier,
    )
    return results, consensus
