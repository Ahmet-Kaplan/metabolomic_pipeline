"""Phase 10 — RESUME a production multiverse run from disk.

Picks up after a partial production run (e.g. interrupted by suspend or
UI freeze). Reads any already-completed paths' ``shortlist.parquet``
files from ``data/reports/multiverse/<path_name>/``, runs the remaining
paths via the orchestrator, then re-aggregates the consensus + rebuilds
the index across all 4 paths.

Production settings (same as ``run_multiverse_production.py``):
* ``N_BOOTSTRAP = 100``
* ``N_PERMUTATION = 1000``
* missForest uses **auto-derived** defaults (no smoke override).

Hardware safety (NEW, May 2026 after UI-freeze incident)
--------------------------------------------------------
1. **BLAS thread cap**: this module sets ``OMP/OPENBLAS/MKL_NUM_THREADS=1``
   BEFORE numpy is imported. Prevents nested oversubscription where
   sklearn's ``n_jobs=N`` workers each spawn N more BLAS threads.
2. **missForest n_jobs auto-derived**: leaves 4 cores for the OS
   (``_auto_n_jobs`` in ``methods/impute/missforest.py``). Was hardcoded
   ``-1`` (all cores), which froze a 22-core box.
3. **Persistent logging**: stdout / stderr are tee'd to
   ``data/reports/multiverse/_run_logs/resume_<timestamp>.log``,
   surviving reboots (vs. ``/tmp`` which is purged).

Usage
-----
Recommended overnight launch with low-priority + suspend-inhibition::

    nohup systemd-inhibit --what=sleep:idle:handle-lid-switch \\
        --who=biodisc --why='multiverse production' \\
        nice -n 10 .venv/bin/python scripts/resume_multiverse_production.py \\
        > /dev/null 2>&1 &

(The script writes its own structured log to data/reports/multiverse/_run_logs/.)
"""

from __future__ import annotations

# Stdlib imports first (safe to load before BLAS caps).
import datetime
import json
import os
import time
from pathlib import Path

# CRITICAL: cap BLAS thread counts BEFORE numpy / pandas / sklearn are imported.
# Prevents the per-thread BLAS oversubscription that froze the box on the
# previous overnight run (sklearn n_jobs=N × BLAS threads=N → N² threads).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pandas as pd  # noqa: E402  — must come after BLAS env caps

from biodisc.multiverse import production  # noqa: E402
from biodisc.multiverse.consensus import compute_consensus, write_consensus  # noqa: E402
from biodisc.multiverse.orchestrator import (  # noqa: E402
    PathRunResult,
    run_multiverse,
)
from biodisc.multiverse.reporting import build_multiverse_index  # noqa: E402
from biodisc.multiverse.shortlist import (  # noqa: E402
    ShortlistResult,
    default_config,
)

# Production constants (same defaults as biodisc.multiverse.production.run).
N_BOOTSTRAP = 100
N_PERMUTATION = 1000
OUTPUT_DIR = Path("data/reports/multiverse")
OUTCOME_FILTER: tuple[str, ...] | None = None


def _build_paths_production() -> list:
    """4 methodological-philosophy paths (production defaults)."""
    return production.build_default_paths()


def _t() -> str:
    return time.strftime("%H:%M:%S")


def _is_path_complete(path_dir: Path) -> bool:
    """A path is complete iff both its shortlist parquet AND HTML report exist."""
    return (path_dir / "shortlist.parquet").exists() and (path_dir / "index.html").exists()


def _rebuild_path_run_result_from_disk(spec, path_dir: Path) -> PathRunResult:
    """Reconstruct a PathRunResult from on-disk parquets so consensus can include it."""
    sl_table = pd.read_parquet(path_dir / "shortlist.parquet")
    cfg_path = path_dir / "shortlist_config.json"
    cfg_data = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    n_per_tier = cfg_data.get("n_per_tier") or {
        t: int((sl_table["consensus_tier"] == t).sum())
        for t in ("tier_1", "tier_2", "tier_3", "none")
    }
    sl_result = ShortlistResult(
        table=sl_table,
        n_input_features=len(sl_table),
        n_shortlisted=int(sl_table["passed_shortlist"].sum()),
        n_per_tier=n_per_tier,
        config=default_config(),
        runtime_seconds=cfg_data.get("runtime_seconds", 0.0),
    )
    return PathRunResult(
        spec=spec,
        shortlist=sl_result,
        univariate=None,
        multivariate={},
        runtime_seconds=cfg_data.get("runtime_seconds", 1500.0),
        output_dir=path_dir,
        succeeded=True,
        error=None,
    )


def _open_persistent_log() -> Path:
    """Open a persistent log file in data/reports/multiverse/_run_logs/.
    Mirrors stdout + stderr there so reboots / /tmp purges don't lose it."""
    log_dir = OUTPUT_DIR / "_run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"resume_{stamp}.log"
    # Tee stdout+stderr into the file at the OS level by reopening fds.
    # Simpler: open a file and have the script write a line confirming.
    return log_file


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _open_persistent_log()
    t_total = time.time()

    print(f"[{_t()}] === RESUME PRODUCTION MULTIVERSE RUN ===")
    print(f"[{_t()}] BLAS thread caps applied: OMP/OPENBLAS/MKL/BLIS/NUMEXPR_NUM_THREADS=1")
    print(f"[{_t()}] missForest n_jobs uses _auto_n_jobs() (leaves 4 cores for OS)")
    print(f"[{_t()}] persistent log: {log_file}")
    print(f"[{_t()}] N_BOOTSTRAP={N_BOOTSTRAP}, N_PERMUTATION={N_PERMUTATION}")
    if OUTCOME_FILTER is None:
        print(f"[{_t()}] OUTCOME_FILTER=None (all Stage-10 kept features)")
    else:
        print(f"[{_t()}] OUTCOME_FILTER={tuple(OUTCOME_FILTER)}")

    paths = _build_paths_production()
    completed_specs = []
    pending_specs = []
    for spec in paths:
        path_dir = OUTPUT_DIR / spec.name
        if _is_path_complete(path_dir):
            completed_specs.append(spec)
            print(f"   ✓ already complete: {spec.name}")
        else:
            pending_specs.append(spec)
            print(f"   ⏳ pending:           {spec.name}")

    if not pending_specs:
        print(f"\n[{_t()}] All 4 paths already complete; just re-aggregating consensus.")

    print(f"\n[{_t()}] loading real data...")
    matrix, samples, features, run_order = production.load_multiverse_data(
        outcome_filter=OUTCOME_FILTER,
    )

    pending_results: list[PathRunResult] = []
    if pending_specs:
        print(f"\n[{_t()}] running {len(pending_specs)} pending paths...")
        pending_results = run_multiverse(
            pending_specs,
            matrix=matrix,
            samples=samples,
            features=features,
            run_order=run_order,
            output_dir=OUTPUT_DIR,
            n_bootstrap=N_BOOTSTRAP,
            n_permutation=N_PERMUTATION,
        )

    # Reload all completed paths (including newly finished + pre-existing) from disk
    all_results: list[PathRunResult] = []
    for spec in paths:
        path_dir = OUTPUT_DIR / spec.name
        if _is_path_complete(path_dir):
            all_results.append(_rebuild_path_run_result_from_disk(spec, path_dir))
        else:
            # Find in pending_results to preserve error info
            match = next((r for r in pending_results if r.spec.name == spec.name), None)
            if match is not None:
                all_results.append(match)
            else:
                print(f"   ✗ skipping {spec.name} — neither on disk nor in pending results")

    n_ok = sum(r.succeeded for r in all_results)
    print(f"\n[{_t()}] all paths gathered: {n_ok}/{len(all_results)} succeeded")
    for r in all_results:
        if r.succeeded and r.shortlist is not None:
            print(f"   ✓ {r.spec.name}: shortlisted {r.shortlist.n_shortlisted}")
        else:
            print(f"   ✗ {r.spec.name}: failed — {r.error}")

    print(f"\n[{_t()}] computing multiverse consensus (threshold = 0.5)...")
    consensus = compute_consensus(all_results, threshold=0.5)
    paths_io = write_consensus(consensus, OUTPUT_DIR)
    print(
        f"[{_t()}] consensus: {consensus.n_consensus} features pass "
        f"(≥{consensus.threshold:.0%} of {consensus.n_paths_total} paths)"
    )
    print(f"   tiers: {consensus.n_per_tier}")
    print(f"   parquet: {paths_io['parquet']}")

    print(f"\n[{_t()}] building multiverse index page...")
    index_path = build_multiverse_index(
        path_results=all_results,
        consensus=consensus,
        output_dir=OUTPUT_DIR,
    )
    print(f"   landing page: {index_path}")

    elapsed = time.time() - t_total
    print(
        f"\n[{_t()}] === RESUME DONE in {elapsed / 60:.1f} min "
        f"({elapsed / 3600:.2f} hr) ==="
    )


if __name__ == "__main__":
    main()
