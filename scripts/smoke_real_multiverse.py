"""Phase 10 — multiverse smoke on real Progenesis data (4-philosophy design).

Runs 4 paths that each represent a complete **methodological philosophy**
— not single-axis swaps off a baseline. A feature surviving all 4 paths
is robust to the entire analytical worldview, not just one parameter
choice. This is the locked design (agreed with Saeed):

* path_a_classical  — Classical mQACC textbook
    halfmin · QC-RSC · PQN · log2
    (the workflow most papers default to; simplest possible methods)

* path_b_robust_ml  — Robust ML / tree-based throughout
    missForest · SERRF · MSTUS · glog
    (every step uses a robust / ML-style method; trees handle non-
    linearity, MSTUS is robust to outliers, glog stabilises variance)

* path_c_mnar_aware — MNAR-aware (zero / below-LOD friendly)
    QRILC · QC-RSC · PQN · glog
    (QRILC explicitly samples from the LOD distribution; glog avoids
    log2-floor artefacts on imputed zeros — see Finding #9 of the
    manual audit)

* path_d_middle     — Middle ground (hybrid choices)
    NS-kNN · SERRF · median · log2
    (NS-kNN combines KNN and below-LOD sampling; median is the
    simplest sensible normaliser; together a hybrid worldview)

Coverage of the locked benchmarks (plan §1)
-------------------------------------------
* imputation: 4/4 of {halfmin, missforest, qrilc, ns_knn} represented
* drift: both {qc_rsc, serrf}
* normalisation: 3/3 of {pqn, mstus, median}
* transformation: both {log2, glog}
* scaling: pareto (only registered option)

Output
------
``data/reports/multiverse/`` contains:
  * one subdirectory per path with full HTML report + shortlist parquet
  * ``consensus.parquet`` / ``consensus.csv`` / ``consensus_passed.parquet``
    / ``consensus_config.json`` — multiverse consensus across the 4 paths
  * ``index.html`` — top-level landing page (Phase 8e.7)
"""

from __future__ import annotations

import time
from pathlib import Path

from biodisc.multiverse.consensus import compute_consensus, write_consensus
from biodisc.multiverse.orchestrator import PathSpec, run_multiverse
from biodisc.multiverse.production import load_multiverse_data
from biodisc.multiverse.reporting import build_multiverse_index

# Reduced bootstrap / permutation counts for fast iteration. Production runs
# should use 100 / 1000 per the plan (recommended for the 1000-permutation
# null in Westerhuis 2008's framework).
N_BOOTSTRAP = 30
N_PERMUTATION = 100

# Smoke-only override for missForest. The production-grade auto defaults
# are ``n_estimators=auto≈123, max_iter=auto≈21`` for our 4644×47 data —
# correct for the production rerun (~5 hr per call) but too slow for a
# pipeline-validation smoke. These smaller values exercise the full
# missForest code path in ~1 hour instead. **Production rerun must drop
# this dict** so auto defaults kick in.
_SMOKE_MISSFOREST_PARAMS: dict[str, int] = {"n_estimators": 50, "max_iter": 10}

OUTPUT_DIR = Path("data/reports/multiverse")


def _t() -> str:
    return time.strftime("%H:%M:%S")


def _build_paths() -> list[PathSpec]:
    """4 methodological-philosophy paths (locked design).

    See module docstring for the rationale of each philosophy.
    """
    return [
        # A — Classical mQACC textbook
        PathSpec(
            name="path_a_classical",
            imputation="halfmin",
            drift="qc_rsc",
            normalisation="pqn",
            transformation="log2",
        ),
        # B — Robust ML / tree-based throughout
        PathSpec(
            name="path_b_robust_ml",
            imputation="missforest",
            imputation_params=_SMOKE_MISSFOREST_PARAMS,  # smoke-only override
            drift="serrf",
            normalisation="mstus",
            transformation="glog",
        ),
        # C — MNAR-aware (zero/below-LOD friendly)
        PathSpec(
            name="path_c_mnar_aware",
            imputation="qrilc",
            drift="qc_rsc",
            normalisation="pqn",
            transformation="glog",
        ),
        # D — Middle ground (hybrid choices)
        PathSpec(
            name="path_d_middle",
            imputation="ns_knn",
            drift="serrf",
            normalisation="median",
            transformation="log2",
        ),
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print(f"[{_t()}] loading real data...")
    matrix, samples, features, run_order = load_multiverse_data()

    paths = _build_paths()
    print(f"[{_t()}] running {len(paths)} paths (n_bootstrap={N_BOOTSTRAP}, "
          f"n_perm={N_PERMUTATION})...")
    for p in paths:
        print(f"   - {p.name}: {' → '.join(p.chain)}")

    results = run_multiverse(
        paths,
        matrix=matrix,
        samples=samples,
        features=features,
        run_order=run_order,
        output_dir=OUTPUT_DIR,
        n_bootstrap=N_BOOTSTRAP,
        n_permutation=N_PERMUTATION,
    )

    n_ok = sum(r.succeeded for r in results)
    print(f"\n[{_t()}] paths complete: {n_ok}/{len(results)} succeeded")
    for r in results:
        if r.succeeded:
            print(f"   ✓ {r.spec.name}: shortlisted {r.shortlist.n_shortlisted} "  # type: ignore[union-attr]
                  f"in {r.runtime_seconds / 60:.1f} min")
        else:
            print(f"   ✗ {r.spec.name}: failed — {r.error}")

    print(f"\n[{_t()}] computing multiverse consensus (threshold = 0.5)...")
    consensus = compute_consensus(results, threshold=0.5)
    paths_io = write_consensus(consensus, OUTPUT_DIR)

    print(f"[{_t()}] consensus: {consensus.n_consensus} features pass "
          f"(≥{consensus.threshold:.0%} of {consensus.n_paths_total} paths)")
    print(f"   tiers: {consensus.n_per_tier}")
    print(f"   parquet: {paths_io['parquet']}")
    print(f"   passed parquet: {paths_io['passed_parquet']}")

    print(f"\n[{_t()}] building multiverse index page...")
    index_path = build_multiverse_index(
        path_results=results,
        consensus=consensus,
        output_dir=OUTPUT_DIR,
    )
    print(f"   landing page: {index_path}")

    elapsed = time.time() - t_total
    print(f"\n[{_t()}] DONE in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
