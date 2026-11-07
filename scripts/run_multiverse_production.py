"""Phase 10 — PRODUCTION multiverse run on real Progenesis data.

Identical to ``smoke_real_multiverse.py`` in path design (4 methodological
philosophies), but uses **production-grade settings** throughout:

* ``N_BOOTSTRAP = 100``  (smoke used 30) — Nogueira 2018 stability precision.
* ``N_PERMUTATION = 1000`` (smoke used 100) — drops perm-p floor from
  ``1/(100+1) = 0.0099`` to ``1/(1001) = 0.000999``, giving sPLS-DA / RF
  evidence room to clear ``perm_p < 0.05``.
* ``missForest`` uses **data-aware auto-derived defaults** (no smoke
  override) — ``_auto_n_estimators(n_features)`` and
  ``_auto_max_iter(n_samples)`` from ``methods/impute/missforest.py``.
  At our 4644×47 data: ``n_estimators=123, max_iter=21``. Should converge
  (smoke at 50/10 hit ``converged=False``).

Every other method is fully data-aware-by-default; the path specs use
no explicit method params, so all auto-formulas fire.

Expected runtime: ~10–14 hours sequentially on this hardware. The dominant
cost is missForest (auto, ~5–6 hr) + RF at n_perm=1000 (~80 min × 4
paths). Run overnight; ``index.html`` + ``consensus.parquet`` ready in the
morning.

Output
------
``data/reports/multiverse/`` — same structure as smoke output:
  * one subdirectory per philosophy with full HTML report + shortlist parquet
  * top-level ``consensus.parquet`` / ``.csv`` / ``_passed.parquet`` /
    ``_config.json``
  * ``index.html`` — Phase 8e.7 landing page

Audit
-----
``.venv/bin/python scripts/verify_multiverse.py`` runs in ~10 s and verifies
46 invariants (per-path file presence, shortlist-vs-consensus row-by-row
fidelity, tier definitions, threshold logic, index page content).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

# CRITICAL: cap BLAS thread counts BEFORE numpy / pandas / sklearn are imported.
# Prevents nested oversubscription (sklearn n_jobs=N × BLAS threads=N → N²
# threads). Combined with missForest's `_auto_n_jobs()` (leaves 4 cores for
# OS), this keeps the desktop responsive during overnight runs.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from biodisc.multiverse.consensus import compute_consensus, write_consensus  # noqa: E402
from biodisc.multiverse.orchestrator import PathSpec, run_multiverse  # noqa: E402
from biodisc.multiverse.production import load_multiverse_data  # noqa: E402
from biodisc.multiverse.reporting import build_multiverse_index  # noqa: E402

# Production multivariate settings (locked plan §16)
N_BOOTSTRAP = 100
N_PERMUTATION = 1000

OUTPUT_DIR = Path("data/reports/multiverse")

# ============================================================
# Stage-10 origin outcome filter for multiverse input
# ============================================================
# Restrict which Stage-10 outcome categories feed the multiverse. ``None``
# (default) is backward-compatible: every feature in
# ``data/interim/10_origin/kept_features.parquet`` is used (4,644 features in
# the current pipeline). Pass a subset of the three categories below to run a
# stricter input.
#
# Available outcomes (counts from the current Stage-10 metrics):
#   * 'endogenous_evidence' — ≥1 candidate tagged endogenous in HMDB    (905)
#   * 'exo_or_unknown'      — exogenous-tagged + unannotated mix        (1,211)
#   * 'passthrough_l4'      — no candidate at all (MSI L4 / unidentified) (2,528)
#
# Common settings (run `scripts/diagnose_outcome_filter.py` to preview impact):
#   OUTCOME_FILTER = None                                          # default — all 4,644
#   OUTCOME_FILTER = ("endogenous_evidence", "exo_or_unknown")     # 2,116 — drops L4 unknowns
#   OUTCOME_FILTER = ("endogenous_evidence",)                      #   905 — annotated endogenous only
#
# Choosing a non-None value requires a full multiverse re-run; the consensus
# output in ``data/reports/multiverse/`` is bound to the active input set.
OUTCOME_FILTER: tuple[str, ...] | None = None


def _t() -> str:
    return time.strftime("%H:%M:%S")


def _build_paths_production() -> list[PathSpec]:
    """4 methodological-philosophy paths — production version.

    Thin wrapper around :func:`biodisc.multiverse.production.build_default_paths`,
    which is the single source of truth shared with the ``biodisc multiverse run``
    CLI command. missForest uses NO explicit params so its data-aware auto-defaults
    fire (``n_estimators=auto, max_iter=auto``).
    """
    from biodisc.multiverse import production

    return production.build_default_paths()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print(f"[{_t()}] === PRODUCTION MULTIVERSE RUN ===")
    print(f"[{_t()}] N_BOOTSTRAP={N_BOOTSTRAP}, N_PERMUTATION={N_PERMUTATION}")
    print(f"[{_t()}] missForest will use AUTO defaults "
          "(n_estimators=auto, max_iter=auto)")
    if OUTCOME_FILTER is None:
        print(f"[{_t()}] OUTCOME_FILTER=None (all Stage-10 kept features)")
    else:
        print(f"[{_t()}] OUTCOME_FILTER={tuple(OUTCOME_FILTER)}")
    print()
    print(f"[{_t()}] loading real data...")
    matrix, samples, features, run_order = load_multiverse_data(
        outcome_filter=OUTCOME_FILTER,
    )

    paths = _build_paths_production()
    print(f"[{_t()}] running {len(paths)} philosophy paths:")
    for p in paths:
        explicit = {
            "imp": p.imputation_params, "drift": p.drift_params,
            "norm": p.normalisation_params, "trans": p.transformation_params,
            "scale": p.scaling_params,
        }
        explicit = {k: v for k, v in explicit.items() if v}
        marker = "(all auto)" if not explicit else f"explicit: {explicit}"
        print(f"   - {p.name:<20} {' → '.join(p.chain)}  {marker}")

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
            print(
                f"   ✓ {r.spec.name}: shortlisted "
                f"{r.shortlist.n_shortlisted} in "  # type: ignore[union-attr]
                f"{r.runtime_seconds / 60:.1f} min"
            )
        else:
            print(f"   ✗ {r.spec.name}: failed — {r.error}")

    print(f"\n[{_t()}] computing multiverse consensus (threshold = 0.5)...")
    consensus = compute_consensus(results, threshold=0.5)
    paths_io = write_consensus(consensus, OUTPUT_DIR)

    print(
        f"[{_t()}] consensus: {consensus.n_consensus} features pass "
        f"(≥{consensus.threshold:.0%} of {consensus.n_paths_total} paths)"
    )
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
    print(f"\n[{_t()}] === PRODUCTION DONE in {elapsed / 60:.1f} min "
          f"({elapsed / 3600:.2f} hr) ===")


if __name__ == "__main__":
    main()
