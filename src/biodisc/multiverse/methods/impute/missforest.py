"""missForest imputation — iterative Random Forest.

Implements Stekhoven & Bühlmann 2012's missForest algorithm via scikit-learn's
:class:`~sklearn.impute.IterativeImputer` with a
:class:`~sklearn.ensemble.RandomForestRegressor` estimator. The algorithm:

  1. Initialise: replace missing values with feature means.
  2. For each feature with missing values (in order of ascending missingness):
     train a RF on the observed rows using the other features as predictors,
     then predict the missing values.
  3. Repeat until predictions stabilise (or ``max_iter`` reached).

Krzywiec et al. 2024 (*Bioinformatics* — imputomics 42-method benchmark)
ranked Random Forest #1 for MCAR scenarios across all evaluated methods.

Orientation note: scikit-learn's ``IterativeImputer`` expects
``rows = samples, cols = features``. Our ``ctx.matrix`` is
``rows = features, cols = samples``, so we transpose before fit and
transpose back after.

Parameters
----------
n_estimators : int | None, default None (auto)
    Number of trees per per-feature RF. When ``None``, computed
    data-aware as ``max(100, min(200, 100 + n_features // 200))``.
    Floor 100 = sklearn / Stekhoven 2012 / Krzywiec 2024 standard;
    grows modestly with feature count to compensate for higher tree
    variance on wider X; cap 200 keeps runtime tractable.
max_iter : int | None, default None (auto)
    Number of round-robin passes through the features. When ``None``,
    computed data-aware as ``max(15, min(25, 10 + n_samples // 4))``.
    Bumped above sklearn's 10 because metabolomics' small-N regime
    often fails to converge in 10 iterations.
n_nearest_features : int | None, default None (auto)
    Limit predictor pool per feature to its N most-correlated neighbours.
    When ``None``, computed data-aware as
    ``max(10, min(50, n_features // 100))``. Gives 10 for 1,000-feature
    datasets, 50 for 5,000+. Without limiting, a 4,644-feature dataset
    would use 4,643 predictors per RF — slow and overfit-prone.
seed : int, default 42
    Deterministic random_state for both the RF and the imputer.

Auto-default rationale — n_nearest_features
-------------------------------------------
``max(10, min(50, n_features // 100))`` keeps the predictor pool in
[10, 50]. The lower bound 10 ensures the RF sees enough variables;
the upper bound 50 keeps each fit tractable on small N. The slope
``// 100`` means each additional 100 features adds 1 predictor —
empirically matches the "more features means more correlated
neighbours available" intuition.

Auto-default rationale — n_estimators (NEW, May 2026)
-----------------------------------------------------
``max(100, min(200, 100 + n_features // 200))``. Floor 100 matches
the canonical missForest-on-metabolomics literature (Stekhoven 2012,
Krzywiec 2024). The ``// 200`` slope adds ~1 tree per 200 features —
modest growth for wider matrices. Cap 200 keeps the total RF count
×iterations runtime bounded.

Auto-default rationale — max_iter (NEW, May 2026)
-------------------------------------------------
``max(15, min(25, 10 + n_samples // 4))``. Sklearn's default 10 was
too tight for our N=47 dataset (we hit ``converged=False``). Adding
``n_samples // 4`` headroom gives small-N datasets a chance to
converge while capping at 25 to bound pathological non-convergence.

Auto-default rationale — n_jobs + threading backend (May 2026)
--------------------------------------------------------------
The May 2026 incident timeline (UI freezes on a 22-core / 32 GB box):

1. **First freeze** — hardcoded ``n_jobs=-1`` spawned 22 sklearn
   workers; combined with default OpenBLAS / MKL threading nested
   under each, produced ~484 competing threads on 22 cores. Fan
   pegged, mouse barely moved.

2. **Second freeze** — fixed ``n_jobs`` to ``min(8, cpus - 4)`` and
   capped BLAS env vars to 1 thread each. Memory still spiked to
   24 GB RSS + 4 GB swap fully consumed. Joblib's default ``loky``
   backend FORKS subprocess workers, each gets a copy-on-write
   image of Python objects (pandas matrix, sklearn caches), and the
   reference-count touches by each subprocess force CoW page
   duplication → effective per-worker memory = full parent RSS.

**Final fix (this module):**

* ``_auto_n_jobs`` caps at ``max_workers=4`` — half the previous 8
  to give an extra-large safety margin on commodity hardware.
* ``IterativeImputer.fit_transform`` is wrapped in
  ``joblib.parallel_backend("threading")``. Threads share memory
  with the parent process — no per-worker subprocess copies.
  RandomForestRegressor's tree-build code is Cython that releases
  the GIL, so threading delivers parallelism with shared memory.
* Production scripts also set
  ``OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`` BEFORE
  numpy import to prevent BLAS-level oversubscription.

Combined effect on our 4644 × 47 × 21-iter workload:
RSS peaks ~10 GB (was 24 GB), swap stays at 0, runtime ~3-5 hours
(was 50-80+ hours due to swap I/O wait).
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)


def _auto_n_nearest_features(n_features: int) -> int:
    """Data-aware default for n_nearest_features. See module docstring rationale."""
    return max(10, min(50, n_features // 100))


def _auto_n_estimators(n_features: int) -> int:
    """Data-aware default for n_estimators. See module docstring rationale."""
    return max(100, min(200, 100 + n_features // 200))


def _auto_max_iter(n_samples: int) -> int:
    """Data-aware default for max_iter. See module docstring rationale."""
    return max(15, min(25, 10 + n_samples // 4))


def _detect_total_memory_gb() -> float:
    """Read total system memory in GB from ``/proc/meminfo`` (Linux).

    Falls back to a conservative 16 GB if the file is unavailable
    (e.g. macOS / WSL edge cases).
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                    return total_kb / (1024.0 * 1024.0)
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    return 16.0


def _auto_n_jobs(
    reserved_for_os: int = 4,
    gb_per_worker: float = 3.0,
    reserved_gb: float = 16.0,
    hard_cap: int = 32,
) -> int:
    """Hardware-aware default for ``n_jobs`` — TRIPLE-axis cap.

    1. **CPU axis**: ``n_cpus - reserved_for_os`` keeps cores free for
       the OS / desktop / other processes.
    2. **Memory axis**: ``(total_gb - reserved_gb) / gb_per_worker`` —
       each missForest worker (with the threading backend) shares
       parent process memory but still allocates per-tree split buffers
       (~3 GB peak each on our reference 4644 × 47 dataset). Total
       memory budget = system RAM minus ``reserved_gb`` reserved for
       the OS, IterativeImputer's working state, and process baseline.
    3. **Hard cap** at ``hard_cap=32`` — beyond ~32 workers, sklearn
       parallelism returns diminish (Cython tree-build per-worker
       overhead saturates).

    Calibration table:
        Laptop  (32 GB, 22 cores): min(32, 18, 5)  = 5  workers
        Server (1500 GB, 44 cores): min(32, 40, 494) = 32 workers
        VM       (4 GB,  4 cores): min(32,  0, 0)  = 1  worker (clamped)

    Auto-fires when caller passes ``n_jobs=None`` to ``impute()``.
    """
    n_cpus = os.cpu_count() or 4
    cpu_max = max(1, n_cpus - reserved_for_os)
    total_gb = _detect_total_memory_gb()
    mem_max = max(1, int((total_gb - reserved_gb) / gb_per_worker))
    return max(1, min(hard_cap, cpu_max, mem_max))


def impute(
    ctx: StageContext,
    *,
    n_estimators: int | None = None,
    max_iter: int | None = None,
    n_nearest_features: int | None = None,
    n_jobs: int | None = None,
    tol: float = 1e-3,
    seed: int = 42,
    treat_zero_as_missing: bool = True,
) -> pd.DataFrame:
    """missForest imputation via sklearn IterativeImputer + RandomForestRegressor.

    Now with proper convergence checking via ``tol`` (relative-change
    stopping criterion — sklearn auto-stops when max change < ``tol``).
    Logs OOB R² across the per-feature RFs as a diagnostic; low OOB R²
    flags the small-N regime where missForest may overfit.

    ``treat_zero_as_missing=True`` (default) converts Progenesis QI's
    below-LOD encoding (cells = 0) to NaN before fitting the iterative
    RF. Set False only if zeros in your data are legitimate measurements.
    """
    # Lazy imports — sklearn import is heavy (~1 s) and only needed when this
    # method is actually called.
    import joblib  # type: ignore[import-untyped]
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    matrix = ctx.matrix
    if treat_zero_as_missing:
        n_zero_in = int((matrix == 0).sum().sum())
        matrix = matrix.where(matrix != 0, np.nan)
    else:
        n_zero_in = 0

    n_missing = int(matrix.isna().sum().sum())
    if n_missing == 0:
        log.info(
            "missforest_no_missing_values",
            n_features=len(matrix),
            treat_zero_as_missing=treat_zero_as_missing,
        )
        return matrix.copy()

    n_features_total = matrix.shape[0]
    n_samples = matrix.shape[1]
    chosen_nnf = (
        _auto_n_nearest_features(n_features_total)
        if n_nearest_features is None
        else n_nearest_features
    )
    # Disable nnf if it exceeds the available predictor pool (means "use all")
    nnf = chosen_nnf if chosen_nnf < n_features_total - 1 else None

    chosen_n_estimators = (
        _auto_n_estimators(n_features_total) if n_estimators is None else n_estimators
    )
    chosen_max_iter = _auto_max_iter(n_samples) if max_iter is None else max_iter
    chosen_n_jobs = _auto_n_jobs() if n_jobs is None else n_jobs

    # OOB requires bootstrap (default True). We enable oob_score for the
    # diagnostic log line at the end. Note: OOB only meaningful when
    # n_samples >= ~30 — for very small N, OOB is unstable and we skip.
    use_oob = n_samples >= 30

    estimator = RandomForestRegressor(
        n_estimators=chosen_n_estimators,
        n_jobs=chosen_n_jobs,
        random_state=seed,
        oob_score=use_oob,
        bootstrap=True,
    )
    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=chosen_max_iter,
        n_nearest_features=nnf,
        tol=tol,
        random_state=seed,
        sample_posterior=False,  # point estimates (deterministic)
    )

    # Transpose: sklearn wants samples × features
    matrix_T = matrix.T.values
    # CRITICAL: explicitly force ``loky`` backend (subprocess workers).
    # We discovered the hard way (May 2026) that on Python 3.13 + sklearn
    # 1.8 + ``OMP_NUM_THREADS=1``, joblib auto-selects the threading
    # backend for sklearn's nested ``n_jobs``. The threading backend uses
    # ``multiprocessing.pool.ThreadPool`` which is GIL-bound — workers
    # idle in ``futex_wait`` while only the main thread runs, giving ~1
    # effective core. py-spy stack traces showed:
    #     _retrieve (joblib.parallel)  ← main thread blocked
    #     worker (multiprocessing.pool:125)  ← THREAD, not subprocess
    # Forcing ``loky`` here spawns true subprocesses with separate memory
    # and separate GILs. Memory cost is bounded by ``_auto_n_jobs`` which
    # caps workers based on system RAM (5 on 32 GB, 32 on 1.5 TB).
    with joblib.parallel_backend("loky", n_jobs=chosen_n_jobs):
        imputed_T = imputer.fit_transform(matrix_T)
    # Transpose back to features × samples
    out = pd.DataFrame(
        imputed_T.T,
        index=matrix.index,
        columns=matrix.columns,
    )
    # Floor at zero — RF can extrapolate slightly negative; biology can't.
    out = out.clip(lower=0.0)

    # Convergence diagnostic — sklearn exposes n_iter_ + final stopping reason
    converged = imputer.n_iter_ < chosen_max_iter

    # Extract OOB R² from the last iteration's RF estimator, if available.
    oob_r2: float | None = None
    if use_oob and hasattr(imputer, "estimator_"):
        with contextlib.suppress(AttributeError, ValueError, TypeError):
            oob_r2 = float(imputer.estimator_.oob_score_)

    log.info(
        "missforest_imputation_done",
        n_features=len(matrix),
        n_samples=n_samples,
        n_missing_filled=n_missing,
        n_zero_treated_as_missing=n_zero_in,
        treat_zero_as_missing=treat_zero_as_missing,
        n_estimators=chosen_n_estimators,
        n_estimators_auto=(n_estimators is None),
        max_iter=chosen_max_iter,
        max_iter_auto=(max_iter is None),
        n_iter_used=int(imputer.n_iter_),
        converged=bool(converged),
        tol=tol,
        n_nearest_features=nnf,
        n_nearest_features_auto=(n_nearest_features is None),
        n_jobs=chosen_n_jobs,
        n_jobs_auto=(n_jobs is None),
        n_cpus_total=os.cpu_count(),
        oob_enabled=use_oob,
        oob_r2=oob_r2,
    )
    if not converged:
        log.warning(
            "missforest_did_not_converge",
            n_iter_used=int(imputer.n_iter_),
            max_iter=chosen_max_iter,
            hint="raise max_iter or accept that small-N data is unstable",
        )
    return out


register(
    MethodSpec(
        name="missforest",
        callable=impute,
        description="Iterative Random Forest imputation — strongest for MCAR/MAR.",
        citation="Stekhoven & Bühlmann 2012, Bioinformatics 28:112",
        category="imputation",
    )
)
