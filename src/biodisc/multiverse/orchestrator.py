"""Phase 10 — Multiverse orchestrator.

Runs N parallel paths through the preprocessing/discovery pipeline, each
defined by a fixed combination of method choices (imputation × drift ×
normalisation × transformation × scaling). Multivariate methods (PLS-DA,
sPLS-DA, RF) are run on every path. Each path produces:

* a full per-path HTML report (Stages 1–11, Phase 9 shortlist)
* a per-path shortlist parquet at ``<path_dir>/shortlist.parquet``
* a ``PathRunResult`` returned in-memory for the consensus aggregator

The multiverse design (Steegen 2016, Simonsohn 2020) makes biomarker
discovery robust to methodological choices: a feature flagged across
many paths is unlikely to be an artefact of any single method's
pathology. The consensus aggregator (``compute_consensus`` in
``consensus.py``) takes the per-path shortlists and computes which
features survived ≥ M-of-N paths.

Path enumeration
----------------
Following the plan §1, the candidate-method axes are:

* imputation: ``halfmin``, ``qrilc``, ``ns_knn``, ``missforest``
* drift: ``qc_rsc``, ``serrf``
* normalisation: ``pqn``, ``mstus``, ``median``
* transformation: ``log2``, ``glog``
* scaling: ``pareto``

Cartesian product = 4 × 2 × 3 × 2 × 1 = 48 paths. Each takes ~25 min
on the real data at production bootstrap counts (100 / 1000), so a
full sweep is ~20 hours sequentially — practical with day-long DVC
runs but not for fast iteration. ``run_multiverse`` therefore accepts
an explicit list of paths so you can pick a small subset for
exploration and the full grid for the final run.

References
----------
* Steegen S et al. 2016. *Increasing transparency through a multiverse
  analysis.* **Perspect Psychol Sci** 11:702.
* Simonsohn U et al. 2020. *Specification curve analysis.* **Nat Hum
  Behav** 4:1208.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.methods.multivariate.plsda import fit as plsda_fit
from biodisc.multiverse.methods.multivariate.rf import fit as rf_fit
from biodisc.multiverse.methods.multivariate.splsda import fit as splsda_fit
from biodisc.multiverse.multivariate_utils import MultivariateResult
from biodisc.multiverse.registry import apply
from biodisc.multiverse.reporting import (
    PathReport,
    PathSummary,
    add_drift_section,
    add_imputation_section,
    add_normalisation_section,
    add_outlier_section,
    add_path_summary_section,
    add_plsda_section,
    add_rf_section,
    add_scaling_section,
    add_splsda_section,
    add_transformation_section,
    add_univariate_section,
)
from biodisc.multiverse.shortlist import (
    ShortlistConfig,
    ShortlistResult,
    compute_shortlist,
    default_config,
    write_shortlist,
)
from biodisc.multiverse.stats_univariate import UnivariateResult, compute_univariate

log = structlog.get_logger(__name__)


# ============================================================
# Path specification
# ============================================================
@dataclass(frozen=True)
class PathSpec:
    """One pipeline path through the multiverse.

    A path is a tuple of method choices. The orchestrator invokes each
    method via the global registry (``biodisc.multiverse.registry.apply``)
    so adding a new method to a category requires only registering it.

    Attributes
    ----------
    name
        Human-readable / filesystem-safe path identifier (e.g.
        ``"halfmin_qc_rsc_pqn_log2_pareto"``). Used as the report's
        directory name and the path id in the consensus aggregator.
    imputation
        Registry name (e.g. ``"halfmin"``).
    drift, normalisation, transformation, scaling
        Same convention.
    *_params
        Optional kwarg dicts passed through to each method. ``None`` /
        empty defaults to the method's auto-derived parameters.
    """

    name: str
    imputation: str
    drift: str
    normalisation: str
    transformation: str
    scaling: str = "pareto"
    imputation_params: dict[str, Any] = field(default_factory=dict)
    drift_params: dict[str, Any] = field(default_factory=dict)
    normalisation_params: dict[str, Any] = field(default_factory=dict)
    transformation_params: dict[str, Any] = field(default_factory=dict)
    scaling_params: dict[str, Any] = field(default_factory=dict)

    @property
    def chain(self) -> list[str]:
        """Method chain for the report banner."""
        return [
            self.imputation,
            self.drift,
            self.normalisation,
            self.transformation,
            self.scaling,
        ]


# ============================================================
# Result containers
# ============================================================
@dataclass(frozen=True)
class PathRunResult:
    """Output of one path run.

    Attributes
    ----------
    spec
        The ``PathSpec`` that produced this result.
    shortlist
        ``ShortlistResult`` from Phase 9 — the per-path biomarker list.
    univariate
        Univariate ANOVA + BH-FDR + η² result (passed to consensus
        aggregator for evidence reconstruction).
    multivariate
        Map ``{"plsda": MultivariateResult, ...}``.
    runtime_seconds
        Wall time of ``run_path``.
    output_dir
        Directory containing the per-path HTML report + shortlist
        parquet.
    succeeded
        ``True`` if the path completed; ``False`` if any stage raised.
        ``error`` is set when ``succeeded=False``.
    error
        Human-readable error message when the path failed; ``None``
        otherwise.
    """

    spec: PathSpec
    shortlist: ShortlistResult | None
    univariate: UnivariateResult | None
    multivariate: dict[str, MultivariateResult]
    runtime_seconds: float
    output_dir: Path
    succeeded: bool
    error: str | None = None


# ============================================================
# Single-path execution
# ============================================================
def run_path(
    spec: PathSpec,
    *,
    matrix: pd.DataFrame,
    samples: pd.DataFrame,
    features: pd.DataFrame,
    run_order: pd.DataFrame | None,
    output_dir: Path,
    n_bootstrap: int = 100,
    n_permutation: int = 1000,
    shortlist_config: ShortlistConfig | None = None,
    write_report: bool = True,
    seed: int = 42,
) -> PathRunResult:
    """Execute one path end-to-end. Returns ``PathRunResult``.

    Errors at any stage are captured and surfaced via
    ``PathRunResult.succeeded=False`` so the multiverse orchestrator can
    continue with other paths even when one fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    log.info(
        "path_run_start",
        path=spec.name,
        chain=spec.chain,
        n_bootstrap=n_bootstrap,
        n_permutation=n_permutation,
    )

    rpt = PathReport(
        name=spec.name,
        output_dir=output_dir,
        title=f"Path {spec.name} — {' → '.join(spec.chain)}",
        path_chain=spec.chain,
    ) if write_report else None

    summary = PathSummary()
    summary.attrition_history.append(("input", matrix.shape[0]))

    try:
        # Imputation
        # NOTE: Imputation is applied BEFORE drift correction. This is the
        # reverse of the standard metabolomics workflow (drift-first, then
        # impute). The impact is limited because QC-RSC and SERRF estimate
        # drift from QC samples (which have no missing values), so the drift
        # model is not affected by imputed biological-sample values. However,
        # this ordering should be validated in a sensitivity analysis before
        # any biomarker claim.
        ctx = StageContext(
            matrix=matrix, samples=samples, features=features, run_order=run_order
        )
        t0 = time.time()
        imputed = apply("imputation", spec.imputation, ctx, **spec.imputation_params)
        summary.runtime_history.append(("imputation", time.time() - t0))
        summary.attrition_history.append(("imputed", imputed.shape[0]))
        summary.method_choices["imputation"] = spec.imputation
        summary.extra_params["imputation"] = dict(spec.imputation_params)
        if rpt is not None:
            add_imputation_section(
                rpt, before=matrix, after=imputed, samples=samples,
                method=spec.imputation,
                parameters=dict(spec.imputation_params),
            )

        # Drift
        ctx2 = StageContext(
            matrix=imputed, samples=samples, features=features, run_order=run_order
        )
        t0 = time.time()
        drift = apply("drift", spec.drift, ctx2, **spec.drift_params)
        summary.runtime_history.append(("drift_correct", time.time() - t0))
        summary.attrition_history.append(("drift_corrected", drift.shape[0]))
        summary.method_choices["drift"] = spec.drift
        summary.extra_params["drift"] = dict(spec.drift_params)
        if rpt is not None:
            add_drift_section(
                rpt, before=imputed, after=drift, samples=samples,
                run_order=run_order, method=spec.drift,
                parameters=dict(spec.drift_params),
            )

        # Normalisation
        ctx3 = StageContext(
            matrix=drift, samples=samples, features=features, run_order=run_order
        )
        t0 = time.time()
        norm = apply("normalisation", spec.normalisation, ctx3, **spec.normalisation_params)
        summary.runtime_history.append(("normalise", time.time() - t0))
        summary.attrition_history.append(("normalised", norm.shape[0]))
        summary.method_choices["normalisation"] = spec.normalisation
        summary.extra_params["normalisation"] = dict(spec.normalisation_params)
        if rpt is not None:
            add_normalisation_section(
                rpt, before=drift, after=norm, samples=samples,
                method=spec.normalisation,
                parameters=dict(spec.normalisation_params),
            )

        # Transformation
        ctx4 = StageContext(
            matrix=norm, samples=samples, features=features, run_order=run_order
        )
        t0 = time.time()
        transformed = apply(
            "transformation", spec.transformation, ctx4, **spec.transformation_params
        )
        summary.runtime_history.append(("transform", time.time() - t0))
        summary.attrition_history.append(("transformed", transformed.shape[0]))
        summary.method_choices["transformation"] = spec.transformation
        summary.extra_params["transformation"] = dict(spec.transformation_params)
        if rpt is not None:
            add_transformation_section(
                rpt, before=norm, after=transformed, samples=samples,
                method=spec.transformation,
                parameters=dict(spec.transformation_params),
            )

        # Scaling
        ctx5 = StageContext(
            matrix=transformed, samples=samples, features=features, run_order=run_order
        )
        t0 = time.time()
        scaled = apply("scaling", spec.scaling, ctx5, **spec.scaling_params)
        summary.runtime_history.append(("scale", time.time() - t0))
        summary.attrition_history.append(("scaled", scaled.shape[0]))
        summary.method_choices["scaling"] = spec.scaling
        if rpt is not None:
            add_scaling_section(
                rpt, before=transformed, after=scaled, samples=samples,
                method=spec.scaling, parameters=dict(spec.scaling_params),
            )

        # Outlier
        if rpt is not None:
            t0 = time.time()
            add_outlier_section(rpt, matrix=scaled, samples=samples)
            summary.runtime_history.append(("outlier", time.time() - t0))

        # Univariate (on log-transformed, NOT scaled)
        t0 = time.time()
        univariate_result = compute_univariate(
            StageContext(matrix=transformed, samples=samples, features=features),
            is_log_scale=True, tukey_post_hoc="auto",
        )
        if rpt is not None:
            add_univariate_section(
                rpt, matrix=transformed, samples=samples, tukey_post_hoc="auto",
            )
        summary.runtime_history.append(("univariate", time.time() - t0))

        # Multivariate (on Pareto-scaled)
        multi_ctx = StageContext(
            matrix=scaled, samples=samples, features=features, run_order=run_order
        )
        t0 = time.time()
        plsda_result = plsda_fit(
            multi_ctx, n_bootstrap=n_bootstrap, n_permutation=n_permutation, seed=seed,
        )
        summary.runtime_history.append(("plsda", time.time() - t0))
        summary.method_choices["multivariate.plsda"] = "plsda"
        if rpt is not None:
            add_plsda_section(rpt, ctx=multi_ctx, result=plsda_result)

        t0 = time.time()
        splsda_result = splsda_fit(
            multi_ctx, n_bootstrap=n_bootstrap, n_permutation=n_permutation, seed=seed,
        )
        summary.runtime_history.append(("splsda", time.time() - t0))
        summary.method_choices["multivariate.splsda"] = "splsda"
        if rpt is not None:
            add_splsda_section(rpt, ctx=multi_ctx, result=splsda_result)

        t0 = time.time()
        rf_result = rf_fit(
            multi_ctx, n_bootstrap=n_bootstrap, n_permutation=n_permutation, seed=seed,
        )
        summary.runtime_history.append(("rf", time.time() - t0))
        summary.method_choices["multivariate.rf"] = "rf"
        if rpt is not None:
            add_rf_section(rpt, ctx=multi_ctx, result=rf_result)

        # Path summary
        if rpt is not None:
            add_path_summary_section(
                rpt, summary=summary,
                univariate_result=univariate_result,
                multivariate_results={
                    "plsda": plsda_result,
                    "splsda": splsda_result,
                    "rf": rf_result,
                },
            )

        # Phase 9 shortlist
        shortlist_result = compute_shortlist(
            univariate_result=univariate_result,
            multivariate_results={
                "plsda": plsda_result,
                "splsda": splsda_result,
                "rf": rf_result,
            },
            config=shortlist_config or default_config(),
        )
        write_shortlist(shortlist_result, output_dir, name="shortlist")

        if rpt is not None:
            rpt.write(export_png=False)

        elapsed = time.time() - t_total
        log.info(
            "path_run_done",
            path=spec.name,
            elapsed_s=round(elapsed, 1),
            n_shortlisted=shortlist_result.n_shortlisted,
            n_per_tier=shortlist_result.n_per_tier,
        )
        return PathRunResult(
            spec=spec,
            shortlist=shortlist_result,
            univariate=univariate_result,
            multivariate={
                "plsda": plsda_result,
                "splsda": splsda_result,
                "rf": rf_result,
            },
            runtime_seconds=elapsed,
            output_dir=output_dir,
            succeeded=True,
            error=None,
        )

    except Exception as exc:  # pragma: no cover — defensive, multiverse continues
        elapsed = time.time() - t_total
        log.error(
            "path_run_failed",
            path=spec.name,
            elapsed_s=round(elapsed, 1),
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return PathRunResult(
            spec=spec,
            shortlist=None,
            univariate=None,
            multivariate={},
            runtime_seconds=elapsed,
            output_dir=output_dir,
            succeeded=False,
            error=f"{type(exc).__name__}: {exc}",
        )


# ============================================================
# Multi-path orchestration
# ============================================================
def run_multiverse(
    paths: list[PathSpec],
    *,
    matrix: pd.DataFrame,
    samples: pd.DataFrame,
    features: pd.DataFrame,
    run_order: pd.DataFrame | None,
    output_dir: Path,
    n_bootstrap: int = 100,
    n_permutation: int = 1000,
    shortlist_config: ShortlistConfig | None = None,
    write_report: bool = True,
    seed: int = 42,
) -> list[PathRunResult]:
    """Run all ``paths`` sequentially. Returns list of results.

    Each path runs in its own subdirectory of ``output_dir``. Failures
    in one path don't stop the rest — see ``PathRunResult.succeeded``.

    Sequential execution is the simplest correct behaviour. Parallel
    execution via joblib / multiprocessing can be added later if N >
    handful of paths makes wall-time prohibitive — but joblib's
    fork-based pool combined with sklearn's BLAS-threaded internals
    deadlocks more often than not, so we keep it sequential by default.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "multiverse_run_start",
        n_paths=len(paths),
        path_names=[p.name for p in paths],
        n_bootstrap=n_bootstrap,
        n_permutation=n_permutation,
    )
    t0 = time.time()

    results: list[PathRunResult] = []
    for i, spec in enumerate(paths, start=1):
        log.info(
            "multiverse_path_start",
            i=i,
            n_total=len(paths),
            path=spec.name,
        )
        path_dir = output_dir / spec.name
        result = run_path(
            spec,
            matrix=matrix,
            samples=samples,
            features=features,
            run_order=run_order,
            output_dir=path_dir,
            n_bootstrap=n_bootstrap,
            n_permutation=n_permutation,
            shortlist_config=shortlist_config,
            write_report=write_report,
            seed=seed,
        )
        results.append(result)

    elapsed = time.time() - t0
    n_ok = sum(r.succeeded for r in results)
    log.info(
        "multiverse_run_done",
        n_paths=len(paths),
        n_succeeded=n_ok,
        n_failed=len(paths) - n_ok,
        elapsed_s=round(elapsed, 1),
    )
    return results


# ============================================================
# Path-grid helpers
# ============================================================
def cartesian_paths(
    *,
    imputation: list[str] | None = None,
    drift: list[str] | None = None,
    normalisation: list[str] | None = None,
    transformation: list[str] | None = None,
    scaling: list[str] | None = None,
    name_template: str = "{imputation}_{drift}_{normalisation}_{transformation}_{scaling}",
) -> list[PathSpec]:
    """Enumerate the Cartesian product of method choices into ``PathSpec``s.

    Defaults reflect the multiverse design in plan §1: 4 imputation × 2
    drift × 3 normalisation × 2 transformation × 1 scaling = 48 paths.
    Pass shorter lists to run a focused subset.
    """
    imputation = imputation or ["halfmin", "qrilc", "ns_knn", "missforest"]
    drift = drift or ["qc_rsc", "serrf"]
    normalisation = normalisation or ["pqn", "mstus", "median"]
    transformation = transformation or ["log2", "glog"]
    scaling = scaling or ["pareto"]

    paths: list[PathSpec] = []
    for imp in imputation:
        for d in drift:
            for n in normalisation:
                for t in transformation:
                    for s in scaling:
                        name = name_template.format(
                            imputation=imp, drift=d, normalisation=n,
                            transformation=t, scaling=s,
                        )
                        paths.append(
                            PathSpec(
                                name=name,
                                imputation=imp,
                                drift=d,
                                normalisation=n,
                                transformation=t,
                                scaling=s,
                            )
                        )
    return paths
