"""8e.8 — render Path A on the real Progenesis-export data.

Pipeline: halfmin → QC-RSC → PQN → log2 → Pareto → outlier → univariate
→ PLS-DA → sPLS-DA → RF → path summary.

Inputs
------
* ``data/interim/01_parsed/`` — Stage 1 parser output (30,370 features × 53
  samples).
* ``data/interim/10_origin/kept_features.parquet`` — feature-id subset
  surviving the upstream filter chain (4,644 features).
* ``data/external/run_order.parquet`` — chronological injection sequence.

Output
------
``data/reports/multiverse/path_a/`` — one full multi-section path report
on real data. The same audit script (``verify_path_a_figures.py``) then
verifies the figures match recomputed values bit-for-bit.

Bootstrap / permutation defaults
--------------------------------
Reduced from production values for fast iteration. The production
multiverse run will use ``n_bootstrap=100, n_permutation=1000`` per the
plan; here we drop to 30 / 100 so the report renders in &lt; 10 minutes.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from biodisc.multiverse import production
from biodisc.multiverse._types import StageContext
from biodisc.multiverse.methods.drift.qc_rsc import correct as qc_rsc_correct
from biodisc.multiverse.methods.impute.halfmin import impute as halfmin_impute
from biodisc.multiverse.methods.multivariate.plsda import fit as plsda_fit
from biodisc.multiverse.methods.multivariate.rf import fit as rf_fit
from biodisc.multiverse.methods.multivariate.splsda import fit as splsda_fit
from biodisc.multiverse.methods.normalise.pqn import normalise as pqn_normalise
from biodisc.multiverse.methods.scale.pareto import scale as pareto_scale
from biodisc.multiverse.methods.transform.log2 import transform as log2_transform
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
from biodisc.multiverse.shortlist import compute_shortlist, default_config, write_shortlist
from biodisc.multiverse.stats_univariate import compute_univariate

# Tuneable: production runs use 100 / 1000; smoke trades fidelity for speed.
N_BOOTSTRAP = 30
N_PERMUTATION = 100

OUTPUT_DIR = Path("data/reports/multiverse/path_a")


def _t() -> str:
    return time.strftime("%H:%M:%S")


def _load_real_data(
    outcome_filter: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load multiverse data using production defaults.

    Thin wrapper around :func:`biodisc.multiverse.production.load_multiverse_data`.
    """
    matrix, samples, features, run_order = production.load_multiverse_data(
        outcome_filter=outcome_filter,
    )
    print(
        f"  loaded: {matrix.shape[0]:,} features × {matrix.shape[1]} samples; "
        f"run_order: {len(run_order)}"
    )
    return matrix, samples, features, run_order


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print(f"[{_t()}] loading real data...")
    matrix, samples, features, run_order = _load_real_data()

    rpt = PathReport(
        name="path_a",
        output_dir=OUTPUT_DIR,
        title="Path A — halfmin → QC-RSC → PQN → log2 → Pareto",
        path_chain=["halfmin", "qc_rsc", "pqn", "log2", "pareto"],
    )

    summary = PathSummary()
    summary.attrition_history.append(("input (post-filters)", matrix.shape[0]))

    # ── Imputation ────────────────────────────────────────────────
    print(f"[{_t()}] Stage 1: imputation (halfmin)...")
    t0 = time.time()
    ctx = StageContext(
        matrix=matrix, samples=samples, features=features, run_order=run_order
    )
    imputed = halfmin_impute(ctx, factor=0.5)
    summary.runtime_history.append(("imputation", time.time() - t0))
    summary.attrition_history.append(("imputed", imputed.shape[0]))
    summary.method_choices["imputation"] = "halfmin"
    summary.extra_params["imputation"] = {"factor": 0.5}
    add_imputation_section(
        rpt, before=matrix, after=imputed, samples=samples,
        method="halfmin", parameters={"factor": 0.5, "fallback_value": 0.0},
    )

    # ── Drift ─────────────────────────────────────────────────────
    print(f"[{_t()}] Stage 2: drift correction (QC-RSC)...")
    t0 = time.time()
    ctx_imp = StageContext(
        matrix=imputed, samples=samples, features=features, run_order=run_order
    )
    drift = qc_rsc_correct(ctx_imp, smoothing_factor=None, spline_order=3)
    summary.runtime_history.append(("drift_correct", time.time() - t0))
    summary.attrition_history.append(("drift_corrected", drift.shape[0]))
    summary.method_choices["drift"] = "qc_rsc"
    summary.extra_params["drift"] = {"smoothing_factor": "auto", "spline_order": 3}
    add_drift_section(
        rpt, before=imputed, after=drift, samples=samples, run_order=run_order,
        method="qc_rsc",
        parameters={"smoothing_factor": "auto", "spline_order": 3, "correction_cap": 10.0},
    )

    # ── Normalisation ─────────────────────────────────────────────
    print(f"[{_t()}] Stage 3: normalisation (PQN)...")
    t0 = time.time()
    ctx_drift = StageContext(
        matrix=drift, samples=samples, features=features, run_order=run_order
    )
    norm = pqn_normalise(ctx_drift, reference="auto")
    summary.runtime_history.append(("normalise", time.time() - t0))
    summary.attrition_history.append(("normalised", norm.shape[0]))
    summary.method_choices["normalisation"] = "pqn"
    summary.extra_params["normalisation"] = {"reference": "qc_median (auto)"}
    add_normalisation_section(
        rpt, before=drift, after=norm, samples=samples,
        method="pqn",
        parameters={"reference": "qc_median (auto)", "min_features_for_factor": "auto"},
    )

    # ── Transformation ────────────────────────────────────────────
    print(f"[{_t()}] Stage 4: transformation (log2)...")
    t0 = time.time()
    ctx_norm = StageContext(
        matrix=norm, samples=samples, features=features, run_order=run_order
    )
    transformed = log2_transform(ctx_norm, epsilon=None)
    summary.runtime_history.append(("transform", time.time() - t0))
    summary.attrition_history.append(("transformed", transformed.shape[0]))
    summary.method_choices["transformation"] = "log2"
    summary.extra_params["transformation"] = {"epsilon": "auto"}
    add_transformation_section(
        rpt, before=norm, after=transformed, samples=samples,
        method="log2", parameters={"epsilon": "auto"},
    )

    # ── Scaling ───────────────────────────────────────────────────
    print(f"[{_t()}] Stage 5: scaling (Pareto)...")
    t0 = time.time()
    ctx_tx = StageContext(
        matrix=transformed, samples=samples, features=features, run_order=run_order
    )
    scaled = pareto_scale(ctx_tx)
    summary.runtime_history.append(("scale", time.time() - t0))
    summary.attrition_history.append(("scaled", scaled.shape[0]))
    summary.method_choices["scaling"] = "pareto"
    add_scaling_section(
        rpt, before=transformed, after=scaled, samples=samples,
        method="pareto", parameters={},
    )

    # ── Outlier ──────────────────────────────────────────────────
    print(f"[{_t()}] Stage 6: SIMCA outlier diagnostic...")
    t0 = time.time()
    add_outlier_section(rpt, matrix=scaled, samples=samples)
    summary.runtime_history.append(("outlier", time.time() - t0))

    # ── Univariate ────────────────────────────────────────────────
    print(f"[{_t()}] Stage 7: univariate (ANOVA + BH-FDR + Tukey)...")
    t0 = time.time()
    add_univariate_section(
        rpt, matrix=transformed, samples=samples, tukey_post_hoc="auto",
    )
    summary.runtime_history.append(("univariate", time.time() - t0))
    univariate_result = compute_univariate(
        StageContext(matrix=transformed, samples=samples, features=features),
        is_log_scale=True, tukey_post_hoc="auto",
    )

    # ── Multivariate ──────────────────────────────────────────────
    multi_ctx = StageContext(
        matrix=scaled, samples=samples, features=features, run_order=run_order
    )
    print(f"[{_t()}] Stage 8: PLS-DA (n_bootstrap={N_BOOTSTRAP}, n_perm={N_PERMUTATION})...")
    t0 = time.time()
    plsda_result = plsda_fit(
        multi_ctx, n_bootstrap=N_BOOTSTRAP, n_permutation=N_PERMUTATION,
    )
    summary.runtime_history.append(("plsda", time.time() - t0))
    summary.method_choices["multivariate.plsda"] = "plsda"
    add_plsda_section(rpt, ctx=multi_ctx, result=plsda_result)

    print(f"[{_t()}] Stage 9: sPLS-DA...")
    t0 = time.time()
    splsda_result = splsda_fit(
        multi_ctx, n_bootstrap=N_BOOTSTRAP, n_permutation=N_PERMUTATION,
    )
    summary.runtime_history.append(("splsda", time.time() - t0))
    summary.method_choices["multivariate.splsda"] = "splsda"
    add_splsda_section(rpt, ctx=multi_ctx, result=splsda_result)

    print(f"[{_t()}] Stage 10: Random Forest (slowest stage)...")
    t0 = time.time()
    rf_result = rf_fit(
        multi_ctx, n_bootstrap=N_BOOTSTRAP, n_permutation=N_PERMUTATION,
    )
    summary.runtime_history.append(("rf", time.time() - t0))
    summary.method_choices["multivariate.rf"] = "rf"
    add_rf_section(rpt, ctx=multi_ctx, result=rf_result)

    # ── Path summary ──────────────────────────────────────────────
    print(f"[{_t()}] Stage 11: path summary...")
    add_path_summary_section(
        rpt, summary=summary,
        univariate_result=univariate_result,
        multivariate_results={
            "plsda": plsda_result,
            "splsda": splsda_result,
            "rf": rf_result,
        },
    )

    # ── Phase 9: per-path biomarker shortlist ────────────────────
    print(f"[{_t()}] Phase 9: biomarker shortlist (k=2 of 4 methods)...")
    shortlist_result = compute_shortlist(
        univariate_result=univariate_result,
        multivariate_results={
            "plsda": plsda_result,
            "splsda": splsda_result,
            "rf": rf_result,
        },
        config=default_config(),
    )
    paths = write_shortlist(shortlist_result, OUTPUT_DIR, name="shortlist")
    print(
        f"    n_shortlisted={shortlist_result.n_shortlisted} / "
        f"{shortlist_result.n_input_features}; tiers={shortlist_result.n_per_tier}"
    )
    print(f"    parquet: {paths['parquet']}")

    # ── Render ────────────────────────────────────────────────────
    print(f"[{_t()}] writing report...")
    out_path = rpt.write(export_png=False)
    elapsed = time.time() - t_total
    print(f"\n[{_t()}] DONE in {elapsed/60:.1f} min")
    print(f"    {out_path}")
    print(f"    sections: {len(rpt.sections)}")
    figures_dir = OUTPUT_DIR / "figures"
    print(f"    figures: {len(list(figures_dir.glob('*.html'))) if figures_dir.exists() else 0}")


if __name__ == "__main__":
    main()
