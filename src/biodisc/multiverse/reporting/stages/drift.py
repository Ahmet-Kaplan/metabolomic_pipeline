"""Drift-correction section — the mQACC story before/after one method.

Drift correction is the highest-stakes stage in the LC-MS preprocessing
pipeline: a method that wins on QC tightness alone (D-ratio / PMSD)
might be silently destroying biology. The whole reason
``diagnostics.compute_drift_diagnostics`` returns six numbers is so the
report can show all six side-by-side and the reader can decide whether
the trade-off is honest.

Figures provided
----------------
1. **mQACC headline panel** — paired before/after bars for D-ratio, PMSD,
   median QC-CV, biological silhouette. Lower is better for the first
   three; higher is better for silhouette.
2. **QC trajectories** — 8 representative features, run-order vs
   intensity, before (dotted) and after (solid). Reviewers see drift
   visually — slope flattening = correction working.
3. **D-ratio per-feature distribution** — KDE before vs after across the
   feature space. The bulk should shift left.
4. **QC-CV histograms** — before vs after, with the mQACC 30 % threshold
   line.
5. **PCA by run-order on QCs** — QCs alone, coloured by chronological
   position. Pre-correction the points smear along PC1; post-correction
   they cluster tightly with no run-order gradient.
6. **Capped-feature audit** — bar (or note) of how many features had
   their drift estimate clamped by the QC-RSC ``correction_cap``.

Each figure is annotated with the headline metric directly so reviewers
don't need to cross-reference the metric panel.

References
----------
* Broadhurst et al. 2018, *Metabolomics* 14:72 — mQACC framework, D-ratio
  threshold of 0.5.
* Kirwan et al. 2013, *Anal Bioanal Chem* 405:5147 — PCA / QC-tightness
  framing.
* Dunn et al. 2011, *Nat Protoc* 6:1060 — QC-CV ≤ 30 % criterion.
* De Livera et al. 2015, *Anal Chem* 87:3606 — biological silhouette
  guardrail (≤ 5 % relative drop = no overcorrection).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.diagnostics import (
    biological_silhouette,
    d_ratio,
    pmsd,
    qc_cv_median,
    qc_cv_pct_passing,
)
from biodisc.multiverse.reporting.core import PathReport
from biodisc.multiverse.reporting.figures import (
    histogram_overlay,
    kde_overlay,
    paired_metric_bar,
    pca_class_and_qc_ellipses,
    pca_scores_2d,
    run_order_trajectories,
)


def _per_feature_d_ratio(
    matrix: pd.DataFrame, qc_ids: list[str], sample_ids: list[str]
) -> np.ndarray:
    """Per-feature D-ratio = SD(QC) / SD(sample); broadcast median is the headline."""
    qc_cols = [c for c in qc_ids if c in matrix.columns]
    sa_cols = [c for c in sample_ids if c in matrix.columns]
    if not qc_cols or not sa_cols:
        return np.array([])
    qc_sd = matrix[qc_cols].std(axis=1, ddof=1)
    sa_sd = matrix[sa_cols].std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = qc_sd.to_numpy(dtype=float) / sa_sd.replace(0, np.nan).to_numpy(dtype=float)
    return np.asarray(ratio, dtype=float)


def _per_feature_qc_cv(matrix: pd.DataFrame, qc_ids: list[str]) -> np.ndarray:
    """Per-feature CV % across QCs."""
    cols = [c for c in qc_ids if c in matrix.columns]
    if not cols:
        return np.array([])
    sub = matrix[cols]
    means = sub.mean(axis=1)
    sds = sub.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = 100.0 * sds.to_numpy(dtype=float) / means.replace(0, np.nan).to_numpy(dtype=float)
    return np.asarray(cv, dtype=float)


def _select_trajectory_features(
    matrix: pd.DataFrame, qc_ids: list[str], n: int
) -> list[str]:
    """Pick a span of features ordered by QC-CV — low / mid / high — to show drift behaviour."""
    cv = _per_feature_qc_cv(matrix, qc_ids)
    fids = matrix.index.to_numpy()
    finite = np.isfinite(cv)
    if int(finite.sum()) < n:
        return list(fids[:n])
    cv_finite = cv[finite]
    fids_finite = fids[finite]
    order = np.argsort(cv_finite)
    fids_ranked = fids_finite[order]
    # Even span: pick n features at percentiles 5, 20, 35, 50, 65, 80, 95, etc.
    if len(fids_ranked) <= n:
        return list(fids_ranked)
    idx = np.linspace(0, len(fids_ranked) - 1, n).astype(int)
    return [str(fids_ranked[i]) for i in idx]


def add_drift_section(
    report: PathReport,
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    samples: pd.DataFrame,
    run_order: pd.DataFrame | None,
    method: str,
    parameters: dict[str, Any] | None = None,
    section_name: str = "drift",
    n_capped_features: int | None = None,
) -> None:
    """Add the drift-correction section to ``report``.

    Parameters
    ----------
    report
        Accumulator.
    before, after
        Pre / post drift-correction matrices (features × samples).
    samples
        Sample metadata. Must include ``injection_type``, ``cell_line``.
    run_order
        Chronological injection sequence. ``None`` means trajectories are
        plotted against ``column_order`` instead — flagged in the caption.
    method
        Method id (``"qc_rsc"``, ``"serrf"``).
    parameters
        Method parameters. Recorded under "parameters used".
    n_capped_features
        Optional — number of features whose drift estimate was clipped by
        QC-RSC's ``correction_cap``. If supplied (and > 0), surfaces as a
        metric and a quality flag.
    """
    parameters = dict(parameters or {})

    title_map = {
        "qc_rsc": "QC-RSC drift correction (cubic-spline anchored on QC injections)",
        "serrf": "SERRF — Systematic Error Removal via Random Forest",
    }
    description_map = {
        "qc_rsc": (
            "Per-feature cubic spline through QC intensities at their run-order positions, "
            "evaluated at every sample's position to estimate drift, then divided out "
            "(Kirwan 2013, <em>Analyst</em> 138:5337). The transparent gold standard for "
            "single-batch acquisitions."
        ),
        "serrf": (
            "Random-forest model of per-feature QC trends across the run, with "
            "feature-feature correlations as predictors (Fan 2019, <em>Anal Chem</em> "
            "91:3590). Flexible but opaque — the silhouette guardrail matters more here."
        ),
    }

    report.add_section(
        section_name,
        title=title_map.get(method, f"Drift correction — {method}"),
        description=description_map.get(method, ""),
        method=method,
        parameters=parameters,
    )

    qc_ids = samples.loc[samples["injection_type"] == "qc", "sample_id"].tolist()
    bio_ids = samples.loc[samples["injection_type"] == "biological", "sample_id"].tolist()
    qc_ids = [s for s in qc_ids if s in after.columns]
    bio_ids = [s for s in bio_ids if s in after.columns]

    # ── Headline mQACC metrics (before vs after) ──────────────────
    metrics_pairs = [
        ("D-ratio", d_ratio(before, qc_ids, bio_ids), d_ratio(after, qc_ids, bio_ids)),
        ("PMSD", pmsd(before, qc_ids), pmsd(after, qc_ids)),
        ("median QC-CV %", qc_cv_median(before, qc_ids), qc_cv_median(after, qc_ids)),
        (
            "biological silhouette",
            biological_silhouette(before, samples),
            biological_silhouette(after, samples),
        ),
        (
            "% features QC-CV ≤ 30",
            qc_cv_pct_passing(before, qc_ids, 30.0) * 100,
            qc_cv_pct_passing(after, qc_ids, 30.0) * 100,
        ),
    ]
    for label, before_v, after_v in metrics_pairs:
        report.add_metric(
            section_name,
            f"{label} (before)",
            round(float(before_v), 4) if np.isfinite(before_v) else "n/a",
        )
        report.add_metric(
            section_name,
            f"{label} (after)",
            round(float(after_v), 4) if np.isfinite(after_v) else "n/a",
        )

    if n_capped_features is not None:
        report.add_metric(
            section_name,
            "n_features_drift_capped",
            int(n_capped_features),
            unit="features",
            description="features whose drift estimate hit the correction_cap (spline overshoot)",
        )

    # ── Figure 1: paired metric bars ──────────────────────────────
    metric_fig = paired_metric_bar(
        [m[0] for m in metrics_pairs[:4]],  # bars only for first 4 — % feat passing has different scale
        [float(m[1]) if np.isfinite(m[1]) else 0.0 for m in metrics_pairs[:4]],
        [float(m[2]) if np.isfinite(m[2]) else 0.0 for m in metrics_pairs[:4]],
        title="mQACC metric panel — before vs after correction",
        yaxis_title="metric value",
    )
    report.add_figure(
        section_name,
        title="mQACC metric panel",
        fig=metric_fig,
        caption=(
            "Headline drift-quality metrics. Lower is better for D-ratio (Broadhurst 2018: "
            "&lt; 0.5 acceptable, &lt; 0.4 good), PMSD, and median QC-CV. Higher is better "
            "for biological silhouette — if it drops by &gt; 5 % the correction is "
            "suspected of erasing biology (De Livera 2015 guardrail)."
        ),
        method_metric="D-ratio · PMSD · QC-CV · silhouette (Broadhurst 2018, Kirwan 2013)",
    )

    # ── Figure 2: QC trajectories (visual proof of drift) ─────────
    if run_order is not None:
        traj_features = _select_trajectory_features(before, qc_ids, n=8)
        traj_fig = run_order_trajectories(
            matrix_before=before,
            matrix_after=after,
            samples=samples,
            run_order=run_order,
            feature_ids=traj_features,
            title="QC drift trajectories — 8 features (dotted = before, solid = after)",
            yaxis_title="intensity (raw)",
        )
        report.add_figure(
            section_name,
            title="Run-order trajectories for 8 representative features",
            fig=traj_fig,
            caption=(
                "Eight features sampled across the QC-CV percentile range. Dotted line "
                "is pre-correction, solid is post-correction. Vertical grey hairlines "
                "mark QC injection positions. Successful correction = the post-trace "
                "is flat across the run; pre-trace shows a slope or curve."
            ),
            method_metric="run-order visual diagnostic",
        )
    else:
        report.add_metric(
            section_name,
            "run_order_available",
            "no",
            description="trajectories skipped — no run_order provided",
        )

    # ── Figure 3: D-ratio per-feature distribution ────────────────
    d_before = _per_feature_d_ratio(before, qc_ids, bio_ids)
    d_after = _per_feature_d_ratio(after, qc_ids, bio_ids)
    finite_b = d_before[np.isfinite(d_before)]
    finite_a = d_after[np.isfinite(d_after)]
    # Clip extreme tail values for plotting only — avoids one feature with sample SD≈0
    if finite_b.size > 0:
        finite_b = finite_b[finite_b <= np.nanpercentile(finite_b, 99)]
    if finite_a.size > 0:
        finite_a = finite_a[finite_a <= np.nanpercentile(finite_a, 99)]
    if finite_b.size > 0 or finite_a.size > 0:
        d_fig = kde_overlay(
            [finite_b, finite_a],
            ["before", "after"],
            title="Per-feature D-ratio distribution — before vs after",
            xaxis_title="D-ratio (SD_QC / SD_sample)",
            colors=["#999999", "#0072B2"],
        )
        # Add reference line at 0.5 (Broadhurst threshold)
        d_fig.add_vline(
            x=0.5,
            line={"color": "#D55E00", "width": 1.5, "dash": "dash"},
            annotation={
                "text": "mQACC threshold 0.5",
                "font": {"size": 10, "color": "#D55E00"},
                "bgcolor": "rgba(255,255,255,0.8)",
            },
            annotation_position="top right",
        )
        report.add_figure(
            section_name,
            title="Per-feature D-ratio distribution",
            fig=d_fig,
            caption=(
                "The bulk of features should shift left (lower D-ratio) post-correction. "
                "Mass to the right of the 0.5 threshold = features still drift-dominated. "
                "Top 1 % truncated for readability."
            ),
            method_metric="D-ratio distribution (Broadhurst 2018)",
        )

    # ── Figure 4: QC-CV histograms before/after ───────────────────
    cv_before = _per_feature_qc_cv(before, qc_ids)
    cv_after = _per_feature_qc_cv(after, qc_ids)
    cv_before_clip = cv_before[np.isfinite(cv_before)]
    cv_after_clip = cv_after[np.isfinite(cv_after)]
    if cv_before_clip.size > 0:
        cv_before_clip = cv_before_clip[cv_before_clip <= 200.0]
    if cv_after_clip.size > 0:
        cv_after_clip = cv_after_clip[cv_after_clip <= 200.0]
    if cv_before_clip.size > 0 or cv_after_clip.size > 0:
        cv_fig = histogram_overlay(
            [cv_before_clip, cv_after_clip],
            ["before", "after"],
            title="Per-feature QC %CV — before vs after",
            xaxis_title="QC-CV %",
            colors=["#999999", "#0072B2"],
            nbins=60,
        )
        cv_fig.add_vline(
            x=30.0,
            line={"color": "#D55E00", "width": 1.5, "dash": "dash"},
            annotation={"text": "mQACC 30 %", "font": {"color": "#D55E00", "size": 10}},
            annotation_position="top right",
        )
        report.add_figure(
            section_name,
            title="Per-feature QC %CV distribution",
            fig=cv_fig,
            caption=(
                "Per-feature coefficient of variation across the QC pool. Mass below 30 % "
                "= drift-corrected features that are also analytically reproducible (the "
                "two-pass mQACC criterion). Tail clipped to 200 % for readability."
            ),
            method_metric="QC-CV distribution (Dunn 2011)",
        )

    # ── Figure 5: PCA on QCs before / after, colour by run-order ──
    if len(qc_ids) >= 4:
        # Build a coloured-by-position view if run_order available
        if run_order is not None and "raw_sample_id" in samples.columns:
            # Inject position into samples for the PCA helper to color by
            pos_map = run_order.set_index("raw_sample_id")["position"].to_dict()
            samples_extra = samples.copy()
            samples_extra["run_position"] = (
                samples_extra["raw_sample_id"].map(pos_map).fillna(-1).astype(int)
            )
            color_col = "run_position"
        else:
            samples_extra = samples
            color_col = "injection_type"

        for label, mat in (("before", before), ("after", after)):
            qc_only = mat[[c for c in qc_ids if c in mat.columns]]
            if qc_only.shape[1] < 3:
                continue
            pca_fig = pca_scores_2d(
                qc_only,
                samples_extra,
                title=f"QC-only PCA — {label} correction",
                color_by=color_col,
                add_class_ellipses=False,
            )
            report.add_figure(
                section_name,
                title=f"QC-only PCA — {label}",
                fig=pca_fig,
                caption=(
                    "PCA fit on the QC pool only. "
                    + ("Pre-correction: " if label == "before" else "Post-correction: ")
                    + (
                        "QCs typically smear along PC1 if drift is unaddressed."
                        if label == "before"
                        else "QCs should cluster tightly with no run-order gradient — "
                        "cluster spread = residual drift."
                    )
                ),
                method_metric="QC-only PC1×PC2 (Kirwan 2013)",
            )

    # ── Figures 6 + 7: full-sample PCA with class + QC ellipses ───
    # Shows the WHOLE picture (cell lines + QC + blank + dilution) before
    # vs after drift correction. Three things become visible at once:
    #   1. Cell-line ellipses get more separated (drift across run reduces
    #      noise variance, so PC1/PC2 reflect more biology, less drift).
    #   2. QC ellipse shrinks (the headline drift-correction win).
    #   3. Blank / dilution clusters stay where they are — they're not in
    #      the QC-anchored model but they are visible for sanity.
    for label, mat in (("before", before), ("after", after)):
        if mat.shape[1] < 4:
            continue
        full_pca_fig = pca_class_and_qc_ellipses(
            mat,
            samples,
            title=f"Full-sample PCA — {label} drift correction (class + QC ellipses)",
            use_log1p=True,
        )
        report.add_figure(
            section_name,
            title=f"Full-sample PCA with class + QC ellipses — {label}",
            fig=full_pca_fig,
            caption=(
                "Full sample set (biological cell lines + QC pool + blank "
                "+ dilution) projected onto PC1 × PC2. Each class gets a "
                "95 % confidence ellipse (Kirwan 2013). Compare "
                + (
                    "with the previous figure for"
                    if label == "after"
                    else "this against the next figure to see"
                )
                + " the QC (grey) ellipse: tightening = drift correction "
                "working; biology ellipses should remain visibly separated "
                "(no overcorrection)."
            ),
            method_metric="Full-sample PC1×PC2 (Kirwan 2013)",
        )


def add_drift_section_from_context(
    report: PathReport,
    *,
    ctx_before: StageContext,
    ctx_after: StageContext,
    method: str,
    parameters: dict[str, Any] | None = None,
    section_name: str = "drift",
    n_capped_features: int | None = None,
) -> None:
    """Convenience wrapper: pass two StageContexts instead of unpacking everything."""
    add_drift_section(
        report,
        before=ctx_before.matrix,
        after=ctx_after.matrix,
        samples=ctx_after.samples,
        run_order=ctx_after.run_order,
        method=method,
        parameters=parameters,
        section_name=section_name,
        n_capped_features=n_capped_features,
    )
