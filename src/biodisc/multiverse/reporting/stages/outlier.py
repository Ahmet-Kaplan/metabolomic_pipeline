"""Outlier section — Hotelling T² × DModX SIMCA diagnostic + class-ellipse PCA.

Wraps ``biodisc.multiverse.outlier.detect_outliers`` into a publication-grade
report section. The reviewer's job is to decide whether a flagged sample is
worth excluding via ``parse.exclude_samples`` — the section's figures are
designed to make that judgement defensible.

What the section shows
----------------------
1. **T² × DModX scatter** — the headline SIMCA panel. One marker per
   biological sample, x = Hotelling's T² (in-model distance, ``Eriksson
   2006``), y = DModX (out-of-model distance). Critical-value lines at
   the chosen α split the plane into four quadrants:

     - lower-left: well-explained, central → "normal" sample
     - lower-right: extreme position but PC model fits → biological extreme
     - upper-left: shape doesn't fit the PC model but position is normal
       → likely technical (contamination, instrument hiccup)
     - upper-right: comprehensive outlier → almost always investigate

   Markers coloured by ``cell_line``; flagged samples labelled with the
   sample id so reviewers can cross-reference the run-order plot.

2. **PCA scores with 95 % confidence ellipses** (Kirwan 2013). One
   ellipse per biological cell line plus one for the QC pool. The
   ellipses make two things obvious at a glance:

     - QC ellipse should be small and central — confirms drift was
       corrected and normalisation worked.
     - Cell-line ellipses should be separated and roughly the same size
       — confirms biology survived the preprocessing.

   Sample markers are labelled where flagged, and the QC pool is drawn
   in neutral grey so it doesn't visually compete with the biological
   classes.

3. **T² bar** and **DModX bar** — per-sample views of each statistic
   with the critical line drawn for reference. Useful for reading off
   which specific sample is the offender.

4. **Flagged-samples table** — sample_id, cell_line, T², DModX, which
   test flagged it.

References
----------
* Wold S. 1976. *Pattern recognition by means of disjoint principal
  components models.* **Pattern Recognition** 8:127.
* Eriksson L et al. 2006. *Multi- and Megavariate Data Analysis.* Umetrics.
* Kirwan JA et al. 2013, *Anal Bioanal Chem* 405:5147 — class-ellipse PCA
  framing.
* Westerhuis JA et al. 1999. *Hierarchical PCA…* and SIMCA literature
  for the DModX critical-value derivation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.outlier import OutlierResult, detect_outliers
from biodisc.multiverse.reporting.core import PathReport
from biodisc.multiverse.reporting.helpers import confidence_ellipse_2d
from biodisc.multiverse.reporting.theme import apply_theme, color_for, display_class


def _t2_dmodx_scatter(
    res: OutlierResult,
    samples: pd.DataFrame,
    *,
    title: str,
    qc_projection: dict[str, np.ndarray] | None = None,
) -> go.Figure:
    """T² × DModX scatter — markers coloured by display class, critical lines drawn.

    Biological samples come from ``res`` (the PCA model is fit on them so
    flagging is meaningful). When ``qc_projection`` is supplied (typically
    from :func:`_project_qc_onto_outlier_model`), QC samples are added as
    a separate grey-diamond trace projected onto the same biological-fit
    model — visible as a sanity check (QCs should cluster tightly near
    the centroid; widely-scattered QCs flag a poorly-conditioned model).
    """
    samples_idx = samples.set_index("sample_id")
    fig = go.Figure()

    # Build (sample_id, display_class, t2, dmodx, flagged) tuples for biologicals
    rows = []
    for sid, t2, dx, fl in zip(
        res.sample_ids,
        res.t_squared,
        res.dmodx,
        res.flagged_either,
        strict=True,
    ):
        if sid in samples_idx.index:
            cls = display_class(
                samples_idx.loc[sid, "cell_line"],
                samples_idx.loc[sid, "injection_type"],
            )
        else:
            cls = "unknown"
        rows.append((sid, cls, float(t2), float(dx), bool(fl)))

    # Group by display class
    by_cl: dict[str, list[tuple[str, float, float, bool]]] = {}
    for sid, cl, t2, dx, fl in rows:
        by_cl.setdefault(cl, []).append((sid, t2, dx, fl))

    for j, (cl, items) in enumerate(sorted(by_cl.items())):
        col = color_for(cl, fallback_index=j)
        sids = [it[0] for it in items]
        t2s = [it[1] for it in items]
        dxs = [it[2] for it in items]
        flags = [it[3] for it in items]
        # Outline flagged points
        line_widths = [2.0 if fl else 0.7 for fl in flags]
        line_colors = ["#D55E00" if fl else "#222" for fl in flags]
        labels = [s if fl else "" for s, fl in zip(sids, flags, strict=True)]
        fig.add_trace(
            go.Scatter(
                x=t2s,
                y=dxs,
                mode="markers+text",
                marker={
                    "color": col,
                    "size": 11,
                    "line": {"width": line_widths, "color": line_colors},
                },
                text=labels,
                textposition="top center",
                textfont={"size": 9, "color": "#444"},
                name=str(cl),
                hovertemplate=(
                    "%{customdata}<br>"
                    f"class: {cl}<br>"
                    "T²: %{x:.3f}<br>DModX: %{y:.3f}<extra></extra>"
                ),
                customdata=sids,
            )
        )

    # QC projections (sanity check: QC pool should cluster tight near centroid)
    if qc_projection is not None and len(qc_projection.get("sample_ids", [])):
        qc_sids = list(qc_projection["sample_ids"])
        qc_t2 = qc_projection["t_squared"].astype(float).tolist()
        qc_dx = qc_projection["dmodx"].astype(float).tolist()
        fig.add_trace(
            go.Scatter(
                x=qc_t2,
                y=qc_dx,
                mode="markers",
                marker={
                    "color": color_for("qc"),
                    "size": 11,
                    "symbol": "diamond",
                    "line": {"width": 0.7, "color": "#222"},
                },
                name="qc (projected)",
                customdata=qc_sids,
                hovertemplate=(
                    "%{customdata}<br>class: qc (projected onto biological model)<br>"
                    "T²: %{x:.3f}<br>DModX: %{y:.3f}<extra></extra>"
                ),
            )
        )

    # Critical-value lines
    if np.isfinite(res.t_squared_critical):
        fig.add_vline(
            x=float(res.t_squared_critical),
            line={"color": "#D55E00", "width": 1.5, "dash": "dash"},
            annotation={
                "text": f"T² crit (α={res.alpha:g})",
                "font": {"color": "#D55E00", "size": 10},
            },
            annotation_position="top right",
        )
    if np.isfinite(res.dmodx_critical):
        fig.add_hline(
            y=float(res.dmodx_critical),
            line={"color": "#D55E00", "width": 1.5, "dash": "dash"},
            annotation={
                "text": f"DModX crit (α={res.alpha:g})",
                "font": {"color": "#D55E00", "size": 10},
            },
            annotation_position="top left",
        )

    apply_theme(
        fig,
        title=title,
        xaxis_title="Hotelling's T² (in-model distance)",
        yaxis_title="DModX (out-of-model distance)",
    )
    return fig


def _project_qc_onto_outlier_model(
    matrix: pd.DataFrame,
    samples: pd.DataFrame,
    bio_result: OutlierResult,
    *,
    use_log1p: bool = False,
) -> dict[str, np.ndarray] | None:
    """Project QC samples onto the biological-fit PCA model and return T²/DModX.

    Why project, not refit-with-QCs
    ------------------------------
    QCs are a pooled mixture aliquoted to be identical, so their
    feature-feature variance structure is artificially compressed.
    Including them in the PCA fit would compress the principal
    components toward the QC mean, biasing the &quot;typical biological
    variance&quot; model used for outlier flagging. Projecting them
    instead — same components, same centroid, same s0 — keeps the
    flagging unbiased while letting reviewers see where QCs land.

    Returns ``None`` if no QC samples are available.
    """
    from sklearn.decomposition import PCA

    bio_ids = list(bio_result.sample_ids)
    qc_ids_all = samples.loc[
        samples["injection_type"] == "qc", "sample_id"
    ].astype(str).tolist()
    qc_ids = [s for s in qc_ids_all if s in matrix.columns]
    if not qc_ids:
        return None

    bio_arr_raw = matrix[bio_ids].T.to_numpy(dtype=float)
    qc_arr_raw = matrix[qc_ids].T.to_numpy(dtype=float)
    if use_log1p and float(np.nanmin(bio_arr_raw)) >= 0:
        bio_arr = np.log1p(bio_arr_raw)
        qc_arr = np.log1p(qc_arr_raw)
    else:
        bio_arr = bio_arr_raw
        qc_arr = qc_arr_raw

    means = bio_arr.mean(axis=0)
    centered = bio_arr - means
    pca = PCA(n_components=bio_result.n_components)
    bio_scores = pca.fit_transform(centered)
    eigvals = np.maximum(pca.explained_variance_, 1e-12)

    # Project QC: subtract biological centroid, then onto components.
    qc_centered = qc_arr - means
    qc_scores = qc_centered @ pca.components_.T

    # T² (Hotelling) using biological-fit eigenvalues
    inv_eigvals = 1.0 / eigvals
    qc_t2 = (qc_scores * qc_scores * inv_eigvals[None, :]).sum(axis=1)

    # DModX requires the same s0 as the biological fit
    n_features = bio_arr.shape[1]
    df_residual = max(1, n_features - bio_result.n_components)
    bio_recon = bio_scores @ pca.components_ + means
    bio_res = bio_arr - bio_recon
    bio_sse_total = (bio_res * bio_res).sum()
    n_bio = len(bio_ids)
    s0_squared = bio_sse_total / max(
        1, (n_bio - bio_result.n_components - 1) * df_residual
    )
    s0 = float(np.sqrt(s0_squared)) if s0_squared > 0 else 1e-12

    qc_recon = qc_scores @ pca.components_ + means
    qc_res = qc_arr - qc_recon
    qc_sse = (qc_res * qc_res).sum(axis=1)
    qc_dmodx = np.sqrt(qc_sse / df_residual) / s0

    return {
        "sample_ids": np.asarray(qc_ids, dtype=object),
        "t_squared": np.asarray(qc_t2, dtype=float),
        "dmodx": np.asarray(qc_dmodx, dtype=float),
    }


def _per_sample_bar(
    res: OutlierResult,
    samples: pd.DataFrame,
    *,
    statistic: str,
    title: str,
    yaxis_title: str,
    qc_projection: dict[str, np.ndarray] | None = None,
) -> go.Figure:
    """Per-sample bar (T² or DModX) with critical-value reference line.

    If ``qc_projection`` is supplied, QC samples are appended as
    additional bars (grouped under "qc") with their projected T²/DModX
    values from the biological-fit model.
    """
    if statistic == "t2":
        values = res.t_squared
        critical = res.t_squared_critical
        flagged = res.flagged_t2
        qc_values = qc_projection["t_squared"] if qc_projection else None
    elif statistic == "dmodx":
        values = res.dmodx
        critical = res.dmodx_critical
        flagged = res.flagged_dmodx
        qc_values = qc_projection["dmodx"] if qc_projection else None
    else:
        raise ValueError(f"unknown statistic {statistic!r}")

    samples_idx = samples.set_index("sample_id")
    fig = go.Figure()
    by_cl: dict[str, list[tuple[str, float, bool]]] = {}
    for sid, v, fl in zip(res.sample_ids, values, flagged, strict=True):
        cls = (
            display_class(
                samples_idx.loc[sid, "cell_line"],
                samples_idx.loc[sid, "injection_type"],
            )
            if sid in samples_idx.index
            else "unknown"
        )
        by_cl.setdefault(cls, []).append((sid, float(v), bool(fl)))

    # Append QC projections under display class "qc" (no flagging — they
    # weren't part of the fit, just projected for visibility)
    if qc_projection is not None and qc_values is not None:
        qc_sids = list(qc_projection["sample_ids"])
        for sid, v in zip(qc_sids, qc_values, strict=True):
            by_cl.setdefault("qc", []).append((str(sid), float(v), False))

    for j, (cl, items) in enumerate(sorted(by_cl.items())):
        col = color_for(cl, fallback_index=j)
        sids = [it[0] for it in items]
        vs = [it[1] for it in items]
        flags = [it[2] for it in items]
        fig.add_trace(
            go.Bar(
                x=sids,
                y=vs,
                marker={
                    "color": col,
                    "line": {
                        "color": ["#D55E00" if f else col for f in flags],
                        "width": [2 if f else 0 for f in flags],
                    },
                },
                name=cl,
                hovertemplate="%{x}<br>" + statistic + ": %{y:.3f}<extra></extra>",
            )
        )

    if np.isfinite(critical):
        fig.add_hline(
            y=float(critical),
            line={"color": "#D55E00", "width": 1.5, "dash": "dash"},
            annotation={
                "text": f"crit (α={res.alpha:g}) = {critical:.3f}",
                "font": {"color": "#D55E00", "size": 10},
            },
            annotation_position="top right",
        )

    apply_theme(fig, title=title, xaxis_title="sample", yaxis_title=yaxis_title)
    fig.update_layout(xaxis={"tickangle": -55, "tickfont": {"size": 9}})
    return fig


def _pca_with_class_and_qc_ellipses(
    matrix: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    title: str,
    flagged_sample_ids: set[str],
    use_log1p: bool = False,
) -> go.Figure:
    """PCA scores plot with 95 % confidence ellipses per cell line and per QC pool.

    Differs from the generic ``pca_scores_2d`` helper in two ways:
      1. **QC samples are included** (other helpers usually drop them) — the
         QC ellipse is the headline drift / normalisation diagnostic.
      2. **Flagged samples get text labels** so the reviewer can read off
         their identity without consulting the table.

    ``use_log1p=False`` by default — outlier detection runs on the
    Pareto-scaled matrix which is already on a stabilised scale; further
    log1p would distort the geometry.
    """
    from sklearn.decomposition import PCA

    cols = [c for c in matrix.columns if c in samples["sample_id"].values]
    if len(cols) < 4:
        fig = go.Figure()
        apply_theme(fig, title=title + " (too few samples)", xaxis_title="PC1", yaxis_title="PC2")
        return fig

    sub = matrix[cols]
    arr = sub.to_numpy(dtype=float)
    if use_log1p and float(np.nanmin(arr)) >= 0:
        arr = np.log1p(arr)
    X = arr.T  # rows = samples, cols = features
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_
    samples_idx = samples.set_index("sample_id")
    # Real-data convention: biological samples have cell_line set, QCs
    # have cell_line = None / NaN. ``display_class`` resolves both to a
    # single per-row label ("MCF7" / "MCF10A" / ... / "qc" / "blank" /
    # "dilution") that maps directly into ``CLASS_COLORS``.
    display_labels = np.array([
        display_class(
            samples_idx.loc[sid, "cell_line"],
            samples_idx.loc[sid, "injection_type"],
        )
        for sid in cols
    ])
    inj_types = samples_idx.loc[cols, "injection_type"].astype(str).to_numpy()

    # Build groups: each unique display class gets its own series + ellipse.
    # Sort biological classes first, then QC / blank / dilution so the
    # neutral-grey QC ellipse is drawn on top of the coloured biological
    # ellipses (last-drawn wins z-order in Plotly).
    _GROUP_ORDER = {"qc": 1000, "blank": 1001, "dilution": 1002}
    unique_groups = sorted(set(display_labels), key=lambda g: (_GROUP_ORDER.get(g, 0), g))
    fig = go.Figure()
    for j, group in enumerate(unique_groups):
        mask = display_labels == group
        if not mask.any():
            continue
        pts_x = scores[mask, 0]
        pts_y = scores[mask, 1]
        col = color_for(group, fallback_index=j)
        sids = np.array(cols)[mask]
        # Label flagged samples with their sample_id; others stay anonymous
        labels = np.array([sid if sid in flagged_sample_ids else "" for sid in sids])
        # Choose marker shape per injection type (uniform within a group)
        sample_inj = inj_types[mask]
        symbol = (
            "circle" if (sample_inj == "biological").all()
            else "diamond" if (sample_inj == "qc").all()
            else "square"
        )
        fig.add_trace(
            go.Scatter(
                x=pts_x,
                y=pts_y,
                mode="markers+text",
                marker={
                    "color": col,
                    "size": 11,
                    "symbol": symbol,
                    "line": {"width": 0.7, "color": "#222"},
                },
                text=labels,
                textposition="top center",
                textfont={"size": 9, "color": "#D55E00"},
                name=group,
                hovertemplate=(
                    "%{customdata}<br>"
                    f"class: {group}<br>"
                    "PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>"
                ),
                customdata=sids,
            )
        )
        # 95 % confidence ellipse if enough points
        if pts_x.size >= 3:
            rx, ry = confidence_ellipse_2d(pts_x, pts_y)
            if rx.size > 0:
                fig.add_trace(
                    go.Scatter(
                        x=rx,
                        y=ry,
                        mode="lines",
                        line={"color": col, "width": 1.5, "dash": "dot"},
                        name=f"{group} 95 % CI",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    apply_theme(
        fig,
        title=title,
        xaxis_title=f"PC1 ({var_ratio[0] * 100:.1f} %)",
        yaxis_title=f"PC2 ({var_ratio[1] * 100:.1f} %)",
    )
    return fig


def add_outlier_section(
    report: PathReport,
    *,
    matrix: pd.DataFrame,
    samples: pd.DataFrame,
    section_name: str = "outlier",
    n_components: int | str = "auto",
    alpha: float = 0.05,
    biological_only: bool = True,
    pca_use_log1p: bool = False,
    project_qc: bool = True,
) -> OutlierResult:
    """Add the outlier-detection section to ``report``. Returns the ``OutlierResult``.

    The matrix passed in must be NaN-free (typically post-Pareto-scaling).
    Outlier detection runs on biological samples only (QC samples cluster
    tightly by design and would compress the &quot;typical biological
    variance&quot; PCA model used for flagging). When ``project_qc=True``
    (default), QC samples are projected onto the same biological-fit
    model and shown alongside biologicals in the T² × DModX scatter +
    per-sample bars — a visual sanity check that the QC pool is tight.

    Returns the ``OutlierResult`` (biological-only) so callers can use
    the flag mask downstream (e.g. for cross-method consensus in Phase 9).
    """
    # Wrap the matrix in a transient StageContext so we can reuse detect_outliers
    ctx = StageContext(
        matrix=matrix,
        samples=samples,
        features=pd.DataFrame({"feature_id": matrix.index}),
        run_order=None,
    )
    res = detect_outliers(
        ctx,
        biological_only=biological_only,
        n_components=n_components,
        alpha=alpha,
    )

    # Project QC samples onto the biological-fit model for visualisation only
    qc_proj = (
        _project_qc_onto_outlier_model(
            matrix, samples, res, use_log1p=pca_use_log1p
        )
        if project_qc and biological_only
        else None
    )

    report.add_section(
        section_name,
        title="SIMCA outlier diagnostic — Hotelling T² + DModX",
        description=(
            "Per-sample outlier flagging from PCA scores (T²) and "
            "residuals (DModX). Samples are <em>flagged for review</em>, "
            "never auto-removed — the user decides whether to add a "
            "flagged sample to <code>parse.exclude_samples</code> and "
            "rerun the pipeline. Includes a class-ellipse PCA "
            "(Kirwan 2013) showing biology + QC structure."
        ),
        method="simca_t2_dmodx",
        parameters={
            "n_components": res.n_components,
            "alpha": res.alpha,
            "biological_only": biological_only,
        },
    )

    report.add_metric(section_name, "n_samples_analysed", len(res.sample_ids))
    report.add_metric(section_name, "n_components", res.n_components)
    report.add_metric(
        section_name,
        "explained_variance_total %",
        round(float(res.explained_variance_ratio.sum()) * 100, 2),
    )
    report.add_metric(
        section_name,
        "T² critical (α)",
        round(float(res.t_squared_critical), 4) if np.isfinite(res.t_squared_critical) else "inf",
    )
    report.add_metric(
        section_name,
        "DModX critical (α)",
        round(float(res.dmodx_critical), 4) if np.isfinite(res.dmodx_critical) else "inf",
    )
    report.add_metric(section_name, "n_outliers_T²_only", int(res.flagged_t2.sum()))
    report.add_metric(section_name, "n_outliers_DModX_only", int(res.flagged_dmodx.sum()))
    report.add_metric(section_name, "n_outliers_either", int(res.flagged_either.sum()))

    # ── Figure 1: T² × DModX scatter (with QC projections) ───────
    scatter_fig = _t2_dmodx_scatter(
        res,
        samples,
        title=f"T² × DModX SIMCA outlier diagnostic ({res.n_components} PCs, α={res.alpha:g})",
        qc_projection=qc_proj,
    )
    report.add_figure(
        section_name,
        title="T² × DModX outlier scatter",
        fig=scatter_fig,
        caption=(
            "Biological samples (circles, coloured by cell line) flagged by "
            "the SIMCA model. QC samples (grey diamonds) are projected onto "
            "the same biological-fit model as a sanity check — they should "
            "cluster tightly in the lower-left quadrant. Critical-value "
            "lines split the plane: <em>lower-left</em> = normal; "
            "<em>lower-right</em> = biological extreme (high T², shape "
            "fits); <em>upper-left</em> = technical artifact (DModX high, "
            "position OK); <em>upper-right</em> = comprehensive outlier. "
            "Flagged biological samples are labelled with their sample id."
        ),
        method_metric="Hotelling T² (Wold 1976) · DModX (Eriksson 2006)",
    )

    # ── Figure 2: PCA with class+QC ellipses (Kirwan 2013) ────────
    flagged_set = set(res.outlier_sample_ids)
    pca_fig = _pca_with_class_and_qc_ellipses(
        matrix,
        samples,
        title="PCA scores with 95 % confidence ellipses (cell lines + QC pool)",
        flagged_sample_ids=flagged_set,
        use_log1p=pca_use_log1p,
    )
    report.add_figure(
        section_name,
        title="PCA scores with class + QC ellipses",
        fig=pca_fig,
        caption=(
            "PC1 × PC2 with one 95 % confidence ellipse per cell line and one "
            "for the QC pool. The QC ellipse should be small and central — "
            "tight QCs confirm the upstream drift correction and normalisation "
            "did their job. Cell-line ellipses should be reasonably "
            "separated and similar in size — confirms biology survived "
            "preprocessing. Flagged outlier samples (from the T² × DModX "
            "panel above) carry their sample id as a label."
        ),
        method_metric="Kirwan 2013 class-ellipse PCA",
    )

    # ── Figure 3: T² bar (per-sample, with QC projections) ───────
    t2_bar = _per_sample_bar(
        res, samples, statistic="t2",
        title="Hotelling's T² per sample",
        yaxis_title="T² value",
        qc_projection=qc_proj,
    )
    report.add_figure(
        section_name,
        title="Hotelling's T² per sample",
        fig=t2_bar,
        caption=(
            "Per-sample T² with the dashed critical line. Bars exceeding "
            "the line are flagged in the scatter above; the colour is the "
            "cell line. QC bars (grey) are projections onto the biological-"
            "fit model — they should sit well below the threshold."
        ),
        method_metric="Hotelling T² (Eriksson 2006)",
    )

    # ── Figure 4: DModX bar (per-sample, with QC projections) ────
    dmodx_bar = _per_sample_bar(
        res, samples, statistic="dmodx",
        title="DModX per sample",
        yaxis_title="DModX value",
        qc_projection=qc_proj,
    )
    report.add_figure(
        section_name,
        title="DModX per sample",
        fig=dmodx_bar,
        caption=(
            "Per-sample DModX (out-of-model distance). Bars above the "
            "critical line indicate samples whose shape isn't well "
            "explained by the PC model — usually a technical issue. QC "
            "DModX values (grey) should be small (QCs reconstruct well "
            "from the biological PC model)."
        ),
        method_metric="DModX (Eriksson 2006)",
    )

    # ── Table: flagged samples ────────────────────────────────────
    if int(res.flagged_either.sum()) > 0:
        samples_idx = samples.set_index("sample_id")
        rows = []
        for sid, t2, dx, ft, fd in zip(
            res.sample_ids,
            res.t_squared,
            res.dmodx,
            res.flagged_t2,
            res.flagged_dmodx,
            strict=True,
        ):
            if not (ft or fd):
                continue
            cl = (
                str(samples_idx.loc[sid, "cell_line"])
                if sid in samples_idx.index
                else ""
            )
            why_parts = []
            if ft:
                why_parts.append("T²")
            if fd:
                why_parts.append("DModX")
            rows.append(
                {
                    "sample_id": sid,
                    "cell_line": cl,
                    "T²": round(float(t2), 4),
                    "T²_crit_exceeded": "yes" if ft else "no",
                    "DModX": round(float(dx), 4),
                    "DModX_crit_exceeded": "yes" if fd else "no",
                    "flagged_by": " + ".join(why_parts),
                }
            )
        report.add_table(
            section_name,
            title=f"Flagged samples (n = {len(rows)})",
            df=pd.DataFrame(rows),
            caption=(
                "Action: review the run-order trajectories and PCA panel for "
                "each flagged sample. To exclude a sample from the entire "
                "pipeline, add its ``sample_id`` to ``parse.exclude_samples`` "
                "in <code>conf/parse.yaml</code> and re-run via "
                "<code>dvc repro</code> from Stage 1."
            ),
        )

    return res
