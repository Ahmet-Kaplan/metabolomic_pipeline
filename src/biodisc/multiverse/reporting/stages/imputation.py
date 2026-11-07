"""Imputation section — visualises the missing → filled transition.

The report needs to convince a reviewer of three things, in this order:

1. **What was missing?** — show the pre-imputation gap pattern. We use a
   compact missingness heatmap (top-N most-missing features) plus a
   per-feature missing-rate histogram.
2. **What got filled?** — overlay the KDE of imputed values against the
   KDE of observed values on the same matrix scale. If the imputed
   distribution sits *below* the observed range with no overlap, MNAR
   imputation is behaving as designed (LOD surrogate). If it overlaps,
   the method is making MAR-style guesses.
3. **Where did the fills land?** — per-sample bar of imputed-cell count.
   Even spread = healthy; spikes on a particular sample = that sample
   had an instrument hiccup or low loading.

Headline metrics (mQACC + sanity):
  * ``n_missing_before``, ``frac_missing_before``
  * ``n_imputed``, ``frac_imputed``
  * ``imputed_min`` / ``imputed_max`` / ``imputed_median``
  * ``observed_min`` / ``observed_max`` / ``observed_median``

References
----------
* Wei et al. 2018, *Sci Rep* 8:663 — NRMSE / PCC framework for evaluating
  imputation. We compute distribution overlap as a stand-in
  (we don't have ground-truth held-out values at runtime).
* Krzywiec et al. 2024, *Bioinformatics* — half-minimum's MNAR-default
  rationale that drives the KDE-below-observed expectation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biodisc.multiverse.reporting.core import PathReport
from biodisc.multiverse.reporting.figures import (
    histogram_overlay,
    kde_overlay,
    missingness_heatmap,
    per_sample_bar,
)
from biodisc.multiverse.reporting.theme import apply_theme


def _imputed_mask(
    before: pd.DataFrame, after: pd.DataFrame, *, treat_zero_as_missing: bool = True
) -> pd.DataFrame:
    """Boolean mask: True where the imputer actually filled a missing cell.

    "Actually filled" = ``before`` was missing AND ``after`` differs from
    ``before``. The value-change check matters because some imputers
    (half-min) only replace NaN cells and leave zero-encoded missingness
    untouched — counting those as "imputed" would inflate the
    per-feature ratios with 0/min = 0 entries that don't reflect any
    real method behaviour.
    """
    before_missing = (
        before.isna() | (before == 0) if treat_zero_as_missing else before.isna()
    )
    # An actual fill means after has a finite value AND that value differs
    # from the (possibly 0 / NaN) before value.
    after_finite = after.notna() & np.isfinite(after.to_numpy())
    # Compare cell-by-cell: NaN-aware "values differ" check.
    diff = (before.fillna(np.nan) != after.fillna(np.nan)) | (
        before.isna() & after.notna()
    )
    out = before_missing & after_finite & diff
    return pd.DataFrame(out, index=after.index, columns=after.columns)


def add_imputation_section(
    report: PathReport,
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    samples: pd.DataFrame,
    method: str,
    parameters: dict[str, Any] | None = None,
    section_name: str = "imputation",
    treat_zero_as_missing: bool = True,
) -> None:
    """Add the imputation section to ``report``.

    Parameters
    ----------
    report
        The accumulator to append into.
    before
        Pre-imputation matrix (features × samples). NaN or zeros mark
        missing values.
    after
        Post-imputation matrix. Must have the same index/columns as
        ``before``.
    samples
        Sample metadata table.
    method
        Method identifier (e.g. ``"halfmin"``, ``"qrilc"``).
    parameters
        Method parameters. Recorded under "parameters used" in the report.
    section_name
        Section slug. Default ``"imputation"``.
    treat_zero_as_missing
        Whether to treat 0s as below-LOD missingness (Progenesis
        convention). Default ``True``.
    """
    parameters = dict(parameters or {})

    if before.shape != after.shape:
        raise ValueError(
            f"shape mismatch: before {before.shape} vs after {after.shape}"
        )

    title_map = {
        "halfmin": "Half-minimum imputation (½ × per-feature min)",
        "qrilc": "QRILC — Quantile Regression for Left-Censored data",
        "ns_knn": "NS-kNN — Neighbourhood-Selective k-NN imputation",
        "missforest": "missForest — random-forest imputation",
        "mean": "Mean imputation (per-feature)",
    }
    description_map = {
        "halfmin": (
            "MNAR LOD surrogate (Hrydziuszko & Viant 2012). For each feature, "
            "missing values are replaced by ½ × the smallest observed value. "
            "Imputed cells should fall <em>below</em> the observed distribution."
        ),
        "qrilc": (
            "MNAR-aware imputation drawn from a truncated log-normal "
            "(Lazar 2016, <em>J Proteome Res</em> 15:1116). Missing cells "
            "are sampled from the lower tail estimated per-feature."
        ),
        "ns_knn": (
            "kNN restricted to neighbours from the same group "
            "(Lee &amp; Styczynski 2018, <em>Metabolomics</em> 14:153). "
            "Mixes MAR/MNAR depending on neighbour overlap."
        ),
        "missforest": (
            "Random-forest imputation (Stekhoven &amp; Bühlmann 2012, "
            "<em>Bioinformatics</em> 28:112). Iterative — assumes MAR."
        ),
        "mean": "Per-feature mean imputation. MAR-only baseline; rarely justified for LC-MS.",
    }
    title = title_map.get(method, f"Imputation — {method}")
    description = description_map.get(method, "")

    report.add_section(
        section_name,
        title=title,
        description=description,
        method=method,
        parameters=parameters,
    )

    # ── Headline counts ───────────────────────────────────────────
    before_missing_mask = (
        before.isna() | (before == 0) if treat_zero_as_missing else before.isna()
    )
    n_missing_before = int(before_missing_mask.sum().sum())
    frac_missing_before = (
        n_missing_before / float(before.size) if before.size else 0.0
    )
    imputed_mask = _imputed_mask(before, after, treat_zero_as_missing=treat_zero_as_missing)
    n_imputed = int(imputed_mask.sum().sum())
    n_left_missing = n_missing_before - n_imputed

    report.add_metric(
        section_name, "n_missing_before", n_missing_before, unit="cells"
    )
    report.add_metric(
        section_name, "frac_missing_before", round(frac_missing_before * 100, 3), unit="%"
    )
    report.add_metric(section_name, "n_imputed", n_imputed, unit="cells")
    report.add_metric(
        section_name,
        "frac_imputed",
        round(n_imputed / float(before.size) * 100, 3) if before.size else 0.0,
        unit="%",
    )
    if n_left_missing != 0:
        report.add_metric(
            section_name,
            "n_left_missing",
            n_left_missing,
            unit="cells",
            description="cells still NaN/0 after imputation (e.g. all-NaN feature falling back to zero)",
        )

    imputed_values = after.values[imputed_mask.values]
    observed_values = after.values[~before_missing_mask.values]
    if imputed_values.size > 0:
        report.add_metric(
            section_name,
            "imputed_median",
            round(float(np.nanmedian(imputed_values)), 4),
        )
        report.add_metric(
            section_name,
            "imputed_p05",
            round(float(np.nanpercentile(imputed_values, 5)), 4),
        )
        report.add_metric(
            section_name,
            "imputed_p95",
            round(float(np.nanpercentile(imputed_values, 95)), 4),
        )
    if observed_values.size > 0:
        report.add_metric(
            section_name,
            "observed_median",
            round(float(np.nanmedian(observed_values)), 4),
        )

    # ── Figure 1: pre-imputation missingness heatmap ──────────────
    miss_fig = missingness_heatmap(
        before,
        samples,
        title="Pre-imputation missingness heatmap",
        max_features=600,
        treat_zero_as_missing=treat_zero_as_missing,
    )
    report.add_figure(
        section_name,
        title="Pre-imputation missingness heatmap (top 600 features)",
        fig=miss_fig,
        caption=(
            "Dark = detected, white = missing. Rows ranked by missing %. "
            "Columns grouped by injection type (blank / QC / biological / dilution). "
            "Whole-sample white stripes flag a failed injection; whole-feature "
            "white bands are common for group-private metabolites."
        ),
        method_metric="diagnostic — input state",
    )

    # ── Figure 2: per-feature missing-rate histogram ──────────────
    feature_miss_pct = before_missing_mask.mean(axis=1).to_numpy(dtype=float) * 100.0
    miss_hist = histogram_overlay(
        [feature_miss_pct],
        ["features"],
        title="Per-feature missing rate (pre-imputation)",
        xaxis_title="missing %",
        nbins=40,
        colors=["#0072B2"],
    )
    report.add_figure(
        section_name,
        title="Per-feature missing-rate distribution",
        fig=miss_hist,
        caption=(
            "How concentrated is the missingness? A right-leaning distribution "
            "(many features mostly-missing) means the matrix is sparse and "
            "MNAR-imputation accuracy matters more."
        ),
        method_metric="diagnostic — input state",
    )

    # ── Figure 3: imputed vs observed on log10 scale ───────────────
    # Linear scale is unreadable for LC-MS data: dynamic range spans 4–6
    # orders of magnitude, so on linear axes the bulk of values stack into
    # a thin spike near zero with the long tail dominating the x-axis.
    # log10 keeps both the imputed lower tail and the observed bulk visible
    # in a single panel.
    if imputed_values.size > 0 and observed_values.size > 0:
        observed_pos = observed_values[observed_values > 0]
        imputed_pos = imputed_values[imputed_values > 0]
        if observed_pos.size > 0 and imputed_pos.size > 0:
            log_kde = kde_overlay(
                [np.log10(observed_pos), np.log10(imputed_pos)],
                ["observed", "imputed"],
                title=f"Imputed vs observed value distribution (log10) — {method}",
                xaxis_title="log10(abundance)",
                colors=["#0072B2", "#D55E00"],
            )
            report.add_figure(
                section_name,
                title="Imputed vs observed distribution (log10 scale)",
                fig=log_kde,
                caption=(
                    "Cross-feature view of where imputed values land relative to "
                    "the observed signal. LC-MS abundances span 4–6 orders of "
                    "magnitude so log10 is required for readability — on linear "
                    "axes both curves collapse to a single spike near zero. "
                    "Half-min and QRILC are <em>per-feature</em> LOD surrogates: "
                    "for each feature, imputed values are smaller than that "
                    "feature's own observed minimum. Across the whole feature set, "
                    "the imputed distribution should sit clearly to the left of "
                    "the bulk of the observed distribution — broad overlap (rather "
                    "than a left-shifted bump) would indicate a MAR-style "
                    "estimator instead of LOD-surrogate behaviour."
                ),
                method_metric="diagnostic — Wei et al. 2018 (log10 view)",
            )

    # ── Figure 3b: per-feature imputed-to-min-observed ratio ──────
    # The per-feature property test: for halfmin with factor=0.5, every
    # imputed value should equal 0.5 × the per-feature observed minimum.
    # We compute imputed / per-feature-min-observed and expect a tight
    # mass near 0.5 (or whatever factor was chosen). Wider spread or mass
    # > 1 would mean the imputer is making MAR-style guesses.
    per_feat_ratios: list[float] = []
    for fid in after.index:
        if fid not in before.index:
            continue
        before_row = before.loc[fid]
        if treat_zero_as_missing:
            obs = before_row[(before_row.notna()) & (before_row > 0)]
        else:
            obs = before_row.dropna()
        if obs.empty:
            continue
        feature_min = float(obs.min())
        if feature_min <= 0:
            continue
        # Find imputed cells for this feature
        miss_mask = imputed_mask.loc[fid]
        if not miss_mask.any():
            continue
        imp_vals = after.loc[fid][miss_mask].astype(float)
        per_feat_ratios.extend((imp_vals / feature_min).tolist())
    if per_feat_ratios:
        import plotly.graph_objects as _go

        ratios_arr = np.asarray(per_feat_ratios, dtype=float)
        ratios_finite = ratios_arr[np.isfinite(ratios_arr)]
        if ratios_finite.size > 0:
            # Use a histogram, not a KDE: for textbook half-min the ratios
            # are *constant* (all equal to factor), so the KDE is degenerate.
            # A histogram surfaces that property — a single bar at 0.5 is
            # the correct, informative picture; a smoothed KDE would refuse
            # to plot anything at all.
            ratio_fig = _go.Figure(
                data=[
                    _go.Histogram(
                        x=ratios_finite,
                        marker={"color": "#D55E00"},
                        opacity=0.85,
                        xbins={
                            "start": 0.0,
                            "end": max(1.5, float(ratios_finite.max()) + 0.05),
                            "size": 0.05,
                        },
                        hovertemplate="ratio: %{x:.3f}<br>n cells: %{y}<extra></extra>",
                    )
                ]
            )
            apply_theme(
                ratio_fig,
                title=(
                    "Per-feature property test — imputed / per-feature min(observed) "
                    f"({method})"
                ),
                xaxis_title="imputed value / per-feature min(observed)",
                yaxis_title="imputed cell count",
            )
            ratio_fig.add_vline(
                x=1.0,
                line={"color": "#444", "width": 1.0, "dash": "dot"},
                annotation={
                    "text": "= observed min",
                    "font": {"size": 10, "color": "#444"},
                },
                annotation_position="top right",
            )
            report.add_metric(
                section_name,
                "median imputed / min(observed)",
                round(float(np.median(ratios_finite)), 4),
                description="0.5 = textbook half-min; lower = more conservative",
            )
            report.add_figure(
                section_name,
                title="Per-feature property test (imputed / observed min)",
                fig=ratio_fig,
                caption=(
                    "Per-feature LOD-surrogate sanity check. For a textbook half-"
                    "min imputer (<code>factor=0.5</code>) every imputed cell "
                    "equals half its own feature's smallest observed value, so "
                    "the histogram is a single bar at 0.5. QRILC samples from a "
                    "left-truncated normal, so its bars spread below 1 with no "
                    "single mode. Mass above 1 (right of the dotted line) means "
                    "some imputed value exceeded that feature's observed minimum "
                    "— a red flag for MNAR methods, expected for kNN / missForest "
                    "under MAR-like patterns."
                ),
                method_metric="per-feature LOD-surrogate test",
            )

    # ── Figure 5: per-sample imputed-cell count ───────────────────
    per_sample_imp = imputed_mask.sum(axis=0).astype(int)
    if int(per_sample_imp.sum()) > 0:
        bar_fig = per_sample_bar(
            per_sample_imp.index.tolist(),
            per_sample_imp.values.astype(float).tolist(),
            samples,
            title="Per-sample imputed-cell count",
            yaxis_title="cells imputed",
            color_by="injection_type",
        )
        report.add_figure(
            section_name,
            title="Per-sample imputed-cell count",
            fig=bar_fig,
            caption=(
                "Bars stacked by injection type. Even spread = healthy. "
                "Spikes on a single biological sample suggest a bad injection; "
                "spikes on QCs are unusual and worth investigating."
            ),
            method_metric="diagnostic — operator-level QA",
        )

    # ── Table: top-N features by imputed-cell count (helps tracing) ──
    per_feature_imp = imputed_mask.sum(axis=1)
    top_n = (
        per_feature_imp[per_feature_imp > 0]
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )
    top_n.columns = ["feature_id", "n_imputed"]
    if not top_n.empty:
        report.add_table(
            section_name,
            title="Top-20 features by imputed-cell count",
            df=top_n,
            caption=(
                "Features that received the most imputation — useful when "
                "auditing whether a specific putative biomarker is heavily "
                "imputation-driven."
            ),
        )
