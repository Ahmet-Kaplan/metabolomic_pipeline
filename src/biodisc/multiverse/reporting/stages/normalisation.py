"""Normalisation section — sample-loading correction quality.

The reviewer wants to know:

1. **Did per-sample loading vary?** — boxplots of per-sample row sums on
   the linear scale, before vs after. Variance across samples drops if
   normalisation worked.
2. **What were the correction factors?** — strip plot or histogram of
   per-sample factors, coloured by injection type (QCs should sit near
   the median, biological samples spread).
3. **Did QC tightness improve?** — PMSD (pooled median SD of QCs) and
   median per-feature QC-CV, paired bars before vs after.
4. **Did biology survive?** — biological silhouette before/after as the
   guardrail (De Livera 2015 — &lt; 5 % drop is acceptable).
5. **Was anything extreme?** — surface n_extreme_factors so an obvious
   outlier sample doesn't slip past unflagged.

References
----------
* Dieterle 2006, *Anal Chem* 78:4281 — PQN.
* Warrack 2009, *J Chromatogr B* 877:547 — MSTUS.
* Li et al. 2017, *Sci Rep* 6:38881 — PMSD.
* Yang et al. 2020, *NAR* 48:W436 — NOREVA 2.0 framework.
* De Livera et al. 2015, *Anal Chem* 87:3606 — silhouette guardrail.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biodisc.multiverse.diagnostics import (
    biological_silhouette,
    pmsd,
    qc_cv_median,
)
from biodisc.multiverse.reporting.core import PathReport
from biodisc.multiverse.reporting.figures import (
    histogram_overlay,
    kde_overlay,
    paired_metric_bar,
    per_sample_boxplot,
)


def _per_sample_factor(before: pd.DataFrame, after: pd.DataFrame) -> pd.Series:
    """Recover the per-sample correction factor: factor_s = sum_before(s) / sum_after(s).

    Works for any per-sample multiplicative scaling (PQN, MSTUS, median).
    Returns ``Series[sample_id → factor]``. Falls back to 1.0 on degenerate samples.
    """
    sum_before = before.sum(axis=0)
    sum_after = after.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = sum_before / sum_after.replace(0, np.nan)
    return factors.fillna(1.0)


def add_normalisation_section(
    report: PathReport,
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    samples: pd.DataFrame,
    method: str,
    parameters: dict[str, Any] | None = None,
    section_name: str = "normalisation",
) -> None:
    """Add the normalisation section to ``report``."""
    parameters = dict(parameters or {})

    title_map = {
        "pqn": "PQN — Probabilistic Quotient Normalization",
        "mstus": "MSTUS — MS Total Useful Signal normalisation",
        "median": "Median normalisation",
    }
    description_map = {
        "pqn": (
            "Per-sample factor = median of per-feature quotients against a reference "
            "spectrum (Dieterle 2006, <em>Anal Chem</em> 78:4281). The robust default "
            "for clinical and discovery metabolomics; the median makes the estimate "
            "insensitive to a few differentially-expressed features."
        ),
        "mstus": (
            "Per-sample factor = sum of &quot;useful&quot; (consistently-detected) "
            "feature intensities (Warrack 2009). Robust to dilution effects but "
            "vulnerable to a few highly-abundant differentially-expressed features."
        ),
        "median": (
            "Per-sample factor = median feature intensity. Simple baseline; "
            "less robust than PQN for compositional data."
        ),
    }

    report.add_section(
        section_name,
        title=title_map.get(method, f"Normalisation — {method}"),
        description=description_map.get(method, ""),
        method=method,
        parameters=parameters,
    )

    qc_ids = samples.loc[samples["injection_type"] == "qc", "sample_id"].tolist()
    qc_ids = [s for s in qc_ids if s in after.columns]

    # ── Headline metrics ──────────────────────────────────────────
    pmsd_before = pmsd(before, qc_ids)
    pmsd_after = pmsd(after, qc_ids)
    cv_before = qc_cv_median(before, qc_ids)
    cv_after = qc_cv_median(after, qc_ids)
    sil_before = biological_silhouette(before, samples)
    sil_after = biological_silhouette(after, samples)
    factors = _per_sample_factor(before, after)
    n_extreme = int(((factors < 0.1) | (factors > 10)).sum())

    for label, value in [
        ("PMSD (before)", pmsd_before),
        ("PMSD (after)", pmsd_after),
        ("median QC-CV % (before)", cv_before),
        ("median QC-CV % (after)", cv_after),
        ("biological silhouette (before)", sil_before),
        ("biological silhouette (after)", sil_after),
        ("factor median", float(factors.median())),
        ("factor min", float(factors.min())),
        ("factor max", float(factors.max())),
    ]:
        report.add_metric(
            section_name,
            label,
            round(float(value), 4) if np.isfinite(value) else "n/a",
        )
    if n_extreme > 0:
        report.add_metric(
            section_name,
            "n_extreme_sample_factors",
            n_extreme,
            unit="samples",
            description="samples with correction factor < 0.1 or > 10",
        )

    # ── Figure 1: paired metric bars (PMSD, QC-CV, silhouette) ────
    metric_pairs = [
        ("PMSD", pmsd_before, pmsd_after),
        ("median QC-CV %", cv_before, cv_after),
        ("biological silhouette", sil_before, sil_after),
    ]
    metric_fig = paired_metric_bar(
        [m[0] for m in metric_pairs],
        [float(m[1]) if np.isfinite(m[1]) else 0.0 for m in metric_pairs],
        [float(m[2]) if np.isfinite(m[2]) else 0.0 for m in metric_pairs],
        title="Normalisation metric panel — before vs after",
        yaxis_title="metric value",
    )
    report.add_figure(
        section_name,
        title="Headline metric panel",
        fig=metric_fig,
        caption=(
            "PMSD (Li 2017) and QC-CV should drop after normalisation; biological "
            "silhouette should hold or rise. A silhouette drop &gt; 5 % flags potential "
            "overcorrection (De Livera 2015 guardrail)."
        ),
        method_metric="PMSD · QC-CV · silhouette (NOREVA 2.0 framework)",
    )

    # ── Figure 2: per-sample row distribution boxplots ────────────
    box_before = per_sample_boxplot(
        before,
        samples,
        title="Per-sample feature distribution — before normalisation",
        yaxis_title="abundance",
        use_log1p=True,
    )
    report.add_figure(
        section_name,
        title="Per-sample row distribution — before",
        fig=box_before,
        caption=(
            "One box per sample, log1p of feature intensities. Bare-eye check for "
            "loading differences: shifted medians = loading variation that the "
            "normaliser is supposed to remove."
        ),
        method_metric="per-sample loading visualisation",
    )
    box_after = per_sample_boxplot(
        after,
        samples,
        title="Per-sample feature distribution — after normalisation",
        yaxis_title="abundance",
        use_log1p=True,
    )
    report.add_figure(
        section_name,
        title="Per-sample row distribution — after",
        fig=box_after,
        caption=(
            "Same plot post-normalisation. Sample medians should align (or align "
            "more tightly than before). Persistent shift on a single biological "
            "sample = sample-level outlier worth flagging."
        ),
        method_metric="per-sample loading visualisation",
    )

    # ── Figure 3: per-sample correction factor distribution ───────
    finite_factors_raw = np.asarray(factors.values, dtype=float)
    finite_factors = finite_factors_raw[np.isfinite(finite_factors_raw)]
    if finite_factors.size > 0:
        factor_fig = histogram_overlay(
            [finite_factors],
            ["per-sample factor"],
            title="Per-sample correction factor distribution",
            xaxis_title="factor value",
            colors=["#0072B2"],
            nbins=30,
        )
        # Reference line at 1.0
        factor_fig.add_vline(
            x=1.0,
            line={"color": "#444444", "width": 1.0, "dash": "dot"},
            annotation={"text": "factor = 1.0", "font": {"size": 10, "color": "#444"}},
            annotation_position="top right",
        )
        report.add_figure(
            section_name,
            title="Per-sample correction factor distribution",
            fig=factor_fig,
            caption=(
                "Distribution of the per-sample correction factor recovered from "
                "row-sum ratios. Mass should concentrate near 1.0; tails &gt; 10 or "
                "&lt; 0.1 are flagged as extreme (possible loading outliers)."
            ),
            method_metric="factor distribution (sanity check)",
        )

    # ── Figure 4: per-feature QC-CV before vs after ───────────────
    if qc_ids:
        cv_b = before[qc_ids].std(axis=1, ddof=1) / before[qc_ids].mean(axis=1).replace(0, np.nan) * 100
        cv_a = after[qc_ids].std(axis=1, ddof=1) / after[qc_ids].mean(axis=1).replace(0, np.nan) * 100
        cv_b_arr = cv_b.dropna().to_numpy()
        cv_a_arr = cv_a.dropna().to_numpy()
        cv_b_arr = cv_b_arr[cv_b_arr <= 200.0]
        cv_a_arr = cv_a_arr[cv_a_arr <= 200.0]
        if cv_b_arr.size > 0 or cv_a_arr.size > 0:
            cv_kde = kde_overlay(
                [cv_b_arr, cv_a_arr],
                ["before", "after"],
                title="Per-feature QC %CV — before vs after",
                xaxis_title="QC-CV %",
                colors=["#999999", "#0072B2"],
            )
            cv_kde.add_vline(
                x=30.0,
                line={"color": "#D55E00", "width": 1.5, "dash": "dash"},
                annotation={"text": "mQACC 30 %", "font": {"color": "#D55E00", "size": 10}},
                annotation_position="top right",
            )
            report.add_figure(
                section_name,
                title="Per-feature QC %CV distribution",
                fig=cv_kde,
                caption=(
                    "Mass should shift left after normalisation. Tail clipped at 200 % "
                    "for readability."
                ),
                method_metric="RSD-QC distribution",
            )

    # ── Table: top samples by absolute factor deviation ───────────
    factor_dev = (factors - 1.0).abs().sort_values(ascending=False).head(10)
    if not factor_dev.empty:
        meta = samples.set_index("sample_id")
        rows = []
        for sid, dev in factor_dev.items():
            sid_str = str(sid)
            inj = str(meta.loc[sid_str, "injection_type"]) if sid_str in meta.index else ""
            cl = str(meta.loc[sid_str, "cell_line"]) if sid_str in meta.index else ""
            rows.append(
                {
                    "sample_id": sid_str,
                    "injection_type": inj,
                    "cell_line": cl,
                    "factor": round(float(factors.loc[sid_str]), 4),
                    "abs_dev_from_1": round(float(dev), 4),
                }
            )
        report.add_table(
            section_name,
            title="Top-10 samples by |factor − 1|",
            df=pd.DataFrame(rows),
            caption=(
                "Samples with the largest correction factors. QCs should not appear "
                "near the top (they are pooled and uniform); biological samples with "
                "extreme factors warrant a column-loading audit."
            ),
        )
