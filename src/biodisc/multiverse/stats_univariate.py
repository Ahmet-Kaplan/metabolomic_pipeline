"""Univariate statistics for the multiverse pipeline (Stage 18 — univariate).

Computes per-feature **parametric one-way ANOVA** across cell-line groups,
then applies BH-FDR correction, eta-squared effect size, per-pair log2
fold-change + Cohen's d, and Tukey HSD post-hoc.

Rationale for parametric ANOVA (not Kruskal-Wallis)
---------------------------------------------------
With n=6-7 per group, non-parametric tests (KW) have low power. ANOVA
on log2-transformed intensities is standard in metabolomics (Broadhurst
2018 mQACC) and the parametric F-statistic is more interpretable for
effect-size estimation (eta-squared). The log2 transformation typically
makes metabolomics data approximately normal. Users should verify this
assumption via the variance-heterogeneity diagnostic.

Processing stack
----------------
* **One-way ANOVA** — per-feature F-statistic and p-value
* **BH-FDR correction** (Benjamini & Hochberg 1995) — controls expected
  false-discovery rate; standard for ω-omics multiple-testing.
* **η² effect size** — standardised effect magnitude for ANOVA, a more
  biologically-interpretable companion to the p-value.
* **Per-pair log2 FC + Cohen's d** — direction and standardised effect
  for every pair of groups, so the user can answer "MCF7 vs MCF10A is
  the contrast that drives this feature's significance".
* **Tukey HSD post-hoc** — which specific groups differ (controls
  family-wise error within the post-hoc family).

Why a stack and not just BH-FDR + log2 FC
-----------------------------------------
At our N (5 groups × 6–7 reps = 30 samples), even truly null features
have noisy F-statistics, and BH-FDR thresholds at q < 0.05 will admit
some methodological false positives. Adding η² (effect size) and per-
pair Cohen's d lets the shortlist filter ALSO require a meaningful
biological magnitude — not just a small p-value with marginal effect.

Output schema (one row per feature)
-----------------------------------
::

    feature_id           string
    f_statistic          float    one-way ANOVA F
    p_value              float    raw p-value
    q_value              float    BH-FDR-corrected
    eta_squared          float    standardised effect size
    n_groups             int      groups with ≥ ``min_per_group`` samples
    pair_<A>_<B>_log2_fc float    log2(mean_A / mean_B), per pair
    pair_<A>_<B>_cohens_d float   standardised mean diff, per pair
    pair_<A>_<B>_tukey_p float    Tukey HSD post-hoc p-value
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class UnivariateResult:
    """Per-feature univariate stats."""

    table: pd.DataFrame
    n_features: int
    n_significant_at_q05: int
    n_groups: int
    group_names: list[str]
    fdr_method: str


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return pvals.copy()
    sort_idx = np.argsort(pvals)
    sorted_p = pvals[sort_idx]
    ranks = np.arange(1, n + 1)
    q_sorted = sorted_p * n / ranks
    # Enforce monotonicity (cumulative minimum from the right)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    q: np.ndarray = np.empty_like(q_sorted)
    q[sort_idx] = q_sorted
    return q


def _eta_squared(values: list[np.ndarray]) -> float:
    """One-way ANOVA η² = SS_between / SS_total. Range [0, 1]."""
    all_vals = np.concatenate(values)
    grand_mean = all_vals.mean()
    ss_total = float(((all_vals - grand_mean) ** 2).sum())
    ss_between = 0.0
    for g in values:
        if len(g) == 0:
            continue
        ss_between += len(g) * (g.mean() - grand_mean) ** 2
    if ss_total <= 0:
        return float("nan")
    return float(ss_between / ss_total)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference for two groups (pooled SD, weighted by n)."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    n_a, n_b = len(a), len(b)
    sd_a = a.std(ddof=1) if n_a > 1 else 0.0
    sd_b = b.std(ddof=1) if n_b > 1 else 0.0
    if (sd_a + sd_b) == 0:
        return float("nan")
    # Correct pooled SD: weighted by degrees of freedom (n-1 each),
    # not a simple average of variances.
    pooled = np.sqrt(((n_a - 1) * sd_a**2 + (n_b - 1) * sd_b**2) / (n_a + n_b - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def compute_univariate(
    ctx: StageContext,
    *,
    min_per_group: int = 3,
    fdr_method: str = "bh",
    biological_only: bool = True,
    sample_ids: Iterable[str] | None = None,
    log_for_fc: bool = True,
    is_log_scale: bool | None = None,
    tukey_post_hoc: str = "auto",
    tukey_q_threshold: float = 0.05,
) -> UnivariateResult:
    """Compute one-way ANOVA + BH-FDR + per-pair effect sizes per feature.

    Parameters
    ----------
    min_per_group : int, default 3
        Groups with fewer than this many samples are dropped before ANOVA
        (variance estimate would be unreliable).
    fdr_method : str, default "bh"
        Multiple-testing correction. Currently only ``"bh"`` (Benjamini-
        Hochberg) is implemented.
    biological_only : bool, default True
        If True and ``sample_ids`` is None, restrict to ``injection_type
        == 'biological'`` samples.
    sample_ids : iterable of str, optional
        Explicit sample subset.
    log_for_fc : bool, default True
        If True, ``log2 FC`` is computed assuming the input matrix is
        ALREADY log-transformed (so ``FC = mean_A - mean_B`` directly).
        If False, computes ``log2(mean_A / mean_B)`` from raw values.
    is_log_scale : bool | None, default None
        Explicit override for whether the input matrix is on log scale.
        ``None`` (default) auto-detects via two cues: (a) any negative
        values OR (b) max absolute value &lt; 30 (raw LC-MS abundances
        span 10²–10⁶ — anything below 30 is almost certainly logged).
        Passing ``True``/``False`` skips auto-detection entirely. The
        biological subset can be all-positive even when QC/blank cells
        dipped negative post-log; this override avoids that edge case.
    tukey_post_hoc : str, default "auto"
        When to compute the per-pair Tukey HSD post-hoc. Three modes:

        * ``"auto"`` (default) — compute only for features whose ANOVA
          q-value is below ``tukey_q_threshold`` (the standard
          mQACC pattern: post-hoc is meaningful only after omnibus
          significance). Other features get NaN. Fast, defensible.
        * ``"all"`` — compute for every feature. Slow on big datasets
          (~10 min for 5000 features × 5 groups).
        * ``"none"`` — skip entirely; ``pair_*_tukey_p`` columns absent.
    tukey_q_threshold : float, default 0.05
        Q-value cutoff for ``tukey_post_hoc='auto'``. Tukey is
        computed only for features with ANOVA q < this threshold.
    """
    from scipy.stats import tukey_hsd

    if tukey_post_hoc not in ("auto", "all", "none"):
        raise ValueError(
            f"tukey_post_hoc must be 'auto', 'all', or 'none'; got {tukey_post_hoc!r}"
        )

    if ctx.matrix.isna().any().any():
        raise ValueError("Univariate stats require a NaN-free matrix.")

    # 1) Pick samples to analyse
    if sample_ids is not None:
        cols = [s for s in sample_ids if s in ctx.matrix.columns]
        sub_samples = ctx.samples[ctx.samples["sample_id"].isin(cols)]
    elif biological_only:
        sub_samples = ctx.samples[ctx.samples["injection_type"] == "biological"]
        cols = [s for s in sub_samples["sample_id"] if s in ctx.matrix.columns]
        sub_samples = sub_samples[sub_samples["sample_id"].isin(cols)]
    else:
        cols = list(ctx.matrix.columns)
        sub_samples = ctx.samples[ctx.samples["sample_id"].isin(cols)]

    if len(cols) < 4:
        raise ValueError(
            f"Univariate ANOVA needs ≥ 4 samples; found {len(cols)} after filtering."
        )

    sub_samples = sub_samples.set_index("sample_id").loc[cols]
    if "cell_line" not in sub_samples.columns:
        raise ValueError("samples DataFrame must have a 'cell_line' column.")

    groups = sub_samples["cell_line"].astype(str)
    group_counts = groups.value_counts()
    valid_groups = group_counts[group_counts >= min_per_group].index.tolist()
    if len(valid_groups) < 2:
        raise ValueError(
            f"Need ≥ 2 groups with ≥ {min_per_group} samples each; "
            f"found groups {dict(group_counts)}"
        )
    group_names = sorted(valid_groups)
    log.info(
        "univariate_groups_resolved",
        n_groups=len(group_names),
        group_names=group_names,
        group_counts={g: int(group_counts[g]) for g in group_names},
    )

    # 2) Build per-group sample-id lookup
    group_to_samples: dict[str, list[str]] = {
        g: groups[groups == g].index.tolist() for g in group_names
    }

    # 3) Resolve "is the matrix already log-transformed?". Caller may
    #    pass ``is_log_scale`` to override; otherwise auto-detect using
    #    two cues that catch typical pipeline outputs without false
    #    positives:
    #      * Any negative values anywhere in the FULL matrix → logged.
    #        (Don't restrict to biological cells — QC/blank may carry
    #         the negatives even when biological is all-positive, e.g.
    #         after log2 of pqn-normalised data with a small ε.)
    #      * Max absolute value below 30 → logged. Raw LC-MS abundances
    #        span 10²–10⁶; values < 30 are practically always log-units.
    sub_matrix = ctx.matrix[cols]
    if is_log_scale is None:
        full_matrix = ctx.matrix
        has_negatives = bool((full_matrix < 0).any().any())
        max_abs_val = float(full_matrix.abs().max().max())
        resolved_is_log = has_negatives or max_abs_val < 30.0
    else:
        resolved_is_log = bool(is_log_scale)
    use_subtract_for_fc = resolved_is_log and log_for_fc

    # 4) Vectorised ANOVA + per-pair stats (replaces previous per-feature loop)
    #
    # Build per-group submatrices once, then compute everything in numpy
    # using axis=1 reductions across samples-within-group. For 4644
    # features × 5 groups this brings the loop from ~5 min to ~1 s.
    from scipy.stats import f as f_dist

    n_groups = len(group_names)
    matrix_arr = sub_matrix.to_numpy(dtype=float)  # shape (n_features, n_samples)
    n_features = matrix_arr.shape[0]

    # Per-group: indices, sizes, means, variances (all per-feature)
    group_indices = {
        g: [cols.index(s) for s in group_to_samples[g]] for g in group_names
    }
    group_sizes = np.array([len(group_indices[g]) for g in group_names])  # (k,)
    group_means = np.zeros((n_features, n_groups))
    group_vars = np.zeros((n_features, n_groups))  # ddof=1
    group_sum_sq_centered = np.zeros((n_features, n_groups))  # for SS_within
    for j, g in enumerate(group_names):
        sub = matrix_arr[:, group_indices[g]]  # (n_features, n_g)
        group_means[:, j] = sub.mean(axis=1)
        group_vars[:, j] = sub.var(axis=1, ddof=1) if sub.shape[1] > 1 else 0.0
        # SS_within contribution: sum((x - mean_g)^2) per feature
        group_sum_sq_centered[:, j] = ((sub - group_means[:, j:j + 1]) ** 2).sum(axis=1)

    n_total = int(group_sizes.sum())
    # Grand mean per feature (weighted by group sizes)
    grand_mean = (group_means * group_sizes).sum(axis=1) / n_total  # (n_features,)

    # SS_between per feature: Σ n_g * (mean_g - grand_mean)^2
    ss_between = (group_sizes * (group_means - grand_mean[:, None]) ** 2).sum(axis=1)
    ss_within = group_sum_sq_centered.sum(axis=1)
    ss_total = ss_between + ss_within
    df_between = n_groups - 1
    df_within = n_total - n_groups

    with np.errstate(divide="ignore", invalid="ignore"):
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        f_statistics = ms_between / ms_within
        p_values = f_dist.sf(f_statistics, df_between, df_within)
        # Replace inf / nan with nan for downstream logic
        f_statistics = np.where(np.isfinite(f_statistics), f_statistics, np.nan)
        p_values = np.where(np.isfinite(p_values), p_values, np.nan)
        eta_squareds = np.where(ss_total > 0, ss_between / ss_total, np.nan)

    # Build base table
    base_data = {
        "feature_id": sub_matrix.index.astype(str).tolist(),
        "f_statistic": f_statistics,
        "p_value": p_values,
        "eta_squared": eta_squareds,
        "n_groups": np.full(n_features, n_groups, dtype=int),
    }
    table = pd.DataFrame(base_data)

    # Per-pair: log2 FC + Cohen's d (vectorised across features)
    for i, g_a in enumerate(group_names):
        mean_a = group_means[:, i]
        var_a = group_vars[:, i]
        for j_idx in range(i + 1, n_groups):
            g_b = group_names[j_idx]
            mean_b = group_means[:, j_idx]
            var_b = group_vars[:, j_idx]
            # log2 FC
            if use_subtract_for_fc:
                fc = mean_a - mean_b
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    safe_b = np.where(mean_b > 0, mean_b, np.nan)
                    ratio = np.where(mean_a > 0, mean_a, np.nan) / safe_b
                    fc = np.log2(ratio)
            # Cohen's d
            with np.errstate(divide="ignore", invalid="ignore"):
                pooled_sd = np.sqrt((var_a + var_b) / 2.0)
                d = np.where(pooled_sd > 0, (mean_a - mean_b) / pooled_sd, np.nan)
            table[f"pair_{g_a}_{g_b}_log2_fc"] = fc
            table[f"pair_{g_a}_{g_b}_cohens_d"] = d

    # Tukey placeholder columns (filled in second pass)
    if tukey_post_hoc != "none":
        for i, g_a in enumerate(group_names):
            for g_b in group_names[i + 1:]:
                table[f"pair_{g_a}_{g_b}_tukey_p"] = np.nan

    # 5) BH-FDR
    pvals = table["p_value"].to_numpy()
    valid = np.isfinite(pvals)
    qvals = np.full(len(pvals), np.nan)
    qvals[valid] = _bh_fdr(pvals[valid])
    table["q_value"] = qvals

    # 6) Tukey HSD second pass (auto-mode: only for q-significant features)
    n_tukey_computed = 0
    if tukey_post_hoc != "none":
        if tukey_post_hoc == "auto":
            features_to_tukey = table.index[
                table["q_value"].fillna(1.0) < tukey_q_threshold
            ].tolist()
        else:  # "all"
            features_to_tukey = table.index.tolist()

        for table_idx in features_to_tukey:
            fid = table.at[table_idx, "feature_id"]
            row_vals = sub_matrix.loc[fid]
            arrays_in_order = [
                row_vals.loc[group_to_samples[g]].to_numpy(dtype=float)
                for g in group_names
            ]
            try:
                tukey_result = tukey_hsd(*arrays_in_order)
                tukey_pvals = tukey_result.pvalue
                for i, g_a in enumerate(group_names):
                    for j, g_b in enumerate(group_names):
                        if i < j:
                            table.at[
                                table_idx, f"pair_{g_a}_{g_b}_tukey_p"
                            ] = float(tukey_pvals[i, j])
                n_tukey_computed += 1
            except Exception:  # pragma: no cover — defensive
                pass

    # Reorder: feature_id, f, p, q, eta², n_groups, then per-pair cols
    base_cols = ["feature_id", "f_statistic", "p_value", "q_value", "eta_squared", "n_groups"]
    pair_cols = [c for c in table.columns if c not in base_cols]
    table = table[base_cols + pair_cols]

    n_sig = int((table["q_value"] < 0.05).sum())
    log.info(
        "univariate_done",
        n_features=len(table),
        n_significant_at_q05=n_sig,
        n_groups=len(group_names),
        is_log_scale_input=resolved_is_log,
        is_log_scale_override=(is_log_scale is not None),
        tukey_post_hoc=tukey_post_hoc,
        n_tukey_computed=n_tukey_computed,
    )

    return UnivariateResult(
        table=table,
        n_features=len(table),
        n_significant_at_q05=n_sig,
        n_groups=len(group_names),
        group_names=group_names,
        fdr_method=fdr_method,
    )
