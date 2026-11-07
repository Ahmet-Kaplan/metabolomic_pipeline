"""Unit tests for univariate stats: ANOVA + BH-FDR + per-pair effect sizes."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from biodisc.multiverse import StageContext
from biodisc.multiverse.stats_univariate import (
    UnivariateResult,
    _bh_fdr,
    _cohens_d,
    _eta_squared,
    compute_univariate,
)


# ============================================================
# Helpers
# ============================================================
def _make_grouped_ctx(
    *,
    n_features: int = 30,
    n_per_group: int = 6,
    group_names: tuple[str, ...] = ("A", "B", "C"),
    n_differential_features: int = 5,
    effect_size: float = 3.0,
    seed: int = 0,
) -> StageContext:
    """Build a context with KNOWN differential features for testing.

    The first ``n_differential_features`` features have a strong group
    effect: each group has its own mean shifted by ``effect_size``.
    Remaining features are noise (no group effect).
    """
    rng = np.random.default_rng(seed)
    n_groups = len(group_names)
    sample_ids: list[str] = []
    cell_lines: list[str] = []
    for g_name in group_names:
        for r in range(n_per_group):
            sample_ids.append(f"{g_name}_{r + 1}")
            cell_lines.append(g_name)
    n_total = len(sample_ids)

    matrix_data = np.zeros((n_features, n_total))
    base_means = rng.lognormal(8.0, 1.0, size=n_features)

    for f in range(n_features):
        for j, sid in enumerate(sample_ids):
            g_idx = group_names.index(sid.rsplit("_", 1)[0])
            if f < n_differential_features:
                # Group-shifted: each group has mean = base + g_idx × effect
                mean_val = base_means[f] * (1 + g_idx * effect_size / n_groups)
            else:
                # No group effect — pure noise
                mean_val = base_means[f]
            matrix_data[f, j] = max(0.001, mean_val * np.exp(rng.normal(0, 0.1)))

    matrix = pd.DataFrame(
        matrix_data,
        index=[f"F{i}" for i in range(n_features)],
        columns=sample_ids,
    )
    samples = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "raw_sample_id": sample_ids,
            "injection_type": ["biological"] * n_total,
            "cell_line": cell_lines,
            "replicate": [int(s.rsplit("_", 1)[1]) for s in sample_ids],
            "dilution_level": [None] * n_total,
            "acq_date": [datetime(2026, 1, 9)] * n_total,
            "column_order": list(range(n_total)),
        }
    )
    return StageContext(matrix=matrix, samples=samples, features=pd.DataFrame())


# ============================================================
# Helper functions
# ============================================================
class TestBHFDR:
    def test_monotonic(self) -> None:
        pvals = np.array([0.001, 0.01, 0.02, 0.05, 0.1])
        q = _bh_fdr(pvals)
        # q-values should be non-decreasing in the same order
        assert (np.diff(q) >= -1e-9).all()

    def test_q_geq_p_for_small_n(self) -> None:
        pvals = np.array([0.01, 0.02, 0.03, 0.04])
        q = _bh_fdr(pvals)
        # q ≥ p (BH never reduces a p-value below itself)
        assert (q >= pvals - 1e-9).all()

    def test_all_significant_remain_close(self) -> None:
        pvals = np.array([0.001] * 100)
        q = _bh_fdr(pvals)
        # All identical p-values → all identical q-values
        assert np.allclose(q, q[0])


class TestEtaSquared:
    def test_no_group_effect_eta_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        groups = [rng.standard_normal(10) for _ in range(3)]
        eta = _eta_squared(groups)
        assert 0 <= eta < 0.3

    def test_large_group_effect_eta_high(self) -> None:
        rng = np.random.default_rng(1)
        groups = [rng.standard_normal(10) + i * 5.0 for i in range(3)]
        eta = _eta_squared(groups)
        assert eta > 0.7


class TestCohensD:
    def test_no_difference_d_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal(20)
        b = rng.standard_normal(20)
        assert abs(_cohens_d(a, b)) < 0.5

    def test_large_difference_d_high(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.standard_normal(20)
        b = rng.standard_normal(20) + 3.0
        d = _cohens_d(a, b)
        assert abs(d) > 2.0

    def test_returns_nan_when_no_variance(self) -> None:
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([2.0, 2.0, 2.0])
        assert np.isnan(_cohens_d(a, b))


# ============================================================
# Main entry point
# ============================================================
class TestComputeUnivariate:
    def test_returns_univariate_result(self) -> None:
        ctx = _make_grouped_ctx(seed=10)
        result = compute_univariate(ctx)
        assert isinstance(result, UnivariateResult)
        assert result.n_features == 30
        assert result.n_groups == 3
        assert set(result.group_names) == {"A", "B", "C"}

    def test_table_has_expected_columns(self) -> None:
        ctx = _make_grouped_ctx(seed=11)
        result = compute_univariate(ctx)
        cols = set(result.table.columns)
        # Required base columns
        for c in [
            "feature_id", "f_statistic", "p_value", "q_value", "eta_squared", "n_groups"
        ]:
            assert c in cols
        # Per-pair columns: A vs B / A vs C / B vs C
        assert "pair_A_B_log2_fc" in cols
        assert "pair_A_B_cohens_d" in cols
        assert "pair_A_B_tukey_p" in cols

    def test_differential_features_have_low_q(self) -> None:
        """The first 5 features have a strong group effect — should all get q < 0.05."""
        ctx = _make_grouped_ctx(
            n_features=30, n_per_group=6, n_differential_features=5,
            effect_size=3.0, seed=12,
        )
        result = compute_univariate(ctx)
        diff_q = result.table.loc[result.table["feature_id"].isin(["F0", "F1", "F2", "F3", "F4"])]["q_value"]
        # All differential features should be significant
        assert (diff_q < 0.05).all(), f"differential features didn't pass q<0.05: {diff_q.to_list()}"

    def test_null_features_mostly_non_significant(self) -> None:
        ctx = _make_grouped_ctx(
            n_features=50, n_per_group=6, n_differential_features=5,
            effect_size=3.0, seed=13,
        )
        result = compute_univariate(ctx)
        null_q = result.table.loc[~result.table["feature_id"].isin([f"F{i}" for i in range(5)])]["q_value"]
        # Most null features should NOT be significant (BH-FDR controls expected FDR ≤ 0.05)
        false_positive_rate = (null_q < 0.05).mean()
        assert false_positive_rate < 0.20, f"too many null features flagged: {false_positive_rate}"

    def test_q_values_monotone_with_p(self) -> None:
        ctx = _make_grouped_ctx(seed=14)
        result = compute_univariate(ctx)
        # Sort by p; q should be non-decreasing in the same order
        sorted_table = result.table.sort_values("p_value")
        q_sorted = sorted_table["q_value"].dropna().to_numpy()
        assert (np.diff(q_sorted) >= -1e-9).all()

    def test_eta_squared_high_for_differential_features(self) -> None:
        ctx = _make_grouped_ctx(
            n_features=30, n_per_group=6, n_differential_features=5,
            effect_size=4.0, seed=15,
        )
        result = compute_univariate(ctx)
        diff_eta = result.table.loc[
            result.table["feature_id"].isin(["F0", "F1", "F2"])
        ]["eta_squared"]
        # Strong group effect → η² > 0.5
        assert (diff_eta > 0.5).all()

    def test_log_fc_signs_match_direction(self) -> None:
        """For features with mean_A < mean_B, log2 FC(A vs B) should be negative."""
        ctx = _make_grouped_ctx(seed=16, effect_size=4.0)
        result = compute_univariate(ctx)
        # Differential features have monotonic group means: A < B < C
        # so pair_A_B_log2_fc should be negative
        diff_row = result.table.loc[result.table["feature_id"] == "F0"].iloc[0]
        assert diff_row["pair_A_B_log2_fc"] < 0
        assert diff_row["pair_A_C_log2_fc"] < 0
        assert diff_row["pair_B_C_log2_fc"] < 0

    def test_too_few_groups_raises(self) -> None:
        ctx = _make_grouped_ctx(group_names=("A",), seed=17)
        with pytest.raises(ValueError, match="≥ 2 groups"):
            compute_univariate(ctx)

    def test_too_few_samples_raises(self) -> None:
        ctx = _make_grouped_ctx(n_per_group=1, group_names=("A", "B"), seed=18)
        with pytest.raises(ValueError, match="≥ 4 samples"):
            compute_univariate(ctx)

    def test_nan_input_raises(self) -> None:
        ctx = _make_grouped_ctx(seed=19)
        ctx.matrix.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN-free"):
            compute_univariate(ctx)

    def test_log_scale_input_uses_subtractive_fc(self) -> None:
        """If matrix has negative values, treat as already-log and use mean_A - mean_B."""
        ctx = _make_grouped_ctx(seed=20, effect_size=2.0)
        # Force input to look log-scale (subtract a constant)
        ctx.matrix = ctx.matrix - 1000.0  # introduces negatives
        result = compute_univariate(ctx, log_for_fc=True)
        # Should still produce sensible (finite) results
        assert result.table["p_value"].notna().any()
