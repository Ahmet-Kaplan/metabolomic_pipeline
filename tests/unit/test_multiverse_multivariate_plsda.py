"""Unit tests for PLS-DA feature-importance generator.

Note these tests verify FEATURE IMPORTANCE properties, NOT predictive
metrics like AUC / accuracy / confusion matrix. PLS-DA here is a
biomarker-discovery tool, not a deployable classifier.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from biodisc.multiverse import StageContext
from biodisc.multiverse.multivariate_utils import (
    MultivariateResult,
    _auto_top_k,
    bootstrap_stability,
    permutation_test,
)
from biodisc.multiverse.registry import apply, list_methods


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
    """Build a context with KNOWN differential features."""
    rng = np.random.default_rng(seed)
    n_groups = len(group_names)
    sample_ids: list[str] = []
    cell_lines: list[str] = []
    for g_name in group_names:
        for r in range(n_per_group):
            sample_ids.append(f"{g_name}_{r + 1}")
            cell_lines.append(g_name)
    n_total = len(sample_ids)

    # Generate features. First n_differential_features have group effects;
    # rest are noise.
    matrix_data = np.zeros((n_features, n_total))
    base_means = rng.lognormal(0, 0.5, size=n_features)
    for f in range(n_features):
        for j, sid in enumerate(sample_ids):
            g_idx = group_names.index(sid.rsplit("_", 1)[0])
            offset = (g_idx - n_groups / 2) * effect_size if f < n_differential_features else 0
            matrix_data[f, j] = base_means[f] + offset + rng.normal(0, 0.5)

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
# multivariate_utils
# ============================================================
class TestAutoTopK:
    def test_floor(self) -> None:
        assert _auto_top_k(50) == 10

    def test_scales(self) -> None:
        assert _auto_top_k(2000) == 20

    def test_ceiling(self) -> None:
        assert _auto_top_k(20_000) == 100


class TestBootstrapStability:
    def test_known_signal_features_get_high_stability(self) -> None:
        """Features with strong group effect should show high stability."""
        ctx = _make_grouped_ctx(n_features=30, n_differential_features=5, effect_size=5.0, seed=10)
        X = ctx.matrix.T.to_numpy()
        labels = np.array(
            [list(("A", "B", "C")).index(s.rsplit("_", 1)[0]) for s in ctx.matrix.columns]
        )

        def vip_fn(Xb: np.ndarray, yb: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
            from biodisc.multiverse.methods.multivariate.plsda import _fit_plsda
            if len(np.unique(yb)) < 2:
                return np.zeros(Xb.shape[1])
            try:
                _, vip, _, _ = _fit_plsda(Xb, yb, n_components=2)
                return vip
            except Exception:
                return np.zeros(Xb.shape[1])

        result = bootstrap_stability(
            importance_fn=vip_fn,
            matrix_arr=X,
            labels=labels,
            feature_ids=ctx.matrix.index.tolist(),
            n_bootstrap=30,
            top_k=5,
            seed=42,
        )
        # The 5 differential features should be the most stable
        stability_per_feat = pd.Series(result.stability_score, index=result.feature_ids)
        diff_features = ["F0", "F1", "F2", "F3", "F4"]
        diff_mean_stab = stability_per_feat.loc[diff_features].mean()
        null_mean_stab = stability_per_feat.drop(diff_features).mean()
        assert diff_mean_stab > null_mean_stab + 0.3, (
            f"differential features (mean={diff_mean_stab:.3f}) didn't beat null "
            f"(mean={null_mean_stab:.3f}) by enough"
        )


class TestPermutationTest:
    def test_real_signal_low_pvalue(self) -> None:
        """Features with strong group effect should get low permutation p-values."""
        ctx = _make_grouped_ctx(n_features=20, n_differential_features=3, effect_size=8.0, seed=20)
        X = ctx.matrix.T.to_numpy()
        labels = np.array(
            [list(("A", "B", "C")).index(s.rsplit("_", 1)[0]) for s in ctx.matrix.columns]
        )

        def vip_fn(Xb: np.ndarray, yb: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
            from biodisc.multiverse.methods.multivariate.plsda import _fit_plsda
            if len(np.unique(yb)) < 2:
                return np.zeros(Xb.shape[1])
            try:
                _, vip, _, _ = _fit_plsda(Xb, yb, n_components=2)
                return vip
            except Exception:
                return np.zeros(Xb.shape[1])

        # Get observed VIP first
        from biodisc.multiverse.methods.multivariate.plsda import _fit_plsda
        _, vip_obs, _, _ = _fit_plsda(X, labels, n_components=2)

        result = permutation_test(
            importance_fn=vip_fn,
            matrix_arr=X,
            labels=labels,
            observed_importance=vip_obs,
            feature_ids=ctx.matrix.index.tolist(),
            n_permutation=50,
            seed=42,
        )
        diff_pvals = result.p_value[:3]
        null_pvals = result.p_value[3:]
        # Differential features should have systematically lower p-values
        assert diff_pvals.mean() < null_pvals.mean(), (
            f"differential p-values (mean={diff_pvals.mean():.3f}) should be lower than "
            f"null (mean={null_pvals.mean():.3f})"
        )


# ============================================================
# PLS-DA registration + basic shape
# ============================================================
class TestPLSDA:
    def test_registered(self) -> None:
        names = {s.name for s in list_methods("multivariate")}
        assert "plsda" in names

    def test_returns_multivariate_result(self) -> None:
        ctx = _make_grouped_ctx(seed=30)
        # Tiny n_bootstrap/n_permutation to keep test fast
        result = apply("multivariate", "plsda", ctx, n_bootstrap=10, n_permutation=20)
        assert isinstance(result, MultivariateResult)
        assert result.method_name == "plsda"
        assert "vip" in result.per_feature.columns
        assert "perm_importance" in result.per_feature.columns
        assert "stability_score" in result.per_feature.columns
        assert "perm_p_value" in result.per_feature.columns

    def test_per_class_beta_columns(self) -> None:
        ctx = _make_grouped_ctx(seed=31)
        result = apply("multivariate", "plsda", ctx, n_bootstrap=5, n_permutation=10)
        # 3 groups → beta_A, beta_B, beta_C
        for cls in ["A", "B", "C"]:
            assert f"beta_{cls}" in result.per_feature.columns
        assert "abs_beta_max" in result.per_feature.columns

    def test_n_components_auto_equals_k_minus_1(self) -> None:
        ctx = _make_grouped_ctx(group_names=("A", "B", "C", "D"), n_per_group=4, seed=32)
        result = apply("multivariate", "plsda", ctx, n_bootstrap=3, n_permutation=5)
        # K=4 → n_components should be 3
        assert result.params_used["n_components"] == 3

    def test_explicit_n_components_used(self) -> None:
        ctx = _make_grouped_ctx(seed=33)
        result = apply("multivariate", "plsda", ctx, n_components=2, n_bootstrap=3, n_permutation=5)
        assert result.params_used["n_components"] == 2

    def test_differential_features_high_vip(self) -> None:
        """Features with known group effect should get higher VIP than null features."""
        ctx = _make_grouped_ctx(n_features=20, n_differential_features=5, effect_size=4.0, seed=34)
        result = apply("multivariate", "plsda", ctx, n_bootstrap=5, n_permutation=10)
        diff_vip = result.per_feature.iloc[:5]["vip"].mean()
        null_vip = result.per_feature.iloc[5:]["vip"].mean()
        assert diff_vip > null_vip * 1.5, (
            f"differential VIP (mean={diff_vip:.2f}) should be > 1.5× null (mean={null_vip:.2f})"
        )

    def test_runtime_recorded(self) -> None:
        ctx = _make_grouped_ctx(seed=35)
        result = apply("multivariate", "plsda", ctx, n_bootstrap=3, n_permutation=5)
        assert result.runtime_seconds > 0
        assert result.n_bootstrap == 3
        assert result.n_permutation == 5
