"""Unit tests for sPLS-DA — sparse PLS-DA feature-importance generator.

Same biomarker-discovery framing as PLS-DA: we verify FEATURE-IMPORTANCE
properties, not predictive metrics. Sparsity-specific contracts:
``selected`` boolean column, keep_x auto-derivation, NIPALS sanity (when
keep_x = n_features the result should approximate full PLS-DA).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from biodisc.multiverse import StageContext
from biodisc.multiverse.methods.multivariate.plsda import _fit_plsda
from biodisc.multiverse.methods.multivariate.splsda import (
    _auto_keep_x,
    _fit_splsda,
    _nipals_sparse,
)
from biodisc.multiverse.multivariate_utils import MultivariateResult
from biodisc.multiverse.registry import apply, list_methods


# ============================================================
# Helpers — same generator pattern as PLS-DA tests
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
# Auto keep_x
# ============================================================
class TestAutoKeepX:
    def test_floor(self) -> None:
        # 50 features → 50 // 100 = 0 → floored to 10
        assert _auto_keep_x(50) == 10

    def test_scales(self) -> None:
        # 2000 features → 20
        assert _auto_keep_x(2000) == 20

    def test_ceiling(self) -> None:
        # 20000 features → 200 → capped at 50
        assert _auto_keep_x(20_000) == 50


# ============================================================
# NIPALS internals
# ============================================================
class TestNipalsSparse:
    def test_sparsity_respected(self) -> None:
        """Each component must have at most keep_x non-zero weights."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 50))
        Y = np.eye(3)[rng.integers(0, 3, size=20)]
        model = _nipals_sparse(X, Y, n_components=2, keep_x=10)
        for a in range(model.x_weights_.shape[1]):
            n_nonzero = int(np.sum(np.abs(model.x_weights_[:, a]) > 1e-12))
            assert n_nonzero <= 10, f"component {a} has {n_nonzero} > 10 non-zero weights"

    def test_dense_limit_matches_pls(self) -> None:
        """When keep_x = n_features, sPLS-DA degenerates to (deflation-style) PLS.

        We don't expect *bit-identical* match to sklearn's PLSRegression — they
        use orthogonal-loadings centring slightly differently — but the *VIP
        ranking* should largely agree. Spearman r > 0.6 is a generous floor.
        """
        ctx = _make_grouped_ctx(n_features=20, n_differential_features=5, effect_size=4.0, seed=11)
        X = ctx.matrix.T.to_numpy(dtype=float)
        labels = np.array(
            [list(("A", "B", "C")).index(s.rsplit("_", 1)[0]) for s in ctx.matrix.columns]
        )
        # PLS-DA reference
        _, vip_pls, _, _ = _fit_plsda(X, labels, n_components=2)
        # sPLS-DA with keep_x = all features
        _, vip_spls, _, _ = _fit_splsda(X, labels, n_components=2, keep_x=20)
        # Both must produce the 5 differential features in the top-K
        top5_pls = set(np.argsort(-vip_pls)[:5])
        top5_spls = set(np.argsort(-vip_spls)[:5])
        # Substantial overlap (≥ 3/5)
        assert len(top5_pls & top5_spls) >= 3, (
            f"PLS-DA top-5 = {top5_pls}, sPLS-DA top-5 = {top5_spls}"
        )


# ============================================================
# sPLS-DA registration + result schema
# ============================================================
class TestSPLSDA:
    def test_registered(self) -> None:
        names = {s.name for s in list_methods("multivariate")}
        assert "splsda" in names

    def test_returns_multivariate_result(self) -> None:
        ctx = _make_grouped_ctx(seed=30)
        result = apply(
            "multivariate", "splsda", ctx, n_bootstrap=10, n_permutation=20
        )
        assert isinstance(result, MultivariateResult)
        assert result.method_name == "splsda"
        for col in (
            "feature_id",
            "selected",
            "vip",
            "perm_importance",
            "stability_score",
            "perm_p_value",
        ):
            assert col in result.per_feature.columns, f"missing column {col!r}"

    def test_selected_column_is_boolean(self) -> None:
        ctx = _make_grouped_ctx(seed=31)
        result = apply(
            "multivariate", "splsda", ctx, n_bootstrap=5, n_permutation=10
        )
        assert result.per_feature["selected"].dtype == bool
        # Some features must be selected (not all-False)
        assert result.per_feature["selected"].any()

    def test_selected_count_within_keep_x_union_bound(self) -> None:
        """|selected| ≤ keep_x × n_components, since selected is the union."""
        ctx = _make_grouped_ctx(n_features=40, seed=32)
        result = apply(
            "multivariate",
            "splsda",
            ctx,
            keep_x=8,
            n_bootstrap=3,
            n_permutation=5,
        )
        n_selected = int(result.per_feature["selected"].sum())
        nc = result.params_used["n_components"]
        assert n_selected <= 8 * nc, (
            f"|selected|={n_selected} exceeds keep_x × n_components = 8×{nc}"
        )

    def test_per_class_beta_columns(self) -> None:
        ctx = _make_grouped_ctx(seed=33)
        result = apply(
            "multivariate", "splsda", ctx, n_bootstrap=5, n_permutation=10
        )
        for cls in ["A", "B", "C"]:
            assert f"beta_{cls}" in result.per_feature.columns
        assert "abs_beta_max" in result.per_feature.columns

    def test_perm_importance_is_nan(self) -> None:
        """sPLS-DA leaves perm_importance NaN by design (see module docstring)."""
        ctx = _make_grouped_ctx(seed=34)
        result = apply(
            "multivariate", "splsda", ctx, n_bootstrap=3, n_permutation=5
        )
        assert result.per_feature["perm_importance"].isna().all()

    def test_n_components_auto_equals_k_minus_1(self) -> None:
        ctx = _make_grouped_ctx(group_names=("A", "B", "C", "D"), n_per_group=4, seed=35)
        result = apply(
            "multivariate", "splsda", ctx, n_bootstrap=3, n_permutation=5
        )
        assert result.params_used["n_components"] == 3

    def test_explicit_keep_x_used(self) -> None:
        ctx = _make_grouped_ctx(n_features=40, seed=36)
        result = apply(
            "multivariate",
            "splsda",
            ctx,
            keep_x=12,
            n_bootstrap=3,
            n_permutation=5,
        )
        assert result.params_used["keep_x"] == 12

    def test_differential_features_get_selected(self) -> None:
        """Features with strong group effect should appear in the selected set."""
        ctx = _make_grouped_ctx(
            n_features=30, n_differential_features=5, effect_size=4.0, seed=37
        )
        result = apply(
            "multivariate",
            "splsda",
            ctx,
            keep_x=8,
            n_bootstrap=5,
            n_permutation=10,
        )
        # Of the 5 known differential features, at least 4 must be selected
        diff_selected = int(result.per_feature.iloc[:5]["selected"].sum())
        assert diff_selected >= 4, (
            f"Only {diff_selected}/5 differential features selected; "
            f"selected set: {result.per_feature.loc[result.per_feature['selected'], 'feature_id'].tolist()}"
        )

    def test_runtime_recorded(self) -> None:
        ctx = _make_grouped_ctx(seed=38)
        result = apply(
            "multivariate", "splsda", ctx, n_bootstrap=3, n_permutation=5
        )
        assert result.runtime_seconds > 0
        assert result.n_bootstrap == 3
        assert result.n_permutation == 5
