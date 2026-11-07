"""Unit tests for the multiverse registry + StageContext + halfmin imputer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biodisc.multiverse import MethodSpec, StageContext, get, list_methods, register
from biodisc.multiverse.registry import apply


# ============================================================
# Helpers
# ============================================================
def _make_matrix(rows: int = 5, cols: int = 4, n_missing: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.uniform(100, 1000, size=(rows, cols))
    matrix = pd.DataFrame(
        data,
        index=[f"F{i}" for i in range(rows)],
        columns=[f"S{j}" for j in range(cols)],
    )
    # Punch n_missing holes deterministically
    flat_idx = rng.choice(rows * cols, size=n_missing, replace=False)
    for idx in flat_idx:
        r, c = divmod(idx, cols)
        matrix.iloc[r, c] = np.nan
    return matrix


def _make_ctx(matrix: pd.DataFrame | None = None) -> StageContext:
    matrix = matrix if matrix is not None else _make_matrix()
    return StageContext(
        matrix=matrix,
        samples=pd.DataFrame(),
        features=pd.DataFrame(),
    )


# ============================================================
# Registry mechanics
# ============================================================
class TestRegistry:
    def test_halfmin_is_registered(self) -> None:
        spec = get("imputation", "halfmin")
        assert spec.name == "halfmin"
        assert spec.category == "imputation"
        assert callable(spec.callable)

    def test_unknown_method_raises_with_helpful_message(self) -> None:
        with pytest.raises(KeyError, match="Unknown imputation method"):
            get("imputation", "no_such_method")

    def test_list_methods_filters_by_category(self) -> None:
        imputers = list_methods(category="imputation")
        assert all(s.category == "imputation" for s in imputers)
        assert any(s.name == "halfmin" for s in imputers)

    def test_list_all_methods(self) -> None:
        all_methods = list_methods()
        assert len(all_methods) >= 1

    def test_register_is_idempotent(self) -> None:
        # Re-registering with the same key overwrites
        original = get("imputation", "halfmin")
        original_callable = original.callable
        # Re-register with a different docstring
        register(
            MethodSpec(
                name="halfmin",
                callable=original_callable,
                description="Re-registered",
                citation="test",
                category="imputation",
            )
        )
        spec = get("imputation", "halfmin")
        assert spec.description == "Re-registered"
        # Restore so other tests aren't affected
        register(original)

    def test_apply_convenience(self) -> None:
        ctx = _make_ctx()
        out = apply("imputation", "halfmin", ctx)
        assert out.shape == ctx.matrix.shape
        assert int(out.isna().sum().sum()) == 0


# ============================================================
# StageContext properties
# ============================================================
class TestStageContext:
    def test_n_features_and_samples(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=10, cols=6, n_missing=0))
        assert ctx.n_features == 10
        assert ctx.n_samples == 6

    def test_missing_counts(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=5, cols=4, n_missing=3))
        assert ctx.n_missing == 3
        assert ctx.pct_missing == pytest.approx(3 / 20 * 100)

    def test_no_missing(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=5, cols=4, n_missing=0))
        assert ctx.n_missing == 0
        assert ctx.pct_missing == 0.0


# ============================================================
# Halfmin imputer correctness
# ============================================================
class TestHalfmin:
    def test_imputed_matrix_has_no_nans(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=5, cols=4, n_missing=4))
        out = apply("imputation", "halfmin", ctx)
        assert int(out.isna().sum().sum()) == 0

    def test_imputed_value_equals_half_per_feature_min(self) -> None:
        # F0 has known min = 100; missing should be filled with 50
        matrix = pd.DataFrame(
            {
                "S0": [100.0, np.nan, 200.0],
                "S1": [200.0, 50.0, np.nan],
                "S2": [np.nan, 80.0, 300.0],
            },
            index=["F0", "F1", "F2"],
        )
        ctx = _make_ctx(matrix)
        out = apply("imputation", "halfmin", ctx)
        # F0 has min 100 → filled = 50
        assert out.loc["F0", "S2"] == pytest.approx(50.0)
        # F1 has min 50 → filled = 25
        assert out.loc["F1", "S0"] == pytest.approx(25.0)
        # F2 has min 200 → filled = 100
        assert out.loc["F2", "S1"] == pytest.approx(100.0)

    def test_factor_param_is_respected(self) -> None:
        matrix = pd.DataFrame(
            {"S0": [100.0, np.nan], "S1": [200.0, 50.0]},
            index=["F0", "F1"],
        )
        ctx = _make_ctx(matrix)
        out = apply("imputation", "halfmin", ctx, factor=0.25)
        # F1 min = 50 → filled = 12.5
        assert out.loc["F1", "S0"] == pytest.approx(12.5)

    def test_observed_values_preserved(self) -> None:
        matrix = pd.DataFrame(
            {
                "S0": [100.0, np.nan, 200.0],
                "S1": [200.0, 50.0, np.nan],
                "S2": [np.nan, 80.0, 300.0],
            },
            index=["F0", "F1", "F2"],
        )
        ctx = _make_ctx(matrix)
        out = apply("imputation", "halfmin", ctx)
        # Original observed values must not change
        assert out.loc["F0", "S0"] == 100.0
        assert out.loc["F0", "S1"] == 200.0
        assert out.loc["F1", "S1"] == 50.0
        assert out.loc["F1", "S2"] == 80.0
        assert out.loc["F2", "S0"] == 200.0
        assert out.loc["F2", "S2"] == 300.0

    def test_all_nan_feature_uses_fallback(self) -> None:
        matrix = pd.DataFrame(
            {
                "S0": [100.0, np.nan],
                "S1": [200.0, np.nan],
            },
            index=["F0", "F1"],
        )
        ctx = _make_ctx(matrix)
        out = apply("imputation", "halfmin", ctx, fallback_value=42.0)
        assert out.loc["F1", "S0"] == 42.0
        assert out.loc["F1", "S1"] == 42.0
        # F0 still gets normal half-min treatment (no NaN to fill anyway)
        assert out.loc["F0", "S0"] == 100.0

    def test_no_missing_returns_unchanged_copy(self) -> None:
        matrix = pd.DataFrame({"S0": [1.0, 2.0], "S1": [3.0, 4.0]}, index=["F0", "F1"])
        ctx = _make_ctx(matrix)
        out = apply("imputation", "halfmin", ctx)
        assert out.equals(matrix)
        # But it's a copy — modifying out shouldn't change matrix
        out.iloc[0, 0] = 999.0
        assert matrix.iloc[0, 0] == 1.0

    def test_shape_preserved(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=8, cols=6, n_missing=10))
        out = apply("imputation", "halfmin", ctx)
        assert out.shape == ctx.matrix.shape
        assert list(out.index) == list(ctx.matrix.index)
        assert list(out.columns) == list(ctx.matrix.columns)
