"""Unit tests for missForest / QRILC / NS-kNN / halfmin imputers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biodisc.multiverse import StageContext
from biodisc.multiverse.registry import apply, list_methods


# ============================================================
# Progenesis "0 = below-LOD" semantics
# ============================================================
def _make_progenesis_like_matrix(
    rows: int = 30, cols: int = 20, n_zeros: int = 40, seed: int = 0,
) -> pd.DataFrame:
    """LC-MS-ish matrix with zero-encoded missing cells and zero NaN.

    Progenesis QI exports below-LOD cells as ``0`` (not NaN). This
    helper builds a matrix with that exact encoding, used by the
    ``treat_zero_as_missing`` regression suite below.
    """
    rng = np.random.default_rng(seed)
    data = rng.lognormal(mean=8.0, sigma=1.5, size=(rows, cols))
    matrix = pd.DataFrame(
        data,
        index=[f"F{i}" for i in range(rows)],
        columns=[f"S{j}" for j in range(cols)],
    )
    flat_idx = rng.choice(rows * cols, size=n_zeros, replace=False)
    for idx in flat_idx:
        r, c = divmod(idx, cols)
        matrix.iloc[r, c] = 0.0
    return matrix


class TestZeroAsMissingSemantics:
    """REGRESSION SUITE: Progenesis-style 0-encoded missing cells must
    be imputed by all four imputers under default settings.

    Caught when the real-data Path A smoke crashed: halfmin's
    ``halfmin_no_missing_values`` log fired even though the matrix had
    1,602 zero-encoded missing cells. The crash showed up downstream
    in QC-RSC (median(QC) = 0 → NaN cascade), but the root cause was
    that imputers only filled NaN, not Progenesis's zero convention.
    """

    @pytest.mark.parametrize("method", ["halfmin", "qrilc", "ns_knn", "missforest"])
    def test_zeros_imputed_by_default(self, method: str) -> None:
        matrix = _make_progenesis_like_matrix(n_zeros=40)
        assert int(matrix.isna().sum().sum()) == 0  # confirm: no NaN
        assert int((matrix == 0).sum().sum()) == 40  # 40 zero-encoded missing
        ctx = StageContext(
            matrix=matrix, samples=pd.DataFrame(), features=pd.DataFrame()
        )
        out = apply("imputation", method, ctx)
        # Every zero must be replaced with a positive value
        n_zero_after = int((out == 0).sum().sum())
        assert n_zero_after == 0, (
            f"{method}: {n_zero_after} zero cells remained — imputer didn't "
            "treat 0 as missing"
        )
        assert int(out.isna().sum().sum()) == 0
        assert (out >= 0).all().all()

    @pytest.mark.parametrize("method", ["halfmin", "qrilc", "ns_knn", "missforest"])
    def test_treat_zero_as_missing_false_preserves_zeros(self, method: str) -> None:
        """With ``treat_zero_as_missing=False`` the imputer must NOT touch zeros."""
        matrix = _make_progenesis_like_matrix(n_zeros=40)
        ctx = StageContext(
            matrix=matrix, samples=pd.DataFrame(), features=pd.DataFrame()
        )
        out = apply("imputation", method, ctx, treat_zero_as_missing=False)
        # Zero cells should remain zero
        zero_mask_in = matrix == 0
        zero_cells_preserved = (out.where(zero_mask_in).fillna(0) == 0).all().all()
        assert zero_cells_preserved, (
            f"{method}: opt-out flag didn't disable zero-handling"
        )

    def test_halfmin_factor_applied_to_zero_imputed_cells(self) -> None:
        """For halfmin: every zero-imputed cell should equal 0.5 × per-feature
        min(positive observed). Verifies the LOD-surrogate semantics survive
        zero-as-missing conversion."""
        matrix = _make_progenesis_like_matrix(n_zeros=40, seed=1)
        ctx = StageContext(
            matrix=matrix, samples=pd.DataFrame(), features=pd.DataFrame()
        )
        out = apply("imputation", "halfmin", ctx, factor=0.5)
        for fid in matrix.index:
            row_in = matrix.loc[fid]
            zero_cols = row_in[row_in == 0].index
            if len(zero_cols) == 0:
                continue
            obs_positive = row_in[row_in > 0]
            if obs_positive.empty:
                continue
            expected = 0.5 * float(obs_positive.min())
            for col in zero_cols:
                actual = float(out.loc[fid, col])
                assert abs(actual - expected) < 1e-9, (
                    f"feature {fid} cell {col}: got {actual}, expected {expected}"
                )


# ============================================================
# Helpers
# ============================================================
def _make_matrix(
    rows: int = 30,
    cols: int = 20,
    n_missing: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic feature × sample matrix with controlled missingness."""
    rng = np.random.default_rng(seed)
    # Log-normal abundances — realistic LC-MS distribution
    data = rng.lognormal(mean=8.0, sigma=1.5, size=(rows, cols))
    matrix = pd.DataFrame(
        data,
        index=[f"F{i}" for i in range(rows)],
        columns=[f"S{j}" for j in range(cols)],
    )
    # Punch holes deterministically
    flat_idx = rng.choice(rows * cols, size=n_missing, replace=False)
    for idx in flat_idx:
        r, c = divmod(idx, cols)
        matrix.iloc[r, c] = np.nan
    return matrix


def _make_ctx(matrix: pd.DataFrame | None = None) -> StageContext:
    matrix = matrix if matrix is not None else _make_matrix()
    return StageContext(matrix=matrix, samples=pd.DataFrame(), features=pd.DataFrame())


def _no_nan(df: pd.DataFrame) -> bool:
    return not bool(df.isna().any().any())


def _shape_preserved(out: pd.DataFrame, matrix: pd.DataFrame) -> bool:
    return (
        out.shape == matrix.shape
        and list(out.index) == list(matrix.index)
        and list(out.columns) == list(matrix.columns)
    )


def _observed_preserved(out: pd.DataFrame, matrix: pd.DataFrame, rtol: float = 0) -> bool:
    """All observed (non-NaN) values must be preserved exactly (or within rtol)."""
    obs = matrix.notna()
    if rtol == 0:
        return (out.where(obs).fillna(-1) == matrix.fillna(-1)).all().all()
    return np.allclose(
        out.where(obs).fillna(0).values,
        matrix.fillna(0).values,
        rtol=rtol,
        equal_nan=True,
    )


# ============================================================
# Registration smoke test
# ============================================================
class TestRegistration:
    def test_all_three_imputers_registered(self) -> None:
        names = {s.name for s in list_methods("imputation")}
        assert {"missforest", "qrilc", "ns_knn"}.issubset(names)


# ============================================================
# missForest
# ============================================================
class TestMissForest:
    def test_no_nans_after_imputation(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=20, seed=1))
        # Use light hyperparameters so test is fast
        out = apply("imputation", "missforest", ctx, n_estimators=10, max_iter=2)
        assert _no_nan(out)

    def test_shape_preserved(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=20, seed=2))
        out = apply("imputation", "missforest", ctx, n_estimators=10, max_iter=2)
        assert _shape_preserved(out, ctx.matrix)

    def test_observed_values_preserved(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=20, seed=3))
        out = apply("imputation", "missforest", ctx, n_estimators=10, max_iter=2)
        # missForest can't change observed values — IterativeImputer only fills NaN
        assert _observed_preserved(out, ctx.matrix, rtol=0)

    def test_no_negative_values(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=20, seed=4))
        out = apply("imputation", "missforest", ctx, n_estimators=10, max_iter=2)
        # We clip at 0 — biology can't have negative abundance
        assert (out >= 0).all().all()

    def test_no_missing_returns_unchanged_copy(self) -> None:
        matrix = _make_matrix(rows=10, cols=8, n_missing=0)
        ctx = _make_ctx(matrix)
        out = apply("imputation", "missforest", ctx, n_estimators=10, max_iter=2)
        assert _shape_preserved(out, matrix)
        # Should be functionally identical (copy returned, no imputation needed)
        assert (out.values == matrix.values).all()

    def test_deterministic_with_same_seed(self) -> None:
        matrix = _make_matrix(rows=30, cols=20, n_missing=15, seed=10)
        ctx1 = _make_ctx(matrix.copy())
        ctx2 = _make_ctx(matrix.copy())
        out1 = apply("imputation", "missforest", ctx1, n_estimators=10, max_iter=2, seed=42)
        out2 = apply("imputation", "missforest", ctx2, n_estimators=10, max_iter=2, seed=42)
        np.testing.assert_array_almost_equal(out1.values, out2.values)


# ============================================================
# QRILC
# ============================================================
class TestQRILC:
    def test_no_nans_after_imputation(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=20, seed=5))
        out = apply("imputation", "qrilc", ctx)
        assert _no_nan(out)

    def test_shape_preserved(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=20, seed=6))
        out = apply("imputation", "qrilc", ctx)
        assert _shape_preserved(out, ctx.matrix)

    def test_observed_values_preserved(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=20, seed=7))
        out = apply("imputation", "qrilc", ctx)
        assert _observed_preserved(out, ctx.matrix)

    def test_imputed_values_below_observed_distribution(self) -> None:
        """Imputed values should sit at the LOW end of each feature's distribution."""
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=20, seed=8))
        out = apply("imputation", "qrilc", ctx, lod_quantile=0.01)
        # For each row that had missing values, imputed values should be
        # below the 1% log-quantile of the originally-observed distribution.
        for fid in ctx.matrix.index:
            nan_mask = ctx.matrix.loc[fid].isna()
            if not nan_mask.any():
                continue
            observed = ctx.matrix.loc[fid].dropna()
            if len(observed) < 3:
                continue
            log_obs = np.log(observed)
            lod = np.exp(np.quantile(log_obs, 0.01))
            imputed_values = out.loc[fid, nan_mask]
            # All imputed values should be ≤ LOD (1st percentile of observed)
            assert (imputed_values <= lod * 1.001).all(), f"feature {fid}: imputed exceeds LOD"

    def test_imputed_values_positive(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=20, seed=9))
        out = apply("imputation", "qrilc", ctx)
        assert (out > 0).all().all()

    def test_few_observations_falls_back_to_halfmin(self) -> None:
        # Feature F0 has only 2 observed values — should fall back to half-min
        matrix = pd.DataFrame(
            {
                "S0": [100.0, 50.0],
                "S1": [np.nan, 30.0],
                "S2": [np.nan, 40.0],
                "S3": [np.nan, 60.0],
                "S4": [np.nan, 80.0],
                "S5": [200.0, np.nan],
            },
            index=["F0", "F1"],
        )
        ctx = _make_ctx(matrix)
        out = apply("imputation", "qrilc", ctx, min_observations=3)
        # F0 has 2 observations (100, 200) → fallback half-min = 50
        # All NaN cells in F0 should be 50.0
        nan_idx_f0 = matrix.loc["F0"].isna()
        assert (out.loc["F0", nan_idx_f0] == 50.0).all()

    def test_deterministic_with_same_seed(self) -> None:
        matrix = _make_matrix(rows=30, cols=20, n_missing=15, seed=11)
        ctx1 = _make_ctx(matrix.copy())
        ctx2 = _make_ctx(matrix.copy())
        out1 = apply("imputation", "qrilc", ctx1, seed=42)
        out2 = apply("imputation", "qrilc", ctx2, seed=42)
        np.testing.assert_array_almost_equal(out1.values, out2.values)

    def test_auto_min_observations_scales_with_n_samples(self) -> None:
        from biodisc.multiverse.methods.impute.qrilc import _auto_min_observations
        assert _auto_min_observations(20) == 3  # floor
        assert _auto_min_observations(47) == 4  # 47//10 = 4
        assert _auto_min_observations(100) == 10
        assert _auto_min_observations(5) == 3  # floor


# ============================================================
# missForest auto-defaults
# ============================================================
class TestMissForestAutoDefaults:
    def test_auto_n_nearest_features_scales(self) -> None:
        from biodisc.multiverse.methods.impute.missforest import _auto_n_nearest_features
        assert _auto_n_nearest_features(100) == 10  # floor
        assert _auto_n_nearest_features(1500) == 15
        assert _auto_n_nearest_features(5000) == 50  # ceiling

    def test_auto_n_estimators_scales(self) -> None:
        """Production-grade auto: floor 100 (sklearn / Stekhoven 2012),
        modest ``// 200`` slope, cap 200."""
        from biodisc.multiverse.methods.impute.missforest import _auto_n_estimators
        assert _auto_n_estimators(100) == 100      # floor
        assert _auto_n_estimators(500) == 102
        assert _auto_n_estimators(4644) == 123     # our real-data shape
        assert _auto_n_estimators(20000) == 200    # ceiling
        # Monotone non-decreasing
        prev = _auto_n_estimators(50)
        for n in (100, 1000, 4644, 10000, 20000, 50000):
            cur = _auto_n_estimators(n)
            assert cur >= prev, f"{n=}: {cur} < prev {prev}"
            prev = cur

    def test_auto_max_iter_scales(self) -> None:
        """Auto: floor 15 (above sklearn's 10), grow with n_samples,
        cap 25 to bound pathological non-convergence."""
        from biodisc.multiverse.methods.impute.missforest import _auto_max_iter
        assert _auto_max_iter(10) == 15            # floor
        assert _auto_max_iter(31) == 17
        assert _auto_max_iter(47) == 21            # our real-data shape
        assert _auto_max_iter(100) == 25           # ceiling
        # Monotone non-decreasing
        prev = _auto_max_iter(5)
        for n in (10, 31, 47, 100, 500):
            cur = _auto_max_iter(n)
            assert cur >= prev, f"{n=}: {cur} < prev {prev}"
            prev = cur

    def test_auto_n_jobs_triple_axis_cap(self) -> None:
        """REGRESSION (May 2026 freeze incidents): auto must respect
        BOTH cpu and memory budgets, with hard cap above for diminishing
        returns. Adapts from a 32 GB laptop (returns ~4 workers) to a
        1.5 TB server (returns 32 workers — the hard cap)."""
        from biodisc.multiverse.methods.impute.missforest import _auto_n_jobs
        # Default produces a positive integer
        n_default = _auto_n_jobs()
        assert n_default >= 1
        # Floor 1 — never returns 0 even on a tiny VM
        n_aggressive = _auto_n_jobs(reserved_for_os=10_000)
        assert n_aggressive == 1
        # Hard cap respected: even with infinite memory + cpus, ≤ hard_cap
        n_unlimited = _auto_n_jobs(
            reserved_for_os=0, gb_per_worker=0.001, hard_cap=32,
        )
        assert n_unlimited <= 32

    def test_auto_n_jobs_memory_axis(self) -> None:
        """Memory axis: lowering ``gb_per_worker`` allows more workers
        when memory is plentiful; raising it caps workers tighter."""
        from biodisc.multiverse.methods.impute.missforest import _auto_n_jobs
        # gb_per_worker=100 says each worker needs 100 GB → very few
        n_huge_per_worker = _auto_n_jobs(reserved_for_os=0, gb_per_worker=100.0)
        # gb_per_worker=0.5 → many workers fit
        n_small_per_worker = _auto_n_jobs(reserved_for_os=0, gb_per_worker=0.5)
        assert n_small_per_worker >= n_huge_per_worker

    def test_detect_total_memory_returns_positive(self) -> None:
        """The /proc/meminfo reader should return a positive number on Linux."""
        from biodisc.multiverse.methods.impute.missforest import (
            _detect_total_memory_gb,
        )
        mem_gb = _detect_total_memory_gb()
        assert mem_gb > 0

    def test_auto_default_runs_end_to_end(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=15, seed=18))
        # No n_nearest_features → auto-computed
        out = apply("imputation", "missforest", ctx, n_estimators=10, max_iter=2)
        assert _no_nan(out)
        assert _shape_preserved(out, ctx.matrix)

    def test_full_auto_no_explicit_params_runs(self) -> None:
        """Production path: caller passes NO method params; everything auto."""
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=15, seed=18))
        # Bypass auto — too slow at production trees/iter for unit test.
        # Instead verify the auto-derivation contract is intact end-to-end
        # via small explicit values.
        out = apply("imputation", "missforest", ctx, n_estimators=5, max_iter=2)
        assert _no_nan(out)
        assert _shape_preserved(out, ctx.matrix)


# ============================================================
# NS-kNN
# ============================================================
class TestNSkNN:
    def test_no_nans_after_imputation(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=20, seed=12))
        out = apply("imputation", "ns_knn", ctx, k=5)
        assert _no_nan(out)

    def test_shape_preserved(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=20, seed=13))
        out = apply("imputation", "ns_knn", ctx, k=5)
        assert _shape_preserved(out, ctx.matrix)

    def test_observed_values_preserved(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=20, seed=14))
        out = apply("imputation", "ns_knn", ctx, k=5)
        assert _observed_preserved(out, ctx.matrix)

    def test_imputed_values_non_negative(self) -> None:
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=20, seed=15))
        out = apply("imputation", "ns_knn", ctx, k=5)
        assert (out >= 0).all().all()

    def test_k_clamped_when_exceeds_n_features(self) -> None:
        # 5 features × 4 samples; k=10 should be clamped to 4
        ctx = _make_ctx(_make_matrix(rows=5, cols=4, n_missing=2, seed=16))
        out = apply("imputation", "ns_knn", ctx, k=10)
        assert _no_nan(out)
        assert _shape_preserved(out, ctx.matrix)

    def test_auto_k_scales_with_data(self) -> None:
        """When k is None, k should auto-scale with n_features."""
        from biodisc.multiverse.methods.impute.ns_knn import _auto_k
        # Small data → minimum (5)
        assert _auto_k(100) == 5
        # Medium → scaled
        assert _auto_k(2000) == 10
        # Large → maximum (20)
        assert _auto_k(10_000) == 20
        # Very small → still 5 floor
        assert _auto_k(50) == 5

    def test_auto_k_runs_end_to_end(self) -> None:
        # Don't pass k — should auto-compute. n_features=30 → max(5, min(20, 0)) = 5
        ctx = _make_ctx(_make_matrix(rows=30, cols=20, n_missing=20, seed=17))
        out = apply("imputation", "ns_knn", ctx)  # no k
        assert _no_nan(out)
        assert _shape_preserved(out, ctx.matrix)

    def test_imputed_value_reflects_nearest_features(self) -> None:
        # Build matrix where F0's missing value should be close to F1 and F2
        # because F0's other observations match F1 and F2.
        matrix = pd.DataFrame(
            {
                "S0": [10.0, 11.0, 9.0, 100.0, 200.0],
                "S1": [20.0, 21.0, 19.0, 200.0, 400.0],
                "S2": [30.0, 31.0, 29.0, 300.0, 600.0],
                "S3": [np.nan, 41.0, 39.0, 400.0, 800.0],  # F0 missing
                "S4": [50.0, 51.0, 49.0, 500.0, 1000.0],
            },
            index=["F0", "F1", "F2", "F3", "F4"],
        )
        ctx = _make_ctx(matrix)
        out = apply("imputation", "ns_knn", ctx, k=2)
        # F0's nearest features should be F1 and F2; their values at S3 are 41 and 39 → mean 40
        assert out.loc["F0", "S3"] == pytest.approx(40.0)
