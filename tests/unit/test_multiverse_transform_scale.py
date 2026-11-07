"""Unit tests for log2 / glog transforms + Pareto scaling + transform diagnostics."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from biodisc.multiverse import StageContext
from biodisc.multiverse.diagnostics import (
    TransformationDiagnostics,
    compute_transformation_diagnostics,
    heteroscedasticity_slope,
    ks_normality_fraction,
    mean_abs_skewness,
)
from biodisc.multiverse.registry import apply, list_methods


# ============================================================
# Helpers
# ============================================================
def _make_ln_matrix(
    n_features: int = 50,
    n_qc: int = 6,
    n_bio: int = 12,
    additive_noise_sd: float = 0.0,
    seed: int = 0,
) -> StageContext:
    """LC-MS abundance matrix with configurable noise model.

    Per-sample values follow the Durbin-Huber 2002 noise model:

        x_{f,s} = baseline_f × multiplicative_noise + additive_noise

    Multiplicative is log-normal (ion-source / dilution variability).
    Additive is Gaussian (LOD/electronic noise floor — set to 0 to get
    pure-multiplicative data, which is log2's ideal regime; set non-zero
    to get the regime where glog wins over log2).
    """
    rng = np.random.default_rng(seed)
    baseline = rng.lognormal(mean=8.0, sigma=1.5, size=n_features)
    n_total = n_qc + n_bio
    sample_ids = [f"QC_{i + 1}" for i in range(n_qc)] + [
        f"S_{i + 1}" for i in range(n_bio)
    ]
    matrix_data = np.zeros((n_features, n_total))
    for j, sid in enumerate(sample_ids):
        mult_sigma = 0.1 if sid.startswith("QC_") else 0.6
        mult_noise = rng.lognormal(mean=0.0, sigma=mult_sigma, size=n_features)
        if additive_noise_sd > 0:
            add_noise = rng.normal(0.0, additive_noise_sd, size=n_features)
        else:
            add_noise = np.zeros(n_features)
        matrix_data[:, j] = np.maximum(baseline * mult_noise + add_noise, 1e-3)
    matrix = pd.DataFrame(
        matrix_data,
        index=[f"F{i}" for i in range(n_features)],
        columns=sample_ids,
    )
    samples = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "raw_sample_id": sample_ids,
            "injection_type": ["qc"] * n_qc + ["biological"] * n_bio,
            "cell_line": [None] * n_qc + ["X"] * n_bio,
            "replicate": list(range(1, n_qc + 1)) + list(range(1, n_bio + 1)),
            "dilution_level": [None] * n_total,
            "acq_date": [datetime(2026, 1, 9)] * n_total,
            "column_order": list(range(n_total)),
        }
    )
    return StageContext(matrix=matrix, samples=samples, features=pd.DataFrame())


def _no_nan(df: pd.DataFrame) -> bool:
    return not bool(df.isna().any().any())


def _shape_preserved(out: pd.DataFrame, ctx: StageContext) -> bool:
    return (
        out.shape == ctx.matrix.shape
        and list(out.index) == list(ctx.matrix.index)
        and list(out.columns) == list(ctx.matrix.columns)
    )


# ============================================================
# Registration
# ============================================================
class TestRegistration:
    def test_two_transforms_registered(self) -> None:
        names = {s.name for s in list_methods("transformation")}
        assert {"log2", "glog"}.issubset(names)

    def test_pareto_registered(self) -> None:
        names = {s.name for s in list_methods("scaling")}
        assert "pareto" in names


# ============================================================
# log2
# ============================================================
class TestLog2:
    def test_no_nans(self) -> None:
        ctx = _make_ln_matrix(seed=1)
        out = apply("transformation", "log2", ctx)
        assert _no_nan(out)

    def test_shape_preserved(self) -> None:
        ctx = _make_ln_matrix(seed=2)
        out = apply("transformation", "log2", ctx)
        assert _shape_preserved(out, ctx)

    def test_reduces_skewness(self) -> None:
        """log2 should reduce mean |skewness| of right-skewed log-normal data."""
        ctx = _make_ln_matrix(seed=3)
        skew_before = mean_abs_skewness(ctx.matrix)
        out = apply("transformation", "log2", ctx)
        skew_after = mean_abs_skewness(out)
        assert skew_after < skew_before

    def test_reduces_heteroscedasticity_slope(self) -> None:
        """log2 should pull the heteroscedasticity slope toward 0."""
        ctx = _make_ln_matrix(seed=4)
        slope_before = heteroscedasticity_slope(ctx.matrix)
        out = apply("transformation", "log2", ctx)
        slope_after = heteroscedasticity_slope(out)
        # Pre-log slope on log-normal data is typically ~1; post-log should be much smaller.
        assert abs(slope_after) < abs(slope_before)

    def test_auto_epsilon_below_smallest(self) -> None:
        from biodisc.multiverse.methods.transform.log2 import _auto_epsilon
        matrix = pd.DataFrame({"S1": [1.0, 100.0], "S2": [50.0, 200.0]})
        eps = _auto_epsilon(matrix)
        assert eps == pytest.approx(0.5)  # 1.0 / 2

    def test_user_epsilon_used(self) -> None:
        ctx = _make_ln_matrix(seed=5)
        out_default = apply("transformation", "log2", ctx)
        out_user = apply("transformation", "log2", ctx, epsilon=10.0)
        # Different ε → different output
        assert not np.allclose(out_default.values, out_user.values)

    def test_zero_epsilon_raises(self) -> None:
        ctx = _make_ln_matrix(seed=6)
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            apply("transformation", "log2", ctx, epsilon=0.0)

    def test_negative_input_raises(self) -> None:
        ctx = _make_ln_matrix(seed=7)
        ctx.matrix.iloc[0, 0] = -1.0
        with pytest.raises(ValueError, match="negative input"):
            apply("transformation", "log2", ctx)

    def test_nan_input_raises(self) -> None:
        ctx = _make_ln_matrix(seed=8)
        ctx.matrix.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN-free"):
            apply("transformation", "log2", ctx)


# ============================================================
# glog
# ============================================================
class TestGLog:
    def test_no_nans(self) -> None:
        ctx = _make_ln_matrix(seed=10)
        out = apply("transformation", "glog", ctx)
        assert _no_nan(out)

    def test_shape_preserved(self) -> None:
        ctx = _make_ln_matrix(seed=11)
        out = apply("transformation", "glog", ctx)
        assert _shape_preserved(out, ctx)

    def test_reduces_skewness(self) -> None:
        ctx = _make_ln_matrix(seed=12)
        skew_before = mean_abs_skewness(ctx.matrix)
        out = apply("transformation", "glog", ctx)
        skew_after = mean_abs_skewness(out)
        assert skew_after < skew_before

    def test_reduces_heteroscedasticity_slope(self) -> None:
        """glog with auto-λ is designed for additive+multiplicative noise — the
        regime where pure log2 over-corrects. Test on that data."""
        ctx = _make_ln_matrix(additive_noise_sd=100.0, seed=13)
        slope_before = heteroscedasticity_slope(ctx.matrix)
        out = apply("transformation", "glog", ctx)
        slope_after = heteroscedasticity_slope(out)
        # On additive+mult data, glog should give |slope| < |slope_before|
        assert abs(slope_after) < abs(slope_before)

    def test_auto_lambda_uses_qc_when_available(self) -> None:
        from biodisc.multiverse.methods.transform.glog import _auto_lambda
        ctx = _make_ln_matrix(n_qc=6, n_bio=12, seed=14)
        lambda_val, source = _auto_lambda(ctx)
        assert source == "qc_median_sd"
        assert lambda_val > 0

    def test_auto_lambda_falls_back_when_few_qcs(self) -> None:
        from biodisc.multiverse.methods.transform.glog import _auto_lambda
        ctx = _make_ln_matrix(n_qc=2, n_bio=12, seed=15)
        lambda_val, source = _auto_lambda(ctx)
        assert source == "all_median_sd"
        assert lambda_val > 0

    def test_user_lambda_used(self) -> None:
        ctx = _make_ln_matrix(seed=16)
        out_a = apply("transformation", "glog", ctx, lambda_param=10.0)
        out_b = apply("transformation", "glog", ctx, lambda_param=1000.0)
        # Different lambdas → different output
        assert not np.allclose(out_a.values, out_b.values)

    def test_zero_lambda_raises(self) -> None:
        ctx = _make_ln_matrix(seed=17)
        with pytest.raises(ValueError, match="lambda_param must be > 0"):
            apply("transformation", "glog", ctx, lambda_param=0.0)

    def test_at_high_intensity_glog_approaches_log2(self) -> None:
        """At x >> λ, glog should give roughly log2(x)."""
        ctx = _make_ln_matrix(seed=18)
        # Force matrix to large values; small lambda
        ctx.matrix.iloc[:, :] = 1e6
        out = apply("transformation", "glog", ctx, lambda_param=1.0)
        # log2(1e6) ≈ 19.93
        assert (out > 19).all().all()
        assert (out < 20.5).all().all()


# ============================================================
# Pareto
# ============================================================
class TestPareto:
    def test_no_nans(self) -> None:
        ctx = _make_ln_matrix(seed=20)
        out = apply("scaling", "pareto", ctx)
        assert _no_nan(out)

    def test_shape_preserved(self) -> None:
        ctx = _make_ln_matrix(seed=21)
        out = apply("scaling", "pareto", ctx)
        assert _shape_preserved(out, ctx)

    def test_per_feature_mean_zero(self) -> None:
        """Pareto centres each feature on zero."""
        ctx = _make_ln_matrix(seed=22)
        out = apply("scaling", "pareto", ctx)
        per_feature_means = out.mean(axis=1)
        # Should all be ~0 within float tolerance
        assert per_feature_means.abs().max() < 1e-9

    def test_per_feature_sd_equals_sqrt_original_sd(self) -> None:
        """The defining property of Pareto: SD_after = sqrt(SD_before)."""
        ctx = _make_ln_matrix(seed=23)
        sd_before = ctx.matrix.std(axis=1, ddof=1)
        out = apply("scaling", "pareto", ctx)
        sd_after = out.std(axis=1, ddof=1)
        # sd_after should equal sqrt(sd_before)
        np.testing.assert_allclose(
            sd_after.to_numpy(),
            np.sqrt(sd_before.to_numpy()),
            rtol=1e-6,
        )

    def test_constant_feature_does_not_explode(self) -> None:
        """A feature with SD=0 should be centered to 0, not divided by 0."""
        ctx = _make_ln_matrix(seed=24)
        ctx.matrix.iloc[0, :] = 100.0  # all same
        out = apply("scaling", "pareto", ctx)
        assert _no_nan(out)
        assert (out.iloc[0, :] == 0.0).all()

    def test_nan_input_raises(self) -> None:
        ctx = _make_ln_matrix(seed=25)
        ctx.matrix.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN-free"):
            apply("scaling", "pareto", ctx)


# ============================================================
# Transformation diagnostics
# ============================================================
class TestTransformationDiagnostics:
    def test_heteroscedasticity_slope_log_normal_pre_log_is_high(self) -> None:
        """Raw log-normal data has slope ~1; post-log it drops near 0."""
        ctx = _make_ln_matrix(n_features=200, seed=30)
        slope_pre = heteroscedasticity_slope(ctx.matrix)
        # Slope on log-normal is typically near 1
        assert 0.5 < abs(slope_pre) < 1.5

        out = apply("transformation", "log2", ctx)
        slope_post = heteroscedasticity_slope(out)
        # Could even be slightly negative on log values
        assert abs(slope_post) < abs(slope_pre)

    def test_mean_abs_skewness_decreases_with_log2(self) -> None:
        ctx = _make_ln_matrix(n_features=100, seed=31)
        skew_pre = mean_abs_skewness(ctx.matrix)
        out = apply("transformation", "log2", ctx)
        skew_post = mean_abs_skewness(out)
        assert skew_post < skew_pre

    def test_ks_normality_fraction_does_not_decrease_with_log2(self) -> None:
        """Post-log normality fraction should be ≥ pre-log (within tolerance).

        With the small N (18 samples) the KS test has weak power, so the
        absolute fraction can saturate near 1 even for log-normal data. We
        only require that log2 doesn't make things worse by a meaningful
        margin (>5% drop would be a red flag).
        """
        ctx = _make_ln_matrix(n_features=100, seed=32)
        frac_pre = ks_normality_fraction(ctx.matrix)
        out = apply("transformation", "log2", ctx)
        frac_post = ks_normality_fraction(out)
        # Allow tiny tolerance for the saturated-near-1 case
        assert frac_post >= frac_pre - 0.05

    def test_compute_transformation_diagnostics_returns_dataclass(self) -> None:
        ctx = _make_ln_matrix(seed=33)
        d = compute_transformation_diagnostics(ctx.matrix)
        assert isinstance(d, TransformationDiagnostics)
        assert np.isfinite(d.heteroscedasticity_slope)
        assert np.isfinite(d.mean_abs_skewness)
        assert 0 <= d.ks_normality_fraction <= 1
