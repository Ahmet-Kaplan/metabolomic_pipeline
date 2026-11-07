"""Unit tests for QC-RSC + SERRF drift correctors + StageContext helpers."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from biodisc.multiverse import StageContext
from biodisc.multiverse.registry import apply, list_methods


# ============================================================
# Helpers
# ============================================================
def _make_drift_data(
    *,
    n_features: int = 30,
    n_samples: int = 20,
    n_qc: int = 6,
    drift_slope_pct: float = 50.0,
    biological_var: float = 0.1,
    seed: int = 0,
) -> StageContext:
    """Build a StageContext with QC + biological samples + simulated drift.

    Drift model: each feature's intensity multiplied by
    ``1 + drift_slope_pct/100 * (position / n_total)``. So at position N
    each feature is `(1 + drift_slope_pct/100)` × its baseline. QCs sample
    this drift cleanly; biological samples have additional small Gaussian
    biological variation on top.
    """
    rng = np.random.default_rng(seed)
    n_total = n_qc + n_samples
    # Baseline log-normal abundances per feature
    baseline = rng.lognormal(mean=8.0, sigma=1.0, size=n_features)

    # Build per-position multipliers (drift)
    positions = np.arange(1, n_total + 1)
    drift_factor = 1.0 + (drift_slope_pct / 100.0) * (positions - 1) / max(n_total - 1, 1)

    # Generate matrix: features × samples
    sample_ids = [f"QC_{i + 1}" for i in range(n_qc)] + [
        f"S_{i + 1}" for i in range(n_samples)
    ]
    matrix_data = np.zeros((n_features, n_total))
    # QCs at positions 1..n_qc; samples at positions n_qc+1..n_total
    # (just for simplicity; real QCs are interleaved but the algorithm
    # only cares about position assignments)
    for j, pos in enumerate(positions):
        if j < n_qc:
            # QC sample: just baseline × drift (no biology)
            matrix_data[:, j] = baseline * drift_factor[pos - 1]
        else:
            # Biological sample: baseline × drift × biological noise
            bio_noise = rng.normal(1.0, biological_var, size=n_features)
            matrix_data[:, j] = baseline * drift_factor[pos - 1] * np.abs(bio_noise)

    matrix = pd.DataFrame(
        matrix_data,
        index=[f"F{i}" for i in range(n_features)],
        columns=sample_ids,
    )

    samples = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "raw_sample_id": sample_ids,  # 1:1 for tests
            "injection_type": ["qc"] * n_qc + ["biological"] * n_samples,
            "cell_line": [None] * n_qc + ["X"] * n_samples,
            "replicate": [i + 1 for i in range(n_qc)]
            + [i + 1 for i in range(n_samples)],
            "dilution_level": [None] * n_total,
            "acq_date": [datetime(2026, 1, 9)] * n_total,
            "column_order": list(range(n_total)),
        }
    )
    run_order = pd.DataFrame(
        {
            "position": positions,
            "raw_sample_id": sample_ids,
            "acq_datetime": [datetime(2026, 1, 9, h % 24) for h in range(n_total)],
            "header_file": [f"_HEADER({i}).TXT" for i in range(n_total)],
        }
    )
    return StageContext(
        matrix=matrix,
        samples=samples,
        features=pd.DataFrame(),
        run_order=run_order,
    )


# ============================================================
# StageContext helpers
# ============================================================
class TestStageContextHelpers:
    def test_sample_positions_returns_aligned_series(self) -> None:
        ctx = _make_drift_data()
        positions = ctx.sample_positions()
        assert positions.index.name == "sample_id"
        assert set(positions.index) == set(ctx.matrix.columns)

    def test_sample_positions_raises_without_run_order(self) -> None:
        ctx = _make_drift_data()
        ctx.run_order = None  # mutate (dataclass not frozen)
        with pytest.raises(ValueError, match="run_order is None"):
            ctx.sample_positions()

    def test_qc_sample_ids_filters_to_matrix_cols(self) -> None:
        ctx = _make_drift_data()
        qc_ids = ctx.qc_sample_ids()
        assert all(s.startswith("QC_") for s in qc_ids)
        assert len(qc_ids) == 6

    def test_sample_positions_raises_on_missing_column(self) -> None:
        ctx = _make_drift_data()
        # Drop one sample from run_order so a matrix column is unmapped
        ctx.run_order = ctx.run_order.iloc[:-1]
        with pytest.raises(ValueError, match="lack a run_order position"):
            ctx.sample_positions()


# ============================================================
# Registration
# ============================================================
class TestRegistration:
    def test_qc_rsc_registered(self) -> None:
        names = {s.name for s in list_methods("drift")}
        assert "qc_rsc" in names
        assert "serrf" in names


# ============================================================
# QC-RSC behaviour
# ============================================================
class TestQCRSC:
    def test_no_nans_in_output(self) -> None:
        ctx = _make_drift_data(seed=1)
        out = apply("drift", "qc_rsc", ctx)
        assert not out.isna().any().any()

    def test_shape_preserved(self) -> None:
        ctx = _make_drift_data(seed=2)
        out = apply("drift", "qc_rsc", ctx)
        assert out.shape == ctx.matrix.shape
        assert list(out.index) == list(ctx.matrix.index)
        assert list(out.columns) == list(ctx.matrix.columns)

    def test_output_non_negative(self) -> None:
        ctx = _make_drift_data(seed=3)
        out = apply("drift", "qc_rsc", ctx)
        assert (out >= 0).all().all()

    def test_correction_reduces_qc_variance(self) -> None:
        """The hallmark test: QC variance after correction < before."""
        ctx = _make_drift_data(drift_slope_pct=80.0, seed=4)  # strong drift
        qc_ids = ctx.qc_sample_ids()

        # Variance per feature across QCs, before
        var_before = ctx.matrix[qc_ids].var(axis=1)
        out = apply("drift", "qc_rsc", ctx)
        var_after = out[qc_ids].var(axis=1)

        # On most features, variance should drop substantially
        improved = (var_after < var_before).sum() / len(var_before)
        # At least 80% of features should show variance reduction
        assert improved > 0.8, f"only {improved:.0%} of features improved"

    def test_zero_variance_qc_passes_through(self) -> None:
        """A feature with constant QC values shouldn't be touched."""
        ctx = _make_drift_data(seed=5)
        qc_ids = ctx.qc_sample_ids()
        # Force F0's QCs to be all the same value
        ctx.matrix.loc["F0", qc_ids] = 100.0
        out = apply("drift", "qc_rsc", ctx)
        # Sample-side values should be unchanged for F0
        assert (out.loc["F0", qc_ids] == 100.0).all()

    def test_requires_imputed_matrix(self) -> None:
        ctx = _make_drift_data(seed=6)
        ctx.matrix.iloc[0, 0] = np.nan  # introduce a NaN
        with pytest.raises(ValueError, match="imputed matrix"):
            apply("drift", "qc_rsc", ctx)

    def test_requires_min_qc_count(self) -> None:
        ctx = _make_drift_data(n_qc=3, seed=7)
        with pytest.raises(ValueError, match="needs ≥ 4 QC injections"):
            apply("drift", "qc_rsc", ctx)

    def test_auto_smoothing_factor_scales(self) -> None:
        from biodisc.multiverse.methods.drift.qc_rsc import _auto_smoothing_factor
        # n_qc=6 → 1.0 (max smoothing for sparse QCs)
        assert _auto_smoothing_factor(6) == pytest.approx(1.0)
        # n_qc=12 → 0.5
        assert _auto_smoothing_factor(12) == pytest.approx(0.5)
        # n_qc=30 → 0.3 (clamped)
        assert _auto_smoothing_factor(30) == pytest.approx(0.3)

    def test_correction_cap_prevents_runaway_inflation(self) -> None:
        """REGRESSION TEST for Apr-2026 bug: spline near-zero prediction
        was inflating samples by 10⁶–10⁸×. Cap should clip drift estimate
        to [ref/cap, ref*cap], so corrected values never exceed
        original × cap."""
        # Build pathological data: feature with QCs that the spline will
        # overshoot near zero between QC anchors
        ctx = _make_drift_data(n_qc=8, n_samples=12, drift_slope_pct=10.0, seed=99)

        # Force one feature's QC values to be a sawtooth (alternating high/low)
        # so the spline has steep oscillations with potential undershoot
        qc_ids = ctx.qc_sample_ids()
        sawtooth = [1000.0, 100.0, 1000.0, 100.0, 1000.0, 100.0, 1000.0, 100.0]
        for sid, val in zip(qc_ids, sawtooth, strict=True):
            ctx.matrix.loc["F0", sid] = val

        original_max = ctx.matrix.loc["F0"].max()
        out = apply("drift", "qc_rsc", ctx, correction_cap=10.0)
        corrected_max = out.loc["F0"].max()

        # No value should exceed original_max × cap (with small tolerance)
        assert corrected_max <= original_max * 10.5, (
            f"correction_cap failed: max went from {original_max} to {corrected_max}"
        )


# ============================================================
# SERRF behaviour
# ============================================================
class TestSERRF:
    def test_no_nans_in_output(self) -> None:
        ctx = _make_drift_data(n_qc=8, seed=10)
        out = apply("drift", "serrf", ctx, n_estimators=10, n_correlated_features=5)
        assert not out.isna().any().any()

    def test_shape_preserved(self) -> None:
        ctx = _make_drift_data(n_qc=8, seed=11)
        out = apply("drift", "serrf", ctx, n_estimators=10, n_correlated_features=5)
        assert out.shape == ctx.matrix.shape

    def test_output_non_negative(self) -> None:
        ctx = _make_drift_data(n_qc=8, seed=12)
        out = apply("drift", "serrf", ctx, n_estimators=10, n_correlated_features=5)
        assert (out >= 0).all().all()

    def test_correction_reduces_qc_variance(self) -> None:
        """SERRF with correlated-feature predictors should reduce QC variance."""
        ctx = _make_drift_data(n_qc=10, drift_slope_pct=80.0, seed=13)
        qc_ids = ctx.qc_sample_ids()

        var_before = ctx.matrix[qc_ids].var(axis=1)
        out = apply(
            "drift", "serrf", ctx, n_estimators=20, n_correlated_features=5, seed=42
        )
        var_after = out[qc_ids].var(axis=1)

        # SERRF is designed to drive QC variance down — most features should improve
        # Note: with synthetic baseline-only QCs (no drift dependency on co-features),
        # SERRF may be slightly less aggressive than QC-RSC; relax the threshold.
        improved = (var_after < var_before).sum() / len(var_before)
        assert improved > 0.5, f"only {improved:.0%} of features improved"

    def test_zero_variance_qc_passes_through(self) -> None:
        ctx = _make_drift_data(n_qc=8, seed=14)
        qc_ids = ctx.qc_sample_ids()
        ctx.matrix.loc["F0", qc_ids] = 100.0
        out = apply(
            "drift", "serrf", ctx, n_estimators=10, n_correlated_features=5
        )
        assert (out.loc["F0", qc_ids] == 100.0).all()

    def test_requires_imputed_matrix(self) -> None:
        ctx = _make_drift_data(n_qc=8, seed=15)
        ctx.matrix.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="imputed matrix"):
            apply("drift", "serrf", ctx)

    def test_requires_min_qc_count(self) -> None:
        ctx = _make_drift_data(n_qc=3, seed=16)
        with pytest.raises(ValueError, match="needs ≥ 5 QC injections"):
            apply("drift", "serrf", ctx)

    def test_deterministic_with_same_seed(self) -> None:
        ctx1 = _make_drift_data(n_qc=8, seed=20)
        ctx2 = _make_drift_data(n_qc=8, seed=20)
        out1 = apply(
            "drift", "serrf", ctx1, n_estimators=10, n_correlated_features=5, seed=42
        )
        out2 = apply(
            "drift", "serrf", ctx2, n_estimators=10, n_correlated_features=5, seed=42
        )
        np.testing.assert_array_almost_equal(out1.values, out2.values)

    def test_auto_n_correlated_scales(self) -> None:
        from biodisc.multiverse.methods.drift.serrf import _auto_n_correlated
        assert _auto_n_correlated(100) == 5  # floor
        assert _auto_n_correlated(2000) == 10
        assert _auto_n_correlated(10_000) == 30  # ceiling

    def test_auto_n_estimators_scales(self) -> None:
        from biodisc.multiverse.methods.drift.serrf import _auto_n_estimators
        # n_qc=4 → max(20, 5*2 + 20) = max(20, 30) = 30
        assert _auto_n_estimators(4) == 30
        # n_qc=16 → 5*4 + 20 = 40
        assert _auto_n_estimators(16) == 40
        # n_qc=400 → 5*20 + 20 = 120 → clamped to 100
        assert _auto_n_estimators(400) == 100

    def test_auto_max_depth_scales(self) -> None:
        from biodisc.multiverse.methods.drift.serrf import _auto_max_depth
        assert _auto_max_depth(2) == 2  # floor
        assert _auto_max_depth(8) == 3
        assert _auto_max_depth(16) == 4
        assert _auto_max_depth(64) == 6

    def test_auto_min_samples_leaf_scales(self) -> None:
        from biodisc.multiverse.methods.drift.serrf import _auto_min_samples_leaf
        assert _auto_min_samples_leaf(2) == 1  # floor
        assert _auto_min_samples_leaf(16) == 2
        assert _auto_min_samples_leaf(64) == 8

    def test_correlation_pool_all_vs_qc_both_run(self) -> None:
        ctx = _make_drift_data(n_qc=8, seed=21)
        out_all = apply(
            "drift", "serrf", ctx, n_estimators=10, n_correlated_features=5,
            correlation_pool="all", max_depth=3,
        )
        out_qc = apply(
            "drift", "serrf", ctx, n_estimators=10, n_correlated_features=5,
            correlation_pool="qc", max_depth=3,
        )
        # Both should produce valid output; values may differ
        assert out_all.shape == out_qc.shape
        assert (out_all >= 0).all().all()
        assert (out_qc >= 0).all().all()

    def test_invalid_correlation_pool_raises(self) -> None:
        ctx = _make_drift_data(n_qc=8, seed=22)
        import pytest
        with pytest.raises(ValueError, match="correlation_pool must be"):
            apply("drift", "serrf", ctx, correlation_pool="invalid")


# ============================================================
# LOQO predictors (added in hardening pass)
# ============================================================
class TestQCRSCLoqoPredict:
    def test_returns_per_feature_array(self) -> None:
        from biodisc.multiverse.methods.drift.qc_rsc import loqo_predict
        ctx = _make_drift_data(n_qc=8, seed=30)
        qc_ids = ctx.qc_sample_ids()
        pred = loqo_predict(ctx, qc_ids[3])
        assert pred.shape == (ctx.matrix.shape[0],)
        assert (pred >= 0).all()
        assert np.isfinite(pred).all()

    def test_loqo_residual_via_diagnostics(self) -> None:
        """End-to-end: diagnostics module + qc_rsc.loqo_predict."""
        from biodisc.multiverse.diagnostics import loqo_residuals_from_predictor
        from biodisc.multiverse.methods.drift.qc_rsc import loqo_predict
        ctx = _make_drift_data(n_qc=8, drift_slope_pct=20.0, seed=31)
        result = loqo_residuals_from_predictor(ctx, predict_held_qc=loqo_predict)
        # With 20% drift over 18 samples, LOQO should be modest but finite
        assert np.isfinite(result.median_residual_pct)
        assert result.n_qc == 8


class TestSERRFLoqoPredict:
    def test_returns_per_feature_array(self) -> None:
        from biodisc.multiverse.methods.drift.serrf import loqo_predict
        ctx = _make_drift_data(n_qc=8, seed=40)
        qc_ids = ctx.qc_sample_ids()
        pred = loqo_predict(
            ctx, qc_ids[3], n_estimators=10, n_correlated_features=5, max_depth=3
        )
        assert pred.shape == (ctx.matrix.shape[0],)
        assert (pred >= 0).all()


# ============================================================
# Silhouette guardrail
# ============================================================
class TestSilhouetteGuardrail:
    @staticmethod
    def _multi_group_ctx(seed: int = 0) -> StageContext:
        """Build a context with 2 biological groups (silhouette is defined)."""
        rng = np.random.default_rng(seed)
        n_feat = 15
        n_qc = 4
        # Two distinct biological groups with different baselines
        baseline_a = rng.lognormal(mean=8.0, sigma=0.5, size=n_feat)
        baseline_b = rng.lognormal(mean=11.0, sigma=0.5, size=n_feat)
        baseline_qc = (baseline_a + baseline_b) / 2
        sample_ids = [f"QC_{i + 1}" for i in range(n_qc)] + [
            f"A_{i + 1}" for i in range(4)
        ] + [f"B_{i + 1}" for i in range(4)]
        n_total = len(sample_ids)
        cols: list[np.ndarray] = []
        for sid in sample_ids:
            if sid.startswith("QC_"):
                base = baseline_qc
            elif sid.startswith("A_"):
                base = baseline_a
            else:
                base = baseline_b
            noise = 1.0 + 0.05 * rng.standard_normal(n_feat)
            cols.append(base * noise)
        matrix = pd.DataFrame(
            np.array(cols).T,
            index=[f"F{i}" for i in range(n_feat)],
            columns=sample_ids,
        )
        cell_lines = (
            [None] * n_qc + ["A"] * 4 + ["B"] * 4
        )
        samples = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "raw_sample_id": sample_ids,
                "injection_type": ["qc"] * n_qc + ["biological"] * 8,
                "cell_line": cell_lines,
                "replicate": list(range(1, n_qc + 1)) + [1, 2, 3, 4] * 2,
                "dilution_level": [None] * n_total,
                "acq_date": [datetime(2026, 1, 9)] * n_total,
                "column_order": list(range(n_total)),
            }
        )
        run_order = pd.DataFrame(
            {
                "position": list(range(1, n_total + 1)),
                "raw_sample_id": sample_ids,
                "acq_datetime": [datetime(2026, 1, 9)] * n_total,
                "header_file": [f"_HEADER({i}).TXT" for i in range(n_total)],
            }
        )
        return StageContext(
            matrix=matrix, samples=samples, features=pd.DataFrame(), run_order=run_order
        )

    def test_no_change_no_drop(self) -> None:
        from biodisc.multiverse.diagnostics import silhouette_guardrail
        ctx = self._multi_group_ctx(seed=50)
        result = silhouette_guardrail(ctx.matrix, ctx.matrix.copy(), ctx.samples)
        assert abs(result.relative_drop) < 1e-9
        assert not result.overcorrected

    def test_destroyed_signal_triggers(self) -> None:
        from biodisc.multiverse.diagnostics import silhouette_guardrail
        ctx = self._multi_group_ctx(seed=51)
        rng = np.random.default_rng(0)
        noise = pd.DataFrame(
            rng.uniform(100, 200, size=ctx.matrix.shape),
            index=ctx.matrix.index,
            columns=ctx.matrix.columns,
        )
        result = silhouette_guardrail(ctx.matrix, noise, ctx.samples)
        # Random noise destroys group structure → significant drop
        assert result.silhouette_after < result.silhouette_before
        assert result.overcorrected  # > 5% drop
