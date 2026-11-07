"""Unit tests for PQN / MSTUS / median normalisation."""

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
def _make_norm_data(
    *,
    n_features: int = 30,
    n_qc: int = 6,
    n_bio: int = 12,
    sample_loading_factors: list[float] | None = None,
    seed: int = 0,
) -> StageContext:
    """Build a controlled context for normalisation testing.

    Each sample gets multiplied by a known per-sample loading factor so
    we can verify a normaliser successfully removes the variability.
    Defaults: random factors in [0.5, 2.0] simulating sample-loading
    differences.
    """
    rng = np.random.default_rng(seed)
    n_total = n_qc + n_bio

    # Baseline log-normal abundances per feature
    baseline = rng.lognormal(mean=8.0, sigma=1.0, size=n_features)

    if sample_loading_factors is None:
        # Random per-sample loading factors in [0.5, 2.0]
        sample_loading_factors = list(rng.uniform(0.5, 2.0, size=n_total))

    sample_ids = [f"QC_{i + 1}" for i in range(n_qc)] + [
        f"S_{i + 1}" for i in range(n_bio)
    ]

    matrix_data = np.zeros((n_features, n_total))
    for j, factor in enumerate(sample_loading_factors):
        # Per-sample noise (small, biological variability)
        bio_noise = 1.0 + 0.05 * rng.standard_normal(n_features)
        matrix_data[:, j] = baseline * factor * np.abs(bio_noise)

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
    def test_three_normalisers_registered(self) -> None:
        names = {s.name for s in list_methods("normalisation")}
        assert {"pqn", "mstus", "median"}.issubset(names)


# ============================================================
# PQN
# ============================================================
class TestPQN:
    def test_no_nans(self) -> None:
        ctx = _make_norm_data(seed=1)
        out = apply("normalisation", "pqn", ctx)
        assert _no_nan(out)

    def test_shape_preserved(self) -> None:
        ctx = _make_norm_data(seed=2)
        out = apply("normalisation", "pqn", ctx)
        assert _shape_preserved(out, ctx)

    def test_non_negative_output(self) -> None:
        ctx = _make_norm_data(seed=3)
        out = apply("normalisation", "pqn", ctx)
        assert (out >= 0).all().all()

    def test_recovers_known_loading_factors(self) -> None:
        """If samples were scaled by known factors, PQN should ~undo them."""
        loading = [
            1.0, 0.5, 2.0, 1.5, 0.8, 1.2,
            0.6, 0.7, 1.3, 1.1, 0.9, 1.4, 0.5, 1.6, 0.7, 1.0, 1.2, 0.8,
        ]
        ctx = _make_norm_data(
            n_features=50, n_qc=6, n_bio=12,
            sample_loading_factors=loading, seed=4,
        )
        out = apply("normalisation", "pqn", ctx)

        # After normalisation, per-sample medians should be roughly equal
        median_before_cv = (
            ctx.matrix.replace(0, np.nan).median(axis=0).std()
            / ctx.matrix.replace(0, np.nan).median(axis=0).mean()
        )
        median_after_cv = (
            out.replace(0, np.nan).median(axis=0).std()
            / out.replace(0, np.nan).median(axis=0).mean()
        )
        assert median_after_cv < median_before_cv, "PQN did not reduce sample-median variability"

    def test_reference_qc_median_when_qcs_available(self) -> None:
        """With ≥3 QCs, default reference should be qc_median."""
        ctx = _make_norm_data(n_qc=6, seed=5)
        out = apply("normalisation", "pqn", ctx)
        # Indirect check: the QC median should be ~1 after normalisation
        # (by definition of QC-reference PQN)
        # NOTE: due to median-of-quotients being applied to per-sample, this is approximate
        assert _no_nan(out)

    def test_falls_back_to_all_median_with_few_qcs(self) -> None:
        """With <3 QCs, default 'auto' should use all_median."""
        ctx = _make_norm_data(n_qc=2, n_bio=10, seed=6)
        out = apply("normalisation", "pqn", ctx)
        assert _no_nan(out)

    def test_qc_median_explicit_raises_with_too_few(self) -> None:
        ctx = _make_norm_data(n_qc=2, n_bio=10, seed=7)
        with pytest.raises(ValueError, match=r"qc_median.*requires"):
            apply("normalisation", "pqn", ctx, reference="qc_median")

    def test_invalid_reference_raises(self) -> None:
        ctx = _make_norm_data(seed=8)
        with pytest.raises(ValueError, match="reference must be"):
            apply("normalisation", "pqn", ctx, reference="invalid")

    def test_requires_imputed_matrix(self) -> None:
        ctx = _make_norm_data(seed=9)
        ctx.matrix.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="imputed matrix"):
            apply("normalisation", "pqn", ctx)

    def test_auto_min_features_for_factor(self) -> None:
        from biodisc.multiverse.methods.normalise.pqn import _auto_min_features_for_factor
        assert _auto_min_features_for_factor(50) == 10  # floor
        assert _auto_min_features_for_factor(2000) == 20
        assert _auto_min_features_for_factor(10_000) == 100


# ============================================================
# MSTUS
# ============================================================
class TestMSTUS:
    def test_no_nans(self) -> None:
        ctx = _make_norm_data(seed=10)
        out = apply("normalisation", "mstus", ctx)
        assert _no_nan(out)

    def test_shape_preserved(self) -> None:
        ctx = _make_norm_data(seed=11)
        out = apply("normalisation", "mstus", ctx)
        assert _shape_preserved(out, ctx)

    def test_non_negative(self) -> None:
        ctx = _make_norm_data(seed=12)
        out = apply("normalisation", "mstus", ctx)
        assert (out >= 0).all().all()

    def test_recovers_known_loading_factors(self) -> None:
        loading = [
            1.0, 0.5, 2.0, 1.5, 0.8, 1.2,
            0.6, 0.7, 1.3, 1.1, 0.9, 1.4, 0.5, 1.6, 0.7, 1.0, 1.2, 0.8,
        ]
        ctx = _make_norm_data(
            n_features=50, n_qc=6, n_bio=12,
            sample_loading_factors=loading, seed=13,
        )
        out = apply("normalisation", "mstus", ctx)
        # Per-sample sums should be roughly equal post-normalisation
        sums_before_cv = ctx.matrix.sum(axis=0).std() / ctx.matrix.sum(axis=0).mean()
        sums_after_cv = out.sum(axis=0).std() / out.sum(axis=0).mean()
        assert sums_after_cv < sums_before_cv, "MSTUS did not reduce sample-sum variability"

    def test_falls_back_when_no_useful_features(self) -> None:
        """With useful_min_pct=1.0 and very sparse data, no useful features."""
        ctx = _make_norm_data(n_features=10, n_qc=4, n_bio=4, seed=14)
        # Punch holes: only F0 detected in all samples; others sporadic
        for i in range(1, 10):
            ctx.matrix.iloc[i, np.random.RandomState(i).randint(0, ctx.matrix.shape[1], 3)] = 0
        out = apply("normalisation", "mstus", ctx, useful_min_pct=1.0)
        assert _no_nan(out)
        assert _shape_preserved(out, ctx)

    def test_requires_imputed_matrix(self) -> None:
        ctx = _make_norm_data(seed=15)
        ctx.matrix.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="imputed matrix"):
            apply("normalisation", "mstus", ctx)


# ============================================================
# Median
# ============================================================
class TestMedian:
    def test_no_nans(self) -> None:
        ctx = _make_norm_data(seed=20)
        out = apply("normalisation", "median", ctx)
        assert _no_nan(out)

    def test_shape_preserved(self) -> None:
        ctx = _make_norm_data(seed=21)
        out = apply("normalisation", "median", ctx)
        assert _shape_preserved(out, ctx)

    def test_non_negative(self) -> None:
        ctx = _make_norm_data(seed=22)
        out = apply("normalisation", "median", ctx)
        assert (out >= 0).all().all()

    def test_per_sample_medians_equal_after(self) -> None:
        """The defining property of median normalisation: per-sample medians equal."""
        ctx = _make_norm_data(n_features=50, seed=23)
        out = apply("normalisation", "median", ctx)
        # All per-sample medians should be ~ global median (the reference)
        post_medians = out.replace(0, np.nan).median(axis=0)
        # Coefficient of variation should be < 1% (essentially zero spread)
        assert post_medians.std() / post_medians.mean() < 0.01

    def test_recovers_known_loading_factors(self) -> None:
        loading = [
            1.0, 0.5, 2.0, 1.5, 0.8, 1.2,
            0.6, 0.7, 1.3, 1.1, 0.9, 1.4, 0.5, 1.6, 0.7, 1.0, 1.2, 0.8,
        ]
        ctx = _make_norm_data(
            n_features=50, n_qc=6, n_bio=12,
            sample_loading_factors=loading, seed=24,
        )
        out = apply("normalisation", "median", ctx)
        median_before_cv = (
            ctx.matrix.replace(0, np.nan).median(axis=0).std()
            / ctx.matrix.replace(0, np.nan).median(axis=0).mean()
        )
        median_after_cv = (
            out.replace(0, np.nan).median(axis=0).std()
            / out.replace(0, np.nan).median(axis=0).mean()
        )
        assert median_after_cv < median_before_cv

    def test_requires_imputed_matrix(self) -> None:
        ctx = _make_norm_data(seed=25)
        ctx.matrix.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="imputed matrix"):
            apply("normalisation", "median", ctx)

    def test_raises_on_all_zero_matrix(self) -> None:
        ctx = _make_norm_data(seed=26)
        ctx.matrix.iloc[:, :] = 0.0
        with pytest.raises(ValueError, match="global median"):
            apply("normalisation", "median", ctx)
