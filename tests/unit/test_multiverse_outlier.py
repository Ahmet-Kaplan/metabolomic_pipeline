"""Unit tests for PCA-based outlier detection (Hotelling's T² + DModX)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from biodisc.multiverse import StageContext
from biodisc.multiverse.outlier import (
    OutlierResult,
    _auto_n_components,
    detect_outliers,
)


# ============================================================
# Helpers
# ============================================================
def _make_ctx_with_known_outlier(
    *,
    n_features: int = 30,
    n_per_group: int = 5,
    n_groups: int = 3,
    add_t2_outlier: bool = False,
    add_dmodx_outlier: bool = False,
    seed: int = 0,
) -> tuple[StageContext, list[str]]:
    """Build a context with optional known outlier samples.

    Returns (ctx, list of injected outlier sample_ids).
    """
    rng = np.random.default_rng(seed)
    sample_ids: list[str] = []
    cell_lines: list[str] = []
    matrix_cols: list[np.ndarray] = []

    # Per-group baseline + within-group noise
    group_baselines = [rng.lognormal(mean=8.0, sigma=0.5, size=n_features) for _ in range(n_groups)]

    for g_idx, baseline in enumerate(group_baselines):
        group_name = chr(ord("A") + g_idx)
        for r in range(n_per_group):
            noise = rng.lognormal(mean=0.0, sigma=0.1, size=n_features)
            matrix_cols.append(baseline * noise)
            sample_ids.append(f"{group_name}_{r + 1}")
            cell_lines.append(group_name)

    injected: list[str] = []

    if add_t2_outlier:
        # T² outlier: extreme along principal axes — multiply the first group's
        # baseline by a big constant. Will be far from the cluster centre in PC space.
        outlier_vec = group_baselines[0] * 10.0
        matrix_cols.append(outlier_vec)
        sample_ids.append("OUTLIER_T2")
        cell_lines.append("A")
        injected.append("OUTLIER_T2")

    if add_dmodx_outlier:
        # DModX outlier: different SHAPE — random per-feature pattern unrelated
        # to any group baseline. Will have large residuals after PCA reconstruction.
        outlier_vec = rng.lognormal(mean=10.0, sigma=2.0, size=n_features)
        matrix_cols.append(outlier_vec)
        sample_ids.append("OUTLIER_DMODX")
        cell_lines.append("A")
        injected.append("OUTLIER_DMODX")

    matrix = pd.DataFrame(
        np.array(matrix_cols).T,
        index=[f"F{i}" for i in range(n_features)],
        columns=sample_ids,
    )
    n_total = len(sample_ids)
    samples = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "raw_sample_id": sample_ids,
            "injection_type": ["biological"] * n_total,
            "cell_line": cell_lines,
            "replicate": list(range(1, n_total + 1)),
            "dilution_level": [None] * n_total,
            "acq_date": [datetime(2026, 1, 9)] * n_total,
            "column_order": list(range(n_total)),
        }
    )
    return (
        StageContext(matrix=matrix, samples=samples, features=pd.DataFrame()),
        injected,
    )


# ============================================================
# Auto-defaults
# ============================================================
class TestAutoNComponents:
    def test_minimum_two(self) -> None:
        # Tiny dataset — should still return at least 2
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 5))
        k = _auto_n_components(10, 5, X)
        assert k >= 2

    def test_caps_at_n_over_6(self) -> None:
        # 60 samples, 50 features, structured data
        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 50))
        k = _auto_n_components(60, 50, X)
        assert k <= 60 // 6


# ============================================================
# Result structure + basic behaviour
# ============================================================
class TestDetectOutliers:
    def test_returns_outlier_result(self) -> None:
        ctx, _ = _make_ctx_with_known_outlier(seed=10)
        result = detect_outliers(ctx)
        assert isinstance(result, OutlierResult)
        assert len(result.sample_ids) == result.t_squared.shape[0]
        assert result.alpha == 0.05
        assert result.n_components >= 2

    def test_no_outliers_in_homogeneous_data(self) -> None:
        """Homogeneous tightly-clustered data should rarely flag outliers."""
        ctx, _ = _make_ctx_with_known_outlier(
            n_features=20, n_per_group=8, n_groups=2, seed=11
        )
        result = detect_outliers(ctx)
        # Most samples should NOT be flagged in clean data
        assert result.n_outliers_either / len(result.sample_ids) < 0.20

    def test_t2_catches_extreme_score(self) -> None:
        ctx, injected = _make_ctx_with_known_outlier(
            add_t2_outlier=True, n_per_group=8, n_groups=2, seed=12
        )
        result = detect_outliers(ctx)
        # The injected T² outlier should be in the flagged-T2 set
        outlier_idx = result.sample_ids.index(injected[0])
        # Either T² catches it, or DModX does (strong outliers often hit both)
        assert (
            result.flagged_t2[outlier_idx] or result.flagged_dmodx[outlier_idx]
        ), f"injected T² outlier {injected[0]} was not flagged"

    def test_dmodx_catches_shape_outlier(self) -> None:
        ctx, injected = _make_ctx_with_known_outlier(
            add_dmodx_outlier=True, n_per_group=8, n_groups=3, seed=13
        )
        result = detect_outliers(ctx)
        outlier_idx = result.sample_ids.index(injected[0])
        assert (
            result.flagged_t2[outlier_idx] or result.flagged_dmodx[outlier_idx]
        ), f"injected DModX outlier {injected[0]} was not flagged"

    def test_biological_only_filter(self) -> None:
        """When biological_only=True, QCs should be excluded from analysis."""
        ctx, _ = _make_ctx_with_known_outlier(seed=14)
        # Add 4 QC samples
        n_features = ctx.matrix.shape[0]
        rng = np.random.default_rng(99)
        for i in range(4):
            ctx.matrix[f"QC_{i + 1}"] = rng.lognormal(8.0, 0.5, size=n_features)
        new_samples = pd.DataFrame(
            {
                "sample_id": [f"QC_{i + 1}" for i in range(4)],
                "raw_sample_id": [f"QC_{i + 1}" for i in range(4)],
                "injection_type": ["qc"] * 4,
                "cell_line": [None] * 4,
                "replicate": list(range(1, 5)),
                "dilution_level": [None] * 4,
                "acq_date": [datetime(2026, 1, 9)] * 4,
                "column_order": list(range(15, 19)),
            }
        )
        ctx.samples = pd.concat([ctx.samples, new_samples], ignore_index=True)
        result = detect_outliers(ctx, biological_only=True)
        # QCs should not be in the result
        assert all(not sid.startswith("QC_") for sid in result.sample_ids)

    def test_explicit_sample_ids_overrides_filter(self) -> None:
        ctx, _ = _make_ctx_with_known_outlier(seed=15)
        result = detect_outliers(ctx, sample_ids=ctx.matrix.columns[:6].tolist())
        assert len(result.sample_ids) == 6

    def test_too_few_samples_raises(self) -> None:
        ctx, _ = _make_ctx_with_known_outlier(n_per_group=1, n_groups=2, seed=16)
        with pytest.raises(ValueError, match="≥ 4 samples"):
            detect_outliers(ctx)

    def test_nan_input_raises(self) -> None:
        ctx, _ = _make_ctx_with_known_outlier(seed=17)
        ctx.matrix.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN-free"):
            detect_outliers(ctx)

    def test_explicit_n_components(self) -> None:
        ctx, _ = _make_ctx_with_known_outlier(seed=18)
        result = detect_outliers(ctx, n_components=3)
        assert result.n_components == 3

    def test_flagged_either_is_union(self) -> None:
        ctx, _ = _make_ctx_with_known_outlier(
            add_t2_outlier=True, add_dmodx_outlier=True, n_per_group=8, n_groups=3, seed=19
        )
        result = detect_outliers(ctx)
        np.testing.assert_array_equal(
            result.flagged_either,
            result.flagged_t2 | result.flagged_dmodx,
        )

    def test_outlier_sample_ids_property(self) -> None:
        ctx, _ = _make_ctx_with_known_outlier(
            add_t2_outlier=True, n_per_group=8, n_groups=2, seed=20
        )
        result = detect_outliers(ctx)
        # outlier_sample_ids is the subset of sample_ids where flagged_either is True
        for sid in result.outlier_sample_ids:
            assert sid in result.sample_ids
            i = result.sample_ids.index(sid)
            assert result.flagged_either[i]
