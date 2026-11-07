"""Unit tests for the Stage 4 QC-CV filter."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from biodisc.audit import AuditLogger
from biodisc.filters.qc_cv import (
    QcCvFilterConfig,
    _compute_cv_and_stats,
    compute_qc_cv_metrics,
    run,
)


def _make_samples(n_bio: int = 2, n_qc: int = 16) -> pd.DataFrame:
    """Samples table with biological + QC rows."""
    rows = []
    col = 0
    for rep in range(1, n_bio + 1):
        rows.append(
            {
                "sample_id": f"Bio_{rep}",
                "raw_sample_id": f"01012026_Bio_{rep}",
                "injection_type": "biological",
                "cell_line": "MCF7",
                "replicate": rep,
                "dilution_level": None,
                "acq_date": datetime(2026, 1, 1),
                "column_order": col,
            }
        )
        col += 1
    for rep in range(1, n_qc + 1):
        rows.append(
            {
                "sample_id": f"QC_{rep}",
                "raw_sample_id": f"01012026_QC_{rep}",
                "injection_type": "qc",
                "cell_line": None,
                "replicate": rep,
                "dilution_level": None,
                "acq_date": datetime(2026, 1, 1),
                "column_order": col,
            }
        )
        col += 1
    return pd.DataFrame(rows)


def _make_abundances(features: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    rows = []
    for fid, per_sample in features.items():
        for sid, inten in per_sample.items():
            rows.append({"feature_id": fid, "sample_id": sid, "intensity": inten})
    return pd.DataFrame(rows)


# ================================================================
# _compute_cv_and_stats
# ================================================================
class TestComputeCvAndStats:
    def test_constant_values_zero_cv(self) -> None:
        vals = np.array([100.0] * 5)
        mean, sd, cv, n = _compute_cv_and_stats(vals)
        assert mean == pytest.approx(100.0)
        assert sd == pytest.approx(0.0)
        assert cv == pytest.approx(0.0)
        assert n == 5

    def test_known_cv(self) -> None:
        # sd (ddof=1) of [10, 12, 11] = 1.0, mean = 11, CV = 100/11 ≈ 9.09%
        vals = np.array([10.0, 12.0, 11.0])
        _, _, cv, n = _compute_cv_and_stats(vals)
        assert cv == pytest.approx(100 / 11, rel=1e-4)
        assert n == 3

    def test_zero_and_nan_excluded(self) -> None:
        # 0 and NaN must not count as detected
        vals = np.array([10.0, 0.0, np.nan, 11.0, 12.0])
        mean, _, _, n = _compute_cv_and_stats(vals)
        assert n == 3
        assert mean == pytest.approx((10 + 11 + 12) / 3)

    def test_single_detection_returns_nan_cv(self) -> None:
        # Only one valid point — SD undefined (n-1 = 0)
        vals = np.array([10.0, 0.0, 0.0])
        _, _, cv, n = _compute_cv_and_stats(vals)
        assert np.isnan(cv)
        assert n == 1

    def test_all_zero_returns_nan(self) -> None:
        vals = np.array([0.0, 0.0, 0.0])
        mean, sd, cv, n = _compute_cv_and_stats(vals)
        assert n == 0
        assert np.isnan(mean) and np.isnan(sd) and np.isnan(cv)


# ================================================================
# compute_qc_cv_metrics
# ================================================================
class TestComputeQcCvMetrics:
    @pytest.fixture
    def samples(self) -> pd.DataFrame:
        return _make_samples(n_bio=0, n_qc=16)

    def test_stable_feature(self, samples: pd.DataFrame) -> None:
        abund = _make_abundances({"f_stable": {f"QC_{i}": 1000.0 for i in range(1, 17)}})
        m = compute_qc_cv_metrics(abund, samples)
        row = m.iloc[0]
        assert row["qc_n_detected"] == 16
        assert row["qc_mean"] == pytest.approx(1000.0)
        assert row["qc_cv_pct"] == pytest.approx(0.0, abs=1e-10)

    def test_noisy_feature(self, samples: pd.DataFrame) -> None:
        # 16 values: [70,100,100,130] × 4
        # mean = 100; sum of sq deviations = 4 × (30² + 0 + 0 + 30²) = 7200
        # sd (ddof=1) = sqrt(7200/15) ≈ 21.91 → CV ≈ 21.91 %
        vals = [70.0, 100.0, 100.0, 130.0] * 4
        abund = _make_abundances({"f_noisy": {f"QC_{i + 1}": vals[i] for i in range(16)}})
        m = compute_qc_cv_metrics(abund, samples)
        row = m.iloc[0]
        import math

        expected_cv = 100.0 * math.sqrt(7200.0 / 15) / 100.0
        assert row["qc_cv_pct"] == pytest.approx(expected_cv, rel=1e-6)

    def test_filter_to_kept_features(self, samples: pd.DataFrame) -> None:
        abund = _make_abundances(
            {
                "keep_me": {f"QC_{i}": 100.0 for i in range(1, 17)},
                "ignore_me": {f"QC_{i}": 100.0 for i in range(1, 17)},
            }
        )
        m = compute_qc_cv_metrics(abund, samples, kept_features={"keep_me"})
        assert len(m) == 1
        assert m.iloc[0]["feature_id"] == "keep_me"

    def test_no_qc_raises(self) -> None:
        s = _make_samples(n_bio=2, n_qc=0)
        a = _make_abundances({"f": {"Bio_1": 100.0, "Bio_2": 110.0}})
        with pytest.raises(ValueError, match="no QC samples"):
            compute_qc_cv_metrics(a, s)


# ================================================================
# run
# ================================================================
class TestRun:
    @pytest.fixture
    def samples(self) -> pd.DataFrame:
        return _make_samples(n_bio=0, n_qc=16)

    def test_three_canonical_features_at_30pct_threshold(self, samples: pd.DataFrame) -> None:
        """Stable (kept), just-under-threshold (kept), way-above (dropped), rare (dropped)."""
        # f_stable: CV = 0%
        stable = {f"QC_{i + 1}": 100.0 for i in range(16)}
        # f_sub30: [75,100,100,125]×4 → sd=sqrt(5000/15)=18.26 → CV ≈ 18.3% (kept at 30%)
        sub30_vals = [75.0, 100.0, 100.0, 125.0] * 4
        sub30 = {f"QC_{i + 1}": sub30_vals[i] for i in range(16)}
        # f_noisy: CV way above 30% — use [20,100,50,180]×4 for a very spiky pattern
        noisy_vals = [20.0, 100.0, 50.0, 180.0] * 4
        noisy = {f"QC_{i + 1}": noisy_vals[i] for i in range(16)}
        # f_rare: only 1 detection → dropped on detection count
        rare = {f"QC_{i + 1}": (100.0 if i == 0 else 0.0) for i in range(16)}

        abund = _make_abundances({"stable": stable, "sub30": sub30, "noisy": noisy, "rare": rare})
        audit = AuditLogger(stage="test_04", params={})
        res = run(
            abundances_long=abund,
            samples=samples,
            kept_features={"stable", "sub30", "noisy", "rare"},
            config=QcCvFilterConfig(max_cv_pct=30.0, min_qc_detections=3),
            audit=audit,
        )
        kept = set(res.kept_features["feature_id"])
        assert "stable" in kept
        assert "sub30" in kept
        assert "noisy" not in kept
        assert "rare" not in kept
        assert res.n_dropped_cv == 1  # noisy
        assert res.n_dropped_detections == 1  # rare

    def test_threshold_configurability(self, samples: pd.DataFrame) -> None:
        """Feature with CV ≈ 36.5% is dropped at 30% but kept at 50%.

        values: [50,100,100,150] × 4 → sum_sq = 4 × (50² + 0 + 0 + 50²) = 20,000;
        sd (ddof=1) = sqrt(20000/15) ≈ 36.51 → CV ≈ 36.51 %.
        """
        vals = [50.0, 100.0, 100.0, 150.0] * 4
        abund = _make_abundances({"f": {f"QC_{i + 1}": vals[i] for i in range(16)}})
        # At 30 %, drop
        r1 = run(
            abundances_long=abund,
            samples=samples,
            kept_features={"f"},
            config=QcCvFilterConfig(max_cv_pct=30.0, min_qc_detections=3),
            audit=AuditLogger(stage="test", params={}),
        )
        assert r1.n_kept == 0
        assert r1.n_dropped_cv == 1
        # At 50 %, keep
        r2 = run(
            abundances_long=abund,
            samples=samples,
            kept_features={"f"},
            config=QcCvFilterConfig(max_cv_pct=50.0, min_qc_detections=3),
            audit=AuditLogger(stage="test", params={}),
        )
        assert r2.n_kept == 1
