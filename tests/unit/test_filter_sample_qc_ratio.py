"""Unit tests for the Stage 5 Sample-RSD / QC-RSD ratio filter."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from biodisc.audit import AuditLogger
from biodisc.filters.sample_qc_ratio import (
    SampleQcRatioFilterConfig,
    _compute_ratio,
    _rsd_pct,
    compute_metrics,
    run,
)


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def _make_samples(n_bio: int = 10, n_qc: int = 8) -> pd.DataFrame:
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


def _make_abundances(features: dict[str, dict[str, list[float] | float]]) -> pd.DataFrame:
    rows = []
    for fid, per_sample in features.items():
        for sid, inten in per_sample.items():
            rows.append({"feature_id": fid, "sample_id": sid, "intensity": inten})
    return pd.DataFrame(rows)


# ============================================================
# _rsd_pct
# ============================================================
class TestRsdPct:
    def test_constant_zero_rsd(self) -> None:
        mean, sd, rsd, n = _rsd_pct(np.array([100.0, 100.0, 100.0, 100.0]))
        assert mean == pytest.approx(100.0)
        assert sd == pytest.approx(0.0)
        assert rsd == pytest.approx(0.0)
        assert n == 4

    def test_known_rsd(self) -> None:
        # [10, 20, 30] → mean=20, sd(ddof=1)=10, RSD=50%
        mean, sd, rsd, n = _rsd_pct(np.array([10.0, 20.0, 30.0]))
        assert mean == pytest.approx(20.0)
        assert sd == pytest.approx(10.0)
        assert rsd == pytest.approx(50.0)
        assert n == 3

    def test_zero_and_nan_excluded(self) -> None:
        _, _, _, n = _rsd_pct(np.array([10.0, 0.0, np.nan, 20.0, 30.0]))
        assert n == 3

    def test_single_detection(self) -> None:
        mean, sd, rsd, n = _rsd_pct(np.array([100.0, 0.0, 0.0]))
        assert n == 1
        assert math.isnan(sd) and math.isnan(rsd)
        assert mean == pytest.approx(100.0)

    def test_all_zero(self) -> None:
        mean, sd, rsd, n = _rsd_pct(np.array([0.0, 0.0, 0.0]))
        assert n == 0
        assert math.isnan(mean) and math.isnan(sd) and math.isnan(rsd)


# ============================================================
# _compute_ratio (edge cases)
# ============================================================
class TestComputeRatio:
    def test_normal(self) -> None:
        assert _compute_ratio(50.0, 25.0) == pytest.approx(2.0)

    def test_qc_zero_bio_positive_returns_inf(self) -> None:
        # Perfect QC (no technical noise), any biology — definitely keep
        assert math.isinf(_compute_ratio(50.0, 0.0))

    def test_qc_zero_bio_zero_returns_nan(self) -> None:
        # Both flat — no information either way
        assert math.isnan(_compute_ratio(0.0, 0.0))

    def test_nan_propagates(self) -> None:
        assert math.isnan(_compute_ratio(float("nan"), 10.0))
        assert math.isnan(_compute_ratio(10.0, float("nan")))


# ============================================================
# compute_metrics
# ============================================================
class TestComputeMetrics:
    @pytest.fixture
    def samples(self) -> pd.DataFrame:
        return _make_samples(n_bio=4, n_qc=4)

    def test_ratio_two(self, samples: pd.DataFrame) -> None:
        # Bio: [100, 200, 300, 400] → mean=250, sd≈129.10, RSD≈51.64%
        # QC : [180, 200, 220, 200] → mean=200, sd≈16.33,  RSD≈ 8.16%
        # ratio ≈ 6.33
        bio_vals = [100.0, 200.0, 300.0, 400.0]
        qc_vals = [180.0, 200.0, 220.0, 200.0]
        abund = _make_abundances(
            {
                "f": {
                    **{f"Bio_{i + 1}": bio_vals[i] for i in range(4)},
                    **{f"QC_{i + 1}": qc_vals[i] for i in range(4)},
                },
            }
        )
        m = compute_metrics(abund, samples)
        row = m.iloc[0]
        # Check rsd values
        assert row["sample_n_detected"] == 4
        assert row["qc_n_detected"] == 4
        assert row["sample_rsd_pct"] == pytest.approx(
            100 * np.std(bio_vals, ddof=1) / np.mean(bio_vals), rel=1e-6
        )
        assert row["qc_rsd_pct"] == pytest.approx(
            100 * np.std(qc_vals, ddof=1) / np.mean(qc_vals), rel=1e-6
        )
        # ratio
        expected_ratio = (100 * np.std(bio_vals, ddof=1) / np.mean(bio_vals)) / (
            100 * np.std(qc_vals, ddof=1) / np.mean(qc_vals)
        )
        assert row["ratio"] == pytest.approx(expected_ratio, rel=1e-6)

    def test_filter_to_kept_features(self, samples: pd.DataFrame) -> None:
        abund = _make_abundances(
            {
                "keep_me": {
                    **{f"Bio_{i + 1}": 100.0 for i in range(4)},
                    **{f"QC_{i + 1}": 100.0 for i in range(4)},
                },
                "ignore_me": {
                    **{f"Bio_{i + 1}": 100.0 for i in range(4)},
                    **{f"QC_{i + 1}": 100.0 for i in range(4)},
                },
            }
        )
        m = compute_metrics(abund, samples, kept_features={"keep_me"})
        assert len(m) == 1
        assert m.iloc[0]["feature_id"] == "keep_me"

    def test_no_bio_raises(self) -> None:
        s = _make_samples(n_bio=0, n_qc=4)
        a = _make_abundances({"f": {f"QC_{i + 1}": 100.0 for i in range(4)}})
        with pytest.raises(ValueError, match="no biological"):
            compute_metrics(a, s)

    def test_no_qc_raises(self) -> None:
        s = _make_samples(n_bio=4, n_qc=0)
        a = _make_abundances({"f": {f"Bio_{i + 1}": 100.0 for i in range(4)}})
        with pytest.raises(ValueError, match="no QC"):
            compute_metrics(a, s)


# ============================================================
# run
# ============================================================
class TestRun:
    @pytest.fixture
    def samples(self) -> pd.DataFrame:
        return _make_samples(n_bio=6, n_qc=6)

    def test_canonical_features(self, samples: pd.DataFrame) -> None:
        """Strong-bio (kept), weak-bio (dropped), undefined (dropped), too-rare (dropped)."""
        # keep_strong: bio var >> qc var
        bio_strong = [50, 100, 150, 200, 250, 300]
        qc_tight = [99, 100, 101, 100, 99, 101]
        # drop_weak: bio var < qc var
        bio_weak = [99, 100, 101, 99, 100, 101]
        qc_noisy = [50, 100, 150, 200, 250, 300]
        # drop_undefined: both flat (zero rsd in both)
        bio_flat = [100, 100, 100, 100, 100, 100]
        qc_flat = [100, 100, 100, 100, 100, 100]
        # drop_rare: only 1 detection in bio
        bio_rare = [100, 0, 0, 0, 0, 0]
        qc_ok = [100, 110, 90, 100, 105, 95]

        abund = _make_abundances(
            {
                "keep_strong": {
                    **{f"Bio_{i + 1}": bio_strong[i] for i in range(6)},
                    **{f"QC_{i + 1}": qc_tight[i] for i in range(6)},
                },
                "drop_weak": {
                    **{f"Bio_{i + 1}": bio_weak[i] for i in range(6)},
                    **{f"QC_{i + 1}": qc_noisy[i] for i in range(6)},
                },
                "drop_undefined": {
                    **{f"Bio_{i + 1}": bio_flat[i] for i in range(6)},
                    **{f"QC_{i + 1}": qc_flat[i] for i in range(6)},
                },
                "drop_rare": {
                    **{f"Bio_{i + 1}": bio_rare[i] for i in range(6)},
                    **{f"QC_{i + 1}": qc_ok[i] for i in range(6)},
                },
            }
        )
        audit = AuditLogger(stage="test_05", params={})
        res = run(
            abundances_long=abund,
            samples=samples,
            kept_features={"keep_strong", "drop_weak", "drop_undefined", "drop_rare"},
            config=SampleQcRatioFilterConfig(
                min_ratio=1.0, min_sample_detections=3, min_qc_detections=3
            ),
            audit=audit,
        )
        kept = set(res.kept_features["feature_id"])
        assert "keep_strong" in kept
        assert "drop_weak" not in kept
        assert "drop_undefined" not in kept
        assert "drop_rare" not in kept
        assert res.n_dropped_ratio == 1  # drop_weak
        assert res.n_dropped_undefined == 1  # drop_undefined (both RSD=0)
        assert res.n_dropped_detections == 1  # drop_rare (n_bio=1<3)

    def test_threshold_configurability(self, samples: pd.DataFrame) -> None:
        # Construct: bio_rsd / qc_rsd ≈ 1.5
        # Bio: [80,100,120] repeated → flatten to 6: [80,100,120,80,100,120]
        # mean=100, sd=sqrt(sum_sq_dev/5); sum_sq=2*(400+0+400)=1600 → sd=sqrt(320)=17.89, RSD=17.89%
        # QC: [88,100,112] → at half spread of bio: mean=100, sd=sqrt(2*(144+0+144)/5)=sqrt(115.2)=10.73, RSD=10.73%
        # ratio ≈ 17.89 / 10.73 ≈ 1.667
        bio = [80, 100, 120, 80, 100, 120]
        qc = [88, 100, 112, 88, 100, 112]
        abund = _make_abundances(
            {
                "f": {
                    **{f"Bio_{i + 1}": bio[i] for i in range(6)},
                    **{f"QC_{i + 1}": qc[i] for i in range(6)},
                }
            }
        )
        # At threshold 1.0 → kept (ratio ≈ 1.67)
        r1 = run(
            abundances_long=abund,
            samples=samples,
            kept_features={"f"},
            config=SampleQcRatioFilterConfig(min_ratio=1.0),
            audit=AuditLogger(stage="t", params={}),
        )
        assert r1.n_kept == 1
        # At threshold 2.0 → dropped (ratio < 2.0)
        r2 = run(
            abundances_long=abund,
            samples=samples,
            kept_features={"f"},
            config=SampleQcRatioFilterConfig(min_ratio=2.0),
            audit=AuditLogger(stage="t", params={}),
        )
        assert r2.n_kept == 0
        assert r2.n_dropped_ratio == 1
