"""Unit tests for the Stage 6 blank-percentage filter."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from biodisc.audit import AuditLogger
from biodisc.filters.blank import (
    BlankFilterConfig,
    _compute_blank_pct,
    _mean_detected,
    compute_metrics,
    run,
)


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def _make_samples(n_bio: int = 4, n_qc: int = 4) -> pd.DataFrame:
    rows = []
    col = 0
    for rep in range(1, n_bio + 1):
        rows.append(
            {
                "sample_id": f"Bio_{rep}",
                "raw_sample_id": f"01_Bio_{rep}",
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
                "raw_sample_id": f"01_QC_{rep}",
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


def _make_abundances(features: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for fid, per_sample in features.items():
        for sid, inten in per_sample.items():
            rows.append({"feature_id": fid, "sample_id": sid, "intensity": inten})
    return pd.DataFrame(rows)


def _make_blanks(features: dict[str, list[float]]) -> pd.DataFrame:
    """blank intensities per feature: 3 replicates per feature in raw block."""
    rows = []
    for fid, vals in features.items():
        for rep, v in enumerate(vals, 1):
            rows.append(
                {
                    "feature_id": fid,
                    "blank_replicate": rep,
                    "block": "raw",
                    "intensity": v,
                }
            )
    return pd.DataFrame(rows)


# ============================================================
# _mean_detected
# ============================================================
class TestMeanDetected:
    def test_normal(self) -> None:
        m, n = _mean_detected(np.array([10.0, 20.0, 30.0]))
        assert m == pytest.approx(20.0) and n == 3

    def test_zero_excluded(self) -> None:
        m, n = _mean_detected(np.array([0.0, 10.0, 0.0, 20.0]))
        assert m == pytest.approx(15.0) and n == 2

    def test_all_zero(self) -> None:
        m, n = _mean_detected(np.array([0.0, 0.0, 0.0]))
        assert n == 0 and math.isnan(m)


# ============================================================
# _compute_blank_pct (edge cases)
# ============================================================
class TestComputeBlankPct:
    def test_normal(self) -> None:
        assert _compute_blank_pct(10.0, 100.0) == pytest.approx(10.0)

    def test_clean(self) -> None:
        # blank=0 → pct=0
        assert _compute_blank_pct(0.0, 100.0) == pytest.approx(0.0)

    def test_no_signal(self) -> None:
        # mean_sq=0, mean_blank>0 → +inf (drop)
        assert math.isinf(_compute_blank_pct(5.0, 0.0))

    def test_both_zero(self) -> None:
        # mean_sq=0, mean_blank=0 → NaN
        assert math.isnan(_compute_blank_pct(0.0, 0.0))

    def test_mean_sq_nan(self) -> None:
        assert math.isnan(_compute_blank_pct(10.0, float("nan")))


# ============================================================
# compute_metrics
# ============================================================
class TestComputeMetrics:
    @pytest.fixture
    def samples(self) -> pd.DataFrame:
        return _make_samples(n_bio=2, n_qc=2)

    def test_clean_feature(self, samples: pd.DataFrame) -> None:
        # Feature with all blanks zero, sample+QC = 100
        abund = _make_abundances(
            {
                "f": {f"Bio_{i + 1}": 100.0 for i in range(2)}
                | {f"QC_{i + 1}": 100.0 for i in range(2)}
            }
        )
        bk = _make_blanks({"f": [0.0, 0.0, 0.0]})
        m = compute_metrics(abund, bk, samples)
        row = m.iloc[0]
        assert row["mean_blank"] == pytest.approx(0.0)
        assert row["mean_sq"] == pytest.approx(100.0)
        assert row["blank_pct"] == pytest.approx(0.0)

    def test_borderline_20pct(self, samples: pd.DataFrame) -> None:
        # blank=10, sample=50 → 20%
        abund = _make_abundances(
            {
                "f": {f"Bio_{i + 1}": 50.0 for i in range(2)}
                | {f"QC_{i + 1}": 50.0 for i in range(2)}
            }
        )
        bk = _make_blanks({"f": [10.0, 10.0, 10.0]})
        m = compute_metrics(abund, bk, samples)
        row = m.iloc[0]
        assert row["blank_pct"] == pytest.approx(20.0)

    def test_no_sq_raises(self) -> None:
        # Samples table with only a blank injection — no biological or QC
        s = pd.DataFrame(
            [
                {
                    "sample_id": "Blank_1",
                    "raw_sample_id": "01_Blank_1",
                    "injection_type": "blank",
                    "cell_line": None,
                    "replicate": 1,
                    "dilution_level": None,
                    "acq_date": datetime(2026, 1, 1),
                    "column_order": 0,
                }
            ]
        )
        a = _make_abundances({"f": {"Blank_1": 100.0}})
        bk = _make_blanks({"f": [0.0, 0.0, 0.0]})
        with pytest.raises(ValueError, match="no biological/QC"):
            compute_metrics(a, bk, s)


# ============================================================
# run
# ============================================================
class TestRun:
    @pytest.fixture
    def samples(self) -> pd.DataFrame:
        return _make_samples(n_bio=4, n_qc=4)

    def test_three_canonical_features_at_20pct(self, samples: pd.DataFrame) -> None:
        abund = _make_abundances(
            {
                "clean": {f"Bio_{i + 1}": 100.0 for i in range(4)}
                | {f"QC_{i + 1}": 100.0 for i in range(4)},
                "border": {f"Bio_{i + 1}": 50.0 for i in range(4)}
                | {f"QC_{i + 1}": 50.0 for i in range(4)},
                "high": {f"Bio_{i + 1}": 10.0 for i in range(4)}
                | {f"QC_{i + 1}": 10.0 for i in range(4)},
                "rare": {f"Bio_{i + 1}": (100.0 if i == 0 else 0.0) for i in range(4)}
                | {f"QC_{i + 1}": 0.0 for i in range(4)},
            }
        )
        bk = _make_blanks(
            {
                "clean": [0.0, 0.0, 0.0],  # blank_pct = 0
                "border": [10.0, 10.0, 10.0],  # blank_pct = 20
                "high": [10.0, 10.0, 10.0],  # blank_pct = 100
                "rare": [5.0, 5.0, 5.0],
            }
        )
        audit = AuditLogger(stage="test_06", params={})
        res = run(
            abundances_long=abund,
            blanks_long=bk,
            samples=samples,
            kept_features={"clean", "border", "high", "rare"},
            config=BlankFilterConfig(max_blank_pct=20.0, min_sq_detections=3),
            audit=audit,
        )
        kept = set(res.kept_features["feature_id"])
        assert "clean" in kept
        assert "border" in kept  # exactly 20% — rule is `>` so 20 keeps
        assert "high" not in kept
        assert "rare" not in kept
        assert res.n_dropped_pct == 1  # high
        assert res.n_dropped_detections == 1  # rare (n_sq=1<3)

    def test_threshold_configurability(self, samples: pd.DataFrame) -> None:
        # blank=10, sample=20 → blank_pct = 50%
        abund = _make_abundances(
            {
                "f": {f"Bio_{i + 1}": 20.0 for i in range(4)}
                | {f"QC_{i + 1}": 20.0 for i in range(4)},
            }
        )
        bk = _make_blanks({"f": [10.0, 10.0, 10.0]})
        # At 20%, drop
        r1 = run(
            abundances_long=abund,
            blanks_long=bk,
            samples=samples,
            kept_features={"f"},
            config=BlankFilterConfig(max_blank_pct=20.0),
            audit=AuditLogger(stage="t", params={}),
        )
        assert r1.n_kept == 0
        # At 50%, keep
        r2 = run(
            abundances_long=abund,
            blanks_long=bk,
            samples=samples,
            kept_features={"f"},
            config=BlankFilterConfig(max_blank_pct=50.0),
            audit=AuditLogger(stage="t", params={}),
        )
        assert r2.n_kept == 1
