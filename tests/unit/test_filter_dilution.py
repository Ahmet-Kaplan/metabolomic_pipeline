"""Unit tests for the Stage 7 dilution-linearity filter."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from biodisc.audit import AuditLogger
from biodisc.filters.dilution import (
    DILUTION_FACTORS,
    DilutionFilterConfig,
    _fit_r2_linear,
    compute_metrics,
    run,
)


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def _make_dilution_long(features: dict[str, list[float]]) -> pd.DataFrame:
    """Build dilution_ladder_long for tests.

    `features = {feature_id: [y_at_factor_1, y_at_0.5, y_at_0.25, y_at_0.125]}`
    Single QC injection per factor (factor=1.0 not duplicated as in the real
    data — that's fine, mean of 1 element is the element itself).
    """
    rows = []
    for fid, ys in features.items():
        for x, y in zip(DILUTION_FACTORS, ys, strict=True):
            rows.append(
                {
                    "feature_id": fid,
                    "dilution_level": 1.0 / x,
                    "dilution_factor": x,
                    "block": "raw",
                    "intensity": y,
                }
            )
    return pd.DataFrame(rows)


# ============================================================
# _fit_r2_linear
# ============================================================
class TestFitR2:
    def test_perfect_line(self) -> None:
        x = np.array(DILUTION_FACTORS)
        y = 100.0 * x  # perfectly linear, no intercept
        slope, intercept, r2, n = _fit_r2_linear(x, y, min_detections=3)
        assert n == 4
        assert slope == pytest.approx(100.0)
        assert intercept == pytest.approx(0.0, abs=1e-10)
        assert r2 == pytest.approx(1.0)

    def test_known_imperfect(self) -> None:
        # Compute by hand: y = [100, 60, 30, 10]; mean=50; expected sd large
        x = np.array(DILUTION_FACTORS)
        y = np.array([100.0, 60.0, 30.0, 10.0])
        _, _, r2, n = _fit_r2_linear(x, y, min_detections=3)
        # numpy gives a specific value; just check ≥ 0.9 since this is roughly linear
        assert n == 4
        assert r2 >= 0.9
        assert r2 <= 1.0

    def test_constant_y_undefined(self) -> None:
        x = np.array(DILUTION_FACTORS)
        y = np.array([50.0, 50.0, 50.0, 50.0])
        _, _, r2, n = _fit_r2_linear(x, y, min_detections=3)
        assert math.isnan(r2)
        assert n == 4

    def test_zero_excluded(self) -> None:
        x = np.array(DILUTION_FACTORS)
        y = np.array([100.0, 50.0, 0.0, 0.0])
        _, _, r2, n = _fit_r2_linear(x, y, min_detections=3)
        # Only 2 detections — below threshold of 3 → NaN
        assert math.isnan(r2)
        assert n == 2

    def test_3_of_4_passes(self) -> None:
        x = np.array(DILUTION_FACTORS)
        y = np.array([100.0, 50.0, 25.0, 0.0])
        _, _, r2, n = _fit_r2_linear(x, y, min_detections=3)
        assert n == 3
        # 3 points are linear-ish; R² should be high
        assert not math.isnan(r2)

    def test_all_zero(self) -> None:
        x = np.array(DILUTION_FACTORS)
        y = np.zeros(4)
        _, _, r2, n = _fit_r2_linear(x, y, min_detections=3)
        assert n == 0
        assert math.isnan(r2)


# ============================================================
# compute_metrics
# ============================================================
class TestComputeMetrics:
    def test_one_feature_perfect(self) -> None:
        dl = _make_dilution_long({"f1": [100.0, 50.0, 25.0, 12.5]})
        m = compute_metrics(dl)
        row = m.iloc[0]
        assert row["feature_id"] == "f1"
        assert row["n_detected"] == 4
        assert row["r2_linear"] == pytest.approx(1.0)
        assert row["y_factor_1"] == pytest.approx(100.0)
        assert row["y_factor_0_125"] == pytest.approx(12.5)

    def test_kept_features_filter(self) -> None:
        dl = _make_dilution_long(
            {
                "keep_me": [100.0, 50.0, 25.0, 12.5],
                "ignore_me": [100.0, 50.0, 25.0, 12.5],
            }
        )
        m = compute_metrics(dl, kept_features={"keep_me"})
        assert len(m) == 1
        assert m.iloc[0]["feature_id"] == "keep_me"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="no rows"):
            compute_metrics(
                pd.DataFrame(columns=["feature_id", "dilution_factor", "block", "intensity"])
            )


# ============================================================
# run
# ============================================================
class TestRun:
    def test_canonical_features(self) -> None:
        dl = _make_dilution_long(
            {
                "perfect": [100.0, 50.0, 25.0, 12.5],  # R² = 1.0
                "good": [100.0, 60.0, 30.0, 10.0],  # R² ~ 0.99
                "constant": [50.0, 50.0, 50.0, 50.0],  # R² undefined (variance zero)
                "bad": [100.0, 80.0, 90.0, 85.0],  # R² ≈ 0.46 (saturated — barely dilutes)
                "rare": [100.0, 0.0, 0.0, 0.0],  # only 1 detection
            }
        )
        audit = AuditLogger(stage="test_07", params={})
        res = run(
            dilution_long=dl,
            kept_features={"perfect", "good", "constant", "bad", "rare"},
            config=DilutionFilterConfig(min_r2_linear=0.7, min_dilution_detections=3),
            audit=audit,
        )
        kept = set(res.kept_features["feature_id"])
        assert "perfect" in kept
        assert "good" in kept
        assert "constant" not in kept  # variance zero → undefined
        assert "bad" not in kept  # R² < 0.7 expected
        assert "rare" not in kept  # n=1 < 3
        # Verify drop reasons partitioned correctly
        assert res.n_dropped_undefined == 1  # constant
        assert res.n_dropped_detections == 1  # rare
        assert res.n_dropped_r2 == 1  # bad

    def test_threshold_configurability(self) -> None:
        # y = [100, 50, 30, 25]; not perfectly linear but reasonable
        dl = _make_dilution_long({"f": [100.0, 50.0, 30.0, 25.0]})
        # At threshold 0.95, may drop or keep depending on exact R²
        # Compute true R² manually
        import numpy as np

        x = np.array(DILUTION_FACTORS)
        y = np.array([100.0, 50.0, 30.0, 25.0])
        slope, intercept = np.polyfit(x, y, 1)
        yp = slope * x + intercept
        r2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()

        # Pick a threshold strictly above r2 so it drops
        thr_drop = r2 + 0.01
        thr_keep = r2 - 0.01

        r1 = run(
            dilution_long=dl,
            kept_features={"f"},
            config=DilutionFilterConfig(min_r2_linear=thr_drop),
            audit=AuditLogger(stage="t", params={}),
        )
        assert r1.n_kept == 0

        r2_run = run(
            dilution_long=dl,
            kept_features={"f"},
            config=DilutionFilterConfig(min_r2_linear=thr_keep),
            audit=AuditLogger(stage="t", params={}),
        )
        assert r2_run.n_kept == 1
