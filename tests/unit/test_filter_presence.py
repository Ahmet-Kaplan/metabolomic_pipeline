"""Unit tests for the Stage 3 presence (modified 80 %) filter."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from biodisc.audit import AuditLogger
from biodisc.filters.presence import (
    PresenceFilterConfig,
    _is_detected,
    compute_presence_metrics,
    run,
)


# ================================================================
# Helpers
# ================================================================
def _make_samples(group_sizes: dict[str, int]) -> pd.DataFrame:
    """Build a `samples` table with only biological samples for tests."""
    rows = []
    col = 0
    for cell_line, n in group_sizes.items():
        for rep in range(1, n + 1):
            rows.append(
                {
                    "sample_id": f"{cell_line}_{rep}",
                    "raw_sample_id": f"01012026_{cell_line}_{rep}",
                    "injection_type": "biological",
                    "cell_line": cell_line,
                    "replicate": rep,
                    "dilution_level": None,
                    "acq_date": datetime(2026, 1, 1),
                    "column_order": col,
                }
            )
            col += 1
    return pd.DataFrame(rows)


def _make_abundances(
    features: dict[str, dict[str, list[float]]],
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """Build abundances_raw_long where features={feature_id: {cell_line: [per-rep intensities]}}."""
    rows = []
    for fid, group_vals in features.items():
        for cell_line, intensities in group_vals.items():
            sub = samples[samples["cell_line"] == cell_line].sort_values("replicate")
            if len(intensities) != len(sub):
                raise ValueError(
                    f"{fid}/{cell_line}: got {len(intensities)} values, expected {len(sub)}"
                )
            for (_, s), inten in zip(sub.iterrows(), intensities, strict=True):
                rows.append({"feature_id": fid, "sample_id": s["sample_id"], "intensity": inten})
    return pd.DataFrame(rows)


# ================================================================
# _is_detected
# ================================================================
class TestIsDetected:
    def test_positive_is_detected(self) -> None:
        s = pd.Series([1.0, 100.0, 1e9])
        assert _is_detected(s).all()

    def test_zero_is_not_detected(self) -> None:
        # Progenesis exports below-LOD as 0 (not NaN). Must count as missing.
        s = pd.Series([0.0, 0, 0.0])
        assert not _is_detected(s).any()

    def test_nan_is_not_detected(self) -> None:
        s = pd.Series([np.nan, np.nan])
        assert not _is_detected(s).any()

    def test_mixed(self) -> None:
        s = pd.Series([1.0, 0.0, np.nan, 100.0])
        assert _is_detected(s).tolist() == [True, False, False, True]

    def test_negative_not_detected(self) -> None:
        # Defensive: negative intensities shouldn't occur, but if they do, don't count
        s = pd.Series([-1.0, 0.0, 1.0])
        assert _is_detected(s).tolist() == [False, False, True]


# ================================================================
# compute_presence_metrics
# ================================================================
class TestComputePresenceMetrics:
    @pytest.fixture
    def samples(self) -> pd.DataFrame:
        # 3 MCF7 + 2 MCF10A
        return _make_samples({"MCF7": 3, "MCF10A": 2})

    def test_all_present(self, samples: pd.DataFrame) -> None:
        abund = _make_abundances(
            {"f1": {"MCF7": [100.0, 200.0, 300.0], "MCF10A": [50.0, 75.0]}},
            samples,
        )
        m = compute_presence_metrics(abund, samples)
        row = m.iloc[0]
        assert row["MCF7_present"] == 3
        assert row["MCF10A_present"] == 2
        assert row["MCF7_pct"] == pytest.approx(1.0)
        assert row["MCF10A_pct"] == pytest.approx(1.0)
        assert row["max_presence_pct"] == pytest.approx(1.0)

    def test_group_private(self, samples: pd.DataFrame) -> None:
        # MCF7-private feature: present in all MCF7, absent in MCF10A
        abund = _make_abundances(
            {"f_mcf7": {"MCF7": [100.0, 200.0, 300.0], "MCF10A": [0.0, 0.0]}},
            samples,
        )
        m = compute_presence_metrics(abund, samples)
        row = m.iloc[0]
        assert row["MCF7_pct"] == pytest.approx(1.0)
        assert row["MCF10A_pct"] == pytest.approx(0.0)
        assert row["max_presence_pct"] == pytest.approx(1.0)

    def test_nan_treated_as_missing(self, samples: pd.DataFrame) -> None:
        abund = _make_abundances(
            {"f": {"MCF7": [100.0, np.nan, 300.0], "MCF10A": [np.nan, np.nan]}},
            samples,
        )
        m = compute_presence_metrics(abund, samples)
        row = m.iloc[0]
        assert row["MCF7_present"] == 2
        assert row["MCF10A_present"] == 0
        assert row["MCF7_pct"] == pytest.approx(2 / 3)
        assert row["max_presence_pct"] == pytest.approx(2 / 3)

    def test_no_biological_raises(self) -> None:
        # Edge case: samples table with no biological rows
        s = pd.DataFrame(
            [
                {
                    "sample_id": "QC_1",
                    "raw_sample_id": "01012026_QC_1",
                    "injection_type": "qc",
                    "cell_line": None,
                    "replicate": 1,
                    "dilution_level": None,
                    "acq_date": datetime(2026, 1, 1),
                    "column_order": 0,
                }
            ]
        )
        a = pd.DataFrame({"feature_id": ["f"], "sample_id": ["QC_1"], "intensity": [100.0]})
        with pytest.raises(ValueError, match="no biological samples"):
            compute_presence_metrics(a, s)


# ================================================================
# End-to-end run: 3 canonical features + threshold
# ================================================================
class TestRun:
    @pytest.fixture
    def samples(self) -> pd.DataFrame:
        return _make_samples({"MCF7": 7, "MCF10A": 6})

    def test_three_canonical_features(self, samples: pd.DataFrame) -> None:
        """A = group-private (keep), B = ubiquitous (keep), C = sparse noise (drop)."""
        abund = _make_abundances(
            {
                # A: MCF7-private — 7/7 in MCF7, 1/6 in MCF10A
                "A_mcf7_private": {
                    "MCF7": [100, 200, 300, 400, 500, 600, 700],
                    "MCF10A": [0, 0, 0, 100, 0, 0],
                },
                # B: everywhere
                "B_everywhere": {
                    "MCF7": [50, 60, 70, 80, 90, 100, 110],
                    "MCF10A": [55, 65, 75, 85, 95, 105],
                },
                # C: sparse noise — 2/7 in MCF7 (29%), 2/6 in MCF10A (33%)
                "C_noise": {
                    "MCF7": [0, 0, 100, 0, 0, 200, 0],
                    "MCF10A": [150, 0, 0, 250, 0, 0],
                },
            },
            samples,
        )
        audit = AuditLogger(stage="test_03", params={"threshold": 0.8})
        res = run(
            abundances_long=abund,
            samples=samples,
            config=PresenceFilterConfig(presence_pct=0.80),
            audit=audit,
        )
        assert res.n_input == 3
        assert res.n_kept == 2
        assert res.n_dropped == 1
        kept = set(res.kept_features["feature_id"])
        assert kept == {"A_mcf7_private", "B_everywhere"}

        # Metrics should reflect math
        metrics_by_fid = res.metrics.set_index("feature_id")
        assert metrics_by_fid.loc["A_mcf7_private", "max_presence_pct"] == pytest.approx(1.0)
        assert metrics_by_fid.loc["B_everywhere", "max_presence_pct"] == pytest.approx(1.0)
        assert metrics_by_fid.loc["C_noise", "max_presence_pct"] == pytest.approx(2 / 6)

        # Audit rows produced correctly
        audit_df = audit.to_frame()
        assert len(audit_df) == 3
        assert set(audit_df["action"]) == {"kept", "dropped"}

    def test_threshold_variation_flips_borderline(self, samples: pd.DataFrame) -> None:
        """Feature at 5/7 MCF7 (71 %) — dropped at 0.80, kept at 0.70."""
        abund = _make_abundances(
            {
                "border": {
                    "MCF7": [100, 100, 100, 100, 100, 0, 0],  # 5/7 ≈ 71 %
                    "MCF10A": [0, 0, 0, 0, 0, 0],
                }
            },
            samples,
        )

        # At 0.80 → dropped
        audit1 = AuditLogger(stage="t", params={})
        res1 = run(abund, samples, config=PresenceFilterConfig(presence_pct=0.80), audit=audit1)
        assert res1.n_kept == 0

        # At 0.70 → kept
        audit2 = AuditLogger(stage="t", params={})
        res2 = run(abund, samples, config=PresenceFilterConfig(presence_pct=0.70), audit=audit2)
        assert res2.n_kept == 1

    def test_zero_is_missing(self, samples: pd.DataFrame) -> None:
        """Sanity: 0-intensity features should fail the filter."""
        abund = _make_abundances(
            {
                "all_zero": {
                    "MCF7": [0, 0, 0, 0, 0, 0, 0],
                    "MCF10A": [0, 0, 0, 0, 0, 0],
                }
            },
            samples,
        )
        audit = AuditLogger(stage="t", params={})
        res = run(abund, samples, config=PresenceFilterConfig(presence_pct=0.80), audit=audit)
        assert res.n_kept == 0
        assert res.metrics.iloc[0]["max_presence_pct"] == 0.0
