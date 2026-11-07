"""Unit tests for the master feature ledger."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from biodisc.audit_ledger import (
    StageSpec,
    _short,
    build_ledger,
    discover_available_stages,
    rebuild,
)


def _write_metrics(
    interim_dir: Path,
    stage_id: str,
    rows: list[dict[str, object]],
) -> None:
    """Write a synthetic metrics.parquet for a given stage."""
    sub = interim_dir / stage_id
    sub.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(sub / "metrics.parquet", engine="pyarrow", compression="zstd")


@pytest.fixture
def synthetic_pipeline(tmp_path: Path) -> Path:
    """Build a 3-stage synthetic pipeline state on disk.

    5 features. Stage 3 keeps {f1,f2,f3,f4} and drops f5.
    Stage 4 keeps {f1,f2,f3} and drops f4.
    Stage 5 keeps {f1,f2} and drops f3.
    """
    interim = tmp_path / "interim"
    _write_metrics(
        interim,
        "03_presence",
        [
            {"feature_id": "f1", "max_presence_pct": 1.00, "kept": True},
            {"feature_id": "f2", "max_presence_pct": 1.00, "kept": True},
            {"feature_id": "f3", "max_presence_pct": 0.90, "kept": True},
            {"feature_id": "f4", "max_presence_pct": 0.85, "kept": True},
            {"feature_id": "f5", "max_presence_pct": 0.30, "kept": False},
        ],
    )
    _write_metrics(
        interim,
        "04_qc_cv",
        [
            {"feature_id": "f1", "qc_cv_pct": 5.0, "kept": True},
            {"feature_id": "f2", "qc_cv_pct": 12.0, "kept": True},
            {"feature_id": "f3", "qc_cv_pct": 28.0, "kept": True},
            {"feature_id": "f4", "qc_cv_pct": 45.0, "kept": False},
        ],
    )
    _write_metrics(
        interim,
        "05_sample_qc_ratio",
        [
            {"feature_id": "f1", "ratio": 5.0, "kept": True},
            {"feature_id": "f2", "ratio": 1.5, "kept": True},
            {"feature_id": "f3", "ratio": 0.8, "kept": False},
        ],
    )
    return interim


# ============================================================
# discover_available_stages
# ============================================================
class TestDiscover:
    def test_finds_existing_stages(self, synthetic_pipeline: Path) -> None:
        registry = [
            StageSpec("03_presence", "metrics.parquet", "max_presence_pct"),
            StageSpec("04_qc_cv", "metrics.parquet", "qc_cv_pct"),
            StageSpec("05_sample_qc_ratio", "metrics.parquet", "ratio"),
            StageSpec("06_blank", "metrics.parquet", "blank_pct"),  # not on disk
        ]
        avail = discover_available_stages(synthetic_pipeline, registry)
        assert [s.stage_id for s in avail] == ["03_presence", "04_qc_cv", "05_sample_qc_ratio"]

    def test_empty_when_none(self, tmp_path: Path) -> None:
        assert discover_available_stages(tmp_path / "nope") == []


# ============================================================
# _short
# ============================================================
class TestShort:
    def test_normal(self) -> None:
        assert _short("03_presence") == "s3"
        assert _short("04_qc_cv") == "s4"
        assert _short("11_post_drift") == "s11"

    def test_non_numeric_falls_back(self) -> None:
        assert _short("alpha_filter") == "alpha_filter"


# ============================================================
# build_ledger — lean
# ============================================================
class TestBuildLedgerLean:
    @pytest.fixture
    def registry(self) -> list[StageSpec]:
        return [
            StageSpec("03_presence", "metrics.parquet", "max_presence_pct"),
            StageSpec("04_qc_cv", "metrics.parquet", "qc_cv_pct"),
            StageSpec("05_sample_qc_ratio", "metrics.parquet", "ratio"),
        ]

    def test_shape(self, synthetic_pipeline: Path, registry: list[StageSpec]) -> None:
        ledger = build_ledger(synthetic_pipeline, full=False, registry=registry)
        # 5 features (Stage 3 had all 5; later stages have subsets — outer join keeps all)
        assert len(ledger) == 5
        # feature_id + 2 cols/stage × 3 stages + dropped_at_stage + active = 1 + 6 + 2 = 9
        assert len(ledger.columns) == 9
        assert set(ledger.columns) == {
            "feature_id",
            "s3_max_presence_pct",
            "s3_kept",
            "s4_qc_cv_pct",
            "s4_kept",
            "s5_ratio",
            "s5_kept",
            "dropped_at_stage",
            "active",
        }

    def test_dropped_at_stage(self, synthetic_pipeline: Path, registry: list[StageSpec]) -> None:
        ledger = build_ledger(synthetic_pipeline, full=False, registry=registry)
        d = ledger.set_index("feature_id")["dropped_at_stage"].to_dict()
        assert d["f1"] is None
        assert d["f2"] is None
        assert d["f3"] == "05_sample_qc_ratio"
        assert d["f4"] == "04_qc_cv"
        assert d["f5"] == "03_presence"

    def test_active_flag(self, synthetic_pipeline: Path, registry: list[StageSpec]) -> None:
        ledger = build_ledger(synthetic_pipeline, full=False, registry=registry)
        active = ledger.set_index("feature_id")["active"].to_dict()
        assert active == {"f1": True, "f2": True, "f3": False, "f4": False, "f5": False}

    def test_metric_values_preserved(
        self, synthetic_pipeline: Path, registry: list[StageSpec]
    ) -> None:
        ledger = build_ledger(synthetic_pipeline, full=False, registry=registry).set_index(
            "feature_id"
        )
        assert ledger.loc["f1", "s3_max_presence_pct"] == 1.0
        assert ledger.loc["f1", "s4_qc_cv_pct"] == 5.0
        assert ledger.loc["f1", "s5_ratio"] == 5.0
        # f5 was dropped at stage 3 → no s4/s5 metrics
        assert pd.isna(ledger.loc["f5", "s4_qc_cv_pct"])
        assert pd.isna(ledger.loc["f5", "s5_ratio"])


# ============================================================
# build_ledger — full
# ============================================================
class TestBuildLedgerFull:
    def test_full_includes_all_columns(self, synthetic_pipeline: Path) -> None:
        registry = [
            StageSpec("03_presence", "metrics.parquet", "max_presence_pct"),
            StageSpec("04_qc_cv", "metrics.parquet", "qc_cv_pct"),
            StageSpec("05_sample_qc_ratio", "metrics.parquet", "ratio"),
        ]
        ledger = build_ledger(synthetic_pipeline, full=True, registry=registry)
        # In full mode every metric column gets prefixed: identical to lean here
        # because synthetic data has only 2 cols per stage; verify prefix scheme works
        assert "s3_max_presence_pct" in ledger.columns
        assert "s3_kept" in ledger.columns
        assert "s4_qc_cv_pct" in ledger.columns
        assert "s5_ratio" in ledger.columns
        assert "dropped_at_stage" in ledger.columns


# ============================================================
# rebuild — end-to-end (lean + persistence)
# ============================================================
def test_rebuild_writes_both(tmp_path: Path, synthetic_pipeline: Path) -> None:
    audit = tmp_path / "audit"
    out = rebuild(
        interim_dir=synthetic_pipeline,
        audit_dir=audit,
        full=False,
        emit_csv=True,
    )
    assert out["parquet"].exists()
    assert out["csv"].exists()
    df = pd.read_parquet(out["parquet"])
    assert len(df) == 5  # all features represented


def test_rebuild_is_idempotent(tmp_path: Path, synthetic_pipeline: Path) -> None:
    """Running rebuild twice must produce bit-identical output."""
    audit = tmp_path / "audit"
    out1 = rebuild(interim_dir=synthetic_pipeline, audit_dir=audit, emit_csv=False)
    df1 = pd.read_parquet(out1["parquet"])
    out2 = rebuild(interim_dir=synthetic_pipeline, audit_dir=audit, emit_csv=False)
    df2 = pd.read_parquet(out2["parquet"])
    assert df1.equals(df2)
