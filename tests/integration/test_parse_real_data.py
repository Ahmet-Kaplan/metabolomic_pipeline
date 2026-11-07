"""Integration test — Stage 1 parser on the real 4-CSV Progenesis export.

Catches silent regressions: if a future refactor to the parser produces a
different feature count / sample count / shape, this test fails loudly.

Skipped automatically if the real data isn't on disk (e.g. CI without data).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# --- paths relative to project root ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "progenesisqi_output"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "01_parsed"

REQUIRED_RAW_FILES = [
    "I_66_CELL_METABOLOMICS_COMPOUND_MEASUREMENT(in).csv",
    "Blank(in).csv",
    "dİLUTION(in).csv",
    "I_66_CELL_METABOLOMICS_COMPOUND_IDENTIFICATION.csv",
]

# Known invariants of the I_66 Progenesis export, from the 2026-04-24 snapshot.
# If any of these numbers drifts, investigate BEFORE updating the expected value.
EXPECTED_FEATURE_COUNT = 30_370
EXPECTED_SAMPLE_COUNT = 53
EXPECTED_SAMPLES_BY_TYPE = {
    "biological": 31,
    "qc": 16,
    "blank": 3,
    "dilution": 3,
}
EXPECTED_IDENTIFICATION_ROWS = 1_004_678
EXPECTED_UNIQUE_IDENTIFIED_FEATURES = 20_274

# Sentinel feature seen in the raw data — known neutral mass & measured CCS
SENTINEL_FEATURE = "11.04_665.3056n"
SENTINEL_NEUTRAL_MASS_PPM_TOL = 10  # ppm tolerance for the mass check


@pytest.fixture(scope="module")
def ensure_stage1_output() -> Path:
    """Skip the integration suite cleanly when raw data or interim outputs are absent.

    This lets unit tests pass in a clean CI environment while still running the
    full regression on a developer workstation.
    """
    missing_raw = [f for f in REQUIRED_RAW_FILES if not (RAW_DIR / f).exists()]
    if missing_raw:
        pytest.skip(f"raw Progenesis files missing: {missing_raw}")
    if not INTERIM_DIR.exists():
        pytest.skip(f"stage-1 interim output missing at {INTERIM_DIR} — run `biodisc parse` first")
    return INTERIM_DIR


@pytest.mark.integration
class TestStage1Snapshot:
    def test_features_count_and_uniqueness(self, ensure_stage1_output: Path) -> None:
        features = pd.read_parquet(ensure_stage1_output / "features.parquet")
        assert len(features) == EXPECTED_FEATURE_COUNT, (
            f"feature count changed: {len(features)} vs expected {EXPECTED_FEATURE_COUNT}"
        )
        assert features["feature_id"].is_unique

    def test_samples_count_and_injection_types(self, ensure_stage1_output: Path) -> None:
        samples = pd.read_parquet(ensure_stage1_output / "samples.parquet")
        assert len(samples) == EXPECTED_SAMPLE_COUNT
        assert samples["sample_id"].is_unique
        counts = samples["injection_type"].value_counts().to_dict()
        for injection_type, expected in EXPECTED_SAMPLES_BY_TYPE.items():
            assert counts.get(injection_type, 0) == expected, (
                f"{injection_type}: got {counts.get(injection_type, 0)} expected {expected}"
            )

    def test_abundances_long_shape(self, ensure_stage1_output: Path) -> None:
        raw = pd.read_parquet(ensure_stage1_output / "abundances_raw_long.parquet")
        norm = pd.read_parquet(ensure_stage1_output / "abundances_normalised_long.parquet")
        # long = features × biological+QC samples only (47)
        expected_long = EXPECTED_FEATURE_COUNT * 47
        assert len(raw) == expected_long
        assert len(norm) == expected_long
        assert set(raw.columns) == {"feature_id", "sample_id", "intensity"}

    def test_abundances_wide_shape(self, ensure_stage1_output: Path) -> None:
        raw_wide = pd.read_parquet(ensure_stage1_output / "abundances_raw_wide.parquet")
        # 30,370 rows (features) × 48 columns (feature_id + 47 samples)
        assert raw_wide.shape == (EXPECTED_FEATURE_COUNT, 48)

    def test_identification_long_shape(self, ensure_stage1_output: Path) -> None:
        ident = pd.read_parquet(ensure_stage1_output / "identification_long.parquet")
        assert len(ident) == EXPECTED_IDENTIFICATION_ROWS
        assert ident["feature_id"].nunique() == EXPECTED_UNIQUE_IDENTIFIED_FEATURES

    def test_sentinel_feature_survives(self, ensure_stage1_output: Path) -> None:
        features = pd.read_parquet(ensure_stage1_output / "features.parquet")
        row = features[features["feature_id"] == SENTINEL_FEATURE]
        assert len(row) == 1, "sentinel feature missing from parsed output"
        # Known values from the 2026-04-24 snapshot
        assert row["neutral_mass"].iloc[0] == pytest.approx(665.305569, rel=1e-6)
        assert row["m_over_z"].iloc[0] == pytest.approx(648.307451, rel=1e-6)
        assert row["measured_ccs"].iloc[0] == pytest.approx(257.151485, rel=1e-6)

    def test_audit_every_feature_kept(self, ensure_stage1_output: Path) -> None:
        audit_path = PROJECT_ROOT / "data" / "audit" / "01_parse.parquet"
        if not audit_path.exists():
            pytest.skip("stage-1 audit missing — re-run `biodisc parse`")
        audit = pd.read_parquet(audit_path)
        assert len(audit) == EXPECTED_FEATURE_COUNT
        assert (audit["action"] == "kept").all()
        assert (audit["reason"] == "parsed_ok").all()

    def test_dilution_ladder_and_blanks_survive(self, ensure_stage1_output: Path) -> None:
        dilution = pd.read_parquet(ensure_stage1_output / "dilution_ladder_long.parquet")
        blanks = pd.read_parquet(ensure_stage1_output / "blanks_long.parquet")
        # 3 dilution factors × 30,370 features × 2 blocks = 182,220 (dilutions < 1.0 only)
        assert (dilution["dilution_factor"] < 1.0).sum() == 30_370 * 3 * 2
        # 3 blank replicates × 30,370 features × 2 blocks = 182,220
        assert len(blanks) == 30_370 * 3 * 2
