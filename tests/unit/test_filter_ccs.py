"""Unit tests for the Stage 8 CCS error filter."""

from __future__ import annotations

import pandas as pd
import pytest

from biodisc.audit import AuditLogger
from biodisc.external.ccs_reference import normalise_adduct
from biodisc.filters.ccs import (
    CCSFilterConfig,
    _bias_factor,
    _split_progenesis_adducts,
    run,
)


# ============================================================
# Helpers
# ============================================================
def _make_features(d: dict[str, float | None]) -> pd.DataFrame:
    """Build features.parquet-equivalent from {feature_id: measured_ccs}."""
    return pd.DataFrame({"feature_id": list(d.keys()), "measured_ccs": list(d.values())})


def _make_identification(rows: list[dict]) -> pd.DataFrame:
    """rows: list of dicts with feature_id, compound_id, adducts, score, mass_error_ppm."""
    cols = ["feature_id", "compound_id", "adducts", "score", "mass_error_ppm"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for col, default in [("score", 50.0), ("mass_error_ppm", 1.0)]:
        if col not in df.columns:
            df[col] = default
    return df


def _make_reference(rows: list[dict]) -> pd.DataFrame:
    """rows: list of dicts with hmdb_id, adduct, reference_ccs, instrument_type."""
    cols = [
        "hmdb_id",
        "adduct",
        "reference_ccs",
        "instrument_type",
        "compound_name",
        "source_db",
        "confidence_tier",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for col, default in [
        ("compound_name", "test_compound"),
        ("source_db", "TEST"),
        ("confidence_tier", 2),
    ]:
        if col not in df.columns:
            df[col] = default
    return df


# ============================================================
# Adduct normalisation + splitting
# ============================================================
class TestAdductHelpers:
    def test_normalise_strips_brackets(self) -> None:
        assert normalise_adduct("[M+H]") == "M+H"
        assert normalise_adduct(" [M-H] ") == "M-H"
        assert normalise_adduct("[M+Na]") == "M+Na"

    def test_normalise_handles_none(self) -> None:
        assert normalise_adduct(None) is None
        assert normalise_adduct(float("nan")) is None

    def test_split_progenesis_adducts(self) -> None:
        out = _split_progenesis_adducts("M+H, M+K, M+Na, M+H-H2O")
        assert out == ["M+H", "M+K", "M+Na", "M+H-H2O"]

    def test_split_handles_none_and_empty(self) -> None:
        assert _split_progenesis_adducts(None) == []
        assert _split_progenesis_adducts(float("nan")) == []
        assert _split_progenesis_adducts("") == []

    def test_split_strips_whitespace(self) -> None:
        assert _split_progenesis_adducts(" M+H ,  M+Na  ") == ["M+H", "M+Na"]


# ============================================================
# Bias-factor lookup
# ============================================================
class TestBiasFactor:
    def test_dtims(self) -> None:
        cfg = CCSFilterConfig(twims_bias_dtims=1.015)
        assert _bias_factor("DTIMS", cfg) == pytest.approx(1.015)

    def test_tims(self) -> None:
        cfg = CCSFilterConfig(twims_bias_tims=1.010)
        assert _bias_factor("TIMS", cfg) == pytest.approx(1.010)

    def test_twims_no_correction(self) -> None:
        cfg = CCSFilterConfig()
        assert _bias_factor("TWIMS", cfg) == 1.0
        assert _bias_factor("Unknown", cfg) == 1.0


# ============================================================
# End-to-end filter behaviour
# ============================================================
class TestRunFilter:
    """Exercise the full run() with synthetic features + ident + reference."""

    def _audit(self) -> AuditLogger:
        return AuditLogger(stage="test_08", params={})

    def test_passthrough_no_candidate(self) -> None:
        """Feature with measured_ccs but no identification → pass-through."""
        features = _make_features({"feat_no_cand": 150.0})
        ident = _make_identification([])  # empty
        reference = _make_reference([])

        res = run(
            features=features,
            identification=ident,
            reference=reference,
            kept_features=["feat_no_cand"],
            config=CCSFilterConfig(),
            audit=self._audit(),
        )
        assert res.n_kept == 1
        assert res.n_passthrough_no_candidate == 1
        assert res.n_passthrough_no_reference == 0
        assert res.n_dropped == 0

    def test_passthrough_no_reference(self) -> None:
        """Feature with candidate but no entry in reference → pass-through."""
        features = _make_features({"feat_no_ref": 150.0})
        ident = _make_identification(
            [
                {
                    "feature_id": "feat_no_ref",
                    "compound_id": "HMDB9999999",
                    "adducts": "M+H",
                    "score": 50.0,
                    "mass_error_ppm": 1.0,
                },
            ]
        )
        reference = _make_reference([])  # empty — no ref entry exists

        res = run(
            features=features,
            identification=ident,
            reference=reference,
            kept_features=["feat_no_ref"],
            config=CCSFilterConfig(),
            audit=self._audit(),
        )
        assert res.n_kept == 1
        assert res.n_passthrough_no_reference == 1
        assert res.n_dropped == 0

    def test_kept_when_ccs_matches(self) -> None:
        """Measured CCS within threshold of corrected reference → keep."""
        # Reference DTIMS 142.0 → corrected (×1.015) = 144.13.
        # Measured 144.5 → err = |144.5 - 144.13|/144.13 ≈ 0.26 % < 2 %.
        features = _make_features({"glucose_feat": 144.5})
        ident = _make_identification(
            [
                {
                    "feature_id": "glucose_feat",
                    "compound_id": "HMDB0000122",
                    "adducts": "M+H",
                    "score": 65.0,
                    "mass_error_ppm": 0.8,
                },
            ]
        )
        reference = _make_reference(
            [
                {
                    "hmdb_id": "HMDB0000122",
                    "adduct": "M+H",
                    "reference_ccs": 142.0,
                    "instrument_type": "DTIMS",
                },
            ]
        )

        res = run(
            features=features,
            identification=ident,
            reference=reference,
            kept_features=["glucose_feat"],
            config=CCSFilterConfig(max_ccs_err_pct=2.0),
            audit=self._audit(),
        )
        assert res.n_kept == 1
        assert res.metrics.iloc[0]["decision"] == "kept"
        assert res.metrics.iloc[0]["best_ccs_err_pct"] < 1.0

    def test_dropped_when_ccs_far(self) -> None:
        """Measured CCS far from corrected reference → drop."""
        features = _make_features({"bad_feat": 200.0})
        ident = _make_identification(
            [
                {
                    "feature_id": "bad_feat",
                    "compound_id": "HMDB0000001",
                    "adducts": "M+H",
                    "score": 70.0,
                    "mass_error_ppm": 0.5,
                },
            ]
        )
        reference = _make_reference(
            [
                {
                    "hmdb_id": "HMDB0000001",
                    "adduct": "M+H",
                    "reference_ccs": 142.0,
                    "instrument_type": "DTIMS",
                },
            ]
        )
        res = run(
            features=features,
            identification=ident,
            reference=reference,
            kept_features=["bad_feat"],
            config=CCSFilterConfig(max_ccs_err_pct=2.0),
            audit=self._audit(),
        )
        assert res.n_kept == 0
        assert res.n_dropped == 1
        assert res.metrics.iloc[0]["decision"] == "dropped_ccs_err"

    def test_quality_floor_blocks_low_score_rescue(self) -> None:
        """A coincidentally-CCS-matching candidate with low Progenesis score must NOT rescue."""
        features = _make_features({"feat": 144.5})
        ident = _make_identification(
            [
                # Junk candidate with low score but coincidental CCS match
                {
                    "feature_id": "feat",
                    "compound_id": "HMDB0000122",
                    "adducts": "M+H",
                    "score": 15.0,
                    "mass_error_ppm": 0.5,
                },
            ]
        )
        reference = _make_reference(
            [
                {
                    "hmdb_id": "HMDB0000122",
                    "adduct": "M+H",
                    "reference_ccs": 142.0,
                    "instrument_type": "DTIMS",
                },
            ]
        )
        res = run(
            features=features,
            identification=ident,
            reference=reference,
            kept_features=["feat"],
            config=CCSFilterConfig(min_progenesis_score=30.0),
            audit=self._audit(),
        )
        assert res.n_kept == 0
        assert res.metrics.iloc[0]["decision"] == "dropped_quality_floor"

    def test_ppm_floor_blocks_high_mass_error_rescue(self) -> None:
        """Candidate with terrible ppm must not rescue."""
        features = _make_features({"feat": 144.5})
        ident = _make_identification(
            [
                {
                    "feature_id": "feat",
                    "compound_id": "HMDB0000122",
                    "adducts": "M+H",
                    "score": 55.0,
                    "mass_error_ppm": 12.0,
                },  # 12 ppm too high
            ]
        )
        reference = _make_reference(
            [
                {
                    "hmdb_id": "HMDB0000122",
                    "adduct": "M+H",
                    "reference_ccs": 142.0,
                    "instrument_type": "DTIMS",
                },
            ]
        )
        res = run(
            features=features,
            identification=ident,
            reference=reference,
            kept_features=["feat"],
            config=CCSFilterConfig(max_mass_error_ppm=5.0),
            audit=self._audit(),
        )
        assert res.n_kept == 0
        assert res.metrics.iloc[0]["decision"] == "dropped_quality_floor"

    def test_any_candidate_passes_rescues_feature(self) -> None:
        """One good candidate (out of many) is enough to keep the feature."""
        features = _make_features({"feat": 144.5})
        ident = _make_identification(
            [
                # Three candidates, only the third has a matching adduct/CCS
                {
                    "feature_id": "feat",
                    "compound_id": "HMDB000A",
                    "adducts": "M+H",
                    "score": 50.0,
                    "mass_error_ppm": 0.5,
                },
                {
                    "feature_id": "feat",
                    "compound_id": "HMDB000B",
                    "adducts": "M+H",
                    "score": 55.0,
                    "mass_error_ppm": 0.5,
                },
                {
                    "feature_id": "feat",
                    "compound_id": "HMDB0000122",
                    "adducts": "M+H",
                    "score": 60.0,
                    "mass_error_ppm": 0.5,
                },
            ]
        )
        reference = _make_reference(
            [
                # Wrong CCS for first two
                {
                    "hmdb_id": "HMDB000A",
                    "adduct": "M+H",
                    "reference_ccs": 100.0,
                    "instrument_type": "DTIMS",
                },
                {
                    "hmdb_id": "HMDB000B",
                    "adduct": "M+H",
                    "reference_ccs": 200.0,
                    "instrument_type": "DTIMS",
                },
                # Right CCS for the third
                {
                    "hmdb_id": "HMDB0000122",
                    "adduct": "M+H",
                    "reference_ccs": 142.0,
                    "instrument_type": "DTIMS",
                },
            ]
        )
        res = run(
            features=features,
            identification=ident,
            reference=reference,
            kept_features=["feat"],
            config=CCSFilterConfig(max_ccs_err_pct=2.0),
            audit=self._audit(),
        )
        assert res.n_kept == 1
        assert res.metrics.iloc[0]["best_compound_id"] == "HMDB0000122"

    def test_adduct_specific_match_required(self) -> None:
        """A reference entry with the wrong adduct must not be used to rescue."""
        features = _make_features({"feat": 144.5})
        ident = _make_identification(
            [
                {
                    "feature_id": "feat",
                    "compound_id": "HMDB0000122",
                    "adducts": "M+H",
                    "score": 60.0,
                    "mass_error_ppm": 0.5,
                },
            ]
        )
        # Reference only has [M+Na] for this compound, NOT [M+H]
        reference = _make_reference(
            [
                {
                    "hmdb_id": "HMDB0000122",
                    "adduct": "M+Na",
                    "reference_ccs": 144.0,
                    "instrument_type": "DTIMS",
                },
            ]
        )
        res = run(
            features=features,
            identification=ident,
            reference=reference,
            kept_features=["feat"],
            config=CCSFilterConfig(),
            audit=self._audit(),
        )
        # No M+H reference exists → pass-through, not kept-by-CCS, not dropped
        assert res.n_passthrough_no_reference == 1
        assert res.n_dropped == 0

    def test_metrics_partition_is_exhaustive(self) -> None:
        """Every input feature lands in exactly one decision bucket."""
        features = _make_features(
            {
                "kept_feat": 144.5,
                "dropped_feat": 200.0,
                "no_cand": 150.0,
                "no_ref": 160.0,
            }
        )
        ident = _make_identification(
            [
                {
                    "feature_id": "kept_feat",
                    "compound_id": "HMDB000A",
                    "adducts": "M+H",
                    "score": 60.0,
                    "mass_error_ppm": 0.5,
                },
                {
                    "feature_id": "dropped_feat",
                    "compound_id": "HMDB000B",
                    "adducts": "M+H",
                    "score": 60.0,
                    "mass_error_ppm": 0.5,
                },
                {
                    "feature_id": "no_ref",
                    "compound_id": "HMDB000C",
                    "adducts": "M+H",
                    "score": 60.0,
                    "mass_error_ppm": 0.5,
                },
            ]
        )
        reference = _make_reference(
            [
                {
                    "hmdb_id": "HMDB000A",
                    "adduct": "M+H",
                    "reference_ccs": 142.0,
                    "instrument_type": "DTIMS",
                },
                {
                    "hmdb_id": "HMDB000B",
                    "adduct": "M+H",
                    "reference_ccs": 142.0,
                    "instrument_type": "DTIMS",
                },
            ]
        )
        res = run(
            features=features,
            identification=ident,
            reference=reference,
            kept_features=["kept_feat", "dropped_feat", "no_cand", "no_ref"],
            config=CCSFilterConfig(),
            audit=self._audit(),
        )
        # 1 kept + 1 dropped + 2 pass-through = 4
        assert res.n_kept + res.n_dropped == 4
        assert res.n_passthrough_no_candidate == 1
        assert res.n_passthrough_no_reference == 1
        # Each metric row has exactly one of: kept, dropped_ccs_err, dropped_quality_floor,
        # passthrough_no_candidate, passthrough_no_reference
        valid = {
            "kept",
            "dropped_ccs_err",
            "dropped_quality_floor",
            "passthrough_no_candidate",
            "passthrough_no_reference",
        }
        assert set(res.metrics["decision"].unique()).issubset(valid)
        assert res.metrics["decision"].notna().all()
