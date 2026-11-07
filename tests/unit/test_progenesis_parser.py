"""Unit tests for the Progenesis CSV parser."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from biodisc.audit import AuditLogger
from biodisc.io.progenesis import (
    ParseConfig,
    ProgenesisParser,
    _is_outer_wrapped_line,
    _parse_acq_date,
    _unwrap_outer_quoted,
    decode_sample_id,
    read_progenesis_csv,
)

DEFAULT_SAMPLE_PATTERN = (
    r"^(?P<acq>\d{8})_(?P<group>[A-Za-z0-9]+)(?:_(?P<dilution>\d+))?_(?P<rep>\d+)$"
)


# ================================================================
# Outer-quote handling
# ================================================================
class TestOuterQuoteHandling:
    def test_detects_progenesis_wrapped_line(self) -> None:
        line = '"11.04_665n,""665.31"",""648.31"",""1""\n'
        assert _is_outer_wrapped_line(line)

    def test_ignores_normal_csv_line(self) -> None:
        line = "11.04_665n,665.31,648.31,1\n"
        assert not _is_outer_wrapped_line(line)

    def test_ignores_unquoted_but_comma_containing_line(self) -> None:
        line = "Compound ID,Formula,Score\n"
        assert not _is_outer_wrapped_line(line)

    def test_ignores_short_quoted_string(self) -> None:
        line = '""\n'
        assert not _is_outer_wrapped_line(line)

    def test_unwraps_wrapped_line_round_trip(self) -> None:
        wrapped = '"11.04_665n,""665.31"",""648.31"",""1""\n'
        got = _unwrap_outer_quoted(wrapped)
        assert got == '11.04_665n,"665.31","648.31","1"\n'

    def test_unwrap_leaves_unwrapped_line_alone(self) -> None:
        line = "Compound,Neutral mass,m/z\n"
        assert _unwrap_outer_quoted(line) == line


# ================================================================
# Sample-id decoding
# ================================================================
class TestDecodeSampleId:
    def test_biological(self) -> None:
        d = decode_sample_id("09012026_MCF7_1", pattern=DEFAULT_SAMPLE_PATTERN)
        assert d.injection_type == "biological"
        assert d.cell_line == "MCF7"
        assert d.replicate == 1
        assert d.sample_id == "MCF7_1"
        assert d.acq_date.year == 2026
        assert d.acq_date.month == 1
        assert d.acq_date.day == 9

    def test_qc(self) -> None:
        d = decode_sample_id("09012026_QC_3", pattern=DEFAULT_SAMPLE_PATTERN)
        assert d.injection_type == "qc"
        assert d.cell_line is None
        assert d.replicate == 3
        assert d.sample_id == "QC_3"

    def test_blank(self) -> None:
        d = decode_sample_id("09012026_Blank_2", pattern=DEFAULT_SAMPLE_PATTERN)
        assert d.injection_type == "blank"
        assert d.cell_line is None
        assert d.replicate == 2
        assert d.sample_id == "Blank_2"

    def test_dilution_half(self) -> None:
        d = decode_sample_id("09012026_QC_1_2", pattern=DEFAULT_SAMPLE_PATTERN)
        assert d.injection_type == "dilution"
        assert d.dilution_level == 2.0
        assert d.sample_id == "QC_1_2"

    def test_dilution_eighth(self) -> None:
        d = decode_sample_id("09012026_QC_1_8", pattern=DEFAULT_SAMPLE_PATTERN)
        assert d.injection_type == "dilution"
        assert d.dilution_level == 8.0

    def test_malformed_raises(self) -> None:
        with pytest.raises(ValueError):
            decode_sample_id("totally_bogus", pattern=DEFAULT_SAMPLE_PATTERN)


class TestParseAcqDate:
    def test_day_month_year(self) -> None:
        d = _parse_acq_date("09012026")
        assert d.year == 2026 and d.month == 1 and d.day == 9

    def test_malformed_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_acq_date("2026-01-09")


# ================================================================
# End-to-end with a synthetic Progenesis CSV
# ================================================================
@pytest.fixture
def synthetic_measurement_csv(tmp_path: Path) -> Path:
    """A 3-feature × 3-sample Progenesis-format measurement CSV.

    Triple-single-quoted to avoid conflict with the trailing ``\"\"\"`` inside
    each Progenesis-wrapped data row.
    """
    csv_body = textwrap.dedent(
        '''\
        ,,,,,,,,,,,,,,,"Normalised abundance",,,"Raw abundance",,,
        ,,,,,,,,,,,,,,,"MCF7",,,"MCF7",,,
        "Compound","Neutral mass (Da)","m/z","Charge","Retention time (min)","CCS (angstrom^2)","Chromatographic peak width (min)","Identifications","Anova (p)","Max Fold Change","Highest Mean","Lowest Mean","Isotope Distribution","Maximum Abundance","Minimum CV%","09012026_MCF7_1","09012026_MCF7_2","09012026_MCF7_3","09012026_MCF7_1","09012026_MCF7_2","09012026_MCF7_3"
        "11.04_665n,""665.31"",""648.31"",""1"",""11.04"",""257.15"",""0.26"",""2"",""0.00001"",""2.4"",""MCF7"",""MCF7"",""100"",""70000"",""14.0"",""22000"",""26000"",""21000"",""42000"",""42900"",""42100"""
        "13.22_314n,""314.25"",""279.23"",""1"",""13.22"",""189.62"",""0.31"",""6"",""0.005"",""1.3"",""MCF7"",""MCF7"",""100"",""14000"",""7.8"",""13000"",""14000"",""14000"",""15000"",""15500"",""16000"""
        "11.51_366n,""366.13"",""367.14"",""1"",""11.51"",""193.65"",""0.29"",""2"",""0.0001"",""4.2"",""MCF7"",""MCF7"",""100"",""60000"",""9.5"",""45000"",""48000"",""44000"",""52000"",""53000"",""51800"""
        '''
    )
    path = tmp_path / "MEASUREMENT.csv"
    path.write_text(csv_body)
    return path


def test_read_progenesis_csv_unwraps(synthetic_measurement_csv: Path) -> None:
    df, banners = read_progenesis_csv(synthetic_measurement_csv, header_rows=3, banner_row_count=2)
    assert df.shape[0] == 3  # 3 features
    assert df["Compound"].tolist() == ["11.04_665n", "13.22_314n", "11.51_366n"]
    assert df["Neutral mass (Da)"].iloc[0] == pytest.approx(665.31)
    # banner row 0 should contain "Normalised abundance" somewhere
    assert any("Normalised abundance" in str(x) for x in banners[0])


def test_end_to_end_parse(tmp_path: Path, synthetic_measurement_csv: Path) -> None:
    """Test parse_all end-to-end with the synthetic measurement file only.

    Blank/dilution/identification paths point at the same file as a stand-in;
    only the measurement parser is exercised here (integration test covers the
    others).
    """
    config = ParseConfig(
        measurement_file=synthetic_measurement_csv.name,
        blank_file=synthetic_measurement_csv.name,
        dilution_file=synthetic_measurement_csv.name,
        identification_file=synthetic_measurement_csv.name,
        sample_id_pattern=DEFAULT_SAMPLE_PATTERN,
    )
    audit = AuditLogger(stage="01_parse_test", params={})
    parser = ProgenesisParser(config, audit)

    # Monkey-patch out blank/dilution/ident parsers since the fixture isn't
    # shaped for them. Only exercise _parse_measurement.
    features, samples, abund_raw, abund_norm = parser._parse_measurement(synthetic_measurement_csv)
    assert len(features) == 3
    assert set(samples["sample_id"]) == {"MCF7_1", "MCF7_2", "MCF7_3"}
    assert len(abund_raw) == 9  # 3 features × 3 samples
    assert len(abund_norm) == 9

    # Spot-check values
    raw_11_04 = abund_raw[
        (abund_raw["feature_id"] == "11.04_665n") & (abund_raw["sample_id"] == "MCF7_1")
    ]
    assert raw_11_04["intensity"].iloc[0] == pytest.approx(42000)

    norm_11_04 = abund_norm[
        (abund_norm["feature_id"] == "11.04_665n") & (abund_norm["sample_id"] == "MCF7_1")
    ]
    assert norm_11_04["intensity"].iloc[0] == pytest.approx(22000)


# ================================================================
# Sample exclusions (parse-time)
# ================================================================
class TestSampleExclusions:
    """Tests for ``ProgenesisParser._apply_sample_exclusions``.

    Verifies that excluded sample IDs are dropped from samples + abundance
    tables (long and wide), with corresponding filtering of blanks_long
    and dilution_ladder_long via their typed-replicate / dilution-level
    columns. Both decoded (``QC_5``) and raw (``09012026_QC_5``) ids are
    accepted; unknown ids log warnings without failing.
    """

    @staticmethod
    def _build_inputs() -> dict[str, pd.DataFrame]:
        """Synthetic samples + abundance + blanks + dilution tables."""
        from datetime import datetime as _dt

        samples = pd.DataFrame(
            {
                "sample_id": ["MCF7_1", "QC_1", "QC_2", "Blank_1", "QC_1_2"],
                "raw_sample_id": [
                    "09012026_MCF7_1",
                    "09012026_QC_1",
                    "09012026_QC_2",
                    "09012026_Blank_1",
                    "09012026_QC_1_2",
                ],
                "injection_type": ["biological", "qc", "qc", "blank", "dilution"],
                "cell_line": ["MCF7", None, None, None, None],
                "replicate": [1, 1, 2, 1, 1],
                "dilution_level": [None, None, None, None, 2.0],
                "acq_date": [_dt(2026, 1, 9)] * 5,
                "column_order": [0, 1, 2, 3, 4],
            }
        )
        abund = pd.DataFrame(
            {
                "feature_id": ["F1"] * 3,
                "sample_id": ["MCF7_1", "QC_1", "QC_2"],
                "intensity": [100.0, 200.0, 300.0],
            }
        )
        blanks = pd.DataFrame(
            {
                "feature_id": ["F1"],
                "blank_replicate": [1],
                "block": ["raw"],
                "intensity": [10.0],
            }
        )
        dilution = pd.DataFrame(
            {
                "feature_id": ["F1"],
                "dilution_level": [2.0],
                "dilution_factor": [0.5],
                "block": ["raw"],
                "intensity": [50.0],
            }
        )
        return {
            "samples": samples,
            "abund_raw": abund.copy(),
            "abund_norm": abund.copy(),
            "blanks": blanks,
            "dilution": dilution,
        }

    @staticmethod
    def _make_parser() -> ProgenesisParser:
        cfg = ParseConfig(
            measurement_file="m",
            blank_file="b",
            dilution_file="d",
            identification_file="i",
            sample_id_pattern=DEFAULT_SAMPLE_PATTERN,
        )
        audit = AuditLogger(stage="01_parse_test", params={})
        return ProgenesisParser(cfg, audit)

    def test_empty_exclusions_changes_nothing(self) -> None:
        inputs = self._build_inputs()
        parser = self._make_parser()
        s, ar, _an, b, d = parser._apply_sample_exclusions(
            samples_df=inputs["samples"],
            abund_raw=inputs["abund_raw"],
            abund_norm=inputs["abund_norm"],
            blanks_df=inputs["blanks"],
            dilution_df=inputs["dilution"],
            requested=(),
        )
        # Counts unchanged across every output table
        assert len(s) == 5
        assert len(ar) == 3
        assert len(b) == 1
        assert len(d) == 1
        # _an is the abundances_norm output; we just check it's a DataFrame
        assert _an.shape[0] == 3

    def test_exclude_qc_by_decoded_id(self) -> None:
        inputs = self._build_inputs()
        parser = self._make_parser()
        s, ar, an, b, d = parser._apply_sample_exclusions(
            samples_df=inputs["samples"],
            abund_raw=inputs["abund_raw"],
            abund_norm=inputs["abund_norm"],
            blanks_df=inputs["blanks"],
            dilution_df=inputs["dilution"],
            requested=("QC_1",),
        )
        # QC_1 dropped from samples
        assert "QC_1" not in s["sample_id"].values
        assert len(s) == 4
        # And from abundances (both raw and normalised blocks)
        assert "QC_1" not in ar["sample_id"].values
        assert "QC_1" not in an["sample_id"].values
        # Blanks + dilution unchanged (different injection types)
        assert len(b) == 1
        assert len(d) == 1
        # column_order renumbered contiguous
        assert list(s["column_order"]) == [0, 1, 2, 3]

    def test_exclude_qc_by_raw_id(self) -> None:
        inputs = self._build_inputs()
        parser = self._make_parser()
        s, ar, _an, _b, _d = parser._apply_sample_exclusions(
            samples_df=inputs["samples"],
            abund_raw=inputs["abund_raw"],
            abund_norm=inputs["abund_norm"],
            blanks_df=inputs["blanks"],
            dilution_df=inputs["dilution"],
            requested=("09012026_QC_2",),
        )
        # Raw form should match the same row
        assert "QC_2" not in s["sample_id"].values
        assert "QC_2" not in ar["sample_id"].values

    def test_exclude_unknown_id_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        inputs = self._build_inputs()
        parser = self._make_parser()
        s, ar, _an, _b, _d = parser._apply_sample_exclusions(
            samples_df=inputs["samples"],
            abund_raw=inputs["abund_raw"],
            abund_norm=inputs["abund_norm"],
            blanks_df=inputs["blanks"],
            dilution_df=inputs["dilution"],
            requested=("NOT_A_REAL_SAMPLE",),
        )
        # Nothing dropped, no exception
        assert len(s) == 5
        assert len(ar) == 3

    def test_exclude_blank_filters_blanks_long(self) -> None:
        inputs = self._build_inputs()
        parser = self._make_parser()
        s, _ar, _an, b, _d = parser._apply_sample_exclusions(
            samples_df=inputs["samples"],
            abund_raw=inputs["abund_raw"],
            abund_norm=inputs["abund_norm"],
            blanks_df=inputs["blanks"],
            dilution_df=inputs["dilution"],
            requested=("Blank_1",),
        )
        # Blank_1 gone from samples
        assert "Blank_1" not in s["sample_id"].values
        # blanks_long filtered out replicate 1
        assert len(b) == 0

    def test_exclude_dilution_filters_dilution_long(self) -> None:
        inputs = self._build_inputs()
        parser = self._make_parser()
        s, _ar, _an, _b, d = parser._apply_sample_exclusions(
            samples_df=inputs["samples"],
            abund_raw=inputs["abund_raw"],
            abund_norm=inputs["abund_norm"],
            blanks_df=inputs["blanks"],
            dilution_df=inputs["dilution"],
            requested=("QC_1_2",),
        )
        assert "QC_1_2" not in s["sample_id"].values
        # dilution_long filtered to remove level 2.0
        assert 2.0 not in d["dilution_level"].astype(float).values

    def test_exclude_multiple_samples(self) -> None:
        inputs = self._build_inputs()
        parser = self._make_parser()
        s, ar, _an, b, d = parser._apply_sample_exclusions(
            samples_df=inputs["samples"],
            abund_raw=inputs["abund_raw"],
            abund_norm=inputs["abund_norm"],
            blanks_df=inputs["blanks"],
            dilution_df=inputs["dilution"],
            requested=("QC_1", "QC_2", "Blank_1"),
        )
        assert len(s) == 2  # MCF7_1 + QC_1_2 remain
        assert set(s["sample_id"]) == {"MCF7_1", "QC_1_2"}
        assert len(ar) == 1  # only MCF7_1 row remains in abundances
        assert len(b) == 0  # blank filtered
        assert len(d) == 1  # dilution untouched

    def test_parse_config_accepts_exclude_samples_tuple(self) -> None:
        cfg = ParseConfig(
            measurement_file="m",
            blank_file="b",
            dilution_file="d",
            identification_file="i",
            sample_id_pattern=DEFAULT_SAMPLE_PATTERN,
            exclude_samples=("QC_5", "Blank_2"),
        )
        # ParseConfig is frozen — must remain hashable so it can be compared
        # in tests / used as dict key in the future.
        assert hash(cfg) is not None
        assert cfg.exclude_samples == ("QC_5", "Blank_2")


# ================================================================
# Audit logger
# ================================================================
class TestAuditLogger:
    def test_kept_and_dropped_roundtrip(self, tmp_path: Path) -> None:
        audit = AuditLogger(stage="testst", params={"threshold": 5})
        audit.kept(feature_ids=["a", "b"], reason="ok")
        audit.dropped(feature_ids=["c"], reason="bad", metric=1.2, threshold=5.0)
        df = audit.to_frame()
        assert len(df) == 3
        assert set(df["action"]) == {"kept", "dropped"}
        assert df[df["feature_id"] == "c"]["metric"].iloc[0] == pytest.approx(1.2)

    def test_flush_writes_parquet(self, tmp_path: Path) -> None:
        audit = AuditLogger(stage="teststage", params={"p": 1})
        audit.kept(feature_ids=["f1"], reason="ok")
        path = audit.flush(output_dir=tmp_path, emit_csv_mirror=True)
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) == 1
        assert df["stage"].iloc[0] == "teststage"
        # CSV mirror
        assert (tmp_path / "teststage.csv").exists()

    def test_params_hash_stable(self) -> None:
        a1 = AuditLogger(stage="s", params={"a": 1, "b": 2})
        a2 = AuditLogger(stage="s", params={"b": 2, "a": 1})
        assert a1._params_hash == a2._params_hash

    def test_params_hash_changes_with_content(self) -> None:
        a1 = AuditLogger(stage="s", params={"threshold": 5})
        a2 = AuditLogger(stage="s", params={"threshold": 10})
        assert a1._params_hash != a2._params_hash
