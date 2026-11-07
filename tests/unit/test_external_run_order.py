"""Unit tests for external/run_order.py — Waters _HEADER.TXT → run-order parquet."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from biodisc.external.run_order import (
    HeaderRecord,
    RunOrderConfig,
    _dedupe_by_datetime,
    _filter_to_samples,
    build,
    parse_all_headers,
    parse_header_file,
)


# ============================================================
# Helpers
# ============================================================
def _write_header(
    path: Path,
    *,
    name: str,
    date: str,
    time: str,
) -> None:
    """Emit a minimal _HEADER.TXT with just the three lines we parse."""
    path.write_text(
        f"$$ Version: 01.00\n"
        f"$$ Acquired Name: {name}\n"
        f"$$ Acquired Date: {date}\n"
        f"$$ Acquired Time: {time}\n"
        f"$$ Other Field: not_parsed\n"
    )


# ============================================================
# Single-file parser
# ============================================================
class TestParseHeaderFile:
    def test_minimal_header(self, tmp_path: Path) -> None:
        p = tmp_path / "_HEADER.TXT"
        _write_header(p, name="09012026_QC_5", date="10-Jan-2026", time="00:56:59")
        rec = parse_header_file(p)
        assert isinstance(rec, HeaderRecord)
        assert rec.acquired_name == "09012026_QC_5"
        assert rec.acq_datetime == datetime(2026, 1, 10, 0, 56, 59)
        assert rec.file == "_HEADER.TXT"

    def test_missing_field_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "_HEADER.TXT"
        p.write_text("$$ Version: 01.00\n$$ Acquired Name: X\n$$ Acquired Time: 12:00:00\n")
        with pytest.raises(ValueError, match="missing Acquired"):
            parse_header_file(p)

    def test_malformed_date_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "_HEADER.TXT"
        _write_header(p, name="X", date="2026-01-10", time="00:56:59")  # ISO format, not Waters
        with pytest.raises(ValueError, match="cannot parse acq date/time"):
            parse_header_file(p)


# ============================================================
# Directory scan
# ============================================================
class TestParseAllHeaders:
    def test_scans_directory(self, tmp_path: Path) -> None:
        for i, t in enumerate(["00:00:00", "00:01:00", "00:02:00"]):
            _write_header(
                tmp_path / f"_HEADER({i}).TXT",
                name=f"sample_{i}",
                date="09-Jan-2026",
                time=t,
            )
        df = parse_all_headers(tmp_path)
        assert len(df) == 3
        assert set(df["acquired_name"]) == {"sample_0", "sample_1", "sample_2"}

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_all_headers(tmp_path / "nope")

    def test_no_files_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No header files"):
            parse_all_headers(tmp_path)

    def test_skips_unparseable_files(self, tmp_path: Path) -> None:
        _write_header(tmp_path / "_HEADER(1).TXT", name="X", date="09-Jan-2026", time="00:00:00")
        # Bad file: missing required field
        (tmp_path / "_HEADER(2).TXT").write_text("$$ Version: 01.00\n")
        df = parse_all_headers(tmp_path)
        assert len(df) == 1


# ============================================================
# Dedup helper
# ============================================================
class TestDedupe:
    def test_keeps_first_of_duplicate_pair(self) -> None:
        df = pd.DataFrame(
            {
                "file": ["a.TXT", "b.TXT", "c.TXT"],
                "acquired_name": ["X", "X", "Y"],
                "acq_datetime": [
                    datetime(2026, 1, 9, 12),
                    datetime(2026, 1, 9, 12),  # duplicate
                    datetime(2026, 1, 9, 13),
                ],
            }
        )
        out = _dedupe_by_datetime(df)
        assert len(out) == 2
        # First instance kept
        assert out.iloc[0]["file"] == "a.TXT"

    def test_no_duplicates_unchanged(self) -> None:
        df = pd.DataFrame(
            {
                "file": ["a.TXT", "b.TXT"],
                "acquired_name": ["X", "Y"],
                "acq_datetime": [datetime(2026, 1, 9, 12), datetime(2026, 1, 9, 13)],
            }
        )
        out = _dedupe_by_datetime(df)
        assert len(out) == 2


# ============================================================
# Filter against samples
# ============================================================
class TestFilterToSamples:
    def test_drops_unknown_names(self) -> None:
        df = pd.DataFrame(
            {
                "file": ["a", "b", "c"],
                "acquired_name": ["sample_A", "Blank_0", "sample_B"],
                "acq_datetime": [
                    datetime(2026, 1, 9, 12),
                    datetime(2026, 1, 9, 13),
                    datetime(2026, 1, 9, 14),
                ],
            }
        )
        out, excluded = _filter_to_samples(df, ["sample_A", "sample_B"])
        assert set(out["acquired_name"]) == {"sample_A", "sample_B"}
        assert excluded == {"Blank_0"}


# ============================================================
# End-to-end build
# ============================================================
class TestBuild:
    def _samples_parquet(self, tmp: Path, raw_ids: list[str]) -> Path:
        # Build a minimal samples.parquet that satisfies SamplesSchema-relevant cols
        df = pd.DataFrame(
            {
                "sample_id": raw_ids,
                "raw_sample_id": raw_ids,
                "injection_type": ["biological"] * len(raw_ids),
                "cell_line": ["X"] * len(raw_ids),
                "replicate": [1] * len(raw_ids),
                "dilution_level": [None] * len(raw_ids),
                "acq_date": [datetime(2026, 1, 9)] * len(raw_ids),
                "column_order": list(range(len(raw_ids))),
            }
        )
        p = tmp / "samples.parquet"
        df.to_parquet(p)
        return p

    def test_end_to_end_build_writes_parquet(self, tmp_path: Path) -> None:
        # 3 headers, all valid samples, no extras, no duplicates
        for i, (n, t) in enumerate(
            [("S_A", "12:00:00"), ("S_B", "13:00:00"), ("S_C", "14:00:00")]
        ):
            _write_header(
                tmp_path / f"_HEADER({i}).TXT", name=n, date="09-Jan-2026", time=t
            )
        samples_path = self._samples_parquet(tmp_path, ["S_A", "S_B", "S_C"])
        out_path = tmp_path / "run_order.parquet"

        cfg = RunOrderConfig(
            headers_dir=tmp_path,
            output_parquet=out_path,
            samples_parquet=samples_path,
        )
        df = build(cfg)
        assert out_path.exists()
        # Position contiguous 1..3, ordered by datetime
        assert list(df["position"]) == [1, 2, 3]
        assert list(df["raw_sample_id"]) == ["S_A", "S_B", "S_C"]

    def test_drops_pre_acquisition_blank(self, tmp_path: Path) -> None:
        # 1 extra header (Blank_0 not in samples) → should be dropped
        _write_header(tmp_path / "_HEADER(0).TXT", name="Blank_0", date="09-Jan-2026", time="11:00:00")
        _write_header(tmp_path / "_HEADER(1).TXT", name="S_A", date="09-Jan-2026", time="12:00:00")
        samples_path = self._samples_parquet(tmp_path, ["S_A"])
        out_path = tmp_path / "run_order.parquet"

        df = build(
            RunOrderConfig(
                headers_dir=tmp_path,
                output_parquet=out_path,
                samples_parquet=samples_path,
            )
        )
        assert len(df) == 1
        assert df["raw_sample_id"].iloc[0] == "S_A"
        assert df["position"].iloc[0] == 1

    def test_fail_on_missing_sample(self, tmp_path: Path) -> None:
        # samples claims S_B exists but no header for it → should raise
        _write_header(tmp_path / "_HEADER(0).TXT", name="S_A", date="09-Jan-2026", time="12:00:00")
        samples_path = self._samples_parquet(tmp_path, ["S_A", "S_B"])
        out_path = tmp_path / "run_order.parquet"

        with pytest.raises(ValueError, match="have NO _HEADER"):
            build(
                RunOrderConfig(
                    headers_dir=tmp_path,
                    output_parquet=out_path,
                    samples_parquet=samples_path,
                    fail_on_missing_samples=True,
                )
            )

    def test_warning_on_missing_sample_when_not_failing(self, tmp_path: Path) -> None:
        _write_header(tmp_path / "_HEADER(0).TXT", name="S_A", date="09-Jan-2026", time="12:00:00")
        samples_path = self._samples_parquet(tmp_path, ["S_A", "S_B"])
        out_path = tmp_path / "run_order.parquet"

        # Should NOT raise when fail_on_missing_samples=False
        df = build(
            RunOrderConfig(
                headers_dir=tmp_path,
                output_parquet=out_path,
                samples_parquet=samples_path,
                fail_on_missing_samples=False,
            )
        )
        assert len(df) == 1

    def test_dedupes_duplicate_headers(self, tmp_path: Path) -> None:
        _write_header(tmp_path / "_HEADER(0).TXT", name="S_A", date="09-Jan-2026", time="12:00:00")
        _write_header(tmp_path / "_HEADER(1).TXT", name="S_A", date="09-Jan-2026", time="12:00:00")  # dup
        _write_header(tmp_path / "_HEADER(2).TXT", name="S_B", date="09-Jan-2026", time="13:00:00")
        samples_path = self._samples_parquet(tmp_path, ["S_A", "S_B"])
        out_path = tmp_path / "run_order.parquet"

        df = build(
            RunOrderConfig(
                headers_dir=tmp_path,
                output_parquet=out_path,
                samples_parquet=samples_path,
            )
        )
        assert len(df) == 2  # dup dropped
        assert list(df["raw_sample_id"]) == ["S_A", "S_B"]

    def test_no_validation_skips_samples_check(self, tmp_path: Path) -> None:
        # No samples_parquet → no validation, all headers retained
        _write_header(tmp_path / "_HEADER(0).TXT", name="X", date="09-Jan-2026", time="12:00:00")
        _write_header(tmp_path / "_HEADER(1).TXT", name="Y", date="09-Jan-2026", time="13:00:00")
        out_path = tmp_path / "run_order.parquet"

        df = build(
            RunOrderConfig(
                headers_dir=tmp_path,
                output_parquet=out_path,
                samples_parquet=None,
            )
        )
        assert len(df) == 2
        assert list(df["position"]) == [1, 2]
