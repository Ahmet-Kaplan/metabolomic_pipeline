"""Progenesis QI CSV parser.

Handles the four files Progenesis QI exports for an untargeted LC-MS dataset:

    1. *_COMPOUND_MEASUREMENT(in).csv   — main feature × sample matrix
    2. Blank(in).csv                    — same matrix + 3 blank injections
    3. dİLUTION(in).csv                 — same matrix + QC dilution series
    4. *_COMPOUND_IDENTIFICATION.csv    — long-format candidate IDs

Known Progenesis quirks we handle:

    - **Outer-quote wrapping**: each data row in the measurement/blank/dilution
      files is wrapped in outer ``"..."`` with inner ``""`` as escape-for-quote.
      The identification file does NOT have this quirk. We auto-detect.
    - **Three-row banner header**: row 0 = section label
      ("Normalised abundance" | "Raw abundance"), row 1 = group label
      (MCF7 | MCF10A | ... | QC | Blank | Dilution), row 2 = column names.
    - **Two abundance blocks**: Normalised and Raw columns repeat per sample.
    - **Turkish ``İ`` in dilution filename** — UTF-8 path handling works natively
      when using pathlib; we do NOT rename files.
    - **Sample ids** like ``09012026_MCF7_1`` or ``09012026_QC_1_2`` follow a
      configurable regex so new datasets with a different prefix still work.

Public API
----------
``ProgenesisParser`` — stateful parser, constructed with a ``ParseConfig``.
``parse_all(input_dir, output_dir)`` — run all four file parsers and write
outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from biodisc.audit import AuditLogger
from biodisc.io.schemas import (
    AbundancesLongSchema,
    BlanksLongSchema,
    DilutionLadderLongSchema,
    FeaturesSchema,
    IdentificationLongSchema,
    SamplesSchema,
)

# Sentinel tokens Progenesis uses for "no value" in otherwise-numeric columns.
# Why: Progenesis renders "---" when a feature didn't pass a per-column quality
# gate (e.g., CV% undefined); empty string means "not computed". Convert to NaN
# before any dtype coercion.
_PROGENESIS_NA_SENTINELS = {"---", "", "—", "N/A", "n/a", "#N/A"}

log = structlog.get_logger(__name__)


# ============================================================
# Config dataclass (mirrors conf/config.yaml:parse)
# ============================================================
@dataclass(frozen=True)
class ParseConfig:
    """Runtime-validated config for the Progenesis parser."""

    measurement_file: str
    blank_file: str
    dilution_file: str
    identification_file: str
    sample_id_pattern: str
    emit_parquet: bool = True
    emit_csv_mirror: bool = True
    emit_wide: bool = True
    emit_long: bool = True
    canonical_abundance: str = "raw"  # or "normalised"
    # Sample IDs to drop at parse time. Accepts both decoded (``QC_5``) and
    # raw (``09012026_QC_5``) forms. Excluded samples are removed from every
    # output table — samples.parquet, abundances, blanks, dilution. The
    # primary use case is dropping multiverse-flagged technical outliers and
    # re-running from Stage 1 so all downstream filters recompute on a
    # cleaner set. Tuple (immutable) so the dataclass stays hashable.
    exclude_samples: tuple[str, ...] = ()


# ============================================================
# Low-level: read Progenesis CSV, transparently handling outer quoting
# ============================================================
def _is_outer_wrapped_line(line: str) -> bool:
    # Progenesis data rows start and end with a literal " and contain the
    # escaped-quote sequence ("" meaning one literal ") in the interior.
    # Minimum length of a meaningful wrapped row is 6. A two-character line of
    # just two double-quotes is a null field, not a wrapped row.
    stripped = line.rstrip("\r\n")
    if len(stripped) < 6 or not (stripped.startswith('"') and stripped.endswith('"')):
        return False
    # The ``""`` escape must appear in the interior of the string, not at the
    # absolute boundary.
    return '""' in stripped[1:-1]


def _unwrap_outer_quoted(line: str) -> str:
    """Strip the Progenesis outer-quote wrapping from one raw data line."""
    stripped = line.rstrip("\r\n")
    if _is_outer_wrapped_line(stripped):
        return stripped[1:-1].replace('""', '"') + "\n"
    return line


def _sniff_outer_quoting(path: Path, data_start_row: int) -> bool:
    """Peek the first data line and decide whether the file is outer-wrapped."""
    with path.open("r", encoding="utf-8", newline="") as f:
        for _ in range(data_start_row):
            f.readline()
        first_data = f.readline()
    return _is_outer_wrapped_line(first_data)


def _scrub_na_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Progenesis sentinel tokens with ``pd.NA`` in every object column."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(list(_PROGENESIS_NA_SENTINELS), np.nan)
    return df


def read_progenesis_csv(
    path: Path,
    *,
    header_rows: int,
    banner_row_count: int = 0,
) -> tuple[pd.DataFrame, list[list[str]]]:
    """Read a Progenesis CSV, returning (body_df, banner_rows).

    Parameters
    ----------
    path
        Path to the CSV.
    header_rows
        Number of rows BEFORE the data header (i.e. rows to skip plus the header
        row). For the measurement/blank/dilution files this is typically 3
        (row 0 = section banner, row 1 = group banner, row 2 = column names).
        For the identification file it is 2 (row 0 = sparse "Tags" marker, row 1
        = column names).
    banner_row_count
        How many of the leading rows are banner rows we want to return separately
        (used for column-level metadata).
    """
    # Collect banner rows raw (these are short, low-cost)
    banners: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for _ in range(banner_row_count):
            raw = f.readline()
            unwrapped = _unwrap_outer_quoted(raw)
            # parse with csv via pandas
            banners.append(
                next(iter(pd.read_csv(StringIO(unwrapped), header=None).values.tolist()))
            )

    # Data body: read with header row at the first non-banner row
    is_wrapped = _sniff_outer_quoting(path, data_start_row=header_rows)

    if is_wrapped:
        # Stream lines, unwrap each, feed to pandas as a single string buffer.
        # For very large files (identification file is 335 MB) this approach
        # would be memory-heavy; however the identification file isn't wrapped,
        # so the wrapped branch only needs to handle the 60 MB measurement files.
        log.debug("reading_outer_wrapped", path=str(path))
        with path.open("r", encoding="utf-8", newline="") as f:
            # Skip banner lines
            for _ in range(header_rows - 1):
                f.readline()
            # Header line is the (header_rows-1 + 1)-th, i.e. still wrapped
            header_line = f.readline()
            header_unwrapped = _unwrap_outer_quoted(header_line)
            # Build a buffer of header + unwrapped data lines
            buf_parts: list[str] = [header_unwrapped]
            for raw in f:
                buf_parts.append(_unwrap_outer_quoted(raw))
        body_text = "".join(buf_parts)
        df = pd.read_csv(StringIO(body_text), low_memory=False)
    else:
        # Unwrapped: pandas reads directly
        log.debug("reading_standard_csv", path=str(path))
        df = pd.read_csv(path, skiprows=header_rows - 1, low_memory=False)

    # Normalise Progenesis' missing-value sentinels before any downstream coercion
    df = _scrub_na_sentinels(df)
    return df, banners


# ============================================================
# Sample-id decoding
# ============================================================
_METADATA_COLS_MEASUREMENT = [
    "Compound",
    "Neutral mass (Da)",
    "m/z",
    "Charge",
    "Retention time (min)",
    "CCS (angstrom^2)",
    "Chromatographic peak width (min)",
    "Identifications",
    "Anova (p)",
    "Max Fold Change",
    "Highest Mean",
    "Lowest Mean",
    "Isotope Distribution",
    "Maximum Abundance",
    "Minimum CV%",
]

_TAIL_COLS = [
    "Minimum CV < 30",
    "Not fragmented",
    "Not identified",
    "Separated by drift time",
    "Accepted Compound ID",
    "Accepted Description",
    "Adducts",
    "Formula",
    "Score",
    "Fragmentation Score",
    "Mass Error (ppm)",
    "Isotope Similarity",
    "Retention Time Error (mins)",
    "dCCS (angstrom^2)",
    "Compound Link",
]


@dataclass(frozen=True)
class DecodedSample:
    """Result of parsing a raw injection id like ``09012026_MCF7_1``."""

    sample_id: str
    raw_sample_id: str
    injection_type: str  # biological | qc | blank | dilution
    cell_line: str | None
    replicate: int
    dilution_level: float | None
    acq_date: datetime


def decode_sample_id(raw: str, *, pattern: str) -> DecodedSample:
    """Parse a raw Progenesis column header into structured metadata.

    Examples
    --------
    >>> decode_sample_id("09012026_MCF7_1", pattern=DEFAULT_PATTERN).injection_type
    'biological'
    >>> decode_sample_id("09012026_QC_3", pattern=DEFAULT_PATTERN).injection_type
    'qc'
    >>> decode_sample_id("09012026_Blank_2", pattern=DEFAULT_PATTERN).injection_type
    'blank'
    >>> decode_sample_id("09012026_QC_1_2", pattern=DEFAULT_PATTERN).dilution_level
    2.0
    """
    # Special-case dilution: raw ends with _1_<N> where N is dilution denominator
    dilution_match = re.match(r"^(?P<acq>\d{8})_QC_1_(?P<dilution>\d+)$", raw)
    if dilution_match:
        g = dilution_match.groupdict()
        return DecodedSample(
            sample_id=f"QC_1_{g['dilution']}",
            raw_sample_id=raw,
            injection_type="dilution",
            cell_line=None,
            replicate=1,
            dilution_level=float(g["dilution"]),
            acq_date=_parse_acq_date(g["acq"]),
        )

    m = re.match(pattern, raw)
    if not m:
        raise ValueError(f"sample id {raw!r} does not match pattern {pattern!r}")
    gd = m.groupdict()
    group = gd["group"]
    rep = int(gd["rep"])
    acq = _parse_acq_date(gd["acq"])

    if group.upper() == "QC":
        return DecodedSample(
            sample_id=f"QC_{rep}",
            raw_sample_id=raw,
            injection_type="qc",
            cell_line=None,
            replicate=rep,
            dilution_level=None,
            acq_date=acq,
        )
    if group.lower() == "blank":
        return DecodedSample(
            sample_id=f"Blank_{rep}",
            raw_sample_id=raw,
            injection_type="blank",
            cell_line=None,
            replicate=rep,
            dilution_level=None,
            acq_date=acq,
        )
    # Biological
    return DecodedSample(
        sample_id=f"{group}_{rep}",
        raw_sample_id=raw,
        injection_type="biological",
        cell_line=group,
        replicate=rep,
        dilution_level=None,
        acq_date=acq,
    )


def _parse_acq_date(yyyymmdd: str) -> datetime:
    """Parse a Progenesis acquisition date prefix.

    Progenesis prefixes with ``DDMMYYYY`` (day-month-year). The sample
    ``09012026`` → 9 January 2026.
    """
    if len(yyyymmdd) != 8 or not yyyymmdd.isdigit():
        raise ValueError(f"acquisition date prefix malformed: {yyyymmdd!r}")
    day = int(yyyymmdd[0:2])
    month = int(yyyymmdd[2:4])
    year = int(yyyymmdd[4:8])
    return datetime(year, month, day)


# ============================================================
# Column-metadata extractor (banner row interpretation)
# ============================================================
@dataclass(frozen=True)
class SampleColumn:
    """A resolved data column pointing at one (sample, abundance block)."""

    dataframe_column: str  # the exact pd.DataFrame column name
    raw_sample_id: str  # stripped of any '.1' etc. suffix
    block: str  # 'raw' or 'normalised'


def resolve_sample_columns(
    df: pd.DataFrame,
    banners: list[list[Any]],
) -> list[SampleColumn]:
    """Walk the 2-banner header and return a list of SampleColumn entries.

    The banner rows come in as raw Python lists. Row 0 marks section boundaries
    (``Normalised abundance`` → block='normalised'; ``Raw abundance`` →
    block='raw'). Row 1 is informative only (group label) — we decode the
    sample id from the column name directly via the regex.
    """
    section_banner = banners[0]
    n = min(len(section_banner), len(df.columns))

    # Walk columns; current_block is set whenever we meet a section banner cell.
    current_block: str | None = None
    resolved: list[SampleColumn] = []
    for i in range(n):
        banner_cell = section_banner[i]
        if isinstance(banner_cell, str) and banner_cell.strip():
            cell = banner_cell.strip().lower()
            if "normalised abundance" in cell or "normalized abundance" in cell:
                current_block = "normalised"
            elif "raw abundance" in cell:
                current_block = "raw"
        col = df.columns[i]
        # Heuristic: sample columns are those matching the Progenesis sample
        # pattern (ddmmyyyy_group_rep...). Metadata and tail columns never do.
        if re.match(r"^\d{8}_", str(col)):
            # Strip pandas '.1', '.2' duplicate-column suffixes that appear
            # when the same header (eg '09012026_MCF7_1') occurs in both blocks.
            raw = re.sub(r"\.\d+$", "", str(col))
            resolved.append(
                SampleColumn(
                    dataframe_column=str(col),
                    raw_sample_id=raw,
                    block=current_block or "raw",
                )
            )
    return resolved


# ============================================================
# Core parser
# ============================================================
class ProgenesisParser:
    """Parse all four Progenesis QI CSV exports into tidy parquet tables."""

    # Canonical output file names under ``output_dir``
    OUT_FEATURES = "features.parquet"
    OUT_SAMPLES = "samples.parquet"
    OUT_ABUND_RAW_LONG = "abundances_raw_long.parquet"
    OUT_ABUND_NORM_LONG = "abundances_normalised_long.parquet"
    OUT_ABUND_RAW_WIDE = "abundances_raw_wide.parquet"
    OUT_ABUND_NORM_WIDE = "abundances_normalised_wide.parquet"
    OUT_IDENT_LONG = "identification_long.parquet"
    OUT_DILUTION = "dilution_ladder_long.parquet"
    OUT_BLANKS = "blanks_long.parquet"

    def __init__(self, config: ParseConfig, audit: AuditLogger) -> None:
        self.config = config
        self.audit = audit

    # ---------------------------------------------------------
    # Top-level: parse all files, write outputs
    # ---------------------------------------------------------
    def parse_all(self, input_dir: Path, output_dir: Path) -> dict[str, Path]:
        """Parse all 4 input files; return a mapping of logical name → output path."""
        out: dict[str, Path] = {}
        output_dir.mkdir(parents=True, exist_ok=True)

        measurement_path = input_dir / self.config.measurement_file
        blank_path = input_dir / self.config.blank_file
        dilution_path = input_dir / self.config.dilution_file
        identification_path = input_dir / self.config.identification_file

        # --- main matrix ---
        log.info("parse_measurement_start", path=str(measurement_path))
        features_df, samples_measurement, abund_raw, abund_norm = self._parse_measurement(
            measurement_path
        )

        # --- blanks ---
        log.info("parse_blanks_start", path=str(blank_path))
        blanks_df, samples_blanks = self._parse_blanks(blank_path)

        # --- dilution ---
        log.info("parse_dilution_start", path=str(dilution_path))
        dilution_df, samples_dilution = self._parse_dilution(dilution_path)

        # --- samples (merged from all three files) ---
        samples_df = self._merge_samples(samples_measurement, samples_blanks, samples_dilution)

        # --- sample exclusions (drop user-flagged samples from EVERY table) ---
        # Done AFTER merging samples and BEFORE writing — every downstream
        # output (abundances, blanks, dilution) sees the cleaner set.
        if self.config.exclude_samples:
            samples_df, abund_raw, abund_norm, blanks_df, dilution_df = (
                self._apply_sample_exclusions(
                    samples_df=samples_df,
                    abund_raw=abund_raw,
                    abund_norm=abund_norm,
                    blanks_df=blanks_df,
                    dilution_df=dilution_df,
                    requested=self.config.exclude_samples,
                )
            )

        # --- write blanks + dilution (after exclusions applied) ---
        out.update(self._write(blanks_df, output_dir, self.OUT_BLANKS, BlanksLongSchema))
        out.update(
            self._write(dilution_df, output_dir, self.OUT_DILUTION, DilutionLadderLongSchema)
        )

        out.update(
            self._write_measurement_outputs(
                output_dir, features_df, samples_df, abund_raw, abund_norm
            )
        )

        # --- identification ---
        log.info("parse_identification_start", path=str(identification_path))
        ident_df = self._parse_identification(identification_path)
        out.update(self._write(ident_df, output_dir, self.OUT_IDENT_LONG, IdentificationLongSchema))

        # --- audit: every feature that passed parsing ---
        self.audit.kept(
            feature_ids=features_df["feature_id"].tolist(),
            reason="parsed_ok",
        )

        return out

    def _apply_sample_exclusions(
        self,
        *,
        samples_df: pd.DataFrame,
        abund_raw: pd.DataFrame,
        abund_norm: pd.DataFrame,
        blanks_df: pd.DataFrame,
        dilution_df: pd.DataFrame,
        requested: tuple[str, ...] | list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Drop user-specified samples from every sample-bearing table.

        ``requested`` accepts either decoded ids (``QC_5``) or raw Progenesis
        column ids (``09012026_QC_5``). Both forms are matched against
        ``samples_df.sample_id`` and ``samples_df.raw_sample_id``. Unknown
        ids log a warning and are skipped — the parser does not fail because
        the user might be exclusion-list-rotating across datasets.

        Blank and dilution tables don't carry a ``sample_id`` column, so we
        translate the matching ``samples_df`` rows into the appropriate filter
        keys (``blank_replicate`` and ``dilution_level``) and drop on those.
        """
        requested_set = set(requested)
        # Match against either decoded sample_id or raw_sample_id
        match_mask = samples_df["sample_id"].isin(requested_set) | samples_df[
            "raw_sample_id"
        ].isin(requested_set)
        matched = samples_df[match_mask].copy()

        unmatched = requested_set - (
            set(matched["sample_id"]) | set(matched["raw_sample_id"])
        )
        for u in sorted(unmatched):
            log.warning("exclude_sample_unknown_id", id=u)

        if matched.empty:
            log.info(
                "exclude_samples_no_matches",
                requested=sorted(requested_set),
            )
            return samples_df, abund_raw, abund_norm, blanks_df, dilution_df

        # Per-sample log of what's being excluded — appears in audit log
        for _, r in matched.iterrows():
            log.info(
                "exclude_sample_dropped",
                sample_id=r["sample_id"],
                raw_sample_id=r["raw_sample_id"],
                injection_type=r["injection_type"],
            )

        # 1) samples table
        excluded_sample_ids = set(matched["sample_id"])
        samples_df = samples_df[~samples_df["sample_id"].isin(excluded_sample_ids)].copy()
        # Keep column_order contiguous
        samples_df = samples_df.reset_index(drop=True)
        samples_df["column_order"] = range(len(samples_df))

        # 2) abundance long tables (sample_id is a column)
        abund_raw = abund_raw[~abund_raw["sample_id"].isin(excluded_sample_ids)].copy()
        abund_norm = abund_norm[~abund_norm["sample_id"].isin(excluded_sample_ids)].copy()

        # 3) blanks_long — filter by blank_replicate where excluded was a blank
        blank_excluded = matched[matched["injection_type"] == "blank"]
        if not blank_excluded.empty:
            blank_reps_to_drop = set(blank_excluded["replicate"].astype(int))
            blanks_df = blanks_df[
                ~blanks_df["blank_replicate"].astype(int).isin(blank_reps_to_drop)
            ].copy()

        # 4) dilution_ladder_long — filter by dilution_level where excluded was a dilution
        dilution_excluded = matched[matched["injection_type"] == "dilution"]
        if not dilution_excluded.empty:
            dilution_levels_to_drop = set(
                dilution_excluded["dilution_level"].dropna().astype(float)
            )
            dilution_df = dilution_df[
                ~dilution_df["dilution_level"].astype(float).isin(dilution_levels_to_drop)
            ].copy()

        log.info(
            "exclude_samples_applied",
            n_requested=len(requested_set),
            n_excluded=len(excluded_sample_ids),
            n_unknown=len(unmatched),
            samples_remaining=len(samples_df),
        )
        return samples_df, abund_raw, abund_norm, blanks_df, dilution_df

    def _merge_samples(self, *sample_frames: pd.DataFrame) -> pd.DataFrame:
        """Concat per-file sample frames, dedup on sample_id, renumber column_order."""
        merged = pd.concat(sample_frames, ignore_index=True)
        merged = merged.drop_duplicates(subset="sample_id", keep="first").reset_index(drop=True)
        # Reassign a globally consistent column_order = row index
        merged["column_order"] = range(len(merged))
        return SamplesSchema.validate(merged)  # type: ignore[no-any-return]

    # =========================================================
    # Measurement file parser
    # =========================================================
    def _parse_measurement(
        self,
        path: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Parse the main Progenesis MEASUREMENT file.

        Returns (features, samples, abundances_raw_long, abundances_norm_long).
        """
        df, banners = read_progenesis_csv(path, header_rows=3, banner_row_count=2)

        # --- features table ---------------------------------------
        features = self._extract_features(df)

        # --- sample metadata --------------------------------------
        sample_cols = resolve_sample_columns(df, banners)
        samples = self._extract_samples(sample_cols)

        # --- abundances long (raw + normalised) -------------------
        abund_raw = self._extract_abundances_long(df, sample_cols, block="raw")
        abund_norm = self._extract_abundances_long(df, sample_cols, block="normalised")

        return features, samples, abund_raw, abund_norm

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pull the feature-metadata columns out of a measurement-format df."""
        rename_map = {
            "Compound": "feature_id",
            "Neutral mass (Da)": "neutral_mass",
            "m/z": "m_over_z",
            "Charge": "charge",
            "Retention time (min)": "rt_min",
            "CCS (angstrom^2)": "measured_ccs",
            "Chromatographic peak width (min)": "peak_width_min",
            "Identifications": "identifications_count",
            "Anova (p)": "anova_p_progenesis",
            "Max Fold Change": "max_fold_change_progenesis",
            "Highest Mean": "highest_mean_group",
            "Lowest Mean": "lowest_mean_group",
            "Isotope Distribution": "isotope_distribution",
            "Maximum Abundance": "maximum_abundance",
            "Minimum CV%": "minimum_cv_pct",
        }
        cols = [c for c in rename_map if c in df.columns]
        feats = df[cols].rename(columns=rename_map).copy()
        return FeaturesSchema.validate(feats)  # type: ignore[no-any-return]

    def _extract_samples(self, sample_cols: list[SampleColumn]) -> pd.DataFrame:
        """One row per unique raw sample id (dedup across blocks)."""
        seen: dict[str, DecodedSample] = {}
        ordering: dict[str, int] = {}
        for idx, sc in enumerate(sample_cols):
            if sc.raw_sample_id in seen:
                continue
            decoded = decode_sample_id(sc.raw_sample_id, pattern=self.config.sample_id_pattern)
            seen[sc.raw_sample_id] = decoded
            ordering[sc.raw_sample_id] = idx

        rows = []
        for raw, dec in seen.items():
            rows.append(
                {
                    "sample_id": dec.sample_id,
                    "raw_sample_id": dec.raw_sample_id,
                    "injection_type": dec.injection_type,
                    "cell_line": dec.cell_line,
                    "replicate": dec.replicate,
                    "dilution_level": dec.dilution_level,
                    "acq_date": dec.acq_date,
                    "column_order": ordering[raw],
                }
            )
        samples_df = pd.DataFrame(rows)
        return SamplesSchema.validate(samples_df)  # type: ignore[no-any-return]

    def _extract_abundances_long(
        self,
        df: pd.DataFrame,
        sample_cols: list[SampleColumn],
        *,
        block: str,
    ) -> pd.DataFrame:
        """Melt wide sample columns for a given block → long (feature, sample, intensity)."""
        cols_for_block = [sc for sc in sample_cols if sc.block == block]
        if not cols_for_block:
            log.warning("no_columns_for_block", block=block)
            return pd.DataFrame(columns=["feature_id", "sample_id", "intensity"])

        # Build a rename map from dataframe_column → sample_id
        rename = {
            sc.dataframe_column: decode_sample_id(
                sc.raw_sample_id, pattern=self.config.sample_id_pattern
            ).sample_id
            for sc in cols_for_block
        }
        sub = df[["Compound", *[sc.dataframe_column for sc in cols_for_block]]].rename(
            columns={"Compound": "feature_id", **rename}
        )
        long = sub.melt(id_vars="feature_id", var_name="sample_id", value_name="intensity")
        long["intensity"] = pd.to_numeric(long["intensity"], errors="coerce")
        return AbundancesLongSchema.validate(long)  # type: ignore[no-any-return]

    # =========================================================
    # Output writing (measurement family)
    # =========================================================
    def _write_measurement_outputs(
        self,
        output_dir: Path,
        features: pd.DataFrame,
        samples: pd.DataFrame,
        abund_raw: pd.DataFrame,
        abund_norm: pd.DataFrame,
    ) -> dict[str, Path]:
        out: dict[str, Path] = {}
        out["features"] = self._persist(features, output_dir / self.OUT_FEATURES)
        out["samples"] = self._persist(samples, output_dir / self.OUT_SAMPLES)
        if self.config.emit_long:
            out["abundances_raw_long"] = self._persist(
                abund_raw, output_dir / self.OUT_ABUND_RAW_LONG
            )
            out["abundances_normalised_long"] = self._persist(
                abund_norm, output_dir / self.OUT_ABUND_NORM_LONG
            )
        if self.config.emit_wide:
            out["abundances_raw_wide"] = self._persist(
                abund_raw.pivot(index="feature_id", columns="sample_id", values="intensity"),
                output_dir / self.OUT_ABUND_RAW_WIDE,
                index_name="feature_id",
            )
            out["abundances_normalised_wide"] = self._persist(
                abund_norm.pivot(index="feature_id", columns="sample_id", values="intensity"),
                output_dir / self.OUT_ABUND_NORM_WIDE,
                index_name="feature_id",
            )
        return out

    # =========================================================
    # Blank file parser
    # =========================================================
    def _parse_blanks(self, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract the 3 Blank injection columns into (long_df, samples_df)."""
        df, banners = read_progenesis_csv(path, header_rows=3, banner_row_count=2)
        sample_cols = resolve_sample_columns(df, banners)
        blank_cols = [
            sc for sc in sample_cols if re.search(r"_Blank_\d+", sc.raw_sample_id, re.IGNORECASE)
        ]
        if not blank_cols:
            log.warning("no_blank_columns_found", path=str(path))
            empty = pd.DataFrame(columns=list(BlanksLongSchema.to_schema().columns.keys()))
            return empty, self._extract_samples([])

        rows: list[dict[str, Any]] = []
        for sc in blank_cols:
            # Blank_1, Blank_2, Blank_3 → replicate int
            m = re.search(r"Blank_(\d+)", sc.raw_sample_id, re.IGNORECASE)
            if not m:
                continue
            rep = int(m.group(1))
            series = pd.to_numeric(df[sc.dataframe_column], errors="coerce")
            sub = pd.DataFrame(
                {
                    "feature_id": df["Compound"].values,
                    "blank_replicate": rep,
                    "block": sc.block,
                    "intensity": series.values,
                }
            )
            rows.append(sub)
        long = pd.concat(rows, ignore_index=True)
        samples = self._extract_samples(blank_cols)
        return BlanksLongSchema.validate(long), samples  # type: ignore[return-value]

    # =========================================================
    # Dilution file parser
    # =========================================================
    def _parse_dilution(self, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract QC + dilution columns into (long_df, samples_df)."""
        df, banners = read_progenesis_csv(path, header_rows=3, banner_row_count=2)
        sample_cols = resolve_sample_columns(df, banners)

        # Dilution series columns look like `<date>_QC_1_2` / `_QC_1_4` / `_QC_1_8`.
        # The undiluted reference is the set of `<date>_QC_<n>` (16 of them).
        dilution_map: dict[float, list[SampleColumn]] = {1.0: [], 0.5: [], 0.25: [], 0.125: []}
        dilution_pattern = re.compile(r"_QC_1_(?P<dn>\d+)$")
        qc_only_pattern = re.compile(r"_QC_\d+$")
        for sc in sample_cols:
            m = dilution_pattern.search(sc.raw_sample_id)
            if m:
                dn = int(m.group("dn"))
                factor = 1.0 / dn
                if factor in dilution_map:
                    dilution_map[factor].append(sc)
                continue
            if qc_only_pattern.search(sc.raw_sample_id):
                dilution_map[1.0].append(sc)

        rows: list[pd.DataFrame] = []
        dilution_only_cols: list[SampleColumn] = []  # for samples.parquet — only QC_1:N entries
        for factor, cols in dilution_map.items():
            if not cols:
                continue
            for sc in cols:
                series = pd.to_numeric(df[sc.dataframe_column], errors="coerce")
                level = 1.0 / factor
                rows.append(
                    pd.DataFrame(
                        {
                            "feature_id": df["Compound"].values,
                            "dilution_level": level,
                            "dilution_factor": factor,
                            "block": sc.block,
                            "intensity": series.values,
                        }
                    )
                )
                # Only the genuinely-diluted columns (factor < 1) are new samples
                # we need to add to samples.parquet; the factor=1.0 QCs are
                # already there from the measurement file.
                if factor < 1.0:
                    dilution_only_cols.append(sc)
        long = pd.concat(rows, ignore_index=True)
        samples = self._extract_samples(dilution_only_cols)
        return DilutionLadderLongSchema.validate(long), samples  # type: ignore[return-value]

    # =========================================================
    # Identification file parser  (no outer-quote wrapping)
    # =========================================================
    def _parse_identification(self, path: Path) -> pd.DataFrame:
        """Extract the long-format identification candidates table.

        The identification file has two leading rows:
          row 0 — sparse marker row with "Tags" in a trailing column
          row 1 — actual column headers
        We skip row 0 and use row 1 as header.
        """
        df = pd.read_csv(path, skiprows=1, low_memory=False)

        rename = {
            "Compound": "feature_id",
            "Compound ID": "compound_id",
            "Accepted?": "accepted",
            "Adducts": "adducts",
            "Formula": "formula",
            "Score": "score",
            "Fragmentation Score": "fragmentation_score",
            "Mass Error (ppm)": "mass_error_ppm",
            "Theoretical CCS (angstrom^2)": "theoretical_ccs",
            "dCCS (angstrom^2)": "dccs",
            "Isotope Similarity": "isotope_similarity",
            "Theoretical Isotope Distribution": "theoretical_isotope_distribution",
            "Link": "link",
            "Description": "description",
        }
        cols = [c for c in rename if c in df.columns]
        out = df[cols].rename(columns=rename).copy()

        # Accepted is 'x' / empty → True / False
        out["accepted"] = out["accepted"].notna() & (out["accepted"].astype(str) != "")

        # candidate_idx = running count within feature_id group, preserving source order
        out["candidate_idx"] = out.groupby("feature_id").cumcount()

        # Reorder cols
        ordered = [
            "feature_id",
            "candidate_idx",
            "compound_id",
            "accepted",
            "adducts",
            "formula",
            "score",
            "fragmentation_score",
            "mass_error_ppm",
            "theoretical_ccs",
            "dccs",
            "isotope_similarity",
            "theoretical_isotope_distribution",
            "link",
            "description",
        ]
        out = out[ordered]
        return IdentificationLongSchema.validate(out)  # type: ignore[no-any-return]

    # =========================================================
    # Persistence helpers
    # =========================================================
    def _persist(
        self,
        df: pd.DataFrame,
        path: Path,
        *,
        index_name: str | None = None,
    ) -> Path:
        """Write a DataFrame to parquet and optionally to CSV mirror."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if index_name is not None:
            # Wide tables need the index preserved as a column on parquet read.
            df = df.reset_index()
        if self.config.emit_parquet:
            df.to_parquet(path, engine="pyarrow", compression="zstd")
            log.info("wrote_parquet", path=str(path), rows=len(df), cols=len(df.columns))
        if self.config.emit_csv_mirror:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            log.debug("wrote_csv_mirror", path=str(csv_path))
        return path

    def _write(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        filename: str,
        schema: type,  # pa.DataFrameModel subclass — typed loosely to avoid meta-type gymnastics
    ) -> dict[str, Path]:
        validated = schema.validate(df)
        return {Path(filename).stem: self._persist(validated, output_dir / filename)}
