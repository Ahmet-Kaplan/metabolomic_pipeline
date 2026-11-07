"""Build the chronological injection run-order from Waters _HEADER.TXT files.

Each Waters MassLynx ``_HEADER.TXT`` file (one per injection) contains lines:

    $$ Acquired Name: 09012026_MCF7_4
    $$ Acquired Date: 10-Jan-2026
    $$ Acquired Time: 00:56:59

We parse all such files in a directory, sort by combined date+time, dedupe,
filter to samples present in the project's ``samples.parquet`` (so pre-
acquisition equilibration blanks etc. are excluded), and assign contiguous
``position`` ``1..N``.

Why we need run-order
---------------------
Drift correction (Stage 12 — QC-RSC, SERRF, TIGER, etc.) anchors a per-
feature trend on the QC-pool injections positioned across the sequence.
Without the chronological order, drift cannot be computed.

Output schema (``run_order.parquet``):

    position         int   1..N (contiguous, after dedupe + filter)
    raw_sample_id    str   matches Progenesis column header
    acq_datetime     dt    combined acquired date + time
    header_file      str   source _HEADER file (for traceability)

Validation
----------
The build emits a warning + exits non-zero if any sample in
``samples.parquet`` lacks a header — drift correction would otherwise be
silently incomplete. A warning (not error) is emitted when extra headers
are dropped (typical: pre-acquisition equilibration blanks).
"""

from __future__ import annotations

import re
import time as _time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# Patterns for the three lines we care about in each header file.
# Anchored with $ on each side of `Acquired` to avoid catching e.g.
# `$$ Cal Date:` lines.
_ACQUIRED_NAME_RE = re.compile(r"^\$\$ Acquired Name:\s*(.+?)\s*$", re.MULTILINE)
_ACQUIRED_DATE_RE = re.compile(r"^\$\$ Acquired Date:\s*(.+?)\s*$", re.MULTILINE)
_ACQUIRED_TIME_RE = re.compile(r"^\$\$ Acquired Time:\s*(.+?)\s*$", re.MULTILINE)

# Waters MassLynx date format: "10-Jan-2026"
_DATE_FORMAT = "%d-%b-%Y"
_TIME_FORMAT = "%H:%M:%S"


# ============================================================
# Config + record dataclasses
# ============================================================
@dataclass(frozen=True)
class RunOrderConfig:
    """Runtime config for the run-order builder."""

    headers_dir: Path
    output_parquet: Path
    samples_parquet: Path | None = None
    drop_unknown_samples: bool = True
    fail_on_missing_samples: bool = True


@dataclass(frozen=True)
class HeaderRecord:
    """One parsed _HEADER.TXT file."""

    file: str
    acquired_name: str
    acq_datetime: datetime


# ============================================================
# Per-file parser
# ============================================================
def parse_header_file(path: Path) -> HeaderRecord:
    """Extract ``Acquired Name`` + combined date/time from one _HEADER file.

    Raises
    ------
    ValueError
        If any of the three required fields is missing or malformed.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    name_m = _ACQUIRED_NAME_RE.search(text)
    date_m = _ACQUIRED_DATE_RE.search(text)
    time_m = _ACQUIRED_TIME_RE.search(text)
    if not (name_m and date_m and time_m):
        missing = [
            n
            for n, m in [("Name", name_m), ("Date", date_m), ("Time", time_m)]
            if m is None
        ]
        raise ValueError(f"{path.name}: missing Acquired {missing}")

    name = name_m.group(1)
    date_str = date_m.group(1)
    time_str = time_m.group(1)
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", f"{_DATE_FORMAT} {_TIME_FORMAT}")
    except ValueError as exc:  # pragma: no cover — defensive
        raise ValueError(
            f"{path.name}: cannot parse acq date/time '{date_str} {time_str}': {exc}"
        ) from exc
    return HeaderRecord(file=path.name, acquired_name=name, acq_datetime=dt)


# ============================================================
# Directory scan
# ============================================================
def parse_all_headers(
    headers_dir: Path,
    *,
    pattern: str = "_HEADER*.TXT",
) -> pd.DataFrame:
    """Return a DataFrame with one row per parsed _HEADER file."""
    if not headers_dir.exists():
        raise FileNotFoundError(f"Headers dir not found: {headers_dir}")
    files = sorted(headers_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No header files matching '{pattern}' in {headers_dir}"
        )
    log.info("run_order_parse_start", n_files=len(files), dir=str(headers_dir))
    t0 = _time.time()

    records: list[HeaderRecord] = []
    bad_files: list[str] = []
    for p in files:
        try:
            records.append(parse_header_file(p))
        except ValueError as exc:
            bad_files.append(p.name)
            log.warning("header_parse_failed", file=p.name, error=str(exc))
    if bad_files:
        log.warning("header_parse_partial", n_failed=len(bad_files), n_ok=len(records))

    df = pd.DataFrame(
        [
            {"file": r.file, "acquired_name": r.acquired_name, "acq_datetime": r.acq_datetime}
            for r in records
        ]
    )
    log.info(
        "run_order_parse_done",
        n_files=len(files),
        n_parsed=len(df),
        n_failed=len(bad_files),
        elapsed_s=round(_time.time() - t0, 2),
    )
    return df


# ============================================================
# Build pipeline: parse → sort → dedupe → filter → assign position
# ============================================================
def _dedupe_by_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (acquired_name, acq_datetime) rows; keep the first file."""
    n_before = len(df)
    deduped = df.drop_duplicates(
        subset=["acquired_name", "acq_datetime"], keep="first"
    ).reset_index(drop=True)
    n_dropped = n_before - len(deduped)
    if n_dropped > 0:
        dropped = df[df.duplicated(subset=["acquired_name", "acq_datetime"], keep="first")]
        for _, r in dropped.iterrows():
            log.warning(
                "run_order_duplicate_dropped",
                file=r["file"],
                acquired_name=r["acquired_name"],
                acq_datetime=str(r["acq_datetime"]),
            )
    return deduped


def _filter_to_samples(
    df: pd.DataFrame,
    valid_ids: Iterable[str],
) -> tuple[pd.DataFrame, set[str]]:
    """Keep only headers whose ``acquired_name`` matches a real sample id.

    Returns ``(filtered_df, excluded_names)``. Pre-acquisition blanks
    (``Blank_0``, system equilibration injections) are typical exclusions.
    """
    valid_set = set(valid_ids)
    in_valid = df["acquired_name"].isin(valid_set)
    excluded = set(df.loc[~in_valid, "acquired_name"])
    if excluded:
        for n in sorted(excluded):
            log.info("run_order_extra_header_dropped", acquired_name=n)
    return df[in_valid].copy(), excluded


def build(config: RunOrderConfig) -> pd.DataFrame:
    """End-to-end: parse headers → sort → dedupe → filter → write parquet.

    Returns the run-order DataFrame.

    Raises
    ------
    ValueError
        If ``samples_parquet`` is provided, ``fail_on_missing_samples`` is
        True (default), and any sample from ``samples.parquet`` lacks a
        header. Drift correction would otherwise silently miss those.
    """
    log.info("run_order_build_start", config=str(config))
    df = parse_all_headers(config.headers_dir)

    # Sort chronologically (stable for tied timestamps — first file wins on dedupe)
    df = df.sort_values("acq_datetime", kind="stable").reset_index(drop=True)
    df = _dedupe_by_datetime(df)

    # Validate / filter against samples.parquet
    excluded: set[str] = set()
    missing: set[str] = set()
    if config.samples_parquet is not None:
        if not config.samples_parquet.exists():
            raise FileNotFoundError(
                f"samples.parquet not found at {config.samples_parquet}; "
                f"build Stage 1 first"
            )
        samples = pd.read_parquet(config.samples_parquet)
        valid_ids = samples["raw_sample_id"].tolist()
        sample_set = set(valid_ids)

        if config.drop_unknown_samples:
            df, excluded = _filter_to_samples(df, valid_ids)

        # Headers covering all samples?
        header_set = set(df["acquired_name"])
        missing = sample_set - header_set
        if missing and config.fail_on_missing_samples:
            raise ValueError(
                f"{len(missing)} samples in samples.parquet have NO _HEADER file: "
                f"{sorted(missing)}. Drift correction would be incomplete."
            )
        if missing:  # warning case
            for s in sorted(missing):
                log.warning("run_order_sample_missing_header", raw_sample_id=s)

    # Assign contiguous position 1..N
    df = df.reset_index(drop=True)
    df["position"] = range(1, len(df) + 1)
    df = df.rename(columns={"acquired_name": "raw_sample_id", "file": "header_file"})

    # Final column order
    out = df[["position", "raw_sample_id", "acq_datetime", "header_file"]].copy()

    config.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.output_parquet, index=False, compression="zstd")
    log.info(
        "run_order_built",
        rows=len(out),
        path=str(config.output_parquet),
        n_extra_dropped=len(excluded),
        n_missing_samples=len(missing),
        first_acq=str(out["acq_datetime"].min()),
        last_acq=str(out["acq_datetime"].max()),
        duration_min=round(
            (out["acq_datetime"].max() - out["acq_datetime"].min()).total_seconds() / 60, 1
        ),
    )
    return out
