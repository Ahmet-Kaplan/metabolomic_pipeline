"""Extract HMDB_ID → InChiKey bridge from the HMDB metabolites XML.

HMDB v5's bulk-distribution XML does NOT contain CCS values, but it does
contain the canonical InChiKey for every metabolite. We need this bridge
to join Progenesis identifications (which key on HMDB IDs) against the
McLean CCS Compendium (which keys on InChiKey).

Output schema (``hmdb_inchikey.parquet``):
    hmdb_id      string  (e.g. "HMDB0000001")
    inchikey     string  (e.g. "BRMWTNUJHUMWMS-LURJTMIESA-N", may be NULL)

The XML is large (~6 GB). We stream-parse with ``iterparse`` and clear
each ``<metabolite>`` element on close, keeping memory bounded.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

NS = "{http://www.hmdb.ca}"


def extract(xml_path: Path) -> pd.DataFrame:
    """Stream-parse HMDB XML; return one row per metabolite (hmdb_id, inchikey)."""
    if not xml_path.exists():
        raise FileNotFoundError(f"HMDB XML not found: {xml_path}")

    log.info("hmdb_inchikey_extract_start", path=str(xml_path))
    t0 = time.time()

    rows: list[tuple[str, str | None]] = []
    ctx = ET.iterparse(str(xml_path), events=("end",))
    for _, elem in ctx:
        if elem.tag != f"{NS}metabolite":
            continue
        accession_el = elem.find(f"{NS}accession")
        inchikey_el = elem.find(f"{NS}inchikey")
        if accession_el is not None and accession_el.text:
            inchikey = inchikey_el.text if inchikey_el is not None else None
            rows.append((accession_el.text.strip(), inchikey.strip() if inchikey else None))
        elem.clear()  # bound memory

    df = pd.DataFrame(rows, columns=["hmdb_id", "inchikey"])
    log.info(
        "hmdb_inchikey_extract_done",
        n_metabolites=len(df),
        n_with_inchikey=int(df["inchikey"].notna().sum()),
        elapsed_s=round(time.time() - t0, 1),
    )
    return df


def write(df: pd.DataFrame, out_path: Path) -> None:
    """Write the bridge as parquet (small, ~few MB)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="zstd")
    log.info("hmdb_inchikey_written", path=str(out_path), rows=len(df))


def build(xml_path: Path, out_path: Path) -> pd.DataFrame:
    """End-to-end: parse XML, write parquet, return the dataframe."""
    df = extract(xml_path)
    write(df, out_path)
    return df
