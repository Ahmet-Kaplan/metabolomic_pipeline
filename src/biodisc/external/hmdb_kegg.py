"""Extract HMDB_ID → KEGG Compound ID bridge from the HMDB metabolites XML.

HMDB v5 entries contain ``<kegg_id>`` elements linking to KEGG compound
identifiers (prefixed ``C``, e.g., C00001 for 1,3-Propanediol). This
bridge lets us map the pipeline's HMDB-based identifications into the
KEGG pathway and reaction network for enrichment analysis.

Output schema (``hmdb_kegg.parquet``)::

    hmdb_id      string   (e.g. "HMDB0000001")
    kegg_id      string   (e.g. "C00001", may be NULL)

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
    """Stream-parse HMDB XML; return one row per metabolite (hmdb_id, kegg_id)."""
    if not xml_path.exists():
        raise FileNotFoundError(f"HMDB XML not found: {xml_path}")

    log.info("hmdb_kegg_extract_start", path=str(xml_path))
    t0 = time.time()

    rows: list[tuple[str, str | None]] = []
    ctx = ET.iterparse(str(xml_path), events=("end",))
    for _, elem in ctx:
        if elem.tag != f"{NS}metabolite":
            continue
        accession_el = elem.find(f"{NS}accession")
        kegg_el = elem.find(f"{NS}kegg_id")
        if accession_el is not None and accession_el.text:
            hmdb_id = accession_el.text.strip()
            kegg = kegg_el.text.strip() if kegg_el is not None and kegg_el.text else None
            rows.append((hmdb_id, kegg if kegg else None))
        elem.clear()

    df = pd.DataFrame(rows, columns=["hmdb_id", "kegg_id"])
    log.info(
        "hmdb_kegg_extract_done",
        n_metabolites=len(df),
        n_with_kegg=int(df["kegg_id"].notna().sum()),
        elapsed_s=round(time.time() - t0, 1),
    )
    return df


def write(df: pd.DataFrame, out_path: Path) -> None:
    """Write the bridge as parquet."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="zstd")
    log.info("hmdb_kegg_written", path=str(out_path), rows=len(df))


def build(xml_path: Path, out_path: Path) -> pd.DataFrame:
    """End-to-end: parse XML, write parquet, return the dataframe."""
    df = extract(xml_path)
    write(df, out_path)
    return df
