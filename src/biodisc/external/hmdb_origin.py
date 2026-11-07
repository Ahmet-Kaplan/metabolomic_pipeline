"""Extract HMDB compound → biological origin classification.

HMDB v5's bulk-distribution XML contains an ontology block with multiple
roots. The "Source" categorisation is NESTED under the "Disposition"
root (verified empirically — it's not a top-level root):

    <ontology>
      <root>
        <term>Disposition</term>
        <descendants>
          <descendant>
            <term>Source</term>
            <descendants>
              <descendant><term>Endogenous</term> …</descendant>
              <descendant><term>Food</term> …</descendant>
              <descendant><term>Microbial</term> …</descendant>
              …
            </descendants>
          </descendant>
          <descendant><term>Route of exposure</term> …</descendant>
          <descendant><term>Biological location</term> …</descendant>
        </descendants>
      </root>
      <root>
        <term>Role</term> …
      </root>
    </ontology>

For human breast-cancer cell-line metabolomics, "Endogenous" is the
biology we care about; everything else (Food, Drug, Microbial, Plant,
Cosmetic, Synthetic, Industrial, Pesticide, Animal-derived = FBS) is
exogenous contamination. Stage 10 uses these labels to drop features
whose ALL surviving cascade candidates are pure-exogenous.

Output schema (``hmdb_origin.parquet``):

    hmdb_id          string                              (e.g. "HMDB0000122")
    source_terms     list[string]                        (e.g. ["Endogenous", "Food"])
    is_endogenous    bool      "Endogenous" tag present
    has_exogenous_tag bool     ≥ 1 non-Endogenous tag
    origin_class     string    one of:
        - "endogenous_only" — only "Endogenous" tag (or co-tagged with
          biological-context tags only)
        - "both"            — "Endogenous" AND ≥ 1 exogenous tag
        - "exogenous_only"  — exogenous tag(s), no "Endogenous"
        - "unannotated"     — no Source tags at all (HMDB hasn't classified)

The XML is large (~6 GB). We stream-parse with iterparse + clear so
memory stays bounded. Expected runtime ~3 min on this hardware.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

NS = "{http://www.hmdb.ca}"

# HMDB Source-ontology terms we treat as ENDOGENOUS evidence.
# In HMDB v5, "Endogenous" is the canonical tag for host-produced metabolites.
ENDOGENOUS_TERMS: frozenset[str] = frozenset({"Endogenous"})

# All other top-level Source descendants are exogenous in cell-culture context.
# Common HMDB tags include: Food, Drug, Drug metabolite, Microbial,
# Plant, Cosmetic, Synthetic, Industrial, Pesticide, Toxin, Pollutant,
# Animal, Mammal, Biological. We don't enumerate explicitly — anything
# in source_terms that's NOT in ENDOGENOUS_TERMS counts as exogenous.


def _extract_source_terms(metabolite_el: ET.Element) -> list[str]:
    """Return top-level Source descendants for one <metabolite> element.

    The Source category is nested under the Disposition root in HMDB v5.
    We walk: ontology → root[term=Disposition] → descendants → descendant
    [term=Source] → descendants → descendant → term.

    Only immediate children of the Source descendant's <descendants> are
    returned — sub-categories (e.g. "Beverage" under "Food") are ignored.
    """
    onto = metabolite_el.find(f"{NS}ontology")
    if onto is None:
        return []
    for root in onto.findall(f"{NS}root"):
        term_el = root.find(f"{NS}term")
        if term_el is None or term_el.text != "Disposition":
            continue
        # Walk into Disposition's descendants → find the Source one
        disp_descs = root.find(f"{NS}descendants")
        if disp_descs is None:
            return []
        for child in disp_descs.findall(f"{NS}descendant"):
            child_term = child.find(f"{NS}term")
            if child_term is None or child_term.text != "Source":
                continue
            # Found the Source descendant — collect its top-level descendant terms
            src_descs = child.find(f"{NS}descendants")
            if src_descs is None:
                return []
            out: list[str] = []
            for src_child in src_descs.findall(f"{NS}descendant"):
                t = src_child.find(f"{NS}term")
                if t is not None and t.text:
                    out.append(t.text.strip())
            return out
        return []  # Disposition root present but no Source descendant
    return []  # No Disposition root


def _classify(source_terms: list[str]) -> str:
    """Bucket a metabolite into endogenous_only / both / exogenous_only / unannotated."""
    if not source_terms:
        return "unannotated"
    has_endo = any(t in ENDOGENOUS_TERMS for t in source_terms)
    has_exo = any(t not in ENDOGENOUS_TERMS for t in source_terms)
    if has_endo and has_exo:
        return "both"
    if has_endo:
        return "endogenous_only"
    return "exogenous_only"


def extract(xml_path: Path) -> pd.DataFrame:
    """Stream-parse HMDB XML; return one row per metabolite with origin labels."""
    if not xml_path.exists():
        raise FileNotFoundError(f"HMDB XML not found: {xml_path}")

    log.info("hmdb_origin_extract_start", path=str(xml_path))
    t0 = time.time()

    rows: list[dict[str, object]] = []
    ctx = ET.iterparse(str(xml_path), events=("end",))
    for _, elem in ctx:
        if elem.tag != f"{NS}metabolite":
            continue
        accession_el = elem.find(f"{NS}accession")
        if accession_el is None or not accession_el.text:
            elem.clear()
            continue
        hmdb_id = accession_el.text.strip()
        terms = _extract_source_terms(elem)
        rows.append(
            {
                "hmdb_id": hmdb_id,
                "source_terms": terms,
                "is_endogenous": any(t in ENDOGENOUS_TERMS for t in terms),
                "has_exogenous_tag": any(t not in ENDOGENOUS_TERMS for t in terms),
                "origin_class": _classify(terms),
            }
        )
        elem.clear()

    df = pd.DataFrame(rows)
    log.info(
        "hmdb_origin_extract_done",
        n_metabolites=len(df),
        n_endogenous=int(df["is_endogenous"].sum()),
        n_exogenous_only=int((df["origin_class"] == "exogenous_only").sum()),
        n_both=int((df["origin_class"] == "both").sum()),
        n_unannotated=int((df["origin_class"] == "unannotated").sum()),
        elapsed_s=round(time.time() - t0, 1),
    )
    return df


def write(df: pd.DataFrame, out_path: Path) -> None:
    """Persist as parquet (lists are stored natively by pyarrow)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="zstd")
    log.info("hmdb_origin_written", path=str(out_path), rows=len(df))


def build(xml_path: Path, out_path: Path) -> pd.DataFrame:
    """End-to-end: parse XML → write parquet → return DataFrame."""
    df = extract(xml_path)
    write(df, out_path)
    return df
