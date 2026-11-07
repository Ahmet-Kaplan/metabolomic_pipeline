"""Build the unified CCS reference table that Stage 8 consumes.

Currently sources only the McLean Unified CCS Compendium (Picache et al.
2019) — gold-standard experimental Tier-2 data. When AllCCS2 bulk arrives
or when Hines/Paglia get manually downloaded, this module will absorb
them via additional ``_ingest_*`` functions and concatenate.

Output schema (``ccs_reference_unified.parquet``) — one row per
(hmdb_id, adduct, source_db) triple:

    hmdb_id           string   (the join key for the discovery pipeline)
    inchikey          string   (original key in the source DB; for audit)
    compound_name     string   (for audit / report readability)
    formula           string
    adduct            string   (CANONICAL FORM: "M+H", "M-H", "M+Na", … no brackets)
    charge            int      (+1, -1, +2, …)
    reference_ccs     float    (the experimental CCS value, Å²)
    ccs_sd            float    (replicate standard deviation; may be NaN)
    n_replicates      int      (how many measurements went into reference_ccs)
    source_db         string   ("McLean_Compendium" for now)
    instrument_type   string   ("DTIMS" for McLean — Agilent 6560 platform)
    confidence_tier   int      (2 = experimental, 4 = predicted; all 2 today)

Adduct normalisation: McLean uses bracketed notation ("[M+H]"); Progenesis
emits unbracketed ("M+H, M+Na, …"). We strip brackets and whitespace so
both sides of the join use the same canonical form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger(__name__)


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class CCSReferenceConfig:
    """Inputs + output for the build."""

    mclean_csv: Path
    hmdb_inchikey_parquet: Path
    output_parquet: Path


# ============================================================
# Helpers
# ============================================================
_BRACKET_RE = re.compile(r"[\[\]\s]")


def normalise_adduct(raw: str | float | None) -> str | None:
    """Canonicalise an adduct string for cross-source matching.

    Examples:
        "[M+H]"      → "M+H"
        " [M-H] "    → "M-H"
        "[M+Na]+"    → "M+Na+"   (charge sign preserved if present)
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    return _BRACKET_RE.sub("", str(raw))


# ============================================================
# Per-source ingestion
# ============================================================
def ingest_mclean(csv_path: Path) -> pd.DataFrame:
    """Load and normalise the McLean Compendium allMASTER.csv."""
    log.info("ingest_mclean_start", path=str(csv_path))
    raw = pd.read_csv(csv_path)
    # Strip BOM from header
    raw.columns = [c.lstrip("﻿").strip() for c in raw.columns]

    out = pd.DataFrame(
        {
            "inchikey": raw["InChiKey"].astype("string"),
            "compound_name": raw["Compound"].astype("string"),
            "formula": raw["Neutral.Formula"].astype("string"),
            "adduct": raw["Ion.Species"].apply(normalise_adduct).astype("string"),
            "charge": pd.to_numeric(raw["Charge"], errors="coerce").astype("Int16"),
            "reference_ccs": pd.to_numeric(raw["CCS"], errors="coerce").astype("float64"),
            "ccs_sd": pd.to_numeric(raw["SD"], errors="coerce").astype("float64"),
            "n_replicates": pd.to_numeric(raw["N.Rep"], errors="coerce").astype("Int16"),
        }
    )
    out["source_db"] = "McLean_Compendium"
    out["instrument_type"] = "DTIMS"  # Agilent 6560 platform — Picache 2019
    out["confidence_tier"] = 2  # experimental

    # Drop rows where the join key is missing or CCS is invalid
    before = len(out)
    out = out[out["inchikey"].notna() & out["reference_ccs"].notna() & out["adduct"].notna()].copy()
    log.info(
        "ingest_mclean_done",
        rows_in=before,
        rows_out=len(out),
        dropped=before - len(out),
    )
    return out


# ============================================================
# Build
# ============================================================
def build(config: CCSReferenceConfig) -> pd.DataFrame:
    """Build unified CCS reference: ingest McLean → bridge to HMDB → write parquet."""
    log.info("ccs_reference_build_start")

    mclean = ingest_mclean(config.mclean_csv)

    # Bridge InChiKey → HMDB ID via the HMDB-derived map
    if not config.hmdb_inchikey_parquet.exists():
        raise FileNotFoundError(
            f"HMDB→InChiKey bridge missing: {config.hmdb_inchikey_parquet}\n"
            f"Run `biodisc external build-hmdb-inchikey` first."
        )
    bridge = pd.read_parquet(config.hmdb_inchikey_parquet)
    bridge = bridge.dropna(subset=["inchikey"]).drop_duplicates("inchikey")
    log.info("bridge_loaded", rows=len(bridge))

    unified = mclean.merge(bridge, on="inchikey", how="inner")
    log.info(
        "bridge_join_done",
        mclean_rows=len(mclean),
        unified_rows=len(unified),
        n_joined_inchikeys=unified["inchikey"].nunique(),
    )

    # Final column order
    unified = unified[
        [
            "hmdb_id",
            "inchikey",
            "compound_name",
            "formula",
            "adduct",
            "charge",
            "reference_ccs",
            "ccs_sd",
            "n_replicates",
            "source_db",
            "instrument_type",
            "confidence_tier",
        ]
    ].copy()

    # Deduplicate: same (hmdb_id, adduct, source_db) → keep highest n_replicates
    unified = unified.sort_values("n_replicates", ascending=False, na_position="last")
    unified = unified.drop_duplicates(subset=["hmdb_id", "adduct", "source_db"], keep="first")
    unified = unified.reset_index(drop=True)

    config.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(config.output_parquet, index=False, compression="zstd")
    log.info(
        "ccs_reference_build_done",
        rows=len(unified),
        unique_hmdb=unified["hmdb_id"].nunique(),
        unique_adducts=unified["adduct"].nunique(),
        path=str(config.output_parquet),
    )
    return unified
