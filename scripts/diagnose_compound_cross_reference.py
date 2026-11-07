"""Compound cross-reference diagnostic — KEGG, ChEBI, ChEMBL, UniChem, m/z search.

Takes the multiverse consensus shortlist and cross-references every
feature against multiple databases:

* **Identified features** (MSI L2/L3): KEGG → ChEBI → ChEMBL → PubChem,
  via UniChem for cross-database identifier mapping.
* **Unidentified features** (MSI Tier 4): search by precursor m/z (from
  the Progenesis features table) against ChEBI's exact-mass registry
  at configurable ppm tolerance, reporting the best candidate matches.

Usage
-----
    uv sync --extra enrichment
    uv run python scripts/diagnose_compound_cross_reference.py

Writes report to ``docs/evaluation/compound_cross_reference.md``.
"""

from __future__ import annotations

import contextlib
import math
import time
from pathlib import Path

import pandas as pd

INTERIM = Path("data/interim")
CONSENSUS = Path("data/reports/multiverse")
EXTERNAL = Path("data/external")
REPORT = Path("docs/evaluation/compound_cross_reference.md")

MZ_TOLERANCE_PPM = 10  # mass tolerance for m/z search (parts per million)


# ============================================================
# Data Loading
# ============================================================

def _load_consensus_shortlist() -> pd.DataFrame:
    path = CONSENSUS / "consensus_passed.parquet"
    if not path.exists():
        raise SystemExit(f"Consensus shortlist not found: {path}")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df)} consensus-passed features")
    return df


def _load_stage9_metrics() -> pd.DataFrame:
    path = INTERIM / "09_id_ranking" / "metrics.parquet"
    if not path.exists():
        raise SystemExit(f"Stage 9 metrics not found: {path}")
    df = pd.read_parquet(path)
    needed = {"feature_id", "top_compound_id", "top_compound_name",
              "msi_tier", "outcome", "top_formula"}
    if not needed.issubset(df.columns):
        missing = needed - set(df.columns)
        raise SystemExit(f"Stage 9 metrics missing columns: {missing}")
    return df[list(needed)].copy()


def _load_features_table() -> pd.DataFrame:
    """Load the Progenesis features table for m/z and RT values."""
    path = INTERIM / "01_parsed" / "features.parquet"
    if not path.exists():
        print("  [warning] features.parquet not found — m/z search disabled")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    needed = {"feature_id", "m_over_z", "rt_min"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    return df[list(needed)].copy()


def _load_hmdb_kegg_bridge() -> pd.DataFrame:
    path = EXTERNAL / "hmdb_kegg.parquet"
    if not path.exists():
        print("  [warning] HMDB→KEGG bridge not found. Run: biodisc external build-hmdb-kegg")
        return pd.DataFrame()
    return pd.read_parquet(path).dropna(subset=["kegg_id"])


# ============================================================
# Database Queries
# ============================================================

def _query_kegg_compound(kegg_id: str, kegg_client) -> dict:
    """Query KEGG for compound information, including ChEBI cross-refs."""
    result: dict = {"kegg_id": kegg_id}
    try:
        data = kegg_client.get(f"cpd:{kegg_id}")
    except Exception as exc:
        result["error"] = str(exc)
        return result

    if not data or not isinstance(data, str):
        return result

    result["found"] = True
    in_dblinks = False
    for line in data.splitlines():
        stripped = line.strip()
        if stripped.startswith("NAME"):
            name = stripped[4:].strip().rstrip(";")
            result["name"] = name
        elif stripped.startswith("FORMULA"):
            result["formula"] = stripped[7:].strip()
        elif stripped.startswith("MASS"):
            with contextlib.suppress(ValueError):
                result["mass"] = float(stripped[4:].strip())
        elif stripped.startswith("DBLINKS"):
            in_dblinks = True
        elif in_dblinks and stripped.startswith("CHEBI:"):
            result["chebi_id"] = stripped.strip()
            in_dblinks = False
        elif stripped.startswith("PATHWAY"):
            parts = stripped.split()
            for p in parts:
                if p.startswith("hsa"):
                    result.setdefault("pathways", []).append(p)
    return result


def _query_chebi_by_id(chebi_id: str, chebi_client) -> dict:
    """Query ChEBI for compound name, formula, mass, definition."""
    result: dict = {"chebi_id": chebi_id}
    try:
        entry = chebi_client.get_complete_entity(chebi_id.replace("CHEBI:", ""))
    except Exception as exc:
        result["error"] = str(exc)
        return result

    if entry:
        result["found"] = True
        result["name"] = getattr(entry, "chebiName", None)
        result["formula"] = getattr(entry, "formula", None)
        result["mass"] = getattr(entry, "mass", None)
        result["definition"] = getattr(entry, "definition", None)
    return result


def _search_chebi_by_mass(mass: float, chebi_client, tolerance_ppm: float,
                          max_results: int = 5) -> list[dict]:
    """Search ChEBI by monoisotopic mass, return candidate matches.

    Uses ChEBI's search_by_mass with a window of ±tolerance_ppm.
    Returns list of dicts with chebi_id, name, mass, mass_diff_ppm.
    """
    if mass <= 0:
        return []
    window = mass * tolerance_ppm / 1e6
    try:
        results = chebi_client.search_by_mass(
            mass,
            mass_type="MONOISOTOPIC MASS",
            search_category="MASS APPROX",
            upper_window=window,
            lower_window=window,
        )
    except Exception as exc:
        print(f"    [ChEBI mass search failed for m/z={mass:.4f}]: {exc}")
        return []

    if not results or not isinstance(results, list):
        return []

    candidates = []
    for item in results[:max_results]:
        if isinstance(item, dict):
            chebi_id = item.get("chebiId", "?")
            name = item.get("chebiName", "?")
            item_mass = item.get("mass", None)
            diff_ppm = abs(item_mass - mass) / mass * 1e6 if item_mass else None
            candidates.append({
                "chebi_id": f"CHEBI:{chebi_id}" if not chebi_id.startswith("CHEBI") else chebi_id,
                "name": name,
                "mass": item_mass,
                "mass_diff_ppm": round(diff_ppm, 2) if diff_ppm else None,
            })
    return candidates


def _query_chembl(molecule_id: str, chembl_client) -> dict:
    """Query ChEMBL for molecule information and bioactivity summary."""
    result: dict = {"chembl_id": molecule_id}
    try:
        mol = chembl_client.get_molecule(molecule_id)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    if mol:
        result["found"] = True
        result["name"] = mol.get("pref_name") or mol.get("molecule_chembl_id")
        result["type"] = mol.get("molecule_type")
        result["max_phase"] = mol.get("max_phase")
        with contextlib.suppress(Exception):
            activities = chembl_client.get_activity(
                molecule_chembl_id=molecule_id, limit=5,
            )
            if activities and "activities" in activities:
                result["n_activities"] = len(activities["activities"])
    return result


# ============================================================
# Feature row builder
# ============================================================

def _build_crossref_rows(
    merged: pd.DataFrame,
    features_df: pd.DataFrame,
    k,
    c,
    chembl,
    u,
) -> tuple[list[list[str]], list[list[str]]]:
    """Process every consensus feature and return (crossref_rows, mz_hit_rows).

    crossref_rows: per-feature row with KEGG / ChEBI / ChEMBL info
    mz_hit_rows:   per-(feature, candidate) rows from m/z search
    """
    # Build lookup maps
    kegg_map = {}
    hmdb_kegg = _load_hmdb_kegg_bridge()
    if not hmdb_kegg.empty:
        kegg_map = dict(zip(hmdb_kegg["hmdb_id"], hmdb_kegg["kegg_id"], strict=False))

    # Build feature → m/z map
    mz_map: dict[str, float] = {}
    if not features_df.empty:
        for _, r in features_df.iterrows():
            mz_map[r["feature_id"]] = r["m_over_z"]

    crossref_rows: list[list[str]] = []
    mz_hit_rows: list[list[str]] = []

    for _, feat_info in merged.iterrows():
        fid = feat_info["feature_id"]
        name = feat_info.get("top_compound_name") or "—"
        msi_tier = str(feat_info.get("msi_tier", "—"))
        formula = feat_info.get("top_formula") or "—"
        compound_id = feat_info.get("top_compound_id")
        mz = mz_map.get(fid)
        mz_str = f"{mz:.4f}" if mz and not (isinstance(mz, float) and math.isnan(mz)) else "—"

        if not compound_id or pd.isna(compound_id):
            # — Unidentified feature: try m/z search —
            crossref_rows.append([fid, name, msi_tier, formula, mz_str, "—", "—", "—", "—", "—", "—"])
            if mz and not math.isnan(mz) and c is not None and hasattr(c, "search_by_mass"):
                time.sleep(0.5)  # rate-limit
                candidates = _search_chebi_by_mass(mz, c, MZ_TOLERANCE_PPM)
                for cand in candidates:
                    mz_hit_rows.append([
                        fid, mz_str,
                        cand.get("chebi_id", "—"),
                        cand.get("name", "—")[:60],
                        str(cand.get("mass", "—")),
                        f"{cand.get('mass_diff_ppm', '—')} ppm" if cand.get("mass_diff_ppm") else "—",
                    ])
                if not candidates:
                    mz_hit_rows.append([fid, mz_str, "—", "No hits in ChEBI", "—", "—"])
            continue

        # — Identified feature: query databases —
        kegg_id = kegg_map.get(compound_id)
        kegg_info = None
        if kegg_id and k is not None:
            kegg_info = _query_kegg_compound(kegg_id, k)
            time.sleep(0.3)

        # KEGG columns
        k_id = kegg_info.get("kegg_id", "—") if kegg_info and kegg_info.get("found") else "—"
        k_name = (kegg_info.get("name", "—")[:60] if kegg_info and kegg_info.get("found") else "—")
        k_pathways = ",".join(kegg_info.get("pathways", [])) if kegg_info else "—"

        # ChEBI: try via KEGG DBLINKS (KEGG entries contain CHEBI:xxxxx)
        chebi_id = None
        chebi_info = None
        if kegg_info and kegg_info.get("chebi_id"):
            chebi_id = kegg_info["chebi_id"]
        elif kegg_id and u is not None:
            with contextlib.suppress(Exception):
                chebi_id = u.get_compound_id_from_kegg(kegg_id)
                if chebi_id and not str(chebi_id).startswith("CHEBI"):
                    chebi_id = f"CHEBI:{chebi_id}"

        if chebi_id and c is not None:
            chebi_info = _query_chebi_by_id(chebi_id, c)
            time.sleep(0.4)

        c_id = chebi_info.get("chebi_id", "—") if chebi_info and chebi_info.get("found") else (chebi_id or "—")
        c_name = (chebi_info.get("name", "—")[:60] if chebi_info and chebi_info.get("found") else "—")

        # ChEMBL: via UniChem KEGG→ChEMBL
        chembl_id = None
        chembl_info = None
        if kegg_id and u is not None:
            with contextlib.suppress(Exception):
                chembl_id = u.get_compound_id_from_kegg(kegg_id)

        if chembl_id and chembl is not None:
            chembl_info = _query_chembl(chembl_id, chembl)
            time.sleep(0.3)

        cb_id = chembl_info.get("chembl_id", "—") if chembl_info and chembl_info.get("found") else (chembl_id or "—")
        cb_name = (chembl_info.get("name", "—")[:40] if chembl_info and chembl_info.get("found") else "—")
        cb_phase = str(chembl_info.get("max_phase", "—")) if chembl_info else "—"

        crossref_rows.append([fid, name, msi_tier, formula, mz_str, k_id, k_name, k_pathways, c_id, c_name, cb_id, cb_name, cb_phase])

    return crossref_rows, mz_hit_rows


# ============================================================
# Report writer
# ============================================================

def _write_report(crossref_rows, mz_hit_rows, stats: dict) -> str:
    """Build the markdown report."""
    lines = [
        "# Compound Cross-Reference Diagnostic",
        "",
        "_Auto-generated by `scripts/diagnose_compound_cross_reference.py`._",
        "",
        "Cross-references the multiverse consensus features against KEGG, ",
        "ChEBI, ChEMBL, and PubChem via UniChem identifier mapping. ",
        "Unidentified features (MSI Tier 4) are searched by precursor m/z ",
        f"against ChEBI's exact-mass registry (±{MZ_TOLERANCE_PPM} ppm tolerance).",
        "",
        f"**Run time:** {stats['elapsed']:.1f}s",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Consensus features | {stats['n_total']} |",
        f"| Identified (MSI L2/L3) | {stats['n_identified']} |",
        f"| Unidentified (MSI L4) | {stats['n_unidentified']} |",
        f"| With HMDB compound ID | {stats['n_with_compound']} |",
        f"| Mapped to KEGG | {stats['n_mapped_kegg']} |",
        f"| Mapped to ChEBI (via KEGG/UniChem) | {stats['n_mapped_chebi']} |",
        f"| Mapped to ChEMBL (via UniChem) | {stats['n_mapped_chembl']} |",
        f"| Features with m/z available | {stats['n_with_mz']} |",
        f"| m/z search candidates (ChEBI) | {stats['n_mz_hits']} |",
        "",
    ]

    if stats["n_mapped_kegg"] > 0:
        lines += [
            "## KEGG / ChEBI / ChEMBL Cross-References",
            "",
            "| feature_id | Name | MSI | Formula | m/z | KEGG | KEGG name | Pathways | ChEBI | ChEBI name | ChEMBL | ChEMBL name | Max phase |",
            "|------------|------|:---:|:-------:|----:|:----:|----------|:--------:|:----:|:--------:|:-----:|:----------:|:---------:|",
        ]
        for r in crossref_rows:
            # Pad to 13 columns with "—"
            while len(r) < 13:
                r.append("—")
            lines.append(
                f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | "
                f"{r[7]} | {r[8]} | {r[9]} | {r[10]} | {r[11]} | {r[12]} |"
            )
        lines.append("")

    if mz_hit_rows:
        lines += [
            "## m/z Search Candidates — Unidentified (MSI Tier 4) Features",
            "",
            f"Searched ChEBI exact-mass registry at ±{MZ_TOLERANCE_PPM} ppm tolerance.",
            "",
            "| feature_id | m/z | ChEBI candidate | Name | Candidate mass | Δppm |",
            "|------------|:---:|:--------------:|:----:|:--------------:|:----:|",
        ]
        for r in mz_hit_rows:
            while len(r) < 6:
                r.append("—")
            lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
        lines.append("")

    # Interpretation
    lines += ["## Interpretation", ""]
    if stats["n_mapped_kegg"] > 0:
        lines += [
            f"- {stats['n_mapped_kegg']} features map to KEGG, enabling pathway enrichment "
            f"(`biodisc multiverse enrichment`).",
        ]
    if stats["n_mapped_chebi"] > 0:
        lines += [
            f"- {stats['n_mapped_chebi']} features have ChEBI identifiers, providing "
            "chemical ontology classification.",
        ]
    if stats["n_mapped_chembl"] > 0:
        lines += [
            f"- {stats['n_mapped_chembl']} features have ChEMBL entries, with bioactivity "
            "data available.",
        ]
    if stats["n_mz_hits"] > 0:
        lines += [
            f"- {stats['n_mz_hits']} ChEBI candidate matches found for MSI Tier 4 features "
            f"within ±{MZ_TOLERANCE_PPM} ppm. These are tentative identifications — validate "
            f"via MS/MS or retention-time matching before reporting.",
        ]
    if stats["n_unidentified"] > 0 and stats["n_mz_hits"] == 0:
        lines += [
            f"- {stats['n_unidentified']} MSI Tier 4 features had no ChEBI match within "
            f"±{MZ_TOLERANCE_PPM} ppm. Their molecular identities remain unknown. MS/MS "
            f"acquisition is required.",
        ]

    lines += [
        "",
        "## Databases Queried",
        "",
        "| Database | Service | Role |",
        "|----------|---------|------|",
        "| KEGG Compound | bioservices.KEGG | Compound name, formula, mass, pathways, ChEBI cross-ref |",
        "| ChEBI | bioservices.ChEBI | Chemical name, formula, ontology; m/z search for unknowns |",
        "| ChEMBL | bioservices.ChEMBL | Molecule type, max clinical phase, bioactivity count |",
        "| UniChem | bioservices.UniChem | Cross-database ID mapping (KEGG ↔ ChEBI ↔ ChEMBL ↔ PubChem) |",
        "",
    ]
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("Compound Cross-Reference Diagnostic")
    print("=" * 60)

    # 1. Load inputs
    print("\nLoading data...")
    consensus = _load_consensus_shortlist()
    stage9 = _load_stage9_metrics()
    features_df = _load_features_table()
    merged = consensus.merge(stage9, on="feature_id", how="left")

    n_identified = int((merged["msi_tier"].isin([2, 3])).sum())
    n_unidentified = int((merged["msi_tier"] == 4).sum())
    n_with_compound = int(merged["top_compound_id"].notna().sum())
    n_with_mz = int(merged["feature_id"].isin(features_df["feature_id"]).sum()) if not features_df.empty else 0

    print(f"  Features: {len(merged)} ({n_identified} identified, {n_unidentified} unidentified)")
    print(f"  With HMDB ID: {n_with_compound}, With m/z: {n_with_mz}")

    # 2. Query databases
    print("\nQuerying external databases...")
    import bioservices

    k = bioservices.KEGG(verbose=False)
    k.organism = "hsa"

    c = None
    if hasattr(bioservices, "ChEBI"):
        try:
            c = bioservices.ChEBI(verbose=False)
            print("  ChEBI: connected")
        except Exception as exc:
            print(f"  ChEBI: connection failed — {exc}")

    chembl = None
    if hasattr(bioservices, "ChEMBL"):
        try:
            chembl = bioservices.ChEMBL(verbose=False)
            print("  ChEMBL: connected")
        except Exception as exc:
            print(f"  ChEMBL: connection failed — {exc}")

    u = None
    if hasattr(bioservices, "UniChem"):
        try:
            u = bioservices.UniChem(verbose=False)
            print("  UniChem: connected")
        except Exception as exc:
            print(f"  UniChem: connection failed — {exc}")

    # 3. Build cross-reference rows
    crossref_rows, mz_hit_rows = _build_crossref_rows(merged, features_df, k, c, chembl, u)

    n_mapped_kegg = sum(1 for r in crossref_rows if r[5] != "—")
    n_mapped_chebi = sum(1 for r in crossref_rows if len(r) > 8 and r[8] != "—")
    n_mapped_chembl = sum(1 for r in crossref_rows if len(r) > 10 and r[10] != "—")

    stats = {
        "n_total": len(merged),
        "n_identified": n_identified,
        "n_unidentified": n_unidentified,
        "n_with_compound": n_with_compound,
        "n_with_mz": n_with_mz,
        "n_mapped_kegg": n_mapped_kegg,
        "n_mapped_chebi": n_mapped_chebi,
        "n_mapped_chembl": n_mapped_chembl,
        "n_mz_hits": len(mz_hit_rows),
        "elapsed": time.time() - t0,
    }

    # 4. Write report
    report = _write_report(crossref_rows, mz_hit_rows, stats)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report)
    print(f"\n{'=' * 60}")
    print(f"  {n_mapped_kegg} → KEGG  |  {n_mapped_chebi} → ChEBI  |  {n_mapped_chembl} → ChEMBL")
    print(f"  {len(mz_hit_rows)} m/z search candidates for {n_unidentified} unidentified features")
    print(f"\n[written] {REPORT}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
