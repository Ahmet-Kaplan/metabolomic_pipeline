"""KEGG pathway enrichment analysis for multiverse consensus features.

Takes the consensus shortlist from Phase 2 and computes over-representation
of KEGG pathways using a hypergeometric test (one-sided Fisher's exact).

Data flow
---------
1. Read consensus shortlist features (consensus_passed.parquet)
2. Join with Stage 9 metrics to get HMDB compound IDs
3. Map HMDB IDs → KEGG Compound IDs via hmdb_kegg.parquet bridge
4. Query KEGG REST API for each compound's pathway membership
5. Compute enrichment p-values (hypergeometric / Fisher's exact)
6. BH-FDR correct across pathways
7. Write enrichment table + summary report

Reference
---------
KEGG API: https://www.kegg.jp/kegg/rest/keggapi.html
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# --------------- pathway name mapping for common metabolic pathways ---------------
# These supplement the KEGG API response with readable names since the API
# returns short descriptions that may need disambiguation.

_ORGANISM_CODE = "hsa"  # Homo sapiens KEGG organism prefix


# ============================================================
# Data structures
# ============================================================
@dataclass(frozen=True)
class EnrichmentResult:
    """Output of KEGG pathway enrichment analysis."""

    table: pd.DataFrame
    n_foreground_features: int
    n_foreground_mapped_to_kegg: int
    n_background_kegg_compounds: int
    n_pathways_tested: int
    n_pathways_enriched_at_q05: int
    runtime_seconds: float


# ============================================================
# Load data
# ============================================================
def _load_consensus_features(consensus_dir: Path) -> pd.DataFrame:
    """Load consensus-passed features with their feature IDs."""
    path = consensus_dir / "consensus_passed.parquet"
    if not path.exists():
        log.warning("consensus_passed_not_found", path=str(path))
        return pd.DataFrame()
    df = pd.read_parquet(path)
    log.info("loaded_consensus_features", n=len(df))
    return df


def _load_stage9_metrics(interim_dir: Path) -> pd.DataFrame:
    """Load Stage 9 ID ranking metrics to get HMDB IDs per feature."""
    path = interim_dir / "09_id_ranking" / "metrics.parquet"
    if not path.exists():
        log.warning("stage9_metrics_not_found", path=str(path))
        return pd.DataFrame()
    df = pd.read_parquet(path)
    needed = {"feature_id", "top_compound_id", "outcome", "msi_tier"}
    missing = needed - set(df.columns)
    if missing:
        log.warning("stage9_metrics_missing_columns", missing=sorted(missing))
        return pd.DataFrame()
    log.info("loaded_stage9_metrics", n=len(df))
    return df[["feature_id", "top_compound_id", "outcome", "msi_tier"]].copy()


def _load_hmdb_kegg_bridge(hmdb_kegg_path: Path) -> pd.DataFrame:
    """Load the HMDB → KEGG bridge table."""
    if not hmdb_kegg_path.exists():
        log.warning("hmdb_kegg_bridge_not_found", path=str(hmdb_kegg_path))
        return pd.DataFrame()
    df = pd.read_parquet(hmdb_kegg_path)
    df = df.dropna(subset=["kegg_id"])
    log.info("loaded_hmdb_kegg_bridge", n=len(df), n_unique_kegg=df["kegg_id"].nunique())
    return df


# ============================================================
# KEGG API queries (via bioservices)
# ============================================================
def _compound_to_pathways(kegg_id: str, kegg_client: Any) -> list[str]:
    """Return list of KEGG pathway IDs containing a given compound.

    Uses bioservices KEGG client. If the compound map is not found
    (e.g. discontinued KEGG ID), returns empty list.
    """
    try:
        data = kegg_client.get(f"cpd:{kegg_id}")
    except Exception as exc:
        log.warning("kegg_query_failed", kegg_id=kegg_id, error=str(exc))
        return []

    if data is None or not isinstance(data, str):
        return []

    pathways: list[str] = []
    for line in data.splitlines():
        stripped = line.strip()
        # KEGG pathway entries look like: PATHWAY  hsa00010  Glycolysis
        if stripped.startswith("PATHWAY") or stripped.startswith("  PATHWAY"):
            parts = stripped.split()
            for p in parts:
                if p.startswith(f"{_ORGANISM_CODE}"):
                    pathways.append(p)
    return pathways


def _fetch_compound_name(kegg_id: str, kegg_client: Any) -> str | None:
    """Fetch the compound name from KEGG for display purposes."""
    try:
        data = kegg_client.get(f"cpd:{kegg_id}")
    except Exception:
        return None
    if data is None or not isinstance(data, str):
        return None
    for line in data.splitlines():
        # First NAME line after the entry header
        if line.startswith("NAME"):
            name = line[4:].strip()
            if name.endswith(";"):
                name = name[:-1]
            return name
    return None


def _fetch_pathway_name(pathway_id: str, kegg_client: Any) -> str:
    """Fetch the pathway title from KEGG."""
    try:
        data = kegg_client.get(pathway_id)
    except Exception:
        return pathway_id
    if data is None or not isinstance(data, str):
        return pathway_id
    for line in data.splitlines():
        if line.startswith("NAME"):
            name = line[4:].strip()
            if name.endswith(";"):
                name = name[:-1]
            return name
    return pathway_id


def _fetch_background_pathway_sizes(
    pathway_ids: set[str], kegg_client: Any,
) -> dict[str, int]:
    """Fetch the number of compounds in each KEGG pathway (the pathway size)."""
    sizes: dict[str, int] = {}
    for pid in sorted(pathway_ids):
        try:
            data = kegg_client.get(pid)
        except Exception:
            continue
        if data is None or not isinstance(data, str):
            continue
        # Count COMPOUND entries
        compound_count = 0
        in_compound = False
        for line in data.splitlines():
            stripped = line.strip()
            if stripped.startswith("COMPOUND"):
                in_compound = True
                compound_count += 1
            elif in_compound:
                if stripped.startswith("(") or stripped == "":
                    in_compound = False
                elif stripped and stripped[0].isalnum():
                    compound_count += 1
                else:
                    in_compound = False
        if compound_count > 0:
            sizes[pid] = compound_count
    return sizes


# ============================================================
# Enrichment test
# ============================================================
def _hypergeom_pval(
    k: int,  # foreground hits in this pathway
    pop_size: int,  # background total compounds (population)
    sample_size: int,  # foreground total compounds (sample)
    n_success_pop: int,  # pathway size (successes in population)
) -> float:
    """One-sided hypergeometric (Fisher's exact) p-value for over-representation.

    P(X >= k) where X ~ Hypergeometric(pop_size, n_success_pop, sample_size).
    Uses the survival function (sf) of the hypergeometric distribution.
    """
    if k <= 0:
        return 1.0
    from scipy.stats import hypergeom

    p = hypergeom.sf(k - 1, pop_size, n_success_pop, sample_size)
    return float(p)


# ============================================================
# Main analysis
# ============================================================
def run_enrichment(
    consensus_dir: Path,
    interim_dir: Path,
    hmdb_kegg_path: Path,
    kegg_client: Any | None = None,
    q_threshold: float = 0.05,
    min_compound_overlap: int = 1,
) -> EnrichmentResult:
    """Run KEGG pathway enrichment on the multiverse consensus shortlist.

    Parameters
    ----------
    consensus_dir
        Path to data/reports/multiverse/.
    interim_dir
        Path to data/interim/.
    hmdb_kegg_path
        Path to hmdb_kegg.parquet (HMDB→KEGG bridge).
    kegg_client
        Pre-configured bioservices KEGG client. If None, creates one.
    q_threshold
        BH-FDR threshold for reporting enriched pathways.
    min_compound_overlap
        Minimum number of consensus compounds in a pathway to test it.

    Returns
    -------
    EnrichmentResult with enrichment table.
    """
    t0 = time.time()

    # 1. Load inputs
    consensus = _load_consensus_features(consensus_dir)
    if consensus.empty:
        return EnrichmentResult(
            table=pd.DataFrame(), n_foreground_features=0,
            n_foreground_mapped_to_kegg=0, n_background_kegg_compounds=0,
            n_pathways_tested=0, n_pathways_enriched_at_q05=0,
            runtime_seconds=time.time() - t0,
        )

    stage9 = _load_stage9_metrics(interim_dir)
    hmdb_kegg = _load_hmdb_kegg_bridge(hmdb_kegg_path)

    if stage9.empty or hmdb_kegg.empty:
        return EnrichmentResult(
            table=pd.DataFrame(), n_foreground_features=len(consensus),
            n_foreground_mapped_to_kegg=0, n_background_kegg_compounds=0,
            n_pathways_tested=0, n_pathways_enriched_at_q05=0,
            runtime_seconds=time.time() - t0,
        )

    # 2. Map consensus features → HMDB IDs → KEGG IDs
    merged = consensus.merge(stage9, on="feature_id", how="left")
    has_compound = merged["top_compound_id"].notna()
    n_with_compound = int(has_compound.sum())

    compound_to_kegg = dict(zip(hmdb_kegg["hmdb_id"], hmdb_kegg["kegg_id"], strict=False))
    merged["kegg_id"] = merged["top_compound_id"].map(compound_to_kegg)
    has_kegg = merged["kegg_id"].notna()
    foreground_kegg = merged.loc[has_kegg, "kegg_id"].unique().tolist()
    n_fg_kegg = len(foreground_kegg)

    log.info(
        "enrichment_input_summary",
        n_consensus_features=len(consensus),
        n_with_hmdb_compound=n_with_compound,
        n_mapped_to_kegg=n_fg_kegg,
    )

    if n_fg_kegg == 0:
        return EnrichmentResult(
            table=pd.DataFrame(), n_foreground_features=len(consensus),
            n_foreground_mapped_to_kegg=0, n_background_kegg_compounds=0,
            n_pathways_tested=0, n_pathways_enriched_at_q05=0,
            runtime_seconds=time.time() - t0,
        )

    # 3. Query KEGG API for pathway membership
    import bioservices

    k = kegg_client or bioservices.KEGG(verbose=False)

    # For each foreground KEGG compound, find its pathways
    compound_to_pathways_map: dict[str, list[str]] = {}
    for cpd_id in foreground_kegg:
        pathways = _compound_to_pathways(cpd_id, k)
        if pathways:
            compound_to_pathways_map[cpd_id] = pathways
            log.debug("kegg_compound_mapped", kegg_id=cpd_id, n_pathways=len(pathways))

    # Collect all unique pathways
    all_pathway_ids: set[str] = set()
    for pw_list in compound_to_pathways_map.values():
        all_pathway_ids.update(pw_list)

    # Fetch pathway names
    pathway_names: dict[str, str] = {}
    for pid in sorted(all_pathway_ids):
        pathway_names[pid] = _fetch_pathway_name(pid, k)

    # Fetch background pathway sizes
    pathway_sizes = _fetch_background_pathway_sizes(all_pathway_ids, k)

    # 4. Build background universe: all KEGG compounds in HMDB that map to human pathways
    #    We approximate the background as all KEGG compounds from HMDB that belong
    #    to any tested pathway.
    background_kegg_set = set(hmdb_kegg["kegg_id"].unique())
    n_background = len(background_kegg_set)

    # 5. Compute enrichment per pathway
    rows: list[dict[str, Any]] = []
    for pid in sorted(all_pathway_ids):
        pw_size = pathway_sizes.get(pid, 0)
        # Foreground: how many of our consensus compounds are in this pathway
        fg_in_pathway = sum(
            1 for pw_list in compound_to_pathways_map.values() if pid in pw_list
        )
        if fg_in_pathway < min_compound_overlap:
            continue
        pval = _hypergeom_pval(
            k=fg_in_pathway,
            pop_size=n_background,
            sample_size=n_fg_kegg,
            n_success_pop=pw_size,
        )
        fg_compounds = [
            cpd for cpd, pw_list in compound_to_pathways_map.items() if pid in pw_list
        ]
        rows.append({
            "pathway_id": pid,
            "pathway_name": pathway_names.get(pid, pid),
            "pathway_size": pw_size,
            "n_foreground_in_pathway": fg_in_pathway,
            "fg_fraction": fg_in_pathway / max(n_fg_kegg, 1),
            "p_value": pval,
            "foreground_kegg_ids": ",".join(fg_compounds),
        })

    if not rows:
        return EnrichmentResult(
            table=pd.DataFrame(), n_foreground_features=len(consensus),
            n_foreground_mapped_to_kegg=n_fg_kegg,
            n_background_kegg_compounds=n_background,
            n_pathways_tested=0, n_pathways_enriched_at_q05=0,
            runtime_seconds=time.time() - t0,
        )

    enrichment = pd.DataFrame(rows)

    # 6. BH-FDR correction
    n_tests = len(enrichment)
    if n_tests > 0:
        sort_idx = np.argsort(enrichment["p_value"].to_numpy())
        sorted_p = enrichment["p_value"].to_numpy()[sort_idx]
        ranks = np.arange(1, n_tests + 1)
        q_sorted = sorted_p * n_tests / ranks
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
        q_sorted = np.minimum(q_sorted, 1.0)
        q_vals = np.empty_like(q_sorted)
        q_vals[sort_idx] = q_sorted
        enrichment["q_value"] = q_vals
    else:
        enrichment["q_value"] = np.nan

    enrichment["enriched"] = enrichment["q_value"] <= q_threshold
    enrichment = enrichment.sort_values("q_value").reset_index(drop=True)

    n_enriched = int(enrichment["enriched"].sum())

    elapsed = time.time() - t0
    log.info(
        "kegg_enrichment_done",
        n_background_compounds=n_background,
        n_foreground_mapped=n_fg_kegg,
        n_pathways_tested=n_tests,
        n_enriched=n_enriched,
        q_threshold=q_threshold,
        elapsed_s=round(elapsed, 1),
    )

    return EnrichmentResult(
        table=enrichment,
        n_foreground_features=len(consensus),
        n_foreground_mapped_to_kegg=n_fg_kegg,
        n_background_kegg_compounds=n_background,
        n_pathways_tested=n_tests,
        n_pathways_enriched_at_q05=n_enriched,
        runtime_seconds=elapsed,
    )


# ============================================================
# Writer
# ============================================================
def write_enrichment(
    result: EnrichmentResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Persist enrichment results as parquet + CSV + HTML summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if result.table.empty:
        return paths

    parquet_path = output_dir / "kegg_enrichment.parquet"
    result.table.to_parquet(parquet_path, index=False, compression="zstd")
    paths["parquet"] = parquet_path

    csv_path = output_dir / "kegg_enrichment.csv"
    result.table.to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    # HTML summary
    n_enriched = result.n_pathways_enriched_at_q05
    enriched_tbl = result.table[result.table["enriched"]].copy() if n_enriched > 0 else result.table.head(20).copy()

    title = f"KEGG Pathway Enrichment — {result.n_foreground_features} consensus features"
    enriched_str = (
        f"<p>{n_enriched} pathways enriched at q &le; 0.05"
        f" ({result.n_pathways_tested} tested)</p>"
    )

    table_html = enriched_tbl.to_html(
        classes="dataframe",
        float_format=lambda x: f"{x:.4g}" if isinstance(x, float) else str(x),
        index=False,
        na_rep="—",
    )

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1200px; margin: 2em auto; padding: 0 1em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 6px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
  th {{ background-color: #f5f5f5; }}
  tr.enriched {{ background-color: #e8f5e9; }}
  .summary {{ margin-bottom: 2em; }}
</style></head>
<body>
<h1>{title}</h1>
<div class="summary">
  <p>Foreground: {result.n_foreground_features} consensus features
     → {result.n_foreground_mapped_to_kegg} KEGG compounds</p>
  <p>Background: {result.n_background_kegg_compounds} HMDB-to-KEGG mapped compounds</p>
  {enriched_str}
</div>
{table_html}
</body>
</html>"""

    html_path = output_dir / "kegg_enrichment.html"
    html_path.write_text(html)
    paths["html"] = html_path

    log.info(
        "kegg_enrichment_written",
        parquet=str(parquet_path),
        html=str(html_path),
    )
    return paths
