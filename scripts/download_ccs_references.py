#!/usr/bin/env python3
"""Download gold-standard CCS reference datasets for Stage 8.

Sources downloaded by default (small, fully open):
  1. Picache 2019 main + supplementary (RSC, ~3,800 experimental CCS)
  2. Hines 2017            (ACS, ~450 lipid CCS)              [may need manual]
  3. Paglia 2014           (ACS, ~120 metabolite CCS)         [may need manual]

Optional (large, opt-in via --include-hmdb):
  4. HMDB metabolites XML  (~150 MB compressed, ~5 GB unzipped)

Usage:
    .venv/bin/python scripts/download_ccs_references.py
    .venv/bin/python scripts/download_ccs_references.py --force          # re-download
    .venv/bin/python scripts/download_ccs_references.py --include-hmdb   # also fetch HMDB

Outputs:
    data/external/raw/<source>.{xlsx,zip}
    data/external/raw/_manifest.json   (sha256 + size + timestamp + source URL)

For sources where automated download fails (typically ACS papers blocking the
publisher's domain on bots), the script prints precise manual download steps
and continues with the rest. Re-run after manual placement to update the
manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEST_DIR = PROJECT_ROOT / "data" / "external" / "raw"
MANIFEST_PATH = DEST_DIR / "_manifest.json"

# A realistic UA helps with some publishers; harmless for open ones.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
    "biodisc-CCS-downloader/0.1 (academic use)"
)
TIMEOUT_SEC = 120

# Terminal colors
GREEN, RED, YELLOW, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m",
)


@dataclass
class CCSSource:
    name: str
    citation: str
    primary_url: str
    output_filename: str
    expected_min_size: int
    description: str
    optional: bool = False
    manual_instructions: str = ""


# Ordered: small/reliable first, large/maybe-manual last.
SOURCES: list[CCSSource] = [
    CCSSource(
        name="mclean_compendium",
        citation="Picache et al. 2019, Chem Sci 10:983, doi:10.1039/C8SC04396E "
                 "(distributed as allMASTER.csv via McLean lab GitHub)",
        primary_url=(
            "https://raw.githubusercontent.com/McLeanResearchGroup/"
            "CCS-Compendium/master/Shiny/data/allMASTER.csv"
        ),
        output_filename="mclean_allMASTER.csv",
        expected_min_size=1_000_000,
        description="McLean Unified CCS Compendium — 3,727 experimental CCS rows "
                    "(~2,300 unique compounds across multiple adducts)",
    ),
    CCSSource(
        name="picache_2019_main",
        citation="Picache et al. 2019, Chem Sci 10:983, doi:10.1039/C8SC04396E",
        primary_url="https://www.rsc.org/suppdata/c8/sc/c8sc04396e/c8sc04396e2.xlsx",
        output_filename="picache_2019_main.xlsx",
        expected_min_size=40_000,
        description="Picache 2019 RSC supplement — submission TEMPLATE (not data; "
                    "kept for citation completeness)",
    ),
    CCSSource(
        name="picache_2019_supp",
        citation="Picache et al. 2019, Chem Sci 10:983, doi:10.1039/C8SC04396E",
        primary_url="https://www.rsc.org/suppdata/c8/sc/c8sc04396e/c8sc04396e1.xlsx",
        output_filename="picache_2019_supp.xlsx",
        expected_min_size=20_000,
        description="Picache 2019 supplementary tables (figures + secondary data)",
    ),
    CCSSource(
        name="hines_2017_lipids",
        citation="Hines et al. 2017, Anal Chem 89:9023, doi:10.1021/acs.analchem.7b01390",
        primary_url=(
            "https://pubs.acs.org/doi/suppl/10.1021/acs.analchem.7b01390/"
            "suppl_file/ac7b01390_si_001.xlsx"
        ),
        output_filename="hines_2017_lipids.xlsx",
        expected_min_size=10_000,
        description="~450 lipid CCS, multi-adduct (TWIMS-relevant)",
        manual_instructions=(
            "1. Open https://pubs.acs.org/doi/10.1021/acs.analchem.7b01390 in a browser\n"
            "2. Click the 'Supporting Information' link (usually right side of page)\n"
            "3. Download the Excel SI file (filename ends in _si_001.xlsx)\n"
            "4. Save it as: data/external/raw/hines_2017_lipids.xlsx\n"
            "5. Re-run this script — it will recognize the file and update the manifest."
        ),
    ),
    CCSSource(
        name="paglia_2014",
        citation="Paglia et al. 2014, Anal Chem 86:3985, doi:10.1021/ac500405x",
        primary_url=(
            "https://pubs.acs.org/doi/suppl/10.1021/ac500405x/"
            "suppl_file/ac500405x_si_001.xlsx"
        ),
        output_filename="paglia_2014_ccs.xlsx",
        expected_min_size=8_000,
        description="~120 metabolite CCS — foundational reference for metabolomics",
        manual_instructions=(
            "1. Open https://pubs.acs.org/doi/10.1021/ac500405x in a browser\n"
            "2. Click the 'Supporting Information' link\n"
            "3. Download the Excel SI file (filename ends in _si_001.xlsx)\n"
            "4. Save it as: data/external/raw/paglia_2014_ccs.xlsx\n"
            "5. Re-run this script."
        ),
    ),
    CCSSource(
        name="hmdb_metabolites",
        citation="Wishart et al. 2022, Nucleic Acids Res 50:D622 (HMDB v5)",
        primary_url="https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip",
        output_filename="hmdb_metabolites.zip",
        expected_min_size=100_000_000,  # ~150 MB compressed
        description="HMDB metabolites XML — bundles experimental + AllCCS2-predicted CCS",
        optional=True,
        manual_instructions=(
            "1. Open https://hmdb.ca/downloads in a browser\n"
            "2. Find 'All Metabolites' under 'XML download'\n"
            "3. Download hmdb_metabolites.zip (~150 MB)\n"
            "4. Save it as: data/external/raw/hmdb_metabolites.zip\n"
            "5. Re-run this script."
        ),
    ),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict[str, dict]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict[str, dict]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def update_manifest_entry(
    manifest: dict, src: CCSSource, path: Path, status: str, note: str = ""
) -> None:
    entry = {
        "status": status,
        "url": src.primary_url,
        "citation": src.citation,
        "filename": src.output_filename,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if path.exists():
        entry["size_bytes"] = path.stat().st_size
        entry["sha256"] = sha256_of(path)
    if note:
        entry["note"] = note
    manifest[src.name] = entry


def download_one(src: CCSSource, *, force: bool) -> tuple[str, str]:
    """Returns (status, message) where status ∈ {ok, skip, fail, manual}."""
    out_path = DEST_DIR / src.output_filename

    # Idempotent: skip if file present and looks valid
    if out_path.exists() and not force:
        size = out_path.stat().st_size
        if size >= src.expected_min_size:
            return "skip", f"already present ({size / 1024 / 1024:.2f} MB)"
        return "fail", f"existing file too small ({size} B); rerun with --force"

    # Attempt download
    try:
        req = urllib.request.Request(
            src.primary_url, headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        return "manual", f"HTTP {e.code} ({e.reason}) — see manual instructions"
    except urllib.error.URLError as e:
        return "manual", f"URLError: {e.reason} — see manual instructions"
    except (TimeoutError, ConnectionResetError) as e:
        return "manual", f"network: {type(e).__name__} — see manual instructions"

    # Validate size
    if len(data) < src.expected_min_size:
        return "fail", (
            f"downloaded file too small ({len(data)} B; expected >= "
            f"{src.expected_min_size} B) — likely a redirect/HTML error page"
        )

    out_path.write_bytes(data)
    return "ok", f"downloaded {len(data) / 1024 / 1024:.2f} MB → {out_path.name}"


def status_glyph(status: str) -> str:
    return {
        "ok":     f"{GREEN}✓{RESET}",
        "skip":   f"{YELLOW}⊙{RESET}",
        "fail":   f"{RED}✗{RESET}",
        "manual": f"{YELLOW}!{RESET}",
    }.get(status, "?")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if file already exists locally",
    )
    parser.add_argument(
        "--include-hmdb", action="store_true",
        help="Also download HMDB metabolites XML (~150 MB; opt-in)",
    )
    args = parser.parse_args()

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    sources = [s for s in SOURCES if not s.optional or args.include_hmdb]

    print(f"{BOLD}Downloading gold-standard CCS reference datasets{RESET}")
    print(f"  Destination: {DEST_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  Force re-download: {args.force}")
    print(f"  HMDB included: {args.include_hmdb}")
    print(f"  Sources to attempt: {len(sources)}")
    print("=" * 72)

    results: list[tuple[CCSSource, str, str]] = []
    for src in sources:
        print(f"\n{BOLD}{src.name}{RESET}")
        print(f"  {src.description}")
        print(f"  Citation: {src.citation}")
        print(f"  URL: {src.primary_url}")

        status, msg = download_one(src, force=args.force)
        print(f"  {status_glyph(status)} {msg}")

        # Manual fallback?
        if status == "manual" and src.manual_instructions:
            print(f"  {YELLOW}Manual download instructions:{RESET}")
            for line in src.manual_instructions.splitlines():
                print(f"    {line}")

        # Update manifest if file ended up on disk for any reason
        out_path = DEST_DIR / src.output_filename
        if out_path.exists():
            ok_size = out_path.stat().st_size >= src.expected_min_size
            entry_status = "ok" if ok_size else "invalid_size"
            update_manifest_entry(manifest, src, out_path, entry_status, note=msg)

        results.append((src, status, msg))

    save_manifest(manifest)

    # Summary
    n_ok     = sum(1 for _, s, _ in results if s == "ok")
    n_skip   = sum(1 for _, s, _ in results if s == "skip")
    n_manual = sum(1 for _, s, _ in results if s == "manual")
    n_fail   = sum(1 for _, s, _ in results if s == "fail")
    n_have   = sum(
        1 for s in sources
        if (DEST_DIR / s.output_filename).exists()
        and (DEST_DIR / s.output_filename).stat().st_size >= s.expected_min_size
    )

    print()
    print("=" * 72)
    print(f"{BOLD}Summary{RESET}")
    print(f"  Just downloaded:        {GREEN}{n_ok}{RESET}")
    print(f"  Already present:        {YELLOW}{n_skip}{RESET}")
    print(f"  Need manual download:   {YELLOW}{n_manual}{RESET}  (instructions printed above)")
    print(f"  Other failures:         {RED}{n_fail}{RESET}")
    print(f"  Files present + valid:  {GREEN}{n_have}/{len(sources)}{RESET}")
    print(f"  Manifest:               {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")

    if n_have == len(sources):
        print(f"\n{GREEN}{BOLD}All sources present.{RESET} Stage 8 unification can proceed.")
        return 0
    print(
        f"\n{YELLOW}{n_manual + n_fail} source(s) still missing.{RESET} "
        "Follow manual steps above, then re-run this script to update the manifest."
    )
    # Non-zero exit only if real failures (manual is expected for ACS)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
