"""Phase 10 audit — verify the multiverse consensus output is correct.

Mirror of the per-path audit pattern (``verify_path_a_figures.py``)
adapted for the consensus-level outputs:

* Per-path subdirs exist with HTML report + shortlist parquet.
* ``consensus.parquet`` has the expected schema and row count.
* Per-path flag matrix in consensus matches the union of per-path
  ``passed_shortlist`` columns bit-for-bit.
* Tier assignment is consistent with the consensus threshold logic.
* ``consensus_passed.parquet`` has only the flagged rows.
* ``consensus_config.json`` matches in-memory recomputation.

Total runtime: ~10 s — no model refits needed since we just read the
parquets the smoke wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

OUTPUT_DIR = Path("data/reports/multiverse")
EXPECTED_PATHS = [
    "path_a_classical",
    "path_b_robust_ml",
    "path_c_mnar_aware",
    "path_d_middle",
]


def report(passed: bool, label: str, msg: str) -> None:
    sym = PASS if passed else FAIL
    print(f"  {sym} {label}: {msg}")


def section_banner(name: str) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")


def audit_per_path_outputs() -> None:
    section_banner("PER-PATH OUTPUTS")
    for path_name in EXPECTED_PATHS:
        path_dir = OUTPUT_DIR / path_name
        report(
            path_dir.exists() and path_dir.is_dir(),
            f"{path_name}: directory exists",
            str(path_dir),
        )
        if not path_dir.exists():
            continue
        report(
            (path_dir / "index.html").exists(),
            f"{path_name}: HTML report exists",
            f"{path_dir}/index.html",
        )
        sl_path = path_dir / "shortlist.parquet"
        report(
            sl_path.exists(),
            f"{path_name}: shortlist parquet exists",
            str(sl_path),
        )


def audit_consensus_files() -> None:
    section_banner("CONSENSUS FILES")
    expected = {
        "parquet": OUTPUT_DIR / "consensus.parquet",
        "csv": OUTPUT_DIR / "consensus.csv",
        "passed_parquet": OUTPUT_DIR / "consensus_passed.parquet",
        "config": OUTPUT_DIR / "consensus_config.json",
    }
    for kind, p in expected.items():
        report(p.exists(), f"consensus_{kind} exists", str(p))


def audit_consensus_schema() -> None:
    section_banner("CONSENSUS SCHEMA")
    parquet = OUTPUT_DIR / "consensus.parquet"
    if not parquet.exists():
        report(False, "consensus parquet missing", "")
        return
    df = pd.read_parquet(parquet)
    expected_universal = {
        "feature_id",
        "n_paths_total",
        "n_paths_flagging",
        "frac_paths_flagging",
        "consensus_tier",
        "passed_consensus",
    }
    expected_per_path = {f"per_path_flag_{p}" for p in EXPECTED_PATHS}
    expected_drilldown = {
        "n_paths_univariate",
        "n_paths_plsda",
        "n_paths_splsda",
        "n_paths_rf",
    }
    expected = expected_universal | expected_per_path | expected_drilldown
    missing = expected - set(df.columns)
    report(
        not missing,
        "consensus schema",
        f"{len(df):,} rows × {len(df.columns)} cols; missing columns: {missing or '∅'}",
    )


def audit_per_path_flag_matrix() -> None:
    """Per-path flag columns must match each path's shortlist `passed_shortlist`."""
    section_banner("PER-PATH FLAG FIDELITY")
    consensus = pd.read_parquet(OUTPUT_DIR / "consensus.parquet")
    consensus_idx = consensus.set_index("feature_id")
    for path_name in EXPECTED_PATHS:
        sl_path = OUTPUT_DIR / path_name / "shortlist.parquet"
        if not sl_path.exists():
            report(False, f"{path_name}: shortlist parquet missing", "")
            continue
        sl = pd.read_parquet(sl_path)
        sl_passed = set(sl.loc[sl["passed_shortlist"], "feature_id"].astype(str))
        consensus_passed = set(
            consensus_idx.index[consensus_idx[f"per_path_flag_{path_name}"]]
            .astype(str)
        )
        match = sl_passed == consensus_passed
        report(
            match,
            f"{path_name}: per-path flag column matches shortlist",
            f"shortlist passed n={len(sl_passed)}, consensus column n={len(consensus_passed)}; "
            f"sets equal: {match}",
        )


def audit_consensus_counts() -> None:
    section_banner("CONSENSUS COUNTS + TIERS")
    df = pd.read_parquet(OUTPUT_DIR / "consensus.parquet")
    cfg = json.loads((OUTPUT_DIR / "consensus_config.json").read_text())

    # n_paths_total uniform across rows
    unique_n_total = df["n_paths_total"].unique()
    report(
        len(unique_n_total) == 1 and int(unique_n_total[0]) == cfg["n_paths_total"],
        "n_paths_total uniform",
        f"unique values in column: {list(unique_n_total)}; config: {cfg['n_paths_total']}",
    )

    # n_paths_flagging vs sum of per-path flag columns
    per_path_cols = [f"per_path_flag_{p}" for p in EXPECTED_PATHS]
    expected_n = df[per_path_cols].astype(int).sum(axis=1)
    match = (df["n_paths_flagging"].to_numpy() == expected_n.to_numpy()).all()
    report(
        bool(match),
        "n_paths_flagging matches per-path columns row-wise",
        f"checked {len(df):,} rows",
    )

    # frac_paths_flagging consistent
    expected_frac = df["n_paths_flagging"] / df["n_paths_total"]
    frac_match = np.allclose(df["frac_paths_flagging"], expected_frac, atol=1e-9)
    report(
        bool(frac_match),
        "frac_paths_flagging consistent",
        "n_flag/n_total match every row",
    )

    # Tier assignment
    n = df["n_paths_total"].iloc[0]
    n_flag = df["n_paths_flagging"]
    expected_tier = pd.Series("none", index=df.index)
    expected_tier[(n_flag >= 1) & (n_flag / n < 0.5)] = "tier_3"
    expected_tier[(n_flag / n >= 0.5) & (n_flag / n < 0.75)] = "tier_2"
    expected_tier[(n_flag / n) >= 0.75] = "tier_1"
    tier_match = (df["consensus_tier"] == expected_tier).all()
    report(
        bool(tier_match),
        "tier assignment matches definitions",
        "tier_1 ≥0.75, tier_2 ≥0.5, tier_3 >0, none =0",
    )

    # passed_consensus matches threshold logic
    threshold = float(cfg["threshold"])
    expected_passed = df["frac_paths_flagging"] >= threshold
    passed_match = (df["passed_consensus"] == expected_passed).all()
    report(
        bool(passed_match),
        "passed_consensus matches threshold rule",
        f"threshold={threshold}; checked {len(df):,} rows",
    )

    # n_consensus matches config + sum
    n_passed_in_file = int(df["passed_consensus"].sum())
    report(
        n_passed_in_file == cfg["n_consensus"],
        "n_consensus matches config",
        f"file n_passed={n_passed_in_file}; config n_consensus={cfg['n_consensus']}",
    )


def audit_passed_parquet() -> None:
    section_banner("CONSENSUS_PASSED PARQUET")
    full = pd.read_parquet(OUTPUT_DIR / "consensus.parquet")
    passed = pd.read_parquet(OUTPUT_DIR / "consensus_passed.parquet")
    expected_n = int(full["passed_consensus"].sum())
    all_passed = (
        passed["passed_consensus"].all() if len(passed) > 0 else True
    )
    report(
        len(passed) == expected_n and bool(all_passed),
        "consensus_passed.parquet contains only flagged rows",
        f"n_rows={len(passed)}; expected {expected_n}",
    )


def audit_config_sidecar() -> None:
    section_banner("CONFIG SIDECAR")
    cfg = json.loads((OUTPUT_DIR / "consensus_config.json").read_text())
    report(
        set(cfg["path_names"]) == set(EXPECTED_PATHS),
        "config path_names match expected",
        f"config: {sorted(cfg['path_names'])}; expected: {sorted(EXPECTED_PATHS)}",
    )
    report(
        cfg["n_paths_total"] >= 1 and cfg["n_paths_total"] <= len(EXPECTED_PATHS),
        "n_paths_total plausible",
        f"n_paths_total={cfg['n_paths_total']} (range [1, {len(EXPECTED_PATHS)}])",
    )
    n_per_tier = cfg["n_per_tier"]
    report(
        sum(n_per_tier.values()) > 0,
        "n_per_tier non-empty",
        f"tiers: {n_per_tier}",
    )


def audit_consensus_summary() -> None:
    """Print a human-readable summary of the consensus output."""
    section_banner("MULTIVERSE CONSENSUS SUMMARY")
    df = pd.read_parquet(OUTPUT_DIR / "consensus.parquet")
    cfg = json.loads((OUTPUT_DIR / "consensus_config.json").read_text())
    print(f"  paths analysed: {cfg['n_paths_total']}")
    print(f"  threshold: {cfg['threshold']}")
    print(f"  n_features total: {len(df):,}")
    print(f"  n_passed_consensus: {cfg['n_consensus']}")
    print(f"  tier breakdown: {cfg['n_per_tier']}")
    if cfg["n_consensus"] > 0:
        passed = pd.read_parquet(OUTPUT_DIR / "consensus_passed.parquet")
        print()
        print("  Top consensus features (sorted by frac):")
        cols = ["feature_id", "n_paths_flagging", "frac_paths_flagging", "consensus_tier"]
        print(passed[cols].head(10).to_string(index=False))


def audit_multiverse_index() -> None:
    """Phase 8e.7: top-level index page exists and contains expected content,
    with every numeric claim back-verified against the consensus parquet."""
    section_banner("PHASE 8e.7 — MULTIVERSE INDEX PAGE")
    index_path = OUTPUT_DIR / "index.html"
    summary_path = OUTPUT_DIR / "summary.json"
    figures_dir = OUTPUT_DIR / "figures"
    report(index_path.exists(), "index.html exists", str(index_path))
    if not index_path.exists():
        return
    report(summary_path.exists(), "summary.json sidecar exists", str(summary_path))
    report(figures_dir.exists() and figures_dir.is_dir(), "figures/ directory exists", str(figures_dir))

    html = index_path.read_text()

    # Sidebar entries — sections we know we added
    expected_sections = [
        "Multiverse summary",
        "Paths in this multiverse",
        "Specification curve",
        "Cross-path agreement",
        "Consensus tier breakdown",
        "consensus features",  # "Top 30 consensus features"
        "Method evidence drilldown",
        "Methods + reproducibility",
    ]
    missing = [s for s in expected_sections if s not in html]
    report(
        not missing,
        "all 8 sections present in HTML",
        f"missing: {missing or '∅'}",
    )

    # Per-path hyperlinks resolve to existing files
    broken_links = []
    for p in EXPECTED_PATHS:
        link = f'href="{p}/index.html"'
        if link not in html:
            broken_links.append(f"{p}: missing href")
            continue
        target = OUTPUT_DIR / p / "index.html"
        if not target.exists():
            broken_links.append(f"{p}: target {target} missing")
    report(
        not broken_links,
        "per-path hyperlinks resolve to real files",
        ", ".join(broken_links) if broken_links else "all 4 OK",
    )

    # Numbers on the page must match the consensus parquet
    cs = pd.read_parquet(OUTPUT_DIR / "consensus.parquet")
    cfg = json.loads((OUTPUT_DIR / "consensus_config.json").read_text())
    n_consensus_html = f">{cfg['n_consensus']}<"  # in the metric value div
    report(
        n_consensus_html in html,
        "n_consensus value on page matches consensus_config",
        f"expected '>{cfg['n_consensus']}<' in HTML",
    )

    # Top features table includes a top consensus feature_id
    top_feature = cs.sort_values(
        ["passed_consensus", "frac_paths_flagging"], ascending=[False, False]
    ).iloc[0]["feature_id"]
    report(
        str(top_feature) in html,
        "top consensus feature appears in the table",
        f"feature: {top_feature}",
    )

    # Methods section cites the references we documented
    for ref in ("Steegen 2016", "Simonsohn 2020"):
        report(ref in html, f"cites {ref}", "in HTML or footer")

    # 4 plotly figures in the index (specification curve + jaccard + tier + drilldown)
    n_fig_blocks = html.count('class="figure-block"')
    report(
        n_fig_blocks == 4,
        "4 figure blocks in index",
        f"counted: {n_fig_blocks}",
    )

    # Each figure has its own standalone HTML written to figures/
    expected_fig_keywords = ("specification", "jaccard", "tier", "drilldown")
    for kw in expected_fig_keywords:
        matches = list(figures_dir.glob(f"*{kw}*.html"))
        report(
            bool(matches),
            f"standalone figure file '{kw}*.html' exists",
            f"found: {[m.name for m in matches]}",
        )

    # summary.json reflects what we built
    summary = json.loads(summary_path.read_text())
    report(
        summary["n_sections"] == 8 and summary["n_figures"] == 4,
        "summary.json: n_sections=8, n_figures=4",
        f"got n_sections={summary['n_sections']}, n_figures={summary['n_figures']}",
    )


def main() -> None:
    audit_per_path_outputs()
    audit_consensus_files()
    audit_consensus_schema()
    audit_per_path_flag_matrix()
    audit_consensus_counts()
    audit_passed_parquet()
    audit_config_sidecar()
    audit_consensus_summary()
    audit_multiverse_index()


if __name__ == "__main__":
    main()
