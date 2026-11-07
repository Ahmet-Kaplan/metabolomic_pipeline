"""Preview the impact of each ``OUTCOME_FILTER`` setting before launching
a (~10–14 hr) multiverse re-run.

Reads ``data/interim/10_origin/metrics.parquet`` and prints:
  1. The kept-only outcome breakdown for the current Stage-10 run.
  2. How many features each canonical filter setting would select.
  3. MSI-tier × outcome cross-tabulation so the user sees what's leaving
     when they exclude a bucket.

Usage
-----
``.venv/bin/python scripts/diagnose_outcome_filter.py``

Optional ``--stage-10-dir`` to point at a non-default location.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from biodisc.multiverse.io import VALID_OUTCOMES, load_active_feature_ids

CANONICAL_SETTINGS: list[tuple[str, tuple[str, ...] | None]] = [
    ("None (current default — all kept features)", None),
    ("(endogenous_evidence, exo_or_unknown, passthrough_l4)",
     ("endogenous_evidence", "exo_or_unknown", "passthrough_l4")),
    ("(endogenous_evidence, exo_or_unknown)  — drops L4 unknowns",
     ("endogenous_evidence", "exo_or_unknown")),
    ("(endogenous_evidence, passthrough_l4)  — drops 'mixed exo/unknown'",
     ("endogenous_evidence", "passthrough_l4")),
    ("(endogenous_evidence,)                 — annotated endogenous only",
     ("endogenous_evidence",)),
    ("(passthrough_l4,)                      — L4 unidentified only",
     ("passthrough_l4",)),
]


def _print_outcome_breakdown(metrics: pd.DataFrame) -> None:
    print("=" * 72)
    print("Stage-10 outcome breakdown (kept=True only)")
    print("=" * 72)
    kept = metrics[metrics["kept"]]
    counts = kept["outcome"].value_counts()
    total = len(kept)
    for outcome in sorted(VALID_OUTCOMES):
        n = int(counts.get(outcome, 0))
        pct = 100.0 * n / total if total else 0.0
        print(f"  {outcome:<22} {n:>6,d}  ({pct:5.1f} %)")
    print(f"  {'TOTAL':<22} {total:>6,d}")
    print()


def _print_msi_crosstab(metrics: pd.DataFrame) -> None:
    print("=" * 72)
    print("MSI-tier × outcome (kept only)")
    print("=" * 72)
    kept = metrics[metrics["kept"]]
    ct = pd.crosstab(kept["msi_tier"], kept["outcome"], margins=True)
    # Sort columns to a stable order for readability
    cols = [c for c in ["endogenous_evidence", "exo_or_unknown", "passthrough_l4", "All"]
            if c in ct.columns]
    print(ct[cols].to_string())
    print()


def _print_canonical_settings(stage_10_dir: Path, metrics: pd.DataFrame) -> None:
    print("=" * 72)
    print("Canonical OUTCOME_FILTER settings — feature count preview")
    print("=" * 72)
    baseline = int(metrics["kept"].sum())
    header = f"  {'Setting':<58} {'n':>6}  {'Δ vs all':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, setting in CANONICAL_SETTINGS:
        try:
            ids = load_active_feature_ids(stage_10_dir, outcome_filter=setting)
            n = len(ids)
            delta = n - baseline
            delta_str = f"{delta:+,d}" if delta else "—"
            print(f"  {label:<58} {n:>6,d}  {delta_str:>9}")
        except ValueError as e:
            print(f"  {label:<58} {'ERR':>6}  ({e})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-10-dir",
        type=Path,
        default=Path("data/interim/10_origin"),
        help="Stage-10 outputs directory (default: data/interim/10_origin).",
    )
    args = parser.parse_args()

    if not (args.stage_10_dir / "metrics.parquet").exists():
        raise SystemExit(
            f"metrics.parquet not found in {args.stage_10_dir}. "
            "Run Stage 10 first (``biodisc filter origin``)."
        )

    metrics = pd.read_parquet(args.stage_10_dir / "metrics.parquet")

    _print_outcome_breakdown(metrics)
    _print_msi_crosstab(metrics)
    _print_canonical_settings(args.stage_10_dir, metrics)

    print(
        "Edit ``OUTCOME_FILTER`` in ``scripts/run_multiverse_production.py`` "
        "to apply one of these settings. ``None`` (current) keeps every "
        "kept Stage-10 feature."
    )


if __name__ == "__main__":
    main()
