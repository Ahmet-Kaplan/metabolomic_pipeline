"""Replication-structure & run-order diagnostic — documents the experimental
design and tests for the two design-level threats that NO amount of correct
analysis code can fix:

  1. **Run-order confounding** — is acquisition order aliased with cell line?
     (If a group is injected in one contiguous block, analytical drift is
     indistinguishable from biology. Dunn 2011 / Broadhurst 2018 require
     randomised or interleaved run order.)

  2. **Pseudo-replication** — are the per-group replicates independent
     biological cultures, or repeated injections of one prepared sample?
     If the latter, group p-values measure *analytical reproducibility*,
     not *biological difference* (Lazic 2010; Blainey 2014, Nat Methods).
     This is THE silent killer of cell-line metabolomics inference.

The data alone cannot *prove* biological vs technical replication — only the
wet-lab protocol can. But this script provides quantitative *evidence*: if the
within-group variance is no larger than the QC (injection) variance, the
replicates behave like technical replicates. If it is meaningfully larger,
there is biological/prep variance beyond the instrument.

Usage
-----
``.venv/bin/python scripts/diagnose_replication_structure.py``

Writes a markdown report to ``docs/evaluation/replication_structure_check.md``
and prints the same to stdout. Read-only — touches no pipeline code or outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

INTERIM = Path("data/interim")
EXTERNAL = Path("data/external")
DEFAULT_REPORT = Path("docs/evaluation/replication_structure_check.md")

# Heuristic interpretation thresholds for the within-group / QC variance ratio.
# < LEAN_TECHNICAL → behaves like technical replicates (high pseudo-rep risk).
# > LEAN_BIOLOGICAL → clear biological/prep variance beyond the instrument.
LEAN_TECHNICAL = 1.2
LEAN_BIOLOGICAL = 2.0


def _rsd(x: pd.Series) -> float:
    """Relative standard deviation (%) of detected values; NaN if <3 detected."""
    x = x.dropna()
    if len(x) < 3 or x.mean() == 0:
        return float("nan")
    return float(100.0 * x.std(ddof=1) / x.mean())


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = pd.read_parquet(INTERIM / "01_parsed/samples.parquet")
    run_order = pd.read_parquet(EXTERNAL / "run_order.parquet")
    abund = pd.read_parquet(INTERIM / "01_parsed/abundances_raw_long.parquet")
    return samples, run_order, abund


def section_acquisition_design(samples: pd.DataFrame, run_order: pd.DataFrame) -> list[str]:
    """Reconstruct the TRUE acquisition sequence (Waters timestamps) and QC spacing."""
    seq = (
        run_order.merge(
            samples[["raw_sample_id", "injection_type", "cell_line"]],
            on="raw_sample_id",
            how="left",
        )
        .sort_values("position")
        .reset_index(drop=True)
    )
    lines = ["## 1. Acquisition design (true Waters run order)", ""]

    n_by_type = seq["injection_type"].value_counts().to_dict()
    lines.append(f"- Total injections: **{len(seq)}** — {n_by_type}")

    # QC spacing among the analytical (non-conditioning) portion
    qc_positions = seq.loc[seq["injection_type"] == "qc", "position"].to_numpy()
    bio_positions = seq.loc[seq["injection_type"] == "biological", "position"].to_numpy()
    if len(qc_positions) >= 2:
        gaps = np.diff(np.sort(qc_positions))
        lines.append(
            f"- QC injections: **{len(qc_positions)}**, "
            f"position range {int(qc_positions.min())}–{int(qc_positions.max())}, "
            f"max gap between consecutive QCs: **{int(gaps.max())}** injections"
        )
    if len(bio_positions):
        lines.append(
            f"- Biological injections span positions "
            f"**{int(bio_positions.min())}–{int(bio_positions.max())}** "
            f"of {len(seq)}"
        )

    # Compact picture of the biological block pattern
    bio_seq = seq[seq["injection_type"] == "biological"]
    order_str = " → ".join(bio_seq["cell_line"].head(10).tolist())
    lines.append(f"- First 10 biological injections (order): {order_str} …")
    lines.append("")
    return lines, seq


def section_run_order_confounding(seq: pd.DataFrame) -> list[str]:
    """Is acquisition position aliased with cell line?"""
    lines = ["## 2. Run-order confounding check", ""]
    bio = seq[seq["injection_type"] == "biological"].copy()
    groups = sorted(bio["cell_line"].unique())

    total_lo, total_hi = bio["position"].min(), bio["position"].max()
    total_span = max(1, total_hi - total_lo)

    lines.append("| Cell line | n | mean position | position span | % of run covered |")
    lines.append("|-----------|--:|--------------:|--------------:|-----------------:|")
    for g in groups:
        pos = bio.loc[bio["cell_line"] == g, "position"]
        span = pos.max() - pos.min()
        lines.append(
            f"| {g} | {len(pos)} | {pos.mean():.1f} | "
            f"{int(pos.min())}–{int(pos.max())} | {100*span/total_span:.0f}% |"
        )

    # Kruskal-Wallis: do groups differ systematically in run position?
    pos_by_group = [bio.loc[bio["cell_line"] == g, "position"].to_numpy() for g in groups]
    try:
        h, p = stats.kruskal(*pos_by_group)
        lines.append("")
        lines.append(
            f"- Kruskal-Wallis (position ~ cell line): H={h:.2f}, **p={p:.3f}**"
        )
        if p < 0.05:
            lines.append(
                "  - ⚠️ Groups occupy systematically different run positions — "
                "**run-order is confounded with cell line.**"
            )
        else:
            lines.append(
                "  - ✅ No systematic position difference between groups — "
                "**run order is NOT confounded with cell line** (interleaved design)."
            )
    except ValueError:
        lines.append("- Kruskal-Wallis could not be computed.")
    lines.append("")
    return lines


def section_within_round_order(seq: pd.DataFrame) -> list[str]:
    """Residual confound: does each cell line always sit at the same within-round slot?"""
    lines = ["## 3. Within-round ordering (residual confound)", ""]
    seq = seq.sort_values("position").reset_index(drop=True)
    # "Injections since last QC" = within-round slot index
    since_qc = []
    counter = 0
    for _, row in seq.iterrows():
        if row["injection_type"] == "qc":
            counter = 0
        else:
            counter += 1
        since_qc.append(counter)
    seq = seq.assign(slot_since_qc=since_qc)

    bio = seq[seq["injection_type"] == "biological"]
    groups = sorted(bio["cell_line"].unique())
    lines.append("| Cell line | mean slot-since-QC | std | fixed slot? |")
    lines.append("|-----------|-------------------:|----:|:-----------:|")
    fixed_any = False
    for g in groups:
        slots = bio.loc[bio["cell_line"] == g, "slot_since_qc"]
        is_fixed = slots.std() < 0.5
        fixed_any = fixed_any or is_fixed
        lines.append(
            f"| {g} | {slots.mean():.1f} | {slots.std():.2f} | "
            f"{'⚠️ yes' if is_fixed else 'no'} |"
        )
    lines.append("")
    if fixed_any:
        lines.append(
            "- ⚠️ At least one cell line always occupies the same slot after each QC. "
            "This is a **minor residual confound** (fixed within-round position). "
            "Note it in Methods; magnitude is usually small vs between-group effects."
        )
    else:
        lines.append("- ✅ Within-round positions vary — no fixed-slot residual confound.")
    lines.append("")
    return lines


def section_pseudoreplication(samples: pd.DataFrame, abund: pd.DataFrame) -> list[str]:
    """The key quantitative test: within-group variance vs QC (analytical) variance."""
    lines = ["## 4. Pseudo-replication evidence (within-group vs analytical variance)", ""]

    # Restrict to Stage-10 survivors if available (denoised, fair comparison)
    kept_path = INTERIM / "10_origin/kept_features.parquet"
    if kept_path.exists():
        kept = set(pd.read_parquet(kept_path)["feature_id"])
        abund = abund[abund["feature_id"].isin(kept)]
        scope = f"{len(kept)} Stage-10 surviving features"
    else:
        scope = "all parsed features"

    df = abund.merge(
        samples[["sample_id", "injection_type", "cell_line"]], on="sample_id"
    )
    df.loc[df["intensity"] == 0, "intensity"] = np.nan  # Progenesis 0 = below LOD

    bio = df[df["injection_type"] == "biological"]
    qc = df[df["injection_type"] == "qc"]

    qc_rsd = qc.groupby("feature_id")["intensity"].apply(_rsd).rename("qc_rsd")
    wg = (
        bio.groupby(["feature_id", "cell_line"])["intensity"]
        .apply(_rsd)
        .rename("wg_rsd")
        .reset_index()
    )
    wg_med = wg.groupby("feature_id")["wg_rsd"].median().rename("within_group_rsd")

    comp = pd.concat([wg_med, qc_rsd], axis=1).dropna()
    comp["ratio"] = comp["within_group_rsd"] / comp["qc_rsd"]
    ratio_med = float(comp["ratio"].median())

    lines.append(f"Scope: {scope} ({len(comp)} features with ≥3 detections in both).")
    lines.append("")
    lines.append("| Quantity | Value |")
    lines.append("|----------|------:|")
    lines.append(f"| Median within-group RSD | {comp['within_group_rsd'].median():.1f}% |")
    lines.append(f"| Median QC (analytical) RSD | {comp['qc_rsd'].median():.1f}% |")
    lines.append(f"| **Median ratio (within-group / QC)** | **{ratio_med:.2f}** |")
    lines.append(f"| % features ratio < {LEAN_TECHNICAL} | {100*(comp['ratio']<LEAN_TECHNICAL).mean():.0f}% |")
    lines.append(f"| % features ratio > {LEAN_BIOLOGICAL} | {100*(comp['ratio']>LEAN_BIOLOGICAL).mean():.0f}% |")
    lines.append("")

    if ratio_med < LEAN_TECHNICAL:
        verdict = (
            "⚠️ **HIGH pseudo-replication risk.** Within-group variance is "
            "barely above analytical variance — the replicates behave like "
            "TECHNICAL (injection) replicates. Group p-values may reflect "
            "instrument reproducibility, not biology. **Confirm the wet-lab "
            "protocol before any biomarker claim.**"
        )
    elif ratio_med < LEAN_BIOLOGICAL:
        verdict = (
            "⚖️ **AMBIGUOUS.** Within-group variance exceeds analytical "
            f"variance (ratio {ratio_med:.2f}), so there is real variance beyond "
            "the instrument — but it is modest. This is consistent with EITHER "
            "biological replicates with low biological variance OR technical "
            "replicates plus sample-prep variance. **The data cannot resolve "
            "this; the wet-lab protocol must confirm biological independence.**"
        )
    else:
        verdict = (
            "✅ **Consistent with biological replication.** Within-group variance "
            f"clearly exceeds analytical variance (ratio {ratio_med:.2f}). The "
            "replicates carry substantial variance beyond the instrument, as "
            "expected for independent biological cultures. (Still confirm the "
            "protocol, but this is reassuring.)"
        )
    lines.append(f"**Interpretation:** {verdict}")
    lines.append("")
    lines.append(
        "> Caveat: the QC is a *pooled* sample, so QC RSD estimates injection "
        "reproducibility of the pooled matrix, which may slightly differ from "
        "per-cell-line injection RSD. Treat this as evidence, not proof."
    )
    lines.append("")
    return lines


def section_summary(ratio_known: bool) -> list[str]:
    return [
        "## 5. What this script can and cannot tell you",
        "",
        "- **Can** confirm run-order is/ isn't confounded with cell line (Sec. 2–3).",
        "- **Can** give quantitative evidence on technical-vs-biological replication "
        "via within-group/analytical variance (Sec. 4).",
        "- **Cannot** prove biological independence — only the wet-lab protocol can.",
        "",
        "### The one question to resolve with the lab",
        "",
        "> Were the 6–7 replicates per cell line **independent biological cultures** "
        "(separate flasks / passages / harvests), or **repeated injections / aliquots "
        "of a single culture** per cell line?",
        "",
        "- Independent cultures → statistics are valid for these 5 lines; next step is "
        "an external validation cohort.",
        "- Injection/aliquot replicates → reframe as analytical/exploratory; biomarker "
        "claims require a biologically-replicated follow-up.",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT,
        help=f"Markdown report output path (default: {DEFAULT_REPORT}).",
    )
    args = parser.parse_args()

    for p in [
        INTERIM / "01_parsed/samples.parquet",
        EXTERNAL / "run_order.parquet",
        INTERIM / "01_parsed/abundances_raw_long.parquet",
    ]:
        if not p.exists():
            raise SystemExit(f"Required input not found: {p}")

    samples, run_order, abund = load_inputs()

    out: list[str] = [
        "# Replication-structure & run-order diagnostic",
        "",
        "_Auto-generated by `scripts/diagnose_replication_structure.py` — read-only._",
        "",
        "Tests the two design-level threats correct analysis code cannot fix: "
        "run-order confounding and pseudo-replication.",
        "",
    ]
    design_lines, seq = section_acquisition_design(samples, run_order)
    out += design_lines
    out += section_run_order_confounding(seq)
    out += section_within_round_order(seq)
    out += section_pseudoreplication(samples, abund)
    out += section_summary(ratio_known=True)

    report = "\n".join(out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(report)
    print(f"\n[written] {args.report}")


if __name__ == "__main__":
    main()
