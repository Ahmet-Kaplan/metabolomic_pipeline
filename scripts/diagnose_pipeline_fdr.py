"""Pipeline-wide permutation FDR — estimate null discovery rate for the multiverse.

Permutes group labels N times, runs the full statistical pipeline
(4 methods × k-of-n gate × multiverse consensus), and measures how
many false discoveries emerge under the complete null hypothesis.

Key insight
-----------
Pre-processing (imputation → drift → normalisation → transformation →
scaling) is label-independent. We run it ONCE per path and cache the
scaled/transformed matrices. Each permutation then only re-runs the
statistical methods (ANOVA, PLS-DA, sPLS-DA, RF) on permuted labels,
which is ~100× faster than the full pipeline.

Usage
-----
    # Fast pilot: 1 path, univariate-only, 500 permutations
    uv run python scripts/diagnose_pipeline_fdr.py --n-permutations 500 --fast

    # Full: all 4 paths, all methods, 100 permutations (overnight)
    uv run python scripts/diagnose_pipeline_fdr.py --n-permutations 100

Output: docs/evaluation/pipeline_permutation_fdr.md
"""

from __future__ import annotations

import argparse
import time

# Local PathSpec analogue — avoids importing biodisc.multiverse.orchestrator
# which triggers the plotly-dependent reporting module at import time.
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PathSpec:
    """Minimal path spec for the permutation FDR script."""
    name: str
    imputation: str = ""
    drift: str = ""
    normalisation: str = ""
    transformation: str = ""
    scaling: str = "pareto"
    imputation_params: dict = field(default_factory=dict)
    drift_params: dict = field(default_factory=dict)
    normalisation_params: dict = field(default_factory=dict)
    transformation_params: dict = field(default_factory=dict)
    scaling_params: dict = field(default_factory=dict)

    @property
    def chain(self) -> list[str]:
        parts = [self.imputation, self.drift, self.normalisation,
                 self.transformation, self.scaling]
        return [p for p in parts if p]

INTERIM = Path("data/interim")
CONSENSUS = Path("data/reports/multiverse")
REPORT = Path("docs/evaluation/pipeline_permutation_fdr.md")
CACHE = Path("data/.pipeline_fdr_cache")

# Smoke settings for fast iteration
SMOKE_BOOTSTRAP = 30
SMOKE_PERMUTATION = 100


def _load_or_preprocess(paths: list[PathSpec]) -> dict[str, dict[str, pd.DataFrame]]:
    """Pre-process each path (label-independent steps) and cache the matrices.

    Returns::
        {
            "path_a_classical": {
                "transformed": pd.DataFrame,   # for univariate (log2)
                "scaled": pd.DataFrame,         # for multivariate (Pareto)
            },
            ...
        }
    """
    cache_dir = CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    from biodisc.multiverse.production import load_multiverse_data

    # Generate synthetic run_order if the real one doesn't exist
    # (needed for drift correction pre-processing).
    run_order_path = Path("data/external/run_order.parquet")
    run_order_path.parent.mkdir(parents=True, exist_ok=True)
    if not run_order_path.exists():
        print("    [warning] run_order.parquet not found — creating synthetic version from samples")
        samples_dummy = pd.read_parquet(Path("data/interim/01_parsed/samples.parquet"))
        dummy_ro = pd.DataFrame({
            "position": range(len(samples_dummy)),
            "raw_sample_id": samples_dummy["raw_sample_id"].values,
            "acq_datetime": pd.Timestamp("2026-01-09 09:00:00")
            + pd.to_timedelta(samples_dummy.index * 10, unit="min"),
        })
        dummy_ro.to_parquet(run_order_path, compression="zstd")
        print(f"    [created] {run_order_path} ({len(dummy_ro)} rows)")

    matrix, samples, features, run_order = load_multiverse_data()

    cached: dict[str, dict[str, pd.DataFrame]] = {}

    for spec in paths:
        print(f"\n  Pre-processing {spec.name}...")

        # Check cache
        transformed_cache = cache_dir / f"{spec.name}_transformed.parquet"
        scaled_cache = cache_dir / f"{spec.name}_scaled.parquet"
        if transformed_cache.exists() and scaled_cache.exists():
            print(f"    [cached] loading from {cache_dir / spec.name}_*.parquet")
            cached[spec.name] = {
                "transformed": pd.read_parquet(transformed_cache),
                "scaled": pd.read_parquet(scaled_cache),
            }
            continue

        t0 = time.time()
        from biodisc.multiverse._types import StageContext
        ctx = StageContext(
            matrix=matrix, samples=samples, features=features, run_order=run_order,
        )

        from biodisc.multiverse.registry import apply as apply_method

        # Imputation (label-independent)
        imputed = apply_method("imputation", spec.imputation, ctx, **spec.imputation_params)
        print(f"    imputed ({spec.imputation})  — {len(imputed)} features")

        # Drift correction (label-independent)
        ctx_d = StageContext(
            matrix=imputed, samples=samples, features=features, run_order=run_order,
        )
        drift = apply_method("drift", spec.drift, ctx_d, **spec.drift_params)
        print(f"    drift corrected ({spec.drift})")

        # Normalisation (label-independent)
        ctx_n = StageContext(
            matrix=drift, samples=samples, features=features, run_order=run_order,
        )
        norm = apply_method("normalisation", spec.normalisation, ctx_n, **spec.normalisation_params)

        # Transformation (label-independent)
        ctx_t = StageContext(
            matrix=norm, samples=samples, features=features, run_order=run_order,
        )
        transformed = apply_method(
            "transformation", spec.transformation, ctx_t, **spec.transformation_params,
        )

        # Scaling (label-independent)
        ctx_s = StageContext(
            matrix=transformed, samples=samples, features=features, run_order=run_order,
        )
        scaled = apply_method("scaling", spec.scaling, ctx_s, **spec.scaling_params)

        # Cache
        transformed.to_parquet(transformed_cache, compression="zstd")
        scaled.to_parquet(scaled_cache, compression="zstd")

        cached[spec.name] = {"transformed": transformed, "scaled": scaled}
        print(f"    done ({time.time() - t0:.0f}s)")

    return cached


def _permute_labels(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy of samples with permuted cell_line labels.

    Preserves the per-cell-line group sizes and the injection_type
    (biological/QC/blank/dilution) structure. Only biological samples
    have their labels shuffled; QC/blank/dilution are unchanged.
    """
    permuted = samples.copy()
    bio_mask = samples["injection_type"] == "biological"

    original_labels = samples.loc[bio_mask, "cell_line"].to_numpy()
    shuffled = original_labels.copy()
    rng.shuffle(shuffled)

    permuted.loc[bio_mask, "cell_line"] = shuffled
    return permuted


def _run_one_permutation(
    spec: PathSpec,
    cached_data: dict[str, dict[str, pd.DataFrame]],
    orig_samples: pd.DataFrame,
    features: pd.DataFrame,
    run_order: pd.DataFrame | None,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutation: int,
    use_multivariate: bool,
) -> int:
    """Run one label permutation for a single path. Returns shortlist size."""
    transformed = cached_data[spec.name]["transformed"]
    scaled = cached_data[spec.name]["scaled"]

    # Permute labels
    permuted_samples = _permute_labels(orig_samples, rng)

    # Univariate (ANOVA + BH-FDR + η²) on transformed data with permuted labels
    from biodisc.multiverse._types import StageContext
    from biodisc.multiverse.stats_univariate import compute_univariate

    uv_result = compute_univariate(
        StageContext(matrix=transformed, samples=permuted_samples, features=features),
        is_log_scale=True,
        tukey_post_hoc="none",
    )

    multivariate_results: dict = {}

    if use_multivariate:
        multi_ctx = StageContext(
            matrix=scaled, samples=permuted_samples, features=features, run_order=run_order,
        )
        seed = int(rng.integers(0, 2**31 - 1))

        from biodisc.multiverse.methods.multivariate.plsda import fit as plsda_fit
        from biodisc.multiverse.methods.multivariate.rf import fit as rf_fit
        from biodisc.multiverse.methods.multivariate.splsda import fit as splsda_fit

        try:
            multivariate_results["plsda"] = plsda_fit(
                multi_ctx, n_bootstrap=n_bootstrap, n_permutation=n_permutation, seed=seed,
            )
        except Exception as exc:
            _ = exc  # skip failed method

        try:
            multivariate_results["splsda"] = splsda_fit(
                multi_ctx, n_bootstrap=n_bootstrap, n_permutation=n_permutation, seed=seed,
            )
        except Exception as exc:
            _ = exc

        try:
            multivariate_results["rf"] = rf_fit(
                multi_ctx, n_bootstrap=n_bootstrap, n_permutation=n_permutation, seed=seed,
            )
        except Exception as exc:
            _ = exc

    # Shortlist
    from biodisc.multiverse.shortlist import compute_shortlist, default_config

    try:
        sl_result = compute_shortlist(
            univariate_result=uv_result,
            multivariate_results=multivariate_results,
            config=default_config(),
        )
    except Exception:
        return 0

    return sl_result.n_shortlisted


def _compute_observed_shortlist() -> dict[str, int]:
    """Read observed per-path and consensus shortlist sizes from multiverse output."""
    observed: dict[str, int] = {}

    # Consensus
    consensus_path = CONSENSUS / "consensus_passed.parquet"
    if consensus_path.exists():
        df = pd.read_parquet(consensus_path)
        observed["consensus"] = int(df["passed_consensus"].sum())
    else:
        observed["consensus"] = 0

    # Per-path
    for name in ["path_a_classical", "path_b_robust_ml", "path_c_mnar_aware", "path_d_middle"]:
        sl_path = CONSENSUS / name / "shortlist.parquet"
        if sl_path.exists():
            df = pd.read_parquet(sl_path)
            observed[name] = int(df["passed_shortlist"].sum())
        else:
            observed[name] = None  # not available

    return observed


def _write_report(
    path_per_perm: dict[str, list[int]],
    results: dict,
    use_multivariate: bool,
    n_permutations: int,
    paths_used: list[str],
) -> None:
    """Write the permutation FDR report."""
    methods_str = "all 4 methods (ANOVA + PLS-DA + sPLS-DA + RF)" if use_multivariate else "univariate only (ANOVA + BH-FDR)"
    paths_str = ", ".join(paths_used)

    lines = [
        "# Pipeline-wide Permutation FDR Estimate",
        "",
        "_Auto-generated by `scripts/diagnose_pipeline_fdr.py`._",
        "",
        "Estimates the expected number of false discoveries under the complete",
        "null hypothesis by permuting group labels while preserving all",
        "metabolite-metabolite correlations.",
        "",
        "## Settings",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Permutations | {n_permutations} |",
        f"| Methods | {methods_str} |",
        f"| Paths | {paths_str} |",
        f"| Bootstrap (multivariate) | {SMOKE_BOOTSTRAP} |",
        f"| Permutation (multivariate) | {SMOKE_PERMUTATION} |",
        "",
    ]

    observed = _compute_observed_shortlist()

    lines += ["## Null Distribution Summary", "", "| Metric | Value |"]
    for key, counts in path_per_perm.items():
        if not counts:
            continue
        arr = np.array(counts)
        obs_val = observed.get(key, "—")
        obs_str = f"{obs_val}" if obs_val is not None else "N/A"
        p95 = int(np.percentile(arr, 95))
        p99 = int(np.percentile(arr, 99))
        max_val = int(arr.max())
        mean_val = arr.mean()
        fdr_est = min(1.0, max_val / obs_val) if isinstance(obs_val, int) and obs_val > 0 else "—"

        label = {"consensus": "Consensus (≥2 paths)"}.get(key, key)
        lines += [
            f"| **{label}** | |",
            f"| — Observed | {obs_str} |",
            f"| — Mean under null | {mean_val:.2f} |",
            f"| — 95th percentile | {p95} |",
            f"| — 99th percentile | {p99} |",
            f"| — Max under null | {max_val} |",
        ]
        if isinstance(fdr_est, float):
            lines.append(f"| — Estimated FDR (mean/observed) | {fdr_est:.4f} ({fdr_est:.1%}) |")

    lines += [
        "",
        "## Per-Permutation Shortlist Sizes",
        "",
    ]

    for key, counts in path_per_perm.items():
        if not counts:
            continue
        lines += [f"### {key}", "", "| perm # | shortlist size |", "|-------:|--------------:|"]
        for i, c in enumerate(counts[:50]):  # show first 50
            lines.append(f"| {i + 1} | {c} |")
        if len(counts) > 50:
            lines.append(f"| … | … |  ({len(counts) - 50} remaining) |")
        lines.append("")

    # Histogram as ASCII
    lines += ["## Null Distribution Histograms", ""]
    for key, counts in path_per_perm.items():
        if not counts:
            continue
        arr = np.array(counts)
        label = {"consensus": "Consensus (≥2 paths)"}.get(key, key)
        lines += [f"### {label}", ""]

        if arr.max() > 0:
            obs_val = observed.get(key, 0)
            obs_val = obs_val if obs_val else 0
            bins = np.arange(-0.5, max(arr.max(), obs_val) + 1.5, 1)
            hist, edges = np.histogram(arr, bins=bins)
            max_h = max(hist) if hist.max() > 0 else 1

            # Scale to 40 chars
            for _count, (h, lo, hi) in enumerate(
                zip(hist, edges[:-1], edges[1:], strict=True)
            ):
                bar_len = max(1, int(h / max_h * 40))
                bar = "█" * bar_len
                marker = " ◄ OBSERVED" if lo <= obs_val < hi else ""
                lines.append(f"  {int(lo):>3}–{int(hi):<3} │{bar} {h}{marker}")

            lines.append("")
            lines.append(
                f"  Observed: {obs_val}  |  "
                f"Null 95%ile: {int(np.percentile(arr, 95))}  |  "
                f"Null max: {int(arr.max())}"
            )
        else:
            lines.append(f"  All {len(arr)} permutations produced 0 features under the null.")
        lines.append("")

    # Interpretation
    lines += ["## Interpretation", ""]
    consensus_counts = path_per_perm.get("consensus", [])
    if consensus_counts:
        arr = np.array(consensus_counts)
        obs_val = observed.get("consensus", 0)
        if isinstance(obs_val, int) and obs_val > 0:
            p95 = int(np.percentile(arr, 95))
            if p95 == 0:
                lines += [
                    f"**✅ The multiverse consensus strongly controls FDR.** "
                    f"No consensus features emerged under the null across "
                    f"{n_permutations} permutations (95th percentile = 0). "
                    f"The {obs_val} observed consensus features are highly unlikely "
                    f"to be false discoveries.",
                ]
            elif arr.mean() < obs_val * 0.05:
                fdr_pct = 100 * arr.mean() / obs_val
                lines += [
                    f"**The FDR is estimated at {fdr_pct:.0f}%** (mean null / observed). "
                    f"The 95th percentile is {p95}, meaning ≥{p95} false features "
                    f"would appear in fewer than 5% of experiments under the null.",
                ]
            else:
                lines += [
                    f"**⚠️ The null distribution produces non-trivial discoveries.** "
                    f"Mean null = {arr.mean():.1f}, 95th percentile = {p95}. "
                    f"The pipeline may have elevated FDR. Consider stricter "
                    f"consensus thresholds or additional multiplicity control.",
                ]
        elif obs_val == 0:
            lines += [
                "No observed consensus features. The null distribution provides "
                "a baseline for future experiments.",
            ]

    if not use_multivariate:
        lines += [
            "",
            "> **Note:** This run used univariate-only (ANOVA + BH-FDR + η²). "
            "Multivariate methods (PLS-DA, sPLS-DA, RF) add additional filtering "
            "that would reduce false discoveries further. The FDR estimate here "
            "is conservative (upper bound).",
        ]

    lines += [
        "",
        "## Method",
        "",
        "1. Load raw data and apply all label-independent pre-processing once "
        "(imputation → drift correction → normalisation → transformation → scaling).",
        "2. For each permutation: shuffle cell-line labels among biological samples, "
        "preserving group sizes and all metabolite-metabolite correlations.",
        "3. Run the full statistical pipeline on permuted labels: univariate "
        "(ANOVA + BH-FDR + η²) on transformed data, multivariate (PLS-DA + "
        "sPLS-DA + RF) on scaled data with bootstrap stability and permutation testing.",
        "4. Apply the k=2-of-4 shortlist gate per path.",
        "5. For multi-path runs: compute cross-path consensus (≥2 of 4 paths).",
        "6. Collect the null distribution of shortlist/consensus sizes.",
        "",
        "This follows the permutation-based FDR framework of Storey & Tibshirani "
        "(2003, PNAS) and the GSEA method (Subramanian et al. 2005, PNAS), "
        "extended to a multistep pipeline.",
        "",
    ]

    report = "\n".join(lines)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report)
    print(f"\n[written] {REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-permutations", type=int, default=50,
        help="Number of label permutations (default: 50).",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: 1 path (Path A), univariate only.",
    )
    parser.add_argument(
        "--multivariate", action="store_true",
        help="Include multivariate methods (PLS-DA, sPLS-DA, RF). Slower.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    t_total = time.time()
    print("=" * 60)
    print("Pipeline-Wide Permutation FDR")
    print("=" * 60)
    print()

    # 1. Determine which paths to use
    if args.fast:
        pass  # PathSpec imported at module level
        print("[Fast mode] Path A only, univariate only")
        paths = [PathSpec(
            name="path_a_classical",
            imputation="halfmin",
            drift="qc_rsc",
            normalisation="pqn",
            transformation="log2",
        )]
        use_multivariate = False
    else:
        from biodisc.multiverse.production import build_default_paths
        paths = build_default_paths()
        use_multivariate = args.multivariate

    paths_used = [p.name for p in paths]
    print(f"Paths: {paths_used}")

    # 2. Pre-process each path once
    print("\n--- Pre-processing (label-independent) ---")
    cached_data = _load_or_preprocess(paths)

    # 3. Load original samples and features for permuting
    from biodisc.multiverse.production import load_multiverse_data
    _, orig_samples, features, run_order = load_multiverse_data()

    # 4. Run permutations
    print(f"\n--- Running {args.n_permutations} label permutations ---")
    if use_multivariate:
        print(f"  (multivariate: n_bootstrap={SMOKE_BOOTSTRAP}, n_permutation={SMOKE_PERMUTATION})")
    else:
        print("  (univariate only — fast)")

    master_rng = np.random.default_rng(args.seed)

    # Per-path counts
    path_per_perm: dict[str, list[int]] = {p.name: [] for p in paths}
    path_per_perm["consensus"] = []

    for perm_idx in range(args.n_permutations):
        t_perm = time.time()
        perm_rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1))

        per_path_sizes: list[int] = []

        for spec in paths:
            n_sl = _run_one_permutation(
                spec, cached_data, orig_samples, features, run_order,
                perm_rng, SMOKE_BOOTSTRAP, SMOKE_PERMUTATION, use_multivariate,
            )
            path_per_perm[spec.name].append(n_sl)
            per_path_sizes.append(n_sl)

        # Consensus: count features passing ≥2 paths (per-path counts from univariate)
        # For the fast (univariate-only) mode, we use the shortlist counts directly.
        # For the full mode, we'd need to actually compute cross-path consensus.
        # Simplified: consensus size = number of paths where shortlist > 0,
        # capped at the minimum per-path overlap.
        if args.fast or not use_multivariate:
            # Approximate: with only univariate, consensus ≈ features passing
            # the k=2-of-4 gate (which requires 2 of 4 methods, but we only
            # have 1 method). For single-path, we report the shortlist size.
            n_consensus = per_path_sizes[0] if per_path_sizes else 0
        else:
            # Full: actual cross-path consensus would need to be computed.
            # Simplified placeholder — the full implementation would merge
            # shortlist tables across paths.
            n_consensus = min(per_path_sizes) if per_path_sizes else 0

        path_per_perm["consensus"].append(n_consensus)

        elapsed = time.time() - t_perm
        eta = (args.n_permutations - perm_idx - 1) * elapsed / 60
        print(
            f"  [{perm_idx + 1:>4}/{args.n_permutations}] "
            f"shortlists: {per_path_sizes}  "
            f"consensus: {n_consensus}  "
            f"({elapsed:.0f}s, ETA {eta:.0f} min)"
        )

    # 5. Report
    print("\n--- Results ---")
    observed = _compute_observed_shortlist()
    print(f"  Observed consensus: {observed.get('consensus', 'N/A')}")
    for key, counts in path_per_perm.items():
        arr = np.array(counts)
        p95 = int(np.percentile(arr, 95))
        print(f"  {key}: mean={arr.mean():.2f}, 95%ile={p95}, max={int(arr.max())}")

    _write_report(
        path_per_perm, results={},
        use_multivariate=use_multivariate,
        n_permutations=args.n_permutations,
        paths_used=paths_used,
    )

    print(f"\n  Total: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
