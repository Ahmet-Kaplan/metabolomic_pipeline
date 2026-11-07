"""Stage 2 runner — orchestrates every plot + profile + index report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

from biodisc.eda import (
    blank_sample_ratio,
    dilution_response,
    missingness,
    mz_rt_density,
    pca,
    qc_vs_sample,
    run_order_drift,
)
from biodisc.eda._loader import load_stage1
from biodisc.eda.profile import profile_table
from biodisc.eda.report import ReportEntry, build_index

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class EdaConfig:
    """Toggle switches + parameters for Stage 2. Mirrors conf/eda/default.yaml."""

    emit_png: bool = True
    run_profiles: bool = True
    profile_features: bool = True
    profile_samples: bool = True
    profile_identification: bool = True  # slow; minimal mode
    run_plots: bool = True
    max_missingness_features: int = 5_000
    dilution_n_sample_features: int = 30
    dilution_r2_threshold: float = 0.7
    blank_pct_threshold: float = 20.0
    seed: int = 42


def run(
    stage_01_dir: Path,
    output_dir: Path,
    config: EdaConfig | None = None,
) -> dict[str, Path]:
    """Run the full Stage 2 EDA.

    Returns a map of logical-output-name → path (for CLI display).
    """
    cfg = config or EdaConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("eda_run_start", stage_01_dir=str(stage_01_dir), output_dir=str(output_dir))

    data = load_stage1(Path(stage_01_dir))
    log.info(
        "eda_stage1_loaded",
        features=len(data.features),
        samples=len(data.samples),
        abundances_long_rows=len(data.abundances_raw_long),
        identification_rows=len(data.identification),
    )

    outputs: dict[str, Path] = {}
    entries: list[ReportEntry] = []

    # ------- Layer A: ydata-profiling auto-reports --------
    profile_dir = output_dir / "profiles"
    if cfg.run_profiles:
        if cfg.profile_features:
            path = profile_table(
                data.features,
                output_path=profile_dir / "features.html",
                title="Profile — features.parquet",
            )
            outputs["profile_features"] = path
            entries.append(
                ReportEntry(
                    key="profile-A",
                    title="Auto-profile — features",
                    description=(
                        "ydata-profiling auto-report on per-feature metadata "
                        "(RT, m/z, CCS, charge, peak width, etc.). Fast sanity-check "
                        "for dtype / range regressions."
                    ),
                    html_href="profiles/features.html",
                )
            )
        if cfg.profile_samples:
            path = profile_table(
                data.samples,
                output_path=profile_dir / "samples.html",
                title="Profile — samples.parquet",
            )
            outputs["profile_samples"] = path
            entries.append(
                ReportEntry(
                    key="profile-B",
                    title="Auto-profile — samples",
                    description=(
                        "ydata-profiling auto-report on the 53-injection metadata "
                        "table (cell_line, injection_type, replicate, acq_date)."
                    ),
                    html_href="profiles/samples.html",
                )
            )
        if cfg.profile_identification:
            path = profile_table(
                data.identification,
                output_path=profile_dir / "identification.html",
                title="Profile — identification_long.parquet (minimal)",
                minimal=True,
            )
            outputs["profile_identification"] = path
            entries.append(
                ReportEntry(
                    key="profile-C",
                    title="Auto-profile — identification (minimal)",
                    description=(
                        "1 M-row table of candidate IDs. Minimal mode (no correlations "
                        "or interactions — otherwise slow). Good for per-column stats."
                    ),
                    html_href="profiles/identification.html",
                )
            )

    # ------- Layer B: metabolomics plots --------
    plots_dir = output_dir / "plots"
    if cfg.run_plots:
        spec: list[tuple[str, str, str, callable, dict]] = [  # type: ignore[valid-type]
            (
                "01_missingness",
                "1. Missingness heatmap",
                (
                    "Where are the NaN values? Rows = features sorted by missingness %, "
                    "columns = injections ordered by type. Dark = present, white = missing."
                ),
                missingness.plot,
                {"max_features": cfg.max_missingness_features},
            ),
            (
                "02_pca",
                "2. PCA by group",
                (
                    "2-component PCA on log-transformed abundances. Expected: tight QC "
                    "cluster, five cell-line clusters, blanks far from biology, dilutions "
                    "trailing from QC."
                ),
                pca.plot,
                {},
            ),
            (
                "03_qc_vs_sample_box",
                "3. QC vs sample intensity",
                (
                    "Box per injection, log₁₀(intensity). Healthy: QC boxes identical, "
                    "biological boxes similar, blanks much lower, dilutions descending."
                ),
                qc_vs_sample.plot,
                {},
            ),
            (
                "04_mz_rt_density",
                "4. m/z × RT feature map",
                (
                    "Every feature plotted at its (RT, m/z) coordinate, coloured by "
                    "log mean intensity across biology. The chemical-space star map."
                ),
                mz_rt_density.plot,
                {},
            ),
            (
                "05_qc_drift",
                "5. QC run-order drift",
                (
                    "Median log₁₀(intensity) across QC injections vs QC sequence number. "
                    "Flat = stable; trend = detector/column drift. "
                    "Caveat: assumes QC numbering = chronological injection order "
                    "(to confirm with MS operator)."
                ),
                run_order_drift.plot,
                {},
            ),
            (
                "06_blank_sample_ratio",
                "6. Blank vs sample ratio (Stage 6 preview)",
                (
                    "log₁₀(mean_sample / mean_blank) per feature. Overlay marks the Stage 6 "
                    "drop threshold equivalent to Saeed's blank-pct formula. Preview of how "
                    "many features will survive the blank filter."
                ),
                blank_sample_ratio.plot,
                {"default_blank_pct_threshold": cfg.blank_pct_threshold},
            ),
            (
                "07_dilution_response",
                "7. Dilution response + R² distribution (Stage 7 preview)",
                (
                    "Stratified sample of per-feature dilution curves plus R²_linear and "
                    "R²_quadratic histograms across all features. Previews how many features "
                    "will survive the R² ≥ 0.7 rule."
                ),
                dilution_response.plot,
                {
                    "n_sample_features": cfg.dilution_n_sample_features,
                    "r2_threshold": cfg.dilution_r2_threshold,
                    "seed": cfg.seed,
                },
            ),
        ]
        for key, title, desc, plot_fn, extra_kwargs in spec:
            paths = plot_fn(
                data,
                output_dir=plots_dir,
                emit_png=cfg.emit_png,
                **extra_kwargs,
            )
            outputs[key] = paths["html"]
            entries.append(
                ReportEntry(
                    key=key,
                    title=title,
                    description=desc,
                    html_href=f"plots/{paths['html'].name}",
                    png_href=(f"plots/{paths['png'].name}" if "png" in paths else None),
                )
            )

    # ------- Index report --------
    extra_info = {
        "Features": f"{len(data.features):,}",
        "Samples": f"{len(data.samples):,}",
        "Abundance rows (long)": f"{len(data.abundances_raw_long):,}",
        "Identification candidates": f"{len(data.identification):,}",
    }
    index_path = build_index(
        output_path=output_dir / "index.html",
        entries=entries,
        extra_info=extra_info,
    )
    outputs["index"] = index_path
    log.info("eda_run_complete", outputs=len(outputs))
    return outputs
