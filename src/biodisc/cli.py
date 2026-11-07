"""Typer CLI entrypoint — composes with Hydra config.

Command structure (designed for Stages 1–17):
::

    biodisc parse                         # Stage 1  (implemented)
    biodisc eda                           # Stage 2  (pending)
    biodisc filter presence               # Stage 3  (pending)
    biodisc filter qc-cv                  # Stage 4  (pending)
    biodisc filter sample-qc-ratio        # Stage 5  (pending)
    biodisc filter blank                  # Stage 6  (pending)
    biodisc filter dilution               # Stage 7  (pending)
    biodisc bench imputation              # Stage 8b (pending)
    biodisc bench drift                   # Stage 9b (pending)
    biodisc bench normalisation           # Stage 11b(pending)
    biodisc bench transformation          # Stage 12b(pending)
    biodisc run all                       # end-to-end orchestrator
    biodisc report                        # HTML + PDF report builder
    biodisc version

All commands accept the same Hydra-style overrides at the end:

    biodisc parse parse.emit_csv_mirror=false
    biodisc filter presence filter.presence_pct=0.70
"""

from __future__ import annotations

from pathlib import Path

import typer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from biodisc.audit import AuditLogger
from biodisc.audit_ledger import rebuild as rebuild_ledger
from biodisc.io.progenesis import ParseConfig, ProgenesisParser
from biodisc.logging_setup import configure_logging
from biodisc.seeds import set_seed

# ============================================================
# Root app + sub-apps (placeholders for future stages)
# ============================================================
app = typer.Typer(
    name="biodisc",
    help="Biomarker Discovery MLOps — untargeted LC-MS pipeline.",
    add_completion=False,
    no_args_is_help=True,
)

# Stage 3–7 filters live under `biodisc filter <stage>`
filter_app = typer.Typer(
    name="filter", help="Feature-level filters (Stages 3–7).", no_args_is_help=True
)
app.add_typer(filter_app, name="filter")

# Stage 8b, 9b, 11b, 12b benchmarks live under `biodisc bench <stage>`
bench_app = typer.Typer(name="bench", help="Method-selection benchmarks.", no_args_is_help=True)
app.add_typer(bench_app, name="bench")

# Orchestration
run_app = typer.Typer(name="run", help="Run multi-stage pipelines.", no_args_is_help=True)
app.add_typer(run_app, name="run")

# Audit utilities (`biodisc audit ledger ...`)
audit_app = typer.Typer(
    name="audit",
    help="Cross-stage audit utilities (master ledger).",
    no_args_is_help=True,
)
app.add_typer(audit_app, name="audit")

# External-reference data builders (`biodisc external <build-step>`)
external_app = typer.Typer(
    name="external",
    help="Build external reference tables (HMDB, CCS) for Stage 8 + later stages.",
    no_args_is_help=True,
)
app.add_typer(external_app, name="external")

# Phase 2 multiverse (`biodisc multiverse run`). Launched explicitly — the full
# run takes hours, so it is deliberately NOT part of the auto `dvc repro` DAG.
multiverse_app = typer.Typer(
    name="multiverse",
    help="Phase 2 — multiverse discovery (Stages 11–17). Run explicitly; not in dvc repro.",
    no_args_is_help=True,
)
app.add_typer(multiverse_app, name="multiverse")


# ============================================================
# Shared helper — auto-rebuild the master ledger after a filter stage
# ============================================================
def _maybe_rebuild_ledger(cfg: DictConfig, *, full: bool, skip: bool) -> None:
    """Rebuild the master feature ledger unless `skip` (--no-ledger) is set."""
    if skip:
        return
    try:
        outs = rebuild_ledger(
            interim_dir=Path(cfg.paths.interim_dir),
            audit_dir=Path(cfg.paths.audit_dir),
            full=full,
            emit_csv=cfg.parse.emit_csv_mirror,
        )
        for k, p in outs.items():
            console.print(f"[dim]  ledger {k}:[/] {p}")
    except Exception as exc:  # pragma: no cover — defensive only
        console.print(f"[yellow]⚠ Ledger rebuild failed (non-fatal):[/] {exc}")


console = Console()

# Default conf directory — overridable via env var PROJECT_ROOT or --config-dir
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "conf"


# ============================================================
# Shared helpers
# ============================================================
def _load_config(config_dir: Path, overrides: list[str]) -> DictConfig:
    """Load the Hydra base config plus optional CLI overrides."""
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def _bootstrap(cfg: DictConfig) -> None:
    """Logging + seed setup common to every subcommand."""
    configure_logging(
        level=cfg.logging.level,
        json_output=cfg.logging.json_output,
        include_caller=cfg.logging.include_caller,
        log_file=Path(cfg.logging.log_file) if cfg.logging.log_file else None,
    )
    set_seed(cfg.seed)


# ============================================================
# STAGE 1 — parse
# ============================================================
@app.command()
def parse(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR,
        "--config-dir",
        "-c",
        help="Hydra config directory.",
    ),
    overrides: list[str] = typer.Argument(
        None,
        help="Hydra-style overrides, e.g. parse.emit_wide=false",
    ),
) -> None:
    """Stage 1 — parse Progenesis QI CSVs into tidy parquet + CSV outputs."""
    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    console.rule("[bold cyan]Stage 1 — Parse Progenesis QI CSVs")
    console.print(f"[dim]Config dir:[/] {config_dir}")
    console.print(f"[dim]Raw dir:[/]    {cfg.paths.raw_dir}")
    console.print(f"[dim]Output dir:[/] {cfg.paths.stage_01_parsed}")
    console.print(f"[dim]Audit dir:[/]  {cfg.paths.audit_dir}")
    console.print(f"[dim]Seed:[/]       {cfg.seed}")

    # Sample exclusions: optional, defaults to empty tuple. Hydra emits a list;
    # we coerce to tuple for the frozen dataclass.
    excludes = tuple(
        OmegaConf.to_container(cfg.parse.get("exclude_samples", []), resolve=True) or []
    )
    if excludes:
        console.print(f"[dim]Excludes:[/]   {list(excludes)}")

    parse_cfg = ParseConfig(
        measurement_file=cfg.parse.measurement_file,
        blank_file=cfg.parse.blank_file,
        dilution_file=cfg.parse.dilution_file,
        identification_file=cfg.parse.identification_file,
        sample_id_pattern=cfg.parse.sample_id_pattern,
        emit_parquet=cfg.parse.emit_parquet,
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
        emit_wide=cfg.parse.emit_wide,
        emit_long=cfg.parse.emit_long,
        canonical_abundance=cfg.parse.canonical_abundance,
        exclude_samples=excludes,
    )
    audit = AuditLogger(
        stage="01_parse",
        params=OmegaConf.to_container(cfg.parse, resolve=True),  # type: ignore[arg-type]
    )
    parser = ProgenesisParser(parse_cfg, audit)
    outputs = parser.parse_all(
        input_dir=Path(cfg.paths.raw_dir),
        output_dir=Path(cfg.paths.stage_01_parsed),
    )
    audit.flush(
        output_dir=Path(cfg.paths.audit_dir),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )

    # Summary table
    t = Table(title="Stage 1 outputs", show_lines=False, title_style="bold green")
    t.add_column("Logical name", style="cyan")
    t.add_column("Path", style="dim")
    t.add_column("Size (MB)", justify="right")
    for name, path in outputs.items():
        size_mb = path.stat().st_size / 1024 / 1024 if path.exists() else 0
        t.add_row(name, str(path), f"{size_mb:.2f}")
    console.print(t)
    console.print("[bold green]✓[/] Stage 1 complete.")


# ============================================================
# STAGE 2 — EDA
# ============================================================
@app.command()
def eda(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR,
        "--config-dir",
        "-c",
        help="Hydra config directory.",
    ),
    overrides: list[str] = typer.Argument(
        None,
        help="Hydra-style overrides, e.g. eda.emit_png=false eda.run_profiles=false",
    ),
) -> None:
    """Stage 2 — exploratory data analysis (ydata profiles + 7 metabolomics plots)."""
    from biodisc.eda import run as run_eda
    from biodisc.eda._runner import EdaConfig

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    console.rule("[bold cyan]Stage 2 — Exploratory Data Analysis")
    console.print(f"[dim]Stage-1 dir:[/] {cfg.paths.stage_01_parsed}")
    console.print(f"[dim]Output dir:[/] {cfg.paths.stage_02_eda}")
    console.print(f"[dim]Seed:[/]       {cfg.seed}")

    eda_cfg = EdaConfig(
        emit_png=cfg.eda.emit_png,
        run_profiles=cfg.eda.run_profiles,
        profile_features=cfg.eda.profile_features,
        profile_samples=cfg.eda.profile_samples,
        profile_identification=cfg.eda.profile_identification,
        run_plots=cfg.eda.run_plots,
        max_missingness_features=cfg.eda.max_missingness_features,
        dilution_n_sample_features=cfg.eda.dilution_n_sample_features,
        dilution_r2_threshold=cfg.eda.dilution_r2_threshold,
        blank_pct_threshold=cfg.eda.blank_pct_threshold,
        seed=cfg.seed,
    )
    outputs = run_eda(
        stage_01_dir=Path(cfg.paths.stage_01_parsed),
        output_dir=Path(cfg.paths.stage_02_eda),
        config=eda_cfg,
    )

    t = Table(title="Stage 2 outputs", show_lines=False, title_style="bold green")
    t.add_column("Key", style="cyan")
    t.add_column("Path", style="dim")
    for key, path in outputs.items():
        t.add_row(key, str(path))
    console.print(t)
    console.print(f"[bold green]✓[/] Stage 2 complete. Open {outputs['index']} to browse.")


# ============================================================
# STAGES 3–7 — filter sub-commands (placeholders)
# ============================================================
@filter_app.command("presence")
def filter_presence(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR,
        "--config-dir",
        "-c",
        help="Hydra config directory.",
    ),
    no_ledger: bool = typer.Option(
        False, "--no-ledger", help="Skip auto-rebuild of master_feature_ledger after this stage."
    ),
    ledger_full: bool = typer.Option(
        False,
        "--ledger-full",
        help="Also emit the wide (--full) ledger including every metric column.",
    ),
    overrides: list[str] = typer.Argument(
        None,
        help="Hydra-style overrides, e.g. filter.presence.presence_pct=0.70",
    ),
) -> None:
    """Stage 3 — modified 80 % rule (keep features present in ≥ T of any group)."""
    import pandas as pd

    from biodisc.filters._presence_report import build_report
    from biodisc.filters.presence import (
        PresenceFilterConfig,
        write_outputs,
    )
    from biodisc.filters.presence import (
        run as run_presence,
    )

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    console.rule("[bold cyan]Stage 3 — Modified 80 % rule (presence filter)")
    console.print(f"[dim]Stage-1 dir:[/] {cfg.paths.stage_01_parsed}")
    console.print(f"[dim]Output dir:[/] {cfg.paths.stage_03_presence}")
    console.print(f"[dim]Threshold:[/]  presence_pct = {cfg.filter.presence.presence_pct}")

    stage01 = Path(cfg.paths.stage_01_parsed)
    abundances = pd.read_parquet(stage01 / "abundances_raw_long.parquet")
    samples = pd.read_parquet(stage01 / "samples.parquet")
    features = pd.read_parquet(stage01 / "features.parquet")

    pf_cfg = PresenceFilterConfig(
        presence_pct=cfg.filter.presence.presence_pct,
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit = AuditLogger(
        stage="03_presence",
        params=OmegaConf.to_container(cfg.filter.presence, resolve=True),  # type: ignore[arg-type]
    )
    result = run_presence(
        abundances_long=abundances,
        samples=samples,
        config=pf_cfg,
        audit=audit,
    )
    paths = write_outputs(
        result,
        output_dir=Path(cfg.paths.stage_03_presence),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit.flush(
        output_dir=Path(cfg.paths.audit_dir),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )

    # Build the HTML report (plots + index)
    report_dir = Path(cfg.paths.reports_dir) / "03_presence"
    report_outputs = build_report(
        metrics=result.metrics,
        features=features,
        output_dir=report_dir,
        threshold=pf_cfg.presence_pct,
        n_input=result.n_input,
        n_kept=result.n_kept,
        n_dropped=result.n_dropped,
        emit_png=True,
    )

    # Summary table
    t = Table(title="Stage 3 summary", show_lines=False, title_style="bold green")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Input features", f"{result.n_input:,}")
    t.add_row("Kept", f"{result.n_kept:,}")
    t.add_row("Dropped", f"{result.n_dropped:,}")
    t.add_row("Drop %", f"{result.drop_pct:.2f} %")
    t.add_row("Threshold", f"{pf_cfg.presence_pct}")
    console.print(t)
    for k, p in paths.items():
        console.print(f"[dim]  {k}:[/] {p}")
    console.print(f"[dim]  report:[/] {report_outputs['index']}")
    _maybe_rebuild_ledger(cfg, full=False, skip=no_ledger)
    if ledger_full and not no_ledger:
        _maybe_rebuild_ledger(cfg, full=True, skip=False)
    console.print("[bold green]✓[/] Stage 3 complete.")


@filter_app.command("qc-cv")
def filter_qc_cv(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR,
        "--config-dir",
        "-c",
        help="Hydra config directory.",
    ),
    no_ledger: bool = typer.Option(False, "--no-ledger", help="Skip auto-rebuild of ledger."),
    ledger_full: bool = typer.Option(False, "--ledger-full", help="Also emit the full ledger."),
    overrides: list[str] = typer.Argument(
        None,
        help="Hydra-style overrides, e.g. filter.qc_cv.max_cv_pct=50",
    ),
) -> None:
    """Stage 4 — drop features with CV % across the 16 QCs > max_cv_pct."""
    import pandas as pd

    from biodisc.filters._qc_cv_report import build_report
    from biodisc.filters.qc_cv import (
        QcCvFilterConfig,
        write_outputs,
    )
    from biodisc.filters.qc_cv import (
        run as run_qc_cv,
    )

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    console.rule("[bold cyan]Stage 4 — QC-CV filter")
    console.print(f"[dim]Stage-1 dir:[/]   {cfg.paths.stage_01_parsed}")
    console.print(f"[dim]Stage-3 dir:[/]   {cfg.paths.stage_03_presence}")
    console.print(f"[dim]Output dir:[/]    {cfg.paths.stage_04_qc_cv}")
    console.print(
        f"[dim]Threshold:[/]    max_cv_pct = {cfg.filter.qc_cv.max_cv_pct} %  · "
        f"min_qc_detections = {cfg.filter.qc_cv.min_qc_detections}"
    )

    stage01 = Path(cfg.paths.stage_01_parsed)
    abundances = pd.read_parquet(stage01 / "abundances_raw_long.parquet")
    samples = pd.read_parquet(stage01 / "samples.parquet")
    stage03_kept = pd.read_parquet(Path(cfg.paths.stage_03_presence) / "kept_features.parquet")

    qc_cfg = QcCvFilterConfig(
        max_cv_pct=cfg.filter.qc_cv.max_cv_pct,
        min_qc_detections=cfg.filter.qc_cv.min_qc_detections,
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit = AuditLogger(
        stage="04_qc_cv",
        params=OmegaConf.to_container(cfg.filter.qc_cv, resolve=True),  # type: ignore[arg-type]
    )
    result = run_qc_cv(
        abundances_long=abundances,
        samples=samples,
        kept_features=stage03_kept,
        config=qc_cfg,
        audit=audit,
    )
    paths = write_outputs(
        result,
        output_dir=Path(cfg.paths.stage_04_qc_cv),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit.flush(
        output_dir=Path(cfg.paths.audit_dir),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )

    report_dir = Path(cfg.paths.reports_dir) / "04_qc_cv"
    report_outputs = build_report(
        metrics=result.metrics,
        abundances_long=abundances,
        samples=samples,
        output_dir=report_dir,
        threshold=qc_cfg.max_cv_pct,
        min_qc_detections=qc_cfg.min_qc_detections,
        n_input=result.n_input,
        n_kept=result.n_kept,
        n_dropped_cv=result.n_dropped_cv,
        n_dropped_detections=result.n_dropped_detections,
        emit_png=True,
    )

    t = Table(title="Stage 4 summary", show_lines=False, title_style="bold green")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Input (Stage-3-kept)", f"{result.n_input:,}")
    t.add_row("Kept", f"{result.n_kept:,}")
    t.add_row("Dropped — CV exceeds thr", f"{result.n_dropped_cv:,}")
    t.add_row("Dropped — insufficient QC detections", f"{result.n_dropped_detections:,}")
    t.add_row("Drop %", f"{result.drop_pct:.2f} %")
    t.add_row("CV threshold", f"{qc_cfg.max_cv_pct} %")
    console.print(t)
    for k, p in paths.items():
        console.print(f"[dim]  {k}:[/] {p}")
    console.print(f"[dim]  report:[/] {report_outputs['index']}")
    _maybe_rebuild_ledger(cfg, full=False, skip=no_ledger)
    if ledger_full and not no_ledger:
        _maybe_rebuild_ledger(cfg, full=True, skip=False)
    console.print("[bold green]✓[/] Stage 4 complete.")


@filter_app.command("sample-qc-ratio")
def filter_sample_qc_ratio(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR,
        "--config-dir",
        "-c",
        help="Hydra config directory.",
    ),
    no_ledger: bool = typer.Option(False, "--no-ledger", help="Skip auto-rebuild of ledger."),
    ledger_full: bool = typer.Option(False, "--ledger-full", help="Also emit the full ledger."),
    overrides: list[str] = typer.Argument(
        None,
        help="Hydra-style overrides, e.g. filter.sample_qc_ratio.min_ratio=2.0",
    ),
) -> None:
    """Stage 5 — drop features where RSD_samples / RSD_QC ≤ min_ratio."""
    import pandas as pd

    from biodisc.filters._sample_qc_ratio_report import build_report
    from biodisc.filters.sample_qc_ratio import (
        SampleQcRatioFilterConfig,
        write_outputs,
    )
    from biodisc.filters.sample_qc_ratio import (
        run as run_filter,
    )

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    console.rule("[bold cyan]Stage 5 — Sample-RSD / QC-RSD ratio")
    console.print(f"[dim]Stage-1 dir:[/]   {cfg.paths.stage_01_parsed}")
    console.print(f"[dim]Stage-4 dir:[/]   {cfg.paths.stage_04_qc_cv}")
    console.print(f"[dim]Output dir:[/]    {cfg.paths.stage_05_sample_qc_ratio}")
    console.print(
        f"[dim]Threshold:[/]    min_ratio = {cfg.filter.sample_qc_ratio.min_ratio} · "
        f"min_sample_det = {cfg.filter.sample_qc_ratio.min_sample_detections} · "
        f"min_qc_det = {cfg.filter.sample_qc_ratio.min_qc_detections}"
    )

    stage01 = Path(cfg.paths.stage_01_parsed)
    abundances = pd.read_parquet(stage01 / "abundances_raw_long.parquet")
    samples = pd.read_parquet(stage01 / "samples.parquet")
    stage04_kept = pd.read_parquet(Path(cfg.paths.stage_04_qc_cv) / "kept_features.parquet")

    f_cfg = SampleQcRatioFilterConfig(
        min_ratio=cfg.filter.sample_qc_ratio.min_ratio,
        min_sample_detections=cfg.filter.sample_qc_ratio.min_sample_detections,
        min_qc_detections=cfg.filter.sample_qc_ratio.min_qc_detections,
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit = AuditLogger(
        stage="05_sample_qc_ratio",
        params=OmegaConf.to_container(cfg.filter.sample_qc_ratio, resolve=True),  # type: ignore[arg-type]
    )
    result = run_filter(
        abundances_long=abundances,
        samples=samples,
        kept_features=stage04_kept["feature_id"].tolist(),
        config=f_cfg,
        audit=audit,
    )
    paths = write_outputs(
        result,
        output_dir=Path(cfg.paths.stage_05_sample_qc_ratio),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit.flush(
        output_dir=Path(cfg.paths.audit_dir),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )

    report_dir = Path(cfg.paths.reports_dir) / "05_sample_qc_ratio"
    report_outputs = build_report(
        metrics=result.metrics,
        output_dir=report_dir,
        threshold=f_cfg.min_ratio,
        n_input=result.n_input,
        n_kept=result.n_kept,
        n_dropped_ratio=result.n_dropped_ratio,
        n_dropped_detections=result.n_dropped_detections,
        n_dropped_undefined=result.n_dropped_undefined,
        emit_png=True,
    )

    t = Table(title="Stage 5 summary", show_lines=False, title_style="bold green")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Input (Stage-4-kept)", f"{result.n_input:,}")
    t.add_row("Kept", f"{result.n_kept:,}")
    t.add_row("Dropped — ratio ≤ threshold", f"{result.n_dropped_ratio:,}")
    t.add_row("Dropped — insufficient detections", f"{result.n_dropped_detections:,}")
    t.add_row("Dropped — ratio undefined", f"{result.n_dropped_undefined:,}")
    t.add_row("Drop %", f"{result.drop_pct:.2f} %")
    t.add_row("Threshold", f"{f_cfg.min_ratio}")
    console.print(t)
    for k, p in paths.items():
        console.print(f"[dim]  {k}:[/] {p}")
    console.print(f"[dim]  report:[/] {report_outputs['index']}")
    _maybe_rebuild_ledger(cfg, full=False, skip=no_ledger)
    if ledger_full and not no_ledger:
        _maybe_rebuild_ledger(cfg, full=True, skip=False)
    console.print("[bold green]✓[/] Stage 5 complete.")


@filter_app.command("blank")
def filter_blank(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR,
        "--config-dir",
        "-c",
        help="Hydra config directory.",
    ),
    no_ledger: bool = typer.Option(False, "--no-ledger", help="Skip auto-rebuild of ledger."),
    ledger_full: bool = typer.Option(False, "--ledger-full", help="Also emit the full ledger."),
    overrides: list[str] = typer.Argument(
        None,
        help="Hydra-style overrides, e.g. filter.blank.max_blank_pct=50",
    ),
) -> None:
    """Stage 6 — drop features where 100 × mean(blank) / mean(sample+QC) > max_blank_pct."""
    import pandas as pd

    from biodisc.filters._blank_report import build_report
    from biodisc.filters.blank import BlankFilterConfig, write_outputs
    from biodisc.filters.blank import run as run_filter

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    console.rule("[bold cyan]Stage 6 — Blank filter (% formula)")
    console.print(f"[dim]Stage-1 dir:[/]   {cfg.paths.stage_01_parsed}")
    console.print(f"[dim]Stage-5 dir:[/]   {cfg.paths.stage_05_sample_qc_ratio}")
    console.print(f"[dim]Output dir:[/]    {cfg.paths.stage_06_blank}")
    console.print(
        f"[dim]Threshold:[/]    max_blank_pct = {cfg.filter.blank.max_blank_pct} %  · "
        f"min_sq_det = {cfg.filter.blank.min_sq_detections}"
    )

    stage01 = Path(cfg.paths.stage_01_parsed)
    abundances = pd.read_parquet(stage01 / "abundances_raw_long.parquet")
    blanks = pd.read_parquet(stage01 / "blanks_long.parquet")
    samples = pd.read_parquet(stage01 / "samples.parquet")
    stage5_kept = pd.read_parquet(
        Path(cfg.paths.stage_05_sample_qc_ratio) / "kept_features.parquet"
    )

    f_cfg = BlankFilterConfig(
        max_blank_pct=cfg.filter.blank.max_blank_pct,
        min_sq_detections=cfg.filter.blank.min_sq_detections,
        block=cfg.filter.blank.block,
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit = AuditLogger(
        stage="06_blank",
        params=OmegaConf.to_container(cfg.filter.blank, resolve=True),  # type: ignore[arg-type]
    )
    result = run_filter(
        abundances_long=abundances,
        blanks_long=blanks,
        samples=samples,
        kept_features=stage5_kept["feature_id"].tolist(),
        config=f_cfg,
        audit=audit,
    )
    paths = write_outputs(
        result,
        output_dir=Path(cfg.paths.stage_06_blank),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit.flush(
        output_dir=Path(cfg.paths.audit_dir),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )

    report_dir = Path(cfg.paths.reports_dir) / "06_blank"
    report_outputs = build_report(
        metrics=result.metrics,
        output_dir=report_dir,
        threshold=f_cfg.max_blank_pct,
        n_input=result.n_input,
        n_kept=result.n_kept,
        n_dropped_pct=result.n_dropped_pct,
        n_dropped_detections=result.n_dropped_detections,
        n_dropped_undefined=result.n_dropped_undefined,
        emit_png=True,
    )

    t = Table(title="Stage 6 summary", show_lines=False, title_style="bold green")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Input (Stage-5-kept)", f"{result.n_input:,}")
    t.add_row("Kept", f"{result.n_kept:,}")
    t.add_row("Dropped — blank_pct > thr", f"{result.n_dropped_pct:,}")
    t.add_row("Dropped — insufficient sample/QC det", f"{result.n_dropped_detections:,}")
    t.add_row("Dropped — undefined", f"{result.n_dropped_undefined:,}")
    t.add_row("Drop %", f"{result.drop_pct:.2f} %")
    t.add_row("Threshold", f"blank_pct > {f_cfg.max_blank_pct} %")
    console.print(t)
    for k, p in paths.items():
        console.print(f"[dim]  {k}:[/] {p}")
    console.print(f"[dim]  report:[/] {report_outputs['index']}")
    _maybe_rebuild_ledger(cfg, full=False, skip=no_ledger)
    if ledger_full and not no_ledger:
        _maybe_rebuild_ledger(cfg, full=True, skip=False)
    console.print("[bold green]✓[/] Stage 6 complete.")


@filter_app.command("dilution")
def filter_dilution(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR,
        "--config-dir",
        "-c",
        help="Hydra config directory.",
    ),
    no_ledger: bool = typer.Option(False, "--no-ledger", help="Skip auto-rebuild of ledger."),
    ledger_full: bool = typer.Option(False, "--ledger-full", help="Also emit the full ledger."),
    overrides: list[str] = typer.Argument(
        None,
        help="Hydra-style overrides, e.g. filter.dilution.min_r2_linear=0.6",
    ),
) -> None:
    """Stage 7 — drop features with R²_linear < min_r2_linear over the 4 dilution points."""
    import pandas as pd

    from biodisc.filters._dilution_report import build_report
    from biodisc.filters.dilution import DilutionFilterConfig, write_outputs
    from biodisc.filters.dilution import run as run_filter

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    console.rule("[bold cyan]Stage 7 — Dilution linearity (R²_linear)")
    console.print(f"[dim]Stage-1 dir:[/]   {cfg.paths.stage_01_parsed}")
    console.print(f"[dim]Stage-6 dir:[/]   {cfg.paths.stage_06_blank}")
    console.print(f"[dim]Output dir:[/]    {cfg.paths.stage_07_dilution}")
    console.print(
        f"[dim]Threshold:[/]    min_r2_linear = {cfg.filter.dilution.min_r2_linear}  · "
        f"min_dilution_det = {cfg.filter.dilution.min_dilution_detections}"
    )

    stage01 = Path(cfg.paths.stage_01_parsed)
    dilution = pd.read_parquet(stage01 / "dilution_ladder_long.parquet")
    stage6_kept = pd.read_parquet(
        Path(cfg.paths.stage_06_blank) / "kept_features.parquet"
    )

    f_cfg = DilutionFilterConfig(
        min_r2_linear=cfg.filter.dilution.min_r2_linear,
        min_dilution_detections=cfg.filter.dilution.min_dilution_detections,
        block=cfg.filter.dilution.block,
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit = AuditLogger(
        stage="07_dilution",
        params=OmegaConf.to_container(cfg.filter.dilution, resolve=True),  # type: ignore[arg-type]
    )
    result = run_filter(
        dilution_long=dilution,
        kept_features=stage6_kept["feature_id"].tolist(),
        config=f_cfg,
        audit=audit,
    )
    paths = write_outputs(
        result,
        output_dir=Path(cfg.paths.stage_07_dilution),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit.flush(
        output_dir=Path(cfg.paths.audit_dir),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )

    report_dir = Path(cfg.paths.reports_dir) / "07_dilution"
    report_outputs = build_report(
        metrics=result.metrics,
        output_dir=report_dir,
        threshold=f_cfg.min_r2_linear,
        n_input=result.n_input,
        n_kept=result.n_kept,
        n_dropped_r2=result.n_dropped_r2,
        n_dropped_detections=result.n_dropped_detections,
        n_dropped_undefined=result.n_dropped_undefined,
        emit_png=True,
    )

    t = Table(title="Stage 7 summary", show_lines=False, title_style="bold green")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Input (Stage-6-kept)", f"{result.n_input:,}")
    t.add_row("Kept", f"{result.n_kept:,}")
    t.add_row("Dropped — R² < thr", f"{result.n_dropped_r2:,}")
    t.add_row("Dropped — insufficient detections", f"{result.n_dropped_detections:,}")
    t.add_row("Dropped — R² undefined", f"{result.n_dropped_undefined:,}")
    t.add_row("Drop %", f"{result.drop_pct:.2f} %")
    t.add_row("Threshold", f"R² ≥ {f_cfg.min_r2_linear}")
    console.print(t)
    for k, p in paths.items():
        console.print(f"[dim]  {k}:[/] {p}")
    console.print(f"[dim]  report:[/] {report_outputs['index']}")
    _maybe_rebuild_ledger(cfg, full=False, skip=no_ledger)
    if ledger_full and not no_ledger:
        _maybe_rebuild_ledger(cfg, full=True, skip=False)
    console.print("[bold green]✓[/] Stage 7 complete.")


# ============================================================
# STAGE 8 — CCS error filter (adduct-aware, quality-floored)
# ============================================================
@filter_app.command("ccs")
def filter_ccs(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR, "--config-dir", "-c", help="Hydra config directory.",
    ),
    no_ledger: bool = typer.Option(False, "--no-ledger", help="Skip auto-rebuild of ledger."),
    ledger_full: bool = typer.Option(False, "--ledger-full", help="Also emit the full ledger."),
    overrides: list[str] = typer.Argument(
        None,
        help="Hydra-style overrides, e.g. filter.ccs.max_ccs_err_pct=2.5",
    ),
) -> None:
    """Stage 8 — drop features whose best (candidate, adduct) match has CCS error > threshold."""
    import pandas as pd

    from biodisc.filters._ccs_report import build_report
    from biodisc.filters.ccs import CCSFilterConfig, write_outputs
    from biodisc.filters.ccs import run as run_filter

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    console.rule("[bold cyan]Stage 8 — CCS error filter (adduct-aware)")
    console.print(f"[dim]Stage-1 dir:[/]   {cfg.paths.stage_01_parsed}")
    console.print(f"[dim]Stage-7 dir:[/]   {cfg.paths.stage_07_dilution}")
    console.print(f"[dim]Output dir:[/]    {cfg.paths.stage_08_ccs}")
    console.print(f"[dim]Reference:[/]     {cfg.paths.ccs_reference_parquet}")
    console.print(
        f"[dim]Thresholds:[/]   ccs_err ≤ {cfg.filter.ccs.max_ccs_err_pct} % · "
        f"score ≥ {cfg.filter.ccs.min_progenesis_score} · "
        f"|ppm| ≤ {cfg.filter.ccs.max_mass_error_ppm}"
    )

    stage01 = Path(cfg.paths.stage_01_parsed)
    features = pd.read_parquet(stage01 / "features.parquet")
    ident = pd.read_parquet(stage01 / "identification_long.parquet")
    stage7_kept = pd.read_parquet(
        Path(cfg.paths.stage_07_dilution) / "kept_features.parquet"
    )

    ref_path = Path(cfg.paths.ccs_reference_parquet)
    if not ref_path.exists():
        console.print(
            f"[bold red]✗[/] Unified CCS reference not found at {ref_path}.\n"
            f"[dim]Build it first:[/] biodisc external build-ccs-reference"
        )
        raise typer.Exit(code=2)
    reference = pd.read_parquet(ref_path)

    f_cfg = CCSFilterConfig(
        max_ccs_err_pct=float(cfg.filter.ccs.max_ccs_err_pct),
        min_progenesis_score=float(cfg.filter.ccs.min_progenesis_score),
        max_mass_error_ppm=float(cfg.filter.ccs.max_mass_error_ppm),
        twims_bias_dtims=float(cfg.filter.ccs.twims_bias_dtims),
        twims_bias_tims=float(cfg.filter.ccs.twims_bias_tims),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit = AuditLogger(
        stage="08_ccs",
        params=OmegaConf.to_container(cfg.filter.ccs, resolve=True),  # type: ignore[arg-type]
    )
    result = run_filter(
        features=features,
        identification=ident,
        reference=reference,
        kept_features=stage7_kept["feature_id"].tolist(),
        config=f_cfg,
        audit=audit,
    )
    paths = write_outputs(
        result,
        output_dir=Path(cfg.paths.stage_08_ccs),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit.flush(
        output_dir=Path(cfg.paths.audit_dir),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )

    n_dropped_ccs = int((result.metrics["decision"] == "dropped_ccs_err").sum())
    n_dropped_quality = int((result.metrics["decision"] == "dropped_quality_floor").sum())
    n_kept_ccs = int((result.metrics["decision"] == "kept").sum())

    report_dir = Path(cfg.paths.reports_dir) / "08_ccs"
    counts = {
        "n_input": result.n_input,
        "n_kept_total": result.n_kept,
        "n_kept_ccs": n_kept_ccs,
        "n_passthrough_no_candidate": result.n_passthrough_no_candidate,
        "n_passthrough_no_reference": result.n_passthrough_no_reference,
        "n_dropped_ccs_err": n_dropped_ccs,
        "n_dropped_quality_floor": n_dropped_quality,
        "min_score": int(f_cfg.min_progenesis_score),
        "max_ppm": int(f_cfg.max_mass_error_ppm),
    }
    report_outputs = build_report(
        metrics=result.metrics,
        output_dir=report_dir,
        threshold=f_cfg.max_ccs_err_pct,
        counts=counts,
        emit_png=True,
    )

    t = Table(title="Stage 8 summary", show_lines=False, title_style="bold green")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Input (Stage-7-active)", f"{result.n_input:,}")
    t.add_row("Filterable (CCS check applied)", f"{result.n_filterable:,}")
    t.add_row("Kept — CCS validated", f"{n_kept_ccs:,}")
    t.add_row("Kept — pass-through (no candidate)", f"{result.n_passthrough_no_candidate:,}")
    t.add_row("Kept — pass-through (no reference)", f"{result.n_passthrough_no_reference:,}")
    t.add_row("Kept (TOTAL)", f"{result.n_kept:,}")
    t.add_row("Dropped — CCS err > thr", f"{n_dropped_ccs:,}")
    t.add_row("Dropped — quality floor failed", f"{n_dropped_quality:,}")
    t.add_row("Drop %", f"{result.drop_pct:.2f} %")
    t.add_row("Threshold", f"err ≤ {f_cfg.max_ccs_err_pct} %")
    console.print(t)
    for k, p in paths.items():
        console.print(f"[dim]  {k}:[/] {p}")
    console.print(f"[dim]  report:[/] {report_outputs['index']}")
    _maybe_rebuild_ledger(cfg, full=False, skip=no_ledger)
    if ledger_full and not no_ledger:
        _maybe_rebuild_ledger(cfg, full=True, skip=False)
    console.print("[bold green]✓[/] Stage 8 complete.")


# ============================================================
# STAGE 9 — Identification candidate ranking (annotation-only, no drops)
# ============================================================
@filter_app.command("id-ranking")
def filter_id_ranking(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR, "--config-dir", "-c", help="Hydra config directory.",
    ),
    no_ledger: bool = typer.Option(False, "--no-ledger", help="Skip auto-rebuild of ledger."),
    ledger_full: bool = typer.Option(False, "--ledger-full", help="Also emit the full ledger."),
    overrides: list[str] = typer.Argument(
        None,
        help="Hydra-style overrides, e.g. filter.id_ranking.min_score=35",
    ),
) -> None:
    """Stage 9 — rank Progenesis candidates (Score → Frag → ppm) and assign MSI tier."""
    import pandas as pd

    from biodisc.filters._id_ranking_report import build_report
    from biodisc.filters.id_ranking import IdRankingConfig, write_outputs
    from biodisc.filters.id_ranking import run as run_filter

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    console.rule("[bold cyan]Stage 9 — Identification candidate ranking")
    console.print(f"[dim]Stage-1 dir:[/]   {cfg.paths.stage_01_parsed}")
    console.print(f"[dim]Stage-8 dir:[/]   {cfg.paths.stage_08_ccs}")
    console.print(f"[dim]Output dir:[/]    {cfg.paths.stage_09_id_ranking}")
    console.print(
        f"[dim]Quality floor:[/] Score ≥ {cfg.filter.id_ranking.min_score} · "
        f"|ppm| ≤ {cfg.filter.id_ranking.max_mass_error_ppm}"
    )

    stage01 = Path(cfg.paths.stage_01_parsed)
    ident = pd.read_parquet(stage01 / "identification_long.parquet")
    stage8_kept = pd.read_parquet(Path(cfg.paths.stage_08_ccs) / "kept_features.parquet")

    f_cfg = IdRankingConfig(
        min_score=float(cfg.filter.id_ranking.min_score),
        max_mass_error_ppm=float(cfg.filter.id_ranking.max_mass_error_ppm),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit = AuditLogger(
        stage="09_id_ranking",
        params=OmegaConf.to_container(cfg.filter.id_ranking, resolve=True),  # type: ignore[arg-type]
    )
    result = run_filter(
        identification=ident,
        kept_features=stage8_kept["feature_id"].tolist(),
        config=f_cfg,
        audit=audit,
    )
    paths = write_outputs(
        result,
        output_dir=Path(cfg.paths.stage_09_id_ranking),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit.flush(
        output_dir=Path(cfg.paths.audit_dir),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )

    report_dir = Path(cfg.paths.reports_dir) / "09_id_ranking"
    counts = {
        "n_input": result.n_input,
        "n_with_candidates": result.n_with_candidates,
        "n_msi_l2": result.n_msi_l2,
        "n_msi_l3": result.n_msi_l3,
        "n_msi_l4_no_candidate": result.n_msi_l4_no_candidate,
        "n_msi_l4_no_floor_survivor": result.n_msi_l4_no_floor_survivor,
        "n_resolved_single_after_floor": result.n_resolved_single_after_floor,
        "n_resolved_at_score": result.n_resolved_at_score,
        "n_resolved_at_frag": result.n_resolved_at_frag,
        "n_resolved_at_ppm": result.n_resolved_at_ppm,
        "n_still_tied": result.n_still_tied,
    }
    config_summary = {
        "min_score": f_cfg.min_score,
        "max_mass_error_ppm": f_cfg.max_mass_error_ppm,
    }
    report_outputs = build_report(
        metrics=result.metrics,
        identification=ident,
        output_dir=report_dir,
        counts=counts,
        config_summary=config_summary,
        emit_png=True,
    )

    t = Table(title="Stage 9 summary", show_lines=False, title_style="bold green")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Input (Stage-8-active)", f"{result.n_input:,}")
    t.add_row("With Progenesis candidate(s)", f"{result.n_with_candidates:,}")
    t.add_row("MSI L2 (unique)", f"{result.n_msi_l2:,}")
    t.add_row("MSI L3 (compound class)", f"{result.n_msi_l3:,}")
    t.add_row("MSI L4 (no floor survivor)", f"{result.n_msi_l4_no_floor_survivor:,}")
    t.add_row("MSI L4 (no candidate)", f"{result.n_msi_l4_no_candidate:,}")
    t.add_row("  ↳ single after floor", f"{result.n_resolved_single_after_floor:,}")
    t.add_row("  ↳ resolved at Score", f"{result.n_resolved_at_score:,}")
    t.add_row("  ↳ resolved at Frag", f"{result.n_resolved_at_frag:,}")
    t.add_row("  ↳ resolved at ppm", f"{result.n_resolved_at_ppm:,}")
    t.add_row("  ↳ still tied (L3)", f"{result.n_still_tied:,}")
    console.print(t)
    for k, p in paths.items():
        console.print(f"[dim]  {k}:[/] {p}")
    console.print(f"[dim]  report:[/] {report_outputs['index']}")
    _maybe_rebuild_ledger(cfg, full=False, skip=no_ledger)
    if ledger_full and not no_ledger:
        _maybe_rebuild_ledger(cfg, full=True, skip=False)
    console.print("[bold green]✓[/] Stage 9 complete.")


# ============================================================
# STAGE 10 — Origin filter (annotation-driven, drops pure-exogenous)
# ============================================================
@filter_app.command("origin")
def filter_origin(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR, "--config-dir", "-c", help="Hydra config directory.",
    ),
    no_ledger: bool = typer.Option(False, "--no-ledger", help="Skip auto-rebuild of ledger."),
    ledger_full: bool = typer.Option(False, "--ledger-full", help="Also emit the full ledger."),
    overrides: list[str] = typer.Argument(
        None,
        help="Hydra-style overrides, e.g. filter.origin.drop_pure_exogenous=false",
    ),
) -> None:
    """Stage 10 — drop features whose all surviving candidates are pure-exogenous."""
    import pandas as pd

    from biodisc.filters._origin_report import build_report
    from biodisc.filters.origin import OriginFilterConfig, write_outputs
    from biodisc.filters.origin import run as run_filter

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    console.rule("[bold cyan]Stage 10 — Origin filter (annotation-driven)")
    console.print(f"[dim]Stage-9 dir:[/]   {cfg.paths.stage_09_id_ranking}")
    console.print(f"[dim]Output dir:[/]    {cfg.paths.stage_10_origin}")
    console.print(f"[dim]HMDB origin:[/]   {cfg.paths.hmdb_origin_parquet}")
    console.print(
        f"[dim]Drop mode:[/]    "
        f"{'HARD drop' if cfg.filter.origin.drop_pure_exogenous else 'SOFT tag (no drops)'}"
    )

    stage9 = Path(cfg.paths.stage_09_id_ranking)
    metrics_in = pd.read_parquet(stage9 / "metrics.parquet")
    ranked = pd.read_parquet(stage9 / "identification_ranked.parquet")
    stage9_kept = pd.read_parquet(stage9 / "kept_features.parquet")

    origin_path = Path(cfg.paths.hmdb_origin_parquet)
    if not origin_path.exists():
        console.print(
            f"[bold red]✗[/] HMDB origin parquet not found at {origin_path}.\n"
            f"[dim]Build it first:[/] biodisc external build-hmdb-origin"
        )
        raise typer.Exit(code=2)
    hmdb_origin = pd.read_parquet(origin_path)

    f_cfg = OriginFilterConfig(
        drop_pure_exogenous=bool(cfg.filter.origin.drop_pure_exogenous),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit = AuditLogger(
        stage="10_origin",
        params=OmegaConf.to_container(cfg.filter.origin, resolve=True),  # type: ignore[arg-type]
    )
    result = run_filter(
        metrics_in=metrics_in,
        ranked_identification=ranked,
        hmdb_origin=hmdb_origin,
        kept_features=stage9_kept["feature_id"].tolist(),
        config=f_cfg,
        audit=audit,
    )
    paths = write_outputs(
        result,
        output_dir=Path(cfg.paths.stage_10_origin),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )
    audit.flush(
        output_dir=Path(cfg.paths.audit_dir),
        emit_csv_mirror=cfg.parse.emit_csv_mirror,
    )

    report_dir = Path(cfg.paths.reports_dir) / "10_origin"
    counts = {
        "n_input": result.n_input,
        "n_kept": result.n_kept,
        "n_dropped": result.n_dropped,
        "n_endogenous_evidence": result.n_endogenous_evidence,
        "n_exo_or_unknown": result.n_exo_or_unknown,
        "n_passthrough_l4": result.n_passthrough_l4,
        "n_all_exogenous": result.n_all_exogenous,
    }
    config_summary = {"drop_pure_exogenous": f_cfg.drop_pure_exogenous}
    report_outputs = build_report(
        metrics=result.metrics,
        ranked_identification=ranked,
        output_dir=report_dir,
        counts=counts,
        config_summary=config_summary,
        emit_png=True,
    )

    t = Table(title="Stage 10 summary", show_lines=False, title_style="bold green")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Input (Stage-9-active)", f"{result.n_input:,}")
    t.add_row("Kept (TOTAL)", f"{result.n_kept:,}")
    t.add_row("  • endogenous evidence", f"{result.n_endogenous_evidence:,}")
    t.add_row("  • exo-or-unknown (kept)", f"{result.n_exo_or_unknown:,}")
    t.add_row("  • pass-through L4", f"{result.n_passthrough_l4:,}")
    t.add_row("Dropped — all-exogenous", f"{result.n_dropped:,}")
    t.add_row("Drop %", f"{result.drop_pct:.2f} %")
    console.print(t)
    for k, p in paths.items():
        console.print(f"[dim]  {k}:[/] {p}")
    console.print(f"[dim]  report:[/] {report_outputs['index']}")
    _maybe_rebuild_ledger(cfg, full=False, skip=no_ledger)
    if ledger_full and not no_ledger:
        _maybe_rebuild_ledger(cfg, full=True, skip=False)
    console.print("[bold green]✓[/] Stage 10 complete.")


# ============================================================
# External-reference builders — biodisc external <build-step>
# ============================================================
@external_app.command("build-hmdb-inchikey")
def external_build_hmdb_inchikey(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR, "--config-dir", "-c", help="Hydra config directory.",
    ),
    overrides: list[str] = typer.Argument(None, help="Hydra-style path overrides."),
) -> None:
    """Stream-parse HMDB metabolites XML → small parquet of (hmdb_id, inchikey)."""
    from biodisc.external import hmdb_inchikey

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    xml_path = Path(cfg.paths.hmdb_xml)
    out_path = Path(cfg.paths.hmdb_inchikey_parquet)

    console.rule("[bold cyan]External — build HMDB → InChiKey bridge")
    console.print(f"[dim]Input XML:[/]   {xml_path}")
    console.print(f"[dim]Output:[/]      {out_path}")

    if not xml_path.exists():
        console.print(
            f"[bold red]✗[/] HMDB XML not found at {xml_path}.\n"
            f"[dim]Download via:[/] python scripts/download_ccs_references.py --include-hmdb"
        )
        raise typer.Exit(code=2)

    df = hmdb_inchikey.build(xml_path, out_path)
    console.print(f"[bold green]✓[/] Wrote {len(df):,} rows "
                  f"({df['inchikey'].notna().sum():,} with InChiKey).")


@external_app.command("build-hmdb-origin")
def external_build_hmdb_origin(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR, "--config-dir", "-c", help="Hydra config directory.",
    ),
    overrides: list[str] = typer.Argument(None, help="Hydra-style path overrides."),
) -> None:
    """Stream-parse HMDB metabolites XML → parquet of (hmdb_id, source_terms, origin_class)."""
    from biodisc.external import hmdb_origin

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    xml_path = Path(cfg.paths.hmdb_xml)
    out_path = Path(cfg.paths.hmdb_origin_parquet)

    console.rule("[bold cyan]External — build HMDB origin classification")
    console.print(f"[dim]Input XML:[/]   {xml_path}")
    console.print(f"[dim]Output:[/]      {out_path}")

    if not xml_path.exists():
        console.print(
            f"[bold red]✗[/] HMDB XML not found at {xml_path}.\n"
            f"[dim]Download via:[/] python scripts/download_ccs_references.py --include-hmdb"
        )
        raise typer.Exit(code=2)

    df = hmdb_origin.build(xml_path, out_path)
    console.print(
        f"[bold green]✓[/] Wrote {len(df):,} rows · "
        f"endogenous_only={int((df['origin_class'] == 'endogenous_only').sum()):,} · "
        f"both={int((df['origin_class'] == 'both').sum()):,} · "
        f"exogenous_only={int((df['origin_class'] == 'exogenous_only').sum()):,} · "
        f"unannotated={int((df['origin_class'] == 'unannotated').sum()):,}."
    )


@external_app.command("build-run-order")
def external_build_run_order(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR, "--config-dir", "-c", help="Hydra config directory.",
    ),
    no_validate: bool = typer.Option(
        False,
        "--no-validate",
        help="Skip cross-check against samples.parquet (default: validates).",
    ),
    overrides: list[str] = typer.Argument(None, help="Hydra-style path overrides."),
) -> None:
    """Parse Waters _HEADER.TXT files → chronological run-order parquet for drift correction."""
    from biodisc.external.run_order import RunOrderConfig, build

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    headers_dir = Path(cfg.paths.run_order_headers_dir)
    out_path = Path(cfg.paths.run_order_parquet)
    samples_path = Path(cfg.paths.stage_01_parsed) / "samples.parquet"

    console.rule("[bold cyan]External — build chronological run-order")
    console.print(f"[dim]Headers dir:[/]      {headers_dir}")
    console.print(f"[dim]Samples file:[/]     {samples_path}")
    console.print(f"[dim]Output:[/]           {out_path}")

    if not headers_dir.exists():
        console.print(
            f"[bold red]✗[/] Headers directory not found at {headers_dir}.\n"
            f"[dim]Place Waters _HEADER.TXT files in this directory and re-run."
        )
        raise typer.Exit(code=2)

    rcfg = RunOrderConfig(
        headers_dir=headers_dir,
        output_parquet=out_path,
        samples_parquet=None if no_validate else samples_path,
        drop_unknown_samples=True,
        fail_on_missing_samples=True,
    )
    df = build(rcfg)
    duration_min = (
        (df["acq_datetime"].max() - df["acq_datetime"].min()).total_seconds() / 60
    )
    console.print(
        f"[bold green]✓[/] Wrote {len(df):,} injections · "
        f"chronological window: "
        f"{df['acq_datetime'].min().strftime('%Y-%m-%d %H:%M')} → "
        f"{df['acq_datetime'].max().strftime('%Y-%m-%d %H:%M')} "
        f"({duration_min:.0f} min total)."
    )


@external_app.command("build-ccs-reference")
def external_build_ccs_reference(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR, "--config-dir", "-c", help="Hydra config directory.",
    ),
    overrides: list[str] = typer.Argument(None, help="Hydra-style path overrides."),
) -> None:
    """Build the unified CCS reference parquet from McLean Compendium + HMDB bridge."""
    from biodisc.external.ccs_reference import CCSReferenceConfig, build

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    rcfg = CCSReferenceConfig(
        mclean_csv=Path(cfg.paths.mclean_csv),
        hmdb_inchikey_parquet=Path(cfg.paths.hmdb_inchikey_parquet),
        output_parquet=Path(cfg.paths.ccs_reference_parquet),
    )

    console.rule("[bold cyan]External — build unified CCS reference")
    console.print(f"[dim]McLean CSV:[/]      {rcfg.mclean_csv}")
    console.print(f"[dim]HMDB bridge:[/]     {rcfg.hmdb_inchikey_parquet}")
    console.print(f"[dim]Output:[/]          {rcfg.output_parquet}")

    if not rcfg.mclean_csv.exists():
        console.print(
            f"[bold red]✗[/] McLean CSV not found at {rcfg.mclean_csv}.\n"
            f"[dim]Download via:[/] python scripts/download_ccs_references.py"
        )
        raise typer.Exit(code=2)

    df = build(rcfg)
    console.print(
        f"[bold green]✓[/] Wrote {len(df):,} unified rows · "
        f"{df['hmdb_id'].nunique():,} unique HMDB compounds · "
        f"{df['adduct'].nunique():,} unique adducts."
    )


@external_app.command("build-hmdb-kegg")
def external_build_hmdb_kegg(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR, "--config-dir", "-c", help="Hydra config directory.",
    ),
    overrides: list[str] = typer.Argument(None, help="Hydra-style path overrides."),
) -> None:
    """Stream-parse HMDB metabolites XML → parquet of (hmdb_id, kegg_id)."""
    from biodisc.external import hmdb_kegg

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    xml_path = Path(cfg.paths.hmdb_xml)
    out_path = Path(cfg.paths.hmdb_kegg_parquet)

    console.rule("[bold cyan]External — build HMDB → KEGG Compound ID bridge")
    console.print(f"[dim]Input XML:[/]   {xml_path}")
    console.print(f"[dim]Output:[/]      {out_path}")

    if not xml_path.exists():
        console.print(
            f"[bold red]✗[/] HMDB XML not found at {xml_path}.\n"
            f"[dim]Download via:[/] python scripts/download_ccs_references.py --include-hmdb"
        )
        raise typer.Exit(code=2)

    df = hmdb_kegg.build(xml_path, out_path)
    console.print(f"[bold green]✓[/] Wrote {len(df):,} rows "
                  f"({df['kegg_id'].notna().sum():,} with KEGG ID).")


# ============================================================
# STAGES 8b, 9b, 11b, 12b — method benchmarks (placeholders)
# ============================================================
@bench_app.command("imputation")
def bench_imputation() -> None:
    """Stage 8b — imputation method benchmark (pending)."""
    console.print("[yellow]Stage 8b (imputation benchmark) not yet implemented.[/]")
    raise typer.Exit(code=1)


@bench_app.command("drift")
def bench_drift() -> None:
    """Stage 9b — drift correction method benchmark (pending)."""
    console.print("[yellow]Stage 9b (drift benchmark) not yet implemented.[/]")
    raise typer.Exit(code=1)


@bench_app.command("normalisation")
def bench_normalisation() -> None:
    """Stage 11b — normalisation method benchmark (pending)."""
    console.print("[yellow]Stage 11b (normalisation benchmark) not yet implemented.[/]")
    raise typer.Exit(code=1)


@bench_app.command("transformation")
def bench_transformation() -> None:
    """Stage 12b — transformation method benchmark (pending)."""
    console.print("[yellow]Stage 12b (transformation benchmark) not yet implemented.[/]")
    raise typer.Exit(code=1)


# ============================================================
# Orchestration + reporting (placeholders)
# ============================================================
@run_app.command("all")
def run_all() -> None:
    """Run the full 17-stage pipeline end-to-end (pending)."""
    console.print("[yellow]Full orchestrator not yet implemented. Use `dvc repro` for now.[/]")
    raise typer.Exit(code=1)


# ============================================================
# PHASE 2 — Multiverse discovery (Stages 11–17)
# ============================================================
@multiverse_app.command("run")
def multiverse_run(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR, "--config-dir", "-c", help="Hydra config directory.",
    ),
    overrides: list[str] = typer.Argument(
        None,
        help=(
            "Hydra-style overrides, e.g. multiverse.n_permutation=200 "
            "'multiverse.outcome_filter=[endogenous_evidence]'"
        ),
    ),
) -> None:
    """Run the full multiverse: load Stage-10 features → paths → consensus → index.

    Slow (hours) and launched explicitly — it is NOT part of `dvc repro`.
    All settings live in conf/multiverse/default.yaml and are overridable.
    """
    from biodisc.multiverse import production

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    mv = cfg.multiverse
    paths_cfg = OmegaConf.to_container(mv.paths, resolve=True)
    specs = production.build_paths_from_config(paths_cfg)  # type: ignore[arg-type]
    outcome_filter = (
        None
        if mv.outcome_filter is None
        else list(OmegaConf.to_container(mv.outcome_filter, resolve=True))  # type: ignore[arg-type]
    )
    n_threads = None if mv.get("n_threads", None) is None else int(mv.n_threads)
    output_dir = Path(mv.output_dir)

    console.rule("[bold cyan]Phase 2 — Multiverse discovery")
    console.print(f"[dim]Interim dir:[/]    {cfg.paths.interim_dir}")
    console.print(f"[dim]Output dir:[/]     {output_dir}")
    console.print(
        f"[dim]Bootstrap:[/]      {mv.n_bootstrap}    "
        f"[dim]Permutation:[/] {mv.n_permutation}    "
        f"[dim]Threshold:[/] {mv.threshold}"
    )
    console.print(
        f"[dim]Outcome filter:[/] "
        f"{outcome_filter if outcome_filter else 'all Stage-10 survivors'}"
    )
    console.print(f"[dim]Thread cap:[/]     {n_threads if n_threads else 'none'}")
    console.print(f"[dim]Paths:[/]          {', '.join(s.name for s in specs)}")
    console.print(
        "[yellow]This is the slow phase (hours). "
        "Run on a machine with cores to spare.[/]"
    )

    results, consensus = production.run(
        paths=specs,
        output_dir=output_dir,
        interim_dir=Path(cfg.paths.interim_dir),
        run_order_path=Path(cfg.paths.run_order_parquet),
        n_bootstrap=int(mv.n_bootstrap),
        n_permutation=int(mv.n_permutation),
        threshold=float(mv.threshold),
        outcome_filter=outcome_filter,
        seed=int(cfg.seed),
        n_threads=n_threads,
    )

    n_ok = sum(r.succeeded for r in results)
    t = Table(title="Multiverse summary", show_lines=False, title_style="bold green")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Paths succeeded", f"{n_ok}/{len(results)}")
    t.add_row("Consensus features", f"{consensus.n_consensus:,}")
    for tier, n in consensus.n_per_tier.items():
        t.add_row(f"  {tier}", f"{n:,}")
    console.print(t)
    console.print(f"[dim]  index:[/]     {output_dir / 'index.html'}")
    console.print(f"[dim]  consensus:[/] {output_dir / 'consensus_passed.parquet'}")
    console.print("[bold green]✓[/] Multiverse complete.")


@multiverse_app.command("enrichment")
def multiverse_enrichment(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR, "--config-dir", "-c", help="Hydra config directory.",
    ),
    q_threshold: float = typer.Option(
        0.05, "--q-threshold", help="BH-FDR q-value threshold for enrichment.",
    ),
    overrides: list[str] = typer.Argument(None, help="Hydra-style path overrides."),
) -> None:
    """KEGG pathway enrichment for the multiverse consensus shortlist.

    Maps consensus features to KEGG compounds via HMDB, queries KEGG REST
    API for pathway membership, and computes hypergeometric enrichment
    with BH-FDR correction. Writes results to the multiverse output dir.
    """
    from biodisc.external import kegg_enrichment as ke

    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    consensus_dir = Path(cfg.paths.reports_dir) / "multiverse"
    interim_dir = Path(cfg.paths.interim_dir)
    hmdb_kegg_path = Path(cfg.paths.hmdb_kegg_parquet)
    output_dir = consensus_dir

    console.rule("[bold cyan]KEGG Pathway Enrichment")

    if not consensus_dir.exists():
        console.print("[bold red]✗[/] Multiverse output not found. Run biodisc multiverse run first.")
        raise typer.Exit(code=1)

    import bioservices
    k = bioservices.KEGG(verbose=False)

    result = ke.run_enrichment(
        consensus_dir=consensus_dir,
        interim_dir=interim_dir,
        hmdb_kegg_path=hmdb_kegg_path,
        kegg_client=k,
        q_threshold=q_threshold,
    )

    console.print(f"[dim]Consensus features:[/]  {result.n_foreground_features}")
    console.print(f"[dim]Mapped to KEGG:[/]     {result.n_foreground_mapped_to_kegg}")
    console.print(f"[dim]Pathways tested:[/]    {result.n_pathways_tested}")
    console.print(f"[dim]Enriched (q ≤ {q_threshold}):[/]   {result.n_pathways_enriched_at_q05}")

    if result.table.empty:
        console.print("[yellow]No enrichment results. Check HMDB→KEGG bridge exists and multiverse was run.[/]")
        return

    # Show top enriched pathways
    top = result.table[result.table["enriched"]].head(10) if result.n_pathways_enriched_at_q05 > 0 else result.table.head(10)
    from rich.table import Table
    tbl = Table(title=f"Top {'enriched' if result.n_pathways_enriched_at_q05 > 0 else ''} pathways", show_lines=False)
    tbl.add_column("Pathway", style="cyan")
    tbl.add_column("Name")
    tbl.add_column("Size", justify="right")
    tbl.add_column("Hits", justify="right")
    tbl.add_column("p-value", justify="right")
    tbl.add_column("q-value", justify="right")
    for _, r in top.iterrows():
        tbl.add_row(
            r["pathway_id"],
            str(r.get("pathway_name", ""))[:60],
            str(r["pathway_size"]),
            str(r["n_foreground_in_pathway"]),
            f"{r['p_value']:.4g}",
            f"{r['q_value']:.4g}",
        )
    console.print(tbl)

    paths = ke.write_enrichment(result, output_dir)
    console.print(f"[bold green]✓[/] Enrichment written — {result.n_pathways_tested} pathways tested, "
                  f"{result.n_pathways_enriched_at_q05} enriched.")
    for k, v in paths.items():
        console.print(f"[dim]  {k}:[/] {v}")


@app.command()
def report() -> None:
    """Build the HTML + PDF report (pending)."""
    console.print("[yellow]Report builder not yet implemented.[/]")
    raise typer.Exit(code=1)


@audit_app.command("ledger")
def audit_ledger(
    config_dir: Path = typer.Option(
        _DEFAULT_CONFIG_DIR,
        "--config-dir",
        "-c",
        help="Hydra config directory.",
    ),
    full: bool = typer.Option(
        False, "--full", help="Emit the wide ledger (every metric column from every stage)."
    ),
    overrides: list[str] = typer.Argument(
        None,
        help="Hydra-style overrides for path config.",
    ),
) -> None:
    """Rebuild the master feature ledger from current per-stage outputs."""
    cfg = _load_config(config_dir, overrides or [])
    _bootstrap(cfg)

    console.rule("[bold cyan]Master feature ledger — rebuild")
    console.print(f"[dim]Interim dir:[/] {cfg.paths.interim_dir}")
    console.print(f"[dim]Audit dir:[/]   {cfg.paths.audit_dir}")
    console.print(f"[dim]Mode:[/]        {'full' if full else 'lean'}")

    outs = rebuild_ledger(
        interim_dir=Path(cfg.paths.interim_dir),
        audit_dir=Path(cfg.paths.audit_dir),
        full=full,
        emit_csv=cfg.parse.emit_csv_mirror,
    )
    for k, p in outs.items():
        console.print(f"[dim]  {k}:[/] {p}")
    console.print("[bold green]✓[/] Ledger rebuilt.")


@app.command()
def version() -> None:
    """Print the biodisc package version."""
    from biodisc import __version__

    console.print(f"biodisc {__version__}")


if __name__ == "__main__":
    app()
