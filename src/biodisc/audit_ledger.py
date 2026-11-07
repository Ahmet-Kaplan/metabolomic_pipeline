"""Master feature ledger — one row per feature, columns growing per stage.

A cross-stage view that joins every per-stage ``metrics.parquet`` into a single
wide table. Each filter stage contributes:

  * ``s<N>_<metric>`` — the headline metric value (lean) or every metric column
    from that stage (full).
  * ``s<N>_kept`` — boolean, whether the feature survived this stage.

Plus two derived columns:

  * ``dropped_at_stage`` — the stage_id where the feature was first dropped,
    or ``None`` if still active.
  * ``active`` — True iff the feature passed every stage it was evaluated in.

The ledger is **derived** from the source-of-truth per-stage outputs; no stage
needs to write to it directly. Rebuild is idempotent and cheap (~1 s per
30k features × N stages).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger(__name__)


# ============================================================
# Stage registry — declares which stages exist and their headline metric
# ============================================================
@dataclass(frozen=True)
class StageSpec:
    """Where to find a stage's metrics + which column is the headline."""

    stage_id: str  # e.g. "03_presence" — must match the dir name
    metrics_filename: str  # usually "metrics.parquet"
    headline_metric: str  # e.g. "max_presence_pct"
    kept_col: str = "kept"

    def metrics_path(self, interim_dir: Path) -> Path:
        return Path(interim_dir) / self.stage_id / self.metrics_filename


# Known filter stages.  When a new filter stage lands, append here.
# Order in this list defines order in the ledger.
STAGE_REGISTRY: list[StageSpec] = [
    StageSpec("03_presence", "metrics.parquet", "max_presence_pct"),
    StageSpec("04_qc_cv", "metrics.parquet", "qc_cv_pct"),
    StageSpec("05_sample_qc_ratio", "metrics.parquet", "ratio"),
    StageSpec("06_blank", "metrics.parquet", "blank_pct"),
    StageSpec("07_dilution", "metrics.parquet", "r2_linear"),
    StageSpec("08_ccs", "metrics.parquet", "ccs_err_pct"),
    StageSpec("09_id_ranking", "metrics.parquet", "n_candidates_final"),
    StageSpec("10_origin", "metrics.parquet", "n_exogenous_only"),
]


def _short(stage_id: str) -> str:
    """Return the column prefix for a stage, e.g. '03_presence' -> 's3'."""
    head = stage_id.split("_", 1)[0]
    try:
        return f"s{int(head)}"
    except ValueError:
        return stage_id


def discover_available_stages(
    interim_dir: Path,
    registry: list[StageSpec] | None = None,
) -> list[StageSpec]:
    """Return the subset of registered stages whose metrics file exists on disk."""
    reg = registry or STAGE_REGISTRY
    return [s for s in reg if s.metrics_path(interim_dir).exists()]


# ============================================================
# Build the ledger
# ============================================================
def build_ledger(
    interim_dir: Path,
    *,
    full: bool = False,
    registry: list[StageSpec] | None = None,
) -> pd.DataFrame:
    """Construct the master ledger by joining all available per-stage metrics.

    Parameters
    ----------
    interim_dir
        Path containing per-stage subdirs (``data/interim/``).
    full
        If False (default), include only ``feature_id``, the headline metric,
        and the kept flag from each stage.
        If True, include every column from each stage's metrics.parquet,
        prefixed with the stage's short id (e.g. ``s3_MCF7_pct``).
    registry
        Override the default STAGE_REGISTRY for tests.
    """
    available = discover_available_stages(interim_dir, registry)
    if not available:
        log.warning("ledger_no_stages_found", interim_dir=str(interim_dir))
        return pd.DataFrame(columns=["feature_id", "dropped_at_stage", "active"])

    log.info(
        "ledger_build_start",
        n_stages=len(available),
        stages=[s.stage_id for s in available],
        mode="full" if full else "lean",
    )

    # Start from the broadest superset of feature_ids — usually Stage 3 input.
    # Each stage's metrics.parquet contains exactly the features that ENTERED
    # that stage. By taking the first available stage's full set, we capture
    # the largest universe; later stages naturally have a subset.
    ledger: pd.DataFrame | None = None
    for spec in available:
        df = pd.read_parquet(spec.metrics_path(interim_dir))
        if "feature_id" not in df.columns:
            log.warning("ledger_stage_missing_feature_id", stage=spec.stage_id)
            continue

        prefix = _short(spec.stage_id)

        if full:
            # Take every column except feature_id, prefix it
            keep_cols = [c for c in df.columns if c != "feature_id"]
            renamed = df[["feature_id", *keep_cols]].rename(
                columns={c: f"{prefix}_{c}" for c in keep_cols}
            )
        else:
            # Lean: just the headline metric + kept
            cols_we_want = [spec.headline_metric, spec.kept_col]
            cols_present = [c for c in cols_we_want if c in df.columns]
            if len(cols_present) < 2:
                log.warning(
                    "ledger_stage_missing_columns",
                    stage=spec.stage_id,
                    missing=set(cols_we_want) - set(df.columns),
                )
                continue
            renamed = df[["feature_id", *cols_present]].rename(
                columns={
                    spec.headline_metric: f"{prefix}_{spec.headline_metric}",
                    spec.kept_col: f"{prefix}_{spec.kept_col}",
                }
            )

        ledger = renamed if ledger is None else ledger.merge(renamed, on="feature_id", how="outer")

    assert ledger is not None  # available was non-empty
    ledger = _add_derived_columns(ledger, available)
    log.info("ledger_build_complete", rows=len(ledger), cols=len(ledger.columns))
    return ledger


def _add_derived_columns(ledger: pd.DataFrame, stages: list[StageSpec]) -> pd.DataFrame:
    """Add ``dropped_at_stage`` and ``active`` from per-stage kept columns."""
    kept_cols = [(s, f"{_short(s.stage_id)}_{s.kept_col}") for s in stages]

    def first_drop(row: pd.Series) -> str | None:
        for spec, col in kept_cols:
            v = row.get(col)
            # NaN means the feature was never evaluated at this stage
            # (because it was dropped earlier). Skip — its drop is already
            # captured by an earlier stage.
            if pd.isna(v):
                continue
            if not bool(v):
                return spec.stage_id
        return None

    ledger["dropped_at_stage"] = ledger.apply(first_drop, axis=1)
    ledger["active"] = ledger["dropped_at_stage"].isna()
    return ledger


# ============================================================
# Persistence
# ============================================================
def write_ledger(
    ledger: pd.DataFrame,
    audit_dir: Path,
    *,
    suffix: str = "",
    emit_csv: bool = True,
) -> dict[str, Path]:
    """Write the ledger to ``<audit_dir>/master_feature_ledger[<suffix>].{parquet,csv}``."""
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    base = f"master_feature_ledger{suffix}"
    out: dict[str, Path] = {}

    parquet_path = audit_dir / f"{base}.parquet"
    ledger.to_parquet(parquet_path, engine="pyarrow", compression="zstd")
    out["parquet"] = parquet_path
    log.info("ledger_written_parquet", path=str(parquet_path), rows=len(ledger))

    if emit_csv:
        csv_path = audit_dir / f"{base}.csv"
        ledger.to_csv(csv_path, index=False)
        out["csv"] = csv_path

    return out


def rebuild(
    interim_dir: Path,
    audit_dir: Path,
    *,
    full: bool = False,
    emit_csv: bool = True,
) -> dict[str, Path]:
    """Convenience: build + write in one call. Use from filter CLIs."""
    ledger = build_ledger(interim_dir, full=full)
    suffix = "_full" if full else ""
    return write_ledger(ledger, audit_dir, suffix=suffix, emit_csv=emit_csv)
