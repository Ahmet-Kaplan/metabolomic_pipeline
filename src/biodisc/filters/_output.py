"""Shared filter-output persistence.

Every filter stage writes the same two files (kept_features + metrics)
as parquet with an optional CSV mirror. Stage 9 additionally writes
identification_ranked. Each filter's dataclass uses the same attribute
names, so a single generic writer works for all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import structlog

log = structlog.get_logger(__name__)


class FilterResult(Protocol):
    """Minimal interface required by :func:`write_outputs`.

    Every filter result dataclass provides these attributes.
    """

    kept_features: Any  # pd.DataFrame
    metrics: Any  # pd.DataFrame


def write_outputs(
    result: FilterResult,
    output_dir: Path,
    *,
    emit_csv_mirror: bool = True,
) -> dict[str, Path]:
    """Persist kept_features + metrics as parquet (+ optional CSV mirror).

    Parameters
    ----------
    result
        Filter result with ``.kept_features`` and ``.metrics`` DataFrames.
        May also have ``.ranked_identification`` (Stage 9).
    output_dir
        Directory to write into (created if needed).
    emit_csv_mirror
        If True, also write ``.csv`` mirrors of each parquet file.

    Returns
    -------
    dict[str, Path]
        Map of logical name → path for each written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    kept_path = output_dir / "kept_features.parquet"
    result.kept_features.to_parquet(kept_path, engine="pyarrow", compression="zstd")
    paths["kept_features"] = kept_path
    log.info("wrote_kept_features", path=str(kept_path), rows=len(result.kept_features))

    metrics_path = output_dir / "metrics.parquet"
    result.metrics.to_parquet(metrics_path, engine="pyarrow", compression="zstd")
    paths["metrics"] = metrics_path
    log.info("wrote_metrics", path=str(metrics_path), rows=len(result.metrics))

    ranked = getattr(result, "ranked_identification", None)
    if ranked is not None:
        ranked_path = output_dir / "identification_ranked.parquet"
        ranked.to_parquet(ranked_path, engine="pyarrow", compression="zstd")
        paths["identification_ranked"] = ranked_path
        log.info(
            "wrote_identification_ranked",
            path=str(ranked_path),
            rows=len(ranked),
        )

    if emit_csv_mirror:
        result.kept_features.to_csv(output_dir / "kept_features.csv", index=False)
        result.metrics.to_csv(output_dir / "metrics.csv", index=False)
        if ranked is not None:
            ranked.to_csv(output_dir / "identification_ranked.csv", index=False)

    return paths
