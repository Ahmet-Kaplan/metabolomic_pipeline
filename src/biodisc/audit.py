"""Per-feature audit log.

Every pipeline stage records, for each feature it touched, one row of:

    (feature_id, stage, action, reason, metric, threshold, timestamp, params_hash)

Stages use a single :class:`AuditLogger` instance. Rows are accumulated in
memory and flushed to parquet + (optional) CSV mirror at the end of the stage.
A master audit (all stages concatenated) can be reconstructed downstream by
scanning ``data/audit/``.

Example
-------
>>> audit = AuditLogger(stage="tidy", params={"emit_csv_mirror": True})
>>> audit.kept(feature_ids=["11.04_665.3056n", "13.22_314.2477n"], reason="parsed_ok")
>>> audit.dropped(feature_ids=["X_YY_bad"], reason="missing_rt", metric=None)
>>> audit.flush(output_dir=Path("data/audit"))
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from biodisc.io.schemas import AuditSchema

log = structlog.get_logger(__name__)


# ============================================================
# Helpers
# ============================================================
def _hash_params(params: dict[str, Any]) -> str:
    """Short deterministic hash of a stage's params dict.

    Used so a row's lineage can be traced back to the exact configuration that
    produced it. Implemented as the first 12 hex chars of SHA-256 of the JSON-
    serialised (sort_keys=True) params.
    """
    blob = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


# ============================================================
# AuditLogger
# ============================================================
@dataclass
class AuditLogger:
    """Accumulate per-feature audit rows for a single stage, then flush."""

    stage: str
    params: dict[str, Any] = field(default_factory=dict)
    _rows: list[dict[str, Any]] = field(default_factory=list, init=False)
    _params_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self._params_hash = _hash_params(self.params)
        log.debug(
            "audit_logger_initialised",
            stage=self.stage,
            params_hash=self._params_hash,
            params_keys=list(self.params.keys()),
        )

    # --- one-shot record methods ---------------------------------
    def kept(
        self,
        feature_ids: Iterable[str],
        reason: str,
        metric: float | None = None,
        threshold: float | None = None,
    ) -> None:
        self._append(feature_ids, "kept", reason, metric, threshold)

    def dropped(
        self,
        feature_ids: Iterable[str],
        reason: str,
        metric: float | None = None,
        threshold: float | None = None,
    ) -> None:
        self._append(feature_ids, "dropped", reason, metric, threshold)

    def modified(
        self,
        feature_ids: Iterable[str],
        reason: str,
        metric: float | None = None,
        threshold: float | None = None,
    ) -> None:
        self._append(feature_ids, "modified", reason, metric, threshold)

    def created(
        self,
        feature_ids: Iterable[str],
        reason: str,
        metric: float | None = None,
        threshold: float | None = None,
    ) -> None:
        self._append(feature_ids, "created", reason, metric, threshold)

    # --- bulk with per-feature metrics ---------------------------
    def record_bulk(
        self,
        feature_ids: Iterable[str],
        actions: Iterable[str],
        reasons: Iterable[str],
        metrics: Iterable[float | None] | None = None,
        thresholds: Iterable[float | None] | None = None,
    ) -> None:
        """Record many rows at once where each feature has its own action/metric.

        Preferred for stages where metrics (e.g. CV%, blank ratio) are per-feature
        and should be preserved verbatim for later analysis.
        """
        fids = list(feature_ids)
        acts = list(actions)
        reas = list(reasons)
        if not (len(fids) == len(acts) == len(reas)):
            raise ValueError(
                f"record_bulk lengths mismatch: {len(fids)=}, {len(acts)=}, {len(reas)=}"
            )
        met = list(metrics) if metrics is not None else [None] * len(fids)
        thr = list(thresholds) if thresholds is not None else [None] * len(fids)
        if len(met) != len(fids) or len(thr) != len(fids):
            raise ValueError("record_bulk metric/threshold length mismatch")

        now = datetime.now(UTC)
        for fid, a, r, m, t in zip(fids, acts, reas, met, thr, strict=True):
            self._rows.append(
                {
                    "feature_id": fid,
                    "stage": self.stage,
                    "action": a,
                    "reason": r,
                    "metric": m,
                    "threshold": t,
                    "timestamp": now,
                    "stage_params_hash": self._params_hash,
                }
            )

    # --- flush ----------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """Return rows as a validated DataFrame (pandera-checked)."""
        if not self._rows:
            df = pd.DataFrame(
                columns=[
                    "feature_id",
                    "stage",
                    "action",
                    "reason",
                    "metric",
                    "threshold",
                    "timestamp",
                    "stage_params_hash",
                ]
            )
        else:
            df = pd.DataFrame(self._rows)
        # Validate against schema (coerces + checks ranges + types)
        return AuditSchema.validate(df)  # type: ignore[no-any-return]

    def flush(
        self,
        output_dir: Path,
        *,
        emit_csv_mirror: bool = True,
    ) -> Path:
        """Write the accumulated rows to ``<output_dir>/<stage>.parquet`` (+ optional CSV)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        df = self.to_frame()
        parquet_path = output_dir / f"{self.stage}.parquet"
        df.to_parquet(parquet_path, engine="pyarrow", compression="zstd")
        log.info(
            "audit_flushed_parquet",
            stage=self.stage,
            path=str(parquet_path),
            rows=len(df),
            kept=int((df["action"] == "kept").sum()),
            dropped=int((df["action"] == "dropped").sum()),
            modified=int((df["action"] == "modified").sum()),
            created=int((df["action"] == "created").sum()),
        )
        if emit_csv_mirror:
            csv_path = output_dir / f"{self.stage}.csv"
            df.to_csv(csv_path, index=False)
            log.debug("audit_flushed_csv", stage=self.stage, path=str(csv_path))
        return parquet_path

    # --- internals ------------------------------------------------
    def _append(
        self,
        feature_ids: Iterable[str],
        action: str,
        reason: str,
        metric: float | None,
        threshold: float | None,
    ) -> None:
        now = datetime.now(UTC)
        for fid in feature_ids:
            self._rows.append(
                {
                    "feature_id": fid,
                    "stage": self.stage,
                    "action": action,
                    "reason": reason,
                    "metric": metric,
                    "threshold": threshold,
                    "timestamp": now,
                    "stage_params_hash": self._params_hash,
                }
            )
