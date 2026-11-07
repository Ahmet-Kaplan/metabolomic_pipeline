"""Stage 4 — QC-CV filter.

For each feature, compute the coefficient of variation (CV %) across the 16
QC pool injections. Drop features that exceed the threshold — their signal is
too technically unreliable to trust for biological comparisons.

Default threshold: 30 % (mQACC convention; Broadhurst et al. 2018,
*Metabolomics* 14:72). Configurable via Hydra. For the pre-drift pass the
"loose" variant of 50 % is recommended in the mQACC two-pass protocol; the
strict post-drift pass at Stage 10 then applies CV ≤ 30 %.

A feature is dropped if EITHER:
    - it has < ``min_qc_detections`` non-zero, non-NaN values in QCs (can't
      estimate CV meaningfully), OR
    - its QC-CV % exceeds ``max_cv_pct``.

Consumes kept_features from Stage 3 so the filter only runs on features that
survived presence filtering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from biodisc.audit import AuditLogger
from biodisc.filters._output import write_outputs  # noqa: F401 — re-exported for CLI

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class QcCvFilterConfig:
    """Runtime config for Stage 4. Mirrors conf/filter/default.yaml:qc_cv."""

    max_cv_pct: float = 30.0
    min_qc_detections: int = 3  # need ≥ 3 non-zero QC values to compute CV
    emit_csv_mirror: bool = True


@dataclass(frozen=True)
class QcCvFilterResult:
    """Structured result of running the QC-CV filter."""

    kept_features: pd.DataFrame
    metrics: pd.DataFrame
    n_input: int
    n_kept: int
    n_dropped_cv: int
    n_dropped_detections: int

    @property
    def n_dropped(self) -> int:
        return self.n_dropped_cv + self.n_dropped_detections

    @property
    def drop_pct(self) -> float:
        return self.n_dropped / self.n_input * 100 if self.n_input else 0.0


def _compute_cv_and_stats(vals: np.ndarray) -> tuple[float, float, float, int]:
    """Compute (mean, sd, cv_pct, n_detected) from a vector of QC intensities.

    "Detected" = non-NaN AND > 0 (Stage 2 showed Progenesis encodes below-LOD
    as 0). SD uses Bessel's correction (ddof=1). Returns NaN for mean/sd/cv
    when fewer than 2 detections (SD undefined).
    """
    arr = np.asarray(vals, dtype=float)
    detected = arr[~np.isnan(arr) & (arr > 0)]
    n = len(detected)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"), n)
    mean = float(detected.mean())
    sd = float(detected.std(ddof=1))
    cv_pct = 100.0 * sd / mean if mean > 0 else float("nan")
    return (mean, sd, cv_pct, n)


def compute_qc_cv_metrics(
    abundances_long: pd.DataFrame,
    samples: pd.DataFrame,
    kept_features: set[str] | None = None,
) -> pd.DataFrame:
    """Per feature, compute (qc_mean, qc_sd, qc_cv_pct, n_qc_detected).

    Only rows where ``samples.injection_type == 'qc'`` are used. If
    ``kept_features`` is provided the computation is restricted to those
    feature ids (so Stage 4 operates on Stage-3-kept features).
    """
    qc_ids = samples[samples["injection_type"] == "qc"]["sample_id"]
    if qc_ids.empty:
        raise ValueError("no QC samples in samples metadata — cannot compute QC-CV")

    qc = abundances_long[abundances_long["sample_id"].isin(qc_ids)].copy()
    if kept_features is not None:
        qc = qc[qc["feature_id"].isin(kept_features)]
    qc["intensity"] = pd.to_numeric(qc["intensity"], errors="coerce")

    # Wide pivot: one row per feature, 16 QC cols
    wide = qc.pivot(index="feature_id", columns="sample_id", values="intensity")
    qc_cols = [c for c in qc_ids if c in wide.columns]
    wide = wide[qc_cols]

    # Vectorised stats (faster than apply for 28k features)
    arr = wide.to_numpy(dtype=float)
    detected_mask = (~np.isnan(arr)) & (arr > 0)
    n_detected = detected_mask.sum(axis=1)

    # We need per-row mean/std over detected values only — use masked arrays
    masked = np.where(detected_mask, arr, np.nan)
    mean = np.nanmean(masked, axis=1)
    # ddof=1 for sample std; nanstd with ddof=1 handles this
    sd = np.nanstd(masked, axis=1, ddof=1)
    cv_pct = np.where(mean > 0, 100.0 * sd / mean, np.nan)

    metrics = pd.DataFrame(
        {
            "feature_id": wide.index.to_list(),
            "qc_n_detected": n_detected.astype(int),
            "qc_mean": mean,
            "qc_sd": sd,
            "qc_cv_pct": cv_pct,
        }
    )
    return metrics


def run(
    abundances_long: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    kept_features: set[str] | pd.DataFrame,
    config: QcCvFilterConfig,
    audit: AuditLogger,
) -> QcCvFilterResult:
    """Apply the QC-CV filter to Stage-3-kept features."""
    if isinstance(kept_features, pd.DataFrame):
        kept_features = set(kept_features["feature_id"].tolist())

    metrics = compute_qc_cv_metrics(abundances_long, samples, kept_features=kept_features)

    # Decisions
    cv_pass = metrics["qc_cv_pct"] <= config.max_cv_pct
    det_pass = metrics["qc_n_detected"] >= config.min_qc_detections
    kept_mask = cv_pass.fillna(False) & det_pass

    metrics["kept"] = kept_mask
    # Encode reason categorically
    metrics["drop_reason"] = np.where(
        kept_mask,
        "kept",
        np.where(
            ~det_pass,
            "insufficient_qc_detections",
            "cv_exceeds_threshold",
        ),
    )

    n_input = len(metrics)
    n_kept = int(kept_mask.sum())
    n_dropped_det = int((~det_pass).sum())
    # For features that fail both, we attribute to detections first (stricter reason)
    n_dropped_cv = int((det_pass & ~cv_pass.fillna(False)).sum())
    assert n_kept + n_dropped_det + n_dropped_cv == n_input, "fate partition broken"

    log.info(
        "qc_cv_filter_result",
        n_input=n_input,
        n_kept=n_kept,
        n_dropped_cv=n_dropped_cv,
        n_dropped_detections=n_dropped_det,
        drop_pct=round((n_dropped_cv + n_dropped_det) / n_input * 100, 2) if n_input else 0.0,
        max_cv_pct=config.max_cv_pct,
        min_qc_detections=config.min_qc_detections,
    )

    # Audit: one row per feature
    reasons = metrics["drop_reason"].tolist()
    actions = ["kept" if k else "dropped" for k in kept_mask]
    human_reasons = []
    for k, reason, cv in zip(
        kept_mask.to_numpy(), reasons, metrics["qc_cv_pct"].to_numpy(), strict=True
    ):
        if k:
            human_reasons.append(f"cv={cv:.2f}_le_{config.max_cv_pct:.1f}")
        elif reason == "insufficient_qc_detections":
            human_reasons.append("insufficient_qc_detections")
        else:
            human_reasons.append(f"cv={cv:.2f}_gt_{config.max_cv_pct:.1f}")
    audit.record_bulk(
        feature_ids=metrics["feature_id"].tolist(),
        actions=actions,
        reasons=human_reasons,
        metrics=metrics["qc_cv_pct"].tolist(),
        thresholds=[config.max_cv_pct] * n_input,
    )

    kept_out = metrics.loc[kept_mask, ["feature_id"]].reset_index(drop=True)
    return QcCvFilterResult(
        kept_features=kept_out,
        metrics=metrics,
        n_input=n_input,
        n_kept=n_kept,
        n_dropped_cv=n_dropped_cv,
        n_dropped_detections=n_dropped_det,
    )
