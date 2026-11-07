"""Stage 5 — Sample-RSD / QC-RSD ratio filter.

For each feature, compare the relative standard deviation across biological
samples to the same quantity across QC injections::

                        RSD_samples
            ratio  =  ──────────────
                          RSD_QC

Keep iff ``ratio > min_ratio`` (default 1.0). The intuition: when biological
variation exceeds technical (QC) variation, the feature carries usable
biology; otherwise the signal is dominated by instrument noise.

This is the **inverted D-ratio** of Broadhurst et al. 2018 (mQACC). The
canonical D-ratio = SD_QC / SD_samples ≤ 0.5 is equivalent to our ratio ≥ 2;
the threshold here is more permissive (ratio > 1 ↔ D-ratio < 1) per Saeed's
the domain-expert specification.

Edge cases handled:
  - mean(bio) == 0 or mean(QC) == 0 → RSD undefined → drop
  - n_bio_detected < min_sample_detections → drop (insufficient evidence)
  - n_qc_detected  < min_qc_detections     → drop (can't trust the denominator)
  - RSD_QC == 0 (perfect QC stability) → ratio = +inf → kept
  - Both RSDs == 0 → ratio undefined (NaN) → drop
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from biodisc.audit import AuditLogger
from biodisc.filters._output import write_outputs  # noqa: F401 — re-exported for CLI

log = structlog.get_logger(__name__)


# ============================================================
# Config + result dataclasses
# ============================================================
@dataclass(frozen=True)
class SampleQcRatioFilterConfig:
    """Runtime config; mirrors conf/filter/default.yaml:sample_qc_ratio."""

    min_ratio: float = 1.0
    min_sample_detections: int = 3
    min_qc_detections: int = 3
    emit_csv_mirror: bool = True


@dataclass(frozen=True)
class SampleQcRatioFilterResult:
    """Structured output of the filter run."""

    kept_features: pd.DataFrame  # one column: feature_id
    metrics: pd.DataFrame
    n_input: int
    n_kept: int
    n_dropped_ratio: int  # ratio ≤ threshold
    n_dropped_detections: int  # detection count guard tripped
    n_dropped_undefined: int  # both RSDs zero / mean zero — ratio NaN

    @property
    def n_dropped(self) -> int:
        return self.n_dropped_ratio + self.n_dropped_detections + self.n_dropped_undefined

    @property
    def drop_pct(self) -> float:
        return self.n_dropped / self.n_input * 100 if self.n_input else 0.0


# ============================================================
# Math helpers
# ============================================================
def _rsd_pct(values: np.ndarray) -> tuple[float, float, float, int]:
    """Return (mean, sd_ddof1, rsd_pct, n_detected) for an array of intensities.

    Detection: value > 0 AND not NaN. SD uses Bessel's correction (ddof=1).
    Returns NaNs if n < 2 or mean == 0.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    n = arr.size
    if n < 2:
        mean = float(arr.mean()) if n else float("nan")
        return mean, float("nan"), float("nan"), n
    mean = float(arr.mean())
    if mean == 0:
        return mean, float("nan"), float("nan"), n
    sd = float(arr.std(ddof=1))
    rsd = 100.0 * sd / mean
    return mean, sd, rsd, n


def _compute_ratio(rsd_bio: float, rsd_qc: float) -> float:
    """Sample-RSD / QC-RSD ratio with documented edge-case handling.

    - both NaN → NaN (undefined)
    - rsd_qc == 0 and rsd_bio > 0 → +inf (perfect QC, any biology — KEEP)
    - rsd_qc == 0 and rsd_bio == 0 → NaN (no information either way)
    """
    if math.isnan(rsd_bio) or math.isnan(rsd_qc):
        return float("nan")
    if rsd_qc == 0:
        return float("inf") if rsd_bio > 0 else float("nan")
    return rsd_bio / rsd_qc


# ============================================================
# Per-feature metrics
# ============================================================
def compute_metrics(
    abundances_long: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    kept_features: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Per feature: RSD over biological samples and over QC samples + ratio.

    Returns a DataFrame with columns:
      feature_id, sample_mean, sample_sd, sample_rsd_pct, sample_n_detected,
      qc_mean, qc_sd, qc_rsd_pct, qc_n_detected, ratio
    """
    bio_ids = samples.loc[samples["injection_type"] == "biological", "sample_id"]
    qc_ids = samples.loc[samples["injection_type"] == "qc", "sample_id"]
    if bio_ids.empty:
        raise ValueError("no biological samples in samples metadata")
    if qc_ids.empty:
        raise ValueError("no QC samples in samples metadata")

    abund = abundances_long
    if kept_features is not None:
        kf = set(kept_features)
        abund = abund[abund["feature_id"].isin(kf)]

    bio_mask = abund["sample_id"].isin(bio_ids)
    qc_mask = abund["sample_id"].isin(qc_ids)

    def _per_feature(group: pd.DataFrame, mask: pd.Series) -> pd.Series:
        # called inside groupby — `mask` aligned by groupby index
        vals = group["intensity"].to_numpy(dtype=float)
        sub = vals[mask.loc[group.index].to_numpy()]
        mean, sd, rsd, n = _rsd_pct(sub)
        return pd.Series({"mean": mean, "sd": sd, "rsd_pct": rsd, "n_detected": n})

    # Compute per-feature stats for bio and QC separately (vectorised)
    bio_long = abund[bio_mask]
    qc_long = abund[qc_mask]

    def _agg(long_frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        rows = []
        for fid, g in long_frame.groupby("feature_id", sort=False):
            mean, sd, rsd, n = _rsd_pct(g["intensity"].to_numpy(dtype=float))
            rows.append(
                {
                    "feature_id": fid,
                    f"{prefix}_mean": mean,
                    f"{prefix}_sd": sd,
                    f"{prefix}_rsd_pct": rsd,
                    f"{prefix}_n_detected": n,
                }
            )
        return pd.DataFrame(rows)

    bio_stats = _agg(bio_long, "sample")
    qc_stats = _agg(qc_long, "qc")

    metrics = bio_stats.merge(qc_stats, on="feature_id", how="outer")
    # Fill missing groups (e.g. feature absent entirely from QCs) with NaN/0
    for col in ["sample_mean", "sample_sd", "sample_rsd_pct", "qc_mean", "qc_sd", "qc_rsd_pct"]:
        if col not in metrics.columns:
            metrics[col] = np.nan
    for col in ["sample_n_detected", "qc_n_detected"]:
        if col not in metrics.columns:
            metrics[col] = 0
        metrics[col] = metrics[col].fillna(0).astype(int)

    metrics["ratio"] = [
        _compute_ratio(b, q)
        for b, q in zip(metrics["sample_rsd_pct"], metrics["qc_rsd_pct"], strict=True)
    ]
    return metrics


# ============================================================
# Apply filter
# ============================================================
def run(
    abundances_long: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    kept_features: Iterable[str],
    config: SampleQcRatioFilterConfig,
    audit: AuditLogger,
) -> SampleQcRatioFilterResult:
    """Apply Stage 5 filter and emit per-feature audit rows."""
    metrics = compute_metrics(abundances_long, samples, kept_features=kept_features)

    threshold = float(config.min_ratio)
    enough_bio = metrics["sample_n_detected"] >= config.min_sample_detections
    enough_qc = metrics["qc_n_detected"] >= config.min_qc_detections
    ratio_defined = metrics["ratio"].notna()
    ratio_passes = metrics["ratio"] > threshold

    kept_mask = enough_bio & enough_qc & ratio_defined & ratio_passes
    metrics["kept"] = kept_mask

    n_input = len(metrics)
    n_kept = int(kept_mask.sum())

    # Mutually-exclusive drop reasons (priority: detection guard > undefined > ratio)
    drop_detections_mask = ~(enough_bio & enough_qc) & ~kept_mask
    drop_undefined_mask = ~drop_detections_mask & ~ratio_defined & ~kept_mask
    drop_ratio_mask = ~drop_detections_mask & ~drop_undefined_mask & ~kept_mask

    n_dropped_detections = int(drop_detections_mask.sum())
    n_dropped_undefined = int(drop_undefined_mask.sum())
    n_dropped_ratio = int(drop_ratio_mask.sum())

    # Build per-feature audit rows
    actions: list[str] = []
    reasons: list[str] = []
    for keep, det_drop, und_drop, r in zip(
        kept_mask,
        drop_detections_mask,
        drop_undefined_mask,
        metrics["ratio"],
        strict=True,
    ):
        if keep:
            r_str = "inf" if math.isinf(r) else f"{r:.3f}"
            actions.append("kept")
            reasons.append(f"ratio={r_str}_gt_{threshold:.2f}")
        elif det_drop:
            actions.append("dropped")
            reasons.append("insufficient_detections")
        elif und_drop:
            actions.append("dropped")
            reasons.append("ratio_undefined")
        else:
            r_str = "nan" if math.isnan(r) else f"{r:.3f}"
            actions.append("dropped")
            reasons.append(f"ratio={r_str}_le_{threshold:.2f}")

    # Use ratio as the audit metric (NaN survives parquet round-trip)
    audit.record_bulk(
        feature_ids=metrics["feature_id"].tolist(),
        actions=actions,
        reasons=reasons,
        metrics=metrics["ratio"].tolist(),
        thresholds=[threshold] * n_input,
    )

    log.info(
        "sample_qc_ratio_filter_result",
        n_input=n_input,
        n_kept=n_kept,
        n_dropped_ratio=n_dropped_ratio,
        n_dropped_detections=n_dropped_detections,
        n_dropped_undefined=n_dropped_undefined,
        min_ratio=threshold,
        drop_pct=round((n_input - n_kept) / n_input * 100, 2) if n_input else 0.0,
    )

    kept_features_df = metrics.loc[kept_mask, ["feature_id"]].reset_index(drop=True)
    return SampleQcRatioFilterResult(
        kept_features=kept_features_df,
        metrics=metrics,
        n_input=n_input,
        n_kept=n_kept,
        n_dropped_ratio=n_dropped_ratio,
        n_dropped_detections=n_dropped_detections,
        n_dropped_undefined=n_dropped_undefined,
    )
