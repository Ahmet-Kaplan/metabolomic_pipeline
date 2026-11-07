"""Stage 6 — Blank filter (Saeed's percentage formula).

For each feature, compute::

                       100 × mean(blank intensities)
        blank_pct  =  ────────────────────────────────
                       mean(biological + QC intensities)

Drop iff ``blank_pct > max_blank_pct`` (default 20 %, mQACC equivalent of
sample/blank ratio ≥ 5; Broadhurst et al. 2018, *Metabolomics* 14:72).

The numerator uses the 3 blank injections (raw block) and the denominator
uses the 31 biological + 16 QC injections (47 total) — matching the Excel
formula: ``ACM2 / ORTALAMA(N2:ACL2) * 100``.

Edge cases handled:
  - mean(samples+QC) == 0  → blank_pct = +inf → drop (no biological signal)
  - mean(blank) == 0       → blank_pct = 0    → kept (perfectly clean)
  - both == 0              → blank_pct = NaN  → drop (no information)
  - n_sq_detected < min_sq_detections → drop (insufficient evidence)
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
class BlankFilterConfig:
    """Mirrors conf/filter/default.yaml:blank."""

    max_blank_pct: float = 20.0
    min_sq_detections: int = 3
    block: str = "raw"  # which abundance block to use for blanks
    emit_csv_mirror: bool = True


@dataclass(frozen=True)
class BlankFilterResult:
    """Structured output of the filter run."""

    kept_features: pd.DataFrame
    metrics: pd.DataFrame
    n_input: int
    n_kept: int
    n_dropped_pct: int  # blank_pct > threshold
    n_dropped_detections: int
    n_dropped_undefined: int  # both means zero / blank_pct NaN

    @property
    def n_dropped(self) -> int:
        return self.n_dropped_pct + self.n_dropped_detections + self.n_dropped_undefined

    @property
    def drop_pct(self) -> float:
        return self.n_dropped / self.n_input * 100 if self.n_input else 0.0


# ============================================================
# Math helper
# ============================================================
def _mean_detected(values: np.ndarray) -> tuple[float, int]:
    """Return (mean, n_detected) where detected = >0 AND not NaN."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    n = arr.size
    return (float(arr.mean()) if n else float("nan"), n)


def _compute_blank_pct(mean_blank_inc_zero: float, mean_sq: float) -> float:
    """blank_pct = 100 × mean_blank / mean_sq, with edge-case handling.

    Note: ``mean_blank_inc_zero`` is the *unconditional* mean across all 3
    blank injections, INCLUDING zeros (Saeed's formula uses raw mean). If all
    3 blanks are zero, this is 0.0 (clean feature). Detection-count guard
    (min_sq_detections) is only applied to the denominator.
    """
    if math.isnan(mean_sq):
        return float("nan")
    if mean_sq == 0:
        # No biological signal — drop regardless of blank
        return float("inf") if mean_blank_inc_zero > 0 else float("nan")
    return 100.0 * mean_blank_inc_zero / mean_sq


# ============================================================
# Per-feature metrics
# ============================================================
def compute_metrics(
    abundances_long: pd.DataFrame,
    blanks_long: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    block: str = "raw",
    kept_features: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Per feature: mean_blank, mean_sq (samples+QC), blank_pct."""
    sq_ids = samples.loc[samples["injection_type"].isin(["biological", "qc"]), "sample_id"].tolist()
    if not sq_ids:
        raise ValueError("no biological/QC samples in samples metadata")

    # Restrict to kept feature universe + block
    blanks = blanks_long[blanks_long["block"] == block].copy()
    abund = abundances_long[abundances_long["sample_id"].isin(sq_ids)].copy()
    if kept_features is not None:
        kf = set(kept_features)
        blanks = blanks[blanks["feature_id"].isin(kf)]
        abund = abund[abund["feature_id"].isin(kf)]

    # Mean blank intensity per feature (raw mean over all 3 replicates,
    # zeros included — Saeed's Excel ORTALAMA semantics)
    blank_means = blanks.groupby("feature_id")["intensity"].mean().rename("mean_blank")
    blank_n = (
        blanks[blanks["intensity"] > 0]
        .groupby("feature_id")["intensity"]
        .size()
        .rename("blank_n_detected")
    )

    # Mean sample+QC intensity per feature (detection-only mean)
    rows: list[dict[str, object]] = []
    for fid, g in abund.groupby("feature_id", sort=False):
        mean_sq, n = _mean_detected(g["intensity"].to_numpy(dtype=float))
        rows.append({"feature_id": fid, "mean_sq": mean_sq, "sq_n_detected": n})
    sq_stats = pd.DataFrame(rows)

    metrics = sq_stats.merge(blank_means.reset_index(), on="feature_id", how="outer").merge(
        blank_n.reset_index(), on="feature_id", how="left"
    )
    metrics["blank_n_detected"] = metrics["blank_n_detected"].fillna(0).astype(int)
    metrics["sq_n_detected"] = metrics["sq_n_detected"].fillna(0).astype(int)
    metrics["mean_blank"] = metrics["mean_blank"].fillna(0.0)
    metrics["blank_pct"] = [
        _compute_blank_pct(b, s)
        for b, s in zip(metrics["mean_blank"], metrics["mean_sq"], strict=True)
    ]
    return metrics


# ============================================================
# Apply filter
# ============================================================
def run(
    abundances_long: pd.DataFrame,
    blanks_long: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    kept_features: Iterable[str],
    config: BlankFilterConfig,
    audit: AuditLogger,
) -> BlankFilterResult:
    """Apply Stage 6 filter and emit per-feature audit rows."""
    metrics = compute_metrics(
        abundances_long,
        blanks_long,
        samples,
        block=config.block,
        kept_features=kept_features,
    )

    threshold = float(config.max_blank_pct)
    enough_sq = metrics["sq_n_detected"] >= config.min_sq_detections
    pct_defined = metrics["blank_pct"].notna()
    pct_passes = metrics["blank_pct"] <= threshold

    kept_mask = enough_sq & pct_defined & pct_passes
    metrics["kept"] = kept_mask

    n_input = len(metrics)
    n_kept = int(kept_mask.sum())

    drop_det = (~enough_sq) & ~kept_mask
    drop_und = (~drop_det) & (~pct_defined) & ~kept_mask
    drop_pct_mask = (~drop_det) & (~drop_und) & ~kept_mask

    n_dropped_detections = int(drop_det.sum())
    n_dropped_undefined = int(drop_und.sum())
    n_dropped_pct = int(drop_pct_mask.sum())

    actions: list[str] = []
    reasons: list[str] = []
    for keep, dd, du, p in zip(kept_mask, drop_det, drop_und, metrics["blank_pct"], strict=True):
        if keep:
            p_str = "inf" if math.isinf(p) else f"{p:.3f}"
            actions.append("kept")
            reasons.append(f"blank_pct={p_str}_le_{threshold:.2f}")
        elif dd:
            actions.append("dropped")
            reasons.append("insufficient_sample_qc_detections")
        elif du:
            actions.append("dropped")
            reasons.append("blank_pct_undefined")
        else:
            p_str = "inf" if math.isinf(p) else ("nan" if math.isnan(p) else f"{p:.3f}")
            actions.append("dropped")
            reasons.append(f"blank_pct={p_str}_gt_{threshold:.2f}")

    audit.record_bulk(
        feature_ids=metrics["feature_id"].tolist(),
        actions=actions,
        reasons=reasons,
        metrics=metrics["blank_pct"].tolist(),
        thresholds=[threshold] * n_input,
    )

    log.info(
        "blank_filter_result",
        n_input=n_input,
        n_kept=n_kept,
        n_dropped_pct=n_dropped_pct,
        n_dropped_detections=n_dropped_detections,
        n_dropped_undefined=n_dropped_undefined,
        max_blank_pct=threshold,
    )

    kept_features_df = metrics.loc[kept_mask, ["feature_id"]].reset_index(drop=True)
    return BlankFilterResult(
        kept_features=kept_features_df,
        metrics=metrics,
        n_input=n_input,
        n_kept=n_kept,
        n_dropped_pct=n_dropped_pct,
        n_dropped_detections=n_dropped_detections,
        n_dropped_undefined=n_dropped_undefined,
    )
