"""Stage 7 — Dilution linearity (linear R²).

For each feature, fit a least-squares line ``y = a·x + b`` to the four
dilution observations::

       i | x_i  | y_i
       --+------+--------------------------------------------
       1 | 1.000| mean intensity across the 16 undiluted QC injections
       2 | 0.500| QC_1:2 intensity
       3 | 0.250| QC_1:4 intensity
       4 | 0.125| QC_1:8 intensity

then compute::

                   SS_res     Σ (y_i − ŷ_i)²
       R² = 1 − ─────────  = 1 − ───────────────
                   SS_tot     Σ (y_i − ȳ)²

Drop iff ``R² < min_r2_linear`` (default 0.7; domain-expert-confirmed linear-only).

Edge cases handled:
  - n_detected < min_dilution_detections (typically 3) → drop, reason
    ``insufficient_dilution_detections`` (need at least 3 of 4 points to fit
    + assess fit honestly).
  - All y values equal (variance 0) → SS_tot = 0 → R² undefined → drop, reason
    ``r2_undefined_constant_y`` (a flat response to dilution is suspicious).
  - All-zero y → drop trivially (counts as 0 detections).

The domain expert confirmed 2026-04-24 that with only 4 points the quadratic
test is mathematically redundant (R²_quad ≥ R²_lin always), so we apply
linear-only.
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
class DilutionFilterConfig:
    """Mirrors conf/filter/default.yaml:dilution."""

    min_r2_linear: float = 0.70
    min_dilution_detections: int = 3
    block: str = "raw"
    emit_csv_mirror: bool = True


@dataclass(frozen=True)
class DilutionFilterResult:
    """Structured output of the filter run."""

    kept_features: pd.DataFrame
    metrics: pd.DataFrame
    n_input: int
    n_kept: int
    n_dropped_r2: int  # R² < threshold
    n_dropped_detections: int  # too few non-zero dilution points
    n_dropped_undefined: int  # constant y / NaN

    @property
    def n_dropped(self) -> int:
        return self.n_dropped_r2 + self.n_dropped_detections + self.n_dropped_undefined

    @property
    def drop_pct(self) -> float:
        return self.n_dropped / self.n_input * 100 if self.n_input else 0.0


# ============================================================
# Math helper
# ============================================================
# Canonical dilution-factor order (matches Stage 1 emission)
DILUTION_FACTORS: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)


def _fit_r2_linear(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_detections: int,
) -> tuple[float, float, float, int]:
    """Return (slope, intercept, r2, n_detected) for a linear fit on detected points.

    Detection: y > 0 AND finite. Drops the corresponding x values too.
    Returns NaNs if n < min_detections OR Var(y) == 0.
    """
    mask = np.isfinite(y) & (y > 0)
    n = int(mask.sum())
    if n < min_detections:
        return float("nan"), float("nan"), float("nan"), n
    xv, yv = x[mask], y[mask]
    var_y = float(np.var(yv, ddof=0))
    if var_y == 0.0:
        return float("nan"), float("nan"), float("nan"), n
    slope, intercept = np.polyfit(xv, yv, 1)
    yp = slope * xv + intercept
    ss_res = float(np.sum((yv - yp) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2, n


# ============================================================
# Per-feature metrics
# ============================================================
def compute_metrics(
    dilution_long: pd.DataFrame,
    *,
    block: str = "raw",
    kept_features: Iterable[str] | None = None,
    min_detections: int = 3,
) -> pd.DataFrame:
    """Per feature: 4-point dilution intensities + linear fit + R²."""
    raw = dilution_long[dilution_long["block"] == block].copy()
    if kept_features is not None:
        kf = set(kept_features)
        raw = raw[raw["feature_id"].isin(kf)]
    if raw.empty:
        raise ValueError("dilution_long has no rows for the requested block / kept features")

    # Aggregate: at factor=1.0 there are 16 QC injections per feature; mean them.
    # At other factors there's 1 measurement each, but mean is a no-op there.
    agg = (
        raw.groupby(["feature_id", "dilution_factor"])["intensity"]
        .mean()
        .unstack()
        .reindex(columns=list(DILUTION_FACTORS))
    )

    x_arr = np.array(DILUTION_FACTORS, dtype=float)
    rows: list[dict[str, float | str | int]] = []
    for fid, vals in agg.iterrows():
        y_arr = vals.to_numpy(dtype=float)
        slope, intercept, r2, n = _fit_r2_linear(x_arr, y_arr, min_detections=min_detections)
        rows.append(
            {
                "feature_id": str(fid),
                "y_factor_1": float(y_arr[0]) if not math.isnan(y_arr[0]) else float("nan"),
                "y_factor_0_5": float(y_arr[1]) if not math.isnan(y_arr[1]) else float("nan"),
                "y_factor_0_25": float(y_arr[2]) if not math.isnan(y_arr[2]) else float("nan"),
                "y_factor_0_125": float(y_arr[3]) if not math.isnan(y_arr[3]) else float("nan"),
                "n_detected": n,
                "slope": slope,
                "intercept": intercept,
                "r2_linear": r2,
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# Apply filter
# ============================================================
def run(
    dilution_long: pd.DataFrame,
    *,
    kept_features: Iterable[str],
    config: DilutionFilterConfig,
    audit: AuditLogger,
) -> DilutionFilterResult:
    """Apply Stage 7 filter and emit per-feature audit rows."""
    metrics = compute_metrics(
        dilution_long,
        block=config.block,
        kept_features=kept_features,
        min_detections=config.min_dilution_detections,
    )

    threshold = float(config.min_r2_linear)
    enough = metrics["n_detected"] >= config.min_dilution_detections
    r2_defined = metrics["r2_linear"].notna()
    r2_passes = metrics["r2_linear"] >= threshold

    kept_mask = enough & r2_defined & r2_passes
    metrics["kept"] = kept_mask

    n_input = len(metrics)
    n_kept = int(kept_mask.sum())

    drop_det = (~enough) & ~kept_mask
    drop_und = (~drop_det) & (~r2_defined) & ~kept_mask
    drop_r2 = (~drop_det) & (~drop_und) & ~kept_mask

    n_dropped_detections = int(drop_det.sum())
    n_dropped_undefined = int(drop_und.sum())
    n_dropped_r2 = int(drop_r2.sum())

    actions: list[str] = []
    reasons: list[str] = []
    for keep, dd, du, r in zip(kept_mask, drop_det, drop_und, metrics["r2_linear"], strict=True):
        if keep:
            actions.append("kept")
            reasons.append(f"r2={r:.3f}_ge_{threshold:.2f}")
        elif dd:
            actions.append("dropped")
            reasons.append("insufficient_dilution_detections")
        elif du:
            actions.append("dropped")
            reasons.append("r2_undefined_constant_y")
        else:
            r_str = "nan" if math.isnan(r) else f"{r:.3f}"
            actions.append("dropped")
            reasons.append(f"r2={r_str}_lt_{threshold:.2f}")

    audit.record_bulk(
        feature_ids=metrics["feature_id"].tolist(),
        actions=actions,
        reasons=reasons,
        metrics=metrics["r2_linear"].tolist(),
        thresholds=[threshold] * n_input,
    )

    log.info(
        "dilution_filter_result",
        n_input=n_input,
        n_kept=n_kept,
        n_dropped_r2=n_dropped_r2,
        n_dropped_detections=n_dropped_detections,
        n_dropped_undefined=n_dropped_undefined,
        min_r2=threshold,
    )

    kept_features_df = metrics.loc[kept_mask, ["feature_id"]].reset_index(drop=True)
    return DilutionFilterResult(
        kept_features=kept_features_df,
        metrics=metrics,
        n_input=n_input,
        n_kept=n_kept,
        n_dropped_r2=n_dropped_r2,
        n_dropped_detections=n_dropped_detections,
        n_dropped_undefined=n_dropped_undefined,
    )
