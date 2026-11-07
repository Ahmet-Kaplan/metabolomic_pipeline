"""Stage 3 — Modified 80 % rule.

For each feature, compute the percentage of biological samples in each
cell-line group where the feature is **detected** (intensity > 0 AND not NaN).
Keep the feature if its maximum presence across groups is ≥ ``presence_pct``.

Rationale (Bijlsma 2006, *Anal Chem* 78:567): the "modified" rule differs from
the strict 80 % rule in requiring ≥ 80 % presence in *at least one* group
rather than in *all* groups. This preserves group-private candidate biomarkers,
which is exactly what we want to find in biomarker discovery.

QC / blank / dilution injections are NOT considered. They are technical
controls, not biological groups — their presence patterns are handled in
Stages 4-6.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import structlog

from biodisc.audit import AuditLogger
from biodisc.filters._output import write_outputs  # noqa: F401 — re-exported for CLI

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PresenceFilterConfig:
    """Runtime config for Stage 3. Mirrors conf/filter/default.yaml:presence."""

    presence_pct: float = 0.80
    emit_csv_mirror: bool = True


@dataclass(frozen=True)
class PresenceFilterResult:
    """Structured result of running the presence filter."""

    kept_features: pd.DataFrame  # one column: feature_id
    metrics: pd.DataFrame  # per-feature per-group presence + max
    n_input: int
    n_kept: int
    n_dropped: int

    @property
    def drop_pct(self) -> float:
        return self.n_dropped / self.n_input * 100 if self.n_input else 0.0


def _is_detected(s: pd.Series) -> pd.Series:
    """True where intensity is > 0 AND not NaN.

    Progenesis encodes "below LOD" as 0 (not NaN) — Stage 2 EDA confirmed this
    empirically. Both are treated as "undetected" for the presence calculation.
    """
    return (s.notna()) & (s > 0)


def compute_presence_metrics(
    abundances_long: pd.DataFrame,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """Per feature × group, compute (present_count, group_size, pct).

    Returns a wide DataFrame with one row per feature, columns:
      feature_id, <group>_present, <group>_size, <group>_pct (for each group),
      max_presence_pct
    """
    bio = samples[samples["injection_type"] == "biological"][["sample_id", "cell_line"]]
    if bio.empty:
        raise ValueError("no biological samples in samples metadata — cannot compute presence")

    # Narrow abundances to biological samples only
    bio_ids = set(bio["sample_id"])
    abund = abundances_long[abundances_long["sample_id"].isin(bio_ids)].copy()
    abund["detected"] = _is_detected(abund["intensity"]).astype("int8")

    # Join in cell_line
    abund = abund.merge(bio, on="sample_id", how="left")

    # Group sizes (denominator per cell line)
    group_sizes = bio.groupby("cell_line").size().to_dict()

    # Aggregate: per (feature_id, cell_line) count detections
    pivot = (
        abund.groupby(["feature_id", "cell_line"])["detected"]
        .sum()
        .unstack(fill_value=0)
        .astype("int32")
    )
    # pivot columns are cell_line names; ensure deterministic order
    groups = sorted(pivot.columns.tolist())
    pivot = pivot[groups]

    # Build the metrics frame
    out = pd.DataFrame(index=pivot.index)
    pct_cols: list[str] = []
    for g in groups:
        size = int(group_sizes.get(g, 0))
        out[f"{g}_present"] = pivot[g]
        out[f"{g}_size"] = size
        if size > 0:
            out[f"{g}_pct"] = pivot[g] / size
        else:
            # Should never happen with a well-formed samples table, but guard anyway
            log.warning("group_has_zero_size", cell_line=g)
            out[f"{g}_pct"] = pd.Series(0.0, index=pivot.index)
        pct_cols.append(f"{g}_pct")

    out["max_presence_pct"] = out[pct_cols].max(axis=1)
    return out.reset_index()


def run(
    abundances_long: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    config: PresenceFilterConfig,
    audit: AuditLogger,
) -> PresenceFilterResult:
    """Apply the modified 80 % rule and emit a result + audit rows."""
    metrics = compute_presence_metrics(abundances_long, samples)
    threshold = float(config.presence_pct)
    metrics["kept"] = metrics["max_presence_pct"] >= threshold

    n_input = len(metrics)
    kept_mask = metrics["kept"].to_numpy()
    n_kept = int(kept_mask.sum())
    n_dropped = n_input - n_kept

    log.info(
        "presence_filter_result",
        n_input=n_input,
        n_kept=n_kept,
        n_dropped=n_dropped,
        drop_pct=round(n_dropped / n_input * 100, 2) if n_input else 0.0,
        threshold=threshold,
    )

    # Audit rows: per-feature
    actions = ["kept" if k else "dropped" for k in kept_mask]
    reasons = [
        f"max_presence={p:.3f}_ge_{threshold:.2f}"
        if k
        else f"max_presence={p:.3f}_lt_{threshold:.2f}"
        for k, p in zip(kept_mask, metrics["max_presence_pct"].to_numpy(), strict=True)
    ]
    audit.record_bulk(
        feature_ids=metrics["feature_id"].tolist(),
        actions=actions,
        reasons=reasons,
        metrics=metrics["max_presence_pct"].tolist(),
        thresholds=[threshold] * n_input,
    )

    kept_features = metrics.loc[kept_mask, ["feature_id"]].reset_index(drop=True)
    return PresenceFilterResult(
        kept_features=kept_features,
        metrics=metrics,
        n_input=n_input,
        n_kept=n_kept,
        n_dropped=n_dropped,
    )
