"""Shared types for the multiverse pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class StageContext:
    """Everything a multiverse-stage method might need.

    Methods should read what they need and ignore the rest. ``run_order`` may
    be ``None`` until the injection-sequence list is available — drift
    methods (and only drift methods) raise if it's missing.

    Attributes
    ----------
    matrix
        Wide abundance matrix: rows indexed by ``feature_id``, columns are
        sample IDs (strings). Missing values are ``NaN``.
    samples
        Long-format sample metadata: ``sample_id``, ``injection_type``,
        ``cell_line``, ``replicate``, ``acq_date``, ``column_order`` (what
        Progenesis exported, not necessarily true injection order).
    features
        Long-format feature metadata: ``feature_id``, ``m_over_z``, ``rt_min``,
        ``measured_ccs``, etc.
    run_order
        Optional injection sequence: ``sample_id`` → integer position
        ``1..53``. ``None`` until provided. Drift correction needs this.
    extra
        Catch-all for path-specific or stage-specific carry-along data.
    """

    matrix: pd.DataFrame
    samples: pd.DataFrame
    features: pd.DataFrame
    run_order: pd.DataFrame | None = None
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.matrix)

    @property
    def n_samples(self) -> int:
        return self.matrix.shape[1]

    @property
    def n_missing(self) -> int:
        """Total number of NaN cells in the abundance matrix."""
        return int(self.matrix.isna().sum().sum())

    @property
    def pct_missing(self) -> float:
        total = self.matrix.size
        return self.n_missing / total * 100 if total else 0.0

    def sample_positions(self) -> pd.Series:
        """Return ``Series[sample_id → position]`` from joined run_order + samples.

        Used by drift-correction methods to align matrix columns with
        chronological injection order.

        Raises
        ------
        ValueError
            If ``run_order`` is None — drift correction cannot run without it.
            Build it first via ``biodisc external build-run-order``.
        ValueError
            If a matrix column has no corresponding row in run_order
            (typically a sample present in samples.parquet but missing a
            ``_HEADER.TXT`` file).
        """
        if self.run_order is None:
            raise ValueError(
                "ctx.run_order is None — drift correction needs the chronological "
                "injection order. Build it first: `biodisc external build-run-order`."
            )
        merged = self.samples.merge(
            self.run_order[["raw_sample_id", "position"]],
            on="raw_sample_id",
            how="inner",
        )
        positions = merged.set_index("sample_id")["position"].astype(int)

        # Verify every matrix column has a position
        missing = set(self.matrix.columns) - set(positions.index)
        if missing:
            raise ValueError(
                f"{len(missing)} matrix columns lack a run_order position: "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return positions

    def qc_sample_ids(self) -> list[str]:
        """Return sample_ids whose injection_type == 'qc' AND are in matrix.columns."""
        qc = self.samples.loc[self.samples["injection_type"] == "qc", "sample_id"].tolist()
        return [s for s in qc if s in self.matrix.columns]
