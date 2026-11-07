"""Pandera DataFrameModel schemas for Stage 1 output tables.

Each schema defines the contract for one parquet file. Violations at a stage
boundary fail-fast with a readable error — the pipeline never silently drifts.

All schemas coerce string → numeric / datetime on load (Progenesis export gives
us strings). ``strict=True`` means unknown columns raise; ``coerce=True`` means
dtypes are enforced on load.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.engines.pandas_engine import PydanticModel  # noqa: F401  (future use)
from pandera.typing import DataFrame, Series  # noqa: F401


# ============================================================
# 1. features.parquet  —  per-compound metadata (one row per feature)
# ============================================================
class FeaturesSchema(pa.DataFrameModel):
    """One row per feature (Progenesis peak). 30,370 rows for the I_66 dataset."""

    feature_id: Series[str] = pa.Field(unique=True, nullable=False)
    # Neutral mass is only populated when Progenesis resolved the adduct; ~24 %
    # of features in a typical untargeted dataset have only m/z (suffix
    # ``m/z`` in the compound id) and neutral mass is NaN. Keep nullable.
    neutral_mass: Series[float] = pa.Field(gt=0, nullable=True)
    m_over_z: Series[float] = pa.Field(gt=0, nullable=False)
    charge: Series[int] = pa.Field(ge=0, nullable=False)  # 0 for uncharged
    rt_min: Series[float] = pa.Field(ge=0, nullable=False)
    measured_ccs: Series[float] = pa.Field(ge=0, nullable=True)
    peak_width_min: Series[float] = pa.Field(ge=0, nullable=True)
    identifications_count: Series[int] = pa.Field(ge=0, nullable=True)
    anova_p_progenesis: Series[float] = pa.Field(ge=0, le=1, nullable=True)
    max_fold_change_progenesis: Series[float] = pa.Field(ge=0, nullable=True)
    highest_mean_group: Series[str] = pa.Field(nullable=True)
    lowest_mean_group: Series[str] = pa.Field(nullable=True)
    isotope_distribution: Series[str] = pa.Field(nullable=True)
    maximum_abundance: Series[float] = pa.Field(ge=0, nullable=True)
    minimum_cv_pct: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = True
        coerce = True


# ============================================================
# 2. samples.parquet  —  per-injection metadata (one row per injection)
# ============================================================
class SamplesSchema(pa.DataFrameModel):
    """One row per injection. ~53 rows: 31 biological + 16 QC + 3 blank + 3 dilution."""

    sample_id: Series[str] = pa.Field(unique=True, nullable=False)
    raw_sample_id: Series[str] = pa.Field(nullable=False)
    injection_type: Series[str] = pa.Field(
        isin=["biological", "qc", "blank", "dilution"], nullable=False
    )
    cell_line: Series[str] = pa.Field(nullable=True)
    replicate: Series[int] = pa.Field(ge=1, nullable=False)
    dilution_level: Series[float] = pa.Field(ge=1, nullable=True)  # 2, 4, 8 (only for dilution)
    acq_date: Series[pa.DateTime] = pa.Field(nullable=False)
    column_order: Series[int] = pa.Field(ge=0, nullable=False)

    class Config:
        strict = True
        coerce = True


# ============================================================
# 3. abundances_{raw,normalised}_long.parquet  —  tidy long
# ============================================================
class AbundancesLongSchema(pa.DataFrameModel):
    """Tidy long-format abundances. Rows = features × samples."""

    feature_id: Series[str] = pa.Field(nullable=False)
    sample_id: Series[str] = pa.Field(nullable=False)
    intensity: Series[float] = pa.Field(ge=0, nullable=True)  # missing values allowed

    class Config:
        strict = True
        coerce = True


# ============================================================
# 4. identification_long.parquet  —  one row per (feature, candidate)
# ============================================================
class IdentificationLongSchema(pa.DataFrameModel):
    """One row per (feature, candidate_idx). ~1M rows for the I_66 dataset."""

    feature_id: Series[str] = pa.Field(nullable=False)
    candidate_idx: Series[int] = pa.Field(ge=0, nullable=False)
    compound_id: Series[str] = pa.Field(nullable=True)  # e.g. HMDB0242755
    accepted: Series[bool] = pa.Field(nullable=True)
    adducts: Series[str] = pa.Field(nullable=True)
    formula: Series[str] = pa.Field(nullable=True)
    score: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    fragmentation_score: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    mass_error_ppm: Series[float] = pa.Field(nullable=True)
    theoretical_ccs: Series[float] = pa.Field(ge=0, nullable=True)
    dccs: Series[float] = pa.Field(nullable=True)
    isotope_similarity: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    theoretical_isotope_distribution: Series[str] = pa.Field(nullable=True)
    link: Series[str] = pa.Field(nullable=True)
    description: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True


# ============================================================
# 5. dilution_ladder_long.parquet  —  dilution series (long)
# ============================================================
class DilutionLadderLongSchema(pa.DataFrameModel):
    """Long format: one row per (feature, dilution level, block)."""

    feature_id: Series[str] = pa.Field(nullable=False)
    dilution_level: Series[float] = pa.Field(ge=1, nullable=False)  # 1, 2, 4, 8
    dilution_factor: Series[float] = pa.Field(gt=0, le=1, nullable=False)  # 1.0, 0.5, 0.25, 0.125
    block: Series[str] = pa.Field(isin=["raw", "normalised"], nullable=False)
    intensity: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = True
        coerce = True


# ============================================================
# 6. blanks_long.parquet  —  blank injections (long)
# ============================================================
class BlanksLongSchema(pa.DataFrameModel):
    """Long format: one row per (feature, blank replicate, block)."""

    feature_id: Series[str] = pa.Field(nullable=False)
    blank_replicate: Series[int] = pa.Field(ge=1, nullable=False)
    block: Series[str] = pa.Field(isin=["raw", "normalised"], nullable=False)
    intensity: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = True
        coerce = True


# ============================================================
# 7. audit/<stage>.parquet  —  per-feature, per-stage audit row
# ============================================================
class AuditSchema(pa.DataFrameModel):
    """Per-feature audit row. One row per (feature_id, stage, action)."""

    feature_id: Series[str] = pa.Field(nullable=False)
    stage: Series[str] = pa.Field(nullable=False)
    action: Series[str] = pa.Field(isin=["kept", "dropped", "modified", "created"], nullable=False)
    reason: Series[str] = pa.Field(nullable=False)
    metric: Series[float] = pa.Field(nullable=True)
    threshold: Series[float] = pa.Field(nullable=True)
    timestamp: Series[pa.DateTime] = pa.Field(nullable=False)
    stage_params_hash: Series[str] = pa.Field(nullable=False)

    class Config:
        strict = True
        coerce = True
