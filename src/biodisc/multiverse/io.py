"""Multiverse I/O — build the initial ``StageContext`` from upstream outputs.

The Progenesis CSV pipeline (Stages 1–10) encodes missing abundances as **zero**,
not NaN. There is no concept of "true zero" abundance in LC-MS — zero means
"no peak detected", which is operationally identical to "below LOD". So the
first thing the multiverse does is **convert zeros to NaN** so imputers can
treat missingness explicitly.

This module is the canonical entry point: every multiverse path orchestrator
calls ``build_initial_context()`` to get a clean, ready-to-impute matrix.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext

log = structlog.get_logger(__name__)

# Sample types that proceed to multivariate analysis. Blanks and dilutions
# served their filtering purpose in Stages 6–7 and don't continue.
DEFAULT_SAMPLE_TYPES: tuple[str, ...] = ("biological", "qc")

# Stage 10 origin-filter outcome categories that ``load_active_feature_ids``
# can select on. Mirrors the four buckets produced by ``filters.origin.run``.
# ``all_exogenous`` is omitted: those features are already dropped at Stage 10
# and never appear in either ``kept_features.parquet`` or in metrics rows with
# ``kept=True``.
VALID_OUTCOMES: frozenset[str] = frozenset({
    "endogenous_evidence",
    "exo_or_unknown",
    "passthrough_l4",
})


def load_active_feature_ids(
    stage_10_dir: Path,
    outcome_filter: Iterable[str] | None = None,
) -> list[str]:
    """Load Stage-10 active feature IDs, optionally restricted by origin outcome.

    Default (``outcome_filter=None``) preserves the historical contract: it
    returns every feature_id in ``kept_features.parquet``. Passing an explicit
    set of outcomes filters ``metrics.parquet`` to those rows whose ``outcome``
    column is in the requested set.

    Parameters
    ----------
    stage_10_dir
        Directory containing Stage-10 outputs
        (``kept_features.parquet`` and ``metrics.parquet``).
    outcome_filter
        ``None`` to keep all (current default) or a subset of
        :data:`VALID_OUTCOMES`. Unknown names raise ``ValueError``.

    Returns
    -------
    list[str]
        Active feature IDs, ordered as in the source file.

    Raises
    ------
    ValueError
        If ``outcome_filter`` contains an unknown outcome name, or if the
        resulting feature list would be empty (so the caller fails loudly
        instead of silently feeding the multiverse zero features).
    """
    if outcome_filter is None:
        kept = pd.read_parquet(stage_10_dir / "kept_features.parquet")
        ids = kept["feature_id"].astype(str).tolist()
        log.info(
            "active_features_loaded",
            source="kept_features.parquet",
            n=len(ids),
            outcome_filter=None,
        )
        return ids

    requested = set(outcome_filter)
    unknown = requested - VALID_OUTCOMES
    if unknown:
        raise ValueError(
            f"Unknown outcome(s) in outcome_filter: {sorted(unknown)}. "
            f"Valid options: {sorted(VALID_OUTCOMES)}."
        )

    metrics = pd.read_parquet(stage_10_dir / "metrics.parquet")
    # Stage-10 metrics.parquet contains both kept and dropped rows; only kept
    # ones are eligible. The 'all_exogenous' rows have kept=False and are
    # excluded automatically by the .isin(requested) filter as well.
    kept_only = metrics[metrics["kept"]]
    filtered = kept_only.loc[
        kept_only["outcome"].isin(requested), "feature_id"
    ].astype(str).tolist()

    available_counts = kept_only["outcome"].value_counts().to_dict()
    if not filtered:
        raise ValueError(
            f"outcome_filter={sorted(requested)} produced 0 features. "
            f"Available kept-outcome counts in this Stage-10 run: "
            f"{available_counts}."
        )

    breakdown = {o: int(available_counts.get(o, 0)) for o in sorted(VALID_OUTCOMES)}
    log.info(
        "active_features_filtered_by_outcome",
        outcomes_allowed=sorted(requested),
        n_total_kept=len(kept_only),
        n_selected=len(filtered),
        breakdown=breakdown,
    )
    return filtered


def build_initial_context(
    *,
    abundances_long: pd.DataFrame,
    samples: pd.DataFrame,
    features: pd.DataFrame,
    active_feature_ids: Iterable[str],
    sample_types: Iterable[str] = DEFAULT_SAMPLE_TYPES,
    treat_zero_as_missing: bool = True,
    run_order: pd.DataFrame | None = None,
) -> StageContext:
    """Pivot upstream long-form abundances into the multiverse's wide matrix.

    Parameters
    ----------
    abundances_long
        Long-format abundance table from Stage 1.
    samples
        Sample metadata from Stage 1.
    features
        Feature metadata from Stage 1.
    active_feature_ids
        Feature IDs that survived Stage 10 (origin filter).
    sample_types
        Which injection types to include. Default biological + QC.
    treat_zero_as_missing
        If True (default), zero intensities are converted to NaN. This is
        the metabolomics-standard convention — zero means "below LOD".
        Set False only when you know your data has true-zero abundances
        (very unusual in LC-MS).
    run_order
        Optional. ``DataFrame[sample_id, position]`` for drift correction.
        ``None`` until the injection-sequence list is available.
    """
    active_set = set(active_feature_ids)
    sample_types_set = set(sample_types)

    bio_qc_samples = samples[samples["injection_type"].isin(sample_types_set)].copy()
    sample_ids = set(bio_qc_samples["sample_id"])

    sub = abundances_long[
        abundances_long["feature_id"].isin(active_set)
        & abundances_long["sample_id"].isin(sample_ids)
    ].copy()

    n_zero = int((sub["intensity"] == 0).sum())
    if treat_zero_as_missing and n_zero > 0:
        sub.loc[sub["intensity"] == 0, "intensity"] = np.nan
        log.info(
            "zero_to_nan_conversion",
            n_zeros_converted=n_zero,
            pct=n_zero / len(sub) * 100,
        )

    wide = sub.pivot(index="feature_id", columns="sample_id", values="intensity")

    feats_subset = features[features["feature_id"].isin(active_set)]

    log.info(
        "stage_context_built",
        n_features=len(wide),
        n_samples=wide.shape[1],
        n_missing=int(wide.isna().sum().sum()),
        pct_missing=float(wide.isna().mean().mean() * 100),
        sample_types=sorted(sample_types_set),
    )

    return StageContext(
        matrix=wide,
        samples=bio_qc_samples,
        features=feats_subset,
        run_order=run_order,
    )


def build_initial_context_from_paths(
    *,
    stage_01_dir: Path,
    stage_10_dir: Path,
    sample_types: Iterable[str] = DEFAULT_SAMPLE_TYPES,
    treat_zero_as_missing: bool = True,
    run_order_path: Path | None = None,
    outcome_filter: Iterable[str] | None = None,
) -> StageContext:
    """Convenience: load the four upstream parquets and call ``build_initial_context``.

    ``run_order_path`` accepts either ``.parquet`` (canonical) or ``.csv``.
    If ``None`` and a default ``data/external/run_order.parquet`` exists at
    the project root, that's used automatically.

    ``outcome_filter`` is forwarded to :func:`load_active_feature_ids`; pass
    a subset of :data:`VALID_OUTCOMES` to restrict the multiverse input by
    Stage-10 origin category. ``None`` (default) keeps every feature in
    ``kept_features.parquet`` — backward-compatible.
    """
    abundances = pd.read_parquet(stage_01_dir / "abundances_raw_long.parquet")
    samples = pd.read_parquet(stage_01_dir / "samples.parquet")
    features = pd.read_parquet(stage_01_dir / "features.parquet")
    kept_ids = load_active_feature_ids(stage_10_dir, outcome_filter=outcome_filter)

    run_order: pd.DataFrame | None = None
    if run_order_path is not None and run_order_path.exists():
        if run_order_path.suffix == ".parquet":
            run_order = pd.read_parquet(run_order_path)
        else:
            run_order = pd.read_csv(run_order_path)
        log.info("run_order_loaded", rows=len(run_order), path=str(run_order_path))

    return build_initial_context(
        abundances_long=abundances,
        samples=samples,
        features=features,
        active_feature_ids=kept_ids,
        sample_types=sample_types,
        treat_zero_as_missing=treat_zero_as_missing,
        run_order=run_order,
    )
