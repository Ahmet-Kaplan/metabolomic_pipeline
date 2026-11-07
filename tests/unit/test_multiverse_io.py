"""Unit tests — multiverse.io.load_active_feature_ids + outcome filter wiring.

These guard the contract that the multiverse loader can optionally restrict
its input to a subset of Stage-10 outcome categories without changing the
default ('all kept features') behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from biodisc.multiverse.io import (
    VALID_OUTCOMES,
    load_active_feature_ids,
)


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def stage_10_dir(tmp_path: Path) -> Path:
    """Synthetic Stage-10 outputs covering all four outcome categories."""
    metrics = pd.DataFrame(
        [
            # feature_id, outcome, kept
            ("f_endo_1", "endogenous_evidence", True),
            ("f_endo_2", "endogenous_evidence", True),
            ("f_endo_3", "endogenous_evidence", True),
            ("f_exoUnk_1", "exo_or_unknown", True),
            ("f_exoUnk_2", "exo_or_unknown", True),
            ("f_l4_1", "passthrough_l4", True),
            ("f_l4_2", "passthrough_l4", True),
            ("f_l4_3", "passthrough_l4", True),
            ("f_l4_4", "passthrough_l4", True),
            # Two dropped — all_exogenous never appears as kept=True downstream
            ("f_dropped_1", "all_exogenous", False),
            ("f_dropped_2", "all_exogenous", False),
        ],
        columns=["feature_id", "outcome", "kept"],
    )
    metrics.to_parquet(tmp_path / "metrics.parquet")

    kept = metrics.loc[metrics["kept"], ["feature_id"]].reset_index(drop=True)
    kept.to_parquet(tmp_path / "kept_features.parquet")
    return tmp_path


# ============================================================
# Default behaviour (None) — backward-compatible
# ============================================================
def test_default_returns_all_kept_features(stage_10_dir: Path) -> None:
    """outcome_filter=None must read kept_features.parquet, preserving the
    historical contract (every feature surviving Stage 10)."""
    ids = load_active_feature_ids(stage_10_dir, outcome_filter=None)
    assert len(ids) == 9
    assert set(ids) == {
        "f_endo_1", "f_endo_2", "f_endo_3",
        "f_exoUnk_1", "f_exoUnk_2",
        "f_l4_1", "f_l4_2", "f_l4_3", "f_l4_4",
    }
    # Dropped features must never appear.
    assert "f_dropped_1" not in ids


def test_default_returns_list_of_strings(stage_10_dir: Path) -> None:
    """Downstream callers index pandas with this list, so it must be a
    plain ``list[str]`` (not a pandas Index)."""
    ids = load_active_feature_ids(stage_10_dir)
    assert isinstance(ids, list)
    assert all(isinstance(x, str) for x in ids)


# ============================================================
# Filtered behaviour — single outcome
# ============================================================
def test_filter_endogenous_only(stage_10_dir: Path) -> None:
    ids = load_active_feature_ids(
        stage_10_dir, outcome_filter={"endogenous_evidence"}
    )
    assert set(ids) == {"f_endo_1", "f_endo_2", "f_endo_3"}


def test_filter_exo_or_unknown_only(stage_10_dir: Path) -> None:
    ids = load_active_feature_ids(
        stage_10_dir, outcome_filter={"exo_or_unknown"}
    )
    assert set(ids) == {"f_exoUnk_1", "f_exoUnk_2"}


def test_filter_passthrough_l4_only(stage_10_dir: Path) -> None:
    ids = load_active_feature_ids(
        stage_10_dir, outcome_filter={"passthrough_l4"}
    )
    assert set(ids) == {"f_l4_1", "f_l4_2", "f_l4_3", "f_l4_4"}


# ============================================================
# Filtered behaviour — combinations
# ============================================================
def test_filter_endo_plus_exo_or_unknown(stage_10_dir: Path) -> None:
    """The combination requested in the user's example —
    {endogenous_evidence, exo_or_unknown} drops L4 only."""
    ids = load_active_feature_ids(
        stage_10_dir,
        outcome_filter=("endogenous_evidence", "exo_or_unknown"),
    )
    assert set(ids) == {
        "f_endo_1", "f_endo_2", "f_endo_3", "f_exoUnk_1", "f_exoUnk_2"
    }
    assert not any(x.startswith("f_l4_") for x in ids)


def test_filter_endo_plus_l4(stage_10_dir: Path) -> None:
    """Strictest exogenous control — drops the exo_or_unknown bucket."""
    ids = load_active_feature_ids(
        stage_10_dir,
        outcome_filter=("endogenous_evidence", "passthrough_l4"),
    )
    assert set(ids) == {
        "f_endo_1", "f_endo_2", "f_endo_3",
        "f_l4_1", "f_l4_2", "f_l4_3", "f_l4_4",
    }
    assert not any(x.startswith("f_exoUnk_") for x in ids)


def test_filter_all_three_outcomes_equals_kept_features(stage_10_dir: Path) -> None:
    """All three valid outcomes selected == every kept row. This is the
    'self-documenting' equivalent of None."""
    ids_all = load_active_feature_ids(
        stage_10_dir, outcome_filter=tuple(VALID_OUTCOMES)
    )
    ids_default = load_active_feature_ids(stage_10_dir, outcome_filter=None)
    assert set(ids_all) == set(ids_default)


def test_filter_excludes_dropped_features(stage_10_dir: Path) -> None:
    """Even if 'all_exogenous' were passed it should never resurrect dropped
    features (defence-in-depth — the filter only sees kept=True rows)."""
    ids = load_active_feature_ids(
        stage_10_dir,
        outcome_filter=("endogenous_evidence", "exo_or_unknown", "passthrough_l4"),
    )
    assert "f_dropped_1" not in ids
    assert "f_dropped_2" not in ids


# ============================================================
# Validation
# ============================================================
def test_unknown_outcome_raises(stage_10_dir: Path) -> None:
    with pytest.raises(ValueError, match="Unknown outcome"):
        load_active_feature_ids(
            stage_10_dir, outcome_filter={"not_a_real_outcome"}
        )


def test_unknown_outcome_lists_valid_options_in_error(stage_10_dir: Path) -> None:
    with pytest.raises(ValueError, match="endogenous_evidence"):
        load_active_feature_ids(
            stage_10_dir, outcome_filter={"endogenoous_evidence"}  # typo
        )


def test_typo_all_exogenous_raises(stage_10_dir: Path) -> None:
    """'all_exogenous' is a valid Stage-10 outcome BUT only on dropped rows;
    we explicitly excluded it from VALID_OUTCOMES to prevent confusion."""
    with pytest.raises(ValueError, match="Unknown outcome"):
        load_active_feature_ids(
            stage_10_dir, outcome_filter={"all_exogenous"}
        )


def test_empty_filter_raises(stage_10_dir: Path) -> None:
    """Empty allowlist would select zero features — must fail loudly so the
    multiverse never silently runs on an empty matrix."""
    with pytest.raises(ValueError, match="produced 0 features"):
        load_active_feature_ids(stage_10_dir, outcome_filter=set())


def test_filter_that_matches_nothing_raises(tmp_path: Path) -> None:
    """If the Stage-10 run happens to have zero features in the requested
    bucket, fail loudly rather than feeding the multiverse an empty matrix."""
    metrics = pd.DataFrame(
        [
            ("f_endo_1", "endogenous_evidence", True),
            ("f_endo_2", "endogenous_evidence", True),
        ],
        columns=["feature_id", "outcome", "kept"],
    )
    metrics.to_parquet(tmp_path / "metrics.parquet")
    metrics[["feature_id"]].to_parquet(tmp_path / "kept_features.parquet")

    with pytest.raises(ValueError, match="produced 0 features"):
        load_active_feature_ids(
            tmp_path, outcome_filter={"passthrough_l4"}
        )


# ============================================================
# Public constant
# ============================================================
def test_valid_outcomes_set_is_locked() -> None:
    """The three Stage-10 outcomes that yield kept=True features — locked
    so an accidental rename in origin.py forces a coordinated test update."""
    assert frozenset({
        "endogenous_evidence",
        "exo_or_unknown",
        "passthrough_l4",
    }) == VALID_OUTCOMES
