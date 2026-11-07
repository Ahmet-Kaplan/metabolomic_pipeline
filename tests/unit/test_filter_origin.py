"""Unit tests for Stage 10 — origin filter."""

from __future__ import annotations

import pandas as pd

from biodisc.audit import AuditLogger
from biodisc.external.hmdb_origin import _classify
from biodisc.filters.origin import (
    OriginFilterConfig,
    _aggregate_origin,
    run,
    write_outputs,
)


# ============================================================
# Helpers
# ============================================================
def _make_metrics(rows: list[dict]) -> pd.DataFrame:
    """Build a Stage-9-style metrics DataFrame."""
    cols = ["feature_id", "msi_tier", "outcome"]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return pd.DataFrame(rows)


def _make_ranked(rows: list[dict]) -> pd.DataFrame:
    """Build a Stage-9-style identification_ranked DataFrame."""
    cols = ["feature_id", "compound_id", "description"]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return pd.DataFrame(rows)


def _make_hmdb_origin(rows: list[dict]) -> pd.DataFrame:
    """Build an HMDB-origin-style DataFrame."""
    cols = ["hmdb_id", "origin_class", "is_endogenous", "source_terms"]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return pd.DataFrame(rows)


def _audit() -> AuditLogger:
    return AuditLogger(stage="10_origin", params={"drop_pure_exogenous": True})


# ============================================================
# Classification helper tests
# ============================================================
class TestClassify:
    def test_endogenous_only(self) -> None:
        assert _classify(["Endogenous"]) == "endogenous_only"

    def test_both(self) -> None:
        assert _classify(["Endogenous", "Food"]) == "both"
        assert _classify(["Endogenous", "Drug", "Microbial"]) == "both"

    def test_exogenous_only(self) -> None:
        assert _classify(["Food"]) == "exogenous_only"
        assert _classify(["Drug", "Microbial", "Plant"]) == "exogenous_only"

    def test_unannotated(self) -> None:
        assert _classify([]) == "unannotated"


# ============================================================
# Aggregation tests
# ============================================================
class TestAggregateOrigin:
    def test_endogenous_evidence(self) -> None:
        cands = pd.DataFrame({"origin_class": ["endogenous_only", "exogenous_only"]})
        outcome, n_endo, _n_both, n_exo, _n_unann, _ = _aggregate_origin(cands)
        assert outcome == "endogenous_evidence"
        assert n_endo == 1
        assert n_exo == 1

    def test_both_counts_as_endogenous_evidence(self) -> None:
        cands = pd.DataFrame({"origin_class": ["both", "exogenous_only"]})
        outcome, _n_endo, n_both, _n_exo, _n_unann, _ = _aggregate_origin(cands)
        assert outcome == "endogenous_evidence"
        assert n_both == 1

    def test_all_exogenous(self) -> None:
        cands = pd.DataFrame({"origin_class": ["exogenous_only", "exogenous_only"]})
        outcome, _n_endo, _n_both, n_exo, n_unann, _ = _aggregate_origin(cands)
        assert outcome == "all_exogenous"
        assert n_exo == 2
        assert n_unann == 0

    def test_exo_or_unknown(self) -> None:
        # Mixed: one exo, one unannotated, no endo → conservative keep
        cands = pd.DataFrame({"origin_class": ["exogenous_only", "unannotated"]})
        outcome, n_endo, _n_both, n_exo, n_unann, _ = _aggregate_origin(cands)
        assert outcome == "exo_or_unknown"
        assert n_endo == 0
        assert n_exo == 1
        assert n_unann == 1

    def test_all_unannotated_keeps(self) -> None:
        # No annotations at all → conservative keep (exo_or_unknown bucket)
        cands = pd.DataFrame({"origin_class": ["unannotated", "unannotated"]})
        outcome, _n_endo, _n_both, _n_exo, n_unann, _ = _aggregate_origin(cands)
        assert outcome == "exo_or_unknown"
        assert n_unann == 2


# ============================================================
# End-to-end run() tests
# ============================================================
class TestRun:
    def test_endogenous_evidence_keeps(self) -> None:
        metrics = _make_metrics(
            [{"feature_id": "F1", "msi_tier": 2, "outcome": "l2_unique"}]
        )
        ranked = _make_ranked(
            [{"feature_id": "F1", "compound_id": "HMDB_E1", "description": "endo"}]
        )
        origin = _make_hmdb_origin(
            [
                {
                    "hmdb_id": "HMDB_E1",
                    "origin_class": "endogenous_only",
                    "is_endogenous": True,
                    "source_terms": ["Endogenous"],
                }
            ]
        )
        result = run(
            metrics_in=metrics,
            ranked_identification=ranked,
            hmdb_origin=origin,
            kept_features=["F1"],
            config=OriginFilterConfig(),
            audit=_audit(),
        )
        assert result.n_kept == 1
        assert result.n_dropped == 0
        assert result.n_endogenous_evidence == 1

    def test_all_exogenous_drops(self) -> None:
        metrics = _make_metrics(
            [{"feature_id": "F1", "msi_tier": 2, "outcome": "l2_unique"}]
        )
        ranked = _make_ranked(
            [{"feature_id": "F1", "compound_id": "HMDB_X1", "description": "drug"}]
        )
        origin = _make_hmdb_origin(
            [
                {
                    "hmdb_id": "HMDB_X1",
                    "origin_class": "exogenous_only",
                    "is_endogenous": False,
                    "source_terms": ["Drug"],
                }
            ]
        )
        result = run(
            metrics_in=metrics,
            ranked_identification=ranked,
            hmdb_origin=origin,
            kept_features=["F1"],
            config=OriginFilterConfig(),
            audit=_audit(),
        )
        assert result.n_kept == 0
        assert result.n_dropped == 1
        assert result.n_all_exogenous == 1

    def test_soft_mode_does_not_drop(self) -> None:
        # Same as test_all_exogenous_drops but with drop_pure_exogenous=False
        metrics = _make_metrics(
            [{"feature_id": "F1", "msi_tier": 2, "outcome": "l2_unique"}]
        )
        ranked = _make_ranked(
            [{"feature_id": "F1", "compound_id": "HMDB_X1", "description": "drug"}]
        )
        origin = _make_hmdb_origin(
            [
                {
                    "hmdb_id": "HMDB_X1",
                    "origin_class": "exogenous_only",
                    "is_endogenous": False,
                    "source_terms": ["Drug"],
                }
            ]
        )
        result = run(
            metrics_in=metrics,
            ranked_identification=ranked,
            hmdb_origin=origin,
            kept_features=["F1"],
            config=OriginFilterConfig(drop_pure_exogenous=False),
            audit=_audit(),
        )
        assert result.n_kept == 1
        assert result.n_dropped == 0
        assert result.n_all_exogenous == 1  # still tagged, not dropped

    def test_l4_no_candidate_passes_through(self) -> None:
        metrics = _make_metrics(
            [{"feature_id": "F1", "msi_tier": 4, "outcome": "l4_no_candidate"}]
        )
        ranked = _make_ranked([])
        origin = _make_hmdb_origin([])
        result = run(
            metrics_in=metrics,
            ranked_identification=ranked,
            hmdb_origin=origin,
            kept_features=["F1"],
            config=OriginFilterConfig(),
            audit=_audit(),
        )
        assert result.n_kept == 1
        assert result.n_passthrough_l4 == 1

    def test_l4_no_floor_survivor_passes_through(self) -> None:
        metrics = _make_metrics(
            [{"feature_id": "F1", "msi_tier": 4, "outcome": "l4_no_floor_survivor"}]
        )
        ranked = _make_ranked([])
        origin = _make_hmdb_origin([])
        result = run(
            metrics_in=metrics,
            ranked_identification=ranked,
            hmdb_origin=origin,
            kept_features=["F1"],
            config=OriginFilterConfig(),
            audit=_audit(),
        )
        assert result.n_kept == 1
        assert result.n_passthrough_l4 == 1

    def test_any_endo_candidate_keeps_l3(self) -> None:
        # L3 with mixed candidates: 1 endo, 2 exo → should be KEPT
        metrics = _make_metrics(
            [{"feature_id": "F1", "msi_tier": 3, "outcome": "l3_ambiguous"}]
        )
        ranked = _make_ranked(
            [
                {"feature_id": "F1", "compound_id": "HMDB_E1", "description": "endo"},
                {"feature_id": "F1", "compound_id": "HMDB_X1", "description": "exo1"},
                {"feature_id": "F1", "compound_id": "HMDB_X2", "description": "exo2"},
            ]
        )
        origin = _make_hmdb_origin(
            [
                {
                    "hmdb_id": "HMDB_E1",
                    "origin_class": "endogenous_only",
                    "is_endogenous": True,
                    "source_terms": ["Endogenous"],
                },
                {
                    "hmdb_id": "HMDB_X1",
                    "origin_class": "exogenous_only",
                    "is_endogenous": False,
                    "source_terms": ["Drug"],
                },
                {
                    "hmdb_id": "HMDB_X2",
                    "origin_class": "exogenous_only",
                    "is_endogenous": False,
                    "source_terms": ["Food"],
                },
            ]
        )
        result = run(
            metrics_in=metrics,
            ranked_identification=ranked,
            hmdb_origin=origin,
            kept_features=["F1"],
            config=OriginFilterConfig(),
            audit=_audit(),
        )
        assert result.n_kept == 1
        assert result.n_endogenous_evidence == 1

    def test_mixed_exo_unknown_keeps_conservative(self) -> None:
        # 1 exo (known), 1 candidate not in HMDB origin table (unannotated)
        # → no endo evidence, but we don't drop because we can't be sure
        metrics = _make_metrics(
            [{"feature_id": "F1", "msi_tier": 3, "outcome": "l3_ambiguous"}]
        )
        ranked = _make_ranked(
            [
                {"feature_id": "F1", "compound_id": "HMDB_X1", "description": "exo"},
                {"feature_id": "F1", "compound_id": "HMDB_UNKNOWN", "description": "unknown"},
            ]
        )
        origin = _make_hmdb_origin(
            [
                {
                    "hmdb_id": "HMDB_X1",
                    "origin_class": "exogenous_only",
                    "is_endogenous": False,
                    "source_terms": ["Drug"],
                }
                # HMDB_UNKNOWN not in table → falls to "unannotated" via merge
            ]
        )
        result = run(
            metrics_in=metrics,
            ranked_identification=ranked,
            hmdb_origin=origin,
            kept_features=["F1"],
            config=OriginFilterConfig(),
            audit=_audit(),
        )
        assert result.n_kept == 1
        assert result.n_exo_or_unknown == 1

    def test_partition_exhaustivity(self) -> None:
        """All active features must land in exactly one outcome bucket."""
        metrics = _make_metrics(
            [
                {"feature_id": "F1", "msi_tier": 2, "outcome": "l2_unique"},  # endo
                {"feature_id": "F2", "msi_tier": 2, "outcome": "l2_unique"},  # exo - drop
                {"feature_id": "F3", "msi_tier": 4, "outcome": "l4_no_candidate"},  # passthrough
                {"feature_id": "F4", "msi_tier": 3, "outcome": "l3_ambiguous"},  # exo+unknown
            ]
        )
        ranked = _make_ranked(
            [
                {"feature_id": "F1", "compound_id": "HMDB_E1", "description": "endo"},
                {"feature_id": "F2", "compound_id": "HMDB_X1", "description": "exo"},
                {"feature_id": "F4", "compound_id": "HMDB_X2", "description": "exo"},
                {"feature_id": "F4", "compound_id": "HMDB_UNK", "description": "unknown"},
            ]
        )
        origin = _make_hmdb_origin(
            [
                {
                    "hmdb_id": "HMDB_E1",
                    "origin_class": "endogenous_only",
                    "is_endogenous": True,
                    "source_terms": ["Endogenous"],
                },
                {
                    "hmdb_id": "HMDB_X1",
                    "origin_class": "exogenous_only",
                    "is_endogenous": False,
                    "source_terms": ["Drug"],
                },
                {
                    "hmdb_id": "HMDB_X2",
                    "origin_class": "exogenous_only",
                    "is_endogenous": False,
                    "source_terms": ["Drug"],
                },
            ]
        )
        result = run(
            metrics_in=metrics,
            ranked_identification=ranked,
            hmdb_origin=origin,
            kept_features=["F1", "F2", "F3", "F4"],
            config=OriginFilterConfig(),
            audit=_audit(),
        )
        # Exactly one of each outcome
        assert result.n_endogenous_evidence == 1  # F1
        assert result.n_all_exogenous == 1  # F2
        assert result.n_passthrough_l4 == 1  # F3
        assert result.n_exo_or_unknown == 1  # F4
        assert (
            result.n_endogenous_evidence
            + result.n_all_exogenous
            + result.n_passthrough_l4
            + result.n_exo_or_unknown
        ) == result.n_input == 4


# ============================================================
# write_outputs
# ============================================================
class TestWriteOutputs:
    def test_writes_two_parquets(self, tmp_path) -> None:
        metrics = _make_metrics(
            [{"feature_id": "F1", "msi_tier": 2, "outcome": "l2_unique"}]
        )
        ranked = _make_ranked(
            [{"feature_id": "F1", "compound_id": "HMDB_E1", "description": "endo"}]
        )
        origin = _make_hmdb_origin(
            [
                {
                    "hmdb_id": "HMDB_E1",
                    "origin_class": "endogenous_only",
                    "is_endogenous": True,
                    "source_terms": ["Endogenous"],
                }
            ]
        )
        result = run(
            metrics_in=metrics,
            ranked_identification=ranked,
            hmdb_origin=origin,
            kept_features=["F1"],
            config=OriginFilterConfig(),
            audit=_audit(),
        )
        paths = write_outputs(result, tmp_path, emit_csv_mirror=False)
        assert "kept_features" in paths
        assert "metrics" in paths
        for p in paths.values():
            assert p.exists()
            assert p.suffix == ".parquet"

    def test_metrics_has_headline_column_for_ledger(self, tmp_path) -> None:
        # Ledger registers ("10_origin", "metrics.parquet", "n_exogenous_only")
        metrics = _make_metrics(
            [{"feature_id": "F1", "msi_tier": 2, "outcome": "l2_unique"}]
        )
        ranked = _make_ranked(
            [{"feature_id": "F1", "compound_id": "HMDB_X1", "description": "exo"}]
        )
        origin = _make_hmdb_origin(
            [
                {
                    "hmdb_id": "HMDB_X1",
                    "origin_class": "exogenous_only",
                    "is_endogenous": False,
                    "source_terms": ["Drug"],
                }
            ]
        )
        result = run(
            metrics_in=metrics,
            ranked_identification=ranked,
            hmdb_origin=origin,
            kept_features=["F1"],
            config=OriginFilterConfig(),
            audit=_audit(),
        )
        m = pd.read_parquet(write_outputs(result, tmp_path)["metrics"])
        assert "n_exogenous_only" in m.columns
        assert "kept" in m.columns
