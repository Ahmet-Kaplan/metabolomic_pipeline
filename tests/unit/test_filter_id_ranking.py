"""Unit tests for Stage 9 — identification candidate ranking."""

from __future__ import annotations

import pandas as pd
import pytest

from biodisc.audit import AuditLogger
from biodisc.filters.id_ranking import (
    IdRankingConfig,
    _rank_one_compound,
    run,
    write_outputs,
)


# ============================================================
# Helpers
# ============================================================
def _make_ident(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic identification_long DataFrame with sensible defaults."""
    cols = [
        "feature_id",
        "candidate_idx",
        "compound_id",
        "score",
        "fragmentation_score",
        "mass_error_ppm",
        "description",
        "formula",
        "adducts",
    ]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                "feature_id": r["feature_id"],
                "candidate_idx": r.get("candidate_idx", i),
                "compound_id": r.get("compound_id", f"HMDB{i:05d}"),
                "score": r.get("score"),
                "fragmentation_score": r.get("frag"),
                "mass_error_ppm": r.get("ppm"),
                "description": r.get("name", f"Compound_{i}"),
                "formula": r.get("formula", "C6H12O6"),
                "adducts": r.get("adducts", "M+H"),
            }
        )
    return pd.DataFrame(out)


def _audit() -> AuditLogger:
    return AuditLogger(stage="09_id_ranking", params={"min_score": 30.0})


# ============================================================
# _rank_one_compound — pure cascade behaviour
# ============================================================
class TestRankOneCompound:
    def test_empty_returns_no_floor_survivor(self) -> None:
        empty = _make_ident([])
        out, label = _rank_one_compound(empty)
        assert out.empty
        assert label == "no_floor_survivor"

    def test_single_returns_single_after_floor(self) -> None:
        df = _make_ident([{"feature_id": "F1", "score": 50, "ppm": 2.0}])
        out, label = _rank_one_compound(df)
        assert len(out) == 1
        assert label == "single_after_floor"

    def test_resolves_at_score(self) -> None:
        df = _make_ident(
            [
                {"feature_id": "F1", "score": 55, "ppm": 1.0},
                {"feature_id": "F1", "score": 40, "ppm": 0.5},
                {"feature_id": "F1", "score": 35, "ppm": 0.1},
            ]
        )
        out, label = _rank_one_compound(df)
        assert len(out) == 1
        assert label == "score"
        assert out.iloc[0]["score"] == 55

    def test_resolves_at_frag(self) -> None:
        # Tied at Score 50 → frag breaks the tie
        df = _make_ident(
            [
                {"feature_id": "F1", "score": 50, "frag": 38, "ppm": 2.0},
                {"feature_id": "F1", "score": 50, "frag": 42, "ppm": 2.5},
                {"feature_id": "F1", "score": 50, "frag": 40, "ppm": 1.0},
            ]
        )
        out, label = _rank_one_compound(df)
        assert len(out) == 1
        assert label == "frag"
        assert out.iloc[0]["fragmentation_score"] == 42

    def test_resolves_at_ppm_when_no_frag(self) -> None:
        # Tied at Score 50, no frag scores → falls through to ppm tiebreak
        df = _make_ident(
            [
                {"feature_id": "F1", "score": 50, "frag": None, "ppm": 2.5},
                {"feature_id": "F1", "score": 50, "frag": None, "ppm": -1.0},
                {"feature_id": "F1", "score": 50, "frag": None, "ppm": 4.5},
            ]
        )
        out, label = _rank_one_compound(df)
        assert len(out) == 1
        assert label == "ppm"
        assert abs(out.iloc[0]["mass_error_ppm"]) == pytest.approx(1.0)

    def test_resolves_at_ppm_after_frag_tie(self) -> None:
        # Tied at Score AND Frag → ppm breaks it
        df = _make_ident(
            [
                {"feature_id": "F1", "score": 50, "frag": 40, "ppm": 4.0},
                {"feature_id": "F1", "score": 50, "frag": 40, "ppm": 1.5},
            ]
        )
        out, label = _rank_one_compound(df)
        assert len(out) == 1
        assert label == "ppm"
        assert out.iloc[0]["mass_error_ppm"] == 1.5

    def test_remains_tied_after_all_three_steps(self) -> None:
        df = _make_ident(
            [
                {"feature_id": "F1", "score": 50, "frag": 40, "ppm": 2.0},
                {"feature_id": "F1", "score": 50, "frag": 40, "ppm": -2.0},
            ]
        )
        out, label = _rank_one_compound(df)
        assert len(out) == 2
        assert label == "tied"

    def test_frag_skipped_when_none_have_frag(self) -> None:
        # No candidates have frag → Step 2 skipped, Step 3 resolves
        df = _make_ident(
            [
                {"feature_id": "F1", "score": 50, "frag": None, "ppm": 1.0},
                {"feature_id": "F1", "score": 50, "frag": None, "ppm": 3.0},
            ]
        )
        out, label = _rank_one_compound(df)
        assert len(out) == 1
        assert label == "ppm"
        assert out.iloc[0]["mass_error_ppm"] == 1.0

    def test_frag_only_keeps_those_with_frag_when_some_have_it(self) -> None:
        # Mix of frag-having and no-frag at the SAME max Score —
        # candidates without frag should be dropped at Step 2 because
        # MS/MS evidence is categorically stronger than no MS/MS.
        df = _make_ident(
            [
                {"feature_id": "F1", "score": 50, "frag": 40, "ppm": 5.0},
                {"feature_id": "F1", "score": 50, "frag": None, "ppm": 1.0},
            ]
        )
        out, label = _rank_one_compound(df)
        assert len(out) == 1
        assert label == "frag"
        assert out.iloc[0]["fragmentation_score"] == 40


# ============================================================
# run() — end-to-end behaviour with quality floor + summaries + audit
# ============================================================
class TestRun:
    def test_quality_floor_drops_low_score(self) -> None:
        # All candidates fail the floor (Score < 30 OR |ppm| > 5) → L4
        ident = _make_ident(
            [
                {"feature_id": "F1", "score": 18, "ppm": 2.0},
                {"feature_id": "F1", "score": 12, "ppm": 4.0},
            ]
        )
        result = run(
            identification=ident,
            kept_features=["F1"],
            config=IdRankingConfig(),
            audit=_audit(),
        )
        assert result.n_msi_l4_no_floor_survivor == 1
        assert result.n_msi_l2 == 0
        assert result.n_msi_l3 == 0
        # No surviving candidates → ranked_identification is empty
        assert result.ranked_identification.empty

    def test_quality_floor_drops_high_ppm(self) -> None:
        # Score is fine but |ppm| exceeds floor
        ident = _make_ident(
            [
                {"feature_id": "F1", "score": 50, "ppm": 8.0},
                {"feature_id": "F1", "score": 45, "ppm": -7.0},
            ]
        )
        result = run(
            identification=ident,
            kept_features=["F1"],
            config=IdRankingConfig(),
            audit=_audit(),
        )
        assert result.n_msi_l4_no_floor_survivor == 1

    def test_l2_unique_after_cascade(self) -> None:
        ident = _make_ident(
            [
                {"feature_id": "F1", "score": 55, "ppm": 1.0},
                {"feature_id": "F1", "score": 40, "ppm": 0.5},
            ]
        )
        result = run(
            identification=ident,
            kept_features=["F1"],
            config=IdRankingConfig(),
            audit=_audit(),
        )
        assert result.n_msi_l2 == 1
        assert result.n_msi_l3 == 0
        assert result.n_resolved_at_score == 1
        assert len(result.ranked_identification) == 1
        assert result.metrics.iloc[0]["top_score"] == 55

    def test_l3_ambiguous_when_truly_tied(self) -> None:
        ident = _make_ident(
            [
                {"feature_id": "F1", "score": 50, "frag": 40, "ppm": 2.0},
                {"feature_id": "F1", "score": 50, "frag": 40, "ppm": -2.0},
            ]
        )
        result = run(
            identification=ident,
            kept_features=["F1"],
            config=IdRankingConfig(),
            audit=_audit(),
        )
        assert result.n_msi_l3 == 1
        assert result.n_msi_l2 == 0
        assert result.n_still_tied == 1
        assert len(result.ranked_identification) == 2

    def test_no_candidate_features_become_l4(self) -> None:
        ident = _make_ident([{"feature_id": "F1", "score": 50, "ppm": 1.0}])
        result = run(
            identification=ident,
            kept_features=["F1", "F2", "F3"],  # F2, F3 have no Progenesis candidate
            config=IdRankingConfig(),
            audit=_audit(),
        )
        assert result.n_input == 3
        assert result.n_msi_l2 == 1
        assert result.n_msi_l4_no_candidate == 2
        # The L2 row + 2 L4 rows = 3 metrics rows total
        assert len(result.metrics) == 3

    def test_kept_features_are_pass_through(self) -> None:
        # Stage 9 never drops — kept_features should == input
        ident = _make_ident(
            [
                {"feature_id": "F1", "score": 55, "ppm": 1.0},
                {"feature_id": "F2", "score": 18, "ppm": 1.0},  # fails floor
            ]
        )
        result = run(
            identification=ident,
            kept_features=["F1", "F2", "F3"],
            config=IdRankingConfig(),
            audit=_audit(),
        )
        assert sorted(result.kept_features["feature_id"].tolist()) == ["F1", "F2", "F3"]

    def test_audit_records_one_row_per_feature(self) -> None:
        ident = _make_ident(
            [
                {"feature_id": "F1", "score": 55, "ppm": 1.0},
                {"feature_id": "F2", "score": 12, "ppm": 1.0},  # fails floor
            ]
        )
        audit = _audit()
        run(
            identification=ident,
            kept_features=["F1", "F2", "F3"],
            config=IdRankingConfig(),
            audit=audit,
        )
        df = audit.to_frame()
        assert len(df) == 3
        assert set(df["feature_id"]) == {"F1", "F2", "F3"}
        # All actions are kept (annotation-only stage)
        assert (df["action"] == "kept").all()

    def test_partition_exhaustivity(self) -> None:
        """Every active feature lands in exactly one MSI bucket."""
        ident = _make_ident(
            [
                # F1 → L2
                {"feature_id": "F1", "score": 55, "ppm": 1.0},
                # F2 → L3 (truly tied)
                {"feature_id": "F2", "score": 50, "frag": 40, "ppm": 2.0},
                {"feature_id": "F2", "score": 50, "frag": 40, "ppm": -2.0},
                # F3 → L4 no_floor_survivor (Score too low)
                {"feature_id": "F3", "score": 18, "ppm": 1.0},
            ]
        )
        # F4 → L4 no_candidate (no rows in identification)
        result = run(
            identification=ident,
            kept_features=["F1", "F2", "F3", "F4"],
            config=IdRankingConfig(),
            audit=_audit(),
        )
        total = (
            result.n_msi_l2
            + result.n_msi_l3
            + result.n_msi_l4_no_candidate
            + result.n_msi_l4_no_floor_survivor
        )
        assert total == result.n_input == 4
        assert result.n_msi_l2 == 1
        assert result.n_msi_l3 == 1
        assert result.n_msi_l4_no_floor_survivor == 1
        assert result.n_msi_l4_no_candidate == 1

    def test_msms_flag_set_when_any_candidate_has_frag(self) -> None:
        ident = _make_ident(
            [
                {"feature_id": "F1", "score": 50, "frag": 35, "ppm": 1.0},
                {"feature_id": "F1", "score": 45, "frag": None, "ppm": 1.5},
                {"feature_id": "F2", "score": 50, "frag": None, "ppm": 1.0},
            ]
        )
        result = run(
            identification=ident,
            kept_features=["F1", "F2"],
            config=IdRankingConfig(),
            audit=_audit(),
        )
        f1 = result.metrics[result.metrics["feature_id"] == "F1"].iloc[0]
        f2 = result.metrics[result.metrics["feature_id"] == "F2"].iloc[0]
        assert bool(f1["had_msms"]) is True
        assert bool(f2["had_msms"]) is False


# ============================================================
# write_outputs — file structure
# ============================================================
class TestWriteOutputs:
    def test_writes_three_parquets(self, tmp_path) -> None:
        ident = _make_ident([{"feature_id": "F1", "score": 50, "ppm": 1.0}])
        result = run(
            identification=ident,
            kept_features=["F1"],
            config=IdRankingConfig(),
            audit=_audit(),
        )
        paths = write_outputs(result, tmp_path, emit_csv_mirror=False)
        assert "kept_features" in paths
        assert "identification_ranked" in paths
        assert "metrics" in paths
        for p in paths.values():
            assert p.exists()
            assert p.suffix == ".parquet"

    def test_csv_mirror_emitted_when_requested(self, tmp_path) -> None:
        ident = _make_ident([{"feature_id": "F1", "score": 50, "ppm": 1.0}])
        result = run(
            identification=ident,
            kept_features=["F1"],
            config=IdRankingConfig(),
            audit=_audit(),
        )
        write_outputs(result, tmp_path, emit_csv_mirror=True)
        assert (tmp_path / "kept_features.csv").exists()
        assert (tmp_path / "identification_ranked.csv").exists()
        assert (tmp_path / "metrics.csv").exists()

    def test_metrics_has_headline_column_for_ledger(self, tmp_path) -> None:
        # The ledger registers ("09_id_ranking", "metrics.parquet", "n_candidates_final")
        # — the headline column must exist + the kept column must exist.
        ident = _make_ident([{"feature_id": "F1", "score": 50, "ppm": 1.0}])
        result = run(
            identification=ident,
            kept_features=["F1"],
            config=IdRankingConfig(),
            audit=_audit(),
        )
        metrics = pd.read_parquet(write_outputs(result, tmp_path)["metrics"])
        assert "n_candidates_final" in metrics.columns
        assert "kept" in metrics.columns
        assert metrics["kept"].all()  # pass-through stage
