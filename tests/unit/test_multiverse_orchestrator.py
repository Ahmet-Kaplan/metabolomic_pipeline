"""Phase 10 unit tests — orchestrator + consensus aggregator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biodisc.multiverse.consensus import (
    compute_consensus,
    write_consensus,
)
from biodisc.multiverse.orchestrator import (
    PathRunResult,
    PathSpec,
    cartesian_paths,
    run_multiverse,
    run_path,
)
from biodisc.multiverse.shortlist import (
    MethodGate,
    ShortlistConfig,
    ShortlistResult,
)


# ============================================================
# Synthetic dataset (real-data convention: QC has cell_line=None)
# ============================================================
def _make_real_convention_synth(
    *, n_features: int = 12, n_per_class: int = 3, n_qc: int = 5, n_signal: int = 4,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tiny LC-MS-ish dataset with class signal + QC pool + run order."""
    rng = np.random.default_rng(seed)
    cell_lines = ["MCF7", "MCF10A", "MDAMB231"]
    sample_recs: list[dict] = []
    for cl in cell_lines:
        for r in range(1, n_per_class + 1):
            sample_recs.append(
                {
                    "sample_id": f"BIO_{cl}_{r}",
                    "raw_sample_id": f"raw_BIO_{cl}_{r}",
                    "injection_type": "biological",
                    "cell_line": cl,
                    "replicate": r,
                    "column_order": len(sample_recs),
                }
            )
    for q in range(1, n_qc + 1):
        sample_recs.append(
            {
                "sample_id": f"QC_{q}",
                "raw_sample_id": f"raw_QC_{q}",
                "injection_type": "qc",
                "cell_line": None,
                "replicate": q,
                "column_order": len(sample_recs),
            }
        )
    samples = pd.DataFrame(sample_recs)
    n_samples = len(samples)
    fids = [f"F{i:03d}" for i in range(n_features)]
    matrix = np.zeros((n_features, n_samples), dtype=float)
    cell_arr = samples["cell_line"].fillna("QC").to_numpy()
    for i in range(n_features):
        base = float(rng.lognormal(mean=8.0, sigma=0.5))
        signal_class = cell_lines[i % len(cell_lines)] if i < n_signal else None
        for j in range(n_samples):
            cl = cell_arr[j]
            boost = 5.0 if cl == signal_class else 0.0
            noise = rng.normal(scale=0.05) if cl == "QC" else rng.normal(scale=0.15)
            matrix[i, j] = max(0.001, base * (1.0 + noise) + boost)
    matrix_df = pd.DataFrame(matrix, index=fids, columns=samples["sample_id"])
    features = pd.DataFrame({"feature_id": fids})
    run_order = pd.DataFrame(
        {
            "raw_sample_id": samples["raw_sample_id"].tolist(),
            "position": list(range(1, n_samples + 1)),
        }
    )
    return matrix_df, samples, features, run_order


# ============================================================
# PathSpec
# ============================================================
class TestPathSpec:
    def test_chain_property(self) -> None:
        spec = PathSpec(
            name="t", imputation="halfmin", drift="qc_rsc",
            normalisation="pqn", transformation="log2",
        )
        assert spec.chain == ["halfmin", "qc_rsc", "pqn", "log2", "pareto"]

    def test_frozen_dataclass(self) -> None:
        from dataclasses import FrozenInstanceError

        spec = PathSpec(
            name="t", imputation="halfmin", drift="qc_rsc",
            normalisation="pqn", transformation="log2",
        )
        with pytest.raises(FrozenInstanceError):
            spec.imputation = "qrilc"  # type: ignore[misc]


# ============================================================
# cartesian_paths
# ============================================================
class TestCartesianPaths:
    def test_default_size(self) -> None:
        paths = cartesian_paths()
        # 4 imputers × 2 drift × 3 norm × 2 transform × 1 scaling = 48
        assert len(paths) == 48
        # All names unique
        names = [p.name for p in paths]
        assert len(set(names)) == 48

    def test_focused_subset(self) -> None:
        paths = cartesian_paths(
            imputation=["halfmin"],
            drift=["qc_rsc"],
            normalisation=["pqn"],
            transformation=["log2"],
        )
        assert len(paths) == 1
        assert paths[0].name == "halfmin_qc_rsc_pqn_log2_pareto"

    def test_method_choice_mix(self) -> None:
        paths = cartesian_paths(
            imputation=["halfmin", "qrilc"],
            drift=["qc_rsc"],
            normalisation=["pqn", "mstus"],
            transformation=["log2"],
        )
        assert len(paths) == 4


# ============================================================
# run_path (single path on synthetic)
# ============================================================
class TestRunPath:
    def test_path_runs_end_to_end(self, tmp_path: Path) -> None:
        matrix, samples, features, run_order = _make_real_convention_synth(
            n_features=10, n_per_class=3, n_qc=4, n_signal=3,
        )
        spec = PathSpec(
            name="halfmin_qc_rsc_pqn_log2_pareto",
            imputation="halfmin",
            drift="qc_rsc",
            normalisation="pqn",
            transformation="log2",
        )
        result = run_path(
            spec,
            matrix=matrix, samples=samples, features=features, run_order=run_order,
            output_dir=tmp_path,
            n_bootstrap=4, n_permutation=10, write_report=False,
        )
        assert result.succeeded, f"path failed: {result.error}"
        assert result.shortlist is not None
        assert result.univariate is not None
        assert set(result.multivariate.keys()) == {"plsda", "splsda", "rf"}
        # Shortlist parquet was written
        assert (tmp_path / "shortlist.parquet").exists()

    def test_failure_captured(self, tmp_path: Path) -> None:
        """An invalid method name should be caught and surfaced cleanly."""
        matrix, samples, features, run_order = _make_real_convention_synth()
        spec = PathSpec(
            name="invalid", imputation="not_a_real_method", drift="qc_rsc",
            normalisation="pqn", transformation="log2",
        )
        result = run_path(
            spec,
            matrix=matrix, samples=samples, features=features, run_order=run_order,
            output_dir=tmp_path,
            n_bootstrap=4, n_permutation=10, write_report=False,
        )
        assert not result.succeeded
        assert result.error is not None
        assert "not_a_real_method" in result.error


# ============================================================
# run_multiverse
# ============================================================
class TestRunMultiverse:
    def test_two_paths_run_sequentially(self, tmp_path: Path) -> None:
        matrix, samples, features, run_order = _make_real_convention_synth(
            n_features=8, n_per_class=3, n_qc=5, n_signal=2,
        )
        paths = [
            PathSpec(
                name="halfmin_qc_rsc_pqn_log2",
                imputation="halfmin", drift="qc_rsc",
                normalisation="pqn", transformation="log2",
            ),
            PathSpec(
                name="halfmin_qc_rsc_mstus_log2",
                imputation="halfmin", drift="qc_rsc",
                normalisation="mstus", transformation="log2",
            ),
        ]
        results = run_multiverse(
            paths,
            matrix=matrix, samples=samples, features=features, run_order=run_order,
            output_dir=tmp_path,
            n_bootstrap=4, n_permutation=10, write_report=False,
        )
        assert len(results) == 2
        # Both succeed
        assert all(r.succeeded for r in results)
        # Per-path subdirs created
        for spec in paths:
            assert (tmp_path / spec.name / "shortlist.parquet").exists()

    def test_failure_in_one_path_doesnt_stop_others(self, tmp_path: Path) -> None:
        matrix, samples, features, run_order = _make_real_convention_synth()
        paths = [
            PathSpec(
                name="ok_path",
                imputation="halfmin", drift="qc_rsc",
                normalisation="pqn", transformation="log2",
            ),
            PathSpec(
                name="bad_path",
                imputation="bogus", drift="qc_rsc",
                normalisation="pqn", transformation="log2",
            ),
            PathSpec(
                name="ok_path_2",
                imputation="halfmin", drift="qc_rsc",
                normalisation="mstus", transformation="log2",
            ),
        ]
        results = run_multiverse(
            paths,
            matrix=matrix, samples=samples, features=features, run_order=run_order,
            output_dir=tmp_path,
            n_bootstrap=3, n_permutation=8, write_report=False,
        )
        assert results[0].succeeded
        assert not results[1].succeeded
        assert results[2].succeeded


# ============================================================
# Consensus aggregator
# ============================================================
def _make_fake_path_result(
    name: str, flagged_features: list[str], all_features: list[str],
) -> PathRunResult:
    """Build a minimal PathRunResult with a hand-crafted shortlist."""
    n = len(all_features)
    table = pd.DataFrame(
        {
            "feature_id": all_features,
            "univariate_q": [0.001 if f in flagged_features else 0.5 for f in all_features],
            "univariate_eta_sq": [0.5 if f in flagged_features else 0.05 for f in all_features],
            "univariate_passed": [f in flagged_features for f in all_features],
            "plsda_vip": [3.0 if f in flagged_features else 0.5 for f in all_features],
            "plsda_stability": [0.95 if f in flagged_features else 0.1 for f in all_features],
            "plsda_perm_p": [0.001 if f in flagged_features else 0.5 for f in all_features],
            "plsda_passed": [f in flagged_features for f in all_features],
            "splsda_selected": [False] * n,
            "splsda_stability": [0.0] * n,
            "splsda_perm_p": [1.0] * n,
            "splsda_passed": [False] * n,
            "rf_gini": [0.001] * n,
            "rf_stability": [0.0] * n,
            "rf_perm_p": [1.0] * n,
            "rf_passed": [False] * n,
            "n_methods_passed": [2 if f in flagged_features else 0 for f in all_features],
            "consensus_tier": [
                "tier_2" if f in flagged_features else "none" for f in all_features
            ],
            "composite_score": [2.0 if f in flagged_features else 0.0 for f in all_features],
            "passed_shortlist": [f in flagged_features for f in all_features],
        }
    )
    sl = ShortlistResult(
        table=table,
        n_input_features=len(all_features),
        n_shortlisted=len(flagged_features),
        n_per_tier={
            "tier_1": 0,
            "tier_2": len(flagged_features),
            "tier_3": 0,
            "none": n - len(flagged_features),
        },
        config=ShortlistConfig(
            method_gates={
                "univariate": MethodGate("univariate"),
                "plsda": MethodGate("plsda"),
            },
            combination="k_of_n",
            k=2,
        ),
        runtime_seconds=0.1,
    )
    spec = PathSpec(
        name=name,
        imputation="halfmin",
        drift="qc_rsc",
        normalisation="pqn",
        transformation="log2",
    )
    return PathRunResult(
        spec=spec,
        shortlist=sl,
        univariate=None,
        multivariate={},
        runtime_seconds=0.1,
        output_dir=Path("/tmp/fake"),
        succeeded=True,
        error=None,
    )


class TestConsensus:
    def test_basic_consensus(self, tmp_path: Path) -> None:
        all_features = [f"F{i:03d}" for i in range(10)]
        # 4 paths, 3 features (F000-F002) flagged in 4/4, 2 features (F003-F004) in 2/4
        path_a = _make_fake_path_result("a", ["F000", "F001", "F002", "F003"], all_features)
        path_b = _make_fake_path_result("b", ["F000", "F001", "F002", "F003", "F004"], all_features)
        path_c = _make_fake_path_result("c", ["F000", "F001", "F002"], all_features)
        path_d = _make_fake_path_result("d", ["F000", "F001", "F002", "F004"], all_features)

        res = compute_consensus([path_a, path_b, path_c, path_d], threshold=0.5)
        assert res.n_paths_total == 4

        rows = res.table.set_index("feature_id")
        # F000-F002 flagged in 4/4 (1.0) → tier_1
        for f in ("F000", "F001", "F002"):
            assert rows.loc[f, "n_paths_flagging"] == 4
            assert rows.loc[f, "frac_paths_flagging"] == 1.0
            assert rows.loc[f, "consensus_tier"] == "tier_1"
            assert rows.loc[f, "passed_consensus"]
        # F003 in 2/4 (0.5) → tier_2 (≥0.5 < 0.75) and passed (>=0.5)
        assert rows.loc["F003", "n_paths_flagging"] == 2
        assert rows.loc["F003", "frac_paths_flagging"] == 0.5
        assert rows.loc["F003", "consensus_tier"] == "tier_2"
        assert rows.loc["F003", "passed_consensus"]
        # F004 in 2/4 (0.5) → tier_2 (passed)
        assert rows.loc["F004", "n_paths_flagging"] == 2
        # F005-F009 not flagged → tier="none", didn't pass
        for f in ("F005", "F006", "F007", "F008", "F009"):
            assert rows.loc[f, "n_paths_flagging"] == 0
            assert rows.loc[f, "consensus_tier"] == "none"
            assert not rows.loc[f, "passed_consensus"]

    def test_threshold_strictness(self) -> None:
        all_features = ["F0", "F1"]
        # F0 in 1/3 (0.33), F1 in 2/3 (0.67)
        a = _make_fake_path_result("a", ["F0", "F1"], all_features)
        b = _make_fake_path_result("b", ["F1"], all_features)
        c = _make_fake_path_result("c", [], all_features)

        # threshold = 0.5
        res_05 = compute_consensus([a, b, c], threshold=0.5)
        rows = res_05.table.set_index("feature_id")
        assert not rows.loc["F0", "passed_consensus"]  # 0.33 < 0.5
        assert rows.loc["F1", "passed_consensus"]  # 0.67 >= 0.5

        # threshold = 0.75 (stricter)
        res_75 = compute_consensus([a, b, c], threshold=0.75)
        rows = res_75.table.set_index("feature_id")
        assert not rows.loc["F1", "passed_consensus"]  # 0.67 < 0.75

    def test_per_path_columns_present(self) -> None:
        all_features = ["F0", "F1"]
        results = [
            _make_fake_path_result("path_x", ["F0"], all_features),
            _make_fake_path_result("path_y", ["F0", "F1"], all_features),
        ]
        res = compute_consensus(results)
        cols = set(res.table.columns)
        assert "per_path_flag_path_x" in cols
        assert "per_path_flag_path_y" in cols

    def test_method_drilldown_columns(self) -> None:
        all_features = ["F0", "F1"]
        results = [_make_fake_path_result("a", ["F0"], all_features)]
        res = compute_consensus(results, method_drilldown=True)
        cols = set(res.table.columns)
        for axis in ("univariate", "plsda", "splsda", "rf"):
            assert f"n_paths_{axis}" in cols

    def test_failed_paths_excluded_by_default(self) -> None:
        all_features = ["F0", "F1"]
        ok = _make_fake_path_result("ok", ["F0"], all_features)
        # Forge a failed result
        bad_spec = PathSpec(
            name="bad", imputation="halfmin", drift="qc_rsc",
            normalisation="pqn", transformation="log2",
        )
        bad = PathRunResult(
            spec=bad_spec, shortlist=None, univariate=None, multivariate={},
            runtime_seconds=0.0, output_dir=Path("/tmp/x"),
            succeeded=False, error="boom",
        )
        res = compute_consensus([ok, bad])
        # Only ok counted (denominator = 1)
        assert res.n_paths_total == 1
        rows = res.table.set_index("feature_id")
        assert rows.loc["F0", "n_paths_flagging"] == 1
        assert rows.loc["F0", "frac_paths_flagging"] == 1.0

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            compute_consensus([], threshold=1.5)

    def test_no_paths_raises(self) -> None:
        with pytest.raises(ValueError, match="No succeeded paths"):
            compute_consensus([])


class TestWriteConsensus:
    def test_write_creates_files(self, tmp_path: Path) -> None:
        all_features = ["F0", "F1"]
        results = [
            _make_fake_path_result("a", ["F0"], all_features),
            _make_fake_path_result("b", ["F0", "F1"], all_features),
        ]
        res = compute_consensus(results)
        paths = write_consensus(res, tmp_path)
        for key in ("parquet", "csv", "passed_parquet", "config"):
            assert paths[key].exists()
        # Round-trip parquet
        df = pd.read_parquet(paths["parquet"])
        assert "feature_id" in df.columns
        assert "passed_consensus" in df.columns
        # Config sidecar
        import json
        cfg = json.loads(paths["config"].read_text())
        assert cfg["threshold"] == 0.5
        assert cfg["n_paths_total"] == 2
        assert cfg["path_names"] == ["a", "b"]
