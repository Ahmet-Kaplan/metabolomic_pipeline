"""Phase 8e.7 — multiverse index unit tests.

Tests the four new figure helpers + the end-to-end ``build_multiverse_index``
function against synthetic ``PathRunResult`` / ``ConsensusResult`` objects.
Mirrors the same testing strategy used elsewhere in the multiverse stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biodisc.multiverse.consensus import compute_consensus
from biodisc.multiverse.orchestrator import PathRunResult, PathSpec
from biodisc.multiverse.reporting import build_multiverse_index
from biodisc.multiverse.reporting.figures import (
    consensus_tier_bar,
    jaccard_similarity_heatmap,
    method_drilldown_heatmap,
    specification_curve_bar,
)
from biodisc.multiverse.shortlist import (
    MethodGate,
    ShortlistConfig,
    ShortlistResult,
)


# ============================================================
# Fixture builders
# ============================================================
def _make_synth_shortlist(
    *, all_features: list[str], passed: list[str], path_name: str = "p",
) -> ShortlistResult:
    """Build a tiny ShortlistResult where ``passed`` features are flagged
    by univariate AND PLS-DA only (matches the real-data UP__ pattern)."""
    n = len(all_features)
    table = pd.DataFrame(
        {
            "feature_id": all_features,
            "univariate_q": [0.001 if f in passed else 0.5 for f in all_features],
            "univariate_eta_sq": [0.5 if f in passed else 0.05 for f in all_features],
            "univariate_passed": [f in passed for f in all_features],
            "plsda_vip": [3.0 if f in passed else 0.5 for f in all_features],
            "plsda_stability": [0.95 if f in passed else 0.1 for f in all_features],
            "plsda_perm_p": [0.001 if f in passed else 0.5 for f in all_features],
            "plsda_passed": [f in passed for f in all_features],
            "splsda_selected": [False] * n,
            "splsda_stability": [0.0] * n,
            "splsda_perm_p": [1.0] * n,
            "splsda_passed": [False] * n,
            "rf_gini": [0.001] * n,
            "rf_stability": [0.0] * n,
            "rf_perm_p": [1.0] * n,
            "rf_passed": [False] * n,
            "n_methods_passed": [2 if f in passed else 0 for f in all_features],
            "consensus_tier": ["tier_2" if f in passed else "none" for f in all_features],
            "composite_score": [2.0 if f in passed else 0.0 for f in all_features],
            "passed_shortlist": [f in passed for f in all_features],
        }
    )
    return ShortlistResult(
        table=table,
        n_input_features=n,
        n_shortlisted=len(passed),
        n_per_tier={
            "tier_1": 0, "tier_2": len(passed), "tier_3": 0, "none": n - len(passed),
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


def _make_synth_path_result(
    name: str, all_features: list[str], passed: list[str],
    *, imp: str = "halfmin", drift: str = "qc_rsc",
    norm: str = "pqn", trans: str = "log2",
) -> PathRunResult:
    spec = PathSpec(
        name=name, imputation=imp, drift=drift,
        normalisation=norm, transformation=trans,
    )
    sl = _make_synth_shortlist(all_features=all_features, passed=passed, path_name=name)
    return PathRunResult(
        spec=spec, shortlist=sl, univariate=None, multivariate={},
        runtime_seconds=120.5, output_dir=Path(f"/tmp/{name}"),
        succeeded=True, error=None,
    )


# ============================================================
# Figure helper tests
# ============================================================
class TestSpecificationCurveBar:
    def test_basic(self) -> None:
        fig = specification_curve_bar(
            path_names=["p_a", "p_b", "p_c"],
            n_shortlisted=[10, 5, 20],
        )
        # 1 trace (the bar)
        assert len(fig.data) == 1
        # Bars are sorted ascending
        assert list(fig.data[0].x) == ["p_b", "p_a", "p_c"]
        assert list(fig.data[0].y) == [5, 10, 20]

    def test_stacked_with_tiers(self) -> None:
        fig = specification_curve_bar(
            path_names=["p_a", "p_b"],
            n_shortlisted=[5, 10],
            n_per_tier_per_path=[
                {"tier_1": 1, "tier_2": 2, "tier_3": 2},
                {"tier_1": 3, "tier_2": 4, "tier_3": 3},
            ],
        )
        # 3 stacked traces (tier_1, tier_2, tier_3)
        assert len(fig.data) == 3


class TestJaccardHeatmap:
    def test_diagonal_one(self) -> None:
        fig = jaccard_similarity_heatmap(
            path_names=["a", "b"],
            feature_sets=[{"F1", "F2"}, {"F2", "F3"}],
        )
        z = np.asarray(fig.data[0].z)
        # Diagonal = 1
        assert z[0, 0] == pytest.approx(1.0)
        assert z[1, 1] == pytest.approx(1.0)
        # Off-diagonal: J = |{F2}| / |{F1, F2, F3}| = 1/3
        assert z[0, 1] == pytest.approx(1 / 3, abs=1e-6)
        assert z[1, 0] == pytest.approx(1 / 3, abs=1e-6)

    def test_disjoint_sets_zero_jaccard(self) -> None:
        fig = jaccard_similarity_heatmap(
            path_names=["a", "b"],
            feature_sets=[{"F1"}, {"F2"}],
        )
        z = np.asarray(fig.data[0].z)
        assert z[0, 1] == pytest.approx(0.0)


class TestConsensusTierBar:
    def test_render(self) -> None:
        fig = consensus_tier_bar(
            n_per_tier={"tier_1": 13, "tier_2": 1, "tier_3": 3, "none": 4627},
        )
        # Single bar trace, 4 categories
        assert len(fig.data) == 1
        assert len(fig.data[0].y) == 4
        assert list(fig.data[0].x) == [13, 1, 3, 4627]


class TestMethodDrilldownHeatmap:
    def test_shape(self) -> None:
        n_feat, n_paths, n_methods = 3, 2, 4
        cube = np.zeros((n_feat, n_paths, n_methods), dtype=int)
        cube[0, 0, 0] = 1  # feature 0 flagged by method 0 in path 0
        fig = method_drilldown_heatmap(
            feature_ids=["F0", "F1", "F2"],
            path_names=["p_a", "p_b"],
            method_names=["univariate", "plsda", "splsda", "rf"],
            flag_cube=cube,
        )
        z = np.asarray(fig.data[0].z)
        # Shape: features × (paths * methods)
        assert z.shape == (n_feat, n_paths * n_methods)
        # Cell (0, 0) (= path 0, method 0) is 1
        assert z[0, 0] == 1


# ============================================================
# build_multiverse_index end-to-end test
# ============================================================
class TestBuildMultiverseIndex:
    def test_renders_index_html(self, tmp_path: Path) -> None:
        all_features = [f"F{i:03d}" for i in range(20)]
        results = [
            _make_synth_path_result("path_a", all_features, ["F000", "F001", "F002"], imp="halfmin"),
            _make_synth_path_result("path_b", all_features, ["F000", "F001", "F003"], imp="qrilc"),
            _make_synth_path_result("path_c", all_features, ["F000", "F004"], imp="halfmin", drift="serrf"),
            _make_synth_path_result("path_d", all_features, ["F000", "F001"], imp="halfmin", norm="mstus"),
        ]
        # Drop required per-path subdirs (the index links to them)
        for r in results:
            (tmp_path / r.spec.name).mkdir(parents=True, exist_ok=True)
            (tmp_path / r.spec.name / "index.html").write_text("<html>stub</html>")

        consensus = compute_consensus(results, threshold=0.5)
        index_path = build_multiverse_index(
            path_results=results,
            consensus=consensus,
            output_dir=tmp_path,
            write_png=False,
        )
        assert index_path.exists()
        html = index_path.read_text()
        # Sanity-check key markers (mirrors the real-data spot check)
        assert "Multiverse" in html
        assert "Specification curve" in html
        assert 'href="path_a/index.html"' in html
        assert 'href="path_c/index.html"' in html
        # 4 figures (spec curve + jaccard + tiers + drilldown)
        assert html.count('class="figure-block"') == 4
        # summary.json sidecar
        assert (tmp_path / "summary.json").exists()
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["n_sections"] == 8

    def test_handles_failed_path(self, tmp_path: Path) -> None:
        all_features = ["F0", "F1"]
        ok = _make_synth_path_result("ok", all_features, ["F0"])
        bad = PathRunResult(
            spec=PathSpec("bad", "halfmin", "qc_rsc", "pqn", "log2"),
            shortlist=None, univariate=None, multivariate={},
            runtime_seconds=0.5, output_dir=tmp_path / "bad",
            succeeded=False, error="boom",
        )
        (tmp_path / "ok").mkdir(parents=True, exist_ok=True)
        (tmp_path / "ok" / "index.html").write_text("<html>stub</html>")
        consensus = compute_consensus([ok, bad])
        index_path = build_multiverse_index(
            path_results=[ok, bad], consensus=consensus,
            output_dir=tmp_path, write_png=False,
        )
        html = index_path.read_text()
        # Failed path appears with FAIL marker in the table
        assert "FAIL" in html or "✗" in html
