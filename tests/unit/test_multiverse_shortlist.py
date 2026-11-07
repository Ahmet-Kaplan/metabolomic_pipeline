"""Phase 9 unit tests — shortlist combination rules + per-method gating."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biodisc.multiverse.multivariate_utils import MultivariateResult
from biodisc.multiverse.shortlist import (
    MethodGate,
    ShortlistConfig,
    compute_shortlist,
    default_config,
    write_shortlist,
)
from biodisc.multiverse.stats_univariate import UnivariateResult


# ============================================================
# Fixture builders
# ============================================================
def _make_univariate_result(n: int = 12, n_signal: int = 4) -> UnivariateResult:
    """First ``n_signal`` features pass; remaining fail."""
    feature_ids = [f"F{i:03d}" for i in range(n)]
    q_values = np.array(
        [0.001] * n_signal + [0.5] * (n - n_signal),
        dtype=float,
    )
    eta = np.array(
        [0.5] * n_signal + [0.05] * (n - n_signal),
        dtype=float,
    )
    table = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "f_statistic": [10.0] * n,
            "p_value": q_values,
            "q_value": q_values,
            "eta_squared": eta,
            "n_groups": [3] * n,
        }
    )
    return UnivariateResult(
        table=table,
        n_features=n,
        n_significant_at_q05=int((q_values < 0.05).sum()),
        n_groups=3,
        group_names=["A", "B", "C"],
        fdr_method="bh",
    )


def _make_multivariate_result(
    method: str, n: int = 12, n_signal: int = 4, *, signal_features: list[int] | None = None,
) -> MultivariateResult:
    """Build a MultivariateResult where specified features pass the default gate."""
    feature_ids = [f"F{i:03d}" for i in range(n)]
    if signal_features is None:
        signal_features = list(range(n_signal))

    base = pd.DataFrame({"feature_id": feature_ids})
    if method in {"plsda", "splsda"}:
        base["vip"] = [3.0 if i in signal_features else 0.5 for i in range(n)]
        base["abs_beta_max"] = [0.5] * n
    if method == "splsda":
        base["selected"] = [i in signal_features for i in range(n)]
    if method == "rf":
        base["gini_importance"] = [
            0.05 if i in signal_features else 0.001 for i in range(n)
        ]
    base["perm_importance"] = [0.5 if i in signal_features else 0.0 for i in range(n)]
    base["stability_score"] = [0.95 if i in signal_features else 0.1 for i in range(n)]
    base["perm_p_value"] = [0.001 if i in signal_features else 0.5 for i in range(n)]

    return MultivariateResult(
        method_name=method,
        per_feature=base,
        params_used={},
        runtime_seconds=0.0,
        n_bootstrap=1,
        n_permutation=1,
    )


# ============================================================
# Default behaviour
# ============================================================
class TestDefaultConfig:
    def test_default_returns_k_of_n_with_4_methods(self) -> None:
        cfg = default_config()
        assert cfg.combination == "k_of_n"
        assert cfg.k == 2
        assert set(cfg.method_gates.keys()) == {"univariate", "plsda", "splsda", "rf"}

    def test_default_thresholds_are_textbook(self) -> None:
        cfg = default_config()
        assert cfg.method_gates["univariate"].rules["q_threshold"] == 0.05
        assert cfg.method_gates["univariate"].rules["eta_threshold"] == 0.14
        assert cfg.method_gates["plsda"].rules["vip_threshold"] == 1.0
        assert cfg.method_gates["plsda"].rules["stability_threshold"] == 0.8


# ============================================================
# Combination rules
# ============================================================
class TestCombinationRules:
    def test_and_rule_requires_all_methods(self) -> None:
        # All 4 methods agree on F000-F003 only
        univ = _make_univariate_result(n=12, n_signal=4)
        multi = {
            m: _make_multivariate_result(m, n=12, n_signal=4)
            for m in ("plsda", "splsda", "rf")
        }
        cfg = ShortlistConfig(
            method_gates=default_config().method_gates,
            combination="and",
        )
        res = compute_shortlist(univ, multi, cfg)
        assert res.n_shortlisted == 4
        assert set(res.table[res.table["passed_shortlist"]]["feature_id"]) == {
            f"F{i:03d}" for i in range(4)
        }

    def test_or_rule_accepts_any_method(self) -> None:
        # univariate flags F000-F003; PLS-DA flags F004-F007 (non-overlapping)
        univ = _make_univariate_result(n=12, n_signal=4)
        plsda = _make_multivariate_result(
            "plsda", n=12, signal_features=[4, 5, 6, 7]
        )
        # sPLS-DA + RF empty (no signal features → no passes)
        splsda = _make_multivariate_result("splsda", n=12, signal_features=[])
        rf = _make_multivariate_result("rf", n=12, signal_features=[])
        cfg = ShortlistConfig(
            method_gates=default_config().method_gates,
            combination="or",
        )
        res = compute_shortlist(
            univ, {"plsda": plsda, "splsda": splsda, "rf": rf}, cfg
        )
        # Union: F000-F003 (univariate) ∪ F004-F007 (PLS-DA) = 8 features
        assert res.n_shortlisted == 8

    def test_k_of_n_with_k_2_two_method_overlap(self) -> None:
        """k=2 of 4: feature must pass at least 2 methods."""
        # Univariate flags F000-F005; PLS-DA flags F003-F008.
        # Overlap = F003-F005 (3 features) — these alone pass k=2.
        univ = _make_univariate_result(n=12, n_signal=6)
        plsda = _make_multivariate_result(
            "plsda", n=12, signal_features=[3, 4, 5, 6, 7, 8]
        )
        splsda = _make_multivariate_result("splsda", n=12, signal_features=[])
        rf = _make_multivariate_result("rf", n=12, signal_features=[])
        cfg = ShortlistConfig(
            method_gates=default_config().method_gates,
            combination="k_of_n",
            k=2,
        )
        res = compute_shortlist(
            univ, {"plsda": plsda, "splsda": splsda, "rf": rf}, cfg
        )
        # Features that passed both univariate AND PLS-DA = F003, F004, F005
        passed = set(res.table[res.table["passed_shortlist"]]["feature_id"])
        assert passed == {"F003", "F004", "F005"}, f"got {passed}"

    def test_k_of_n_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be in"):
            ShortlistConfig(
                method_gates={"univariate": MethodGate("univariate", {})},
                combination="k_of_n",
                k=5,
            )

    def test_invalid_combination_raises(self) -> None:
        with pytest.raises(ValueError, match="combination must be"):
            ShortlistConfig(
                method_gates={"univariate": MethodGate("univariate", {})},
                combination="xor",
            )


# ============================================================
# Consensus tiers
# ============================================================
class TestConsensusTiers:
    def test_tiers_assigned_correctly(self) -> None:
        # Build features with 0/1/2/3/4 method passes by construction
        n = 5
        univ = _make_univariate_result(n=n, n_signal=4)  # F000-F003 pass
        plsda = _make_multivariate_result("plsda", n=n, signal_features=[0, 1, 2])
        splsda = _make_multivariate_result("splsda", n=n, signal_features=[0, 1])
        rf = _make_multivariate_result("rf", n=n, signal_features=[0])

        res = compute_shortlist(
            univ, {"plsda": plsda, "splsda": splsda, "rf": rf}, default_config()
        )
        rows = res.table.set_index("feature_id")
        # F000: all 4 → tier_1; F001: 3 → tier_1; F002: 2 → tier_2;
        # F003: 1 → tier_3; F004: 0 → none
        assert rows.loc["F000", "consensus_tier"] == "tier_1"
        assert rows.loc["F001", "consensus_tier"] == "tier_1"
        assert rows.loc["F002", "consensus_tier"] == "tier_2"
        assert rows.loc["F003", "consensus_tier"] == "tier_3"
        assert rows.loc["F004", "consensus_tier"] == "none"

    def test_n_per_tier_sums_to_total(self) -> None:
        univ = _make_univariate_result(n=12, n_signal=4)
        plsda = _make_multivariate_result("plsda", n=12, signal_features=[0, 1, 2])
        splsda = _make_multivariate_result("splsda", n=12, signal_features=[0, 1])
        rf = _make_multivariate_result("rf", n=12, signal_features=[0])
        res = compute_shortlist(
            univ, {"plsda": plsda, "splsda": splsda, "rf": rf}, default_config()
        )
        assert sum(res.n_per_tier.values()) == res.n_input_features


# ============================================================
# Edge cases
# ============================================================
class TestEdgeCases:
    def test_univariate_only_works(self) -> None:
        univ = _make_univariate_result(n=10, n_signal=3)
        # k=1 because only one method available
        cfg = ShortlistConfig(
            method_gates={"univariate": MethodGate("univariate", {})},
            combination="k_of_n",
            k=1,
        )
        res = compute_shortlist(univ, {}, cfg)
        assert res.n_shortlisted == 3

    def test_multivariate_only_works(self) -> None:
        plsda = _make_multivariate_result("plsda", n=10, n_signal=3)
        cfg = ShortlistConfig(
            method_gates={"plsda": MethodGate("plsda", {})},
            combination="k_of_n",
            k=1,
        )
        res = compute_shortlist(None, {"plsda": plsda}, cfg)
        assert res.n_shortlisted == 3

    def test_no_inputs_raises(self) -> None:
        cfg = ShortlistConfig(
            method_gates={"plsda": MethodGate("plsda", {})},
            combination="or",
        )
        with pytest.raises(ValueError, match="No method evidence"):
            compute_shortlist(None, {}, cfg)


# ============================================================
# Composite score
# ============================================================
class TestCompositeScore:
    def test_unweighted_score_equals_n_methods_passed(self) -> None:
        univ = _make_univariate_result(n=5, n_signal=3)
        plsda = _make_multivariate_result("plsda", n=5, signal_features=[0, 1])
        cfg = ShortlistConfig(
            method_gates={
                "univariate": MethodGate("univariate", {}),
                "plsda": MethodGate("plsda", {}),
            },
            combination="or",
        )
        res = compute_shortlist(univ, {"plsda": plsda}, cfg)
        # Unweighted composite = n_methods_passed
        for _, row in res.table.iterrows():
            assert (
                row["composite_score"] == row["n_methods_passed"]
            ), f"feature {row['feature_id']}: score {row['composite_score']} ≠ count {row['n_methods_passed']}"

    def test_weighted_score(self) -> None:
        univ = _make_univariate_result(n=5, n_signal=3)
        plsda = _make_multivariate_result("plsda", n=5, signal_features=[0, 1])
        cfg = ShortlistConfig(
            method_gates={
                "univariate": MethodGate("univariate", {}),
                "plsda": MethodGate("plsda", {}),
            },
            combination="or",
            weights={"univariate": 1.0, "plsda": 3.0},
        )
        res = compute_shortlist(univ, {"plsda": plsda}, cfg)
        rows = res.table.set_index("feature_id")
        # F000: both methods pass → 1*1 + 3*1 = 4
        # F001: both → 4
        # F002: univariate only → 1
        # F003: neither → 0
        # F004: neither → 0
        assert rows.loc["F000", "composite_score"] == 4.0
        assert rows.loc["F001", "composite_score"] == 4.0
        assert rows.loc["F002", "composite_score"] == 1.0
        assert rows.loc["F003", "composite_score"] == 0.0


# ============================================================
# I/O
# ============================================================
class TestWriteShortlist:
    def test_write_creates_all_files(self, tmp_path: Path) -> None:
        univ = _make_univariate_result(n=8, n_signal=3)
        plsda = _make_multivariate_result("plsda", n=8, n_signal=3)
        res = compute_shortlist(
            univ, {"plsda": plsda}, ShortlistConfig(
                method_gates={
                    "univariate": MethodGate("univariate", {}),
                    "plsda": MethodGate("plsda", {}),
                },
                combination="and",
            ),
        )
        paths = write_shortlist(res, tmp_path)
        for key in ("parquet", "csv", "passed_parquet", "config"):
            assert paths[key].exists()
        # Round-trip parquet
        df = pd.read_parquet(paths["parquet"])
        assert "feature_id" in df.columns
        assert "passed_shortlist" in df.columns
        # Passed-only parquet has just the passing features
        passed = pd.read_parquet(paths["passed_parquet"])
        assert int(passed["passed_shortlist"].sum()) == len(passed)
        # Config sidecar contains the rules
        import json
        cfg_data = json.loads(paths["config"].read_text())
        assert cfg_data["combination"] == "and"
        assert cfg_data["n_shortlisted"] == res.n_shortlisted
