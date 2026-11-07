"""Unit tests for Random Forest feature-importance generator.

Same biomarker-discovery framing — we verify FEATURE-IMPORTANCE properties,
not predictive metrics. Specific to RF: Gini importance + permutation
importance + bootstrap stability + permutation p-value.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from biodisc.multiverse import StageContext
from biodisc.multiverse.multivariate_utils import MultivariateResult, _auto_n_estimators
from biodisc.multiverse.registry import apply, list_methods


# ============================================================
# Helpers
# ============================================================
def _make_grouped_ctx(
    *,
    n_features: int = 30,
    n_per_group: int = 6,
    group_names: tuple[str, ...] = ("A", "B", "C"),
    n_differential_features: int = 5,
    effect_size: float = 3.0,
    seed: int = 0,
) -> StageContext:
    rng = np.random.default_rng(seed)
    n_groups = len(group_names)
    sample_ids: list[str] = []
    cell_lines: list[str] = []
    for g_name in group_names:
        for r in range(n_per_group):
            sample_ids.append(f"{g_name}_{r + 1}")
            cell_lines.append(g_name)
    n_total = len(sample_ids)

    matrix_data = np.zeros((n_features, n_total))
    base_means = rng.lognormal(0, 0.5, size=n_features)
    for f in range(n_features):
        for j, sid in enumerate(sample_ids):
            g_idx = group_names.index(sid.rsplit("_", 1)[0])
            offset = (g_idx - n_groups / 2) * effect_size if f < n_differential_features else 0
            matrix_data[f, j] = base_means[f] + offset + rng.normal(0, 0.5)

    matrix = pd.DataFrame(
        matrix_data,
        index=[f"F{i}" for i in range(n_features)],
        columns=sample_ids,
    )
    samples = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "raw_sample_id": sample_ids,
            "injection_type": ["biological"] * n_total,
            "cell_line": cell_lines,
            "replicate": [int(s.rsplit("_", 1)[1]) for s in sample_ids],
            "dilution_level": [None] * n_total,
            "acq_date": [datetime(2026, 1, 9)] * n_total,
            "column_order": list(range(n_total)),
        }
    )
    return StageContext(matrix=matrix, samples=samples, features=pd.DataFrame())


# ============================================================
# Auto n_estimators
# ============================================================
class TestAutoNEstimators:
    def test_floor(self) -> None:
        # 5 samples → 50 → floored to 100
        assert _auto_n_estimators(5) == 100

    def test_scales(self) -> None:
        # 30 samples → 300
        assert _auto_n_estimators(30) == 300

    def test_large(self) -> None:
        # 100 samples → 1000
        assert _auto_n_estimators(100) == 1000


# ============================================================
# RF registration + result schema
# ============================================================
class TestRF:
    def test_registered(self) -> None:
        names = {s.name for s in list_methods("multivariate")}
        assert "rf" in names

    def test_returns_multivariate_result(self) -> None:
        ctx = _make_grouped_ctx(seed=40)
        result = apply(
            "multivariate",
            "rf",
            ctx,
            n_estimators=50,
            n_bootstrap=10,
            n_permutation=20,
            n_perm_importance_repeats=3,
        )
        assert isinstance(result, MultivariateResult)
        assert result.method_name == "rf"
        for col in (
            "feature_id",
            "gini_importance",
            "perm_importance",
            "stability_score",
            "perm_p_value",
        ):
            assert col in result.per_feature.columns, f"missing column {col!r}"

    def test_n_estimators_auto(self) -> None:
        ctx = _make_grouped_ctx(n_per_group=4, seed=41)  # 12 samples
        result = apply(
            "multivariate",
            "rf",
            ctx,
            n_bootstrap=3,
            n_permutation=5,
            n_perm_importance_repeats=2,
        )
        # 12 samples → max(100, 120) = 120
        assert result.params_used["n_estimators"] == 120

    def test_explicit_n_estimators_used(self) -> None:
        ctx = _make_grouped_ctx(seed=42)
        result = apply(
            "multivariate",
            "rf",
            ctx,
            n_estimators=42,
            n_bootstrap=3,
            n_permutation=5,
            n_perm_importance_repeats=2,
        )
        assert result.params_used["n_estimators"] == 42

    def test_gini_importance_normalised(self) -> None:
        """sklearn's feature_importances_ sum to 1."""
        ctx = _make_grouped_ctx(seed=43)
        result = apply(
            "multivariate",
            "rf",
            ctx,
            n_estimators=50,
            n_bootstrap=3,
            n_permutation=5,
            n_perm_importance_repeats=2,
        )
        assert abs(result.per_feature["gini_importance"].sum() - 1.0) < 1e-6

    def test_differential_features_higher_gini(self) -> None:
        """Features with strong group effect should get higher Gini importance."""
        ctx = _make_grouped_ctx(
            n_features=20, n_differential_features=5, effect_size=4.0, seed=44
        )
        result = apply(
            "multivariate",
            "rf",
            ctx,
            n_estimators=200,
            n_bootstrap=5,
            n_permutation=10,
            n_perm_importance_repeats=3,
        )
        diff_gini = result.per_feature.iloc[:5]["gini_importance"].mean()
        null_gini = result.per_feature.iloc[5:]["gini_importance"].mean()
        # Differential features should have substantially higher Gini importance
        assert diff_gini > null_gini * 2.0, (
            f"differential Gini (mean={diff_gini:.4f}) should be > 2× null "
            f"(mean={null_gini:.4f})"
        )

    def test_perm_importance_present(self) -> None:
        """perm_importance should not be NaN for RF (it's the model-agnostic measure)."""
        ctx = _make_grouped_ctx(seed=45)
        result = apply(
            "multivariate",
            "rf",
            ctx,
            n_estimators=50,
            n_bootstrap=3,
            n_permutation=5,
            n_perm_importance_repeats=3,
        )
        assert not result.per_feature["perm_importance"].isna().any()

    def test_runtime_recorded(self) -> None:
        ctx = _make_grouped_ctx(seed=46)
        result = apply(
            "multivariate",
            "rf",
            ctx,
            n_estimators=30,
            n_bootstrap=3,
            n_permutation=5,
            n_perm_importance_repeats=2,
        )
        assert result.runtime_seconds > 0
        assert result.n_bootstrap == 3
        assert result.n_permutation == 5
