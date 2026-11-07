"""Unit tests — biodisc.multiverse.production path-building.

Covers the config → PathSpec logic that the `biodisc multiverse run` CLI and
the standalone scripts both rely on. The full run() is not unit-tested here
(it needs real data and takes hours); these guard the pure logic.
"""

from __future__ import annotations

import pytest

from biodisc.multiverse.orchestrator import PathSpec
from biodisc.multiverse.production import build_default_paths, build_paths_from_config


# ============================================================
# Locked defaults
# ============================================================
def test_build_default_paths_returns_four_locked_philosophies() -> None:
    paths = build_default_paths()
    assert [p.name for p in paths] == [
        "path_a_classical",
        "path_b_robust_ml",
        "path_c_mnar_aware",
        "path_d_middle",
    ]


def test_default_path_method_chains() -> None:
    """The exact method chains that produced the validated results — locked."""
    paths = {p.name: p for p in build_default_paths()}
    assert paths["path_a_classical"].chain == ["halfmin", "qc_rsc", "pqn", "log2", "pareto"]
    assert paths["path_b_robust_ml"].chain == ["missforest", "serrf", "mstus", "glog", "pareto"]
    assert paths["path_c_mnar_aware"].chain == ["qrilc", "qc_rsc", "pqn", "glog", "pareto"]
    assert paths["path_d_middle"].chain == ["ns_knn", "serrf", "median", "log2", "pareto"]


def test_default_paths_use_auto_params() -> None:
    """No explicit method params → methods fire their data-aware auto-defaults."""
    for p in build_default_paths():
        assert p.imputation_params == {}
        assert p.drift_params == {}
        assert p.normalisation_params == {}
        assert p.transformation_params == {}
        assert p.scaling_params == {}


# ============================================================
# Config → PathSpec
# ============================================================
def _valid_entry(name: str = "p1") -> dict[str, str]:
    return {
        "name": name,
        "imputation": "halfmin",
        "drift": "qc_rsc",
        "normalisation": "pqn",
        "transformation": "log2",
        "scaling": "pareto",
    }


def test_build_from_config_matches_defaults() -> None:
    """The shipped config must reproduce the locked default paths exactly."""
    cfg = [
        {"name": "path_a_classical", "imputation": "halfmin", "drift": "qc_rsc",
         "normalisation": "pqn", "transformation": "log2", "scaling": "pareto"},
        {"name": "path_b_robust_ml", "imputation": "missforest", "drift": "serrf",
         "normalisation": "mstus", "transformation": "glog", "scaling": "pareto"},
        {"name": "path_c_mnar_aware", "imputation": "qrilc", "drift": "qc_rsc",
         "normalisation": "pqn", "transformation": "glog", "scaling": "pareto"},
        {"name": "path_d_middle", "imputation": "ns_knn", "drift": "serrf",
         "normalisation": "median", "transformation": "log2", "scaling": "pareto"},
    ]
    built = build_paths_from_config(cfg)
    defaults = build_default_paths()
    assert [p.chain for p in built] == [p.chain for p in defaults]
    assert [p.name for p in built] == [p.name for p in defaults]


def test_build_from_config_returns_pathspec() -> None:
    built = build_paths_from_config([_valid_entry()])
    assert len(built) == 1
    assert isinstance(built[0], PathSpec)
    assert built[0].name == "p1"


def test_scaling_defaults_to_pareto_when_omitted() -> None:
    entry = _valid_entry()
    del entry["scaling"]
    built = build_paths_from_config([entry])
    assert built[0].scaling == "pareto"


def test_optional_params_passed_through() -> None:
    entry = _valid_entry()
    entry_with_params = {**entry, "imputation_params": {"factor": 0.3}}
    built = build_paths_from_config([entry_with_params])
    assert built[0].imputation_params == {"factor": 0.3}


def test_missing_required_field_raises() -> None:
    entry = _valid_entry()
    del entry["drift"]
    with pytest.raises(ValueError, match="missing required field"):
        build_paths_from_config([entry])


def test_none_required_field_raises() -> None:
    entry = _valid_entry()
    entry["normalisation"] = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="missing required field"):
        build_paths_from_config([entry])


def test_empty_config_raises() -> None:
    with pytest.raises(ValueError, match="no paths"):
        build_paths_from_config([])
