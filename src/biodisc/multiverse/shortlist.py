"""Phase 9 — per-path biomarker shortlist (multi-metric AND / OR / k-of-N rules).

Combines per-method evidence (univariate ANOVA + BH-FDR + η²;
multivariate PLS-DA / sPLS-DA / RF with bootstrap stability + permutation
p-values) into a single per-path biomarker shortlist with explicit
provenance for every flag.

Why a multi-criteria approach
-----------------------------
At the discovery N (5 cell lines × 6–7 reps = 31 samples), each method
on its own has noise: BH-FDR admits some false positives at q &lt; 0.05;
PLS-DA's VIP score is biased toward features with large variance not
necessarily large effect (Filzmoser 2009); RF's Gini importance is
biased toward high-cardinality features (Strobl 2007). **Cross-method
agreement** is the strongest defensible filter — features flagged by
multiple methods are robust to any single method's pathology
(Steegen 2016 multiverse principle).

The default config asks "≥ 2 of 4 methods agree" (k-of-N with k=2). This
is the canonical mQACC consensus pattern — strict enough to filter out
single-method noise, loose enough to surface biology that one method
sees clearly while another marginally misses (sample size limits).

Output
------
``ShortlistResult.table`` is a per-feature DataFrame with explicit
columns for every method's score AND verdict, plus the cross-method
consensus tier. A reviewer can answer "why did feature X make the
shortlist?" by looking at one row.

Tier definitions (k_of_n with k=2 default)
------------------------------------------
* **tier_1** — ≥ 3 methods agree (high-confidence, multi-method consensus).
* **tier_2** — exactly 2 methods agree (default shortlist threshold).
* **tier_3** — exactly 1 method agrees (single-method-only — kept for
  auditing, NOT in the default shortlist).
* **none** — 0 methods agree.

References
----------
* Steegen S et al. 2016. *Increasing transparency through a multiverse
  analysis.* **Perspect Psychol Sci** 11:702.
* Westerhuis JA et al. 2008. *Assessment of PLSDA cross validation.*
  **Metabolomics** 4:81.
* Saccenti E et al. 2014. *Reflections on univariate and multivariate
  analysis of metabolomics data.* **Metabolomics** 10:361.
* Nogueira S, Sechidis K, Brown G. 2018. *On the stability of feature
  selection algorithms.* **J Mach Learn Res** 18:1.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse.multivariate_utils import MultivariateResult
from biodisc.multiverse.stats_univariate import UnivariateResult

log = structlog.get_logger(__name__)


# ============================================================
# Configuration
# ============================================================
@dataclass(frozen=True)
class MethodGate:
    """Per-method gating thresholds.

    ``rules`` is a free-form dict of threshold names to values. Each
    method's ``passed`` mask is computed by ``_apply_<method>_gate``
    against this dict; unknown keys are ignored (so the same
    ``ShortlistConfig`` can be reused across pipelines that have
    different sub-method capabilities).
    """

    name: str
    rules: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ShortlistConfig:
    """Per-path shortlist rule definition.

    Attributes
    ----------
    method_gates
        Map of method name → ``MethodGate`` with thresholds.
    combination
        How to combine per-method ``passed`` flags into the final
        shortlist:

        * ``"and"`` — feature must pass ALL methods.
        * ``"or"`` — feature must pass AT LEAST ONE method.
        * ``"k_of_n"`` — feature must pass at least ``k`` of the
          configured methods. Default ``k=2``. Most useful in practice.
    k
        For ``combination="k_of_n"`` only.
    weights
        Optional method weights for the ``composite_score`` column.
        Each method's contribution to the score is ``weight * passed``.
        ``None`` (default) means equal weights = 1.0 per method.
    """

    method_gates: dict[str, MethodGate]
    combination: str = "k_of_n"
    k: int = 2
    weights: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.combination not in ("and", "or", "k_of_n"):
            raise ValueError(
                f"combination must be 'and', 'or', or 'k_of_n'; got "
                f"{self.combination!r}"
            )
        if self.combination == "k_of_n" and (self.k < 1 or self.k > len(self.method_gates)):
            raise ValueError(
                f"k must be in [1, {len(self.method_gates)}]; got {self.k}"
            )


def default_config() -> ShortlistConfig:
    """Return the textbook mQACC default config: k=2 of 4 methods."""
    return ShortlistConfig(
        method_gates={
            "univariate": MethodGate(
                name="univariate",
                rules={"q_threshold": 0.05, "eta_threshold": 0.14},
            ),
            "plsda": MethodGate(
                name="plsda",
                rules={
                    "vip_threshold": 1.0,
                    "stability_threshold": 0.8,
                    "perm_p_threshold": 0.05,
                },
            ),
            "splsda": MethodGate(
                name="splsda",
                rules={
                    "stability_threshold": 0.8,
                    "perm_p_threshold": 0.05,
                },
            ),
            "rf": MethodGate(
                name="rf",
                rules={
                    "stability_threshold": 0.8,
                    "perm_p_threshold": 0.05,
                },
            ),
        },
        combination="k_of_n",
        k=2,
    )


# ============================================================
# Per-method gate evaluation
# ============================================================
def _apply_univariate_gate(
    result: UnivariateResult, gate: MethodGate
) -> pd.DataFrame:
    """Return per-feature evidence DataFrame for univariate."""
    table = result.table.copy()
    q_thr = float(gate.rules.get("q_threshold", 0.05))
    eta_thr = float(gate.rules.get("eta_threshold", 0.14))
    out = pd.DataFrame(
        {
            "feature_id": table["feature_id"].astype(str),
            "univariate_q": table["q_value"],
            "univariate_eta_sq": table["eta_squared"],
        }
    )
    out["univariate_passed"] = (
        (out["univariate_q"] < q_thr) & (out["univariate_eta_sq"] >= eta_thr)
    ).fillna(False)
    return out


def _apply_plsda_gate(
    result: MultivariateResult, gate: MethodGate
) -> pd.DataFrame:
    """Return per-feature evidence DataFrame for PLS-DA."""
    table = result.per_feature.copy()
    vip_thr = float(gate.rules.get("vip_threshold", 1.0))
    stab_thr = float(gate.rules.get("stability_threshold", 0.8))
    p_thr = float(gate.rules.get("perm_p_threshold", 0.05))
    out = pd.DataFrame(
        {
            "feature_id": table["feature_id"].astype(str),
            "plsda_vip": table.get("vip", pd.Series(np.nan, index=table.index)),
            "plsda_stability": table.get(
                "stability_score", pd.Series(np.nan, index=table.index)
            ),
            "plsda_perm_p": table.get(
                "perm_p_value", pd.Series(np.nan, index=table.index)
            ),
        }
    )
    out["plsda_passed"] = (
        (out["plsda_vip"] >= vip_thr)
        & (out["plsda_stability"] >= stab_thr)
        & (out["plsda_perm_p"] < p_thr)
    ).fillna(False)
    return out


def _apply_splsda_gate(
    result: MultivariateResult, gate: MethodGate
) -> pd.DataFrame:
    """Return per-feature evidence DataFrame for sPLS-DA."""
    table = result.per_feature.copy()
    stab_thr = float(gate.rules.get("stability_threshold", 0.8))
    p_thr = float(gate.rules.get("perm_p_threshold", 0.05))
    selected = table.get(
        "selected", pd.Series(False, index=table.index)
    ).astype(bool)
    out = pd.DataFrame(
        {
            "feature_id": table["feature_id"].astype(str),
            "splsda_selected": selected,
            "splsda_stability": table.get(
                "stability_score", pd.Series(np.nan, index=table.index)
            ),
            "splsda_perm_p": table.get(
                "perm_p_value", pd.Series(np.nan, index=table.index)
            ),
        }
    )
    out["splsda_passed"] = (
        selected
        & (out["splsda_stability"] >= stab_thr)
        & (out["splsda_perm_p"] < p_thr)
    ).fillna(False)
    return out


def _apply_rf_gate(
    result: MultivariateResult, gate: MethodGate
) -> pd.DataFrame:
    """Return per-feature evidence DataFrame for RF."""
    table = result.per_feature.copy()
    stab_thr = float(gate.rules.get("stability_threshold", 0.8))
    p_thr = float(gate.rules.get("perm_p_threshold", 0.05))
    out = pd.DataFrame(
        {
            "feature_id": table["feature_id"].astype(str),
            "rf_gini": table.get(
                "gini_importance", pd.Series(np.nan, index=table.index)
            ),
            "rf_stability": table.get(
                "stability_score", pd.Series(np.nan, index=table.index)
            ),
            "rf_perm_p": table.get(
                "perm_p_value", pd.Series(np.nan, index=table.index)
            ),
        }
    )
    out["rf_passed"] = (
        (out["rf_stability"] >= stab_thr) & (out["rf_perm_p"] < p_thr)
    ).fillna(False)
    return out


_MultivariateGateFn = Callable[[MultivariateResult, MethodGate], pd.DataFrame]
_GATE_DISPATCH: dict[str, _MultivariateGateFn] = {
    "plsda": _apply_plsda_gate,
    "splsda": _apply_splsda_gate,
    "rf": _apply_rf_gate,
}


# ============================================================
# Result container
# ============================================================
@dataclass(frozen=True)
class ShortlistResult:
    """Output of :func:`compute_shortlist`.

    Attributes
    ----------
    table
        Per-feature DataFrame with all evidence columns + verdict.
    n_input_features
        Number of features fed in (union across method results).
    n_shortlisted
        ``int(table["passed_shortlist"].sum())``.
    n_per_tier
        Map ``{"tier_1": int, "tier_2": int, "tier_3": int, "none": int}``.
    config
        The ``ShortlistConfig`` used.
    runtime_seconds
        Wall time of :func:`compute_shortlist`.
    """

    table: pd.DataFrame
    n_input_features: int
    n_shortlisted: int
    n_per_tier: dict[str, int]
    config: ShortlistConfig
    runtime_seconds: float


# ============================================================
# Main entry point
# ============================================================
def compute_shortlist(
    univariate_result: UnivariateResult | None,
    multivariate_results: dict[str, MultivariateResult],
    config: ShortlistConfig | None = None,
) -> ShortlistResult:
    """Compute per-path biomarker shortlist from per-method evidence.

    Parameters
    ----------
    univariate_result
        ANOVA + BH-FDR + η² results. ``None`` skips the univariate gate
        entirely (useful for pathways that didn't run univariate).
    multivariate_results
        Map of method name → ``MultivariateResult``. Recognised keys:
        ``"plsda"``, ``"splsda"``, ``"rf"``. Other keys are ignored.
    config
        Per-path config. ``None`` uses :func:`default_config` (k=2 of 4
        methods, mQACC textbook thresholds).
    """
    config = config or default_config()
    t0 = time.time()

    # Build per-method evidence DataFrames
    evidence_frames: list[pd.DataFrame] = []
    methods_used: list[str] = []
    for method_name, gate in config.method_gates.items():
        if method_name == "univariate":
            if univariate_result is None:
                continue
            df = _apply_univariate_gate(univariate_result, gate)
        elif method_name in multivariate_results:
            df = _GATE_DISPATCH[method_name](
                multivariate_results[method_name], gate
            )
        else:
            continue
        evidence_frames.append(df)
        methods_used.append(method_name)

    if not evidence_frames:
        raise ValueError(
            "No method evidence available — pass at least one of "
            "univariate_result or multivariate_results."
        )

    # Warn when a method was requested but produced no passing features
    # (e.g., RF with zero stability at low sample size).
    for method_name in methods_used:
        df = evidence_frames[methods_used.index(method_name)]
        n_passed = int(df[f"{method_name}_passed"].sum()) if f"{method_name}_passed" in df.columns else 0
        if n_passed == 0:
            log.warning(
                "method_contributed_no_features",
                method=method_name,
                n_evidence_rows=len(df),
            )

    # Outer-join on feature_id (in case methods see different feature sets,
    # which shouldn't happen but is defensive).
    table = evidence_frames[0]
    for df in evidence_frames[1:]:
        table = table.merge(df, on="feature_id", how="outer")

    # Per-feature pass count
    pass_cols = [f"{m}_passed" for m in methods_used]
    for col in pass_cols:
        if col not in table.columns:
            table[col] = False
    table["n_methods_passed"] = (
        table[pass_cols].fillna(False).astype(int).sum(axis=1)
    )

    # Final shortlist verdict per combination rule
    if config.combination == "and":
        table["passed_shortlist"] = table["n_methods_passed"] == len(methods_used)
    elif config.combination == "or":
        table["passed_shortlist"] = table["n_methods_passed"] >= 1
    else:  # k_of_n
        table["passed_shortlist"] = table["n_methods_passed"] >= config.k

    # Composite score (weighted sum across methods)
    weights = config.weights or {m: 1.0 for m in methods_used}
    table["composite_score"] = sum(
        float(weights.get(m, 1.0)) * table[f"{m}_passed"].astype(float)
        for m in methods_used
    )

    # Consensus tier (mQACC style — describes evidence-strength of each feature)
    n_methods = len(methods_used)
    table["consensus_tier"] = "none"
    if n_methods >= 1:
        table.loc[table["n_methods_passed"] >= 1, "consensus_tier"] = "tier_3"
    if n_methods >= 2:
        table.loc[table["n_methods_passed"] >= 2, "consensus_tier"] = "tier_2"
    if n_methods >= 3:
        table.loc[table["n_methods_passed"] >= 3, "consensus_tier"] = "tier_1"

    # Sort by score descending (then by feature_id for determinism)
    table = table.sort_values(
        ["passed_shortlist", "composite_score", "n_methods_passed", "feature_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    # Counts
    n_per_tier = {
        tier: int((table["consensus_tier"] == tier).sum())
        for tier in ("tier_1", "tier_2", "tier_3", "none")
    }
    n_shortlisted = int(table["passed_shortlist"].sum())

    elapsed = time.time() - t0
    log.info(
        "shortlist_computed",
        n_input_features=len(table),
        n_methods_used=len(methods_used),
        methods=methods_used,
        combination=config.combination,
        k=config.k if config.combination == "k_of_n" else None,
        n_shortlisted=n_shortlisted,
        n_per_tier=n_per_tier,
        elapsed_s=round(elapsed, 2),
    )
    return ShortlistResult(
        table=table,
        n_input_features=len(table),
        n_shortlisted=n_shortlisted,
        n_per_tier=n_per_tier,
        config=config,
        runtime_seconds=elapsed,
    )


# ============================================================
# I/O helpers
# ============================================================
def write_shortlist(
    result: ShortlistResult, output_dir: Any, *, name: str = "shortlist"
) -> dict[str, Any]:
    """Write ``result.table`` to parquet + CSV alongside a config sidecar.

    Output files in ``output_dir`` (``pathlib.Path``-coercible):
      * ``{name}.parquet`` — full per-feature table
      * ``{name}.csv`` — same, human-readable
      * ``{name}_passed.parquet`` — just the rows where
        ``passed_shortlist=True``, sorted by composite_score desc
      * ``{name}_config.json`` — config used (rules + combination + k)
    """
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {}

    parquet_path = output_dir / f"{name}.parquet"
    csv_path = output_dir / f"{name}.csv"
    result.table.to_parquet(parquet_path, index=False)
    result.table.to_csv(csv_path, index=False)
    paths["parquet"] = parquet_path
    paths["csv"] = csv_path

    passed_path = output_dir / f"{name}_passed.parquet"
    result.table[result.table["passed_shortlist"]].to_parquet(
        passed_path, index=False
    )
    paths["passed_parquet"] = passed_path

    config_path = output_dir / f"{name}_config.json"
    config_path.write_text(
        json.dumps(
            {
                "method_gates": {
                    name: {"rules": gate.rules}
                    for name, gate in result.config.method_gates.items()
                },
                "combination": result.config.combination,
                "k": result.config.k,
                "weights": result.config.weights,
                "n_input_features": result.n_input_features,
                "n_shortlisted": result.n_shortlisted,
                "n_per_tier": result.n_per_tier,
                "runtime_seconds": result.runtime_seconds,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    paths["config"] = config_path

    log.info(
        "shortlist_written",
        output_dir=str(output_dir),
        n_features=len(result.table),
        n_passed=result.n_shortlisted,
    )
    return paths
