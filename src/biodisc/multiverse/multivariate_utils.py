"""Shared multivariate-discovery utilities — bootstrap stability + permutation testing.

NOT cross-validation. NOT prediction. NOT classifier evaluation.
====================================================================
This module operationalises **biomarker discovery** as a feature-ranking
problem on a single dataset. The fitted multivariate models (PLS-DA,
sPLS-DA, RF) are *importance generators* — we never deploy them. So
the relevant questions are:

  1. **Is this feature's importance score stable** under data resampling?
     → Bootstrap stability (Meinshausen & Bühlmann 2010 / Nogueira 2018).
     Repeat: sample with replacement, refit the model, record which
     features rank in the top-K. ``stability_score`` = fraction of
     bootstrap resamples in which the feature appears.

  2. **Is the observed importance score better than chance?**
     → Permutation test. Shuffle the class labels, refit, measure
     importance. Repeat 1000×. ``perm_p_value`` = fraction of
     permutations whose importance ≥ the observed value. Standard
     metabolomics practice (Westerhuis 2008 *Metabolomics* 4:81).

NO held-out test set is involved in either. NO hyperparameters are
tuned via CV (would leak with single-dataset discovery — see Westerhuis
2008's discussion of the discovery-vs-validation distinction).

Performance
-----------
For RF with 4644 features × 31 samples × n_estimators=300:
  * Single fit: ~5 s
  * 100 bootstraps: ~8 min
  * 1000 permutations: ~80 min
Total per-method ~90 min on this hardware. Cached output by default;
``n_bootstrap`` and ``n_permutation`` reducible during iteration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class StabilityResult:
    """Bootstrap-stability assessment for one feature-importance method."""

    feature_ids: list[str]
    stability_score: np.ndarray  # fraction of bootstraps in top-K, length n_features
    n_bootstrap: int
    top_k: int


@dataclass(frozen=True)
class PermutationResult:
    """Permutation-test (label-shuffle null) for one feature-importance method."""

    feature_ids: list[str]
    p_value: np.ndarray  # one-sided: fraction of perms with importance ≥ observed
    n_permutation: int


def _auto_top_k(n_features: int) -> int:
    """Data-aware default for stability top-K cutoff.

    ``max(10, min(100, n_features // 100))``. For 5000 features → 50;
    for 100 features → 10 (floor). The rationale: top-K should be a
    realistic biomarker shortlist size, not a single feature.
    """
    return int(max(10, min(100, n_features // 100)))


def _auto_n_estimators(n_samples: int) -> int:
    """Data-aware default for RF n_estimators in our small-N regime.

    ``max(100, n_samples * 10)``. Small N → fewer trees needed to
    saturate; over-large ensembles just inflate runtime without
    improving the importance ranking signal.
    """
    return int(max(100, n_samples * 10))


def bootstrap_stability(
    *,
    importance_fn: Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray],
    matrix_arr: np.ndarray,  # (n_samples, n_features) — sklearn convention
    labels: np.ndarray,  # (n_samples,) class labels
    feature_ids: list[str],
    n_bootstrap: int = 100,
    top_k: int | None = None,
    seed: int = 42,
) -> StabilityResult:
    """Compute bootstrap-stability score per feature.

    Parameters
    ----------
    importance_fn
        Callable ``(X, y, rng) -> array of length n_features``. Should
        fit the multivariate model and return per-feature importance.
        Receives the bootstrap-resampled (X, y) and a method-specific
        RNG (so RF / sklearn estimators stay deterministic).
    matrix_arr
        Sample × feature numpy array (rows = samples).
    labels
        Per-sample integer or string class labels.
    feature_ids
        List of feature_id strings (length = n_features); used in result.
    n_bootstrap
        Number of bootstrap resamples. 100 is conventional for stability
        score precision (Nogueira 2018).
    top_k
        Top-K cutoff for "ranks high". When ``None``, computed via
        ``_auto_top_k(n_features)``.
    seed
        Master RNG seed; per-bootstrap RNGs spawn from this.
    """
    n_samples, n_features = matrix_arr.shape
    chosen_top_k = _auto_top_k(n_features) if top_k is None else top_k
    chosen_top_k = min(chosen_top_k, n_features)
    master_rng = np.random.default_rng(seed)

    # Track how often each feature is in the top-K across bootstraps
    in_top_k_count = np.zeros(n_features, dtype=int)
    n_successful = 0

    for b in range(n_bootstrap):
        # Bootstrap resample WITH REPLACEMENT
        boot_rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1))
        boot_idx = boot_rng.integers(0, n_samples, size=n_samples)
        X_boot = matrix_arr[boot_idx]
        y_boot = labels[boot_idx]
        try:
            importances = importance_fn(X_boot, y_boot, boot_rng)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("bootstrap_iter_failed", iter=b, error=str(exc))
            continue
        n_successful += 1
        # Find indices of top-K features by importance
        top_idx = np.argsort(-importances)[:chosen_top_k]
        in_top_k_count[top_idx] += 1

    stability = in_top_k_count / max(1, n_successful)
    n_failed = n_bootstrap - n_successful
    log.info(
        "bootstrap_stability_done",
        n_bootstrap=n_bootstrap,
        n_successful=n_successful,
        n_failed=n_failed,
        n_features=n_features,
        n_samples=n_samples,
        top_k=chosen_top_k,
        n_features_consistently_in_top_k=int((stability >= 0.8).sum()),
    )
    return StabilityResult(
        feature_ids=feature_ids,
        stability_score=stability,
        n_bootstrap=n_bootstrap,
        top_k=chosen_top_k,
    )


def permutation_test(
    *,
    importance_fn: Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray],
    matrix_arr: np.ndarray,  # (n_samples, n_features)
    labels: np.ndarray,
    observed_importance: np.ndarray,  # (n_features,) — from full-data fit
    feature_ids: list[str],
    n_permutation: int = 1000,
    seed: int = 42,
) -> PermutationResult:
    """One-sided permutation p-value per feature.

    Shuffles labels ``n_permutation`` times, refits the model, records
    per-feature importance under the null. p-value = fraction of
    permutations where the null importance ≥ observed.

    Tied permutations count toward the p-value (conservative, standard).
    p-value floor = ``1 / (n_permutation + 1)`` (we can never claim p=0
    from finite resampling).
    """
    _n_samples, n_features = matrix_arr.shape
    master_rng = np.random.default_rng(seed)

    # Counts of permutations whose importance >= observed (per feature)
    ge_count = np.zeros(n_features, dtype=int)

    for p in range(n_permutation):
        perm_rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1))
        # Shuffle labels (keep matrix order)
        shuffled_labels = perm_rng.permutation(labels)
        try:
            null_importance = importance_fn(matrix_arr, shuffled_labels, perm_rng)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("permutation_iter_failed", iter=p, error=str(exc))
            continue
        ge_count += (null_importance >= observed_importance).astype(int)

    # One-sided p-value with conventional +1 floor
    p_values = (ge_count + 1) / (n_permutation + 1)
    log.info(
        "permutation_test_done",
        n_permutation=n_permutation,
        n_features=n_features,
        n_features_p_lt_05=int((p_values < 0.05).sum()),
        n_features_p_lt_01=int((p_values < 0.01).sum()),
        n_features_p_lt_001=int((p_values < 0.001).sum()),
    )
    return PermutationResult(
        feature_ids=feature_ids,
        p_value=p_values,
        n_permutation=n_permutation,
    )


@dataclass(frozen=True)
class MultivariateResult:
    """Result from one multivariate feature-importance method on full data.

    Per-feature DataFrame columns vary by method; the schema is open
    so each method can add its own (VIP for PLS, Gini for RF, etc.)
    while sharing the universal columns:

      * ``feature_id`` (string)
      * ``perm_importance`` — sklearn permutation_importance score
      * ``stability_score`` — bootstrap stability (∈ [0, 1])
      * ``perm_p_value`` — vs label-shuffle null (∈ [1/(N+1), 1])

    Attributes
    ----------
    method_name
        Identifier (e.g. ``"plsda"``).
    per_feature
        One row per feature; columns at minimum include the universals
        above plus method-specific (VIP, beta_*, gini_importance, etc.).
    params_used
        Serialisable dict of method parameters (n_components etc.).
    runtime_seconds
        Wall-clock time for the full computation.
    n_bootstrap
        Number of bootstrap resamples for stability.
    n_permutation
        Number of label permutations for p-values.
    """

    method_name: str
    per_feature: pd.DataFrame
    params_used: dict[str, Any]
    runtime_seconds: float
    n_bootstrap: int
    n_permutation: int


# Re-export auto-defaults so methods can use the same formulas
__all__ = [
    "MultivariateResult",
    "PermutationResult",
    "StabilityResult",
    "_auto_n_estimators",
    "_auto_top_k",
    "bootstrap_stability",
    "permutation_test",
]
