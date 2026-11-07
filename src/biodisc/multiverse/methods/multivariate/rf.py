"""Random Forest — non-linear feature-importance generator (Breiman 2001).

Third multivariate method. Complements PLS-DA / sPLS-DA which are
covariance-based linear methods; RF is **tree-based, non-linear, and
sensitive to interactions**. A biomarker that ranks high under all three
is methodologically robust — that's the multiverse insight.

What we report (NOT prediction metrics)
---------------------------------------
RF here is a **feature-importance generator**, not a deployable
classifier. We do NOT report OOB error, confusion matrix, or AUC. We DO
report:

  * **Gini importance** — RF's built-in mean decrease in impurity. Fast
    (single fit) but biased toward high-cardinality features (Strobl 2007).
  * **Permutation importance** — sklearn's model-agnostic measure;
    less biased on small N. Computed once on the full-data fit.
  * **Bootstrap stability** — fraction of bootstrap resamples in which
    the feature ranks in the top-K by Gini importance (Nogueira 2018).
  * **Permutation p-value** — fraction of label-shuffled fits whose
    Gini importance ≥ observed (Westerhuis 2008).

Auto-derived parameters
-----------------------
* ``n_estimators``: ``max(100, n_samples * 10)`` — small-N regime saturates
  fast; over-large ensembles inflate runtime without improving the ranking
  signal.
* ``max_features``: ``"sqrt"`` (Breiman default for classification).
* ``min_samples_leaf``: ``1`` — defaults; we don't tune via CV (would leak
  on a single discovery dataset).

References
----------
* Breiman L. 2001. *Mach. Learn.* 45:5. — Random Forests.
* Strobl C, Boulesteix A-L, Zeileis A, Hothorn T. 2007. *BMC Bioinformatics*
  8:25. — Bias of Gini vs. permutation importance.
* Strobl C, Boulesteix A-L, Kneib T, Augustin T, Zeileis A. 2008. *BMC
  Bioinformatics* 9:307. — Conditional variable importance.
* Touw WG et al. 2013. *Brief. Bioinform.* 14:315. — RF in metabolomics
  classification.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import structlog

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.methods.multivariate.plsda import _build_xy
from biodisc.multiverse.multivariate_utils import (
    MultivariateResult,
    _auto_n_estimators,
    bootstrap_stability,
    permutation_test,
)
from biodisc.multiverse.registry import MethodSpec, register

log = structlog.get_logger(__name__)


def _fit_rf(
    X: np.ndarray,  # noqa: N803 — sklearn X convention
    y: np.ndarray,
    n_estimators: int,
    seed: int = 0,
    n_jobs: int = 1,
) -> tuple[Any, np.ndarray]:
    """Fit RandomForestClassifier and return (model, gini_importance)."""
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=1,
        bootstrap=True,
        random_state=seed,
        n_jobs=n_jobs,
    )
    rf.fit(X, y)
    return rf, rf.feature_importances_.astype(float)


def fit(
    ctx: StageContext,
    *,
    n_estimators: int | None = None,
    n_bootstrap: int = 100,
    n_permutation: int = 1000,
    n_perm_importance_repeats: int = 10,
    seed: int = 42,
    biological_only: bool = True,
    n_jobs: int = 1,
) -> MultivariateResult:
    """Fit Random Forest and produce per-feature importance + stability.

    Parameters
    ----------
    n_estimators : int | None, default None (auto)
        ``max(100, n_samples * 10)`` when ``None``. Small-N regime
        saturates fast.
    n_bootstrap : int, default 100
        Bootstrap resamples for Gini-stability.
    n_permutation : int, default 1000
        Label permutations for Gini-importance p-values.
    n_perm_importance_repeats : int, default 10
        Repeats for sklearn's permutation_importance on the primary fit.
        (Single computation, not part of the bootstrap/permutation grids.)
    seed : int, default 42
        Master RNG seed.
    biological_only : bool, default True
        Restrict to biological samples (drop QCs from feature ranking).
    n_jobs : int, default 1
        Parallelism for RF + permutation_importance. Keep 1 by default —
        bootstrap/permutation are the outer loops, parallelism inside
        each sklearn call competes with that.
    """
    t0 = time.time()
    X, y, feature_ids, _sample_ids, class_names = _build_xy(ctx, biological_only=biological_only)
    n_samples, n_features = X.shape
    n_classes = len(class_names)

    chosen_n_estimators = (
        _auto_n_estimators(n_samples) if n_estimators is None else int(n_estimators)
    )

    log.info(
        "rf_fit_start",
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        class_names=class_names,
        n_estimators=chosen_n_estimators,
        n_estimators_auto=(n_estimators is None),
    )

    # Primary fit on full data — produces Gini importance + permutation importance
    rf, gini_obs = _fit_rf(X, y, n_estimators=chosen_n_estimators, seed=seed, n_jobs=n_jobs)

    from sklearn.inspection import permutation_importance

    perm_imp_result = permutation_importance(
        rf,
        X,
        y,
        n_repeats=n_perm_importance_repeats,
        random_state=seed,
        n_jobs=n_jobs,
    )
    perm_imp_obs = perm_imp_result.importances_mean.astype(float)

    # Bootstrap stability via Gini importance
    def gini_importance_fn(
        X_b: np.ndarray,  # noqa: N803 — sklearn X convention
        y_b: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if len(np.unique(y_b)) < 2:
            return np.zeros(n_features)
        try:
            sub_seed = int(rng.integers(0, 2**31 - 1))
            _, g = _fit_rf(X_b, y_b, n_estimators=chosen_n_estimators, seed=sub_seed, n_jobs=1)
            return g
        except Exception:
            return np.zeros(n_features)

    stability = bootstrap_stability(
        importance_fn=gini_importance_fn,
        matrix_arr=X,
        labels=y,
        feature_ids=feature_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    perm_pvals = permutation_test(
        importance_fn=gini_importance_fn,
        matrix_arr=X,
        labels=y,
        observed_importance=gini_obs,
        feature_ids=feature_ids,
        n_permutation=n_permutation,
        seed=seed + 1,
    )

    table = pd.DataFrame({"feature_id": feature_ids})
    table["gini_importance"] = gini_obs
    table["perm_importance"] = perm_imp_obs
    table["stability_score"] = stability.stability_score
    table["perm_p_value"] = perm_pvals.p_value

    elapsed = time.time() - t0
    log.info(
        "rf_fit_done",
        elapsed_s=round(elapsed, 1),
        n_features_high_gini=int((gini_obs >= np.percentile(gini_obs, 95)).sum()),
        n_features_stable=int((stability.stability_score >= 0.8).sum()),
        n_features_p_lt_05=int((perm_pvals.p_value < 0.05).sum()),
    )

    return MultivariateResult(
        method_name="rf",
        per_feature=table,
        params_used={
            "n_estimators": chosen_n_estimators,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "biological_only": biological_only,
            "stability_top_k": stability.top_k,
        },
        runtime_seconds=elapsed,
        n_bootstrap=n_bootstrap,
        n_permutation=n_permutation,
    )


register(
    MethodSpec(
        name="rf",
        callable=fit,
        description="Random Forest feature-importance generator (Gini + perm importance).",
        citation="Breiman 2001, Mach Learn 45:5; Strobl 2007 BMC Bioinformatics 8:25",
        category="multivariate",
    )
)
