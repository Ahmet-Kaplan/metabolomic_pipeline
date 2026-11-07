"""Multivariate feature-importance methods for biomarker discovery.

Each module here implements:

    def fit(ctx: StageContext, **params) -> MultivariateResult:
        '''Returns per-feature importance + stability + permutation p-values.'''

CRITICAL FRAMING — these are NOT classifiers we deploy. They are
**feature-importance generators** for biomarker discovery on a single
dataset. We never predict on new patients; we never compute confusion
matrix / AUC / Q² — those metrics presuppose a deployment context that
doesn't exist for this pipeline.

The methods give complementary perspectives on which features
discriminate the cell-line groups:

  * **PLS-DA** — covariance-based (variance × correlation with class)
  * **sPLS-DA** — sparse / hard-selected variant of PLS-DA
  * **Random Forest** — non-linear, tree-split-based

A biomarker robust across all three is methodologically robust —
that's the multiverse insight applied to feature-importance methods.
"""
