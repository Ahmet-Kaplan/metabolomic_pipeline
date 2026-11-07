"""Scaling methods for the multiverse pipeline (Stage 16).

Each module implements:

    def scale(ctx: StageContext, **params) -> pd.DataFrame:
        '''Returns a per-feature-scaled matrix; same shape as ctx.matrix.'''

and self-registers via ``register(MethodSpec(category="scaling", ...))``.

Scaling is the FINAL preprocessing step before statistical analysis. It
centers each feature on zero and rescales its variance — controlling
which features contribute most strongly to multivariate models like
PLS-DA.

Why scaling matters
-------------------
Without scaling, a multivariate model (PLS-DA, PCA) is dominated by
the highest-variance features — typically a few abundant metabolites,
not the discriminating signal. Pareto/auto scaling lets every feature
contribute proportionally; tree-based methods (Random Forest) don't
need scaling.
"""
