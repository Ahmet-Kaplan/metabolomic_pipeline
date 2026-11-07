"""Transformation methods for the multiverse pipeline.

Each module implements:

    def transform(ctx: StageContext, **params) -> pd.DataFrame:
        '''Returns a variance-stabilised matrix; same shape as ctx.matrix.'''

and self-registers via ``register(MethodSpec(category="transformation", ...))``.

Transformations come AFTER normalisation in the standard pipeline. Their
job is to stabilise the per-feature variance across the dynamic range
(LC-MS intensities are typically log-normal — without log-like
transformation, downstream stats are dominated by the most-abundant
features). The transformed matrix can then be scaled (Pareto / auto)
for multivariate analysis.
"""
