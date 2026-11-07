"""Imputation methods for the multiverse pipeline.

Each module in this package implements one imputer following the contract:

    def impute(ctx: StageContext, **params) -> pd.DataFrame:
        '''Returns a wide matrix with NaN replaced; same shape as ctx.matrix.'''

and self-registers via ``register(MethodSpec(category="imputation", ...))``.
"""
