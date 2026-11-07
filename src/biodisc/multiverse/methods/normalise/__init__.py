"""Normalisation methods for the multiverse pipeline.

Each module implements the contract:

    def normalise(ctx: StageContext, **params) -> pd.DataFrame:
        '''Returns a per-sample-scaled matrix; same shape as ctx.matrix.'''

and self-registers via ``register(MethodSpec(category="normalisation", ...))``.

All normalisers operate on a feature × sample matrix where missing values
have already been imputed (Stage 11). Output is a per-sample-scaled matrix
of the same shape — observed positive values only.
"""
