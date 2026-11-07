"""Drift correction methods for the multiverse pipeline.

Each module implements the contract:

    def correct(ctx: StageContext, **params) -> pd.DataFrame:
        '''Returns a drift-corrected matrix; same shape as ctx.matrix.'''

and self-registers via ``register(MethodSpec(category="drift", ...))``.

All drift correctors require ``ctx.run_order`` (chronological injection
sequence) and ``ctx.samples`` (to identify QC anchors). They operate on
a feature × sample matrix where missing values have already been imputed
(Stage 11). Output is a per-sample-rescaled matrix of the same shape:
QC-anchored drift trends are removed, biological signal preserved.
"""
