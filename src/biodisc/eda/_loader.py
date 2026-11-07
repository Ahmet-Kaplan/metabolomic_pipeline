"""Shared loader for Stage-1 parquet tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Stage1Data:
    """Container bundling all Stage-1 parquet outputs.

    Loaders default to the paths Stage 1 writes to; plot modules take this
    container as input and never hit disk themselves.
    """

    features: pd.DataFrame
    samples: pd.DataFrame
    abundances_raw_long: pd.DataFrame
    abundances_raw_wide: pd.DataFrame
    identification: pd.DataFrame
    blanks: pd.DataFrame
    dilution_ladder: pd.DataFrame


def load_stage1(stage_01_dir: Path) -> Stage1Data:
    """Load every Stage-1 parquet into a Stage1Data bundle."""
    d = Path(stage_01_dir)
    return Stage1Data(
        features=pd.read_parquet(d / "features.parquet"),
        samples=pd.read_parquet(d / "samples.parquet"),
        abundances_raw_long=pd.read_parquet(d / "abundances_raw_long.parquet"),
        abundances_raw_wide=pd.read_parquet(d / "abundances_raw_wide.parquet"),
        identification=pd.read_parquet(d / "identification_long.parquet"),
        blanks=pd.read_parquet(d / "blanks_long.parquet"),
        dilution_ladder=pd.read_parquet(d / "dilution_ladder_long.parquet"),
    )
