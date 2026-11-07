"""Deterministic seed utilities.

Every pipeline stage that touches randomness (imputation benchmark, permutation
tests, bootstrap stability sweeps, CV folds) must call :func:`set_seed` at the
top so reruns of `dvc repro` produce bit-identical outputs.

Usage
-----
>>> from biodisc.seeds import set_seed
>>> set_seed(cfg.seed)
"""

from __future__ import annotations

import os
import random

import numpy as np
import structlog

log = structlog.get_logger(__name__)

_DEFAULT_SEED = 42


def set_seed(seed: int | None = None) -> int:
    """Pin every RNG the pipeline depends on to a single integer seed.

    Covers: Python's ``random``, NumPy's legacy global state, and
    ``PYTHONHASHSEED`` (for dict/set order stability in downstream processes
    we spawn — child processes inherit env).

    Returns the effective seed (the passed-in value or the default).
    """
    effective = _DEFAULT_SEED if seed is None else int(seed)
    random.seed(effective)
    np.random.seed(effective)
    os.environ["PYTHONHASHSEED"] = str(effective)
    log.debug("seed_set", seed=effective)
    return effective


def spawn_rng(parent_seed: int, tag: str) -> np.random.Generator:
    """Derive a named sub-RNG from a parent seed.

    Use when a stage needs multiple independent RNG streams (e.g. imputation
    benchmark: one per (method, mechanism, rate) combination). ``tag`` is
    mixed into the parent seed so different tags → statistically independent
    streams, but the whole thing remains deterministic under the parent seed.
    """
    mixed = hash((parent_seed, tag)) & 0xFFFFFFFF
    return np.random.default_rng(seed=mixed)
