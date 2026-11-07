"""Multiverse pipeline — run N parallel paths post-Stage-10, converge at consensus.

Each path is a fixed combination of methods (imputation × drift × normalisation
× transformation). Paths run end-to-end independently; their per-path biomarker
shortlists converge at the consensus stage. A biomarker that appears in more
paths is more robust to methodological choice — that's the multiverse insight
(Steegen et al. 2016).

Module layout
-------------
* ``_types``  — shared dataclasses (StageContext, PathSpec)
* ``registry`` — central method registry; methods self-register on import
* ``methods/``  — one file per method, each calling ``register(...)`` at module
  load
* ``orchestrator`` — runs a single path or all paths
* ``consensus`` — merges per-path shortlists into Tier-1/2/3

To add a new method
-------------------
1. Drop a file in ``methods/<category>/<name>.py``
2. Implement ``def fn(ctx: StageContext, **params) -> pd.DataFrame`` and call
   ``register(MethodSpec(...))`` at the end
3. Reference ``<name>`` in any path YAML

Methods auto-load at package import — see ``methods/__init__.py``.
"""

# Importing methods triggers their self-registration.
from biodisc.multiverse import methods  # noqa: F401
from biodisc.multiverse._types import StageContext
from biodisc.multiverse.registry import MethodSpec, get, list_methods, register

__all__ = ["MethodSpec", "StageContext", "get", "list_methods", "register"]
