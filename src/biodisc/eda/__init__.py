"""Stage 2 — Exploratory Data Analysis.

Layer A: ydata-profiling wrappers (``profile.py``) — generic stats per table.
Layer B: seven metabolomics-specific plots, one module per plot.

Policy: Stage 2 is strictly diagnostic. No sample or feature is dropped here;
issues are flagged in the HTML index and the run proceeds.
"""

from __future__ import annotations

__all__ = [
    "run",
]

from biodisc.eda._runner import run
