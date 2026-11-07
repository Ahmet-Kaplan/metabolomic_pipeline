"""ydata-profiling wrapper — generic auto-reports per table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger(__name__)


_DEFAULT_CONFIG = {
    # Reasonable defaults for exploratory use. See ydata-profiling docs for more.
    "title": "Stage 2 EDA — auto profile",
    "explorative": True,
    "correlations": {"pearson": {"calculate": True}, "spearman": {"calculate": False}},
    "interactions": {"continuous": False},
    "samples": {"head": 10, "tail": 0},
    "plot": {"histogram": {"bins": 50}},
}


def profile_table(
    df: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    minimal: bool = False,
) -> Path:
    """Write a ydata-profiling HTML report for ``df`` to ``output_path``.

    Parameters
    ----------
    df
        DataFrame to profile. For very long tables (> 200k rows) set
        ``minimal=True`` to skip expensive correlation / interaction analysis.
    output_path
        Path to the output HTML.
    title
        Human-readable title for the report.
    minimal
        If True, disables slow expensive analyses (correlations, interactions,
        duplicates). Use for the 1M-row identification table.
    """
    from ydata_profiling import ProfileReport

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "ydata_profile_start",
        title=title,
        rows=len(df),
        cols=len(df.columns),
        minimal=minimal,
    )
    report = ProfileReport(
        df,
        title=title,
        minimal=minimal,
        explorative=not minimal,
        correlations=None if minimal else {"pearson": {"calculate": True}},
        interactions=None if minimal else {"continuous": False},
        progress_bar=False,
    )
    report.to_file(str(output_path))
    log.info("ydata_profile_written", title=title, path=str(output_path))
    return output_path
