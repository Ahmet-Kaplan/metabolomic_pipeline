"""Plotly theme + class colour palette for multiverse path reports.

The palette is **fixed and shared** across every figure in the report so
the same cell-line colour appears in PCA, score plots, volcano legends,
β-coefficient heatmaps, etc. This is essential for paper-grade
consistency.

Colours from Wong's 2011 colour-blind-friendly palette
(https://www.nature.com/articles/nmeth.1618) — also exposed by Plotly
as ``plotly.colors.qualitative.Safe``. We pin five for the cell lines
plus a neutral grey for QC pool and a black for blanks.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# ===== Class colour palette =====================================================
# Wong 2011 / Plotly Safe (color-blind friendly). Fixed mapping across all figures.
CLASS_COLORS: dict[str, str] = {
    # 5 breast cell lines
    "MCF7":     "#0072B2",  # blue
    "MCF10A":   "#009E73",  # bluish-green
    "MDAMB231": "#D55E00",  # vermilion
    "SKBR3":    "#CC79A7",  # reddish-purple
    "UACC":     "#E69F00",  # orange
    # technical injection types
    "QC":      "#888888",  # neutral grey — drift reference
    "qc":      "#888888",
    "blank":   "#000000",  # black — instrument background
    "dilution": "#F0E442",  # yellow — linearity ladder
}

# Default fallback for any class not in the map (cyan-magenta cycle).
_FALLBACK_PALETTE: list[str] = [
    "#56B4E9", "#882255", "#117733", "#999933", "#AA4499", "#332288",
]


def color_for(name: str, fallback_index: int = 0) -> str:
    """Return the canonical colour for a class label, with a stable fallback."""
    if name in CLASS_COLORS:
        return CLASS_COLORS[name]
    return _FALLBACK_PALETTE[fallback_index % len(_FALLBACK_PALETTE)]


def display_class(cell_line: object, injection_type: object) -> str:
    """Resolve a per-sample display label that always identifies the QC pool.

    Real-data convention: ``cell_line`` is the cell-line name for biological
    samples and ``None`` (or NaN) for QC / blank / dilution rows. Plain
    string-conversion would render those as ``"None"`` in legends and
    ellipse labels — visually wrong because the QC pool is a meaningful
    group that needs its own colour and ellipse.

    This helper returns:
      * ``cell_line`` for biological samples (e.g. ``"MCF7"``)
      * the lowercase injection type otherwise (``"qc"``, ``"blank"``,
        ``"dilution"``)

    The returned string is used directly as a key into ``CLASS_COLORS``,
    so ``"qc"`` resolves to the neutral grey, etc.
    """
    if isinstance(injection_type, str) and injection_type == "biological":
        # Biological row: trust the cell_line column
        if cell_line is None:
            return "unknown"
        if isinstance(cell_line, float) and cell_line != cell_line:  # NaN check
            return "unknown"
        return str(cell_line)
    if injection_type is None:
        return "unknown"
    if isinstance(injection_type, float) and injection_type != injection_type:
        return "unknown"
    return str(injection_type)


# ===== Plotly publication theme =================================================
PLOTLY_TEMPLATE = "simple_white"

PUBLICATION_FONT: dict[str, Any] = {
    # Browser maps "serif" → Times/Liberation Serif on most systems; matches
    # the Computer-Modern aesthetic of an ACS / Nature figure caption.
    "family": "serif",
    "size": 12,
    "color": "#222222",
}

TITLE_FONT: dict[str, Any] = {
    "family": "serif",
    "size": 14,
    "color": "#111111",
}

AXIS_FONT: dict[str, Any] = {
    "family": "serif",
    "size": 11,
    "color": "#333333",
}

# Margins tuned for ~600 × 400 figures embedded in HTML reports.
DEFAULT_MARGIN: dict[str, int] = {"l": 60, "r": 20, "t": 50, "b": 60}


def apply_theme(
    fig: go.Figure,
    *,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Apply the publication theme in-place and return the figure.

    Sets template, fonts, margins; optionally sets title + axis titles +
    figure size. Idempotent — safe to call multiple times.
    """
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=PUBLICATION_FONT,
        title_font=TITLE_FONT,
        margin=DEFAULT_MARGIN,
    )
    if title is not None:
        fig.update_layout(title=title)
    if xaxis_title is not None:
        fig.update_xaxes(title=xaxis_title, title_font=AXIS_FONT)
    if yaxis_title is not None:
        fig.update_yaxes(title=yaxis_title, title_font=AXIS_FONT)
    if width is not None:
        fig.update_layout(width=width)
    if height is not None:
        fig.update_layout(height=height)
    return fig
