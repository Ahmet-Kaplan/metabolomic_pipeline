"""Shared plot output helper — writes HTML + optional PNG for a plotly figure."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import structlog

log = structlog.get_logger(__name__)


def write_plot(
    fig: go.Figure,
    output_dir: Path,
    name: str,
    *,
    emit_png: bool = True,
    png_width: int = 1200,
    png_height: int = 700,
) -> dict[str, Path]:
    """Write a plotly figure to HTML and (optionally) PNG.

    Returns a ``{'html': ..., 'png': ...}`` dict of paths (png key only present
    if ``emit_png`` and kaleido export succeeds).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{name}.html"
    # write_html with include_plotlyjs='cdn' keeps files small; use 'inline' if offline.
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    log.info("eda_plot_html_written", plot=name, path=str(html_path))

    out: dict[str, Path] = {"html": html_path}
    if emit_png:
        png_path = output_dir / f"{name}.png"
        try:
            fig.write_image(str(png_path), width=png_width, height=png_height, scale=2)
            out["png"] = png_path
            log.info("eda_plot_png_written", plot=name, path=str(png_path))
        except Exception as exc:  # pragma: no cover — kaleido issues shouldn't crash the run
            log.warning("eda_plot_png_failed", plot=name, error=str(exc))
    return out
