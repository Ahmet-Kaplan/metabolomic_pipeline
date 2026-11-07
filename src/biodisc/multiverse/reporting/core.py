"""``PathReport`` — accumulates per-section figures/tables/metrics, writes HTML.

Design notes
------------
* **One report per pipeline path.** A path is a specific choice of
  imputer + drift method + normaliser + transform + scaler. The report's
  title makes that choice explicit (e.g. "path_A: halfmin → QC-RSC → PQN
  → log2 → Pareto").

* **Sections are ordered as added.** The renderer respects insertion
  order, mirroring the actual pipeline flow. No reordering magic.

* **Figures get exported twice** — once interactively (Plotly inline div
  + plotly.js from CDN) for the HTML report, and once as PNG (kaleido)
  for paper insertion. ``export_png=False`` skips the kaleido call (used
  in fast-iteration tests).

* **Tables are rendered as HTML server-side** via pandas — no client-side
  data shipped, so reports are self-contained and load instantly.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import structlog
from jinja2 import Environment, FileSystemLoader, select_autoescape

from biodisc.multiverse.reporting.helpers import slugify

log = structlog.get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class FigureItem:
    title: str
    figure: go.Figure
    caption: str = ""
    method_metric: str = ""  # e.g., "D-ratio (Broadhurst 2018)"


@dataclass
class TableItem:
    title: str
    df: pd.DataFrame
    caption: str = ""
    max_rows: int = 30  # truncate long tables; full data lives in the parquets
    escape: bool = True  # set False to permit raw HTML in cells (e.g. hyperlinks)


@dataclass
class MetricItem:
    name: str
    value: Any
    unit: str = ""
    description: str = ""


@dataclass
class Section:
    name: str  # slug, e.g. "imputation"
    title: str  # display, e.g. "Half-min imputation"
    description: str = ""
    method: str = ""  # short identifier, e.g. "halfmin"
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: list[MetricItem] = field(default_factory=list)
    figures: list[FigureItem] = field(default_factory=list)
    tables: list[TableItem] = field(default_factory=list)


class PathReport:
    """Accumulator + renderer for one multiverse pipeline path's report."""

    def __init__(
        self,
        name: str,
        output_dir: Path,
        *,
        title: str | None = None,
        path_chain: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        name
            Path slug (e.g. ``"path_A"``). Used for the output directory name.
        output_dir
            Directory where ``index.html`` and ``figures/`` will be written.
        title
            Human-friendly title for the top of the page. Defaults to
            ``f"Multiverse path report: {name}"``.
        path_chain
            Optional list of method choices in pipeline order, e.g.
            ``["halfmin", "qc_rsc", "pqn", "log2", "pareto"]``. Rendered
            as a banner so the path's identity is obvious at a glance.
        """
        self.name = name
        self.output_dir = Path(output_dir)
        self.title = title or f"Multiverse path report: {name}"
        self.path_chain = path_chain or []
        self.sections: list[Section] = []
        self._sections_by_name: dict[str, Section] = {}
        self.created_at = _dt.datetime.now().isoformat(timespec="seconds")

    # ── Section management ─────────────────────────────────────────────────
    def add_section(
        self,
        name: str,
        title: str,
        *,
        description: str = "",
        method: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Register a new section. Idempotent on ``name``."""
        if name in self._sections_by_name:
            log.warning("section_already_exists_overwriting", section=name)
        section = Section(
            name=name,
            title=title,
            description=description,
            method=method,
            parameters=parameters or {},
        )
        self.sections.append(section)
        self._sections_by_name[name] = section

    def _section(self, name: str) -> Section:
        if name not in self._sections_by_name:
            raise KeyError(
                f"Unknown section {name!r}; call add_section() first. "
                f"Known sections: {sorted(self._sections_by_name)}"
            )
        return self._sections_by_name[name]

    # ── Content adders ─────────────────────────────────────────────────────
    def add_figure(
        self,
        section: str,
        title: str,
        fig: go.Figure,
        *,
        caption: str = "",
        method_metric: str = "",
    ) -> None:
        self._section(section).figures.append(
            FigureItem(title=title, figure=fig, caption=caption, method_metric=method_metric)
        )

    def add_table(
        self,
        section: str,
        title: str,
        df: pd.DataFrame,
        *,
        caption: str = "",
        max_rows: int = 30,
        escape: bool = True,
    ) -> None:
        self._section(section).tables.append(
            TableItem(title=title, df=df, caption=caption, max_rows=max_rows, escape=escape)
        )

    def add_metric(
        self,
        section: str,
        name: str,
        value: Any,
        *,
        unit: str = "",
        description: str = "",
    ) -> None:
        self._section(section).metrics.append(
            MetricItem(name=name, value=value, unit=unit, description=description)
        )

    # ── Rendering ──────────────────────────────────────────────────────────
    def write(
        self,
        *,
        export_png: bool = True,
        png_scale: int = 2,
        template_name: str = "path_index.html.j2",
    ) -> Path:
        """Render index.html + write all figures + tables. Returns index path.

        Parameters
        ----------
        export_png
            If True, also write each figure as a PNG via kaleido (slower).
            Set False for fast-iteration tests.
        png_scale
            Resolution multiplier for PNG export (2 → 2× DPI).
        template_name
            Jinja template under ``reporting/templates/``. Default
            ``path_index.html.j2`` for per-path reports; multiverse index
            renders ``multiverse_index.html.j2`` over the same Section
            machinery.
        """
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write each figure as standalone interactive HTML + PNG, and
        #    capture an inline div+script blob for embedding.
        rendered_sections: list[dict[str, Any]] = []
        n_figures_written = 0

        for sec in self.sections:
            sec_dict: dict[str, Any] = {
                "name": sec.name,
                "title": sec.title,
                "description": sec.description,
                "method": sec.method,
                "parameters": sec.parameters,
                "metrics": [
                    {"name": m.name, "value": m.value, "unit": m.unit, "description": m.description}
                    for m in sec.metrics
                ],
                "figures": [],
                "tables": [],
            }

            for i, fig_item in enumerate(sec.figures):
                slug = f"{sec.name}_{i:02d}_{slugify(fig_item.title)}"
                html_path = figures_dir / f"{slug}.html"
                png_path = figures_dir / f"{slug}.png"

                fig_item.figure.write_html(
                    html_path,
                    include_plotlyjs="cdn",
                    full_html=True,
                )
                if export_png:
                    try:
                        fig_item.figure.write_image(png_path, scale=png_scale)
                    except Exception as exc:  # pragma: no cover — kaleido env issue
                        log.warning("png_export_failed", figure=slug, error=str(exc))

                # Inline embed (div + JS) for the report body
                inline_html = fig_item.figure.to_html(
                    include_plotlyjs=False,  # plotly.js loaded once at top of page
                    full_html=False,
                    div_id=slug,
                )
                sec_dict["figures"].append(
                    {
                        "slug": slug,
                        "title": fig_item.title,
                        "caption": fig_item.caption,
                        "method_metric": fig_item.method_metric,
                        "inline_html": inline_html,
                        "png_relpath": f"figures/{slug}.png",
                        "html_relpath": f"figures/{slug}.html",
                    }
                )
                n_figures_written += 1

            for tbl_item in sec.tables:
                shown = tbl_item.df.head(tbl_item.max_rows)
                truncated = len(tbl_item.df) > tbl_item.max_rows
                sec_dict["tables"].append(
                    {
                        "title": tbl_item.title,
                        "caption": tbl_item.caption,
                        "html": shown.to_html(
                            classes="report-table",
                            border=0,
                            index=False,
                            escape=tbl_item.escape,
                            float_format=lambda x: f"{x:.4g}",
                        ),
                        "n_rows_total": len(tbl_item.df),
                        "n_rows_shown": len(shown),
                        "truncated": truncated,
                    }
                )

            rendered_sections.append(sec_dict)

        # 2. Render the Jinja template
        env = Environment(
            loader=FileSystemLoader(_TEMPLATE_DIR),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template(template_name)
        index_html = template.render(
            title=self.title,
            path_name=self.name,
            path_chain=self.path_chain,
            created_at=self.created_at,
            sections=rendered_sections,
            n_figures=n_figures_written,
            n_sections=len(self.sections),
        )

        index_path = self.output_dir / "index.html"
        index_path.write_text(index_html, encoding="utf-8")

        # 3. Drop a sibling summary.json for machine consumption (Phase 10
        #    orchestrator can roll this up across paths).
        summary = {
            "name": self.name,
            "title": self.title,
            "path_chain": self.path_chain,
            "created_at": self.created_at,
            "n_sections": len(self.sections),
            "n_figures": n_figures_written,
            "sections": [
                {
                    "name": s.name,
                    "title": s.title,
                    "method": s.method,
                    "parameters": s.parameters,
                    "metrics": [{"name": m.name, "value": m.value, "unit": m.unit} for m in s.metrics],
                    "n_figures": len(s.figures),
                    "n_tables": len(s.tables),
                }
                for s in self.sections
            ],
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

        log.info(
            "path_report_written",
            path=self.name,
            output_dir=str(self.output_dir),
            n_sections=len(self.sections),
            n_figures=n_figures_written,
            export_png=export_png,
        )
        return index_path
