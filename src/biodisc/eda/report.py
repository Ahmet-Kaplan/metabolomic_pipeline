"""Stage 2 index report — Jinja2 HTML linking every plot + profile."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Any


@dataclass(frozen=True)
class ReportEntry:
    """One section in the index."""

    key: str
    title: str
    description: str
    html_href: str
    png_href: str | None = None


def build_index(
    *,
    output_path: Path,
    entries: list[ReportEntry],
    project_name: str = "Biomarker Discovery MLOps",
    stage_label: str = "Stage 2 — Exploratory Data Analysis",
    extra_info: dict[str, Any] | None = None,
) -> Path:
    """Write the top-level index.html linking every plot + profile."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # Hand-rolled HTML so we don't need to ship a templates/ dir for such a
    # simple page. Uses the same navy-and-cream palette as plan.pdf.
    cards_html = []
    for e in entries:
        png_snippet = (
            f'<a class="png" href="{e.png_href}" target="_blank">PNG</a>' if e.png_href else ""
        )
        cards_html.append(
            dedent(
                f"""
                <article class="card">
                  <header>
                    <h2>{e.title}</h2>
                    <span class="tag">{e.key}</span>
                  </header>
                  <p>{e.description}</p>
                  <footer>
                    <a class="primary" href="{e.html_href}" target="_blank">Open interactive HTML ↗</a>
                    {png_snippet}
                  </footer>
                </article>
                """
            ).strip()
        )

    info_html = ""
    if extra_info:
        info_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in extra_info.items())
        info_html = f"<table class='info'>{info_rows}</table>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{stage_label} — {project_name}</title>
<style>
  :root {{
    --navy: #0b2545;
    --navy-2: #13315c;
    --blue: #1e4377;
    --accent: #c9a94e;
    --paper: #f6f8fb;
    --border: #d6dde6;
    --text: #1a1a1a;
    --muted: #555;
  }}
  body {{
    font-family: "Source Sans Pro", "Helvetica Neue", sans-serif;
    background: var(--paper); color: var(--text);
    margin: 0; padding: 0; line-height: 1.55;
  }}
  header.top {{
    background: linear-gradient(180deg, var(--navy) 0%, var(--navy-2) 100%);
    color: #fff; padding: 3rem 2rem;
  }}
  header.top h1 {{ margin: 0 0 .3em 0; font-weight: 700; font-size: 2.1rem; }}
  header.top p  {{ margin: 0; color: #a8c4e6; font-size: 1.05rem; }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
  table.info {{
    border-collapse: collapse; margin: 1.5em 0; font-size: .95rem;
  }}
  table.info th {{ background: var(--navy); color: #fff; padding: .4em .8em; text-align: left; }}
  table.info td {{ padding: .4em .8em; border-bottom: 1px solid var(--border); }}
  .cards {{
    display: grid; gap: 1.5rem;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    margin-top: 2rem;
  }}
  .card {{
    background: #fff; border: 1px solid var(--border); border-radius: 4px;
    padding: 1.3rem; box-shadow: 0 1px 3px rgba(11, 37, 69, .06);
  }}
  .card header {{ display:flex; justify-content:space-between; align-items:baseline; }}
  .card h2 {{ margin: 0; font-size: 1.08rem; color: var(--navy); }}
  .card .tag {{ font-size: .78rem; color: var(--muted); font-family: ui-monospace, monospace; }}
  .card p  {{ color: var(--muted); font-size: .92rem; margin: .7em 0 1em; }}
  .card footer {{ display: flex; gap: .8em; }}
  .card a {{
    padding: .35em .8em; border-radius: 3px; text-decoration: none; font-size: .88rem;
  }}
  .card a.primary {{ background: var(--navy); color: #fff; }}
  .card a.primary:hover {{ background: var(--blue); }}
  .card a.png {{ background: #eee; color: var(--text); }}
  footer.foot {{
    padding: 2rem; text-align: center; color: var(--muted); font-size: .85rem;
    border-top: 1px solid var(--border); margin-top: 3rem;
  }}
</style>
</head>
<body>
<header class="top">
  <h1>{project_name}</h1>
  <p>{stage_label}</p>
</header>
<main>
  {info_html}
  <div class="cards">
    {"".join(cards_html)}
  </div>
</main>
<footer class="foot">
  Generated {generated} · Stage 2 is strictly diagnostic — no features or samples are dropped.
</footer>
</body>
</html>
"""
    output_path.write_text(html)
    return output_path
