"""Unit tests for multiverse path-report infrastructure (Phase 8e.1).

These verify the SHELL of the reporting system — the figure types and
section coverage will get their own tests in 8e.2 onwards.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from biodisc.multiverse.reporting import CLASS_COLORS, PathReport, apply_theme
from biodisc.multiverse.reporting.helpers import confidence_ellipse_2d, slugify
from biodisc.multiverse.reporting.theme import color_for


# ============================================================
# Theme + colours
# ============================================================
class TestColours:
    def test_five_cell_lines_present(self) -> None:
        for cl in ("MCF7", "MCF10A", "MDAMB231", "SKBR3", "UACC"):
            assert cl in CLASS_COLORS
            # 7-char hex
            assert CLASS_COLORS[cl].startswith("#") and len(CLASS_COLORS[cl]) == 7

    def test_qc_neutral_grey(self) -> None:
        # Both QC and qc resolve to the same grey
        assert CLASS_COLORS["QC"] == CLASS_COLORS["qc"] == "#888888"

    def test_color_for_known_class(self) -> None:
        assert color_for("MCF7") == CLASS_COLORS["MCF7"]

    def test_color_for_unknown_falls_back(self) -> None:
        c = color_for("UnknownClass", fallback_index=2)
        assert c.startswith("#")


class TestApplyTheme:
    def test_sets_template_and_font(self) -> None:
        fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[1, 2])])
        apply_theme(fig, title="t", xaxis_title="x", yaxis_title="y")
        assert fig.layout.template is not None
        assert fig.layout.font.family == "serif"
        assert fig.layout.title.text == "t"
        assert fig.layout.xaxis.title.text == "x"
        assert fig.layout.yaxis.title.text == "y"


# ============================================================
# Helpers
# ============================================================
class TestSlugify:
    def test_basic(self) -> None:
        assert slugify("Hello, World!") == "hello__world"

    def test_strip_edges(self) -> None:
        assert slugify("  /path/  ") == "path"


class TestConfidenceEllipse:
    def test_returns_two_arrays(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 1, 50)
        rx, ry = confidence_ellipse_2d(x, y)
        assert len(rx) == len(ry) == 120  # default n_points

    def test_handles_too_few_points(self) -> None:
        rx, ry = confidence_ellipse_2d(np.array([1.0]), np.array([1.0]))
        assert len(rx) == 0 and len(ry) == 0

    def test_unsupported_confidence_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError, match=r"0\.95"):
            confidence_ellipse_2d(np.zeros(10), np.zeros(10), confidence=0.99)

    def test_ellipse_centred_on_mean(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(5.0, 1.0, 200)
        y = rng.normal(-3.0, 0.8, 200)
        rx, ry = confidence_ellipse_2d(x, y)
        # Mid-point of the ellipse should be near (mean_x, mean_y)
        assert abs(rx.mean() - x.mean()) < 0.1
        assert abs(ry.mean() - y.mean()) < 0.1


# ============================================================
# PathReport — accumulator + rendering
# ============================================================
class TestPathReport:
    def test_empty_report_writes(self, tmp_path: Path) -> None:
        rpt = PathReport(name="test_path", output_dir=tmp_path)
        out = rpt.write(export_png=False)
        assert out.exists() and out.name == "index.html"
        assert (tmp_path / "summary.json").exists()
        # No figures expected
        figs_dir = tmp_path / "figures"
        if figs_dir.exists():
            assert list(figs_dir.glob("*.html")) == []

    def test_unknown_section_raises(self, tmp_path: Path) -> None:
        import pytest
        rpt = PathReport(name="x", output_dir=tmp_path)
        with pytest.raises(KeyError, match="Unknown section"):
            rpt.add_metric("nope", "n_features", 10)

    def test_full_section_round_trips(self, tmp_path: Path) -> None:
        rpt = PathReport(
            name="path_A",
            output_dir=tmp_path,
            path_chain=["halfmin", "qc_rsc", "pqn", "log2", "pareto"],
        )
        rpt.add_section(
            "imputation",
            title="Half-min imputation",
            description="Replaces missing values with half the per-feature minimum.",
            method="halfmin",
            parameters={"factor": 0.5, "fallback_value": 0.0},
        )
        rpt.add_metric("imputation", "n_imputed_cells", 1602, unit="cells")
        rpt.add_metric(
            "imputation", "missing_rate", 0.73, unit="%",
            description="Fraction of zero entries pre-imputation",
        )
        fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[10, 20, 30])])
        apply_theme(fig, title="dummy", xaxis_title="x", yaxis_title="y")
        rpt.add_figure(
            "imputation",
            title="Demo figure",
            fig=fig,
            caption="Just a placeholder — real figures arrive in 8e.2.",
            method_metric="placeholder",
        )
        rpt.add_table(
            "imputation",
            title="Top-3 imputed features",
            df=pd.DataFrame(
                {"feature_id": ["F0", "F1", "F2"], "n_imputed": [10, 5, 2]}
            ),
        )
        out = rpt.write(export_png=False)

        # Inspect HTML
        html_text = out.read_text(encoding="utf-8")
        assert "Half-min imputation" in html_text
        assert "1602" in html_text
        assert "halfmin" in html_text
        assert "qc_rsc" in html_text  # path_chain rendered
        assert "Demo figure" in html_text
        assert "Top-3 imputed features" in html_text

        # Figure files
        figs_dir = tmp_path / "figures"
        html_figs = list(figs_dir.glob("*.html"))
        assert len(html_figs) == 1
        # PNG export was disabled for speed
        assert list(figs_dir.glob("*.png")) == []

        # Summary JSON
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["name"] == "path_A"
        assert summary["n_sections"] == 1
        assert summary["n_figures"] == 1
        assert summary["sections"][0]["method"] == "halfmin"
        assert summary["sections"][0]["parameters"]["factor"] == 0.5

    def test_multiple_sections_preserve_order(self, tmp_path: Path) -> None:
        rpt = PathReport(name="ordered", output_dir=tmp_path)
        for stage in ("a", "b", "c"):
            rpt.add_section(stage, title=stage.upper())
        out = rpt.write(export_png=False)
        text = out.read_text(encoding="utf-8")
        # Verify section order: A appears before B appears before C
        a_idx = text.index('id="a"')
        b_idx = text.index('id="b"')
        c_idx = text.index('id="c"')
        assert a_idx < b_idx < c_idx

    def test_table_truncation_note(self, tmp_path: Path) -> None:
        rpt = PathReport(name="trunc", output_dir=tmp_path)
        rpt.add_section("x", title="X")
        big = pd.DataFrame({"f": range(100), "v": range(100)})
        rpt.add_table("x", title="big", df=big, max_rows=5)
        out = rpt.write(export_png=False)
        text = out.read_text(encoding="utf-8")
        assert "showing first 5 of 100 rows" in text


# ============================================================
# PNG export (smoke — kaleido is heavy; one test only)
# ============================================================
class TestPngExport:
    def test_png_emitted_when_enabled(self, tmp_path: Path) -> None:
        rpt = PathReport(name="png", output_dir=tmp_path)
        rpt.add_section("x", title="X")
        fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[1, 2])])
        rpt.add_figure("x", title="t", fig=fig)
        rpt.write(export_png=True, png_scale=1)
        figs = list((tmp_path / "figures").glob("*.png"))
        assert len(figs) == 1, f"expected 1 PNG, got: {figs}"
