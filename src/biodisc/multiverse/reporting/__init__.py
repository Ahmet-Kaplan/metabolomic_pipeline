"""Multiverse path-report builder.

Each pipeline path through the multiverse (e.g. halfmin → QC-RSC → PQN →
log2 → Pareto → 3 multivariate methods) deserves its own publication-grade
HTML report — analogous to the per-stage reports we produce for the
upstream filter pipeline (Stages 02–10).

Usage::

    from biodisc.multiverse.reporting import PathReport

    rpt = PathReport(name="path_A", output_dir=Path("data/reports/multiverse/path_A"))
    rpt.add_section("imputation", title="Half-min imputation",
                    method="halfmin", parameters={"factor": 0.5})
    rpt.add_metric("imputation", "n_imputed_cells", 1602, unit="cells")
    rpt.add_figure("imputation", title="Missingness heatmap (pre/post)", fig=plotly_fig)
    rpt.write()  # → data/reports/multiverse/path_A/{index.html, figures/*.html, figures/*.png}

Each figure is exported as both an interactive ``.html`` (Plotly + CDN
plotly.js) AND a static ``.png`` (kaleido) for paper inclusion.
"""

from biodisc.multiverse.reporting.core import PathReport
from biodisc.multiverse.reporting.multiverse_index import build_multiverse_index
from biodisc.multiverse.reporting.stages import (
    PathSummary,
    add_drift_section,
    add_imputation_section,
    add_normalisation_section,
    add_outlier_section,
    add_path_summary_section,
    add_plsda_section,
    add_rf_section,
    add_scaling_section,
    add_splsda_section,
    add_transformation_section,
    add_univariate_section,
)
from biodisc.multiverse.reporting.theme import CLASS_COLORS, apply_theme

__all__ = [
    "CLASS_COLORS",
    "PathReport",
    "PathSummary",
    "add_drift_section",
    "add_imputation_section",
    "add_normalisation_section",
    "add_outlier_section",
    "add_path_summary_section",
    "add_plsda_section",
    "add_rf_section",
    "add_scaling_section",
    "add_splsda_section",
    "add_transformation_section",
    "add_univariate_section",
    "apply_theme",
    "build_multiverse_index",
]
