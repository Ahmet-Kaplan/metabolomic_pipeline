"""Per-stage section builders for the multiverse path report.

Each module in this subpackage exposes one entry point —
``add_<stage>_section(report, *, before, after, samples, method, parameters,
**stage_kwargs)`` — that adds the corresponding section to a
``PathReport`` with method-specific publication-grade Plotly figures,
mQACC metrics, and any auxiliary tables.

Stages 8e.2 covers the pre-multivariate transformations:

* ``imputation`` — half-min / QRILC / NS-kNN / missForest
* ``drift`` — QC-RSC / SERRF
* ``normalisation`` — PQN / MSTUS / median
* ``transformation`` — log2 / glog
* ``scaling`` — Pareto

Outlier (8e.3), univariate (8e.4), multivariate (8e.5) and the path-level
summary (8e.6) live alongside in this package once their phases land.
"""

from biodisc.multiverse.reporting.stages.drift import add_drift_section
from biodisc.multiverse.reporting.stages.imputation import add_imputation_section
from biodisc.multiverse.reporting.stages.multivariate import (
    add_plsda_section,
    add_rf_section,
    add_splsda_section,
)
from biodisc.multiverse.reporting.stages.normalisation import add_normalisation_section
from biodisc.multiverse.reporting.stages.outlier import add_outlier_section
from biodisc.multiverse.reporting.stages.path_summary import (
    PathSummary,
    add_path_summary_section,
)
from biodisc.multiverse.reporting.stages.scaling import add_scaling_section
from biodisc.multiverse.reporting.stages.transformation import add_transformation_section
from biodisc.multiverse.reporting.stages.univariate import add_univariate_section

__all__ = [
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
]
