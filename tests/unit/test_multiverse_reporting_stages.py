"""Unit tests for the Phase 8e.2 stage section builders.

Each builder is exercised on a small synthetic dataset that mimics the
properties Path A's real data will exhibit (MNAR missingness, mild drift,
loading variation). We don't try to validate the *content* of every
figure — just:

  * the section is registered with the right method + parameters,
  * headline metrics land in the right slots,
  * figures are emitted (count and basic shape),
  * tables, when present, contain the expected columns.

End-to-end correctness is verified by ``scripts/smoke_8e2.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biodisc.multiverse._types import StageContext
from biodisc.multiverse.methods.multivariate.plsda import fit as _plsda_fit
from biodisc.multiverse.methods.multivariate.rf import fit as _rf_fit
from biodisc.multiverse.methods.multivariate.splsda import fit as _splsda_fit
from biodisc.multiverse.reporting import (
    PathReport,
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


# ============================================================
# Fixtures
# ============================================================
def _make_synth(
    *, with_run_order: bool = True, n_features: int = 20, n_per_class: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Build a deterministic feature × sample synthetic LC-MS-ish matrix."""
    rng = np.random.default_rng(0)
    cell_lines = ["MCF7", "MCF10A", "MDAMB231"]
    n_qc = 6
    sample_recs: list[dict] = []
    for cl in cell_lines:
        for r in range(1, n_per_class + 1):
            sample_recs.append(
                {
                    "sample_id": f"BIO_{cl}_{r}",
                    "raw_sample_id": f"raw_BIO_{cl}_{r}",
                    "injection_type": "biological",
                    "cell_line": cl,
                    "replicate": r,
                    "column_order": len(sample_recs),
                }
            )
    for q in range(1, n_qc + 1):
        sample_recs.append(
            {
                "sample_id": f"QC_{q}",
                "raw_sample_id": f"raw_QC_{q}",
                "injection_type": "qc",
                "cell_line": "QC",
                "replicate": q,
                "column_order": len(sample_recs),
            }
        )
    samples = pd.DataFrame(sample_recs)
    n_samples = len(samples)

    fids = [f"F{i:03d}" for i in range(n_features)]
    matrix = np.zeros((n_features, n_samples), dtype=float)
    for i in range(n_features):
        base = float(rng.lognormal(mean=8.0 + i * 0.05, sigma=0.7))
        for j in range(n_samples):
            inj = samples.iloc[j]["injection_type"]
            noise = 0.05 if inj == "qc" else 0.20
            matrix[i, j] = base * (1.0 + rng.normal(scale=noise))
    # MNAR missing on bottom 25% of features
    feature_means = matrix.mean(axis=1)
    bottom = np.argsort(feature_means)[: n_features // 4]
    for fi in bottom:
        miss_idx = rng.choice(n_samples, size=2, replace=False)
        matrix[fi, miss_idx] = np.nan
    matrix_df = pd.DataFrame(matrix, index=fids, columns=samples["sample_id"])

    run_order = (
        pd.DataFrame(
            {
                "raw_sample_id": samples["raw_sample_id"].tolist(),
                "position": list(range(1, n_samples + 1)),
            }
        )
        if with_run_order
        else None
    )
    return matrix_df, samples, run_order


def _impute_halfmin(matrix: pd.DataFrame) -> pd.DataFrame:
    """Tiny halfmin reproduced inline to keep the test independent of method modules."""
    out = matrix.copy()
    for fid in out.index:
        row = out.loc[fid]
        if row.isna().any():
            row_min = float(row.min(skipna=True))
            out.loc[fid] = row.fillna(0.5 * (row_min if np.isfinite(row_min) else 0.0))
    return out


# ============================================================
# Imputation section
# ============================================================
class TestImputationSection:
    def test_basic_section_lands(self, tmp_path: Path) -> None:
        matrix, samples, _ = _make_synth(with_run_order=False)
        imputed = _impute_halfmin(matrix)
        rpt = PathReport(name="imp", output_dir=tmp_path)
        add_imputation_section(
            rpt,
            before=matrix,
            after=imputed,
            samples=samples,
            method="halfmin",
            parameters={"factor": 0.5},
        )
        out = rpt.write(export_png=False)
        assert out.exists()

        # One section, expected metrics
        assert len(rpt.sections) == 1
        sec = rpt.sections[0]
        assert sec.name == "imputation"
        assert sec.method == "halfmin"
        metric_names = {m.name for m in sec.metrics}
        assert "n_missing_before" in metric_names
        assert "n_imputed" in metric_names
        # Imputation shouldn't leave anything missing on this fixture
        n_imputed = next(m for m in sec.metrics if m.name == "n_imputed").value
        assert n_imputed > 0

        # Multiple figures generated
        assert len(sec.figures) >= 4

    def test_shape_mismatch_raises(self, tmp_path: Path) -> None:
        matrix, samples, _ = _make_synth(with_run_order=False)
        bad = matrix.iloc[:5]  # wrong shape
        rpt = PathReport(name="x", output_dir=tmp_path)
        with pytest.raises(ValueError, match="shape mismatch"):
            add_imputation_section(
                rpt,
                before=matrix,
                after=bad,
                samples=samples,
                method="halfmin",
            )

    def test_treats_zero_as_missing_by_default(self, tmp_path: Path) -> None:
        # All-zero feature: pre-imputation = "all missing"; imputed = something else
        matrix = pd.DataFrame(
            {"S1": [0.0, 1.0], "S2": [0.0, 2.0], "S3": [0.0, 3.0]},
            index=["F0_zero", "F1"],
        )
        after = pd.DataFrame(
            {"S1": [0.5, 1.0], "S2": [0.5, 2.0], "S3": [0.5, 3.0]},
            index=["F0_zero", "F1"],
        )
        samples = pd.DataFrame(
            {
                "sample_id": ["S1", "S2", "S3"],
                "raw_sample_id": ["S1", "S2", "S3"],
                "injection_type": ["biological", "biological", "biological"],
                "cell_line": ["MCF7", "MCF7", "MCF7"],
                "replicate": [1, 2, 3],
                "column_order": [0, 1, 2],
            }
        )
        rpt = PathReport(name="z", output_dir=tmp_path)
        add_imputation_section(
            rpt,
            before=matrix,
            after=after,
            samples=samples,
            method="halfmin",
        )
        sec = rpt.sections[0]
        n_imp = next(m for m in sec.metrics if m.name == "n_imputed").value
        assert n_imp == 3, f"expected 3 zeros treated as missing-then-imputed, got {n_imp}"


# ============================================================
# Drift section
# ============================================================
class TestDriftSection:
    def test_basic_with_run_order(self, tmp_path: Path) -> None:
        matrix, samples, run_order = _make_synth()
        # Pretend imputed = the matrix + a fudged "after" with smaller QC variance
        imputed = matrix.fillna(matrix.mean(axis=1).fillna(0).iloc[0])
        # Build "after" by tightening the QC columns
        qc_cols = samples.loc[samples["injection_type"] == "qc", "sample_id"].tolist()
        corrected = imputed.copy()
        qc_means = corrected[qc_cols].mean(axis=1)
        corrected[qc_cols] = qc_means.values[:, None] * (
            1 + np.random.default_rng(1).normal(scale=0.01, size=(len(qc_means), len(qc_cols)))
        )

        rpt = PathReport(name="drift", output_dir=tmp_path)
        add_drift_section(
            rpt,
            before=imputed,
            after=corrected,
            samples=samples,
            run_order=run_order,
            method="qc_rsc",
            parameters={"smoothing_factor": "auto"},
            n_capped_features=2,
        )

        sec = rpt.sections[0]
        assert sec.method == "qc_rsc"
        metric_names = {m.name for m in sec.metrics}
        assert "D-ratio (before)" in metric_names
        assert "D-ratio (after)" in metric_names
        assert "n_features_drift_capped" in metric_names
        # Headline + trajectories + d-ratio + cv + 2 PCAs
        assert len(sec.figures) >= 5

    def test_no_run_order_skips_trajectories(self, tmp_path: Path) -> None:
        matrix, samples, _ = _make_synth(with_run_order=False)
        imputed = matrix.fillna(matrix.mean(axis=1).fillna(0).iloc[0])
        rpt = PathReport(name="drift_no_ro", output_dir=tmp_path)
        add_drift_section(
            rpt,
            before=imputed,
            after=imputed,  # no-op correction
            samples=samples,
            run_order=None,
            method="qc_rsc",
        )
        sec = rpt.sections[0]
        # run_order_available metric should reflect the gap
        ros = [m for m in sec.metrics if m.name == "run_order_available"]
        assert ros and ros[0].value == "no"


# ============================================================
# Normalisation section
# ============================================================
class TestNormalisationSection:
    def test_basic(self, tmp_path: Path) -> None:
        matrix, samples, _ = _make_synth(with_run_order=False)
        imputed = matrix.fillna(matrix.mean(axis=1).iloc[0])
        # Synthetic "normalised" — divide each column by its sum
        sums = imputed.sum(axis=0)
        norm = imputed.div(sums / sums.median(), axis=1)

        rpt = PathReport(name="norm", output_dir=tmp_path)
        add_normalisation_section(
            rpt,
            before=imputed,
            after=norm,
            samples=samples,
            method="pqn",
            parameters={"reference": "qc_median"},
        )
        sec = rpt.sections[0]
        assert sec.method == "pqn"
        metric_names = {m.name for m in sec.metrics}
        assert "PMSD (before)" in metric_names
        assert "PMSD (after)" in metric_names
        assert "factor median" in metric_names
        # Boxplots before + after, factor histo, paired-bar, CV histo
        assert len(sec.figures) >= 3
        # Top-10 table by |factor − 1|
        assert len(sec.tables) >= 1
        tbl = sec.tables[0]
        assert {"sample_id", "injection_type", "factor"} <= set(tbl.df.columns)


# ============================================================
# Transformation section
# ============================================================
class TestTransformationSection:
    def test_log2_records_metrics_and_figures(self, tmp_path: Path) -> None:
        matrix, samples, _ = _make_synth(with_run_order=False)
        # Use halfmin instead of mean-fill so feature variances aren't flattened
        imputed = _impute_halfmin(matrix)
        log_matrix = np.log2(imputed.clip(lower=1e-6))

        rpt = PathReport(name="tx", output_dir=tmp_path)
        add_transformation_section(
            rpt,
            before=imputed,
            after=log_matrix,
            samples=samples,
            method="log2",
            parameters={"epsilon": "auto"},
        )
        sec = rpt.sections[0]
        slope_before = next(
            m.value for m in sec.metrics if m.name == "heteroscedasticity slope (before)"
        )
        slope_after = next(
            m.value for m in sec.metrics if m.name == "heteroscedasticity slope (after)"
        )
        # Both slopes must be finite numbers — content correctness checked by smoke
        assert isinstance(slope_before, float) and np.isfinite(slope_before)
        assert isinstance(slope_after, float) and np.isfinite(slope_after)
        # All metric pair labels present (before / after for the 3 transformation metrics)
        labels = {m.name for m in sec.metrics}
        for stem in ("heteroscedasticity slope", "mean |skewness|", "KS-normal fraction"):
            assert f"{stem} (before)" in labels
            assert f"{stem} (after)" in labels
        # Heteroscedasticity scatter (×2) + skewness violin + mean histogram + headline = 5
        assert len(sec.figures) >= 4


# ============================================================
# Scaling section
# ============================================================
class TestScalingSection:
    def test_pareto_identity(self, tmp_path: Path) -> None:
        matrix, samples, _ = _make_synth(with_run_order=False)
        imputed = matrix.fillna(matrix.mean(axis=1).iloc[0])
        log_matrix = np.log2(imputed.clip(lower=1e-6))
        means = log_matrix.mean(axis=1)
        sds = log_matrix.std(axis=1, ddof=1).replace(0, 1)
        pareto = log_matrix.subtract(means, axis=0).divide(np.sqrt(sds), axis=0)

        rpt = PathReport(name="scl", output_dir=tmp_path)
        add_scaling_section(
            rpt,
            before=log_matrix,
            after=pareto,
            samples=samples,
            method="pareto",
        )
        sec = rpt.sections[0]
        # Mean-zero metric should be tiny
        mean_after = next(
            m.value
            for m in sec.metrics
            if m.name == "per-feature mean (after) — median |·|"
        )
        assert isinstance(mean_after, float)
        assert mean_after < 1e-4

        # Pareto identity figure should be present (3 figures total — mean check, SD KDE, identity)
        assert len(sec.figures) == 3


# ============================================================
# Outlier section
# ============================================================
def _make_synth_real_convention(
    *, n_features: int = 20, n_per_class: int = 4, n_qc: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Synthetic data using REAL-data convention: QC has cell_line=None.

    Mirrors what the Stage 1 parser produces on Progenesis output —
    biological rows have a cell_line, QC / blank / dilution rows have
    cell_line=None. Use this to catch QC-display bugs that the synthetic
    cell_line="QC" hack hides.
    """
    rng = np.random.default_rng(0)
    cell_lines = ["MCF7", "MCF10A", "MDAMB231"]
    sample_recs: list[dict] = []
    for cl in cell_lines:
        for r in range(1, n_per_class + 1):
            sample_recs.append(
                {
                    "sample_id": f"BIO_{cl}_{r}",
                    "raw_sample_id": f"raw_BIO_{cl}_{r}",
                    "injection_type": "biological",
                    "cell_line": cl,
                    "replicate": r,
                    "column_order": len(sample_recs),
                }
            )
    for q in range(1, n_qc + 1):
        sample_recs.append(
            {
                "sample_id": f"QC_{q}",
                "raw_sample_id": f"raw_QC_{q}",
                "injection_type": "qc",
                "cell_line": None,  # ← real-data convention
                "replicate": q,
                "column_order": len(sample_recs),
            }
        )
    samples = pd.DataFrame(sample_recs)
    n_samples = len(samples)
    fids = [f"F{i:03d}" for i in range(n_features)]
    matrix = np.zeros((n_features, n_samples), dtype=float)
    for i in range(n_features):
        base = float(rng.lognormal(mean=8.0 + i * 0.05, sigma=0.7))
        for j in range(n_samples):
            inj = samples.iloc[j]["injection_type"]
            noise = 0.05 if inj == "qc" else 0.20
            matrix[i, j] = base * (1.0 + rng.normal(scale=noise))
    matrix_df = pd.DataFrame(matrix, index=fids, columns=samples["sample_id"])
    return matrix_df, samples, None


class TestQcDisplayLabeling:
    """REGRESSION SUITE: Caught when real-data Path A showed 'None' as a class
    name where 'qc' should appear. The bug was that QC samples have
    cell_line=None in real data — synthetic test fixtures hid this by setting
    cell_line='QC'. These tests use the real-data convention to lock in
    correct behaviour."""

    def test_outlier_pca_labels_qc_correctly(self, tmp_path: Path) -> None:
        from biodisc.multiverse.reporting.theme import display_class

        matrix, samples, _ = _make_synth_real_convention(n_qc=4)
        # Verify the synthetic uses real-data convention
        qc_rows = samples[samples["injection_type"] == "qc"]
        assert qc_rows["cell_line"].isna().all() or (
            qc_rows["cell_line"].apply(lambda x: x is None)
        ).all()
        # display_class should resolve to "qc"
        labels = samples.apply(
            lambda r: display_class(r["cell_line"], r["injection_type"]), axis=1
        )
        assert (labels[samples["injection_type"] == "qc"] == "qc").all()
        assert (
            labels[samples["injection_type"] == "biological"]
            .isin({"MCF7", "MCF10A", "MDAMB231"})
        ).all()

        # Build Pareto-style matrix
        log_matrix = np.log2(matrix.clip(lower=1e-6))
        means = log_matrix.mean(axis=1)
        sds = log_matrix.std(axis=1, ddof=1).replace(0, 1)
        pareto = log_matrix.subtract(means, axis=0).divide(np.sqrt(sds), axis=0)

        rpt = PathReport(name="qc_label_test", output_dir=tmp_path)
        add_outlier_section(rpt, matrix=pareto, samples=samples)
        rpt.write(export_png=False)

        # Check the PCA-with-ellipses figure: should have a "qc" trace, not "None"
        import json
        import re
        pca_path = next((tmp_path / "figures").glob("outlier_01*.html"))
        text = pca_path.read_text()
        m = re.search(r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*(\[.*?\])\s*,\s*\{', text, re.DOTALL)
        traces = json.loads(m.group(1))
        names = {t.get("name") for t in traces}
        assert "qc" in names, f"expected 'qc' trace; got: {names}"
        assert "None" not in names, f"unexpected 'None' trace: {names}"
        assert "qc 95 % CI" in names, "expected QC ellipse trace"

    def test_outlier_t2_dmodx_includes_qc_projection(self, tmp_path: Path) -> None:
        matrix, samples, _ = _make_synth_real_convention(n_qc=4)
        log_matrix = np.log2(matrix.clip(lower=1e-6))
        means = log_matrix.mean(axis=1)
        sds = log_matrix.std(axis=1, ddof=1).replace(0, 1)
        pareto = log_matrix.subtract(means, axis=0).divide(np.sqrt(sds), axis=0)

        rpt = PathReport(name="qc_proj_test", output_dir=tmp_path)
        add_outlier_section(rpt, matrix=pareto, samples=samples)
        rpt.write(export_png=False)

        import json
        import re
        scatter_path = next((tmp_path / "figures").glob("outlier_00*.html"))
        text = scatter_path.read_text()
        m = re.search(r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*(\[.*?\])\s*,\s*\{', text, re.DOTALL)
        traces = json.loads(m.group(1))
        names = {t.get("name") for t in traces}
        assert any("qc" in str(n).lower() and "projected" in str(n).lower() for n in names), (
            f"expected 'qc (projected)' trace; got: {names}"
        )

    def test_project_qc_disabled(self, tmp_path: Path) -> None:
        """When ``project_qc=False`` no QC projection should appear."""
        matrix, samples, _ = _make_synth_real_convention(n_qc=4)
        log_matrix = np.log2(matrix.clip(lower=1e-6))
        means = log_matrix.mean(axis=1)
        sds = log_matrix.std(axis=1, ddof=1).replace(0, 1)
        pareto = log_matrix.subtract(means, axis=0).divide(np.sqrt(sds), axis=0)

        rpt = PathReport(name="no_qc_proj", output_dir=tmp_path)
        add_outlier_section(rpt, matrix=pareto, samples=samples, project_qc=False)
        rpt.write(export_png=False)

        import json
        import re
        scatter_path = next((tmp_path / "figures").glob("outlier_00*.html"))
        text = scatter_path.read_text()
        m = re.search(r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*(\[.*?\])\s*,\s*\{', text, re.DOTALL)
        traces = json.loads(m.group(1))
        names = {t.get("name") for t in traces}
        assert not any("projected" in str(n).lower() for n in names), (
            f"unexpected projection trace: {names}"
        )


class TestOutlierSection:
    def test_basic_outlier_section(self, tmp_path: Path) -> None:
        matrix, samples, _ = _make_synth(with_run_order=False, n_per_class=4)
        # Pareto-scale a complete matrix so outlier detection has clean input
        imputed = _impute_halfmin(matrix)
        log_matrix = np.log2(imputed.clip(lower=1e-6))
        means = log_matrix.mean(axis=1)
        sds = log_matrix.std(axis=1, ddof=1).replace(0, 1)
        pareto = log_matrix.subtract(means, axis=0).divide(np.sqrt(sds), axis=0)

        rpt = PathReport(name="outlier", output_dir=tmp_path)
        result = add_outlier_section(
            rpt,
            matrix=pareto,
            samples=samples,
        )
        # Section state
        sec = rpt.sections[0]
        assert sec.method == "simca_t2_dmodx"
        metric_names = {m.name for m in sec.metrics}
        for required in (
            "n_samples_analysed",
            "n_components",
            "T² critical (α)",
            "DModX critical (α)",
            "n_outliers_either",
        ):
            assert required in metric_names

        # Result shape
        n_bio = int((samples["injection_type"] == "biological").sum())
        assert len(result.sample_ids) == n_bio
        assert result.t_squared.shape == (n_bio,)
        assert result.dmodx.shape == (n_bio,)

        # Figures: scatter + PCA-with-ellipses + T² bar + DModX bar
        assert len(sec.figures) == 4

        # If any flagged, the table is added; if none, no table
        if result.flagged_either.any():
            assert len(sec.tables) == 1
            tbl = sec.tables[0]
            assert {"sample_id", "cell_line", "T²", "DModX", "flagged_by"} <= set(tbl.df.columns)
        else:
            assert len(sec.tables) == 0

        # HTML lands
        out = rpt.write(export_png=False)
        text = out.read_text(encoding="utf-8")
        assert 'id="outlier"' in text


# ============================================================
# Univariate section
# ============================================================
class TestUnivariateSection:
    def test_basic_univariate_section(self, tmp_path: Path) -> None:
        # Build a deliberately-signal-bearing log-transformed matrix:
        # 4 cell lines × 5 reps each, half features carry a strong group effect.
        rng = np.random.default_rng(0)
        cell_lines = ["MCF7", "MCF10A", "MDAMB231", "SKBR3"]
        n_per = 5
        sample_recs = []
        for cl in cell_lines:
            for r in range(1, n_per + 1):
                sample_recs.append(
                    {
                        "sample_id": f"BIO_{cl}_{r}",
                        "raw_sample_id": f"raw_BIO_{cl}_{r}",
                        "injection_type": "biological",
                        "cell_line": cl,
                        "replicate": r,
                        "column_order": len(sample_recs),
                    }
                )
        samples = pd.DataFrame(sample_recs)
        n_samples = len(samples)
        n_features = 24
        feat_ids = [f"F{i:03d}" for i in range(n_features)]
        # First half: signal features (boost a single group); second half: pure noise
        matrix = np.zeros((n_features, n_samples), dtype=float)
        for i in range(n_features):
            base = float(rng.normal(loc=10.0, scale=0.5))
            signal_group = cell_lines[i % len(cell_lines)] if i < n_features // 2 else None
            for j, row in enumerate(sample_recs):
                cl = row["cell_line"]
                noise = rng.normal(scale=0.30)
                boost = 2.0 if cl == signal_group else 0.0
                matrix[i, j] = base + boost + noise
        log_matrix = pd.DataFrame(matrix, index=feat_ids, columns=samples["sample_id"])

        rpt = PathReport(name="uv", output_dir=tmp_path)
        result = add_univariate_section(
            rpt,
            matrix=log_matrix,
            samples=samples,
        )
        sec = rpt.sections[0]
        assert sec.method == "anova_bh_eta_tukey"
        # Group count + signal detection sanity
        assert result.n_groups == 4
        assert result.n_features == n_features
        # We boosted 12 features by 2 SD — expect most of them to be q-significant
        n_sig = int((result.table["q_value"] < 0.05).sum())
        assert n_sig >= 6, f"signal features expected ≥ 6 q<0.05, got {n_sig}"

        # Headline metrics present
        names = {m.name for m in sec.metrics}
        for required in (
            "n_features",
            "n_groups",
            "n_significant (q<0.05)",
            "n_large_eta_squared (η²≥0.14)",
        ):
            assert required in names, f"missing metric {required}"

        # Figures: p-hist + η² + volcano + top-30 + Tukey heatmap = 5
        assert len(sec.figures) == 5
        # Top-N table
        assert len(sec.tables) == 1
        tbl = sec.tables[0]
        assert {"feature_id", "q_value", "eta_squared"} <= set(tbl.df.columns)

        # HTML lands
        out = rpt.write(export_png=False)
        text = out.read_text(encoding="utf-8")
        assert 'id="univariate"' in text

    def test_log_scale_override_is_passed(self, tmp_path: Path) -> None:
        """Regression: section must pass is_log_scale=True so log2 FC = mean_A − mean_B."""
        # Construct a log-transformed matrix where biological subset is all positive
        # (would confuse the auto-detect — exactly the bug we fixed).
        rng = np.random.default_rng(1)
        sample_recs = [
            {
                "sample_id": f"BIO_{cl}_{r}",
                "raw_sample_id": f"raw_BIO_{cl}_{r}",
                "injection_type": "biological",
                "cell_line": cl,
                "replicate": r,
                "column_order": idx,
            }
            for idx, (cl, r) in enumerate(
                [(c, r) for c in ("MCF7", "MCF10A", "MDAMB231") for r in (1, 2, 3, 4)]
            )
        ]
        samples = pd.DataFrame(sample_recs)
        n_samples = len(samples)
        log_matrix = pd.DataFrame(
            rng.normal(loc=15.0, scale=0.5, size=(10, n_samples)),  # all positive but logged
            index=[f"F{i:03d}" for i in range(10)],
            columns=samples["sample_id"],
        )
        # Boost MCF7 in a single feature
        mcf7_cols = samples.loc[samples["cell_line"] == "MCF7", "sample_id"].tolist()
        log_matrix.loc["F000", mcf7_cols] += 3.0

        rpt = PathReport(name="uv_lo", output_dir=tmp_path)
        result = add_univariate_section(rpt, matrix=log_matrix, samples=samples)

        # F000 should have a large positive log2 FC for MCF7 vs others (~3.0)
        # If the bug were present, the section would compute log2(mean_a/mean_b)
        # of values around ~15, giving log2 of ratios near 1 — much smaller than 3.
        row = result.table.set_index("feature_id").loc["F000"]
        # Check at least one MCF7-vs-other log2 FC column is near +3
        mcf7_pair_cols = [c for c in result.table.columns if "pair_MCF7_" in c and c.endswith("_log2_fc")]
        assert mcf7_pair_cols, "expected MCF7 pair columns"
        max_fc = max(abs(float(row[c])) for c in mcf7_pair_cols)
        assert max_fc > 2.0, (
            f"MCF7 boost of +3 in log space should produce log2 FC > 2; "
            f"got max |FC| = {max_fc:.3f} — log_scale override likely lost"
        )


# ============================================================
# Multivariate sections (PLS-DA, sPLS-DA, RF)
# ============================================================
def _make_multivariate_synth(
    *, n_features: int = 12, n_per_class: int = 4, signal_strength: float = 2.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small Pareto-style matrix with embedded class signal."""
    rng = np.random.default_rng(0)
    cell_lines = ["MCF7", "MCF10A", "MDAMB231"]
    sample_recs = [
        {
            "sample_id": f"BIO_{cl}_{r}",
            "raw_sample_id": f"raw_BIO_{cl}_{r}",
            "injection_type": "biological",
            "cell_line": cl,
            "replicate": r,
            "column_order": idx,
        }
        for idx, (cl, r) in enumerate(
            (cl, r) for cl in cell_lines for r in range(1, n_per_class + 1)
        )
    ]
    samples = pd.DataFrame(sample_recs)
    n_samples = len(samples)
    matrix = rng.normal(size=(n_features, n_samples))
    # Embed signal: features 0..2 boost MCF7, 3..5 boost MCF10A, 6..8 boost MDAMB231
    cell_arr = samples["cell_line"].to_numpy()
    for i in range(n_features):
        target = cell_lines[(i // 3) % len(cell_lines)] if i < 9 else None
        if target is not None:
            mask = cell_arr == target
            matrix[i, mask] += signal_strength
    log_matrix = pd.DataFrame(
        matrix,
        index=[f"F{i:03d}" for i in range(n_features)],
        columns=samples["sample_id"],
    )
    return log_matrix, samples


class TestMultivariateSections:
    def test_plsda_section(self, tmp_path: Path) -> None:
        matrix, samples = _make_multivariate_synth()
        ctx = StageContext(
            matrix=matrix, samples=samples,
            features=pd.DataFrame({"feature_id": matrix.index}),
        )
        result = _plsda_fit(ctx, n_bootstrap=8, n_permutation=20)
        rpt = PathReport(name="pls", output_dir=tmp_path)
        add_plsda_section(rpt, ctx=ctx, result=result)
        sec = rpt.sections[0]
        assert sec.method == "plsda"
        names = {m.name for m in sec.metrics}
        for required in ("n_VIP≥1", "n_stable (≥0.8)", "n_shortlist (VIP+stab+p)"):
            assert required in names
        # 5 figures: VIP bar + stability scatter + p-hist + score plot + β heatmap
        assert len(sec.figures) >= 4
        # Top-N table
        assert len(sec.tables) == 1

    def test_splsda_section(self, tmp_path: Path) -> None:
        matrix, samples = _make_multivariate_synth()
        ctx = StageContext(
            matrix=matrix, samples=samples,
            features=pd.DataFrame({"feature_id": matrix.index}),
        )
        result = _splsda_fit(ctx, n_bootstrap=8, n_permutation=20)
        rpt = PathReport(name="spls", output_dir=tmp_path)
        add_splsda_section(rpt, ctx=ctx, result=result)
        sec = rpt.sections[0]
        assert sec.method == "splsda"
        names = {m.name for m in sec.metrics}
        for required in ("n_selected", "sparsity %", "n_shortlist (sel+stab+p)"):
            assert required in names
        # The result table must carry the "selected" column
        assert "selected" in result.per_feature.columns

    def test_rf_section(self, tmp_path: Path) -> None:
        matrix, samples = _make_multivariate_synth(n_features=8)
        ctx = StageContext(
            matrix=matrix, samples=samples,
            features=pd.DataFrame({"feature_id": matrix.index}),
        )
        result = _rf_fit(ctx, n_bootstrap=4, n_permutation=10)
        rpt = PathReport(name="rf", output_dir=tmp_path)
        add_rf_section(rpt, ctx=ctx, result=result)
        sec = rpt.sections[0]
        assert sec.method == "rf"
        names = {m.name for m in sec.metrics}
        for required in ("median Gini", "median perm importance", "n_shortlist (stab+p)"):
            assert required in names
        # Should have at least 4 figures (Gini bar, stability scatter, p-hist,
        # Gini-vs-perm scatter); OOB trajectory is best-effort
        assert len(sec.figures) >= 4

    def test_section_method_name_validation(self, tmp_path: Path) -> None:
        """Each add_*_section must reject mismatched method names."""
        matrix, samples = _make_multivariate_synth()
        ctx = StageContext(
            matrix=matrix, samples=samples,
            features=pd.DataFrame({"feature_id": matrix.index}),
        )
        plsda = _plsda_fit(ctx, n_bootstrap=4, n_permutation=10)
        rpt = PathReport(name="x", output_dir=tmp_path)
        with pytest.raises(ValueError, match="splsda"):
            add_splsda_section(rpt, ctx=ctx, result=plsda)
        with pytest.raises(ValueError, match="rf"):
            add_rf_section(rpt, ctx=ctx, result=plsda)


# ============================================================
# Path summary
# ============================================================
class TestPathSummarySection:
    def test_basic_summary(self, tmp_path: Path) -> None:
        from biodisc.multiverse._types import StageContext as _Ctx
        from biodisc.multiverse.stats_univariate import compute_univariate

        # Build small set of inputs
        matrix, samples = _make_multivariate_synth(n_features=12)
        ctx = _Ctx(
            matrix=matrix, samples=samples,
            features=pd.DataFrame({"feature_id": matrix.index}),
        )
        univariate_result = compute_univariate(
            ctx, is_log_scale=True, tukey_post_hoc="auto"
        )
        plsda_result = _plsda_fit(ctx, n_bootstrap=4, n_permutation=10)
        splsda_result = _splsda_fit(ctx, n_bootstrap=4, n_permutation=10)
        rf_result = _rf_fit(ctx, n_bootstrap=4, n_permutation=10)

        summary = PathSummary(
            attrition_history=[
                ("input", 30),
                ("imputed", 30),
                ("filtered", 25),
                ("scaled", 22),
            ],
            runtime_history=[
                ("imputation", 0.5),
                ("drift_correct", 1.2),
                ("scale", 0.3),
            ],
            method_choices={
                "imputation": "halfmin",
                "drift": "qc_rsc",
                "normalisation": "pqn",
            },
            extra_params={
                "imputation": {"factor": 0.5},
                "drift": {"smoothing_factor": "auto"},
            },
        )

        rpt = PathReport(name="ps", output_dir=tmp_path)
        shortlists = add_path_summary_section(
            rpt,
            summary=summary,
            univariate_result=univariate_result,
            multivariate_results={
                "plsda": plsda_result,
                "splsda": splsda_result,
                "rf": rf_result,
            },
        )

        # Returned shortlists dict keyed by method
        assert set(shortlists.keys()) == {"univariate", "plsda", "splsda", "rf"}
        # All values are sets of feature_ids
        for sl in shortlists.values():
            assert isinstance(sl, set)

        sec = rpt.sections[0]
        assert sec.method == "path_summary"

        # Headline metrics
        names = {m.name for m in sec.metrics}
        for required in (
            "n_features_input",
            "n_features_final",
            "overall retention %",
            "total runtime (s)",
            "n_features_in_any_shortlist",
            "n_features_in_all_shortlists",
        ):
            assert required in names, f"missing metric {required}"

        # Per-method shortlist metrics
        for method in ("plsda", "splsda", "rf", "univariate"):
            assert f"shortlist size — {method}" in names

        # 4 figures expected: attrition + Jaccard + consensus bar + runtime
        assert len(sec.figures) == 4
        # 2 tables: config + per-method shortlist sizes
        assert len(sec.tables) == 2

        # HTML lands
        out = rpt.write(export_png=False)
        text = out.read_text(encoding="utf-8")
        assert 'id="path_summary"' in text

    def test_empty_inputs_dont_crash(self, tmp_path: Path) -> None:
        """The section must render gracefully with no multivariate results."""
        rpt = PathReport(name="empty", output_dir=tmp_path)
        shortlists = add_path_summary_section(
            rpt,
            summary=PathSummary(),
        )
        assert shortlists == {}
        # Section gets created with no figures (no data)
        assert len(rpt.sections) == 1
        # Renders cleanly
        out = rpt.write(export_png=False)
        assert out.exists()


# ============================================================
# End-to-end: 11 sections in one report
# ============================================================
class TestFullReportComposition:
    def test_seven_sections_compose(self, tmp_path: Path) -> None:
        matrix, samples, run_order = _make_synth()
        imputed = _impute_halfmin(matrix)
        rpt = PathReport(name="all6", output_dir=tmp_path, path_chain=["A", "B", "C", "D", "E", "F"])
        add_imputation_section(
            rpt, before=matrix, after=imputed, samples=samples, method="halfmin"
        )
        add_drift_section(
            rpt,
            before=imputed,
            after=imputed,
            samples=samples,
            run_order=run_order,
            method="qc_rsc",
        )
        add_normalisation_section(
            rpt, before=imputed, after=imputed, samples=samples, method="pqn"
        )
        log_matrix = np.log2(imputed.clip(lower=1e-6))
        add_transformation_section(
            rpt, before=imputed, after=log_matrix, samples=samples, method="log2"
        )
        means = log_matrix.mean(axis=1)
        sds = log_matrix.std(axis=1, ddof=1).replace(0, 1)
        pareto = log_matrix.subtract(means, axis=0).divide(np.sqrt(sds), axis=0)
        add_scaling_section(
            rpt, before=log_matrix, after=pareto, samples=samples, method="pareto"
        )
        add_outlier_section(rpt, matrix=pareto, samples=samples)
        add_univariate_section(rpt, matrix=log_matrix, samples=samples)
        out = rpt.write(export_png=False)
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        for sec_name in (
            "imputation", "drift", "normalisation", "transformation",
            "scaling", "outlier", "univariate",
        ):
            assert f'id="{sec_name}"' in text, f"section {sec_name} missing in HTML"
        # Sidebar nav entry too
        assert text.count("href=\"#imputation\"") >= 1
        # Check path_chain rendered
        for step in ("A", "B", "C", "D", "E", "F"):
            assert f"<span class=\"step\">{step}</span>" in text
