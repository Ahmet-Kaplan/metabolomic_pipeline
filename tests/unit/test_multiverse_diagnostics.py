"""Unit tests for multiverse/diagnostics.py — mQACC validation metrics."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from biodisc.multiverse import StageContext
from biodisc.multiverse.diagnostics import (
    DriftDiagnostics,
    LOQOResult,
    biological_silhouette,
    compute_drift_diagnostics,
    d_ratio,
    loqo_residuals_from_predictor,
    pca_qc_ellipse_area,
    pmsd,
    qc_cv_median,
    qc_cv_pct_passing,
)


# ============================================================
# Helpers
# ============================================================
def _make_matrix(
    n_features: int = 20,
    n_qc: int = 6,
    n_bio: int = 12,
    qc_noise: float = 0.05,
    bio_noise: float = 0.3,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build a controlled synthetic matrix for diagnostic testing.

    Parameters
    ----------
    qc_noise
        Fractional noise applied to QC samples (low = tight QCs).
    bio_noise
        Fractional noise applied to biological samples — should be much
        larger than qc_noise for D-ratio to be sensible.
    """
    rng = np.random.default_rng(seed)
    baseline = rng.lognormal(mean=8.0, sigma=1.0, size=n_features)
    n_total = n_qc + n_bio
    qc_ids = [f"QC_{i + 1}" for i in range(n_qc)]
    bio_ids = [f"S_{i + 1}" for i in range(n_bio)]
    sample_ids = qc_ids + bio_ids

    matrix_data = np.zeros((n_features, n_total))
    for j, sid in enumerate(sample_ids):
        if sid.startswith("QC_"):
            noise = 1.0 + qc_noise * rng.standard_normal(n_features)
        else:
            noise = 1.0 + bio_noise * rng.standard_normal(n_features)
        matrix_data[:, j] = baseline * np.abs(noise)

    matrix = pd.DataFrame(
        matrix_data,
        index=[f"F{i}" for i in range(n_features)],
        columns=sample_ids,
    )
    # Two cell-line groups for silhouette test
    cell_lines = (
        [None] * n_qc
        + ["A"] * (n_bio // 2)
        + ["B"] * (n_bio - n_bio // 2)
    )
    samples = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "raw_sample_id": sample_ids,
            "injection_type": ["qc"] * n_qc + ["biological"] * n_bio,
            "cell_line": cell_lines,
            "replicate": list(range(1, n_qc + 1)) + list(range(1, n_bio + 1)),
            "dilution_level": [None] * n_total,
            "acq_date": [datetime(2026, 1, 9)] * n_total,
            "column_order": list(range(n_total)),
        }
    )
    return matrix, samples, qc_ids, bio_ids


# ============================================================
# D-ratio
# ============================================================
class TestDRatio:
    def test_tight_qcs_low_d_ratio(self) -> None:
        matrix, _, qc_ids, bio_ids = _make_matrix(qc_noise=0.02, bio_noise=0.5)
        r = d_ratio(matrix, qc_ids, bio_ids)
        assert 0.0 < r < 0.3, f"expected D-ratio in (0, 0.3), got {r}"

    def test_loose_qcs_high_d_ratio(self) -> None:
        matrix, _, qc_ids, bio_ids = _make_matrix(qc_noise=0.5, bio_noise=0.5)
        r = d_ratio(matrix, qc_ids, bio_ids)
        assert r > 0.5

    def test_returns_nan_on_empty(self) -> None:
        empty = pd.DataFrame()
        assert np.isnan(d_ratio(empty, [], []))


# ============================================================
# PMSD + QC-CV
# ============================================================
class TestPMSD:
    def test_lower_qc_noise_lower_pmsd(self) -> None:
        m_low, _, qc_low, _ = _make_matrix(qc_noise=0.02, seed=1)
        m_high, _, qc_high, _ = _make_matrix(qc_noise=0.3, seed=1)
        assert pmsd(m_low, qc_low) < pmsd(m_high, qc_high)


class TestQCCV:
    def test_median_cv_correct(self) -> None:
        # Construct so we know the answer: feature with 4 QCs of [100, 100, 100, 100] → CV=0
        # Add a feature with [100, 110, 90, 100] → CV ≈ 8.16%
        matrix = pd.DataFrame(
            {
                "QC_1": [100.0, 100.0],
                "QC_2": [100.0, 110.0],
                "QC_3": [100.0, 90.0],
                "QC_4": [100.0, 100.0],
            },
            index=["F0", "F1"],
        )
        cv = qc_cv_median(matrix, ["QC_1", "QC_2", "QC_3", "QC_4"])
        # F0 CV=0; F1 CV ≈ 8.16%; median = (0+8.16)/2 = 4.08
        assert 3 < cv < 5

    def test_pct_passing_threshold(self) -> None:
        matrix, _, qc_ids, _ = _make_matrix(qc_noise=0.05)
        # With 5% noise, all features should easily pass 30% threshold
        passing = qc_cv_pct_passing(matrix, qc_ids, threshold=30.0)
        assert passing > 0.9


# ============================================================
# Biological silhouette
# ============================================================
class TestBiologicalSilhouette:
    def test_well_separated_groups_high_silhouette(self) -> None:
        # Two groups with very different baselines → high silhouette
        rng = np.random.default_rng(0)
        n_feat = 10
        group_a_baseline = rng.lognormal(mean=8.0, sigma=0.5, size=n_feat)
        group_b_baseline = rng.lognormal(mean=11.0, sigma=0.5, size=n_feat)
        sample_ids = [f"A_{i + 1}" for i in range(4)] + [f"B_{i + 1}" for i in range(4)]
        cols = []
        for sid in sample_ids:
            base = group_a_baseline if sid.startswith("A") else group_b_baseline
            noise = 1.0 + 0.05 * rng.standard_normal(n_feat)
            cols.append(base * noise)
        matrix = pd.DataFrame(
            np.array(cols).T,
            index=[f"F{i}" for i in range(n_feat)],
            columns=sample_ids,
        )
        samples = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "raw_sample_id": sample_ids,
                "injection_type": ["biological"] * len(sample_ids),
                "cell_line": ["A"] * 4 + ["B"] * 4,
                "replicate": [1, 2, 3, 4] * 2,
                "dilution_level": [None] * len(sample_ids),
                "acq_date": [datetime(2026, 1, 9)] * len(sample_ids),
                "column_order": list(range(len(sample_ids))),
            }
        )
        sil = biological_silhouette(matrix, samples)
        assert sil > 0.5, f"expected high silhouette, got {sil}"

    def test_returns_nan_on_single_group(self) -> None:
        matrix, samples, _, _ = _make_matrix()
        # Set all bio cell_lines to the same group → silhouette undefined
        samples.loc[samples["injection_type"] == "biological", "cell_line"] = "A"
        sil = biological_silhouette(matrix, samples)
        assert np.isnan(sil)


# ============================================================
# PCA QC ellipse area
# ============================================================
class TestPCAQCEllipseArea:
    def test_tight_qcs_smaller_area(self) -> None:
        m_tight, _, qc_tight, _ = _make_matrix(qc_noise=0.02, n_qc=8, seed=1)
        m_loose, _, qc_loose, _ = _make_matrix(qc_noise=0.3, n_qc=8, seed=1)
        a_tight = pca_qc_ellipse_area(m_tight, qc_tight)
        a_loose = pca_qc_ellipse_area(m_loose, qc_loose)
        assert a_tight < a_loose

    def test_returns_nan_with_too_few_qcs(self) -> None:
        matrix, _, qc_ids, _ = _make_matrix(n_qc=2)
        assert np.isnan(pca_qc_ellipse_area(matrix, qc_ids))


# ============================================================
# Full diagnostic stack
# ============================================================
class TestComputeDriftDiagnostics:
    def test_returns_dataclass_with_all_fields(self) -> None:
        matrix, samples, qc_ids, bio_ids = _make_matrix(n_qc=8)
        diag = compute_drift_diagnostics(matrix, samples, qc_ids, bio_ids)
        assert isinstance(diag, DriftDiagnostics)
        for v in [
            diag.d_ratio,
            diag.pmsd,
            diag.qc_cv_median,
            diag.qc_cv_pct_passing_30,
            diag.biological_silhouette,
            diag.pca_qc_ellipse_area,
        ]:
            assert not np.isnan(v), f"unexpected NaN in {v}"


# ============================================================
# LOQO residuals
# ============================================================
class TestLOQOResidualsFromPredictor:
    """Tests for the generic LOQO runner with a fake predictor."""

    @staticmethod
    def _ctx_with_run_order(
        matrix: pd.DataFrame, samples: pd.DataFrame, qc_ids: list[str], bio_ids: list[str]
    ) -> StageContext:
        n_total = len(qc_ids) + len(bio_ids)
        run_order = pd.DataFrame(
            {
                "position": list(range(1, n_total + 1)),
                "raw_sample_id": qc_ids + bio_ids,
                "acq_datetime": [datetime(2026, 1, 9)] * n_total,
                "header_file": [f"_HEADER({i}).TXT" for i in range(n_total)],
            }
        )
        return StageContext(
            matrix=matrix, samples=samples, features=pd.DataFrame(), run_order=run_order
        )

    def test_perfect_predictor_yields_zero_residuals(self) -> None:
        matrix, samples, qc_ids, bio_ids = _make_matrix(n_qc=6, n_bio=8, seed=2)
        ctx = self._ctx_with_run_order(matrix, samples, qc_ids, bio_ids)

        # Predictor that returns the true held-QC value → residual = 0
        def perfect(ctx_in: StageContext, held_qc: str) -> np.ndarray:
            return ctx_in.matrix[held_qc].to_numpy(dtype=float)

        result = loqo_residuals_from_predictor(ctx, predict_held_qc=perfect)
        assert isinstance(result, LOQOResult)
        assert result.n_qc == 6
        assert result.median_residual_pct == 0.0
        assert result.fraction_features_passing == 1.0

    def test_too_few_qcs_returns_nan_loqo(self) -> None:
        matrix, samples, qc_ids, bio_ids = _make_matrix(n_qc=3, n_bio=4, seed=3)
        ctx = self._ctx_with_run_order(matrix, samples, qc_ids, bio_ids)

        def trivial(ctx_in: StageContext, held_qc: str) -> np.ndarray:
            return np.zeros(ctx_in.matrix.shape[0])

        result = loqo_residuals_from_predictor(ctx, predict_held_qc=trivial)
        assert np.isnan(result.median_residual_pct)
        assert result.n_qc == 3

    def test_threshold_passes_when_predictor_close(self) -> None:
        matrix, samples, qc_ids, bio_ids = _make_matrix(n_qc=6, n_bio=8, seed=4)
        ctx = self._ctx_with_run_order(matrix, samples, qc_ids, bio_ids)

        # Predictor with 5% systematic error → all features pass 30% threshold
        def slightly_off(ctx_in: StageContext, held_qc: str) -> np.ndarray:
            return ctx_in.matrix[held_qc].to_numpy(dtype=float) * 1.05

        result = loqo_residuals_from_predictor(
            ctx, predict_held_qc=slightly_off, threshold_pct=30.0
        )
        # 5% error → median residual ≈ 5%
        assert 4.0 < result.median_residual_pct < 6.0
        assert result.fraction_features_passing == 1.0
