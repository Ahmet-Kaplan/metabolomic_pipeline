"""Unit tests for Stage 2 EDA plot modules.

Each plot module gets a smoke test on a tiny synthetic Stage1Data bundle: we
verify (a) the function runs without error, (b) HTML output exists, (c) the
file is non-empty HTML content.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biodisc.eda import (
    blank_sample_ratio,
    dilution_response,
    missingness,
    mz_rt_density,
    pca,
    qc_vs_sample,
    run_order_drift,
)
from biodisc.eda._loader import Stage1Data
from biodisc.eda.report import ReportEntry, build_index


# ================================================================
# Synthetic Stage-1 bundle
# ================================================================
@pytest.fixture
def synthetic_data() -> Stage1Data:
    """Tiny but realistic-shaped Stage-1 bundle.

    50 features × (3 MCF7 + 3 MCF10A + 4 QC + 2 Blank + 2 Dilution) = 14 injections.
    """
    rng = np.random.default_rng(42)
    n_features = 50
    feature_ids = [f"feat_{i:03d}" for i in range(n_features)]

    features = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "neutral_mass": rng.uniform(100, 800, n_features),
            "m_over_z": rng.uniform(100, 800, n_features),
            "charge": np.ones(n_features, dtype=int),
            "rt_min": rng.uniform(0.5, 14.0, n_features),
            "measured_ccs": rng.uniform(150, 300, n_features),
            "peak_width_min": rng.uniform(0.1, 0.4, n_features),
            "identifications_count": rng.integers(0, 5, n_features),
            "anova_p_progenesis": rng.uniform(0, 1, n_features),
            "max_fold_change_progenesis": rng.uniform(1, 10, n_features),
            "highest_mean_group": ["MCF7"] * n_features,
            "lowest_mean_group": ["MCF10A"] * n_features,
            "isotope_distribution": ["100 - 20"] * n_features,
            "maximum_abundance": rng.uniform(1e3, 1e5, n_features),
            "minimum_cv_pct": rng.uniform(5, 30, n_features),
        }
    )

    samples_rows = []
    for i, (group, typ) in enumerate(
        [
            ("MCF7", "biological"),
            ("MCF7", "biological"),
            ("MCF7", "biological"),
            ("MCF10A", "biological"),
            ("MCF10A", "biological"),
            ("MCF10A", "biological"),
            ("QC", "qc"),
            ("QC", "qc"),
            ("QC", "qc"),
            ("QC", "qc"),
            ("Blank", "blank"),
            ("Blank", "blank"),
            ("QC_1_2", "dilution"),
            ("QC_1_8", "dilution"),
        ]
    ):
        rep = (i % 3) + 1 if typ == "biological" else i + 1
        if typ == "qc":
            sid = f"QC_{i - 5}"
        elif typ == "blank":
            sid = f"Blank_{i - 9}"
        elif typ == "dilution":
            sid = group
        else:
            sid = f"{group}_{rep}"
        samples_rows.append(
            {
                "sample_id": sid,
                "raw_sample_id": f"01012026_{sid}",
                "injection_type": typ,
                "cell_line": group if typ == "biological" else None,
                "replicate": rep,
                "dilution_level": 2.0 if sid == "QC_1_2" else (8.0 if sid == "QC_1_8" else None),
                "acq_date": datetime(2026, 1, 1),
                "column_order": i,
            }
        )
    samples = pd.DataFrame(samples_rows)

    # Abundances long (only biological + QC go here — 10 injections)
    long_rows = []
    for _, s in samples[samples["injection_type"].isin(["biological", "qc"])].iterrows():
        for fid in feature_ids:
            long_rows.append(
                {
                    "feature_id": fid,
                    "sample_id": s["sample_id"],
                    "intensity": float(rng.uniform(100, 1e5)),
                }
            )
    abund_long = pd.DataFrame(long_rows)
    abund_wide = abund_long.pivot(
        index="feature_id", columns="sample_id", values="intensity"
    ).reset_index()

    # Blanks (raw block)
    blanks_rows = []
    for rep in (1, 2):
        for fid in feature_ids:
            blanks_rows.append(
                {
                    "feature_id": fid,
                    "blank_replicate": rep,
                    "block": "raw",
                    "intensity": float(rng.uniform(10, 100)),
                }
            )
            blanks_rows.append(
                {
                    "feature_id": fid,
                    "blank_replicate": rep,
                    "block": "normalised",
                    "intensity": float(rng.uniform(10, 100)),
                }
            )
    blanks = pd.DataFrame(blanks_rows)

    # Dilution ladder (raw block): include factor 1.0 for QCs + dilutions
    dil_rows = []
    for factor, level in [(1.0, 1), (0.5, 2), (0.25, 4), (0.125, 8)]:
        base = rng.uniform(100, 1000, n_features)
        for fid, b in zip(feature_ids, base, strict=False):
            dil_rows.append(
                {
                    "feature_id": fid,
                    "dilution_level": float(level),
                    "dilution_factor": factor,
                    "block": "raw",
                    "intensity": float(b * factor + rng.normal(0, b * 0.05)),
                }
            )
    dilution_ladder = pd.DataFrame(dil_rows)

    # Identification: one candidate per feature
    ident = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "candidate_idx": 0,
            "compound_id": [f"HMDB{i:07d}" for i in range(n_features)],
            "accepted": False,
            "adducts": ["M+H"] * n_features,
            "formula": ["CxHyOz"] * n_features,
            "score": rng.uniform(30, 80, n_features),
            "fragmentation_score": rng.uniform(20, 90, n_features),
            "mass_error_ppm": rng.uniform(-5, 5, n_features),
            "theoretical_ccs": np.nan,
            "dccs": np.nan,
            "isotope_similarity": rng.uniform(50, 99, n_features),
            "theoretical_isotope_distribution": ["100 - 20"] * n_features,
            "link": [""] * n_features,
            "description": ["x"] * n_features,
        }
    )

    return Stage1Data(
        features=features,
        samples=samples,
        abundances_raw_long=abund_long,
        abundances_raw_wide=abund_wide,
        identification=ident,
        blanks=blanks,
        dilution_ladder=dilution_ladder,
    )


# ================================================================
# Individual plot smoke tests
# ================================================================
class TestPlotSmoke:
    @pytest.mark.parametrize(
        "module, name",
        [
            (missingness, "01_missingness"),
            (pca, "02_pca"),
            (qc_vs_sample, "03_qc_vs_sample_box"),
            (mz_rt_density, "04_mz_rt_density"),
            (run_order_drift, "05_qc_drift"),
            (blank_sample_ratio, "06_blank_sample_ratio"),
            (dilution_response, "07_dilution_response"),
        ],
    )
    def test_plot_produces_html(
        self, synthetic_data: Stage1Data, tmp_path: Path, module, name: str
    ) -> None:
        # Skip PNG in tests — kaleido's Chromium startup adds ~2 s per test.
        out = module.plot(synthetic_data, output_dir=tmp_path, emit_png=False)
        assert "html" in out
        assert out["html"].exists()
        content = out["html"].read_text()
        assert "<html" in content.lower() or "plotly" in content.lower()
        assert out["html"].stat().st_size > 500  # non-trivial content


# ================================================================
# Index report builder
# ================================================================
class TestReportBuilder:
    def test_index_builds(self, tmp_path: Path) -> None:
        entries = [
            ReportEntry(
                key="01_x",
                title="Test",
                description="desc",
                html_href="plots/x.html",
                png_href=None,
            ),
        ]
        path = build_index(output_path=tmp_path / "index.html", entries=entries)
        assert path.exists()
        body = path.read_text()
        assert "<html" in body
        assert "Test" in body
        assert "plots/x.html" in body
