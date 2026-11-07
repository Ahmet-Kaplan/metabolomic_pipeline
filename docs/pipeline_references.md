# Pipeline Stage References

Literature references for every stage of the biomarker discovery pipeline.
Last updated: 2026-05-17.

---

## Status at a glance

| | |
|---|---|
| Papers already in `data/papers/` | **6 papers** covering 8 stages |
| Papers **missing** from `data/papers/` | **15 papers** |

---

## Part 1 — Filtering pipeline (Stages 1–10)

### Stage 1 — Parse Progenesis QI CSVs

> Converts the four Progenesis QI CSV exports into typed parquet tables.

No academic citation required — this stage handles software output format only.
The instrument platform is a **Waters Synapt G2-Si** (TWIMS-IMS, ESI+).

---

### Stage 2 — Exploratory Data Analysis (EDA)

> PCA, QC drift plots, missingness heatmap, m/z × RT density, ydata-profiling reports.

| # | Reference | In `data/papers/`? |
|---|-----------|:-----------------:|
| 1 | Broadhurst D et al. (2018). *Guidelines and considerations for the use of system suitability and quality control samples in mass spectrometry assays applied in untargeted clinical metabolomic studies.* **Metabolomics** 14:72. https://doi.org/10.1007/s11306-018-1367-3 | ✅ `Broadhurst_et_al_Guidelines_considerations_Metabolomics.pdf` |
| 2 | Broadhurst DI & Kell DB (2006). *Statistical strategies for avoiding false discoveries in metabolomics and related experiments.* **Metabolomics** 2(4):171–196. https://doi.org/10.1007/s11306-006-0037-z | ✅ `broadhurst_kell_metabolomics06.pdf` |

---

### Stage 3 — 80 % Presence Rule

> Drops features absent in ≥ 80 % of samples in **every** biological group (modified per-group criterion).

| # | Reference | In `data/papers/`? |
|---|-----------|:-----------------:|
| 1 | **Bijlsma S et al. (2006).** *Large-scale human metabolomics studies: a strategy for data (pre-)processing and validation.* **Analytical Chemistry** 78(2):567–574. https://doi.org/10.1021/ac051495j | ❌ **Missing** |

> **Key quote from Bijlsma 2006:** Features are retained only if detected in a minimum fraction of samples within each class — the origin of the "80% rule" in metabolomics preprocessing.

---

### Stage 4 — QC-CV Filter

> Drops features with coefficient of variation (CV%) > 30 % across the 16 QC pool injections.

| # | Reference | In `data/papers/`? |
|---|-----------|:-----------------:|
| 1 | Broadhurst D et al. (2018). *Guidelines and considerations…* **Metabolomics** 14:72. | ✅ `Broadhurst_et_al_Guidelines_considerations_Metabolomics.pdf` |

> **Key quote from Broadhurst 2018, p.9:** *"The acceptance criterion for RSD is typically set to < 20 % (Sangster et al. 2006; Dunn et al. 2011b) or **< 30 %** (Lewis et al. 2016; Want et al. 2010)"*

Our threshold of **30 %** is directly cited in the mQACC guidelines.

---

### Stage 5 — Sample-RSD / QC-RSD Ratio (D-ratio)

> Drops features where biological variation ≤ technical (QC) variation (ratio ≤ 1).

| # | Reference | In `data/papers/`? |
|---|-----------|:-----------------:|
| 1 | Broadhurst D et al. (2018). *Guidelines and considerations…* **Metabolomics** 14:72. | ✅ `Broadhurst_et_al_Guidelines_considerations_Metabolomics.pdf` |

> **Key section:** Section 3, Equations 3–5. The D-ratio is formally defined as:
> `D-ratio = (s_QC / s_sample) × 100%`
> where a D-ratio approaching 0 % means the feature carries biological signal, not noise.

---

### Stage 6 — Blank Filter

> Drops features where `mean(blank) / mean(sample + QC) × 100` exceeds 20 %.

| # | Reference | In `data/papers/`? |
|---|-----------|:-----------------:|
| 1 | Broadhurst D et al. (2018). *Guidelines and considerations…* **Metabolomics** 14:72. | ✅ `Broadhurst_et_al_Guidelines_considerations_Metabolomics.pdf` |

> **Key quote from Broadhurst 2018, Section 2.2, p.4 (criterion b):**
> *"The signal in the blank sample is greater than a percentage of the average signal from the complete set of biological samples (e.g. a **5 % of median** acceptance criteria could be set)."*

> **Important note:** The paper explicitly states *"no defined criteria have been reported or agreed as a standard."* Our 20 % threshold equals the widely-used **5× sample-to-blank fold-change rule**. Whatever threshold is used, Broadhurst 2018 requires it to be **reported** — which our pipeline does automatically in the audit log.

---

### Stage 7 — Dilution Linearity (R² linear)

> Drops features with R²\_linear < 0.7 across the 4-point dilution series.

| # | Reference | In `data/papers/`? |
|---|-----------|:-----------------:|
| 1 | **Lewis MR et al. (2016).** *Development and application of UPLC-TOF MS for precision large scale urinary metabolic phenotyping.* **Analytical Chemistry** 88(18):9004–9013. https://doi.org/10.1021/acs.analchem.6b01481 | ❌ **Missing** |
| 2 | **Want EJ et al. (2010).** *Global metabolic profiling procedures for urine using UPLC-MS.* **Nature Protocols** 5(6):1005–1018. https://doi.org/10.1038/nprot.2010.50 | ❌ **Missing** |
| 3 | Broadhurst D et al. (2018). Section 6 — QC serial dilution. **Metabolomics** 14:72. | ✅ `Broadhurst_et_al_Guidelines_considerations_Metabolomics.pdf` |

---

### Stage 8 — CCS Error Filter

> Drops features whose best (candidate × adduct) match has CCS error > 3 % against the McLean CCS Compendium.

| # | Reference | In `data/papers/`? |
|---|-----------|:-----------------:|
| 1 | **Picache JA, Bhimsaria D et al. (2019).** *Collision cross section compendium to annotate and predict multi-platform ion mobility spectra of metabolites.* **Chemical Science** 10:983–993. https://doi.org/10.1039/C8SC04396E | ❌ **Missing** |
| 2 | **Hines KM et al. (2017).** *Characterization of the mechanisms of daptomycin resistance among Gram-positive bacterial pathogens by multidimensional lipidomics.* **mSphere** — (for TWIMS CCS methodology) | ❌ **Missing** |

> **Note:** The external reference database used is the **McLean CCS Compendium** (Picache et al. 2019), downloaded via `scripts/download_ccs_references.py`.

---

### Stage 9 — Identification Candidate Ranking / MSI Tiers

> Ranks Progenesis candidates per feature (Score → Frag → ppm cascade) and assigns an MSI identification confidence tier (L2 / L3 / L4). Annotation-only — zero feature drops.

| # | Reference | In `data/papers/`? |
|---|-----------|:-----------------:|
| 1 | **Sumner LW et al. (2007).** *Proposed minimum reporting standards for chemical analysis.* **Metabolomics** 3(3):211–221. https://doi.org/10.1007/s11306-007-0082-2 | ❌ **Missing** |
| 2 | **Schymanski EL et al. (2014).** *Identifying Small Molecules via High Resolution Mass Spectrometry: Communicating Confidence.* **Environmental Science & Technology** 48(4):2097–2098. https://doi.org/10.1021/es5002105 | ❌ **Missing** |

> **MSI tier definitions (Sumner et al. 2007):**
> - **L2** — Putatively annotated compound (unique candidate after ranking)
> - **L3** — Putatively characterised compound class (ambiguous, tied candidates)
> - **L4** — Unknown (no Progenesis candidate or no floor survivor)

---

### Stage 10 — Origin Filter

> Drops features whose **all** surviving Stage-9 candidates are pure-exogenous in the HMDB Source ontology (drugs, FBS components, plasticisers, food).

| # | Reference | In `data/papers/`? |
|---|-----------|:-----------------:|
| 1 | **Wishart DS et al. (2022).** *HMDB 5.0: the Human Metabolome Database for 2022.* **Nucleic Acids Research** 50:D622–D631. https://doi.org/10.1093/nar/gkab1062 | ❌ **Missing** |

---

## Part 2 — Multiverse analysis (Stages 11–17)

---

### Imputation

| Method | Reference | In `data/papers/`? |
|--------|-----------|:-----------------:|
| **Half-minimum** | Standard convention in metabolomics (no single canonical paper) | — |
| **NS-KNN** (non-parametric KNN) | **Troyanskaya O et al. (2001).** *Missing value estimation methods for DNA microarrays.* **Bioinformatics** 17(6):520–525. https://doi.org/10.1093/bioinformatics/17.6.520 | ❌ **Missing** |
| **QRILC** (left-censored / MNAR) | **Lazar C et al. (2016).** *Accounting for the Multiple Natures of Missing Values in Label-Free Quantitative Proteomics Data Sets to Compare Imputation Strategies.* **Journal of Proteome Research** 15(4):1116–1125. https://doi.org/10.1021/acs.jproteome.5b00981 | ❌ **Missing** |
| **missForest** (MNAR-aware RF) | **Stekhoven DJ & Bühlmann P (2012).** *MissForest — non-parametric missing value imputation for mixed-type data.* **Bioinformatics** 28(1):112–118. https://doi.org/10.1093/bioinformatics/btr597 | ❌ **Missing** |

---

### Drift Correction

| Method | Reference | In `data/papers/`? |
|--------|-----------|:-----------------:|
| **QC-RSC** (QC-based robust spline correction) | **Dunn WB et al. (2011).** *Procedures for large-scale metabolic profiling of serum and plasma using gas chromatography and liquid chromatography coupled to mass spectrometry.* **Nature Protocols** 6(7):1060–1083. https://doi.org/10.1038/nprot.2011.335 | ❌ **Missing** |
| **SERRF** (systematic error removal using random forest) | **Fan S et al. (2019).** *Systematic Error Removal Using Random Forest for Normalizing Large-Scale Untargeted Lipidomics Data.* **Analytical Chemistry** 91(5):3590–3596. https://doi.org/10.1021/acs.analchem.8b05592 | ❌ **Missing** |

---

### Normalisation

| Method | Reference | In `data/papers/`? |
|--------|-----------|:-----------------:|
| **Median** | Standard practice | — |
| **MSTUS** (median of total useful signal) | **Torgrip RJO et al. (2008).** *A note on normalization of biofluid 1D ¹H-NMR data.* **Metabolomics** 4:114–121. https://doi.org/10.1007/s11306-008-0109-5 | ❌ **Missing** |
| **PQN** (probabilistic quotient normalisation) | **Dieterle F et al. (2006).** *Probabilistic Quotient Normalization as Robust Method to Account for Dilution of Complex Biological Mixtures.* **Analytical Chemistry** 78(13):4281–4290. https://doi.org/10.1021/ac051632c | ❌ **Missing** |

---

### Transformation

| Method | Reference | In `data/papers/`? |
|--------|-----------|:-----------------:|
| **log2** | Standard practice — see Broadhurst & Kell (2006) | ✅ `broadhurst_kell_metabolomics06.pdf` |
| **glog** (generalised logarithm / variance-stabilising) | **Parsons HM et al. (2007).** *Improved classification accuracy in 1- and 2-dimensional NMR metabolomics data using the variance stabilising generalised logarithm transformation.* **BMC Bioinformatics** 8:234. https://doi.org/10.1186/1471-2105-8-234 | ❌ **Missing** |

---

### Scaling

| Method | Reference | In `data/papers/`? |
|--------|-----------|:-----------------:|
| **Pareto scaling** | **van den Berg RA et al. (2006).** *Centering, scaling, and transformations: improving the biological information content of metabolomics data.* **BMC Genomics** 7:142. https://doi.org/10.1186/1471-2164-7-142 | ❌ **Missing** |

---

### Univariate Statistics

| Test | Reference | In `data/papers/`? |
|------|-----------|:-----------------:|
| Mann-Whitney U, fold-change, FDR (Benjamini-Hochberg) | Broadhurst DI & Kell DB (2006). **Metabolomics** 2(4):171–196. | ✅ `broadhurst_kell_metabolomics06.pdf` |
| Univariate vs multivariate rationale | Saccenti E et al. (2013). *Reflections on univariate and multivariate analysis of metabolomics data.* **Metabolomics** 10(3):361–374. https://doi.org/10.1007/s11306-013-0598-6 | ✅ `Reflections on univariate and multivariate analysis of metabolomics data.pdf` |

---

### Multivariate — PLS-DA

| Reference | In `data/papers/`? |
|-----------|:-----------------:|
| **Westerhuis JA et al. (2008).** *Assessment of PLSDA cross validation.* **Metabolomics** 4(1):81–89. https://doi.org/10.1007/s11306-007-0099-6 | ✅ `Assessment_of_PLSDA_cross_validation.pdf` |
| Broadhurst DI & Kell DB (2006). **Metabolomics** 2(4):171–196. | ✅ `broadhurst_kell_metabolomics06.pdf` |
| Saccenti E et al. (2013). **Metabolomics** 10(3):361–374. | ✅ `Reflections on univariate and multivariate analysis of metabolomics data.pdf` |

---

### Multivariate — sPLS-DA

| Reference | In `data/papers/`? |
|-----------|:-----------------:|
| **Lê Cao KA et al. (2011).** *Sparse PLS discriminant analysis: biologically relevant feature selection and graphical displays for multiclass problems.* **BMC Bioinformatics** 12:253. https://doi.org/10.1186/1471-2105-12-253 | ❌ **Missing** |

---

### Multivariate — Random Forest

| Reference | In `data/papers/`? |
|-----------|:-----------------:|
| **Breiman L (2001).** *Random Forests.* **Machine Learning** 45:5–32. https://doi.org/10.1023/A:1010933404324 | ❌ **Missing** |

---

### Permutation Testing

| Reference | In `data/papers/`? |
|-----------|:-----------------:|
| Broadhurst DI & Kell DB (2006). **Metabolomics** 2(4):171–196. | ✅ `broadhurst_kell_metabolomics06.pdf` |
| Westerhuis JA et al. (2008). **Metabolomics** 4(1):81–89. | ✅ `Assessment_of_PLSDA_cross_validation.pdf` |

---

### Consensus Shortlist

| Reference | In `data/papers/`? |
|-----------|:-----------------:|
| Broadhurst DI & Kell DB (2006). **Metabolomics** 2(4):171–196. | ✅ `broadhurst_kell_metabolomics06.pdf` |

---

## Missing papers — download list

The following 15 papers are referenced by the pipeline but **not yet in `data/papers/`**.
Download and save each PDF using the DOI links below.

### High priority (core filter justifications)

| Paper | DOI |
|-------|-----|
| Bijlsma et al. 2006 — **80% presence rule** (Stage 3) | https://doi.org/10.1021/ac051495j |
| Sumner et al. 2007 — **MSI identification tiers** (Stage 9) | https://doi.org/10.1007/s11306-007-0082-2 |
| Dieterle et al. 2006 — **PQN normalisation** | https://doi.org/10.1021/ac051632c |
| van den Berg et al. 2006 — **Pareto scaling** | https://doi.org/10.1186/1471-2164-7-142 |
| Lê Cao et al. 2011 — **sPLS-DA** | https://doi.org/10.1186/1471-2105-12-253 |

### Medium priority

| Paper | DOI |
|-------|-----|
| Lewis et al. 2016 — **Dilution linearity** (Stage 7) | https://doi.org/10.1021/acs.analchem.6b01481 |
| Picache et al. 2019 — **McLean CCS Compendium** (Stage 8) | https://doi.org/10.1039/C8SC04396E |
| Wishart et al. 2022 — **HMDB 5.0** (Stage 10) | https://doi.org/10.1093/nar/gkab1062 |
| Dunn et al. 2011 — **QC-RSC drift correction** | https://doi.org/10.1038/nprot.2011.335 |
| Stekhoven & Bühlmann 2012 — **missForest** | https://doi.org/10.1093/bioinformatics/btr597 |
| Lazar et al. 2016 — **QRILC imputation** | https://doi.org/10.1021/acs.jproteome.5b00981 |
| Schymanski et al. 2014 — **Identification confidence levels** (Stage 9) | https://doi.org/10.1021/es5002105 |

### Lower priority

| Paper | DOI |
|-------|-----|
| Breiman 2001 — **Random Forest** | https://doi.org/10.1023/A:1010933404324 |
| Parsons et al. 2007 — **glog transformation** | https://doi.org/10.1186/1471-2105-8-234 |
| Troyanskaya et al. 2001 — **KNN imputation** | https://doi.org/10.1093/bioinformatics/17.6.520 |
