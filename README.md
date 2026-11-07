# Biomarker Discovery MLOps

A reproducible pipeline that turns raw **Progenesis QI** output (untargeted LC-MS metabolomics) into a short list of robust candidate biomarker features.

You give it four CSV files from Progenesis. It cleans the data, removes noise and contamination, annotates compounds, runs the statistics four different ways, and returns the handful of features that hold up no matter which analysis choices you make.

---

## What it does, in one picture

```
4 Progenesis CSVs
      │
      ▼
┌──────────────────────────────┐
│ PHASE 1 — Filter cascade      │   Stages 1–10  (CLI / DVC)
│ clean, de-noise, annotate     │   ~30,000 features → ~4,600 clean, vetted features
└──────────────────────────────┘
      │
      ▼
┌──────────────────────────────┐
│ PHASE 2 — Multiverse          │   Stages 11–17  (biodisc multiverse run)
│ 4 analysis philosophies +     │   ~4,600 features → a small consensus shortlist
│ statistics + consensus        │
└──────────────────────────────┘
      │
      ▼
Consensus shortlist  (the candidates that survive every analysis)
```

**Phase 1** is deterministic and fast (minutes). Every feature that is dropped is logged with the exact reason. **Phase 2** is the heavy statistical step (hours) — it runs the whole analysis four times with different, equally-defensible methods and keeps only what is robust to that choice.

> **Study this repo was built on:** 5 breast-cancer cell lines (MCF7, MCF10A, MDA-MB-231, SKBR3, UACC) — 31 biological injections + 16 QC pool injections + 3 blanks + 3 dilution points, acquired on a Waters Synapt G2-Si (ion-mobility, ESI+) on 2026-01-09.

---

## What you need

- **Python 3.12** (`.python-version` specifies 3.12; 3.11 may work but is not tested)
- **git**
- **DVC** (optional, but the easiest way to run Phase 1)
- Your four Progenesis QI CSV exports

---

## Install

```bash
# 1. Clone
git clone https://github.com/Ahmet-Kaplan/metabolomic_pipeline.git
cd metabolomic_pipeline

# 2. Create an isolated environment with uv (recommended) or pip
#    Using uv (fast, lock-file reproducible):
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv if needed
uv sync --frozen --all-extras   # exact versions from uv.lock
source .venv/bin/activate

#    Or using pip:
# python3 -m venv .venv
# source .venv/bin/activate
# pip install -e ".[dev,eda,enrichment]"

# 3. Check it works
biodisc --help
```

Or use Docker: `docker build -t metabolomic-pipeline . && docker run --rm metabolomic-pipeline --help`.

---

## Reproducing the published results

The repository contains every intermediate table of the published run, so the results can be checked at four depths:

| Depth | What it checks | Command | Time |
|---|---|---|---|
| 1 | Code behaves as specified | `uv run pytest -q tests` | ~1 min |
| 2 | Every derived table and figure follows from the committed multiverse outputs | run the six scripts listed in [`.github/workflows/ci.yml`](.github/workflows/ci.yml), then `git diff --exit-code -- 'data/**/*.csv' 'data/**/*.json'` (no diff = identical) | ~2 min |
| 3 | Phase 2 (multiverse + consensus) from the committed Stage-10 survivors | `biodisc multiverse run multiverse.output_dir=data/reports/multiverse_rerun` and compare `consensus.csv` with `data/reports/multiverse/consensus.csv` | hours |
| 4 | Phase 1 (filter cascade) from the raw Progenesis exports | download the raw exports (see *Data availability*) into `data/progenesisqi_output/`, build the references (Step 2), then `dvc repro` | ~30 min |

Depths 1 and 2 run automatically on every push (GitHub Actions).

---

## Step 1 — Put your data in

Place your four Progenesis QI exports in `data/progenesisqi_output/`:

```
data/progenesisqi_output/
├── I_66_CELL_METABOLOMICS_COMPOUND_MEASUREMENT(in).csv   # sample abundances
├── Blank(in).csv                                          # blank injections
├── dİLUTION(in).csv                                       # dilution series  (note the Turkish İ)
└── I_66_CELL_METABOLOMICS_COMPOUND_IDENTIFICATION.csv     # compound identifications
```

**Your filenames are different?** No problem — edit the four `parse.*_file` lines in [`conf/config.yaml`](conf/config.yaml).

**How sample names are read.** The parser reads the injection type and group straight from the sample names using a regex (`parse.sample_id_pattern` in `conf/config.yaml`). The default expects names like:

| Example name | Read as |
|--------------|---------|
| `09012026_MCF7_1`  | biological, group = MCF7, replicate 1 |
| `09012026_QC_3`    | QC pool injection |
| `09012026_Blank_1` | blank |
| `09012026_QC_1_2`  | dilution point (1:2) |

If your naming differs, update that one regex.

---

## Step 2 — Build the reference databases (once)

Three stages need external reference data: **CCS filter (8)**, **origin filter (10)**, **multiverse drift correction**, and **KEGG pathway enrichment**. Build them once.

```bash
# 2a. Download the open CCS reference (small). Add --include-hmdb for the HMDB XML (large, ~150 MB).
#     Note: HMDB requires manual download from https://hmdb.ca/downloads (CAPTCHA-protected).
#     Save as data/external/raw/hmdb_metabolites.zip.
python scripts/download_ccs_references.py --include-hmdb

# 2b. Build the reference tables
biodisc external build-hmdb-inchikey     # HMDB XML  → hmdb_inchikey.parquet
biodisc external build-hmdb-origin       # HMDB XML  → hmdb_origin.parquet        (Stage 10)
biodisc external build-hmdb-kegg         # HMDB XML  → hmdb_kegg.parquet          (KEGG enrichment)
biodisc external build-ccs-reference     # McLean+HMDB → ccs_reference_unified.parquet (Stage 8)

# 2c. Build the chronological run order (needed for drift correction in Phase 2)
#     Reads the Waters _HEADER.TXT acquisition files (path: run_order_headers_dir in conf/paths/default.yaml).
#     The run order of the published dataset is already committed as data/external/run_order.parquet.
biodisc external build-run-order
```

> If you don't have ion-mobility CCS values, the CCS filter (Stage 8) simply passes those features through — it never drops a feature it can't check. Same idea for origin: features with no annotation are kept.

---

## Step 3 — How the pipeline works, stage by stage

Every filter stage follows the same simple contract:

> **reads** the previous stage's surviving features → **applies one rule** → **writes** the survivors, a per-feature log of why each was kept or dropped, and an HTML report with plots.

| # | Stage | Command | The rule (plain English) | Default cut | Drops features? |
|---|-------|---------|--------------------------|-------------|:---------------:|
| 1 | Parse | `biodisc parse` | Read the 4 CSVs into clean tables | — | no |
| 2 | EDA | `biodisc eda` | Auto data-profile + 7 overview plots | — | no |
| 3 | Presence | `biodisc filter presence` | Keep a feature if it is detected in ≥ 80 % of **at least one** group | 80 % | yes |
| 4 | QC-CV | `biodisc filter qc-cv` | Drop features that are too noisy across the QC pool | CV ≤ 30 % | yes |
| 5 | D-ratio | `biodisc filter sample-qc-ratio` | Keep features where **biological** variation beats **technical** variation | ratio > 1 | yes |
| 6 | Blank | `biodisc filter blank` | Drop features that are mostly background (blank signal) | blank ≤ 20 % | yes |
| 7 | Dilution | `biodisc filter dilution` | Keep features whose signal scales with concentration (a straight line) | R² ≥ 0.70 | yes |
| 8 | CCS | `biodisc filter ccs` | Drop features whose ion-mobility size disagrees with the reference | error ≤ 2 % | yes |
| 9 | ID ranking | `biodisc filter id-ranking` | Pick the best compound match and grade its confidence (MSI L2/L3/L4) | Score ≥ 30, ppm ≤ 5 | **no — annotation only** |
| 10 | Origin | `biodisc filter origin` | Drop features that are purely **exogenous** (drug, plastic, food, serum) | — | yes |
| 11–17 | Multiverse | `biodisc multiverse run` | Run the whole analysis 4 ways and keep the consensus | see below | shortlists |

### A little more on each stage

- **Stage 1 — Parse.** Turns the four CSVs into tidy tables: features, samples, abundances (raw and Progenesis-normalised), blanks, dilution ladder, and the long identification table. Everything downstream reads these.
- **Stage 2 — EDA.** A look-before-you-leap report: missingness, distributions, QC behaviour, dilution response. It changes nothing; it just helps you sanity-check the data.
- **Stage 3 — Presence.** A real metabolite should be detected reliably in at least one group. Features seen only sporadically are dropped.
- **Stage 4 — QC-CV.** The QC pool is the same sample injected many times, so its signal *should* be stable. Features that wobble a lot across QCs (CV > 30 %) are unreliable and removed. This is usually the biggest cut.
- **Stage 5 — D-ratio.** Compares how much a feature varies between your samples vs. across the QCs. If biological variation isn't bigger than technical noise, the feature can't carry useful biology.
- **Stage 6 — Blank.** If a feature is almost as strong in the blank as in real samples, it's contamination or carry-over, not biology.
- **Stage 7 — Dilution.** Dilute the QC 1:2, 1:4, 1:8 — a true analyte's signal should fall in a straight line. Features that don't track concentration are artefacts.
- **Stage 8 — CCS.** Uses the ion-mobility "size" (collision cross section) as an extra identity check against a reference database. Features whose measured size disagrees by more than 2 % are dropped. Features with no reference entry pass through untouched.
- **Stage 9 — ID ranking.** For each feature, ranks the candidate compounds (best Score, then fragmentation, then mass accuracy) and assigns a confidence grade: **L2** (one strong match), **L3** (a tie / compound class), or **L4** (no confident match). It never drops a feature — it only labels them.
- **Stage 10 — Origin.** In cell-culture work, drug/plastic/food/serum signals are real peaks but not the biology you care about. If *every* candidate for a feature is flagged exogenous-only in HMDB, the feature is dropped. Unannotated features are kept (we don't drop on a check we can't make).

After every filter stage, a **master ledger** (`data/audit/master_feature_ledger.*`) is rebuilt automatically, showing each feature's fate across all stages in one table.

---

## Step 4 — Run Phase 1 (Stages 1–10)

**The easy way — one command (recommended):**

```bash
dvc repro
```

DVC runs Stages 1 → 10 in order, skips anything already up to date, and re-runs only what changed.

**Or run them yourself, one at a time:**

```bash
biodisc parse
biodisc eda
biodisc filter presence
biodisc filter qc-cv
biodisc filter sample-qc-ratio
biodisc filter blank
biodisc filter dilution
biodisc filter ccs            # needs the CCS reference from Step 2
biodisc filter id-ranking
biodisc filter origin         # needs the HMDB origin table from Step 2
```

---

## Step 5 — Run Phase 2 (the multiverse, Stages 11–17)

This is where the real discovery happens. It takes the clean features from Stage 10 and runs the **same analysis four times**, each with a different, well-established methodology ("philosophy"):

| Path | Philosophy | Imputation → Drift → Normalise → Transform → Scale |
|------|-----------|----------------------------------------------------|
| A | Classical | half-min → QC-RSC → PQN → log2 → Pareto |
| B | Robust ML | missForest → SERRF → MSTUS → glog → Pareto |
| C | MNAR-aware | QRILC → QC-RSC → PQN → glog → Pareto |
| D | Middle | NS-kNN → SERRF → median → log2 → Pareto |

Within each path it runs four feature-ranking methods — **one-way ANOVA** (with FDR correction), **PLS-DA**, **sPLS-DA**, and **Random Forest** — each backed by **bootstrap stability** and a **1000-label permutation test**. A feature must clear several methods to make a path's shortlist.

The **consensus** then keeps features that show up across paths:

- **Tier 1** — flagged in ≥ 75 % of paths (≥ 3 of 4)
- **Tier 2** — flagged in ≥ 50 % of paths (2 of 4)

It's a normal `biodisc` command like the filter stages — just config-driven and slow:

```bash
# Full run (4 paths, production settings from conf/multiverse/default.yaml). Takes several hours.
biodisc multiverse run
```

Everything is configurable in [`conf/multiverse/default.yaml`](conf/multiverse/default.yaml) and overridable on the command line:

```bash
biodisc multiverse run multiverse.n_permutation=200          # quicker, lower-resolution run
biodisc multiverse run multiverse.n_threads=1                # cap BLAS threads (desktop-friendly)
biodisc multiverse run multiverse.output_dir=data/reports/multiverse_test
```

> **Why it isn't part of `dvc repro`:** a full run takes hours, so it's a deliberate manual trigger — you never want a routine Stage-4 tweak to launch a multi-hour job. Run it explicitly, here or on a server.

If a long run is interrupted, resume only the unfinished paths:

```bash
python scripts/resume_multiverse_production.py
```

Results land in `data/reports/multiverse/` — open `index.html` for the interactive summary, or read `consensus_passed.parquet` for the shortlist.

### Post-multiverse analysis

After the multiverse completes, run these diagnostics to interpret the results:

```bash
# KEGG pathway enrichment for the consensus shortlist
uv sync --extra enrichment                    # install bioservices
biodisc multiverse enrichment                  # KEGG → pathway enrichment test

# Pipeline-wide permutation FDR estimate (how many discoveries are expected under null?)
python scripts/diagnose_pipeline_fdr.py --fast --n-permutations 500    # univariate pilot (minutes)
python scripts/diagnose_pipeline_fdr.py --multivariate --n-permutations 20  # full (hours)

# Compound cross-references: KEGG, ChEBI, ChEMBL, PubChem + m/z search for unknowns
uv run python scripts/diagnose_compound_cross_reference.py

# Rank consensus features for validation prioritization
python scripts/diagnose_candidate_prioritization.py
```

### Optional: restrict what enters the multiverse

Limit the multiverse to certain Stage-10 categories (e.g. annotated endogenous features only) with the `outcome_filter` config key — no code edits:

```bash
biodisc multiverse run                                                  # all surviving features (default)
biodisc multiverse run 'multiverse.outcome_filter=[endogenous_evidence,exo_or_unknown]'  # drop unidentified (L4)
biodisc multiverse run 'multiverse.outcome_filter=[endogenous_evidence]'                 # annotated endogenous only
```

Preview the effect of each setting before committing to a multi-hour run:

```bash
python scripts/diagnose_outcome_filter.py
```

---

## Where the results land

```
data/
├── interim/            # the cleaned tables at each stage (parquet)
│   ├── 01_parsed/      #   features, samples, abundances, blanks, dilution, identification
│   ├── 03_presence/ … 10_origin/   #   kept_features + metrics for each filter
├── audit/              # WHY each feature was kept/dropped, per stage
│   ├── 01_parse.* … 10_origin.*
│   └── master_feature_ledger.*     # every feature's fate across all stages
├── reports/            # human-readable HTML reports + plots
│   ├── 02_eda/ … 10_origin/
│   └── multiverse/     # the consensus shortlist, per-path reports, index.html
└── external/           # the reference databases you built in Step 2
```

---

## Reading the results

```bash
# Rebuild the cross-stage ledger by hand (it also rebuilds automatically after each filter)
biodisc audit ledger            # lean: kept/dropped per stage
biodisc audit ledger --full     # full: every metric column from every stage

# The multiverse shortlist
#   data/reports/multiverse/consensus_passed.parquet   ← the candidates
#   data/reports/multiverse/index.html                 ← interactive overview
```

> **Note on identity:** the multiverse shortlist lists features by ID (e.g. `10.76_1918.0639n`). The compound names and MSI grades live in the Stage 9 table (`data/interim/09_id_ranking/metrics.parquet`); join the two on `feature_id` to see what each candidate is.

---

## Changing thresholds

Every threshold lives in [`conf/filter/default.yaml`](conf/filter/default.yaml) and can be overridden on the command line — no code edits:

```bash
biodisc filter presence  filter.presence.presence_pct=0.70
biodisc filter qc-cv     filter.qc_cv.max_cv_pct=50
biodisc filter blank     filter.blank.max_blank_pct=50
biodisc filter dilution  filter.dilution.min_r2_linear=0.6
```

To exclude a bad injection from the whole pipeline, add its ID to `parse.exclude_samples` in `conf/config.yaml` and re-run — it disappears from every stage.

---

## Diagnostics & utilities

```bash
# Preview how each multiverse input-filter setting changes the feature count
python scripts/diagnose_outcome_filter.py

# Audit the experimental design: run-order confounding + pseudo-replication checks
python scripts/diagnose_replication_structure.py

# Pipeline-wide permutation FDR: estimate false discovery rate under the null
#   --fast: univariate-only pilot (minutes)   --multivariate: full 4 methods (hours)
python scripts/diagnose_pipeline_fdr.py --fast --n-permutations 500
python scripts/diagnose_pipeline_fdr.py --n-permutations 100

# Rank consensus features by composite score for validation prioritization
python scripts/diagnose_candidate_prioritization.py
```

---

## Project layout

```
.
├── conf/                  # all settings (Hydra): paths, thresholds, EDA, logging
├── data/
│   ├── progenesisqi_output/  # ← your 4 input CSVs go here
│   ├── interim/ audit/ reports/ external/   # pipeline outputs
├── docs/
│   └── pipeline_references.md  # the paper behind every stage
├── scripts/               # multiverse runner, downloaders, diagnostics
├── src/biodisc/
│   ├── io/                 # Progenesis parser + schemas
│   ├── filters/            # one module per filter stage (3–10)
│   ├── eda/                # EDA plots + profiling
│   ├── external/           # HMDB / CCS / run-order builders
│   └── multiverse/         # imputation, drift, normalise, stats, consensus
├── tests/                 # unit + integration tests
├── dvc.yaml               # Phase 1 pipeline definition (Stages 1–10)
└── pyproject.toml
```

---

## Tests

```bash
pytest tests/unit/ -q                 # fast unit tests
pytest tests/ -q                      # everything (needs parsed data)
pytest tests/unit/ --cov=src/biodisc  # with coverage
```

---

## The tools used

| Job | Tool |
|-----|------|
| Tables | pandas + pyarrow (Parquet) |
| Schema checks | pandera |
| Settings | Hydra |
| Pipeline orchestration | DVC |
| Logging | structlog |
| CLI | Typer + Rich |
| Statistics | scipy |
| Machine learning | scikit-learn |

---

## Good to know (honest notes)

- **`dvc repro` covers Stages 1–10 only.** Run the multiverse (Phase 2) separately with `biodisc multiverse run` (Step 5) — it's deliberately kept out of the auto DAG because it takes hours.
- **`biodisc run all` and `biodisc report` are placeholders** — they are not implemented yet. Use `dvc repro` (Phase 1) and `biodisc multiverse run` (Phase 2).
- **Build the reference databases (Step 2) before Stages 8, 10, and the multiverse**, or those steps will stop and tell you what's missing.
- **The multiverse is the slow part** (hours). Use `resume_multiverse_production.py` if a run is interrupted.
- **About replicates and claims.** This pipeline tells you which features *differ between your groups*. Whether that supports a biomarker *claim* depends on your study design — in particular, whether your replicates are independent biological samples or repeated injections of the same sample. Run `scripts/diagnose_replication_structure.py` before drawing biological conclusions, and validate any candidate in an independent cohort.
- **Single-batch data.** All injections were acquired on one instrument day (2026-01-09). Within-run drift correction handles temporal variation during that run, but no between-batch replicates exist. Results may not generalise to another instrument day without additional experiments.
- **Imputation ordering.** The multiverse pipeline applies imputation before drift correction (reverse of the standard metabolomics workflow). The impact is limited because QC samples have no missing values, so the drift model is not affected. However, this ordering should be validated in a sensitivity analysis before any biomarker claim.
- **Reference PDFs** are not redistributed; DOIs for every cited method are listed in [`docs/pipeline_references.md`](docs/pipeline_references.md).
- **Blank filter threshold.** The Stage 6 blank filter threshold (blank ≤ 20 % of sample+QC mean) is derived from the mQACC guideline (Broadhurst 2018) with adaptation by the project's domain expert. As noted in Broadhurst 2018, "no defined criteria have been reported or agreed as a standard" for blank filtering.

---

## References

Every stage's threshold and method is backed by a published source — see [`docs/pipeline_references.md`](docs/pipeline_references.md) for the full mapping.

---

## Data availability

Derived Parquet tables, per-stage audit ledgers and HTML reports for the published run are committed under `data/`. The raw Progenesis QI exports and Waters acquisition logs are being deposited in MetaboLights; the accession number will be added here.

## Code availability and licence

MIT licence (see [`LICENSE`](LICENSE)). Each release is archived on Zenodo.
