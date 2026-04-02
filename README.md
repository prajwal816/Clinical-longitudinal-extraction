<h1 align="center">
  <a href="./#gh-light-mode-only">
    <img src="./.github/assets/clinical-nlp-logo-light-mode.svg" alt="Clinical NLP Extraction" />
  </a>
  <a href="./#gh-dark-mode-only">
    <img src="./.github/assets/clinical-nlp-logo-dark-mode.svg" alt="Clinical NLP Extraction" />
  </a>
</h1>

<p align="center">
  <b>Production-grade clinical condition extraction from longitudinal patient notes using a two-pass LLM pipeline with deterministic hardening.</b>
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>&nbsp;
  <a href="./requirements.txt"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"></a>&nbsp;
  <a href="./Data/taxonomy.json"><img src="https://img.shields.io/badge/Taxonomy-13_Categories-2ea44f.svg?style=for-the-badge" alt="Taxonomy"></a>&nbsp;
  <a href="./Data/problem_statement.md"><img src="https://img.shields.io/badge/Spec-Problem_Statement-e05d44.svg?style=for-the-badge" alt="Problem Statement"></a>
</p>

<br/>

<p align="center">
  <img src="./report/assets/pipeline.png" alt="System Architecture — Two-Pass LLM Pipeline" width="85%" />
</p>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
  - [Pass 1 — Per-Note Extraction](#pass-1--per-note-extraction-map)
  - [Pass 2 — Patient-Level Consolidation](#pass-2--patient-level-consolidation-reduce)
  - [Pass 3 — Deterministic Evidence Hardening](#pass-3--deterministic-evidence-hardening)
- [Repository Structure](#repository-structure)
- [Module Reference](#module-reference)
- [Taxonomy & Schema](#taxonomy--schema)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
- [Usage](#usage)
  - [Run Inference](#1-run-inference-real-extraction)
  - [Dry-Run Mode](#2-dry-run-mode-no-api-calls)
  - [Validate Outputs](#3-validate-outputs)
  - [Evaluate on Training Data](#4-evaluate-on-training-data)
- [CLI Reference](#cli-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Design Decisions & Rationale](#design-decisions--rationale)
- [Data Format](#data-format)
- [Output Schema](#output-schema)
- [Performance & Optimization](#performance--optimization)
- [Git Conventions](#git-conventions)
- [License](#license)

---

## Overview

This repository implements a **production-grade clinical NLP pipeline** for the assignment defined in [`Data/problem_statement.md`](./Data/problem_statement.md). Given a patient's chronologically ordered clinical notes (`text_0.md`, `text_1.md`, ..., `text_N.md`), the system produces a **comprehensive structured condition summary** — one JSON file per patient — containing:

| Field | Description |
|-------|-------------|
| `condition_name` | Human-readable, clinically specific name |
| `category` / `subcategory` | Strict mapping to `taxonomy.json` (13 categories, 60+ subcategories) |
| `status` | `active`, `resolved`, or `suspected` — as of the **latest note** mentioning it |
| `onset` | Earliest explicit documentation date, following priority rules |
| `evidence` | Verbatim text spans with `note_id` + `line_no` grounding every extraction |

---

## System Architecture

The pipeline implements a resilient **three-stage processing architecture** — two LLM-driven passes followed by a deterministic hardening pass — ensuring high fidelity with robust fallback mechanisms.

```
┌──────────────────────────────────────────────────────────────────┐
│                  CLINICAL NLP EXTRACTION PIPELINE                │
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                      │
│  │ text_0  │    │ text_1  │    │ text_N  │   Patient Notes       │
│  └────┬────┘    └────┬────┘    └────┬────┘                       │
│       │              │              │                            │
│       ▼              ▼              ▼                            │
│  ┌─────────────────────────────────────────────────────┐         │
│  │         PASS 1: Per-Note Extraction (Map)           │         │
│  │  • LLM extracts conditions from each note           │         │
│  │  • Evidence span coercion (exact line matching)      │         │
│  │  • Fuzzy taxonomy recovery (rapidfuzz)               │         │
│  │  • Pydantic schema validation                        │         │
│  │  • Concurrent execution (ThreadPoolExecutor)         │         │
│  └───────────────────────┬─────────────────────────────┘         │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────┐         │
│  │       PASS 2: Patient Consolidation (Reduce)        │         │
│  │  • LLM merges candidates across all notes            │         │
│  │  • Deduplication (fuzzy name + taxonomy slot)         │         │
│  │  • Status: latest note wins                           │         │
│  │  • Onset: earliest stated date wins                   │         │
│  │  • Deterministic fallback if LLM output invalid       │         │
│  └───────────────────────┬─────────────────────────────┘         │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────┐         │
│  │     PASS 3: Evidence Hardening (Deterministic)      │         │
│  │  • Scan all notes for missing evidence links          │         │
│  │  • Fuzzy partial matching (threshold ≥ 90)            │         │
│  │  • Deduplicate by (note_id, line_no)                  │         │
│  │  • Zero additional LLM calls                          │         │
│  └───────────────────────┬─────────────────────────────┘         │
│                          │                                       │
│                          ▼                                       │
│                ┌──────────────────┐                               │
│                │ patient_XX.json  │   Structured Output           │
│                └──────────────────┘                               │
└──────────────────────────────────────────────────────────────────┘
```

### Pass 1 — Per-Note Extraction (`Map`)

Each clinical note is processed independently through the LLM. This stage:

1. **Loads** the note with 1-indexed line numbers for precise evidence tracking.
2. **Constructs a rich system prompt** (`prompts.py`) embedding:
   - Full taxonomy with descriptions, subcategories, and examples
   - Status value definitions with clinical signal keywords
   - Disambiguation rules verbatim from the taxonomy (e.g., heart failure by cause, diabetic complications under `metabolic_endocrine.diabetes`)
   - Onset priority rules and date formatting requirements
   - A carefully curated **few-shot example** demonstrating expected extraction behavior
3. **Validates & coerces** every extraction:
   - **Evidence span coercion** — ensures `evidence.span` is an exact substring of the referenced line; falls back to the full line text if the LLM paraphrased
   - **Fuzzy taxonomy recovery** — recovers near-miss taxonomy keys (e.g., `"primary"` → `"primary_malignancy"`) via `rapidfuzz` (threshold ≥ 75)
   - **Status correction** — fuzzy-matches invalid status strings to valid values
   - **Pydantic validation** — enforces schema constraints with graceful rejection of invalid entries

### Pass 2 — Patient-Level Consolidation (`Reduce`)

Per-note candidates are merged into a unified patient summary:

1. **Deduplication** — conditions with the same taxonomy slot and fuzzy-similar names (threshold ≥ 88) are merged, keeping the most descriptive name.
2. **Status resolution** — adopts the status from the chronologically **latest** note (`text_0` = earliest, `text_N` = latest).
3. **Onset resolution** — selects the **earliest** non-null onset date, following priority rules (stated date > note encounter date > relative date).
4. **Evidence accumulation** — the LLM is explicitly instructed to retain evidence from **every** note where the condition appears.
5. **Deterministic fallback** — if the LLM's consolidation output fails Pydantic validation, the system automatically falls back to a rule-based merge using deduplication, chronological status selection, and evidence union.

### Pass 3 — Deterministic Evidence Hardening

A zero-cost deterministic post-pass that strengthens evidence completeness:

1. For each predicted condition, **scans all note lines** for fuzzy matches against the normalized condition name (partial ratio ≥ 90).
2. **Adds missing evidence** from notes that the LLM omitted (at most one evidence entry per note).
3. **Deduplicates** evidence by `(note_id, line_no)` to prevent duplicates.
4. Uses exact note line text as the span — guaranteeing verbatim evidence.

---

## Repository Structure

```
NLP-Assignment/
├── main.py                          # CLI entrypoint — evaluator runs this
├── requirements.txt                 # Python dependencies
├── patients_dev.json                # Dev patient list [patient_02, patient_08, patient_15]
├── LICENSE                          # MIT License
├── .gitignore                       # Ignores outputs, caches, credentials
│
├── Clinical_Nlp_Extraction/         # Core Python package
│   ├── __init__.py                  # Package init
│   ├── data_loader.py               # Note/taxonomy/label loading with line indexing
│   ├── llm_client.py                # OpenAI-compatible client with retry & token tracking
│   ├── model.py                     # Client factory from environment variables
│   ├── prompts.py                   # Prompt construction (taxonomy, few-shot, rules)
│   ├── schemas.py                   # Pydantic models (Condition, Evidence, PatientOutput)
│   ├── extractor.py                 # Two-pass extraction engine + evidence hardening
│   ├── inference.py                 # Patient pipeline orchestrator (parallel execution)
│   ├── evaluate.py                  # Multi-dimensional metrics (P/R/F1, status, onset, evidence)
│   ├── train.py                     # Training set evaluation with detailed reporting
│   ├── validate_outputs.py          # Schema + taxonomy validation for output files
│   └── utils.py                     # Hashing, normalization, file I/O helpers
│
├── Data/
│   ├── problem_statement.md         # Assignment specification
│   ├── taxonomy.json                # Clinical condition taxonomy (13 categories)
│   ├── train/                       # Labeled training data
│   │   ├── patient_XX/              #   Patient notes (text_0.md, text_1.md, ...)
│   │   └── labels/                  #   Ground truth (patient_XX.json)
│   └── dev/                         # Unlabeled development data
│       └── patient_XX/              #   Patient notes
│
├── Report/
│   ├── REPORT.md                    # Detailed technical report (Markdown)
│   ├── make_figures.py              # Report figure generation script
│   ├── assets/                      # Generated figures (pipeline, distributions, metrics)
│   └── overleaf/                    # Overleaf-ready LaTeX report
│
└── .github/
    └── assets/                      # Logo SVGs (light/dark mode)
```

---

## Module Reference

| Module | Responsibility | Key Features |
|--------|---------------|--------------|
| [`main.py`](./main.py) | CLI entrypoint | Argument parsing, patient loop, dry-run mode, manifest generation |
| [`data_loader.py`](./Clinical_Nlp_Extraction/data_loader.py) | Data I/O | 1-indexed line loading, chronological note ordering, taxonomy/label parsing |
| [`llm_client.py`](./Clinical_Nlp_Extraction/llm_client.py) | LLM communication | OpenAI-compatible client, exponential backoff retry (4 attempts), `response_format` fallback, robust JSON extraction (fenced blocks, brace detection), cumulative token/latency tracking |
| [`model.py`](./Clinical_Nlp_Extraction/model.py) | Client factory | Builds `OpenAICompatibleClient` from `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` |
| [`prompts.py`](./Clinical_Nlp_Extraction/prompts.py) | Prompt engineering | Full taxonomy formatting, status/disambiguation/onset rules, few-shot example, compact key reference |
| [`schemas.py`](./Clinical_Nlp_Extraction/schemas.py) | Data validation | Pydantic v2 models — `Condition`, `Evidence`, `PatientOutput`, `Taxonomy`; strict field validators |
| [`extractor.py`](./Clinical_Nlp_Extraction/extractor.py) | Extraction engine | Per-note extraction, patient consolidation, fuzzy taxonomy recovery, evidence coercion & hardening, deduplication, deterministic fallback |
| [`inference.py`](./Clinical_Nlp_Extraction/inference.py) | Orchestration | `ThreadPoolExecutor`-based parallel per-note extraction, sequential consolidation, output writing |
| [`evaluate.py`](./Clinical_Nlp_Extraction/evaluate.py) | Evaluation | Multi-dimensional metrics: condition P/R/F1, status accuracy, onset accuracy (exact + partial), evidence recall & precision, macro-averaging |
| [`train.py`](./Clinical_Nlp_Extraction/train.py) | Training evaluation | End-to-end eval on labeled data, per-patient scoring, macro-average reporting, results JSON export |
| [`validate_outputs.py`](./Clinical_Nlp_Extraction/validate_outputs.py) | Output validation | Schema + taxonomy validation for all `patient_XX.json` files in a directory |
| [`utils.py`](./Clinical_Nlp_Extraction/utils.py) | Utilities | SHA-256 hashing, condition name normalization, markdown cleaning, JSON I/O, `ConditionKey` dataclass |

---

## Taxonomy & Schema

The system enforces a **strict clinical taxonomy** defined in [`taxonomy.json`](./Data/taxonomy.json):

| Category | Subcategories | Examples |
|----------|:---:|---------|
| `cancer` | 5 | Primary malignancy, metastasis, pre-malignant, benign, CUP |
| `cardiovascular` | 6 | Coronary, hypertensive, rhythm, vascular, structural, inflammatory |
| `infectious` | 5 | Bacterial, viral, fungal, parasitic, spirochetal |
| `metabolic_endocrine` | 7 | Diabetes, thyroid, genetic, nutritional, lipid, adrenal, pituitary |
| `neurological` | 6 | Cerebrovascular, traumatic, seizure, functional, degenerative, neuromuscular |
| `pulmonary` | 5 | Obstructive, acute respiratory, structural, occupational, cystic |
| `gastrointestinal` | 6 | Hepatic, biliary, upper GI, lower GI, inflammatory bowel, functional GI |
| `renal` | 4 | Renal failure, structural, glomerular, renovascular |
| `hematological` | 3 | Cytopenia, coagulation, hemoglobinopathy |
| `immunological` | 5 | Immunodeficiency, allergic, autoimmune, autoinflammatory, complement |
| `musculoskeletal` | 4 | Fracture, degenerative, crystal arthropathy, connective tissue |
| `toxicological` | 2 | Poisoning, environmental exposure |
| `dental_oral` | 2 | Dental, temporomandibular |

**Status values:** `active` · `resolved` · `suspected`

**Key disambiguation rules:**
- Heart failure → categorized by underlying **cause**, not as `cardiovascular.structural`
- Diabetic complications → always under `metabolic_endocrine.diabetes`, not the affected organ

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- Access to an **OpenAI-compatible LLM API** (OpenAI, OpenRouter, Azure OpenAI, vLLM, Ollama, etc.)

### Installation

```bash
git clone https://github.com/your-username/NLP-Assignment.git
cd NLP-Assignment
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ≥ 1.0.0 | OpenAI-compatible API client |
| `tqdm` | ≥ 4.60.0 | Progress bars |
| `tenacity` | ≥ 8.0.0 | Retry logic with exponential backoff |
| `pydantic` | ≥ 2.6.0 | Data validation and schema enforcement |
| `rapidfuzz` | ≥ 3.6.0 | Fuzzy string matching for taxonomy recovery |

### Environment Configuration

The pipeline reads three **required** environment variables:

```bash
# Standard OpenAI
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o"

# — OR — OpenRouter
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="sk-or-..."
export OPENAI_MODEL="meta-llama/llama-3-8b-instruct"

# — OR — Local (vLLM / Ollama)
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="not-needed"
export OPENAI_MODEL="mistral-7b-instruct"
```

> **Note:** Model names and endpoints are **never** hardcoded. The pipeline works with any provider exposing the `/chat/completions` endpoint.

---

## Usage

### 1. Run Inference (Real Extraction)

```bash
python main.py \
  --data-dir ./Data/dev \
  --patient-list ./patients_dev.json \
  --output-dir ./output_real \
  --cache-dir ./.cache_run \
  --temperature 0 \
  --concurrency 4
```

**Output:**
```
output_real/
├── patient_02.json        # Structured condition summary
├── patient_08.json
├── patient_15.json
└── _manifest.json         # Run metadata (model, patients, settings)
```

### 2. Dry-Run Mode (No API Calls)

Validates CLI wiring, output directory creation, and file formatting without incurring any LLM costs:

```bash
python main.py \
  --data-dir ./Data/dev \
  --patient-list ./patients_dev.json \
  --output-dir ./output_dry \
  --dry-run
```

**Expected:** Each output file contains `{"patient_id": "...", "conditions": []}` — schema-valid empty output.

### 3. Validate Outputs

Cross-validate all generated outputs against the Pydantic schema **and** taxonomy constraints:

```bash
python -m Clinical_Nlp_Extraction.validate_outputs \
  --output-dir ./output_real \
  --taxonomy-path ./Data/taxonomy.json
```

**Expected:** `Validated N files OK.`

### 4. Evaluate on Training Data

Run the full pipeline against labeled training data and compute multi-dimensional metrics:

```bash
python -m Clinical_Nlp_Extraction.train \
  --data-dir ./Data/train \
  --taxonomy-path ./Data/taxonomy.json \
  --results-json ./results.json \
  --verbose
```

**Output:** Per-patient scores + macro-averaged metrics + token usage summary + detailed `results.json`.

---

## CLI Reference

### `main.py` — Primary Entrypoint

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--data-dir` | ✅ | — | Path to data directory (`./Data/dev` or `./Data/train`) |
| `--patient-list` | ✅ | — | JSON file with list of patient IDs to process |
| `--output-dir` | ✅ | — | Directory for `patient_XX.json` output files |
| `--taxonomy-path` | ❌ | `./Data/taxonomy.json` | Path to taxonomy definition |
| `--cache-dir` | ❌ | `./.cache` | Directory for SHA-256-keyed LLM response cache |
| `--temperature` | ❌ | `0.0` | LLM sampling temperature |
| `--max-output-tokens` | ❌ | `4096` | Max tokens per LLM response |
| `--concurrency` | ❌ | `4` | Parallel threads for per-note extraction |
| `--verbose` | ❌ | `false` | Enable DEBUG-level logging |
| `--dry-run` | ❌ | `false` | Write empty outputs without LLM calls |

### `validate_outputs` — Schema Validator

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--output-dir` | ✅ | — | Directory containing `patient_XX.json` files |
| `--taxonomy-path` | ❌ | `./Data/taxonomy.json` | Path to taxonomy definition |

### `train` — Training Evaluation

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--data-dir` | ✅ | — | Path to training directory (must contain `labels/`) |
| `--taxonomy-path` | ✅ | — | Path to taxonomy definition |
| `--cache-dir` | ❌ | `./.cache` | LLM response cache directory |
| `--results-json` | ❌ | — | Path to save detailed results JSON |
| `--temperature` | ❌ | `0.0` | LLM sampling temperature |
| `--max-output-tokens` | ❌ | `4096` | Max tokens per LLM response |
| `--verbose` | ❌ | `false` | Enable DEBUG-level logging |

---

## Evaluation Metrics

The evaluation framework (`evaluate.py`) computes multi-dimensional metrics aligned with the official scoring criteria:

| Metric | Description | Matching |
|--------|-------------|----------|
| **Condition Precision** | Fraction of predicted conditions that are correct | Greedy matching by category + subcategory + fuzzy name (≥ 88) |
| **Condition Recall** | Fraction of ground-truth conditions found | Same matching logic |
| **Condition F1** | Harmonic mean of precision and recall | — |
| **Status Accuracy** | % of matched conditions with correct status | Exact string match |
| **Onset Accuracy (Exact)** | % of matched conditions with exact onset date | Full string comparison |
| **Onset Accuracy (Partial)** | % with at least the correct year | Year extraction via regex |
| **Evidence Recall** | Fraction of GT evidence `note_id`s covered per condition | Set intersection ratio |
| **Evidence Precision** | Fraction of predicted evidence `note_id`s that are in GT | Set intersection ratio |

All metrics are computed per-patient and then **macro-averaged** across the cohort.

---

## Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **Full taxonomy in prompts** | Including descriptions + examples (not just keys) significantly improves category assignment accuracy |
| **Disambiguation rules verbatim** | Ensures domain-specific logic (e.g., diabetic nephropathy → `metabolic_endocrine.diabetes`) is applied consistently |
| **Few-shot example** | Grounds output format and demonstrates nuanced extraction behavior (lab values → conditions, procedure exclusion) |
| **Fuzzy taxonomy recovery** | LLMs output near-miss keys ~5% of the time; recovery prevents silent data loss |
| **Evidence hardening** | Deterministically fills gaps for conditions mentioned across many notes without additional LLM cost |
| **Disk caching (SHA-256)** | Makes iterative development fast and cost-effective; identical prompts never re-invoke the API |
| **Deterministic fallback** | If the consolidation LLM output fails validation, a rule-based merge guarantees valid output |
| **Robust JSON parsing** | Handles fenced code blocks, markdown wrapping, and brace extraction for cross-model compatibility |
| **`response_format` fallback** | Auto-detects providers that don't support `{"type": "json_object"}` and falls back gracefully |
| **Temperature 0.0** | Minimizes variance across runs for reproducible extractions |
| **4096 max tokens** | Prevents truncation for patients with 20+ conditions |
| **Threaded parallelism** | Per-note extraction is embarrassingly parallel; 2–4× speedup on multi-note patients |

---

## Data Format

### Input Notes

Each note is a Markdown file representing a clinical document. Notes are ordered chronologically by filename (`text_0.md` = earliest). Typical sections include:

- **Diagnoses** — Primary diagnoses for the current encounter
- **Other Diagnoses** — Comorbidities and secondary conditions
- **Medical History** — Past conditions and prior procedures
- **Lab Results** — Laboratory values (may indicate unnamed conditions)
- **Imaging** — Radiology findings
- **Therapy and Progression** — Treatment course

### Patient List

A JSON array of patient IDs:

```json
["patient_02", "patient_08", "patient_15"]
```

---

## Output Schema

Each `patient_XX.json` conforms to:

```json
{
  "patient_id": "patient_02",
  "conditions": [
    {
      "condition_name": "Arterial hypertension",
      "category": "cardiovascular",
      "subcategory": "hypertensive",
      "status": "active",
      "onset": "March 2014",
      "evidence": [
        {
          "note_id": "text_0",
          "line_no": 6,
          "span": "Arterial hypertension"
        },
        {
          "note_id": "text_3",
          "line_no": 12,
          "span": "Known arterial hypertension under Ramipril 5mg"
        }
      ]
    }
  ]
}
```

**Schema enforcement:** All outputs are validated by Pydantic models (`schemas.py`) ensuring type safety, non-empty evidence, valid status literals, and positive line numbers.

---

## Performance & Optimization

| Feature | Implementation | Impact |
|---------|---------------|--------|
| **LLM Response Caching** | SHA-256 hash of full prompt → disk cache | Zero repeated API calls; enables fast iteration |
| **Parallel Extraction** | `ThreadPoolExecutor` with configurable `--concurrency` | 2–4× speedup for multi-note patients |
| **Retry with Backoff** | `tenacity` — 4 attempts, exponential jitter (1s–30s) | Resilience against transient API failures |
| **Token Tracking** | Cumulative prompt + completion + latency tracking | Full cost visibility per run |
| **Lazy Taxonomy Loading** | Taxonomy parsed once per patient, shared across notes | Minimal I/O overhead |
| **Deterministic Fallback** | Rule-based merge when LLM consolidation fails | Guaranteed valid output, zero wasted API cost |

---

## Git Conventions

**Tracked (push to repository):**
- `Data/` — Assignment spec, taxonomy, patient notes
- `Clinical_Nlp_Extraction/` — Pipeline source code
- `Report/` — Technical report and figures
- `main.py`, `requirements.txt`, `patients_dev.json`

**Ignored (via `.gitignore`):**
- `output*/`, `out_*/` — Generated predictions
- `.cache*/` — LLM response cache
- `.env` — API credentials
- `__pycache__/`, `*.pyc` — Python bytecode

---

## License

This project is licensed under the **MIT License** — see [`LICENSE`](./LICENSE) for details.

---

<p align="center">
  <sub>Built with ❤️ for clinical NLP research</sub>
</p>
