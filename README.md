<h1 align="center">
  <img src="./.github/assets/clinical-nlp-logo-light-mode.svg" alt="Clinical NLP Extraction" onerror="this.src='https://img.shields.io/badge/NLP_Extraction-Pipeline-blue?style=for-the-badge&logo=appveyor'"/>
</h1>

<p align="center">
  <i>A production-grade, two-pass LLM-based clinical NLP pipeline for extracting structured condition summaries from longitudinal patient notes.</i>
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="license"></a>
  <a href="./requirements.txt"><img src="https://img.shields.io/badge/Python-3.10%2B-informational.svg?style=flat-square" alt="python"></a>
  <a href="./Data/taxonomy.json"><img src="https://img.shields.io/badge/Taxonomy-Strict-success.svg?style=flat-square" alt="taxonomy strict"></a>
  <a href="./Data/problem_statement.md"><img src="https://img.shields.io/badge/Spec-Problem_Statement-important.svg?style=flat-square" alt="spec"></a>
  <img src="https://img.shields.io/badge/Status-Production_Ready-brightgreen.svg?style=flat-square" alt="Production status">
</p>

<hr>

## 📖 Introduction

This repository contains a **highly robust clinical NLP pipeline** designed to tackle the complex task of medical condition extraction across longitudinal clinical notes, as defined in the [Problem Statement](./Data/problem_statement.md).

Given a chronological series of medical notes for a patient (e.g., `text_0.md`, `text_1.md`), the system synthesizes a **comprehensive structured condition summary** mapping back to a strict clinical taxonomy.

### 🎯 Key Extractions Per Patient
- **Condition Identification:** Deep extraction mapped exactly to standard taxonomies (`Data/taxonomy.json`).
- **Clinical Status:** Determines if a condition is `active`, `resolved`, or `suspected` as of its *latest* mention.
- **Onset Date Detection:** Infers or explicitly extracts the earliest chronological anchor for when a condition manifested.
- **Verbatim Evidence:** Exact textual span extraction and linking (via `note_id` and `line_no`) grounding every condition in factual data.

## ⚙️ Architecture & Pipeline Workings

The system implements a resilient **two-pass processing architecture** ensuring high fidelity and robust fallback mechanisms against hallucinations.

### 1. Pass One: Per-Note Extraction (`map`)
Every note is parsed and sent through the LLM independently. This granular pass ensures:
- Deep contextual understanding of isolated encounters.
- Strict token-exact **evidence coercing**: Span text mapping exactly back to line numbers.
- Fuzzy taxonomy recovery: Overcoming minor LLM spelling mistakes using `rapidfuzz` to securely map to valid categories/subcategories.

### 2. Pass Two: Patient-Level Consolidation (`reduce`)
Extracted condition candidates from chronologically ordered encounters are dynamically unified:
- Resolves conflicts by propagating the most up-to-date **status** (based on document chronology).
- Infers absolute **onset** anchors across differing time references.
- **Deterministic Evidence Hardening:** Automatically spans the consolidated results against all notes to forcefully fill in missed evidence links missed by the LLM.

<p align="center">
  <img src="./report/assets/pipeline.png" alt="Architecture Pipeline diagram" />
</p>

## 📂 Repository Layout

```text
📦 NLP-Assignment
 ┣ 📂 Clinical_Nlp_Extraction/  # Core framework: pipeline, utils, prompts, validation
 ┃ ┣ 📜 data_loader.py          # Fast file parsing and sanitization
 ┃ ┣ 📜 extractor.py            # The 2-pass engine & deterministic fallback logic
 ┃ ┣ 📜 llm_client.py           # Robust OpenAI-compatible client wrapper
 ┃ ┗ 📜 schemas.py              # Pydantic schemas enforcing extraction structures
 ┣ 📂 Data/                     # Problem specs, taxonomy configurations & patient data
 ┣ 📂 report/                   # Generated evaluation reports and artifacts
 ┣ 📜 main.py                   # High-level pipeline CLI entrypoint
 ┣ 📜 requirements.txt          # Python dependencies
 ┗ 📜 README.md                 # Project documentation
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository and navigate inside
cd NLP-Assignment

# Install the required dependencies
pip install -r requirements.txt
```

### Environment Configurations

The pipeline natively interfaces with *any* **OpenAI API Compatible Endpoint** (e.g., standard OpenAI, OpenRouter, self-hosted vLLM). Set the context variables:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1" # Or custom endpoint
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini" # Or preferred model
```

## 🛠️ CLI Usage

The primary entry point is `main.py`. The system exposes different operational execution environments based on validation needs.

### 1. Perform Real Extraction (API Inference)
To perform actual API calls leveraging your selected LLM:

```bash
python main.py \
  --data-dir ./Data/dev \
  --patient-list ./patients_dev.json \
  --output-dir ./output_real \
  --cache-dir ./.cache_run \
  --temperature 0 \
  --concurrency 4
```
**Artifacts Generated**: Outputs `output_real/patient_XX.json` for every patient + `_manifest.json` tracker.

### 2. Perform a Dry-Run (Fast Wiring Test)
Ensure internal path wiring and validation mechanics operate correctly without hitting the billable LLM endpoint:

```bash
python main.py \
  --data-dir ./Data/dev \
  --patient-list ./patients_dev.json \
  --output-dir ./output_dry \
  --dry-run
```

### 3. Data Validation
To rigorously cross-examine the `output_real` extraction data strictly against the definitions mapped inside the `taxonomy.json`:

```bash
python -m Clinical_Nlp_Extraction.validate_outputs \
  --output-dir ./output_real \
  --taxonomy-path ./Data/taxonomy.json
```

## 🧪 Advanced Recovery Mechanisms
- **LLM Fail-Safes:** If LLM merging fails or returns junk during Pass Two (patient consolidation), the system gracefully regresses to an intelligent `_deterministic_fallback()` utilizing string deduplication and rule-based chronological prioritization to guarantee robust outputs.
- **Smart Evidence Linking:** Identifies instances when an LLM drops evidence, explicitly searching line histories using `rapidfuzz` to construct missing context.

## 🛡️ License & Contributing Notes

Distributed under the MIT License. See `LICENSE` for more information.

> **Important**: The `Data/` folder operates as the source of truth for problem criteria and must remain version-controlled. Outputs folders like `output*/` or `.cache*/` are safely ignored by git.
