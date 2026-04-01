# Clinical Condition Extraction from Longitudinal Notes — Report (Draft)

This repository implements the pipeline described in `clinical_nlp_assignment/problem_statement.md`:
given all longitudinal notes for a patient, produce a structured JSON condition summary constrained to the provided taxonomy and supported by line-numbered evidence spans.

> This file is the Markdown version of the report. An Overleaf-ready LaTeX version lives in `report/overleaf/main.tex`.

## 1. Problem summary

**Goal.** For each patient, extract a condition inventory across all notes and output one JSON file `patient_XX.json` containing:

- `condition_name`: human-readable and as specific as possible
- `category` / `subcategory`: strict keys from `taxonomy.json`
- `status`: one of `active`, `resolved`, `suspected` (as of the latest note where the condition appears)
- `onset`: earliest explicit documentation date, or `null`
- `evidence`: exhaustive supporting excerpts across notes; each excerpt includes `note_id`, `line_no`, and `span` copied from the note text

**Constraints.**

- All LLM calls must be via an **OpenAI-compatible API**, using environment variables:
  - `OPENAI_BASE_URL`
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL`
- Do **not** hardcode model names/endpoints.
- Output must match schema and taxonomy strictly.

## 2. Data exploration notes

**Directory structure (as provided).**

- `clinical_nlp_assignment/train/patient_XX/text_N.md`
- `clinical_nlp_assignment/train/labels/patient_XX.json` (ground truth)
- `clinical_nlp_assignment/dev/patient_XX/text_N.md` (unlabeled)

**Chronology.** Notes are ordered by filename: `text_0.md` is earliest.

**Label schema.** Training labels match the target output schema and are used to understand taxonomy mapping and evidence formatting.

## 3. Approach & system design

The implemented system is a **two-stage LLM extraction pipeline** with strict post-validation:

### 3.1 Stage A — Per-note extraction

For each note:

1. Load the note with **1-indexed line numbers**.
2. Prompt an LLM to extract only taxonomy-valid conditions for this single note.
3. Enforce that `evidence.span` is copied verbatim from the referenced `line_no` (fallback to the full line if needed).
4. Validate each extracted condition against:
   - schema (Pydantic)
   - taxonomy keys (`category`, `subcategory`)
   - status values

This stage intentionally over-includes candidates to avoid missing conditions later.

### 3.2 Stage B — Patient-level consolidation

All per-note candidates are consolidated into a final patient summary:

- Dedupe synonymous conditions within the same taxonomy slot (light fuzzy match)
- **Status selection** guided by chronological note ordering: latest mention wins
- **Onset selection** guided by the spec’s priority rules (model instructed)
- **Evidence accumulation**: model instructed to include evidence from *every* note where mentioned

If the patient-level model output fails validation, the system falls back to a deterministic merge:

- Dedupe by normalized name + taxonomy slot
- Onset: earliest non-null onset among candidates
- Status: taken from the latest note where the condition key appears
- Evidence: union of evidence candidates

## 4. Ensuring “comprehensive evidence”

The spec requires: **Include evidence from every note where a condition is mentioned**.

Current implementation uses two safeguards:

1. **Instructional control** in the patient-level consolidation prompt: the model is explicitly required to include evidence from every note where the condition appears, using only provided candidates.
2. **Candidate supply**: we provide candidates from each note to the patient consolidator so it can “see” cross-note mentions.

### Recommended hardening (future improvement)

To reduce dependence on model compliance, add a deterministic post-check:

- For each predicted condition, scan all note lines for:
  - normalized condition name variants (string match)
  - common abbreviations/ICD tokens observed in training data
- If a mention is detected in a note without evidence, append an evidence entry referencing the matching line.

This adds recall for evidence completeness with minimal extra cost and no extra LLM calls.

## 5. Experiments & results (template)

This repo includes a lightweight evaluation helper for iterative development using the labeled training patients.

**Script:** `python -m clinical_nlp_assignment.train`

**Metric:** A practical dev metric computing Precision/Recall/F1 over condition inventory using:

- exact `category/subcategory` match
- fuzzy match over normalized `condition_name`

> Note: The official evaluation may differ; this is for internal iteration.

**Results (fill in after running):**

- Train macro Precision: `__`
- Train macro Recall: `__`
- Train macro F1: `__`

## 6. Reproducibility, speed, and cost

### 6.1 Caching

All LLM calls are cached on disk (`--cache-dir`), keyed by a hash of the prompt content.
Re-running on the same inputs avoids repeated token spend.

### 6.2 Determinism

Default temperature is `0.0` to reduce variance.

## 7. How to run

### 7.1 Install

```bash
pip install -r requirements.txt
```

### 7.2 Environment variables (provided by evaluator)

```bash
export OPENAI_BASE_URL="..."
export OPENAI_API_KEY="..."
export OPENAI_MODEL="..."
```

### 7.3 Run inference

```bash
python main.py \
  --data-dir ./clinical_nlp_assignment/dev \
  --patient-list ./patients.json \
  --output-dir ./output \
  --cache-dir ./.cache
```

### 7.4 Validate outputs (schema + taxonomy)

```bash
python -m clinical_nlp_assignment.validate_outputs \
  --output-dir ./output \
  --taxonomy-path ./clinical_nlp_assignment/taxonomy.json
```

## 8. What worked / what didn’t

### Worked

- Two-stage extraction reduces missed conditions in long notes.
- Taxonomy and schema validation prevents invalid outputs.
- Disk caching improves speed and cost substantially on re-runs.

### Didn’t / risks

- Some providers/models may not support `response_format={"type":"json_object"}`; if so, adjust the client to fall back to strict JSON prompting + parsing.
- Evidence completeness is still partially dependent on model compliance unless the deterministic post-check described above is added.

