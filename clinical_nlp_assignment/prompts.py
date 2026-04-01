"""
Prompt construction for the clinical condition extraction pipeline.

Separates prompt engineering from extraction logic for clarity and maintainability.
Includes full taxonomy formatting, disambiguation rules, few-shot examples,
and explicit onset/date/status instructions.
"""

from __future__ import annotations

import json
from typing import Any

from .data_loader import get_valid_categories, get_valid_statuses


# ---------------------------------------------------------------------------
# Taxonomy formatting
# ---------------------------------------------------------------------------


def format_taxonomy_full(taxonomy: dict[str, Any]) -> str:
    """Format the full taxonomy with descriptions, subcategories, and examples.

    Produces a compact but complete reference that the LLM can use for
    accurate category/subcategory assignment.
    """
    lines: list[str] = []
    cats = taxonomy.get("condition_categories", {})
    for cat_key, cat_data in cats.items():
        desc = cat_data.get("description", "")
        lines.append(f"### {cat_key}")
        lines.append(f"  Description: {desc}")
        subcats = cat_data.get("subcategories", {})
        for sub_key, sub_desc in subcats.items():
            lines.append(f"  - {sub_key}: {sub_desc}")
        lines.append("")

    return "\n".join(lines)


def _format_status_values(taxonomy: dict[str, Any]) -> str:
    """Format status values with descriptions and signals."""
    lines: list[str] = []
    statuses = taxonomy.get("status_values", {})
    for status_key, status_data in statuses.items():
        desc = status_data.get("description", "")
        signals = status_data.get("signals", [])
        lines.append(f"- **{status_key}**: {desc}")
        if signals:
            lines.append(f"  Signals: {', '.join(signals)}")
    return "\n".join(lines)


def _format_disambiguation_rules(taxonomy: dict[str, Any]) -> str:
    """Format disambiguation rules verbatim from taxonomy."""
    rules = taxonomy.get("disambiguation_rules", [])
    notes = taxonomy.get("notes", [])
    lines: list[str] = []
    for r in rules:
        lines.append(f"- RULE: {r['rule']}")
        lines.append(f"  {r['explanation']}")
    for n in notes:
        lines.append(f"- NOTE: {n}")
    return "\n".join(lines)


def _format_onset_rules(taxonomy: dict[str, Any]) -> str:
    """Format onset/date rules from taxonomy."""
    date_fmt = taxonomy.get("date_format", {})
    rules = date_fmt.get("onset_rules", [])
    lines: list[str] = [
        "Date formats: full=\"16 March 2026\", month_year=\"March 2014\", year_only=\"2014\", unknown=null",
        "",
        "Onset priority rules:",
    ]
    for i, r in enumerate(rules, 1):
        lines.append(f"  {i}. {r}")
    return "\n".join(lines)


def _format_valid_keys_compact(taxonomy: dict[str, Any]) -> str:
    """Compact listing of valid category→subcategory keys and valid statuses."""
    cats = get_valid_categories(taxonomy)
    statuses = get_valid_statuses(taxonomy)
    return json.dumps({"valid_categories": cats, "valid_statuses": statuses}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Few-shot example
# ---------------------------------------------------------------------------

_FEW_SHOT_NOTE = """\
1: **Dear colleague,**
2:
3: We are reporting on Mr. Smith, born 01/15/1950, inpatient from 05/12/2014 to 05/20/2014.
4:
5: **Diagnoses:**
6: - Arterial hypertension
7: - Non-insulin-dependent diabetes mellitus type II
8:
9: **Medical History:**
10: - Hypothyroidism
11: - Status post cholecystectomy in 2010
12:
13: **Therapy:** Blood pressure well controlled on Ramipril 5 mg.
"""

_FEW_SHOT_OUTPUT = """\
{
  "conditions": [
    {
      "condition_name": "Arterial hypertension",
      "category": "cardiovascular",
      "subcategory": "hypertensive",
      "status": "active",
      "onset": null,
      "evidence": [{"note_id": "text_0", "line_no": 6, "span": "Arterial hypertension"}]
    },
    {
      "condition_name": "Non-insulin-dependent diabetes mellitus type II",
      "category": "metabolic_endocrine",
      "subcategory": "diabetes",
      "status": "active",
      "onset": null,
      "evidence": [{"note_id": "text_0", "line_no": 7, "span": "Non-insulin-dependent diabetes mellitus type II"}]
    },
    {
      "condition_name": "Hypothyroidism",
      "category": "metabolic_endocrine",
      "subcategory": "thyroid",
      "status": "active",
      "onset": null,
      "evidence": [{"note_id": "text_0", "line_no": 10, "span": "Hypothyroidism"}]
    }
  ]
}"""


def _build_few_shot_block() -> str:
    """Return a formatted few-shot example block."""
    return (
        "=== FEW-SHOT EXAMPLE ===\n"
        "INPUT NOTE (note_id: text_0):\n"
        f"{_FEW_SHOT_NOTE}\n"
        "EXPECTED OUTPUT:\n"
        f"{_FEW_SHOT_OUTPUT}\n"
        "NOTE: 'Status post cholecystectomy' is a surgical procedure, not a condition in the taxonomy, so it is NOT extracted.\n"
        "NOTE: Hypothyroidism in Medical History is marked 'active' because it is a chronic ongoing condition being managed, not a past event.\n"
        "=== END EXAMPLE ===\n"
    )


# ---------------------------------------------------------------------------
# Per-note extraction prompts
# ---------------------------------------------------------------------------


def build_note_system_prompt(taxonomy: dict[str, Any]) -> str:
    """Construct the system prompt for per-note condition extraction."""
    taxonomy_full = format_taxonomy_full(taxonomy)
    status_info = _format_status_values(taxonomy)
    disambig = _format_disambiguation_rules(taxonomy)
    onset_rules = _format_onset_rules(taxonomy)
    few_shot = _build_few_shot_block()

    return f"""\
You are an expert clinical information extraction system.

TASK: Extract ALL clinically significant diagnoses and findings from ONE clinical note that map to the provided taxonomy.

## TAXONOMY (use ONLY these category/subcategory keys)

{taxonomy_full}

## STATUS VALUES

{status_info}

## DISAMBIGUATION RULES (follow these strictly)

{disambig}

## ONSET / DATE RULES

{onset_rules}

## EXTRACTION GUIDELINES

1. Extract conditions from ALL sections: Diagnoses, Other Diagnoses, Medical History, narrative text, imaging reports, lab results.
2. Conditions in "Medical History" or preceded by "status post" / "history of" → status "resolved" unless later notes show recurrence.
3. Conditions in "Diagnoses" or "Other Diagnoses" → status "active" unless explicitly noted as resolved/suspected.
4. For lab abnormalities (e.g., low hemoglobin, low platelets), extract the clinical condition they indicate (e.g., anemia, thrombocytopenia) using the hematological category.
5. For imaging findings that indicate a condition (e.g., "liver cirrhosis" on CT), extract the condition.
6. Do NOT invent conditions not mentioned. Do NOT extract symptoms alone (e.g., "cough") — only named diagnoses or clinically significant findings.
7. One entry per distinct condition. Separate entries for different anatomical sites (e.g., brain metastasis vs liver metastasis).

## OUTPUT FORMAT

Return ONLY valid JSON:
{{"conditions": [
  {{
    "condition_name": "Human-readable, specific name",
    "category": "exact_taxonomy_key",
    "subcategory": "exact_taxonomy_key",
    "status": "active|resolved|suspected",
    "onset": "date or null",
    "evidence": [{{"note_id": "...", "line_no": N, "span": "exact text from note line"}}]
  }}
]}}

HARD CONSTRAINTS:
- evidence.span MUST be copied EXACTLY from the provided note line (no edits to spelling/case/punctuation).
- evidence.line_no MUST match the line number prefix shown.
- Only use category/subcategory keys from the taxonomy above.
- status MUST be one of: active, resolved, suspected.
- If onset cannot be determined from this note alone, use null.
- Include at least one evidence entry per condition.

{few_shot}"""


def build_note_user_prompt(
    note_id: str, note_text: str, taxonomy: dict[str, Any]
) -> str:
    """Construct the user message for per-note extraction."""
    keys_compact = _format_valid_keys_compact(taxonomy)
    return (
        f"Valid taxonomy keys (for reference): {keys_compact}\n\n"
        f"note_id: {note_id}\n"
        "NOTE (each line starts with line_no: ):\n"
        f"{note_text}\n"
    )


# ---------------------------------------------------------------------------
# Patient consolidation prompts
# ---------------------------------------------------------------------------


def build_patient_system_prompt(taxonomy: dict[str, Any]) -> str:
    """Construct the system prompt for patient-level consolidation."""
    taxonomy_full = format_taxonomy_full(taxonomy)
    status_info = _format_status_values(taxonomy)
    disambig = _format_disambiguation_rules(taxonomy)
    onset_rules = _format_onset_rules(taxonomy)

    return f"""\
You are an expert clinical information extraction system.

TASK: Build a longitudinal condition summary for ONE PATIENT by consolidating condition mentions across multiple notes.

## TAXONOMY

{taxonomy_full}

## STATUS VALUES

{status_info}

## DISAMBIGUATION RULES

{disambig}

## ONSET / DATE RULES

{onset_rules}

## CONSOLIDATION RULES

1. **Deduplication**: If two candidates refer to the same underlying condition (e.g., "Diabetes mellitus type II" and "Non-insulin-dependent diabetes mellitus type II"), merge them into ONE entry. Use the most specific/complete name.
2. **Status**: Use the status from the chronologically LATEST note where the condition appears. Notes are provided in chronological order (earliest to latest).
3. **Onset**: Use the EARLIEST explicit documentation date, following the onset priority rules above. If one candidate has a stated date and another has only a note date, prefer the stated date.
4. **Evidence**: MUST include excerpts from EVERY note where the condition is mentioned. Do NOT drop evidence from earlier notes. Use ONLY the provided evidence spans verbatim.
5. **Separate entries**: Maintain separate entries for different metastatic sites, different anatomical sites, or different conditions even if related.
6. **Do not add conditions**: Only consolidate conditions present in the evidence candidates. Do not invent new conditions.

## OUTPUT FORMAT

Return ONLY valid JSON:
{{"patient_id": "...", "conditions": [
  {{
    "condition_name": "Most specific name",
    "category": "exact_taxonomy_key",
    "subcategory": "exact_taxonomy_key",
    "status": "active|resolved|suspected",
    "onset": "earliest date or null",
    "evidence": [{{"note_id": "...", "line_no": N, "span": "exact text"}}]
  }}
]}}

HARD CONSTRAINTS:
- Only taxonomy-valid category/subcategory keys.
- evidence.span MUST be copied EXACTLY from the provided evidence candidates (NO edits).
- One entry per distinct condition.
- Evidence must cover ALL notes where the condition appears."""


def build_patient_user_prompt(
    patient_id: str,
    note_ids_in_order: list[str],
    candidates: list[dict[str, Any]],
    taxonomy: dict[str, Any],
) -> str:
    """Construct the user message for patient-level consolidation."""
    keys_compact = _format_valid_keys_compact(taxonomy)
    return (
        f"patient_id: {patient_id}\n"
        f"Valid taxonomy keys: {keys_compact}\n\n"
        "Notes in chronological order (earliest → latest):\n"
        f"{json.dumps(note_ids_in_order, ensure_ascii=False)}\n\n"
        "Evidence candidates (use ONLY these spans verbatim; include ALL notes where mentioned):\n"
        f"{json.dumps(candidates, ensure_ascii=False)}\n"
    )
