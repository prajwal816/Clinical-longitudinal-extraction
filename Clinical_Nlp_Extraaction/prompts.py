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
        'Date formats: full="16 March 2026", month_year="March 2014", year_only="2014", unknown=null',
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
# Few-shot example (curated from real training data patterns)
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
9: **Other Diagnoses:**
10: - Hypothyroidism
11: - Status post cholecystectomy in 2010
12:
13: **Medical History:**
14: - Liver cirrhosis
15: - Idiopathic thrombocytopenia
16:
17: **Ultrasound abdomen:**
18: Image of liver cirrhosis. Hepatosplenomegaly. Moderate aortic sclerosis.
19:
20: **Lab results:**
21:
22:   Parameter        Result         Reference Range
23:   Hemoglobin       8.2 g/dL       14 - 18 g/dL
24:   RBC              2.7M /μL       4.5M - 5.9M /μL
25:
26: **Therapy:** Blood pressure well controlled on Ramipril 5 mg.
"""

_FEW_SHOT_OUTPUT = """\
{
  "conditions": [
    {
      "condition_name": "Arterial hypertension",
      "category": "cardiovascular",
      "subcategory": "hypertensive",
      "status": "active",
      "onset": "May 2014",
      "evidence": [{"note_id": "text_0", "line_no": 6, "span": "Arterial hypertension"}]
    },
    {
      "condition_name": "Non-insulin-dependent diabetes mellitus type II",
      "category": "metabolic_endocrine",
      "subcategory": "diabetes",
      "status": "active",
      "onset": "May 2014",
      "evidence": [{"note_id": "text_0", "line_no": 7, "span": "Non-insulin-dependent diabetes mellitus type II"}]
    },
    {
      "condition_name": "Hypothyroidism",
      "category": "metabolic_endocrine",
      "subcategory": "thyroid",
      "status": "active",
      "onset": "May 2014",
      "evidence": [{"note_id": "text_0", "line_no": 10, "span": "Hypothyroidism"}]
    },
    {
      "condition_name": "Liver cirrhosis",
      "category": "gastrointestinal",
      "subcategory": "hepatic",
      "status": "active",
      "onset": "May 2014",
      "evidence": [
        {"note_id": "text_0", "line_no": 14, "span": "Liver cirrhosis"},
        {"note_id": "text_0", "line_no": 18, "span": "Image of liver cirrhosis. Hepatosplenomegaly. Moderate aortic sclerosis."}
      ]
    },
    {
      "condition_name": "Idiopathic thrombocytopenia",
      "category": "hematological",
      "subcategory": "cytopenia",
      "status": "active",
      "onset": "May 2014",
      "evidence": [{"note_id": "text_0", "line_no": 15, "span": "Idiopathic thrombocytopenia"}]
    },
    {
      "condition_name": "Hepatosplenomegaly",
      "category": "gastrointestinal",
      "subcategory": "hepatic",
      "status": "active",
      "onset": "May 2014",
      "evidence": [{"note_id": "text_0", "line_no": 18, "span": "Image of liver cirrhosis. Hepatosplenomegaly. Moderate aortic sclerosis."}]
    },
    {
      "condition_name": "Aortic sclerosis",
      "category": "cardiovascular",
      "subcategory": "vascular",
      "status": "active",
      "onset": "May 2014",
      "evidence": [{"note_id": "text_0", "line_no": 18, "span": "Image of liver cirrhosis. Hepatosplenomegaly. Moderate aortic sclerosis."}]
    },
    {
      "condition_name": "Anemia",
      "category": "hematological",
      "subcategory": "cytopenia",
      "status": "active",
      "onset": "May 2014",
      "evidence": [
        {"note_id": "text_0", "line_no": 23, "span": "Hemoglobin       8.2 g/dL       14 - 18 g/dL"},
        {"note_id": "text_0", "line_no": 24, "span": "RBC              2.7M /μL       4.5M - 5.9M /μL"}
      ]
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
        f"{_FEW_SHOT_OUTPUT}\n\n"
        "Key observations from this example:\n"
        "- 'Status post cholecystectomy' is a surgical PROCEDURE, not a taxonomy condition → NOT extracted.\n"
        "- Hypothyroidism in 'Other Diagnoses' is a chronic ongoing condition → status 'active'.\n"
        "- Liver cirrhosis appears in Medical History AND on imaging → evidence from BOTH lines.\n"
        "- Hepatosplenomegaly is a SEPARATE condition from liver cirrhosis → separate entry.\n"
        "- Aortic sclerosis → cardiovascular.vascular (NOT structural).\n"
        "- Anemia is extracted from abnormal lab values (Hgb 8.2, low RBC) even though 'anemia' isn't explicitly named.\n"
        "- Onset uses the note's encounter date (May 2014) since no stated date exists for any condition.\n"
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

TASK: Extract ALL clinically significant diagnoses and findings from ONE clinical note that map to the provided taxonomy. Be thorough — it is better to over-include than to miss conditions.

## TAXONOMY (use ONLY these category/subcategory keys)

{taxonomy_full}

## STATUS VALUES

{status_info}

## DISAMBIGUATION RULES (follow these strictly)

{disambig}

## ONSET / DATE RULES

{onset_rules}

## EXTRACTION GUIDELINES

1. Extract conditions from ALL sections: Diagnoses, Other Diagnoses, Medical History, narrative text, imaging reports, lab results, physical examination findings.
2. Conditions in "Medical History" or preceded by "status post" / "history of" → status "resolved" UNLESS they are chronic/ongoing conditions being managed (e.g., hypothyroidism, diabetes, hypertension → these remain "active").
3. Conditions in "Diagnoses" or "Other Diagnoses" → status "active" unless explicitly noted as resolved/suspected.
4. For lab abnormalities significantly outside reference ranges that indicate a clinical condition (e.g., low hemoglobin → anemia, low platelets → thrombocytopenia, low lymphocytes → lymphopenia), extract the condition even if not explicitly named.
5. For imaging findings that name a condition (e.g., "liver cirrhosis" on CT, "cardiomegaly" on X-ray), extract the condition.
6. Do NOT extract symptoms alone (e.g., "cough", "fatigue") — only named diagnoses or clinically significant findings.
7. Do NOT extract surgical procedures (e.g., "cholecystectomy", "tracheotomy") — only the underlying conditions.
8. One entry per distinct condition. Separate entries for different anatomical sites (e.g., brain metastasis vs liver metastasis).
9. When a condition name includes a site or qualifier, preserve it (e.g., "Squamous cell carcinoma of the left tongue base", not just "Squamous cell carcinoma").

## OUTPUT FORMAT

Return ONLY valid JSON:
{{"conditions": [
  {{
    "condition_name": "Human-readable, specific name",
    "category": "exact_taxonomy_key",
    "subcategory": "exact_taxonomy_key",
    "status": "active|resolved|suspected",
    "onset": "date string or null",
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

TASK: Build a FINAL longitudinal condition summary for ONE PATIENT by consolidating condition mentions extracted from multiple clinical notes. This is the definitive output — it must be complete, accurate, and precisely formatted.

## TAXONOMY

{taxonomy_full}

## STATUS VALUES

{status_info}

## DISAMBIGUATION RULES

{disambig}

## ONSET / DATE RULES

{onset_rules}

## CONSOLIDATION RULES

1. **Deduplication**: If two or more candidates refer to the SAME underlying condition (e.g., "Diabetes mellitus type II" and "Non-insulin-dependent diabetes mellitus type II"; or "Tongue base carcinoma" and "Squamous cell carcinoma of the left tongue base"), merge them into ONE entry. Use the most specific/complete/descriptive name.
2. **Status**: Use the status from the chronologically LATEST note where the condition appears. Notes are provided in chronological order (text_0 = earliest, text_N = latest). If a condition appears as "suspected" in text_0 and "active" in text_3, report "active". If it appears "active" in text_1 but is never mentioned again, keep "active".
3. **Onset**: Use the EARLIEST explicit documentation date, following the onset priority rules above. If one candidate has a stated date (e.g., "first diagnosed 03/2021") and another has only a note date, prefer the stated date.
4. **Evidence MUST be comprehensive**: Include evidence from EVERY note where the condition is mentioned. If a condition appears in 6 notes, there MUST be evidence entries from all 6 notes. Do NOT drop evidence from earlier notes.
5. **Evidence spans**: Use ONLY the provided evidence spans verbatim. Do NOT edit, paraphrase, truncate, or combine spans.
6. **Separate entries**: Maintain separate entries for:
   - Different metastatic sites (brain metastasis vs liver metastasis)
   - Different anatomical sites of the same condition type
   - Distinct conditions even if related (e.g., liver cirrhosis vs hepatosplenomegaly)
7. **Do not add conditions**: Only consolidate conditions present in the evidence candidates. Do not invent new conditions.
8. **Date format**: Use "16 March 2026", "March 2014", "2014", or null.

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
