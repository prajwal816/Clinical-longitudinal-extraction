from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from .data_loader import format_note_with_line_numbers, get_valid_categories, get_valid_statuses
from .llm_client import OpenAICompatibleClient
from .schemas import Condition, PatientOutput, Taxonomy, validate_condition_taxonomy
from .utils import ConditionKey, clean_markdown_line, dump_json, ensure_dir, load_json, normalize_condition_name, sha256_text


@dataclass(frozen=True)
class ExtractorConfig:
    cache_dir: Path
    max_conditions_per_note: int = 60


NOTE_SYSTEM_PROMPT = """You are an expert clinical information extraction system.

Task: Extract ALL clinically significant diagnoses and findings that map to the provided taxonomy.

Hard constraints:
- Output MUST be valid JSON.
- ONLY include conditions that fit the taxonomy (category/subcategory keys).
- status MUST be one of: active, resolved, suspected.
- evidence.span MUST be copied EXACTLY from the provided note line (without modifying spelling/case/punctuation).
- evidence.line_no MUST match the line number shown.
- Do NOT invent facts or dates. If onset cannot be determined, use null.

You are extracting from ONE NOTE only. Do not merge across notes.
Return a JSON object with key: "conditions" as an array of condition objects:
{condition_name, category, subcategory, status, onset, evidence:[{note_id,line_no,span}]}.
Include at least one evidence entry per condition, from this note.
"""


PATIENT_SYSTEM_PROMPT = """You are an expert clinical information extraction system.

Task: Build a longitudinal condition summary for ONE PATIENT by consolidating condition mentions across notes.

Hard constraints:
- Output MUST be valid JSON matching:
  { "patient_id": "...", "conditions": [ {condition_name, category, subcategory, status, onset, evidence:[...]} ] }
- Only taxonomy-valid category/subcategory keys.
- status reflects the condition's state as of the LATEST note where it appears.
- onset uses the EARLIEST explicit documentation date, following the onset priority rules in the instructions.
- evidence MUST include excerpts from EVERY note where the condition is mentioned (comprehensive).
- evidence.span MUST be copied EXACTLY from the provided evidence candidates (no edits).
- One entry per distinct condition; separate entries for different metastatic sites / anatomical sites when applicable.

Do not add any conditions that are not supported by evidence candidates.
"""


def _taxonomy_brief(taxonomy: dict[str, Any]) -> str:
    # Keep prompt size small: only keys
    cats = get_valid_categories(taxonomy)
    statuses = get_valid_statuses(taxonomy)
    return json.dumps({"valid_categories": cats, "valid_statuses": statuses}, ensure_ascii=False)


def _cache_get(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path = cache_dir / f"{key}.json"
    if path.exists():
        return load_json(path)
    return None


def _cache_set(cache_dir: Path, key: str, obj: dict[str, Any]) -> None:
    ensure_dir(cache_dir)
    dump_json(cache_dir / f"{key}.json", obj)


def _coerce_evidence_spans(note: dict[str, Any], extracted: dict[str, Any]) -> dict[str, Any]:
    """
    Enforce evidence.span is an exact substring of the referenced line.
    If not, replace with the full (cleaned) line content to maintain exactness.
    """
    note_lines: dict[int, str] = note["lines"]
    note_id = note["note_id"]
    for c in extracted.get("conditions", []) or []:
        ev_list = c.get("evidence", []) or []
        fixed = []
        for ev in ev_list:
            try:
                line_no = int(ev.get("line_no"))
            except Exception:
                continue
            if line_no not in note_lines:
                continue
            raw_line = note_lines[line_no]
            span = str(ev.get("span") or "").strip()
            if not span or span not in raw_line:
                span = clean_markdown_line(raw_line)
                if span and span not in raw_line:
                    # fall back to raw line verbatim if cleaning removed chars
                    span = raw_line
            fixed.append({"note_id": note_id, "line_no": line_no, "span": span})
        c["evidence"] = fixed
    return extracted


def extract_conditions_from_note(
    *,
    llm: OpenAICompatibleClient,
    taxonomy: dict[str, Any],
    note: dict[str, Any],
    config: ExtractorConfig,
) -> list[Condition]:
    taxonomy_brief = _taxonomy_brief(taxonomy)
    note_text = format_note_with_line_numbers(note)
    user = (
        "Valid taxonomy keys (do not output anything else):\n"
        f"{taxonomy_brief}\n\n"
        f"note_id: {note['note_id']}\n"
        "NOTE (each line starts with line_no: ): \n"
        f"{note_text}\n"
    )
    cache_key = sha256_text(NOTE_SYSTEM_PROMPT + "\n" + user)
    cached = _cache_get(config.cache_dir / "note", cache_key)
    if cached is None:
        cached = llm.json_chat(system=NOTE_SYSTEM_PROMPT, user=user)
        _cache_set(config.cache_dir / "note", cache_key, cached)

    cached = _coerce_evidence_spans(note, cached)

    valid_cat_to_subcats = get_valid_categories(taxonomy)
    valid_statuses = set(get_valid_statuses(taxonomy))

    out: list[Condition] = []
    for item in (cached.get("conditions", []) or [])[: config.max_conditions_per_note]:
        try:
            c = Condition.model_validate(item)
        except Exception:
            continue
        if c.status not in valid_statuses:
            continue
        try:
            validate_condition_taxonomy(c, valid_cat_to_subcats)
        except Exception:
            continue
        out.append(c)
    return out


def _key_for_condition(c: Condition) -> ConditionKey:
    return ConditionKey(
        category=c.category,
        subcategory=c.subcategory,
        name_norm=normalize_condition_name(c.condition_name),
    )


def _dedupe_conditions(conditions: list[Condition]) -> list[Condition]:
    """
    Quick local dedupe before patient-level LLM consolidation.
    Keeps all evidence, picks latest status by note ordering later.
    """
    by_key: dict[ConditionKey, Condition] = {}
    for c in conditions:
        k = _key_for_condition(c)
        if k not in by_key:
            by_key[k] = c
            continue
        existing = by_key[k]
        merged = existing.model_copy(deep=True)
        merged.evidence.extend(c.evidence)
        by_key[k] = merged

    # fuzzy merge within same category/subcategory for small spelling diffs
    keys = list(by_key.keys())
    merged_map: dict[ConditionKey, Condition] = {}
    used: set[ConditionKey] = set()
    for i, k in enumerate(keys):
        if k in used:
            continue
        base = by_key[k].model_copy(deep=True)
        used.add(k)
        for j in range(i + 1, len(keys)):
            k2 = keys[j]
            if k2 in used:
                continue
            if k.category != k2.category or k.subcategory != k2.subcategory:
                continue
            if fuzz.ratio(k.name_norm, k2.name_norm) >= 94:
                base.evidence.extend(by_key[k2].evidence)
                used.add(k2)
        merged_map[_key_for_condition(base)] = base
    return list(merged_map.values())


def consolidate_patient(
    *,
    llm: OpenAICompatibleClient,
    taxonomy: dict[str, Any],
    patient_id: str,
    note_ids_in_order: list[str],
    note_conditions: dict[str, list[Condition]],
    config: ExtractorConfig,
) -> PatientOutput:
    taxonomy_brief = _taxonomy_brief(taxonomy)

    # Provide candidate evidence only (keep spans exact), plus note order for status rule.
    candidates: list[dict[str, Any]] = []
    for note_id in note_ids_in_order:
        conds = note_conditions.get(note_id, [])
        for c in conds:
            candidates.append(
                {
                    "note_id": note_id,
                    "condition_name": c.condition_name,
                    "category": c.category,
                    "subcategory": c.subcategory,
                    "status": c.status,
                    "onset": c.onset,
                    "evidence": [e.model_dump() for e in c.evidence],
                }
            )

    user = (
        f"patient_id: {patient_id}\n"
        "Valid taxonomy keys (do not output anything else):\n"
        f"{taxonomy_brief}\n\n"
        "Notes are in chronological order (earliest to latest):\n"
        f"{json.dumps(note_ids_in_order, ensure_ascii=False)}\n\n"
        "Evidence candidates (you MUST ONLY use these spans verbatim; include ALL notes where mentioned):\n"
        f"{json.dumps(candidates, ensure_ascii=False)}\n"
    )

    cache_key = sha256_text(PATIENT_SYSTEM_PROMPT + "\n" + user)
    cached = _cache_get(config.cache_dir / "patient", cache_key)
    if cached is None:
        cached = llm.json_chat(system=PATIENT_SYSTEM_PROMPT, user=user)
        _cache_set(config.cache_dir / "patient", cache_key, cached)

    valid_cat_to_subcats = get_valid_categories(taxonomy)
    Taxonomy.model_validate(taxonomy)  # sanity

    # Validate output schema + taxonomy; if model returns junk, fall back to deterministic merge.
    try:
        out = PatientOutput.model_validate(cached)
        for c in out.conditions:
            validate_condition_taxonomy(c, valid_cat_to_subcats)
        return out
    except Exception:
        # deterministic fallback: dedupe candidates, keep status as latest mention, onset earliest non-null
        flat: list[Condition] = []
        latest_idx: dict[ConditionKey, int] = {}
        onset_best: dict[ConditionKey, str | None] = {}
        for idx, note_id in enumerate(note_ids_in_order):
            for c in note_conditions.get(note_id, []):
                k = _key_for_condition(c)
                flat.append(c)
                latest_idx[k] = max(latest_idx.get(k, -1), idx)
                if c.onset:
                    onset_best[k] = onset_best.get(k) or c.onset
        merged = _dedupe_conditions(flat)
        by_k = {_key_for_condition(c): c for c in merged}
        # status: pick from a condition instance in the latest note if possible
        for k, c in list(by_k.items()):
            li = latest_idx.get(k, -1)
            if li >= 0:
                note_id = note_ids_in_order[li]
                for c2 in note_conditions.get(note_id, []):
                    if _key_for_condition(c2) == k:
                        by_k[k] = c.model_copy(update={"status": c2.status, "onset": onset_best.get(k)})
                        break
            else:
                by_k[k] = c.model_copy(update={"onset": onset_best.get(k)})
        return PatientOutput(patient_id=patient_id, conditions=list(by_k.values()))

