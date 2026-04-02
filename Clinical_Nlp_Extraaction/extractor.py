"""
Two-pass condition extraction engine.

Pass 1 (per-note): Extract condition candidates from each clinical note.
Pass 2 (patient):  Consolidate candidates into a final patient condition summary.

Includes deterministic evidence hardening, fuzzy taxonomy recovery, and
robust deduplication.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process as fuzz_process

from .data_loader import format_note_with_line_numbers, get_valid_categories, get_valid_statuses
from .llm_client import OpenAICompatibleClient
from .prompts import (
    build_note_system_prompt,
    build_note_user_prompt,
    build_patient_system_prompt,
    build_patient_user_prompt,
)
from .schemas import Condition, Evidence, PatientOutput, Taxonomy, validate_condition_taxonomy
from .utils import (
    ConditionKey,
    clean_markdown_line,
    dump_json,
    ensure_dir,
    load_json,
    normalize_condition_name,
    sha256_text,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractorConfig:
    cache_dir: Path
    max_conditions_per_note: int = 80


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def _cache_get(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path = cache_dir / f"{key}.json"
    if path.exists():
        return load_json(path)
    return None


def _cache_set(cache_dir: Path, key: str, obj: dict[str, Any]) -> None:
    ensure_dir(cache_dir)
    dump_json(cache_dir / f"{key}.json", obj)


# ---------------------------------------------------------------------------
# Fuzzy taxonomy recovery
# ---------------------------------------------------------------------------


def _try_fix_taxonomy(
    condition: dict[str, Any],
    valid_cat_to_subcats: dict[str, list[str]],
) -> dict[str, Any] | None:
    """Attempt to fix invalid category/subcategory via fuzzy matching.

    Returns corrected dict or None if unfixable.
    """
    cat = condition.get("category", "")
    sub = condition.get("subcategory", "")

    all_cats = list(valid_cat_to_subcats.keys())

    # Fix category
    if cat not in valid_cat_to_subcats:
        match = fuzz_process.extractOne(cat, all_cats, score_cutoff=75)
        if match:
            logger.debug("Taxonomy fix: category '%s' → '%s' (score=%d)", cat, match[0], match[1])
            cat = match[0]
            condition = {**condition, "category": cat}
        else:
            return None

    # Fix subcategory
    valid_subs = valid_cat_to_subcats.get(cat, [])
    if sub not in valid_subs:
        match = fuzz_process.extractOne(sub, valid_subs, score_cutoff=70)
        if match:
            logger.debug("Taxonomy fix: subcategory '%s' → '%s' (score=%d)", sub, match[0], match[1])
            condition = {**condition, "subcategory": match[0]}
        else:
            return None

    return condition


# ---------------------------------------------------------------------------
# Evidence coercion & hardening
# ---------------------------------------------------------------------------


def _coerce_evidence_spans(note: dict[str, Any], extracted: dict[str, Any]) -> dict[str, Any]:
    """Enforce evidence.span is an exact substring of the referenced line.

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
            except (TypeError, ValueError):
                continue
            if line_no not in note_lines:
                continue
            raw_line = note_lines[line_no]
            span = str(ev.get("span") or "").strip()
            if not span or span not in raw_line:
                # Try cleaned version
                span = clean_markdown_line(raw_line)
                if not span:
                    span = raw_line.strip()
                elif span not in raw_line:
                    # Cleaning changed chars; use raw line verbatim
                    span = raw_line.strip()
            if span:
                fixed.append({"note_id": note_id, "line_no": line_no, "span": span})
        c["evidence"] = fixed
    return extracted


def _harden_evidence_completeness(
    patient_output: PatientOutput,
    all_notes: list[dict[str, Any]],
) -> PatientOutput:
    """Deterministic post-check: ensure every note mentioning a condition has evidence.

    Scans all note lines for fuzzy matches of each condition name and adds
    missing evidence entries from notes that were missed by the LLM.
    Also deduplicates evidence entries.
    """
    for condition in patient_output.conditions:
        cond_name_norm = normalize_condition_name(condition.condition_name)
        # Track which notes already have evidence
        existing_note_ids = {ev.note_id for ev in condition.evidence}

        for note in all_notes:
            note_id = note["note_id"]
            # Don't add duplicates — only scan notes missing from evidence
            if note_id in existing_note_ids:
                continue

            note_lines: dict[int, str] = note["lines"]
            for line_no, line_text in note_lines.items():
                line_norm = normalize_condition_name(line_text)
                if not line_norm or len(line_norm) < 3:
                    continue

                # Check if condition name appears in this line (fuzzy)
                score = fuzz.partial_ratio(cond_name_norm, line_norm)
                if score >= 90 and len(cond_name_norm) >= 4:
                    span = line_text.strip()
                    if span:
                        condition.evidence.append(
                            Evidence(note_id=note_id, line_no=line_no, span=span)
                        )
                        existing_note_ids.add(note_id)
                        break  # One evidence per note is sufficient for hardening

        # Deduplicate evidence: same (note_id, line_no)
        seen: set[tuple[str, int]] = set()
        deduped: list[Evidence] = []
        for ev in condition.evidence:
            key = (ev.note_id, ev.line_no)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        condition.evidence = deduped

    return patient_output


# ---------------------------------------------------------------------------
# Pass 1: Per-note extraction
# ---------------------------------------------------------------------------


def extract_conditions_from_note(
    *,
    llm: OpenAICompatibleClient,
    taxonomy: dict[str, Any],
    note: dict[str, Any],
    config: ExtractorConfig,
) -> list[Condition]:
    """Extract conditions from a single clinical note using the LLM."""
    system_prompt = build_note_system_prompt(taxonomy)
    note_text = format_note_with_line_numbers(note)
    user_prompt = build_note_user_prompt(note["note_id"], note_text, taxonomy)

    cache_key = sha256_text(system_prompt + "\n" + user_prompt)
    cached = _cache_get(config.cache_dir / "note", cache_key)
    if cached is None:
        cached = llm.json_chat(system=system_prompt, user=user_prompt)
        _cache_set(config.cache_dir / "note", cache_key, cached)

    cached = _coerce_evidence_spans(note, cached)

    valid_cat_to_subcats = get_valid_categories(taxonomy)
    valid_statuses = set(get_valid_statuses(taxonomy))

    out: list[Condition] = []
    for item in (cached.get("conditions", []) or [])[:config.max_conditions_per_note]:
        # Attempt fuzzy taxonomy recovery before validation
        fixed_item = _try_fix_taxonomy(item, valid_cat_to_subcats)
        if fixed_item is None:
            logger.debug("Dropping condition (unfixable taxonomy): %s", item.get("condition_name", "?"))
            continue
        item = fixed_item

        # Fix status if invalid
        status = item.get("status", "")
        if status not in valid_statuses:
            # Try common misspellings
            status_match = fuzz_process.extractOne(status, list(valid_statuses), score_cutoff=70)
            if status_match:
                item["status"] = status_match[0]
            else:
                item["status"] = "active"  # Safe default

        try:
            c = Condition.model_validate(item)
        except Exception as e:
            logger.debug("Condition validation failed: %s — %s", item.get("condition_name", "?"), e)
            continue

        try:
            validate_condition_taxonomy(c, valid_cat_to_subcats)
        except Exception:
            continue

        out.append(c)

    return out


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------


def _key_for_condition(c: Condition) -> ConditionKey:
    return ConditionKey(
        category=c.category,
        subcategory=c.subcategory,
        name_norm=normalize_condition_name(c.condition_name),
    )


def _dedupe_conditions(conditions: list[Condition]) -> list[Condition]:
    """Quick local dedupe before patient-level LLM consolidation.

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

    # Fuzzy merge within same category/subcategory for small spelling diffs
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
            if fuzz.ratio(k.name_norm, k2.name_norm) >= 88:
                base.evidence.extend(by_key[k2].evidence)
                used.add(k2)
                # Use the longer/more descriptive name
                if len(by_key[k2].condition_name) > len(base.condition_name):
                    base.condition_name = by_key[k2].condition_name
        merged_map[_key_for_condition(base)] = base
    return list(merged_map.values())


# ---------------------------------------------------------------------------
# Pass 2: Patient-level consolidation
# ---------------------------------------------------------------------------


def consolidate_patient(
    *,
    llm: OpenAICompatibleClient,
    taxonomy: dict[str, Any],
    patient_id: str,
    note_ids_in_order: list[str],
    note_conditions: dict[str, list[Condition]],
    all_notes: list[dict[str, Any]],
    config: ExtractorConfig,
) -> PatientOutput:
    """Consolidate per-note conditions into a final patient summary."""
    system_prompt = build_patient_system_prompt(taxonomy)

    # Build candidate evidence, preserving note ordering
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

    user_prompt = build_patient_user_prompt(patient_id, note_ids_in_order, candidates, taxonomy)

    cache_key = sha256_text(system_prompt + "\n" + user_prompt)
    cached = _cache_get(config.cache_dir / "patient", cache_key)
    if cached is None:
        cached = llm.json_chat(system=system_prompt, user=user_prompt)
        _cache_set(config.cache_dir / "patient", cache_key, cached)

    valid_cat_to_subcats = get_valid_categories(taxonomy)

    # Validate output schema + taxonomy; if model returns junk, fall back to deterministic merge.
    try:
        out = PatientOutput.model_validate(cached)
        # Validate each condition's taxonomy
        valid_conditions: list[Condition] = []
        for c in out.conditions:
            try:
                validate_condition_taxonomy(c, valid_cat_to_subcats)
                valid_conditions.append(c)
            except Exception:
                # Try fuzzy fix
                fixed = _try_fix_taxonomy(c.model_dump(), valid_cat_to_subcats)
                if fixed:
                    try:
                        fc = Condition.model_validate(fixed)
                        validate_condition_taxonomy(fc, valid_cat_to_subcats)
                        valid_conditions.append(fc)
                    except Exception:
                        logger.debug("Dropping condition after fix attempt: %s", c.condition_name)
                else:
                    logger.debug("Dropping unfixable condition: %s", c.condition_name)
        out.conditions = valid_conditions
        # Apply evidence hardening
        out = _harden_evidence_completeness(out, all_notes)
        return out
    except Exception as e:
        logger.warning("Patient consolidation LLM output failed validation: %s. Using deterministic fallback.", e)
        return _deterministic_fallback(
            patient_id=patient_id,
            note_ids_in_order=note_ids_in_order,
            note_conditions=note_conditions,
            all_notes=all_notes,
            valid_cat_to_subcats=valid_cat_to_subcats,
        )


def _deterministic_fallback(
    *,
    patient_id: str,
    note_ids_in_order: list[str],
    note_conditions: dict[str, list[Condition]],
    all_notes: list[dict[str, Any]],
    valid_cat_to_subcats: dict[str, list[str]],
) -> PatientOutput:
    """Deterministic consolidation when LLM output fails validation.

    - Dedupes by normalized name + taxonomy slot
    - Onset: earliest non-null
    - Status: from latest note where condition appears
    - Evidence: union of all candidates
    """
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

    # Apply latest-note status and earliest onset
    for k, c in list(by_k.items()):
        li = latest_idx.get(k, -1)
        updates: dict[str, Any] = {"onset": onset_best.get(k)}
        if li >= 0:
            note_id = note_ids_in_order[li]
            for c2 in note_conditions.get(note_id, []):
                if _key_for_condition(c2) == k:
                    updates["status"] = c2.status
                    break
        by_k[k] = c.model_copy(update=updates)

    result = PatientOutput(patient_id=patient_id, conditions=list(by_k.values()))
    # Apply evidence hardening
    result = _harden_evidence_completeness(result, all_notes)
    return result
