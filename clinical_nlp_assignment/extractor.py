from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from .data_loader import format_note_with_line_numbers, get_valid_categories, get_valid_statuses
from .llm_client import OpenAICompatibleClient
from .prompts import (
    build_note_system_prompt,
    build_note_user_prompt,
    build_patient_system_prompt,
    build_patient_user_prompt,
)
from .schemas import Condition, PatientOutput, Taxonomy, validate_condition_taxonomy
from .utils import ConditionKey, clean_markdown_line, dump_json, ensure_dir, load_json, normalize_condition_name, sha256_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractorConfig:
    cache_dir: Path
    max_conditions_per_note: int = 60


# ---------------------------------------------------------------------------
# Cache helpers
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
# Evidence span coercion
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Condition recovery: fuzzy taxonomy key matching
# ---------------------------------------------------------------------------

def _try_recover_taxonomy_keys(
    item: dict[str, Any], valid_cat_to_subcats: dict[str, list[str]]
) -> dict[str, Any] | None:
    """Attempt to fix invalid category/subcategory keys via fuzzy matching.

    Returns the fixed item dict if recoverable, None otherwise.
    """
    cat = str(item.get("category", "")).strip().lower()
    subcat = str(item.get("subcategory", "")).strip().lower()

    # Try exact cat first
    if cat in valid_cat_to_subcats:
        if subcat in valid_cat_to_subcats[cat]:
            return item  # already valid
        # Try fuzzy subcat within this category
        best_sub, best_score = None, 0
        for valid_sub in valid_cat_to_subcats[cat]:
            score = fuzz.ratio(subcat, valid_sub)
            if score > best_score:
                best_sub, best_score = valid_sub, score
        if best_score >= 75 and best_sub:
            item["subcategory"] = best_sub
            logger.info("Recovered subcategory: '%s' → '%s' (score=%d)", subcat, best_sub, best_score)
            return item

    # Try fuzzy cat
    best_cat, best_cat_score = None, 0
    for valid_cat in valid_cat_to_subcats:
        score = fuzz.ratio(cat, valid_cat)
        if score > best_cat_score:
            best_cat, best_cat_score = valid_cat, score

    if best_cat_score >= 75 and best_cat:
        item["category"] = best_cat
        # Now try subcat within the matched category
        if subcat in valid_cat_to_subcats[best_cat]:
            logger.info("Recovered category: '%s' → '%s' (score=%d)", cat, best_cat, best_cat_score)
            return item
        best_sub, best_sub_score = None, 0
        for valid_sub in valid_cat_to_subcats[best_cat]:
            score = fuzz.ratio(subcat, valid_sub)
            if score > best_sub_score:
                best_sub, best_sub_score = valid_sub, score
        if best_sub_score >= 75 and best_sub:
            item["subcategory"] = best_sub
            logger.info(
                "Recovered category+subcategory: '%s.%s' → '%s.%s'",
                cat, subcat, best_cat, best_sub,
            )
            return item

    return None


# ---------------------------------------------------------------------------
# Per-note extraction
# ---------------------------------------------------------------------------


def extract_conditions_from_note(
    *,
    llm: OpenAICompatibleClient,
    taxonomy: dict[str, Any],
    note: dict[str, Any],
    config: ExtractorConfig,
) -> list[Condition]:
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
    for item in (cached.get("conditions", []) or [])[: config.max_conditions_per_note]:
        # Attempt recovery for invalid taxonomy keys
        if not _is_taxonomy_valid(item, valid_cat_to_subcats):
            recovered = _try_recover_taxonomy_keys(item, valid_cat_to_subcats)
            if recovered is None:
                logger.debug("Dropping condition with unrecoverable taxonomy: %s", item.get("condition_name"))
                continue
            item = recovered

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


def _is_taxonomy_valid(item: dict[str, Any], valid_cat_to_subcats: dict[str, list[str]]) -> bool:
    """Quick check if an item has valid category/subcategory."""
    cat = str(item.get("category", ""))
    subcat = str(item.get("subcategory", ""))
    return cat in valid_cat_to_subcats and subcat in valid_cat_to_subcats.get(cat, [])


# ---------------------------------------------------------------------------
# Condition keying and deduplication
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Evidence hardening (post-LLM deterministic pass)
# ---------------------------------------------------------------------------

# Common clinical abbreviations mapping condition keywords to abbreviations
_ABBREVIATION_MAP: dict[str, list[str]] = {
    "hypertension": ["htn"],
    "diabetes mellitus": ["dm ii", "dm2", "niddm", "iddm", "dm type"],
    "atrial fibrillation": ["afib", "a-fib"],
    "chronic obstructive pulmonary disease": ["copd"],
    "congestive heart failure": ["chf"],
    "coronary artery disease": ["cad"],
    "myocardial infarction": ["mi ", "stemi", "nstemi"],
    "deep vein thrombosis": ["dvt"],
    "pulmonary embolism": ["pulm. embolism"],
    "urinary tract infection": ["uti"],
    "obstructive sleep apnea": ["osa ", "osas"],
    "gastroesophageal reflux disease": ["gerd"],
    "chronic kidney disease": ["ckd"],
    "end-stage renal disease": ["esrd"],
    "acute respiratory distress syndrome": ["ards"],
    "thrombocytopenia": ["low platelets"],
    "hypothyroidism": ["hypothyr."],
    "hyperthyroidism": ["hyperthy."],
}

# Minimum length for a search term to be used (avoids short false matches)
_MIN_TERM_LEN = 5


def _harden_evidence(
    patient_output: PatientOutput,
    notes: list[dict[str, Any]],
) -> PatientOutput:
    """Deterministic post-pass to ensure evidence completeness.

    Conservative approach: only matches the full condition name (case-insensitive)
    and known clinical abbreviations. Does NOT decompose condition names into
    individual words to avoid noisy false matches.
    """
    from .schemas import Evidence

    note_map: dict[str, dict[str, Any]] = {n["note_id"]: n for n in notes}

    for condition in patient_output.conditions:
        cond_name_lower = condition.condition_name.lower()

        # Build search terms: full condition name + abbreviations only
        search_terms: list[str] = [cond_name_lower]

        # Add abbreviations only if the condition name matches a known mapping
        for full_name, abbrevs in _ABBREVIATION_MAP.items():
            if full_name in cond_name_lower or cond_name_lower in full_name:
                search_terms.extend(abbrevs)

        # Filter out terms that are too short to be meaningful
        search_terms = [t for t in search_terms if len(t) >= _MIN_TERM_LEN]

        if not search_terms:
            continue  # Skip if no usable search terms

        existing_keys: set[tuple[str, int]] = set()
        for ev in condition.evidence:
            existing_keys.add((ev.note_id, ev.line_no))

        new_evidence = list(condition.evidence)

        for note_id, note_data in note_map.items():
            note_lines: dict[int, str] = note_data["lines"]
            for line_no, line_content in note_lines.items():
                if (note_id, line_no) in existing_keys:
                    continue
                line_lower = line_content.lower()
                stripped = line_content.strip()
                if len(stripped) < 5:
                    continue

                # Check if any search term appears in this line
                matched_term = None
                for term in search_terms:
                    if term in line_lower:
                        matched_term = term
                        break
                if matched_term is None:
                    continue

                # Use the full stripped line as span (guaranteed to be in raw_line)
                new_evidence.append(Evidence(note_id=note_id, line_no=line_no, span=stripped))
                existing_keys.add((note_id, line_no))

        # Deduplicate by (note_id, line_no) and verify span exactness
        seen: set[tuple[str, int]] = set()
        deduped = []
        for ev in new_evidence:
            key = (ev.note_id, ev.line_no)
            if key in seen:
                continue
            seen.add(key)
            note_data = note_map.get(ev.note_id)
            if note_data is None:
                continue
            note_lines = note_data["lines"]
            if ev.line_no not in note_lines:
                continue
            raw_line = note_lines[ev.line_no]
            if ev.span not in raw_line:
                # Fix span to be the raw line
                span = clean_markdown_line(raw_line)
                if span and span not in raw_line:
                    span = raw_line
                ev = Evidence(note_id=ev.note_id, line_no=ev.line_no, span=span if span else raw_line)
            deduped.append(ev)

        condition.evidence = deduped

    return patient_output


# ---------------------------------------------------------------------------
# Patient consolidation
# ---------------------------------------------------------------------------


def consolidate_patient(
    *,
    llm: OpenAICompatibleClient,
    taxonomy: dict[str, Any],
    patient_id: str,
    note_ids_in_order: list[str],
    note_conditions: dict[str, list[Condition]],
    notes: list[dict[str, Any]] | None = None,
    config: ExtractorConfig,
) -> PatientOutput:
    system_prompt = build_patient_system_prompt(taxonomy)

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

    user_prompt = build_patient_user_prompt(patient_id, note_ids_in_order, candidates, taxonomy)

    cache_key = sha256_text(system_prompt + "\n" + user_prompt)
    cached = _cache_get(config.cache_dir / "patient", cache_key)
    if cached is None:
        cached = llm.json_chat(system=system_prompt, user=user_prompt)
        _cache_set(config.cache_dir / "patient", cache_key, cached)

    valid_cat_to_subcats = get_valid_categories(taxonomy)
    Taxonomy.model_validate(taxonomy)  # sanity

    # Validate output schema + taxonomy; if model returns junk, fall back to deterministic merge.
    try:
        out = PatientOutput.model_validate(cached)
        valid_conditions = []
        for c in out.conditions:
            try:
                validate_condition_taxonomy(c, valid_cat_to_subcats)
                valid_conditions.append(c)
            except Exception:
                # Try recovery
                item = c.model_dump()
                recovered = _try_recover_taxonomy_keys(item, valid_cat_to_subcats)
                if recovered:
                    try:
                        rc = Condition.model_validate(recovered)
                        validate_condition_taxonomy(rc, valid_cat_to_subcats)
                        valid_conditions.append(rc)
                    except Exception:
                        logger.debug("Dropping condition after failed recovery: %s", c.condition_name)
                else:
                    logger.debug("Dropping unrecoverable condition: %s", c.condition_name)
        out.conditions = valid_conditions

        # Apply evidence hardening if notes are available
        if notes:
            out = _harden_evidence(out, notes)

        return out
    except Exception:
        logger.warning("Patient-level LLM output failed validation; using deterministic fallback for %s", patient_id)
        # deterministic fallback: dedupe candidates, keep status as latest mention, onset earliest
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

        result = PatientOutput(patient_id=patient_id, conditions=list(by_k.values()))

        # Apply evidence hardening if notes are available
        if notes:
            result = _harden_evidence(result, notes)

        return result
