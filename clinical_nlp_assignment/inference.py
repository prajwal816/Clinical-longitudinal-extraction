"""
Inference orchestrator — runs the two-pass extraction pipeline for a patient.

Supports concurrent per-note extraction via ThreadPoolExecutor for speed.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .data_loader import load_patient_notes, load_taxonomy
from .extractor import ExtractorConfig, consolidate_patient, extract_conditions_from_note
from .llm_client import OpenAICompatibleClient
from .schemas import Condition
from .utils import dump_json, ensure_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceConfig:
    data_dir: Path
    taxonomy_path: Path
    output_dir: Path
    cache_dir: Path
    concurrency: int = 4


def _extract_single_note(
    llm: OpenAICompatibleClient,
    taxonomy: dict,
    note: dict,
    extractor_cfg: ExtractorConfig,
) -> tuple[str, list[Condition]]:
    """Extract conditions from a single note. Returns (note_id, conditions)."""
    conds = extract_conditions_from_note(
        llm=llm, taxonomy=taxonomy, note=note, config=extractor_cfg,
    )
    return note["note_id"], conds


def run_patient(
    *,
    llm: OpenAICompatibleClient,
    patient_id: str,
    cfg: InferenceConfig,
) -> Path:
    """Run the full extraction pipeline for a single patient.

    1. Load all notes
    2. Extract conditions per-note (optionally in parallel)
    3. Consolidate into patient summary
    4. Write output JSON
    """
    taxonomy = load_taxonomy(str(cfg.taxonomy_path))
    notes = load_patient_notes(str(cfg.data_dir), patient_id)
    extractor_cfg = ExtractorConfig(cache_dir=cfg.cache_dir)

    note_conditions: dict[str, list[Condition]] = {}
    note_ids = [n["note_id"] for n in notes]

    if cfg.concurrency > 1 and len(notes) > 1:
        # Parallel per-note extraction
        with ThreadPoolExecutor(max_workers=min(cfg.concurrency, len(notes))) as pool:
            futures = {
                pool.submit(_extract_single_note, llm, taxonomy, note, extractor_cfg): note["note_id"]
                for note in notes
            }
            for future in as_completed(futures):
                nid = futures[future]
                try:
                    note_id, conds = future.result()
                    note_conditions[note_id] = conds
                    logger.info("  %s: %d conditions extracted", note_id, len(conds))
                except Exception as e:
                    logger.error("  %s extraction failed: %s", nid, e)
                    note_conditions[nid] = []
    else:
        # Sequential
        for note in notes:
            conds = extract_conditions_from_note(
                llm=llm, taxonomy=taxonomy, note=note, config=extractor_cfg,
            )
            note_conditions[note["note_id"]] = conds
            logger.info("  %s: %d conditions extracted", note["note_id"], len(conds))

    patient_out = consolidate_patient(
        llm=llm,
        taxonomy=taxonomy,
        patient_id=patient_id,
        note_ids_in_order=note_ids,
        note_conditions=note_conditions,
        all_notes=notes,
        config=extractor_cfg,
    )

    logger.info(
        "  %s: %d conditions consolidated from %d notes",
        patient_id, len(patient_out.conditions), len(notes),
    )

    ensure_dir(cfg.output_dir)
    out_path = cfg.output_dir / f"{patient_id}.json"
    dump_json(out_path, patient_out.model_dump())
    return out_path
