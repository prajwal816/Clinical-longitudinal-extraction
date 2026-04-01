from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .data_loader import load_patient_notes, load_taxonomy
from .extractor import ExtractorConfig, consolidate_patient, extract_conditions_from_note
from .llm_client import OpenAICompatibleClient
from .utils import dump_json, ensure_dir


@dataclass(frozen=True)
class InferenceConfig:
    data_dir: Path
    taxonomy_path: Path
    output_dir: Path
    cache_dir: Path


def run_patient(
    *,
    llm: OpenAICompatibleClient,
    patient_id: str,
    cfg: InferenceConfig,
) -> Path:
    taxonomy = load_taxonomy(str(cfg.taxonomy_path))
    notes = load_patient_notes(str(cfg.data_dir), patient_id)
    extractor_cfg = ExtractorConfig(cache_dir=cfg.cache_dir)

    note_conditions = {}
    note_ids = []
    for note in notes:
        note_ids.append(note["note_id"])
        conds = extract_conditions_from_note(
            llm=llm, taxonomy=taxonomy, note=note, config=extractor_cfg
        )
        note_conditions[note["note_id"]] = conds

    patient_out = consolidate_patient(
        llm=llm,
        taxonomy=taxonomy,
        patient_id=patient_id,
        note_ids_in_order=note_ids,
        note_conditions=note_conditions,
        config=extractor_cfg,
    )

    ensure_dir(cfg.output_dir)
    out_path = cfg.output_dir / f"{patient_id}.json"
    dump_json(out_path, patient_out.model_dump())
    return out_path

