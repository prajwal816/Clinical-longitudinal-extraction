from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .data_loader import load_ground_truth, load_patient_notes, load_taxonomy
from .evaluate import compute_prf1, macro_average
from .extractor import ExtractorConfig, consolidate_patient, extract_conditions_from_note
from .model import build_llm_client


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate extraction on labeled train set")
    p.add_argument(
        "--data-dir",
        required=True,
        help="Path to train directory containing patient_XX/ and labels/",
    )
    p.add_argument(
        "--taxonomy-path",
        required=True,
        help="Path to taxonomy.json",
    )
    p.add_argument(
        "--cache-dir",
        default=str(Path(".cache")),
        help="Cache directory (default: ./.cache)",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-output-tokens", type=int, default=2048)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    labels_dir = data_dir / "labels"
    taxonomy = load_taxonomy(str(Path(args.taxonomy_path)))

    llm = build_llm_client(
        temperature=float(args.temperature), max_output_tokens=int(args.max_output_tokens)
    )
    extractor_cfg = ExtractorConfig(cache_dir=Path(args.cache_dir))

    patient_ids = sorted([p.name for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("patient_")])
    scores = []
    for pid in tqdm(patient_ids, desc="Evaluating patients"):
        gt = load_ground_truth(str(labels_dir), pid)
        if not gt:
            continue

        notes = load_patient_notes(str(data_dir), pid)
        note_ids = [n["note_id"] for n in notes]
        note_conditions = {}
        for note in notes:
            note_conditions[note["note_id"]] = extract_conditions_from_note(
                llm=llm, taxonomy=taxonomy, note=note, config=extractor_cfg
            )

        pred = consolidate_patient(
            llm=llm,
            taxonomy=taxonomy,
            patient_id=pid,
            note_ids_in_order=note_ids,
            note_conditions=note_conditions,
            config=extractor_cfg,
        )

        score = compute_prf1(
            y_true=pred.__class__.model_validate(gt),
            y_pred=pred,
        )
        scores.append(score)
        tqdm.write(f"{pid}: P={score.precision:.3f} R={score.recall:.3f} F1={score.f1:.3f}")

    macro = macro_average(scores)
    print(f"MACRO: P={macro.precision:.3f} R={macro.recall:.3f} F1={macro.f1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

