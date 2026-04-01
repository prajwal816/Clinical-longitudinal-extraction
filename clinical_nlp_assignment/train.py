from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from .data_loader import load_ground_truth, load_patient_notes, load_taxonomy
from .evaluate import compute_detailed_score, compute_prf1, macro_average, macro_average_detailed
from .extractor import ExtractorConfig, consolidate_patient, extract_conditions_from_note
from .model import build_llm_client
from .utils import dump_json

logger = logging.getLogger(__name__)


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
    p.add_argument(
        "--results-json",
        default=None,
        help="Path to save detailed results JSON (optional)",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-output-tokens", type=int, default=4096)
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    data_dir = Path(args.data_dir)
    labels_dir = data_dir / "labels"
    taxonomy = load_taxonomy(str(Path(args.taxonomy_path)))

    llm = build_llm_client(
        temperature=float(args.temperature), max_output_tokens=int(args.max_output_tokens)
    )
    extractor_cfg = ExtractorConfig(cache_dir=Path(args.cache_dir))

    patient_ids = sorted([p.name for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("patient_")])
    detailed_scores = []
    basic_scores = []
    results_per_patient = []

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
            notes=notes,
            config=extractor_cfg,
        )

        gt_output = pred.__class__.model_validate(gt)
        score = compute_prf1(y_true=gt_output, y_pred=pred)
        detailed = compute_detailed_score(y_true=gt_output, y_pred=pred)

        basic_scores.append(score)
        detailed_scores.append(detailed)

        tqdm.write(
            f"{pid}: "
            f"P={score.precision:.3f} R={score.recall:.3f} F1={score.f1:.3f} | "
            f"Status={detailed.status_accuracy:.3f} "
            f"Onset={detailed.onset_accuracy:.3f} "
            f"EvRecall={detailed.evidence_recall:.3f} "
            f"EvPrec={detailed.evidence_precision:.3f}"
        )

        results_per_patient.append({
            "patient_id": pid,
            "num_true_conditions": len(gt_output.conditions),
            "num_pred_conditions": len(pred.conditions),
            "precision": round(score.precision, 4),
            "recall": round(score.recall, 4),
            "f1": round(score.f1, 4),
            "status_accuracy": round(detailed.status_accuracy, 4),
            "onset_accuracy": round(detailed.onset_accuracy, 4),
            "onset_partial": round(detailed.onset_partial, 4),
            "evidence_recall": round(detailed.evidence_recall, 4),
            "evidence_precision": round(detailed.evidence_precision, 4),
        })

    macro = macro_average(basic_scores)
    macro_det = macro_average_detailed(detailed_scores)

    print("\n" + "=" * 70)
    print("MACRO AVERAGES")
    print("=" * 70)
    print(f"  Condition ID:  P={macro.precision:.3f}  R={macro.recall:.3f}  F1={macro.f1:.3f}")
    if macro_det:
        print(f"  Status:        {macro_det['status_accuracy']:.3f}")
        print(f"  Onset (exact): {macro_det['onset_accuracy']:.3f}")
        print(f"  Onset (partial): {macro_det['onset_partial']:.3f}")
        print(f"  Evidence recall:    {macro_det['evidence_recall']:.3f}")
        print(f"  Evidence precision: {macro_det['evidence_precision']:.3f}")
    print("=" * 70)

    # Token usage
    print(f"\n{llm.token_summary()}")

    # Save results JSON
    if args.results_json:
        output = {
            "macro": {
                "precision": round(macro.precision, 4),
                "recall": round(macro.recall, 4),
                "f1": round(macro.f1, 4),
                **{k: round(v, 4) for k, v in macro_det.items()},
            },
            "per_patient": results_per_patient,
            "token_usage": {
                "total_calls": llm.total_calls,
                "prompt_tokens": llm.total_prompt_tokens,
                "completion_tokens": llm.total_completion_tokens,
                "total_tokens": llm.total_tokens,
            },
        }
        dump_json(Path(args.results_json), output)
        print(f"Detailed results saved to: {args.results_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
