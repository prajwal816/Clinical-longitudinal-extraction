from __future__ import annotations

import argparse
from pathlib import Path

from clinical_nlp_assignment.data_loader import get_valid_categories, load_taxonomy
from clinical_nlp_assignment.schemas import PatientOutput, validate_condition_taxonomy


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate output JSON files against schema and taxonomy.")
    p.add_argument("--output-dir", required=True, help="Directory containing patient_XX.json outputs")
    p.add_argument(
        "--taxonomy-path",
        default=str(Path(__file__).parent / "taxonomy.json"),
        help="Path to taxonomy.json (default: clinical_nlp_assignment/taxonomy.json)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    taxonomy = load_taxonomy(str(Path(args.taxonomy_path)))
    valid = get_valid_categories(taxonomy)

    files = sorted([p for p in out_dir.glob("patient_*.json") if p.is_file()])
    if not files:
        raise SystemExit(f"No patient_*.json found under {out_dir}")

    for fp in files:
        obj = fp.read_text(encoding="utf-8")
        parsed = PatientOutput.model_validate_json(obj)
        for c in parsed.conditions:
            validate_condition_taxonomy(c, valid)
    print(f"Validated {len(files)} files OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

