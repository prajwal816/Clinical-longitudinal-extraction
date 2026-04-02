from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from tqdm import tqdm

from Clinical_Nlp_Extraction.data_loader import load_patient_list
from Clinical_Nlp_Extraction.inference import InferenceConfig, run_patient
from Clinical_Nlp_Extraction.llm_client import OpenAICompatibleClient, load_llm_config_from_env


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clinical condition extraction pipeline")
    p.add_argument("--data-dir", required=True, help="Path to data directory (e.g. ./Data/dev)")
    p.add_argument("--patient-list", required=True, help="Path to JSON list of patient IDs")
    p.add_argument("--output-dir", required=True, help="Directory to write patient_XX.json outputs")
    p.add_argument(
        "--taxonomy-path",
        default=str(Path(__file__).parent / "Data" / "taxonomy.json"),
        help="Path to taxonomy.json (default: Data/taxonomy.json)",
    )
    p.add_argument(
        "--cache-dir",
        default=str(Path(__file__).parent / ".cache"),
        help="Directory for cached LLM responses (default: ./.cache)",
    )
    p.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="Max output tokens per call (default: 4096)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent per-note extraction threads (default: 4)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for detailed output.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without LLM calls: write empty condition summaries (schema-valid) for wiring tests.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    data_dir = Path(args.data_dir)
    patient_list_path = Path(args.patient_list)
    output_dir = Path(args.output_dir)
    taxonomy_path = Path(args.taxonomy_path)
    cache_dir = Path(args.cache_dir)

    patient_ids = load_patient_list(str(patient_list_path))
    if not patient_ids:
        raise SystemExit("Patient list is empty.")

    llm = None
    if not args.dry_run:
        llm_cfg = load_llm_config_from_env(
            temperature=float(args.temperature), max_output_tokens=int(args.max_output_tokens)
        )
        llm = OpenAICompatibleClient(llm_cfg)

    cfg = InferenceConfig(
        data_dir=data_dir,
        taxonomy_path=taxonomy_path,
        output_dir=output_dir,
        cache_dir=cache_dir,
        concurrency=int(args.concurrency),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for pid in tqdm(patient_ids, desc="Processing patients"):
        if args.dry_run:
            # Schema-valid empty output for integration testing without API access.
            out_path = output_dir / f"{pid}.json"
            out_path.write_text(
                json.dumps({"patient_id": pid, "conditions": []}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            assert llm is not None
            run_patient(llm=llm, patient_id=pid, cfg=cfg)

    # Token usage summary
    if llm is not None:
        print(f"\n{llm.token_summary()}")

    # Write a small manifest for convenience (not used in evaluation).
    (output_dir / "_manifest.json").write_text(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "patient_list": str(patient_list_path),
                "taxonomy_path": str(taxonomy_path),
                "model": os.getenv("OPENAI_MODEL"),
                "patients": patient_ids,
                "dry_run": bool(args.dry_run),
                "concurrency": int(args.concurrency),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
