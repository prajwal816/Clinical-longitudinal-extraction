"""
Data loading utilities for clinical condition extraction pipeline.

Handles reading patient notes, taxonomy, and patient lists with proper
line-number indexing for evidence span tracking.
"""

import json
import os
import re
from pathlib import Path
from typing import Any


def load_patient_list(patient_list_path: str) -> list[str]:
    """Load a JSON file containing a list of patient IDs.

    Args:
        patient_list_path: Path to JSON file with list of patient IDs.

    Returns:
        List of patient ID strings (e.g., ["patient_02", "patient_08"]).
    """
    with open(patient_list_path, "r", encoding="utf-8") as f:
        patient_ids = json.load(f)

    if not isinstance(patient_ids, list):
        raise ValueError(f"Patient list must be a JSON array, got {type(patient_ids)}")

    return patient_ids


def load_note(note_path: str) -> dict[str, Any]:
    """Load a single clinical note file and return its lines (1-indexed).

    Args:
        note_path: Path to a .md clinical note file.

    Returns:
        Dictionary with:
            - 'lines': dict mapping 1-indexed line numbers to line content
            - 'raw_text': the full raw text
            - 'note_id': filename without extension
    """
    with open(note_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    raw_lines = raw_text.split("\n")
    # 1-indexed line mapping
    lines = {i + 1: line for i, line in enumerate(raw_lines)}

    note_id = Path(note_path).stem  # e.g., "text_0"

    return {
        "note_id": note_id,
        "lines": lines,
        "raw_text": raw_text,
        "num_lines": len(raw_lines),
    }


def load_patient_notes(data_dir: str, patient_id: str) -> list[dict[str, Any]]:
    """Load all clinical notes for a patient, ordered chronologically.

    Notes are ordered by filename (text_0.md, text_1.md, ...) which
    corresponds to chronological order per the problem statement.

    Args:
        data_dir: Root data directory containing patient folders.
        patient_id: Patient identifier (e.g., "patient_06").

    Returns:
        List of note dictionaries ordered chronologically.
    """
    patient_dir = os.path.join(data_dir, patient_id)

    if not os.path.isdir(patient_dir):
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    # Find all .md files and sort by numeric index
    note_files = []
    for fname in os.listdir(patient_dir):
        if fname.endswith(".md"):
            note_files.append(fname)

    # Sort by numeric index in filename (text_0.md, text_1.md, ...)
    def sort_key(fname: str) -> int:
        match = re.search(r"(\d+)", fname)
        return int(match.group(1)) if match else 0

    note_files.sort(key=sort_key)

    if not note_files:
        raise FileNotFoundError(f"No .md note files found in {patient_dir}")

    notes = []
    for fname in note_files:
        note_path = os.path.join(patient_dir, fname)
        note = load_note(note_path)
        notes.append(note)

    return notes


def load_taxonomy(taxonomy_path: str) -> dict[str, Any]:
    """Load the condition taxonomy from JSON.

    Args:
        taxonomy_path: Path to taxonomy.json.

    Returns:
        Parsed taxonomy dictionary.
    """
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    return taxonomy


def get_valid_categories(taxonomy: dict) -> dict[str, list[str]]:
    """Extract valid category → subcategory mappings from taxonomy.

    Args:
        taxonomy: Parsed taxonomy dictionary.

    Returns:
        Dict mapping category keys to lists of valid subcategory keys.
    """
    categories = {}
    for cat_key, cat_data in taxonomy.get("condition_categories", {}).items():
        subcats = list(cat_data.get("subcategories", {}).keys())
        categories[cat_key] = subcats

    return categories


def get_valid_statuses(taxonomy: dict) -> list[str]:
    """Extract valid status values from taxonomy.

    Args:
        taxonomy: Parsed taxonomy dictionary.

    Returns:
        List of valid status strings.
    """
    return list(taxonomy.get("status_values", {}).keys())


def format_note_with_line_numbers(note: dict) -> str:
    """Format a clinical note with line numbers for LLM prompts.

    Args:
        note: Note dictionary from load_note().

    Returns:
        String with each line prefixed by its 1-indexed line number.
    """
    formatted_lines = []
    for line_no in sorted(note["lines"].keys()):
        line_content = note["lines"][line_no]
        formatted_lines.append(f"{line_no}: {line_content}")

    return "\n".join(formatted_lines)


def load_ground_truth(labels_dir: str, patient_id: str) -> dict[str, Any] | None:
    """Load ground truth labels for a patient (training data only).

    Args:
        labels_dir: Path to labels directory.
        patient_id: Patient identifier.

    Returns:
        Parsed ground truth JSON, or None if not found.
    """
    label_path = os.path.join(labels_dir, f"{patient_id}.json")
    if not os.path.exists(label_path):
        return None

    with open(label_path, "r", encoding="utf-8") as f:
        return json.load(f)
