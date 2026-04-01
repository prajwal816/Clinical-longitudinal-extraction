from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def read_env_required(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


_MD_BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.*?)\*(?!\*)")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)


def clean_markdown_line(line: str) -> str:
    """
    Lightweight normalization for prompting while preserving line text enough that
    evidence spans can be copied verbatim from the original line.
    """
    s = line.replace("\u00a0", " ")
    s = _HTML_COMMENT_RE.sub("", s)
    s = _MD_BOLD_RE.sub(r"\1", s)
    s = _MD_ITALIC_RE.sub(r"\1", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def normalize_condition_name(name: str) -> str:
    s = name.casefold().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-()/]", "", s)
    return s


@dataclass(frozen=True)
class ConditionKey:
    category: str
    subcategory: str
    name_norm: str

