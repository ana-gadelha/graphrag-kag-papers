from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def clean_text(value: str) -> str:
    """Remove espaços repetidos sem alterar acentos ou capitalização."""
    return re.sub(r"\s+", " ", value).strip()


def comparison_key(value: str) -> str:
    """Cria uma chave apenas para comparação e deduplicação."""
    normalized = unicodedata.normalize("NFKD", clean_text(value))
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_marks.casefold()


def unique_clean_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_text(value)
        key = comparison_key(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
