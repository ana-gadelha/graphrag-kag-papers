from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .common import clean_text, comparison_key, read_json, unique_clean_strings, write_json


REQUIRED_FIELDS = (
    "id",
    "source",
    "url",
    "title",
    "authors",
    "year",
    "abstract",
    "keywords",
)


def normalize_paper(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": clean_text(str(raw.get("id", ""))),
        "source": clean_text(str(raw.get("source", ""))),
        "url": clean_text(str(raw.get("url", ""))),
        "title": clean_text(str(raw.get("title", ""))),
        "authors": unique_clean_strings(raw.get("authors", [])),
        "year": raw.get("year"),
        "abstract": clean_text(str(raw.get("abstract", ""))),
        "keywords": unique_clean_strings(raw.get("keywords", [])),
    }


def validate_and_normalize(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("O arquivo deve conter uma lista JSON de papers.")

    errors: list[str] = []
    warnings: list[str] = []
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for position, raw in enumerate(data, start=1):
        if not isinstance(raw, dict):
            errors.append(f"Item {position}: não é um objeto JSON.")
            continue

        missing = [field for field in REQUIRED_FIELDS if field not in raw]
        if missing:
            errors.append(f"Item {position}: campos ausentes: {', '.join(missing)}.")
            continue

        if not isinstance(raw["authors"], list) or not all(
            isinstance(item, str) for item in raw["authors"]
        ):
            errors.append(f"Item {position}: authors deve ser uma lista de textos.")
            continue

        if not isinstance(raw["keywords"], list) or not all(
            isinstance(item, str) for item in raw["keywords"]
        ):
            errors.append(f"Item {position}: keywords deve ser uma lista de textos.")
            continue

        paper = normalize_paper(raw)
        paper_id = comparison_key(paper["id"])

        for field in ("id", "source", "url", "title", "abstract"):
            if not paper[field]:
                errors.append(f"Item {position}: campo {field} vazio.")

        if not paper["authors"]:
            errors.append(f"Item {position}: lista de autores vazia.")
        if not paper["keywords"]:
            warnings.append(f"Item {position} ({paper['id']}): sem palavras-chave.")

        if not isinstance(paper["year"], int) or not 2023 <= paper["year"] <= 2026:
            errors.append(
                f"Item {position} ({paper['id']}): ano fora do intervalo 2023–2026."
            )

        if paper_id in seen_ids:
            errors.append(f"Item {position}: id duplicado: {paper['id']}.")
        seen_ids.add(paper_id)
        normalized.append(paper)

    year_counts = Counter(paper["year"] for paper in normalized)
    source_counts = Counter(paper["source"] for paper in normalized)
    report = {
        "valid": not errors,
        "papers_received": len(data),
        "papers_normalized": len(normalized),
        "errors": errors,
        "warnings": warnings,
        "papers_by_year": dict(sorted(year_counts.items())),
        "papers_by_source": dict(source_counts),
        "unique_authors": len(
            {
                comparison_key(author)
                for paper in normalized
                for author in paper["authors"]
            }
        ),
        "unique_concepts": len(
            {
                comparison_key(keyword)
                for paper in normalized
                for keyword in paper["keywords"]
            }
        ),
    }
    return normalized, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida e normaliza papers.json.")
    parser.add_argument("--input", type=Path, default=Path("data/papers.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("output/papers_normalized.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("output/validation_report.json")
    )
    args = parser.parse_args()

    normalized, report = validate_and_normalize(read_json(args.input))
    write_json(args.report, report)
    if not report["valid"]:
        raise SystemExit(
            f"Validação falhou com {len(report['errors'])} erro(s). "
            f"Consulte {args.report}."
        )
    write_json(args.output, normalized)
    print(
        f"Validação concluída: {len(normalized)} papers, "
        f"{report['unique_authors']} autores e "
        f"{report['unique_concepts']} conceitos únicos."
    )


if __name__ == "__main__":
    main()
