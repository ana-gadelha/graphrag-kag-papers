from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .common import (
    clean_text,
    read_json,
    unique_clean_strings,
    write_json,
)

METHOD_ALIASES: dict[str, tuple[str, ...]] = {
    "GraphRAG": ("graphrag", "graph rag", "graph-rag"),
    "LightRAG": ("lightrag",),
    "HippoRAG": ("hipporag",),
    "KAG": ("knowledge augmented generation", "knowledge-augmented generation"),
    "PathRAG": ("pathrag",),
    "GRAG": ("grag",),
    "G-Retriever": ("g-retriever",),
    "Think-on-Graph": ("think-on-graph",),
    "DynaGRAG": ("dynagrag",),
    "Youtu-GraphRAG": ("youtu-graphrag",),
    "GraphSearch": ("graphsearch",),
    "LinearRAG": ("linearrag",),
    "CG-RAG": ("cg-rag",),
    "GFM-RAG": ("gfm-rag",),
    "MedRAG": ("medrag",),
    "KET-RAG": ("ket-rag",),
    "E²GraphRAG": ("e²graphrag", "e^2graphrag", "e2graphrag"),
}

DATASET_ALIASES: dict[str, tuple[str, ...]] = {
    "HotpotQA": ("hotpotqa",),
    "MuSiQue": ("musique",),
    "2WikiMultiHopQA": ("2wikimultihopqa", "2wiki multihopqa"),
    "WebQSP": ("webqsp", "webquestionssp"),
    "ComplexWebQuestions": ("complexwebquestions", "cwq"),
    "TriviaQA": ("triviaqa",),
    "Natural Questions": ("natural questions",),
    "PubMedQA": ("pubmedqa",),
    "MedQA": ("medqa",),
    "MetaQA": ("metaqa",),
    "WikiMovies": ("wikimovies",),
    "CLUTRR": ("clutrr",),
    "FactKG": ("factkg",),
    "CRAG": ("crag benchmark",),
}


def contains_alias(text: str, alias: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text))


def extract_by_rules(paper: dict[str, Any]) -> dict[str, list[str]]:
    title = clean_text(paper["title"]).casefold()
    text = f"{paper['title']} {paper['abstract']}".casefold()

    proposed: list[str] = []
    used: list[str] = []
    for canonical, aliases in METHOD_ALIASES.items():
        in_title = any(contains_alias(title, alias.casefold()) for alias in aliases)
        in_text = any(contains_alias(text, alias.casefold()) for alias in aliases)
        if in_title:
            proposed.append(canonical)
        elif in_text:
            used.append(canonical)

    datasets = [
        canonical
        for canonical, aliases in DATASET_ALIASES.items()
        if any(contains_alias(text, alias.casefold()) for alias in aliases)
    ]
    return {
        "proposed_methods": unique_clean_strings(proposed),
        "used_methods": unique_clean_strings(used),
        "datasets": unique_clean_strings(datasets),
    }


def load_curated_entities(path: Path) -> dict[str, dict[str, list[str]]]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("O arquivo curado deve ser um objeto indexado por paper id.")
    return data


def extraction_prompt(paper: dict[str, Any]) -> str:
    return f"""Você está realizando extração de informação científica para um grafo.

Analise SOMENTE o título e o abstract fornecidos. Não use conhecimento externo e não
invente entidades. Extraia:
- proposed_methods: métodos, sistemas ou frameworks apresentados como contribuição do paper;
- used_methods: métodos que o paper declara usar, comparar ou tomar como baseline;
- datasets: conjuntos de dados ou benchmarks explicitamente mencionados.

Retorne somente JSON válido neste formato:
{{"proposed_methods": [], "used_methods": [], "datasets": []}}

Título: {paper["title"]}
Abstract: {paper["abstract"]}
"""


def extract_with_openai(paper: dict[str, Any], model: str) -> dict[str, list[str]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Instale as dependências com pip install -r requirements.txt"
        ) from exc

    with OpenAI() as client:
        response = client.responses.create(model=model, input=extraction_prompt(paper))
    if response.status != "completed":
        raise ValueError("O provedor não concluiu a extração.")
    raw = response.output_text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    data = json.loads(raw)
    return validate_entities(data)


ENTITY_FIELDS = ("proposed_methods", "used_methods", "datasets")


def validate_entities(data: Any) -> dict[str, list[str]]:
    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), list)
        or not all(isinstance(value, str) for value in data[key])
        for key in ENTITY_FIELDS
    ):
        raise ValueError(
            "Extração deve conter três listas de textos: proposed_methods, used_methods, datasets."
        )
    return {key: unique_clean_strings(data[key]) for key in ENTITY_FIELDS}


def enrich_papers(
    papers: list[dict[str, Any]],
    provider: str,
    model: str,
    curated: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for paper in papers:
        try:
            if provider == "openai":
                entities = extract_with_openai(paper, model)
            elif provider == "curated":
                if curated is None or paper["id"] not in curated:
                    raise ValueError("Paper ausente da curadoria.")
                entities = validate_entities(curated[paper["id"]])
            else:
                entities = extract_by_rules(paper)
            entities = validate_entities(entities)
        except Exception as exc:
            failures.append(
                {
                    "paper_id": paper["id"],
                    "error": f"Extração falhou ({type(exc).__name__}); confira curadoria e provedor.",
                }
            )
            # Distingue falha de uma extração válida sem entidades explícitas.
            entities = {}
        enriched.append({**paper, "extracted": entities})

    report = {
        "valid": not failures,
        "provider": provider,
        "model": model if provider == "openai" else None,
        "papers_processed": len(enriched),
        "failures": failures,
        "papers_with_proposed_methods": sum(
            bool(p["extracted"].get("proposed_methods", [])) for p in enriched
        ),
        "papers_with_used_methods": sum(
            bool(p["extracted"].get("used_methods", [])) for p in enriched
        ),
        "papers_with_datasets": sum(
            bool(p["extracted"].get("datasets", [])) for p in enriched
        ),
        "review_required": bool(failures) or provider != "curated",
        "review_note": (
            "A extração curada foi conferida contra título e abstract. "
            "No modo heuristic ou openai, confirme as entidades antes da importação. "
            "Lista vazia significa que a entidade não foi explicitamente identificada."
        ),
    }
    return enriched, report


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Extrai Method e Dataset dos abstracts."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("output/papers_normalized.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/papers_enriched.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("output/extraction_report.json")
    )
    parser.add_argument(
        "--provider", choices=("curated", "heuristic", "openai"), default="curated"
    )
    parser.add_argument(
        "--curated",
        type=Path,
        default=Path("data/curated_entities.json"),
        help="Entidades revisadas manualmente para o corpus do MVP.",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    args = parser.parse_args()

    try:
        if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY ausente.")
        from .validate_data import validate_and_normalize

        papers, validation = validate_and_normalize(read_json(args.input))
        if not validation["valid"]:
            raise ValueError("Corpus inválido: " + "; ".join(validation["errors"]))
        curated = (
            load_curated_entities(args.curated) if args.provider == "curated" else None
        )
        enriched, report = enrich_papers(papers, args.provider, args.model, curated)
    except Exception as exc:
        report = {
            "valid": False,
            "provider": args.provider,
            "failures": [
                {"error": f"Preparação da extração falhou ({type(exc).__name__})."}
            ],
        }
        write_json(args.report, report)
        raise SystemExit(f"Extração falhou; consulte {args.report}.") from None
    write_json(args.report, report)
    if report["failures"]:
        raise SystemExit(f"Extração falhou; consulte {args.report}.")
    write_json(args.output, enriched)
    print(
        f"Extração concluída para {len(enriched)} papers. "
        f"Falhas: {len(report['failures'])}. Revise {args.output} antes da importação."
    )


if __name__ == "__main__":
    main()
