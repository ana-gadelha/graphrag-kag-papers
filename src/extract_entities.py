from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .common import clean_text, comparison_key, read_json, unique_clean_strings, write_json


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

Título: {paper['title']}
Abstract: {paper['abstract']}
"""


def extract_with_openai(paper: dict[str, Any], model: str) -> dict[str, list[str]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Instale as dependências com pip install -r requirements.txt") from exc

    client = OpenAI()
    response = client.responses.create(model=model, input=extraction_prompt(paper))
    raw = response.output_text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    data = json.loads(raw)
    expected = ("proposed_methods", "used_methods", "datasets")
    if any(not isinstance(data.get(key), list) for key in expected):
        raise ValueError("Resposta da LLM não possui as três listas esperadas.")
    return {key: unique_clean_strings(data[key]) for key in expected}


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
                entities = {
                    key: unique_clean_strings(curated[paper["id"]].get(key, []))
                    for key in ("proposed_methods", "used_methods", "datasets")
                }
            else:
                entities = extract_by_rules(paper)
        except Exception as exc:  # mantém os outros papers processáveis
            failures.append({"paper_id": paper["id"], "error": str(exc)})
            entities = {
                "proposed_methods": [],
                "used_methods": [],
                "datasets": [],
            }
        enriched.append({**paper, "extracted": entities})

    report = {
        "provider": provider,
        "model": model if provider == "openai" else None,
        "papers_processed": len(enriched),
        "failures": failures,
        "papers_with_proposed_methods": sum(
            bool(p["extracted"]["proposed_methods"]) for p in enriched
        ),
        "papers_with_used_methods": sum(
            bool(p["extracted"]["used_methods"]) for p in enriched
        ),
        "papers_with_datasets": sum(bool(p["extracted"]["datasets"]) for p in enriched),
        "review_required": provider != "curated",
        "review_note": (
            "A extração curada foi conferida contra título e abstract. "
            "No modo heuristic ou openai, confirme as entidades antes da importação. "
            "Lista vazia significa que a entidade não foi explicitamente identificada."
        ),
    }
    return enriched, report


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Extrai Method e Dataset dos abstracts.")
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
    parser.add_argument(
        "--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    )
    args = parser.parse_args()

    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY não foi definida no arquivo .env.")

    papers = read_json(args.input)
    curated = load_curated_entities(args.curated) if args.provider == "curated" else None
    enriched, report = enrich_papers(papers, args.provider, args.model, curated)
    write_json(args.output, enriched)
    write_json(args.report, report)
    print(
        f"Extração concluída para {len(enriched)} papers. "
        f"Falhas: {len(report['failures'])}. Revise {args.output} antes da importação."
    )


if __name__ == "__main__":
    main()
