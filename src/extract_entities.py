from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .common import clean_text, read_json, unique_clean_strings, write_json
from .entity_resolution import (
    DEFAULT_REGISTRY_PATH,
    Entity,
    EntityRegistry,
    resolve_metadata_entities,
)

# The seed registry is the sole source of reviewed Method/Dataset aliases.
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
    "KG²RAG": ("KG$^2$RAG",),
    "Microsoft GraphRAG": ("Microsoft's Graph-RAG", "Microsoft’s Graph-RAG"),
    "Cross-encoder reranking": ("cross-encoder re-ranking", "cross encoder reranking"),
}

DATASET_ALIASES: dict[str, tuple[str, ...]] = {
    "HotpotQA": ("hotpotqa",),
    "MuSiQue": ("musique",),
    "2WikiMultiHopQA": ("2wikimultihopqa", "2wiki multihopqa", "2wiki"),
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

ENTITY_FIELDS = ("proposed_methods", "used_methods", "datasets")
RELATION_TO_FIELD = {
    "PROPOSES": "proposed_methods",
    "USES_METHOD": "used_methods",
    "EVALUATED_ON": "datasets",
}
RELATION_LABEL = {
    "PROPOSES": "Method",
    "USES_METHOD": "Method",
    "EVALUATED_ON": "Dataset",
}
EXTRACTION_SCHEMA_VERSION = "1"

_METHOD_PROPOSAL_CUE = re.compile(
    r"\b(?:propos\w*|present\w*|introduc\w*|develop\w*|design\w*|"
    r"our\s+(?:method|approach|framework|system|model)|new\s+"
    r"(?:method|approach|framework|system|model))\b",
    re.IGNORECASE,
)
_METHOD_USAGE_CUE = re.compile(
    r"\b(?:us(?:e|es|ed|ing)|adopt\w*|compar\w*|baseline\w*|"
    r"benchmark\w*|evaluat\w*|experiment\w*)\b",
    re.IGNORECASE,
)
_METHOD_ADDITIONAL_USAGE_CUE = re.compile(
    r"\b(?:integrat(?:e|es|ed|ing)|combin(?:e|es|ed|ing)|"
    r"utili[sz](?:e|es|ed|ing)|appl(?:y|ies|ied|ying)|"
    r"incorporat(?:e|es|ed|ing)|outperform(?:s|ed|ing)?|"
    r"(?:faster|slower|better|worse)(?:\s+[\w-]+){0,3}\s+than|"
    r"(?:speedup|improvement)(?:\s+[\w-]+){0,3}\s+(?:over|against))\b",
    re.IGNORECASE,
)
_DATASET_CUE = re.compile(
    r"\b(?:dataset\w*|benchmark\w*|evaluat\w*|test\w*|experiment\w*|"
    r"results?\s+on|performance\s+on)\b",
    re.IGNORECASE,
)
_NUMERIC_DATASET_CUE = re.compile(
    r"\b(?:improvement\s+of\s+\d+(?:\.\d+)?\s*%\s+on\s+\S+|"
    r"\d+(?:\.\d+)?\s*%\s+on\s+\S+\s+in\s+terms\s+of\s+"
    r"(?:F1(?:\s+score)?|accuracy|exact\s+match)\b)",
    re.IGNORECASE,
)
_NEGATED_CUE_PREFIX = re.compile(
    r"\b(?:not|never|without|no|rather\s+than|instead\s+of)\s+(?:\w+\s+){0,2}$",
    re.IGNORECASE,
)


def contains_alias(text: str, alias: str) -> bool:
    """Match whole entity tokens; hyphens stay part of a name."""
    return bool(
        re.search(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", text, re.IGNORECASE)
    )


def _quote_for(text: str, start: int, end: int) -> str:
    """Return a sentence-sized exact quote containing the selected mention."""
    left_candidates = [text.rfind(mark, 0, start) for mark in ".!?\n"]
    left = max(left_candidates) + 1
    right_candidates = [
        position for mark in ".!?\n" if (position := text.find(mark, end)) >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    quote = text[left:right].strip()
    if not quote or text.find(quote) < 0:
        raise ValueError("Could not locate a local evidence quote.")
    return quote


def _find_surface(
    text: str, entity: Entity, preferred_surface: str | None = None
) -> tuple[str, int, int] | None:
    candidates = [preferred_surface] if preferred_surface else []
    candidates.extend(
        sorted(entity.aliases, key=lambda alias: (-len(alias), alias.casefold()))
    )
    checked: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate.casefold() in checked:
            continue
        checked.add(candidate.casefold())
        pattern = re.compile(
            rf"(?<![\w-])({re.escape(candidate)})(?![\w-])", re.IGNORECASE
        )
        match = pattern.search(text)
        if match:
            surface = match.group(1)
            return surface, match.start(1), match.end(1)
    return None


def _relation_is_supported(relation: str, quote: str, field: str | None = None) -> bool:
    if relation == "PROPOSES":
        title_cue = field == "title" and re.search(
            r"\b(?:approach|framework|method|system|model)\b", quote, re.IGNORECASE
        )
        return bool(_METHOD_PROPOSAL_CUE.search(quote) or title_cue)
    if relation == "USES_METHOD":
        if _METHOD_USAGE_CUE.search(quote):
            return True
        additional_cue = _METHOD_ADDITIONAL_USAGE_CUE
    elif relation == "EVALUATED_ON":
        if _DATASET_CUE.search(quote):
            return True
        additional_cue = _NUMERIC_DATASET_CUE
    else:
        return False
    # New cues must describe an affirmative action or a numeric result, not a denial.
    return any(
        not _NEGATED_CUE_PREFIX.search(quote[: match.start()])
        for match in additional_cue.finditer(quote)
    )


def _locate_mention(
    raw: Any, paper: dict[str, Any], *, require_relation_cue: bool = True
) -> dict[str, Any]:
    required = {"surface_form", "label", "relation", "field", "quote"}
    if not isinstance(raw, dict) or not required.issubset(raw):
        raise ValueError(
            "Each mention must include surface, label, relation, field and quote."
        )
    surface = raw.get("surface_form")
    label = raw.get("label")
    relation = raw.get("relation")
    field = raw.get("field")
    quote = raw.get("quote")
    if (
        not isinstance(surface, str)
        or not clean_text(surface)
        or not isinstance(quote, str)
        or not quote
        or field not in {"title", "abstract"}
        or relation not in RELATION_TO_FIELD
        or label != RELATION_LABEL[relation]
    ):
        raise ValueError(
            "Mention has an invalid field, label, relation or evidence quote."
        )
    source = paper[field]
    quote_start = source.find(quote)
    if quote_start < 0:
        raise ValueError(
            "Evidence quote is not an exact substring of its source field."
        )
    local_start = quote.find(surface)
    if local_start < 0:
        raise ValueError("Surface form does not occur exactly in its evidence quote.")
    start = quote_start + local_start
    end = start + len(surface)
    if source[start:end] != surface:
        raise ValueError("Mention offsets do not identify the exact surface form.")
    if require_relation_cue and not _relation_is_supported(relation, quote, field):
        raise ValueError("Evidence quote does not support the requested relation.")
    return {
        "surface_form": surface,
        "label": label,
        "relation": relation,
        "field": field,
        "quote": quote,
        "start": start,
        "end": end,
    }


def validate_mentions(
    data: Any, paper: dict[str, Any], *, require_relation_cues: bool = True
) -> list[dict[str, Any]]:
    if not isinstance(data, dict) or not isinstance(data.get("mentions"), list):
        raise ValueError(
            "Extraction must contain a mentions list; an empty list is valid."
        )
    mentions = [
        _locate_mention(raw, paper, require_relation_cue=require_relation_cues)
        for raw in data["mentions"]
    ]
    seen: set[tuple[str, str, str, int]] = set()
    for mention in mentions:
        key = (
            mention["label"],
            mention["relation"],
            mention["field"],
            mention["start"],
        )
        if key in seen:
            raise ValueError("Duplicate entity mention in extraction response.")
        seen.add(key)
    return mentions


def _derive_lists(mentions: list[dict[str, Any]]) -> dict[str, list[str]]:
    values = {field: [] for field in ENTITY_FIELDS}
    for mention in mentions:
        field = RELATION_TO_FIELD[mention["relation"]]
        values[field].append(mention["canonical_name"])
    return {field: unique_clean_strings(names) for field, names in values.items()}


def _resolve_mentions(
    mentions: list[dict[str, Any]], registry: EntityRegistry
) -> list[dict[str, Any]]:
    resolved = []
    for mention in mentions:
        entity = registry.resolve(mention["label"], mention["surface_form"])
        resolved.append(
            {**mention, **entity.as_dict(), "origin": "title_abstract_extraction"}
        )
    return resolved


def _curated_mentions(
    paper: dict[str, Any], entities: dict[str, list[str]], registry: EntityRegistry
):
    mentions: list[dict[str, Any]] = []
    for field, names in entities.items():
        relation = {
            "proposed_methods": "PROPOSES",
            "used_methods": "USES_METHOD",
            "datasets": "EVALUATED_ON",
        }[field]
        label = RELATION_LABEL[relation]
        for name in names:
            entity = registry.resolve(label, name)
            located = None
            for source_field in ("title", "abstract"):
                found = _find_surface(paper[source_field], entity, name)
                if found is None:
                    continue
                surface, start, end = found
                quote = _quote_for(paper[source_field], start, end)
                located = {
                    "surface_form": surface,
                    "label": label,
                    "relation": relation,
                    "field": source_field,
                    "quote": quote,
                }
                # Curated labels are human-reviewed; the lexical cue check is
                # reserved for automated/heuristic relation assignment.
                break
            if located is None:
                raise ValueError(
                    f"Curated {label} has no local quote supporting {relation}: {name}"
                )
            mentions.append(_locate_mention(located, paper, require_relation_cue=False))
    return _resolve_mentions(mentions, registry)


def _heuristic_surfaces(paper: dict[str, Any], entities: list[Entity]):
    """Prefer complete registered names over nested short aliases (RAG in Graph RAG)."""
    selected: dict[tuple[str, str], list[tuple[str, int, int]]] = {}
    for field in ("title", "abstract"):
        for label in ("Method", "Dataset"):
            candidates = []
            seen = set()
            for entity in entities:
                if entity.label != label:
                    continue
                for alias in entity.aliases:
                    pattern = rf"(?<![\w-])({re.escape(alias)})(?![\w-])"
                    for match in re.finditer(pattern, paper[field], re.IGNORECASE):
                        start, end = match.span(1)
                        key = (entity.entity_id, start, end)
                        if key not in seen:
                            seen.add(key)
                            candidates.append(
                                (entity.entity_id, match.group(1), start, end)
                            )
            occupied = []
            for entity_id, surface, start, end in sorted(
                candidates, key=lambda item: (-(item[3] - item[2]), item[2], item[0])
            ):
                if any(
                    start < other_end and other_start < end
                    for other_start, other_end in occupied
                ):
                    continue
                occupied.append((start, end))
                selected.setdefault((entity_id, field), []).append(
                    (surface, start, end)
                )
    return selected


def _heuristic_mentions(paper: dict[str, Any], registry: EntityRegistry):
    mentions: list[dict[str, Any]] = []
    entities = [e for e in registry.entities() if e.label in {"Method", "Dataset"}]
    surfaces = _heuristic_surfaces(paper, entities)
    for entity in entities:
        relation_candidates = (
            (("PROPOSES", "title"), ("USES_METHOD", "abstract"))
            if entity.label == "Method"
            else (("EVALUATED_ON", "abstract"), ("EVALUATED_ON", "title"))
        )
        accepted = False
        for relation, source_field in relation_candidates:
            for surface, start, end in sorted(
                surfaces.get((entity.entity_id, source_field), []),
                key=lambda item: item[1],
            ):
                quote = _quote_for(paper[source_field], start, end)
                if _relation_is_supported(relation, quote, source_field):
                    mentions.append(
                        _locate_mention(
                            {
                                "surface_form": surface,
                                "label": entity.label,
                                "relation": relation,
                                "field": source_field,
                                "quote": quote,
                            },
                            paper,
                        )
                    )
                    accepted = True
                    break
            if accepted:
                break
    return _resolve_mentions(mentions, registry)


def extraction_prompt(paper: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    system = """You extract Method and Dataset relations from one scientific paper.
Use only the supplied title and abstract. Return JSON with exactly one top-level
key, `mentions`, containing objects with `surface_form`, `label`, `relation`,
`field`, and `quote`. Labels are Method or Dataset. Relations are PROPOSES,
USES_METHOD, and EVALUATED_ON. A Method is PROPOSES only when the title/abstract
states that the paper presents, proposes, introduces, develops, or calls it its
own approach. A Method is USES_METHOD only when the paper says it uses, adopts,
compares, evaluates, or uses it as a baseline. A Dataset is EVALUATED_ON only
when the paper says it evaluates, tests, benchmarks, or reports results on it.
The quote must be copied exactly from the named title or abstract and contain
the exact surface form. Do not infer from outside knowledge. Return an empty
mentions array when no relation is supported by an exact quote."""
    payload = {
        "paper_id": paper["id"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "output_schema": {
            "mentions": [
                {
                    "surface_form": "exact text",
                    "label": "Method | Dataset",
                    "relation": "PROPOSES | USES_METHOD | EVALUATED_ON",
                    "field": "title | abstract",
                    "quote": "exact contiguous source text",
                }
            ]
        },
    }
    return system, payload


def extract_with_llm(
    paper: dict[str, Any], model: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from retrieval.llm import accept_json, generate_json, stage_settings

    settings = stage_settings("extraction")
    if model:
        settings = replace(settings, model=model)
    system, payload = extraction_prompt(paper)
    response = generate_json(
        settings,
        system,
        payload,
        namespace="text-to-graph-extraction",
        version=EXTRACTION_SCHEMA_VERSION,
    )
    mentions = validate_mentions(response, paper)
    # Cache promotion happens only after source evidence and relation validation.
    accept_json(response)
    receipt = getattr(response, "receipt", None)
    return mentions, {
        "model": (receipt.get("effective_model") if isinstance(receipt, dict) else None)
        or settings.model,
        "cache_hit": receipt.get("cache_hit", False)
        if isinstance(receipt, dict)
        else False,
        "usage": receipt.get("usage") if isinstance(receipt, dict) else None,
        "seconds": receipt.get("seconds") if isinstance(receipt, dict) else None,
    }


def extract_with_openai(
    paper: dict[str, Any], model: str | None = None
) -> dict[str, list[str]]:
    """Compatibility wrapper; extraction now uses the stage-scoped JSON client."""
    mentions, _ = extract_with_llm(paper, model)
    # Kept for older call sites; enrich_papers uses the evidence-bearing variant.
    return _derive_lists(_resolve_mentions(mentions, EntityRegistry.load()))


def load_curated_entities(path: Path) -> dict[str, dict[str, list[str]]]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("Curated file must be an object indexed by paper id.")
    return data


def validate_entities(data: Any) -> dict[str, list[str]]:
    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), list)
        or not all(isinstance(value, str) for value in data[key])
        for key in ENTITY_FIELDS
    ):
        raise ValueError("Curated extraction must contain the three entity lists.")
    return {key: unique_clean_strings(data[key]) for key in ENTITY_FIELDS}


def extract_by_rules(paper: dict[str, Any]) -> dict[str, list[str]]:
    registry = EntityRegistry.load()
    mentions = _heuristic_mentions(paper, registry)
    return _derive_lists(mentions)


def enrich_papers(
    papers: list[dict[str, Any]],
    provider: str = "auto",
    model: str | None = None,
    curated: dict[str, dict[str, list[str]]] | None = None,
    registry: EntityRegistry | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if provider not in {"auto", "openai", "curated", "heuristic"}:
        raise ValueError(f"Unsupported extraction provider: {provider}")
    from .validate_data import validate_and_normalize

    normalized, validation = validate_and_normalize(papers)
    if not validation["valid"]:
        raise ValueError("Corpus inválido: " + "; ".join(validation["errors"]))
    papers = normalized
    registry = registry or EntityRegistry.load()
    enriched: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    receipts: list[dict[str, Any]] = []
    for paper in papers:
        try:
            if provider in {"auto", "openai"}:
                mentions, receipt = extract_with_llm(paper, model)
                receipts.append({"paper_id": paper["id"], **receipt})
            elif provider == "curated":
                if curated is None or paper["id"] not in curated:
                    raise ValueError("Paper absent from curated extraction.")
                entities = validate_entities(curated[paper["id"]])
                mentions = _curated_mentions(paper, entities, registry)
            else:
                mentions = _heuristic_mentions(paper, registry)

            # Revalidate all modes and enrich the IDs only after quote checks pass.
            raw_mentions = [
                {
                    key: mention[key]
                    for key in ("surface_form", "label", "relation", "field", "quote")
                }
                for mention in mentions
            ]
            located = validate_mentions(
                {"mentions": raw_mentions},
                paper,
                require_relation_cues=provider != "curated",
            )
            resolved = _resolve_mentions(located, registry)
            extracted = {
                "mentions": resolved,
                **_derive_lists(resolved),
                "provenance": {
                    "origin": "title_abstract_extraction",
                    "provider": "llm" if provider in {"auto", "openai"} else provider,
                    "model": receipts[-1]["model"]
                    if provider in {"auto", "openai"}
                    else None,
                    "schema_version": EXTRACTION_SCHEMA_VERSION,
                },
            }
            extracted["metadata_entities"] = resolve_metadata_entities(paper, registry)
        except Exception as exc:
            failures.append(
                {
                    "paper_id": paper.get("id", "unknown"),
                    "error": f"Extração inválida ({type(exc).__name__}); confira evidências e configuração.",
                }
            )
            extracted = {}
        enriched.append({**paper, "extracted": extracted})

    selected_model = next(
        (item["model"] for item in receipts if item.get("model")), None
    )
    report = {
        "valid": not failures,
        "provider": "llm" if provider in {"auto", "openai"} else provider,
        "model": selected_model,
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
        "papers_with_no_supported_mentions": sum(
            "mentions" in p["extracted"] and not p["extracted"]["mentions"]
            for p in enriched
        ),
        "receipts": receipts,
        "review_required": provider != "curated" or bool(failures),
        "review_note": (
            "Method/Dataset extraídos de título/abstract com quote e offsets verificáveis. "
            "Autores e palavras-chave vêm dos metadados do paper. Lista vazia é válida "
            "quando nenhum relacionamento tem evidência explícita."
        ),
    }
    return enriched, report


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Extrai relações Method/Dataset com evidência local."
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
        "--provider",
        choices=("auto", "curated", "heuristic", "openai"),
        default="auto",
        help="auto usa o LLM configurado para extração; curated/heuristic são modos offline explícitos.",
    )
    parser.add_argument(
        "--curated", type=Path, default=Path("data/curated_entities.json")
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--model", help="Override explícito do modelo configurado para extraction."
    )
    args = parser.parse_args()

    report: dict[str, Any]
    try:
        from .validate_data import validate_and_normalize

        papers, validation = validate_and_normalize(read_json(args.input))
        if not validation["valid"]:
            raise ValueError("Corpus inválido: " + "; ".join(validation["errors"]))
        registry = EntityRegistry.load(args.registry)
        curated = (
            load_curated_entities(args.curated) if args.provider == "curated" else None
        )
        enriched, report = enrich_papers(
            papers, args.provider, args.model, curated, registry
        )
    except Exception as exc:
        report = {
            "valid": False,
            "provider": args.provider,
            "papers_processed": 0,
            "failures": [
                {
                    "paper_id": "unknown",
                    "error": f"Extração não iniciada ({type(exc).__name__}).",
                }
            ],
            "review_required": True,
        }
        enriched = []
    write_json(args.report, report)
    if not report["valid"]:
        raise SystemExit(
            f"Extração falhou em {len(report['failures'])} paper(s). Consulte {args.report}."
        )
    # Save only the complete, validated extraction so unknown exact names persist.
    registry.save(args.registry)
    write_json(args.output, enriched)
    print(f"Extração concluída: {len(enriched)} papers, provider={report['provider']}.")


if __name__ == "__main__":
    main()
