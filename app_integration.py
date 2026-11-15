"""Adapta os resultados do retrieval ao formato consumido pela interface."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.parse import urlparse

from retrieval import answer_question


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _authors_text(value: Any) -> str:
    if isinstance(value, str):
        return _text(value, "Autoria não informada")
    if isinstance(value, (list, tuple)):
        authors = [_text(author) for author in value]
        authors = [author for author in authors if author]
        return ", ".join(authors) or "Autoria não informada"
    return "Autoria não informada"


def _node_id(node: dict[str, Any], properties: dict[str, Any], fallback: str) -> str:
    return _text(
        node.get("id"),
        _text(
            node.get("entity_id"),
            _text(properties.get("entity_id"), _text(properties.get("name"), fallback)),
        ),
    )


def _canonical_edge_key(edge: dict[str, Any]) -> str:
    properties = _mapping(edge.get("properties"))
    payload = {
        "source": _text(edge.get("source"), _text(edge.get("origem"), "unknown")),
        "target": _text(edge.get("target"), _text(edge.get("destino"), "unknown")),
        "type": _text(edge.get("type"), _text(edge.get("relacao"), "RELATED_TO")),
        "properties": properties,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _edge_id(edge: dict[str, Any], fallback: str) -> str:
    return _text(edge.get("id"), _text(edge.get("edge_id"), fallback))


def _query_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _planner_details(result: dict[str, Any]) -> dict[str, Any]:
    raw = _mapping(result.get("planner"))
    generated = result.get("generated_queries", raw.get("generated_queries"))
    if generated is None:
        generated = result.get("generated_query", raw.get("generated_query"))
    if generated is None:
        generated = raw.get("queries", raw.get("query", raw.get("cypher")))

    # Keep only diagnostic fields. Runtime settings and credentials stay in Python.
    return {
        "status": _text(raw.get("status"), _text(result.get("planner_status"))),
        "reason": _text(raw.get("reason"), _text(raw.get("warning"))),
        "generated_queries": _query_list(generated),
    }


def adaptar_resultado(result: dict[str, Any]) -> dict[str, Any]:
    """Converte o contrato público do retrieval para os campos da interface.

    Campos novos são opcionais para que resultados gravados antes do planejador
    e das afirmações continuem renderizando sem condições especiais no app.
    """
    raw_sources = result.get("sources")
    sources = raw_sources if isinstance(raw_sources, list) else []
    fontes = []
    for position, raw_source in enumerate(sources, start=1):
        source = _mapping(raw_source)
        citation_id = _text(source.get("citation_id"), f"S{position}")
        title = _text(
            source.get("title"), _text(source.get("paper_id"), "Artigo sem título")
        )
        fontes.append(
            {
                **source,
                "citation_id": citation_id,
                "titulo": title,
                "autores": _authors_text(source.get("authors")),
                "ano": source.get("year"),
            }
        )

    raw_subgraph = _mapping(result.get("subgraph"))
    raw_nodes = raw_subgraph.get("nodes")
    raw_edges = raw_subgraph.get("edges")
    nos = []
    for position, raw_node in enumerate(
        raw_nodes if isinstance(raw_nodes, list) else [], start=1
    ):
        node = _mapping(raw_node)
        properties = _mapping(node.get("properties"))
        node_id = _node_id(node, properties, f"node-{position}")
        display_name = _text(
            properties.get("title"), _text(properties.get("name"), node_id)
        )
        connector = node.get("connector") is True
        nos.append(
            {
                "id": node_id,
                "label": display_name,
                "tipo": _text(node.get("label"), "Unknown"),
                "properties": properties,
                "connector": connector,
                "source_ids": node.get("source_ids", node.get("source_id", [])),
                "path_ids": node.get("path_ids", []),
                "provenance": node.get("provenance", {}),
            }
        )

    arestas = []
    for raw_edge in raw_edges if isinstance(raw_edges, list) else []:
        edge = _mapping(raw_edge)
        properties = _mapping(edge.get("properties"))
        fallback_id = (
            "edge:"
            + hashlib.sha256(_canonical_edge_key(edge).encode("utf-8")).hexdigest()[:16]
        )
        arestas.append(
            {
                "id": _edge_id(edge, fallback_id),
                "origem": _text(
                    edge.get("source"), _text(edge.get("origem"), "unknown")
                ),
                "destino": _text(
                    edge.get("target"), _text(edge.get("destino"), "unknown")
                ),
                "relacao": _text(
                    edge.get("type"), _text(edge.get("relacao"), "RELATED_TO")
                ),
                "properties": properties,
                "source_ids": edge.get("source_ids", edge.get("source_id", [])),
                "path_ids": edge.get("path_ids", []),
                "provenance": edge.get("provenance", {}),
            }
        )

    raw_warnings = result.get("warnings")
    warnings = (
        [_text(warning) for warning in raw_warnings if _text(warning)]
        if isinstance(raw_warnings, list)
        else []
    )
    raw_claims = result.get("claims")
    claims = (
        [_mapping(claim) for claim in raw_claims if isinstance(claim, dict)]
        if isinstance(raw_claims, list)
        else []
    )
    queries = result.get("executed_queries", result.get("queries", []))
    return {
        "resposta": _text(result.get("answer")),
        "fontes": fontes,
        "nos": nos,
        "arestas": arestas,
        "warnings": warnings,
        "queries": queries if isinstance(queries, list) else [],
        "paths": result.get("paths", [])
        if isinstance(result.get("paths", []), list)
        else [],
        "claims": claims,
        "planner": _planner_details(result),
        "confidence": _mapping(result.get("confidence")),
    }


def consultar(pergunta: str) -> dict[str, Any]:
    """Consulta o backend diretamente e adapta sua resposta para a UI."""
    return adaptar_resultado(answer_question(pergunta))


def url_http_valida(value: Any) -> bool:
    """Só oferece links externos com esquema HTTP ou HTTPS."""
    if not isinstance(value, str):
        return False
    try:
        parsed = urlparse(value.strip())
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
