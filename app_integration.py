"""Adapta os resultados do retrieval ao formato consumido pela interface."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from graphviz import Digraph, escape, nohtml
from graphviz.quoting import attr_list, quote

from retrieval import answer_question

NODE_COLORS = {
    "Paper": "#cfe8ff",
    "Author": "#e8d5ff",
    "Concept": "#fff1c2",
    "Method": "#ffe8b3",
    "Dataset": "#d5f5d5",
}


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


def adaptar_resultado(result: dict[str, Any]) -> dict[str, Any]:
    """Converte o contrato público do retrieval para os campos da interface."""
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
    for raw_node in raw_nodes if isinstance(raw_nodes, list) else []:
        node = _mapping(raw_node)
        node_id = _text(node.get("id"), "unknown")
        properties = _mapping(node.get("properties"))
        display_name = _text(
            properties.get("title"),
            _text(properties.get("name"), node_id),
        )
        connector = node.get("connector") is True
        if connector:
            display_name += " (conector)"
        nos.append(
            {
                "id": node_id,
                "label": display_name,
                "tipo": _text(node.get("label"), "Unknown"),
                "properties": properties,
                "connector": connector,
            }
        )

    arestas = []
    for raw_edge in raw_edges if isinstance(raw_edges, list) else []:
        edge = _mapping(raw_edge)
        arestas.append(
            {
                "origem": _text(edge.get("source"), "unknown"),
                "destino": _text(edge.get("target"), "unknown"),
                "relacao": _text(edge.get("type"), "RELATED_TO"),
            }
        )

    raw_warnings = result.get("warnings")
    warnings = (
        [_text(warning) for warning in raw_warnings if _text(warning)]
        if isinstance(raw_warnings, list)
        else []
    )
    return {
        "resposta": _text(result.get("answer")),
        "fontes": fontes,
        "nos": nos,
        "arestas": arestas,
        "warnings": warnings,
        "queries": result.get("queries", []),
        "paths": result.get("paths", []),
        "confidence": _mapping(result.get("confidence")),
    }


def consultar(pergunta: str) -> dict[str, Any]:
    """Consulta o backend diretamente e adapta sua resposta para a UI."""
    return adaptar_resultado(answer_question(pergunta))


def montar_dot(nos: list[dict[str, Any]], arestas: list[dict[str, Any]]) -> Digraph:
    """Cria um grafo direcionado usando o escaping do pacote Graphviz."""

    def dot_text(value: Any) -> str:
        # graphviz.escape protege barras e sequências especiais; a forma \n
        # mantém quebras de linha dentro de um único identificador/atributo DOT.
        escaped = escape(_text(value))
        return nohtml(escaped.replace("\n", r"\n"))

    graph = Digraph("G")
    graph.attr(rankdir="LR")
    graph.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontname="Helvetica",
    )

    for node in nos:
        node_id = dot_text(_text(node.get("id"), "unknown"))
        node_type = _text(node.get("tipo"), "")
        label = dot_text(_text(node.get("label"), _text(node.get("id"), "unknown")))
        graph.node(
            node_id,
            label=label,
            fillcolor=NODE_COLORS.get(node_type, "#eeeeee"),
        )

    for edge in arestas:
        source = dot_text(_text(edge.get("origem"), "unknown"))
        target = dot_text(_text(edge.get("destino"), "unknown"))
        relation = dot_text(_text(edge.get("relacao"), "RELATED_TO"))
        graph.body.append(
            f"\t{quote(source)} -> {quote(target)}"
            f"{attr_list(kwargs={'label': relation})}\n"
        )
    return graph


def url_http_valida(value: Any) -> bool:
    """Só oferece links externos com esquema HTTP ou HTTPS."""
    if not isinstance(value, str):
        return False
    try:
        parsed = urlparse(value.strip())
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
