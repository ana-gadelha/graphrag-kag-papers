"""Validate citation structure and exact evidence before rendering model claims.

Literal evidence checks cannot establish semantic entailment. Evaluation therefore
also requires independent human review of scientific statements.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote as url_quote

from .config import RetrievalError
from .cypher_guard import RELATIONSHIPS

ABSTENTION = "Não encontrei evidências suficientes nos artigos recuperados para responder a essa pergunta."


def _strings(value: Any, *, pattern: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(v, str) or not re.fullmatch(pattern, v) for v in value
    ):
        raise RetrievalError("Identificadores de evidência inválidos na afirmação.")
    if len(value) != len(set(value)):
        raise RetrievalError("Identificadores de evidência duplicados na afirmação.")
    return value


def _validate_path(
    path: dict, graph: dict, source_ids: set[str], sources: dict
) -> None:
    nodes = path.get("nodes", [])
    edges = path.get("edges", [])
    known_nodes = {n.get("id"): n for n in graph.get("nodes", [])}
    known_edges = {
        (e.get("source"), e.get("type"), e.get("target"))
        for e in graph.get("edges", [])
    }
    if (
        type(path.get("length")) is not int
        or not 1 <= path["length"] <= 4
        or len(nodes) != path["length"] + 1
        or len(edges) != path["length"]
    ):
        raise RetrievalError("Caminho citado tem estrutura ou comprimento inválido.")
    kind = path.get("kind", "shortest")
    if (
        kind not in {"fact", "shortest"}
        or (
            kind == "fact"
            and (path["length"] != 1 or path.get("source") != path.get("target"))
        )
        or (kind == "shortest" and path.get("source") == path.get("target"))
    ):
        raise RetrievalError(
            "Tipo de caminho incompatível com seus extremos e comprimento."
        )
    for node in nodes:
        if node.get("id") not in known_nodes:
            raise RetrievalError("Caminho cita um nó ausente do subgrafo.")
    for index, edge in enumerate(edges):
        signature = (edge.get("source"), edge.get("type"), edge.get("target"))
        adjacent = {nodes[index].get("id"), nodes[index + 1].get("id")}
        if signature not in known_edges or {signature[0], signature[2]} != adjacent:
            raise RetrievalError("Caminho cita relações ausentes ou descontínuas.")
        expected_labels = RELATIONSHIPS.get(signature[1])
        actual_labels = (
            known_nodes[signature[0]].get("label"),
            known_nodes[signature[2]].get("label"),
        )
        if actual_labels != expected_labels:
            raise RetrievalError(
                "Relação citada incompatível com os rótulos ou a direção."
            )
    endpoints = {path.get("source"), path.get("target")}
    if not endpoints <= source_ids:
        raise RetrievalError(
            "As fontes da afirmação não correspondem aos extremos do caminho."
        )
    paper_nodes = {
        n["id"] for n in nodes if known_nodes[n["id"]].get("label") == "Paper"
    }
    for citation in endpoints:
        expected = "paper:" + url_quote(str(sources[citation]["paper_id"]), safe="")
        if expected not in paper_nodes:
            raise RetrievalError("Extremo do caminho não corresponde ao artigo citado.")
    if path.get("kind") != "fact":
        expected_ends = {
            "paper:" + url_quote(str(sources[c]["paper_id"]), safe="")
            for c in endpoints
        }
        if {nodes[0]["id"], nodes[-1]["id"]} != expected_ends:
            raise RetrievalError(
                "Extremos do percurso não correspondem às fontes citadas."
            )


def _literal(value: str) -> str:
    return re.sub(r"([\\`*_{\[\]<>])", r"\\\1", value)


def _render_graph(paths: list[dict], graph: dict) -> str:
    # Structural statements are rendered from verified edges, rather than trusting
    # an arbitrary scientific assertion merely decorated with a valid path ID.
    known = {node["id"]: node for node in graph.get("nodes", [])}
    descriptions = []
    for path in paths:
        parts = []
        for index, candidate in enumerate(path["nodes"]):
            node = known[candidate["id"]]
            properties = node.get("properties", {})
            name = str(properties.get("title") or properties.get("name") or node["id"])
            parts.append(_literal(name))
            if index < len(path["edges"]):
                edge = path["edges"][index]
                parts.append(
                    f"— {edge['type']} →"
                    if edge["source"] == node["id"]
                    else f"← {edge['type']} —"
                )
        descriptions.append(" ".join(parts))
    return "; ".join(descriptions)


def validate_claims(payload: Any, evidence: dict) -> dict:
    """Return only validated claims and deterministic display text, or fail closed."""
    if (
        not isinstance(payload, dict)
        or type(payload.get("abstained")) is not bool
        or not isinstance(payload.get("claims"), list)
        or len(payload["claims"]) > 20
    ):
        raise RetrievalError(
            "A resposta deve conter claims estruturadas e abstained booleano."
        )
    if payload["abstained"]:
        if payload["claims"]:
            raise RetrievalError("Abstenção não pode conter afirmações factuais.")
        return {"answer": ABSTENTION, "abstained": True, "claims": []}
    if not payload["claims"]:
        raise RetrievalError("A resposta não contém afirmações sustentadas.")
    sources = {
        s.get("citation_id", f"S{i}"): s
        for i, s in enumerate(evidence.get("sources", []), 1)
    }
    paths = {p.get("path_id"): p for p in evidence.get("paths", [])}
    graph = evidence.get("subgraph", {})
    validated = []
    rendered = []
    for claim in payload["claims"]:
        if (
            not isinstance(claim, dict)
            or claim.get("kind") not in {"document", "graph"}
            or not isinstance(claim.get("text"), str)
            or not claim["text"].strip()
            or len(claim["text"]) > 3000
        ):
            raise RetrievalError("Afirmação deve conter kind e text válidos.")
        source_ids = _strings(claim.get("source_ids"), pattern=r"S[1-9]\d*")
        path_ids = _strings(claim.get("path_ids", []), pattern=r"P[1-9]\d*")
        if not source_ids or not set(source_ids) <= sources.keys():
            raise RetrievalError("A resposta citou uma fonte que não foi recuperada.")
        if not set(path_ids) <= paths.keys():
            raise RetrievalError("A resposta citou um caminho que não foi recuperado.")
        for path_id in path_ids:
            _validate_path(paths[path_id], graph, set(source_ids), sources)
        quotes = claim.get("quotes", [])
        if not isinstance(quotes, list):
            raise RetrievalError("Os trechos de evidência devem ser uma lista.")
        covered = set()
        normalized_quotes = []
        for item in quotes:
            if not isinstance(item, dict):
                raise RetrievalError("Trecho de evidência inválido.")
            citation, field, text = (
                item.get("source_id"),
                item.get("field"),
                item.get("quote"),
            )
            if (
                citation not in source_ids
                or field not in {"title", "abstract"}
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise RetrievalError(
                    "Trecho deve identificar fonte, campo textual e citação literal."
                )
            original = sources[citation].get(field)
            if not isinstance(original, str) or text not in original:
                raise RetrievalError(
                    "Trecho de evidência não existe no título/abstract da fonte citada."
                )
            covered.add(citation)
            start = original.index(text)
            normalized_quotes.append(
                {
                    "source_id": citation,
                    "field": field,
                    "quote": text,
                    "start": start,
                    "end": start + len(text),
                }
            )
        if claim["kind"] == "document" and covered != set(source_ids):
            raise RetrievalError(
                "Cada fonte da afirmação científica precisa de um trecho literal."
            )
        if claim["kind"] == "graph" and not path_ids:
            raise RetrievalError(
                "Afirmações sobre conexões precisam citar um caminho do grafo."
            )
        if claim["kind"] == "graph":
            used_sources = {
                paths[pid][end] for pid in path_ids for end in ("source", "target")
            }
            if used_sources != set(source_ids):
                raise RetrievalError(
                    "Afirmação de grafo contém fontes que não pertencem aos caminhos citados."
                )
        text = claim["text"].strip()
        if re.search(r"\[(?:S|P)\d+", text):
            raise RetrievalError(
                "Citações devem estar nos campos estruturados da afirmação."
            )
        if claim["kind"] == "graph":
            text = _render_graph([paths[pid] for pid in path_ids], graph)
        validated.append(
            {
                "kind": claim["kind"],
                "text": text,
                "source_ids": source_ids,
                "path_ids": path_ids,
                "quotes": normalized_quotes,
            }
        )
        # Document text is escaped to keep model output from injecting fake links.
        visible = _literal(text) if claim["kind"] == "document" else text
        rendered.append(
            visible + " " + " ".join(f"[{cid}]" for cid in source_ids + path_ids)
        )
    return {"answer": "\n\n".join(rendered), "abstained": False, "claims": validated}
