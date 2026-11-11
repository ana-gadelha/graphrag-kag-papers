"""Recuperação vetorial e por relações, sem escrita ou Cypher gerado por LLM."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any
from urllib.parse import quote

from src.common import comparison_key

from .config import RetrievalError
from .embeddings import (
    EMBEDDING_MODEL, check_index, content_hash, validate_vector,
)


PAPERS_QUERY = """
MATCH (p:Paper)
OPTIONAL MATCH (p)-[:HAS_CONCEPT]->(c:Concept)
RETURN p.id AS id, p.title AS title, p.abstract AS abstract,
       p.year AS year, p.url AS url, p.source AS source,
       p.embedding_model AS embedding_model, p.embedding_hash AS embedding_hash,
       p.embedding AS embedding, collect(DISTINCT c.name) AS keywords
ORDER BY id
"""

VECTOR_QUERY = """
CALL db.index.vector.queryNodes('paper_embedding', $limit, $embedding)
YIELD node, score
RETURN node.id AS paper_id, score ORDER BY score DESC, paper_id
"""

ENTITIES_QUERY = """
MATCH (e)
WHERE e:Author OR e:Concept OR e:Method OR e:Dataset
RETURN labels(e)[0] AS label, e.name AS name ORDER BY label, name
"""

DIRECT_QUERY = """
UNWIND $entities AS entity
MATCH (p:Paper)-[:AUTHORED|HAS_CONCEPT|PROPOSES|USES_METHOD|EVALUATED_ON]-(e)
WHERE entity.label IN labels(e) AND e.name = entity.name
RETURN p.id AS paper_id, count(DISTINCT e) AS direct_matches
ORDER BY direct_matches DESC, paper_id LIMIT $limit
"""

EXPAND_QUERY = """
UNWIND $seed_ids AS seed_id
MATCH (seed:Paper {id: seed_id})
CALL (seed) {
    MATCH (seed)-[:AUTHORED|HAS_CONCEPT|PROPOSES|USES_METHOD|EVALUATED_ON]-(e)
          -[:AUTHORED|HAS_CONCEPT|PROPOSES|USES_METHOD|EVALUATED_ON]-(p:Paper)
    WHERE p <> seed AND (e:Author OR e:Concept OR e:Method OR e:Dataset)
    RETURN p.id AS paper_id, count(DISTINCT e) AS connections
    ORDER BY connections DESC, paper_id LIMIT $per_seed
}
RETURN paper_id, sum(connections) AS shared_connections
ORDER BY shared_connections DESC, paper_id LIMIT $limit
"""

FACTS_QUERY = """
MATCH (p:Paper)-[r:AUTHORED|HAS_CONCEPT|PROPOSES|USES_METHOD|EVALUATED_ON]-(e)
WHERE p.id IN $paper_ids AND (e:Author OR e:Concept OR e:Method OR e:Dataset)
RETURN DISTINCT p.id AS paper_id, type(r) AS type,
       labels(e)[0] AS entity_label, e.name AS entity_name,
       startNode(r) = p AS paper_is_source
ORDER BY paper_id, type, entity_label, entity_name
"""


def _records(session: Any, query: str, **parameters: Any) -> list[dict]:
    return [record.data() for record in session.run(query, **parameters)]


def _normalized(text: str) -> str:
    value = re.sub(r"[-_\s]+", " ", comparison_key(text)).strip()
    value = re.sub(r"\bgraph rag\b", "graphrag", value)
    return re.sub(r"\bknowledge augmented generation\b", "kag", value)


def match_entities(question: str, entities: list[dict]) -> list[dict]:
    """Nomes literais e aliases mínimos, sem confundir KAG com outra palavra."""
    question_key = _normalized(question)
    matches = []
    for entity in entities:
        name = _normalized(entity.get("name") or "")
        if name and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", question_key):
            matches.append(entity)
    return matches


def fuse_rankings(vector_ids: list[str], graph_ids: list[str]) -> list[dict]:
    """RRF de pesos iguais; pontuação é ranking, não probabilidade."""
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for channel, paper_ids in (("vector", vector_ids), ("graph", graph_ids)):
        for rank, paper_id in enumerate(dict.fromkeys(paper_ids), start=1):
            scores[paper_id] += 1 / (60 + rank)
            ranks[paper_id][channel] = rank
    return [
        {"paper_id": paper_id, "score": scores[paper_id],
         "ranks": ranks[paper_id], "retrieved_by": list(ranks[paper_id])}
        for paper_id in sorted(scores, key=lambda pid: (-scores[pid], pid))
    ]


def _node_id(label: str, key: str) -> str:
    return f"{label.lower()}:{quote(key, safe='')}"


def assemble_evidence(
    ranked: list[dict], papers: dict[str, dict], facts: list[dict],
) -> dict:
    sources = []
    nodes: dict[str, dict] = {}
    edges: dict[tuple, dict] = {}
    warnings = [
        "Resultados limitados às fontes recuperadas e aos metadados/abstracts; "
        "não representam necessariamente todos os trabalhos do corpus."
    ]
    for position, item in enumerate(ranked, start=1):
        paper = papers[item["paper_id"]]
        paper_facts = [f for f in facts if f["paper_id"] == item["paper_id"]]
        relations = [
            {key: f[key] for key in ("type", "entity_label", "entity_name")}
            for f in paper_facts
        ]
        source = {
            **item, "citation_id": f"S{position}",
            **{key: paper.get(key) for key in ("title", "year", "url", "abstract")},
            "authors": sorted({f["entity_name"] for f in paper_facts
                               if f["type"] == "AUTHORED"}),
            "relations": relations,
        }
        sources.append(source)
        paper_node = _node_id("Paper", item["paper_id"])
        nodes[paper_node] = {
            "id": paper_node, "label": "Paper",
            "properties": {key: paper.get(key) for key in
                           ("id", "title", "year", "url", "abstract", "source")},
        }
        for fact in paper_facts:
            entity_node = _node_id(fact["entity_label"], fact["entity_name"])
            nodes[entity_node] = {
                "id": entity_node, "label": fact["entity_label"],
                "properties": {"name": fact["entity_name"]},
            }
            start, end = ((paper_node, entity_node) if fact["paper_is_source"]
                          else (entity_node, paper_node))
            key = (start, fact["type"], end)
            edges[key] = {"source": start, "target": end, "type": fact["type"]}
        if not any(f["type"] == "EVALUATED_ON" for f in paper_facts):
            warnings.append(
                f"[S{position}] Dataset não identificado nas evidências disponíveis."
            )
    return {"sources": sources,
            "subgraph": {"nodes": list(nodes.values()), "edges": list(edges.values())},
            "warnings": warnings if sources else []}


def retrieve(driver: Any, database: str, encoder: Any, question: str, top_k: int) -> dict:
    """Consulta o índice e o grafo, verifica embeddings e materializa o contexto."""
    with driver.session(database=database, default_access_mode="READ") as session:
        check_index(session)
        rows = _records(session, PAPERS_QUERY)
        if not rows:
            return {"sources": [], "subgraph": {"nodes": [], "edges": []}, "warnings": []}
        papers = {p["id"]: p for p in rows}
        for paper in rows:
            try:
                validate_vector(paper.get("embedding"))
            except (RetrievalError, TypeError, ValueError) as exc:
                raise RetrievalError(
                    "Embeddings ausentes ou inválidos. Execute python -m retrieval.index_embeddings."
                ) from exc
            if (paper.get("embedding_model") != EMBEDDING_MODEL
                    or paper.get("embedding_hash") != content_hash(paper)):
                raise RetrievalError(
                    "Embeddings incompatíveis ou desatualizados. Execute "
                    "python -m retrieval.index_embeddings (ou --force para trocar o modelo)."
                )
        vector = validate_vector(encoder.encode(question, normalize_embeddings=True))
        vector_rows = _records(session, VECTOR_QUERY, limit=8, embedding=vector)
        vector_ids = [r["paper_id"] for r in vector_rows if r["paper_id"] in papers]
        entities = match_entities(question, _records(session, ENTITIES_QUERY))
        direct = _records(session, DIRECT_QUERY, entities=entities, limit=20) if entities else []
        seeds = list(dict.fromkeys([r["paper_id"] for r in direct] + vector_ids))
        expanded = (_records(session, EXPAND_QUERY, seed_ids=seeds, per_seed=5, limit=20)
                    if seeds else [])
        direct_scores = {r["paper_id"]: r["direct_matches"] for r in direct}
        shared_scores = {r["paper_id"]: r["shared_connections"] for r in expanded}
        graph_ids = sorted(
            (direct_scores.keys() | shared_scores.keys()) & papers.keys(),
            key=lambda pid: (-direct_scores.get(pid, 0), -shared_scores.get(pid, 0), pid),
        )[:20]
        ranked = fuse_rankings(vector_ids, graph_ids)
        # Vizinhos de expansão presentes nos dois canais não devem excluir os
        # próprios papers ligados às entidades explicitamente citadas na pergunta.
        ranked.sort(key=lambda item: (
            -direct_scores.get(item["paper_id"], 0), -item["score"], item["paper_id"],
        ))
        ranked = ranked[:top_k]
        facts = (_records(session, FACTS_QUERY, paper_ids=[r["paper_id"] for r in ranked])
                 if ranked else [])
        # DISTINCT in Cypher plus deterministic sorting make evidence stable across runs.
        return assemble_evidence(ranked, papers, facts)
