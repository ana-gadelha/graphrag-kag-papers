"""Read-only vector/graph retrieval with an opt-in validated LLM channel."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from typing import Any
from urllib.parse import quote

from src.common import comparison_key

from .config import RetrievalError
from .embeddings import (
    EMBEDDING_MODEL,
    INDEX_INFO_QUERY,
    check_index,
    content_hash,
    validate_vector,
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
RETURN labels(e)[0] AS label, e.name AS name, e.entity_id AS entity_id,
       coalesce(e.aliases, []) AS aliases ORDER BY label, name
"""

DIRECT_QUERY = """
UNWIND $entities AS entity
MATCH (p:Paper)-[:AUTHORED|HAS_CONCEPT|PROPOSES|USES_METHOD|EVALUATED_ON]-(e)
WHERE entity.label IN labels(e)
  AND ((entity.entity_id IS NOT NULL AND e.entity_id = entity.entity_id)
       OR (entity.entity_id IS NULL AND e.name = entity.name))
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
       labels(e)[0] AS entity_label, e.name AS entity_name, e.entity_id AS entity_id,
       properties(r) AS provenance,
       startNode(r) = p AS paper_is_source
ORDER BY paper_id, type, entity_label, entity_name
"""


PATH_SOURCE_LIMIT = 6

PATHS_QUERY = """
UNWIND $pairs AS pair
MATCH (a:Paper {id: pair.source_id}), (b:Paper {id: pair.target_id})
MATCH path = shortestPath(
    (a)-[:AUTHORED|HAS_CONCEPT|PROPOSES|USES_METHOD|EVALUATED_ON*1..4]-(b)
)
WHERE all(n IN nodes(path) WHERE n:Paper OR n:Author OR n:Concept OR n:Method OR n:Dataset)
RETURN pair.source_id AS source_id, pair.target_id AS target_id,
       length(path) AS length,
       [n IN nodes(path) | {
           db_id: elementId(n), label: labels(n)[0],
           key: CASE WHEN n:Paper THEN n.id ELSE coalesce(n.entity_id, n.name) END,
           name: n.name, entity_id: n.entity_id,
           title: n.title, year: n.year, url: n.url
       }] AS path_nodes,
       [r IN relationships(path) | {
           source: elementId(startNode(r)), target: elementId(endNode(r)), type: type(r),
           provenance: properties(r)
       }] AS path_edges
ORDER BY source_id, target_id
"""

_QUERY_STAGES = {
    INDEX_INFO_QUERY: "Verificação do índice vetorial",
    PAPERS_QUERY: "Leitura dos artigos",
    VECTOR_QUERY: "Busca vetorial",
    ENTITIES_QUERY: "Leitura das entidades",
    DIRECT_QUERY: "Correspondências diretas",
    EXPAND_QUERY: "Expansão no grafo",
    FACTS_QUERY: "Relações das fontes",
    PATHS_QUERY: "Caminhos mínimos entre fontes",
}


class _TracedSession:
    """Registra somente Cypher e parâmetros de consulta, sem dados da conexão."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self.queries: list[dict] = []

    def run(self, query: Any, **parameters: Any) -> Any:
        diagnostic = deepcopy(parameters)
        if "embedding" in diagnostic:
            diagnostic["embedding"] = {
                "dimensions": len(parameters["embedding"]),
                "values_omitted": True,
            }
        result = self.session.run(query, **parameters)
        text = str(query)
        self.queries.append(
            {
                "stage": _QUERY_STAGES.get(text, "Cypher gerado pelo LLM"),
                "cypher": text.strip(),
                "parameters": diagnostic,
            }
        )
        return result


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
        names = [entity.get("name") or "", *(entity.get("aliases") or [])]
        if any(
            name and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", question_key)
            for name in map(_normalized, names)
        ):
            matches.append(entity)
    return matches


def match_explicit_papers(question: str, papers: dict[str, dict]) -> list[str]:
    """Resolve only literal corpus IDs/URLs, in their order in the question."""
    matches = []
    for paper_id, paper in papers.items():
        aliases = {paper_id, paper.get("url") or ""}
        if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", paper_id):
            aliases.update(
                f"{scheme}://arxiv.org/{path}/{paper_id}{suffix}"
                for scheme in ("http", "https")
                for path, suffix in (("abs", ""), ("pdf", ""), ("pdf", ".pdf"))
            )
        elif re.match(r"10\.\d{4,9}/", paper_id):
            aliases.add(f"https://doi.org/{paper_id}")
        positions = [
            match.start()
            for alias in aliases
            if alias
            if (
                match := re.search(
                    rf"(?<![\w./-]){re.escape(alias)}(?![\w/-]|\.\w)",
                    question,
                    re.IGNORECASE,
                )
            )
        ]
        if positions:
            matches.append((min(positions), paper_id))
    return [paper_id for _, paper_id in sorted(matches)]


def fuse_rankings(vector_ids: list[str], graph_ids: list[str]) -> list[dict]:
    """RRF de pesos iguais; pontuação é ranking, não probabilidade."""
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for channel, paper_ids in (("vector", vector_ids), ("graph", graph_ids)):
        for rank, paper_id in enumerate(dict.fromkeys(paper_ids), start=1):
            scores[paper_id] += 1 / (60 + rank)
            ranks[paper_id][channel] = rank
    return [
        {
            "paper_id": paper_id,
            "score": scores[paper_id],
            "ranks": ranks[paper_id],
            "retrieved_by": list(ranks[paper_id]),
        }
        for paper_id in sorted(scores, key=lambda pid: (-scores[pid], pid))
    ]


def _node_id(label: str, key: str) -> str:
    return f"{label.lower()}:{quote(key, safe='')}"


def assemble_evidence(
    ranked: list[dict],
    papers: dict[str, dict],
    facts: list[dict],
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
            {
                **{key: f[key] for key in ("type", "entity_label", "entity_name")},
                "entity_id": f.get("entity_id"),
                "provenance": f.get("provenance", {}),
            }
            for f in paper_facts
        ]
        source = {
            **item,
            "citation_id": f"S{position}",
            **{key: paper.get(key) for key in ("title", "year", "url", "abstract")},
            "authors": sorted(
                {f["entity_name"] for f in paper_facts if f["type"] == "AUTHORED"}
            ),
            "relations": relations,
        }
        sources.append(source)
        paper_node = _node_id("Paper", item["paper_id"])
        nodes[paper_node] = {
            "id": paper_node,
            "label": "Paper",
            "properties": {
                key: paper.get(key)
                for key in ("id", "title", "year", "url", "abstract", "source")
            },
        }
        for fact in paper_facts:
            entity_node = _node_id(
                fact["entity_label"], fact.get("entity_id") or fact["entity_name"]
            )
            nodes[entity_node] = {
                "id": entity_node,
                "label": fact["entity_label"],
                "properties": {
                    "name": fact["entity_name"],
                    "entity_id": fact.get("entity_id"),
                },
            }
            start, end = (
                (paper_node, entity_node)
                if fact["paper_is_source"]
                else (entity_node, paper_node)
            )
            key = (start, fact["type"], end)
            edges[key] = {
                "source": start,
                "target": end,
                "type": fact["type"],
                "provenance": fact.get("provenance", {}),
            }
        if not any(f["type"] == "EVALUATED_ON" for f in paper_facts):
            warnings.append(
                f"[S{position}] Dataset não identificado nas evidências disponíveis."
            )
    return {
        "sources": sources,
        "subgraph": {"nodes": list(nodes.values()), "edges": list(edges.values())},
        "warnings": warnings if sources else [],
    }


def connect_sources(session: Any, evidence: dict) -> list[dict]:
    """Materializa caminhos e conectores sem promover novos artigos a fontes."""
    sources = evidence["sources"]
    pairs = [
        {"source_id": a["paper_id"], "target_id": b["paper_id"]}
        for a, b in combinations(sources[:PATH_SOURCE_LIMIT], 2)
    ]
    if not pairs:
        return []
    citations = {source["paper_id"]: source["citation_id"] for source in sources}
    graph = evidence["subgraph"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = {
        (edge["source"], edge["type"], edge["target"]): edge for edge in graph["edges"]
    }
    paths = []
    for row in _records(session, PATHS_QUERY, pairs=pairs):
        path_nodes = []
        ids = {}
        for raw in row["path_nodes"]:
            label, key = raw["label"], raw["key"]
            node_id = _node_id(label, key)
            ids[raw["db_id"]] = node_id
            if node_id not in nodes:
                properties = (
                    {"id": key, **{k: raw.get(k) for k in ("title", "year", "url")}}
                    if label == "Paper"
                    else {
                        "name": raw.get("name") or key,
                        "entity_id": raw.get("entity_id"),
                    }
                )
                nodes[node_id] = {
                    "id": node_id,
                    "label": label,
                    "properties": properties,
                    "connector": label == "Paper" and key not in citations,
                }
            path_nodes.append(nodes[node_id])
        path_edges = []
        for raw in row["path_edges"]:
            edge = {
                "source": ids[raw["source"]],
                "target": ids[raw["target"]],
                "type": raw["type"],
                "provenance": raw.get("provenance", {}),
            }
            edge = edges.get((edge["source"], edge["type"], edge["target"]), edge)
            edges[(edge["source"], edge["type"], edge["target"])] = edge
            path_edges.append(edge)
        paths.append(
            {
                "source": citations[row["source_id"]],
                "target": citations[row["target_id"]],
                "length": row["length"],
                "nodes": path_nodes,
                "edges": path_edges,
            }
        )
    graph.update(nodes=list(nodes.values()), edges=list(edges.values()))
    return paths


def retrieve(
    driver: Any,
    database: str,
    encoder: Any,
    question: str,
    top_k: int,
    *,
    use_planner: bool = False,
    mode: str = "hybrid",
) -> dict:
    """Consulta o índice e o grafo, verifica embeddings e materializa o contexto."""
    if mode not in {"hybrid", "vector"}:
        raise RetrievalError("Modo de recuperação deve ser hybrid ou vector.")
    with driver.session(database=database, default_access_mode="READ") as raw_session:
        session = _TracedSession(raw_session)
        check_index(session)
        rows = _records(session, PAPERS_QUERY)
        if not rows:
            return {
                "sources": [],
                "subgraph": {"nodes": [], "edges": []},
                "warnings": [],
                "queries": session.queries,
                "paths": [],
            }
        papers = {p["id"]: p for p in rows}
        for paper in rows:
            try:
                validate_vector(paper.get("embedding"))
            except (RetrievalError, TypeError, ValueError) as exc:
                raise RetrievalError(
                    "Embeddings ausentes ou inválidos. Execute python -m retrieval.index_embeddings."
                ) from exc
            if paper.get("embedding_model") != EMBEDDING_MODEL or paper.get(
                "embedding_hash"
            ) != content_hash(paper):
                raise RetrievalError(
                    "Embeddings incompatíveis ou desatualizados. Execute "
                    "python -m retrieval.index_embeddings (ou --force para trocar o modelo)."
                )
        vector = validate_vector(encoder.encode(question, normalize_embeddings=True))
        vector_rows = _records(session, VECTOR_QUERY, limit=8, embedding=vector)
        vector_ids = [r["paper_id"] for r in vector_rows if r["paper_id"] in papers]
        explicit_ids = (
            match_explicit_papers(question, papers) if mode == "hybrid" else []
        )
        vocabulary = _records(session, ENTITIES_QUERY) if mode == "hybrid" else []
        entities = match_entities(question, vocabulary)
        direct = (
            _records(session, DIRECT_QUERY, entities=entities, limit=20)
            if entities
            else []
        )
        seeds = list(
            dict.fromkeys(explicit_ids + [r["paper_id"] for r in direct] + vector_ids)
        )
        expanded = (
            _records(session, EXPAND_QUERY, seed_ids=seeds, per_seed=5, limit=20)
            if seeds and mode == "hybrid"
            else []
        )
        planner = {"status": "disabled", "paper_ids": [], "queries": [], "warnings": []}
        if use_planner and mode == "hybrid":
            from .planner import plan_and_retrieve

            planner = plan_and_retrieve(
                raw_session,
                question,
                seed_ids=seeds[:8],
                vocabulary=vocabulary,
                corpus_ids=list(papers),
            )
            session.queries.extend(planner.get("queries", []))
        direct_scores = {r["paper_id"]: r["direct_matches"] for r in direct}
        shared_scores = {r["paper_id"]: r["shared_connections"] for r in expanded}
        graph_ids = sorted(
            (direct_scores.keys() | shared_scores.keys()) & papers.keys(),
            key=lambda pid: (
                -direct_scores.get(pid, 0),
                -shared_scores.get(pid, 0),
                pid,
            ),
        )[:20]
        planned_ids = [pid for pid in planner.get("paper_ids", []) if pid in papers]
        graph_ids = list(dict.fromkeys(explicit_ids + planned_ids + graph_ids))
        ranked = fuse_rankings(vector_ids, graph_ids)
        for item in ranked:
            pid = item["paper_id"]
            item["graph_origins"] = (
                (["explicit_paper_id"] if pid in explicit_ids else [])
                + (["llm_cypher"] if pid in planned_ids else [])
                + (
                    ["fixed_graph"]
                    if pid in direct_scores or pid in shared_scores
                    else []
                )
            )
        # Vizinhos de expansão presentes nos dois canais não devem excluir os
        # próprios papers ligados às entidades explicitamente citadas na pergunta.
        ranked.sort(
            key=lambda item: (
                explicit_ids.index(item["paper_id"])
                if item["paper_id"] in explicit_ids
                else len(explicit_ids),
                -int(item["paper_id"] in planned_ids),
                -direct_scores.get(item["paper_id"], 0),
                -item["score"],
                item["paper_id"],
            )
        )
        ranked = ranked[:top_k]
        facts = (
            _records(session, FACTS_QUERY, paper_ids=[r["paper_id"] for r in ranked])
            if ranked and mode == "hybrid"
            else []
        )
        evidence = assemble_evidence(ranked, papers, facts)
        evidence["paths"] = (
            connect_sources(session, evidence) if mode == "hybrid" else []
        )
        assign_evidence_ids(evidence)
        evidence["planner"] = planner
        evidence["warnings"].extend(planner.get("warnings", []))
        if len(explicit_ids) > top_k:
            evidence["warnings"].append(
                "Há mais artigos explicitamente citados que o limite de fontes; "
                "foram priorizados na ordem em que aparecem na pergunta."
            )
        evidence["queries"] = session.queries
        return evidence


def _edge_id(edge: dict) -> str:
    signature = [edge["source"], edge["type"], edge["target"]]
    return "edge:" + hashlib.sha256(repr(signature).encode()).hexdigest()[:20]


def assign_evidence_ids(evidence: dict) -> None:
    """Make shortest paths and one-edge graph facts citeable with stable edge IDs."""
    graph = evidence.get("subgraph", {})
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    edges = graph.get("edges", [])
    for edge in edges:
        edge["id"] = _edge_id(edge)
    paths = [path for path in evidence.get("paths", []) if path.get("kind") != "fact"]
    evidence["paths"] = paths
    for path in paths:
        path["kind"] = path.get("kind", "shortest")
        for edge in path.get("edges", []):
            edge["id"] = _edge_id(edge)
    citations = {
        _node_id("Paper", s["paper_id"]): s["citation_id"]
        for s in evidence.get("sources", [])
    }
    for edge in edges:
        paper_id = edge["source"] if edge["source"] in citations else edge["target"]
        if paper_id not in citations:
            continue
        paths.append(
            {
                "kind": "fact",
                "source": citations[paper_id],
                "target": citations[paper_id],
                "length": 1,
                "nodes": [nodes[edge["source"]], nodes[edge["target"]]],
                "edges": [edge],
            }
        )
    for position, path in enumerate(paths, 1):
        path["path_id"] = f"P{position}"
