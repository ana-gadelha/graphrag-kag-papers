"""Read-only, content-addressed snapshot of the graph used by retrieval."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable

from .benchmark import canonical_json

RELATIONS = ("AUTHORED", "HAS_CONCEPT", "PROPOSES", "USES_METHOD", "EVALUATED_ON")
ENTITY_LABELS = ("Author", "Concept", "Method", "Dataset")

PAPER_SNAPSHOT_QUERY = """
MATCH (p:Paper)
OPTIONAL MATCH (p)-[:HAS_CONCEPT]->(c:Concept)
RETURN p.id AS id, p.title AS title, p.abstract AS abstract,
       p.year AS year, p.url AS url, p.source AS source,
       p.embedding_model AS embedding_model, p.embedding_hash AS embedding_hash,
       p.embedding AS embedding, collect(DISTINCT c.name) AS keywords
ORDER BY id
"""

ENTITY_SNAPSHOT_QUERY = """
MATCH (e)
WHERE e:Author OR e:Concept OR e:Method OR e:Dataset
RETURN labels(e) AS labels, e.entity_id AS entity_id, e.name AS name,
       coalesce(e.aliases, []) AS aliases
ORDER BY entity_id, name
"""

RELATION_SNAPSHOT_QUERY = """
MATCH (p:Paper)-[r]-(e)
WHERE (e:Author OR e:Concept OR e:Method OR e:Dataset)
  AND type(r) IN $relations
RETURN p.id AS paper_id, type(r) AS relation,
       labels(e) AS entity_labels, e.entity_id AS entity_id,
       e.name AS entity_name, coalesce(e.aliases, []) AS entity_aliases,
       startNode(r) = p AS paper_is_source, properties(r) AS provenance
ORDER BY paper_id, relation, entity_id, entity_name
"""

RELATION_TYPES_QUERY = """
MATCH ()-[r]->()
RETURN DISTINCT type(r) AS relation ORDER BY relation
"""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _sorted_values(values: Iterable[Any]) -> list[Any]:
    return sorted(
        (_json_safe(value) for value in values), key=lambda value: canonical_json(value)
    )


def _row(record: Any) -> dict:
    return record.data() if hasattr(record, "data") else dict(record)


def build_snapshot(
    paper_rows: list[dict],
    entity_rows: list[dict],
    relation_rows: list[dict],
    relation_type_rows: list[dict],
    expected_paper_ids: set[str],
) -> dict[str, Any]:
    """Normalize snapshot rows and require an exact corpus Paper ID match."""
    papers: list[dict] = []
    for row in paper_rows:
        vector = row.get("embedding")
        papers.append(
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "abstract": row.get("abstract"),
                "year": _json_safe(row.get("year")),
                "url": row.get("url"),
                "source": row.get("source"),
                "keywords": sorted(str(value) for value in (row.get("keywords") or [])),
                "embedding_model": row.get("embedding_model"),
                "embedding_hash": row.get("embedding_hash"),
                "embedding_vector_sha256": hashlib.sha256(
                    canonical_json(_json_safe(vector))
                ).hexdigest(),
            }
        )
    papers.sort(key=lambda paper: str(paper.get("id")))
    paper_ids = [paper.get("id") for paper in papers]
    if any(not isinstance(paper_id, str) for paper_id in paper_ids):
        raise ValueError("Graph snapshot contains a Paper without a string ID.")
    if len(paper_ids) != len(set(paper_ids)):
        raise ValueError("Graph snapshot contains duplicate Paper IDs.")
    if set(paper_ids) != expected_paper_ids:
        missing = sorted(expected_paper_ids - set(paper_ids))
        extra = sorted(set(paper_ids) - expected_paper_ids)
        raise ValueError(
            f"Graph Paper IDs differ from the corpus (missing={missing}, extra={extra})."
        )

    entities = []
    for row in entity_rows:
        labels = row.get("labels") or []
        if not isinstance(labels, list):
            labels = [str(labels)]
        if not labels or labels[0] not in ENTITY_LABELS:
            raise ValueError(
                "Graph entity is missing a retrieval label in its primary label position."
            )
        if not isinstance(row.get("entity_id"), str) or not isinstance(
            row.get("name"), str
        ):
            raise ValueError("Graph entity is missing its canonical ID or name.")
        entities.append(
            {
                "labels": [str(label) for label in labels],
                "entity_id": row.get("entity_id"),
                "canonical_name": row.get("name"),
                "aliases": sorted(str(alias) for alias in (row.get("aliases") or [])),
            }
        )
    entities.sort(key=lambda entity: canonical_json(entity))

    relations = []
    for row in relation_rows:
        relation = row.get("relation")
        if relation not in RELATIONS:
            raise ValueError(
                f"Graph snapshot contains unsupported relation: {relation!r}."
            )
        entity_labels = row.get("entity_labels") or []
        if (
            row.get("paper_id") not in expected_paper_ids
            or not isinstance(row.get("entity_id"), str)
            or not isinstance(row.get("entity_name"), str)
            or not entity_labels
            or entity_labels[0] not in ENTITY_LABELS
        ):
            raise ValueError(
                "Graph relation has an invalid paper or canonical entity endpoint."
            )
        relations.append(
            {
                "paper_id": row.get("paper_id"),
                "relation": relation,
                "entity_labels": sorted(
                    str(label) for label in (row.get("entity_labels") or [])
                ),
                "entity_id": row.get("entity_id"),
                "entity_name": row.get("entity_name"),
                "entity_aliases": sorted(
                    str(alias) for alias in (row.get("entity_aliases") or [])
                ),
                "paper_is_source": bool(row.get("paper_is_source")),
                "provenance": _json_safe(row.get("provenance") or {}),
            }
        )
    relations.sort(key=lambda relation: canonical_json(relation))

    graph_relation_types = sorted(
        {
            row.get("relation")
            for row in relation_type_rows
            if isinstance(row.get("relation"), str)
        }
    )
    ignored_relation_types = sorted(set(graph_relation_types) - set(RELATIONS))
    counts = Counter(item["relation"] for item in relations)
    snapshot = {
        "paper_count": len(papers),
        "entity_count": len(entities),
        "relation_count": len(relations),
        "papers": papers,
        "entities": entities,
        "relations": relations,
        "relation_types": list(RELATIONS),
        "observed_relation_types": sorted(set(graph_relation_types) & set(RELATIONS)),
        "ignored_relation_types": ignored_relation_types,
        "relation_counts": {
            relation: counts.get(relation, 0) for relation in RELATIONS
        },
    }
    identity = {
        key: value for key, value in snapshot.items() if key != "ignored_relation_types"
    }
    return {"sha256": hashlib.sha256(canonical_json(identity)).hexdigest(), **snapshot}


def capture_snapshot(
    driver: Any, database: str, expected_paper_ids: set[str]
) -> dict[str, Any]:
    """Read the retrieval graph using a Neo4j READ session only."""
    with driver.session(database=database, default_access_mode="READ") as session:
        paper_rows = [_row(row) for row in session.run(PAPER_SNAPSHOT_QUERY)]
        entity_rows = [_row(row) for row in session.run(ENTITY_SNAPSHOT_QUERY)]
        relation_rows = [
            _row(row)
            for row in session.run(RELATION_SNAPSHOT_QUERY, relations=list(RELATIONS))
        ]
        relation_type_rows = [_row(row) for row in session.run(RELATION_TYPES_QUERY)]
    return build_snapshot(
        paper_rows, entity_rows, relation_rows, relation_type_rows, expected_paper_ids
    )


def snapshot_from_environment(expected_paper_ids: set[str]) -> dict[str, Any]:
    """Open a configured driver, capture one read-only graph snapshot, and close it."""
    from retrieval.config import Settings, create_driver

    settings = Settings.from_env()
    driver = create_driver(settings)
    try:
        driver.verify_connectivity()
        return capture_snapshot(driver, settings.database, expected_paper_ids)
    finally:
        driver.close()
