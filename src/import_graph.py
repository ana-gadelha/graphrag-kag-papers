from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from .common import read_json
from .entity_resolution import DEFAULT_REGISTRY_PATH, EntityRegistry, entity_id_for
from .extract_entities import (
    ENTITY_FIELDS,
    RELATION_LABEL,
    RELATION_TO_FIELD,
    validate_mentions,
    validate_entities,
)
from .validate_data import validate_and_normalize


def schema_statements(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"//[^\n]*", "", text)
    return [statement.strip() for statement in text.split(";") if statement.strip()]


def apply_schema(driver: Any, database: str, path: Path) -> None:
    with driver.session(database=database) as session:
        for statement in schema_statements(path):
            session.run(statement).consume()


OWNED_TYPES = ("AUTHORED", "HAS_CONCEPT", "PROPOSES", "USES_METHOD", "EVALUATED_ON")
OWNED_EDGE_PREDICATE = """
(type(r) = 'AUTHORED' AND n:Author AND endNode(r) = p)
OR (type(r) = 'HAS_CONCEPT' AND n:Concept AND startNode(r) = p)
OR (type(r) IN ['PROPOSES', 'USES_METHOD'] AND n:Method AND startNode(r) = p)
OR (type(r) = 'EVALUATED_ON' AND n:Dataset AND startNode(r) = p)
"""
PAPER_QUERY = (
    """
MERGE (p:Paper {id: $id})
SET p.title=$title, p.year=$year, p.abstract=$abstract, p.url=$url, p.source=$source
WITH p
OPTIONAL MATCH (p)-[r]-(n)
WHERE """
    + OWNED_EDGE_PREDICATE
    + """
DELETE r
"""
)

RELATION_LABELS = {
    "AUTHORED": "Author",
    "HAS_CONCEPT": "Concept",
    **{relation: label for relation, label in RELATION_LABEL.items()},
}


def _validate_entity_identity(entity: Any, *, label: str, field: str, origin: str) -> None:
    if not isinstance(entity, dict):
        raise ValueError(f"Invalid {label} entity object.")
    required = {"entity_id", "canonical_name", "surface_form", "label", "relation", "field"}
    if not required.issubset(entity):
        raise ValueError(f"{label} entity is missing identity or provenance fields.")
    canonical = entity.get("canonical_name")
    surface = entity.get("surface_form")
    if (
        entity.get("label") != label
        or entity.get("field") != field
        or entity.get("origin") != origin
        or not isinstance(canonical, str)
        or not canonical.strip()
        or not isinstance(surface, str)
        or not surface.strip()
        or entity.get("entity_id") != entity_id_for(label, canonical)
    ):
        raise ValueError(f"Invalid {label} entity identity or provenance.")


def _validate_extracted(
    paper: dict[str, Any], raw: Any, registry: EntityRegistry
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{paper['id']}: extracted must be an object.")
    if not isinstance(raw.get("mentions"), list):
        raise ValueError(f"{paper['id']}: extracted.mentions must be present, even when empty.")
    provenance = raw.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("origin") != "title_abstract_extraction"
        or provenance.get("provider") not in {"llm", "curated", "heuristic"}
        or provenance.get("schema_version") != "1"
    ):
        raise ValueError(f"{paper['id']}: invalid text extraction provenance.")

    plain_mentions = []
    for mention in raw["mentions"]:
        if not isinstance(mention, dict):
            raise ValueError(f"{paper['id']}: mention must be an object.")
        plain_mentions.append(
            {
                key: mention.get(key)
                for key in ("surface_form", "label", "relation", "field", "quote")
            }
        )
    checked = validate_mentions(
        {"mentions": plain_mentions},
        paper,
        require_relation_cues=provenance.get("provider") != "curated",
    )
    for supplied, located in zip(raw["mentions"], checked):
        for key in ("start", "end"):
            if type(supplied.get(key)) is not int or supplied[key] != located[key]:
                raise ValueError(f"{paper['id']}: mention {key} offset is invalid.")
        if supplied.get("origin") != "title_abstract_extraction":
            raise ValueError(f"{paper['id']}: mention origin is missing or invalid.")
        _validate_entity_identity(
            supplied,
            label=located["label"],
            field=located["field"],
            origin="title_abstract_extraction",
        )
        if supplied["entity_id"] != entity_id_for(located["label"], supplied["canonical_name"]):
            raise ValueError(f"{paper['id']}: mention entity_id differs from its canonical name.")
        resolved = registry.resolve(located["label"], located["surface_form"])
        if (
            resolved.entity_id != supplied["entity_id"]
            or resolved.canonical_name != supplied["canonical_name"]
        ):
            raise ValueError(f"{paper['id']}: mention does not resolve through the global registry.")

    expected = {field: [] for field in ENTITY_FIELDS}
    for mention in raw["mentions"]:
        expected[RELATION_TO_FIELD[mention["relation"]]].append(mention["canonical_name"])
    expected = {field: unique for field, unique in expected.items()}
    from .common import unique_clean_strings

    expected = {field: unique_clean_strings(names) for field, names in expected.items()}
    lists = validate_entities({field: raw.get(field) for field in ENTITY_FIELDS})
    if lists != expected:
        raise ValueError(f"{paper['id']}: derived entity lists differ from structured mentions.")

    metadata = raw.get("metadata_entities")
    if not isinstance(metadata, list):
        raise ValueError(f"{paper['id']}: metadata_entities must be present.")
    expected_metadata = {
        ("Author", "AUTHORED", "authors", surface) for surface in paper["authors"]
    } | {
        ("Concept", "HAS_CONCEPT", "keywords", surface) for surface in paper["keywords"]
    }
    actual_metadata = set()
    for entity in metadata:
        if not isinstance(entity, dict):
            raise ValueError(f"{paper['id']}: metadata entity must be an object.")
        label = entity.get("label")
        field = entity.get("field")
        relation = "AUTHORED" if label == "Author" else "HAS_CONCEPT" if label == "Concept" else None
        if relation is None:
            raise ValueError(f"{paper['id']}: invalid metadata entity label.")
        _validate_entity_identity(
            entity, label=label, field=field, origin="paper_metadata"
        )
        if entity.get("relation") != relation:
            raise ValueError(f"{paper['id']}: metadata relation differs from its label.")
        resolved = registry.resolve(label, entity["surface_form"])
        if (
            resolved.entity_id != entity["entity_id"]
            or resolved.canonical_name != entity["canonical_name"]
        ):
            raise ValueError(f"{paper['id']}: metadata entity conflicts with the global registry.")
        actual_metadata.add((label, relation, field, entity["surface_form"]))
    if actual_metadata != expected_metadata or len(metadata) != len(expected_metadata):
        raise ValueError(f"{paper['id']}: metadata entity list does not match authors/keywords.")

    return {**raw, **lists, "mentions": raw["mentions"], "metadata_entities": metadata}


def validate_import_input(data: Any) -> list[dict[str, Any]]:
    """Validate the entire corpus and its evidence before any database access."""
    papers, report = validate_and_normalize(data)
    if not report["valid"]:
        raise ValueError("Entrada inválida: " + "; ".join(report["errors"]))
    registry = EntityRegistry.load()
    for paper, raw in zip(papers, data):
        paper["extracted"] = _validate_extracted(paper, raw.get("extracted"), registry)
    return papers


def _merge_entity(tx: Any, entity: dict[str, Any]) -> None:
    label = entity["label"]
    if label not in RELATION_LABELS.values():
        raise ValueError("Unsupported entity label in import.")
    tx.run(
        f"""MERGE (n:{label} {{entity_id: $entity_id}})
        ON CREATE SET n.name=$canonical_name, n.canonical_name=$canonical_name,
                      n.aliases=$aliases, n.identity_origin=$identity_origin
        ON MATCH SET n.name=$canonical_name, n.canonical_name=$canonical_name,
                     n.aliases=reduce(acc=coalesce(n.aliases, []), alias IN $aliases |
                       CASE WHEN alias IN acc THEN acc ELSE acc + alias END),
                     n.identity_origin=coalesce(n.identity_origin, $identity_origin)""",
        entity_id=entity["entity_id"],
        canonical_name=entity["canonical_name"],
        aliases=entity.get("aliases", [entity["canonical_name"]]),
        identity_origin=entity.get("identity_origin", entity.get("origin", "observed_exact_name")),
    ).consume()


def _write_paper(tx: Any, paper: dict[str, Any]) -> None:
    tx.run(
        PAPER_QUERY,
        id=paper["id"],
        title=paper["title"],
        year=paper["year"],
        abstract=paper["abstract"],
        url=paper["url"],
        source=paper["source"],
    ).consume()
    for entity in paper["extracted"]["metadata_entities"]:
        _merge_entity(tx, entity)
        if entity["label"] == "Author":
            tx.run(
                "MATCH (n:Author {entity_id:$entity_id}), (p:Paper {id:$paper_id}) "
                "MERGE (n)-[:AUTHORED {origin:'paper_metadata', field:'authors'}]->(p)",
                entity_id=entity["entity_id"],
                paper_id=paper["id"],
            ).consume()
        else:
            tx.run(
                "MATCH (p:Paper {id:$paper_id}), (n:Concept {entity_id:$entity_id}) "
                "MERGE (p)-[:HAS_CONCEPT {origin:'paper_metadata', field:'keywords'}]->(n)",
                entity_id=entity["entity_id"],
                paper_id=paper["id"],
            ).consume()
    for mention in paper["extracted"]["mentions"]:
        _merge_entity(tx, mention)
        relation = mention["relation"]
        label = mention["label"]
        tx.run(
            f"MATCH (p:Paper {{id:$paper_id}}), (n:{label} {{entity_id:$entity_id}}) "
            f"CREATE (p)-[:{relation} {{origin:'title_abstract_extraction', "
            "field:$field, quote:$quote, start:$start, end:$end, "
            "surface_form:$surface_form}]->(n)",
            paper_id=paper["id"],
            entity_id=mention["entity_id"],
            field=mention["field"],
            quote=mention["quote"],
            start=mention["start"],
            end=mention["end"],
            surface_form=mention["surface_form"],
        ).consume()


def import_papers(driver: Any, database: str, papers: list[dict[str, Any]]) -> None:
    """Reconcile app-owned edges while preserving Paper identity and embeddings."""
    papers = validate_import_input(papers)
    with driver.session(database=database) as session:
        for paper in papers:
            session.execute_write(_write_paper, paper)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Applies the schema and imports validated papers.")
    parser.add_argument("--input", type=Path, default=Path("output/papers_enriched.json"))
    parser.add_argument("--schema", type=Path, default=Path("schema/schema.cypher"))
    parser.add_argument("--skip-schema", action="store_true", help="Do not reapply schema.cypher.")
    args = parser.parse_args()

    driver = None
    try:
        papers = validate_import_input(read_json(args.input))
        required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
        if any(not os.getenv(name) for name in required):
            raise ValueError("Neo4j configuration is incomplete.")
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
        driver.verify_connectivity()
        if not args.skip_schema:
            apply_schema(driver, database, args.schema)
        import_papers(driver, database, papers)
        with driver.session(database=database) as session:
            session.run("CALL db.awaitIndexes(300)").consume()
    except Exception as exc:
        raise SystemExit(
            f"Import failed ({type(exc).__name__}); check input, migration and Neo4j configuration."
        ) from None
    finally:
        if driver is not None:
            error_in_progress = sys.exc_info()[0] is not None
            try:
                driver.close()
            except Exception:
                if not error_in_progress:
                    raise SystemExit("Falha ao encerrar a conexão com o Neo4j.") from None
    print(f"Import completed: {len(papers)} papers processed.")


if __name__ == "__main__":
    main()
