"""Check graph contents, schema constraints, provenance and vector index."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from .common import read_json, write_json
from .entity_resolution import entity_id_for
from .import_graph import OWNED_EDGE_PREDICATE, OWNED_TYPES, validate_import_input


ENTITY_CONSTRAINTS = {
    "Paper": ("paper_id_unique", "id"),
    "Author": ("author_entity_id_unique", "entity_id"),
    "Concept": ("concept_entity_id_unique", "entity_id"),
    "Method": ("method_entity_id_unique", "entity_id"),
    "Dataset": ("dataset_entity_id_unique", "entity_id"),
}


def _expected_edges(paper: dict[str, Any]) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for entity in paper["extracted"]["metadata_entities"]:
        expected.append(
            {
                "type": entity["relation"],
                "label": entity["label"],
                "entity_id": entity["entity_id"],
                "canonical_name": entity["canonical_name"],
                "outgoing": entity["label"] != "Author",
                "properties": {"origin": "paper_metadata", "field": entity["field"]},
            }
        )
    for mention in paper["extracted"]["mentions"]:
        expected.append(
            {
                "type": mention["relation"],
                "label": mention["label"],
                "entity_id": mention["entity_id"],
                "canonical_name": mention["canonical_name"],
                "outgoing": True,
                "properties": {
                    "origin": "title_abstract_extraction",
                    "field": mention["field"],
                    "quote": mention["quote"],
                    "start": mention["start"],
                    "end": mention["end"],
                    "surface_form": mention["surface_form"],
                },
            }
        )
    return expected


def validate_graph(
    driver: Any, database: str, papers: list[dict], *, check_embeddings: bool = False
) -> dict:
    papers = validate_import_input(papers)
    errors = []
    with driver.session(database=database) as session:
        constraints = {row["name"]: row.data() for row in session.run("SHOW CONSTRAINTS")}
        indexes = {row["name"]: row.data() for row in session.run("SHOW INDEXES")}
        for label, (name, key) in ENTITY_CONSTRAINTS.items():
            constraint = constraints.get(name, {})
            if (
                constraint.get("type") not in {"UNIQUENESS", "NODE_PROPERTY_UNIQUENESS"}
                or constraint.get("entityType") != "NODE"
                or constraint.get("labelsOrTypes") != [label]
                or constraint.get("properties") != [key]
            ):
                errors.append(f"Incompatible constraint: {name}")
            backing = indexes.get(constraint.get("ownedIndex"), {})
            if backing.get("state") != "ONLINE":
                errors.append(f"Constraint index unavailable: {name}")

        for name, kind, labels, properties in [
            ("paper_year_range", "RANGE", ["Paper"], ["year"]),
            ("paper_fulltext", "FULLTEXT", ["Paper"], ["title", "abstract"]),
            (
                "entity_name_fulltext",
                "FULLTEXT",
                ["Concept", "Method", "Dataset"],
                ["name"],
            ),
        ]:
            index = indexes.get(name, {})
            if (
                index.get("state") != "ONLINE"
                or index.get("type") != kind
                or set(index.get("labelsOrTypes", [])) != set(labels)
                or set(index.get("properties", [])) != set(properties)
                or index.get("entityType") != "NODE"
            ):
                errors.append(f"Incompatible index: {name}")

        from retrieval.config import RetrievalError
        from retrieval.embeddings import check_index

        try:
            check_index(session)
        except RetrievalError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"Vector index check failed ({type(exc).__name__}).")

        for paper in papers:
            rows = list(
                session.run(
                    "MATCH (p:Paper {id:$id}) RETURN properties(p) AS properties",
                    id=paper["id"],
                )
            )
            if len(rows) != 1:
                errors.append(
                    f"{paper['id']}: expected one Paper, found {len(rows)}"
                )
                continue
            props = rows[0]["properties"]
            for key in ("id", "title", "abstract", "year", "source", "url"):
                if props.get(key) != paper[key]:
                    errors.append(f"{paper['id']}: property differs: {key}")

            actual = list(
                session.run(
                    """MATCH (p:Paper {id:$id})-[r]-(n)
                    WHERE """
                    + OWNED_EDGE_PREDICATE
                    + """RETURN type(r) AS type, labels(n) AS labels,
                       n.entity_id AS entity_id, n.name AS name,
                       startNode(r) = p AS outgoing,
                       properties(r) AS relationship_properties""",
                    id=paper["id"],
                )
            )
            expected = _expected_edges(paper)
            remaining = list(expected)
            for edge in actual:
                matches = [
                    item
                    for item in remaining
                    if item["type"] == edge["type"]
                    and item["label"] in edge["labels"]
                    and item["entity_id"] == edge.get("entity_id")
                    and item["canonical_name"] == edge.get("name")
                    and item["outgoing"] == edge.get("outgoing")
                    and all(
                        (edge.get("relationship_properties") or {}).get(key) == value
                        for key, value in item["properties"].items()
                    )
                ]
                if matches:
                    remaining.remove(matches[0])
                else:
                    errors.append(f"{paper['id']}: unexpected relationship: {dict(edge)}")
            if remaining:
                errors.append(f"{paper['id']}: missing relationships: {remaining}")

            if check_embeddings:
                from retrieval.embeddings import EMBEDDING_MODEL, content_hash, validate_vector

                try:
                    validate_vector(props.get("embedding"))
                    if props.get("embedding_model") != EMBEDDING_MODEL or props.get(
                        "embedding_hash"
                    ) != content_hash(paper):
                        raise ValueError("model/hash mismatch")
                except (RetrievalError, ValueError) as exc:
                    errors.append(f"{paper['id']}: invalid embedding: {exc}")
    return {"valid": not errors, "papers_checked": len(papers), "errors": errors}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/papers_enriched.json"))
    parser.add_argument("--report", type=Path, default=Path("output/graph_validation_report.json"))
    parser.add_argument("--check-embeddings", action="store_true")
    args = parser.parse_args()
    driver = None
    report = {"valid": False, "errors": []}
    try:
        papers = validate_import_input(read_json(args.input))
        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
        report = validate_graph(
            driver,
            os.getenv("NEO4J_DATABASE", "neo4j"),
            papers,
            check_embeddings=args.check_embeddings,
        )
    except Exception as exc:
        report = {
            "valid": False,
            "errors": [f"Graph validation did not complete ({type(exc).__name__})."],
        }
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                report["valid"] = False
                report["errors"].append("Falha ao encerrar a conexão com o Neo4j.")
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
