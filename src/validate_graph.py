"""Confere o corpus importado e o schema contra o snapshot esperado."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from .common import read_json, write_json
from .import_graph import (
    OWNED_EDGE_PREDICATE,
    OWNED_TYPES,
    RELATIONS,
    validate_import_input,
)


def validate_graph(
    driver: Any, database: str, papers: list[dict], *, check_embeddings: bool = False
) -> dict:
    papers = validate_import_input(papers)
    errors = []
    with driver.session(database=database) as session:
        constraints = {r["name"]: r.data() for r in session.run("SHOW CONSTRAINTS")}
        indexes = {r["name"]: r.data() for r in session.run("SHOW INDEXES")}
        for label, key in [
            ("Paper", "id"),
            ("Author", "name"),
            ("Concept", "name"),
            ("Method", "name"),
            ("Dataset", "name"),
        ]:
            name = f"{label.lower()}_{key}_unique"
            c = constraints.get(name, {})
            if (
                c.get("type") not in {"UNIQUENESS", "NODE_PROPERTY_UNIQUENESS"}
                or c.get("entityType") != "NODE"
                or c.get("labelsOrTypes") != [label]
                or c.get("properties") != [key]
            ):
                errors.append(f"Constraint incompatível: {name}")
            backing = indexes.get(c.get("ownedIndex"), {})
            if backing.get("state") != "ONLINE":
                errors.append(f"Índice da constraint indisponível: {name}")
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
                errors.append(f"Índice incompatível: {name}")
        from retrieval.config import RetrievalError
        from retrieval.embeddings import check_index

        try:
            check_index(session)
        except RetrievalError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(
                f"Conferência do índice vetorial falhou ({type(exc).__name__})."
            )
        for paper in papers:
            rows = list(
                session.run(
                    "MATCH (p:Paper {id:$id}) RETURN properties(p) AS properties",
                    id=paper["id"],
                )
            )
            if len(rows) != 1:
                errors.append(
                    f"{paper['id']}: esperado exatamente um Paper; encontrado {len(rows)}"
                )
                continue
            props = rows[0]["properties"]
            for key in ("id", "title", "abstract", "year", "source", "url"):
                if props.get(key) != paper[key]:
                    errors.append(f"{paper['id']}: propriedade divergente: {key}")
            actual = list(
                session.run(
                    """MATCH (p:Paper {id:$id})-[r]- (n)
                WHERE """
                    + OWNED_EDGE_PREDICATE
                    + """
                RETURN type(r) AS type, labels(n) AS labels, n.name AS name,
                       startNode(r) = p AS outgoing""",
                    id=paper["id"],
                    types=list(OWNED_TYPES),
                )
            )
            expected = []
            for field, label, rel in RELATIONS:
                for name in paper.get(field, paper["extracted"].get(field, [])):
                    expected.append((rel, label, name, rel != "AUTHORED"))
            remaining = list(expected)
            for edge in actual:
                matches = [
                    item
                    for item in remaining
                    if item[0] == edge["type"]
                    and item[1] in edge["labels"]
                    and item[2] == edge["name"]
                    and item[3] == edge["outgoing"]
                ]
                if matches:
                    remaining.remove(matches[0])
                else:
                    errors.append(f"{paper['id']}: relação inesperada: {dict(edge)}")
            if remaining:
                errors.append(f"{paper['id']}: relações ausentes: {remaining}")
            if check_embeddings:
                from retrieval.embeddings import (
                    EMBEDDING_MODEL,
                    content_hash,
                    validate_vector,
                )

                try:
                    validate_vector(props.get("embedding"))
                    if props.get("embedding_model") != EMBEDDING_MODEL or props.get(
                        "embedding_hash"
                    ) != content_hash(paper):
                        raise ValueError("modelo/hash divergente")
                except (RetrievalError, ValueError) as exc:
                    errors.append(f"{paper['id']}: embedding inválido: {exc}")
    return {"valid": not errors, "papers_checked": len(papers), "errors": errors}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("output/papers_enriched.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("output/graph_validation_report.json")
    )
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
            "errors": [f"Validação não concluída ({type(exc).__name__})."],
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
