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
from .extract_entities import validate_entities
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
RELATIONS = (
    ("authors", "Author", "AUTHORED"),
    ("keywords", "Concept", "HAS_CONCEPT"),
    ("proposed_methods", "Method", "PROPOSES"),
    ("used_methods", "Method", "USES_METHOD"),
    ("datasets", "Dataset", "EVALUATED_ON"),
)
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


def validate_import_input(data: Any) -> list[dict[str, Any]]:
    """Valida o corpus inteiro antes de permitir qualquer acesso ao banco."""
    papers, report = validate_and_normalize(data)
    if not report["valid"]:
        raise ValueError("Entrada inválida: " + "; ".join(report["errors"]))
    for paper, raw in zip(papers, data):
        paper["extracted"] = validate_entities(raw.get("extracted"))
    return papers


def _write_paper(tx: Any, paper: dict[str, Any]) -> None:
    tx.run(PAPER_QUERY, **paper).consume()
    for field, label, relationship in RELATIONS:
        names = paper.get(field, paper["extracted"].get(field, []))
        edge = (
            f"(n)-[:{relationship}]->(p)"
            if relationship == "AUTHORED"
            else f"(p)-[:{relationship}]->(n)"
        )
        tx.run(
            f"MATCH (p:Paper {{id: $id}}) UNWIND $names AS name "
            f"MERGE (n:{label} {{name: name}}) MERGE {edge}",
            id=paper["id"],
            names=names,
        ).consume()


def import_papers(driver: Any, database: str, papers: list[dict[str, Any]]) -> None:
    """Reconcilia cada paper em uma transação, preservando dados não gerenciados."""
    papers = validate_import_input(papers)
    with driver.session(database=database) as session:
        for paper in papers:
            session.execute_write(_write_paper, paper)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Aplica o schema e popula o Neo4j.")
    parser.add_argument(
        "--input", type=Path, default=Path("output/papers_enriched.json")
    )
    parser.add_argument("--schema", type=Path, default=Path("schema/schema.cypher"))
    parser.add_argument(
        "--skip-schema", action="store_true", help="Não reaplica schema.cypher."
    )
    args = parser.parse_args()

    driver = None
    try:
        papers = validate_import_input(read_json(args.input))
        required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
        if any(not os.getenv(name) for name in required):
            raise ValueError("Configuração Neo4j incompleta.")
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
            f"Importação falhou ({type(exc).__name__}); confira entrada e configuração."
        ) from None
    finally:
        if driver is not None:
            error_in_progress = sys.exc_info()[0] is not None
            try:
                driver.close()
            except Exception:
                if not error_in_progress:
                    raise SystemExit(
                        "Falha ao encerrar a conexão com o Neo4j."
                    ) from None
    print(f"Importação concluída: {len(papers)} papers processados.")


if __name__ == "__main__":
    main()
