from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from .common import read_json


def schema_statements(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"//[^\n]*", "", text)
    return [statement.strip() for statement in text.split(";") if statement.strip()]


def apply_schema(driver: Any, database: str, path: Path) -> None:
    with driver.session(database=database) as session:
        for statement in schema_statements(path):
            session.run(statement).consume()


PAPER_QUERY = """
MERGE (p:Paper {id: $id})
SET p.title = $title,
    p.year = $year,
    p.abstract = $abstract,
    p.url = $url,
    p.source = $source
WITH p
UNWIND $authors AS author_name
MERGE (a:Author {name: author_name})
MERGE (a)-[:AUTHORED]->(p)
WITH DISTINCT p
UNWIND $keywords AS concept_name
MERGE (c:Concept {name: concept_name})
MERGE (p)-[:HAS_CONCEPT]->(c)
"""

PROPOSES_QUERY = """
MATCH (p:Paper {id: $paper_id})
UNWIND $names AS entity_name
MERGE (m:Method {name: entity_name})
MERGE (p)-[:PROPOSES]->(m)
"""

USES_METHOD_QUERY = """
MATCH (p:Paper {id: $paper_id})
UNWIND $names AS entity_name
MERGE (m:Method {name: entity_name})
MERGE (p)-[:USES_METHOD]->(m)
"""

DATASET_QUERY = """
MATCH (p:Paper {id: $paper_id})
UNWIND $names AS entity_name
MERGE (d:Dataset {name: entity_name})
MERGE (p)-[:EVALUATED_ON]->(d)
"""


def import_papers(driver: Any, database: str, papers: list[dict[str, Any]]) -> None:
    with driver.session(database=database) as session:
        for paper in papers:
            session.execute_write(lambda tx, p=paper: tx.run(PAPER_QUERY, **p).consume())
            extracted = paper.get("extracted", {})
            relations = (
                (PROPOSES_QUERY, extracted.get("proposed_methods", [])),
                (USES_METHOD_QUERY, extracted.get("used_methods", [])),
                (DATASET_QUERY, extracted.get("datasets", [])),
            )
            for query, names in relations:
                if names:
                    session.execute_write(
                        lambda tx, q=query, n=names, pid=paper["id"]: tx.run(
                            q, paper_id=pid, names=n
                        ).consume()
                    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Aplica o schema e popula o Neo4j.")
    parser.add_argument(
        "--input", type=Path, default=Path("output/papers_enriched.json")
    )
    parser.add_argument(
        "--schema", type=Path, default=Path("schema/schema.cypher")
    )
    parser.add_argument(
        "--skip-schema", action="store_true", help="Não reaplica schema.cypher."
    )
    args = parser.parse_args()

    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise SystemExit(f"Variáveis ausentes no .env: {', '.join(missing)}")

    database = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        driver.verify_connectivity()
        if not args.skip_schema:
            apply_schema(driver, database, args.schema)
        papers = read_json(args.input)
        import_papers(driver, database, papers)
    finally:
        driver.close()
    print(f"Importação concluída: {len(papers)} papers processados.")


if __name__ == "__main__":
    main()
