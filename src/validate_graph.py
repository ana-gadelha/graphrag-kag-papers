from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


QUERIES = {
    "nodes_by_label": """
        MATCH (n)
        UNWIND labels(n) AS label
        RETURN label, count(*) AS quantity ORDER BY label
    """,
    "relationships_by_type": """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(*) AS quantity ORDER BY type
    """,
    "papers_without_authors": """
        MATCH (p:Paper)
        WHERE NOT EXISTS { MATCH (:Author)-[:AUTHORED]->(p) }
        RETURN p.id AS id, p.title AS title ORDER BY p.id
    """,
    "papers_without_concepts": """
        MATCH (p:Paper)
        WHERE NOT EXISTS { MATCH (p)-[:HAS_CONCEPT]->(:Concept) }
        RETURN p.id AS id, p.title AS title ORDER BY p.id
    """,
    "papers_without_abstract": """
        MATCH (p:Paper)
        WHERE p.abstract IS NULL OR trim(p.abstract) = ''
        RETURN p.id AS id, p.title AS title ORDER BY p.id
    """,
}


def main() -> None:
    load_dotenv()
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise SystemExit(f"Variáveis ausentes no .env: {', '.join(missing)}")

    database = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    report = {}
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            for name, query in QUERIES.items():
                report[name] = [record.data() for record in session.run(query)]
    finally:
        driver.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
