"""Confere o retrieval real após o bootstrap em um serviço TEST_NEO4J_* isolado.

Este comando não gera respostas nem usa credenciais LLM. Os testes unitários
continuam independentes do download do encoder e da disponibilidade do banco.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neo4j import GraphDatabase

from retrieval.embeddings import load_encoder
from retrieval.hybrid import retrieve
from retrieval.index_embeddings import index_papers


def main() -> None:
    required = ("TEST_NEO4J_URI", "TEST_NEO4J_USERNAME", "TEST_NEO4J_PASSWORD")
    if not all(os.getenv(name) for name in required):
        raise SystemExit("Configure TEST_NEO4J_* para um banco de teste isolado.")
    database = os.getenv("TEST_NEO4J_DATABASE", "neo4j")
    encoder = load_encoder()
    checks = []
    with GraphDatabase.driver(
        os.environ["TEST_NEO4J_URI"],
        auth=(os.environ["TEST_NEO4J_USERNAME"], os.environ["TEST_NEO4J_PASSWORD"]),
    ) as driver:
        driver.verify_connectivity()
        reindex = index_papers(driver, database, encoder=encoder)
        assert reindex["papers"] == 26, reindex
        assert reindex["embedded"] == 0 and reindex["skipped"] == 26, reindex
        for question in (
            "Quais artigos utilizam GraphRAG e quais datasets foram utilizados?",
            "Quais trabalhos usam HotpotQA?",
            "Quais artigos têm Cehao Yang como autor?",
        ):
            result = retrieve(driver, database, encoder, question, 6)
            sources = result["sources"]
            assert len(sources) == 6
            assert len({source["paper_id"] for source in sources}) == 6
            assert [source["citation_id"] for source in sources] == [
                f"S{i}" for i in range(1, 7)
            ]
            nodes = {node["id"]: node for node in result["subgraph"]["nodes"]}
            for edge in result["subgraph"]["edges"]:
                assert edge["source"] in nodes and edge["target"] in nodes
                if edge["type"] == "AUTHORED":
                    assert nodes[edge["source"]]["label"] == "Author"
                    assert nodes[edge["target"]]["label"] == "Paper"
            if "Cehao Yang" in question:
                with driver.session(database=database) as session:
                    expected = {
                        row["id"]
                        for row in session.run(
                            "MATCH (:Author {name: 'Cehao Yang'})-[:AUTHORED]->(p:Paper) RETURN p.id AS id"
                        )
                    }
                assert expected and expected.issubset(
                    {source["paper_id"] for source in sources}
                )
            checks.append(
                {"question": question, "sources": len(sources), "nodes": len(nodes)}
            )
    print(
        json.dumps(
            {"valid": True, "reindex": reindex, "checks": checks},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
