"""Opt-in tests: TEST_NEO4J_* must describe an isolated test service."""

import os
import unittest
import uuid
from copy import deepcopy

from neo4j import GraphDatabase

from src.import_graph import _write_paper, import_papers
from tests.test_graph_import import fixture


@unittest.skipUnless(
    all(
        os.getenv(k)
        for k in ("TEST_NEO4J_URI", "TEST_NEO4J_USERNAME", "TEST_NEO4J_PASSWORD")
    ),
    "isolated TEST_NEO4J_* service not configured",
)
class GraphIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.driver = GraphDatabase.driver(
            os.environ["TEST_NEO4J_URI"],
            auth=(os.environ["TEST_NEO4J_USERNAME"], os.environ["TEST_NEO4J_PASSWORD"]),
        )
        self.database = os.getenv("TEST_NEO4J_DATABASE", "neo4j")
        self.token = "pipeline-test-" + uuid.uuid4().hex
        self.paper = fixture()
        self.paper.update(
            id=self.token,
            authors=[self.token + "author"],
            keywords=[self.token + "concept"],
        )
        self.paper["extracted"] = {
            "proposed_methods": [self.token + "method"],
            "used_methods": [self.token + "baseline"],
            "datasets": [self.token + "dataset"],
        }

    def tearDown(self):
        with self.driver.session(database=self.database) as s:
            s.run(
                "MATCH (n) WHERE n.id STARTS WITH $token OR n.name STARTS WITH $token DETACH DELETE n",
                token=self.token,
            ).consume()
        self.driver.close()

    def test_reconcile_rollback_and_preserve_unrelated(self):
        import_papers(self.driver, self.database, [self.paper])
        with self.driver.session(database=self.database) as s:
            s.run(
                "MATCH (p:Paper {id:$id}) CREATE (q:Paper {id:$other}) CREATE (p)-[:CITES]->(q) CREATE (p)-[:PROPOSES]->(q)",
                id=self.token,
                other=self.token + "other",
            ).consume()
        changed = deepcopy(self.paper)
        changed["title"] = "Changed"
        changed["authors"] = [self.token + "new-author"]
        changed["keywords"] = []
        changed["extracted"] = {
            "proposed_methods": [],
            "used_methods": [self.token + "method"],
            "datasets": [],
        }
        import_papers(self.driver, self.database, [changed])
        import_papers(self.driver, self.database, [changed])
        with self.driver.session(database=self.database) as s:
            self.assertEqual(
                s.run(
                    "MATCH (:Paper {id:$id})-[:PROPOSES]->(:Method) RETURN count(*) AS n",
                    id=self.token,
                ).single()["n"],
                0,
            )
            self.assertEqual(
                s.run(
                    "MATCH (:Paper {id:$id})-[:CITES]->(:Paper) RETURN count(*) AS n",
                    id=self.token,
                ).single()["n"],
                1,
            )
            self.assertEqual(
                s.run(
                    "MATCH (:Author {name:$name}) RETURN count(*) AS n",
                    name=self.token + "author",
                ).single()["n"],
                1,
            )
            self.assertEqual(
                s.run(
                    "MATCH (a:Author)-[:AUTHORED]->(:Paper {id:$id}) RETURN a.name AS name",
                    id=self.token,
                ).single()["name"],
                self.token + "new-author",
            )
            self.assertEqual(
                s.run(
                    "MATCH (:Paper {id:$id})-[:PROPOSES]->(:Paper) RETURN count(*) AS n",
                    id=self.token,
                ).single()["n"],
                1,
            )
            self.assertEqual(
                s.run(
                    "MATCH (:Paper {id:$id})-[:HAS_CONCEPT|EVALUATED_ON]->() RETURN count(*) AS n",
                    id=self.token,
                ).single()["n"],
                0,
            )
            self.assertEqual(
                s.run(
                    "MATCH (:Paper {id:$id})-[:USES_METHOD]->(m:Method) RETURN collect(m.name) AS names",
                    id=self.token,
                ).single()["names"],
                [self.token + "method"],
            )

            def fail(tx):
                _write_paper(tx, self.paper)
                raise RuntimeError("rollback")

            with self.assertRaises(RuntimeError):
                s.execute_write(fail)
            self.assertEqual(
                s.run(
                    "MATCH (p:Paper {id:$id}) RETURN p.title AS title", id=self.token
                ).single()["title"],
                "Changed",
            )
