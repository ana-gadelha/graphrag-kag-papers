"""Opt-in Neo4j checks for compiled generated Cypher; no model inference."""

import os
import unittest
import uuid

from neo4j import GraphDatabase, Query

from retrieval.cypher_guard import InvalidCypher, validate_and_compile


@unittest.skipUnless(
    all(
        os.getenv(k)
        for k in ("TEST_NEO4J_URI", "TEST_NEO4J_USERNAME", "TEST_NEO4J_PASSWORD")
    ),
    "isolated TEST_NEO4J_* service not configured",
)
class PlannerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.driver = GraphDatabase.driver(
            os.environ["TEST_NEO4J_URI"],
            auth=(os.environ["TEST_NEO4J_USERNAME"], os.environ["TEST_NEO4J_PASSWORD"]),
        )
        self.session = self.driver.session(
            database=os.getenv("TEST_NEO4J_DATABASE", "neo4j")
        )
        self.prefix = "planner-test-" + uuid.uuid4().hex
        self.ids = [self.prefix + x for x in ("a", "b", "c")]
        self.vocab = [
            {
                "label": label,
                "name": self.prefix + suffix,
                "entity_id": self.prefix + suffix,
            }
            for label, suffix in [
                ("Concept", "concept"),
                ("Dataset", "dataset"),
                ("Method", "method"),
                ("Author", "author"),
            ]
        ]
        self.session.run(
            """
            CREATE (a:Paper {id:$a, year:2024}), (b:Paper {id:$b, year:2025}), (c:Paper {id:$c, year:2026}),
                   (x:Concept {name:$x, entity_id:$x}), (y:Dataset {name:$y, entity_id:$y}),
                   (m:Method {name:$m, entity_id:$m}), (u:Author {name:$u, entity_id:$u})
            CREATE (a)-[:HAS_CONCEPT]->(x), (b)-[:HAS_CONCEPT]->(x),
                   (b)-[:EVALUATED_ON]->(y), (c)-[:EVALUATED_ON]->(y),
                   (a)-[:PROPOSES]->(m), (u)-[:AUTHORED]->(a)
        """,
            a=self.ids[0],
            b=self.ids[1],
            c=self.ids[2],
            x=self.vocab[0]["entity_id"],
            y=self.vocab[1]["entity_id"],
            m=self.vocab[2]["entity_id"],
            u=self.vocab[3]["entity_id"],
        ).consume()

    def tearDown(self):
        self.session.run(
            "MATCH (n) WHERE n.id STARTS WITH $prefix OR n.entity_id STARTS WITH $prefix DETACH DELETE n",
            prefix=self.prefix,
        ).consume()
        self.session.close()
        self.driver.close()

    def run_query(self, query, params):
        compiled = validate_and_compile(
            query,
            params,
            seed_ids=self.ids[:1],
            corpus_ids=self.ids,
            vocabulary=self.vocab,
        )
        return [
            r["paper_id"]
            for r in self.session.run(
                Query(compiled.executed, timeout=5), **compiled.parameters
            )
        ]

    def test_direct_year_and_entity_intersection(self):
        direct = "MATCH (p:Paper) WHERE p.year >= $year RETURN DISTINCT p.id AS paper_id LIMIT 40"
        self.assertEqual(self.run_query(direct, {"year": 2025}), self.ids[1:])
        intersection = "MATCH (u:Author)-[:AUTHORED]->(p:Paper)-[:PROPOSES]->(m:Method) WHERE u.entity_id = $author AND m.entity_id = $method RETURN DISTINCT p.id AS paper_id LIMIT 40"
        self.assertEqual(
            self.run_query(
                intersection,
                {
                    "author": self.vocab[3]["entity_id"],
                    "method": self.vocab[2]["entity_id"],
                },
            ),
            self.ids[:1],
        )

    def test_two_and_four_hops_preserve_relationship_uniqueness(self):
        query = "MATCH (a:Paper)-[:HAS_CONCEPT]->(x:Concept)<-[:HAS_CONCEPT]-(p:Paper) WHERE a.id = $id RETURN DISTINCT p.id AS paper_id LIMIT 40"
        self.assertEqual(self.run_query(query, {"id": self.ids[0]}), self.ids[1:2])
        query = "MATCH (a:Paper)-[:HAS_CONCEPT]->(x:Concept)<-[:HAS_CONCEPT]-(b:Paper)-[:EVALUATED_ON]->(y:Dataset)<-[:EVALUATED_ON]-(p:Paper) WHERE a.id = $id RETURN DISTINCT p.id AS paper_id LIMIT 40"
        self.assertEqual(self.run_query(query, {"id": self.ids[0]}), self.ids[2:])

    def test_outside_corpus_and_write_queries_reject_before_execution(self):
        for query, params in [
            ("MATCH (p:Paper) DELETE p RETURN DISTINCT p.id AS paper_id LIMIT 1", {}),
            (
                "MATCH (p:Paper) WHERE p.id = $id RETURN DISTINCT p.id AS paper_id LIMIT 1",
                {"id": "unknown"},
            ),
        ]:
            with self.assertRaises(InvalidCypher):
                self.run_query(query, params)


if __name__ == "__main__":
    unittest.main()
