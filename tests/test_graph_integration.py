"""Opt-in tests: TEST_NEO4J_* must describe an isolated test service."""

import os
import json
import unittest
import uuid
from copy import deepcopy

from neo4j import GraphDatabase

from src.entity_resolution import EntityRegistry, resolve_metadata_entities
from src.import_graph import _write_paper, import_papers
from tests.test_graph_import import fixture
from retrieval.hybrid import PATHS_QUERY, connect_sources


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
        self.paper["abstract"] = (
            f"We present {self.token}method and evaluate it on {self.token}dataset."
        )
        self._set_extraction(
            [
                (self.token + "method", "Method", "PROPOSES"),
                (self.token + "dataset", "Dataset", "EVALUATED_ON"),
            ]
        )

    def _set_extraction(self, relation_mentions):
        registry = EntityRegistry.load()
        quote = self.paper["abstract"]
        mentions = []
        for surface, label, relation in relation_mentions:
            start = quote.index(surface)
            mentions.append(
                {
                    "surface_form": surface,
                    "label": label,
                    "relation": relation,
                    "field": "abstract",
                    "quote": quote,
                    "start": start,
                    "end": start + len(surface),
                    **registry.resolve(label, surface).as_dict(),
                    "origin": "title_abstract_extraction",
                }
            )
        fields = {"proposed_methods": [], "used_methods": [], "datasets": []}
        for mention in mentions:
            field = {
                "PROPOSES": "proposed_methods",
                "USES_METHOD": "used_methods",
                "EVALUATED_ON": "datasets",
            }[mention["relation"]]
            fields[field].append(mention["canonical_name"])
        self.paper["extracted"] = {
            **fields,
            "mentions": mentions,
            "provenance": {
                "origin": "title_abstract_extraction",
                "provider": "heuristic",
                "model": None,
                "schema_version": "1",
            },
            "metadata_entities": resolve_metadata_entities(self.paper, registry),
        }

    def tearDown(self):
        with self.driver.session(database=self.database) as s:
            s.run(
                "MATCH (n) WHERE n.id STARTS WITH $token OR n.name STARTS WITH $token DETACH DELETE n",
                token=self.token,
            ).consume()
        self.driver.close()

    def _path_evidence(self, *suffixes):
        return {
            "sources": [{"paper_id": self.token + suffix, "citation_id": f"S{i}"}
                        for i, suffix in enumerate(suffixes, start=1)],
            "subgraph": {"nodes": [], "edges": []},
        }

    def test_shortest_path_prefers_two_hops_and_preserves_author_direction(self):
        with self.driver.session(database=self.database) as session:
            session.run("""
                CREATE (a:Paper {id: $token + 'a'}), (b:Paper {id: $token + 'b'}),
                       (c:Paper {id: $token + 'c'}),
                       (author:Author {name: $token + 'author'}),
                       (x:Concept {name: $token + 'x'}),
                       (y:Dataset {name: $token + 'y'})
                CREATE (author)-[:AUTHORED]->(a), (author)-[:AUTHORED]->(b),
                       (a)-[:HAS_CONCEPT]->(x), (c)-[:HAS_CONCEPT]->(x),
                       (c)-[:EVALUATED_ON]->(y), (b)-[:EVALUATED_ON]->(y),
                       (a)-[:CITES]->(b)
            """, token=self.token).consume()
            evidence = self._path_evidence("a", "b")
            original_sources = deepcopy(evidence["sources"])
            paths = connect_sources(session, evidence)
            self.assertEqual(len(paths), 1)
            path = paths[0]
            self.assertEqual(path["length"], 2)
            self.assertEqual((path["source"], path["target"]), ("S1", "S2"))
            self.assertEqual([n["label"] for n in path["nodes"]], ["Paper", "Author", "Paper"])
            author_id = path["nodes"][1]["id"]
            self.assertTrue(all(e["source"] == author_id for e in path["edges"]))
            self.assertTrue(all(e["type"] == "AUTHORED" for e in path["edges"]))
            self.assertEqual(evidence["sources"], original_sources)
            self.assertNotIn("db_id", json.dumps(paths))
            # A tie may select either author, while the minimum length stays fixed.
            session.run("""
                MATCH (a:Paper {id:$a}), (b:Paper {id:$b})
                CREATE (x:Author {name:$name})-[:AUTHORED]->(a), (x)-[:AUTHORED]->(b)
            """, a=self.token + "a", b=self.token + "b", name=self.token + "tie").consume()
            tied = connect_sources(session, evidence)
            self.assertEqual(len(tied), 1)
            self.assertEqual(tied[0]["length"], 2)
            self.assertEqual(len(evidence["subgraph"]["nodes"]),
                             len({n["id"] for n in evidence["subgraph"]["nodes"]}))

    def test_four_hop_limit_connectors_and_disconnected_sources(self):
        with self.driver.session(database=self.database) as session:
            session.run("""
                CREATE (a:Paper {id:$token + 'a'}), (b:Paper {id:$token + 'b'}),
                       (c:Paper {id:$token + 'c'}), (d:Paper {id:$token + 'd'}),
                       (mid:Paper {id:$token + 'mid', title:'Artigo intermediário'}),
                       (x:Concept {name:$token + 'x'}),
                       (y:Concept {name:$token + 'y'}),
                       (z:Concept {name:$token + 'z'})
                CREATE (a)-[:HAS_CONCEPT]->(x), (mid)-[:HAS_CONCEPT]->(x),
                       (mid)-[:HAS_CONCEPT]->(y), (b)-[:HAS_CONCEPT]->(y),
                       (b)-[:HAS_CONCEPT]->(z), (d)-[:HAS_CONCEPT]->(z)
            """, token=self.token).consume()
            evidence = self._path_evidence("a", "b", "c")
            paths = connect_sources(session, evidence)
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0]["length"], 4)
            connectors = [n for n in paths[0]["nodes"] if n.get("connector")]
            self.assertEqual([n["properties"]["id"] for n in connectors], [self.token + "mid"])
            self.assertEqual(len(evidence["sources"]), 3)
            for _ in range(2):
                connect_sources(session, evidence)
            graph = evidence["subgraph"]
            self.assertEqual(len(graph["nodes"]), 5)
            self.assertEqual(len(graph["edges"]), 4)
            self.assertEqual(list(session.run(PATHS_QUERY, pairs=[{
                "source_id": self.token + "a", "target_id": self.token + "d",
            }])), [])

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
        changed["abstract"] = f"We use {self.token}method as a baseline."
        self.paper = changed
        self._set_extraction([(self.token + "method", "Method", "USES_METHOD")])
        changed = deepcopy(self.paper)
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
