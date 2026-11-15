from __future__ import annotations

import unittest

from retrieval.cypher_guard import InvalidCypher, validate_and_compile

VOCABULARY = [
    {"label": "Author", "name": "Alice", "entity_id": "author:alice", "aliases": []},
    {
        "label": "Concept",
        "name": "GraphRAG",
        "entity_id": "concept:graphrag",
        "aliases": ["Graph RAG"],
    },
    {"label": "Method", "name": "KAG", "entity_id": "method:kag", "aliases": []},
    {
        "label": "Dataset",
        "name": "HotpotQA",
        "entity_id": "dataset:hotpot",
        "aliases": [],
    },
]


def compile_query(query, params=None, seeds=None):
    return validate_and_compile(
        query,
        params or {},
        seed_ids=seeds if seeds is not None else ["a"],
        vocabulary=VOCABULARY,
        corpus_ids=["a", "b", "c"],
    )


class CypherGuardTests(unittest.TestCase):
    def test_id_and_year_filters_preserved_in_direct_read(self):
        query = "MATCH (p:Paper {id:$paper}) WHERE p.year >= $year RETURN p.id AS paper_id LIMIT 12"
        plan = compile_query(query, {"paper": "a", "year": 2024})
        self.assertIn("p.id = $paper AND p.year >= $year", plan.executed)
        self.assertIn("p.id IN $guard_scope_p", plan.executed)
        self.assertIn("LIMIT 12", plan.executed)
        self.assertNotIn("CALL", plan.executed)
        self.assertEqual(plan.original_parameters, {"paper": "a", "year": 2024})

    def test_three_way_intersection_retains_all_predicates(self):
        plan = compile_query(
            "MATCH (p:Paper)-[:HAS_CONCEPT]->(c:Concept), "
            "(p)-[:USES_METHOD]->(m:Method), (a:Author)-[:AUTHORED]->(p) "
            "WHERE c.entity_id = $concept AND m.name = $method AND a.entity_id = $author "
            "AND p.year >= $year RETURN DISTINCT p.id AS paper_id LIMIT 40",
            {
                "concept": "concept:graphrag",
                "method": "KAG",
                "author": "author:alice",
                "year": 2024,
            },
        )
        self.assertEqual(plan.executed.count("LIMIT 20"), 3)
        self.assertEqual(plan.executed.count("LIMIT 40"), 4)
        for predicate in (
            "c.entity_id = $concept",
            "m.name = $method",
            "a.entity_id = $author",
            "p.year >= $year",
        ):
            self.assertIn(predicate, plan.executed)
        self.assertEqual(plan.executed.count("RETURN DISTINCT p.id AS paper_id"), 1)
        self.assertEqual(plan.root_alias, "c")

    def test_four_hops_bound_every_frontier_before_next_expansion(self):
        plan = compile_query(
            "MATCH (s:Paper)-[:HAS_CONCEPT]->(c:Concept)<-[:HAS_CONCEPT]-(m:Paper)"
            "-[:USES_METHOD]->(e:Method)<-[:USES_METHOD]-(p:Paper) "
            "WHERE s.id IN $seeds RETURN DISTINCT p.id AS paper_id LIMIT 40",
            {"seeds": ["a"]},
        )
        self.assertEqual(plan.executed.count("CALL ("), 4)
        self.assertEqual(plan.executed.count("LIMIT 20"), 4)
        self.assertEqual(plan.executed.count("LIMIT 40"), 5)
        self.assertLess(plan.executed.index("LIMIT 8"), plan.executed.index("CALL ("))
        self.assertIn("LIMIT 20\n}\nWITH DISTINCT", plan.executed)
        self.assertIn("c:Concept)<-[:HAS_CONCEPT]-(m:Paper)", plan.canonical)

    def test_seed_scope_clamped_before_any_fanout(self):
        plan = validate_and_compile(
            "MATCH (p:Paper)-[:HAS_CONCEPT]->(c:Concept)<-[:HAS_CONCEPT]-(q:Paper) "
            "RETURN DISTINCT q.id AS paper_id LIMIT 40",
            {},
            seed_ids=[str(index) for index in range(20)],
            vocabulary=VOCABULARY,
            corpus_ids=[str(index) for index in range(30)],
        )
        self.assertEqual(len(plan.parameters["guard_scope_q"]), 8)
        self.assertEqual(len(plan.parameters["guard_scope_p"]), 30)

    def test_relationship_uniqueness_preserved_within_each_original_match(self):
        query = "MATCH (s:Paper)-[:HAS_CONCEPT]->(c:Concept)<-[:HAS_CONCEPT]-(p:Paper) WHERE s.id = $seed RETURN p.id LIMIT 40"
        plan = compile_query(query, {"seed": "a"})
        self.assertIn("guard_rel_1 <> guard_rel_0", plan.executed)
        self.assertIn("RETURN DISTINCT p, guard_rel_1", plan.executed)
        separate = compile_query(
            "MATCH (s:Paper)-[:HAS_CONCEPT]->(c:Concept) MATCH (c)<-[:HAS_CONCEPT]-(p:Paper) WHERE s.id = $seed RETURN p.id LIMIT 40",
            {"seed": "a"},
        )
        self.assertNotIn("guard_rel_1 <> guard_rel_0", separate.executed)
        self.assertEqual(separate.canonical.count("MATCH "), 2)

    def test_four_entity_intersection_rejected(self):
        with self.assertRaises(InvalidCypher):
            compile_query(
                "MATCH (a:Author)-[:AUTHORED]->(p:Paper)-[:HAS_CONCEPT]->(c:Concept), "
                "(m:Method)<-[:USES_METHOD]-(p)-[:EVALUATED_ON]->(d:Dataset) "
                "WHERE a.entity_id = $a AND c.entity_id = $c AND m.entity_id = $m AND d.entity_id = $d "
                "RETURN p.id LIMIT 40",
                {
                    "a": "author:alice",
                    "c": "concept:graphrag",
                    "m": "method:kag",
                    "d": "dataset:hotpot",
                },
            )

    def test_all_relationships_have_the_correct_direction(self):
        for label, relationship, direction, entity_id in (
            ("Author", "AUTHORED", "incoming", "author:alice"),
            ("Concept", "HAS_CONCEPT", "outgoing", "concept:graphrag"),
            ("Method", "PROPOSES", "outgoing", "method:kag"),
            ("Method", "USES_METHOD", "outgoing", "method:kag"),
            ("Dataset", "EVALUATED_ON", "outgoing", "dataset:hotpot"),
        ):
            with self.subTest(relationship=relationship):
                edge = (
                    f"<-[:{relationship}]-"
                    if direction == "incoming"
                    else f"-[:{relationship}]->"
                )
                plan = compile_query(
                    f"MATCH (p:Paper){edge}(e:{label}) WHERE e.entity_id = $entity RETURN p.id LIMIT 40",
                    {"entity": entity_id},
                )
                self.assertIn(relationship, plan.executed)

    def test_unsafe_or_out_of_subset_query_never_compiles(self):
        queries = [
            "MATCH (p:Paper) DELETE p RETURN p.id LIMIT 40",
            "MATCH (p:Paper) DETACH DELETE p RETURN p.id LIMIT 40",
            "CALL db.labels() YIELD label RETURN label LIMIT 40",
            "MATCH (p:Paper) RETURN p.id LIMIT 40 UNION MATCH (n) RETURN n.id LIMIT 40",
            "MATCH (p:Paper) WHERE p.id = $id RETURN p.id LIMIT 40 // comment",
            "/*comment*/ MATCH (p:Paper) WHERE p.id = $id RETURN p.id LIMIT 40",
            "MATCH (p:`Paper`) WHERE p.id = $id RETURN p.id LIMIT 40",
            "MATCH (p:Paper) WHERE p.id = $id RETURN p.id LIMIT 40;",
            "MATCH (p:Secret) WHERE p.id = $id RETURN p.id LIMIT 40",
            "MATCH (p:Paper) WHERE p.embedding = $id RETURN p.id LIMIT 40",
            "MATCH (p:Paper) WHERE p.id = $id RETURN p LIMIT 40",
            "MATCH (p:Paper) WHERE p.id = $id OR true RETURN p.id LIMIT 40",
            "MATCH (p:Paper) WHERE p.id = 'a' RETURN p.id LIMIT 40",
            "MATCH (p:Paper) WHERE p.id = $id RETURN p.id LIMIT " + "9" * 5000,
            "MATCH (p:Paper) WHERE p.id = $id RETURN p.id LIMIT 41",
            "MATCH (p:Paper) WHERE p.id = $id RETURN p.id LIMIT 0",
            "MATCH (p:Paper)-[:HAS_CONCEPT*1..4]->(c:Concept) WHERE p.id = $id RETURN p.id LIMIT 40",
            "MATCH (p:Paper)-[:UNRELATED]->(c:Concept) WHERE p.id = $id RETURN p.id LIMIT 40",
            "MATCH (p:Paper)-[:AUTHORED]->(a:Author) WHERE p.id = $id RETURN p.id LIMIT 40",
            "MATCH (p:Paper), (q:Paper) WHERE p.id = $id RETURN p.id LIMIT 40",
            "MATCH (p:Paper)-[:HAS_CONCEPT]->(c:Dataset) WHERE p.id = $id RETURN p.id LIMIT 40",
            "MATCH (guard_scope:Paper) WHERE guard_scope.id = $id RETURN guard_scope.id LIMIT 40",
        ]
        for query in queries:
            with self.subTest(query=query), self.assertRaises(InvalidCypher):
                compile_query(query, {"id": "a"})

    def test_parameters_are_scoped_typed_and_all_used(self):
        cases = [
            ("p.id = $id", {"id": "missing"}),
            ("p.id = $id", {"id": "a", "extra": "a"}),
            ("p.id = $id", {"id": ["a"]}),
            ("p.id IN $id", {"id": "a"}),
            ("p.id IN $id", {"id": []}),
            ("p.id IN $id", {"id": ["a"] * 9}),
            ("p.year >= $year", {"year": True}),
            ("p.year >= $year", {"year": "2024"}),
            ("p.year IN $year", {"year": [2024]}),
            ("p.id = $id", {}),
            ("z.id = $id", {"id": "a"}),
            ("p.id = $guard_input", {"guard_input": "a"}),
        ]
        for predicate, params in cases:
            with (
                self.subTest(predicate=predicate, params=params),
                self.assertRaises(InvalidCypher),
            ):
                compile_query(
                    f"MATCH (p:Paper) WHERE {predicate} RETURN p.id LIMIT 40", params
                )

    def test_alias_spelling_cannot_be_used_as_canonical_name(self):
        with self.assertRaises(InvalidCypher):
            compile_query(
                "MATCH (p:Paper)-[:HAS_CONCEPT]->(c:Concept) WHERE c.name = $name RETURN p.id LIMIT 40",
                {"name": "Graph RAG"},
            )

    def test_invalid_shapes_do_not_raise_uncontrolled_exceptions(self):
        for query, params in (
            (None, {}),
            ("", {}),
            ("x" * 13000, {}),
            ("MATCH (p:Paper) RETURN p.id LIMIT 40", []),
        ):
            with self.subTest(query=str(query)[:30]), self.assertRaises(InvalidCypher):
                compile_query(query, params)
