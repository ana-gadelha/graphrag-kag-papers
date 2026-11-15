import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from retrieval import hybrid
from tests.test_hybrid import Record, fact, paper


class HybridPlannerTests(unittest.TestCase):
    def run_retrieval(self, mode="hybrid", planner_status="executed", planner_ids=None):
        session = Mock()
        responses = {
            hybrid.PAPERS_QUERY: [paper("a"), paper("b")],
            hybrid.VECTOR_QUERY: [{"paper_id": "a", "score": 0.9}],
            hybrid.ENTITIES_QUERY: [
                {
                    "label": "Method",
                    "name": "M",
                    "entity_id": "m:1",
                    "aliases": ["method alias"],
                }
            ],
            hybrid.EXPAND_QUERY: [],
            hybrid.FACTS_QUERY: [
                dict(fact("b", "PROPOSES", "Method", "M"), entity_id="m:1")
            ],
            hybrid.PATHS_QUERY: [],
        }
        session.run.side_effect = lambda q, **kw: [Record(r) for r in responses[q]]
        driver = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=False)
        encoder = Mock()
        encoder.encode.return_value = [1.0] + [0.0] * 383
        plan = {
            "status": planner_status,
            "paper_ids": planner_ids if planner_ids is not None else ["b"],
            "queries": [{"stage": "LLM", "cypher": "MATCH ...", "executed": True}],
            "warnings": [] if planner_status == "executed" else ["Operação degradada."],
        }
        with (
            patch("retrieval.hybrid.check_index"),
            patch("retrieval.planner.plan_and_retrieve", return_value=plan) as planner,
        ):
            result = hybrid.retrieve(
                driver,
                "neo4j",
                encoder,
                "papers after 2025?",
                1,
                use_planner=True,
                mode=mode,
            )
        return result, session, planner

    def test_generated_candidates_contribute_and_keep_origin_trace(self):
        result, _, planner = self.run_retrieval()
        self.assertEqual(result["sources"][0]["paper_id"], "b")
        self.assertEqual(result["sources"][0]["graph_origins"], ["llm_cypher"])
        self.assertEqual(result["planner"]["status"], "executed")
        self.assertTrue(any(q["stage"] == "LLM" for q in result["queries"]))
        self.assertEqual(result["paths"][0]["kind"], "fact")
        self.assertEqual(result["paths"][0]["path_id"], "P1")
        self.assertIn("method:m%3A1", {n["id"] for n in result["subgraph"]["nodes"]})
        planner.assert_called_once()

    def test_failure_keeps_visible_degraded_status_and_fixed_results(self):
        result, _, _ = self.run_retrieval(planner_status="quota", planner_ids=[])
        self.assertEqual(result["sources"][0]["paper_id"], "a")
        self.assertEqual(result["planner"]["status"], "quota")
        self.assertIn("Operação degradada.", result["warnings"])

    def test_vector_baseline_never_plans_or_expands_graph(self):
        result, session, planner = self.run_retrieval(mode="vector")
        planner.assert_not_called()
        self.assertEqual(result["sources"][0]["paper_id"], "a")
        self.assertEqual(result["paths"], [])
        self.assertEqual(result["subgraph"]["edges"], [])
        calls = [c.args[0] for c in session.run.call_args_list]
        self.assertNotIn(hybrid.ENTITIES_QUERY, calls)
        self.assertNotIn(hybrid.EXPAND_QUERY, calls)
        self.assertNotIn(hybrid.FACTS_QUERY, calls)

    def test_alias_matching_and_fact_ids_are_idempotent(self):
        vocabulary = [
            {
                "label": "Method",
                "name": "M",
                "aliases": ["method alias"],
                "entity_id": "m:1",
            }
        ]
        self.assertEqual(
            hybrid.match_entities("Uses method alias?", vocabulary), vocabulary
        )
        result, _, _ = self.run_retrieval()
        before = deepcopy(result)
        hybrid.assign_evidence_ids(result)
        self.assertEqual(before, result)


if __name__ == "__main__":
    unittest.main()
