from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from retrieval import planner
from retrieval.config import LLMSettings
from tests.test_cypher_guard import VOCABULARY

VALID = {
    "query": "MATCH (p:Paper)-[:HAS_CONCEPT]->(c:Concept) WHERE c.entity_id = $concept AND p.year >= $year RETURN DISTINCT p.id AS paper_id LIMIT 40",
    "params": {"concept": "concept:graphrag", "year": 2024},
}


class PlannerTests(unittest.TestCase):
    def call(self, session=None, **kwargs):
        return planner.plan_and_retrieve(
            session or Mock(),
            "Quais artigos de Graph RAG desde 2024?",
            seed_ids=["a"],
            vocabulary=VOCABULARY,
            corpus_ids=["a", "b", "c"],
            llm_settings=LLMSettings(api_key="test"),
            **kwargs,
        )

    @patch("retrieval.llm.accept_json")
    @patch("retrieval.llm.generate_json", return_value=VALID)
    def test_executes_compiled_actual_generated_query_and_retains_diagnostics(
        self, generate, accept
    ):
        session = Mock()
        session.run.return_value = [
            {"paper_id": "b"},
            {"paper_id": "a"},
            {"paper_id": "b"},
        ]
        result = self.call(session)
        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["paper_ids"], ["b", "a"])
        generated_query = result["queries"][0]
        submitted = session.run.call_args.args[0]
        self.assertEqual(submitted.timeout, 5)
        self.assertEqual(submitted.text, generated_query["cypher"])
        self.assertEqual(generated_query["original"], VALID["query"])
        self.assertEqual(generated_query["original_parameters"], VALID["params"])
        self.assertIn("p.year >= $year", submitted.text)
        self.assertIn("c.entity_id = $concept", submitted.text)
        self.assertIn("LIMIT 20", submitted.text)
        self.assertTrue(generated_query["executed"])
        accept.assert_called_once_with(VALID)
        request = generate.call_args.args[2]
        self.assertEqual(request["schema"]["relationships"][0]["type"], "AUTHORED")
        self.assertEqual(request["seed_ids"], ["a"])
        self.assertNotIn("abstract", str(request))
        self.assertEqual(request["vocabulary"][1]["aliases"], ["Graph RAG"])

    @patch("retrieval.llm.accept_json")
    @patch("retrieval.llm.generate_json")
    def test_invalid_generated_response_is_never_executed_or_cached(
        self, generate, accept
    ):
        adversarial = [
            {"query": "MATCH (p:Paper) DELETE p RETURN p.id LIMIT 40", "params": {}},
            {
                "query": "MATCH (p:Paper) WHERE p.id = $id RETURN p.id LIMIT 40",
                "params": {"id": "unrelated"},
            },
            {"query": "CALL dbms.components()", "params": {}},
            {
                "query": VALID["query"],
                "params": VALID["params"],
                "template": "anything",
            },
            {"template": "entity_search", "params": {}},
            {
                "query": VALID["query"],
                "params": {"concept": "concept:graphrag", "year": 2024, "x": "unused"},
            },
        ]
        for generated in adversarial:
            with self.subTest(generated=generated):
                generate.return_value = generated
                session = Mock()
                result = self.call(session)
                self.assertEqual(result["status"], "invalid")
                self.assertEqual(result["paper_ids"], [])
                session.run.assert_not_called()
        accept.assert_not_called()

    @patch("retrieval.llm.generate_json", return_value=VALID)
    @patch("retrieval.llm.accept_json")
    def test_empty_is_distinct_from_success(self, accept, generate):
        session = Mock()
        session.run.return_value = []
        result = self.call(session)
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["paper_ids"], [])
        self.assertTrue(result["queries"][0]["executed"])

    @patch("retrieval.llm.generate_json")
    def test_quota_timeout_and_other_failures_remain_visible(self, generate):
        for exception, status in (
            (RuntimeError("429 insufficient_quota"), "quota"),
            (TimeoutError("request timed out"), "timeout"),
            (RuntimeError("network failure with private endpoint"), "error"),
        ):
            with self.subTest(status=status):
                generate.side_effect = exception
                session = Mock()
                result = self.call(session)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["paper_ids"], [])
                self.assertNotIn("private endpoint", str(result))
                session.run.assert_not_called()

    @patch("retrieval.llm.generate_json", return_value=VALID)
    @patch("retrieval.llm.accept_json")
    def test_database_timeout_does_not_fall_through_as_success(self, accept, generate):
        session = Mock()
        session.run.side_effect = RuntimeError(
            "Neo.ClientError.Transaction.TransactionTimedOut"
        )
        result = self.call(session)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["paper_ids"], [])
        self.assertTrue(result["queries"][0]["attempted"])
        self.assertFalse(result["queries"][0]["executed"])

    @patch("retrieval.llm.generate_json", return_value=VALID)
    @patch("retrieval.llm.accept_json")
    def test_partial_results_discarded_on_iteration_timeout(self, accept, generate):
        def rows():
            yield {"paper_id": "a"}
            raise TimeoutError()

        session = Mock()
        session.run.return_value = rows()
        result = self.call(session)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["paper_ids"], [])

    @patch("retrieval.llm.generate_json", return_value=VALID)
    @patch("retrieval.llm.accept_json")
    def test_completion_error_after_the_final_row_discards_results(
        self, accept, generate
    ):
        def rows():
            yield {"paper_id": "a"}
            raise TimeoutError("server timeout after final record")

        session = Mock()
        session.run.return_value = rows()
        limited = {
            "query": "MATCH (p:Paper) WHERE p.id = $id RETURN p.id LIMIT 1",
            "params": {"id": "a"},
        }
        generate.return_value = limited
        result = self.call(session)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["paper_ids"], [])

    @patch("retrieval.llm.generate_json", return_value=VALID)
    @patch("retrieval.llm.accept_json")
    def test_corpus_scope_checked_on_results(self, accept, generate):
        session = Mock()
        session.run.return_value = [{"paper_id": "a"}, {"paper_id": "secret"}]
        result = self.call(session)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["paper_ids"], [])

    @patch("retrieval.llm.generate_json", return_value=VALID)
    @patch("retrieval.llm.accept_json")
    def test_stage_settings_used_when_not_passed(self, accept, generate):
        session = Mock()
        session.run.return_value = []
        with patch(
            "retrieval.llm.stage_settings", return_value=LLMSettings(api_key="test")
        ) as settings:
            result = planner.plan_and_retrieve(
                session,
                "GraphRAG?",
                seed_ids=["a"],
                vocabulary=VOCABULARY,
                corpus_ids=["a"],
            )
        settings.assert_called_once_with("cypher")
        self.assertEqual(result["status"], "empty")
