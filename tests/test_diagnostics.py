import json
import unittest
from copy import deepcopy
from unittest.mock import Mock

from retrieval import hybrid
from tests.test_embeddings import FakeResult, valid_index
from tests.test_hybrid import Record, paper


class DiagnosticsTests(unittest.TestCase):
    def test_trace_matches_executed_queries_and_omits_vector_values(self):
        session = Mock()
        responses = {
            hybrid.INDEX_INFO_QUERY: [valid_index()],
            hybrid.PAPERS_QUERY: [paper("a"), paper("b")],
            hybrid.VECTOR_QUERY: [{"paper_id": "a", "score": 0.9},
                                  {"paper_id": "b", "score": 0.8}],
            hybrid.ENTITIES_QUERY: [],
            hybrid.EXPAND_QUERY: [],
            hybrid.FACTS_QUERY: [],
            hybrid.PATHS_QUERY: [],
        }
        session.run.side_effect = lambda query, **params: FakeResult(
            [Record(row) for row in responses[query]]
        )
        driver = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=False)
        encoder = Mock()
        encoder.encode.return_value = [1.0] + [0.0] * 383
        result = hybrid.retrieve(driver, "neo4j", encoder, "Sem entidades", 6)
        self.assertEqual([q["cypher"] for q in result["queries"]],
                         [call.args[0].strip() for call in session.run.call_args_list])
        self.assertNotIn(hybrid.DIRECT_QUERY.strip(), [q["cypher"] for q in result["queries"]])
        vector_trace = next(q for q in result["queries"] if q["stage"] == "Busca vetorial")
        self.assertEqual(vector_trace["parameters"], {
            "limit": 8, "embedding": {"dimensions": 384, "values_omitted": True},
        })
        vector_call = next(c for c in session.run.call_args_list if c.args[0] == hybrid.VECTOR_QUERY)
        self.assertEqual(vector_call.kwargs["embedding"], encoder.encode.return_value)
        self.assertEqual([s["paper_id"] for s in result["sources"]], ["a", "b"])
        self.assertEqual([s["score"] for s in result["sources"]], [1 / 61, 1 / 62])
        json.dumps(result)

    def test_empty_database_keeps_only_index_and_papers_trace(self):
        session = Mock()
        session.run.side_effect = lambda query, **params: FakeResult(
            [Record(valid_index())] if query == hybrid.INDEX_INFO_QUERY else []
        )
        driver = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=False)
        encoder = Mock()
        result = hybrid.retrieve(driver, "neo4j", encoder, "Pergunta", 6)
        self.assertEqual([q["cypher"] for q in result["queries"]], [
            hybrid.INDEX_INFO_QUERY.strip(), hybrid.PAPERS_QUERY.strip(),
        ])
        self.assertEqual(result["paths"], [])
        encoder.encode.assert_not_called()

    def test_paths_are_limited_to_fifteen_pairs_without_changing_sources(self):
        sources = [{"paper_id": str(i), "citation_id": f"S{i + 1}"} for i in range(20)]
        evidence = {"sources": sources, "subgraph": {"nodes": [], "edges": []}}
        original = deepcopy(evidence)
        session = Mock()
        session.run.return_value = []
        self.assertEqual(hybrid.connect_sources(session, evidence), [])
        pairs = session.run.call_args.kwargs["pairs"]
        self.assertEqual(len(pairs), 15)
        self.assertEqual({p[k] for p in pairs for k in ("source_id", "target_id")},
                         {str(i) for i in range(6)})
        self.assertEqual(evidence, original)

    def test_zero_or_one_source_skips_paths_query(self):
        for sources in ([], [{"paper_id": "a", "citation_id": "S1"}]):
            session = Mock()
            evidence = {"sources": sources, "subgraph": {"nodes": [], "edges": []}}
            self.assertEqual(hybrid.connect_sources(session, evidence), [])
            session.run.assert_not_called()

    def test_path_failure_is_not_reported_as_an_empty_success(self):
        session = Mock()
        session.run.side_effect = RuntimeError("database unavailable")
        evidence = {"sources": [{"paper_id": "a", "citation_id": "S1"},
                                {"paper_id": "b", "citation_id": "S2"}],
                    "subgraph": {"nodes": [], "edges": []}}
        with self.assertRaises(RuntimeError):
            hybrid.connect_sources(session, evidence)
