from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

from retrieval import hybrid
from retrieval.config import RetrievalError
from retrieval.embeddings import EMBEDDING_MODEL, content_hash


class Record(dict):
    def data(self):
        return dict(self)


def paper(paper_id):
    result = {
        "id": paper_id, "title": f"Paper {paper_id}", "abstract": "Abstract disponível.",
        "keywords": ["GraphRAG"], "year": 2025, "source": "arXiv",
        "url": f"https://example.org/{paper_id}",
        "embedding": [1.0] + [0.0] * 383, "embedding_model": EMBEDDING_MODEL,
    }
    result["embedding_hash"] = content_hash(result)
    return result


def fact(paper_id, relation, label, name, paper_is_source=True):
    return {"paper_id": paper_id, "type": relation, "entity_label": label,
            "entity_name": name, "paper_is_source": paper_is_source}


class HybridTests(unittest.TestCase):
    def test_entity_matching_boundaries_accents_and_aliases(self):
        entities = [{"label": "Method", "name": "GraphRAG"},
                    {"label": "Concept", "name": "KAG"},
                    {"label": "Author", "name": "João Silva"}]
        self.assertEqual(hybrid.match_entities(
            "GRAPH-RAG, knowledge-augmented generation e Joao Silva?", entities), entities)
        self.assertEqual(hybrid.match_entities("HippoGraphRAG e SKAG", entities), [])

    def test_rrf_deduplication_and_tie_break(self):
        ranked = hybrid.fuse_rankings(["b", "a", "b"], ["c", "a"])
        self.assertEqual([r["paper_id"] for r in ranked], ["a", "b", "c"])
        self.assertEqual(ranked[0]["retrieved_by"], ["vector", "graph"])
        self.assertAlmostEqual(ranked[0]["score"], 2 / 62)
        self.assertEqual(ranked[0]["ranks"], {"vector": 2, "graph": 2})

    def test_subgraph_direction_unique_ids_and_source_alignment(self):
        papers = {"a": paper("a"), "b": paper("b")}
        facts = [fact("a", "AUTHORED", "Author", "Shared", False),
                 fact("b", "AUTHORED", "Author", "Shared", False),
                 fact("a", "USES_METHOD", "Method", "Shared"),
                 fact("b", "EVALUATED_ON", "Dataset", "HotpotQA")]
        result = hybrid.assemble_evidence(hybrid.fuse_rankings(["a", "b"], []), papers, facts)
        self.assertEqual(len(result["subgraph"]["nodes"]), 5)
        self.assertEqual(result["subgraph"]["edges"][0],
                         {"source": "author:Shared", "target": "paper:a", "type": "AUTHORED"})
        self.assertEqual(result["sources"][0]["authors"], ["Shared"])
        self.assertEqual(result["sources"][1]["citation_id"], "S2")
        self.assertTrue(any("[S1] Dataset não identificado" in w for w in result["warnings"]))
        self.assertFalse(any("[S2]" in w for w in result["warnings"]))
        encoded = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("embedding", encoded)
        self.assertEqual(len(result["subgraph"]["edges"]), 4)

    @patch("retrieval.hybrid.check_index")
    def test_graph_independent_of_vector_and_shared_entity_expansion(self, check_index):
        session = Mock()
        responses = {
            hybrid.PAPERS_QUERY: [paper("a"), paper("b"), paper("c")],
            hybrid.VECTOR_QUERY: [{"paper_id": "a", "score": 0.9}],
            hybrid.ENTITIES_QUERY: [{"label": "Method", "name": "GraphRAG"}],
            hybrid.DIRECT_QUERY: [{"paper_id": "b", "direct_matches": 1}],
            hybrid.EXPAND_QUERY: [{"paper_id": "c", "shared_connections": 1}],
            hybrid.FACTS_QUERY: [fact("b", "USES_METHOD", "Method", "GraphRAG")],
        }
        session.run.side_effect = lambda query, **params: [Record(r) for r in responses[query]]
        driver = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=False)
        encoder = Mock()
        encoder.encode.return_value = [1.0] + [0.0] * 383
        result = hybrid.retrieve(driver, "neo4j", encoder, "Quais usam GraphRAG?", 6)
        self.assertEqual({s["paper_id"] for s in result["sources"]}, {"a", "b", "c"})
        by_id = {s["paper_id"]: s for s in result["sources"]}
        self.assertEqual(by_id["b"]["retrieved_by"], ["graph"])
        self.assertEqual(by_id["c"]["retrieved_by"], ["graph"])
        calls = {c.args[0]: c.kwargs for c in session.run.call_args_list}
        self.assertEqual(calls[hybrid.DIRECT_QUERY]["entities"], responses[hybrid.ENTITIES_QUERY])
        self.assertEqual(calls[hybrid.EXPAND_QUERY]["seed_ids"], ["b", "a"])
        self.assertEqual(calls[hybrid.EXPAND_QUERY]["per_seed"], 5)
        self.assertEqual(calls[hybrid.EXPAND_QUERY]["limit"], 20)
        self.assertEqual(calls[hybrid.DIRECT_QUERY]["limit"], 20)
        self.assertEqual(calls[hybrid.VECTOR_QUERY]["limit"], 8)
        driver.session.assert_called_once_with(database="neo4j", default_access_mode="READ")

    @patch("retrieval.hybrid.check_index")
    def test_stale_embedding_stops_before_query_encoding(self, check_index):
        stale = paper("a")
        stale["abstract"] = "Conteúdo alterado desde a indexação."
        session = Mock()
        session.run.return_value = [Record(stale)]
        driver = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=False)
        encoder = Mock()
        with self.assertRaisesRegex(RetrievalError, "desatualizados"):
            hybrid.retrieve(driver, "neo4j", encoder, "GraphRAG", 6)
        encoder.encode.assert_not_called()

    @patch("retrieval.hybrid.check_index")
    def test_direct_entity_evidence_survives_higher_rrf_expanded_neighbors(self, check_index):
        for direct_matches in ([], [{"paper_id": "b", "direct_matches": 1}]):
            with self.subTest(direct=bool(direct_matches)):
                session = Mock()
                responses = {
                    hybrid.PAPERS_QUERY: [paper("a"), paper("b"), paper("c")],
                    hybrid.VECTOR_QUERY: [{"paper_id": "a", "score": 0.9}],
                    hybrid.ENTITIES_QUERY: [{"label": "Author", "name": "Autor citado"}],
                    hybrid.DIRECT_QUERY: direct_matches,
                    hybrid.EXPAND_QUERY: [
                        {"paper_id": "a", "shared_connections": 2},
                        {"paper_id": "c", "shared_connections": 1},
                    ],
                    hybrid.FACTS_QUERY: [],
                }
                session.run.side_effect = lambda query, **params: [Record(r) for r in responses[query]]
                driver = Mock()
                driver.session.return_value.__enter__ = Mock(return_value=session)
                driver.session.return_value.__exit__ = Mock(return_value=False)
                encoder = Mock()
                encoder.encode.return_value = [1.0] + [0.0] * 383
                result = hybrid.retrieve(driver, "neo4j", encoder, "Artigos de Autor citado?", 1)
                selected = result["sources"][0]
                self.assertEqual(selected["paper_id"], "b" if direct_matches else "a")
                self.assertEqual(selected["retrieved_by"], ["graph"] if direct_matches else ["vector", "graph"])

    @patch("retrieval.hybrid.check_index")
    def test_empty_database_returns_no_evidence(self, check_index):
        session = Mock()
        session.run.return_value = []
        driver = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=False)
        encoder = Mock()
        result = hybrid.retrieve(driver, "neo4j", encoder, "GraphRAG", 6)
        self.assertEqual(result, {"sources": [], "subgraph": {"nodes": [], "edges": []}, "warnings": []})
        encoder.encode.assert_not_called()

    @patch("retrieval.hybrid.check_index")
    def test_rejects_model_mismatch_changed_concepts_and_invalid_stored_vectors(self, check_index):
        for change in (
            {"embedding_model": "outro-modelo"},
            {"keywords": ["Outro conceito"]},
            {"embedding": [0.0] * 384},
            {"embedding": [1.0] * 383},
            {"embedding": [float("nan")] * 384},
            {"embedding": ["1.0"] * 384},
        ):
            with self.subTest(change=list(change)):
                stored = {**paper("a"), **change}
                session = Mock()
                session.run.return_value = [Record(stored)]
                driver = Mock()
                driver.session.return_value.__enter__ = Mock(return_value=session)
                driver.session.return_value.__exit__ = Mock(return_value=False)
                encoder = Mock()
                with self.assertRaises(RetrievalError):
                    hybrid.retrieve(driver, "neo4j", encoder, "GraphRAG", 6)
                encoder.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
