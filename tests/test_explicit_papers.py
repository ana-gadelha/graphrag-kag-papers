from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from retrieval import hybrid
from tests.test_hybrid import Record, fact, paper

EXPLICIT_IDS = ["2407.10805", "2509.22009"]


class ExplicitPaperTests(unittest.TestCase):
    def retrieve(
        self, question, *, status="executed", planned_ids=None, top_k=2, mode="hybrid"
    ):
        neighbors = [f"neighbor-{n}" for n in range(40)]
        session = Mock()
        responses = {
            hybrid.PAPERS_QUERY: [paper(pid) for pid in EXPLICIT_IDS + neighbors],
            hybrid.VECTOR_QUERY: [{"paper_id": neighbors[0], "score": 0.9}],
            hybrid.ENTITIES_QUERY: [],
            hybrid.EXPAND_QUERY: [{"paper_id": neighbors[0], "shared_connections": 10}],
            hybrid.PATHS_QUERY: [],
        }

        def run(query, **params):
            rows = (
                [
                    fact(pid, "AUTHORED", "Author", f"Author {pid}", False)
                    for pid in params["paper_ids"]
                ]
                if query == hybrid.FACTS_QUERY
                else responses[query]
            )
            return [Record(row) for row in rows]

        session.run.side_effect = run
        driver = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=False)
        encoder = Mock()
        encoder.encode.return_value = [1.0] + [0.0] * 383
        plan = {
            "status": status,
            "paper_ids": planned_ids or [],
            "queries": [],
            "warnings": [],
        }
        with (
            patch("retrieval.hybrid.check_index"),
            patch("retrieval.planner.plan_and_retrieve", return_value=plan) as planner,
        ):
            result = hybrid.retrieve(
                driver,
                "neo4j",
                encoder,
                question,
                top_k,
                use_planner=True,
                mode=mode,
            )
        return result, session, planner

    def test_both_explicit_sources_and_author_facts_survive_planner_failure(self):
        for status in ("quota", "truncation", "provider_error"):
            with self.subTest(status=status):
                result, session, planner = self.retrieve(
                    "Quais os coautores de 2407.10805 e 2509.22009?", status=status
                )
                self.assertEqual(
                    [s["paper_id"] for s in result["sources"]], EXPLICIT_IDS
                )
                for source in result["sources"]:
                    self.assertEqual(source["graph_origins"], ["explicit_paper_id"])
                    self.assertEqual(source["retrieved_by"], ["graph"])
                    self.assertEqual(
                        source["authors"], [f"Author {source['paper_id']}"]
                    )
                    self.assertTrue(source["abstract"])
                calls = {
                    call.args[0]: call.kwargs for call in session.run.call_args_list
                }
                self.assertEqual(calls[hybrid.FACTS_QUERY]["paper_ids"], EXPLICIT_IDS)
                self.assertEqual(
                    calls[hybrid.EXPAND_QUERY]["seed_ids"][:2], EXPLICIT_IDS
                )
                self.assertEqual(planner.call_args.kwargs["seed_ids"][:2], EXPLICIT_IDS)
                self.assertEqual(len(result["subgraph"]["edges"]), 2)

    def test_many_planned_neighbors_cannot_displace_explicit_ids(self):
        result, _, _ = self.retrieve(
            "2407.10805 e 2509.22009", planned_ids=[f"neighbor-{n}" for n in range(40)]
        )
        self.assertEqual([s["paper_id"] for s in result["sources"]], EXPLICIT_IDS)

    def test_explicit_overflow_uses_question_order_and_warns(self):
        result, _, _ = self.retrieve("2509.22009 antes de 2407.10805", top_k=1)
        self.assertEqual([s["paper_id"] for s in result["sources"]], [EXPLICIT_IDS[1]])
        self.assertTrue(
            any("mais artigos explicitamente" in w for w in result["warnings"])
        )

    def test_exact_ids_doi_and_urls_match_but_unknown_or_partial_ids_do_not(self):
        corpus = {pid: paper(pid) for pid in EXPLICIT_IDS + ["10.1234/example.paper"]}
        cases = [
            ("arXiv:2407.10805; 2509.22009.", EXPLICIT_IDS),
            ("https://arxiv.org/pdf/2407.10805.pdf", [EXPLICIT_IDS[0]]),
            ("https://example.org/2509.22009", [EXPLICIT_IDS[1]]),
            ("https://doi.org/10.1234/example.paper", ["10.1234/example.paper"]),
            ("10.1234/example.paper.", ["10.1234/example.paper"]),
            ("9999.99999 2407.1080 12407.10805 2407.108050", []),
            ("2407.10805v2 2407.10805.1 prefix2407.10805", []),
            ("10.1234/example.paper.extra 10.1234/example.paper/suffix", []),
            ("2407.10805 2407.10805", [EXPLICIT_IDS[0]]),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                self.assertEqual(
                    hybrid.match_explicit_papers(question, corpus), expected
                )

    def test_vector_baseline_does_not_promote_literal_ids(self):
        result, session, planner = self.retrieve(
            "2407.10805 e 2509.22009", mode="vector"
        )
        self.assertEqual([s["paper_id"] for s in result["sources"]], ["neighbor-0"])
        self.assertEqual(result["sources"][0]["graph_origins"], [])
        self.assertEqual(result["sources"][0]["retrieved_by"], ["vector"])
        planner.assert_not_called()
        calls = [call.args[0] for call in session.run.call_args_list]
        self.assertNotIn(hybrid.EXPAND_QUERY, calls)
        self.assertNotIn(hybrid.FACTS_QUERY, calls)


if __name__ == "__main__":
    unittest.main()
