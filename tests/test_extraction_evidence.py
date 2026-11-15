from __future__ import annotations

import unittest
from unittest.mock import patch

from retrieval.config import LLMSettings
from src.entity_resolution import EntityRegistry
from src.extract_entities import enrich_papers, validate_mentions
from src.import_graph import validate_import_input
from tests.test_pipeline import SAMPLE


class ExtractionEvidenceTests(unittest.TestCase):
    def test_heuristic_prefers_compound_names_but_keeps_independent_short_alias(self):
        from src.extract_entities import _heuristic_mentions

        registry = EntityRegistry(
            {
                "version": 1,
                "entities": [
                    {
                        "label": "Method",
                        "canonical_name": "GraphRAG",
                        "aliases": ["Graph RAG"],
                    },
                    {"label": "Method", "canonical_name": "RAG", "aliases": []},
                ],
            }
        )
        paper = {**SAMPLE, "abstract": "We compare RAG with another method."}
        mentions = _heuristic_mentions(paper, registry)
        self.assertEqual(
            {(m["canonical_name"], m["relation"], m["field"]) for m in mentions},
            {
                ("GraphRAG", "PROPOSES", "title"),
                ("RAG", "USES_METHOD", "abstract"),
            },
        )
        paper["abstract"] = "We present GraphRAG and evaluate it on HotpotQA."
        self.assertEqual(
            [m["canonical_name"] for m in _heuristic_mentions(paper, registry)],
            ["GraphRAG"],
        )

    def test_exact_quote_offsets_and_relation_are_retained(self):
        paper = {
            **SAMPLE,
            "title": SAMPLE["title"].strip(),
            "abstract": SAMPLE["abstract"].strip(),
        }
        quote = "We present GraphRAG and evaluate it on HotpotQA."
        mentions = validate_mentions(
            {
                "mentions": [
                    {
                        "surface_form": "GraphRAG",
                        "label": "Method",
                        "relation": "PROPOSES",
                        "field": "abstract",
                        "quote": quote,
                    }
                ]
            },
            paper,
        )
        mention = mentions[0]
        self.assertEqual(
            paper["abstract"][mention["start"] : mention["end"]], "GraphRAG"
        )
        self.assertIn(mention["surface_form"], mention["quote"])

    def test_missing_quote_surface_and_unsupported_relation_are_rejected(self):
        paper = {
            **SAMPLE,
            "title": SAMPLE["title"].strip(),
            "abstract": SAMPLE["abstract"].strip(),
        }
        cases = [
            {
                "surface_form": "GraphRAG",
                "label": "Method",
                "relation": "PROPOSES",
                "field": "abstract",
                "quote": "not present",
            },
            {
                "surface_form": "Missing",
                "label": "Method",
                "relation": "PROPOSES",
                "field": "abstract",
                "quote": paper["abstract"],
            },
            {
                "surface_form": "GraphRAG",
                "label": "Dataset",
                "relation": "PROPOSES",
                "field": "abstract",
                "quote": paper["abstract"],
            },
            {
                "surface_form": "GraphRAG",
                "label": "Method",
                "relation": "PROPOSES",
                "field": "title",
                "quote": paper["title"],
            },
        ]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_mentions({"mentions": [raw]}, paper)

    def test_empty_mentions_is_valid_and_not_a_failed_extraction(self):
        paper = {
            **SAMPLE,
            "title": SAMPLE["title"].strip(),
            "abstract": "No explicit methods or datasets are named.",
        }
        self.assertEqual(validate_mentions({"mentions": []}, paper), [])

    def test_explicit_integration_application_and_comparison_support_method_usage(self):
        quotes = [
            "Integrating Graph Convolutional Networks (GCNs) and Large Language Models (LLMs) through hard prompting",
            "We combine LLMs with graph retrieval.",
            "Our method has faster indexing than GRAG.",
            "The speedup over GRAG in retrieval is substantial.",
            "We utilize LLMs for query expansion.",
            "We apply LLMs to query expansion.",
            "We incorporate LLMs into the retrieval pipeline.",
            "Our GRAG approach significantly outperforms current state-of-the-art RAG methods.",
        ]
        for quote in quotes:
            surface = "GRAG" if "GRAG" in quote else "LLMs"
            with self.subTest(quote=quote):
                mentions = validate_mentions(
                    {
                        "mentions": [
                            {
                                "surface_form": surface,
                                "label": "Method",
                                "relation": "USES_METHOD",
                                "field": "abstract",
                                "quote": quote,
                            }
                        ]
                    },
                    {**SAMPLE, "abstract": quote},
                )
                self.assertEqual(mentions[0]["quote"], quote)
                self.assertEqual(mentions[0]["relation"], "USES_METHOD")

    def test_numeric_dataset_performance_quotes_support_evaluation(self):
        cases = [
            ("2wiki", "achieving a relative improvement of 19.6% on 2wiki"),
            ("hotpotQA", "and 33.5% on hotpotQA in terms of F1 score"),
        ]
        for surface, quote in cases:
            with self.subTest(quote=quote):
                mentions = validate_mentions(
                    {
                        "mentions": [
                            {
                                "surface_form": surface,
                                "label": "Dataset",
                                "relation": "EVALUATED_ON",
                                "field": "abstract",
                                "quote": quote,
                            }
                        ]
                    },
                    {**SAMPLE, "abstract": quote},
                )
                self.assertEqual(mentions[0]["quote"], quote)
                self.assertEqual(mentions[0]["relation"], "EVALUATED_ON")

    def test_discussion_bare_names_and_negated_new_cues_remain_unsupported(self):
        cases = [
            (
                "GraphRAG",
                "Method",
                "USES_METHOD",
                "abstract",
                "we systematically analyze the technical foundations of GraphRAG and examine current implementations",
            ),
            ("HippoRAG", "Method", "PROPOSES", "title", "HippoRAG"),
            (
                "GraphRAG-Filtering",
                "Method",
                "PROPOSES",
                "abstract",
                "consisting of GraphRAG-Filtering and GraphRAG-Integration.",
            ),
            (
                "Personalized PageRank",
                "Method",
                "USES_METHOD",
                "abstract",
                "Personalized PageRank",
            ),
            ("LLMs", "Method", "USES_METHOD", "abstract", "LLMs"),
            ("IRCoT", "Method", "USES_METHOD", "abstract", "IRCoT"),
            (
                "LLMs",
                "Method",
                "USES_METHOD",
                "abstract",
                "We do not integrate LLMs into the pipeline.",
            ),
            (
                "LLMs",
                "Method",
                "USES_METHOD",
                "abstract",
                "The pipeline operates without incorporating LLMs.",
            ),
            (
                "GRAG",
                "Method",
                "USES_METHOD",
                "abstract",
                "GRAG does not significantly outperform other approaches.",
            ),
            (
                "HotpotQA",
                "Dataset",
                "EVALUATED_ON",
                "abstract",
                "We discuss HotpotQA and define the F1 score.",
            ),
            (
                "HotpotQA",
                "Dataset",
                "EVALUATED_ON",
                "abstract",
                "HotpotQA contains 33.5% multi-hop questions.",
            ),
            (
                "2wiki",
                "Dataset",
                "EVALUATED_ON",
                "abstract",
                "There is no improvement of 19.6% on 2wiki.",
            ),
        ]
        for surface, label, relation, field, quote in cases:
            with (
                self.subTest(quote=quote),
                self.assertRaisesRegex(ValueError, "does not support"),
            ):
                validate_mentions(
                    {
                        "mentions": [
                            {
                                "surface_form": surface,
                                "label": label,
                                "relation": relation,
                                "field": field,
                                "quote": quote,
                            }
                        ]
                    },
                    {**SAMPLE, field: quote},
                )

    def test_curated_mode_keeps_compatibility_lists_and_metadata_origin(self):
        registry = EntityRegistry.load()
        enriched, report = enrich_papers(
            [SAMPLE],
            "curated",
            curated={
                SAMPLE["id"]: {
                    "proposed_methods": ["GraphRAG"],
                    "used_methods": [],
                    "datasets": ["HotpotQA"],
                }
            },
            registry=registry,
        )
        self.assertTrue(report["valid"])
        extracted = enriched[0]["extracted"]
        self.assertEqual(extracted["proposed_methods"], ["GraphRAG"])
        self.assertEqual(extracted["datasets"], ["HotpotQA"])
        self.assertTrue(
            all(e["origin"] == "paper_metadata" for e in extracted["metadata_entities"])
        )
        self.assertTrue(validate_import_input(enriched))

    def test_invalid_relation_response_is_not_importable(self):
        enriched, report = enrich_papers(
            [SAMPLE],
            "curated",
            curated={
                SAMPLE["id"]: {
                    "proposed_methods": ["NotInSource"],
                    "used_methods": [],
                    "datasets": [],
                }
            },
            registry=EntityRegistry.load(),
        )
        self.assertFalse(report["valid"])
        with self.assertRaises(ValueError):
            validate_import_input(enriched)

    def test_auto_uses_stage_client_and_accepts_only_valid_evidence(self):
        class Generated(dict):
            receipt = {
                "effective_model": "mock-extractor-v1",
                "cache_hit": True,
                "usage": {"input_tokens": 20, "output_tokens": 12},
                "seconds": 0.1,
            }

        valid_response = Generated(
            mentions=[
                {
                    "surface_form": "GraphRAG",
                    "label": "Method",
                    "relation": "PROPOSES",
                    "field": "abstract",
                    "quote": "We present GraphRAG and evaluate it on HotpotQA.",
                },
                {
                    "surface_form": "HotpotQA",
                    "label": "Dataset",
                    "relation": "EVALUATED_ON",
                    "field": "abstract",
                    "quote": "We present GraphRAG and evaluate it on HotpotQA.",
                },
            ]
        )
        settings = LLMSettings(model="configured-extractor-v0", api_key="test")
        with (
            patch("retrieval.llm.stage_settings", return_value=settings) as stage,
            patch(
                "retrieval.llm.generate_json", return_value=valid_response
            ) as generate,
            patch("retrieval.llm.accept_json") as accept,
        ):
            enriched, report = enrich_papers(
                [SAMPLE], "auto", model="cli-override", registry=EntityRegistry.load()
            )
        stage.assert_called_once_with("extraction")
        self.assertEqual(
            generate.call_args.kwargs["namespace"], "text-to-graph-extraction"
        )
        self.assertEqual(generate.call_args.kwargs["version"], "1")
        self.assertEqual(generate.call_args.args[0].model, "cli-override")
        accept.assert_called_once_with(valid_response)
        self.assertTrue(report["valid"])
        self.assertEqual(report["model"], "mock-extractor-v1")
        self.assertTrue(report["receipts"][0]["cache_hit"])
        mention = enriched[0]["extracted"]["mentions"][0]
        self.assertTrue(mention["entity_id"].startswith("method:"))
        self.assertEqual(enriched[0]["extracted"]["proposed_methods"], ["GraphRAG"])
        self.assertTrue(validate_import_input(enriched))

    def test_auto_invalid_response_is_not_cached_or_importable(self):
        class Generated(dict):
            pass

        invalid_response = Generated(
            mentions=[
                {
                    "surface_form": "Graph RAG",
                    "label": "Method",
                    "relation": "USES_METHOD",
                    "field": "title",
                    "quote": "From Local to Global: A Graph RAG Approach",
                }
            ]
        )
        with (
            patch(
                "retrieval.llm.stage_settings",
                return_value=LLMSettings(model="mock", api_key="test"),
            ),
            patch("retrieval.llm.generate_json", return_value=invalid_response),
            patch("retrieval.llm.accept_json") as accept,
        ):
            enriched, report = enrich_papers(
                [SAMPLE], "auto", registry=EntityRegistry.load()
            )
        accept.assert_not_called()
        self.assertFalse(report["valid"])
        with self.assertRaises(ValueError):
            validate_import_input(enriched)


if __name__ == "__main__":
    unittest.main()
