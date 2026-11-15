from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app_integration import adaptar_resultado, consultar, url_http_valida
from retrieval.config import RetrievalError

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
BACKEND_RESULT = {
    "answer": "O artigo avalia GraphRAG no HotpotQA [S1].",
    "sources": [
        {
            "citation_id": "S1",
            "paper_id": "paper:2404.16130",
            "title": "From Local to Global: A Graph RAG Approach",
            "authors": ["John Smith", "Ana Doe"],
            "year": 2024,
            "url": "https://arxiv.org/abs/2404.16130",
        }
    ],
    "subgraph": {
        "nodes": [
            {
                "id": "paper:2404.16130",
                "label": "Paper",
                "properties": {"title": "From Local to Global"},
            },
            {
                "id": "dataset:HotpotQA",
                "label": "Dataset",
                "properties": {"name": "HotpotQA"},
            },
        ],
        "edges": [
            {
                "id": "edge:paper-hotpot",
                "source": "paper:2404.16130",
                "target": "dataset:HotpotQA",
                "type": "EVALUATED_ON",
                "provenance": {"source_ids": ["S1"]},
            }
        ],
    },
    "warnings": ["Resultados limitados às fontes recuperadas."],
}


class AdapterTests(unittest.TestCase):
    def test_maps_legacy_result_and_preserves_old_fields(self):
        adapted = adaptar_resultado(BACKEND_RESULT)
        self.assertEqual(adapted["resposta"], BACKEND_RESULT["answer"])
        self.assertEqual(adapted["fontes"][0]["citation_id"], "S1")
        self.assertEqual(
            adapted["fontes"][0]["titulo"], BACKEND_RESULT["sources"][0]["title"]
        )
        self.assertEqual(adapted["fontes"][0]["autores"], "John Smith, Ana Doe")
        self.assertEqual(adapted["nos"][1]["label"], "HotpotQA")
        self.assertEqual(adapted["arestas"][0]["id"], "edge:paper-hotpot")
        self.assertEqual(adapted["arestas"][0]["relacao"], "EVALUATED_ON")
        self.assertEqual(adapted["warnings"], BACKEND_RESULT["warnings"])
        self.assertEqual(adapted["claims"], [])

    def test_entity_id_then_name_are_stable_identity_fallbacks(self):
        result = {
            "subgraph": {
                "nodes": [
                    {
                        "label": "Concept",
                        "entity_id": "entity-42",
                        "properties": {"name": "KAG"},
                    },
                    {"label": "Dataset", "properties": {"name": "HotpotQA"}},
                ],
                "edges": [],
            }
        }
        adapted = adaptar_resultado(result)
        self.assertEqual(
            [node["id"] for node in adapted["nos"]], ["entity-42", "HotpotQA"]
        )

    def test_new_claim_planner_paths_and_query_trace_are_preserved(self):
        result = {
            **deepcopy(BACKEND_RESULT),
            "claims": [
                {
                    "kind": "dataset",
                    "text": "Usou HotpotQA.",
                    "source_ids": ["S1"],
                    "path_ids": ["P1"],
                    "quotes": ["evaluated on HotpotQA"],
                }
            ],
            "planner": {
                "status": "degraded",
                "reason": "Sem geração.",
                "queries": [
                    {
                        "original": "MATCH (p:Paper) RETURN p",
                        "cypher": None,
                        "executed": False,
                    }
                ],
            },
            "queries": [
                {"stage": "Busca vetorial", "cypher": "MATCH (p:Paper) RETURN p"}
            ],
            "paths": [
                {
                    "path_id": "P1",
                    "kind": "fact",
                    "source": "S1",
                    "target": "S1",
                    "length": 1,
                    "nodes": [],
                    "edges": [],
                }
            ],
        }
        adapted = adaptar_resultado(result)
        self.assertEqual(adapted["claims"], result["claims"])
        self.assertEqual(adapted["planner"]["status"], "degraded")
        self.assertEqual(
            adapted["planner"]["generated_queries"], result["planner"]["queries"]
        )
        self.assertEqual(adapted["queries"], result["queries"])
        self.assertEqual(adapted["paths"], result["paths"])

    def test_consultar_calls_backend_and_external_links_are_restricted(self):
        with patch(
            "app_integration.answer_question", return_value=BACKEND_RESULT
        ) as backend:
            self.assertEqual(consultar("Pergunta")["fontes"][0]["citation_id"], "S1")
        backend.assert_called_once_with("Pergunta")
        self.assertTrue(url_http_valida("https://example.org/paper"))
        self.assertFalse(url_http_valida("javascript:alert(1)"))
        self.assertFalse(url_http_valida("/paper"))


class AppFlowTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(APP_PATH)).run()

    def test_graph_claim_planner_and_fact_path_render_without_graphviz(self):
        result = deepcopy(BACKEND_RESULT)
        result["sources"][0].update(score=2 / 61, retrieved_by=["vector", "graph"])
        result["confidence"] = {
            "level": "medium",
            "reason": "Uma fonte citada nos dois canais.",
        }
        result["planner"] = {
            "status": "configuration_error",
            "reason": "Planner temporariamente indisponível.",
            "queries": [
                {
                    "original": "MATCH (q:Paper) RETURN q",
                    "cypher": None,
                    "executed": False,
                }
            ],
        }
        result["queries"] = [
            {
                "stage": "Busca vetorial",
                "cypher": "MATCH (p:Paper) RETURN p",
                "parameters": {"limit": 8},
                "executed": True,
            }
        ]
        result["paths"] = [
            {
                "path_id": "P1",
                "kind": "fact",
                "source": "S1",
                "target": "S1",
                "length": 1,
                "nodes": [],
                "edges": [result["subgraph"]["edges"][0]],
                "edge_ids": ["edge:paper-hotpot"],
            }
        ]
        result["claims"] = [
            {
                "kind": "dataset",
                "text": "A avaliação usou HotpotQA. [link](javascript:alert(1))",
                "source_ids": ["S1"],
                "path_ids": ["P1"],
                "quotes": [
                    {
                        "source_id": "S1",
                        "field": "abstract",
                        "quote": "evaluated on HotpotQA",
                        "start": 23,
                        "end": 45,
                    }
                ],
            }
        ]
        with patch("app_integration.answer_question", return_value=result):
            app = self.app()
            app.text_input[0].set_value("Pergunta").run()
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "Uma fonte citada nos dois canais." in item.value
                for item in app.markdown
            )
        )
        self.assertIn(
            "Relevância relativa (RRF): 0.032787", [item.value for item in app.caption]
        )
        self.assertIn(
            "Canais de recuperação: vetorial, grafo",
            [item.value for item in app.caption],
        )
        self.assertIn(
            "Planner temporariamente indisponível.",
            [item.value for item in app.warning],
        )
        self.assertTrue(
            any("Fatos de uma relação" in item.value for item in app.markdown)
        )
        self.assertIn(
            "dataset — A avaliação usou HotpotQA. [link](javascript:alert(1))",
            [item.value for item in app.text],
        )
        self.assertIn(
            "Trecho (S1 · abstract · 23–45): evaluated on HotpotQA",
            [item.value for item in app.text],
        )
        self.assertIn("MATCH (q:Paper) RETURN q", [item.value for item in app.code])
        self.assertIn("MATCH (p:Paper) RETURN p", [item.value for item in app.code])
        self.assertEqual(len(app.get("link_button")), 1)
        self.assertEqual(len(app.dataframe), 2)
        self.assertFalse(app.get("graphviz_chart"))

    def test_query_only_runs_on_button_and_reruns_preserve_answer(self):
        app = AppTest.from_file(str(APP_PATH))
        with patch(
            "app_integration.answer_question", return_value=BACKEND_RESULT
        ) as backend:
            app.run()
            app.text_input[0].set_value("Pergunta").run()
            backend.assert_not_called()
            app.button[0].click().run()
            backend.assert_called_once_with("Pergunta")
            app.run()
            backend.assert_called_once_with("Pergunta")
        self.assertIn(BACKEND_RESULT["answer"], [item.value for item in app.markdown])
        self.assertIn(
            "[S1] From Local to Global: A Graph RAG Approach",
            [item.label for item in app.get("link_button")],
        )
        self.assertFalse(app.exception)

    def test_no_path_messages_and_abstention_remain_available(self):
        for count, expected in (
            (1, "São necessárias pelo menos duas fontes para calcular conexões."),
            (
                2,
                "Nenhuma conexão encontrada entre as fontes no limite de quatro saltos.",
            ),
        ):
            result = deepcopy(BACKEND_RESULT)
            if count == 2:
                result["sources"].append(
                    {**result["sources"][0], "citation_id": "S2", "paper_id": "second"}
                )
            with patch("app_integration.answer_question", return_value=result):
                app = self.app()
                app.text_input[0].set_value("Pergunta").run()
                app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIn(expected, [item.value for item in app.info])

        abstention = {
            "answer": "Não encontrei evidências suficientes.",
            "sources": [],
            "subgraph": {"nodes": [], "edges": []},
            "warnings": ["Sem evidências."],
            "queries": [{"stage": "Leitura", "cypher": "MATCH (p:Paper) RETURN p"}],
        }
        with patch("app_integration.answer_question", return_value=abstention):
            app = self.app()
            app.text_input[0].set_value("Pergunta sem evidências").run()
            app.button[0].click().run()
        self.assertIn(abstention["answer"], [item.value for item in app.markdown])
        self.assertIn("Sem evidências.", [item.value for item in app.warning])
        self.assertFalse(app.exception)

    def test_query_errors_clear_old_answer_and_do_not_expose_secrets(self):
        app = AppTest.from_file(str(APP_PATH))
        with patch(
            "app_integration.answer_question",
            side_effect=[BACKEND_RESULT, RetrievalError("Falha segura.")],
        ):
            app.run()
            app.text_input[0].set_value("Pergunta antiga").run()
            app.button[0].click().run()
            app.text_input[0].set_value("Pergunta nova").run()
            app.button[0].click().run()
        self.assertIn(
            "Não foi possível consultar: Falha segura.",
            [item.value for item in app.error],
        )
        self.assertNotIn(
            BACKEND_RESULT["answer"], [item.value for item in app.markdown]
        )

        with patch(
            "app_integration.answer_question",
            side_effect=RuntimeError("api_key=do-not-show"),
        ):
            app = self.app()
            app.text_input[0].set_value("Pergunta").run()
            app.button[0].click().run()
        self.assertIn(
            "Não foi possível consultar agora. Tente novamente.",
            [item.value for item in app.error],
        )
        self.assertNotIn("do-not-show", " ".join(item.value for item in app.error))


if __name__ == "__main__":
    unittest.main()
