from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from graphviz import Digraph
from streamlit.testing.v1 import AppTest

from app_integration import adaptar_resultado, consultar, montar_dot, url_http_valida
from retrieval.config import RetrievalError

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
BACKEND_RESULT = {
    "question": "Quais artigos usam GraphRAG?",
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
                "source": "paper:2404.16130",
                "target": "dataset:HotpotQA",
                "type": "EVALUATED_ON",
            }
        ],
    },
    "warnings": ["Resultados limitados às fontes recuperadas."],
}


class AdapterTests(unittest.TestCase):
    def test_maps_public_retrieval_shape_to_portuguese_interface_shape(self) -> None:
        adapted = adaptar_resultado(BACKEND_RESULT)

        self.assertEqual(adapted["resposta"], BACKEND_RESULT["answer"])
        self.assertEqual(adapted["fontes"][0]["citation_id"], "S1")
        self.assertEqual(
            adapted["fontes"][0]["titulo"], BACKEND_RESULT["sources"][0]["title"]
        )
        self.assertEqual(adapted["fontes"][0]["autores"], "John Smith, Ana Doe")
        self.assertEqual(adapted["fontes"][0]["ano"], 2024)
        self.assertEqual(
            adapted["fontes"][0]["url"], BACKEND_RESULT["sources"][0]["url"]
        )
        self.assertEqual(adapted["nos"][0]["label"], "From Local to Global")
        self.assertEqual(adapted["nos"][1]["label"], "HotpotQA")
        self.assertEqual(adapted["nos"][1]["tipo"], "Dataset")
        self.assertEqual(
            adapted["arestas"],
            [
                {
                    "origem": "paper:2404.16130",
                    "destino": "dataset:HotpotQA",
                    "relacao": "EVALUATED_ON",
                }
            ],
        )
        self.assertEqual(adapted["warnings"], BACKEND_RESULT["warnings"])

    def test_consultar_calls_the_public_backend_directly(self) -> None:
        with patch(
            "app_integration.answer_question", return_value=BACKEND_RESULT
        ) as backend:
            result = consultar("Quais artigos usam GraphRAG?")

        backend.assert_called_once_with("Quais artigos usam GraphRAG?")
        self.assertEqual(result["fontes"][0]["citation_id"], "S1")

    def test_dot_graph_preserves_direction_and_escapes_quotes_backslashes_and_newlines(
        self,
    ) -> None:
        nodes = [
            {
                "id": 'paper:"\\\n1',
                "label": 'Título "A" \\ linha\nseguinte',
                "tipo": "Paper",
            },
            {"id": "author:1", "label": "Ana", "tipo": "Author"},
            {"id": "concept:1", "label": "GraphRAG", "tipo": "Concept"},
            {"id": "method:1", "label": "KAG", "tipo": "Method"},
            {"id": "dataset:1", "label": "HotpotQA", "tipo": "Dataset"},
        ]
        edges = [
            {
                "origem": nodes[0]["id"],
                "destino": "author:1",
                "relacao": 'AUTHORED "by"',
            }
        ]

        graph = montar_dot(nodes, edges)

        self.assertIsInstance(graph, Digraph)
        source = graph.source
        self.assertIn("rankdir=LR", source)
        self.assertIn('fillcolor="#cfe8ff"', source)
        self.assertIn('fillcolor="#e8d5ff"', source)
        self.assertIn('fillcolor="#fff1c2"', source)
        self.assertIn('fillcolor="#ffe8b3"', source)
        self.assertIn('fillcolor="#d5f5d5"', source)
        self.assertIn('"paper:\\"\\\\\\n1"', source)
        self.assertIn(r'label="Título \"A\" \\ linha\nseguinte"', source)
        self.assertIn('"paper:\\"\\\\\\n1" -> "author:1"', source)
        self.assertNotIn('Título "A" \\ linha\nseguinte', source)

    def test_external_links_accept_only_http_and_https(self) -> None:
        self.assertTrue(url_http_valida("https://example.org/paper"))
        self.assertTrue(url_http_valida("http://example.org/paper"))
        self.assertFalse(url_http_valida("javascript:alert(1)"))
        self.assertFalse(url_http_valida("/paper"))
        self.assertFalse(url_http_valida(None))


class AppFlowTests(unittest.TestCase):
    def _app(self) -> AppTest:
        return AppTest.from_file(str(APP_PATH)).run()

    def test_displays_queries_confidence_rrf_paths_and_connector_labels(self):
        result = deepcopy(BACKEND_RESULT)
        result["sources"][0].update(score=2 / 61, retrieved_by=["vector", "graph"])
        result["confidence"] = {
            "level": "medium", "method": "cited-channel-agreement-v1",
            "reason": "Uma fonte citada nos dois canais.",
        }
        result["queries"] = [{
            "stage": "Caminhos mínimos entre fontes", "cypher": "MATCH path = shortestPath(...)",
            "parameters": {"pairs": [{"source_id": "a", "target_id": "b"}]},
        }]
        connector = {
            "id": "paper:connector", "label": "Paper", "connector": True,
            "properties": {"title": "Artigo intermediário"},
        }
        second_source = {**deepcopy(result["sources"][0]), "citation_id": "S2",
                         "paper_id": "second", "title": "Segundo artigo"}
        result["sources"].append(second_source)
        concept = {"id": "concept:shared", "label": "Concept", "properties": {"name": "Conceito"}}
        endpoint = {"id": "paper:second", "label": "Paper", "properties": {"title": "Segundo artigo"}}
        result["subgraph"]["nodes"].extend([connector, concept, endpoint])
        result["subgraph"]["edges"].extend([
            {"source": connector["id"], "target": "dataset:HotpotQA", "type": "EVALUATED_ON"},
            {"source": connector["id"], "target": concept["id"], "type": "HAS_CONCEPT"},
            {"source": endpoint["id"], "target": concept["id"], "type": "HAS_CONCEPT"},
        ])
        result["paths"] = [{
            "source": "S1", "target": "S2", "length": 4,
            "nodes": result["subgraph"]["nodes"], "edges": result["subgraph"]["edges"],
        }]
        with patch("app_integration.answer_question", return_value=result):
            app = self._app()
            app.text_input[0].set_value("Pergunta").run()
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("Uma fonte citada nos dois canais.", " ".join(m.value for m in app.markdown))
        self.assertIn("Relevância relativa (RRF): 0.032787", [c.value for c in app.caption])
        self.assertIn("Canais de recuperação: vetorial, grafo", [c.value for c in app.caption])
        self.assertEqual([c.value for c in app.code], [result["queries"][0]["cypher"]])
        self.assertIn("[S1] ↔ [S2] · 4 saltos", [m.value for m in app.markdown])
        self.assertIn("Artigo intermediário (conector)", app.text[0].value)
        graph = app.get("graphviz_chart")[0].proto.spec
        self.assertIn("Artigo intermediário (conector)", graph)
        adapted = adaptar_resultado(result)
        for key in ("queries", "paths", "confidence"):
            self.assertEqual(adapted[key], result[key])

    def test_no_path_has_distinct_messages_for_one_source_and_disconnection(self):
        for count, expected in (
            (1, "São necessárias pelo menos duas fontes para calcular conexões."),
            (2, "Nenhuma conexão encontrada entre as fontes no limite de quatro saltos."),
        ):
            result = deepcopy(BACKEND_RESULT)
            if count == 2:
                result["sources"].append({**deepcopy(result["sources"][0]), "citation_id": "S2",
                                          "paper_id": "second"})
            with self.subTest(count=count), patch("app_integration.answer_question", return_value=result):
                app = self._app()
                app.text_input[0].set_value("Pergunta").run()
                app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIn(expected, [i.value for i in app.info])

    def test_abstention_preserves_trace_and_shows_low_confidence(self):
        result = {
            "answer": "Não encontrei evidências suficientes.", "sources": [],
            "subgraph": {"nodes": [], "edges": []}, "warnings": [], "paths": [],
            "queries": [{"stage": "Leitura dos artigos", "cypher": "MATCH (p:Paper) RETURN p",
                         "parameters": {}}],
            "confidence": {"level": "low", "reason": "O sistema se absteve.",
                           "method": "cited-channel-agreement-v1"},
        }
        with patch("app_integration.answer_question", return_value=result):
            app = self._app()
            app.text_input[0].set_value("Pergunta").run()
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("**Baixa** — O sistema se absteve.", [m.value for m in app.markdown])
        self.assertEqual(app.code[0].value, result["queries"][0]["cypher"])

    def test_query_runs_only_on_button_and_normal_reruns_preserve_result(self) -> None:
        app = AppTest.from_file(str(APP_PATH))
        with patch(
            "app_integration.answer_question", return_value=BACKEND_RESULT
        ) as backend:
            app.run()
            app.text_input[0].set_value("Quais artigos usam GraphRAG?").run()
            backend.assert_not_called()

            app.button[0].click().run()
            backend.assert_called_once_with("Quais artigos usam GraphRAG?")
            app.run()
            backend.assert_called_once_with("Quais artigos usam GraphRAG?")

        self.assertIn(
            "O artigo avalia GraphRAG no HotpotQA [S1].",
            [item.value for item in app.markdown],
        )
        self.assertIn("Fontes recuperadas", [item.value for item in app.subheader])
        self.assertIn(
            "Resultados limitados às fontes recuperadas.",
            [item.value for item in app.warning],
        )
        self.assertIn(
            "[S1] From Local to Global: A Graph RAG Approach",
            [item.label for item in app.get("link_button")],
        )
        self.assertIn(
            "John Smith, Ana Doe · 2024", [item.value for item in app.caption]
        )

        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("graphviz_chart")), 1)
        self.assertEqual(len(app.dataframe), 2)

    def test_empty_query_skips_backend_and_clears_previous_results(self) -> None:
        app = AppTest.from_file(str(APP_PATH))
        with patch(
            "app_integration.answer_question", return_value=BACKEND_RESULT
        ) as backend:
            app.run()
            app.text_input[0].set_value("Pergunta anterior").run()
            app.button[0].click().run()
            backend.assert_called_once_with("Pergunta anterior")

            app.text_input[0].set_value("   ").run()
            app.button[0].click().run()
            backend.assert_called_once_with("Pergunta anterior")

        self.assertIn(
            "Digite uma pergunta antes de consultar.",
            [item.value for item in app.warning],
        )
        self.assertNotIn(
            "O artigo avalia GraphRAG no HotpotQA [S1].",
            [item.value for item in app.markdown],
        )

    def test_abstention_and_warnings_are_visible_without_sources(self) -> None:
        abstention = {
            "answer": "Não encontrei evidências suficientes para responder.",
            "sources": [],
            "subgraph": {"nodes": [], "edges": []},
            "warnings": ["O grafo não retornou evidências para esta consulta."],
        }
        app = AppTest.from_file(str(APP_PATH))
        with patch("app_integration.answer_question", return_value=abstention):
            app.run()
            app.text_input[0].set_value("Pergunta sem evidências").run()
            app.button[0].click().run()

        self.assertIn(
            "Não encontrei evidências suficientes para responder.",
            [item.value for item in app.markdown],
        )
        self.assertIn(
            "O grafo não retornou evidências para esta consulta.",
            [item.value for item in app.warning],
        )
        self.assertIn(
            "Nenhuma fonte foi recuperada para esta pergunta.",
            [item.value for item in app.info],
        )

        self.assertFalse(app.exception)

    def test_abstention_with_retrieved_sources_keeps_context_visible(self) -> None:
        result = {
            **BACKEND_RESULT,
            "answer": "Não encontrei evidências suficientes para responder.",
            "warnings": ["As fontes recuperadas não permitem responder à pergunta."],
        }
        app = AppTest.from_file(str(APP_PATH))
        with patch("app_integration.answer_question", return_value=result):
            app.run()
            app.text_input[0].set_value("Pergunta fora do domínio").run()
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn(result["answer"], [item.value for item in app.markdown])
        self.assertEqual(len(app.get("link_button")), 1)
        self.assertEqual(len(app.get("graphviz_chart")), 1)
        self.assertIn(result["warnings"][0], [item.value for item in app.warning])

    def test_retrieval_errors_are_shown_safely_and_unknown_errors_are_generic(
        self,
    ) -> None:
        app = AppTest.from_file(str(APP_PATH))
        with patch(
            "app_integration.answer_question",
            side_effect=[
                BACKEND_RESULT,
                RetrievalError("Falha segura de recuperação."),
            ],
        ):
            app.run()
            app.text_input[0].set_value("Pergunta anterior").run()
            app.button[0].click().run()
            self.assertIn(
                "O artigo avalia GraphRAG no HotpotQA [S1].",
                [item.value for item in app.markdown],
            )

            app.text_input[0].set_value("Nova pergunta").run()
            app.button[0].click().run()
        self.assertIn(
            "Não foi possível consultar: Falha segura de recuperação.",
            [item.value for item in app.error],
        )
        self.assertNotIn(
            "O artigo avalia GraphRAG no HotpotQA [S1].",
            [item.value for item in app.markdown],
        )

        app = AppTest.from_file(str(APP_PATH))
        with patch(
            "app_integration.answer_question",
            side_effect=RuntimeError("api_key=segredo-nao-exibir"),
        ):
            app.run()
            app.text_input[0].set_value("Pergunta").run()
            app.button[0].click().run()
        errors = [item.value for item in app.error]
        self.assertIn("Não foi possível consultar agora. Tente novamente.", errors)
        self.assertNotIn("segredo-nao-exibir", " ".join(errors))


if __name__ == "__main__":
    unittest.main()
