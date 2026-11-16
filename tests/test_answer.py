from __future__ import annotations

import io
import json
import re
import unittest
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from retrieval.__main__ import main as cli_main
from retrieval.answer import answer_question
from retrieval.config import RetrievalError, Settings
from retrieval.confidence import assess_confidence

SOURCES = [
    {
        "citation_id": "S1",
        "paper_id": "2404.16130",
        "title": "From Local to Global: A Graph RAG Approach",
        "authors": ["John Smith"],
        "year": 2024,
        "url": "https://arxiv.org/abs/2404.16130",
        "abstract": "We evaluate GraphRAG on HotpotQA.",
        "relations": [
            {
                "type": "EVALUATED_ON",
                "entity_label": "Dataset",
                "entity_name": "HotpotQA",
            }
        ],
        "retrieved_by": ["vector"],
        "ranks": {"vector": 1},
        "score": 1 / 61,
    }
]
EVIDENCE = {
    "sources": SOURCES,
    "subgraph": {"nodes": [{"id": "paper:1"}], "edges": []},
    "warnings": ["Resultados limitados às fontes recuperadas."],
    "queries": [{"stage": "Teste", "cypher": "RETURN $value", "parameters": {"value": 1}}],
    "paths": [],
}


class FakeDriver:
    def __init__(self) -> None:
        self.verify_connectivity = Mock()
        self.close = Mock()


class FakeOpenAI:
    def __init__(
        self,
        output_text: str = "O artigo avalia GraphRAG no HotpotQA [S1].",
        status: str | None = "completed",
        *,
        raw: bool = False,
        abstained: bool = False,
    ) -> None:
        self.output_text = (
            output_text
            if raw
            else json.dumps(
                {
                    "claims": ([] if abstained and not re.findall(r"S\d+", output_text) else [{
                        "kind": "document", "text": re.sub(r"\[S\d+\]", "", output_text).strip(),
                        "source_ids": list(dict.fromkeys(re.findall(r"S\d+", output_text))), "path_ids": [],
                        "quotes": [{"source_id": "S1", "field": "abstract", "quote": SOURCES[0]["abstract"]}],
                    }]),
                    "abstained": abstained,
                }
            )
        )
        response_attrs = {"output_text": self.output_text}
        if status is not None:
            response_attrs["status"] = status
        self.responses = SimpleNamespace(
            create=Mock(return_value=SimpleNamespace(**response_attrs))
        )

    def __enter__(self) -> FakeOpenAI:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class AnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        runtime_patch = patch("retrieval.llm._runtime_directory", return_value=None)
        runtime_patch.start()
        self.addCleanup(runtime_patch.stop)
        self.settings = Settings(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="not-a-real-password",
            database="neo4j",
            openai_model="gpt-4.1-mini",
            openai_api_key="not-a-real-api-key",
        )
        self.driver = FakeDriver()
        self.settings_patch = patch(
            "retrieval.answer.Settings.from_env",
            return_value=self.settings,
        )
        self.driver_patch = patch(
            "retrieval.answer._new_driver", return_value=self.driver
        )
        self.encoder_patch = patch(
            "retrieval.answer._load_encoder", return_value=object()
        )
        self.retrieve_patch = patch("retrieval.answer._retrieve", return_value=EVIDENCE)
        self.client = FakeOpenAI()
        self.client_patch = patch(
            "retrieval.answer._openai_client", return_value=self.client
        )

        self.settings_from_env = self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)
        self.new_driver = self.driver_patch.start()
        self.addCleanup(self.driver_patch.stop)
        self.load_encoder = self.encoder_patch.start()
        self.addCleanup(self.encoder_patch.stop)
        self.retrieve = self.retrieve_patch.start()
        self.addCleanup(self.retrieve_patch.stop)
        self.openai_client = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)

    def test_validates_question_and_top_k_before_loading_settings(self) -> None:
        invalid_calls = [
            (None, 6),
            ("", 6),
            (" \t\n ", 6),
            ("x" * 4_001, 6),
            ("question", True),
            ("question", False),
            ("question", 0),
            ("question", 21),
            ("question", 1.5),
            ("question", "6"),
        ]
        for question, top_k in invalid_calls:
            with self.subTest(question_type=type(question).__name__, top_k=top_k):
                with self.assertRaises(RetrievalError):
                    answer_question(question, top_k=top_k)  # type: ignore[arg-type]

        self.settings_from_env.assert_not_called()
        self.new_driver.assert_not_called()
        self.retrieve.assert_not_called()
        self.openai_client.assert_not_called()

    def test_retrieves_with_requested_top_k_and_returns_complete_contract(self) -> None:
        result = answer_question("  O que foi avaliado?  ", top_k=3)

        self.driver.verify_connectivity.assert_called_once_with()
        self.retrieve.assert_called_once_with(
            self.driver,
            "neo4j",
            self.load_encoder.return_value,
            "O que foi avaliado?",
            3,
        )
        self.driver.close.assert_called_once_with()
        self.openai_client.assert_called_once_with(self.settings.generation)
        self.assertEqual(
            {k: result[k] for k in ("question", "answer", "sources", "subgraph", "warnings", "queries", "paths", "confidence")},
            {
                "question": "O que foi avaliado?",
                "answer": "O artigo avalia GraphRAG no HotpotQA . [S1]",
                "sources": SOURCES,
                "subgraph": EVIDENCE["subgraph"],
                "warnings": EVIDENCE["warnings"],
                "queries": EVIDENCE["queries"],
                "paths": EVIDENCE["paths"],
                "confidence": assess_confidence(result["answer"], SOURCES, abstained=False),
            },
        )

        request = self.client.responses.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-4.1-mini")
        self.assertEqual(request["store"], False)
        self.assertEqual(request["max_output_tokens"], 2_000)
        self.assertIn("não confiável", request["instructions"])
        prompt = json.loads(request["input"])
        self.assertEqual(prompt["question"], "O que foi avaliado?")
        self.assertEqual(prompt["sources"], SOURCES)
        self.assertEqual(prompt["paths"], EVIDENCE["paths"])
        self.assertNotIn("queries", prompt)
        self.assertNotIn("confidence", prompt)

    def test_answer_recovers_literal_superscript_quote_without_another_llm_call(self) -> None:
        data = deepcopy(EVIDENCE)
        literal_quote = "We evaluate KG$^2$RAG on HotpotQA."
        data["sources"][0]["abstract"] = "Prefix. " + literal_quote
        self.retrieve.return_value = data
        payload = {
            "abstained": False,
            "claims": [{
                "kind": "document", "text": "O trabalho usa HotpotQA.",
                "source_ids": ["S1"], "path_ids": [],
                "quotes": [{"source_id": "S1", "field": "abstract",
                            "quote": "We evaluate KG²RAG on HotpotQA."}],
            }],
        }
        client = FakeOpenAI(json.dumps(payload), raw=True)
        self.openai_client.return_value = client

        result = answer_question("Quais trabalhos usam HotpotQA?")

        self.assertFalse(result["abstained"])
        quote = result["claims"][0]["quotes"][0]
        self.assertEqual(quote["quote"], literal_quote)
        self.assertEqual(quote["start"], len("Prefix. "))
        self.assertEqual(quote["end"], len("Prefix. ") + len(literal_quote))
        client.responses.create.assert_called_once()

    def test_no_evidence_abstains_without_api_key_or_llm_call(self) -> None:
        self.settings = Settings(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="not-a-real-password",
            openai_api_key=None,
        )
        self.settings_from_env.return_value = self.settings
        evidence = {
            "sources": [],
            "subgraph": {"nodes": [], "edges": []},
            "warnings": [],
        }
        self.retrieve.return_value = evidence

        result = answer_question("Qual método foi proposto?")

        self.assertEqual(
            result["answer"],
            "Não encontrei evidências suficientes nos artigos recuperados "
            "para responder a essa pergunta.",
        )
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["subgraph"], evidence["subgraph"])
        self.assertEqual(result["confidence"]["level"], "low")
        self.assertEqual(result["queries"], [])
        self.openai_client.assert_not_called()
        self.driver.close.assert_called_once_with()

    def test_paths_reach_generator_and_public_json_without_changing_confidence(self):
        evidence = deepcopy(EVIDENCE)
        evidence["sources"].append({**deepcopy(SOURCES[0]), "citation_id": "S2", "paper_id": "second"})
        nodes = [{"id": "paper:1", "properties": {"abstract": "Abstract do primeiro artigo."}},
                 {"id": "author:shared", "properties": {"name": "Autor"}},
                 {"id": "paper:second", "properties": {"abstract": "Abstract do segundo artigo."}}]
        evidence["subgraph"]["nodes"] = nodes
        evidence["paths"] = [{"source": "S1", "target": "S2", "length": 2,
                              "nodes": nodes, "edges": []}]
        self.retrieve.return_value = evidence
        result = answer_question("Pergunta?")
        context = json.loads(self.client.responses.create.call_args.kwargs["input"])
        self.assertEqual(context["paths"], [{
            "path_id": None, "kind": "shortest", "source": "S1", "target": "S2", "length": 2,
            "node_ids": [node["id"] for node in nodes], "edge_ids": [],
        }])
        self.assertNotIn("abstract", context["subgraph"]["nodes"][0]["properties"])
        self.assertEqual(result["paths"], evidence["paths"])
        self.assertEqual(result["confidence"]["level"], "low")
        self.assertEqual(json.loads(json.dumps(result))["queries"], evidence["queries"])
        instructions = self.client.responses.create.call_args.kwargs["instructions"]
        self.assertIn("conexões estruturais", instructions)
        self.assertIn("connector", instructions)

    def test_missing_api_key_is_checked_only_when_sources_exist(self) -> None:
        self.settings = Settings(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="not-a-real-password",
            openai_api_key=None,
        )
        self.settings_from_env.return_value = self.settings

        with self.assertRaisesRegex(RetrievalError, "OPENAI_API_KEY"):
            answer_question("O que foi avaliado?")

        self.openai_client.assert_not_called()
        self.driver.close.assert_called_once_with()

    def test_rejects_empty_unknown_and_malformed_citations(self) -> None:
        for answer in (
            "",
            "A resposta é GraphRAG.",
            "A resposta usa [S2].",
            "Uma fonte [S1]. Outra afirmação [S1, S999].",
            "Uma fonte [S1]. Outra afirmação [ S999 ].",
        ):
            with self.subTest(answer=answer):
                fake_client = FakeOpenAI(answer)
                self.openai_client.return_value = fake_client
                with self.assertRaises(RetrievalError):
                    answer_question("O que foi avaliado?")

    def test_rejects_incomplete_or_failed_provider_status_even_with_citation(
        self,
    ) -> None:
        for status in (
            "incomplete",
            "failed",
            "queued",
            "in_progress",
            "cancelled",
            None,
        ):
            with self.subTest(status=status):
                fake_client = FakeOpenAI(
                    "Texto parcial com citação válida [S1].",
                    status=status,
                )
                self.openai_client.return_value = fake_client
                with self.assertRaisesRegex(RetrievalError, "não concluiu"):
                    answer_question("O que foi avaliado?")

    def test_abstains_with_sources_without_citations_and_preserves_evidence(
        self,
    ) -> None:
        self.openai_client.return_value = FakeOpenAI(
            "Não há informações suficientes.",
            abstained=True,
        )
        result = answer_question("Qual receita de bolo é recomendada?")
        self.assertIn("Não encontrei evidências suficientes", result["answer"])
        self.assertEqual(result["sources"], SOURCES)
        self.assertEqual(result["subgraph"], EVIDENCE["subgraph"])
        self.assertEqual(result["queries"], EVIDENCE["queries"])
        self.assertEqual(result["confidence"]["level"], "low")
        self.assertEqual(len(result["warnings"]), len(EVIDENCE["warnings"]) + 1)
        self.assertEqual(
            EVIDENCE["warnings"], ["Resultados limitados às fontes recuperadas."]
        )

    def test_abstention_does_not_publish_uncited_model_claims(self) -> None:
        self.openai_client.return_value = FakeOpenAI(
            "Afirmação sem base.", abstained=True
        )
        result = answer_question("Pergunta?")
        self.assertNotIn("Afirmação sem base", result["answer"])

    def test_rejects_invalid_json_and_malformed_generation_objects(self) -> None:
        values = [
            "not JSON",
            "[]",
            "null",
            "{}",
            json.dumps({"answer": "x", "abstained": "false"}),
            json.dumps({"answer": "x", "abstained": 1}),
            json.dumps({"answer": None, "abstained": True}),
        ]
        for raw in values:
            with self.subTest(raw=raw):
                self.openai_client.return_value = FakeOpenAI(raw, raw=True)
                with self.assertRaises(RetrievalError):
                    answer_question("Pergunta?")

    def test_fenced_json_is_accepted(self) -> None:
        valid = {"abstained": False, "claims": [{"kind": "document", "text": "Resposta.",
            "source_ids": ["S1"], "path_ids": [], "quotes": [{"source_id": "S1",
            "field": "abstract", "quote": SOURCES[0]["abstract"]}]}]}
        self.openai_client.return_value = FakeOpenAI("```json\n" + json.dumps(valid) + "\n```", raw=True)
        self.assertEqual(answer_question("Pergunta?")["answer"], "Resposta. [S1]")

    def test_neo4j_authentication_and_connectivity_errors_are_safe(self) -> None:
        from neo4j.exceptions import AuthError, ServiceUnavailable

        for error, message in (
            (AuthError("secret"), "Autenticação no Neo4j"),
            (ServiceUnavailable("secret"), "conectar ao Neo4j"),
        ):
            with self.subTest(error=type(error).__name__):
                self.retrieve.side_effect = error
                with self.assertRaisesRegex(RetrievalError, message) as caught:
                    answer_question("Pergunta?")
                self.assertNotIn("secret", str(caught.exception))

    def test_cli_prints_only_sanitized_error_and_returns_failure(self) -> None:
        try:
            raise RetrievalError("Falha ao gerar resposta segura.") from RuntimeError(
                "api-key=secret-value"
            )
        except RetrievalError as error:
            error_to_raise = error

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("retrieval.__main__.answer_question", side_effect=error_to_raise):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = cli_main(["pergunta", "--top-k", "6"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "Falha ao gerar resposta segura.\n")
        self.assertNotIn("secret-value", stderr.getvalue())

    def test_always_closes_driver_when_retrieval_fails_and_sanitizes_error(
        self,
    ) -> None:
        self.retrieve.side_effect = RuntimeError(
            "bolt://neo4j:not-a-real-password@internal-host"
        )

        with self.assertRaises(RetrievalError) as caught:
            answer_question("O que foi avaliado?")

        self.assertEqual(
            str(caught.exception), "Falha ao recuperar evidências do Neo4j."
        )
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertNotIn("not-a-real-password", str(caught.exception))
        self.driver.close.assert_called_once_with()
        self.openai_client.assert_not_called()

    def test_sanitizes_provider_errors_but_keeps_internal_cause(self) -> None:
        fake_client = FakeOpenAI()
        fake_client.responses.create.side_effect = RuntimeError("api-key=secret-value")
        self.openai_client.return_value = fake_client

        with self.assertRaises(RetrievalError) as caught:
            answer_question("O que foi avaliado?")

        self.assertEqual(
            str(caught.exception),
            "Falha ao gerar a resposta com o provedor de linguagem.",
        )
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertNotIn("secret-value", str(caught.exception))
        self.driver.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
