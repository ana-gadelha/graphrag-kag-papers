"""Configuração isolada e protocolos reais do SDK com transporte HTTP em memória."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

from retrieval.answer import _generate_answer, answer_question
from retrieval.config import LLMSettings, OPENAI_BASE_URL, RetrievalError, Settings


ENV = {
    "NEO4J_URI": "bolt://localhost:7687", "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "test-password", "OPENAI_MODEL": "cloud-model",
    "OPENAI_API_KEY": "cloud-test-key",
}
EVIDENCE = {
    "sources": [{"citation_id": "S1", "paper_id": "test-paper",
                 "title": "Paper sintético", "abstract": "Abstract sintético."}],
    "subgraph": {"nodes": [], "edges": []}, "warnings": [],
}
TEXT = "O paper contém um abstract sintético [S1]."


def chat_payload(text=TEXT, finish_reason="stop"):
    return {"id": "chatcmpl-test", "object": "chat.completion", "created": 0,
            "model": "local-model", "choices": [
                {"index": 0, "finish_reason": finish_reason,
                 "message": {"role": "assistant", "content": text}},
            ]}


def responses_payload(text=TEXT, status="completed"):
    return {"id": "resp_test", "object": "response", "created_at": 0,
            "model": "local-model", "status": status,
            "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                        "status": "completed", "content": [
                            {"type": "output_text", "text": text, "annotations": []},
                        ]}]}


@contextmanager
def environment(**overrides):
    with patch.dict(os.environ, {**ENV, **overrides}, clear=True), patch(
        "dotenv.load_dotenv", return_value=False,
    ):
        yield


@contextmanager
def sdk_transport(payload, *, status_code=200):
    """Usa o SDK instalado, sem acessar rede ou substituir seus endpoints."""
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status_code, json=payload)

    def make_client(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))

    with patch("openai.OpenAI", side_effect=make_client) as constructor:
        yield requests, constructor


class LLMConfigurationTests(unittest.TestCase):
    def test_legacy_cloud_defaults_ignore_ambient_base_url(self):
        with environment(OPENAI_BASE_URL="http://unrelated-service/v1"):
            settings = Settings.from_env()
            self.assertEqual(settings.generation.base_url, OPENAI_BASE_URL)
            self.assertEqual(settings.generation.api, "responses")
            self.assertEqual(settings.generation.model, "cloud-model")
            self.assertEqual(settings.generation.api_key, "cloud-test-key")
            self.assertEqual(settings.generation.timeout, 30)
            self.assertEqual(os.environ["OPENAI_BASE_URL"], "http://unrelated-service/v1")

    def test_custom_url_defaults_do_not_reuse_cloud_key_or_change_extraction_settings(self):
        with environment(RETRIEVAL_LLM_BASE_URL="http://localhost:1234/v1/",
                         RETRIEVAL_LLM_MODEL=" local-model "):
            original_env = dict(os.environ)
            settings = Settings.from_env()
            self.assertEqual(settings.generation.base_url, "http://localhost:1234/v1")
            self.assertEqual(settings.generation.api, "chat_completions")
            self.assertEqual(settings.generation.timeout, 120)
            self.assertEqual(settings.generation.api_key, "local")
            self.assertEqual(settings.generation.model, "local-model")
            self.assertEqual(settings.openai_model, "cloud-model")
            self.assertEqual(settings.openai_api_key, "cloud-test-key")
            self.assertEqual(dict(os.environ), original_env)

    def test_explicit_scoped_overrides_and_responses(self):
        with environment(RETRIEVAL_LLM_BASE_URL="http://localhost:11434/v1",
                         RETRIEVAL_LLM_MODEL="local-model", RETRIEVAL_LLM_API="responses",
                         RETRIEVAL_LLM_API_KEY="private-local-key", RETRIEVAL_LLM_TIMEOUT="240.5"):
            settings = Settings.from_env()
        self.assertEqual(settings.generation.api, "responses")
        self.assertEqual(settings.generation.api_key, "private-local-key")
        self.assertEqual(settings.generation.timeout, 240.5)
        self.assertNotIn("private-local-key", repr(settings.generation))

    def test_scoped_model_and_key_can_override_cloud_defaults(self):
        with environment(RETRIEVAL_LLM_MODEL="scoped-model", RETRIEVAL_LLM_API_KEY="scoped-key"):
            settings = Settings.from_env()
        self.assertEqual(settings.generation.model, "scoped-model")
        self.assertEqual(settings.generation.api_key, "scoped-key")
        self.assertEqual(settings.generation.base_url, OPENAI_BASE_URL)

    def test_invalid_settings_are_actionable_and_do_not_echo_values(self):
        bad_configs = [
            {"RETRIEVAL_LLM_BASE_URL": "http://localhost:1234/v1"},
            {"RETRIEVAL_LLM_API": "unknown"},
            *[{"RETRIEVAL_LLM_TIMEOUT": value} for value in ("zero", "0", "-1", "nan", "inf")],
            *[{"RETRIEVAL_LLM_BASE_URL": value, "RETRIEVAL_LLM_MODEL": "local-model"}
              for value in ("localhost:1234/v1", "file:///tmp/v1", "http://user:secret@localhost/v1",
                            "http://localhost:bad/v1", "http://localhost:99999/v1",
                            "http://localhost/v1?key=secret", "http://localhost/v1#secret")],
        ]
        for config in bad_configs:
            with self.subTest(keys=list(config)), environment(**config):
                with self.assertRaises(RetrievalError) as caught:
                    Settings.from_env().generation
                self.assertIn("RETRIEVAL_LLM_", str(caught.exception))
                self.assertNotIn("secret", str(caught.exception))


class LocalLLMProtocolTests(unittest.TestCase):
    def settings(self, *, api="chat_completions"):
        return Settings(
            uri="bolt://localhost:7687", username="neo4j", password="test-password",
            openai_api_key="cloud-test-key",
            llm=LLMSettings(model="local-model", base_url="http://localhost:1234/v1",
                            api_key="local", api=api, timeout=120),
        )

    def test_chat_request_and_text_via_real_sdk(self):
        with sdk_transport(chat_payload()) as (requests, constructor):
            answer = _generate_answer(self.settings(), "Pergunta de teste?", EVIDENCE)
        self.assertEqual(answer, TEXT)
        self.assertEqual(len(requests), 1)
        self.assertEqual(str(requests[0].url), "http://localhost:1234/v1/chat/completions")
        self.assertEqual(requests[0].headers["authorization"], "Bearer local")
        body = json.loads(requests[0].content)
        self.assertEqual(body["model"], "local-model")
        self.assertEqual(body["max_tokens"], 2000)
        self.assertNotIn("store", body)
        self.assertNotIn("max_output_tokens", body)
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertIn("não confiável", body["messages"][0]["content"])
        self.assertEqual(body["messages"][1]["role"], "user")
        context = json.loads(body["messages"][1]["content"])
        self.assertEqual(context, {"question": "Pergunta de teste?", **EVIDENCE})
        self.assertEqual(constructor.call_args.kwargs["timeout"], 120)

    def test_local_responses_request_via_real_sdk(self):
        with sdk_transport(responses_payload()) as (requests, _):
            answer = _generate_answer(self.settings(api="responses"), "Pergunta?", EVIDENCE)
        self.assertEqual(answer, TEXT)
        self.assertEqual(requests[0].url.path, "/v1/responses")
        body = json.loads(requests[0].content)
        self.assertEqual(body["model"], "local-model")
        self.assertFalse(body["store"])
        self.assertEqual(body["max_output_tokens"], 2000)
        self.assertIn("instructions", body)

    def test_chat_rejects_truncation_blocking_and_unexpected_choices(self):
        payloads = [chat_payload(finish_reason=reason)
                    for reason in ("length", "content_filter", "tool_calls", None)]
        payloads.append({**chat_payload(), "choices": []})
        for payload in payloads:
            with self.subTest(payload=payload), sdk_transport(payload):
                with self.assertRaisesRegex(RetrievalError, "não concluiu"):
                    _generate_answer(self.settings(), "Pergunta?", EVIDENCE)

    def test_both_protocols_reject_invalid_citations_and_empty_answers(self):
        for api, factory in (("chat_completions", chat_payload), ("responses", responses_payload)):
            for text in ("", "Resposta sem citações.", "Fonte inventada [S999].", "Texto [S1] e [S1, S999]."):
                with self.subTest(api=api, text=text), sdk_transport(factory(text)):
                    with self.assertRaises(RetrievalError):
                        _generate_answer(self.settings(api=api), "Pergunta?", EVIDENCE)

    def test_http_error_does_not_fallback_or_expose_server_details(self):
        payload = {"error": {"message": "private-server-details", "type": "not_found"}}
        with sdk_transport(payload, status_code=404) as (requests, _):
            with self.assertRaises(RetrievalError) as caught:
                _generate_answer(self.settings(), "Pergunta?", EVIDENCE)
        self.assertNotIn("private-server-details", str(caught.exception))
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/v1/chat/completions")

    def test_public_contract_is_preserved_with_local_generation(self):
        driver = Mock()
        with patch("retrieval.answer.Settings.from_env", return_value=self.settings()), patch(
            "retrieval.answer._new_driver", return_value=driver,
        ), patch("retrieval.answer._load_encoder", return_value=object()), patch(
            "retrieval.answer._retrieve", return_value=EVIDENCE,
        ), sdk_transport(chat_payload()):
            result = answer_question("  Pergunta?  ")
        self.assertEqual(result, {"question": "Pergunta?", "answer": TEXT, **EVIDENCE})
        driver.close.assert_called_once_with()

    def test_no_evidence_does_not_require_a_valid_local_llm_configuration(self):
        driver = Mock()
        evidence = {"sources": [], "subgraph": {"nodes": [], "edges": []}, "warnings": []}
        with environment(RETRIEVAL_LLM_BASE_URL="http://localhost:1234/v1"), patch(
            "retrieval.answer._new_driver", return_value=driver,
        ), patch("retrieval.answer._load_encoder", return_value=object()), patch(
            "retrieval.answer._retrieve", return_value=evidence,
        ), patch("retrieval.answer._openai_client") as client:
            result = answer_question("Pergunta sem evidências?")
        self.assertEqual(result["sources"], [])
        self.assertIn("Não encontrei evidências", result["answer"])
        client.assert_not_called()

    def test_embedding_configuration_does_not_require_valid_local_llm_configuration(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from retrieval.index_embeddings import main

        driver = Mock()
        report = {"papers": 0, "embedded": 0, "skipped": 0}
        with environment(RETRIEVAL_LLM_BASE_URL="http://localhost:1234/v1"), patch(
            "retrieval.index_embeddings.create_driver", return_value=driver,
        ), patch("retrieval.index_embeddings.index_papers", return_value=report), redirect_stdout(StringIO()):
            self.assertEqual(main([]), 0)
        driver.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
