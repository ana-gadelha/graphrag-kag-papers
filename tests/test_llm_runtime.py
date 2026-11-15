from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from retrieval.config import LLMSettings, RetrievalError
from retrieval.llm import LLMError, accept_json, generate_json, stage_settings


def client_for(data):
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=json.dumps(data), refusal=None),
            )
        ],
        model="served-free-model",
    )
    return client


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(
            os.environ,
            {
                "LLM_CACHE_DIR": self.tmp.name,
                "LLM_DAILY_REQUEST_BUDGET": "40",
                "LLM_REQUESTS_PER_MINUTE": "10",
            },
            clear=True,
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.settings = LLMSettings(
            model="free-model", api_key="test-secret", api="chat_completions"
        )

    def test_only_validated_cache_is_replayed_and_keys_are_not_persisted(self):
        factory = Mock(return_value=client_for({"value": 1}))

        def call():
            return generate_json(
                self.settings,
                "system",
                {"x": "text"},
                namespace="test",
                version="v1",
                client_factory=factory,
            )

        first = call()
        second = call()
        self.assertEqual(factory.call_count, 2)
        accept_json(second)
        replay = call()
        self.assertEqual(factory.call_count, 2)
        self.assertTrue(replay.receipt["cache_hit"])
        self.assertEqual(replay["value"], 1)
        serialized = "".join(p.read_text() for p in Path(self.tmp.name).glob("*.json"))
        self.assertNotIn("test-secret", serialized)
        self.assertEqual(first.receipt["effective_model"], "served-free-model")

    def test_budget_counts_failed_attempts_and_retry_is_explicit(self):
        os.environ["LLM_DAILY_REQUEST_BUDGET"] = "1"
        factory = Mock(side_effect=RuntimeError("private-message"))
        with self.assertRaises(LLMError):
            generate_json(
                self.settings,
                "sys",
                {},
                namespace="t",
                version="1",
                client_factory=factory,
            )
        with self.assertRaisesRegex(LLMError, "orçamento local"):
            generate_json(
                self.settings,
                "sys",
                {},
                namespace="t",
                version="1",
                client_factory=factory,
            )
        self.assertEqual(factory.call_count, 1)

    def test_rate_limit_and_invalid_budget_are_safe(self):
        os.environ["LLM_REQUESTS_PER_MINUTE"] = "1"
        factory = Mock(return_value=client_for({}))
        generate_json(
            self.settings, "sys", {}, namespace="t", version="1", client_factory=factory
        )
        with self.assertRaisesRegex(LLMError, "por minuto"):
            generate_json(
                self.settings,
                "sys",
                {},
                namespace="t",
                version="2",
                client_factory=factory,
            )
        os.environ["LLM_REQUESTS_PER_MINUTE"] = "secret"
        with self.assertRaisesRegex(LLMError, "inteiro positivo"):
            generate_json(
                self.settings,
                "sys",
                {},
                namespace="t",
                version="3",
                client_factory=factory,
            )

    def test_stage_scopes_are_explicit_and_do_not_cross_credentials(self):
        with patch("dotenv.load_dotenv", return_value=False):
            with self.assertRaisesRegex(LLMError, "explicitamente"):
                stage_settings("cypher")
            os.environ.update(
                RETRIEVAL_LLM_BASE_URL="https://free.example/v1",
                RETRIEVAL_LLM_MODEL="free",
                RETRIEVAL_LLM_API_KEY="retrieval-secret",
            )
            self.assertEqual(stage_settings("extraction").api_key, "retrieval-secret")
            os.environ.update(
                EXTRACTION_LLM_BASE_URL="http://localhost:1234/v1",
                EXTRACTION_LLM_MODEL="other",
            )
            settings = stage_settings("extraction")
            self.assertEqual(settings.api_key, "local")
            self.assertEqual(settings.model, "other")
            self.assertEqual(stage_settings("cypher").model, "free")
            os.environ["CYPHER_LLM_TIMEOUT"] = "nan"
            with self.assertRaisesRegex(RetrievalError, "CYPHER_LLM_TIMEOUT"):
                stage_settings("cypher")

    def test_empty_stage_url_does_not_redirect_an_inherited_credential(self):
        os.environ.update(
            RETRIEVAL_LLM_BASE_URL="https://free.example/v1",
            RETRIEVAL_LLM_MODEL="free",
            RETRIEVAL_LLM_API_KEY="source-service-key",
        )
        for stage in ("extraction", "cypher"):
            for empty in ("", "   "):
                with (
                    self.subTest(stage=stage, empty=empty),
                    patch.dict(os.environ, {stage.upper() + "_LLM_BASE_URL": empty}),
                    patch("dotenv.load_dotenv", return_value=False),
                ):
                    resolved = stage_settings(stage)
                    self.assertEqual(resolved.base_url, "https://free.example/v1")
                    self.assertEqual(resolved.api_key, "source-service-key")

    def test_truncation_audit_keeps_usage_and_output_limit(self):
        client = client_for({})
        response = client.chat.completions.create.return_value
        response.choices[0].finish_reason = "length"
        response.usage = Mock()
        response.usage.model_dump.return_value = {"completion_tokens": 2000}
        with self.assertRaises(LLMError) as caught:
            generate_json(
                self.settings,
                "sys",
                {},
                namespace="t",
                version="1",
                client_factory=lambda _: client,
            )
        self.assertEqual(caught.exception.kind, "truncation")
        failed = list(Path(self.tmp.name).glob("*.failed.json"))
        self.assertEqual(len(failed), 1)
        receipt = json.loads(failed[0].read_text())["receipt"]
        self.assertEqual(receipt["usage"]["completion_tokens"], 2000)
        self.assertEqual(receipt["max_output_tokens"], 2000)
        self.assertEqual(receipt["finish_reason"], "length")

    def test_cache_version_and_configuration_invalidate(self):
        factory = Mock(return_value=client_for({}))
        first = generate_json(
            self.settings, "sys", {}, namespace="t", version="1", client_factory=factory
        )
        accept_json(first)
        generate_json(
            self.settings, "sys", {}, namespace="t", version="2", client_factory=factory
        )
        self.assertEqual(factory.call_count, 2)


if __name__ == "__main__":
    unittest.main()
