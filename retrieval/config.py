"""Configuration shared by the graph retrieval commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import os
from typing import Any
from urllib.parse import urlsplit


class RetrievalError(RuntimeError):
    """A configuration or retrieval operation failed with a safe message."""


OPENAI_BASE_URL = "https://api.openai.com/v1"


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Configuração de geração exclusiva de retrieval, já resolvida."""

    model: str = "gpt-4.1-mini"
    api_key: str | None = field(default=None, repr=False)
    base_url: str = field(default=OPENAI_BASE_URL, repr=False)
    api: str = "responses"
    timeout: float = 30.0

    def __post_init__(self) -> None:
        if self.api not in {"responses", "chat_completions"}:
            raise RetrievalError(
                "RETRIEVAL_LLM_API deve ser responses ou chat_completions."
            )
        if not self.model.strip():
            raise RetrievalError("Defina RETRIEVAL_LLM_MODEL com o modelo servido.")
        try:
            parsed = urlsplit(self.base_url)
            valid_url = (
                parsed.scheme in {"http", "https"} and parsed.hostname
                and not parsed.username and not parsed.password
                and not parsed.query and not parsed.fragment
            )
            parsed.port  # Valida também a sintaxe e o intervalo da porta.
        except ValueError:
            valid_url = False
        if not valid_url:
            raise RetrievalError(
                "RETRIEVAL_LLM_BASE_URL deve ser uma URL HTTP(S) sem credenciais, "
                "query ou fragmento, incluindo o prefixo da API (ex.: /v1)."
            )
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise RetrievalError("RETRIEVAL_LLM_TIMEOUT deve ser um número positivo e finito.")

    @classmethod
    def from_environment(
        cls, environment: dict[str, str], *, default_model: str, default_key: str | None,
    ) -> "LLMSettings":
        def value(name: str) -> str:
            return (environment.get(name) or "").strip()

        base_url = value("RETRIEVAL_LLM_BASE_URL").rstrip("/")
        model = value("RETRIEVAL_LLM_MODEL")
        if base_url and not model:
            raise RetrievalError(
                "Defina RETRIEVAL_LLM_MODEL com o identificador do modelo no servidor."
            )
        try:
            timeout = float(value("RETRIEVAL_LLM_TIMEOUT") or ("120" if base_url else "30"))
        except ValueError:
            raise RetrievalError(
                "RETRIEVAL_LLM_TIMEOUT deve ser um número positivo e finito."
            ) from None
        # Um endereço alternativo nunca herda a chave destinada à OpenAI.
        api_key = value("RETRIEVAL_LLM_API_KEY") or ("local" if base_url else default_key)
        return cls(
            model=model or default_model,
            api_key=api_key,
            base_url=base_url or OPENAI_BASE_URL,
            api=value("RETRIEVAL_LLM_API") or ("chat_completions" if base_url else "responses"),
            timeout=timeout,
        )


@dataclass(frozen=True, slots=True)
class Settings:
    uri: str = field(repr=False)
    username: str
    password: str = field(repr=False)
    database: str = "neo4j"
    openai_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = field(default=None, repr=False)
    llm: LLMSettings | None = field(default=None, repr=False)
    llm_environment: dict[str, str] | None = field(default=None, repr=False)

    @property
    def generation(self) -> LLMSettings:
        """Preserva os construtores existentes e os padrões da OpenAI."""
        if self.llm is not None:
            return self.llm
        if self.llm_environment is not None:
            return LLMSettings.from_environment(
                self.llm_environment, default_model=self.openai_model,
                default_key=self.openai_api_key,
            )
        return LLMSettings(model=self.openai_model, api_key=self.openai_api_key)

    @classmethod
    def from_env(cls) -> "Settings":
        """Load the repository's .env without replacing existing environment values."""
        try:
            from dotenv import load_dotenv
        except ImportError:
            raise RetrievalError(
                "A dependência python-dotenv é necessária para carregar a configuração."
            ) from None

        repository_root = Path(__file__).resolve().parents[1]
        load_dotenv(dotenv_path=repository_root / ".env", override=False)

        required_names = {
            "uri": "NEO4J_URI",
            "username": "NEO4J_USERNAME",
            "password": "NEO4J_PASSWORD",
        }
        missing = [
            env_name
            for env_name in required_names.values()
            if not (os.environ.get(env_name) or "").strip()
        ]
        if missing:
            joined = ", ".join(missing)
            raise RetrievalError(
                f"Configuração Neo4j incompleta; defina: {joined}."
            )

        default_model = _env("OPENAI_MODEL") or "gpt-4.1-mini"
        default_key = _env("OPENAI_API_KEY") or None
        return cls(
            uri=os.environ[required_names["uri"]],
            username=os.environ[required_names["username"]],
            password=os.environ[required_names["password"]],
            database=os.environ.get("NEO4J_DATABASE", "neo4j") or "neo4j",
            openai_model=default_model,
            openai_api_key=default_key,
            # Captura a configuração sem exigir um LLM para indexar ou abster-se.
            llm_environment={name: value for name, value in os.environ.items()
                             if name.startswith("RETRIEVAL_LLM_")},
        )


def create_driver(settings: Settings) -> Any:
    """Create a Neo4j driver without opening a connection."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise RetrievalError(
            "A dependência neo4j é necessária para acessar o grafo."
        ) from None

    try:
        return GraphDatabase.driver(
            settings.uri,
            auth=(settings.username, settings.password),
        )
    except Exception:
        # Driver exceptions can include connection details. Keep the public error safe.
        raise RetrievalError("Não foi possível inicializar o driver Neo4j.") from None
