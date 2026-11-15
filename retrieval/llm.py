"""OpenAI-compatible JSON transport, stage isolation and resumable free-tier calls.

Only application-validated JSON is replayed. The local ledger accounts for failed
attempts too; it is an operational budget, not a copy of the provider's quota.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import LLMSettings, RetrievalError


class LLMError(RetrievalError):
    """Safe public message plus machine-readable failure classification."""

    def __init__(self, message: str, kind: str, *, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


def stage_settings(stage: str) -> LLMSettings:
    """Resolve stage overrides, inheriting only explicitly configured retrieval.

    A new endpoint never inherits another endpoint's credential. Implicit paid
    OpenAI defaults are intentionally unavailable to extraction and planning.
    """
    if stage not in {"extraction", "cypher", "retrieval"}:
        raise ValueError("Etapa de linguagem desconhecida.")
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    prefix = {"extraction": "EXTRACTION", "cypher": "CYPHER", "retrieval": "RETRIEVAL"}[
        stage
    ] + "_LLM_"
    names = (
        "MODEL",
        "BASE_URL",
        "API_KEY",
        "API",
        "TIMEOUT",
        "MAX_OUTPUT_TOKENS",
        "MAX_RETRIES",
    )
    resolved = {
        f"RETRIEVAL_LLM_{n}": os.environ.get(f"RETRIEVAL_LLM_{n}", "") for n in names
    }
    own_url = os.environ.get(prefix + "BASE_URL", "").strip()
    if own_url:
        # Inherit neither a protocol nor a credential across endpoints.
        resolved["RETRIEVAL_LLM_API_KEY"] = ""
        resolved["RETRIEVAL_LLM_API"] = ""
    for name in names:
        if prefix + name in os.environ:
            if name == "BASE_URL" and not os.environ[prefix + name].strip():
                # An empty optional override means inherit, not switch to the
                # SDK's OpenAI endpoint while retaining another service's key.
                continue
            resolved["RETRIEVAL_LLM_" + name] = os.environ[prefix + name]
    model = resolved["RETRIEVAL_LLM_MODEL"].strip()
    if not model:
        raise LLMError(
            f"Defina {prefix}MODEL ou RETRIEVAL_LLM_MODEL explicitamente.",
            "configuration_error",
        )
    try:
        return LLMSettings.from_environment(
            resolved, default_model=model, default_key=None
        )
    except RetrievalError as exc:
        raise LLMError(
            str(exc).replace("RETRIEVAL_LLM_", prefix), "configuration_error"
        ) from None


def openai_client(settings: LLMSettings) -> Any:
    from openai import OpenAI

    # Retries are performed below so each HTTP attempt consumes the local budget.
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout,
        max_retries=0,
    )


def provider_error(exc: Exception) -> LLMError:
    from openai import APIConnectionError, APITimeoutError

    if isinstance(exc, APITimeoutError):
        return LLMError(
            "O provedor de linguagem excedeu o tempo limite; tente novamente.",
            "timeout",
        )
    if isinstance(exc, APIConnectionError):
        return LLMError(
            "Não foi possível conectar ao provedor de linguagem; confira o servidor e a rede.",
            "connection_error",
        )
    status = getattr(exc, "status_code", None)
    messages = {
        401: (
            "Autenticação no provedor de linguagem falhou; confira a chave configurada.",
            "authentication_error",
        ),
        403: (
            "O provedor de linguagem negou acesso; confira as permissões do modelo.",
            "access_denied",
        ),
        404: (
            "Modelo ou endpoint de linguagem indisponível; confira a configuração do provedor.",
            "model_unavailable",
        ),
        400: (
            "O provedor rejeitou a requisição; confira modelo, parâmetros e tamanho do contexto.",
            "invalid_request",
        ),
        413: ("O contexto excedeu o limite do provedor de linguagem.", "context_limit"),
    }
    if status == 429:
        exhausted = getattr(exc, "code", None) in {
            "insufficient_quota",
            "credit_balance_exhausted",
            "billing_hard_limit_reached",
            "organization_usage_limit_exceeded",
        }
        message = (
            "O provedor de linguagem está sem saldo ou quota; confira o faturamento da conta."
            if exhausted
            else "O provedor de linguagem atingiu um limite de uso; confira a quota e tente novamente."
        )
        return LLMError(message, "quota", status_code=429)
    if isinstance(status, int) and status >= 500:
        return LLMError(
            "O provedor de linguagem está temporariamente indisponível; tente novamente.",
            "provider_error",
            status_code=status,
        )
    message, kind = messages.get(
        status,
        ("Falha ao gerar a resposta com o provedor de linguagem.", "provider_error"),
    )
    return LLMError(message, kind, status_code=status)


def parse_json(raw: Any) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        raise LLMError(
            "O provedor de linguagem retornou uma resposta vazia.", "empty_response"
        )
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", raw.strip(), re.I | re.S)
    try:
        data = json.loads(fenced.group(1) if fenced else raw)
    except (ValueError, TypeError):
        raise LLMError(
            "O provedor de linguagem retornou JSON inválido.", "invalid_json"
        ) from None
    if not isinstance(data, dict):
        raise LLMError("O provedor deve retornar um objeto JSON.", "invalid_json")
    return data


def _runtime_directory() -> Path | None:
    if os.getenv("LLM_CACHE_ENABLED", "1") == "0":
        return None
    return Path(os.getenv("LLM_CACHE_DIR", "output/llm_cache"))


def _positive_env(name: str, default: int) -> int:
    try:
        number = int(os.getenv(name, str(default)))
        if number <= 0:
            raise ValueError
        return number
    except ValueError:
        raise LLMError(
            f"{name} deve ser um inteiro positivo.", "configuration_error"
        ) from None


@contextmanager
def _ledger(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(directory / "requests.sqlite3", timeout=10)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS attempts (at REAL, namespace TEXT, model TEXT, digest TEXT, status TEXT, metadata TEXT)"
    )
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _reserve(
    directory: Path | None, namespace: str, model: str, digest: str
) -> int | None:
    if directory is None:
        return None
    daily = _positive_env("LLM_DAILY_REQUEST_BUDGET", 40)
    per_minute = _positive_env("LLM_REQUESTS_PER_MINUTE", 10)
    now = time.time()
    midnight = (
        datetime.now(timezone.utc)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .timestamp()
    )
    with _ledger(directory) as db:
        db.execute("BEGIN IMMEDIATE")
        if (
            db.execute(
                "SELECT count(*) FROM attempts WHERE at>=?", (midnight,)
            ).fetchone()[0]
            >= daily
        ):
            raise LLMError(
                "O orçamento local diário de chamadas foi atingido; retome após a renovação.",
                "local_budget",
            )
        if (
            db.execute(
                "SELECT count(*) FROM attempts WHERE at>?", (now - 60,)
            ).fetchone()[0]
            >= per_minute
        ):
            raise LLMError(
                "O limite local de chamadas por minuto foi atingido; retome em instantes.",
                "local_rate_limit",
            )
        cursor = db.execute(
            "INSERT INTO attempts VALUES (?, ?, ?, ?, 'started', '{}')",
            (now, namespace, model, digest),
        )
        return cursor.lastrowid


def _record(
    directory: Path | None, row_id: int | None, status: str, metadata: dict
) -> None:
    if directory is not None and row_id is not None:
        with _ledger(directory) as db:
            db.execute(
                "UPDATE attempts SET status=?,metadata=? WHERE rowid=?",
                (status, json.dumps(metadata), row_id),
            )


def _save(path: Path, payload: dict) -> None:
    # Replace atomically; interrupted writes must never become valid cache entries.
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False)
    temporary.replace(path)


class GeneratedJSON(dict):
    """JSON plus private receipt; callers explicitly accept validated responses."""

    def __init__(self, data: dict, *, path: Path | None, envelope: dict):
        super().__init__(data)
        self._path = path
        self._envelope = envelope
        self.receipt = envelope["receipt"]


def accept_json(payload: dict) -> None:
    """Mark reusable only after the caller's schema/evidence validation passed."""
    if isinstance(payload, GeneratedJSON) and payload._path is not None:
        payload._envelope["validated"] = True
        _save(payload._path, payload._envelope)


def _response_metadata(metadata: dict, response: Any) -> None:
    """Keep usage even when the provider stops at its output-token limit."""
    for attribute, key in (("model", "effective_model"), ("id", "response_id")):
        value = getattr(response, attribute, None)
        if isinstance(value, str):
            metadata[key] = value
    usage = getattr(response, "usage", None)
    if usage is not None and hasattr(usage, "model_dump"):
        metadata["usage"] = usage.model_dump()


def generate_json(
    settings: LLMSettings,
    system: str,
    payload: dict | str,
    *,
    namespace: str,
    version: str,
    client_factory: Callable | None = None,
) -> dict:
    """Generate strict JSON, with no provider fallback and bounded explicit retries."""
    if not settings.api_key:
        raise LLMError(
            "Defina RETRIEVAL_LLM_API_KEY ou OPENAI_API_KEY para a geração.",
            "configuration_error",
        )
    public_settings = asdict(settings)
    # Different accounts do not share results, but no credential is written to disk.
    public_settings["api_key"] = hashlib.sha256(settings.api_key.encode()).hexdigest()
    content = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if isinstance(payload, dict)
        else payload
    )
    digest = hashlib.sha256(
        json.dumps(
            [namespace, version, public_settings, system, content], sort_keys=True
        ).encode()
    ).hexdigest()
    directory = _runtime_directory()
    path = directory / (digest + ".json") if directory is not None else None
    if path is not None and path.is_file():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("validated") is True and isinstance(saved.get("data"), dict):
                saved["receipt"]["cache_hit"] = True
                return GeneratedJSON(saved["data"], path=path, envelope=saved)
        except (ValueError, KeyError, OSError):
            pass
    factory = client_factory or openai_client
    for attempt in range(settings.max_retries + 1):
        row_id = _reserve(directory, namespace, settings.model, digest)
        started = time.monotonic()
        metadata: dict = {
            "model": settings.model,
            "cache_hit": False,
            "attempt": attempt + 1,
            "max_output_tokens": settings.max_output_tokens,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        raw = None
        try:
            with factory(settings) as client:
                if settings.api == "chat_completions":
                    response = client.chat.completions.create(
                        model=settings.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": content},
                        ],
                        max_tokens=settings.max_output_tokens,
                    )
                    _response_metadata(metadata, response)
                    choice = response.choices[0] if response.choices else None
                    raw = getattr(getattr(choice, "message", None), "content", None)
                    finish = getattr(choice, "finish_reason", None)
                    metadata["finish_reason"] = finish
                    if finish != "stop":
                        kind = (
                            "truncation"
                            if finish == "length"
                            else "refusal"
                            if finish == "content_filter"
                            else "unexpected_finish"
                        )
                        raise LLMError(
                            "O provedor de linguagem não concluiu a resposta (truncamento, bloqueio ou retorno inesperado).",
                            kind,
                        )
                    if getattr(choice.message, "refusal", None):
                        raise LLMError(
                            "O provedor de linguagem bloqueou a resposta.", "refusal"
                        )
                else:
                    response = client.responses.create(
                        model=settings.model,
                        instructions=system,
                        input=content,
                        store=False,
                        max_output_tokens=settings.max_output_tokens,
                    )
                    _response_metadata(metadata, response)
                    raw = getattr(response, "output_text", None)
                    metadata["finish_reason"] = getattr(response, "status", None)
                    if getattr(response, "status", None) != "completed":
                        raise LLMError(
                            "O provedor de linguagem não concluiu a resposta.",
                            "truncation",
                        )
                    if any(
                        getattr(part, "type", None) == "refusal"
                        for item in (getattr(response, "output", []) or [])
                        for part in (getattr(item, "content", []) or [])
                    ):
                        raise LLMError(
                            "O provedor de linguagem bloqueou a resposta.", "refusal"
                        )
                data = parse_json(raw)
            metadata["seconds"] = round(time.monotonic() - started, 3)
            envelope = {
                "validated": False,
                "data": data,
                "receipt": metadata,
                "request_hash": digest,
                "namespace": namespace,
                "version": version,
            }
            if path is not None:
                _save(path, envelope)
            _record(directory, row_id, "json_unvalidated", metadata)
            return GeneratedJSON(data, path=path, envelope=envelope)
        except Exception as exc:
            error = exc if isinstance(exc, LLMError) else provider_error(exc)
            metadata.update(
                kind=error.kind, seconds=round(time.monotonic() - started, 3)
            )
            _record(directory, row_id, error.kind, metadata)
            if path is not None:
                _save(
                    path.with_suffix(".failed.json"),
                    {"validated": False, "receipt": metadata, "raw": raw},
                )
            if attempt < settings.max_retries and error.kind in {
                "connection_error",
                "timeout",
                "provider_error",
            }:
                time.sleep(min(2**attempt, 4))
                continue
            if error is exc:
                raise
            raise error from exc
    raise AssertionError("Unreachable retry state")
