"""Local, deterministic embeddings for paper retrieval."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from typing import Any, Iterable, Sequence

from retrieval.config import RetrievalError


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIMENSIONS = 384


@lru_cache(maxsize=1)
def load_encoder() -> Any:
    """Load and cache the multilingual Sentence Transformers encoder on demand."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RetrievalError(
            "A dependência sentence-transformers é necessária para gerar embeddings."
        ) from None

    try:
        encoder = SentenceTransformer(EMBEDDING_MODEL)
        dimension_getter = getattr(encoder, "get_embedding_dimension", None)
        if not callable(dimension_getter):
            dimension_getter = getattr(
                encoder, "get_sentence_embedding_dimension", None
            )
        if not callable(dimension_getter):
            raise AttributeError("SentenceTransformer has no dimension method")
        dimension = dimension_getter()
    except Exception:
        raise RetrievalError(
            f"Não foi possível carregar o modelo local {EMBEDDING_MODEL}."
        ) from None

    if dimension != DIMENSIONS:
        raise RetrievalError(
            f"O modelo {EMBEDDING_MODEL} retornou dimensão {dimension}; "
            f"eram esperadas {DIMENSIONS}."
        )
    return encoder


def _string_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def embedding_text(paper: dict[str, Any]) -> str:
    """Build the stable text used to embed a paper and its Concept names."""
    title = _string_value(paper.get("title"))
    abstract = _string_value(paper.get("abstract"))
    raw_keywords = paper.get("keywords") or []
    if isinstance(raw_keywords, str):
        raw_keywords = [raw_keywords]

    keywords: set[str] = set()
    if isinstance(raw_keywords, Iterable):
        for raw_keyword in raw_keywords:
            if isinstance(raw_keyword, dict):
                raw_keyword = raw_keyword.get("name")
            keyword = _string_value(raw_keyword)
            if keyword:
                keywords.add(keyword)

    sections: list[str] = []
    if title:
        sections.append(f"Title: {title}")
    if abstract:
        sections.append(f"Abstract: {abstract}")
    if keywords:
        ordered_keywords = sorted(keywords, key=lambda item: (item.casefold(), item))
        sections.append(f"Keywords: {', '.join(ordered_keywords)}")
    return "\n".join(sections)


def content_hash(paper: dict[str, Any]) -> str:
    """Return the SHA-256 hash of exactly the text supplied to the encoder."""
    return hashlib.sha256(embedding_text(paper).encode("utf-8")).hexdigest()


def validate_vector(vector: Any) -> list[float]:
    """Convert a vector to floats after checking dimension and numeric validity."""
    try:
        values = list(vector)
    except (TypeError, ValueError):
        raise RetrievalError("Embedding inválido: era esperado um vetor numérico.") from None

    if len(values) != DIMENSIONS:
        raise RetrievalError(
            f"Embedding inválido: dimensão {len(values)}; eram esperadas {DIMENSIONS}."
        )

    converted: list[float] = []
    for value in values:
        if isinstance(value, (bool, str, bytes, bytearray, memoryview)):
            raise RetrievalError("Embedding inválido: o vetor contém valor não numérico.")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise RetrievalError(
                "Embedding inválido: o vetor contém valor não numérico."
            ) from None
        if not math.isfinite(number):
            raise RetrievalError("Embedding inválido: o vetor contém valor não finito.")
        converted.append(number)

    if not any(value != 0.0 for value in converted):
        raise RetrievalError("Embedding inválido: o vetor não pode ser nulo.")
    return converted


def encode_texts(texts: Sequence[str], *, encoder: Any | None = None) -> list[list[float]]:
    """Encode texts with the shared model and return normalized float vectors."""
    if not texts:
        return []
    active_encoder = encoder if encoder is not None else load_encoder()
    try:
        encoded = active_encoder.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = list(encoded)
    except Exception:
        raise RetrievalError(
            f"Falha ao gerar embeddings com o modelo {EMBEDDING_MODEL}."
        ) from None

    if len(vectors) != len(texts):
        raise RetrievalError(
            "O modelo retornou uma quantidade de embeddings diferente da quantidade de textos."
        )

    normalized: list[list[float]] = []
    for vector in vectors:
        values = validate_vector(vector)
        norm = math.sqrt(sum(value * value for value in values))
        normalized.append([value / norm for value in values])
    return normalized


INDEX_INFO_QUERY = """
SHOW INDEXES YIELD name, state, type, options, labelsOrTypes, properties
WHERE name = $index_name
RETURN name, state, type, options, labelsOrTypes, properties
"""


def _record_value(record: Any, key: str, default: Any = None) -> Any:
    try:
        return record.get(key, default)
    except AttributeError:
        try:
            return record[key]
        except (KeyError, TypeError, IndexError):
            return default


def _single_record(result: Any) -> Any | None:
    single = getattr(result, "single", None)
    if callable(single):
        return single()
    try:
        return next(iter(result), None)
    except TypeError:
        return None


def check_index(session: Any) -> None:
    """Require the configured online paper vector index; never alter the schema."""
    result = session.run(INDEX_INFO_QUERY, index_name="paper_embedding")
    record = _single_record(result)
    if record is None:
        raise RetrievalError(
            "Índice vetorial 'paper_embedding' não encontrado. Crie manualmente o "
            "índice para Paper.embedding com dimensão 384 e similaridade cosine; "
            "nenhuma alteração no schema foi feita."
        )

    name = _record_value(record, "name")
    state = _record_value(record, "state")
    index_type = _record_value(record, "type")
    labels = _record_value(record, "labelsOrTypes") or []
    properties = _record_value(record, "properties") or []
    options = _record_value(record, "options") or {}
    index_config = options.get("indexConfig", {}) if hasattr(options, "get") else {}
    dimensions = index_config.get("vector.dimensions") if hasattr(index_config, "get") else None
    similarity = (
        index_config.get("vector.similarity_function")
        if hasattr(index_config, "get")
        else None
    )

    problems: list[str] = []
    if name != "paper_embedding":
        problems.append("nome")
    if state != "ONLINE":
        problems.append(f"estado {state!r} (esperado ONLINE)")
    if str(index_type).upper() != "VECTOR":
        problems.append(f"tipo {index_type!r} (esperado VECTOR)")
    if "Paper" not in labels:
        problems.append("label Paper ausente")
    if "embedding" not in properties:
        problems.append("propriedade embedding ausente")
    try:
        dimensions_match = int(dimensions) == DIMENSIONS
    except (TypeError, ValueError):
        dimensions_match = False
    if not dimensions_match:
        problems.append(f"dimensão {dimensions!r} (esperada {DIMENSIONS})")
    if str(similarity).lower() != "cosine":
        problems.append(f"similaridade {similarity!r} (esperada cosine)")

    if problems:
        details = "; ".join(problems)
        raise RetrievalError(
            f"Índice vetorial 'paper_embedding' incompatível: {details}. "
            "Ajuste o índice manualmente; nenhuma alteração no schema foi feita."
        )
