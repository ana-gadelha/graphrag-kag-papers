"""Populate Paper embeddings using the configured local multilingual model."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Iterable

from retrieval.config import RetrievalError, Settings, create_driver
from retrieval.embeddings import (
    EMBEDDING_MODEL,
    check_index,
    content_hash,
    embedding_text,
    encode_texts,
    validate_vector,
)


PAPERS_QUERY = """
MATCH (p:Paper)
OPTIONAL MATCH (p)-[:HAS_CONCEPT]->(c:Concept)
WITH p, collect(DISTINCT c.name) AS keywords
RETURN p.id AS id, p.title AS title, p.abstract AS abstract, keywords,
       p.embedding AS embedding, p.embedding_model AS embedding_model,
       p.embedding_hash AS embedding_hash
ORDER BY id
"""

UPDATE_EMBEDDINGS_QUERY = """
UNWIND $rows AS row
MATCH (p:Paper {id: row.id})
OPTIONAL MATCH (p)-[:HAS_CONCEPT]->(c:Concept)
WITH p, row, collect(DISTINCT c.name) AS current_keywords
WHERE coalesce(p.title, '') = coalesce(row.title, '')
  AND coalesce(p.abstract, '') = coalesce(row.abstract, '')
  AND size(current_keywords) = size(row.keywords)
  AND all(keyword IN current_keywords WHERE keyword IN row.keywords)
  AND (
    row.force
    OR coalesce(p.embedding_model, '') = coalesce(row.previous_embedding_model, '')
  )
SET p.embedding = row.embedding,
    p.embedding_model = row.embedding_model,
    p.embedding_hash = row.embedding_hash
RETURN count(p) AS updated
"""


def _value(record: Any, key: str, default: Any = None) -> Any:
    try:
        return record.get(key, default)
    except AttributeError:
        try:
            return record[key]
        except (KeyError, TypeError, IndexError):
            return default


def _records(result: Any) -> list[Any]:
    try:
        return list(result)
    except TypeError:
        return []


def _read_papers(session: Any) -> list[dict[str, Any]]:
    return _records(session.run(PAPERS_QUERY))


def _stored_vector_is_valid(vector: Any) -> bool:
    if vector is None:
        return False
    try:
        validate_vector(vector)
    except RetrievalError:
        return False
    return True


def _check_model_compatibility(papers: Iterable[Any], *, force: bool) -> None:
    if force:
        return
    for paper in papers:
        stored_model = _value(paper, "embedding_model")
        stored_vector = _value(paper, "embedding")
        if stored_model and stored_model != EMBEDDING_MODEL:
            raise RetrievalError(
                f"O Paper {_value(paper, 'id')!r} tem embedding do modelo "
                f"{stored_model!r}; o índice usa {EMBEDDING_MODEL!r}. "
                "Use --force para re-embedar todos os papers explicitamente."
            )
        if stored_vector is not None and stored_model != EMBEDDING_MODEL:
            raise RetrievalError(
                f"O Paper {_value(paper, 'id')!r} tem embedding sem metadados "
                "compatíveis. Use --force para re-embedar todos os papers explicitamente."
            )


def _updates_for_papers(
    papers: list[Any], *, force: bool
) -> tuple[list[dict[str, Any]], int]:
    updates: list[dict[str, Any]] = []
    skipped = 0
    for source in papers:
        paper_id = _value(source, "id")
        if paper_id is None:
            raise RetrievalError("Foi encontrado um Paper sem a propriedade id.")

        title = _value(source, "title")
        abstract = _value(source, "abstract")
        keywords = _value(source, "keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        text_paper = {"title": title, "abstract": abstract, "keywords": keywords}
        source_hash = content_hash(text_paper)
        stored_hash = _value(source, "embedding_hash")
        stored_model = _value(source, "embedding_model")
        stored_vector = _value(source, "embedding")

        can_skip = (
            not force
            and stored_hash == source_hash
            and stored_model == EMBEDDING_MODEL
            and _stored_vector_is_valid(stored_vector)
        )
        if can_skip:
            skipped += 1
            continue

        updates.append(
            {
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "keywords": list(keywords),
                "text": embedding_text(text_paper),
                "embedding_hash": source_hash,
                "previous_embedding_model": stored_model,
                "force": force,
            }
        )
    return updates, skipped


def _write_batches(session: Any, updates: list[dict[str, Any]], batch_size: int) -> int:
    updated_total = 0
    for offset in range(0, len(updates), batch_size):
        batch = updates[offset : offset + batch_size]
        rows = [
            {
                "id": item["id"],
                "title": item["title"],
                "abstract": item["abstract"],
                "keywords": item["keywords"],
                "embedding": item["embedding"],
                "embedding_model": EMBEDDING_MODEL,
                "embedding_hash": item["embedding_hash"],
                "previous_embedding_model": item["previous_embedding_model"],
                "force": item["force"],
            }
            for item in batch
        ]
        result = session.run(UPDATE_EMBEDDINGS_QUERY, rows=rows)
        single = getattr(result, "single", None)
        record = single() if callable(single) else next(iter(result), None)
        updated = _value(record, "updated") if record is not None else None
        if updated is None:
            raise RetrievalError("Neo4j não confirmou a atualização dos embeddings.")
        updated_total += int(updated)

    if updated_total != len(updates):
        raise RetrievalError(
            "Um ou mais papers mudaram enquanto os embeddings eram gerados; "
            "as linhas incompatíveis foram ignoradas, embora outros lotes possam "
            "ter sido atualizados. Execute o indexador novamente."
        )
    return updated_total


def index_papers(
    driver: Any,
    database: str,
    *,
    force: bool = False,
    batch_size: int = 64,
    encoder: Any | None = None,
) -> dict[str, int]:
    """Check the existing index, then insert or refresh stale Paper embeddings."""
    if batch_size < 1:
        raise RetrievalError("batch_size deve ser maior que zero.")

    with driver.session(database=database) as session:
        check_index(session)
        papers = _read_papers(session)
        _check_model_compatibility(papers, force=force)
        updates, skipped = _updates_for_papers(papers, force=force)

        if updates:
            vectors = encode_texts(
                [item["text"] for item in updates],
                encoder=encoder,
            )
            for item, vector in zip(updates, vectors, strict=True):
                item["embedding"] = vector
            embedded = _write_batches(session, updates, batch_size)
        else:
            embedded = 0

    return {"papers": len(papers), "embedded": embedded, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera embeddings locais para os nós Paper do Neo4j."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-embeda todos os papers, mesmo quando há modelo incompatível",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="quantidade de papers por atualização (padrão: 64)",
    )
    args = parser.parse_args(argv)

    driver = None
    try:
        settings = Settings.from_env()
        driver = create_driver(settings)
        report = index_papers(
            driver,
            settings.database,
            force=args.force,
            batch_size=args.batch_size,
        )
    except RetrievalError as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1
    except Exception:
        # Database and driver exceptions may carry authentication details.
        print(
            "Erro: falha inesperada ao indexar embeddings; detalhes de conexão omitidos.",
            file=sys.stderr,
        )
        return 1
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass

    print(
        "Indexação concluída: "
        f"{report['embedded']} atualizados, {report['skipped']} já válidos, "
        f"{report['papers']} papers lidos."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
