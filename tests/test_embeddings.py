from __future__ import annotations

import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from retrieval.config import RetrievalError, Settings
from retrieval.embeddings import (
    DIMENSIONS,
    EMBEDDING_MODEL,
    check_index,
    content_hash,
    embedding_text,
    validate_vector,
)
from retrieval.index_embeddings import index_papers


def valid_index() -> dict[str, object]:
    return {
        "name": "paper_embedding",
        "state": "ONLINE",
        "type": "VECTOR",
        "labelsOrTypes": ["Paper"],
        "properties": ["embedding"],
        "options": {
            "indexConfig": {
                "vector.dimensions": DIMENSIONS,
                "vector.similarity_function": "cosine",
            }
        },
    }


class FakeResult(list):
    def single(self):
        return self[0] if self else None


class FakeSession:
    def __init__(self, papers: list[dict[str, object]]) -> None:
        self.papers = papers
        self.queries: list[str] = []

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def run(self, query: str, **parameters: object) -> FakeResult:
        self.queries.append(query)
        if query.lstrip().startswith("SHOW INDEXES"):
            return FakeResult([valid_index()])
        if "ORDER BY id" in query:
            return FakeResult([dict(paper) for paper in self.papers])
        if "UNWIND $rows AS row" in query:
            updated = 0
            for row in parameters["rows"]:
                paper = next(
                    (item for item in self.papers if item["id"] == row["id"]),
                    None,
                )
                if paper is None:
                    continue
                current_keywords = list(paper.get("keywords", []))
                if (
                    (paper.get("title") or "") != (row.get("title") or "")
                    or (paper.get("abstract") or "") != (row.get("abstract") or "")
                    or len(current_keywords) != len(row["keywords"])
                    or any(keyword not in row["keywords"] for keyword in current_keywords)
                    or (
                        not row["force"]
                        and paper.get("embedding_model")
                        != row["previous_embedding_model"]
                    )
                ):
                    continue
                paper["embedding"] = row["embedding"]
                paper["embedding_model"] = row["embedding_model"]
                paper["embedding_hash"] = row["embedding_hash"]
                updated += 1
            return FakeResult([{"updated": updated}])
        raise AssertionError(f"Unexpected query: {query}")


class FakeDriver:
    def __init__(self, papers: list[dict[str, object]]) -> None:
        self.fake_session = FakeSession(papers)

    def session(self, *, database: str) -> FakeSession:
        self.database = database
        return self.fake_session


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **_: object) -> list[list[float]]:
        self.calls.append(texts)
        return [
            [float(sum(text.encode("utf-8")) % 997 + 1)] + [0.0] * (DIMENSIONS - 1)
            for text in texts
        ]


class EmbeddingTests(unittest.TestCase):
    def test_embedding_text_sorts_concept_names_and_hashes_content(self) -> None:
        first = {
            "title": " Paper title ",
            "abstract": "Short abstract.",
            "keywords": ["Zeta", "Alpha", "Alpha"],
        }
        second = {**first, "keywords": ["Alpha", "Zeta"]}

        self.assertEqual(
            embedding_text(first),
            "Title: Paper title\nAbstract: Short abstract.\nKeywords: Alpha, Zeta",
        )
        self.assertEqual(content_hash(first), content_hash(second))
        self.assertNotEqual(content_hash(first), content_hash({**first, "abstract": "Changed."}))

    def test_validate_vector_checks_dimension_finiteness_and_nonzero(self) -> None:
        vector = validate_vector([1] + [0] * (DIMENSIONS - 1))
        self.assertEqual(vector[0], 1.0)
        self.assertEqual(len(vector), DIMENSIONS)

        invalid_vectors = (
            [1.0] * (DIMENSIONS - 1),
            [0.0] * DIMENSIONS,
            [float("nan")] + [1.0] * (DIMENSIONS - 1),
            ["1.0"] + [0.0] * (DIMENSIONS - 1),
        )
        for invalid in invalid_vectors:
            with self.subTest(vector_length=len(invalid)):
                with self.assertRaises(RetrievalError):
                    validate_vector(invalid)

    def test_check_index_accepts_expected_online_vector_index(self) -> None:
        driver = FakeDriver([])
        check_index(driver.session(database="neo4j"))

    def test_check_index_rejects_wrong_dimension_without_schema_mutation(self) -> None:
        bad_index = valid_index()
        bad_index["options"] = {
            "indexConfig": {
                "vector.dimensions": 1536,
                "vector.similarity_function": "cosine",
            }
        }

        class InvalidIndexSession:
            queries: list[str] = []

            def run(self, query: str, **_: object) -> FakeResult:
                self.queries.append(query)
                return FakeResult([bad_index])

        session = InvalidIndexSession()
        with self.assertRaisesRegex(RetrievalError, "dimensão 1536"):
            check_index(session)
        self.assertEqual(len(session.queries), 1)
        self.assertTrue(session.queries[0].lstrip().startswith("SHOW INDEXES"))

    def test_missing_neo4j_config_reports_names_without_secret_values(self) -> None:
        fake_dotenv = SimpleNamespace(load_dotenv=lambda **_: False)
        with patch.dict(sys.modules, {"dotenv": fake_dotenv}), patch.dict(
            os.environ, {}, clear=True
        ):
            with self.assertRaises(RetrievalError) as caught:
                Settings.from_env()

        message = str(caught.exception)
        self.assertIn("NEO4J_URI", message)
        self.assertIn("NEO4J_PASSWORD", message)
        self.assertNotIn("secret", message.lower())

    def test_settings_repr_hides_connection_and_api_secrets(self) -> None:
        settings = Settings(
            uri="neo4j+s://private-uri",
            username="neo4j",
            password="db-secret",
            openai_api_key="api-secret",
        )

        representation = repr(settings)
        self.assertNotIn("private-uri", representation)
        self.assertNotIn("db-secret", representation)
        self.assertNotIn("api-secret", representation)

    def test_indexing_is_idempotent_and_refreshes_changed_hash(self) -> None:
        paper: dict[str, object] = {
            "id": "paper-1",
            "title": "Graph retrieval",
            "abstract": "A graph retrieval method.",
            "keywords": ["GraphRAG", "Retrieval"],
            "embedding": None,
            "embedding_model": None,
            "embedding_hash": None,
        }
        driver = FakeDriver([paper])
        encoder = FakeEncoder()

        first = index_papers(driver, "neo4j", encoder=encoder)
        first_hash = paper["embedding_hash"]
        second = index_papers(driver, "neo4j", encoder=encoder)

        self.assertEqual(first, {"papers": 1, "embedded": 1, "skipped": 0})
        self.assertEqual(second, {"papers": 1, "embedded": 0, "skipped": 1})
        self.assertEqual(len(encoder.calls), 1)
        self.assertEqual(paper["embedding_model"], EMBEDDING_MODEL)
        self.assertEqual(len(paper["embedding"]), DIMENSIONS)

        paper["abstract"] = "The abstract changed."
        third = index_papers(driver, "neo4j", encoder=encoder)

        self.assertEqual(third, {"papers": 1, "embedded": 1, "skipped": 0})
        self.assertNotEqual(first_hash, paper["embedding_hash"])
        self.assertEqual(len(encoder.calls), 2)

    def test_incompatible_model_requires_explicit_force(self) -> None:
        paper: dict[str, object] = {
            "id": "paper-2",
            "title": "Graph retrieval",
            "abstract": "A graph retrieval method.",
            "keywords": [],
            "embedding": [1.0] + [0.0] * (DIMENSIONS - 1),
            "embedding_model": "another-embedding-model",
            "embedding_hash": "old-hash",
        }
        driver = FakeDriver([paper])
        encoder = FakeEncoder()

        with self.assertRaisesRegex(RetrievalError, "--force"):
            index_papers(driver, "neo4j", encoder=encoder)
        self.assertEqual(encoder.calls, [])

        report = index_papers(driver, "neo4j", force=True, encoder=encoder)
        self.assertEqual(report, {"papers": 1, "embedded": 1, "skipped": 0})
        self.assertEqual(paper["embedding_model"], EMBEDDING_MODEL)

    def test_string_embedding_is_replaced_even_when_hash_matches(self) -> None:
        paper: dict[str, object] = {
            "id": "paper-3",
            "title": "Graph retrieval",
            "abstract": "A graph retrieval method.",
            "keywords": [],
            "embedding": ["1.0"] + ["0.0"] * (DIMENSIONS - 1),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_hash": content_hash(
                {"title": "Graph retrieval", "abstract": "A graph retrieval method."}
            ),
        }
        driver = FakeDriver([paper])
        encoder = FakeEncoder()

        report = index_papers(driver, "neo4j", encoder=encoder)

        self.assertEqual(report, {"papers": 1, "embedded": 1, "skipped": 0})
        self.assertEqual(len(encoder.calls), 1)
        self.assertTrue(all(isinstance(value, float) for value in paper["embedding"]))


if __name__ == "__main__":
    unittest.main()
