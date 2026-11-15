import unittest
from copy import deepcopy

from retrieval.claims import validate_claims
from retrieval.config import RetrievalError
from retrieval.hybrid import assign_evidence_ids


def evidence():
    paper = {
        "id": "paper:a",
        "label": "Paper",
        "properties": {"id": "a", "title": "Paper A"},
    }
    entity = {
        "id": "dataset:d",
        "label": "Dataset",
        "properties": {"name": "Dataset D"},
    }
    edge = {"source": paper["id"], "target": entity["id"], "type": "EVALUATED_ON"}
    result = {
        "sources": [
            {
                "citation_id": "S1",
                "paper_id": "a",
                "abstract": "We evaluate method M on dataset D.",
                "title": "Paper A",
            }
        ],
        "subgraph": {"nodes": [paper, entity], "edges": [edge]},
        "paths": [],
    }
    assign_evidence_ids(result)
    return result


def document():
    return {
        "abstained": False,
        "claims": [
            {
                "kind": "document",
                "text": "O estudo avalia M no dataset D.",
                "source_ids": ["S1"],
                "path_ids": [],
                "quotes": [
                    {
                        "source_id": "S1",
                        "field": "abstract",
                        "quote": "We evaluate method M on dataset D.",
                    }
                ],
            }
        ],
    }


class ClaimsTests(unittest.TestCase):
    def test_document_records_exact_offsets_and_does_not_accept_legacy_free_answer(
        self,
    ):
        result = validate_claims(document(), evidence())
        self.assertEqual(result["answer"], "O estudo avalia M no dataset D. [S1]")
        self.assertEqual(result["claims"][0]["quotes"][0]["start"], 0)
        with self.assertRaises(RetrievalError):
            validate_claims(
                {"answer": "A Lua é feita de queijo [S1].", "abstained": False},
                evidence(),
            )

    def test_quotes_must_cover_every_source_and_exist_in_original_text(self):
        for modification in [
            [],
            [{"source_id": "S1", "field": "abstract", "quote": "not in the abstract"}],
            [{"source_id": "S1", "field": "year", "quote": "2025"}],
        ]:
            payload = document()
            payload["claims"][0]["quotes"] = modification
            with (
                self.subTest(modification=modification),
                self.assertRaises(RetrievalError),
            ):
                validate_claims(payload, evidence())

    def test_graph_claim_is_rendered_from_verified_relations(self):
        data = evidence()
        payload = {
            "abstained": False,
            "claims": [
                {
                    "kind": "graph",
                    "text": "an untrusted invented conclusion",
                    "source_ids": ["S1"],
                    "path_ids": ["P1"],
                    "quotes": [],
                }
            ],
        }
        result = validate_claims(payload, data)
        self.assertNotIn("invented", result["answer"])
        self.assertIn("EVALUATED_ON", result["answer"])
        self.assertIn("[P1]", result["answer"])
        for mutate in [
            lambda d: d["paths"][0].update(length=2),
            lambda d: d["paths"][0]["edges"][0].update(type="PROPOSES"),
            lambda d: d["paths"][0].update(source="S999"),
            lambda d: d["paths"][0]["nodes"][0].update(id="paper:other"),
        ]:
            broken = deepcopy(data)
            mutate(broken)
            with self.subTest(mutate=mutate), self.assertRaises(RetrievalError):
                validate_claims(payload, broken)

    def test_graph_claim_cannot_pad_sources_to_inflate_confidence(self):
        data = evidence()
        data["sources"].append(
            {"citation_id": "S2", "paper_id": "b", "title": "Unrelated"}
        )
        claim = {
            "kind": "graph",
            "text": "Conexão.",
            "source_ids": ["S1", "S2"],
            "path_ids": ["P1"],
            "quotes": [],
        }
        with self.assertRaisesRegex(RetrievalError, "não pertencem"):
            validate_claims({"abstained": False, "claims": [claim]}, data)

    def test_unknown_sources_paths_and_embedded_fake_citations_rejected(self):
        for key, value in [
            ("source_ids", ["S999"]),
            ("path_ids", ["P999"]),
            ("text", "Falso [S1]"),
            ("source_ids", ["S1", "S1"]),
        ]:
            payload = document()
            payload["claims"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RetrievalError):
                validate_claims(payload, evidence())

    def test_abstention_is_fact_free_and_strict(self):
        self.assertTrue(
            validate_claims({"abstained": True, "claims": []}, evidence())["abstained"]
        )
        payload = document()
        payload["abstained"] = True
        with self.assertRaises(RetrievalError):
            validate_claims(payload, evidence())

    def test_model_links_are_rendered_as_literal_text(self):
        payload = document()
        payload["claims"][0]["text"] = "[Click](https://evil.test) <script>"
        answer = validate_claims(payload, evidence())["answer"]
        self.assertIn(r"\[Click\]", answer)
        self.assertIn(r"\<script\>", answer)


if __name__ == "__main__":
    unittest.main()
