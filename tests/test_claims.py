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

    def test_superscript_quotes_restore_source_spelling_and_offsets(self):
        for field in ("title", "abstract"):
            for source_form in ("²", "$^2$", "$^{2}$"):
                for model_form in ("²", "$^2$", "$^{2}$"):
                    with self.subTest(field=field, source=source_form, model=model_form):
                        data = evidence()
                        original_quote = f"We evaluate KG{source_form}RAG on HotpotQA."
                        source = f"Prefix. {original_quote} Suffix."
                        data["sources"][0][field] = source
                        payload = document()
                        supplied = payload["claims"][0]["quotes"][0]
                        supplied.update(
                            field=field,
                            quote=f"We evaluate KG{model_form}RAG on HotpotQA.",
                        )
                        result = validate_claims(payload, data)
                        actual = result["claims"][0]["quotes"][0]
                        self.assertEqual(actual["quote"], original_quote)
                        self.assertEqual(actual["start"], len("Prefix. "))
                        self.assertEqual(actual["end"], len("Prefix. ") + len(original_quote))
                        self.assertEqual(source[actual["start"]:actual["end"]], actual["quote"])
                        self.assertEqual(supplied["quote"], f"We evaluate KG{model_form}RAG on HotpotQA.")

    def test_superscript_recovery_preserves_all_other_characters(self):
        data = evidence()
        data["sources"][0]["abstract"] = "We evaluate KG$^2$RAG on HotpotQA."
        for quote in (
            "We evaluate KG²RAG on MuSiQue.",
            "We evaluate KG²RAG on hotpotqa.",
            "We evaluate KG²RAG on HotpotQA!",
            "We do not evaluate KG²RAG on HotpotQA.",
            "We evaluate KG³RAG on HotpotQA.",
            "We evaluate KG2RAG on HotpotQA.",
            "We evaluate KG^2RAG on HotpotQA.",
        ):
            payload = document()
            payload["claims"][0]["quotes"][0]["quote"] = quote
            with self.subTest(quote=quote), self.assertRaises(RetrievalError):
                validate_claims(payload, data)

    def test_superscript_recovery_rejects_ambiguous_source_spans(self):
        data = evidence()
        data["sources"][0]["abstract"] = (
            "We evaluate KG$^2$RAG on HotpotQA. "
            "We evaluate KG$^{2}$RAG on HotpotQA."
        )
        payload = document()
        payload["claims"][0]["quotes"][0]["quote"] = "We evaluate KG²RAG on HotpotQA."
        with self.assertRaises(RetrievalError):
            validate_claims(payload, data)

    def test_exact_quote_takes_precedence_over_superscript_recovery(self):
        data = evidence()
        prefix = "We evaluate KG$^2$RAG on HotpotQA. "
        quote = "We evaluate KG²RAG on HotpotQA."
        data["sources"][0]["abstract"] = prefix + quote
        payload = document()
        payload["claims"][0]["quotes"][0]["quote"] = quote
        actual = validate_claims(payload, data)["claims"][0]["quotes"][0]
        self.assertEqual(actual["quote"], quote)
        self.assertEqual(actual["start"], len(prefix))

    def test_superscript_recovery_cannot_borrow_evidence_from_another_source(self):
        data = evidence()
        data["sources"].append({
            "citation_id": "S2", "paper_id": "b",
            "abstract": "We evaluate KG$^2$RAG on HotpotQA.",
        })
        payload = document()
        payload["claims"][0]["quotes"][0]["quote"] = "We evaluate KG²RAG on HotpotQA."
        with self.assertRaises(RetrievalError):
            validate_claims(payload, data)

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
