from __future__ import annotations

import unittest

from src.common import comparison_key, unique_clean_strings
from src.extract_entities import extract_by_rules
from src.validate_data import validate_and_normalize


SAMPLE = {
    "id": "2404.16130",
    "source": "arXiv",
    "url": "https://arxiv.org/abs/2404.16130",
    "title": " From Local to Global: A Graph RAG Approach ",
    "authors": ["John Smith", "John  Smith", "Maria Silva"],
    "year": 2024,
    "abstract": "We present GraphRAG and evaluate it on HotpotQA.",
    "keywords": ["GraphRAG", " GraphRAG ", "knowledge graphs"],
}


class PipelineTests(unittest.TestCase):
    def test_comparison_key_ignores_case_accents_and_spaces(self) -> None:
        self.assertEqual(comparison_key("  MÉTODO   A "), "metodo a")

    def test_unique_clean_strings(self) -> None:
        self.assertEqual(unique_clean_strings(["A", " A ", "B"]), ["A", "B"])

    def test_validation_normalizes_and_accepts_sample(self) -> None:
        papers, report = validate_and_normalize([SAMPLE])
        self.assertTrue(report["valid"])
        self.assertEqual(papers[0]["authors"], ["John Smith", "Maria Silva"])
        self.assertEqual(papers[0]["keywords"], ["GraphRAG", "knowledge graphs"])

    def test_rule_extraction(self) -> None:
        entities = extract_by_rules(SAMPLE)
        self.assertEqual(entities["proposed_methods"], ["GraphRAG"])
        self.assertEqual(entities["datasets"], ["HotpotQA"])

    def test_duplicate_ids_are_rejected(self) -> None:
        _, report = validate_and_normalize([SAMPLE, SAMPLE])
        self.assertFalse(report["valid"])
        self.assertTrue(any("duplicado" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
