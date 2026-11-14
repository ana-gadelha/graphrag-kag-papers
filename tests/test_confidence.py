import unittest

from retrieval.confidence import CONFIDENCE_METHOD, assess_confidence


def source(number, channels):
    return {"citation_id": f"S{number}", "retrieved_by": channels}


class ConfidenceTests(unittest.TestCase):
    def test_levels_depend_on_distinct_cited_sources_and_channel_agreement(self):
        dual = ["vector", "graph"]
        cases = [
            ("[S1]", [source(1, ["vector"])], "low"),
            ("[S1] [S1]", [source(1, ["graph"])], "low"),
            ("[S1]", [source(1, dual)], "medium"),
            ("[S1] [S2]", [source(1, ["vector"]), source(2, ["graph"])], "medium"),
            ("[S1] [S2]", [source(1, dual), source(2, ["graph"])], "medium"),
            ("[S1] [S2]", [source(1, dual), source(2, dual)], "high"),
            ("[S1]", [source(1, ["vector"]), source(2, dual)], "low"),
            ("Sem fontes", [], "low"),
            ("[S1]", [{"citation_id": "S1"}], "low"),
        ]
        for answer, sources, level in cases:
            with self.subTest(answer=answer, sources=sources):
                result = assess_confidence(answer, sources, abstained=False)
                self.assertEqual(result["level"], level)
                self.assertEqual(result["method"], CONFIDENCE_METHOD)
                self.assertTrue(result["reason"])

    def test_abstention_overrides_apparent_channel_agreement(self):
        result = assess_confidence(
            "[S1] [S2]", [source(1, ["vector", "graph"]), source(2, ["vector", "graph"])],
            abstained=True,
        )
        self.assertEqual(result["level"], "low")
        self.assertIn("absteve", result["reason"])
