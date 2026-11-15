from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from graph_component.data import (
    build_graph_payload,
    graph_payload_fingerprint,
    selected_graph_record,
)

ROOT = Path(__file__).resolve().parents[1]


class GraphPayloadTests(unittest.TestCase):
    def fixture(self):
        nodes = [
            {
                "id": "paper:one",
                "label": "Artigo",
                "tipo": "Paper",
                "properties": {
                    "title": "Artigo <script>alert(1)</script>",
                    "metadata": {"api_key": "must-not-leave-python", "year": 2025},
                },
                "source_ids": "S1",
                "provenance": {"source_ids": ["S1"], "access_token": "hidden"},
            },
            {
                "id": "dataset:one",
                "label": "HotpotQA",
                "tipo": "Dataset",
                "properties": {"name": "HotpotQA"},
            },
        ]
        edges = [
            {
                "id": "edge:stable-1",
                "origem": "paper:one",
                "destino": "dataset:one",
                "relacao": "EVALUATED_ON",
                "properties": {"weight": 1, "password": "hidden"},
                "source_ids": ["S1"],
                "provenance": {"citation_id": "S1"},
            }
        ]
        paths = [
            {
                "path_id": "P1",
                "kind": "fact",
                "source": "S1",
                "target": "S1",
                "length": 1,
                "nodes": nodes,
                "edges": edges,
            }
        ]
        claims = [
            {
                "kind": "dataset",
                "text": "Usou HotpotQA.",
                "source_ids": ["S1"],
                "path_ids": ["P1"],
                "quotes": [
                    {
                        "source_id": "S1",
                        "field": "abstract",
                        "quote": "evaluated on HotpotQA",
                        "start": 11,
                        "end": 34,
                    }
                ],
            }
        ]
        return nodes, edges, paths, claims

    def test_payload_is_deterministic_highlights_path_and_resolves_selection(self):
        nodes, edges, paths, claims = self.fixture()
        first = build_graph_payload(nodes, edges, paths, claims)
        second = build_graph_payload(nodes, edges, paths, claims)

        self.assertEqual(first, second)
        self.assertEqual(first["paths"][0]["path_id"], "P1")
        self.assertEqual(first["paths"][0]["kind"], "fact")
        self.assertEqual(first["paths"][0]["edge_ids"], ["edge:stable-1"])
        self.assertEqual(first["edges"][0]["source"], "paper:one")
        self.assertEqual(first["edges"][0]["target"], "dataset:one")
        self.assertEqual(first["edges"][0]["type"], "EVALUATED_ON")
        self.assertEqual(
            graph_payload_fingerprint(first), graph_payload_fingerprint(second)
        )
        self.assertEqual(
            selected_graph_record(first, {"type": "edge", "id": "edge:stable-1"})[
                "item_type"
            ],
            "edge",
        )
        self.assertIsNone(
            selected_graph_record(first, {"type": "node", "id": "missing"})
        )

    def test_credentials_are_removed_and_claim_payload_is_narrow(self):
        nodes, edges, paths, claims = self.fixture()
        payload = build_graph_payload(nodes, edges, paths, claims)
        serialized = json.dumps(payload)

        for sensitive in (
            "must-not-leave-python",
            "access_token",
            "hidden",
            "api_key",
            "password",
        ):
            self.assertNotIn(sensitive, serialized)
        self.assertEqual(
            payload["claims"],
            [
                {
                    "kind": "dataset",
                    "text": "Usou HotpotQA.",
                    "source_ids": ["S1"],
                    "path_ids": ["P1"],
                    "quotes": [
                        {
                            "source_id": "S1",
                            "field": "abstract",
                            "quote": "evaluated on HotpotQA",
                            "start": 11,
                            "end": 34,
                        }
                    ],
                }
            ],
        )
        self.assertNotIn("planner", payload)
        self.assertIn(
            "<script>alert(1)</script>", payload["nodes"][0]["properties"]["title"]
        )

    def test_missing_edge_ids_are_deterministic_and_shared_with_path_edges(self):
        nodes, _, _, claims = self.fixture()
        edges = [
            {
                "origem": "paper:one",
                "destino": "dataset:one",
                "relacao": "EVALUATED_ON",
                "properties": {"weight": 1},
            }
        ]
        paths = [{"path_id": "P1", "kind": "fact", "edges": edges}]

        first = build_graph_payload(nodes, edges, paths, claims)
        second = build_graph_payload(nodes, edges, paths, claims)

        self.assertEqual(first["edges"][0]["id"], second["edges"][0]["id"])
        self.assertEqual(first["paths"][0]["edge_ids"], [first["edges"][0]["id"]])

    def test_frontend_uses_text_nodes_and_contains_no_network_request_path(self):
        javascript = (ROOT / "graph_component" / "viewer.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("textContent", javascript)
        self.assertNotIn("innerHTML", javascript)
        self.assertNotIn("fetch(", javascript)
        self.assertNotIn("XMLHttpRequest", javascript)
        self.assertNotIn("WebSocket", javascript)

    def test_component_keeps_cytoscape_mouse_events_in_document_dom(self):
        from graph_component.component import _graph_component

        _graph_component.cache_clear()
        try:
            with patch("graph_component.component.component") as register:
                register.return_value = object()
                registered = _graph_component()

            self.assertIs(registered, register.return_value)
            self.assertFalse(register.call_args.kwargs["isolate_styles"])
        finally:
            _graph_component.cache_clear()


if __name__ == "__main__":
    unittest.main()
