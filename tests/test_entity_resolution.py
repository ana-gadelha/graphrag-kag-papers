from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.entity_resolution import EntityRegistry, entity_id_for


class EntityResolutionTests(unittest.TestCase):
    def test_reviewed_alias_resolves_to_same_deterministic_entity(self):
        registry = EntityRegistry.load()
        canonical = registry.resolve("Method", "GraphRAG")
        spaced = registry.resolve("Method", "Graph RAG")
        dashed = registry.resolve("Method", "graph-rag")
        self.assertEqual(canonical.entity_id, spaced.entity_id)
        self.assertEqual(canonical.entity_id, dashed.entity_id)
        self.assertEqual(canonical.canonical_name, "GraphRAG")

    def test_unreviewed_near_match_does_not_fuzzy_merge(self):
        registry = EntityRegistry.load()
        reviewed = registry.resolve("Method", "GraphRAG")
        typo = registry.resolve("Method", "GraphRagX")
        self.assertNotEqual(reviewed.entity_id, typo.entity_id)
        self.assertEqual(typo.canonical_name, "GraphRagX")

    def test_alias_ambiguity_blocks_registry_load(self):
        data = {
            "version": 1,
            "identity_policy": {},
            "entities": [
                {"label": "Method", "canonical_name": "One", "aliases": ["shared"]},
                {"label": "Method", "canonical_name": "Two", "aliases": ["shared"]},
            ],
        }
        with self.assertRaisesRegex(ValueError, "Ambiguous Method alias"):
            EntityRegistry(data)

    def test_ids_are_stable_and_author_name_limit_is_documented(self):
        self.assertEqual(entity_id_for("Author", "J. Smith"), entity_id_for("Author", "j.  smith"))
        registry = EntityRegistry.load()
        self.assertIn("homonyms", registry.identity_policy["Author"])

    def test_registry_save_is_deterministic_and_keeps_observed_names(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "registry.json"
            registry = EntityRegistry.load()
            one = registry.resolve("Dataset", "A New Benchmark")
            registry.save(destination)
            saved = json.loads(destination.read_text(encoding="utf-8"))
            self.assertIn(one.entity_id, {entry["entity_id"] for entry in saved["entities"]})
            reloaded = EntityRegistry.load(destination)
            self.assertEqual(
                reloaded.resolve("Dataset", "A New Benchmark").entity_id,
                one.entity_id,
            )


if __name__ == "__main__":
    unittest.main()
