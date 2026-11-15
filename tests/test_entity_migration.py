from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.entity_resolution import EntityRegistry
from src.migrate_entities import build_migration_plan, migrate_entities


def row(node_id, label, name, *, relations=None, extra_properties=None):
    return {
        "node_id": node_id,
        "labels": [label],
        "properties": {"name": name, **(extra_properties or {})},
        "relations": relations or [],
    }


class EntityMigrationTests(unittest.TestCase):
    def test_reviewed_aliases_plan_a_deterministic_merge(self):
        registry = EntityRegistry.load()
        plan = build_migration_plan(
            [row("node-b", "Method", "Graph RAG"), row("node-a", "Method", "GraphRAG")],
            registry,
        )
        self.assertTrue(plan["valid"])
        self.assertEqual(plan["nodes_to_merge"], 1)
        group = plan["groups"][0]
        self.assertEqual(group["canonical_name"], "GraphRAG")
        self.assertEqual(group["survivor_id"], "node-a")
        self.assertEqual(group["merge_node_ids"], ["node-b"])
        self.assertIn("Graph RAG", group["aliases"])

    def test_observed_entity_own_name_is_not_an_unreviewed_historical_alias(self):
        source = row("one", "Author", "Synthetic Normal Author", extra_properties={"aliases": ["Synthetic Normal Author"]})
        plan = build_migration_plan([source], EntityRegistry.load())
        self.assertTrue(plan["valid"], plan["conflicts"])
        self.assertIn("Synthetic Normal Author", plan["groups"][0]["aliases"])

    def test_unknown_near_names_are_not_fuzzy_merged(self):
        plan = build_migration_plan(
            [row("one", "Method", "GraphRAG"), row("two", "Method", "GraphRagX")],
            EntityRegistry.load(),
        )
        self.assertTrue(plan["valid"])
        self.assertEqual(plan["entities_after_migration"], 2)
        self.assertEqual(plan["nodes_to_merge"], 0)

    def test_duplicate_external_relation_and_properties_abort_before_changes(self):
        external = {
            "type": "CITES",
            "start_id": "node-two",
            "end_id": "paper-x",
            "start_labels": ["Method"],
            "end_labels": ["Paper"],
        }
        plan = build_migration_plan(
            [
                row("node-one", "Method", "GraphRAG"),
                row("node-two", "Method", "Graph RAG", relations=[external]),
            ],
            EntityRegistry.load(),
        )
        self.assertFalse(plan["valid"])
        self.assertTrue(any("external relationship" in item["reason"] for item in plan["conflicts"]))

        plan = build_migration_plan(
            [
                row("node-one", "Method", "GraphRAG"),
                row("node-two", "Method", "Graph RAG", extra_properties={"owner": "team"}),
            ],
            EntityRegistry.load(),
        )
        self.assertFalse(plan["valid"])
        self.assertTrue(any("unrecognized properties" in item["reason"] for item in plan["conflicts"]))

    def test_dry_run_is_read_only_and_reports_the_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "entity_registry.json"
            EntityRegistry.load().save(registry_path)
            driver = MagicMock()
            session = driver.session.return_value.__enter__.return_value
            session.run.return_value = [
                row("node-one", "Method", "GraphRAG"),
                row("node-two", "Method", "Graph RAG"),
            ]
            report = migrate_entities(driver, "neo4j", registry_path)
            self.assertTrue(report["valid"])
            self.assertEqual(report["mode"], "dry_run")
            self.assertEqual(report["nodes_to_merge"], 1)
            session.execute_write.assert_not_called()
            self.assertFalse(any(call.args and "CREATE CONSTRAINT" in str(call.args[0]) for call in session.run.call_args_list))

    def test_apply_runs_data_transaction_then_idempotent_entity_constraints(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "entity_registry.json"
            EntityRegistry.load().save(registry_path)
            driver = MagicMock()
            session = driver.session.return_value.__enter__.return_value
            tx = MagicMock()
            session.execute_write.side_effect = lambda fn, *args: fn(tx, *args)
            records = [row("node-one", "Method", "GraphRAG"), row("node-two", "Method", "Graph RAG")]
            with (
                patch("src.migrate_entities._records", return_value=records),
                patch("src.migrate_entities._schema_migration") as schema,
            ):
                report = migrate_entities(driver, "neo4j", registry_path, apply=True)
            self.assertTrue(report["applied"])
            session.execute_write.assert_called_once()
            schema.assert_called_once_with(driver, "neo4j")
            queries = [call.args[0] for call in tx.run.call_args_list]
            self.assertTrue(any("CREATE (other)-[new:PROPOSES]->(survivor)" in query for query in queries))
            self.assertTrue(any("DELETE old" in query and "DETACH" not in query for query in queries))
            self.assertFalse(any("apoc." in query.casefold() for query in queries))
            self.assertEqual(EntityRegistry.load(registry_path).resolve("Method", "Graph RAG").canonical_name, "GraphRAG")

    def test_apply_rechecks_for_new_external_relationships_inside_transaction(self):
        from src.migrate_entities import _apply_data_migration

        registry = EntityRegistry.load()
        original = [row("node-one", "Method", "GraphRAG"), row("node-two", "Method", "Graph RAG")]
        plan = build_migration_plan(original, registry)
        group = plan["groups"][0]
        external = {
            "type": "CITES",
            "start_id": "node-two",
            "end_id": "paper-x",
            "start_labels": ["Method"],
            "end_labels": ["Paper"],
            "properties": {},
        }
        changed = [
            original[0],
            {**original[1], "relations": [external]},
        ]
        tx = MagicMock()
        with patch("src.migrate_entities._records", return_value=changed):
            with self.assertRaisesRegex(RuntimeError, "conflicts changed"):
                _apply_data_migration(
                    tx,
                    plan["groups"],
                    registry.to_dict(),
                    {group["entity_id"]: group["_snapshot"]},
                )
        tx.run.assert_not_called()

    def test_conflicted_dry_run_never_opens_write_transaction(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "entity_registry.json"
            EntityRegistry.load().save(registry_path)
            driver = MagicMock()
            driver.session.return_value.__enter__.return_value.run.return_value = [
                row("node-one", "Method", "GraphRAG"),
                row("node-two", "Method", "Graph RAG", extra_properties={"custom": "value"}),
            ]
            report = migrate_entities(driver, "neo4j", registry_path, apply=True)
            self.assertFalse(report["valid"])
            driver.session.return_value.__enter__.return_value.execute_write.assert_not_called()

    def test_unreviewed_historical_alias_is_previewed_but_blocks_apply(self):
        records = [
            row("node-one", "Method", "GraphRAG"),
            row(
                "node-two",
                "Method",
                "Graph RAG",
                extra_properties={"aliases": ["lost-custom-alias"]},
            ),
        ]
        plan = build_migration_plan(records, EntityRegistry.load())
        group = plan["groups"][0]
        self.assertIn("lost-custom-alias", group["aliases"])
        self.assertFalse(plan["valid"])
        self.assertTrue(
            any("not in the reviewed registry" in item["reason"] for item in plan["conflicts"])
        )

        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "entity_registry.json"
            EntityRegistry.load().save(registry_path)
            original_registry = registry_path.read_text(encoding="utf-8")
            driver = MagicMock()
            driver.session.return_value.__enter__.return_value.run.return_value = records
            report = migrate_entities(driver, "neo4j", registry_path, apply=True)
            self.assertFalse(report["valid"])
            driver.session.return_value.__enter__.return_value.execute_write.assert_not_called()
            self.assertEqual(registry_path.read_text(encoding="utf-8"), original_registry)

    def test_reviewed_historical_alias_survives_merge_and_repeat_apply(self):
        historical_alias = "historical-rag-name"
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "entity_registry.json"
            registry_data = EntityRegistry.load().to_dict()
            graphrag = next(
                entity
                for entity in registry_data["entities"]
                if entity["label"] == "Method" and entity["canonical_name"] == "GraphRAG"
            )
            graphrag["aliases"].append(historical_alias)
            registry_path.write_text(json.dumps(registry_data), encoding="utf-8")
            records = [
                row("node-one", "Method", "GraphRAG"),
                row(
                    "node-two",
                    "Method",
                    "Graph RAG",
                    extra_properties={"aliases": [historical_alias]},
                ),
            ]
            driver = MagicMock()
            session = driver.session.return_value.__enter__.return_value
            tx = MagicMock()
            session.execute_write.side_effect = lambda fn, *args: fn(tx, *args)
            with (
                patch("src.migrate_entities._records", return_value=records),
                patch("src.migrate_entities._schema_migration"),
            ):
                first = migrate_entities(driver, "neo4j", registry_path, apply=True)
            self.assertTrue(first["applied"])
            self.assertIn(historical_alias, first["changes"][0]["aliases"])
            survivor_update = next(
                call for call in tx.run.call_args_list if "SET n.entity_id=$entity_id" in call.args[0]
            )
            self.assertIn(historical_alias, survivor_update.kwargs["aliases"])
            persisted = EntityRegistry.load(registry_path)
            self.assertEqual(
                persisted.lookup("Method", historical_alias).canonical_name,
                "GraphRAG",
            )

            canonical_id = persisted.resolve("Method", "GraphRAG").entity_id
            after_merge = [
                row(
                    "node-one",
                    "Method",
                    "GraphRAG",
                    extra_properties={
                        "entity_id": canonical_id,
                        "canonical_name": "GraphRAG",
                        "aliases": ["GraphRAG", "Graph RAG", historical_alias],
                        "identity_origin": "reviewed_seed",
                    },
                )
            ]
            repeat_driver = MagicMock()
            repeat_session = repeat_driver.session.return_value.__enter__.return_value
            repeat_tx = MagicMock()
            repeat_session.execute_write.side_effect = lambda fn, *args: fn(repeat_tx, *args)
            with (
                patch("src.migrate_entities._records", return_value=after_merge),
                patch("src.migrate_entities._schema_migration"),
            ):
                second = migrate_entities(
                    repeat_driver, "neo4j", registry_path, apply=True
                )
            self.assertTrue(second["applied"])
            self.assertEqual(second["nodes_to_merge"], 0)
            self.assertIn(historical_alias, second["changes"][0]["aliases"])
            repeat_update = next(
                call
                for call in repeat_tx.run.call_args_list
                if "SET n.entity_id=$entity_id" in call.args[0]
            )
            self.assertIn(historical_alias, repeat_update.kwargs["aliases"])
            self.assertEqual(
                EntityRegistry.load(registry_path)
                .lookup("Method", historical_alias)
                .canonical_name,
                "GraphRAG",
            )

    def test_historical_alias_owned_by_another_registry_entity_blocks_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "entity_registry.json"
            data = EntityRegistry.load().to_dict()
            from src.entity_resolution import entity_id_for

            data["entities"].append(
                {
                    "entity_id": entity_id_for("Method", "OtherMethod"),
                    "label": "Method",
                    "canonical_name": "OtherMethod",
                    "aliases": ["OtherMethod", "historical-rag-name"],
                    "identity_origin": "reviewed_seed",
                }
            )
            registry_path.write_text(json.dumps(data), encoding="utf-8")
            plan = build_migration_plan(
                [
                    row("node-one", "Method", "GraphRAG"),
                    row(
                        "node-two",
                        "Method",
                        "Graph RAG",
                        extra_properties={"aliases": ["historical-rag-name"]},
                    ),
                ],
                EntityRegistry.load(registry_path),
            )
        self.assertFalse(plan["valid"])
        self.assertTrue(
            any(
                "different registry entity" in item["reason"]
                for item in plan["conflicts"]
            )
        )

    def test_alias_claimed_by_another_graph_group_is_blocked(self):
        plan = build_migration_plan(
            [
                row(
                    "node-one",
                    "Method",
                    "GraphRAG",
                    extra_properties={"aliases": ["Novel Method"]},
                ),
                row("node-other", "Method", "Novel Method"),
            ],
            EntityRegistry.load(),
        )
        self.assertFalse(plan["valid"])
        self.assertTrue(
            any(
                "claimed by another graph entity" in item["reason"]
                for item in plan["conflicts"]
            )
        )

    def test_malformed_historical_alias_property_blocks_migration(self):
        plan = build_migration_plan(
            [row("node-one", "Method", "GraphRAG", extra_properties={"aliases": "Graph RAG"})],
            EntityRegistry.load(),
        )
        self.assertFalse(plan["valid"])
        self.assertTrue(
            any("list of non-empty strings" in item["reason"] for item in plan["conflicts"])
        )


if __name__ == "__main__":
    unittest.main()
