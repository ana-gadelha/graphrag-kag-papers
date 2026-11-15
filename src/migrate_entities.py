"""Backfill stable entity IDs and merge only provably compatible aliases."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from .common import clean_text, comparison_key
from .entity_resolution import DEFAULT_REGISTRY_PATH, EntityRegistry
from .import_graph import OWNED_TYPES


LABELS = ("Author", "Concept", "Method", "Dataset")
OLD_CONSTRAINTS = (
    "author_name_unique",
    "concept_name_unique",
    "method_name_unique",
    "dataset_name_unique",
)
NEW_CONSTRAINTS = {
    "Author": "author_entity_id_unique",
    "Concept": "concept_entity_id_unique",
    "Method": "method_entity_id_unique",
    "Dataset": "dataset_entity_id_unique",
}
MANAGED_PROPERTIES = {"name", "entity_id", "canonical_name", "aliases", "identity_origin"}
EDGE_LAYOUTS = {
    "AUTHORED": ("Author", "Paper", "outgoing"),
    "HAS_CONCEPT": ("Paper", "Concept", "incoming"),
    "PROPOSES": ("Paper", "Method", "incoming"),
    "USES_METHOD": ("Paper", "Method", "incoming"),
    "EVALUATED_ON": ("Paper", "Dataset", "incoming"),
}


def _records(session: Any) -> list[dict[str, Any]]:
    rows = session.run(
        """MATCH (n)
        WHERE n:Author OR n:Concept OR n:Method OR n:Dataset
        OPTIONAL MATCH (n)-[r]-(other)
        RETURN elementId(n) AS node_id, labels(n) AS labels,
               properties(n) AS properties,
               collect(CASE WHEN r IS NULL THEN null ELSE {
                   type: type(r),
                   start_id: elementId(startNode(r)),
                   end_id: elementId(endNode(r)),
                   start_labels: labels(startNode(r)),
                   end_labels: labels(endNode(r)),
                   properties: properties(r)
               } END) AS relations"""
    )
    return [dict(row) for row in rows]


def _relation_error(relation: dict[str, Any]) -> str | None:
    layout = EDGE_LAYOUTS.get(relation.get("type"))
    if layout is None:
        return f"external relationship {relation.get('type')}"
    start_label, end_label, _ = layout
    if start_label not in relation.get("start_labels", []):
        return f"malformed {relation['type']} start endpoint"
    if end_label not in relation.get("end_labels", []):
        return f"malformed {relation['type']} end endpoint"
    return None


def _stable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _stable_value(value[key])) for key in sorted(value))
    if isinstance(value, (list, tuple)):
        return tuple(_stable_value(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def _record_snapshot(row: dict[str, Any]) -> tuple[Any, ...]:
    properties = row.get("properties") or {}
    managed = {
        key: properties[key]
        for key in MANAGED_PROPERTIES
        if key in properties
    }
    relations = [relation for relation in row.get("relations") or [] if relation]
    relation_snapshots = sorted(
        (
            relation.get("type"),
            relation.get("start_id"),
            relation.get("end_id"),
            tuple(sorted(relation.get("start_labels") or [])),
            tuple(sorted(relation.get("end_labels") or [])),
            _stable_value(relation.get("properties") or {}),
        )
        for relation in relations
    )
    return (
        tuple(sorted(row.get("labels") or [])),
        _stable_value(managed),
        tuple(relation_snapshots),
    )


def build_migration_plan(
    records: list[dict[str, Any]], registry: EntityRegistry
) -> dict[str, Any]:
    """Build a read-only plan and collect conflicts before mutation."""
    reviewed_registry = EntityRegistry(registry.to_dict())
    conflicts: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    by_entity_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    snapshots: dict[str, tuple[Any, ...]] = {}

    records = sorted(
        records,
        key=lambda row: (
            next((label for label in LABELS if label in (row.get("labels") or [])), ""),
            comparison_key((row.get("properties") or {}).get("name", "")),
            clean_text((row.get("properties") or {}).get("name", "")).casefold(),
            clean_text((row.get("properties") or {}).get("name", "")),
            str(row.get("node_id", "")),
        ),
    )
    for row in records:
        labels = row.get("labels", [])
        matched_labels = [label for label in LABELS if label in labels]
        if len(matched_labels) != 1 or set(labels) != {matched_labels[0]}:
            conflicts.append(
                {
                    "node_id": row.get("node_id"),
                    "reason": "entity node has unexpected or ambiguous labels",
                }
            )
            continue
        label = matched_labels[0]
        properties = row.get("properties") or {}
        name = properties.get("name")
        if not isinstance(name, str) or not name.strip():
            conflicts.append({"node_id": row.get("node_id"), "reason": "missing name property"})
            continue
        raw_aliases = properties.get("aliases", [])
        if not isinstance(raw_aliases, list) or any(
            not isinstance(alias, str) or not clean_text(alias)
            for alias in raw_aliases
        ):
            conflicts.append(
                {
                    "node_id": row.get("node_id"),
                    "reason": "invalid aliases property; expected a list of non-empty strings",
                }
            )
            node_aliases = []
        else:
            node_aliases = list(dict.fromkeys(clean_text(alias) for alias in raw_aliases))
        entity = registry.resolve(label, name)
        existing_id = properties.get("entity_id")
        if existing_id is not None and existing_id != entity.entity_id:
            conflicts.append(
                {
                    "node_id": row.get("node_id"),
                    "reason": "existing entity_id disagrees with exact registry identity",
                }
            )
        existing_canonical = properties.get("canonical_name")
        if existing_canonical is not None and existing_canonical != entity.canonical_name:
            conflicts.append(
                {
                    "node_id": row.get("node_id"),
                    "reason": "existing canonical_name disagrees with exact registry identity",
                }
            )
        row = {
            **row,
            "label": label,
            "entity": entity,
            "name": name,
            "node_aliases": node_aliases,
        }
        by_entity_id[entity.entity_id].append(row)
        snapshots[row["node_id"]] = _record_snapshot(row)

    for entity_id, members in sorted(by_entity_id.items()):
        label = members[0]["label"]
        entity = registry.resolve(label, members[0]["name"])
        member_ids = sorted(member["node_id"] for member in members)
        survivor_id = member_ids[0]
        aliases = list(entity.aliases)
        aliases.extend(member["name"] for member in members)
        aliases.extend(
            alias for member in members for alias in member["node_aliases"]
        )
        aliases = list(dict.fromkeys(alias for alias in aliases if alias))

        if len(members) > 1:
            extra_properties = sorted(
                {
                    key
                    for member in members
                    for key in member["properties"]
                    if key not in MANAGED_PROPERTIES
                }
            )
            if extra_properties:
                conflicts.append(
                    {
                        "entity_id": entity_id,
                        "reason": "duplicate nodes carry unrecognized properties",
                        "property_names": extra_properties,
                    }
                )
            for member in members:
                for relation in member.get("relations") or []:
                    if relation is None:
                        continue
                    reason = _relation_error(relation)
                    if reason:
                        conflicts.append(
                            {
                                "entity_id": entity_id,
                                "node_id": member["node_id"],
                                "reason": reason,
                            }
                        )

        groups[entity_id] = {
            "label": label,
            "entity_id": entity_id,
            "canonical_name": entity.canonical_name,
            "identity_origin": entity.identity_origin,
            "aliases": aliases,
            "historical_aliases": list(
                dict.fromkeys(
                    alias
                    for member in members
                    for alias in member["node_aliases"]
                )
            ),
            "node_ids": member_ids,
            "survivor_id": survivor_id,
            "merge_node_ids": member_ids[1:],
            "_snapshot": {member_id: snapshots[member_id] for member_id in member_ids},
            "relationship_transfers": sum(
                len([relation for relation in member.get("relations") or [] if relation])
                for member in members[1:]
            ),
        }

    graph_alias_owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    for group in groups.values():
        members = by_entity_id[group["entity_id"]]
        for member in members:
            claimed_aliases = [member["name"], *member["node_aliases"]]
            for alias in claimed_aliases:
                graph_alias_owners[(group["label"], comparison_key(alias))].add(
                    group["entity_id"]
                )

    for group in groups.values():
        for member in by_entity_id[group["entity_id"]]:
            for alias in member["node_aliases"]:
                registered = reviewed_registry.lookup(group["label"], alias)
                direct_name = comparison_key(alias) in {
                    comparison_key(member["name"]), comparison_key(group["canonical_name"])
                }
                if registered is None and not direct_name:
                    conflicts.append(
                        {
                            "entity_id": group["entity_id"],
                            "node_id": member["node_id"],
                            "alias": alias,
                            "reason": (
                                "historical alias is not in the reviewed registry; "
                                "add it to the correct registry entity before applying"
                            ),
                        }
                    )
                elif registered is not None and registered.entity_id != group["entity_id"]:
                    conflicts.append(
                        {
                            "entity_id": group["entity_id"],
                            "node_id": member["node_id"],
                            "alias": alias,
                            "registered_entity_id": registered.entity_id,
                            "reason": "historical alias belongs to a different registry entity",
                        }
                    )
                other_owners = graph_alias_owners[
                    (group["label"], comparison_key(alias))
                ] - {group["entity_id"]}
                if other_owners:
                    conflicts.append(
                        {
                            "entity_id": group["entity_id"],
                            "node_id": member["node_id"],
                            "alias": alias,
                            "conflicting_entity_ids": sorted(other_owners),
                            "reason": "historical alias is claimed by another graph entity",
                        }
                    )

    return {
        "valid": not conflicts,
        "nodes_seen": len(records),
        "entities_after_migration": len(groups),
        "nodes_to_merge": sum(len(group["merge_node_ids"]) for group in groups.values()),
        "relationship_transfers": sum(group["relationship_transfers"] for group in groups.values()),
        "groups": list(groups.values()),
        "conflicts": conflicts,
    }


def _relationship_transfer_query(label: str, relationship: str, direction: str) -> str:
    if relationship not in OWNED_TYPES or label not in LABELS:
        raise ValueError("Unsupported migration relationship layout.")
    other_label = "Paper"
    if direction == "outgoing":
        pattern = f"(old:{label})-[r:{relationship}]->(other:{other_label})"
        create = f"(survivor)-[new:{relationship}]->(other)"
    else:
        pattern = f"(other:{other_label})-[r:{relationship}]->(old:{label})"
        create = f"(other)-[new:{relationship}]->(survivor)"
    return f"""MATCH {pattern}
    WHERE elementId(old) IN $old_ids
    WITH old, r, other, properties(r) AS rel_properties
    MATCH (survivor:{label}) WHERE elementId(survivor) = $survivor_id
    CREATE {create}
    SET new = rel_properties
    DELETE r"""


def _apply_data_migration(
    tx: Any,
    groups: list[dict[str, Any]],
    registry_data: dict[str, Any],
    expected_snapshots: dict[str, dict[str, tuple[Any, ...]]],
) -> None:
    fresh_plan = build_migration_plan(
        _records(tx), EntityRegistry(registry_data)
    )
    if not fresh_plan["valid"]:
        raise RuntimeError("Migration conflicts changed after preview.")
    fresh_groups = {group["entity_id"]: group for group in fresh_plan["groups"]}
    for expected_group in groups:
        current = fresh_groups.get(expected_group["entity_id"])
        if current is None or current["node_ids"] != expected_group["node_ids"]:
            raise RuntimeError("Migration nodes changed after preview.")
        for node_id, snapshot in expected_snapshots[expected_group["entity_id"]].items():
            if current["_snapshot"].get(node_id) != snapshot:
                raise RuntimeError("Migration properties or relationships changed after preview.")

    for group in groups:
        old_ids = group["merge_node_ids"]
        if old_ids:
            for relationship, (start_label, end_label, direction) in EDGE_LAYOUTS.items():
                if group["label"] == start_label:
                    query = _relationship_transfer_query(group["label"], relationship, "outgoing")
                elif group["label"] == end_label:
                    query = _relationship_transfer_query(group["label"], relationship, "incoming")
                else:
                    continue
                tx.run(
                    query,
                    old_ids=old_ids,
                    survivor_id=group["survivor_id"],
                ).consume()
            tx.run(
                "MATCH (old) WHERE elementId(old) IN $old_ids DELETE old",
                old_ids=old_ids,
            ).consume()
        tx.run(
            "MATCH (n) WHERE elementId(n) = $survivor_id "
            "SET n.entity_id=$entity_id, n.name=$canonical_name, "
            "n.canonical_name=$canonical_name, "
            "n.aliases=reduce(acc=coalesce(n.aliases, []), alias IN $aliases | "
            "CASE WHEN alias IN acc THEN acc ELSE acc + alias END), "
            "n.identity_origin=coalesce(n.identity_origin, $identity_origin)",
            survivor_id=group["survivor_id"],
            entity_id=group["entity_id"],
            canonical_name=group["canonical_name"],
            aliases=group["aliases"],
            identity_origin=group["identity_origin"],
        ).consume()


def _schema_migration(driver: Any, database: str) -> None:
    with driver.session(database=database) as session:
        for label, constraint_name in NEW_CONSTRAINTS.items():
            session.run(
                f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.entity_id IS UNIQUE"
            ).consume()
        for constraint_name in OLD_CONSTRAINTS:
            session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS").consume()


def migrate_entities(
    driver: Any,
    database: str,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    registry = EntityRegistry.load(registry_path)
    with driver.session(database=database) as session:
        records = _records(session)
    plan = build_migration_plan(records, registry)
    report = {
        key: value for key, value in plan.items() if key != "groups"
    }
    report["changes"] = [
        {key: value for key, value in group.items() if not key.startswith("_") and key != "node_ids"}
        for group in plan["groups"]
    ]
    report["mode"] = "apply" if apply else "dry_run"
    if not plan["valid"] or not apply:
        return report

    expected_snapshots = {
        group["entity_id"]: group["_snapshot"] for group in plan["groups"]
    }
    registry_data = registry.to_dict()
    with driver.session(database=database) as session:
        session.execute_write(
            _apply_data_migration,
            plan["groups"],
            registry_data,
            expected_snapshots,
        )
    _schema_migration(driver, database)
    # Registry additions are deterministic and are only persisted after DB success.
    registry.save(registry_path)
    report["applied"] = True
    return report


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--apply", action="store_true", help="Apply after a conflict-free preview.")
    args = parser.parse_args()

    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
    if any(not os.getenv(name) for name in required):
        raise SystemExit("Neo4j configuration is incomplete.")
    driver = None
    try:
        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
        driver.verify_connectivity()
        report = migrate_entities(
            driver, args.database, args.registry, apply=args.apply
        )
    except Exception as exc:
        report = {
            "valid": False,
            "mode": "apply" if args.apply else "dry_run",
            "conflicts": [{"reason": f"migration failed ({type(exc).__name__})"}],
        }
    finally:
        if driver is not None:
            driver.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("valid"):
        raise SystemExit(1)
    if args.apply and not report.get("applied"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
