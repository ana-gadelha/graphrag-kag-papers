"""Deterministic, exact-match resolution for graph entities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import clean_text, comparison_key


DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data/entity_registry.json"
ENTITY_LABELS = {"Author", "Concept", "Method", "Dataset"}


def entity_id_for(label: str, canonical_name: str) -> str:
    """Return a stable ID derived only from the label and normalized name."""
    if label not in ENTITY_LABELS:
        raise ValueError(f"Unsupported entity label: {label}")
    normalized = comparison_key(canonical_name)
    if not normalized:
        raise ValueError("Entity canonical name cannot be empty.")
    digest = hashlib.sha256(f"{label}\0{normalized}".encode("utf-8")).hexdigest()[:24]
    return f"{label.casefold()}:{digest}"


@dataclass(frozen=True, slots=True)
class Entity:
    entity_id: str
    label: str
    canonical_name: str
    aliases: tuple[str, ...]
    identity_origin: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "identity_origin": self.identity_origin,
        }


class EntityRegistry:
    """Resolve exact normalized aliases and retain dynamically observed names."""

    def __init__(self, data: dict[str, Any], *, path: Path | None = None) -> None:
        if data.get("version") != 1 or not isinstance(data.get("entities"), list):
            raise ValueError("Entity registry must use version 1 and contain entities.")
        self.path = path
        self.identity_policy = data.get("identity_policy", {})
        self._entities: dict[str, Entity] = {}
        self._aliases: dict[tuple[str, str], str] = {}
        for raw in data["entities"]:
            self._add_loaded_entity(raw)

    @classmethod
    def load(cls, path: Path = DEFAULT_REGISTRY_PATH) -> "EntityRegistry":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Entity registry must be a JSON object.")
        return cls(data, path=path)

    def _add_loaded_entity(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            raise ValueError("Every entity registry entry must be an object.")
        label = raw.get("label")
        canonical = clean_text(raw.get("canonical_name", ""))
        aliases = raw.get("aliases", [])
        identity_origin = raw.get("identity_origin", "reviewed")
        if (
            label not in ENTITY_LABELS
            or not canonical
            or not isinstance(aliases, list)
            or not all(isinstance(alias, str) for alias in aliases)
            or not isinstance(identity_origin, str)
            or not identity_origin
        ):
            raise ValueError("Invalid entity registry entry.")
        clean_aliases = tuple(dict.fromkeys([canonical, *(clean_text(a) for a in aliases)]))
        expected_id = entity_id_for(label, canonical)
        entity_id = raw.get("entity_id", expected_id)
        if entity_id != expected_id:
            raise ValueError(f"Entity ID does not match canonical identity: {canonical}")
        if entity_id in self._entities:
            raise ValueError(f"Duplicate entity ID in registry: {entity_id}")
        entity = Entity(entity_id, label, canonical, clean_aliases, identity_origin)
        self._entities[entity_id] = entity
        for alias in clean_aliases:
            key = (label, comparison_key(alias))
            previous = self._aliases.get(key)
            if previous is not None and previous != entity_id:
                raise ValueError(
                    f"Ambiguous {label} alias in entity registry: {alias}"
                )
            self._aliases[key] = entity_id

    def resolve(self, label: str, surface_form: str) -> Entity:
        """Resolve a known alias or register an exact observed name as new."""
        if label not in ENTITY_LABELS:
            raise ValueError(f"Unsupported entity label: {label}")
        surface = clean_text(surface_form)
        key = comparison_key(surface)
        if not key:
            raise ValueError("Entity surface form cannot be empty.")
        existing_id = self._aliases.get((label, key))
        if existing_id is not None:
            existing = self._entities[existing_id]
            if existing.identity_origin == "reviewed_seed" or surface in existing.aliases:
                return existing
            aliases = (*existing.aliases, surface)
            updated = Entity(
                existing.entity_id,
                existing.label,
                existing.canonical_name,
                aliases,
                existing.identity_origin,
            )
            self._entities[existing_id] = updated
            for alias in aliases:
                self._aliases[(label, comparison_key(alias))] = existing_id
            return updated

        entity = Entity(
            entity_id=entity_id_for(label, surface),
            label=label,
            canonical_name=surface,
            aliases=(surface,),
            identity_origin=(
                "name_only_metadata" if label == "Author" else "observed_exact_name"
            ),
        )
        self._entities[entity.entity_id] = entity
        self._aliases[(label, key)] = entity.entity_id
        return entity

    def lookup(self, label: str, surface_form: str) -> Entity | None:
        """Look up an already registered exact alias without registering new names."""
        if label not in ENTITY_LABELS:
            raise ValueError(f"Unsupported entity label: {label}")
        key = comparison_key(surface_form)
        if not key:
            return None
        entity_id = self._aliases.get((label, key))
        return self._entities.get(entity_id) if entity_id is not None else None

    def to_dict(self) -> dict[str, Any]:
        entities = sorted(
            (entity.as_dict() for entity in self._entities.values()),
            key=lambda item: (item["label"], comparison_key(item["canonical_name"])),
        )
        return {
            "version": 1,
            "identity_policy": self.identity_policy,
            "entities": entities,
        }

    def save(self, path: Path | None = None) -> None:
        destination = path or self.path
        if destination is None:
            raise ValueError("Entity registry path is not configured.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def entities(self) -> list[Entity]:
        return list(self._entities.values())


def resolve_metadata_entities(
    paper: dict[str, Any], registry: EntityRegistry
) -> list[dict[str, Any]]:
    """Resolve authors and keywords while retaining their metadata origin."""
    resolved: list[dict[str, Any]] = []
    for field, label, relationship in (
        ("authors", "Author", "AUTHORED"),
        ("keywords", "Concept", "HAS_CONCEPT"),
    ):
        for surface in paper[field]:
            entity = registry.resolve(label, surface)
            resolved.append(
                {
                    **entity.as_dict(),
                    "surface_form": surface,
                    "relation": relationship,
                    "field": field,
                    "origin": "paper_metadata",
                }
            )
    return resolved
