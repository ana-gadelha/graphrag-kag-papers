"""Prepare a small, credential-free graph payload for the browser component."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_SENSITIVE_KEY = re.compile(
    r"password|secret|api[_-]?key|token|authorization|credential|"
    r"connection[_-]?string|database[_-]?url|private[_-]?key",
    re.IGNORECASE,
)


def _safe_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return "[valor aninhado omitido]"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _safe_value(item, depth=depth + 1)
            for key, item in value.items()
            if not _SENSITIVE_KEY.search(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, depth=depth + 1) for item in value]
    return str(value)


def _text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _ids(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return []
    return [_text(item) for item in value if _text(item)]


def _node_identifier(node: dict[str, Any]) -> str:
    properties = (
        node.get("properties") if isinstance(node.get("properties"), dict) else {}
    )
    return _text(
        node.get("id"),
        _text(
            node.get("entity_id"),
            _text(
                properties.get("entity_id"), _text(properties.get("name"), "unknown")
            ),
        ),
    )


def _edge_signature(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _text(edge.get("source"), _text(edge.get("origem"), "unknown")),
        _text(edge.get("type"), _text(edge.get("relacao"), "RELATED_TO")),
        _text(edge.get("target"), _text(edge.get("destino"), "unknown")),
    )


def _edge_identifier(edge: dict[str, Any], signature: tuple[str, str, str]) -> str:
    explicit = _text(edge.get("id"), _text(edge.get("edge_id")))
    if explicit:
        return explicit
    digest = hashlib.sha256("\x1f".join(signature).encode("utf-8")).hexdigest()[:16]
    return f"edge:{digest}"


def _path_node_ids(path: dict[str, Any]) -> list[str]:
    raw = path.get("node_ids")
    if isinstance(raw, list):
        return [_text(item) for item in raw if _text(item)]
    nodes = path.get("nodes")
    if not isinstance(nodes, list):
        return []
    return [
        _node_identifier(node) if isinstance(node, dict) else _text(node)
        for node in nodes
    ]


def _path_edge_ids(
    path: dict[str, Any], edge_ids_by_signature: dict[tuple[str, str, str], list[str]]
) -> list[str]:
    raw = path.get("edge_ids")
    if isinstance(raw, list):
        return [_text(item) for item in raw if _text(item)]
    edges = path.get("edges")
    if not isinstance(edges, list):
        return []
    edge_ids: list[str] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        signature = _edge_signature(edge)
        explicit = _text(edge.get("id"), _text(edge.get("edge_id")))
        candidates = edge_ids_by_signature.get(signature, [])
        if explicit:
            edge_ids.append(explicit)
        elif candidates:
            edge_ids.append(candidates[0])
        else:
            edge_ids.append(_edge_identifier(edge, signature))
    return edge_ids


def build_graph_payload(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    paths: list[dict[str, Any]],
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build graph elements with only display data, never runtime settings."""
    graph_nodes: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        properties = (
            node.get("properties") if isinstance(node.get("properties"), dict) else {}
        )
        graph_nodes.append(
            {
                "id": _node_identifier(node),
                "label": _text(node.get("label"), _node_identifier(node)),
                "type": _text(node.get("tipo"), "Unknown"),
                "connector": node.get("connector") is True,
                "properties": _safe_value(properties),
                "source_ids": _ids(
                    node.get("source_ids", properties.get("source_ids"))
                ),
                "path_ids": _ids(node.get("path_ids")),
                "provenance": _safe_value(node.get("provenance", {})),
            }
        )

    graph_edges: list[dict[str, Any]] = []
    edge_ids_by_signature: dict[tuple[str, str, str], list[str]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        signature = _edge_signature(edge)
        edge_id = _edge_identifier(edge, signature)
        edge_ids_by_signature.setdefault(signature, []).append(edge_id)
        properties = (
            edge.get("properties") if isinstance(edge.get("properties"), dict) else {}
        )
        graph_edges.append(
            {
                "id": edge_id,
                "source": signature[0],
                "target": signature[2],
                "type": signature[1],
                "properties": _safe_value(properties),
                "source_ids": _ids(
                    edge.get("source_ids", properties.get("source_ids"))
                ),
                "path_ids": _ids(edge.get("path_ids")),
                "provenance": _safe_value(edge.get("provenance", {})),
            }
        )

    graph_paths = []
    for position, path in enumerate(paths, start=1):
        if not isinstance(path, dict):
            continue
        graph_paths.append(
            {
                "path_id": _text(path.get("path_id"), f"P{position}"),
                "source": _text(path.get("source")),
                "target": _text(path.get("target")),
                "length": path.get("length"),
                "kind": _text(path.get("kind"), "shortest"),
                "node_ids": _path_node_ids(path),
                "edge_ids": _path_edge_ids(path, edge_ids_by_signature),
            }
        )

    graph_claims = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        graph_claims.append(
            {
                "kind": _text(claim.get("kind"), "Afirmação"),
                "text": _text(claim.get("text")),
                "source_ids": _ids(claim.get("source_ids")),
                "path_ids": _ids(claim.get("path_ids")),
                "quotes": _safe_value(claim.get("quotes", [])),
            }
        )

    return {
        "nodes": graph_nodes,
        "edges": graph_edges,
        "paths": graph_paths,
        "claims": graph_claims,
    }


def graph_payload_fingerprint(payload: dict[str, Any]) -> str:
    """Return a stable key so a new graph cannot inherit a stale selection."""
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def selected_graph_record(
    payload: dict[str, Any], selection: Any
) -> dict[str, Any] | None:
    """Resolve a browser selection back to a safe payload record for Streamlit."""
    if not isinstance(selection, dict):
        return None
    item_type = selection.get("type")
    identifier = _text(selection.get("id"))
    if item_type == "node":
        records = payload.get("nodes", [])
        key = "id"
    elif item_type == "edge":
        records = payload.get("edges", [])
        key = "id"
    else:
        return None
    for record in records:
        if isinstance(record, dict) and record.get(key) == identifier:
            return {"item_type": item_type, **record}
    return None
