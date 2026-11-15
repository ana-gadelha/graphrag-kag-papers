"""Local-only browser fixture for the Cytoscape component; it never calls retrieval."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_component import build_graph_payload, render_graph, selected_graph_record

st.set_page_config(page_title="Graph component fixture", layout="wide")
st.title("Graph component browser fixture")
st.caption("Synthetic data only. This page does not connect to Neo4j or an LLM.")

nodes = [
    {
        "id": "paper:alpha",
        "label": "GraphRAG paper",
        "tipo": "Paper",
        "properties": {
            "title": "GraphRAG paper",
            "year": 2024,
            "doi": "10.0000/example",
        },
        "source_ids": ["S1"],
        "provenance": {"citation_id": "S1", "source": "fixture"},
    },
    {
        "id": "paper:connector",
        "label": "Connector paper",
        "tipo": "Paper",
        "connector": True,
        "properties": {"title": "Connector paper", "year": 2023},
        "source_ids": [],
        "provenance": {"kind": "connector"},
    },
    {
        "id": "concept:rag",
        "label": "Graph RAG",
        "tipo": "Concept",
        "properties": {"name": "Graph RAG", "entity_id": "concept:rag"},
        "source_ids": ["S1", "S2"],
    },
    {
        "id": "paper:beta",
        "label": "KAG paper",
        "tipo": "Paper",
        "properties": {"title": "KAG paper", "year": 2025},
        "source_ids": ["S2"],
    },
]
edges = [
    {
        "id": "edge:alpha-concept",
        "origem": "paper:alpha",
        "destino": "concept:rag",
        "relacao": "HAS_CONCEPT",
        "properties": {"confidence": 0.98},
        "source_ids": ["S1"],
        "provenance": {"citation_id": "S1", "field": "concepts"},
    },
    {
        "id": "edge:concept-beta",
        "origem": "concept:rag",
        "destino": "paper:beta",
        "relacao": "USES_METHOD",
        "properties": {"confidence": 0.86},
        "source_ids": ["S2"],
        "provenance": {"citation_id": "S2", "field": "methods"},
    },
    {
        "id": "edge:connector-concept",
        "origem": "paper:connector",
        "destino": "concept:rag",
        "relacao": "DISCUSSES",
        "properties": {},
        "source_ids": ["S2"],
    },
]
paths = [
    {
        "path_id": "P1",
        "kind": "shortest",
        "source": "S1",
        "target": "S2",
        "length": 2,
        "nodes": [nodes[0], nodes[2], nodes[3]],
        "edges": edges[:2],
        "edge_ids": ["edge:alpha-concept", "edge:concept-beta"],
    },
    *[
        {
            "path_id": f"P{position}",
            "kind": "fact",
            "source": "S1",
            "target": "S1",
            "length": 1,
            "nodes": [nodes[0], nodes[2]],
            "edges": [edges[0]],
            "edge_ids": [edges[0]["id"]],
        }
        for position in range(2, 102)
    ],
]
claims = [
    {
        "kind": "relationship",
        "text": "Os artigos compartilham o conceito Graph RAG.",
        "source_ids": ["S1", "S2"],
        "path_ids": ["P1"],
        "quotes": ["synthetic evidence quote"],
    }
]

payload = build_graph_payload(nodes, edges, paths, claims)
state = render_graph(payload)
selection = selected_graph_record(payload, state.get("selection"))
if selection:
    st.caption(
        f"Seleção recebida pelo Streamlit: {selection['item_type']} · {selection['id']}"
    )
