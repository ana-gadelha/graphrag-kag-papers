"""Register and render the local Streamlit components.v2 graph viewer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.components.v2 import component

from .data import graph_payload_fingerprint

_ASSET_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _graph_component():
    """Create one component definition; all browser code is bundled locally."""
    vendor = (_ASSET_DIR / "vendor" / "cytoscape.min.js").read_text(encoding="utf-8")
    viewer = (_ASSET_DIR / "viewer.js").read_text(encoding="utf-8")
    return component(
        "interactive_evidence_graph",
        html=(_ASSET_DIR / "index.html").read_text(encoding="utf-8"),
        css=(_ASSET_DIR / "viewer.css").read_text(encoding="utf-8"),
        js=f"{vendor}\n{viewer}",
        # Keep Cytoscape's canvas in the document DOM. Streamlit's isolated
        # shadow root retargets native mouse events before Cytoscape handles
        # them, which prevents node/edge taps and background panning.
        isolate_styles=False,
    )


def render_graph(payload: dict[str, Any]):
    """Mount the graph and expose browser selection as persistent component state."""
    graph_viewer = _graph_component()
    fingerprint = graph_payload_fingerprint(payload)
    key = f"evidence-graph-{fingerprint}"
    state = st.session_state.get(key, {})
    selection = state.get("selection") if isinstance(state, dict) else None
    return graph_viewer(
        data={**payload, "fingerprint": fingerprint, "selection": selection},
        default={"selection": None},
        key=key,
        height=640,
        on_selection_change=lambda: None,
    )
