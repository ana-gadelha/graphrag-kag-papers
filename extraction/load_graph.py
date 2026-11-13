"""Prepara o corpus, carrega no Neo4j e executa a conferência final.

Uso, a partir da raiz do projeto:
    python extraction/load_graph.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from run_pipeline import run_pipeline  # noqa: E402

if __name__ == "__main__":
    run_pipeline(import_neo4j=True, index_embeddings=True)
