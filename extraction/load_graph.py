"""Prepara o corpus, carrega no Neo4j e executa a conferência final.

Uso, a partir da raiz do projeto:
    python extraction/load_graph.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.import_graph import main as import_main  # noqa: E402
from src.validate_graph import main as validate_main  # noqa: E402


if __name__ == "__main__":
    print("1/4 - Validando e normalizando o corpus...")
    subprocess.run([sys.executable, "-m", "src.validate_data"], check=True)
    print("\n2/4 - Aplicando a extração curada de métodos e datasets...")
    subprocess.run(
        [sys.executable, "-m", "src.extract_entities", "--provider", "curated"],
        check=True,
    )
    print("\n3/4 - Aplicando o schema e carregando o Neo4j...")
    import_main()
    print("\n4/4 - Conferindo o grafo:")
    validate_main()
