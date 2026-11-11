"""Testa as credenciais do Neo4j sem alterar o banco."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402


def main() -> None:
    load_dotenv()
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise SystemExit(f"Variáveis ausentes no .env: {', '.join(missing)}")

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        server = driver.get_server_info()
        print(f"Conexão realizada com sucesso: {server.address}")
        print(f"Agente do servidor: {server.agent}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
