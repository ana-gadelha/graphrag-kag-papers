from __future__ import annotations

import argparse
import subprocess
import sys


def run(*args: str) -> None:
    print("\n>", " ".join(args))
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o pipeline Text-to-Graph.")
    parser.add_argument(
        "--provider", choices=("curated", "heuristic", "openai"), default="curated"
    )
    parser.add_argument(
        "--import-neo4j",
        action="store_true",
        help="Após preparar os dados, conecta e importa no Neo4j.",
    )
    args = parser.parse_args()

    python = sys.executable
    run(python, "-m", "src.validate_data")
    run(python, "-m", "src.extract_entities", "--provider", args.provider)
    if args.import_neo4j:
        run(python, "-m", "src.import_graph")
        run(python, "-m", "src.validate_graph")
    else:
        print(
            "\nDados preparados. Revise output/papers_enriched.json. "
            "Quando o banco estiver acessível, use --import-neo4j."
        )


if __name__ == "__main__":
    main()
