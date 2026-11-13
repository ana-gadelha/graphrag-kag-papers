from __future__ import annotations

import argparse
import subprocess
import sys


def run(*args: str) -> None:
    print("\n>", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def run_pipeline(
    provider: str = "curated",
    *,
    import_neo4j: bool = False,
    index_embeddings: bool = False,
) -> None:
    if index_embeddings and not import_neo4j:
        raise ValueError("--index-embeddings requer --import-neo4j")
    python = sys.executable
    run(python, "-m", "src.validate_data")
    run(python, "-m", "src.extract_entities", "--provider", provider)
    if import_neo4j:
        run(python, "-m", "src.import_graph")
        run(python, "-m", "src.validate_graph")
        if index_embeddings:
            run(python, "-m", "retrieval.index_embeddings")
            run(python, "-m", "src.validate_graph", "--check-embeddings")
    else:
        print(
            "\nDados preparados. Revise output/papers_enriched.json. "
            "Quando o banco estiver acessível, use --import-neo4j."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o pipeline Text-to-Graph.")
    parser.add_argument(
        "--provider", choices=("curated", "heuristic", "openai"), default="curated"
    )
    parser.add_argument("--import-neo4j", action="store_true")
    parser.add_argument("--index-embeddings", action="store_true")
    args = parser.parse_args()
    if args.index_embeddings and not args.import_neo4j:
        parser.error("--index-embeddings requer --import-neo4j")
    run_pipeline(
        args.provider,
        import_neo4j=args.import_neo4j,
        index_embeddings=args.index_embeddings,
    )


if __name__ == "__main__":
    main()
