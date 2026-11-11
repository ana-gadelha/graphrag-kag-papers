"""Interface de linha de comando para perguntas ao corpus."""

from __future__ import annotations

import argparse
import json
import sys

from .answer import answer_question
from .config import RetrievalError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Responde perguntas usando artigos recuperados no Neo4j."
    )
    parser.add_argument("question", help="Pergunta em linguagem natural")
    parser.add_argument(
        "--top-k", type=int, default=6,
        help="Quantidade máxima de artigos recuperados (1–20; padrão: 6)",
    )
    args = parser.parse_args(argv)

    try:
        result = answer_question(args.question, top_k=args.top_k)
    except RetrievalError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("Falha ao responder à pergunta.", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
