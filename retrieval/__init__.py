"""Recuperação e resposta com evidências do grafo de papers."""

from __future__ import annotations

from typing import Any

__all__ = ["answer_question"]


def answer_question(question: str, *, top_k: int = 6) -> dict[str, Any]:
    """Importa a implementação apenas quando a API é chamada."""
    from .answer import answer_question as _answer_question

    return _answer_question(question, top_k=top_k)
