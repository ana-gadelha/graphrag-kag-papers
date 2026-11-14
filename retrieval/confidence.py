"""Heurística de convergência dos canais nas fontes efetivamente citadas."""

import re


CONFIDENCE_METHOD = "cited-channel-agreement-v1"


def assess_confidence(answer: str, sources: list[dict], *, abstained: bool) -> dict:
    """Classifica as evidências; não estima probabilidade de correção da resposta."""
    cited_ids = set(re.findall(r"\[(S\d+)\]", answer))
    cited = {source["citation_id"]: source for source in sources
             if source.get("citation_id") in cited_ids}
    dual_count = sum(
        {"vector", "graph"}.issubset(source.get("retrieved_by", []))
        for source in cited.values()
    )
    if abstained:
        level = "low"
        reason = "O sistema se absteve por insuficiência de evidências."
    elif len(cited) >= 2 and dual_count == len(cited):
        level = "high"
        reason = (f"As {len(cited)} fontes distintas citadas foram recuperadas "
                  "pelos canais vetorial e grafo.")
    elif dual_count or len(cited) >= 2:
        level = "medium"
        reason = (f"{len(cited)} fonte(s) distinta(s) citada(s); "
                  f"{dual_count} recuperada(s) pelos canais vetorial e grafo.")
    else:
        level = "low"
        reason = ("Há menos de duas fontes distintas citadas e nenhuma delas "
                  "foi recuperada pelos dois canais.")
    return {"level": level, "method": CONFIDENCE_METHOD, "reason": reason}
