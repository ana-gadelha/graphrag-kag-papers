"""Question and extraction annotation starters grounded in the local corpus.

All references created here are drafts. Human review must approve them before a
provider-backed run can treat them as benchmark gold.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

QUESTION_SCHEMA = "retrieval-question-bank-v1"
ANNOTATION_SCHEMA = "extraction-annotations-v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def corpus_hashes(root: Path) -> dict[str, str]:
    paths = {
        "papers": root / "data/papers.json",
        "curated_entities": root / "data/curated_entities.json",
        "entity_registry": root / "data/entity_registry.json",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def load_corpus(root: Path) -> tuple[list[dict], dict[str, dict]]:
    papers = json.loads((root / "data/papers.json").read_text(encoding="utf-8"))
    curated = json.loads(
        (root / "data/curated_entities.json").read_text(encoding="utf-8")
    )
    if not isinstance(papers, list) or not isinstance(curated, dict):
        raise ValueError("Corpus files must contain a papers list and entity object.")
    if any(
        not isinstance(paper, dict) or not isinstance(paper.get("id"), str)
        for paper in papers
    ):
        raise ValueError("Every paper must have a string id.")
    if len({paper["id"] for paper in papers}) != len(papers):
        raise ValueError("Paper IDs must be unique.")
    return papers, curated


def _paper_index(papers: list[dict]) -> dict[str, dict]:
    return {paper["id"]: paper for paper in papers}


def _quote(
    papers: dict[str, dict], paper_id: str, field: str, literal: str
) -> dict[str, str]:
    paper = papers[paper_id]
    value = paper.get(field)
    if field in {"authors", "keywords"}:
        found = isinstance(value, list) and literal in value
    else:
        found = isinstance(value, str) and literal in value
    if not found:
        raise ValueError(
            f"Reference draft quote is absent: {paper_id}.{field}: {literal!r}"
        )
    return {"paper_id": paper_id, "field": field, "quote": literal}


def _question(
    papers: dict[str, dict],
    question_id: str,
    split: str,
    category: str,
    text: str,
    answer: str,
    required_papers: list[str],
    evidence: list[tuple[str, str, str]],
    *,
    expected_abstention: bool = False,
    note: str = "",
) -> dict[str, Any]:
    return {
        "id": question_id,
        "split": split,
        "category": category,
        "question": text,
        "reference_draft": {
            "answer": answer,
            "required_papers": required_papers,
            "expected_abstention": expected_abstention,
            "evidence": [
                _quote(papers, paper_id, field, literal)
                for paper_id, field, literal in evidence
            ],
            "review_note": note
            or "Rascunho derivado do título, abstract e metadados; revisar semanticamente.",
        },
        "human_review": {"status": "PENDING", "reviewers": []},
    }


def draft_questions(papers_list: list[dict]) -> list[dict[str, Any]]:
    """Create 6 development and 16 frozen evaluation question drafts."""
    papers = _paper_index(papers_list)
    q = lambda *args, **kwargs: _question(papers, *args, **kwargs)
    questions = [
        q(
            "dev-direct-01",
            "dev",
            "direct",
            "No artigo 2405.14831, qual abordagem é introduzida e qual algoritmo de ranking ela integra?",
            "O artigo introduz HippoRAG e descreve que ele integra LLMs, grafos de conhecimento e Personalized PageRank.",
            ["2405.14831"],
            [
                ("2405.14831", "abstract", "In this work, we introduce HippoRAG"),
                ("2405.14831", "abstract", "the Personalized PageRank algorithm"),
            ],
        ),
        q(
            "dev-direct-02",
            "dev",
            "direct",
            "Quais benchmarks KAG nomeia ao relatar seus resultados de F1 em QA multihop?",
            "KAG relata melhorias relativas de F1 em 2wiki e hotpotQA.",
            ["2409.13731"],
            [
                (
                    "2409.13731",
                    "abstract",
                    "19.6% on 2wiki and 33.5% on hotpotQA in terms of F1 score",
                )
            ],
        ),
        q(
            "dev-multihop-01",
            "dev",
            "multi_hop",
            "O paper GraphSearch compartilha quais coautores com Think-on-Graph 2.0, e que workflow ele propõe?",
            "Os dois registros compartilham Cehao Yang, Chengjin Xu, Xuhui Jiang e Jian Guo; o paper 2509.22009 propõe GraphSearch.",
            ["2407.10805", "2509.22009"],
            [
                ("2407.10805", "authors", "Cehao Yang"),
                ("2407.10805", "authors", "Chengjin Xu"),
                ("2407.10805", "authors", "Xuhui Jiang"),
                ("2407.10805", "authors", "Jian Guo"),
                ("2509.22009", "abstract", "we propose \\textsc{GraphSearch}"),
            ],
        ),
        q(
            "dev-multihop-02",
            "dev",
            "multi_hop",
            "Qual coautor conecta o paper GFM-RAG ao paper Youtu-GraphRAG, e qual método o primeiro propõe?",
            "Linhao Luo aparece nos dois registros; 2502.01113 propõe GFM-RAG.",
            ["2502.01113", "2508.19855"],
            [
                ("2502.01113", "authors", "Linhao Luo"),
                ("2508.19855", "authors", "Linhao Luo"),
                ("2502.01113", "abstract", "we introduce GFM-RAG"),
            ],
        ),
        q(
            "dev-unsupported-01",
            "dev",
            "unsupported",
            "Usando somente o abstract e metadados do paper LightRAG (2410.05779), quais GPU e quantidade de VRAM foram usadas nos experimentos?",
            "Abster-se: o corpus disponível não informa GPU nem VRAM para os experimentos de LightRAG.",
            [],
            [
                (
                    "2410.05779",
                    "abstract",
                    "Extensive experimental validation demonstrates considerable improvements",
                )
            ],
            expected_abstention=True,
            note="A ausência precisa ser confirmada por revisores no escopo completo do registro local.",
        ),
        q(
            "dev-trap-01",
            "dev",
            "interpretation_trap",
            "No paper HippoRAG, IRCoT é uma contribuição proposta pelo artigo ou um método existente com que ele compara/integrar?",
            "A evidência classifica IRCoT como método existente comparado/integrado, não como contribuição proposta por HippoRAG.",
            ["2405.14831"],
            [
                (
                    "2405.14831",
                    "abstract",
                    "Single-step retrieval with HippoRAG achieves comparable or better performance than iterative retrieval like IRCoT",
                ),
                (
                    "2405.14831",
                    "abstract",
                    "integrating HippoRAG into IRCoT brings further substantial gains",
                ),
            ],
            note="Armadilha de relação: a menção explícita sustenta comparação/uso, não PROPOSES IRCoT.",
        ),
        q(
            "eval-direct-01",
            "eval",
            "direct",
            "Segundo o paper 2404.16130, que abordagem ele propõe e em quais duas etapas constrói seu índice?",
            "Propõe GraphRAG; primeiro deriva um grafo de conhecimento de entidades e depois pré-computa resumos de comunidades.",
            ["2404.16130"],
            [
                ("2404.16130", "abstract", "we propose GraphRAG"),
                (
                    "2404.16130",
                    "abstract",
                    "first, to derive an entity knowledge graph from the source documents, then to pregenerate community summaries",
                ),
            ],
        ),
        q(
            "eval-direct-02",
            "eval",
            "direct",
            "Que melhorias relativas de F1 KAG relata em 2wiki e hotpotQA?",
            "O abstract relata 19,6% em 2wiki e 33,5% em hotpotQA.",
            ["2409.13731"],
            [
                (
                    "2409.13731",
                    "abstract",
                    "achieving a relative improvement of 19.6% on 2wiki and 33.5% on hotpotQA in terms of F1 score",
                )
            ],
        ),
        q(
            "eval-intersection-01",
            "eval",
            "intersection",
            "Quais papers do corpus nomeiam HotpotQA nos resultados e qual abordagem cada um propõe?",
            "Os registros são 2409.13731, que introduz KAG, e 2502.06864, que propõe KG²RAG.",
            ["2409.13731", "2502.06864"],
            [
                (
                    "2409.13731",
                    "abstract",
                    "we introduce a professional domain knowledge service framework called Knowledge Augmented Generation (KAG)",
                ),
                ("2409.13731", "abstract", "hotpotQA in terms of F1 score"),
                (
                    "2502.06864",
                    "abstract",
                    "we propose a novel Knowledge Graph-Guided Retrieval Augmented Generation (KG$^2$RAG) framework",
                ),
                (
                    "2502.06864",
                    "abstract",
                    "Extensive experiments conducted on the HotpotQA dataset",
                ),
            ],
        ),
        q(
            "eval-intersection-02",
            "eval",
            "intersection",
            "PathRAG identifica redundância ou insuficiência como limitação central, e que mecanismos cita para tratá-la?",
            "O abstract aponta redundância na informação recuperada e descreve flow-based pruning junto com path-based prompting.",
            ["2502.14902"],
            [
                (
                    "2502.14902",
                    "abstract",
                    "the redundancy of the retrieved information, rather than its insufficiency",
                ),
                ("2502.14902", "abstract", "flow-based pruning"),
                ("2502.14902", "abstract", "path-based prompting"),
            ],
        ),
        q(
            "eval-multihop-01",
            "eval",
            "multi_hop",
            "Partindo do coautor Junnan Dong no registro 2501.13958, que método o outro paper do corpus propõe?",
            "O outro registro é 2508.19855 e propõe Youtu-GraphRAG.",
            ["2501.13958", "2508.19855"],
            [
                ("2501.13958", "authors", "Junnan Dong"),
                ("2508.19855", "authors", "Junnan Dong"),
                (
                    "2508.19855",
                    "abstract",
                    "we propose a vertically unified agentic paradigm, Youtu-GraphRAG",
                ),
            ],
        ),
        q(
            "eval-multihop-02",
            "eval",
            "multi_hop",
            "Qual método é proposto pelo paper do corpus ligado por coautoria ao survey 2501.13958, e qual coautor em comum permite esse percurso?",
            "O paper 2510.10114 propõe LinearRAG; Qinggang Zhang é coautor nos dois registros.",
            ["2501.13958", "2510.10114"],
            [
                ("2501.13958", "authors", "Qinggang Zhang"),
                ("2510.10114", "authors", "Qinggang Zhang"),
                (
                    "2510.10114",
                    "abstract",
                    "we revisit the pipeline of existing GraphRAG systems and propose LinearRAG",
                ),
            ],
        ),
        q(
            "eval-multihop-03",
            "eval",
            "multi_hop",
            "Qual workflow o paper ligado a Think-on-Graph 2.0 pela coautoria de Cehao Yang propõe, e qual outro autor em comum consta nos metadados?",
            "O paper 2509.22009 propõe GraphSearch; Chengjin Xu, Xuhui Jiang e Jian Guo também aparecem como coautores nos dois registros.",
            ["2407.10805", "2509.22009"],
            [
                ("2407.10805", "authors", "Cehao Yang"),
                ("2509.22009", "authors", "Cehao Yang"),
                ("2407.10805", "authors", "Chengjin Xu"),
                ("2509.22009", "authors", "Chengjin Xu"),
                ("2407.10805", "authors", "Xuhui Jiang"),
                ("2509.22009", "authors", "Xuhui Jiang"),
                ("2407.10805", "authors", "Jian Guo"),
                ("2509.22009", "authors", "Jian Guo"),
                ("2509.22009", "abstract", "we propose \\textsc{GraphSearch}"),
            ],
        ),
        q(
            "eval-multihop-04",
            "eval",
            "multi_hop",
            "Que método o paper 2501.15067 propõe, e qual método é proposto pelo paper do corpus conectado a ele pelo coautor Yuntong Hu?",
            "O registro 2501.15067 propõe CG-RAG/LeSeGR; 2405.16506 compartilha Yuntong Hu e propõe GRAG.",
            ["2501.15067", "2405.16506"],
            [
                ("2501.15067", "authors", "Yuntong Hu"),
                ("2405.16506", "authors", "Yuntong Hu"),
                (
                    "2501.15067",
                    "abstract",
                    "we introduce Contextualized Graph Retrieval-Augmented Generation (CG-RAG)",
                ),
                (
                    "2501.15067",
                    "abstract",
                    "Next, we introduce Lexical-Semantic Graph Retrieval (LeSeGR)",
                ),
                (
                    "2405.16506",
                    "abstract",
                    "we introduce Graph Retrieval-Augmented Generation (GRAG)",
                ),
            ],
        ),
        q(
            "eval-multihop-05",
            "eval",
            "multi_hop",
            "Quais métodos são propostos pelos dois papers ligados pela coautoria de Zirui Guo, e qual registro propõe cada método?",
            "Zirui Guo conecta 2410.05779, que propõe LightRAG, a 2502.14902, que propõe PathRAG.",
            ["2410.05779", "2502.14902"],
            [
                ("2410.05779", "authors", "Zirui Guo"),
                ("2502.14902", "authors", "Zirui Guo"),
                ("2410.05779", "abstract", "we propose LightRAG"),
                ("2502.14902", "abstract", "we propose PathRAG"),
            ],
        ),
        q(
            "eval-multihop-06",
            "eval",
            "multi_hop",
            "Que dois papers são conectados pelo benchmark HotpotQA, e que método cada paper apresenta?",
            "HotpotQA conecta 2409.13731 (KAG) a 2502.06864 (KG²RAG).",
            ["2409.13731", "2502.06864"],
            [
                ("2409.13731", "abstract", "Knowledge Augmented Generation (KAG)"),
                ("2409.13731", "abstract", "hotpotQA"),
                (
                    "2502.06864",
                    "abstract",
                    "Knowledge Graph-Guided Retrieval Augmented Generation (KG$^2$RAG)",
                ),
                ("2502.06864", "abstract", "HotpotQA dataset"),
            ],
        ),
        q(
            "eval-unsupported-01",
            "eval",
            "unsupported",
            "Usando somente o registro local de PathRAG (2502.14902), quais são os nomes exatos dos seis datasets citados?",
            "Abster-se: o abstract informa que os experimentos cobrem seis datasets, mas não os nomeia.",
            [],
            [
                (
                    "2502.14902",
                    "abstract",
                    "across six datasets and five evaluation dimensions",
                )
            ],
            expected_abstention=True,
            note="A não identificação dos seis nomes no abstract deve ser confirmada por revisores.",
        ),
        q(
            "eval-unsupported-02",
            "eval",
            "unsupported",
            "Qual p-valor e intervalo de confiança MedRAG relata para a redução de erros diagnósticos?",
            "Abster-se: o registro local não fornece p-valor ou intervalo de confiança.",
            [],
            [
                (
                    "2502.04413",
                    "abstract",
                    "outperforms state-of-the-art models in reducing misdiagnosis rates",
                )
            ],
            expected_abstention=True,
        ),
        q(
            "eval-unsupported-03",
            "eval",
            "unsupported",
            "Qual é a VRAM da GPU usada nos experimentos de LinearRAG, segundo o abstract local?",
            "Abster-se: o abstract não identifica hardware ou VRAM.",
            [],
            [("2510.10114", "abstract", "Extensive experiments on four datasets")],
            expected_abstention=True,
        ),
        q(
            "eval-unsupported-04",
            "eval",
            "unsupported",
            "Qual linguagem de programação é usada na implementação de KET-RAG, segundo o registro local?",
            "Abster-se: o registro menciona resultados e comparação, mas não informa a linguagem da implementação.",
            [],
            [
                (
                    "2502.09304",
                    "abstract",
                    "We evaluate 13 solutions on three real-world datasets",
                )
            ],
            expected_abstention=True,
        ),
        q(
            "eval-trap-01",
            "eval",
            "interpretation_trap",
            "No paper HippoRAG, IRCoT deve ser anotado como PROPOSES ou como USES_METHOD? Qual trecho distingue a relação?",
            "IRCoT é USES_METHOD: HippoRAG é comparado com retrieval iterativo como IRCoT e também é integrado a IRCoT.",
            ["2405.14831"],
            [
                (
                    "2405.14831",
                    "abstract",
                    "comparable or better performance than iterative retrieval like IRCoT",
                ),
                ("2405.14831", "abstract", "integrating HippoRAG into IRCoT"),
            ],
            note="Usar a taxonomia: comparação/integração sustenta USES_METHOD; não sustenta que HippoRAG propôs IRCoT.",
        ),
        q(
            "eval-trap-02",
            "eval",
            "interpretation_trap",
            "No paper KET-RAG (2502.09304), Graph-RAG da Microsoft é o método proposto pelo artigo ou um método existente usado como comparação?",
            "É um método existente usado como comparação; KET-RAG é o framework proposto.",
            ["2502.09304"],
            [
                (
                    "2502.09304",
                    "abstract",
                    "we propose KET-RAG, a multi-granular indexing framework",
                ),
                (
                    "2502.09304",
                    "abstract",
                    "comparable or superior retrieval quality to Microsoft's Graph-RAG",
                ),
            ],
            note="Usar a taxonomia: KET-RAG é PROPOSES; Microsoft Graph-RAG é referência comparativa/USES_METHOD.",
        ),
    ]
    return questions


def make_question_bank(root: Path) -> dict[str, Any]:
    papers, _ = load_corpus(root)
    return {
        "schema_version": QUESTION_SCHEMA,
        "source_hashes": corpus_hashes(root),
        "review_policy": "Every reference is a draft until two independent human reviewers approve it.",
        "questions": draft_questions(papers),
    }


def _normal_key(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    return "".join(char for char in decomposed if char.isalnum())


def _find_surface(source: str, candidate: str) -> tuple[str, int, int] | None:
    normalized: list[str] = []
    offsets: list[tuple[int, int]] = []
    for index, char in enumerate(source):
        for normalized_char in _normal_key(char):
            normalized.append(normalized_char)
            offsets.append((index, index + 1))
    target = _normal_key(candidate)
    if not target:
        return None
    haystack = "".join(normalized)
    start = haystack.find(target)
    if start < 0:
        return None
    end = start + len(target)
    return (
        source[offsets[start][0] : offsets[end - 1][1]],
        offsets[start][0],
        offsets[end - 1][1],
    )


_PROPOSAL_CUE = re.compile(
    r"\b(?:propos\w*|present\w*|introduc\w*|develop\w*|design\w*|"
    r"our\s+(?:method|approach|framework|system|model)|new\s+"
    r"(?:method|approach|framework|system|model))\b",
    re.IGNORECASE,
)
_USAGE_CUE = re.compile(
    r"\b(?:us(?:e|es|ed|ing)|adopt\w*|compar\w*|baseline\w*|"
    r"benchmark\w*|evaluat\w*|experiment\w*)\b",
    re.IGNORECASE,
)
_DATASET_CUE = re.compile(
    r"\b(?:dataset\w*|benchmark\w*|evaluat\w*|test\w*|experiment\w*|"
    r"results?\s+on|performance\s+on)\b",
    re.IGNORECASE,
)


def _sentence_quote(source: str, start: int, end: int) -> str:
    left = max(source.rfind(mark, 0, start) for mark in ".!?\n") + 1
    right_candidates = [
        position + 1 for mark in ".!?\n" if (position := source.find(mark, end)) >= 0
    ]
    right = min(right_candidates) if right_candidates else len(source)
    return source[left:right].strip()


def _candidate_mentions(paper: dict, draft: dict) -> list[dict[str, Any]]:
    candidates: list[dict[str, str]] = []
    for entity_field, relation, label in (
        ("proposed_methods", "PROPOSES", "Method"),
        ("used_methods", "USES_METHOD", "Method"),
        ("datasets", "EVALUATED_ON", "Dataset"),
    ):
        for candidate in draft.get(entity_field, []):
            candidates.append(
                {"candidate": candidate, "relation": relation, "label": label}
            )

    mentions = []
    for item in candidates:
        located = None
        for source_field in ("title", "abstract"):
            source = paper.get(source_field, "")
            found = _find_surface(source, item["candidate"])
            if found:
                surface, start, end = found
                quote = _sentence_quote(source, start, end)
                cue = {
                    "PROPOSES": _PROPOSAL_CUE,
                    "USES_METHOD": _USAGE_CUE,
                    "EVALUATED_ON": _DATASET_CUE,
                }[item["relation"]]
                located = {
                    "surface_form": surface,
                    "label": item["label"],
                    "relation": item["relation"],
                    "field": source_field,
                    "quote": quote,
                    "literal_match": True,
                    "cue_match": bool(cue.search(quote)),
                }
                break
        mentions.append(
            {
                **item,
                **(
                    located
                    or {
                        "surface_form": None,
                        "field": None,
                        "quote": None,
                        "literal_match": False,
                        "cue_match": False,
                    }
                ),
                "review_status": "PENDING",
                "review_note": "Candidate copied from curated_entities.json; verify identity and relation.",
            }
        )
    return mentions


def make_extraction_annotations(root: Path) -> dict[str, Any]:
    papers, curated = load_corpus(root)
    annotations = []
    for paper in papers:
        annotations.append(
            {
                "paper_id": paper["id"],
                "status": "PENDING",
                "reviewers": [],
                "draft_from_curated_entities": curated.get(paper["id"], {}),
                "mentions": _candidate_mentions(paper, curated.get(paper["id"], {})),
            }
        )
    return {
        "schema_version": ANNOTATION_SCHEMA,
        "source_hashes": corpus_hashes(root),
        "review_policy": "Curated lists are annotation candidates, not gold. Review each relation and quote; two independent approvals are required.",
        "papers": annotations,
    }


def prepare_files(root: Path, output_dir: Path) -> list[Path]:
    """Write missing offline starter files without overwriting reviewer work."""
    output_dir.mkdir(parents=True, exist_ok=True)
    starters = {
        output_dir / "questions.json": make_question_bank(root),
        output_dir / "extraction_annotations.json": make_extraction_annotations(root),
    }
    written = []
    for path, data in starters.items():
        if path.exists():
            continue
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        written.append(path)
    return written
