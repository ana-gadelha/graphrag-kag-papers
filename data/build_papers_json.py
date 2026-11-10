"""Build the team's 26-paper MVP corpus from arXiv and the SBC journal.

Run from any directory with the project's Python environment and `arxiv`
package installed. The output follows the schema requested for papers.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import arxiv

ARXIV_KEYWORDS: dict[str, list[str]] = {
    "2501.13958": ["GraphRAG", "survey", "knowledge graphs", "multi-hop retrieval"],
    "2408.08921": ["GraphRAG", "survey", "graph-based retrieval", "knowledge integration"],
    "2501.00309": ["GraphRAG", "retrieval-augmented generation", "knowledge graphs"],
    "2404.16130": ["GraphRAG", "query-focused summarization", "community detection"],
    "2410.05779": ["LightRAG", "hybrid retrieval", "knowledge graphs", "vector search"],
    "2405.14831": ["HippoRAG", "long-term memory", "knowledge graphs", "personalized PageRank"],
    "2409.13731": ["KAG", "Knowledge-Augmented Generation", "logic forms", "professional domains"],
    "2502.14902": ["PathRAG", "relational paths", "graph pruning", "multi-hop reasoning"],
    "2405.16506": ["GRAG", "graph retrieval", "question answering", "knowledge graphs"],
    "2402.07630": ["G-Retriever", "textual graphs", "subgraph retrieval", "question answering"],
    "2407.10805": ["Think-on-Graph", "knowledge graphs", "multi-hop reasoning", "retrieval-augmented generation"],
    "2412.18644": ["DynaGRAG", "graph topology", "dynamic retrieval", "language understanding"],
    "2503.13804": ["GraphRAG", "knowledge filtering", "knowledge integration", "hallucination reduction"],
    "2506.19967": ["GraphRAG", "multi-hop question answering", "inference-time scaling", "knowledge graphs"],
    "2508.19855": ["Youtu-GraphRAG", "agentic retrieval", "graph construction", "complex reasoning"],
    "2509.22009": ["GraphSearch", "agentic search", "hybrid retrieval", "multi-hop reasoning"],
    "2510.10114": ["LinearRAG", "GraphRAG", "scalable retrieval", "large-scale corpora"],
    "2501.15067": ["citation graph", "GraphRAG", "research question answering", "scientific literature"],
    "2502.01113": ["GFM-RAG", "graph foundation model", "graph retrieval", "RAG"],
    "2502.04413": ["medical RAG", "knowledge graphs", "clinical reasoning", "healthcare"],
    "2502.06864": ["knowledge graph-guided RAG", "hybrid retrieval", "knowledge graphs"],
    "2502.09304": ["KET-RAG", "multi-granular indexing", "graph retrieval", "efficiency"],
    "2505.24226": ["E²GraphRAG", "graph-based RAG", "efficiency", "retrieval"],
    "2606.25656": ["GraphRAG", "context optimization", "agentic retrieval", "RAG comparison"],
    "2608.28766": ["hybrid GraphRAG", "vector search", "climate science", "question answering"],
}

BRAZILIAN_PAPER = {
    "id": "doi:10.5753/reic.2026.8503",
    "source": "SBC",
    "url": "https://doi.org/10.5753/reic.2026.8503",
    "title": "Multimodal RAG with Knowledge Graphs for Portuguese Maintenance Manuals in the Context of Industry 4.0",
    "authors": ["Christian Freitas", "Lilian Berton"],
    "year": 2026,
    "abstract": (
        "Navigating dense, multimodal technical documentation remains a critical bottleneck for industrial maintenance in non-English contexts. "
        "This paper presents a multimodal Retrieval-Augmented Generation (RAG) framework for Portuguese maintenance manuals, combining dense vector search over text and tables with knowledge graph traversal. "
        "We evaluate a local multilingual transformer (paraphrase-multilingual-MiniLM-L12-v2) and OpenAI’s text-embedding-3-small across 8 configurations (2 models × 4 retrieval modes) using a ground-truth set of 50 domain-specific queries. "
        "The best configurations (openai/text_only and openai/text_table) achieve a BERTScore-F1 of 0.70, ROUGE-L of 0.36, and an MRR of 0.65, answering 45 of 50 queries. "
        "Analysis shows that while high-dimensional proprietary embeddings excel at capturing specialized technical jargon, standard Reciprocal Rank Fusion (RRF) introduces rank contamination when fusing sparse graph signals with dense text vectors. Source code is publicly available."
    ),
    "keywords": [
        "Portuguese NLP", "multimodal RAG", "knowledge graphs", "vector retrieval",
        "hybrid retrieval", "industrial maintenance",
    ],
}


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    results = client.results(arxiv.Search(id_list=list(ARXIV_KEYWORDS)))
    found: dict[str, arxiv.Result] = {}
    for result in results:
        paper_id = result.entry_id.rstrip("/").split("/")[-1].split("v")[0]
        if paper_id in ARXIV_KEYWORDS:
            found[paper_id] = result

    missing = sorted(set(ARXIV_KEYWORDS) - found.keys())
    if missing:
        raise RuntimeError(f"arXiv did not return selected records: {', '.join(missing)}")

    papers: list[dict[str, object]] = []
    for paper_id, keywords in ARXIV_KEYWORDS.items():
        result = found[paper_id]
        papers.append(
            {
                # Identificador único do paper (usado na constraint do Neo4j)
                "id": paper_id,
                "source": "arXiv",
                # Link para o artigo original (usado na provenance)
                "url": f"https://arxiv.org/abs/{paper_id}",
                "title": clean_text(result.title),
                "authors": [author.name for author in result.authors],
                # The arXiv identifier year avoids a timezone boundary shifting
                # a late-December submission into the previous calendar year.
                "year": 2000 + int(paper_id[:2]),
                "abstract": clean_text(result.summary),
                "keywords": keywords,
            }
        )

    papers.append(BRAZILIAN_PAPER)
    output = Path(__file__).with_name("papers.json")
    output.write_text(json.dumps(papers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(papers)} papers to {output}")


if __name__ == "__main__":
    main()
