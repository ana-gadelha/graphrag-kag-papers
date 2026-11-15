"""Real graph/encoder + in-memory OpenAI transport, without exporting the corpus.

Requires the fixed corpus bootstrapped in TEST_NEO4J_*. It is a separate smoke,
not a source of empirical LLM quality results.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from openai import OpenAI

from retrieval.answer import answer_question
from retrieval.config import LLMSettings, Settings


def main():
    names = ("TEST_NEO4J_URI", "TEST_NEO4J_USERNAME", "TEST_NEO4J_PASSWORD")
    if not all(os.getenv(n) for n in names):
        raise SystemExit("Configure TEST_NEO4J_* para banco isolado com os 26 papers.")
    llm = LLMSettings(
        model="synthetic-free-model",
        api_key="in-memory-test",
        api="chat_completions",
        base_url="http://in-memory.invalid/v1",
        max_output_tokens=8192,
    )
    settings = Settings(
        uri=os.environ[names[0]],
        username=os.environ[names[1]],
        password=os.environ[names[2]],
        database=os.getenv("TEST_NEO4J_DATABASE", "neo4j"),
        llm=llm,
    )
    requests = []

    def handle(request):
        requests.append(request)
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        if "vocabulary" in payload:
            dataset = next(
                e
                for e in payload["vocabulary"]
                if e["label"] == "Dataset" and e["name"] == "HotpotQA"
            )
            response = {
                "query": "MATCH (p:Paper)-[:EVALUATED_ON]->(d:Dataset) WHERE d.entity_id = $dataset RETURN DISTINCT p.id AS paper_id LIMIT 40",
                "params": {"dataset": dataset["entity_id"]},
            }
        else:
            node_map = {n["id"]: n for n in payload["subgraph"]["nodes"]}
            edge_map = {e["id"]: e for e in payload["subgraph"]["edges"]}
            candidates = [
                p
                for p in payload["paths"]
                if p["kind"] == "fact"
                and any(
                    node_map[n].get("properties", {}).get("name") == "HotpotQA"
                    for n in p["node_ids"]
                )
                and any(edge_map[e]["type"] == "EVALUATED_ON" for e in p["edge_ids"])
            ]
            assert candidates, "HotpotQA fact path was not recovered"
            response = {
                "abstained": False,
                "claims": [
                    {
                        "kind": "graph",
                        "text": "Relação de avaliação.",
                        "source_ids": [p["source"]],
                        "path_ids": [p["path_id"]],
                        "quotes": [],
                    }
                    for p in candidates
                ],
            }
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-in-memory",
                "object": "chat.completion",
                "created": 0,
                "model": llm.model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(response),
                        },
                    }
                ],
            },
        )

    def client(configuration):
        return OpenAI(
            api_key=configuration.api_key,
            base_url=configuration.base_url,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )

    with (
        patch("retrieval.answer.Settings.from_env", return_value=settings),
        patch("retrieval.llm.stage_settings", return_value=llm),
        patch("retrieval.llm.openai_client", side_effect=client),
        patch("retrieval.answer._openai_client", side_effect=client),
        patch("retrieval.llm._runtime_directory", return_value=None),
    ):
        result = answer_question(
            "Quais trabalhos foram avaliados em HotpotQA?", top_k=6
        )
    assert len(requests) == 2, len(requests)
    assert result["planner"]["status"] == "executed", result["planner"]
    assert not result["abstained"] and result["claims"]
    assert all(c["path_ids"] for c in result["claims"])
    assert "EVALUATED_ON" in result["answer"]
    assert any(q.get("executed") is True for q in result["queries"])
    print(
        json.dumps(
            {
                "valid": True,
                "generation": "in_memory_only",
                "model_requests": len(requests),
                "sources": len(result["sources"]),
                "claims": len(result["claims"]),
                "planner_status": result["planner"]["status"],
                "paths": len(result["paths"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
