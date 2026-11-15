"""Hash-bound paired execution with an explicit provider-export gate."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from .benchmark import corpus_hashes, file_sha256
from .metrics import question_fingerprint, sha256_json
from .neo4j_snapshot import capture_snapshot

RUN_SCHEMA = "paired-retrieval-run-v1"
SAFE_LLM_FAILURE_KINDS = {
    "access_denied",
    "authentication_error",
    "configuration_error",
    "connection_error",
    "context_limit",
    "empty_response",
    "invalid_json",
    "invalid_request",
    "local_budget",
    "local_rate_limit",
    "model_unavailable",
    "provider_error",
    "quota",
    "refusal",
    "timeout",
    "truncation",
    "unexpected_finish",
}


def _safe_llm_settings(settings: Any) -> dict[str, Any]:
    endpoint = getattr(settings, "base_url", "")
    return {
        "model": getattr(settings, "model", None),
        "api": getattr(settings, "api", None),
        "endpoint_sha256": hashlib.sha256(str(endpoint).encode("utf-8")).hexdigest(),
        "timeout": getattr(settings, "timeout", None),
        "max_output_tokens": getattr(settings, "max_output_tokens", None),
        "max_retries": getattr(settings, "max_retries", None),
    }


def build_manifest(
    root: Path,
    question_bank: dict,
    split: str,
    top_k: int,
    *,
    retrieval_settings: Any,
    cypher_settings: Any,
    embedding_model: str,
    graph_snapshot: dict[str, Any],
) -> dict[str, Any]:
    if split not in {"dev", "eval"}:
        raise ValueError("split must be dev or eval")
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    selected = [q for q in question_bank["questions"] if q.get("split") == split]
    files = {
        "answer": root / "retrieval/answer.py",
        "planner": root / "retrieval/planner.py",
        "claims": root / "retrieval/claims.py",
        "grammar": root / "retrieval/cypher_subset.lark",
        "cypher_guard": root / "retrieval/cypher_guard.py",
        "retrieval_config": root / "retrieval/config.py",
        "hybrid": root / "retrieval/hybrid.py",
        "embeddings": root / "retrieval/embeddings.py",
        "llm_runtime": root / "retrieval/llm.py",
    }
    file_hashes = {key: file_sha256(path) for key, path in files.items()}
    model_inputs = {
        "retrieval": _safe_llm_settings(retrieval_settings),
        "cypher": _safe_llm_settings(cypher_settings),
        "embedding_model": embedding_model,
    }
    bank_content = [
        {key: value for key, value in question.items() if key != "human_review"}
        for question in selected
    ]
    return {
        "schema_version": RUN_SCHEMA,
        "split": split,
        "top_k": top_k,
        "question_ids": [q["id"] for q in selected],
        "question_hashes": {q["id"]: question_fingerprint(q) for q in selected},
        "question_bank_sha256": sha256_json(bank_content),
        "corpus_hashes": corpus_hashes(root),
        "graph_snapshot_sha256": graph_snapshot["sha256"],
        "graph_snapshot_counts": {
            key: graph_snapshot.get(key)
            for key in (
                "paper_count",
                "entity_count",
                "relation_count",
                "relation_counts",
            )
        },
        "model_ids": {
            "retrieval": model_inputs["retrieval"]["model"],
            "cypher": model_inputs["cypher"]["model"],
            "embedding": embedding_model,
        },
        "model_configuration": {
            "retrieval": model_inputs["retrieval"],
            "cypher": model_inputs["cypher"],
        },
        "model_sha256": sha256_json(model_inputs),
        "prompt_sha256": sha256_json(
            {
                "answer": file_hashes["answer"],
                "planner": file_hashes["planner"],
                "llm_runtime": file_hashes["llm_runtime"],
            }
        ),
        "schema_sha256": sha256_json(
            {
                "answer": file_hashes["answer"],
                "claims": file_hashes["claims"],
            }
        ),
        "grammar_config_sha256": sha256_json(
            {
                "grammar": file_hashes["grammar"],
                "cypher_guard": file_hashes["cypher_guard"],
                "retrieval_config": file_hashes["retrieval_config"],
                "hybrid": file_hashes["hybrid"],
                "embeddings": file_hashes["embeddings"],
            }
        ),
    }


def new_checkpoint(manifest: dict) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA,
        "manifest": manifest,
        "manifest_sha256": sha256_json(manifest),
        "pairs": [],
    }


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        try:
            Path(temporary_name).unlink()
        except OSError:
            pass
        raise


def load_or_create_checkpoint(path: Path, manifest: dict, *, resume: bool) -> dict:
    identity = sha256_json(manifest)
    if path.exists():
        if not resume:
            raise ValueError(
                f"{path} already exists; pass --resume or choose another output path."
            )
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("schema_version") != RUN_SCHEMA
            or checkpoint.get("manifest_sha256") != identity
        ):
            raise ValueError(
                "Existing checkpoint hashes do not match this corpus/model/prompt/schema/grammar configuration."
            )
        if checkpoint.get("invalidated"):
            raise ValueError(
                "Existing checkpoint was invalidated by a graph snapshot change; choose a new output file."
            )
        if checkpoint.get("manifest") != manifest or not isinstance(
            checkpoint.get("pairs"), list
        ):
            raise ValueError("Existing checkpoint manifest or pairs are malformed.")
        return checkpoint
    return new_checkpoint(manifest)


def _initial_claim_reviews(
    question_id: str, mode: str, claims: Any
) -> tuple[list[str], list[dict]]:
    if not isinstance(claims, list):
        return [], []
    claim_ids = []
    reviews = []
    for index, claim in enumerate(claims):
        identity = sha256_json(
            {
                "question_id": question_id,
                "mode": mode,
                "index": index,
                "claim": claim,
            }
        )
        claim_ids.append(identity)
        reviews.append(
            {
                "claim_id": identity,
                "claim_index": index,
                "reviewers": [
                    {"reviewer_id": "", "supported": None, "notes": ""},
                    {"reviewer_id": "", "supported": None, "notes": ""},
                ],
            }
        )
    return claim_ids, reviews


def _mode_outcome(
    question: dict,
    mode: str,
    manifest: dict,
    answer_function: Callable[..., dict],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = answer_function(
            question["question"], top_k=manifest["top_k"], mode=mode
        )
        elapsed = round(time.perf_counter() - started, 4)
        if not isinstance(response, dict):
            raise TypeError("answer_question returned a non-object result")
        claim_ids, reviews = _initial_claim_reviews(
            question["id"], mode, response.get("claims", [])
        )
        return {
            "mode": mode,
            "question_hash": question_fingerprint(question),
            "manifest_sha256": sha256_json(manifest),
            "model_sha256": manifest["model_sha256"],
            "latency_seconds": elapsed,
            "response": response,
            "claim_ids": claim_ids,
            "claim_reviews": reviews,
        }
    except Exception as exc:  # Provider exceptions may contain endpoint/key details.
        failure = {"type": type(exc).__name__}
        kind = getattr(exc, "kind", None)
        if isinstance(kind, str) and kind in SAFE_LLM_FAILURE_KINDS:
            failure["kind"] = kind
        status_code = getattr(exc, "status_code", None)
        if type(status_code) is int and 100 <= status_code <= 599:
            failure["status_code"] = status_code
        return {
            "mode": mode,
            "question_hash": question_fingerprint(question),
            "manifest_sha256": sha256_json(manifest),
            "model_sha256": manifest["model_sha256"],
            "latency_seconds": round(time.perf_counter() - started, 4),
            "failure": failure,
        }


def _mode_outcome_complete(outcome: Any, mode: str) -> bool:
    if not isinstance(outcome, dict) or outcome.get("failure"):
        return False
    response = outcome.get("response")
    if not isinstance(response, dict) or response.get("mode") != mode:
        return False
    if mode == "vector":
        return True
    planner = response.get("planner")
    status = planner.get("status") if isinstance(planner, dict) else None
    return status in {"executed", "empty"}


def run_paired(
    questions: list[dict],
    manifest: dict,
    checkpoint: dict,
    checkpoint_path: Path,
    answer_function: Callable[..., dict],
    snapshot_check: Callable[[], dict[str, Any]],
) -> dict:
    """Run each missing mode and persist after every provider call."""
    identity = sha256_json(manifest)
    if checkpoint.get("manifest_sha256") != identity:
        raise ValueError("Checkpoint manifest changed before execution.")
    if checkpoint.get("invalidated"):
        raise ValueError(
            "Checkpoint was invalidated by a previous graph snapshot mismatch."
        )
    indexed = {
        pair.get("question_id"): pair
        for pair in checkpoint.get("pairs", [])
        if isinstance(pair, dict)
    }
    for question in questions:
        question_id = question["id"]
        try:
            before = snapshot_check()
            if before.get("sha256") != manifest.get("graph_snapshot_sha256"):
                raise ValueError("Graph snapshot hash changed before a question pair.")
        except Exception as exc:
            checkpoint["invalidated"] = {
                "reason": "graph_snapshot_mismatch_before_pair",
                "error_type": type(exc).__name__,
            }
            save_checkpoint(checkpoint_path, checkpoint)
            raise ValueError(
                "Graph snapshot changed or could not be validated before a question pair."
            ) from None
        pair = indexed.get(question_id)
        if pair is None:
            pair = {
                "question_id": question_id,
                "question_hash": question_fingerprint(question),
                "manifest_sha256": identity,
                "top_k": manifest["top_k"],
            }
            checkpoint["pairs"].append(pair)
            indexed[question_id] = pair
            save_checkpoint(checkpoint_path, checkpoint)
        elif (
            pair.get("question_hash") != question_fingerprint(question)
            or pair.get("manifest_sha256") != identity
        ):
            raise ValueError(f"Checkpoint pair hashes changed for {question_id}.")
        pair.setdefault("graph_snapshot_before_sha256", before["sha256"])
        if pair["graph_snapshot_before_sha256"] != manifest.get(
            "graph_snapshot_sha256"
        ):
            checkpoint["invalidated"] = {
                "reason": "graph_snapshot_mismatch_in_checkpoint_pair",
                "error_type": "ValueError",
            }
            save_checkpoint(checkpoint_path, checkpoint)
            raise ValueError(
                f"Checkpoint pair {question_id} used a different graph snapshot."
            )
        for mode in ("vector", "hybrid"):
            existing = pair.get(mode)
            if _mode_outcome_complete(existing, mode):
                continue
            history = (
                list(existing.get("attempt_history", []))
                if isinstance(existing, dict)
                else []
            )
            if isinstance(existing, dict):
                history.append(
                    {
                        key: value
                        for key, value in existing.items()
                        if key != "attempt_history"
                    }
                )
            attempt = _mode_outcome(question, mode, manifest, answer_function)
            attempt["attempt_number"] = len(history) + 1
            if history:
                attempt["attempt_history"] = history
            pair[mode] = attempt
            save_checkpoint(checkpoint_path, checkpoint)
        try:
            after = snapshot_check()
            if after.get("sha256") != manifest.get("graph_snapshot_sha256"):
                raise ValueError("Graph snapshot hash changed after a question pair.")
            pair["graph_snapshot_after_sha256"] = after["sha256"]
            save_checkpoint(checkpoint_path, checkpoint)
        except Exception as exc:
            checkpoint["invalidated"] = {
                "reason": "graph_snapshot_mismatch_after_pair",
                "question_id": question_id,
                "error_type": type(exc).__name__,
            }
            pair["graph_snapshot_after_sha256"] = None
            save_checkpoint(checkpoint_path, checkpoint)
            raise ValueError(
                f"Graph snapshot changed or could not be validated after {question_id}."
            ) from None
    return checkpoint


def stage_settings_for_run() -> tuple[Any, Any, str]:
    """Resolve the same stage settings used by retrieval without making calls."""
    from retrieval.embeddings import EMBEDDING_MODEL
    from retrieval.llm import stage_settings

    return stage_settings("retrieval"), stage_settings("cypher"), EMBEDDING_MODEL


@contextmanager
def graph_snapshot_session(expected_paper_ids: set[str]):
    """Keep one configured driver for repeatable read-only before/after checks."""
    from retrieval.config import Settings, create_driver

    settings = Settings.from_env()
    driver = create_driver(settings)
    try:
        driver.verify_connectivity()
        initial = capture_snapshot(driver, settings.database, expected_paper_ids)
        yield (
            initial,
            lambda: capture_snapshot(driver, settings.database, expected_paper_ids),
        )
    finally:
        driver.close()
