"""Synthetic, provider-free tests for evaluation integrity and metrics."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path

from evaluation.benchmark import QUESTION_SCHEMA
from evaluation.metrics import (
    build_retrieval_report,
    compare_extractions,
    question_fingerprint,
    reviewers_approved,
    sha256_json,
)
from evaluation.neo4j_snapshot import build_snapshot
from evaluation.runner import load_or_create_checkpoint, new_checkpoint, run_paired

SOURCE_HASHES = {
    "papers": "paper-hash",
    "curated_entities": "curated-hash",
    "entity_registry": "registry-hash",
}
PAPERS = [
    {"id": "p1", "title": "Paper one", "abstract": "proof one"},
    {"id": "p2", "title": "Paper two", "abstract": "proof two"},
]
GRAPH_HASH = "a" * 64


def _questions() -> list[dict]:
    questions = []
    for index in range(6):
        multi_hop = index == 0
        unsupported = index == 5
        required = ["p1", "p2"] if multi_hop else [] if unsupported else ["p1"]
        questions.append(
            {
                "id": f"dev-{index}",
                "split": "dev",
                "category": "multi_hop"
                if multi_hop
                else "unsupported"
                if unsupported
                else "direct",
                "question": f"Question {index}?",
                "reference_draft": {
                    "answer": "A reviewed synthetic reference.",
                    "required_papers": required,
                    "expected_abstention": unsupported,
                    "evidence": [
                        {"paper_id": "p1", "field": "abstract", "quote": "proof one"}
                    ],
                },
                "human_review": {
                    "status": "APPROVED",
                    "reviewers": ["reviewer-a", "reviewer-b"],
                },
            }
        )
    return questions


def _bank() -> dict:
    return {
        "schema_version": QUESTION_SCHEMA,
        "source_hashes": SOURCE_HASHES,
        "questions": _questions(),
    }


def _response(
    mode: str, *, planner_status: str | None = None, include_path: bool = True
) -> dict:
    sources = [
        {
            "citation_id": "S1",
            "paper_id": "p1",
            "title": "Paper one",
            "abstract": "proof one",
        },
        {
            "citation_id": "S2",
            "paper_id": "p2",
            "title": "Paper two",
            "abstract": "proof two",
        },
    ]
    nodes = [
        {"id": "paper:p1", "label": "Paper", "properties": {"id": "p1"}},
        {"id": "author:shared", "label": "Author", "properties": {"name": "Shared"}},
        {"id": "paper:p2", "label": "Paper", "properties": {"id": "p2"}},
    ]
    edges = [
        {"source": "paper:p1", "target": "author:shared", "type": "AUTHORED"},
        {"source": "paper:p2", "target": "author:shared", "type": "AUTHORED"},
    ]
    claims = [
        {
            "kind": "document",
            "text": "Literal claim one",
            "source_ids": ["S1"],
            "path_ids": [],
            "quotes": [{"source_id": "S1", "field": "abstract", "quote": "proof one"}],
        },
    ]
    paths = []
    if include_path:
        paths = [
            {
                "path_id": "P1",
                "kind": "shortest",
                "source": "S1",
                "target": "S2",
                "length": 2,
                "nodes": nodes,
                "edges": edges,
            }
        ]
        claims.append(
            {
                "kind": "graph",
                "text": "Connected papers",
                "source_ids": ["S1", "S2"],
                "path_ids": ["P1"],
                "quotes": [],
            }
        )
    planner = (
        {"status": planner_status}
        if planner_status is not None
        else {"status": "disabled"}
    )
    return {
        "mode": mode,
        "abstained": False,
        "claims": claims,
        "generation": {
            "model": "synthetic-model",
            "effective_model": "synthetic-model",
        },
        "planner": planner,
        "sources": sources,
        "paths": paths,
        "subgraph": {"nodes": nodes, "edges": edges},
    }


def _outcome(
    question: dict, mode: str, manifest_hash: str, *, planner_status: str | None = None
) -> dict:
    response = _response(mode, planner_status=planner_status)
    if question["reference_draft"]["expected_abstention"]:
        response["abstained"] = True
        response["claims"] = []
        response["generation"] = {}
    claim_ids = [
        sha256_json(
            {
                "question_id": question["id"],
                "mode": mode,
                "index": index,
                "claim": claim,
            }
        )
        for index, claim in enumerate(response["claims"])
    ]
    claim_reviews = [
        {
            "claim_id": claim_id,
            "claim_index": index,
            "reviewers": [
                {"reviewer_id": "reviewer-a", "supported": True, "notes": "supported"},
                {"reviewer_id": "reviewer-b", "supported": True, "notes": "supported"},
            ],
        }
        for index, claim_id in enumerate(claim_ids)
    ]
    return {
        "mode": mode,
        "question_hash": question_fingerprint(question),
        "manifest_sha256": manifest_hash,
        "model_sha256": "b" * 64,
        "latency_seconds": 0.2,
        "response": response,
        "claim_ids": claim_ids,
        "claim_reviews": claim_reviews,
    }


def _run(
    *,
    hybrid_status: str = "executed",
    drop_pair: bool = False,
    bad_pair_hash: bool = False,
) -> tuple[dict, dict]:
    bank = _bank()
    questions = bank["questions"]
    manifest = {
        "schema_version": "paired-retrieval-run-v1",
        "split": "dev",
        "top_k": 6,
        "question_ids": [question["id"] for question in questions],
        "question_hashes": {
            question["id"]: question_fingerprint(question) for question in questions
        },
        "corpus_hashes": SOURCE_HASHES,
        "model_ids": {"retrieval": "synthetic-model", "cypher": "synthetic-model"},
        "model_sha256": "b" * 64,
        "prompt_sha256": "c" * 64,
        "schema_sha256": "d" * 64,
        "grammar_config_sha256": "e" * 64,
        "graph_snapshot_sha256": GRAPH_HASH,
        "graph_snapshot_counts": {
            "paper_count": 2,
            "entity_count": 1,
            "relation_count": 2,
        },
    }
    manifest_hash = sha256_json(manifest)
    pairs = []
    for index, question in enumerate(questions):
        if drop_pair and index == 5:
            continue
        pair = {
            "question_id": question["id"],
            "question_hash": "wrong"
            if bad_pair_hash and index == 0
            else question_fingerprint(question),
            "manifest_sha256": manifest_hash,
            "top_k": 6,
            "graph_snapshot_before_sha256": GRAPH_HASH,
            "graph_snapshot_after_sha256": GRAPH_HASH,
            "vector": _outcome(question, "vector", manifest_hash),
            "hybrid": _outcome(
                question,
                "hybrid",
                manifest_hash,
                planner_status=hybrid_status if index == 0 else "empty",
            ),
        }
        pairs.append(pair)
    return {
        "schema_version": "paired-retrieval-run-v1",
        "manifest": manifest,
        "manifest_sha256": manifest_hash,
        "pairs": pairs,
    }, bank


class SnapshotTests(unittest.TestCase):
    def _rows(self):
        papers = [
            {
                "id": "p1",
                "title": "A",
                "abstract": "B",
                "embedding_model": "m",
                "embedding_hash": "content",
                "embedding": [0.1, 0.2],
                "keywords": ["RAG"],
            }
        ]
        entities = [
            {
                "labels": ["Author"],
                "entity_id": "author:1",
                "name": "Ada",
                "aliases": ["ADA"],
            }
        ]
        relations = [
            {
                "paper_id": "p1",
                "relation": "AUTHORED",
                "entity_labels": ["Author"],
                "entity_id": "author:1",
                "entity_name": "Ada",
                "entity_aliases": ["ADA"],
                "paper_is_source": True,
                "provenance": {"quote": "Ada"},
            }
        ]
        relation_types = [
            {"relation": relation}
            for relation in (
                "AUTHORED",
                "HAS_CONCEPT",
                "PROPOSES",
                "USES_METHOD",
                "EVALUATED_ON",
            )
        ]
        return papers, entities, relations, relation_types

    def test_snapshot_hash_covers_embedding_provenance_and_graph_edges(self):
        data = self._rows()
        first = build_snapshot(*data, {"p1"})
        changed = deepcopy(data)
        changed[0][0]["embedding"] = [0.1, 0.3]
        second = build_snapshot(*changed, {"p1"})
        self.assertNotEqual(first["sha256"], second["sha256"])
        changed = deepcopy(data)
        changed[2][0]["provenance"]["quote"] = "different"
        third = build_snapshot(*changed, {"p1"})
        self.assertNotEqual(first["sha256"], third["sha256"])

    def test_snapshot_requires_exact_paper_id_set(self):
        with self.assertRaisesRegex(ValueError, "Paper IDs differ"):
            build_snapshot(*self._rows(), {"p1", "p2"})

    def test_nonretrieval_relationship_types_are_recorded_but_not_identity_blockers(
        self,
    ):
        data = self._rows()
        first = build_snapshot(*data, {"p1"})
        changed = deepcopy(data)
        changed[3].append({"relation": "CITES"})
        second = build_snapshot(*changed, {"p1"})
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(second["ignored_relation_types"], ["CITES"])


class PairedReportTests(unittest.TestCase):
    def test_complete_pair_reports_deterministic_citations_and_real_multi_hop_path(
        self,
    ):
        run, bank = _run()
        report = build_retrieval_report(run, bank, SOURCE_HASHES, papers_list=PAPERS)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["complete_pairs"], 6)
        self.assertEqual(report["availability"]["planner_eligible_pairs"], 6)
        self.assertEqual(report["availability"]["hybrid_planner"]["attempted"], 6)
        self.assertEqual(
            report["availability"]["hybrid_planner"][
                "successful_execution_rate_among_attempts"
            ],
            1.0,
        )
        self.assertEqual(
            report["availability"]["hybrid_planner"]["nonempty_executions"], 1
        )
        self.assertEqual(
            report["metrics"]["hybrid"]["multi_hop_valid_path_coverage"], 1.0
        )
        self.assertEqual(
            report["metrics"]["hybrid"]["multi_hop_valid_path_cited_coverage"], 1.0
        )
        self.assertEqual(
            report["metrics"]["hybrid"]["deterministic_citation_validity"], 1.0
        )
        self.assertEqual(report["metrics"]["hybrid"]["abstention_accuracy"], 1.0)
        self.assertEqual(
            report["availability"]["hybrid_planner_status_counts"],
            {"empty": 5, "executed": 1},
        )
        self.assertEqual(report["paired_deltas"]["source_recall_macro"], 0.0)

    def test_pending_gold_and_missing_pair_are_inconclusive_and_count_failures(self):
        run, bank = _run(drop_pair=True)
        bank["questions"][0]["human_review"] = {"status": "PENDING", "reviewers": []}
        report = build_retrieval_report(run, bank, SOURCE_HASHES, papers_list=PAPERS)
        self.assertEqual(report["status"], "inconclusive")
        self.assertEqual(report["availability"]["missing_pairs"], 1)
        self.assertEqual(report["paired_deltas"], None)
        self.assertTrue(any("reference" in item.lower() for item in report["problems"]))

    def test_reviewer_ids_are_normalized_before_counting_independent_reviewers(self):
        self.assertFalse(
            reviewers_approved({"status": "APPROVED", "reviewers": ["Ana", " ANA "]})
        )
        self.assertTrue(
            reviewers_approved({"status": "APPROVED", "reviewers": ["Ana", "Bia"]})
        )

    def test_mismatched_pair_hash_is_inconclusive(self):
        run, bank = _run(bad_pair_hash=True)
        report = build_retrieval_report(run, bank, SOURCE_HASHES, papers_list=PAPERS)
        self.assertEqual(report["status"], "inconclusive")
        self.assertTrue(any("hash mismatch" in item for item in report["problems"]))

    def test_degraded_planner_is_excluded_and_keeps_report_inconclusive(self):
        run, bank = _run(hybrid_status="invalid")
        report = build_retrieval_report(run, bank, SOURCE_HASHES, papers_list=PAPERS)
        self.assertEqual(report["status"], "inconclusive")
        self.assertEqual(report["planner_eligible_pairs"], 5)
        self.assertEqual(report["metrics"]["hybrid"]["planner"]["attempted"], 5)
        self.assertEqual(
            report["availability"]["hybrid_planner_status_counts"]["invalid"], 1
        )
        self.assertIsNone(report["paired_deltas"])

    def test_report_counts_degraded_attempt_history_after_successful_resume(self):
        run, bank = _run()
        pair = run["pairs"][0]
        prior_attempt = deepcopy(pair["hybrid"])
        prior_attempt["response"]["planner"]["status"] = "quota"
        pair["hybrid"]["attempt_history"] = [prior_attempt]
        pair["hybrid"]["attempt_number"] = 2

        report = build_retrieval_report(run, bank, SOURCE_HASHES, papers_list=PAPERS)

        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            report["availability"]["hybrid_planner_status_counts"]["quota"], 1
        )
        self.assertEqual(report["availability"]["hybrid_planner"]["attempted"], 7)
        self.assertEqual(report["planner_eligible_pairs"], 6)

    def test_no_valid_planner_execution_has_no_main_metrics(self):
        run, bank = _run(hybrid_status="quota")
        for pair in run["pairs"]:
            pair["hybrid"]["response"]["planner"]["status"] = "quota"
        report = build_retrieval_report(run, bank, SOURCE_HASHES, papers_list=PAPERS)
        self.assertEqual(report["status"], "inconclusive")
        self.assertEqual(report["planner_eligible_pairs"], 0)
        self.assertIsNone(report["metrics"])
        self.assertTrue(
            any("No hybrid planner query" in item for item in report["problems"])
        )

    def test_missing_planner_status_is_reported_but_not_an_attempt_denominator(self):
        run, bank = _run()
        run["pairs"][0]["hybrid"]["response"]["planner"] = {}
        report = build_retrieval_report(run, bank, SOURCE_HASHES, papers_list=PAPERS)
        self.assertEqual(report["status"], "inconclusive")
        self.assertEqual(
            report["availability"]["hybrid_planner_status_counts"]["missing"], 1
        )
        self.assertEqual(report["metrics"]["hybrid"]["planner"]["attempted"], 5)
        self.assertEqual(report["availability"]["hybrid_planner"]["attempted"], 5)

    def test_mode_failure_is_counted_over_full_selected_set(self):
        run, bank = _run()
        failed = run["pairs"][0]["vector"]
        run["pairs"][0]["vector"] = {
            "mode": "vector",
            "question_hash": failed["question_hash"],
            "manifest_sha256": failed["manifest_sha256"],
            "model_sha256": failed["model_sha256"],
            "latency_seconds": 0.1,
            "failure": {"type": "TimeoutError"},
        }
        report = build_retrieval_report(run, bank, SOURCE_HASHES, papers_list=PAPERS)
        self.assertEqual(report["status"], "inconclusive")
        self.assertEqual(report["availability"]["vector_failures"], 1)
        self.assertEqual(report["availability"]["vector_responses"], 5)
        self.assertEqual(report["availability"]["selected_questions"], 6)

    def test_effective_model_mismatch_invalidates_pair_comparison(self):
        run, bank = _run()
        run["pairs"][0]["hybrid"]["response"]["generation"]["effective_model"] = (
            "other-model"
        )
        report = build_retrieval_report(run, bank, SOURCE_HASHES, papers_list=PAPERS)
        self.assertEqual(report["status"], "inconclusive")
        self.assertTrue(
            any("different effective models" in item for item in report["problems"])
        )

    def test_effective_model_drift_between_pairs_invalidates_comparison(self):
        run, bank = _run()
        for mode in ("vector", "hybrid"):
            run["pairs"][1][mode]["response"]["generation"]["effective_model"] = (
                "model-v2"
            )
        report = build_retrieval_report(run, bank, SOURCE_HASHES, papers_list=PAPERS)
        self.assertEqual(report["status"], "inconclusive")
        self.assertTrue(any("across pairs" in item for item in report["problems"]))

    def test_abstention_without_generation_receipt_does_not_require_model_id(self):
        run, bank = _run()
        for mode in ("vector", "hybrid"):
            outcome = run["pairs"][0][mode]
            response = outcome["response"]
            response["abstained"] = True
            response["claims"] = []
            response["generation"] = {}
            outcome["claim_ids"] = []
            outcome["claim_reviews"] = []
        report = build_retrieval_report(run, bank, SOURCE_HASHES, papers_list=PAPERS)
        self.assertEqual(report["status"], "complete")

    def test_path_absence_counts_zero_instead_of_claiming_multihop_coverage(self):
        run, bank = _run()
        pair = run["pairs"][0]
        for mode in ("vector", "hybrid"):
            response = pair[mode]["response"]
            response["paths"] = []
            response["claims"] = response["claims"][:1]
            pair[mode]["claim_ids"] = pair[mode]["claim_ids"][:1]
            pair[mode]["claim_reviews"] = pair[mode]["claim_reviews"][:1]
        report = build_retrieval_report(run, bank, SOURCE_HASHES, papers_list=PAPERS)
        self.assertEqual(
            report["metrics"]["hybrid"]["multi_hop_valid_path_coverage"], 0.0
        )
        self.assertEqual(
            report["metrics"]["hybrid"]["multi_hop_valid_path_cited_coverage"], 0.0
        )

    def test_planner_status_matrix_keeps_disabled_and_missing_out_of_denominator(self):
        run, bank = _run()
        statuses = ["executed", "empty", "invalid", "timeout", "disabled", "missing"]
        for pair, status in zip(run["pairs"], statuses):
            planner = pair["hybrid"]["response"]["planner"]
            if status == "missing":
                pair["hybrid"]["response"]["planner"] = {}
            else:
                planner["status"] = status
        report = build_retrieval_report(run, bank, SOURCE_HASHES, papers_list=PAPERS)
        planner = report["availability"]["hybrid_planner"]
        self.assertEqual(report["status"], "inconclusive")
        self.assertEqual(planner["attempted"], 4)
        self.assertEqual(planner["validated_query_rate_among_attempts"], 0.5)
        self.assertEqual(planner["successful_execution_rate_among_attempts"], 0.5)
        self.assertEqual(planner["nonempty_execution_rate_among_attempts"], 0.25)

    def test_invalid_literal_claim_quote_is_measured_as_invalid(self):
        run, bank = _run()
        for mode in ("vector", "hybrid"):
            run["pairs"][0][mode]["response"]["claims"][0]["quotes"][0]["quote"] = (
                "invented quote"
            )
        report = build_retrieval_report(run, bank, SOURCE_HASHES, papers_list=PAPERS)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            report["metrics"]["vector"]["deterministic_citation_validity"], 0.9
        )


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_rejects_changed_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run.json"
            first = {"model_sha256": "one"}
            path.write_text(
                __import__("json").dumps(new_checkpoint(first)), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "hashes do not match"):
                load_or_create_checkpoint(path, {"model_sha256": "two"}, resume=True)

    def test_pair_runner_checks_snapshot_before_and_after_each_pair(self):
        question = _questions()[0]
        manifest = {
            "top_k": 6,
            "model_sha256": "b" * 64,
            "graph_snapshot_sha256": GRAPH_HASH,
        }
        checkpoint = new_checkpoint(manifest)
        calls = []
        snapshots = []

        def answer(text, *, top_k, mode):
            calls.append((text, top_k, mode))
            return _response(
                mode, planner_status="executed" if mode == "hybrid" else "disabled"
            )

        def snapshot_check():
            snapshots.append(1)
            return {"sha256": GRAPH_HASH}

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run.json"
            run_paired([question], manifest, checkpoint, path, answer, snapshot_check)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(
            checkpoint["pairs"][0]["graph_snapshot_before_sha256"], GRAPH_HASH
        )
        self.assertEqual(
            checkpoint["pairs"][0]["graph_snapshot_after_sha256"], GRAPH_HASH
        )

    def test_pair_runner_invalidates_checkpoint_when_graph_changes(self):
        question = _questions()[0]
        manifest = {
            "top_k": 6,
            "model_sha256": "b" * 64,
            "graph_snapshot_sha256": GRAPH_HASH,
        }
        checkpoint = new_checkpoint(manifest)
        sequence = iter([GRAPH_HASH, "f" * 64])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run.json"
            with self.assertRaisesRegex(ValueError, "after dev-0"):
                run_paired(
                    [question],
                    manifest,
                    checkpoint,
                    path,
                    lambda text, *, top_k, mode: _response(mode),
                    lambda: {"sha256": next(sequence)},
                )
            self.assertEqual(
                checkpoint["invalidated"]["reason"],
                "graph_snapshot_mismatch_after_pair",
            )

    def test_pair_runner_refuses_changed_snapshot_before_any_model_call(self):
        question = _questions()[0]
        manifest = {
            "top_k": 6,
            "model_sha256": "b" * 64,
            "graph_snapshot_sha256": GRAPH_HASH,
        }
        checkpoint = new_checkpoint(manifest)
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "before a question pair"):
                run_paired(
                    [question],
                    manifest,
                    checkpoint,
                    Path(temp) / "run.json",
                    lambda text, *, top_k, mode: calls.append(mode),
                    lambda: {"sha256": "f" * 64},
                )
        self.assertEqual(calls, [])
        self.assertTrue(checkpoint.get("invalidated"))

    def test_resume_retries_failed_mode_and_keeps_completed_sibling_mode(self):
        question = _questions()[0]
        manifest = {
            "top_k": 6,
            "model_sha256": "b" * 64,
            "graph_snapshot_sha256": GRAPH_HASH,
        }
        initial = new_checkpoint(manifest)
        first_calls = []

        def first_answer(text, *, top_k, mode):
            first_calls.append(mode)
            if mode == "vector":
                raise TimeoutError
            return _response(mode, planner_status="empty")

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run.json"
            run_paired(
                [question],
                manifest,
                initial,
                path,
                first_answer,
                lambda: {"sha256": GRAPH_HASH},
            )
            resumed = load_or_create_checkpoint(path, manifest, resume=True)
            second_calls = []

            def second_answer(text, *, top_k, mode):
                second_calls.append(mode)
                return _response(mode, planner_status="executed")

            run_paired(
                [question],
                manifest,
                resumed,
                path,
                second_answer,
                lambda: {"sha256": GRAPH_HASH},
            )
        self.assertEqual(first_calls, ["vector", "hybrid"])
        self.assertEqual(second_calls, ["vector"])
        self.assertEqual(
            resumed["pairs"][0]["hybrid"]["response"]["planner"]["status"], "empty"
        )
        self.assertIsNone(resumed["pairs"][0]["vector"].get("failure"))

    def test_resume_retries_quota_planner_and_preserves_previous_attempt_reviews(self):
        question = _questions()[0]
        manifest = {
            "top_k": 6,
            "model_sha256": "b" * 64,
            "graph_snapshot_sha256": GRAPH_HASH,
        }
        checkpoint = new_checkpoint(manifest)
        first_calls = []

        def quota_answer(text, *, top_k, mode):
            first_calls.append(mode)
            return _response(
                mode, planner_status="quota" if mode == "hybrid" else "disabled"
            )

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run.json"
            run_paired(
                [question],
                manifest,
                checkpoint,
                path,
                quota_answer,
                lambda: {"sha256": GRAPH_HASH},
            )
            pair = checkpoint["pairs"][0]
            prior_hybrid = pair["hybrid"]
            prior_hybrid["claim_reviews"][0]["reviewers"] = [
                {"reviewer_id": "reviewer-a", "supported": True, "notes": "yes"},
                {"reviewer_id": "reviewer-b", "supported": False, "notes": "no"},
            ]
            prior_vector = pair["vector"]
            prior_vector["claim_reviews"][0]["reviewers"] = [
                {"reviewer_id": "reviewer-a", "supported": True, "notes": "yes"},
                {"reviewer_id": "reviewer-b", "supported": True, "notes": "yes"},
            ]
            from evaluation.runner import save_checkpoint

            save_checkpoint(path, checkpoint)
            resumed = load_or_create_checkpoint(path, manifest, resume=True)
            second_calls = []

            def recovered_answer(text, *, top_k, mode):
                second_calls.append(mode)
                return _response(mode, planner_status="executed")

            run_paired(
                [question],
                manifest,
                resumed,
                path,
                recovered_answer,
                lambda: {"sha256": GRAPH_HASH},
            )

        self.assertEqual(first_calls, ["vector", "hybrid"])
        self.assertEqual(second_calls, ["hybrid"])
        final_hybrid = resumed["pairs"][0]["hybrid"]
        self.assertEqual(final_hybrid["response"]["planner"]["status"], "executed")
        self.assertEqual(final_hybrid["attempt_number"], 2)
        self.assertEqual(len(final_hybrid["attempt_history"]), 1)
        prior_attempt = final_hybrid["attempt_history"][0]
        self.assertEqual(prior_attempt["response"]["planner"]["status"], "quota")
        self.assertEqual(
            prior_attempt["claim_reviews"][0]["reviewers"][1]["supported"], False
        )
        self.assertEqual(
            resumed["pairs"][0]["vector"]["claim_reviews"][0]["reviewers"][1][
                "supported"
            ],
            True,
        )

    def test_failure_records_allowlisted_llm_metadata_without_exception_text(self):
        class LLMError(Exception):
            kind = "local_budget"
            status_code = 429

        question = _questions()[0]
        manifest = {
            "top_k": 6,
            "model_sha256": "b" * 64,
            "graph_snapshot_sha256": GRAPH_HASH,
        }
        checkpoint = new_checkpoint(manifest)

        def failing_answer(text, *, top_k, mode):
            raise LLMError("https://private.invalid/key=secret response body")

        with tempfile.TemporaryDirectory() as temp:
            run_paired(
                [question],
                manifest,
                checkpoint,
                Path(temp) / "run.json",
                failing_answer,
                lambda: {"sha256": GRAPH_HASH},
            )
        failure = checkpoint["pairs"][0]["vector"]["failure"]
        self.assertEqual(
            failure, {"type": "LLMError", "kind": "local_budget", "status_code": 429}
        )
        self.assertNotIn("private", str(checkpoint))

    def test_cli_run_requires_explicit_export_flag(self):
        from evaluation.__main__ import main

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as exit_info:
            main(["run", "--split", "dev"])
        self.assertEqual(exit_info.exception.code, 2)


class ExtractionReportTests(unittest.TestCase):
    def _annotations(self):
        return {
            "schema_version": "extraction-annotations-v1",
            "source_hashes": SOURCE_HASHES,
            "papers": [
                {
                    "paper_id": "p1",
                    "status": "APPROVED",
                    "reviewers": ["a", "b"],
                    "mentions": [
                        {
                            "surface_form": "Method A",
                            "canonical_name": "Method A",
                            "label": "Method",
                            "relation": "PROPOSES",
                            "field": "abstract",
                            "quote": "We propose Method A.",
                        }
                    ],
                },
                {
                    "paper_id": "p2",
                    "status": "APPROVED",
                    "reviewers": ["a", "b"],
                    "mentions": [],
                },
            ],
        }

    def test_extraction_metrics_are_by_label_relation_and_literal_evidence(self):
        papers = [
            {"id": "p1", "title": "Paper one", "abstract": "We propose Method A."},
            {"id": "p2", "title": "Paper two", "abstract": "No relation here."},
        ]
        predictions = [
            {
                **papers[0],
                "extracted": {
                    "mentions": [
                        {
                            "surface_form": "Method A",
                            "canonical_name": "Method A",
                            "label": "Method",
                            "relation": "PROPOSES",
                            "field": "abstract",
                            "quote": "We propose Method A.",
                        }
                    ]
                },
            },
            {**papers[1], "extracted": {"mentions": []}},
        ]
        report = compare_extractions(
            predictions, self._annotations(), papers, SOURCE_HASHES
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["metrics"]["f1"], 1.0)
        self.assertEqual(report["metrics"]["by_label"]["Method"]["f1"], 1.0)
        self.assertEqual(
            report["metrics"]["by_relation"]["PROPOSES"]["predicted_evidence_coverage"],
            1.0,
        )

    def test_pending_extraction_gold_remains_inconclusive(self):
        annotations = self._annotations()
        annotations["papers"][0]["status"] = "PENDING"
        report = compare_extractions([], annotations, [], SOURCE_HASHES)
        self.assertEqual(report["status"], "inconclusive")
        self.assertIsNone(report["metrics"])


if __name__ == "__main__":
    unittest.main()
