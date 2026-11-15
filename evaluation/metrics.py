"""Offline validation and descriptive metrics for reviewed evaluation data."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter
from typing import Any

from .benchmark import ANNOTATION_SCHEMA, QUESTION_SCHEMA, canonical_json

EXPECTED_EVAL_CATEGORIES = {
    "direct": 2,
    "intersection": 2,
    "multi_hop": 6,
    "unsupported": 4,
    "interpretation_trap": 2,
}
RELATION_LABEL = {
    "PROPOSES": "Method",
    "USES_METHOD": "Method",
    "EVALUATED_ON": "Dataset",
}


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def question_fingerprint(question: dict) -> str:
    """Hash the question and reference content, excluding mutable review fields."""
    content = {key: value for key, value in question.items() if key != "human_review"}
    return sha256_json(content)


def reviewers_approved(review: Any) -> bool:
    if (
        not isinstance(review, dict)
        or str(review.get("status", "")).upper() != "APPROVED"
    ):
        return False
    reviewers = review.get("reviewers")
    if not isinstance(reviewers, list):
        return False
    ids = [
        item.strip().casefold()
        for item in reviewers
        if isinstance(item, str) and item.strip()
    ]
    return len(set(ids)) >= 2


def _quote_exists(papers: dict[str, dict], evidence: Any) -> bool:
    if not isinstance(evidence, dict):
        return False
    paper_id = evidence.get("paper_id")
    if not isinstance(paper_id, str):
        return False
    paper = papers.get(paper_id)
    field = evidence.get("field")
    quote = evidence.get("quote")
    if (
        paper is None
        or not isinstance(field, str)
        or not isinstance(quote, str)
        or not quote
    ):
        return False
    source = paper.get(field)
    if field in {"authors", "keywords"}:
        return isinstance(source, list) and quote in source
    return isinstance(source, str) and quote in source


def validate_question_bank(
    bank: Any,
    papers_list: list[dict],
    current_source_hashes: dict[str, str],
    *,
    split: str | None = None,
    require_approved: bool = False,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(bank, dict) or bank.get("schema_version") != QUESTION_SCHEMA:
        return [f"Question bank schema must be {QUESTION_SCHEMA}."]
    if bank.get("source_hashes") != current_source_hashes:
        errors.append(
            "Question bank corpus hashes do not match the current corpus files."
        )
    questions = bank.get("questions")
    if not isinstance(questions, list):
        return errors + ["Question bank must contain a questions list."]
    ids = [question.get("id") for question in questions if isinstance(question, dict)]
    if len(ids) != len(questions) or len(ids) != len(set(ids)):
        errors.append("Question IDs must be present and unique.")
    paper_ids = {paper.get("id") for paper in papers_list}
    selected = [
        q
        for q in questions
        if isinstance(q, dict) and (split is None or q.get("split") == split)
    ]
    if split not in {None, "dev", "eval"}:
        errors.append("Split must be dev or eval.")
    if split is None or split == "dev":
        if sum(isinstance(q, dict) and q.get("split") == "dev" for q in questions) != 6:
            errors.append(
                "The benchmark must contain exactly six development questions."
            )
    if split is None or split == "eval":
        eval_questions = [
            q for q in questions if isinstance(q, dict) and q.get("split") == "eval"
        ]
        if len(eval_questions) != 16:
            errors.append(
                "The frozen evaluation set must contain exactly 16 questions."
            )
        counts = Counter(q.get("category") for q in eval_questions)
        if dict(counts) != EXPECTED_EVAL_CATEGORIES:
            errors.append(
                "Evaluation categories must contain 2 direct, 2 intersection, "
                "6 multi_hop, 4 unsupported and 2 interpretation_trap questions."
            )
    indexed_papers = {paper["id"]: paper for paper in papers_list}
    for question in selected:
        question_id = question.get("id", "unknown")
        reference = question.get("reference_draft")
        if (
            not isinstance(question.get("question"), str)
            or not question["question"].strip()
        ):
            errors.append(f"{question_id}: question text is empty.")
        if not isinstance(reference, dict) or not isinstance(
            reference.get("answer"), str
        ):
            errors.append(f"{question_id}: reference draft is missing.")
            continue
        required = reference.get("required_papers")
        evidence = reference.get("evidence")
        if not isinstance(required, list):
            errors.append(f"{question_id}: required_papers contains an unknown ID.")
            required = []
        elif any(not isinstance(pid, str) or pid not in paper_ids for pid in required):
            errors.append(f"{question_id}: required_papers contains an unknown ID.")
        if all(isinstance(pid, str) for pid in required) and len(required) != len(
            set(required)
        ):
            errors.append(f"{question_id}: required_papers contains duplicate IDs.")
        if type(reference.get("expected_abstention")) is not bool:
            errors.append(f"{question_id}: expected_abstention must be boolean.")
        if question.get("category") == "unsupported":
            if reference.get("expected_abstention") is not True or required:
                errors.append(
                    f"{question_id}: unsupported questions must expect abstention and no required papers."
                )
        elif not reference.get("expected_abstention") and not required:
            errors.append(
                f"{question_id}: answerable questions need required paper IDs."
            )
        if question.get("category") == "multi_hop" and len(required or []) < 2:
            errors.append(
                f"{question_id}: multi-hop path coverage requires at least two required papers."
            )
        if not isinstance(evidence, list) or not evidence:
            errors.append(
                f"{question_id}: reference draft needs literal evidence or a scope basis."
            )
        elif any(not _quote_exists(indexed_papers, item) for item in evidence):
            errors.append(
                f"{question_id}: a reference quote is not literal corpus evidence."
            )
        review = question.get("human_review")
        if require_approved and not reviewers_approved(review):
            errors.append(f"{question_id}: reference is not approved by two reviewers.")
    return errors


def validate_extraction_annotations(
    annotations: Any,
    papers_list: list[dict],
    current_source_hashes: dict[str, str],
    *,
    require_approved: bool = False,
) -> list[str]:
    errors: list[str] = []
    if (
        not isinstance(annotations, dict)
        or annotations.get("schema_version") != ANNOTATION_SCHEMA
    ):
        return [f"Annotation schema must be {ANNOTATION_SCHEMA}."]
    if annotations.get("source_hashes") != current_source_hashes:
        errors.append("Annotation corpus hashes do not match the current corpus files.")
    records = annotations.get("papers")
    if not isinstance(records, list):
        return errors + ["Extraction annotations must contain a papers list."]
    expected = {paper.get("id") for paper in papers_list}
    record_ids = [
        record.get("paper_id") for record in records if isinstance(record, dict)
    ]
    if len(record_ids) != len(records) or any(
        not isinstance(paper_id, str) for paper_id in record_ids
    ):
        errors.append("Annotation paper IDs must be present and unique.")
    elif len(record_ids) != len(set(record_ids)):
        errors.append("Annotation paper IDs must be present and unique.")
    if (
        all(isinstance(paper_id, str) for paper_id in record_ids)
        and set(record_ids) != expected
    ):
        errors.append(
            "Extraction annotations must cover every current paper exactly once."
        )
    paper_index = {paper["id"]: paper for paper in papers_list}
    for record in records:
        if not isinstance(record, dict):
            continue
        paper_id = record.get("paper_id", "unknown")
        if paper_id not in paper_index:
            errors.append(f"{paper_id}: annotation paper is absent from the corpus.")
            continue
        if require_approved and not reviewers_approved(record):
            errors.append(
                f"{paper_id}: extraction annotation is not approved by two reviewers."
            )
        mentions = record.get("mentions")
        if not isinstance(mentions, list):
            errors.append(
                f"{paper_id}: mentions must be a list; [] is a valid reviewed empty set."
            )
            continue
        seen = set()
        paper = paper_index[paper_id]
        for mention in mentions:
            if not isinstance(mention, dict):
                errors.append(f"{paper_id}: mention must be an object.")
                continue
            relation, label = mention.get("relation"), mention.get("label")
            if relation not in RELATION_LABEL or label != RELATION_LABEL.get(relation):
                errors.append(f"{paper_id}: mention label/relation pair is invalid.")
                continue
            surface = mention.get("surface_form")
            field = mention.get("field")
            quote = mention.get("quote")
            canonical_name = mention.get("canonical_name")
            if canonical_name is not None and not isinstance(canonical_name, str):
                errors.append(
                    f"{paper_id}: canonical_name must be a string when present."
                )
            valid_source = (
                field in {"title", "abstract"}
                and isinstance(surface, str)
                and bool(surface.strip())
                and isinstance(quote, str)
                and bool(quote)
                and isinstance(paper.get(field), str)
                and quote in paper[field]
                and surface in quote
            )
            unresolved_pending_candidate = (
                not reviewers_approved(record)
                and mention.get("literal_match") is False
                and surface is None
                and field is None
                and quote is None
            )
            if not valid_source and not unresolved_pending_candidate:
                errors.append(
                    f"{paper_id}: mention lacks exact title/abstract evidence containing its surface form."
                )
            entity_name = (
                canonical_name
                if isinstance(canonical_name, str) and canonical_name
                else surface
            )
            key = (
                relation,
                _entity_key(entity_name if isinstance(entity_name, str) else ""),
            )
            if key in seen:
                errors.append(f"{paper_id}: duplicate relation/entity annotation.")
            seen.add(key)
    return errors


def _entity_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    return "".join(char for char in decomposed if char.isalnum())


def _paper_ids(response: dict) -> list[str]:
    sources = response.get("sources")
    if not isinstance(sources, list):
        return []
    return list(
        dict.fromkeys(
            source.get("paper_id") or source.get("id")
            for source in sources
            if isinstance(source, dict)
            and isinstance(source.get("paper_id") or source.get("id"), str)
        )
    )


def _valid_path(path: Any, response: dict, source_by_id: dict[str, dict]) -> bool:
    if not isinstance(path, dict):
        return False
    length, nodes, edges = path.get("length"), path.get("nodes"), path.get("edges")
    if (
        type(length) is not int
        or length < 1
        or not isinstance(nodes, list)
        or not isinstance(edges, list)
    ):
        return False
    if len(nodes) != length + 1 or len(edges) != length:
        return False
    graph = (
        response.get("subgraph") if isinstance(response.get("subgraph"), dict) else {}
    )
    known_nodes = {
        node.get("id"): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    known_edges = {
        (edge.get("source"), edge.get("type"), edge.get("target"))
        for edge in graph.get("edges", [])
        if isinstance(edge, dict)
        and all(isinstance(edge.get(key), str) for key in ("source", "type", "target"))
    }
    if any(
        not isinstance(node, dict)
        or not isinstance(node.get("id"), str)
        or node.get("id") not in known_nodes
        for node in nodes
    ):
        return False
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict) or any(
            not isinstance(edge.get(key), str) for key in ("source", "type", "target")
        ):
            return False
        signature = (edge.get("source"), edge.get("type"), edge.get("target"))
        adjacent = {nodes[index].get("id"), nodes[index + 1].get("id")}
        if signature not in known_edges or {signature[0], signature[2]} != adjacent:
            return False
    source, target = path.get("source"), path.get("target")
    if (
        not isinstance(source, str)
        or not isinstance(target, str)
        or source not in source_by_id
        or target not in source_by_id
    ):
        return False
    if path.get("kind") != "fact":
        expected_papers = {
            str(source_by_id[source].get("paper_id")),
            str(source_by_id[target].get("paper_id")),
        }
        endpoint_papers = {
            str(node.get("properties", {}).get("id"))
            for node in (nodes[0], nodes[-1])
            if known_nodes[node["id"]].get("label") == "Paper"
        }
        if endpoint_papers != expected_papers:
            return False
    return True


def _response_source_index(response: dict) -> dict[str, dict]:
    sources = response.get("sources") if isinstance(response.get("sources"), list) else []
    indexed = {}
    for index, item in enumerate(sources, start=1):
        if not isinstance(item, dict):
            continue
        citation_id = item.get("citation_id")
        indexed[citation_id if isinstance(citation_id, str) else f"S{index}"] = item
    return indexed


def _citation_validity(response: dict) -> tuple[int, int]:
    claims = response.get("claims")
    if not isinstance(claims, list):
        return 0, 0
    paths = response.get("paths") if isinstance(response.get("paths"), list) else []
    source_by_id = _response_source_index(response)
    path_by_id = {
        path.get("path_id"): path
        for path in paths
        if isinstance(path, dict) and isinstance(path.get("path_id"), str)
    }
    valid_path_ids = {
        path_id
        for path_id, path in path_by_id.items()
        if isinstance(path_id, str) and _valid_path(path, response, source_by_id)
    }
    valid_count = 0
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        source_ids = claim.get("source_ids")
        path_refs = claim.get("path_ids", [])
        quotes = claim.get("quotes", [])
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or any(not isinstance(item, str) for item in source_ids)
            or not isinstance(path_refs, list)
            or any(not isinstance(item, str) for item in path_refs)
            or not isinstance(quotes, list)
        ):
            continue
        if (
            not set(source_ids) <= source_by_id.keys()
            or not set(path_refs) <= valid_path_ids
        ):
            continue
        quote_sources = set()
        literal_quotes_valid = True
        for quote in quotes:
            if not isinstance(quote, dict):
                literal_quotes_valid = False
                break
            citation, field, text = (
                quote.get("source_id"),
                quote.get("field"),
                quote.get("quote"),
            )
            source = source_by_id.get(citation) if isinstance(citation, str) else None
            if (
                source is None
                or not isinstance(field, str)
                or field not in {"title", "abstract"}
                or not isinstance(text, str)
                or not text
            ):
                literal_quotes_valid = False
                break
            original = source.get(field)
            if not isinstance(original, str) or text not in original:
                literal_quotes_valid = False
                break
            quote_sources.add(citation)
        if not literal_quotes_valid:
            continue
        kind = claim.get("kind")
        if kind == "document" and quote_sources != set(source_ids):
            continue
        if kind == "graph":
            if not path_refs:
                continue
            path_citations_valid = True
            for path_id in path_refs:
                path = path_by_id[path_id]
                if not {path.get("source"), path.get("target")} <= set(source_ids):
                    path_citations_valid = False
                    break
            if not path_citations_valid:
                continue
        if not isinstance(kind, str) or kind not in {"document", "graph"}:
            continue
        valid_count += 1
    return valid_count, len(claims)


def _claim_review_counts(
    pair_mode: dict, response: dict
) -> tuple[int, int, int, int, int]:
    claims = response.get("claims") if isinstance(response.get("claims"), list) else []
    reviews = pair_mode.get("claim_reviews")
    if not isinstance(reviews, list):
        return 0, len(claims), 0, 0, 0
    review_by_id = {
        item.get("claim_id"): item
        for item in reviews
        if isinstance(item, dict) and isinstance(item.get("claim_id"), str)
    }
    reviewed = total_ratings = supported = agreement = claim_total = 0
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            claim_total += 1
            continue
        claim_ids = pair_mode.get("claim_ids", [])
        identity = (
            claim_ids[index]
            if isinstance(claim_ids, list) and index < len(claim_ids)
            else None
        )
        review = review_by_id.get(identity) if isinstance(identity, str) else None
        if not review:
            claim_total += 1
            continue
        ratings = review.get("reviewers", [])
        unique_ratings = {}
        if isinstance(ratings, list):
            for rating in ratings:
                if (
                    not isinstance(rating, dict)
                    or not isinstance(rating.get("reviewer_id"), str)
                    or not rating.get("reviewer_id", "").strip()
                    or type(rating.get("supported")) is not bool
                ):
                    continue
                reviewer = rating["reviewer_id"].strip().casefold()
                unique_ratings.setdefault(reviewer, rating["supported"])
        distinct = set(unique_ratings)
        claim_total += 1
        if len(distinct) >= 2:
            values = list(unique_ratings.values())
            reviewed += 1
            total_ratings += len(values)
            supported += sum(values)
            agreement += int(len(set(values)) == 1)
    return reviewed, claim_total, supported, total_ratings, agreement


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _mode_metrics(question_pairs: list[tuple[dict, dict]], mode: str) -> dict[str, Any]:
    precision_values = []
    recall_values = []
    multipath_coverage = []
    multipath_complete = []
    multipath_path_coverage = []
    multipath_path_cited = []
    abstentions = 0
    abstention_correct = 0
    citation_valid = citation_total = 0
    latencies = []
    failures = 0
    planner_statuses: Counter[str] = Counter()
    planner_valid = planner_attempted = planner_successfully_executed = 0
    claim_count = claims_supported = claims_reviewed = claim_reviews_total = (
        reviewers_agree
    ) = 0
    for question, outcome in question_pairs:
        if outcome.get("failure"):
            failures += 1
            continue
        response = outcome.get("response")
        if not isinstance(response, dict):
            failures += 1
            continue
        if isinstance(outcome.get("latency_seconds"), (int, float)):
            latencies.append(float(outcome["latency_seconds"]))
        claims = (
            response.get("claims") if isinstance(response.get("claims"), list) else []
        )
        claim_count += len(claims)
        if isinstance(response.get("abstained"), bool):
            abstentions += int(response["abstained"])
            abstention_correct += int(
                response["abstained"]
                == question["reference_draft"]["expected_abstention"]
            )
        valid, total = _citation_validity(response)
        citation_valid += valid
        citation_total += total
        returned = set(_paper_ids(response))
        expected = set(question["reference_draft"].get("required_papers", []))
        if expected:
            precision = _ratio(len(returned & expected), len(returned))
            recall = _ratio(len(returned & expected), len(expected))
            if precision is not None:
                precision_values.append(precision)
            if recall is not None:
                recall_values.append(recall)
        if question.get("category") == "multi_hop" and expected:
            coverage = _ratio(len(returned & expected), len(expected))
            multipath_coverage.append(coverage if coverage is not None else 0.0)
            multipath_complete.append(int(expected <= returned))
            source_by_id = _response_source_index(response)
            valid_paths = []
            for path in response.get("paths", []):
                if (
                    not _valid_path(path, response, source_by_id)
                    or path.get("length", 0) < 2
                ):
                    continue
                path_papers = {
                    source_by_id[citation].get("paper_id")
                    for citation in (path.get("source"), path.get("target"))
                    if citation in source_by_id
                }
                if len(path_papers) == 2 and path_papers <= expected:
                    valid_paths.append(path)
            multipath_path_coverage.append(int(bool(valid_paths)))
            cited_path_ids = set()
            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                source_ids = claim.get("source_ids", [])
                path_ids = claim.get("path_ids", [])
                if (
                    not isinstance(source_ids, list)
                    or any(not isinstance(item, str) for item in source_ids)
                    or not isinstance(path_ids, list)
                    or any(not isinstance(item, str) for item in path_ids)
                ):
                    continue
                for path_id in path_ids:
                    path = next(
                        (
                            item
                            for item in valid_paths
                            if item.get("path_id") == path_id
                        ),
                        None,
                    )
                    if path and {path.get("source"), path.get("target")} <= set(
                        source_ids
                    ):
                        cited_path_ids.add(path_id)
            multipath_path_cited.append(
                int(any(path.get("path_id") in cited_path_ids for path in valid_paths))
            )
        planner = (
            response.get("planner") if isinstance(response.get("planner"), dict) else {}
        )
        status = str(planner.get("status", "missing")).lower()
        planner_statuses[status] += 1
        if status not in {"disabled", "missing"}:
            planner_attempted += 1
            if status in {"executed", "empty"}:
                planner_valid += 1
                planner_successfully_executed += 1
        reviewed, total_claims, supported, ratings, agree = _claim_review_counts(
            outcome, response
        )
        claims_reviewed += reviewed
        claim_count += 0
        claims_supported += supported
        claim_reviews_total += ratings
        reviewers_agree += agree
    return {
        "completed_responses": len(question_pairs) - failures,
        "failures": failures,
        "source_precision_macro": _ratio(sum(precision_values), len(precision_values)),
        "source_recall_macro": _ratio(sum(recall_values), len(recall_values)),
        "source_metric_questions": len(precision_values),
        "multi_hop_required_paper_coverage_macro": _ratio(
            sum(multipath_coverage), len(multipath_coverage)
        ),
        "multi_hop_all_required_papers_covered": _ratio(
            sum(multipath_complete), len(multipath_complete)
        ),
        "multi_hop_valid_path_coverage": _ratio(
            sum(multipath_path_coverage), len(multipath_path_coverage)
        ),
        "multi_hop_valid_path_cited_coverage": _ratio(
            sum(multipath_path_cited), len(multipath_path_cited)
        ),
        "multi_hop_questions": len(multipath_coverage),
        "deterministic_citation_validity": _ratio(citation_valid, citation_total),
        "deterministic_claims_checked": citation_total,
        "abstention_rate": _ratio(abstentions, len(question_pairs) - failures),
        "abstention_accuracy": _ratio(
            abstention_correct, len(question_pairs) - failures
        ),
        "mean_latency_seconds": round(sum(latencies) / len(latencies), 4)
        if latencies
        else None,
        "planner": {
            "status_counts": dict(sorted(planner_statuses.items())),
            "valid_rate_among_attempts": _ratio(planner_valid, planner_attempted),
            "execution_success_rate_among_attempts": _ratio(
                planner_successfully_executed, planner_attempted
            ),
            "attempted": planner_attempted,
            "successfully_executed": planner_successfully_executed,
            "disabled_or_missing_not_in_denominator": planner_statuses.get(
                "disabled", 0
            )
            + planner_statuses.get("missing", 0),
        },
        "human_claim_review": {
            "claims": claim_count,
            "claims_reviewed_by_two": claims_reviewed,
            "review_completion": _ratio(claims_reviewed, claim_count),
            "supported_label_fraction": _ratio(claims_supported, claim_reviews_total),
            "reviewer_agreement": _ratio(reviewers_agree, claims_reviewed),
        },
    }


def _mode_outcome(pair: dict, mode: str) -> dict | None:
    item = pair.get(mode)
    return item if isinstance(item, dict) else None


def _outcome_attempts(outcome: dict | None) -> list[dict]:
    if not isinstance(outcome, dict):
        return []
    history = outcome.get("attempt_history", [])
    prior = (
        [item for item in history if isinstance(item, dict)]
        if isinstance(history, list)
        else []
    )
    return [*prior, outcome]


def build_retrieval_report(
    run: Any,
    question_bank: dict,
    current_source_hashes: dict[str, str],
    *,
    papers_list: list[dict],
) -> dict[str, Any]:
    """Summarize only hash-matched complete pairs; pending judgments stay inconclusive."""
    problems: list[str] = []
    if (
        not isinstance(run, dict)
        or run.get("schema_version") != "paired-retrieval-run-v1"
    ):
        return {
            "status": "inconclusive",
            "problems": ["Run file schema is invalid."],
            "metrics": None,
        }
    manifest = run.get("manifest")
    if not isinstance(manifest, dict):
        return {
            "status": "inconclusive",
            "problems": ["Run manifest is missing."],
            "metrics": None,
        }
    manifest_hash = sha256_json(manifest)
    if run.get("manifest_sha256") != manifest_hash:
        problems.append("Run manifest hash does not match its contents.")
    if manifest.get("corpus_hashes") != current_source_hashes:
        problems.append("Run corpus hashes do not match the current corpus files.")
    graph_snapshot_hash = manifest.get("graph_snapshot_sha256")
    if not isinstance(graph_snapshot_hash, str) or len(graph_snapshot_hash) != 64:
        problems.append("Run manifest has no valid graph snapshot hash.")
    if run.get("invalidated"):
        problems.append("Run checkpoint was invalidated by a graph snapshot mismatch.")
    split = manifest.get("split")
    if split not in {"dev", "eval"}:
        problems.append("Run manifest split is invalid.")
    bank_errors = validate_question_bank(
        question_bank,
        papers_list,
        current_source_hashes,
        split=split if split in {"dev", "eval"} else None,
    )
    if bank_errors:
        problems.extend(f"Question bank: {message}" for message in bank_errors)
    questions = {
        question["id"]: question
        for question in question_bank.get("questions", [])
        if question.get("split") == split
    }
    if any(not reviewers_approved(q.get("human_review")) for q in questions.values()):
        problems.append(
            "One or more reference drafts lack approval from two independent reviewers."
        )
    question_ids = manifest.get("question_ids")
    if question_ids != list(questions):
        # Keep stable bank order for manifest validation.
        expected_ids = [
            q["id"]
            for q in question_bank.get("questions", [])
            if q.get("split") == split
        ]
        if question_ids != expected_ids:
            problems.append(
                "Run manifest question IDs do not match the selected split."
            )
    expected_question_hashes = {
        qid: question_fingerprint(question) for qid, question in questions.items()
    }
    if manifest.get("question_hashes") != expected_question_hashes:
        problems.append(
            "Run manifest question/reference hashes do not match the selected split."
        )
    pairs = run.get("pairs")
    if not isinstance(pairs, list):
        return {
            "status": "inconclusive",
            "problems": problems + ["Run has no pairs list."],
            "metrics": None,
        }
    indexed: dict[str, dict] = {}
    planner_status_counts: Counter[str] = Counter()
    observed_retrieval_effective_models: set[str] = set()
    observed_cypher_effective_models: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, dict) or not isinstance(pair.get("question_id"), str):
            problems.append("Run contains a malformed pair entry.")
            continue
        qid = pair["question_id"]
        if qid in indexed:
            problems.append(f"Run contains duplicate pair for {qid}.")
            continue
        indexed[qid] = pair
        question = questions.get(qid)
        if question is None:
            problems.append(f"Run contains question {qid} outside the selected split.")
            continue
        if pair.get("question_hash") != question_fingerprint(question):
            problems.append(f"Pair {qid} question/reference hash mismatch.")
        if pair.get("manifest_sha256") != manifest_hash:
            problems.append(f"Pair {qid} run-manifest hash mismatch.")
        if pair.get("top_k") != manifest.get("top_k"):
            problems.append(f"Pair {qid} top_k does not match the run manifest.")
        if pair.get("graph_snapshot_before_sha256") != graph_snapshot_hash:
            problems.append(
                f"Pair {qid} has no matching pre-pair graph snapshot check."
            )
        if pair.get("graph_snapshot_after_sha256") != graph_snapshot_hash:
            problems.append(
                f"Pair {qid} has no matching post-pair graph snapshot check."
            )
        for mode in ("vector", "hybrid"):
            outcome = _mode_outcome(pair, mode)
            if outcome is None:
                continue
            if not isinstance(outcome.get("attempt_history", []), list):
                problems.append(f"Pair {qid} {mode} attempt history is malformed.")
            for attempt in _outcome_attempts(outcome):
                if attempt.get("question_hash") != question_fingerprint(question):
                    problems.append(f"Pair {qid} {mode} question hash mismatch.")
                if attempt.get("manifest_sha256") != manifest_hash:
                    problems.append(f"Pair {qid} {mode} manifest hash mismatch.")
                if attempt.get("model_sha256") != manifest.get("model_sha256"):
                    problems.append(
                        f"Pair {qid} {mode} model configuration hash mismatch."
                    )
                if attempt.get("failure") is None:
                    response = attempt.get("response")
                    if not isinstance(response, dict) or response.get("mode") != mode:
                        problems.append(
                            f"Pair {qid} {mode} response mode is missing or mismatched."
                        )
                    if isinstance(response, dict):
                        generation = response.get("generation")
                        if isinstance(generation, dict):
                            observed = generation.get("effective_model")
                            if isinstance(observed, str) and observed.strip():
                                observed_retrieval_effective_models.add(
                                    observed.strip()
                                )
                        planner = response.get("planner")
                        planner_receipt = (
                            planner.get("llm") if isinstance(planner, dict) else None
                        )
                        if isinstance(planner_receipt, dict):
                            observed = planner_receipt.get("effective_model")
                            if isinstance(observed, str) and observed.strip():
                                observed_cypher_effective_models.add(observed.strip())
                elif not isinstance(attempt.get("failure"), dict):
                    problems.append(f"Pair {qid} {mode} failure record is malformed.")
    if len(observed_retrieval_effective_models) > 1:
        problems.append(
            "Vector/hybrid retrieval receipts report different effective models across pairs."
        )
    if len(observed_cypher_effective_models) > 1:
        problems.append(
            "Hybrid Cypher planner receipts report different effective models across pairs."
        )
    missing_ids = set(questions) - set(indexed)
    if missing_ids:
        problems.append(f"Run is missing {len(missing_ids)} question pairs.")

    complete_pairs = []
    planner_eligible_pairs = []
    for qid, question in questions.items():
        pair = indexed.get(qid)
        if pair is None:
            continue
        vector, hybrid = _mode_outcome(pair, "vector"), _mode_outcome(pair, "hybrid")
        if not vector or not hybrid or vector.get("failure") or hybrid.get("failure"):
            continue
        if not isinstance(vector.get("response"), dict) or not isinstance(
            hybrid.get("response"), dict
        ):
            continue
        complete_pairs.append((question, pair))
        planner = hybrid["response"].get("planner")
        status = (
            str(planner.get("status", "missing")).lower()
            if isinstance(planner, dict)
            else "missing"
        )
        if status in {"executed", "empty"}:
            planner_eligible_pairs.append((question, pair))

    mode_question_pairs = {
        mode: [(question, pair[mode]) for question, pair in complete_pairs]
        for mode in ("vector", "hybrid")
    }
    eligible_mode_question_pairs = {
        mode: [(question, pair[mode]) for question, pair in planner_eligible_pairs]
        for mode in ("vector", "hybrid")
    }
    metrics = (
        {
            mode: _mode_metrics(items, mode)
            for mode, items in eligible_mode_question_pairs.items()
        }
        if planner_eligible_pairs
        else None
    )
    if len(complete_pairs) != len(questions):
        problems.append(
            "Not every selected question has a complete vector/hybrid pair."
        )
    claim_review_incomplete = False
    for mode, items in mode_question_pairs.items():
        for _, outcome in items:
            response = outcome.get("response", {})
            human = _claim_review_counts(outcome, response)
            if human[0] != human[1]:
                claim_review_incomplete = True
    if claim_review_incomplete:
        problems.append(
            "Some generated claims have not been labeled by two independent reviewers."
        )
    if len(planner_eligible_pairs) != len(questions):
        problems.append(
            "Hybrid planner did not validly execute for every selected pair; degraded/disabled pairs are excluded from the main comparison."
        )
    if not planner_eligible_pairs:
        problems.append(
            "No hybrid planner query has a valid executed or empty result; no main comparison is available."
        )
    if not manifest.get("model_sha256"):
        problems.append("Run manifest has no model configuration hash.")
    if (
        not manifest.get("prompt_sha256")
        or not manifest.get("schema_sha256")
        or not manifest.get("grammar_config_sha256")
    ):
        problems.append(
            "Run manifest is missing prompt, schema, or grammar configuration hashes."
        )
    status = (
        "complete"
        if not problems and len(complete_pairs) == len(questions)
        else "inconclusive"
    )
    availability = {
        "selected_questions": len(questions),
        "pairs_present": len(indexed),
        "complete_vector_hybrid_pairs": len(complete_pairs),
        "planner_eligible_pairs": len(planner_eligible_pairs),
        "missing_pairs": len(set(questions) - set(indexed)),
        "vector_responses": 0,
        "vector_failures": 0,
        "vector_missing": 0,
        "hybrid_responses": 0,
        "hybrid_failures": 0,
        "hybrid_missing": 0,
    }
    planner_attempts = planner_valid_queries = planner_execution_attempts = 0
    planner_execution_successes = planner_nonempty_executions = 0
    for qid in questions:
        pair = indexed.get(qid, {})
        outcome = _mode_outcome(pair, "hybrid") if pair else None
        attempts = _outcome_attempts(outcome)
        if not attempts:
            planner_status_counts["missing"] += 1
            continue
        for attempt in attempts:
            if attempt.get("failure"):
                planner_status_counts["call_failure"] += 1
                continue
            response = attempt.get("response")
            planner = response.get("planner") if isinstance(response, dict) else None
            planner_status = (
                str(planner.get("status", "missing")).lower()
                if isinstance(planner, dict)
                else "missing"
            )
            planner_status_counts[planner_status] += 1
            if planner_status in {"disabled", "missing"}:
                continue
            planner_attempts += 1
            diagnostics = (
                planner.get("queries", []) if isinstance(planner, dict) else []
            )
            query_was_validated = any(
                isinstance(query, dict)
                and isinstance(query.get("canonical"), str)
                and isinstance(query.get("cypher"), str)
                for query in diagnostics
            )
            query_was_executed = any(
                isinstance(query, dict) and query.get("attempted") is True
                for query in diagnostics
            )
            if planner_status in {"executed", "empty"} or query_was_validated:
                planner_valid_queries += 1
            if planner_status in {"executed", "empty"} or query_was_executed:
                planner_execution_attempts += 1
            if planner_status in {"executed", "empty"}:
                planner_execution_successes += 1
            if planner_status == "executed":
                planner_nonempty_executions += 1
    availability["hybrid_planner_status_counts"] = dict(
        sorted(planner_status_counts.items())
    )
    availability["hybrid_planner"] = {
        "attempted": planner_attempts,
        "disabled_or_missing_not_in_denominator": (
            planner_status_counts.get("disabled", 0)
            + planner_status_counts.get("missing", 0)
        ),
        "validated_query_rate_among_attempts": _ratio(
            planner_valid_queries, planner_attempts
        ),
        "execution_attempt_rate_among_attempts": _ratio(
            planner_execution_attempts, planner_attempts
        ),
        "successful_execution_rate_among_attempts": _ratio(
            planner_execution_successes, planner_attempts
        ),
        "nonempty_execution_rate_among_attempts": _ratio(
            planner_nonempty_executions, planner_attempts
        ),
        "successfully_executed": planner_execution_successes,
        "nonempty_executions": planner_nonempty_executions,
    }
    for qid in questions:
        pair = indexed.get(qid, {})
        for mode in ("vector", "hybrid"):
            outcome = _mode_outcome(pair, mode) if pair else None
            attempts = _outcome_attempts(outcome)
            availability[f"{mode}_attempts"] = availability.get(
                f"{mode}_attempts", 0
            ) + len(attempts)
            for attempt in attempts:
                if isinstance(attempt.get("latency_seconds"), (int, float)):
                    availability.setdefault(f"_{mode}_latencies", []).append(
                        float(attempt["latency_seconds"])
                    )
                if attempt.get("failure"):
                    availability[f"{mode}_failures"] += 1
            if (
                outcome is None
                or outcome.get("failure")
                or not isinstance(outcome.get("response"), dict)
            ):
                availability[f"{mode}_missing"] += 1
            else:
                availability[f"{mode}_responses"] += 1
    for mode in ("vector", "hybrid"):
        latencies = availability.pop(f"_{mode}_latencies", [])
        availability[f"{mode}_latency_observations"] = len(latencies)
        availability[f"{mode}_mean_latency_seconds"] = (
            round(sum(latencies) / len(latencies), 4) if latencies else None
        )
    availability["mode_failures_total"] = (
        availability["vector_failures"] + availability["hybrid_failures"]
    )
    return {
        "status": status,
        "split": split,
        "question_count": len(questions),
        "complete_pairs": len(complete_pairs),
        "planner_eligible_pairs": len(planner_eligible_pairs),
        "ignored_graph_relationship_types": run.get(
            "ignored_graph_relationship_types", []
        ),
        "availability": availability,
        "problems": problems,
        "metrics": metrics,
        "paired_deltas": _paired_deltas(metrics)
        if status == "complete" and metrics
        else None,
        "interpretation": (
            "Comparative deltas are available only when every pair, gold reference, model/config hash, and human claim label is complete."
            if status != "complete"
            else "Descriptive paired results; claim support labels are human judgments, while citation validity is deterministic structure/evidence validation."
        ),
    }


def _paired_deltas(metrics: dict) -> dict[str, float | None]:
    fields = (
        "source_precision_macro",
        "source_recall_macro",
        "multi_hop_required_paper_coverage_macro",
        "multi_hop_all_required_papers_covered",
        "deterministic_citation_validity",
        "abstention_accuracy",
    )
    deltas = {}
    for field in fields:
        vector, hybrid = metrics["vector"].get(field), metrics["hybrid"].get(field)
        deltas[field] = (
            round(hybrid - vector, 4)
            if isinstance(vector, (int, float)) and isinstance(hybrid, (int, float))
            else None
        )
    vector_claims = metrics["vector"]["human_claim_review"]["supported_label_fraction"]
    hybrid_claims = metrics["hybrid"]["human_claim_review"]["supported_label_fraction"]
    deltas["human_supported_claim_label_fraction"] = (
        round(hybrid_claims - vector_claims, 4)
        if isinstance(vector_claims, (int, float))
        and isinstance(hybrid_claims, (int, float))
        else None
    )
    return deltas


def compare_extractions(
    predictions: Any,
    annotations: dict,
    papers_list: list[dict],
    current_source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Compare enriched mentions only against two-reviewer-approved annotations."""
    errors = validate_extraction_annotations(
        annotations, papers_list, current_source_hashes, require_approved=True
    )
    if errors:
        return {"status": "inconclusive", "problems": errors, "metrics": None}
    if not isinstance(predictions, list):
        return {
            "status": "inconclusive",
            "problems": ["Prediction file must be a JSON list."],
            "metrics": None,
        }
    paper_index = {paper["id"]: paper for paper in papers_list}
    prediction_index = {
        item.get("id"): item
        for item in predictions
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    problems = []
    if set(prediction_index) != set(paper_index) or len(prediction_index) != len(
        predictions
    ):
        problems.append(
            "Prediction papers do not cover the current corpus exactly once."
        )
    for paper_id, predicted in prediction_index.items():
        original = paper_index.get(paper_id)
        if original is None or any(
            predicted.get(field) != original.get(field)
            for field in ("title", "abstract")
        ):
            problems.append(
                f"Prediction content for {paper_id} does not match the current corpus."
            )
    if problems:
        return {"status": "inconclusive", "problems": problems, "metrics": None}

    expected_mentions: set[tuple[str, str, str, str]] = set()
    predicted_mentions: set[tuple[str, str, str, str]] = set()
    gold_evidence = Counter()
    predicted_evidence = Counter()
    gold_counts = Counter()
    predicted_counts = Counter()
    for record in annotations["papers"]:
        paper_id = record["paper_id"]
        for mention in record["mentions"]:
            canonical_name = mention.get("canonical_name")
            entity_name = (
                canonical_name
                if isinstance(canonical_name, str) and canonical_name
                else mention.get("surface_form")
            )
            key = (
                paper_id,
                mention["label"],
                mention["relation"],
                _entity_key(entity_name if isinstance(entity_name, str) else ""),
            )
            expected_mentions.add(key)
            gold_counts[(mention["label"], mention["relation"])] += 1
            gold_evidence[(mention["label"], mention["relation"])] += int(
                _gold_evidence_valid(mention, paper_index[paper_id])
            )
    for paper_id, record in prediction_index.items():
        extracted = (
            record.get("extracted") if isinstance(record.get("extracted"), dict) else {}
        )
        mentions = (
            extracted.get("mentions")
            if isinstance(extracted.get("mentions"), list)
            else []
        )
        for mention in mentions:
            if (
                not isinstance(mention, dict)
                or mention.get("relation") not in RELATION_LABEL
            ):
                continue
            label = mention.get("label")
            if label != RELATION_LABEL[mention["relation"]]:
                continue
            surface = mention.get("canonical_name") or mention.get("surface_form")
            if not isinstance(surface, str):
                continue
            key = (paper_id, label, mention["relation"], _entity_key(surface))
            predicted_mentions.add(key)
            group = (label, mention["relation"])
            predicted_counts[group] += 1
            predicted_evidence[group] += int(
                _prediction_evidence_valid(mention, paper_index[paper_id])
            )

    tp = len(expected_mentions & predicted_mentions)
    overall = _prf(tp, len(predicted_mentions), len(expected_mentions))
    labels = sorted({key[1] for key in expected_mentions | predicted_mentions})
    relations = sorted(RELATION_LABEL)
    by_label = {}
    for label in labels:
        gold = {key for key in expected_mentions if key[1] == label}
        pred = {key for key in predicted_mentions if key[1] == label}
        by_label[label] = {
            **_prf(len(gold & pred), len(pred), len(gold)),
            "predicted_evidence_coverage": _ratio(
                sum(
                    value
                    for (candidate_label, _), value in predicted_evidence.items()
                    if candidate_label == label
                ),
                sum(
                    value
                    for (candidate_label, _), value in predicted_counts.items()
                    if candidate_label == label
                ),
            ),
            "gold_evidence_coverage": _ratio(
                sum(
                    value
                    for (candidate_label, _), value in gold_evidence.items()
                    if candidate_label == label
                ),
                sum(
                    value
                    for (candidate_label, _), value in gold_counts.items()
                    if candidate_label == label
                ),
            ),
        }
    by_relation = {}
    for relation in relations:
        gold = {key for key in expected_mentions if key[2] == relation}
        pred = {key for key in predicted_mentions if key[2] == relation}
        label = RELATION_LABEL[relation]
        group = (label, relation)
        by_relation[relation] = {
            **_prf(len(gold & pred), len(pred), len(gold)),
            "predicted_evidence_coverage": _ratio(
                predicted_evidence[group], predicted_counts[group]
            ),
            "gold_evidence_coverage": _ratio(gold_evidence[group], gold_counts[group]),
        }
    return {
        "status": "complete",
        "problems": [],
        "metrics": {
            **overall,
            "gold_mentions": len(expected_mentions),
            "predicted_mentions": len(predicted_mentions),
            "by_label": by_label,
            "by_relation": by_relation,
            "predicted_evidence_coverage": _ratio(
                sum(predicted_evidence.values()), sum(predicted_counts.values())
            ),
            "gold_evidence_coverage": _ratio(
                sum(gold_evidence.values()), sum(gold_counts.values())
            ),
        },
    }


def _gold_evidence_valid(mention: dict, paper: dict) -> bool:
    field, quote, surface = (
        mention.get("field"),
        mention.get("quote"),
        mention.get("surface_form"),
    )
    return (
        field in {"title", "abstract"}
        and isinstance(quote, str)
        and quote in paper.get(field, "")
        and isinstance(surface, str)
        and surface in quote
    )


def _prediction_evidence_valid(mention: dict, paper: dict) -> bool:
    return _gold_evidence_valid(mention, paper)


def _prf(true_positive: int, predicted: int, gold: int) -> dict[str, float | None]:
    precision = _ratio(true_positive, predicted)
    recall = _ratio(true_positive, gold)
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {"precision": precision, "recall": recall, "f1": f1}
