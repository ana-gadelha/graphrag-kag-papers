"""Offline-first command line for retrieval and extraction evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .benchmark import (
    corpus_hashes,
    load_corpus,
    prepare_files,
)
from .metrics import (
    build_retrieval_report,
    compare_extractions,
    reviewers_approved,
    validate_extraction_annotations,
    validate_question_bank,
)
from .runner import (
    build_manifest,
    graph_snapshot_session,
    load_or_create_checkpoint,
    run_paired,
    save_checkpoint,
    stage_settings_for_run,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "evaluation/data/questions.json"
DEFAULT_ANNOTATIONS = ROOT / "evaluation/data/extraction_annotations.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_starters(
    args: argparse.Namespace,
) -> tuple[list[dict], dict, dict, dict[str, str]]:
    papers, _ = load_corpus(ROOT)
    hashes = corpus_hashes(ROOT)
    bank = _read_json(args.questions)
    annotations = _read_json(args.annotations)
    return papers, bank, annotations, hashes


def _review_status(bank: dict, annotations: dict) -> dict[str, int]:
    counts = {
        "question_approved": 0,
        "question_pending": 0,
        "extraction_approved": 0,
        "extraction_pending": 0,
    }
    for question in bank.get("questions", []):
        if reviewers_approved(question.get("human_review")):
            counts["question_approved"] += 1
        else:
            counts["question_pending"] += 1
    for record in annotations.get("papers", []):
        if reviewers_approved(record):
            counts["extraction_approved"] += 1
        else:
            counts["extraction_pending"] += 1
    return counts


def _command_prepare(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    written = prepare_files(ROOT, output_dir)
    if written:
        print("Created offline starter files:")
        for path in written:
            print(f"- {path}")
    else:
        print(
            f"Preserved existing files in {output_dir}; no reviewer data was overwritten."
        )
    return 0


def _command_check(args: argparse.Namespace) -> int:
    try:
        papers, bank, annotations, hashes = _load_starters(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {"status": "invalid", "error": str(exc)}, ensure_ascii=False, indent=2
            )
        )
        return 2
    errors = validate_question_bank(bank, papers, hashes)
    errors += validate_extraction_annotations(annotations, papers, hashes)
    review = _review_status(bank, annotations)
    pending = review["question_pending"] + review["extraction_pending"]
    if args.require_reviewed and pending:
        errors.append(
            "Human review is incomplete; two reviewers must approve every question and paper annotation."
        )
    report = {
        "status": "valid_with_pending_review"
        if not errors and pending
        else "valid"
        if not errors
        else "invalid",
        "question_count": len(bank.get("questions", [])),
        "extraction_paper_count": len(annotations.get("papers", [])),
        "review": review,
        "errors": errors,
        "provider_calls": 0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if errors else 0


def _command_report(args: argparse.Namespace) -> int:
    try:
        papers, bank, annotations, hashes = _load_starters(args)
        if args.run_file:
            run = _read_json(args.run_file)
            result = build_retrieval_report(run, bank, hashes, papers_list=papers)
            result["provider_calls"] = 0
        elif args.predictions:
            predictions = _read_json(args.predictions)
            result = compare_extractions(predictions, annotations, papers, hashes)
            result["provider_calls"] = 0
        else:
            raise ValueError(
                "Choose --run-file or --predictions for an offline report."
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "inconclusive",
                    "problems": [str(exc)],
                    "metrics": None,
                    "provider_calls": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "complete" else 3


def _selected_questions(bank: dict, split: str) -> list[dict]:
    return [
        question for question in bank["questions"] if question.get("split") == split
    ]


def _command_run(args: argparse.Namespace) -> int:
    if not args.allow_provider_export:
        print(
            "Run blocked: pass --allow-provider-export to permit sending benchmark questions and retrieved corpus context to configured providers.",
            file=sys.stderr,
        )
        return 2
    try:
        papers, bank, _, hashes = _load_starters(args)
        errors = validate_question_bank(
            bank, papers, hashes, split=args.split, require_approved=True
        )
        if errors:
            raise ValueError("Gold/reference gate failed: " + " ".join(errors))
        questions = _selected_questions(bank, args.split)
        retrieval_settings, cypher_settings, embedding_model = stage_settings_for_run()
        output = args.output or ROOT / f"evaluation/results/{args.split}.json"
        output = output.resolve()
        expected_ids = {paper["id"] for paper in papers}
        with graph_snapshot_session(expected_ids) as (graph_snapshot, snapshot_check):
            manifest = build_manifest(
                ROOT,
                bank,
                args.split,
                args.top_k,
                retrieval_settings=retrieval_settings,
                cypher_settings=cypher_settings,
                embedding_model=embedding_model,
                graph_snapshot=graph_snapshot,
            )
            checkpoint = load_or_create_checkpoint(output, manifest, resume=args.resume)
            checkpoint["ignored_graph_relationship_types"] = graph_snapshot.get(
                "ignored_relation_types", []
            )
            # This import is deliberately below both the provider-export and reviewed-gold gates.
            from retrieval.answer import answer_question

            checkpoint = run_paired(
                questions, manifest, checkpoint, output, answer_question, snapshot_check
            )
            save_checkpoint(output, checkpoint)
        complete = 0
        failures = 0
        for pair in checkpoint["pairs"]:
            if all(
                isinstance(pair.get(mode), dict)
                and not pair[mode].get("failure")
                and isinstance(pair[mode].get("response"), dict)
                for mode in ("vector", "hybrid")
            ):
                complete += 1
            failures += sum(
                bool(pair.get(mode, {}).get("failure")) for mode in ("vector", "hybrid")
            )
        print(
            json.dumps(
                {
                    "status": "checkpoint_saved",
                    "split": args.split,
                    "questions": len(questions),
                    "complete_pairs": complete,
                    "mode_failures": failures,
                    "output": str(output),
                    "manifest_sha256": checkpoint["manifest_sha256"],
                    "provider_calls": "not counted here; consult the LLM ledger",
                    "note": "A saved run is not a quality result; run `report` after human claim review.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        # The checkpoint is retained for resume; exception details may expose provider settings.
        print(
            json.dumps(
                {
                    "status": "run_incomplete",
                    "error_type": type(exc).__name__,
                    "message": str(exc)
                    if isinstance(exc, ValueError)
                    else "Review configuration and checkpoint; details omitted.",
                    "provider_export_authorized": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation",
        description="Prepare and audit retrieval evaluation offline; provider-backed runs require explicit authorization.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Create missing starter data without network access or overwriting review work.",
    )
    prepare.add_argument("--output-dir", type=Path, default=ROOT / "evaluation/data")
    prepare.set_defaults(handler=_command_prepare)

    check = commands.add_parser(
        "check",
        help="Check corpus hashes, schema, evidence quotes and review status offline.",
    )
    check.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    check.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    check.add_argument("--require-reviewed", action="store_true")
    check.set_defaults(handler=_command_check)

    report = commands.add_parser(
        "report",
        help="Generate an offline report from a saved run or extraction output.",
    )
    source = report.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-file", type=Path)
    source.add_argument(
        "--predictions",
        type=Path,
        help="Enriched JSON output from src.extract_entities.",
    )
    report.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    report.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    report.add_argument("--output", type=Path)
    report.set_defaults(handler=_command_report)

    run = commands.add_parser(
        "run",
        help="Execute vector/hybrid pairs after review and explicit provider-export authorization.",
    )
    run.add_argument("--split", choices=("dev", "eval"), default="eval")
    run.add_argument("--top-k", type=int, default=6)
    run.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    run.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    run.add_argument("--output", type=Path)
    run.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when all saved input hashes match.",
    )
    run.add_argument(
        "--allow-provider-export",
        action="store_true",
        required=True,
        help="Authorize sending these questions and retrieved evidence to the configured retrieval/Cypher providers.",
    )
    run.set_defaults(handler=_command_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
