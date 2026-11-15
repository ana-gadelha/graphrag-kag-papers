# Retrieval and extraction evaluation

This directory contains an offline-first evaluation harness for the 26 records
in `data/papers.json`. It compares vector-only and hybrid retrieval with the
same question, `top_k`, retrieval model settings, prompts, and configuration
hashes. The question bank contains six development drafts and 16 frozen
evaluation drafts: 2 direct, 2 intersection, 6 multi-hop, 4 unsupported, and 2
`USES_METHOD`/`PROPOSES` interpretation traps.

**The bundled references and extraction annotations are unreviewed drafts.**
Every question and every paper annotation starts as `PENDING`. The current
`data/curated_entities.json` values are candidate labels only; this package does
not treat them as confirmed gold. A `check` result means the files are
well-formed and their literal quotes match the current corpus. It does not mean
that a person has verified their meaning.

## Offline commands

Run these from the repository root. `prepare`, `check`, and `report` do not call
Neo4j or language model providers.

```bash
python3 -m evaluation prepare
python3 -m evaluation check
python3 -m evaluation check --require-reviewed
```

`prepare` creates missing starter files and preserves any file that already
exists. To create a separate copy for review:

```bash
python3 -m evaluation prepare --output-dir /tmp/graphrag-evaluation-review
```

`check` validates source hashes, the 6/16 question counts, category counts,
paper IDs, literal evidence, annotation coverage for all 26 papers, and review
status. `--require-reviewed` also requires two distinct reviewer IDs for every
question and paper annotation.

## Human review before a run

Review each reference in `evaluation/data/questions.json` against the local
title, abstract, author metadata, and keywords. Confirm the required paper IDs,
the expected answer or abstention, and the supplied evidence. For each reviewed
question, set `human_review.status` to `APPROVED` and list two distinct reviewer
IDs in `human_review.reviewers`. Do not approve a question just because its
quoted text appears literally; the reviewers must confirm that it supports the
reference and that an unsupported question really is unanswerable from this
corpus.

For extraction evaluation, review all 26 records in
`evaluation/data/extraction_annotations.json`. Correct the candidate mention
list from the paper's title and abstract, including relation direction:

- `PROPOSES` means the paper presents its own method.
- `USES_METHOD` means it uses, evaluates, compares with, or adopts an existing
  method. A baseline or comparison does not become a proposed method.
- `EVALUATED_ON` requires explicit evidence that the named dataset was used for
  evaluation.

Every accepted mention needs a literal `surface_form`, `label`, `relation`,
`field` (`title` or `abstract`), and exact `quote` containing that surface.
Empty `mentions` is valid only after reviewers confirm that no relation is
supported. Mark the paper record `status: "APPROVED"` and add two distinct
reviewer IDs only after its entire mention list has been checked. The draft
`draft_from_curated_entities` field is retained for provenance and is not
scored as gold.

Recheck before running:

```bash
python3 -m evaluation check --require-reviewed
```

## Paired retrieval run

The run command requires a deliberate provider-export flag and two-reviewer
approval for every question in the selected split. It sends each question to
the configured retrieval and Cypher providers through the application's
`answer_question()` path. Retrieved titles, abstracts, graph evidence, and
questions may be sent to those providers. The existing LLM cache and request
budget still apply. No inference is started by this README's other commands.

After the selected question references are approved, execute the frozen set:

```bash
python3 -m evaluation run --split eval --top-k 6 --allow-provider-export
```

The default checkpoint is `evaluation/results/eval.json`. It saves after each
mode call, so an interrupted run can be resumed with:

```bash
python3 -m evaluation run --split eval --top-k 6 --resume --allow-provider-export
```

On resume, a completed sibling mode is kept. A hybrid response whose planner
status is degraded (for example `quota`, `local_budget`, `timeout`, or `invalid`)
is retried; earlier attempts, including claim-review labels, remain in that
mode's `attempt_history`. Sanitized failures may include a known LLM error kind
and HTTP status code, but never exception text or response bodies.

Resume is accepted only when the corpus files, question/reference content,
retrieval and Cypher model configuration, prompt source, answer schema,
Cypher grammar, and retrieval configuration hashes match the checkpoint. A
different configuration needs a different `--output` file. The manifest stores
model IDs and hashes of provider endpoints, not API keys or raw endpoints.

The manifest also binds a read-only Neo4j snapshot. It requires the graph's
`Paper.id` set to equal the 26 corpus IDs and hashes paper content and embedding
metadata/vector digests, canonical entity names and aliases, the five retrieval
relationship types, edge direction, and relation provenance. The runner checks
that hash before and after each question pair; a change invalidates the
checkpoint and prevents resuming or mixing results from different graph states.
Other relationship types are ignored by this retrieval snapshot and listed in
the run checkpoint; they do not block the evaluation.
When providers return `effective_model` receipts, the report checks for model
drift between vector and hybrid and across pairs. Abstentions with no generation
receipt do not fail this check; the manifest still records the configured model.
If a provider omits `effective_model` for a generated response, the report can
verify only the configured model hash and cannot detect that provider's hidden
alias/version drift.

The vector response reports planner status `disabled` by design. Hybrid planner
statuses `executed` and `empty` both mean validated Cypher actually ran; they
remain separate in the report because only `executed` returned paper IDs.
Statuses such as `invalid`, `quota`, `timeout`, and `error` are degraded and do
not contribute to the main paired comparison. If any selected hybrid planner
did not validly execute, the report is `inconclusive`.

## Reviewing generated claims and reporting

Each saved claim has two blank reviewer slots under its `claim_reviews` entry.
Reviewers should label whether the complete claim is semantically supported by
the cited corpus evidence, set `reviewer_id`, and set `supported` to `true` or
`false`. Distinct reviewers must label every claim. Literal quote and citation
checks are deterministic, but they do not establish scientific entailment; no
LLM-only judge is used.

Generate the retrieval report offline after completing those labels:

```bash
python3 -m evaluation report --run-file evaluation/results/eval.json
```

An incomplete pair, mismatched hash, pending gold review, missing two-reviewer
claim judgment, or degraded hybrid planner keeps the status `inconclusive` and
suppresses paired deltas. Reports count failures and availability across all
selected questions, while retrieval metrics use only complete pairs whose
hybrid planner validly executed. Source precision/recall are calculated for
questions with reviewed required papers. Multi-hop metrics include required
paper coverage and a check for an actual validated graph path of at least two
hops between required source papers, plus whether a claim cites that path.
Citation validity checks source IDs, exact title/abstract quotes, and graph path
structure. Abstention, latency, planner status, and independently reviewed
claim support are reported separately. Unsupported questions are scored for
abstention and are excluded from source precision/recall.

To report extraction precision, recall, and F1 from the enriched output created
by `src.extract_entities`, with per-label, per-relation, and evidence coverage:

```bash
python3 -m evaluation report --predictions output/papers_enriched.json
```

Extraction metrics remain inconclusive until all 26 annotation records have
two-reviewer approval and the prediction file matches the current corpus.

## Scope and interpretation

The benchmark is grounded in titles, abstracts, authors, and keywords saved in
the repository. It does not establish claims that require the full paper. The
reports are descriptive for this 26-record corpus; do not generalize them to
other datasets or state that hallucinations were reduced unless the complete
paired run and human claim review support that conclusion.
