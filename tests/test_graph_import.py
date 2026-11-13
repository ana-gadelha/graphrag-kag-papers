import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

import run_pipeline
from src.extract_entities import enrich_papers, validate_entities
from src.import_graph import OWNED_TYPES, import_papers, validate_import_input
from tests.test_pipeline import SAMPLE


def fixture():
    return {
        **deepcopy(SAMPLE),
        "extracted": {
            "proposed_methods": ["GraphRAG"],
            "used_methods": [],
            "datasets": [],
        },
    }


class ImportTests(unittest.TestCase):
    def test_extraction_failure_is_sanitized_and_cannot_be_imported(self):
        with patch(
            "src.extract_entities.extract_with_openai",
            side_effect=RuntimeError("secret-provider-detail"),
        ):
            enriched, report = enrich_papers([SAMPLE], "openai", "test-model")
        self.assertFalse(report["valid"])
        self.assertTrue(report["review_required"])
        self.assertNotIn("secret-provider-detail", str(report))
        with self.assertRaises(ValueError):
            validate_import_input(enriched)
        _, report = enrich_papers([SAMPLE], "curated", "", {})
        self.assertTrue(report["review_required"])
        self.assertFalse(report["valid"])

    def test_extraction_rejects_incomplete_provider_response(self):
        from src.extract_entities import extract_with_openai

        for status in ("incomplete", "failed", "queued", None):
            with self.subTest(status=status), patch("openai.OpenAI") as client:
                response = client.return_value.__enter__.return_value.responses.create.return_value
                response.status = status
                response.output_text = (
                    '{"proposed_methods":[],"used_methods":[],"datasets":[]}'
                )
                with self.assertRaises(ValueError):
                    extract_with_openai(SAMPLE, "test-model")

    def test_invalid_input_never_opens_session(self):
        for invalid in (
            [],
            [fixture(), {**fixture(), "year": []}],
            [{**fixture(), "extracted": {"datasets": []}}],
        ):
            driver = MagicMock()
            with self.assertRaises(ValueError):
                import_papers(driver, "neo4j", invalid)
            driver.session.assert_not_called()

    def test_single_transaction_for_each_paper(self):
        driver = MagicMock()
        session = driver.session.return_value.__enter__.return_value
        tx = MagicMock()
        session.execute_write.side_effect = lambda fn, paper: fn(tx, paper)
        import_papers(driver, "neo4j", [fixture()])
        self.assertEqual(session.execute_write.call_count, 1)
        self.assertEqual(tx.run.call_count, 6)
        first = tx.run.call_args_list[0].args[0]
        self.assertIn("DELETE r", first)
        for rel in OWNED_TYPES:
            self.assertIn(rel, first)
        self.assertNotIn("DETACH", first)
        self.assertNotIn("embedding_hash", first)

    def test_curated_missing_lists_and_nonstring_members_fail(self):
        for data in (
            {},
            {"proposed_methods": [True], "used_methods": [], "datasets": []},
        ):
            with self.assertRaises(ValueError):
                validate_entities(data)
            _, report = enrich_papers([SAMPLE], "curated", "", {SAMPLE["id"]: data})
            self.assertEqual(len(report["failures"]), 1)
        self.assertEqual(
            validate_entities(
                {"proposed_methods": [], "used_methods": [], "datasets": []}
            )["datasets"],
            [],
        )

    def test_failed_extraction_reports_without_overwriting_valid_snapshot(self):
        import json
        import tempfile
        from pathlib import Path

        from src.extract_entities import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, curated, output, report = [
                root / name
                for name in (
                    "source.json",
                    "curated.json",
                    "output.json",
                    "report.json",
                )
            ]
            source.write_text(json.dumps([SAMPLE]))
            curated.write_text(json.dumps({SAMPLE["id"]: {}}))
            output.write_text("previous valid snapshot")
            with patch(
                "sys.argv",
                [
                    "extract",
                    "--input",
                    str(source),
                    "--curated",
                    str(curated),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                ],
            ):
                with self.assertRaises(SystemExit):
                    main()
            self.assertEqual(output.read_text(), "previous valid snapshot")
            self.assertTrue(json.loads(report.read_text())["failures"])

    def test_pipeline_stops_at_extraction_failure(self):
        with patch(
            "run_pipeline.run", side_effect=[None, RuntimeError("extraction failed")]
        ) as run:
            with self.assertRaises(RuntimeError):
                run_pipeline.run_pipeline(import_neo4j=True, index_embeddings=True)
        self.assertEqual(run.call_count, 2)

    def test_index_requires_import_and_final_validation(self):
        with self.assertRaises(ValueError):
            run_pipeline.run_pipeline(index_embeddings=True)
        with patch("run_pipeline.run") as run:
            run_pipeline.run_pipeline(import_neo4j=True, index_embeddings=True)
        self.assertEqual(
            [c.args[2] for c in run.call_args_list],
            [
                "src.validate_data",
                "src.extract_entities",
                "src.import_graph",
                "src.validate_graph",
                "retrieval.index_embeddings",
                "src.validate_graph",
            ],
        )
        self.assertEqual(run.call_args.args[-1], "--check-embeddings")


class ValidatorTests(unittest.TestCase):
    def test_unexpected_index_error_is_sanitized_in_report(self):
        from src.validate_graph import validate_graph

        paper = validate_import_input([fixture()])[0]
        with patch(
            "retrieval.embeddings.check_index",
            side_effect=RuntimeError("secret-connection-detail"),
        ):
            report = validate_graph(self.make_driver(paper), "neo4j", [paper])
        self.assertFalse(report["valid"])
        self.assertNotIn("secret-connection-detail", str(report))
        self.assertIn("RuntimeError", str(report))

    def test_cli_close_errors_are_sanitized_and_return_nonzero(self):
        from contextlib import redirect_stdout
        from io import StringIO

        from src import import_graph, validate_graph

        driver = MagicMock()
        driver.close.side_effect = RuntimeError("secret-close-detail")
        with (
            patch("sys.argv", ["import"]),
            patch("src.import_graph.read_json", return_value=[fixture()]),
            patch.dict(
                "os.environ",
                {
                    "NEO4J_URI": "bolt://isolated",
                    "NEO4J_USERNAME": "test",
                    "NEO4J_PASSWORD": "test",
                },
            ),
            patch("src.import_graph.GraphDatabase.driver", return_value=driver),
            patch("src.import_graph.apply_schema"),
            patch("src.import_graph.import_papers"),
        ):
            with self.assertRaises(SystemExit) as failure:
                import_graph.main()
            self.assertIn("encerrar", str(failure.exception))
            self.assertNotIn("secret-close-detail", str(failure.exception))

        with (
            patch("sys.argv", ["validate"]),
            patch("src.validate_graph.read_json", return_value=[fixture()]),
            patch("src.validate_graph.GraphDatabase.driver", return_value=driver),
            patch(
                "src.validate_graph.validate_graph",
                return_value={"valid": True, "errors": []},
            ),
            patch("src.validate_graph.write_json") as write,
            redirect_stdout(StringIO()) as output,
        ):
            with self.assertRaises(SystemExit) as failure:
                validate_graph.main()
            self.assertEqual(failure.exception.code, 1)
            report = write.call_args.args[1]
            self.assertFalse(report["valid"])
            self.assertNotIn("secret-close-detail", output.getvalue())
            self.assertIn("encerrar", output.getvalue())

    def make_driver(
        self,
        paper,
        *,
        props=None,
        edges=None,
        empty=False,
        bad_schema=False,
        constraint_type="UNIQUENESS",
    ):
        from src.import_graph import RELATIONS

        constraints, indexes = [], []
        for label, key in [
            ("Paper", "id"),
            ("Author", "name"),
            ("Concept", "name"),
            ("Method", "name"),
            ("Dataset", "name"),
        ]:
            name = f"{label.lower()}_{key}_unique"
            constraints.append(
                dict(
                    name=name,
                    type=constraint_type,
                    entityType="NODE",
                    labelsOrTypes=[label],
                    properties=[key],
                    ownedIndex=name,
                )
            )
            indexes.append(dict(name=name, state="ONLINE"))
        for name, kind, labels, properties in [
            ("paper_year_range", "RANGE", ["Paper"], ["year"]),
            ("paper_fulltext", "FULLTEXT", ["Paper"], ["title", "abstract"]),
            (
                "entity_name_fulltext",
                "FULLTEXT",
                ["Concept", "Method", "Dataset"],
                ["name"],
            ),
        ]:
            indexes.append(
                dict(
                    name=name,
                    type=kind,
                    state="ONLINE",
                    labelsOrTypes=labels,
                    properties=properties,
                    entityType="NODE",
                )
            )
        if bad_schema:
            indexes[-1]["properties"] = ["wrong"]
        if edges is None:
            edges = [
                dict(type=rel, labels=[label], name=name, outgoing=rel != "AUTHORED")
                for field, label, rel in RELATIONS
                for name in paper.get(field, paper["extracted"].get(field, []))
            ]

        class Record(dict):
            def data(self):
                return dict(self)

        def run(query, **kwargs):
            if query == "SHOW CONSTRAINTS":
                return [Record(row) for row in constraints]
            if query == "SHOW INDEXES":
                return [Record(row) for row in indexes]
            if "properties(p)" in query:
                return (
                    []
                    if empty
                    else [Record(properties=paper if props is None else props)]
                )
            return [Record(row) for row in edges]

        driver = MagicMock()
        driver.session.return_value.__enter__.return_value.run.side_effect = run
        return driver

    def test_schema_snapshot_edges_and_embeddings(self):
        from retrieval.embeddings import EMBEDDING_MODEL, content_hash
        from src.validate_graph import validate_graph

        paper = validate_import_input([fixture()])[0]
        with patch("retrieval.embeddings.check_index"):
            self.assertTrue(
                validate_graph(self.make_driver(paper), "neo4j", [paper])["valid"]
            )
            self.assertTrue(
                validate_graph(
                    self.make_driver(paper, constraint_type="NODE_PROPERTY_UNIQUENESS"),
                    "neo4j",
                    [paper],
                )["valid"]
            )
            self.assertFalse(
                validate_graph(
                    self.make_driver(paper, constraint_type="NODE_PROPERTY_EXISTENCE"),
                    "neo4j",
                    [paper],
                )["valid"]
            )
            for kwargs in (
                {"empty": True},
                {"bad_schema": True},
                {"props": {}},
                {"edges": []},
                {
                    "edges": [
                        dict(
                            type="PROPOSES",
                            labels=["Method"],
                            name="Unexpected",
                            outgoing=True,
                        )
                    ]
                },
            ):
                self.assertFalse(
                    validate_graph(self.make_driver(paper, **kwargs), "neo4j", [paper])[
                        "valid"
                    ],
                    kwargs,
                )
            props = {
                **paper,
                "embedding": [0.1] * 384,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_hash": "stale",
            }
            self.assertFalse(
                validate_graph(
                    self.make_driver(paper, props=props),
                    "neo4j",
                    [paper],
                    check_embeddings=True,
                )["valid"]
            )
            props["embedding_hash"] = content_hash(paper)
            self.assertTrue(
                validate_graph(
                    self.make_driver(paper, props=props),
                    "neo4j",
                    [paper],
                    check_embeddings=True,
                )["valid"]
            )
            paper["keywords"] = []
            self.assertTrue(
                validate_graph(self.make_driver(paper), "neo4j", [paper])["valid"]
            )

    def test_cli_failure_writes_report_and_nonzero(self):
        import json
        import tempfile
        from pathlib import Path

        from src.validate_graph import main

        with tempfile.TemporaryDirectory() as directory:
            source, report = (
                Path(directory) / "input.json",
                Path(directory) / "report.json",
            )
            source.write_text(json.dumps([fixture()]))
            with (
                patch(
                    "sys.argv",
                    ["validate", "--input", str(source), "--report", str(report)],
                ),
                patch(
                    "src.validate_graph.GraphDatabase.driver", return_value=MagicMock()
                ),
                patch(
                    "src.validate_graph.validate_graph",
                    return_value={"valid": False, "errors": ["missing Paper"]},
                ),
            ):
                with self.assertRaises(SystemExit) as failure:
                    main()
            self.assertEqual(failure.exception.code, 1)
            self.assertFalse(json.loads(report.read_text())["valid"])

    def test_import_cli_validates_before_connection_and_sanitizes_failure(self):
        import json
        import tempfile
        from pathlib import Path

        from src.import_graph import main

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            source.write_text("[]")
            with (
                patch("sys.argv", ["import", "--input", str(source)]),
                patch("src.import_graph.GraphDatabase.driver") as connect,
                patch("src.import_graph.apply_schema") as schema,
            ):
                with self.assertRaises(SystemExit):
                    main()
                connect.assert_not_called()
                schema.assert_not_called()
            source.write_text(json.dumps([fixture()]))
            with (
                patch("sys.argv", ["import", "--input", str(source)]),
                patch.dict(
                    "os.environ",
                    {
                        "NEO4J_URI": "bolt://isolated",
                        "NEO4J_USERNAME": "test",
                        "NEO4J_PASSWORD": "test",
                    },
                ),
                patch(
                    "src.import_graph.GraphDatabase.driver",
                    side_effect=RuntimeError("secret-password"),
                ),
            ):
                with self.assertRaises(SystemExit) as failure:
                    main()
                self.assertNotIn("secret-password", str(failure.exception))
