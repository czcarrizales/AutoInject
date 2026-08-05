import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlpi.agentdojo.continuation_reporting import (
    PUBLICATION_REQUEST_SCHEMA,
    canonical_json_bytes,
    finalize_stage_record,
    publish_stage_record,
    sha256_file,
    validate_record_component,
    write_stage_publication_request,
)

try:
    from omegaconf import OmegaConf
    from rlpi.agentdojo.adaptive_agentdojo import validate_continuation_config
except ImportError:
    OmegaConf = None
    validate_continuation_config = None


AGGREGATOR_PATH = Path(__file__).parents[1] / "analysis/continuation/aggregate_continuation_results.py"
SPEC = importlib.util.spec_from_file_location("continuation_aggregate", AGGREGATOR_PATH)
aggregate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(aggregate)


class ContinuationReportingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "run"
        outputs = self.run / "grpo_training_outputs"
        outputs.mkdir(parents=True)
        (outputs / "experience_history.jsonl").write_text(
            '{"success_rate": 0, "utility_score": 1, "reward": 0}\n', encoding="utf-8"
        )
        (outputs / "grpo_evaluations.jsonl").write_text(
            '{"success_rate": 1, "utility_score": 0, "reward": 2}\n', encoding="utf-8"
        )
        (self.run / "checkpoint.pt").write_bytes(b"policy")
        (self.run / "checkpoint_state.json").write_text(json.dumps({
            "learner_state": {"queries_used": 5, "early_stopped": False},
            "experiment_reporting": {
                "victim_queries_total": 5, "pipeline_runs_attempted": 5,
                "pipeline_runs_completed": 4, "pipeline_run_errors": 1, "budget_overshoot": 0,
            },
            "continuation": {
                "source_checkpoint_path": "/stage-a/checkpoint.pt",
                "source_checkpoint_sha256": "a" * 64,
                "source_checkpoint_state_path": "/stage-a/checkpoint_state.json",
                "source_checkpoint_state_sha256": "b" * 64,
                "source_checkpoint_size_bytes": 123,
                "source_checkpoint_state_size_bytes": 456,
            },
        }), encoding="utf-8")

    def finalize(self, *, stage="B", run_id="run-b", source_total=5, source_stage="A"):
        return finalize_stage_record(
            records_root=self.root / "continuations/v1", chain_id="travel-u19-i5-chain-1",
            stage=stage, run_id=run_id, suite="travel", user_task="user_task_19",
            injection_task="injection_task_5", initialization_mode="policy_warm",
            reference_mode="source", source_stage=source_stage, source_cumulative_queries_used=source_total,
            git_commit="abc123", resolved_config={
                "query_budget": 5, "nested": {"b": 2, "a": 1},
                "source_checkpoint_path": "/stage-a/checkpoint.pt",
                "source_checkpoint_sha256": "a" * 64,
                "source_checkpoint_state_path": "/stage-a/checkpoint_state.json",
                "source_checkpoint_state_sha256": "b" * 64,
                "source_checkpoint_size_bytes": 123,
                "source_checkpoint_state_size_bytes": 456,
            },
            run_dir=self.run, stage_start_time="2026-08-04T00:00:00Z",
            stage_end_time="2026-08-04T00:00:02Z", stage_wall_clock_seconds=2.0,
            gpu_name="Test GPU", gpu_count=1,
            configuration_fingerprint="c" * 64,
        )

    def test_canonical_hashing_and_atomic_finalization(self):
        self.assertEqual(canonical_json_bytes({"b": 2, "a": 1}), b'{"a":1,"b":2}\n')
        path = self.finalize()
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["cumulative_queries_used"], 10)
        self.assertEqual(record["experience_history_sha256"], sha256_file(self.run / "grpo_training_outputs/experience_history.jsonl"))
        self.assertFalse(any(path.parent.glob(".*.json.*")))

    def test_record_component_validation(self):
        validate_record_component("chain_id", "travel-u19-i5.1")
        for value in ("", "a/b", ".."):
            with self.assertRaises(ValueError):
                validate_record_component("chain_id", value)

    def test_refuses_overwrite_and_missing_or_changed_raw_files(self):
        self.finalize()
        with self.assertRaises(FileExistsError):
            self.finalize()
        (self.run / "grpo_training_outputs/grpo_evaluations.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            self.finalize(run_id="missing")

    def test_shell_publisher_requires_all_postconditions_and_is_no_replace(self):
        stdout = self.run / "stdout.log"
        stdout.write_text("finalized log\n", encoding="utf-8")
        records_root = self.root / "continuations/v1"
        record_path = records_root / "records/stages/travel-u19-i5-chain-1--stage-B--run-b.json"
        request = {
            "schema_version": PUBLICATION_REQUEST_SCHEMA,
            "records_root": str(records_root),
            "chain_id": "travel-u19-i5-chain-1",
            "stage": "B",
            "run_id": "run-b",
            "suite": "travel",
            "user_task": "user_task_19",
            "injection_task": "injection_task_5",
            "initialization_mode": "policy_warm",
            "reference_mode": "source",
            "source_stage": "A",
            "source_cumulative_queries_used": 5,
            "git_commit": "abc123",
            "resolved_config": {
                "query_budget": 5, "source_checkpoint_path": "/stage-a/checkpoint.pt",
                "source_checkpoint_sha256": "a" * 64,
                "source_checkpoint_state_path": "/stage-a/checkpoint_state.json",
                "source_checkpoint_state_sha256": "b" * 64,
                "source_checkpoint_size_bytes": 123,
                "source_checkpoint_state_size_bytes": 456,
            },
            "run_dir": str(self.run),
            "result_root": str(self.run.parent),
            "stage_record_path": str(record_path),
            "configuration_fingerprint": "c" * 64,
            "stage_start_time": "2026-08-04T00:00:00Z",
            "stage_end_time": "2026-08-04T00:00:02Z",
            "stage_wall_clock_seconds": 2.0,
        }
        request_path = self.run / "stage-publication-request.json"
        write_stage_publication_request(request_path, request)
        published = publish_stage_record(request_path, stdout)
        self.assertEqual(published, record_path)
        record = json.loads(published.read_text())
        self.assertEqual(record["stdout_log_sha256"], sha256_file(stdout))
        with self.assertRaises(FileExistsError):
            publish_stage_record(request_path, stdout)

    def test_publisher_does_not_publish_after_postcondition_failure(self):
        stdout = self.run / "stdout.log"
        stdout.write_text("finalized log\n", encoding="utf-8")
        (self.run / "grpo_training_outputs/grpo_evaluations.jsonl").unlink()
        records_root = self.root / "continuations/v1"
        record_path = records_root / "records/stages/travel-u19-i5-chain-1--stage-B--run-b.json"
        request = {
            "schema_version": PUBLICATION_REQUEST_SCHEMA,
            "records_root": str(records_root), "chain_id": "travel-u19-i5-chain-1",
            "stage": "B", "run_id": "run-b", "suite": "travel",
            "user_task": "user_task_19", "injection_task": "injection_task_5",
            "initialization_mode": "policy_warm", "reference_mode": "source",
            "source_stage": "A", "source_cumulative_queries_used": 5,
            "git_commit": "abc123", "resolved_config": {
                "query_budget": 5, "source_checkpoint_path": "/stage-a/checkpoint.pt",
                "source_checkpoint_sha256": "a" * 64,
                "source_checkpoint_state_path": "/stage-a/checkpoint_state.json",
                "source_checkpoint_state_sha256": "b" * 64,
                "source_checkpoint_size_bytes": 123,
                "source_checkpoint_state_size_bytes": 456,
            },
            "run_dir": str(self.run), "result_root": str(self.run.parent),
            "stage_record_path": str(record_path), "configuration_fingerprint": "c" * 64,
            "stage_start_time": "2026-08-04T00:00:00Z",
            "stage_end_time": "2026-08-04T00:00:02Z", "stage_wall_clock_seconds": 2.0,
        }
        request_path = self.run / "stage-publication-request.json"
        write_stage_publication_request(request_path, request)
        with self.assertRaises(FileNotFoundError):
            publish_stage_record(request_path, stdout)
        self.assertFalse(record_path.exists())

    def test_post_rename_directory_fsync_failure_does_not_fail_publication(self):
        stdout = self.run / "stdout.log"
        stdout.write_text("finalized log\n", encoding="utf-8")
        records_root = self.root / "continuations/v1"
        record_path = records_root / "records/stages/travel-u19-i5-chain-1--stage-B--run-b.json"
        request = {
            "schema_version": PUBLICATION_REQUEST_SCHEMA,
            "records_root": str(records_root), "chain_id": "travel-u19-i5-chain-1",
            "stage": "B", "run_id": "run-b", "suite": "travel",
            "user_task": "user_task_19", "injection_task": "injection_task_5",
            "initialization_mode": "policy_warm", "reference_mode": "source",
            "source_stage": "A", "source_cumulative_queries_used": 5,
            "git_commit": "abc123", "resolved_config": {
                "query_budget": 5, "source_checkpoint_path": "/stage-a/checkpoint.pt",
                "source_checkpoint_sha256": "a" * 64,
                "source_checkpoint_state_path": "/stage-a/checkpoint_state.json",
                "source_checkpoint_state_sha256": "b" * 64,
                "source_checkpoint_size_bytes": 123,
                "source_checkpoint_state_size_bytes": 456,
            },
            "run_dir": str(self.run), "result_root": str(self.run.parent),
            "stage_record_path": str(record_path), "configuration_fingerprint": "c" * 64,
            "stage_start_time": "2026-08-04T00:00:00Z",
            "stage_end_time": "2026-08-04T00:00:02Z", "stage_wall_clock_seconds": 2.0,
        }
        request_path = self.run / "stage-publication-request.json"
        write_stage_publication_request(request_path, request)
        real_fsync = __import__("os").fsync
        calls = 0

        def fail_directory_fsync(descriptor):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated directory fsync failure")
            return real_fsync(descriptor)

        with patch("rlpi.agentdojo.continuation_reporting.os.fsync", fail_directory_fsync):
            self.assertEqual(publish_stage_record(request_path, stdout), record_path)
        self.assertTrue(record_path.is_file())

    def test_discovery_hash_validation_ordering_metrics_and_cumulative_chains(self):
        self.finalize(stage="C", run_id="z", source_total=10, source_stage="B")
        second = self.finalize(stage="B", run_id="a", source_total=5)
        records = aggregate.load_records(self.root / "continuations/v1")
        self.assertEqual([record["stage"] for record in records], ["B", "C"])
        stage, evaluations = aggregate.aggregate_stage(records[0])
        self.assertEqual((stage["pair_asr"], stage["pair_utility"]), (1, 0))
        self.assertEqual(stage["cumulative_queries_used"], 10)
        self.assertEqual(records[1]["source_stage"], "B")
        self.assertEqual(records[1]["cumulative_queries_used"], 15)
        self.assertEqual(len(evaluations), 2)
        (self.run / "grpo_training_outputs/experience_history.jsonl").write_text(
            '{"success_rate": 0, "utility_score": 1, "reward": 99}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "incorrect hash"):
            aggregate.load_records(self.root / "continuations/v1")

    def test_duplicate_identity_is_rejected(self):
        record = self.finalize()
        duplicate = record.with_name("other-record.json")
        duplicate.write_bytes(record.read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate stage identity"):
            aggregate.load_records(self.root / "continuations/v1")

    def test_csv_order_is_deterministic(self):
        self.finalize(stage="C", run_id="z", source_total=10)
        self.finalize(stage="B", run_id="a", source_total=5)
        records = aggregate.load_records(self.root / "continuations/v1")
        rows, evaluations = zip(*(aggregate.aggregate_stage(record) for record in records))
        output = self.root / "derived"
        aggregate.write_csv(output / "stages.csv", list(rows), aggregate.STAGE_FIELDS)
        aggregate.write_csv(output / "evaluations.csv", [item for group in evaluations for item in group], aggregate.EVALUATION_FIELDS)
        with (output / "stages.csv").open(newline="", encoding="utf-8") as handle:
            self.assertEqual([row["stage"] for row in csv.DictReader(handle)], ["B", "C"])

    def test_empty_csv_headers_and_null_technical_invalidation(self):
        output = self.root / "derived"
        aggregate.write_csv(output / "stages.csv", [], aggregate.STAGE_FIELDS)
        aggregate.write_csv(output / "evaluations.csv", [], aggregate.EVALUATION_FIELDS)
        self.assertEqual((output / "stages.csv").read_text(encoding="utf-8").strip().split(","), aggregate.STAGE_FIELDS)
        self.assertEqual((output / "evaluations.csv").read_text(encoding="utf-8").strip().split(","), aggregate.EVALUATION_FIELDS)
        record = self.finalize()
        row, _ = aggregate.aggregate_stage(aggregate.load_records(self.root / "continuations/v1")[0])
        aggregate.write_csv(output / "one-stage.csv", [row], aggregate.STAGE_FIELDS)
        with (output / "one-stage.csv").open(newline="", encoding="utf-8") as handle:
            self.assertEqual(next(csv.DictReader(handle))["technical_invalidation"], "")

    def test_supported_quality_classifications(self):
        record = json.loads(self.finalize().read_text(encoding="utf-8"))
        reporting = {"pipeline_run_errors": 0}
        learner_state = {"early_stopped": False}
        self.assertEqual(aggregate.quality_classification(record, False, reporting, learner_state), "cannot_determine")
        record["suite"], record["injection_task"] = "slack", "injection_task_5"
        self.assertEqual(aggregate.quality_classification(record, True, reporting, learner_state), "evaluator_implementation_limitation")
        record["suite"], record["injection_task"] = "travel", "injection_task_5"
        self.assertEqual(aggregate.quality_classification(record, True, reporting, {"early_stopped": True}), "early_stopped")
        self.assertEqual(aggregate.quality_classification(record, True, {"pipeline_run_errors": 1}, learner_state), "evaluator_errors_present")
        self.assertEqual(aggregate.quality_classification(record, True, reporting, learner_state), "standard")

    @unittest.skipUnless(validate_continuation_config is not None, "Hydra/OmegaConf runtime dependencies unavailable")
    def test_disabled_and_cold_recorded_config_validation(self):
        disabled = OmegaConf.create({"initialization_mode": "cold", "source_checkpoint_path": None, "reference_mode": "base", "continuation_records_root": None})
        validate_continuation_config(disabled)
        incomplete = OmegaConf.create({"initialization_mode": "cold", "source_checkpoint_path": None, "reference_mode": "base", "continuation_records_root": "/records", "chain_id": None, "stage": "A", "run_id": "run-a", "git_commit": "abc"})
        with self.assertRaisesRegex(ValueError, "metadata is incomplete"):
            validate_continuation_config(incomplete)


if __name__ == "__main__":
    unittest.main()
