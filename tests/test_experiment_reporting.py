import logging
import unittest

from rlpi.agentdojo.experiment_reporting import (
    ExperimentReporting,
    log_final_experiment_summary,
)


class ExperimentReportingTests(unittest.TestCase):
    def test_counter_increments_and_cycle_deltas(self):
        reporting = ExperimentReporting()
        before = reporting.counter_snapshot()

        reporting.record_pipeline_attempt("outer")
        reporting.record_pipeline_completed()
        reporting.record_feedback_model_call("outer")
        reporting.record_pipeline_attempt("grpo")
        reporting.record_pipeline_error()
        reporting.record_feedback_model_call("grpo")
        reporting.record_cycle(3, before)

        self.assertEqual(reporting.victim_queries_outer, 1)
        self.assertEqual(reporting.victim_queries_grpo, 1)
        self.assertEqual(reporting.victim_queries_total, 2)
        self.assertEqual(reporting.feedback_model_calls_total, 2)
        self.assertEqual(reporting.pipeline_runs_attempted, 2)
        self.assertEqual(reporting.pipeline_runs_completed, 1)
        self.assertEqual(reporting.pipeline_run_errors, 1)
        self.assertEqual(reporting.cycles[0]["iteration"], 3)
        self.assertEqual(reporting.cycles[0]["victim_queries_outer"], 1)
        self.assertEqual(reporting.cycles[0]["victim_queries_grpo"], 1)

    def test_checkpoint_without_reporting_is_backward_compatible(self):
        reporting = ExperimentReporting.from_checkpoint(
            {
                "checkpoint_version": "1.0",
                "learner_state": {"queries_used": 7},
            }
        )

        self.assertEqual(
            reporting.to_dict(),
            {
                "victim_queries_outer": 0,
                "victim_queries_grpo": 0,
                "feedback_model_calls_outer": 0,
                "feedback_model_calls_grpo": 0,
                "pipeline_runs_attempted": 0,
                "pipeline_runs_completed": 0,
                "pipeline_run_errors": 0,
                "victim_queries_total": 0,
                "feedback_model_calls_total": 0,
                "cycles": [],
            },
        )

    def test_checkpoint_reporting_round_trip(self):
        original = ExperimentReporting(
            victim_queries_outer=2,
            victim_queries_grpo=4,
            feedback_model_calls_outer=1,
            feedback_model_calls_grpo=3,
            pipeline_runs_attempted=6,
            pipeline_runs_completed=5,
            pipeline_run_errors=1,
            cycles=[{"iteration": 1, "victim_queries_outer": 2}],
        )

        restored = ExperimentReporting.from_checkpoint(
            {"experiment_reporting": original.to_dict()}
        )

        self.assertEqual(restored.to_dict(), original.to_dict())

    def test_final_summary_reports_overshoot_and_mismatch(self):
        reporting = ExperimentReporting(victim_queries_outer=2)
        test_logger = logging.getLogger("reporting-test")

        with self.assertLogs(test_logger, level=logging.INFO) as captured:
            log_final_experiment_summary(
                test_logger,
                reporting,
                query_budget=1,
                queries_used=3,
            )

        output = "\n".join(captured.output)
        self.assertIn("budget_overshoot: 2", output)
        self.assertIn("victim_queries_total: 2", output)
        self.assertIn("differs from queries_used (3)", output)


if __name__ == "__main__":
    unittest.main()
