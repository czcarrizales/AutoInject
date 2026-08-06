import copy
import unittest

from scripts import cross_pair_experiment
from scripts import generate_continuation_jobs as continuation
from scripts import generate_cross_pair_jobs


def plain_environment(job):
    container = job["spec"]["template"]["spec"]["containers"][0]
    return {
        entry["name"]: entry["value"]
        for entry in container["env"]
        if isinstance(entry, dict)
        and set(entry) == {"name", "value"}
        and entry["name"] in continuation.INDEXED_TASK_ENV_NAMES
    }


class CrossPairJobGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = continuation.load_stage_a_inventory(
            continuation.DEFAULT_STAGE_A_INVENTORY
        )
        cls.specs = cross_pair_experiment.build_cross_pair_specs(cls.records)
        cls.template = continuation.load_behavioral_template(
            continuation.DEFAULT_TEMPLATE
        )

    def test_environment_separates_source_artifacts_from_target_task(self):
        spec = self.specs[0]
        environment = generate_cross_pair_jobs.expected_cross_pair_environment(spec)

        self.assertEqual(
            spec["source"],
            {
                "suite": "workspace",
                "user_task": "user_task_39",
                "injection_task": "injection_task_10",
            },
        )
        self.assertEqual(
            spec["target"],
            {
                "suite": "workspace",
                "user_task": "user_task_38",
                "injection_task": "injection_task_10",
            },
        )
        self.assertEqual(
            environment["AI_SOURCE_CHECKPOINT"],
            spec["source_checkpoint_path"],
        )
        self.assertEqual(
            environment["AI_SOURCE_SHA256"],
            spec["source_checkpoint_sha256"],
        )
        self.assertEqual(environment["AI_SUITE"], "workspace")
        self.assertEqual(environment["AI_USER_TASK"], "user_task_38")
        self.assertEqual(environment["AI_INJECTION_TASK"], "injection_task_10")
        self.assertNotEqual(
            environment["AI_USER_TASK"],
            spec["source"]["user_task"],
        )

    def test_rendered_job_uses_foreign_checkpoint_for_target_task(self):
        spec = self.specs[0]
        job, audit = generate_cross_pair_jobs.render_cross_pair_job(
            self.template,
            spec,
        )
        environment = plain_environment(job)

        self.assertEqual(
            environment,
            generate_cross_pair_jobs.expected_cross_pair_environment(spec),
        )
        self.assertEqual(
            environment["AI_SOURCE_CHECKPOINT"],
            spec["source_checkpoint_path"],
        )
        self.assertEqual(
            environment["AI_USER_TASK"],
            spec["target"]["user_task"],
        )
        self.assertEqual(audit["source"], spec["source"])
        self.assertEqual(audit["target"], spec["target"])
        self.assertEqual(audit["suite"], spec["target"]["suite"])
        self.assertEqual(audit["user_task"], spec["target"]["user_task"])
        self.assertEqual(
            audit["injection_task"],
            spec["target"]["injection_task"],
        )
        self.assertEqual(
            audit["configuration_fingerprint"],
            spec["configuration_fingerprint"],
        )

    def test_cross_pair_job_keeps_outputs_outside_continuations(self):
        job, audit = generate_cross_pair_jobs.render_cross_pair_job(
            self.template,
            self.specs[0],
        )
        container = job["spec"]["template"]["spec"]["containers"][0]
        shell = container["args"][0]

        self.assertIn(
            f"RECORDS_ROOT={cross_pair_experiment.RECORDS_ROOT}",
            shell,
        )
        self.assertNotIn(continuation.RECORDS_ROOT, shell)
        self.assertTrue(
            audit["output_checkpoint_path"].startswith(
                cross_pair_experiment.RECORDS_ROOT + "/"
            )
        )
        self.assertNotIn(
            "/continuations/",
            audit["output_checkpoint_path"],
        )

    def test_all_32_specs_render_with_unique_jobs_and_target_tasks(self):
        names = set()

        for spec in self.specs:
            job, audit = generate_cross_pair_jobs.render_cross_pair_job(
                self.template,
                spec,
            )
            environment = plain_environment(job)

            names.add(job["metadata"]["name"])
            self.assertEqual(
                environment["AI_SUITE"],
                spec["target"]["suite"],
            )
            self.assertEqual(
                environment["AI_USER_TASK"],
                spec["target"]["user_task"],
            )
            self.assertEqual(
                environment["AI_INJECTION_TASK"],
                spec["target"]["injection_task"],
            )
            self.assertEqual(
                environment["AI_SOURCE_CHECKPOINT"],
                spec["source_checkpoint_path"],
            )
            self.assertEqual(
                audit["code_commit"],
                continuation.runtime_commit(),
            )

        self.assertEqual(len(names), 32)

    def test_rejects_self_warm_spec(self):
        spec = copy.deepcopy(self.specs[0])
        spec["source"] = copy.deepcopy(spec["target"])

        with self.assertRaisesRegex(
            generate_cross_pair_jobs.CrossPairJobGenerationError,
            "cannot equal",
        ):
            generate_cross_pair_jobs.expected_cross_pair_environment(spec)

    def test_rejects_output_root_outside_cross_pair_tree(self):
        spec = copy.deepcopy(self.specs[0])
        spec["result_root"] = continuation.RECORDS_ROOT + "/unexpected"

        with self.assertRaisesRegex(
            generate_cross_pair_jobs.CrossPairJobGenerationError,
            "escapes",
        ):
            generate_cross_pair_jobs.expected_cross_pair_environment(spec)


if __name__ == "__main__":
    unittest.main()
