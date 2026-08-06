import copy
import hashlib
import json
import unittest

import yaml

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


    def test_packages_one_immutable_config_map_and_one_indexed_job(self):
        rendered = [
            generate_cross_pair_jobs.render_cross_pair_job(
                self.template,
                spec,
            )
            for spec in self.specs
        ]
        resources, indexed_records = (
            generate_cross_pair_jobs.package_cross_pair_indexed_resources(
                [item[0] for item in rendered],
                [item[1] for item in rendered],
                parallelism=8,
            )
        )
        config_map, indexed_job = resources

        self.assertEqual(
            [resource["kind"] for resource in resources],
            ["ConfigMap", "Job"],
        )
        self.assertTrue(config_map["immutable"])
        self.assertEqual(len(indexed_records), 32)
        self.assertEqual(indexed_job["spec"]["completionMode"], "Indexed")
        self.assertEqual(indexed_job["spec"]["completions"], 32)
        self.assertEqual(indexed_job["spec"]["parallelism"], 8)
        self.assertEqual(indexed_job["spec"]["backoffLimitPerIndex"], 0)
        self.assertEqual(indexed_job["spec"]["maxFailedIndexes"], 32)
        self.assertEqual(
            indexed_job["spec"]["template"]["spec"]["activeDeadlineSeconds"],
            continuation.INDEXED_POD_DEADLINE_SECONDS,
        )

    def test_index_zero_keeps_source_checkpoint_and_target_task(self):
        rendered = [
            generate_cross_pair_jobs.render_cross_pair_job(
                self.template,
                spec,
            )
            for spec in self.specs
        ]
        resources, indexed_records = (
            generate_cross_pair_jobs.package_cross_pair_indexed_resources(
                [item[0] for item in rendered],
                [item[1] for item in rendered],
                parallelism=8,
            )
        )
        task_document = json.loads(
            resources[0]["data"][continuation.INDEXED_TASK_KEY]
        )
        task = task_document["tasks"][0]
        spec = self.specs[0]
        record = indexed_records[0]

        self.assertEqual(task["completion_index"], 0)
        self.assertEqual(task["pair"], spec["target"])
        self.assertEqual(
            task["env"]["AI_SOURCE_CHECKPOINT"],
            spec["source_checkpoint_path"],
        )
        self.assertEqual(
            task["env"]["AI_SOURCE_SHA256"],
            spec["source_checkpoint_sha256"],
        )
        self.assertEqual(
            task["env"]["AI_USER_TASK"],
            spec["target"]["user_task"],
        )
        self.assertNotEqual(
            task["env"]["AI_USER_TASK"],
            spec["source"]["user_task"],
        )
        self.assertEqual(record["target"], spec["target"])
        self.assertEqual(record["source"], spec["source"])

    def test_indexed_resources_use_cross_pair_names_and_labels(self):
        rendered = [
            generate_cross_pair_jobs.render_cross_pair_job(
                self.template,
                spec,
            )
            for spec in self.specs
        ]
        resources, indexed_records = (
            generate_cross_pair_jobs.package_cross_pair_indexed_resources(
                [item[0] for item in rendered],
                [item[1] for item in rendered],
                parallelism=8,
            )
        )
        config_map, indexed_job = resources
        task_map_text = config_map["data"][continuation.INDEXED_TASK_KEY]
        task_map_sha256 = hashlib.sha256(task_map_text.encode()).hexdigest()
        expected_job, expected_config_map = (
            generate_cross_pair_jobs.cross_pair_indexed_resource_names(
                task_map_sha256
            )
        )

        self.assertEqual(indexed_job["metadata"]["name"], expected_job)
        self.assertEqual(config_map["metadata"]["name"], expected_config_map)
        self.assertTrue(expected_config_map.endswith(task_map_sha256[:12]))

        for resource in resources:
            labels = resource["metadata"]["labels"]
            self.assertEqual(
                labels["autoinject-run-kind"],
                "cross-pair-policy-warm-v1",
            )
            self.assertEqual(labels["autoinject-stage"], "x")
            self.assertNotIn(
                "continuation-stage-x",
                labels["autoinject-run-kind"],
            )

        self.assertEqual(
            indexed_job["metadata"]["labels"]["autoinject-suite"],
            "multi",
        )
        self.assertTrue(
            all(
                record["indexed_job_name"] == expected_job
                and record["task_config_map"] == expected_config_map
                and record["task_map_sha256"] == task_map_sha256
                for record in indexed_records
            )
        )

    def test_all_indexed_records_execute_targets_with_named_sources(self):
        rendered = [
            generate_cross_pair_jobs.render_cross_pair_job(
                self.template,
                spec,
            )
            for spec in self.specs
        ]
        resources, indexed_records = (
            generate_cross_pair_jobs.package_cross_pair_indexed_resources(
                [item[0] for item in rendered],
                [item[1] for item in rendered],
                parallelism=8,
            )
        )
        task_document = json.loads(
            resources[0]["data"][continuation.INDEXED_TASK_KEY]
        )

        self.assertEqual(
            [record["completion_index"] for record in indexed_records],
            list(range(32)),
        )
        for spec, record, task in zip(
            self.specs,
            indexed_records,
            task_document["tasks"],
        ):
            self.assertEqual(task["pair"], spec["target"])
            self.assertEqual(record["target"], spec["target"])
            self.assertEqual(record["source"], spec["source"])
            self.assertEqual(
                task["env"]["AI_SOURCE_CHECKPOINT"],
                spec["source_checkpoint_path"],
            )
            self.assertEqual(
                (
                    record["suite"],
                    record["user_task"],
                    record["injection_task"],
                ),
                (
                    spec["target"]["suite"],
                    spec["target"]["user_task"],
                    spec["target"]["injection_task"],
                ),
            )

    def test_cross_pair_generation_is_deterministic(self):
        first = generate_cross_pair_jobs.generate_cross_pair(
            parallelism=8,
        )
        second = generate_cross_pair_jobs.generate_cross_pair(
            parallelism=8,
        )

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[3], second[3])
        generate_cross_pair_jobs.verify_cross_pair_generated_pair(
            first[0],
            first[1],
        )

    def test_inventory_uses_cross_pair_semantics_and_32_records(self):
        manifest, inventory, _, _ = (
            generate_cross_pair_jobs.generate_cross_pair(
                parallelism=8,
            )
        )
        document = json.loads(inventory)

        self.assertEqual(
            document["condition"],
            cross_pair_experiment.CONDITION,
        )
        self.assertEqual(document["record_stage"], "X")
        self.assertEqual(document["source_stage"], "A")
        self.assertEqual(document["query_budget"], 260)
        self.assertEqual(document["completion_mode"], "Indexed")
        self.assertEqual(document["completions"], 32)
        self.assertEqual(document["parallelism"], 8)
        self.assertEqual(len(document["records"]), 32)
        self.assertNotIn("target_stage", document)

        first = document["records"][0]
        self.assertEqual(first["target"], self.specs[0]["target"])
        self.assertEqual(first["source"], self.specs[0]["source"])
        self.assertEqual(
            first["source_checkpoint_path"],
            self.specs[0]["source_checkpoint_path"],
        )
        self.assertTrue(manifest.startswith("---"))

    def test_manifest_resources_are_bound_to_inventory_hash(self):
        manifest, inventory, _, _ = (
            generate_cross_pair_jobs.generate_cross_pair(
                parallelism=8,
            )
        )
        document = json.loads(inventory)
        resources = list(yaml.safe_load_all(manifest))

        self.assertEqual(
            [resource["kind"] for resource in resources],
            ["ConfigMap", "Job"],
        )
        for resource in resources:
            annotations = resource["metadata"]["annotations"]
            self.assertEqual(
                annotations[
                    "autoinject.ucr.edu/job-inventory-payload-sha256"
                ],
                document["job_inventory_payload_sha256"],
            )
            self.assertEqual(
                annotations[
                    continuation.INDEXED_TASK_MAP_HASH_ANNOTATION
                ],
                document["task_map_sha256"],
            )

    def test_verifier_rejects_tampered_manifest(self):
        manifest, inventory, _, _ = (
            generate_cross_pair_jobs.generate_cross_pair(
                parallelism=8,
            )
        )
        tampered = manifest.replace(
            "autoinject-cross-pair-v1",
            "autoinject-cross-pair-bad",
            1,
        )

        with self.assertRaisesRegex(
            generate_cross_pair_jobs.CrossPairJobGenerationError,
            "SHA-256 mismatch",
        ):
            generate_cross_pair_jobs.verify_cross_pair_generated_pair(
                tampered,
                inventory,
            )

    def test_verifier_rejects_tampered_inventory(self):
        manifest, inventory, _, _ = (
            generate_cross_pair_jobs.generate_cross_pair(
                parallelism=8,
            )
        )
        document = json.loads(inventory)
        document["indexed_job_name"] = "autoinject-cross-pair-bad"
        tampered = json.dumps(
            document,
            indent=2,
            sort_keys=True,
        ) + "\n"

        with self.assertRaisesRegex(
            generate_cross_pair_jobs.CrossPairJobGenerationError,
            "payload SHA-256 mismatch",
        ):
            generate_cross_pair_jobs.verify_cross_pair_generated_pair(
                manifest,
                tampered,
            )

if __name__ == "__main__":
    unittest.main()
