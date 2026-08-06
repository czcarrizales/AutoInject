import copy
import hashlib
import json
import os
import subprocess
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts import generate_continuation_jobs as generator


class ContinuationJobGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.original_records_root = generator.RECORDS_ROOT
        cls.original_runtime_values = (
            generator.INTENDED_RUNTIME_COMMIT,
            generator.CODE_COMMIT,
            generator.RUNTIME_HARDENING_COMMIT,
        )
        generator.RECORDS_ROOT = str(root / "continuations/v1")
        generator.INTENDED_RUNTIME_COMMIT = generator.TEMPLATE_RUNTIME_COMMIT
        generator.CODE_COMMIT = generator.TEMPLATE_RUNTIME_COMMIT
        generator.RUNTIME_HARDENING_COMMIT = generator.TEMPLATE_RUNTIME_COMMIT
        selection = json.loads(
            (generator.REPOSITORY_ROOT / "continuation/stage-a-source-selection.json").read_text()
        )["records"]
        old = json.loads(
            (generator.REPOSITORY_ROOT / "continuation/stage-a-inventory.json").read_text()
        )["records"]
        old_by_pair = {generator.pair_key(record): record for record in old}
        records = []
        for selected in selection:
            prior = old_by_pair[generator.pair_key(selected)]
            checkpoint = Path(selected["checkpoint_path"])
            state_hash = hashlib.sha256((selected["chain_id"] + "/state").encode()).hexdigest()
            records.append(
                {
                    "suite": selected["suite"],
                    "user_task": selected["user_task"],
                    "injection_task": selected["injection_task"],
                    "stage": "A",
                    "run_id": checkpoint.parent.parent.name,
                    "result_root": str(checkpoint.parent.parent),
                    "stage_record_id": f"stage-a-inventory-v3.json#{selected['chain_id']}",
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_state_path": selected["checkpoint_state_path"],
                    "checkpoint_sha256": prior["checkpoint_sha256"],
                    "checkpoint_size_bytes": prior["checkpoint_size_bytes"],
                    "checkpoint_state_sha256": state_hash,
                    "checkpoint_state_size_bytes": 4096,
                    "cumulative_queries_used": selected["expected_queries_used"],
                    "chain_id": selected["chain_id"],
                    "campaign_fingerprint": prior["checkpoint_sha256"],
                }
            )
        cls.inventory_path = root / "stage-a-inventory-v3.json"
        cls.inventory_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "source_stage": "A",
                    "inspection_job_name": "autoinject-stage-a-checkpoint-hash-inspection-v3",
                    "records": records,
                }
            )
        )
        cls.base_records = generator.load_stage_a_inventory(cls.inventory_path)
        cls.template = generator.load_behavioral_template(generator.DEFAULT_TEMPLATE)

    @classmethod
    def tearDownClass(cls):
        generator.RECORDS_ROOT = cls.original_records_root
        (
            generator.INTENDED_RUNTIME_COMMIT,
            generator.CODE_COMMIT,
            generator.RUNTIME_HARDENING_COMMIT,
        ) = cls.original_runtime_values
        cls.directory.cleanup()

    def generate_stage_b(self):
        return generator.generate(
            target_stage="B",
            stage_a_inventory=self.inventory_path,
            template_path=generator.DEFAULT_TEMPLATE,
            predecessor_job_inventory=None,
            parallelism=8,
            enforce_runtime_pin=False,
        )

    def make_stage_b_predecessor(self, directory, base, *, stage_queries=87):
        root = Path(directory)
        generator.RECORDS_ROOT = str(root / "continuations/v1")
        source = generator.source_from_stage_a(base)
        identifiers = generator.target_identifiers(source, "B")
        result_root = Path(identifiers["result_root"])
        run = result_root / "run"
        outputs = run / "grpo_training_outputs"
        outputs.mkdir(parents=True)
        checkpoint = run / "checkpoint.pt"
        checkpoint.write_bytes(b"synthetic-stage-b-checkpoint")
        state = run / "checkpoint_state.json"
        state_document = {
            "learner_state": {"queries_used": stage_queries},
            "experiment_reporting": {"pipeline_run_errors": 0},
            "continuation": {
                "source_checkpoint_path": base["checkpoint_path"],
                "source_checkpoint_sha256": base["checkpoint_sha256"],
                "source_checkpoint_state_path": base["checkpoint_state_path"],
                "source_checkpoint_state_sha256": base["checkpoint_state_sha256"],
                "source_checkpoint_size_bytes": base["checkpoint_size_bytes"],
                "source_checkpoint_state_size_bytes": base["checkpoint_state_size_bytes"],
            },
        }
        state.write_text(json.dumps(state_document), encoding="utf-8")
        run_id = identifiers["run_id"]
        record_path = Path(identifiers["stage_record_path"])
        record_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = generator.configuration_fingerprint(source, "B", identifiers)
        nomination = {
            "suite": base["suite"],
            "user_task": base["user_task"],
            "injection_task": base["injection_task"],
            "chain_id": base["chain_id"],
            "stage": "B",
            "run_id": run_id,
            "job_name": identifiers["job_name"],
            "experiment_id": identifiers["experiment_id"],
            "result_root": str(result_root),
            "stage_record_path": str(record_path),
            "output_checkpoint_path": str(checkpoint),
            "output_checkpoint_state_path": str(state),
            "output_checkpoint_sha256": None,
            "output_checkpoint_state_sha256": None,
            "configuration_fingerprint": fingerprint,
            "source_stage": "A",
            "source_descriptor": source,
        }
        record = {
            "schema_version": generator.STAGE_RECORD_SCHEMA,
            "chain_id": base["chain_id"],
            "stage": "B",
            "run_id": run_id,
            "suite": base["suite"],
            "user_task": base["user_task"],
            "injection_task": base["injection_task"],
            "initialization_mode": "policy_warm",
            "reference_mode": "source",
            "source_stage": "A",
            "source_checkpoint_path": base["checkpoint_path"],
            "source_checkpoint_sha256": base["checkpoint_sha256"],
            "source_checkpoint_size_bytes": base["checkpoint_size_bytes"],
            "source_checkpoint_state_path": base["checkpoint_state_path"],
            "source_checkpoint_state_sha256": base["checkpoint_state_sha256"],
            "source_checkpoint_state_size_bytes": base["checkpoint_state_size_bytes"],
            "result_root": str(result_root),
            "output_checkpoint_path": str(checkpoint),
            "output_checkpoint_sha256": generator.sha256_file(checkpoint),
            "output_checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_state_path": str(state),
            "checkpoint_state_sha256": generator.sha256_file(state),
            "checkpoint_state_size_bytes": state.stat().st_size,
            "stage_query_budget": 260,
            "stage_queries_used": stage_queries,
            "cumulative_queries_used": base["cumulative_queries_used"] + stage_queries,
            "git_commit": generator.runtime_commit(),
            "resolved_config_sha256": hashlib.sha256(b"resolved").hexdigest(),
            "configuration_fingerprint": fingerprint,
            "completion_status": "completed",
            "technical_invalidation": None,
            "technical_invalidation_reason": None,
            "experiment_reporting": {"pipeline_run_errors": 0},
        }
        record_path.write_text(json.dumps(record), encoding="utf-8")
        inventory = root / "stage-b-job-inventory.json"
        document = {"schema_version": 2, "target_stage": "B", "records": [nomination]}
        document["job_inventory_payload_sha256"] = hashlib.sha256(
            (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        manifest = yaml.safe_dump_all(
            [{"apiVersion": "batch/v1", "kind": "Job", "metadata": {"annotations": {
                "autoinject.ucr.edu/job-inventory-payload-sha256": document["job_inventory_payload_sha256"]
            }}}],
            explicit_start=True,
        )
        document["generated_yaml_sha256"] = hashlib.sha256(manifest.encode()).hexdigest()
        inventory.write_text(json.dumps(document))
        (root / "stage-b-jobs.yaml").write_text(manifest)
        return inventory, nomination, record, record_path, state

    def resolve_one(self, inventory, base):
        return generator.resolve_predecessor_sources("C", [base], inventory)

    def test_stage_ordering_is_generic(self):
        self.assertEqual(generator.stage_index("D"), 3)
        self.assertEqual(generator.predecessor_stage("B"), "A")
        self.assertEqual(generator.predecessor_stage("C"), "B")
        with self.assertRaises(generator.GenerationError):
            generator.predecessor_stage("A")

    def test_launchable_generation_requires_committed_hardened_runtime_pin(self):
        original = (
            generator.INTENDED_RUNTIME_COMMIT,
            generator.CODE_COMMIT,
            generator.RUNTIME_HARDENING_COMMIT,
        )
        try:
            generator.INTENDED_RUNTIME_COMMIT = None
            generator.CODE_COMMIT = None
            generator.RUNTIME_HARDENING_COMMIT = None
            with self.assertRaisesRegex(generator.GenerationError, "not yet committed and pinned"):
                generator.generate(
                    target_stage="B",
                    stage_a_inventory=self.inventory_path,
                    template_path=generator.DEFAULT_TEMPLATE,
                    predecessor_job_inventory=None,
                )
        finally:
            (
                generator.INTENDED_RUNTIME_COMMIT,
                generator.CODE_COMMIT,
                generator.RUNTIME_HARDENING_COMMIT,
            ) = original

    def test_stage_b_integrity_budget_and_three_early_stops(self):
        manifest, inventory_text, resources, records = self.generate_stage_b()
        documents = list(yaml.safe_load_all(manifest))
        self.assertEqual(len(resources), 2)
        self.assertEqual([document["kind"] for document in documents], ["ConfigMap", "Job"])
        generator.verify_generated_pair(manifest, inventory_text)

        inventory = json.loads(inventory_text)
        self.assertEqual(inventory["code_commit"], generator.runtime_commit())
        self.assertEqual(inventory["completion_mode"], "Indexed")
        self.assertEqual(inventory["completions"], 32)
        self.assertEqual(inventory["parallelism"], 8)
        self.assertRegex(inventory["task_map_sha256"], r"[0-9a-f]{64}")

        config_map, indexed_job = documents
        self.assertIs(config_map["immutable"], True)
        task_map_text = config_map["data"][generator.INDEXED_TASK_KEY]
        self.assertEqual(
            hashlib.sha256(task_map_text.encode()).hexdigest(),
            inventory["task_map_sha256"],
        )
        self.assertTrue(
            config_map["metadata"]["name"].endswith(
                inventory["task_map_sha256"][:12]
            )
        )
        for resource in (config_map, indexed_job):
            self.assertEqual(
                resource["metadata"]["annotations"][
                    generator.INDEXED_TASK_MAP_HASH_ANNOTATION
                ],
                inventory["task_map_sha256"],
            )
        task_document = json.loads(task_map_text)
        self.assertEqual(task_document["task_count"], 32)
        self.assertEqual(
            [task["completion_index"] for task in task_document["tasks"]],
            list(range(32)),
        )

        spec = indexed_job["spec"]
        self.assertEqual(spec["completionMode"], "Indexed")
        self.assertEqual(spec["completions"], 32)
        self.assertEqual(spec["parallelism"], 8)
        self.assertEqual(spec["backoffLimitPerIndex"], 0)
        self.assertEqual(spec["maxFailedIndexes"], 32)
        self.assertNotIn("activeDeadlineSeconds", spec)

        pod_spec = spec["template"]["spec"]
        self.assertEqual(pod_spec["activeDeadlineSeconds"], 252000)
        container = pod_spec["containers"][0]
        self.assertEqual(
            container["resources"]["requests"],
            {"cpu": "2", "memory": "24Gi", "nvidia.com/gpu": 1},
        )
        self.assertEqual(
            container["resources"]["limits"],
            {"cpu": "2", "memory": "24Gi", "nvidia.com/gpu": 1},
        )
        affinity_values = (
            pod_spec["affinity"]["nodeAffinity"]
            ["requiredDuringSchedulingIgnoredDuringExecution"]
            ["nodeSelectorTerms"][0]["matchExpressions"][0]["values"]
        )
        self.assertEqual(affinity_values, ["NVIDIA-L40", "NVIDIA-L40S"])

        shell = container["args"][0]
        self.assertIn('pipeline_status=("${PIPESTATUS[@]}")', shell)
        self.assertIn(
            'exec "${PYTHON}" -m rlpi.agentdojo.continuation_reporting', shell
        )
        self.assertNotIn("cp /work/stage-b-stdout.log", shell)
        self.assertIn("JOB_COMPLETION_INDEX", shell)
        self.assertIn(generator.INDEXED_TASK_FILE, shell)
        self.assertIn(inventory["task_map_sha256"], shell)

        early = {
            generator.pair_key(record)
            for record in records
            if record["source_cumulative_queries_used"] == 65
        }
        self.assertEqual(
            early,
            {
                ("banking", "user_task_14", "injection_task_7"),
                ("slack", "user_task_20", "injection_task_2"),
                ("slack", "user_task_20", "injection_task_3"),
            },
        )
        self.assertEqual(
            [record["completion_index"] for record in records], list(range(32))
        )
        self.assertEqual(
            {generator.pair_key(record) for record in records},
            generator.expected_pairs(),
        )
        for task, record in zip(task_document["tasks"], records):
            self.assertEqual(record["query_budget"], 260)
            self.assertIn("source_checkpoint_state_sha256", record)
            self.assertEqual(record["code_commit"], generator.runtime_commit())
            self.assertEqual(
                task["env"], generator.expected_dynamic_environment(record)
            )

    def test_parallelism_bounds_and_required_cli_argument(self):
        for value in (0, 33):
            with self.subTest(parallelism=value), self.assertRaisesRegex(
                generator.GenerationError, "parallelism"
            ):
                generator.generate(
                    target_stage="B",
                    stage_a_inventory=self.inventory_path,
                    template_path=generator.DEFAULT_TEMPLATE,
                    predecessor_job_inventory=None,
                    parallelism=value,
                    enforce_runtime_pin=False,
                )
        with self.assertRaises(SystemExit):
            generator.parse_args(["--stage", "B"])

    def test_unexpected_common_template_drift_fails_closed(self):
        rendered = [
            generator.render_job(
                self.template, generator.source_from_stage_a(record), "B"
            )
            for record in self.base_records
        ]
        jobs = [item[0] for item in rendered]
        records = [item[1] for item in rendered]
        jobs[1]["spec"]["template"]["spec"]["containers"][0]["resources"][
            "requests"
        ]["memory"] = "16Gi"
        with self.assertRaisesRegex(
            generator.GenerationError, "Unexpected non-task-specific difference"
        ):
            generator.package_indexed_resources(jobs, records, "B", 8)

    def test_indexed_task_bootstrap_fails_closed(self):
        manifest, inventory_text, _, _ = self.generate_stage_b()
        config_map = list(yaml.safe_load_all(manifest))[0]
        inventory = json.loads(inventory_text)
        with tempfile.TemporaryDirectory() as directory:
            task_map = Path(directory) / "tasks.json"
            task_map.write_text(
                config_map["data"][generator.INDEXED_TASK_KEY], encoding="utf-8"
            )
            bootstrap = generator.indexed_task_bootstrap(
                inventory["task_map_sha256"]
            ).replace(generator.INDEXED_TASK_FILE, str(task_map))
            script = (
                "set -Eeuo pipefail\n"
                f"PYTHON={shlex.quote(sys.executable)}\n"
                f"{bootstrap}"
                'printf "OK:%s:%s\\n" "${AI_SUITE}" "${AI_USER_TASK}"\n'
            )

            valid_environment = os.environ.copy()
            valid_environment["JOB_COMPLETION_INDEX"] = "0"
            valid = subprocess.run(
                ["bash", "-c", script],
                check=False,
                capture_output=True,
                text=True,
                env=valid_environment,
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertTrue(valid.stdout.startswith("OK:"))

            for value in (None, "abc", "-1", "32"):
                with self.subTest(index=value):
                    environment = os.environ.copy()
                    if value is not None:
                        environment["JOB_COMPLETION_INDEX"] = value
                    else:
                        environment.pop("JOB_COMPLETION_INDEX", None)
                    failed = subprocess.run(
                        ["bash", "-c", script],
                        check=False,
                        capture_output=True,
                        text=True,
                        env=environment,
                    )
                    self.assertNotEqual(failed.returncode, 0)

            task_map.write_text(
                task_map.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            tampered = subprocess.run(
                ["bash", "-c", script],
                check=False,
                capture_output=True,
                text=True,
                env=valid_environment,
            )
            self.assertNotEqual(tampered.returncode, 0)

    def test_behavioral_template_invariants_fail_closed(self):
        mutations = (
            (lambda value: value["spec"].__setitem__("backoffLimit", 1), "backoffLimit"),
            (
                lambda value: value["spec"]["template"]["spec"]["containers"][0][
                    "resources"
                ]["requests"].__setitem__("memory", "16Gi"),
                "resource requests",
            ),
            (
                lambda value: value["spec"]["template"]["spec"]["affinity"][
                    "nodeAffinity"
                ]["requiredDuringSchedulingIgnoredDuringExecution"][
                    "nodeSelectorTerms"
                ][0]["matchExpressions"][0]["values"].__setitem__(0, "NVIDIA-A100"),
                "affinity",
            ),
            (
                lambda value: value["spec"]["template"]["spec"].__setitem__(
                    "restartPolicy", "Always"
                ),
                "restartPolicy",
            ),
            (
                lambda value: value["spec"]["template"]["spec"].__setitem__(
                    "automountServiceAccountToken", True
                ),
                "automountServiceAccountToken",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                template = copy.deepcopy(self.template)
                mutate(template)
                with self.assertRaisesRegex(generator.GenerationError, message):
                    generator.validate_behavioral_template_invariants(template)

    def test_completion_index_order_is_canonical_across_inventory_order(self):
        baseline = self.generate_stage_b()[3]
        with tempfile.TemporaryDirectory() as directory:
            shuffled_path = Path(directory) / "stage-a-inventory-v3.json"
            document = json.loads(self.inventory_path.read_text(encoding="utf-8"))
            document["records"] = list(reversed(document["records"]))
            shuffled_path.write_text(json.dumps(document), encoding="utf-8")
            shuffled = generator.generate(
                target_stage="B",
                stage_a_inventory=shuffled_path,
                template_path=generator.DEFAULT_TEMPLATE,
                predecessor_job_inventory=None,
                parallelism=8,
                enforce_runtime_pin=False,
            )[3]
        self.assertEqual(
            [generator.pair_key(record) for record in shuffled],
            [generator.pair_key(record) for record in baseline],
        )
        self.assertEqual(
            [record["completion_index"] for record in shuffled], list(range(32))
        )

    def test_task_map_hash_and_launch_wiring_are_verified(self):
        manifest, inventory_text, _, _ = self.generate_stage_b()
        resources = list(yaml.safe_load_all(manifest))
        inventory = json.loads(inventory_text)

        tampered_map = copy.deepcopy(resources)
        tampered_map[0]["data"][generator.INDEXED_TASK_KEY] += " "
        tampered_manifest = generator.render_yaml(tampered_map)
        tampered_inventory = copy.deepcopy(inventory)
        tampered_inventory["generated_yaml_sha256"] = hashlib.sha256(
            tampered_manifest.encode()
        ).hexdigest()
        with self.assertRaisesRegex(generator.GenerationError, "ConfigMap hash"):
            generator.verify_generated_pair(
                tampered_manifest, json.dumps(tampered_inventory)
            )

        drifted_job = copy.deepcopy(resources)
        drifted_job[1]["spec"]["template"]["spec"]["containers"][0][
            "resources"
        ]["limits"]["memory"] = "16Gi"
        drifted_manifest = generator.render_yaml(drifted_job)
        drifted_inventory = copy.deepcopy(inventory)
        drifted_inventory["generated_yaml_sha256"] = hashlib.sha256(
            drifted_manifest.encode()
        ).hexdigest()
        with self.assertRaisesRegex(generator.GenerationError, "resources"):
            generator.verify_generated_pair(
                drifted_manifest, json.dumps(drifted_inventory)
            )

    def test_rendered_runtime_branch_is_single_binding(self):
        job, record = generator.render_job(
            self.template, generator.source_from_stage_a(self.base_records[0]), "B"
        )
        init_shell = job["spec"]["template"]["spec"]["initContainers"][0]["args"][0]
        self.assertIn(
            "BRANCH=experiment/stage-b-policy-warm-start", init_shell
        )
        self.assertNotIn(
            "BRANCH=experiment/travel-u19-i5-stage-b-validation", init_shell
        )

        stale = copy.deepcopy(job)
        stale_shell = stale["spec"]["template"]["spec"]["initContainers"][0]["args"][0]
        stale["spec"]["template"]["spec"]["initContainers"][0]["args"][0] = (
            stale_shell.replace(
                f"BRANCH={generator.RUNTIME_BRANCH}",
                f"BRANCH={generator.TEMPLATE_RUNTIME_BRANCH}",
            )
        )
        with self.assertRaisesRegex(generator.GenerationError, "runtime branch"):
            generator.verify_rendered_runtime_commit(
                stale, record, generator.runtime_commit()
            )

    def test_rendered_runtime_commit_is_single_binding(self):
        original = (
            generator.INTENDED_RUNTIME_COMMIT,
            generator.CODE_COMMIT,
            generator.RUNTIME_HARDENING_COMMIT,
        )
        commit = "c" * 40
        try:
            generator.INTENDED_RUNTIME_COMMIT = commit
            generator.CODE_COMMIT = commit
            generator.RUNTIME_HARDENING_COMMIT = commit
            job, record = generator.render_job(
                self.template, generator.source_from_stage_a(self.base_records[0]), "B"
            )
            serialized = json.dumps(job)
            self.assertIn(f"COMMIT={commit}", serialized)
            self.assertIn(f"CODE_COMMIT={commit}", serialized)
            self.assertNotIn(generator.TEMPLATE_RUNTIME_COMMIT, serialized)
            self.assertEqual(record["code_commit"], commit)
            self.assertEqual(
                job["metadata"]["annotations"]["autoinject.ucr.edu/code-commit"],
                commit,
            )
            missing_annotation = copy.deepcopy(job)
            del missing_annotation["metadata"]["annotations"]["autoinject.ucr.edu/code-commit"]
            with self.assertRaisesRegex(generator.GenerationError, "metadata has the wrong"):
                generator.verify_rendered_runtime_commit(missing_annotation, record, commit)
            stale_annotation = copy.deepcopy(job)
            stale_annotation["metadata"]["annotations"][
                "autoinject.ucr.edu/code-commit"
            ] = generator.TEMPLATE_RUNTIME_COMMIT
            with self.assertRaisesRegex(generator.GenerationError, "metadata has the wrong"):
                generator.verify_rendered_runtime_commit(stale_annotation, record, commit)
        finally:
            (
                generator.INTENDED_RUNTIME_COMMIT,
                generator.CODE_COMMIT,
                generator.RUNTIME_HARDENING_COMMIT,
            ) = original

    def test_identifiers_are_unique_and_generation_is_deterministic(self):
        first = self.generate_stage_b()
        second = self.generate_stage_b()
        self.assertEqual(first[:2], second[:2])
        for field in ("job_name", "run_id", "experiment_id", "result_root", "stage_record_path", "chain_id"):
            values = [record[field] for record in first[3]]
            self.assertEqual(len(values), len(set(values)), field)

    def test_exact_v1_nomination_ignores_r01_and_r01_cannot_substitute(self):
        base = self.base_records[0]
        with tempfile.TemporaryDirectory() as directory:
            inventory, nomination, _, record_path, _ = self.make_stage_b_predecessor(directory, base)
            self.assertEqual(self.resolve_one(inventory, base)[0]["run_id"], nomination["run_id"])
            r01 = record_path.with_name(
                f"{base['chain_id']}--stage-B--{base['chain_id']}-stage-b-r01.json"
            )
            r01.write_bytes(record_path.read_bytes())
            nomination["run_id"] = f"{base['chain_id']}-stage-b-r01"
            nomination["job_name"] = f"autoinject-{generator.pair_slug(base)}-stage-b-r01"
            nomination["experiment_id"] = "validation-r01"
            document = {"schema_version": 2, "target_stage": "B", "records": [nomination]}
            document["job_inventory_payload_sha256"] = hashlib.sha256(
                (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest()
            manifest = yaml.safe_dump_all(
                [{"apiVersion": "batch/v1", "kind": "Job", "metadata": {"annotations": {
                    "autoinject.ucr.edu/job-inventory-payload-sha256": document["job_inventory_payload_sha256"]
                }}}], explicit_start=True,
            )
            document["generated_yaml_sha256"] = hashlib.sha256(manifest.encode()).hexdigest()
            inventory.write_text(json.dumps(document))
            (Path(directory) / "stage-b-jobs.yaml").write_text(manifest)
            with self.assertRaisesRegex(generator.GenerationError, "canonical production nomination"):
                self.resolve_one(inventory, base)
            record_path.unlink()

    def test_mismatched_nomination_stage_cannot_generate_stage_c_outputs(self):
        base = self.base_records[0]
        with tempfile.TemporaryDirectory() as directory:
            inventory, nomination, _, _, _ = self.make_stage_b_predecessor(directory, base)
            stage_b_source = self.resolve_one(inventory, base)[0]
            identifiers = generator.target_identifiers(stage_b_source, "C")
            nomination.update(
                {
                    "stage": "C",
                    "source_stage": "B",
                    "source_descriptor": stage_b_source,
                    "job_name": identifiers["job_name"],
                    "run_id": identifiers["run_id"],
                    "experiment_id": identifiers["experiment_id"],
                    "result_root": identifiers["result_root"],
                    "stage_record_path": identifiers["stage_record_path"],
                    "output_checkpoint_path": f"{identifiers['result_root']}/run/checkpoint.pt",
                    "output_checkpoint_state_path": f"{identifiers['result_root']}/run/checkpoint_state.json",
                    "configuration_fingerprint": generator.configuration_fingerprint(
                        stage_b_source, "C", identifiers
                    ),
                }
            )
            records = [nomination]
            records.extend(
                {**nomination, "chain_id": record["chain_id"]}
                for record in self.base_records[1:]
            )
            document = {"schema_version": 2, "target_stage": "B", "records": records}
            document["job_inventory_payload_sha256"] = hashlib.sha256(
                (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest()
            manifest = yaml.safe_dump_all(
                [{"apiVersion": "batch/v1", "kind": "Job", "metadata": {"annotations": {
                    "autoinject.ucr.edu/job-inventory-payload-sha256": document["job_inventory_payload_sha256"]
                }}}], explicit_start=True,
            )
            document["generated_yaml_sha256"] = hashlib.sha256(manifest.encode()).hexdigest()
            inventory.write_text(json.dumps(document))
            (Path(directory) / "stage-b-jobs.yaml").write_text(manifest)
            output = Path(directory) / "stage-c-jobs.yaml"
            index = Path(directory) / "stage-c-job-inventory.json"
            self.assertEqual(
                generator.main(
                    [
                        "--stage", "C",
                        "--parallelism", "8",
                        "--stage-a-inventory", str(self.inventory_path),
                        "--predecessor-job-inventory", str(inventory),
                        "--output", str(output),
                        "--job-inventory-output", str(index),
                    ]
                ),
                1,
            )
            self.assertFalse(output.exists())
            self.assertFalse(index.exists())

    def test_recursive_lineage_and_cumulative_fail_closed(self):
        base = self.base_records[0]
        mutations = [
            ("source_checkpoint_path", self.base_records[1]["checkpoint_path"], "nominated prior"),
            ("source_checkpoint_state_path", self.base_records[1]["checkpoint_state_path"], "nominated prior"),
            ("source_stage", "Z", "source stage"),
            ("cumulative_queries_used", 999, "cumulative"),
            ("run_id", "wrong-run", "run_id"),
            ("result_root", "/wrong/root", "result_root"),
            ("configuration_fingerprint", "0" * 64, "configuration_fingerprint"),
            ("completion_status", "running", "not successfully completed"),
            ("technical_invalidation", True, "invalidated"),
        ]
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                inventory, _, record, record_path, _ = self.make_stage_b_predecessor(directory, base)
                record[field] = value
                record_path.write_text(json.dumps(record))
                with self.assertRaisesRegex(generator.GenerationError, message):
                    self.resolve_one(inventory, base)

    def test_nomination_path_and_artifact_confinement_fail_closed(self):
        base = self.base_records[0]
        with tempfile.TemporaryDirectory() as directory:
            inventory, nomination, record, _, _ = self.make_stage_b_predecessor(directory, base)
            outside_record = Path(directory) / "wrong.json"
            outside_record.write_text(json.dumps(record))
            nomination["stage_record_path"] = str(outside_record)
            json.loads(inventory.read_text())["records"]
            document = {"schema_version": 2, "target_stage": "B", "records": [nomination]}
            document["job_inventory_payload_sha256"] = hashlib.sha256(
                (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest()
            manifest = yaml.safe_dump_all(
                [{"apiVersion": "batch/v1", "kind": "Job", "metadata": {"annotations": {
                    "autoinject.ucr.edu/job-inventory-payload-sha256": document["job_inventory_payload_sha256"]
                }}}], explicit_start=True,
            )
            document["generated_yaml_sha256"] = hashlib.sha256(manifest.encode()).hexdigest()
            inventory.write_text(json.dumps(document))
            (Path(directory) / "stage-b-jobs.yaml").write_text(manifest)
            with self.assertRaisesRegex(generator.GenerationError, "canonical production nomination"):
                self.resolve_one(inventory, base)
        with tempfile.TemporaryDirectory() as directory:
            inventory, nomination, record, record_path, _ = self.make_stage_b_predecessor(directory, base)
            outside = Path(directory) / "outside.pt"
            outside.write_bytes(b"synthetic-stage-b-checkpoint")
            record["output_checkpoint_path"] = str(outside)
            record["output_checkpoint_sha256"] = generator.sha256_file(outside)
            record_path.write_text(json.dumps(record))
            with self.assertRaisesRegex(generator.GenerationError, "invalid|exact nomination"):
                self.resolve_one(inventory, base)
        with tempfile.TemporaryDirectory() as directory:
            inventory, _, record, record_path, state = self.make_stage_b_predecessor(directory, base)
            outside_dir = Path(directory) / "outside"
            outside_dir.mkdir()
            outside = outside_dir / "checkpoint_state.json"
            outside.write_bytes(state.read_bytes())
            record["checkpoint_state_path"] = str(outside)
            record["checkpoint_state_sha256"] = generator.sha256_file(outside)
            record["checkpoint_state_size_bytes"] = outside.stat().st_size
            record_path.write_text(json.dumps(record))
            with self.assertRaisesRegex(generator.GenerationError, "exact nomination"):
                self.resolve_one(inventory, base)

    def test_internally_consistent_record_with_wrong_prior_source_is_rejected(self):
        base = self.base_records[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, nomination, record, record_path, state = self.make_stage_b_predecessor(directory, base)
            wrong_path = "/workspace/results/wrong-task/run/checkpoint.pt"
            wrong_hash = "d" * 64
            nomination["source_descriptor"]["checkpoint_path"] = wrong_path
            nomination["source_descriptor"]["checkpoint_sha256"] = wrong_hash
            record["source_checkpoint_path"] = wrong_path
            record["source_checkpoint_sha256"] = wrong_hash
            state_value = json.loads(state.read_text())
            state_value["continuation"]["source_checkpoint_path"] = wrong_path
            state_value["continuation"]["source_checkpoint_sha256"] = wrong_hash
            state.write_text(json.dumps(state_value))
            record["checkpoint_state_sha256"] = generator.sha256_file(state)
            record["checkpoint_state_size_bytes"] = state.stat().st_size
            record_path.write_text(json.dumps(record))
            document = {"schema_version": 2, "target_stage": "B", "records": [nomination]}
            document["job_inventory_payload_sha256"] = hashlib.sha256(
                (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest()
            manifest = yaml.safe_dump_all(
                [{"apiVersion": "batch/v1", "kind": "Job", "metadata": {"annotations": {
                    "autoinject.ucr.edu/job-inventory-payload-sha256": document["job_inventory_payload_sha256"]
                }}}], explicit_start=True,
            )
            document["generated_yaml_sha256"] = hashlib.sha256(manifest.encode()).hexdigest()
            inventory.write_text(json.dumps(document))
            (root / "stage-b-jobs.yaml").write_text(manifest)
            with self.assertRaisesRegex(generator.GenerationError, "Nominated Stage A lineage"):
                self.resolve_one(inventory, base)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_symlink_escape_is_rejected(self):
        base = self.base_records[0]
        with tempfile.TemporaryDirectory() as directory:
            inventory, _, record, record_path, state = self.make_stage_b_predecessor(directory, base)
            outside = Path(directory) / "outside-state.json"
            outside.write_bytes(state.read_bytes())
            state.unlink()
            state.symlink_to(outside)
            record["checkpoint_state_sha256"] = generator.sha256_file(state)
            record_path.write_text(json.dumps(record))
            with self.assertRaisesRegex(generator.GenerationError, "regular nonempty"):
                self.resolve_one(inventory, base)

    def test_changed_state_bytes_with_same_query_count_is_rejected(self):
        base = self.base_records[0]
        with tempfile.TemporaryDirectory() as directory:
            inventory, _, record, record_path, state = self.make_stage_b_predecessor(directory, base)
            value = json.loads(state.read_text())
            value["unrecorded_change"] = True
            state.write_text(json.dumps(value))
            record["checkpoint_state_size_bytes"] = state.stat().st_size
            record_path.write_text(json.dumps(record))
            with self.assertRaisesRegex(generator.GenerationError, "checkpoint-state hash mismatch"):
                self.resolve_one(inventory, base)

    def test_stage_c_uses_actual_cumulative_and_not_stage_arithmetic(self):
        base = self.base_records[0]
        with tempfile.TemporaryDirectory() as directory:
            inventory, _, _, _, _ = self.make_stage_b_predecessor(directory, base, stage_queries=87)
            source = self.resolve_one(inventory, base)[0]
            self.assertEqual(source["cumulative_queries_used"], base["cumulative_queries_used"] + 87)
            self.assertNotEqual(source["cumulative_queries_used"], 520)

    def test_shell_metacharacters_remain_environment_data(self):
        source = generator.source_from_stage_a(copy.deepcopy(self.base_records[0]))
        source["checkpoint_path"] = "/workspace/results/a $(touch /tmp/pwned);'\"`x`/checkpoint.pt"
        job, _ = generator.render_job(self.template, source, "B")
        container = job["spec"]["template"]["spec"]["containers"][0]
        shell = container["args"][0]
        self.assertNotIn(source["checkpoint_path"], shell)
        env = {entry["name"]: entry.get("value") for entry in container["env"]}
        self.assertEqual(env["AI_SOURCE_CHECKPOINT"], source["checkpoint_path"])
        source["chain_id"] = "chain;$(touch-pwned)"
        with self.assertRaisesRegex(generator.GenerationError, "Unsafe"):
            generator.render_job(self.template, source, "B")

    def test_cross_hash_mismatch_is_rejected(self):
        manifest, inventory, _, _ = self.generate_stage_b()
        document = json.loads(inventory)
        document["generated_yaml_sha256"] = "0" * 64
        with self.assertRaisesRegex(generator.GenerationError, "YAML/job-inventory"):
            generator.verify_generated_pair(manifest, json.dumps(document))

    def test_stage_c_failure_creates_no_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage-c.yaml"
            index = Path(directory) / "stage-c.json"
            with patch.object(generator, "RUNTIME_HARDENING_COMMIT", generator.CODE_COMMIT):
                status = generator.main(
                    [
                        "--stage", "C",
                        "--parallelism", "8",
                        "--stage-a-inventory", str(self.inventory_path),
                        "--predecessor-job-inventory", str(Path(directory) / "missing.json"),
                        "--output", str(output),
                        "--job-inventory-output", str(index),
                    ]
                )
            self.assertEqual(status, 1)
            self.assertFalse(output.exists())
            self.assertFalse(index.exists())

    def test_tee_failure_prevents_completion_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "completed.json"
            script = f'''set -Eeuo pipefail
set +e
bash -c 'exit 0' | bash -c 'cat >/dev/null; exit 7'
pipeline_status=("${{PIPESTATUS[@]}}")
set -e
test "${{#pipeline_status[@]}}" -eq 2
test "${{pipeline_status[0]}}" -eq 0
test "${{pipeline_status[1]}}" -eq 0
touch {marker}
'''
            result = subprocess.run(["bash", "-c", script], check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())

    def test_postcondition_failure_prevents_completion_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "completed.json"
            missing = Path(directory) / "missing-checkpoint.pt"
            script = f'''set -Eeuo pipefail
set +e
bash -c 'exit 0' | tee /dev/null
pipeline_status=("${{PIPESTATUS[@]}}")
set -e
test "${{pipeline_status[0]}}" -eq 0
test "${{pipeline_status[1]}}" -eq 0
test -s {missing}
touch {marker}
'''
            result = subprocess.run(["bash", "-c", script], check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
