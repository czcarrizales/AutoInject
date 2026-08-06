import copy
import re
import unittest

from scripts import cross_pair_experiment
from scripts import cross_pair_mapping
from scripts import generate_continuation_jobs as continuation


class CrossPairExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = continuation.load_stage_a_inventory(
            continuation.DEFAULT_STAGE_A_INVENTORY
        )
        cls.records_by_pair = {
            continuation.pair_key(record): record
            for record in cls.records
        }

    def test_builds_32_complete_specs(self):
        specs = cross_pair_experiment.build_cross_pair_specs(self.records)

        self.assertEqual(len(specs), 32)
        self.assertEqual(
            [spec["completion_index"] for spec in specs],
            list(range(32)),
        )
        self.assertEqual(
            {
                (
                    spec["target"]["suite"],
                    spec["target"]["user_task"],
                    spec["target"]["injection_task"],
                )
                for spec in specs
            },
            continuation.expected_pairs(),
        )

    def test_records_source_and_target_separately(self):
        specs = cross_pair_experiment.build_cross_pair_specs(self.records)

        for spec in specs:
            source = spec["source"]
            target = spec["target"]

            self.assertEqual(source["suite"], target["suite"])
            self.assertEqual(
                source["injection_task"],
                target["injection_task"],
            )
            self.assertNotEqual(
                source["user_task"],
                target["user_task"],
            )

    def test_first_spec_has_expected_workspace_transfer(self):
        spec = cross_pair_experiment.build_cross_pair_specs(
            self.records
        )[0]

        self.assertEqual(
            spec["target"],
            {
                "suite": "workspace",
                "user_task": "user_task_38",
                "injection_task": "injection_task_10",
            },
        )
        self.assertEqual(
            spec["source"],
            {
                "suite": "workspace",
                "user_task": "user_task_39",
                "injection_task": "injection_task_10",
            },
        )
        self.assertIn(
            "workspace-u38-i10-from-workspace-u39-i10",
            spec["run_id"],
        )

    def test_checkpoint_provenance_matches_named_source(self):
        specs = cross_pair_experiment.build_cross_pair_specs(self.records)

        for spec in specs:
            source_key = (
                spec["source"]["suite"],
                spec["source"]["user_task"],
                spec["source"]["injection_task"],
            )
            source_record = self.records_by_pair[source_key]

            self.assertEqual(
                spec["source_checkpoint_path"],
                source_record["checkpoint_path"],
            )
            self.assertEqual(
                spec["source_checkpoint_sha256"],
                source_record["checkpoint_sha256"],
            )
            self.assertEqual(
                spec["source_checkpoint_state_path"],
                source_record["checkpoint_state_path"],
            )
            self.assertEqual(
                spec["source_checkpoint_state_sha256"],
                source_record["checkpoint_state_sha256"],
            )

    def test_identifiers_and_output_paths_are_unique(self):
        specs = cross_pair_experiment.build_cross_pair_specs(self.records)

        for field in (
            "chain_id",
            "run_id",
            "experiment_id",
            "result_root",
            "stage_record_path",
            "configuration_fingerprint",
        ):
            values = [spec[field] for spec in specs]
            self.assertEqual(len(values), len(set(values)))

    def test_cross_pair_outputs_are_separate_from_continuations(self):
        specs = cross_pair_experiment.build_cross_pair_specs(self.records)

        for spec in specs:
            self.assertTrue(
                spec["result_root"].startswith(
                    cross_pair_experiment.RECORDS_ROOT + "/"
                )
            )
            self.assertNotIn("/continuations/", spec["result_root"])
            self.assertNotIn("/stage-b/", spec["result_root"])
            self.assertNotIn("/stage-c/", spec["result_root"])

    def test_fingerprint_is_bound_to_source_checkpoint(self):
        mappings = cross_pair_mapping.build_cross_pair_mappings(
            self.records
        )
        mapping = mappings[0]
        resolved_identifiers = cross_pair_experiment.identifiers(mapping)

        original = cross_pair_experiment.configuration_fingerprint(
            mapping,
            resolved_identifiers,
        )

        changed = copy.deepcopy(mapping)
        changed["source_descriptor"]["checkpoint_sha256"] = "0" * 64

        modified = cross_pair_experiment.configuration_fingerprint(
            changed,
            resolved_identifiers,
        )

        self.assertNotEqual(original, modified)
        self.assertRegex(original, re.compile(r"^[0-9a-f]{64}$"))

    def test_rejects_self_warm_disguised_as_cross_pair(self):
        mapping = cross_pair_mapping.build_cross_pair_mappings(
            self.records
        )[0]
        tampered = copy.deepcopy(mapping)
        tampered["source"] = copy.deepcopy(tampered["target"])
        tampered["source_descriptor"] = copy.deepcopy(
            self.records_by_pair[
                (
                    tampered["target"]["suite"],
                    tampered["target"]["user_task"],
                    tampered["target"]["injection_task"],
                )
            ]
        )

        with self.assertRaisesRegex(
            cross_pair_experiment.CrossPairExperimentError,
            "cannot equal",
        ):
            cross_pair_experiment.identifiers(tampered)


if __name__ == "__main__":
    unittest.main()
