import copy
import unittest

from scripts import cross_pair_mapping
from scripts import generate_continuation_jobs as continuation


def pair_from_section(section):
    return (
        section["suite"],
        section["user_task"],
        section["injection_task"],
    )


class CrossPairMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = continuation.load_stage_a_inventory(
            continuation.DEFAULT_STAGE_A_INVENTORY
        )

    def test_builds_complete_32_target_mapping(self):
        mappings = cross_pair_mapping.build_cross_pair_mappings(self.records)

        self.assertEqual(len(mappings), 32)
        self.assertEqual(
            [mapping["completion_index"] for mapping in mappings],
            list(range(32)),
        )
        self.assertEqual(
            {pair_from_section(mapping["target"]) for mapping in mappings},
            continuation.expected_pairs(),
        )
        self.assertEqual(
            {pair_from_section(mapping["source"]) for mapping in mappings},
            continuation.expected_pairs(),
        )

    def test_only_user_task_changes(self):
        mappings = cross_pair_mapping.build_cross_pair_mappings(self.records)

        for mapping in mappings:
            target = mapping["target"]
            source = mapping["source"]

            self.assertEqual(source["suite"], target["suite"])
            self.assertEqual(
                source["injection_task"],
                target["injection_task"],
            )
            self.assertNotEqual(
                source["user_task"],
                target["user_task"],
            )

    def test_known_workspace_pair_and_reverse_direction(self):
        mappings = cross_pair_mapping.build_cross_pair_mappings(self.records)

        self.assertEqual(
            pair_from_section(mappings[0]["target"]),
            ("workspace", "user_task_38", "injection_task_10"),
        )
        self.assertEqual(
            pair_from_section(mappings[0]["source"]),
            ("workspace", "user_task_39", "injection_task_10"),
        )

        self.assertEqual(
            pair_from_section(mappings[4]["target"]),
            ("workspace", "user_task_39", "injection_task_10"),
        )
        self.assertEqual(
            pair_from_section(mappings[4]["source"]),
            ("workspace", "user_task_38", "injection_task_10"),
        )

    def test_mapping_is_reciprocal(self):
        mappings = cross_pair_mapping.build_cross_pair_mappings(self.records)
        source_for_target = {
            pair_from_section(mapping["target"]): pair_from_section(
                mapping["source"]
            )
            for mapping in mappings
        }

        for target, source in source_for_target.items():
            self.assertEqual(source_for_target[source], target)

    def test_rejects_missing_record(self):
        with self.assertRaisesRegex(
            cross_pair_mapping.CrossPairMappingError,
            "exactly 32",
        ):
            cross_pair_mapping.build_cross_pair_mappings(self.records[:-1])

    def test_rejects_duplicate_record(self):
        records = copy.deepcopy(self.records)
        records[-1] = copy.deepcopy(records[0])

        with self.assertRaisesRegex(
            cross_pair_mapping.CrossPairMappingError,
            "Duplicate",
        ):
            cross_pair_mapping.build_cross_pair_mappings(records)

    def test_rejects_unsupported_identity(self):
        records = copy.deepcopy(self.records)
        records[0]["user_task"] = "user_task_999"

        with self.assertRaisesRegex(
            cross_pair_mapping.CrossPairMappingError,
            "Unexpected Stage A task identity",
        ):
            cross_pair_mapping.build_cross_pair_mappings(records)


if __name__ == "__main__":
    unittest.main()
