import copy
import hashlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path

from scripts import import_stage_a_checkpoint_inventory as importer


class StageAInventoryImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selections = importer.load_selection(importer.DEFAULT_SELECTION)

    def records(self):
        records = []
        for selection in self.selections:
            identity = "/".join(importer.pair_key(selection))
            digest = hashlib.sha256(identity.encode()).hexdigest()
            if "known_checkpoint_sha256" in selection:
                digest = selection["known_checkpoint_sha256"]
            records.append(
                {
                    "suite": selection["suite"],
                    "user_task": selection["user_task"],
                    "injection_task": selection["injection_task"],
                    "checkpoint_path": selection["checkpoint_path"],
                    "checkpoint_state_path": selection["checkpoint_state_path"],
                    "checkpoint_sha256": digest,
                    "checkpoint_size_bytes": 3_087_560_784,
                    "checkpoint_state_sha256": hashlib.sha256((identity + "/state").encode()).hexdigest(),
                    "checkpoint_state_size_bytes": 4096,
                    "queries_used": selection["expected_queries_used"],
                    "chain_id": selection["chain_id"],
                }
            )
        return records

    def test_accepts_complete_output_and_orders_it_by_selection(self):
        records = list(reversed(self.records()))
        validated = importer.validate_records(records, self.selections)
        self.assertEqual(len(validated), 32)
        self.assertEqual(
            [importer.pair_key(record) for record in validated],
            [importer.pair_key(record) for record in self.selections],
        )
        self.assertEqual(validated[0]["stage"], "A")
        self.assertIn("cumulative_queries_used", validated[0])
        rendered = importer.render_inventory(validated)
        self.assertEqual(rendered, importer.render_inventory(validated))
        document = json.loads(rendered)
        self.assertEqual(document["source_stage"], "A")
        self.assertEqual(document["schema_version"], 2)
        self.assertEqual(document["inspection_job_name"], importer.INSPECTION_JOB_NAME)
        self.assertIn("checkpoint_state_sha256", document["records"][0])
        self.assertEqual(
            document["records"][0]["stage_record_id"],
            f"stage-a-inventory-v3.json#{document['records'][0]['chain_id']}",
        )

    def test_v2_evidence_cannot_satisfy_v3_inventory_schema(self):
        inspection_path = (
            importer.REPOSITORY_ROOT
            / "continuation/stage-a-checkpoint-hash-inspection-v2.jsonl"
        )
        inventory_path = importer.REPOSITORY_ROOT / "continuation/stage-a-inventory.json"
        with inspection_path.open(encoding="utf-8") as stream:
            inspected = importer.read_jsonl(stream)
        with self.assertRaisesRegex(importer.InventoryError, "invalid fields"):
            importer.validate_records(inspected, self.selections)
        durable = json.loads(inventory_path.read_text(encoding="utf-8"))
        self.assertNotEqual(durable.get("inspection_job_name"), importer.INSPECTION_JOB_NAME)

    def test_refuses_partial_output(self):
        content = "".join(json.dumps(record) + "\n" for record in self.records()[:-1])
        with self.assertRaisesRegex(importer.InventoryError, "exactly 32"):
            importer.read_jsonl(io.StringIO(content))

    def test_refuses_duplicate_pair(self):
        records = self.records()
        records[-1] = copy.deepcopy(records[0])
        with self.assertRaisesRegex(importer.InventoryError, "duplicate"):
            importer.validate_records(records, self.selections)

    def test_refuses_invalid_sha256(self):
        records = self.records()
        records[0]["checkpoint_sha256"] = "not-a-sha256"
        with self.assertRaisesRegex(importer.InventoryError, "invalid SHA-256"):
            importer.validate_records(records, self.selections)

    def test_refuses_duplicate_checkpoint_hash(self):
        records = self.records()
        records[1]["checkpoint_sha256"] = records[0]["checkpoint_sha256"]
        with self.assertRaisesRegex(importer.InventoryError, "duplicate checkpoint hashes"):
            importer.validate_records(records, self.selections)

    def test_refuses_changed_state_hash_and_duplicate_state_hash(self):
        records = self.records()
        records[0]["checkpoint_state_sha256"] = "INVALID"
        with self.assertRaisesRegex(importer.InventoryError, "checkpoint-state SHA-256"):
            importer.validate_records(records, self.selections)
        records = self.records()
        records[1]["checkpoint_state_sha256"] = records[0]["checkpoint_state_sha256"]
        with self.assertRaisesRegex(importer.InventoryError, "duplicate checkpoint-state hashes"):
            importer.validate_records(records, self.selections)

    def test_refuses_source_path_change(self):
        records = self.records()
        records[0]["checkpoint_path"] = "/workspace/results/other/checkpoint.pt"
        with self.assertRaisesRegex(importer.InventoryError, "selected source"):
            importer.validate_records(records, self.selections)

    def test_refuses_query_count_change(self):
        records = self.records()
        records[0]["queries_used"] += 260
        with self.assertRaisesRegex(importer.InventoryError, "finalized report"):
            importer.validate_records(records, self.selections)

    def test_refuses_extra_non_json_log_output(self):
        content = "unexpected preamble\n" + "".join(
            json.dumps(record) + "\n" for record in self.records()
        )
        with self.assertRaisesRegex(importer.InventoryError, "not JSON"):
            importer.read_jsonl(io.StringIO(content))

    def test_inventory_is_immutable(self):
        content = importer.render_inventory(
            importer.validate_records(self.records(), self.selections)
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage-a-inventory.json"
            importer.write_immutable(output, content)
            self.assertFalse(list(Path(directory).glob(".stage-a-inventory.json.*")))
            importer.write_immutable(output, content)
            self.assertEqual(output.read_text(), content)
            self.assertFalse(list(Path(directory).glob(".stage-a-inventory.json.*")))
            with self.assertRaisesRegex(importer.InventoryError, "Refusing to overwrite"):
                importer.write_immutable(output, content + " ")
            self.assertFalse(list(Path(directory).glob(".stage-a-inventory.json.*")))

    def test_concurrent_differing_creation_never_replaces_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage-a-inventory.json"
            barrier = threading.Barrier(2)
            results = []

            def publish(content):
                barrier.wait()
                try:
                    importer.write_immutable(output, content)
                    results.append(("ok", content))
                except importer.InventoryError:
                    results.append(("rejected", content))

            threads = [
                threading.Thread(target=publish, args=("first\n",)),
                threading.Thread(target=publish, args=("second\n",)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(sorted(status for status, _ in results), ["ok", "rejected"])
            winner = next(content for status, content in results if status == "ok")
            self.assertEqual(output.read_text(), winner)
            self.assertFalse(list(Path(directory).glob(".stage-a-inventory.json.*")))


if __name__ == "__main__":
    unittest.main()
