import ast
import json
import tempfile
import unittest
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPOSITORY_ROOT / "autoinject-stage-a-checkpoint-hash-inspection-v2.yaml"
)
V3_MANIFEST_PATH = (
    REPOSITORY_ROOT / "autoinject-stage-a-checkpoint-hash-inspection-v3.yaml"
)
HISTORICAL_WORKSPACE_CONFIG = REPOSITORY_ROOT / (
    "frozen-baseline-v1-analysis/raw/autoinject-frozen-baseline-v1/"
    "run-02-workspace-u38-i10/seed-1/20260721T013706Z/run/.hydra/config.yaml"
)


def load_embedded_validator():
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    shell = document["spec"]["template"]["spec"]["containers"][0]["args"][0]
    prefix = "python3 - <<'PY'\n"
    start = shell.index(prefix) + len(prefix)
    end = shell.rindex("\nPY\n")
    embedded_python = shell[start:end]
    parsed = ast.parse(embedded_python, filename=str(MANIFEST_PATH))

    retained = []
    constant_names = {
        "CANONICAL_SUITES",
        "HYDRA_SUITE_LINE",
        "HYDRA_PLAIN_SCALAR",
        "HYDRA_INTEGER_SCALAR",
        "HYDRA_FLOAT_SCALAR",
    }
    for node in parsed.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            retained.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constant_names
            for target in node.targets
        ):
            retained.append(node)

    namespace = {}
    validator_module = ast.Module(body=retained, type_ignores=[])
    exec(compile(validator_module, str(MANIFEST_PATH), "exec"), namespace)
    return namespace


class HydraSuiteValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = load_embedded_validator()

    def parse(self, text, expected="workspace"):
        return self.validator["parse_hydra_suite"](
            text, expected, Path("/evidence/run/.hydra/config.yaml")
        )

    def assert_suite_failure(self, text, expected="workspace"):
        with self.assertRaises(RuntimeError) as caught:
            self.parse(text, expected)
        message = str(caught.exception)
        self.assertIn(f"expected_suite={expected!r}", message)
        self.assertIn("raw_suite_value=", message)
        self.assertIn("normalized_suite=", message)
        self.assertIn("raw_value_type=", message)
        self.assertIn("config_path=/evidence/run/.hydra/config.yaml", message)
        return message

    def test_exact_retained_workspace_config(self):
        source = {
            "suite": "workspace",
            "user_task": "user_task_38",
            "injection_task": "injection_task_10",
        }
        self.validator["validate_hydra_identity"](
            HISTORICAL_WORKSPACE_CONFIG, source
        )

    def test_all_canonical_direct_suite_scalars(self):
        for suite in ("workspace", "banking", "travel", "slack"):
            with self.subTest(suite=suite):
                self.assertEqual(self.parse(f"suite: {suite}\n", suite), suite)

    def test_quoted_direct_suite_scalars(self):
        self.assertEqual(self.parse("suite: 'workspace'\n"), "workspace")
        self.assertEqual(self.parse('suite: "workspace"\n'), "workspace")

    def test_malformed_or_structured_values_fail_closed(self):
        cases = (
            "suite:\n",
            "suite: [workspace]\n",
            "suite: {name: workspace}\n",
            "suite: true\n",
            "suite: 1\n",
            "suite: workspace extra\n",
            'suite: "workspace\n',
        )
        for text in cases:
            with self.subTest(text=text):
                self.assert_suite_failure(text)

    def test_missing_and_nested_suite_fail_closed(self):
        self.assert_suite_failure("user_tasks: [user_task_38]\n")
        self.assert_suite_failure("config:\n  suite: workspace\n")

    def test_duplicate_suite_keys_fail_closed(self):
        message = self.assert_suite_failure("suite: workspace\nsuite: workspace\n")
        self.assertIn("duplicate top-level suite keys", message)
        self.assertIn("raw_value_type=list", message)

    def test_unknown_aliases_fail_closed(self):
        for alias in ("Workspace", "workplace", "workspace_v1", "email"):
            with self.subTest(alias=alias):
                message = self.assert_suite_failure(f"suite: {alias}\n")
                self.assertIn(f"raw_suite_value={alias!r}", message)
                self.assertIn("normalized_suite=None", message)
                self.assertIn("raw_value_type=str", message)

    def test_cross_suite_mismatches_fail(self):
        for actual, expected in (
            ("workspace", "banking"),
            ("banking", "travel"),
            ("travel", "slack"),
            ("slack", "workspace"),
        ):
            with self.subTest(actual=actual, expected=expected):
                message = self.assert_suite_failure(
                    f"suite: {actual}\n", expected=expected
                )
                self.assertIn(f"raw_suite_value={actual!r}", message)
                self.assertIn(f"normalized_suite={actual!r}", message)
                self.assertIn("raw_value_type=str", message)

    def test_user_and_injection_mismatches_still_fail(self):
        config = (
            "suite: workspace\n"
            "user_tasks:\n"
            "- user_task_39\n"
            "injection_tasks:\n"
            "- injection_task_11\n"
        )
        expected = {
            "suite": "workspace",
            "user_task": "user_task_38",
            "injection_task": "injection_task_10",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(config, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "user-task identity mismatch"):
                self.validator["validate_hydra_identity"](path, expected)

        expected["user_task"] = "user_task_39"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(config, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "injection-task identity mismatch"):
                self.validator["validate_hydra_identity"](path, expected)

    def test_v3_preserves_validator_and_adds_state_byte_evidence(self):
        document = yaml.safe_load(V3_MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            document["metadata"]["name"],
            "autoinject-stage-a-checkpoint-hash-inspection-v3",
        )
        shell = document["spec"]["template"]["spec"]["containers"][0]["args"][0]
        prefix = "python3 - <<'PY'\n"
        embedded = shell[shell.index(prefix) + len(prefix):shell.rindex("\nPY\n")]
        compile(embedded, str(V3_MANIFEST_PATH), "exec")
        self.assertIn('"checkpoint_state_sha256": hash_file(state_path)', embedded)
        self.assertIn('"checkpoint_state_size_bytes": state_metadata.st_size', embedded)
        mount = document["spec"]["template"]["spec"]["containers"][0]["volumeMounts"][0]
        self.assertTrue(mount["readOnly"])
        self.assertNotIn("nvidia.com/gpu", json.dumps(document))

if __name__ == "__main__":
    unittest.main()
