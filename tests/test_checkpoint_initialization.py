import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from omegaconf import OmegaConf

from rlpi.agentdojo import adaptive_agentdojo


class CheckpointInitializationValidationTests(unittest.TestCase):
    def _config(self, mode, source_checkpoint_path):
        return OmegaConf.create(
            {
                "initialization_mode": mode,
                "source_checkpoint_path": source_checkpoint_path,
            }
        )

    def test_cold_with_null_path_passes(self):
        adaptive_agentdojo.validate_checkpoint_initialization(
            self._config("cold", None)
        )

    def test_cold_with_non_null_path_fails(self):
        with self.assertRaisesRegex(ValueError, "cold.*must be null"):
            adaptive_agentdojo.validate_checkpoint_initialization(
                self._config("cold", "/tmp/checkpoint.pt")
            )

    def test_policy_with_null_path_fails(self):
        with self.assertRaisesRegex(ValueError, "policy.*requires"):
            adaptive_agentdojo.validate_checkpoint_initialization(
                self._config("policy", None)
            )

    def test_policy_with_missing_file_fails(self):
        with TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing.pt"

            with self.assertRaisesRegex(ValueError, "does not exist"):
                adaptive_agentdojo.validate_checkpoint_initialization(
                    self._config("policy", str(missing_path))
                )

    def test_policy_with_directory_fails(self):
        with TemporaryDirectory(suffix=".pt") as directory:
            with self.assertRaisesRegex(ValueError, "regular file"):
                adaptive_agentdojo.validate_checkpoint_initialization(
                    self._config("policy", directory)
                )

    def test_policy_with_empty_file_fails(self):
        with TemporaryDirectory() as directory:
            empty_path = Path(directory) / "empty.pt"
            empty_path.touch()

            with self.assertRaisesRegex(ValueError, "non-empty"):
                adaptive_agentdojo.validate_checkpoint_initialization(
                    self._config("policy", str(empty_path))
                )

    def test_policy_with_non_pt_file_fails(self):
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.bin"
            checkpoint_path.write_bytes(b"checkpoint")

            with self.assertRaisesRegex(ValueError, r"end in \.pt"):
                adaptive_agentdojo.validate_checkpoint_initialization(
                    self._config("policy", str(checkpoint_path))
                )

    def test_policy_with_unreadable_file_fails(self):
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            checkpoint_path.write_bytes(b"checkpoint")

            with (
                patch.object(
                    Path,
                    "open",
                    side_effect=PermissionError("permission denied"),
                ),
                self.assertRaisesRegex(ValueError, "not readable"),
            ):
                adaptive_agentdojo.validate_checkpoint_initialization(
                    self._config("policy", str(checkpoint_path))
                )

    def test_policy_with_valid_file_passes(self):
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            checkpoint_path.write_bytes(b"checkpoint")

            adaptive_agentdojo.validate_checkpoint_initialization(
                self._config("policy", str(checkpoint_path))
            )

    def test_unsupported_mode_fails(self):
        with self.assertRaisesRegex(ValueError, "Unsupported initialization_mode"):
            adaptive_agentdojo.validate_checkpoint_initialization(
                self._config("full", None)
            )

    def test_invalid_config_fails_before_any_experiment_setup(self):
        cfg = self._config("policy", None)

        with (
            patch.object(adaptive_agentdojo, "set_seed") as set_seed,
            patch.object(adaptive_agentdojo, "get_suite") as get_suite,
            patch.object(adaptive_agentdojo, "ModelsEnum") as models_enum,
            patch.object(
                adaptive_agentdojo, "setup_pipeline_and_attacker"
            ) as setup_pipeline,
            patch.object(
                adaptive_agentdojo, "get_attack_learner"
            ) as get_attack_learner,
            patch.object(
                adaptive_agentdojo, "execute_single_benchmark"
            ) as evaluate_victim,
            patch.object(
                adaptive_agentdojo, "run_adaptive_attack"
            ) as run_adaptive_attack,
            self.assertRaises(ValueError),
        ):
            adaptive_agentdojo.main.__wrapped__(cfg)

        set_seed.assert_not_called()
        get_suite.assert_not_called()
        models_enum.assert_not_called()
        setup_pipeline.assert_not_called()
        get_attack_learner.assert_not_called()
        evaluate_victim.assert_not_called()
        run_adaptive_attack.assert_not_called()


if __name__ == "__main__":
    unittest.main()
