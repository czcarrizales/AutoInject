import builtins
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf
from torch import nn

from rlpi.agentdojo import adaptive_agentdojo
from rlpi.attack.learners.trl_suffix.learner import TRLSuffixLearner


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.config = SimpleNamespace(model_type="tiny-policy")


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

    def test_legacy_config_with_both_fields_absent_passes(self):
        adaptive_agentdojo.validate_checkpoint_initialization(
            OmegaConf.create({})
        )

    def test_initialization_mode_without_source_path_fails(self):
        with self.assertRaisesRegex(ValueError, "both be present or both"):
            adaptive_agentdojo.validate_checkpoint_initialization(
                OmegaConf.create({"initialization_mode": "cold"})
            )

    def test_source_path_without_initialization_mode_fails(self):
        with self.assertRaisesRegex(ValueError, "both be present or both"):
            adaptive_agentdojo.validate_checkpoint_initialization(
                OmegaConf.create({"source_checkpoint_path": None})
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


class CheckpointInitializationDispatchTests(unittest.TestCase):
    def _learner(self):
        return SimpleNamespace(
            load_policy_weights_only=MagicMock(),
            load_model=MagicMock(),
            modify_tasks=MagicMock(),
            update_scores=MagicMock(),
        )

    def _run_config(self, mode, source_path, query_budget=0):
        return OmegaConf.create(
            {
                "initialization_mode": mode,
                "source_checkpoint_path": source_path,
                "query_budget": query_budget,
                "attack": "direct",
                "defense": None,
                "system_message": None,
                "model": "test-model",
                "max_tokens": None,
                "learner": {"type": "test"},
            }
        )

    def test_cold_dispatch_calls_no_checkpoint_loader(self):
        learner = self._learner()

        adaptive_agentdojo.initialize_learner_from_checkpoint(
            self._run_config("cold", None), learner
        )

        learner.load_policy_weights_only.assert_not_called()
        learner.load_model.assert_not_called()

    def test_policy_dispatch_loads_policy_exactly_once(self):
        learner = self._learner()
        checkpoint_path = "/checkpoints/source.pt"

        adaptive_agentdojo.initialize_learner_from_checkpoint(
            self._run_config("policy", checkpoint_path), learner
        )

        learner.load_policy_weights_only.assert_called_once_with(
            checkpoint_path
        )
        learner.load_model.assert_not_called()

    def test_unsupported_dispatch_mode_fails_clearly(self):
        learner = self._learner()

        with self.assertRaisesRegex(
            ValueError, "Unsupported initialization_mode"
        ):
            adaptive_agentdojo.initialize_learner_from_checkpoint(
                self._run_config("full", None), learner
            )

        learner.load_policy_weights_only.assert_not_called()
        learner.load_model.assert_not_called()

    def test_policy_dispatch_rejects_incompatible_learner(self):
        learner = SimpleNamespace(load_model=MagicMock())

        with self.assertRaisesRegex(
            TypeError, "does not support policy-only checkpoint initialization"
        ):
            adaptive_agentdojo.initialize_learner_from_checkpoint(
                self._run_config("policy", "/checkpoints/source.pt"),
                learner,
            )

        learner.load_model.assert_not_called()

    def _assert_legacy_sidecar_is_bypassed(self, mode):
        learner = self._learner()
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "source.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            legacy_sidecar = Path(directory) / "checkpoint_state.json"
            legacy_sidecar.write_text("not valid json")
            cfg = self._run_config(
                mode,
                str(checkpoint_path) if mode == "policy" else None,
            )

            with (
                patch.object(
                    adaptive_agentdojo,
                    "get_user_tasks",
                    return_value=[],
                ),
                patch.object(
                    adaptive_agentdojo,
                    "get_wrapped_injection_tasks",
                    return_value={},
                ),
                patch.object(
                    adaptive_agentdojo,
                    "setup_pipeline_and_attacker",
                    return_value=(object(), object()),
                ),
                patch.object(
                    adaptive_agentdojo,
                    "get_attack_learner",
                    return_value=learner,
                ),
                patch.object(
                    adaptive_agentdojo,
                    "_find_latest_checkpoint",
                    side_effect=AssertionError(
                        "explicit initialization searched for legacy checkpoint"
                    ),
                ) as find_latest_checkpoint,
                patch.object(adaptive_agentdojo, "_save_checkpoint"),
            ):
                adaptive_agentdojo.run_adaptive_attack(
                    cfg=cfg,
                    suite=object(),
                    model=object(),
                    logdir=Path(directory),
                )

        learner.load_model.assert_not_called()
        find_latest_checkpoint.assert_not_called()
        if mode == "policy":
            learner.load_policy_weights_only.assert_called_once_with(
                str(checkpoint_path)
            )
        else:
            learner.load_policy_weights_only.assert_not_called()

    def test_cold_mode_bypasses_existing_legacy_sidecar(self):
        self._assert_legacy_sidecar_is_bypassed("cold")

    def test_policy_mode_bypasses_existing_legacy_sidecar(self):
        self._assert_legacy_sidecar_is_bypassed("policy")

    def test_missing_initialization_fields_use_legacy_resume(self):
        learner = self._learner()
        cfg = self._run_config("cold", None)
        del cfg.initialization_mode
        del cfg.source_checkpoint_path

        with TemporaryDirectory() as directory:
            legacy_sidecar = Path(directory) / "checkpoint_state.json"
            legacy_sidecar.write_text(
                json.dumps(
                    {
                        "checkpoint_version": "1.0",
                        "learner_state": {"queries_used": 0},
                    }
                )
            )

            with (
                patch.object(
                    adaptive_agentdojo,
                    "get_user_tasks",
                    return_value=[],
                ),
                patch.object(
                    adaptive_agentdojo,
                    "get_wrapped_injection_tasks",
                    return_value={},
                ),
                patch.object(
                    adaptive_agentdojo,
                    "setup_pipeline_and_attacker",
                    return_value=(object(), object()),
                ),
                patch.object(
                    adaptive_agentdojo,
                    "get_attack_learner",
                    return_value=learner,
                ),
            ):
                adaptive_agentdojo.run_adaptive_attack(
                    cfg=cfg,
                    suite=object(),
                    model=object(),
                    logdir=Path(directory),
                )

        learner.load_model.assert_called_once_with(str(legacy_sidecar))
        learner.load_policy_weights_only.assert_not_called()

    def test_initialization_finishes_before_controller_execution(self):
        events = []
        learner = self._learner()
        learner.load_policy_weights_only.side_effect = lambda path: events.append(
            "initialized"
        )
        learner.modify_tasks.side_effect = lambda **kwargs: events.append(
            "modify_tasks"
        )
        user_task = SimpleNamespace(PROMPT="user prompt")
        cfg = self._run_config("policy", "/checkpoints/source.pt", 1)

        def construct_learner(**kwargs):
            events.append("constructed")
            return learner

        def run_controller(**kwargs):
            events.append("controller")
            return {}, {}

        with (
            patch.object(
                adaptive_agentdojo,
                "get_user_tasks",
                return_value=[user_task],
            ),
            patch.object(
                adaptive_agentdojo,
                "get_wrapped_injection_tasks",
                return_value={"injection": object()},
            ),
            patch.object(
                adaptive_agentdojo,
                "setup_pipeline_and_attacker",
                return_value=(object(), object()),
            ),
            patch.object(
                adaptive_agentdojo,
                "get_attack_learner",
                side_effect=construct_learner,
            ),
            patch.object(
                adaptive_agentdojo,
                "_run_benchmarks",
                side_effect=run_controller,
            ),
            patch.object(
                adaptive_agentdojo,
                "calculate_average_scores",
                return_value=(0.0, 0.0),
            ),
            patch.object(adaptive_agentdojo, "_save_checkpoint"),
        ):
            adaptive_agentdojo.run_adaptive_attack(
                cfg=cfg,
                suite=object(),
                model=object(),
                logdir=Path("/unused"),
            )

        self.assertEqual(
            events,
            ["constructed", "initialized", "modify_tasks", "controller"],
        )
        learner.load_model.assert_not_called()


class PolicyWeightsOnlyCheckpointTests(unittest.TestCase):
    def _learner(self):
        learner = TRLSuffixLearner.__new__(TRLSuffixLearner)
        learner.attack_model_name = "tiny-attack-model"
        learner.policy = TinyPolicy()
        learner.best_suffix = "keep-best-suffix"
        learner.best_reward = 0.75
        learner.training_count = 4
        learner.experience_count = 9
        learner.early_stopped = True
        learner.current_prompt = "keep-current-prompt"
        learner.current_suffix = "keep-current-suffix"
        learner.current_user_task = object()
        learner.injection_task = object()
        learner.evaluator = object()
        learner.experiment_reporting = object()
        learner._checkpoint_queries_used = 17
        learner.max_suffix_length = 20
        learner.temperature = 0.7
        learner.top_p = 0.9
        learner.grpo_num_generations = 2
        learner.grpo_num_iterations = 1
        learner.grpo_learning_rate = 1e-7
        learner.min_experiences_for_training = 2
        return learner

    def _checkpoint(self, source_policy=None):
        source_policy = source_policy or TinyPolicy()
        return {
            "policy_checkpoint_format_version": "1.0",
            "policy_state_dict": {
                key: tensor.detach().clone()
                for key, tensor in source_policy.state_dict().items()
            },
            "model_config": {
                "attack_model_name": "tiny-attack-model",
                "policy_class": (
                    f"{TinyPolicy.__module__}.{TinyPolicy.__qualname__}"
                ),
                "model_type": "tiny-policy",
                "dtype": "torch.float32",
            },
        }

    def _save_checkpoint(self, directory, checkpoint):
        path = Path(directory) / "checkpoint.pt"
        torch.save(checkpoint, path)
        return path

    def _tensor_snapshot(self, learner):
        return {
            key: tensor.detach().clone()
            for key, tensor in learner.policy.state_dict().items()
        }

    def _assert_tensors_unchanged(self, learner, snapshot):
        current = learner.policy.state_dict()
        self.assertEqual(current.keys(), snapshot.keys())
        for key, original in snapshot.items():
            current_bytes = (
                current[key]
                .detach()
                .cpu()
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            original_bytes = (
                original.detach()
                .cpu()
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            self.assertEqual(current_bytes, original_bytes, key)

    def _assert_load_fails_transactionally(self, checkpoint, message):
        learner = self._learner()
        snapshot = self._tensor_snapshot(learner)
        with TemporaryDirectory() as directory:
            path = self._save_checkpoint(directory, checkpoint)
            with patch.object(
                learner.policy,
                "load_state_dict",
                wraps=learner.policy.load_state_dict,
            ) as load_state_dict:
                with self.assertRaisesRegex(ValueError, message):
                    learner.load_policy_weights_only(path)
            load_state_dict.assert_not_called()
        self._assert_tensors_unchanged(learner, snapshot)

    def test_valid_tiny_model_checkpoint_loads_policy_only(self):
        learner = self._learner()
        source_policy = TinyPolicy()
        with torch.no_grad():
            for tensor in source_policy.parameters():
                tensor.fill_(3.0)

        unchanged = {
            "best_suffix": learner.best_suffix,
            "best_reward": learner.best_reward,
            "training_count": learner.training_count,
            "experience_count": learner.experience_count,
            "early_stopped": learner.early_stopped,
            "current_prompt": learner.current_prompt,
            "current_suffix": learner.current_suffix,
            "current_user_task": learner.current_user_task,
            "injection_task": learner.injection_task,
            "evaluator": learner.evaluator,
            "experiment_reporting": learner.experiment_reporting,
            "_checkpoint_queries_used": learner._checkpoint_queries_used,
        }

        with TemporaryDirectory() as directory:
            path = self._save_checkpoint(
                directory, self._checkpoint(source_policy)
            )
            sidecar = Path(directory) / "checkpoint_state.json"
            sidecar.write_text("not valid json")

            read_paths = []
            original_builtin_open = builtins.open
            original_path_open = Path.open
            original_io_open = io.open

            def record_read(file, mode):
                if "r" not in mode or isinstance(file, int):
                    return
                file_path = Path(file)
                if file_path.suffix == ".json":
                    raise AssertionError("policy loader opened JSON sidecar")
                if file_path != path:
                    raise AssertionError(
                        f"policy loader read unexpected path: {file_path}"
                    )
                read_paths.append(file_path)

            def guarded_builtin_open(file, mode="r", *args, **kwargs):
                record_read(file, mode)
                return original_builtin_open(file, mode, *args, **kwargs)

            def guarded_path_open(path_obj, mode="r", *args, **kwargs):
                record_read(path_obj, mode)
                return original_path_open(path_obj, mode, *args, **kwargs)

            def guarded_io_open(file, mode="r", *args, **kwargs):
                record_read(file, mode)
                return original_io_open(file, mode, *args, **kwargs)

            with (
                patch("builtins.open", side_effect=guarded_builtin_open),
                patch.object(
                    Path,
                    "open",
                    autospec=True,
                    side_effect=guarded_path_open,
                ),
                patch("io.open", side_effect=guarded_io_open),
                patch.object(
                    json,
                    "load",
                    side_effect=AssertionError(
                        "policy loader called json.load"
                    ),
                ),
                patch(
                    "rlpi.attack.learners.trl_suffix.learner.torch.load",
                    wraps=torch.load,
                ) as torch_load,
                patch.object(
                    learner,
                    "load_model",
                    side_effect=AssertionError("instance load_model called"),
                ) as instance_load_model,
                patch.object(
                    TRLSuffixLearner,
                    "load_model",
                    side_effect=AssertionError("class load_model called"),
                ) as class_load_model,
                patch.object(
                    learner.policy,
                    "load_state_dict",
                    wraps=learner.policy.load_state_dict,
                ) as load_state_dict,
            ):
                learner.load_policy_weights_only(path)

        for key, source_tensor in source_policy.state_dict().items():
            self.assertTrue(
                torch.equal(learner.policy.state_dict()[key], source_tensor),
                key,
            )
        torch_load.assert_called_once_with(
            path, map_location="cpu", weights_only=True
        )
        load_state_dict.assert_called_once()
        self.assertTrue(load_state_dict.call_args.kwargs["strict"])
        instance_load_model.assert_not_called()
        class_load_model.assert_not_called()
        self.assertTrue(read_paths)
        self.assertEqual(set(read_paths), {path})
        self.assertEqual(learner.best_suffix, unchanged["best_suffix"])
        self.assertEqual(learner.best_reward, unchanged["best_reward"])
        self.assertEqual(learner.training_count, unchanged["training_count"])
        self.assertEqual(
            learner.experience_count, unchanged["experience_count"]
        )
        self.assertEqual(learner.early_stopped, unchanged["early_stopped"])
        self.assertEqual(learner.current_prompt, unchanged["current_prompt"])
        self.assertEqual(learner.current_suffix, unchanged["current_suffix"])
        for field in (
            "current_user_task",
            "injection_task",
            "evaluator",
            "experiment_reporting",
        ):
            self.assertIs(getattr(learner, field), unchanged[field])
        self.assertEqual(
            learner._checkpoint_queries_used,
            unchanged["_checkpoint_queries_used"],
        )
        self.assertEqual(
            learner.policy_initialization_provenance["source_checkpoint_path"],
            str(path),
        )

    def test_save_model_includes_policy_compatibility_metadata(self):
        learner = self._learner()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            learner.save_model(str(path))
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=True
            )

        self.assertEqual(
            checkpoint["policy_checkpoint_format_version"], "1.0"
        )
        self.assertEqual(
            checkpoint["model_config"],
            {
                "attack_model_name": "tiny-attack-model",
                "policy_class": (
                    f"{TinyPolicy.__module__}.{TinyPolicy.__qualname__}"
                ),
                "model_type": "tiny-policy",
                "dtype": "torch.float32",
            },
        )

    def test_missing_policy_state_dict_fails_transactionally(self):
        checkpoint = self._checkpoint()
        del checkpoint["policy_state_dict"]
        self._assert_load_fails_transactionally(
            checkpoint, "policy_state_dict.*required"
        )

    def test_wrong_policy_state_dict_type_fails_transactionally(self):
        checkpoint = self._checkpoint()
        checkpoint["policy_state_dict"] = []
        self._assert_load_fails_transactionally(
            checkpoint, "policy_state_dict.*mapping"
        )

    def test_non_mapping_checkpoint_fails_transactionally(self):
        self._assert_load_fails_transactionally([], "checkpoint.*mapping")

    def test_unsupported_format_version_fails_transactionally(self):
        checkpoint = self._checkpoint()
        checkpoint["policy_checkpoint_format_version"] = "2.0"
        self._assert_load_fails_transactionally(
            checkpoint, "Unsupported policy checkpoint format version"
        )

    def test_attack_model_name_mismatch_fails_transactionally(self):
        checkpoint = self._checkpoint()
        checkpoint["model_config"]["attack_model_name"] = "wrong-model"
        self._assert_load_fails_transactionally(
            checkpoint, "attack model name mismatch"
        )

    def test_policy_class_mismatch_fails_transactionally(self):
        checkpoint = self._checkpoint()
        checkpoint["model_config"]["policy_class"] = "wrong.Policy"
        self._assert_load_fails_transactionally(
            checkpoint, "policy class mismatch"
        )

    def test_load_state_dict_failure_rolls_back_target_tensors(self):
        learner = self._learner()
        snapshot = self._tensor_snapshot(learner)
        checkpoint = self._checkpoint()
        original_load_state_dict = learner.policy.load_state_dict
        calls = 0

        def fail_after_loading(state_dict, strict=True):
            nonlocal calls
            calls += 1
            result = original_load_state_dict(state_dict, strict=strict)
            if calls == 1:
                raise RuntimeError("simulated load failure")
            return result

        with TemporaryDirectory() as directory:
            path = self._save_checkpoint(directory, checkpoint)
            with patch.object(
                learner.policy,
                "load_state_dict",
                side_effect=fail_after_loading,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "simulated load failure"
                ):
                    learner.load_policy_weights_only(path)

        self.assertEqual(calls, 2)
        self._assert_tensors_unchanged(learner, snapshot)
        self.assertFalse(
            hasattr(learner, "policy_initialization_provenance")
        )

    def test_load_and_rollback_failure_reports_integrity_error(self):
        learner = self._learner()
        checkpoint = self._checkpoint()
        original_load_state_dict = learner.policy.load_state_dict
        calls = 0

        def fail_load_and_rollback(state_dict, strict=True):
            nonlocal calls
            calls += 1
            if calls == 1:
                original_load_state_dict(state_dict, strict=strict)
                raise RuntimeError("simulated original load failure")
            raise RuntimeError("simulated rollback failure")

        with TemporaryDirectory() as directory:
            path = self._save_checkpoint(directory, checkpoint)
            with patch.object(
                learner.policy,
                "load_state_dict",
                side_effect=fail_load_and_rollback,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "transactional integrity failure.*rollback failed",
                ) as raised:
                    learner.load_policy_weights_only(path)

        self.assertEqual(calls, 2)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertIn(
            "simulated original load failure",
            str(raised.exception.__cause__),
        )
        self.assertFalse(
            hasattr(learner, "policy_initialization_provenance")
        )

    def test_missing_parameter_key_fails_transactionally(self):
        checkpoint = self._checkpoint()
        checkpoint["policy_state_dict"].pop("linear.bias")
        self._assert_load_fails_transactionally(checkpoint, "missing keys")

    def test_extra_parameter_key_fails_transactionally(self):
        checkpoint = self._checkpoint()
        checkpoint["policy_state_dict"]["extra"] = torch.zeros(1)
        self._assert_load_fails_transactionally(checkpoint, "extra keys")

    def test_tensor_shape_mismatch_fails_transactionally(self):
        checkpoint = self._checkpoint()
        checkpoint["policy_state_dict"]["linear.weight"] = torch.zeros(3, 2)
        self._assert_load_fails_transactionally(checkpoint, "shape mismatch")

    def test_tensor_dtype_mismatch_fails_transactionally(self):
        checkpoint = self._checkpoint()
        checkpoint["policy_state_dict"]["linear.weight"] = checkpoint[
            "policy_state_dict"
        ]["linear.weight"].double()
        self._assert_load_fails_transactionally(checkpoint, "dtype mismatch")


if __name__ == "__main__":
    unittest.main()
