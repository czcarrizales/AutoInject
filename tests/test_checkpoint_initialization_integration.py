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


class CheckpointInitializationIntegrationTests(unittest.TestCase):
    def _config(self, mode="cold", source_checkpoint_path=None):
        return OmegaConf.create(
            {
                "initialization_mode": mode,
                "source_checkpoint_path": source_checkpoint_path,
                "seed": 1,
                "suite": "test-suite",
                "user_tasks": ["user-task"],
                "injection_tasks": ["injection-task"],
                "query_budget": 1,
                "attack": "direct",
                "defense": None,
                "system_message": None,
                "model": "gpt-5-nano",
                "max_tokens": None,
                "learner": {"type": "test"},
            }
        )

    def _learner(self):
        learner = TRLSuffixLearner.__new__(TRLSuffixLearner)
        learner.attack_model_name = "tiny-attack-model"
        learner.policy = TinyPolicy()
        learner.best_suffix = None
        learner.best_reward = -float("inf")
        learner.training_count = 0
        learner.experience_count = 0
        learner.early_stopped = False
        learner.current_prompt = None
        learner.current_suffix = None
        learner.current_user_task = None
        learner.injection_task = None
        learner.evaluator = None
        learner.experiment_reporting = None
        learner.max_suffix_length = 20
        learner.temperature = 0.7
        learner.top_p = 0.9
        learner.grpo_num_generations = 2
        learner.grpo_num_iterations = 1
        learner.grpo_learning_rate = 1e-7
        learner.min_experiences_for_training = 2
        learner.load_policy_weights_only = MagicMock()
        learner.load_model = MagicMock()
        learner.set_current_user_task_obj = MagicMock()
        learner.modify_tasks = MagicMock()
        learner.update_scores = MagicMock()
        learner.set_experiment_reporting = MagicMock(
            side_effect=lambda reporting: setattr(
                learner, "experiment_reporting", reporting
            )
        )
        learner.set_evaluation_infrastructure = MagicMock()
        return learner

    def _assert_fresh_at_controller(self, learner):
        self.assertIsNone(learner.best_suffix)
        self.assertEqual(learner.best_reward, -float("inf"))
        self.assertEqual(learner.experience_count, 0)
        self.assertEqual(learner.training_count, 0)
        self.assertFalse(learner.early_stopped)
        self.assertIsNone(learner.current_prompt)
        self.assertIsNone(learner.current_suffix)
        self.assertFalse(hasattr(learner, "_checkpoint_queries_used"))
        reporting = learner.experiment_reporting.to_dict()
        self.assertEqual(reporting["victim_queries_total"], 0)
        self.assertEqual(reporting["feedback_model_calls_total"], 0)
        self.assertEqual(reporting["cycles"], [])

    def _run_main(self, cfg, learner, logdir, controller_side_effect):
        user_task = SimpleNamespace(ID="user-task", PROMPT="user prompt")
        injection_task = object()
        pipeline = SimpleNamespace(name="tiny-pipeline")
        attacker = object()

        with (
            patch.object(adaptive_agentdojo, "set_seed") as set_seed,
            patch.object(
                adaptive_agentdojo, "get_suite", return_value=object()
            ) as get_suite,
            patch.object(
                adaptive_agentdojo,
                "get_user_tasks",
                return_value=[user_task],
            ),
            patch.object(
                adaptive_agentdojo,
                "get_wrapped_injection_tasks",
                return_value={"injection-task": injection_task},
            ),
            patch.object(
                adaptive_agentdojo,
                "setup_pipeline_and_attacker",
                return_value=(pipeline, attacker),
            ),
            patch.object(
                adaptive_agentdojo,
                "get_attack_learner",
                return_value=learner,
            ) as get_attack_learner,
            patch.object(
                adaptive_agentdojo,
                "_run_benchmarks",
                side_effect=controller_side_effect,
            ) as run_controller,
            patch.object(
                adaptive_agentdojo,
                "_find_latest_checkpoint",
                wraps=adaptive_agentdojo._find_latest_checkpoint,
            ) as find_latest_checkpoint,
            patch.object(
                adaptive_agentdojo.os,
                "getcwd",
                return_value=str(logdir),
            ),
        ):
            adaptive_agentdojo.main.__wrapped__(cfg)

        set_seed.assert_called_once_with(1)
        get_suite.assert_called_once_with("v1.2", "test-suite")
        get_attack_learner.assert_called_once()
        run_controller.assert_called_once()
        return find_latest_checkpoint

    def test_explicit_cold_runs_controller_fresh_and_persists_provenance(self):
        learner = self._learner()
        events = []

        def controller(**kwargs):
            events.append("controller")
            self._assert_fresh_at_controller(learner)
            return ({("user-task", "injection-task"): 1.0}, {})

        learner.modify_tasks.side_effect = lambda **kwargs: events.append(
            "modify_tasks"
        )

        with TemporaryDirectory() as directory:
            logdir = Path(directory)
            (logdir / "checkpoint_state.json").write_text("not valid json")
            find_latest_checkpoint = self._run_main(
                self._config(), learner, logdir, controller
            )
            checkpoint_state = json.loads(
                (logdir / "checkpoint_state.json").read_text()
            )

        learner.load_policy_weights_only.assert_not_called()
        learner.load_model.assert_not_called()
        find_latest_checkpoint.assert_not_called()
        self.assertEqual(events, ["modify_tasks", "controller"])
        self.assertEqual(
            checkpoint_state["checkpoint_initialization"],
            {"mode": "cold", "source_checkpoint_path": None},
        )
        self.assertEqual(
            checkpoint_state["learner_state"],
            {
                "experience_count": 0,
                "training_count": 0,
                "best_suffix": None,
                "best_reward": -float("inf"),
                "early_stopped": False,
                "queries_used": 1,
            },
        )

    def test_explicit_policy_loads_before_controller_and_persists_provenance(self):
        learner = self._learner()
        events = []
        source_state = {
            "best_suffix": "source suffix",
            "best_reward": 99.0,
            "experience_count": 8,
            "training_count": 5,
            "queries_used": 42,
            "early_stopped": True,
            "reporting": {"victim_queries_total": 42},
        }

        with TemporaryDirectory() as directory:
            logdir = Path(directory)
            source_path = logdir / "source.pt"
            source_path.write_bytes(b"non-empty checkpoint")
            (logdir / "checkpoint_state.json").write_text(
                json.dumps({"learner_state": source_state})
            )

            def load_policy(path):
                events.append("policy-loaded")
                learner.policy_initialization_provenance = {
                    "source_checkpoint_path": str(path),
                    "policy_checkpoint_format_version": "1.0",
                    "attack_model_name": "tiny-attack-model",
                    "policy_class": "tests.TinyPolicy",
                    "model_type": "tiny-policy",
                    "dtype": "torch.float32",
                }

            def controller(**kwargs):
                events.append("controller")
                self._assert_fresh_at_controller(learner)
                for field in (
                    "best_suffix",
                    "best_reward",
                    "experience_count",
                    "training_count",
                    "early_stopped",
                ):
                    self.assertNotEqual(getattr(learner, field), source_state[field])
                return ({("user-task", "injection-task"): 1.0}, {})

            learner.load_policy_weights_only.side_effect = load_policy
            learner.modify_tasks.side_effect = lambda **kwargs: events.append(
                "modify_tasks"
            )
            find_latest_checkpoint = self._run_main(
                self._config("policy", str(source_path)),
                learner,
                logdir,
                controller,
            )
            checkpoint_state = json.loads(
                (logdir / "checkpoint_state.json").read_text()
            )

        learner.load_policy_weights_only.assert_called_once_with(str(source_path))
        learner.load_model.assert_not_called()
        find_latest_checkpoint.assert_not_called()
        self.assertEqual(
            events, ["policy-loaded", "modify_tasks", "controller"]
        )
        self.assertEqual(
            checkpoint_state["checkpoint_initialization"],
            {
                "mode": "policy",
                "source_checkpoint_path": str(source_path.resolve()),
                "policy_checkpoint_format_version": "1.0",
                "attack_model_name": "tiny-attack-model",
                "policy_class": "tests.TinyPolicy",
                "model_type": "tiny-policy",
                "dtype": "torch.float32",
            },
        )
        self.assertEqual(checkpoint_state["learner_state"]["best_suffix"], None)
        self.assertEqual(
            checkpoint_state["learner_state"]["best_reward"], -float("inf")
        )
        self.assertEqual(checkpoint_state["learner_state"]["experience_count"], 0)
        self.assertEqual(checkpoint_state["learner_state"]["training_count"], 0)
        self.assertFalse(checkpoint_state["learner_state"]["early_stopped"])
        self.assertEqual(checkpoint_state["learner_state"]["queries_used"], 1)
        self.assertNotEqual(
            checkpoint_state["experiment_reporting"], source_state["reporting"]
        )

    def test_failed_policy_load_stops_before_controller_or_checkpoint_save(self):
        learner = self._learner()
        learner.load_policy_weights_only.side_effect = RuntimeError("load failed")

        with TemporaryDirectory() as directory:
            logdir = Path(directory)
            source_path = logdir / "source.pt"
            source_path.write_bytes(b"non-empty checkpoint")
            controller = MagicMock()

            with self.assertRaisesRegex(RuntimeError, "load failed"):
                self._run_main(
                    self._config("policy", str(source_path)),
                    learner,
                    logdir,
                    controller,
                )

            self.assertFalse((logdir / "checkpoint_state.json").exists())
            self.assertFalse((logdir / "checkpoint.pt").exists())

        controller.assert_not_called()
        learner.load_model.assert_not_called()
        self.assertFalse(
            hasattr(learner, "checkpoint_initialization_provenance")
        )

    def test_absent_initialization_fields_retain_legacy_resume(self):
        learner = self._learner()
        cfg = self._config()
        del cfg.initialization_mode
        del cfg.source_checkpoint_path

        with TemporaryDirectory() as directory:
            logdir = Path(directory)
            sidecar = logdir / "checkpoint_state.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "checkpoint_version": "1.0",
                        "learner_state": {"queries_used": 0},
                    }
                )
            )

            def controller(**kwargs):
                self._assert_fresh_at_controller(learner)
                return ({("user-task", "injection-task"): 1.0}, {})

            find_latest_checkpoint = self._run_main(
                cfg, learner, logdir, controller
            )

        find_latest_checkpoint.assert_called_once_with(logdir)
        learner.load_model.assert_called_once_with(str(sidecar))
        learner.load_policy_weights_only.assert_not_called()
        self.assertFalse(
            hasattr(learner, "checkpoint_initialization_provenance")
        )


if __name__ == "__main__":
    unittest.main()
