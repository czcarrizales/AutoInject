import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from accelerate.state import AcceleratorState

from agentdojo.agent_pipeline.agent_pipeline import LocalLLM
from rlpi.agentdojo import utils as agentdojo_utils
from rlpi.attack.learners.trl_suffix.learner import TRLSuffixLearner
from rlpi.attack.learners.trl_suffix.utils import create_grpo_config


class TwoGpuPlacementTests(unittest.TestCase):
    def test_local_victim_uses_whole_model_device_map(self):
        model = MagicMock()
        model.parameters.return_value = []
        tokenizer = SimpleNamespace(
            pad_token="pad",
            eos_token="eos",
        )

        with (
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=model,
            ) as load_victim,
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ),
        ):
            LocalLLM(
                "facebook/Meta-SecAlign-8B",
                device="cuda:0",
            )

        self.assertEqual(
            load_victim.call_args.kwargs["device_map"],
            {"": "cuda:0"},
        )
        self.assertIs(
            load_victim.call_args.kwargs["dtype"],
            torch.float16,
        )

    def test_victim_device_is_passed_to_local_llm(self):
        local_llm = MagicMock(spec=agentdojo_utils.LocalLLM)
        model = SimpleNamespace(value="facebook/Meta-SecAlign-8B")

        with (
            patch.object(agentdojo_utils, "LocalLLM", return_value=local_llm) as local,
            patch.object(
                agentdojo_utils.AgentPipeline,
                "from_config",
                return_value=MagicMock(),
            ),
            patch.object(agentdojo_utils, "load_attack", return_value=MagicMock()),
        ):
            agentdojo_utils.setup_pipeline_and_attacker(
                model=model,
                defense=None,
                system_message=None,
                attack="direct",
                suite=MagicMock(),
                victim_device="cuda:0",
            )

        local.assert_called_once_with(
            "facebook/Meta-SecAlign-8B", device="cuda:0"
        )

    def test_policy_uses_whole_model_learner_device_map(self):
        learner = TRLSuffixLearner.__new__(TRLSuffixLearner)
        learner.attack_model_name = "Qwen/Qwen2-1.5B"
        learner.device = "cuda:1"
        learner.seed = 42
        policy = MagicMock()
        policy.parameters.return_value = []
        policy.generation_config = SimpleNamespace(
            pad_token_id=0,
            eos_token_id=0,
        )
        tokenizer = SimpleNamespace(
            pad_token="pad",
            eos_token="eos",
            pad_token_id=0,
            eos_token_id=0,
        )

        with (
            patch(
                "rlpi.attack.learners.trl_suffix.learner."
                "AutoModelForCausalLM.from_pretrained",
                return_value=policy,
            ) as load_policy,
            patch(
                "rlpi.attack.learners.trl_suffix.learner."
                "AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ),
            patch(
                "rlpi.attack.learners.trl_suffix.learner.set_seed"
            ),
        ):
            learner._init_model_and_tokenizer()

        self.assertEqual(
            load_policy.call_args.kwargs["device_map"],
            {"": "cuda:1"},
        )
        self.assertIs(
            load_policy.call_args.kwargs["torch_dtype"],
            torch.float16,
        )

    def test_trainer_and_reference_target_learner_device(self):
        learner = TRLSuffixLearner.__new__(TRLSuffixLearner)
        learner.device = "cuda:1"
        learner.policy = MagicMock()
        learner.policy.parameters.side_effect = lambda: iter(
            [SimpleNamespace(device=torch.device("cuda:1"))]
        )
        learner.tokenizer = MagicMock()
        learner.log_dir = "/tmp"
        learner.grpo_per_device_train_batch_size = 2
        learner.grpo_learning_rate = 1.0e-5
        learner.grpo_num_generations = 2
        learner.grpo_num_iterations = 1
        learner.grpo_gradient_accumulation_steps = 2
        learner.grpo_warmup_steps = 4
        learner.max_suffix_length = 10
        learner.grpo_max_grad_norm = 0.1
        learner.grpo_beta = 1.0
        learner.grpo_adam_epsilon = 1.0e-5
        learner.grpo_adam_beta2 = 0.98
        learner.verbose = False
        dataset = MagicMock()
        dataset.__len__.return_value = 1
        grpo_config = MagicMock()
        grpo_config.distributed_state.device = torch.device("cuda:0")
        trainer = MagicMock()
        trainer.model.parameters.return_value = [
            SimpleNamespace(device=torch.device("cuda:1"))
        ]
        trainer.ref_model.parameters.return_value = [
            SimpleNamespace(device=torch.device("cuda:1"))
        ]
        trainer.args.device = torch.device("cuda:1")
        trainer.args.n_gpu = 1
        trainer.accelerator.device = torch.device("cuda:1")
        trainer._prepare_input.return_value = SimpleNamespace(
            device=torch.device("cuda:1")
        )

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "empty_cache"),
            patch.object(torch.cuda, "set_device") as set_device,
            patch(
                "rlpi.attack.learners.trl_suffix.learner.create_grpo_config",
                return_value=grpo_config,
            ),
            patch(
                "rlpi.attack.learners.trl_suffix.learner.GRPOTrainer",
                return_value=trainer,
            ),
        ):
            learner._create_grpo_trainer_with_reward_function(
                MagicMock(),
                dataset,
                max_steps=1,
            )
            self.assertEqual(
                os.environ["ACCELERATE_TORCH_DEVICE"],
                "cuda:1",
            )

        set_device.assert_called_once_with("cuda:1")
        self.assertEqual(
            grpo_config._setup_devices,
            torch.device("cuda:1"),
        )
        self.assertEqual(
            grpo_config.distributed_state.device,
            torch.device("cuda:1"),
        )
        self.assertEqual(grpo_config._n_gpu, 1)

    def test_non_default_trainer_device_is_not_cuda_zero(self):
        AcceleratorState._reset_state(reset_partial_state=True)
        try:
            with (
                patch.dict(
                    os.environ,
                    {"ACCELERATE_TORCH_DEVICE": "cuda:1"},
                    clear=False,
                ),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "device_count", return_value=2),
                patch.object(torch.cuda, "set_device"),
            ):
                config = create_grpo_config(
                    output_dir="/tmp/grpo-device-regression",
                    grpo_per_device_train_batch_size=2,
                    grpo_learning_rate=1.0e-5,
                    grpo_num_generations=2,
                    grpo_num_iterations=1,
                    grpo_gradient_accumulation_steps=2,
                    grpo_warmup_steps=4,
                    max_suffix_length=10,
                    max_steps=1,
                    beta=1.0,
                )

            self.assertEqual(config.device, torch.device("cuda:0"))
            self.assertEqual(config.n_gpu, 2)
            config._setup_devices = torch.device("cuda:1")
            config.distributed_state.device = torch.device("cuda:1")
            config._n_gpu = 1
            self.assertEqual(config.device, torch.device("cuda:1"))
            self.assertEqual(config.n_gpu, 1)
            self.assertEqual(
                config.distributed_state.device,
                torch.device("cuda:1"),
            )
        finally:
            AcceleratorState._reset_state(reset_partial_state=True)


if __name__ == "__main__":
    unittest.main()
