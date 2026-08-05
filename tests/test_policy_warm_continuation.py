import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from datasets import Dataset
from omegaconf import OmegaConf
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import GRPOConfig, GRPOTrainer

from rlpi.agentdojo import adaptive_agentdojo
from rlpi.attack.learners.trl_suffix.learner import TRLSuffixLearner


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    @property
    def dtype(self):
        return self.linear.weight.dtype


def checkpoint_for(policy):
    return {
        "policy_state_dict": copy.deepcopy(policy.state_dict()),
        "model_config": {
            "attack_model_name": "tiny-policy",
            "dtype": "torch.float32",
        },
    }


class PolicyWarmContinuationTests(unittest.TestCase):
    def learner(self):
        learner = TRLSuffixLearner.__new__(TRLSuffixLearner)
        learner.attack_model_name = "tiny-policy"
        learner.policy = TinyPolicy()
        learner.experience_count = learner.training_count = 0
        learner.best_suffix = learner.current_prompt = learner.current_suffix = None
        learner.best_reward = -float("inf")
        learner.early_stopped = False
        return learner

    def write_checkpoint(self, checkpoint):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "checkpoint.pt"
        torch.save(checkpoint, path)
        return path

    def test_warm_load_creates_equal_separate_frozen_reference_and_fresh_state(self):
        source = TinyPolicy()
        with torch.no_grad():
            source.linear.weight.fill_(3.0)
        learner = self.learner()
        learner.load_policy_warm_start(self.write_checkpoint(checkpoint_for(source)))

        for policy, reference in zip(
            learner.policy.state_dict().values(),
            learner._warm_reference.state_dict().values(),
        ):
            self.assertTrue(torch.equal(policy, reference))
            self.assertNotEqual(policy.data_ptr(), reference.data_ptr())
        self.assertFalse(learner._warm_reference.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in learner._warm_reference.parameters())
        )
        with torch.no_grad():
            learner.policy.linear.weight.add_(1)
        self.assertFalse(
            torch.equal(learner.policy.linear.weight, learner._warm_reference.linear.weight)
        )
        self.assertEqual(learner.experience_count, 0)
        self.assertEqual(learner.training_count, 0)
        self.assertIsNone(learner.best_suffix)
        self.assertEqual(learner.best_reward, -float("inf"))
        self.assertFalse(learner.early_stopped)

    def test_incompatible_checkpoint_fails_before_policy_mutation(self):
        learner = self.learner()
        checkpoint = checkpoint_for(TinyPolicy())
        checkpoint["policy_state_dict"]["linear.weight"] = torch.empty(1)
        before = copy.deepcopy(learner.policy.state_dict())

        with self.assertRaisesRegex(ValueError, "incompatible"):
            learner.load_policy_warm_start(self.write_checkpoint(checkpoint))

        self.assertTrue(
            all(torch.equal(tensor, before[name]) for name, tensor in learner.policy.state_dict().items())
        )
        self.assertFalse(hasattr(learner, "_warm_reference"))

    def test_cold_config_is_valid_and_incomplete_warm_config_is_rejected(self):
        cold = OmegaConf.create(
            {"initialization_mode": "cold", "source_checkpoint_path": None, "reference_mode": "base"}
        )
        adaptive_agentdojo.validate_continuation_config(cold)
        warm = OmegaConf.create(
            {"initialization_mode": "policy_warm", "source_checkpoint_path": None, "reference_mode": "source"}
        )
        with self.assertRaisesRegex(ValueError, "source_checkpoint_path"):
            adaptive_agentdojo.validate_continuation_config(warm)


class ExplicitReferencePatchTests(unittest.TestCase):
    def tiny_model(self):
        config = GPT2Config(
            vocab_size=4, n_positions=16, n_embd=8, n_layer=1, n_head=1,
            bos_token_id=1, eos_token_id=1, pad_token_id=0,
        )
        config._name_or_path = "tiny-base"
        return GPT2LMHeadModel(config)

    def trainer(self, model, **kwargs):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer(), pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
        )
        config = GRPOConfig(
            output_dir=directory.name, per_device_train_batch_size=1, num_generations=2,
            generation_batch_size=2, max_completion_length=2, max_steps=1, beta=1.0,
            report_to=[], gradient_checkpointing=False, use_vllm=False, bf16=False, fp16=False,
        )
        return GRPOTrainer(
            model=model, reward_funcs=lambda completions, **unused: [0.0] * len(completions),
            args=config, train_dataset=Dataset.from_dict({"prompt": ["hello"]}),
            processing_class=tokenizer, **kwargs,
        )

    @staticmethod
    def tokenizer():
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel

        return Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "hello": 2, "<unk>": 3}, "<unk>"))

    def test_explicit_reference_is_used_and_default_path_is_preserved(self):
        reference = self.tiny_model()
        with patch("trl.trainer.grpo_trainer.AutoConfig.from_pretrained") as auto_config:
            explicit = self.trainer(self.tiny_model(), ref_model=reference)
        auto_config.assert_not_called()
        self.assertIs(explicit.ref_model, reference)

        import trl.trainer.grpo_trainer as grpo_module

        fallback_reference = self.tiny_model()
        architecture = SimpleNamespace(from_pretrained=MagicMock(return_value=fallback_reference))
        with (
            patch(
                "trl.trainer.grpo_trainer.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(architectures=["TinyReference"]),
            ) as auto_config,
            patch.object(grpo_module.transformers, "TinyReference", architecture, create=True),
        ):
            fallback = self.trainer(self.tiny_model())
        auto_config.assert_called_once_with("tiny-base")
        architecture.from_pretrained.assert_called_once()
        self.assertIs(fallback.ref_model, fallback_reference)


if __name__ == "__main__":
    unittest.main()
