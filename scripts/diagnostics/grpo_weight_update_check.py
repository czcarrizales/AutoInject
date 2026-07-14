"""Check whether TRL GRPO can update Qwen weights when rewards vary.

This is a diagnostic test, not an AutoInject experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


MODEL_NAME = "Qwen/Qwen2-0.5B"
OUTPUT_DIR = Path("/workspace/results/grpo-weight-update-check")


def alternating_reward(completions: list[str], **kwargs) -> list[float]:
    """Guarantee reward variation inside each two-completion group."""
    del kwargs

    rewards = [float(index % 2) for index in range(len(completions))]

    print("\n=== SYNTHETIC REWARDS ===")
    for index, (completion, reward) in enumerate(zip(completions, rewards)):
        preview = completion.replace("\n", "\\n")[:100]
        print(f"{index}: reward={reward:.1f}, completion={preview!r}")

    return rewards


def get_trainable_parameter(
    trainer: GRPOTrainer,
) -> tuple[str, torch.nn.Parameter]:
    """Return a stable trainable matrix parameter from the unwrapped model."""
    model = trainer.accelerator.unwrap_model(trainer.model)

    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.ndim >= 2:
            return name, parameter

    raise RuntimeError("No trainable matrix parameter was found.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== SOFTWARE ===")
    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a CUDA GPU.")

    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Two prompts and two generations produce four completions.
    # Rewards are returned as 0, 1, 0, 1, providing variation in each pair.
    dataset = Dataset.from_dict(
        {
            "prompt": [
                "Generate a short nonsensical adversarial suffix:",
                "Generate a short nonsensical adversarial suffix:",
            ]
        }
    )

    config = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        model_init_kwargs={"dtype": torch.float32},
        per_device_train_batch_size=2,
        num_generations=2,
        generation_batch_size=4,
        max_prompt_length=64,
        max_completion_length=8,
        max_steps=1,
        learning_rate=1e-4,
        beta=0.0,
        gradient_checkpointing=False,
        fp16=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=42,
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[alternating_reward],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    parameter_name, parameter = get_trainable_parameter(trainer)

    before = (
        parameter.detach()
        .float()
        .reshape(-1)[:4096]
        .cpu()
        .clone()
    )

    trainable_count = sum(
        item.numel()
        for item in trainer.accelerator.unwrap_model(
            trainer.model
        ).parameters()
        if item.requires_grad
    )

    print("\n=== BEFORE TRAINING ===")
    print("Sampled parameter:", parameter_name)
    print("Trainable parameters:", trainable_count)
    print("Sample mean:", before.mean().item())
    print("Sample norm:", before.norm().item())

    result = trainer.train()

    _, parameter_after = get_trainable_parameter(trainer)
    after = (
        parameter_after.detach()
        .float()
        .reshape(-1)[:4096]
        .cpu()
        .clone()
    )

    difference = (after - before).abs()
    changed_values = int(torch.count_nonzero(difference).item())
    max_change = float(difference.max().item())
    mean_change = float(difference.mean().item())
    weights_changed = changed_values > 0

    report = {
        "model": MODEL_NAME,
        "sampled_parameter": parameter_name,
        "trainable_parameters": trainable_count,
        "sample_values_checked": before.numel(),
        "changed_values": changed_values,
        "max_absolute_change": max_change,
        "mean_absolute_change": mean_change,
        "weights_changed": weights_changed,
        "training_metrics": result.metrics,
    }

    report_path = OUTPUT_DIR / "weight_update_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n=== AFTER TRAINING ===")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to: {report_path}")

    if not weights_changed:
        raise RuntimeError(
            "Rewards varied, but the sampled model weights did not change."
        )


if __name__ == "__main__":
    main()
