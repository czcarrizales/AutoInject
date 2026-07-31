# AutoInject Qwen baseline freeze

Status: materialized for review; not yet committed, pushed, or launched.

The immutable baseline identifier will be the single Git commit created after
this review.  An official run is permitted only when the local checkout is
clean, its `HEAD` equals the fetched remote branch tip, and that commit contains
this manifest, the approved placement-only source patch, its focused tests, and
the documents in this directory.

## Scientific freeze

- Frozen source parent: `c4eedbdd9822cfdaaa4ba229d88d31beccf84bb7`
- Approved placement-patch SHA-256:
  `1a5c6d59c5fde4e36538d6fa2ac3e281b6573373495ecddc8b03b26685a0eb1b`
- Container: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
- Runtime: Python 3.11.10, torch 2.5.1+cu124, CUDA build 12.4,
  Transformers 4.57.0, Accelerate 1.14.0, TRL 0.24.0, PEFT 0.20.0,
  Hydra 1.3.4.
- Victim: `facebook/Meta-SecAlign-8B` on `cuda:0`.
- Policy/reference/Trainer/optimizer: `Qwen/Qwen2-1.5B` on `cuda:1`.
- Feedback: `qwen3-small` through the approved OpenAI-compatible endpoint;
  `enable_thinking=False` is preserved by the existing request path.
- Run 01: `banking`, `user_task_14`, `injection_task_5`; experiment seed 1,
  learner seed 42; query budget 260.

The command in `BASELINE_MANIFEST.yaml` is the normative resolved invocation.
No checkpoint, learner state, suffix state, Trainer state, optimizer state, or
proof harness is admitted at startup. Read-only model-download caches may be
reused; policy weights are initialized with `from_pretrained`.

## Freeze procedure

The reviewer should first run every static check in `RUNBOOK.md`. Then create
one commit containing exactly the reviewed files listed there. After pushing,
record the resulting commit in the run provenance. Do not edit the manifest or
source after the commit; any change requires a new reviewed freeze commit.
