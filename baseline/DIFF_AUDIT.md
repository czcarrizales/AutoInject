# Semantic diff audit

Comparison base: `c4eedbdd9822cfdaaa4ba229d88d31beccf84bb7`.

The five approved production files have one semantic difference from the
frozen source: explicit two-GPU placement. Tests, manifests, and documentation
do not alter runtime algorithm semantics.

| Production file | Changed behavior | Line classification |
|---|---|---|
| `agentdojo/src/agentdojo/agent_pipeline/agent_pipeline.py` | Load the victim on the requested device instead of automatic placement. | required explicit placement |
| `src/rlpi/agentdojo/adaptive_agentdojo.py` | Read and forward `victim_device`. | required device propagation |
| `src/rlpi/agentdojo/config/config.yaml` | Add the disabled-by-default `victim_device: null` key. | required device propagation |
| `src/rlpi/agentdojo/utils.py` | Accept the device and construct `LocalLLM` only when it is provided. | required device propagation |
| `src/rlpi/attack/learners/trl_suffix/learner.py` | Put policy on the requested device; bind Accelerate/Trainer/current CUDA device to it; suppress automatic two-device DataParallel while both GPUs remain visible. | required explicit placement; required Trainer device pinning |
| same learner assertions | Verify policy/reference/Trainer/Accelerator/input placement and absence of DataParallel. | read-only assertion |

No changed production line alters the objective, sampling, rewards, feedback,
optimizer construction or kwargs, query accounting, stopping, suffix creation,
checkpoint format, retry/parsing behavior, model dtype, or hyperparameter
defaults. In particular, `_n_gpu = 1` changes only Transformers' automatic
wrapper choice; it does not hide `cuda:0` from the process.

The accepted adaptation outside these five files is the existing qwen3-small
feedback request option test/behavior (`enable_thinking=False`) already reviewed
against the frozen baseline. The baseline manifest supplies approved runtime
parameters; it contains no production-proof monkeypatch or reduced bound.
