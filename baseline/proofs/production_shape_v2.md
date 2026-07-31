# Production-shape proof v2

The infrastructure proof completed successfully on 2026-07-31. It was an
infrastructure test, not a cold or warm scientific result.

## Recovered control-plane evidence

- Job: `autoinject-qwen3-small-two-v100-production-shape-proof-v2`
- Pod: `autoinject-qwen3-small-two-v100-production-shape-proof-v2-z7jx4`
- Namespace: `ucr-rai-chazz`
- Job start: `2026-07-31T16:37:11Z`
- Job completion: `2026-07-31T17:14:54Z`
- Pod scheduled: `2026-07-31T16:37:17Z`
- Main container: `2026-07-31T16:37:28Z` to `2026-07-31T17:14:50Z`, exit 0
- Init container: exit 0
- Frozen source: `c4eedbdd9822cfdaaa4ba229d88d31beccf84bb7`
- Placement patch: `1a5c6d59c5fde4e36538d6fa2ac3e281b6573373495ecddc8b03b26685a0eb1b`

The retained main log reports `two_gpu_preflight: PASS`, 16 generated
completions/evaluations, one optimizer update, checkpoint creation,
`production_shape_training: PASS`, `fresh_checkpoint_reload: PASS`, and
`gpu_memory_validation: PASS`. The proof exercised two gradient-accumulation
microsteps and one update at the production generation, batch, iteration, and
suffix-length shape.

## Evidence availability

The Job and Pod objects and both container logs were available through the
Kubernetes API during materialization. Historical Kubernetes events had
expired. The PVC is not mounted in this checkout, and no new reader Pod was
created, so the small JSON/CSV files could not be copied without launching a
new Kubernetes resource. Their known persistent root is:

`/workspace/results/autoinject-qwen3-small/two-v100-production-shape-proof-v2/qwen3-small-two-v100-20260731T163728Z-389e3537-1755-4b40-af90-15f1f3b70395`

See `production_shape_v2_evidence_index.json` for retained hashes and the
unavailable-file inventory. This limitation is historical-evidence retention,
not a scientific blocker.
