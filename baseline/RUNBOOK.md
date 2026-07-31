# Official cold-baseline runbook

This runbook is intentionally fail-closed. Do not launch until the reviewed
freeze commit exists on `experiments/qwen3-small-baseline-v1` and the successful
bootstrap-v3 environment report remains on the PVC.

## 1. Freeze and static gate

From the repository root, run:

```bash
git status --porcelain=v1
git fetch --no-tags origin experiments/qwen3-small-baseline-v1
test "$(git rev-parse HEAD)" = \
  "$(git rev-parse origin/experiments/qwen3-small-baseline-v1)"
test -z "$(git status --porcelain=v1)"
git show HEAD:baseline/BASELINE_MANIFEST.yaml >/dev/null
bash -n baseline/watch_k8s_timing.sh
PYTHONPATH="$PWD/src:$PWD/agentdojo/src" \
  PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest -v \
    tests.test_qwen3_small_feedback_options \
    tests.test_two_gpu_placement
kubectl apply --dry-run=client -f baseline/BASELINE_MANIFEST.yaml
```

The empty `git status` and local/remote commit equality are mandatory. Record
`git rev-parse HEAD`; it is the baseline freeze commit. The Job independently
fetches that branch tip, verifies the frozen parent and patch, and archives the
manifest and Git hashes in its result provenance.

Files intended for the single freeze commit are:

```text
agentdojo/src/agentdojo/agent_pipeline/agent_pipeline.py
src/rlpi/agentdojo/adaptive_agentdojo.py
src/rlpi/agentdojo/config/config.yaml
src/rlpi/agentdojo/utils.py
src/rlpi/attack/learners/trl_suffix/learner.py
tests/test_qwen3_small_feedback_options.py
tests/test_two_gpu_placement.py
baseline/BASELINE_FREEZE.md
baseline/BASELINE_MANIFEST.yaml
baseline/DIFF_AUDIT.md
baseline/KNOWN_ORIGINAL_QUIRKS.md
baseline/RUNBOOK.md
baseline/watch_k8s_timing.sh
baseline/proofs/production_shape_v2.md
baseline/proofs/production_shape_v2_evidence_index.json
baseline/proofs/production_shape_v2_init_container.log
```

Do not include historical V1-V6 manifests/evidence or unrelated working-tree
changes in this commit. Proposed commit message:

```text
Freeze two-V100 Qwen cold baseline
```

## 2. Start the passive timing collector

The watcher uses only local `kubectl` read calls; the experiment Pod receives no
service-account token or Kubernetes API permission. Start it immediately before
submission. The evidence directory is unique and outside the Git checkout:

```bash
NAMESPACE=ucr-rai-chazz
JOB=autoinject-qwen3-small-two-v100-cold-baseline-v1-r01
EVIDENCE_DIR=/home/chazz/research/ucr-rai/results/autoinject-qwen3-small/cold-baseline-v1/run-01-banking-u14-i5/provenance/kubernetes
LAUNCH_START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
mkdir -p "${EVIDENCE_DIR}"
printf '%s\n' "${LAUNCH_START_UTC}" > "${EVIDENCE_DIR}/launch-start-utc.txt"
nohup baseline/watch_k8s_timing.sh \
  --namespace "${NAMESPACE}" \
  --job "${JOB}" \
  --evidence-dir "${EVIDENCE_DIR}" \
  --launch-start-utc "${LAUNCH_START_UTC}" \
  --timeout-seconds 604800 \
  > "${EVIDENCE_DIR}/watcher-supervisor.log" 2>&1 &
WATCHER_PID=$!
printf '%s\n' "${WATCHER_PID}" > "${EVIDENCE_DIR}/watcher.pid"
```

The collector polls Job and label-selected Pod state every five seconds and
captures events during execution. It writes:

```text
kubernetes-timeline.jsonl
kubernetes-events.txt
job-final.yaml
pod-final.yaml
timing-summary.json
timing-summary.txt
```

It also keeps `timing-watcher.log`. It never deletes or mutates Kubernetes
objects and retains partial records on Job failure or watcher timeout.

## 3. Submit and keep the watcher active

Run exactly this command only after the freeze gate and watcher startup pass:

```bash
kubectl apply -f baseline/BASELINE_MANIFEST.yaml
```

Then monitor without changing the workload:

```bash
kubectl -n ucr-rai-chazz get pods \
  -l job-name=autoinject-qwen3-small-two-v100-cold-baseline-v1-r01 \
  -o wide --watch
kubectl -n ucr-rai-chazz logs \
  -l job-name=autoinject-qwen3-small-two-v100-cold-baseline-v1-r01 \
  -c autoinject-qwen3-small --timestamps --follow
wait "${WATCHER_PID}"
```

The wrapper emits `BASELINE_WRAPPER_START_UTC`,
`AUTOINJECT_COMMAND_START_UTC`, `AUTOINJECT_COMMAND_END_UTC`,
`BASELINE_WRAPPER_END_UTC`, and `AUTOINJECT_EXIT_CODE`. The application is run
inside an `if` so strict Bash mode does not hide its status; the wrapper exits
with the original AutoInject exit code.

## 4. Recover after terminal disconnection

Because the watcher is started with `nohup`, reconnect and inspect it with:

```bash
EVIDENCE_DIR=/home/chazz/research/ucr-rai/results/autoinject-qwen3-small/cold-baseline-v1/run-01-banking-u14-i5/provenance/kubernetes
ps -p "$(cat "${EVIDENCE_DIR}/watcher.pid")" -o pid=,stat=,etime=,command=
tail -n 100 "${EVIDENCE_DIR}/timing-watcher.log"
kubectl -n ucr-rai-chazz get job \
  autoinject-qwen3-small-two-v100-cold-baseline-v1-r01 -o wide
```

If the watcher itself stopped, do not overwrite its partial files. Start a new
collector in a timestamped recovery subdirectory, using the saved original
launch timestamp:

```bash
RECOVERY_DIR="${EVIDENCE_DIR}/recovery-$(date -u +%Y%m%dT%H%M%SZ)"
baseline/watch_k8s_timing.sh \
  --namespace ucr-rai-chazz \
  --job autoinject-qwen3-small-two-v100-cold-baseline-v1-r01 \
  --evidence-dir "${RECOVERY_DIR}" \
  --launch-start-utc "$(cat "${EVIDENCE_DIR}/launch-start-utc.txt")"
```

Recovery can preserve current/final objects and logs, but it cannot recreate
missed five-second observations. Kubernetes events have finite retention and
may expire, which is why they must be captured while the Job is active.

## Timing interpretation

Kubernetes exposes exact object creation, condition-transition, and container
start/finish timestamps. Pod `Running`/terminal phase points in the summary are
first five-second observations because phases do not expose a separate exact
transition timestamp. Kubernetes also does not expose one authoritative
timestamp for every image-pull, mount, init, and container-creation boundary;
the watcher therefore reports scheduling wait separately and labels the
post-scheduling interval as combined startup overhead, with captured events as
supporting detail. Victim-load and first-GRPO points are populated only when a
naturally occurring log message identifies them; unavailable is preferable to
an inferred timestamp.
