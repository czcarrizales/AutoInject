#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  watch_k8s_timing.sh \
    --namespace NAMESPACE \
    --job JOB_NAME \
    --evidence-dir DIRECTORY \
    --launch-start-utc YYYY-MM-DDTHH:MM:SSZ \
    [--timeout-seconds SECONDS]

Start this watcher immediately before submitting the Job. It is read-only with
respect to Kubernetes and never deletes or modifies a Job or Pod.
EOF
}

NAMESPACE=""
JOB_NAME=""
EVIDENCE_DIR=""
LAUNCH_START_UTC=""
TIMEOUT_SECONDS=604800
POLL_SECONDS=5

while (( $# > 0 )); do
  case "$1" in
    --namespace)
      NAMESPACE="${2:?missing namespace}"
      shift 2
      ;;
    --job)
      JOB_NAME="${2:?missing Job name}"
      shift 2
      ;;
    --evidence-dir)
      EVIDENCE_DIR="${2:?missing evidence directory}"
      shift 2
      ;;
    --launch-start-utc)
      LAUNCH_START_UTC="${2:?missing launch timestamp}"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="${2:?missing timeout}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

for value_name in NAMESPACE JOB_NAME EVIDENCE_DIR LAUNCH_START_UTC; do
  if [[ -z "${!value_name}" ]]; then
    printf 'Required argument is empty: %s\n' "${value_name}" >&2
    usage >&2
    exit 64
  fi
done
if [[ ! "${TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Invalid --timeout-seconds: %s\n' "${TIMEOUT_SECONDS}" >&2
  exit 64
fi

for utility in kubectl python3 date mktemp mkdir mv rm rmdir sleep tee touch; do
  command -v "${utility}" >/dev/null
done
python3 - "${LAUNCH_START_UTC}" <<'PY'
import datetime as dt
import sys

try:
    dt.datetime.strptime(sys.argv[1], "%Y-%m-%dT%H:%M:%SZ")
except ValueError as error:
    raise SystemExit(
        "--launch-start-utc must be a valid UTC timestamp in "
        "YYYY-MM-DDTHH:MM:SSZ form"
    ) from error
PY

mkdir -p "${EVIDENCE_DIR}"
readonly TIMELINE="${EVIDENCE_DIR}/kubernetes-timeline.jsonl"
readonly EVENTS="${EVIDENCE_DIR}/kubernetes-events.txt"
readonly JOB_FINAL="${EVIDENCE_DIR}/job-final.yaml"
readonly POD_FINAL="${EVIDENCE_DIR}/pod-final.yaml"
readonly SUMMARY_JSON="${EVIDENCE_DIR}/timing-summary.json"
readonly SUMMARY_TEXT="${EVIDENCE_DIR}/timing-summary.txt"
readonly WATCHER_LOG="${EVIDENCE_DIR}/timing-watcher.log"

for output in \
  "${TIMELINE}" "${EVENTS}" "${JOB_FINAL}" "${POD_FINAL}" \
  "${SUMMARY_JSON}" "${SUMMARY_TEXT}" "${WATCHER_LOG}"; do
  if [[ -e "${output}" ]]; then
    printf 'Refusing to overwrite timing evidence: %s\n' "${output}" >&2
    exit 65
  fi
done

touch "${TIMELINE}" "${EVENTS}" "${WATCHER_LOG}"
exec > >(tee -a "${WATCHER_LOG}") 2>&1

utc_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

readonly COLLECTOR_START_UTC="$(utc_now)"
readonly START_EPOCH="$(date -u +%s)"
readonly TMP_DIR="$(mktemp -d)"
POD_NAME=""
WATCHER_OUTCOME="running"

cleanup() {
  local status=$?
  set +e
  rm -f \
    "${TMP_DIR}/job.json" "${TMP_DIR}/pods.json" \
    "${TMP_DIR}/pod.json" "${TMP_DIR}/events.json" \
    "${TMP_DIR}/main.log" "${TMP_DIR}/seen-events"
  rmdir "${TMP_DIR}" 2>/dev/null || true
  return "${status}"
}
trap cleanup EXIT

printf 'timing_collector_start_utc=%s\n' "${COLLECTOR_START_UTC}"
printf 'local_launch_command_start_utc=%s\n' "${LAUNCH_START_UTC}"
printf 'namespace=%s job=%s poll_seconds=%s timeout_seconds=%s\n' \
  "${NAMESPACE}" "${JOB_NAME}" "${POLL_SECONDS}" "${TIMEOUT_SECONDS}"

python3 - "${TIMELINE}" "${COLLECTOR_START_UTC}" \
  "${LAUNCH_START_UTC}" "${NAMESPACE}" "${JOB_NAME}" <<'PY'
import json
import sys

path, observed, launch, namespace, job = sys.argv[1:]
with open(path, "a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "record_type": "collector_start",
        "observed_at_utc": observed,
        "local_launch_command_start_utc": launch,
        "namespace": namespace,
        "job_name": job,
    }, sort_keys=True) + "\n")
PY

capture_events() {
  local observed_at="$1"

  # Kubernetes field selectors cannot OR names, so obtain the namespace event
  # set and filter both involved-object names locally.
  if ! kubectl -n "${NAMESPACE}" get events -o json \
      > "${TMP_DIR}/events.json" 2>>"${WATCHER_LOG}"; then
    return 0
  fi
  python3 - "${TMP_DIR}/events.json" "${TMP_DIR}/seen-events" \
    "${EVENTS}" "${observed_at}" "${JOB_NAME}" "${POD_NAME}" <<'PY'
import json
import pathlib
import sys

source, seen_path, output, observed, job_name, pod_name = sys.argv[1:]
data = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
seen_file = pathlib.Path(seen_path)
seen = set(seen_file.read_text(encoding="utf-8").splitlines()) if seen_file.exists() else set()
new_ids = []
with open(output, "a", encoding="utf-8") as stream:
    for event in data.get("items", []):
        involved = event.get("involvedObject", {})
        if involved.get("name") not in {job_name, pod_name}:
            continue
        metadata = event.get("metadata", {})
        uid = metadata.get("uid") or ":".join([
            involved.get("name", ""),
            event.get("reason", ""),
            event.get("firstTimestamp", ""),
            event.get("message", ""),
        ])
        if uid in seen:
            continue
        seen.add(uid)
        new_ids.append(uid)
        event_time = (
            event.get("eventTime")
            or event.get("lastTimestamp")
            or event.get("firstTimestamp")
            or metadata.get("creationTimestamp")
        )
        stream.write(
            f"observed_at_utc={observed}\n"
            f"event_time_utc={event_time}\n"
            f"object={involved.get('kind')}/{involved.get('name')}\n"
            f"type={event.get('type')} reason={event.get('reason')} "
            f"count={event.get('count')}\n"
            f"message={event.get('message')}\n---\n"
        )
if new_ids:
    seen_file.write_text("\n".join(sorted(seen)) + "\n", encoding="utf-8")
PY
}

while true; do
  observed_at="$(utc_now)"
  job_available=false
  pod_available=false

  if kubectl -n "${NAMESPACE}" get job "${JOB_NAME}" -o json \
      > "${TMP_DIR}/job.json" 2>>"${WATCHER_LOG}"; then
    job_available=true
  else
    printf '%s Job not visible yet: %s\n' "${observed_at}" "${JOB_NAME}"
    printf '{}' > "${TMP_DIR}/job.json"
  fi

  if kubectl -n "${NAMESPACE}" get pods \
      -l "job-name=${JOB_NAME}" -o json \
      > "${TMP_DIR}/pods.json" 2>>"${WATCHER_LOG}"; then
    candidate="$({
      python3 - "${TMP_DIR}/pods.json" <<'PY'
import json
import pathlib
import sys

items = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")).get("items", [])
items.sort(key=lambda item: item.get("metadata", {}).get("creationTimestamp", ""))
if items:
    print(items[0].get("metadata", {}).get("name", ""))
PY
    } || true)"
    if [[ -n "${candidate}" ]]; then
      POD_NAME="${candidate}"
    fi
  else
    printf '{"items":[]}' > "${TMP_DIR}/pods.json"
  fi

  if [[ -n "${POD_NAME}" ]] && \
      kubectl -n "${NAMESPACE}" get pod "${POD_NAME}" -o json \
        > "${TMP_DIR}/pod.json" 2>>"${WATCHER_LOG}"; then
    pod_available=true
  else
    printf '{}' > "${TMP_DIR}/pod.json"
  fi

  python3 - "${TIMELINE}" "${observed_at}" "${JOB_NAME}" \
    "${POD_NAME}" "${TMP_DIR}/job.json" "${TMP_DIR}/pod.json" <<'PY'
import json
import pathlib
import sys

timeline, observed, job_name, pod_name, job_path, pod_path = sys.argv[1:]
job = json.loads(pathlib.Path(job_path).read_text(encoding="utf-8"))
pod = json.loads(pathlib.Path(pod_path).read_text(encoding="utf-8"))

def condition_map(resource):
    return {
        item.get("type"): {
            "status": item.get("status"),
            "reason": item.get("reason"),
            "last_transition_time": item.get("lastTransitionTime"),
        }
        for item in resource.get("status", {}).get("conditions", [])
    }

main = None
for status in pod.get("status", {}).get("containerStatuses", []):
    if status.get("name") == "autoinject-qwen3-small":
        main = status
        break

record = {
    "record_type": "poll",
    "observed_at_utc": observed,
    "job_name": job_name,
    "job_exists": bool(job),
    "job_creation_timestamp": job.get("metadata", {}).get("creationTimestamp"),
    "job_start_time": job.get("status", {}).get("startTime"),
    "job_active": job.get("status", {}).get("active", 0),
    "job_succeeded": job.get("status", {}).get("succeeded", 0),
    "job_failed": job.get("status", {}).get("failed", 0),
    "job_conditions": condition_map(job),
    "pod_name": pod_name or None,
    "pod_exists": bool(pod),
    "pod_creation_timestamp": pod.get("metadata", {}).get("creationTimestamp"),
    "pod_phase": pod.get("status", {}).get("phase"),
    "pod_reason": pod.get("status", {}).get("reason"),
    "pod_conditions": condition_map(pod),
    "main_container_state": main.get("state") if main else None,
}
with open(timeline, "a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
PY

  capture_events "${observed_at}"

  if [[ "${job_available}" == true ]]; then
    terminal="$({
      python3 - "${TMP_DIR}/job.json" <<'PY'
import json
import pathlib
import sys

job = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for condition in job.get("status", {}).get("conditions", []):
    if condition.get("status") == "True" and condition.get("type") in {
        "Complete", "Failed"
    }:
        print(condition.get("type"))
        break
PY
    } || true)"
    if [[ -n "${terminal}" ]]; then
      WATCHER_OUTCOME="job_terminal:${terminal}"
      break
    fi
  fi

  now_epoch="$(date -u +%s)"
  if (( now_epoch - START_EPOCH >= TIMEOUT_SECONDS )); then
    WATCHER_OUTCOME="watcher_timeout"
    printf '%s Watcher timeout reached\n' "${observed_at}" >&2
    break
  fi
  sleep "${POLL_SECONDS}"
done

collector_end_utc="$(utc_now)"
capture_events "${collector_end_utc}"

if ! kubectl -n "${NAMESPACE}" get job "${JOB_NAME}" -o yaml > "${JOB_FINAL}"; then
  printf '# Job unavailable at collector end: %s/%s\n' \
    "${NAMESPACE}" "${JOB_NAME}" > "${JOB_FINAL}"
fi
if [[ -n "${POD_NAME}" ]] && \
    kubectl -n "${NAMESPACE}" get pod "${POD_NAME}" -o yaml > "${POD_FINAL}"; then
  kubectl -n "${NAMESPACE}" logs "pod/${POD_NAME}" \
    -c autoinject-qwen3-small --timestamps > "${TMP_DIR}/main.log" 2>>"${WATCHER_LOG}" || true
else
  printf '# Pod unavailable at collector end for Job: %s/%s\n' \
    "${NAMESPACE}" "${JOB_NAME}" > "${POD_FINAL}"
  : > "${TMP_DIR}/main.log"
fi

python3 - "${TIMELINE}" "${TMP_DIR}/main.log" "${SUMMARY_JSON}" \
  "${SUMMARY_TEXT}" "${LAUNCH_START_UTC}" "${collector_end_utc}" \
  "${WATCHER_OUTCOME}" "${NAMESPACE}" "${JOB_NAME}" "${POD_NAME}" <<'PY'
import datetime as dt
import json
import pathlib
import re
import sys

(
    timeline_path,
    log_path,
    summary_json_path,
    summary_text_path,
    launch_start,
    collector_end,
    watcher_outcome,
    namespace,
    job_name,
    pod_name,
) = sys.argv[1:]

def parse_time(value):
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)

def iso(value):
    return value.isoformat().replace("+00:00", "Z") if value else None

def seconds(start, end):
    return round((end - start).total_seconds(), 3) if start and end else None

records = [
    json.loads(line)
    for line in pathlib.Path(timeline_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
polls = [record for record in records if record.get("record_type") == "poll"]

def first_value(extractor):
    for record in polls:
        value = extractor(record)
        if value:
            return value
    return None

job_created = parse_time(first_value(lambda r: r.get("job_creation_timestamp")))
pod_created = parse_time(first_value(lambda r: r.get("pod_creation_timestamp")))
scheduled = parse_time(first_value(
    lambda r: r.get("pod_conditions", {}).get("PodScheduled", {}).get("last_transition_time")
    if r.get("pod_conditions", {}).get("PodScheduled", {}).get("status") == "True" else None
))
initialized = parse_time(first_value(
    lambda r: r.get("pod_conditions", {}).get("Initialized", {}).get("last_transition_time")
    if r.get("pod_conditions", {}).get("Initialized", {}).get("status") == "True" else None
))
running_observed = parse_time(first_value(
    lambda r: r.get("observed_at_utc") if r.get("pod_phase") == "Running" else None
))
pod_terminal_observed = parse_time(first_value(
    lambda r: r.get("observed_at_utc") if r.get("pod_phase") in {"Succeeded", "Failed"} else None
))
main_started = parse_time(first_value(
    lambda r: (r.get("main_container_state") or {}).get("running", {}).get("startedAt")
    or (r.get("main_container_state") or {}).get("terminated", {}).get("startedAt")
))
main_finished = parse_time(first_value(
    lambda r: (r.get("main_container_state") or {}).get("terminated", {}).get("finishedAt")
))

job_terminal_type = None
job_terminal = None
for record in polls:
    for condition_type in ("Complete", "Failed"):
        condition = record.get("job_conditions", {}).get(condition_type, {})
        if condition.get("status") == "True":
            candidate = parse_time(condition.get("last_transition_time"))
            if candidate and (job_terminal is None or candidate < job_terminal):
                job_terminal = candidate
                job_terminal_type = condition_type

log_text = pathlib.Path(log_path).read_text(encoding="utf-8", errors="replace")
markers = {}
for key in (
    "BASELINE_WRAPPER_START_UTC",
    "AUTOINJECT_COMMAND_START_UTC",
    "AUTOINJECT_COMMAND_END_UTC",
    "BASELINE_WRAPPER_END_UTC",
    "AUTOINJECT_EXIT_CODE",
):
    match = re.search(rf"(?:^|\s){key}=([^\s]+)", log_text, flags=re.MULTILINE)
    markers[key] = match.group(1) if match else None

def first_log_timestamp(pattern):
    for line in log_text.splitlines():
        if pattern in line:
            kubectl_prefix = re.match(r"^(\S+)\s", line)
            app_prefix = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?)\]", line)
            if app_prefix:
                return parse_time(app_prefix.group(1).replace(",", ".") + "+00:00")
            if kubectl_prefix:
                return parse_time(kubectl_prefix.group(1))
    return None

victim_load_completed = first_log_timestamp("Victim model load complete")
first_grpo_start = first_log_timestamp("Starting GRPO training session #1")

launch = parse_time(launch_start)
application_start = parse_time(markers["AUTOINJECT_COMMAND_START_UTC"])
application_end = parse_time(markers["AUTOINJECT_COMMAND_END_UTC"])
wrapper_start = parse_time(markers["BASELINE_WRAPPER_START_UTC"])
wrapper_end = parse_time(markers["BASELINE_WRAPPER_END_UTC"])
collector_finished = parse_time(collector_end)

timestamps = {
    "local_launch_command_start": iso(launch),
    "job_creation": iso(job_created),
    "pod_creation": iso(pod_created),
    "pod_scheduled": iso(scheduled),
    "pod_initialized": iso(initialized),
    "first_pod_running_observation": iso(running_observed),
    "main_container_started": iso(main_started),
    "baseline_wrapper_start": iso(wrapper_start),
    "autoinject_command_start": iso(application_start),
    "victim_model_load_completed_from_log": iso(victim_load_completed),
    "first_grpo_training_session_start_from_log": iso(first_grpo_start),
    "autoinject_command_end": iso(application_end),
    "baseline_wrapper_end": iso(wrapper_end),
    "main_container_finished": iso(main_finished),
    "first_terminal_pod_phase_observation": iso(pod_terminal_observed),
    "job_terminal_transition": iso(job_terminal),
    "timing_collector_end": iso(collector_finished),
}
durations = {
    "submission_to_job_creation": seconds(launch, job_created),
    "job_creation_to_pod_creation": seconds(job_created, pod_created),
    "pod_pending_total": seconds(pod_created, main_started),
    "pod_creation_to_scheduled": seconds(pod_created, scheduled),
    "scheduled_to_main_container_start": seconds(scheduled, main_started),
    "main_container_runtime": seconds(main_started, main_finished),
    "application_runtime": seconds(application_start, application_end),
    "job_total_wall_clock": seconds(launch, job_terminal),
    "scheduling_wait": seconds(pod_created, scheduled),
    "startup_overhead_before_application": seconds(main_started, application_start),
    "shutdown_overhead_after_application": seconds(application_end, main_finished),
}

summary = {
    "schema_version": 1,
    "namespace": namespace,
    "job_name": job_name,
    "pod_name": pod_name or None,
    "watcher_outcome": watcher_outcome,
    "job_terminal_condition": job_terminal_type,
    "autoinject_exit_code": int(markers["AUTOINJECT_EXIT_CODE"]) if markers["AUTOINJECT_EXIT_CODE"] is not None else None,
    "timestamps_utc": timestamps,
    "durations_seconds": durations,
    "definitions": {
        "pod_pending_total": "pod creation to main-container startedAt",
        "scheduling_wait": "pod creation to first PodScheduled=True transition",
        "scheduled_to_main_container_start": "post-scheduling image pull, volume mount, init-container, and container-creation interval combined",
        "job_total_wall_clock": "local launch-command start to Job terminal-condition transition",
    },
    "limitations": [
        "Pod phase=Running and terminal Pod phase are five-second polling observations, not exact server-side transition timestamps.",
        "Kubernetes does not expose one authoritative timestamp separating image pull, volume mount, init, and container creation; retained events may refine that interval.",
        "Victim-load and first-GRPO timestamps are parsed only from naturally occurring application log messages; this baseline currently has no explicit victim-load-complete message, so that field is normally unavailable.",
        "Historical Kubernetes events can expire; kubernetes-events.txt is captured while the watcher is active.",
    ],
}
pathlib.Path(summary_json_path).write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
lines = [
    f"namespace: {namespace}",
    f"job_name: {job_name}",
    f"pod_name: {pod_name or 'unavailable'}",
    f"watcher_outcome: {watcher_outcome}",
    f"job_terminal_condition: {job_terminal_type or 'unavailable'}",
    f"autoinject_exit_code: {summary['autoinject_exit_code']}",
    "",
    "UTC lifecycle timestamps:",
]
lines.extend(f"  {key}: {value or 'unavailable'}" for key, value in timestamps.items())
lines.extend(["", "Durations (seconds):"])
lines.extend(f"  {key}: {value if value is not None else 'unavailable'}" for key, value in durations.items())
lines.extend(["", "Limitations:"])
lines.extend(f"  - {value}" for value in summary["limitations"])
pathlib.Path(summary_text_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

python3 - "${TIMELINE}" "${collector_end_utc}" "${WATCHER_OUTCOME}" <<'PY'
import json
import sys

path, observed, outcome = sys.argv[1:]
with open(path, "a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "record_type": "collector_end",
        "observed_at_utc": observed,
        "watcher_outcome": outcome,
    }, sort_keys=True) + "\n")
PY

printf 'timing_collector_end_utc=%s outcome=%s\n' \
  "${collector_end_utc}" "${WATCHER_OUTCOME}"
