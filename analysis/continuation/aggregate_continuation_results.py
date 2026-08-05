#!/usr/bin/env python3
"""Regenerate continuation CSVs from immutable flat stage records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


INVALID_JSON = object()

STAGE_FIELDS = """chain_id stage run_id source_stage source_checkpoint_path
source_checkpoint_sha256 suite user_task injection_task pair initialization_mode reference_mode
completion_status technical_invalidation technical_invalidation_reason pair_asr pair_utility
outer_evaluation_count grpo_evaluation_count valid_metric_records invalid_json_records
missing_metric_records unsupported_metric_type_records nonfinite_metric_records out_of_range_metric_records
record_count_reconciled maximum_security_score queries_used victim_queries_total pipeline_runs_attempted
pipeline_runs_completed pipeline_run_errors pipeline_error_fraction budget_overshoot early_stopped
data_quality_classification stage_query_budget cumulative_queries_used stage_start_time stage_end_time
stage_wall_clock_seconds git_commit resolved_config_sha256 gpu_name gpu_count""".split()
EVALUATION_FIELDS = """chain_id stage run_id suite user_task injection_task phase sequence metric_status
success_rate utility_score reward timestamp source_raw_path""".split()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(record: dict[str, Any], canonical: str, alias: str, *, utility: bool):
    if canonical not in record and alias not in record:
        return None, "missing"
    value = record.get(canonical, record.get(alias))
    if value is None:
        return None, "missing"
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, (int, float)):
        return None, "unsupported"
    if not math.isfinite(value):
        return None, "nonfinite"
    if utility:
        return (int(value), "valid") if value in (0, 1) else (None, "unsupported")
    return (float(value), "valid") if 0 <= value <= 1 else (None, "out_of_range")


def read_jsonl(path: Path) -> list[Any]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                records.append(INVALID_JSON)
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                records.append(INVALID_JSON)
                continue
            records.append(value)
    return records


def load_records(records_root: Path) -> list[dict[str, Any]]:
    records = []
    identities: dict[tuple[Any, Any, Any], Path] = {}
    for path in sorted((records_root / "records" / "stages").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"Stage record is not an object: {path}")
        identity = (record.get("chain_id"), record.get("stage"), record.get("run_id"))
        previous = identities.get(identity)
        if previous is not None:
            raise ValueError(f"Duplicate stage identity {identity}: {previous} and {path}")
        identities[identity] = path
        for key in (
            "experience_history", "grpo_evaluations", "checkpoint_state", "output_checkpoint"
        ):
            raw_path = Path(record[f"{key}_path"])
            if not raw_path.is_file():
                raise FileNotFoundError(f"{path}: missing {key}: {raw_path}")
            if sha256_file(raw_path) != record[f"{key}_sha256"]:
                raise ValueError(f"{path}: incorrect hash for {key}: {raw_path}")
        if record.get("stdout_log_path") is not None:
            stdout_path = Path(record["stdout_log_path"])
            if not stdout_path.is_file():
                raise FileNotFoundError(f"{path}: missing stdout_log: {stdout_path}")
            if sha256_file(stdout_path) != record.get("stdout_log_sha256"):
                raise ValueError(f"{path}: incorrect hash for stdout_log: {stdout_path}")
        records.append(record)
    return sorted(records, key=lambda row: (row["chain_id"], str(row["stage"]), row["run_id"]))


def quality_classification(record: dict[str, Any], determined: bool, reporting: dict[str, Any], learner_state: dict[str, Any]) -> str:
    if not determined:
        return "cannot_determine"
    if record["suite"] == "slack" and record["injection_task"] == "injection_task_5":
        return "evaluator_implementation_limitation"
    if learner_state.get("early_stopped") is True:
        return "early_stopped"
    if isinstance(reporting.get("pipeline_run_errors"), int) and reporting["pipeline_run_errors"] > 0:
        return "evaluator_errors_present"
    return "standard"


def aggregate_stage(record: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    streams = [("outer", "experience_history"), ("grpo", "grpo_evaluations")]
    valid: list[tuple[float, int]] = []
    evaluations: list[dict[str, Any]] = []
    counts = {name: 0 for name in ("invalid_json", "missing", "unsupported", "nonfinite", "out_of_range")}
    stream_counts = {}
    for phase, key in streams:
        raw_records = read_jsonl(Path(record[f"{key}_path"]))
        stream_counts[phase] = len(raw_records)
        for sequence, raw in enumerate(raw_records, start=1):
            if raw is INVALID_JSON:
                counts["invalid_json"] += 1
                status = "invalid_json"
                security = utility = None
            elif not isinstance(raw, dict):
                counts["unsupported"] += 1
                status = "unsupported_metric"
                security = utility = None
            else:
                security, security_status = metric(raw, "success_rate", "security", utility=False)
                utility, utility_status = metric(raw, "utility_score", "utility", utility=True)
                if security_status != "valid" or utility_status != "valid":
                    status = next(
                        kind for kind in ("missing", "nonfinite", "out_of_range", "unsupported")
                        if kind in (security_status, utility_status)
                    )
                    counts[status] += 1
                else:
                    valid.append((security, utility))
                    status = "valid"
            evaluations.append({
                "chain_id": record["chain_id"], "stage": record["stage"], "run_id": record["run_id"],
                "suite": record["suite"], "user_task": record["user_task"], "injection_task": record["injection_task"],
                "phase": phase, "sequence": sequence, "metric_status": status,
                "success_rate": security, "utility_score": utility,
                "reward": raw.get("reward") if isinstance(raw, dict) else None,
                "timestamp": raw.get("timestamp") if isinstance(raw, dict) else None,
                "source_raw_path": record[f"{key}_path"],
            })
    total = sum(stream_counts.values())
    reconciled = total == len(valid) + sum(counts.values())
    determined = reconciled and not any(counts.values())
    complete_security = any(security == 1.0 for security, _ in valid)
    pair_asr = int(complete_security) if determined else None
    if not determined:
        pair_utility = None
    elif any(security == 1.0 and utility == 1 for security, utility in valid):
        pair_utility = 1
    elif complete_security:
        pair_utility = 0
    else:
        pair_utility = int(any(utility == 1 for _, utility in valid))
    state = json.loads(Path(record["checkpoint_state_path"]).read_text(encoding="utf-8"))
    reporting = state["experiment_reporting"]
    learner_state = state["learner_state"]
    attempted = reporting.get("pipeline_runs_attempted")
    errors = reporting.get("pipeline_run_errors")
    fraction = errors / attempted if isinstance(errors, int) and isinstance(attempted, int) and attempted else None
    row = {
        "chain_id": record["chain_id"], "stage": record["stage"], "run_id": record["run_id"],
        "source_stage": record["source_stage"], "source_checkpoint_path": record["source_checkpoint_path"],
        "source_checkpoint_sha256": record["source_checkpoint_sha256"], "suite": record["suite"],
        "user_task": record["user_task"], "injection_task": record["injection_task"],
        "pair": f"{record['suite']}-u{record['user_task'].removeprefix('user_task_')}-i{record['injection_task'].removeprefix('injection_task_')}",
        "initialization_mode": record["initialization_mode"], "reference_mode": record["reference_mode"],
        "completion_status": record["completion_status"], "technical_invalidation": record["technical_invalidation"],
        "technical_invalidation_reason": record["technical_invalidation_reason"],
        "pair_asr": pair_asr, "pair_utility": pair_utility,
        "outer_evaluation_count": stream_counts["outer"], "grpo_evaluation_count": stream_counts["grpo"],
        "valid_metric_records": len(valid), "invalid_json_records": counts["invalid_json"],
        "missing_metric_records": counts["missing"], "unsupported_metric_type_records": counts["unsupported"],
        "nonfinite_metric_records": counts["nonfinite"], "out_of_range_metric_records": counts["out_of_range"],
        "record_count_reconciled": reconciled,
        "maximum_security_score": max((security for security, _ in valid), default=None),
        "queries_used": record["stage_queries_used"], "victim_queries_total": reporting.get("victim_queries_total"),
        "pipeline_runs_attempted": attempted, "pipeline_runs_completed": reporting.get("pipeline_runs_completed"),
        "pipeline_run_errors": errors, "pipeline_error_fraction": fraction,
        "budget_overshoot": reporting.get("budget_overshoot"), "early_stopped": learner_state.get("early_stopped"),
        "data_quality_classification": quality_classification(record, determined, reporting, learner_state),
        "stage_query_budget": record["stage_query_budget"], "cumulative_queries_used": record["cumulative_queries_used"],
        "stage_start_time": record["stage_start_time"], "stage_end_time": record["stage_end_time"],
        "stage_wall_clock_seconds": record["stage_wall_clock_seconds"], "git_commit": record["git_commit"],
        "resolved_config_sha256": record["resolved_config_sha256"], "gpu_name": record["gpu_name"], "gpu_count": record["gpu_count"],
    }
    return row, evaluations


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    stage_rows, evaluation_rows = [], []
    for record in load_records(args.records_root):
        stage_row, evaluations = aggregate_stage(record)
        stage_rows.append(stage_row)
        evaluation_rows.extend(evaluations)
    write_csv(args.output_dir / "continuation-stage-results-v1.csv", stage_rows, STAGE_FIELDS)
    write_csv(args.output_dir / "continuation-evaluations-v1.csv", evaluation_rows, EVALUATION_FIELDS)


if __name__ == "__main__":
    main()
