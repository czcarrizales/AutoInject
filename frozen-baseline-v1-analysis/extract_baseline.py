#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


ANALYSIS_ROOT = Path("frozen-baseline-v1-analysis")
RAW_ROOT = ANALYSIS_ROOT / "raw"
OUTPUT_ROOT = ANALYSIS_ROOT / "derived"

SOURCE_COMMIT = "33c2cbdb332d0a9b06f498d4eeac1f11c4491247"
RUN_01_MANIFEST_SHA256 = (
    "2ea408e5503f2e4782279bed2417837150e9b92b189a83228a58b6f881df4e2c"
)
RUN_02_10_MANIFEST_SHA256 = (
    "5ba68ed173787e337725b01ba95b87a6af45d3d3e8f85961bb65e9da3da78a89"
)

RUN_SPECS = [
    {
        "run_number": 1,
        "suite": "banking",
        "user_task": "user_task_14",
        "injection_task": "injection_task_5",
        "pattern": (
            "autoinject-baseline-candidate-v1/"
            "banking-u14-i5/seed-1/*"
        ),
        "expected_candidates": 256,
        "manifest_sha256": RUN_01_MANIFEST_SHA256,
    },
    {
        "run_number": 2,
        "suite": "workspace",
        "user_task": "user_task_38",
        "injection_task": "injection_task_10",
        "pattern": (
            "autoinject-frozen-baseline-v1/"
            "run-02-workspace-u38-i10/seed-1/*"
        ),
        "expected_candidates": 256,
        "manifest_sha256": RUN_02_10_MANIFEST_SHA256,
    },
    {
        "run_number": 3,
        "suite": "travel",
        "user_task": "user_task_18",
        "injection_task": "injection_task_3",
        "pattern": (
            "autoinject-frozen-baseline-v1/"
            "run-03-travel-u18-i3/seed-1/*"
        ),
        "expected_candidates": 256,
        "manifest_sha256": RUN_02_10_MANIFEST_SHA256,
    },
    {
        "run_number": 4,
        "suite": "slack",
        "user_task": "user_task_19",
        "injection_task": "injection_task_2",
        "pattern": (
            "autoinject-frozen-baseline-v1/"
            "run-04-slack-u19-i2/seed-1/*"
        ),
        "expected_candidates": 256,
        "manifest_sha256": RUN_02_10_MANIFEST_SHA256,
    },
    {
        "run_number": 5,
        "suite": "banking",
        "user_task": "user_task_15",
        "injection_task": "injection_task_6",
        "pattern": (
            "autoinject-frozen-baseline-v1/"
            "run-05-banking-u15-i6/seed-1/*"
        ),
        "expected_candidates": 256,
        "manifest_sha256": RUN_02_10_MANIFEST_SHA256,
    },
    {
        "run_number": 6,
        "suite": "workspace",
        "user_task": "user_task_39",
        "injection_task": "injection_task_11",
        "pattern": (
            "autoinject-frozen-baseline-v1/"
            "run-06-workspace-u39-i11/seed-1/*"
        ),
        "expected_candidates": 256,
        "manifest_sha256": RUN_02_10_MANIFEST_SHA256,
    },
    {
        "run_number": 7,
        "suite": "travel",
        "user_task": "user_task_19",
        "injection_task": "injection_task_4",
        "pattern": (
            "autoinject-frozen-baseline-v1/"
            "run-07-travel-u19-i4/seed-1/*"
        ),
        "expected_candidates": 256,
        "manifest_sha256": RUN_02_10_MANIFEST_SHA256,
    },
    {
        "run_number": 8,
        "suite": "slack",
        "user_task": "user_task_20",
        "injection_task": "injection_task_3",
        "pattern": (
            "autoinject-frozen-baseline-v1/"
            "run-08-slack-u20-i3/seed-1/*"
        ),
        "expected_candidates": 64,
        "manifest_sha256": RUN_02_10_MANIFEST_SHA256,
    },
    {
        "run_number": 9,
        "suite": "banking",
        "user_task": "user_task_14",
        "injection_task": "injection_task_7",
        "pattern": (
            "autoinject-frozen-baseline-v1/"
            "run-09-banking-u14-i7/seed-1/*"
        ),
        "expected_candidates": 256,
        "manifest_sha256": RUN_02_10_MANIFEST_SHA256,
    },
    {
        "run_number": 10,
        "suite": "workspace",
        "user_task": "user_task_38",
        "injection_task": "injection_task_12",
        "pattern": (
            "autoinject-frozen-baseline-v1/"
            "run-10-workspace-u38-i12/seed-1/*"
        ),
        "expected_candidates": 256,
        "manifest_sha256": RUN_02_10_MANIFEST_SHA256,
    },
]


def resolve_single_directory(pattern: str) -> Path:
    matches = sorted(path for path in RAW_ROOT.glob(pattern) if path.is_dir())

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one directory for {pattern!r}, found {len(matches)}: "
            f"{matches}"
        )

    return matches[0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def last_regex_value(
    text: str,
    pattern: str,
    *,
    cast: type = int,
    default: Any = None,
) -> Any:
    matches = re.findall(pattern, text, flags=re.MULTILINE | re.IGNORECASE)

    if not matches:
        return default

    value = matches[-1]

    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def parse_outer_cycles(text: str) -> list[dict[str, Any]]:
    cycle_start = re.compile(r"Iter\s+(\d+)\s+-\s+Running benchmark")
    outcome = re.compile(
        r"ASR:\s*([0-9.]+)\s*,\s*Utility:\s*([0-9.]+)"
    )

    cycles: list[dict[str, Any]] = []
    pending_cycle: int | None = None

    for line in text.splitlines():
        start_match = cycle_start.search(line)
        if start_match:
            pending_cycle = int(start_match.group(1))
            continue

        outcome_match = outcome.search(line)
        if outcome_match and pending_cycle is not None:
            asr = float(outcome_match.group(1))
            utility = float(outcome_match.group(2))

            cycles.append(
                {
                    "cycle": pending_cycle,
                    "outer_asr": asr,
                    "outer_utility": utility,
                    "outer_joint_success": int(asr >= 1.0 and utility >= 1.0),
                }
            )
            pending_cycle = None

    return cycles


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_one(value: float) -> bool:
    return value >= 1.0 - 1e-9


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percent(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for spec in RUN_SPECS:
        run_root = resolve_single_directory(spec["pattern"])

        evaluations_path = (
            run_root
            / "artifacts"
            / "grpo_training_outputs"
            / "grpo_evaluations.jsonl"
        )
        job_log_path = run_root / "job.log"
        adaptive_log_path = run_root / "run" / "adaptive_agentdojo.log"

        required_paths = [
            evaluations_path,
            job_log_path,
            adaptive_log_path,
        ]

        for required_path in required_paths:
            if not required_path.is_file():
                raise FileNotFoundError(
                    f"Missing required file for run {spec['run_number']:02d}: "
                    f"{required_path}"
                )

        evaluations = read_jsonl(evaluations_path)
        job_log = read_text(job_log_path)
        adaptive_log = read_text(adaptive_log_path)
        outer_cycles = parse_outer_cycles(adaptive_log)

        expected_candidates = int(spec["expected_candidates"])
        if len(evaluations) != expected_candidates:
            raise RuntimeError(
                f"Run {spec['run_number']:02d} has {len(evaluations)} candidates; "
                f"expected {expected_candidates}"
            )

        run_candidate_rows: list[dict[str, Any]] = []

        for global_index, record in enumerate(evaluations, start=1):
            session_id = int(record.get("training_session_id", 0))

            iteration_label = str(record.get("iteration_label", ""))
            iteration_match = re.search(r"iter(\d+)$", iteration_label)

            if iteration_match:
                candidate_within_session = int(iteration_match.group(1)) + 1
            else:
                candidate_within_session = (
                    sum(
                        1
                        for previous in run_candidate_rows
                        if previous["training_session_id"] == session_id
                    )
                    + 1
                )

            # Every normal cycle has:
            #   1 outer evaluation, then 64 GRPO candidate evaluations.
            #
            # Cycle 1: outer query 1, candidates 2–65
            # Cycle 2: outer query 66, candidates 67–130
            candidate_victim_query = (
                (session_id - 1) * 65
                + 1
                + candidate_within_session
            )

            attack_success = as_float(record.get("success_rate"))
            utility_success = as_float(record.get("utility_score"))
            joint_success = int(
                is_one(attack_success) and is_one(utility_success)
            )

            prob_0 = as_float(record.get("prob_0"))
            prob_1 = as_float(record.get("prob_1"))

            feedback_usable = int((prob_0 + prob_1) > 0.0)
            feedback_continuous = int(
                0.0 < prob_0 < 1.0 and 0.0 < prob_1 < 1.0
            )

            candidate_row = {
                "run_number": spec["run_number"],
                "suite": spec["suite"],
                "user_task": spec["user_task"],
                "injection_task": spec["injection_task"],
                "global_seed": 1,
                "learner_seed": 42,
                "candidate_global_index": global_index,
                "training_session_id": session_id,
                "candidate_within_session": candidate_within_session,
                "candidate_victim_query": candidate_victim_query,
                "iteration_label": iteration_label,
                "timestamp": record.get("timestamp"),
                "suffix": record.get("suffix"),
                "attack_success": attack_success,
                "utility_success": utility_success,
                "joint_success": joint_success,
                "reward": as_float(record.get("reward")),
                "prob_0": prob_0,
                "prob_1": prob_1,
                "feedback_usable": feedback_usable,
                "feedback_continuous": feedback_continuous,
                "is_better_than_previous": int(
                    bool(record.get("is_better_than_previous", False))
                ),
                "prompt": record.get("prompt"),
                "feedback_reasoning": record.get("gpt_reasoning"),
                "source_jsonl": str(evaluations_path),
                "source_commit": SOURCE_COMMIT,
                "manifest_sha256": spec["manifest_sha256"],
            }

            candidate_rows.append(candidate_row)
            run_candidate_rows.append(candidate_row)

        for cycle in outer_cycles:
            cycle_rows.append(
                {
                    "run_number": spec["run_number"],
                    "suite": spec["suite"],
                    "user_task": spec["user_task"],
                    "injection_task": spec["injection_task"],
                    **cycle,
                    "outer_victim_query": (cycle["cycle"] - 1) * 65 + 1,
                    "source_log": str(adaptive_log_path),
                    "source_commit": SOURCE_COMMIT,
                    "manifest_sha256": spec["manifest_sha256"],
                }
            )

        attack_candidates = [
            row for row in run_candidate_rows if is_one(row["attack_success"])
        ]
        utility_candidates = [
            row for row in run_candidate_rows if is_one(row["utility_success"])
        ]
        joint_candidates = [
            row for row in run_candidate_rows if row["joint_success"] == 1
        ]

        best_reward_candidate = max(
            run_candidate_rows,
            key=lambda row: row["reward"],
        )

        first_attack = min(
            attack_candidates,
            key=lambda row: row["candidate_victim_query"],
            default=None,
        )
        first_joint = min(
            joint_candidates,
            key=lambda row: row["candidate_victim_query"],
            default=None,
        )

        outer_any_attack = int(
            any(is_one(cycle["outer_asr"]) for cycle in outer_cycles)
        )
        outer_any_utility = int(
            any(is_one(cycle["outer_utility"]) for cycle in outer_cycles)
        )
        outer_any_joint = int(
            any(cycle["outer_joint_success"] == 1 for cycle in outer_cycles)
        )

        candidate_any_attack = int(bool(attack_candidates))
        candidate_any_utility = int(bool(utility_candidates))
        candidate_any_joint = int(bool(joint_candidates))

        # These "combined observed" fields mean that the outcome appeared
        # either in an outer evaluation or in a saved internal GRPO candidate.
        observed_any_attack = max(candidate_any_attack, outer_any_attack)
        observed_any_utility = max(candidate_any_utility, outer_any_utility)
        observed_any_joint = max(candidate_any_joint, outer_any_joint)

        queries_used = last_regex_value(
            job_log,
            r"^\s*queries_used:\s*(\d+)\s*$",
            cast=int,
        )
        query_budget = last_regex_value(
            job_log,
            r"^\s*query_budget:\s*(\d+)\s*$",
            cast=int,
            default=260,
        )
        pipeline_errors = last_regex_value(
            job_log,
            r"^\s*pipeline_run_errors:\s*(\d+)\s*$",
            cast=int,
            default=0,
        )
        usable_feedback = last_regex_value(
            job_log,
            r"^\s*usable_feedback_probabilities:\s*(\d+)\s*$",
            cast=int,
            default=sum(row["feedback_usable"] for row in run_candidate_rows),
        )
        continuous_feedback = last_regex_value(
            job_log,
            r"^\s*continuous_feedback_probabilities:\s*(\d+)\s*$",
            cast=int,
            default=sum(
                row["feedback_continuous"] for row in run_candidate_rows
            ),
        )
        runtime_seconds = last_regex_value(
            job_log,
            r"^\s*runtime_seconds:\s*(\d+)\s*$",
            cast=int,
        )
        exit_code = last_regex_value(
            job_log,
            r"^\s*exit_code:\s*(\d+)\s*$",
            cast=int,
        )

        early_stopped = int(
            "Early stopping condition met during training" in job_log
            or (
                queries_used is not None
                and query_budget is not None
                and queries_used < query_budget
                and candidate_any_joint == 1
            )
        )

        gpu_match = re.findall(
            r"^\s*gpu:\s*(.+?)\s*$",
            job_log,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        gpu = gpu_match[-1].strip() if gpu_match else None

        final_outer = outer_cycles[-1] if outer_cycles else None

        run_rows.append(
            {
                "run_number": spec["run_number"],
                "suite": spec["suite"],
                "user_task": spec["user_task"],
                "injection_task": spec["injection_task"],
                "global_seed": 1,
                "learner_seed": 42,
                "candidate_count": len(run_candidate_rows),
                "outer_evaluation_count": len(outer_cycles),
                "candidate_attack_success_count": len(attack_candidates),
                "candidate_utility_success_count": len(utility_candidates),
                "candidate_joint_success_count": len(joint_candidates),
                "candidate_attack_success_rate": percent(
                    len(attack_candidates), len(run_candidate_rows)
                ),
                "candidate_utility_success_rate": percent(
                    len(utility_candidates), len(run_candidate_rows)
                ),
                "candidate_joint_success_rate": percent(
                    len(joint_candidates), len(run_candidate_rows)
                ),
                "candidate_any_attack": candidate_any_attack,
                "candidate_any_utility": candidate_any_utility,
                "candidate_any_joint": candidate_any_joint,
                "outer_any_attack": outer_any_attack,
                "outer_any_utility": outer_any_utility,
                "outer_any_joint": outer_any_joint,
                "observed_any_attack": observed_any_attack,
                "observed_any_utility": observed_any_utility,
                "observed_any_joint": observed_any_joint,
                "final_outer_asr": (
                    final_outer["outer_asr"] if final_outer else None
                ),
                "final_outer_utility": (
                    final_outer["outer_utility"] if final_outer else None
                ),
                "best_reward": best_reward_candidate["reward"],
                "best_reward_candidate_index": (
                    best_reward_candidate["candidate_global_index"]
                ),
                "best_reward_attack_success": (
                    best_reward_candidate["attack_success"]
                ),
                "best_reward_utility_success": (
                    best_reward_candidate["utility_success"]
                ),
                "best_reward_joint_success": (
                    best_reward_candidate["joint_success"]
                ),
                "best_reward_suffix": best_reward_candidate["suffix"],
                "first_attack_candidate_index": (
                    first_attack["candidate_global_index"]
                    if first_attack
                    else None
                ),
                "first_attack_victim_query": (
                    first_attack["candidate_victim_query"]
                    if first_attack
                    else None
                ),
                "first_attack_suffix": (
                    first_attack["suffix"] if first_attack else None
                ),
                "first_joint_candidate_index": (
                    first_joint["candidate_global_index"]
                    if first_joint
                    else None
                ),
                "first_joint_victim_query": (
                    first_joint["candidate_victim_query"]
                    if first_joint
                    else None
                ),
                "first_joint_suffix": (
                    first_joint["suffix"] if first_joint else None
                ),
                "query_budget": query_budget,
                "queries_used": queries_used,
                "early_stopped": early_stopped,
                "pipeline_errors": pipeline_errors,
                "usable_feedback_probabilities": usable_feedback,
                "continuous_feedback_probabilities": continuous_feedback,
                "runtime_seconds": runtime_seconds,
                "runtime_minutes": (
                    runtime_seconds / 60.0
                    if runtime_seconds is not None
                    else None
                ),
                "exit_code": exit_code,
                "gpu": gpu,
                "result_root": str(run_root),
                "source_commit": SOURCE_COMMIT,
                "manifest_sha256": spec["manifest_sha256"],
            }
        )

    candidate_fields = [
        "run_number",
        "suite",
        "user_task",
        "injection_task",
        "global_seed",
        "learner_seed",
        "candidate_global_index",
        "training_session_id",
        "candidate_within_session",
        "candidate_victim_query",
        "iteration_label",
        "timestamp",
        "suffix",
        "attack_success",
        "utility_success",
        "joint_success",
        "reward",
        "prob_0",
        "prob_1",
        "feedback_usable",
        "feedback_continuous",
        "is_better_than_previous",
        "prompt",
        "feedback_reasoning",
        "source_jsonl",
        "source_commit",
        "manifest_sha256",
    ]

    cycle_fields = [
        "run_number",
        "suite",
        "user_task",
        "injection_task",
        "cycle",
        "outer_victim_query",
        "outer_asr",
        "outer_utility",
        "outer_joint_success",
        "source_log",
        "source_commit",
        "manifest_sha256",
    ]

    run_fields = list(run_rows[0].keys())

    write_csv(
        OUTPUT_ROOT / "baseline_candidates.csv",
        candidate_rows,
        candidate_fields,
    )
    write_csv(
        OUTPUT_ROOT / "baseline_cycles.csv",
        cycle_rows,
        cycle_fields,
    )
    write_csv(
        OUTPUT_ROOT / "baseline_runs.csv",
        run_rows,
        run_fields,
    )

    summary_rows: list[dict[str, Any]] = []

    groups: list[tuple[str, list[dict[str, Any]]]] = [
        ("all", run_rows),
        *[
            (
                suite,
                [row for row in run_rows if row["suite"] == suite],
            )
            for suite in ["banking", "workspace", "travel", "slack"]
        ],
    ]

    for group_name, rows in groups:
        pair_count = len(rows)

        summary_rows.append(
            {
                "group": group_name,
                "pair_count": pair_count,
                "observed_attack_success_pairs": sum(
                    row["observed_any_attack"] for row in rows
                ),
                "observed_asr": percent(
                    sum(row["observed_any_attack"] for row in rows),
                    pair_count,
                ),
                "observed_utility_success_pairs": sum(
                    row["observed_any_utility"] for row in rows
                ),
                "observed_utility_rate": percent(
                    sum(row["observed_any_utility"] for row in rows),
                    pair_count,
                ),
                "observed_joint_success_pairs": sum(
                    row["observed_any_joint"] for row in rows
                ),
                "observed_joint_success_rate": percent(
                    sum(row["observed_any_joint"] for row in rows),
                    pair_count,
                ),
                "best_reward_attack_success_pairs": sum(
                    int(is_one(row["best_reward_attack_success"]))
                    for row in rows
                ),
                "best_reward_asr": percent(
                    sum(
                        int(is_one(row["best_reward_attack_success"]))
                        for row in rows
                    ),
                    pair_count,
                ),
                "best_reward_utility_success_pairs": sum(
                    int(is_one(row["best_reward_utility_success"]))
                    for row in rows
                ),
                "best_reward_utility_rate": percent(
                    sum(
                        int(is_one(row["best_reward_utility_success"]))
                        for row in rows
                    ),
                    pair_count,
                ),
                "best_reward_joint_success_pairs": sum(
                    row["best_reward_joint_success"] for row in rows
                ),
                "best_reward_joint_success_rate": percent(
                    sum(row["best_reward_joint_success"] for row in rows),
                    pair_count,
                ),
                "early_stop_count": sum(row["early_stopped"] for row in rows),
                "mean_queries_used": statistics.mean(
                    row["queries_used"]
                    for row in rows
                    if row["queries_used"] is not None
                ),
                "median_queries_used": statistics.median(
                    row["queries_used"]
                    for row in rows
                    if row["queries_used"] is not None
                ),
                "mean_runtime_seconds": statistics.mean(
                    row["runtime_seconds"]
                    for row in rows
                    if row["runtime_seconds"] is not None
                ),
                "median_runtime_seconds": statistics.median(
                    row["runtime_seconds"]
                    for row in rows
                    if row["runtime_seconds"] is not None
                ),
                "pipeline_errors_total": sum(
                    row["pipeline_errors"] for row in rows
                ),
            }
        )

    summary_fields = list(summary_rows[0].keys())

    write_csv(
        OUTPUT_ROOT / "baseline_summary.csv",
        summary_rows,
        summary_fields,
    )

    expected_total_candidates = 9 * 256 + 64
    if len(candidate_rows) != expected_total_candidates:
        raise RuntimeError(
            f"Expected {expected_total_candidates} candidate rows, "
            f"found {len(candidate_rows)}"
        )

    if len(run_rows) != 10:
        raise RuntimeError(f"Expected 10 run rows, found {len(run_rows)}")

    print()
    print("Frozen Baseline v1 extraction complete")
    print("=" * 118)
    print(
        f"{'Run':>3}  {'Pair':<29} "
        f"{'Cand':>4} {'Atk':>4} {'Util':>4} {'Joint':>5} "
        f"{'AnyA':>4} {'AnyU':>4} {'AnyJ':>4} "
        f"{'Queries':>7} {'Early':>5} {'Errors':>6}"
    )
    print("-" * 118)

    for row in run_rows:
        pair = (
            f"{row['suite']}/"
            f"{row['user_task'].replace('user_task_', 'u')}/"
            f"{row['injection_task'].replace('injection_task_', 'i')}"
        )

        print(
            f"{row['run_number']:>3}  "
            f"{pair:<29} "
            f"{row['candidate_count']:>4} "
            f"{row['candidate_attack_success_count']:>4} "
            f"{row['candidate_utility_success_count']:>4} "
            f"{row['candidate_joint_success_count']:>5} "
            f"{row['observed_any_attack']:>4} "
            f"{row['observed_any_utility']:>4} "
            f"{row['observed_any_joint']:>4} "
            f"{str(row['queries_used']):>7} "
            f"{row['early_stopped']:>5} "
            f"{row['pipeline_errors']:>6}"
        )

    all_summary = summary_rows[0]

    print()
    print("Aggregate observed outcomes across the 10 task pairs")
    print("-" * 58)
    print(
        "Any attack success: "
        f"{all_summary['observed_attack_success_pairs']}/10 "
        f"= {100 * all_summary['observed_asr']:.1f}%"
    )
    print(
        "Any utility success: "
        f"{all_summary['observed_utility_success_pairs']}/10 "
        f"= {100 * all_summary['observed_utility_rate']:.1f}%"
    )
    print(
        "Any joint success:  "
        f"{all_summary['observed_joint_success_pairs']}/10 "
        f"= {100 * all_summary['observed_joint_success_rate']:.1f}%"
    )
    print(
        "Best-reward ASR:    "
        f"{all_summary['best_reward_attack_success_pairs']}/10 "
        f"= {100 * all_summary['best_reward_asr']:.1f}%"
    )
    print(
        "Best-reward utility:"
        f" {all_summary['best_reward_utility_success_pairs']}/10 "
        f"= {100 * all_summary['best_reward_utility_rate']:.1f}%"
    )
    print(
        "Mean queries used:  "
        f"{all_summary['mean_queries_used']:.1f}"
    )
    print(
        "Median queries used:"
        f" {all_summary['median_queries_used']:.1f}"
    )
    print(
        "Early-stop runs:    "
        f"{all_summary['early_stop_count']}/10"
    )
    print(
        "Pipeline errors:    "
        f"{all_summary['pipeline_errors_total']}"
    )

    print()
    print("Generated files:")
    for path in sorted(OUTPUT_ROOT.glob("*.csv")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
