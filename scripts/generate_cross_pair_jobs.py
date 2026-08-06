#!/usr/bin/env python3
"""Render controlled single-job cross-pair policy warm-starts."""

from __future__ import annotations

import copy
from typing import Any

from scripts import cross_pair_experiment
from scripts import generate_continuation_jobs as continuation


class CrossPairJobGenerationError(ValueError):
    """Raised when a cross-pair Job cannot be rendered unambiguously."""


def _validate_spec(spec: dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise CrossPairJobGenerationError("Cross-pair specification must be an object")

    target = spec.get("target")
    source = spec.get("source")
    if not isinstance(target, dict) or not isinstance(source, dict):
        raise CrossPairJobGenerationError(
            "Cross-pair specification requires source and target objects"
        )

    target_key = cross_pair_experiment.section_key(target)
    source_key = cross_pair_experiment.section_key(source)
    if target_key not in continuation.expected_pairs():
        raise CrossPairJobGenerationError(
            f"Unexpected cross-pair target identity: {target_key}"
        )
    if source_key not in continuation.expected_pairs():
        raise CrossPairJobGenerationError(
            f"Unexpected cross-pair source identity: {source_key}"
        )
    if source_key == target_key:
        raise CrossPairJobGenerationError(
            f"Cross-pair source cannot equal its target: {target_key}"
        )
    if source["suite"] != target["suite"]:
        raise CrossPairJobGenerationError("Cross-pair source and target suites differ")
    if source["injection_task"] != target["injection_task"]:
        raise CrossPairJobGenerationError(
            "Cross-pair source and target injection tasks differ"
        )
    if source["user_task"] == target["user_task"]:
        raise CrossPairJobGenerationError(
            "Cross-pair source did not switch user task"
        )

    expected_values = {
        "condition": cross_pair_experiment.CONDITION,
        "stage": cross_pair_experiment.RECORD_STAGE,
        "source_stage": "A",
        "initialization_mode": "policy_warm",
        "reference_mode": "source",
        "query_budget": cross_pair_experiment.QUERY_BUDGET,
    }
    for field, expected in expected_values.items():
        if spec.get(field) != expected:
            raise CrossPairJobGenerationError(
                f"Cross-pair specification has invalid {field}: {spec.get(field)!r}"
            )

    index = spec.get("completion_index")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < continuation.INDEXED_COMPLETIONS
    ):
        raise CrossPairJobGenerationError(
            f"Cross-pair completion index is invalid: {index!r}"
        )

    continuation.validate_sha256(
        spec.get("source_checkpoint_sha256"),
        "cross-pair source checkpoint hash",
    )
    continuation.validate_sha256(
        spec.get("source_checkpoint_state_sha256"),
        "cross-pair source checkpoint-state hash",
    )
    continuation.validate_sha256(
        spec.get("configuration_fingerprint"),
        "cross-pair configuration fingerprint",
    )
    continuation.validate_positive_integer(
        spec.get("source_checkpoint_size_bytes"),
        "cross-pair source checkpoint size",
    )
    continuation.validate_positive_integer(
        spec.get("source_checkpoint_state_size_bytes"),
        "cross-pair source checkpoint-state size",
    )
    continuation.validate_positive_integer(
        spec.get("source_stage_queries_used"),
        "cross-pair source stage queries",
    )
    continuation.validate_positive_integer(
        spec.get("source_cumulative_queries_used"),
        "cross-pair source cumulative queries",
    )

    result_root = spec.get("result_root")
    stage_record_path = spec.get("stage_record_path")
    if (
        not isinstance(result_root, str)
        or not result_root.startswith(cross_pair_experiment.RECORDS_ROOT + "/")
    ):
        raise CrossPairJobGenerationError(
            "Cross-pair result root escapes the cross-pair records root"
        )
    if (
        not isinstance(stage_record_path, str)
        or not stage_record_path.startswith(
            cross_pair_experiment.RECORDS_ROOT + "/records/stages/"
        )
    ):
        raise CrossPairJobGenerationError(
            "Cross-pair stage record escapes the cross-pair records root"
        )


def _job_name(spec: dict[str, Any]) -> str:
    target_slug = cross_pair_experiment.section_slug(spec["target"])
    source_slug = cross_pair_experiment.section_slug(spec["source"])
    value = (
        f"autoinject-x-{target_slug}-from-{source_slug}-"
        f"{cross_pair_experiment.CAMPAIGN_VERSION}"
    )
    if continuation.SAFE_COMPONENT.fullmatch(value) is None or len(value) > 63:
        raise CrossPairJobGenerationError(
            f"Unsafe cross-pair Kubernetes Job name: {value!r}"
        )
    return value


def expected_cross_pair_environment(spec: dict[str, Any]) -> dict[str, str]:
    """Bind source artifacts to the sibling source and execution tasks to the target."""
    _validate_spec(spec)
    target = spec["target"]
    environment = {
        "AI_SOURCE_CHECKPOINT": str(spec["source_checkpoint_path"]),
        "AI_SOURCE_STATE": str(spec["source_checkpoint_state_path"]),
        "AI_SOURCE_SHA256": str(spec["source_checkpoint_sha256"]),
        "AI_SOURCE_SIZE": str(spec["source_checkpoint_size_bytes"]),
        "AI_SOURCE_STATE_SHA256": str(spec["source_checkpoint_state_sha256"]),
        "AI_SOURCE_STATE_SIZE": str(spec["source_checkpoint_state_size_bytes"]),
        "AI_SOURCE_STAGE_QUERIES": str(spec["source_stage_queries_used"]),
        "AI_SOURCE_CUMULATIVE": str(spec["source_cumulative_queries_used"]),
        "AI_CHAIN_ID": str(spec["chain_id"]),
        "AI_RUN_ID": str(spec["run_id"]),
        "AI_RESULT_ROOT": str(spec["result_root"]),
        "AI_STAGE_RECORD": str(spec["stage_record_path"]),
        "AI_STAGE": str(spec["stage"]),
        "AI_SOURCE_STAGE": str(spec["source_stage"]),
        "AI_EXPERIMENT_ID": str(spec["experiment_id"]),
        "AI_SUITE": str(target["suite"]),
        "AI_USER_TASK": str(target["user_task"]),
        "AI_INJECTION_TASK": str(target["injection_task"]),
        "AI_CONFIGURATION_FINGERPRINT": str(spec["configuration_fingerprint"]),
    }
    if set(environment) != set(continuation.INDEXED_TASK_ENV_NAMES):
        raise CrossPairJobGenerationError(
            "Cross-pair environment does not match the reviewed runtime interface"
        )
    return environment


def _continuation_adapter_source(spec: dict[str, Any]) -> dict[str, Any]:
    """Build a private adapter used only to reuse the hardened runtime renderer."""
    target = spec["target"]
    return {
        "suite": target["suite"],
        "user_task": target["user_task"],
        "injection_task": target["injection_task"],
        "stage": "A",
        "checkpoint_path": spec["source_checkpoint_path"],
        "checkpoint_state_path": spec["source_checkpoint_state_path"],
        "checkpoint_sha256": spec["source_checkpoint_sha256"],
        "checkpoint_size_bytes": spec["source_checkpoint_size_bytes"],
        "checkpoint_state_sha256": spec["source_checkpoint_state_sha256"],
        "checkpoint_state_size_bytes": spec[
            "source_checkpoint_state_size_bytes"
        ],
        "stage_queries_used": spec["source_stage_queries_used"],
        "cumulative_queries_used": spec["source_cumulative_queries_used"],
        "chain_id": spec["chain_id"],
        "predecessor_record_path": spec["source_stage_record_id"],
    }


def render_cross_pair_job(
    template: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render one hardened Job with foreign source weights and target task identity."""
    _validate_spec(spec)
    expected_environment = expected_cross_pair_environment(spec)

    job, _ = continuation.render_job(
        template,
        _continuation_adapter_source(spec),
        "B",
    )

    job_name = _job_name(spec)
    job["metadata"]["name"] = job_name
    job["metadata"]["labels"]["autoinject-run-kind"] = (
        f"cross-pair-policy-warm-{cross_pair_experiment.CAMPAIGN_VERSION}"
    )
    job["metadata"]["labels"]["autoinject-stage"] = (
        cross_pair_experiment.RECORD_STAGE.lower()
    )
    job["metadata"]["labels"]["autoinject-suite"] = spec["target"]["suite"]

    container = job["spec"]["template"]["spec"]["containers"][0]
    container["name"] = "cross-pair"
    shell = container["args"][0]
    shell = continuation.replace_once(
        shell,
        f"RECORDS_ROOT={continuation.RECORDS_ROOT}",
        f"RECORDS_ROOT={cross_pair_experiment.RECORDS_ROOT}",
        "cross-pair records root",
    )
    container["args"][0] = shell

    rebound: set[str] = set()
    for entry in container.get("env", []):
        name = entry.get("name") if isinstance(entry, dict) else None
        if name not in expected_environment:
            continue
        if set(entry) != {"name", "value"}:
            raise CrossPairJobGenerationError(
                f"Cross-pair environment {name!r} is not a plain string value"
            )
        entry["value"] = expected_environment[name]
        rebound.add(name)

    if rebound != set(expected_environment):
        missing = sorted(set(expected_environment) - rebound)
        raise CrossPairJobGenerationError(
            f"Rendered cross-pair Job is missing runtime environment: {missing}"
        )
    if continuation.RECORDS_ROOT in shell:
        raise CrossPairJobGenerationError(
            "Rendered cross-pair Job retains the continuation records root"
        )
    if f"RECORDS_ROOT={cross_pair_experiment.RECORDS_ROOT}" not in shell:
        raise CrossPairJobGenerationError(
            "Rendered cross-pair Job lacks its isolated records root"
        )

    target = spec["target"]
    audit_record = {
        **copy.deepcopy(spec),
        "job_name": job_name,
        "suite": target["suite"],
        "user_task": target["user_task"],
        "injection_task": target["injection_task"],
        "output_checkpoint_path": f"{spec['result_root']}/run/checkpoint.pt",
        "output_checkpoint_state_path": (
            f"{spec['result_root']}/run/checkpoint_state.json"
        ),
        "output_checkpoint_sha256": None,
        "output_checkpoint_state_sha256": None,
        "code_commit": continuation.runtime_commit(),
    }
    return job, audit_record
