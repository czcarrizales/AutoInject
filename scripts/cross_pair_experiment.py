#!/usr/bin/env python3
"""Build deterministic audit specifications for cross-pair policy warm-starts."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from scripts import cross_pair_mapping
from scripts import generate_continuation_jobs as continuation


CAMPAIGN_VERSION = "v1"
CONDITION = "cross_pair_policy_warm"
RECORD_STAGE = "X"
RECORDS_ROOT = (
    "/workspace/results/autoinject-qwen3-small/"
    "cross-pair-policy-warm/v1"
)
QUERY_BUDGET = continuation.QUERY_BUDGET


class CrossPairExperimentError(ValueError):
    """Raised when a cross-pair experiment specification is unsafe or ambiguous."""


def section_key(section: dict[str, Any]) -> tuple[str, str, str]:
    """Return the suite, user-task, and injection-task identity."""
    try:
        return (
            section["suite"],
            section["user_task"],
            section["injection_task"],
        )
    except KeyError as error:
        raise CrossPairExperimentError(
            f"Cross-pair identity is missing field: {error}"
        ) from error


def section_slug(section: dict[str, Any]) -> str:
    """Return the established safe task-pair slug."""
    return continuation.pair_slug(
        {
            "suite": section["suite"],
            "user_task": section["user_task"],
            "injection_task": section["injection_task"],
        }
    )


def validate_mapping(mapping: dict[str, Any]) -> None:
    """Require one controlled sibling transfer before creating output identities."""
    target = mapping.get("target")
    source = mapping.get("source")
    descriptor = mapping.get("source_descriptor")

    if not all(isinstance(value, dict) for value in (target, source, descriptor)):
        raise CrossPairExperimentError(
            "Cross-pair mapping requires target, source, and source_descriptor objects"
        )

    target_key = section_key(target)
    source_key = section_key(source)

    if target_key not in continuation.expected_pairs():
        raise CrossPairExperimentError(
            f"Unexpected cross-pair target identity: {target_key}"
        )
    if source_key not in continuation.expected_pairs():
        raise CrossPairExperimentError(
            f"Unexpected cross-pair source identity: {source_key}"
        )
    if source_key == target_key:
        raise CrossPairExperimentError(
            f"Cross-pair source cannot equal its target: {target_key}"
        )
    if source["suite"] != target["suite"]:
        raise CrossPairExperimentError(
            f"Cross-pair source and target suites differ: {source_key} -> {target_key}"
        )
    if source["injection_task"] != target["injection_task"]:
        raise CrossPairExperimentError(
            "Cross-pair source and target injection tasks differ: "
            f"{source_key} -> {target_key}"
        )
    if source["user_task"] == target["user_task"]:
        raise CrossPairExperimentError(
            f"Cross-pair source did not switch user task: {source_key}"
        )
    if continuation.pair_key(descriptor) != source_key:
        raise CrossPairExperimentError(
            "Cross-pair source descriptor does not match the named source"
        )
    if descriptor.get("stage") != "A":
        raise CrossPairExperimentError(
            "Cross-pair source checkpoint must come from Stage A"
        )


def identifiers(mapping: dict[str, Any]) -> dict[str, str]:
    """Create output identities that include both source and target task IDs."""
    validate_mapping(mapping)

    target_slug = section_slug(mapping["target"])
    source_slug = section_slug(mapping["source"])
    chain_id = f"{target_slug}-from-{source_slug}-cross-001"
    run_id = f"{chain_id}-{CAMPAIGN_VERSION}"
    experiment_id = (
        "meta-secalign-qwen3-small-cross-pair-policy-warm-"
        f"{target_slug}-from-{source_slug}-{CAMPAIGN_VERSION}"
    )
    result_root = (
        f"{RECORDS_ROOT}/targets/{target_slug}/"
        f"from/{source_slug}/{run_id}"
    )
    stage_record_path = (
        f"{RECORDS_ROOT}/records/stages/"
        f"{chain_id}--stage-{RECORD_STAGE}--{run_id}.json"
    )

    for label, value in (
        ("chain ID", chain_id),
        ("run ID", run_id),
    ):
        if continuation.SAFE_COMPONENT.fullmatch(value) is None:
            raise CrossPairExperimentError(
                f"Unsafe cross-pair {label}: {value!r}"
            )

    return {
        "chain_id": chain_id,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "result_root": result_root,
        "stage_record_path": stage_record_path,
    }


def fingerprint_payload(
    mapping: dict[str, Any],
    resolved_identifiers: dict[str, str],
) -> dict[str, Any]:
    """Return the complete immutable configuration identity for one transfer."""
    validate_mapping(mapping)
    source = mapping["source_descriptor"]

    return {
        "campaign_version": CAMPAIGN_VERSION,
        "condition": CONDITION,
        "record_stage": RECORD_STAGE,
        "source_stage": "A",
        "query_budget": QUERY_BUDGET,
        "runtime_config_map": continuation.RUNTIME_CONFIG_MAP,
        "code_commit": continuation.runtime_commit(),
        "completion_index": mapping["completion_index"],
        "target": copy.deepcopy(mapping["target"]),
        "source": copy.deepcopy(mapping["source"]),
        "source_checkpoint_path": source["checkpoint_path"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_checkpoint_size_bytes": source["checkpoint_size_bytes"],
        "source_checkpoint_state_path": source["checkpoint_state_path"],
        "source_checkpoint_state_sha256": source[
            "checkpoint_state_sha256"
        ],
        "source_checkpoint_state_size_bytes": source[
            "checkpoint_state_size_bytes"
        ],
        "source_stage_queries_used": source["cumulative_queries_used"],
        "source_cumulative_queries_used": source[
            "cumulative_queries_used"
        ],
        "source_stage_record_id": source["stage_record_id"],
        "source_campaign_fingerprint": source["campaign_fingerprint"],
        **resolved_identifiers,
    }


def configuration_fingerprint(
    mapping: dict[str, Any],
    resolved_identifiers: dict[str, str],
) -> str:
    """Hash the complete source, target, checkpoint, and output configuration."""
    payload = fingerprint_payload(mapping, resolved_identifiers)
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_cross_pair_specs(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one complete cross-pair audit specification for every target."""
    mappings = cross_pair_mapping.build_cross_pair_mappings(records)
    specs: list[dict[str, Any]] = []

    for mapping in mappings:
        resolved_identifiers = identifiers(mapping)
        source = mapping["source_descriptor"]

        spec = {
            "completion_index": mapping["completion_index"],
            "condition": CONDITION,
            "stage": RECORD_STAGE,
            "source_stage": "A",
            "initialization_mode": "policy_warm",
            "reference_mode": "source",
            "query_budget": QUERY_BUDGET,
            "target": copy.deepcopy(mapping["target"]),
            "source": copy.deepcopy(mapping["source"]),
            "source_checkpoint_path": source["checkpoint_path"],
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "source_checkpoint_size_bytes": source[
                "checkpoint_size_bytes"
            ],
            "source_checkpoint_state_path": source[
                "checkpoint_state_path"
            ],
            "source_checkpoint_state_sha256": source[
                "checkpoint_state_sha256"
            ],
            "source_checkpoint_state_size_bytes": source[
                "checkpoint_state_size_bytes"
            ],
            "source_stage_queries_used": source[
                "cumulative_queries_used"
            ],
            "source_cumulative_queries_used": source[
                "cumulative_queries_used"
            ],
            "source_stage_record_id": source["stage_record_id"],
            "source_campaign_fingerprint": source[
                "campaign_fingerprint"
            ],
            **resolved_identifiers,
        }
        spec["configuration_fingerprint"] = configuration_fingerprint(
            mapping,
            resolved_identifiers,
        )
        specs.append(spec)

    for field in (
        "chain_id",
        "run_id",
        "experiment_id",
        "result_root",
        "stage_record_path",
        "configuration_fingerprint",
    ):
        values = [spec[field] for spec in specs]
        if len(set(values)) != len(values):
            raise CrossPairExperimentError(
                f"Cross-pair specifications contain duplicate {field}"
            )

    if [spec["completion_index"] for spec in specs] != list(range(32)):
        raise CrossPairExperimentError(
            "Cross-pair specifications lost canonical completion ordering"
        )

    return specs
