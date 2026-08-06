#!/usr/bin/env python3
"""Render controlled single-job cross-pair policy warm-starts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

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

def cross_pair_indexed_resource_names(
    task_map_sha256: str | None = None,
) -> tuple[str, str]:
    """Return explicit Kubernetes identities for the cross-pair campaign."""
    job_name = (
        f"autoinject-cross-pair-{cross_pair_experiment.CAMPAIGN_VERSION}"
    )
    suffix = ""
    if task_map_sha256 is not None:
        suffix = (
            "-"
            + continuation.validate_sha256(
                task_map_sha256,
                "cross-pair task-map hash",
            )[:12]
        )
    config_map_name = f"{job_name}-tasks{suffix}"

    for label, value in (
        ("cross-pair Indexed Job name", job_name),
        ("cross-pair task ConfigMap name", config_map_name),
    ):
        if (
            continuation.SAFE_COMPONENT.fullmatch(value) is None
            or len(value) > 63
        ):
            raise CrossPairJobGenerationError(
                f"Unsafe {label}: {value!r}"
            )
    return job_name, config_map_name


def package_cross_pair_indexed_resources(
    jobs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    parallelism: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Package 32 rendered cross-pair Jobs into one immutable Indexed Job."""
    parallelism = continuation.validate_parallelism(parallelism)
    resources, indexed_records = continuation.package_indexed_resources(
        jobs,
        records,
        cross_pair_experiment.RECORD_STAGE,
        parallelism,
    )
    if (
        len(resources) != 2
        or resources[0].get("kind") != "ConfigMap"
        or resources[1].get("kind") != "Job"
    ):
        raise CrossPairJobGenerationError(
            "Cross-pair packaging did not produce one ConfigMap and one Job"
        )

    config_map, indexed_job = resources
    try:
        task_map_text = config_map["data"][continuation.INDEXED_TASK_KEY]
    except (KeyError, TypeError) as error:
        raise CrossPairJobGenerationError(
            f"Cross-pair task ConfigMap is incomplete: {error}"
        ) from error
    if not isinstance(task_map_text, str):
        raise CrossPairJobGenerationError(
            "Cross-pair task map must be UTF-8 JSON text"
        )

    task_map_sha256 = hashlib.sha256(task_map_text.encode()).hexdigest()
    indexed_job_name, config_map_name = cross_pair_indexed_resource_names(
        task_map_sha256
    )
    old_config_map_name = config_map.get("metadata", {}).get("name")

    config_map["metadata"]["name"] = config_map_name
    config_map["metadata"]["labels"]["autoinject-run-kind"] = (
        f"cross-pair-policy-warm-{cross_pair_experiment.CAMPAIGN_VERSION}"
    )
    config_map["metadata"]["labels"]["autoinject-stage"] = (
        cross_pair_experiment.RECORD_STAGE.lower()
    )

    indexed_job["metadata"]["name"] = indexed_job_name
    indexed_job["metadata"]["labels"]["autoinject-run-kind"] = (
        f"cross-pair-policy-warm-{cross_pair_experiment.CAMPAIGN_VERSION}"
    )
    indexed_job["metadata"]["labels"]["autoinject-stage"] = (
        cross_pair_experiment.RECORD_STAGE.lower()
    )
    indexed_job["metadata"]["labels"]["autoinject-suite"] = "multi"

    pod_spec = indexed_job["spec"]["template"]["spec"]
    task_volumes = [
        volume
        for volume in pod_spec.get("volumes", [])
        if isinstance(volume, dict)
        and volume.get("name") == continuation.INDEXED_TASK_VOLUME
    ]
    if len(task_volumes) != 1:
        raise CrossPairJobGenerationError(
            "Cross-pair Indexed Job has invalid task ConfigMap volume wiring"
        )
    if task_volumes[0].get("configMap", {}).get("name") != old_config_map_name:
        raise CrossPairJobGenerationError(
            "Cross-pair Indexed Job lost the packaged task ConfigMap identity"
        )
    task_volumes[0]["configMap"]["name"] = config_map_name

    for record in indexed_records:
        record["indexed_job_name"] = indexed_job_name
        record["task_config_map"] = config_map_name
        record["task_map_sha256"] = task_map_sha256

    continuation.validate_indexed_job_invariants(
        indexed_job,
        config_map_name,
        task_map_sha256,
        parallelism,
    )
    return resources, indexed_records

def cross_pair_job_inventory_payload(
    records: list[dict[str, Any]],
    template_path: Path,
    parallelism: int,
    indexed_job_name: str,
    task_config_map: str,
    task_map_sha256: str,
) -> dict[str, Any]:
    """Build the deterministic audit payload for the cross-pair campaign."""
    parallelism = continuation.validate_parallelism(parallelism)
    task_map_sha256 = continuation.validate_sha256(
        task_map_sha256,
        "cross-pair task-map hash",
    )
    if len(records) != continuation.INDEXED_COMPLETIONS:
        raise CrossPairJobGenerationError(
            "Cross-pair inventory requires exactly 32 records"
        )

    for index, record in enumerate(records):
        _validate_spec(record)
        if record.get("completion_index") != index:
            raise CrossPairJobGenerationError(
                f"Cross-pair inventory record {index} has the wrong index"
            )
        if record.get("indexed_job_name") != indexed_job_name:
            raise CrossPairJobGenerationError(
                f"Cross-pair inventory record {index} has the wrong Job name"
            )
        if record.get("task_config_map") != task_config_map:
            raise CrossPairJobGenerationError(
                f"Cross-pair inventory record {index} has the wrong ConfigMap"
            )
        if record.get("task_map_sha256") != task_map_sha256:
            raise CrossPairJobGenerationError(
                f"Cross-pair inventory record {index} has the wrong task-map hash"
            )
        if record.get("code_commit") != continuation.runtime_commit():
            raise CrossPairJobGenerationError(
                f"Cross-pair inventory record {index} has the wrong runtime commit"
            )

    resolved_template = Path(template_path).resolve()
    repository_root = continuation.REPOSITORY_ROOT.resolve()
    try:
        relative_template = resolved_template.relative_to(repository_root)
    except ValueError as error:
        raise CrossPairJobGenerationError(
            "Cross-pair behavioral template escapes the repository"
        ) from error

    return {
        "schema_version": 1,
        "condition": cross_pair_experiment.CONDITION,
        "campaign_version": cross_pair_experiment.CAMPAIGN_VERSION,
        "record_stage": cross_pair_experiment.RECORD_STAGE,
        "source_stage": "A",
        "initialization_mode": "policy_warm",
        "reference_mode": "source",
        "query_budget": cross_pair_experiment.QUERY_BUDGET,
        "behavioral_template": str(relative_template),
        "runtime_config_map": continuation.RUNTIME_CONFIG_MAP,
        "code_commit": continuation.runtime_commit(),
        "completion_mode": "Indexed",
        "completions": continuation.INDEXED_COMPLETIONS,
        "parallelism": parallelism,
        "indexed_job_name": indexed_job_name,
        "task_config_map": task_config_map,
        "task_map_sha256": task_map_sha256,
        "records": copy.deepcopy(records),
    }


def verify_cross_pair_generated_pair(manifest: str, inventory: str) -> None:
    """Verify manifest, audit inventory, hashes, and cross-pair semantics together."""
    try:
        document = json.loads(inventory)
    except json.JSONDecodeError as error:
        raise CrossPairJobGenerationError(
            f"Cross-pair inventory is invalid JSON: {error}"
        ) from error
    if not isinstance(document, dict):
        raise CrossPairJobGenerationError(
            "Cross-pair inventory must be a JSON object"
        )

    yaml_sha256 = hashlib.sha256(manifest.encode()).hexdigest()
    if document.get("generated_yaml_sha256") != yaml_sha256:
        raise CrossPairJobGenerationError(
            "Cross-pair generated YAML/inventory SHA-256 mismatch"
        )

    payload = {
        key: value
        for key, value in document.items()
        if key
        not in {
            "generated_yaml_sha256",
            "job_inventory_payload_sha256",
        }
    }
    payload_sha256 = hashlib.sha256(
        (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()
    if document.get("job_inventory_payload_sha256") != payload_sha256:
        raise CrossPairJobGenerationError(
            "Cross-pair job-inventory payload SHA-256 mismatch"
        )

    expected_header = {
        "schema_version": 1,
        "condition": cross_pair_experiment.CONDITION,
        "campaign_version": cross_pair_experiment.CAMPAIGN_VERSION,
        "record_stage": cross_pair_experiment.RECORD_STAGE,
        "source_stage": "A",
        "initialization_mode": "policy_warm",
        "reference_mode": "source",
        "query_budget": cross_pair_experiment.QUERY_BUDGET,
        "runtime_config_map": continuation.RUNTIME_CONFIG_MAP,
        "code_commit": continuation.runtime_commit(),
        "completion_mode": "Indexed",
        "completions": continuation.INDEXED_COMPLETIONS,
    }
    for field, expected in expected_header.items():
        if document.get(field) != expected:
            raise CrossPairJobGenerationError(
                f"Cross-pair inventory has invalid {field}: "
                f"{document.get(field)!r}"
            )

    parallelism = continuation.validate_parallelism(
        document.get("parallelism")
    )
    task_map_sha256 = continuation.validate_sha256(
        document.get("task_map_sha256"),
        "cross-pair inventory task-map hash",
    )
    expected_job_name, expected_config_map_name = (
        cross_pair_indexed_resource_names(task_map_sha256)
    )
    if document.get("indexed_job_name") != expected_job_name:
        raise CrossPairJobGenerationError(
            "Cross-pair inventory has the wrong Indexed Job name"
        )
    if document.get("task_config_map") != expected_config_map_name:
        raise CrossPairJobGenerationError(
            "Cross-pair inventory has the wrong task ConfigMap name"
        )

    try:
        resources = list(yaml.safe_load_all(manifest))
    except yaml.YAMLError as error:
        raise CrossPairJobGenerationError(
            f"Cross-pair generated YAML is invalid: {error}"
        ) from error
    if (
        len(resources) != 2
        or not all(isinstance(resource, dict) for resource in resources)
        or [resource.get("kind") for resource in resources]
        != ["ConfigMap", "Job"]
    ):
        raise CrossPairJobGenerationError(
            "Cross-pair YAML must contain one ConfigMap and one Job"
        )
    config_map, indexed_job = resources

    if config_map.get("metadata", {}).get("name") != expected_config_map_name:
        raise CrossPairJobGenerationError(
            "Cross-pair task ConfigMap identity differs from its inventory"
        )
    if config_map.get("immutable") is not True:
        raise CrossPairJobGenerationError(
            "Cross-pair task ConfigMap must be immutable"
        )
    if indexed_job.get("metadata", {}).get("name") != expected_job_name:
        raise CrossPairJobGenerationError(
            "Cross-pair Indexed Job identity differs from its inventory"
        )

    for resource in resources:
        metadata = resource.get("metadata", {})
        annotations = metadata.get("annotations", {})
        labels = metadata.get("labels", {})
        if (
            annotations.get(continuation.INDEXED_TASK_MAP_HASH_ANNOTATION)
            != task_map_sha256
        ):
            raise CrossPairJobGenerationError(
                "Cross-pair resource task-map hash annotation differs"
            )
        if (
            annotations.get(
                "autoinject.ucr.edu/job-inventory-payload-sha256"
            )
            != payload_sha256
        ):
            raise CrossPairJobGenerationError(
                "Cross-pair resource inventory hash annotation differs"
            )
        if (
            annotations.get("autoinject.ucr.edu/code-commit")
            != continuation.runtime_commit()
        ):
            raise CrossPairJobGenerationError(
                "Cross-pair resource runtime commit annotation differs"
            )
        if (
            labels.get("autoinject-run-kind")
            != (
                "cross-pair-policy-warm-"
                f"{cross_pair_experiment.CAMPAIGN_VERSION}"
            )
        ):
            raise CrossPairJobGenerationError(
                "Cross-pair resource run-kind label differs"
            )
        if (
            labels.get("autoinject-stage")
            != cross_pair_experiment.RECORD_STAGE.lower()
        ):
            raise CrossPairJobGenerationError(
                "Cross-pair resource stage label differs"
            )

    try:
        task_map_text = config_map["data"][continuation.INDEXED_TASK_KEY]
        task_document = json.loads(task_map_text)
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise CrossPairJobGenerationError(
            f"Cross-pair task ConfigMap is invalid: {error}"
        ) from error
    if (
        not isinstance(task_map_text, str)
        or hashlib.sha256(task_map_text.encode()).hexdigest()
        != task_map_sha256
    ):
        raise CrossPairJobGenerationError(
            "Cross-pair task ConfigMap hash disagrees with inventory"
        )

    records = document.get("records")
    if (
        not isinstance(records, list)
        or len(records) != continuation.INDEXED_COMPLETIONS
    ):
        raise CrossPairJobGenerationError(
            "Cross-pair inventory must contain exactly 32 records"
        )
    continuation.validate_task_document(task_document, records)
    continuation.validate_indexed_job_invariants(
        indexed_job,
        expected_config_map_name,
        task_map_sha256,
        parallelism,
    )

    tasks = task_document["tasks"]
    for index, (record, task) in enumerate(zip(records, tasks)):
        _validate_spec(record)
        if record.get("completion_index") != index:
            raise CrossPairJobGenerationError(
                f"Cross-pair record {index} has the wrong completion index"
            )
        if task.get("pair") != record.get("target"):
            raise CrossPairJobGenerationError(
                f"Cross-pair task {index} does not execute its target"
            )
        if task.get("env") != expected_cross_pair_environment(record):
            raise CrossPairJobGenerationError(
                f"Cross-pair task {index} environment differs from its record"
            )
        if record.get("indexed_job_name") != expected_job_name:
            raise CrossPairJobGenerationError(
                f"Cross-pair record {index} has the wrong Indexed Job name"
            )
        if record.get("task_config_map") != expected_config_map_name:
            raise CrossPairJobGenerationError(
                f"Cross-pair record {index} has the wrong task ConfigMap"
            )
        if record.get("task_map_sha256") != task_map_sha256:
            raise CrossPairJobGenerationError(
                f"Cross-pair record {index} has the wrong task-map hash"
            )


def generate_cross_pair(
    *,
    stage_a_inventory: Path = continuation.DEFAULT_STAGE_A_INVENTORY,
    template_path: Path = continuation.DEFAULT_TEMPLATE,
    parallelism: int = 8,
    enforce_runtime_pin: bool = True,
) -> tuple[
    str,
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Generate deterministic cross-pair YAML and its bound audit inventory."""
    if enforce_runtime_pin:
        continuation.runtime_commit()
    parallelism = continuation.validate_parallelism(parallelism)

    stage_a_records = continuation.load_stage_a_inventory(stage_a_inventory)
    specs = cross_pair_experiment.build_cross_pair_specs(stage_a_records)
    template = continuation.load_behavioral_template(template_path)
    rendered = [
        render_cross_pair_job(template, spec)
        for spec in specs
    ]
    resources, indexed_records = package_cross_pair_indexed_resources(
        [item[0] for item in rendered],
        [item[1] for item in rendered],
        parallelism=parallelism,
    )

    config_map, indexed_job = resources
    task_map_text = config_map["data"][continuation.INDEXED_TASK_KEY]
    task_map_sha256 = hashlib.sha256(task_map_text.encode()).hexdigest()
    payload = cross_pair_job_inventory_payload(
        indexed_records,
        template_path,
        parallelism,
        indexed_job["metadata"]["name"],
        config_map["metadata"]["name"],
        task_map_sha256,
    )
    payload_sha256 = hashlib.sha256(
        (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()
    for resource in resources:
        resource["metadata"].setdefault("annotations", {})[
            "autoinject.ucr.edu/job-inventory-payload-sha256"
        ] = payload_sha256

    manifest = continuation.render_yaml(resources)
    manifest_sha256 = hashlib.sha256(manifest.encode()).hexdigest()
    inventory = continuation.render_job_inventory(
        payload,
        manifest_sha256,
    )
    verify_cross_pair_generated_pair(manifest, inventory)
    return manifest, inventory, resources, indexed_records
