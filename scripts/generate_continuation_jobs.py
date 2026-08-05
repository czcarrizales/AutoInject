#!/usr/bin/env python3
"""Generate deterministic sequential continuation Jobs from verified sources."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_A_INVENTORY = REPOSITORY_ROOT / "continuation/stage-a-inventory-v3.json"
DEFAULT_TEMPLATE = (
    REPOSITORY_ROOT / "autoinject-travel-u19-i5-stage-b-validation-v2.yaml"
)
RECORDS_ROOT = "/workspace/results/autoinject-qwen3-small/continuations/v1"
RUNTIME_CONFIG_MAP = "autoinject-continuation-runtime-v1"
# This is deliberately unset until a committed hardened runtime revision is
# available. CODE_COMMIT and RUNTIME_HARDENING_COMMIT are derived aliases so
# every rendered location has one generator-supplied source of truth.
INTENDED_RUNTIME_COMMIT: str | None = "66f4e040e891926bd493b5466d501d11b3f7fa96"
CODE_COMMIT = INTENDED_RUNTIME_COMMIT
RUNTIME_HARDENING_COMMIT = INTENDED_RUNTIME_COMMIT
TEMPLATE_RUNTIME_COMMIT = "abfbad93c88766ba82f4195e43355d5d269a8c8b"
QUERY_BUDGET = 260
CAMPAIGN_VERSION = "v1"
STAGE_RECORD_SCHEMA = "continuation-stage-record/v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_COMPONENT = re.compile(r"[a-z0-9][a-z0-9-]*\Z")

STAGE_A_FIELDS = {
    "suite",
    "user_task",
    "injection_task",
    "stage",
    "checkpoint_path",
    "checkpoint_state_path",
    "checkpoint_state_sha256",
    "checkpoint_state_size_bytes",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "cumulative_queries_used",
    "chain_id",
    "run_id",
    "result_root",
    "stage_record_id",
    "campaign_fingerprint",
}
PREDECESSOR_FIELDS = {
    "schema_version",
    "chain_id",
    "stage",
    "run_id",
    "suite",
    "user_task",
    "injection_task",
    "initialization_mode",
    "reference_mode",
    "source_stage",
    "source_checkpoint_path",
    "source_checkpoint_sha256",
    "source_checkpoint_size_bytes",
    "source_checkpoint_state_path",
    "source_checkpoint_state_sha256",
    "source_checkpoint_state_size_bytes",
    "result_root",
    "output_checkpoint_path",
    "output_checkpoint_sha256",
    "output_checkpoint_size_bytes",
    "checkpoint_state_path",
    "checkpoint_state_sha256",
    "checkpoint_state_size_bytes",
    "stage_query_budget",
    "stage_queries_used",
    "cumulative_queries_used",
    "git_commit",
    "resolved_config_sha256",
    "configuration_fingerprint",
    "completion_status",
    "technical_invalidation",
    "technical_invalidation_reason",
    "experiment_reporting",
}


class GenerationError(ValueError):
    """Raised when a complete, unambiguous continuation cannot be generated."""


def runtime_commit() -> str:
    """Return the single committed runtime revision required for launchable Jobs."""
    if (
        INTENDED_RUNTIME_COMMIT is None
        or CODE_COMMIT != INTENDED_RUNTIME_COMMIT
        or RUNTIME_HARDENING_COMMIT != INTENDED_RUNTIME_COMMIT
    ):
        raise GenerationError(
            "Hardened runtime code is not yet committed and pinned; refusing launchable output"
        )
    return INTENDED_RUNTIME_COMMIT


class LiteralDumper(yaml.SafeDumper):
    """Emit embedded shell scripts as readable literal YAML blocks."""


def _represent_string(dumper: yaml.SafeDumper, value: str):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


LiteralDumper.add_representer(str, _represent_string)


def expected_pairs() -> set[tuple[str, str, str]]:
    return {
        *(("workspace", f"user_task_{u}", f"injection_task_{i}") for u in (38, 39) for i in range(10, 14)),
        *(("banking", f"user_task_{u}", f"injection_task_{i}") for u in (14, 15) for i in range(5, 9)),
        *(("travel", f"user_task_{u}", f"injection_task_{i}") for u in (18, 19) for i in range(3, 7)),
        *(("slack", f"user_task_{u}", f"injection_task_{i}") for u in (19, 20) for i in range(2, 6)),
    }


def pair_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return record["suite"], record["user_task"], record["injection_task"]


def pair_slug(record: dict[str, Any]) -> str:
    user_match = re.fullmatch(r"user_task_([0-9]+)", record["user_task"])
    injection_match = re.fullmatch(
        r"injection_task_([0-9]+)", record["injection_task"]
    )
    if user_match is None or injection_match is None:
        raise GenerationError(f"Invalid task identity: {pair_key(record)}")
    slug = (
        f"{record['suite']}-u{user_match.group(1)}-i{injection_match.group(1)}"
    )
    if SAFE_COMPONENT.fullmatch(slug) is None:
        raise GenerationError(f"Unsafe task-pair slug: {slug!r}")
    return slug


def stage_index(stage: str) -> int:
    if not isinstance(stage, str) or re.fullmatch(r"[A-Z]", stage) is None:
        raise GenerationError(f"Stage must be one uppercase letter: {stage!r}")
    return ord(stage) - ord("A")


def stage_from_index(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 26:
        raise GenerationError(f"Unsupported stage index: {index!r}")
    return chr(ord("A") + index)


def predecessor_stage(stage: str) -> str:
    index = stage_index(stage)
    if index == 0:
        raise GenerationError("Stage A is the cold-start source and has no predecessor")
    return stage_from_index(index - 1)


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GenerationError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise GenerationError(f"{label} must be a JSON object: {path}")
    return value


def validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise GenerationError(f"{label} is not a lowercase SHA-256: {value!r}")
    return value


def validate_positive_integer(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GenerationError(f"{label} must be an integer >= {minimum}: {value!r}")
    return value


def validate_unique(records: list[dict[str, Any]], fields: Iterable[str], label: str) -> None:
    pairs = [pair_key(record) for record in records]
    if len(set(pairs)) != len(pairs):
        raise GenerationError(f"{label} contains duplicate task identities")
    for field in fields:
        values = [record[field] for record in records]
        if len(set(values)) != len(values):
            raise GenerationError(f"{label} contains duplicate {field} values")


def load_stage_a_inventory(path: Path) -> list[dict[str, Any]]:
    document = load_json_object(path, "Stage A inventory")
    if document.get("schema_version") != 2 or document.get("source_stage") != "A":
        raise GenerationError("Unsupported Stage A inventory schema or source stage")
    if document.get("inspection_job_name") != "autoinject-stage-a-checkpoint-hash-inspection-v3":
        raise GenerationError("Stage A inventory was not produced by inspection v3")
    records = document.get("records")
    if not isinstance(records, list) or len(records) != 32:
        raise GenerationError("Stage A inventory must contain exactly 32 records")
    if not all(isinstance(record, dict) for record in records):
        raise GenerationError("Every Stage A inventory record must be an object")

    for index, record in enumerate(records, 1):
        if set(record) != STAGE_A_FIELDS:
            raise GenerationError(f"Stage A inventory record {index} has invalid fields")
        if record["stage"] != "A":
            raise GenerationError(f"Stage A inventory record {index} has wrong stage")
        if pair_key(record) not in expected_pairs():
            raise GenerationError(f"Stage A inventory record {index} has unexpected pair")
        pair_slug(record)
        validate_sha256(record["checkpoint_sha256"], f"record {index} checkpoint hash")
        validate_sha256(record["checkpoint_state_sha256"], f"record {index} state hash")
        validate_positive_integer(record["checkpoint_size_bytes"], f"record {index} checkpoint size")
        validate_positive_integer(record["checkpoint_state_size_bytes"], f"record {index} state size")
        validate_positive_integer(record["cumulative_queries_used"], f"record {index} cumulative queries")
        checkpoint = Path(record["checkpoint_path"])
        state = Path(record["checkpoint_state_path"])
        if not checkpoint.is_absolute() or checkpoint.name != "checkpoint.pt":
            raise GenerationError(f"Stage A record {index} has invalid checkpoint path")
        if not state.is_absolute() or state.name != "checkpoint_state.json":
            raise GenerationError(f"Stage A record {index} has invalid state path")
        if checkpoint.parent != state.parent:
            raise GenerationError(f"Stage A record {index} paths do not share a run directory")
        if record["result_root"] != str(checkpoint.parent.parent):
            raise GenerationError(f"Stage A record {index} has inconsistent result root")
        if record["run_id"] != checkpoint.parent.parent.name:
            raise GenerationError(f"Stage A record {index} has inconsistent run ID")
        if record["stage_record_id"] != f"stage-a-inventory-v3.json#{record['chain_id']}":
            raise GenerationError(f"Stage A record {index} has inconsistent record ID")
        validate_sha256(record["campaign_fingerprint"], f"record {index} campaign fingerprint")
        if not isinstance(record["chain_id"], str) or not record["chain_id"]:
            raise GenerationError(f"Stage A record {index} has invalid chain ID")

    validate_unique(
        records,
        ("chain_id", "checkpoint_path", "checkpoint_state_path", "checkpoint_sha256", "checkpoint_state_sha256"),
        "Stage A inventory",
    )
    if {pair_key(record) for record in records} != expected_pairs():
        raise GenerationError("Stage A inventory pair set differs from the established 32")
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_nonempty(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise GenerationError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise GenerationError(f"{label} must be a regular nonempty file: {path}")
    return metadata


def source_from_stage_a(record: dict[str, Any]) -> dict[str, Any]:
    source = copy.deepcopy(record)
    source["stage_queries_used"] = record["cumulative_queries_used"]
    source["predecessor_record_path"] = record["stage_record_id"]
    return source


def validate_descriptor_lineage(
    descriptor: dict[str, Any], base_record: dict[str, Any]
) -> None:
    """Recursively anchor a nominated source descriptor to imported Stage A."""
    if pair_key(descriptor) != pair_key(base_record) or descriptor.get("chain_id") != base_record["chain_id"]:
        raise GenerationError("Nominated predecessor lineage crosses task or chain")
    if descriptor.get("stage") == "A":
        for field in STAGE_A_FIELDS:
            if descriptor.get(field) != base_record[field]:
                raise GenerationError(f"Nominated Stage A lineage differs at {field}")
        return
    prior = descriptor.get("source_descriptor")
    if not isinstance(prior, dict):
        raise GenerationError("Nominated predecessor lineage is incomplete")
    if prior.get("stage") != predecessor_stage(descriptor["stage"]):
        raise GenerationError("Nominated predecessor lineage skips a stage")
    validate_descriptor_lineage(prior, base_record)


def validate_predecessor_record(
    nomination: dict[str, Any],
    base_record: dict[str, Any],
) -> dict[str, Any]:
    expected_stage = nomination["stage"]
    validate_canonical_nomination(nomination, base_record)
    record_path = Path(nomination["stage_record_path"])
    if not record_path.is_absolute():
        raise GenerationError("Nominated predecessor stage-record path must be absolute")
    regular_nonempty(record_path, "Nominated predecessor stage record")
    record = load_json_object(record_path, "nominated predecessor stage record")
    missing = sorted(PREDECESSOR_FIELDS - set(record))
    if missing:
        raise GenerationError(f"Incomplete predecessor record {record_path}: missing={missing}")
    if record["schema_version"] != STAGE_RECORD_SCHEMA:
        raise GenerationError(f"Unsupported predecessor schema: {record_path}")
    if record["stage"] != expected_stage:
        raise GenerationError(f"Predecessor stage mismatch: {record_path}")
    if record["chain_id"] != base_record["chain_id"] or pair_key(record) != pair_key(base_record):
        raise GenerationError(f"Predecessor chain or task identity mismatch: {record_path}")
    for field in ("chain_id", "run_id", "result_root", "configuration_fingerprint"):
        if record[field] != nomination[field]:
            raise GenerationError(f"Predecessor {field} differs from exact nomination: {record_path}")
    if record["completion_status"] != "completed":
        raise GenerationError(f"Predecessor is not successfully completed: {record_path}")
    if record["technical_invalidation"] is not None or record["technical_invalidation_reason"] is not None:
        raise GenerationError(f"Predecessor is technically invalidated: {record_path}")
    if record["initialization_mode"] != "policy_warm" or record["reference_mode"] != "source":
        raise GenerationError(f"Predecessor continuation mode is inconsistent: {record_path}")
    prior = nomination.get("source_descriptor")
    if not isinstance(prior, dict):
        raise GenerationError(f"Predecessor nomination lacks source lineage: {record_path}")
    if record["source_stage"] != predecessor_stage(expected_stage) or record["source_stage"] != prior.get("stage"):
        raise GenerationError(f"Predecessor source stage is inconsistent: {record_path}")
    validate_descriptor_lineage(prior, base_record)
    source_pairs = (
        ("source_checkpoint_path", "checkpoint_path"),
        ("source_checkpoint_sha256", "checkpoint_sha256"),
        ("source_checkpoint_size_bytes", "checkpoint_size_bytes"),
        ("source_checkpoint_state_path", "checkpoint_state_path"),
        ("source_checkpoint_state_sha256", "checkpoint_state_sha256"),
        ("source_checkpoint_state_size_bytes", "checkpoint_state_size_bytes"),
    )
    for record_field, prior_field in source_pairs:
        if record[record_field] != prior.get(prior_field):
            raise GenerationError(
                f"Predecessor {record_field} differs from nominated prior stage: {record_path}"
            )
    if record["git_commit"] != runtime_commit():
        raise GenerationError(f"Predecessor code commit is inconsistent: {record_path}")
    if record["stage_query_budget"] != QUERY_BUDGET:
        raise GenerationError(f"Predecessor query budget is inconsistent: {record_path}")
    if not isinstance(record["experiment_reporting"], dict):
        raise GenerationError(f"Predecessor experiment reporting is incomplete: {record_path}")
    validate_sha256(record["source_checkpoint_sha256"], "predecessor source hash")
    validate_sha256(record["resolved_config_sha256"], "predecessor config hash")
    checkpoint_hash = validate_sha256(
        record["output_checkpoint_sha256"], "predecessor output checkpoint hash"
    )
    state_hash = validate_sha256(
        record["checkpoint_state_sha256"], "predecessor checkpoint-state hash"
    )
    stage_queries = validate_positive_integer(
        record["stage_queries_used"], "predecessor stage queries", allow_zero=True
    )
    if stage_queries > QUERY_BUDGET:
        raise GenerationError(f"Predecessor stage queries exceed its budget: {record_path}")
    cumulative_queries = validate_positive_integer(
        record["cumulative_queries_used"], "predecessor cumulative queries"
    )
    expected_cumulative = prior.get("cumulative_queries_used")
    if not isinstance(expected_cumulative, int) or cumulative_queries != expected_cumulative + stage_queries:
        raise GenerationError(f"Predecessor cumulative queries are inconsistent: {record_path}")

    checkpoint = Path(record["output_checkpoint_path"])
    state_path = Path(record["checkpoint_state_path"])
    if not checkpoint.is_absolute() or checkpoint.name != "checkpoint.pt":
        raise GenerationError(f"Predecessor output checkpoint path is invalid: {record_path}")
    if not state_path.is_absolute() or state_path.name != "checkpoint_state.json":
        raise GenerationError(f"Predecessor checkpoint-state path is invalid: {record_path}")
    result_root = Path(nomination["result_root"])
    if not result_root.is_absolute():
        raise GenerationError(f"Predecessor result root is not absolute: {record_path}")
    expected_run = result_root / "run"
    expected_checkpoint = expected_run / "checkpoint.pt"
    expected_state = expected_run / "checkpoint_state.json"
    if checkpoint != expected_checkpoint or checkpoint != Path(nomination["output_checkpoint_path"]):
        raise GenerationError(f"Predecessor checkpoint differs from exact nomination: {record_path}")
    if state_path != expected_state or state_path != Path(nomination["output_checkpoint_state_path"]):
        raise GenerationError(f"Predecessor state differs from exact nomination: {record_path}")
    if checkpoint.parent != state_path.parent:
        raise GenerationError(f"Predecessor artifacts are from different runs: {record_path}")
    expected_name = f"{record['chain_id']}--stage-{expected_stage}--{record['run_id']}.json"
    if record_path.name != expected_name:
        raise GenerationError(f"Predecessor record filename is inconsistent: {record_path}")

    checkpoint_metadata = regular_nonempty(checkpoint, "Predecessor checkpoint")
    state_metadata = regular_nonempty(state_path, "Predecessor checkpoint state")
    if record["output_checkpoint_size_bytes"] != checkpoint_metadata.st_size:
        raise GenerationError(f"Predecessor checkpoint size mismatch: {checkpoint}")
    if record["checkpoint_state_size_bytes"] != state_metadata.st_size:
        raise GenerationError(f"Predecessor checkpoint-state size mismatch: {state_path}")
    resolved_root = result_root.resolve(strict=True)
    if checkpoint.resolve(strict=True) != expected_checkpoint or state_path.resolve(strict=True) != expected_state:
        raise GenerationError(f"Predecessor artifact path or symlink escapes nominated run: {record_path}")
    if expected_run.resolve(strict=True).parent != resolved_root:
        raise GenerationError(f"Predecessor run escapes nominated result root: {record_path}")
    if sha256_file(checkpoint) != checkpoint_hash:
        raise GenerationError(f"Predecessor checkpoint hash mismatch: {checkpoint}")
    if sha256_file(state_path) != state_hash:
        raise GenerationError(f"Predecessor checkpoint-state hash mismatch: {state_path}")
    if nomination.get("output_checkpoint_sha256") not in (None, checkpoint_hash):
        raise GenerationError(f"Predecessor checkpoint hash differs from nomination: {record_path}")
    if nomination.get("output_checkpoint_state_sha256") not in (None, state_hash):
        raise GenerationError(f"Predecessor state hash differs from nomination: {record_path}")
    state = load_json_object(state_path, "predecessor checkpoint state")
    learner_state = state.get("learner_state")
    if not isinstance(learner_state, dict) or learner_state.get("queries_used") != stage_queries:
        raise GenerationError(f"Predecessor state query count mismatch: {state_path}")
    continuation = state.get("continuation")
    if not isinstance(continuation, dict):
        raise GenerationError(f"Predecessor state lacks continuation provenance: {state_path}")
    if continuation.get("source_checkpoint_path") != record["source_checkpoint_path"]:
        raise GenerationError(f"Predecessor state source path mismatch: {state_path}")
    if continuation.get("source_checkpoint_sha256") != record["source_checkpoint_sha256"]:
        raise GenerationError(f"Predecessor state source hash mismatch: {state_path}")
    if continuation.get("source_checkpoint_state_path") != record["source_checkpoint_state_path"]:
        raise GenerationError(f"Predecessor state source-state path mismatch: {state_path}")
    if continuation.get("source_checkpoint_state_sha256") != record["source_checkpoint_state_sha256"]:
        raise GenerationError(f"Predecessor state source-state hash mismatch: {state_path}")
    if continuation.get("source_checkpoint_size_bytes") != record["source_checkpoint_size_bytes"]:
        raise GenerationError(f"Predecessor state source size mismatch: {state_path}")
    if continuation.get("source_checkpoint_state_size_bytes") != record["source_checkpoint_state_size_bytes"]:
        raise GenerationError(f"Predecessor state source-state size mismatch: {state_path}")

    return {
        "suite": record["suite"],
        "user_task": record["user_task"],
        "injection_task": record["injection_task"],
        "stage": expected_stage,
        "checkpoint_path": str(checkpoint),
        "checkpoint_state_path": str(state_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_size_bytes": checkpoint_metadata.st_size,
        "checkpoint_state_sha256": state_hash,
        "checkpoint_state_size_bytes": state_metadata.st_size,
        "stage_queries_used": stage_queries,
        "cumulative_queries_used": cumulative_queries,
        "chain_id": record["chain_id"],
        "run_id": record["run_id"],
        "result_root": record["result_root"],
        "stage_record_id": str(record_path.resolve(strict=True)),
        "campaign_fingerprint": record["configuration_fingerprint"],
        "predecessor_record_path": str(record_path.resolve()),
        "source_descriptor": copy.deepcopy(prior),
    }


def validate_canonical_nomination(
    nomination: dict[str, Any], base_record: dict[str, Any]
) -> None:
    """Require the official deterministic predecessor nomination before reading it."""
    expected_stage = nomination.get("stage")
    if not isinstance(expected_stage, str):
        raise GenerationError("Predecessor nomination has an invalid stage")
    prior = nomination.get("source_descriptor")
    if not isinstance(prior, dict):
        raise GenerationError("Predecessor nomination lacks source lineage")
    validate_descriptor_lineage(prior, base_record)
    if prior.get("stage") != predecessor_stage(expected_stage):
        raise GenerationError("Predecessor nomination source stage is inconsistent")
    identifiers = target_identifiers(prior, expected_stage)
    expected_fingerprint = configuration_fingerprint(prior, expected_stage, identifiers)
    expected = {
        "suite": prior["suite"],
        "user_task": prior["user_task"],
        "injection_task": prior["injection_task"],
        "chain_id": prior["chain_id"],
        "stage": expected_stage,
        "source_stage": prior["stage"],
        "configuration_fingerprint": expected_fingerprint,
        **identifiers,
        "output_checkpoint_path": f"{identifiers['result_root']}/run/checkpoint.pt",
        "output_checkpoint_state_path": f"{identifiers['result_root']}/run/checkpoint_state.json",
    }
    for field, value in expected.items():
        if nomination.get(field) != value:
            raise GenerationError(
                f"Predecessor {field} differs from canonical production nomination"
            )
    record_path = Path(nomination["stage_record_path"])
    canonical_records = Path(RECORDS_ROOT) / "records" / "stages"
    if record_path.parent != canonical_records:
        raise GenerationError("Predecessor record path escapes canonical records/stages")


def resolve_predecessor_sources(
    target_stage: str,
    base_records: list[dict[str, Any]],
    predecessor_job_inventory: Path | None,
) -> list[dict[str, Any]]:
    previous_stage = predecessor_stage(target_stage)
    if stage_index(previous_stage) == stage_index("A"):
        return [source_from_stage_a(record) for record in base_records]
    if predecessor_job_inventory is None:
        raise GenerationError(
            f"Stage {target_stage} requires the Stage {previous_stage} generated-job inventory"
        )
    document = load_json_object(predecessor_job_inventory, "predecessor generated-job inventory")
    if document.get("schema_version") != 2 or document.get("target_stage") != previous_stage:
        raise GenerationError("Predecessor generated-job inventory has wrong schema or stage")
    recorded_payload_hash = document.get("job_inventory_payload_sha256")
    validate_sha256(recorded_payload_hash, "predecessor job-inventory payload hash")
    payload = {
        key: value
        for key, value in document.items()
        if key not in {"generated_yaml_sha256", "job_inventory_payload_sha256"}
    }
    actual_payload_hash = hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    if actual_payload_hash != recorded_payload_hash:
        raise GenerationError("Predecessor job-inventory payload hash mismatch")
    predecessor_yaml = predecessor_job_inventory.with_name(
        f"stage-{previous_stage.lower()}-jobs.yaml"
    )
    regular_nonempty(predecessor_yaml, "Predecessor generated YAML")
    verify_generated_pair(
        predecessor_yaml.read_text(encoding="utf-8"),
        predecessor_job_inventory.read_text(encoding="utf-8"),
    )
    nominations = document.get("records")
    if not isinstance(nominations, list) or len(nominations) != len(base_records):
        raise GenerationError("Predecessor generated-job inventory is incomplete")
    by_chain: dict[str, dict[str, Any]] = {}
    for nomination in nominations:
        chain_id = nomination.get("chain_id") if isinstance(nomination, dict) else None
        if not isinstance(chain_id, str) or chain_id in by_chain:
            raise GenerationError("Predecessor generated-job inventory has duplicate or invalid chains")
        by_chain[chain_id] = nomination
    resolved = []
    for base_record in base_records:
        chain_id = base_record["chain_id"]
        nomination = by_chain.get(chain_id)
        if nomination is None:
            raise GenerationError(f"Missing exact Stage {previous_stage} nomination for {chain_id}")
        if nomination.get("stage") != previous_stage:
            raise GenerationError(
                f"Predecessor nomination stage must be {previous_stage}: {chain_id}"
            )
        resolved.append(validate_predecessor_record(nomination, base_record))
    validate_unique(
        resolved,
        ("chain_id", "checkpoint_path", "checkpoint_state_path"),
        f"Stage {previous_stage} predecessors",
    )
    return resolved


def target_identifiers(source: dict[str, Any], target_stage: str) -> dict[str, str]:
    slug = pair_slug(source)
    stage_lower = target_stage.lower()
    run_id = f"{source['chain_id']}-stage-{stage_lower}-{CAMPAIGN_VERSION}"
    job_name = f"autoinject-{slug}-stage-{stage_lower}-{CAMPAIGN_VERSION}"
    experiment_id = (
        f"meta-secalign-qwen3-small-policy-warm-{slug}-stage-{stage_lower}-{CAMPAIGN_VERSION}"
    )
    result_root = (
        f"{RECORDS_ROOT}/chains/{source['chain_id']}/stage-{stage_lower}/{run_id}"
    )
    stage_record_path = (
        f"{RECORDS_ROOT}/records/stages/{source['chain_id']}--stage-{target_stage}--{run_id}.json"
    )
    for label, value in (("Job name", job_name), ("run ID", run_id)):
        if SAFE_COMPONENT.fullmatch(value) is None:
            raise GenerationError(f"Unsafe {label}: {value!r}")
    if len(job_name) > 63:
        raise GenerationError(f"Kubernetes Job name is too long: {job_name}")
    return {
        "job_name": job_name,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "result_root": result_root,
        "stage_record_path": stage_record_path,
    }


def configuration_fingerprint(
    source: dict[str, Any], target_stage: str, identifiers: dict[str, str]
) -> str:
    payload = {
        "campaign_version": CAMPAIGN_VERSION,
        "code_commit": runtime_commit(),
        "query_budget": QUERY_BUDGET,
        "runtime_config_map": RUNTIME_CONFIG_MAP,
        "code_commit": runtime_commit(),
        "stage": target_stage,
        "source_stage": source["stage"],
        "chain_id": source["chain_id"],
        "task": pair_key(source),
        "source_checkpoint_path": source["checkpoint_path"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_checkpoint_state_path": source["checkpoint_state_path"],
        "source_checkpoint_state_sha256": source["checkpoint_state_sha256"],
        "source_cumulative_queries_used": source["cumulative_queries_used"],
        "run_id": identifiers["run_id"],
        "result_root": identifiers["result_root"],
        "stage_record_path": identifiers["stage_record_path"],
    }
    return hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise GenerationError(
            f"Behavioral template occurrence count for {label} is {count}, expected 1"
        )
    return text.replace(old, new)


def replace_exact_count(text: str, old: str, new: str, count: int, label: str) -> str:
    actual = text.count(old)
    if actual != count:
        raise GenerationError(
            f"Behavioral template occurrence count for {label} is {actual}, expected {count}"
        )
    return text.replace(old, new)


def load_behavioral_template(path: Path) -> dict[str, Any]:
    try:
        documents = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError) as error:
        raise GenerationError(f"Cannot read behavioral template {path}: {error}") from error
    if len(documents) != 1 or not isinstance(documents[0], dict):
        raise GenerationError("Behavioral template must contain exactly one YAML document")
    template = documents[0]
    if template.get("kind") != "Job" or template.get("metadata", {}).get("name") != "autoinject-travel-u19-i5-stage-b-r02":
        raise GenerationError("Unexpected behavioral template Job")
    pod_spec = template["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    if container["resources"]["requests"] != container["resources"]["limits"]:
        raise GenerationError("Behavioral template resource requests and limits differ")
    if container["resources"]["requests"].get("nvidia.com/gpu") != 1:
        raise GenerationError("Behavioral template does not request exactly one GPU")
    runtime = next(volume for volume in pod_spec["volumes"] if volume["name"] == "runtime")
    if runtime["configMap"]["name"] != RUNTIME_CONFIG_MAP:
        raise GenerationError("Behavioral template runtime ConfigMap differs")
    return template


def verify_rendered_runtime_commit(
    job: dict[str, Any], index_record: dict[str, Any], commit: str
) -> None:
    """Ensure every runtime commit use was substituted from the single source."""
    init_shell = job["spec"]["template"]["spec"]["initContainers"][0]["args"][0]
    runtime_shell = job["spec"]["template"]["spec"]["containers"][0]["args"][0]
    if f"COMMIT={commit}" not in init_shell:
        raise GenerationError("Rendered Job init container has the wrong runtime commit")
    if 'checkout --detach "${COMMIT}"' not in init_shell or 'rev-parse HEAD' not in init_shell:
        raise GenerationError("Rendered Job does not verify its detached runtime commit")
    if f"CODE_COMMIT={commit}" not in runtime_shell:
        raise GenerationError("Rendered Job runtime has the wrong CODE_COMMIT")
    if job.get("metadata", {}).get("annotations", {}).get(
        "autoinject.ucr.edu/code-commit"
    ) != commit:
        raise GenerationError("Rendered Job metadata has the wrong runtime commit")
    rendered = json.dumps(job, sort_keys=True)
    if TEMPLATE_RUNTIME_COMMIT != commit and TEMPLATE_RUNTIME_COMMIT in rendered:
        raise GenerationError("Rendered Job retains the old template runtime commit")
    if index_record.get("code_commit") != commit:
        raise GenerationError("Generated inventory record has the wrong runtime commit")
    identifiers = {
        field: index_record[field]
        for field in ("job_name", "run_id", "experiment_id", "result_root", "stage_record_path")
    }
    if index_record.get("configuration_fingerprint") != configuration_fingerprint(
        index_record["source_descriptor"], index_record["stage"], identifiers
    ):
        raise GenerationError("Generated fingerprint does not use the runtime commit")


def render_job(
    template: dict[str, Any], source: dict[str, Any], target_stage: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    commit = runtime_commit()
    source_stage = predecessor_stage(target_stage)
    if source["stage"] != source_stage:
        raise GenerationError("Resolved source stage does not immediately precede target")
    identifiers = target_identifiers(source, target_stage)
    fingerprint = configuration_fingerprint(source, target_stage, identifiers)
    job = copy.deepcopy(template)
    job["metadata"]["name"] = identifiers["job_name"]
    job["metadata"].setdefault("annotations", {})[
        "autoinject.ucr.edu/code-commit"
    ] = commit
    job["metadata"]["labels"]["autoinject-run-kind"] = (
        f"continuation-stage-{target_stage.lower()}-{CAMPAIGN_VERSION}"
    )
    job["metadata"]["labels"]["autoinject-stage"] = target_stage.lower()
    job["metadata"]["labels"]["autoinject-suite"] = source["suite"]
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["name"] = f"stage-{target_stage.lower()}"
    init_container = job["spec"]["template"]["spec"]["initContainers"][0]
    init_shell = replace_once(
        init_container["args"][0],
        f"COMMIT={TEMPLATE_RUNTIME_COMMIT}",
        f"COMMIT={commit}",
        "init-container runtime commit",
    )
    init_container["args"][0] = init_shell
    shell = container["args"][0]
    shell = replace_once(
        shell,
        f"CODE_COMMIT={TEMPLATE_RUNTIME_COMMIT}",
        f"CODE_COMMIT={commit}",
        "runtime CODE_COMMIT",
    )
    shell = replace_once(
        shell,
        "SOURCE_CHECKPOINT=/workspace/results/autoinject-qwen3-small/l40-cold-baseline-v1/indexed-remaining-31-attempt-02/travel-u19-i5-attempt-02/run/checkpoint.pt",
        'SOURCE_CHECKPOINT="${AI_SOURCE_CHECKPOINT}"',
        "source checkpoint",
    )
    shell = replace_once(
        shell,
        "SOURCE_STATE=/workspace/results/autoinject-qwen3-small/l40-cold-baseline-v1/indexed-remaining-31-attempt-02/travel-u19-i5-attempt-02/run/checkpoint_state.json",
        'SOURCE_STATE="${AI_SOURCE_STATE}"',
        "source state",
    )
    shell = replace_once(
        shell,
        "SOURCE_SHA256=b5fbf2a24ca23aed07b45fd0991b50a81e0abc34a6202760b73888a02af1a746",
        'SOURCE_SHA256="${AI_SOURCE_SHA256}"\n          SOURCE_SIZE="${AI_SOURCE_SIZE}"\n          SOURCE_STATE_SHA256="${AI_SOURCE_STATE_SHA256}"\n          SOURCE_STATE_SIZE="${AI_SOURCE_STATE_SIZE}"',
        "source hash",
    )
    shell = replace_once(
        shell,
        "CHAIN_ID=travel-u19-i5-chain-001",
        'CHAIN_ID="${AI_CHAIN_ID}"',
        "chain ID",
    )
    shell = replace_once(
        shell,
        "RUN_ID=travel-u19-i5-chain-001-stage-b-r02",
        'RUN_ID="${AI_RUN_ID}"',
        "run ID",
    )
    shell = replace_once(
        shell,
        "RESULT_ROOT=${RECORDS_ROOT}/chains/${CHAIN_ID}/stage-b/${RUN_ID}",
        'RESULT_ROOT="${AI_RESULT_ROOT}"',
        "result root",
    )
    shell = replace_once(
        shell,
        "STAGE_RECORD=${RECORDS_ROOT}/records/stages/${CHAIN_ID}--stage-B--${RUN_ID}.json",
        'STAGE_RECORD="${AI_STAGE_RECORD}"',
        "stage record",
    )
    shell = replace_once(
        shell,
        "import json,sys; assert json.load(open(sys.argv[1]))[\"learner_state\"][\"queries_used\"] == 260",
        "import json,sys; state=json.load(open(sys.argv[1])); assert state[\"learner_state\"][\"queries_used\"] == int(sys.argv[2])",
        "source state query validation",
    )
    shell = replace_once(
        shell,
        "exp_ident=meta-secalign-qwen3-small-policy-warm-travel-u19-i5-stage-b-r02",
        'exp_ident="${AI_EXPERIMENT_ID}"',
        "experiment ID",
    )
    shell = replace_once(
        shell,
        "suite=travel 'user_tasks=[user_task_19]' 'injection_tasks=[injection_task_5]'",
        'suite="${AI_SUITE}" "user_tasks=[${AI_USER_TASK}]" "injection_tasks=[${AI_INJECTION_TASK}]"',
        "task identity",
    )
    shell = replace_once(
        shell,
        "chain_id=\"${CHAIN_ID}\" stage=B run_id=\"${RUN_ID}\" source_stage=A",
        'chain_id="${CHAIN_ID}" stage="${AI_STAGE}" run_id="${RUN_ID}" source_stage="${AI_SOURCE_STAGE}"',
        "stage metadata",
    )
    shell = replace_once(
        shell,
        "source_cumulative_queries_used=260 git_commit=\"${CODE_COMMIT}\"",
        'source_cumulative_queries_used="${AI_SOURCE_CUMULATIVE}" git_commit="${CODE_COMMIT}"',
        "source cumulative queries",
    )
    shell = replace_once(
        shell,
        'test "$(sha256sum "${SOURCE_CHECKPOINT}" | awk \'{print $1}\')" = "${SOURCE_SHA256}"',
        'test "$(stat -c %s "${SOURCE_CHECKPOINT}")" = "${SOURCE_SIZE}"\n          test "$(sha256sum "${SOURCE_CHECKPOINT}" | awk \'{print $1}\')" = "${SOURCE_SHA256}"\n          test "$(stat -c %s "${SOURCE_STATE}")" = "${SOURCE_STATE_SIZE}"\n          test "$(sha256sum "${SOURCE_STATE}" | awk \'{print $1}\')" = "${SOURCE_STATE_SHA256}"',
        "source checkpoint and state integrity",
    )
    shell = replace_once(
        shell,
        '"${PYTHON}" -c \'import json,sys; state=json.load(open(sys.argv[1])); assert state["learner_state"]["queries_used"] == int(sys.argv[2])\' "${SOURCE_STATE}"',
        '"${PYTHON}" -c \'import json,sys; state=json.load(open(sys.argv[1])); assert state["learner_state"]["queries_used"] == int(sys.argv[2])\' "${SOURCE_STATE}" "${AI_SOURCE_STAGE_QUERIES}"',
        "source state query argument",
    )
    shell = replace_once(
        shell,
        'source_checkpoint_path="${SOURCE_CHECKPOINT}" \\\n  continuation_records_root="${RECORDS_ROOT}"',
        'source_checkpoint_path="${SOURCE_CHECKPOINT}" source_checkpoint_sha256="${SOURCE_SHA256}" \\\n  source_checkpoint_size_bytes="${SOURCE_SIZE}" source_stage_queries_used="${AI_SOURCE_STAGE_QUERIES}" \\\n  source_checkpoint_state_path="${SOURCE_STATE}" source_checkpoint_state_sha256="${SOURCE_STATE_SHA256}" \\\n  source_checkpoint_state_size_bytes="${SOURCE_STATE_SIZE}" \\\n  continuation_records_root="${RECORDS_ROOT}" result_root="${RESULT_ROOT}" \\\n  stage_record_path="${STAGE_RECORD}" configuration_fingerprint="${AI_CONFIGURATION_FINGERPRINT}"',
        "source provenance arguments",
    )
    shell = replace_once(
        shell,
        '2>&1 | tee /work/stage-b-stdout.log\nstatus=${PIPESTATUS[0]}\nset -e\ntest "${status}" -eq 0',
        '2>&1 | tee "${RUN_DIR}/stdout.log"\npipeline_status=("${PIPESTATUS[@]}")\nset -e\ntest "${#pipeline_status[@]}" -eq 2\ntest "${pipeline_status[0]}" -eq 0\ntest "${pipeline_status[1]}" -eq 0',
        "pipeline status capture",
    )
    shell = replace_once(
        shell,
        'test -s "${STAGE_RECORD}"\ncp /work/stage-b-stdout.log "${RUN_DIR}/stdout.log"\ntest -s "${RUN_DIR}/stdout.log"',
        '"${PYTHON}" - "${RUN_DIR}/checkpoint.pt" "${RUN_DIR}/checkpoint_state.json" <<\'PY\'\nimport json\nimport sys\nimport torch\ncheckpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True, mmap=True)\nassert isinstance(checkpoint, dict)\nassert isinstance(checkpoint.get("policy_state_dict"), dict) and checkpoint["policy_state_dict"]\nassert all(torch.is_tensor(value) for value in checkpoint["policy_state_dict"].values())\nassert isinstance(checkpoint.get("model_config"), dict)\nwith open(sys.argv[2], encoding="utf-8") as handle:\n    state = json.load(handle)\nassert isinstance(state, dict)\nassert isinstance(state.get("learner_state"), dict)\nqueries = state["learner_state"].get("queries_used")\nassert isinstance(queries, int) and not isinstance(queries, bool) and 0 <= queries <= 260\nassert isinstance(state.get("experiment_reporting"), dict)\nassert isinstance(state.get("continuation"), dict)\nPY\ntest -s "${RUN_DIR}/stage-publication-request.json"\ntest -s "${RUN_DIR}/stdout.log"\ntest ! -e "${STAGE_RECORD}"\nexec "${PYTHON}" -m rlpi.agentdojo.continuation_reporting \\\n  --publish-request "${RUN_DIR}/stage-publication-request.json" \\\n  --stdout-log "${RUN_DIR}/stdout.log"',
        "deferred completion publication",
    )
    container["args"][0] = shell

    dynamic_environment = {
        "AI_SOURCE_CHECKPOINT": source["checkpoint_path"],
        "AI_SOURCE_STATE": source["checkpoint_state_path"],
        "AI_SOURCE_SHA256": source["checkpoint_sha256"],
        "AI_SOURCE_SIZE": str(source["checkpoint_size_bytes"]),
        "AI_SOURCE_STATE_SHA256": source["checkpoint_state_sha256"],
        "AI_SOURCE_STATE_SIZE": str(source["checkpoint_state_size_bytes"]),
        "AI_SOURCE_STAGE_QUERIES": str(source["stage_queries_used"]),
        "AI_SOURCE_CUMULATIVE": str(source["cumulative_queries_used"]),
        "AI_CHAIN_ID": source["chain_id"],
        "AI_RUN_ID": identifiers["run_id"],
        "AI_RESULT_ROOT": identifiers["result_root"],
        "AI_STAGE_RECORD": identifiers["stage_record_path"],
        "AI_STAGE": target_stage,
        "AI_SOURCE_STAGE": source_stage,
        "AI_EXPERIMENT_ID": identifiers["experiment_id"],
        "AI_SUITE": source["suite"],
        "AI_USER_TASK": source["user_task"],
        "AI_INJECTION_TASK": source["injection_task"],
        "AI_CONFIGURATION_FINGERPRINT": fingerprint,
    }
    container["env"].extend(
        {"name": name, "value": value} for name, value in dynamic_environment.items()
    )

    index_record = {
        "suite": source["suite"],
        "user_task": source["user_task"],
        "injection_task": source["injection_task"],
        "chain_id": source["chain_id"],
        "source_stage": source_stage,
        "source_checkpoint_path": source["checkpoint_path"],
        "source_checkpoint_state_path": source["checkpoint_state_path"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_checkpoint_size_bytes": source["checkpoint_size_bytes"],
        "source_checkpoint_state_sha256": source["checkpoint_state_sha256"],
        "source_checkpoint_state_size_bytes": source["checkpoint_state_size_bytes"],
        "source_stage_queries_used": source["stage_queries_used"],
        "source_cumulative_queries_used": source["cumulative_queries_used"],
        "predecessor_record_path": source["predecessor_record_path"],
        "source_descriptor": copy.deepcopy(source),
        "stage": target_stage,
        "query_budget": QUERY_BUDGET,
        **identifiers,
        "output_checkpoint_path": f"{identifiers['result_root']}/run/checkpoint.pt",
        "output_checkpoint_state_path": f"{identifiers['result_root']}/run/checkpoint_state.json",
        "output_checkpoint_sha256": None,
        "output_checkpoint_state_sha256": None,
        "configuration_fingerprint": fingerprint,
        "code_commit": commit,
    }
    verify_rendered_runtime_commit(job, index_record, commit)
    return job, index_record


def validate_generated(
    jobs: list[dict[str, Any]], index_records: list[dict[str, Any]], target_stage: str
) -> None:
    if len(jobs) != 32 or len(index_records) != 32:
        raise GenerationError("Generation must produce exactly 32 Jobs and index records")
    if {pair_key(record) for record in index_records} != expected_pairs():
        raise GenerationError("Generated pair set differs from the established 32")
    for field in (
        "job_name",
        "run_id",
        "experiment_id",
        "result_root",
        "stage_record_path",
        "chain_id",
    ):
        values = [record[field] for record in index_records]
        if len(set(values)) != len(values):
            raise GenerationError(f"Generated records contain duplicate {field}")
    names = [job["metadata"]["name"] for job in jobs]
    if names != [record["job_name"] for record in index_records]:
        raise GenerationError("Generated Job names disagree with the index")
    for record in index_records:
        if record["stage"] != target_stage or record["source_stage"] != predecessor_stage(target_stage):
            raise GenerationError("Generated stage ordering is inconsistent")
        if record["query_budget"] != QUERY_BUDGET:
            raise GenerationError("Generated Job does not use a fresh 260-query budget")


def render_yaml(jobs: list[dict[str, Any]]) -> str:
    return yaml.dump_all(
        jobs,
        Dumper=LiteralDumper,
        explicit_start=True,
        sort_keys=False,
        width=1000,
    )


def job_inventory_payload(
    records: list[dict[str, Any]], target_stage: str, template_path: Path
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "target_stage": target_stage,
        "source_stage": predecessor_stage(target_stage),
        "query_budget": QUERY_BUDGET,
        "behavioral_template": str(template_path.relative_to(REPOSITORY_ROOT)),
        "runtime_config_map": RUNTIME_CONFIG_MAP,
        "code_commit": runtime_commit(),
        "records": records,
    }


def render_job_inventory(payload: dict[str, Any], yaml_sha256: str) -> str:
    payload_hash = hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    document = {
        **payload,
        "job_inventory_payload_sha256": payload_hash,
        "generated_yaml_sha256": yaml_sha256,
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def verify_generated_pair(manifest: str, inventory: str) -> None:
    document = json.loads(inventory)
    yaml_hash = hashlib.sha256(manifest.encode()).hexdigest()
    if document.get("generated_yaml_sha256") != yaml_hash:
        raise GenerationError("Generated YAML/job-inventory SHA-256 mismatch")
    payload = {
        key: value
        for key, value in document.items()
        if key not in {"generated_yaml_sha256", "job_inventory_payload_sha256"}
    }
    payload_hash = hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    if document.get("job_inventory_payload_sha256") != payload_hash:
        raise GenerationError("Generated job-inventory payload SHA-256 mismatch")
    jobs = list(yaml.safe_load_all(manifest))
    if any(
        job.get("metadata", {}).get("annotations", {}).get(
            "autoinject.ucr.edu/job-inventory-payload-sha256"
        )
        != payload_hash
        for job in jobs
    ):
        raise GenerationError("Generated YAML does not nominate its job inventory")


def write_deterministic(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def generate(
    *,
    target_stage: str,
    stage_a_inventory: Path,
    template_path: Path,
    predecessor_job_inventory: Path | None,
    enforce_runtime_pin: bool = True,
) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]:
    if enforce_runtime_pin:
        runtime_commit()
    if stage_index(target_stage) == stage_index("A"):
        raise GenerationError("Stage A is not generated by the continuation generator")
    base_records = load_stage_a_inventory(stage_a_inventory)
    sources = resolve_predecessor_sources(
        target_stage, base_records, predecessor_job_inventory
    )
    template = load_behavioral_template(template_path)
    rendered = [render_job(template, source, target_stage) for source in sources]
    jobs = [item[0] for item in rendered]
    index_records = [item[1] for item in rendered]
    validate_generated(jobs, index_records, target_stage)
    payload = job_inventory_payload(index_records, target_stage, template_path)
    if payload["code_commit"] != runtime_commit():
        raise GenerationError("Generated inventory has the wrong runtime commit")
    payload_hash = hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    for job in jobs:
        job["metadata"].setdefault("annotations", {})[
            "autoinject.ucr.edu/job-inventory-payload-sha256"
        ] = payload_hash
    manifest = render_yaml(jobs)
    inventory = render_job_inventory(
        payload, hashlib.sha256(manifest.encode()).hexdigest()
    )
    verify_generated_pair(manifest, inventory)
    return (
        manifest,
        inventory,
        jobs,
        index_records,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, help="Target stage letter after A")
    parser.add_argument("--stage-a-inventory", type=Path, default=DEFAULT_STAGE_A_INVENTORY)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--predecessor-job-inventory", type=Path
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job-inventory-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        target_stage = args.stage
        stage_index(target_stage)
        output = args.output or (
            REPOSITORY_ROOT / f"continuation/stage-{target_stage.lower()}-jobs.yaml"
        )
        index_output = args.job_inventory_output or (
            REPOSITORY_ROOT
            / f"continuation/stage-{target_stage.lower()}-job-inventory.json"
        )
        predecessor_inventory = args.predecessor_job_inventory
        if stage_index(target_stage) > stage_index("B") and predecessor_inventory is None:
            previous = predecessor_stage(target_stage).lower()
            predecessor_inventory = (
                REPOSITORY_ROOT / f"continuation/stage-{previous}-job-inventory.json"
            )
        manifest, inventory, jobs, _ = generate(
            target_stage=target_stage,
            stage_a_inventory=args.stage_a_inventory,
            template_path=args.template,
            predecessor_job_inventory=predecessor_inventory,
            enforce_runtime_pin=True,
        )
        write_deterministic(output, manifest)
        write_deterministic(index_output, inventory)
    except (GenerationError, OSError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"WROTE_CONTINUATION_JOBS={output}")
    print(f"WROTE_CONTINUATION_JOB_INVENTORY={index_output}")
    print(f"GENERATED_JOB_COUNT={len(jobs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
