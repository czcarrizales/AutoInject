"""Immutable, flat result records for continuation stages."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import subprocess
import tempfile
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "continuation-stage-record/v1"
PUBLICATION_REQUEST_SCHEMA = "continuation-stage-publication-request/v1"
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9._-]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable JSON representation used for configuration hashes."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def detect_gpus() -> tuple[str | None, int]:
    """Return a stable GPU description without making GPU availability mandatory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, 0
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return "; ".join(sorted(set(names))) or None, len(names)


def raw_file(path: Path, *, required: bool) -> dict[str, str | None]:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if required:
            raise FileNotFoundError(f"Required continuation raw artifact is missing: {path}")
        return {"path": None, "sha256": None}
    if not path.is_file() or path.is_symlink() or metadata.st_size <= 0:
        raise ValueError(f"Continuation raw artifact must be a regular nonempty file: {path}")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256_file(path),
        "size_bytes": metadata.st_size,
    }


def validate_record_component(name: str, value: str) -> None:
    if not isinstance(value, str) or value in ("", ".", "..") or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"Unsafe continuation record {name}: {value!r}")


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish source, refusing an existing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        result = renameat2(
            -100, os.fsencode(source), -100, os.fsencode(destination), 1  # RENAME_NOREPLACE
        )
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"Finalized continuation record already exists: {destination}")
        if error not in (errno.ENOSYS, errno.EINVAL):
            raise OSError(error, os.strerror(error), destination)

    # link() is an atomic create-without-overwrite fallback on filesystems lacking renameat2.
    try:
        os.link(source, destination)
    except FileExistsError:
        raise FileExistsError(f"Finalized continuation record already exists: {destination}") from None
    try:
        source.unlink()
    except OSError:
        # The destination is already the immutable publication. Do not report a
        # failed Job after it has become visible.
        pass


def write_stage_publication_request(path: Path, request: Mapping[str, Any]) -> Path:
    """Create a non-completion publication request without replacement."""
    if request.get("schema_version") != PUBLICATION_REQUEST_SCHEMA:
        raise ValueError("Unsupported continuation publication-request schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(dict(request)))
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def finalize_stage_record(
    *,
    records_root: Path,
    chain_id: str,
    stage: str,
    run_id: str,
    suite: str,
    user_task: str,
    injection_task: str,
    initialization_mode: str,
    reference_mode: str,
    source_stage: str | None,
    source_cumulative_queries_used: int,
    git_commit: str | None,
    resolved_config: Mapping[str, Any],
    run_dir: Path,
    stage_start_time: str,
    stage_end_time: str,
    stage_wall_clock_seconds: float,
    stdout_log_path: Path | None = None,
    gpu_name: str | None = None,
    gpu_count: int | None = None,
    expected_record_path: Path | None = None,
    expected_result_root: Path | None = None,
    configuration_fingerprint: str | None = None,
) -> Path:
    """Publish one immutable record after a successfully completed stage."""
    for name, value in (("chain_id", chain_id), ("stage", stage), ("run_id", run_id)):
        validate_record_component(name, value)
    if not isinstance(configuration_fingerprint, str) or not _SHA256.fullmatch(
        configuration_fingerprint
    ):
        raise ValueError("Continuation configuration fingerprint is not a SHA-256")
    run_dir = run_dir.resolve(strict=True)
    result_root = run_dir.parent.resolve(strict=True)
    if (
        expected_result_root is not None
        and result_root != expected_result_root.resolve(strict=True)
    ):
        raise ValueError("Continuation result root differs from the nominated result root")
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_state_path = run_dir / "checkpoint_state.json"
    experience_path = run_dir / "grpo_training_outputs" / "experience_history.jsonl"
    grpo_path = run_dir / "grpo_training_outputs" / "grpo_evaluations.jsonl"
    checkpoint_state = json.loads(checkpoint_state_path.read_text(encoding="utf-8"))
    learner_state = checkpoint_state.get("learner_state")
    reporting = checkpoint_state.get("experiment_reporting")
    if not isinstance(learner_state, dict) or not isinstance(reporting, dict):
        raise ValueError("checkpoint_state.json lacks learner_state or experiment_reporting")
    queries_used = learner_state.get("queries_used")
    if not isinstance(queries_used, int) or queries_used < 0:
        raise ValueError("checkpoint_state.json has invalid learner_state.queries_used")

    continuation = checkpoint_state.get("continuation", {})
    if initialization_mode == "policy_warm":
        if not isinstance(continuation, dict):
            raise ValueError("policy_warm checkpoint lacks continuation provenance")
        source_path = continuation.get("source_checkpoint_path")
        source_hash = continuation.get("source_checkpoint_sha256")
        source_state_path = continuation.get("source_checkpoint_state_path")
        source_state_hash = continuation.get("source_checkpoint_state_sha256")
        source_size = continuation.get("source_checkpoint_size_bytes")
        source_state_size = continuation.get("source_checkpoint_state_size_bytes")
        if not all(
            isinstance(value, str)
            for value in (source_path, source_hash, source_state_path, source_state_hash)
        ):
            raise ValueError("policy_warm checkpoint lacks source checkpoint provenance")
        for state_value, config_key in (
            (source_path, "source_checkpoint_path"),
            (source_hash, "source_checkpoint_sha256"),
            (source_state_path, "source_checkpoint_state_path"),
            (source_state_hash, "source_checkpoint_state_sha256"),
            (source_size, "source_checkpoint_size_bytes"),
            (source_state_size, "source_checkpoint_state_size_bytes"),
        ):
            if state_value != resolved_config.get(config_key):
                raise ValueError(
                    f"checkpoint continuation provenance differs from {config_key}"
                )
    else:
        source_path = source_hash = source_state_path = source_state_hash = None
        source_size = source_state_size = None

    if gpu_count is None:
        gpu_name, gpu_count = detect_gpus()
    raw = {
        "experience_history": raw_file(experience_path, required=True),
        "grpo_evaluations": raw_file(grpo_path, required=True),
        "checkpoint_state": raw_file(checkpoint_state_path, required=True),
        "output_checkpoint": raw_file(checkpoint_path, required=True),
        "stdout_log": raw_file(stdout_log_path, required=False)
        if stdout_log_path is not None
        else {"path": None, "sha256": None},
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "chain_id": chain_id,
        "stage": stage,
        "run_id": run_id,
        "suite": suite,
        "user_task": user_task,
        "injection_task": injection_task,
        "initialization_mode": initialization_mode,
        "reference_mode": reference_mode,
        "source_stage": source_stage,
        "source_checkpoint_path": source_path,
        "source_checkpoint_sha256": source_hash,
        "source_checkpoint_state_path": source_state_path,
        "source_checkpoint_state_sha256": source_state_hash,
        "source_checkpoint_size_bytes": source_size,
        "source_checkpoint_state_size_bytes": source_state_size,
        "result_root": str(result_root),
        "output_checkpoint_path": raw["output_checkpoint"]["path"],
        "output_checkpoint_sha256": raw["output_checkpoint"]["sha256"],
        "experience_history_path": raw["experience_history"]["path"],
        "experience_history_sha256": raw["experience_history"]["sha256"],
        "grpo_evaluations_path": raw["grpo_evaluations"]["path"],
        "grpo_evaluations_sha256": raw["grpo_evaluations"]["sha256"],
        "checkpoint_state_path": raw["checkpoint_state"]["path"],
        "checkpoint_state_sha256": raw["checkpoint_state"]["sha256"],
        "checkpoint_state_size_bytes": raw["checkpoint_state"]["size_bytes"],
        "output_checkpoint_size_bytes": raw["output_checkpoint"]["size_bytes"],
        "stdout_log_path": raw["stdout_log"]["path"],
        "stdout_log_sha256": raw["stdout_log"]["sha256"],
        "stage_query_budget": resolved_config["query_budget"],
        "stage_queries_used": queries_used,
        "cumulative_queries_used": source_cumulative_queries_used + queries_used,
        "stage_start_time": stage_start_time,
        "stage_end_time": stage_end_time,
        "stage_wall_clock_seconds": stage_wall_clock_seconds,
        "git_commit": git_commit,
        "resolved_config_sha256": hashlib.sha256(canonical_json_bytes(resolved_config)).hexdigest(),
        "configuration_fingerprint": configuration_fingerprint,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "completion_status": "completed",
        # Successful runtime finalization does not adjudicate technical invalidation in v1.
        "technical_invalidation": None,
        "technical_invalidation_reason": None,
        "experiment_reporting": reporting,
    }
    stages_directory = (records_root / "records" / "stages").resolve()
    stages_directory.mkdir(parents=True, exist_ok=True)
    destination = stages_directory / f"{chain_id}--stage-{stage}--{run_id}.json"
    if destination.parent.resolve() != stages_directory:
        raise ValueError("Continuation record destination escapes records/stages")
    if expected_record_path is not None and destination != expected_record_path.resolve():
        raise ValueError("Continuation record path differs from the nominated record path")
    if destination.exists():
        raise FileExistsError(f"Finalized continuation record already exists: {destination}")
    directory_descriptor = os.open(destination.parent, os.O_DIRECTORY)
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(record))
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(temporary, destination)
        try:
            os.fsync(directory_descriptor)
        except OSError:
            # Rename is the completion publication point. A durability warning
            # must not make Kubernetes classify the already-published Job failed.
            pass
    finally:
        try:
            os.close(directory_descriptor)
        except OSError:
            pass
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return destination


_PUBLICATION_KEYS = {
    "schema_version",
    "records_root",
    "chain_id",
    "stage",
    "run_id",
    "suite",
    "user_task",
    "injection_task",
    "initialization_mode",
    "reference_mode",
    "source_stage",
    "source_cumulative_queries_used",
    "git_commit",
    "resolved_config",
    "run_dir",
    "result_root",
    "stage_record_path",
    "configuration_fingerprint",
    "stage_start_time",
    "stage_end_time",
    "stage_wall_clock_seconds",
}


def publish_stage_record(request_path: Path, stdout_log_path: Path) -> Path:
    """Validate finalized artifacts and publish the nominated completion record."""
    request_metadata = request_path.lstat()
    if request_path.is_symlink() or not request_path.is_file() or request_metadata.st_size <= 0:
        raise ValueError("Publication request must be a regular nonempty file")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict) or set(request) != _PUBLICATION_KEYS:
        raise ValueError("Publication request has missing or unexpected fields")
    if request["schema_version"] != PUBLICATION_REQUEST_SCHEMA:
        raise ValueError("Unsupported continuation publication-request schema")
    run_dir = Path(request["run_dir"]).resolve(strict=True)
    result_root = Path(request["result_root"]).resolve(strict=True)
    if run_dir != result_root / "run" or run_dir.parent != result_root:
        raise ValueError("Publication request run directory escapes its result root")
    expected_stdout = run_dir / "stdout.log"
    if stdout_log_path.resolve(strict=True) != expected_stdout.resolve(strict=True):
        raise ValueError("stdout path differs from the finalized permanent log path")
    for relative in (
        "checkpoint.pt",
        "checkpoint_state.json",
        "grpo_training_outputs/experience_history.jsonl",
        "grpo_training_outputs/grpo_evaluations.jsonl",
    ):
        artifact = run_dir / relative
        resolved = artifact.resolve(strict=True)
        if resolved != artifact or resolved.parent != artifact.parent.resolve(strict=True):
            raise ValueError(f"Publication artifact escapes nominated run directory: {artifact}")
        raw_file(artifact, required=True)
    return finalize_stage_record(
        records_root=Path(request["records_root"]),
        chain_id=request["chain_id"],
        stage=request["stage"],
        run_id=request["run_id"],
        suite=request["suite"],
        user_task=request["user_task"],
        injection_task=request["injection_task"],
        initialization_mode=request["initialization_mode"],
        reference_mode=request["reference_mode"],
        source_stage=request["source_stage"],
        source_cumulative_queries_used=request["source_cumulative_queries_used"],
        git_commit=request["git_commit"],
        resolved_config=request["resolved_config"],
        run_dir=run_dir,
        stage_start_time=request["stage_start_time"],
        stage_end_time=request["stage_end_time"],
        stage_wall_clock_seconds=request["stage_wall_clock_seconds"],
        stdout_log_path=stdout_log_path,
        expected_record_path=Path(request["stage_record_path"]),
        expected_result_root=result_root,
        configuration_fingerprint=request["configuration_fingerprint"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish a validated continuation record")
    parser.add_argument("--publish-request", type=Path, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    args = parser.parse_args(argv)
    publish_stage_record(args.publish_request, args.stdout_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
