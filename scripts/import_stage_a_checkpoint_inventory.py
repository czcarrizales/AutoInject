#!/usr/bin/env python3
"""Validate Stage A inspection JSONL and create an immutable inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, TextIO


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = REPOSITORY_ROOT / "continuation/stage-a-source-selection.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "continuation/stage-a-inventory-v3.json"
INSPECTION_JOB_NAME = "autoinject-stage-a-checkpoint-hash-inspection-v3"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED_RECORD_FIELDS = {
    "suite",
    "user_task",
    "injection_task",
    "checkpoint_path",
    "checkpoint_state_path",
    "checkpoint_state_sha256",
    "checkpoint_state_size_bytes",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "queries_used",
    "chain_id",
}


class InventoryError(ValueError):
    """Raised when inspection output cannot form a complete inventory."""


def expected_pairs() -> set[tuple[str, str, str]]:
    return {
        *(
            ("workspace", f"user_task_{user}", f"injection_task_{injection}")
            for user in (38, 39)
            for injection in range(10, 14)
        ),
        *(
            ("banking", f"user_task_{user}", f"injection_task_{injection}")
            for user in (14, 15)
            for injection in range(5, 9)
        ),
        *(
            ("travel", f"user_task_{user}", f"injection_task_{injection}")
            for user in (18, 19)
            for injection in range(3, 7)
        ),
        *(
            ("slack", f"user_task_{user}", f"injection_task_{injection}")
            for user in (19, 20)
            for injection in range(2, 6)
        ),
    }


def pair_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return record["suite"], record["user_task"], record["injection_task"]


def load_selection(path: Path) -> list[dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InventoryError(f"Cannot read source selection {path}: {error}") from error
    if not isinstance(document, dict):
        raise InventoryError("Source selection must be a JSON object")
    if document.get("schema_version") != 1 or document.get("source_stage") != "A":
        raise InventoryError("Unsupported source-selection schema or stage")
    records = document.get("records")
    if not isinstance(records, list) or len(records) != 32:
        raise InventoryError("Source selection must contain exactly 32 records")
    if not all(isinstance(record, dict) for record in records):
        raise InventoryError("Every source selection must be a JSON object")

    keys = [pair_key(record) for record in records]
    if len(set(keys)) != len(keys):
        raise InventoryError("Source selection contains duplicate task pairs")
    if set(keys) != expected_pairs():
        raise InventoryError("Source selection differs from the established 32 pairs")
    for field in ("checkpoint_path", "checkpoint_state_path", "chain_id"):
        values = [record.get(field) for record in records]
        if any(not isinstance(value, str) or not value for value in values):
            raise InventoryError(f"Source selection has an invalid {field}")
        if len(set(values)) != len(values):
            raise InventoryError(f"Source selection contains duplicate {field} values")
    return records


def read_jsonl(stream: TextIO) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(stream, 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise InventoryError(
                f"Inspection output line {line_number} is not JSON: {error}"
            ) from error
        if not isinstance(record, dict):
            raise InventoryError(
                f"Inspection output line {line_number} is not a JSON object"
            )
        records.append(record)
    if len(records) != 32:
        raise InventoryError(
            f"Inspection output must contain exactly 32 records, found {len(records)}"
        )
    return records


def validate_records(
    records: list[dict[str, Any]], selections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    selection_by_pair = {pair_key(record): record for record in selections}
    observed_pairs: list[tuple[str, str, str]] = []
    observed_paths: set[str] = set()
    observed_state_paths: set[str] = set()
    observed_chains: set[str] = set()
    observed_hashes: set[str] = set()
    observed_state_hashes: set[str] = set()

    for index, record in enumerate(records, 1):
        if set(record) != REQUIRED_RECORD_FIELDS:
            missing = sorted(REQUIRED_RECORD_FIELDS - set(record))
            extra = sorted(set(record) - REQUIRED_RECORD_FIELDS)
            raise InventoryError(
                f"Record {index} has invalid fields: missing={missing}, extra={extra}"
            )
        key = pair_key(record)
        observed_pairs.append(key)
        selection = selection_by_pair.get(key)
        if selection is None:
            raise InventoryError(f"Unexpected task pair in record {index}: {key}")

        for field in ("checkpoint_path", "checkpoint_state_path", "chain_id"):
            if record[field] != selection[field]:
                raise InventoryError(
                    f"Record {index} {field} differs from selected source: "
                    f"{record[field]!r} != {selection[field]!r}"
                )
        if record["queries_used"] != selection["expected_queries_used"]:
            raise InventoryError(
                f"Record {index} queries_used differs from finalized report"
            )
        known_hash = selection.get("known_checkpoint_sha256")
        if known_hash is not None and record["checkpoint_sha256"] != known_hash:
            raise InventoryError(
                f"Record {index} hash differs from retained inspection evidence"
            )

        digest = record["checkpoint_sha256"]
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise InventoryError(f"Record {index} has an invalid SHA-256")
        state_digest = record["checkpoint_state_sha256"]
        if not isinstance(state_digest, str) or SHA256.fullmatch(state_digest) is None:
            raise InventoryError(f"Record {index} has an invalid checkpoint-state SHA-256")
        size = record["checkpoint_size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise InventoryError(f"Record {index} has an invalid checkpoint size")
        state_size = record["checkpoint_state_size_bytes"]
        if isinstance(state_size, bool) or not isinstance(state_size, int) or state_size <= 0:
            raise InventoryError(f"Record {index} has an invalid checkpoint-state size")
        queries = record["queries_used"]
        if isinstance(queries, bool) or not isinstance(queries, int) or queries <= 0:
            raise InventoryError(f"Record {index} has invalid queries_used")

        if record["checkpoint_path"] in observed_paths:
            raise InventoryError("Inspection output contains duplicate checkpoint paths")
        if record["checkpoint_state_path"] in observed_state_paths:
            raise InventoryError("Inspection output contains duplicate state paths")
        if record["chain_id"] in observed_chains:
            raise InventoryError("Inspection output contains duplicate chain IDs")
        if record["checkpoint_sha256"] in observed_hashes:
            raise InventoryError("Inspection output contains duplicate checkpoint hashes")
        if record["checkpoint_state_sha256"] in observed_state_hashes:
            raise InventoryError("Inspection output contains duplicate checkpoint-state hashes")
        observed_paths.add(record["checkpoint_path"])
        observed_state_paths.add(record["checkpoint_state_path"])
        observed_chains.add(record["chain_id"])
        observed_hashes.add(record["checkpoint_sha256"])
        observed_state_hashes.add(record["checkpoint_state_sha256"])

    if len(set(observed_pairs)) != len(observed_pairs):
        raise InventoryError("Inspection output contains duplicate task pairs")
    if set(observed_pairs) != expected_pairs():
        missing = sorted(expected_pairs() - set(observed_pairs))
        extra = sorted(set(observed_pairs) - expected_pairs())
        raise InventoryError(
            f"Inspection output pair set is incomplete: missing={missing}, extra={extra}"
        )

    by_pair = {pair_key(record): record for record in records}
    normalized = []
    for selection in selections:
        record = by_pair[pair_key(selection)]
        normalized_record = {
                "suite": record["suite"],
                "user_task": record["user_task"],
                "injection_task": record["injection_task"],
                "stage": "A",
                "run_id": Path(record["checkpoint_path"]).parent.parent.name,
                "result_root": str(Path(record["checkpoint_path"]).parent.parent),
                "stage_record_id": f"stage-a-inventory-v3.json#{record['chain_id']}",
                "checkpoint_path": record["checkpoint_path"],
                "checkpoint_state_path": record["checkpoint_state_path"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "checkpoint_size_bytes": record["checkpoint_size_bytes"],
                "checkpoint_state_sha256": record["checkpoint_state_sha256"],
                "checkpoint_state_size_bytes": record["checkpoint_state_size_bytes"],
                "cumulative_queries_used": record["queries_used"],
                "chain_id": record["chain_id"],
            }
        normalized_record["campaign_fingerprint"] = hashlib.sha256(
            (
                json.dumps(normalized_record, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        normalized.append(normalized_record)
    return normalized


def render_inventory(records: list[dict[str, Any]]) -> str:
    document = {
        "schema_version": 2,
        "source_stage": "A",
        "inspection_job_name": INSPECTION_JOB_NAME,
        "records": records,
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def write_immutable(path: Path, content: str) -> None:
    """Atomically create *path* without replacing a concurrent publisher."""
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
        try:
            os.link(temporary, path)
        except FileExistsError:
            try:
                existing = path.read_text(encoding="utf-8")
            except OSError as error:
                raise InventoryError(
                    f"Cannot read existing inventory {path}: {error}"
                ) from error
            if existing != content:
                raise InventoryError(f"Refusing to overwrite different inventory: {path}")
            return
        directory_descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise InventoryError(f"Cannot write inventory {path}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inspection_jsonl", type=Path)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        selections = load_selection(args.selection)
        with args.inspection_jsonl.open(encoding="utf-8") as stream:
            records = read_jsonl(stream)
        inventory = validate_records(records, selections)
        write_immutable(args.output, render_inventory(inventory))
    except (InventoryError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"WROTE_STAGE_A_INVENTORY={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
