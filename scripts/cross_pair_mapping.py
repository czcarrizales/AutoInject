#!/usr/bin/env python3
"""Deterministic Stage A source-to-target mappings for cross-pair warm-starts."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

from scripts import generate_continuation_jobs as continuation


SIBLING_USER_TASKS = {
    "workspace": {
        "user_task_38": "user_task_39",
        "user_task_39": "user_task_38",
    },
    "banking": {
        "user_task_14": "user_task_15",
        "user_task_15": "user_task_14",
    },
    "travel": {
        "user_task_18": "user_task_19",
        "user_task_19": "user_task_18",
    },
    "slack": {
        "user_task_19": "user_task_20",
        "user_task_20": "user_task_19",
    },
}


class CrossPairMappingError(ValueError):
    """Raised when a complete and controlled cross-pair mapping cannot be built."""


def sibling_user_task(suite: str, user_task: str) -> str:
    """Return the only allowed sibling user task within a suite."""
    suite_mapping = SIBLING_USER_TASKS.get(suite)
    if suite_mapping is None:
        raise CrossPairMappingError(f"Unsupported cross-pair suite: {suite!r}")

    sibling = suite_mapping.get(user_task)
    if sibling is None:
        raise CrossPairMappingError(
            f"Unsupported user task for suite {suite!r}: {user_task!r}"
        )
    return sibling


def identity(record: dict[str, Any]) -> tuple[str, str, str]:
    """Return the suite, user-task, and injection-task identity."""
    try:
        return continuation.pair_key(record)
    except KeyError as error:
        raise CrossPairMappingError(
            f"Cross-pair record is missing identity field: {error}"
        ) from error


def build_cross_pair_mappings(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one reciprocal Stage A source assignment for every target pair."""
    materialized = [copy.deepcopy(record) for record in records]
    expected = continuation.expected_pairs()

    if len(materialized) != continuation.INDEXED_COMPLETIONS:
        raise CrossPairMappingError(
            "Cross-pair mapping requires exactly 32 Stage A records"
        )

    by_pair: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in materialized:
        key = identity(record)

        if key not in expected:
            raise CrossPairMappingError(
                f"Unexpected Stage A task identity in cross-pair mapping: {key}"
            )
        if key in by_pair:
            raise CrossPairMappingError(
                f"Duplicate Stage A task identity in cross-pair mapping: {key}"
            )

        by_pair[key] = record

    if set(by_pair) != expected:
        missing = sorted(expected - set(by_pair))
        extra = sorted(set(by_pair) - expected)
        raise CrossPairMappingError(
            f"Stage A pair set is incomplete: missing={missing}, extra={extra}"
        )

    mappings: list[dict[str, Any]] = []

    for index, target in enumerate(
        sorted(materialized, key=continuation.canonical_task_key)
    ):
        target_key = identity(target)
        suite, target_user_task, injection_task = target_key

        source_key = (
            suite,
            sibling_user_task(suite, target_user_task),
            injection_task,
        )
        source = by_pair.get(source_key)

        if source is None:
            raise CrossPairMappingError(
                f"No allowed sibling source exists for target {target_key}"
            )
        if source_key == target_key:
            raise CrossPairMappingError(
                f"Cross-pair mapping accidentally selected the target itself: {target_key}"
            )

        mappings.append(
            {
                "completion_index": index,
                "target": {
                    "suite": suite,
                    "user_task": target_user_task,
                    "injection_task": injection_task,
                },
                "source": {
                    "suite": source["suite"],
                    "user_task": source["user_task"],
                    "injection_task": source["injection_task"],
                },
                "source_descriptor": copy.deepcopy(source),
            }
        )

    target_keys = [
        (
            mapping["target"]["suite"],
            mapping["target"]["user_task"],
            mapping["target"]["injection_task"],
        )
        for mapping in mappings
    ]
    source_keys = [
        (
            mapping["source"]["suite"],
            mapping["source"]["user_task"],
            mapping["source"]["injection_task"],
        )
        for mapping in mappings
    ]

    if len(set(target_keys)) != len(target_keys) or set(target_keys) != expected:
        raise CrossPairMappingError(
            "Cross-pair mapping does not cover every target exactly once"
        )

    if len(set(source_keys)) != len(source_keys) or set(source_keys) != expected:
        raise CrossPairMappingError(
            "Cross-pair mapping does not use every source exactly once"
        )

    source_for_target = dict(zip(target_keys, source_keys))
    for target_key, source_key in source_for_target.items():
        if source_for_target.get(source_key) != target_key:
            raise CrossPairMappingError(
                f"Cross-pair mapping is not reciprocal: {target_key} -> {source_key}"
            )

    return mappings
