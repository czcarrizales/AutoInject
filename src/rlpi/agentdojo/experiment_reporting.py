"""Passive query and pipeline reporting for adaptive attack experiments."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


COUNTER_FIELDS = (
    "victim_queries_outer",
    "victim_queries_grpo",
    "feedback_model_calls_outer",
    "feedback_model_calls_grpo",
    "pipeline_runs_attempted",
    "pipeline_runs_completed",
    "pipeline_run_errors",
)


@dataclass
class ExperimentReporting:
    """Accumulate telemetry without influencing experiment control flow."""

    victim_queries_outer: int = 0
    victim_queries_grpo: int = 0
    feedback_model_calls_outer: int = 0
    feedback_model_calls_grpo: int = 0
    pipeline_runs_attempted: int = 0
    pipeline_runs_completed: int = 0
    pipeline_run_errors: int = 0
    cycles: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def victim_queries_total(self) -> int:
        return self.victim_queries_outer + self.victim_queries_grpo

    @property
    def feedback_model_calls_total(self) -> int:
        return (
            self.feedback_model_calls_outer
            + self.feedback_model_calls_grpo
        )

    def record_pipeline_attempt(self, phase: str) -> None:
        self.pipeline_runs_attempted += 1
        if phase == "grpo":
            self.victim_queries_grpo += 1
        else:
            self.victim_queries_outer += 1

    def record_pipeline_completed(self) -> None:
        self.pipeline_runs_completed += 1

    def record_pipeline_error(self) -> None:
        self.pipeline_run_errors += 1

    def record_feedback_model_call(self, phase: str) -> None:
        if phase == "grpo":
            self.feedback_model_calls_grpo += 1
        else:
            self.feedback_model_calls_outer += 1

    def counter_snapshot(self) -> Dict[str, int]:
        return {name: int(getattr(self, name)) for name in COUNTER_FIELDS}

    def record_cycle(
        self, iteration: int, before: Mapping[str, int]
    ) -> None:
        after = self.counter_snapshot()
        cycle = {"iteration": iteration}
        cycle.update(
            {
                name: after[name] - before.get(name, 0)
                for name in COUNTER_FIELDS
            }
        )
        cycle["victim_queries_total"] = (
            cycle["victim_queries_outer"] + cycle["victim_queries_grpo"]
        )
        cycle["feedback_model_calls_total"] = (
            cycle["feedback_model_calls_outer"]
            + cycle["feedback_model_calls_grpo"]
        )
        self.cycles.append(cycle)

    def to_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = self.counter_snapshot()
        state.update(
            {
                "victim_queries_total": self.victim_queries_total,
                "feedback_model_calls_total": self.feedback_model_calls_total,
                "cycles": list(self.cycles),
            }
        )
        return state

    def restore(self, state: Mapping[str, Any]) -> None:
        """Restore additive reporting state; absent fields default to zero."""
        for name in COUNTER_FIELDS:
            setattr(self, name, int(state.get(name, 0)))
        cycles = state.get("cycles", [])
        self.cycles = list(cycles) if isinstance(cycles, list) else []

    @classmethod
    def from_checkpoint(
        cls, checkpoint_state: Mapping[str, Any]
    ) -> "ExperimentReporting":
        reporting = cls()
        state = checkpoint_state.get("experiment_reporting", {})
        if isinstance(state, Mapping):
            reporting.restore(state)
        return reporting


def log_final_experiment_summary(
    logger, reporting: ExperimentReporting, query_budget: int, queries_used: int
) -> None:
    """Log compatibility and passive reporting totals at experiment end."""
    budget_overshoot = max(0, queries_used - query_budget)
    fields = {
        "query_budget": query_budget,
        "queries_used": queries_used,
        **reporting.to_dict(),
        "budget_overshoot": budget_overshoot,
    }
    fields.pop("cycles", None)
    logger.info(
        "Final experiment summary:\n%s",
        "\n".join(f"  {name}: {value}" for name, value in fields.items()),
    )
    if reporting.victim_queries_total != queries_used:
        logger.warning(
            "victim_queries_total (%s) differs from queries_used (%s); "
            "queries_used remains the compatibility-facing total.",
            reporting.victim_queries_total,
            queries_used,
        )
