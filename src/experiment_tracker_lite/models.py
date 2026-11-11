"""Data entities mirroring MLflow Run, RunInfo, and RunData."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RunStatus = Literal["RUNNING", "FINISHED", "FAILED", "KILLED"]

VALID_STATUSES: frozenset[str] = frozenset({"RUNNING", "FINISHED", "FAILED", "KILLED"})


@dataclass(frozen=True)
class RunInfo:
    """Metadata about an experiment run."""

    run_id: str
    run_name: str
    experiment_id: str
    status: RunStatus
    start_time: int
    end_time: int | None = None


@dataclass(frozen=True)
class RunData:
    """Logged parameters, latest metrics, and tags for a run."""

    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Run:
    """A tracked experiment run containing RunInfo and RunData."""

    info: RunInfo
    data: RunData

    @property
    def run_id(self) -> str:
        """Convenience accessor for info.run_id."""
        return self.info.run_id


@dataclass(frozen=True)
class MetricPoint:
    """A single logged metric observation."""

    step: int
    timestamp: int
    value: float
