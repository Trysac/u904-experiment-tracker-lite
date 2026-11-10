"""File-based experiment tracker with a subset of the MLflow fluent API."""

from __future__ import annotations

import os
from pathlib import Path

from .cli import main
from .errors import ErrorCode, TrackerError
from .models import MetricPoint, Run, RunData, RunInfo, RunStatus
from .tracker import ActiveRun, Tracker

__all__ = [
    "ActiveRun",
    "ErrorCode",
    "MetricPoint",
    "Run",
    "RunData",
    "RunInfo",
    "RunStatus",
    "Tracker",
    "TrackerError",
    "__version__",
    "build_site",
    "end_run",
    "get_run",
    "log_artifact",
    "log_metric",
    "log_metrics",
    "log_param",
    "log_params",
    "main",
    "search_runs",
    "set_experiment",
    "set_tag",
    "set_tracking_root",
    "start_run",
]

__version__ = "0.1.0"

_DEFAULT_TRACKER = Tracker()


def set_tracking_root(path: str | os.PathLike[str]) -> None:
    """Set the root directory for the module-level default tracker (default './mlruns')."""
    _DEFAULT_TRACKER.set_tracking_root(path)


def set_experiment(name: str) -> str:
    """Set or create the active experiment by name and return its experiment_id."""
    return _DEFAULT_TRACKER.set_experiment(name)


def start_run(
    run_name: str | None = None,
    *,
    experiment_name: str | None = None,
) -> ActiveRun:
    """Start a new run on the default tracker and return an ActiveRun context manager."""
    return _DEFAULT_TRACKER.start_run(run_name=run_name, experiment_name=experiment_name)


def end_run(status: str = "FINISHED") -> None:
    """End the currently active run on the default tracker."""
    _DEFAULT_TRACKER.end_run(status=status)


def log_param(key: str, value: object) -> None:
    """Log a parameter key-value pair to the active run."""
    _DEFAULT_TRACKER.log_param(key, value)


def log_params(params: dict[str, object]) -> None:
    """Log a dictionary of parameters to the active run."""
    _DEFAULT_TRACKER.log_params(params)


def log_metric(key: str, value: float, step: int = 0) -> None:
    """Log a metric key-value observation at a given step to the active run."""
    _DEFAULT_TRACKER.log_metric(key, value, step=step)


def log_metrics(metrics: dict[str, float], step: int = 0) -> None:
    """Log a dictionary of metrics at a given step to the active run."""
    _DEFAULT_TRACKER.log_metrics(metrics, step=step)


def set_tag(key: str, value: object) -> None:
    """Set a tag key-value pair on the active run."""
    _DEFAULT_TRACKER.set_tag(key, value)


def log_artifact(local_path: str | os.PathLike[str], artifact_path: str | None = None) -> None:
    """Copy a local file or directory into the active run's artifacts directory."""
    _DEFAULT_TRACKER.log_artifact(local_path, artifact_path=artifact_path)


def get_run(run_id: str) -> Run:
    """Retrieve a stored Run by its run_id."""
    return _DEFAULT_TRACKER.get_run(run_id)


def search_runs(
    experiment_names: list[str] | None = None,
    filter_string: str = "",
    order_by: list[str] | None = None,
    max_results: int = 1000,
) -> list[Run]:
    """Search runs using MLflow filter syntax and return a list of Run objects."""
    return _DEFAULT_TRACKER.search_runs(
        experiment_names=experiment_names,
        filter_string=filter_string,
        order_by=order_by,
        max_results=max_results,
    )


def build_site(out_dir: str | os.PathLike[str]) -> list[Path]:
    """Generate a deterministic static HTML comparison site in out_dir."""
    return _DEFAULT_TRACKER.build_site(out_dir)
