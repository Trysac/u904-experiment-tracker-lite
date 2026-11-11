"""Core Tracker class and ActiveRun context manager."""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

from .errors import TrackerError
from .models import MetricPoint, Run, RunInfo
from .search import matches_filter, parse_filter, parse_order_by, sort_runs
from .site import build_site_for_root
from .storage import (
    append_metric_point,
    atomic_write_json,
    copy_artifact_atomically,
    load_run,
    read_json,
    save_run_json,
)
from .validation import (
    MAX_SEARCH_RESULTS,
    validate_experiment_name,
    validate_key,
    validate_metric_value,
    validate_param_value,
    validate_run_name,
    validate_status,
    validate_step,
    validate_tag_value,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_EXPERIMENT_NAME = "Default"
DEFAULT_EXPERIMENT_ID = "0"


def _default_clock() -> int:
    return int(time.time() * 1000)


def _default_new_id() -> str:
    return uuid.uuid4().hex


class ActiveRun:
    """Context manager representing the currently active run."""

    def __init__(self, tracker: Tracker, info: RunInfo) -> None:
        self._tracker = tracker
        self._info = info

    @property
    def info(self) -> RunInfo:
        """Return latest RunInfo for this active run."""
        return self._info

    @property
    def run_id(self) -> str:
        """Return the run_id of this active run."""
        return self._info.run_id

    def _refresh_info(self, info: RunInfo) -> None:
        self._info = info

    def __enter__(self) -> ActiveRun:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._tracker.active_run is self:
            status = "FINISHED" if exc_type is None else "FAILED"
            self._tracker.end_run(status=status)


class Tracker:
    """File-backed experiment tracker managing experiments, runs, metrics, and search."""

    def __init__(
        self,
        root: str | os.PathLike[str] = "./mlruns",
        *,
        clock: Callable[[], int] = _default_clock,
        new_id: Callable[[], str] = _default_new_id,
    ) -> None:
        self._root = Path(root)
        self._clock = clock
        self._new_id = new_id
        self._current_experiment_id: str | None = None
        self._active_run: ActiveRun | None = None
        self._active_run_dir: Path | None = None
        self._active_params: dict[str, str] = {}
        self._active_tags: dict[str, str] = {}

    @property
    def root(self) -> Path:
        """Return the root tracking directory."""
        return self._root

    @property
    def active_run(self) -> ActiveRun | None:
        """Return the currently active run, if any."""
        return self._active_run

    def set_tracking_root(self, path: str | os.PathLike[str]) -> None:
        """Update the tracking root directory. Fails if a run is currently active."""
        if self._active_run is not None:
            raise TrackerError(
                "invalid_state",
                "Cannot change tracking root while a run is active.",
            )
        self._root = Path(path)
        self._current_experiment_id = None

    def _iter_experiments(self) -> list[tuple[str, str, Path]]:
        """Return sorted list of (experiment_id, experiment_name, exp_dir)."""
        if not self._root.exists():
            return []
        exps: list[tuple[str, str, Path]] = []
        for child in sorted(self._root.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or child.name.startswith("."):
                continue
            meta_path = child / "meta.json"
            if meta_path.exists():
                meta = read_json(meta_path)
                exp_id = str(meta["experiment_id"])
                exp_name = str(meta["name"])
                exps.append((exp_id, exp_name, child))
        return exps

    def _ensure_default_experiment(self) -> str:
        """Ensure the default experiment exists and return its experiment_id."""
        exps = self._iter_experiments()
        for exp_id, exp_name, _ in exps:
            if exp_name == DEFAULT_EXPERIMENT_NAME:
                return exp_id
        exp_id = DEFAULT_EXPERIMENT_ID
        exp_dir = self._root / exp_id
        atomic_write_json(
            exp_dir / "meta.json",
            {
                "creation_time": self._clock(),
                "experiment_id": exp_id,
                "name": DEFAULT_EXPERIMENT_NAME,
            },
        )
        return exp_id

    def set_experiment(self, name: str) -> str:
        """Set the active experiment by name, creating it if needed, and return experiment_id."""
        validate_experiment_name(name)
        exps = self._iter_experiments()
        for exp_id, exp_name, _ in exps:
            if exp_name == name:
                self._current_experiment_id = exp_id
                return exp_id
        # Assign deterministic sequential integer ID or '0' for Default if unused
        existing_ids = {exp_id for exp_id, _, _ in exps}
        if name == DEFAULT_EXPERIMENT_NAME and DEFAULT_EXPERIMENT_ID not in existing_ids:
            new_exp_id = DEFAULT_EXPERIMENT_ID
        else:
            numeric_ids = [int(eid) for eid in existing_ids if eid.isdigit()]
            next_num = (max(numeric_ids) + 1) if numeric_ids else 1
            while str(next_num) in existing_ids:
                next_num += 1
            new_exp_id = str(next_num)
        exp_dir = self._root / new_exp_id
        atomic_write_json(
            exp_dir / "meta.json",
            {
                "creation_time": self._clock(),
                "experiment_id": new_exp_id,
                "name": name,
            },
        )
        self._current_experiment_id = new_exp_id
        return new_exp_id

    def start_run(
        self,
        run_name: str | None = None,
        *,
        experiment_name: str | None = None,
    ) -> ActiveRun:
        """Start a new run and return an ActiveRun context manager."""
        if self._active_run is not None:
            raise TrackerError(
                "invalid_state",
                "A run is already active. Call end_run() before starting a new run.",
            )
        validate_run_name(run_name)
        if experiment_name is not None:
            exp_id = self.set_experiment(experiment_name)
        elif self._current_experiment_id is not None:
            exp_id = self._current_experiment_id
        else:
            exp_id = self._ensure_default_experiment()
            self._current_experiment_id = exp_id

        run_id = self._new_id()
        if not isinstance(run_id, str) or not run_id.strip() or "/" in run_id or ".." in run_id:
            raise TrackerError("invalid_parameter_value", "Generated run_id is invalid.")

        effective_name = run_name if run_name is not None else f"run-{run_id[:8]}"
        start_ms = self._clock()
        info = RunInfo(
            run_id=run_id,
            run_name=effective_name,
            experiment_id=exp_id,
            status="RUNNING",
            start_time=start_ms,
            end_time=None,
        )
        run_dir = self._root / exp_id / run_id
        if run_dir.exists():
            raise TrackerError("invalid_state", f"Run ID {run_id!r} already exists.")
        self._active_params = {}
        self._active_tags = {}
        save_run_json(run_dir, info, self._active_params, self._active_tags)
        active = ActiveRun(self, info)
        self._active_run = active
        self._active_run_dir = run_dir
        logger.debug("Started run %s in experiment %s", run_id, exp_id)
        return active

    def _require_active(self) -> tuple[ActiveRun, Path]:
        if self._active_run is None or self._active_run_dir is None:
            raise TrackerError(
                "invalid_state",
                "No active run. Call start_run() before logging or ending a run.",
            )
        return self._active_run, self._active_run_dir

    def end_run(self, status: str = "FINISHED") -> None:
        """End the currently active run with the given status."""
        validated_status = validate_status(status)
        if validated_status == "RUNNING":
            raise TrackerError(
                "invalid_parameter_value",
                "Cannot end a run with status 'RUNNING'.",
            )
        if self._active_run is None or self._active_run_dir is None:
            return
        active = self._active_run
        run_dir = self._active_run_dir
        end_ms = self._clock()
        updated_info = RunInfo(
            run_id=active.info.run_id,
            run_name=active.info.run_name,
            experiment_id=active.info.experiment_id,
            status=validated_status,
            start_time=active.info.start_time,
            end_time=end_ms,
        )
        save_run_json(run_dir, updated_info, self._active_params, self._active_tags)
        active._refresh_info(updated_info)
        self._active_run = None
        self._active_run_dir = None
        self._active_params = {}
        self._active_tags = {}
        logger.debug("Ended run %s with status %s", updated_info.run_id, validated_status)

    def log_param(self, key: str, value: object) -> None:
        """Log a single parameter to the active run."""
        active, run_dir = self._require_active()
        validate_key(key, "param key")
        str_val = validate_param_value(value)
        if key in self._active_params:
            if self._active_params[key] != str_val:
                raise TrackerError(
                    "invalid_parameter_value",
                    f"Changing param value for key {key!r} is not allowed.",
                )
            return
        self._active_params[key] = str_val
        save_run_json(run_dir, active.info, self._active_params, self._active_tags)

    def log_params(self, params: dict[str, object]) -> None:
        """Log multiple parameters to the active run."""
        if not isinstance(params, dict):
            raise TrackerError("invalid_parameter_value", "params must be a dict.")
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        """Log a single metric value at the given step to the active run."""
        _, run_dir = self._require_active()
        validate_key(key, "metric key")
        fval = validate_metric_value(value)
        istep = validate_step(step)
        point = MetricPoint(step=istep, timestamp=self._clock(), value=fval)
        append_metric_point(run_dir, key, point)

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Log multiple metrics at the given step to the active run."""
        if not isinstance(metrics, dict):
            raise TrackerError("invalid_parameter_value", "metrics must be a dict.")
        istep = validate_step(step)
        for k, v in metrics.items():
            self.log_metric(k, v, step=istep)

    def set_tag(self, key: str, value: object) -> None:
        """Set a tag on the active run."""
        active, run_dir = self._require_active()
        validate_key(key, "tag key")
        str_val = validate_tag_value(value)
        self._active_tags[key] = str_val
        save_run_json(run_dir, active.info, self._active_params, self._active_tags)

    def log_artifact(
        self, local_path: str | os.PathLike[str], artifact_path: str | None = None
    ) -> None:
        """Copy a local file or directory into the active run's artifact store."""
        _, run_dir = self._require_active()
        copy_artifact_atomically(run_dir, local_path, artifact_path)

    def get_run(self, run_id: str) -> Run:
        """Load and return a Run by its run_id across all experiments."""
        if not isinstance(run_id, str) or not run_id.strip():
            raise TrackerError("invalid_parameter_value", "run_id must be a non-empty string.")
        for _, _, exp_dir in self._iter_experiments():
            candidate = exp_dir / run_id
            if (candidate / "run.json").exists():
                return load_run(candidate)
        raise TrackerError("resource_does_not_exist", f"Run {run_id!r} does not exist.")

    def search_runs(
        self,
        experiment_names: list[str] | None = None,
        filter_string: str = "",
        order_by: list[str] | None = None,
        max_results: int = 1000,
    ) -> list[Run]:
        """Search runs matching filter_string and ordered by order_by."""
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise TrackerError(
                "invalid_parameter_value",
                "max_results must be a positive integer.",
            )
        if max_results > MAX_SEARCH_RESULTS:
            raise TrackerError(
                "limit_exceeded",
                f"max_results cannot exceed {MAX_SEARCH_RESULTS}.",
            )
        clauses = parse_filter(filter_string)
        order_clauses = parse_order_by(order_by)

        exps = self._iter_experiments()
        if experiment_names is not None:
            if not isinstance(experiment_names, list):
                raise TrackerError(
                    "invalid_parameter_value", "experiment_names must be a list of strings."
                )
            name_set = set(experiment_names)
            target_exps = [exp_dir for _, exp_name, exp_dir in exps if exp_name in name_set]
        elif self._current_experiment_id is not None and not exps:
            target_exps = []
        else:
            target_exps = [exp_dir for _, _, exp_dir in exps]

        matched: list[Run] = []
        for exp_dir in target_exps:
            for child in sorted(exp_dir.iterdir(), key=lambda p: p.name):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                if (child / "run.json").exists():
                    run = load_run(child)
                    if matches_filter(run, clauses):
                        matched.append(run)

        ordered = sort_runs(matched, order_clauses)
        return ordered[:max_results]

    def build_site(self, out_dir: str | os.PathLike[str]) -> list[Path]:
        """Build a static, deterministic HTML comparison site for all experiments in this store."""
        return build_site_for_root(self._root, Path(out_dir))
