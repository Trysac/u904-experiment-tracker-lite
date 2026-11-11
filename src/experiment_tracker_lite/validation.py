"""Validation utilities enforcing MLflow-compatible limits and formats."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

from .errors import TrackerError
from .models import VALID_STATUSES, RunStatus

MAX_KEY_LENGTH = 250
MAX_PARAM_VALUE_LENGTH = 6000
MAX_TAG_VALUE_LENGTH = 6000
MAX_EXPERIMENT_NAME_LENGTH = 250
MAX_RUN_NAME_LENGTH = 250
MAX_METRIC_POINTS = 100_000
MAX_ARTIFACT_BYTES = 104_857_600
MAX_FILTER_LENGTH = 1000
MAX_SEARCH_RESULTS = 10_000

_KEY_RE = re.compile(r"^[a-zA-Z0-9_\-\. /]+$")


def validate_key(key: str, kind: str = "key") -> str:
    """Validate parameter, metric, or tag key according to MLflow rules."""
    if not isinstance(key, str) or not (1 <= len(key) <= MAX_KEY_LENGTH):
        raise TrackerError(
            "invalid_parameter_value",
            f"Invalid {kind} length: must be between 1 and {MAX_KEY_LENGTH} characters.",
        )
    if not _KEY_RE.match(key):
        raise TrackerError(
            "invalid_parameter_value",
            f"Invalid {kind} {key!r}: only letters, digits, '_', '-', '.', ' ', '/' allowed.",
        )
    segments = [seg for seg in key.split("/") if seg != ""]
    if not segments or any(seg.strip() == ".." or seg == "." for seg in key.split("/")):
        raise TrackerError(
            "invalid_parameter_value",
            f"Invalid {kind} {key!r}: path traversal segments are not allowed.",
        )
    return key


def validate_param_value(value: object) -> str:
    """Convert a parameter value with str() and enforce length limit."""
    str_val = str(value)
    if len(str_val) > MAX_PARAM_VALUE_LENGTH:
        raise TrackerError(
            "invalid_parameter_value",
            f"Param value length {len(str_val)} exceeds maximum of {MAX_PARAM_VALUE_LENGTH}.",
        )
    return str_val


def validate_tag_value(value: object) -> str:
    """Convert a tag value with str() and enforce length limit."""
    str_val = str(value)
    if len(str_val) > MAX_TAG_VALUE_LENGTH:
        raise TrackerError(
            "invalid_parameter_value",
            f"Tag value length {len(str_val)} exceeds maximum of {MAX_TAG_VALUE_LENGTH}.",
        )
    return str_val


def validate_metric_value(value: float | int) -> float:
    """Validate that a metric value is a finite number or NaN (not infinity or boolean)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackerError(
            "invalid_parameter_value",
            "Metric value must be a float or int.",
        )
    fval = float(value)
    if math.isinf(fval):
        raise TrackerError(
            "invalid_parameter_value",
            "Metric value must be finite or NaN, not infinity.",
        )
    return fval


def validate_step(step: int) -> int:
    """Validate that a metric step is an integer."""
    if isinstance(step, bool) or not isinstance(step, int):
        raise TrackerError(
            "invalid_parameter_value",
            "Metric step must be an integer.",
        )
    return step


def validate_status(status: str) -> RunStatus:
    """Validate run status string."""
    if status not in VALID_STATUSES:
        raise TrackerError(
            "invalid_parameter_value",
            f"Invalid status {status!r}: expected one of {sorted(VALID_STATUSES)}.",
        )
    return status  # type: ignore[return-value]


def validate_experiment_name(name: str) -> str:
    """Validate experiment name."""
    if not isinstance(name, str) or not name.strip() or len(name) > MAX_EXPERIMENT_NAME_LENGTH:
        raise TrackerError(
            "invalid_parameter_value",
            "Experiment name must be a non-empty string of at most 250 characters.",
        )
    return name


def validate_run_name(name: str | None) -> str | None:
    """Validate optional run name."""
    if name is None:
        return None
    if not isinstance(name, str) or len(name) > MAX_RUN_NAME_LENGTH:
        raise TrackerError(
            "invalid_parameter_value",
            "Run name must be a string of at most 250 characters.",
        )
    return name


def validate_artifact_subpath(artifact_path: str | None) -> Path:
    """Validate relative artifact destination path against traversal."""
    if artifact_path is None or artifact_path == "":
        return Path(".")
    if not isinstance(artifact_path, str) or len(artifact_path) > MAX_KEY_LENGTH:
        raise TrackerError(
            "invalid_parameter_value",
            "Invalid artifact_path length.",
        )
    norm = os.path.normpath(artifact_path)
    if (
        os.path.isabs(artifact_path)
        or norm.startswith("..")
        or "/.." in artifact_path.replace("\\", "/")
    ):
        raise TrackerError(
            "invalid_parameter_value",
            f"Invalid artifact_path {artifact_path!r}: relative subpaths only.",
        )
    parts = Path(artifact_path).parts
    if any(p in ("..", "/") or "\\" in p for p in parts):
        raise TrackerError(
            "invalid_parameter_value",
            f"Invalid artifact_path {artifact_path!r}: path traversal not allowed.",
        )
    return Path(*parts) if parts else Path(".")
