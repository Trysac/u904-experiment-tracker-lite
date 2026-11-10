"""Atomic file-based persistence for experiments, runs, metrics, and artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .errors import TrackerError
from .models import MetricPoint, Run, RunData, RunInfo
from .validation import (
    MAX_ARTIFACT_BYTES,
    MAX_METRIC_POINTS,
    validate_artifact_subpath,
)


def atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write bytes to a temporary file in the same directory and atomically replace target."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp_", suffix=".part")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
        os.replace(tmp_path, target)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(target: Path, payload: dict[str, Any]) -> None:
    """Serialize payload deterministically to JSON and write atomically."""
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n"
    atomic_write_bytes(target, content.encode("utf-8"))


def read_json(target: Path) -> dict[str, Any]:
    """Read a JSON dictionary from target."""
    with target.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


def _metric_filename_for_key(run_dir: Path, key: str) -> Path:
    """Return the .jsonl path for a metric key inside <run_dir>/metrics/."""
    metrics_dir = run_dir / "metrics"
    # Safe filename: if simple key without slashes or special collision risks, use <key>.jsonl;
    # if key has slashes, create nested directories under metrics/.
    rel_parts = [p for p in key.split("/") if p]
    target = metrics_dir.joinpath(*rel_parts)
    return target.with_name(target.name + ".jsonl")


def append_metric_point(run_dir: Path, key: str, point: MetricPoint) -> None:
    """Atomically append a metric point to <run_dir>/metrics/<key>.jsonl enforcing point limits."""
    target = _metric_filename_for_key(run_dir, key)
    existing_bytes = b""
    line_count = 0
    if target.exists():
        existing_bytes = target.read_bytes()
        line_count = existing_bytes.count(b"\n")
    if line_count >= MAX_METRIC_POINTS:
        raise TrackerError(
            "limit_exceeded",
            f"Metric {key!r} reached maximum of {MAX_METRIC_POINTS} logged points.",
        )
    record = {"key": key, "step": point.step, "timestamp": point.timestamp, "value": point.value}
    new_line = (json.dumps(record, sort_keys=True, allow_nan=True) + "\n").encode("utf-8")
    atomic_write_bytes(target, existing_bytes + new_line)


def load_latest_metrics(run_dir: Path) -> dict[str, float]:
    """Load metric jsonl files and compute latest value by (step, timestamp)."""
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        return {}
    latest: dict[str, tuple[int, int, int, float]] = {}
    for jsonl_path in sorted(metrics_dir.rglob("*.jsonl")):
        rel = jsonl_path.relative_to(metrics_dir).as_posix()
        default_key = rel[: -len(".jsonl")]
        with jsonl_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                k = str(obj.get("key", default_key))
                step = int(obj["step"])
                ts = int(obj["timestamp"])
                val = float(obj["value"])
                candidate = (step, ts, idx, val)
                prev = latest.get(k)
                if prev is None or (candidate[0], candidate[1], candidate[2]) >= (
                    prev[0],
                    prev[1],
                    prev[2],
                ):
                    latest[k] = candidate
    return {k: v[3] for k, v in sorted(latest.items())}


def save_run_json(
    run_dir: Path,
    info: RunInfo,
    params: dict[str, str],
    tags: dict[str, str],
) -> None:
    """Write <run_dir>/run.json atomically."""
    payload = {
        "info": {
            "end_time": info.end_time,
            "experiment_id": info.experiment_id,
            "run_id": info.run_id,
            "run_name": info.run_name,
            "start_time": info.start_time,
            "status": info.status,
        },
        "params": dict(sorted(params.items())),
        "tags": dict(sorted(tags.items())),
    }
    atomic_write_json(run_dir / "run.json", payload)


def load_run(run_dir: Path) -> Run:
    """Load a Run entity from <run_dir>/run.json and <run_dir>/metrics/."""
    run_json_path = run_dir / "run.json"
    if not run_json_path.exists():
        raise TrackerError("resource_does_not_exist", f"Run directory {run_dir.name!r} not found.")
    raw = read_json(run_json_path)
    info_raw = raw["info"]
    info = RunInfo(
        run_id=str(info_raw["run_id"]),
        run_name=str(info_raw["run_name"]),
        experiment_id=str(info_raw["experiment_id"]),
        status=info_raw["status"],
        start_time=int(info_raw["start_time"]),
        end_time=int(info_raw["end_time"]) if info_raw.get("end_time") is not None else None,
    )
    params: dict[str, str] = {str(k): str(v) for k, v in raw.get("params", {}).items()}
    tags: dict[str, str] = {str(k): str(v) for k, v in raw.get("tags", {}).items()}
    metrics = load_latest_metrics(run_dir)
    return Run(info=info, data=RunData(params=params, metrics=metrics, tags=tags))


def copy_artifact_atomically(
    run_dir: Path, local_path: str | os.PathLike[str], artifact_path: str | None = None
) -> None:
    """Copy a file or directory into <run_dir>/artifacts/[artifact_path] enforcing size limits."""
    src = Path(local_path)
    if not src.exists():
        raise TrackerError(
            "invalid_parameter_value",
            f"Artifact source path {str(local_path)!r} does not exist.",
        )
    subpath = validate_artifact_subpath(artifact_path)
    artifacts_root = run_dir / "artifacts"
    dest_dir = artifacts_root / subpath

    # Calculate existing artifact total size + new source size
    existing_total = 0
    if artifacts_root.exists():
        for p in artifacts_root.rglob("*"):
            if p.is_file():
                existing_total += p.stat().st_size

    if src.is_file():
        new_size = src.stat().st_size
        if existing_total + new_size > MAX_ARTIFACT_BYTES:
            raise TrackerError(
                "limit_exceeded",
                f"Artifact size exceeds maximum limit of {MAX_ARTIFACT_BYTES} bytes.",
            )
        atomic_write_bytes(dest_dir / src.name, src.read_bytes())
    elif src.is_dir():
        files_to_copy: list[tuple[Path, Path]] = []
        added_size = 0
        for item in sorted(src.rglob("*")):
            if item.is_file():
                added_size += item.stat().st_size
                if existing_total + added_size > MAX_ARTIFACT_BYTES:
                    raise TrackerError(
                        "limit_exceeded",
                        f"Artifact directory exceeds maximum limit of {MAX_ARTIFACT_BYTES} bytes.",
                    )
                rel = item.relative_to(src)
                files_to_copy.append((item, dest_dir / src.name / rel))
        for src_file, dst_file in files_to_copy:
            atomic_write_bytes(dst_file, src_file.read_bytes())
    else:
        raise TrackerError(
            "invalid_parameter_value", "Artifact source must be a regular file or directory."
        )


def list_run_artifacts(run_dir: Path) -> list[tuple[str, int, str]]:
    """Return sorted list of (relative_posix_path, size_bytes, sha256_hex) for a run's artifacts."""
    artifacts_root = run_dir / "artifacts"
    if not artifacts_root.exists():
        return []
    results: list[tuple[str, int, str]] = []
    for p in sorted(artifacts_root.rglob("*")):
        if p.is_file() and not p.name.startswith(".tmp_"):
            rel = p.relative_to(artifacts_root).as_posix()
            data = p.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            results.append((rel, len(data), digest))
    return results


__all__ = [
    "append_metric_point",
    "atomic_write_bytes",
    "atomic_write_json",
    "copy_artifact_atomically",
    "list_run_artifacts",
    "load_latest_metrics",
    "load_run",
    "read_json",
    "save_run_json",
    "shutil",
]
