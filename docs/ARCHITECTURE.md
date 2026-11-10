# Architecture

## Overview

`experiment_tracker_lite` is structured into six focused modules with zero external runtime dependencies:

1. `errors.py`: Defines `TrackerError(code, message)` with machine-readable `ErrorCode` literals (`invalid_parameter_value`, `resource_does_not_exist`, `invalid_state`, `bad_filter`, `limit_exceeded`).
2. `models.py`: Immutable dataclasses mirroring MLflow entities: `Run`, `RunInfo`, `RunData`, and `MetricPoint`.
3. `validation.py`: Enforces MLflow parameter/key/tag character and length limits, finite/NaN metric constraints, path traversal protections, and artifact size caps.
4. `storage.py`: Implements atomic file writes (`tempfile.mkstemp` + `os.replace`) for `<root>/<experiment_id>/meta.json`, `<root>/<experiment_id>/<run_id>/run.json`, and `<root>/<experiment_id>/<run_id>/metrics/<key>.jsonl`.
5. `search.py`: Implements a lexer, parser, and evaluator for the MLflow `filter_string` and `order_by` subset.
6. `site.py` & `cli.py`: Reproducible HTML comparison site builder (`html.escape` on all user strings, deterministic sorting, no wall-clock timestamps) and CLI comparison tool.
