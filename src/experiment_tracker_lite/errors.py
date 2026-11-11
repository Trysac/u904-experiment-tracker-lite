"""Error definitions for experiment_tracker_lite."""

from __future__ import annotations

from typing import Literal

ErrorCode = Literal[
    "invalid_parameter_value",
    "resource_does_not_exist",
    "invalid_state",
    "bad_filter",
    "limit_exceeded",
]


class TrackerError(Exception):
    """Base exception raised by experiment_tracker_lite operations."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        self.code: ErrorCode = code
        safe_msg = message[:200] if len(message) > 200 else message
        super().__init__(f"[{code}] {safe_msg}")
        self.message = safe_msg
