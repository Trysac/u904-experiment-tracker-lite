"""Behavior tests for the public API of experiment_tracker_lite."""

from __future__ import annotations

import pytest

import experiment_tracker_lite


def test_normalize_trims_and_lowercases() -> None:
    assert experiment_tracker_lite.normalize("  Hello World ") == "hello world"


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_normalize_rejects_blank_input(value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        experiment_tracker_lite.normalize(value)


def test_version_is_semver() -> None:
    parts = experiment_tracker_lite.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
