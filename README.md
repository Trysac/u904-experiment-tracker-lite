# experiment-tracker-lite

[![PyPI version](https://img.shields.io/pypi/v/trysac-experiment-tracker-lite)](https://pypi.org/project/trysac-experiment-tracker-lite/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-experiment-tracker-lite)](LICENSE)
[![CI](https://github.com/Trysac/u904-experiment-tracker-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u904-experiment-tracker-lite/actions/workflows/ci.yml)

File-based experiment tracker with a subset of the MLflow fluent API, run search with MLflow filter syntax and a reproducible static comparison site

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-experiment-tracker-lite
# or
uv add trysac-experiment-tracker-lite
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import experiment_tracker_lite

# normalize() trims surrounding whitespace and lower-cases the text.
print(experiment_tracker_lite.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    experiment_tracker_lite.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
