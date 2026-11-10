# Use Atomic JSON/JSONL File Layout and List-Based Search Return Type

Status: accepted (generated, pending human review)

## Context

Small ML projects need MLflow-compatible experiment tracking without running a tracking server, database, or installing heavy dataframe dependencies.

## Decision

Store experiment and run metadata in deterministic JSON files and metric histories in append-atomically-replaced JSONL files using stdlib only, inject clock and ID generators into Tracker for deterministic testing, and return list[Run] from search_runs.

## Consequences

Scripts use familiar MLflow call names and filter syntax with zero runtime dependencies, git-reviewable run files, and byte-identical static HTML comparison sites.
