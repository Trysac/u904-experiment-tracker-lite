# Project record\n\n## Objective\nFile-based experiment tracker with a subset of the MLflow fluent API, run search with MLflow filter syntax, and a reproducible static comparison site.\n\n## Scope\n- Public API documented in README.md, fully typed and tested.\n\n## Out of Scope\n- Tracking server.\n- Model registry.\n- Autologging.\n- Nested runs.\n- The MLflow on-disk format.\n- Charts in the site (tables only).\n\n## Effort metrics\n| Metric | Value |\n|---|---|\n| Tokens (input / output) | pending measurement |\n| Hours (human) | pending measurement |\n| Cost (USD) | pending measurement |

## Factory run u904 (measured from API usage)

| Date (UTC) | Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|---|
| 2026-11-10 | generation | google/gemini-4-argon | yes | 5453 | 28772 | 0 | 0 | 0.5973 |
| 2026-11-10 | fix-1 | google/gemini-3.8-flash | no | 214742 | 1186 | 0 | 107845 | 0.1736 |
| 2026-11-10 | audit | google/gemini-3.8-flash | no | 295311 | 1360 | 0 | 549200 | 0.2678 |
| 2026-11-10 | docs | google/gemini-3.1-flash-lite | no | 201363 | 1294 | 0 | 0 | 0.0523 |

Total: $1.0909. Prices: vertex line, as measured by its adapter. Providers serve the current version of each model id, so the date tells which version answered; versions, launch stages and price sources per line are in the factory's model register (docs/MODELS.md).
