"""Deterministic static HTML comparison site generator."""

from __future__ import annotations

import html
import math
from pathlib import Path

from .models import Run
from .storage import atomic_write_bytes, list_run_artifacts, load_run, read_json

_CSS = """:root {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: #1f2937;
  background: #f9fafb;
}
body {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}
h1, h2, h3 {
  color: #111827;
}
a {
  color: #2563eb;
  text-decoration: none;
}
a:hover {
  text-decoration: underline;
}
table {
  width: 100%;
  border-collapse: collapse;
  background: #ffffff;
  margin-bottom: 2rem;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}
th, td {
  border: 1px solid #e5e7eb;
  padding: 0.6rem 0.8rem;
  text-align: left;
  font-size: 0.9rem;
}
th {
  background: #f3f4f6;
  font-weight: 600;
}
code {
  background: #f3f4f6;
  padding: 0.1rem 0.3rem;
  border-radius: 3px;
}
.status-FINISHED { color: #059669; font-weight: 600; }
.status-FAILED { color: #dc2626; font-weight: 600; }
.status-RUNNING { color: #d97706; font-weight: 600; }
.status-KILLED { color: #6b7280; font-weight: 600; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    if math.isnan(value):
        return "NaN"
    return f"{value:.6g}"


def _collect_experiments_and_runs(
    root: Path,
) -> list[tuple[str, str, list[tuple[Run, list[tuple[str, int, str]]]]]]:
    """Load all experiments and runs in deterministic order."""
    if not root.exists():
        return []
    experiments: list[tuple[str, str, list[tuple[Run, list[tuple[str, int, str]]]]]] = []
    for exp_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        meta_path = exp_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = read_json(meta_path)
        exp_id = str(meta["experiment_id"])
        exp_name = str(meta["name"])
        runs_with_artifacts: list[tuple[Run, list[tuple[str, int, str]]]] = []
        for run_dir in sorted(exp_dir.iterdir(), key=lambda p: p.name):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            if (run_dir / "run.json").exists():
                run = load_run(run_dir)
                artifacts = list_run_artifacts(run_dir)
                runs_with_artifacts.append((run, artifacts))
        # Sort runs deterministically: start_time DESC, run_id ASC
        runs_with_artifacts.sort(key=lambda item: (-item[0].info.start_time, item[0].info.run_id))
        experiments.append((exp_id, exp_name, runs_with_artifacts))
    experiments.sort(key=lambda item: (item[0], item[1]))
    return experiments


def _render_index(
    experiments: list[tuple[str, str, list[tuple[Run, list[tuple[str, int, str]]]]]],
) -> str:
    lines: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        "  <title>Experiment Tracker Lite - Overview</title>",
        f"  <style>{_CSS}</style>",
        "</head>",
        "<body>",
        "  <h1>Experiment Tracker Lite</h1>",
        "  <h2>Experiments</h2>",
    ]
    if not experiments:
        lines.append("  <p>No experiments recorded yet.</p>")
    else:
        lines.extend(
            [
                "  <table>",
                "    <thead>",
                "      <tr>"
                "<th>Experiment ID</th><th>Name</th><th>Runs</th><th>Compare Page</th>"
                "</tr>",
                "    </thead>",
                "    <tbody>",
            ]
        )
        for exp_id, exp_name, runs in experiments:
            page_href = f"experiment_{_esc(exp_id)}.html"
            lines.append(
                f"      <tr>"
                f"<td><code>{_esc(exp_id)}</code></td>"
                f"<td>{_esc(exp_name)}</td>"
                f"<td>{len(runs)}</td>"
                f'<td><a href="{page_href}">Compare Runs</a></td>'
                f"</tr>"
            )
        lines.extend(["    </tbody>", "  </table>"])

        lines.append("  <h2>All Runs</h2>")
        lines.extend(
            [
                "  <table>",
                "    <thead>",
                "      <tr>"
                "<th>Experiment</th>"
                "<th>Run ID</th>"
                "<th>Run Name</th>"
                "<th>Status</th>"
                "<th>Start Time (ms)</th>"
                "<th>Duration (ms)</th>"
                "<th>Params</th>"
                "<th>Latest Metrics</th>"
                "</tr>",
                "    </thead>",
                "    <tbody>",
            ]
        )
        for exp_id, exp_name, runs in experiments:
            for run, _ in runs:
                duration_str = (
                    str(run.info.end_time - run.info.start_time)
                    if run.info.end_time is not None
                    else "-"
                )
                params_summary = ", ".join(
                    f"{_esc(k)}={_esc(v)}" for k, v in sorted(run.data.params.items())
                )
                metrics_summary = ", ".join(
                    f"{_esc(k)}={_esc(_format_metric(v))}"
                    for k, v in sorted(run.data.metrics.items())
                )
                status_cls = f"status-{_esc(run.info.status)}"
                lines.append(
                    f"      <tr>"
                    f'<td><a href="experiment_{_esc(exp_id)}.html">{_esc(exp_name)}</a></td>'
                    f"<td><code>{_esc(run.info.run_id)}</code></td>"
                    f"<td>{_esc(run.info.run_name)}</td>"
                    f'<td class="{status_cls}">{_esc(run.info.status)}</td>'
                    f"<td>{run.info.start_time}</td>"
                    f"<td>{_esc(duration_str)}</td>"
                    f"<td>{params_summary or '-'}</td>"
                    f"<td>{metrics_summary or '-'}</td>"
                    f"</tr>"
                )
        lines.extend(["    </tbody>", "  </table>"])

    lines.extend(["</body>", "</html>", ""])
    return "\n".join(lines)


def _render_experiment_page(
    exp_id: str,
    exp_name: str,
    runs: list[tuple[Run, list[tuple[str, int, str]]]],
) -> str:
    param_keys = sorted({k for run, _ in runs for k in run.data.params})
    metric_keys = sorted({k for run, _ in runs for k in run.data.metrics})
    tag_keys = sorted({k for run, _ in runs for k in run.data.tags})

    lines: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        f"  <title>Experiment {_esc(exp_name)} ({_esc(exp_id)}) - Comparison</title>",
        f"  <style>{_CSS}</style>",
        "</head>",
        "<body>",
        '  <p><a href="index.html">&larr; Back to Overview</a></p>',
        f"  <h1>Experiment: {_esc(exp_name)} (<code>{_esc(exp_id)}</code>)</h1>",
    ]

    if not runs:
        lines.append("  <p>No runs recorded in this experiment.</p>")
    else:
        lines.append("  <h2>Run Comparison Table</h2>")
        header_cells = [
            "<th>Run ID</th>",
            "<th>Run Name</th>",
            "<th>Status</th>",
            "<th>Start Time (ms)</th>",
            "<th>End Time (ms)</th>",
        ]
        for pk in param_keys:
            header_cells.append(f"<th>param: {_esc(pk)}</th>")
        for mk in metric_keys:
            header_cells.append(f"<th>metric: {_esc(mk)}</th>")
        for tk in tag_keys:
            header_cells.append(f"<th>tag: {_esc(tk)}</th>")
        header_cells.append("<th>Artifacts</th>")

        lines.extend(
            [
                "  <table>",
                "    <thead>",
                "      <tr>" + "".join(header_cells) + "</tr>",
                "    </thead>",
                "    <tbody>",
            ]
        )
        for run, artifacts in runs:
            status_cls = f"status-{_esc(run.info.status)}"
            end_str = str(run.info.end_time) if run.info.end_time is not None else "-"
            row_cells = [
                f"<td><code>{_esc(run.info.run_id)}</code></td>",
                f"<td>{_esc(run.info.run_name)}</td>",
                f'<td class="{status_cls}">{_esc(run.info.status)}</td>',
                f"<td>{run.info.start_time}</td>",
                f"<td>{_esc(end_str)}</td>",
            ]
            for pk in param_keys:
                val = run.data.params.get(pk, "-")
                row_cells.append(f"<td>{_esc(val)}</td>")
            for mk in metric_keys:
                mval = run.data.metrics.get(mk)
                row_cells.append(f"<td>{_esc(_format_metric(mval))}</td>")
            for tk in tag_keys:
                tval = run.data.tags.get(tk, "-")
                row_cells.append(f"<td>{_esc(tval)}</td>")
            art_str = (
                ", ".join(f"{_esc(rel)} ({sz}B)" for rel, sz, _ in artifacts) if artifacts else "-"
            )
            row_cells.append(f"<td>{art_str}</td>")
            lines.append("      <tr>" + "".join(row_cells) + "</tr>")
        lines.extend(["    </tbody>", "  </table>"])

    lines.extend(["</body>", "</html>", ""])
    return "\n".join(lines)


def build_site_for_root(root: Path, out_dir: Path) -> list[Path]:
    """Generate index.html and per-experiment comparison pages deterministically."""
    out_dir.mkdir(parents=True, exist_ok=True)
    experiments = _collect_experiments_and_runs(root)

    written: list[Path] = []
    index_path = out_dir / "index.html"
    atomic_write_bytes(index_path, _render_index(experiments).encode("utf-8"))
    written.append(index_path)

    for exp_id, exp_name, runs in experiments:
        exp_page_path = out_dir / f"experiment_{exp_id}.html"
        html_text = _render_experiment_page(exp_id, exp_name, runs)
        atomic_write_bytes(exp_page_path, html_text.encode("utf-8"))
        written.append(exp_page_path)

    return sorted(written)
