"""Command-line interface for experiment_tracker_lite."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

from .errors import TrackerError
from .models import Run
from .tracker import Tracker


def _format_cell(val: str | float | None) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        return f"{val:.6g}"
    return val


def _render_compare_table(runs: list[Run]) -> str:
    param_keys = sorted({k for r in runs for k in r.data.params})
    metric_keys = sorted({k for r in runs for k in r.data.metrics})

    headers = ["Field", *[r.info.run_id for r in runs]]
    rows: list[list[str]] = [
        ["run_name", *[r.info.run_name for r in runs]],
        ["status", *[r.info.status for r in runs]],
    ]
    for pk in param_keys:
        rows.append([f"param.{pk}", *[_format_cell(r.data.params.get(pk)) for r in runs]])
    for mk in metric_keys:
        rows.append([f"metric.{mk}", *[_format_cell(r.data.metrics.get(mk)) for r in runs]])

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            if len(cell) > widths[idx]:
                widths[idx] = len(cell)

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    sep = "-+-".join("-" * w for w in widths)
    output_lines = [fmt_row(headers), sep, *[fmt_row(r) for r in rows]]
    return "\n".join(output_lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment_tracker_lite CLI.

    Returns 0 on success, 1 on runtime error, 2 on usage error.
    """
    parser = argparse.ArgumentParser(
        prog="experiment-tracker-lite",
        description="Compare experiment runs or build a static comparison site.",
    )
    subparsers = parser.add_subparsers(dest="command")

    compare_parser = subparsers.add_parser("compare", help="Compare two or more runs side by side.")
    compare_parser.add_argument(
        "--root",
        default="./mlruns",
        help="Tracking root directory (default: ./mlruns).",
    )
    compare_parser.add_argument(
        "run_ids",
        nargs="+",
        help="At least two run IDs to compare.",
    )

    site_parser = subparsers.add_parser("site", help="Build a static HTML comparison site.")
    site_parser.add_argument(
        "--root",
        default="./mlruns",
        help="Tracking root directory (default: ./mlruns).",
    )
    site_parser.add_argument(
        "--out",
        required=True,
        help="Output directory for generated HTML files.",
    )

    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    if args.command is None:
        parser.print_usage(sys.stderr)
        return 2

    if args.command == "compare":
        if len(args.run_ids) < 2:
            compare_parser.print_usage(sys.stderr)
            return 2
        tracker = Tracker(root=args.root)
        runs: list[Run] = []
        try:
            for rid in args.run_ids:
                runs.append(tracker.get_run(rid))
        except TrackerError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(_render_compare_table(runs))
        return 0

    if args.command == "site":
        tracker = Tracker(root=args.root)
        try:
            tracker.build_site(args.out)
        except TrackerError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    return 2
