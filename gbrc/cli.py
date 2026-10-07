"""Argparse CLI for GBRC operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import resolve_fund
from .data_quality import audit_all
from .etf.engine import BriefingEngine
from .etf.migration import (
    migrate_legacy_snapshots,
    validate_normalized_directory,
)
from .etf.workflow import backfill_range, manifest_as_text, run_daily_etf_workflow
from .system import create_baseline_manifest


def _add_daily_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--date", help="requested date in YYYY-MM-DD")
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="render from the latest validated snapshots without provider calls",
    )
    parser.add_argument("--json", action="store_true", help="print run JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gbrc",
        description="Growth-Based Research Center operations",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    briefing = commands.add_parser(
        "briefing",
        help="run the complete two-fund ETF daily briefing",
    )
    _add_daily_flags(briefing)

    etf = commands.add_parser("etf", help="ETF data and reporting commands")
    etf_commands = etf.add_subparsers(dest="etf_command", required=True)
    daily = etf_commands.add_parser("daily", help="fetch, validate, and report both funds")
    _add_daily_flags(daily)
    etf_commands.add_parser("migrate", help="migrate v1 snapshots non-destructively")
    etf_commands.add_parser("validate", help="validate all normalized snapshots")
    trades = etf_commands.add_parser(
        "trades",
        help="real-trade timing, monthly contribution, and US Top N holding analysis",
    )
    trades.add_argument("--fund", default="time", help="fund alias (default: time)")
    trades.add_argument("--end", help="last snapshot date YYYY-MM-DD (default: latest)")
    trades.add_argument("--top", type=int, default=10, help="US-listed Top N (default: 10)")
    report = etf_commands.add_parser(
        "report",
        help="print one fund's report section from stored snapshots (no collection)",
    )
    report.add_argument("--fund", default="time", help="fund alias (default: time)")
    report.add_argument("--date", help="snapshot date YYYY-MM-DD (default: latest)")
    backfill = etf_commands.add_parser(
        "backfill",
        help="fill weekdays without a validated snapshot (provider calls)",
    )
    backfill.add_argument("start", help="first date YYYY-MM-DD")
    backfill.add_argument("end", help="last date YYYY-MM-DD")
    backfill.add_argument("--fund", default="time", help="fund alias (default: time)")
    cumulative = etf_commands.add_parser(
        "cumulative",
        help="one report for the span since the last delivered briefing",
    )
    cumulative.add_argument("--start", help="start date (default: previous common snapshot)")
    cumulative.add_argument("--end", help="end date (default: latest common snapshot)")
    cumulative.add_argument("--out-dir", help="output folder (default: data/derived/briefings)")

    system = commands.add_parser("system", help="system maintenance commands")
    system_commands = system.add_subparsers(dest="system_command", required=True)
    system_commands.add_parser("baseline", help="record a non-destructive baseline manifest")
    system_commands.add_parser(
        "audit",
        help="audit legacy option and earnings logs without modifying them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "system" and args.system_command == "baseline":
        path, payload = create_baseline_manifest()
        print(f"baseline: {path}")
        print(json.dumps(payload["worktree_status_counts"], ensure_ascii=False))
        return 0

    if args.command == "system" and args.system_command == "audit":
        summary = audit_all(Path(__file__).resolve().parents[1])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2 if summary["needs_migration"] else 0

    if args.command == "etf" and args.etf_command == "migrate":
        summary = migrate_legacy_snapshots()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["failed"] == 0 else 2

    if args.command == "etf" and args.etf_command == "validate":
        summary = validate_normalized_directory()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["invalid"] == 0 and summary["valid"] > 0 else 2

    if args.command == "etf" and args.etf_command == "report":
        print(BriefingEngine().fund_report(resolve_fund(args.fund), args.date))
        return 0

    if args.command == "etf" and args.etf_command == "backfill":
        records = backfill_range(resolve_fund(args.fund), args.start, args.end)
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0 if all(record["status"] != "failed" for record in records) else 2

    if args.command == "etf" and args.etf_command == "cumulative":
        from .etf.cumulative import run as run_cumulative  # prices may need the network

        options = {"out_dir": args.out_dir} if args.out_dir else {}
        print(json.dumps(run_cumulative(args.start, args.end, **options),
                         ensure_ascii=False, indent=2))
        return 0

    if args.command == "etf" and args.etf_command == "trades":
        from .etf.trades import write_analysis  # network (yfinance) only on demand

        path, summary = write_analysis(args.fund, args.end, args.top)
        print(f"report: {path}")
        print(f"period: {summary['start']} → {summary['end']} ({summary['snapshots']} snapshots)")
        print(f"holding-based {summary['holding_based_pct']:+.1f}%, "
              f"monthly trade effect sum {summary['trade_effect_sum_pp']:+.1f}%p, "
              f"real-trade events {summary['event_count']}")
        return 0

    if args.command == "briefing" or (
        args.command == "etf" and args.etf_command == "daily"
    ):
        code, manifest = run_daily_etf_workflow(
            requested_date=args.date,
            fetch=not args.no_fetch,
        )
        if args.json:
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
        else:
            print(manifest_as_text(manifest))
        return code

    raise AssertionError("unreachable command")
