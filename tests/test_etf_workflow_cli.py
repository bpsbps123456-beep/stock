"""Network-free regression tests for the ETF workflow and CLI routing."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from gbrc import cli
from gbrc.config import load_fund_configs
from gbrc.etf import workflow
from gbrc.etf.snapshot import normalize_snapshot
from gbrc.etf.storage import save_normalized_snapshot as save_to_root
from gbrc.etf.storage import save_raw_capture as save_raw_to_root


PERFORMANCE_MARKER = "PERFORMANCE_FIXTURE_TABLE"
DIGEST_MARKER = "DIGEST_FIXTURE_CONSENSUS"


class FakeCrawler:
    """Small deterministic replacement for the network-backed crawler."""

    def __init__(
        self,
        *,
        failing_alias: str | None = None,
        snapshot_dates: dict[str, list[str]] | None = None,
        performance_error: Exception | None = None,
        digest_error: Exception | None = None,
    ) -> None:
        self.failing_alias = failing_alias
        self.snapshot_dates = snapshot_dates or {"time": [], "koact": []}
        self.performance_error = performance_error
        self.digest_error = digest_error
        self.digest_calls: list[str] = []
        self.active_alias: str | None = None
        self.holdings_count = {"time": 2, "koact": 2}
        self.fetch_calls: list[tuple[str, str]] = []
        self.compare_calls: list[tuple[str, str]] = []
        self.performance_calls: list[str | None] = []

    def set_active_etf(self, alias: str) -> None:
        self.active_alias = alias

    def fetch_holdings(self, requested_date: str) -> dict[str, Any]:
        if self.active_alias is None:
            raise AssertionError("set_active_etf must be called before fetch_holdings")
        alias = self.active_alias
        self.fetch_calls.append((alias, requested_date))
        if self.failing_alias in {alias, "*"}:
            raise RuntimeError(f"fixture provider failure: {alias}")

        fund = load_fund_configs()[alias]
        return {
            "date": requested_date,
            "actual_date": requested_date,
            "etf_ticker": fund.snapshot_ticker,
            "etf_name": fund.display_name,
            "source_url": f"https://example.invalid/{alias}",
            "fetched_at": f"{requested_date}T18:00:00+09:00",
            "holdings_count": self.holdings_count[alias],
            "holdings": [
                {
                    "code": f"{alias.upper()} TEST EQUITY",
                    "name": f"{alias.upper()} Test Security",
                    "qty": 9,
                    "value_krw": 900,
                    "weight": 90.0,
                },
                {
                    "code": "",
                    "name": "Cash",
                    "qty": 100,
                    "value_krw": 100,
                    "weight": 10.0,
                },
            ],
        }

    def list_snapshot_dates(self) -> list[str]:
        if self.active_alias is None:
            raise AssertionError("set_active_etf must be called before list_snapshot_dates")
        return list(self.snapshot_dates.get(self.active_alias, []))

    def compare_report(self, report_date: str) -> None:
        if self.active_alias is None:
            raise AssertionError("set_active_etf must be called before compare_report")
        self.compare_calls.append((self.active_alias, report_date))
        print(f"### [현재 비중 Top 10] — {report_date}")
        print("  1  FIXTURE1  Security  90.00%")
        print("### [현금 보유액 별도 점검]")
        print("Cash  10.00%")
        print("### [섹터별 비중/변화 요약]")
        print("AI 반도체  90.00%")
        print("### [수량 기준 실매매]")
        print("  D-5 실매매 없음")
        print("### [D-20 신규 편입 클러스터]")
        print("신규 편입 없음")
        print("### [One-glance table]")
        print(
            "  Rank  Code  Name  Weight  D-1 dW(ret)  D-3 dW(ret)  "
            "D-5 dW(ret)  D-20 dW(ret)"
        )
        print("  ----  ----  ----  ------  -----------  -----------  -----------  ------------")
        for rank in range(1, self.holdings_count[self.active_alias] + 1):
            code = "CASH" if rank == self.holdings_count[self.active_alias] else "SECURITY"
            name = "Cash" if code == "CASH" else f"Fixture {rank}"
            print(
                f"{rank:6d}  {code}  {name}  "
                "50.00%  +0.00  +0.00  +0.00  +0.00"
            )
        print()
        print("### [포트폴리오 해석]")
        print("합성 포트폴리오 해석")
        print("### [직전 변화 복기]")
        print("합성 직전 변화 복기")

    def core_digest_report(self, report_date: str) -> None:
        self.digest_calls.append(report_date)
        if self.digest_error is not None:
            raise self.digest_error
        print("### [두 펀드 실매매 합의] — fixture")
        print(DIGEST_MARKER)

    def fund_performance_report(self, report_date: str | None = None) -> None:
        self.performance_calls.append(report_date)
        if self.performance_error is not None:
            raise self.performance_error
        print("### [두 펀드 성과 비교] — fixture")
        print(PERFORMANCE_MARKER)
        for horizon in ("D-1", "D-3", "D-5", "D-20"):
            print(f"{horizon}  TIME +0.00%  KoAct +0.00%  SAME")


def seed_normalized_snapshots(root: Path, snapshot_date: str) -> None:
    crawler = FakeCrawler()
    for alias in ("time", "koact"):
        crawler.set_active_etf(alias)
        raw = crawler.fetch_holdings(snapshot_date)
        save_to_root(normalize_snapshot(raw), root=root)


def run_isolated_workflow(
    crawler: FakeCrawler,
    base: Path,
    requested_date: str,
    *,
    fetch: bool,
) -> tuple[int, dict[str, Any]]:
    raw_root = base / "raw"
    normalized_root = base / "normalized"
    reports_root = base / "briefings"
    runs_root = base / "runs"

    def isolated_save(snapshot: dict[str, Any]) -> tuple[Path, str]:
        return save_to_root(snapshot, root=normalized_root)

    def isolated_raw_save(snapshot: dict[str, Any]) -> tuple[Path | None, str]:
        return save_raw_to_root(snapshot, root=raw_root)

    with (
        mock.patch.object(workflow, "_crawler", return_value=crawler),
        mock.patch.object(
            workflow, "save_normalized_snapshot", side_effect=isolated_save
        ),
        mock.patch.object(workflow, "save_raw_capture", side_effect=isolated_raw_save),
        mock.patch.object(workflow, "NORMALIZED_HOLDINGS_DIR", normalized_root),
        mock.patch.object(workflow, "DERIVED_BRIEFINGS_DIR", reports_root),
        mock.patch.object(workflow, "STATE_RUNS_DIR", runs_root),
    ):
        return workflow.run_daily_etf_workflow(requested_date, fetch=fetch)


def markdown_headings(report: str) -> list[str]:
    return [line.strip() for line in report.splitlines() if line.lstrip().startswith("#")]


def assert_etf_only_headings(test: unittest.TestCase, report: str) -> None:
    headings = markdown_headings(report)
    test.assertFalse(
        any("옵션" in heading and "ETF" in heading for heading in headings),
        headings,
    )
    test.assertFalse(any("PER" in heading.upper() for heading in headings), headings)


class DailyWorkflowTests(unittest.TestCase):
    requested_date = "2026-07-10"

    def test_two_fund_success_generates_performance_manifest_and_atomic_report(
        self,
    ) -> None:
        crawler = FakeCrawler()
        funds = load_fund_configs()

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            raw_root = base / "raw"
            normalized_root = base / "normalized"
            reports_root = base / "briefings"
            runs_root = base / "runs"

            def isolated_save(snapshot: dict[str, Any]) -> tuple[Path, str]:
                return save_to_root(snapshot, root=normalized_root)

            def isolated_raw_save(
                snapshot: dict[str, Any],
            ) -> tuple[Path | None, str]:
                return save_raw_to_root(snapshot, root=raw_root)

            original_replace = os.replace
            with (
                mock.patch.object(workflow, "_crawler", return_value=crawler),
                mock.patch.object(
                    workflow, "save_normalized_snapshot", side_effect=isolated_save
                ),
                mock.patch.object(
                    workflow, "save_raw_capture", side_effect=isolated_raw_save
                ),
                mock.patch.object(
                    workflow, "DERIVED_BRIEFINGS_DIR", reports_root
                ),
                mock.patch.object(workflow, "STATE_RUNS_DIR", runs_root),
                mock.patch(
                    "gbrc.etf.storage.os.replace", wraps=original_replace
                ) as replace,
            ):
                exit_code, manifest = workflow.run_daily_etf_workflow(
                    self.requested_date,
                    fetch=True,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(manifest["performance_status"], "generated")
            self.assertEqual(manifest["errors"], [])
            self.assertEqual(set(manifest["funds"]), {"time", "koact"})
            self.assertTrue(
                all(
                    fund_run["report_status"] == "generated"
                    for fund_run in manifest["funds"].values()
                )
            )
            self.assertEqual(
                crawler.fetch_calls,
                [
                    ("time", self.requested_date),
                    ("koact", self.requested_date),
                ],
            )
            self.assertEqual(
                crawler.compare_calls,
                [
                    ("time", self.requested_date),
                    ("koact", self.requested_date),
                ],
            )
            self.assertEqual(crawler.performance_calls, [self.requested_date])

            report_path = Path(manifest["report_path"])
            manifest_path = Path(manifest["manifest_path"])
            self.assertEqual(report_path, reports_root / f"{self.requested_date}.md")
            self.assertTrue(report_path.is_file())
            self.assertTrue(manifest_path.is_file())
            on_disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk_manifest["status"], "success")
            self.assertEqual(on_disk_manifest["manifest_path"], str(manifest_path))

            replaced_targets = [Path(call.args[1]) for call in replace.call_args_list]
            self.assertIn(report_path, replaced_targets)
            self.assertEqual(list(reports_root.glob(".*.tmp")), [])

            report = report_path.read_text(encoding="utf-8")
            level_two = [
                line.strip()
                for line in report.splitlines()
                if line.startswith("## ")
            ]
            self.assertEqual(
                level_two,
                [
                    "## 핵심 결론",
                    "## 두 펀드 가격 성과 비교",
                    f"## {funds['time'].display_name} (456600)",
                    f"## {funds['koact'].display_name} (2ETFQ1)",
                ],
            )
            self.assertIn(PERFORMANCE_MARKER, report)
            assert_etf_only_headings(self, report)
            self.assertEqual(manifest["core_digest_status"], "generated")
            self.assertEqual(crawler.digest_calls, [self.requested_date])
            core = report.split("## 핵심 결론", 1)[1].split("## 두 펀드 가격 성과 비교", 1)[0]
            self.assertIn(DIGEST_MARKER, core)

    def test_digest_failure_keeps_success_report_and_logs_the_stage(self) -> None:
        crawler = FakeCrawler(digest_error=RuntimeError("fixture digest failure"))

        with tempfile.TemporaryDirectory() as temp_dir:
            exit_code, manifest = run_isolated_workflow(
                crawler,
                Path(temp_dir),
                self.requested_date,
                fetch=True,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(manifest["core_digest_status"], "failed")
            self.assertEqual(
                [error["stage"] for error in manifest["errors"]], ["core_digest"]
            )
            report = Path(manifest["report_path"]).read_text(encoding="utf-8")
            self.assertNotIn(DIGEST_MARKER, report)
            self.assertIn(PERFORMANCE_MARKER, report)

    def test_one_fund_failure_is_partial_exit_two_without_performance_table(
        self,
    ) -> None:
        crawler = FakeCrawler(failing_alias="koact")

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            raw_root = base / "raw"
            normalized_root = base / "normalized"
            reports_root = base / "briefings"
            runs_root = base / "runs"

            def isolated_save(snapshot: dict[str, Any]) -> tuple[Path, str]:
                return save_to_root(snapshot, root=normalized_root)

            def isolated_raw_save(
                snapshot: dict[str, Any],
            ) -> tuple[Path | None, str]:
                return save_raw_to_root(snapshot, root=raw_root)

            with (
                mock.patch.object(workflow, "_crawler", return_value=crawler),
                mock.patch.object(
                    workflow, "save_normalized_snapshot", side_effect=isolated_save
                ),
                mock.patch.object(
                    workflow, "save_raw_capture", side_effect=isolated_raw_save
                ),
                mock.patch.object(
                    workflow, "DERIVED_BRIEFINGS_DIR", reports_root
                ),
                mock.patch.object(workflow, "STATE_RUNS_DIR", runs_root),
            ):
                exit_code, manifest = workflow.run_daily_etf_workflow(
                    self.requested_date,
                    fetch=True,
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(manifest["status"], "partial_success")
            self.assertEqual(
                manifest["performance_status"], "not_generated_partial_run"
            )
            self.assertEqual(manifest["funds"]["time"]["report_status"], "generated")
            self.assertEqual(
                manifest["funds"]["koact"]["report_status"], "not_generated"
            )
            self.assertEqual(manifest["funds"]["koact"]["quality_status"], "failed")
            self.assertEqual(len(manifest["errors"]), 1)
            self.assertEqual(manifest["errors"][0]["fund"], "koact")
            self.assertEqual(crawler.performance_calls, [])
            self.assertEqual(crawler.compare_calls, [("time", self.requested_date)])

            report = Path(manifest["report_path"]).read_text(encoding="utf-8")
            self.assertNotIn(PERFORMANCE_MARKER, report)
            self.assertIn("fixture provider failure: koact", report)

            on_disk_manifest = json.loads(
                Path(manifest["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(on_disk_manifest["status"], "partial_success")
            self.assertEqual(
                on_disk_manifest["performance_status"],
                "not_generated_partial_run",
            )

    def test_no_fetch_older_snapshots_are_existing_stale_and_run_is_stale(
        self,
    ) -> None:
        snapshot_date = "2026-07-09"
        crawler = FakeCrawler(
            snapshot_dates={
                "time": [snapshot_date, "2026-07-11"],
                "koact": [snapshot_date, "2026-07-11"],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            seed_normalized_snapshots(base / "normalized", snapshot_date)

            exit_code, manifest = run_isolated_workflow(
                crawler,
                base,
                self.requested_date,
                fetch=False,
            )

            self.assertEqual(exit_code, 2)
            self.assertEqual(manifest["status"], "stale")
            self.assertEqual(manifest["performance_status"], "generated_stale")
            self.assertEqual(crawler.fetch_calls, [])
            self.assertEqual(
                crawler.compare_calls,
                [("time", snapshot_date), ("koact", snapshot_date)],
            )
            self.assertEqual(crawler.performance_calls, [snapshot_date])
            for fund_run in manifest["funds"].values():
                self.assertEqual(fund_run["quality_status"], "existing_stale")
                self.assertEqual(fund_run["as_of_date"], snapshot_date)
                self.assertTrue(fund_run["stale_for_requested_date"])

            report_path = Path(manifest["report_path"])
            self.assertEqual(
                report_path.name,
                f"{self.requested_date}_{manifest['run_id']}_stale.md",
            )
            self.assertTrue(report_path.is_file())
            self.assertFalse(
                (base / "briefings" / f"{self.requested_date}.md").exists()
            )

    def test_performance_exception_is_partial_and_preserves_success_report(
        self,
    ) -> None:
        crawler = FakeCrawler(
            performance_error=RuntimeError("fixture performance failure")
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            canonical_report = base / "briefings" / f"{self.requested_date}.md"
            canonical_report.parent.mkdir(parents=True)
            original_report = b"known-good-success-report\n"
            canonical_report.write_bytes(original_report)

            exit_code, manifest = run_isolated_workflow(
                crawler,
                base,
                self.requested_date,
                fetch=True,
            )

            self.assertEqual(exit_code, 2)
            self.assertEqual(manifest["status"], "partial_success")
            self.assertEqual(manifest["performance_status"], "failed")
            self.assertEqual(crawler.performance_calls, [self.requested_date])
            self.assertEqual(len(manifest["errors"]), 1)
            self.assertEqual(manifest["errors"][0]["stage"], "performance")
            self.assertIn("fixture performance failure", manifest["errors"][0]["error"])

            report_path = Path(manifest["report_path"])
            self.assertNotEqual(report_path, canonical_report)
            self.assertEqual(
                report_path.name,
                f"{self.requested_date}_{manifest['run_id']}_partial_success.md",
            )
            self.assertTrue(report_path.is_file())
            self.assertIn(
                "fixture performance failure",
                report_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(canonical_report.read_bytes(), original_report)

            manifest_path = Path(manifest["manifest_path"])
            on_disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk_manifest["manifest_path"], str(manifest_path))

    def test_failed_run_preserves_success_report_and_uses_run_id_suffix(
        self,
    ) -> None:
        crawler = FakeCrawler(failing_alias="*")

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            canonical_report = base / "briefings" / f"{self.requested_date}.md"
            canonical_report.parent.mkdir(parents=True)
            original_report = b"known-good-success-report\n"
            canonical_report.write_bytes(original_report)

            exit_code, manifest = run_isolated_workflow(
                crawler,
                base,
                self.requested_date,
                fetch=True,
            )

            self.assertEqual(exit_code, 2)
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(
                manifest["performance_status"], "not_generated_partial_run"
            )
            self.assertEqual(len(manifest["errors"]), 2)
            self.assertEqual(crawler.performance_calls, [])

            report_path = Path(manifest["report_path"])
            self.assertNotEqual(report_path, canonical_report)
            self.assertEqual(
                report_path.name,
                f"{self.requested_date}_{manifest['run_id']}_failed.md",
            )
            self.assertTrue(report_path.is_file())
            self.assertEqual(canonical_report.read_bytes(), original_report)


class CliParserTests(unittest.TestCase):
    def test_all_supported_command_routes_parse(self) -> None:
        parser = cli.build_parser()
        cases = [
            (
                ["briefing", "--date", "2026-07-09", "--no-fetch", "--json"],
                {
                    "command": "briefing",
                    "date": "2026-07-09",
                    "no_fetch": True,
                    "json": True,
                },
            ),
            (
                ["etf", "daily", "--date", "2026-07-10"],
                {
                    "command": "etf",
                    "etf_command": "daily",
                    "date": "2026-07-10",
                    "no_fetch": False,
                    "json": False,
                },
            ),
            (["etf", "migrate"], {"command": "etf", "etf_command": "migrate"}),
            (["etf", "validate"], {"command": "etf", "etf_command": "validate"}),
            (
                ["etf", "trades", "--fund", "koact", "--end", "2026-09-23", "--top", "5"],
                {
                    "command": "etf",
                    "etf_command": "trades",
                    "fund": "koact",
                    "end": "2026-09-23",
                    "top": 5,
                },
            ),
            (
                ["etf", "trades"],
                {"command": "etf", "etf_command": "trades", "fund": "time", "top": 10},
            ),
            (
                ["system", "baseline"],
                {"command": "system", "system_command": "baseline"},
            ),
            (
                ["system", "audit"],
                {"command": "system", "system_command": "audit"},
            ),
        ]

        for argv, expected in cases:
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                for name, value in expected.items():
                    self.assertEqual(getattr(args, name), value)

    def test_cli_propagates_partial_workflow_exit_code(self) -> None:
        partial_manifest = {
            "status": "partial_success",
            "requested_date": "2026-07-10",
            "funds": {},
            "errors": [{"fund": "koact"}],
        }

        with (
            mock.patch.object(
                cli,
                "run_daily_etf_workflow",
                return_value=(2, partial_manifest),
            ) as run,
            redirect_stdout(io.StringIO()) as output,
        ):
            exit_code = cli.main(["etf", "daily", "--no-fetch", "--json"])

        self.assertEqual(exit_code, 2)
        run.assert_called_once_with(requested_date=None, fetch=False)
        self.assertEqual(json.loads(output.getvalue())["status"], "partial_success")

    def test_system_audit_needs_migration_prints_json_and_exits_two(self) -> None:
        summary = {
            "status": "needs_migration",
            "needs_migration": True,
            "options": {"needs_migration": True},
            "earnings": {"needs_migration": False},
        }

        with (
            mock.patch.object(cli, "audit_all", return_value=summary) as audit,
            redirect_stdout(io.StringIO()) as output,
        ):
            exit_code = cli.main(["system", "audit"])

        self.assertEqual(exit_code, 2)
        self.assertEqual(json.loads(output.getvalue()), summary)
        audit.assert_called_once_with(Path(cli.__file__).resolve().parents[1])


class ExistingNoFetchReportTests(unittest.TestCase):
    def test_latest_available_no_fetch_report_is_etf_only(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        run_root = project_root / "data" / "state" / "runs"
        candidates: list[tuple[Path, dict[str, Any]]] = []

        for manifest_path in sorted(run_root.glob("*_ETF_DAILY.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            fund_runs = list(manifest.get("funds", {}).values())
            if len(fund_runs) != 2 or not all(
                fund.get("fetch_requested") is False for fund in fund_runs
            ):
                continue

            configured_path = Path(str(manifest.get("report_path") or ""))
            fallback_path = (
                project_root
                / "data"
                / "derived"
                / "briefings"
                / f"{manifest.get('requested_date')}.md"
            )
            report_path = (
                configured_path if configured_path.is_file() else fallback_path
            )
            if report_path.is_file():
                candidates.append((report_path, manifest))

        if not candidates:
            self.skipTest("no generated no-fetch ETF report is available")

        report_path, _ = candidates[-1]
        report = report_path.read_text(encoding="utf-8")
        funds = load_fund_configs()
        level_two = [
            line.strip() for line in report.splitlines() if line.startswith("## ")
        ]

        self.assertIn(
            f"## {funds['time'].display_name} ({funds['time'].snapshot_ticker})",
            level_two,
        )
        self.assertIn(
            f"## {funds['koact'].display_name} ({funds['koact'].snapshot_ticker})",
            level_two,
        )
        self.assertIn("## 두 펀드 가격 성과 비교", level_two)
        assert_etf_only_headings(self, report)


class HolidayCrawler(FakeCrawler):
    """Answers a holiday with the previous session's basket, as the providers do."""

    def __init__(self, holidays: set[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.holidays = holidays

    def fetch_holdings(self, requested_date: str) -> dict[str, Any]:
        raw = super().fetch_holdings(requested_date)
        if requested_date in self.holidays:
            raw["date"] = raw["actual_date"] = "2026-09-23"
        return raw


class TodayRunTests(unittest.TestCase):
    """A run without --date fills skipped sessions and puts the desk block first."""

    def test_missing_weekdays_skip_weekends_and_keep_the_recent_end(self) -> None:
        self.assertEqual(workflow._missing_weekdays("2026-09-23", "2026-09-29"),
                         ["2026-09-24", "2026-09-25", "2026-09-28"])
        self.assertEqual(workflow._missing_weekdays("2026-09-23", "2026-09-29", limit=1), ["2026-09-28"])
        self.assertEqual(workflow._missing_weekdays(None, "2026-09-29"), [])

    def test_backfill_stores_only_days_the_provider_really_published(self) -> None:
        crawler = HolidayCrawler({"2026-09-24", "2026-09-25"},
                                 snapshot_dates={"time": ["2026-09-23"], "koact": ["2026-09-23"]})
        manifest: dict[str, Any] = {}
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(workflow, "BACKFILL_PAUSE_SEC", 0), \
                mock.patch.object(workflow, "save_normalized_snapshot",
                                  side_effect=lambda s: save_to_root(s, root=Path(temp_dir))), \
                mock.patch.object(workflow, "save_raw_capture",
                                  side_effect=lambda s: save_raw_to_root(s, root=Path(temp_dir) / "raw")):
            workflow._backfill_missing(crawler, "2026-09-29", manifest)
            stored = sorted(p.name for p in Path(temp_dir).glob("*.json"))
        self.assertEqual({(r["fund"], r["date"], r["status"]) for r in manifest["backfill"]}, {
            ("time", "2026-09-24", "not_published"), ("time", "2026-09-25", "not_published"),
            ("time", "2026-09-28", "stored"), ("koact", "2026-09-24", "not_published"),
            ("koact", "2026-09-25", "not_published"), ("koact", "2026-09-28", "stored")})
        funds = load_fund_configs()
        self.assertEqual(stored, sorted(f"2026-09-28_{funds[a].snapshot_ticker}.json" for a in ("time", "koact")))

    def test_backfill_stops_a_fund_at_its_first_provider_error(self) -> None:
        crawler = FakeCrawler(failing_alias="koact",
                              snapshot_dates={"time": ["2026-09-24"], "koact": ["2026-09-23"]})
        manifest: dict[str, Any] = {}
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(workflow, "BACKFILL_PAUSE_SEC", 0), \
                mock.patch.object(workflow, "save_normalized_snapshot",
                                  side_effect=lambda s: save_to_root(s, root=Path(temp_dir))), \
                mock.patch.object(workflow, "save_raw_capture",
                                  side_effect=lambda s: save_raw_to_root(s, root=Path(temp_dir) / "raw")):
            workflow._backfill_missing(crawler, "2026-09-29", manifest)
        koact = [r for r in manifest["backfill"] if r["fund"] == "koact"]
        self.assertEqual([r["status"] for r in koact], ["failed"])
        self.assertEqual([r["date"] for r in manifest["backfill"] if r["fund"] == "time"],
                         ["2026-09-25", "2026-09-28"])

    def test_desk_block_opens_the_core_conclusion_on_todays_run(self) -> None:
        def fake_desk(manifest: dict[str, Any]) -> list[str]:
            manifest["desk_digest_status"] = "generated"
            return ["### [내 7종목 점검] — fixture", "| DELL | 1 |", ""]

        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(workflow, "_desk_digest", side_effect=fake_desk):
            code, manifest = run_isolated_workflow(FakeCrawler(), Path(temp_dir), None, fetch=True)
            report = Path(manifest["report_path"]).read_text(encoding="utf-8")
        self.assertEqual((code, manifest["status"], manifest["report_contract_status"]), (0, "success", "passed"))
        core = report.split("## 두 펀드 가격 성과 비교")[0]
        self.assertLess(core.index("[내 7종목 점검]"), core.index(DIGEST_MARKER))
        self.assertIn("내 7종목 점검과 실매매 합의", core)

    def test_a_failing_desk_block_leaves_the_briefing_standing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch("desk.digest.briefing_block", side_effect=RuntimeError("no prices")):
            code, manifest = run_isolated_workflow(FakeCrawler(), Path(temp_dir), None, fetch=True)
            report = Path(manifest["report_path"]).read_text(encoding="utf-8")
        self.assertEqual((code, manifest["status"], manifest["desk_digest_status"]), (0, "success", "failed"))
        self.assertEqual([e["stage"] for e in manifest["errors"]], ["desk_digest"])
        self.assertNotIn("[내 7종목 점검]", report)

    def test_explicit_date_reruns_neither_backfill_nor_add_the_desk_block(self) -> None:
        crawler = FakeCrawler(snapshot_dates={"time": ["2026-07-01"], "koact": ["2026-07-01"]})
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(workflow, "_desk_digest") as desk:
            run_isolated_workflow(crawler, Path(temp_dir), "2026-07-10", fetch=True)
        desk.assert_not_called()
        self.assertEqual(crawler.fetch_calls, [("time", "2026-07-10"), ("koact", "2026-07-10")])


if __name__ == "__main__":
    unittest.main()
