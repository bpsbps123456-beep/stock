"""Orchestrate the two-fund daily ETF workflow (docs/etf_system.md §2)."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

from gbrc.config import FundConfig, load_fund_configs

from .engine import BriefingEngine
from .report_contract import (
    validate_daily_report,
    validate_fund_report,
    validate_performance_report,
)
from .snapshot import normalize_snapshot, snapshot_summary
from .storage import (
    DERIVED_BRIEFINGS_DIR,
    NORMALIZED_HOLDINGS_DIR,
    STATE_RUNS_DIR,
    atomic_write_json,
    atomic_write_text,
    load_normalized_snapshot,
    save_raw_capture,
    save_normalized_snapshot,
)


KST = timezone(timedelta(hours=9))
BACKFILL_LIMIT = 10        # weekdays; a longer gap is filled from the most recent end
BACKFILL_PAUSE_SEC = 1.0   # between provider calls (KoAct has answered HTTP 429 before)


def _engine() -> BriefingEngine:
    """The engine every stage of a run shares; tests replace it with a fake."""
    return BriefingEngine(NORMALIZED_HOLDINGS_DIR)


def _latest_date(
    engine: BriefingEngine,
    fund: FundConfig,
    through_date: str | None = None,
) -> str | None:
    dates = engine.snapshot_dates(fund)
    if through_date:
        dates = [date for date in dates if date <= through_date]
    return dates[-1] if dates else None


def _missing_weekdays(latest: str | None, today: str, limit: int = BACKFILL_LIMIT) -> list[str]:
    """Weekdays after the latest stored snapshot and before today. KRX holidays are
    rejected later, when the provider answers with an earlier as-of date."""
    if not latest:
        return []
    day, end = date.fromisoformat(latest) + timedelta(days=1), date.fromisoformat(today)
    days = []
    while day < end:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day += timedelta(days=1)
    return days[-limit:]


def _store_published_day(engine: BriefingEngine, fund: FundConfig, day: str) -> dict[str, Any]:
    """Fetch one provider day; store it only if the provider really published that day.

    A holiday's answer is the previous basket (as-of differs) and is never saved as that day.
    """
    raw = engine.fetch_holdings(fund, day)
    if not raw or not raw.get("holdings"):
        return {"fund": fund.alias, "date": day, "status": "not_published"}
    normalized = normalize_snapshot(raw)
    if normalized["as_of_date"] != day:
        return {"fund": fund.alias, "date": day, "status": "not_published",
                "provider_as_of": normalized["as_of_date"]}
    save_raw_capture(raw)
    _, storage_status = save_normalized_snapshot(normalized)
    return {"fund": fund.alias, "date": day, "status": "stored", "storage_status": storage_status}


def _fill_days(engine: BriefingEngine, fund: FundConfig, days: list[str],
               records: list[dict[str, Any]]) -> None:
    """The first provider error stops this fund (usually rate limiting)."""
    for day in days:
        try:
            records.append(_store_published_day(engine, fund, day))
        except Exception as exc:
            records.append({"fund": fund.alias, "date": day, "status": "failed",
                            "error_type": type(exc).__name__, "error": str(exc)})
            break
        finally:
            time.sleep(BACKFILL_PAUSE_SEC)


def _backfill_missing(engine: BriefingEngine, today: str, manifest: dict[str, Any]) -> None:
    """Store validated snapshots for sessions a skipped briefing left empty; today's run
    carries on whatever happens."""
    records = manifest.setdefault("backfill", [])
    for fund in load_fund_configs().values():
        _fill_days(engine, fund, _missing_weekdays(_latest_date(engine, fund), today), records)


def backfill_range(fund: FundConfig, start: str, end: str,
                   engine: BriefingEngine | None = None) -> list[dict[str, Any]]:
    """Manual restore: every weekday in [start, end] without a validated snapshot."""
    engine = engine or _engine()
    have = set(engine.snapshot_dates(fund))
    days, day = [], date.fromisoformat(start)
    while day <= date.fromisoformat(end):
        if day.weekday() < 5 and day.isoformat() not in have:
            days.append(day.isoformat())
        day += timedelta(days=1)
    records: list[dict[str, Any]] = []
    _fill_days(engine, fund, days, records)
    return records


def _desk_digest(manifest: dict[str, Any]) -> list[str]:
    """'내 7종목 점검' from the user's night desk, first under the core conclusion.

    The desk is optional here: without it, or when it fails, the briefing stands and the
    manifest records why.
    """
    try:
        from desk.digest import briefing_block
    except ImportError:
        manifest["desk_digest_status"] = "unavailable"
        return []
    try:
        lines = briefing_block(_engine())
    except Exception as exc:
        manifest["desk_digest_status"] = "failed"
        manifest["errors"].append({
            "stage": "desk_digest",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        return []
    manifest["desk_digest_status"] = "generated"
    return [*lines, ""]


def _core_summary(manifest: dict[str, Any]) -> str:
    status = manifest["status"]
    dates = sorted({
        str(fund.get("as_of_date"))
        for fund in manifest.get("funds", {}).values()
        if fund.get("as_of_date")
    })
    date_text = ", ".join(dates) if dates else "N/A"
    filled = sorted({r["date"] for r in manifest.get("backfill", []) if r["status"] == "stored"})
    filled_text = f" 건너뛴 날 {', '.join(filled)} 자료를 채웠습니다." if filled else ""
    first = ("내 7종목 점검과 실매매 합의·관찰군을" if manifest.get("desk_digest_status") == "generated"
             else "실매매 합의와 관찰군을")
    if status == "success":
        return (
            f"두 펀드 자료가 {date_text} 기준으로 검증됐습니다.{filled_text} "
            f"아래 {first} 먼저 보고, 가격 성과는 흐름 해석 뒤 참고자료로 확인하세요."
        )
    if status == "stale":
        return (
            f"데이터 지연 — 최신 검증 기준일은 {date_text}입니다.{filled_text} "
            "요청일의 신규 투자 판단으로 사용하지 않습니다."
        )
    if status == "partial_success":
        return "부분 실패 — 정상 생성된 범위만 보존했으며 신규 투자 판단은 보류합니다."
    return "실행 실패 — 유효한 두 펀드 브리핑을 생성하지 못했습니다."


def _assemble_report(
    header_parts: list[str],
    performance_parts: list[str],
    fund_parts: list[str],
    manifest: dict[str, Any],
    digest_parts: list[str] | None = None,
) -> list[str]:
    digest = digest_parts if manifest.get("status") in {"success", "stale"} else []
    return [
        *header_parts,
        "## 핵심 결론",
        "",
        _core_summary(manifest),
        "",
        *(digest or []),
        *performance_parts,
        *fund_parts,
    ]


def _core_digest(engine: BriefingEngine, target: str, manifest: dict[str, Any]) -> list[str]:
    """Cross-fund real-trade consensus and the US-listed watch list."""
    try:
        text = engine.core_digest(target).strip()
    except Exception as exc:  # the briefing stands without the digest
        manifest["core_digest_status"] = "failed"
        manifest["errors"].append({
            "stage": "core_digest",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        return []
    manifest["core_digest_status"] = "generated"
    return [text, ""] if text else []


def run_daily_etf_workflow(
    requested_date: str | None = None,
    *,
    fetch: bool = True,
) -> tuple[int, dict[str, Any]]:
    engine = _engine()
    run_started = datetime.now(KST)
    # Today's run (no --date) also fills skipped sessions and adds the desk block;
    # explicit-date reruns reproduce the historical report and do neither.
    today_run = requested_date is None
    requested_date = requested_date or run_started.strftime("%Y-%m-%d")
    datetime.strptime(requested_date, "%Y-%m-%d")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_started.strftime("%Y%m%dT%H%M%S%f%z"),
        "workflow": "ETF_DAILY",
        "requested_date": requested_date,
        "started_at": run_started.isoformat(timespec="seconds"),
        "funds": {},
        "errors": [],
    }
    header_parts = [
        f"# GBRC 두 펀드 일일 브리핑 — 실행일 {requested_date}",
        "",
        "> ETF/NAV 구성종목과 가격 성과만 사용합니다. 옵션·PER는 포함하지 않습니다.",
        "",
    ]
    fund_parts: list[str] = []
    performance_parts: list[str] = []
    if today_run and fetch:
        _backfill_missing(engine, requested_date, manifest)

    report_aliases: list[str] = []
    current_aliases: list[str] = []
    for alias, fund in load_fund_configs().items():
        fund_run: dict[str, Any] = {
            "fund_id": fund.snapshot_ticker,
            "expected": True,
            "fetch_requested": fetch,
        }
        manifest["funds"][alias] = fund_run
        try:
            if fetch:
                raw = engine.fetch_holdings(fund, requested_date)
                if not raw or not raw.get("holdings"):
                    raise RuntimeError("provider returned no holdings")
                raw_path, raw_status = save_raw_capture(raw)
                normalized = normalize_snapshot(raw)
                path, storage_status = save_normalized_snapshot(normalized)
                fund_run.update(snapshot_summary(normalized))
                fund_run["raw_capture_status"] = raw_status
                fund_run["raw_capture_path"] = str(raw_path) if raw_path else None
                fund_run["storage_status"] = storage_status
                fund_run["snapshot_path"] = str(path)
                report_date = normalized["as_of_date"]
                if report_date == requested_date:
                    current_aliases.append(alias)
                else:
                    fund_run["stale_for_requested_date"] = True
            else:
                report_date = _latest_date(engine, fund, requested_date)
                if not report_date:
                    raise RuntimeError("no normalized snapshot available")
                fund_run.update({
                    "quality_status": (
                        "existing" if report_date == requested_date else "existing_stale"
                    ),
                    "as_of_date": report_date,
                    "storage_status": "skipped",
                })
                if report_date == requested_date:
                    current_aliases.append(alias)
                else:
                    fund_run["stale_for_requested_date"] = True

            fund_report = engine.fund_report(fund, report_date).strip()
            expected_holdings = fund_run.get("holdings_count")
            if expected_holdings is None:
                snapshot_path = (
                    NORMALIZED_HOLDINGS_DIR
                    / f"{report_date}_{fund.snapshot_ticker}.json"
                )
                expected_holdings = load_normalized_snapshot(
                    snapshot_path
                )["holdings_count"]
                fund_run["holdings_count"] = expected_holdings
            validate_fund_report(fund_report, int(expected_holdings))
            fund_parts.extend([
                f"## {fund.display_name} ({fund.snapshot_ticker})",
                "",
                fund_report,
                "",
            ])
            fund_run["report_status"] = "generated"
            report_aliases.append(alias)
        except Exception as exc:  # run manifest must retain provider failures
            fund_run.update({
                "quality_status": "failed",
                "report_status": "not_generated",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            manifest["errors"].append({
                "fund": alias,
                "stage": "fetch_validate_compare",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            fund_parts.extend([
                f"## {fund.display_name} ({fund.snapshot_ticker})",
                "",
                f"> 생성 실패: {type(exc).__name__}: {exc}",
                "",
            ])

    digest_parts: list[str] = []
    if len(report_aliases) == 2:
        performance_dates = [
            manifest["funds"][alias].get("as_of_date")
            for alias in report_aliases
        ]
        performance_target = (
            performance_dates[0]
            if performance_dates[0] == performance_dates[1]
            else min(date for date in performance_dates if date)
        )
        digest_parts = _core_digest(engine, performance_target, manifest)
        try:
            performance_report = engine.performance_report(performance_target).strip()
            validate_performance_report(performance_report)
            performance_parts.extend([
                "## 두 펀드 가격 성과 비교",
                "",
                performance_report,
                "",
            ])
            manifest["performance_status"] = (
                "generated" if len(current_aliases) == 2 else "generated_stale"
            )
        except Exception as exc:
            manifest["performance_status"] = "failed"
            manifest["errors"].append({
                "stage": "performance",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            performance_parts.extend([
                "## 두 펀드 가격 성과 비교",
                "",
                f"> 생성 실패: {type(exc).__name__}: {exc}",
                "",
            ])
    else:
        manifest["performance_status"] = "not_generated_partial_run"
        performance_parts.extend([
            "## 두 펀드 가격 성과 비교",
            "",
            "> 두 펀드가 모두 유효하지 않아 성과 승자를 계산하지 않았습니다.",
            "",
        ])
    if today_run and "time" in report_aliases:
        digest_parts = [*_desk_digest(manifest), *digest_parts]

    finished = datetime.now(KST)
    manifest["finished_at"] = finished.isoformat(timespec="seconds")
    if (
        len(report_aliases) == 2
        and len(current_aliases) == 2
        and manifest["performance_status"] == "generated"
    ):
        manifest["status"] = "success"
    elif (
        len(report_aliases) == 2
        and len(current_aliases) < 2
        and manifest["performance_status"] == "generated_stale"
    ):
        manifest["status"] = "stale"
    elif report_aliases:
        manifest["status"] = "partial_success"
    else:
        manifest["status"] = "failed"

    report_parts = _assemble_report(
        header_parts, performance_parts, fund_parts, manifest, digest_parts,
    )
    fund_headings = [
        f"## {fund.display_name} ({fund.snapshot_ticker})"
        for fund in load_fund_configs().values()
    ]
    try:
        validate_daily_report("\n".join(report_parts), fund_headings)
        manifest["report_contract_status"] = "passed"
    except Exception as exc:
        manifest["report_contract_status"] = "failed"
        manifest["errors"].append({
            "stage": "report_contract",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        manifest["status"] = "partial_success" if report_aliases else "failed"
        report_parts = _assemble_report(
            header_parts, performance_parts, fund_parts, manifest, digest_parts,
        )

    if manifest["status"] == "success":
        report_path = DERIVED_BRIEFINGS_DIR / f"{requested_date}.md"
    else:
        report_path = DERIVED_BRIEFINGS_DIR / (
            f"{requested_date}_{manifest['run_id']}_{manifest['status']}.md"
        )
    atomic_write_text(report_path, "\n".join(report_parts))
    manifest["report_path"] = str(report_path)
    manifest_path = STATE_RUNS_DIR / f"{manifest['run_id']}_ETF_DAILY.json"
    manifest["manifest_path"] = str(manifest_path)
    atomic_write_json(manifest_path, manifest)

    exit_code = 0 if manifest["status"] == "success" else 2
    return exit_code, manifest


def manifest_as_text(manifest: dict[str, Any]) -> str:
    lines = [
        f"ETF_DAILY {manifest['status']} — {manifest['requested_date']}",
        f"report: {manifest.get('report_path')}",
        f"manifest: {manifest.get('manifest_path')}",
    ]
    for alias, fund in manifest.get("funds", {}).items():
        lines.append(
            f"- {alias}: {fund.get('quality_status')} "
            f"as_of={fund.get('as_of_date', 'N/A')} "
            f"storage={fund.get('storage_status', 'N/A')}"
        )
        if fund.get("error"):
            lines.append(f"  error: {fund['error']}")
    return "\n".join(lines)
