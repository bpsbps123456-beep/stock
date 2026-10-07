"""Network-free end-to-end test of the fund report and the cross-fund digest."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from gbrc.config import load_fund_configs
from gbrc.etf import prices
from gbrc.etf.engine import BriefingEngine
from gbrc.etf.report_contract import validate_fund_report
from gbrc.etf.snapshot import normalize_snapshot
from gbrc.etf.storage import save_normalized_snapshot

SESSIONS = ["2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
            "2026-09-21", "2026-09-22", "2026-09-23"]
TARGET = SESSIONS[-1]
US_NAMES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def raw_snapshot(fund_id: str, date: str, qty: dict[str, float]) -> dict:
    holdings = [
        {"code": code, "name": code, "qty": q, "value_krw": q * 1000.0, "weight": 0.0}
        for code, q in qty.items()
    ]
    holdings.append({"code": "", "name": "Cash", "qty": 500.0, "value_krw": 500.0,
                     "weight": 0.0})
    return {"date": date, "actual_date": date, "etf_ticker": fund_id,
            "etf_name": fund_id, "holdings": holdings,
            "fetched_at": f"{date}T18:00:00+09:00"}


def seed(root: Path) -> None:
    for date in SESSIONS:
        time_qty = {f"{name} US EQUITY": 100.0 for name in US_NAMES}
        time_qty["009150"] = 100.0
        koact_qty = {f"{name} US EQUITY": 100.0 for name in US_NAMES}
        if date >= "2026-09-18":  # TIME 설정 유입: 전 종목 x1.1
            time_qty = {code: q * 1.1 for code, q in time_qty.items()}
        if date == TARGET:
            time_qty["AAA US EQUITY"] *= 1.3  # TIME 선별 매수 +30%
            koact_qty["AAA US EQUITY"] *= 1.2  # KoAct 선별 매수 +20%
        for fund_id, qty in (("456600", time_qty), ("2ETFQ1", koact_qty)):
            save_normalized_snapshot(normalize_snapshot(raw_snapshot(fund_id, date, qty)), root)


def fake_close_history(tickers, start_date, end_date):
    """Close = day of month, so each cell reveals which dates were used."""
    history = {}
    for ticker in tickers:
        if not ticker:
            continue
        if ticker.endswith(".KS"):
            index = pd.to_datetime(SESSIONS)
        else:
            index = pd.bdate_range("2026-09-01", "2026-09-22")
        history[ticker] = pd.Series([float(ts.day) for ts in index], index=index)
    return history


class BriefingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        seed(root)
        self.engine = BriefingEngine(root)
        self.time = load_fund_configs()["time"]
        self.patch = mock.patch.object(prices, "load_close_history", side_effect=fake_close_history)
        self.patch.start()
        prices._SESSION_CACHE.clear()

    def tearDown(self) -> None:
        self.patch.stop()
        prices._SESSION_CACHE.clear()
        self.temp.cleanup()

    def test_fund_report_separates_real_trades_from_creation_scaling(self) -> None:
        report = self.engine.fund_report(self.time, TARGET)
        validate_fund_report(report.strip(), len(US_NAMES) + 2)

        flow_section = report.split("[수량 기준 실매매]", 1)[1].split("### D-1 compare", 1)[0]
        self.assertIn("AAA US EQUITY", flow_section)
        self.assertIn("+30.0%", flow_section)
        self.assertNotIn("BBB US EQUITY", flow_section)  # 설정 유입 x1.1은 실매매가 아니다
        self.assertIn("공통 배율 x1.1000", flow_section)

    def test_one_glance_returns_follow_the_nav_valuation_dates(self) -> None:
        report = self.engine.fund_report(self.time, TARGET)
        glance = report.split("[One-glance table]", 1)[1].split("[포트폴리오 해석]", 1)[0]
        aaa = next(line for line in glance.splitlines() if " AAA US EQUITY " in line)
        # 해외: 9/22 PDF=9/18 종가, 9/23 PDF=9/21 종가 → 21/18-1 = +16.7%
        self.assertIn("(+16.7%)", aaa)
        kr = next(line for line in glance.splitlines() if " 009150 " in line)
        # 국내: 9/22 PDF=9/21 종가, 9/23 PDF=9/22 종가 → 22/21-1 = +4.8%
        self.assertIn("(+4.8%)", kr)

    def test_digest_lists_common_buy_and_us_watch_list(self) -> None:
        digest = self.engine.core_digest(TARGET)
        self.assertIn("공통 실매수 1: AAA (TIME 매수 +30%, KoAct 매수 +20%)", digest)
        self.assertIn("[TIME 미국 상장 Top 10 관찰군]", digest)
        self.assertNotIn("009150", digest.split("관찰군]", 1)[1])


if __name__ == "__main__":
    unittest.main()
