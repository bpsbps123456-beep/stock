"""바닥 감지기(desk/bottom.py, §13.6): 연구 숫자 재현, 상태 판정, 장중 봉·분할 섞임 방어, HTTP 연결."""
from __future__ import annotations

import json
import threading
import unittest
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from desk import bottom, server

ROOT = Path(__file__).resolve().parents[1]


def series(values, start="2026-01-02"):
    return pd.Series(np.asarray(values, float), index=pd.bdate_range(start, periods=len(values)))


def drawdown_path(rebound: float):
    """60일 동안 100→110으로 오르고, 6일 동안 95까지 빠진 뒤 저점 대비 rebound만큼 오른다."""
    up = list(np.linspace(100, 110, 60))
    down = list(np.linspace(108, 95, 6))
    return up + down + [95 * (1 + rebound)]


class ResearchParityTests(unittest.TestCase):
    """연구 엔진(bc_lib)과 같은 SOXX 스냅샷에서 같은 숫자가 나와야 한다(두 정의가 갈라지지 않게)."""

    @classmethod
    def setUpClass(cls):
        p = pd.read_csv(ROOT / "research/news_turning_points_20261025/prices.csv", header=[0, 1], index_col=0)
        p.index = pd.to_datetime(p.index, errors="coerce")
        p = p[p.index.notna()].dropna().sort_index()
        u = pd.read_csv(ROOT / "data/derived/analysis/leverage_flood/results/us_prices_close.csv",
                        encoding="utf-8-sig", index_col=0, parse_dates=True)["SOXX"].dropna()
        close = p["close"]["SOXX"].astype(float).combine_first(u).sort_index().loc[:"2026-10-23"]  # bc_lib.load_soxx와 같은 이음
        cls.fr = bottom.frame(close)
        cls.mask = bottom._between(cls.fr.index, "2024-01-02", "2026-10-23")
        cls.calib = pd.read_csv(ROOT / "data/derived/analysis/bottom_confirm/results/calib_up.csv", encoding="utf-8-sig")

    def test_band_rates_match_research_table(self):
        rates = bottom.hold_rates(self.fr, self.mask)
        ref = self.calib[(self.calib["kind"] == "SOXX") & (self.calib["period"] == "2024~2026")].set_index("bin")
        for i, name in ((0, "0~3%"), (1, "3~5%")):
            self.assertEqual(rates["bands"][i]["n"], int(ref.loc[name, "days"]))
            self.assertAlmostEqual(rates["bands"][i]["rate"], float(ref.loc[name, "hold20"]), places=12)

    def test_seven_percent_days_match_report(self):
        band = bottom.hold_rates(self.fr, self.mask)["bands"][3]  # 7% 이상
        self.assertEqual(band["n"], 147)                          # report.md: 147일 99.3%, 실패 하락 1개
        self.assertEqual(round(band["rate"] * band["n"]), 146)


class StatusTests(unittest.TestCase):
    def test_kinds_follow_drawdown_and_rebound(self):
        self.assertEqual(bottom.status(bottom.frame(series(np.linspace(100, 130, 80))))["kind"], "calm")
        falling = bottom.status(bottom.frame(series(drawdown_path(0.05))))
        self.assertEqual(falling["kind"], "falling")
        self.assertAlmostEqual(falling["low"], 95)
        self.assertAlmostEqual(falling["lines"][0]["price"], 95 * 1.07)
        confirmed = bottom.status(bottom.frame(series(drawdown_path(0.074))))
        self.assertEqual(confirmed["kind"], "confirmed")
        self.assertEqual(confirmed["band"], 3)
        self.assertAlmostEqual(confirmed["up"], 0.074)

    def test_calm_trigger_and_days_left(self):
        st = bottom.status(bottom.frame(series(np.linspace(100, 130, 80))))
        self.assertAlmostEqual(st["dd_trigger"], 130 * 0.9)
        self.assertIsNone(st["dd_left"])
        deep_today = bottom.status(bottom.frame(series(list(np.linspace(100, 110, 60)) + [98])))
        self.assertEqual(deep_today["dd_left"], 19)              # −10% 날이 오늘이면 19거래일 더 하락 국면

    def test_signal_outcomes(self):
        base = drawdown_path(0.08)
        held = bottom.signals(bottom.frame(series(base + [100] * 20)), start="2026-01-02")
        self.assertEqual([s["status"] for s in held], ["held"])
        broke = bottom.signals(bottom.frame(series(base + [99, 94, 100])), start="2026-01-02")
        self.assertEqual((broke[0]["status"], broke[0]["day"]), ("broke", 2))
        watching = bottom.signals(bottom.frame(series(base + [101] * 3)), start="2026-01-02")
        self.assertEqual((watching[0]["status"], watching[0]["day"]), ("watching", 3))

    def test_new_peak_inside_cooldown_starts_new_signal(self):
        first = drawdown_path(0.08)                               # 67번째 날 확인
        rally = list(np.linspace(103, 125, 8))                   # 새 60일 고점
        second = list(np.linspace(120, 108, 4)) + [108 * 1.08]   # 다시 −13.6% 뒤 8% 반등(20거래일 안)
        sigs = bottom.signals(bottom.frame(series(first + rally + second)), start="2026-01-02")
        self.assertEqual(len(sigs), 2)
        self.assertLess(len(first + rally + second) - 1 - (len(first) - 1), bottom.COOL + 1)


class DataGuardTests(unittest.TestCase):
    def test_unfinished_session_is_dropped(self):
        s = pd.Series([1.0, 2.0, 3.0], index=pd.DatetimeIndex(["2026-10-26", "2026-10-27", "2026-10-28"]))
        during = datetime(2026, 10, 28, 15, 0, tzinfo=timezone.utc)   # 뉴욕 11시, 장중
        self.assertEqual(list(bottom.completed(s, during).index.strftime("%m-%d")), ["10-26", "10-27"])
        after = datetime(2026, 10, 28, 21, 0, tzinfo=timezone.utc)
        self.assertEqual(len(bottom.completed(s, after)), 3)

    def test_split_mix_is_refused(self):
        with self.assertRaises(bottom.BottomError):
            bottom.check_split(series([510, 512, 171, 172]), "SOXX")
        bottom.check_split(series([510, 512, 520, 470]), "SOXX")   # 큰 하락이어도 40% 안이면 통과

    def test_provisional_uses_quote_after_last_close_only(self):
        close = drawdown_path(0.05)
        dates = list(pd.bdate_range("2026-06-01", periods=len(close)).strftime("%Y-%m-%d"))
        data = {"tail": {"dates": dates, "close": close}}
        cal = bottom.market_calendar()
        closed = cal.session_close(dates[-1]).timestamp()
        quote = {"price": 95 * 1.075, "session": "장전", "time_kst": "10-28 22:00", "epoch": closed + 3600}
        prov = bottom.provisional(data, quote)
        self.assertEqual(prov["kind"], "confirmed")
        self.assertIsNone(bottom.provisional(data, {**quote, "epoch": closed - 60}))

    def test_live_ratio_mismatch_is_flagged(self):
        data = {"tail": {"dates": ["2026-11-04"], "close": [510.0]}}
        quote = {"price": 171.0, "session": "장전", "time_kst": "11-05 18:00", "epoch": 1.0}
        with mock.patch.dict(bottom._LIVE, {"epoch": 9e18, "quote": quote, "error": None}):
            self.assertEqual(bottom.live(data), (None, "MISMATCH"))


class BottomHttpTests(unittest.TestCase):
    def test_endpoint_serves_payload(self):
        fake = {"symbol": "SOXX", "status": {"kind": "calm"}}
        httpd = server.DeskServer(("127.0.0.1", 0), server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            with mock.patch.object(bottom, "payload", return_value=fake) as called:
                with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_port}/api/bottom?refresh=1") as response:
                    self.assertEqual(json.load(response), fake)
                called.assert_called_once_with(force=True)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(3)


if __name__ == "__main__":
    unittest.main()
