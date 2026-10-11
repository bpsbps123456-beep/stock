"""추세 등급 contract: the score, ladder, gates and holdings must stay what docs/desk_system.md §11 says,
using only prices the desk really has on each day."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from desk import trend
from gbrc.etf.storage import NORMALIZED_HOLDINGS_DIR


def bars(closes, start="2025-01-02"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = np.asarray(closes, float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c}, index=idx)


def trend_path(n=260, daily=0.002, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    r = daily + noise * rng.standard_normal(n)
    return 100 * np.exp(np.cumsum(r))


class ScoreTest(unittest.TestCase):
    def test_clenow_is_annual_growth_times_fit(self):
        c = pd.Series(trend_path(200, daily=0.001))
        q = trend.clenow(c, 126)
        self.assertTrue(np.isnan(q.iloc[124]))
        self.assertAlmostEqual(q.iloc[-1], np.exp(0.252) - 1, places=6)  # 매끄러운 직선이면 R² = 1
        noisy = trend.clenow(pd.Series(trend_path(200, daily=0.001, noise=0.03)), 126)
        self.assertLess(noisy.iloc[-1], q.iloc[-1])  # 같은 기울기라도 들쭉날쭉하면 작아진다

    def test_long_averages_start_at_sma_min_bars(self):
        c = pd.Series(np.arange(1, 151, dtype=float))
        self.assertTrue(np.isnan(trend.sma(c, 200).iloc[trend.SMA_MIN - 2]))
        self.assertAlmostEqual(trend.sma(c, 200).iloc[trend.SMA_MIN - 1], np.mean(np.arange(1, trend.SMA_MIN + 1)))

    def test_indicators_use_only_past_closes(self):
        b = bars(trend_path(300, noise=0.02, seed=3))
        full = trend.indicators(b)
        cut = trend.indicators(b.iloc[:250])
        cols = ["L", "M", "H", "Q", "up", "down"]
        pd.testing.assert_frame_equal(full.iloc[:250][cols], cut[cols])

    def test_strongest_trend_ranks_first_and_short_listings_are_left_out(self):
        ind = {f"S{i}": trend.indicators(bars(trend_path(260, daily=0.0005 * i, noise=0.004, seed=i))) for i in range(12)}
        ind["NEW"] = trend.indicators(bars(trend_path(100, daily=0.01), start="2025-06-02"))  # 상장 100거래일
        cal = ind["S0"].index[-40:]
        pct, raw, parts = trend.score_table(ind, cal)
        self.assertEqual(pct.iloc[-1].idxmax(), "S11")
        self.assertEqual(raw.iloc[-1].idxmax(), "S11")
        self.assertEqual(pct.iloc[-1].idxmin(), "S0")
        self.assertTrue(pct["NEW"].isna().all())
        self.assertEqual(set(parts), set(trend.PARTS))

    def test_smoothing_keeps_one_good_day_from_jumping_to_the_top(self):
        cal = pd.bdate_range("2026-01-01", periods=20)
        ind = {}
        for i in range(10):
            d = pd.DataFrame({k: np.full(20, i / 10) for k in trend.PARTS}, index=cal)
            d["bars"] = 500
            ind[f"T{i}"] = d
        jump = ind["T0"].copy()
        jump.iloc[-1, :len(trend.PARTS)] = 9.0  # 마지막 하루만 1등
        ind["T0"] = jump
        pct, raw, _ = trend.score_table(ind, cal)
        self.assertLess(pct["T0"].iloc[-1], 0.9)  # 10일 평균이라 하루로 S가 되지 않는다
        self.assertEqual(raw["T0"].iloc[-1], 1.0)  # 당일 점수는 바로 1등
        # 순서 계약: 재료 백분위 평균 → 최근 SMOOTH일 평균 → 다시 백분위
        parts = [pd.DataFrame({s: d[k] for s, d in ind.items()}).rank(axis=1, pct=True) for k in trend.PARTS]
        raw_composite = sum(parts) / len(parts)
        expected = raw_composite.rolling(trend.SMOOTH, min_periods=1).mean().rank(axis=1, pct=True)
        pd.testing.assert_frame_equal(pct, expected[pct.columns], check_names=False)

    def test_outside_names_are_placed_in_the_reference_ranking_without_moving_it(self):
        cal = pd.bdate_range("2026-01-01", periods=1)
        x = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 2.5, 9.0, np.nan]], index=cal,
                         columns=["A", "B", "C", "D", "JP", "HK", "KR"])
        ref = {"A", "B", "C", "D"}
        out = trend.rank_vs(x, ref)
        pd.testing.assert_series_equal(out.loc[cal[0], list(ref)].sort_index(),
                                       x[sorted(ref)].rank(axis=1, pct=True).iloc[0].sort_index(), check_names=False)
        self.assertAlmostEqual(out.at[cal[0], "JP"], 0.625)  # B와 C 사이 = 4종목 중 2.5등
        self.assertEqual(out.at[cal[0], "HK"], 1.0)
        self.assertTrue(np.isnan(out.at[cal[0], "KR"]))


class LadderTest(unittest.TestCase):
    def test_percentile_floors(self):
        cases = {0.95: "S", 0.9: "A", 0.71: "A", 0.7: "B", 0.31: "B", 0.3: "C", 0.11: "C", 0.1: "D", 0.0: "D"}
        for p, g in cases.items():
            self.assertEqual(trend.grade_of(p), g, p)
        self.assertIsNone(trend.grade_of(float("nan")))
        self.assertIsNone(trend.grade_of(None))

    def test_gates_need_a_rising_stack_for_s_and_a_and_force_d_below_a_falling_stack(self):
        self.assertEqual(trend.gate("S", up=True, down=False), "S")
        self.assertEqual(trend.gate("S", up=False, down=False), "B")
        self.assertEqual(trend.gate("A", up=np.False_, down=np.False_), "B")
        self.assertEqual(trend.gate("A", up=np.True_, down=np.False_), "A")
        self.assertEqual(trend.gate("B", up=False, down=True), "D")
        self.assertEqual(trend.gate("S", up=True, down=True), "D")
        self.assertEqual(trend.gate("C", up=float("nan"), down=float("nan")), "C")
        self.assertIsNone(trend.gate(None, up=True, down=False))

    def test_grade_table_applies_gates_per_day(self):
        cal = pd.bdate_range("2026-01-01", periods=3)
        pct = pd.DataFrame({"X": [0.95, 0.95, 0.05]}, index=cal)
        ind = {"X": pd.DataFrame({"up": [True, False, True], "down": [False, False, True]}, index=cal)}
        self.assertEqual(list(trend.grade_table(pct, ind, cal)["X"]), ["S", "B", "D"])

    def test_missing_scores_stay_none_not_nan(self):
        # pandas 3은 글자·None 목록을 문자열형으로 바꾸며 None을 NaN으로 만든다. 미산정이 '등급 있음'으로 세어지면 안 된다.
        cal = pd.bdate_range("2026-01-01", periods=3)
        pct = pd.DataFrame({"X": [0.95, 0.5, np.nan]}, index=cal)
        ind = {"X": pd.DataFrame({"up": [True] * 3, "down": [False] * 3}, index=cal)}
        g = trend.grade_table(pct, ind, cal)["X"]
        self.assertEqual(list(g), ["S", "B", None])
        row = trend.stock_row({"symbol": "X", "funds": {}}, None, [np.nan], np.nan, {}, np.nan)
        self.assertIsNone(row["grade"])
        self.assertIsNone(row["raw"])

    def test_streak_counts_the_unbroken_run_ending_today(self):
        self.assertEqual(trend._streak(["A", "S", "S", "S"]), 3)
        self.assertEqual(trend._streak(["S", None]), 0)
        self.assertEqual(trend._streak([]), 0)

    def test_realized_groups_returns_by_the_grade_days_ago(self):
        cal = pd.bdate_range("2026-01-01", periods=4)
        letters = pd.DataFrame({"X": ["S", "S", "S", "S"], "Y": ["D", "D", "D", "C"], "Z": [None] * 4}, index=cal)
        close = pd.DataFrame({"X": [100, 101, 105, 110.0], "Y": [50, 50, 49, 45.0], "Z": [1, 1, 1, 1.0]}, index=cal)
        r = trend.realized(letters, close, pd.Series([10, 10, 10, 11.0], index=cal), days=3)
        self.assertEqual(r["from"], "2026-01-01")
        self.assertEqual(r["by_grade"]["S"], {"n": 1, "ret": 10.0})
        self.assertEqual(r["by_grade"]["D"], {"n": 1, "ret": -10.0})
        self.assertEqual(r["by_grade"]["A"]["n"], 0)
        self.assertEqual(r["all"], 0.0)
        self.assertEqual(r["bench"], 10.0)
        self.assertIsNone(trend.realized(letters, close, close["X"], days=4))


class SessionTest(unittest.TestCase):
    def test_todays_bar_is_dropped_until_that_market_closes(self):
        b = bars([1, 2, 3], start="2026-10-07")  # 10/7·8·9
        ny_open = datetime(2026, 10, 9, 23, 0, tzinfo=trend.config.KST)  # 뉴욕 10:00
        ny_closed = datetime(2026, 10, 10, 6, 0, tzinfo=trend.config.KST)  # 뉴욕 17:00
        self.assertEqual(len(trend.settled(b, "미국", ny_open)), 2)
        self.assertEqual(len(trend.settled(b, "미국", ny_closed)), 3)
        seoul_midday = datetime(2026, 10, 9, 12, 0, tzinfo=trend.config.KST)
        self.assertEqual(len(trend.settled(b, "한국", seoul_midday)), 2)
        next_morning = datetime(2026, 10, 10, 9, 0, tzinfo=trend.config.KST)
        self.assertEqual(len(trend.settled(b, "한국", next_morning)), 3)  # 지난 날짜 봉은 그대로


class HoldingsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for fund in ("456600", "2ETFQ1"):
            src = sorted(NORMALIZED_HOLDINGS_DIR.glob(f"*_{fund}.json"))[-1]
            shutil.copy(src, self.tmp / src.name)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_both_funds_merge_into_one_row_per_company_without_futures_or_cash(self):
        rows, dates = trend.holdings(self.tmp)
        self.assertEqual(set(dates), {"TIME", "KoAct"})
        keys = [r["key"] for r in rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertFalse(any("FUTURES" in k or "INDEX" in k for k in keys))
        self.assertTrue(all(r["symbol"] for r in rows))
        both = [r for r in rows if len(r["funds"]) == 2]
        self.assertTrue(both)  # 두 펀드가 함께 든 종목은 한 줄에 두 비중
        for r in rows:
            if r["market"] == "한국":
                self.assertTrue(r["symbol"].endswith((".KS", ".KQ")))
        weights = [max(r["funds"].values()) for r in rows]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_share_classes_join_and_short_listings_get_a_proxy(self):
        rows, _ = trend.holdings(self.tmp)
        self.assertLessEqual(sum(1 for r in rows if r["symbol"] in ("GOOG", "GOOGL")), 1)
        sk = [r for r in rows if r["symbol"] == "SKHY"]
        if sk:
            self.assertEqual(sk[0].get("proxy"), "000660.KS")


def fake_bars(n, end="2026-10-09", daily=0.001, seed=0):
    idx = pd.bdate_range(end=end, periods=n)
    c = trend_path(n, daily=daily, noise=0.01, seed=seed)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c}, index=idx)


class ComputeTest(unittest.TestCase):
    """compute() 끝단: 해외·대체 이력 종목이 미국 순위를 바꾸지 않고, 화면 숫자가 등급 날짜에 맞는다."""

    NOW = datetime(2026, 10, 12, 20, 0, tzinfo=trend.config.KST)  # 한국 장 마감 뒤, 미국 월요일 장 전

    def rows(self, extra=True):
        us = [{"key": f"U{i} US EQUITY", "symbol": f"U{i}", "name": f"U{i}", "market": "미국", "funds": {"TIME": 1.0},
               "sector": "x", "sub": "x", "ko": None} for i in range(12)]
        if not extra:
            return us
        return us + [
            {"key": "000001", "symbol": "000001.KS", "name": "KR", "market": "한국", "funds": {"TIME": 1.0}, "sector": "x", "sub": "x", "ko": None},
            {"key": "ADR US EQUITY", "symbol": "ADR", "name": "ADR", "market": "미국", "funds": {"TIME": 1.0}, "sector": "x", "sub": "x",
             "ko": None, "proxy": "999999.KS"},
        ]

    def bars(self):
        out = {f"U{i}": fake_bars(300, daily=0.0004 * i, seed=i) for i in range(12)}
        out["000001.KS"] = fake_bars(300, end="2026-10-12", daily=0.003, seed=40)  # 오늘(월) 한국 봉까지 있음
        out["ADR"] = fake_bars(60, daily=0.002, seed=41)                           # 상장 60거래일
        out["999999.KS"] = fake_bars(300, end="2026-10-08", daily=0.002, seed=42)  # 원주(10/9 휴장)
        out["QQQ"] = fake_bars(300, daily=0.0005, seed=43)
        return out

    def run_compute(self, rows, bars):
        with mock.patch.object(trend, "holdings", return_value=(rows, {"TIME": "2026-10-08"})),                 mock.patch.object(trend, "fetch", side_effect=lambda syms: {s: bars[s].copy() for s in syms if s in bars}):
            return trend.compute(now=self.NOW)

    def test_outside_and_proxy_names_do_not_move_us_grades_and_rows_match_the_grade_date(self):
        data, charts = self.run_compute(self.rows(), self.bars())
        alone, _ = self.run_compute(self.rows(extra=False), self.bars())
        self.assertEqual(data["as_of"], "2026-10-09")
        by = {s["symbol"]: s for s in data["stocks"]}
        us_alone = {s["symbol"]: (s["grade"], s["hist"]) for s in alone["stocks"]}
        for sym, (g, hist) in us_alone.items():
            self.assertEqual((by[sym]["grade"], by[sym]["hist"]), (g, hist), sym)
        self.assertEqual(by["000001.KS"]["date"], "2026-10-09")  # 한국 오늘 봉이 아니라 등급 날짜 기준
        self.assertEqual(charts["000001.KS"]["dates"][-1], "2026-10-09")
        self.assertEqual(by["ADR"]["proxy_used"], {"symbol": "999999.KS", "own_bars": 60})
        adr = self.bars()["ADR"]["Close"]
        self.assertAlmostEqual(by["ADR"]["close"], adr.iloc[-1])  # 종가는 자체 상장 가격(원주 아님)
        self.assertEqual(by["ADR"]["date"], "2026-10-09")
        self.assertIsNotNone(by["ADR"]["grade"])
        self.assertEqual(charts["ADR"]["proxy"], "999999.KS")
        self.assertTrue(all("raw" in s for s in data["stocks"]))
        self.assertTrue(all(len(c["dates"]) <= trend.CHART_DAYS + 1 for c in charts.values()))
        self.assertTrue(all(c["grades"][0] is not None for c in charts.values()))  # 차트 첫 칸부터 등급이 있다
        self.assertEqual(sum(v for k, v in data["dist"].items() if k != "none") + data["dist"]["none"], len(data["stocks"]))

    def test_too_many_missing_prices_fail_instead_of_caching_a_thin_ranking(self):
        bars = self.bars()
        for i in range(4):
            bars.pop(f"U{i}")
        with self.assertRaises(RuntimeError):
            self.run_compute(self.rows(), bars)


class PayloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.patches = [mock.patch.object(trend, "CACHE_PATH", self.tmp / "trend.json"),
                        mock.patch.object(trend, "CHART_PATH", self.tmp / "trend_chart.json"),
                        mock.patch.object(trend.config, "CACHE", self.tmp)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fake(self, n=1):
        return ({"as_of": "2026-10-09", "n": n, "stocks": []}, {"OKTA": {"symbol": "OKTA", "dates": []}})

    def test_payload_reuses_a_fresh_cache_and_force_recomputes(self):
        with mock.patch.object(trend, "compute", side_effect=[self.fake(1), self.fake(2)]) as calc:
            self.assertEqual(trend.payload()["n"], 1)
            cached = trend.payload()
            self.assertEqual(cached["n"], 1)
            self.assertEqual(cached["check"], trend.CHECK)  # 검증 문구는 캐시가 아니라 지금 코드에서
            self.assertEqual(trend.payload(force=True)["n"], 2)
            self.assertEqual(calc.call_count, 2)

    def test_chart_reads_the_saved_charts_and_explains_missing_ones(self):
        data, charts = self.fake()
        data["stocks"] = [{"symbol": "OKTA"}, {"symbol": "NOPX", "why": "시세 없음"}]
        with mock.patch.object(trend, "compute", return_value=(data, charts)):
            trend.payload()
        self.assertEqual(trend.chart("OKTA")["symbol"], "OKTA")
        with self.assertRaisesRegex(trend.TrendError, "시세를 받지 못해"):
            trend.chart("NOPX")
        with self.assertRaisesRegex(trend.TrendError, "보유 종목이 아닙니다"):
            trend.chart("ZZZZ")
        with self.assertRaisesRegex(trend.TrendError, "비었습니다"):
            trend.chart("")

    def test_a_cache_from_another_spec_is_recomputed(self):
        with mock.patch.object(trend, "compute", side_effect=[self.fake(1), self.fake(2)]) as calc:
            trend.payload()
            with mock.patch.object(trend, "SPEC", "다른 계산식"):
                self.assertEqual(trend.payload()["n"], 2)
            self.assertEqual(calc.call_count, 2)

    def test_check_tables_are_well_formed(self):
        for t in trend.CHECK["tables"]:
            for label, values in t["rows"]:
                self.assertEqual(len(values), len(t["columns"]), label)
        self.assertEqual(json.loads(json.dumps(trend.rules()))["ladder"][0], ["S", 0.9])


if __name__ == "__main__":
    unittest.main()
