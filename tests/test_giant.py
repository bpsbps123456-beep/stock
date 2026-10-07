"""거인의 어깨 contract: the correction anchor, strength measures, whale signals and the ★
checklist must stay what docs/desk_system.md §9 says, using only data the desk really has."""

from __future__ import annotations

import tempfile
import unittest
import copy
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import ai_ath_gap
import watchlist
from desk import giant


def bars(closes, volumes=None, start="2026-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    closes = np.asarray(closes, float)
    return pd.DataFrame({"Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
                         "Close": closes, "Volume": volumes if volumes is not None else np.ones(len(closes))},
                        index=idx)


class AnchorTest(unittest.TestCase):
    def test_anchor_is_the_deepest_fall_and_survives_new_highs(self):
        path = [100] * 5 + [120, 130, 90, 80, 85] + list(range(90, 141, 10))  # fall -38%, then new high 140
        a = giant.find_anchor(bars(path)["Close"])
        self.assertAlmostEqual(a["peak"], 130)
        self.assertAlmostEqual(a["low"], 80)
        self.assertAlmostEqual(a["crash"], -38.46, places=2)
        self.assertAlmostEqual(a["since_low"], 75.0, places=2)
        self.assertGreater(a["vs_peak"], 0)  # a later new high does not move the anchor
        self.assertEqual(a["age"], 7)  # sessions since the low

    def test_on_the_day_of_the_low_since_low_is_zero_so_the_page_must_say_hold(self):
        # This is why the page shows the hold note: every stock's since-low is 0 on the low day.
        a = giant.find_anchor(bars([100, 120, 130, 90, 80])["Close"])
        self.assertEqual((a["age"], a["since_low"]), (0, 0.0))
        self.assertIn("universe", giant.HOLD_CHECK)

    def test_up_down_volume_ratio(self):
        b = bars([10, 11, 10, 12], volumes=[1, 300, 100, 200])
        self.assertAlmostEqual(giant.ud_ratio(b, days=3), 5.0)  # (300+200)/100

    def test_stock_measures_use_the_last_close_on_or_before_anchor_dates(self):
        b = bars([100] * 60 + [80] * 10 + [120] * 10)
        days = b.index
        anchor = {"peak_date": days[59].date().isoformat(), "low_date": days[65].date().isoformat(),
                  "grade_from": days[30].date().isoformat(), "grade_to": days[70].date().isoformat()}
        m = giant.stock_metrics(b, anchor)
        self.assertAlmostEqual(m["crash"], -20.0)
        self.assertAlmostEqual(m["since_low"], 50.0)
        self.assertAlmostEqual(m["cycle"], 20.0)
        self.assertAlmostEqual(m["r1w"], 0.0)
        self.assertTrue(m["above50"])
        self.assertIn(m["bucket"], ai_ath_gap.BUCKETS)


class WhaleSignalTest(unittest.TestCase):
    ROWS = [  # newest first, as the Naver API returns them
        {"bizdate": "20260803", "foreignerPureBuyQuant": "+1,000", "organPureBuyQuant": "-500",
         "individualPureBuyQuant": "-300", "closePrice": "10,000", "foreignerHoldRatio": "11.50%"},
        {"bizdate": "20260730", "foreignerPureBuyQuant": "+2,000", "organPureBuyQuant": "+100",
         "individualPureBuyQuant": "-2,500", "closePrice": "9,000", "foreignerHoldRatio": "11.00%"},
        {"bizdate": "20260729", "foreignerPureBuyQuant": "-9,999", "organPureBuyQuant": "-9,999",
         "closePrice": "8,000", "foreignerHoldRatio": "10.00%"},
    ]

    def test_recovery_grade_uses_each_stocks_own_crash(self):
        # at its high, falls 40% to its own low, then regains half of the drop
        b = bars([50] * 20 + [100] * 10 + [60] * 10 + [80] * 5)
        days = b.index
        anchor = {"peak_date": days[25].date().isoformat(), "low_date": days[32].date().isoformat(),
                  "grade_from": days[10].date().isoformat(), "grade_to": days[39].date().isoformat()}
        self.assertAlmostEqual(giant.own_recovery(b, anchor), 50.0)
        mild = bars([100] * 30 + [90] * 15)          # fell only 10%: not graded
        self.assertIsNone(giant.own_recovery(mild, anchor))
        far = bars([200] * 5 + [100] * 25 + [60] * 15)  # already -50% from its high before the fall
        self.assertIsNone(giant.own_recovery(far, anchor))
        rows = [{"recovery": v} for v in (95.0, 100.0, 30.0)]
        self.assertEqual(giant.recovery_grade(rows), {"recovery": 95.0, "recovery_n": 3, "grade": "S"})
        self.assertIsNone(giant.recovery_grade(rows[:2])["grade"])  # fewer than 3 graded names
        self.assertEqual([giant.grade_of(v) for v in (90, 60, 40, 20, 19.9)], ["S", "A", "B", "C", "D"])

    def test_kr_flows_start_the_day_after_the_low(self):
        f = giant.kr_flow_from_rows(self.ROWS, "2026-07-29", mcap=1_000_000_000)
        self.assertAlmostEqual(f["foreign_pct"], (1000 * 10000 + 2000 * 9000) / 1e9 * 100)
        self.assertAlmostEqual(f["inst_pct"], (-500 * 10000 + 100 * 9000) / 1e9 * 100)
        self.assertAlmostEqual(f["foreign_hold_pp"], 1.5)
        self.assertTrue(f["complete"])
        self.assertEqual((f["from"], f["to"]), ("20260730", "20260803"))
        self.assertAlmostEqual(f["indiv_pct"], (-300 * 10000 - 2500 * 9000) / 1e9 * 100)
        # whoever is not individual/foreign/institution (buybacks, ADR conversion) took the other side
        self.assertAlmostEqual(f["other_pct"], -(f["foreign_pct"] + f["inst_pct"] + f["indiv_pct"]), places=2)

    def test_breadth_counts_stocks_near_their_own_high(self):
        cal = pd.bdate_range("2026-03-02", periods=80)
        up = bars(np.linspace(100, 200, 80), start="2026-03-02")             # always at a high
        down = bars(np.r_[np.linspace(100, 200, 70), [150] * 10], start="2026-03-02")  # -25% at the end
        young = bars(np.linspace(10, 20, 30), start="2026-05-25")             # <60 bars: ignored
        b = giant.breadth_series({"U": up, "D": down, "Y": young}, ["U", "D", "Y"], cal)
        self.assertTrue(np.isnan(b.iloc[0]))   # nobody has 60 bars yet
        self.assertAlmostEqual(b.iloc[65], 100.0)
        self.assertAlmostEqual(b.iloc[-1], 50.0)

    def test_kr_flows_flag_a_window_that_misses_the_low(self):
        f = giant.kr_flow_from_rows(self.ROWS[:2], "2026-07-29", mcap=1e9)
        self.assertFalse(f["complete"])
        self.assertIsNone(f["foreign_hold_pp"])
        self.assertIsNone(giant.kr_flow_from_rows(self.ROWS, "2026-07-29", mcap=None))

    def test_us_short_interest_reads_shrinking_shorts_as_buying(self):
        # US exchanges publish no buyer types; covered shorts are the one post-low buyer we can see.
        epoch = lambda d: pd.Timestamp(d, tz="UTC").timestamp()
        info = {"sharesShort": 80, "sharesShortPriorMonth": 100, "sharesOutstanding": 1000,
                "shortPercentOfFloat": 0.12, "dateShortInterest": epoch("2026-09-15"),
                "sharesShortPreviousMonthDate": epoch("2026-08-14")}
        s = giant.short_interest(info)
        self.assertEqual((s["cover_pct"], s["chg_pct"], s["float_pct"]), (2.0, -20.0, 12.0))
        self.assertEqual((s["from"], s["to"]), ("2026-08-14", "2026-09-15"))
        self.assertLess(giant.short_interest({**info, "sharesShort": 130})["cover_pct"], 0)  # new shorting = selling
        self.assertIsNone(giant.short_interest({"sharesShort": 80}))  # non-US listings have no FINRA data

    def test_price_impact_is_the_median_move_per_100m_dollars(self):
        closes = 100 * 1.01 ** np.arange(61)  # every day +1%
        b = bars(closes, volumes=1400e8 / closes)  # 1,400억 원 = 1억 달러 at 1,400 KRW/USD each day
        self.assertAlmostEqual(giant.price_impact(b, per_usd=1400.0, days=50), 1.0, places=3)
        self.assertAlmostEqual(giant.price_impact(b, per_usd=2800.0, days=50), 2.0, places=3)  # half the dollars
        thin = bars(closes, volumes=np.r_[np.zeros(40), 1400e8 / closes[40:]])
        self.assertIsNone(giant.price_impact(thin, per_usd=1400.0, days=50))  # <25 traded days

    def test_tokyo_forward_pe_uses_the_consensus_next_fiscal_year(self):
        # Yahoo's forwardEps for Tokyo listings is not the consensus FY EPS (Advantest 237.7 vs 1,202.6 yen).
        est = pd.DataFrame({"avg": [933.2, 1202.6]}, index=["0y", "+1y"])
        info = {"forwardPE": 169.5, "currentPrice": 40300.0, "marketCap": 1, "currency": "JPY"}
        with mock.patch("yfinance.Ticker") as ticker:
            ticker.return_value.info = info
            ticker.return_value.earnings_estimate = est
            self.assertEqual(giant.fundamentals("6857.T")["fwd_pe"], 33.5)
            self.assertEqual(giant.fundamentals("NVDA")["fwd_pe"], 169.5)  # US: Yahoo's own forward P/E
            ticker.return_value.earnings_estimate = None
            self.assertIsNone(giant.fundamentals("6857.T")["fwd_pe"])  # no estimate: blank, not the bad number
        self.assertIsNone(giant.next_fy_pe(100.0, -2.0))

    def test_fx_turns_local_units_into_dollars_and_inverts_only_eur(self):
        with mock.patch("yfinance.Ticker") as ticker:
            ticker.return_value.history.return_value = pd.DataFrame({"Close": [1.25]})
            fx = giant.fx_per_usd(["EUR", "KRW", "USD", None, "XXX"])
        self.assertEqual(fx, {"USD": 1.0, "EUR": 0.8, "KRW": 1.25})

    def test_fund_codes_map_to_universe_symbols(self):
        uni = {s for subs in watchlist.AI_UNIVERSE.values() for syms in subs.values() for s in syms}
        cases = {"INTC US EQUITY": "INTC", "SKHY US EQUITY": "000660.KS", "GOOG US EQUITY": "GOOGL",
                 "009150": "009150.KS", "285A JP EQUITY": "285A.T", "100 HK EQUITY": "0100.HK",
                 "300502 CH EQUITY": "300502.SZ", "0043Y0": None, "NQ FUTURES": None}
        for key, want in cases.items():
            self.assertEqual(giant.universe_symbol(key, uni), want, key)


class ChecklistTest(unittest.TestCase):
    def stock(self, sym, sub, since, gap, ud, flows=None):
        return {"symbol": sym, "group": "G", "sub": sub, "since_low": since, "crash": -30.0, "cycle": 0.0,
                "r1m": 1.0, "r3m": 1.0, "gap_pct": gap, "ud": ud, "above50": True, "rev_growth": 10.0,
                "funds": flows or {}, "kr_flow": None, "mcap": None}

    def test_star_needs_strong_sector_sub_leader_near_high_and_accumulation(self):
        stocks = {s["symbol"]: s for s in [
            self.stock("A", "hot", 80, -2, 1.5), self.stock("B", "hot", 60, -5, 1.1),
            self.stock("C", "hot", 50, -1, 1.3),  # third in its sub
            self.stock("D", "cold", 10, -1, 2.0),  # leader of a weak sub
            self.stock("E", "hot2", 90, -25, 1.4),  # far from its high
            self.stock("F", "hot2", 85, -3, 0.8),  # no accumulation
        ]}
        uni = {"since_low": 55.0}
        layout = {"G": {"hot": ["A", "B", "C"], "cold": ["D"], "hot2": ["E", "F"]}}
        with mock.patch.object(watchlist, "AI_UNIVERSE", layout):
            groups = giant.rank_and_check(stocks, uni)
        self.assertEqual([s for s, m in stocks.items() if m["star"]], ["A", "B"])
        self.assertEqual(stocks["C"]["checks"], [True, False, True, True])
        self.assertEqual(stocks["D"]["checks"][0], False)
        self.assertEqual(groups[0]["subs"][0]["members"], ["A", "B", "C"])
        self.assertEqual(groups[0]["leader"], "E")

    def test_fund_counts_and_kr_weighting(self):
        a = self.stock("A", "s", 1, 0, 1, {"TIME": {"since_low": {"kind": "매수"}, "d20": {"kind": "매도"}},
                                            "KoAct": {"since_low": {"kind": "신규"}}})
        a.update(kr_flow={"foreign_pct": 2.0, "inst_pct": 0.0}, mcap=300)
        b = self.stock("B", "s", 1, 0, 1, {"TIME": {"since_low": {"kind": "편출"}}})
        b.update(kr_flow={"foreign_pct": -2.0, "inst_pct": 1.0}, mcap=100)
        row = giant.summarize([a, b], {"since_low": 0.0})
        self.assertEqual((row["fund_buy"], row["fund_sell"], row["fund_buy_20"], row["fund_sell_20"]), (2, 1, 0, 1))
        self.assertAlmostEqual(row["kr_foreign_pct"], 1.0)  # (2*300 - 2*100) / 400
        self.assertAlmostEqual(row["kr_inst_pct"], 0.25)

    def test_us_short_cover_is_weighted_by_dollar_cap(self):
        a = self.stock("A", "s", 1, 0, 1)
        a.update(short={"cover_pct": 3.0}, mcap_usd=100)
        b = self.stock("B", "s", 1, 0, 1)
        b.update(short={"cover_pct": -1.0}, mcap_usd=300)
        c = self.stock("C", "s", 1, 0, 1)  # a Korean name: no FINRA data, not counted
        row = giant.summarize([a, b, c], {"since_low": 0.0})
        self.assertEqual(row["us_n"], 2)
        self.assertAlmostEqual(row["us_short_cover"], 0.0)  # (3*100 - 1*300) / 400


class CacheTest(unittest.TestCase):
    def test_fresh_price_cache_gains_current_tour_roles_without_fetching_prices(self):
        from desk import tour
        stock = ChecklistTest().stock("NVDA", "GPU", 80, -2, 1.5)
        stock.update(recovery=95.0, checks=[True] * 4, star=True)
        cached = {"stocks": {"NVDA": stock}, "universe": {"since_low": 50.0}, "_epoch": 1e12}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(giant, "CACHE_PATH", Path(tmp) / "giant.json"), \
                mock.patch.object(giant, "compute") as compute:
            import json
            giant.CACHE_PATH.write_text(json.dumps(cached), encoding="utf-8")
            result = giant.payload()
        compute.assert_not_called()
        self.assertEqual(result["tour_as_of"], tour.load_map()["as_of"])
        self.assertEqual(result["stocks"]["NVDA"]["grade"], "S")
        self.assertGreater(len(result["stocks"]["NVDA"]["tour_fields"]), 1)

    def test_payload_reuses_a_fresh_cache_and_force_recomputes(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(giant, "CACHE_PATH", Path(tmp) / "giant.json"), \
                mock.patch.object(giant.config, "CACHE", Path(tmp)), \
                mock.patch.object(giant, "compute", return_value={"x": float("nan")}) as compute:
            first = giant.payload()
            giant.payload()
            self.assertEqual(compute.call_count, 1)
            self.assertIsNone(first["x"])  # NaN never reaches the page as invalid JSON
            giant.payload(force=True)
            self.assertEqual(compute.call_count, 2)


class UniverseContractTest(unittest.TestCase):
    def test_tour_roles_cover_only_the_universe_and_allow_cpu_gpu_overlap(self):
        from desk import tour
        fields = watchlist.ai_tour_sectors(tour.load_map(), tour.universe_symbol)
        placed = {s for r in fields for s in r["symbols"]}
        uni = {s for subs in watchlist.AI_UNIVERSE.values() for syms in subs.values() for s in syms}
        self.assertEqual(placed, uni)
        by_key = {r["key"]: r["symbols"] for r in fields}
        self.assertEqual(len(by_key), len(fields))
        self.assertIn("AMD", by_key["tour:cpu"])
        self.assertIn("AMD", by_key["tour:gpu"])
        self.assertIn("000660.KS", by_key["tour:hbm"])
        self.assertIn("GOOGL", by_key["tour:asic"])
        self.assertIn("ENR.DE", by_key["tour:grid"])
        self.assertIn("SU.PA", by_key["tour:cooling"])
        self.assertNotIn("OPENAI", placed)
        for r in fields:
            self.assertEqual(len(r["symbols"]), len(set(r["symbols"])))

    def test_field_and_stock_grades_differ_and_role_comparison_preserves_star_checks(self):
        data = {"universe": {"since_low": 50.0}, "stocks": {}}
        for sym, rec in [("A", 100.0), ("B", 65.0), ("C", 30.0), ("D", None)]:
            s = ChecklistTest().stock(sym, "primary", 80, -2, 1.5)
            s.update(recovery=rec, checks=[True, False, True, True], star=False, rank_in_sub=3)
            data["stocks"][sym] = s
        before = copy.deepcopy(data["stocks"])
        m = {"as_of": "2026-10-07", "companies": {s: {} for s in data["stocks"]},
             "stops": [{"name": "G", "zones": [
                 {"id": "wide", "name": "전체 역할", "members": [{"t": s} for s in data["stocks"]]},
                 {"id": "small", "name": "작은 역할", "members": [{"t": "A"}, {"t": "B"}]},
             ]}]}
        layout = {"G": {"primary": list(data["stocks"])}}
        with mock.patch.object(watchlist, "AI_UNIVERSE", layout):
            out = giant.add_comparison_views(data, m)
        self.assertEqual(out["tour_sectors"][0]["grade"], "A")
        self.assertIsNone(out["tour_sectors"][1]["grade"])
        self.assertEqual([out["stocks"][s]["grade"] for s in before], ["S", "A", "C", None])
        for sym, s in out["stocks"].items():
            for key in ("checks", "star", "rank_in_sub", "sub", "group"):
                self.assertEqual(s[key], before[sym][key])
        self.assertEqual(out["tour_unplaced"], [])

    def test_every_non_us_symbol_has_a_display_name(self):
        rows = ai_ath_gap.universe_rows()  # also fails on duplicates
        missing = [s for _, _, s in rows if "." in s and s not in ai_ath_gap.NAMES]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
