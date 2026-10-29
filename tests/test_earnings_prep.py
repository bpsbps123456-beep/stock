"""160-minute support zones and the earnings-reaction draft (desk_system.md §14.2)."""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from desk import earnings_prep as ep
from desk import zones
from desk.news import market_calendar

NY = "America/New_York"


def hourly(days: list[str], lows: list[float], highs: list[float] | None = None) -> pd.DataFrame:
    """Regular-session 1-hour bars (09:30…15:30), three 160-minute groups a day; one low/high per group."""
    rows, idx = [], []
    starts = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    group = [0, 0, 0, 1, 1, 2, 2]
    k = 0
    for d in days:
        for hm, g in zip(starts, group):
            lo = lows[3 * k + g] if 3 * k + g < len(lows) else lows[-1]
            hi = (highs[3 * k + g] if highs else lo * 1.01)
            idx.append(pd.Timestamp(f"{d} {hm}", tz=NY))
            rows.append({"Open": lo, "High": hi, "Low": lo, "Close": lo})
        k += 1
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


class ZoneTests(unittest.TestCase):
    def test_three_bars_a_day(self):
        b = zones.bars160(hourly(["2026-10-26"], [10, 11, 12]))
        self.assertEqual(list(b["g"]), [0, 1, 2])
        self.assertEqual(list(b["low"]), [10, 11, 12])

    def test_pivot_needs_five_bars_each_side(self):
        lows = [10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10]
        b = pd.DataFrame({"low": lows, "high": [x + 1 for x in lows]})
        piv = zones.pivots(b)
        self.assertEqual(piv[:, 0].tolist(), [5.0])  # 가운데 저점 하나
        self.assertEqual(zones.zones(piv, 9, 20.0)[2], 0)  # 오른쪽 5봉이 끝나기 전(9번 봉)에는 모른다
        self.assertEqual(zones.zones(piv, 10, 20.0)[2], 1)

    def test_merge_gap_and_order(self):
        # 스윙 (봉, 0=저점, 가격): 100·100.3은 한 구역, 101은 1.2% 안이라 약한 쪽이 빠진다, 95는 따로
        piv = np.array([(10, 0, 100.0), (20, 0, 100.3), (30, 0, 101.0), (40, 0, 95.0), (50, 1, 110.0)])
        sup, res, total = zones.zones(piv, 60, 105.0)
        self.assertEqual(total, 5)
        self.assertEqual([round(z[0], 1) for z in sup], [100.0, 95.0])  # 가까운 지지부터
        self.assertEqual(sup[0][2], 2)
        self.assertEqual([z[0] for z in res], [110.0])

    def test_forming_bar_is_dropped(self):
        h1 = hourly(["2026-10-26", "2026-10-27"], [10, 11, 12, 13, 14, 15])
        at = pd.Timestamp("2026-10-27 13:00", tz=NY)  # 둘째 날 둘째 봉(12:30~14:30)이 아직 만들어지는 중
        self.assertEqual(zones.current(h1, now=at)["asof"], "2026-10-27")
        b = zones.bars160(h1)
        ends = len([1 for d, g in zip(b["day"], b["g"]) if pd.Timestamp(d).tz_localize(NY)
                    + pd.Timedelta(minutes=zones.ENDS[int(g)]) <= at])
        self.assertEqual(ends, 4)


class PrepTests(unittest.TestCase):
    def test_reaction_day(self):
        cal = market_calendar()
        at = lambda s: pd.Timestamp(s, tz=NY)
        self.assertEqual(ep.reaction_day(at("2026-10-29 16:05"), cal), "2026-10-30")
        self.assertEqual(ep.reaction_day(at("2026-10-29 07:00"), cal), "2026-10-29")
        self.assertIsNone(ep.reaction_day(at("2026-10-29 12:00"), cal))
        self.assertEqual(ep.reaction_day(at("2026-10-31 09:00"), cal), "2026-11-02")  # 토요일 발표

    def test_touch_table_interpolates_and_clamps(self):
        self.assertAlmostEqual(ep.p_touch(0.07, 0.07), ep.TOUCH[1.0])
        self.assertAlmostEqual(ep.p_touch(1.125 * 0.07, 0.07), round((ep.TOUCH[1.0] + ep.TOUCH[1.25]) / 2, 3))
        self.assertEqual(ep.p_touch(0.001, 0.07), ep.TOUCH[0.25])  # 0.25s 안쪽은 0.70에 머문다
        self.assertEqual(ep.p_touch(0.5, 0.07), ep.TOUCH[3.0])
        self.assertIsNone(ep.p_touch(-0.01, 0.07))
        values = [ep.TOUCH[k] for k in sorted(ep.TOUCH)]
        self.assertTrue(all(b < a for a, b in zip(values, values[1:])))

    def test_distribution_scales_the_2026_quantiles(self):
        d = ep.distribution(0.1)
        self.assertEqual(d["p_up"], 0.5)
        self.assertAlmostEqual(d["q"]["0.1"], -0.167)
        self.assertAlmostEqual(d["q"]["0.9"], 0.184)

    def test_implied_move_removes_normal_days(self):
        chain = lambda price: pd.DataFrame({"strike": [95.0, 100.0, 105.0], "bid": [0.0] * 3, "ask": [0.0] * 3,
                                            "lastPrice": price})
        fake = SimpleNamespace(options=("2026-10-30", "2026-11-06"),
                               option_chain=lambda e: SimpleNamespace(calls=chain([7.0, 4.0, 2.0]), puts=chain([2.0, 4.0, 7.0])))
        with mock.patch("yfinance.Ticker", return_value=fake):
            im = ep.implied_move("XYZ", "2026-10-30", 100.0, "2026-10-28", sigma=0.03)
        self.assertEqual(im["expiry"], "2026-10-30")
        self.assertAlmostEqual(im["implied_move"], 0.08)
        self.assertEqual(im["normal_days"], 1)  # 10/29 하루가 평소 날
        self.assertAlmostEqual(im["implied_move_event"], round(0.8 * math.sqrt((0.08 / 0.8) ** 2 - 0.03 ** 2), 4))


if __name__ == "__main__":
    unittest.main()
