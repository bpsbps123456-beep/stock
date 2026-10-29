"""실적 뒤 흐름(§15) — 가격 특징·규칙 판정·결과 닫기·경보·반응일·원장 갱신(네트워크 없음)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from desk import erules
from desk.news import market_calendar


def series(n: int = 140, start: float = 100.0) -> dict[str, np.ndarray]:
    c = np.full(n, start)
    return {"c": c.copy(), "h": c * 1.01, "lo": c * 0.99, "v": np.full(n, 1_000.0), "spy": np.full(n, 400.0)}


class FeatureTests(unittest.TestCase):
    def test_needs_61_sessions_of_history(self) -> None:
        s = series()
        self.assertIsNone(erules.features(s["c"], s["h"], s["lo"], s["v"], s["spy"], 60, 5e9, 100.0))
        self.assertIsNotNone(erules.features(s["c"], s["h"], s["lo"], s["v"], s["spy"], 61, 5e9, 100.0))

    def test_reaction_day_values(self) -> None:
        s = series()
        k = 100
        s["c"][k], s["h"][k], s["lo"][k], s["v"][k] = 90.0, 100.0, 89.0, 4_000.0
        s["spy"][k] = 404.0  # SPY +1%
        s["h"][40] = 150.0   # 52주 고점(발표 전날 종가 100 → −33.3%)
        f = erules.features(s["c"], s["h"], s["lo"], s["v"], s["spy"], k, 9e9, 90.0)
        self.assertAlmostEqual(f["r0_x"], -11.0, places=3)
        self.assertAlmostEqual(f["vol_ratio"], 4.0, places=3)
        self.assertAlmostEqual(f["dist_hi"], -33.3333, places=3)
        self.assertAlmostEqual(f["clv"], 1 / 11, places=3)
        self.assertAlmostEqual(f["mcap_b"], 10.0, places=3)  # 지금 9B × 100/90
        self.assertIn("d5_x", f)

    def test_d5_missing_until_five_sessions_pass(self) -> None:
        s = series(105)
        f = erules.features(s["c"], s["h"], s["lo"], s["v"], s["spy"], 100, 5e9, 100.0)
        self.assertNotIn("d5_x", f)


class MatchTests(unittest.TestCase):
    base = {"vol60": 4.0, "dist_hi": -35.0, "clv": 0.2, "vol_ratio": 1.0, "mcap_b": 50.0, "r0_x": -9.0}

    def test_b1_needs_all_three(self) -> None:
        self.assertEqual(erules.matches(self.base, "Industrials")[0], ["B1"])
        self.assertEqual(erules.matches({**self.base, "clv": 0.3}, "Industrials")[0], [])
        self.assertEqual(erules.matches({**self.base, "vol60": 3.0}, "Industrials")[0], [])

    def test_b4_technology_with_volume_spike(self) -> None:
        f = {**self.base, "dist_hi": -5.0, "vol_ratio": 3.2}
        self.assertEqual(erules.matches(f, "Technology")[0], ["B4"])
        self.assertEqual(erules.matches(f, "Finance")[0], [])

    def test_a2_waits_for_five_days(self) -> None:
        f = {"vol60": 2.0, "mcap_b": 12.0, "r0_x": 4.0}
        self.assertEqual(erules.matches(f, "Technology"), ([], True))
        self.assertEqual(erules.matches({**f, "d5_x": 3.5}, "Technology"), (["A2"], False))
        self.assertEqual(erules.matches({**f, "d5_x": 1.0}, "Technology"), ([], False))
        self.assertEqual(erules.matches({**f, "mcap_b": 40.0, "d5_x": 5.0}, "Technology"), ([], False))


class OutcomeTests(unittest.TestCase):
    def test_running_then_closed(self) -> None:
        c = np.linspace(100, 130, 31)
        spy = np.full(31, 400.0)
        running = erules.outcome(c, spy, 0, 40, 30)
        self.assertFalse(running["closed"])
        self.assertAlmostEqual(running["x"], 30.0, places=2)
        done = erules.outcome(c, spy, 0, 20, 30)
        self.assertTrue(done["closed"])
        self.assertEqual(done["days"], 20)
        self.assertIsNone(erules.outcome(c, spy, 35, 50, 30)["x"])

    def test_excess_subtracts_spy(self) -> None:
        c = np.array([100.0, 110.0])
        spy = np.array([400.0, 420.0])
        self.assertAlmostEqual(erules.excess(c, spy, 0, 1), 5.0, places=3)


class StatsTests(unittest.TestCase):
    @staticmethod
    def ev(code: str | None, x: float, key: str = "h20") -> dict:
        return {"matches": [code] if code else [], "out": {key: {"closed": True, "x": x}}}

    def test_building_until_ten(self) -> None:
        st = erules.rule_stats([self.ev("B1", -1.0)] * 9, "B1")
        self.assertEqual(st["state"], "building")

    def test_buy_alarm_below_half(self) -> None:
        evs = [self.ev("B1", -1.0)] * 6 + [self.ev("B1", 2.0)] * 4 + [self.ev(None, 1.0)] * 10
        st = erules.rule_stats(evs, "B1")
        self.assertEqual((st["n"], st["win"], st["state"]), (10, 0.4, "alarm"))
        self.assertEqual((st["base_n"], st["base_win"]), (20, 0.7))

    def test_avoid_alarm_above_half(self) -> None:
        evs = [self.ev("A2", 1.0, "e5h20")] * 6 + [self.ev("A2", -1.0, "e5h20")] * 4
        self.assertEqual(erules.rule_stats(evs, "A2")["state"], "alarm")
        evs = [self.ev("A2", -1.0, "e5h20")] * 7 + [self.ev("A2", 1.0, "e5h20")] * 3
        self.assertEqual(erules.rule_stats(evs, "A2")["state"], "holding")


class ReactionDayTests(unittest.TestCase):
    cal = market_calendar()

    def test_sec_after_close_is_next_session(self) -> None:
        self.assertEqual(erules.reaction_day(self.cal, "2026-10-22 16:21", "2026-10-22", "time-not-supplied"), ("2026-10-23", "sec"))
        self.assertEqual(erules.reaction_day(self.cal, "2026-10-23 16:05", "2026-10-23", ""), ("2026-10-26", "sec"))  # 금 → 월

    def test_sec_before_open_is_same_session(self) -> None:
        self.assertEqual(erules.reaction_day(self.cal, "2026-10-28 06:02", "2026-10-28", ""), ("2026-10-28", "sec"))
        self.assertEqual(erules.reaction_day(self.cal, "2026-10-29 08:00", "2026-10-28", "time-after-hours"), ("2026-10-29", "sec"))

    def test_fallbacks(self) -> None:
        self.assertEqual(erules.reaction_day(self.cal, None, "2026-10-28", "time-after-hours"), ("2026-10-29", "nasdaq"))
        self.assertEqual(erules.reaction_day(self.cal, None, "2026-10-28", "time-pre-market"), ("2026-10-28", "nasdaq"))
        self.assertEqual(erules.reaction_day(self.cal, None, "2026-10-22", "time-not-supplied", "2026-10-22 16:00"), ("2026-10-23", "yf"))
        self.assertEqual(erules.reaction_day(self.cal, None, "2026-10-22", "time-not-supplied", "2026-10-22 12:00"), (None, "unknown"))
        self.assertEqual(erules.reaction_day(self.cal, None, "2026-10-22", "time-not-supplied"), (None, "unknown"))


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cal = market_calendar()
        self.uni = {"AAA": {"name": "Aaa Corp", "cap": 9e9, "sector": "Technology"},
                    "BBB": {"name": "Bbb Inc", "cap": 50e9, "sector": "Industrials"}}

    def state_with(self, calendar: dict, accepts: dict | None = None) -> dict:
        st = erules.empty_state()
        st["calendar"] = calendar
        st["sec"] = {s: {"fetched": "2099-01-01", "accepts": a} for s, a in (accepts or {}).items()}
        return st

    def test_dedup_within_31_sessions_keeps_first(self) -> None:
        st = self.state_with({"2026-10-06": [{"symbol": "AAA", "time": "time-pre-market"}],
                              "2026-10-20": [{"symbol": "AAA", "time": "time-pre-market"}]})
        erules.build_events(st, self.uni, self.cal)
        self.assertEqual(list(st["events"]), ["AAA|2026-10-06"])

    def test_unknown_timing_recorded(self) -> None:
        st = self.state_with({"2026-10-06": [{"symbol": "BBB", "time": "time-not-supplied"}]})
        erules.build_events(st, self.uni, self.cal)
        self.assertEqual(st["events"], {})
        self.assertEqual(st["unknown_timing"], ["BBB 2026-10-06"])

    def test_score_and_payload(self) -> None:
        days = pd.bdate_range("2026-05-01", periods=140)
        st = self.state_with({days[110].strftime("%Y-%m-%d"): [{"symbol": "AAA", "time": "time-pre-market"}]})
        with mock.patch.object(erules, "LIVE_START", "2026-01-01"):
            erules.build_events(st, self.uni, self.cal)
        key = next(iter(st["events"]))
        k = list(days).index(pd.Timestamp(st["events"][key]["reaction"]))
        c = np.full(140, 100.0)
        c[k:] = 110.0
        vol = np.full(140, 1_000.0)
        vol[k] = 5_000.0
        rng = np.random.default_rng(1)
        c[:k] *= 1 + rng.normal(0, 0.05, k)  # 하루 변동 3.5% 넘는 고변동
        c[k - 1] = 100.0
        prices = {"Close": pd.DataFrame({"AAA": c, "SPY": np.full(140, 400.0)}, index=days),
                  "High": pd.DataFrame({"AAA": c * 1.01, "SPY": np.full(140, 401.0)}, index=days),
                  "Low": pd.DataFrame({"AAA": c * 0.99, "SPY": np.full(140, 399.0)}, index=days),
                  "Volume": pd.DataFrame({"AAA": vol, "SPY": np.full(140, 1e6)}, index=days)}
        erules.score_events(st, prices, self.uni)
        ev = st["events"][key]
        self.assertEqual(ev["status"], "ok")
        self.assertIn("B4", ev["matches"])
        self.assertTrue(ev["out"]["h20"]["closed"])
        self.assertAlmostEqual(ev["out"]["h20"]["x"], 0.0, places=3)  # 반응일 종가 뒤로 평평
        with tempfile.TemporaryDirectory() as tmp:
            data = erules.payload(path=Path(tmp) / "x.json", state=st)
        b4 = next(r for r in data["rules"] if r["code"] == "B4")
        self.assertEqual(len(b4["closed"]), 0)  # nx(다음 실적까지)는 다음 발표가 없어 90거래일 상한까지 진행 중
        self.assertEqual(len(b4["open"]), 1)
        self.assertEqual(data["events_n"], 1)

    def test_read_state_rejects_other_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.json"
            p.write_text('{"version": 2}', encoding="utf-8")
            with self.assertRaises(erules.ErulesError):
                erules.read_state(p)
            self.assertEqual(erules.read_state(Path(tmp) / "none.json")["version"], 1)


if __name__ == "__main__":
    unittest.main()
