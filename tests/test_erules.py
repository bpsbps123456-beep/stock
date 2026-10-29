"""실적 뒤 흐름(§15) — 가격 특징·조건 판정(8개 규칙)·시가 진입 결과·경보·반응일·원장 갱신(네트워크 없음)."""
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
    return {"c": c.copy(), "o": c.copy(), "h": c * 1.01, "lo": c * 0.99, "v": np.full(n, 1_000.0), "spy": np.full(n, 400.0)}


def feat(s: dict, k: int, cap: float = 5e9, px: float = 100.0) -> dict | None:
    return erules.features(s["c"], s["o"], s["h"], s["lo"], s["v"], s["spy"], k, cap, px)


class FeatureTests(unittest.TestCase):
    def test_needs_61_sessions_of_history(self) -> None:
        s = series()
        self.assertIsNone(feat(s, 60))
        self.assertIsNotNone(feat(s, 61))

    def test_reaction_day_values(self) -> None:
        s = series()
        k = 100
        s["o"][k], s["c"][k], s["h"][k], s["lo"][k], s["v"][k] = 95.0, 90.0, 100.0, 89.0, 4_000.0
        s["spy"][k] = 404.0          # SPY +1%
        s["c"][k - 61] = 80.0        # 발표 전 60일 +25%
        f = feat(s, k, 9e9, 90.0)
        self.assertAlmostEqual(f["r0_x"], -11.0, places=3)
        self.assertAlmostEqual(f["gap"], -5.0, places=3)
        self.assertAlmostEqual(f["o2c"], (90 / 95 - 1) * 100, places=3)
        self.assertAlmostEqual(f["vol_ratio"], 4.0, places=3)
        self.assertAlmostEqual(f["pre60_x"], 25.0, places=3)
        self.assertAlmostEqual(f["mcap_b"], 10.0, places=3)  # 지금 9B × 100/90
        self.assertEqual(f["below_ma"], 0)
        self.assertIn("d5_x", f)

    def test_d5_missing_until_five_sessions_pass(self) -> None:
        s = series(105)
        self.assertNotIn("d5_x", feat(s, 100))


class JudgeTests(unittest.TestCase):
    def test_k2_tech_volume_spike(self) -> None:
        f = {"vol_ratio": 3.2, "vol60": 4.0}
        self.assertEqual(erules.judge("K2", f, "Technology"), "hit")
        self.assertEqual(erules.judge("K2", f, "Finance"), "no")
        self.assertEqual(erules.judge("K2", {**f, "vol60": 3.0}, "Technology"), "no")

    def test_k1_waits_for_d5_and_eps(self) -> None:
        f = {"vol_ratio": 3.5}
        self.assertEqual(erules.judge("K1", f, "Industrials"), "wait")
        self.assertTrue(erules.eps_needed(f, "Industrials"))
        self.assertEqual(erules.judge("K1", {**f, "eps": 8.0, "d5_x": -4.0}, "Industrials"), "hit")
        self.assertEqual(erules.judge("K1", {**f, "eps": 20.0, "d5_x": -4.0}, "Industrials"), "no")
        self.assertEqual(erules.judge("K1", {**f, "eps": 8.0, "d5_x": -1.0}, "Industrials"), "no")

    def test_eps_not_needed_when_other_conditions_fail(self) -> None:
        f = {"vol_ratio": 1.0, "pre60_x": 0.0}
        self.assertFalse(erules.eps_needed(f, "Finance"))

    def test_k3_capitulation_gap(self) -> None:
        f = {"r0_sig": -3.5, "gap": -4.0, "o2c": -1.5, "below_ma": 1}
        self.assertEqual(erules.judge("K3", f, "Health Care"), "hit")
        self.assertEqual(erules.judge("K3", {**f, "o2c": 0.5}, "Health Care"), "no")
        self.assertEqual(erules.judge("K3", {**f, "below_ma": 0}, "Health Care"), "no")

    def test_avoid_rules(self) -> None:
        self.assertEqual(erules.judge("X1", {"r0_sig": 3.5, "gap": 4.0, "o2c": 1.5}, "Industrials"), "hit")
        self.assertEqual(erules.judge("X2", {"r0_x": 4.0, "mcap_b": 12.0, "d5_x": 3.5}, "Technology"), "hit")
        self.assertEqual(erules.judge("X2", {"r0_x": 4.0, "mcap_b": 40.0, "d5_x": 3.5}, "Technology"), "no")
        self.assertEqual(erules.judge("X3", {"d5_x": 3.0}, "Finance"), "hit")
        self.assertEqual(erules.judge("X3", {}, "Finance"), "wait")

    def test_every_rule_condition_is_defined(self) -> None:
        for code, rule in erules.RULES.items():
            self.assertTrue(set(rule["preds"]) <= set(erules.PRED), code)
            self.assertIn(rule["out"], erules.OUTCOMES, code)


class OutcomeTests(unittest.TestCase):
    def test_open_entry_running_then_closed(self) -> None:
        c = np.linspace(100, 130, 31)
        o = c - 1.0
        sc = so = np.full(31, 400.0)
        running = erules.outcome(c, o, sc, so, 1, 40, 30)
        self.assertTrue(running["entered"])
        self.assertFalse(running["closed"])
        self.assertAlmostEqual(running["x"], (130 / o[1] - 1) * 100, places=2)
        done = erules.outcome(c, o, sc, so, 1, 20, 30)
        self.assertTrue(done["closed"])
        self.assertEqual(done["days"], 19)
        later = erules.outcome(c, o, sc, so, 35, 50, 30)
        self.assertFalse(later["entered"])
        self.assertIsNone(later["x"])

    def test_excess_subtracts_spy_open_to_close(self) -> None:
        c, o = np.array([100.0, 110.0]), np.array([100.0, 100.0])
        sc, so = np.array([400.0, 420.0]), np.array([400.0, 400.0])
        self.assertAlmostEqual(erules.excess_open(c, o, sc, so, 0, 1), 5.0, places=3)


class StatsTests(unittest.TestCase):
    @staticmethod
    def ev(code: str | None, x: float, key: str) -> dict:
        return {"hits": [code] if code else [], "out": {key: {"closed": True, "x": x}}}

    def test_building_until_ten(self) -> None:
        self.assertEqual(erules.rule_stats([self.ev("K4", -1.0, "o_h20")] * 9, "K4")["state"], "building")

    def test_buy_alarm_below_half(self) -> None:
        evs = [self.ev("K4", -1.0, "o_h20")] * 6 + [self.ev("K4", 2.0, "o_h20")] * 4 + [self.ev(None, 1.0, "o_h20")] * 10
        st = erules.rule_stats(evs, "K4")
        self.assertEqual((st["n"], st["win"], st["state"]), (10, 0.4, "alarm"))
        self.assertEqual((st["base_n"], st["base_win"]), (20, 0.7))

    def test_avoid_alarm_above_half(self) -> None:
        evs = [self.ev("X2", 1.0, "o5_h20")] * 6 + [self.ev("X2", -1.0, "o5_h20")] * 4
        self.assertEqual(erules.rule_stats(evs, "X2")["state"], "alarm")
        evs = [self.ev("X2", -1.0, "o5_h20")] * 7 + [self.ev("X2", 1.0, "o5_h20")] * 3
        self.assertEqual(erules.rule_stats(evs, "X2")["state"], "holding")


class ReactionDayTests(unittest.TestCase):
    cal = market_calendar()

    def test_sec_after_close_is_next_session(self) -> None:
        self.assertEqual(erules.reaction_day(self.cal, "2026-10-22 16:21", "2026-10-22", "time-not-supplied"), ("2026-10-23", "sec"))
        self.assertEqual(erules.reaction_day(self.cal, "2026-10-23 16:05", "2026-10-23", ""), ("2026-10-26", "sec"))  # 금 → 월

    def test_sec_before_open_is_same_session(self) -> None:
        self.assertEqual(erules.reaction_day(self.cal, "2026-10-28 06:02", "2026-10-28", ""), ("2026-10-28", "sec"))

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

    def state_with(self, calendar: dict) -> dict:
        st = erules.empty_state()
        st["calendar"] = calendar
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
        rng = np.random.default_rng(1)
        c = np.full(140, 100.0)
        c[:k] *= 1 + rng.normal(0, 0.05, k)  # 하루 변동 3.5% 넘는 고변동
        c[k - 1], c[k:] = 100.0, 110.0
        vol = np.full(140, 1_000.0)
        vol[k] = 5_000.0
        flat = np.full(140, 400.0)
        frame = lambda a, b: pd.DataFrame({"AAA": a, "SPY": b}, index=days)  # noqa: E731
        prices = {"Close": frame(c, flat), "Open": frame(c, flat), "High": frame(c * 1.01, flat + 1),
                  "Low": frame(c * 0.99, flat - 1), "Volume": frame(vol, np.full(140, 1e6))}
        need = erules.score_events(st, prices, self.uni, self.cal)
        ev = st["events"][key]
        self.assertEqual(ev["status"], "ok")
        self.assertIn("K2", ev["hits"])
        self.assertTrue(ev["out"]["o_h20"]["closed"])
        self.assertAlmostEqual(ev["out"]["o_h20"]["x"], 0.0, places=3)  # 반응 다음 날 시가 뒤로 평평
        self.assertIsInstance(need, list)
        with tempfile.TemporaryDirectory() as tmp:
            data = erules.payload(path=Path(tmp) / "x.json", state=st)
        k2 = next(r for r in data["rules"] if r["code"] == "K2")
        self.assertEqual((len(k2["open"]), len(k2["closed"])), (1, 0))  # 다음 실적까지는 90거래일 상한까지 보유 중
        self.assertEqual(data["events_n"], 1)
        self.assertEqual(data["upcoming"], [])

    def test_read_state_rejects_other_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.json"
            p.write_text('{"version": 2}', encoding="utf-8")
            with self.assertRaises(erules.ErulesError):
                erules.read_state(p)
            self.assertEqual(erules.read_state(Path(tmp) / "none.json")["version"], 1)


if __name__ == "__main__":
    unittest.main()
