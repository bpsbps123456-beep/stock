"""Locked forecasts: recording time, tamper check, first close after recording, and scoring."""
from __future__ import annotations

import copy
import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from desk import forecast, news, server

DAYS = ["2026-10-28", "2026-10-29", "2026-10-30", "2026-11-02"]


def example(horizon: int = 3) -> dict:
    universe = {t: {"beta": b, "rev": r, "si": s} for t, b, r, s in
                [("AAA", 1.5, 0.9, 0.02), ("BBB", 1.2, 0.7, 0.03), ("CCC", 1.0, 0.5, 0.05),
                 ("DDD", 0.8, 0.3, 0.10), ("EEE", 0.6, 0.1, 0.20), ("FFF", 1.1, 0.2, 0.04)]}
    pick = lambda t: {"ticker": t, "thesis": "근거", "reasons": ["전망 상향"], "risks": ["금리"]}
    return {
        "id": "test", "title": "시험", "author": "claude", "request": "탑픽", "benchmark": "SOXX",
        "horizon_sessions": horizon, "summary": "요약", "universe": universe, "primary_basket": "picks",
        "baskets": {"picks": {"label": "탑픽", "tickers": ["AAA", "BBB"], "beta": 1.35},
                    "rest": {"label": "나머지", "tickers": ["CCC", "DDD", "EEE", "FFF"], "beta": 0.875}},
        "picks": [pick("AAA"), pick("BBB")],
        "claims": [
            {"id": "m1", "kind": "benchmark_return", "op": ">", "value": 0.0, "probability": 0.6, "text": "SOXX 상승"},
            {"id": "m2", "kind": "benchmark_min_close", "op": ">", "value": 95.0, "probability": 0.8, "text": "저점 유지"},
            {"id": "b1", "kind": "basket_excess", "basket": "picks", "op": ">", "value": 0.0, "probability": 0.6, "text": "초과"},
            {"id": "b2", "kind": "basket_excess", "basket": "picks", "adjust": "beta", "op": ">", "value": 0.0,
             "probability": 0.55, "text": "베타 조정"},
            {"id": "b3", "kind": "basket_excess", "basket": "picks", "versus": "rest", "op": ">", "value": 0.0,
             "probability": 0.55, "text": "나머지보다"},
            {"id": "b4", "kind": "basket_count_beat", "basket": "picks", "op": ">=", "value": 2, "probability": 0.5, "text": "둘 다"},
            {"id": "f1", "kind": "factor_rho", "factor": "rev", "op": ">", "value": 0.0, "probability": 0.6, "text": "전망"},
            {"id": "e1", "kind": "event_excess", "ticker": "AAA", "start": "2026-10-28", "end": "2026-10-29",
             "op": ">", "value": 0.0, "probability": 0.5, "text": "실적 반응"},
            {"id": "eps", "kind": "eps_change", "ticker": "AAA", "base": 2.0, "op": ">", "value": 0.0,
             "probability": 0.7, "text": "EPS 상향"},
            {"id": "n1", "kind": "manual", "probability": 0.8, "text": "설비투자 유지", "how": "뉴스"},
        ],
    }


def locked(fc: dict, at: str = "2026-10-28T14:30:00+00:00") -> dict:
    fc = copy.deepcopy(fc)
    fc["recorded_at"] = at
    fc["sha256"] = forecast.digest(fc)
    return fc


def prices(paths: dict[str, list[float]]) -> dict:
    series = {t: dict(zip(DAYS, v)) for t, v in paths.items()}
    return {"adj": series, "close": copy.deepcopy(series)}


PRICES = prices({"SOXX": [100, 102, 101, 104], "AAA": [10, 11, 12, 13], "BBB": [20, 21, 20, 22],
                 "CCC": [30, 30, 31, 31], "DDD": [40, 39, 40, 41], "EEE": [50, 48, 47, 46], "FFF": [60, 61, 60, 61]})
AFTER = news.stamp("2026-11-02T22:00:00Z")


class ScheduleTests(unittest.TestCase):
    def test_baseline_is_first_close_after_recording(self):
        during = forecast.schedule(locked(example(21)))
        self.assertEqual((during["baseline"], during["target"]), ("2026-10-28", "2026-11-27"))
        after = forecast.schedule(locked(example(21), "2026-10-28T20:00:01+00:00"))
        self.assertEqual(after["baseline"], "2026-10-29")

    def test_nothing_is_scored_before_the_baseline_close(self):
        result = forecast.evaluate(locked(example()), PRICES, now=news.stamp("2026-10-28T19:59:00Z"))
        self.assertEqual(result["status"], "기준 종가 대기")
        self.assertTrue(all(c["state"] == "pending" for c in result["claims"]))


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.fc = locked(example())

    def claims(self, result):
        return {c["id"]: c for c in result["claims"]}

    def test_baskets_alpha_counts_and_final_states(self):
        result = forecast.evaluate(self.fc, PRICES, {"values": {"AAA": 2.2}}, now=AFTER)
        self.assertTrue(result["complete"])
        picks = result["baskets"]["picks"]
        self.assertAlmostEqual(picks["return"], (0.3 + 0.1) / 2)
        self.assertAlmostEqual(picks["excess"], 0.2 - 0.04)
        self.assertAlmostEqual(picks["alpha"], 0.2 - 1.35 * 0.04)
        self.assertEqual(picks["beat"], 2)
        c = self.claims(result)
        self.assertEqual({k: c[k]["state"] for k in ("m1", "m2", "b1", "b2", "b3", "b4", "f1", "e1", "eps")},
                         dict.fromkeys(("m1", "m2", "b1", "b2", "b3", "b4", "f1", "e1", "eps"), "hit"))
        self.assertEqual(c["n1"]["state"], "manual")
        self.assertAlmostEqual(c["eps"]["observed"], 0.1)
        self.assertGreater(c["f1"]["observed"], 0.5)
        self.assertEqual(result["resolved"], 9)
        self.assertAlmostEqual(result["brier"], sum((c[k]["probability"] - 1) ** 2 for k in
                                                    ("m1", "m2", "b1", "b2", "b3", "b4", "f1", "e1", "eps")) / 9)
        self.assertEqual([p["rank"] for p in result["picks"]], [1, 2])

    def test_eps_change_sign_is_right_for_losses(self):
        fc = example()
        fc["claims"].append({"id": "loss", "kind": "eps_change", "ticker": "BBB", "base": -6.0, "op": "<", "value": 0.0,
                             "probability": 0.55, "text": "적자 확대"})
        result = forecast.evaluate(locked(fc), PRICES, {"values": {"AAA": 2.2, "BBB": -6.6}}, now=AFTER)
        loss = {c["id"]: c for c in result["claims"]}["loss"]
        self.assertAlmostEqual(loss["observed"], -0.1, msg="bigger loss is a negative revision")
        self.assertEqual(loss["state"], "hit")

    def test_eps_base_must_be_nonzero(self):
        fc = example()
        fc["claims"][-2]["base"] = 0
        with self.assertRaises(forecast.ForecastError):
            forecast.validate_forecast(fc, locked=False)

    def test_interim_values_are_tracking_not_final(self):
        result = forecast.evaluate(self.fc, PRICES, now=news.stamp("2026-10-30T21:00:00Z"))
        c = self.claims(result)
        self.assertEqual(result["status"], "진행 중")
        self.assertEqual(result["elapsed"], 2)
        self.assertEqual(c["m1"]["state"], "on_track")
        self.assertEqual(c["e1"]["state"], "hit", "event window ended on 10/29")
        self.assertEqual(c["eps"]["state"], "pending", "no estimate fetched yet")

    def test_broken_low_is_a_final_miss_at_once(self):
        broken = prices({"SOXX": [100, 94, 101, 104], "AAA": [10, 11, 12, 13], "BBB": [20, 21, 20, 22],
                         "CCC": [30, 30, 31, 31], "DDD": [40, 39, 40, 41], "EEE": [50, 48, 47, 46], "FFF": [60, 61, 60, 61]})
        result = forecast.evaluate(self.fc, broken, now=news.stamp("2026-10-29T21:00:00Z"))
        self.assertEqual(self.claims(result)["m2"]["state"], "miss")

    def test_missing_stock_price_carries_last_close(self):
        gap = copy.deepcopy(PRICES)
        del gap["adj"]["BBB"]["2026-10-30"]
        result = forecast.evaluate(self.fc, gap, now=news.stamp("2026-10-30T21:00:00Z"))
        self.assertAlmostEqual(result["paths"]["BBB"][-1], 0.05)


class CacheTests(unittest.TestCase):
    def test_intraday_bar_is_not_final(self):
        self.assertFalse(forecast.final_close("2026-10-28", news.stamp("2026-10-28T19:00:00Z")))
        self.assertFalse(forecast.final_close("2026-10-28", news.stamp("2026-10-28T20:10:00Z")), "auction still settling")
        self.assertTrue(forecast.final_close("2026-10-28", news.stamp("2026-10-28T20:20:00Z")))
        self.assertFalse(forecast.final_close("2026-11-26", news.stamp("2026-11-30T00:00:00Z")), "Thanksgiving")

    def test_cache_fetched_before_a_settled_close_is_refetched(self):
        early = {"fetched_epoch": news.stamp("2026-10-28T19:30:00Z").timestamp()}
        self.assertTrue(forecast._missed_close(early, news.stamp("2026-10-28T21:00:00Z")))
        self.assertFalse(forecast._missed_close(early, news.stamp("2026-10-28T20:05:00Z")))
        late = {"fetched_epoch": news.stamp("2026-10-28T21:00:00Z").timestamp()}
        self.assertFalse(forecast._missed_close(late, news.stamp("2026-10-29T13:00:00Z")))


class SplitTests(unittest.TestCase):
    """SOXX splits 3:1 from 2026-11-05; Yahoo then rescales all earlier closes."""

    def setUp(self):
        self.fc = locked(example())

    def run_with(self, soxx, aaa=(10, 11, 12, 13), splits=None):
        p = prices({"SOXX": list(soxx), "AAA": list(aaa), "BBB": [20, 21, 20, 22], "CCC": [30, 30, 31, 31],
                    "DDD": [40, 39, 40, 41], "EEE": [50, 25, 24, 23], "FFF": [60, 61, 60, 61]})
        p["splits"] = splits or {}
        return forecast.evaluate(self.fc, p, now=AFTER)

    def test_recorded_level_is_rescaled_after_a_split(self):
        result = self.run_with([v / 3 for v in (100, 102, 101, 104)], splits={"SOXX": {"2026-10-30": 3.0}})
        m2 = {c["id"]: c for c in result["claims"]}["m2"]
        self.assertAlmostEqual(m2["threshold"], 95 / 3)
        self.assertEqual(m2["state"], "hit", "33.67 > 31.67 after rescaling; 33.67 < 95 would be a false miss")
        self.assertIn("3:1", m2["note"])

    def test_history_not_yet_rescaled_holds_scoring(self):
        result = self.run_with([100, 102, 101 / 3, 104 / 3], splits={"SOXX": {"2026-10-30": 3.0}})
        self.assertEqual(result["status"], "분할 반영 대기")
        self.assertTrue(all(c["state"] == "pending" for c in result["claims"]))

    def test_unadjusted_stock_is_left_out_not_scored_as_a_crash(self):
        result = self.run_with([100, 102, 101, 104], aaa=(10, 11, 6, 6.5), splits={"AAA": {"2026-10-30": 2.0}})
        self.assertEqual(result["price_issues"], ["AAA"])
        self.assertNotIn("AAA", result["returns"])
        self.assertAlmostEqual(result["baskets"]["picks"]["return"], 0.1)

    def test_real_crash_without_split_is_kept(self):
        result = self.run_with([100, 102, 101, 104])
        self.assertEqual(result["price_issues"], [])
        self.assertAlmostEqual(result["returns"]["EEE"], -0.54)


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ledger.json"

    def test_record_stamps_time_hash_and_refuses_rewrite(self):
        with mock.patch.object(forecast, "now_iso", return_value="2026-10-28T14:30:00+00:00"):
            fc = forecast.record(example(), self.path)
        self.assertEqual(fc["recorded_at"], "2026-10-28T14:30:00+00:00")
        self.assertEqual(fc["sha256"], forecast.digest(fc))
        before = self.path.read_bytes()
        changed = example()
        changed["summary"] = "나중에 바꾼 판단"
        with self.assertRaises(forecast.ForecastError):
            forecast.record(changed, self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.path.with_suffix(".lock").exists())

    def test_backdated_or_supplied_stamp_is_ignored(self):
        fc = example()
        fc["recorded_at"] = "2026-01-01T00:00:00+00:00"
        with mock.patch.object(forecast, "now_iso", return_value="2026-10-28T14:30:00+00:00"):
            self.assertEqual(forecast.record(fc, self.path)["recorded_at"], "2026-10-28T14:30:00+00:00")

    def test_edit_after_recording_is_detected(self):
        forecast.record(example(), self.path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["forecasts"][0]["picks"][0]["thesis"] = "사후 수정"
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(forecast.ForecastError):
            forecast.read_ledger(self.path)

    def test_invalid_claim_never_creates_ledger(self):
        fc = example()
        fc["claims"][0]["probability"] = 1.0
        with self.assertRaises(forecast.ForecastError):
            forecast.record(fc, self.path)
        self.assertFalse(self.path.exists())

    def test_picks_must_match_primary_basket(self):
        fc = example()
        fc["picks"].reverse()
        with self.assertRaises(forecast.ForecastError):
            forecast.validate_forecast(fc, locked=False)


class PayloadTests(unittest.TestCase):
    def test_refresh_failure_keeps_cached_prices(self):
        data = {"version": 1, "forecasts": [locked(example())]}
        cached = {**PRICES, "fetched_at": "2026-11-02T22:00:00Z", "fetched_epoch": 0}
        with mock.patch.object(forecast, "_read", return_value=cached), \
                mock.patch.object(forecast, "refresh_prices", side_effect=OSError("offline")), \
                mock.patch.object(forecast, "refresh_estimates", side_effect=OSError("offline")):
            result = forecast.payload(force=True, data=data, now=AFTER)
        self.assertIn("offline", result["warning"])
        self.assertEqual(result["forecasts"][0]["result"]["status"], "완료")

    def test_recorded_ledger_is_valid_and_locked(self):
        data = forecast.read_ledger()
        for fc in data["forecasts"]:
            if forecast.is_earnings(fc):
                for ev in fc["events"]:
                    self.assertLess(news.stamp(fc["recorded_at"]), forecast.event_times(ev)["cutoff"])
                continue
            sch = forecast.schedule(fc)
            self.assertEqual(len(sch["days"]), fc["horizon_sessions"] + 1)
            self.assertLess(news.stamp(fc["recorded_at"]), sch["closes"][sch["baseline"]])


def earnings_example() -> dict:
    dist = lambda p, lo, mid, hi: {"p_up": p, "q": {"0.1": lo, "0.25": lo / 2, "0.5": mid, "0.75": hi / 2, "0.9": hi}}
    event = lambda t, timing, at, levels=(): {
        "ticker": t, "timing": timing, "announce_at": at, "base_date": "2026-10-28", "reaction_date": "2026-10-29",
        "inputs": {"scale": 0.08}, "mechanical": dist(0.5, -0.1, 0.0, 0.1), "forecast": dist(0.6, -0.08, 0.01, 0.12),
        "levels": list(levels), "thesis": "근거", "reasons": ["옵션 내재 변동"], "risks": ["가이던스"]}
    return {"id": "earn", "type": "earnings", "title": "실적", "author": "claude", "request": "실적 반응", "benchmark": "QQQ",
            "summary": "요약", "model": {"text": "기계 기준 + 판단"},
            "events": [event("AAA", "amc", "2026-10-28T20:05:00Z",
                             [{"name": "Z1", "price": 9.5, "p_touch": 0.4}, {"name": "Z3", "price": 8.5, "p_touch": 0.1}]),
                       event("BBB", "bmo", "2026-10-29T11:00:00Z")]}


EARN_PRICES = {"close": {"AAA": {"2026-10-28": 10.0, "2026-10-29": 11.0}, "BBB": {"2026-10-28": 20.0, "2026-10-29": 19.0},
                         "QQQ": {"2026-10-28": 100.0, "2026-10-29": 101.0}},
               "low": {"AAA": {"2026-10-29": 9.4}, "BBB": {"2026-10-29": 18.0}},
               "ext": {"AAA|2026-10-29": {"post": {"low": 8.0, "floor": 8.9}, "pre": {"low": 9.45, "floor": 9.45},
                                          "reg": {"low": 9.4, "floor": 9.4 / 0.999}}}}
EARN_AFTER = news.stamp("2026-10-29T20:30:00Z")


class EarningsTests(unittest.TestCase):
    def test_record_must_precede_release_and_base_close(self):
        forecast.validate_forecast(locked(earnings_example()))
        for at in ("2026-10-28T20:00:00+00:00", "2026-10-28T21:00:00+00:00"):  # 기준일 마감 뒤: amc·bmo 모두 거부
            with self.assertRaises(forecast.ForecastError):
                forecast.validate_forecast(locked(earnings_example(), at))

    def test_session_rules(self):
        cases = [("reaction_date", "2026-10-30"), ("announce_at", "2026-10-28T18:00:00Z"), ("timing", "dmh")]
        for key, value in cases:
            fc = earnings_example()
            fc["events"][0][key] = value
            with self.assertRaises(forecast.ForecastError, msg=key):
                forecast.validate_forecast(fc, locked=False)
        fc = earnings_example()
        fc["events"][1]["announce_at"] = "2026-10-29T14:00:00Z"  # 장중(10:00 ET) 발표는 bmo가 아니다
        with self.assertRaises(forecast.ForecastError):
            forecast.validate_forecast(fc, locked=False)

    def test_distribution_rules(self):
        for bad in ({"p_up": 1.0}, {"q": {"0.1": 0.1, "0.25": 0.0, "0.5": 0.0, "0.75": 0.1, "0.9": 0.2}}, {"q": {"0.5": 0.0}}):
            fc = earnings_example()
            fc["events"][0]["forecast"].update(bad)
            with self.assertRaises(forecast.ForecastError):
                forecast.validate_forecast(fc, locked=False)

    def test_waiting_until_reaction_close_settles(self):
        result = forecast.evaluate_earnings(locked(earnings_example()), EARN_PRICES, now=news.stamp("2026-10-29T19:59:00Z"))
        self.assertEqual(result["status"], "발표 대기")
        self.assertTrue(all(x["state"] == "waiting" for x in result["events"]))

    def test_scoring_against_mechanical_and_levels(self):
        result = forecast.evaluate_earnings(locked(earnings_example()), EARN_PRICES, now=EARN_AFTER)
        self.assertEqual(result["status"], "완료")
        a, b = result["events"]
        self.assertAlmostEqual(a["return"], 0.10)
        self.assertAlmostEqual(a["bench_return"], 0.01)
        self.assertAlmostEqual(a["forecast"]["brier"], 0.16)
        self.assertTrue(a["forecast"]["dir_hit"])
        self.assertIsNone(a["mechanical"]["dir_hit"])  # 50%는 방향을 고르지 않은 것
        self.assertTrue(a["forecast"]["in80"] and not a["forecast"]["in50"])
        self.assertFalse(a["mechanical"]["in80"])
        q = earnings_example()["events"][0]["forecast"]["q"]
        self.assertAlmostEqual(a["forecast"]["pinball"], forecast.pinball(q, 0.10))
        z1, z3 = a["levels"]
        self.assertTrue(z1["touched"] and z1["touched_post"])  # 정규장 저가 9.4가 9.5보다 0.1% 넘게 아래
        self.assertAlmostEqual(z1["to_close"], 11 / 9.5 - 1)
        self.assertFalse(z3["touched"])  # 프리장·정규장은 못 닿음
        self.assertFalse(z3["touched_post"])  # 애프터 원 저가 8.0은 한 번 찍힌 값, 확인 저가 8.9는 8.5 위
        self.assertAlmostEqual(b["return"], -0.05)
        self.assertFalse(b["forecast"]["dir_hit"])
        score = result["score"]
        self.assertEqual(score["n"], 2)
        self.assertEqual((score["forecast"]["calls"], score["forecast"]["hits"]), (2, 1))
        self.assertEqual(score["mechanical"]["calls"], 0)
        self.assertEqual((score["levels"]["n"], score["levels"]["touched"]), (2, 1))
        self.assertAlmostEqual(score["size_ratio"], (0.10 / 0.08 + 0.05 / 0.08) / 2)

    def test_regular_low_needs_to_go_through_the_limit(self):
        prices = copy.deepcopy(EARN_PRICES)
        prices["ext"] = {}
        prices["low"]["AAA"]["2026-10-29"] = 9.5  # 지정가와 같은 저가는 체결로 보지 않는다
        z1 = forecast.evaluate_earnings(locked(earnings_example()), prices, now=EARN_AFTER)["events"][0]["levels"][0]
        self.assertFalse(z1["touched"])

    def test_missing_price_after_close_is_not_scored(self):
        prices = copy.deepcopy(EARN_PRICES)
        del prices["close"]["BBB"]["2026-10-29"]
        result = forecast.evaluate_earnings(locked(earnings_example()), prices, now=EARN_AFTER)
        self.assertEqual([x["state"] for x in result["events"]], ["scored", "no_price"])
        self.assertEqual(result["status"], "진행 중")

    def test_record_refused_after_cutoff_and_payload_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            with mock.patch.object(forecast, "now_iso", return_value="2026-10-28T20:00:00+00:00"):
                with self.assertRaises(forecast.ForecastError):
                    forecast.record(earnings_example(), path)
            self.assertFalse(path.exists())
            with mock.patch.object(forecast, "now_iso", return_value="2026-10-28T14:30:00+00:00"):
                forecast.record(earnings_example(), path)
                forecast.record(example(), path)
            data = forecast.read_ledger(path)
        self.assertEqual(forecast.tickers_of(data),  # 실적 기록이 있으면 시장 몫 후보 지수도 함께 받는다
                         sorted({"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "QQQ", "SOXX", *forecast.MARKET_PROXIES}))
        out = forecast.payload(data=data, prices={**EARN_PRICES, "adj": PRICES["adj"]}, estimates={}, now=EARN_AFTER)
        self.assertEqual(out["earnings"]["score"]["n"], 2)
        self.assertEqual({x["record"] for x in out["earnings"]["events"]}, {"earn"})


def market_frame(n: int = 160, shock: tuple[int, float, float] | None = None):
    """Prices where AAA moves 1.5× SOXX and has nothing to do with RSP; shock = (day, AAA move, SOXX move)."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2026-03-02", periods=n).strftime("%Y-%m-%d")
    rsp, soxx = rng.normal(0, 0.01, n), rng.normal(0, 0.02, n)
    aaa = 1.5 * soxx + rng.normal(0, 0.002, n)
    if shock:
        i, a, s = shock
        aaa[i], soxx[i] = a, s
    path = lambda r: 100 * np.cumprod(1 + r)
    return pd.DataFrame({"AAA": path(aaa), "RSP": path(rsp), "SOXX": path(soxx)}, index=days), list(days)


class MarketSplitTests(unittest.TestCase):
    def test_fit_picks_the_closest_proxy_and_skips_past_report_days(self):
        frame, days = market_frame(shock=(100, -0.30, 0.05))
        base = days[140]
        fit = forecast.market_fit(frame, "AAA", base, {days[100]})
        self.assertEqual(fit["proxy"], "SOXX")
        self.assertAlmostEqual(fit["beta"], 1.5, delta=0.05)
        self.assertEqual((fit["n"], fit["to"]), (forecast.BETA_SESSIONS, base))
        raw = forecast.market_fit(frame, "AAA", base, set())  # 지난 실적 갭을 넣으면 베타가 흔들린다
        self.assertGreater(abs(raw["beta"] - 1.5), 0.2)

    def test_fit_uses_closes_up_to_the_base_day_only(self):
        frame, days = market_frame()
        later = frame.copy()
        later.loc[days[150]:, "AAA"] *= 0.5  # 기준일 뒤 급락은 고르는 데 쓰이지 않는다
        self.assertEqual(forecast.market_fit(frame, "AAA", days[140], set()),
                         forecast.market_fit(later, "AAA", days[140], set()))

    def test_short_history_and_missing_prices(self):
        frame, days = market_frame(n=40)
        self.assertIsNone(forecast.market_fit(frame, "AAA", days[-1], set())["proxy"])
        self.assertIsNone(forecast.market_fit(frame, "ZZZ", days[-1], set()))  # 가격이 없으면 다음 갱신에 다시

    def test_scored_report_is_split_into_market_and_own(self):
        market = {"events": {"AAA|2026-10-29": {"proxy": "QQQ", "beta": 2.0, "corr": 0.8},
                             "BBB|2026-10-29": {"proxy": "SOXX", "beta": 1.0, "corr": 0.5}}}
        result = forecast.evaluate_earnings(locked(earnings_example()), EARN_PRICES, now=EARN_AFTER, market=market)
        a, b = result["events"]
        self.assertAlmostEqual(a["market"]["part"], 2.0 * 0.01)
        self.assertAlmostEqual(a["market"]["own"], 0.10 - 0.02)
        self.assertIsNone(b["market"])  # 비교 지수 가격이 없으면 나누지 않는다
        own = result["score"]["own"]
        self.assertEqual((own["n"], own["calls"], own["hits"]), (1, 1, 1))

    def test_proxy_split_not_rescaled_leaves_the_split_out(self):
        prices = copy.deepcopy(EARN_PRICES)
        prices["close"]["QQQ"] = {"2026-10-28": 100.0, "2026-10-29": 34.0}
        prices["splits"] = {"QQQ": {"2026-10-29": 3.0}}
        market = {"events": {"AAA|2026-10-29": {"proxy": "QQQ", "beta": 1.0, "corr": 0.8}}}
        a = forecast.evaluate_earnings(locked(earnings_example()), prices, now=EARN_AFTER, market=market)["events"][0]
        self.assertIsNone(a["market"])

    def test_pending_only_after_the_base_close(self):
        data = {"version": 1, "forecasts": [locked(earnings_example())]}
        self.assertEqual(forecast.pending_market(data, None, news.stamp("2026-10-28T19:00:00Z")), [])
        settled = news.stamp("2026-10-28T20:30:00Z")
        self.assertEqual([e["ticker"] for e in forecast.pending_market(data, None, settled)], ["AAA", "BBB"])
        done = {"events": {"AAA|2026-10-29": {"proxy": None}, "BBB|2026-10-29": {"proxy": "QQQ"}}}
        self.assertEqual(forecast.pending_market(data, done, settled), [])


class ForecastHttpTests(unittest.TestCase):
    def test_endpoint_serves_payload(self):
        httpd = server.DeskServer(("127.0.0.1", 0), server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            with mock.patch.object(forecast, "payload", return_value={"forecasts": []}) as called:
                with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_port}/api/forecast?refresh=1") as response:
                    self.assertEqual(json.load(response), {"forecasts": []})
            called.assert_called_once_with(force=True)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(3)


if __name__ == "__main__":
    unittest.main()
