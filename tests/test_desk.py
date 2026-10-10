"""Night desk rules without network: journal accounting, the 7-stock selection and caps,
the pause rule, and the US session calendar."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from desk import alerts, config, journal
from desk import strategies as S


class JournalTest(unittest.TestCase):
    def test_average_cost_realized_and_cash(self):
        entries = [
            {"date": "2026-09-21", "type": "deposit", "amount": 1000.0},
            {"date": "2026-09-22", "type": "buy", "ticker": "AAA", "qty": 10.0, "price": 10.0},
            {"date": "2026-09-23", "type": "buy", "ticker": "AAA", "qty": 10.0, "price": 20.0},
            {"date": "2026-09-24", "type": "sell", "ticker": "AAA", "qty": 5.0, "price": 30.0},
        ]
        out = journal.compute(entries, {"AAA": 25.0})
        pos = out["positions"][0]
        self.assertAlmostEqual(pos["avg_cost"], 15.0)
        self.assertAlmostEqual(pos["qty"], 15.0)
        self.assertAlmostEqual(out["realized"], 5 * (30 - 15))
        self.assertAlmostEqual(out["cash"], 1000 - 100 - 200 + 150)
        self.assertAlmostEqual(out["equity"], out["cash"] + 15 * 25)

    def test_oversell_is_rejected(self):
        with self.assertRaises(journal.JournalError):
            journal.compute([{"date": "2026-09-22", "type": "sell", "ticker": "AAA", "qty": 1, "price": 1}], {})

    def test_unfunded_account_does_not_net_negative_cash(self):
        out = journal.compute([{"date": "2026-09-22", "type": "buy", "ticker": "AAA", "qty": 2, "price": 50}],
                              {"AAA": 60})
        self.assertFalse(out["funded"])
        self.assertAlmostEqual(out["equity"], 120)
        self.assertAlmostEqual(out["positions"][0]["weight"], 1.0)

    def test_add_entry_validates_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(config, "TRADES", Path(tmp) / "t.json"):
            journal.add_entry({"date": "2026-09-21", "type": "deposit", "amount": 100})
            with self.assertRaises(journal.JournalError):
                journal.add_entry({"date": "2026-09-22", "type": "buy", "ticker": "", "qty": 1, "price": 1})
            with self.assertRaises(journal.JournalError):
                journal.add_entry({"date": "2026-13-01", "type": "deposit", "amount": 1})
            self.assertEqual(len(journal.load_entries()), 1)

    def test_settings_ignore_retired_strategy_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({"strategy": "c239120", "split": 0.5, "tolerance": 0.03}), encoding="utf-8")
            with mock.patch.object(config, "SETTINGS", path):
                self.assertEqual(journal.load_settings(), {"tolerance": 0.03})

    def test_history_twr_ignores_money_moving_in_and_out(self):
        days = ["2026-09-21", "2026-09-22", "2026-09-23"]
        closes = pd.DataFrame({"AAA": [10.0, 11.0, 11.0]})
        entries = [
            {"date": "2026-09-21", "type": "deposit", "amount": 1000.0},
            {"date": "2026-09-21", "type": "buy", "ticker": "AAA", "qty": 50.0, "price": 10.0},
            {"date": "2026-09-23", "type": "withdraw", "amount": 400.0},
        ]
        out = journal.history(entries, closes, days, np.ones(3))
        self.assertAlmostEqual(out["usd"][2], 650.0)
        self.assertAlmostEqual(out["twr"][1], 1.05)
        self.assertAlmostEqual(out["twr"][2], 1.05)  # the withdrawal is not a loss

    def test_history_twr_unknown_without_a_deposit(self):
        closes = pd.DataFrame({"AAA": [10.0, 12.0]})
        out = journal.history([{"date": "2026-09-21", "type": "buy", "ticker": "AAA", "qty": 1, "price": 10}],
                              closes, ["2026-09-21", "2026-09-22"], np.ones(2))
        self.assertTrue(all(math.isnan(v) for v in out["twr"]))


class RuleTest(unittest.TestCase):
    def assertWeights(self, got, want):
        self.assertEqual(set(got), set(want))
        for key, value in want.items():
            self.assertAlmostEqual(got[key], value, places=9)

    def test_select_keeps_last_weeks_names_inside_top12(self):
        ranked = [{"key": k} for k in "ABCDEFGHIJKLMNO"]
        self.assertEqual(S.select(ranked, set()), list("ABCDEFG"))
        # H (8th) and L (12th) stay, M (13th) is replaced by the best free names.
        self.assertEqual(S.select(ranked, {"H", "L", "M"}), ["H", "L", "A", "B", "C", "D", "E"])

    def test_semiconductor_budget_leaves_cash(self):
        sector = {"S1": "메모리", "S2": "메모리", "S3": "AI 반도체", "S4": "반도체(적자)", "N": "AI 인프라"}
        got = S.project_caps({k: 0.2 for k in sector}, sector)
        # four semis share the 50% budget; the other name stops at the 25% stock cap; 25% stays cash
        self.assertWeights(got, {"S1": 0.125, "S2": 0.125, "S3": 0.125, "S4": 0.125, "N": 0.25})

    def test_semiconductor_budget_is_50_percent_and_774_keeps_60(self):
        # user decision 2026-10-10 (docs/desk_system.md §3); `desk validate` replays #774 with 60%
        self.assertEqual(config.RULE["semis_cap"], 0.50)
        self.assertEqual(config.CODEX_774_RULE, {"lookback": 120, "n": 7, "buffer": 5, "stock_cap": 0.25,
                                                 "sector_cap": 0.60, "semis_cap": 0.60})

    def test_codex_caps_switch_to_774_and_always_restore(self):
        from desk.validate import codex_caps
        before = dict(config.RULE)
        with self.assertRaises(RuntimeError):
            with codex_caps():
                self.assertEqual(config.RULE, config.CODEX_774_RULE)
                raise RuntimeError("replay failed")
        self.assertEqual(config.RULE, before)

    def test_cap_notes_read_the_live_rule(self):
        from desk.state import _cap_note
        self.assertIn("반도체 합계 50% 상한", _cap_note(0.10, 0.20, "메모리"))
        self.assertIn("같은 섹터 60% 상한", _cap_note(0.10, 0.20, "엔터프라이즈 SW"))
        self.assertIn("한 종목 25% 상한", _cap_note(0.25, 0.40, "AI 인프라"))

    def test_stock_cap_redistributes_then_leaves_cash(self):
        sector = {"A": "빅테크", "B": "엔터프라이즈 SW", "C": "우주/방산"}
        got = S.project_caps({"A": 0.5, "B": 0.3, "C": 0.2}, sector)
        self.assertWeights(got, {"A": 0.25, "B": 0.25, "C": 0.25})

    def test_sector_cap_stops_a_whole_sector(self):
        sector = {"A": "엔터프라이즈 SW", "B": "엔터프라이즈 SW", "C": "엔터프라이즈 SW", "D": "빅테크"}
        got = S.project_caps({"A": 0.3, "B": 0.3, "C": 0.3, "D": 0.1}, sector)
        self.assertWeights(got, {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.25})

    def test_unclassified_names_share_the_semiconductor_budget(self):
        got = S.project_caps({k: 0.25 for k in "WXYZ"}, {})
        self.assertWeights(got, {k: 0.125 for k in "WXYZ"})

    def test_sector_lookup_follows_recorded_renames(self):
        self.assertEqual(S.sector_of("ECHO US EQUITY", "ECHO"), "우주/방산")   # formerly SATS
        self.assertEqual(S.sector_of("MOG/A US EQUITY", "MOG-A"), "우주/방산")  # source code spelling
        self.assertEqual(S.sector_of("ZZZZ US EQUITY", "ZZZZ"), S.UNKNOWN)

    def test_subsector_is_display_only_and_falls_back_to_cap_sector(self):
        self.assertEqual(S.subsector_of("SNDK US EQUITY", "SNDK"), "NAND·SSD")
        self.assertEqual(S.subsector_of("GOOG US EQUITY", "GOOG"), "미국 빅테크")  # other listing of GOOGL
        self.assertEqual(S.subsector_of("SKHY US EQUITY", "SKHY"), "DRAM·HBM")   # SK hynix ADR
        self.assertEqual(S.subsector_of("COIN US EQUITY", "COIN"), "핀테크/디지털자산")  # outside AI
        self.assertEqual(S.subsector_of("ZZZZ US EQUITY", "ZZZZ"), S.UNKNOWN)
        # caps keep grouping by the broader watchlist sector
        self.assertEqual(S.sector_of("SNDK US EQUITY", "SNDK"), "메모리")
        self.assertEqual(S.sector_of("MU US EQUITY", "MU"), "메모리")


class DigestTest(unittest.TestCase):
    def test_holding_notes_flag_the_hold_line_exits_and_joint_selling(self):
        from desk.digest import holding_notes

        self.assertEqual(holding_notes(3, "매수 +10%", "매수 +20%", "보유 안 함"), [])
        self.assertEqual(holding_notes(11, "가격효과", "가격효과", "가격효과"), ["유지선(12위) 근접"])
        self.assertEqual(holding_notes(13, "가격효과", "가격효과", "가격효과"), ["유지선(12위) 밖 → 다음 재조정에서 교체"])
        self.assertEqual(holding_notes(None, "편출", "편출", "보유 안 함"),
                         ["TIME 목록에서 빠짐 → 다음 재조정에서 매도", "TIME 편출"])
        self.assertEqual(holding_notes(8, "매도 -17%", "매도 -39%", "매도 -26%"), ["두 펀드 모두 D-20 매도"])

    def test_rebalance_changes_list_entries_and_exits(self):
        from desk.digest import rebalance_changes

        self.assertEqual(rebalance_changes(["A", "B", "C"], {"A", "B", "D"}), (["C"], ["D"]))
        self.assertEqual(rebalance_changes(["A", "B"], {"A", "B"}), ([], []))


def pause_ctx(**changes):
    ctx = {"account_drawdown": None, "model_drawdown": -0.05, "model_vs_etf_20d": 0.02, "model_vs_etf_120d": 0.40}
    return {**ctx, **changes}


class PauseRuleTest(unittest.TestCase):
    def test_quiet_inside_the_2026_range(self):
        self.assertEqual(alerts.my_pause_rule(pause_ctx()), [])
        self.assertEqual(alerts.my_pause_rule(pause_ctx(model_vs_etf_120d=None, model_vs_etf_20d=None)), [])

    def test_short_lag_is_a_keep_going_note(self):
        out = alerts.my_pause_rule(pause_ctx(model_vs_etf_20d=-0.12))
        self.assertEqual([a["level"] for a in out], ["warning"])
        self.assertIn("매매 규칙 유지", out[0]["title"])

    def test_drawdown_takes_the_worse_of_account_and_model(self):
        # 2026 worst drawdown -30.3%: out of range only past -35.3%
        self.assertEqual(alerts.my_pause_rule(pause_ctx(model_drawdown=-0.34)), [])
        out = alerts.my_pause_rule(pause_ctx(account_drawdown=-0.36, model_drawdown=-0.10))
        self.assertEqual([a["level"] for a in out], ["serious"])
        self.assertIn("내 계좌", out[0]["title"])

    def test_six_months_behind_replaces_the_keep_going_note(self):
        out = alerts.my_pause_rule(pause_ctx(model_vs_etf_20d=-0.12, model_vs_etf_120d=-0.01))
        self.assertEqual([a["level"] for a in out], ["serious"])
        self.assertIn("120거래일", out[0]["title"])


class CalendarTest(unittest.TestCase):
    @staticmethod
    def at(text):
        return datetime.fromisoformat(text).replace(tzinfo=config.NY)

    def test_weekend_rolls_to_monday_open_in_kst(self):
        day, open_kst = config.next_us_session(self.at("2026-09-27T02:44"))
        self.assertEqual(day, "2026-09-28")
        self.assertEqual(open_kst.strftime("%H:%M"), "22:30")

    def test_before_open_is_today_after_open_is_next_session(self):
        self.assertEqual(config.next_us_session(self.at("2026-09-29T09:00"))[0], "2026-09-29")
        self.assertEqual(config.next_us_session(self.at("2026-09-29T10:00"))[0], "2026-09-30")

    def test_holiday_is_skipped_and_winter_open_moves(self):
        day, open_kst = config.next_us_session(self.at("2026-11-25T17:00"))
        self.assertEqual(day, "2026-11-27")
        self.assertEqual(open_kst.strftime("%H:%M"), "23:30")

    def test_next_rebalance_is_the_first_session_of_next_week(self):
        self.assertEqual(config.next_rebalance("2026-09-28"), "2026-10-05")
        self.assertEqual(config.next_rebalance("2026-08-31"), "2026-09-08")  # Labor Day Monday


if __name__ == "__main__":
    unittest.main()
