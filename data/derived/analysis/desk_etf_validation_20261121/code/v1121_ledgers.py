"""Rebuild the §3.1/§3.2 desk-switch ledgers on top of a chosen weekly desk rule.

The weekly desk path comes from desk.strategies.run_model, which reads desk.config.RULE
at call time. Setting RULE to the pre-2026-11-21 literal (120-session return) must
reproduce the frozen 2026-11-20 study files exactly; the current RULE (60-session
residual momentum) then gives the new numbers with the same code. Everything else
(SOXX 100-session gate, monthly FINVIZ rankings, share/cash ledger, fees, FX) is the
prior research code, imported unchanged:
  finviz_trends_20261120 (gates, ScreenGrid) -> finviz_recipes_20261120 (TimingEngine,
  recipes) -> seven_sleeves_20261120 (market, baselines._ledger) -> desk_defense_20261119.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parents[1]
TRENDS = ANALYSIS / "finviz_trends_20261120"
RECIPES = ANALYSIS / "finviz_recipes_20261120"
sys.path.insert(0, str(TRENDS))
from gates import TimingEngine, close_terminated, load_gates, load_inputs, recipes, switched_targets  # noqa: E402
from ranking import ScreenGrid  # noqa: E402
import desk.config as desk_config  # noqa: E402

# The literal RULE in desk/config.py from 2026-10-10 until commit 6d9a1fb1 (2026-11-21).
OLD_RULE = {"lookback": 120, "n": 7, "buffer": 5, "stock_cap": 0.25, "sector_cap": 0.60, "semis_cap": 0.50}
NEW_RULE = {"score": "res", "lookback": 60, "n": 7, "buffer": 5, "stock_cap": 0.25, "sector_cap": 0.60,
            "semis_cap": 0.50}
RULES = {"old120": OLD_RULE, "res60": NEW_RULE}
TREND_KEY = "30-40-175/base/loose/rv/monthly1"
# label -> (alternative used while SOXX is below its 100-session mean)
STRATEGIES = {
    "base": "주간 모델만(전환 없음)",
    "switch": "§3.1 전환: 대피 때 재무 강화 FINVIZ",
    "exit": "대피 때 달러 현금",
    "soxx": "대피 때 SOXX 보유",
    "original": "대피 때 원본 FINVIZ",
    "trend": "§3.2 30·40·175 FINVIZ 후보",
}
EXECUTIONS = ("timely", "lag1", "lag1_all")


def load(rule_name: str):
    """Research panel with the weekly desk built under one RULE; RULE is restored afterwards."""
    if desk_config.RULE != NEW_RULE:
        raise ValueError(f"desk/config.py RULE changed since 2026-11-21: {desk_config.RULE}")
    saved = desk_config.RULE
    desk_config.RULE = dict(RULES[rule_name])
    try:
        src = load_inputs()
    finally:
        desk_config.RULE = saved
    return src


def with_soxx(src):
    """Append SOXX as an extra executable column (split-adjusted Close/Open of the frozen desk)."""
    if "SOXX" in src.loc:
        raise ValueError("SOXX already in the research universe")
    tail = pd.read_csv(ANALYSIS / "seven_sleeves_20261120/tail_prices.csv", index_col="symbol")
    ref_days = src.days[:-1]
    o = np.r_[src.book.opn["SOXX"].reindex(ref_days).to_numpy(float), tail.at["SOXX", "open"]]
    c = np.r_[src.book.cls["SOXX"].reindex(ref_days).to_numpy(float), tail.at["SOXX", "close"]]
    if not (np.isfinite(o).all() and np.isfinite(c).all() and (o > 0).all() and (c > 0).all()):
        raise ValueError("Missing SOXX execution prices")
    n = len(src.days)
    add = lambda arr, value: np.concatenate([arr, np.full((n, 1), value, dtype=arr.dtype)], axis=1)  # noqa: E731
    out = SimpleNamespace(**vars(src))
    out.symbols = [*src.symbols, "SOXX"]
    out.loc = {**src.loc, "SOXX": len(src.symbols)}
    out.opn = np.concatenate([src.opn, o[:, None]], axis=1)
    out.close = np.concatenate([src.close, c[:, None]], axis=1)
    out.tradable = add(src.tradable, True)
    out.settlements = add(src.settlements, False)
    out.membership = add(src.membership, False)
    out.raw_symbols = add(src.raw_symbols, "SOXX")
    out.sectors = add(src.sectors, "Unknown/SOXX")
    out.issuers = add(src.issuers, "Ticker/SOXX")
    out.sector_known = add(src.sector_known, False)
    out.ma_score = add(src.ma_score, np.nan)
    return out


def _delay(plan, reviews, src):
    """Fill each published decision one auction later (finviz_trends_20261120 lag1 convention)."""
    days = len(src.days)
    return close_terminated({j + 1: plan[j] for j in reviews if j + 1 < days}, src.opn, src.tradable)


def build(rule_name: str):
    """Daily ledgers {execution: {strategy: frame}} plus gates, for one weekly RULE."""
    base_src = load(rule_name)
    grid = ScreenGrid(base_src)
    lookup = recipes(base_src)
    src = with_soxx(base_src)
    gates = {0: load_gates(base_src)["sma100"], 1: load_gates(base_src, lag=1)["sma100"]}
    trend_ranks = [line[line >= 0].tolist() for line in grid.ranks(TREND_KEY)]
    assert grid.screens.at[TREND_KEY, "schedule"] == "monthly1"
    soxx = np.zeros(len(src.symbols))
    soxx[src.loc["SOXX"]] = 1.0
    out = {}
    for execution in EXECUTIONS:
        lag = 0 if execution == "timely" else 1
        gate = gates[lag]
        engine = TimingEngine(src, gate)
        desk = engine.desk
        if execution == "lag1_all":
            desk = _delay(desk, sorted(desk), src)
        months = engine.reviews["monthly1"]
        screens = {}
        for name, ranks in [("switch", lookup["balance"]), ("original", lookup["original"]), ("trend", trend_ranks)]:
            plan = engine.targets(ranks, "monthly1")
            screens[name] = _delay(plan, months, src) if lag else plan
        plans = {"base": desk, "exit": switched_targets(src.days, src.opn, desk, {}, gate),
                 "soxx": switched_targets(src.days, src.opn, desk, {j: soxx for j in range(len(src.days))}, gate)}
        for name, screen in screens.items():
            plans[name] = switched_targets(src.days, src.opn, desk, screen, gate)
        frames = {}
        for name in STRATEGIES:
            frame = engine._price(plans[name], detail=True, stress=execution == "timely")
            frame["usd10"] = frame.krw10 / src.fx
            frame["gate_on"] = True if name == "base" else gate
            frames[name] = frame
        out[execution] = frames
    return SimpleNamespace(src=src, gates=gates, ledgers=out, rule=rule_name)


def reproduce_old(built):
    """The old-rule ledgers must equal the frozen 2026-11-20 study files."""
    assert built.rule == "old120"
    led = built.ledgers["timely"]
    archive = pd.read_csv(RECIPES / "selected_daily.csv", index_col=["key", "date"])
    checks = {}
    for mine, theirs in [("base", "base"), ("exit", "exit"), ("switch", "balance/monthly1"),
                         ("original", "original/monthly1")]:
        ref = archive.loc[theirs]
        cols = ["krw", "usd", "krw10", "turn", "exposure"]
        assert list(led[mine].index) == list(ref.index)
        err = float(np.max(np.abs(led[mine][cols].to_numpy() - ref[cols].to_numpy())))
        np.testing.assert_allclose(led[mine][cols], ref[cols], rtol=2e-10, atol=1e-12)
        checks[f"{mine}=finviz_recipes:{theirs}"] = err
    final = pd.read_csv(TRENDS / "finalists.csv").set_index(["key", "test"])
    days = pd.Index(built.src.days)
    for key, mine in [(TREND_KEY, "trend"), ("20-50-200/base/balance/rv/monthly1", "switch")]:
        for test, execution in [("timely_10bp", "timely"), ("lag1_10bp", "lag1")]:
            nav = built.ledgers[execution][mine].krw10.to_numpy()
            ytd = nav[-1] - 1
            h2 = nav[-1] / nav[days.searchsorted("2026-07-01") - 1] - 1
            ref = final.loc[(key, test)]
            np.testing.assert_allclose([ytd, h2], [ref.year_ret, ref.h2_ret], atol=1e-10, rtol=0)
            checks[f"{mine}/{execution}=finviz_trends:finalists"] = float(max(abs(ytd - ref.year_ret), abs(h2 - ref.h2_ret)))
    return checks


if __name__ == "__main__":
    old = build("old120")
    print(reproduce_old(old))
