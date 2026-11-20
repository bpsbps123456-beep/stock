"""Hold size, monthly timing and ranking fixed while crossing the two balance-sheet thresholds."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RECIPES = HERE.parent / "finviz_recipes_20261120"
sys.path.insert(0, str(RECIPES))
from research import ORIGINAL, TimingEngine, describe, load_inputs, market_gate, spells
from backtest import forward_checks
from baselines import _ledger
from compare import close_terminated, switched_targets
from sensitivity import exposure_match


def execution_checks(engine, ranks, switched):
    first = engine.reviews["monthly1"]
    src = engine.src
    rows = []
    previous = pd.read_csv(RECIPES / "sensitivity.csv").set_index(["recipe", "timing"])
    balance_plan = engine.targets(ranks["balance"], "monthly1")
    for label in ("balance", "current_only"):
        monthly = balance_plan if label == "balance" else engine.targets(ranks[label], "monthly1")
        shifted = {j + 1: monthly[j].copy() for j in first if j + 1 < len(src.days)}
        plans = {"one_open_later": close_terminated(shifted, src.opn, src.tradable)}
        if label == "current_only":
            plans["balance_slot_count"] = exposure_match(src, ranks[label], first, balance_plan)
        comparisons = {"normal": switched[label]}
        for timing, plan in plans.items():
            routed = switched_targets(src.days, src.opn, engine.desk, plan, engine.gate)
            comparisons[timing] = _ledger(src, routed)
        for timing, frame in comparisons.items():
            stats = describe(frame)
            if label == "balance" and timing == "one_open_later":
                check = previous.loc[("balance", "stale2")]
                np.testing.assert_allclose(stats["year_ret"], check.year_ret, atol=1e-10)
            rows.append(dict(quality=label, check=timing, year_ret=stats["year_ret"],
                             h2_ret=stats["h2_ret"], q4_ret=stats["oct_ret"],
                             year_dd=stats["year_dd"], off_exposure=frame.exposure.loc[~engine.gate].mean()))
    pd.DataFrame(rows).to_csv(HERE / "quality_execution.csv", index=False, float_format="%.12g")


def run():
    candidates = pd.read_csv(HERE / "size_inputs.csv", float_precision="round_trip")
    prior = pd.read_csv(RECIPES / "selected_daily.csv", index_col=["key", "date"])
    src = load_inputs()
    gate = market_gate(src)
    engine = TimingEngine(src, gate)
    slots = {symbol: i for i, symbol in enumerate(src.symbols)}
    months = candidates.sort_values("rank_original").groupby("trade_open")
    assert set(months.groups) == {src.days[i] for i in engine.reviews["monthly1"]}
    results = []
    switch_nav = {}
    all_ranks, standalone_paths, switch_paths = {}, {}, {}
    for label, current, debt in [("original", 1., 1.), ("current_only", 1.5, 1.),
                                 ("debt_only", 1., .5), ("balance", 1.5, .5)]:
        ranks = {}
        for day, group in months:
            qualified = group[group.current_ratio.gt(current) & group.debt_equity.ge(0)
                              & group.debt_equity.lt(debt)]
            if label == "original":
                assert qualified.ticker.tolist() == group.ticker.tolist()
            if label == "balance":
                archived = group.dropna(subset=["rank_balance"]).sort_values("rank_balance")
                assert qualified.ticker.tolist() == archived.ticker.tolist()
            ranks[day] = [slots[symbol] for symbol in qualified.ticker]
        all_ranks[label] = [ranks.get(day, []) for day in src.days]
        curves = engine.evaluate(all_ranks[label], "monthly1", detail=True, stress=True)
        switched = curves["switch"]
        standalone_paths[label], switch_paths[label] = curves["always"], switched
        switch_nav[label] = switched.krw10
        if label in ("original", "balance"):
            ref = prior.loc[f"{label}/monthly1"]
            assert switched.index.equals(ref.index)
            np.testing.assert_allclose(switched.krw10, ref.krw10, rtol=1e-10)
            assert switched.holdings.fillna("").equals(ref.holdings.fillna(""))
        metrics = describe(switched)
        standalone = describe(curves["always"])
        results.append(dict(quality=label, min_current_exclusive=current, max_debt_exclusive=debt,
                            candidate_months=sum(map(len, ranks.values())),
                            year_ret=metrics["year_ret"], h1_ret=metrics["h1_ret"],
                            h2_ret=metrics["h2_ret"], q3_ret=metrics["q3_ret"],
                            q4_to_date_ret=metrics["oct_ret"], year_dd=metrics["year_dd"],
                            h2_dd=metrics["h2_dd"], off_exposure=switched.loc[~gate, "exposure"].mean(),
                            gross_year_ret=metrics["year_gross_ret"], stress_year_ret=metrics["year_stress_ret"],
                            always_year_ret=standalone["year_ret"], always_h2_ret=standalone["h2_ret"]))
    # The shared attribution routine takes ORIGINAL as the reference key.
    # Here the comparator is the currently configured balance screen.
    switch_nav[ORIGINAL] = switch_nav.pop("balance")
    periods, scores = spells(src.days, gate, switch_nav)
    periods["key"] = periods.key.replace({ORIGINAL: "balance"})
    periods.rename(columns={"key": "quality", "relative_return": "relative_to_balance"}).to_csv(
        HERE / "quality_episodes.csv", index=False, float_format="%.12g")
    result = pd.DataFrame(results)
    result = result.join(scores.rename(index={ORIGINAL: "balance"}), on="quality")
    result.to_csv(HERE / "quality_results.csv", index=False, float_format="%.12g")
    # Exploratory reuse of the size-selection procedure after inspecting the four fixed curves.
    # Tie priority starts with the live rule.
    forward, train = forward_checks(engine, all_ranks, standalone_paths, switch_paths["balance"],
                                   "balance", ("balance", "original", "current_only", "debt_only"))
    forward.to_csv(HERE / "quality_forward.csv", index=False, float_format="%.12g")
    train.to_csv(HERE / "quality_training.csv", index=False, float_format="%.12g")
    execution_checks(engine, all_ranks, switch_paths)
    return result


if __name__ == "__main__":
    print(run().to_string(index=False))
