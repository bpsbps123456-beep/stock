"""Monthly FINVIZ capitalization filters on the frozen 2026 SOXX/desk ledger.

Each size screen uses the same seven-slot allocation, previous-close ranking,
portfolio engine, settlement handling and FX series as the published controls.
The calendar-window choices are made using only standalone FINVIZ training returns.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "finviz_recipes_20261120"
sys.path.insert(0, str(PRIOR))
from research import ORIGINAL, describe, spells
from market import load_inputs
from baselines import market_gate, _ledger
from compare import switched_targets
from recipes import recipes
from sensitivity import exposure_match
from timing import TimingEngine


SIZES = {
    "published": None,
    "over2b": (2e9, np.inf),
    "mid2to10b": (2e9, 10e9),
    "over10b": (10e9, np.inf),
    "over200b": (200e9, np.inf),
}
SCREENS = ("original", "balance")
WINDOWS = {
    "h1": ("2026-01-01", "2026-06-30"), "h2": ("2026-07-01", "2026-11-18"),
    "q1": ("2026-01-01", "2026-03-31"), "q2": ("2026-04-01", "2026-06-30"),
    "q3": ("2026-07-01", "2026-09-30"), "q4_partial": ("2026-10-01", "2026-11-18"),
}


def rankings(src, engine, published, cap_rows):
    """Check that cap inputs cover every name in each published monthly ranking."""
    reviews = engine.reviews["monthly1"]
    if cap_rows.duplicated(["trade_open", "ticker"]).any():
        raise ValueError("Duplicate capitalization observation")
    assert set(cap_rows.trade_open) == {src.days[j] for j in reviews}
    if not (np.isfinite(cap_rows.cap_usd) & cap_rows.cap_usd.gt(0)).all():
        raise ValueError("Missing capitalization in a published eligible name")
    result = {(name, size): [[] for _ in src.days] for name in SCREENS for size in SIZES}
    for j in reviews:
        day = cap_rows[cap_rows.trade_open.eq(src.days[j])].set_index("ticker")
        assert day.signal_date.eq(src.signals[j]).all() and src.signals[j] < src.days[j]
        assert set(day.index) == {src.symbols[i] for name in SCREENS for i in published[name][j]}
        for name in SCREENS:
            rank = published[name][j]
            expected = [src.symbols[i] for i in rank]
            recorded = day["rank_" + name].dropna().sort_values()
            assert recorded.index.tolist() == expected, (name, src.days[j])
            np.testing.assert_array_equal(recorded.values, np.arange(1, len(rank) + 1))
            for size, bounds in SIZES.items():
                result[name, size][j] = (rank if bounds is None else [
                    i for i in rank if bounds[0] <= day.at[src.symbols[i], "cap_usd"] < bounds[1]])
            # Mid-only and large-plus form an exhaustive, disjoint partition.
            mid = set(result[name, "mid2to10b"][j])
            large = set(result[name, "over10b"][j])
            assert mid.isdisjoint(large) and mid | large == set(result[name, "over2b"][j])
            assert set(result[name, "over200b"][j]) <= large
    return result


def period(curve, name):
    start, end = WINDOWS[name]
    indices = np.flatnonzero((curve.index >= start) & (curve.index <= end))
    first, last = indices[0], indices[-1]
    net = curve.krw10.to_numpy()
    anchor = net[first - 1] if first else 1.
    wealth = np.r_[anchor, net[first:last + 1]]
    return {"ret": float(wealth[-1] / anchor - 1),
            "dd": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
            "exposure": float(curve.exposure.iloc[first:last + 1].mean())}


def forward_checks(engine, ranked, standalone, baseline, reference, candidates):
    """Choose by prior standalone return at >=60% exposure, then change on the next open.

    Each candidate and the reference need a daily ranking. Candidates also need
    a standalone curve and their iteration order is the declared tie-breaker.
    The baseline is the switched reference portfolio for the comparison.
    """
    candidates = tuple(candidates)
    assert candidates and set(candidates) <= set(standalone) and reference in ranked
    src, gate = engine.src, engine.gate
    rows, training = [], []
    for train, test in [("h1", "h2"), ("q1", "q2"), ("q2", "q3"), ("q3", "q4_partial")]:
        winner, best = reference, -np.inf
        for candidate in candidates:
            fit = period(standalone[candidate], train)
            enough = fit["exposure"] >= .60
            training.append(dict(train=train, test=test, candidate=candidate, eligible=enough, **fit))
            if enough and fit["ret"] > best + 1e-12:
                winner, best = candidate, fit["ret"]
        # Only holdings selected on/after the first eligible auction can change.
        # E.g. March 31 is risk-off, and a choice learned at its close cannot
        # earn the selected screen's overnight gap into the April 1 recovery.
        start, end = WINDOWS[test]
        eligible_auctions = [j for j, day in enumerate(src.days) if start <= day <= end]
        first = eligible_auctions[0]
        assert first in engine.reviews["monthly1"]
        applied = [ranked[winner if start <= day <= end else reference][j] for j, day in enumerate(src.days)]
        selected = engine.evaluate(applied, "monthly1")["switch"]
        np.testing.assert_allclose(selected.krw10.iloc[:first], baseline.krw10.iloc[:first], atol=1e-12, rtol=0)
        actual = period(selected, test)
        control = period(baseline, test)
        def session_counts(window):
            start, end = WINDOWS[window]
            view = (selected.index >= start) & (selected.index <= end)
            return int(view.sum()), int((view & ~gate).sum())
        train_days, train_off = session_counts(train)
        test_days, test_off = session_counts(test)
        rows.append(dict(train=train, test=test, train_start=WINDOWS[train][0], train_end=WINDOWS[train][1],
                         test_start=WINDOWS[test][0], test_end=WINDOWS[test][1], selected=winner,
                         first_eligible_open=src.days[first],
                         train_days=train_days, train_off_days=train_off, test_days=test_days,
                         test_off_days=test_off, selected_ret=actual["ret"], published_ret=control["ret"],
                         selected_dd=actual["dd"], published_dd=control["dd"],
                         return_spread=actual["ret"] - control["ret"], dd_spread=actual["dd"] - control["dd"]))
    return pd.DataFrame(rows), pd.DataFrame(training)


def check_reference(name, path, reference):
    archive = reference.loc[name + "/monthly1"]
    assert path.index.equals(archive.index)
    fields = ["krw", "usd", "turn", "exposure", "krw10", "active", "sectors", "max_sector_weight", "krw25"]
    np.testing.assert_allclose(path[fields], archive[fields], rtol=2e-10, atol=1e-12)
    assert path.holdings.fillna("").tolist() == archive.holdings.fillna("").tolist()
    return float((path.krw10 - archive.krw10).abs().max())


def run():
    src = load_inputs()
    gate = market_gate(src)
    engine = TimingEngine(src, gate)
    published = recipes(src)
    cap_rows = pd.read_csv(HERE / "size_inputs.csv", float_precision="round_trip")
    ranked = rankings(src, engine, published, cap_rows)
    reference = pd.read_csv(PRIOR / "selected_daily.csv", index_col=["key", "date"])
    reviews = engine.reviews["monthly1"]
    paths, summaries, baskets, matched, checks = {}, [], [], [], {}
    for (name, size), rank in ranked.items():
        plan = engine.targets(rank, "monthly1")
        sizes = [len(rank[j]) for j in reviews]
        for j in reviews:
            holds = [i for i in rank[j][:7] if plan[j][i] > 0]
            assert len(holds) == np.count_nonzero(plan[j]) <= 7
            baskets.append(dict(recipe=name, size=size, trade_open=src.days[j], signal_date=src.signals[j],
                                passed=len(rank[j]), held=len(holds), initial_cash_slots=7 - len(holds),
                                selected="|".join(src.symbols[i] for i in rank[j][:7]),
                                executable="|".join(src.symbols[i] for i in holds)))
        curves = engine.evaluate(rank, "monthly1", detail=True, stress=True)
        for mode, path in curves.items():
            paths[name, size, mode] = path
            off = path.loc[~gate]
            summaries.append(dict(recipe=name, size=size, mode=mode, review_count=len(reviews),
                                  mean_eligible=np.mean(sizes), reviews_below_seven=sum(n < 7 for n in sizes),
                                  off_mean_names=off.active.mean(), off_mean_exposure=off.exposure.mean(),
                                  off_mean_sectors=off.sectors.mean(), off_max_sector_weight=off.max_sector_weight.max(),
                                  **describe(path)))
        if size == "published":
            checks[name] = check_reference(name, curves["switch"], reference)
        else:
            # Same number of executable slots each month from the broader published screen.
            equal_count = exposure_match(src, published[name], reviews, plan)
            assert all(np.count_nonzero(equal_count[j]) == np.count_nonzero(plan[j]) for j in reviews)
            routed = switched_targets(src.days, src.opn, engine.desk, equal_count, gate)
            matched_curve = _ledger(src, routed)
            control, restricted = describe(matched_curve), describe(curves["switch"])
            matched.append(dict(recipe=name, size=size, matched_off_names=matched_curve.active.loc[~gate].mean(),
                                matched_off_exposure=matched_curve.exposure.loc[~gate].mean(),
                                restricted_off_exposure=curves["switch"].exposure.loc[~gate].mean(),
                                **{f"matched_{k}": control[k] for k in ["year_ret", "h2_ret", "year_dd", "h2_dd"]},
                                **{f"restricted_{k}": restricted[k] for k in ["year_ret", "h2_ret", "year_dd", "h2_dd"]}))
    # The spell attribution helper reconciles cumulative excess to the full-year wealth ratio.
    # Name its internal reference key after the *current* balance/published control for this run.
    names = {ORIGINAL if (n, s) == ("balance", "published") else f"{n}/{s}": (n, s)
             for n in SCREENS for s in SIZES}
    switch_nav = {label: paths[n, s, "switch"].krw10 for label, (n, s) in names.items()}
    episodes, attribution = spells(src.days, gate, switch_nav)
    episodes["recipe"] = episodes.key.map(lambda k: names[k][0])
    episodes["size"] = episodes.key.map(lambda k: names[k][1])
    for i, row in episodes.iterrows():
        off = paths[row.recipe, row["size"], "switch"].loc[row["exit"]:row.last_off]
        assert len(off) == row.sessions_off
        episodes.at[i, "off_mean_names"] = off.active.mean()
        episodes.at[i, "off_mean_exposure"] = off.exposure.mean()
    episodes = episodes.drop(columns="key").rename(columns={"relative_return": "relative_to_balance_published"})
    for candidate in summaries:
        if candidate["mode"] == "switch":
            label = next(k for k, v in names.items() if v == (candidate["recipe"], candidate["size"]))
            candidate.update(attribution.loc[label].to_dict())
    choices = [s for s in SIZES if s != "published"]
    forwards, train = forward_checks(engine, {s: ranked["balance", s] for s in SIZES},
                                     {s: paths["balance", s, "always"] for s in choices},
                                     paths["balance", "published", "switch"], "published", choices)
    forwards = forwards.rename(columns={"selected": "selected_size"})
    train = train.rename(columns={"candidate": "size"})
    def save(frame, filename):
        frame.to_csv(HERE / filename, index=False, float_format="%.12g")
    save(pd.DataFrame(summaries), "size_results.csv")
    save(pd.DataFrame(baskets), "monthly_baskets.csv")
    save(episodes, "episodes.csv")
    save(pd.DataFrame(matched), "exposure_matched.csv")
    save(forwards, "forward_windows.csv")
    save(train, "training_windows.csv")
    fields = ["krw", "krw10", "krw25", "exposure", "active", "turn", "holdings"]
    daily = pd.concat({k: v[fields] for k, v in paths.items()}, names=["recipe", "size", "mode", "date"])
    save(daily.reset_index(), "daily.csv")
    validation = dict(sessions=len(src.days), first_day=src.days[0], last_day=src.days[-1],
                      risk_off_sessions=int((~gate).sum()), risk_off_spells=int(episodes.episode.nunique()),
                      cap_rows=len(cap_rows), actual_minimum_eligible_cap_usd=float(cap_rows.cap_usd.min()),
                      max_daily_krw10_error_vs_published=checks,
                      cap_floor_rank_differences={n: sum(ranked[n, "published"][j] != ranked[n, "over2b"][j]
                                                          for j in reviews) for n in SCREENS})
    (HERE / "ledger_checks.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    return pd.DataFrame(summaries), forwards


if __name__ == "__main__":
    results, forward = run()
    print(results[results["mode"].eq("switch")][
        ["recipe", "size", "year_ret", "h1_ret", "h2_ret", "year_dd", "h2_dd", "oct_ret",
         "off_mean_names", "off_mean_exposure"]].round(4).to_string(index=False))
    print(forward.round(4).to_string(index=False))
