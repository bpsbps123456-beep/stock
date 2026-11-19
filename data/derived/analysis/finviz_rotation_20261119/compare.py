"""Reprice Finviz-like screens and the reference desk using the same open/close ledger."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "desk_defense_20261119"
sys.path.insert(0, str(PREVIOUS))
from inputs import load_book
from evaluate import summarize
from study import replay, market_gates
from switch import switched_targets
from desk import strategies as desk

from technicals import load_panels
from fundamentals import load_fund_pass
from universe import broad_members


def scheduled(days, i, frequency):
    if i == 0 or frequency == "daily":
        return True
    if frequency == "weekly":
        return desk.iso_week(days[i]) != desk.iso_week(days[i-1])
    return days[i][:7] != days[i-1][:7]


def relative_target(pool, scores):
    selected = sorted((s for s in pool if pd.notna(scores.get(s))), key=lambda s: (-scores[s], s))[:7]
    return dict.fromkeys(selected, 1/7)


def desk_filtered(book, rows, eligible, kept):
    ranked = [r for r in rows if r["ticker"] in eligible]
    selected = desk.select(ranked, kept)
    raw = {r["key"]: r["time_weight"] for r in ranked if r["key"] in selected}
    total = sum(raw.values())
    capped = desk.project_caps({k: v / total for k, v in raw.items()}, book.sector)
    return {book.ticker[k]: v for k, v in capped.items()}, set(selected)


def blend(days, prices, base, screen):
    # One-unit funding per sleeve. At either scheduled event reset both to equal capital.
    events = set(base) | set(screen)
    plans = (base, screen)
    holdings = np.zeros((2, prices.shape[1]))
    cash = np.ones(2)
    out = {}
    for i in range(len(days)):
        if i not in events:
            continue
        o = prices[i]
        active = (holdings != 0).any(axis=0)
        if not np.isfinite(o[active]).all():
            raise ValueError(f"Missing mixed-sleeve mark on {days[i]}")
        marked = holdings * np.nan_to_num(o)
        nav = marked.sum(axis=1)+cash
        weights = marked / nav[:, None]
        for sleeve, plan in enumerate(plans):
            if i in plan:
                weights[sleeve] = plan[i]
        out[i] = weights.mean(axis=0)
        holdings = np.divide(weights, o, where=weights > 0, out=np.zeros_like(weights))
        cash = 1-weights.sum(axis=1)
    return out


def close_terminated(plan, opn, tradable):
    """Remove an unavailable order or mandatorily liquidate a terminated holding at its settlement mark."""
    if tradable.all():
        return plan
    updated = {}
    shares = np.zeros(opn.shape[1])
    cash = 1.
    for i, allowed in enumerate(tradable):
        if i not in plan and not ((shares != 0) & ~allowed).any():
            continue
        o = opn[i]
        if not np.isfinite(o[shares != 0]).all():
            raise ValueError(f"Missing liquidation or adjustment mark on row {i}")
        held = shares * np.nan_to_num(o)
        net = held.sum()+cash
        w = np.where(allowed, plan.get(i, held/net), 0.)
        if not (o[w > 0] > 0).all():
            raise ValueError(f"Missing live execution on row {i}")
        updated[i] = w
        shares = np.divide(net*w, o, where=w > 0, out=np.zeros_like(w))
        cash = net*(1-w.sum())
    return updated


def run():
    book = load_book()
    ref = desk.run_model(book)
    days = ref.days
    start = book.days.index(days[0])
    signals = book.days[start-1:-1]
    panels = load_panels()
    close = panels["close"].reindex(days)
    symbols = sorted(set(close.columns) | set(book.ticker.values()))
    loc = {s: i for i, s in enumerate(symbols)}

    def vector(target):
        w = np.zeros(len(symbols))
        for s, value in target.items():
            w[loc[s]] = value
        return w

    def prices(field, original):
        # The desk vintage takes precedence where it has an actual observed price.
        main = original.loc[days].reindex(columns=symbols)
        return main.combine_first(panels[field].reindex(index=days, columns=symbols))[symbols].to_numpy()

    opn, cls = prices("open", book.opn), prices("close", book.cls)
    tradable = panels["tradable"].reindex(index=days, columns=symbols, fill_value=True).to_numpy(bool)
    fx = book.fx[start:] / book.fx[start-1]
    base = {days.index(d): vector({book.ticker[k]: v for k, v in w.items()}) for d, w in ref.rebalances.items()}
    gate = np.ones(len(days), dtype=bool)
    market = market_gates(book, days)["SOXX100r1"]
    results, ledgers, shortlist, selection_log, diagnostics = {}, {}, {}, [], {}

    def store(key, plan, allow=gate):
        frame = replay(days, opn, cls, fx, plan, allow)
        results[key] = summarize(frame)
        ledgers[key] = frame

    store("base", base)
    np.testing.assert_allclose(ledgers["base"].krw, desk.model_index(book, ref)/100, rtol=1e-10)
    store("base_SOXX100", base, market)
    store("base_half", {i: w*.5 for i, w in base.items()})
    store("cash", {})
    store("SPY", {0: vector({"SPY": 1.})})
    financial = load_fund_pass(panels["price"])
    technical = panels["technical_pass"].eq(1)
    ownership = pd.read_csv(HERE / "ownership_proxy.csv", index_col=0)
    if not set(signals).issubset(ownership.index):
        raise ValueError("Ownership observations do not cover all signal dates")
    ownership = ownership.reindex(index=technical.index, columns=technical.columns)
    screens = {"ma": panels["ma_pass"].eq(1), "technical": technical,
               "quality": technical & financial.eq(True)}
    screens["ownership"] = screens["quality"] & (ownership > 0.4)
    rankings = [desk.momentum_ranking(book, start+i, d)[1] for i, d in enumerate(days)]
    broad = [broad_members(d) for d in days]
    time = []
    for d in days:
        pdf = desk.latest_before(book.pdfs, d)
        time.append({book.ticker[k] for k, weight in book.weights(pdf).items() if weight > 0} if pdf else set())
    scores = panels["relvol"].reindex(index=signals, columns=symbols).to_dict(orient="records")
    for screen, mask in screens.items():
        filtered = mask.reindex(index=signals, columns=symbols).eq(True)
        eligible = [set(filtered.columns[row]) for row in filtered.to_numpy(bool)]
        for freq in ("daily", "weekly", "monthly"):
            plans = {k: {} for k in ("broad", "time", "desk")}
            kept = set()
            for i, d in enumerate(days):
                if not scheduled(days, i, freq):
                    # TIME first published its 2026 basket after Jan 2, so all TIME modes
                    # get an initial decision on the first executable roster (like the desk).
                    if not (i and not time[i-1] and time[i]):
                        continue
                    scopes = ("time", "desk")
                else:
                    scopes = plans.keys()
                for scope in scopes:
                    pool = eligible[i] & (broad[i] if scope == "broad" else time[i])
                    if scope == "desk":
                        target, kept = desk_filtered(book, rankings[i], pool, kept)
                    else:
                        target = relative_target(pool, scores[i])
                    plans[scope][i] = vector(target)
                    executable = [s for s in target if tradable[i, loc[s]]]
                    selection_log.append((scope, screen, freq, d, signals[i], len(pool), "|".join(sorted(executable))))
            raw_broad = plans["broad"]
            for scope, plan in plans.items():
                plans[scope] = close_terminated(plan, opn, tradable)
                store(f"{scope}_{screen}_{freq}", plans[scope])
            if screen in ("quality", "ownership"):
                store(f"blend_{screen}_{freq}", blend(days, opn, base, plans["broad"]))
                store(f"exit_{screen}_{freq}", plans["broad"], market)
                store(f"switch_{screen}_{freq}", switched_targets(days, opn, base, plans["broad"], market))
            if screen == "ownership":
                refreshed = raw_broad.copy()
                for i in range(1, len(days)):
                    if market[i-1] and not market[i]:
                        refreshed[i] = vector(relative_target(eligible[i] & broad[i], scores[i]))
                refreshed = close_terminated(refreshed, opn, tradable)
                schedule = switched_targets(days, opn, base, refreshed, market)
                fresh = replay(days, opn, cls, fx, schedule, gate)
                diagnostics[f"rescreen_on_exit_{freq}"] = summarize(fresh)
                if freq == "daily":
                    np.testing.assert_allclose(fresh.krw, ledgers["switch_ownership_daily"].krw)

    stats = pd.DataFrame.from_dict(results, orient="index").sort_index()
    stats.index.name = "strategy"
    choices = stats.drop(index=["cash","base_SOXX100","base_half","SPY"])
    def choose(rows):
        eligible = rows.loc[rows.h1_mdd >= stats.loc["base","h1_mdd"]-1e-10]
        if eligible.empty:
            return "base"
        return eligible.index[eligible.h1_return_10bp >= eligible.h1_return_10bp.max()-1e-12].min()
    for group in ["all","broad","time","desk","blend","exit","switch"]:
        rows = choices if group == "all" else choices.loc[choices.index.str.startswith(group+"_")]
        shortlist[group] = choose(rows)
    stats.to_csv(HERE/"results.csv")
    # Fix coverage first, then apply the percentage threshold, keeping unknown distinct.
    sensitivity = {
        "ownership_known_only": screens["quality"] & ownership.notna(),
        "help_debt_and_ownership": technical & load_fund_pass(panels["price"], definition="help").eq(True)
                                   & (ownership > .4),
        "without_ma": panels["non_ma_pass"].eq(1) & financial.eq(True) & (ownership > .4),
    }
    for tag, mask in sensitivity.items():
        passes = mask.reindex(index=signals, columns=symbols).eq(True).to_numpy(bool)
        for freq in ("daily", "weekly", "monthly"):
            plan = {i: vector(relative_target(set(np.asarray(symbols)[passes[i]]) & broad[i], scores[i]))
                    for i in range(len(days)) if scheduled(days, i, freq)}
            p = close_terminated(plan, opn, tradable)
            diagnostics[f"{tag}_{freq}"] = summarize(replay(days, opn, cls, fx, p, gate))
    pd.DataFrame.from_dict(diagnostics, orient="index").to_csv(HERE/"sensitivity.csv")
    pd.DataFrame(shortlist.items(), columns=["group","strategy"]).to_csv(HERE/"selection.csv", index=False)
    pd.DataFrame(selection_log, columns=["universe","filter","frequency","trade_open","signal_close",
                                         "eligible_count","selected_tickers"]).to_csv(HERE/"decisions.csv", index=False)
    pd.concat(ledgers, names=["strategy","date"]).to_csv(HERE/"daily.csv")
    from rotation_report import write_report
    write_report(stats, shortlist, ledgers, market)


if __name__ == "__main__":
    run()
