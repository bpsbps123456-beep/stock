"""Independent raw-bar audit for the frozen close/+2%/-3% SOXL study.

The reference below does not import the study's execution or accounting engine.
Only the fixed protocol, raw input prices, and final outputs are shared.
"""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ET = "America/New_York"


def independent_touch(o, h, l, stop, target, priority="stop", gap_tp="open"):
    """Opening print precedes bar extrema; otherwise chronology is unknowable."""
    if o <= stop:
        return "stop_gap", float(o), False
    if o >= target:
        return "target_gap", float(o if gap_tp == "open" else target), False
    low_hit, high_hit = l <= stop, h >= target
    if low_hit and high_hit:
        return (("stop", float(stop), True) if priority == "stop"
                else ("target", float(target), True))
    if low_hit:
        return "stop", float(stop), False
    if high_hit:
        return "target", float(target), False
    return None


def synthetic_checks():
    r = independent_touch(94, 99, 93, 97, 102)
    assert r == ("stop_gap", 94.0, False)
    assert r[1] / 100 - 1 < -.03
    assert independent_touch(106, 108, 105, 97, 102) == ("target_gap", 106.0, False)
    assert independent_touch(106, 108, 105, 97, 102, gap_tp="target") == ("target_gap", 102.0, False)
    assert independent_touch(100, 103, 96, 97, 102) == ("stop", 97.0, True)
    assert independent_touch(100, 103, 96, 97, 102, "target") == ("target", 102.0, True)
    assert independent_touch(100, 102, 98, 97, 102) == ("target", 102.0, False)
    assert independent_touch(100, 101, 97, 97, 102) == ("stop", 97.0, False)
    assert independent_touch(100, 101, 98, 97, 102) is None
    # The cost is a separate proportional debit, not a changed barrier price.
    q = 1 / (100 * 1.001)
    assert np.isclose(q * 102 * .999 - 1, .01796203796203799)
    assert np.isclose(q * 97 * .999 - 1, -.03193806193806192)
    check = {"gap_stop_worse_than_minus3": True, "gap_target_open_improvement": True,
             "gap_target_conservative_sensitivity": True,
            "same_bar_both_priorities": True, "inclusive_touches": True,
            "no_hit_remains_open": True, "separate_cost_debit": True}
    days = ("2026-01-02", "2026-01-05", "2026-01-06")
    prices = pd.DataFrame({"Close": [100., 100., 100.]},
                          index=pd.DatetimeIndex(days, tz=ET))

    def bars(extra=()):
        rows = [(f"{day} 15:49", 100., 100., 100., 100.) for day in days]
        rows.extend(extra)
        return pd.DataFrame({"Open": [r[1] for r in rows], "High": [r[2] for r in rows],
                             "Low": [r[3] for r in rows], "Close": [r[4] for r in rows]},
                            index=pd.DatetimeIndex([r[0] for r in rows], tz=ET)).sort_index()

    # This extreme price interval precedes the initial close entry.
    before = bars([("2026-01-02 15:59", 100., 200., 1., 100.)])
    got = independent_simulate(before, prices, days[0], days[-1])
    assert len(got["trades"]) == 0
    assert len([e for e in got["events"] if e[0] == "entry"]) == 1
    assert got["open_position"]["entry_at"].startswith(days[0])
    assert np.isclose(got["equity"][-1]["nav"], 1 / 1.001)
    assert np.isclose(got["open_position"]["liquidation_nav"], .999 / 1.001)
    check.update(preentry_extrema_ignored=True, no_additions_while_held=True,
                 unclosed_position_marked_without_exit_debit=True,
                 liquidation_value_separate=True)

    late = bars([("2026-01-05 15:50", 100., 103., 100., 102.)])
    got = independent_simulate(late, prices, days[0], days[-1])
    assert [e[1][:10] for e in got["events"] if e[0] == "entry"] == [days[0], days[-1]]
    check["unconfirmed_cutoff_bar_cannot_reenter_same_close"] = True

    # A newly bought position cannot use any part of the closing minute bar.
    after = bars([("2026-01-02 16:00", 100., 200., 1., 100.),
                  ("2026-01-06 16:01", 100., 200., 1., 100.)])
    got = independent_simulate(after, prices, days[0], days[-1], "EXT")
    assert len(got["trades"]) == 0
    check.update(new_entry_closing_minute_excluded=True, terminal_postclose_excluded=True)
    return check


def independent_simulate(minute, daily, start, end, window="RTH", cost=.001,
                         priority="stop", gap_tp="open"):
    """One cash ledger, frozen close-10-minute entry eligibility, no additions."""
    schedule = xc.get_calendar("XNYS").schedule.loc[start:end]
    grouped = {str(day): frame for day, frame in minute.groupby(minute.index.date)}
    cash, pos = 1.0, None
    trades, equity, events = [], [], []
    for j, (session, calendar_row) in enumerate(schedule.iterrows()):
        day = str(session.date())
        op = calendar_row["open"].tz_convert(ET)
        cl = calendar_row["close"].tz_convert(ET)
        cutoff = cl - pd.Timedelta(minutes=10)
        today = grouped[day]
        close = float(daily.loc[day, "Close"])
        pre = today[(today.index >= (op if window == "RTH" else op.normalize() + pd.Timedelta(hours=4)))
                    & (today.index < cl)]
        eligible = None

        def process(t, b, point=False):
            nonlocal cash, pos
            if pos is None:
                return
            if point:
                # Closing snapshot follows the same gap-price convention.
                hit = independent_touch(float(b), float(b), float(b), pos["stop"], pos["target"], priority, gap_tp)
            else:
                assert t >= pos["entry_at"]
                hit = independent_touch(float(b.Open), float(b.High), float(b.Low),
                                        pos["stop"], pos["target"], priority, gap_tp)
            if hit is None:
                return
            reason, xp, ambiguous = hit
            cash = pos["qty"] * xp * (1 - cost)
            trades.append({"entry_at": pos["entry_at"].isoformat(), "exit_at": t.isoformat(),
                           "entry_price": pos["price"], "exit_price": xp,
                           "reason": reason, "ambiguous": ambiguous,
                           "exit_clock": "closing_snapshot" if point else "minute_start_label",
                           "net_pct": (cash / pos["capital"] - 1) * 100,
                           "cash": cash})
            events.append(("exit", t.isoformat()))
            pos = None

        for t, b in pre.iterrows():
            if t >= cutoff and eligible is None:
                eligible = pos is None
            process(t, b)
        if eligible is None:
            eligible = pos is None
        # Auction is a price observation only for an already-held position.
        process(cl, close, point=True)
        entered_at_close = eligible and pos is None
        if entered_at_close:
            capital = cash
            pos = {"entry_at": cl, "price": close, "stop": close * .97,
                   "target": close * 1.02, "qty": capital / (close * (1 + cost)),
                   "capital": capital}
            events.append(("entry", cl.isoformat()))
            cash = 0.0
        nav = cash if pos is None else pos["qty"] * close
        equity.append({"day": day, "nav": nav, "held": pos is not None,
                       "closed_so_far": len(trades)})
        if window != "RTH" and j < len(schedule) - 1:
            after = cl + pd.Timedelta(minutes=1) if entered_at_close else cl
            post = today[(today.index >= after)
                         & (today.index < cl.normalize() + pd.Timedelta(hours=20))]
            for t, b in post.iterrows():
                process(t, b)
    tail = None if pos is None else {
        "entry_at": pos["entry_at"].isoformat(), "entry_price": pos["price"],
        "mark_price": close, "mark_nav": equity[-1]["nav"],
        "unrealized_pct": (equity[-1]["nav"] / pos["capital"] - 1) * 100,
        "liquidation_nav": equity[-1]["nav"] * (1 - cost)}
    return {"trades": trades, "equity": equity, "open_position": tail, "events": events}


def independent_statistics(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0}
    means = np.random.default_rng(20261105).choice(a, (10000, len(a)), replace=True).mean(axis=1)
    eq = np.r_[1., np.cumprod(1 + a / 100)]
    return {"n": len(a), "mean_pct": float(a.mean()), "median_pct": float(np.median(a)),
            "win_rate_pct": float(np.mean(a > 0) * 100), "worst_pct": float(a.min()),
            "best_pct": float(a.max()), "mean_ci95_pct": np.quantile(means, [.025, .975]).tolist(),
            "compound_pct": float((eq[-1] - 1) * 100),
            "closed_trade_mdd_pct": float(np.min(eq / np.maximum.accumulate(eq) - 1) * 100)}


def equal_number(a, b, name):
    if a is None or b is None:
        assert a is None and b is None, (name, a, b)
    else:
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, err_msg=name)


def raw_price_consistency(minute, daily):
    assert str(minute.index.tz) == str(daily.index.tz) == ET
    assert not minute.index.has_duplicates and not daily.index.has_duplicates
    v = minute[["Open", "High", "Low", "Close"]].to_numpy(float)
    assert np.isfinite(v).all() and (v > 0).all()
    assert (v[:, 1] >= np.maximum(v[:, 0], v[:, 3])).all()
    assert (v[:, 2] <= np.minimum(v[:, 0], v[:, 3])).all()
    schedule = xc.get_calendar("XNYS").schedule.loc["2026-01-02":"2026-11-10"]
    grouped = {str(day): f for day, f in minute.groupby(minute.index.date)}
    comparisons = []
    gaps = []
    prior_close = None
    for day, cal in schedule.iterrows():
        ds = str(day.date())
        op, cl = cal["open"].tz_convert(ET), cal["close"].tz_convert(ET)
        f = grouped[ds]
        reg = f[(f.index >= op) & (f.index < cl)]
        assert reg.index.equals(pd.date_range(op, cl, freq="1min", inclusive="left"))
        p = float(daily.loc[ds, "Close"])
        last = float(reg.iloc[-1].Close)
        post = float(f.at[cl, "Open"]) if cl in f.index else None
        comparisons.append({"day": ds, "daily_close": p, "last_regular_minute_close": last,
                            "closing_minute_open": post,
                            "last_minute_difference_pct": (p / last - 1) * 100,
                            "closing_minute_difference_pct": None if post is None else (p / post - 1) * 100})
        if prior_close is not None:
            gaps.append({"day": ds, "previous_daily_close": prior_close,
                         "regular_open": float(reg.iloc[0].Open),
                         "gap_pct": (float(reg.iloc[0].Open) / prior_close - 1) * 100})
        prior_close = p
    # Scale mismatches would be many percent every day, not isolated differences
    # between consolidated daily benchmarks and individual minute aggregates.
    assert max(abs(r["last_minute_difference_pct"]) for r in comparisons) < 5
    return {"sessions": len(schedule), "all_regular_minutes_present": True,
            "max_abs_daily_vs_last_minute_pct": max(abs(r["last_minute_difference_pct"]) for r in comparisons),
            "max_abs_daily_vs_closing_minute_open_pct": max(abs(r["closing_minute_difference_pct"]) for r in comparisons if r["closing_minute_difference_pct"] is not None),
            "worst_five_regular_open_gaps": sorted(gaps, key=lambda x: x["gap_pct"])[:5],
            "largest_five_regular_open_gaps": sorted(gaps, key=lambda x: -x["gap_pct"])[:5],
            "daily_minute_comparisons": comparisons}


def audit_model(model, all_trades, all_equity, minute, daily):
    w, p, start, gap_tp = (model[k] for k in ("window", "policy", "run_start", "gap_tp"))
    filt_t = all_trades.window.eq(w) & all_trades.policy.eq(p) & all_trades.run_start.eq(start) & all_trades.gap_tp.eq(gap_tp)
    filt_e = all_equity.window.eq(w) & all_equity.policy.eq(p) & all_equity.run_start.eq(start) & all_equity.gap_tp.eq(gap_tp)
    observed_t = all_trades[filt_t].reset_index(drop=True)
    observed_e = all_equity[filt_e].reset_index(drop=True)
    ref = independent_simulate(minute, daily, start, "2026-11-10", w,
                               priority="stop" if p == "stop_first" else "target", gap_tp=gap_tp)
    rt = pd.DataFrame(ref["trades"])
    re = pd.DataFrame(ref["equity"])
    done = observed_t[observed_t.reason.ne("open_censored")].reset_index(drop=True)
    assert len(done) == len(rt)
    assert len(observed_t) == len(rt) + int(ref["open_position"] is not None)
    assert done.entry_time.tolist() == rt.entry_at.tolist()
    assert done.exit_time.tolist() == rt.exit_at.tolist()
    equal_number(done.entry_price.to_numpy(), rt.entry_price.to_numpy(), "entry prices")
    equal_number(done.exit_price.to_numpy(), rt.exit_price.to_numpy(), "first hit prices")
    assert done.ambiguity.tolist() == rt.ambiguous.tolist()
    assert done.reason.str.startswith("take_profit").tolist() == rt.reason.str.startswith("target").tolist()
    assert done.exit_clock.tolist() == rt.exit_clock.tolist()
    assert observed_e.day.tolist() == re.day.tolist()
    assert observed_e.position.astype(bool).tolist() == re.held.tolist()
    independent_nav = {}
    independent_ret = {}
    for bp in (0, 5, 10, 20):
        c = bp / 10000
        nav = re.nav.to_numpy(float) * (((1-c)/(1+c)) / (.999/1.001)) ** re.closed_so_far.to_numpy(int)
        nav *= np.where(re.held, 1.001/(1+c), 1)
        independent_nav[bp] = nav * 100
        equal_number(observed_e[f"nav_{bp}bp"].to_numpy(), nav * 100, f"NAV {bp}bp")
        equal_number(observed_e[f"cash_{bp}bp"].to_numpy(), np.where(re.held, 0, nav * 100), f"cash {bp}bp")
        ret = (rt.exit_price.to_numpy() * (1-c) / (rt.entry_price.to_numpy() * (1+c)) - 1) * 100
        independent_ret[bp] = ret
        equal_number(done[f"net_{bp}bp_pct"].to_numpy(), ret, f"returns {bp}bp")
    g = (rt.exit_price.to_numpy() / rt.entry_price.to_numpy() - 1) * 100
    equal_number(done.gross_pct.to_numpy(), g, "gross returns")
    periods_checked = 0
    for period, a, b in (("all", start, "2026-11-10"),
                         ("H1", start, "2026-06-30"),
                         ("H2", "2026-07-01", "2026-11-10")):
        mask_e = re.day.between(a, b)
        if not mask_e.any():
            assert period not in model
            continue
        m = model[period]
        mask_t = rt.exit_at.str[:10].between(a, b).to_numpy()
        assert m["closed"] == int(mask_t.sum())
        assert m["target"] == int(rt[mask_t].reason.str.startswith("target").sum())
        assert m["stop"] == int(rt[mask_t].reason.str.startswith("stop").sum())
        assert m["ambiguous"] == int(rt[mask_t].ambiguous.sum())
        for key, value in independent_statistics(g[mask_t]).items():
            equal_number(m["gross"][key], value, f"{period} gross {key}")
        for bp in (0, 5, 10, 20):
            vals = independent_ret[bp][mask_t]
            for key, value in independent_statistics(vals).items():
                equal_number(m["costs"][str(bp)][key], value, f"{period} {bp}bp {key}")
            loss = -vals[vals < 0].sum()
            pf = float(vals[vals > 0].sum() / loss) if loss else None
            equal_number(m["costs"][str(bp)]["profit_factor"], pf, f"{period} PF")
            nav = independent_nav[bp]
            before = np.flatnonzero(re.day.to_numpy() < a)
            initial = nav[before[-1]] if len(before) else 100.
            path = np.r_[initial, nav[mask_e.to_numpy()]]
            expected = {"start_nav": initial, "end_nav": path[-1],
                        "return_pct": (path[-1] / initial - 1) * 100,
                        "daily_close_mdd_pct": np.min(path / np.maximum.accumulate(path) - 1) * 100}
            for key, value in expected.items():
                equal_number(m["portfolio"][str(bp)][key], value, f"{period} portfolio {key}")
        periods_checked += 1
    for bp in (0, 5, 10, 20):
        liquid = independent_nav[bp][-1] * (1 - bp/10000) if ref["open_position"] else independent_nav[bp][-1]
        equal_number(model["terminal_liquidation_nav"][str(bp)], liquid, "terminal liquidation")
    if ref["open_position"]:
        tail = observed_t[observed_t.reason.eq("open_censored")].iloc[0]
        assert tail.entry_time == ref["open_position"]["entry_at"]
        equal_number(tail.mark_price, ref["open_position"]["mark_price"], "terminal mark")
    return {"window": w, "policy": p, "run_start": start, "gap_tp": gap_tp,
            "closed_trade_rows_checked": len(rt), "daily_equity_rows_checked": len(re),
            "periods_checked": periods_checked, "all_four_costs_match": True,
            "return_10bp_pct": model["all"]["portfolio"]["10"]["return_pct"]}


def future_mutation_check(minute, daily):
    spec = importlib.util.spec_from_file_location("audit_observed_study", HERE / "study.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    schedule = xc.get_calendar("XNYS").schedule.loc["2026-01-02":"2026-11-10"]
    close = pd.Series(daily.Close.to_numpy(), index=daily.index.strftime("%Y-%m-%d"))
    b = minute.copy()
    later = b.index >= pd.Timestamp("2026-07-01", tz=ET)
    b.loc[later, ["Open", "High", "Low", "Close"]] *= .5
    changed_close = close.copy()
    changed_close.loc[changed_close.index >= "2026-07-01"] *= .5
    checked = []
    for w in ("RTH", "EXT"):
        for gap_tp in ("open", "target"):
            # Invoke the production entry point only for this metamorphic test.
            ta, ea, _ = engine.simulate(minute, close, schedule, w, "stop_first", "2026-01-02", gap_tp=gap_tp)
            tb, eb, _ = engine.simulate(b, changed_close, schedule, w, "stop_first", "2026-01-02", gap_tp=gap_tp)
            a, z = ea[ea.day.le("2026-06-30")], eb[eb.day.le("2026-06-30")]
            pd.testing.assert_frame_equal(a, z)
            a, z = ta[ta.exit_day.le("2026-06-30")], tb[tb.exit_day.le("2026-06-30")]
            pd.testing.assert_frame_equal(a, z)
            assert not np.allclose(ea.nav_10bp, eb.nav_10bp)
            checked.append({"window": w, "gap_tp": gap_tp})
    return {"H1_all_outputs_unchanged": True, "H2_outputs_changed": True, "models": checked}


def main():
    minute = pd.read_parquet(HERE / "cache/minute.parquet").sort_index()
    daily = pd.read_parquet(HERE / "cache/daily.parquet").sort_index()
    trades = pd.read_csv(HERE / "results/trades.csv")
    equity = pd.read_csv(HERE / "results/equity.csv")
    summary = json.loads((HERE / "results/summary.json").read_text(encoding="utf-8"))
    checks = synthetic_checks()
    source = raw_price_consistency(minute, daily)
    rows = []
    assert len(summary["models"]) == 16
    for model in summary["models"]:
        rows.append(audit_model(model, trades, equity, minute, daily))
        print(json.dumps(rows[-1]), flush=True)
    mutation = future_mutation_check(minute, daily)
    result = {"pass": True, "independent_execution_and_cash_ledger": True,
              "synthetic_checks": checks, "raw_price_consistency": source,
              "models": rows, "future_price_mutation": mutation,
              "trade_rows_checked": sum(r["closed_trade_rows_checked"] for r in rows),
              "daily_equity_rows_checked": sum(r["daily_equity_rows_checked"] for r in rows)}
    (HERE / "results/audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"pass": True, "models": len(rows), "trade_rows": result["trade_rows_checked"],
                      "equity_rows": result["daily_equity_rows_checked"]}), flush=True)


if __name__ == "__main__":
    main()
