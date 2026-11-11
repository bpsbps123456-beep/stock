"""SOXL close entry with a user-fixed -3%/+2% bracket; see protocol.json.

Costs are a separate cash debit. Price barriers do not move with cost assumptions.
No broker orders, account quantities or operational engine are changed.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ET = "America/New_York"
COSTS = (0, 5, 10, 20)
START, END = "2026-01-02", "2026-11-10"


def shared_stats(values):
    spec = importlib.util.spec_from_file_location("close_bracket_shared_stats", ROOT.parent / "soxl_close_rsi/code/study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.stats(values)


def inputs():
    minute = pd.read_parquet(ROOT / "cache/minute.parquet").sort_index()
    daily = pd.read_parquet(ROOT / "cache/daily.parquet").sort_index()
    schedule = xc.get_calendar("XNYS").schedule.loc[START:END]
    close = pd.Series(daily.Close.to_numpy(), index=daily.index.strftime("%Y-%m-%d"))
    assert not minute.index.has_duplicates and not close.index.has_duplicates
    assert str(minute.index.tz) == ET
    assert set(schedule.index.strftime("%Y-%m-%d")) <= set(close.index)
    return minute, close, schedule


def first_exit(frame, entry_price, policy="stop_first", gap_tp="open"):
    """First observable barrier touch; opening price precedes intrabar extremes."""
    if frame.empty:
        return None
    stop, target = entry_price * .97, entry_price * 1.02
    v = frame[["Open", "High", "Low"]].to_numpy(dtype=float)
    hits = np.flatnonzero((v[:, 2] <= stop) | (v[:, 1] >= target))
    if not len(hits):
        return None
    i = int(hits[0])
    op, hi, lo = v[i]
    both = bool(lo <= stop and hi >= target and stop < op < target)
    if op <= stop:
        px, reason = op, "stop_gap" if op < stop else "stop"
    elif op >= target:
        px, reason = op if gap_tp == "open" else target, "take_profit_gap"
    elif lo <= stop and (hi < target or policy == "stop_first"):
        px, reason = stop, "stop"
    else:
        px, reason = target, "take_profit"
    return {"exit_time": frame.index[i].isoformat(), "exit_price": float(px), "reason": reason,
            "ambiguity": both, "gap_stop": reason == "stop_gap", "exit_clock": "minute_start_label"}


def simulate(minute, close, schedule, window="RTH", policy="stop_first", run_start=START, gap_tp="open"):
    """One position, preset closing-order eligibility, sequential cash ledger."""
    assert window in ("RTH", "EXT") and policy in ("stop_first", "target_first")
    assert gap_tp in ("open", "target")
    schedule = schedule.loc[run_start:END]
    last_day = str(schedule.index[-1].date())
    groups = {k: v for k, v in minute.groupby(minute.index.strftime("%Y-%m-%d"))}
    day_num = {d: i for i, d in enumerate(schedule.index.strftime("%Y-%m-%d"))}
    cash = {bp: 100.0 for bp in COSTS}
    quantity = {bp: 0.0 for bp in COSTS}
    trades, equity = [], []
    position = None
    n_cutoff_blocked = 0

    def sell(hit, day):
        nonlocal position
        record = position.copy()
        record.update(hit)
        record["exit_day"] = day
        record["holding_sessions"] = day_num[day] - day_num[record["entry_day"]]
        record["holding_calendar_hours"] = (pd.Timestamp(record["exit_time"]) - pd.Timestamp(record["entry_time"])).total_seconds() / 3600
        record["gross_pct"] = (record["exit_price"] / record["entry_price"] - 1) * 100
        for bp in COSTS:
            c = bp / 10000
            record[f"net_{bp}bp_pct"] = (record["exit_price"] * (1-c) / (record["entry_price"] * (1+c)) - 1) * 100
            cash[bp] = quantity[bp] * record["exit_price"] * (1-c)
            quantity[bp] = 0.0
        trades.append(record)
        position = None

    def check(frame, day):
        if position is not None:
            hit = first_exit(frame, position["entry_price"], policy, gap_tp)
            if hit is not None:
                sell(hit, day)

    for session_day, session in schedule.iterrows():
        day = str(session_day.date())
        op, cl = session["open"].tz_convert(ET), session["close"].tz_convert(ET)
        frame = groups.get(day)
        if frame is None:
            raise ValueError(f"Missing minute session: {day}")
        assert (frame.Volume > 0).all()
        regular = frame[(frame.index >= op) & (frame.index < cl)]
        assert len(regular) == int((cl-op).total_seconds()/60), f"Incomplete RTH: {day}"
        if window == "RTH":
            active = regular
        else:
            active = frame[(frame.index.hour >= 4) & (frame.index.hour < 20)]
        cutoff = cl - pd.Timedelta(minutes=10)
        check(active[active.index < cutoff], day)
        eligible = position is None
        check(active[(active.index >= cutoff) & (active.index < cl)], day)
        # Closing daily price can differ from the last regular minute.
        p = float(close.at[day])
        if position is not None:
            p0 = position["entry_price"]
            if p <= p0*.97 or p >= p0*1.02:
                hit = {"exit_time": cl.isoformat(), "exit_price": min(p, p0*.97) if p <= p0*.97 else (p if gap_tp == "open" else p0*1.02),
                       "reason": "stop_close" if p <= p0*.97 else "take_profit_close", "ambiguity": False,
                       "gap_stop": bool(p < p0*.97), "exit_clock": "closing_snapshot"}
                sell(hit, day)
        if position is None and not eligible:
            n_cutoff_blocked += 1
        new_entry = position is None and eligible
        if new_entry:
            tid = len(trades) + 1
            position = {"window": window, "policy": policy, "gap_tp": gap_tp, "run_start": run_start, "trade_id": tid,
                        "entry_day": day, "entry_time": cl.isoformat(), "entry_price": p}
            for bp in COSTS:
                quantity[bp] = cash[bp] / (p * (1+bp/10000))
                cash[bp] = 0.0
        row = {"window": window, "policy": policy, "gap_tp": gap_tp, "run_start": run_start, "day": day,
               "session_close": cl.isoformat(), "close": p, "position": int(position is not None),
               "trade_id": position["trade_id"] if position else None}
        for bp in COSTS:
            row[f"cash_{bp}bp"] = cash[bp]
            row[f"nav_{bp}bp"] = cash[bp] + quantity[bp]*p
        equity.append(row)
        if window == "EXT" and day != last_day:
            after = active[active.index >= cl + (pd.Timedelta(minutes=1) if new_entry else pd.Timedelta(0))]
            check(after, day)
    if position:
        record = position.copy()
        record.update({"exit_day": None, "exit_time": None, "exit_price": None, "reason": "open_censored",
                       "ambiguity": False, "gap_stop": False, "exit_clock": None,
                       "holding_sessions": day_num[last_day]-day_num[record["entry_day"]],
                       "holding_calendar_hours": (pd.Timestamp(equity[-1]["session_close"])-pd.Timestamp(record["entry_time"])).total_seconds()/3600,
                       "gross_pct": None, "mark_price": float(close.at[last_day]),
                       "unrealized_gross_pct": (float(close.at[last_day])/record["entry_price"]-1)*100})
        trades.append(record)
    return pd.DataFrame(trades), pd.DataFrame(equity), {"late_exit_reentry_deferred_days": n_cutoff_blocked}


def trade_metrics(trades):
    done = trades[trades.reason.ne("open_censored")]
    result = {"closed": len(done), "open": int(trades.reason.eq("open_censored").sum()),
              "target": int(done.reason.str.startswith("take_profit").sum()),
              "stop": int(done.reason.str.startswith("stop").sum()),
              "gap_stops": int(done.gap_stop.sum()), "ambiguous": int(done.ambiguity.sum())}
    result["gross"] = shared_stats(done.gross_pct)
    result["costs"] = {str(bp): shared_stats(done[f"net_{bp}bp_pct"]) for bp in COSTS}
    for bp in COSTS:
        a = done[f"net_{bp}bp_pct"].to_numpy(float)
        loss = -a[a < 0].sum()
        result["costs"][str(bp)]["profit_factor"] = float(a[a > 0].sum()/loss) if loss else None
    if len(done):
        result["average_hold_sessions"] = float(done.holding_sessions.mean())
        result["median_hold_hours"] = float(done.holding_calendar_hours.median())
        result["max_hold_sessions"] = int(done.holding_sessions.max())
    return result


def summarize(trades, equity, extra, close):
    periods = {"all": (START, END), "H1": (START, "2026-06-30"), "H2": ("2026-07-01", END)}
    metrics = {}
    for label, (start, end) in periods.items():
        q = equity[equity.day.between(start, end)]
        if q.empty:
            continue
        before = equity[equity.day.lt(start)]
        stats = trade_metrics(trades[trades.exit_day.between(start, end) | (trades.reason.eq("open_censored") & trades.entry_day.between(start, end))])
        stats["portfolio"] = {}
        for bp in COSTS:
            initial = float(before.iloc[-1][f"nav_{bp}bp"]) if len(before) else 100.0
            path = np.r_[initial, q[f"nav_{bp}bp"].to_numpy(float)]
            stats["portfolio"][str(bp)] = {"start_nav": initial, "end_nav": float(path[-1]),
                "return_pct": float((path[-1]/initial-1)*100),
                "daily_close_mdd_pct": float((path/np.maximum.accumulate(path)-1).min()*100)}
        stats["trade_attribution"] = "closed trades assigned by exit date; open trade kept separately"
        metrics[label] = stats
    if trades.reason.eq("open_censored").any():
        metrics["terminal_open"] = trades[trades.reason.eq("open_censored")].iloc[-1].dropna().to_dict()
    metrics["terminal_liquidation_nav"] = {str(bp): float(equity.iloc[-1][f"nav_{bp}bp"])*(1-bp/10000) if equity.iloc[-1].position else float(equity.iloc[-1][f"nav_{bp}bp"]) for bp in COSTS}
    metrics.update(extra)
    return metrics


def main():
    minute, close, schedule = inputs()
    root = ROOT / "results"
    root.mkdir(exist_ok=True)
    all_t, all_e, models = [], [], []
    for window in ("RTH", "EXT"):
        for policy in ("stop_first", "target_first"):
            for gap_tp in ("open", "target"):
                for start in (START, "2026-07-01"):
                    t, e, extra = simulate(minute, close, schedule, window, policy, start, gap_tp)
                    all_t.append(t); all_e.append(e)
                    models.append({"window": window, "policy": policy, "gap_tp": gap_tp, "run_start": start,
                                   **summarize(t, e, extra, close)})
    pd.concat(all_t, ignore_index=True).to_csv(root / "trades.csv", index=False)
    pd.concat(all_e, ignore_index=True).to_csv(root / "equity.csv", index=False)
    benchmark = {}
    for name, start, end in (("all", START, END), ("H1", START, "2026-06-30"), ("H2", "2026-07-01", END)):
        # H2 continuous B&H uses June30 close; freshH2 entry uses July1 close.
        dates = close.index[(close.index >= start) & (close.index <= end)]
        p = close.loc[dates].to_numpy(float)
        benchmark[name] = {}
        for bp in COSTS:
            path = np.r_[100.0, p / (p[0]*(1+bp/10000))*100]
            benchmark[name][str(bp)] = {"return_pct": float((path[-1]/100-1)*100),
                "daily_close_mdd_pct": float((path/np.maximum.accumulate(path)-1).min()*100)}
    # Continuous H2 B&H marks the same exposure from June30, no additional buy cost.
    p = close.loc[(close.index >= "2026-06-30") & (close.index <= END)].to_numpy(float)
    benchmark["H2_continuous"] = {"return_pct": float((p[-1]/p[0]-1)*100),
        "daily_close_mdd_pct": float((p/np.maximum.accumulate(p)-1).min()*100)}
    output = {"protocol": json.loads((ROOT / "protocol.json").read_text(encoding="utf-8")),
              "sessions": len(schedule), "models": models, "buy_hold": benchmark}
    (root / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for m in models:
        if m["run_start"] == START and m["policy"] == "stop_first":
            s = m["all"]
            print(json.dumps({"window": m["window"], "gap_tp": m["gap_tp"], "closed": s["closed"], "TP": s["target"], "SL": s["stop"],
                              "mean10": s["costs"]["10"].get("mean_pct"), "portfolio10": s["portfolio"]["10"],
                              "H2": m["H2"]["portfolio"]["10"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
