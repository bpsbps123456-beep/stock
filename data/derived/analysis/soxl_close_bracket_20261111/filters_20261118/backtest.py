"""Apply the frozen entry filters to the existing SOXL close-bracket engine."""
import argparse
import importlib.util
import json
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ENGINE_SPEC = importlib.util.spec_from_file_location("close_bracket_engine", HERE.parent / "study.py")
ENGINE = importlib.util.module_from_spec(ENGINE_SPEC)
ENGINE_SPEC.loader.exec_module(ENGINE)
PLAN = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
JAN, JUL, SEP, END = PLAN["start"], "2026-07-01", "2026-09-01", PLAN["end"]
SCENARIOS = ((JAN, "RTH", "open"), (JUL, "RTH", "open"), (JUL, "EXT", "open"),
             (JUL, "RTH", "target"), (SEP, "RTH", "open"))


def period_stats(trades, equity, flags, end):
    equity = equity[equity.day.le(end)]
    # Truncate by exit time as well as entry time to keep rankings point-in-time.
    entered = trades[trades.entry_day.astype("string").le(end)]
    closed = entered[entered.exit_day.astype("string").le(end).fillna(False)]
    rets = closed.net_10bp_pct.dropna().astype(float)
    path = np.r_[100., equity.nav_10bp.to_numpy()]
    final = equity.iloc[-1]
    stats = dict(sessions=len(equity), signals=int(flags.loc[equity.day].sum()),
                 entries=len(entered), closed=len(closed), open_positions=int(final.position),
                 win_pct=float(rets.gt(0).mean()*100), mdd10=float((path/np.maximum.accumulate(path)-1).min()*100),
                 best10=rets.max(), worst10=rets.min())
    for bp in ENGINE.COSTS:
        stats[f"net{bp}"] = final[f"nav_{bp}bp"] - 100
    # Leave all other entries and terminal marks on their original dates.
    stats["without_best10"] = path[-1] / (1 + stats["best10"]/100) - 100
    assert stats["entries"] == stats["closed"] + stats["open_positions"]
    return stats


def choose(main, period, minimum):
    candidates = [f for f in PLAN["filters"] if f not in ("baseline", "bb1")]
    scores = main.loc[main.period.eq(period)].set_index("filter").loc[candidates]
    eligible = scores[scores.closed.ge(minimum) & scores.net10.gt(0)]
    return None if eligible.empty else eligible.net10.idxmax()


def inputs():
    minute = pd.read_parquet(HERE / "cache/minute.parquet").sort_index()
    daily = pd.read_parquet(HERE / "cache/daily.parquet").sort_index()
    close = pd.Series(daily.Close.to_numpy(), index=daily.index.strftime("%Y-%m-%d"))
    schedule = xc.get_calendar("XNYS").schedule.loc[JAN:END]
    sessions = schedule.index.strftime("%Y-%m-%d")
    assert close.index.is_unique and sessions.isin(close.index).all()
    assert minute.index.is_unique and str(minute.index.tz) == ENGINE.ET
    return minute, close, schedule, sessions


def buffer_check():
    minute, close, schedule, sessions = inputs()
    features = pd.read_csv(OUT / "earlier_cutoff_features.csv", index_col="day")
    assert features.index.is_unique and sessions.isin(features.index).all()
    rows = []
    for name in ("above_vwap", "late_up", "rsi_turn", "down5"):
        raw = features.loc[sessions, name]
        assert raw.isin([0, 1]).all()
        flags = raw.eq(1)
        trades, equity, _ = ENGINE.simulate(minute, close, schedule, run_start=JUL,
                                            run_end=END, entry_days=set(flags.index[flags]), cutoff_minutes=15)
        rows.append(dict(filter=name, cutoff_minutes=15, **period_stats(trades, equity, flags, END)))
    pd.DataFrame(rows).to_csv(OUT / "earlier_cutoff_metrics.csv", index=False)


def compare_belt_components():
    """Post-result attribution on the same dates with both BB-1 inputs."""
    minute, close, schedule, _ = inputs()
    features = pd.read_csv(OUT / "features.csv", index_col="day")
    end = "2026-11-11"  # last observation in the frozen historical belt study
    panel = features.loc[JUL:end]
    known = panel.bb1.notna()
    deep = known & panel.bb1_prior_soxx_drawdown60.le(-.20)
    belt = deep & panel.bb1_flow_percentile.ge(80)
    assert np.array_equal(belt, panel.bb1.eq(1))
    rows = []
    for name, flags in (("drawdown_only", deep), ("bb1", belt)):
        for gap in ("open", "target"):
            trades, equity, _ = ENGINE.simulate(minute, close, schedule, run_start=JUL, run_end=end,
                                                gap_tp=gap, entry_days=set(flags.index[flags]))
            rows.append(dict(filter=name, start=JUL, end=end, flow_known_days=int(known.sum()), gap_tp=gap,
                             **period_stats(trades, equity, flags, end)))
    pd.DataFrame(rows).to_csv(OUT / "belt_components.csv", index=False)
    # A price-only filter can also be evaluated after the flow archive ends.
    sector = pd.read_parquet(HERE / "cache/SOXX_warm_daily.parquet").Close
    drawdown = (sector/sector.rolling(60, min_periods=60).max()-1).shift()
    drawdown.index = drawdown.index.strftime("%Y-%m-%d")
    drawdown = drawdown.loc[JUL:END]
    assert drawdown.notna().all()
    flags = drawdown.le(-.20)
    assert np.array_equal(flags.loc[panel.index], panel.bb1_prior_soxx_drawdown60.le(-.20))
    rows = []
    for start, window, gap in ((JUL, "RTH", "open"), (SEP, "RTH", "open"),
                               (JUL, "RTH", "target"), (JUL, "EXT", "open")):
        trades, equity, _ = ENGINE.simulate(minute, close, schedule, run_start=start, run_end=END,
                                            window=window, gap_tp=gap, entry_days=set(flags.index[flags]))
        rows.append(dict(start=start, end=END, window=window, gap_tp=gap,
                         **period_stats(trades, equity, flags, END)))
    pd.DataFrame(rows).to_csv(OUT / "drawdown_only.csv", index=False)


def main():
    minute, close, schedule, sessions = inputs()
    features = pd.read_csv(OUT / "features.csv", index_col="day")
    assert features.index.is_unique and sessions.isin(features.index).all()
    metrics, trade_books, equity_books, monthly = [], [], [], []
    fills = ["entry_day", "exit_day", "net_10bp_pct"]
    for name in PLAN["filters"]:
        if name == "baseline":
            flags = pd.Series(True, index=sessions)
        else:
            raw = features.loc[sessions, name]
            assert raw.dropna().isin([False, True, 0, 1]).all(), name
            flags = raw.eq(1).fillna(False)
        allowed = set(flags.index[flags])
        for start, window, gap in SCENARIOS:
            trades, equity, _ = ENGINE.simulate(minute, close, schedule, window=window, gap_tp=gap,
                                                run_start=start, run_end=END, entry_days=allowed)
            equity_books.append(equity.assign(filter=name))
            if trades.empty:
                trades = pd.DataFrame(columns=fills)
            else:
                trade_books.append(trades.assign(filter=name))
                if "net_10bp_pct" not in trades:
                    trades["net_10bp_pct"] = np.nan
            if start == JAN:
                periods = [("h1", "2026-06-30"), ("full", END)]
            elif start == JUL:
                periods = [("h2", END)]
                if window == "RTH" and gap == "open":
                    periods.append(("jul_aug", "2026-08-31"))
                    prior = 100.
                    for month, part in equity.groupby(equity.day.str[:7], sort=True):
                        nav = float(part.iloc[-1].nav_10bp)
                        monthly.append(dict(filter=name, month=month, net10=100*(nav/prior-1),
                                            closed=int(trades.exit_day.astype("string").str.startswith(month).sum())))
                        prior = nav
            else:
                periods = [("sep_on", END)]
            for period, end in periods:
                metrics.append(dict(filter=name, period=period, window=window, gap_tp=gap,
                                    **period_stats(trades, equity, flags, end)))
    results = pd.DataFrame(metrics)
    main = results[results.window.eq("RTH") & results.gap_tp.eq("open")]
    h2_prices = close.loc[JUL:END]
    summary = dict(h1_pick=choose(main, "h1", 20), summer_pick=choose(main, "jul_aug", 10),
                   buy_hold_h2=float((h2_prices.iloc[-1]/(h2_prices.iloc[0]*1.001)-1)*100),
                   h2_sessions=len(h2_prices), last_session=END)
    pd.concat(trade_books, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    pd.concat(equity_books, ignore_index=True).to_csv(OUT / "equity.csv", index=False)
    results.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / "monthly.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(main.pivot(index="filter", columns="period", values="net10").round(2).to_string())
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--buffer", action="store_true", help="run the separate five-minute submission buffer check")
    checks.add_argument("--components", action="store_true", help="compare the registered signal to its price component")
    args = parser.parse_args()
    if args.buffer:
        buffer_check()
    elif args.components:
        compare_belt_components()
    else:
        main()
