"""Frozen SOXL close-minus-30 RSI study. See ../BRIEF.md; returns in percent."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT.parent
ET = "America/New_York"
EXITS = ("16:30", "17:00", "18:00", "19:00")
TF = ("1m", "5m", "daily")
COSTS = (0.0, 0.0005, 0.001, 0.002)
END = "2026-11-03"


def shared_rma():
    spec = importlib.util.spec_from_file_location("close_rsi_shared_htf", ANALYSIS / "dip_reason/code/htf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.rma_state


def load_bars():
    base = ANALYSIS / "sox_open_switch/cache"
    paths = [base / p / "SOXL.parquet" for p in ("alp1m", "alp1m_recent")]
    frames, meta = [], {}
    for path in paths:
        f = pd.read_parquet(path).loc["2025-01-01":END].sort_index()
        assert not f.index.has_duplicates
        assert str(f.index.tz) == ET
        frames.append(f)
        meta[str(path.relative_to(ANALYSIS))] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                               "bars_used": len(f)}
    common = frames[0].index.intersection(frames[1].index)
    assert len(common) > 0
    a, b = (f.loc[common, ["Open", "High", "Low", "Close"]].to_numpy() for f in frames)
    assert np.allclose(a, b, rtol=1e-10, atol=1e-10), "overlap/split mismatch"
    meta["overlap_bars"] = len(common)
    meta["overlap_max_relative_diff"] = float(np.max(np.abs(a / b - 1)))
    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    o, h, l, c = (full[x].to_numpy() for x in ("Open", "High", "Low", "Close"))
    assert np.isfinite(np.c_[o, h, l, c]).all()
    assert (l > 0).all() and (h >= np.maximum(o, c)).all() and (l <= np.minimum(o, c)).all()
    return full, meta


def regular_bars(full):
    schedule = xc.get_calendar("XNYS").schedule.loc["2025-01-01":END]
    chunks, counts = [], []
    for day, row in schedule.iterrows():
        op, cl = row["open"].tz_convert(ET), row["close"].tz_convert(ET)
        f = full[(full.index >= op) & (full.index < cl)]
        chunks.append(f)
        counts.append({"day": str(day.date()), "bars": len(f), "expected": int((cl - op).total_seconds() / 60),
                       "close": cl.strftime("%H:%M")})
    return pd.concat(chunks), schedule, counts


def signals(full):
    reg, schedule, counts = regular_bars(full)
    rma = shared_rma()
    r1 = pd.Series(rma(reg.Close.to_numpy(), 14)[2], index=reg.index)
    f5 = reg.Close.resample("5min", closed="left", label="left").last().dropna()
    r5 = pd.Series(rma(f5.to_numpy(), 14)[2], index=f5.index)
    daily = reg.Close.groupby(reg.index.date).last()
    g, l, _ = rma(daily.to_numpy(), 14)
    rows = []
    for day, row in schedule.iterrows():
        ds = str(day.date())
        if ds < "2026-01-01":
            continue
        cl = row["close"].tz_convert(ET)
        if cl.strftime("%H:%M") != "16:00":
            continue
        decision = cl - pd.Timedelta(minutes=30)
        last = decision - pd.Timedelta(minutes=1)
        if last not in reg.index:
            continue
        j = daily.index.get_loc(day.date())
        delta = float(reg.at[last, "Close"] - daily.iloc[j - 1])
        ag, al = (g[j - 1] * 13 + max(delta, 0)) / 14, (l[j - 1] * 13 + max(-delta, 0)) / 14
        rd = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
        vals = {"1m": r1.get(last, np.nan), "5m": r5.get(decision - pd.Timedelta(minutes=5), np.nan), "daily": rd}
        for tf, val in vals.items():
            source_start = {"1m": last, "5m": decision - pd.Timedelta(minutes=5),
                            "daily": row["open"].tz_convert(ET)}[tf]
            rows.append({"day": ds, "tf": tf, "decision": decision.isoformat(),
                         "last_observed_1m": last.isoformat(), "signal_bar_start": source_start.isoformat(),
                         "rsi": float(val), "signal": bool(val <= 30)})
    return pd.DataFrame(rows), counts


def make_trades(full, sig):
    rows, missing = [], []
    for s in sig.to_dict("records"):
        for delay in (0, 1, 2):
            entry = pd.Timestamp(s["decision"]) + pd.Timedelta(minutes=delay)
            for hhmm in ("16:00", *EXITS):
                exit_ = pd.Timestamp(f'{s["day"]} {hhmm}', tz=ET)
                if entry not in full.index or exit_ not in full.index or min(full.at[entry, "Volume"], full.at[exit_, "Volume"]) <= 0:
                    missing.append({"day": s["day"], "tf": s["tf"], "delay": delay, "exit_hm": hhmm})
                    continue
                ep, xp = float(full.at[entry, "Open"]), float(full.at[exit_, "Open"])
                path = full[(full.index >= entry) & (full.index < exit_)]
                rows.append({**s, "delay": delay, "exit_hm": hhmm,
                             "entry": entry.isoformat(), "exit": exit_.isoformat(), "entry_price": ep, "exit_price": xp,
                             "gross_pct": (xp / ep - 1) * 100,
                             "mae_pct": (float(path.Low.min()) / ep - 1) * 100,
                             "mfe_pct": (float(path.High.max()) / ep - 1) * 100,
                             **{f"net_{int(c * 10000)}bp_pct": (xp * (1 - c) / (ep * (1 + c)) - 1) * 100 for c in COSTS}})
    return pd.DataFrame(rows), missing


def stats(ret):
    a = np.asarray(ret, dtype=float)
    if len(a) == 0:
        return {"n": 0}
    rng = np.random.default_rng(20261105)
    means = rng.choice(a, (10000, len(a)), replace=True).mean(axis=1)
    eq = np.r_[1, np.cumprod(1 + a / 100)]
    return {"n": len(a), "mean_pct": float(a.mean()), "median_pct": float(np.median(a)),
            "win_rate_pct": float(np.mean(a > 0) * 100), "worst_pct": float(a.min()), "best_pct": float(a.max()),
            "mean_ci95_pct": np.quantile(means, [0.025, 0.975]).tolist(),
            "compound_pct": float((eq[-1] - 1) * 100), "closed_trade_mdd_pct": float((eq / np.maximum.accumulate(eq) - 1).min() * 100)}


def summarize(trades):
    periods = {"H1": ("2026-01-01", "2026-06-30"), "H2": ("2026-07-01", END), "all": ("2026-01-01", END)}
    cells = []
    for tf in TF:
        for ex in ("16:00", *EXITS):
            for p, (a, b) in periods.items():
                all_ = trades[(trades.tf == tf) & (trades.delay == 1) & (trades.exit_hm == ex) & trades.day.between(a, b)]
                hit = all_[all_.signal]
                item = {"tf": tf, "exit_hm": ex, "period": p, "available_days": len(all_),
                        "signal_days": hit.day.tolist(), "mae_worst_pct": float(hit.mae_pct.min()) if len(hit) else None,
                        "gross": stats(hit.gross_pct), "all_day_gross": stats(all_.gross_pct),
                        "costs": {str(int(c * 10000)): stats(hit[f"net_{int(c * 10000)}bp_pct"]) for c in COSTS},
                        "all_day_net10bp": stats(all_.net_10bp_pct)}
                # Comparing with the unconditional mean is descriptive, not causal.
                if len(hit) and len(all_):
                    item["excess_vs_all_day_net10bp_pct"] = float(hit.net_10bp_pct.mean() - all_.net_10bp_pct.mean())
                cells.append(item)
    wf = []
    for tf in TF:
        eligible = [x for x in cells if x["tf"] == tf and x["period"] == "H1" and x["exit_hm"] in EXITS and x["costs"]["10"]["n"] > 0]
        if not eligible:
            wf.append({"tf": tf, "selected_exit": None, "reason": "no_H1_signal"})
            continue
        best = max(eligible, key=lambda x: x["costs"]["10"]["mean_pct"])
        test = next(x for x in cells if x["tf"] == tf and x["period"] == "H2" and x["exit_hm"] == best["exit_hm"])
        st = test["costs"]["10"]
        passed = st["n"] >= 20 and st["mean_pct"] > 0 and st["mean_ci95_pct"][0] > 0 and test.get("excess_vs_all_day_net10bp_pct", -999) > 0
        wf.append({"tf": tf, "selected_exit": best["exit_hm"], "H1": best, "H2": test, "passes_frozen_gate": bool(passed)})
    return {"cells": cells, "walk_forward": wf}


def main():
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    full, meta = load_bars()
    sig, counts = signals(full)
    trades, missing = make_trades(full, sig)
    sig.to_csv(out / "signals.csv", index=False)
    trades.to_csv(out / "trades.csv", index=False)
    # Unique quote requests: do not multiply the same day by RSI timeframe.
    hit = trades[trades.signal & trades.delay.eq(1)]
    targets = sorted(set(hit.entry) | set(hit.exit))
    (out / "quote_targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")
    result = {"end": END, "data": meta, "regular_session_counts": counts,
              "missing_trade_bars": missing, "n_signal_days": sig[sig.signal].groupby("tf").size().to_dict(),
              **summarize(trades)}
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"signal_counts": result["n_signal_days"], "unique_quote_targets": len(targets),
                      "wf": [{"tf": x["tf"], "exit": x["selected_exit"], "H2": x.get("H2", {}).get("costs", {}).get("10"),
                              "pass": x.get("passes_frozen_gate")} for x in result["walk_forward"]]}, indent=2))


if __name__ == "__main__":
    main()
