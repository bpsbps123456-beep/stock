"""BRIEF8 frozen premarket SOXL-gap -> SOXS experiment; no parameter selection.

Run from any directory: py .../code/premarket_fade.py
All returns are percentage points. No historical performance before 2025 is evaluated.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("premarket_fade_shared_lib", HERE / "lib.py")
LIB = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LIB
SPEC.loader.exec_module(LIB)
SEED, NBOOT = 20261104, 10000
THRESHOLDS, EXITS = (5, 3, 8), ("09:59", "09:45")
SIGNAL_BAR, ENTRY_BAR = LIB.idx("08:59"), LIB.idx("09:00")
OHLC = ["Open", "High", "Low", "Close"]


class FixedWindow:
    """Already-eligible day: observe 08:59, fill next close, exit at actual ET."""
    start = "08:58"

    def __init__(self, pos=-1, exit_actual="09:59"):
        self.pos = pos
        self.exit_decision = LIB.idx(exit_actual) - 2

    def on_bar(self, ctx, i):
        if i == SIGNAL_BAR:
            return LIB.Order(self.pos)
        if i == self.exit_decision:
            return LIB.Order(0)
        return None


def merge_cache(cache_root=None):
    """Use identical overlapping adjusted OHLC; later file wins only after check."""
    cache = Path(cache_root) if cache_root else ROOT / "cache"
    out = cache / "alp1m_premarket_fade"
    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    for sym in LIB.SYMS:
        a = pd.read_parquet(cache / "alp1m" / f"{sym}.parquet")
        b = pd.read_parquet(cache / "alp1m_recent" / f"{sym}.parquet")
        for frame in (a, b):
            if str(frame.index.tz) != "America/New_York":
                raise ValueError(f"{sym}: timezone must be America/New_York")
            if frame.index.has_duplicates:
                raise ValueError(f"{sym}: duplicate raw bars")
        common = a.index.intersection(b.index)
        aa, bb = a.loc[common, OHLC].to_numpy(), b.loc[common, OHLC].to_numpy()
        if len(common) == 0 or not np.allclose(aa, bb, rtol=1e-10, atol=1e-10, equal_nan=True):
            raise ValueError(f"{sym}: split adjustment/overlap mismatch")
        rel = float(np.nanmax(np.abs(aa / bb - 1)))
        combined = pd.concat([a, b]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(out / f"{sym}.parquet")
        checks[sym] = {"overlap_bars": len(common), "max_ohlc_rel_diff": rel,
                       "first": str(combined.index[0]), "last": str(combined.index[-1])}
    return out, checks


def valid_bar(data, sym, d, i):
    return bool(data.has[sym][d, i] and data.A[sym]["Volume"][d, i] > 0
                and np.isfinite(data.A[sym]["Close"][d, i]) and data.A[sym]["Close"][d, i] > 0)


def gap_for_day(data, d):
    """Only prior regular-session close and the completed 08:59 signal bar."""
    if d <= 0 or not valid_bar(data, "SOXL", d, SIGNAL_BAR):
        return np.nan
    last = int(data.rth_end[d - 1]) - 1
    if not valid_bar(data, "SOXL", d - 1, last):
        return np.nan
    return float((data.A["SOXL"]["Close"][d, SIGNAL_BAR]
                  / data.A["SOXL"]["Close"][d - 1, last] - 1) * 100)


def timestamp(day, actual):
    return pd.Timestamp(f"{day} {actual}", tz="America/New_York")


def max_dd(r):
    equity = np.r_[1.0, np.cumprod(1 + np.asarray(r) / 100)]
    return float((equity / np.maximum.accumulate(equity) - 1).min() * 100)


def bootstrap(r, nboot=NBOOT, seed=SEED):
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return [None, None], None
    rng = np.random.default_rng(seed)
    draws = r[rng.integers(0, len(r), (nboot, len(r)))].mean(1)
    return np.percentile(draws, [2.5, 97.5]).tolist(), float((draws <= 0).mean())


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return adjusted.tolist()


def summarize(days, r, c2, signal, mirror, all_short, all_long, null=None):
    r, c2, signal = np.asarray(r), np.asarray(c2), np.asarray(signal, dtype=bool)
    ci, tail = bootstrap(r)
    traded = r[signal]
    keep = np.ones(len(r), dtype=bool)
    top = np.argsort(-r, kind="stable")[:min(2, len(r))]
    keep[top] = False
    monthly = {}
    for month in pd.DatetimeIndex(days).strftime("%Y-%m").unique():
        mask = pd.DatetimeIndex(days).strftime("%Y-%m") == month
        monthly[month] = {"signals": int(signal[mask].sum()), "per_day_pct": float(r[mask].mean())}
    result = {"days": len(r), "signals": int(signal.sum()), "per_day_pct": float(r.mean()),
              "per_trade_pct": float(traded.mean()) if len(traded) else None,
              "median_trade_pct": float(np.median(traded)) if len(traded) else None,
              "win_rate": float((traded > 0).mean()) if len(traded) else None,
              "worst_trade_pct": float(traded.min()) if len(traded) else None,
              "ci95_per_day_pct": ci, "bootstrap_tail_le_zero_reference": tail,
              "cost2_per_day_pct": float(c2.mean()),
              "drop_top2_per_day_pct": float(r[keep].mean()) if keep.any() else None,
              "removed_top2": [{"day": str(pd.Timestamp(days[i]).date()), "return_pct": float(r[i])} for i in top],
              "compound_pct": float((np.prod(1 + r / 100) - 1) * 100), "max_dd_pct": max_dd(r),
              "signal_mirror_SOXL_per_day_pct": float(np.asarray(mirror).mean()),
              "everyday_SOXS_per_day_pct": float(np.asarray(all_short).mean()),
              "everyday_SOXL_per_day_pct": float(np.asarray(all_long).mean()), "monthly": monthly}
    if null is not None:
        means = np.asarray(null).mean(1)
        result["month_matched_null"] = {"draws": len(means), "mean_per_day_pct": float(means.mean()),
                                             "ci95_per_day_pct": np.percentile(means, [2.5, 97.5]).tolist(),
                                             "observed_minus_null_mean_pct": float(r.mean() - means.mean()),
                                             "upper_tail_reference": float((1 + (means >= r.mean()).sum()) / (len(means) + 1))}
    return result


def month_matched_null(days, signal, baseline, eligible, nboot=NBOOT, seed=SEED):
    """Draw the same count of unique valid sessions per calendar month."""
    days = pd.DatetimeIndex(days)
    months = days.strftime("%Y-%m")
    signal, baseline, eligible = map(np.asarray, (signal, baseline, eligible))
    out = np.zeros((nboot, len(days)), dtype=float)
    rng = np.random.default_rng(seed)
    for month in months.unique():
        sel = np.flatnonzero((months == month) & eligible)
        count = int(signal[months == month].sum())
        if count > len(sel):
            raise ValueError("monthly signal count exceeds eligible sessions")
        if count:
            draws = np.argsort(rng.random((nboot, len(sel))), axis=1)[:, :count]
            chosen = sel[draws]
            out[np.arange(nboot)[:, None], chosen] = baseline[chosen]
    return out


def runs(data, indices, exit_actual, pos, cost_mult):
    engine = LIB.Engine(data, scheme="std", cost_mult=cost_mult)
    exit_bar = LIB.idx(exit_actual) - 1
    returns = np.zeros(len(indices))
    eligible = np.zeros(len(indices), dtype=bool)
    trades = {}
    sym = LIB.INST[pos]
    for k, d in enumerate(indices):
        if not (valid_bar(data, sym, d, ENTRY_BAR) and valid_bar(data, sym, d, exit_bar)):
            continue
        ret, trade = engine.run_day(FixedWindow(pos, exit_actual), d)
        if len(trade) != 1:
            raise AssertionError((data.days[d], pos, exit_actual, trade))
        row = trade[0]
        if row[2] != "09:00" or row[3] != LIB.hm(exit_bar) or row[-1] != "signal":
            raise AssertionError(row)
        returns[k], eligible[k], trades[k] = ret * 100, True, row
    return returns, eligible, trades


def main():
    source, checks = merge_cache()
    data = LIB.Data(src=source.name)
    indices = [i for i, day in enumerate(data.days) if pd.Timestamp("2025-01-02").date() <= day <= pd.Timestamp("2026-11-03").date()]
    dates = pd.DatetimeIndex([data.days[i] for i in indices])
    gaps = np.array([gap_for_day(data, i) for i in indices])
    result = {"brief": "BRIEF8.md", "code_sha256_16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
              "seed": SEED, "bootstrap_draws": NBOOT, "units": "percentage points (1.0 = 1%)",
              "information_time": "08:59 bar closes at09:00; entry09:00bar close=09:01actual; exit09:58bar close=09:59actual",
              "history_status": "2026 reused historical review, not a new independent holdout; 2025 auxiliary only",
              "overlap_checks": checks, "rows": [], "data_checks": {"days_2026": int((dates.year == 2026).sum()),
              "days_2025": int((dates.year == 2025).sum()), "signal_gap_missing": int((~np.isfinite(gaps)).sum())}}
    daily_rows, trade_rows = [], []
    periods = {"2026_all": ("2026-01-02", "2026-11-03"), "2026_front": ("2026-01-02", "2026-06-30"),
               "2026_back": ("2026-07-01", "2026-11-03"), "2026_jul_aug": ("2026-07-01", "2026-08-31"),
               "2026_sep_nov": ("2026-09-01", "2026-11-03"), "2025_auxiliary": ("2025-01-02", "2025-12-31")}
    for exit_actual in EXITS:
        print("fixed exit", exit_actual, flush=True)
        short, eligible, trades = runs(data, indices, exit_actual, -1, 1.0)
        short2, eligible2, _ = runs(data, indices, exit_actual, -1, 2.0)
        long, eligible_long, _ = runs(data, indices, exit_actual, 1, 1.0)
        assert np.array_equal(eligible, eligible2)
        result["data_checks"][exit_actual] = {"SOXS_invalid_execution_days": [str(d.date()) for d in dates[~eligible]],
                                                     "SOXL_invalid_execution_days": [str(d.date()) for d in dates[~eligible_long]]}
        common_eligible = eligible & eligible_long & np.isfinite(gaps)
        short = np.where(common_eligible, short, 0)
        short2 = np.where(common_eligible, short2, 0)
        long = np.where(common_eligible, long, 0)
        for threshold in THRESHOLDS:
            sid = f"g{threshold}_exit{exit_actual.replace(':', '')}"
            raw_signal = np.isfinite(gaps) & (gaps >= threshold - 1e-10)
            signal = raw_signal & common_eligible
            ret, c2, mirror = np.where(signal, short, 0), np.where(signal, short2, 0), np.where(signal, long, 0)
            null = month_matched_null(dates, signal, short, common_eligible)
            row = {"strategy_id": sid, "threshold_pct": threshold, "exit_actual_et": exit_actual,
                   "primary": threshold == 5 and exit_actual == "09:59", "periods": {}}
            for period, (start, end) in periods.items():
                mask = (dates >= start) & (dates <= end)
                row["periods"][period] = summarize(dates[mask], ret[mask], c2[mask], signal[mask], mirror[mask],
                                                      short[mask], long[mask], null[:, mask] if period.startswith("2026") else None)
            row["raw_signals_missing_execution"] = [str(d.date()) for d in dates[raw_signal & ~common_eligible]]
            result["rows"].append(row)
            for k, day in enumerate(dates):
                daily_rows.append({"strategy_id": sid, "day": str(day.date()), "year": day.year,
                                   "threshold_pct": threshold, "exit_actual_et": exit_actual, "signal_gap_pct": gaps[k],
                                   "raw_signal": bool(raw_signal[k]), "execution_valid": bool(common_eligible[k]), "signal": bool(signal[k]),
                                   "return_pct": ret[k], "return_cost2_pct": c2[k], "mirror_return_pct": mirror[k],
                                   "everyday_SOXS_return_pct": short[k], "everyday_SOXL_return_pct": long[k]})
                if not signal[k]:
                    continue
                tr = trades[k]
                entry_bar = timestamp(day.date(), tr[2])
                exit_bar = timestamp(day.date(), tr[3])
                trade_rows.append({"strategy_id": sid, "day": str(day.date()), "year": day.year,
                                   "threshold_pct": threshold, "exit_actual_et": exit_actual, "sym": "SOXS", "signal_gap_pct": gaps[k],
                                   "signal_timestamp": timestamp(day.date(), "09:00").isoformat(),
                                   "entry_timestamp": (entry_bar + pd.Timedelta(minutes=1)).isoformat(),
                                   "exit_timestamp": (exit_bar + pd.Timedelta(minutes=1)).isoformat(),
                                   "entry_bar_timestamp": entry_bar.isoformat(), "exit_bar_timestamp": exit_bar.isoformat(),
                                   "entry_bar_close": float(data.A["SOXS"]["Close"][indices[k], ENTRY_BAR]),
                                   "exit_bar_close": float(data.A["SOXS"]["Close"][indices[k], LIB.idx(exit_actual) - 1]),
                                   "entry_px": tr[4], "exit_px": tr[5], "return_pct": ret[k], "return_cost2_pct": c2[k],
                                   "mirror_return_pct": mirror[k]})
    adjusted = holm([r["periods"]["2026_back"]["bootstrap_tail_le_zero_reference"] for r in result["rows"]])
    for row, adj in zip(result["rows"], adjusted):
        row["periods"]["2026_back"]["holm6_bootstrap_tail_reference"] = adj
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "premarket_fade.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(daily_rows).to_csv(out / "premarket_fade_daily.csv", index=False, encoding="utf-8")
    pd.DataFrame(trade_rows).to_csv(out / "premarket_fade_trades.csv", index=False, encoding="utf-8")
    print("wrote", out / "premarket_fade.json", "trades", len(trade_rows), flush=True)


if __name__ == "__main__":
    main()
