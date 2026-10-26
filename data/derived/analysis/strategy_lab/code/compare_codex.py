"""Head-to-head: Codex zero-cost US-only candidates vs this lab's momentum rules, 2026.

Codex paths are regenerated with Codex's own engine (read-only import) and checked
against its saved grid. Timing: Codex 'base' uses the snapshot strictly before the
trade date and fills at the close, i.e. this lab's close +1 row; Codex 'lag1' ~ +2.
'open' rows execute the same decisions at the US open of the decision day, which
fits the user's 08:00-23:00 KST order window better than a close fill.
"""

import json
import sys

import numpy as np
import pandas as pd

import common as C
import data as D
import engine as E

sys.path.insert(0, str(D.REPO))
from gbrc.etf.research_engine import build_rankings, config_grid, simulate_detail  # noqa: E402

SRC = D.REPO / "data" / "derived" / "analysis" / "2026-09-23_strategy_research" / "us_only_experiment"
CODEX = {"239120": "Codex 주 후보(신규편입 1종목·매일)", "240024": "Codex 강건 선택(반복매수 2종목·주1회)",
         "201609": "Codex 3종목(실매도 역발상·매일)", "162139": "Codex 5종목 강건(동반매수·월1회)",
         "246798": "Codex 5종목 기본(동반매수·주1회·추세현금)"}
MINE = {"H6e 모멘텀 Top5·주1회·동일": ("T", "MOM120", 5, 1, 5, 0),
        "H6 모멘텀 Top5·매일·펀드비중": ("T", "MOM120", 5, 0, 5, 1),
        "TIME 비중 Top5·주1회": ("T", "W", 5, 1, 5, 1),
        "TIME 복제·매일": ("T", "W", 500, 0, 0, 1)}
SEGMENTS = {"1~5월": ("2026-01-01", "2026-05-31"), "6~7월": ("2026-06-01", "2026-07-31"),
            "8~9월": ("2026-08-01", "2026-09-22")}
END = "2026-09-22"


def metrics(dates, r):
    """Total/MDD/vol plus segment returns over dates <= END (common window)."""
    dates = np.asarray(dates)
    keep = dates <= END
    dates, r = dates[keep], np.asarray(r)[keep]
    eq = np.cumprod(1 + r)
    mdd = (eq / np.maximum.accumulate(np.maximum(eq, 1.0)) - 1).min()
    seg = {k: np.prod(1 + r[(dates >= a) & (dates <= b)]) - 1 for k, (a, b) in SEGMENTS.items()}
    return {"total": eq[-1] - 1, "mdd": mdd, "vol": r[1:].std() * np.sqrt(252), **seg}


def codex_paths():
    arrays = dict(np.load(SRC / "bundle.npz"))
    raw = json.loads((SRC / "raw_prices.json").read_text(encoding="utf-8"))
    dates = arrays["dates"].astype(str)
    arrays["qqq_usd"] = np.asarray([raw["QQQ"][str(day)] for day in dates])
    saved = {"base": np.load(SRC / "zero_cost" / "grid_returns.npy", mmap_mode="r"),
             "lag1": np.load(SRC / "zero_cost" / "delayed_grid_returns.npy", mmap_mode="r")}
    out = {}
    for name, lag in (("base", 0), ("lag1", 1)):
        ranking = build_rankings(arrays, extra_lag=lag)
        params, _ = config_grid(ranking["rank_configs"])
        for cid in CODEX:
            det = simulate_detail(params[int(cid)], arrays, ranking, cost=0.0)
            assert np.allclose(det["returns"], saved[name][int(cid)], atol=1e-6), f"{cid} {name} mismatch"
            out[(cid, name)] = det
    return dates, arrays["symbols"].astype(str), out


def open_returns_for(allocs_by_day, dates, symbols, d):
    """KRW open-to-open path when each day's decision is executed at that day's US open."""
    op = d["prices"]["Open"]
    cl = d["prices"]["Close"]
    adj = d["prices"]["Adj Close"]
    idx = pd.to_datetime(dates)
    alias = {"GOOG": "GOOGL", "SATS": "ECHO"}  # this lab prices merged classes under one symbol
    cols = [alias.get(s, s).replace("/", "-").replace(".", "-") for s in symbols]
    missing = sorted({c for c in cols if c not in adj.columns})
    if missing:
        print("  no open price in lab cache (held weight gets 0 return):", missing)
    o = (op.reindex(idx, columns=cols) * (adj.reindex(idx, columns=cols) / cl.reindex(idx, columns=cols))).to_numpy(float)
    fx = adj["KRW=X"].ffill().reindex(idx).ffill().to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ro = np.nan_to_num(o[1:] / o[:-1] - 1, nan=0.0)
    r = np.zeros(len(dates))
    for t in range(1, len(dates)):
        a = allocs_by_day[t - 1]
        r[t] = (1 + float(a @ ro[t - 1])) * (fx[t] / fx[t - 1]) - 1
    return r


def main():
    d = D.load()
    dates, symbols, cx = codex_paths()
    print(f"Codex engine reproduced saved grid paths for {len(CODEX)} candidates x 2 timings (|err|<=1e-6)")
    rows = {}
    for cid, label in CODEX.items():
        for name in ("base", "lag1"):
            rows[(label, "종가 · Codex " + name)] = metrics(dates, cx[(cid, name)]["returns"])
        det = cx[(cid, "base")]
        alloc = np.zeros((len(dates), len(symbols)))
        for t in range(len(dates)):
            for slot in range(det["holdings"].shape[1]):
                j = int(det["holdings"][t, slot])
                if j >= 0:
                    alloc[t, j] += float(det["allocations"][t, slot])
        rows[(label, "시가 · Codex base 결정")] = metrics(dates, open_returns_for(alloc, dates, symbols, d))
        if cid == "239120":
            held = [(dates[t], symbols[int(det["holdings"][t, 0])] if det["holdings"][t, 0] >= 0 else "현금")
                    for t in range(len(dates))]
            runs, prev, start, last = [], None, None, None
            for day, s in held:
                if s != prev:
                    if prev is not None:
                        runs.append((start, last, prev))
                    prev, start = s, day
                last = day
            runs.append((start, last, prev))
            contrib = {}
            for t in range(1, len(dates)):
                j = int(det["holdings"][t - 1, 0])
                if j >= 0:
                    contrib[symbols[j]] = contrib.get(symbols[j], 0.0) + float(det["returns"][t])
            print("\n239120 holding runs (base, 0원):")
            for a, b, s in runs:
                print(f"  {a} ~ {b}  {s}")
            print("  switches:", len(runs) - 1, "| cash days:", sum(1 for _, s in held if s == "현금"),
                  "| sum of daily returns by name:",
                  ", ".join(f"{k} {v * 100:+.0f}%p" for k, v in sorted(contrib.items(), key=lambda x: -x[1])[:8]))

    for delay, lab in ((1, "종가 · +1행(≈Codex base)"), (2, "종가 · +2행(≈Codex lag1)")):
        m = E.build_market(d)
        names, stack = E.build_scores(m)
        stack = E.apply_delay(m, stack, delay)
        for k, (s, sc, N, rb, buf, wt) in MINE.items():
            r = C.simulate(m, stack, [names.index((s, "all", sc)), N, wt, buf, rb, 0, E.SOURCES.index(s)], cost=0.0)[0]
            rows[(k, lab)] = metrics(m.days, r)
    m = E.build_market(d, exec_mode="open")
    names, stack = E.build_scores(m)
    stack = E.apply_delay(m, stack, 1)
    for k, (s, sc, N, rb, buf, wt) in MINE.items():
        r = C.simulate(m, stack, [names.index((s, "all", sc)), N, wt, buf, rb, 0, E.SOURCES.index(s)], cost=0.0)[0]
        rows[(k, "시가 · +1행")] = metrics(m.days, r)

    etf = 0.684
    print("\n== 2026-01-02 ~ 09-22, 수수료 0원 (TIME ETF +68.4%, 6~7월 -18.4%, 8~9월 +5.0%) ==")
    print("strategy                              | execution                 | total   xETF |  MDD  | vol | 1~5월   6~7월   8~9월")
    for (k, lab), v in rows.items():
        print(f"{k:37s} | {lab:25s} | {v['total'] * 100:+6.0f}% {v['total'] / etf:4.1f}x | {v['mdd'] * 100:4.0f}% | {v['vol'] * 100:3.0f}% |"
              f" {v['1~5월'] * 100:+6.0f}% {v['6~7월'] * 100:+6.0f}% {v['8~9월'] * 100:+6.0f}%")


if __name__ == "__main__":
    main()
