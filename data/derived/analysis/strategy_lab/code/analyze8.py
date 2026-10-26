"""Stage 8 (etf_system §11 stress): +1 decision-day delay, 15/30bp costs, the
Aug-Sep retrospective holdout, and a fresh cash start in August.
Multiples only when the benchmark return is positive; otherwise excess %p."""

import copy

import numpy as np

import common as C
import data as D
import engine as E
import grid as G

M0, P, meta = C.load_grid()
cols = {c: C.col(meta, c) for c in meta["metrics"]}
scen = {"기본10bp": M0,
        "15bp": np.load(C.GRID / "metrics_c15.npy", mmap_mode="r"),
        "+1일·10bp": np.load(C.GRID / "metrics_d1.npy", mmap_mode="r"),
        "+1일·30bp": np.load(C.GRID / "metrics_d1c30.npy", mmap_mode="r")}
src = np.array([meta["score_names"][s][0] for s in P[:, 0]])
pre = np.array([meta["score_names"][s][1] for s in P[:, 0]])
score = np.array([meta["score_names"][s][2] for s in P[:, 0]])
reb = np.array(meta["rebs"])[P[:, 4]]
key = np.array([f"{a}/{b}/{c}/{r}" for a, b, c, r in zip(src, pre, score, reb)])
fam = np.array([C.score_family(s) for s in score])

print("== Rule plateaus (median over N>=3 neighbours, no overlay) ==")
print("rule                | " + " | ".join(f"{s:>9s}" for s in scen) + " | delay keeps?")
rules = ["T/all/MOM120/D", "T/all/MOM120/W", "T/all/MOM120/M", "I/all/MOM120/D", "T/all/RADJ120/D",
         "I/all/RADJ120/D", "I/all/RADJ120/W", "I/all/PULL/D", "I/all/PULL/W", "K/all/PULL/D",
         "K/all/REV5/D", "U/all/REV5/D", "T/all/W/D", "T/all/W/W", "I/all/W/D", "T/all/FLOW5/D",
         "U/all/CONS20/D", "T/all/NEW10/D", "T/all/MOM20/D", "T/all/WMOM60_50/D"]
for r in rules:
    sel = (key == r) & (P[:, 1] >= 3)
    v = [np.median(np.asarray(X[sel, 0, cols["total"]])) for X in scen.values()]
    print(f"{r:19s} | " + " | ".join(f"{x * 100:+8.0f}%" for x in v) + f" | {v[2] / v[0] * 100:4.0f}% of base")
print("\n== Family medians (N>=3, no overlay, daily) ==")
for f in sorted(set(fam)):
    sel = (fam == f) & (P[:, 1] >= 3) & (reb == "D")
    v = [np.median(np.asarray(X[sel, 0, cols["total"]])) for X in scen.values()]
    print(f"{f:6s} " + " | ".join(f"{s} {x * 100:+5.0f}%" for s, x in zip(scen, v)))

# ---- finalists: per-window returns, fresh August start, under each execution scenario ----
d = D.load()
adj = d["prices"]["Adj Close"]


def etf_window(tk, start, end):
    s = adj[tk].dropna()
    s = s[(s.index >= start) & (s.index <= end)]
    return s.iloc[-1] / s.iloc[0] - 1


# KRX windows aligned to the same US information span (KRX close D reflects US close D-1)
etf = {w: {tag: etf_window(tk, a, b) for tag, tk in D.ETF_TICKERS.items()}
       for w, (a, b) in {"1~5월": ("2026-01-05", "2026-06-01"), "6~7월": ("2026-06-01", "2026-08-03"),
                         "8~9월": ("2026-08-03", "2026-09-23"), "전체": ("2026-01-05", "2026-09-23")}.items()}
print("\nETF windows (KRW): " + "; ".join(f"{w} TIME {v['T'] * 100:+.1f}% KoAct {v['K'] * 100:+.1f}%" for w, v in etf.items()))


def fmt_vs(x, bench):
    """Multiple when the benchmark is positive, else excess in %p (etf_system §11)."""
    return f"{x / bench:4.1f}x" if bench > 0 else f"{(x - bench) * 100:+5.1f}%p"


def build(delay):
    m = E.build_market(d)
    names, stack = E.build_scores(m)
    stack = E.apply_delay(m, stack, delay)
    return m, names, stack


def fresh_from(m, start_day):
    """Market sliced to start in cash at start_day (a decision day)."""
    t0 = m.days.index(start_day)
    s = copy.copy(m)
    s.days = m.days[t0:]
    s.R = m.R[t0:].copy()
    s.R[0] = 0.0
    s.fxr = m.fxr[t0:].copy()
    s.fxr[0] = 0.0
    s.dec_row = m.dec_row[t0:].copy()
    s.reb = {k: v[t0:].copy() for k, v in m.reb.items()}
    for v in s.reb.values():
        v[0] = True
    return s


finalists = {"REPL_T TIME 미국 전량": ("T", "W", 500, 0, 0), "H1b TIME 비중 Top5·매일": ("T", "W", 5, 0, 5),
             "H6 TIME 모멘텀120 Top5·매일": ("T", "MOM120", 5, 0, 5),
             "H6b TIME 모멘텀120 Top5·주1회": ("T", "MOM120", 5, 1, 5),
             "H6c TIME 모멘텀120 Top10·월1회": ("T", "MOM120", 10, 3, 10),
             "H6e TIME 모멘텀120 Top5·주1회·동일": ("T", "MOM120", 5, 1, 5, 0),
             "H7 공통 위험조정모멘텀 Top5": ("I", "RADJ120", 5, 0, 5),
             "H9c 공통 눌림목 Top5·주1회": ("I", "PULL", 5, 1, 5),
             "H10 KoAct 단기반전 Top5·매일": ("K", "REV5", 5, 0, 5)}
for delay, cost, label in ((0, 0.001, "기본 10bp"), (0, 0.0015, "15bp"), (1, 0.001, "+1일 지연·10bp"), (1, 0.003, "+1일 지연·30bp")):
    m, names, stack = build(delay)
    aug = next(x for x in m.days if x >= "2026-08-03")
    fm = fresh_from(m, aug)
    idx = {w: [i for i, x in enumerate(m.days) if a <= x <= b] for w, (a, b) in
           {"1~5월": ("2026-01-05", "2026-05-29"), "6~7월": ("2026-06-01", "2026-07-31"),
            "8~9월": ("2026-08-03", "2026-09-22")}.items()}
    print(f"\n== Finalists · {label} ==")
    print("rule                            | 전체    xTIME | MDD  | 1~5월  6~7월(vsTIME) | 8~9월 xTIME xKoAct | 8월현금출발 xTIME")
    for k, spec in finalists.items():
        s_, sc, N, rb, buf = spec[:5]
        wt = spec[5] if len(spec) > 5 else 1
        param = [names.index((s_, "all", sc)), N, wt, buf, rb, 0, E.SOURCES.index(s_)]
        krw, _, _ = C.simulate(m, stack, param, cost=cost)
        st = C.stats(krw)
        seg = {w: np.prod(1 + krw[i]) - 1 for w, i in idx.items()}
        fresh, _, _ = C.simulate(fm, stack, param, cost=cost)
        fr = np.prod(1 + fresh) - 1
        print(f"{k:31s} | {st['total'] * 100:+6.0f}% {fmt_vs(st['total'], etf['전체']['T'])} | {st['mdd'] * 100:4.0f}% | "
              f"{seg['1~5월'] * 100:+5.0f}% {seg['6~7월'] * 100:+5.0f}%({fmt_vs(seg['6~7월'], etf['6~7월']['T'])}) | "
              f"{seg['8~9월'] * 100:+5.1f}% {fmt_vs(seg['8~9월'], etf['8~9월']['T'])} {fmt_vs(seg['8~9월'], etf['8~9월']['K'])} | "
              f"{fr * 100:+5.1f}% {fmt_vs(fr, etf['8~9월']['T'])}")
