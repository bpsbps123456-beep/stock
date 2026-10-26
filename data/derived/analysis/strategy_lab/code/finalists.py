"""Finalist rules under every execution variant, plus the H6e profile and path counts.

Execution variants reflect the user's order window (08:00-23:00 KST): the US close
is not reachable, so open execution and 1-3 decision-day delays are the realistic
cases; 30bp one-way stands in for pre-market spreads.
"""

import numpy as np

import common as C
import data as D
import engine as E
import grid as G

d = D.load()
FIN = {"REPL_T TIME 미국 전량": ("T", "W", 500, 0, 0, 1),
       "H1b TIME 비중 Top5·매일": ("T", "W", 5, 0, 5, 1),
       "H6 TIME 모멘텀120 Top5·매일·펀드비중": ("T", "MOM120", 5, 0, 5, 1),
       "H6e TIME 모멘텀120 Top5·주1회·동일": ("T", "MOM120", 5, 1, 5, 0),
       "H6f TIME 모멘텀120 Top10·주1회·동일": ("T", "MOM120", 10, 1, 10, 0),
       "H7 공통보유 위험조정모멘텀 Top5·매일": ("I", "RADJ120", 5, 0, 5, 1),
       "H10 KoAct 단기반전 Top5·매일": ("K", "REV5", 5, 0, 5, 1)}


def param(names, spec):
    s, sc, N, rb, buf, wt = spec
    return [names.index((s, "all", sc)), N, wt, buf, rb, 0, E.SOURCES.index(s)]


print("== Finalists by universe / execution (cost 0.10%) : total (MDD, Sharpe) ==")
for label, kw in (("전체", {}), ("SNDK 제외", {"exclude": ("SNDK",)}),
                  ("메모리5 제외", {"exclude": ("SNDK", "WDC", "STX", "MU", "SKHY")}),
                  ("시가 체결", {"exec_mode": "open"})):
    m = E.build_market(d, **kw)
    names, stack = E.build_scores(m)
    print(f"-- {label}")
    for k, spec in FIN.items():
        st = C.stats(C.simulate(m, stack, param(names, spec))[0])
        print(f"   {k:34s} {st['total'] * 100:+6.0f}%  (MDD {st['mdd'] * 100:4.0f}%, Sharpe {st['sharpe']:.2f})")

print("\n== User-executable variants: cost 30bp, open/close x extra decision-day delay ==")
print("execution | " + " | ".join(k.split()[0] for k in FIN))
for ex in ("close", "open"):
    for delay in (0, 1, 2, 3):
        m = E.build_market(d, exec_mode=ex)
        names, stack = E.build_scores(m)
        stack = E.apply_delay(m, stack, delay)
        vals = []
        for spec in FIN.values():
            st = C.stats(C.simulate(m, stack, param(names, spec), cost=0.003)[0])
            vals.append(f"{st['total'] * 100:+5.0f}%({st['mdd'] * 100:3.0f})")
        print(f"{ex:5s} +{delay}d  | " + " | ".join(vals))

print("\n== Trade frequency (cost 0.10%, close) ==")
m = E.build_market(d)
names, stack = E.build_scores(m)
ctx = G.Context(m)
weeks = len(m.days) / 5
for k, spec in FIN.items():
    if k.startswith("REPL"):
        continue
    W, g, tu = C.weights_path(m, stack, param(names, spec))
    held = W > 0
    entries = int((held[1:] & ~held[:-1]).sum() + held[0].sum())
    spells = []
    for j in range(W.shape[1]):
        run = 0
        for t in range(W.shape[0]):
            if held[t, j]:
                run += 1
            elif run:
                spells.append(run)
                run = 0
        if run:
            spells.append(run)
    print(f"   {k:34s} new buys {entries:3d} ({entries / weeks:.1f}/week), avg hold {np.mean(spells):5.1f} days, "
          f"turnover {tu.sum():5.1f}x, names {int(held.any(axis=0).sum())}")

print("\n== H6e profile ==")
p = param(names, FIN["H6e TIME 모멘텀120 Top5·주1회·동일"])
krw = C.simulate(m, stack, p)[0]
ref = C.simulate(m, stack, param(names, FIN["REPL_T TIME 미국 전량"]))[0]
st = C.stats(krw)
eq = np.cumprod(1 + krw)
dd = eq / np.maximum.accumulate(np.maximum(eq, 1)) - 1
t_mdd = int(np.argmin(dd))
t_pk = int(np.argmax(eq[:t_mdd + 1]))
print(f"total {st['total'] * 100:+.1f}%  MDD {st['mdd'] * 100:.1f}% ({m.days[t_pk]} -> {m.days[t_mdd]})  "
      f"Sharpe {st['sharpe']:.2f}  vol {st['vol'] * 100:.0f}%  (REPL_T vol {C.stats(ref)['vol'] * 100:.0f}%)")
mo = [np.expm1(np.log1p(krw)[ctx.month_idx == i].sum()) * 100 for i in range(9)]
mr = [np.expm1(np.log1p(ref)[ctx.month_idx == i].sum()) * 100 for i in range(9)]
print("month   " + " ".join(f"{x[5:]:>6}" for x in ctx.months))
print("H6e     " + " ".join(f"{x:+6.1f}" for x in mo))
print("REPL_T  " + " ".join(f"{x:+6.1f}" for x in mr))
print("excess  " + " ".join(f"{a - b:+6.1f}" for a, b in zip(mo, mr)) + f"   beat {sum(a > b for a, b in zip(mo, mr))}/9")
W, _, _ = C.weights_path(m, stack, p)
contrib = C.contributions(m, W)
print("top contributions (USD, %p of start): "
      + ", ".join(f"{k} {v * 100:+.0f}" for k, v in list(contrib.items())[:6]))
print("month-start holdings (historical simulation, equal weight):")
seen = set()
for t, day in enumerate(m.days):
    if day[:7] in seen:
        continue
    seen.add(day[:7])
    js = np.where(W[t] > 0)[0]
    print("  ", day, ", ".join(m.tickers[j] for j in js))

M, P, meta = C.load_grid()
X = np.asarray(M, dtype=np.float64).reshape(-1, M.shape[2])
cols = [C.col(meta, f"m{i}") for i in range(1, 10)] + [C.col(meta, f"bs{i}") for i in range(10)]
base = np.asarray(M[:, 0, :], dtype=np.float64)
print(f"\ndistinct return paths: {np.unique(np.round(X[:, cols], 6), axis=0).shape[0]:,} of {X.shape[0]:,} variants; "
      f"base rules {np.unique(np.round(base[:, cols], 6), axis=0).shape[0]:,} of {base.shape[0]:,}")
