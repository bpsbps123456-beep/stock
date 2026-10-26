"""과매도(분할매수 후보)·과매수(분할매도 후보) 지표 비교: 신호 뒤 초과수익과 t값, 세 창(40분 7~9월, 60분 2026, 60분 2024~25)."""
import numpy as np
import pandas as pd
from numba import njit

from core import *
from evalcfg import data
from osc import rsi, stoch_rsi, forward

D, HALF = data()


@njit(cache=True)
def streak(c):
    m = len(c)
    out = np.zeros(m)
    for i in range(1, m):
        if c[i] > c[i - 1]:
            out[i] = out[i - 1] + 1 if out[i - 1] > 0 else 1
        elif c[i] < c[i - 1]:
            out[i] = out[i - 1] - 1 if out[i - 1] < 0 else -1
        else:
            out[i] = 0
    return out


@njit(cache=True)
def pct_rank(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    for i in range(n, m):
        if np.isnan(x[i]):
            continue
        cnt, ok = 0, True
        for j in range(i - n, i):
            if np.isnan(x[j]):
                ok = False
                break
            if x[j] <= x[i]:
                cnt += 1
        if ok:
            out[i] = 100.0 * cnt / n
    return out


def connors(c):
    roc = np.r_[np.nan, c[1:] / c[:-1] - 1]
    return (rsi(c, 3) + rsi(streak(c), 2) + pct_rank(roc, 100)) / 3


def fast_z(c, bpd):
    L = max(3, bpd)
    K = kalman(c, gain_to_q(length_to_gain(L)), 3.0)
    return (c - K) / rms(c - K, max(10, 5 * bpd))


def signals(c, bpd):
    r14, r2 = rsi(c, 14), rsi(c, 2)
    k, _ = stoch_rsi(r14, 14, 3, 3)
    cr, fz = connors(c), fast_z(c, bpd)
    return {
        "매수후보 RSI14<30 & 스토RSI<30 (가설)": (r14 < 30) & (k < 30),
        "매수후보 RSI14<30": r14 < 30,
        "매수후보 스토RSI<20": k < 20,
        "매수후보 RSI2<10": r2 < 10,
        "매수후보 ConnorsRSI<15": cr < 15,
        "매수후보 짧은칼만 편차<-2": fz < -2,
        "매도후보 RSI14>70 & 스토RSI>70": (r14 > 70) & (k > 70),
        "매도후보 스토RSI>80": k > 80,
        "매도후보 RSI2>90": r2 > 90,
        "매도후보 ConnorsRSI>85": cr > 85,
        "매도후보 짧은칼만 편차>+2": fz > 2,
    }


def study(tf, eth, src, years, H=(3, 6)):
    acc = {}
    for s in SYMS:
        tb = tf_bars(D[(s, src)], tf, eth, HALF)
        c, h, l = tb.C.values, tb.H.values, tb.L.values
        yr = pd.to_datetime(tb.end_ns.values, utc=True).year.isin(years)
        fw = {x: forward(c, h, l, x)[0] for x in H}
        bpd = int(np.ceil((960 if eth else 390) / tf))
        for name, cond in signals(c, bpd).items():
            cond = np.nan_to_num(cond.astype(float), nan=0).astype(bool)
            onset = cond & ~np.r_[False, cond[:-1]] & yr
            for x in H:
                base = fw[x][yr]
                acc.setdefault((name, x), {"ev": [], "base": []})
                acc[(name, x)]["ev"].append(fw[x][onset])
                acc[(name, x)]["base"].append(base)
    rows = []
    for (name, x), v in acc.items():
        ev = np.concatenate(v["ev"]); ev = ev[~np.isnan(ev)]
        bs = np.concatenate(v["base"]); bs = bs[~np.isnan(bs)]
        ex = ev.mean() - bs.mean()
        t = ex / (ev.std(ddof=1) / np.sqrt(len(ev))) if len(ev) > 2 else np.nan
        rows.append({"신호": name, "봉": x, "n": len(ev), "초과": ex, "t": t, "상승확률": (ev > 0).mean(), "평소상승": (bs > 0).mean()})
    return pd.DataFrame(rows)


out = []
for label, args in [("40분 7~9월", (40, False, "5m", [2026])), ("60분 2026", (60, False, "1h", [2026])),
                    ("60분 2024~25", (60, False, "1h", [2024, 2025]))]:
    t = study(*args)
    t["창"] = label
    out.append(t)
T = pd.concat(out)
T.to_csv("ev2.csv", index=False)
for x in (3, 6):
    p = T[T["봉"] == x].pivot(index="신호", columns="창", values=["초과", "t", "n"])
    p = p.reindex(columns=["40분 7~9월", "60분 2026", "60분 2024~25"], level=1)
    print(f"\n[{x}봉 뒤] 평소 대비 초과수익 (매도후보는 음수여야 좋음) / t값 / 신호 수")
    fmt = pd.DataFrame(index=p.index)
    for w in ["40분 7~9월", "60분 2026", "60분 2024~25"]:
        fmt[w] = [f"{a:+.2%} (t{b:+.1f}, {int(c)}번)" for a, b, c in zip(p[("초과", w)], p[("t", w)], p[("n", w)])]
    print(fmt.to_string())
