"""패닉 매수 단독 모의매매: 신호 다음 봉 시가 매수 → 목표가(지정가) 또는 N봉 뒤 시가 정리. 깨어 있는 프리장·정규장만(야간 데이터 없음)."""
import itertools
import numpy as np, pandas as pd
from numba import njit
from core import *
from evalcfg import data
from osc import rsi
from run_v5 import eth_native
from ev_panic import series

D, H = data()


@njit(cache=True)
def sim_panic(o, h, l, c, sig, ok, awake, tgt, nmax, close_fill):
    n = len(o)
    eq = np.empty(n)
    cash, sh = 1.0, 0.0
    held, entry, bars = False, 0.0, 0
    ntr, nwin = 0, 0
    for j in range(n):
        if held and ok[j]:
            bars += 1
            hit = (c[j] >= entry * (1 + tgt)) if close_fill else (h[j] >= entry * (1 + tgt))
            if bars > 1 and hit:
                px = c[j] if close_fill else max(o[j], entry * (1 + tgt))
                cash = sh * px; sh = 0.0; held = False; ntr += 1; nwin += 1 if px > entry else 0
            elif bars > nmax and awake[j]:
                cash = sh * o[j]; nwin += 1 if o[j] > entry else 0; sh = 0.0; held = False; ntr += 1
        if (not held) and j > 0 and sig[j - 1] and ok[j] and awake[j]:
            sh = cash / o[j]; cash = 0.0; held = True; entry = o[j]; bars = 1
            hit = (c[j] >= entry * (1 + tgt)) if close_fill else (h[j] >= entry * (1 + tgt))
        eq[j] = cash + sh * c[j]
    return eq, ntr, nwin


def run(kind, years, close_fill):
    out = []
    SIG = {}
    DATA = {}
    for s in SYMS:
        tb, et = series(s, kind)
        o, h, l, c = tb.O.values, tb.H.values, tb.L.values, tb.C.values
        m = np.asarray(et.hour * 60 + et.minute)        # 봉 끝 시각(ET)
        start = m - (60 if kind == "60연장" else 40)
        kst_h = np.asarray(et.tz_convert("Asia/Seoul").hour)
        awake = kst_h >= 8
        pre = (start >= 240) & (start < 570)
        reg = (start >= 570) & (start < 960)
        ok = reg | (pre & awake)
        r = rsi(c, 14)
        d3 = r - np.r_[np.nan, np.nan, np.nan, r[:-3]]
        s1 = np.nan_to_num(((d3 <= -25) & (r < 30)).astype(float)).astype(bool)
        s2 = np.nan_to_num((r < 25).astype(float)).astype(bool)
        SIG[(s, "3봉 RSI -25 & <30")] = s1 & ~np.r_[False, s1[:-1]]
        SIG[(s, "RSI<25")] = s2 & ~np.r_[False, s2[:-1]]
        yr = np.asarray(et.year.isin(years))
        DATA[s] = (o, h, l, c, ok, yr, et, awake)
    for name, tgt, nmax in itertools.product(["RSI<25"], [0.005, 0.01, 0.02], [3, 6]):
        eqs, nt, nw = [], 0, 0
        for s in SYMS:
            o, h, l, c, ok, yr, et, awake = DATA[s]
            i0 = np.flatnonzero(yr)[0]; i1 = np.flatnonzero(yr)[-1] + 1
            e, t, w = sim_panic(o[i0:i1], h[i0:i1], l[i0:i1], c[i0:i1], SIG[(s, name)][i0:i1], ok[i0:i1], awake[i0:i1], tgt, nmax, close_fill)
            d = pd.Series(e, index=et[i0:i1].date).groupby(level=0).last()
            eqs.append(d); nt += t; nw += w
        port = pd.concat(eqs, axis=1).ffill().fillna(1.0).mean(axis=1)
        r = port.pct_change().dropna()
        mdd = (port / port.cummax() - 1).min()
        out.append({"신호": name, "목표": f"{tgt*100:g}%", "최대보유": f"{nmax}봉", "수익": f"{port.iloc[-1]-1:+.0%}",
                    "샤프": round(r.mean() / r.std() * np.sqrt(252), 2), "낙폭": f"{mdd:.0%}",
                    "거래": nt, "승률": f"{nw / max(nt, 1):.0%}"})
    return pd.DataFrame(out)


pd.set_option("display.width", 200)
for kind, years, lab in (("60연장", [2026], "2026 (60분)"), ("60연장", [2025], "2025 (60분)"), ("40연장", [2026], "최근 60일 (40분)")):
    a = run(kind, years, False); b = run(kind, years, True)
    a["종가체결 수익"] = b["수익"].values
    print(f"\n===== {lab}: 패닉 매수만 (7종목 동일 비중, 수수료 0%) =====")
    print(a.to_string(index=False))
