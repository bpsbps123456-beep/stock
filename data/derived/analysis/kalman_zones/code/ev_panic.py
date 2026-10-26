"""RSI 급락(패닉셀) 신호 뒤 빠른 반등이 있나: 다음 봉 시가 매수 → 1·3·6봉 뒤 종가, 평소 대비 초과·t값, 세션별."""
import numpy as np, pandas as pd
from core import *
from evalcfg import data
from osc import rsi
from run_v5 import eth_native

D, H = data()


def series(s, kind):
    if kind == "60연장":
        tb = eth_native(s)
    else:
        tb = tf_bars(D[(s, "5m")], 40, True, H)
    st = pd.to_datetime(tb.end_ns.values, utc=True).tz_convert("America/New_York")
    return tb, st


def signals(c, h, l):
    r = rsi(c, 14)
    d1 = r - np.r_[np.nan, r[:-1]]
    d2 = r - np.r_[np.nan, np.nan, r[:-2]]
    d3 = r - np.r_[np.nan, np.nan, np.nan, r[:-3]]
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr14 = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values
    raw = {"1봉 -15 & <40": (d1 <= -15) & (r < 40), "2봉 -20 & <35": (d2 <= -20) & (r < 35),
           "3봉 -25 & <30": (d3 <= -25) & (r < 30), "RSI<25 (수준)": r < 25,
           "한 봉 ATRx2 급락": (c - pc) <= -2 * np.r_[np.nan, atr14[:-1]]}
    out = {}
    for k, v in raw.items():
        v = np.nan_to_num(v.astype(float)).astype(bool)
        out[k] = v & ~np.r_[False, v[:-1]]
    return out


def study(kind, years):
    acc = {}
    for s in SYMS:
        tb, et = series(s, kind)
        o, c, h, l = tb.O.values, tb.C.values, tb.H.values, tb.L.values
        yr = np.asarray(et.year.isin(years))
        m = np.asarray(et.hour * 60 + et.minute)            # 봉 끝 시각(ET)
        ses = np.where((m > 570) & (m <= 960), "정규", np.where(m <= 570, "프리", "애프터"))
        n = len(c)
        fw = {}
        for H_ in (1, 3, 6):
            f = np.full(n, np.nan)
            f[:n - H_] = c[H_:] / o[1:n - H_ + 1] - 1       # 다음 봉 시가 매수 → H봉 뒤 종가
            fw[H_] = f
        for name, sig in signals(c, h, l).items():
            for H_ in (1, 3, 6):
                key = (name, H_)
                acc.setdefault(key, {"ev": [], "base": [], "ses": []})
                acc[key]["ev"].append(fw[H_][sig & yr]); acc[key]["ses"].append(ses[sig & yr])
                acc[key]["base"].append(fw[H_][yr])
    rows = []
    for (name, H_), v in acc.items():
        ev = np.concatenate(v["ev"]); se = np.concatenate(v["ses"]); bs = np.concatenate(v["base"])
        ok = ~np.isnan(ev); ev, se = ev[ok], se[ok]; bs = bs[~np.isnan(bs)]
        ex = ev.mean() - bs.mean()
        t = ex / (ev.std(ddof=1) / np.sqrt(len(ev))) if len(ev) > 2 else np.nan
        by = {k: f"{(ev[se == k].mean() - bs.mean()):+.2%}({(se == k).sum()})" for k in ("프리", "정규", "애프터")}
        rows.append({"신호": name, "봉": H_, "n": len(ev), "초과": f"{ex:+.2%}", "t": round(t, 1),
                     "상승확률": f"{(ev > 0).mean():.0%}/{(bs > 0).mean():.0%}", **by})
    return pd.DataFrame(rows)


pd.set_option("display.width", 220)
for kind, years, lab in (("60연장", [2026], "60분 연장봉 2026"), ("60연장", [2025], "60분 연장봉 2025"),
                         ("40연장", [2026], "40분 연장봉 최근 60일")):
    t = study(kind, years)
    print(f"\n===== {lab} (다음 봉 시가 매수 → N봉 뒤, 평소 대비 초과 / 세션별 초과(건수)) =====")
    print(t.sort_values(["봉", "신호"]).to_string(index=False))
