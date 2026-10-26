"""가설 검증: 40분봉 RSI14<30 & 스토캐스틱RSI K<30 동시 진입 뒤 반등하는가 (시간 제약 없이, 평소 봉과 비교).

같은 신호를 60분봉 정규장(1시간봉 기반) 2026년, 2024~2025년에도 본다. 큰 추세는 240분 정규장 칼만 5단계 구간.
"""
import numpy as np
import pandas as pd

from core import *
from evalcfg import data, indicator
from osc import rsi, stoch_rsi, forward

pd.set_option("display.width", 220)
D, HALF = data()
ZN = {2: "과매수", 1: "매수", 0: "평범", -1: "매도", -2: "패닉셀"}
HS = (1, 3, 6, 10)


def big_zone(s):
    tb = tf_bars(D[(s, "1h")], 240, False, HALF)
    _, _, z = indicator(tb, 21, "rms", 50, 0.75, 1.75)
    return tb.end_ns.values, z


def events(tf, eth, src, years, rl=14, st=14, thr_r=30, thr_k=30):
    rows = []
    for s in SYMS:
        tb = tf_bars(D[(s, src)], tf, eth, HALF)
        c, h, l = tb.C.values, tb.H.values, tb.L.values
        r = rsi(c, rl)
        k, _ = stoch_rsi(r, st, 3, 3)
        cond = (r < thr_r) & (k < thr_k)
        onset = cond & ~np.r_[False, cond[:-1]]
        be, bz = big_zone(s)
        zi = np.searchsorted(be, tb.end_ns.values, side="right") - 1
        z = np.where(zi >= 0, bz[np.maximum(zi, 0)], -9)
        yr = pd.to_datetime(tb.end_ns.values, utc=True).year
        df = pd.DataFrame({"s": s, "onset": onset, "zone": z, "yr": yr})
        for H in HS:
            f, up, dn = forward(c, h, l, H)
            df[f"f{H}"], df[f"up{H}"], df[f"dn{H}"] = f, up, dn
        rows.append(df[df.yr.isin(years)])
    return pd.concat(rows, ignore_index=True)


def report(name, df):
    ev, base = df[df.onset], df
    print(f"\n=== {name}: 신호 {len(ev)}번 / 전체 봉 {len(base)} ===")
    print("            " + "  ".join(f"{H:>2}봉뒤 수익(평소)  상승확률(평소)  최대반등(평소)  최대하락(평소)" for H in (1, 3)))
    for H in HS:
        e, b = ev[f"f{H}"].dropna(), base[f"f{H}"].dropna()
        print(f"  {H:>2}봉 뒤: 수익 {e.mean():+.2%} ({b.mean():+.2%})  상승확률 {(e > 0).mean():.0%} ({(b > 0).mean():.0%})  "
              f"그사이 최대반등 {ev[f'up{H}'].mean():+.2%} ({base[f'up{H}'].mean():+.2%})  최대하락 {ev[f'dn{H}'].mean():+.2%} ({base[f'dn{H}'].mean():+.2%})")
    g = ev.groupby("zone")[["f3", "f6", "up3", "dn3"]].agg(["mean", "count"])
    for zc in (2, 1, 0, -1, -2):
        if zc in g.index:
            r = g.loc[zc]
            print(f"   큰추세 {ZN[zc]:<4} 신호 {int(r[('f3','count')]):>3}번  3봉뒤 {r[('f3','mean')]:+.2%}  6봉뒤 {r[('f6','mean')]:+.2%}  3봉내 최대반등 {r[('up3','mean')]:+.2%}  최대하락 {r[('dn3','mean')]:+.2%}")


report("40분 정규장 (7/2~9/25)", events(40, False, "5m", [2026]))
report("40분 연장시간 (7/2~9/25)", events(40, True, "5m", [2026]))
report("60분 정규장 2026년", events(60, False, "1h", [2026]))
report("60분 정규장 2024~2025년 (다른 장세)", events(60, False, "1h", [2024, 2025]))
