"""프리장 바닥 시각과 개장 반등: 7종목, 5분봉(최근 60일)·1시간봉(2025·2026). 시각은 한국시간(KST)으로 본다."""
import numpy as np, pandas as pd
from core import *
from evalcfg import data

D, H = data()
pd.set_option("display.width", 220)


def days(s, key):
    d = D[(s, key)].copy()
    t = d.index
    d["m"] = t.hour * 60 + t.minute
    d["date"] = t.date
    d["kst_h"] = t.tz_convert("Asia/Seoul").hour
    d["kst"] = t.tz_convert("Asia/Seoul").strftime("%H:%M")
    return d


def study(key, years):
    rows = []
    for s in SYMS:
        d = days(s, key)
        d = d[pd.to_datetime(d.index).year.isin(years)] if key == "1h" else d
        prev_close = None
        for date, g in d.groupby("date"):
            reg = g[(g.m >= 570) & (g.m < 960)]
            pre = g[(g.m >= 240) & (g.m < 570)]
            if prev_close is not None and len(pre) and len(reg):
                lo_i = pre.Low.idxmin()
                op = reg.Open.iloc[0]
                first = reg[reg.m < 570 + 60]                     # 개장 후 첫 1시간
                rows.append({"s": s, "date": date, "pc": prev_close, "pre_low": pre.Low.min(),
                             "pre_low_kst_h": pre.loc[lo_i, "kst_h"], "open": op,
                             "hi60": first.High.max(), "lo60": first.Low.min(), "c60": first.Close.iloc[-1],
                             "close": reg.Close.iloc[-1],
                             "p19": pre[pre.kst_h == 19].Open.iloc[0] if (pre.kst_h == 19).any() else np.nan,
                             "p20": pre[pre.kst_h == 20].Open.iloc[0] if (pre.kst_h == 20).any() else np.nan})
            if len(reg):
                prev_close = reg.Close.iloc[-1]
    return pd.DataFrame(rows)


for key, years, lab in (("5m", [2026], "5분봉 최근 60일"), ("1h", [2026], "1시간봉 2026"), ("1h", [2025], "1시간봉 2025")):
    r = study(key, years)
    down = r[r.pre_low < r.pc * 0.99]                             # 프리장에서 전일 종가 -1% 아래로 빠진 날
    print(f"\n===== {lab}: 종목·일 {len(r)}개, 그중 프리장에서 -1% 넘게 빠진 날 {len(down)}개 =====")
    dist = down.pre_low_kst_h.value_counts(normalize=True).sort_index()
    print("  프리장 최저가 시각(KST, 서머·표준시 섞임) 분포:", " ".join(f"{int(h)}시 {v:.0%}" for h, v in dist.items()))
    for name, x in (("프리장 최저가에 샀다면(가장 운 좋은 경우)", down.pre_low), ("19시 가격에 샀다면", down.p19), ("20시 가격에 샀다면", down.p20)):
        ok = x.notna()
        print(f"  {name}: 개장가까지 {((down.open[ok] / x[ok]) - 1).mean():+.2%} | 개장 1시간 최고가까지 {((down.hi60[ok] / x[ok]) - 1).mean():+.2%}"
              f" | 개장 1시간 뒤 {((down.c60[ok] / x[ok]) - 1).mean():+.2%} | 그날 종가 {((down.close[ok] / x[ok]) - 1).mean():+.2%}  (n={ok.sum()})")
    print(f"  개장 1시간 안에 개장가보다 +1% 넘게 오른 날 비율: {(down.hi60 >= down.open * 1.01).mean():.0%} (모든 날 {(r.hi60 >= r.open * 1.01).mean():.0%})")
    print(f"  개장가 대비 개장 1시간 최고 평균 {((down.hi60 / down.open) - 1).mean():+.2%}, 최저 평균 {((down.lo60 / down.open) - 1).mean():+.2%}")
