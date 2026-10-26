"""(1) 개장 시점 구간별 이후 수익률  (2) 손절 뒤 '팔았더니 반등' 비율: 새 설정 vs 지금 쓰는 원본."""
import numpy as np, pandas as pd
from core import SYMS
from evalcfg import evaluate, prep
NAME = {2: "과매수", 1: "매수", 0: "평범", -1: "매도", -2: "패닉셀"}

def open_bars(S, s):
    reg = np.flatnonzero(S[s]["is_reg"]); d = S[s]["day_idx"][reg]
    first = reg[np.r_[True, d[1:] != d[:-1]]]
    return first[first >= S[s]["j0"]]

def zone_forward(win, cfg, H=(1, 3, 5)):
    S = prep(win)[0]
    _, _, _, zinfo, *_ = evaluate(win, **cfg)
    rows = []
    for s in SYMS:
        tb, z, idx = zinfo[s]; ob = open_bars(S, s); o = S[s]["o"]
        for k, j in enumerate(ob):
            i = idx[j]
            if i < 0 or z[i] == -9:
                continue
            r = {"s": s, "zone": int(z[i])}
            for h in H:
                r[f"f{h}"] = o[ob[k + h]] / o[j] - 1 if k + h < len(ob) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)

def regret(win, cfg, days=3, up=0.03):
    """판단 시가 매도(손절·빨강) 뒤 3거래일 안에 매도가보다 3% 넘게 오른 비율과 5일 뒤 수익률."""
    S = prep(win)[0]
    _, _, logs, *_ = evaluate(win, **cfg, rec=True)
    out = []
    for s in SYMS:
        ob = open_bars(S, s); o, h = S[s]["o"], S[s]["h"]
        day_of = {j: k for k, j in enumerate(ob)}
        for j, side, px, new, why, zn in logs[s]:
            j = int(j)
            if side < 0 and why == 0 and new == 0 and zn == -1 and j in day_of:
                k = day_of[j]
                if k + days >= len(ob):
                    continue
                hi = h[j:ob[k + days]].max()
                f5 = o[ob[k + 5]] / px - 1 if k + 5 < len(ob) else np.nan
                out.append((hi / px - 1 > up, f5))
    a = np.array(out, dtype=float)
    return len(a), a[:, 0].mean() if len(a) else np.nan, np.nanmean(a[:, 1]) if len(a) else np.nan

NEW = dict(tf=240, eth=False, L=21, norm="rms", n=50, a=0.75, b=1.75)
for win in ("long", "short"):
    f = zone_forward(win, NEW)
    base = f[["f1", "f3", "f5"]].mean()
    g = f.groupby("zone")[["f1", "f3", "f5"]].agg(["mean", "count"])
    print(f"\n[{win}] 240분 정규장 새 설정 — 개장 때 보이는 구간별 이후 평균 수익률 (전체 평균 1일 {base.f1:+.2%} / 3일 {base.f3:+.2%} / 5일 {base.f5:+.2%})")
    for zc in (2, 1, 0, -1, -2):
        if zc in g.index:
            r = g.loc[zc]
            print(f"   {NAME[zc]:<4} 표본 {int(r[('f1','count')]):>4}  1일 {r[('f1','mean')]:+.2%}  3일 {r[('f3','mean')]:+.2%}  5일 {r[('f5','mean')]:+.2%}")
print("\n손절 뒤 3거래일 안에 매도가 +3% 넘게 반등한 비율 ('팔았더니 반등')")
cases = [("long", "새 설정 240분 정규장", NEW), ("long", "원본 60분 정규장", dict(tf=60, eth=False, original=True)),
         ("long", "원본 240분 정규장", dict(tf=240, eth=False, original=True)),
         ("short", "새 설정 240분 정규장", NEW), ("short", "원본 40분 연장(지금 차트)", dict(tf=40, eth=True, original=True)),
         ("short", "원본 40분 정규장", dict(tf=40, eth=False, original=True))]
for win, label, cfg in cases:
    n, rate, f5 = regret(win, cfg)
    print(f"   [{win}] {label:<18} 손절 {n:>3}번  반등 비율 {rate:.0%}  손절가 대비 5일 뒤 {f5:+.2%}")
