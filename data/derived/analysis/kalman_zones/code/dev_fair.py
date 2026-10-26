"""공정 비교: 깨어 있는 프리·정규장의 '모든 봉'에서 각각 샀다고 치고(겹침 허용) 같은 매도·자기 전 정책을 적용한 평균 vs 이격 매수.
이격 매수 건들은 같은 날 여러 종목이 함께 걸리므로(9/28 7종목 모두) t값은 실제보다 부풀려진다."""
import contextlib, io
import numpy as np, pandas as pd
from numba import njit
import dev_sim, dev_5m
from dev_stop import sim


@njit(cache=True)
def exit_from(i, o, h, c, ema, r, reg, pre, awake, x, bedpol):
    n = len(c); lim = np.nan
    for j in range(i + 1, n):
        ses = reg[j] or pre[j]
        bed = awake[j] and j + 1 < n and not awake[j + 1]
        if awake[j] and not bed:
            lim = np.nan
        if r[j - 1] > x and ses and awake[j]:
            return o[j]
        if not np.isnan(lim) and reg[j] and h[j] >= lim:
            return max(o[j], lim)
        if bed:
            if bedpol == 2:
                return c[j]
            if bedpol == 1:
                lim = max(ema[j], c[j] * 1.001)
    return np.nan


def base(D, bedpol):
    out = []
    for d in D.values():
        ok = np.where((d["reg"] | d["pre"]) & d["awake"])[0]
        for i in ok[ok > 61]:
            px = exit_from(i, d["o"], d["h"], d["c"], d["ema"], d["r"], d["reg"], d["pre"], d["awake"], 70.0, bedpol)
            if not np.isnan(px):
                out.append(px / d["o"][i] - 1)
    return np.array(out)


D3 = {s: dev_sim.prep(s, 60) for s in dev_sim.SYMS}
with contextlib.redirect_stdout(io.StringIO()):
    D5 = {s: dev_5m.prep(s, 36, 9) for s in dev_5m.SYMS}
for lab, D in (("3분봉 19일", D3), ("5분봉 근사 60일", D5)):
    print(f"== {lab} ==")
    for bp, bl, key in ((0, "A 그대로 보유", None), (1, "C 보라선 지정가", "ema"), (2, "B 자기 전 정리", "now")):
        b = base(D, bp)
        g = np.array([px / ent - 1 for d in D.values() for ei, xi, ent, px in sim(d, 0.02, 70, 0, key)])
        t = (g.mean() - b.mean()) / (g.std(ddof=1) / np.sqrt(len(g)))
        print(f"  {bl:12s} 이격 {len(g):3d}회 평균 {g.mean():+.2%} 승률 {(g > 0).mean():.0%} | 아무 봉 {len(b):6d}개 평균 {b.mean():+.2%} "
              f"승률 {(b > 0).mean():.0%} | 초과 {g.mean() - b.mean():+.2%} t={t:.1f}")
