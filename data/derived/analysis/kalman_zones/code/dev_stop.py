"""(1) 5분봉 근사 60일에서도 이격 매수가 '아무 봉 매수 + 같은 매도'보다 나은가 (2) 손절(깨어 있을 때 종가가 매수가 -s%면 다음 봉 시가)."""
import numpy as np, pandas as pd
import dev_sim, dev_5m


def sim(d, g, x, stop, bedpol, any_entry=False):
    o, h, l, c, ema, r, reg, pre, awake = (d[k] for k in ("o", "h", "l", "c", "ema", "r", "reg", "pre", "awake"))
    n = len(c); out = []; hold = False; lim = np.nan; ent = ei = 0
    for i in range(61, n):
        ses = reg[i] or pre[i]
        bed = awake[i] and i + 1 < n and not awake[i + 1]
        if hold:
            if awake[i] and not bed:
                lim = np.nan
            px = np.nan
            if ses and awake[i] and i - 1 >= ei and (r[i - 1] > x or (stop and c[i - 1] <= ent * (1 - stop))):
                px = o[i]
            elif not np.isnan(lim) and reg[i] and h[i] >= lim:
                px = max(o[i], lim)
            if not np.isnan(px):
                out.append((ei, i, ent, px)); hold = False; lim = np.nan; continue
            if bed:
                if bedpol == "now":
                    out.append((ei, i, ent, c[i])); hold = False; continue
                lim = max(ema[i], c[i] * 1.001) if bedpol == "ema" else np.nan
            continue
        if ses and awake[i]:
            L = ema[i - 1] * (1 - g)
            if any_entry:
                ent, ei, hold, lim = o[i], i, True, np.nan
            elif l[i] <= L:
                ent, ei, hold, lim = min(o[i], L), i, True, np.nan
    return out


def stats(D, **kw):
    rs = []
    for s, d in D.items():
        for ei, xi, ent, px in sim(d, 0.02, 70, **kw):
            rs.append(px / ent - 1)
    t = np.array(rs)
    return len(t), (t > 0).mean(), t.mean(), np.quantile(t, 0.05), t.min(), t.std(ddof=1)


if __name__ == "__main__":
    D3 = {s: dev_sim.prep(s, 60) for s in dev_sim.SYMS}
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        D5 = {s: dev_5m.prep(s, 36, 9) for s in dev_5m.SYMS}
    for lab, D in (("3분봉 19일", D3), ("5분봉 근사 60일", D5)):
        print(f"== {lab} ==")
        for bedpol, bl in ((None, "A 보유"), ("ema", "C 보라선 지정가"), ("now", "B 자기 전 정리")):
            base = stats(D, stop=0, bedpol=bedpol, any_entry=True)
            for stop in (0, 0.03, 0.05):
                n, w, a, q5, mn, sd = stats(D, stop=stop, bedpol=bedpol)
                tv = (a - base[2]) / (sd / np.sqrt(n))
                print(f"  {bl:12s} 손절 {('없음' if not stop else f'-{stop:.0%}'):4s}: {n:3d}회 승률 {w:.0%} 평균 {a:+.2%} "
                      f"나쁜5% {q5:+.2%} 최악 {mn:+.2%} | 아무 봉 매수(손절 없음) 평균 {base[2]:+.2%} 승률 {base[1]:.0%} → 초과 t={tv:.1f}")
