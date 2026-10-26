"""'얇은 세션 이격 → 그날 18시(한국) 프리장에 매수' 규칙의 대리 시험. 주간거래 봉이 없어 애프터마켓(16~20시 ET) 이격으로 본다.
매수: 보라선(EMA) -2% (닿으면 min(시가, 매수선)) / 매도: RSI>70 다음 봉 시가, 없으면 매수 뒤 첫 8시(한국) 이후 첫 봉 시가. 24시간, 수수료 0%.
wait=True면 애프터 이격은 표시만 하고, 다음 18시(한국) 이후 첫 봉 시가에 산다. 그 전에 다른 세션에서 이격이 나면 그때 산다."""
import contextlib, io
import numpy as np, pandas as pd
import dev_sim, dev_5m

DAY = pd.Timedelta(days=1)


def next_at(t, hr):
    d = t.normalize() + pd.Timedelta(hours=hr)
    return d + DAY if t >= d else d


def sim(d, wait, g=0.02, x=70):
    o, l, c, ema, r, kst = d["o"], d["l"], d["c"], d["ema"], d["r"], d["kst"]
    post = ~(d["reg"] | d["pre"])
    n = len(c); out = []; hold = False; pend = False; ent = ei = 0; dead = pt = None; ppx = np.nan; src = ""
    for i in range(61, n):
        t = kst[i]
        if hold:
            if (i - 1 >= ei and r[i - 1] > x) or t >= dead:
                out.append(dict(ei=ei, xi=i, ent=ent, px=o[i], src=src, trig=ppx)); hold = False
            continue
        L = ema[i - 1] * (1 - g)
        hit = l[i] <= L
        if pend and t >= pt:                                   # 예약 시각: 무조건 시가에 산다
            ent, ei, hold, pend, src = o[i], i, True, False, "18시"
        elif hit and wait and post[i]:
            if not pend:
                pend, pt, ppx = True, next_at(t, 18), min(o[i], L)
            continue
        elif hit:
            ent, ei, hold, pend, src, ppx = min(o[i], L), i, True, False, ("애프터" if post[i] else "그외"), min(o[i], L)
        if hold:
            dead = next_at(t, 8)
    return out


def run(D, lab):
    res = {}
    for wait in (False, True):
        rows = []
        for s, d in D.items():
            for tr in sim(d, wait):
                tr["sym"] = s; tr["ret"] = tr["px"] / tr["ent"] - 1; rows.append(tr)
        res[wait] = pd.DataFrame(rows)
    a, b = res[False], res[True]
    print(f"== {lab} ==")
    for k, t in (("바로 매수", a), ("애프터 이격은 18시 매수", b)):
        print(f"  {k:14s}: {len(t)}회 · 승률 {(t.ret > 0).mean():.0%} · 평균 {t.ret.mean():+.2%} · 최악 {t.ret.min():+.2%} · 합계 {t.ret.sum():+.0%}")
    ah = a[a.src == "애프터"]; w = b[b.src == "18시"]
    print(f"  애프터 이격 바로 매수 {len(ah)}회 평균 {ah.ret.mean():+.2%} | 18시 매수 {len(w)}회 평균 {w.ret.mean():+.2%} 승률 {(w.ret > 0).mean():.0%}")
    if len(w):
        gap = w.ent / w.trig - 1
        print(f"  18시 매수가 vs 애프터 이격가: 평균 {gap.mean():+.2%}, 중앙 {gap.median():+.2%}, 더 싸게 산 비율 {(gap < 0).mean():.0%} ({(gap < 0).sum()}/{len(gap)})")


if __name__ == "__main__":
    D3 = {s: dev_sim.prep(s, 60) for s in dev_sim.SYMS}
    run(D3, "3분봉 9/1~9/28")
    with contextlib.redirect_stdout(io.StringIO()):
        D5 = {s: dev_5m.prep(s, 36, 9) for s in dev_5m.SYMS}
    run(D5, "5분봉 근사 7/2~9/25")
