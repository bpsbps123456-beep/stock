"""24시간 이격 매수 + '매수 뒤 처음 오는 아침 8시(KST)에 정리'(RSI 매도가 먼저 오면 RSI). 8시 이후 첫 봉 시가에 판다."""
import contextlib, io
import numpy as np, pandas as pd
import dev_sim, dev_5m


def sim8(d, hr=8, use_dead=True, g=0.02, x=70):
    o, h, l, c, ema, r, kst = (d[k] for k in ("o", "h", "l", "c", "ema", "r", "kst"))
    reg, pre = d["reg"], d["pre"]
    tk = kst.as_unit("ns").asi8
    day = 86_400_000_000_000
    n = len(c); out = []; hold = False; ent = ei = 0; dead = 0
    eq = np.ones(n); cash = 1.0; sh = 0.0
    for i in range(61, n):
        if hold:
            px = why = None
            if i - 1 >= ei and r[i - 1] > x:
                px, why = o[i], "RSI"
            elif use_dead and tk[i] >= dead:
                px, why = o[i], f"{hr}시"
            if px is not None:
                out.append((ei, i, ent, px, why, "정규" if reg[ei] else "프리" if pre[ei] else "애프터"))
                cash = sh * px; sh = 0.0; hold = False
        elif ema[i - 1] * (1 - g) >= l[i]:
            ent, ei, hold = min(o[i], ema[i - 1] * (1 - g)), i, True
            sh = cash / ent; cash = 0.0
            t = kst[i]
            d8 = t.normalize() + pd.Timedelta(hours=hr)
            if t >= d8:
                d8 += pd.Timedelta(days=1)
            dead = d8.as_unit("ns").value
        eq[i] = cash + sh * c[i]
    return out, eq


def report(D, lab, mins, **kw):
    rows, cols = [], []
    for s, d in D.items():
        tr, eq = sim8(d, **kw)
        rows += [(s, px / ent - 1, xi - ei, why, ses, d["kst"][ei], d["kst"][xi]) for ei, xi, ent, px, why, ses in tr]
        e = pd.Series(eq, index=d["kst"]); cols.append(e.groupby(e.index.date).last())
    t = pd.DataFrame(rows, columns=["sym", "ret", "bars", "why", "ses", "buy", "sell"])
    P = pd.concat(cols, axis=1).ffill().fillna(1.0).mean(axis=1)
    by = " · ".join(f"{k} {len(v)}회 {v.ret.mean():+.2%}" for k, v in t.groupby("why"))
    print(f"{lab}: {len(t)}회 · 승률 {(t.ret > 0).mean():.0%} · 평균 {t.ret.mean():+.2%} · 최악 {t.ret.min():+.2%} · 보유 중앙 {t.bars.median() * mins:.0f}분"
          f" | 계좌 {P.iloc[-1] - 1:+.1%} · 낙폭 {(P / P.cummax() - 1).min():+.1%} | 매도 이유: {by}")
    return t


if __name__ == "__main__":
    D3 = {s: dev_sim.prep(s, 60) for s in dev_sim.SYMS}
    with contextlib.redirect_stdout(io.StringIO()):
        D5 = {s: dev_5m.prep(s, 36, 9) for s in dev_5m.SYMS}
    for lab, D, mins in (("3분봉 9/1~9/28", D3, 3), ("5분봉 근사 7/2~9/25", D5, 5)):
        report(D, f"{lab} RSI만      ", mins, use_dead=False)
        t = report(D, f"{lab} RSI+8시 정리", mins, use_dead=True)
        if D is D3:
            sn = t[(t.sym == "SNDK") & (t.buy >= pd.Timestamp("2026-09-28 22:00", tz="Asia/Seoul"))]
            print("   SNDK 9/28 밤:", ", ".join(f"{b:%m/%d %H:%M} 매수 → {s:%m/%d %H:%M} {w} 매도 {r:+.2%}" for b, s, w, r in zip(sn.buy, sn.sell, sn.why, sn.ret)))
