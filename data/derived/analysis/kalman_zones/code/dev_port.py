"""이격 매수(자기 전 그대로 보유, 손절 없음/−3%)를 7종목 같은 비중 계좌로: 누적 수익·최대 낙폭.
체결: 닿으면(저가 ≤ 매수선 → min(시가, 매수선)) / 종가(종가 ≤ 매수선일 때 그 종가, 프리장 꼬리 착시 배제).
기존 지정가 매수(run_sr2 'short' 60일, 추매 없음·새 저항선)와 같은 5분봉 60일로 비교한다."""
import contextlib, io
import numpy as np, pandas as pd
import dev_sim, dev_5m


def equity(d, g, x, stop, close_fill):
    o, h, l, c, ema, r, reg, pre, awake = (d[k] for k in ("o", "h", "l", "c", "ema", "r", "reg", "pre", "awake"))
    n = len(c); eq = np.ones(n); cash = 1.0; sh = 0.0; ent = 0.0; ei = 0; hold = False; nt = 0
    for i in range(61, n):
        ses = reg[i] or pre[i]
        if hold:
            if ses and awake[i] and i - 1 >= ei and (r[i - 1] > x or (stop and c[i - 1] <= ent * (1 - stop))):
                cash = sh * o[i]; sh = 0.0; hold = False; nt += 1
        elif ses and awake[i]:
            L = ema[i - 1] * (1 - g)
            if close_fill and c[i] <= L:
                ent = c[i]; sh = cash / ent; cash = 0.0; hold = True; ei = i
            elif not close_fill and l[i] <= L:
                ent = min(o[i], L); sh = cash / ent; cash = 0.0; hold = True; ei = i
        eq[i] = cash + sh * c[i]
    return pd.Series(eq, index=d["et"]), nt


def port(D, **kw):
    cols, nt = [], 0
    for s, d in D.items():
        e, t = equity(d, 0.02, 70, **kw)
        cols.append(e.groupby(e.index.date).last()); nt += t
    P = pd.concat(cols, axis=1).ffill().fillna(1.0).mean(axis=1)
    return P.iloc[-1] / P.iloc[0] - 1, (P / P.cummax() - 1).min(), nt, P


with contextlib.redirect_stdout(io.StringIO()):
    D5 = {s: dev_5m.prep(s, 36, 9) for s in dev_5m.SYMS}
    for s in D5:
        D5[s]["et"] = D5[s]["kst"].tz_convert("America/New_York")
D3 = {s: dev_sim.prep(s, 60) for s in dev_sim.SYMS}
for s in D3:
    D3[s]["et"] = D3[s]["kst"].tz_convert("America/New_York")
for lab, D in (("5분봉 근사 60일 (7/2~9/25)", D5), ("3분봉 19일 (9/1~9/28)", D3)):
    print(f"== {lab} ==")
    for stop in (0, 0.03):
        for cf in (False, True):
            ret, mdd, nt, P = port(D, stop=stop, close_fill=cf)
            print(f"  손절 {'없음' if not stop else '-3%'} · {'종가 체결' if cf else '닿으면 체결'}: 누적 {ret:+.1%} · 최대 낙폭 {mdd:+.1%} · {nt}회")
    if D is D5:
        ret, mdd, nt, P = port(D, stop=0, close_fill=True)
        m = P.groupby(pd.to_datetime(P.index).month).apply(lambda v: v.iloc[-1] / v.iloc[0] - 1)
        print("   월별(손절 없음·종가 체결):", {k: f"{v:+.1%}" for k, v in m.items()})
print("\n기존 지정가 매수 같은 60일(run_sr2 short, 추매 없음·새 저항선): 닿으면 체결 +74.3%/낙폭 -6.5%, 종가 체결 +7.2%/낙폭 -11.1%")


def port_from(D, start, **kw):
    """start(ET 날짜)부터 매매를 시작한 계좌: 그 전 봉은 지표 계산에만 쓴다."""
    cols = []
    for s, d in D.items():
        d2 = dict(d)
        et = d["et"]
        cut = int(np.searchsorted(np.asarray(et.date), start))
        m = np.zeros(len(et), bool); m[cut:] = True
        d2["awake"] = d["awake"] & m                       # 시작 전에는 매수 안 함
        e, _ = equity(d2, 0.02, 70, **kw)
        e = e[e.index.date >= start]
        cols.append(e.groupby(e.index.date).last() / e.iloc[0])
    P = pd.concat(cols, axis=1).ffill().fillna(1.0).mean(axis=1)
    return P.iloc[-1] - 1, (P / P.cummax() - 1).min()


if __name__ == "__main__":
    st = pd.Timestamp("2026-07-16").date()
    print("\n== 5분봉 근사, 기존 지정가 매수와 같은 7/16~9/25 ==")
    for stop in (0, 0.03):
        for cf in (False, True):
            ret, mdd = port_from(D5, st, stop=stop, close_fill=cf)
            print(f"  손절 {'없음' if not stop else '-3%'} · {'종가 체결' if cf else '닿으면 체결'}: 누적 {ret:+.1%} · 최대 낙폭 {mdd:+.1%}")
