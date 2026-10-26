"""이격 매수를 24시간 거래 가능으로(자는 시간·세션 제한 없음): 저가가 직전 봉 EMA x 0.98에 닿으면 min(시가, 매수선),
RSI(14) > 70인 봉 다음 봉 시가에 매도. 프리·정규·애프터 모든 봉(야간 세션은 데이터 없음). 수수료 0%, 손절 없음."""
import contextlib, io
import numpy as np, pandas as pd
import dev_sim, dev_5m


def sim24(d, g=0.02, x=70):
    o, h, l, c, ema, r = (d[k] for k in ("o", "h", "l", "c", "ema", "r"))
    n = len(c); out = []; hold = False; ent = ei = 0
    eq = np.ones(n); cash = 1.0; sh = 0.0
    for i in range(61, n):
        if hold and i - 1 >= ei and r[i - 1] > x:
            out.append((ei, i, ent, o[i])); cash = sh * o[i]; sh = 0.0; hold = False
        elif not hold:
            L = ema[i - 1] * (1 - g)
            if l[i] <= L:
                ent, ei, hold = min(o[i], L), i, True; sh = cash / ent; cash = 0.0
        eq[i] = cash + sh * c[i]
    return out, eq


def report(D, lab, sn_key):
    rows, cols = [], []
    for s, d in D.items():
        tr, eq = sim24(d)
        for ei, xi, ent, px in tr:
            rows.append((s, px / ent - 1, (xi - ei), d["kst"][ei], d["kst"][xi]))
        e = pd.Series(eq, index=d["kst"])
        cols.append(e.groupby(e.index.date).last())
    t = pd.DataFrame(rows, columns=["sym", "ret", "bars", "buy", "sell"])
    P = pd.concat(cols, axis=1).ffill().fillna(1.0).mean(axis=1)
    print(f"== {lab} == {len(t)}회 · 승률 {(t.ret > 0).mean():.0%} · 평균 {t.ret.mean():+.2%} · 최악 {t.ret.min():+.2%} · "
          f"보유 중앙 {t.bars.median() * sn_key:.0f}분 | 7종목 계좌 누적 {P.iloc[-1] - 1:+.1%} · 최대 낙폭 {(P / P.cummax() - 1).min():+.1%}")
    return t


if __name__ == "__main__":
    D3 = {s: dev_sim.prep(s, 60) for s in dev_sim.SYMS}
    t3 = report(D3, "3분봉 9/1~9/28", 3)
    sn = t3[(t3.sym == "SNDK") & (t3.buy >= pd.Timestamp("2026-09-28 22:00", tz="Asia/Seoul"))]
    print("  SNDK 9/28 밤:", ", ".join(f"{b:%m-%d %H:%M} 매수 → {s:%m-%d %H:%M} 매도 {r:+.2%}" for b, s, r in zip(sn.buy, sn.sell, sn.ret)))
    with contextlib.redirect_stdout(io.StringIO()):
        D5 = {s: dev_5m.prep(s, 36, 9) for s in dev_5m.SYMS}
    report(D5, "5분봉 근사 7/2~9/25", 5)


def sim24x(d, close_fill, g=0.02, x=70):
    """세션별 분해용: 매수 봉의 세션을 같이 돌려준다. close_fill이면 종가 <= 매수선일 때 그 종가에 산다."""
    o, h, l, c, ema, r, reg, pre = (d[k] for k in ("o", "h", "l", "c", "ema", "r", "reg", "pre"))
    n = len(c); out = []; hold = False; ent = ei = 0
    for i in range(61, n):
        if hold and i - 1 >= ei and r[i - 1] > x:
            out.append((ei, i, ent, o[i], "정규" if reg[ei] else "프리" if pre[ei] else "애프터")); hold = False
        elif not hold:
            L = ema[i - 1] * (1 - g)
            if close_fill and c[i] <= L:
                ent, ei, hold = c[i], i, True
            elif not close_fill and l[i] <= L:
                ent, ei, hold = min(o[i], L), i, True
    return out


def split(D, lab):
    for cf in (False, True):
        rs = [(ses, px / ent - 1) for d in D.values() for ei, xi, ent, px, ses in sim24x(d, cf)]
        t = pd.DataFrame(rs, columns=["ses", "ret"])
        by = t.groupby("ses").ret.agg(["count", "mean"])
        txt = " · ".join(f"{k} {int(v['count'])}회 {v['mean']:+.2%}" for k, v in by.iterrows())
        print(f"  {lab} {'종가 체결' if cf else '닿으면 체결'}: 전체 {len(t)}회 평균 {t.ret.mean():+.2%} | {txt}")


if __name__ == "__main__":
    print("\n== 매수 세션별 (24시간) ==")
    split(D3, "3분봉")
    split(D5, "5분봉 근사")
