"""이격 매수의 '자기 전 처리' 비교. 매수: 보라선(EMA) -2% 지정가 / 매도: RSI > 70 다음 봉 시가(깨어 있을 때).
자기 전(잠들기 직전 봉)에 보유 중이면 정책이 정한다: None=그대로, 'now'=그 봉 종가 정리, 숫자=밤새 지정가(정규장 체결, 깨면 취소).
3분봉(정확, 19거래일)과 5분봉 근사(EMA 36·RSI 9, 60일, 7월 급락 포함) 두 가지로."""
import numpy as np, pandas as pd
import dev_sim, dev_5m

POL = {
    "A 그대로 보유(말씀하신 규칙)": lambda nt, ent, c, ema: None,
    "B 자기 전 정리": lambda nt, ent, c, ema: "now",
    "C 매일 밤 보라선 지정가": lambda nt, ent, c, ema: max(ema, c * 1.001),
    "D 첫날 밤 보라선 지정가, 이틀째 밤 정리": lambda nt, ent, c, ema: max(ema, c * 1.001) if nt == 1 else "now",
}


def sim(d, g, x, pol):
    o, h, l, c, ema, r, reg, pre, awake = (d[k] for k in ("o", "h", "l", "c", "ema", "r", "reg", "pre", "awake"))
    n = len(c); out = []; hold = False; lim = np.nan; ent = ei = nights = 0
    for i in range(61, n):
        ses = reg[i] or pre[i]
        bed = awake[i] and i + 1 < n and not awake[i + 1]
        if hold:
            if awake[i] and not bed:
                lim = np.nan
            px = np.nan
            if r[i - 1] > x and i - 1 >= ei and ses and awake[i]:
                px, why = o[i], "rsi"
            elif not np.isnan(lim) and reg[i] and h[i] >= lim:
                px, why = max(o[i], lim), "lim"
            if not np.isnan(px):
                out.append((ei, i, ent, px, why)); hold = False; lim = np.nan; continue
            if bed:
                nights += 1
                a = pol(nights, ent, c[i], ema[i])
                if a == "now":
                    out.append((ei, i, ent, c[i], "bed")); hold = False; continue
                lim = np.nan if a is None else a
            continue
        if ses and awake[i] and l[i] <= ema[i - 1] * (1 - g):
            ent, ei, hold, lim, nights = min(o[i], ema[i - 1] * (1 - g)), i, True, np.nan, 0
    return out


def table(D, kst_key="kst"):
    rows = []
    for name, pol in POL.items():
        rs = []
        for s, d in D.items():
            for ei, xi, ent, px, why in sim(d, 0.02, 70, pol):
                rs.append((px / ent - 1, (xi - ei), d["l"][ei:xi + 1].min() / ent - 1, why))
        t = pd.DataFrame(rs, columns=["ret", "bars", "mae", "why"])
        rows.append(dict(정책=name, 횟수=len(t), 승률=f"{(t.ret > 0).mean():.0%}", 평균=f"{t.ret.mean():+.2%}",
                         중앙=f"{t.ret.median():+.2%}", 최악=f"{t.ret.min():+.2%}", 나쁜5퍼=f"{t.ret.quantile(0.05):+.2%}",
                         합계=f"{t.ret.sum():+.0%}", 보유중앙_봉=int(t.bars.median())))
    return pd.DataFrame(rows)


pd.set_option("display.width", 220)
D3 = {s: dev_sim.prep(s, 60) for s in dev_sim.SYMS}
print("== 3분봉 EMA60, RSI14 (9/1~9/28, 19거래일) ==")
print(table(D3).to_string(index=False))
D5 = {s: dev_5m.prep(s, 36, 9) for s in dev_5m.SYMS}
print("\n== 5분봉 근사 EMA36, RSI9 (7/2~9/25, 60일) ==")
print(table(D5).to_string(index=False))
