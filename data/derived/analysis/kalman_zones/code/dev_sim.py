"""이격 매수 모의매매: 3분봉 저가가 보라선(EMA) 대비 g% 아래로 내려가면 사고, 3분봉 RSI(14)가 X를 넘으면 판다.
데이터: yfinance 1분봉(최근 19거래일, 프리·정규·애프터) → 3분봉. 야간 세션(주간거래) 봉은 없다.
체결: 매수 = 직전 봉 EMA x (1-g) 지정가에 닿으면 min(시가, 지정가) / 매도 = RSI가 X를 넘은 봉 다음 봉 시가.
깨어 있는 08~24시 KST에만 매수·RSI 매도. 자는 동안 처리(sleep)는 비교용으로 네 가지."""
import sys
import numpy as np, pandas as pd
from osc import rsi
from dev_check import bars3

SYMS = ["AMD", "ARM", "DELL", "INTC", "SNDK", "STX", "WDC"]


def prep(s, ema_len):
    b = bars3(s)
    et = b.index
    m = et.hour * 60 + et.minute
    kst = et.tz_convert("Asia/Seoul")
    return dict(
        o=b.Open.values, h=b.High.values, l=b.Low.values, c=b.Close.values,
        ema=b.Close.ewm(span=ema_len, adjust=False).mean().values,
        r=rsi(b.Close.values, 14),
        reg=np.asarray((m >= 570) & (m < 960)), pre=np.asarray((m >= 240) & (m < 570)),
        awake=np.asarray(kst.hour >= 8), kst=kst, day=np.asarray(et.date))


def sim(d, g, x, sleep):
    """sleep: 'hold' 자는 동안 그냥 보유(깨면 RSI 규칙) / 'ema' 자기 전 보라선에 지정가 매도(정규장 체결) /
    'flat' 잠들기 직전 봉 종가에 정리 / 'none' 매도만 잠 무시(밤새 RSI 매도, 참고용). 매수는 모두 깨어 있을 때만."""
    o, h, l, c, ema, r = d["o"], d["h"], d["l"], d["c"], d["ema"], d["r"]
    reg, pre, awake = d["reg"], d["pre"], d["awake"]
    n = len(c)
    trades = []
    hold = False
    ent = ei = 0
    lim = np.nan
    for i in range(61, n):
        ses = reg[i] or pre[i]
        if hold:
            px = np.nan
            if r[i - 1] > x and i - 1 >= ei and ses and (awake[i] or sleep == "none"):
                px, why = o[i], "rsi"                      # 직전 봉 RSI > x → 이번 봉 시가
            elif not np.isnan(lim) and reg[i] and h[i] >= lim:
                px, why = max(o[i], lim), "lim"            # 자는 동안 걸어 둔 지정가
            if not np.isnan(px):
                trades.append((ei, i, ent, px, why))
                hold, lim = False, np.nan
            elif awake[i] and i + 1 < n and not awake[i + 1]:   # 잠들기 직전 봉
                if sleep == "flat" and ses:
                    trades.append((ei, i, ent, c[i], "flat"))
                    hold = False
                elif sleep == "ema":
                    lim = max(ema[i], c[i] * 1.001)
            if not hold:
                continue
        elif ses and awake[i]:
            L = ema[i - 1] * (1 - g)
            if l[i] <= L:
                ent, ei, hold, lim = min(o[i], L), i, True, np.nan
    if hold:
        trades.append((ei, n - 1, ent, c[-1], "open"))
    return trades


def summarize(rows):
    t = pd.DataFrame(rows, columns=["sym", "ei", "xi", "ent", "px", "why", "bars", "mae", "ekst"])
    t["ret"] = t.px / t.ent - 1
    return t


def run(g=0.02, x=70, ema_len=60, sleep="hold", D=None):
    rows = []
    for s in SYMS:
        d = D[(s, ema_len)]
        for ei, xi, ent, px, why in sim(d, g, x, sleep):
            mae = d["l"][ei:xi + 1].min() / ent - 1
            rows.append((s, ei, xi, ent, px, why, xi - ei, mae, d["kst"][ei]))
    return summarize(rows)


if __name__ == "__main__":
    D = {}
    for s in SYMS:
        for L in (25, 60, 100):
            D[(s, L)] = prep(s, L)
    pd.set_option("display.width", 220)
    out = []
    for sleep in ("hold", "ema", "flat", "none"):
        for L in (25, 60, 100):
            for g in (0.015, 0.02, 0.025, 0.03):
                for x in (60, 65, 70, 75):
                    t = run(g, x, L, sleep, D)
                    if len(t) == 0:
                        continue
                    out.append(dict(sleep=sleep, ema=L, g=g, x=x, n=len(t), win=(t.ret > 0).mean(), avg=t.ret.mean(),
                                    med=t.ret.median(), worst=t.ret.min(), cum=np.prod(1 + t.ret) - 1,
                                    hold_med=t.bars.median() * 3, mae_med=t.mae.median(),
                                    lim=(t.why == "lim").mean(), flat=(t.why == "flat").mean()))
    res = pd.DataFrame(out)
    res.to_csv("../results/dev_sim.csv", index=False)
    f = lambda v: f"{v:+.2%}"
    show = res.copy()
    for k in ("avg", "med", "worst", "cum", "mae_med"):
        show[k] = show[k].map(f)
    show["win"] = show.win.map(lambda v: f"{v:.0%}")
    print(show[(show.ema == 60) & (show.x == 70)].to_string(index=False))
    print()
    print(show[(show.ema == 60) & (show.g == 0.02)].to_string(index=False))
    print()
    print(show[(show.x == 70) & (show.g == 0.02)].to_string(index=False))
