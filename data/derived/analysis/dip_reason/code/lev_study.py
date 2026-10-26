"""3배 레버리지 ETF는 이격 매수 기준을 2%가 아니라 3%(또는 다른 값)로 둘까(2026-10-08 사용자 질문).
지금 지표 규칙(보라선 EMA60 × (1-이격) 지정가 매수, RSI14>70 다음 봉 시가 매도, 매수 뒤 첫 아침 8시 정리,
전날 정규장 +H% 이상이면 다음 날 정규장 매수 안 함)을 그대로 두고 이격 %와 전날 급등 기준 H만 바꾼다.
기간: 앞 = 5분봉 근사(EMA36·RSI9), 뒤 = 3분봉(EMA60·RSI14, 1분봉 합성). 사용: py lev_study.py <raw 폴더>"""
import sys
from datetime import date
import numpy as np, pandas as pd
from events import bars, prep
from rules import hot_days

LEV = ["SOXL", "SOXS", "TQQQ", "SQQQ", "TECL", "LABU", "NAIL", "DPST", "KORU", "SPXL", "TNA"]
GAPS = [0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06]
HOTS = [0.07, 0.15, 0.21, None]


def arrays(raw, s, mins, el, rl):
    d = prep(bars(raw, s, mins), el, rl)
    kst = d.index.tz_convert("Asia/Seoul")
    t8 = kst.normalize() + pd.Timedelta(hours=8)
    dead = np.where(kst < t8, t8.as_unit("ns").asi8, (t8 + pd.Timedelta(days=1)).as_unit("ns").asi8)
    H = {h: (hot_days(raw, s, d, h) if h else np.zeros(len(d), bool)) for h in HOTS}
    return d, dict(o=d.o.values, l=d.l.values, r=d.r.values, ema=d.ema.values, tk=d.index.as_unit("ns").asi8,
                   dead=dead, reg=d.ses.to_numpy(dtype="U8") == "정규", day=np.asarray(d.date), H=H)


def sim(A, gap, hot, reg_only):
    o, l, r, ema, tk, dd, reg, hb = A["o"], A["l"], A["r"], A["ema"], A["tk"], A["dead"], A["reg"], A["H"][hot]
    out, hold, ent, ei, dead = [], False, 0.0, 0, 0
    for i in range(61, len(o)):
        if hold:
            if (i - 1 >= ei and r[i - 1] > 70) or tk[i] >= dead:
                out.append((ei, o[i] / ent - 1)); hold = False
        elif not hb[i] and not (reg_only and not reg[i]):
            lvl = ema[i - 1] * (1 - gap)
            if l[i] <= lvl:
                ent, ei, hold, dead = min(o[i], lvl), i, True, dd[i]
    return out


if __name__ == "__main__":
    raw = sys.argv[1]
    P = {"앞 7/16~9/8 (5분봉 근사)": (5, 36, 9, date(2026, 7, 16), date(2026, 9, 8)),
         "뒤 9/11~10/7 (3분봉)": (3, 60, 14, date(2026, 9, 11), date(2026, 10, 7))}
    rows = []
    for pname, (mins, el, rl, lo, hi) in P.items():
        for s in LEV:
            d, A = arrays(raw, s, mins, el, rl)
            for gap in GAPS:
                for hot in HOTS:
                    for reg_only in (False, True):
                        for ei, ret in sim(A, gap, hot, reg_only):
                            if lo <= A["day"][ei] <= hi:
                                rows.append((pname, s, gap, hot if hot else 0.0, "정규장만" if reg_only else "24시간", A["day"][ei], ret))
    t = pd.DataFrame(rows, columns=["기간", "sym", "gap", "hot", "범위", "day", "ret"])
    t.to_csv("../results/lev_trades.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for (pname, scope), v in t.groupby(["기간", "범위"], sort=False):
        print(f"\n===== {pname} · {scope} · 3배 ETF {v.sym.nunique()}개 합산 (전날 급등 기준별)")
        g = v.groupby(["hot", "gap"]).ret.agg(건수="size", 평균="mean", 승률=lambda r: (r > 0).mean(), 최악="min", 합="sum").reset_index()
        g["hot"] = g.hot.map(lambda h: f"+{h:.0%}" if h else "끔")
        for c, f in (("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("합", "{:+.2f}")):
            g[c] = g[c].map(f.format)
        g["gap"] = g.gap.map("{:.1%}".format)
        print(g.pivot_table(index="gap", columns="hot", values="합", aggfunc="first", sort=False).to_string())
        print("  (전날 급등 +7% 기준에서 이격별 건수·평균·승률·최악)")
        print(g[g.hot == "+7%"][["gap", "건수", "평균", "승률", "최악", "합"]].to_string(index=False))
    # 종목별: 24시간, 전날 급등 끔, 이격별 합
    for pname, v in t[(t.범위 == "24시간") & (t.hot == 0.0)].groupby("기간", sort=False):
        print(f"\n종목별 합 ({pname}, 24시간, 전날 급등 규칙 끔)")
        print(v.pivot_table(index="sym", columns="gap", values="ret", aggfunc="sum").round(2).to_string())
