"""3배 레버리지 이격별 '총수익률(복리)'(사용자 10/9: 총 수익률로 보면 2.3%가 낫지 않나).
ETF마다 자금을 따로 두고 매매마다 전액 복리(1+r 곱). 11개 같은 비중 계좌 = ETF별 복리 수익의 평균. 최대 낙폭은 매매 단위 자산곡선.
lev_study 규칙 그대로(전날 +7% 포함). 사용: py lev_compound.py <raw 폴더>"""
import sys
from datetime import date
import numpy as np, pandas as pd
from lev_study import LEV, arrays, sim

GAPS = [0.020, 0.022, 0.023, 0.025, 0.027, 0.030, 0.035]
P = {"앞(5분봉)": (5, 36, 9, date(2026, 7, 16), date(2026, 9, 8)), "뒤(3분봉)": (3, 60, 14, date(2026, 9, 11), date(2026, 10, 7))}

if __name__ == "__main__":
    raw = sys.argv[1]
    rows = []
    for pn, (mins, el, rl, lo, hi) in P.items():
        for s in LEV:
            d, A = arrays(raw, s, mins, el, rl)
            for g in GAPS:
                for reg in (False, True):
                    r = np.array([ret for ei, ret in sim(A, g, 0.07, reg) if lo <= A["day"][ei] <= hi])
                    eq = np.cumprod(1 + r) if len(r) else np.array([1.0])
                    mdd = (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min() if len(r) else 0.0
                    rows.append(dict(기간=pn, 범위="정규장만" if reg else "24시간", sym=s, gap=g, 복리=eq[-1] - 1, 낙폭=mdd, 건수=len(r)))
    t = pd.DataFrame(rows)
    g = t.groupby(["기간", "범위", "gap"], sort=False).agg(계좌=("복리", "mean"), 중앙=("복리", "median"), 최악낙폭=("낙폭", "min"), 건수=("건수", "sum")).reset_index()
    base = g[g.gap == 0.02].set_index(["기간", "범위"]).계좌
    g["2%대비"] = g.apply(lambda x: x.계좌 - base[(x.기간, x.범위)], axis=1)
    g["gap"] = g.gap.map("{:.1%}".format)
    for c in ("계좌", "중앙", "최악낙폭", "2%대비"):
        g[c] = g[c].map("{:+.1%}".format)
    pd.set_option("display.width", 200)
    print(g.to_string(index=False))
    s23 = t[(t.gap.isin([0.02, 0.023]))].pivot_table(index=["기간", "범위", "sym"], columns="gap", values="복리")
    s23["2.3%-2%"] = s23[0.023] - s23[0.02]
    print("\nETF별 2.3% - 2% 복리 차이:")
    print(s23["2.3%-2%"].unstack(["기간", "범위"]).map("{:+.1%}".format).to_string())
