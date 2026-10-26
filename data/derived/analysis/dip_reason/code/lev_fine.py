"""3배 레버리지 이격을 2.0~3.0% 사이에서 촘촘히(사용자 10/9 질문: 2.3%는?). lev_study.py 규칙 그대로, 전날 +7% 포함.
2% 대비 합 차이, 날짜별 t, 2%보다 나은 ETF 수, 가장 이득 본 하루를 뺀 값. 사용: py lev_fine.py <raw 폴더>"""
import sys
from datetime import date
import numpy as np, pandas as pd
from lev_study import LEV, arrays, sim

GAPS = [0.020, 0.021, 0.022, 0.023, 0.024, 0.025, 0.027, 0.030]
P = {"앞(5분봉)": (5, 36, 9, date(2026, 7, 16), date(2026, 9, 8)), "뒤(3분봉)": (3, 60, 14, date(2026, 9, 11), date(2026, 10, 7))}

if __name__ == "__main__":
    raw = sys.argv[1]
    rows = []
    for pn, (mins, el, rl, lo, hi) in P.items():
        for s in LEV:
            d, A = arrays(raw, s, mins, el, rl)
            for g in GAPS:
                for reg in (False, True):
                    for ei, ret in sim(A, g, 0.07, reg):
                        if lo <= A["day"][ei] <= hi:
                            rows.append((pn, "정규장만" if reg else "24시간", s, g, A["day"][ei], ret))
    t = pd.DataFrame(rows, columns=["기간", "범위", "sym", "gap", "day", "ret"])
    out = []
    for (pn, sc), v in t.groupby(["기간", "범위"], sort=False):
        b = v[v.gap == 0.02]
        bd, bs = b.groupby("day").ret.sum(), b.groupby("sym").ret.sum()
        days = sorted(set(v.day))
        for g in GAPS:
            x = v[v.gap == g]
            dd = x.groupby("day").ret.sum().reindex(days, fill_value=0) - bd.reindex(days, fill_value=0)
            tt = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if dd.std(ddof=1) > 0 else 0
            es = x.groupby("sym").ret.sum().reindex(LEV, fill_value=0) - bs.reindex(LEV, fill_value=0)
            out.append(dict(기간=pn, 범위=sc, 이격=f"{g:.1%}", 건수=len(x), 합=x.ret.sum(), 차이=dd.sum(), t=tt,
                            나은ETF=int((es > 1e-9).sum()), 나쁜ETF=int((es < -1e-9).sum()),
                            최대하루뺀차이=dd.sum() - dd.max() if dd.sum() > 0 else dd.sum() - dd.min(),
                            SOXL=es["SOXL"], SOXS=es["SOXS"]))
    r = pd.DataFrame(out)
    pd.set_option("display.width", 220)
    print(r.round(2).to_string(index=False))
    piv = r.pivot_table(index="이격", columns=["기간", "범위"], values="차이", sort=False).round(2)
    print("\n2% 대비 합 차이\n" + piv.to_string())
