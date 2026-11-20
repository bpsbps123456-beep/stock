"""관문 A·B를 모두 넘은 규칙(펀드 10건 이상)의 강건성: 같은 날짜 군집, 날짜 평균을 뺀 결과, 분기 나눔, 조기 실전.
결과: results/robust_wide.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]


def mask_of(p, rule):
    m = np.ones(len(p), bool)
    for c in rule.split(" & "):
        if "≤" in c:
            f, v = c.split("≤")
            m &= (p[f].to_numpy(float) <= float(v))
        elif "≥" in c:
            f, v = c.split("≥")
            m &= (p[f].to_numpy(float) >= float(v))
        else:
            f, v = c.split("=")
            if f == "sector":
                m &= (p.sector == v).to_numpy()
            else:
                b = p[f].astype(bool).to_numpy()
                m &= b if v == "참" else ~b
    return m


def main() -> None:
    p = pd.read_csv(BASE / "results/panel_wide.csv")
    r = pd.read_csv(BASE / "results/rules_wide_pass.csv")
    r = r[(r.n_B >= 10) & (r.hit_B >= 0.65)]
    rd = p.reaction_date.astype(str)
    A = ((p.src == "wide") & ~p.fund_ticker & (rd >= "2026-07-01") & (rd <= "2026-10-21")).to_numpy()
    S = ((p.src == "wide") & (rd < "2026-07-01")).to_numpy()
    Bm = ((p.src == "fund") & (rd >= "2026-07-01") & (rd <= "2026-10-21")).to_numpy()
    L = ((p.src == "fund") & (rd >= "2026-10-22") & (rd <= "2026-11-12")).to_numpy()
    wk = pd.to_datetime(p.reaction_date).dt.strftime("%G-W%V").to_numpy()
    rows = []
    for x in r.itertuples():
        ent, h = x.family.split("_")[0], int(x.family.split("_")[1][1:])
        y = p[f"{ent}_h{h}"].to_numpy()
        # 같은 반응일(같은 표본 묶음) 평균을 뺀 결과: 같은 기간 시장·업종 공통 움직임 제거
        grp = np.where(S, "S", np.where(A, "A", np.where(Bm, "B", "x")))
        dm = pd.Series(y).groupby([grp, rd.to_numpy()]).transform("mean").to_numpy()
        yd = y - dm
        m = mask_of(p, x.rule)
        rec = {"family": x.family, "rule": x.rule}
        for nm, s in (("S", S), ("A", A), ("B", Bm)):
            sel = m & s & ~np.isnan(y)
            rec[f"n_{nm}"] = int(sel.sum())
            rec[f"hit_{nm}"] = round(float((y[sel] < 0).mean()), 3)
            rec[f"hit_{nm}_datedm"] = round(float((yd[sel] < 0).mean()), 3)
            rec[f"weeks_{nm}"] = int(len(set(wk[sel])))
            top = pd.Series(wk[sel]).value_counts()
            rec[f"topweek_share_{nm}"] = round(float(top.iloc[0] / sel.sum()), 2) if sel.sum() else np.nan
        for nm, lo, hi in (("A_Q3", "2026-07-01", "2026-09-30"), ("A_Q4", "2026-10-01", "2026-10-21")):
            sel = m & A & ~np.isnan(y) & (rd >= lo).to_numpy() & (rd <= hi).to_numpy()
            rec[f"n_{nm}"], rec[f"hit_{nm}"] = int(sel.sum()), round(float((y[sel] < 0).mean()), 3) if sel.sum() else np.nan
        rec["n_live_signal"] = int((m & L).sum())
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "results/robust_wide.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.T.to_string())


if __name__ == "__main__":
    main()
