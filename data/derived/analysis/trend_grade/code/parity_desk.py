"""데스크 desk/trend.py 계산과 연구 FINAL(final.py)의 일치 대조. 같은 150종목·같은 기간, 보유 시점 대신 봉 130개 기준."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.insert(0, ROOT)
import harness as H
import fam_composite as FC
import fam_ladder as FL
from desk import trend

P = H.load_panel()
C = P["C"]
plain = lambda P: (P["C"].notna().cumsum() >= H.MIN_BARS) & P["C"].notna()
H.universe_mask = plain  # 보유 시점 대신 봉 기준(데스크와 같은 모집단)
score = FC.make(FC.GRID["E14"], 10)(P)
mask = plain(P)
res_pct = H.pct_rank(score, mask & score.notna())
res_grade = FL.hyb_grades(score, P, mask)

ind = {}
for s in C.columns:
    b = pd.DataFrame({"Open": P["O"][s], "High": P["H"][s], "Low": P["L"][s], "Close": C[s]}).dropna(subset=["Close"])
    ind[s] = trend.indicators(b)
cal = C.index[C.index >= "2026-01-02"]
cal_full = C.index[C.index >= "2025-10-01"]
pct, _, _ = trend.score_table(ind, cal_full)
grades = trend.grade_table(pct, ind, cal_full)
a, b = res_pct.loc[cal], pct.loc[cal].reindex(columns=res_pct.columns)
diff = (a - b).abs().max().max()
ga, gb = res_grade.loc[cal], grades.loc[cal].reindex(columns=res_grade.columns)
both = ga.notna() & gb.notna()
agree = float((ga[both] == gb[both]).sum().sum() / both.sum().sum())
print("max |pct diff|", diff, " grade agreement", round(agree * 100, 3), "%  cells", int(both.sum().sum()),
      " nan mismatch", int((ga.notna() != gb.notna()).sum().sum()))
d = (a - b).abs()
big = d.stack()
big = big[big > 1e-9]
print("cells differing", len(big), "of", int(a.notna().sum().sum()))
print(big.sort_values(ascending=False).head(8))
t, s = big.idxmax()
print("research", a.at[t, s], "desk", b.at[t, s], "raw score research", score.at[t, s])
