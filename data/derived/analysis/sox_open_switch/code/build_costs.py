"""cache/quotes.parquet(실측 NBBO 표본) → cost_table.json: 종목 × 시간대(pre·open5·early·late)의 반 호가 비율 중앙값.
시간대: pre = 09:30 전, open5 = 09:30~09:34, early = 09:35~09:59, late = 10:00 이후(호가 시각 = 그 분의 직전 NBBO).
사용: py build_costs.py [입력이름 출력표 출력세부] — BRIEF7: py build_costs.py quotes_2016 cost_table_2016.json cost_detail_2016.json"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC, TAB, DET = (sys.argv[1], sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("quotes", "cost_table.json", "cost_detail.json")
q = pd.read_parquet(os.path.join(ROOT, f"cache/{SRC}.parquet")).dropna()
q = q[(q.ask > q.bid) & (q.bid > 0)]
q["hs"] = (q.ask - q.bid) / 2 / ((q.ask + q.bid) / 2)
t = pd.to_datetime(q.t)
m = t.dt.hour * 60 + t.dt.minute
q["bucket"] = np.select([m <= 570, m <= 575, m <= 600], ["pre", "open5", "early"], "late")   # 호가 시각 T = 봉 (T−1분) 종가 체결
tab, detail = {}, {}
for s, g in q.groupby("sym"):
    tab[s] = {b: float(g[g.bucket == b].hs.median()) for b in ["pre", "open5", "early", "late"]}
    detail[s] = {b: {"n": int((g.bucket == b).sum()), "p75": float(g[g.bucket == b].hs.quantile(0.75))} for b in tab[s]}
json.dump(tab, open(os.path.join(ROOT, TAB), "w"), indent=1)
json.dump({"median_half_spread": tab, "detail": detail}, open(os.path.join(ROOT, "results", DET), "w"), indent=1)
print(json.dumps(tab, indent=1))
