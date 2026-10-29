"""cache/quotes.parquet(실측 NBBO 표본) → cost_table.json: 종목 × 시간대(pre·open5·early·late)의 반 호가 비율 중앙값.
시간대: pre = 09:30 전, open5 = 09:30~09:34, early = 09:35~09:59, late = 10:00 이후(호가 시각 = 그 분의 직전 NBBO)."""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
q = pd.read_parquet(os.path.join(ROOT, "cache/quotes.parquet")).dropna()
q = q[(q.ask > q.bid) & (q.bid > 0)]
q["hs"] = (q.ask - q.bid) / 2 / ((q.ask + q.bid) / 2)
t = pd.to_datetime(q.t)
m = t.dt.hour * 60 + t.dt.minute
q["bucket"] = np.select([m <= 570, m <= 575, m <= 600], ["pre", "open5", "early"], "late")   # 호가 시각 T = 봉 (T−1분) 종가 체결
tab, detail = {}, {}
for s, g in q.groupby("sym"):
    tab[s] = {b: float(g[g.bucket == b].hs.median()) for b in ["pre", "open5", "early", "late"]}
    detail[s] = {b: {"n": int((g.bucket == b).sum()), "p75": float(g[g.bucket == b].hs.quantile(0.75))} for b in tab[s]}
json.dump(tab, open(os.path.join(ROOT, "cost_table.json"), "w"), indent=1)
json.dump({"median_half_spread": tab, "detail": detail}, open(os.path.join(ROOT, "results/cost_detail.json"), "w"), indent=1)
print(json.dumps(tab, indent=1))
