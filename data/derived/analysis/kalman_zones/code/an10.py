import numpy as np, pandas as pd
from core import *
from evalcfg import evaluate, prep
from evalcfg2 import evaluate2
from grid_short import portfolio
S, common, sel = prep("long")
P = dict(tf=240, eth=False, L=21, norm="rms", n=50, a=0.75, b=1.75)
bh = portfolio([buy_hold(S[s]["o"], S[s]["c"], S[s]["is_reg"], S[s]["is_end"], S[s]["j0"]) for s in SYMS], S)
v0 = evaluate2("long", **P, stop_mode=0, buy_filter=0, dip_buy=0)[4]
v1 = evaluate2("long", **P, stop_mode=1, buy_filter=0, dip_buy=0)[4]
og = evaluate("long", 60, False, original=True)[4]
df = pd.DataFrame({"보유": bh, "새 규칙": v0, "새 규칙-손절없음": v1, "원본 60분": og}, index=pd.to_datetime(common))
m = df.resample("ME").last()
m = pd.concat([pd.DataFrame([[1.0] * 4], columns=m.columns, index=[pd.Timestamp("2025-12-31")]), m])
print((m.pct_change().dropna() * 100).round(1).to_string())
dd = (df / df.cummax() - 1).min() * 100
print("최대 낙폭(%):", dd.round(1).to_dict())
jul = df.loc["2026-06-30":"2026-08-10"]; print("6/30~8/10 최저점 대비:", ((jul.min() / jul.iloc[0] - 1) * 100).round(1).to_dict())
