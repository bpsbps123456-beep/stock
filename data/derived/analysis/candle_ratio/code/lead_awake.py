"""반박 갈래가 찾은 사후 단서(정규장 60분 N5 body fade 0.5/0.5 밤 보유)를 144종목에서 이상적 체결과
깨어 있음 체결(08~24시 KST에만, 자는 동안 신호는 다음 깨어 있는 봉 시가)로 잰다. 비용 0과 편도 0.05%.
py code/lead_awake.py → results/lead_awake.csv"""
import numpy as np
import pandas as pd

from engine import *

SYMS = sorted(p.stem[:-3] for p in CACHE.glob("*_5m.parquet"))
TF, N, KIND, B, S = 60, 5, "body", 0.5, 0.5
rows = []
for sym in SYMS:
    tb = tf_bars(load(sym, "5m"), TF, "rth")
    tb = tb[tb.day >= WIN["5m"]["warm"]].reset_index(drop=True)
    O, C, V = (tb[k].values.astype(np.float64) for k in ("O", "C", "V"))
    seg = segments(tb.day.values, WIN["5m"])
    p = ratio(O, C, V, N, KIND)
    bh_sel, bh_conf = bh(O, C, seg)
    r = {"sym": sym, "bh_sel": bh_sel, "bh_conf": bh_conf}
    for mode, aw in (("ideal", np.ones(len(tb), np.bool_)), ("awake", awake_mask(tb.start))):
        for cost in (0.0, 0.0005):
            out = np.zeros(11)
            sim(O, C, p, tb.lod.values, seg, aw, B, S, False, False, cost, out)
            k = f"{mode}_{cost}"
            r[k + "_sel"], r[k + "_conf"] = out[0], out[1]
            r[k + "_ex_sel"] = out[0] - out[4] / out[6] * bh_sel
            r[k + "_ex_conf"] = out[1] - out[5] / out[7] * bh_conf
            r[k + "_ent"] = out[2] + out[3]
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(HERE / "results" / "lead_awake.csv", index=False)
pc = lambda v: round(float(np.expm1(v) * 100), 1)
print("종목", len(df), "보유 중앙 고르기/확인", pc(df.bh_sel.median()), pc(df.bh_conf.median()))
for k in ("ideal_0.0", "ideal_0.0005", "awake_0.0", "awake_0.0005"):
    print(k, "수익 중앙", pc(df[k + "_sel"].median()), pc(df[k + "_conf"].median()),
          "| 초과 중앙", pc(df[k + "_ex_sel"].median()), pc(df[k + "_ex_conf"].median()),
          "| 초과+ 비율", round((df[k + "_ex_sel"] > 0).mean(), 2), round((df[k + "_ex_conf"] > 0).mean(), 2),
          "| 보유 이김", round((df[k + "_conf"] > df.bh_conf).mean(), 2), "| 진입 중앙", df[k + "_ent"].median())
mu = df[df.sym == "MU"].iloc[0]
print("MU", {k: pc(mu[k]) for k in df.columns if k.endswith(("_sel", "_conf")) and not k.startswith("bh")}, "보유", pc(mu.bh_sel), pc(mu.bh_conf))
