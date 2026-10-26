import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import frontier as FR
A = FR.A
raw = pd.read_parquet(A.PANEL); raw["ts"] = pd.to_datetime(raw["ts"])
def show(sym, a, b):
    g = FR.per_symbol(raw[raw.sym == sym])
    R = g["rsi"].to_numpy()
    for k, c in (("K3n", "T3_20"), ("K0", "T0_20"), ("K0_10", "T0_10")):
        g[k] = A.stage5(R, g[c].to_numpy(), 0.5)
    g["C20"] = g["C"].shift(20)
    g["MA20"] = np.exp(np.log(g["C"]).rolling(20).mean())
    print(sym); print(g[(g.date >= a) & (g.date <= b)][["date", "C", "C20", "MA20", "rsi", "T0_20", "T3_20", "K0", "K3n", "K0_10"]].round(3).to_string(index=False))
show("SMR", "2025-08-25", "2025-10-08")
show("CRCL", "2025-12-01", "2026-01-12")
show("LUNR", "2025-02-20", "2025-05-05")
show("SNDK", "2025-11-20", "2026-01-12")
