import sys, numpy as np, pandas as pd
sys.path.insert(0, "code"); import grid as G
tot = ne_f = ne_b = ne_both = big = 0
for pat in ("A", "B"):
    for N in (60, 120, 240, 390):
        for tol in (0.0, 0.0005, 0.001):
            g = G.grid_table(f"cache/cand_yf1m/{pat}_{N}_{tol}.parquet", pat, N, tol, 20261014)
            tot += len(g); ne_f += int((g.f_n > 0).sum()); ne_b += int((g.b_n > 0).sum()); ne_both += int(((g.f_n>1)&(g.b_n>1)).sum()); big += int(((g.f_n+g.b_n)>=100).sum())
print("cells", tot, "front n>0", ne_f, "back n>0", ne_b, "both n>1", ne_both, "total n>=100", big)
g = G.grid_table("cache/cand_yf1m/A_120_0.0005.parquet", "A", 120, 0.0005, 20261014)
o = g[(g.K=="level3_2")&(g.ema=="persist")&(g.D==0.02)&(g.v==0)&(g.rm=="wick")&(g.r==0.5)&(g.m==1.0)&(g.above==False)]
print("orig_A cell rows", len(o), "signals after tradable/sane/valid/dedup: front", int(o.f_n.iloc[0]), "back", int(o.b_n.iloc[0]))
g = G.grid_table("cache/cand_yf1m/B_120_0.0005.parquet", "B", 120, 0.0005, 20261014)
o = g[(g.K=="k1")&(g.ema=="persist")&(g.D==0.02)&(g.v==0)&(g.below==True)&(g.eng=="basic")]
print("orig_B cell rows", len(o), "signals: front", int(o.f_n.iloc[0]), "back", int(o.b_n.iloc[0]))
