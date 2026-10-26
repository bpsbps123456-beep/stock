import pandas as pd, numpy as np
pd.set_option("display.width", 250)
L = pd.read_parquet("grid_long.parquet"); S = pd.read_parquet("grid_short.parquet")
def piv(df, val, **flt):
    g = df[df.kind == "zone5"]
    for k, v in flt.items(): g = g[g[k] == v]
    return g.pivot_table(index="a", columns="b", values=val, aggfunc="median")
print("LONG 240 RTH  median over L/norm/rules — SEL sharpe"); print(piv(L, "sel_sh", tf=240).round(2))
print("LONG 240 RTH — VAL return"); print((piv(L, "val_ret", tf=240)*100).round(1))
print("LONG all TF — VAL return by tf x rules (median)")
g = L[L.kind=="zone5"]; print((g.pivot_table(index="tf", columns=["tp_half","panic_hold"], values="val_ret", aggfunc="median")*100).round(1))
print("LONG all TF — SEL return by tf x rules (median)"); print((g.pivot_table(index="tf", columns=["tp_half","panic_hold"], values="sel_ret", aggfunc="median")*100).round(1))
print("LONG VAL return by tf x L (median)"); print((g.pivot_table(index="tf", columns="L", values="val_ret", aggfunc="median")*100).round(1))
print("LONG VAL return by tf x norm (median)"); print((g.pivot_table(index="tf", columns=["norm","n"], values="val_ret", aggfunc="median")*100).round(1))
print("SHORT 40 ETH — SEL sharpe (median)"); print(piv(S, "sel_sh", tf=40, eth=True).round(2))
print("SHORT 40 ETH — VAL return (median)"); print((piv(S, "val_ret", tf=40, eth=True)*100).round(1))
o = S[S.kind=="original"]; print("SHORT original all/val:"); print(o[["tf","eth","all_ret","all_sh","all_mdd","sel_ret","val_ret","trades","expo"]].round(3).to_string(index=False))
