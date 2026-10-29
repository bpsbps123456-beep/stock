import sys, numpy as np, pandas as pd
sys.path.insert(0, "code"); sys.path.insert(0, "audit/claude_skeptic")
import engine as E, ref as R
d = R.load("yf1m", "NVDA"); a = E.prep(d); r = R.per_symbol(d)
i = np.nonzero(a["mins"] == 570)[0][:3]
for j in i:
    print(j, "prev20 vol sum", d.Volume.iloc[j-20:j].sum(), "v", d.Volume.iloc[j], "engine volr", a["volr"][j], "ref volr", r["volr"][j])
print(d.Volume.iloc[i[0]-25:i[0]+1].tolist())
