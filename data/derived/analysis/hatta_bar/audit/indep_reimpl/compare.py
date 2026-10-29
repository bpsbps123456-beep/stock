"""Compare my signal sets with the engine dump; agreement counts only (no outcomes)."""
import os
import sys

import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(OUT))
eng = pd.read_csv(os.path.join(ROOT, "cache", "engine_signals_yf1m.csv"))
mode = sys.argv[1] if len(sys.argv) > 1 else "a"
mine = pd.read_csv(os.path.join(OUT, f"mine_signals_{mode}.csv"))
cells = ["orig_A", "orig_B", "alt_A1", "alt_A2", "alt_B1", "alt_B2"]
rows = []
mism = []
for c in cells:
    e = eng[eng.cell == c]
    m = mine[mine.cell == c]
    es = set(zip(e.sym, e.ts_et))
    ms = set(zip(m.sym, m.ts))
    rows.append(dict(cell=c, engine_n=len(es), mine_n=len(ms), matched=len(es & ms),
                     engine_only=len(es - ms), mine_only=len(ms - es)))
    for s, t in sorted(es - ms):
        mism.append(dict(cell=c, side="engine_only", sym=s, ts=t))
    for s, t in sorted(ms - es):
        mism.append(dict(cell=c, side="mine_only", sym=s, ts=t))
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(mism).to_csv(os.path.join(OUT, f"mismatch_{mode}.csv"), index=False)
print(len(mism), "mismatches")
