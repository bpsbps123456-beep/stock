"""Classify K-flips under the alternative 'S position' reading (counts only)."""
import os
import numpy as np, pandas as pd
import detect as D
diag = pd.read_csv(os.path.join(D.OUT, "diag.csv"))
cellp = {c["name"]: c for c in D.CELLS}
cls = []
for sym in sorted(set(diag.sym)):
    sub = diag[(diag.sym == sym) & diag.cell.isin(["alt_A1", "alt_B2"])]
    d = D.load(sym); l = d["Low"].to_numpy(float); sws = {}
    for _, r in sub.iterrows():
        cp = cellp[r.cell]; k, N, p = int(r.k), cp["N"], cp["p"]
        if not (r.tok and r.dok and r.vok and r.shape and r.hyg):
            continue
        S = l[k - N:k - 1].min()
        if p not in sws: sws[p] = D.swing_mask(l, p)
        sw = sws[p]
        w = np.arange(k - N, k - 1)
        ties = w[l[w] == S]
        last_arg = ties[-1]
        js = np.nonzero(sw[k - N:min(k - 2, k - 1 - p) + 1])[0] + (k - N)
        if last_arg in set(js):
            continue
        pts = list(l[js]) + [S]
        tp = int(np.sum(np.array(pts) <= S * 1.003))
        if (tp >= cp["K"]) == (r.touch >= cp["K"]):
            continue
        if last_arg > k - 1 - p:
            kind = "latest exact re-test unconfirmed"
        elif l[last_arg - 1] == S:
            kind = "adjacent tie (left neighbour equal)"
        else:
            kind = "other"
        cls.append((r.cell, kind))
print(pd.DataFrame(cls, columns=["cell", "kind"]).value_counts().to_string())
