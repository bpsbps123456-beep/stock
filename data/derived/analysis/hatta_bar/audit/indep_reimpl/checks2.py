"""Extra structural checks (counts only, no outcomes)."""
import glob, os
import numpy as np, pandas as pd
import detect as D

eng = pd.read_csv(os.path.join(D.ROOT, "cache", "engine_signals_yf1m.csv"))
diag = pd.read_csv(os.path.join(D.OUT, "diag.csv"))
cellp = {c["name"]: c for c in D.CELLS}

# 1) volume-ratio inf signals: time of day
inf = diag[diag.pass_c & np.isinf(diag.vr)]
print("vr=inf signals by cell/time:"); print(inf.groupby("cell").ts.apply(lambda s: sorted(set(x[11:] for x in s))).to_string())

# 2) Yahoo extended-hours volume: share of bars with volume 0 by session
tot = {"pre": [0, 0], "rth": [0, 0], "post": [0, 0]}
for f in sorted(glob.glob(os.path.join(D.DATA, "*.parquet"))):
    s = os.path.basename(f)[:-8]
    if s in ("SPY", "QQQ"):
        continue
    d = pd.read_parquet(f)
    mins = d.index.hour * 60 + d.index.minute
    for name, msk in (("pre", mins < 570), ("rth", (mins >= 570) & (mins < 960)), ("post", mins >= 960)):
        tot[name][0] += int(msk.sum()); tot[name][1] += int(((d.Volume.to_numpy() == 0) & msk).sum())
print("bars / zero-volume bars by session:", tot)

# 3) engine signal ET time-of-day distribution
eng["hm"] = eng.ts_et.str[11:]
def sess(hm):
    h, m = int(hm[:2]), int(hm[3:])
    x = h * 60 + m
    return "04-09:30" if x < 570 else ("09:30-11" if x < 660 else ("11-16" if x < 960 else "16-20"))
eng["sess"] = eng.hm.map(sess)
print(pd.crosstab(eng.cell, eng.sess))

# 4) 'S position' reading flips in touch cells (all candidates)
flips = []
for sym in sorted(set(diag.sym)):
    sub = diag[(diag.sym == sym) & diag.cell.isin(["alt_A1", "alt_B2"])]
    if sub.empty:
        continue
    d = D.load(sym); l = d["Low"].to_numpy(float)
    sws = {}
    for _, r in sub.iterrows():
        cp = cellp[r.cell]; k, N, p = int(r.k), cp["N"], cp["p"]
        others = r.tok and r.dok and r.vok and r.shape and r.hyg
        if not others:
            continue
        S = l[k - N:k - 1].min()
        if p not in sws: sws[p] = D.swing_mask(l, p)
        sw = sws[p]
        w = np.arange(k - N, k - 1)
        last_arg = w[np.nonzero(l[w] == S)[0][-1]]
        js = np.nonzero(sw[k - N:min(k - 2, k - 1 - p) + 1])[0] + (k - N)
        pts = list(l[js])
        if last_arg not in set(js): pts.append(S)
        tp = int(np.sum(np.array(pts) <= S * 1.003))
        if (tp >= cp["K"]) != (r.touch >= cp["K"]):
            flips.append((r.cell, sym, r.ts, r.touch, tp))
fl = pd.DataFrame(flips, columns=["cell", "sym", "ts", "touch_price", "touch_pos"])
print("K-condition flips under 'last-argmin position' reading:", fl.groupby("cell").size().to_dict())
print(fl.head(10).to_string())
