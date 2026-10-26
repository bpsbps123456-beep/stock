"""Stage 5: leave-winner-out. Is the edge a rule, or one memory rally?

Compare each rule to following TIME's US book in the SAME reduced universe.
"""

import numpy as np

import common as C
import data as D
import engine as E

M0, P, meta = C.load_grid()
scen = {"full": (M0, ()),
        "xSNDK": (np.load(C.GRID / "metrics_xSNDK.npy", mmap_mode="r"), ("SNDK",)),
        "xMEM": (np.load(C.GRID / "metrics_xMEM.npy", mmap_mode="r"), ("SNDK", "WDC", "STX", "MU", "SKHY"))}
cols = {c: C.col(meta, c) for c in meta["metrics"]}
src = np.array([meta["score_names"][s][0] for s in P[:, 0]])
pre = np.array([meta["score_names"][s][1] for s in P[:, 0]])
score = np.array([meta["score_names"][s][2] for s in P[:, 0]])
reb = np.array(meta["rebs"])[P[:, 4]]
fam = np.array([C.score_family(s) for s in score])
key = np.array([f"{a}/{b}/{c}/{r}" for a, b, c, r in zip(src, pre, score, reb)])
d = D.load()

repl = {}
for sc, (_, excl) in scen.items():
    m = E.build_market(d, exclude=excl)
    names, stack = E.build_scores(m)
    for s in ("T", "K"):
        krw, _, _ = C.simulate(m, stack, [names.index((s, "all", "W")), 500, 1, 0, 0, 0, E.SOURCES.index(s)])
        repl[(sc, s)] = np.prod(1 + krw) - 1
    print(f"{sc:6s} REPL_T {repl[(sc, 'T')] * 100:+.1f}%  REPL_K {repl[(sc, 'K')] * 100:+.1f}%")

print("\n== Family median total (N>=3, no overlay, daily|weekly) and ratio to REPL_T of the same universe ==")
print("family | full D (x)      | xSNDK D (x)     | xMEM D (x)      || full W  | xSNDK W | xMEM W")
for f in sorted(set(fam)):
    parts, partsw = [], []
    for sc, (X, _) in scen.items():
        sel = (fam == f) & (P[:, 1] >= 3) & (reb == "D")
        v = np.median(np.asarray(X[sel, 0, cols["total"]]))
        parts.append(f"{v * 100:+5.0f}% ({(1 + v) / (1 + repl[(sc, 'T')]):.2f})")
        selw = (fam == f) & (P[:, 1] >= 3) & (reb == "W")
        partsw.append(f"{np.median(np.asarray(X[selw, 0, cols['total']])) * 100:+5.0f}%")
    print(f"{f:6s} | " + " | ".join(parts) + " || " + " | ".join(partsw))
print("(x) = (1+rule)/(1+REPL_T): wealth ratio vs following TIME's US book in the same universe")

print("\n== Selected rules across universes (median over N>=3 neighbours) ==")
rules = ["T/all/MOM120/D", "I/all/MOM120/D", "T/all/RADJ120/D", "I/all/RADJ120/D", "T/all/MOM120/M",
         "T/all/RADJ120/W", "I/all/PULL/D", "K/all/PULL/D", "U/all/PULL/D", "I/all/PULL/W", "K/all/REV5/D",
         "I/all/REV5/D", "T/all/W/D", "I/all/W/D", "T/all/WMOM60_50/D", "I/all/WMOM60_50/D",
         "T/all/MOM60/D", "T/all/MOM20/D", "I/all/MOM20/W", "T/all/FLOW5/D", "T/all/NEW10/D"]
print("rule                 | full           | xSNDK          | xMEM           | beat REPL_T in all 3?")
for rule in rules:
    out, beats = [], []
    for sc, (X, _) in scen.items():
        sel = (key == rule) & (P[:, 1] >= 3)
        v = np.median(np.asarray(X[sel, 0, cols["total"]]))
        mdd = np.median(np.asarray(X[sel, 0, cols["mdd"]]))
        out.append(f"{v * 100:+5.0f}% ({mdd * 100:3.0f}%)")
        beats.append(v > repl[(sc, "T")])
    print(f"{rule:20s} | " + " | ".join(out) + f" | {'YES' if all(beats) else 'no'}")

print("\n== Share of base rules (N>=3, no overlay) beating REPL_T of the same universe ==")
for sc, (X, _) in scen.items():
    t = np.asarray(X[:, 0, cols["total"]])
    sel = P[:, 1] >= 3
    print(f"{sc:6s}: {np.mean(t[sel] > repl[(sc, 'T')]) * 100:.1f}%   >=1.5x wealth: "
          f"{np.mean((1 + t[sel]) >= 1.5 * (1 + repl[(sc, 'T')])) * 100:.1f}%")
