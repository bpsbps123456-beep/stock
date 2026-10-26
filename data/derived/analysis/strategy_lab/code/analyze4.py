"""Stage 4: do the robust rule families survive costs, open execution, and
contribution concentration?"""

import json

import numpy as np

import common as C
import data as D
import engine as E

M0, P, meta = C.load_grid()
scen = {"c0": np.load(C.GRID / "metrics_c0.npy", mmap_mode="r"),
        "c10": M0,
        "c25": np.load(C.GRID / "metrics_c25.npy", mmap_mode="r"),
        "open": np.load(C.GRID / "metrics_open.npy", mmap_mode="r")}
cols = {c: C.col(meta, c) for c in meta["metrics"]}
src = np.array([meta["score_names"][s][0] for s in P[:, 0]])
pre = np.array([meta["score_names"][s][1] for s in P[:, 0]])
score = np.array([meta["score_names"][s][2] for s in P[:, 0]])
reb = np.array(meta["rebs"])[P[:, 4]]
key = np.array([f"{a}/{b}/{c}/{r}" for a, b, c, r in zip(src, pre, score, reb)])
plateaus = json.load(open(D.CACHE / "rule_plateaus.json"))

print("== Rule plateaus under cost / execution scenarios (median over N>=3 neighbours, no overlay) ==")
print("rule                        | total c0 / c10 / c25 / open(c10) | OOS c10 / c25 / open | turnover")
watch = [r["rule"] for r in plateaus[:30]]
extra = ["T/all/W/D", "T/all/W/M", "I/all/W/D", "K/all/W/D", "U/all/PULL/D", "T/all/PULL/D",
         "T/all/REV5/D", "I/all/PULL/W", "K/all/PULL/W", "K/all/REV5/W"]
for rule in watch + [e for e in extra if e not in watch]:
    sel = (key == rule) & (P[:, 1] >= 3)
    if not sel.any():
        continue
    t = {s: np.median(np.asarray(X[sel, 0, cols["total"]])) for s, X in scen.items()}
    oos = {s: np.median(np.asarray(X[sel, 0, cols["oos_ret"]])) for s, X in scen.items()}
    turn = np.median(np.asarray(M0[sel, 0, cols["turnover"]]))
    print(f"{rule:27s} | {t['c0'] * 100:+5.0f}% {t['c10'] * 100:+5.0f}% {t['c25'] * 100:+5.0f}% {t['open'] * 100:+5.0f}%"
          f"           | {oos['c10'] * 100:+4.0f}% {oos['c25'] * 100:+4.0f}% {oos['open'] * 100:+4.0f}% | {turn:5.0f}")

print("\n== Family medians by scenario (N>=3, no overlay) ==")
fam = np.array([C.score_family(s) for s in score])
for f in sorted(set(fam)):
    sel = (fam == f) & (P[:, 1] >= 3) & (reb == "D")
    selw = (fam == f) & (P[:, 1] >= 3) & (reb == "W")
    line = " ".join(f"{s}:{np.median(np.asarray(X[sel, 0, cols['total']])) * 100:+4.0f}%" for s, X in scen.items())
    linew = " ".join(f"{s}:{np.median(np.asarray(X[selw, 0, cols['total']])) * 100:+4.0f}%" for s, X in scen.items())
    print(f"{f:6s} daily  {line}   | weekly {linew}")

print("\n== Contribution concentration of representative rules (N=5, FW, buffer=N, cost 0.10%) ==")
d = D.load()
m = E.build_market(d)
names, stack = E.build_scores(m)
reps = [("T", "all", "MOM120", "D"), ("I", "all", "MOM120", "D"), ("T", "all", "RADJ120", "D"),
        ("K", "all", "REV5", "D"), ("I", "all", "REV5", "D"), ("K", "all", "PULL", "D"),
        ("I", "all", "PULL", "D"), ("T", "all", "W", "D")]
for s, p_, sc, rb in reps:
    sid = names.index((s, p_, sc))
    param = [sid, 5, 1, 5, meta["rebs"].index(rb), 0, E.SOURCES.index(s)]
    W, gross, turn = C.weights_path(m, stack, param)
    contrib = C.contributions(m, W)
    total_usd = np.prod(1 + gross - 0.001 * turn) - 1
    vals = np.array(list(contrib.values()))
    top = list(contrib.items())[:5]
    pos = np.sort(vals)[::-1]
    share_top3 = pos[:3].sum() / vals.sum() if vals.sum() > 0 else np.nan
    names_held = (W > 0).sum(axis=0)
    print(f"{s}/{p_}/{sc}/{rb} N5: USD total {total_usd * 100:+.0f}%, sum of contributions {vals.sum() * 100:+.0f}%p, "
          f"top-3 winners = {share_top3 * 100:.0f}% of net, distinct names {int((names_held > 0).sum())}")
    print("    largest |contrib|: " + ", ".join(f"{k} {v * 100:+.0f}%p" for k, v in top))
    # robustness: drop the single best contributor's days (set its returns to 0) and rerun
    best = max(contrib, key=contrib.get)
    j = m.tickers.index(best)
    saved = m.R[:, j].copy()
    m.R[:, j] = 0.0
    W2, g2, t2 = C.weights_path(m, stack, param)
    m.R[:, j] = saved
    print(f"    without {best}'s price moves: USD total {(np.prod(1 + g2 - 0.001 * t2) - 1) * 100:+.0f}%")
