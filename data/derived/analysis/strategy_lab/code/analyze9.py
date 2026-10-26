"""Stage 9: (a) rebalance-date luck for monthly/weekly rules, (b) plateau-level
selection using Jan-May only -> Jun-Sep and Aug-Sep, under base and +1-day delay."""

import numpy as np

import common as C
import data as D
import engine as E
import grid as G

d = D.load()
m = E.build_market(d)
names, stack = E.build_scores(m)

# (a) phase luck: rebalance on the k-th decision day of each month / week
print("== (a) Rebalance-date luck: same rule, different day of the month/week ==")
dec_days = [t for t in range(len(m.days)) if m.dec_row[t] >= 0]


def phase_mask(period, k):
    mask = np.zeros(len(m.days), bool)
    groups = {}
    for t in dec_days:
        stamp = m.days[t][:7] if period == "M" else np.datetime64(m.days[t], "W")
        groups.setdefault(stamp, []).append(t)
    for ts in groups.values():
        if k < len(ts):
            mask[ts[k]] = True
    mask[0] = True
    return mask


for (s, sc, N, wt), period, ks in (((("T", "MOM120", 10, 1)), "M", range(0, 15, 2)),
                                   ((("T", "MOM120", 5, 0)), "M", range(0, 15, 2)),
                                   ((("T", "MOM120", 5, 0)), "W", range(5)),
                                   ((("T", "W", 10, 1)), "M", range(0, 15, 2))):
    vals = []
    for k in ks:
        m.reb["X"] = phase_mask(period, k)
        G.REBS = G.REBS + ("X",) if "X" not in G.REBS else G.REBS
        param = [names.index((s, "all", sc)), N, wt, N, G.REBS.index("X"), 0, E.SOURCES.index(s)]
        vals.append(np.prod(1 + C.simulate(m, stack, param)[0]) - 1)
    print(f"  {s}/{sc} N{N} {'FW' if wt else 'EQ'} {period}: " + " ".join(f"{v * 100:+.0f}%" for v in vals)
          + f"   (min {min(vals) * 100:+.0f}%, max {max(vals) * 100:+.0f}%)")

# (b) plateau selection on Jan-May only
M0, P, meta = C.load_grid()
cols = {c: C.col(meta, c) for c in meta["metrics"]}
key = np.array(["/".join(meta["score_names"][p[0]]) + "/" + meta["rebs"][p[4]] for p in P])
div = P[:, 1] >= 3
aug_cols = [cols["m8"], cols["m9"]]
jj_cols = [cols["m6"], cols["m7"]]
for tag, X in (("기본", M0), ("+1일 지연", np.load(C.GRID / "metrics_d1.npy", mmap_mode="r")),
               ("+1일·30bp", np.load(C.GRID / "metrics_d1c30.npy", mmap_mode="r"))):
    base = np.asarray(X[:, 0, :], dtype=np.float64)
    rows = []
    for k in np.unique(key[div]):
        sel = (key == k) & div
        rows.append((k, np.percentile(base[sel, cols["is_ret"]], 25), np.median(base[sel, cols["is_ret"]]),
                     np.median(base[sel, cols["oos_ret"]]), np.median(np.expm1(base[sel][:, jj_cols].sum(axis=1))),
                     np.median(np.expm1(base[sel][:, aug_cols].sum(axis=1)))))
    rows.sort(key=lambda r: -r[1])
    oos_all = np.median(base[div, cols["oos_ret"]])
    print(f"\n== (b) {tag}: rule plateaus ranked by Jan-May p25 only (N>=3). all-rule OOS median {oos_all * 100:+.1f}% ==")
    print("  rank rule                  | IS p25  IS med | Jun-Sep  Jun-Jul  Aug-Sep")
    for i, r in enumerate(rows[:12], 1):
        print(f"  {i:4d} {r[0]:22s}| {r[1] * 100:+6.0f}% {r[2] * 100:+6.0f}% | {r[3] * 100:+6.1f}% {r[4] * 100:+6.1f}% {r[5] * 100:+6.1f}%")
    top = rows[:10]
    print(f"  top-10 plateaus avg: Jun-Sep {np.mean([r[3] for r in top]) * 100:+.1f}%, Aug-Sep {np.mean([r[5] for r in top]) * 100:+.1f}%"
          f"  | all plateaus avg Jun-Sep {np.mean([r[3] for r in rows]) * 100:+.1f}%, Aug-Sep {np.mean([r[5] for r in rows]) * 100:+.1f}%")
