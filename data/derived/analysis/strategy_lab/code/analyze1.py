"""Stage 1: benchmarks, brute-force leaderboard, target hit counts, marginal effects."""

import json

import numpy as np

import common as C
import data as D
import engine as E

M, P, meta = C.load_grid()
d = D.load()
m = E.build_market(d)
names, stack = E.build_scores(m)

bm = C.benchmarks(d, m, stack, names)
print("== Benchmarks (KRW) ==")
for k, v in bm.items():
    extra = f" hold {v['holdings']:.1f}" if "holdings" in v else ""
    sh = f" sharpe {v['sharpe']:.2f}" if "sharpe" in v else ""
    print(f"{k:18s} total {v['total'] * 100:+7.1f}%  mdd {v['mdd'] * 100:6.1f}%{sh}{extra}")
(D.CACHE / "benchmarks.json").write_text(json.dumps(bm, default=float, indent=1), encoding="utf-8")

tot = M[:, :, C.col(meta, "total")]
flat = np.asarray(tot).ravel()
etfT = bm["ETF_T_adj"]["total"]
etfK = bm["ETF_K_adj"]["total"]
blend = bm["ETF_blend"]["total"]
print(f"\n== Variants: {flat.size:,} ==")
print("total return percentiles %:", np.percentile(flat * 100, [1, 10, 25, 50, 75, 90, 99, 99.9]).round(1))
for nm, base in (("TIME ETF", etfT), ("KoAct ETF", etfK), ("50/50", blend)):
    for mult in (1, 2, 3):
        share = (flat >= mult * base).mean() * 100
        print(f"  >= {mult}x {nm} ({mult * base * 100:+.0f}%): {share:5.2f}% of variants")

print("\n== Top 25 by full-period total (cost 0.10%/side) ==")
order = np.argsort(-flat)[:25]
cols = {c: C.col(meta, c) for c in ("total", "mdd", "sharpe", "is_ret", "oos_ret", "holdings", "turnover")}
for rank, f in enumerate(order, 1):
    b, o = divmod(int(f), M.shape[1])
    row = M[b, o]
    print(f"{rank:2d} {row[cols['total']] * 100:+7.1f}% mdd {row[cols['mdd']] * 100:6.1f}% "
          f"sh {row[cols['sharpe']]:.2f} IS {row[cols['is_ret']] * 100:+6.1f}% OOS {row[cols['oos_ret']] * 100:+6.1f}% "
          f"hold {row[cols['holdings']]:.1f} | {C.label(meta, P[b], o)}")

print("\n== Marginal effect: median / p90 total by dimension (all overlays) ==")
src = np.array([meta["score_names"][s][0] for s in P[:, 0]])
pre = np.array([meta["score_names"][s][1] for s in P[:, 0]])
fam = np.array([C.score_family(meta["score_names"][s][2]) for s in P[:, 0]])
dims = {"source": src, "prefilter": pre, "family": fam, "N": P[:, 1],
        "weighting": np.array(meta["weightings"])[P[:, 2]],
        "buffer": np.where(P[:, 3] > 0, "yes", "no"), "rebalance": np.array(meta["rebs"])[P[:, 4]],
        "exit": np.array(meta["exits"])[P[:, 5]]}
tot_none = np.asarray(tot[:, 0])
for dim, vals in dims.items():
    print(f"-- {dim}")
    for v in sorted(set(vals.tolist()), key=str):
        sel = vals == v
        a = np.asarray(tot[sel]).ravel()
        print(f"   {str(v):8s} n={sel.sum():6d}  median {np.median(a) * 100:+6.1f}%  p90 {np.percentile(a, 90) * 100:+6.1f}%"
              f"  (no overlay median {np.median(tot_none[sel]) * 100:+6.1f}%)")
print("-- overlay")
for o, nm in enumerate(meta["overlays"]):
    a = np.asarray(tot[:, o])
    mdd = np.asarray(M[:, o, C.col(meta, "mdd")])
    print(f"   {nm:15s} median {np.median(a) * 100:+6.1f}%  p90 {np.percentile(a, 90) * 100:+6.1f}%  median MDD {np.median(mdd) * 100:6.1f}%"
          f"  exposure {np.median(M[:, o, C.col(meta, 'exposure')]):.2f}")
