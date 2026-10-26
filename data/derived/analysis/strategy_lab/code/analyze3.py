"""Stage 3: robustness at the rule-family level and regime behaviour.

A rule is robust when it beats following TIME's US book (REPL_T) in both halves,
in most months, and when its parameter neighbours do too (a plateau, not a spike).
"""

import json

import numpy as np

import common as C
import data as D
import engine as E
import grid as G

M, P, meta = C.load_grid()
d = D.load()
m = E.build_market(d)
names, stack = E.build_scores(m)
ctx = G.Context(m)

# ---- reference: follow each fund's full US book (fund weights, daily) ----
def ref_stats(src):
    sid = names.index((src, "all", "W"))
    krw, _, _ = C.simulate(m, stack, [sid, 500, 1, 0, 0, 0, E.SOURCES.index(src)])
    lr = np.log1p(krw)
    out = {"total": np.expm1(lr.sum()), "is": np.expm1(lr[:ctx.is_end + 1].sum()),
           "oos": np.expm1(lr[ctx.is_end + 1:].sum()),
           "months": np.array([lr[ctx.month_idx == i].sum() for i in range(9)]),
           "reg": np.array([lr[ctx.regime == g].sum() for g in range(3)])}
    return out

REF = {s: ref_stats(s) for s in ("T", "K")}
for s, r in REF.items():
    print(f"REPL_{s}: total {r['total'] * 100:+.1f}%  IS {r['is'] * 100:+.1f}%  OOS {r['oos'] * 100:+.1f}%  months "
          + " ".join(f"{x * 100:+.1f}" for x in np.expm1(r['months'])))
reg_days = np.bincount(ctx.regime, minlength=3)
print("regime days up/box/down:", reg_days.tolist())
for s, r in REF.items():
    print(f"REPL_{s} annualized log return by regime up/box/down: "
          + " ".join(f"{x / n * 252 * 100:+.0f}%" for x, n in zip(r["reg"], reg_days)))

o = 0  # no overlay: base rules only
cols = {c: C.col(meta, c) for c in meta["metrics"]}
base = np.asarray(M[:, o, :], dtype=np.float64)
tot, is_r, oos_r = base[:, cols["total"]], base[:, cols["is_ret"]], base[:, cols["oos_ret"]]
mon = base[:, [cols[f"m{i}"] for i in range(1, 10)]]
reg = base[:, [cols["reg_up"], cols["reg_box"], cols["reg_down"]]]
ref = REF["T"]
beat_both = (is_r > ref["is"]) & (oos_r > ref["oos"])
beat_months = (mon > ref["months"][None, :]).sum(axis=1)
print(f"\nbase rules {len(base):,}: beat REPL_T in both halves {beat_both.mean() * 100:.1f}%, "
      f">=6/9 months {(beat_months >= 6).mean() * 100:.1f}%, both {((beat_months >= 6) & beat_both).mean() * 100:.1f}%")

# ---- family table: (source, prefilter, score, rebalance) plateaus over N/weighting/buffer/exit ----
src = np.array([meta["score_names"][s][0] for s in P[:, 0]])
pre = np.array([meta["score_names"][s][1] for s in P[:, 0]])
score = np.array([meta["score_names"][s][2] for s in P[:, 0]])
reb = np.array(meta["rebs"])[P[:, 4]]
key = np.array([f"{a}/{b}/{c}/{r}" for a, b, c, r in zip(src, pre, score, reb)])
rows = []
for k in np.unique(key):
    sel = (key == k) & (P[:, 1] >= 3)  # N in {3,5,7,10}: diversified enough to be investable
    if sel.sum() == 0:
        continue
    rows.append({
        "rule": k, "n": int(sel.sum()),
        "med_total": np.median(tot[sel]), "p25_total": np.percentile(tot[sel], 25),
        "med_is": np.median(is_r[sel]), "med_oos": np.median(oos_r[sel]),
        "share_beat_both": beat_both[sel].mean(), "med_beat_months": np.median(beat_months[sel]),
        "med_mdd": np.median(base[sel, cols["mdd"]]), "med_sharpe": np.median(base[sel, cols["sharpe"]]),
        "reg_ann": [np.median(reg[sel, g]) / reg_days[g] * 252 for g in range(3)],
        "med_turnover": np.median(base[sel, cols["turnover"]]),
    })
rows.sort(key=lambda r: -r["p25_total"])
print("\n== Rule plateaus (N>=3), ranked by 25th-percentile total across neighbours ==")
print("rule                         | p25tot  medtot | medIS  medOOS | beat-both beatM | medMDD Sharpe | up/box/down ann.  | turn")
for r in rows[:40]:
    print(f"{r['rule']:28s} | {r['p25_total'] * 100:+6.0f}% {r['med_total'] * 100:+6.0f}% | {r['med_is'] * 100:+5.0f}% {r['med_oos'] * 100:+6.0f}% |"
          f"   {r['share_beat_both'] * 100:4.0f}%   {r['med_beat_months']:3.0f}  | {r['med_mdd'] * 100:5.0f}% {r['med_sharpe']:5.2f} | "
          + "/".join(f"{x * 100:+.0f}" for x in r["reg_ann"]) + f" | {r['med_turnover']:.0f}")
json.dump(rows, open(D.CACHE / "rule_plateaus.json", "w"), default=float)

# ---- family-level regime profile (N>=3, no overlay) ----
fam = np.array([C.score_family(s) for s in score])
print("\n== Score family x regime (median annualized log return, N>=3, no overlay) ==")
print("family  | up      box     down   | total  | beat-both")
for f in sorted(set(fam)):
    sel = (fam == f) & (P[:, 1] >= 3)
    ra = [np.median(reg[sel, g]) / reg_days[g] * 252 * 100 for g in range(3)]
    print(f"{f:7s} | {ra[0]:+6.0f}% {ra[1]:+6.0f}% {ra[2]:+6.0f}% | {np.median(tot[sel]) * 100:+5.0f}% | {beat_both[sel].mean() * 100:4.0f}%")
print("REPL_T  | " + " ".join(f"{x / n * 252 * 100:+6.0f}%" for x, n in zip(REF['T']['reg'], reg_days)))

print("\n== Source x rebalance (median total, N>=3, no overlay) ==")
for s in E.SOURCES:
    print(s, " ".join(f"{rb}:{np.median(tot[(src == s) & (reb == rb) & (P[:, 1] >= 3)]) * 100:+.0f}%" for rb in meta["rebs"]))
