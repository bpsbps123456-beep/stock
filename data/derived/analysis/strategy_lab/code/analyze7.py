"""Stage 7: parameter plateaus of the finalists, monthly consistency, regime-label
sensitivity (QQQ vs SMH vs fund universe), and walk-forward without the 1-stock rule."""

import numpy as np

import common as C
import data as D
import engine as E
import grid as G

M, P, meta = C.load_grid()
cols = {c: C.col(meta, c) for c in meta["metrics"]}
d = D.load()
m = E.build_market(d)
names, stack = E.build_scores(m)
ctx = G.Context(m)

print("== Parameter plateaus (total %, cost 0.10%, no overlay; rows N, cols rebalance; FW / buffer=N / exit none) ==")
for src, pre, sc in (("T", "all", "MOM120"), ("I", "all", "RADJ120"), ("I", "all", "PULL"), ("K", "all", "REV5"), ("T", "all", "W")):
    sid = meta["score_names"].index((src, pre, sc))
    print(f"-- {src}/{pre}/{sc}")
    print("   N  | " + "  ".join(f"{r:>6}" for r in meta["rebs"]) + " | no-buffer D  EQ D   MDD(D)")
    for N in meta["NS"]:
        vals = []
        for ri in range(4):
            i = np.where((P[:, 0] == sid) & (P[:, 1] == N) & (P[:, 2] == 1) & (P[:, 3] == N) & (P[:, 4] == ri) & (P[:, 5] == 0))[0][0]
            vals.append(M[i, 0, cols["total"]])
        i_nb = np.where((P[:, 0] == sid) & (P[:, 1] == N) & (P[:, 2] == 1) & (P[:, 3] == 0) & (P[:, 4] == 0))[0][0]
        i_eq = np.where((P[:, 0] == sid) & (P[:, 1] == N) & (P[:, 2] == 0) & (P[:, 3] == N) & (P[:, 4] == 0))[0][0]
        i_d = np.where((P[:, 0] == sid) & (P[:, 1] == N) & (P[:, 2] == 1) & (P[:, 3] == N) & (P[:, 4] == 0))[0][0]
        print(f"  {N:3d} | " + "  ".join(f"{v * 100:+5.0f}%" for v in vals)
              + f" |   {M[i_nb, 0, cols['total']] * 100:+5.0f}%   {M[i_eq, 0, cols['total']] * 100:+5.0f}%  {M[i_d, 0, cols['mdd']] * 100:5.0f}%")


def rule(src, score, N=5, wt=1, buf=None, reb=0, ex=0, pre="all"):
    return [names.index((src, pre, score)), N, wt, N if buf is None else buf, reb, ex, E.SOURCES.index(src)]


S = {"REPL_T": C.simulate(m, stack, rule("T", "W", N=500, buf=0))[0],
     "H6 T/MOM120 N5 D": C.simulate(m, stack, rule("T", "MOM120"))[0],
     "H6b T/MOM120 N5 W": C.simulate(m, stack, rule("T", "MOM120", reb=1))[0],
     "H6c T/MOM120 N10 M": C.simulate(m, stack, rule("T", "MOM120", N=10, reb=3))[0],
     "H7 I/RADJ120 N5 D": C.simulate(m, stack, rule("I", "RADJ120"))[0],
     "H9b I/PULL N5 D": C.simulate(m, stack, rule("I", "PULL"))[0],
     "H9c I/PULL N5 W": C.simulate(m, stack, rule("I", "PULL", reb=1))[0],
     "H10 K/REV5 N5 D": C.simulate(m, stack, rule("K", "REV5"))[0]}

print("\n== Monthly excess over REPL_T (%p) ==")
lr = {k: np.log1p(v) for k, v in S.items()}
mret = {k: np.expm1([lr[k][ctx.month_idx == i].sum() for i in range(9)]) for k in S}
print(" " * 20 + " ".join(f"{mm[5:]:>6}" for mm in ctx.months) + "  beat")
for k in S:
    if k == "REPL_T":
        continue
    ex = (mret[k] - mret["REPL_T"]) * 100
    print(f"{k:20s}" + " ".join(f"{x:+6.1f}" for x in ex) + f"  {int((ex > 0).sum())}/9")

# ---- regime labels on three bases ----
i0, _ = m.feat["ref_idx"]
Pall = m.feat["P"]
held_any = m.held["U"].any(axis=0)
with np.errstate(invalid="ignore", divide="ignore"):
    lr_all = np.log(Pall[1:] / Pall[:-1])
uni = np.concatenate([[0.0], np.nanmean(np.where(np.isfinite(lr_all[:, held_any]), lr_all[:, held_any], np.nan), axis=1)])
uni_idx = np.exp(np.nancumsum(uni))


def labels(series, thr):
    n = len(series)
    out = np.zeros(len(m.days), int)
    for t in range(len(m.days)):
        i = i0 + t
        r = series[min(n - 1, i + 10)] / series[max(0, i - 10)] - 1
        out[t] = 0 if r > thr else (2 if r < -thr else 1)
    return out


bases = {"QQQ ±3%": labels(m.ref["QQQ"], 0.03), "SMH ±6%": labels(m.ref["SMH"], 0.06),
         "펀드유니버스EW ±6%": labels(uni_idx, 0.06)}
for nm, lab in bases.items():
    cnt = np.bincount(lab, minlength=3)
    print(f"\n== Regime profile by {nm} labels (days up/box/down = {cnt.tolist()}), annualized log return ==")
    for k in S:
        vals = [lr[k][lab == g].sum() / max(1, (lab == g).sum()) * 252 * 100 for g in range(3)]
        print(f"  {k:20s} up {vals[0]:+6.0f}%  box {vals[1]:+6.0f}%  down {vals[2]:+6.0f}%")
    months_by = {g: sorted({m.days[t][:7] for t in range(len(m.days)) if lab[t] == g}) for g in (2,)}
    print(f"  down-labelled months: {months_by[2]}")

# ---- walk-forward over pre-registered rules without the 1-stock rule ----
cand = {k: v for k, v in S.items()}
extra = {"H1b T/W N5 D": rule("T", "W"), "H2 I/W N5 D": rule("I", "W"), "H3 T/FLOW5 N5 D": rule("T", "FLOW5"),
         "H5 T/NEW10 N5 D": rule("T", "NEW10"), "H8 I/WMOM60_50 N5 D": rule("I", "WMOM60_50"),
         "H9 K/PULL N5 D": rule("K", "PULL")}
for k, p in extra.items():
    cand[k] = C.simulate(m, stack, p)[0]
mon = {k: np.array([np.log1p(v)[ctx.month_idx == i].sum() for i in range(9)]) for k, v in cand.items()}
print("\n== Walk-forward over 14 pre-registered rules (no 1-stock rule), switch cost 0.2%/month ==")
for lb in (1, 2, 3, 99):
    for topk in (1, 3):
        real, picks = [], []
        for mi in range(1, 9):
            lo = max(0, mi - lb)
            ranked = sorted(cand, key=lambda k: -mon[k][lo:mi].sum())[:topk]
            picks.append("+".join(x.split()[0] for x in ranked))
            real.append(np.mean([np.expm1(mon[x][mi]) for x in ranked]) - 0.002)
        tot = np.prod(1 + np.array(real)) - 1
        print(f"  lookback {'all' if lb == 99 else str(lb) + 'm':>3} top{topk}: Feb-Sep {tot * 100:+6.1f}%  {' '.join(picks)}")
print(f"  static REPL_T Feb-Sep {np.expm1(mon['REPL_T'][1:].sum()) * 100:+.1f}%,  equal-weight all 14 rules Feb-Sep "
      f"{(np.prod([1 + np.mean([np.expm1(mon[k][i]) for k in cand]) for i in range(1, 9)]) - 1) * 100:+.1f}%")
