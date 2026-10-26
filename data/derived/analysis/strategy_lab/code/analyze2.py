"""Stage 2: overfitting diagnostics.

1. IS (Jan-May) -> OOS (Jun-Sep) decay of the IS winners, both directions.
2. Walk-forward: each month pick the trailing winner(s) among all variants, hold one month.
3. CSCV probability of backtest overfitting (Bailey et al. 2015), 10 blocks, 252 splits.
4. Deflated Sharpe ratio of the best full-period Sharpe.
"""

import itertools
import json
import math

import numpy as np
from scipy.stats import norm, spearmanr

import common as C
import data as D

M, P, meta = C.load_grid()
nB, nO, _ = M.shape
bm = json.loads((D.CACHE / "benchmarks.json").read_text(encoding="utf-8"))


def flat(name):
    return np.asarray(M[:, :, C.col(meta, name)], dtype=np.float64).ravel()


tot, is_r, oos_r = flat("total"), flat("is_ret"), flat("oos_ret")
is_s, oos_s = flat("is_sharpe"), flat("oos_sharpe")
N = tot.size
rng = np.random.default_rng(7)
sample = rng.choice(N, 200_000, replace=False)
print(f"variants {N:,}")
print(f"Spearman IS vs OOS total return (200k sample): {spearmanr(is_r[sample], oos_r[sample])[0]:+.3f}")
print(f"Spearman IS vs OOS Sharpe       (200k sample): {spearmanr(is_s[sample], oos_s[sample])[0]:+.3f}")

repl = {k: v for k, v in bm.items() if k.startswith("REPL") and k.endswith("0.001")}
# benchmark variants inside the grid for period-matched comparisons
def find(src, score, n=10, wt=1, buf=0, reb=0, ex=0, o=0):
    sid = meta["score_names"].index((src, "all", score))
    hit = np.where((P[:, 0] == sid) & (P[:, 1] == n) & (P[:, 2] == wt) & (P[:, 3] == buf)
                   & (P[:, 4] == reb) & (P[:, 5] == ex))[0][0]
    return hit * nO + o

ref_T10 = find("T", "W")  # TIME US top10, fund weights, daily
print(f"reference TIME top10 FW daily: IS {is_r[ref_T10] * 100:+.1f}%  OOS {oos_r[ref_T10] * 100:+.1f}%")
print(f"all-variant median: IS {np.median(is_r) * 100:+.1f}%  OOS {np.median(oos_r) * 100:+.1f}%")

print("\n== 1. IS winners -> OOS ==")
for metric_name, is_metric, oos_metric in (("return", is_r, oos_r), ("sharpe", is_s, oos_s)):
    order = np.argsort(-is_metric)
    for k in (1, 10, 100, 1000, 10000):
        top = order[:k]
        pct = (oos_r[:, None] < oos_r[top][None, :]).mean(axis=0) if k <= 100 else None
        print(f"  by IS {metric_name:6s} top{k:>6}: IS ret mean {is_r[top].mean() * 100:+7.1f}%  "
              f"OOS ret mean {oos_r[top].mean() * 100:+6.1f}%  median {np.median(oos_r[top]) * 100:+6.1f}%"
              + (f"  OOS pct-rank mean {pct.mean() * 100:.0f}" if pct is not None else ""))
print("  reverse (select on OOS period, test on IS period):")
order = np.argsort(-oos_r)
for k in (1, 10, 100, 1000, 10000):
    top = order[:k]
    print(f"  by OOS return top{k:>6}: OOS ret mean {oos_r[top].mean() * 100:+7.1f}%  -> IS ret mean {is_r[top].mean() * 100:+6.1f}%"
          f"  median {np.median(is_r[top]) * 100:+6.1f}%")

print("\n== 2. Walk-forward (monthly re-selection over all variants) ==")
mon = np.stack([flat(f"m{i}") for i in range(1, 10)], axis=1)  # log returns Jan..Sep
SWITCH = 2 * 0.001  # full turnover each month at 0.10%/side (conservative)
months = meta["months"]
ref_path = mon[ref_T10, 1:]
results = {}
for lookback in (1, 2, 3, 99):
    for k in (1, 10, 100, 1000):
        realized = []
        for mi in range(1, 9):  # Feb..Sep
            lo = max(0, mi - lookback)
            trail = mon[:, lo:mi].sum(axis=1)
            top = np.argpartition(-trail, k)[:k] if k < N else np.arange(N)
            nxt = np.expm1(mon[top, mi]).mean() - SWITCH
            realized.append(nxt)
        total = np.prod(1 + np.array(realized)) - 1
        results[(lookback, k)] = (total, realized)
        lb = "all" if lookback == 99 else f"{lookback}m"
        print(f"  lookback {lb:>3} top{k:>5}: Feb-Sep {total * 100:+7.1f}%   monthly "
              + " ".join(f"{x * 100:+5.1f}" for x in realized))
print(f"  reference TIME top10 FW daily Feb-Sep: {np.expm1(ref_path.sum()) * 100:+.1f}%  monthly "
      + " ".join(f"{x * 100:+5.1f}" for x in np.expm1(ref_path)))
print(f"  all-variant median Feb-Sep: {np.median(np.expm1(mon[:, 1:].sum(axis=1))) * 100:+.1f}%")

print("\n== 3. CSCV probability of backtest overfitting ==")
bs = np.stack([flat(f"bs{i}") for i in range(10)], axis=1).astype(np.float32)
bq = np.stack([flat(f"bq{i}") for i in range(10)], axis=1).astype(np.float32)
edges = meta["block_edges"]
sizes = np.diff(edges).astype(np.float32)
splits = list(itertools.combinations(range(10), 5))
for metric_name in ("return", "sharpe"):
    logits, oos_pct = [], []
    for chunk in range(0, len(splits), 12):
        group = splits[chunk:chunk + 12]
        mask = np.zeros((10, len(group)), np.float32)
        for gi, s in enumerate(group):
            mask[list(s), gi] = 1
        is_sum, oos_sum = bs @ mask, bs @ (1 - mask)
        if metric_name == "return":
            is_m, oos_m = is_sum, oos_sum
        else:
            n_is, n_oos = sizes @ mask, sizes @ (1 - mask)
            def sh(s, q, n):
                mu = s / n
                var = np.maximum(q / n - mu ** 2, 1e-12)
                return mu / np.sqrt(var)
            is_m, oos_m = sh(is_sum, bq @ mask, n_is), sh(oos_sum, bq @ (1 - mask), n_oos)
        best = is_m.argmax(axis=0)
        for gi in range(len(group)):
            v = oos_m[best[gi], gi]
            w = (oos_m[:, gi] < v).mean()
            w = min(max(w, 1e-6), 1 - 1e-6)
            oos_pct.append(w)
            logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    print(f"  metric {metric_name:6s}: PBO = {np.mean(logits <= 0) * 100:.1f}%   "
          f"median OOS percentile of IS-best = {np.median(oos_pct) * 100:.0f}   (splits {len(splits)})")

print("\n== 4. Deflated Sharpe ratio of the best full-period Sharpe ==")
sh_ann = flat("sharpe")
T = len(meta["days"]) - 1
sr = sh_ann / math.sqrt(252)
best = int(np.argmax(sr))
b, o = divmod(best, nO)
print("  best Sharpe variant:", C.label(meta, P[b], o), f"annualized {sh_ann[best]:.2f}, total {tot[best] * 100:+.1f}%")
var_sr = sr.var()
g = 0.5772156649
# skew/kurtosis of the best variant's daily returns
import engine as E
d = D.load(); m = E.build_market(d); names, stack = E.build_scores(m)
ov = meta["overlays"][o]
from grid import market_overlays
mk = market_overlays(m, d)
krw, _, _ = C.simulate(m, stack, list(P[b]), overlay=(ov if ov.startswith("SELF") else mk[ov]))
r = krw[1:]
g3 = ((r - r.mean()) ** 3).mean() / r.std() ** 3
g4 = ((r - r.mean()) ** 4).mean() / r.std() ** 4
srb = r.mean() / r.std()
for n_trials, label in ((N, "all variants"), (2000, "~2000 effective"), (200, "~200 effective")):
    sr0 = math.sqrt(var_sr) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * math.e)))
    z = (srb - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * srb + (g4 - 1) / 4 * srb ** 2)
    print(f"  trials {label:16s}: expected max Sharpe under null {sr0 * math.sqrt(252):.2f} (ann.)  DSR = {norm.cdf(z):.3f}")
