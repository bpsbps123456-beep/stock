"""Real-time regime classifiers and regime-dependent rule switching.

A classifier sees closes up to the previous US day only and returns one label per
backtest day: 0 = bull (상승장), 1 = box (박스장), 2 = bear (하락장).
The regime -> rule map is chosen on the first half (Jan-May) and tested on Jun-Sep.
"""

from __future__ import annotations

import numpy as np

import common as C
import data as D
import engine as E
import grid as G

BULL, BOX, BEAR = 0, 1, 2


def _trend_labels(m, x):
    """Price vs 50-day MA, confirmed by the 20-day MA vs 50-day MA."""
    i0, _ = m.feat["ref_idx"]
    out = np.full(len(m.days), BOX)
    for t in range(len(m.days)):
        i = i0 + t
        last, ma20, ma50 = x[i - 1], np.mean(x[i - 20:i]), np.mean(x[i - 50:i])
        if last > ma50 and ma20 > ma50:
            out[t] = BULL
        elif last < ma50 and ma20 < ma50:
            out[t] = BEAR
    return out


def classify_qqq(m):
    return _trend_labels(m, m.ref["QQQ"])


def classify_smh(m):
    return _trend_labels(m, m.ref["SMH"])


def classify_breadth(m):
    """Share of either fund's US holdings above their 20-day MA: >60% bull, <40% bear."""
    i0, _ = m.feat["ref_idx"]
    P = m.feat["P"]
    out = np.full(len(m.days), BOX)
    row = 0
    for t in range(len(m.days)):
        if m.dec_row[t] >= 0:
            row = m.dec_row[t]
        i = i0 + t
        cols = np.where(m.held["U"][row])[0]
        last, ma = P[i - 1, cols], np.nanmean(P[i - 20:i, cols], axis=0)
        ok = ~np.isnan(last) & ~np.isnan(ma)
        share = np.mean(last[ok] > ma[ok])
        out[t] = BULL if share > 0.6 else (BEAR if share < 0.4 else BOX)
    return out


def switch_returns(labels, mapping, series, cost=G.COST):
    """Daily KRW returns when the rule for each regime is held; 2*cost per switch.
    labels[t] is known at close t-1 (it uses closes up to t-1), so day t's
    return belongs to the rule chosen for labels[t-1]."""
    prev = np.concatenate([[labels[0]], labels[:-1]])
    out = np.zeros(len(labels))
    for t in range(len(labels)):
        rule = mapping[prev[t]]
        out[t] = 0.0 if rule == "CASH" else series[rule][t]
        if t > 0 and mapping[prev[t]] != mapping[prev[t - 1]]:
            out[t] -= 2 * cost
    return out


def fit_mapping(labels, series, is_end, candidates):
    """For each regime pick the rule with the best in-sample mean daily log return."""
    prev = np.concatenate([[labels[0]], labels[:-1]])
    mapping = {}
    for g in (BULL, BOX, BEAR):
        days = [t for t in range(1, is_end + 1) if prev[t] == g]
        if not days:
            mapping[g] = candidates[0]
            continue
        def score(rule):
            return 0.0 if rule == "CASH" else np.log1p(series[rule][days]).mean()
        mapping[g] = max(candidates, key=score)
    return mapping


def evaluate(m, labels, series, ctx, candidates):
    mapping = fit_mapping(labels, series, ctx.is_end, candidates)
    r = switch_returns(labels, mapping, series)
    lr = np.log1p(r)
    return mapping, {"total": np.expm1(lr.sum()), "is": np.expm1(lr[:ctx.is_end + 1].sum()),
                     "oos": np.expm1(lr[ctx.is_end + 1:].sum()), "mdd": C.stats(r)["mdd"]}


if __name__ == "__main__":
    d = D.load()
    m = E.build_market(d)
    names, stack = E.build_scores(m)
    ctx = G.Context(m)

    def rule(src, score, N=5, reb=0, buf=None):
        return [names.index((src, "all", score)), N, 1, N if buf is None else buf, reb, 0,
                E.SOURCES.index(src)]

    rules = {"REPL_T": rule("T", "W", N=500, buf=0), "H1b_T_W5": rule("T", "W"),
             "H6_T_MOM120": rule("T", "MOM120"), "H6c_T_MOM120_N10M": rule("T", "MOM120", N=10, reb=3),
             "H7_I_RADJ120": rule("I", "RADJ120"), "H9b_I_PULL": rule("I", "PULL"),
             "H9c_I_PULL_W": rule("I", "PULL", reb=1), "H10_K_REV5": rule("K", "REV5")}
    series = {k: C.simulate(m, stack, p)[0] for k, p in rules.items()}
    cands = list(rules) + ["CASH"]
    print("static rules: " + ", ".join(
        f"{k} {np.expm1(np.log1p(v).sum()) * 100:+.0f}% (IS {np.expm1(np.log1p(v)[:ctx.is_end + 1].sum()) * 100:+.0f}%"
        f" / OOS {np.expm1(np.log1p(v)[ctx.is_end + 1:].sum()) * 100:+.0f}%)" for k, v in series.items()))
    classifiers = {"QQQ trend": classify_qqq, "SMH trend": classify_smh, "breadth": classify_breadth}
    for name, fn in classifiers.items():
        lab = fn(m)
        cnt = np.bincount(lab, minlength=3)
        mapping, res = evaluate(m, lab, series, ctx, cands)
        flips = int((np.diff(lab) != 0).sum())
        print(f"\n[{name}] days bull/box/bear {cnt.tolist()}, label changes {flips}")
        print(f"  IS-fitted map: bull->{mapping[BULL]}, box->{mapping[BOX]}, bear->{mapping[BEAR]}")
        print(f"  total {res['total'] * 100:+.0f}%  IS {res['is'] * 100:+.0f}%  OOS {res['oos'] * 100:+.0f}%  MDD {res['mdd'] * 100:.0f}%")
        for fixed in ({BULL: "H6_T_MOM120", BOX: "H9c_I_PULL_W", BEAR: "CASH"},
                      {BULL: "H6_T_MOM120", BOX: "H9c_I_PULL_W", BEAR: "H10_K_REV5"},
                      {BULL: "H6_T_MOM120", BOX: "H6_T_MOM120", BEAR: "CASH"}):
            r = switch_returns(lab, fixed, series)
            lr = np.log1p(r)
            print(f"  fixed {fixed[BULL]}/{fixed[BOX]}/{fixed[BEAR]}: total {np.expm1(lr.sum()) * 100:+.0f}%"
                  f"  IS {np.expm1(lr[:ctx.is_end + 1].sum()) * 100:+.0f}%  OOS {np.expm1(lr[ctx.is_end + 1:].sum()) * 100:+.0f}%"
                  f"  MDD {C.stats(r)['mdd'] * 100:.0f}%")
