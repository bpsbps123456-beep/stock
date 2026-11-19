"""Attack the season-dial results: delay, exposure-matched control, joint placebo, concentration, look-ahead.

Families checked in full (no hand-picking): M2 with ALT=REV (9), M5 (24), M1 on set S2 (10).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from species import Lab, HERE, build_plans, extended_bars
from meta import build, windows
from season import momentum_ic, known_ic
import common  # noqa: E402

OUT = HERE / "out"
FAMILIES = {
    "M2_REV": lambda k, s: s["family"] == "M2" and s["alt"] == "REV",
    "M5": lambda k, s: s["family"] == "M5",
    "M1_S2": lambda k, s: s["family"] == "M1" and s["set"] == "S2",
}


def delayed(choice, n):
    return np.r_[np.repeat(choice[0], n), choice[:-n]] if n else choice


def main():
    lab = Lab()
    specs, ic_raw = build(lab)
    base = lab.ledgers["MOM"]
    T = len(lab.ctx.days)
    shifts = list(range(10, T - 9))
    rows, joint = [], {}
    for fam, keep in FAMILIES.items():
        keys = [k for k, s in specs.items() if keep(k, s)]
        joint[fam] = {"keys": keys, "actual": [], "shifted": np.zeros((len(shifts), len(keys), 3))}
        for i, key in enumerate(keys):
            sp = specs[key]
            me = windows(lab.follow(sp["choice"], sp["names"]))
            d1 = windows(lab.follow(delayed(sp["choice"], 1), sp["names"]))
            d2 = windows(lab.follow(delayed(sp["choice"], 2), sp["names"]))
            ctrl = {w: common.matched_control(base, me[f"{w}_expo"], w) for w in ("full", "h2", "sep_nov")}
            for s_i, s in enumerate(shifts):
                st = windows(lab.follow(np.roll(sp["choice"], s), sp["names"]))
                joint[fam]["shifted"][s_i, i] = (st["full_usd"], st["h2_usd"], st["full_usd_mdd"])
            pl = joint[fam]["shifted"][:, i]
            joint[fam]["actual"].append((me["full_usd"], me["h2_usd"], me["full_usd_mdd"]))
            rows.append({"family": fam, "strategy": key,
                         "full_usd": me["full_usd"], "full_usd_mdd": me["full_usd_mdd"], "h1_usd": me["h1_usd"],
                         "h2_usd": me["h2_usd"], "sep_nov_usd": me["sep_nov_usd"], "hell1_usd": me["hell1_usd"],
                         "rebound_usd": me["rebound_usd"], "hell2_usd": me["hell2_usd"],
                         "full_expo": me["full_expo"], "h2_expo": me["h2_expo"], "full_usd10": me["full_usd10"],
                         "ctrl_full_usd": ctrl["full"][0], "ctrl_h2_usd": ctrl["h2"][0], "ctrl_sep_nov_usd": ctrl["sep_nov"][0],
                         "d1_full_usd": d1["full_usd"], "d1_full_usd_mdd": d1["full_usd_mdd"], "d1_h2_usd": d1["h2_usd"],
                         "d2_full_usd": d2["full_usd"], "d2_h2_usd": d2["h2_usd"],
                         "pct_full": float((pl[:, 0] < me["full_usd"]).mean()), "pct_h2": float((pl[:, 1] < me["h2_usd"]).mean()),
                         "pct_mdd": float((pl[:, 2] < me["full_usd_mdd"]).mean())})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "robust_candidates.csv", index=False)

    fw = []
    for fam, j in joint.items():
        act = np.array(j["actual"])
        sh = j["shifted"]
        for m, name in ((0, "full_usd"), (1, "h2_usd"), (2, "full_usd_mdd")):
            best, med = act[:, m].max(), np.median(act[:, m])
            fw.append({"family": fam, "n": len(j["keys"]), "metric": name,
                       "actual_best": best, "pct_best_vs_shifted_best": float((sh[:, :, m].max(axis=1) < best).mean()),
                       "actual_median": med, "pct_median_vs_shifted_median": float((np.median(sh[:, :, m], axis=1) < med).mean())})
    fw = pd.DataFrame(fw)
    fw.to_csv(OUT / "robust_familywise.csv", index=False)

    # REV species: which names made its H2, and how concentrated the gain was.
    c = lab.ctx
    rev = lab.plans["REV"]
    held = []
    days = c.days
    keys = list(c.col)
    for j, w in sorted(rev.items()):
        for i in np.flatnonzero(w[:len(keys)] > 0):
            held.append({"rebalance": days[j], "ticker": c.tickers[i], "weight": float(w[i])})
    held = pd.DataFrame(held)
    held.to_csv(OUT / "rev_holdings.csv", index=False)
    contrib = []
    dates = sorted(rev)
    for n, j in enumerate(dates):
        nxt = dates[n + 1] if n + 1 < len(dates) else T - 1
        w = rev[j][:len(keys)]
        o0, o1 = lab.opn[j, :len(keys)], (lab.opn[nxt, :len(keys)] if nxt > j else lab.close[T - 1, :len(keys)])
        for i in np.flatnonzero(w > 0):
            contrib.append({"rebalance": days[j], "ticker": c.tickers[i], "weight": float(w[i]),
                            "ret": float(o1[i] / o0[i] - 1), "contrib": float(w[i] * (o1[i] / o0[i] - 1))})
    contrib = pd.DataFrame(contrib)
    contrib.to_csv(OUT / "rev_contrib.csv", index=False)

    # Look-ahead self-test: scramble every price after session T0, decisions up to T0 must not move.
    T0 = 150
    book = c.book
    k0 = c.start + T0
    rng = np.random.default_rng(7)
    saved = {n: getattr(book, n).copy() for n in ("adj", "cls", "opn")}
    for n in ("adj", "cls", "opn"):
        frame = getattr(book, n)
        frame.iloc[k0:] = frame.iloc[k0:].to_numpy() * rng.uniform(.5, 1.5, size=frame.iloc[k0:].shape)
    ic_b = {h: momentum_ic(lab, h) for h in (5, 10, 20)}
    opn_b, _, etf_b = extended_bars(c)
    plans_b = build_plans(c, opn_b.shape[1], etf_b)
    for n, frame in saved.items():
        setattr(book, n, frame)
    known_same = all(np.allclose(known_ic(ic_raw[h], h, K)[:T0 + 1], known_ic(ic_b[h], h, K)[:T0 + 1], equal_nan=True)
                     for h in (5, 10, 20) for K in (5, 10, 20))
    rev_same = all(np.array_equal(lab.plans["REV"][j], plans_b["REV"][j]) for j in lab.plans["REV"] if j <= T0)
    selftest = {"T0": days[T0], "known_ic_identical_through_T0": bool(known_same), "rev_targets_identical_through_T0": bool(rev_same)}

    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 200,
                  "display.float_format", lambda v: f"{v:+.3f}")
    show = ["strategy", "full_usd", "full_usd_mdd", "h2_usd", "sep_nov_usd", "hell1_usd", "rebound_usd", "hell2_usd",
            "h2_expo", "ctrl_h2_usd", "ctrl_full_usd", "d1_full_usd", "d1_h2_usd", "d2_full_usd", "pct_full", "pct_h2"]
    for fam in FAMILIES:
        print(f"\n== {fam}\n", table[table.family == fam][show].to_string(index=False))
    print("\n== family-wise joint-shift placebo\n", fw.to_string(index=False))
    h2 = contrib[(contrib.rebalance >= "2026-06-26") & (contrib.rebalance <= "2026-10-05")]
    by = h2.groupby("ticker").contrib.sum().sort_values(ascending=False)
    print("\n== REV reversal-season contribution by ticker (sum of weekly weight x return)\n", by.head(10).to_string(),
          "\n ... bottom\n", by.tail(5).to_string(), f"\n total {by.sum():+.3f}, top 3 share {by.head(3).sum() / by.sum():.2f}")
    print("\nself-test:", selftest)
    (OUT / "robust_selftest.json").write_text(json.dumps(selftest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
