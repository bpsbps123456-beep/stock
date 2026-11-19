"""Point-in-time season following: hold the species a detector names, decided from closes through j-1.

Pre-declared grid (138 candidates, fixed before scoring):
  M1 species momentum: hold the species with the best trailing L-session return.
     7 species sets x L in {3, 5, 10, 20, 40} x {daily, weekly decision} = 70
  M2 momentum-IC dial: MOM while the matured trailing IC of 120-session momentum is >= 0, else ALT.
     horizon h in {5, 10, 20} x window K in {5, 10, 20} x ALT in {CASH, HALF, REV, NONSEMI} = 36
  M3 learned seasons: state = (SOXX >= SMA n) x (IC_10,20 >= 0); in each state hold the species with the
     best (decayed) mean daily return over earlier sessions of that state, MOM until 10 are seen.
     n in {20, 50} x half-life in {all, 40} x species set {S2, S6} = 8
  M5 two dials: SOXX gate off -> DOWN species; gate on -> MOM if IC >= 0 else REV.
     gate in {SOXX10r1, SOXX20r1, SOXX20r3} x h in {5, 10} x K in {10, 20} x DOWN in {CASH, HALF} = 24
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from species import Lab, HERE, CHECK  # first: puts the shared harness on sys.path
from season import momentum_ic, known_ic, species_momentum, UNLEV
import common  # noqa: E402
import study  # noqa: E402

OUT = HERE / "out"
CACHE = HERE / "cache"
SETS = {
    "S1": ["MOM", "CASH"],
    "S2": ["MOM", "REV", "CASH"],
    "S3": ["MOM", "REV", "PULL", "NONSEMI", "CASH"],
    "S4": ["MOM", "SOXX", "QQQ", "CASH"],
    "S5": ["MOM", "REV", "SOXL", "SOXS", "CASH"],
    "S6": UNLEV,
    "S7": UNLEV + ["SOXL", "SOXS", "PSQ"],
}
EXTRA = {"hell1": ("2026-06-23", "2026-07-29"), "rebound": ("2026-07-30", "2026-09-22"),
         "hell2": ("2026-10-13", "2026-11-17"), "reversal_season": ("2026-06-26", "2026-10-05")}


def weekly_mask(days):
    from desk import strategies as S
    return np.array([j == 0 or S.iso_week(d) != S.iso_week(days[j - 1]) for j, d in enumerate(days)])


def hold_weekly(choice, mask):
    out = choice.copy()
    for j in range(1, len(out)):
        if not mask[j]:
            out[j] = out[j - 1]
    return out


def m3_choice(usd, names, state, half_life, min_seen=10):
    T = usd.shape[0]
    logret = np.log(usd / np.vstack([np.ones(usd.shape[1]), usd[:-1]]))
    choice = np.zeros(T, dtype=int)
    mom = names.index("MOM")
    for t in range(T):
        past = np.flatnonzero(state[:t] == state[t]) if state[t] >= 0 else np.array([], int)
        if len(past) < min_seen:
            choice[t] = mom
            continue
        w = np.ones(len(past)) if half_life is None else 0.5 ** ((t - 1 - past) / half_life)
        choice[t] = int(np.argmax(w @ logret[past] / w.sum()))
    return choice


def build(lab):
    days, ctx = lab.ctx.days, lab.ctx
    T = len(days)
    wk = weekly_mask(days)
    ic_raw = {h: momentum_ic(lab, h) for h in (5, 10, 20)}
    soxx = ctx.book.cls["SOXX"]
    k = np.arange(ctx.start, ctx.start + T) - 1
    trend = {n: (soxx.to_numpy() >= soxx.rolling(n).mean().to_numpy())[k] for n in (20, 50)}
    specs = {}
    for sname, names in SETS.items():
        usd = np.column_stack([lab.ledgers[n].usd.to_numpy() for n in names])
        for L in (3, 5, 10, 20, 40):
            daily = species_momentum(usd, L)
            for freq, ch in (("d", daily), ("w", hold_weekly(daily, wk))):
                specs[f"M1_{sname}_L{L}_{freq}"] = dict(family="M1", names=names, choice=ch, set=sname, L=L, freq=freq)
    for h in (5, 10, 20):
        for K in (5, 10, 20):
            ic = known_ic(ic_raw[h], h, K)
            works = ~(ic < 0)  # NaN (not yet matured) counts as momentum working
            for alt in ("CASH", "HALF", "REV", "NONSEMI"):
                names = ["MOM", alt]
                specs[f"M2_h{h}_K{K}_{alt}"] = dict(family="M2", names=names, choice=np.where(works, 0, 1),
                                                    h=h, K=K, alt=alt)
    ic1020 = known_ic(ic_raw[10], 10, 20)
    for n in (20, 50):
        state = np.where(np.isnan(ic1020), -1, 2 * trend[n] + (ic1020 >= 0)).astype(int)
        for hl in (None, 40):
            for sname in ("S2", "S6"):
                names = SETS[sname]
                usd = np.column_stack([lab.ledgers[x].usd.to_numpy() for x in names])
                specs[f"M3_n{n}_hl{hl or 'all'}_{sname}"] = dict(family="M3", names=names,
                                                                  choice=m3_choice(usd, names, state, hl),
                                                                  n=n, half_life=hl or 0, set=sname)
    for g in ("SOXX10r1", "SOXX20r1", "SOXX20r3"):
        up = ctx.gates[g]
        for h in (5, 10):
            for K in (10, 20):
                works = ~(known_ic(ic_raw[h], h, K) < 0)
                for down in ("CASH", "HALF"):
                    names = ["MOM", "REV", down]
                    ch = np.where(~up, 2, np.where(works, 0, 1))
                    specs[f"M5_{g}_h{h}_K{K}_{down}"] = dict(family="M5", names=names, choice=ch,
                                                            gate=g, h=h, K=K, down=down)
    return specs, ic_raw


def references(lab):
    ctx = lab.ctx
    refs = {"base": lab.ledgers["MOM"], "half": lab.ledgers["HALF"]}
    for g in ("SOXX10r1", "SOXX20r3", "SOXX100r1"):
        refs[f"gate_{g}"] = study.replay(ctx.days, lab.opn, lab.close, ctx.fx, lab.plans["MOM"], ctx.gates[g])
    crowd = pd.read_parquet(CHECK / "cache" / "crowd_daily.parquet")
    for key in ("C4r_SOXX10_cash", "C4o_SOXX10_cash"):
        part = crowd[crowd.strategy == key].set_index("date")[common.LEDGER].astype(float)
        refs[f"bb1_{key[:3]}_SOXX10"] = part
    for n in ("REV", "PULL", "SOXX", "CASH"):
        refs[f"species_{n}"] = lab.ledgers[n]
    return refs


def windows(curve):
    out = {}
    for w, (a, b) in {**common.WINDOWS, **EXTRA}.items():
        s = common.window_stats(curve, a, b)
        out[f"{w}_usd"], out[f"{w}_usd_mdd"], out[f"{w}_krw"] = s["usd"], s["usd_mdd"], s["krw"]
        out[f"{w}_krw_mdd"], out[f"{w}_expo"] = s["krw_mdd"], s["expo"]
    out["turn"] = float(curve.turn.sum())
    usd10 = curve.usd * curve.krw10 / curve.krw
    out["full_usd10"] = float(usd10.iloc[-1] - 1)
    return out


def switches(choice):
    return int((np.diff(choice) != 0).sum())


def placebo(lab, spec, draws=None):
    """Circular shifts of the choice series keep run lengths and the species mix, not the timing."""
    ch = spec["choice"]
    T = len(ch)
    shifts = range(10, T - 9) if draws is None else draws
    vals = []
    for s in shifts:
        led = lab.follow(np.roll(ch, s), spec["names"])
        st = windows(led)
        vals.append((st["full_usd"], st["h2_usd"], st["sep_nov_usd"]))
    return np.array(vals)


def main():
    lab = Lab()
    specs, ic_raw = build(lab)
    curves = {key: lab.follow(sp["choice"], sp["names"]) for key, sp in specs.items()}
    refs = references(lab)
    CACHE.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    every = {**{f"ref:{k}": v for k, v in refs.items()}, **curves}
    pd.concat(every, names=["strategy", "date"]).reset_index().to_parquet(CACHE / "meta_daily.parquet", index=False)
    rows = {}
    for key, curve in every.items():
        meta = specs.get(key, {})
        rows[key] = {"family": meta.get("family", "ref"),
                     **{k: v for k, v in meta.items() if k not in ("names", "choice", "family")},
                     "species": "/".join(meta.get("names", [])),
                     "switches": switches(meta["choice"]) if meta else np.nan, **windows(curve)}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "strategy"

    # Forward selection inside the 138 (base included as a candidate, as in Codex's rule).
    st = common.stats_table({**curves, "base": refs["base"]})
    picks = []
    for train, test in (("h1", "h2"), ("jan_aug", "sep_nov")):
        for currency in ("krw", "usd"):
            for rule, key in common.select(st, train, currency).items():
                r = st.loc[key]
                ctrl = common.matched_control(refs["base"], r[f"{test}_expo"], test)
                picks.append({"train": train, "test": test, "select_in": currency, "rule": rule, "strategy": key,
                              "train_usd": r[f"{train}_usd"], "train_usd_mdd": r[f"{train}_usd_mdd"],
                              "test_usd": r[f"{test}_usd"], "test_usd_mdd": r[f"{test}_usd_mdd"],
                              "test_expo": r[f"{test}_expo"], "matched_usd": ctrl[0], "matched_usd_mdd": ctrl[1],
                              "base_test_usd": st.loc["base", f"{test}_usd"]})
    picks = pd.DataFrame(picks)

    # Robustness for every selected rule and the hindsight leaders: one-session delay and placebo.
    focus = list(dict.fromkeys([*picks.strategy[picks.strategy != "base"],
                                *table.loc[list(curves)].full_usd.nlargest(3).index,
                                *table.loc[list(curves)].h2_usd.nlargest(3).index]))
    robust = []
    for key in focus:
        sp = specs[key]
        late = lab.follow(np.r_[sp["choice"][0], sp["choice"][:-1]], sp["names"])
        wl = windows(late)
        pl = placebo(lab, sp)
        me = table.loc[key]
        robust.append({"strategy": key, "full_usd": me.full_usd, "h2_usd": me.h2_usd, "sep_nov_usd": me.sep_nov_usd,
                       "full_usd_mdd": me.full_usd_mdd, "delay1_full_usd": wl["full_usd"], "delay1_h2_usd": wl["h2_usd"],
                       "delay1_sep_nov_usd": wl["sep_nov_usd"], "delay1_full_usd_mdd": wl["full_usd_mdd"],
                       "placebo_n": len(pl), "pct_full": float((pl[:, 0] < me.full_usd).mean()),
                       "pct_h2": float((pl[:, 1] < me.h2_usd).mean()),
                       "pct_sep_nov": float((pl[:, 2] < me.sep_nov_usd).mean()),
                       "placebo_full_median": float(np.median(pl[:, 0])), "switches": me.switches})
    robust = pd.DataFrame(robust)

    table.to_csv(OUT / "meta_candidates.csv")
    picks.to_csv(OUT / "meta_selection.csv", index=False)
    robust.to_csv(OUT / "meta_robustness.csv", index=False)
    choices = pd.DataFrame({k: [specs[k]["names"][c] for c in specs[k]["choice"]] for k in focus},
                           index=lab.ctx.days)
    choices.to_csv(OUT / "meta_focus_choices.csv", index_label="session")

    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 200,
                  "display.float_format", lambda v: f"{v:+.3f}")
    cols = ["full_usd", "full_usd_mdd", "h1_usd", "hell1_usd", "rebound_usd", "hell2_usd", "h2_usd",
            "sep_nov_usd", "reversal_season_usd", "h2_expo", "turn", "full_usd10", "switches"]
    print("== references\n", table[table.family == "ref"][cols].to_string())
    fam = table[table.family != "ref"]
    print("\n== family medians\n", fam.groupby("family")[cols[:-1]].median().to_string())
    b = table.loc["ref:base"]
    beat = fam[(fam.full_usd > b.full_usd) & (fam.full_usd_mdd > b.full_usd_mdd)]
    beat2 = fam[(fam.hell1_usd > b.hell1_usd) & (fam.hell2_usd > b.hell2_usd) & (fam.h1_usd >= .9 * b.h1_usd)]
    print(f"\nbeat base on full-year return and MDD: {len(beat)}/{len(fam)}; "
          f"beat base in both hells while keeping 90% of H1: {len(beat2)}/{len(fam)}")
    print("\n== top 12 by full-year USD\n", fam.sort_values("full_usd", ascending=False)[cols].head(12).to_string())
    print("\n== forward selection\n", picks.to_string(index=False))
    print("\n== robustness (delay 1 session, placebo = all circular shifts of the choice series)\n",
          robust.to_string(index=False))
    meta = {"candidates": len(specs), "by_family": fam.family.value_counts().to_dict(),
            "beat_full_ret_mdd": len(beat), "beat_both_hells_keep90_h1": len(beat2)}
    (OUT / "meta_summary.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
