"""Family "reentry": leave the market on a fixed exit, buy back on one of the desk's bottom signals.

Run from the repo root:  py data/derived/analysis/desk_defense_check_20261119/ext_reentry.py

Engine: Codex's weekly 7-stock plan (ctx.weekly) replayed with common.replay_frac. Only the exposure
fraction frac[j] (0, 1/3, 2/3, 1) changes; no leverage, no instrument outside the book.

Timing: every signal array S[j] below holds information through the close of session j-1 (BB-1: the
calendar row of session j, which the registered definition builds from the previous close and the SEIBRO
table published 09:00 KST before that session). A decision in S[j] is filled at the open of session j.

Exits (fixed, not optimised)
  E1..E4  SOXX close < SMA10/20/50/100 (ctx.gates SOXX{n}r1; recomputed here and asserted equal)
  E5      SOXX close <= 90% of its 60-session high close
  E6      base shadow NAV (USD close) <= 88% of its peak; the peak restarts at the re-buy close
          (stop_research branch G "trailing -12%"); "gate recovered" = within 12% of the all-time peak
  E1..E5 re-arm only after the exit condition has switched off once (a cross-down event, as in branch G).

Re-entries (each crossed with E1..E6)
  R0  gate's own recovery
  R1  fixed wait N sessions                      N in {3, 5, 10, 20}
  R2  SOXX >= (1+x) * lowest close since exit    x in {5, 7, 10}%
  R3  R2 and the low is at least h sessions old  x in {5, 7, 10}% x h in {5, 10}
  R4  BB-1 on (pure)   /  R5 = BB-1 on or gate recovered (the "orgate" mode of R4)
  R6  bottom detector state "confirmed" (desk.bottom.frame: in_dd and up >= 7%)
  R7  ladder: 1/3 back at each of SOXX -15/-20/-25% from the 60-session high (latched), rest on gate recovery
  R8  AI breadth below T during the spell and then up p points from its spell low   T in {10,15,19} x p in {3,5}
  R9  SOXX RSI14 below L during the spell and then above L                          L in {30, 35, 40}
  mode "pure": only the signal re-buys.  mode "orgate": the signal or the gate's recovery, whichever first.
  Grid: 6 exits x (R0 1 + R7 1 + 24 signal rules x 2 modes) = 6 x 50 = 300 candidates + base = 301 ledgers.

Extra outputs (cache/): reentry_detail.csv (one row per candidate: windows, round trips, episode factors,
random re-buy percentiles), reentry_trips.csv (one row per cash spell), reentry_summary.json.
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import study  # noqa: E402
from desk import bottom  # noqa: E402  pure function frame() only

FAMILY = "reentry"
BB1 = HERE.parent / "bottom_zone_20261111" / "belt" / "now" / "bb1_calendar.json"
BREADTH = common.CODEX / "breadth.csv"
SMA = {"E1": 10, "E2": 20, "E3": 50, "E4": 100}
EXITS = {"E1": "SOXX10r1", "E2": "SOXX20r1", "E3": "SOXX50r1", "E4": "SOXX100r1",
         "E5": "SOXXdd60_10", "E6": "shadow_trail12"}
LADDER = (-.15, -.20, -.25)
EPISODES = (("pre", "2026-01-01", "2026-06-21"), ("jul", "2026-06-22", "2026-08-31"),
            ("sep", "2026-09-01", "2026-09-30"), ("octnov", "2026-10-01", "2026-12-31"))
HELL = {"hell1": ("2026-06-22", "2026-07-29"), "hell2": ("2026-10-13", "2026-11-17")}
DRAWS = 1000


def rules():
    """(rule id, parameter dict, modes)."""
    out = [("R0", {}, ("gate",))]
    out += [("R1", {"wait": n}, ("pure", "orgate")) for n in (3, 5, 10, 20)]
    out += [("R2", {"up": x}, ("pure", "orgate")) for x in (.05, .07, .10)]
    out += [("R3", {"up": x, "hold": h}, ("pure", "orgate")) for x in (.05, .07, .10) for h in (5, 10)]
    out += [("R4", {}, ("pure", "orgate"))]
    out += [("R6", {}, ("pure", "orgate"))]
    out += [("R7", {}, ("orgate",))]
    out += [("R8", {"below": t, "rise": p}, ("pure", "orgate")) for t in (10, 15, 19) for p in (3, 5)]
    out += [("R9", {"rsi": level}, ("pure", "orgate")) for level in (30, 35, 40)]
    return out


def rsi14(close):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return 100 - 100 / (1 + up / down)


def build_signals(soxx, start, n, shadow_close, breadth_close, bb1_on):
    """All arrays have one value per replay session j and use closes through j-1 only.

    soxx: SOXX closes on book.days (position start+j is session j). shadow_close[j]: base USD NAV at the
    close of session j. breadth_close[j+1]: breadth at the close of session j (index 0 = close before
    session 0). bb1_on[j]: BB-1 calendar state of session j (pre-open information).
    """
    prev = slice(start - 1, start - 1 + n)                     # close of session j-1
    sig = {"soxx": soxx.to_numpy()[prev]}
    for name, m in SMA.items():
        above = (soxx >= soxx.rolling(m).mean()).to_numpy()[prev]
        sig[name] = ~above                                      # exit condition on
    sig["dd60"] = (soxx / soxx.rolling(60).max() - 1).to_numpy()[prev]
    sig["E5"] = sig["dd60"] <= -.10
    sig["rsi"] = rsi14(soxx).to_numpy()[prev]
    frame = bottom.frame(pd.Series(soxx.to_numpy(), index=pd.to_datetime(soxx.index)))
    sig["confirmed"] = (frame.in_dd.to_numpy() & (frame.up.to_numpy() >= bottom.CONFIRM))[prev]
    sig["shadow"] = np.r_[1., shadow_close[:-1]]
    sig["runmax"] = np.maximum.accumulate(sig["shadow"])
    sig["breadth"] = np.asarray(breadth_close[:n], dtype=float)
    sig["bb1"] = np.asarray(bb1_on, dtype=bool)
    for key, values in sig.items():
        if len(values) != n or (values.dtype != bool and not np.isfinite(values).all()):
            raise ValueError(f"signal {key} incomplete")
    return sig


def simulate(sig, exit_id, rule, par, mode):
    """Exposure fraction per session and the list of cash spells (exit session, full re-buy session or None)."""
    n = len(sig["soxx"])
    soxx, dd60, rsi, breadth, shadow = (sig[k] for k in ("soxx", "dd60", "rsi", "breadth", "shadow"))
    trail = exit_id == "E6"
    cond = (shadow <= .88 * sig["runmax"]) if trail else sig[exit_id]   # gate off (all-time-peak version for E6)
    frac = np.ones(n)
    spells, tranches = [], []
    invested, armed, peak = True, True, 1.
    for j in range(n):
        if invested:
            if trail:
                peak = max(peak, shadow[j])
                hit = shadow[j] <= .88 * peak
            else:
                armed = armed or not cond[j]
                hit = armed and cond[j]
            if hit:
                invested, armed, a = False, False, j
                low, low_j, bmin, was_low, stage = soxx[j], j, breadth[j], rsi[j] < par.get("rsi", 0), 0
                frac[j] = 0.
            continue
        if soxx[j] < low:
            low, low_j = soxx[j], j
        bmin = min(bmin, breadth[j])
        recovered = not cond[j]
        if rule == "R0":
            back = recovered
        elif rule == "R1":
            back = j - a >= par["wait"]
        elif rule == "R2":
            back = soxx[j] >= low * (1 + par["up"])
        elif rule == "R3":
            back = soxx[j] >= low * (1 + par["up"]) and j - low_j >= par["hold"]
        elif rule == "R4":
            back = bool(sig["bb1"][j])
        elif rule == "R6":
            back = bool(sig["confirmed"][j])
        elif rule == "R7":
            level = sum(dd60[j] <= x for x in LADDER)
            if level > stage:
                tranches += [(a, j)] * (level - stage)
                stage = level
            back = stage >= 3
        elif rule == "R8":
            back = bmin < par["below"] and breadth[j] >= bmin + par["rise"]
        elif rule == "R9":
            back = was_low and rsi[j] > par["rsi"]
            was_low = was_low or rsi[j] < par["rsi"]
        else:
            raise ValueError(rule)
        if mode == "orgate" and recovered:
            back = True
        if back:
            if rule == "R7" and stage < 3:
                tranches += [(a, j)] * (3 - stage)
            invested, peak, armed = True, shadow[j], recovered   # re-armed once the gate has been seen off
            spells.append((a, j))
        else:
            frac[j] = stage / 3
    if not invested:
        spells.append((a, None))
    return frac, spells, tranches


def gate_recovery(sig, exit_id, a):
    """First session after exit a on which the exit condition is off (None: not before the data end)."""
    cond = (sig["shadow"] <= .88 * sig["runmax"]) if exit_id == "E6" else sig[exit_id]
    later = np.flatnonzero(~cond[a + 1:])
    return int(a + 1 + later[0]) if later.size else None


def episode_of(day):
    return next(name for name, lo, hi in EPISODES if lo <= day <= hi)


def trip_rows(key, cfg, sig, days, spells, tranches, curve, base, v_open, rng):
    """One row per cash spell plus the 1,000-draw random re-buy control (0/1 rules only).

    factor = strategy/base ratio at the close of the re-buy session divided by the same ratio at the close
    before the exit: >1 means the round trip beat simply holding the base. For 0/1 rules it equals
    base open NAV at the exit / base open NAV at the re-buy (asserted).
    Control: keep the exit date, re-buy at a uniformly random session in [exit+1, max(re-buy, gate recovery)].
    """
    n = len(days)
    ratio = (curve.usd / base.usd).to_numpy()
    binary = cfg["rule"] != "R7"
    rows, draws = [], np.ones(DRAWS)
    episode_draws = {name: np.ones(DRAWS) for name, _, _ in EPISODES}
    for a, s in spells:
        end = s if s is not None else n - 1
        factor = ratio[end] / (ratio[a - 1] if a else 1.)
        b = gate_recovery(sig, cfg["exit"], a)
        last = n - 1 if (s is None or b is None) else max(s, b)
        window = np.arange(a + 1, last + 1)
        row = dict(strategy=key, exit_day=days[a], rebuy_day=days[s] if s is not None else "",
                   gate_recovery_day=days[b] if b is not None else "", open_at_end=s is None,
                   sessions_out=(s if s is not None else n) - a, episode=episode_of(days[a]), factor=factor,
                   sold=v_open[a], window_sessions=len(window))
        if binary:
            bought = v_open[s] if s is not None else base.usd.iloc[-1]
            np.testing.assert_allclose(factor, v_open[a] / bought, rtol=1e-9)
            row.update(rebought=bought, lower=bool(bought < v_open[a]))
            if len(window) and s is not None:
                # A spell still open at the data end has no re-buy to rank: it stays out of the control.
                alt = v_open[window]
                row["pct_cheaper_than_random"] = float(((alt > bought).sum() + .5 * (alt == bought).sum()) / len(alt))
                row["rebuy_vs_spell_low"] = float(bought / min(alt.min(), bought) - 1)
                pick = v_open[a] / alt[rng.integers(0, len(alt), DRAWS)]
            else:
                pick = np.full(DRAWS, factor)
            draws *= pick
            episode_draws[row["episode"]] *= pick
        rows.append(row)
    for a, j in tranches:
        rows.append(dict(strategy=key, exit_day=days[a], rebuy_day=days[j], tranche=True,
                         episode=episode_of(days[a]), sold=v_open[a], rebought=v_open[j],
                         lower=bool(v_open[j] < v_open[a])))
    return rows, (draws, episode_draws) if binary else None


def detail_row(key, cfg, rows, control, curve, base, stats):
    spells = [r for r in rows if not r.get("tranche")]
    legs = [r for r in rows if r.get("tranche")] if cfg["rule"] == "R7" else [r for r in spells if not r["open_at_end"]]
    done = [r for r in spells if not r["open_at_end"]]
    out = dict(strategy=key, **cfg)
    for w in ("full", "h1", "h2", "jul_aug", "sep_nov", "oct_nov"):
        for m in ("usd", "usd_mdd", "krw", "krw_mdd", "expo", "turn"):
            out[f"{w}_{m}"] = stats[f"{w}_{m}"]
    for name, (lo, hi) in HELL.items():
        w = common.window_stats(curve, lo, hi)
        out[f"{name}_usd"], out[f"{name}_usd_mdd"] = w["usd"], w["usd_mdd"]
    cost = curve.krw10 / curve.krw                                  # cumulative 10bp one-way drag
    out["full_usd_10bp"] = (curve.usd * cost).iloc[-1] - 1
    h2 = curve.index >= "2026-07-01"
    before = (curve.usd * cost)[~h2].iloc[-1]
    out["h2_usd_10bp"] = (curve.usd * cost).iloc[-1] / before - 1
    out["round_trips"] = len(done)
    out["open_at_end"] = int(any(r["open_at_end"] for r in spells))
    out["avg_sessions_out"] = float(np.mean([r["sessions_out"] for r in spells])) if spells else np.nan
    out["sessions_out_total"] = int(sum(r["sessions_out"] for r in spells))
    out["share_rebought_lower"] = float(np.mean([r["lower"] for r in legs])) if legs else np.nan
    out["vs_base_final"] = curve.usd.iloc[-1] / base.usd.iloc[-1]
    for name, _, _ in EPISODES:
        part = [r for r in spells if r["episode"] == name]
        out[f"trips_{name}"] = len(part)
        out[f"open_{name}"] = int(any(r["open_at_end"] for r in part))
        out[f"factor_{name}"] = float(np.prod([r["factor"] for r in part])) if part else 1.
        lower = [r["lower"] for r in (legs if cfg["rule"] == "R7" else done) if r["episode"] == name]
        out[f"lower_{name}"] = float(np.mean(lower)) if lower else np.nan
    if control is not None and spells:
        draws, by_episode = control
        np.testing.assert_allclose(out["vs_base_final"], np.prod([r["factor"] for r in spells]), rtol=1e-9)
        out["random_pct_all"] = float((draws < out["vs_base_final"]).mean() + .5 * (draws == out["vs_base_final"]).mean())
        out["random_median_all"] = float(np.median(draws))
        if not done:
            out["random_pct_all"] = np.nan
        for name in ("jul", "octnov"):
            d, f = by_episode[name], out[f"factor_{name}"]
            closed = any(r["episode"] == name for r in done)
            out[f"random_pct_{name}"] = float((d < f).mean() + .5 * (d == f).mean()) if closed else np.nan
        pct = [r["pct_cheaper_than_random"] for r in spells if "pct_cheaper_than_random" in r]
        out["mean_trip_pct_cheaper"] = float(np.mean(pct)) if pct else np.nan
    return out


def frac_table(sig):
    table = {}
    for exit_id in EXITS:
        for rule, par, modes in rules():
            for mode in modes:
                label = "".join(f"_{k}{round(v * 100) if isinstance(v, float) else v}" for k, v in par.items())
                key = f"{exit_id}_{rule}{label}_{mode}"
                frac, spells, tranches = simulate(sig, exit_id, rule, par, mode)
                cfg = dict(subfamily=f"{rule}_{mode}", exit=exit_id, exit_name=EXITS[exit_id], rule=rule, mode=mode,
                           wait=par.get("wait", ""), up=par.get("up", ""), hold=par.get("hold", ""),
                           below=par.get("below", ""), rise=par.get("rise", ""), rsi=par.get("rsi", ""))
                table[key] = (frac, spells, tranches, cfg)
    return table


def inputs(ctx):
    n = len(ctx.days)
    soxx = ctx.book.cls["SOXX"]
    if soxx.isna().any() or list(soxx.index[ctx.start:ctx.start + n]) != list(ctx.days):
        raise ValueError("SOXX history does not line up with the replay sessions")
    breadth = pd.read_csv(BREADTH, index_col=0).breadth
    closes = [ctx.book.days[ctx.start - 1]] + list(ctx.days)
    breadth = breadth.reindex(closes)
    calendar = {r["day"]: r for r in json.loads(BB1.read_text(encoding="utf-8"))}
    bb1 = np.array([calendar[d]["on"] is True for d in ctx.days])
    unknown = [d for d in ctx.days if calendar[d]["on"] is None]
    dd = (soxx / soxx.rolling(60).max() - 1).shift()
    err = max(abs(dd[d] - calendar[d]["dd60"]) for d in ctx.days if calendar[d]["dd60"] is not None)
    assert err < 1e-9, f"BB-1 calendar drawdown differs from the book SOXX series: {err}"
    return soxx, breadth.to_numpy(), bb1, unknown


def lookahead_test(ctx, soxx, breadth, bb1, reference, t=150):
    """Multiply every price-like input after session t by 1.5 (BB-1: flip later rows) and compare decisions <= t."""
    n = len(ctx.days)
    opn, close = ctx.opn.copy(), ctx.close.copy()
    opn[t + 1:] *= 1.5
    close[t:] *= 1.5                                             # close of session t is unknown at its open
    shadow = study.replay(ctx.days, opn, close, ctx.fx, ctx.weekly, ctx.gates["none"]).usd.to_numpy()
    soxx2 = soxx.copy()
    soxx2.iloc[ctx.start + t:] *= 1.5
    breadth2 = breadth.copy()
    breadth2[t + 1:] *= 1.5
    bb2 = bb1.copy()
    bb2[t + 1:] = ~bb2[t + 1:]
    sig = build_signals(soxx2, ctx.start, n, shadow, breadth2, bb2)
    changed_later = 0
    for key, (frac, _, _, _) in frac_table(sig).items():
        ref = reference[key][0]
        if not np.array_equal(frac[:t + 1], ref[:t + 1]):
            raise AssertionError(f"look-ahead: {key} changed a decision at or before session {t}")
        changed_later += not np.array_equal(frac[t + 1:], ref[t + 1:])
    return {"T": t, "day_T": ctx.days[t], "candidates_checked": len(reference), "decisions_le_T_identical": True,
            "candidates_whose_later_decisions_changed": int(changed_later)}


def main():
    t0 = time.time()
    ctx = common.context()
    days, n = ctx.days, len(ctx.days)
    args = (days, ctx.opn, ctx.close, ctx.fx)
    base = study.replay(*args, ctx.weekly, ctx.gates["none"])
    np.testing.assert_allclose(base.usd, ctx.baseline.close, rtol=1e-10)
    v_open = np.asarray(ctx.baseline.value, dtype=float)       # base USD NAV at each session's open
    soxx, breadth, bb1, bb1_unknown = inputs(ctx)
    sig = build_signals(soxx, ctx.start, n, base.usd.to_numpy(), breadth, bb1)
    for exit_id, m in SMA.items():                              # own gates == Codex gates
        assert np.array_equal(~sig[exit_id], ctx.gates[f"SOXX{m}r1"]), exit_id
    table = frac_table(sig)
    assert len(table) == 300, len(table)

    curves, configs = {"base": base}, {"base": dict(subfamily="base", exit="E0", exit_name="none", rule="none",
                                                    mode="", wait="", up="", hold="", below="", rise="", rsi="")}
    base_frac = common.replay_frac(*args, ctx.weekly, np.ones(n))
    np.testing.assert_allclose(base_frac.to_numpy(), base.to_numpy(), rtol=1e-10, atol=1e-12)
    for key, (frac, _, _, cfg) in table.items():
        curves[key] = common.replay_frac(*args, ctx.weekly, frac)
        configs[key] = cfg
    for exit_id, m in SMA.items():                              # R0 state machine == Codex gate ledger
        ref = study.replay(*args, ctx.weekly, ctx.gates[f"SOXX{m}r1"])
        np.testing.assert_allclose(curves[f"{exit_id}_R0_gate"].to_numpy(), ref.to_numpy(), rtol=1e-10, atol=1e-12)
    common.save(FAMILY, curves, configs)
    selftest = lookahead_test(ctx, soxx, breadth, bb1, table)

    st = common.stats_table(curves)
    details, trips = [], []
    for key, (frac, spells, tranches, cfg) in table.items():
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        rows, control = trip_rows(key, cfg, sig, days, spells, tranches, curves[key], base, v_open, rng)
        trips += rows
        details.append(detail_row(key, cfg, rows, control, curves[key], base, st.loc[key]))
    detail = pd.DataFrame(details).set_index("strategy")
    pd.DataFrame(trips).to_csv(common.CACHE / "reentry_trips.csv", index=False)
    # Fragility diagnostic (not a candidate): every decision filled one session later.
    late = {key: common.replay_frac(*args, ctx.weekly, np.r_[frac[0], frac[:-1]]) for key, (frac, _, _, _) in table.items()}
    late_stats = common.stats_table(late)
    for w in ("full", "h2", "sep_nov"):
        detail[f"{w}_usd_delay1"] = late_stats[f"{w}_usd"]
        detail[f"{w}_usd_mdd_delay1"] = late_stats[f"{w}_usd_mdd"]
    distinct = len({curves[k].usd.round(10).to_numpy().tobytes() for k in table})

    b = st.loc["base"]
    cand = st.drop(index="base")
    out = {"n_ledgers": len(curves), "distinct_candidate_ledgers": distinct, "bb1_unknown_sessions": len(bb1_unknown),
           "lookahead_selftest": selftest,
           "base_hell": {k: round(float(common.window_stats(base, lo, hi)["usd"]), 6) for k, (lo, hi) in HELL.items()},
           "delay1": {"full_usd_above_base": int((detail.full_usd_delay1 > b.full_usd).sum()),
                      "h2_beat_base_return_and_mdd": int(((detail.h2_usd_delay1 > b.h2_usd)
                                                          & (detail.h2_usd_mdd_delay1 > b.h2_usd_mdd)).sum()),
                      "sep_nov_usd_above_base": int((detail.sep_nov_usd_delay1 > b.sep_nov_usd).sum())}}

    def row(key, train=None, test=None):
        r = st.loc[key]
        d = dict(strategy=key)
        for w in ("full", "h1", "h2", "jan_aug", "jul_aug", "sep_nov", "oct_nov"):
            for m in ("krw", "krw_mdd", "usd", "usd_mdd", "expo", "turn"):
                d[f"{w}_{m}"] = round(float(r[f"{w}_{m}"]), 6)
        c = curves[key]
        cost = c.krw10 / c.krw
        for w, (lo, hi) in (("h2", common.WINDOWS["h2"]), ("sep_nov", common.WINDOWS["sep_nov"])):
            x = common.window_stats(pd.DataFrame({"krw": c.krw10, "usd": c.usd * cost, "exposure": c.exposure,
                                                  "turn": c.turn}), lo, hi)
            d[f"{w}_krw_10bp"], d[f"{w}_usd_10bp"] = round(float(x["krw"]), 6), round(float(x["usd"]), 6)
        if test:
            ctl = common.matched_control(base, r[f"{test}_expo"], test)
            d[f"{test}_control_usd"], d[f"{test}_control_usd_mdd"] = (round(float(v), 6) for v in ctl)
        return d

    out["base"] = row("base")
    out["selections"] = [dict(protocol=f"{name} {train}->{test}", rule=rule, **row(key, train, test))
                         for name, train, test in (("P1", "h1", "h2"), ("P2", "jan_aug", "sep_nov"))
                         for rule, key in common.select(st, train).items()]
    out["hindsight"] = {"best_h2_usd": row(cand.h2_usd.idxmax(), "h1", "h2"),
                        "best_sep_nov_usd": row(cand.sep_nov_usd.idxmax(), "jan_aug", "sep_nov"),
                        "best_full_usd": row(cand.full_usd.idxmax(), "h1", "h2")}
    control = {w: pd.DataFrame([common.matched_control(base, cand.at[k, f"{w}_expo"], w) for k in cand.index],
                               index=cand.index, columns=["usd", "mdd"]) for w in ("h2", "jul_aug", "sep_nov")}
    out["counts"] = {"candidates": len(cand)}
    for w in control:
        beat = (cand[f"{w}_usd"] > b[f"{w}_usd"]) & (cand[f"{w}_usd_mdd"] > b[f"{w}_usd_mdd"])
        excess = cand[f"{w}_usd"] - control[w].usd
        detail[f"{w}_control_usd"], detail[f"{w}_excess_vs_control"] = control[w].usd, excess
        out["counts"][w] = {"beat_base_usd_return_and_mdd": int(beat.sum()),
                            "also_beat_matched_control_return": int((beat & (excess > 0)).sum()),
                            "also_beat_control_return_and_mdd": int((beat & (excess > 0)
                                                                     & (cand[f"{w}_usd_mdd"] > control[w].mdd)).sum())}
    out["counts"]["full_usd_above_base"] = int((cand.full_usd > b.full_usd).sum())
    out["counts"]["h1_usd_above_base"] = int((cand.h1_usd > b.h1_usd).sum())
    detail.to_csv(common.CACHE / "reentry_detail.csv")

    def group(by):
        g = detail.groupby(by)
        t = pd.DataFrame({"n": g.size(), "h2_usd_median": g.h2_usd.median(), "h2_usd_best": g.h2_usd.max(),
                          "jul_aug_usd_median": g.jul_aug_usd.median(), "sep_nov_usd_median": g.sep_nov_usd.median(),
                          "full_usd_median": g.full_usd.median(), "h2_expo_median": g.h2_expo.median(),
                          "h2_turn_median": g.h2_turn.median(),
                          "h2_excess_vs_control_median": g.h2_excess_vs_control.median(),
                          "factor_pre_median": g.factor_pre.median(), "factor_jul_median": g.factor_jul.median(),
                          "factor_octnov_median": g.factor_octnov.median(),
                          "open_at_end_share": g.open_at_end.mean(),
                          "random_pct_all_median": g.random_pct_all.median()})
        return t.round(4)

    out["by_subfamily"] = group("subfamily").reset_index().to_dict("records")
    out["by_exit"] = group("exit").reset_index().to_dict("records")
    # Does waiting still lose? Episode factors (>1: the round trips beat holding the base) by exit date.
    waits = detail[detail.rule != "R0"]
    r0 = detail[detail.rule == "R0"].set_index("exit")
    question = {}
    for name, _, _ in EPISODES:
        f, k = waits[f"factor_{name}"], waits[f"trips_{name}"] > 0
        versus = f[k] > waits.loc[k, "exit"].map(r0[f"factor_{name}"]).to_numpy()
        shut = k & (waits[f"open_{name}"] == 0)
        question[name] = {"cells_with_trips": int(k.sum()), "beat_holding": int((f[k] > 1).sum()),
                          "cells_all_trips_closed": int(shut.sum()), "closed_beat_holding": int((f[shut] > 1).sum()),
                          "closed_median_factor": round(float(f[shut].median()), 4) if shut.any() else None,
                          "beat_gate_recovery_R0": int(versus.sum()), "median_factor": round(float(f[k].median()), 4),
                          "R0_factor_by_exit": {e: round(float(r0.at[e, f"factor_{name}"]), 4) for e in r0.index}}
    question["through_sep_product"] = {
        "beat_holding": int(((waits.factor_pre * waits.factor_jul * waits.factor_sep) > 1).sum()), "cells": len(waits)}
    question["full_year"] = {"beat_holding": int((waits.vs_base_final > 1).sum()), "cells": len(waits),
                             "closed_position_cells_beat_holding": int(((waits.vs_base_final > 1) & (waits.open_at_end == 0)).sum()),
                             "closed_position_cells": int((waits.open_at_end == 0).sum())}
    out["waiting_question"] = question
    binary = detail[detail.rule != "R7"]
    out["random_control"] = {
        "cells": int(binary.random_pct_all.notna().sum()),
        "pct_all_median": round(float(binary.random_pct_all.median()), 4),
        "cells_above_90pct": int((binary.random_pct_all > .9).sum()),
        "cells_below_10pct": int((binary.random_pct_all < .1).sum()),
        "pct_jul_median": round(float(binary.random_pct_jul.median()), 4),
        "pct_octnov_median": round(float(binary.random_pct_octnov.median()), 4),
        "by_rule_median_pct_all": binary.groupby("subfamily").random_pct_all.median().round(4).to_dict(),
        "by_rule_median_pct_jul": binary.groupby("subfamily").random_pct_jul.median().round(4).to_dict(),
        "by_rule_median_pct_octnov": binary.groupby("subfamily").random_pct_octnov.median().round(4).to_dict()}
    out["runtime_seconds"] = round(time.time() - t0, 1)
    (common.CACHE / "reentry_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float),
                                                       encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_ledgers", "distinct_candidate_ledgers", "base_hell", "delay1",
                                          "bb1_unknown_sessions", "lookahead_selftest", "counts",
                                          "waiting_question", "random_control", "runtime_seconds")},
                     indent=1, ensure_ascii=False, default=float))
    for s in out["selections"]:
        print(s["protocol"], s["rule"], s["strategy"])
    print("hindsight", {k: v["strategy"] for k, v in out["hindsight"].items()})


if __name__ == "__main__":
    main()
