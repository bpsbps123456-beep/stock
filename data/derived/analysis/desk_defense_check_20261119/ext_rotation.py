"""Family "rotation": faster or different rankings inside the same PIT TIME universe.

Run from the repo root:  py data/derived/analysis/desk_defense_check_20261119/ext_rotation.py

Universe and timing are the base's: US-listed names of the latest TIME PDF dated before the
session, scores from adjusted closes through the previous session, fills at the session open.
The ranking is re-implemented on book.adj (so the look-ahead test can perturb prices) and the
trivial setting (120 sessions / 7 names / buffer 5 / weekly / TIME weights) is asserted equal
to ctx.weekly and to study.replay's base ledger.

Grid: 1,224 cells including base. An R4 "missing" variant whose targets equal its "drop" twin
is removed together with its 5 gated copies (8 twins -> 48 cells on the frozen inputs, because
no ranked name lacks a trend grade on a weekly date), leaving 1,176 saved ledgers = base + 1,175.
  R1 224 = lookback {5,10,20,40,60,90,120} x N {3,5,7,10} x buffer {5,0} x freq {week,change}
           x weight {time,equal}                                  (contains base)
  R2  40 = score {avgrank, skip5, voladj60, voladj120, accel} x N {3,5,7,10} x freq {week,change}
  R3  48 = rank lookback {20,60,120} x abs lookback {20,60} x reference {zero,SOXX}
           x disposition {cash,rotate} x freq {week,change}
  R4  32 = {trend,giant}-grade-then-momentum x lookback {20,60,120} x missing {drop,neutral}
           x freq {week,change} (24) + trend>=A required x lookback {20,60}
           x missing {drop,pass} x freq {week,change} (8)
  R5 880 = (56 R1 weekly TIME-weight + 40 R2 + 48 R3 + 32 R4 = 176) x gate
           {SOXX10r1, SOXX20r3, SOXX100r1, QQQ10r1, breadth19}
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import study  # noqa: E402
from desk import config  # noqa: E402
from desk import strategies as S  # noqa: E402

FAMILY = "rotation"
LOOKBACKS = (5, 10, 20, 40, 60, 90, 120)
NS = (3, 5, 7, 10)
BUFFERS = (5, 0)
FREQS = ("week", "change")
WEIGHTS = ("time", "equal")
COMPOSITES = ("avgrank", "skip5", "voladj60", "voladj120", "accel")
OUTER = ("SOXX10r1", "SOXX20r3", "SOXX100r1", "QQQ10r1", "breadth19")
NEUTRAL = 2.0          # grade B when a missing grade is treated as neutral
BREADTH_SPLIT = 19.0   # Giant tab threshold (calibrated on 2026-02..09: not a clean H1 input)
T_TEST = 150


# ---------------------------------------------------------------------------
# point-in-time signals (everything a decision at the open of session j may read)
# ---------------------------------------------------------------------------

def build(book, days, start, tickers, col, trend_df, giant_df, breadth):
    adj = book.adj.reindex(columns=tickers).to_numpy(float)
    ks = start + np.arange(len(days))
    assert ks[0] - 1 - max(LOOKBACKS) >= 0

    def ret(prices, lookback, skip=0):
        p1, p0 = prices[ks - 1 - skip], prices[ks - 1 - lookback]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = p1 / p0 - 1
        out[~(np.isfinite(p1) & np.isfinite(p0) & (p0 > 0))] = np.nan
        return out

    score = {f"mom{n}": ret(adj, n) for n in LOOKBACKS}
    score["skip5"] = ret(adj, 120, skip=5)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret = np.diff(np.log(adj), axis=0)          # logret[i] = close i+1 over close i
    vol = np.full(score["mom120"].shape, np.nan)
    for j, k in enumerate(ks):
        window = logret[k - 61:k - 1]                  # the 60 returns ending at close k-1
        sd = window.std(axis=0, ddof=1)
        vol[j] = np.where(np.isfinite(window).all(axis=0) & (sd > 0), sd, np.nan)
    score["voladj60"] = score["mom60"] / vol
    score["voladj120"] = score["mom120"] / vol
    score["accel"] = score["mom20"] - ((1 + score["mom120"]) ** (20 / 120) - 1)

    soxx = book.adj["SOXX"].to_numpy(float)[:, None]
    absolute = {}
    for n in (20, 60):
        own, ref = score[f"mom{n}"], ret(soxx, n)
        absolute[n, "zero"] = np.nan_to_num(own, nan=-np.inf) > 0
        absolute[n, "SOXX"] = np.nan_to_num(own, nan=-np.inf) > ref

    def previous(frame):
        return frame.reindex(index=book.days, columns=tickers).shift().loc[days].to_numpy(float)

    pools, tw = [], []
    for day in days:
        pdf = S.latest_before(book.pdfs, day)
        w = book.weights(pdf) if pdf else {}
        pools.append([key for key in sorted(w) if w[key] > 0 and key in col])
        tw.append({key: w[key] for key in pools[-1]})

    gates = study.market_gates(book, days)
    shifted = breadth.shift().reindex(days)
    assert shifted.notna().all()
    gates["breadth19"] = (shifted >= BREADTH_SPLIT).to_numpy()
    return dict(score=score, absolute=absolute, trend=previous(trend_df), giant=previous(giant_df),
                pools=pools, tw=tw, gates=gates, days=days, col=col, sector=book.sector, cache={})


def orders(sig, name):
    """Ranked keys per session; stable sorts keep the base's key order on ties."""
    if name in sig["cache"]:
        return sig["cache"][name]
    col, out = sig["col"], []
    for j, pool in enumerate(sig["pools"]):
        if name == "avgrank":
            parts = [sig["score"][f"mom{n}"][j] for n in (20, 60, 120)]
            keys = [k for k in pool if all(np.isfinite(p[col[k]]) for p in parts)]
            total = dict.fromkeys(keys, 0)
            for p in parts:
                for rank, k in enumerate(sorted(keys, key=lambda k: -p[col[k]]), 1):
                    total[k] += rank
            out.append(sorted(keys, key=lambda k: total[k]))
        elif name.startswith(("trend|", "giant|")):
            panel, base, missing = name.split("|")
            grade, row = sig[panel][j], sig["score"][base][j]
            keys = [k for k in pool if np.isfinite(row[col[k]])
                    and (missing == "neutral" or np.isfinite(grade[col[k]]))]
            g = {k: grade[col[k]] if np.isfinite(grade[col[k]]) else NEUTRAL for k in keys}
            out.append(sorted(keys, key=lambda k: (-g[k], -row[col[k]])))
        else:
            row = sig["score"][name][j]
            keys = [k for k in pool if np.isfinite(row[col[k]])]
            out.append(sorted(keys, key=lambda k: -row[col[k]]))
    sig["cache"][name] = out
    return out


def select(order, current, n, buf):
    """desk.strategies.select with explicit n and buffer."""
    keep = [k for k in order[:n + buf] if k in current][:n]
    for k in order:
        if len(keep) >= n:
            break
        if k not in keep:
            keep.append(k)
    return keep


def plan(sig, order_name, n=7, buf=5, freq="week", weight="time", eligible=None, mode="rotate"):
    """study.plan generalised: ranking, N, buffer and weights are parameters.

    mode "rotate": names failing `eligible` are removed before selection.
    mode "cash": the basket is chosen without the filter on weekly dates and the weight of a
    failing name stays in USD cash (checked weekly, or every session when freq == "change").
    """
    days, col, ranked_all = sig["days"], sig["col"], orders(sig, order_name)
    prev, last_pass, targets, started, target = set(), set(), {}, False, {}
    for j, day in enumerate(days):
        week = j == 0 or S.iso_week(day) != S.iso_week(days[j - 1])
        ranked = ranked_all[j]
        if not ranked or (freq == "week" and started and not week):
            continue
        if mode == "cash" and started and not week:
            passed = {k for k in prev if eligible[j, col[k]]}
            if passed == last_pass:
                continue
        else:
            pool = ranked if (mode == "cash" or eligible is None) else \
                [k for k in ranked if eligible[j, col[k]]]
            selected = set(select(pool, prev, n, buf))
            if mode == "rotate" and started and not week and selected == prev:
                continue
            prev = selected
            if weight == "time":
                raw = {k: sig["tw"][j][k] for k in pool if k in selected}
            else:
                raw = {k: 1.0 for k in pool if k in selected}
            total = sum(raw.values())
            target = S.project_caps({k: v / total for k, v in raw.items()}, sig["sector"])
            passed = {k for k in prev if eligible is None or eligible[j, col[k]]}
        weights = np.zeros(len(col))
        for k in passed:
            weights[col[k]] = target.get(k, 0.)
        targets[j] = weights
        last_pass, started = passed, True
    return targets


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------

def inner_grid(sig):
    """(key, config, plan kwargs) for R1-R4. `r5` marks the configs crossed with gates."""
    blank = dict(score="", lookback=0, n=7, buffer=5, freq="week", weight="time", abs_lookback=0,
                 abs_ref="", abs_mode="", grade_rule="", missing="", gate="none", inner="")
    for lookback in LOOKBACKS:
        for n in NS:
            for buf in BUFFERS:
                for freq in FREQS:
                    for weight in WEIGHTS:
                        base = (lookback, n, buf, freq, weight) == (120, 7, 5, "week", "time")
                        key = "base" if base else f"r1_L{lookback}_N{n}_b{buf}_{freq}_{weight}"
                        cfg = {**blank, "subfamily": "R1", "score": "mom", "lookback": lookback, "n": n,
                               "buffer": buf, "freq": freq, "weight": weight}
                        yield key, cfg, dict(order_name=f"mom{lookback}", n=n, buf=buf, freq=freq,
                                             weight=weight), freq == "week" and weight == "time"
    for name in COMPOSITES:
        for n in NS:
            for freq in FREQS:
                cfg = {**blank, "subfamily": "R2", "score": name, "n": n, "freq": freq}
                yield f"r2_{name}_N{n}_{freq}", cfg, dict(order_name=name, n=n, freq=freq), True
    for lookback in (20, 60, 120):
        for abs_n in (20, 60):
            for ref in ("zero", "SOXX"):
                for mode in ("cash", "rotate"):
                    for freq in FREQS:
                        cfg = {**blank, "subfamily": "R3", "score": "mom", "lookback": lookback, "freq": freq,
                               "abs_lookback": abs_n, "abs_ref": ref, "abs_mode": mode}
                        yield (f"r3_L{lookback}_abs{abs_n}{ref}_{mode}_{freq}", cfg,
                               dict(order_name=f"mom{lookback}", freq=freq,
                                    eligible=sig["absolute"][abs_n, ref], mode=mode), True)
    for panel in ("trend", "giant"):
        for lookback in (20, 60, 120):
            for missing in ("drop", "neutral"):
                for freq in FREQS:
                    cfg = {**blank, "subfamily": "R4", "score": "mom", "lookback": lookback, "freq": freq,
                           "grade_rule": f"{panel}_then_mom", "missing": missing}
                    yield (f"r4_{panel}first_L{lookback}_{missing}_{freq}", cfg,
                           dict(order_name=f"{panel}|mom{lookback}|{missing}", freq=freq), True)
    for lookback in (20, 60):
        for missing in ("drop", "pass"):
            grade = sig["trend"]
            ok = (np.nan_to_num(grade, nan=-1.) >= 3) | (np.isnan(grade) if missing == "pass" else False)
            for freq in FREQS:
                cfg = {**blank, "subfamily": "R4", "score": "mom", "lookback": lookback, "freq": freq,
                       "grade_rule": "trendA_required", "missing": missing}
                yield (f"r4_trendAreq_L{lookback}_{missing}_{freq}", cfg,
                       dict(order_name=f"mom{lookback}", freq=freq, eligible=ok, mode="rotate"), True)


def all_targets(sig):
    return {key: (cfg, plan(sig, **kwargs), r5) for key, cfg, kwargs, r5 in inner_grid(sig)}


def same_targets(a, b, upto=None):
    ka = sorted(j for j in a if upto is None or j <= upto)
    kb = sorted(j for j in b if upto is None or j <= upto)
    return ka == kb and all(np.array_equal(a[j], b[j]) for j in ka)


# ---------------------------------------------------------------------------
# look-ahead self-test
# ---------------------------------------------------------------------------

def perturbed(ctx, trend_df, giant_df, breadth, mode):
    """Change everything observed at or after the close of session T_TEST.

    "x1.5": every adjusted close, close and open from session T on is multiplied by 1.5 (the
    lead's test; it preserves cross-sectional return ranks, so it mainly tests the gates and
    the absolute filters). "random": each name and session gets its own seeded factor in
    [0.5, 2], which reorders every ranking. Grades from close T on are remapped and breadth
    is mirrored, so a missing one-session shift would be caught as well.
    """
    book, t0 = ctx.book, ctx.start + T_TEST
    rng = np.random.default_rng(20261119)
    frames = {}
    for name in ("adj", "cls", "opn"):
        frame = getattr(book, name).copy()
        factor = np.full(frame.shape, 1.5) if mode == "x1.5" else rng.uniform(.5, 2., frame.shape)
        factor[:t0] = 1.
        frames[name] = frame * factor
    late = ctx.days[T_TEST]
    grades = []
    for frame in (trend_df, giant_df):
        changed = frame.copy()
        rows = changed.index >= late
        changed.loc[rows] = ((changed.loc[rows] + 2) % 5).fillna(4.)
        grades.append(changed)
    mirrored = breadth.copy()
    mirrored[mirrored.index >= late] = 100 - mirrored[mirrored.index >= late]
    return build(dataclasses.replace(book, **frames), ctx.days, ctx.start, ctx.tickers, ctx.col,
                 *grades, mirrored)


def lookahead_test(ctx, sig, reference, trend_df, giant_df, breadth):
    out = {}
    for mode in ("x1.5", "random"):
        other = perturbed(ctx, trend_df, giant_df, breadth, mode)
        for name in ("none", *OUTER):
            assert np.array_equal(sig["gates"][name][:T_TEST + 1], other["gates"][name][:T_TEST + 1]), name
        later = 0
        for key, (_, targets, _) in all_targets(other).items():
            assert same_targets(reference[key][1], targets, upto=T_TEST), f"look-ahead in {key} ({mode})"
            later += not same_targets(reference[key][1], targets)
        gates_changed = sum(not np.array_equal(sig["gates"][g], other["gates"][g]) for g in OUTER)
        out[mode] = dict(configs=len(reference), identical_through_T=len(reference),
                         configs_changed_after_T=later, gates_changed_after_T=gates_changed)
    return out


# ---------------------------------------------------------------------------
# build and save
# ---------------------------------------------------------------------------

def run():
    ctx = common.context()
    assert config.RULE == {"lookback": 120, "n": 7, "buffer": 5, "stock_cap": .25, "sector_cap": .6,
                           "semis_cap": .5}
    trend_df = pd.read_csv(common.CODEX / "trend_grades.csv", index_col=0)
    giant_df = pd.read_csv(common.CODEX / "giant_grades.csv", index_col=0)
    breadth = pd.read_csv(common.CODEX / "breadth.csv", index_col=0).breadth
    sig = build(ctx.book, ctx.days, ctx.start, ctx.tickers, ctx.col, trend_df, giant_df, breadth)
    for name, gate in ctx.gates.items():
        assert np.array_equal(gate, sig["gates"][name]), name
    for j, ranked in enumerate(ctx.rankings):      # the re-implemented ranking is the desk's
        assert [r["key"] for r in ranked] == orders(sig, "mom120")[j], ctx.days[j]

    inner = all_targets(sig)
    args = (ctx.days, ctx.opn, ctx.close, ctx.fx)
    assert same_targets(inner["base"][1], ctx.weekly)
    fast = study.plan(ctx.book, ctx.days, ctx.rankings, ctx.col, ctx.ok["any"], "change", "rotate")
    assert same_targets(inner["r1_L120_N7_b5_change_time"][1], fast)       # Codex's fast_none
    cash = study.plan(ctx.book, ctx.days, ctx.rankings, ctx.col, ctx.ok["trendA"], "change", "cash")
    probe = plan(sig, "mom120", freq="change", eligible=ctx.ok["trendA"], mode="cash")
    assert same_targets(probe, cash)                                        # cash disposition
    rotate = study.plan(ctx.book, ctx.days, ctx.rankings, ctx.col, ctx.ok["trendA"], "week", "rotate")
    assert same_targets(plan(sig, "mom120", eligible=ctx.ok["trendA"]), rotate)

    lookahead = lookahead_test(ctx, sig, inner, trend_df, giant_df, breadth)

    # R4 twins: a "missing" variant whose targets equal its sibling is not a separate candidate.
    dropped = []
    for key in sorted(inner):
        for a, b in (("_neutral_", "_drop_"), ("_pass_", "_drop_")):
            twin = key.replace(a, b)
            if key.startswith("r4_") and a in key and same_targets(inner[key][1], inner[twin][1]):
                dropped.append(key)
    for key in dropped:
        del inner[key]

    curves, configs, plans = {}, {}, {}
    for key, (cfg, targets, r5) in inner.items():
        configs[key], plans[key] = cfg, targets
        curves[key] = study.replay(*args, targets, sig["gates"]["none"])
        if r5:
            for gate in OUTER:
                name = f"{key}__{gate}"
                configs[name] = {**cfg, "subfamily": "R5", "gate": gate, "inner": cfg["subfamily"]}
                curves[name] = study.replay(*args, targets, sig["gates"][gate])
    reference = study.replay(*args, ctx.weekly, ctx.gates["none"])
    np.testing.assert_allclose(curves["base"].to_numpy(), reference.to_numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(curves["base"].usd, ctx.baseline.close, rtol=1e-10)
    stats = common.window_stats
    h1, h2 = stats(curves["base"], *common.WINDOWS["h1"]), stats(curves["base"], *common.WINDOWS["h2"])
    assert abs(h1["krw"] - 2.4505) < 5e-5 and abs(h2["krw"] + .3945) < 5e-5, (h1, h2)
    for key, frame in curves.items():
        assert frame.exposure.between(-1e-9, 1 + 1e-9).all(), key
    common.save(FAMILY, curves, configs)
    return ctx, sig, curves, configs, plans, dict(lookahead=lookahead, dropped_twins=dropped,
                                                  base_h1_krw=h1["krw"], base_h2_krw=h2["krw"])


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def cost_stats(curve, window):
    """Return of the 10bp-cost ledgers inside a window (KRW and USD)."""
    a, b = common.WINDOWS[window]
    ix = np.flatnonzero((curve.index >= a) & (curve.index <= b))
    usd10 = (curve.usd * curve.krw10 / curve.krw).to_numpy()
    krw10 = curve.krw10.to_numpy()
    out = {}
    for name, values in (("krw10", krw10), ("usd10", usd10)):
        before = values[ix[0] - 1] if ix[0] else 1.
        path = np.r_[before, values[ix]]
        out[name] = path[-1] / before - 1
        out[f"{name}_mdd"] = np.min(path / np.maximum.accumulate(path) - 1)
    return out


def weekly_usd(curve, window="h2"):
    """USD return per ISO week clipped to the window (anchor: the close before each week)."""
    a, b = common.WINDOWS[window]
    days = list(curve.index)
    ix = [i for i, d in enumerate(days) if a <= d <= b]
    weeks = {}
    for i in ix:
        weeks.setdefault(S.iso_week(days[i]), []).append(i)
    usd = curve.usd.to_numpy()
    return pd.Series({days[v[0]]: usd[v[-1]] / usd[v[0] - 1] - 1 for v in weeks.values()})


def week_compare(base_weeks, other_weeks):
    lose, win = base_weeks < 0, base_weeks > 0
    return dict(base_losing_weeks=int(lose.sum()), base_winning_weeks=int(win.sum()),
                losing_weeks_turned_nonnegative=int((other_weeks[lose] >= 0).sum()),
                losing_weeks_smaller_loss=int((other_weeks[lose] > base_weeks[lose]).sum()),
                winning_weeks_turned_nonpositive=int((other_weeks[win] <= 0).sum()),
                winning_weeks_smaller_gain=int((other_weeks[win] < base_weeks[win]).sum()),
                sum_diff_in_base_losing_weeks=float((other_weeks[lose] - base_weeks[lose]).sum()),
                sum_diff_in_base_winning_weeks=float((other_weeks[win] - base_weeks[win]).sum()))


def name_weeks(ctx, targets, window="h2"):
    """{(rebalance session, column): open-to-next-rebalance-open return} for weekly plans."""
    a, b = common.WINDOWS[window]
    sessions = sorted(j for j in targets if a <= ctx.days[j] <= b)
    every = sorted(targets)
    out = {}
    for j in sessions:
        later = [x for x in every if x > j]
        for c in np.flatnonzero(targets[j] > 0):
            end = ctx.opn[later[0], c] if later else ctx.close[-1, c]
            out[j, c] = end / ctx.opn[j, c] - 1
    return out


def position_compare(base_nw, other_nw):
    lose = {k for k, v in base_nw.items() if v < 0}
    win = {k for k, v in base_nw.items() if v > 0}
    new = {k: v for k, v in other_nw.items() if k not in base_nw}
    return dict(base_losing_name_weeks=len(lose), base_winning_name_weeks=len(win),
                losers_not_held=len(lose - set(other_nw)), winners_not_held=len(win - set(other_nw)),
                new_name_weeks=len(new), new_losers=sum(v < 0 for v in new.values()),
                new_winners=sum(v > 0 for v in new.values()),
                mean_ret_base_dropped=float(np.mean([base_nw[k] for k in base_nw if k not in other_nw]))
                if any(k not in other_nw for k in base_nw) else float("nan"),
                mean_ret_new=float(np.mean(list(new.values()))) if new else float("nan"))


def describe(key, table, curves, train, test):
    r, c = table.loc[key], cost_stats(curves[key], test)
    control = common.matched_control(curves["base"], r[f"{test}_expo"], test)
    return dict(strategy=key, train=train, test=test,
                train_krw=r[f"{train}_krw"], train_krw_mdd=r[f"{train}_krw_mdd"],
                train_usd=r[f"{train}_usd"], train_usd_mdd=r[f"{train}_usd_mdd"],
                test_krw=r[f"{test}_krw"], test_krw_mdd=r[f"{test}_krw_mdd"],
                test_usd=r[f"{test}_usd"], test_usd_mdd=r[f"{test}_usd_mdd"],
                test_expo=r[f"{test}_expo"], test_turn=r[f"{test}_turn"],
                test_krw_10bp=c["krw10"], test_usd_10bp=c["usd10"],
                matched_control_usd=control[0], matched_control_usd_mdd=control[1])


def report(ctx, curves, configs, plans, checks):
    table = common.stats_table(curves)
    cfg = pd.DataFrame.from_dict(configs, orient="index")
    base = table.loc["base"]
    out = dict(checks=checks, n_candidates=len(curves),
               by_subfamily_count=cfg.subfamily.value_counts().sort_index().to_dict())
    out["base"] = {k: float(base[k]) for k in table.columns if k.split("_")[-1] in ("krw", "usd", "mdd", "expo", "turn")}
    out["base_cost"] = {w: cost_stats(curves["base"], w) for w in ("h2", "sep_nov")}

    pools = {"all": table, "no_breadth_gate": table[(cfg.gate != "breadth19").reindex(table.index)],
             "ungated_R1_R4": table[(cfg.gate == "none").reindex(table.index)]}
    selections = []
    for pool_name, pool in pools.items():
        for protocol, (train, test) in (("P1", ("h1", "h2")), ("P2", ("jan_aug", "sep_nov"))):
            for rule, key in common.select(pool, train).items():
                selections.append(dict(pool=pool_name, protocol=protocol, rule=rule,
                                       **describe(key, table, curves, train, test)))
    out["selections"] = selections
    out["hindsight"] = dict(
        best_h2_usd=describe(table.h2_usd.idxmax(), table, curves, "h1", "h2"),
        best_sep_nov_usd=describe(table.sep_nov_usd.idxmax(), table, curves, "jan_aug", "sep_nov"),
        best_h2_usd_ungated=describe(pools["ungated_R1_R4"].h2_usd.idxmax(), table, curves, "h1", "h2"),
        best_sep_nov_usd_ungated=describe(pools["ungated_R1_R4"].sep_nov_usd.idxmax(), table, curves,
                                          "jan_aug", "sep_nov"))

    control = {k: common.matched_control(curves["base"], table.at[k, "h2_expo"], "h2")[0] for k in table.index}
    table["h2_control_usd"] = pd.Series(control)
    table["h2_usd10"] = pd.Series({k: cost_stats(curves[k], "h2")["usd10"] for k in table.index})
    cand = table.drop(index="base")
    both = cand[(cand.h2_usd > base.h2_usd) & (cand.h2_usd_mdd > base.h2_usd_mdd)]
    out["counts"] = dict(
        total=len(cand), beat_base_both_h2_usd=len(both),
        also_beat_matched_control=int((both.h2_usd > both.h2_control_usd).sum()),
        control_undefined_exposure_above_base=int(both.h2_control_usd.isna().sum()),
        beat_base_both_with_10bp=int(((cand.h2_usd10 > cost_stats(curves["base"], "h2")["usd10"])
                                      & (cand.h2_usd_mdd > base.h2_usd_mdd)).sum()),
        all_beating_own_control=int((cand.h2_usd > cand.h2_control_usd).sum()),
        all_with_control_defined=int(cand.h2_control_usd.notna().sum()))

    def group(frame, by, label):
        rows = []
        joined = frame.join(cfg)
        for value, g in joined.groupby(by, sort=True):
            beat = g[(g.h2_usd > base.h2_usd) & (g.h2_usd_mdd > base.h2_usd_mdd)]
            rows.append(dict(group=label, value="|".join(map(str, value)) if isinstance(value, tuple) else str(value),
                             n=len(g), median_h2_usd=g.h2_usd.median(), best_h2_usd=g.h2_usd.max(),
                             worst_h2_usd=g.h2_usd.min(), median_sep_nov_usd=g.sep_nov_usd.median(),
                             median_h2_expo=g.h2_expo.median(), median_h2_turn=g.h2_turn.median(),
                             median_h1_krw=g.h1_krw.median(), median_h2_usd_10bp=g.h2_usd10.median(),
                             beat_base_both=len(beat),
                             beat_control=int((beat.h2_usd > beat.h2_control_usd).sum()),
                             median_excess_vs_control=(g.h2_usd - g.h2_control_usd).median()))
        return rows

    ungated = cand[(cfg.gate == "none").reindex(cand.index)]
    gated = cand[(cfg.gate != "none").reindex(cand.index)]
    r1 = ungated[(cfg.subfamily == "R1").reindex(ungated.index)]
    full = table  # base counted inside its own R1 cells
    r1b = full[((cfg.subfamily == "R1") & (cfg.gate == "none")).reindex(full.index)]
    out["subfamily"] = group(full, "subfamily", "subfamily")
    out["groups"] = (group(r1b, "lookback", "R1 lookback") + group(r1b, "n", "R1 N")
                     + group(r1b, "buffer", "R1 buffer") + group(r1b, "freq", "R1 freq")
                     + group(r1b, "weight", "R1 weight")
                     + group(ungated[(cfg.subfamily == "R2").reindex(ungated.index)], "score", "R2 score")
                     + group(ungated[(cfg.subfamily == "R3").reindex(ungated.index)],
                             ["abs_ref", "abs_mode"], "R3 ref|mode")
                     + group(ungated[(cfg.subfamily == "R4").reindex(ungated.index)], "grade_rule", "R4 rule")
                     + group(gated, "gate", "R5 gate") + group(gated, "inner", "R5 inner"))
    del r1

    # Base's H2 weeks against the faster rankings (same N, buffer, weights and weekly schedule).
    base_weeks = weekly_usd(curves["base"])
    base_nw = name_weeks(ctx, plans["base"])
    headline = sorted({s["strategy"] for s in selections} | {v["strategy"] for v in out["hindsight"].values()})
    weeks, positions = {}, {}
    for lookback in LOOKBACKS[:-1]:
        for freq in FREQS:
            key = f"r1_L{lookback}_N7_b5_{freq}_time"
            weeks[key] = week_compare(base_weeks, weekly_usd(curves[key]))
        key = f"r1_L{lookback}_N7_b5_week_time"
        positions[key] = position_compare(base_nw, name_weeks(ctx, plans[key]))
    for key in ("r2_avgrank_N7_week", "r2_skip5_N7_week", "r2_voladj60_N7_week", "r2_voladj120_N7_week",
                "r2_accel_N7_week", "r4_trendfirst_L120_drop_week", "r4_giantfirst_L120_drop_week",
                "r4_giantfirst_L120_neutral_week", "r4_trendAreq_L60_drop_week"):
        if key in curves:
            weeks[key] = week_compare(base_weeks, weekly_usd(curves[key]))
            positions[key] = position_compare(base_nw, name_weeks(ctx, plans[key]))
    for key in headline:
        weeks.setdefault(key, week_compare(base_weeks, weekly_usd(curves[key])))
    out["base_h2_weeks"] = dict(n=len(base_weeks), losing=int((base_weeks < 0).sum()),
                                winning=int((base_weeks > 0).sum()),
                                sum_losing=float(base_weeks[base_weeks < 0].sum()),
                                sum_winning=float(base_weeks[base_weeks > 0].sum()))
    out["week_compare"], out["position_compare"] = weeks, positions

    rows = []
    for key in sorted(set(headline) | {"base", "r1_L120_N7_b5_change_time"}
                      | {f"r1_L{n}_N7_b5_week_time" for n in LOOKBACKS[:-1]}
                      | {f"r1_L{n}_N7_b5_change_time" for n in LOOKBACKS[:-1]}):
        r = table.loc[key]
        rows.append(dict(strategy=key, h1_krw=r.h1_krw, h2_krw=r.h2_krw, h2_usd=r.h2_usd, h2_usd_mdd=r.h2_usd_mdd,
                         h2_expo=r.h2_expo, h2_turn=r.h2_turn, h2_usd_10bp=r.h2_usd10,
                         h2_control_usd=r.h2_control_usd, sep_nov_usd=r.sep_nov_usd,
                         sep_nov_usd_mdd=r.sep_nov_usd_mdd, jan_aug_krw=r.jan_aug_krw,
                         **{f"sep_nov_{k}": v for k, v in cost_stats(curves[key], "sep_nov").items()
                            if k == "usd10"}))
    out["headline"] = rows
    out["extras"] = extras(table.join(cfg))
    return out, table.join(cfg)


def extras(t):
    """Checks against Codex's published numbers and the paired view: same gate, base ranking."""
    b, c = t.loc["base"], t.drop(index="base")
    ung = t[t.gate == "none"]
    out = dict(
        cross_checks={k: dict(h1_krw=t.at[k, "h1_krw"], h2_krw=t.at[k, "h2_krw"], h2_expo=t.at[k, "h2_expo"],
                              sep_nov_krw=t.at[k, "sep_nov_krw"], sep_nov_usd=t.at[k, "sep_nov_usd"],
                              sep_nov_usd_mdd=t.at[k, "sep_nov_usd_mdd"])
                      for k in ("base__breadth19", "base__SOXX100r1", "base__SOXX10r1",
                                "r1_L120_N7_b5_change_time")},
        positive_h2_usd=int((c.h2_usd > 0).sum()), positive_h2_krw=int((c.h2_krw > 0).sum()),
        positive_sep_nov_usd=int((c.sep_nov_usd > 0).sum()),
        beat_base_h1_krw=int((c.h1_krw > b.h1_krw).sum()),
        beat_base_h1_and_h2_krw=int(((c.h1_krw > b.h1_krw) & (c.h2_krw > b.h2_krw)).sum()),
        beat_base_full_krw=int((c.full_krw > b.full_krw).sum()),
        beat_base_full_krw_and_mdd=int(((c.full_krw > b.full_krw) & (c.full_krw_mdd > b.full_krw_mdd)).sum()),
        ungated_n=len(ung) - 1,
        ungated_beat_base_both_h2=int(((ung.h2_usd > b.h2_usd) & (ung.h2_usd_mdd > b.h2_usd_mdd)).sum()),
        ungated_beat_base_jul_aug_usd=int((ung.jul_aug_usd > b.jul_aug_usd).sum()),
        ungated_beat_base_sep_nov_usd=int((ung.sep_nov_usd > b.sep_nov_usd).sum()),
        ungated_beat_base_both_subperiods=int(((ung.jul_aug_usd > b.jul_aug_usd)
                                               & (ung.sep_nov_usd > b.sep_nov_usd)).sum()),
        ungated_beat_base_h1_krw=int((ung.h1_krw > b.h1_krw).sum()),
        ungated_h2_usd_quantiles={str(q): float(ung.h2_usd.quantile(q)) for q in (0, .1, .25, .5, .75, .9, 1)},
        spearman_ungated={f"{x}~{y}": float(ung[x].corr(ung[y], method="spearman"))
                          for x, y in (("h1_krw", "h2_usd"), ("jan_aug_krw", "sep_nov_usd"),
                                       ("jul_aug_usd", "sep_nov_usd"))},
        ungated_by_n=[dict(n=int(n), count=len(g), best_h2_usd=g.h2_usd.max(), median_h2_usd=g.h2_usd.median(),
                           median_h1_krw=g.h1_krw.median(), best_h1_krw=g.h1_krw.max(),
                           median_h2_expo=g.h2_expo.median()) for n, g in ung.groupby("n")])
    paired = []
    for gate in ("none", *OUTER):
        ref = t.loc["base" if gate == "none" else f"base__{gate}"]
        g = t[t.gate == gate]
        if gate == "none":      # the same 176 inner rules that R5 crosses with the gates
            g = g[(g.subfamily != "R1") | ((g.freq == "week") & (g.weight == "time"))]
        g = g.drop(index=ref.name)
        paired.append(dict(gate=gate, n=len(g), ref_h2_usd=ref.h2_usd, ref_sep_nov_usd=ref.sep_nov_usd,
                           ref_h1_krw=ref.h1_krw, ref_h2_expo=ref.h2_expo,
                           better_h2_usd=int((g.h2_usd > ref.h2_usd).sum()),
                           better_sep_nov_usd=int((g.sep_nov_usd > ref.sep_nov_usd).sum()),
                           better_jul_aug_and_sep_nov=int(((g.jul_aug_usd > ref.jul_aug_usd)
                                                           & (g.sep_nov_usd > ref.sep_nov_usd)).sum()),
                           better_h1_krw=int((g.h1_krw > ref.h1_krw).sum()),
                           median_diff_h2_usd=(g.h2_usd - ref.h2_usd).median(),
                           median_diff_sep_nov_usd=(g.sep_nov_usd - ref.sep_nov_usd).median(),
                           median_diff_h1_krw=(g.h1_krw - ref.h1_krw).median()))
    out["paired_vs_base_same_gate"] = paired
    return out


def main():
    began = time.time()
    ctx, sig, curves, configs, plans, checks = run()
    out, table = report(ctx, curves, configs, plans, checks)
    out["runtime_seconds"] = time.time() - began
    common.CACHE.mkdir(exist_ok=True)
    table.to_csv(common.CACHE / "rotation_stats.csv")
    text = json.dumps(out, indent=1, ensure_ascii=False,
                      default=lambda v: None if isinstance(v, float) and np.isnan(v) else float(v))
    (common.CACHE / "rotation_report.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
