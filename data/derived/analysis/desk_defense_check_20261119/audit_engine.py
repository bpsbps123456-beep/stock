"""Independent re-derivation of the desk_defense_20261119 ledgers (read-only audit).

Run from the repo root:  py data/derived/analysis/desk_defense_check_20261119/audit_engine.py
Writes only cache/audit_engine.json. Never calls Codex's file-writing functions.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import CODEX, S, study  # noqa: E402

OUT = {}
H1 = ("2026-01-01", "2026-06-30")
H2 = ("2026-07-01", "2026-12-31")
WIN = {"full": ("2026-01-01", "2026-12-31"), "h1": H1, "h2": H2, "jul_aug": ("2026-07-01", "2026-08-31"),
       "sep_latest": ("2026-09-01", "2026-12-31")}


def say(*parts):
    print(*parts, flush=True)


# ---------------------------------------------------------------------------------------
# 1. From-scratch share ledger (dicts and loops, no numpy vector maths, no study.replay)
# ---------------------------------------------------------------------------------------

def _value(position, prices, day):
    total = 0.0
    for i, qty in position.items():
        p = prices[i]
        if not (p == p) or p <= 0:
            raise ValueError(f"held column {i} has no price on {day}")
        total += qty * p
    return total


def simulate(days, O, C, fx, targets, gate, scale=1.0, *, gate_fill="open", slip_flip=0.0, slip_all=0.0,
             flip_days=(), trace=None):
    """Real account + shadow account, both in shares.

    gate_fill="prev_close": a market-gate flip decided for session j trades at close[j-1] instead of open[j].
    slip_flip / slip_all: adverse fill cost as a fraction of traded notional, paid from cash, on gate-flip
    trades only / on every trade. flip_days: extra sessions treated as gate trades (switch family).
    """
    real, cash = {}, 1.0
    shadow, shadow_cash = {}, 1.0
    drag = 1.0
    rows = []
    flip_days = set(flip_days)

    def rebalance(prices, want, slip, day):
        nonlocal real, cash
        nav = cash + _value(real, prices, day)
        old = {i: q * prices[i] / nav for i, q in real.items()}
        turn = sum(abs(want.get(i, 0.0) - old.get(i, 0.0)) for i in set(want) | set(old))
        desired = {i: nav * w / prices[i] for i, w in want.items() if w > 0}
        for _ in range(3 if slip else 1):
            traded = sum(abs(desired.get(i, 0.0) - real.get(i, 0.0)) * prices[i] for i in set(desired) | set(real))
            spend = sum(q * prices[i] for i, q in desired.items())
            cost = slip * traded
            if spend + cost <= nav + 1e-15 or spend == 0:
                break
            factor = (nav - cost) / spend
            desired = {i: q * factor for i, q in desired.items()}
        real, cash = desired, nav - spend - cost
        return turn

    for j, day in enumerate(days):
        flip = j > 0 and bool(gate[j]) != bool(gate[j - 1])
        turn, done = 0.0, False
        if flip and gate_fill == "prev_close":
            P = C[j - 1]
            snav = shadow_cash + _value(shadow, P, day)
            want = {i: scale * q * P[i] / snav for i, q in shadow.items()} if gate[j] else {}
            turn += rebalance(P, want, slip_flip + slip_all, day)
            done = True
        P = O[j]
        nav_open = cash + _value(real, P, day)
        snav = shadow_cash + _value(shadow, P, day)
        if j in targets:
            w = targets[j]
            shadow = {}
            invested = 0.0
            for i in range(len(w)):
                if w[i] > 0:
                    if not (P[i] > 0):
                        raise ValueError(f"no executable open for column {i} on {day}")
                    shadow[i] = snav * w[i] / P[i]
                    invested += w[i]
            shadow_cash = snav * (1 - invested)
        flip_trade = flip and not done
        if (gate[j] and j in targets) or flip_trade:
            want = {i: scale * q * P[i] / snav for i, q in shadow.items()} if gate[j] else {}
            slip = slip_all + (slip_flip if (flip_trade or j in flip_days) else 0.0)
            t = rebalance(P, want, slip, day)
            turn += t
            drag *= 1 - 0.001 * t
        if trace is not None:
            after = cash + _value(real, P, day)
            trace.append({i: q * P[i] / after for i, q in real.items()})
        usd = cash + _value(real, C[j], day)
        krw = usd * fx[j]
        rows.append((krw, usd, turn, 1 - cash / nav_open, krw * drag))
    return pd.DataFrame(rows, index=days, columns=common.LEDGER)


def window(values, days, start, end):
    idx = [i for i, d in enumerate(days) if start <= d <= end]
    before = values[idx[0] - 1] if idx[0] > 0 else 1.0
    peak, mdd = before, 0.0
    for i in idx:
        peak = max(peak, values[i])
        mdd = min(mdd, values[i] / peak - 1)
    return values[idx[-1]] / before - 1, mdd, idx


def my_summary(curve, days):
    out = {}
    for name, (a, b) in WIN.items():
        for column, suffix in (("krw", ""), ("krw10", "_10bp"), ("usd", "_usd")):
            ret, mdd, idx = window(curve[column].tolist(), days, a, b)
            out[f"{name}_return{suffix}"], out[f"{name}_mdd{suffix}"] = ret, mdd
        out[f"{name}_exposure"] = sum(curve.exposure.iloc[idx]) / len(idx)
        out[f"{name}_turn"] = sum(curve.turn.iloc[idx])
        out[f"{name}_sessions"] = len(idx)
    return out


def maxdiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


# ---------------------------------------------------------------------------------------
# 2. Inputs straight from the frozen CSVs (not through the Book object)
# ---------------------------------------------------------------------------------------

def frozen_inputs(ctx):
    frames = {n: pd.read_csv(CODEX / "frozen" / f"prices_{n}.csv", index_col=0, parse_dates=True)
              for n in ("open", "close", "adj_close")}
    for f in frames.values():
        f.index = f.index.strftime("%Y-%m-%d")
    all_days = [d for d in frames["adj_close"]["QQQ"].dropna().index]
    assert all_days == ctx.book.days, "session calendar differs"
    k0 = all_days.index(ctx.days[0])
    O = frames["open"].reindex(all_days)[ctx.tickers].to_numpy()[k0:]
    C = frames["close"].reindex(all_days)[ctx.tickers].to_numpy()[k0:]
    raw = frames["adj_close"]["KRW=X"].dropna()
    fx_all, last = [], float("nan")
    stamps = list(raw.index)
    p = 0
    stale = 0
    for d in all_days:
        moved = False
        while p < len(stamps) and stamps[p] <= d:
            last = float(raw.iloc[p])
            moved = stamps[p] == d
            p += 1
        stale += (not moved) and d >= ctx.days[0]
        fx_all.append(last)
    fx = np.array(fx_all[k0:]) / fx_all[k0 - 1]
    return dict(frames=frames, all_days=all_days, k0=k0, O=O, C=C, fx=fx, fx_level=fx_all, fx_stale_sessions=int(stale))


# ---------------------------------------------------------------------------------------
# 3. First-principles schedules (desk.strategies only; no study.plan)
# ---------------------------------------------------------------------------------------

def iso(day):
    return date.fromisoformat(day).isocalendar()[:2]


def fp_weekly(ctx):
    book, prev, started, out = ctx.book, set(), False, {}
    for j, day in enumerate(ctx.days):
        k = ctx.start + j
        _, ranked = S.momentum_ranking(book, k, day)
        new_week = (not started) or iso(day) != iso(book.days[k - 1])
        if not ranked or not new_week:
            continue
        chosen = S.select(ranked, prev)
        tw = {r["key"]: r["time_weight"] for r in ranked}
        total = sum(tw[x] for x in chosen)
        out[j] = (list(chosen), S.project_caps({x: tw[x] / total for x in chosen}, book.sector))
        prev, started = set(chosen), True
    return out


def fp_cash(ctx, weekly, passes):
    """Weekly basket; the share of a failing name is USD cash; republish when the passing set changes."""
    out, current, last = {}, None, None
    for j in range(len(ctx.days)):
        fresh = j in weekly
        if fresh:
            current = weekly[j]
        if current is None:
            continue
        chosen, target = current
        passing = frozenset(x for x in chosen if passes(j, x))
        if fresh or passing != last:
            vec = np.zeros(len(ctx.col))
            for x in passing:
                vec[ctx.col[x]] = target.get(x, 0.0)
            out[j] = vec
            last = passing
    return out


def fp_cash_drift(ctx, weekly, passes, O):
    """Alternative: mid-week only the failing names are sold and recovered names bought back at their weekly
    target share of current NAV (bounded by cash); surviving names are NOT rebalanced back to weekly weights."""
    out, current, last = {}, None, None
    shares, cash = {}, 1.0
    capped = 0
    for j in range(len(ctx.days)):
        nav = cash + sum(q * O[j][i] for i, q in shares.items())
        fresh = j in weekly
        if fresh:
            current = weekly[j]
        if current is None:
            continue
        chosen, target = current
        passing = frozenset(x for x in chosen if passes(j, x))
        if not (fresh or passing != last):
            continue
        if fresh:
            w = {ctx.col[x]: target.get(x, 0.0) for x in passing}
        else:
            keep = {ctx.col[x] for x in passing & last}
            w = {i: q * O[j][i] / nav for i, q in shares.items() if i in keep}
            room = 1 - sum(w.values())
            for x in sorted(passing - last):
                add = min(target.get(x, 0.0), max(room, 0.0))
                capped += add < target.get(x, 0.0) - 1e-12
                w[ctx.col[x]] = add
                room -= add
        w = {i: v for i, v in w.items() if v > 0}
        vec = np.zeros(len(ctx.col))
        for i, v in w.items():
            vec[i] = v
        out[j] = vec
        shares = {i: nav * v / O[j][i] for i, v in w.items()}
        cash = nav * (1 - sum(w.values()))
        last = passing
    return out, capped


def compare_targets(a, b):
    keys = set(a) ^ set(b)
    diff = max((float(np.max(np.abs(a[j] - b[j]))) for j in set(a) & set(b)), default=0.0)
    return {"sessions_mine": len(a), "sessions_codex": len(b), "sessions_only_in_one": len(keys), "max_abs_weight_diff": diff}


# ---------------------------------------------------------------------------------------
def main():
    ctx = common.context()
    days = ctx.days
    fz = frozen_inputs(ctx)
    O, C, fx = fz["O"], fz["C"], fz["fx"]
    OUT["inputs"] = {
        "sessions": len(days), "tickers": len(ctx.tickers), "unique_tickers": len(set(ctx.tickers)),
        "max_abs_open_vs_ctx": float(np.nanmax(np.abs(O - ctx.opn))), "max_abs_close_vs_ctx": float(np.nanmax(np.abs(C - ctx.close))),
        "nan_pattern_equal": bool((np.isnan(O) == np.isnan(ctx.opn)).all() and (np.isnan(C) == np.isnan(ctx.close)).all()),
        "max_abs_fx_vs_ctx": maxdiff(fx, ctx.fx), "fx_sessions_using_stale_quote": fz["fx_stale_sessions"],
        "nan_open_cells_2026": int(np.isnan(O).sum()), "nan_close_cells_2026": int(np.isnan(C).sum()),
        "tickers_with_nan_2026": {t: int(np.isnan(O[:, i]).sum()) for i, t in enumerate(ctx.tickers) if np.isnan(O[:, i]).any()},
        "first_pdf": ctx.book.pdfs[0], "ranking_empty_sessions": [d for d, r in zip(days, ctx.rankings) if not r],
    }
    say("INPUTS", json.dumps(OUT["inputs"], ensure_ascii=False))

    # ---- a) all 303 ledgers: my simulator on study.plan schedules vs study.replay, and vs the CSV ----
    sched, mine, codex = {}, {}, {}
    args = (days, ctx.opn, ctx.close, ctx.fx)
    for mode, freq, method, condition in study.selectors():
        targets = study.plan(ctx.book, days, ctx.rankings, ctx.col, ctx.ok[condition], freq, method)
        for market, gate in ctx.gates.items():
            key = "base" if (mode == "weekly" and market == "none") else f"{mode}_{market}"
            sched[key] = (targets, market, 1.0)
        if freq == "change" and method == "rotate" and condition in ("trendA", "giantA", "bothA", "bothB"):
            for market in ("QQQ20r1", "QQQ50r1", "SOXX20r1", "SOXX50r1"):
                sw = study.switched_targets(days, ctx.opn, ctx.weekly, targets, ctx.gates[market])
                sched[f"switch_{condition}_{market}"] = (sw, "none", 1.0)
                sched[f"_alt_{condition}"] = None
                OUT.setdefault("_alt", {})[condition] = targets
        if mode == "weekly":
            sched["half"] = (targets, "none", .5)
    alt_targets = OUT.pop("_alt")
    sched = {k: v for k, v in sched.items() if v is not None}
    worst = dict.fromkeys(common.LEDGER, 0.0)
    worst_key = {}
    for key, (targets, market, scale) in sched.items():
        mine[key] = simulate(days, O, C, fx, targets, ctx.gates[market], scale)
        codex[key] = study.replay(*args, targets, ctx.gates[market], scale)
        for c in common.LEDGER:
            d = maxdiff(mine[key][c], codex[key][c])
            if d > worst[c]:
                worst[c], worst_key[c] = d, key
    OUT["a_all303_vs_study_replay"] = {"strategies": len(sched), "max_abs_diff": worst, "worst_strategy": worst_key}
    say("A all303", json.dumps(OUT["a_all303_vs_study_replay"]))

    daily = pd.read_csv(CODEX / "selected_daily.csv")
    file_diffs = {}
    for key, frame in daily.groupby("strategy"):
        frame = frame.set_index("date")
        assert list(frame.index) == days
        file_diffs[key] = {c: maxdiff(mine[key][c], frame[c]) for c in common.LEDGER}
    OUT["a_selected_daily_csv"] = file_diffs
    required = ["base", "half", "weekly_SOXX100r1", "stock100_cash_SOXX100r1", "stock100_cash_none", "fast_none",
                "giantA_change_cash_none", "giantA_change_cash_SOXX100r1", "bothA_change_cash_SOXX10r1", "switch_bothA_QQQ50r1"]
    assert all(k in file_diffs for k in required), [k for k in required if k not in file_diffs]
    say("A csv max diff over", len(file_diffs), "strategies:",
        {c: max(v[c] for v in file_diffs.values()) for c in common.LEDGER})

    cand = pd.read_csv(CODEX / "candidates.csv", index_col=0)
    summ = pd.DataFrame({k: my_summary(v, days) for k, v in mine.items()}).T
    cols = [c for c in summ.columns if c in cand.columns]
    d = (summ.loc[cand.index, cols].astype(float) - cand[cols].astype(float)).abs()
    OUT["a_candidates_csv"] = {"columns_compared": len(cols), "rows": len(cand), "max_abs_diff": float(d.max().max()),
                               "worst_column": str(d.max().idxmax())}
    say("A candidates.csv", OUT["a_candidates_csv"])

    # ---- a) first-principles schedules ----
    weekly = fp_weekly(ctx)
    weekly_vec = {}
    for j, (chosen, target) in weekly.items():
        v = np.zeros(len(ctx.col))
        for x, w in target.items():
            v[ctx.col[x]] = w
        weekly_vec[j] = v
    close_all = fz["frames"]["close"].reindex(fz["all_days"])
    px = {t: close_all[t].tolist() for t in ctx.tickers}
    k0 = fz["k0"]

    def sma_pass(n):
        def f(j, key):
            series = px[ctx.book.ticker[key]]
            k = k0 + j
            win = series[k - n:k]
            if len(win) < n or any(not (v == v) for v in win):
                return False
            return series[k - 1] >= sum(win) / n
        return f

    giant = pd.read_csv(CODEX / "giant_grades.csv", index_col=0)
    trend = pd.read_csv(CODEX / "trend_grades.csv", index_col=0)

    def grade_pass(frames, floor):
        def f(j, key):
            prev_day = fz["all_days"][k0 + j - 1]
            for frame in frames:
                if prev_day not in frame.index:
                    return False
                g = frame.at[prev_day, ctx.book.ticker[key]]
                if not (g == g) or g < floor:
                    return False
            return True
        return f

    plans = {
        "weekly": (weekly_vec, study.plan(ctx.book, days, ctx.rankings, ctx.col, ctx.ok["any"], "week", "rotate")),
        "stock100_cash": (fp_cash(ctx, weekly, sma_pass(100)),
                          study.plan(ctx.book, days, ctx.rankings, ctx.col, ctx.ok["sma100"], "change", "cash")),
        "stock20_cash": (fp_cash(ctx, weekly, sma_pass(20)),
                         study.plan(ctx.book, days, ctx.rankings, ctx.col, ctx.ok["sma20"], "change", "cash")),
        "giantA_change_cash": (fp_cash(ctx, weekly, grade_pass([giant], 3)),
                               study.plan(ctx.book, days, ctx.rankings, ctx.col, ctx.ok["giantA"], "change", "cash")),
        "bothA_change_cash": (fp_cash(ctx, weekly, grade_pass([giant, trend], 3)),
                              study.plan(ctx.book, days, ctx.rankings, ctx.col, ctx.ok["bothA"], "change", "cash")),
    }
    OUT["a_first_principles_schedules"] = {k: compare_targets(a, b) for k, (a, b) in plans.items()}
    run_model = {days.index(d): t for d, t in ctx.baseline.rebalances.items()}
    OUT["a_first_principles_schedules"]["weekly_vs_run_model"] = {
        "sessions_only_in_one": len(set(run_model) ^ set(weekly)),
        "max_abs_weight_diff": max(abs(weekly[j][1].get(x, 0) - run_model[j].get(x, 0))
                                   for j in set(weekly) & set(run_model) for x in set(weekly[j][1]) | set(run_model[j]))}
    for name, market in (("weekly", "none"), ("stock100_cash", "SOXX100r1"), ("giantA_change_cash", "SOXX100r1"),
                         ("bothA_change_cash", "SOXX10r1")):
        key = "base" if name == "weekly" else f"{name}_{market}"
        fp_curve = simulate(days, O, C, fx, plans[name][0], ctx.gates[market])
        OUT["a_first_principles_schedules"][f"ledger_{key}_vs_csv"] = {
            c: maxdiff(fp_curve[c], daily[daily.strategy == key][c]) for c in common.LEDGER}
    say("A first-principles", json.dumps(OUT["a_first_principles_schedules"]))

    # independent gates
    sym_close = {s: fz["frames"]["close"].reindex(fz["all_days"])[s].tolist() for s in ("QQQ", "SOXX")}
    gate_mismatch = {}
    for name, flags in ctx.gates.items():
        if name == "none":
            continue
        sym = "QQQ" if name.startswith("QQQ") else "SOXX"
        n, conf = int(name[len(sym):].split("r")[0]), int(name.split("r")[1])
        series, state, run, states = sym_close[sym], None, 0, []
        for k in range(len(series)):
            if k + 1 >= n:
                above = series[k] >= sum(series[k + 1 - n:k + 1]) / n
                run = run + 1 if above else 0
                if not above:
                    state = False
                elif run >= conf:
                    state = True
            states.append(state)
        own = [states[k0 + j - 1] for j in range(len(days))]
        gate_mismatch[name] = int(sum(bool(a) != bool(b) for a, b in zip(own, flags)))
    OUT["a_gate_rederivation_mismatches"] = gate_mismatch
    OUT["gate_off_sessions"] = {name: {"h1": int(sum(not g for g, d in zip(flags, days) if d <= H1[1])),
                                       "h2": int(sum(not g for g, d in zip(flags, days) if d >= H2[0])),
                                       "flips_h1": int(sum(flags[j] != flags[j - 1] for j in range(1, len(days)) if days[j] <= H1[1])),
                                       "flips_h2": int(sum(flags[j] != flags[j - 1] for j in range(1, len(days)) if days[j] >= H2[0]))}
                                for name, flags in ctx.gates.items() if name != "none"}
    say("A gates mismatches", gate_mismatch)
    say("gate off sessions", json.dumps(OUT["gate_off_sessions"]))

    # eligibility arrays vs first principles
    elig = {}
    for n in (10, 20, 50, 100):
        f = sma_pass(n)
        own = np.array([[f(j, key) for key in ctx.col] for j in range(len(days))])
        elig[f"sma{n}"] = int((own != ctx.ok[f"sma{n}"]).sum())
    for name, frames, floor in (("giantA", [giant], 3), ("trendA", [trend], 3), ("bothA", [giant, trend], 3), ("bothB", [giant, trend], 2)):
        f = grade_pass(frames, floor)
        own = np.array([[f(j, key) for key in ctx.col] for j in range(len(days))])
        elig[name] = int((own != ctx.ok[name]).sum())
    OUT["a_eligibility_mismatch_cells"] = elig
    say("A eligibility mismatches", elig)

    # ---- b) defect hunt ----
    b = {}
    # stale weekly target on mid-week pass changes
    for name, passes, cond in (("stock100_cash", sma_pass(100), "sma100"), ("stock50_cash", sma_pass(50), "sma50"),
                               ("stock20_cash", sma_pass(20), "sma20"), ("stock10_cash", sma_pass(10), "sma10"),
                               ("giantA_change_cash", grade_pass([giant], 3), "giantA"),
                               ("bothA_change_cash", grade_pass([giant, trend], 3), "bothA")):
        codex_t = study.plan(ctx.book, days, ctx.rankings, ctx.col, ctx.ok[cond], "change", "cash")
        drift_t, capped = fp_cash_drift(ctx, weekly, passes, O)
        mid = [j for j in codex_t if j not in weekly]
        for market in ("none", "SOXX100r1", "SOXX10r1"):
            a = simulate(days, O, C, fx, codex_t, ctx.gates[market])
            d = simulate(days, O, C, fx, drift_t, ctx.gates[market])
            sa, sd = my_summary(a, days), my_summary(d, days)
            b[f"stale_target_{name}_{market}"] = {
                "midweek_republish_sessions": len(mid), "reentries_capped_by_cash": int(capped),
                "codex": {k: sa[k] for k in ("h1_return", "h2_return", "h2_return_usd", "h2_mdd_usd", "h1_turn", "h2_turn", "h2_exposure", "h2_return_10bp")},
                "survivors_not_rebalanced": {k: sd[k] for k in ("h1_return", "h2_return", "h2_return_usd", "h2_mdd_usd", "h1_turn", "h2_turn", "h2_exposure", "h2_return_10bp")},
                "midweek_turn_codex": float(a.turn.iloc[mid].sum()) if market == "none" else None,
                "midweek_turn_alt": float(d.turn.iloc[mid].sum()) if market == "none" else None,
            }
    # names that leave the ranked TIME list while held
    left = []
    starts = sorted(weekly)
    for a_, nxt in zip(starts, starts[1:] + [len(days)]):
        for x in weekly[a_][0]:
            gone = [days[j] for j in range(a_ + 1, nxt) if x not in {r["key"] for r in ctx.rankings[j]}]
            if gone:
                left.append({"week_start": days[a_], "ticker": ctx.book.ticker[x], "sessions_held_after_leaving": len(gone), "first": gone[0]})
    b["held_names_absent_from_ranked_list_midweek"] = {"events": len(left), "detail": left[:12]}
    b["weekly_targets"] = {"count": len(weekly), "first": days[min(weekly)],
                           "invested_sum_min": float(min(v.sum() for v in weekly_vec.values())),
                           "invested_sum_mean": float(np.mean([v.sum() for v in weekly_vec.values()])),
                           "invested_sum_max": float(max(v.sum() for v in weekly_vec.values())),
                           "weeks_with_fewer_than_7": int(sum(len(c) < 7 for c, _ in weekly.values())),
                           "weeks_not_starting_monday": int(sum(date.fromisoformat(days[j]).weekday() != 0 for j in weekly))}
    for market in ("SOXX100r1", "SOXX10r1", "QQQ50r1"):
        g = ctx.gates[market]
        flips = [j for j in range(1, len(days)) if g[j] != g[j - 1]]
        b[f"gate_flip_on_weekly_target_day_{market}"] = {"flips": len(flips), "on_target_day": int(sum(j in weekly for j in flips)),
                                                        "first_target_gate_on": bool(g[min(weekly)])}
    # drag formula vs an explicit cost paid from cash at each trade
    for key in ("base", "stock100_cash_SOXX100r1", "weekly_SOXX10r1", "switch_bothA_QQQ50r1"):
        targets, market, scale = sched[key]
        exact = simulate(days, O, C, fx, targets, ctx.gates[market], scale, slip_all=.001)
        r_exact = window(exact.krw.tolist(), days, *H2)[0]
        r_drag = window(mine[key].krw10.tolist(), days, *H2)[0]
        b[f"drag_formula_{key}"] = {"h2_krw_drag_formula": r_drag, "h2_krw_cost_paid_from_cash": r_exact, "diff": r_drag - r_exact,
                                    "full_drag": float(mine[key].krw10.iloc[-1] / mine[key].krw.iloc[-1] - 1),
                                    "full_exact": float(exact.krw.iloc[-1] / mine[key].krw.iloc[-1] - 1)}
    # exposure and drawdown conventions
    base = mine["base"]
    close_expo = []
    shares, cash_ = {}, 1.0
    for j in range(len(days)):
        nav = cash_ + sum(q * O[j][i] for i, q in shares.items())
        if j in weekly_vec:
            w = weekly_vec[j]
            shares = {i: nav * w[i] / O[j][i] for i in range(len(w)) if w[i] > 0}
            cash_ = nav * (1 - w.sum())
        stock = sum(q * C[j][i] for i, q in shares.items())
        close_expo.append(stock / (stock + cash_))
    h2i = [i for i, d in enumerate(days) if d >= H2[0]]
    b["exposure_definition_base"] = {"h2_mean_at_open_after_trades": float(base.exposure.iloc[h2i].mean()),
                                     "h2_mean_at_close": float(np.mean([close_expo[i] for i in h2i])),
                                     "h1_mean_at_open": float(base.exposure.iloc[[i for i, d in enumerate(days) if d <= H1[1]]].mean())}
    usd = base.usd.tolist()
    b["drawdown_convention_base"] = {
        "h2_krw_mdd_with_prior_close": window(base.krw.tolist(), days, *H2)[1],
        "h2_krw_mdd_window_only": float(np.min(base.krw.iloc[h2i] / np.maximum.accumulate(base.krw.iloc[h2i]) - 1)),
        "h2_usd_mdd_with_prior_close": window(usd, days, *H2)[1],
        "h2_usd_mdd_window_only": float(np.min(base.usd.iloc[h2i] / np.maximum.accumulate(base.usd.iloc[h2i]) - 1)),
        "h2_usd_mdd_from_running_ytd_peak": float(min(usd[i] / max(usd[:i + 1]) - 1 for i in h2i)),
        "h2_krw_mdd_from_running_ytd_peak": float(min(base.krw.iloc[i] / base.krw.iloc[:i + 1].max() - 1 for i in h2i)),
        "usd_peak_day": days[int(np.argmax(usd))], "krw_peak_day": days[int(base.krw.to_numpy().argmax())],
        "usd_peak_to_last": float(usd[-1] / max(usd) - 1), "krw_peak_to_last": float(base.krw.iloc[-1] / base.krw.max() - 1)}
    # selection and ties (recomputed from my ledgers)
    table = summ.drop(index=[k for k in summ.index if k in ("base", "half")])
    ref = summ.loc["base"]
    eligible = table[table.h1_mdd >= ref.h1_mdd - 1e-9]
    pool = pd.concat([eligible, summ.loc[["base"]]])
    order = pool.sort_values("h1_return", ascending=False)
    retain = pool[(pool.h1_return >= .75 * ref.h1_return - 1e-12) | (pool.index == "base")]
    shallow = retain[retain.h1_mdd >= retain.h1_mdd.max() - 1e-12]
    dup = summ.round(12).duplicated(subset=["h1_return", "h1_mdd"], keep=False)
    b["selection"] = {
        "base_h1_mdd": float(ref.h1_mdd), "eligible": int(len(eligible)), "candidates": int(len(table)),
        "ineligible": [k for k in table.index if k not in eligible.index],
        "return_rule_top5": [(k, float(v)) for k, v in order.h1_return.head(5).items()],
        "return_rule_pick": str(order.index[0]), "matches_codex_flag": bool(cand.loc[order.index[0], "global_selection"]),
        "stability_retained": int(len(retain)), "stability_tied_on_mdd": [(k, float(shallow.loc[k, "h1_mdd"]), float(shallow.loc[k, "h1_return"])) for k in shallow.index],
        "stability_pick": str(shallow.sort_values("h1_return", ascending=False).index[0]),
        "stability_matches_codex_flag": bool(cand.loc[shallow.sort_values("h1_return", ascending=False).index[0], "stability_selection"]),
        "stability_next5_by_mdd": [(k, float(retain.loc[k, "h1_mdd"]), float(retain.loc[k, "h1_return"]), float(retain.loc[k, "h2_return"]))
                                   for k in retain.sort_values("h1_mdd", ascending=False).index[:6]],
        "strategies_sharing_identical_h1_return_and_mdd": int(dup.sum()),
        "strategies_identical_to_base_in_h1": [k for k in summ.index if k != "base" and abs(summ.loc[k, "h1_return"] - ref.h1_return) < 1e-12],
        "h2_double_winners": int(((table.h2_return > ref.h2_return + 1e-9) & (table.h2_mdd > ref.h2_mdd + 1e-9)).sum()),
        "h2_double_winners_usd": int(((table.h2_return_usd > ref.h2_return_usd + 1e-9) & (table.h2_mdd_usd > ref.h2_mdd_usd + 1e-9)).sum()),
    }
    # matched control recomputed
    mc = {}
    base_r = [usd[i] / usd[i - 1] - 1 for i in h2i]
    normal = float(base.exposure.iloc[h2i].mean())
    for key in ("half", "stock100_cash_SOXX100r1", "weekly_SOXX100r1", "giantA_change_cash_SOXX100r1", "bothA_change_cash_SOXX10r1", "switch_bothA_QQQ50r1"):
        f = summ.loc[key, "h2_exposure"] / normal
        nav, peak, mdd = 1.0, 1.0, 0.0
        for r in base_r:
            nav *= 1 + f * r
            peak = max(peak, nav)
            mdd = min(mdd, nav / peak - 1)
        mc[key] = {"fraction": float(f), "mine_ret": nav - 1, "mine_mdd": mdd,
                   "codex_ret": float(cand.loc[key, "h2_matched_return_usd"]), "codex_mdd": float(cand.loc[key, "h2_matched_mdd_usd"]),
                   "candidate_usd": float(summ.loc[key, "h2_return_usd"]),
                   # same average exposure but time-in-market profile: exposure-weighted share of base daily return
                   "candidate_minus_control_pp": float(100 * (summ.loc[key, "h2_return_usd"] - (nav - 1)))}
    b["matched_control"] = mc
    OUT["b_defects"] = b
    for k, v in b.items():
        say("B", k, json.dumps(v, ensure_ascii=False, default=str))

    # ---- c) FX decomposition ----
    c = {}
    i630 = days.index("2026-06-30")
    c["usd_cash_only"] = {"h1_krw": float(fx[i630] - 1), "h2_krw": float(fx[-1] / fx[i630] - 1), "full_krw": float(fx[-1] - 1),
                          "h2_mdd_krw": window(fx.tolist(), days, *H2)[1],
                          "krw_per_usd_start": fz["fx_level"][k0 - 1], "krw_per_usd_0630": fz["fx_level"][k0 + i630], "krw_per_usd_last": fz["fx_level"][-1],
                          "fx_max_abs_daily_move_2026": float(np.max(np.abs(np.diff(np.r_[1., fx]) / np.r_[1., fx[:-1]]))),
                          "fx_sessions_with_stale_quote": fz["fx_stale_sessions"]}
    for key in ("base", "half", "stock100_cash_SOXX100r1", "weekly_SOXX100r1", "weekly_SOXX10r1", "switch_bothA_QQQ50r1",
                "giantA_change_cash_SOXX100r1", "bothA_change_cash_SOXX10r1"):
        row = {}
        for name, (a_, b_) in (("h1", H1), ("h2", H2), ("sep_latest", WIN["sep_latest"])):
            rk = window(mine[key].krw.tolist(), days, a_, b_)[0]
            ru = window(mine[key].usd.tolist(), days, a_, b_)[0]
            rf = window(fx.tolist(), days, a_, b_)[0]
            row[name] = {"krw": rk, "usd": ru, "fx": rf, "identity_error": (1 + ru) * (1 + rf) - 1 - rk,
                         "fx_contribution_pp": 100 * (rk - ru), "krw_recomputed_max_err": maxdiff(mine[key].krw, mine[key].usd * fx)}
        c[key] = row
    OUT["c_fx"] = c
    say("C", json.dumps(c))

    # ---- e) sensitivities ----
    e = {}
    Oprev = O.copy()
    Oprev[1:] = C[:-1]
    headline = ["base", "half", "weekly_SOXX100r1", "stock100_cash_SOXX100r1", "weekly_SOXX10r1", "giantA_change_cash_SOXX100r1",
                "bothA_change_cash_SOXX10r1", "switch_bothA_QQQ50r1"]

    def brief(curve):
        s = my_summary(curve, days)
        return {k: round(float(s[k]), 6) for k in ("h1_return", "h2_return", "h2_return_usd", "h2_mdd_usd", "sep_latest_return_usd", "h2_turn", "h2_exposure")}

    for key in headline:
        targets, market, scale = sched[key]
        gate = ctx.gates[market]
        flip_days, t_prev = (), targets
        if key.startswith("switch_"):
            _, cond, mk = key.split("_")
            g2 = ctx.gates[mk]
            flip_days = [j for j in range(1, len(days)) if g2[j] != g2[j - 1]]
            t_prev = study.switched_targets(days, Oprev, ctx.weekly, alt_targets[cond], g2)
        row = {"next_open(codex)": brief(mine[key])}
        row["all_trades_at_prior_close"] = brief(simulate(days, Oprev, C, fx, t_prev, gate, scale))
        if not key.startswith("switch_"):
            row["gate_flips_at_prior_close"] = brief(simulate(days, O, C, fx, targets, gate, scale, gate_fill="prev_close"))
            lag = np.r_[gate[0], gate[:-1]]
            row["gate_one_session_late"] = brief(simulate(days, O, C, fx, targets, lag, scale))
        row["slip20bp_gate_trades"] = brief(simulate(days, O, C, fx, targets, gate, scale, slip_flip=.002, flip_days=flip_days))
        row["slip20bp_all_trades"] = brief(simulate(days, O, C, fx, targets, gate, scale, slip_all=.002))
        e[key] = row
        say("E", key)
        for k, v in row.items():
            say("   ", f"{k:28s}", json.dumps(v))
    # overnight gap vs intraday split of the base in H2 (what a close fill would have avoided / missed on exit days)
    g = ctx.gates["SOXX100r1"]
    exits = [j for j in range(1, len(days)) if g[j - 1] and not g[j]]
    entries = [j for j in range(1, len(days)) if not g[j - 1] and g[j]]

    def gap(js):
        return [float(ctx.baseline.value[j] / ctx.baseline.close[j - 1] - 1) for j in js]
    e["SOXX100r1_base_overnight_gap_on_flip_days"] = {
        "exit_days": [days[j] for j in exits], "exit_gaps": gap(exits),
        "entry_days": [days[j] for j in entries], "entry_gaps": gap(entries)}
    say("E gaps", json.dumps(e["SOXX100r1_base_overnight_gap_on_flip_days"]))
    OUT["e_sensitivity"] = e

    # ---- f) semantic checks against desk.strategies.run_model (a returns-based engine, not share-based) ----
    f = {}
    bv, bc = ctx.baseline.value, ctx.baseline.close
    worst_sem, worst_w = 0.0, 0.0
    for market, g in ctx.gates.items():
        if market == "none":
            continue
        tr = []
        curve = simulate(days, O, C, fx, ctx.weekly, g, trace=tr)
        u = curve.usd.tolist()
        for j in range(1, len(days)):
            got = u[j] / u[j - 1] - 1
            if g[j] and g[j - 1]:
                want = bc[j] / bc[j - 1] - 1          # invested through: base close-to-close
            elif g[j] and not g[j - 1]:
                want = bc[j] / bv[j] - 1              # re-entry at the open: base open-to-close
            elif not g[j] and g[j - 1]:
                want = bv[j] / bc[j - 1] - 1          # exit at the open: keeps the overnight gap
            else:
                want = 0.0
            worst_sem = max(worst_sem, abs(got - want))
            if g[j] and not g[j - 1]:                 # basket bought back == base's drifted basket that open
                ref_w = {ctx.col[x]: w for x, w in ctx.baseline.weights[j].items()}
                worst_w = max([worst_w] + [abs(tr[j].get(i, 0) - ref_w.get(i, 0)) for i in set(tr[j]) | set(ref_w)])
    f["weekly_gates_daily_return_vs_run_model_max_abs"] = worst_sem
    f["reentry_basket_vs_run_model_drifted_weights_max_abs"] = worst_w

    def spells(g):
        out, start = [], None
        for j, on in enumerate(g):
            if not on and start is None:
                start = j
            if start is not None and (on or j == len(days) - 1):
                end_value = bv[j] if on else bc[j]
                out.append({"exit_open": days[start], "end": days[j], "open_ended": bool(not on), "sessions": j - start + int(not on),
                            "base_usd_return_missed": float(end_value / bv[start] - 1)})
                start = None
        return out

    for market in ("SOXX100r1", "SOXX10r1", "QQQ50r1"):
        sp = spells(ctx.gates[market])
        row = {"spells": sp if market == "SOXX100r1" else len(sp)}
        for name, (a_, b_) in (("h1", H1), ("h2", H2)):
            inside = [s for s in sp if a_ <= s["exit_open"] <= b_]
            factor = float(np.prod([1 / (1 + s["base_usd_return_missed"]) for s in inside]))
            gated = window(mine[f"weekly_{market}"].usd.tolist(), days, a_, b_)[0]
            plain = window(mine["base"].usd.tolist(), days, a_, b_)[0]
            row[name] = {"spells": len(inside), "helped": int(sum(s["base_usd_return_missed"] < 0 for s in inside)),
                         "product_of_spell_factors": factor, "gated_over_base_nav_ratio": (1 + gated) / (1 + plain),
                         "factor_without_last_open_ended_spell": float(np.prod([1 / (1 + s["base_usd_return_missed"]) for s in inside if not s["open_ended"]]))}
        f[f"spells_{market}"] = row
    # family selections
    fam_bad = []
    for family, sub in cand.drop(index=["base", "half"]).groupby("family"):
        ok_ = sub[sub.h1_mdd >= cand.loc["base", "h1_mdd"] - 1e-9]
        pick = "base" if ok_.empty else ok_.index[ok_.h1_return >= ok_.h1_return.max() - 1e-12].min()
        if pick != "base" and not cand.loc[pick, "family_selection"]:
            fam_bad.append((family, pick))
    f["family_selection_mismatches"] = fam_bad
    f["family_sizes"] = {str(k): int(v) for k, v in cand.drop(index=["base", "half"]).groupby("family").size().items()}
    # dividends ignored by the Close-based ledger, and extreme marks among base holdings
    adj = fz["frames"]["adj_close"].reindex(fz["all_days"])
    cls = fz["frames"]["close"].reindex(fz["all_days"])
    missing, worst_gap, extremes = {"h1": 0.0, "h2": 0.0}, 0.0, []
    for j in range(1, len(days)):
        k = k0 + j
        for x, w in ctx.baseline.weights[j].items():
            t = ctx.book.ticker[x]
            ra, rc = adj[t].iloc[k] / adj[t].iloc[k - 1] - 1, cls[t].iloc[k] / cls[t].iloc[k - 1] - 1
            missing["h1" if days[j] <= H1[1] else "h2"] += float(w * (ra - rc))
            worst_gap = max(worst_gap, abs(ra - rc))
            i = ctx.col[x]
            extremes.append((abs(O[j][i] / C[j - 1][i] - 1), days[j], t, "overnight", float(O[j][i] / C[j - 1][i] - 1), round(w, 4)))
            extremes.append((abs(C[j][i] / O[j][i] - 1), days[j], t, "intraday", float(C[j][i] / O[j][i] - 1), round(w, 4)))
    f["dividends_ignored_weighted_sum"] = missing
    f["max_abs_adj_vs_close_daily_return_gap_on_held_names"] = float(worst_gap)
    f["largest_moves_on_base_holdings"] = [e_[1:] for e_ in sorted(extremes, reverse=True)[:8]]
    OUT["f_semantics"] = f
    for k, v in f.items():
        say("F", k, json.dumps(v, ensure_ascii=False, default=str))

    # ---- d) which engine behaviours the three unit tests would notice ----
    # Mutated copies of study.py / switch.py are written under cache/mutants, imported under private names,
    # handed to the test module in place of `study`, then deleted. Codex's files are never touched.
    import importlib
    import importlib.util
    import io
    import shutil
    import unittest
    from types import SimpleNamespace
    tests = importlib.import_module("tests.test_desk_defense_20261119")
    assert tests.study is study
    mutant_dir = HERE / "cache" / "mutants"
    mutant_dir.mkdir(parents=True, exist_ok=True)

    def run_tests():
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
        return result.testsRun, len(result.failures) + len(result.errors)

    mutants = [
        ("study", "replay: scale ignored", "scale * (shadow * o_mark / shadow_nav)", "(shadow * o_mark / shadow_nav)"),
        ("study", "replay: shadow cash reserve dropped", "reserve = shadow_nav * (1 - w.sum())", "reserve = 0."),
        ("study", "replay: no FX conversion", "krw = usd * fx[j]", "krw = usd"),
        ("study", "replay: exposure column zeroed", "1 - cash/nav", "0."),
        ("study", "replay: shadow NAV not compounded", "shadow = np.divide(shadow_nav*w, o,", "shadow = np.divide(w, o,"),
        ("study", "replay: missing-price guard removed", "if not np.isfinite(o[active]).all() or not np.isfinite(c[active]).all():", "if False:"),
        ("study", "replay: 10bp drag doubled", "drag *= 1 - .001 * turn", "drag *= 1 - .002 * turn"),
        ("study", "replay: gate never exits", "if gate[j] else np.zeros_like(shares)", "if True else np.zeros_like(shares)"),
        ("study", "replay: target-day rebalance skipped while invested", "if (gate[j] and trade) or", "if (j == 0 and trade) or"),
        ("study", "market_gates: same-day close (look-ahead)", "state.ffill().shift().reindex(days)", "state.ffill().reindex(days)"),
        ("study", "eligibility: same-day close (look-ahead)", "columns=tickers).shift().loc[days]", "columns=tickers).loc[days]"),
        ("study", "plan: cash overlay never reacts mid-week", "if passed == last_pass:", "if True:"),
        ("study", "plan: caps not applied", "target = S.project_caps({k: v / total for k, v in raw.items()}, book.sector)", "target = {k: v / total for k, v in raw.items()}"),
        ("study", "plan: weekly rule trades every day", '(freq == "week" and started and not week)', "(False)"),
        ("switch", "switched_targets: stale published weights on a switch", "schedule[j] = values[active] / nav[active]",
         "schedule[j] = plans[active][max(k for k in plans[active] if k <= j)].copy()"),
    ]
    d_rows = []
    total, failed = run_tests()
    assert failed == 0, "tests must pass before mutation"
    try:
        for n, (module_name, label, old, new) in enumerate(mutants):
            source = (CODEX / f"{module_name}.py").read_text(encoding="utf-8")
            assert source.count(old) == 1, (label, source.count(old))
            path = mutant_dir / f"mutant_{n}_{module_name}.py"
            path.write_text(source.replace(old, new), encoding="utf-8")
            spec = importlib.util.spec_from_file_location(f"_audit_mutant_{n}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            proxy = SimpleNamespace(replay=getattr(module, "replay", study.replay), market_gates=getattr(module, "market_gates", study.market_gates),
                                    switched_targets=module.switched_targets)
            tests.study = proxy
            try:
                _, bad = run_tests()
            finally:
                tests.study = study
            d_rows.append({"mutation": label, "tests_failed": bad, "detected": bad > 0})
    finally:
        shutil.rmtree(mutant_dir, ignore_errors=True)
    assert run_tests()[1] == 0
    OUT["d_tests"] = {"tests": total, "mutants": d_rows, "undetected": [r["mutation"] for r in d_rows if not r["detected"]]}
    for r in d_rows:
        say("D", r)

    (HERE / "cache").mkdir(exist_ok=True)
    (HERE / "cache" / "audit_engine.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    say("written cache/audit_engine.json")


if __name__ == "__main__":
    main()
