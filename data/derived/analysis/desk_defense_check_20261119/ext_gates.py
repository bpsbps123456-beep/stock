"""Family "gates": market-exit signals Codex did not test, on the weekly 7-stock plan and three more plans.

Run from the repo root:  py data/derived/analysis/desk_defense_check_20261119/ext_gates.py

Every signal is a state after a completed close; the state after close j-1 decides the exposure held
from the open of US session j (study.replay / common.replay_frac). Nothing is written outside this
check folder: cache/gates_daily.parquet, out/gates_candidates.csv, cache/gates_summary.json.

Grid (gate signals x plans):
  G1 SMA band        index{QQQ,SOXX} x n{5,10,20,30,50,100} x band{0,1,2,3}% x exit conf{1,2} x re-entry conf{1,3} = 192
  G2 Donchian        index x exit N{5,10,20,40} x re-entry{high5,high10,high20,sma20}                             = 32
  G3 Drawdown        index x x{5,7.5,10,12.5,15}% off the 60-session high x re-entry{+5,+7.5,+10% off low, sma20}  = 40
  G4 MA cross        index x (fast,slow) in {5,10,20}x{20,50,100}, fast<slow                                      = 16
  G5 Composite       {AND,OR} of QQQ,SOXX >= SMA n{10,20,50,100} x re-entry conf{1,3}                             = 16
  G6 Giant breadth   threshold{10,15,19,20,25,30,35,40} x re-entry conf{1,3} (16) + hysteresis t/t+5 (8)
                     + breadth >= own SMA{10,20} x re-entry conf{1,3} (4)                                         = 28
  G7 Other series    TIME ETF 456600.KS >= SMA n{5,10,20,50} x KRX lag{prev_date,same_date} (8)
                     + median return of PIT TIME US holdings over L{10,20,40} sessions >= 0 x re-entry conf{1,3} (6) = 14
  -> 338 binary gates x 4 plans {weekly, stock100_cash, giantA_cash, bothA_cash}                                  = 1352
  G8 Partial exit    (12 plain G1 gates b=0,1/1 + 28 G6 gates) x off-state exposure{0.25,0.5}, weekly plan        = 80
  G9 Two-speed       index x pair{(20,100),(10,50)}: exposure = share of the two SMAs the index is above, 4 plans = 16
  G0 Plans alone     stock100_cash, giantA_cash, bothA_cash with no gate (Codex's, kept as references)            = 3
  base                                                                                                            = 1
  Total 1452 ledgers (1451 candidates + base). 48 of them (plain G1 n in 10/20/50/100, re-entry 3 only for
  20/50, on four plans) reproduce Codex candidates and are flagged dup_codex=1.
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
from common import S, study  # noqa: E402
from desk import giant  # noqa: E402
from watchlist import AI_SUBSECTOR  # noqa: E402

T_TEST = 150
BARS = common.CODEX / "cache" / "grade_bars"
PLANS = {"weekly": ("any", "week", "rotate"), "stock100_cash": ("sma100", "change", "cash"),
         "giantA_cash": ("giantA", "change", "cash"), "bothA_cash": ("bothA", "change", "cash")}


# ---------------------------------------------------------------- signal primitives
def sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()


def streak(cond, k):
    """cond was true on each of the last k closes."""
    return pd.Series(np.asarray(cond, dtype=float)).rolling(k).min().to_numpy() == 1


def machine(exit_, enter, first, init=True):
    """In/out state after each close. Out on an exit close, back in on a later entry close."""
    state = np.full(len(exit_), np.nan)
    s = bool(init)
    for i in range(first, len(exit_)):
        if s:
            if exit_[i]:
                s = False
        elif enter[i]:
            s = True
        state[i] = s
    return state


def band_state(close, n, band, k_exit, k_in):
    line = sma(close, n)
    first = n - 1 + max(k_exit, k_in) - 1
    return machine(streak(close < line * (1 - band), k_exit), streak(close >= line * (1 + band), k_in),
                   first, close[first] >= line[first])


def donchian_state(close, n_exit, reentry):
    s = pd.Series(close)
    exit_ = close < s.shift().rolling(n_exit).min().to_numpy()
    if reentry == "sma20":
        enter, first = close >= sma(close, 20), max(n_exit, 19)
    else:
        enter, first = close > s.shift().rolling(reentry).max().to_numpy(), max(n_exit, reentry)
    return machine(exit_, enter, first, True)


def drawdown_state(close, x, reentry, window=60):
    """Exit at close <= (1-x) * highest close of the last `window` sessions (window cut at the last re-entry
    close, otherwise the rule would exit again the day after every re-entry). Re-enter at close >= (1+y) *
    lowest close since the exit, or at close >= SMA20."""
    line = sma(close, 20)
    state = np.full(len(close), np.nan)
    s, anchor, low = True, 0, np.nan
    for i in range(window - 1, len(close)):
        if s:
            if close[i] <= (1 - x) * close[max(i - window + 1, anchor):i + 1].max():
                s, low = False, close[i]
        else:
            low = min(low, close[i])
            if (close[i] >= line[i]) if reentry == "sma20" else (close[i] >= (1 + reentry) * low):
                s, anchor = True, i
        state[i] = s
    return state


def cond_state(cond, first, k_in=1):
    """Out on the first close with cond false, back in after k_in consecutive closes with cond true."""
    cond = np.asarray(cond, dtype=bool)
    first += k_in - 1
    return machine(~cond, streak(cond, k_in), first, cond[first])


def breadth_from_bars(bars, days):
    """Giant-tab breadth (share of the AI universe within 10% of its all-time high) on US close dates."""
    series = giant.breadth_series(bars, sorted(AI_SUBSECTOR), pd.DatetimeIndex(days))
    return series.to_numpy()


def load_bars():
    return {s: pd.read_parquet(BARS / f"{s}.parquet") for s in sorted(AI_SUBSECTOR)}


# ---------------------------------------------------------------- gate registry
def build_gates(book, days, start, bars):
    """name -> (subfamily, params, frac[len(days)] in [0,1]); frac[j] uses closes through session j-1."""
    cal = book.days
    n_days = len(days)

    def shifted(state):  # state after close of book index i -> exposure at session j = start + j
        out = state[start - 1:start - 1 + n_days]
        if len(out) != n_days or not np.isfinite(out).all():
            raise ValueError("signal warm-up does not reach the first session")
        return out.astype(float)

    px = {sym: book.cls[sym].to_numpy(float) for sym in ("QQQ", "SOXX")}
    gates = {}

    def add(name, subfamily, frac, **params):
        if name in gates:
            raise ValueError(name)
        gates[name] = (subfamily, params, np.asarray(frac, dtype=float))

    plain = {}
    for sym, close in px.items():
        for n in (5, 10, 20, 30, 50, 100):
            for band in (0, 1, 2, 3):
                for k_exit in (1, 2):
                    for k_in in (1, 3):
                        frac = shifted(band_state(close, n, band / 100, k_exit, k_in))
                        name = f"G1_{sym}_n{n}_b{band}_e{k_exit}_r{k_in}"
                        add(name, "G1_sma_band", frac, index=sym, n=n, band_pct=band, exit_conf=k_exit,
                            reentry=f"conf{k_in}")
                        if band == 0 and k_exit == 1 and k_in == 1:
                            plain[name] = frac
        for n_exit in (5, 10, 20, 40):
            for reentry in (5, 10, 20, "sma20"):
                label = reentry if reentry == "sma20" else f"high{reentry}"
                add(f"G2_{sym}_low{n_exit}_{label}", "G2_donchian", shifted(donchian_state(close, n_exit, reentry)),
                    index=sym, n=n_exit, reentry=label)
        for x in (5, 7.5, 10, 12.5, 15):
            for reentry in (5, 7.5, 10, "sma20"):
                label = reentry if reentry == "sma20" else f"up{reentry:g}"
                state = drawdown_state(close, x / 100, reentry if reentry == "sma20" else reentry / 100)
                add(f"G3_{sym}_dd{x:g}_{label}", "G3_drawdown", shifted(state), index=sym, drop_pct=x, reentry=label)
        for fast in (5, 10, 20):
            for slow in (20, 50, 100):
                if fast < slow:
                    state = np.where(np.isfinite(sma(close, slow)), sma(close, fast) >= sma(close, slow), np.nan)
                    add(f"G4_{sym}_x{fast}_{slow}", "G4_ma_cross", shifted(state), index=sym, n=fast, slow=slow,
                        reentry="conf1")
    for mode in ("AND", "OR"):
        for n in (10, 20, 50, 100):
            above = [px[sym] >= sma(px[sym], n) for sym in ("QQQ", "SOXX")]
            cond = (above[0] & above[1]) if mode == "AND" else (above[0] | above[1])
            for k_in in (1, 3):
                add(f"G5_{mode}_n{n}_r{k_in}", "G5_composite", shifted(cond_state(cond, n - 1, k_in)),
                    index=f"QQQ_{mode}_SOXX", n=n, reentry=f"conf{k_in}")

    breadth = breadth_from_bars(bars, cal)
    first_b = int(np.flatnonzero(np.isfinite(breadth))[0])
    g6 = {}
    for t in (10, 15, 19, 20, 25, 30, 35, 40):
        for k_in in (1, 3):
            name = f"G6_thr{t}_r{k_in}"
            g6[name] = shifted(cond_state(breadth >= t, first_b, k_in))
            add(name, "G6_breadth_level", g6[name], index="breadth", threshold=t, reentry=f"conf{k_in}")
        name = f"G6_hyst{t}_{t + 5}"
        g6[name] = shifted(machine(breadth < t, breadth >= t + 5, first_b, breadth[first_b] >= t))
        add(name, "G6_breadth_level", g6[name], index="breadth", threshold=t, reentry=f"level{t + 5}")
    for n in (10, 20):
        for k_in in (1, 3):
            name = f"G6_sma{n}_r{k_in}"
            g6[name] = shifted(cond_state(breadth >= sma(breadth, n), first_b + n - 1, k_in))
            add(name, "G6_breadth_sma", g6[name], index="breadth", n=n, reentry=f"conf{k_in}")

    # G7a: TIME ETF on its own KRX calendar. prev_date uses KRX closes dated before the US session date;
    # same_date also uses the KRX close of that calendar date (15:30 KST, before the 22:30/23:30 KST US open).
    etf = book.etf.dropna()
    krx, eclose = etf.index, etf.to_numpy(float)
    for n in (5, 10, 20, 50):
        state = cond_state(eclose >= sma(eclose, n), n - 1)
        for lag, side in (("prev_date", "left"), ("same_date", "right")):
            pos = krx.searchsorted(pd.DatetimeIndex(days), side=side) - 1
            if (pos < 0).any() or not np.isfinite(state[pos]).all():
                raise ValueError("TIME ETF warm-up too short")
            add(f"G7_etf_n{n}_{lag}", "G7_time_etf", state[pos].astype(float), index="456600.KS", n=n, krx_lag=lag,
                reentry="conf1")
    # G7b: median trailing return of the PIT TIME US holdings (the ranked rows of the PDF dated before session k).
    # The first TIME PDF is dated 2026-01-02, so the first PIT roster exists at session 1 (2026-01-05, also the
    # base's first investment). There is no warm-up: the state starts from the first observation and the
    # re-entry confirmation applies only after an exit. Session 0 holds nothing in any plan and is set to 1.
    adj = book.adj
    rosters = {k: [row["ticker"] for row in S.momentum_ranking(book, k, cal[k])[1]] for k in range(121, len(cal))}
    first_k = min(k for k, names in rosters.items() if names)
    if first_k > start + 1:
        raise ValueError("PIT roster starts after the first invested session")
    for lookback in (10, 20, 40):
        median = np.full(len(cal), np.nan)
        for k in range(first_k, len(cal)):
            names = rosters[k]
            ret = adj.iloc[k - 1][names].to_numpy(float) / adj.iloc[k - 1 - lookback][names].to_numpy(float) - 1
            median[k] = np.nanmedian(ret) if np.isfinite(ret).any() else np.nan
        if not np.isfinite(median[first_k:]).all():
            raise ValueError("median holdings return undefined")
        for k_in in (1, 3):
            # median[k] already uses closes through k-1, so the state at k applies to session k (no extra shift).
            cond = np.where(np.isfinite(median), median >= 0, False)
            state = machine(~cond, streak(cond, k_in), first_k, cond[first_k])
            state[:first_k] = 1.
            out = state[start:start + n_days]
            if not np.isfinite(out).all():
                raise ValueError("median holdings warm-up too short")
            add(f"G7_med{lookback}_r{k_in}", "G7_holdings_median", out.astype(float), index="TIME_US_median",
                n=lookback, reentry=f"conf{k_in}")
    binary = dict(gates)

    partial = {}
    for name, frac in {**plain, **g6}.items():
        for off in (.25, .5):
            params = dict(gates[name][1], off_exposure=off, source=name)
            partial[f"G8_off{int(off * 100)}_{name}"] = ("G8_partial_" + ("sma" if name.startswith("G1") else "breadth"),
                                                         params, np.where(frac > 0, 1., off))
    two_speed = {}
    for sym, close in px.items():
        for a, b in ((20, 100), (10, 50)):
            frac = (shifted(cond_state(close >= sma(close, a), a - 1))
                    + shifted(cond_state(close >= sma(close, b), b - 1))) / 2
            two_speed[f"G9_{sym}_{a}_{b}"] = ("G9_two_speed", dict(index=sym, n=a, slow=b), frac)
    return binary, partial, two_speed, breadth


def sma_ok(book, days, tickers, n):
    close = book.cls.reindex(columns=tickers)
    return (close >= close.rolling(n).mean()).reindex(index=book.days).shift().loc[days].to_numpy(bool)


def build_plans(ctx, book, rankings, ok):
    return {name: study.plan(book, ctx.days, rankings, ctx.col, ok[cond], freq, method)
            for name, (cond, freq, method) in PLANS.items()}


# ---------------------------------------------------------------- look-ahead self-test
def lookahead_test(ctx, bars, gate_sets, plans, mode="uniform"):
    """Change every price after session T and require identical decisions at sessions <= T.

    uniform: x1.5 everywhere (the required test; it keeps cross-sectional ranks, so it cannot move the
    weekly ranking). scrambled: an independent factor in [0.5, 2.0] per security and day (fixed seed),
    which also reorders momentum ranks, per-stock SMA states, breadth and the holdings median.
    """
    cut = ctx.days[T_TEST]
    book = ctx.book
    rng = np.random.default_rng(20261119)

    def factor(shape):
        return 1.5 if mode == "uniform" else rng.uniform(.5, 2., shape)

    late = book.cls.index > cut
    f_close = factor((int(late.sum()), book.cls.shape[1]))

    def bump(frame, f):
        out = frame.copy()
        out.loc[late] = out.loc[late].to_numpy() * f
        return out

    etf = book.etf.copy()
    late_etf = etf.index > pd.Timestamp(cut)
    etf[late_etf] = etf[late_etf].to_numpy() * factor(int(late_etf.sum()))
    book2 = dataclasses.replace(book, cls=bump(book.cls, f_close), adj=bump(book.adj, f_close),
                                opn=bump(book.opn, factor(f_close.shape) if mode != "uniform" else 1.5), etf=etf)
    bars2 = {}
    for sym, frame in bars.items():
        frame = frame.copy()
        late_bar = frame.index > pd.Timestamp(cut)
        frame.loc[late_bar] = frame.loc[late_bar].to_numpy() * factor((int(late_bar.sum()), 1))
        bars2[sym] = frame
    binary2, partial2, two2, breadth2 = build_gates(book2, ctx.days, ctx.start, bars2)
    changed_after, strict_ok = 0, 0
    for original, perturbed in zip(gate_sets, (binary2, partial2, two2)):
        assert list(original) == list(perturbed)
        for name in original:
            a, b = original[name][2], perturbed[name][2]
            if not np.array_equal(a[:T_TEST + 1], b[:T_TEST + 1]):
                raise AssertionError(f"look-ahead in gate {name}")
            # The decision at T+1 also uses only closes through T, except same_date KRX which may use the
            # (perturbed) KRX close dated T+1.
            strict_ok += "same_date" in name or a[T_TEST + 1] == b[T_TEST + 1]
            changed_after += not np.array_equal(a[T_TEST + 2:], b[T_TEST + 2:])
    n_gates = sum(len(g) for g in gate_sets)
    if strict_ok != n_gates:
        raise AssertionError("a gate decision at T+1 moved with prices after T")
    rankings2 = [S.momentum_ranking(book2, ctx.start + j, day)[1] for j, day in enumerate(ctx.days)]
    ok2 = dict(ctx.ok, sma100=sma_ok(book2, ctx.days, ctx.tickers, 100))
    plans2 = build_plans(ctx, book2, rankings2, ok2)
    plan_changed = {}
    for name, targets in plans.items():
        early = {j: w for j, w in targets.items() if j <= T_TEST + 1}
        early2 = {j: w for j, w in plans2[name].items() if j <= T_TEST + 1}
        if set(early) != set(early2) or any(not np.array_equal(early[j], early2[j]) for j in early):
            raise AssertionError(f"look-ahead in plan {name}")
        late = {j for j in set(targets) | set(plans2[name]) if j > T_TEST + 1}
        plan_changed[name] = sum(1 for j in late if j not in targets or j not in plans2[name]
                                 or not np.array_equal(targets[j], plans2[name][j]))
    return dict(mode=mode, T=T_TEST, cut_date=cut, gates_tested=n_gates, gates_identical_through_T=n_gates,
                gates_identical_at_T_plus_1=int(strict_ok), gates_changed_after_T_plus_1=int(changed_after),
                plans_identical_through_T_plus_1=list(plans), plan_targets_changed_after=plan_changed,
                breadth_identical_through_T=bool(np.array_equal(
                    breadth_from_bars(bars, ctx.book.days)[:ctx.start + T_TEST + 1],
                    breadth2[:ctx.start + T_TEST + 1])))


# ---------------------------------------------------------------- build
def exits(frac, days, window):
    a, b = common.WINDOWS[window]
    return int(sum(1 for j in range(1, len(days)) if a <= days[j] <= b and frac[j] < frac[j - 1] - 1e-12))


def build():
    ctx = common.context()
    args = (ctx.days, ctx.opn, ctx.close, ctx.fx)
    base = study.replay(*args, ctx.weekly, ctx.gates["none"])
    s = common.window_stats(base, *common.WINDOWS["h1"]), common.window_stats(base, *common.WINDOWS["h2"])
    assert abs(s[0]["krw"] - 2.4505) < 5e-5 and abs(s[1]["krw"] + 0.3945) < 5e-5, (s[0]["krw"], s[1]["krw"])
    np.testing.assert_allclose(base.usd, ctx.baseline.close, rtol=1e-10)

    bars = load_bars()
    assert not set(AI_SUBSECTOR) - set(bars)
    binary, partial, two_speed, breadth = build_gates(ctx.book, ctx.days, ctx.start, bars)
    # Independent checks against Codex: his 12 SMA gates, his breadth file, his sma100 eligibility.
    for sym in ("QQQ", "SOXX"):
        for n in (10, 20, 50, 100):
            for k_in in ((1, 3) if n in (20, 50) else (1,)):
                mine = binary[f"G1_{sym}_n{n}_b0_e1_r{k_in}"][2].astype(bool)
                assert np.array_equal(mine, ctx.gates[f"{sym}{n}r{k_in}"]), (sym, n, k_in)
    saved = pd.read_csv(common.CODEX / "breadth.csv", index_col=0).breadth
    np.testing.assert_allclose(pd.Series(breadth, index=ctx.book.days).loc[saved.index[1:]], saved.iloc[1:], rtol=1e-9)
    assert np.array_equal(sma_ok(ctx.book, ctx.days, ctx.tickers, 100), ctx.ok["sma100"])
    plans = build_plans(ctx, ctx.book, ctx.rankings, ctx.ok)
    assert set(plans["weekly"]) == set(ctx.weekly) and all(
        np.array_equal(plans["weekly"][j], ctx.weekly[j]) for j in ctx.weekly)

    selftest = [lookahead_test(ctx, bars, (binary, partial, two_speed), plans, mode)
                for mode in ("uniform", "scrambled")]

    curves, configs = {"base": base}, {"base": dict(subfamily="base", plan="weekly", signal="none")}

    def put(key, plan, name, subfamily, params, frac):
        is_binary = bool(np.isin(frac, (0., 1.)).all())
        curve = (study.replay(*args, plans[plan], frac.astype(bool)) if is_binary
                 else common.replay_frac(*args, plans[plan], frac))
        dup = int(subfamily == "G1_sma_band" and params["band_pct"] == 0 and params["exit_conf"] == 1 and (
            (params["n"] in (10, 100) and params["reentry"] == "conf1") or params["n"] in (20, 50)))
        configs[key] = dict(subfamily=subfamily, plan=plan, signal=name, **params, dup_codex=dup,
                            exits_full=exits(frac, ctx.days, "full"), exits_h1=exits(frac, ctx.days, "h1"),
                            exits_jul_aug=exits(frac, ctx.days, "jul_aug"), exits_sep_nov=exits(frac, ctx.days, "sep_nov"),
                            off_sessions_h2=int(sum(1 for j, d in enumerate(ctx.days) if d >= "2026-07-01" and frac[j] < 1)))
        curves[key] = curve

    for plan in PLANS:
        if plan != "weekly":
            put(f"{plan}__none", plan, "none", "G0_plan_only", {}, np.ones(len(ctx.days)))
        for name, (subfamily, params, frac) in binary.items():
            put(f"{plan}__{name}", plan, name, subfamily, params, frac)
        for name, (subfamily, params, frac) in two_speed.items():
            put(f"{plan}__{name}", plan, name, subfamily, params, frac)
    for name, (subfamily, params, frac) in partial.items():
        put(f"weekly__{name}", "weekly", name, subfamily, params, frac)

    # Cross-checks against numbers Codex published (report.md).
    st = common.stats_table({k: curves[k] for k in ("stock100_cash__none", "giantA_cash__none",
                                                    "bothA_cash__G1_SOXX_n10_b0_e1_r1", "weekly__G6_thr19_r1",
                                                    "stock100_cash__G1_SOXX_n100_b0_e1_r1")})
    for key, expect in (("stock100_cash__none", -.3990), ("giantA_cash__none", -.3020),
                        ("bothA_cash__G1_SOXX_n10_b0_e1_r1", -.1510), ("weekly__G6_thr19_r1", -.1631),
                        ("stock100_cash__G1_SOXX_n100_b0_e1_r1", -.3393)):
        assert abs(st.at[key, "h2_krw"] - expect) < 5e-5, (key, st.at[key, "h2_krw"])
    counts = pd.Series([c["subfamily"] for c in configs.values()]).value_counts().to_dict()
    assert len(curves) == 1452, len(curves)
    common.save("gates", curves, configs)
    return ctx, curves, configs, selftest, counts


# ---------------------------------------------------------------- report
def report(ctx, curves, configs, selftest, counts, elapsed):
    st = common.stats_table(curves)
    cfg = pd.DataFrame.from_dict(configs, orient="index")
    base = st.loc["base"]
    out = {"n_ledgers": len(curves), "subfamily_counts": counts, "lookahead_selftest": selftest}
    out["base"] = {k: round(float(base[k]), 6) for k in st.columns}

    def row(key, train=None, test=None):
        r = st.loc[key]
        d = {"strategy": key}
        for w in ([train, test] if train else ["h1", "h2", "jul_aug", "sep_nov", "oct_nov"]):
            for m in ("krw", "krw_mdd", "usd", "usd_mdd", "expo", "turn"):
                d[f"{w}_{m}"] = round(float(r[f"{w}_{m}"]), 6)
        if test:
            ctl = common.matched_control(curves["base"], r[f"{test}_expo"], test)
            d[f"{test}_control_usd"], d[f"{test}_control_usd_mdd"] = (round(float(v), 6) for v in ctl)
        return d

    protocols = (("P1", "h1", "h2"), ("P2", "jan_aug", "sep_nov"))
    sel = []
    for scope, keys in (("all", st.index), ("weekly_plan_only", [k for k in st.index if cfg.at[k, "plan"] == "weekly"])):
        for name, train, test in protocols:
            for rule, key in common.select(st.loc[keys], train).items():
                sel.append(dict(scope=scope, protocol=f"{name} {train}->{test}", rule=rule, **row(key, train, test)))
    out["selections"] = sel
    cand = st.drop(index="base")
    out["hindsight"] = {"best_h2_usd": row(cand.h2_usd.idxmax(), "h1", "h2"),
                        "best_sep_nov_usd": row(cand.sep_nov_usd.idxmax(), "jan_aug", "sep_nov"),
                        "best_h2_usd_weekly_plan": row(cand.h2_usd[cfg.loc[cand.index, "plan"] == "weekly"].idxmax(), "h1", "h2")}

    control = {w: pd.DataFrame([common.matched_control(curves["base"], cand.at[k, f"{w}_expo"], w) for k in cand.index],
                               index=cand.index, columns=["usd", "mdd"]) for w in ("h2", "jul_aug", "sep_nov")}
    excess = {w: cand[f"{w}_usd"] - control[w].usd for w in control}
    beat = {w: (cand[f"{w}_usd"] > base[f"{w}_usd"]) & (cand[f"{w}_usd_mdd"] > base[f"{w}_usd_mdd"]) for w in control}
    out["counts"] = {"candidates": len(cand)}
    for w in control:
        both = beat[w] & (excess[w] > 0)
        out["counts"][w] = {"beat_base_usd_return_and_mdd": int(beat[w].sum()),
                            "also_beat_matched_control_return": int(both.sum()),
                            "also_beat_control_return_and_mdd": int((both & (cand[f"{w}_usd_mdd"] > control[w].mdd)).sum()),
                            "control_undefined": int(control[w].usd.isna().sum())}
    out["counts"]["beat_control_h2_AND_jul_aug_AND_sep_nov"] = int(
        ((excess["h2"] > 0) & (excess["jul_aug"] > 0) & (excess["sep_nov"] > 0)).sum())
    out["counts"]["jul_aug_usd_worse_than_base"] = int((cand.jul_aug_usd < base.jul_aug_usd).sum())
    out["counts"]["sep_nov_usd_better_than_base"] = int((cand.sep_nov_usd > base.sep_nov_usd).sum())
    out["counts"]["beat_base_h2_krw_return_and_mdd"] = int(
        ((cand.h2_krw > base.h2_krw) & (cand.h2_krw_mdd > base.h2_krw_mdd)).sum())
    weekly = cfg.loc[cand.index, "plan"] == "weekly"
    out["counts"]["weekly_plan"] = {"candidates": int(weekly.sum()), "beat_base_h2": int((beat["h2"] & weekly).sum()),
                                    "also_beat_control_h2": int((beat["h2"] & weekly & (excess["h2"] > 0)).sum())}
    out["rank_corr"] = {
        "h1_krw_vs_h2_krw": round(float(cand.h1_krw.corr(cand.h2_krw, method="spearman")), 4),
        "jan_aug_krw_vs_sep_nov_krw": round(float(cand.jan_aug_krw.corr(cand.sep_nov_krw, method="spearman")), 4),
        "jul_aug_usd_vs_sep_nov_usd": round(float(cand.jul_aug_usd.corr(cand.sep_nov_usd, method="spearman")), 4),
        "jul_aug_excess_vs_sep_nov_excess": round(float(excess["jul_aug"].corr(excess["sep_nov"], method="spearman")), 4),
        "h2_expo_vs_h2_usd": round(float(cand.h2_expo.corr(cand.h2_usd, method="spearman")), 4)}

    frame = cand.join(cfg[["subfamily", "plan", "exits_full", "exits_h1", "exits_jul_aug", "exits_sep_nov"]])
    frame["h2_excess"], frame["jul_aug_excess"], frame["sep_nov_excess"] = excess["h2"], excess["jul_aug"], excess["sep_nov"]

    def table(groups):
        rows = []
        for label, g in frame.groupby(groups, sort=True):
            label = label if isinstance(label, tuple) else (label,)
            rows.append(dict(zip(groups, label), n=len(g),
                             h1_krw_med=g.h1_krw.median(), h2_usd_med=g.h2_usd.median(), h2_usd_best=g.h2_usd.max(),
                             h2_usd_mdd_med=g.h2_usd_mdd.median(), jul_aug_usd_med=g.jul_aug_usd.median(),
                             sep_nov_usd_med=g.sep_nov_usd.median(), h2_expo_med=g.h2_expo.median(),
                             h2_excess_med=g.h2_excess.median(), h2_excess_pos=int((g.h2_excess > 0).sum()),
                             jul_aug_excess_med=g.jul_aug_excess.median(), sep_nov_excess_med=g.sep_nov_excess.median(),
                             exits_full_med=g.exits_full.median(), exits_h1_med=g.exits_h1.median(),
                             exits_jul_aug_med=g.exits_jul_aug.median(), exits_sep_nov_med=g.exits_sep_nov.median()))
        return pd.DataFrame(rows).round(4)

    by_sub, by_sub_plan = table(["subfamily"]), table(["subfamily", "plan"])
    by_plan = table(["plan"])
    out["by_subfamily"] = by_sub.to_dict("records")
    out["by_plan"] = by_plan.to_dict("records")
    out["by_subfamily_plan"] = by_sub_plan.to_dict("records")
    # Weekly-plan slices of the main knobs, to see which parameter carries the effect.
    wk = frame[frame.plan == "weekly"].join(cfg[["index", "n", "band_pct", "exit_conf", "reentry", "threshold"]])
    knobs = {}
    for sub, cols in (("G1_sma_band", ["index", "n"]), ("G1_sma_band", ["band_pct"]), ("G1_sma_band", ["exit_conf", "reentry"]),
                      ("G6_breadth_level", ["threshold"]), ("G2_donchian", ["index", "n"]), ("G3_drawdown", ["index"])):
        g = wk[wk.subfamily == sub].groupby(cols)
        knobs[f"{sub}:{'+'.join(cols)}"] = g.agg(
            count=("h2_usd", "size"), h1_krw_med=("h1_krw", "median"), jul_aug_usd_med=("jul_aug_usd", "median"),
            sep_nov_usd_med=("sep_nov_usd", "median"), h2_usd_med=("h2_usd", "median"), h2_expo_med=("h2_expo", "median"),
            h2_excess_med=("h2_excess", "median"), exits_jul_aug_med=("exits_jul_aug", "median"),
            exits_sep_nov_med=("exits_sep_nov", "median")).round(4).reset_index().to_dict("records")
    out["weekly_plan_knobs"] = knobs
    top = frame.sort_values("h2_excess", ascending=False).head(15)
    out["top15_h2_excess_over_control"] = [
        dict(strategy=k, h1_krw=round(float(r.h1_krw), 4), h2_usd=round(float(r.h2_usd), 4),
             h2_usd_mdd=round(float(r.h2_usd_mdd), 4), h2_expo=round(float(r.h2_expo), 4),
             h2_excess=round(float(r.h2_excess), 4), jul_aug_excess=round(float(r.jul_aug_excess), 4),
             sep_nov_excess=round(float(r.sep_nov_excess), 4)) for k, r in top.iterrows()]
    out["runtime_seconds"] = round(elapsed, 1)
    common.CACHE.mkdir(exist_ok=True)
    (common.CACHE / "gates_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float),
                                                     encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    print(json.dumps({k: out[k] for k in ("n_ledgers", "subfamily_counts", "lookahead_selftest", "counts", "rank_corr",
                                          "runtime_seconds")}, indent=1, default=float))
    print(pd.DataFrame(sel).T.to_string())
    print(pd.DataFrame(out["hindsight"]).to_string())
    print(by_sub.T.to_string())
    print(by_plan.T.to_string())
    print(by_sub_plan.to_string())
    return out


if __name__ == "__main__":
    t0 = time.time()
    built = build()
    report(*built, time.time() - t0)
