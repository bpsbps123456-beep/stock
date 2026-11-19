"""Family "filters": stock-level entry/hold/skip filters on the weekly 7-stock rule.

Run from the repo root:  py data/derived/analysis/desk_defense_check_20261119/ext_filters.py

Every filter is a point-in-time eligibility matrix [session, ticker]: the row used at the open of
US session j is built from closes (and opens) through session j-1, the TIME PDF roster that
desk.strategies.momentum_ranking exposes for session j, and Codex's frozen grade tables shifted one
session. Earnings windows are calendar facts (see earnings_table). Each matrix goes to study.plan in
three modes:
  wk   = ("week", "rotate")    weekly rebalance, failing names replaced by the next-ranked passing one
  cash = ("change", "cash")    base weekly basket, a failing held name sits in USD cash until it passes
  rot  = ("change", "rotate")  checked daily, rebalanced when the selected set changes

Grid (52 single filters):
  F1 extension (9)   close/SMA50-1 <= {20,30,40,50}% ; RSI14 < {75,80,85} ; 10-session return <= {30,50}%
  F2 structure (10)  close >= SMA20 and SMA50 ; SMA20 > SMA50 ; close >= {10,20}-session closing low x (1+{5,10}%) ;
                     10-session closing low above the prior 10-session low ; close within {10,15,20}% of 20-session closing high
  F3 rel. strength (8)  {20,60}-session return > SOXX's, > QQQ's, >= median, >= 67th pct of the PIT TIME US roster
  F4 vol / gap (4)   20-session realised vol <= roster {median, 75th pct} ; no open gap worse than -{7,10}% in the last 5 sessions
  F5 tabs (13)       trend grade >= {B,A} (plain / and not below the grade 5 sessions earlier) ; Giant recovery grade >= {B,A}
                     (missing grade fails / passes) ; close within {10,20,30}% of the all-time high ;
                     share of the name's AI sub-sector within 10% of its high >= {20,40}% (non-AI names pass)
  F6 earnings (8)    windows (before, after) in {(2,1),(5,1),(2,3),(5,3)} sessions around the reaction session:
                     skip (not eligible inside) and only (eligible only inside)
  singles            52 filters x 3 modes x gates {none, SOXX10r1, SOXX20r3}                              = 468
  F7 concentration   max {1,2} names per AI sub-sector x {wk, rot} (4) ; semis cap {25,35}% weekly (2) ;
                     semis cap {25,35}% x the two best training singles, per training list (<= 8)
  F8 combinations    per training list (h1, jan_aug) and mode: up to 6 qualifying filters (see qualifiers()),
                     all pairs and triples (<= 35) x 3 gates                                             <= 630
  refs               base, weekly_SOXX10r1, weekly_SOXX20r3 (Codex duplicates)
Candidates built from a training list carry avail_h1 / avail_ja flags; P1 selects only among avail_h1,
P2 only among avail_ja, so no combination chosen with Jul-Aug data is scored on Jul-Aug.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import S, study  # noqa: E402
import signals  # noqa: E402  Codex frozen grade tables (read only)
import watchlist  # noqa: E402
from desk import config  # noqa: E402

FAMILY = "filters"
T_TEST = 150
BARS = common.CODEX / "cache" / "grade_bars"
EARN_FROZEN = common.CACHE / "filters_earnings_dates.csv"
MODES = {"wk": ("week", "rotate"), "cash": ("change", "cash"), "rot": ("change", "rotate")}
GATES = ("none", "SOXX10r1", "SOXX20r3")
TRAINS = {"h1": "avail_h1", "jan_aug": "avail_ja"}
K_COMBO, MAX_PER_SUB = 6, 2
NEAR = -0.10  # Giant tab: within 10% of the all-time high


# ---------------------------------------------------------------- inputs
def load_bars(symbols):
    return {s: pd.read_parquet(BARS / f"{s}.parquet") for s in sorted(symbols) if (BARS / f"{s}.parquet").exists()}


def asof(series, us_dates):
    """Latest local value on or before each US close date (Asian closes precede the US close)."""
    pos = series.index.searchsorted(us_dates, side="right") - 1
    out = np.full(len(us_dates), np.nan)
    out[pos >= 0] = series.to_numpy()[pos[pos >= 0]]
    return out


def earnings_table(tickers):
    """Reaction sessions (first session trading on the report). Frozen on first build.

    Sources, in order: earnings_trend/results/events_wide.csv (SEC 8-K acceptance or yfinance timestamp,
    built 2026-10-29) and desk erules.json events (SEC / Nasdaq / yfinance, from 2026-09-24). Dates in
    trend_earnings.json (yfinance earnings dates, no time of day) are used only when neither source has a
    reaction within 4 calendar days; those rows carry source "yf_date" and both that session and the next
    are treated as reaction sessions.
    """
    if EARN_FROZEN.exists():
        return pd.read_csv(EARN_FROZEN)
    wide = pd.read_csv(common.ROOT / "data/derived/analysis/earnings_trend/results/events_wide.csv",
                       usecols=["ticker", "reaction_date", "time_src"], low_memory=False)
    wide = wide[wide.ticker.isin(tickers)].assign(source=lambda d: "wide_" + d.time_src)
    ev = json.loads((common.ROOT / "data/derived/desk/erules.json").read_text(encoding="utf-8"))["events"]
    er = pd.DataFrame([(v["ticker"], v["reaction"], "erules_" + str(v["timing"]["src"])) for v in ev.values()
                       if v["ticker"] in tickers and v.get("reaction")], columns=["ticker", "reaction_date", "source"])
    both = pd.concat([wide[["ticker", "reaction_date", "source"]], er]).drop_duplicates(["ticker", "reaction_date"])
    te = json.loads((common.ROOT / "data/derived/desk/trend_earnings.json").read_text(encoding="utf-8"))["dates"]
    extra = []
    for t in tickers:
        known = pd.to_datetime(both.loc[both.ticker == t, "reaction_date"])
        for d in te.get(t, []):
            if not len(known) or (known - pd.Timestamp(d)).abs().min().days > 4:
                extra.append((t, d, "yf_date"))
    table = pd.concat([both, pd.DataFrame(extra, columns=["ticker", "reaction_date", "source"])])
    table = table[(table.reaction_date >= "2025-12-01") & (table.reaction_date <= "2026-11-30")]
    table = table.sort_values(["ticker", "reaction_date"]).reset_index(drop=True)
    common.CACHE.mkdir(exist_ok=True)
    table.to_csv(EARN_FROZEN, index=False)
    return table


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def roster(days, rankings, col, top=None):
    m = np.zeros((len(days), len(col)), dtype=bool)
    for j, ranked in enumerate(rankings):
        for r in (ranked if top is None else ranked[:top]):
            m[j, col[r["key"]]] = True
    return m


# ---------------------------------------------------------------- filters
def build_filters(book, days, tickers, rankings, col, bars, earn, grades):
    """name -> (subfamily, params, bool[session, ticker]); row j is known before the open of session j."""
    start = book.days.index(days[0])
    n = len(days)
    assert list(book.cls.index) == book.days
    C, A, O = (f[tickers] for f in (book.cls, book.adj, book.opn))

    def prev(frame):  # value known after the close of session j-1
        return frame.shift().iloc[start:start + n].to_numpy()

    def need(cond):  # requirement: missing data fails
        return np.nan_to_num(prev(cond.astype(float).where(cond.notna())), nan=0.) > .5

    out = {}

    def put(name, sub, elig, **params):
        assert elig.shape == (n, len(tickers)) and elig.dtype == bool
        out[name] = (sub, params, elig)

    sma20, sma50 = C.rolling(20).mean(), C.rolling(50).mean()
    # F1 extension: exclusion filters, missing data passes
    ext, r14, ret10 = prev(C / sma50 - 1), prev(rsi(C)), prev(A / A.shift(10) - 1)
    for x in (20, 30, 40, 50):
        put(f"ext50_{x}", "F1", ~(ext > x / 100), kind="ext_sma50", x=x)
    for x in (75, 80, 85):
        put(f"rsi_{x}", "F1", ~(r14 >= x), kind="rsi14", x=x)
    for x in (30, 50):
        put(f"run10_{x}", "F1", ~(ret10 > x / 100), kind="ret10", x=x)

    # F2 trend structure: requirements, missing data fails
    def req(frame_bool, valid):
        return need(frame_bool.where(valid))

    put("above20_50", "F2", req((C >= sma20) & (C >= sma50), sma50.notna() & C.notna()), kind="close>=sma20&50")
    put("sma20gt50", "F2", req(sma20 > sma50, sma50.notna()), kind="sma20>sma50")
    for w in (10, 20):
        low = C.rolling(w).min()
        for x in (5, 10):
            put(f"low{w}_p{x}", "F2", req(C >= low * (1 + x / 100), low.notna()), kind="above_low", w=w, x=x)
    low10, prior10 = C.rolling(10).min(), C.shift(10).rolling(10).min()
    put("higher_low10", "F2", req(low10 > prior10, prior10.notna()), kind="higher_low", w=10)
    high20 = C.rolling(20).max()
    for x in (10, 15, 20):
        put(f"hi20_{x}", "F2", req(C / high20 - 1 >= -x / 100, high20.notna()), kind="near_high20", x=x)

    # F3 relative strength (adjusted closes); roster = PIT TIME US holdings ranked for session j
    member = roster(days, rankings, col)
    for L in (20, 60):
        ret = A / A.shift(L) - 1
        r = prev(ret)
        for bench in ("SOXX", "QQQ"):
            b = (book.adj[bench] / book.adj[bench].shift(L) - 1).shift().iloc[start:start + n].to_numpy()
            put(f"rs{L}_{bench.lower()}", "F3", np.nan_to_num(r, nan=-np.inf) > b[:, None], kind="ret>bench", L=L, bench=bench)
        for label, q in (("half", 50), ("third", 200 / 3)):
            thr = np.array([np.nanpercentile(r[j, member[j]], q) if member[j].any() and np.isfinite(r[j, member[j]]).any()
                            else np.inf for j in range(n)])
            put(f"rs{L}_{label}", "F3", np.nan_to_num(r, nan=-np.inf) >= thr[:, None], kind="ret>=roster_pct", L=L, q=round(q, 1))

    # F4 volatility and gaps
    vol = prev(A.pct_change().rolling(20).std())
    for label, q in (("med", 50), ("p75", 75)):
        thr = np.array([np.nanpercentile(vol[j, member[j]], q) if member[j].any() and np.isfinite(vol[j, member[j]]).any()
                        else -np.inf for j in range(n)])
        put(f"vol20_{label}", "F4", np.nan_to_num(vol, nan=np.inf) <= thr[:, None], kind="vol20<=roster_pct", q=q)
    gap5 = prev((O / C.shift() - 1).rolling(5, min_periods=1).min())
    for x in (7, 10):
        put(f"gap5_{x}", "F4", ~(gap5 < -x / 100), kind="no_down_gap", x=x)

    # F5 tab filters. Grades: Codex's frozen replay, indexed by close date, shifted one session like study.eligibility.
    trend, giant = (g.reindex(index=book.days, columns=tickers) for g in grades)
    tr, gi = prev(trend), prev(giant)
    not_falling = prev((trend >= trend.shift(5)).astype(float).where(trend.notna() & trend.shift(5).notna()))
    for label, g in (("B", 2), ("A", 3)):
        put(f"trend{label}", "F5", np.nan_to_num(tr, nan=-1) >= g, kind="trend_grade", g=g)
        put(f"trend{label}_nf5", "F5", (np.nan_to_num(tr, nan=-1) >= g) & (np.nan_to_num(not_falling, nan=0.) > .5),
            kind="trend_grade_not_falling5", g=g)
        put(f"giant{label}", "F5", np.nan_to_num(gi, nan=-1) >= g, kind="giant_grade", g=g)
        put(f"giant{label}_nanok", "F5", np.nan_to_num(gi, nan=9) >= g, kind="giant_grade_missing_passes", g=g)
    us = pd.DatetimeIndex(book.days)
    dist = pd.DataFrame({t: asof(bars[t].Close / bars[t].High.cummax() - 1, us) for t in tickers}, index=book.days)
    d = prev(dist)
    for x in (10, 20, 30):
        put(f"ath_{x}", "F5", np.nan_to_num(d, nan=-np.inf) >= -x / 100, kind="ath_distance", x=x)
    sub_of = {t: watchlist.resolve_subsector(t) for t in tickers}
    members = {}
    for sym, g in watchlist.AI_SUBSECTOR.items():
        members.setdefault(g, []).append(sym)
    breadth = {}
    for g in {v for v in sub_of.values() if v}:
        near = []
        for sym in members[g]:
            b = bars[sym]
            flag = (b.Close / b.High.cummax() - 1 >= NEAR).astype(float)
            flag.iloc[:59] = np.nan  # Giant tab needs 60 bars
            near.append(asof(flag, us))
        breadth[g] = np.nanmean(np.array(near), axis=0) if near else np.full(len(us), np.nan)
    sb = prev(pd.DataFrame({t: breadth[sub_of[t]] if sub_of[t] else np.nan for t in tickers}, index=book.days))
    for x in (20, 40):
        put(f"subbreadth_{x}", "F5", ~(sb < x / 100), kind="subsector_breadth", x=x)

    # F6 earnings windows around the reaction session (calendar facts, not shifted)
    pos = {day: i for i, day in enumerate(book.days)}
    bdays = np.array(book.days)
    react = np.zeros((len(book.days), len(tickers)), dtype=bool)
    tcol = {t: i for i, t in enumerate(tickers)}
    for t, day, source in earn.itertuples(index=False):
        i = int(np.searchsorted(bdays, day))
        if i < len(bdays):
            react[i, tcol[t]] = True
            if source == "yf_date" and i + 1 < len(bdays):
                react[i + 1, tcol[t]] = True
    for before, after in ((2, 1), (5, 1), (2, 3), (5, 3)):
        win = np.zeros_like(react)
        for i, c in zip(*np.nonzero(react)):
            win[max(0, i - before):i + after + 1, c] = True
        win = win[start:start + n]
        put(f"earn_skip_{before}_{after}", "F6", ~win, kind="earnings_skip", before=before, after=after)
        put(f"earn_only_{before}_{after}", "F6", win.copy(), kind="earnings_only", before=before, after=after)
    return out, member


# ---------------------------------------------------------------- plans
@contextmanager
def rule(semis_cap=None, sub_max=None, sub_of=None):
    """Temporarily change the in-memory desk rule (no file is touched)."""
    old_cap, old_select = config.RULE["semis_cap"], S.select
    try:
        if semis_cap is not None:
            config.RULE["semis_cap"] = semis_cap
        if sub_max is not None:
            def select(ranked, current):
                n, buf = config.RULE["n"], config.RULE["buffer"]
                order = [r["key"] for r in ranked]
                keep, count = [], {}

                def take(k):
                    if count.get(sub_of[k], 0) < sub_max:
                        keep.append(k)
                        count[sub_of[k]] = count.get(sub_of[k], 0) + 1
                for k in order[:n + buf]:
                    if k in current and len(keep) < n:
                        take(k)
                for k in order:
                    if len(keep) >= n:
                        break
                    if k not in keep:
                        take(k)
                return keep
            S.select = select
        yield
    finally:
        config.RULE["semis_cap"], S.select = old_cap, old_select


def same_targets(a, b, upto):
    ka, kb = [j for j in a if j <= upto], [j for j in b if j <= upto]
    return ka == kb and all(np.array_equal(a[j], b[j]) for j in ka)


def win(curve, window, column="krw"):
    s = common.window_stats(curve, *common.WINDOWS[window])
    return s[column], s[f"{column}_mdd"]


def qualifiers(single_curves, base_curve, filters, train, mode):
    """Filters allowed into combinations, decided on the training window only.

    A filter qualifies in a mode when its ungated single is no worse than base on training KRW return and
    drawdown (D1), or keeps >= 75% of base's training return with a return/|drawdown| ratio no lower than
    base's (D2). Ranked by training return (ties: ratio, name); at most MAX_PER_SUB per subfamily, K_COMBO total.
    """
    br, bm = win(base_curve, train)
    bcal = br / abs(bm) if bm < 0 else np.inf
    rows = []
    for name in filters:
        r, m = win(single_curves[name, mode], train)
        cal = r / abs(m) if m < 0 else np.inf
        d1 = r >= br - 1e-12 and m >= bm - 1e-12
        d2 = r >= .75 * br and cal >= bcal - 1e-12
        if d1 or d2:
            rows.append((-r, -cal, name))
    picked, per_sub = [], {}
    for _, _, name in sorted(rows):
        sub = filters[name][0]
        if per_sub.get(sub, 0) < MAX_PER_SUB and len(picked) < K_COMBO:
            picked.append(name)
            per_sub[sub] = per_sub.get(sub, 0) + 1
    return picked, len(rows)


def best_singles(single_curves, base_curve, filters, train):
    """Two best distinct filters on the training window (Codex's return rule), inactive ones excluded."""
    br, bm = win(base_curve, train)
    ix = np.flatnonzero((base_curve.index >= common.WINDOWS[train][0]) & (base_curve.index <= common.WINDOWS[train][1]))
    rows = []
    for (name, mode), curve in single_curves.items():
        r, m = win(curve, train)
        active = not np.allclose(curve.usd.to_numpy()[ix], base_curve.usd.to_numpy()[ix], rtol=1e-12)
        if active and m >= bm - 1e-7:
            rows.append((-r, name, mode))
    out = []
    for _, name, mode in sorted(rows):
        if name not in [o[0] for o in out]:
            out.append((name, mode))
        if len(out) == 2:
            break
    return out


# ---------------------------------------------------------------- information content
def info_content(ctx, filters, member, top12):
    """Mean forward open-to-open return of passing vs failing names, by day then averaged."""
    opn = ctx.opn
    n = len(ctx.days)
    days = np.array(ctx.days)
    rows = []
    for h in (5, 20):
        fwd = np.full(opn.shape, np.nan)
        fwd[:n - h] = opn[h:] / opn[:n - h] - 1
        for wname in ("h1", "jul_aug", "sep_nov"):
            a, b = common.WINDOWS[wname]
            js = np.flatnonzero((days >= a) & (days <= b))
            for uname, uni in (("roster", member), ("top12", top12)):
                for name, (sub, _, elig) in filters.items():
                    spread, mp, mf, share = [], [], [], []
                    for j in js:
                        ok = uni[j] & np.isfinite(fwd[j])
                        p, f = ok & elig[j], ok & ~elig[j]
                        if ok.any():
                            share.append(p.sum() / ok.sum())
                        if p.any():
                            mp.append(fwd[j, p].mean())
                        if f.any():
                            mf.append(fwd[j, f].mean())
                        if p.any() and f.any():
                            spread.append(fwd[j, p].mean() - fwd[j, f].mean())
                    x = np.array(spread)
                    t = np.nan
                    if len(x) > h + 2:
                        e = x - x.mean()
                        var = (e @ e) / len(x)
                        for lag in range(1, h):  # Newey-West, overlap h-1
                            var += 2 * (1 - lag / h) * (e[lag:] @ e[:-lag]) / len(x)
                        t = x.mean() / np.sqrt(var / len(x)) if var > 0 else np.nan
                    rows.append(dict(filter=name, subfamily=sub, horizon=h, window=wname, universe=uname,
                                     pass_share=np.mean(share) if share else np.nan,
                                     mean_pass=np.mean(mp) if mp else np.nan, mean_fail=np.mean(mf) if mf else np.nan,
                                     spread=x.mean() if len(x) else np.nan, nw_t=t, days=len(x)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    ctx = common.context()
    book, days, tickers, col = ctx.book, ctx.days, ctx.tickers, ctx.col
    args = (days, ctx.opn, ctx.close, ctx.fx)
    bars = load_bars(set(tickers) | set(watchlist.AI_SUBSECTOR))
    assert not [t for t in tickers if t not in bars] and not [s for s in watchlist.AI_SUBSECTOR if s not in bars]
    earn = earnings_table(tickers)
    grades = signals.build_signals(book)  # frozen CSVs, no rebuild
    filters, member = build_filters(book, days, tickers, ctx.rankings, col, bars, earn, grades)
    assert len(filters) == 52
    np.testing.assert_array_equal(filters["trendA"][2], ctx.ok["trendA"])
    np.testing.assert_array_equal(filters["giantA"][2], ctx.ok["giantA"])
    top12 = roster(days, ctx.rankings, col, top=12)

    # look-ahead self-test: every price after session T_TEST x1.5, decisions through T_TEST must not move
    cut = book.days.index(days[T_TEST])
    scale = pd.Series(np.where(np.arange(len(book.days)) > cut, 1.5, 1.), index=book.cls.index)
    pbook = dataclasses.replace(book, cls=book.cls.mul(scale, axis=0), opn=book.opn.mul(scale, axis=0),
                                adj=book.adj.mul(scale, axis=0))
    pbars = {s: b.mul(np.where(b.index > pd.Timestamp(days[T_TEST]), 1.5, 1.), axis=0) for s, b in bars.items()}
    prank = [S.momentum_ranking(pbook, ctx.start + j, day)[1] for j, day in enumerate(days)]
    assert [[(r["key"], r["rank"]) for r in x] for x in prank[:T_TEST + 1]] == \
           [[(r["key"], r["rank"]) for r in x] for x in ctx.rankings[:T_TEST + 1]]
    pfilters, _ = build_filters(pbook, days, tickers, prank, col, pbars, earn, grades)
    changed_after = 0
    for name in filters:
        np.testing.assert_array_equal(filters[name][2][:T_TEST + 2], pfilters[name][2][:T_TEST + 2], err_msg=name)
        changed_after += int((filters[name][2][T_TEST + 2:] != pfilters[name][2][T_TEST + 2:]).any())
    pgates = study.market_gates(pbook, days)
    for g in GATES:
        np.testing.assert_array_equal(ctx.gates[g][:T_TEST + 2], pgates[g][:T_TEST + 2])
    plans_checked = 0
    for name in filters:
        for mode, (freq, method) in MODES.items():
            a = study.plan(book, days, ctx.rankings, col, filters[name][2], freq, method)
            b = study.plan(pbook, days, prank, col, pfilters[name][2], freq, method)
            assert same_targets(a, b, T_TEST), (name, mode)
            plans_checked += 1
    lookahead = (f"prices after session index {T_TEST} ({days[T_TEST]}) x1.5 in book.cls/opn/adj and all {len(bars)} grade_bars: "
                 f"rankings, {len(filters)} eligibility matrices (rows 0..{T_TEST + 1}), 3 gates and {plans_checked} plans' targets "
                 f"through session {T_TEST} identical; {changed_after}/{len(filters)} matrices did change after T (test has power; "
                 "the unchanged ones are earnings calendars, scale-invariant ratios and the frozen grade tables, which cannot be perturbed)")

    curves, configs = {}, {}
    base = study.replay(*args, ctx.weekly, ctx.gates["none"])
    np.testing.assert_allclose(base.usd, ctx.baseline.close, rtol=1e-10)
    again = study.plan(book, days, ctx.rankings, col, np.ones_like(ctx.ok["any"]), "week", "rotate")
    assert same_targets(again, ctx.weekly, len(days))
    blank = dict(subfamily="base", filter="", mode="wk", gate="none", kind="", params="", n_filters=0,
                 semis_cap=config.RULE["semis_cap"], sub_max=0, avail_h1=1, avail_ja=1, dup_codex=0)
    curves["base"], configs["base"] = base, dict(blank)
    for g in GATES[1:]:
        curves[f"weekly.{g}"] = study.replay(*args, ctx.weekly, ctx.gates[g])
        configs[f"weekly.{g}"] = {**blank, "subfamily": "ref_gate", "gate": g, "dup_codex": 1}

    def add(key, sub, targets, gate, **extra):
        assert key not in curves, key
        curves[key] = study.replay(*args, targets, ctx.gates[gate])
        configs[key] = {**blank, "subfamily": sub, "gate": gate, **extra}

    # singles
    single_targets, single_curves = {}, {}
    for name, (sub, params, elig) in filters.items():
        for mode, (freq, method) in MODES.items():
            targets = study.plan(book, days, ctx.rankings, col, elig, freq, method)
            single_targets[name, mode] = targets
            for g in GATES:
                add(f"{name}.{mode}.{g}", sub, targets, g, filter=name, mode=mode, kind=params.get("kind", ""),
                    params=json.dumps({k: v for k, v in params.items() if k != "kind"}), n_filters=1,
                    dup_codex=int(name in ("trendA", "giantA")))
            single_curves[name, mode] = curves[f"{name}.{mode}.none"]

    # F7 concentration
    sub_of = {key: watchlist.resolve_subsector(book.ticker[key]) or book.sector[key] for key in col}
    everything = np.ones_like(ctx.ok["any"])
    for kmax in (1, 2):
        for mode in ("wk", "rot"):
            with rule(sub_max=kmax, sub_of=sub_of):
                targets = study.plan(book, days, ctx.rankings, col, everything, *MODES[mode])
            add(f"submax{kmax}.{mode}.none", "F7", targets, "none", filter=f"submax{kmax}", mode=mode,
                kind="max_names_per_subsector", sub_max=kmax)
    for cap in (25, 35):
        with rule(semis_cap=cap / 100):
            targets = study.plan(book, days, ctx.rankings, col, everything, "week", "rotate")
        add(f"semis{cap}.wk.none", "F7", targets, "none", filter=f"semis{cap}", mode="wk", kind="semis_cap", semis_cap=cap / 100)
    decisions = {"best_singles": {}, "qualifiers": {}}
    for train, flag in TRAINS.items():
        best = best_singles(single_curves, base, filters, train)
        decisions["best_singles"][train] = [f"{n}.{m}" for n, m in best]
        for name, mode in best:
            for cap in (25, 35):
                key = f"semis{cap}+{name}.{mode}.none"
                if key in curves:
                    configs[key][flag] = 1
                    continue
                with rule(semis_cap=cap / 100):
                    targets = study.plan(book, days, ctx.rankings, col, filters[name][2], *MODES[mode])
                add(key, "F7", targets, "none", filter=f"semis{cap}+{name}", mode=mode, kind="semis_cap+filter",
                    semis_cap=cap / 100, n_filters=1, avail_h1=int(train == "h1"), avail_ja=int(train == "jan_aug"))

    # F8 combinations, chosen per training window and mode
    for train, flag in TRAINS.items():
        for mode, (freq, method) in MODES.items():
            picked, n_qual = qualifiers(single_curves, base, filters, train, mode)
            decisions["qualifiers"][f"{train}.{mode}"] = dict(qualified=n_qual, used=picked)
            for size in (2, 3):
                for combo in itertools.combinations(sorted(picked), size):
                    label = "+".join(combo)
                    if f"{label}.{mode}.none" in curves:
                        for g in GATES:
                            configs[f"{label}.{mode}.{g}"][flag] = 1
                        continue
                    elig = np.logical_and.reduce([filters[c][2] for c in combo])
                    targets = study.plan(book, days, ctx.rankings, col, elig, freq, method)
                    for g in GATES:
                        add(f"{label}.{mode}.{g}", "F8", targets, g, filter=label, mode=mode, kind=f"and{size}",
                            n_filters=size, avail_h1=int(train == "h1"), avail_ja=int(train == "jan_aug"))

    for key, frame in curves.items():
        assert frame.exposure.between(-1e-9, 1 + 1e-9).all(), key
    table = common.save(FAMILY, curves, configs)
    cfg = pd.DataFrame.from_dict(configs, orient="index")
    st = common.stats_table(curves)

    # 10bp one-way cost versions of the window returns
    for w, (a, b) in common.WINDOWS.items():
        vals = {}
        for key, c in curves.items():
            c10 = pd.DataFrame({"krw": c.krw10, "usd": c.usd * c.krw10 / c.krw, "turn": c.turn, "exposure": c.exposure})
            s = common.window_stats(c10, a, b)
            vals[key] = (s["krw"], s["usd"], s["usd_mdd"])
        st[[f"{w}_krw10", f"{w}_usd10", f"{w}_usd10_mdd"]] = pd.DataFrame.from_dict(vals, orient="index")

    # exposure-matched controls (USD); above base's exposure the control is base itself
    b = st.loc["base"]
    for w in ("h2", "sep_nov", "jul_aug"):
        ctrl = np.array([common.matched_control(base, e, w) for e in st[f"{w}_expo"]])
        nan = np.isnan(ctrl[:, 0])
        st[f"{w}_ctrl_usd"] = np.where(nan, b[f"{w}_usd"], ctrl[:, 0])
        st[f"{w}_ctrl_mdd"] = np.where(nan, b[f"{w}_usd_mdd"], ctrl[:, 1])
        st[f"{w}_ctrl_is_base"] = nan
        st[f"{w}_beat_ctrl_ret"] = st[f"{w}_usd"] > st[f"{w}_ctrl_usd"] + 1e-9
        st[f"{w}_beat_ctrl_both"] = st[f"{w}_beat_ctrl_ret"] & (st[f"{w}_usd_mdd"] > st[f"{w}_ctrl_mdd"] + 1e-9)
        st[f"{w}_beat_base_both"] = (st[f"{w}_usd"] > b[f"{w}_usd"] + 1e-9) & (st[f"{w}_usd_mdd"] > b[f"{w}_usd_mdd"] + 1e-9)
    full = cfg.join(st)
    full.index.name = "strategy"
    full.to_csv(common.CACHE / "filters_stats.csv")
    cand = full.drop(index="base")

    def describe(key, train=None, test=None):
        r = full.loc[key]
        d = dict(strategy=key, h1_krw=r.h1_krw, h2_krw=r.h2_krw, h2_usd=r.h2_usd, h2_usd_mdd=r.h2_usd_mdd,
                 jan_aug_krw=r.jan_aug_krw, jul_aug_usd=r.jul_aug_usd, sep_nov_krw=r.sep_nov_krw, sep_nov_usd=r.sep_nov_usd,
                 sep_nov_usd_mdd=r.sep_nov_usd_mdd, oct_nov_usd=r.oct_nov_usd, h2_expo=r.h2_expo, h2_turn=r.h2_turn,
                 sep_nov_expo=r.sep_nov_expo, sep_nov_turn=r.sep_nov_turn, h2_usd10=r.h2_usd10, sep_nov_usd10=r.sep_nov_usd10,
                 h2_krw10=r.h2_krw10, sep_nov_krw10=r.sep_nov_krw10, h2_ctrl_usd=r.h2_ctrl_usd, h2_ctrl_mdd=r.h2_ctrl_mdd,
                 sep_nov_ctrl_usd=r.sep_nov_ctrl_usd, sep_nov_ctrl_mdd=r.sep_nov_ctrl_mdd)
        if train:
            d.update(train_krw=r[f"{train}_krw"], train_krw_mdd=r[f"{train}_krw_mdd"], test_krw=r[f"{test}_krw"],
                     test_krw_mdd=r[f"{test}_krw_mdd"], test_usd=r[f"{test}_usd"], test_usd_mdd=r[f"{test}_usd_mdd"],
                     test_expo=r[f"{test}_expo"], test_turn=r[f"{test}_turn"], test_usd10=r[f"{test}_usd10"],
                     test_krw10=r[f"{test}_krw10"], test_ctrl_usd=r[f"{test}_ctrl_usd"], test_ctrl_mdd=r[f"{test}_ctrl_mdd"])
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}

    selections = []
    for protocol, train, test, flag in (("P1", "h1", "h2", "avail_h1"), ("P2", "jan_aug", "sep_nov", "avail_ja")):
        pool = full[full[flag] == 1]
        for scope, sub in (("all", pool), ("no_gate", pool[pool.gate == "none"]),
                           ("singles_no_gate", pool[(pool.gate == "none") & (pool.n_filters <= 1) & ~pool.subfamily.isin(["F7"])])):
            for rule_name, key in common.select(sub, train).items():
                selections.append(dict(protocol=protocol, scope=scope, rule=rule_name, pool=len(sub), **describe(key, train, test)))

    groups = []

    def group(label, sub):
        if not len(sub):
            return
        groups.append(dict(group=label, n=len(sub), med_h2_usd=sub.h2_usd.median(), best_h2_usd=sub.h2_usd.max(),
                           med_jul_aug_usd=sub.jul_aug_usd.median(), med_sep_nov_usd=sub.sep_nov_usd.median(),
                           med_h2_expo=sub.h2_expo.median(), med_h2_turn=sub.h2_turn.median(),
                           beat_base_both_h2=int(sub.h2_beat_base_both.sum()),
                           beat_ctrl_ret_h2=int(sub.h2_beat_ctrl_ret.sum()), beat_ctrl_both_h2=int(sub.h2_beat_ctrl_both.sum()),
                           beat_ctrl_ret_sep_nov=int(sub.sep_nov_beat_ctrl_ret.sum()),
                           beat_ctrl_both_sep_nov=int(sub.sep_nov_beat_ctrl_both.sum()),
                           beat_base_ret_h2=int((sub.h2_usd > b.h2_usd + 1e-9).sum()),
                           beat_base_ret_sep_nov=int((sub.sep_nov_usd > b.sep_nov_usd + 1e-9).sum())))
    for subf in ("F1", "F2", "F3", "F4", "F5", "F6"):
        for mode in MODES:
            group(f"{subf}.{mode}.none", cand[(cand.subfamily == subf) & (cand["mode"] == mode) & (cand.gate == "none")])
        for g in GATES[1:]:
            group(f"{subf}.all_modes.{g}", cand[(cand.subfamily == subf) & (cand.gate == g)])
    group("F7.none", cand[cand.subfamily == "F7"])
    for mode in MODES:
        for g in GATES:
            group(f"F8.{mode}.{g}", cand[(cand.subfamily == "F8") & (cand["mode"] == mode) & (cand.gate == g)])
    group("ref_gate", cand[cand.subfamily == "ref_gate"])
    group("ALL", cand)
    groups = pd.DataFrame(groups)
    groups.to_csv(common.CACHE / "filters_subfamily.csv", index=False)

    info = info_content(ctx, filters, member, top12)
    info.to_csv(common.CACHE / "filters_info_content.csv", index=False)

    single = cand[(cand.n_filters == 1) & (cand.gate == "none") & cand.subfamily.isin(["F1", "F2", "F3", "F4", "F5", "F6"])]
    earn_cov = earn[earn.reaction_date.between("2026-01-01", "2026-11-17")]
    held = sorted({tickers[i] for w in ctx.weekly.values() for i in np.flatnonzero(w > 0)})
    summary = dict(
        family=FAMILY, n_ledgers=len(curves), n_candidates=len(cand), runtime_seconds=round(time.time() - t0, 1),
        counts_by_subfamily=cand.groupby("subfamily").size().to_dict(),
        counts_avail=dict(h1=int(cand.avail_h1.sum()), jan_aug=int(cand.avail_ja.sum())),
        base=describe("base"), lookahead=lookahead, decisions=decisions, selections=selections,
        hindsight=dict(h2_usd=describe(cand.h2_usd.idxmax()), sep_nov_usd=describe(cand.sep_nov_usd.idxmax()),
                       h2_usd_no_gate=describe(cand[cand.gate == "none"].h2_usd.idxmax()),
                       sep_nov_usd_no_gate=describe(cand[cand.gate == "none"].sep_nov_usd.idxmax()),
                       h2_usd_full_exposure=describe(cand[cand.h2_expo >= b.h2_expo - .02].h2_usd.idxmax()),
                       sep_nov_usd_full_exposure=describe(cand[cand.sep_nov_expo >= b.sep_nov_expo - .02].sep_nov_usd.idxmax())),
        counts=dict(total=len(cand), beat_base_both_h2_usd=int(cand.h2_beat_base_both.sum()),
                    also_beat_ctrl_ret=int((cand.h2_beat_base_both & cand.h2_beat_ctrl_ret).sum()),
                    also_beat_ctrl_both=int((cand.h2_beat_base_both & cand.h2_beat_ctrl_both).sum()),
                    beat_base_both_sep_nov=int(cand.sep_nov_beat_base_both.sum()),
                    sep_nov_also_beat_ctrl_both=int((cand.sep_nov_beat_base_both & cand.sep_nov_beat_ctrl_both).sum()),
                    beat_base_both_h2_usd_10bp=int(((cand.h2_usd10 > b.h2_usd10 + 1e-9) & (cand.h2_usd10_mdd > b.h2_usd10_mdd + 1e-9)).sum()),
                    no_gate_total=int((cand.gate == "none").sum()),
                    no_gate_beat_base_both_h2=int((cand.h2_beat_base_both & (cand.gate == "none")).sum()),
                    no_gate_also_beat_ctrl_both=int((cand.h2_beat_base_both & cand.h2_beat_ctrl_both & (cand.gate == "none")).sum()),
                    beat_ctrl_both_h2_and_sep_nov=int((cand.h2_beat_ctrl_both & cand.sep_nov_beat_ctrl_both).sum()),
                    beat_ctrl_both_h2_jul_aug_sep_nov=int((cand.h2_beat_ctrl_both & cand.sep_nov_beat_ctrl_both & cand.jul_aug_beat_ctrl_both).sum())),
        rank_persistence=dict(
            spearman_h1_vs_h2_usd=float(cand.h1_usd.corr(cand.h2_usd, method="spearman")),
            spearman_jan_aug_vs_sep_nov_usd=float(cand.jan_aug_usd.corr(cand.sep_nov_usd, method="spearman")),
            spearman_jul_aug_vs_sep_nov_usd=float(cand.jul_aug_usd.corr(cand.sep_nov_usd, method="spearman")),
            singles_spearman_jul_aug_excess_vs_sep_nov_excess=float(
                (single.jul_aug_usd - single.jul_aug_ctrl_usd).corr(single.sep_nov_usd - single.sep_nov_ctrl_usd, method="spearman"))),
        earnings=dict(rows_2026=len(earn_cov), sources=earn_cov.source.value_counts().to_dict(),
                      tickers_with_dates=int(earn_cov.ticker.nunique()), tickers=len(tickers),
                      held_ever=len(held), held_without_any_date=[t for t in held if t not in set(earn_cov.ticker)],
                      held_event_counts={t: int((earn_cov.ticker == t).sum()) for t in held}),
        groups=groups.to_dict("records"))
    (common.CACHE / "filters_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float),
                                                       encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_ledgers", "n_candidates", "runtime_seconds", "counts_by_subfamily",
                                               "counts_avail", "decisions", "counts", "rank_persistence")},
                     indent=1, ensure_ascii=False, default=float))
    print(lookahead)
    for s in selections:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
               if k in ("protocol", "scope", "rule", "pool", "strategy", "train_krw", "train_krw_mdd", "test_krw", "test_usd",
                        "test_usd_mdd", "test_expo", "test_turn", "test_ctrl_usd", "test_ctrl_mdd", "test_usd10")})
    return table


if __name__ == "__main__":
    main()
