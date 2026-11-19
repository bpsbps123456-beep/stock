"""ext_stops: exits driven by the account's own equity curve and by per-stock price lines.

Run from the repo root:  py data/derived/analysis/desk_defense_check_20261119/ext_stops.py
    --download   fetch daily OHLC once and freeze it under frozen_ohlc/ (needed only for S6)

Every decision executed at the open of session j uses closes through j-1 (S6: a resting stop whose
level is fixed from closes through j-1 and that is hit by the Low of session j).

Subfamilies and grid (counts in GRID, asserted in main):
  S1 equity drawdown stop   x {8,10,12,15,20,25}% x peak window {20,60,all} x re-entry
                            {bounce 5/10/15%, shadow NAV >= SMA10/SMA20, time-out 5/10/20}      = 144
  S2 equity moving average  n {5,10,20,30,50} x re-entry confirmation {1,3}                     = 10
  S3 volatility target      window {10,20,40} x target {30,40,50,60,80}%                        = 15
  S4 per-stock trailing     x {8,10,12,15,20,25,30}% x after {week,sma10,sma20,spell}
                            x freed weight {cash,rotate}                                        = 56
  S5 per-stock loss cut     x {7,10,15,20}% x after(4) x freed(2)                               = 32
  S6 resting intraday stop  the S4 and S5 grids with Low-triggered fills at min(Open, stop)     = 88
  S7 pieces x market gates  pieces picked per subfamily S1..S5 on the training window only
                            (h1 and jan_aug, rules return/stability; members that never acted in
                            training are skipped) x {SOXX20r3, SOXX100r1, QQQ10r1, breadth>=19},
                            plus stock piece x equity piece. A combination whose ledger equals one
                            of its parts is dropped and listed.                                 (data dependent)
  control                   base x the same four gates (not candidates, never selected)        = 4

Selection pools: breadth>=19 combinations are kept out of both forward selections (Codex fixed the
19% line on Feb-Sep data) and S7 pieces picked on jan_aug are kept out of the h1 selection.
Outputs: cache/stops_daily.parquet and out/stops_candidates.csv via common.save, diagnostics under
cache/stops_*.csv|json. The look-ahead self-test runs on every execution.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common  # noqa: E402

study, S = common.study, common.S
FROZEN = HERE / "frozen_ohlc"
OHLC = FROZEN / "ohlc.csv"
T_TEST = 150                      # look-ahead self-test: prices after this session index are x1.5
S7_GATES = ("SOXX20r3", "SOXX100r1", "QQQ10r1", "breadth19")

S1_X, S1_L = (.08, .10, .12, .15, .20, .25), (20, 60, None)
S1_RE = [("bounce", .05), ("bounce", .10), ("bounce", .15), ("sma", 10), ("sma", 20),
         ("time", 5), ("time", 10), ("time", 20)]
S2_N, S2_R = (5, 10, 20, 30, 50), (1, 3)
S3_W, S3_T = (10, 20, 40), (.30, .40, .50, .60, .80)
S4_X, S5_X = (.08, .10, .12, .15, .20, .25, .30), (.07, .10, .15, .20)
AFTER, FREED = ("week", "sma10", "sma20", "spell"), ("cash", "rotate")
GRID = {"S1": len(S1_X) * len(S1_L) * len(S1_RE), "S2": len(S2_N) * len(S2_R), "S3": len(S3_W) * len(S3_T),
        "S4": len(S4_X) * len(AFTER) * len(FREED), "S5": len(S5_X) * len(AFTER) * len(FREED)}
GRID["S6"] = GRID["S4"] + GRID["S5"]


# ---------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------

def _sma(values, n):
    """Trailing simple mean over n rows (axis 0); NaN until n rows exist. Exact for constant input."""
    out = np.full(values.shape, np.nan)
    view = np.lib.stride_tricks.sliding_window_view(values, n, axis=0)
    out[n - 1:] = view.mean(axis=-1)
    return out


def load_ohlc(ctx):
    """Frozen daily Low, reconciled with the frozen book. Returns (low[days, tickers], usable[tickers], report)."""
    if not OHLC.exists():
        return None, None, {"status": "missing"}
    raw = pd.read_csv(OHLC, dtype={"date": str})
    wide = {f: raw.pivot(index="date", columns="ticker", values=f) for f in ("open", "low", "close")}
    book_days = [d for d in ctx.book.days if d >= "2025-12-01"]
    usable, worst = [], {}
    for t in ctx.tickers:
        if t not in wide["open"].columns:
            usable.append(False)
            worst[t] = None
            continue
        diffs = []
        for field, frame in (("open", ctx.book.opn), ("close", ctx.book.cls)):
            mine = wide[field][t].reindex(book_days).to_numpy(float)
            ref = frame[t].reindex(book_days).to_numpy(float)
            both = np.isfinite(ref)
            if not np.isfinite(mine[both]).all():
                diffs.append(np.inf)
            elif both.any():
                diffs.append(np.max(np.abs(mine[both] / ref[both] - 1)))
        lo = wide["low"][t].reindex(ctx.days).to_numpy(float)
        need = np.isfinite(ctx.close[:, ctx.tickers.index(t)])
        sane = np.isfinite(lo[need]).all() and (lo[need] <= np.minimum(
            ctx.opn[:, ctx.tickers.index(t)], ctx.close[:, ctx.tickers.index(t)])[need] * 1.005).all()
        worst[t] = float(max(diffs)) if diffs else None
        usable.append(bool(diffs) and max(diffs) <= .005 and bool(sane))
    low = wide["low"].reindex(index=ctx.days, columns=ctx.tickers).to_numpy(float)
    finite = [v for v in worst.values() if v is not None and np.isfinite(v)]
    report = {"status": "ok", "tickers": len(ctx.tickers), "usable": int(sum(usable)),
              "dropped": [t for t, u in zip(ctx.tickers, usable) if not u],
              "max_rel_diff_all": float(max(finite)) if finite else None,
              "max_rel_diff_usable": float(max([worst[t] for t, u in zip(ctx.tickers, usable) if u], default=np.nan))}
    return low, np.array(usable), report


def download_ohlc(ctx):
    import yfinance as yf
    FROZEN.mkdir(exist_ok=True)
    frame = yf.download(sorted(ctx.tickers), start="2025-12-01", end="2026-11-18", auto_adjust=False,
                        progress=False, threads=True, group_by="ticker")
    rows = []
    for t in sorted(ctx.tickers):
        if t not in frame.columns.get_level_values(0):
            continue
        part = frame[t].dropna(subset=["Open", "High", "Low", "Close"])
        for day, r in part.iterrows():
            rows.append((day.strftime("%Y-%m-%d"), t, r.Open, r.High, r.Low, r.Close))
    out = pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close"])
    out = out[out.date <= "2026-11-17"].sort_values(["ticker", "date"])
    out.to_csv(OHLC, index=False, float_format="%.6f", lineterminator="\n")
    meta = {"source": "yfinance", "version": yf.__version__, "auto_adjust": False,
            "start": "2025-12-01", "end_exclusive": "2026-11-18", "rows": len(out),
            "tickers": int(out.ticker.nunique()), "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (FROZEN / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def prices(ctx, low=None, bump_after=None, seed=None):
    """Price arrays, optionally perturbed for the look-ahead test.

    bump_after=T, seed=None: every price at session index > T is multiplied by 1.5 (the lead's test).
    bump_after=T, seed=int:  stricter. Closes from session T on and opens/lows after T are multiplied by
        a random factor in [0.6, 1.5] per session and ticker, so a rule that reads the close of the
        session it trades in is caught as well.
    """
    full = ctx.book.cls.reindex(columns=ctx.tickers).to_numpy(float).copy()
    opn, close = ctx.opn.copy(), ctx.close.copy()
    low = None if low is None else low.copy()
    if bump_after is not None:
        T = bump_after
        if seed is None:
            factor, first_close = np.full(close.shape, 1.5), T + 1
        else:
            factor, first_close = np.random.default_rng(seed).uniform(.6, 1.5, close.shape), T
        full[ctx.start + first_close:] *= factor[first_close:]
        close[first_close:] *= factor[first_close:]
        opn[T + 1:] *= factor[T + 1:]
        if low is not None:
            low[T + 1:] *= factor[T + 1:]
    np.testing.assert_array_equal(np.nan_to_num(full[ctx.start:]), np.nan_to_num(close))
    sma = {n: _sma(full, n)[ctx.start:] for n in (10, 20)}
    base = study.replay(ctx.days, opn, close, ctx.fx, ctx.weekly, ctx.gates["none"])
    return SimpleNamespace(opn=opn, close=close, low=low, sma=sma, base=base, nav=base.usd.to_numpy())


# ---------------------------------------------------------------------------
# S1-S3: decisions from the always-invested base shadow NAV (USD closes through j-1)
# ---------------------------------------------------------------------------

PAD = 60  # the account is worth 1.0 before its first session


def s1_gate(nav, x, window, reentry):
    """Drawdown stop. After a re-entry the peak restarts at the close that triggered it; without that
    reset the old peak would stop the account out again on the next close."""
    n = len(nav)
    kind, value = reentry
    line = _sma(np.r_[np.ones(PAD), nav], value)[PAD:] if kind == "sma" else None
    gate, exits = np.ones(n, dtype=bool), []
    invested, anchor, exit_at, low = True, 0, None, None
    for j in range(1, n):
        i = j - 1
        if invested:
            first = anchor if window is None else max(anchor, i - window + 1)
            if nav[i] <= (1 - x) * nav[first:i + 1].max():
                invested, exit_at, low = False, j, nav[i]
                exits.append(j)
        else:
            low = min(low, nav[i])
            back = ((kind == "bounce" and nav[i] >= (1 + value) * low) or
                    (kind == "sma" and nav[i] >= line[i]) or (kind == "time" and j - exit_at >= value))
            if back:
                invested, anchor = True, i
        gate[j] = invested
    return gate, exits


def s2_gate(nav, n_sma, confirm):
    """Invested while shadow NAV >= its SMA; out after one close below, back after `confirm` closes at/above."""
    hist = np.r_[np.ones(PAD), nav]
    above = hist >= _sma(hist, n_sma)
    gate, exits = np.ones(len(nav), dtype=bool), []
    state, run = True, confirm
    for i in range(PAD - 1, len(hist) - 1):          # close i decides session j = i - PAD + 1
        if above[i]:
            run += 1
            if run >= confirm:
                state = True
        else:
            run, state = 0, False
        j = i - PAD + 1
        if not state and gate[j - 1] if j else False:
            exits.append(j)
        gate[j] = state
    return gate, exits


def s3_frac(nav, window, target):
    """min(1, target / annualised realised vol of the last `window` shadow daily returns), nearest 0.25.
    Session 0 is a structural zero (the model is not invested yet) and is kept out of the window."""
    path = np.r_[1., nav]
    ret = path[1:] / path[:-1] - 1
    frac = np.ones(len(nav))
    for j in range(window + 1, len(nav)):
        vol = ret[j - window:j].std(ddof=1) * np.sqrt(252)
        if vol > 0:
            frac[j] = min(1., np.floor(target / vol * 4 + .5) / 4)
    return frac


# ---------------------------------------------------------------------------
# S4-S6: per-stock stops
# ---------------------------------------------------------------------------

def stock_sim(ctx, px, kind, x, after, freed, intraday=False, usable=None):
    """Weekly 7-stock rule with per-stock stops. Returns (target schedule, own ledger, stop events).

    kind    trail: close <= (1-x) * highest close since the name was (re)bought
            loss:  close <= (1-x) * the open at which it was (re)bought
            x=None disables stops (must reproduce the base).
    after   week:  blocked until the next weekly rebalance after the stop
            sma10/sma20: blocked until a later close is back at/above the stock's own SMA
            spell: blocked while the unstopped weekly rule keeps selecting the name
    freed   cash:   the weekly selection is untouched; the stopped weight waits in USD cash and no
                    other holding is traded (drifted weights are republished)
            rotate: the name leaves the selection; the 7 slots are refilled from that session's
                    ranking without blocked names and re-weighted with project_caps
    intraday (S6): a resting stop at (1-x)*reference, fixed before the session, active from the
            session after the entry; hit when Low <= stop, filled at min(Open, stop). Tickers whose
            OHLC did not reconcile keep the close-based trigger.
    """
    days, col, book = ctx.days, ctx.col, ctx.book
    opn, close = px.opn, px.close
    m = opn.shape[1]
    shares, cash, drag = np.zeros(m), 1., 1.
    prev, base_prev, target = set(), set(), {}
    blocked, entry, ref = {}, {}, {}
    pending = started = False
    targets, rows, events = {}, [], []
    for j, day in enumerate(days):
        o = opn[j]
        o_mark = np.nan_to_num(o)
        nav = cash + shares @ o_mark
        ranked = ctx.rankings[j]
        week = j == 0 or S.iso_week(day) != S.iso_week(days[j - 1])
        rebal = bool(ranked) and (not started or week)
        fired = []
        if x is not None and j > 0:
            for k in sorted(entry):
                c = col[k]
                by_close = not intraday or not usable[c]
                if by_close and entry[k] <= j - 1 and close[j - 1, c] <= (1 - x) * ref[k]:
                    fired.append(k)
        # release blocks (information through close j-1 only)
        if after == "week":
            if rebal:
                blocked = {}
        elif after in ("sma10", "sma20"):
            line = px.sma[int(after[3:])]
            for k, at in list(blocked.items()):
                if j - 1 >= at and close[j - 1, col[k]] >= line[j - 1, col[k]]:
                    del blocked[k]
        if rebal:
            base_prev = set(S.select(ranked, base_prev))
            if after == "spell":
                blocked = {k: at for k, at in blocked.items() if k in base_prev}
        w = None
        if freed == "cash":
            if rebal:
                fired = [k for k in fired if k in base_prev]      # leaving anyway: an ordinary exit
            for k in fired:
                blocked[k] = j
            if rebal:
                prev = set(base_prev)
                raw = {r["key"]: r["time_weight"] for r in ranked if r["key"] in prev}
                total = sum(raw.values())
                target = S.project_caps({k: v / total for k, v in raw.items()}, book.sector)
                w = np.zeros(m)
                for k in prev:
                    if k not in blocked:
                        w[col[k]] = target.get(k, 0.)
            elif started:
                drift = shares * o_mark / nav
                changed = False
                for k in fired:
                    drift[col[k]] = 0.
                    changed = True
                for k in sorted(prev):                             # released names return at their weekly weight
                    if k not in blocked and shares[col[k]] == 0 and target.get(k, 0.) > 0:
                        add = min(target[k], 1 - drift.sum())
                        if add > 1e-12:
                            drift[col[k]] = add
                            changed = True
                if changed:
                    w = drift
        else:
            for k in fired:
                blocked[k] = j
                prev.discard(k)
            if ranked and (rebal or fired or pending):
                pool = [r for r in ranked if r["key"] not in blocked]
                prev = set(S.select(pool, prev))
                raw = {r["key"]: r["time_weight"] for r in pool if r["key"] in prev}
                total = sum(raw.values())
                target = S.project_caps({k: v / total for k, v in raw.items()}, book.sector)
                w = np.zeros(m)
                for k in prev:
                    w[col[k]] = target.get(k, 0.)
                pending = False
        for k in fired:
            events.append(dict(key=k, ticker=book.ticker[k], j=j, fill=float(o[col[k]]), how="close"))
        turn = 0.
        if w is not None:
            if not (o[w > 0] > 0).all():
                raise ValueError(f"Missing executable open on {day}")
            targets[j] = w
            turn = np.abs(w - shares * o_mark / nav).sum()
            shares = np.divide(nav * w, o, out=np.zeros_like(w), where=w > 0)
            cash = nav * (1 - w.sum())
            drag *= 1 - .001 * turn
            started = True
        exposure = 1 - cash / nav
        held = {k for k, c in col.items() if shares[c] > 0}
        for k in set(entry) - held:
            del entry[k], ref[k]
        for k in held - set(entry):
            entry[k] = j
            ref[k] = float(o[col[k]]) if kind == "loss" else -np.inf
        if intraday and x is not None:
            for k in sorted(entry):
                c = col[k]
                if not usable[c] or entry[k] >= j:
                    continue
                level = (1 - x) * ref[k]
                if px.low[j, c] <= level:
                    fill = min(o[c], level)
                    sold = shares[c] * fill / nav
                    cash += shares[c] * fill
                    shares[c] = 0.
                    turn += sold
                    drag *= 1 - .001 * sold
                    del entry[k], ref[k]
                    blocked[k] = j
                    events.append(dict(key=k, ticker=book.ticker[k], j=j, fill=float(fill),
                                       how="gap" if fill == o[c] else "intraday"))
                    if freed == "rotate":
                        prev.discard(k)
                        pending = True
        c_mark = np.nan_to_num(close[j])
        if not np.isfinite(close[j][shares != 0]).all():
            raise ValueError(f"Missing closing mark on {day}")
        if kind == "trail":
            for k in entry:
                ref[k] = max(ref[k], float(close[j, col[k]]))
        usd = cash + shares @ c_mark
        rows.append((usd * ctx.fx[j], usd, turn, exposure, usd * ctx.fx[j] * drag))
    return targets, pd.DataFrame(rows, index=days, columns=common.LEDGER), events


def stock_grid():
    for kind, xs in (("trail", S4_X), ("loss", S5_X)):
        for x in xs:
            for after in AFTER:
                for freed in FREED:
                    yield kind, x, after, freed


def stock_key(sub, kind, x, after, freed):
    return f"{sub}_{kind}{round(x * 100):02d}_{after}_{freed}"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def decisions(ctx, px, usable):
    """Every S1-S6 decision as plain arrays: equity gates/fractions, stock schedules and stop events."""
    out = {"gate": {}, "frac": {}, "stock": {}, "exits": {}, "cfg": {}}
    for x in S1_X:
        for window in S1_L:
            for kind, value in S1_RE:
                tag = f"{round(value * 100):02d}" if kind == "bounce" else f"{value}"
                key = f"S1_dd{round(x * 100):02d}_L{window or 'all'}_{kind}{tag}"
                out["gate"][key], out["exits"][key] = s1_gate(px.nav, x, window, (kind, value))
                out["cfg"][key] = dict(subfamily="S1_equity_drawdown", x=x, window=window or "all",
                                       reentry=f"{kind}{tag}", market="none")
    for n_sma in S2_N:
        for confirm in S2_R:
            key = f"S2_navsma{n_sma}_r{confirm}"
            out["gate"][key], out["exits"][key] = s2_gate(px.nav, n_sma, confirm)
            out["cfg"][key] = dict(subfamily="S2_equity_sma", x=n_sma, reentry=f"r{confirm}", market="none")
    for window in S3_W:
        for target in S3_T:
            key = f"S3_vol{window}_t{round(target * 100)}"
            out["frac"][key] = s3_frac(px.nav, window, target)
            out["cfg"][key] = dict(subfamily="S3_vol_target", x=target, window=window, market="none")
    for kind, x, after, freed in stock_grid():
        sub = "S4" if kind == "trail" else "S5"
        key = stock_key(sub, kind, x, after, freed)
        out["stock"][key] = stock_sim(ctx, px, kind, x, after, freed)
        out["cfg"][key] = dict(subfamily=f"{sub}_stock_{'trailing' if kind == 'trail' else 'losscut'}_{freed}",
                               x=x, reentry=after, freed=freed, market="none")
        if px.low is not None:
            key = stock_key("S6", kind, x, after, freed)
            out["stock"][key] = stock_sim(ctx, px, kind, x, after, freed, intraday=True, usable=usable)
            out["cfg"][key] = dict(subfamily=f"S6_intraday_{'trailing' if kind == 'trail' else 'losscut'}_{freed}",
                                   x=x, reentry=after, freed=freed, market="none")
    return out


def _same_through(ref, alt, T):
    """Assert every decision executed at sessions <= T is identical; return how many sets were compared."""
    checked = 0
    for key, gate in ref["gate"].items():
        assert (gate[:T + 1] == alt["gate"][key][:T + 1]).all(), key
        assert [e for e in ref["exits"][key] if e <= T] == [e for e in alt["exits"][key] if e <= T], key
        checked += 1
    for key, frac in ref["frac"].items():
        assert (frac[:T + 1] == alt["frac"][key][:T + 1]).all(), key
        checked += 1
    for key, (targets, _, events) in ref["stock"].items():
        other, _, other_events = alt["stock"][key]
        a, b = [j for j in targets if j <= T], [j for j in other if j <= T]
        assert a == b, key
        for j in a:
            np.testing.assert_array_equal(targets[j], other[j], err_msg=key)
        cut = lambda rows: [(e["key"], e["j"], e["fill"], e["how"]) for e in rows if e["j"] <= T]
        assert cut(events) == cut(other_events), key
        checked += 1
    return checked


def _power(ref, alt, T):
    """How much the perturbation moved later decisions, and how many one-session-leaky rules it would catch."""
    later = sum(not np.array_equal(ref[kind][k], alt[kind][k]) for kind in ("gate", "frac") for k in ref[kind])
    later += sum(sorted(j for j in ref["stock"][k][0] if j > T) != sorted(j for j in alt["stock"][k][0] if j > T)
                 or [e["j"] for e in ref["stock"][k][2]] != [e["j"] for e in alt["stock"][k][2]] for k in ref["stock"])
    # a rule that used the close of its own session would carry decision T+1 at session T
    leaky = sum(ref[kind][k][T + 1] != alt[kind][k][T + 1] for kind in ("gate", "frac") for k in ref[kind])
    for k in ref["stock"]:
        mine, other = ref["stock"][k][0].get(T + 1), alt["stock"][k][0].get(T + 1)
        leaky += (mine is None) != (other is None) or (mine is not None and not np.array_equal(mine, other))
    return int(later), int(leaky)


def lookahead_test(ctx, low, usable, ref):
    """Required test (x1.5 after T=150) plus random-factor variants at four cut points."""
    out = []
    for T, seed in ((T_TEST, None), (60, 11), (110, 12), (T_TEST, 13), (190, 14)):
        alt = decisions(ctx, prices(ctx, low, bump_after=T, seed=seed), usable)
        checked = _same_through(ref, alt, T)
        later, leaky = _power(ref, alt, T)
        out.append(dict(T=T, day=ctx.days[T], perturbation="x1.5 after T" if seed is None else
                        "random 0.6-1.5 on closes >= T and opens/lows > T", decision_sets_checked=checked,
                        identical_through_T=True, sets_changed_after_T=later, one_session_leak_would_be_caught=leaky))
    # the ranking the per-stock rotation reads: bump the adjusted closes it is built from
    T = T_TEST
    adj = ctx.book.adj.copy()
    adj.iloc[ctx.start + T:] *= 1.5
    book = dataclasses.replace(ctx.book, adj=adj)
    same = lambda rows: [(r["key"], r["mom120"], r["time_weight"]) for r in rows]
    for j in range(T + 1):
        assert same(S.momentum_ranking(book, ctx.start + j, ctx.days[j])[1]) == same(ctx.rankings[j]), ctx.days[j]
    return out


def pick_pieces(table, cfg, curves):
    """One or two pieces per subfamily group and training window, chosen on training data only.

    Rule: common.select on {base + group members that traded differently from the base inside the
    training window}; when a rule returns base (no member passes its drawdown test) the group's
    highest training KRW return stands in, so every group contributes a piece. Members that never
    acted in training are base clones there and are left out.
    """
    groups = {"S1": "S1_", "S2": "S2_", "S3": "S3_", "S4": "S4_", "S5": "S5_"}
    pieces = {}
    base = curves["base"]
    for train in ("h1", "jan_aug"):
        inside = (base.index >= common.WINDOWS[train][0]) & (base.index <= common.WINDOWS[train][1])
        for group, prefix in groups.items():
            members = [k for k in table.index if k.startswith(prefix) and cfg[k]["market"] == "none"
                       and not np.allclose(curves[k].to_numpy()[inside], base.to_numpy()[inside], rtol=1e-12, atol=1e-12)]
            chosen = common.select(table.loc[["base"] + members], train)
            for rule, key in chosen.items():
                if key == "base":
                    key = sorted(members, key=lambda k: (-table.at[k, f"{train}_krw"], k))[0]
                pieces.setdefault(key, set()).add(train)
    return pieces


def build(ctx, low, usable):
    px = prices(ctx, low)
    args = (ctx.days, px.opn, px.close, ctx.fx)
    none = ctx.gates["none"]
    base = study.replay(*args, ctx.weekly, none)
    assert abs(common.window_stats(base, *common.WINDOWS["h1"])["krw"] - 2.4505) < 5e-5
    assert abs(common.window_stats(base, *common.WINDOWS["h2"])["krw"] + 0.3945) < 5e-5
    np.testing.assert_allclose(base.usd, ctx.baseline.close, rtol=1e-10)
    # stops disabled: the custom schedule and the custom ledger are the base
    for freed in FREED:
        for intraday in (False, True) if low is not None else (False,):
            targets, ledger, events = stock_sim(ctx, px, "trail", None, "week", freed, intraday, usable)
            assert sorted(targets) == sorted(ctx.weekly) and not events
            for j in targets:
                np.testing.assert_array_equal(targets[j], ctx.weekly[j])
            np.testing.assert_allclose(study.replay(*args, targets, none).to_numpy(), base.to_numpy(), rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(ledger.to_numpy(), base.to_numpy(), rtol=1e-10, atol=1e-12)

    dec = decisions(ctx, px, usable)
    curves, configs = {"base": base}, {"base": dict(subfamily="base", market="none")}
    for key, gate in dec["gate"].items():
        curves[key] = study.replay(*args, ctx.weekly, gate)
    for key, frac in dec["frac"].items():
        curves[key] = common.replay_frac(*args, ctx.weekly, frac)
    worst = 0.
    for key, (targets, ledger, events) in dec["stock"].items():
        if key.startswith("S6"):
            curves[key] = ledger                                   # fills inside the session: own loop
        else:
            curves[key] = study.replay(*args, targets, none)
            np.testing.assert_allclose(ledger.to_numpy(), curves[key].to_numpy(), rtol=1e-9, atol=1e-9, err_msg=key)
            worst = max(worst, float(np.abs(ledger.to_numpy() - curves[key].to_numpy()).max()))
    for key in dec["cfg"]:
        configs[key] = dict(dec["cfg"][key], basis="grid", pit_h1=1, pit_jan_aug=1, control=0)
    configs["base"].update(basis="grid", pit_h1=1, pit_jan_aug=1, control=0)

    # S7: pieces (training-window choices) x market gates
    breadth = pd.read_csv(common.CODEX / "breadth.csv", index_col=0).breadth.shift().reindex(ctx.days)
    assert breadth.notna().all()
    market = {name: ctx.gates[name] for name in S7_GATES[:3]}
    market["breadth19"] = (breadth >= 19).to_numpy()
    table = common.stats_table(curves)
    pieces = pick_pieces(table, configs, curves)

    def flags(trains, name):
        clean = name != "breadth19"         # Codex fixed the 19% line on Feb-Sep data
        return dict(basis="+".join(sorted(trains)), control=0, pit_h1=int(clean and "h1" in trains),
                    pit_jan_aug=int(clean))

    def same(a, b):
        return np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12, atol=1e-12)

    control = {name: study.replay(*args, ctx.weekly, gate) for name, gate in market.items()}
    skipped = []                           # a combination that never departs from one of its parts is not a candidate
    for piece, trains in sorted(pieces.items()):
        for name, gate in market.items():
            key = f"S7_{piece}_{name}"
            if piece in dec["gate"]:
                curve = study.replay(*args, ctx.weekly, dec["gate"][piece] & gate)
            elif piece in dec["frac"]:
                curve = common.replay_frac(*args, ctx.weekly, dec["frac"][piece] * gate)
            else:
                curve = study.replay(*args, dec["stock"][piece][0], gate)
            if same(curve, control[name]) or same(curve, curves[piece]):
                skipped.append(key)
                continue
            curves[key] = curve
            configs[key] = dict(subfamily=f"S7_{piece[:2]}_x_market", piece=piece, market=name, **flags(trains, name))
    for train in ("h1", "jan_aug"):       # stock piece under an equity-curve piece, same training window
        stocks = [p for p, t in pieces.items() if train in t and p in dec["stock"]]
        equity = [p for p, t in pieces.items() if train in t and p not in dec["stock"]]
        for a in sorted(stocks):
            for b in sorted(equity):
                key = f"S7_{a}_x_{b}"
                trains = pieces[a] & pieces[b]
                if key in curves or key in skipped:
                    continue
                if b in dec["gate"]:
                    curve = study.replay(*args, dec["stock"][a][0], dec["gate"][b])
                else:
                    curve = common.replay_frac(*args, dec["stock"][a][0], dec["frac"][b])
                if same(curve, curves[a]) or same(curve, curves[b]):
                    skipped.append(key)
                    continue
                curves[key] = curve
                configs[key] = dict(subfamily="S7_stock_x_equity", piece=a, market=b, **flags(trains, b))
    for name in market:
        key = f"ctl_base_{name}"
        curves[key] = control[name]
        configs[key] = dict(subfamily="control_gate_only", market=name, basis="control", control=1,
                            pit_h1=0, pit_jan_aug=0)
    return px, dec, curves, configs, pieces, worst, skipped


# ---------------------------------------------------------------------------
# reporting (cache/ only)
# ---------------------------------------------------------------------------

def period(day):
    return "h1" if day < "2026-07-01" else "jul_aug" if day < "2026-09-01" else "sep_nov"


def event_table(ctx, px, dec, configs):
    """One row per stop: fill and whether the name (or the shadow account) was lower 10/20 sessions later."""
    rows = []
    n = len(ctx.days)
    for key, (_, _, events) in dec["stock"].items():
        for e in events:
            c = ctx.col[e["key"]]
            row = dict(strategy=key, subfamily=configs[key]["subfamily"], name=e["ticker"], day=ctx.days[e["j"]],
                       how=e["how"], fill=e["fill"])
            for h in (10, 20):
                row[f"fwd{h}"] = px.close[e["j"] + h, c] / e["fill"] - 1 if e["j"] + h < n else np.nan
            rows.append(row)
    value = ctx.baseline.value                                   # shadow USD value at each open
    for key, exits in dec["exits"].items():
        for j in exits:
            row = dict(strategy=key, subfamily=configs[key]["subfamily"], name="ACCOUNT", day=ctx.days[j],
                       how="equity", fill=float(value[j]))
            for h in (10, 20):
                row[f"fwd{h}"] = px.nav[j + h] / value[j] - 1 if j + h < n else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out["period"] = out.day.map(period)
    return out


def event_summary(events, curves_n):
    rows = []
    for (sub, per), g in events.groupby(["subfamily", "period"]):
        rows.append(dict(subfamily=sub, period=per, strategies=curves_n[sub], stops=len(g),
                         stops_per_strategy=len(g) / curves_n[sub],
                         distinct_name_days=g[["name", "day"]].drop_duplicates().shape[0],
                         n10=int(g.fwd10.notna().sum()), lower10=float((g.fwd10.dropna() < 0).mean()),
                         mean_fwd10=float(g.fwd10.mean()),
                         n20=int(g.fwd20.notna().sum()), lower20=float((g.fwd20.dropna() < 0).mean()),
                         mean_fwd20=float(g.fwd20.mean())))
    return pd.DataFrame(rows)


def report(ctx, curves, configs, pieces, extra):
    table = common.stats_table(curves)
    cfg = pd.DataFrame.from_dict(configs, orient="index")
    cand = [k for k in table.index if not cfg.at[k, "control"]]
    out = dict(extra)
    out["n_candidates"] = len(cand) - 1
    out["subfamily_counts"] = cfg.loc[cand].subfamily.value_counts().sort_index().to_dict()
    out["pieces"] = {k: sorted(v) for k, v in sorted(pieces.items())}

    def describe(key, train, test):
        r = table.loc[key]
        return dict(strategy=key, **{f"{side}_{m}": float(r[f"{w}_{m}"]) for side, w in (("train", train), ("test", test))
                                     for m in ("krw", "krw_mdd", "usd", "usd_mdd", "expo", "turn")})

    out["selections"] = []
    for label, train, test, flag in (("P1", "h1", "h2", "pit_h1"), ("P2", "jan_aug", "sep_nov", "pit_jan_aug")):
        pools = {"clean": [k for k in cand if cfg.at[k, flag]], "all_incl_contaminated": cand,
                 "grid_only_S1_S6": [k for k in cand if cfg.at[k, "basis"] == "grid"]}
        for pool_name, pool in pools.items():
            for rule, key in common.select(table.loc[pool], train).items():
                out["selections"].append(dict(protocol=label, train=train, test=test, pool=pool_name,
                                              pool_size=len(pool), rule=rule, **describe(key, train, test)))
    b = table.loc["base"]
    out["base"] = {c: float(b[c]) for c in table.columns}
    others = [k for k in cand if k != "base"]
    out["hindsight"] = {"best_h2_usd": describe(table.loc[others, "h2_usd"].idxmax(), "h1", "h2"),
                        "best_sep_nov_usd": describe(table.loc[others, "sep_nov_usd"].idxmax(), "jan_aug", "sep_nov")}
    better = [k for k in others if table.at[k, "h2_usd"] > b.h2_usd and table.at[k, "h2_usd_mdd"] > b.h2_usd_mdd]
    rest = [k for k in table.index if k != "base"]
    control = {k: common.matched_control(curves["base"], table.at[k, "h2_expo"], "h2") for k in rest}
    table["h2_ctl_usd"] = pd.Series({k: v[0] for k, v in control.items()})
    table["h2_ctl_usd_mdd"] = pd.Series({k: v[1] for k, v in control.items()})
    ctl9 = {k: common.matched_control(curves["base"], table.at[k, "sep_nov_expo"], "sep_nov") for k in rest}
    table["sep_nov_ctl_usd"] = pd.Series({k: v[0] for k, v in ctl9.items()})
    table["jul_aug_ctl_usd"] = pd.Series({k: common.matched_control(curves["base"], table.at[k, "jul_aug_expo"],
                                                                    "jul_aug")[0] for k in rest})
    beat = [k for k in better if np.isfinite(control[k][0]) and table.at[k, "h2_usd"] > control[k][0]]
    undefined = [k for k in better if not np.isfinite(control[k][0])]
    beat_any = [k for k in others if np.isfinite(control[k][0]) and table.at[k, "h2_usd"] > control[k][0]]
    out["counts"] = dict(total=len(others), beat_base_both_h2_usd=len(better), also_beat_matched_control=len(beat),
                         control_undefined_among_better=len(undefined),
                         beat_matched_control_any=len(beat_any),
                         beat_control_return_and_mdd=len([k for k in beat if table.at[k, "h2_usd_mdd"] > control[k][1]]),
                         beat_sep_nov_control=len([k for k in others if np.isfinite(ctl9[k][0])
                                                   and table.at[k, "sep_nov_usd"] > ctl9[k][0]]))
    out["beat_control_keys"] = sorted(beat, key=lambda k: -(table.at[k, "h2_usd"] - control[k][0]))[:25]
    scopes = {"S1_S5_close_based": lambda k: k[:2] in ("S1", "S2", "S3", "S4", "S5"),
              "S6_intraday": lambda k: k.startswith("S6"), "S7_combinations": lambda k: k.startswith("S7")}
    out["counts_by_scope"] = {name: dict(total=sum(map(test, others)), beat_base_both_h2_usd=sum(map(test, better)),
                                         also_beat_matched_control=sum(map(test, beat)),
                                         control_undefined_among_better=sum(map(test, undefined)))
                              for name, test in scopes.items()}
    ledgers = {k: curves[k].usd.round(10).to_numpy().tobytes() for k in cand}
    out["distinct_ledgers"] = dict(candidates_incl_base=len(cand), distinct=len(set(ledgers.values())),
                                   identical_to_base=sum(v == ledgers["base"] for k, v in ledgers.items() if k != "base"))
    for row in out["selections"]:          # exposure-matched control of each selected rule in its test window
        key, test = row["strategy"], row["test"]
        row["test_ctl_usd"], row["test_ctl_usd_mdd"] = (float(v) for v in common.matched_control(
            curves["base"], table.at[key, f"{test}_expo"], test)) if key != "base" else (float(b[f"{test}_usd"]),
                                                                                          float(b[f"{test}_usd_mdd"]))
        row["base_test_usd"], row["base_test_usd_mdd"] = float(b[f"{test}_usd"]), float(b[f"{test}_usd_mdd"])
        row["base_test_krw"], row["base_test_krw_mdd"] = float(b[f"{test}_krw"]), float(b[f"{test}_krw_mdd"])
        row["full_krw"], row["full_krw_mdd"] = float(table.at[key, "full_krw"]), float(table.at[key, "full_krw_mdd"])
    table["subfamily"] = cfg.subfamily
    table["edge_h2"] = table.h2_usd - table.h2_ctl_usd
    table["edge_sep_nov"] = table.sep_nov_usd - table.sep_nov_ctl_usd
    table["edge_jul_aug"] = table.jul_aug_usd - table.jul_aug_ctl_usd
    both = table.loc[others, ["edge_jul_aug", "edge_sep_nov"]].dropna()
    out["edge_sign_by_subperiod"] = dict(
        defined_in_both=len(both), positive_in_both=int(((both.edge_jul_aug > 0) & (both.edge_sep_nov > 0)).sum()),
        positive_jul_aug_only=int(((both.edge_jul_aug > 0) & (both.edge_sep_nov <= 0)).sum()),
        positive_sep_nov_only=int(((both.edge_jul_aug <= 0) & (both.edge_sep_nov > 0)).sum()),
        positive_in_neither=int(((both.edge_jul_aug <= 0) & (both.edge_sep_nov <= 0)).sum()),
        rank_correlation=float(both.edge_jul_aug.rank().corr(both.edge_sep_nov.rank())))
    rows = []
    for sub, g in table.loc[others + [k for k in table.index if cfg.at[k, "control"]]].groupby("subfamily"):
        rows.append(dict(subfamily=sub, n=len(g), h2_usd_median=g.h2_usd.median(), h2_usd_best=g.h2_usd.max(),
                         h2_usd_worst=g.h2_usd.min(), h2_usd_mdd_median=g.h2_usd_mdd.median(),
                         jul_aug_usd_median=g.jul_aug_usd.median(), sep_nov_usd_median=g.sep_nov_usd.median(),
                         h2_expo_median=g.h2_expo.median(), h2_turn_median=g.h2_turn.median(),
                         h1_krw_median=g.h1_krw.median(), h2_krw_median=g.h2_krw.median(),
                         beat_base_both=int(((g.h2_usd > b.h2_usd) & (g.h2_usd_mdd > b.h2_usd_mdd)).sum()),
                         beat_control=int((g.edge_h2 > 0).sum()), edge_h2_median=g.edge_h2.median(),
                         edge_jul_aug_median=g.edge_jul_aug.median(), edge_sep_nov_median=g.edge_sep_nov.median(),
                         beat_base_sep_nov_both=int(((g.sep_nov_usd > b.sep_nov_usd) &
                                                     (g.sep_nov_usd_mdd > b.sep_nov_usd_mdd)).sum())))
    sub = pd.DataFrame(rows)
    common.CACHE.mkdir(exist_ok=True)
    show = ["h1_krw", "jan_aug_krw", "h2_krw", "h2_usd", "h2_usd_mdd", "jul_aug_usd", "sep_nov_usd", "sep_nov_usd_mdd",
            "h2_expo", "edge_h2"]
    # S7: what the piece adds to the gate alone
    rows = []
    for k in cand:
        gate = cfg.at[k, "market"]
        if str(cfg.at[k, "subfamily"]).endswith("_x_market"):
            ref = table.loc[f"ctl_base_{gate}"]
            rows.append(dict(strategy=k, piece=cfg.at[k, "piece"], gate=gate,
                             **{f"d_{c}": float(table.at[k, c] - ref[c]) for c in
                                ("h1_krw", "h2_usd", "h2_usd_mdd", "jul_aug_usd", "sep_nov_usd", "h2_expo")}))
    marginal = pd.DataFrame(rows)
    marginal.to_csv(common.CACHE / "stops_s7_marginal.csv", index=False)
    out["s7_piece_minus_gate_only"] = dict(
        n=len(marginal), h2_usd_better=int((marginal.d_h2_usd > 0).sum()),
        h2_usd_and_mdd_better=int(((marginal.d_h2_usd > 0) & (marginal.d_h2_usd_mdd > 0)).sum()),
        d_h2_usd_median=float(marginal.d_h2_usd.median()), d_sep_nov_usd_median=float(marginal.d_sep_nov_usd.median()),
        d_h1_krw_median=float(marginal.d_h1_krw.median()),
        by_gate={g: dict(n=len(x), d_h2_usd_median=float(x.d_h2_usd.median()), d_h2_usd_max=float(x.d_h2_usd.max()),
                         d_h1_krw_median=float(x.d_h1_krw.median())) for g, x in marginal.groupby("gate")})
    out["gate_only_controls"] = table.loc[[k for k in table.index if cfg.at[k, "control"]], show].to_dict("index")
    # S6 against the close-based rule with the same parameters
    rows = []
    for k in cand:
        if k.startswith("S6_"):
            twin = ("S4_" if "_trail" in k else "S5_") + k[3:]
            rows.append(dict(strategy=k, kind="trailing" if "_trail" in k else "losscut", freed=cfg.at[k, "freed"],
                             **{f"d_{c}": float(table.at[k, c] - table.at[twin, c]) for c in
                                ("h1_krw", "h2_usd", "h2_usd_mdd", "jul_aug_usd", "sep_nov_usd", "full_usd")}))
    pairs = pd.DataFrame(rows)
    if len(pairs):
        pairs.to_csv(common.CACHE / "stops_s6_pairs.csv", index=False)
        out["s6_minus_close_based"] = {
            f"{a}_{b}": dict(n=len(x), h2_usd_better=int((x.d_h2_usd > 0).sum()), d_h2_usd_median=float(x.d_h2_usd.median()),
                             d_sep_nov_usd_median=float(x.d_sep_nov_usd.median()), d_h1_krw_median=float(x.d_h1_krw.median()),
                             d_full_usd_median=float(x.d_full_usd.median()), full_usd_better=int((x.d_full_usd > 0).sum()))
            for (a, b), x in pairs.groupby(["kind", "freed"])}
    # parameter gradients inside the grid
    grid = table.loc[[k for k in cand if cfg.at[k, "basis"] == "grid" and k != "base"]].join(cfg[["x", "reentry", "window"]])
    params = []
    for field in ("x", "reentry", "window"):
        for (subfamily, value), g in grid.groupby(["subfamily", field]):
            params.append(dict(subfamily=subfamily, parameter=field, value=value, n=len(g),
                               **{c + "_median": float(g[c].median()) for c in show}))
    pd.DataFrame(params).to_csv(common.CACHE / "stops_parameters.csv", index=False)
    sub.to_csv(common.CACHE / "stops_subfamily.csv", index=False)
    table.to_csv(common.CACHE / "stops_windows.csv")
    (common.CACHE / "stops_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out, sub, table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="fetch and freeze daily OHLC for S6")
    opts = parser.parse_args()
    began = time.time()
    ctx = common.context()
    if opts.download:
        print("download", download_ohlc(ctx))
    low, usable, ohlc = load_ohlc(ctx)
    if low is None:
        print("S6 skipped: frozen_ohlc/ohlc.csv is missing (run once with --download)")
    px, dec, curves, configs, pieces, worst, skipped = build(ctx, low, usable)
    grid = {s: sum(1 for k in dec["cfg"] if k.startswith(s + "_")) for s in GRID}
    assert all(grid[s] == GRID[s] for s in GRID if s != "S6" or low is not None), grid
    lookahead = lookahead_test(ctx, low, usable, dec)
    common.save("stops", curves, configs)
    events = event_table(ctx, px, dec, configs)
    events.to_csv(common.CACHE / "stops_events.csv", index=False)
    per_sub = pd.Series({configs[k]["subfamily"]: 0 for k in dec["cfg"]})
    for k in dec["cfg"]:
        per_sub[configs[k]["subfamily"]] += 1
    summary = event_summary(events, per_sub)
    summary.to_csv(common.CACHE / "stops_event_summary.csv", index=False)
    extra = dict(grid=grid, ohlc=ohlc, own_loop_max_abs_diff_vs_study_replay=worst, s7_skipped_as_duplicates=skipped,
                 lookahead=lookahead, ranking_lookahead="rankings for sessions <= 150 identical after "
                 "adjusted closes from session 150 on were multiplied by 1.5")
    out, sub, table = report(ctx, curves, configs, pieces, extra)
    out["runtime_seconds"] = time.time() - began
    (common.CACHE / "stops_summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    print(json.dumps({k: out[k] for k in ("n_candidates", "grid", "subfamily_counts", "ohlc", "lookahead", "counts",
                                          "counts_by_scope", "distinct_ledgers", "s7_skipped_as_duplicates",
                                          "own_loop_max_abs_diff_vs_study_replay")}, indent=1, default=float))
    print("pieces", out["pieces"])
    print(pd.DataFrame(out["selections"]).round(4).to_string())
    print("hindsight", json.dumps(out["hindsight"], indent=1, default=float))
    print(sub.round(4).to_string())
    for name in ("s7_piece_minus_gate_only", "gate_only_controls", "s6_minus_close_based"):
        print(name, json.dumps(out.get(name), indent=1, default=float))
    print(pd.read_csv(common.CACHE / "stops_parameters.csv").round(4).to_string())
    print(summary.round(4).to_string())
    print(f"runtime {out['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
