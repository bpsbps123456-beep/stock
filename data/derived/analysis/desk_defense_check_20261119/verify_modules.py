"""verify_modules: can the universe / stops / filters / tabsig ledgers be trusted, and do they say what the lead reads?

Run from the repository root:  py data/derived/analysis/desk_defense_check_20261119/verify_modules.py

Reads the saved ledgers (cache/<family>_daily.parquet) and the frozen Codex engine through common.py.
Writes only cache/verify_modules.json. Nothing here imports or runs the four builders' main().
Sections: 1) independent rebuilds of saved ledgers with loops written here, 2) blunt same-day test on
every ledger, 3) the lead's four readings, 4) regime alarm tables.
"""
from __future__ import annotations

import json
import sys
from bisect import bisect_left
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

study, S = common.study, common.S
CACHE = common.CACHE
FAMILIES = ("universe", "stops", "filters", "tabsig")
EPISODES = {"jul_fall": ("2026-06-23", "2026-07-29"), "aug_sep_rally": ("2026-07-30", "2026-09-22"),
            "octnov_fall": ("2026-10-13", "2026-11-17"), "h2": ("2026-07-01", "2026-11-17"),
            "sep_nov": ("2026-09-01", "2026-11-17"), "full": ("2026-01-02", "2026-11-17")}
OUT = {}
pd.set_option("display.width", 260, "display.max_columns", 60, "display.max_rows", 400)


# ---------------------------------------------------------------------------
# own account loops (no study.replay / replay_frac / switched_targets inside)
# ---------------------------------------------------------------------------

def shadow_weights(plan, opn, n):
    """Open weights (after its own trade) of an always-invested account that follows `plan`."""
    m = opn.shape[1]
    shares, cash = np.zeros(m), 1.0
    out = np.zeros((n, m))
    for j in range(n):
        o = np.nan_to_num(opn[j])
        nav = cash + shares @ o
        if j in plan:
            w = plan[j]
            shares = np.where(w > 0, nav * w / np.where(w > 0, opn[j], 1.0), 0.0)
            cash = nav * (1 - w.sum())
        out[j] = shares * o / nav
    return out


def account(orders, opn, close):
    """orders: j -> absolute weight vector to hold from the open of j. Returns usd close, exposure, weights."""
    n, m = opn.shape
    shares, cash = np.zeros(m), 1.0
    usd, expo, opn_nav, held = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros((n, m))
    for j in range(n):
        o = np.nan_to_num(opn[j])
        nav = cash + shares @ o
        opn_nav[j] = nav
        if j in orders:
            w = orders[j]
            shares = np.where(w > 0, nav * w / np.where(w > 0, opn[j], 1.0), 0.0)
            cash = nav * (1 - w.sum())
        expo[j] = 1 - cash / nav
        usd[j] = cash + shares @ np.nan_to_num(close[j])
        held[j] = shares
    return usd, expo, opn_nav, held


def gated_orders(plan, frac, opn):
    """Hold frac[j] of the shadow basket: trade when the shadow publishes while frac>0 or when frac changes."""
    n = len(frac)
    sw = shadow_weights(plan, opn, n)
    orders = {}
    for j in range(n):
        if (frac[j] > 0 and j in plan) or (j > 0 and abs(frac[j] - frac[j - 1]) > 1e-12):
            orders[j] = frac[j] * sw[j]
    return orders


def switch_orders(plan_a, plan_b, gate, opn):
    """Follow plan_a while gate is on, plan_b while it is off; a flip moves to the other basket's drifted weights."""
    n = len(gate)
    wa, wb = shadow_weights(plan_a, opn, n), shadow_weights(plan_b, opn, n)
    orders, prev = {}, None
    for j in range(n):
        on = bool(gate[j])
        plan, sw = (plan_a, wa) if on else (plan_b, wb)
        if j in plan or (prev is not None and on != prev):
            orders[j] = sw[j].copy()
        prev = on
    return orders


def rel_diff(mine, saved):
    saved = np.asarray(saved, float)
    return float(np.max(np.abs(np.asarray(mine) / saved - 1)))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_ledgers():
    out = {}
    for fam in FAMILIES:
        d = pd.read_parquet(CACHE / f"{fam}_daily.parquet")
        d["strategy"], d["date"] = d.strategy.astype(str), d.date.astype(str)
        out[fam] = {c: d.pivot(index="date", columns="strategy", values=c) for c in ("usd", "exposure", "turn", "krw")}
    return out


def ep_ret(usd, days, a, b):
    """Return and MDD of a usd close path inside [a, b], anchored on the close before a."""
    ix = [i for i, d in enumerate(days) if a <= d <= b]
    before = usd[ix[0] - 1] if ix[0] else 1.0
    path = np.r_[before, usd[ix]]
    return path[-1] / before - 1, float(np.min(path / np.maximum.accumulate(path) - 1))


def control(base_usd, base_expo, expo, days, a, b):
    """Exposure-matched control: hold (rule exposure / base exposure) of the base every day of [a, b]."""
    ix = [i for i, d in enumerate(days) if a <= d <= b]
    before = base_usd[ix[0] - 1] if ix[0] else 1.0
    daily = np.r_[before, base_usd[ix]]
    daily = daily[1:] / daily[:-1] - 1
    scale = expo[ix].mean() / base_expo[ix].mean()
    grow = lambda k: float(np.prod(1 + k * daily) - 1)
    return grow(scale), float(expo[ix].mean()), grow(min(scale, 1.0)), grow(1.0)


def stats(led, base_usd, base_expo, days):
    """Per strategy and episode: return, mdd, mean exposure and three comparisons with the base:
    _xs  minus the control holding (rule exposure / base exposure) of the base (may lever the base above 1),
    _xc  the same with the ratio capped at 1 (a rule more invested than the base is compared with the base),
    _d   minus the base itself."""
    rows = {}
    U, E = led["usd"], led["exposure"]
    for key in U.columns:
        u, e = U[key].to_numpy(), E[key].to_numpy()
        r = {}
        for name, (a, b) in EPISODES.items():
            ret, mdd = ep_ret(u, days, a, b)
            ctl, ex, capped, plain = control(base_usd, base_expo, e, days, a, b)
            r[f"{name}"], r[f"{name}_mdd"], r[f"{name}_expo"], r[f"{name}_xs"] = ret, mdd, ex, ret - ctl
            r[f"{name}_xc"], r[f"{name}_d"] = ret - capped, ret - plain
        rows[key] = r
    return pd.DataFrame.from_dict(rows, orient="index")


def binom_tail(k, n, p):
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))) if n else np.nan


def show(title, frame, digits=3):
    print(f"\n== {title}")
    print(frame.round(digits).to_string())


# ---------------------------------------------------------------------------
# 1) independent rebuilds
# ---------------------------------------------------------------------------

def nonsemi_plan(ctx, length, n, weight):
    """Weekly top-n non-semiconductor TIME names by `length`-session adjusted-close momentum through close j-1."""
    adj = ctx.book.adj.reindex(columns=ctx.tickers).to_numpy(float)
    labels = {ctx.book.ticker[k]: ctx.book.sector[k] for k in ctx.col}
    tcol = {t: i for i, t in enumerate(ctx.tickers)}
    plan, picks = {}, {}
    for j in sorted(ctx.weekly):
        k = ctx.start + j
        rows = []
        for r in ctx.rankings[j]:
            if S.in_semis_budget(r["sector"]):
                continue
            a, b = adj[k - 1, ctx.col[r["key"]]], adj[k - 1 - length, ctx.col[r["key"]]]
            if np.isfinite(a) and np.isfinite(b) and b > 0:
                rows.append((-(a / b - 1), r["key"], r))
        top = [r for _, _, r in sorted(rows, key=lambda x: x[:2])[:n]]
        if weight == "time":
            total = sum(r["time_weight"] for r in top)
            raw = {r["ticker"]: r["time_weight"] / total for r in top}
        else:
            raw = {r["ticker"]: 1 / len(top) for r in top}
        w = np.zeros(len(ctx.tickers))
        for t, v in S.project_caps(raw, labels).items():
            w[tcol[t]] = v
        plan[j], picks[j] = w, [r["ticker"] for r in top]
    return plan, picks


def dd_gate(nav, x, window, bounce):
    """Equity drawdown stop: out after a close <= (1-x) x the window's highest close, back after +bounce from the low."""
    n = len(nav)
    gate = np.ones(n, bool)
    invested, anchor, trough = True, 0, None
    for j in range(1, n):
        i = j - 1
        if invested:
            lo = max(anchor, i - window + 1)
            if nav[i] <= (1 - x) * max(nav[lo:i + 1]):
                invested, trough = False, nav[i]
        else:
            trough = min(trough, nav[i])
            if nav[i] >= (1 + bounce) * trough:
                invested, anchor = True, i
        gate[j] = invested
    return gate


def trail_stop_week_cash(ctx, x, low=None, usable=None):
    """Per-stock trailing stop, stopped weight waits in cash until the next weekly rebalance.
    low=None: close <= (1-x) x highest close since entry, sold at the next open (S4).
    low given: resting stop hit when Low <= level, filled at min(open, level), for `usable` tickers (S6)."""
    opn, close = ctx.opn, ctx.close
    n, m = opn.shape
    shares, cash = np.zeros(m), 1.0
    entry, peak = {}, {}
    usd, expo, stops = np.zeros(n), np.zeros(n), []
    for j in range(n):
        o = np.nan_to_num(opn[j])
        nav = cash + shares @ o
        fired = [c for c in entry if j > 0 and entry[c] <= j - 1 and close[j - 1, c] <= (1 - x) * peak[c]
                 and (low is None or not usable[c])]
        if j in ctx.weekly:
            w = ctx.weekly[j].copy()
            for c in fired:
                w[c] = 0.0
            shares = np.where(w > 0, nav * w / np.where(w > 0, opn[j], 1.0), 0.0)
            cash = nav * (1 - w.sum())
        else:
            for c in fired:
                cash += shares[c] * o[c]
                shares[c] = 0.0
        stops += [(j, c, "close") for c in fired if j not in ctx.weekly or ctx.weekly[j][c] > 0]
        expo[j] = 1 - cash / nav
        now = set(np.flatnonzero(shares > 0))
        for c in set(entry) - now:
            del entry[c], peak[c]
        for c in now - set(entry):
            entry[c], peak[c] = j, -np.inf
        if low is not None:
            for c in sorted(entry):
                if usable[c] and entry[c] < j and low[j, c] <= (1 - x) * peak[c]:
                    cash += shares[c] * min(o[c], (1 - x) * peak[c])
                    shares[c] = 0.0
                    del entry[c], peak[c]
                    stops.append((j, c, "intraday"))
        for c in entry:
            peak[c] = max(peak[c], close[j, c])
        usd[j] = cash + shares @ np.nan_to_num(close[j])
    return usd, expo, stops


def own_filters(ctx):
    """Two eligibility matrices rebuilt cell by cell from closes through session j-1."""
    C = ctx.book.cls.reindex(columns=ctx.tickers).to_numpy(float)
    A = ctx.book.adj.reindex(columns=ctx.tickers).to_numpy(float)
    Q = ctx.book.adj["QQQ"].to_numpy(float)
    n, m = len(ctx.days), len(ctx.tickers)
    above, rs = np.zeros((n, m), bool), np.zeros((n, m), bool)
    for j in range(n):
        i = ctx.start + j - 1
        bench = Q[i] / Q[i - 20] - 1
        for c in range(m):
            w50 = C[i - 49:i + 1, c]
            if np.isfinite(w50).all():
                above[j, c] = C[i, c] >= w50[-20:].mean() and C[i, c] >= w50.mean()
            r = A[i, c] / A[i - 20, c] - 1
            rs[j, c] = bool(np.isfinite(r) and r > bench)
    return {"above20_50": above, "rs20_qqq": rs}


def own_subbreadth(ctx, level):
    """Eligible unless < level of the name's AI sub-sector is within 10% of its all-time high at the close before session j."""
    import watchlist
    folder = common.CODEX / "cache" / "grade_bars"
    members = {}
    for sym, g in watchlist.AI_SUBSECTOR.items():
        members.setdefault(g, []).append(sym)
    us = pd.DatetimeIndex(ctx.book.days)
    n, m = len(ctx.days), len(ctx.tickers)
    elig = np.ones((n, m), bool)
    cache = {}
    for c, tk in enumerate(ctx.tickers):
        g = watchlist.resolve_subsector(tk)
        if not g:
            continue
        if g not in cache:
            flags = []
            for sym in members[g]:
                b = pd.read_parquet(folder / f"{sym}.parquet")
                near = (b.Close / b.High.cummax() - 1 >= -0.10).to_numpy(float)
                near[:59] = np.nan
                pos = b.index.searchsorted(us, side="right") - 1
                flags.append(np.where(pos >= 0, near[np.clip(pos, 0, None)], np.nan))
            arr = np.array(flags)
            cnt = np.isfinite(arr).sum(axis=0)
            cache[g] = np.where(cnt > 0, np.nansum(arr, axis=0) / np.maximum(cnt, 1), np.nan)
        share = cache[g]
        for j in range(n):
            v = share[ctx.start + j - 1]
            elig[j, c] = not (np.isfinite(v) and v < level)
    return elig


def fund_series(fund, days):
    """TIME fund cash weight and semiconductor weight per PDF, mapped to sessions (PDF dated before the next session)."""
    from desk import config as desk_config
    from gbrc.etf.market import is_cash_holding, yahoo_ticker
    from gbrc.etf.storage import load_fund_snapshots
    snaps = load_fund_snapshots(fund, "2026-11-17")
    pdfs = sorted(snaps)
    cash, semis = [], []
    for day in pdfs:
        c = s = 0.0
        for row in snaps[day]["holdings"]:
            w = float(row["weight"])
            if is_cash_holding(row):
                c += w
            elif S.sector_of(S.research_key(row["code"]), yahoo_ticker(row["code"], day) or "") in desk_config.SEMIS_SECTORS:
                s += w
        cash.append(c)
        semis.append(s)
    cash, semis = np.array(cash), np.array(semis)
    chg10 = np.r_[np.full(10, np.nan), semis[10:] - semis[:-10]]
    nxt = days[1:] + ["9999-12-31"]
    pick = [bisect_left(pdfs, d) - 1 for d in nxt]
    take = lambda v: np.array([v[i] if i >= 0 else np.nan for i in pick])
    return take(cash), take(chg10), [pdfs[i] if i >= 0 else None for i in pick]


def rebuilds(ctx, L, base_usd):
    days, opn, close = ctx.days, ctx.opn, ctx.close
    n = len(days)
    rows = []

    def check(fam, key, usd, expo=None, note=""):
        saved = L[fam]["usd"][key].reindex(days).to_numpy()
        row = dict(family=fam, ledger=key, max_rel_diff_usd=rel_diff(usd, saved), note=note)
        if expo is not None:
            row["max_abs_diff_exposure"] = float(np.max(np.abs(expo - L[fam]["exposure"][key].reindex(days).to_numpy())))
        rows.append(row)

    # base in every family
    usd, expo, opn_nav, _ = account(ctx.weekly, opn, close)
    for fam in FAMILIES:
        check(fam, "base", usd, expo, "weekly base through the own loop")
    assert rel_diff(usd, ctx.baseline.close) < 1e-9

    # universe: two U1a switches, one always-on basket, one scaled exit
    plans = {}
    for length, k, weight, gate in ((120, 7, "time", "SOXX20r1"), (20, 5, "equal", "SOXX100r1"), (120, 3, "time", "QQQ10r1")):
        plan, picks = nonsemi_plan(ctx, length, k, weight)
        plans[length, k, weight] = (plan, picks)
        u, e, _, _ = account(switch_orders(ctx.weekly, plan, ctx.gates[gate], opn), opn, close)
        check("universe", f"u1a_L{length}_N{k}_{weight}_{gate}", u, e, "own non-semis ranking + own switch loop")
    u, e, _, _ = account(plans[120, 7, "time"][0], opn, close)
    check("universe", "u1a_L120_N7_time_always", u, e, "own non-semis ranking, always held")
    frac = np.where(ctx.gates["SOXX50r1"], 1.0, 0.25)
    u, e, _, _ = account(gated_orders(ctx.weekly, frac, opn), opn, close)
    check("universe", "u1c_f25_SOXX50r1", u, e, "own fractional-exposure loop")

    # stops: equity drawdown stop, per-stock trailing stop (close) and resting intraday stop
    gate = dd_gate(base_usd, .10, 60, .10)
    u, e, _, _ = account(gated_orders(ctx.weekly, gate.astype(float), opn), opn, close)
    check("stops", "S1_dd10_L60_bounce10", u, e, f"own drawdown gate, {int((~gate).sum())} sessions out")
    for x in (.15, .10):
        u, e, st = trail_stop_week_cash(ctx, x)
        check("stops", f"S4_trail{round(100 * x):02d}_week_cash", u, e, f"own per-stock loop, {len(st)} stops")
    import ext_stops  # read-only helper: frozen daily lows and the reconciliation flags
    low, usable, rep = ext_stops.load_ohlc(ctx)
    if low is not None:
        u, e, st = trail_stop_week_cash(ctx, .15, low, usable)
        check("stops", "S6_trail15_week_cash", u, e,
              f"own resting-stop loop, {len(st)} stops ({sum(1 for s in st if s[2] == 'intraday')} intraday), usable tickers {int(usable.sum())}/{len(usable)}")
        ev = pd.read_csv(CACHE / "stops_events.csv")
        mine = sorted((days[j], ctx.tickers[c]) for j, c, _ in st)
        theirs = sorted(zip(*[ev[ev.strategy == "S6_trail15_week_cash"][c] for c in ("day", "name")]))
        rows.append(dict(family="stops", ledger="S6_trail15_week_cash events", max_rel_diff_usd=float(mine != theirs),
                         note=f"own stop list {len(mine)} vs stops_events.csv {len(theirs)} (0 = identical name-days)"))

    # filters: own eligibility matrices through Codex's plan(), own account loop
    elig = own_filters(ctx)
    for name, mode, gate in (("above20_50", "cash", "none"), ("above20_50", "wk", "SOXX20r3"), ("rs20_qqq", "wk", "none"),
                             ("rs20_qqq", "rot", "SOXX10r1")):
        freq, method = {"wk": ("week", "rotate"), "cash": ("change", "cash"), "rot": ("change", "rotate")}[mode]
        plan = study.plan(ctx.book, days, ctx.rankings, ctx.col, elig[name], freq, method)
        u, e, _, _ = account(gated_orders(plan, ctx.gates[gate].astype(float), opn), opn, close)
        check("filters", f"{name}.{mode}.{gate}", u, e, f"own eligibility (pass share {elig[name].mean():.2f}) + study.plan + own loop")

    for level, mode in ((.20, "wk"), (.40, "rot")):
        sb = own_subbreadth(ctx, level)
        freq, method = {"wk": ("week", "rotate"), "rot": ("change", "rotate")}[mode]
        plan = study.plan(ctx.book, days, ctx.rankings, ctx.col, sb, freq, method)
        u, e, _, _ = account(plan, opn, close)
        check("filters", f"subbreadth_{round(100 * level)}.{mode}.none", u, e,
              f"own sub-sector breadth (pass share {sb.mean():.2f}) + study.plan + own loop")

    # tabsig: fund cash / semis weight from the raw PDFs, breadth from Codex's table
    cash_w, semis_chg10, used = fund_series("456600", days)
    breadth = pd.read_csv(common.CODEX / "breadth.csv", index_col=0).breadth.reindex(days).to_numpy()
    for key, alarm, level in (("time_cash_wge4_x0", cash_w >= 4, 0.0), ("time_semis_w_chg10le-2_x0", semis_chg10 <= -2, 0.0),
                              ("breadthle15_x0.5", breadth <= 15, 0.5)):
        frac = np.ones(n)
        frac[1:] = np.where(alarm[:-1], level, 1.0)
        u, e, _, _ = account(gated_orders(ctx.weekly, frac, opn), opn, close)
        check("tabsig", key, u, e, f"own signal from raw data, alarm on {int(alarm.sum())} closes")
    table = pd.DataFrame(rows)
    show("1) independent rebuilds vs saved ledgers", table, 12)
    OUT["rebuilds"] = table.to_dict(orient="records")
    return opn_nav, plans, (cash_w, used)


# ---------------------------------------------------------------------------
# 2) blunt same-day test
# ---------------------------------------------------------------------------

def blunt(ctx, L, base_usd, base_open, st):
    """Days on which a rule cut exposure at the open: was the base's return THAT day unusually bad,
    compared with the day before (which a legitimate rule may react to)?"""
    days = ctx.days
    n = len(days)
    c2c = np.r_[base_usd[0] - 1, base_usd[1:] / base_usd[:-1] - 1]
    o2c = base_usd / base_open - 1                       # what an exit at the open actually avoids
    bad_c2c, bad_o2c = c2c <= np.quantile(c2c, .10), o2c <= np.quantile(o2c, .10)
    weekly = np.zeros(n, bool)
    weekly[sorted(ctx.weekly)] = True
    late = np.array([d > "2026-08-10" for d in days])    # after the builders' own T=150 self-test cut
    fam_rows, top_rows, date_rows = [], [], []
    after_bad = np.r_[False, bad_c2c[:-1]]
    cluster = dict(p_worst_o2c_given_prev_worst_c2c=float(bad_o2c[after_bad].mean()), n_prev_worst=int(after_bad.sum()),
                   p_worst_o2c_given_prev_worst_c2c_after_0810=float(bad_o2c[after_bad & late].mean()),
                   n_prev_worst_after_0810=int((after_bad & late).sum()), p_worst_o2c_unconditional=float(bad_o2c.mean()))
    print("\nclustering in the base itself:", json.dumps(cluster))
    OUT["blunt_clustering"] = cluster
    for fam in FAMILIES:
        E = L[fam]["exposure"].reindex(days).to_numpy()
        T = L[fam]["turn"].reindex(days).to_numpy()
        keys = list(L[fam]["exposure"].columns)
        cut = np.zeros_like(E, bool)
        cut[1:] = E[1:] < E[:-1] - .05
        mid = (T > .25) & ~weekly[:, None] & ~cut         # big trade off the weekly schedule without an exposure cut
        per = {}
        for name, ev in (("cut", cut), ("midweek_trade", mid)):
            for scope, mask in (("all", np.ones(n, bool)), ("after_0810", late)):
                e = ev & mask[:, None]
                e[0] = False
                prev = np.r_[False, bad_c2c[:-1]]
                k = int(e.sum())
                dates = e.any(axis=1)
                fam_rows.append(dict(
                    family=fam, event=name, scope=scope, ledgers_with_event=int(e.any(axis=0).sum()), events=k,
                    unique_dates=int(dates.sum()),
                    mean_base_same_day_o2c=float((e * o2c[:, None]).sum() / k) if k else np.nan,
                    mean_base_prev_day_c2c=float((e * np.r_[0, c2c[:-1]][:, None]).sum() / k) if k else np.nan,
                    share_same_day_worst_decile_o2c=float((e & bad_o2c[:, None]).sum() / k) if k else np.nan,
                    share_prev_day_worst_decile=float((e & prev[:, None]).sum() / k) if k else np.nan,
                    unique_dates_same_day_worst_decile_o2c=int((dates & bad_o2c).sum()),
                    worst_decile_o2c_rate_in_scope=float(bad_o2c[mask].mean()),
                    worst_decile_prev_rate_in_scope=float(prev[mask].mean()),
                    base_mean_o2c_all_days=float(o2c[mask].mean())))
            per[name] = ev
        # the worst-decile days on which most ledgers of the family cut: what did the base do the day before?
        hits = (per["cut"] & bad_o2c[:, None]).sum(axis=1)
        for j in np.argsort(-hits)[:4]:
            if hits[j]:
                date_rows.append(dict(family=fam, date=days[j], ledgers_cutting=int(hits[j]), base_o2c=float(o2c[j]),
                                      base_prev_c2c=float(c2c[j - 1]), base_prev2_c2c=float(c2c[j - 2]),
                                      base_dd_prev_close=float(base_usd[j - 1] / base_usd[:j].max() - 1)))
        order =st[fam].drop(index="base", errors="ignore").sort_values("h2", ascending=False).index[:20]
        for rank, key in enumerate(order, 1):
            c = keys.index(key)
            e = per["cut"][:, c] | per["midweek_trade"][:, c]
            k = int(e.sum())
            same, prev = int((e & bad_o2c).sum()), int((e[1:] & bad_c2c[:-1]).sum())
            top_rows.append(dict(family=fam, rank=rank, ledger=key, h2=st[fam].at[key, "h2"], h2_expo=st[fam].at[key, "h2_expo"],
                                 action_days=k, exposure_cuts=int(per["cut"][:, c].sum()),
                                 on_worst_decile_o2c_day=same, expected_by_chance=round(.1 * k, 1),
                                 p_binom=binom_tail(same, k, float(bad_o2c.mean())), after_worst_decile_day=prev,
                                 mean_o2c_action=float(o2c[e].mean()) if k else np.nan,
                                 mean_prev_c2c=float(c2c[np.flatnonzero(e) - 1].mean()) if k else np.nan))
    fam_table, top = pd.DataFrame(fam_rows), pd.DataFrame(top_rows)
    show("2) blunt test by family (base return on the day a rule acted at the open vs the day before)", fam_table, 4)
    agg = top.groupby("family").agg(rules=("ledger", "size"), action_days=("action_days", "sum"),
                                    on_worst_decile=("on_worst_decile_o2c_day", "sum"), expected=("expected_by_chance", "sum"),
                                    after_worst_decile=("after_worst_decile_day", "sum"), min_p=("p_binom", "min"),
                                    rules_p_below_05=("p_binom", lambda s: int((s < .05).sum())),
                                    mean_o2c_action=("mean_o2c_action", "mean"), mean_prev_c2c=("mean_prev_c2c", "mean"))
    show("2a) worst-decile days with the most exposure cuts per family", pd.DataFrame(date_rows), 4)
    OUT["blunt_dates"] = date_rows
    show("2b) 20 best H2 rules per family, summed", agg, 4)
    show("2c) 20 best H2 rules per family, each", top.drop(columns=["rank"]), 4)
    print(f"worst-decile thresholds: c2c <= {np.quantile(c2c, .10):+.4f} ({int(bad_c2c.sum())} days), "
          f"open-to-close <= {np.quantile(o2c, .10):+.4f} ({int(bad_o2c.sum())} days)")
    OUT["blunt_family"] = fam_table.to_dict(orient="records")
    OUT["blunt_top20_sum"] = agg.reset_index().to_dict(orient="records")
    OUT["blunt_top20"] = top.to_dict(orient="records")


# ---------------------------------------------------------------------------
# 3) the lead's readings
# ---------------------------------------------------------------------------

EP = ["jul_fall", "aug_sep_rally", "octnov_fall", "h2", "sep_nov", "full"]


def reading_universe(ctx, L, st, plans, base_usd):
    days = ctx.days
    cfg = pd.read_csv(common.OUT / "universe_candidates.csv", index_col=0)
    t = st["universe"].join(cfg[["subfamily", "gate", "lookback", "n", "weight"]])
    u1a = t[t.subfamily.str.startswith("U1a")].copy()
    cols = ["jul_fall", "aug_sep_rally", "octnov_fall", "h2", "h2_mdd", "h2_expo", "h2_xs", "h2_xc", "jul_fall_xs", "jul_fall_xc",
            "octnov_fall_expo", "octnov_fall_xs", "octnov_fall_xc", "octnov_fall_d", "sep_nov_xs", "full", "full_mdd"]
    show("3i) base", t.loc[["base"], cols])
    u1a["L"] = u1a.lookback.astype(int)
    show("3i) U1a switch medians by lookback x N", u1a[u1a.gate != "always"].groupby(["L", "n"])[cols].median())
    show("3i) U1a medians by subfamily", u1a.groupby("subfamily")[cols].median())
    show("3i) U1a switch medians by gate", u1a[u1a.gate != "always"].groupby("gate")[cols].median())
    sw = u1a[u1a.gate != "always"]
    b = t.loc["base"]
    counts = dict(
        n_switch=len(sw), h2_xs_median=float(sw.h2_xs.median()), h2_expo_median=float(sw.h2_expo.median()),
        jul_better_than_base_by_5pts=int((sw.jul_fall > b.jul_fall + .05).sum()),
        octnov_better_than_base_by_5pts=int((sw.octnov_fall > b.octnov_fall + .05).sum()),
        octnov_better_than_base_at_all=int((sw.octnov_fall > b.octnov_fall).sum()),
        octnov_xs_positive=int((sw.octnov_fall_xs > 0).sum()), octnov_xs_median=float(sw.octnov_fall_xs.median()),
        octnov_xc_positive=int((sw.octnov_fall_xc > 0).sum()), octnov_xc_median=float(sw.octnov_fall_xc.median()),
        octnov_xc_above_5pts=int((sw.octnov_fall_xc > .05).sum()), octnov_worst=float(sw.octnov_fall.min()),
        octnov_d_median_L120=float(sw[sw.lookback.astype(int) == 120].octnov_fall_d.median()),
        octnov_d_median_L20=float(sw[sw.lookback.astype(int) == 20].octnov_fall_d.median()),
        jul_d_median=float(sw.jul_fall_d.median()), octnov_d_median=float(sw.octnov_fall_d.median()),
        rally_d_median=float(sw.aug_sep_rally_d.median()), h1_like_full_d_median=float(sw.full_d.median()),
        h2_xc_median=float(sw.h2_xc.median()),
        jul_xs_median=float(sw.jul_fall_xs.median()), rally_worse_than_base=int((sw.aug_sep_rally < b.aug_sep_rally).sum()),
        rally_median=float(sw.aug_sep_rally.median()), full_better_than_base=int((sw.full > b.full).sum()),
        full_better_and_mdd_better=int(((sw.full > b.full) & (sw.full_mdd > b.full_mdd)).sum()),
        octnov_best=sw.octnov_fall.idxmax(), octnov_best_value=float(sw.octnov_fall.max()),
        octnov_median=float(sw.octnov_fall.median()), jul_median=float(sw.jul_fall.median()))
    print("\n3i) U1a switch counts", json.dumps(counts, indent=1, default=float))
    # the same gate with cash instead of the non-semis basket
    same = []
    for gate, g in sw.groupby("gate"):
        cash = t.loc[f"u1c_f00_{gate}"]
        same.append(dict(gate=gate, n=len(g), **{f"{e}_u1a_med": g[e].median() for e in ("jul_fall", "aug_sep_rally", "octnov_fall", "h2", "full")},
                         **{f"{e}_cash": cash[e] for e in ("jul_fall", "aug_sep_rally", "octnov_fall", "h2", "full")}))
    same = pd.DataFrame(same).set_index("gate")
    show("3i) U1a (median of 12 baskets) vs plain cash exit under the same gate", same)
    show("3i) every U1a variant", u1a.sort_values(["subfamily", "gate", "lookback", "n", "weight"])[
        ["gate", "jul_fall", "aug_sep_rally", "octnov_fall", "h2", "h2_expo", "h2_xs", "full", "full_mdd"]])
    OUT["u1a_counts"], OUT["u1a_by_gate"] = counts, sw.groupby("gate")[cols].median().reset_index().to_dict(orient="records")
    OUT["u1a_vs_cash_same_gate"] = same.reset_index().to_dict(orient="records")
    OUT["u1a_all"] = u1a[["gate", "jul_fall", "aug_sep_rally", "octnov_fall", "h2", "h2_expo", "h2_xs", "full", "full_mdd"]].reset_index().to_dict(orient="records")

    OUT["u1a_by_L_n"] = u1a[u1a.gate != "always"].groupby(["L", "n"])[cols].median().reset_index().to_dict(orient="records")
    for spec in ((120, 7, "time"), (20, 5, "equal")):
        _octnov_holdings(ctx, plans[spec], spec)


def _octnov_holdings(ctx, plan_picks, spec):
    """What u1a_L<..>_N<..>_<..>_SOXX20r1 held in 10/13-11/17 (own rebuild)."""
    days = ctx.days
    plan, picks = plan_picks
    name = "u1a_L%d_N%d_%s_SOXX20r1" % spec
    gate = ctx.gates["SOXX20r1"]
    usd, expo, opn_nav, held = account(switch_orders(ctx.weekly, plan, gate, ctx.opn), ctx.opn, ctx.close)
    ix = [i for i, d in enumerate(days) if "2026-10-13" <= d <= "2026-11-17"]
    close, opn = np.nan_to_num(ctx.close), np.nan_to_num(ctx.opn)
    pnl = np.zeros(len(ctx.tickers))
    wsum = np.zeros(len(ctx.tickers))
    for j in ix:
        pnl += held[j - 1] * (opn[j] - close[j - 1]) + held[j] * (close[j] - opn[j])
        wsum += held[j] * close[j] / usd[j]
    start = usd[ix[0] - 1]
    name_ret = ctx.close[ix[-1]] / ctx.close[ix[0] - 1] - 1
    held_tab = pd.DataFrame({"mean_weight": wsum / len(ix), "contribution": pnl / start, "name_return_window": name_ret,
                             "sector": [ctx.book.sector[k] for k in ctx.col]}, index=ctx.tickers)
    held_tab = held_tab[held_tab.mean_weight > 1e-6].sort_values("mean_weight", ascending=False)
    show(f"3i) {name} holdings 10/13-11/17 (gate off {int((~gate[ix]).sum())} of {len(ix)} sessions; "
         f"window return {usd[ix[-1]] / start - 1:+.4f}, contributions sum {pnl.sum() / start:+.4f})", held_tab, 4)
    weeks = [(days[j], "on" if gate[j] else "OFF", ",".join(picks[j]),
              ",".join(f"{ctx.tickers[c]}:{ctx.weekly[j][c]:.2f}" for c in np.flatnonzero(ctx.weekly[j] > 0)))
             for j in sorted(plan) if "2026-10-05" <= days[j] <= "2026-11-17"]
    print("weekly dates: gate, non-semis top-N (alt basket), base basket")
    for w in weeks:
        print("  ", *w, sep=" | ")
    spells = "".join("1" if g else "0" for g in gate[ix])
    print("SOXX20r1 gate by session 10/13-11/17:", spells)
    bh = shadow_weights(ctx.weekly, ctx.opn, len(days))
    semis = np.array([S.in_semis_budget(ctx.book.sector[k]) for k in ctx.col])
    print(f"base semis-budget weight (mean of opens) in window: {bh[ix][:, semis].sum(axis=1).mean():.3f}; "
          f"u1a semis weight: {np.mean([(held[j] * opn[j])[semis].sum() / opn_nav[j] for j in ix]):.3f}")
    OUT.setdefault("u1a_octnov", {})[name] = dict(holdings=held_tab.reset_index().to_dict(orient="records"), weeks=weeks,
                                                  gate=spells, window_return=float(usd[ix[-1]] / start - 1))


def reading_stops(ctx, L, st):
    days = ctx.days
    cfg = pd.read_csv(common.OUT / "stops_candidates.csv", index_col=0)
    t = st["stops"].join(cfg[["subfamily", "control"]])
    t = t[(t.control != 1) & (t.index != "base")]
    cols = ["h2", "h2_expo", "h2_xs", "h2_xc", "h2_d", "jul_fall_xc", "jul_fall_d", "octnov_fall_xc", "octnov_fall_d",
            "aug_sep_rally_d", "full", "full_d", "full_xc"]
    g = t.groupby("subfamily")
    med = g[cols].median()
    med.insert(0, "n", g.size())
    med["h2_xc_pos"] = g.h2_xc.apply(lambda s: int((s > 0).sum()))
    med["both_falls_xc_pos"] = g.apply(lambda d: int(((d.jul_fall_xc > 0) & (d.octnov_fall_xc > 0)).sum()))
    med["both_falls_xc_pos_and_full_ge_base"] = g.apply(
        lambda d: int(((d.jul_fall_xc > 0) & (d.octnov_fall_xc > 0) & (d.full_d >= 0)).sum()))
    med["h2_xc_best"] = g.h2_xc.max()
    show("3ii) stops: medians by subfamily (xs = vs exposure-ratio control, xc = ratio capped at 1, d = vs base)", med)
    OUT["stops_subfamily"] = med.reset_index().to_dict(orient="records")
    stock = t[t.subfamily.str.contains("stock|intraday")]
    best = stock.sort_values("h2_xc", ascending=False).head(10)[["subfamily", "jul_fall", "octnov_fall"] + cols]
    show("3ii) per-stock stops with the largest H2 excess (capped control)", best)
    OUT["stops_best_stock"] = best.reset_index().to_dict(orient="records")
    rot = stock[stock.subfamily.str.contains("trailing_rotate")].copy()
    rot["x"] = rot.index.str.extract(r"(?:trail|loss)(\d+)", expand=False).astype(int).to_numpy()
    rot["after"] = rot.index.str.split("_").str[2]
    show("3ii) trailing stops with rotation, median by stop distance", rot.groupby("x")[cols].median())
    show("3ii) trailing stops with rotation, median by re-entry rule", rot.groupby("after")[cols].median())
    OUT["stops_trailing_rotate_by_x"] = rot.groupby("x")[cols].median().reset_index().to_dict(orient="records")
    OUT["stops_trailing_rotate_by_after"] = rot.groupby("after")[cols].median().reset_index().to_dict(orient="records")

    ev = pd.read_csv(CACHE / "stops_events.csv")
    ev = ev[ev.name != "ACCOUNT"]
    ev["kind"] = ev.subfamily.str.split("_").str[0]
    rows = []
    for (kind, period), g in list(ev.groupby(["kind", "period"])) + [((k, "all"), g) for k, g in ev.groupby("kind")]:
        d = g.dropna(subset=["fwd10"])
        u = d.drop_duplicates(["name", "day"])
        rows.append(dict(kind=kind, period=period, stops=len(g), with_fwd10=len(d), share_lower_10=float((d.fwd10 < 0).mean()),
                         mean_fwd10=float(d.fwd10.mean()), median_fwd10=float(d.fwd10.median()),
                         distinct_name_days=len(u), share_lower_10_distinct=float((u.fwd10 < 0).mean()),
                         no_fwd10_yet=int(g.fwd10.isna().sum())))
    evt = pd.DataFrame(rows)
    # reference: every name-day the base held, same horizon, open j -> close j+10
    held = shadow_weights(ctx.weekly, ctx.opn, len(days)) > 0
    ref = []
    for period, (a, b) in (("h1", ("2026-01-01", "2026-06-30")), ("jul_aug", ("2026-07-01", "2026-08-31")),
                           ("sep_nov", ("2026-09-01", "2026-12-31")), ("all", ("2026-01-01", "2026-12-31"))):
        v = [ctx.close[j + 10, c] / ctx.opn[j, c] - 1 for j, d in enumerate(days) if a <= d <= b and j + 10 < len(days)
             for c in np.flatnonzero(held[j])]
        ref.append(dict(period=period, base_held_name_days=len(v), share_lower_10=float(np.mean(np.array(v) < 0)),
                        mean_fwd10=float(np.mean(v))))
    ref = pd.DataFrame(ref)
    show("3ii) stops_events.csv: was the stopped name lower 10 sessions after the fill?", evt)
    show("3ii) reference: all name-days held by the base (open j -> close j+10)", ref)
    OUT["stops_events"], OUT["stops_events_reference"] = evt.to_dict(orient="records"), ref.to_dict(orient="records")


def reading_filters(ctx, L, st, base_usd, base_expo):
    days = ctx.days
    cfg = pd.read_csv(common.OUT / "filters_candidates.csv", index_col=0)
    t = st["filters"].join(cfg[["subfamily", "filter", "mode", "gate", "n_filters", "avail_h1", "avail_ja", "dup_codex"]])
    t = t[~t.subfamily.isin(["base", "ref_gate"])]
    cols = ["h2", "h2_expo", "h2_xs", "h2_xc", "h2_d", "jul_fall_xc", "octnov_fall_xc", "sep_nov_xc", "full_d"]
    for label, keys in (("subfamily", ["subfamily"]), ("subfamily x gate", ["subfamily", "gate"]), ("subfamily x mode", ["subfamily", "mode"])):
        g = t.groupby(keys)
        med = g[cols].median()
        med.insert(0, "n", g.size())
        med["h2_xc_pos"] = g.h2_xc.apply(lambda s: int((s > 0).sum()))
        med["both_falls_xc_pos"] = g.apply(lambda d: int(((d.jul_fall_xc > 0) & (d.octnov_fall_xc > 0)).sum()))
        show(f"3iii) filters: medians by {label} (xs = exposure-ratio control, xc = ratio capped at 1, d = vs base)", med)
        OUT[f"filters_by_{label.replace(' x ', '_')}"] = med.reset_index().to_dict(orient="records")
    hi = t[(t.jul_fall_expo > .70) & (t.octnov_fall_expo > .70) & (t.h2_expo > .70)].copy()
    hi["worst_of_two"] = hi[["jul_fall_xc", "octnov_fall_xc"]].min(axis=1)
    both = hi[hi.worst_of_two > 0].sort_values("worst_of_two", ascending=False)
    strong = both[both.worst_of_two > .05]
    summary = dict(candidates=len(t), exposure_above_70_in_both_falls_and_h2=len(hi), positive_excess_both_falls=len(both),
                   both_above_5pts=len(strong), both_positive_and_full_year_ge_base=int((both.full_d >= 0).sum()),
                   both_above_5pts_and_full_year_ge_base=int((strong.full_d >= 0).sum()),
                   gates_of_the_positive=both.gate.value_counts().to_dict(), modes_of_the_positive=both["mode"].value_counts().to_dict(),
                   full_d_median_of_the_positive=float(both.full_d.median()),
                   rally_d_median_of_the_positive=float(both.aug_sep_rally_d.median()),
                   h2_xc_median_all=float(t.h2_xc.median()), h2_xc_pos_all=int((t.h2_xc > 0).sum()),
                   ungated=len(t[t.gate == "none"]), ungated_h2_xc_median=float(t[t.gate == "none"].h2_xc.median()),
                   ungated_h2_xc_pos=int((t[t.gate == "none"].h2_xc > 0).sum()),
                   ungated_full_d_median=float(t[t.gate == "none"].full_d.median()),
                   ungated_full_ge_base=int((t[t.gate == "none"].full_d >= 0).sum()))
    print("\n3iii) filters with positive excess (capped control) in BOTH falls at exposure > 70%:",
          json.dumps(summary, indent=1, ensure_ascii=False))
    view = ["subfamily", "mode", "gate", "jul_fall", "jul_fall_expo", "jul_fall_xc", "octnov_fall", "octnov_fall_expo", "octnov_fall_xc",
            "aug_sep_rally_d", "h2", "h2_expo", "h2_d", "full", "full_d", "full_mdd"]
    show("3iii) the list, best worst-of-two first (top 45)", both[view].head(45))
    keep = strong[strong.full_d >= 0]
    show("3iii) above 5 points in both falls AND full year not below the base", keep[view])
    # single filters, ungated: does the result hold in both rebalancing modes?
    single = t[(t.n_filters == 1) & (t.gate == "none") & t["mode"].isin(["wk", "rot"])]
    piv = single.pivot(index="filter", columns="mode", values=["jul_fall_xc", "octnov_fall_xc", "full_d", "h2_expo"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv["worst_of_four"] = piv[["jul_fall_xc_wk", "jul_fall_xc_rot", "octnov_fall_xc_wk", "octnov_fall_xc_rot"]].min(axis=1)
    piv = piv.sort_values("worst_of_four", ascending=False)
    show("3iii) single ungated filters, weekly and daily-rotate modes side by side (top 20 by the worst of the four numbers)", piv.head(20))
    robust = piv[piv.worst_of_four > .05]
    print(f"single filters with > 5 points in both falls in BOTH modes: {len(robust)} of {len(piv)}: {list(robust.index)}; "
          f"of these full year >= base in both modes: {list(robust[(robust.full_d_wk >= 0) & (robust.full_d_rot >= 0)].index)}")
    OUT["filters_both_summary"] = summary
    OUT["filters_both"] = both[view].reset_index().to_dict(orient="records")
    OUT["filters_both_strong_full_ok"] = keep[view].reset_index().to_dict(orient="records")
    OUT["filters_single_two_modes"] = piv.reset_index().to_dict(orient="records")

    # placebo: random weekly eligibility with the same engine. Does knocking out random names do the same?
    rng = np.random.default_rng(20261119)
    sched = sorted(ctx.weekly)
    week_of = np.searchsorted(sched, np.arange(len(days)), side="right") - 1
    rows = []
    for share in (.5, .7, .85):
        for _ in range(150):
            draw = rng.random((len(sched) + 1, len(ctx.tickers))) < share
            elig = draw[week_of + 1]
            plan = study.plan(ctx.book, days, ctx.rankings, ctx.col, elig, "week", "rotate")
            usd, expo, _, _ = account(plan, ctx.opn, ctx.close)
            r = dict(pass_share=share)
            for name in ("jul_fall", "octnov_fall", "aug_sep_rally", "h2", "full"):
                a, b = EPISODES[name]
                ret = ep_ret(usd, days, a, b)[0]
                _, ex, capped, plain = control(base_usd, base_expo, expo, days, a, b)
                r[name], r[f"{name}_expo"], r[f"{name}_xc"], r[f"{name}_d"] = ret, ex, ret - capped, ret - plain
            rows.append(r)
    pl = pd.DataFrame(rows)
    pl["worst_of_two"] = pl[["jul_fall_xc", "octnov_fall_xc"]].min(axis=1)
    g = pl.groupby("pass_share")
    ptab = pd.DataFrame({
        "n": g.size(), "h2_expo_med": g.h2_expo.median(), "jul_xc_med": g.jul_fall_xc.median(), "octnov_xc_med": g.octnov_fall_xc.median(),
        "h2_d_med": g.h2_d.median(), "full_d_med": g.full_d.median(),
        "both_falls_pos": g.worst_of_two.apply(lambda s: float((s > 0).mean())),
        "both_falls_above_5pts": g.worst_of_two.apply(lambda s: float((s > .05).mean())),
        "both_above_5pts_and_full_ge_base": g.apply(lambda d: float(((d.worst_of_two > .05) & (d.full_d >= 0)).mean())),
        "worst_of_two_p90": g.worst_of_two.quantile(.9), "worst_of_two_max": g.worst_of_two.max()})
    show("3iii) placebo: 150 random weekly eligibility matrices per pass share (weekly-rotate mode, ungated)", ptab)
    real = t[(t.n_filters == 1) & (t.gate == "none") & (t["mode"] == "wk")]
    real = real.assign(worst_of_two=real[["jul_fall_xc", "octnov_fall_xc"]].min(axis=1))
    comp = dict(real_single_wk_ungated=len(real), real_both_falls_pos=float((real.worst_of_two > 0).mean()),
                real_both_above_5pts=float((real.worst_of_two > .05).mean()),
                real_both_above_5pts_and_full_ge_base=float(((real.worst_of_two > .05) & (real.full_d >= 0)).mean()),
                real_h2_d_median=float(real.h2_d.median()), real_full_d_median=float(real.full_d.median()),
                placebo_all_both_falls_pos=float((pl.worst_of_two > 0).mean()),
                placebo_all_both_above_5pts=float((pl.worst_of_two > .05).mean()),
                placebo_all_h2_d_median=float(pl.h2_d.median()), placebo_all_full_d_median=float(pl.full_d.median()))
    for key in ("subbreadth_20.wk.none", "vol20_med.wk.none", "hi20_10.wk.none", "above20_50.wk.none", "higher_low10.wk.none"):
        if key in t.index:
            w = min(t.at[key, "jul_fall_xc"], t.at[key, "octnov_fall_xc"])
            comp[f"{key}_worst_of_two"] = float(w)
            comp[f"{key}_placebo_share_at_least_as_good"] = float((pl.worst_of_two >= w).mean())
            comp[f"{key}_placebo_share_as_good_in_falls_and_full_year"] = float(
                ((pl.worst_of_two >= w) & (pl.full_d >= t.at[key, "full_d"])).mean())
    print("3iii) real single weekly filters vs placebo:", json.dumps(comp, indent=1))
    OUT["filters_placebo"] = ptab.reset_index().to_dict(orient="records")
    OUT["filters_placebo_compare"] = comp


def reading_tabsig(ctx, L, st, base_usd):
    days = ctx.days
    cfg = pd.read_csv(common.OUT / "tabsig_candidates.csv", index_col=0)
    t = st["tabsig"].join(cfg[["signal", "level", "source", "alarm_sessions"]]).drop(index="base")
    cols = ["h2", "h2_expo", "h2_xs", "h2_xc", "jul_fall_xc", "octnov_fall_xc", "sep_nov_xc", "full_d"]
    g = t.groupby("source")
    med = g[cols].median()
    med.insert(0, "n", g.size())
    med["h2_xs_pos"] = g.h2_xc.apply(lambda s: int((s > 0).sum()))
    med.loc["ALL"] = [len(t)] + list(t[cols].median()) + [int((t.h2_xc > 0).sum())]
    active = t[t.h2_expo < st["tabsig"].at["base", "h2_expo"] - .01]
    med.loc["ALL_active_in_h2"] = [len(active)] + list(active[cols].median()) + [int((active.h2_xc > 0).sum())]
    show("3iv) tabsig: medians by source", med)
    fund = t[t.source == "fund"]
    best = fund.sort_values("h2_xc", ascending=False).head(8)[["signal", "level", "alarm_sessions", "jul_fall", "octnov_fall"] + cols]
    show("3iv) fund-manager gates with the largest H2 excess", best)
    both = fund[(fund.jul_fall_xc > 0) & (fund.octnov_fall_xc > 0)]
    show(f"3iv) fund gates with positive excess in both falls: {len(both)} of {len(fund)}",
         both[["signal", "level", "alarm_sessions", "jul_fall", "jul_fall_expo", "octnov_fall", "octnov_fall_expo", "aug_sep_rally_d"] + cols])
    OUT["tabsig_fund_both_falls_detail"] = both[["jul_fall", "jul_fall_expo", "jul_fall_xc", "octnov_fall", "octnov_fall_expo",
                                                  "octnov_fall_xc", "h2", "h2_expo", "h2_xc", "full_d"]].reset_index().to_dict(orient="records")
    OUT["tabsig_by_source"] = med.reset_index().to_dict(orient="records")
    OUT["tabsig_fund_best"] = best.reset_index().to_dict(orient="records")
    OUT["tabsig_fund_both_falls"] = list(both.index)

    # what the TIME fund did around the two peaks
    pdf = pd.read_csv(CACHE / "regime_fund_time_by_pdf.csv", dtype={"pdf": str}).set_index("pdf")
    rows, flows = [], []
    for peak, end in (("2026-06-22", "2026-07-29"), ("2026-09-22", "2026-11-17")):
        p, e = days.index(peak), days.index(end)
        first = lambda lvl: next((k for k in range(p, e + 1) if base_usd[k] / base_usd[p] - 1 <= -lvl), None)
        marks = {"P-20": p - 20, "P-10": p - 10, "P-5": p - 5, "P": p, "d5": first(.05), "d10": first(.10), "d10+5": first(.10) + 5,
                 "d10+10": first(.10) + 10, "end": e}
        for name, k in marks.items():
            k = min(k, len(days) - 1)
            i = pdf.index.searchsorted(days[k], side="right") - 1          # latest PDF dated on or before that US session
            r = pdf.iloc[i]
            rows.append(dict(peak=peak, mark=name, session=days[k], pdf=pdf.index[i], base_dd=base_usd[k] / base_usd[p] - 1,
                             us_w=r.us_w, semis_w=r.semis_w, semis_us_w=r.semis_us_w, memory_w=r.memory_w, sw_w=r.sw_w,
                             cash_w=r.cash_w, fut_w=r.fut_w))
        for a, b in (("P-20", "P-10"), ("P-10", "P"), ("P", "d5"), ("d5", "d10"), ("d10", "d10+10"), ("d10+10", "end")):
            seg = pdf[(pdf.index > days[marks[a]]) & (pdf.index <= days[min(marks[b], len(days) - 1)])]
            flows.append(dict(peak=peak, span=f"{a}..{b}", pdfs=len(seg), net_us_pp=seg.net_us.sum(), net_semis_pp=seg.net_semis.sum(),
                              net_us_semis_pp=seg.net_us_semis.sum(), net_all_pp=seg.net_all.sum(),
                              sells=int(seg.sells.sum()), buys=int(seg.buys.sum())))
    levels, flows = pd.DataFrame(rows), pd.DataFrame(flows)
    show("3iv) TIME fund weights (% of NAV) around the two peaks; the weight also moves with prices", levels)
    show("3iv) TIME fund real trades (quantity-based, NAV %p summed over PDFs in the span; negative = net selling)", flows)
    ordinary = pdf[(pdf.index >= "2026-01-15")]
    print("for scale, 10-PDF net real trades over 2026: net_us_10 quantiles",
          ordinary.net_us_10.quantile([.1, .25, .5, .75, .9]).round(2).to_dict(),
          "net_semis_10", ordinary.net_semis_10.quantile([.1, .25, .5, .75, .9]).round(2).to_dict())
    print("us_w range 2026:", round(pdf.us_w.min(), 1), round(pdf.us_w.max(), 1), "semis_w:", round(pdf.semis_w.min(), 1),
          round(pdf.semis_w.max(), 1), "cash_w:", round(pdf.cash_w.min(), 2), round(pdf.cash_w.max(), 2))
    OUT["time_fund_levels"], OUT["time_fund_flows"] = levels.to_dict(orient="records"), flows.to_dict(orient="records")
    et = pd.read_csv(CACHE / "regime_event_table.csv")
    sel = et[et.decline.isin(["2026-06-22", "2026-09-22"]) & et.signal.isin(
        ["time_us_w", "time_semis_w", "time_semis_us_w", "time_cash_w", "time_net_us_10", "time_net_semis_10", "time_net_us_semis_10",
         "time_semis_w_chg10", "time_us_w_chg10"])]
    show("3iv) regime_event_table.csv rows for the TIME fund (values at P-10, P-5, P, first -5%, first -10%, trough)",
         sel.drop(columns=["trough", "depth"]))
    OUT["time_fund_event_table"] = sel.to_dict(orient="records")


def reading_alarms(ctx, base_usd, base_open, base_expo):
    days = ctx.days
    al = pd.read_csv(CACHE / "regime_alarm_stats.csv")
    al = al[al.threshold.notna()].copy()
    al["early_both"] = (al["dd_at_fill_06-22"] > -.10) & (al["dd_at_fill_09-22"] > -.10)
    cols = ["signal", "direction", "threshold", "on_sessions", "onsets_total", "false_onsets", "false_onsets_outside_any_8pct_decline",
            "false_on_sessions", "first_on_06-22", "sessions_vs_peak_06-22", "dd_at_fill_06-22", "first_on_09-22",
            "sessions_vs_peak_09-22", "dd_at_fill_09-22", "base_logret_next_on_outside", "base_logret_next_on_inside"]
    few = al[al.false_onsets <= 5].sort_values(["false_onsets", "false_on_sessions"])
    show(f"4) regime_alarm_stats.csv: {len(al)} series with a threshold; false onsets <= 5: {len(few)}; "
         f"filled before -10% in both declines: {int(al.early_both.sum())}; both: {int((al.early_both & (al.false_onsets <= 5)).sum())}",
         few[cols], 4)
    cand = al[al.early_both & (al.false_onsets <= 5)]
    show("4) candidates: fill above -10% in both declines and <= 5 false onsets (threshold is read off these two declines)", cand[cols], 4)
    quiet = al[al.false_on_sessions <= 10]
    show("4) the other kind: series in alarm on <= 10 sessions outside the two declines", quiet[cols], 4)
    cand = pd.concat([cand, quiet[~quiet.signal.isin(cand.signal)]])
    # recheck the candidates from the saved panel with an own count and a leave-one-out threshold
    panel = pd.read_csv(CACHE / "regime_signal_panel.csv", index_col=0)
    panel.index = panel.index.astype(str)
    hells = (("2026-06-22", "2026-07-29"), ("2026-09-22", "2026-11-17"))
    inside = np.zeros(len(days), bool)
    marks = []
    for a, b in hells:
        p, e = days.index(a), days.index(b)
        inside[p:e + 1] = True
        d10 = next(k for k in range(p, e + 1) if base_usd[k] / base_usd[p] - 1 <= -.10)
        marks.append((p, d10, e))
    rows = []
    for _, r in cand.iterrows():
        sign = 1 if r.direction == "high" else -1
        v = panel[r.signal].reindex(days).to_numpy(float) * sign
        peaks = [np.nanmax(v[p:d + 1]) for p, d, _ in marks]
        row = dict(signal=r.signal, threshold_file=r.threshold, threshold_own=min(peaks) * sign)
        on = np.nan_to_num(v, nan=-np.inf) >= min(peaks) - 1e-12
        onset = on & ~np.r_[False, on[:-1]]
        row["false_onsets_own"], row["false_on_sessions_own"] = int((onset & ~inside).sum()), int((on & ~inside).sum())
        row["false_onset_dates"] = ",".join(d for d, f in zip(days, onset & ~inside) if f)
        row["share_of_other_sessions_in_alarm"] = float((on & ~inside).sum() / (~inside).sum())
        frac = np.ones(len(days))
        frac[1:] = np.where(on[:-1], 0.0, 1.0)
        u, e, _, _ = account(gated_orders(ctx.weekly, frac, ctx.opn), ctx.opn, ctx.close)
        for name in ("jul_fall", "octnov_fall", "h2", "full"):
            a, b = EPISODES[name]
            ret = ep_ret(u, days, a, b)[0]
            _, ex, capped, plain = control(base_usd, base_expo, e, days, a, b)
            row[f"exit_{name}"], row[f"exit_{name}_xc"] = ret, ret - capped
        row["exit_full_expo"] = float(e.mean())
        for (p, d10, e_), tag in zip(marks, ("06-22", "09-22")):
            row[f"on_at_peak_{tag}"] = bool(on[p])
            row[f"on_sessions_in_10_before_peak_{tag}"] = int(on[p - 10:p].sum())
            k0 = next((k for k in range(p - 10, p + 1) if on[k]), None)
            row[f"base_gain_first_on_to_peak_{tag}"] = float(base_usd[p] / base_open[k0 + 1] - 1) if k0 is not None and k0 < p else np.nan
            k = next((k for k in range(p, d10 + 1) if onset[k] or (k == p and on[k])), None)
            row[f"first_on_from_peak_{tag}"] = days[k] if k is not None else None
            row[f"dd_at_that_fill_{tag}"] = float(base_open[k + 1] / base_usd[p] - 1) if k is not None else np.nan
        # leave-one-out: threshold from one decline only, applied to the other
        for i, other in ((0, 1), (1, 0)):
            on_i = np.nan_to_num(v, nan=-np.inf) >= peaks[i] - 1e-12
            p, d, e = marks[other]
            first = next((k for k in range(max(p - 10, 0), e + 1) if on_i[k]), None)
            onset_i = on_i & ~np.r_[False, on_i[:-1]]
            tag = hells[other][0][5:]
            row[f"thr_from_other_first_on_{tag}"] = days[first] if first is not None else None
            row[f"thr_from_other_dd_at_fill_{tag}"] = (base_open[first + 1] / base_usd[p] - 1) if first is not None and first + 1 < len(days) else np.nan
            row[f"thr_from_other_false_onsets_{tag}"] = int((onset_i & ~inside).sum())
        rows.append(row)
    own = pd.DataFrame(rows)
    if len(own):
        show("4) own recount of the candidates and leave-one-decline-out thresholds", own, 4)
    seps = {}
    for mark in ("P", "d5", "d10"):
        s = pd.read_csv(CACHE / f"regime_separation_{mark}.csv")
        ok = s[s.separates]
        seps[mark] = dict(series=len(s), separates=len(ok), wrong_way=int(s.separates_wrong_way.sum()),
                          names=ok.signal.tolist())
        show(f"4) regime_separation_{mark}.csv: series whose value at {mark} separates the two long declines from the V-shaped ones "
             f"({len(ok)} of {len(s)}; wrong way {int(s.separates_wrong_way.sum())})",
             ok[["signal", "n_v", "hell_values", "v_min", "v_max", "v_median"]], 4)
    OUT["alarm_candidates"] = cand[cols].to_dict(orient="records")
    OUT["alarm_candidates_own"] = own.to_dict(orient="records")
    OUT["alarm_counts"] = dict(series=len(al), false_onsets_le5=len(few), early_both=int(al.early_both.sum()),
                               both=int((al.early_both & (al.false_onsets <= 5)).sum()))
    OUT["separation"] = seps
    return cand, own


def main():
    ctx = common.context()
    days = ctx.days
    L = load_ledgers()
    for fam in FAMILIES:
        assert list(L[fam]["usd"].index) == days, fam
    base = study.replay(days, ctx.opn, ctx.close, ctx.fx, ctx.weekly, ctx.gates["none"])
    base_usd, base_expo = base.usd.to_numpy(), base.exposure.to_numpy()
    print(f"sessions {len(days)}; ledgers " + ", ".join(f"{f} {L[f]['usd'].shape[1]}" for f in FAMILIES))
    for fam in FAMILIES:
        d = rel_diff(L[fam]["usd"]["base"].to_numpy(), base_usd)
        print(f"  {fam}: saved base ledger vs study.replay base, max rel diff {d:.2e}")
    head = {name: ep_ret(base_usd, days, a, b) for name, (a, b) in EPISODES.items()}
    print("base (usd return, mdd):", {k: (round(v[0], 4), round(v[1], 4)) for k, v in head.items()})
    OUT["base"] = {k: dict(ret=v[0], mdd=v[1]) for k, v in head.items()}

    base_open, plans, _ = rebuilds(ctx, L, base_usd)
    assert rel_diff(base_open, ctx.baseline.value) < 1e-9
    st = {fam: stats(L[fam], base_usd, base_expo, days) for fam in FAMILIES}
    blunt(ctx, L, base_usd, base_open, st)
    reading_universe(ctx, L, st, plans, base_usd)
    reading_stops(ctx, L, st)
    reading_filters(ctx, L, st, base_usd, base_expo)
    reading_tabsig(ctx, L, st, base_usd)
    reading_alarms(ctx, base_usd, base_open, base_expo)
    (CACHE / "verify_modules.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("\nwrote", CACHE / "verify_modules.json")


if __name__ == "__main__":
    main()
