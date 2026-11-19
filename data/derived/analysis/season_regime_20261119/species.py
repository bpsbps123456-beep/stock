"""Species: independent, point-in-time portfolios that can each be held from any US open.

Each species is a target plan {session index: weight vector} over the 86 frozen book tickers plus
five ETFs. Weekly baskets rebalance on the first session with a ranking and then on the first session
of each ISO week, like the desk rule. Rankings use closes through the previous session only.

    MOM      the desk rule (120-session momentum top 7, TIME weights, caps) = base
    HALF     MOM at half size, rest USD cash
    REV      7 TIME US names with the worst 5-session return, equal weight, caps   (buy the losers)
    PULL     of the 14 strongest by 120-session momentum, the 7 worst over 5 sessions (buy leaders' dips)
    NONSEMI  MOM restricted to names outside the semiconductor budget
    EW       every TIME US name, equal weight
    SOXX, QQQ, SOXL, SOXS, PSQ   buy and hold the ETF
    CASH     USD cash
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CHECK = HERE.parent / "desk_defense_check_20261119"
if str(CHECK) not in sys.path:
    sys.path.insert(0, str(CHECK))

import common  # noqa: E402  (puts the repo root and Codex's study folder on sys.path)
import study  # noqa: E402
from desk import strategies as S  # noqa: E402

ETFS = ("SOXX", "QQQ", "SOXL", "SOXS", "PSQ")
NAMES = ("MOM", "HALF", "REV", "PULL", "NONSEMI", "EW", *ETFS, "CASH")


def extended_bars(ctx):
    """Open/close arrays [session, 86 tickers + 5 ETFs]; SOXX/QQQ from the book, the rest frozen."""
    frozen = pd.read_csv(CHECK / "frozen_etf" / "hedge_ohlc.csv")
    cols_o, cols_c = [], []
    for sym in ETFS:
        if sym in ctx.book.opn.columns:
            o, c = ctx.book.opn.loc[ctx.days, sym], ctx.book.cls.loc[ctx.days, sym]
        else:
            part = frozen[frozen.symbol == sym].set_index("date")
            o, c = part.Open.reindex(ctx.days), part.Close.reindex(ctx.days)
        if o.isna().any() or c.isna().any():
            raise ValueError(f"{sym} has missing bars")
        cols_o.append(o.to_numpy(float))
        cols_c.append(c.to_numpy(float))
    opn = np.column_stack([ctx.opn, *cols_o])
    close = np.column_stack([ctx.close, *cols_c])
    etf_col = {sym: ctx.opn.shape[1] + i for i, sym in enumerate(ETFS)}
    return opn, close, etf_col


def trailing_return(ctx, key, j, n):
    """Return over n sessions ending at the close before session j (adjusted closes)."""
    px = ctx.book.adj[ctx.book.ticker[key]].to_numpy(float)
    k = ctx.start + j
    a, b = px[k - 1 - n], px[k - 1]
    return b / a - 1 if np.isfinite(a) and np.isfinite(b) and a > 0 else np.nan


def weekly_plan(ctx, width, pick, caps=True):
    """Publish pick(j, ranked, previous) weights on the desk's weekly schedule."""
    targets, prev, started = {}, set(), False
    for j, (day, ranked) in enumerate(zip(ctx.days, ctx.rankings)):
        week = j == 0 or S.iso_week(day) != S.iso_week(ctx.days[j - 1])
        if not ranked or (started and not week):
            continue
        raw = pick(j, ranked, prev)
        if not raw:
            continue
        total = sum(raw.values())
        raw = {k: v / total for k, v in raw.items()}
        weights = S.project_caps(raw, ctx.book.sector) if caps else raw
        vec = np.zeros(width)
        for k, v in weights.items():
            vec[ctx.col[k]] = v
        targets[j] = vec
        prev, started = set(raw), True
    return targets


def build_plans(ctx, width, etf_col):
    def pad(plan, scale=1.0):
        out = {}
        for j, w in plan.items():
            v = np.zeros(width)
            v[:len(w)] = scale * w
            out[j] = v
        return out

    def rev(j, ranked, prev):
        scored = [(trailing_return(ctx, r["key"], j, 5), r["key"]) for r in ranked]
        scored = sorted((s, k) for s, k in scored if np.isfinite(s))
        return {k: 1.0 for _, k in scored[:7]}

    def pull(j, ranked, prev):
        top = sorted(ranked, key=lambda r: r["rank"])[:14]
        scored = [(trailing_return(ctx, r["key"], j, 5), r["key"]) for r in top]
        scored = sorted((s, k) for s, k in scored if np.isfinite(s))
        return {k: 1.0 for _, k in scored[:7]}

    def nonsemi(j, ranked, prev):
        pool = [r for r in ranked if not S.in_semis_budget(r["sector"])]
        keep = S.select(pool, prev)
        return {r["key"]: r["time_weight"] for r in pool if r["key"] in keep}

    def ew(j, ranked, prev):
        return {r["key"]: 1.0 for r in ranked}

    def hold(sym):
        v = np.zeros(width)
        v[etf_col[sym]] = 1.0
        return {0: v}

    plans = {"MOM": pad(ctx.weekly), "HALF": pad(ctx.weekly, .5),
             "REV": weekly_plan(ctx, width, rev), "PULL": weekly_plan(ctx, width, pull),
             "NONSEMI": weekly_plan(ctx, width, nonsemi), "EW": weekly_plan(ctx, width, ew, caps=False)}
    plans.update({sym: hold(sym) for sym in ETFS})
    plans["CASH"] = {}
    return plans


def multi_switch(opn, plans, choice):
    """Target schedule that holds plans[choice[j]] from the open of session j.

    Every plan keeps its own shadow shares and cash, so a switch buys the incoming plan's
    currently drifted basket at that open; on the active plan's own decision dates its new
    target is used. choice[j] must be decided from information through the close of j-1.
    """
    n_plans, width = len(plans), opn.shape[1]
    shares, cash = np.zeros((n_plans, width)), np.ones(n_plans)
    schedule, previous = {}, None
    for j in range(len(choice)):
        prices = opn[j]
        held = (shares != 0).any(axis=0)
        if not (np.isfinite(prices[held]) & (prices[held] > 0)).all():
            raise ValueError(f"A shadow basket holds a security without an open on session {j}")
        safe = np.where(np.isfinite(prices), prices, 0.0)
        values = shares * safe
        nav = values.sum(axis=1) + cash
        for i, plan in enumerate(plans):
            if j in plan:
                w = plan[j]
                values[i] = w * nav[i]
                shares[i] = np.divide(values[i], prices, out=np.zeros(width), where=w > 0)
                cash[i] = nav[i] - values[i].sum()
        a = int(choice[j])
        if j in plans[a]:
            schedule[j] = plans[a][j].copy()
        elif a != previous:
            schedule[j] = values[a] / nav[a]
        previous = a
    return schedule


class Lab:
    """Frozen inputs, species plans and their stand-alone ledgers."""

    def __init__(self):
        self.ctx = common.context()
        self.opn, self.close, self.etf_col = extended_bars(self.ctx)
        self.width = self.opn.shape[1]
        self.plans = build_plans(self.ctx, self.width, self.etf_col)
        self.names = list(NAMES)
        self.always = np.ones(len(self.ctx.days), dtype=bool)
        self.ledgers = {n: self.replay(self.plans[n]) for n in self.names}
        base = self.ledgers["MOM"]
        np.testing.assert_allclose(base.usd.to_numpy(), self.ctx.baseline.close, rtol=1e-10)

    def replay(self, targets):
        c = self.ctx
        return study.replay(c.days, self.opn, self.close, c.fx, targets, self.always)

    def follow(self, choice, names):
        """Ledger of holding species names[choice[j]] from each open."""
        plans = [self.plans[n] for n in names]
        return self.replay(multi_switch(self.opn, plans, choice))


if __name__ == "__main__":
    lab = Lab()
    days = lab.ctx.days
    for i, n in enumerate(lab.names):
        alone = lab.ledgers[n]
        same = lab.follow(np.zeros(len(days), dtype=int), [n])
        np.testing.assert_allclose(same.usd.to_numpy(), alone.usd.to_numpy(), rtol=1e-12)
    # A switch must equal holding each species' own path on its sessions (open-to-open accounting).
    half = len(days) // 2
    mixed = lab.follow(np.r_[np.zeros(half), np.ones(len(days) - half)].astype(int), ["MOM", "REV"])
    np.testing.assert_allclose(mixed.usd.to_numpy()[:half], lab.ledgers["MOM"].usd.to_numpy()[:half], rtol=1e-12)
    ratio = mixed.usd.to_numpy()[half:] / lab.ledgers["REV"].usd.to_numpy()[half:]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-10)  # after the switch it is REV's own basket
    print("species:", {n: round(float(lab.ledgers[n].usd.iloc[-1] - 1), 4) for n in lab.names})
    print("mixed MOM->REV at", days[half], "final", round(float(mixed.usd.iloc[-1] - 1), 4))
