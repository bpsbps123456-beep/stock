"""Sparse seven-position replay of the desk's two independently drifting portfolios.

The holding entering an auction earns the overnight return. The post-auction
holding earns the intraday return. Cash fractions and stock weights are invariant
to the scale of a fee-paying account, so all SOXX gates can share each shadow book.
Run this file to check the daily results against the independent full-share ledger.
"""
from pathlib import Path
import sys
from typing import NamedTuple

import numpy as np
from numba import njit

PRIOR = Path(__file__).resolve().parents[1] / "finviz_recipes_20261120"
sys.path.insert(0, str(PRIOR))
from research import TimingEngine, load_inputs, market_gate
from timing import review_days
from baselines import desk_targets


class Sleeve(NamedTuple):
    before_ids: np.ndarray
    before_weights: np.ndarray
    after_ids: np.ndarray
    after_weights: np.ndarray
    overnight: np.ndarray
    intraday: np.ndarray
    exposure: np.ndarray
    traded: np.ndarray


class Result(NamedTuple):
    nav: np.ndarray
    exposure: np.ndarray
    turn: np.ndarray
    always_nav: np.ndarray
    always_exposure: np.ndarray
    always_turn: np.ndarray


@njit(cache=True)
def _shadow(opn, close, tradable, rows, ids, weights, settle_closed):
    days = len(opn)
    slots = ids.shape[1]
    before = np.full((days, slots), -1, np.int32)
    after = before.copy()
    old_weights = np.zeros((days, slots))
    new_weights = old_weights.copy()
    overnight = np.ones(days)
    intraday = np.ones(days)
    exposure = np.zeros(days)
    traded = np.zeros(days, np.bool_)
    closing_weights = np.zeros(slots)
    cash = 1.
    row = 0
    for j in range(days):
        opening = cash
        if j:
            for k in range(slots):
                s = after[j-1, k]
                if s >= 0:
                    o = opn[j, s]
                    if not np.isfinite(o) or o <= 0:
                        raise ValueError("Missing shadow opening mark")
                    before[j, k] = s
                    old_weights[j, k] = closing_weights[k] * o / close[j-1, s]
                    opening += old_weights[j, k]
        overnight[j] = opening
        old_weights[j] /= opening
        if row < len(rows) and j == rows[row]:
            traded[j] = True
            for k in range(slots):
                s = ids[row, k]
                if s >= 0 and (not settle_closed or tradable[j, s]):
                    after[j, k] = s
                    new_weights[j, k] = weights[row, k]
            row += 1
        else:
            for k in range(slots):
                s = before[j, k]
                if s >= 0:
                    if settle_closed and not tradable[j, s]:
                        traded[j] = True
                    else:
                        after[j, k] = s
                        new_weights[j, k] = old_weights[j, k]
        exposure[j] = new_weights[j].sum()
        reserve = 1. - exposure[j]
        ending = reserve
        for k in range(slots):
            s = after[j, k]
            if s < 0:
                closing_weights[k] = 0.
            else:
                o, c = opn[j, s], close[j, s]
                if not (np.isfinite(o) and o > 0 and np.isfinite(c) and c > 0):
                    raise ValueError("Missing executable or closing mark for a held stock")
                closing_weights[k] = new_weights[j, k] * c / o
                ending += closing_weights[k]
        intraday[j] = ending
        closing_weights /= ending
        cash = reserve / ending
    return Sleeve(before, old_weights, after, new_weights, overnight, intraday, exposure, traded)


@njit(cache=True)
def _auction(before_ids, before, after_ids, after, settled, fee):
    """Turnover at gross NAV and net financing factor for the same opening trades."""
    overlap = np.zeros(len(after))
    sells = 0.
    turnover = 0.
    for k in range(len(before)):
        s = before_ids[k]
        if s < 0:
            continue
        retained = False
        for m in range(len(after)):
            if after_ids[m] == s:
                overlap[m] = before[k]
                retained = True
                break
        if not retained:
            turnover += before[k]
            if not settled[s]:
                sells += before[k]
    for m in range(len(after)):
        turnover += abs(after[m] - overlap[m])
    post = 1.
    if fee > 0:
        for _ in range(10):
            chargeable = sells
            for m in range(len(after)):
                s = after_ids[m]
                if s >= 0 and not settled[s]:
                    chargeable += abs(post * after[m] - overlap[m])
            updated = 1. - fee * chargeable
            if abs(updated - post) <= 1e-15:
                break
            post = updated
    return turnover, post


@njit(cache=True)
def _transitions(previous, active, cross, settlements, fee):
    days = len(active.overnight)
    gross = previous.overnight * active.intraday
    turns = np.zeros(days)
    for j in range(days):
        if cross or active.traded[j]:
            turn, factor = _auction(previous.before_ids[j], previous.before_weights[j],
                                    active.after_ids[j], active.after_weights[j], settlements[j], fee)
            gross[j] *= factor
            turns[j] = turn
    return gross, turns


@njit(cache=True)
def _eligible(ranks, rows, issuers, tradable):
    selected = np.full(ranks.shape, -1, np.int32)
    for i in range(len(rows)):
        j = rows[i]
        for k in range(ranks.shape[1]):
            s = ranks[i, k]
            if s < 0:
                continue
            duplicate = False
            for m in range(k):
                prior = ranks[i, m]
                if prior >= 0 and issuers[j, prior] == issuers[j, s]:
                    duplicate = True
                    break
            if not duplicate and tradable[j, s]:
                selected[i, k] = s
    return selected


@njit(cache=True)
def _price_gates(gates, fx, growth, turnover, desk_exposure, screen_exposure):
    count, days = gates.shape
    nav = np.empty((count, days))
    exposure = np.empty((count, days))
    turns = np.empty((count, days))
    for g in range(count):
        value = 1.
        previous = int(gates[g, 0])
        for j in range(days):
            active = int(gates[g, j])
            value *= growth[previous, active, j]
            nav[g, j] = value * fx[j]
            exposure[g, j] = desk_exposure[j] if active else screen_exposure[j]
            turns[g, j] = turnover[previous, active, j]
            previous = active
    return nav, exposure, turns


class FastEngine:
    def __init__(self, src):
        self.src = src
        self.issuers = np.unique(src.issuers, return_inverse=True)[1].reshape(src.issuers.shape).astype(np.int32)
        day_count = len(src.days)
        self.reviews = {name: np.array(review_days(src.days, np.ones(day_count, bool), name), int)
                        for name in ("monthly1", "monthly1_11", "weekly1")}
        plan = desk_targets(src)
        rows = np.array(sorted(plan), int)
        names = [np.flatnonzero(plan[j] > 0) for j in rows]
        slots = max((len(n) for n in names), default=0)
        ids = np.full((len(rows), slots), -1, np.int32)
        weights = np.zeros_like(ids, dtype=float)
        for r, (j, n) in enumerate(zip(rows, names)):
            ids[r, :len(n)] = n
            weights[r, :len(n)] = plan[j][n]
        self.desk = _shadow(src.opn, src.close, src.tradable, rows, ids, weights, False)
        self.desk_costs = {}

    def evaluate(self, picks, reviews, gates, fee=.001, delay=0):
        """KRW end-of-day NAV for every gate, and an independently held screen.

        Daily gates are True for the desk. A delay defers the originally selected
        screen by that many auctions. Change the input gates separately to delay
        market transitions. Each missing ranked slot is represented by -1.
        """
        src = self.src
        gates = np.atleast_2d(np.asarray(gates))
        days = len(src.days)
        if gates.dtype != np.bool_ or gates.shape[1] != days:
            raise ValueError("Supply boolean gates aligned to execution days")
        if not isinstance(delay, (int, np.integer)) or delay < 0 or not 0 <= fee < 1:
            raise ValueError("Invalid delay or fee")
        reviews = self.reviews[reviews] if isinstance(reviews, str) else np.asarray(reviews, int)
        if reviews.ndim != 1 or np.any(np.diff(reviews) <= 0) or np.any((reviews < 0) | (reviews >= days)):
            raise ValueError("Review days must be unique, ordered, and in the execution range")
        if isinstance(picks, dict):
            ranks = np.full((len(reviews), 7), -1, np.int32)
            for i, j in enumerate(reviews):
                row = picks[j][:7]
                ranks[i, :len(row)] = row
        else:
            ranks = np.asarray(picks)
            if ranks.ndim != 2 or not np.issubdtype(ranks.dtype, np.integer):
                raise ValueError("Supply a rectangular integer ranking")
            if len(ranks) == days:
                ranks = ranks[reviews]
            elif len(ranks) != len(reviews):
                raise ValueError("Ranks must cover all days or precisely the review days")
            ranks = ranks[:, :7]
        if np.any((ranks < -1) | (ranks >= len(src.symbols))):
            raise ValueError("Invalid security index in submitted rankings")
        valid = reviews + delay < days
        decision_rows = reviews[valid]
        ids = _eligible(ranks[valid].astype(np.int32), decision_rows, self.issuers, src.tradable)
        screen = _shadow(src.opn, src.close, src.tradable, decision_rows + delay,
                         ids, np.where(ids >= 0, 1/7, 0.), True)
        growth = np.empty((2, 2, days))
        turnover = np.empty_like(growth)
        if fee not in self.desk_costs:
            self.desk_costs[fee] = _transitions(self.desk, self.desk, False, src.settlements, fee)
        growth[1, 1], turnover[1, 1] = self.desk_costs[fee]
        growth[0, 0], turnover[0, 0] = _transitions(screen, screen, False, src.settlements, fee)
        growth[1, 0], turnover[1, 0] = _transitions(self.desk, screen, True, src.settlements, fee)
        growth[0, 1], turnover[0, 1] = _transitions(screen, self.desk, True, src.settlements, fee)
        nav, exposure, turns = _price_gates(gates, src.fx, growth, turnover, self.desk.exposure, screen.exposure)
        return Result(nav, exposure, turns, growth[0, 0].cumprod() * src.fx, screen.exposure, turnover[0, 0])


def verify():
    from time import perf_counter
    from compare import close_terminated, switched_targets
    from recipes import recipes

    src = load_inputs()
    fast = FastEngine(src)
    reference = market_gate(src)
    gates = np.array([reference, ~reference, np.arange(len(src.days)) % 2 == 0])
    canonical = TimingEngine(src, reference)
    designs = recipes(src)
    count = 0
    worst = 0.
    for design, schedule, delay in [("balance", "monthly1", 0), ("original", "weekly1", 1),
                                    ("balance", "monthly1_11", 0)]:
        ranks = designs[design]
        reviews = fast.reviews[schedule]
        picks = {j: ranks[j][:7] for j in reviews}
        if schedule == "monthly1_11":
            # Force the delisting/settlement and duplicate-rank branches.
            retired = np.argwhere(src.settlements)
            exercised = 0
            for day, symbol in retired:
                prior = reviews[reviews < day]
                if len(prior) and src.tradable[prior[-1], symbol] and src.membership[prior[-1], symbol]:
                    j = prior[-1]
                    picks[j] = [symbol, symbol, *picks[j][:5]]
                    exercised += 1
            assert exercised > 0
        ranks = [picks.get(j, []) for j in range(len(src.days))]
        monthly = canonical.targets(ranks, schedule)
        if delay:
            monthly = close_terminated({j+delay: monthly[j] for j in reviews if j+delay < len(src.days)},
                                       src.opn, src.tradable)
        paths = [canonical._price(switched_targets(src.days, src.opn, canonical.desk, monthly, g),
                                  detail=True, stress=True) for g in gates]
        always = canonical._price(monthly, detail=True, stress=True)
        for fee, field in [(0., "krw"), (.001, "krw10"), (.0025, "krw25")]:
            result = fast.evaluate(picks, schedule, gates, fee=fee, delay=delay)
            for curve, nav, exposure, turn in [
                    (always, result.always_nav, result.always_exposure, result.always_turn),
                    *[(path, result.nav[i], result.exposure[i], result.turn[i]) for i, path in enumerate(paths)]]:
                np.testing.assert_allclose(nav, curve[field], rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(exposure, curve.exposure, rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(turn, curve.turn, rtol=1e-11, atol=1e-12)
                worst = max(worst, np.max(np.abs(nav - curve[field])))
                count += len(nav)
    start = perf_counter()
    for _ in range(100):
        fast.evaluate(picks, schedule, gates)
    print(f"Verified {count} daily NAV, exposure and turnover observations (maximum NAV error {worst:.2e}).")
    print(f"Average time per basket across {len(gates)} gates: {(perf_counter()-start)*10:.2f} ms")


if __name__ == "__main__":
    verify()
