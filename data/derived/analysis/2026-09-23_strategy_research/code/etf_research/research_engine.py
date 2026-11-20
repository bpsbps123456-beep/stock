"""Point-in-time, cash-funded research simulator. No brokerage/order integration.

All scores use a snapshot strictly before execution date and prices through the
previous date. Rebalance after earning today's return; costs use drifted holdings.
The grid is deliberately fixed before the retrospective holdout is evaluated.
"""
from __future__ import annotations

from datetime import date
from itertools import product
import numpy as np
from numba import njit, prange

FAMILIES = (
    "conviction", "net_flow", "buy_only", "joint_buy", "joint_hold",
    "new_entry", "repeated_buy", "avoid_sell", "divergence", "buy_momentum",
    "buy_dip", "holding_momentum", "sold_reversal",
)
SIZES = (1, 2, 3, 5, 7, 10)
MIXES = (0.0, 0.25, 0.5, 0.75, 1.0)  # TIME fraction of combined scores
GATES = ("none", "positive_momentum", "above_ma20")
WEIGHTINGS = ("equal", "conviction", "inverse_volatility")
REGIMES = ("always", "trend_half", "trend_cash", "manager_cash")
FREQUENCIES = ("daily", "weekly", "monthly")


def percentile(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average-rank percentiles; missing/ineligible values remain zero."""
    out = np.zeros_like(values, dtype=np.float64)
    for t in range(len(values)):
        ix = np.flatnonzero(mask[t] & np.isfinite(values[t]))
        if not len(ix):
            continue
        x = values[t, ix]
        # Average ties rather than using ticker order as a fictitious signal.
        order = np.argsort(x, kind="stable")
        ranks = np.empty(len(x), dtype=float)
        i = 0
        while i < len(x):
            j = i + 1
            while j < len(x) and x[order[j]] == x[order[i]]:
                j += 1
            ranks[order[i:j]] = (i + j + 1) / (2 * len(x))
            i = j
        out[t, ix] = ranks
    return out


def price_features(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """20-observation features known before each execution date; no warmup padding."""
    t_count, n = prices.shape
    mom = np.full((t_count, n), np.nan)
    ma = np.full_like(mom, np.nan)
    vol = np.full_like(mom, np.nan)
    for t in range(21, t_count):
        hist = prices[t-21:t]
        ok = np.all(np.isfinite(hist) & (hist > 0), axis=0)
        mom[t, ok] = hist[-1, ok] / hist[0, ok] - 1
        ma[t, ok] = hist[-1, ok] / np.mean(hist[-20:, ok], axis=0) - 1
        r = hist[1:, ok] / hist[:-1, ok] - 1
        vol[t, ok] = np.std(r, axis=0, ddof=1)
    return mom, ma, vol


def build_rankings(arrays: dict, extra_lag: int = 0) -> dict:
    dates = arrays["dates"].astype(str)
    sd = arrays["snapshot_dates"].astype(str)
    prices = arrays["prices"].astype(float)
    t_count, n = prices.shape
    # Strictly earlier date, plus optional extra global valuation-session delay.
    decision_dates = dates[np.maximum(0, np.arange(t_count) - extra_lag)]
    si = np.searchsorted(sd, decision_dates, side="left") - 1
    has = si >= 0
    safe = np.maximum(si, 0)
    weights = arrays["weights"][safe].astype(float)
    weights[~has] = 0
    ff = arrays["flow_pp"][safe].astype(float)
    kk = arrays["flow_kind"][safe].astype(float)
    ff[~has] = np.nan
    kk[~has] = np.nan
    cash = arrays["cash"][safe].astype(float)
    cash[~has] = 1.0
    previous_prices = np.vstack([np.full(n, np.nan), prices[:-1]])
    if extra_lag:
        previous_prices = previous_prices[np.maximum(0, np.arange(t_count)-extra_lag)]
    priced = np.isfinite(previous_prices) & (previous_prices > 0)
    mom, ma, vol = price_features(prices)
    if extra_lag:
        idx = np.maximum(0, np.arange(t_count)-extra_lag)
        mom, ma, vol = mom[idx], ma[idx], vol[idx]
    union = (weights[:, 0] + weights[:, 1] > 0) & priced
    mr = percentile(mom, union)
    # Prior 20-date QQQ local USD trend, removing FX from KRW QQQ benchmark.
    q = arrays.get("qqq_usd")
    if q is None:
        q = arrays["benchmarks"][:, 2]  # explicitly recorded KRW QQQ fallback
    qm, qa, _ = price_features(np.asarray(q).reshape(-1, 1))
    state = np.zeros(t_count, dtype=np.int8)  # 0 unknown,1 up,2 down,3 range
    state[(qm[:, 0] > 0.03) & (qa[:, 0] > 0)] = 1
    state[(qm[:, 0] < -0.03) & (qa[:, 0] < 0)] = 2
    state[np.isfinite(qm[:, 0]) & (state == 0)] = 3
    if extra_lag:
        state = state[np.maximum(0, np.arange(t_count)-extra_lag)]
    exposure = np.ones((t_count, 4), dtype=np.float64)
    exposure[state == 2, 1] = 0.5
    exposure[state == 2, 2] = 0.0
    exposure[:, 3] = np.clip(1 - np.mean(cash, axis=1), 0, 1)
    ranks, conviction, inverse_vol, rank_configs = [], [], [], []
    # Consecutive daily buys accumulated in each window, available at snapshot time.
    daily_kind = arrays["flow_kind"][:, 0]
    lookbacks = [1, 3, 5, 10, 20, 40]
    for lb_i, mix in product(range(6), MIXES):
        w = mix*weights[:, 0] + (1-mix)*weights[:, 1]
        held = (w > 0) & priced
        f0, f1 = ff[:, lb_i, 0], ff[:, lb_i, 1]
        k0, k1 = kk[:, lb_i, 0], kk[:, lb_i, 1]
        if mix == 1:
            flow, valid_flow = f0, np.isfinite(f0)
        elif mix == 0:
            flow, valid_flow = f1, np.isfinite(f1)
        else:
            flow = mix*f0 + (1-mix)*f1
            valid_flow = np.isfinite(f0) & np.isfinite(f1)
        wr = percentile(w, held)
        fr = percentile(flow, held & valid_flow)
        buy = ((k0 > 0) & (mix > 0)) | ((k1 > 0) & (mix < 1))
        sell = ((k0 < 0) & (mix > 0)) | ((k1 < 0) & (mix < 1))
        count = np.zeros_like(w)
        lb = lookbacks[lb_i]
        for t, s in enumerate(si):
            if s >= lb:
                base = s-lb
                if 'flow_base_dates' in arrays:
                    bd = arrays['flow_base_dates'][s, lb_i, 0]
                    base = int(np.searchsorted(sd, bd)) if bd else s
                d = daily_kind[base+1:s+1]
                count[t] = mix*np.sum(d[:, 0] > 0, axis=0)+(1-mix)*np.sum(d[:, 1] > 0, axis=0)
        cr = percentile(count, held)
        families = [
            (wr, held),
            (fr, held & valid_flow),
            (fr, held & valid_flow & buy & (flow > 0)),
            (fr + 0.1*wr, held & valid_flow & (k0 > 0) & (k1 > 0)),
            (wr, held & (weights[:, 0] > 0) & (weights[:, 1] > 0)),
            (wr + 0.1*fr, held & valid_flow & (((k0 == 2) & (mix > 0)) | ((k1 == 2) & (mix < 1)))),
            (cr + 0.1*fr, held & valid_flow & (count >= 2) & (flow > 0)),
            (wr + 0.1*fr, held & valid_flow & ~sell),
            (fr, held & valid_flow & buy & (((k0 > 0) & (k1 < 0)) | ((k1 > 0) & (k0 < 0)))),
            (0.5*fr + 0.5*mr, held & valid_flow & buy & np.isfinite(mom)),
            (0.5*fr + 0.5*(1-mr), held & valid_flow & buy & np.isfinite(mom)),
            (0.5*wr + 0.5*mr, held & np.isfinite(mom)),
            (wr + 0.1*mr, held & valid_flow & sell & np.isfinite(mom)),
        ]
        for fi, gate in product(range(len(FAMILIES)), range(len(GATES))):
            score, eligible = families[fi]
            eligible = eligible.copy()
            if gate == 1:
                eligible &= mom > 0
            elif gate == 2:
                eligible &= ma > 0
            order = np.argsort(-np.where(eligible, score, -np.inf), axis=1, kind="stable")[:, :10]
            eligible_top = np.take_along_axis(eligible, order, axis=1)
            order = np.where(eligible_top, order, -1).astype(np.int16)
            ix = np.maximum(order, 0)
            cw = np.take_along_axis(w, ix, axis=1)
            iv = np.take_along_axis(vol, ix, axis=1)
            iv = np.divide(1.0, np.maximum(iv, 0.001), out=np.zeros_like(iv), where=np.isfinite(iv))
            ranks.append(order)
            conviction.append(np.where(order >= 0, cw, 0).astype(np.float32))
            inverse_vol.append(np.where(order >= 0, iv, 0).astype(np.float32))
            rank_configs.append({"family": FAMILIES[fi], "lookback": lb, "time_mix": mix, "gate": GATES[gate]})
    schedule = np.zeros((t_count, 3), dtype=np.bool_)
    periods = [(date.fromisoformat(d).isocalendar()[:2], d[:7]) for d in dates]
    for t in range(t_count):
        schedule[t, 0] = True
        schedule[t, 1] = t == 0 or periods[t][0] != periods[t-1][0]
        schedule[t, 2] = t == 0 or periods[t][1] != periods[t-1][1]
    return {"ranks": np.array(ranks), "conviction": np.array(conviction),
            "inverse_vol": np.array(inverse_vol), "rank_configs": rank_configs,
            "exposure": exposure, "schedule": schedule, "regime_state": state,
            "signal_snapshot_index": si}


def config_grid(rank_configs: list[dict]) -> tuple[np.ndarray, list[dict]]:
    rows, configs = [], []
    for rid, size, freq, weighting, regime in product(range(len(rank_configs)), SIZES, range(3), range(3), range(4)):
        rows.append((rid, size, freq, weighting, regime))
        configs.append({**rank_configs[rid], "max_stocks": size,
                        "rebalance": FREQUENCIES[freq], "weighting": WEIGHTINGS[weighting],
                        "risk_rule": REGIMES[regime]})
    return np.array(rows, dtype=np.int32), configs


@njit(cache=True)
def _simulate_one(params, ranks, conviction, inverse_vol, exposure, schedule,
                  price_returns, tradable, stale, cost, start, end, detail):
    rid, size, freq, weighting, regime = params
    idx = np.full(10, -1, dtype=np.int32)
    val = np.zeros(10)
    cash = 1.0
    out = np.zeros(end-start, dtype=np.float64)
    turnover = np.zeros(end-start)
    counts = np.zeros(end-start)
    cashpath = np.zeros(end-start)
    holdings = np.full((end-start, 10), -1, dtype=np.int32) if detail else np.zeros((1, 1), dtype=np.int32)
    allocations = np.zeros((end-start, 10)) if detail else np.zeros((1, 1))
    initialized = False
    for t in range(start, end):
        before = cash + np.sum(val)
        for j in range(10):
            if idx[j] >= 0:
                if stale[t, idx[j]]:
                    out[t-start:] = np.nan
                    return out, turnover, counts, cashpath, holdings, allocations
                val[j] *= 1 + price_returns[t, idx[j]]
        nav = cash + np.sum(val)
        # Initial entry may occur once the first eligible signal exists, even if
        # the first calendar week/month began before any disclosed snapshot.
        do_trade = schedule[t, freq] or not initialized
        if do_trade:
            frozen_idx = np.full(10, -1, dtype=np.int32)
            frozen_val = np.zeros(10)
            nf, locked = 0, 0.0
            for j in range(10):
                if idx[j] >= 0 and not tradable[t, idx[j]]:
                    frozen_idx[nf] = idx[j]
                    frozen_val[nf] = val[j]
                    locked += val[j]
                    nf += 1
            target_idx = np.full(10, -1, dtype=np.int32)
            target_w = np.zeros(10)
            nt = 0
            for k in range(ranks.shape[2]):
                s = ranks[rid, t, k]
                if s < 0 or nt >= max(0, size-nf):
                    break
                if not tradable[t, s]:
                    continue
                b = 1.0
                if weighting == 1:
                    b = conviction[rid, t, k]
                elif weighting == 2:
                    b = inverse_vol[rid, t, k]
                if b <= 0:
                    continue
                target_idx[nt], target_w[nt] = s, b
                nt += 1
            sw = np.sum(target_w)
            if sw > 0:
                target_w /= sw
            invested = exposure[t, regime] if nt > 0 else 0.0
            # Solve costs and desired dollars jointly. This charges both legs,
            # funds fees from NAV, and never borrows cash at full exposure.
            post = nav
            traded = 0.0
            wanted = np.zeros(10)
            for _ in range(12):
                budget = max(0.0, post*invested-locked)
                wanted = target_w*budget
                traded = 0.0
                for j in range(10):
                    if idx[j] >= 0 and tradable[t, idx[j]]:
                        target = 0.0
                        for k in range(nt):
                            if target_idx[k] == idx[j]:
                                target = wanted[k]
                        traded += abs(target-val[j])
                for k in range(nt):
                    found = False
                    for j in range(10):
                        if idx[j] == target_idx[k]:
                            found = True
                    if not found:
                        traded += wanted[k]
                post = nav - cost*traded
            idx[:] = -1
            val[:] = 0.0
            for k in range(nf):
                idx[k], val[k] = frozen_idx[k], frozen_val[k]
            for k in range(nt):
                if wanted[k] > 1e-14:
                    idx[nf+k], val[nf+k] = target_idx[k], wanted[k]
            cash = nav - cost*traded - np.sum(val)
            if cash < -1e-9:
                out[t-start:] = np.nan
                return out, turnover, counts, cashpath, holdings, allocations
            cash = max(0.0, cash)
            turnover[t-start] = traded/nav if nav > 0 else 0
            if nt > 0:
                initialized = True
        after = cash + np.sum(val)
        out[t-start] = after/before - 1
        counts[t-start] = np.sum(idx >= 0)
        cashpath[t-start] = cash/after
        if detail:
            holdings[t-start] = idx
            allocations[t-start] = val/after
    return out, turnover, counts, cashpath, holdings, allocations


@njit(cache=True, parallel=True)
def simulate_grid(params, ranks, conviction, inverse_vol, exposure, schedule,
                  price_returns, tradable, stale, cost):
    m, t_count = len(params), len(price_returns)
    daily = np.empty((m, t_count), dtype=np.float32)
    stats = np.zeros((m, 3), dtype=np.float64)
    for i in prange(m):
        r, turn, counts, cash, _, _ = _simulate_one(
            params[i], ranks, conviction, inverse_vol, exposure, schedule,
            price_returns, tradable, stale, cost, 0, t_count, False)
        daily[i] = r
        stats[i, 0] = np.sum(turn)
        stats[i, 1] = np.max(counts)
        stats[i, 2] = np.mean(cash)
    return daily, stats


def simulation_inputs(arrays: dict, ranking: dict) -> tuple:
    p = arrays["prices"]
    ret = np.zeros_like(p, dtype=np.float64)
    valid = np.isfinite(p[1:]) & np.isfinite(p[:-1]) & (p[:-1] > 0)
    np.divide(p[1:], p[:-1], out=ret[1:], where=valid)
    ret[1:] = np.where(valid, ret[1:]-1, 0.0)
    # Long missing-price intervals cannot be assumed to have zero return.
    stale = arrays.get("price_age_days", np.zeros_like(p)) > 10
    stale |= arrays.get("fx_age_days", np.zeros_like(p)) > 5
    stale |= ~np.isfinite(p) | (p <= 0)
    stale[1:] |= ~np.isfinite(p[:-1]) | (p[:-1] <= 0)
    return (ranking["ranks"], ranking["conviction"], ranking["inverse_vol"],
            ranking["exposure"], ranking["schedule"], ret,
            (arrays["tradable"] & np.isfinite(p) & (p > 0)).astype(np.bool_), stale.astype(np.bool_))


def simulate_detail(params, arrays, ranking, cost=0.0015, start=0, end=None) -> dict:
    end = len(arrays["dates"]) if end is None else end
    r, turnover, counts, cash, holdings, allocations = _simulate_one(
        np.array(params, dtype=np.int32), *simulation_inputs(arrays, ranking), cost,
        start, end, True)
    return {"returns": r, "turnover": turnover, "counts": counts, "cash": cash,
            "holdings": holdings, "allocations": allocations}
