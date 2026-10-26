"""Grid runner: base strategies x regime overlays -> compact metrics tensor (.npy).

Costs: COST per side on every traded unit of equity (assumption, see report).
Overlay rule: exposure e[t] is decided at close t from data up to t-1 and applies to
(t, t+1]. When exposure drops the shadow portfolio is sold pro rata into USD cash.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import data as D
import engine as E

COST = 0.001  # 0.10% per side (scenario, not data)
STOP = 0.15
NS = (1, 2, 3, 5, 7, 10)
WEIGHTINGS = ("EQ", "FW", "IV")
REBS = ("D", "W", "2W", "M")
EXITS = ("none", "drop", "stop15")
OVERLAYS = ("none", "QQQ>MA20", "QQQ>MA50", "QQQ>MA100", "SMH>MA20", "SMH>MA50",
            "QQQ20>50", "VIX<25", "BREADTH>50", "FUNDCASH", "SELF>MA10", "SELF>MA20",
            "SELFDD15", "QQQ>MA50/half", "SELF>MA20/half")
METRICS = (["total", "sharpe", "mdd", "vol", "is_ret", "oos_ret", "is_sharpe", "oos_sharpe",
            "turnover", "exposure", "holdings"]
           + [f"m{i}" for i in range(1, 10)] + [f"bs{i}" for i in range(10)]
           + [f"bq{i}" for i in range(10)] + ["reg_up", "reg_box", "reg_down"])
BATCH = 4096
OUT = D.CACHE / "grid"


def param_table(names):
    rows = []
    for sid, (src, _pre, _name) in enumerate(names):
        for N in NS:
            for wi in range(len(WEIGHTINGS)):
                for buf in (0, N):
                    for ri, rb in enumerate(REBS):
                        for ex in ((0,) if rb == "D" else (0, 1, 2)):
                            rows.append((sid, N, wi, buf, ri, ex, E.SOURCES.index(src)))
    return np.array(rows, np.int64)


def market_overlays(m, d):
    i0, _ = m.feat["ref_idx"]
    T = len(m.days)
    Q, S, V = m.ref["QQQ"], m.ref["SMH"], m.ref["^VIX"]
    P = m.feat["P"]

    def above_ma(x, L):
        return np.array([1.0 if x[i0 + t - 1] > np.mean(x[i0 + t - L:i0 + t]) else 0.0
                         for t in range(T)])

    out = {"none": np.ones(T)}
    out["QQQ>MA20"], out["QQQ>MA50"], out["QQQ>MA100"] = (above_ma(Q, L) for L in (20, 50, 100))
    out["SMH>MA20"], out["SMH>MA50"] = above_ma(S, 20), above_ma(S, 50)
    out["QQQ20>50"] = np.array([1.0 if np.mean(Q[i0 + t - 20:i0 + t]) > np.mean(Q[i0 + t - 50:i0 + t])
                                else 0.0 for t in range(T)])
    out["VIX<25"] = np.array([1.0 if V[i0 + t - 1] < 25 else 0.0 for t in range(T)])

    held_u = m.held["U"]
    breadth = np.ones(T)
    row = 0
    for t in range(T):
        if m.dec_row[t] >= 0:
            row = m.dec_row[t]
        i = i0 + t
        names = np.where(held_u[row])[0]
        last = P[i - 1, names]
        ma = np.nanmean(P[i - 20:i, names], axis=0)
        ok = ~np.isnan(last) & ~np.isnan(ma)
        breadth[t] = 1.0 if np.mean(last[ok] > ma[ok]) > 0.5 else 0.0
    out["BREADTH>50"] = breadth

    cash = np.array([float(d["panels"]["T"]["metrics"][p]["cash_weight_pct"])
                     + float(d["panels"]["K"]["metrics"][p]["cash_weight_pct"]) for p in d["pdf_dates"]])
    pdf_pos = {p: i for i, p in enumerate(d["pdf_dates"])}
    fund = np.ones(T)
    row = 0
    for t in range(T):
        if m.dec_row[t] >= 0:
            row = m.dec_row[t]
        k = pdf_pos[m.dec_pdf[row]]
        hist = cash[max(0, k - 10):k]
        fund[t] = 1.0 if (len(hist) == 0 or cash[k] <= hist.mean()) else 0.0
    out["FUNDCASH"] = fund
    out["QQQ>MA50/half"] = np.where(out["QQQ>MA50"] > 0, 1.0, 0.5)
    return out


def self_overlay(eq, kind):
    """Equity-curve overlays from the base strategy's own equity up to t-1."""
    Sn, T = eq.shape
    prev = np.concatenate([np.ones((Sn, 1)), eq[:, :-1]], axis=1)  # eq[t-1], eq[-1]=1
    if kind in ("SELF>MA10", "SELF>MA20", "SELF>MA20/half"):
        L = 10 if kind == "SELF>MA10" else 20
        csum = np.cumsum(prev, axis=1)
        idx = np.arange(T)
        lo = np.maximum(0, idx - L + 1)
        window = idx - lo + 1
        lagged = np.where(lo > 0, csum[:, np.maximum(lo - 1, 0)], 0.0)
        ma = (csum - lagged) / window
        on = (prev >= ma).astype(float)
        if kind.endswith("/half"):
            return np.where(on > 0, 1.0, 0.5)
        return on
    if kind == "SELFDD15":
        peak = np.maximum.accumulate(prev, axis=1)
        return (prev / peak - 1 > -0.15).astype(float)
    raise ValueError(kind)


def expost_regimes(m):
    i0, _ = m.feat["ref_idx"]
    Q = m.ref["QQQ"]
    n = len(Q)
    lab = np.zeros(len(m.days), np.int64)
    for t in range(len(m.days)):
        i = i0 + t
        r = Q[min(n - 1, i + 10)] / Q[max(0, i - 10)] - 1
        lab[t] = 0 if r > 0.03 else (2 if r < -0.03 else 1)
    return lab


class Context:
    def __init__(self, m):
        self.T = len(m.days)
        self.is_end = max(i for i, x in enumerate(m.days) if x <= E.IS_END)
        months = sorted({x[:7] for x in m.days})
        self.month_idx = np.array([months.index(x[:7]) for x in m.days])
        self.months = months
        edges = np.linspace(1, self.T, 11).round().astype(int)
        self.block_idx = np.full(self.T, -1)
        for b in range(10):
            self.block_idx[edges[b]:edges[b + 1]] = b
        self.regime = expost_regimes(m)


def metrics(net_krw, eprev, eff_turn, nh, ctx):
    Sn, T = net_krw.shape
    out = np.empty((Sn, len(METRICS)), np.float32)
    lr = np.log1p(net_krw)
    cum = np.cumsum(lr, axis=1)
    out[:, 0] = np.expm1(cum[:, -1])
    r = net_krw[:, 1:]
    sd = r.std(axis=1)
    out[:, 1] = np.where(sd > 0, r.mean(axis=1) / np.where(sd > 0, sd, 1) * np.sqrt(252), 0)
    out[:, 2] = np.expm1((cum - np.maximum.accumulate(np.maximum(cum, 0), axis=1)).min(axis=1))
    out[:, 3] = sd * np.sqrt(252)
    k = ctx.is_end
    out[:, 4] = np.expm1(lr[:, :k + 1].sum(axis=1))
    out[:, 5] = np.expm1(lr[:, k + 1:].sum(axis=1))
    for col, seg in ((6, r[:, :k]), (7, r[:, k:])):
        s = seg.std(axis=1)
        out[:, col] = np.where(s > 0, seg.mean(axis=1) / np.where(s > 0, s, 1) * np.sqrt(252), 0)
    out[:, 8] = eff_turn.sum(axis=1)
    out[:, 9] = eprev[:, 1:].mean(axis=1) if eprev.ndim == 2 else eprev[1:].mean()
    out[:, 10] = nh[:, 1:].mean(axis=1)
    for mi in range(9):
        out[:, 11 + mi] = lr[:, ctx.month_idx == mi].sum(axis=1)
    for b in range(10):
        sel = ctx.block_idx == b
        out[:, 20 + b] = lr[:, sel].sum(axis=1)
        out[:, 30 + b] = (lr[:, sel] ** 2).sum(axis=1)
    for g in range(3):
        out[:, 40 + g] = lr[:, ctx.regime == g].sum(axis=1)
    return out


def run(cost=COST, limit=None, exec_mode="close", tag="", exclude=(), delay=0):
    OUT.mkdir(parents=True, exist_ok=True)
    d = D.load()
    m = E.build_market(d, exec_mode, exclude)
    names, stack = E.build_scores(m)
    stack = E.apply_delay(m, stack, delay)
    params = param_table(names)
    if limit:
        params = params[:limit]
    ctx = Context(m)
    mk = market_overlays(m, d)
    reb_stack = np.stack([m.reb[r] for r in REBS])
    fw = np.stack([m.fw[s] for s in E.SOURCES])
    held = np.stack([m.held[s] for s in E.SOURCES])
    iv = np.nan_to_num(m.iv, nan=0.0, posinf=0.0)
    T = ctx.T
    n = len(params)
    result = np.lib.format.open_memmap(OUT / f"metrics{tag}.npy", mode="w+", dtype=np.float32,
                                       shape=(n, len(OVERLAYS), len(METRICS)))
    started = time.time()
    for b0 in range(0, n, BATCH):
        P = params[b0:b0 + BATCH]
        Sn = len(P)
        gross = np.zeros((Sn, T))
        turn = np.zeros((Sn, T))
        nh = np.zeros((Sn, T), np.int64)
        E.run_batch(m.R, m.dec_row, reb_stack, stack, fw, iv, held, P, STOP, gross, turn, nh)
        eq = np.cumprod(1 + gross - cost * turn, axis=1)
        for o, name in enumerate(OVERLAYS):
            e = self_overlay(eq, name) if name.startswith("SELF") else np.broadcast_to(mk[name], (Sn, T))
            eprev = np.concatenate([np.zeros((Sn, 1)), e[:, :-1]], axis=1)
            eff = np.minimum(eprev, e) * turn + np.abs(e - eprev)
            net = eprev * gross - cost * eff
            net_krw = (1 + net) * (1 + m.fxr) - 1
            result[b0:b0 + Sn, o] = metrics(net_krw, eprev, eff, nh, ctx)
        done = b0 + Sn
        rate = done / (time.time() - started)
        print(f"{done}/{n} base strategies  {rate:.0f}/s  eta {(n - done) / rate:.0f}s", flush=True)
    result.flush()
    np.save(OUT / "params.npy", params)
    if tag:
        print(f"done in {time.time() - started:.0f}s -> {OUT} ({tag})")
        return
    meta = {"score_names": names, "overlays": OVERLAYS, "metrics": METRICS, "cost": cost,
            "stop": STOP, "NS": NS, "weightings": WEIGHTINGS, "rebs": REBS, "exits": EXITS,
            "sources": E.SOURCES, "days": m.days, "months": ctx.months,
            "is_end": E.IS_END, "bt": [E.BT_START, E.BT_END],
            "block_edges": np.linspace(1, T, 11).round().astype(int).tolist(),
            "regime_days": np.bincount(ctx.regime, minlength=3).tolist()}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"done in {time.time() - started:.0f}s -> {OUT}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cost", type=float, default=COST)
    ap.add_argument("--exec", default="close", choices=("close", "open"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--delay", type=int, default=0)
    a = ap.parse_args()
    run(cost=a.cost, limit=a.limit, exec_mode=a.exec, tag=a.tag,
        exclude=tuple(x for x in a.exclude.split(',') if x), delay=a.delay)
