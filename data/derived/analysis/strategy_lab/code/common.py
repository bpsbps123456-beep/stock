"""Shared loaders and helpers for the analysis scripts."""

from __future__ import annotations

import json

import numpy as np

import data as D
import engine as E
import grid as G

GRID = D.CACHE / "grid"


def load_grid():
    meta = json.loads((GRID / "meta.json").read_text(encoding="utf-8"))
    M = np.load(GRID / "metrics.npy", mmap_mode="r")
    P = np.load(GRID / "params.npy")
    meta["score_names"] = [tuple(x) for x in meta["score_names"]]
    return M, P, meta


def col(meta, name):
    return meta["metrics"].index(name)


def label(meta, p, o=None):
    src, pre, score = meta["score_names"][p[0]]
    parts = [f"{src}/{pre}/{score}", f"N{p[1]}", meta["weightings"][p[2]],
             f"buf{p[3]}", meta["rebs"][p[4]], meta["exits"][p[5]]]
    if o is not None:
        parts.append(meta["overlays"][o])
    return " ".join(parts)


def score_family(name):
    for prefix in ("WMOM", "FMOM", "WFLOW", "FLOW", "MOM", "RADJ", "HIGH", "NEW", "DW", "CONS"):
        if name.startswith(prefix):
            return prefix
    return name


def simulate(m, stack, param, cost=G.COST, overlay=None):
    """Daily net KRW returns for one base param (optionally with a market overlay array)."""
    reb_stack = np.stack([m.reb[r] for r in G.REBS])
    fw = np.stack([m.fw[s] for s in E.SOURCES])
    held = np.stack([m.held[s] for s in E.SOURCES])
    iv = np.nan_to_num(m.iv, nan=0.0, posinf=0.0)
    T = len(m.days)
    P = np.array([param], np.int64)
    g = np.zeros((1, T))
    tu = np.zeros((1, T))
    nh = np.zeros((1, T), np.int64)
    E.run_batch(m.R, m.dec_row, reb_stack, stack, fw, iv, held, P, G.STOP, g, tu, nh)
    if overlay is None:
        e = np.ones((1, T))
    elif isinstance(overlay, str) and overlay.startswith("SELF"):
        eq = np.cumprod(1 + g - cost * tu, axis=1)
        e = G.self_overlay(eq, overlay)
    else:
        e = np.asarray(overlay, float).reshape(1, T)
    eprev = np.concatenate([np.zeros((1, 1)), e[:, :-1]], axis=1)
    eff = np.minimum(eprev, e) * tu + np.abs(e - eprev)
    net = eprev * g - cost * eff
    return ((1 + net) * (1 + m.fxr) - 1)[0], nh[0], eff[0]


def stats(krw):
    eq = np.cumprod(1 + krw)
    peak = np.maximum.accumulate(np.maximum(eq, 1.0))
    r = krw[1:]
    return {"total": eq[-1] - 1, "mdd": (eq / peak - 1).min(),
            "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
            "vol": r.std() * np.sqrt(252)}


def benchmarks(d, m, stack, names):
    """ETF buy-and-hold (KRX 1/5 close -> 9/23 close) and same-engine references."""
    adj = d["prices"]["Adj Close"]
    close = d["prices"]["Close"]
    out = {}
    for tag, tk in D.ETF_TICKERS.items():
        for field, frame in (("adj", adj), ("close", close)):
            s = frame[tk].dropna()
            s = s[(s.index >= "2026-01-05") & (s.index <= "2026-09-23")]
            lv = s / s.iloc[0]
            out[f"ETF_{tag}_{field}"] = {"total": lv.iloc[-1] - 1,
                                         "mdd": (lv / lv.cummax() - 1).min()}
    # 50/50 buy-and-hold of both ETFs
    sT = adj["456600.KS"].dropna()
    sK = adj["0015B0.KS"].dropna()
    idx = sT.index.intersection(sK.index)
    idx = idx[(idx >= "2026-01-05") & (idx <= "2026-09-23")]
    blend = 0.5 * sT[idx] / sT[idx].iloc[0] + 0.5 * sK[idx] / sK[idx].iloc[0]
    out["ETF_blend"] = {"total": blend.iloc[-1] - 1, "mdd": (blend / blend.cummax() - 1).min()}
    # QQQ in KRW on the strategy calendar
    i0, i1 = m.feat["ref_idx"]
    q = m.ref["QQQ"][i0 - 1:i1 + 1]
    qr = q[1:] / q[:-1] - 1
    qr[0] = 0.0
    out["QQQ_krw"] = stats((1 + qr) * (1 + m.fxr) - 1)
    # replicate each fund's full US book at fund weights, rebalanced daily
    for src in ("T", "K", "U"):
        sid = names.index((src, "all", "W"))
        param = [sid, 500, 1, 0, 0, 0, E.SOURCES.index(src)]
        for cost in (0.0, G.COST):
            krw, nh, _ = simulate(m, stack, param, cost=cost)
            out[f"REPL_{src}_c{cost}"] = {**stats(krw), "holdings": nh[1:].mean()}
    return out


def weights_path(m, stack, param):
    """End-of-day weights (T, J) and daily gross/turnover for one base param."""
    sid, N, wi, buf, ri, ex, src = [int(x) for x in param]
    W = np.zeros((len(m.days), len(m.companies)))
    iv = np.nan_to_num(m.iv, nan=0.0, posinf=0.0)
    gross, turn = E.run_weights(m.R, m.dec_row, m.reb[G.REBS[ri]], stack[sid], m.fw[E.SOURCES[src]],
                                iv, m.held[E.SOURCES[src]], N, wi, buf, ex, G.STOP, W)
    return W, gross, turn


def contributions(m, W):
    """Per-stock sum of daily contributions w[t-1] * r[t] (USD, before costs)."""
    contrib = (W[:-1] * m.R[1:]).sum(axis=0)
    return {m.tickers[j]: contrib[j] for j in np.argsort(-np.abs(contrib)) if contrib[j] != 0}
