"""5분 간격 시간봉(10~360분) x 정규장/연장 x 칼만 평활(거래일 기준) x 폭 x a·b x 규칙 전수 탐색.

모든 시간봉을 같은 조건에 두려고 평활·폭 창을 봉 수가 아니라 거래일로 정하고, 예열은 5분봉 시작(7/2)부터 똑같이 한다.
최근 창: 평가 7/22~9/25(선택 ~8/21, 확인 8/24~). 2026년 창(60분 배수 정규장만): 선택 1~5월, 확인 6~9월.
"""
import time
from datetime import date
from multiprocessing import Pool

import numpy as np
import pandas as pd

from core import *
from grid_short import prepare, portfolio, window_metrics

HK = [1, 2, 3, 5, 7, 10]                                          # 칼만 평활(거래일)
NORMS = [("rms", 5), ("rms", 10), ("atr", 5), ("atr", 10)]        # 폭 계산 창(거래일)
AB = [(a, b) for a in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0] for b in [1.25, 1.5, 1.75, 2.0, 2.5, 3.0] if b >= a + 0.25]
RULES = [(True, True), (True, False), (False, True), (False, False)]  # (절반 익절, 패닉 보유)
WIN = {"short": dict(base_key="5m", eval_start=date(2026, 7, 22), split=date(2026, 8, 24)),
       "long": dict(base_key="1h", eval_start=date(2026, 1, 2), split=date(2026, 6, 1))}
COST = 0.0005
_ctx = {}


def ctx(win):
    if win not in _ctx:
        D = load()
        H = set()
        for s in SYMS:
            H |= half_days(D[(s, "1h")])
        S, common, sel = prepare(D, H, **WIN[win])
        _ctx[win] = (D, H, S, common, sel)
    return _ctx[win]


def bars_per_day(tf, eth):
    return int(np.ceil((960 if eth else 390) / tf))


def run_one(S, IDX, ZZ, TP, PB, th, ph, use_lim):
    eqs, ntr, exs = [], 0, []
    for s in SYMS:
        eq, nt, _, ex = simulate(S[s]["o"], S[s]["h"], S[s]["l"], S[s]["c"], S[s]["is_reg"], S[s]["is_awake"],
                                 S[s]["is_end"], IDX[s], ZZ[s], TP[s], PB[s], S[s]["j0"], th, ph, use_lim, COST, False)
        eqs.append(eq); ntr += nt; exs.append(ex.mean())
    return eqs, ntr, float(np.mean(exs))


def job(args):
    win, tf, eth = args
    D, H, S, common, sel = ctx(win)
    src = "1h" if win == "long" else "5m"
    TB, IDX, npre = {}, {}, 10 ** 9
    for s in SYMS:
        tb = tf_bars(D[(s, src)], tf, eth, H)
        TB[s] = tb
        IDX[s] = map_completed(S[s]["start_ns"], tb.end_ns.values)
        npre = min(npre, int(np.searchsorted(tb.end_ns.values, S[s]["start_ns"][S[s]["j0"]], side="right")))
    d = bars_per_day(tf, eth)
    rows = []
    # 원본 지표 2색 규칙
    ZZ, nanv = {}, {}
    for s in SYMS:
        tb = TB[s]
        K = kalman(tb.C.values, 0.01, 3.0)
        _, trd = ratchet(K, atr(tb.H.values, tb.L.values, tb.C.values, 5), 0.5)
        z = trd.copy(); z[z == 0] = -9
        ZZ[s] = z; nanv[s] = np.full(len(tb), np.nan)
    eqs, ntr, ex = run_one(S, IDX, ZZ, nanv, nanv, True, False, False)
    rows.append({"win": win, "tf": tf, "eth": eth, "kind": "original", "trades": ntr, "expo": ex,
                 **window_metrics(portfolio(eqs, S), sel)})
    for hk in HK:
        L = max(3, int(round(hk * d)))
        g = length_to_gain(L)
        KK = {s: kalman(TB[s].C.values, gain_to_q(g), 3.0) for s in SYMS}
        for norm, hn in NORMS:
            n = max(5, int(round(hn * d)))
            if npre < n:
                continue
            UU = {}
            for s in SYMS:
                tb = TB[s]
                UU[s] = (atr(tb.H.values, tb.L.values, tb.C.values, n) * band_scale(g)) if norm == "atr" \
                    else rms(tb.C.values - KK[s], n)
            for a, b in AB:
                ZZ = {s: zones(TB[s].C.values, KK[s], UU[s], a, b) for s in SYMS}
                TP = {s: KK[s] + b * UU[s] for s in SYMS}
                PB = {s: KK[s] - b * UU[s] for s in SYMS}
                for th, ph in RULES:
                    eqs, ntr, ex = run_one(S, IDX, ZZ, TP, PB, th, ph, True)
                    rows.append({"win": win, "tf": tf, "eth": eth, "kind": "zone5", "hk": hk, "L": L,
                                 "norm": norm, "hn": hn, "n": n, "a": a, "b": b, "tp_half": th, "panic_hold": ph,
                                 "trades": ntr, "expo": ex, **window_metrics(portfolio(eqs, S), sel)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    jobs = [("long", tf, False) for tf in (60, 120, 180, 240, 300, 360)]
    jobs += [("short", tf, eth) for tf in range(10, 365, 5) for eth in (False, True)]
    bh = []
    for win in ("short", "long"):
        D, H, S, common, sel = ctx(win)
        eq = portfolio([buy_hold(S[s]["o"], S[s]["c"], S[s]["is_reg"], S[s]["is_end"], S[s]["j0"]) for s in SYMS], S)
        bh.append({"win": win, "kind": "buyhold", **window_metrics(eq, sel)})
    parts = [pd.DataFrame(bh)]
    with Pool(11) as pool:
        for i, df in enumerate(pool.imap_unordered(job, jobs), 1):
            parts.append(df)
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} jobs, rows={sum(len(p) for p in parts)}, {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet("dense.parquet")
    print(out.groupby(["win", "kind"]).size())
