"""갈래 trade: 질문 3(매매)과 BRIEF 채택 기준 1~3.

실제: 종목 × 판정 28 × 구간(P1·P2·Y25·Y24·Y23) × 비용(0·0.05·0.1%) → 수익·낙폭·평균 보유 x·진입 수·같은 비중·보유.
귀무: P1·P2·Y25에서 판정 상태를 구간 안 원형 이동(swing.shift_state와 같은 정의, 모든 종목·판정 같은 k, 15 ≤ k ≤ D−15).
→ results/trade_actual.csv, results/trade_null.npz
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from numba import njit

import swing

OUT = swing.HERE / "results"
COSTS = (0.0, 0.0005, 0.001)
PERS = ("P1", "P2", "Y25", "Y24", "Y23")
NULL_PERS = ("P1", "P2", "Y25")
NAMES = [swing.dname(k, p) for k, p in swing.DETECTORS]
WARM = 60


@njit(cache=True)
def null_kernel(o, c, st, j0, j1, cost, ks):
    """ks마다 shift_state(st, j0, j1, k) → sim_period, static_bh(같은 비중 x). 반환: 수익, 낙폭, x, 진입, 같은 비중 수익, 같은 비중 낙폭."""
    nk = len(ks)
    rr = np.empty(nk); dd = np.empty(nk); xs = np.empty(nk); en = np.empty(nk)
    sr = np.empty(nk); sd = np.empty(nk)
    a = max(j0 - 1, 0)
    L = j1 + 1 - a
    seg = st[a:j1 + 1].copy()
    tmp = st.copy()
    n = j1 - j0 + 1
    px = np.empty(n + 1)
    for t in range(n):
        px[t] = o[j0 + t]
    px[n] = c[j1]
    for q in range(nk):
        k = ks[q]
        for i in range(L):
            tmp[a + (i + k) % L] = seg[i]
        ret, mdd, x, ent, eq = swing.sim_period(o, c, tmp, j0, j1, cost)
        pk = 1.0; m = 0.0; e = 1.0
        for t in range(n + 1):
            e = 1.0 + x * (px[t] / px[0] - 1.0)
            if e > pk:
                pk = e
            v = e / pk - 1.0
            if v < m:
                m = v
        rr[q] = ret; dd[q] = mdd; xs[q] = x; en[q] = ent; sr[q] = e - 1.0; sd[q] = m
    return rr, dd, xs, en, sr, sd


def included(g, per, D):
    idx = swing.period_index(g, per)
    a, b = swing.PERIODS[per]
    return len(idx) == D and idx[0] >= WARM and g.date.iloc[-1] >= b


def selfcheck(data):
    """null_kernel이 swing.shift_state + sim_period + static_bh와 같은지."""
    rng = np.random.default_rng(0)
    syms = list(data)
    worst = 0.0
    for _ in range(40):
        s = syms[rng.integers(len(syms))]
        g = data[s]
        per = NULL_PERS[rng.integers(3)]
        idx = swing.period_index(g, per)
        if len(idx) == 0 or idx[0] < WARM:
            continue
        j0, j1 = idx[0], idx[-1]
        st = swing.states(g)[NAMES[rng.integers(28)]]
        o, c = g.O.values.astype(float), g.C.values.astype(float)
        cost = COSTS[rng.integers(3)]
        ks = np.array([15, 37, len(idx) - 15], np.int64)
        res = null_kernel(o, c, st, j0, j1, cost, ks)
        for q, k in enumerate(ks):
            r = swing.sim_period(o, c, swing.shift_state(st, j0, j1, int(k)), j0, j1, cost)
            b = swing.static_bh(o, c, j0, j1, r[2])
            ref = (r[0], r[1], r[2], r[3], b[0], b[1])
            worst = max(worst, max(abs(res[i][q] - ref[i]) for i in range(6)))
    assert worst < 1e-10, worst
    return worst


def main():
    t0 = time.time()
    data = swing.load()
    print("자체 확인 최대 오차", selfcheck(data))
    ref = data["SPY"]
    Dper = {p: len(swing.period_index(ref, p)) for p in PERS}
    incl = {p: [s for s, g in data.items() if included(g, p, Dper[p])] for p in PERS}
    ks = {p: np.arange(15, Dper[p] - 15 + 1, dtype=np.int64) for p in NULL_PERS}
    print({p: (Dper[p], len(incl[p])) for p in PERS}, {p: len(v) for p, v in ks.items()})
    null = {p: np.full((len(incl[p]), len(COSTS), 28, len(ks[p]), 2), np.nan) for p in NULL_PERS}
    rows = []
    for s, g in data.items():
        o, c = g.O.values.astype(float), g.C.values.astype(float)
        sts = swing.states(g)
        for p in PERS:
            if s not in incl[p]:
                continue
            idx = swing.period_index(g, p)
            j0, j1 = int(idx[0]), int(idx[-1])
            si = incl[p].index(s)
            for di, nm in enumerate(NAMES):
                st = sts[nm]
                for ci, cost in enumerate(COSTS):
                    ret, mdd, x, ent, _ = swing.sim_period(o, c, st, j0, j1, cost)
                    sw, swd, bh, bhd = swing.static_bh(o, c, j0, j1, x)
                    rows.append((p, s, nm, nm.split("_")[0], cost, ret, mdd, x, ent, sw, swd, bh, bhd))
                    if p in NULL_PERS:
                        rr, dd, xs, en, sr, sd = null_kernel(o, c, st, j0, j1, cost, ks[p])
                        null[p][si, ci, di, :, 0] = rr - sr
                        null[p][si, ci, di, :, 1] = dd - sd
    df = pd.DataFrame(rows, columns=["per", "sym", "det", "kind", "cost", "ret", "mdd", "x", "ent", "sw_ret", "sw_mdd", "bh_ret", "bh_mdd"])
    df["ex"] = df.ret - df.sw_ret
    df["dd_ex"] = df.mdd - df.sw_mdd
    df.to_csv(OUT / "trade_actual.csv", index=False)
    np.savez_compressed(OUT / "trade_null.npz", names=np.array(NAMES), costs=np.array(COSTS),
                        **{f"{p}_null": null[p] for p in NULL_PERS},
                        **{f"{p}_ks": ks[p] for p in NULL_PERS},
                        **{f"{p}_syms": np.array(incl[p]) for p in NULL_PERS})
    print(len(df), "행", f"{time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
