"""최근 60거래일(5분봉) 창: 시간봉 15~360분 x 정규장/연장 x 지표 x 5단계 기준 x 규칙 전수 시험.

평가 시작 2026-07-16(앞 10거래일은 지표 예열), 선택 구간 07-16~08-19, 확인 구간 08-20~09-25.
"""
import itertools
import time

import numpy as np
import pandas as pd

from core import *

EVAL_START = pd.Timestamp("2026-07-16").date()
SPLIT = pd.Timestamp("2026-08-20").date()
TFS = [15, 20, 30, 40, 60, 80, 90, 120, 160, 180, 240, 320, 360]
LS = [5, 8, 13, 21, 34, 55, 89]
NORMS = [("atr", 14), ("atr", 50), ("rms", 50), ("rms", 100)]
AS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
BS = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
RULES = [(tp_half, panic_hold) for tp_half in (True, False) for panic_hold in (False, True)]
COST = 0.0005  # 시가 체결 편도 0.05%(수수료 0원, 호가 차이만)


def prepare(D, halfs, base_key="5m", eval_start=EVAL_START, split=SPLIT):
    S = {}
    for s in SYMS:
        BA = base_arrays(D[(s, base_key)], halfs)
        j0 = int(np.searchsorted(BA["dates"], eval_start))
        end_js = np.flatnonzero(BA["is_end"][j0:]) + j0
        BA["j0"] = j0
        BA["end_dates"] = BA["dates"][end_js]
        S[s] = BA
    common = sorted(set().union(*[set(S[s]["end_dates"]) for s in SYMS]))
    for s in SYMS:
        ed = S[s]["end_dates"]
        S[s]["map"] = np.searchsorted(ed, common, side="right") - 1  # -1 → 아직 없음(1.0)
    sel = np.array([d < split for d in common])
    return S, common, sel


def portfolio(eqs, S):
    cols = []
    for s, eq in zip(SYMS, eqs):
        m = S[s]["map"]
        cols.append(np.where(m >= 0, eq[np.maximum(m, 0)], 1.0))
    return np.mean(cols, axis=0)


def window_metrics(port, sel):
    last_sel = np.flatnonzero(sel)[-1]
    a = metrics(port)
    m1 = metrics(port[sel])
    m2 = metrics(port[~sel], base=port[last_sel])
    return {"all_ret": a["ret"], "all_sh": a["sharpe"], "all_mdd": a["mdd"],
            "sel_ret": m1["ret"], "sel_sh": m1["sharpe"], "sel_mdd": m1["mdd"],
            "val_ret": m2["ret"], "val_sh": m2["sharpe"], "val_mdd": m2["mdd"]}


def run(D, halfs, tfs, sessions, base_key="5m", eval_start=EVAL_START, split=SPLIT, tag="short"):
    S, common, sel = prepare(D, halfs, base_key, eval_start, split)
    rows = []
    bh = portfolio([buy_hold(S[s]["o"], S[s]["c"], S[s]["is_reg"], S[s]["is_end"], S[s]["j0"]) for s in SYMS], S)
    rows.append({"kind": "buyhold", **window_metrics(bh, sel)})
    t0 = time.time()
    for tf, eth in itertools.product(tfs, sessions):
        TB, IDX, NPRE = {}, {}, {}
        for s in SYMS:
            src_key = "1h" if (not eth and tf % 60 == 0) else base_key
            tb = tf_bars(D[(s, src_key)], tf, eth, halfs)
            TB[s] = tb
            IDX[s] = map_completed(S[s]["start_ns"], tb.end_ns.values)
            NPRE[s] = int(np.searchsorted(tb.end_ns.values, S[s]["start_ns"][S[s]["j0"]], side="right"))
        npre = min(NPRE.values())
        # 원본 지표(0.01/3.0/ATR5/0.5) 2색 규칙
        eqs, ntr, exs = [], 0, []
        for s in SYMS:
            tb = TB[s]
            K = kalman(tb.C.values, 0.01, 3.0)
            _, trd = ratchet(K, atr(tb.H.values, tb.L.values, tb.C.values, 5), 0.5)
            z = trd.copy(); z[z == 0] = -9
            nanv = np.full(len(tb), np.nan)
            eq, n, _, ex = simulate(S[s]["o"], S[s]["h"], S[s]["l"], S[s]["c"], S[s]["is_reg"], S[s]["is_awake"],
                                    S[s]["is_end"], IDX[s], z, nanv, nanv, S[s]["j0"], True, False, False, COST, False)
            eqs.append(eq); ntr += n; exs.append(ex.mean())
        rows.append({"kind": "original", "tf": tf, "eth": eth, "trades": ntr, "expo": float(np.mean(exs)),
                     **window_metrics(portfolio(eqs, S), sel)})
        for L in LS:
            g = length_to_gain(L)
            q = gain_to_q(g)
            KK = {s: kalman(TB[s].C.values, q, 3.0) for s in SYMS}
            for norm, n in NORMS:
                if npre < max(n + 5, L):
                    continue
                UU = {}
                for s in SYMS:
                    tb = TB[s]
                    if norm == "atr":
                        UU[s] = atr(tb.H.values, tb.L.values, tb.C.values, n) * band_scale(g)
                    else:
                        UU[s] = rms(tb.C.values - KK[s], n)
                for a, b in [(a, b) for a in AS for b in BS if b >= a + 0.25]:
                    ZZ, TP, PB = {}, {}, {}
                    for s in SYMS:
                        ZZ[s] = zones(TB[s].C.values, KK[s], UU[s], a, b)
                        TP[s] = KK[s] + b * UU[s]
                        PB[s] = KK[s] - b * UU[s]
                    for tp_half, panic_hold in RULES:
                        eqs, ntr, exs = [], 0, []
                        for s in SYMS:
                            eq, nt, _, ex = simulate(S[s]["o"], S[s]["h"], S[s]["l"], S[s]["c"], S[s]["is_reg"],
                                                     S[s]["is_awake"], S[s]["is_end"], IDX[s], ZZ[s], TP[s], PB[s],
                                                     S[s]["j0"], tp_half, panic_hold, True, COST, False)
                            eqs.append(eq); ntr += nt; exs.append(ex.mean())
                        rows.append({"kind": "zone5", "tf": tf, "eth": eth, "L": L, "norm": norm, "n": n,
                                     "a": a, "b": b, "tp_half": tp_half, "panic_hold": panic_hold,
                                     "trades": ntr, "expo": float(np.mean(exs)),
                                     **window_metrics(portfolio(eqs, S), sel)})
        print(f"[{tag}] tf={tf} eth={eth} npre={npre} rows={len(rows)} {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    D = load()
    halfs = set()
    for s in SYMS:
        halfs |= half_days(D[(s, "1h")])
    df = run(D, halfs, TFS, (False, True))
    df.to_parquet("grid_short2.parquet")
    print(df.groupby("kind").size())
