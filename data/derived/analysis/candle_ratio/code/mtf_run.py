"""BRIEF3 실행: 144종목 × 신호 시간봉 9개 × 전략(원안 + 판정 × A0·A2), 판 3개, 귀무 k 15~54(공통 달력 같은 시각 칸).

py code/mtf_run.py [프로세스 수] → results/mtf_rows.parquet(전략), results/mtf_diag.parquet(진단), results/mtf_meta.json
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import CACHE, HERE, WIN
from zones_mtf import BASE_TFS, diag, prepare, run

OUT = HERE / "results"
SYMS = sorted(p.stem[:-3] for p in CACHE.glob("*_5m.parquet"))
BOARDS = {"real": (0.001, False), "real0": (0.0, False), "ideal": (0.0, True)}
NULL_BOARDS = ("real", "real0")
GUARD = 15
F32 = ["ret_sel", "mdd_sel", "x_sel", "static_sel", "static_mdd_sel", "bh_sel", "turn_sel",
       "ret_conf", "mdd_conf", "x_conf", "static_conf", "static_mdd_conf", "bh_conf", "turn_conf"]


def calendar(syms):
    days = set()
    for s in syms:
        P = prepare(s, 240)
        days.update(P["tb"].day.values[P["seg"] >= 0])
    return sorted(days)


def shifter(P, cal):
    """평가 봉(seg ≥ 0)만 달력 (d − k) mod D 같은 시각 칸에서 flag를 가져온다. 빈 칸은 앞 봉 값을 잇고, 예열 봉은 그대로."""
    tb = P["tb"]
    ev = np.flatnonzero(P["seg"] >= 0)
    dix = {d: i for i, d in enumerate(cal)}
    D = len(cal)
    st = pd.DatetimeIndex(tb.start.values[ev])
    sl = (np.asarray(st.hour * 60 + st.minute) - 240) // P["b"]
    di = np.array([dix[d] for d in tb.day.values[ev]])
    grid = np.full((D, int(sl.max()) + 1), -1, np.int64)
    grid[di, sl] = ev

    def shifted(flags, k):
        src = grid[(di - k) % D, sl]
        pos = np.arange(len(ev))
        last = np.maximum.accumulate(np.where(src >= 0, pos, -1))
        s2 = np.where(last >= 0, src[np.maximum(last, 0)], -1)
        out = {}
        for nm, v in flags.items():
            w = v.copy()
            w[ev] = np.where(s2 >= 0, v[np.maximum(s2, 0)], False)
            out[nm] = w
        return out
    return shifted


def work(args):
    sym, cal, ks = args
    rows, drows = [], []
    for b in BASE_TFS:
        P = prepare(sym, b)
        sh = shifter(P, cal)
        for k in [0] + ks:
            fl = P["flags"] if k == 0 else sh(P["flags"], k)
            for bd, (cost, ideal) in BOARDS.items():
                if k and bd not in NULL_BOARDS:
                    continue
                for r in run(P, cost, ideal, fl):
                    r.update(board=bd, k=k)
                    rows.append(r)
            for bd, ideal in (("real0", False), ("ideal", True)):
                for r in diag(P, ideal, fl):
                    r.update(board=bd, k=k)
                    drows.append(r)
    g = pd.DataFrame(rows); g["sym"] = sym
    d = pd.DataFrame(drows); d["sym"] = sym
    for c in F32:
        g[c] = g[c].astype(np.float32)
    return sym, g, d


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    t0 = time.time()
    cal = calendar(SYMS)
    D = len(cal)
    ks = list(range(GUARD, D - GUARD + 1))
    print("달력", cal[0], "~", cal[-1], "D", D, "이동", ks[0], "~", ks[-1], len(ks), flush=True)
    G, Dg = [], []
    with Pool(nproc) as pool:
        for i, (sym, g, d) in enumerate(pool.imap_unordered(work, [(s, cal, ks) for s in SYMS]), 1):
            G.append(g); Dg.append(d)
            if i % 12 == 0 or i == len(SYMS):
                print(f"{i}/{len(SYMS)} {time.time() - t0:.0f}s", flush=True)
    g = pd.concat(G, ignore_index=True)
    d = pd.concat(Dg, ignore_index=True)
    for c in ("sym", "clf", "action", "board"):
        g[c] = g[c].astype("category")
    for c in ("sym", "clf", "board"):
        d[c] = d[c].astype("category")
    g["k"] = g.k.astype(np.int16); d["k"] = d.k.astype(np.int16)
    g.to_parquet(OUT / "mtf_rows.parquet")
    d.to_parquet(OUT / "mtf_diag.parquet")
    json.dump(dict(D=D, calendar=[str(cal[0]), str(cal[-1])], shifts=[ks[0], ks[-1]], n_shifts=len(ks), n_syms=len(SYMS),
                   base_tfs=BASE_TFS, n_strategies=int(g[(g.k == 0) & (g.board == "real")].groupby("sym").size().iloc[0]),
                   seconds=round(time.time() - t0)), open(OUT / "mtf_meta.json", "w"), ensure_ascii=False, indent=1)
    print("완료", len(g), "행", len(d), "진단 행", round(time.time() - t0), "초", flush=True)
