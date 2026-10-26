"""mtf diag 갈래 보강: '추세 < 애매' 반대 방향이 구성(보유 비율·변동 크기) 차이 때문인지 점검.

zones_mtf.diag의 봉당 노출 맞춘 초과 = 구간 안 공분산 cov_Z(pos, r)이다. 이것은 구간 안 보유 변동(sd pos)과 봉 수익 변동(sd r)에 비례한다.
예: 하락 추세 봉에서 원안이 거의 늘 현금이면 sd pos ≈ 0이라 초과가 0에 가깝게 눌린다.
그래서 같은 봉 집합에서 상관 corr_Z = cov_Z ÷ (sd pos · sd r)도 잰다. 종목별 corr_T − corr_A → 144종목 중앙, 귀무 k(mtf_run.shifter)로 비교.
같이 내는 것: 구간별 평균 보유 x̄, sd pos, sd r, 그리고 cov로 다시 잰 봉당 초과(mtf_diag.parquet와 맞는지 확인용).
py code/mtfdiag_corr.py [프로세스 수=3] → results/mtfdiag_corr_raw.parquet
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import HERE
from mtf_run import GUARD, SYMS, shifter
from zones3 import sim_frac, targets
from zones_mtf import BASE_TFS, prepare

OUT = HERE / "results"
BOARDS = {"real0": False, "ideal": True}


def days_of(sym):
    P = prepare(sym, 240)
    return set(P["tb"].day.values[P["seg"] >= 0])


def set_stats(M, p, r):
    """M: (판정, 봉) 불리언. 각 판정 집합의 n, x̄, sd pos, sd r, cov(pos, r)."""
    Mf = M.astype(np.float64)
    n = Mf.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mp, mr = Mf @ p / n, Mf @ r / n
        vp = np.maximum(Mf @ (p * p) / n - mp * mp, 0.0)
        vr = np.maximum(Mf @ (r * r) / n - mr * mr, 0.0)
        cov = Mf @ (p * r) / n - mp * mr
    return n, mp, np.sqrt(vp), np.sqrt(vr), cov


def work(args):
    sym, cal, ks = args
    rows = []
    for b in BASE_TFS:
        P = prepare(sym, b)
        sh = shifter(P, cal)
        O, C, seg = P["O"], P["C"], P["seg"]
        r = np.r_[O[1:] / O[:-1] - 1, C[-1] / O[-1] - 1]
        pos = {}
        for bd, ideal in BOARDS.items():
            aw = np.ones(len(O), np.bool_) if ideal else P["awake"]
            po = np.zeros(len(O))
            sim_frac(O, C, targets(P["s"], np.zeros(len(O), np.bool_), 3), seg, aw, 0.0, po)
            pos[bd] = po
        names = list(P["flags"])
        for k in [0] + ks:
            fl = P["flags"] if k == 0 else sh(P["flags"], k)
            Z = np.stack([np.r_[False, fl[nm][:-1]] for nm in names])
            for w in (0, 1):
                m = seg == w
                Zw, rw = Z[:, m], r[m]
                for bd, po in pos.items():
                    pw = po[m]
                    for zn, M in (("T", Zw), ("A", ~Zw)):
                        n, mp, sp, sr, cov = set_stats(M, pw, rw)
                        for i, nm in enumerate(names):
                            rows.append((bd, b, nm, k, w, zn, n[i], mp[i], sp[i], sr[i], cov[i]))
    g = pd.DataFrame(rows, columns=["board", "b", "clf", "k", "win", "zone", "n", "xbar", "sd_pos", "sd_r", "cov"])
    g["sym"] = sym
    for c in ("xbar", "sd_pos", "sd_r", "cov"):
        g[c] = g[c].astype(np.float32)
    return g


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    t0 = time.time()
    with Pool(nproc) as pool:
        cal = sorted(set().union(*pool.map(days_of, SYMS)))
        D = len(cal)
        ks = list(range(GUARD, D - GUARD + 1))
        print("달력", cal[0], "~", cal[-1], "D", D, "이동", ks[0], "~", ks[-1], len(ks), f"{time.time() - t0:.0f}s", flush=True)
        G = []
        for i, g in enumerate(pool.imap_unordered(work, [(s, cal, ks) for s in SYMS]), 1):
            G.append(g)
            if i % 24 == 0 or i == len(SYMS):
                print(f"{i}/{len(SYMS)} {time.time() - t0:.0f}s", flush=True)
    g = pd.concat(G, ignore_index=True)
    for c in ("sym", "clf", "board", "zone"):
        g[c] = g[c].astype("category")
    g["k"] = g.k.astype(np.int16); g["win"] = g.win.astype(np.int8); g["n"] = g.n.astype(np.int32)
    g.to_parquet(OUT / "mtfdiag_corr_raw.parquet")
    print("완료", len(g), "행", round(time.time() - t0), "초", flush=True)
