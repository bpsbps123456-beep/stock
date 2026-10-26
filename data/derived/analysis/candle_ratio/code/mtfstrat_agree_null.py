"""AGREE 판정의 보조 귀무(갈래 strat): flag가 아니라 z봉 원안 상태를 k일 원형 이동한 뒤 AGREE = (이동한 z 상태 == s)로 다시 만든다.

왜: mtf_run 귀무는 AGREE flag(= z 상태 == s)를 통째로 민다. 그러면 flag가 s와 따로 뒤집혀 귀무 쪽 매매가 약 1.8배가 된다
(b40 z160 AGREE A0 확인 회전 실제 24, 귀무 중앙 43). 실제 판에서는 그만큼 비용이 귀무에만 더 붙는다.
이 귀무는 z 상태 자체의 뒤집힘 구조를 그대로 두고 시각만 어긋나게 한다. 이동 칸·달력·k는 mtf_run.shifter와 같다.

py code/mtfstrat_agree_null.py [프로세스 수=3] → results/mtfstrat_agree_null.parquet
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import HERE
from mtf_run import F32, GUARD, SYMS, calendar, shifter
from zones_mtf import BASE_TFS, prepare, run

OUT = HERE / "results"
BOARDS = {"real": (0.001, False), "real0": (0.0, False)}


def zstates(P):
    s1 = P["s"] == 1
    return {nm.replace("_AGREE", "_Z"): np.where(fl, s1, ~s1) for nm, fl in P["flags"].items() if nm.endswith("_AGREE")}


def work(args):
    sym, cal, ks = args
    rows = []
    for b in BASE_TFS:
        P = prepare(sym, b)
        s1 = P["s"] == 1
        zs = zstates(P)
        if not zs:
            continue
        sh = shifter(P, cal)
        for k in [0] + ks:
            z = zs if k == 0 else sh(zs, k)
            fl = {nm.replace("_Z", "_AGREE"): (v == s1) for nm, v in z.items()}
            if k == 0:  # 원래 flag 복원 확인
                for nm, v in fl.items():
                    assert np.array_equal(v, P["flags"][nm]), (sym, b, nm)
            for bd, (cost, ideal) in BOARDS.items():
                for r in run(P, cost, ideal, fl):
                    if r["clf"] == "BASE":
                        continue
                    r.update(board=bd, k=k, sym=sym)
                    rows.append(r)
    g = pd.DataFrame(rows)
    for c in F32:
        g[c] = g[c].astype(np.float32)
    return g


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    t0 = time.time()
    cal = calendar(SYMS)
    D = len(cal)
    ks = list(range(GUARD, D - GUARD + 1))
    print("달력", cal[0], "~", cal[-1], "D", D, "이동", ks[0], "~", ks[-1], len(ks), flush=True)
    G = []
    with Pool(nproc) as pool:
        for i, g in enumerate(pool.imap_unordered(work, [(s, cal, ks) for s in SYMS]), 1):
            G.append(g)
            if i % 12 == 0 or i == len(SYMS):
                print(f"{i}/{len(SYMS)} {time.time() - t0:.0f}s", flush=True)
    g = pd.concat(G, ignore_index=True)
    for c in ("sym", "clf", "action", "board"):
        g[c] = g[c].astype("category")
    g["k"] = g.k.astype(np.int16)
    g.to_parquet(OUT / "mtfstrat_agree_null.parquet")
    print("완료", len(g), "행", round(time.time() - t0), "초", flush=True)
