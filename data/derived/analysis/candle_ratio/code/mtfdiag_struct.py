"""mtf diag 갈래 구조 점검: 판정 flag가 신호 뒤집힘에 기계적으로 묶였는지.

뒤집힘 봉 i = 평가 봉 중 s_i ≠ s_{i−1}. 그 봉 마감의 flag가 새 포지션 첫 봉(i+1)의 구간을 정한다(zones_mtf.diag와 같은 정렬).
지표 = 뒤집힘 봉의 애매 비율 − 전체 평가 봉의 애매 비율(%p), 종목별 → 144종목 중앙. 고르기·확인 각각.
귀무 = mtf_run.shifter와 같은 flag 공통 k 이동(15~54). 귀무에서는 flag가 그날의 뒤집힘과 무관해진다.
관측이 귀무 범위를 크게 넘는 판정은 '애매 = 뒤집힘 직후'를 정의로 품고 있어 판정만 옮기는 귀무가 맞지 않는다.
py code/mtfdiag_struct.py [프로세스 수=3] → results/mtfdiag_struct_raw.parquet, mtfdiag_struct.csv
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import HERE
from mtf_run import GUARD, SYMS, shifter
from zones_mtf import BASE_TFS, prepare

OUT = HERE / "results"


def days_of(sym):
    P = prepare(sym, 240)
    return set(P["tb"].day.values[P["seg"] >= 0])


def work(args):
    sym, cal, ks = args
    rows = []
    for b in BASE_TFS:
        P = prepare(sym, b)
        sh = shifter(P, cal)
        s, seg = P["s"], P["seg"]
        flip = np.r_[False, s[1:] != s[:-1]] & (seg >= 0)
        for k in [0] + ks:
            fl = P["flags"] if k == 0 else sh(P["flags"], k)
            for nm, f in fl.items():
                for w in (0, 1):
                    m = seg == w
                    mf = m & flip
                    rows.append((b, nm, k, w, float((~f[m]).mean()), float((~f[mf]).mean()) if mf.any() else np.nan, int(mf.sum()), int(m.sum())))
    g = pd.DataFrame(rows, columns=["b", "clf", "k", "win", "amb_all", "amb_flip", "n_flip", "n_bar"])
    g["sym"] = sym
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
    for c in ("sym", "clf"):
        g[c] = g[c].astype("category")
    g.to_parquet(OUT / "mtfdiag_struct_raw.parquet")
    g["lift"] = (g.amb_flip - g.amb_all) * 100
    med = g.groupby(["win", "b", "clf", "k"], observed=True).agg(lift=("lift", "median"), amb_all=("amb_all", "median"),
                                                                 amb_flip=("amb_flip", "median"), n_flip=("n_flip", "median")).reset_index()
    obs = med[med.k == 0].set_index(["win", "b", "clf"])
    nul = med[med.k > 0].groupby(["win", "b", "clf"]).lift
    out = obs.drop(columns="k").join(nul.median().rename("null_med")).join(nul.quantile(0.95).rename("null_q95")).join(nul.max().rename("null_max")).reset_index()
    out["amb_all"] *= 100; out["amb_flip"] *= 100
    out["type"] = out.clf.astype(str).str.split("_").str[1]
    out["z"] = out.clf.astype(str).str.split("_").str[0].str[1:]
    out["zeqb"] = out.z == out.b.astype(str)
    out["win"] = out.win.map({0: "sel", 1: "conf"})
    out.to_csv(OUT / "mtfdiag_struct.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(out.round(2).to_string(index=False))
    s = out.assign(kind=np.where(out.type == "AGREE", "AGREE", np.where(out.type == "EXT", np.where(out.zeqb, "EXT z=b", "EXT z>b"),
                                                                       out.type + np.where(out.zeqb, " z=b", " z>b"))))
    print(s.groupby(["win", "kind"]).agg(n=("lift", "size"), lift_med=("lift", "median"), lift_min=("lift", "min"), lift_max=("lift", "max"),
                                         null_q95_med=("null_q95", "median"), over_null_max=("lift", lambda x: int((x > s.loc[x.index, "null_max"]).sum()))).round(2))
    print("완료", round(time.time() - t0), "초", flush=True)
