"""MU 원형 이동 귀무 (BRIEF 판정 2).

py code/null_shift.py run 5m      → 이동 k=0..D-1 격자(비용 0)를 돌려 results/null_MU_5m_{sel,conf}.npy, _cells.parquet, _shifts.csv
분석은 code/null_analyze.py.

이동은 run_grid.grid(shift_days=k) 그대로(비율 p를 k×하루 봉 수(중앙값)만큼 np.roll). 가격·보유 기준은 그대로다.
"""
from __future__ import annotations


import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from engine import HERE, WIN, load, tf_bars
from run_grid import grid

SYM = "MU"
OUT = HERE / "results"
WORKERS = 3
ORIG = dict(tf=40, sess="eth", N=10, ratio="count", dir="follow", dayflat=False, b=0.5, s=0.5)
CELL_KEYS = ["tf", "sess", "N", "ratio", "dir", "dayflat", "b", "s"]


def zero_cost(sess):
    return [0.0]


def n_days(base):
    d = load(SYM, base)
    tb = tf_bars(d, 60, "eth")
    return int(len(np.unique(tb.day.values[tb.day.values >= WIN[base]["warm"]])))


def spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize(g: pd.DataFrame, k: int) -> dict:
    rs = g.ret_sel.values
    rc = g.ret_conf.values
    order = np.argsort(-rs, kind="stable")
    top = order[0]
    ex_sel = rs - g.hold_sel.values / g.bars_sel.values * g.bh_sel.values
    ex_conf = rc - g.hold_conf.values / g.bars_conf.values * g.bh_conf.values
    fol = (g["dir"] == "follow").values
    row = dict(k=k, top_sel=float(rs[top]), top_conf=float(rc[top]),
               top_ties=int((rs == rs[top]).sum()),
               top20_conf_med=float(np.median(rc[order[:20]])),
               rho=spearman(rs, rc),
               exf_sel=float(np.median(ex_sel[fol])), exf_conf=float(np.median(ex_conf[fol])),
               exd_sel=float(np.median(ex_sel[~fol])), exd_conf=float(np.median(ex_conf[~fol])),
               top_ent_sel=int(g.ent_sel.values[top]), top_ent_conf=int(g.ent_conf.values[top]))
    for c in CELL_KEYS:
        v = g[c].values[top]
        row["top_" + c] = v.item() if hasattr(v, "item") else v
    return row


def work(args):
    base, k = args
    t0 = time.time()
    g = grid(SYM, base, costs_fn=zero_cost, shift_days=k)
    s = summarize(g, k)
    s["sec"] = round(time.time() - t0, 1)
    return k, s, g.ret_sel.values.astype(np.float32), g.ret_conf.values.astype(np.float32), \
        (g[CELL_KEYS + ["bh_sel", "bh_conf"]] if k == 0 else None)


def run(base):
    D = n_days(base)
    ks = list(range(0, D))  # 0 = 실제, 1..D-1 = 귀무
    t0 = time.time()
    first = work((base, 0))
    M = len(first[2])
    sel = np.full((D, M), np.nan, np.float32)
    conf = np.full((D, M), np.nan, np.float32)
    rows = [first[1]]
    sel[0], conf[0] = first[2], first[3]
    cells = first[4]
    cells.to_parquet(OUT / f"null_{SYM}_{base}_cells.parquet")
    # k=0이 기존 격자(비용 0)와 같은지 확인
    old = pd.read_parquet(OUT / f"grid_{SYM}_{base}.parquet")
    old = old[old.cost == 0.0].reset_index(drop=True)
    same = len(old) == M and np.allclose(old.ret_sel.values, sel[0], atol=1e-5) and np.allclose(old.ret_conf.values, conf[0], atol=1e-5)
    print(base, "D", D, "칸", M, "k=0 기존 격자와 일치:", same, round(time.time() - t0, 1), "초", flush=True)
    with ProcessPoolExecutor(WORKERS) as ex:
        for k, s, a, b, _ in ex.map(work, [(base, k) for k in ks[1:]]):
            sel[k], conf[k] = a, b
            rows.append(s)
            print(base, "k", k, s["sec"], "초", "top_sel", round(s["top_sel"], 3), "top_conf", round(s["top_conf"], 3), flush=True)
    np.save(OUT / f"null_{SYM}_{base}_sel.npy", sel)
    np.save(OUT / f"null_{SYM}_{base}_conf.npy", conf)
    pd.DataFrame(rows).sort_values("k").to_csv(OUT / f"null_{SYM}_{base}_shifts.csv", index=False)
    print(base, "끝", round(time.time() - t0, 1), "초", flush=True)


if __name__ == "__main__":
    mode, base = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        WORKERS = int(sys.argv[3])
    assert mode == "run", mode
    run(base)
