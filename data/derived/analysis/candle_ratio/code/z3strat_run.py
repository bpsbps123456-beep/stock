"""z3strat 갈래 실행: BRIEF2 채택 기준 2(전략)·3(1시간봉 방향)의 원자료.

- 144종목(1시간봉 창은 6/1 전 자료가 있는 143종목) × zones3.run_symbol 전략 85개(28 × 3 + 원안).
- 판 5개: real(깨어 있음 + 편도 0.1%, 주), real0(비용 0), real3(0.03%), ideal(모든 봉, 0), std(11/1 표준시 시간표 + 0.1%).
- 귀무: 추세 flag 28개만 거래일 단위로 원형 이동하고 신호 s는 그대로 둔다. 모든 종목에 같은 k, 15 ≤ k ≤ D−15.
  공통 거래일 달력(전 종목 합집합)에서 (d−k) mod D일의 같은 시각 칸 flag를 가져온다.
  그 칸이 없으면(프리·애프터 봉이 빠진 날) 바로 앞 봉의 이동된 flag를 이어 쓴다(처음이면 추세 아님).
  np.roll(k×하루 봉 수)는 봉이 빠진 종목에서 실제 이동 일수가 어긋나므로 쓰지 않는다(loss_run 점검: 5분봉 창 최소 6.3일, 1시간봉 창 2.6일).
- k = 0은 실제(이동 없음)다. 1시간봉 창은 HTF 160·320(5분봉 전용)을 뺀 26 × 3 + 원안 = 79개.

py code/z3strat_run.py 5m|1h [프로세스 수 ≤ 3] → results/z3strat_rows_{base}.parquet, results/z3strat_meta_{base}.json
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import CACHE, HERE, SESS, WIN, load
from zones3 import TF, prepare, run_symbol

OUT = HERE / "results"
SYMS = sorted(p.stem[:-3] for p in CACHE.glob("*_5m.parquet"))
GUARD = 15
BOARDS = {"real": (0.001, False, "dst"), "real0": (0.0, False, "dst"), "real3": (0.0003, False, "dst"),
          "ideal": (0.0, True, "dst"), "std": (0.001, False, "std")}
KEEP = ["ret_sel", "mdd_sel", "x_sel", "static_sel", "static_mdd_sel", "turn_sel",
        "ret_conf", "mdd_conf", "x_conf", "static_conf", "static_mdd_conf", "turn_conf",
        "ret_all", "mdd_all", "x_all", "static_all", "static_mdd_all"]
BH = ["bh_sel", "bh_mdd_sel", "bh_conf", "bh_mdd_conf", "bh_all", "bh_mdd_all"]


def tf_min(base):
    return TF if base == "5m" else 60


def eth_days(sym, base):
    d = load(sym, base)
    m = np.asarray(d.index.hour * 60 + d.index.minute)
    keep = (m >= SESS["eth"][0]) & (m < SESS["eth"][1])
    days = pd.unique(np.asarray(d.index.date)[keep])
    return [x for x in days if x >= WIN[base]["warm"]]


def calendar(base, syms):
    days = set()
    for s in syms:
        days.update(eth_days(s, base))
    return sorted(days)


def shifter(tb, cal, base):
    """이동 k에 대한 출처 봉 번호(없으면 앞 봉 이어 쓰기, −1이면 추세 아님)를 돌려주는 함수."""
    dix = {d: i for i, d in enumerate(cal)}
    D = len(cal)
    st = pd.DatetimeIndex(tb.start)
    sl = (np.asarray(st.hour * 60 + st.minute) - SESS["eth"][0]) // tf_min(base)
    di = np.array([dix[d] for d in tb.day.values])
    n = len(tb)
    grid = np.full((D, int(sl.max()) + 1), -1, np.int64)
    grid[di, sl] = np.arange(n)
    assert (grid >= 0).sum() == n, "칸 중복"

    def src(k):
        s = grid[(di - k) % D, sl]
        last = np.maximum.accumulate(np.where(s >= 0, np.arange(n), -1))
        out = np.where(last >= 0, s[np.maximum(last, 0)], -1)
        return out, float((s < 0).mean())
    return src


def shifted_flags(flags, idx):
    ok = idx >= 0
    j = np.maximum(idx, 0)
    return {k: (v[j] & ok) for k, v in flags.items()}


def work(args):
    sym, base, cal, ks = args
    P = {"dst": prepare(sym, base, "dst"), "std": prepare(sym, base, "std")}
    drop = {"HTF_tf160", "HTF_tf320"} if base == "1h" else set()
    src = shifter(P["dst"]["tb"], cal, base)
    frames, fill = [], {}
    bh = None
    for k in [0] + ks:
        if k == 0:
            fl = P["dst"]["flags"]
        else:
            idx, miss = src(k)
            fill[k] = miss
            fl = shifted_flags(P["dst"]["flags"], idx)
        for bd, (cost, ideal, ak) in BOARDS.items():
            Q = dict(P[ak], flags=fl)
            df = run_symbol(Q, cost, ideal)
            if drop:
                df = df[~df.clf.isin(drop)]
            if k == 0 and bd == "real":
                bh = df.iloc[0][BH].to_dict()
            df = df[["clf", "action"] + KEEP].copy()
            df["board"] = bd
            df["k"] = k
            frames.append(df)
    g = pd.concat(frames, ignore_index=True)
    g["sym"] = sym
    for c in KEEP:
        g[c] = g[c].astype(np.float32)
    info = dict(n=len(P["dst"]["tb"]), days=int(P["dst"]["tb"].day.nunique()),
                fill_max=max(fill.values()) if fill else 0.0, fill_mean=float(np.mean(list(fill.values()))) if fill else 0.0,
                awake=float(P["dst"]["awake"].mean()), awake_std=float(P["std"]["awake"].mean()), **bh)
    return sym, g, info


def run(base, nproc):
    t0 = time.time()
    syms = SYMS
    if base == "1h":  # 6/1 전 1시간봉이 없는 종목은 창이 달라 뺀다(loss_run과 같음)
        syms = [s for s in SYMS if min(eth_days(s, base)) <= WIN[base]["sel"][0]]
    cal = calendar(base, syms)
    D = len(cal)
    ks = list(range(GUARD, D - GUARD + 1))
    print(base, "달력", cal[0], "~", cal[-1], "D", D, "이동", ks[0], "~", ks[-1], len(ks), "개, 종목", len(syms), flush=True)
    frames, infos = [], {}
    with Pool(min(nproc, 3)) as pool:
        for i, (sym, g, info) in enumerate(pool.imap_unordered(work, [(s, base, cal, ks) for s in syms]), 1):
            frames.append(g)
            infos[sym] = info
            if i % 12 == 0 or i == len(syms):
                print(f"{base} {i}/{len(syms)} {time.time() - t0:.0f}s", flush=True)
    g = pd.concat(frames, ignore_index=True)
    for c in ("sym", "clf", "action", "board"):
        g[c] = g[c].astype("category")
    g["k"] = g.k.astype(np.int16)
    g.to_parquet(OUT / f"z3strat_rows_{base}.parquet")
    meta = dict(base=base, calendar=[str(cal[0]), str(cal[-1])], D=D, shifts=[ks[0], ks[-1]], n_shifts=len(ks),
                n_syms=len(syms), excluded=sorted(set(SYMS) - set(syms)), n_strat=int(g.clf.astype(str).str.cat(g.action.astype(str)).nunique()),
                boards={k: dict(cost=v[0], ideal=v[1], awake=v[2]) for k, v in BOARDS.items()},
                fill_max=max(v["fill_max"] for v in infos.values()), fill_mean=float(np.mean([v["fill_mean"] for v in infos.values()])),
                sec=round(time.time() - t0, 1), per_sym=infos)
    (OUT / f"z3strat_meta_{base}.json").write_text(json.dumps(meta, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    print({k: v for k, v in meta.items() if k != "per_sym"}, flush=True)


if __name__ == "__main__":
    base = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    run(base, nproc)
