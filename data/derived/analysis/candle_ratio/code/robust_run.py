"""robust 갈래 ①: 원안(40분·연장·N10·count·follow·0.5/0.5·밤 보유)과 그 변형들을 144종목 × 이상적/실제 × 귀무 이동으로 잰다.

변형(5분봉 창): 원안, 체결 한 봉 지연, 묶음 시작 +5·10·20·35분, 시간봉 30·35·45·50·60분, N 8·9·11·12, 문턱 0.6/0.4.
변형(1시간봉 창, 60분 원안 꼴): 원안 60분, 한 봉 지연, N 8·9·11·12, 0.6/0.4, 120분.
실행 판: ideal(모든 봉, 비용 0) / real(engine.awake_mask 08~24시 KST에만 체결, 연장 편도 0.1%).
귀무: 비율 p를 k×하루 봉 수(중앙값)만큼 np.roll(run_grid 방식). k = 0(실제)과 15 ≤ k ≤ D−15, 모든 종목에 같은 k.

py code/robust_run.py 5m [프로세스=3] → results/robust_runs_5m.npy  float32 [종목, 변형, 판, k, 창(전체·고르기·확인), 지표 8]
                                          results/robust_runs_5m_meta.json
지표 8은 robust_common.wmetrics 설명 순서.
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from robust_common import COST_REAL, ORIG, RES, SYMS, bars_for, load, null_shifts, pos_ret, ratio, window_masks, wmetrics

VARIANTS = {
    "5m": [("원안", {}), ("지연1봉", {"delay": 1}),
           ("시작+5분", {"off": 5}), ("시작+10분", {"off": 10}), ("시작+20분", {"off": 20}), ("시작+35분", {"off": 35}),
           ("30분", {"tf": 30}), ("35분", {"tf": 35}), ("45분", {"tf": 45}), ("50분", {"tf": 50}), ("60분", {"tf": 60}),
           ("N8", {"N": 8}), ("N9", {"N": 9}), ("N11", {"N": 11}), ("N12", {"N": 12}),
           ("0.6/0.4", {"b": 0.6, "s": 0.4})],
    "1h": [("원안60분", {}), ("지연1봉", {"delay": 1}),
           ("N8", {"N": 8}), ("N9", {"N": 9}), ("N11", {"N": 11}), ("N12", {"N": 12}),
           ("0.6/0.4", {"b": 0.6, "s": 0.4}), ("120분", {"tf": 120})],
}
BASE_TF = {"5m": 40, "1h": 60}
MODES = ("ideal", "real")


def work(args):
    sym, base, ks = args
    t0 = time.time()
    var = VARIANTS[base]
    arr = np.full((len(var), 2, len(ks), 3, 8), np.nan, np.float32)
    d = load(sym, base)
    cache = {}
    m = np.zeros(8)
    for vi, (_, ch) in enumerate(var):
        tf = ch.get("tf", BASE_TF[base]); off = ch.get("off", 0)
        if (tf, off) not in cache:
            cache[(tf, off)] = bars_for(sym, base, tf, "eth", off, d=d)
        B = cache[(tf, off)]
        if len(B["O"]) < 40:
            continue
        p0 = ratio(B["O"], B["C"], B["V"], ch.get("N", ORIG["N"]), ORIG["kind"])
        masks = window_masks(B["seg"])
        for ki, k in enumerate(ks):
            p = np.roll(p0, k * B["bpd"]) if k else p0
            if ch.get("delay"):
                p = np.r_[np.full(ch["delay"], np.nan), p[:-ch["delay"]]]
            for mi, (aw, cost) in enumerate(((B["ones"], 0.0), (B["awake"], COST_REAL["eth"]))):
                pos, r, rb = pos_ret(B["O"], B["C"], p, aw, ch.get("b", ORIG["b"]), ch.get("s", ORIG["s"]), True, cost)
                for wi, mk in enumerate(masks):
                    wmetrics(r, rb, pos, mk, m)
                    arr[vi, mi, ki, wi] = m
    return sym, arr, time.time() - t0


if __name__ == "__main__":
    base = sys.argv[1]
    nproc = min(int(sys.argv[2]) if len(sys.argv) > 2 else 3, 3)
    D, shifts = null_shifts(base)
    ks = [0] + shifts
    print(base, "D", D, "귀무 이동", len(shifts), f"({shifts[0]}~{shifts[-1]})", "종목", len(SYMS), flush=True)
    out = np.full((len(SYMS), len(VARIANTS[base]), 2, len(ks), 3, 8), np.nan, np.float32)
    t0 = time.time()
    with Pool(nproc) as pool:
        for i, (sym, arr, dt) in enumerate(pool.imap_unordered(work, [(s, base, ks) for s in SYMS]), 1):
            out[SYMS.index(sym)] = arr
            if i % 12 == 0 or i == len(SYMS):
                print(f"{i}/{len(SYMS)} {sym} {dt:.1f}s 누적 {time.time() - t0:.0f}s", flush=True)
    np.save(RES / f"robust_runs_{base}.npy", out)
    meta = dict(base=base, D=D, ks=ks, syms=SYMS, variants=[v[0] for v in VARIANTS[base]],
                variant_changes=[v[1] for v in VARIANTS[base]], modes=list(MODES), windows=["full", "sel", "conf"],
                metrics=["ret_log", "mdd_log", "bh_log", "bhmdd_log", "x", "static_ret", "static_mdd", "bars"],
                cost_real=COST_REAL["eth"], orig=ORIG)
    (RES / f"robust_runs_{base}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("끝", round(time.time() - t0), "초", flush=True)
