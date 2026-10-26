"""awake 갈래: 사용자가 실제로 쓸 수 있는 조건(한국시간 08~24시 체결 + 편도 비용)에서 남는 칸이 있는지.

부분 격자(연장 세션만, 밤 보유만)를 144종목에 돌리고, 같은 실행 안에서 공통 k 원형 이동 귀무도 함께 잰다.
- 체결: engine.awake_mask(봉 시가 시각, KST 08~24시)에만 포지션을 바꾼다. 자는 동안 신호는 다음 깨어 있는 봉 시가.
- 비용: 연장 세션 편도 0.1%(run_grid.COST_SENS["eth"]). 단서 칸(정규장)은 0.05%.
- 귀무: 비율 p를 k거래일 × 하루 봉 수(중앙값)만큼 np.roll(run_grid의 shift 방식). 모든 종목에 같은 k.
  누출을 막으려고 15 ≤ k ≤ D−15만 쓴다(D = MU 연장 60분봉 거래일 수, null_shift.n_days와 같은 정의).
- 같은 비중 그냥 보유(static): 전략의 평균 보유 비율 x만큼 처음에 사서 계속 들고 나머지는 현금.
  자산 = 1 + x·(P_t/P_0 − 1), P는 같은 시간봉 시가 경로(bh와 같은 시가→다음 시가 회계). 최대 낙폭은 봉 시가마다 잰다.

py code/awake_grid.py 5m [프로세스 수=3]   (1h도 같다: 시간봉 60·120·240만)
→ results/awake/<base>/<SYM>.npy        float32 (칸, 12)  실제(k=0), 열은 COLS12
→ results/awake/<base>/<SYM>_spec.npy   float32 (1+K, 특수칸, 12)  원안 이상적·단서 칸(정규장 60분 N5 body fade)
→ results/awake/<base>/<SYM>_meta.json  시간봉별 보유 수익·보유 낙폭·하루 봉 수
→ $AWAKE_TMP/<base>/<SYM>.npy           float32 (1+K, 8, 칸)  귀무 포함 전체(열은 COLS8). 크기 때문에 결과 폴더 밖에 둔다.
→ results/awake_cells_<base>.parquet, results/awake_shifts_<base>.json
"""
from __future__ import annotations

import itertools
import json
import os
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from engine import BS, CACHE, HERE, SS, WIN, awake_mask, bh, load, ratio, segments, sim, tf_bars  # noqa: E402
from run_grid import COST_SENS  # noqa: E402

TFS = {"5m": list(range(10, 121, 10)) + [160, 240], "1h": [60, 120, 240]}
NS7 = [3, 5, 8, 10, 12, 15, 20]
KINDS = ["count", "body"]
DIRS = ["follow", "fade"]
GUARD = 15
KEY = ["sess", "tf", "N", "ratio", "dir", "b", "s"]
COLS12 = ["ret_sel", "ret_conf", "x_sel", "x_conf", "mdd_sel", "mdd_conf", "mdd_all",
          "smdd_sel", "smdd_conf", "smdd_all", "ent_sel", "ent_conf"]  # ret·mdd(전략)는 로그, smdd(static)는 단순
COLS8 = ["ret_sel", "ret_conf", "x_sel", "x_conf", "mdd_sel", "mdd_conf", "smdd_sel", "smdd_conf"]
I8 = [COLS12.index(c) for c in COLS8]
ORIG_TF = {"5m": 40, "1h": 60}
# 특수칸: (이름, 세션, 시간봉, N, 비율, follow, b, s, awake, 비용)
def specials(base):
    t = ORIG_TF[base]
    return [("orig_ideal0", "eth", t, 10, "count", True, 0.5, 0.5, False, 0.0),
            ("orig_awake_c", "eth", t, 10, "count", True, 0.5, 0.5, True, COST_SENS["eth"]),
            ("lead_awake_c", "rth", 60, 5, "body", False, 0.5, 0.5, True, COST_SENS["rth"]),
            ("lead_awake0", "rth", 60, 5, "body", False, 0.5, 0.5, True, 0.0),
            ("lead_ideal0", "rth", 60, 5, "body", False, 0.5, 0.5, False, 0.0)]

TMP = pathlib.Path(os.environ.get("AWAKE_TMP", str(HERE / "results" / "awake_tmp")))
OUT = HERE / "results" / "awake"


def cell_table(base):
    rows = list(itertools.product(["eth"], TFS[base], NS7, KINDS, DIRS, BS, SS))
    return pd.DataFrame(rows, columns=KEY)


def n_days(base):
    tb = tf_bars(load("MU", base), 60, "eth")
    return int(len(np.unique(tb.day.values[tb.day.values >= WIN[base]["warm"]])))


def shift_list(base):
    D = n_days(base)
    return D, list(range(GUARD, D - GUARD + 1))


@njit(cache=True)
def static_mdd(P, x):
    """P: 구간 시작을 1로 둔 가격 경로(봉 시가들 + 마지막). 자산 1 + x(P−1)의 최대 낙폭(단순)."""
    if np.isnan(x):
        return np.nan
    peak = 1.0
    mdd = 0.0
    for j in range(len(P)):
        e = 1.0 + x * (P[j] - 1.0)
        if e > peak:
            peak = e
        dd = e / peak - 1.0
        if dd < mdd:
            mdd = dd
    return mdd


@njit(cache=True)
def run_one(o, c, p, lod, seg, awake, b, s, follow, cost, P0, P1, PA, out, row):
    sim(o, c, p, lod, seg, awake, b, s, follow, False, cost, out)
    xs = out[4] / out[6] if out[6] > 0 else np.nan
    xc = out[5] / out[7] if out[7] > 0 else np.nan
    xa = (out[4] + out[5]) / (out[6] + out[7]) if out[6] + out[7] > 0 else np.nan
    row[0] = out[0]; row[1] = out[1]; row[2] = xs; row[3] = xc
    row[4] = out[8]; row[5] = out[9]; row[6] = out[10]
    row[7] = static_mdd(P0, xs); row[8] = static_mdd(P1, xc); row[9] = static_mdd(PA, xa)
    row[10] = out[2]; row[11] = out[3]


@njit(cache=True)
def series_kernel(o, c, p, lod, seg, awake, bs, ss, cost, P0, P1, PA, res):
    """한 비율 계열의 방향 2 × 문턱 16칸(밤 보유). res[32, 12]."""
    out = np.zeros(11)
    r = 0
    for fi in range(2):
        for bi in range(len(bs)):
            for si in range(len(ss)):
                run_one(o, c, p, lod, seg, awake, bs[bi], ss[si], fi == 0, cost, P0, P1, PA, out, res[r])
                r += 1


def paths(O, C, seg):
    """bh 회계(시가→다음 시가, 마지막 봉은 시가→종가)의 구간별 가격 경로. 시작점 1 포함."""
    r = np.r_[np.log(O[1:] / O[:-1]), np.log(C[-1] / O[-1])]
    def mk(m):
        return np.r_[1.0, np.exp(np.cumsum(r[m]))]
    return mk(seg == 0), mk(seg == 1), mk(seg >= 0)


def bars(d, base, tf, sess):
    tb = tf_bars(d, tf, sess)
    tb = tb[tb.day >= WIN[base]["warm"]].reset_index(drop=True)
    if len(tb) < 40:
        return None
    O, C, V = (tb[k].values.astype(np.float64) for k in ("O", "C", "V"))
    seg = segments(tb.day.values, WIN[base])
    bpd = int(pd.Series(tb.day.values).value_counts().median())
    return dict(O=O, C=C, V=V, lod=tb.lod.values, seg=seg, aw=awake_mask(tb.start), bpd=bpd,
                ideal=np.ones(len(tb), np.bool_), P=paths(O, C, seg), n=len(tb))


def work(args):
    sym, base, ks = args
    f = OUT / base / f"{sym}.npy"
    if f.exists() and (TMP / base / f"{sym}.npy").exists():
        return sym, "skip", 0.0
    t0 = time.time()
    d = load(sym, base)
    bs, ss = np.array(BS), np.array(SS)
    cost = COST_SENS["eth"]
    tfs = TFS[base]
    nper = len(NS7) * len(KINDS) * 32
    M = len(tfs) * nper
    K1 = len(ks)
    allk = np.full((K1, 8, M), np.nan, np.float32)
    act = np.full((M, 12), np.nan, np.float32)
    meta = dict(sym=sym, base=base, ks=ks, tf={})
    res = np.zeros((32, 12))
    for ti, tf in enumerate(tfs):
        B = bars(d, base, tf, "eth")
        if B is None:
            continue
        bh_s, bh_c = bh(B["O"], B["C"], B["seg"])
        meta["tf"][str(tf)] = dict(bh_sel=bh_s, bh_conf=bh_c, bpd=B["bpd"], n=B["n"],
                                   bh_mdd_sel=static_mdd(B["P"][0], 1.0), bh_mdd_conf=static_mdd(B["P"][1], 1.0),
                                   bh_mdd_all=static_mdd(B["P"][2], 1.0), bh_all=float(np.log(B["P"][2][-1])))
        for ni, n in enumerate(NS7):
            for ki, kind in enumerate(KINDS):
                p0 = ratio(B["O"], B["C"], B["V"], n, kind)
                lo = ti * nper + (ni * len(KINDS) + ki) * 32
                for kk, k in enumerate(ks):
                    p = np.roll(p0, k * B["bpd"]) if k else p0
                    series_kernel(B["O"], B["C"], p, B["lod"], B["seg"], B["aw"], bs, ss, cost, *B["P"], res)
                    allk[kk, :, lo:lo + 32] = res[:, I8].T
                    if k == 0:
                        act[lo:lo + 32] = res
    # 특수칸
    sp = specials(base)
    spec = np.full((K1, len(sp), 12), np.nan, np.float32)
    cache = {}
    out = np.zeros(11)
    row = np.zeros(12)
    for si_, (nm, sess, tf, n, kind, fol, b, s, aw, cst) in enumerate(sp):
        key = (sess, tf)
        if key not in cache:
            cache[key] = bars(d, base, tf, sess)
        B = cache[key]
        if B is None:
            continue
        if sess == "rth":
            bh_s, bh_c = bh(B["O"], B["C"], B["seg"])
            meta["rth60"] = dict(bh_sel=bh_s, bh_conf=bh_c, bpd=B["bpd"], n=B["n"],
                                 bh_mdd_sel=static_mdd(B["P"][0], 1.0), bh_mdd_conf=static_mdd(B["P"][1], 1.0),
                                 bh_mdd_all=static_mdd(B["P"][2], 1.0), bh_all=float(np.log(B["P"][2][-1])))
        p0 = ratio(B["O"], B["C"], B["V"], n, kind)
        for kk, k in enumerate(ks):
            p = np.roll(p0, k * B["bpd"]) if k else p0
            run_one(B["O"], B["C"], p, B["lod"], B["seg"], B["aw"] if aw else B["ideal"], b, s, fol, cst,
                    *B["P"], out, row)
            spec[kk, si_] = row
    np.save(OUT / base / f"{sym}_spec.npy", spec)
    (OUT / base / f"{sym}_meta.json").write_text(json.dumps(meta, default=float), encoding="utf-8")
    np.save(TMP / base / f"{sym}.npy", allk)
    np.save(f, act)  # 마지막에 써서 중단되면 다시 돈다
    return sym, M, time.time() - t0


if __name__ == "__main__":
    base = sys.argv[1]
    nproc = min(int(sys.argv[2]) if len(sys.argv) > 2 else 3, 3)
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    (OUT / base).mkdir(parents=True, exist_ok=True)
    (TMP / base).mkdir(parents=True, exist_ok=True)
    D, nulls = shift_list(base)
    ks = [0] + nulls
    cell_table(base).to_parquet(HERE / "results" / f"awake_cells_{base}.parquet")
    (HERE / "results" / f"awake_shifts_{base}.json").write_text(
        json.dumps(dict(D=D, guard=GUARD, ks=ks, specials=[s[0] for s in specials(base)], tmp=str(TMP)), ensure_ascii=False),
        encoding="utf-8")
    syms = only or sorted(p.name[: -len(f"_{base}.parquet")] for p in CACHE.glob(f"*_{base}.parquet"))
    print(base, "D", D, "이동", len(nulls), f"({nulls[0]}~{nulls[-1]})", "칸", len(cell_table(base)), "종목", len(syms), flush=True)
    t0 = time.time()
    with Pool(nproc) as pool:
        for i, (sym, n, dt) in enumerate(pool.imap_unordered(work, [(s, base, ks) for s in syms]), 1):
            print(f"{i}/{len(syms)} {sym} {n} {dt:.1f}s 누적 {time.time() - t0:.0f}s", flush=True)
