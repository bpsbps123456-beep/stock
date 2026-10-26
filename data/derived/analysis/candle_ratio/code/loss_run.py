"""loss 갈래: 원안(미리 정한 규칙)이 같은 비중 그냥 보유(static)나 무작위 타이밍(원형 이동 귀무)보다 덜 잃는가.

칸 2개만 본다(이웃 칸은 robust 갈래): 원안 = N10·count·follow·b0.5·s0.5·밤 보유, 세션 eth·rth.
- 5분봉 창(7/17~10/22): 40분봉.  1시간봉 창(6/1~10/22, 예열 5/1~): 60분봉(60분 원안 꼴).
- 판 2개: ideal(모든 봉 체결, 비용 0), real(engine.awake_mask 08~24시 KST에만 체결, 편도 eth 0.1%·rth 0.05%).
- 구간 3개: all(고르기+확인), sel, conf. 지표: 로그수익, 최대 낙폭(로그, 봉 경계 시가 기준), 보유 비율.
- 비교: 보유(bh), 같은 비중 그냥 보유(static: 구간 평균 보유 비율 x만큼 구간 첫 시가에 사서 계속 보유,
  자산 = 1 + x·(P_t/P_0·(1−c) − 1), real 판은 진입 비용 c를 한 번 낸다).
- 귀무: 비율 p를 거래일 단위로 원형 이동. 모든 종목에 같은 k, 15 ≤ k ≤ D−15.
  method 'day'(주): 공통 거래일 달력에서 (d−k) mod D일의 같은 시각 칸 값을 가져온다(그 칸이 없으면 NaN = 상태 유지).
           봉이 다 있는 종목에서는 run_grid의 np.roll(p, k×하루 봉 수)와 같다.
  method 'roll'(점검): run_grid 그대로 np.roll(p, k×하루 봉 수 중앙값). 프리·애프터 봉이 빠진 종목은 실제 이동 일수가 어긋난다.

py code/loss_run.py [프로세스 수=3] → results/loss_rows_5m.parquet, results/loss_rows_1h.parquet, results/loss_run_meta.json
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from engine import CACHE, HERE, SESS, WIN, awake_mask, load, ratio, segments, sim, tf_bars

OUT = HERE / "results"
SYMS = sorted(p.stem[:-3] for p in CACHE.glob("*_5m.parquet"))
TF = {"5m": 40, "1h": 60}
N, KIND, B, S = 10, "count", 0.5, 0.5
COST = {"eth": 0.001, "rth": 0.0005}
GUARD = 15
SEGS = ("all", "sel", "conf")


def prep(sym, base, sess):
    tb = tf_bars(load(sym, base), TF[base], sess)
    return tb[tb.day >= WIN[base]["warm"]].reset_index(drop=True)


def calendar(base):
    days = set()
    for sym in SYMS:
        for sess in ("eth", "rth"):
            days.update(prep(sym, base, sess).day.tolist())
    return sorted(days)


def static_path(px, a, b, x, c):
    """구간 봉 a..b(포함)의 시가 경로 px[a..b+1]에서 비중 x 그냥 보유의 로그수익·최대 낙폭(로그)."""
    path = px[a:b + 2] / px[a]
    e = 1.0 + x * (path * (1.0 - c) - 1.0)
    e = np.r_[1.0, e]  # 시작 자산 1(진입 직전)
    le = np.log(e)
    return float(le[-1]), float(np.min(le - np.maximum.accumulate(le)))


def work(args):
    sym, base, cal = args
    D = len(cal)
    dix = {d: i for i, d in enumerate(cal)}
    ks = list(range(GUARD, D - GUARD + 1))
    rows = []
    info = {}
    for sess in ("eth", "rth"):
        tb = prep(sym, base, sess)
        O, C, V = (tb[k].values.astype(np.float64) for k in ("O", "C", "V"))
        n = len(tb)
        lod = tb.lod.values
        seg = segments(tb.day.values, WIN[base])
        p0 = ratio(O, C, V, N, KIND)
        st = pd.DatetimeIndex(tb.start)
        sl = (np.asarray(st.hour * 60 + st.minute) - SESS[sess][0]) // TF[base]
        di = np.array([dix[d] for d in tb.day.values])
        nsl = int(sl.max()) + 1
        grid = np.full((D, nsl), -1, np.int64)
        grid[di, sl] = np.arange(n)
        assert (grid >= 0).sum() == n, (sym, base, sess, "칸 중복")
        bpd = int(pd.Series(tb.day.values).value_counts().median())
        px = np.r_[O, C[-1]]
        bounds = {}
        for nm, m in (("all", seg >= 0), ("sel", seg == 0), ("conf", seg == 1)):
            w = np.flatnonzero(m)
            assert len(w) and w[-1] - w[0] + 1 == len(w), (sym, base, sess, nm, "구간 불연속")
            bounds[nm] = (int(w[0]), int(w[-1]))
        bh = {nm: static_path(px, a, b, 1.0, 0.0) for nm, (a, b) in bounds.items()}
        bh_c = {nm: static_path(px, a, b, 1.0, COST[sess]) for nm, (a, b) in bounds.items()}
        aw = {"ideal": np.ones(n, np.bool_), "real": awake_mask(tb.start)}
        cost = {"ideal": 0.0, "real": COST[sess]}
        # 실효 이동 거리 점검(roll): 출처 봉과의 거리(일) — 과거 s/bpd, 미래 (n−s)/bpd
        info[sess] = dict(n=n, bpd=bpd, missing=int(bpd * len(np.unique(di)) - n), days=int(len(np.unique(di))),
                          roll_min_fut_days=float(min((n - (k * bpd) % n) / bpd for k in ks)),
                          roll_min_past_days=float(min(((k * bpd) % n) / bpd for k in ks)),
                          awake_frac=float(aw["real"].mean()))
        variants = [("day", 0, p0)]
        for k in ks:
            src = grid[(di - k) % D, sl]
            variants.append(("day", k, np.where(src >= 0, p0[np.maximum(src, 0)], np.nan)))
            variants.append(("roll", k, np.roll(p0, k * bpd)))
        out = np.zeros(11)
        for method, k, p in variants:
            for ver in ("ideal", "real"):
                sim(O, C, p, lod, seg, aw[ver], B, S, True, False, cost[ver], out)
                hold = {"all": (out[4] + out[5]) / (out[6] + out[7]), "sel": out[4] / out[6], "conf": out[5] / out[7]}
                ret = {"all": out[0] + out[1], "sel": out[0], "conf": out[1]}
                mdd = {"all": out[10], "sel": out[8], "conf": out[9]}
                ent = {"all": out[2] + out[3], "sel": out[2], "conf": out[3]}
                for nm in SEGS:
                    a, b = bounds[nm]
                    sr, sm = static_path(px, a, b, hold[nm], cost[ver])
                    bb = bh_c[nm] if ver == "real" else bh[nm]
                    rows.append((sym, sess, method, k, ver, nm, ret[nm], mdd[nm], hold[nm], ent[nm], sr, sm, bb[0], bb[1]))
    df = pd.DataFrame(rows, columns=["sym", "sess", "method", "k", "ver", "seg", "ret", "mdd", "hold", "ent",
                                     "st_ret", "st_mdd", "bh_ret", "bh_mdd"])
    return sym, df, info


def run(base, nproc):
    t0 = time.time()
    cal = calendar(base)
    D = len(cal)
    ks = list(range(GUARD, D - GUARD + 1))
    syms = SYMS
    if base == "1h":  # 6/1 전 1시간봉이 없는 종목은 창이 달라 뺀다
        syms = [s for s in SYMS if prep(s, base, "rth").day.min() <= WIN[base]["sel"][0]]
    print(base, "달력", cal[0], "~", cal[-1], "D", D, "이동", ks[0], "~", ks[-1], len(ks), "개", "종목", len(syms), flush=True)
    frames, infos = [], {}
    with Pool(min(nproc, 3)) as pool:
        for i, (sym, df, info) in enumerate(pool.imap_unordered(work, [(s, base, cal) for s in syms]), 1):
            frames.append(df)
            infos[sym] = info
            if i % 24 == 0 or i == len(syms):
                print(f"{base} {i}/{len(syms)} 누적 {time.time() - t0:.0f}s", flush=True)
    g = pd.concat(frames, ignore_index=True)
    for c in ("sym", "sess", "method", "ver", "seg"):
        g[c] = g[c].astype("category")
    g["k"] = g.k.astype(np.int16)
    g.to_parquet(OUT / f"loss_rows_{base}.parquet")
    excluded = sorted(set(SYMS) - set(syms))
    return dict(calendar=[str(cal[0]), str(cal[-1])], D=D, shifts=[ks[0], ks[-1]], n_shifts=len(ks),
                n_syms=len(syms), excluded=excluded, sec=round(time.time() - t0, 1),
                roll_min_fut_days=min(v[s]["roll_min_fut_days"] for v in infos.values() for s in v),
                roll_min_fut_sym=min(((v[s]["roll_min_fut_days"], f"{k}/{s}") for k, v in infos.items() for s in v))[1],
                per_sym=infos)


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    meta = {b: run(b, nproc) for b in ("5m", "1h")}
    (OUT / "loss_run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    for b, m in meta.items():
        print(b, {k: v for k, v in m.items() if k != "per_sym"})
