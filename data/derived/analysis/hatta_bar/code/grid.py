"""사전 등록 격자(v3: ① 179,712칸, ② 59,904칸 — 이탈 방식 2가지 포함)를 판정 모집단(주문 가능 시간)·E2에서 잰다.
칸마다 앞(7/1~8/31)·뒤(9/1~10/28) 통계를 남기고, 앞 구간 자격 칸 중 (R1 − 변동성 맞춤 R1) 거래일 묶음 t값 1위를 고른다.
사용: py grid.py cand_alp1m 20260831   → results/grid_A.parquet, grid_B.parquet, pick.json"""
import os, sys, json, itertools, time
import numpy as np
import pandas as pd
import numba as nb
import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
GAP = 30
ELIG = dict(n=100, days=25, syms=20, share=0.15)

K_OPTS = [("k1", None, 1, 0)] + [(f"{mode}{p}_{k}", mode, k, p) for mode in ("touch", "level") for p in E.PS for k in (2, 3)]
EMA_OPTS = ("none", "now", "persist")
D_OPTS = (0.0, 0.01, 0.02, 0.03)
V_OPTS = (0.0, 2.0)
A_SHAPES = [(rm, r, m, ab) for rm in ("wick", "cpos") for r in (0.5, 0.65) for m in (0.0, 1.0, 2.0) for ab in (False, True)]
B_SHAPES = [(bl, eng) for bl in (True, False) for eng in ("basic", "body", "full", "reclaim")]
STAT_NAMES = ["n", "days", "syms", "share", "r1", "r1net", "lift", "t", "hit", "bvhit", "race", "r1e0", "hite0",
              "liftnet", "tnet"]


@nb.njit(cache=True)
def cell_stats(sel, sid, ts, dcode, front, r1, r1n, bv, bvn, hit, bvhit, race, r1e0, hite0, nd, nsym):
    """sel(정렬된 행 번호) → 겹침 30분 빼기 → 앞·뒤 통계 [2, 15]."""
    out = np.full((2, 15), np.nan)
    dsum = np.zeros((2, nd))
    dsumn = np.zeros((2, nd))
    dcnt = np.zeros((2, nd))
    symseen = np.zeros((2, nsym), np.bool_)
    acc = np.zeros((2, 12))
    last_s = -1
    last_t = -10 ** 12
    for q in range(len(sel)):
        i = sel[q]
        if sid[i] == last_s and ts[i] - last_t < 30:
            continue
        last_s = sid[i]
        last_t = ts[i]
        w = 0 if front[i] else 1
        x = r1[i] - bv[i]
        xn = r1n[i] - bvn[i]
        dsum[w, dcode[i]] += x
        dsumn[w, dcode[i]] += xn
        dcnt[w, dcode[i]] += 1
        symseen[w, sid[i]] = True
        acc[w, 0] += 1
        acc[w, 1] += r1[i]
        acc[w, 2] += r1n[i]
        acc[w, 3] += x
        acc[w, 4] += hit[i]
        acc[w, 5] += bvhit[i]
        if not np.isnan(race[i]):
            acc[w, 6] += race[i]
            acc[w, 7] += 1
        acc[w, 8] += r1e0[i]
        acc[w, 9] += hite0[i]
        acc[w, 10] += xn
    for w in range(2):
        n = acc[w, 0]
        if n == 0:
            out[w, 0] = 0
            continue
        G = 0
        mx = 0.0
        for d in range(nd):
            if dcnt[w, d] > 0:
                G += 1
                mx = max(mx, dcnt[w, d])
        ns = 0
        for s in range(nsym):
            if symseen[w, s]:
                ns += 1
        m = acc[w, 3] / n
        mn = acc[w, 10] / n
        v = 0.0
        vn = 0.0
        for d in range(nd):
            if dcnt[w, d] > 0:
                e = dsum[w, d] - m * dcnt[w, d]
                v += e * e
                en = dsumn[w, d] - mn * dcnt[w, d]
                vn += en * en
        if G > 1 and v > 0:
            se = np.sqrt(v * G / (G - 1)) / n
            t = m / se
        else:
            t = np.nan
        if G > 1 and vn > 0:
            tn = mn / (np.sqrt(vn * G / (G - 1)) / n)
        else:
            tn = np.nan
        out[w, 0] = n
        out[w, 1] = G
        out[w, 2] = ns
        out[w, 3] = mx / n
        out[w, 4] = acc[w, 1] / n
        out[w, 5] = acc[w, 2] / n
        out[w, 6] = m
        out[w, 7] = t
        out[w, 8] = acc[w, 4] / n
        out[w, 9] = acc[w, 5] / n
        out[w, 10] = acc[w, 6] / acc[w, 7] if acc[w, 7] > 0 else np.nan
        out[w, 11] = acc[w, 8] / n
        out[w, 12] = acc[w, 9] / n
        out[w, 13] = mn
        out[w, 14] = tn
    return out


def k_mask(X, opt):
    name, mode, k, p = opt
    if mode is None:
        return np.ones(len(X), bool)
    return X[f"{mode}{p}"].to_numpy() >= k


def ema_mask(X, e):
    if e == "none":
        return np.ones(len(X), bool)
    return X["stack" if e == "now" else "persist"].to_numpy().astype(bool)


def shape_mask_A(X, sh):
    rm, r, m, ab = sh
    mk = (X[rm].to_numpy() >= r) & (X["rngatr"].to_numpy() >= m)
    if ab:
        mk &= X["above"].to_numpy()
    return mk


def shape_mask_B(X, sh):
    bl, eng = sh
    mk = X[f"eng_{eng}"].to_numpy().astype(bool)
    if bl:
        mk &= X["below"].to_numpy()
    return mk


def context_ok(X, pat, sh):
    """맥락(이탈 정의에 속하는 고정·선택 조건): ①은 봉 안에서 깸, ②는 이탈 마감 선택."""
    if pat == "A":
        return X["openok"].to_numpy().astype(bool)
    bl, _ = sh
    return X["below"].to_numpy().astype(bool) if bl else np.ones(len(X), bool)


def prepare(X, split):
    X = X[X.tradable & X.sane].reset_index(drop=True)
    days = np.sort(X.day.unique())
    dcode = np.searchsorted(days, X.day.to_numpy())
    arr = dict(sid=X.sid.to_numpy().astype(np.int64), ts=X.ts.to_numpy().astype(np.int64), dcode=dcode.astype(np.int64),
               front=(X.day.to_numpy() <= split), r1=X.r1_e2.to_numpy(), r1n=X.r1net_e2.to_numpy(),
               bv=X.bv_r1_e2.to_numpy(), bvn=X.bv_r1net_e2.to_numpy(), hit=X.hit_e2.to_numpy(),
               bvhit=X.bv_hit_e2.to_numpy(), race=X.race_e2.to_numpy(), r1e0=X.r1_e0.to_numpy(),
               hite0=X.hit_e0.to_numpy(), nd=len(days), nsym=int(X.sid.max()) + 1 if len(X) else 1)
    return X, arr


def run_stats(arr, mask):
    sel = np.nonzero(mask)[0].astype(np.int64)
    return cell_stats(sel, arr["sid"], arr["ts"], arr["dcode"], arr["front"], arr["r1"], arr["r1n"], arr["bv"],
                      arr["bvn"], arr["hit"], arr["bvhit"], arr["race"], arr["r1e0"], arr["hite0"], arr["nd"], arr["nsym"])


def grid_table(path, pat, N, tol, split, mode):
    X, arr = prepare(pd.read_parquet(path), split)
    rows = []
    shapes = A_SHAPES if pat == "A" else B_SHAPES
    shm = [shape_mask_A(X, s) if pat == "A" else shape_mask_B(X, s) for s in shapes]
    ctx = [context_ok(X, pat, s) for s in shapes]
    for kopt in K_OPTS:
        km = k_mask(X, kopt)
        for e in EMA_OPTS:
            em = km & ema_mask(X, e)
            for D in D_OPTS:
                dm = em & (X["dd"].to_numpy() >= D)
                for v in V_OPTS:
                    vm = dm & (X["volr"].to_numpy() >= v) if v > 0 else dm
                    for si, sh in enumerate(shapes):
                        st = run_stats(arr, vm & ctx[si] & shm[si])
                        rows.append((mode, N, tol, kopt[0], e, D, v) + tuple(sh) + tuple(st[0]) + tuple(st[1]))
    shape_cols = ["rm", "r", "m", "above"] if pat == "A" else ["below", "eng"]
    cols = ["mode", "N", "tol", "K", "ema", "D", "v"] + shape_cols + [f"f_{s}" for s in STAT_NAMES] + [f"b_{s}" for s in STAT_NAMES]
    return pd.DataFrame(rows, columns=cols)


def eligible(df):
    return ((df.f_n >= ELIG["n"]) & (df.f_days >= ELIG["days"]) & (df.f_syms >= ELIG["syms"])
            & (df.f_share <= ELIG["share"]))


def main(cand, split):
    os.makedirs(RES, exist_ok=True)
    cdir = os.path.join(E.CACHE, cand)
    t0 = time.time()
    picks = {}
    for pat in ("A", "B"):
        parts = []
        for mode in E.MODES:
            for N in E.NS:
                for tol in E.TOLS:
                    path = os.path.join(cdir, f"{pat}_{N}_{tol}_{mode}.parquet")
                    parts.append(grid_table(path, pat, N, tol, split, mode))
                    print(pat, mode, N, tol, len(parts[-1]), f"{time.time() - t0:.0f}s", flush=True)
        g = pd.concat(parts, ignore_index=True)
        g["elig"] = eligible(g)
        g["rank"] = g.f_t.where(g.elig).rank(ascending=False, method="first")
        g.to_parquet(os.path.join(RES, f"grid_{pat}{'' if cand == 'cand_alp1m' else '_' + cand}.parquet"))
        top = g[g["rank"] == 1]
        picks[pat] = top.iloc[0].to_dict() if len(top) else None
        print(pat, "cells", len(g), "eligible", int(g.elig.sum()))
    json.dump(picks, open(os.path.join(RES, f"pick{'' if cand == 'cand_alp1m' else '_' + cand}.json"), "w"),
              ensure_ascii=False, indent=1, default=float)
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
