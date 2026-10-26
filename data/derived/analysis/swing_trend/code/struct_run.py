"""struct 갈래(BRIEF 질문 1): 정말 1주~1달짜리 중단기 추세가 있는가.

- 사후 지그재그(θ 10·15·20%, k 3·4·5 × ATR20)로 구간을 나눠 길이·크기·몫을 잰다(사후 값, 설명용).
- 귀무: 같은 종목의 일봉을 구간 안에서 섞은 가짜 가격 200개(일 단위 iid 섞기)와 200개(5일 묶음 섞기).
  봉은 전일 종가 대비 비율(O/C₋₁, H/C₋₁, L/C₋₁, C/C₋₁)로 옮겨 시가 갭·ATR·구간 총수익을 그대로 둔다.
  한 번의 섞기 순서를 모든 종목에 같이 쓴다(종목끼리의 동조를 남겨 횡단 중앙의 귀무 분포가 맞게).
- 예열 60거래일(실제 봉), 구간을 다 덮는 종목만.
- 시장 공통성: 실제 반전점 날짜가 종목끼리 몰리는지(±2일 동시 반전), SOXX·QQQ 반전점(±3일)과 겹치는지.
  귀무는 종목마다 반전점 날짜를 구간 안에서 따로 원형 이동(200회).
출력: results/struct_*.csv, results/struct_summary.json
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from swing import HERE, PERIODS, atr, load, zigzag  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = HERE / "results"
WARM = 60
B = 200
ZZ = [("P10", 0.10, 0.0), ("P15", 0.15, 0.0), ("P20", 0.20, 0.0), ("A3", 0.0, 3.0), ("A4", 0.0, 4.0), ("A5", 0.0, 5.0)]
ETF = {"SPY", "QQQ", "SOXX", "SMH", "IGV", "IWM", "XBI", "ARKG"}
LMAX = 150
NM = 32
SEED = 20261025
# 지표 칸
(N_SEG, S_LEN, N14, N522, N23, MV14, MV522, MV23, N_PIV, N_SUD, N_JMP, N_GAP, F3, F3SZ,
 NH5, FW5, NH10, FW10, NH20, FW20, N_UP, LEN_UP, MV_UP, N_DAY, N_SUDDAY,
 NS_SEG, NS_LEN, NS_MV, LAG, N_LAG, REM, _SP) = range(NM)


@njit(cache=True)
def metrics(O, C, a20, pi, pc, pt, s0, s1, out, hist):
    npv = len(pi)
    for i in range(npv - 1):
        a = pi[i]; b = pi[i + 1]
        if a < s0 or b > s1:
            continue
        ln = b - a
        sz = abs(np.log(C[b] / C[a]))
        out[0] += 1; out[1] += ln
        if ln <= 4:
            out[2] += 1; out[5] += sz
        elif ln <= 22:
            out[3] += 1; out[6] += sz
        else:
            out[4] += 1; out[7] += sz
        hl = ln if ln < 150 else 150
        hist[0, hl] += 1; hist[1, hl] += sz
        dr = pt[i]
        if dr == 1:
            out[20] += 1; out[21] += ln; out[22] += sz
        if ln >= 5:
            out[12] += dr * np.log(C[a + 3] / C[a]); out[13] += sz
        # 시작 반전이 갑작스러운가
        if a + 1 <= s1:
            jump = dr * (C[a + 1] - C[a]) >= 2.0 * a20[a]
            gap = dr * (O[a + 1] - C[a]) >= a20[a]
            if jump or gap:
                out[25] += 1; out[26] += ln; out[27] += sz
        # 확정 뒤 남은 몫(구간 끝 반전점까지)
        c = pc[i]
        if c <= b:
            out[30] += dr * np.log(C[b] / C[c])
    for i in range(npv):
        p = pi[i]
        if p < s0 or p + 1 > s1:
            continue
        dr = pt[i]
        jump = dr * (C[p + 1] - C[p]) >= 2.0 * a20[p]
        gap = dr * (O[p + 1] - C[p]) >= a20[p]
        out[8] += 1
        if jump or gap:
            out[9] += 1
        if jump:
            out[10] += 1
        if gap:
            out[11] += 1
        out[28] += pc[i] - p; out[29] += 1
    for i in range(npv):
        c = pc[i]
        if c < s0:
            continue
        dr = pt[i]
        if c + 5 <= s1:
            out[14] += 1; out[15] += dr * np.log(C[c + 5] / C[c])
        if c + 10 <= s1:
            out[16] += 1; out[17] += dr * np.log(C[c + 10] / C[c])
        if c + 20 <= s1:
            out[18] += 1; out[19] += dr * np.log(C[c + 20] / C[c])
    for t in range(s0 + 1, s1 + 1):
        out[23] += 1
        if abs(C[t] - C[t - 1]) >= 2.0 * a20[t - 1] or abs(O[t] - C[t - 1]) >= a20[t - 1]:
            out[24] += 1


@njit(cache=True)
def run_sym(O, H, L, C, w, perms, ths, ks, out, hist):
    P = perms.shape[0]; D = perms.shape[1]; n = len(C)
    empty = np.zeros(0)
    O2 = O.copy(); H2 = H.copy(); L2 = L.copy(); C2 = C.copy()
    for b in range(P):
        for t in range(D):
            src = w + perms[b, t]
            base = C[src - 1]
            cp = C2[w + t - 1]
            O2[w + t] = cp * O[src] / base
            H2[w + t] = cp * H[src] / base
            L2[w + t] = cp * L[src] / base
            C2[w + t] = cp * C[src] / base
        a20 = atr(H2, L2, C2, 20)
        for z in range(len(ths)):
            if ks[z] > 0:
                absv = ks[z] * a20
            else:
                absv = empty
            st, pi, pc, pt = zigzag(C2, ths[z], absv)
            metrics(O2, C2, a20, pi, pc, pt, w, n - 1, out[b, z], hist[b, z])


def block_perm(D, bl, rng):
    off = int(rng.integers(0, bl))
    edges = sorted(set([0, D] + list(range(off, D, bl))))
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[k] for k in order])


def acf_vr(R):
    """R: (D, S) 로그 일수익. 반환 acf (20, S), vr (3, S) for q 5·10·20."""
    x = R - R.mean(0)
    den = (x ** 2).sum(0)
    D = len(R)
    ac = np.stack([(x[:-k] * x[k:]).sum(0) / den for k in range(1, 21)])
    cs = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(x, 0)])
    var1 = den / D
    vr = []
    for q in (5, 10, 20):
        sq = cs[q:] - cs[:-q]
        vr.append((sq ** 2).mean(0) / (q * var1))
    return ac, np.stack(vr)


def eligible(G, per):
    a, b = PERIODS[per]
    spy = G["SPY"]
    dP = spy.date[(spy.date >= a) & (spy.date <= b)].values
    ok = {}
    for s, g in G.items():
        d = g.date.values
        p = int(np.searchsorted(d, dP[0]))
        if p >= len(d) or d[p] != dP[0] or p < WARM:
            continue
        seg = d[p:p + len(dP)]
        if len(seg) != len(dP) or not (seg == dP).all():
            continue
        ok[s] = (p, p + len(dP) - 1)
    return dP, ok


def qsum(real, null):
    nm = np.nanmedian(null)
    n = int(np.sum(np.isfinite(null)))
    return dict(real=real, null_med=nm, null_p05=np.nanpercentile(null, 5), null_p95=np.nanpercentile(null, 95),
                p_up=(1 + np.sum(null >= real)) / (1 + n), p_dn=(1 + np.sum(null <= real)) / (1 + n),
                ratio=real / nm if np.isfinite(nm) and nm != 0 else np.nan)


def div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def derived(A):
    """A (..., NM) 합계 → 지표 사전. 종목별이면 종목별 값, 합산이면 합산 값."""
    mvt = A[..., MV14] + A[..., MV522] + A[..., MV23]
    nseg = A[..., N_SEG]
    return {
        "구간수": nseg,
        "평균길이": div(A[..., S_LEN], nseg),
        "평균크기": div(mvt, nseg),
        "몫_1_4일": div(A[..., MV14], mvt),
        "몫_5_22일": div(A[..., MV522], mvt),
        "몫_23일이상": div(A[..., MV23], mvt),
        "개수몫_5_22일": div(A[..., N522], nseg),
        "갑작반전비율": div(A[..., N_SUD], A[..., N_PIV]),
        "큰봉반전비율": div(A[..., N_JMP], A[..., N_PIV]),
        "갭반전비율": div(A[..., N_GAP], A[..., N_PIV]),
        "아무날갑작비율": div(A[..., N_SUDDAY], A[..., N_DAY]),
        "첫3일몫": div(A[..., F3], A[..., F3SZ]),
        "확정뒤5일": div(A[..., FW5], A[..., NH5]),
        "확정뒤10일": div(A[..., FW10], A[..., NH10]),
        "확정뒤20일": div(A[..., FW20], A[..., NH20]),
        "상승평균길이": div(A[..., LEN_UP], A[..., N_UP]),
        "하락평균길이": div(A[..., S_LEN] - A[..., LEN_UP], nseg - A[..., N_UP]),
        "상승평균크기": div(A[..., MV_UP], A[..., N_UP]),
        "하락평균크기": div(mvt - A[..., MV_UP], nseg - A[..., N_UP]),
        "갑작시작_평균크기": div(A[..., NS_MV], A[..., NS_SEG]),
        "보통시작_평균크기": div(mvt - A[..., NS_MV], nseg - A[..., NS_SEG]),
        "갑작시작_평균길이": div(A[..., NS_LEN], A[..., NS_SEG]),
        "보통시작_평균길이": div(A[..., S_LEN] - A[..., NS_LEN], nseg - A[..., NS_SEG]),
        "확정지연일": div(A[..., LAG], A[..., N_LAG]),
        "확정뒤남은몫": div(A[..., REM], mvt),
    }


def hist_rows(per, zname, H):
    """H (P, 2, LMAX+1) 합산 길이 분포 → 칸별 개수몫·움직임몫, 중앙 길이."""
    bins = [(1, 4), (5, 10), (11, 22), (23, 44), (45, LMAX)]
    rows = []
    cnt, mv = H[:, 0, :], H[:, 1, :]
    ctot, mtot = cnt.sum(1), mv.sum(1)
    cum = np.cumsum(cnt, 1)
    med = np.array([np.searchsorted(cum[i], ctot[i] / 2.0) if ctot[i] > 0 else np.nan for i in range(len(cnt))])
    for lab, sl in [("실제", slice(0, 1)), ("섞기", slice(1, B + 1)), ("5일묶음", slice(B + 1, 2 * B + 1))]:
        r = dict(period=per, zz=zname, data=lab, 중앙길이=float(np.nanmedian(med[sl])), 구간수=float(np.median(ctot[sl])))
        for lo, hi in bins:
            c = cnt[sl, lo:hi + 1].sum(1) / np.maximum(ctot[sl], 1)
            m = mv[sl, lo:hi + 1].sum(1) / np.maximum(mtot[sl], 1e-12)
            r[f"개수_{lo}_{hi}"] = float(np.median(c))
            r[f"움직임_{lo}_{hi}"] = float(np.median(m))
        rows.append(r)
    return rows


def window(g, j0, j1):
    w0 = j0 - WARM
    return tuple(np.ascontiguousarray(g[k].values[w0:j1 + 1].astype(np.float64)) for k in ("O", "H", "L", "C"))


def real_pivots(O, H, L, C, th, k):
    a20 = atr(H, L, C, 20)
    st, pi, pc, pt = zigzag(C, th, k * a20 if k > 0 else np.zeros(0))
    m = pi >= WARM
    pi, pc, pt = pi[m], pc[m], pt[m]
    sud = np.zeros(len(pi), bool)
    for i, p in enumerate(pi):
        if p + 1 < len(C):
            sud[i] = (pt[i] * (C[p + 1] - C[p]) >= 2 * a20[p]) or (pt[i] * (O[p + 1] - C[p]) >= a20[p])
    return pi - WARM, pc - WARM, pt, sud


def week_frac(I, wid, nw):
    """주별 반전 종목 비율 (2, nw): 그 주에 해당 종류 반전이 1번 이상인 종목 비율."""
    S = I.shape[1]
    out = np.zeros((2, nw))
    for ty in range(2):
        W = np.zeros((S, nw), np.int32)
        np.add.at(W.T, wid, I[ty].T.astype(np.int32))
        out[ty] = (W > 0).mean(0)
    return out


def cluster(piv, D, ref, rng, wid, win=2, rwin=3):
    """piv: {sym: (days, types, sudden)} 주식만. ref: {"SOXX": (days, types), "QQQ": ...}.
    반환: 동시 반전 지수(실제·귀무), SOXX·QQQ 겹침(실제·귀무), 갑작/보통 반전의 겹침."""
    syms = list(piv)
    S = len(syms)
    I = np.zeros((2, S, D), np.int8)  # 0 저점, 1 고점
    for i, s in enumerate(syms):
        d, t, _ = piv[s]
        for dd, tt in zip(d, t):
            I[0 if tt == 1 else 1, i, dd] = 1

    def coinc(I):
        tot = 0.0; n = 0
        per_piv = []
        for ty in range(2):
            M = I[ty].astype(np.int32)
            cs = np.concatenate([np.zeros((S, 1), np.int32), np.cumsum(M, 1)], 1)
            lo = np.clip(np.arange(D) - win, 0, D); hi = np.clip(np.arange(D) + win + 1, 0, D)
            A = (cs[:, hi] - cs[:, lo]) > 0
            col = A.sum(0)
            ss, dd = np.nonzero(M)
            v = (col[dd] - 1) / (S - 1)
            per_piv.append((ty, ss, dd, v))
            tot += v.sum(); n += len(v)
        return tot / max(n, 1), per_piv

    def overlap(I, rd, rt):
        hit = 0; n = 0
        for ty, code in ((0, 1), (1, -1)):
            rdays = rd[rt == code]
            ok = np.zeros(D, bool)
            for x in rdays:
                ok[max(0, x - rwin):min(D, x + rwin + 1)] = True
            M = I[ty]
            hit += int((M & ok[None, :]).sum()); n += int(M.sum())
        return hit / max(n, 1)

    real_ci, per_piv = coinc(I)
    res = dict(n_stock=S, n_piv=int(I.sum()), ci_real=real_ci)
    nw = int(wid.max()) + 1
    wf = week_frac(I, wid, nw)
    wmax_real = wf.max(1); wtop3_real = np.sort(wf, 1)[:, -3:].mean(1)
    nul_wmax, nul_wtop3 = [], []
    nul_ci, nul_ov = [], {k: [] for k in ref}
    for b in range(B):
        sh = rng.integers(0, D, S)
        J = np.empty_like(I)
        for i in range(S):
            J[:, i] = np.roll(I[:, i], sh[i], axis=1)
        nul_ci.append(coinc(J)[0])
        wfn = week_frac(J, wid, nw)
        nul_wmax.append(wfn.max(1)); nul_wtop3.append(np.sort(wfn, 1)[:, -3:].mean(1))
        for k, (rd, rt) in ref.items():
            nul_ov[k].append(overlap(J, rd, rt))
    nul_ci = np.array(nul_ci)
    nul_wmax = np.array(nul_wmax); nul_wtop3 = np.array(nul_wtop3)
    for ty, nm_ in ((0, "저점"), (1, "고점")):
        res.update({f"주최대_{nm_}_real": float(wmax_real[ty]), f"주최대_{nm_}_null": float(nul_wmax[:, ty].mean()),
                    f"주최대_{nm_}_null_p95": float(np.percentile(nul_wmax[:, ty], 95)),
                    f"주상위3_{nm_}_real": float(wtop3_real[ty]), f"주상위3_{nm_}_null": float(nul_wtop3[:, ty].mean()),
                    f"주상위3_{nm_}_null_p95": float(np.percentile(nul_wtop3[:, ty], 95))})
    res.update(ci_null=float(nul_ci.mean()), ci_null_p95=float(np.percentile(nul_ci, 95)), ci_p=float((1 + (nul_ci >= real_ci).sum()) / (1 + B)))
    for k, (rd, rt) in ref.items():
        r = overlap(I, rd, rt)
        nn = np.array(nul_ov[k])
        res.update({f"{k}_n_piv": int(len(rd)), f"{k}_ov_real": r, f"{k}_ov_null": float(nn.mean()),
                    f"{k}_ov_null_p95": float(np.percentile(nn, 95)), f"{k}_ov_p": float((1 + (nn >= r).sum()) / (1 + B))})
    # 갑작 반전 vs 보통 반전: 동시 반전 지수, SOXX 겹침
    sud = np.zeros((2, S, D), bool)
    for i, s in enumerate(syms):
        d, t, u = piv[s]
        for dd, tt, uu in zip(d, t, u):
            sud[0 if tt == 1 else 1, i, dd] = uu
    vs, vn = [], []
    for ty, ss, dd, v in per_piv:
        m = sud[ty, ss, dd]
        vs.append(v[m]); vn.append(v[~m])
    vs, vn = np.concatenate(vs), np.concatenate(vn)
    res.update(ci_sudden=float(vs.mean()) if len(vs) else np.nan, ci_normal=float(vn.mean()) if len(vn) else np.nan,
               n_sudden=int(len(vs)), n_normal=int(len(vn)))
    for k, (rd, rt) in ref.items():
        Is = (I.astype(bool) & sud).astype(np.int8)
        In = (I.astype(bool) & ~sud).astype(np.int8)
        res[f"{k}_ov_sudden"] = overlap(Is, rd, rt)
        res[f"{k}_ov_normal"] = overlap(In, rd, rt)
    return res, I, syms


def main(periods=("P1", "P2", "Y25", "Y24", "Y23")):
    OUT.mkdir(exist_ok=True)
    G = load()
    ths = np.array([z[1] for z in ZZ]); ks = np.array([z[2] for z in ZZ])
    rows_stat, rows_hist, rows_vr, rows_cl, rows_wk, rows_piv, rows_sym = [], [], [], [], [], [], []
    for pi_, per in enumerate(periods):
        t0 = time.time()
        dP, ok = eligible(G, per)
        D = len(dP)
        syms = sorted(ok)
        rng = np.random.default_rng(SEED + pi_)
        perms = np.vstack([np.arange(D)] + [rng.permutation(D) for _ in range(B)] + [block_perm(D, 5, rng) for _ in range(B)]).astype(np.int64)
        P = len(perms)
        Hpool = np.zeros((P, len(ZZ), 2, LMAX + 1))
        A = np.zeros((len(syms), P, len(ZZ), NM))
        R = np.zeros((D, len(syms)))
        for si, s in enumerate(syms):
            j0, j1 = ok[s]
            O, H, L, C = window(G[s], j0, j1)
            run_sym(O, H, L, C, WARM, perms, ths, ks, A[si], Hpool)
            R[:, si] = np.log(C[WARM:] / C[WARM - 1:-1])
        # --- 지그재그 구조: 합산과 종목 횡단 중앙
        is_etf = np.array([s in ETF for s in syms])
        groups = (("전체", np.ones(len(syms), bool)), ("주식", ~is_etf), ("ETF", is_etf))
        for zi, (zn, _, _) in enumerate(ZZ):
            persym_all = derived(A[:, :, zi, :])
            for grp, gm in groups:
                pooled = derived(A[gm, :, zi, :].sum(0))  # (P,)
                persym = {k: v[gm] for k, v in persym_all.items()}  # (S, P)
                for mname in pooled:
                    pv = pooled[mname]
                    sv = persym[mname]
                    with np.errstate(all="ignore"):
                        med = np.array([np.nanmedian(sv[:, b]) if np.isfinite(sv[:, b]).any() else np.nan for b in range(P)])
                    for lab, sl in (("섞기", slice(1, B + 1)), ("5일묶음", slice(B + 1, 2 * B + 1))):
                        nul_sym_med = np.nanmedian(sv[:, sl], 1)
                        nul_sym_p95 = np.nanpercentile(sv[:, sl], 95, 1)
                        nul_sym_p05 = np.nanpercentile(sv[:, sl], 5, 1)
                        valid = np.isfinite(sv[:, 0]) & np.isfinite(nul_sym_med)
                        r = dict(period=per, group=grp, n_sym=int(gm.sum()), D=D, zz=zn, metric=mname, null=lab)
                        q = qsum(pv[0], pv[sl]); r.update({f"합산_{k}": v for k, v in q.items()})
                        q = qsum(med[0], med[sl]); r.update({f"중앙_{k}": v for k, v in q.items()})
                        r.update(종목_유효=int(valid.sum()),
                                 종목_귀무중앙초과=float((sv[valid, 0] > nul_sym_med[valid]).mean()) if valid.any() else np.nan,
                                 종목_귀무95초과=float((sv[valid, 0] > nul_sym_p95[valid]).mean()) if valid.any() else np.nan,
                                 종목_귀무5미만=float((sv[valid, 0] < nul_sym_p05[valid]).mean()) if valid.any() else np.nan)
                        rows_stat.append(r)
            rows_hist += hist_rows(per, zn, Hpool[:, zi])
            # 종목별 실제 값(설명용)
            for si, s in enumerate(syms):
                d = {k: float(v[si, 0]) for k, v in persym_all.items()}
                d.update(period=per, zz=zn, sym=s)
                rows_sym.append(d)
        # --- 분산비·자기상관
        acs, vrs = [], []
        for b in range(P):
            ac, vr = acf_vr(R[perms[b]])
            acs.append(ac); vrs.append(vr)
        acs = np.array(acs); vrs = np.array(vrs)  # (P, 20, S), (P, 3, S)
        feats = {f"VR{q}": vrs[:, i, :] for i, q in enumerate((5, 10, 20))}
        feats.update({f"ACF{k}": acs[:, k - 1, :] for k in range(1, 21)})
        feats["ACF합1_5"] = acs[:, 0:5, :].sum(1)
        feats["ACF합6_20"] = acs[:, 5:20, :].sum(1)
        for (fname, X0), (grp, gm) in [(f, g) for f in feats.items() for g in groups]:
            X = X0[:, gm]
            for lab, sl in (("섞기", slice(1, B + 1)), ("5일묶음", slice(B + 1, 2 * B + 1))):
                med = np.median(X, 1)
                r = dict(period=per, group=grp, n_sym=int(gm.sum()), D=D, metric=fname, null=lab)
                r.update({f"중앙_{k}": v for k, v in qsum(med[0], med[sl]).items()})
                nm = np.median(X[sl], 0); p95 = np.percentile(X[sl], 95, 0); p05 = np.percentile(X[sl], 5, 0)
                r.update(종목_귀무중앙초과=float((X[0] > nm).mean()), 종목_귀무95초과=float((X[0] > p95).mean()),
                         종목_귀무5미만=float((X[0] < p05).mean()))
                for etf in ("SPY", "QQQ", "SOXX", "SMH", "IGV"):
                    if etf in syms:
                        i = syms.index(etf)
                        r[f"{etf}_실제"] = float(X0[0, i]); r[f"{etf}_귀무p95"] = float(np.percentile(X0[sl, i], 95)); r[f"{etf}_귀무p05"] = float(np.percentile(X0[sl, i], 5))
                rows_vr.append(r)
        # --- 실제 반전점과 시장 공통성
        stocks = [s for s in syms if s not in ETF]
        wk = pd.to_datetime(pd.Series(dP)).dt.isocalendar()
        wkey = (wk.year.astype(str) + "-W" + wk.week.astype(str).str.zfill(2)).values
        wid = pd.factorize(wkey)[0]
        soxx_c = None
        ref2 = {}
        for s in ("SOXX", "QQQ"):
            j0, j1 = ok[s]
            O, H, L, C = window(G[s], j0, j1)
            d, c, t, u = real_pivots(O, H, L, C, 0.0, 2.0)
            ref2[s + "_A2"] = (d, t)
        for zi, (zn, th, k) in enumerate(ZZ):
            piv = {}
            ref = {}
            for s in syms:
                j0, j1 = ok[s]
                O, H, L, C = window(G[s], j0, j1)
                d, c, t, u = real_pivots(O, H, L, C, th, k)
                for dd, cc, tt, uu in zip(d, c, t, u):
                    rows_piv.append(dict(period=per, zz=zn, sym=s, date=dP[dd], type="저점" if tt == 1 else "고점",
                                         confirm=dP[cc] if cc < D else None, lag=int(cc - dd), sudden=bool(uu)))
                if s in ("SOXX", "QQQ"):
                    ref[s] = (d, t)
                if s == "SOXX":
                    soxx_c = C[WARM - 1:]
                if s not in ETF:
                    piv[s] = (d, t, u)
            ref.update(ref2)
            res, I, ssy = cluster(piv, D, ref, np.random.default_rng(SEED + 100 * pi_ + zi), wid)
            res.update(period=per, zz=zn, D=D)
            rows_cl.append(res)
            # 주별
            dfw = pd.DataFrame({"wk": wkey, "i": np.arange(D)})
            S = len(ssy)
            for w_, gi in dfw.groupby("wk").i:
                ii = gi.values
                lo = I[0][:, ii].any(1).mean(); hi = I[1][:, ii].any(1).mean()
                sx = float(soxx_c[ii[-1] + 1] / soxx_c[ii[0]] - 1) if soxx_c is not None else np.nan
                rows_wk.append(dict(period=per, zz=zn, week=w_, start=dP[ii[0]], end=dP[ii[-1]], 저점비율=lo, 고점비율=hi, SOXX주수익=sx, n_stock=S))
        print(per, D, len(syms), f"{time.time() - t0:.1f}s", flush=True)
    st = pd.DataFrame(rows_stat); st.to_csv(OUT / "struct_zz_stats.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_hist).to_csv(OUT / "struct_len_hist.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_vr).to_csv(OUT / "struct_vr_acf.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_cl).to_csv(OUT / "struct_cluster.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_wk).to_csv(OUT / "struct_cluster_weeks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_piv).to_csv(OUT / "struct_pivots_real.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_sym).to_csv(OUT / "struct_per_symbol.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("P1", "P2", "Y25", "Y24", "Y23"))
