"""robust 갈래 ③ 보강: 하루별 몫을 종목 × 날짜 표로 만들어 '공통으로 큰 날'을 뺀다.

종목마다 자기 최고 3일을 빼는 것(robust_decomp)은 귀무에서도 크게 깎이므로 함께 본다. 여기서는
 (a) 144종목 평균 E_d가 가장 큰 시장 공통 3일을 모든 종목에서 빼기,
 (b) 종목마다 보유 |B_d|가 가장 큰 3일(자기 큰 움직임 날)을 빼기,
를 실제(k=0)와 귀무(k=15~54)에 똑같이 적용한다. 날짜 정의는 robust_decomp와 같다(ET 20:00→20:00).
창(전체·고르기·확인)마다 그 창의 봉만으로 하루 표와 x를 따로 만든다(9/5 자료가 없는 종목의 9/4→9/8 갭이 확인 창에 새지 않게).
py code/robust_days.py → results/robust_days.npz, robust_days_summary.json
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from robust_common import COST_REAL, ORIG, RES, SYMS, bars_for, null_shifts, pos_ret, ratio, window_masks
from robust_decomp import steps


def day_table(st, pos, cost, wmask):
    gid = st["gid"]
    w5 = wmask[gid]
    ps = pos[gid].astype(float)
    chg = np.r_[pos[0] != 0, pos[1:] != pos[:-1]]
    c5 = np.where(st["first"], chg[gid] * np.log(1.0 - cost), 0.0)
    k = w5 & st["lastday"]
    days = np.r_[st["d"][w5], st["gday"][k]]
    sv = np.r_[ps[w5] * st["intra"][w5] + c5[w5], ps[k] * st["gap"][k]]
    bv = np.r_[st["intra"][w5], st["gap"][k]]
    return pd.DataFrame({"d": days, "s": sv, "b": bv}).groupby("d").sum(), float(pos[wmask].mean())


def work(args):
    sym, ks, alldays = args
    B = bars_for(sym, "5m", ORIG["tf"], "eth", 0)
    st = steps(B)
    p0 = ratio(B["O"], B["C"], B["V"], ORIG["N"], ORIG["kind"])
    masks = window_masks(B["seg"])
    nd = len(alldays)
    S = np.full((3, len(ks), 2, nd), np.nan); Bm = np.full((3, nd), np.nan); X = np.full((3, len(ks), 2), np.nan)
    pos_idx = {d: i for i, d in enumerate(alldays)}
    for ki, k in enumerate(ks):
        p = np.roll(p0, k * B["bpd"]) if k else p0
        for mi, (aw, cost) in enumerate(((B["ones"], 0.0), (B["awake"], COST_REAL["eth"]))):
            pos, _, _ = pos_ret(B["O"], B["C"], p, aw, ORIG["b"], ORIG["s"], True, cost)
            for wi, wm in enumerate(masks):
                df, x = day_table(st, pos, cost, wm)
                ii = np.array([pos_idx.get(str(d), -1) for d in df.index])
                assert (ii >= 0).all(), (sym, [str(d) for d in df.index[ii < 0]])
                S[wi, ki, mi, ii] = df.s.values; X[wi, ki, mi] = x
                if ki == 0 and mi == 0:
                    Bm[wi, ii] = df.b.values
    return sym, S, Bm, X


def stats(v):
    med = np.median(v, axis=0) * 100; posf = (v > 0).mean(axis=0)
    return dict(med=round(float(med[0]), 2), pos=round(float(posf[0]), 3), null_med=round(float(np.median(med[1:])), 2),
                null_pct=round(float((med[1:] < med[0]).mean()), 3), pos_null_med=round(float(np.median(posf[1:])), 3),
                pos_null_pct=round(float((posf[1:] < posf[0]).mean()), 3))


if __name__ == "__main__":
    D, shifts = null_shifts("5m")
    ks = [0] + shifts
    B0 = bars_for("MU", "5m", 40)
    lo, hi = B0["day"][0], B0["day"][-1]
    alldays = sorted({str(d) for d in pd.bdate_range(str(lo), str(hi)).date})  # 평일 전부(빈 날은 NaN)
    nd = len(alldays)
    S = np.full((len(SYMS), 3, len(ks), 2, nd), np.nan); Bm = np.full((len(SYMS), 3, nd), np.nan)
    X = np.full((len(SYMS), 3, len(ks), 2), np.nan)
    t0 = time.time()
    with Pool(3) as pool:
        for sym, s, b, x in pool.imap_unordered(work, [(s, ks, alldays) for s in SYMS]):
            i = SYMS.index(sym); S[i], Bm[i], X[i] = s, b, x
    np.savez_compressed(RES / "robust_days.npz", S=S, B=Bm, X=X, days=np.array(alldays), syms=np.array(SYMS), ks=np.array(ks))
    # 검산: 하루 합 = robust_decomp 창 합
    A = np.load(RES / "robust_decomp.npy"); F = json.loads((RES / "robust_decomp_meta.json").read_text(encoding="utf-8"))["feat"]
    for wi in range(3):
        for mi in range(2):
            d = np.nansum(S[:, wi, :, mi, :], axis=2) - A[:, :, mi, wi, F.index("S")]
            assert np.nanmax(np.abs(d)) < 1e-9, (wi, mi, np.nanmax(np.abs(d)))
    out = {}
    for wi, wn in enumerate(("full", "sel", "conf")):
        for mi, mode in enumerate(("ideal", "real")):
            Sd = np.nan_to_num(S[:, wi, :, mi, :]); Bd = np.nan_to_num(Bm[:, wi, :])
            E = Sd - X[:, wi, :, mi][:, :, None] * Bd[:, None, :]  # [sym, k, day]
            tot = E.sum(axis=2)
            rm_a = np.zeros_like(tot); top = []
            for ki in range(len(ks)):
                m = E[:, ki, :].mean(axis=0); t3 = np.argsort(-m)[:3]
                rm_a[:, ki] = tot[:, ki] - E[:, ki, t3].sum(axis=1)
                if ki == 0:
                    top = [(alldays[j], round(float(m[j] * 100), 2), round(float((E[:, 0, j] > 0).mean()), 2),
                            round(float(np.median(np.expm1(Bd[:, j])) * 100), 1)) for j in t3]
            t3b = np.argsort(-np.abs(Bd), axis=1)[:, :3]
            rm_b = tot - np.take_along_axis(E, np.repeat(t3b[:, None, :], len(ks), axis=1), axis=2).sum(axis=2)
            bb = np.take_along_axis(Bd, t3b, axis=1); ss = np.take_along_axis(Sd[:, 0, :], t3b, axis=1); dn = bb < 0
            m0 = E[:, 0, :].mean(axis=0)
            r = dict(E=stats(tot), E_rm_market3=stats(rm_a), E_rm_bigmove3=stats(rm_b),
                     market_top3_days=top,  # (날짜, 종목 평균 E_d %p, E_d>0 종목 비율, 보유 중앙 %)
                     bigmove_days=dict(down_share=round(float(dn.mean()), 2),
                                       down_hold=round(float(np.expm1(bb[dn]).mean() * 100), 1),
                                       down_strat=round(float(np.expm1(ss[dn]).mean() * 100), 1),
                                       up_hold=round(float(np.expm1(bb[~dn]).mean() * 100), 1),
                                       up_strat=round(float(np.expm1(ss[~dn]).mean() * 100), 1)),
                     mean_E_total=round(float(m0.sum() * 100), 2), mean_E_top5_days=round(float(np.sort(m0)[-5:].sum() * 100), 2),
                     mean_E_bottom5_days=round(float(np.sort(m0)[:5].sum() * 100), 2))
            out[f"{wn}_{mode}"] = r
            print(wn, mode, json.dumps(r, ensure_ascii=False))
    (RES / "robust_days_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("끝", round(time.time() - t0), "초")
