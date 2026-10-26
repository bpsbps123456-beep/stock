"""BRIEF2 시험 2(a)·(c)·(d): 두 단서 3분위 AND 규칙 전부(18C2 × 3² = 1,377개 × type 2).

고르기: 학습 구간(P1, 또는 Y25+P1)에서 사건 50개·날짜 10일 이상인 규칙 중 고른 사건 평균 y1(로그) 1위와 상위 10개.
지표: 고른 사건 평균 y1 − 안 고른 사건 평균 y1. 확인 P2, 보조 Y25(P1 학습일 때), 참고 Y24·Y23.
귀무(c2combo_lib 설명): ① 고르기 귀무 — 학습 구간 단서를 날짜 원형 이동해 같은 고르기를 반복한 1위 평균,
② 고정 규칙 귀무 — 고른 규칙 그대로, 확인 구간 단서만 날짜 원형 이동(주) 또는 날짜 블록 순열(보조),
③ 전 과정 귀무 — 학습·확인 구간을 모두 이동해 고르기부터 반복.
py code/c2combo_rules.py [ZZA3|ZZP10] → results/c2combo_rules_<det>_all.csv, _top.csv, _summary.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from c2combo_lib import FEATURES, NREP, RES, bin_label, binize, boot_diff_ci, boot_rows, edges, group_rows, load, null_src, pq

I, J = np.triu_indices(len(FEATURES), 1)
NP = len(I)  # 153 쌍
MIN_N, MIN_D, TOPK = 50, 10, 10


def stats(B, y, di):
    """모든 규칙(153쌍 × 9칸)의 고른 사건 수·합·날짜 수와 전체 수·합."""
    _, dd = np.unique(di, return_inverse=True)
    D = dd.max() + 1
    flat = ((B[:, I] * 3 + B[:, J]) + 9 * np.arange(NP)).ravel()
    n = np.bincount(flat, minlength=NP * 9).reshape(NP, 9).astype(float)
    s = np.bincount(flat, weights=np.repeat(y, NP), minlength=NP * 9).reshape(NP, 9)
    key = flat * D + np.repeat(dd, NP)
    nd = (np.bincount(key, minlength=NP * 9 * D) > 0).reshape(NP * 9, D).sum(1).reshape(NP, 9)
    return n, s, nd, float(len(y)), float(y.sum())


def diffs(st):
    n, s, nd, N, S = st
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / n - (S - s) / (N - n)


def select(st, k=TOPK):
    n, s, nd, N, S = st
    elig = (n >= MIN_N) & (nd >= MIN_D) & (n < N)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(elig, s / n, -np.inf).ravel()
    order = np.argsort(-m, kind="stable")
    order = order[np.isfinite(m[order])][:k]
    return order, m[order], int(elig.sum())


def rule_mask(B, rid):
    p, c = divmod(int(rid), 9)
    return (B[:, I[p]] * 3 + B[:, J[p]]) == c


def rule_text(rid, e):
    p, c = divmod(int(rid), 9)
    a, b = divmod(c, 3)
    return bin_label(FEATURES[I[p]], a, e) + " AND " + bin_label(FEATURES[J[p]], b, e)


def side(y, sel):
    sel = sel & np.isfinite(y)
    un = ~sel & np.isfinite(y)
    return (np.mean(y[sel]) if sel.any() else np.nan) - (np.mean(y[un]) if un.any() else np.nan)


def union_mask(B, rids):
    m = np.zeros(len(B), bool)
    for rid in rids:
        m |= rule_mask(B, rid)
    return m


def run(det: str):
    ev = load(det)
    X = ev[FEATURES].values.astype(float)
    y = ev.y1.values
    di = ev.di.values
    all_rows, top_rows, sum_rows = [], [], []
    for t in (1, -1):
        e = edges(ev, t)
        B = binize(X, e)
        R = {p: group_rows(ev, p, t) for p in ["P1", "P2", "Y25", "Y24", "Y23"]}
        seed0 = 1000 * (t + 1) + (0 if det == "ZZA3" else 7)
        SH = {p: null_src(ev, R[p], "shift", NREP, seed0 + i) for i, p in enumerate(R)}
        PM = {p: null_src(ev, R[p], "perm", NREP, seed0 + 50 + i) for i, p in enumerate(R)}
        SA = {p: null_src(ev, R[p], "shiftall", NREP, seed0 + 80 + i) for i, p in enumerate(R)}
        st_test = {p: stats(B[R[p]], y[R[p]], di[R[p]]) for p in R}
        dif_test = {p: diffs(st_test[p]).ravel() for p in R}
        # 전 규칙 표(P1 기준 자격)
        st1 = st_test["P1"]
        for rid in range(NP * 9):
            p, c = divmod(rid, 9)
            row = dict(det=det, type=t, rid=rid, f1=FEATURES[I[p]], b1=c // 3, f2=FEATURES[J[p]], b2=c % 3,
                       P1_elig=bool(st1[0][p, c] >= MIN_N and st1[2][p, c] >= MIN_D and st1[0][p, c] < st1[3]))
            for q in R:
                n, s, nd, N, S = st_test[q]
                row[f"{q}_n"] = int(n[p, c]); row[f"{q}_dates"] = int(nd[p, c])
                row[f"{q}_mean"] = s[p, c] / n[p, c] if n[p, c] else np.nan
                row[f"{q}_diff"] = dif_test[q][rid]
            all_rows.append(row)
        for train in ("P1", "Y25P1"):
            tr = R["P1"] if train == "P1" else np.r_[R["P1"], R["Y25"]]
            st_tr = stats(B[tr], y[tr], di[tr])
            top, topm, n_elig = select(st_tr)
            if len(top) == 0:
                sum_rows.append(dict(det=det, type=t, train=train, n_elig=0))
                continue
            tests = ["P2", "Y25", "Y24", "Y23"] if train == "P1" else ["P2", "Y24", "Y23"]
            # 고르기 귀무(학습 구간 단서 이동 → 같은 고르기) + 전 과정 귀무(P2도 이동)
            sel_null_best, sel_null_top10, pipe1, pipe10, pipe_pos = [], [], [], [], []
            fix1, fix10, fixpos, perm1, perm10, all1, all10 = [], [], [], [], [], [], []
            y25_1, y25_10, uni_fix, uni_all, uni_perm, uni_pipe, uni_y25 = [], [], [], [], [], [], []
            for r in range(NREP):
                src_tr = SH["P1"][r] if train == "P1" else np.r_[SH["P1"][r], SH["Y25"][r]]
                stn = stats(B[src_tr], y[tr], di[tr])
                tn, tm, _ = select(stn)
                sel_null_best.append(tm[0] if len(tm) else np.nan)
                sel_null_top10.append(np.mean(tm) if len(tm) else np.nan)
                st2n = stats(B[SH["P2"][r]], y[R["P2"]], di[R["P2"]])
                d2n = diffs(st2n).ravel()
                if len(tn):
                    pipe1.append(d2n[tn[0]]); pipe10.append(np.nanmean(d2n[tn])); pipe_pos.append(np.sum(d2n[tn] > 0))
                fix1.append(d2n[top[0]]); fix10.append(np.nanmean(d2n[top])); fixpos.append(np.sum(d2n[top] > 0))
                uni_fix.append(side(y[R["P2"]], union_mask(B[SH["P2"][r]], top)))
                uni_all.append(side(y[R["P2"]], union_mask(B[SA["P2"][r]], top)))
                uni_perm.append(side(y[R["P2"]], union_mask(B[PM["P2"][r]], top)))
                if len(tn):
                    uni_pipe.append(side(y[R["P2"]], union_mask(B[SH["P2"][r]], tn)))
                d2p = diffs(stats(B[PM["P2"][r]], y[R["P2"]], di[R["P2"]])).ravel()
                perm1.append(d2p[top[0]]); perm10.append(np.nanmean(d2p[top]))
                d2a = diffs(stats(B[SA["P2"][r]], y[R["P2"]], di[R["P2"]])).ravel()
                all1.append(d2a[top[0]]); all10.append(np.nanmean(d2a[top]))
                if train == "P1":
                    d25 = diffs(stats(B[SH["Y25"][r]], y[R["Y25"]], di[R["Y25"]])).ravel()
                    y25_1.append(d25[top[0]]); y25_10.append(np.nanmean(d25[top]))
                    uni_y25.append(side(y[R["Y25"]], union_mask(B[SH["Y25"][r]], top)))
            # 고른 규칙 상세
            for rank, (rid, m) in enumerate(zip(top, topm), 1):
                msk_tr = rule_mask(B[tr], rid)
                row = dict(det=det, type=t, train=train, rank=rank, rid=int(rid), rule=rule_text(rid, e),
                           train_n=int(msk_tr.sum()), train_dates=int(np.unique(di[tr][msk_tr]).size),
                           train_mean=m, train_diff=side(y[tr], msk_tr))
                for q in tests:
                    rq = R[q]
                    mk = rule_mask(B[rq], rid)
                    row[f"{q}_n"] = int(mk.sum()); row[f"{q}_dates"] = int(np.unique(di[rq][mk]).size)
                    row[f"{q}_mean"] = float(np.mean(y[rq][mk])) if mk.any() else np.nan
                    row[f"{q}_diff"] = side(y[rq], mk)
                top_rows.append(row)
            # 요약(1위와 상위 10개)
            r2 = R["P2"]
            m2 = rule_mask(B[r2], top[0])
            d2 = dif_test["P2"]
            elig1 = ((st_tr[0] >= MIN_N) & (st_tr[2] >= MIN_D) & (st_tr[0] < st_tr[3])).ravel()
            pct = float(np.nanmean(d2[elig1] < d2[top[0]]))
            ci = boot_diff_ci(y[r2], m2, di[r2])
            # 상위 10 평균의 날짜 묶음 부트스트랩
            b10 = []
            for bi in boot_rows(di[r2], 500, seed=3):
                rr = r2[bi]
                b10.append(np.nanmean(diffs(stats(B[rr], y[rr], di[rr])).ravel()[top]))
            q95s, ps = pq(topm[0], sel_null_best)
            q95s10, ps10 = pq(np.mean(topm), sel_null_top10)
            q95f, pf = pq(d2[top[0]], fix1)
            q95f10, pf10 = pq(np.nanmean(d2[top]), fix10)
            q95p, pp = pq(d2[top[0]], pipe1)
            q95p10, pp10 = pq(np.nanmean(d2[top]), pipe10)
            q95m, pm_ = pq(d2[top[0]], perm1)
            q95m10, pm10 = pq(np.nanmean(d2[top]), perm10)
            q95a, pa = pq(d2[top[0]], all1)
            q95a10, pa10 = pq(np.nanmean(d2[top]), all10)
            nsets = len({rule_mask(B[tr], rid).tobytes() for rid in top})
            # 상위 10개 중 하나라도 해당(OR) 고르기
            u2 = union_mask(B[r2], top)
            u_d = side(y[r2], u2)
            u_ci = boot_diff_ci(y[r2], u2, di[r2])
            uq_f, up_f = pq(u_d, uni_fix)
            uq_a, up_a = pq(u_d, uni_all)
            uq_m, up_m = pq(u_d, uni_perm)
            uq_p, up_p = pq(u_d, uni_pipe)
            ok10 = [bool(rule_mask(B[r2], rid).sum() >= 10 and np.unique(di[r2][rule_mask(B[r2], rid)]).size >= 5) for rid in top]
            d_ok = d2[top][np.array(ok10)]
            npos = int(np.sum(d2[top] > 0))
            sr = dict(det=det, type=t, train=train, n_elig=n_elig, n_rules=NP * 9,
                      train_N=len(tr), train_dates=int(np.unique(di[tr]).size),
                      top1=rule_text(top[0], e), top1_train_mean=topm[0],
                      selnull_q95_top1=q95s, selnull_p_top1=ps, top10_train_mean=float(np.mean(topm)), selnull_q95_top10=q95s10, selnull_p_top10=ps10,
                      P2_N=len(r2), P2_dates=int(np.unique(di[r2]).size),
                      P2_top1_n=int(m2.sum()), P2_top1_dates=int(np.unique(di[r2][m2]).size),
                      P2_top1_sel_mean=float(np.mean(y[r2][m2])) if m2.any() else np.nan,
                      P2_top1_uns_mean=float(np.mean(y[r2][~m2])),
                      P2_top1_diff=d2[top[0]], P2_top1_ci_lo=ci[0], P2_top1_ci_hi=ci[1],
                      P2_top1_ok=bool(m2.sum() >= 10 and np.unique(di[r2][m2]).size >= 5), top10_distinct_sets=nsets,
                      P2_top1_fix_q95=q95f, P2_top1_fix_p=pf, P2_top1_fix_nullmean=float(np.nanmean(fix1)),
                      P2_top1_all_q95=q95a, P2_top1_all_p=pa, P2_top1_all_nullmean=float(np.nanmean(all1)),
                      P2_top1_perm_q95=q95m, P2_top1_perm_p=pm_,
                      P2_top1_pipe_q95=q95p, P2_top1_pipe_p=pp, P2_top1_pct_among_elig=pct,
                      P2_top1_diff_y1s=side(ev.y1s.values[r2], m2), P2_top1_diff_y10=side(ev.y10.values[r2], m2),
                      P2_top1_diff_y20=side(ev.y20.values[r2], m2),
                      P2_top10_mean_diff=float(np.nanmean(d2[top])), P2_top10_ci_lo=float(np.quantile(b10, 0.025)),
                      P2_top10_ci_hi=float(np.quantile(b10, 0.975)), P2_top10_npos=npos,
                      P2_top10_fix_q95=q95f10, P2_top10_fix_p=pf10, P2_top10_fix_nullmean=float(np.nanmean(fix10)),
                      P2_top10_all_q95=q95a10, P2_top10_all_p=pa10, P2_top10_perm_q95=q95m10, P2_top10_perm_p=pm10,
                      P2_top10_pipe_q95=q95p10, P2_top10_pipe_p=pp10,
                      P2_top10_npos_null_mean=float(np.mean(fixpos)),
                      P2_top10_n_ok=int(sum(ok10)), P2_top10_ok_mean_diff=float(np.mean(d_ok)) if len(d_ok) else np.nan,
                      P2_top10_ok_npos=int(np.sum(d_ok > 0)),
                      P2_union_n=int(u2.sum()), P2_union_dates=int(np.unique(di[r2][u2]).size), P2_union_diff=u_d,
                      P2_union_ci_lo=u_ci[0], P2_union_ci_hi=u_ci[1], P2_union_fix_q95=uq_f, P2_union_fix_p=up_f,
                      P2_union_fix_nullmean=float(np.nanmean(uni_fix)), P2_union_all_p=up_a, P2_union_perm_p=up_m, P2_union_pipe_p=up_p,
                      P2_union_diff_y10=side(ev.y10.values[r2], u2), P2_union_diff_y20=side(ev.y20.values[r2], u2),
                      P2_elig_mean_diff=float(np.nanmean(d2[elig1])), P2_elig_share_pos=float(np.nanmean(d2[elig1] > 0)))
            if train == "P1":
                r5 = R["Y25"]
                m5 = rule_mask(B[r5], top[0])
                d5 = dif_test["Y25"]
                ci5 = boot_diff_ci(y[r5], m5, di[r5])
                q5, p5 = pq(d5[top[0]], y25_1)
                q510, p510 = pq(np.nanmean(d5[top]), y25_10)
                sr.update(Y25_top1_n=int(m5.sum()), Y25_top1_dates=int(np.unique(di[r5][m5]).size), Y25_top1_diff=d5[top[0]],
                          Y25_top1_ci_lo=ci5[0], Y25_top1_ci_hi=ci5[1], Y25_top1_fix_q95=q5, Y25_top1_fix_p=p5,
                          Y25_top10_mean_diff=float(np.nanmean(d5[top])), Y25_top10_npos=int(np.sum(d5[top] > 0)),
                          Y25_top10_fix_q95=q510, Y25_top10_fix_p=p510,
                          Y25_union_n=int(union_mask(B[r5], top).sum()), Y25_union_diff=side(y[r5], union_mask(B[r5], top)),
                          Y25_union_ci_lo=boot_diff_ci(y[r5], union_mask(B[r5], top), di[r5])[0],
                          Y25_union_ci_hi=boot_diff_ci(y[r5], union_mask(B[r5], top), di[r5])[1],
                          Y25_union_fix_p=pq(side(y[r5], union_mask(B[r5], top)), uni_y25)[1],
                          Y24_union_diff=side(y[R["Y24"]], union_mask(B[R["Y24"]], top)), Y23_union_diff=side(y[R["Y23"]], union_mask(B[R["Y23"]], top)),
                          Y24_top1_diff=dif_test["Y24"][top[0]], Y23_top1_diff=dif_test["Y23"][top[0]],
                          Y24_top10_mean_diff=float(np.nanmean(dif_test["Y24"][top])), Y23_top10_mean_diff=float(np.nanmean(dif_test["Y23"][top])))
            sum_rows.append(sr)
            print(det, t, train, "자격", n_elig, "1위", sr["top1"], f"P1 {topm[0]:+.4f}", f"P2 diff {d2[top[0]]:+.4f} p_fix {pf:.3f}",
                  f"top10 P2 {np.nanmean(d2[top]):+.4f} 양수 {npos}/10 p {pf10:.3f}", f"OR P2 {u_d:+.4f} n {u2.sum()} p {up_f:.3f}/{up_a:.3f}/{up_m:.3f}", flush=True)
    pd.DataFrame(all_rows).to_csv(RES / f"c2combo_rules_{det}_all.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(top_rows).to_csv(RES / f"c2combo_rules_{det}_top.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sum_rows).to_csv(RES / f"c2combo_rules_{det}_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    for det in (sys.argv[1:] or ["ZZA3", "ZZP10"]):
        run(det)
