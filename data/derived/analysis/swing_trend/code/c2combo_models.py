"""BRIEF2 시험 2(b)·(c)·(d): 로지스틱 회귀(L2)와 깊이 2 결정나무로 y1 > 0(이어짐) 맞히기.

- 학습: P1(b) 또는 Y25+P1(c), 매수 확정·매도 확정 따로. 결과표: P2 AUC, 예측 상위 1/3 사건 평균 y1 − 나머지.
- 로지스틱: 18개 단서를 학습 구간 평균·표준편차로 표준화(주). 민감도로 학습 구간 1·99분위에서 자른 뒤 표준화(윈저),
  시장 단서 8개만, 종목 단서 10개만도 본다. 규제 강도 C는 학습 구간 날짜를 시간순 5덩어리로 나눈 교차검증 평균 AUC 최대(같으면 작은 C).
- 나무: 원 단서, 깊이 2, 잎 최소 사건 = 학습 사건의 8%(사전 고정, 고르기 없음). 점수 = 잎의 학습 이어짐 비율.
- 상위 1/3: P2 점수 순위 상위 1/3(로지스틱), 그리고 학습 구간 점수 2/3분위 이상(두 모형 공통, '학습 문턱').
- 귀무: 고정 모형에 확인 구간 단서만 날짜 원형 이동(k 15..D−15, 주)·전 범위 이동·블록 순열(보조),
  그리고 전 과정 귀무(학습·확인 구간 단서를 모두 이동한 뒤 C 고르기·학습부터 반복, 주 모형만).
py code/c2combo_models.py [ZZA3|ZZP10] → results/c2combo_models_<det>.csv, _coef.csv, _tree.txt
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

from c2combo_lib import FEATURES, NAME, NREP, RES, boot_rows, group_rows, load, null_src, pq

warnings.filterwarnings("ignore")
CGRID = np.logspace(-3, 1, 9)
MF = [f for f in FEATURES if f[0] == "M"]
SF = [f for f in FEATURES if f[0] == "S"]
VARIANTS = {"logit": (FEATURES, False), "logit_winsor": (FEATURES, True), "logit_M": (MF, False), "logit_S": (SF, False)}


def auc(lab, sc):
    return roc_auc_score(lab, sc) if 0 < lab.sum() < len(lab) else np.nan


class Prep:
    def __init__(self, X, winsor):
        self.lo, self.hi = (np.quantile(X, 0.01, 0), np.quantile(X, 0.99, 0)) if winsor else (None, None)
        Z = self._clip(X)
        self.mu, self.sd = Z.mean(0), Z.std(0)
        self.sd[self.sd == 0] = 1.0

    def _clip(self, X):
        return np.clip(X, self.lo, self.hi) if self.lo is not None else X

    def __call__(self, X):
        return (self._clip(X) - self.mu) / self.sd


def fit_logit(X, lab, di, winsor):
    """날짜 시간순 5덩어리 교차검증으로 C를 고르고 전체로 다시 학습."""
    du = np.unique(di)
    blocks = np.array_split(du, 5)
    cv = []
    for C in CGRID:
        a = []
        for b in blocks:
            te = np.isin(di, b)
            if lab[~te].sum() in (0, (~te).sum()):
                continue
            pp = Prep(X[~te], winsor)
            m = LogisticRegression(C=C, max_iter=2000).fit(pp(X[~te]), lab[~te])
            a.append(auc(lab[te], m.decision_function(pp(X[te]))))
        cv.append(np.nanmean(a) if len(a) else np.nan)
    cv = np.array(cv)
    best = int(np.nanargmax(cv))  # 같은 값이면 앞(작은 C)
    pp = Prep(X, winsor)
    m = LogisticRegression(C=CGRID[best], max_iter=2000).fit(pp(X), lab)
    return (lambda Z: m.decision_function(pp(Z))), m, CGRID[best], cv


def fit_tree(X, lab):
    m = DecisionTreeClassifier(max_depth=2, min_samples_leaf=max(1, int(round(0.08 * len(lab)))), random_state=0).fit(X, lab)
    return (lambda Z: m.predict_proba(Z)[:, 1]), m


def top_diff(y, sc, thr=None):
    """상위 1/3(thr 없으면 그 표본 점수 2/3분위 이상) 평균 y − 나머지 평균 y, 고른 비율."""
    q = np.quantile(sc, 2 / 3) if thr is None else thr
    sel = sc >= q - 1e-12
    if sel.all() or not sel.any():
        return np.nan, float(sel.mean())
    return float(y[sel].mean() - y[~sel].mean()), float(sel.mean())


def evaluate(score, X, y, lab, thr):
    sc = score(X)
    d_rank, _ = top_diff(y, sc)
    d_thr, share = top_diff(y, sc, thr)
    return auc(lab, sc), d_rank, d_thr, share, sc


def describe_tree(m, cols):
    t = m.tree_
    out = []

    def rec(i, path):
        if t.children_left[i] == -1:
            v = t.value[i][0]
            p = v[1] / v.sum() if v.sum() else np.nan
            out.append(f"  {' 그리고 '.join(path) or '전체'} → 사건 {int(t.n_node_samples[i])}개, 이어짐 비율 {p:.0%}")
            return
        f = cols[t.feature[i]]
        th = t.threshold[i]
        rec(t.children_left[i], path + [f"{f}({NAME[f]}) ≤ {th:.3g}"])
        rec(t.children_right[i], path + [f"{f}({NAME[f]}) > {th:.3g}"])
    rec(0, [])
    return "\n".join(out)


def run(det: str):
    ev = load(det)
    y = ev.y1.values
    lab_all = (y > 0).astype(int)
    di = ev.di.values
    rows, coefs, trees = [], [], []
    for t in (1, -1):
        R = {p: group_rows(ev, p, t) for p in ["P1", "P2", "Y25", "Y24", "Y23"]}
        seed0 = 5000 + 1000 * (t + 1) + (0 if det == "ZZA3" else 7)
        SH = {p: null_src(ev, R[p], "shift", NREP, seed0 + i) for i, p in enumerate(R)}
        SA = {"P2": null_src(ev, R["P2"], "shiftall", NREP, seed0 + 80)}
        PM = {"P2": null_src(ev, R["P2"], "perm", NREP, seed0 + 50)}
        boots = boot_rows(di[R["P2"]], 1000, seed=seed0 + 9)
        for train in ("P1", "Y25P1"):
            tr = R["P1"] if train == "P1" else np.r_[R["Y25"], R["P1"]]
            tests = ["P2", "Y25", "Y24", "Y23"] if train == "P1" else ["P2", "Y24", "Y23"]
            models = {}
            for name, (cols, wz) in VARIANTS.items():
                Xc = ev[cols].values.astype(float)
                score, m, C, cv = fit_logit(Xc[tr], lab_all[tr], di[tr], wz)
                models[name] = (cols, score, dict(C=C, cv_auc=float(np.nanmax(cv))))
                if name == "logit":
                    for f, b in zip(cols, m.coef_[0]):
                        coefs.append(dict(det=det, type=t, train=train, model=name, feature=f, name=NAME[f], coef=b, C=C))
            Xc = ev[FEATURES].values.astype(float)
            score, m = fit_tree(Xc[tr], lab_all[tr])
            models["tree"] = (FEATURES, score, dict(C=np.nan, cv_auc=np.nan))
            trees.append(f"[{det} type {t:+d} 학습 {train}] 학습 사건 {len(tr)}개, 이어짐 비율 {lab_all[tr].mean():.0%}\n" + describe_tree(m, FEATURES))
            for name, (cols, score, info) in models.items():
                Xc = ev[cols].values.astype(float)
                sc_tr = score(Xc[tr])
                thr = np.quantile(sc_tr, 2 / 3)
                row = dict(det=det, type=t, train=train, model=name, n_feat=len(cols), **info,
                           train_N=len(tr), train_dates=int(np.unique(di[tr]).size), train_auc=auc(lab_all[tr], sc_tr))
                for q in tests:
                    rq = R[q]
                    a, dr, dt, sh, sc = evaluate(score, Xc[rq], y[rq], lab_all[rq], thr)
                    row.update({f"{q}_N": len(rq), f"{q}_dates": int(np.unique(di[rq]).size), f"{q}_pos_rate": lab_all[rq].mean(),
                                f"{q}_auc": a, f"{q}_top3_diff": dr, f"{q}_thr_diff": dt, f"{q}_thr_share": sh})
                    if q == "P2":
                        sel = sc >= np.quantile(sc, 2 / 3) - 1e-12
                        row.update(P2_top3_n=int(sel.sum()), P2_top3_dates=int(np.unique(di[rq][sel]).size),
                                   P2_top3_mean=float(y[rq][sel].mean()), P2_rest_mean=float(y[rq][~sel].mean()),
                                   P2_top3_diff_y10=float(np.nanmean(ev.y10.values[rq][sel]) - np.nanmean(ev.y10.values[rq][~sel])),
                                   P2_top3_diff_y20=float(np.nanmean(ev.y20.values[rq][sel]) - np.nanmean(ev.y20.values[rq][~sel])))
                        # 날짜 묶음 부트스트랩
                        ba, bd = [], []
                        for bi in boots:
                            ba.append(auc(lab_all[rq][bi], sc[bi])); bd.append(top_diff(y[rq][bi], sc[bi])[0])
                        row.update(P2_auc_ci_lo=np.nanquantile(ba, 0.025), P2_auc_ci_hi=np.nanquantile(ba, 0.975),
                                   P2_top3_ci_lo=np.nanquantile(bd, 0.025), P2_top3_ci_hi=np.nanquantile(bd, 0.975))
                        # 고정 모형 귀무
                        for tag, MP in (("fix", SH["P2"]), ("all", SA["P2"]), ("perm", PM["P2"])):
                            na, nd = [], []
                            for r in range(NREP):
                                sn = score(Xc[MP[r]])
                                na.append(auc(lab_all[rq], sn)); nd.append(top_diff(y[rq], sn)[0])
                            row[f"P2_auc_{tag}_q95"], row[f"P2_auc_{tag}_p"] = pq(a, na)
                            row[f"P2_top3_{tag}_q95"], row[f"P2_top3_{tag}_p"] = pq(dr, nd)
                            if tag == "fix":
                                row["P2_auc_fix_nullmean"] = float(np.nanmean(na)); row["P2_top3_fix_nullmean"] = float(np.nanmean(nd))
                    if q == "Y25":
                        na, nd = [], []
                        for r in range(NREP):
                            sn = score(Xc[SH["Y25"][r]])
                            na.append(auc(lab_all[rq], sn)); nd.append(top_diff(y[rq], sn)[0])
                        row["Y25_auc_fix_q95"], row["Y25_auc_fix_p"] = pq(a, na)
                        row["Y25_top3_fix_q95"], row["Y25_top3_fix_p"] = pq(dr, nd)
                # 전 과정 귀무(주 모형 logit·tree만): 학습·P2 단서 모두 이동 → 다시 학습
                if name in ("logit", "tree"):
                    pa, pd_ = [], []
                    for r in range(NREP):
                        src = SH["P1"][r] if train == "P1" else np.r_[SH["Y25"][r], SH["P1"][r]]
                        if name == "logit":
                            sn_f = fit_logit(Xc[src], lab_all[tr], di[tr], False)[0]
                        else:
                            sn_f = fit_tree(Xc[src], lab_all[tr])[0]
                        sn = sn_f(Xc[SH["P2"][r]])
                        pa.append(auc(lab_all[R["P2"]], sn)); pd_.append(top_diff(y[R["P2"]], sn)[0])
                    row["P2_auc_pipe_q95"], row["P2_auc_pipe_p"] = pq(row["P2_auc"], pa)
                    row["P2_top3_pipe_q95"], row["P2_top3_pipe_p"] = pq(row["P2_top3_diff"], pd_)
                rows.append(row)
                print(det, t, train, name, f"C {info['C']:.3g}" if np.isfinite(info["C"]) else "", f"P2 AUC {row['P2_auc']:.3f} p {row['P2_auc_fix_p']:.3f}",
                      f"상위1/3 {row['P2_top3_diff']:+.4f} p {row['P2_top3_fix_p']:.3f}",
                      f"Y25 AUC {row.get('Y25_auc', np.nan):.3f}", flush=True)
    pd.DataFrame(rows).to_csv(RES / f"c2combo_models_{det}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coefs).to_csv(RES / f"c2combo_models_{det}_coef.csv", index=False, encoding="utf-8-sig")
    (RES / f"c2combo_models_{det}_tree.txt").write_text("\n\n".join(trees), encoding="utf-8")


if __name__ == "__main__":
    for det in (sys.argv[1:] or ["ZZA3", "ZZP10"]):
        run(det)
