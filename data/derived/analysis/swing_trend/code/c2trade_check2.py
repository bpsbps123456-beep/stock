"""갈래 c2trade 점검 2(사후): 점검 1에서 나온 두 변형의 날짜 블록 순열 귀무 200회(c2trade_run.Donor.perm과 같은 기증 규칙, 같은 seed).

- 누수 제거 and_b: P1 표지를 5/29 종가로 자른 뒤 같은 절차(AND 1위, 장세 전환).
- and_b 매수만: 매수 확정은 AND 1위 규칙(이어짐이면 보유, 아니면 현금), 매도 확정은 모두 V자(보유) — 사전 등록 밖에서 결과를 보고 만든 변형.
귀무는 단서를 섞은 뒤 P1 학습부터 같은 절차를 반복한다. 주식 묶음 중앙(같은 비중 대비·ZZA 대비), 비용 0.
→ results/c2trade_check2.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import c2trade_check as K
import c2trade_run as R
import swing

OUT = swing.HERE / "results"
NPERM = 200


def med(rows, name):
    df = pd.DataFrame(rows, columns=["variant", "per", "sym", "cost", "ex_sw", "ex_zz", "x"])
    df = df[(df.cost == 0) & (df.variant == name)]
    return df.groupby("per")[["ex_sw", "ex_zz"]].median()


def run_variants(ev, X, y_override, keep, evaluate):
    T = len(ev)
    e2 = ev.copy()
    if y_override is not None:
        e2["y1"] = y_override
    e3 = e2[keep].reset_index(drop=True)
    pr, _ = R.fit_predict(e3, X[keep])
    full = np.zeros(T, bool); full[np.flatnonzero(keep)] = pr["and"]
    return full


def main():
    ev, X, info, evaluate = K.setup()
    T = len(ev)
    y_tr, keep_lf, _ = K.leak_free_labels(ev, info)
    keep_all = np.ones(T, bool)
    cal_per = {}
    D = swing.load(); ref = D["SPY"]
    for p in swing.PERIODS:
        cal_per[p] = list(ref.date.values[swing.period_index(ref, p)])
    donor = R.Donor(ev, cal_per)
    real = {}
    f_lf = run_variants(ev, X, y_tr, keep_lf, evaluate)
    f_or = run_variants(ev, X, None, keep_all, evaluate)
    zeros = np.zeros(T, bool)
    real["누수 제거 and_b"] = med(evaluate("v", f_lf, f_lf), "v")
    real["and_b 매수만"] = med(evaluate("v", f_or, zeros), "v")
    nul = {k: [] for k in real}
    rng = np.random.default_rng(R.SEED)
    for rep in range(NPERM):
        dn = donor.perm(rng)
        Xn = X[dn]
        g_lf = run_variants(ev, Xn, y_tr, keep_lf, evaluate)
        g_or = run_variants(ev, Xn, None, keep_all, evaluate)
        nul["누수 제거 and_b"].append(med(evaluate("v", g_lf, g_lf), "v"))
        nul["and_b 매수만"].append(med(evaluate("v", g_or, zeros), "v"))
    rows = []
    for k in real:
        for per in real[k].index:
            for col in ("ex_sw", "ex_zz"):
                a = float(real[k].loc[per, col])
                n = np.array([float(t.loc[per, col]) for t in nul[k]])
                rows.append(dict(variant=k, per=per, metric=col, actual=a, null_q50=float(np.quantile(n, .5)),
                                 null_q95=float(np.quantile(n, .95)), p=float((n >= a).mean())))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "c2trade_check2.csv", index=False)
    print(out.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
