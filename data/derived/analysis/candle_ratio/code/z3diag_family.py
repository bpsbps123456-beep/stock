"""z3 diag 갈래 보강: 통과 후보 수의 계열 단위 귀무(후보끼리 상관 반영)와 사후 표지 ③ 통과 후보 목록.

귀무 이동 k마다 '그 k의 diff가 나머지 이동들의 95분위를 넘는 후보 수'(한 개 빼기)를 세어
관측 통과 수(k=0의 diff > 전체 귀무 95분위)와 비교한다. 채택 기준 1(부호 조건 포함)도 같은 방식.
py code/z3diag_family.py → results/z3diag_family.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from z3diag_analyze import OUT, labels, load, stock_stats


def loo_counts(diff, sign):
    """diff, sign: (k, 후보). k=0 관측, 1.. 귀무. 귀무 k마다 나머지 귀무의 95분위로 센 통과 수."""
    null = diff[1:]
    obs_pass = diff[0] > np.quantile(null, 0.95, axis=0)
    obs = dict(diff=int(obs_pass.sum()), crit1=int((obs_pass & sign[0]).sum()))
    cd, cc = [], []
    for i in range(null.shape[0]):
        rest = np.delete(null, i, axis=0)
        p = null[i] > np.quantile(rest, 0.95, axis=0)
        cd.append(int(p.sum()))
        cc.append(int((p & sign[1 + i]).sum()))
    cd, cc = np.array(cd), np.array(cc)
    return dict(obs=obs, null_diff_mean=round(float(cd.mean()), 2), null_diff_q95=float(np.quantile(cd, 0.95)),
                p_diff=round(float(((cd >= obs["diff"]).sum() + 1) / (len(cd) + 1)), 3),
                null_crit1_mean=round(float(cc.mean()), 2), null_crit1_q95=float(np.quantile(cc, 0.95)),
                p_crit1=round(float(((cc >= obs["crit1"]).sum() + 1) / (len(cc) + 1)), 3),
                null_diff_hist=np.bincount(cd).tolist())


def posthoc_loo(days, Z, board, win, lab=3, zval=0):
    L = labels(days, board)
    m = np.ones(len(days), bool) if win == "all" else (days.seg.values == (0 if win == "sel" else 1))
    Lm, Zm = L[m], Z[:, :, m]
    sel = Lm == lab
    lift = (Zm[:, :, sel] == zval).mean(axis=2) * 100 - (Zm == zval).mean(axis=2) * 100
    return loo_counts(lift, np.ones_like(lift, dtype=bool))


if __name__ == "__main__":
    out = {}
    for base in ("5m", "1h"):
        D = load(base)
        for bi, bn in enumerate(D["boards"]):
            for wi, wn in enumerate(("sel", "conf")):
                S = stock_stats(D["RES"][:, bi, wi])
                sign = (S["amb"] < 0) & (S["trend"] > 0)
                out[f"{base}_{bn}_{wn}"] = loo_counts(S["diff"], sign)
    days = pd.read_parquet(OUT / "z3diag_days_5m.parquet")
    dz = np.load(OUT / "z3diag_days_5m.npz")
    Z = dz["Z"]
    for board in ("real01", "ideal"):
        for win in ("conf", "sel", "all"):
            for lab, zv in ((1, 1), (2, -1), (3, 0)):
                out[f"posthoc_{board}_{win}_L{lab}"] = posthoc_loo(days, Z, board, win, lab, zv)
    ph = pd.read_csv(OUT / "z3diag_posthoc_5m.csv")
    lst = ph[(ph.label == 3) & ph["pass"]].groupby(["board", "win"]).cand.apply(list)
    out["posthoc_L3_pass_list"] = {f"{a}_{b}": v for (a, b), v in lst.items()}
    (OUT / "z3diag_family.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in out.items():
        if k == "posthoc_L3_pass_list":
            print(k, json.dumps(v, ensure_ascii=False))
        else:
            print(k, v["obs"], "귀무 diff 평균", v["null_diff_mean"], "95분위", v["null_diff_q95"], "p", v["p_diff"],
                  "| 기준1 귀무 평균", v["null_crit1_mean"], "95분위", v["null_crit1_q95"], "p", v["p_crit1"])
