"""mtf diag 갈래 보강 분석: results/mtfdiag_corr_raw.parquet → 상관 기준 반대 방향 점검과 구간 구성.

- 확인: cov×100이 mtf_diag.parquet의 봉당 초과(trend/trend_n·amb/amb_n, %)와 같은지(k=0, real0).
- 상관 차이 = corr_T − corr_A(종목별) → 144종목 중앙, 귀무 k 40개로 표준화 점수·반대 방향 통과 수(mtfdiag_analyze와 같은 절차).
- 구성: 구간별 평균 보유 x̄, 보유 변동 sd pos, 봉 수익 변동 sd r의 종목 중앙(관측·귀무 중앙).
py code/mtfdiag_corr_analyze.py → results/mtfdiag_corr_summary.json, mtfdiag_corr_judgments.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mtfdiag_analyze import OUT, fam, judgments, mean_z, pass_low, pass_mat, pval_ge, zscore

BOARDS = ("real0", "ideal")
WN = {0: "sel", 1: "conf"}


def arrays(g, J):
    ks = [0] + sorted(int(k) for k in g.k.unique() if k > 0)
    syms = sorted(g.sym.astype(str).unique())
    jkey = {(b, c): i for i, (b, c) in enumerate(zip(J.b, J.clf))}
    ji = np.array([jkey[(b, c)] for b, c in zip(g.b, g.clf.astype(str))])
    ki = g.k.map({k: i for i, k in enumerate(ks)}).values
    si = g.sym.astype(str).map({s: i for i, s in enumerate(syms)}).values
    out = {}
    for c in ("n", "xbar", "sd_pos", "sd_r", "cov"):
        v = np.full((len(ks), len(J), len(syms)), np.nan)
        v[ki, ji, si] = g[c].values
        out[c] = v
    return out, ks, syms


def main():
    J = judgments()
    raw = pd.read_parquet(OUT / "mtfdiag_corr_raw.parquet")
    raw["board"] = raw.board.astype(str); raw["zone"] = raw.zone.astype(str)
    summ, tabs = {}, []
    # 확인: mtf_diag와 일치
    dg = pd.read_parquet(OUT / "mtf_diag.parquet")
    dg = dg[(dg.k == 0) & (dg.board == "real0")].copy()
    dg["clf"] = dg.clf.astype(str); dg["sym"] = dg.sym.astype(str)
    c0 = raw[(raw.k == 0) & (raw.board == "real0") & (raw.win == 1)].copy()
    c0["clf"] = c0.clf.astype(str); c0["sym"] = c0.sym.astype(str)
    t = c0[c0.zone == "T"].merge(dg, on=["sym", "b", "clf"])
    ok = t.trend_conf_n > 0
    summ["check_cov_vs_diag"] = dict(rows=int(ok.sum()), n_match=bool((t["n"][ok] == t.trend_conf_n[ok]).all()),
                                     max_abs_diff=float(np.nanmax(np.abs(t["cov"][ok] * 100 - t.trend_conf[ok] / t.trend_conf_n[ok] * 100))))
    for bd in BOARDS:
        for w, wn in WN.items():
            x = raw[(raw.board == bd) & (raw.win == w)]
            AT, ks, syms = arrays(x[x.zone == "T"], J)
            AA, _, _ = arrays(x[x.zone == "A"], J)
            with np.errstate(invalid="ignore", divide="ignore"):
                cT = AT["cov"] / (AT["sd_pos"] * AT["sd_r"])
                cA = AA["cov"] / (AA["sd_pos"] * AA["sd_r"])
            cT[~np.isfinite(cT)] = np.nan; cA[~np.isfinite(cA)] = np.nan
            d = cT - cA
            S = dict(diff=np.nanmedian(d, axis=2), trend=np.nanmedian(cT, axis=2), amb=np.nanmedian(cA, axis=2))
            S["zs"] = zscore(S["diff"])
            sign = (S["amb"] < 0) & (S["trend"] > 0)
            rsign = (S["amb"] > 0) & (S["trend"] < 0)
            P, Pr = pass_mat(S["diff"], sign), pass_low(S["diff"], rsign)
            key = f"{bd}_{wn}"
            summ.setdefault("corr_family", {})[key] = {
                nm: dict(pass_=fam(P, J[nm].values), reverse=fam(Pr, J[nm].values), mean_z=mean_z(S, J[nm].values),
                         n_stocks_defined=int(np.median((~np.isnan(d[0][J[nm].values])).sum(axis=1))))
                for nm in ("clean", "zgtb", "noext", "struct")}
            comp = {}
            for nm in ("clean", "zgtb", "struct"):
                ix = J[nm].values
                c = {}
                for zn, A in (("T", AT), ("A", AA)):
                    for col in ("xbar", "sd_pos", "sd_r"):
                        med = np.nanmedian(A[col], axis=2)  # (k, 판정)
                        c[f"{col}_{zn}"] = round(float(np.median(med[0, ix])), 4)
                        c[f"{col}_{zn}_null"] = round(float(np.median(med[1:, ix])), 4)
                # 하락 쪽 쏠림: 구간 안 보유 비율이 0.2 미만이거나 0.8 초과인 종목 비율
                for zn, A in (("T", AT), ("A", AA)):
                    xb = A["xbar"]
                    with np.errstate(invalid="ignore"):
                        ext = np.nanmean(np.where(np.isnan(xb), np.nan, (xb < 0.2) | (xb > 0.8)), axis=2)
                    c[f"lopsided_{zn}"] = round(float(np.median(ext[0, ix])), 3)
                    c[f"lopsided_{zn}_null"] = round(float(np.median(ext[1:, ix])), 3)
                comp[nm] = c
            summ.setdefault("composition", {})[key] = comp
            df = J.copy()
            df.insert(0, "win", wn); df.insert(0, "board", bd)
            df["corr_trend"] = S["trend"][0]; df["corr_amb"] = S["amb"][0]; df["corr_diff"] = S["diff"][0]
            df["corr_diff_null_med"] = np.median(S["diff"][1:], axis=0)
            df["corr_diff_p"] = [pval_ge(S["diff"][0, j], S["diff"][1:, j]) for j in range(len(J))]
            df["corr_z"] = S["zs"][0]
            df["corr_pass"] = P[0]; df["corr_rev_pass"] = Pr[0]
            for zn, A in (("T", AT), ("A", AA)):
                for col in ("xbar", "sd_pos", "sd_r"):
                    df[f"{col}_{zn}"] = np.nanmedian(A[col][0], axis=1)
            tabs.append(df)
    T = pd.concat(tabs, ignore_index=True)
    T.to_csv(OUT / "mtfdiag_corr_judgments.csv", index=False, encoding="utf-8-sig")
    (OUT / "mtfdiag_corr_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1,
                                                              default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(json.dumps(summ["check_cov_vs_diag"], ensure_ascii=False))
    for k, v in summ["corr_family"].items():
        print("corr", k, json.dumps(v, ensure_ascii=False))
    for k, v in summ["composition"].items():
        print("comp", k, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    main()
