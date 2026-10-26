"""cross_section 갈래 보충 점검 (2026-10-13). 새 조합을 고르지 않는다(진단만).

A) 합성 순위 51개 중 h10 IC가 front·back 모두 > 0인 것: 상위 20% 초과(h10) 날짜 계열로
   최대 기여 종목 빼기·SNDK 빼기·상위 5일 빼기·월별·장세(QQQ 종가 > 50일선) 점검,
   주간 S(상위 10%) 포트폴리오에서 최대 기여 종목 빼기.
B) 이중 정렬 36조합 평균 차이(SP_hi − SP_lo, h10)가 시장 방향(베타) 몫인지:
   날짜별 차이를 같은 기간 QQQ 수익(t+1 시가 → t+11 시가)에 회귀, 그리고 t일에 아는 60일 베타 차이.

실행: PYTHONIOENCODING=utf-8 py cross_section_extra.py
결과: ../results/cross_section_extra.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_section as X  # noqa: E402

H = X.H
RESULTS = X.RESULTS


def top20_series(score: pd.DataFrame, h: int, dates, cols, drop=()) -> tuple[pd.Series, pd.Series]:
    """harness.evaluate와 같은 잣대: eligible 안 백분위 > 0.8 평균 − eligible 평균. 종목별 기여(%p)도."""
    P = H.load_panel()
    E = H.eligible().copy()
    if drop:
        E[list(drop)] = False
    score = score.reindex(index=P["C"].index, columns=P["C"].columns)
    mask = E & score.notna()
    pr = H.pct_rank(score, mask)
    cols = [c for c in cols if c not in drop]
    fr = H.forward(h).loc[dates, cols].where(mask.loc[dates, cols])
    sc = pr.loc[dates, cols]
    top = (sc > 0.8) & fr.notna()
    ex = fr.sub(fr.mean(axis=1), axis=0)
    s = ex.where(top).mean(axis=1)
    contrib = (ex.where(top).div(top.sum(axis=1), axis=0).fillna(0).sum(axis=0) / s.notna().sum()) * 100
    return s.dropna(), contrib.sort_values(ascending=False)


def weekly_contrib(score: pd.DataFrame, lo: float, start: str, end: str, drop=()) -> tuple[dict, pd.Series]:
    """harness.weekly_portfolio 결과와, 주별 (종목 수익/보유 수)의 합으로 본 종목별 기여(%p, 단리 근사)."""
    P = H.load_panel()
    O, C = P["O"], P["C"]
    E = H.eligible().copy()
    if drop:
        E[list(drop)] = False
    score = score.reindex(index=C.index, columns=C.columns)
    mask = E & score.notna()
    pr = H.pct_rank(score, mask)
    res = H.weekly_portfolio(pr, lo, start, end, mask)
    cal = C.index
    days = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    week = pd.Series(days.to_period("W-SUN"), index=days)
    firsts = [d for d, w in zip(days, week) if d == week[week == w].index[0]]
    contrib = pd.Series(0.0, index=C.columns)
    for i, w0 in enumerate(firsts[:-1]):
        w1 = firsts[i + 1]
        prev = cal[cal.get_loc(w0) - 1]
        row = pr.loc[prev]
        names = [n for n in row[(row > lo) & mask.loc[prev]].index if np.isfinite(O.at[w0, n])]
        if not names:
            continue
        r = (O.loc[w1, names].fillna(C.loc[cal[cal.get_loc(w1) - 1], names]) / O.loc[w0, names] - 1)
        contrib[names] += r.fillna(0) / len(names)
    return res, (contrib * 100).sort_values(ascending=False)


def stat(x, h):
    return X.stat(x, h)


def part_a():
    P = H.load_panel()
    rk = json.load(open(os.path.join(RESULTS, "cross_section_rankers.json"), encoding="utf-8"))
    cands = [r["name"] for r in rk["rows"]
             if r["result"]["front"]["h10"]["ic"] > 0 and r["result"]["back"]["h10"]["ic"] > 0]
    fns = {}
    for L in X.LS:
        for p in X.P_NAMES:
            fns[f"comp_S{L}+{p}"] = X.make_composite(L, p)
    for L in X.LS:
        fns[f"S{L}"] = X.make_S(L)
    for p in X.P_NAMES:
        fns[p] = X.make_P(p)
    reg = X.regime(P)
    last = P["C"].index[-1].strftime("%Y-%m-%d")
    out = {"candidates_ic10_both_pos": cands, "n_rankers": len(rk["rows"]), "checks": {}}
    for name in cands:
        sc = fns[name](P)
        c = {}
        for w in ("front", "back"):
            dates = H.window_dates(w, 10)
            cols = X.window_cols(P, w)
            s, con = top20_series(sc, 10, dates, cols)
            top = con.index[0]
            s2, _ = top20_series(sc, 10, dates, cols, drop=(top,))
            s3, _ = top20_series(sc, 10, dates, cols, drop=("SNDK",))
            rg = reg.reindex(s.index).fillna(False)
            a, b = H.WINDOWS[w]
            b = min(b, last)
            ps, pcon = weekly_contrib(sc, 0.90, a, b)
            ptop = pcon.index[0]
            ps2, _ = weekly_contrib(sc, 0.90, a, b, drop=(ptop,))
            ps3, _ = weekly_contrib(sc, 0.90, a, b, drop=("SNDK",))
            c[w] = {
                "top20_h10": stat(s, 10),
                "top_contrib": [{"sym": k, "pp": round(float(v), 3)} for k, v in con.head(5).items()],
                "drop_top_stock": {"sym": top, **stat(s2, 10)},
                "drop_SNDK": {"SNDK_contrib_pp": round(float(con.get("SNDK", 0.0)), 3), **stat(s3, 10)},
                "drop_top5_dates": X.drop_dates_stat(s, 10, 5),
                "regime_qqq_above_sma50": stat(s[rg], 10),
                "regime_qqq_below_sma50": stat(s[~rg], 10),
                "port_S": ps,
                "port_S_top_contrib": [{"sym": k, "pp": round(float(v), 2)} for k, v in pcon.head(5).items()],
                "port_S_drop_top_stock": {"sym": ptop, **ps2},
                "port_S_drop_SNDK": ps3,
            }
        cal = P["C"].index
        ex_ = H.exit_date(10)
        dates = cal[(cal >= pd.Timestamp("2024-01-02")) & ex_.notna().to_numpy()]
        sall, _ = top20_series(sc, 10, dates, [k for k in P["C"].columns if k != "DRUG"])
        c["monthly_top20_h10"] = X.monthly(sall)
        out["checks"][name] = c
        print("ranker checked", name, flush=True)
    return out


def part_b():
    """36조합 평균 차이의 베타 몫."""
    P = H.load_panel()
    U = H.universe_mask(P)
    E = H.eligible()
    feat = X.features(P)
    Q = P["bench"]
    qfwd = Q["O"].shift(-11) / Q["O"].shift(-1) - 1.0
    # t일에 아는 60일 베타(일 종가 수익, 종목별)
    r = P["C"].pct_change()
    m = Q["C"].pct_change()
    cov = r.rolling(60, min_periods=60).cov(m)
    var = m.rolling(60, min_periods=60).var()
    beta = cov.div(var, axis=0)
    reg = X.regime(P)
    diffs, bdiff = {}, {}
    cal = P["C"].index
    ex_ = H.exit_date(10)
    dates = cal[(cal >= pd.Timestamp("2024-01-02")) & ex_.notna().to_numpy()]
    cols = [c for c in P["C"].columns if c != "DRUG"]
    for L in X.LS:
        for p in X.P_NAMES:
            g = X.groups(feat["S"][L], feat["P"][p], U)
            n = X.combo_name(L, p)
            diffs[n] = X.all_dates_series(g, 10, P, E)
            bh = beta.loc[dates, cols].where(g["SP_hi"].loc[dates, cols]).mean(axis=1)
            bl = beta.loc[dates, cols].where(g["SP_lo"].loc[dates, cols]).mean(axis=1)
            bdiff[n] = bh - bl
    avg = pd.concat(diffs, axis=1).mean(axis=1)
    bavg = pd.concat(bdiff, axis=1).mean(axis=1)
    out = {}
    for w in ("front", "back", "y2025", "y2024"):
        d = H.window_dates(w, 10)
        y = avg.reindex(d).dropna()
        x = qfwd.reindex(y.index)
        ok = x.notna()
        y, x = y[ok], x[ok]
        A = np.column_stack([np.ones(len(x)), x.to_numpy()])
        coef, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
        resid = pd.Series(y.to_numpy() - coef[1] * x.to_numpy(), index=y.index)  # 절편 + 잔차 = 베타 뺀 몫
        rg = reg.reindex(y.index).fillna(False)
        out[w] = {
            "diff_mean": round(float(y.mean()) * 100, 3),
            "slope_on_qqq_fwd10": round(float(coef[1]), 3),
            "intercept_pp": round(float(coef[0]) * 100, 3),
            "corr": round(float(np.corrcoef(x, y)[0, 1]), 3),
            "beta_neutral": stat(resid, 10),
            "beta_neutral_above_sma50": stat(resid[rg], 10),
            "beta_neutral_below_sma50": stat(resid[~rg], 10),
            "qqq_fwd10_above": round(float(x[rg].mean()) * 100, 2), "qqq_fwd10_below": round(float(x[~rg].mean()) * 100, 2),
            "beta60_SPhi_minus_SPlo": round(float(bavg.reindex(y.index).mean()), 3),
            "beta60_diff_above": round(float(bavg.reindex(y.index)[rg].mean()), 3),
            "beta60_diff_below": round(float(bavg.reindex(y.index)[~rg].mean()), 3),
        }
        print(w, out[w], flush=True)
    return out


if __name__ == "__main__":
    res = {"family": X.FAMILY, "part": "extra",
           "note": "diagnostics only, no new selection; ranker candidates = harness IC10 > 0 in both front and back",
           "rankers": part_a(), "beta_check_all36_h10": part_b()}
    path = os.path.join(RESULTS, f"{X.FAMILY}_extra.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("saved", path)
