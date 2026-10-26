"""대조 확인(결과를 본 뒤 추가, 선택편향 있음): S가 좋아진 것이 '추가 종목이 좋아서'인가 '종목 수가 늘어서'인가.

같은 종목 수끼리 비교한다. 점수(최종식 10일 평균) 순위 상위 k개 중 상승 배열(종가>50일선>200일선)인 종목을 같은
비중으로 매주 교체한다(S와 같은 관문). k = 6(지금 S 크기)·15(세 펀드를 모두 넣은 S 크기).
  U0 상위 6 대 U4 상위 6: 같은 수에서 추가 범위가 더 좋은 종목을 고르는가
  U0 상위 15 대 U4 상위 15: 종목 수를 늘렸을 때 어느 범위가 나은가
그리고 U4 S에서 추가 종목 중 S 일수가 가장 많았던 3개(RVMD·DNTH·CORT)를 빼면 남는가.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare as X  # noqa: E402

H, FC, FL = X.H, X.FC, X.FL


def score_and_gate(P, mask):
    um = mask & (P["C"].notna().cumsum() >= H.MIN_BARS) & P["C"].notna()
    orig = H.universe_mask
    H.universe_mask = lambda _P: um
    try:
        sc = FC.make(FC.GRID["E14"], 10)(P)
    finally:
        H.universe_mask = orig
    C = P["C"]
    s50, s200 = FL.sma(C, 50), FL.sma(C, 200)
    up = (C > s50) & (s50 > s200)
    return sc.where(um), um, up, FL.hyb_grades(sc, P, um)


def weekly_topk(P, sc, um, up, a, b, k, drop=()):
    O, C = P["O"], P["C"]
    cal = C.index
    d = cal[(cal >= pd.Timestamp(a)) & (cal <= pd.Timestamp(b))]
    wk = pd.Series(d.to_period("W-SUN"), index=d)
    firsts = [x for x, w in zip(d, wk) if x == wk[wk == w].index[0]]
    eq, curve, n_list = 1.0, [], []
    for i in range(len(firsts) - 1):
        w0, w1 = firsts[i], firsts[i + 1]
        prev = cal[cal.get_loc(w0) - 1]
        if prev < pd.Timestamp(a):
            continue
        row = sc.loc[prev][um.loc[prev] & up.loc[prev].fillna(False)].dropna().drop(labels=[x for x in drop if x in sc.columns], errors="ignore")
        names = [s for s in row.sort_values(ascending=False).index[:k] if np.isfinite(O.at[w0, s]) and np.isfinite(O.at[w1, s])]
        if not names:
            continue
        n_list.append(len(names))
        for x in d[(d >= w0) & (d < w1)]:
            curve.append(eq * float((C.loc[x, names] / O.loc[w0, names]).mean()))
        eq *= float((O.loc[w1, names] / O.loc[w0, names]).mean())
    c = pd.Series(curve)
    return round((eq - 1) * 100, 1), round(float((c / c.cummax() - 1).min()) * 100, 1), round(float(np.mean(n_list)), 1)


def weekly_s_drop(P, G, um, a, b, drop):
    """S 규칙에서 특정 종목을 뺀 판."""
    G2 = G.copy()
    for s in drop:
        if s in G2.columns:
            G2[s] = None
    return X.weekly(P, G2, um, a, b, "S")


def main():
    extra = {k: X.fund_members(fid) for k, fid in X.EXTRA.items()}
    allx = set().union(*[set().union(*v.values()) for v in extra.values() if v])
    P = X.load_panel(allx)
    out = {}
    for name in ("U0", "U4"):
        m = X.membership(P, X.VARIANTS[name], extra, aux_all=True)
        sc, um, up, G = score_and_gate(P, m)
        res = {}
        for w in ("front", "back", "y2026", "y2025"):
            a, b = X.WINDOWS[w]
            res[w] = {f"top{k}": weekly_topk(P, sc, um, up, a, b, k) for k in (6, 15)}
            if name == "U4":
                r = weekly_s_drop(P, G, um, a, b, ["RVMD", "DNTH", "CORT"])
                res[w]["S_drop3"] = (r["ret"], r["mdd"], r["avg_n"])
        out[name] = res
        for w, r in res.items():
            print(name, w, r, flush=True)
    (HERE.parent / "results" / "control.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
