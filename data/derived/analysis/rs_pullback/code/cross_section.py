"""상대강도 + 눌림 연구 — 갈래 'cross_section' (2026-10-13, 사전 등록은 ../BRIEF.md).

질문: 'QQQ 대비 강한 종목 중 눌린 것'을 그날 보유 종목 단면에서 고르면 이후 더 오르는가.

1) 이중 정렬(36조합 = 강도 L 3개 × 눌림 P 12개)
   그날 universe_mask 안 pct(S_L) >= 0.70 = 강함. 강함 안에서 P 순위 3등분:
   SP_hi(가장 눌림, 상위 1/3) / SP_lo(덜 눌림 = 추격, 하위 1/3). 약함&눌림 = pct(S_L) <= 0.30 이고
   약함 안 P 상위 1/3. h = 5·10·20, 구간 front/back/y2025/y2024. 그날 eligible 평균 대비 초과(%p, 날짜 평균),
   차이 SP_hi − SP_lo의 Newey-West t(lag=h).
   주 지표(사전 등록): h=10 차이. front 1위 조합을 고르고 back에서 확인.
2) 합성 순위 51개: pct(S_L)+pct(P) 36 + S_L 3 + P 12 → harness.evaluate, 각 점수 harness.no_lookahead.
3) front·back 모두 차이(h10) > 0인 조합: 최대 기여 종목 빼기, SNDK 빼기, 상위 5일 빼기, 월별, 장세 1개
   (t일 QQQ 종가 > 50일 단순 이평).
4) 36개 주 차이의 front·back 부호 집계, 부호 일치 이항 검정, 순위 상관.

실행: PYTHONIOENCODING=utf-8 py cross_section.py [sort|rank|all]
결과: ../results/cross_section.json (sort 부분), ../results/cross_section_rankers.json (rank 부분)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TG = os.path.abspath(os.path.join(HERE, "..", "..", "trend_grade", "code"))
sys.path.insert(0, TG)
import harness as H  # noqa: E402

RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FAMILY = "cross_section"
LS = (20, 60, 120)
HS = (5, 10, 20)
WINS = ("front", "back", "y2025", "y2024")
STRONG, WEAK = 0.70, 0.30
P_NAMES = ["P1_3", "P1_4", "P1_5", "P2_5", "P2_10", "P2_20", "P3_5", "P3_10", "P3_20", "P4_5", "P4_10", "P4_20"]


# ---------- 공통 정의 (BRIEF) ----------
def ex(P: dict, n: int) -> pd.DataFrame:
    C, Q = P["C"], P["bench"]["C"].reindex(P["C"].index)
    return 100.0 * (C / C.shift(n)).sub(Q / Q.shift(n), axis=0)


def strength(P: dict, L: int) -> pd.DataFrame:
    return ex(P, L)


def pullback(P: dict, name: str) -> pd.DataFrame:
    kind, k = name.split("_")
    k = int(k)
    C, Q = P["C"], P["bench"]["C"].reindex(P["C"].index)
    if kind == "P1":
        return -ex(P, k)
    if kind == "P2":
        e4 = ex(P, 4)
        return e4.rolling(k, min_periods=k).max() - e4
    if kind == "P3":
        R = C.div(Q, axis=0)
        return 1.0 - R / R.rolling(k, min_periods=k).max()
    if kind == "P4":
        return 1.0 - C / C.rolling(k, min_periods=k).max()
    raise ValueError(name)


def combo_name(L: int, p: str) -> str:
    return f"S{L}x{p}"


# ---------- 1) 이중 정렬 ----------
def groups(S: pd.DataFrame, Pm: pd.DataFrame, U: pd.DataFrame) -> dict:
    """t일까지 아는 값만으로 그룹 bool 표. U = universe_mask (점수 안 단면 순위용)."""
    ps = S.where(U).rank(axis=1, pct=True)
    strong = ps >= STRONG
    weak = ps <= WEAK
    # 동점 처리: P2·P3·P4는 K봉 고점이면 0으로 동점이 많다(강한 종목 절반 이상인 날도 있음).
    # 평균 순위면 동점 묶음이 1/3을 넘어 SP_lo가 비는 날이 생기므로(front 91일 중 11일) 'min' 순위로
    # 고점 동점 종목을 모두 '덜 눌림'에 넣는다. 그래서 SP_lo가 1/3보다 클 수 있다.
    pin_s = Pm.where(strong & Pm.notna()).rank(axis=1, pct=True, method="min")
    pin_w = Pm.where(weak & Pm.notna()).rank(axis=1, pct=True, method="min")
    return {"SP_hi": pin_s > 2 / 3, "SP_lo": pin_s <= 1 / 3, "strong": strong, "weak_pulled": pin_w > 2 / 3}


def daily_excess(mask: pd.DataFrame, h: int, dates, cols, E: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """날짜별 (그룹 평균 − eligible 평균) 수익(소수), 그룹 인원."""
    fr = H.forward(h).loc[dates, cols].where(E.loc[dates, cols])
    m = mask.loc[dates, cols] & fr.notna()
    g = fr.where(m).mean(axis=1)
    return g - fr.mean(axis=1), m.sum(axis=1)


def daily_raw(mask: pd.DataFrame, h: int, dates, cols, E) -> pd.Series:
    fr = H.forward(h).loc[dates, cols].where(E.loc[dates, cols])
    return fr.where(mask.loc[dates, cols] & fr.notna()).mean(axis=1)


def window_cols(P: dict, w: str, drop=()) -> list:
    excl = set(H.EXCLUDE.get(w, set())) | set(drop)
    return [c for c in P["C"].columns if c not in excl]


def stat(x: pd.Series, h: int) -> dict:
    x = x.dropna()
    return {"mean": round(float(x.mean()) * 100, 3) if len(x) else None,
            "t": round(H.newey_west_t(x, h), 2) if len(x) >= 5 else None,
            "pos": round(float((x > 0).mean()) * 100, 1) if len(x) else None, "days": int(len(x))}


def sort_table(feat: dict, U, E, P, drop=(), wins=WINS, hs=HS, keep_series=False) -> dict:
    """36조합 × h × 구간. drop = 우주에서 통째로 뺄 종목(순위부터 다시 계산)."""
    if drop:
        U = U.copy()
        U[list(drop)] = False
    out, series = {}, {}
    for L in LS:
        for p in P_NAMES:
            g = groups(feat["S"][L], feat["P"][p], U)
            name = combo_name(L, p)
            row = {}
            for h in hs:
                for w in wins:
                    dates = H.window_dates(w, h)
                    cols = window_cols(P, w, drop)
                    r = {}
                    hi, nhi = daily_excess(g["SP_hi"], h, dates, cols, E)
                    lo, nlo = daily_excess(g["SP_lo"], h, dates, cols, E)
                    st, _ = daily_excess(g["strong"], h, dates, cols, E)
                    wp, _ = daily_excess(g["weak_pulled"], h, dates, cols, E)
                    diff = (hi - lo).dropna()
                    r["SP_hi"] = stat(hi, h)
                    r["SP_lo"] = stat(lo, h)
                    r["strong"] = stat(st, h)
                    r["weak_pulled"] = stat(wp, h)
                    r["diff"] = stat(diff, h)
                    r["n_hi"] = round(float(nhi[nhi > 0].mean()), 1) if (nhi > 0).any() else 0
                    r["n_lo"] = round(float(nlo[nlo > 0].mean()), 1) if (nlo > 0).any() else 0
                    row[f"h{h}_{w}"] = r
                    if keep_series:
                        series[(name, h, w)] = diff
            out[name] = row
    return (out, series) if keep_series else out


# ---------- 3) 집중 점검 ----------
def contributions(g: dict, h: int, w: str, P, E) -> pd.Series:
    """구간 평균 차이(SP_hi − SP_lo)에 대한 종목별 기여(%p). 합 = 구간 평균 차이."""
    dates = H.window_dates(w, h)
    cols = window_cols(P, w)
    fr = H.forward(h).loc[dates, cols].where(E.loc[dates, cols])
    fx = fr.sub(fr.mean(axis=1), axis=0)
    hi = g["SP_hi"].loc[dates, cols] & fr.notna()
    lo = g["SP_lo"].loc[dates, cols] & fr.notna()
    ok = (hi.sum(axis=1) > 0) & (lo.sum(axis=1) > 0)
    a = fx.where(hi).div(hi.sum(axis=1), axis=0).fillna(0)
    b = fx.where(lo).div(lo.sum(axis=1), axis=0).fillna(0)
    c = (a - b).loc[ok].sum(axis=0) / ok.sum()
    return (c * 100).sort_values(ascending=False)


def drop_dates_stat(x: pd.Series, h: int, k: int = 5) -> dict:
    x = x.dropna()
    top = x.sort_values(ascending=False).index[:k]
    rest = x.drop(top)
    d = stat(rest, h)
    d["dropped"] = [t.strftime("%Y-%m-%d") for t in top]
    d["dropped_vals"] = [round(float(x[t]) * 100, 2) for t in top]
    return d


def monthly(x: pd.Series) -> dict:
    x = x.dropna()
    g = x.groupby(x.index.to_period("M"))
    return {str(k): {"mean": round(float(v.mean()) * 100, 2), "days": int(len(v))} for k, v in g}


def regime(P) -> pd.Series:
    Q = P["bench"]["C"]
    return Q > Q.rolling(50, min_periods=50).mean()


def all_dates_series(g: dict, h: int, P, E) -> pd.Series:
    """2024-01 ~ 끝, 매도일이 자료 안에 있는 모든 신호일의 차이(구간 제외 없음, y2024의 DRUG만 뺌)."""
    cal = P["C"].index
    ex_ = H.exit_date(h)
    dates = cal[(cal >= pd.Timestamp("2024-01-02")) & ex_.notna().to_numpy()]
    cols = [c for c in P["C"].columns if c != "DRUG"]
    hi, _ = daily_excess(g["SP_hi"], h, dates, cols, E)
    lo, _ = daily_excess(g["SP_lo"], h, dates, cols, E)
    return (hi - lo).dropna()


# ---------- 4) 부호 집계 ----------
def binom_two_sided(k: int, n: int) -> float:
    pk = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    obs = pk[k]
    return float(min(1.0, sum(p for p in pk if p <= obs + 1e-15)))


def spearman(a, b) -> float:
    a, b = pd.Series(a), pd.Series(b)
    return float(a.rank().corr(b.rank()))


# ---------- 2) 합성 순위 ----------
def make_composite(L: int, p: str):
    def f(P):
        U = H.universe_mask(P)
        S, Pm = strength(P, L), pullback(P, p)
        return S.where(U).rank(axis=1, pct=True) + Pm.where(U).rank(axis=1, pct=True)
    return f


def make_S(L: int):
    return lambda P: strength(P, L)


def make_P(p: str):
    return lambda P: pullback(P, p)


def run_rank():
    P = H.load_panel()
    fns = {}
    for L in LS:
        for p in P_NAMES:
            fns[f"comp_S{L}+{p}"] = ("pct(S_L)+pct(P)", make_composite(L, p))
    for L in LS:
        fns[f"S{L}"] = ("S_L alone", make_S(L))
    for p in P_NAMES:
        fns[p] = ("P alone", make_P(p))
    rows = []
    t0 = time.time()
    for name, (spec, fn) in fns.items():
        nla = bool(H.no_lookahead(fn))
        res = H.evaluate(fn(P))
        rows.append({"name": name, "spec": spec, "no_lookahead": nla, "result": res})
        f10, b10 = res["front"]["h10"], res["back"]["h10"]
        print(f"{name:22s} nla={nla} IC10 f {f10['ic']:+.3f} ({f10['ic_t']:+.2f}) b {b10['ic']:+.3f} "
              f"({b10['ic_t']:+.2f}) | top20_10 f {f10['top20']:+.2f} b {b10['top20']:+.2f} "
              f"| {time.time()-t0:.0f}s", flush=True)
    path = os.path.join(RESULTS, f"{FAMILY}_rankers.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"family": FAMILY, "part": "rankers", "n_evaluations": len(rows),
                   "benchmarks": H.benchmarks(), "rows": rows}, fh, ensure_ascii=False, indent=1)
    print("saved", path)


# ---------- 실행: 1)·3)·4) ----------
def features(P) -> dict:
    return {"S": {L: strength(P, L) for L in LS}, "P": {p: pullback(P, p) for p in P_NAMES}}


def run_sort():
    t0 = time.time()
    P = H.load_panel()
    U = H.universe_mask(P)
    E = H.eligible()
    feat = features(P)

    # 미래 정보 시험: 그룹 bool을 점수(0/1)로 바꿔 harness.no_lookahead에 넣는다(대표 4조합 × 4그룹).
    nla = {}
    for L, p in ((20, "P1_4"), (60, "P2_10"), (120, "P3_20"), (60, "P4_5")):
        for gname in ("SP_hi", "SP_lo", "strong", "weak_pulled"):
            fn = (lambda L, p, gname: (lambda Pc: groups(strength(Pc, L), pullback(Pc, p),
                                                         H.universe_mask(Pc))[gname].astype(float)))(L, p, gname)
            nla[f"{combo_name(L, p)}:{gname}"] = bool(H.no_lookahead(fn))
    print("no_lookahead groups", all(nla.values()), f"{time.time()-t0:.0f}s", flush=True)

    table, series = sort_table(feat, U, E, P, keep_series=True)
    print("table done", f"{time.time()-t0:.0f}s", flush=True)

    # 4) 부호 집계
    counts = {}
    for h in HS:
        f = {n: table[n][f"h{h}_front"]["diff"]["mean"] for n in table}
        b = {n: table[n][f"h{h}_back"]["diff"]["mean"] for n in table}
        y = {n: table[n][f"h{h}_y2025"]["diff"]["mean"] for n in table}
        names = list(table)
        fp = sum(f[n] > 0 for n in names)
        bp = sum(b[n] > 0 for n in names)
        both = sum(f[n] > 0 and b[n] > 0 for n in names)
        agree = sum((f[n] > 0) == (b[n] > 0) for n in names)
        best = max(names, key=lambda n: f[n])
        back_rank = sorted(names, key=lambda n: -b[n]).index(best) + 1
        counts[f"h{h}"] = {
            "n": len(names), "front_pos": fp, "back_pos": bp, "both_pos": both, "y2025_pos": sum(y[n] > 0 for n in names),
            "sign_agree": agree, "sign_agree_binom_p": round(binom_two_sided(agree, len(names)), 4),
            "front_pos_binom_p": round(binom_two_sided(fp, len(names)), 4),
            "back_pos_binom_p": round(binom_two_sided(bp, len(names)), 4),
            "spearman_front_back": round(spearman([f[n] for n in names], [b[n] for n in names]), 3),
            "spearman_front_y2025": round(spearman([f[n] for n in names], [y[n] for n in names]), 3),
            "front_best": best, "front_best_front": f[best], "front_best_back": b[best],
            "front_best_back_t": table[best][f"h{h}_back"]["diff"]["t"], "front_best_back_rank": back_rank,
            "back_median": round(float(np.median([b[n] for n in names])), 3),
            "front_median": round(float(np.median([f[n] for n in names])), 3),
        }
    print(json.dumps(counts["h10"], ensure_ascii=False), flush=True)

    # 갈래 전체(선택 없음): 날짜마다 36조합 차이의 단순 평균 → NW t. 그리고 L별 12개 평균.
    reg = regime(P)
    family_avg = {}
    for h in HS:
        for w in WINS:
            allc = pd.concat({n: series[(n, h, w)] for n in table}, axis=1)
            avg = allc.mean(axis=1)
            rg = reg.reindex(avg.index).fillna(False)
            d = {"all36": stat(avg, h), "all36_above_sma50": stat(avg[rg], h), "all36_below_sma50": stat(avg[~rg], h)}
            for L in LS:
                d[f"S{L}_12"] = stat(allc[[combo_name(L, p) for p in P_NAMES]].mean(axis=1), h)
            for k in ("P1", "P2", "P3", "P4"):
                d[f"{k}_9"] = stat(allc[[combo_name(L, p) for L in LS for p in P_NAMES if p.startswith(k)]].mean(axis=1), h)
            family_avg[f"h{h}_{w}"] = d
    fam_month = pd.concat({n: all_dates_series(groups(feat["S"][int(n.split('x')[0][1:])], feat["P"][n.split('x')[1]], U),
                                               10, P, E) for n in table}, axis=1).mean(axis=1)
    family_avg["monthly_h10_all36"] = monthly(fam_month)
    # 강함 안에서 SP_hi가 강함 전체보다 나은가(사는 쪽 비교, 날짜 평균의 차)
    hi_vs_strong = {}
    for h in HS:
        f = {n: table[n][f"h{h}_front"]["SP_hi"]["mean"] - table[n][f"h{h}_front"]["strong"]["mean"] for n in table}
        b = {n: table[n][f"h{h}_back"]["SP_hi"]["mean"] - table[n][f"h{h}_back"]["strong"]["mean"] for n in table}
        hi_vs_strong[f"h{h}"] = {"front_pos": sum(v > 0 for v in f.values()), "back_pos": sum(v > 0 for v in b.values()),
                                 "both_pos": sum(f[n] > 0 and b[n] > 0 for n in table),
                                 "front_median": round(float(np.median(list(f.values()))), 3),
                                 "back_median": round(float(np.median(list(b.values()))), 3)}
    print(json.dumps({k: v["all36"] for k, v in family_avg.items() if k.startswith("h10")}, ensure_ascii=False), flush=True)

    # 3) 앞·뒤 모두 양(h10)인 조합 점검
    cands = [n for n in table if table[n]["h10_front"]["diff"]["mean"] > 0 and table[n]["h10_back"]["diff"]["mean"] > 0]
    checks = {}
    for name in cands:
        L = int(name.split("x")[0][1:])
        p = name.split("x")[1]
        g = groups(feat["S"][L], feat["P"][p], U)
        c = {}
        for w in ("front", "back"):
            con = contributions(g, 10, w, P, E)
            top = con.index[0]
            # 한 조합만 다시 계산(우주에서 그 종목을 빼고 순위부터)
            U2 = U.copy()
            U2[top] = False
            g2 = groups(feat["S"][L], feat["P"][p], U2)
            dates = H.window_dates(w, 10)
            cols = window_cols(P, w, (top,))
            hi, _ = daily_excess(g2["SP_hi"], 10, dates, cols, E)
            lo, _ = daily_excess(g2["SP_lo"], 10, dates, cols, E)
            U3 = U.copy()
            U3["SNDK"] = False
            g3 = groups(feat["S"][L], feat["P"][p], U3)
            cols3 = window_cols(P, w, ("SNDK",))
            hi3, _ = daily_excess(g3["SP_hi"], 10, dates, cols3, E)
            lo3, _ = daily_excess(g3["SP_lo"], 10, dates, cols3, E)
            base = series[(name, 10, w)]
            rg = reg.reindex(base.index)
            c[w] = {
                "base": stat(base, 10),
                "top_contrib": [{"sym": s, "pp": round(float(v), 3)} for s, v in con.head(5).items()],
                "bottom_contrib": [{"sym": s, "pp": round(float(v), 3)} for s, v in con.tail(3).items()],
                "drop_top_stock": {"sym": top, **stat((hi - lo).dropna(), 10)},
                "drop_SNDK": {"SNDK_contrib_pp": round(float(con.get("SNDK", 0.0)), 3), **stat((hi3 - lo3).dropna(), 10)},
                "drop_top5_dates": drop_dates_stat(base, 10, 5),
                "regime_qqq_above_sma50": stat(base[rg.fillna(False)], 10),
                "regime_qqq_below_sma50": stat(base[~rg.fillna(False)], 10),
                "SP_hi_h10": table[name][f"h10_{w}"]["SP_hi"], "SP_lo_h10": table[name][f"h10_{w}"]["SP_lo"],
            }
        y = series[(name, 10, "y2025")]
        ry = reg.reindex(y.index).fillna(False)
        c["y2025_regime"] = {"above": stat(y[ry], 10), "below": stat(y[~ry], 10)}
        c["monthly_h10"] = monthly(all_dates_series(g, 10, P, E))
        checks[name] = c
        print("checked", name, f"{time.time()-t0:.0f}s", flush=True)

    # 비교용: 모든 36조합의 h10 차이 월별(2026)과 장세별(front·back) — 후보 아닌 것까지 같은 잣대로
    regime_all = {}
    for name in table:
        out = {}
        for w in ("front", "back", "y2025"):
            s = series[(name, 10, w)]
            rg = reg.reindex(s.index).fillna(False)
            out[w] = {"above": stat(s[rg], 10)["mean"], "above_days": int(rg.sum()),
                      "below": stat(s[~rg], 10)["mean"], "below_days": int((~rg).sum())}
        regime_all[name] = out
    reg_share = {w: round(float(reg.reindex(H.window_dates(w, 10)).fillna(False).mean()) * 100, 1) for w in WINS}

    res = {
        "family": FAMILY,
        "spec": {
            "strong": "pct(S_L) >= 0.70 in universe_mask", "weak": "pct(S_L) <= 0.30",
            "thirds": "within-group pct rank of P: SP_hi > 2/3, SP_lo <= 1/3; weak_pulled = weak & P pct > 2/3",
            "excess": "group mean fwd(t+1 open -> t+1+h open) minus eligible mean, %p, date-averaged; NW t lag=h",
            "primary": "diff SP_hi - SP_lo at h=10, pick front best, confirm on back",
            "L": list(LS), "P": P_NAMES, "h": list(HS), "windows": list(WINS),
            "search_size": "36 combos x 3 h x 4 windows x 5 metrics (double sort) + 51 rankers (harness.evaluate)",
            "regime": "QQQ close(t) > SMA50(t) (only pre-registered split)",
            "cost_note": "one-way 0.1% spread = 0.2%p per round trip; SP_hi vs SP_lo diff is cost-neutral when choosing between two buys",
        },
        "no_lookahead_groups": nla,
        "regime_share_above_pct": reg_share,
        "table": table,
        "counts": counts,
        "family_avg": family_avg,
        "hi_vs_strong": hi_vs_strong,
        "candidates_h10_both_pos": cands,
        "checks": checks,
        "regime_all_h10": regime_all,
    }
    path = os.path.join(RESULTS, f"{FAMILY}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("saved", path, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("sort", "all"):
        run_sort()
    if what in ("rank", "all"):
        run_rank()
