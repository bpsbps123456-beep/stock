"""계열 'ladder' — 점수가 아니라 S~D '등급 사다리' 설계를 잰다 (2026-10-12).

기준 점수 3개 × 매핑(상대 피라미드·5분위·혼합 관문) + 절대 규칙 2개(점수와 무관, 한 번만).
채점은 harness 방식 그대로: 등급은 eligible(보유+봉130+다음 날 시가) 종목에만 매기고,
초과수익은 그날 eligible 평균 대비, t일 종가 등급 → t+1 시가 매수 → t+1+h 시가 매도.

실행: cd data/derived/analysis/trend_grade/code && PYTHONIOENCODING=utf-8 py fam_ladder.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import harness as H

FAMILY = "ladder"
GR = H.GRADES  # S A B C D
CODE = {"S": 4, "A": 3, "B": 2, "C": 1, "D": 0}
LADDER_WINDOWS = ("front", "back", "y2025")
EVENTS = {"july_shock": ("2026-06-22", "2026-07-29"), "rebound": ("2026-07-29", "2026-09-30")}


# ---------------------------------------------------------------- 지표 조각
def sma(C: pd.DataFrame, n: int) -> pd.DataFrame:
    return C.rolling(n, min_periods=min(n, 120)).mean()


def high_n(Hi: pd.DataFrame, n: int) -> pd.DataFrame:
    return Hi.rolling(n, min_periods=min(n, 120)).max()


def smas(P: dict, ns=(10, 20, 50, 100, 200)) -> dict:
    return {n: sma(P["C"], n) for n in ns}


# ---------------------------------------------------------------- 기준 점수
def score_L(P: dict) -> pd.DataFrame:
    C = P["C"]
    return C / sma(C, 200) - 1


def score_T(P: dict) -> pd.DataFrame:
    C = P["C"]
    S = smas(P)
    pos = sum((C > S[n]).astype(float) for n in (10, 20, 50, 100, 200))
    ali = sum((S[a] > S[b]).astype(float) for a, b in ((10, 20), (20, 50), (50, 100), (100, 200)))
    r50 = C / S[50] - 1
    tb = H.pct_rank(r50, H.universe_mask(P) & r50.notna())
    T = pos + ali + 0.001 * tb
    return T.where(S[200].notna() & C.notna() & tb.notna())


def score_E2(P: dict) -> pd.DataFrame:
    C, Hi = P["C"], P["H"]
    m = H.universe_mask(P)
    feats = [C / sma(C, 200) - 1, C / sma(C, 50) - 1, C / sma(C, 20) - 1, C / high_n(Hi, 252) - 1]
    ranks = [H.pct_rank(f, m & f.notna()) for f in feats]
    return sum(ranks) / 4.0  # 하나라도 NaN이면 NaN


SCORES = {
    "L": (score_L, "C/SMA200-1. SMA200 rolling(200, min_periods=120)."),
    "T": (score_T, "SMA{10,20,50,100,200} 위 개수(0-5) + 인접 정배열 개수(10>20,20>50,50>100,100>200; 0-4) "
                   "+ 0.001×pct_rank(C/SMA50-1, universe_mask). SMA rolling(n, min_periods=min(n,120)); "
                   "SMA200 NaN 또는 universe 밖이면 NaN."),
    "E2": (score_E2, "{C/SMA200-1, C/SMA50-1, C/SMA20-1, C/252일 최고 고가-1}의 universe_mask 단면 백분위 평균. "
                     "252일 고가 rolling(252, min_periods=120), 넷 중 하나라도 NaN이면 NaN."),
}


# ---------------------------------------------------------------- 매핑
def _obj(shape_like: pd.DataFrame) -> np.ndarray:
    return np.full(shape_like.shape, None, dtype=object)


def rel_grades(score: pd.DataFrame, cuts, mask: pd.DataFrame) -> pd.DataFrame:
    pr = H.pct_rank(score, mask & score.notna())
    return H.grade_of(pr, cuts)


def hyb_grades(score: pd.DataFrame, P: dict, mask: pd.DataFrame) -> pd.DataFrame:
    """상대 피라미드 + 관문: S·A는 C>SMA50>SMA200일 때만(아니면 B), C<SMA200 그리고 SMA50<SMA200이면 D."""
    C = P["C"]
    s50, s200 = sma(C, 50), sma(C, 200)
    g = rel_grades(score, H.PYRAMID, mask).to_numpy(object).copy()
    gate = ((C > s50) & (s50 > s200)).to_numpy()
    dcond = ((C < s200) & (s50 < s200)).to_numpy()
    graded = pd.notna(g)
    top = graded & ((g == "S") | (g == "A"))
    g[top & ~gate] = "B"
    g[graded & dcond] = "D"
    return pd.DataFrame(g, index=C.index, columns=C.columns)


def abs1_grades(P: dict, mask: pd.DataFrame) -> pd.DataFrame:
    C = P["C"]
    s20, s50, s200 = sma(C, 20), sma(C, 50), sma(C, 200)
    s50_10 = s50.shift(10)
    h252 = high_n(P["H"], 252)
    valid = (mask & C.notna() & s20.notna() & s50.notna() & s200.notna() & s50_10.notna() & h252.notna()).to_numpy()
    S = ((C > s20) & (s20 > s50) & (s50 > s200) & (s50 > s50_10) & (C >= 0.9 * h252)).to_numpy()
    A = ((C > s50) & (s50 > s200)).to_numpy()
    B = (C > s200).to_numpy()
    Cc = ((C > s50) | (s50 > s200)).to_numpy()
    g = _obj(C)
    g[valid] = "D"
    g[valid & Cc] = "C"
    g[valid & B] = "B"
    g[valid & A] = "A"
    g[valid & S] = "S"
    return pd.DataFrame(g, index=C.index, columns=C.columns)


def abs2_grades(P: dict, mask: pd.DataFrame) -> pd.DataFrame:
    C = P["C"]
    s20, s50, s100, s200 = sma(C, 20), sma(C, 50), sma(C, 100), sma(C, 200)
    valid = (mask & C.notna() & s20.notna() & s50.notna() & s100.notna() & s200.notna()).to_numpy()
    k = sum((C > s).astype(int) for s in (s20, s50, s100, s200)).to_numpy()
    aligned = ((s20 > s50) & (s50 > s100) & (s100 > s200)).to_numpy()
    s50_200 = (s50 > s200).to_numpy()
    g = _obj(C)
    g[valid & (k == 0)] = "D"
    g[valid & (k == 1)] = "C"
    g[valid & ((k == 2) | (k == 3))] = "B"
    g[valid & (((k == 4) & ~aligned) | ((k == 3) & s50_200))] = "A"
    g[valid & (k == 4) & aligned] = "S"
    return pd.DataFrame(g, index=C.index, columns=C.columns)


HYST_BAND = 0.05
_LO = {"S": 0.90, "A": 0.70, "B": 0.30, "C": 0.10, "D": -np.inf}
_HI = {"S": np.inf, "A": 0.90, "B": 0.70, "C": 0.30, "D": 0.10}


def hyst_grades(score: pd.DataFrame, mask: pd.DataFrame, band: float = HYST_BAND) -> pd.DataFrame:
    """extra: 피라미드 + 이력(hysteresis). 어제 등급 k이고 오늘 백분위가 k 구간 ±band 안이면 k 유지,
    아니면 보통 피라미드 등급. 어제 등급이 없으면(비보유·비eligible) 보통 등급. 과거 경로만 쓴다."""
    pr = H.pct_rank(score, mask & score.notna())
    vals = pr.to_numpy(float)
    plain = H.grade_of(pr, H.PYRAMID).to_numpy(object)
    out = np.full(vals.shape, None, dtype=object)
    for i in range(vals.shape[0]):
        g = plain[i].copy()
        if i > 0:
            p, v = out[i - 1], vals[i]
            fin = np.isfinite(v)
            for k in GR:
                keep = fin & (p == k) & (v > _LO[k] - band) & (v <= _HI[k] + band)
                g[keep] = k
        out[i] = g
    return pd.DataFrame(out, index=pr.index, columns=pr.columns)


def code_of(G: pd.DataFrame) -> pd.DataFrame:
    v = G.to_numpy(object)
    out = np.full(v.shape, np.nan)
    for k, c in CODE.items():
        out[v == k] = c
    return pd.DataFrame(out, index=G.index, columns=G.columns)


# 미래 정보 점검용: 실제 운영형(순위는 universe_mask) 등급 코드 함수
def prod_code_fn(kind: str, base=None):
    def fn(P):
        m = H.universe_mask(P)
        if kind == "ABS1":
            return code_of(abs1_grades(P, m))
        if kind == "ABS2":
            return code_of(abs2_grades(P, m))
        if kind == "HYB":
            return code_of(hyb_grades(base(P), P, m))
        if kind == "HYST":
            return code_of(hyst_grades(base(P), m))
        raise ValueError(kind)
    return fn


# ---------------------------------------------------------------- 사다리 채점
def _r(x, k=3):
    x = float(x)
    return round(x, k) if np.isfinite(x) else None


def streaks(isin: pd.DataFrame, a: str, b: str) -> dict:
    """연속 유지 일수(거래일). 시작일이 구간 안인 연속 구간만 센다. 패널 끝에서 잘린 구간 포함."""
    arr = isin.to_numpy(bool)
    cal = isin.index
    lo, hi = np.searchsorted(cal, pd.Timestamp(a)), np.searchsorted(cal, pd.Timestamp(b), side="right")
    lens = []
    for j in range(arr.shape[1]):
        x = arr[:, j]
        if not x.any():
            continue
        d = np.diff(np.concatenate(([0], x.astype(int), [0])))
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        for s, e in zip(st, en):
            if lo <= s < hi:
                lens.append(e - s)
    if not lens:
        return {"runs": 0, "mean": None, "median": None}
    lens = np.array(lens)
    return {"runs": int(len(lens)), "mean": _r(lens.mean(), 1), "median": _r(np.median(lens), 1),
            "p90": _r(np.percentile(lens, 90), 1)}


def event_table(G: pd.DataFrame, d0: str, d1: str) -> dict:
    P = H.load_panel()
    C = P["C"]
    r = C.loc[d1] / C.loc[d0] - 1
    g = G.loc[d0]
    ok = g.notna() & r.notna()
    base = float(r[ok].mean())
    out = {"all": {"n": int(ok.sum()), "ret": _r(base * 100, 2)},
           "QQQ": _r((P["bench"]["C"].loc[d1] / P["bench"]["C"].loc[d0] - 1) * 100, 2)}
    for k in GR:
        sel = ok & (g == k)
        if sel.sum() == 0:
            out[k] = {"n": 0, "ret": None, "ex": None}
            continue
        m = float(r[sel].mean())
        out[k] = {"n": int(sel.sum()), "ret": _r(m * 100, 2), "ex": _r((m - base) * 100, 2)}
    vals = [out[k]["ret"] for k in GR]
    out["mono"] = all(v is not None for v in vals) and all(vals[i] > vals[i + 1] for i in range(4))
    return out


def _mono(d: dict) -> tuple[bool, int]:
    vals = [d[k] for k in GR]
    inv = sum(1 for i in range(4) if vals[i] is not None and vals[i + 1] is not None and vals[i] <= vals[i + 1])
    full = all(v is not None for v in vals)
    return (full and inv == 0), inv


def ladder_stats(G: pd.DataFrame, windows=LADDER_WINDOWS) -> dict:
    P = H.load_panel()
    E = H.eligible()
    G = G.reindex(index=P["C"].index, columns=P["C"].columns)
    G = G.where(E)  # 채점은 eligible만
    code = code_of(G)
    mdd = H.forward_mdd20()
    last = P["C"].index[-1].strftime("%Y-%m-%d")
    isS = G == "S"
    isSA = G.isin(["S", "A"])
    res = {}
    for w in windows:
        excl = H.EXCLUDE.get(w, set())
        cols = [c for c in P["C"].columns if c not in excl]
        o = {}
        d1 = H.window_dates(w, 1)
        g1 = G.loc[d1, cols]
        n_el = E.loc[d1, cols].sum(axis=1)
        n_gr = g1.notna().sum(axis=1)
        o["eligible_mean"] = _r(n_el.mean(), 1)
        o["ungraded_mean"] = _r((n_el - n_gr).mean(), 2)
        cnt = {k: (g1 == k).sum(axis=1) for k in GR}
        o["count"] = {k: _r(cnt[k].mean(), 1) for k in GR}
        o["share"] = {k: _r((cnt[k] / n_gr.replace(0, np.nan)).mean() * 100, 1) for k in GR}
        o["S_zero_days"] = _r((cnt["S"] == 0).mean() * 100, 1)
        o["S_count_min_med_max"] = [int(cnt["S"].min()), _r(cnt["S"].median(), 1), int(cnt["S"].max())]
        o["D_zero_days"] = _r((cnt["D"] == 0).mean() * 100, 1)
        for h in (10, 20):
            dates = H.window_dates(w, h)
            fr = H.forward(h).loc[dates, cols].where(E.loc[dates, cols])
            mean_all = fr.mean(axis=1)
            g = G.loc[dates, cols]
            ex, days, hit = {}, {}, {}
            for k in GR:
                s = (fr.where(g == k).mean(axis=1) - mean_all).dropna()
                ex[k] = _r(s.mean() * 100) if len(s) else None
                days[k] = int(len(s))
                hit[k] = _r((s > 0).mean() * 100, 1) if len(s) else None
            mono, inv = _mono(ex)
            ic = H._row_spearman(code.loc[dates, cols], fr).dropna()
            d = {"ex": ex, "days": days, "hit": hit, "mono": mono, "inversions": inv,
                 "S_minus_D": _r(ex["S"] - ex["D"]) if ex["S"] is not None and ex["D"] is not None else None,
                 "grade_ic": _r(ic.mean(), 4), "grade_ic_t": _r(H.newey_west_t(ic, h), 2), "n_dates": int(len(dates))}
            if h == 20:
                md = mdd.loc[dates, cols].where(E.loc[dates, cols])
                d["mdd20"] = {k: _r(md.where(g == k).mean(axis=1).mean() * 100, 2) for k in GR}
                d["mdd20"]["all"] = _r(md.mean(axis=1).mean() * 100, 2)
            o[f"h{h}"] = d
        prev = G.shift(1).loc[d1, cols]
        both = g1.notna() & prev.notna()
        o["turnover"] = _r(((g1 != prev) & both).sum().sum() / max(both.sum().sum(), 1) * 100, 1)
        a, b = H.WINDOWS[w]
        b = min(b, last)
        o["streak_S"] = streaks(isS[cols], a, b)
        o["streak_SA"] = streaks(isSA[cols], a, b)
        o["port_S"] = H.weekly_portfolio(isS.astype(float), 0.5, a, b, E)
        o["port_SA"] = H.weekly_portfolio(isSA.astype(float), 0.5, a, b, E)
        res[w] = o
    res["events"] = {name: event_table(G, d0, d1) for name, (d0, d1) in EVENTS.items()}
    return res


# ---------------------------------------------------------------- 실행
MAP_SPECS = {
    "REL_P": "상대 피라미드: eligible 단면 백분위 상위 10% S / 다음 20% A / 40% B / 20% C / 하위 10% D (harness grade_of PYRAMID).",
    "REL_Q": "상대 5분위: 상위 20%씩 S/A/B/C/D (harness grade_of QUINTILE).",
    "HYB": "상대 피라미드 + 관문: S·A는 C>SMA50 그리고 SMA50>SMA200일 때만(아니면 B); "
           "C<SMA200 그리고 SMA50<SMA200이면 백분위와 무관하게 D.",
    "ABS1": "절대: S=C>SMA20>SMA50>SMA200 & SMA50>SMA50(10일 전) & C>=0.9×252일 최고 고가; A=(S 아님) C>SMA50>SMA200; "
            "B=(위 아님) C>SMA200; C=(위 아님) C>SMA50 또는 SMA50>SMA200; D=나머지. 점수와 무관.",
    "ABS2": "절대: k=SMA{20,50,100,200} 중 C가 위인 개수. S=k4 & SMA20>50>100>200; A=k4(정배열 아님) 또는 (k3 & SMA50>SMA200); "
            "B=그 외 k2·k3; C=k1; D=k0. 점수와 무관.",
}


def main():
    t0 = time.time()
    P = H.load_panel()
    E = H.eligible()
    rows = []
    base_cache = {}
    for sname, (fn, sspec) in SCORES.items():
        sc = fn(P).reindex(index=P["C"].index, columns=P["C"].columns)
        nl = H.no_lookahead(fn)
        ev = H.evaluate(sc)
        base_cache[sname] = (sc, nl, ev)
        print(H.summary_line(f"{sname} (base)", ev), "| no_lookahead", nl, f"| {time.time()-t0:.0f}s")
        for mname in ("REL_P", "REL_Q", "HYB"):
            if mname == "REL_P":
                G = rel_grades(sc, H.PYRAMID, E)
                nl_map, nl_detail = nl, {"base": nl}
            elif mname == "REL_Q":
                G = rel_grades(sc, H.QUINTILE, E)
                nl_map, nl_detail = nl, {"base": nl}
            else:
                G = hyb_grades(sc, P, E)
                nl_prod = H.no_lookahead(prod_code_fn("HYB", fn))
                nl_map, nl_detail = nl and nl_prod, {"base": nl, "hyb_code_universe_mask": nl_prod}
            lad = ladder_stats(G)
            result = dict(ev) if mname != "HYB" else {}
            result["ladder"] = lad
            rows.append({"name": f"{sname}_{mname}", "spec": f"기준 점수 {sname}: {sspec} | 매핑 {mname}: {MAP_SPECS[mname]}",
                         "result": result, "no_lookahead": bool(nl_map), "no_lookahead_detail": nl_detail})
            print_ladder(f"{sname}_{mname}", lad)
    for mname, gfn in (("ABS1", abs1_grades), ("ABS2", abs2_grades)):
        G = gfn(P, E)
        nl = H.no_lookahead(prod_code_fn(mname))
        lad = ladder_stats(G)
        rows.append({"name": mname, "spec": f"매핑 {mname}: {MAP_SPECS[mname]} SMA rolling(n, min_periods=min(n,120)), "
                                            f"252일 고가 rolling(252, min_periods=120).",
                     "result": {"ladder": lad}, "no_lookahead": bool(nl), "no_lookahead_detail": {"code": nl}})
        print_ladder(mname, lad)
    # 격자 밖(extra_): 이력 띠 ±0.05 피라미드. 격자 결과를 본 뒤 추가 — 회전율·S 유지 일수 진단용.
    for sname, (fn, sspec) in SCORES.items():
        sc, nl, _ = base_cache[sname]
        G = hyst_grades(sc, E)
        nl_prod = H.no_lookahead(prod_code_fn("HYST", fn))
        lad = ladder_stats(G)
        rows.append({"name": f"extra_{sname}_REL_P_hyst",
                     "spec": f"기준 점수 {sname}: {sspec} | 매핑: 상대 피라미드 + 이력 띠 ±{HYST_BAND} "
                             f"(어제 등급 구간 ±{HYST_BAND} 안이면 유지). 격자 결과를 본 뒤 추가한 extra.",
                     "result": {"ladder": lad}, "no_lookahead": bool(nl and nl_prod),
                     "no_lookahead_detail": {"base": nl, "hyst_code_universe_mask": nl_prod}})
        print_ladder(f"extra_{sname}_REL_P_hyst", lad)
    path = H.save_results(FAMILY, rows)
    print("saved", path, f"{time.time()-t0:.0f}s")
    return rows


def print_ladder(name: str, lad: dict):
    print(f"== {name}")
    for w in LADDER_WINDOWS:
        o = lad[w]
        h20, h10 = o["h20"], o["h10"]
        print(f"  {w:6s} share {o['share']} Szero {o['S_zero_days']}% S#{o['S_count_min_med_max']} turn {o['turnover']}% "
              f"Srun {o['streak_S']['mean']}")
        print(f"         h20 ex {h20['ex']} mono {h20['mono']} gIC {h20['grade_ic']} (t {h20['grade_ic_t']}) mdd {h20['mdd20']}")
        print(f"         h10 ex {h10['ex']} mono {h10['mono']} gIC {h10['grade_ic']} | portS {o['port_S']} portSA {o['port_SA']}")
    for ev, t in lad["events"].items():
        print(f"  {ev}: " + " ".join(f"{k}:{t[k]['ret']}({t[k]['n']})" for k in GR) + f" all {t['all']['ret']} QQQ {t['QQQ']} mono {t['mono']}")


if __name__ == "__main__":
    main()
