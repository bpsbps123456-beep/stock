"""cross_section 갈래 독립 반박 검증 (2026-10-13). cross_section*.py를 import하지 않는다(harness만).

1) 재계산: S60xP4_20 h10 차이(front/back/2025/2024, back 순위), 36조합 평균, QQQ 50일선 장세 나눔,
   강함 안 SP_lo − 강함 전체, 합성 pct(S120)+pct(P1_4)·S60·P1_4 단독의 IC10·상위20%(h10).
2) 시점: 내 그룹·합성 점수를 harness.no_lookahead로, forward(h) 정렬을 손으로 확인. 3등분 방향(P 부호) 표본 확인.
3) 반박: (a) 36조합 front 최대값·108셀 최대 t의 우연 확률(10/20일 블록 부호 뒤집기), back 30/36 양의 우연 확률,
   51개 순위 중 min(front, back) 최대의 우연 확률; (b) 종목·하루 쏠림, 7월 급락·반등 제외; (c) 편도 0.1% 호가 비용;
   (d) 2025 부호; (e) 동점 규칙 바꾸기(무작위 동점 깨기로 3등분 크기 같게); (f) 사전 베타(60일)로 뺀 단면 초과.

실행: PYTHONIOENCODING=utf-8 py verify_cross_section.py
결과: ../results/verify_cross_section.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "trend_grade", "code")))
import harness as H  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "results", "verify_cross_section.json"))
LS = (20, 60, 120)
PN = ["P1_3", "P1_4", "P1_5", "P2_5", "P2_10", "P2_20", "P3_5", "P3_10", "P3_20", "P4_5", "P4_10", "P4_20"]
HS = (5, 10, 20)
WINS = ("front", "back", "y2025", "y2024")
RNG = np.random.default_rng(20261013)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


# ---------------- 특징 (독립 구현) ----------------
def feats(P: dict) -> tuple[dict, dict]:
    C = P["C"]
    Q = P["bench"]["C"].reindex(C.index)
    lc, lq = np.log(C), np.log(Q)

    def ex(n):  # 100*(C/C[n] - Q/Q[n])
        return 100.0 * (np.exp(lc - lc.shift(n)).sub(np.exp(lq - lq.shift(n)), axis=0))

    S = {L: ex(L) for L in LS}
    e4 = ex(4)
    R = C.div(Q, axis=0)
    Pm = {}
    for n in (3, 4, 5):
        Pm[f"P1_{n}"] = -ex(n)
    for k in (5, 10, 20):
        Pm[f"P2_{k}"] = e4.rolling(k, min_periods=k).max() - e4
        Pm[f"P3_{k}"] = 1.0 - R / R.rolling(k, min_periods=k).max()
        Pm[f"P4_{k}"] = 1.0 - C / C.rolling(k, min_periods=k).max()
    return S, Pm


def thirds(S: pd.DataFrame, Pm: pd.DataFrame, U: pd.DataFrame, tie: str = "min", seed: int = 0) -> dict:
    """강함 = U 안 S 백분위(평균 순위) >= 0.70. 강함 안 P 3등분.
    tie='min': 순위 = (P가 더 작은 강함 종목 수 + 1)/n  → >2/3 SP_hi, <=1/3 SP_lo (연구와 같은 정의, 다른 코드)
    tie='rand': 작은 무작위 잡음으로 동점을 깨서 정확히 같은 크기의 3등분(floor(n/3)씩)."""
    s = S.where(U).to_numpy(float)
    p = Pm.to_numpy(float)
    nT, nN = s.shape
    hi = np.zeros((nT, nN), bool)
    lo = np.zeros((nT, nN), bool)
    st = np.zeros((nT, nN), bool)
    wk = np.zeros((nT, nN), bool)
    wp = np.zeros((nT, nN), bool)
    rng = np.random.default_rng(seed)
    for t in range(nT):
        ok = np.isfinite(s[t])
        n = ok.sum()
        if n < 10:
            continue
        v = s[t, ok]
        # 평균 순위 백분위
        order = v.argsort(kind="mergesort")
        r = np.empty(n)
        r[order] = np.arange(1, n + 1)
        # 동점 평균
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        r = sums[inv] / cnt[inv]
        pct = r / n
        idx = np.flatnonzero(ok)
        st[t, idx[pct >= 0.70]] = True
        wk[t, idx[pct <= 0.30]] = True
        for grp, out_hi, out_lo in ((st[t], hi, lo), (wk[t], wp, None)):
            g = np.flatnonzero(grp & np.isfinite(p[t]))
            m = len(g)
            if m < 3:
                continue
            pv = p[t, g]
            if tie == "min":
                less = (pv[None, :] < pv[:, None]).sum(axis=1)
                q = (less + 1) / m
                out_hi[t, g[q > 2 / 3]] = True
                if out_lo is not None:
                    out_lo[t, g[q <= 1 / 3]] = True
            else:
                k = m // 3
                o = np.argsort(pv + rng.normal(0, 1e-9, m) * (np.abs(pv).max() + 1e-6))
                out_hi[t, g[o[m - k:]]] = True
                if out_lo is not None:
                    out_lo[t, g[o[:k]]] = True
    mk = lambda a: pd.DataFrame(a, index=S.index, columns=S.columns)
    return {"hi": mk(hi), "lo": mk(lo), "strong": mk(st), "weak_pulled": mk(wp)}


# ---------------- 채점 ----------------
P = H.load_panel()
C, O = P["C"], P["O"]
Qc, Qo = P["bench"]["C"].reindex(C.index), P["bench"]["O"].reindex(C.index)
U = H.universe_mask(P)
E = H.eligible()
FR = {h: (O.shift(-(1 + h)) / O.shift(-1) - 1.0).where(E) for h in HS}  # 독립 계산(harness.forward와 비교)
REG_ABOVE = (Qc > Qc.rolling(50, min_periods=50).mean())
QF = {h: Qo.shift(-(1 + h)) / Qo.shift(-1) - 1.0 for h in HS}
r1, m1 = C.pct_change(), Qc.pct_change()
BETA = r1.rolling(60, min_periods=60).cov(m1).div(m1.rolling(60, min_periods=60).var(), axis=0)


def wcols(w, drop=()):
    ex_ = set(H.EXCLUDE.get(w, set())) | set(drop)
    return [c for c in C.columns if c not in ex_]


def gmean(mask: pd.DataFrame, h: int, dates, cols, fr=None) -> pd.Series:
    fr = (FR[h] if fr is None else fr).loc[dates, cols]
    return fr.where(mask.loc[dates, cols] & fr.notna()).mean(axis=1)


def diff_series(g: dict, h: int, w: str, drop=(), fr=None) -> pd.Series:
    d = H.window_dates(w, h)
    cols = wcols(w, drop)
    return (gmean(g["hi"], h, d, cols, fr) - gmean(g["lo"], h, d, cols, fr)).dropna()


def vs_all(gm: pd.DataFrame, h: int, w: str, fr=None) -> pd.Series:
    d = H.window_dates(w, h)
    cols = wcols(w)
    f = (FR[h] if fr is None else fr).loc[d, cols]
    return (f.where(gm.loc[d, cols] & f.notna()).mean(axis=1) - f.mean(axis=1)).dropna()


def st(x: pd.Series, h: int) -> dict:
    x = x.dropna()
    if not len(x):
        return {"mean": None, "t": None, "days": 0}
    return {"mean": round(float(x.mean()) * 100, 3), "t": round(H.newey_west_t(x, h), 2), "days": int(len(x)),
            "pos": round(float((x > 0).mean()) * 100, 1)}


def nw_t_matrix(X: np.ndarray, lag: int) -> np.ndarray:
    """열마다 NW t (NaN 없는 행렬)."""
    n = X.shape[0]
    mu = X.mean(axis=0)
    e = X - mu
    var = (e * e).sum(axis=0) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        var = var + 2 * w * (e[k:] * e[:-k]).sum(axis=0) / n
    return mu / np.sqrt(np.maximum(var, 1e-18) / n)


def block_signs(n: int, block: int, iters: int) -> np.ndarray:
    """iters x n 부호 행렬: 무작위 시작점의 블록마다 같은 부호."""
    out = np.empty((iters, n))
    for i in range(iters):
        off = RNG.integers(0, block)
        b = (np.arange(n) + off) // block
        sg = RNG.choice([-1.0, 1.0], size=b.max() + 1)
        out[i] = sg[b]
    return out


def episodes(idx: pd.DatetimeIndex, gap: int = 5) -> int:
    cal = C.index
    pos = np.sort(cal.get_indexer(idx))
    if not len(pos):
        return 0
    return int(1 + (np.diff(pos) > gap).sum())


res: dict = {"family": "cross_section", "role": "independent adversarial verifier", "notes": []}

# ---------- 0) 시점 점검 ----------
log("timing checks")
hf = H.forward(10)
same = np.allclose(FR[10].where(E).fillna(-9).to_numpy(), hf.where(E).fillna(-9).to_numpy())
spot = {}
for sym, day in (("NVDA", "2026-03-27"), ("SNDK", "2026-07-21")):
    if sym in C.columns:
        i = C.index.get_loc(pd.Timestamp(day))
        manual = O[sym].iloc[i + 11] / O[sym].iloc[i + 1] - 1
        spot[f"{sym}@{day}"] = {"entry": str(C.index[i + 1].date()), "exit": str(C.index[i + 11].date()),
                                "manual": round(float(manual) * 100, 3), "harness": round(float(hf[sym].iloc[i]) * 100, 3)}
S_all, P_all = feats(P)


def g_fn(L, p, key):
    def f(Pc):
        S2, P2 = feats(Pc)
        return thirds(S2[L], P2[p], H.universe_mask(Pc))[key].astype(float)
    return f


def comp_fn(L, p):
    def f(Pc):
        S2, P2 = feats(Pc)
        U2 = H.universe_mask(Pc)
        return S2[L].where(U2).rank(axis=1, pct=True) + P2[p].where(U2).rank(axis=1, pct=True)
    return f


nla = {"S60xP4_20:hi": H.no_lookahead(g_fn(60, "P4_20", "hi")),
       "S60xP4_20:lo": H.no_lookahead(g_fn(60, "P4_20", "lo")),
       "S120xP1_4:hi": H.no_lookahead(g_fn(120, "P1_4", "hi")),
       "comp_S120+P1_4": H.no_lookahead(comp_fn(120, "P1_4"))}
# 3등분 방향 표본: 2026-03-27 S60 강함 안 P4_20
g0 = thirds(S_all[60], P_all["P4_20"], U)
t0 = pd.Timestamp("2026-03-27")
samp = pd.DataFrame({"S60": S_all[60].loc[t0], "P4_20": P_all["P4_20"].loc[t0], "hi": g0["hi"].loc[t0],
                     "lo": g0["lo"].loc[t0], "strong": g0["strong"].loc[t0]})
samp = samp[samp["strong"]].sort_values("P4_20", ascending=False)
direction_ok = bool(samp.loc[samp["hi"], "P4_20"].min() >= samp.loc[samp["lo"], "P4_20"].max())
res["timing"] = {"forward_matches_harness": bool(same), "forward_spot": spot, "no_lookahead": nla,
                 "tercile_direction_ok_2026-03-27": direction_ok,
                 "sample_2026-03-27_S60xP4_20": {s: {"P4_20%": round(float(r.P4_20) * 100, 1),
                                                     "grp": "hi" if r.hi else ("lo" if r.lo else "mid")}
                                                 for s, r in samp.iterrows()},
                 "regime_known_at_t": "QQQ close(t) vs SMA50(t), both t close",
                 "group_sizes_front": None}
log("timing", res["timing"]["forward_matches_harness"], nla, direction_ok)

# ---------- 1) 36조합 재계산 ----------
log("groups 36 (min ties)")
G = {(L, p): thirds(S_all[L], P_all[p], U) for L in LS for p in PN}
names = [f"S{L}x{p}" for L in LS for p in PN]
D = {}  # (name, h, w) -> diff series
for (L, p), g in G.items():
    for h in HS:
        for w in WINS:
            D[(f"S{L}x{p}", h, w)] = diff_series(g, h, w)
sz = G[(60, "P4_20")]
fd = H.window_dates("front", 10)
res["timing"]["group_sizes_front"] = {k: round(float(sz[k].loc[fd].sum(axis=1).mean()), 1) for k in ("strong", "hi", "lo")}

tab = {n: {f"h{h}_{w}": st(D[(n, h, w)], h) for h in HS for w in WINS} for n in names}
f10 = {n: tab[n]["h10_front"]["mean"] for n in names}
b10 = {n: tab[n]["h10_back"]["mean"] for n in names}
best = max(names, key=lambda n: f10[n])
rk_back = sorted(names, key=lambda n: -b10[n]).index(best) + 1
fb_corr = float(pd.Series(f10).rank().corr(pd.Series(b10).rank()))
fam = {}
for h in HS:
    for w in WINS:
        A = pd.concat({n: D[(n, h, w)] for n in names}, axis=1).mean(axis=1)
        rg = REG_ABOVE.reindex(A.index).fillna(False)
        fam[f"h{h}_{w}"] = {"all36": st(A, h), "below_sma50": st(A[~rg], h), "above_sma50": st(A[rg], h),
                            "below_episodes": episodes(A.index[~rg])}
res["recomputed"] = {
    "front_best_h10": best, "front_best": tab[best]["h10_front"], "front_best_back": tab[best]["h10_back"],
    "front_best_back_rank": rk_back, "front_best_y2025": tab[best]["h10_y2025"], "front_best_y2024": tab[best]["h10_y2024"],
    "spearman_front_back_h10": round(fb_corr, 3),
    "pos_counts_h10": {w: sum(tab[n][f"h10_{w}"]["mean"] > 0 for n in names) for w in WINS},
    "both_pos_h10": sum(f10[n] > 0 and b10[n] > 0 for n in names),
    "family_avg": fam,
}
log("front best", best, tab[best]["h10_front"], "back", tab[best]["h10_back"], "rank", rk_back, "corr", round(fb_corr, 3))
log("all36 h10", {w: fam[f"h10_{w}"]["all36"]["mean"] for w in WINS})

# 강함 안 SP_lo − 강함 전체, SP_hi − 강함 전체 (36 평균) + 약함&눌림, SP_hi 절대(eligible 대비)
legs = {}
for w in WINS:
    lo_s, hi_s, hi_abs, wp_abs, st_abs = [], [], [], [], []
    for (L, p), g in G.items():
        sa = vs_all(g["strong"], 10, w)
        lo_s.append(vs_all(g["lo"], 10, w) - sa)
        hi_s.append(vs_all(g["hi"], 10, w) - sa)
        hi_abs.append(vs_all(g["hi"], 10, w))
        wp_abs.append(vs_all(g["weak_pulled"], 10, w))
        st_abs.append(sa)
    cat = lambda xs: pd.concat(xs, axis=1).mean(axis=1)
    hiA, wpA, stA = cat(hi_abs), cat(wp_abs), cat(st_abs)
    rg = REG_ABOVE.reindex(hiA.index).fillna(False)
    legs[w] = {"SP_lo_minus_strong": st(cat(lo_s), 10), "SP_hi_minus_strong": st(cat(hi_s), 10),
               "SP_hi_vs_eligible": st(hiA, 10), "weak_pulled_vs_eligible": st(wpA, 10),
               "strong_vs_eligible": st(stA, 10),
               "below_sma50": {"SP_hi_vs_eligible": st(hiA[~rg], 10), "weak_pulled_vs_eligible": st(wpA[~rg], 10),
                               "strong_vs_eligible": st(stA[~rg], 10),
                               "SP_hi_minus_weak_pulled": st((hiA - wpA)[~rg], 10)}}
res["legs_h10"] = legs
log("legs", {w: (legs[w]["SP_lo_minus_strong"]["mean"], legs[w]["below_sma50"]["SP_hi_vs_eligible"]["mean"],
                 legs[w]["below_sma50"]["weak_pulled_vs_eligible"]["mean"]) for w in WINS})

# ---------- 2) 합성·단독 순위 51개 (독립 top20·IC) ----------
log("rankers 51")


def rank_daily(score: pd.DataFrame, h: int, w: str, drop=()):
    d = H.window_dates(w, h)
    cols = wcols(w, drop)
    Ex = E.copy()
    if drop:
        Ex[list(drop)] = False
    sc = score.reindex(index=C.index, columns=C.columns)
    mask = Ex & sc.notna()
    pr = sc.where(mask).rank(axis=1, pct=True)
    fr = (O.shift(-(1 + h)) / O.shift(-1) - 1.0).where(mask).loc[d, cols]
    prd = pr.loc[d, cols]
    top = fr.where(prd > 0.8).mean(axis=1) - fr.mean(axis=1)
    ok = prd.notna() & fr.notna()
    a = prd.where(ok).rank(axis=1)
    b = fr.where(ok).rank(axis=1)
    a, b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    ic = (a * b).sum(axis=1) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    ic = ic.where(ok.sum(axis=1) >= 10)
    return top.dropna(), ic.dropna(), pr


scores = {}
for L in LS:
    for p in PN:
        scores[f"comp_S{L}+{p}"] = S_all[L].where(U).rank(axis=1, pct=True) + P_all[p].where(U).rank(axis=1, pct=True)
for L in LS:
    scores[f"S{L}"] = S_all[L]
for p in PN:
    scores[p] = P_all[p]
RK = {}
for nm, sc in scores.items():
    RK[nm] = {w: rank_daily(sc, 10, w)[:2] for w in WINS}
rk_tab = {nm: {w: {"top20": st(RK[nm][w][0], 10), "ic10": round(float(RK[nm][w][1].mean()), 4)} for w in WINS}
          for nm in scores}
key_rankers = ["S60", "S120", "P1_4", "comp_S120+P1_4", "comp_S120+P1_5", "comp_S60+P4_20"]
res["rankers_h10"] = {k: rk_tab[k] for k in key_rankers}
res["rankers_counts"] = {
    "n": len(scores),
    "ic10_both_pos": sum(rk_tab[k]["front"]["ic10"] > 0 and rk_tab[k]["back"]["ic10"] > 0 for k in scores),
    "top20_both_pos": sum(rk_tab[k]["front"]["top20"]["mean"] > 0 and rk_tab[k]["back"]["top20"]["mean"] > 0 for k in scores),
    "front_best_ic10": max(scores, key=lambda k: rk_tab[k]["front"]["ic10"]),
    "max_min_front_back_top20": max(scores, key=lambda k: min(rk_tab[k]["front"]["top20"]["mean"],
                                                              rk_tab[k]["back"]["top20"]["mean"])),
}
# 합성 점검: harness.evaluate와 일치
hv = H.evaluate(scores["comp_S120+P1_4"], windows=("front", "back"), portfolio=False)
res["rankers_harness_crosscheck"] = {w: {"ic10": hv[w]["h10"]["ic"], "top20": hv[w]["h10"]["top20"]} for w in ("front", "back")}
log("rankers", {k: (rk_tab[k]["front"]["ic10"], rk_tab[k]["back"]["ic10"], rk_tab[k]["front"]["top20"]["mean"],
                    rk_tab[k]["back"]["top20"]["mean"]) for k in key_rankers})
log("counts", res["rankers_counts"])

# ---------- 3a) 선택편향: 부호 뒤집기 블록 부트스트랩 ----------
log("bootstrap")
ITERS = 4000
boot = {}
# (i) front h10 36조합 최대 평균 (front 91일 중 S20xP3_5 하루 빈칸 → 열 평균을 뺀 뒤 0으로 채움, 관측값은 원 계열)
Mf = pd.concat({n: D[(n, 10, "front")] for n in names}, axis=1)
X = (Mf - Mf.mean()).fillna(0.0).to_numpy()
SG = block_signs(len(X), 10, ITERS)
maxmean = np.array([(X * s[:, None]).mean(axis=0).max() for s in SG]) * 100
obs = float(Mf.mean().max() * 100)
maxt = np.array([nw_t_matrix(X * s[:, None], 10).max() for s in SG])
obs_ts = np.array([H.newey_west_t(Mf[c], 10) for c in Mf.columns])
obs_t = float(obs_ts.max())
null_t = np.array([nw_t_matrix(X * s[:, None], 10) for s in SG[:1000]])
boot["front_h10_max_of_36"] = {"observed_mean": round(obs, 3), "observed_name": str(Mf.mean().idxmax()),
                               "p_max_mean": round(float((maxmean >= obs).mean()), 4),
                               "null_max_mean_median": round(float(np.median(maxmean)), 3),
                               "null_max_mean_95": round(float(np.quantile(maxmean, 0.95)), 3),
                               "observed_max_t": round(obs_t, 2), "p_max_t": round(float((maxt >= obs_t).mean()), 4),
                               "null_cells_t_ge_1.86_mean": round(float((null_t >= 1.86).sum(axis=1).mean()), 2),
                               "obs_cells_t_ge_1.86": int((obs_ts >= 1.86).sum()),
                               "method": "10-day blocks, random offset, joint sign flips across 36 combos, 4000 iters"}
# (ii) front 108셀(36 x h 5/10/20) 최대 t — 20일 블록, 달력 공통
cal = C.index
cols108 = {}
for h in HS:
    for n in names:
        cols108[(n, h)] = D[(n, h, "front")]
M108 = pd.concat(cols108, axis=1)
B = (cal.get_indexer(M108.index))
tmax_obs, tmax_null = [], []
Xs = {k: (v.to_numpy(), cal.get_indexer(v.index)) for k, v in cols108.items()}
obs_tmax = max(H.newey_west_t(v, k[1]) for k, v in cols108.items())
obs_cell = max(cols108, key=lambda k: H.newey_west_t(cols108[k], k[1]))
for i in range(1000):
    off = RNG.integers(0, 20)
    sgn = RNG.choice([-1.0, 1.0], size=len(cal) // 20 + 3)
    best_t = -9
    for (n, h), (x, pos) in Xs.items():
        xx = (x - x.mean()) * sgn[(pos + off) // 20]
        best_t = max(best_t, float(nw_t_matrix(xx[:, None], h)[0]))
    tmax_null.append(best_t)
tmax_null = np.array(tmax_null)
boot["front_108cells_max_t"] = {"observed": round(float(obs_tmax), 2), "cell": f"{obs_cell[0]} h{obs_cell[1]}",
                                "cell_front_mean": round(float(cols108[obs_cell].mean()) * 100, 3),
                                "cell_back": st(D[(obs_cell[0], obs_cell[1], "back")], obs_cell[1]),
                                "p": round(float((tmax_null >= obs_tmax).mean()), 4),
                                "null_median": round(float(np.median(tmax_null)), 2),
                                "null_95": round(float(np.quantile(tmax_null, 0.95)), 2)}
# (iii) back h10: 36 중 양 개수 >= 관측의 우연 확률, 36평균 p
Mb = pd.concat({n: D[(n, 10, "back")] for n in names}, axis=1).dropna()
Xb = Mb.to_numpy() - Mb.to_numpy().mean(axis=0)
SGb = block_signs(len(Xb), 10, ITERS)
# 귀무: 평균 0. 관측 양 개수와 같은 분포의 '평균을 0으로 옮긴' 표본에서 양 개수
npos_obs = int((Mb.mean() > 0).sum())
npos_null = np.array([((Xb * s[:, None]).mean(axis=0) > 0).sum() for s in SGb])
avg_null = np.array([(Xb * s[:, None]).mean() for s in SGb]) * 100
boot["back_h10_pos_count"] = {"observed": npos_obs, "p_ge": round(float((npos_null >= npos_obs).mean()), 4),
                              "null_median": float(np.median(npos_null)),
                              "all36_mean_obs": round(float(Mb.mean().mean() * 100), 3),
                              "all36_mean_p": round(float((avg_null >= Mb.mean().mean() * 100).mean()), 4)}
# (iv) 51개 순위: max_k min(front_k, back_k) 상위20% h10
F51 = pd.concat({k: RK[k]["front"][0] for k in scores}, axis=1)
B51 = pd.concat({k: RK[k]["back"][0] for k in scores}, axis=1)
Xf, Xb51 = (F51 - F51.mean()).fillna(0.0).to_numpy(), (B51 - B51.mean()).fillna(0.0).to_numpy()
SGf, SGb2 = block_signs(len(Xf), 10, ITERS), block_signs(len(Xb51), 10, ITERS)
mm_null = np.array([np.minimum((Xf * a[:, None]).mean(axis=0), (Xb51 * b[:, None]).mean(axis=0)).max()
                    for a, b in zip(SGf, SGb2)]) * 100
mm_obs = float(np.minimum(F51.mean(), B51.mean()).max() * 100)
mm_name = str(np.minimum(F51.mean(), B51.mean()).idxmax())
c_obs = float(min(F51["comp_S120+P1_4"].mean(), B51["comp_S120+P1_4"].mean()) * 100)
boot["rankers51_max_min_front_back_top20"] = {"best_name": mm_name, "observed": round(mm_obs, 3),
                                              "comp_S120+P1_4_min": round(c_obs, 3),
                                              "p_best": round(float((mm_null >= mm_obs).mean()), 4),
                                              "p_comp_S120+P1_4": round(float((mm_null >= c_obs).mean()), 4),
                                              "null_median": round(float(np.median(mm_null)), 3),
                                              "null_95": round(float(np.quantile(mm_null, 0.95)), 3)}
res["selection_bias"] = boot
log("boot", json.dumps(boot, ensure_ascii=False))

# ---------- 3b) 쏠림: 종목·하루·7월 ----------
log("concentration")


def all36_series(Ux, h, w, drop=(), fr=None, tie="min"):
    ss = []
    for L in LS:
        for p in PN:
            g = G[(L, p)] if (Ux is U and tie == "min") else thirds(S_all[L], P_all[p], Ux, tie=tie, seed=L * 100 + PN.index(p))
            ss.append(diff_series(g, h, w, drop, fr))
    return pd.concat(ss, axis=1).mean(axis=1)


def contrib_all36(w, h=10):
    d = H.window_dates(w, h)
    cols = wcols(w)
    fr = FR[h].loc[d, cols]
    fx = fr.sub(fr.mean(axis=1), axis=0)
    tot = pd.Series(0.0, index=cols)
    for g in G.values():
        hi = g["hi"].loc[d, cols] & fr.notna()
        lo = g["lo"].loc[d, cols] & fr.notna()
        ok = (hi.sum(axis=1) > 0) & (lo.sum(axis=1) > 0)
        a = fx.where(hi).div(hi.sum(axis=1), axis=0).fillna(0)
        b = fx.where(lo).div(lo.sum(axis=1), axis=0).fillna(0)
        tot += (a - b).loc[ok].sum(axis=0) / ok.sum()
    return (tot / len(G) * 100).sort_values(ascending=False)


conc = {}
for w in ("front", "back", "y2025"):
    base = pd.concat({n: D[(n, 10, w)] for n in names}, axis=1).mean(axis=1)
    con = contrib_all36(w)
    top = con.index[0]
    Ud = U.copy()
    Ud[top] = False
    s_top = all36_series(Ud, 10, w, (top,))
    Us = U.copy()
    Us["SNDK"] = False
    s_sndk = all36_series(Us, 10, w, ("SNDK",))
    srt = base.sort_values(ascending=False)
    conc[w] = {"base": st(base, 10), "top_contrib": {k: round(float(v), 3) for k, v in con.head(5).items()},
               "bottom_contrib": {k: round(float(v), 3) for k, v in con.tail(3).items()},
               "drop_top_stock": {"sym": top, **st(s_top, 10)},
               "drop_SNDK": {"SNDK_contrib": round(float(con.get("SNDK", 0.0)), 3), **st(s_sndk, 10)},
               "drop_top5_days": {**st(base.drop(srt.index[:5]), 10), "days": [str(x.date()) for x in srt.index[:5]],
                                  "vals": [round(float(v) * 100, 2) for v in srt.iloc[:5]]},
               "top5_days_share_of_sum": round(float(srt.iloc[:5].sum() / base.sum()), 2) if base.sum() > 0 else None}
# 7월 급락·반등 제외 (back)
bs = pd.concat({n: D[(n, 10, "back")] for n in names}, axis=1).mean(axis=1)
exitd = H.exit_date(10)
in_jul = (bs.index >= "2026-07-01") & (bs.index <= "2026-07-31")
span_hits = np.array([(t + pd.Timedelta(days=1) <= pd.Timestamp("2026-08-07")) and (exitd[t] >= pd.Timestamp("2026-07-13"))
                      for t in bs.index])
conc["back_july"] = {"drop_signal_dates_in_july": st(bs[~in_jul], 10), "only_july": st(bs[in_jul], 10),
                     "drop_hold_overlapping_0713_0807": st(bs[~span_hits], 10), "n_dropped": int(span_hits.sum())}
# 합성 S120+P1_4 back: 7월 제외, SNDK·최대 기여 종목 제외
ctop_b = RK["comp_S120+P1_4"]["back"][0]
in_jul2 = (ctop_b.index >= "2026-07-01") & (ctop_b.index <= "2026-07-31")
span2 = np.array([(exitd[t] >= pd.Timestamp("2026-07-13")) and (t <= pd.Timestamp("2026-08-07")) for t in ctop_b.index])
sc_c = scores["comp_S120+P1_4"]
conc["comp_S120+P1_4"] = {}
for w in ("front", "back"):
    s0 = RK["comp_S120+P1_4"][w][0]
    no_sndk = rank_daily(sc_c, 10, w, drop=("SNDK",))[0]
    srt = s0.sort_values(ascending=False)
    conc["comp_S120+P1_4"][w] = {"base": st(s0, 10), "drop_SNDK": st(no_sndk, 10),
                                 "drop_top5_days": st(s0.drop(srt.index[:5]), 10),
                                 "top5_days": [str(x.date()) for x in srt.index[:5]]}
conc["comp_S120+P1_4"]["back_drop_july_signals"] = st(ctop_b[~in_jul2], 10)
conc["comp_S120+P1_4"]["back_drop_hold_overlapping_0713_0807"] = st(ctop_b[~span2], 10)
# S60xP4_20 front: 3/26~4/2 쏠림
s6 = D[("S60xP4_20", 10, "front")]
srt = s6.sort_values(ascending=False)
conc["S60xP4_20_front"] = {"base": st(s6, 10), "top5_days": [str(x.date()) for x in srt.index[:5]],
                           "top5_vals": [round(float(v) * 100, 2) for v in srt.iloc[:5]],
                           "drop_top5": st(s6.drop(srt.index[:5]), 10),
                           "drop_mar_apr": st(s6[(s6.index < "2026-03-01") | (s6.index > "2026-04-30")], 10)}
res["concentration"] = conc
log("conc", json.dumps({w: (conc[w]["drop_SNDK"]["mean"], conc[w]["drop_top_stock"], conc[w]["drop_top5_days"]["mean"])
                        for w in ("front", "back", "y2025")}, ensure_ascii=False))
log("july", conc["back_july"], conc["comp_S120+P1_4"]["back_drop_july_signals"])

# ---------- 3c) 동점 규칙 바꾸기 ----------
log("tie=rand")
alt = {}
GR = {(L, p): thirds(S_all[L], P_all[p], U, tie="rand", seed=L * 100 + PN.index(p)) for L in LS for p in PN}
DR = {(f"S{L}x{p}", w): diff_series(GR[(L, p)], 10, w) for L in LS for p in PN for w in WINS}
f10r = {n: DR[(n, "front")].mean() * 100 for n in names}
b10r = {n: DR[(n, "back")].mean() * 100 for n in names}
bestr = max(names, key=lambda n: f10r[n])
alt = {"front_best": bestr, "front": round(f10r[bestr], 3), "back": round(b10r[bestr], 3),
       "back_rank": sorted(names, key=lambda n: -b10r[n]).index(bestr) + 1,
       "S60xP4_20": {w: st(DR[("S60xP4_20", w)], 10) for w in WINS},
       "all36": {w: st(pd.concat({n: DR[(n, w)] for n in names}, axis=1).mean(axis=1), 10) for w in WINS},
       "spearman_front_back": round(float(pd.Series(f10r).rank().corr(pd.Series(b10r).rank())), 3)}
res["tie_rule_equal_terciles"] = alt
log("alt", alt["front_best"], alt["front"], alt["back"], alt["back_rank"], {w: alt["all36"][w]["mean"] for w in WINS})

# ---------- 3d) 사전 베타(60일)로 뺀 단면 초과 ----------
log("beta")
beta_res = {}
for h in (10,):
    FRb = (FR[h] - BETA.mul(QF[h], axis=0)).where(E & BETA.notna())
    for w in WINS:
        A = pd.concat({f"S{L}x{p}": diff_series(G[(L, p)], h, w, fr=FRb) for L in LS for p in PN}, axis=1).mean(axis=1)
        Araw = pd.concat({n: D[(n, h, w)] for n in names}, axis=1).mean(axis=1).reindex(A.index)
        rg = REG_ABOVE.reindex(A.index).fillna(False)
        d = H.window_dates(w, h)
        bd = []
        for g in G.values():
            bd.append(BETA.loc[d].where(g["hi"].loc[d]).mean(axis=1) - BETA.loc[d].where(g["lo"].loc[d]).mean(axis=1))
        bd = pd.concat(bd, axis=1).mean(axis=1)
        beta_res[w] = {"raw_all36": st(Araw, h), "beta_adj_all36": st(A, h),
                       "raw_below": st(Araw[~rg], h), "beta_adj_below": st(A[~rg], h),
                       "raw_above": st(Araw[rg], h), "beta_adj_above": st(A[rg], h),
                       "beta60_hi_minus_lo": round(float(bd.mean()), 3),
                       "beta60_hi_minus_lo_below": round(float(bd.reindex(A.index)[~rg].mean()), 3),
                       "S60xP4_20_beta_adj": st(diff_series(G[(60, "P4_20")], h, w, fr=FRb), h)}
    # 합성 순위 상위20%도 베타 조정
    for w in ("front", "back", "y2025"):
        d = H.window_dates(w, h)
        cols = wcols(w)
        sc = scores["comp_S120+P1_4"].reindex(index=C.index, columns=C.columns)
        mask = E & sc.notna() & BETA.notna()
        pr = sc.where(mask).rank(axis=1, pct=True).loc[d, cols]
        f = FRb.where(mask).loc[d, cols]
        beta_res[w]["comp_S120+P1_4_top20_beta_adj"] = st((f.where(pr > 0.8).mean(axis=1) - f.mean(axis=1)).dropna(), h)
res["beta_adjusted_h10"] = beta_res
# 조합별 QQQ<50일선 차이 부호: front·back·2025 모두 양인 조합 수 (원값·사전 베타 조정값)
FRb10 = (FR[10] - BETA.mul(QF[10], axis=0)).where(E & BETA.notna())
pc = {"raw": {}, "beta_adj": {}}
for (L, p), g in G.items():
    n = f"S{L}x{p}"
    for kind, fr_ in (("raw", None), ("beta_adj", FRb10)):
        signs = {}
        for w in ("front", "back", "y2025", "y2024"):
            s = D[(n, 10, w)] if fr_ is None else diff_series(g, 10, w, fr=fr_)
            rg = REG_ABOVE.reindex(s.index).fillna(False)
            signs[w] = float(s[~rg].mean())
        pc[kind][n] = signs
res["below_sma50_per_combo_counts"] = {
    kind: {"front_back_2025_all_pos": sum(all(v[w] > 0 for w in ("front", "back", "y2025")) for v in d_.values()),
           "all4_pos": sum(all(v[w] > 0 for w in ("front", "back", "y2025", "y2024")) for v in d_.values()),
           **{f"{w}_pos": sum(v[w] > 0 for v in d_.values()) for w in ("front", "back", "y2025", "y2024")}}
    for kind, d_ in pc.items()}
log("per-combo below", res["below_sma50_per_combo_counts"])
log("beta", {w: (beta_res[w]["beta_adj_below"]["mean"], beta_res[w]["beta_adj_below"]["t"],
                 beta_res[w].get("comp_S120+P1_4_top20_beta_adj", {}).get("mean")) for w in WINS})

# ---------- 3e) 호가 비용: 상위20% 10일 재조정 회전율, 주간 상위10% 순수익 ----------
log("costs")
cost = {}
for nm in ("comp_S120+P1_4", "P1_4", "S120"):
    sc = scores[nm].reindex(index=C.index, columns=C.columns)
    mask = E & sc.notna()
    pr = sc.where(mask).rank(axis=1, pct=True)
    top = (pr > 0.8)
    W = top.div(top.sum(axis=1), axis=0).fillna(0)
    out = {}
    for w in ("front", "back", "y2025"):
        d = H.window_dates(w, 10)
        dW = (W - W.shift(10)).abs().sum(axis=1).loc[d]  # 10일 전 목록 대비 매매 비중 합(사고+팔고)
        gross = RK[nm][w][0] if w in RK[nm] else None
        c10 = float(dW.mean()) * 0.001 * 100  # %p per 10 days
        out[w] = {"gross_top20_h10": st(gross, 10)["mean"], "turnover_sum_abs_dw_10d": round(float(dW.mean()), 2),
                  "cost_pp_per_10d_0.1pct_oneway": round(c10, 3),
                  "net_top20_h10": round(st(gross, 10)["mean"] - c10, 3)}
    cost[nm] = out


def weekly(sc: pd.DataFrame, lo: float, start: str, end: str, oneway: float):
    sc = sc.reindex(index=C.index, columns=C.columns)
    mask = E & sc.notna()
    pr = sc.where(mask).rank(axis=1, pct=True)
    days = C.index[(C.index >= start) & (C.index <= end)]
    wk = pd.Series(days.to_period("W-SUN"), index=days)
    firsts = list(wk.groupby(wk).apply(lambda s: s.index[0]))
    eq, prev_w, eqs = 1.0, pd.Series(dtype=float), []
    for i, w0 in enumerate(firsts[:-1]):
        w1 = firsts[i + 1]
        tprev = C.index[C.index.get_loc(w0) - 1]
        row = pr.loc[tprev]
        nm = [s for s in row[(row > lo) & mask.loc[tprev]].index if np.isfinite(O.at[w0, s])]
        if not nm:
            cur = pd.Series(dtype=float)
            r = 0.0
        else:
            cur = pd.Series(1.0 / len(nm), index=nm)
            px1 = O.loc[w1, nm].fillna(C.loc[C.index[C.index.get_loc(w1) - 1], nm])
            r = float((px1 / O.loc[w0, nm]).mean() - 1)
        turn = float(cur.sub(prev_w, fill_value=0).abs().sum())
        eq *= (1 + r) * (1 - oneway * turn)
        prev_w = cur
        eqs.append(eq)
    return round((eq - 1) * 100, 2), len(firsts) - 1


for nm in ("comp_S120+P1_4", "P1_4"):
    for w, (a, b) in (("front", ("2026-01-02", "2026-05-29")), ("back", ("2026-06-01", "2026-10-09"))):
        g0_, n_ = weekly(scores[nm], 0.90, a, b, 0.0)
        g1_, _ = weekly(scores[nm], 0.90, a, b, 0.001)
        cost[nm][f"weekly_top10_{w}"] = {"gross": g0_, "net_0.1pct": g1_, "weeks": n_}
# 비교: 같은 주간 방식 보유 종목 전체(같은 비중)
ones = pd.DataFrame(1.0, index=C.index, columns=C.columns)
cost["universe_weekly"] = {w: weekly(ones, 0.0, a, b, 0.0)[0] for w, (a, b) in
                           (("front", ("2026-01-02", "2026-05-29")), ("back", ("2026-06-01", "2026-10-09")))}
# 장세 규칙 오버레이 비용: QQQ<50일선일 때 SP_hi 사기는 보유 전체 대비 h10 왕복 0.2%p
cost["note"] = ("SP_hi vs SP_lo 선택은 어차피 살 때 비용 중립. 상위20%를 10일마다 갈면 위 표 비용. "
                "QQQ<50일선 오버레이로 보유 전체 대신 SP_hi를 사면 10일 왕복 최대 0.2%p.")
res["costs"] = cost
log("costs", json.dumps(cost, ensure_ascii=False))

# ---------- 3f) 2025 부호, 2024 ----------
res["y2025_sign"] = {"all36_h10": fam["h10_y2025"]["all36"], "S60xP4_20_h10": tab["S60xP4_20"]["h10_y2025"],
                     "comp_S120+P1_4": rk_tab["comp_S120+P1_4"]["y2025"],
                     "below_sma50_all36": fam["h10_y2025"]["below_sma50"], "above_sma50_all36": fam["h10_y2025"]["above_sma50"],
                     "all36_h5": fam["h5_y2025"]["all36"], "all36_h20": fam["h20_y2025"]["all36"]}

# 장세 나눔: 2026 전체(front+back) below 일수·에피소드
res["regime_episodes"] = {w: {"below_days": fam[f"h10_{w}"]["below_sma50"]["days"],
                              "episodes": fam[f"h10_{w}"]["below_episodes"]} for w in WINS}
# 36평균 h10, 2026 월 x 장세
mt = {}
for w in ("front", "back"):
    A = pd.concat({n: D[(n, 10, w)] for n in names}, axis=1).mean(axis=1)
    rg = REG_ABOVE.reindex(A.index).fillna(False)
    df = pd.DataFrame({"v": A * 100, "reg": np.where(rg, "above", "below"), "m": A.index.strftime("%Y-%m")})
    for (m, r), x in df.groupby(["m", "reg"])["v"]:
        mt[f"{m}_{r}"] = {"mean": round(float(x.mean()), 2), "days": int(len(x))}
res["all36_h10_month_x_regime"] = mt
# 강함 안 SP_hi − 강함 전체, QQQ<50일선 (36평균)
res["below_sma50_SP_hi_minus_strong"] = {
    w: round(legs[w]["below_sma50"]["SP_hi_vs_eligible"]["mean"] - legs[w]["below_sma50"]["strong_vs_eligible"]["mean"], 3)
    for w in WINS}

# harness.weekly_portfolio로 주간 상위 10% 다시 재기(연구 수치 대조)
hw = {}
for nm in ("comp_S120+P1_4", "P1_4"):
    sc = scores[nm].reindex(index=C.index, columns=C.columns)
    mask = E & sc.notna()
    pr = sc.where(mask).rank(axis=1, pct=True)
    hw[nm] = {w: H.weekly_portfolio(pr, 0.90, a, b, mask) for w, (a, b) in
              (("front", ("2026-01-02", "2026-05-29")), ("back", ("2026-06-01", "2026-10-09")))}
    # 주간 back 종목 기여(단리 근사): 상위 3
    a, b = "2026-06-01", "2026-10-09"
    days = C.index[(C.index >= a) & (C.index <= b)]
    wk = pd.Series(days.to_period("W-SUN"), index=days)
    firsts = list(wk.groupby(wk).apply(lambda s: s.index[0]))
    contrib = pd.Series(0.0, index=C.columns)
    for i, w0 in enumerate(firsts[:-1]):
        w1 = firsts[i + 1]
        tp = C.index[C.index.get_loc(w0) - 1]
        row = pr.loc[tp]
        nmz = [s for s in row[(row > 0.9) & mask.loc[tp]].index if np.isfinite(O.at[w0, s])]
        if nmz:
            r = O.loc[w1, nmz].fillna(C.loc[C.index[C.index.get_loc(w1) - 1], nmz]) / O.loc[w0, nmz] - 1
            contrib[nmz] += r.fillna(0) / len(nmz)
    hw[nm]["back_top_contrib_pp_simple"] = {k: round(float(v) * 100, 1) for k, v in contrib.sort_values(ascending=False).head(4).items()}
hw["universe"] = {w: H.bench_weekly(a, b, "universe") for w, (a, b) in
                  (("front", ("2026-01-02", "2026-05-29")), ("back", ("2026-06-01", "2026-10-09")))}
res["weekly_harness_crosscheck"] = hw
log("weekly harness", json.dumps(hw, ensure_ascii=False))

# ---------- 4) 데스크 7종목(#774) 위 사전 지정 2개 시험 ----------
# A: 순위 키를 120일 모멘텀 → pct(120일 수익)+pct(−4일 수익) (= 연구 관찰 후보 (2) pct(S120)+pct(P1_4), QQQ 빼기는 같은 날 순위에 영향 없음)
# B: k−1일 QQQ 종가 < 50일 이평일 때만 A, 아니면 기본. 결정은 k−1 종가·k보다 앞선 PDF(deskx 규칙). 비용: 편도 0.1% x 회전.
log("deskx")
try:
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "rsi_breakout", "code")))
    import deskx  # noqa: E402

    class CompEngine(deskx.Engine):
        mode = "base"

        def ranking(self, p, k, session):
            rows = super().ranking(p, k, session)
            if self.mode == "base" or len(rows) < 3:
                return rows
            if self.mode == "regime":
                q = self.book.adj["QQQ"].to_numpy(float)
                if k - 50 < 0 or not (q[k - 1] < np.nanmean(q[k - 50:k])):
                    return rows
            r4 = []
            for r in rows:
                px = self.col("adj", r["key"])
                ok = px is not None and k - 5 >= 0 and np.isfinite(px[k - 1]) and np.isfinite(px[k - 5]) and px[k - 5] > 0
                r4.append(px[k - 1] / px[k - 5] - 1 if ok else np.nan)
            pm = pd.Series([r["mom"] for r in rows]).rank(pct=True)
            pp = (-pd.Series(r4)).rank(pct=True).fillna(0.5)
            sc = (pm + pp).to_numpy()
            order = np.argsort(-sc, kind="mergesort")
            return [rows[i] for i in order]

    def net_curve(v: pd.Series, rows: list, oneway: float) -> pd.Series:
        r = (v.shift(-1) / v - 1).dropna()
        turn = [0.0]
        for i in range(1, len(rows)):
            a, b = rows[i - 1]["w"], rows[i]["w"]
            turn.append(sum(abs(b.get(x, 0.0) - a.get(x, 0.0)) for x in set(a) | set(b)))
        turn[0] = sum(rows[0]["w"].values())
        t = pd.Series(turn, index=v.index)
        # rows[i] 비중은 i일 시가 재조정 뒤 → i일 수익(i→i+1)에서 비용을 뺀다. 재조정일(ISO 주 첫 거래일)만 센다.
        days_ = [r_["day"] for r_ in rows]
        reb = pd.Series([i == 0 or deskx.iso_week(days_[i]) != deskx.iso_week(days_[i - 1])
                         for i in range(len(rows))], index=v.index)
        cost = (t.where(reb, 0.0) * oneway).reindex(r.index).fillna(0.0)
        return r - cost, float(t.where(reb, 0.0).sum())

    eng = CompEngine()
    desk = {}
    for mode in ("base", "comp", "regime"):
        eng.mode = mode
        v, rows = eng.run()
        ev = eng.evaluate(mode, v, rows)
        rn, tot_turn = net_curve(v, rows, 0.001)
        out = {}
        for lab, a, b in deskx.PERIODS:
            x = rn[(rn.index >= a) & (rn.index <= b)]
            out[lab] = {"gross": round(ev[lab]["수익"] * 100, 1), "mdd": round(ev[lab]["낙폭"] * 100, 1),
                        "net_0.1pct": round(float((1 + x).prod() - 1) * 100, 1)}
        out["monthly_gross"] = {m: round(x * 100, 1) for m, x in ev["월별"].items()}
        out["turnover_sum"] = round(tot_turn, 2)
        out["last_hold"] = ev.get("마지막보유")
        desk[mode] = out
    desk["base_matches_deskx_default"] = True
    desk["note"] = "base = deskx.Engine 기본(=#774, check()로 desk run_model과 같음). comp/regime은 사전 지정 2개만 시험."
    res["desk_overlay"] = desk
    log("desk", json.dumps({m: {k: desk[m][k] for k in ("앞 1/2~5/29", "뒤 6/1~10/9", "9/1~10/9", "전체")}
                            for m in ("base", "comp", "regime")}, ensure_ascii=False))
except Exception as exc:  # noqa: BLE001
    res["desk_overlay"] = {"error": repr(exc)}
    log("desk error", repr(exc))

res["verifier_summary"] = {
    "verdict": "confirmed (연구 판정 '관찰만, 채택 없음' 유지)",
    "search_size_verifier": ("재계산 36조합 x h3 x 구간4 + 순위 51개(h10) + 동점 규칙 대안 36 + 사전 베타 조정 36 + "
                             "부트스트랩 4종(4000/1000회) + 데스크 오버레이 사전 지정 2개(comp, regime). 새 조합 선택 없음."),
    "points": [
        "머리 숫자 전부 재현(S60xP4_20 front +4.254 t1.86, back +0.259 t0.14 26위, 순위상관 -0.095, 36평균 0.827/0.962/-0.02/0.837, "
        "50일선 아래 2.11/2.17/1.83/1.56, 27/36, 합성 IC10·상위20%, 주간 58.56/67.78).",
        "선택편향: front 36개 최대 평균 p 0.057, 최대 t p 0.27(우연 기대 t>=1.86 셀 2.3개, 관측 2개). 108셀 최대 t 2.93 p 0.25. "
        "back 30/36 양 p 0.26. 51개 순위 min(front,back) 최대 p 0.05~0.07(이 갈래 안 보정만).",
        "사전 60일 베타로 단면 조정하면 front 36평균 +0.83 -> -0.01(전부 베타), 50일선 아래 front +0.79(t0.37)·back +0.72(t1.58)·"
        "2025 +2.00(t2.31)·2024 -0.01. 조합별 세 구간 모두 양 27 -> 18, 네 구간 9.",
        "QQQ<50일선에서 강함&눌림(SP_hi)은 back +0.90, 2025 +0.37로 약함&눌림(+3.77, +1.76)보다 못했다. '강함' 조건이 더하는 것이 없다. "
        "남는 것은 강함 안에서 고점 추격 1/3을 피하는 쪽(SP_hi-강함 전체 +1.39/+1.22/+0.88/+0.65, 원값).",
        "쏠림: 36평균 front는 SNDK 빼면 0.41, AXTI 빼면 0.24, 상위 5일 빼면 -0.19. S60xP4_20 front는 3~4월 빼면 +0.74(t0.33). "
        "back은 7월 신호를 빼도 +1.41(t1.94)로 7월 의존은 아니다.",
        "데스크 7종목(#774) 순위를 pct(120일)+pct(-4일)로 바꾸면 전체 +220% -> +129%(front 205 -> 77, back 4.9 -> 29.7, 8월 한 달 몫이 큼). "
        "QQQ<50일선일 때만 바꾸면 +196%. 둘 다 기본보다 나쁘다.",
        "호가 편도 0.1%: 합성 상위20% 10일 교체 비용 약 0.13~0.14%p/10일, 결론을 바꾸지 않는다. 동점 규칙(같은 크기 3등분)으로도 결론 같음.",
    ],
}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
log("saved", OUT)
