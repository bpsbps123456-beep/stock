"""E1 셋 조합 격자(BRIEF.md '격자 결과를 본 뒤 추가하는 갈래'). 24C3 = 2,024 × 방향 8 = 16,192칸.

칸: 세 신호 모두 켜진 날 = 1. 위(+)는 백분위 ≥ 0.65, 아래(−)는 ≤ 0.35. 셋 중 하나라도 NaN이면 NaN.
판정·귀무: grid.select_confirm 그대로(front Welch t 상위 20, 최소 신호일 5 → back 같은 방향 + shift_p < 0.05).
귀무: 2026 안에서 모든 칸을 같은 거리만큼 원형 이동 200번(seed 13).
추가 보고: 고른 칸이 실제로 몇 개의 서로 다른 신호일 묶음인지, SOXX·2025 값, 묶음 구성(옵션·개인 vs 헤지펀드·큰손).
결과: results/triple_combos.json, results/triple_combos_cells.parquet(모든 칸 × h × R/M × 기간),
      results/triple_combos_S.npy(int8 칸 행렬, −1 = NaN, 행 = harness.panel() 색인, 열 = json의 names 순서)
"""
from __future__ import annotations

import time
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import grid as G
import harness as H

TH_UP, TH_DN = 0.65, 0.35
N_NULL, SEED = 200, 13
OPT = "옵션·개인"


def cells3(P: pd.DataFrame):
    """날 × 칸 행렬(1/0/NaN), 이름('a+&b-&c+'), 칸별 (구성원 3개, 부호 3개). 순서: 조합 우선, 방향 '+++' … '---'."""
    X = P.to_numpy()
    U, D, N = X >= TH_UP, X <= TH_DN, np.isnan(X)
    cmb = np.array(list(combinations(range(X.shape[1]), 3)))
    A, B, C = cmb.T
    na = N[:, A] | N[:, B] | N[:, C]
    pats = [(sa, sb, sc) for sa in "+-" for sb in "+-" for sc in "+-"]
    cols = []
    for sa, sb, sc in pats:
        on = (U if sa == "+" else D)[:, A] & (U if sb == "+" else D)[:, B] & (U if sc == "+" else D)[:, C]
        cols.append(np.where(na, np.nan, on.astype(float)))
    S = np.stack(cols, axis=2).reshape(len(P), -1)
    sig = list(P.columns)
    names, meta = [], []
    for a, b, c in cmb:
        for p in pats:
            names.append(f"{sig[a]}{p[0]}&{sig[b]}{p[1]}&{sig[c]}{p[2]}")
            meta.append(((sig[a], sig[b], sig[c]), p))
    return S, names, meta


def n_opt(members) -> int:
    return sum(H.GROUP[m] == OPT for m in members)


def runs(pos: np.ndarray, gap: int = 5) -> int:
    """신호일(거래일 위치) 묶음 수: 이웃 신호일 간격이 gap 거래일 넘으면 새 묶음."""
    if len(pos) == 0:
        return 0
    return int(1 + (np.diff(pos) > gap).sum())


def stat(sig, y, mask, side):
    s, v = np.where(mask, sig, np.nan), np.where(mask, y, np.nan)
    d, t, n = H.welch(s, v)
    p = H.shift_p(s, v, side) if n >= 3 else np.nan
    return d, t, n, p


def main() -> None:
    t0 = time.time()
    P = H.panel()
    idx = P.index
    Y = H.outcomes("QQQ").reindex(idx)
    Ys = H.outcomes("SOXX").reindex(idx)
    S, names, meta = cells3(P)
    k_of = {nm: k for k, nm in enumerate(names)}
    print("cells", S.shape, f"{time.time() - t0:.1f}s")
    np.save(H.RES / "triple_combos_S.npy", np.where(np.isnan(S), -1, S).astype(np.int8))

    # 모든 칸 표(2025~, QQQ)
    sel = np.asarray(idx >= "2025-01-02")
    tab = G.table(S[sel], names, Y[sel], idx[sel])
    tab["n_opt"] = np.tile([n_opt(m) for m, _ in meta], len(tab) // len(names))
    tab.to_parquet(H.RES / "triple_combos_cells.parquet")
    print("table", tab.shape, f"{time.time() - t0:.1f}s")

    # 고르기·확인(2026)
    y26 = np.asarray(idx >= "2026-01-02")
    S26, Y26, i26 = S[y26], Y[y26].reset_index(drop=True), idx[y26]
    fr, bk = H.period_mask(i26, "front"), H.period_mask(i26, "back")
    obs = G.select_confirm(S26, Y26, fr, bk, names, detail=True)

    def summarize(Sm, rows):
        """고른 칸들의 서로 다른 front 신호일 묶음 수, 앞쪽 t."""
        fsets = {tuple(np.flatnonzero(fr & (Sm[:, k_of[r["cell"]]] == 1))) for r in rows}
        ts = [abs(r["front_t"]) for r in rows]
        return len(fsets), max(ts), float(np.mean(ts))

    rng = np.random.default_rng(SEED)
    null = {s: {"conf": [], "n_sets": [], "best_t": [], "mean_t": [], "conf_sets": []} for s in ("top", "bottom")}
    shifts = []
    for i in range(N_NULL):
        k = int(rng.integers(H.MIN_SHIFT, len(i26) - H.MIN_SHIFT))
        shifts.append(k)
        Sr = np.roll(S26, k, axis=0)
        r = G.select_confirm(Sr, Y26, fr, bk, names, detail=True)
        for side in ("top", "bottom"):
            c, rows = r[side]
            ns, bt, mt = summarize(Sr, rows)
            cs = len({tuple(np.flatnonzero(fr & (Sr[:, k_of[x["cell"]]] == 1))) for x in rows if x["confirmed"]})
            for key, v in (("conf", c), ("n_sets", ns), ("best_t", bt), ("mean_t", mt), ("conf_sets", cs)):
                null[side][key].append(v)
        if (i + 1) % 50 == 0:
            print("null", i + 1, f"{time.time() - t0:.1f}s")

    res = {"n_cells": len(names), "thresholds": [TH_UP, TH_DN], "n_null": N_NULL, "seed": SEED,
           "eligible_front_cells": {}, "picks": {}, "null": {}, "sets": {}, "composition": {}, "member_freq": {}}
    # front에서 최소 신호일 5 이상인 칸 수(h·결과별)
    for h in H.HORIZONS:
        for y in ("R", "M"):
            sub = tab[(tab["h"] == h) & (tab["y"] == y)]
            res["eligible_front_cells"][f"{y}{h}"] = int(((sub["front_n"] >= G.MIN_FRONT) & sub["front_t"].notna()).sum())
    # 구성 기준선(전체 칸)
    base = Counter(n_opt(m) for m, _ in meta)
    res["composition"]["universe"] = {f"opt{j}": base[j] for j in range(4)}

    for side in ("top", "bottom"):
        c, rows = obs[side]
        want = "less" if side == "top" else "greater"
        fset_id, full_id = {}, {}
        for r in rows:
            k = k_of[r["cell"]]
            sig = S[:, k]
            members, signs = meta[k]
            r["members"], r["signs"] = list(members), "".join(signs)
            r["n_opt"] = n_opt(members)
            r["groups"] = [H.GROUP[m] for m in members]
            s26 = S26[:, k]
            fpos, bpos = np.flatnonzero(fr & (s26 == 1)), np.flatnonzero(bk & (s26 == 1))
            fkey, allkey = tuple(fpos), tuple(np.flatnonzero(s26 == 1))
            r["front_set"] = fset_id.setdefault(fkey, len(fset_id) + 1)
            r["full2026_set"] = full_id.setdefault(allkey, len(full_id) + 1)
            r["front_days"] = [str(i26[j])[:10] for j in fpos]
            r["back_days"] = [str(i26[j])[:10] for j in bpos]
            r["front_runs"], r["back_runs"] = runs(fpos), runs(bpos)
            yq, ys = Y[f"{r['y']}{r['h']}"].to_numpy(), Ys[f"{r['y']}{r['h']}"].to_numpy()
            m25 = H.period_mask(idx, "y2025")
            d, t, n, p = stat(sig, yq, m25, want)
            r.update({"y2025_d": d, "y2025_t": t, "y2025_n": n, "y2025_p": p})
            for per in ("front", "back", "y2025"):
                d, t, n, p = stat(sig, ys, H.period_mask(idx, per), want)
                r.update({f"soxx_{per}_d": d, f"soxx_{per}_t": t, f"soxx_{per}_n": n, f"soxx_{per}_p": p})
        res["picks"][side] = rows
        nn = {key: np.array(v, dtype=float) for key, v in null[side].items()}
        ns, bt, mt = summarize(S26, rows)
        conf_sets = len({tuple(r["front_days"]) for r in rows if r["confirmed"]})
        res["null"][side] = {
            "confirmed": c, "null_mean": float(nn["conf"].mean()), "null_median": float(np.median(nn["conf"])),
            "null_p95": float(np.percentile(nn["conf"], 95)), "p": float(((nn["conf"] >= c).sum() + 1) / (N_NULL + 1)),
            "null_zero_share": float((nn["conf"] == 0).mean()),
            "confirmed_front_sets": conf_sets, "null_conf_sets_mean": float(nn["conf_sets"].mean()),
            "null_conf_sets_p95": float(np.percentile(nn["conf_sets"], 95)),
            "p_conf_sets": float(((nn["conf_sets"] >= conf_sets).sum() + 1) / (N_NULL + 1)),
            "best_front_abs_t": bt, "null_best_t_mean": float(nn["best_t"].mean()),
            "best_t_pct_in_null": float((nn["best_t"] < bt).mean()),
            "mean_front_abs_t": mt, "null_mean_t_mean": float(nn["mean_t"].mean()),
            "mean_t_pct_in_null": float((nn["mean_t"] < mt).mean()),
            "n_front_sets": ns, "null_n_sets_mean": float(nn["n_sets"].mean()),
            "null_conf_hist": {int(k): int(v) for k, v in sorted(Counter(nn["conf"].astype(int)).items())}}
        res["sets"][side] = {"distinct_front_sets": len(fset_id), "distinct_full2026_sets": len(full_id),
                             "distinct_cells": len({r["cell"] for r in rows}),
                             "distinct_cell_h_y": len({(r["cell"], r["h"], r["y"]) for r in rows}),
                             "distinct_front_set_h_y": len({(r["front_set"], r["h"], r["y"]) for r in rows})}
        res["composition"][side] = {
            "picks": {f"opt{j}": sum(r["n_opt"] == j for r in rows) for j in range(4)},
            "confirmed": {f"opt{j}": sum(r["n_opt"] == j and r["confirmed"] for r in rows) for j in range(4)}}
        res["member_freq"][side] = Counter(f"{m}{s}" for r in rows for m, s in zip(r["members"], r["signs"])).most_common(10)

    # 설명용: 모든 칸에서 front t와 back t의 순위 상관(판정 아님)
    desc = {}
    for h in H.HORIZONS:
        for y in ("R", "M"):
            sub = tab[(tab["h"] == h) & (tab["y"] == y) & (tab["front_n"] >= G.MIN_FRONT) & (tab["back_n"] >= G.MIN_FRONT)]
            sub = sub.dropna(subset=["front_t", "back_t"])
            rho = spearmanr(sub["front_t"], sub["back_t"]).statistic if len(sub) > 10 else np.nan
            desc[f"{y}{h}"] = {"n_cells": int(len(sub)), "spearman_front_back_t": float(rho),
                               "sign_agree": float((np.sign(sub["front_t"]) == np.sign(sub["back_t"])).mean())}
    res["front_back_rank_corr"] = desc
    res["null_shifts"] = shifts
    res["names_order"] = "조합(itertools.combinations(harness.SIGNALS, 3)) 우선, 방향 +++,++-,+-+,+--,-++,-+-,--+,--- 순"
    H.save("triple_combos", res)
    print(f"done {time.time() - t0:.1f}s")

    for side in ("top", "bottom"):
        print(side, {k: v for k, v in res["null"][side].items()})
        print("  sets", res["sets"][side], "comp", res["composition"][side])
        print("  freq", res["member_freq"][side])
        for r in res["picks"][side]:
            bp = r["back_p"] if r["back_p"] == r["back_p"] else float("nan")
            print(f"  {r['cell']:52s} h{r['h']:2d} {r['y']} opt{r['n_opt']} F{r['front_set']:2d} "
                  f"front {r['front_d']:+.2f}(t{r['front_t']:+.1f},n{r['front_n']},runs{r['front_runs']}) "
                  f"back {r['back_d']:+.2f}(n{r['back_n']},runs{r['back_runs']},p{bp:.3f}) "
                  f"25 {r['y2025_d']:+.2f}(n{r['y2025_n']},p{r['y2025_p']:.2f}) "
                  f"SOXX f{r['soxx_front_d']:+.2f} b{r['soxx_back_d']:+.2f}(p{r['soxx_back_p']:.2f}) 25 {r['soxx_y2025_d']:+.2f} "
                  f"{'확인' if r['confirmed'] else ''}")
    print("rank corr", desc)


def followup() -> None:
    """main() 결과(json·npy)를 읽어 확인된 칸의 실제 날짜·2025·SOXX 집계, 둘 조합 확인 칸과 겹침,
    front·back t 순위 상관의 귀무(같은 이동 200번)를 덧붙인다. `py triple_combos.py --report`."""
    import json
    res = json.loads((H.RES / "triple_combos.json").read_text(encoding="utf-8"))
    P = H.panel()
    idx = P.index
    Y = H.outcomes("QQQ").reindex(idx)
    S8 = np.load(H.RES / "triple_combos_S.npy")
    S = np.where(S8 < 0, np.nan, S8.astype(float))
    _, names, meta = cells3(P)
    k_of = {nm: k for k, nm in enumerate(names)}
    y26 = np.asarray(idx >= "2026-01-02")
    S26, Y26, i26 = S[y26], Y[y26].reset_index(drop=True), idx[y26]
    fr, bk = H.period_mask(i26, "front"), H.period_mask(i26, "back")
    fu = {}
    for side in ("top", "bottom"):
        rows = res["picks"][side]
        conf = [r for r in rows if r["confirmed"]]
        sets = {}
        for r in rows:
            sets.setdefault(r["front_set"], {"front_days": r["front_days"], "cells": [], "confirmed": 0})
            sets[r["front_set"]]["cells"].append(f"{r['cell']} {r['y']}{r['h']}")
            sets[r["front_set"]]["confirmed"] += r["confirmed"]
        good = (lambda d: d < 0) if side == "top" else (lambda d: d > 0)
        ok25 = [r for r in conf if r["y2025_n"] >= 3 and r["y2025_d"] == r["y2025_d"]]
        okall25 = [r for r in rows if r["y2025_n"] >= 3 and r["y2025_d"] == r["y2025_d"]]
        bdays = sorted({d for r in conf for d in r["back_days"]})
        fu[side] = {
            "front_sets": sets,
            "confirmed_back_union_days": bdays,
            "confirmed_back_union_runs": runs(np.flatnonzero(np.isin(np.asarray(i26).astype(str), bdays))),
            "confirmed_distinct_back_sets": len({tuple(r["back_days"]) for r in conf}),
            "y2025_confirmed": {"with_3plus_days": len(ok25), "same_dir": sum(good(r["y2025_d"]) for r in ok25),
                                "p_lt_05": sum(good(r["y2025_d"]) and r["y2025_p"] < 0.05 for r in ok25)},
            "y2025_all_picks": {"with_3plus_days": len(okall25), "same_dir": sum(good(r["y2025_d"]) for r in okall25),
                                "p_lt_05": sum(good(r["y2025_d"]) and r["y2025_p"] < 0.05 for r in okall25)},
            "soxx_confirmed": {"front_same_dir": sum(good(r["soxx_front_d"]) for r in conf),
                               "back_same_dir": sum(good(r["soxx_back_d"]) for r in conf if r["soxx_back_d"] == r["soxx_back_d"]),
                               "back_p_lt_05": sum(good(r["soxx_back_d"]) and r["soxx_back_p"] < 0.05 for r in conf
                                                   if r["soxx_back_d"] == r["soxx_back_d"])},
            "soxx_all_picks_back": {"measurable": sum(r["soxx_back_d"] == r["soxx_back_d"] for r in rows),
                                    "same_dir": sum(good(r["soxx_back_d"]) for r in rows if r["soxx_back_d"] == r["soxx_back_d"])},
            "qqq_all_picks_back": {"measurable_n3": sum(r["back_n"] >= 3 and r["back_d"] == r["back_d"] for r in rows),
                                   "same_dir": sum(good(r["back_d"]) for r in rows if r["back_n"] >= 3 and r["back_d"] == r["back_d"])},
        }
    # 둘 조합 격자(문턱 0.7)에서 확인된 3칸과 셋 조합 확인 칸의 날짜 겹침
    X = P.to_numpy()
    col = {s: j for j, s in enumerate(P.columns)}

    def pair_days(a, sa, b, sb, th=0.7):
        xa, xb = X[y26, col[a]], X[y26, col[b]]
        ca = xa >= th if sa == "+" else xa <= 1 - th
        cb = xb >= th if sb == "+" else xb <= 1 - th
        on = ca & cb
        return set(np.asarray(i26)[fr & on].astype(str)), set(np.asarray(i26)[bk & on].astype(str))
    pairs = {"eq_pcr5+&shvr_ai5-": ("eq_pcr5", "+", "shvr_ai5", "-"), "skew+&shvr_ai5-": ("skew", "+", "shvr_ai5", "-"),
             "skew+&nq_am_chg4-": ("skew", "+", "nq_am_chg4", "-")}
    ov = {}
    for pn, spec in pairs.items():
        pf, pb = pair_days(*spec)
        ov[pn] = {"pair_front_n": len(pf), "pair_back_n": len(pb), "vs_confirmed_triples": []}
        for r in res["picks"]["top"]:
            if r["confirmed"]:
                tf, tb = set(r["front_days"]), set(r["back_days"])
                jf = len(pf & tf) / len(pf | tf) if pf | tf else np.nan
                jb = len(pb & tb) / len(pb | tb) if pb | tb else np.nan
                ov[pn]["vs_confirmed_triples"].append({"cell": r["cell"], "front_jaccard": jf, "back_jaccard": jb})
    fu["pair_overlap"] = ov
    # front t·back t 순위 상관(모든 칸) — 관측과 같은 이동 200번 귀무
    rc = {}
    for h in H.HORIZONS:
        for y in ("R", "M"):
            yy = Y26[f"{y}{h}"].to_numpy()

            def rho(Sm):
                _, tf, nf = H.grid_diffs(Sm, yy, fr)
                _, tb, nb = H.grid_diffs(Sm, yy, bk)
                ok = (nf >= G.MIN_FRONT) & (nb >= G.MIN_FRONT) & ~np.isnan(tf) & ~np.isnan(tb)
                return spearmanr(tf[ok], tb[ok]).statistic if ok.sum() > 10 else np.nan, int(ok.sum())
            obs, n = rho(S26)
            nul = np.array([rho(np.roll(S26, k, axis=0))[0] for k in res["null_shifts"]])
            nul = nul[~np.isnan(nul)]
            rc[f"{y}{h}"] = {"rho": float(obs), "n_cells": n, "null_mean": float(nul.mean()), "null_p05": float(np.percentile(nul, 5)),
                             "null_p95": float(np.percentile(nul, 95)), "p_upper": float(((nul >= obs).sum() + 1) / (len(nul) + 1)),
                             "n_null": int(len(nul))}
    fu["front_back_rank_corr_null"] = rc
    res["followup"] = fu
    H.save("triple_combos", res)
    print(json.dumps(fu, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    import sys
    followup() if "--report" in sys.argv else main()
