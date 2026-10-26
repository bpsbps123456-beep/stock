"""3단계 조합 격자(BRIEF.md '조합 격자'·'판정'). 단독 48 + 둘 조합 1,104 칸 × h 3 × 결과(R·M), T1·T2 가설 조합.

판정: 1~5월(front) Welch t 상위 20칸(최소 신호일 5) → 6~10월(back) 같은 방향 + back 순열 p < 0.05 칸 수.
선택편향: 2026 안에서 모든 신호열을 같은 거리만큼 원형 이동한 200번에 같은 절차를 반복한 귀무분포.
결과: results/grid_cells.parquet(모든 칸), results/grid.json(고른 칸·확인·귀무·가설 조합)
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

import harness as H

TH1, TH2 = 0.8, 0.7
TOPK, MIN_FRONT, N_NULL, SEED = 20, 5, 200, 11
RETAIL_EUPH = {"eq_pcr5": -1, "ai_cpcr5": -1, "lev_qqq_bull5": 1, "aaii_spread": 1, "nq_nonrept_net_oi": 1}
SMART_DEF = {"nq_lev_chg4": -1, "nq_am_chg4": -1, "dix": -1, "gvip_rel20": -1, "si_ai_chg": 1, "qai_beta60": -1}


def cells(P: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """신호 칸 행렬(날 × 칸, 1/0/NaN)과 이름. 이름: 'a+' = a 높음, 'a-' = a 낮음, 쌍은 'a+&b-'."""
    cols, names = [], []
    X = P.to_numpy()
    for j, s in enumerate(P.columns):
        x = X[:, j]
        for sign, cond in (("+", x >= TH1), ("-", x <= round(1 - TH1, 6))):  # 1−0.8 = 0.1999… 부동소수 오차 방지
            cols.append(np.where(np.isnan(x), np.nan, cond.astype(float)))
            names.append(f"{s}{sign}")
    for a, b in combinations(range(len(P.columns)), 2):
        xa, xb = X[:, a], X[:, b]
        na = np.isnan(xa) | np.isnan(xb)
        for sa, ca in (("+", xa >= TH2), ("-", xa <= round(1 - TH2, 6))):
            for sb, cb in (("+", xb >= TH2), ("-", xb <= round(1 - TH2, 6))):
                cols.append(np.where(na, np.nan, (ca & cb).astype(float)))
                names.append(f"{P.columns[a]}{sa}&{P.columns[b]}{sb}")
    return np.column_stack(cols), names


def composite(P: pd.DataFrame, spec: dict) -> pd.Series:
    parts = pd.DataFrame({s: (P[s] if d > 0 else 1 - P[s]) for s, d in spec.items()})
    return parts.mean(axis=1).where(parts.notna().sum(axis=1) >= (len(spec) + 1) // 2 + 1)


def table(S: np.ndarray, names: list[str], Y: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
    rows = []
    for h in H.HORIZONS:
        for y in ("R", "M"):
            yy = Y[f"{y}{h}"].to_numpy()
            rec = {"cell": names, "h": h, "y": y}
            for per in ("front", "back", "y2025"):
                d, t, n = H.grid_diffs(S, yy, H.period_mask(idx, per))
                rec.update({f"{per}_d": d, f"{per}_t": t, f"{per}_n": n})
            rows.append(pd.DataFrame(rec))
    return pd.concat(rows, ignore_index=True)


def select_confirm(S26: np.ndarray, Y26: pd.DataFrame, fr: np.ndarray, bk: np.ndarray, names: list[str], detail: bool = False):
    """front에서 고점 쪽(R·M 음수 t)·바닥 쪽(R 양수 t) 상위 20칸 → back 확인 수. detail이면 칸별 기록도 돌려준다."""
    recs = []
    for h in H.HORIZONS:
        for y in ("R", "M"):
            yy = Y26[f"{y}{h}"].to_numpy()
            d, t, n = H.grid_diffs(S26, yy, fr)
            recs.append(pd.DataFrame({"k": np.arange(len(names)), "h": h, "y": y, "t": t, "n": n}))
    g = pd.concat(recs, ignore_index=True)
    g = g[(g["n"] >= MIN_FRONT) & g["t"].notna()]
    out = {}
    for side in ("top", "bottom"):
        sub = g if side == "top" else g[g["y"] == "R"]
        pick = sub.nsmallest(TOPK, "t") if side == "top" else sub.nlargest(TOPK, "t")
        conf, rows = 0, []
        for _, r in pick.iterrows():
            yy = Y26[f"{r['y']}{int(r['h'])}"].to_numpy()
            sig = S26[:, int(r["k"])]
            bs, by = np.where(bk, sig, np.nan), np.where(bk, yy, np.nan)
            d, _, nb = H.welch(bs, by)
            want = (d < 0) if side == "top" else (d > 0)
            p = H.shift_p(bs, by, "less" if side == "top" else "greater") if (nb >= 3 and want) else np.nan
            ok = bool(want and p < 0.05)
            conf += ok
            if detail:
                fd, ft, fn = H.welch(np.where(fr, sig, np.nan), np.where(fr, yy, np.nan))
                rows.append({"cell": names[int(r["k"])], "h": int(r["h"]), "y": r["y"], "front_d": fd, "front_t": ft, "front_n": fn,
                             "back_d": d, "back_n": nb, "back_p": p, "confirmed": ok})
        out[side] = (conf, rows) if detail else conf
    return out


def main() -> None:
    P = H.panel()
    idx = P.index
    Y = H.outcomes("QQQ").reindex(idx)
    S, names = cells(P)
    # 가설 조합 T1·T2
    re, sd = composite(P, RETAIL_EUPH), composite(P, SMART_DEF)
    t1 = ((re >= TH2) & (sd >= TH2)).astype(float).where(re.notna() & sd.notna())
    t2 = ((re <= 1 - TH2) & (sd <= 1 - TH2)).astype(float).where(re.notna() & sd.notna())
    S = np.column_stack([S, t1.to_numpy(), t2.to_numpy()])
    names = names + ["T1_개미환호&큰손방어", "T2_개미공포&큰손매수"]
    tab = table(S, names, Y, idx)
    tab.to_parquet(H.RES / "grid_cells.parquet")
    # 고르기·확인(2026)
    y26 = np.asarray(idx >= "2026-01-02")
    S26, Y26, i26 = S[y26], Y[y26].reset_index(drop=True), idx[y26]
    fr, bk = H.period_mask(i26, "front"), H.period_mask(i26, "back")
    obs = select_confirm(S26, Y26, fr, bk, names, detail=True)
    rng = np.random.default_rng(SEED)
    null = {"top": [], "bottom": []}
    for _ in range(N_NULL):
        k = int(rng.integers(H.MIN_SHIFT, len(i26) - H.MIN_SHIFT))
        r = select_confirm(np.roll(S26, k, axis=0), Y26, fr, bk, names)
        null["top"].append(r["top"])
        null["bottom"].append(r["bottom"])
    res = {"n_cells": len(names), "picks": {}, "null": {}}
    for side in ("top", "bottom"):
        c, rows = obs[side]
        nn = np.array(null[side])
        res["picks"][side] = rows
        res["null"][side] = {"confirmed": c, "null_mean": float(nn.mean()), "null_p95": float(np.percentile(nn, 95)),
                             "p": float(((nn >= c).sum() + 1) / (len(nn) + 1))}
    # T1·T2 판정과 2025, SOXX 보조
    Ys = H.outcomes("SOXX").reindex(idx)
    hyp = {}
    for nm, col, side in (("T1", t1, "top"), ("T2", t2, "bottom")):
        sig = col.to_numpy()
        rows = []
        for h in H.HORIZONS:
            for y in (("R", "M") if side == "top" else ("R",)):
                yy = Y[f"{y}{h}"].to_numpy()
                row = {"h": h, "y": y}
                for per in ("front", "back", "y2025"):
                    m = H.period_mask(idx, per)
                    d, t, n = H.welch(np.where(m, sig, np.nan), np.where(m, yy, np.nan))
                    row.update({f"{per}_d": d, f"{per}_n": n,
                                f"{per}_p": H.shift_p(np.where(m, sig, np.nan), np.where(m, yy, np.nan), "less" if side == "top" else "greater")})
                ys = Ys[f"{y}{h}"].to_numpy()
                for per in ("front", "back", "y2025"):
                    m = H.period_mask(idx, per)
                    row[f"soxx_{per}_d"] = H.welch(np.where(m, sig, np.nan), np.where(m, ys, np.nan))[0]
                good = (lambda d: d < 0) if side == "top" else (lambda d: d > 0)
                row["pass"] = bool(good(row["front_d"]) and good(row["back_d"]) and row["back_p"] < 0.05) if not np.isnan(row["back_d"]) else False
                rows.append(row)
        hyp[nm] = rows
    res["hypotheses"] = hyp
    H.save("grid", res)
    for side in ("top", "bottom"):
        print(side, res["null"][side])
        for r in res["picks"][side][:TOPK]:
            print(f"  {r['cell']:45s} h{r['h']:2d} {r['y']} front {r['front_d']:+.2f}(t{r['front_t']:+.1f},n{r['front_n']}) "
                  f"back {r['back_d']:+.2f}(n{r['back_n']}, p{r['back_p'] if r['back_p'] == r['back_p'] else float('nan'):.2f}) {'확인' if r['confirmed'] else ''}")
    for nm, rows in hyp.items():
        for r in rows:
            print(nm, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
