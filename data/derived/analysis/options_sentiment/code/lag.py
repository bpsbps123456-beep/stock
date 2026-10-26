"""2단계 시차 실험(BRIEF.md '2단계', 2026-10-14). 사용자: '살짝 비틀어서 — 모두가 환호할 때 고점. 콜과 풋 사이
시간 차이를 1일·2일·3일·4일·열흘·한 달까지 다 실험'.

A. 환호(콜 쏠림) 신호 뒤 L거래일 QQQ 수익 R_L·최대 낙폭 M_L. 지표 3개 × 측정 창 W 6개 × 시차 L 8개.
B. 콜 쏠림 뒤 k일 안에 풋 쪽으로 돌아선 첫날. 지표 2개 × W 2개 × k 8개.
진입 기준은 t+1 시가(지표는 t일 마감 뒤 값). 결과: results/lag_A.csv, lag_B.csv, lag.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402
import fetch  # noqa: E402

RES = HERE.parent / "results"
WS = [1, 2, 3, 5, 10, 21]
LS = [1, 2, 3, 4, 5, 10, 15, 21]
KS = [1, 2, 3, 4, 5, 10, 15, 21]
TH, TURN = 0.8, 0.5
MIN_SHIFT, N_NULL = 21, 200
PERIODS = {"front": ("2026-01-02", "2026-05-29"), "back": ("2026-06-01", "2026-12-31"), "y2025": ("2025-01-02", "2025-12-31")}


def ohlc() -> pd.DataFrame:
    f = fetch.CACHE / "yahoo_ohlc.parquet"
    if not f.exists():
        import yfinance as yf
        cols = {}
        for t in ("QQQ", "SOXX"):
            h = yf.Ticker(t).history(start="2024-01-02", auto_adjust=False)
            h.index = h.index.strftime("%Y-%m-%d")
            cols[f"{t}_o"], cols[f"{t}_c"] = h["Open"], h["Close"]
        pd.DataFrame(cols).to_parquet(f)
    return pd.read_parquet(f)


def outcomes(px: pd.DataFrame, days: pd.Index, a: str, base: str = "open") -> dict[str, pd.Series]:
    """R_L·M_L을 분석일(days) 색인으로. base='open'이면 t+1 시가, 'close'면 t+1 종가 기준."""
    o, c = px[f"{a}_o"].to_numpy(), px[f"{a}_c"].to_numpy()
    pos = {d: i for i, d in enumerate(px.index)}
    out = {}
    for L in LS:
        r, m = np.full(len(days), np.nan), np.full(len(days), np.nan)
        for j, d in enumerate(days):
            i = pos.get(d)
            if i is None:
                continue
            if base == "open":
                if i + L >= len(c):
                    continue
                path = np.concatenate([[o[i + 1]], c[i + 1:i + L + 1]])
            else:
                if i + 1 + L >= len(c):
                    continue
                path = c[i + 1:i + L + 2]
            r[j] = (path[-1] / path[0] - 1) * 100
            m[j] = (path / np.maximum.accumulate(path) - 1).min() * 100
        out[f"R{L}"], out[f"M{L}"] = pd.Series(r, index=days), pd.Series(m, index=days)
    return out


def indicators(days: pd.Index) -> dict[str, pd.Series]:
    """방향 맞춘 백분위(1 = 가장 환호). 키: 'EQ_W', 'AI_W', 'AIB_W'."""
    cb = fetch.load_cboe().reindex(days)
    syms = build.ai_us_symbols()
    occ = fetch.load_occ(syms)
    cc = occ.pivot_table(index="date", columns="underlying", values="C_C", aggfunc="sum").reindex(days)
    cp = occ.pivot_table(index="date", columns="underlying", values="C_P", aggfunc="sum").reindex(days)
    have = cc.notna().any(axis=1) | cp.notna().any(axis=1)  # OCC 자료가 있는 날(2024-10-14~)
    cc, cp = cc.fillna(0).where(have, axis=0), cp.fillna(0).where(have, axis=0)
    out = {}
    for W in WS:
        eq = cb["equity_pv"].rolling(W).sum() / cb["equity_cv"].rolling(W).sum()
        out[f"EQ_{W}"] = build.rolling_pct(-eq)
        ai = cp.sum(axis=1, min_count=1).rolling(W).sum() / cc.sum(axis=1, min_count=1).rolling(W).sum()
        out[f"AI_{W}"] = build.rolling_pct(-ai)
        sc, sp = cc.rolling(W).sum(), cp.rolling(W).sum()
        ok = (sc + sp) / W >= 1000
        pcr = (sp / sc.where(sc > 0)).where(ok)
        frenzy = pd.DataFrame({s: build.rolling_pct(-pcr[s]) for s in pcr.columns}) >= TH
        n_ok = ok.sum(axis=1)
        breadth = (frenzy & ok).sum(axis=1) / n_ok.where(n_ok >= 20)
        out[f"AIB_{W}"] = build.rolling_pct(breadth)
    return out


def _period(idx: pd.Index, name: str) -> np.ndarray:
    lo, hi = PERIODS[name]
    return np.asarray((idx >= lo) & (idx <= hi))


def diff_t(sig: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    m = ~np.isnan(sig) & ~np.isnan(y)
    s, v = sig[m] > 0, y[m]
    if s.sum() < 3 or (~s).sum() < 3:
        return np.nan, np.nan, int(s.sum())
    a, b = v[s], v[~s]
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, (d / se if se > 0 else np.nan), int(s.sum())


def shift_p(sig: np.ndarray, y: np.ndarray) -> float:
    m = ~np.isnan(sig) & ~np.isnan(y)
    s, v = sig[m] > 0, y[m]
    n = len(s)
    if s.sum() < 3 or n < 3 * MIN_SHIFT:
        return np.nan
    obs = v[s].mean() - v[~s].mean()
    null = [v[r].mean() - v[~r].mean() for k in range(MIN_SHIFT, n - MIN_SHIFT + 1)
            for r in [np.roll(s, k)] if 0 < r.sum() < n]
    return float((np.sum(np.array(null) <= obs) + 1) / (len(null) + 1))


def grid_A(ind: dict, out: dict, idx: pd.Index) -> pd.DataFrame:
    rows = []
    for key, p in ind.items():
        sig = (p >= TH).astype(float).where(p.notna()).to_numpy()
        for L in LS:
            for y in ("R", "M"):
                yy = out[f"{y}{L}"].to_numpy()
                row = {"ind": key.split("_")[0], "W": int(key.split("_")[1]), "L": L, "y": y}
                for per in PERIODS:
                    msk = _period(idx, per)
                    d, t, n = diff_t(np.where(msk, sig, np.nan), np.where(msk, yy, np.nan))
                    row.update({f"{per}_d": d, f"{per}_t": t, f"{per}_n": n})
                rows.append(row)
    return pd.DataFrame(rows)


def events_B(p: pd.Series, k: int) -> pd.Series:
    v = p.to_numpy()
    fr = np.nan_to_num(v >= TH)
    tn = (v <= TURN) & ~np.isnan(v)
    ev = np.zeros(len(v))
    for t in range(1, len(v)):
        if tn[t] and not tn[t - 1] and fr[max(0, t - k):t].any():
            ev[t] = 1
    return pd.Series(np.where(np.isnan(v), np.nan, ev), index=p.index)


def grid_B(ind: dict, out: dict, idx: pd.Index) -> tuple[pd.DataFrame, dict]:
    rows, evs = [], {}
    for name in ("EQ", "AI"):
        for W in (1, 5):
            p = ind[f"{name}_{W}"]
            for k in KS:
                e = events_B(p, k)
                evs[f"{name}_{W}_{k}"] = e
                for y in ("R21", "M21"):
                    yy = out[y].to_numpy()
                    row = {"ind": name, "W": W, "k": k, "y": y[0]}
                    for per in PERIODS:
                        msk = _period(idx, per)
                        d, t, n = diff_t(np.where(msk, e.to_numpy(), np.nan), np.where(msk, yy, np.nan))
                        hit = yy[msk & (e.to_numpy() > 0) & ~np.isnan(yy)]
                        row.update({f"{per}_d": d, f"{per}_t": t, f"{per}_n": n,
                                    f"{per}_dd5": int((hit <= -5).sum()) if y == "M21" else None})
                    rows.append(row)
    return pd.DataFrame(rows), evs


def pick_and_confirm(g: pd.DataFrame, sigs: dict, out: dict, idx: pd.Index, keycols: list[str], ycol_of) -> dict:
    """front에서 가설 방향 t가 가장 큰(가장 음수) 칸 → back 차이·순위·순열 p, 2025."""
    res = {}
    for y in ("R", "M"):
        sub = g[(g["y"] == y) & g["front_t"].notna()]
        best = sub.loc[sub["front_t"].idxmin()]
        key = "_".join(str(best[c]) for c in keycols)
        sig, yy = sigs[key], out[ycol_of(best)].to_numpy()
        back_rank = int((sub["back_d"] < best["back_d"]).sum() + 1)
        res[y] = {"cell": {c: (int(best[c]) if c != "ind" else best[c]) for c in keycols},
                  "front_d": round(best["front_d"], 2), "front_t": round(best["front_t"], 2), "front_n": int(best["front_n"]),
                  "back_d": round(best["back_d"], 2), "back_n": int(best["back_n"]), "back_rank": f"{back_rank}/{sub['back_d'].notna().sum()}",
                  "back_p": round(shift_p(np.where(_period(idx, "back"), sig, np.nan), np.where(_period(idx, "back"), yy, np.nan)), 3),
                  "y2025_d": None if pd.isna(best["y2025_d"]) else round(best["y2025_d"], 2),
                  "y2025_p": round(shift_p(np.where(_period(idx, "y2025"), sig, np.nan), np.where(_period(idx, "y2025"), yy, np.nan)), 3)}
    return res


def consistency_null(ind: dict, out: dict, idx: pd.Index, seed: int = 7) -> dict:
    """front·back 모두 가설 방향(차이 < 0)인 칸 수 — 관측 vs 모든 신호열을 같은 거리만큼 원형 이동한 200번."""
    y26 = np.asarray(idx >= "2026-01-02")
    sub_idx = idx[y26]
    fr, bk = _period(sub_idx, "front"), _period(sub_idx, "back")
    sigs = {k: (p >= TH).astype(float).where(p.notna()).to_numpy()[y26] for k, p in ind.items()}
    ys = {k: v.to_numpy()[y26] for k, v in out.items()}

    def count(shift: int) -> dict:
        c = {"R": 0, "M": 0}
        for s in sigs.values():
            ss = np.roll(s, shift)
            for L in LS:
                for y in ("R", "M"):
                    yy = ys[f"{y}{L}"]
                    d1 = diff_t(np.where(fr, ss, np.nan), np.where(fr, yy, np.nan))[0]
                    d2 = diff_t(np.where(bk, ss, np.nan), np.where(bk, yy, np.nan))[0]
                    c[y] += int(d1 < 0 and d2 < 0)
        return c

    obs = count(0)
    rng = np.random.default_rng(seed)
    null = [count(int(rng.integers(MIN_SHIFT, len(sub_idx) - MIN_SHIFT))) for _ in range(N_NULL)]
    return {y: {"obs": obs[y], "null_mean": round(float(np.mean([n[y] for n in null])), 1),
                "p": round(float((sum(n[y] >= obs[y] for n in null) + 1) / (N_NULL + 1)), 3), "cells": 3 * len(WS) * len(LS)}
            for y in ("R", "M")}


def main() -> None:
    days = fetch.load_cboe().index
    px = ohlc()
    px = px[px.index <= days[-1]]
    ind = indicators(days)
    out = outcomes(px, days, "QQQ")
    gA = grid_A(ind, out, days)
    sigA = {k: (p >= TH).astype(float).where(p.notna()).to_numpy() for k, p in ind.items()}
    selA = pick_and_confirm(gA, sigA, out, days, ["ind", "W"], lambda b: f"{b['y']}{int(b['L'])}")
    # 칸 열쇠는 지표·W, 결과 열은 L에 달려 있어 L별로 따로 확인한다
    for y in ("R", "M"):
        sub = gA[(gA["y"] == y) & gA["front_t"].notna()]
        best = sub.loc[sub["front_t"].idxmin()]
        selA[y]["cell"]["L"] = int(best["L"])
    gB, evB = grid_B(ind, out, days)
    sigB = {k: v.to_numpy() for k, v in evB.items()}
    selB = pick_and_confirm(gB, sigB, out, days, ["ind", "W", "k"], lambda b: f"{b['y']}21")
    cons = consistency_null(ind, out, days)
    soxx = outcomes(px, days, "SOXX")
    gA_s = grid_A(ind, soxx, days)
    out_c = outcomes(px, days, "QQQ", base="close")
    gA_c = grid_A(ind, out_c, days)
    gA.to_csv(RES / "lag_A.csv", index=False)
    gA_s.to_csv(RES / "lag_A_soxx.csv", index=False)
    gA_c.to_csv(RES / "lag_A_closebase.csv", index=False)
    gB.to_csv(RES / "lag_B.csv", index=False)
    base = {f"{y}{L}|{per}": round(float(np.nanmean(out[f'{y}{L}'].to_numpy()[_period(days, per)])), 2)
            for y in ("R", "M") for L in LS for per in PERIODS}
    res = {"A_pick": selA, "B_pick": selB, "A_consistency": cons, "base": base}
    (RES / "lag.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("A_pick", "B_pick", "A_consistency")}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
