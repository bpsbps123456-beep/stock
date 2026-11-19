"""갈래 bot_confirm — 구조 반전 확인('바닥이 떴다'를 언제 아는가).

모든 신호는 하락 상황 게이트(3분봉 종가 <= 최근 40개 3분봉 고가 최고 x (1-depth), depth 3% 또는 5%)가
'최근 15분(1분봉 15개, 현재 포함) 안에 한 번이라도 참'일 때만 인정한다. 게이트는 체결 1분봉 시각에 이미 완성된 3분봉 값만 쓴다.
신호는 완성봉 기준, 체결은 harness.exec_index(완성 뒤 첫 1분봉 시가), 겹침 제거 dedup(30).
피벗은 Pine ta.pivothigh/pivotlow(R,R)과 같은 판정(high[R] == ta.highest(high, 2R+1))이고 확정 봉(j+R)부터만 쓴다.

실행: py code/bot_confirm_grid.py  →  results/bot_confirm.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

OUT = os.path.join(H.ROOT, "results", "bot_confirm.json")
GATE_MIN = 15  # 게이트 기억 창(분)


# ───────────── 공통 상황 조건(모든 갈래 동일) ─────────────
d1 = H.load_1m()
b3 = H.resample(3)
RALLY = H.to_1m_mask(H.ctx_rally(b3, 20, 0.015), b3)
DECLINE = H.to_1m_mask(H.ctx_decline(b3, 40, 0.03), b3)


def recent_mask(mask: np.ndarray, n: int = GATE_MIN) -> np.ndarray:
    """1분봉 기준 최근 n개 봉(현재 포함) 안에 한 번이라도 참."""
    return pd.Series(mask.astype(float)).rolling(n, min_periods=1).max().to_numpy() > 0


GATES = {
    "d3": recent_mask(DECLINE),
    "d5": recent_mask(H.to_1m_mask(H.ctx_decline(b3, 40, 0.05), b3)),
}


# ───────────── 지표 도우미 ─────────────
def pivot_flags(s: pd.Series, R: int, kind: str) -> pd.Series:
    """확정 봉 t에서 참: s[t-R]이 [t-2R, t] 창의 최고(최저). 값은 s.shift(R)."""
    w = 2 * R + 1
    if kind == "high":
        ext = s.rolling(w).max()
    else:
        ext = s.rolling(w).min()
    return (s.shift(R) == ext) & ext.notna()


def last_two_pivots(s: pd.Series, R: int, kind: str):
    """각 봉에서 지금까지 확정된 마지막 피벗 값과 그 직전 피벗 값."""
    f = pivot_flags(s, R, kind)
    val = s.shift(R).where(f)
    last = val.ffill()
    # 직전 피벗: 피벗 값 시퀀스를 한 칸 밀어 ffill
    seq = val.dropna()
    prev_seq = seq.shift(1)
    prev = prev_seq.reindex(s.index).ffill()
    return f, last, prev


def sig_lh_break(b: pd.DataFrame, R: int) -> pd.Series:
    """마지막 확정 피벗 고점이 그 직전 피벗 고점보다 낮을 때(더 낮은 고점), 종가가 그 고점을 상향 돌파."""
    _, ph, pph = last_two_pivots(b["High"], R, "high")
    lower = ph < pph
    cross = (b["Close"] > ph) & (b["Close"].shift(1) <= ph.shift(1))
    return (cross & lower).fillna(False)


def sig_hl_confirm(b: pd.DataFrame, R: int, tf: int) -> pd.Series:
    """과매도(최근 60분 안 RSI14<=30) 뒤 새 확정 피벗 저점이 직전 피벗 저점보다 높음 → 확정 봉에서 신호."""
    f, pl, ppl = last_two_pivots(b["Low"], R, "low")
    r = H.rsi(b["Close"], 14)
    w = max(1, 60 // tf)
    os_ = (r.rolling(w, min_periods=1).min() <= 30)
    return (f & (pl > ppl) & os_ & (b["Close"] > pl)).fillna(False)


def sig_ema_reclaim(b: pd.DataFrame, n: int) -> pd.Series:
    """직전 3봉 중 하나라도 EMA 아래 마감 + 이번 봉 종가 > EMA(n) 그리고 종가 > 직전 봉 고가."""
    e = H.ema(b["Close"], n)
    below = (b["Close"] < e)
    was_below = below.shift(1, fill_value=False) | below.shift(2, fill_value=False) | below.shift(3, fill_value=False)
    return ((b["Close"] > e) & (b["Close"] > b["High"].shift(1)) & was_below).fillna(False)


def sig_vwap_reclaim(b: pd.DataFrame) -> pd.Series:
    vw = H.session_vwap(b) if "date" in b else None
    c = b["Close"]
    return ((c > vw) & (c.shift(1) <= vw.shift(1)) & (b["date"] == b["date"].shift(1))).fillna(False)


def sig_engulf(b: pd.DataFrame, N: int) -> pd.Series:
    """연속 N봉 음봉(종가<시가) 뒤 첫 장악형 양봉: 종가>시가, 종가>=직전 시가, 시가<=직전 종가."""
    dn = (b["Close"] < b["Open"]).astype(int)
    run = dn.shift(1).rolling(N).sum() == N
    bull = (b["Close"] > b["Open"]) & (b["Close"] >= b["Open"].shift(1)) & (b["Open"] <= b["Close"].shift(1))
    return (run & bull).fillna(False)


def sig_close_gt_prev_high(b: pd.DataFrame) -> pd.Series:
    return (b["Close"] > b["High"].shift(1)).fillna(False)


def sig_bounce(b: pd.DataFrame, x: float, tf: int) -> pd.Series:
    """지연 사다리: 최근 60분(1m 60봉/3m 20봉, 현재 포함) 최저 저가 대비 종가가 +x 이상으로 처음 올라선 봉."""
    w = max(1, 60 // tf)
    lo = b["Low"].rolling(w, min_periods=1).min()
    lvl = lo * (1 + x)
    return ((b["Close"] >= lvl) & (b["Close"].shift(1) < lvl.shift(1))).fillna(False)


def sig_gate_entry(b: pd.DataFrame) -> pd.Series:
    """참조(확인 없음): 3분봉이 하락 상황에 처음 들어간 봉."""
    c = H.ctx_decline(b, 40, 0.03)
    return (c & ~c.shift(1, fill_value=False)).fillna(False)


def build_configs():
    cfgs = []
    for tf in (1, 3):
        for R in (2, 4):
            cfgs.append((f"LHbreak_R{R}", tf, lambda b, R=R: sig_lh_break(b, R)))
        for R in (2, 4):
            cfgs.append((f"HLconfirm_R{R}", tf, lambda b, R=R, tf=tf: sig_hl_confirm(b, R, tf)))
        for n in (10, 25):
            cfgs.append((f"EMA{n}reclaim_hi1", tf, lambda b, n=n: sig_ema_reclaim(b, n)))
        cfgs.append(("VWAPreclaim", tf, sig_vwap_reclaim))
        for N in (3, 5):
            cfgs.append((f"Engulf_after{N}down", tf, lambda b, N=N: sig_engulf(b, N)))
        cfgs.append(("Close_gt_prevHigh", tf, sig_close_gt_prev_high))
    for tf in (1, 3):
        for x in (0.005, 0.01, 0.02, 0.03):
            cfgs.append((f"Bounce{x*100:g}pct_from60mLow", tf, lambda b, x=x, tf=tf: sig_bounce(b, x, tf)))
    return cfgs


# ───────────── 확인 지연·바닥 거리(실시간 값) ─────────────
L1 = d1["Low"].to_numpy(float)
O1 = d1["Open"].to_numpy(float)
T1 = d1.index.values
D1 = d1["date"].values


def lag_stats(pos: np.ndarray, look: int = 60):
    """체결 시점까지 '이미 알 수 있는' 최근 look분(같은 날) 최저가 대비 체결가 거리와 그 저점 뒤 경과 분."""
    dist, lag = [], []
    for p in pos:
        lo = max(0, p - look)
        idx = np.arange(lo, p)
        idx = idx[D1[idx] == D1[p]]
        if not len(idx):
            continue
        k = idx[np.argmin(L1[idx])]
        dist.append(O1[p] / L1[k] - 1)
        lag.append((T1[p] - T1[k]) / np.timedelta64(1, "m"))
    return dist, lag


def score(pos_all: np.ndarray, rng) -> dict:
    ev = H.event_metrics(pos_all)
    s = H.summarize(ev, rng, "bottom", DECLINE)
    e = ev[H.in_range(ev["date"].values, rng) & ev["reg"].values]
    dist, lag = lag_stats(e["pos"].to_numpy())
    out = {k: s.get(k) for k in ("n", "days", "per_day", "r60_mean%", "r60_x_ctx%", "r60_x_dm%", "r60_x_dm_t",
                                  "rclose_mean%", "rclose_x_ctx%", "rclose_x_dm_t", "r30_x_ctx%",
                                  "mae60_med%", "mfe60_med%", "fp_good%", "fp_good_base%", "r60_win%")}
    out["dist_from_low_med%"] = round(100 * float(np.median(dist)), 3) if dist else None
    out["lag_from_low_med_min"] = round(float(np.median(lag)), 1) if lag else None
    return out


def main():
    t0 = time.time()
    bars = {1: H.resample(1), 3: b3}
    res = {"grid": {}, "meta": {}}
    cfgs = build_configs()
    n_cfg = 0
    for name, tf, fn in cfgs:
        b = bars[tf]
        sig = fn(b)
        raw = H.exec_index(sig, b)
        raw = raw[raw >= 0]
        for gname, g in (GATES.items() if not name.startswith("Bounce") else [("d3", GATES["d3"])]):
            pos = H.dedup(raw[g[raw]], 30)
            key = f"{name}/{tf}m/{gname}"
            res["grid"][key] = {nm: score(pos, rng) for nm, rng in (("TRAIN", H.TRAIN), ("TEST", H.TEST), ("PRE", H.PRE))}
            n_cfg += 1
            tr = res["grid"][key]["TRAIN"]
            print(key, tr.get("n"), tr.get("r60_x_ctx%"), tr.get("r60_x_dm_t"), tr.get("dist_from_low_med%"), round(time.time() - t0), flush=True)
    # 참조: 확인 없이 하락 상황 진입 즉시
    sig = sig_gate_entry(b3)
    pos = H.dedup(H.exec_index(sig, b3), 30)
    res["reference"] = {"GateEntry_3m_noconfirm": {nm: score(pos, rng) for nm, rng in (("TRAIN", H.TRAIN), ("TEST", H.TEST), ("PRE", H.PRE))}}
    n_cfg += 1
    res["meta"]["configs_tested"] = n_cfg
    res["meta"]["grid"] = ("6 families x 2 timeframes(1m,3m) x params(LH R{2,4}, HL R{2,4}, EMA{10,25}, VWAP 1, Engulf N{3,5}, C>H[1] 1)"
                           " = 20 signals x 2 gates(decline 3%/5% within 15m) = 40; + delay ladder Bounce x{0.5,1,2,3}% from 60m low x 2 tf, gate d3 only = 8;"
                           " + 1 reference(gate entry, no confirm) = 49")

    # TRAIN만으로 선택: n>=15, per_day 0.3~6, r60_x_ctx 내림차순 상위 3
    rows = []
    for k, v in res["grid"].items():
        tr = v["TRAIN"]
        if tr.get("n", 0) >= 15 and tr.get("per_day") is not None and 0.3 <= tr["per_day"] <= 6 and tr.get("r60_x_ctx%") is not None:
            rows.append((tr["r60_x_ctx%"], k))
    rows.sort(reverse=True)
    res["selection_rule"] = "TRAIN only: n>=15, per_day 0.3~6, sort by r60_x_ctx% desc, top 3 (no TEST reselection)"
    res["selected"] = [k for _, k in rows[:3]]
    res["train_rank"] = [(k, s) for s, k in rows]

    # 지연-위험 균형: TRAIN 설정 전체에서 바닥 거리 vs mae60, r60_x_ctx 상관
    df = pd.DataFrame([{**v["TRAIN"], "key": k} for k, v in res["grid"].items()]).dropna(subset=["dist_from_low_med%"])
    res["tradeoff_train"] = {
        "corr_dist_vs_mae60": round(float(df["dist_from_low_med%"].corr(df["mae60_med%"])), 3),
        "corr_dist_vs_r60x": round(float(df["dist_from_low_med%"].corr(df["r60_x_ctx%"])), 3),
        "corr_lag_vs_r60x": round(float(df["lag_from_low_med_min"].corr(df["r60_x_ctx%"])), 3),
        "by_tf": df.assign(tf=df["key"].str.split("/").str[1]).groupby("tf")[["dist_from_low_med%", "lag_from_low_med_min", "mae60_med%", "r60_x_ctx%"]].median().round(3).to_dict(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    print("selected", res["selected"])
    for k in res["selected"]:
        print(k, json.dumps(res["grid"][k], ensure_ascii=False))
    print("ref", json.dumps(res["reference"], ensure_ascii=False))
    print("tradeoff", json.dumps(res["tradeoff_train"], ensure_ascii=False))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
