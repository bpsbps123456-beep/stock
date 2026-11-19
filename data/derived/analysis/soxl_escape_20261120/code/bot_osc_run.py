"""갈래 bot_osc — 오실레이터로 바닥 확인('언제') (2026-11-20).

신호 가족(1분봉·3분봉 각각, 지표는 연장시간 포함 연속 계산):
  A  RSI14 L 상향 교차(ta.crossover): rsi[1] <= L and rsi > L, L in {15,20,25,30}
  B  Stoch RSI(3,3,14,14): K가 D를 상향 교차(K[1]<=D[1], K>D) and K[1] < thr, thr in {10,20}
  C  강세 다이버전스(확정 피벗만): 피벗 원천 {가격 저가, RSI} x 좌우 R in {3,5} x 과매도 조건 {없음, 현 피벗 RSI<30}
     현 피벗 p(확정 시점 q=p+R)와 직전 피벗 p' 거리 5..60봉, 가격 저가[p] < 저가[p'] and RSI[p] > RSI[p'].
     신호는 확정봉 q 완성 뒤 체결.
  E  RSI 최저 후 반등 폭: m = lowest(rsi, 20), m < L and rsi - m >= J 가 처음 참이 된 봉, L in {20,25}, J in {10,15}
  D  (1분봉 차트 전용) 1분·3분 동시 과매도
     D1(X): 1분 RSI 30 상향 교차 and 직전 완성 3분봉 RSI < X, X in {25,30,35}
     D2   : 1분 Stoch K/D 상향 교차(K[1]<20) and 직전 완성 3분봉 RSI < 30
     D3   : 1분 Stoch K/D 상향 교차(K[1]<20) and 직전 완성 3분봉 Stoch K < 20
각 신호 x 게이트 {조건 없음, DECLINE(체결 1분봉 시점의 하락 상황)} → dedup 30분.
채점: summarize(ev, rng, 'bottom', DECLINE). 1순위 r60_x_ctx%, t = r60_x_dm_t.
TRAIN만으로 상위 3개(n>=10, 하루 0.3~6회) 고르고 TEST·PRE 숫자는 그대로 적는다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

OUT = os.path.join(H.ROOT, "results", "bot_osc.json")


def _f(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    return x


# ───────────── 신호 ─────────────
def crossover(a: pd.Series, b) -> pd.Series:
    """ta.crossover(a, b): a > b and a[1] <= b[1]."""
    b1 = b if np.isscalar(b) else b.shift(1)
    return (a > b) & (a.shift(1) <= b1)


def pivot_low_confirm(src: pd.Series, L: int, R: int) -> pd.Series:
    """확정봉 q에서 참: src[q-R]이 왼쪽 L봉보다 엄격히 낮고 오른쪽 R봉 이하(<=). 미래 없음(q 시점에 다 앎)."""
    c = src.shift(R)
    right = src.rolling(R).min()                  # q-R+1..q
    left = src.shift(R + 1).rolling(L).min()       # q-R-L..q-R-1
    return (c < left) & (c <= right)


def bull_div(bars: pd.DataFrame, r: pd.Series, source: str, R: int, os_lvl: float | None,
             rng_min: int = 5, rng_max: int = 60) -> pd.Series:
    low = bars["Low"]
    src = low if source == "price" else r
    pc = pivot_low_confirm(src, R, R).to_numpy()
    q_idx = np.nonzero(pc)[0]
    lo = low.to_numpy()
    rv = r.to_numpy()
    sig = np.zeros(len(bars), dtype=bool)
    prev = None
    for q in q_idx:
        p = q - R
        if prev is not None:
            pp = prev - R
            dist = p - pp
            if rng_min <= dist <= rng_max and lo[p] < lo[pp] and rv[p] > rv[pp]:
                if os_lvl is None or rv[p] < os_lvl:
                    sig[q] = True
        prev = q
    return pd.Series(sig, index=bars.index)


def rsi_rebound(r: pd.Series, L: float, J: float, W: int = 20) -> pd.Series:
    m = r.rolling(W).min()
    c = (m < L) & (r - m >= J)
    return c & ~c.shift(1, fill_value=False)


def htf_on_1m(series: pd.Series, bars: pd.DataFrame) -> np.ndarray:
    """1분봉 시각(봉 시작)에 이미 완성된 최신 상위봉 값(Pine request.security(expr[1], lookahead_on)와 같음)."""
    d1 = H.load_1m()
    ends = bars["end"].values
    k = np.searchsorted(ends, d1.index.values, side="right") - 1
    v = series.reindex(bars.index).to_numpy(float)
    return np.where(k >= 0, v[np.clip(k, 0, len(v) - 1)], np.nan)


def build_signals():
    d1 = H.load_1m()
    sigs = {}  # name -> (bars, bool Series, tf, params)
    ind = {}
    for tf in (1, 3):
        b = H.resample(tf)
        r = H.rsi(b["Close"], 14)
        K, D = H.stoch_rsi(b["Close"], 3, 3, 14, 14)
        ind[tf] = (b, r, K, D)
        for L in (15, 20, 25, 30):
            sigs[f"A_rsiX{L}_{tf}m"] = (b, crossover(r, L), tf, {"fam": "A", "L": L})
        for thr in (10, 20):
            s = crossover(K, D) & (K.shift(1) < thr)
            sigs[f"B_stochX_k{thr}_{tf}m"] = (b, s, tf, {"fam": "B", "thr": thr})
        for src in ("price", "rsi"):
            for R in (3, 5):
                for osl in (None, 30):
                    nm = f"C_div_{src}_R{R}_{'os30' if osl else 'any'}_{tf}m"
                    sigs[nm] = (b, bull_div(b, r, src, R, osl), tf, {"fam": "C", "src": src, "R": R, "os": osl})
        for L in (20, 25):
            for J in (10, 15):
                sigs[f"E_rsiMin{L}_reb{J}_{tf}m"] = (b, rsi_rebound(r, L, J), tf, {"fam": "E", "L": L, "J": J, "W": 20})
    # D: 1분 차트 + 3분 완성봉
    b1, r1, K1, D1_ = ind[1]
    b3, r3, K3, _ = ind[3]
    r3c = pd.Series(htf_on_1m(r3, b3), index=b1.index)
    k3c = pd.Series(htf_on_1m(K3, b3), index=b1.index)
    x30 = crossover(r1, 30)
    st1 = crossover(K1, D1_) & (K1.shift(1) < 20)
    for X in (25, 30, 35):
        sigs[f"D1_rsi1X30_rsi3lt{X}"] = (b1, x30 & (r3c < X), 1, {"fam": "D1", "X": X})
    sigs["D2_stoch1_rsi3lt30"] = (b1, st1 & (r3c < 30), 1, {"fam": "D2"})
    sigs["D3_stoch1_k3lt20"] = (b1, st1 & (k3c < 20), 1, {"fam": "D3"})
    return sigs, ind


def location_desc(pos: np.ndarray, rng) -> dict:
    """설명용(실시간 값): 체결 시점 그날 고가(04:00부터 누적) 대비 낙폭, 3분 EMA60 대비 ATR 배수."""
    d1 = H.load_1m()
    if "loc" not in H._CACHE:
        dd = d1["High"].groupby(d1["date"]).cummax()
        b3 = H.resample(3)
        e60 = H.ema(b3["Close"], 60)
        a14 = H.atr(b3, 14)
        z = (b3["Close"] - e60) / a14
        H._CACHE["loc"] = (dd.to_numpy(float), htf_on_1m(z, b3))
    cmh, z1 = H._CACHE["loc"]
    p = pos[H.in_range(d1["date"].values[pos], rng) & d1["reg"].values[pos]]
    if not len(p):
        return {}
    O = d1["Open"].to_numpy(float)
    return {"dd_from_dayhigh_med%": round(100 * float(np.median(O[p] / cmh[p - 1] - 1)), 2),
            "z_ema60_atr3_med": round(float(np.nanmedian(z1[p])), 2)}


def main():
    b3 = H.resample(3)
    RALLY = H.to_1m_mask(H.ctx_rally(b3, 20, 0.015), b3)  # noqa: F841 (공통 조건, 바닥 갈래는 미사용)
    DECLINE = H.to_1m_mask(H.ctx_decline(b3, 40, 0.03), b3)
    sigs, _ = build_signals()
    keys = ["n", "days", "per_day", "r60_x_ctx%", "r60_x_dm_t", "r60_x_dm%", "r60_x_all%", "r60_mean%",
            "rclose_x_ctx%", "rclose_x_dm_t", "rclose_x_all%", "mae60_med%", "mfe60_med%", "fp_good%", "fp_good_base%",
            "r30_x_ctx%"]
    rows = []
    for nm, (bars, s, tf, prm) in sigs.items():
        raw = H.exec_index(s, bars)
        raw = raw[raw >= 0]
        for gate in ("all", "decline"):
            p = raw[DECLINE[raw]] if gate == "decline" else raw
            p = H.dedup(p, 30)
            ev = H.event_metrics(p)
            res = {"name": f"{nm}|{gate}", "sig": nm, "gate": gate, "tf": tf, "params": prm}
            for rng, tag in ((H.TRAIN, "train"), (H.TEST, "test"), (H.PRE, "pre")):
                sm = H.summarize(ev, rng, "bottom", DECLINE)
                res[tag] = {k: _f(sm.get(k)) for k in keys}
                res[tag]["loc"] = location_desc(p, rng)
            rows.append(res)
            print(f"{res['name']:40s} TR n={res['train']['n']:3} r60x={res['train'].get('r60_x_ctx%')} t={res['train'].get('r60_x_dm_t')}"
                  f" | TE n={res['test']['n']:3} r60x={res['test'].get('r60_x_ctx%')} t={res['test'].get('r60_x_dm_t')}", flush=True)
    elig = [r for r in rows if (r["train"]["n"] or 0) >= 10 and 0.3 <= (r["train"].get("per_day") or 0) <= 6
            and r["train"].get("r60_x_ctx%") is not None]
    elig.sort(key=lambda r: r["train"]["r60_x_ctx%"], reverse=True)
    raw_top = [r["name"] for r in elig[:6]]
    # 같은 신호의 두 게이트는 사건이 거의 같으므로(깊은 과매도는 대부분 이미 DECLINE) 신호당 TRAIN 1순위가 높은 게이트 하나만 남긴다.
    seen, top = set(), []
    for r in elig:
        if r["sig"] in seen:
            continue
        seen.add(r["sig"])
        top.append(r["name"])
        if len(top) == 3:
            break
    # 상위 후보 설명: TEST 사건별 r60과 그날 하락상황 평균(왜 ctx 대조와 dm 대조가 갈리는지)
    d1 = H.load_1m()
    f = H._fwd_tables()
    detail = {}
    for nm in top:
        sig, gate = nm.split("|")
        bars, s, tf, prm = sigs[sig]
        raw = H.exec_index(s, bars)
        raw = raw[raw >= 0]
        p = H.dedup(raw[DECLINE[raw]] if gate == "decline" else raw, 30)
        ev = H.event_metrics(p)
        det = {}
        for rng, tag in ((H.TRAIN, "train"), (H.TEST, "test")):
            e = ev[H.in_range(ev["date"].values, rng) & ev["reg"].values]
            det[tag] = [{"time": str(t), "r60%": None if np.isnan(a) else round(100 * a, 2),
                         "rclose%": None if np.isnan(c) else round(100 * c, 2),
                         "mae60%": None if np.isnan(m) else round(100 * m, 2)}
                        for t, a, c, m in zip(e["time"], e["r60"], e["rclose"], e["mae60"])]
            det[tag + "_r60_median%"] = round(100 * float(np.nanmedian(e["r60"])), 3) if len(e) else None
        detail[nm] = det
    base = {}
    for rng, tag in ((H.TRAIN, "train"), (H.TEST, "test"), (H.PRE, "pre")):
        bc = H.baseline_positions(rng, DECLINE)
        base[tag] = {"decline_bars": int(len(bc)), "decline_r60_mean%": round(100 * float(np.nanmean(f["r60"][bc])), 3),
                     "decline_rclose_mean%": round(100 * float(np.nanmean(f["rclose"][bc])), 3)}
    out = {"branch": "bot_osc", "configs_tested": len(rows), "n_signals": len(sigs), "gates": ["all", "decline"],
           "selection": "TRAIN r60_x_ctx% 내림차순, 조건 n>=10 & 하루 0.3~6회, 신호당 게이트 1개, 상위 3개(TEST 재선택 없음)",
           "eligible": len(elig), "raw_top6": raw_top, "top3": top, "decline_baseline": base,
           "top3_detail": detail, "rows": rows}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(_f(out), fh, ensure_ascii=False, indent=1, default=str)
    print("TOP3", top)
    print("configs", len(rows), "eligible", len(elig))


if __name__ == "__main__":
    main()
