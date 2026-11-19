"""갈래 bot_dev — 바닥이 뜨는 '위치(어디쯤)' 연구.

1) 설명용(사후): TRAIN 정규장 1분봉 지그재그 3% 저점(= 이후 +3% 반등 전에 더 낮은 저가가 없던 저점)이
   위치 특징값 분포의 어디에 있었는지 하락상황(DECLINE) 정규장 봉 전체와 비교한다. 매매 근거로 쓰지 않는다.
2) 실시간 규칙: 구역 진입 봉(entry) 또는 구역 안 첫 반등봉(rebound: 최근 3봉 안에 구역 + 종가 > 직전봉 고가)을
   완성봉 기준으로 신호화 → 다음 1분봉 시가 체결(H.exec_index) → 정규장·DECLINE 상황만 → dedup 30분.
   채점 H.summarize(ev, rng, 'bottom', DECLINE). 1순위 r60_x_ctx%, t는 r60_x_dm_t.
선택: TRAIN만, n>=10 & per_day<=6, r60_x_ctx% 순, 특징 계열당 1개로 상위 3개(사전 고정). TEST는 그대로 기록.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

OUT = os.path.join(H.ROOT, "results", "bot_dev.json")

d1 = H.load_1m()
b3 = H.resample(3)
RALLY = H.to_1m_mask(H.ctx_rally(b3, 20, 0.015), b3)
DECLINE = H.to_1m_mask(H.ctx_decline(b3, 40, 0.03), b3)
REG1 = d1["reg"].to_numpy()


# ───────────────────────── 특징(완성봉 기준, 미래 없음) ─────────────────────────
def features(b: pd.DataFrame, tf: int) -> pd.DataFrame:
    C, O, Hh, L, V = b["Close"], b["Open"], b["High"], b["Low"], b["Volume"]
    e60 = H.ema(C, 60)
    a14 = H.atr(b, 14)
    f = pd.DataFrame(index=b.index)
    f["devA"] = (C - e60) / a14                       # EMA60 대비 ATR 배수
    f["devP"] = 100 * (C / e60 - 1)                   # EMA60 대비 %
    tp = (Hh + L + C) / 3
    cv = V.groupby(b["date"]).cumsum().replace(0, np.nan)
    vw = (tp * V).groupby(b["date"]).cumsum() / cv
    var = (tp * tp * V).groupby(b["date"]).cumsum() / cv - vw * vw
    sd = np.sqrt(var.clip(lower=0))
    f["vwz"] = (C - vw) / sd.replace(0, np.nan)        # 세션 VWAP σ밴드 배수(ta.vwap stdev 정의)
    f["ddDay"] = 100 * (C / Hh.groupby(b["date"]).cummax() - 1)   # 오늘(04:00~) 지금까지 고가 대비
    nroll = 120 // tf
    f["ddRoll"] = 100 * (C / Hh.rolling(nroll, min_periods=1).max() - 1)  # 최근 120분 고가 대비
    # 전일 정규장 종가(전일 15:59 1분봉 종가)
    reg_close = d1.loc[d1["reg"], "Close"].groupby(d1.loc[d1["reg"], "date"]).last()
    pc = reg_close.shift(1)
    f["ddPC"] = 100 * (C / b["date"].map(pc).to_numpy() - 1)
    dn = (C < C.shift(1)).astype(int)
    grp = (dn == 0).cumsum()
    f["downRun"] = dn.groupby(grp).cumsum()            # 연속 하락 종가 봉 수
    vm, vs = V.rolling(20).mean(), V.rolling(20).std(ddof=0)
    f["volz"] = (V - vm) / vs.replace(0, np.nan)
    rngb = (Hh - L).replace(0, np.nan)
    f["wick"] = (np.minimum(O, C) - L) / rngb          # 아래꼬리 비율
    f["rsi"] = H.rsi(C, 14)
    f["e60"], f["atr"], f["vwap"], f["vwsd"] = e60, a14, vw, sd
    return f


# 구역 정의: (계열, 이름, 함수(f)->bool Series, 사람이 읽는 선)
def zones():
    z = []
    for k in (2, 3, 4, 6):
        z.append(("ema_atr", f"C<=EMA60-{k}ATR", lambda f, k=k: f["devA"] <= -k))
    for x in (1.5, 3.0, 4.5):
        z.append(("ema_pct", f"C<=EMA60*(1-{x}%)", lambda f, x=x: f["devP"] <= -x))
    for k in (1, 2, 3):
        z.append(("vwap_sd", f"C<=VWAP-{k}sd", lambda f, k=k: f["vwz"] <= -k))
    for x in (4, 6, 8, 10):
        z.append(("dd_day", f"C<=dayHigh*(1-{x}%)", lambda f, x=x: f["ddDay"] <= -x))
    for x in (5, 7, 9):
        z.append(("dd_roll", f"C<=high120m*(1-{x}%)", lambda f, x=x: f["ddRoll"] <= -x))
    for x in (3, 6, 9, 12):
        z.append(("dd_pc", f"C<=prevClose*(1-{x}%)", lambda f, x=x: f["ddPC"] <= -x))
    for n in (4, 6, 8):
        z.append(("down_run", f"downRun>={n}", lambda f, n=n: f["downRun"] >= n))
    for zz in (2, 3):
        z.append(("capit", f"volz>={zz}&wick>=0.5", lambda f, zz=zz: (f["volz"] >= zz) & (f["wick"] >= 0.5)))
    return z


def signal(f: pd.DataFrame, b: pd.DataFrame, zfun, timing: str) -> pd.Series:
    zn = zfun(f).fillna(False).astype(bool)
    if timing == "entry":
        return zn & ~zn.shift(1, fill_value=False)
    recent = zn.astype(int).rolling(3, min_periods=1).max().astype(bool)
    return recent & (b["Close"] > b["High"].shift(1))


def positions(sig: pd.Series, b: pd.DataFrame) -> np.ndarray:
    pos = H.exec_index(sig, b)
    pos = pos[pos >= 0]
    pos = pos[REG1[pos] & DECLINE[pos]]
    return H.dedup(pos, 30)


def score(pos, rng):
    ev = H.event_metrics(pos)
    s = H.summarize(ev, rng, "bottom", DECLINE)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in s.items()}


# ───────────────────────── 1) 설명용 사후 바닥 ─────────────────────────
def zigzag_lows_sm(rng, pct=0.03):
    """정확한 지그재그: 방향 상태기계. 저점 = 이후 +pct 반등 전 더 낮은 저가 없음 & 그 전 고점에서 -pct 하락."""
    sel = REG1 & H.in_range(d1["date"].values, rng)
    idx = np.nonzero(sel)[0]
    Hv, Lv = d1["High"].to_numpy(float), d1["Low"].to_numpy(float)
    out = []
    for dt in np.unique(d1["date"].values[idx]):
        ii = idx[d1["date"].values[idx] == dt]
        hi_i, hi = ii[0], Hv[ii[0]]
        lo_i, lo = ii[0], Lv[ii[0]]
        mode = 0  # 0=미정, 1=상승 추적(고점 갱신), -1=하락 추적(저점 갱신)
        for i in ii:
            if mode >= 0 and Hv[i] > hi:
                hi_i, hi = i, Hv[i]
            if mode <= 0 and Lv[i] < lo:
                lo_i, lo = i, Lv[i]
            if mode != 1 and Hv[i] >= lo * (1 + pct) and lo_i <= i:
                if mode == -1 or mode == 0:
                    out.append(lo_i)
                mode, hi_i, hi = 1, i, Hv[i]
            elif mode != -1 and Lv[i] <= hi * (1 - pct):
                mode, lo_i, lo = -1, i, Lv[i]
    return np.array(out, dtype=int)


def explain(rng):
    lows = zigzag_lows_sm(rng)
    rep = {"n_bottoms": int(len(lows)), "days": int(len(np.unique(d1["date"].values[lows]))),
           "share_in_DECLINE%": round(100 * float(DECLINE[lows].mean()), 1)}
    base = H.baseline_positions(rng, DECLINE)
    base_all = H.baseline_positions(rng)
    tabs = {}
    for tf in (1, 3):
        b = H.resample(tf)
        f = features(b, tf)
        # 1분봉 위치 → 그 시각을 포함한 tf봉(그 봉 완성값, 설명용)
        k = np.searchsorted(b.index.values, d1.index.values, side="right") - 1
        fv = f.iloc[k].reset_index(drop=True)
        q = {}
        for col in ("devA", "devP", "vwz", "ddDay", "ddRoll", "ddPC", "downRun", "volz", "wick", "rsi"):
            v = fv[col].to_numpy(float)
            qb = np.nanquantile(v[lows], [0.1, 0.25, 0.5, 0.75, 0.9])
            qd = np.nanquantile(v[base], [0.1, 0.25, 0.5, 0.75, 0.9])
            q[col] = {"bottoms_q10_25_50_75_90": [round(x, 2) for x in qb],
                      "DECLINE_q10_25_50_75_90": [round(x, 2) for x in qd]}
        # 구역별: 바닥 중 구역 안 비율 vs 하락상황 봉 중 구역 안 비율 → 배율(lift)
        zl = {}
        for fam, nm, zf in zones():
            zn = zf(f).fillna(False).to_numpy()[k]
            cov = float(zn[lows].mean())
            br = float(zn[base].mean())
            # 바닥 '근처'(바닥 ±5분 안)를 구역이 포함한 비율
            near = np.array([zn[max(p - 5, 0):p + 6].any() for p in lows]).mean()
            zl[nm] = {"bottom_cover%": round(100 * cov, 1), "decline_bar%": round(100 * br, 1),
                      "lift": round(cov / br, 2) if br > 0 else None, "cover_pm5min%": round(100 * float(near), 1),
                      "allreg_bar%": round(100 * float(zn[base_all].mean()), 1)}
        tabs[f"{tf}m"] = {"quantiles": q, "zone_lift": zl}
    rep["by_tf"] = tabs
    return rep


# ───────────────────────── 2) 실시간 규칙 격자 ─────────────────────────
def grid():
    rows = []
    for tf in (1, 3):
        b = H.resample(tf)
        f = features(b, tf)
        for fam, nm, zf in zones():
            for timing in ("entry", "rebound"):
                sig = signal(f, b, zf, timing)
                pos = positions(sig, b)
                rows.append({"tf": tf, "family": fam, "zone": nm, "timing": timing, "pos": pos,
                             "TRAIN": score(pos, H.TRAIN), "TEST": score(pos, H.TEST), "PRE": score(pos, H.PRE)})
    return rows


def pick(rows):
    ok = [r for r in rows if r["TRAIN"].get("n", 0) >= 10 and r["TRAIN"].get("per_day", 99) <= 6]
    ok.sort(key=lambda r: -r["TRAIN"]["r60_x_ctx%"])
    top, fams = [], set()
    for r in ok:
        if r["family"] in fams:
            continue
        top.append(r)
        fams.add(r["family"])
        if len(top) == 3:
            break
    return ok, top


KEYS = ["n", "days", "per_day", "r60_x_ctx%", "r60_x_dm%", "r60_x_dm_t", "rclose_x_ctx%", "rclose_x_dm_t",
        "r60_mean%", "mae60_med%", "mfe60_med%", "fp_good%", "fp_good_base%", "r30_x_ctx%"]


def slim(s):
    return {k: s.get(k) for k in KEYS}


if __name__ == "__main__":
    ex_train = explain(H.TRAIN)
    rows = grid()
    ok, top = pick(rows)
    # 확인용: 'DECLINE 아무 시점' 1분봉 3분봉 반등봉만(구역 없음) 대조 — 시점 효과만 분리
    ctrl = {}
    for tf in (1, 3):
        b = H.resample(tf)
        sig = b["Close"] > b["High"].shift(1)
        pos = positions(sig, b)
        ctrl[f"{tf}m_rebound_only"] = {"TRAIN": slim(score(pos, H.TRAIN)), "TEST": slim(score(pos, H.TEST))}
    res = {
        "branch": "bot_dev",
        "configs_tested": len(rows),
        "grid": "TF{1m,3m} x 26 zones(ema_atr k{2,3,4,6}; ema_pct {1.5,3,4.5}; vwap_sd {1,2,3}; dd_day {4,6,8,10}; "
                "dd_roll120m {5,7,9}; dd_prevclose {3,6,9,12}; downRun {4,6,8}; capit volz{2,3}&wick>=0.5) "
                "x timing{entry, rebound(최근3봉 구역 & C>H[1])} = 104; 정규장+DECLINE 게이트, dedup30",
        "selection": "TRAIN n>=10, per_day<=6, r60_x_ctx% 내림차순, 특징 계열당 1개 상위 3",
        "explain_TRAIN": ex_train,
        "controls": ctrl,
        "all": [{"tf": r["tf"], "family": r["family"], "zone": r["zone"], "timing": r["timing"],
                 "TRAIN": slim(r["TRAIN"]), "TEST": slim(r["TEST"]), "PRE": slim(r["PRE"])} for r in rows],
        "train_rank_top15": [f'{r["tf"]}m|{r["zone"]}|{r["timing"]}|{r["TRAIN"]["r60_x_ctx%"]}' for r in ok[:15]],
        "top3": [{"tf": r["tf"], "family": r["family"], "zone": r["zone"], "timing": r["timing"],
                  "TRAIN": r["TRAIN"], "TEST": r["TEST"], "PRE": slim(r["PRE"])} for r in top],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    print("configs", len(rows), "eligible", len(ok))
    for s in res["train_rank_top15"]:
        print(s)
    for r in top:
        print("TOP", r["tf"], r["zone"], r["timing"], "\n TRAIN", slim(r["TRAIN"]), "\n TEST", slim(r["TEST"]), "\n PRE", slim(r["PRE"]))
    print("CTRL", json.dumps(ctrl, ensure_ascii=False, default=float))
