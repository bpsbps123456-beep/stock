"""갈래 exit_struct: 구조 이탈선(가격 수준)으로 긴급탈출 (2026-11-20).

실행: py code/exit_struct_scan.py  (작업 폴더 soxl_escape_20261120)
결과: results/exit_struct.json

1단계: 이탈선 16종 x 시간봉 2종(1분·3분) x 확인방식 5종 x 거래량 2종 = 320 사건형 + 상태형 32 = 352 설정.
2단계: 1단계 TRAIN 상위 5개 기준선 x '계속 빠짐' 거르개 10종 = 50 설정.
3단계: 2단계 TRAIN 상위 거르개 2종 묶음 (상위 3 기준선 x 거르개쌍 3) = 9 설정.
선택은 TRAIN hold_sim x_rand_mean% 만(하루 0.3~6회, 사건 10건 이상). TEST는 그대로 적는다.

미래 정보 차단: 피벗은 확정(i+R) 다음 봉부터, 돈치안·전저점은 직전 봉까지의 값, 시가 범위는 범위 끝난 뒤,
전일 값은 전일 정규장, 프리장 저점은 09:30 이후, 3분 맥락은 완성봉만(to_1m_mask).
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

OUT = os.path.join(H.ROOT, "results", "exit_struct.json")

d1 = H.load_1m()
b3 = H.resample(3)
RALLY = H.to_1m_mask(H.ctx_rally(b3, 20, 0.015), b3)
DECLINE = H.to_1m_mask(H.ctx_decline(b3, 40, 0.03), b3)
TF = {1: H.resample(1), 3: b3}


# ───────────── 이탈선 ─────────────
def pivot_low_confirmed(bars: pd.DataFrame, L: int, R: int) -> pd.Series:
    """ta.pivotlow(low, L, R)와 같은 판정. 피벗 j는 j+R 봉에서 확정, 값은 j+R 봉부터 유지(다음 확정까지)."""
    lo = bars["Low"].to_numpy(float)
    n = len(lo)
    out = np.full(n, np.nan)
    piv = np.full(n, np.nan)  # 확정 시점 봉에 피벗 값
    for j in range(L, n - R):
        w = lo[j - L:j + R + 1]
        v = lo[j]
        # Pine: 왼쪽은 v 이하 금지(엄격), 오른쪽은 v 미만 금지 — 대칭 엄격으로 단순화(동률 드묾)
        if v <= w.min() and (lo[j - L:j] > v).all() and (lo[j + 1:j + R + 1] >= v).all():
            piv[j + R] = v
    s = pd.Series(piv, index=bars.index).ffill()
    return s, pd.Series(piv, index=bars.index)


def higher_low_line(bars, L, R):
    _, piv = pivot_low_confirmed(bars, L, R)
    v = piv.to_numpy()
    out = np.full(len(v), np.nan)
    prev = np.nan
    cur = np.nan
    for i, p in enumerate(v):
        if not np.isnan(p):
            cur = p if (not np.isnan(prev) and p > prev) else np.nan  # 더 낮은 저점이면 구조 깨짐 → 선 없음
            prev = p
        out[i] = cur
    return pd.Series(out, index=bars.index)


def rally_anchor_line(bars, lb):
    """랠리 시작 저점: ctx_rally(lb, 1.5%)가 거짓→참이 된 봉에서 최근 lb봉 최저가를 기준선으로 잡고,
    이탈(종가<기준선)하거나 새 랠리 시작이 올 때까지 유지."""
    lo = bars["Low"].rolling(lb, min_periods=1).min()
    st = bars["Close"] >= lo * 1.015
    start = st & ~st.shift(1, fill_value=False)
    c = bars["Close"].to_numpy(float)
    lv = lo.to_numpy(float)
    sv = start.to_numpy()
    out = np.full(len(c), np.nan)
    cur = np.nan
    for i in range(len(c)):
        out[i] = cur  # i 봉 판정에 쓰는 선(직전까지 정해진 값)
        if not np.isnan(cur) and c[i] < cur:
            cur = np.nan
        if sv[i]:
            cur = lv[i]
    return pd.Series(out, index=bars.index)


def day_levels():
    reg = d1[d1["reg"]]
    g = reg.groupby("date")
    plow, pclose = g["Low"].min(), g["Close"].last()
    dates = sorted(d1["date"].unique())
    prev = {dates[i]: dates[i - 1] for i in range(1, len(dates))}
    pm = d1[d1["mod"] < H.REG_OPEN].groupby("date")["Low"].min()
    orl = {}
    for k in (5, 15, 30):
        m = (d1["mod"] >= H.REG_OPEN) & (d1["mod"] < H.REG_OPEN + k)
        orl[k] = d1[m].groupby("date")["Low"].min()
    return plow, pclose, prev, pm, orl


PLOW, PCLOSE, PREV, PMLOW, ORL = day_levels()


def session_lines(bars, tf):
    dt = bars["date"]
    endmod = bars["mod"] + tf
    pv = dt.map(PREV)
    out = {
        "prev_low": pv.map(PLOW).astype(float),
        "prev_close": pv.map(PCLOSE).astype(float),
        "pm_low": dt.map(PMLOW).astype(float).where(bars["mod"] >= H.REG_OPEN),
        "vwap": H.session_vwap(bars),
    }
    for k in (5, 15, 30):
        out[f"or{k}_low"] = dt.map(ORL[k]).astype(float).where(endmod >= H.REG_OPEN + k)
    return out


def build_lines(tf):
    bars = TF[tf]
    mul = 3 // tf  # 분 단위 맞춤: 3분봉 1봉 = 1분봉 3봉
    lines = {}
    for rmin in (9, 15, 30):
        n = rmin // 3 * mul
        s, _ = pivot_low_confirmed(bars, n, n)
        lines[f"pivot{rmin}m"] = s.shift(1)          # 확정 다음 봉부터
    for rmin in (9, 15):
        n = rmin // 3 * mul
        lines[f"hl{rmin}m"] = higher_low_line(bars, n, n).shift(1)
    for nmin in (30, 60, 120):
        n = nmin // 3 * mul
        lines[f"donch{nmin}m"] = bars["Low"].shift(1).rolling(n, min_periods=n).min()
    lines["rally_anchor"] = rally_anchor_line(bars, 20 * mul)
    lines.update(session_lines(bars, tf))
    return lines


# ───────────── 확인 방식 ─────────────
CONF = ("touch", "c0", "c2", "c5", "atr05")


def cond_below(bars, line, conf, atr_):
    c, lo = bars["Close"], bars["Low"]
    if conf == "touch":
        return lo < line
    if conf == "c0":
        return c < line
    if conf == "c2":
        return c < line * (1 - 0.002)
    if conf == "c5":
        return c < line * (1 - 0.005)
    if conf == "atr05":
        return c < line - 0.5 * atr_
    raise ValueError(conf)


def event_of(cond: pd.Series) -> pd.Series:
    c = cond.fillna(False).astype(bool)
    return c & ~c.shift(1, fill_value=False)


# ───────────── 채점 ─────────────
def score(sig: pd.Series, bars, form: str) -> dict:
    """sig: bars 기준 bool. form='event'면 사건 위치, 'state'면 상태 위치를 hold_sim에 넘김."""
    pos_all = H.exec_index(sig, bars)
    pos_all = pos_all[pos_all >= 0]
    pos_d = H.dedup(pos_all, 30)
    ev = H.event_metrics(pos_d)
    out = {}
    for nm, rng in (("train", H.TRAIN), ("test", H.TEST), ("pre", H.PRE)):
        hs = H.hold_sim(pos_all, rng)
        hr = H.hold_sim(pos_all, rng, entry_mask=RALLY)
        sm = H.summarize(ev, rng, "exit", RALLY)
        out[nm] = {
            "primary": hs["x_rand_mean%"], "primary_t": hs["x_rand_t_days"],
            "rally_primary": hr["x_rand_mean%"], "rally_t": hr["x_rand_t_days"],
            "n": sm.get("n", 0), "per_day": sm.get("per_day", 0.0),
            "r60_dm": sm.get("r60_x_dm%"), "r60_dm_t": sm.get("r60_x_dm_t"),
            "rclose": sm.get("rclose_x_dm%"), "fp_good": sm.get("fp_good%"), "fp_base": sm.get("fp_good_base%"),
            "rule_cvar5": hs["rule_cvar5%"], "rand_cvar5": hs["rand_cvar5%"],
            "exit_rate": hs["exit_rate%"], "hold_min": hs["avg_hold_min"],
            "rally_cvar5": hr["rule_cvar5%"], "rally_rand_cvar5": hr["rand_cvar5%"],
        }
    return {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv) for kk, vv in v.items()}
            for k, v in out.items()}


def ok_freq(r):
    t = r["train"]
    return t["n"] >= 10 and 0.3 <= (t["per_day"] or 0) <= 6


# ───────────── 거르개(계속 빠지는 이탈 가르기) ─────────────
def filters_for(tf):
    bars = TF[tf]
    c = bars["Close"]
    e60_3 = H.ema(b3["Close"], 60)
    e25_3 = H.ema(b3["Close"], 25)
    rsi3 = H.rsi(b3["Close"], 14)

    def ctx3(s):  # 3분 완성봉 상태를 tf 봉으로
        if tf == 3:
            return s.fillna(False).astype(bool)
        m = H.to_1m_mask(s, b3)
        # to_1m_mask는 1분봉 '시작 시각'에 이미 완성된 3분봉 → 1분봉 완성 시점에 알려진 값으로 한 칸 늦춤 없이 사용 가능
        return pd.Series(m, index=bars.index)

    rsi_tf = H.rsi(c, 14)
    K, _ = H.stoch_rsi(c)
    vw = H.session_vwap(bars)
    hi30 = bars["High"].rolling(30 // tf, min_periods=1).max()
    f = {
        "below_e60_3m": ctx3(b3["Close"] < e60_3),
        "e25_lt_e60_3m": ctx3(e25_3 < e60_3),
        "rsi3m_gt40": ctx3(rsi3 > 40),
        "rsi_tf_gt30": (rsi_tf > 30),
        "after_1000": bars["mod"] + tf >= 600,
        "slow_drop_lt2pct": (c >= hi30 * 0.98),          # 30분 고점 대비 2% 미만 하락(투매 직후 아님)
        "stoch_gt20": (K > 20),
        "below_vwap": (c < vw),
        "decline_ctx": ctx3(H.ctx_decline(b3, 40, 0.03)),
    }
    return {k: v.fillna(False).astype(bool) for k, v in f.items()}


def retest_fail(bars, line, ev_break, tf):
    """이탈 사건 뒤 30분 안에 선 근처(고가>=선x0.999)까지 되돌았다가 음봉 종가<선 으로 실패한 봉."""
    W = 30 // tf
    recent = ev_break.astype(int).shift(1, fill_value=0).rolling(W, min_periods=1).max().astype(bool)
    return recent & (bars["High"] >= line * 0.999) & (bars["Close"] < line) & (bars["Close"] < bars["Open"])


def two_bar(cond):
    c = cond.fillna(False).astype(bool)
    both = c & c.shift(1, fill_value=False)
    return both & ~both.shift(1, fill_value=False)


def main():
    res = {"stage1": {}, "stage2": {}, "stage3": {}}
    cache = {}
    for tf in (1, 3):
        bars = TF[tf]
        atr_ = H.atr(bars, 14)
        lines = build_lines(tf)
        for ln, line in lines.items():
            for conf in CONF:
                cond = cond_below(bars, line, conf, atr_)
                ev = event_of(cond)
                for vol in (False, True):
                    sig = ev & (bars["Volume"] > 1.5 * bars["Volume"].rolling(20).mean()) if vol else ev
                    key = f"{tf}m|{ln}|{conf}|{'vol' if vol else 'novol'}|event"
                    res["stage1"][key] = score(sig, bars, "event")
                    cache[key] = (tf, ln, conf, vol)
            key = f"{tf}m|{ln}|c0|novol|state"
            res["stage1"][key] = score(cond_below(bars, line, "c0", atr_).fillna(False), bars, "state")
        print(tf, "stage1 done", len(res["stage1"]), flush=True)

    s1 = {k: v for k, v in res["stage1"].items() if ok_freq(v) and k.endswith("event")}
    top5 = sorted(s1, key=lambda k: -s1[k]["train"]["primary"])[:5]
    res["stage1_top5"] = top5

    def base_signal(key):
        tf, ln, conf, vol = cache[key]
        bars = TF[tf]
        atr_ = H.atr(bars, 14)
        line = build_lines(tf)[ln]
        cond = cond_below(bars, line, conf, atr_)
        ev = event_of(cond)
        if vol:
            ev = ev & (bars["Volume"] > 1.5 * bars["Volume"].rolling(20).mean())
        return tf, bars, line, cond, ev

    for key in top5:
        tf, bars, line, cond, ev = base_signal(key)
        F = filters_for(tf)
        for fn, fm in F.items():
            res["stage2"][f"{key}+{fn}"] = score(ev & fm, bars, "event")
        res["stage2"][f"{key}+two_bar"] = score(two_bar(cond), bars, "event")
        res["stage2"][f"{key}+retest_fail"] = score(retest_fail(bars, line, ev, tf), bars, "event")
        print("stage2", key, flush=True)

    s2 = {k: v for k, v in res["stage2"].items() if ok_freq(v)}
    # 거르개별 TRAIN 평균 개선(기준선 대비) → 상위 3 거르개로 쌍
    gain = {}
    for k, v in s2.items():
        base, fn = k.rsplit("+", 1)
        if fn in ("two_bar", "retest_fail"):
            continue
        gain.setdefault(fn, []).append(v["train"]["primary"] - res["stage1"][base]["train"]["primary"])
    fbest = sorted(gain, key=lambda f: -np.mean(gain[f]))[:3]
    res["filter_gain_train"] = {f: round(float(np.mean(g)), 3) for f, g in gain.items()}
    for key in top5[:3]:
        tf, bars, line, cond, ev = base_signal(key)
        F = filters_for(tf)
        for a, b in combinations(fbest, 2):
            res["stage3"][f"{key}+{a}&{b}"] = score(ev & F[a] & F[b], bars, "event")

    allc = {}
    for st in ("stage1", "stage2", "stage3"):
        allc.update(res[st])
    res["configs_tested"] = len(allc)
    elig = {k: v for k, v in allc.items() if ok_freq(v)}
    top3 = sorted(elig, key=lambda k: -elig[k]["train"]["primary"])[:3]
    res["final_top3"] = {k: allc[k] for k in top3}
    res["baseline_ema60_note"] = "BRIEF: 3m 종가 EMA60 하향 이탈 TRAIN x_rand -0.35 (t -4.5)"
    # 순위 분포(선택편향 공개)
    pr = np.array([v["train"]["primary"] for v in elig.values()])
    res["train_primary_dist"] = {"n_eligible": int(len(pr)), "mean": round(float(pr.mean()), 3),
                                 "p90": round(float(np.quantile(pr, 0.9)), 3), "max": round(float(pr.max()), 3),
                                 "share_positive": round(float((pr > 0).mean()), 3)}
    pt = np.array([elig[k]["test"]["primary"] for k in elig])
    res["train_test_rank_corr"] = round(float(pd.Series(pr).corr(pd.Series(pt), method="spearman")), 3)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    print("configs", len(allc))
    for k in top3:
        print(k, json.dumps(allc[k]["train"]), "\n   TEST", json.dumps(allc[k]["test"]), "\n   PRE", allc[k]["pre"]["primary"])


if __name__ == "__main__":
    main()
