"""독립 검증자 — 바닥(BOTTOM) 후보 4개 재계산·미래정보 감사·견고성 (2026-11-20).

harness.py에서는 자료·지표 함수(load_1m, resample, ema, rsi, rma, atr)만 쓴다.
체결·겹침 제거·앞으로 수익·대조군·날짜 맞춤 대조·first passage·하락상황 마스크는 여기서 새로 짠다.

규약(과제서와 같음)
- 신호 봉 완성 시각(end) 이후 첫 1분봉 시가에 체결, 같은 날짜만.
- 하락상황(DECLINE): 3분봉 Close <= 최근 40개 3분봉 High 최고 x 0.97. 1분봉 시각 t에는 end <= t인 최신 3분봉 값.
- r60: 체결봉 시가 -> 체결 + 60분 직전 1분봉 종가, 16:00(그날 마지막 정규장 봉)에서 자름.
- primary = 신호 r60 평균 - 같은 구간 정규장 & DECLINE 1분봉 전체 r60 평균(%).
- 변형 'h'(harness 관례): 진행 중인 마지막 날의 대조군 봉은 잘린 창 그대로 포함(harness _fwd_tables 동작).
  변형 's'(엄격): 대조군도 60분 창이 자료 끝을 넘으면 뺀다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402  (지표·자료 함수만 사용)

ROOT = H.ROOT
OUT = os.path.join(ROOT, "results", "verify_bottom.json")
TRAIN, TEST, PRE = ("2026-10-15", "2026-11-04"), ("2026-11-05", "2026-11-19"), ("2026-08-04", "2026-10-14")
RANGES = {"TRAIN": TRAIN, "TEST": TEST, "PRE": PRE}

D = H.load_1m()
N = len(D)
T = D.index.values                       # datetime64 (UTC 기준 동일 간격)
O = D["Open"].to_numpy(float)
HI = D["High"].to_numpy(float)
LO = D["Low"].to_numpy(float)
C = D["Close"].to_numpy(float)
DATE = D["date"].values
MOD = D["mod"].to_numpy()
REG = (MOD >= 570) & (MOD < 960)


# ───────────── 앞으로 경로(독립 구현) ─────────────
def _last_reg_idx():
    lr = np.full(N, -1)
    codes, uniq = pd.factorize(DATE)
    for c in range(len(uniq)):
        idx = np.nonzero((codes == c) & REG)[0]
        if len(idx):
            lr[codes == c] = idx.max()
    return lr


LAST_REG = _last_reg_idx()
LAST_T = T[-1]
LAST_DATE = DATE[-1]
SESSION_OPEN = MOD[-1] < 960


def fwd(h=60):
    """체결봉 i 시가 기준 h분 수익(16:00에서 자름). 반환 (trunc, strict, jidx)."""
    tgt = T + np.timedelta64(h, "m")
    j = np.searchsorted(T, tgt, side="left") - 1
    j = np.where(LAST_REG >= 0, np.minimum(j, LAST_REG), j)
    r = np.where(LAST_REG >= 0, C[np.clip(j, 0, N - 1)] / O - 1, np.nan)
    strict = r.copy()
    # 진행 중인 날: 60분 창이 자료 끝을 넘으면 비움(harness의 사건 처리와 같은 문턱: t+h > 마지막 봉 시작)
    inc = (tgt > LAST_T) & (DATE == LAST_DATE) & SESSION_OPEN
    strict[inc] = np.nan
    return r, strict, j


R60, R60S, J60 = fwd(60)
RCL = np.where(LAST_REG >= 0, C[np.clip(LAST_REG, 0, N - 1)] / O - 1, np.nan)
RCLS = RCL.copy()
if SESSION_OPEN:
    RCLS[DATE == LAST_DATE] = np.nan


def mae60(i):
    j = J60[i]
    if j < i or np.isnan(R60S[i]):   # 진행 중인 날 미완 창은 비움(harness 사건 처리와 같음)
        return np.nan, np.nan
    return LO[i:j + 1].min() / O[i] - 1, HI[i:j + 1].max() / O[i] - 1


def fpass(i, up=0.015, dn=0.015):
    e = LAST_REG[i]
    if e < i:
        return np.nan
    a = np.nonzero(HI[i:e + 1] >= O[i] * (1 + up))[0]
    b = np.nonzero(LO[i:e + 1] <= O[i] * (1 - dn))[0]
    a = a[0] if len(a) else 10**9
    b = b[0] if len(b) else 10**9
    if a == b == 10**9:
        return np.nan
    return 1.0 if a < b else (0.0 if b < a else 0.5)


def in_rng(dates, rng):
    return (dates >= np.datetime64(rng[0])) & (dates <= np.datetime64(rng[1]))


# ───────────── 체결·겹침 제거·상황 마스크(독립 구현) ─────────────
def fill_index(sig: pd.Series, bars: pd.DataFrame) -> np.ndarray:
    """신호 봉 완성 시각 이후 첫 1분봉(같은 날짜) 정수 위치."""
    s = sig.reindex(bars.index).fillna(False).to_numpy(bool)
    ends = bars["end"].values[s]
    bd = bars["date"].values[s]
    p = np.searchsorted(T, ends, side="left")
    ok = p < N
    p = np.where(ok, p, 0)
    ok &= DATE[p] == bd
    return np.unique(p[ok])


def my_dedup(pos, minutes=30):
    keep, lt, ld = [], None, None
    for p in np.sort(pos):
        if lt is not None and DATE[p] == ld and (T[p] - lt) < np.timedelta64(minutes, "m"):
            continue
        keep.append(p)
        lt, ld = T[p], DATE[p]
    return np.array(keep, dtype=int)


B3 = H.resample(3)
B1 = H.resample(1)


def htf_state_on_1m(vals: np.ndarray, bars: pd.DataFrame) -> np.ndarray:
    """1분봉 시작 시각 t에 end <= t인 최신 상위봉 값."""
    k = np.searchsorted(bars["end"].values, T, side="right") - 1
    v = np.asarray(vals)
    fillv = np.nan if v.dtype.kind == "f" else False
    return np.where(k >= 0, v[np.clip(k, 0, len(v) - 1)], fillv)


def decline_mask(look=40, pct=0.03):
    hi = B3["High"].rolling(look, min_periods=1).max()
    st = (B3["Close"] <= hi * (1 - pct)).to_numpy()
    return htf_state_on_1m(st, B3).astype(bool)


DECL = decline_mask()

_BASE_CACHE = {}


def baseline(rng):
    if rng in _BASE_CACHE:
        return _BASE_CACHE[rng]
    idx = np.nonzero(REG & in_rng(DATE, rng) & DECL)[0]
    allreg = np.nonzero(REG & in_rng(DATE, rng))[0]
    fpb = np.array([fpass(i) for i in allreg[::7]])     # harness와 같은 7봉 간격 표본(전체 정규장 봉)
    res = {"r60_h": float(np.nanmean(R60[idx])), "r60_s": float(np.nanmean(R60S[idx])),
           "rcl_h": float(np.nanmean(RCL[idx])), "rcl_s": float(np.nanmean(RCLS[idx])),
           "fp_base": float(100 * np.nanmean(fpb)), "ndays": int(len(np.unique(DATE[allreg]))), "n_ctx": int(len(idx))}
    _BASE_CACHE[rng] = res
    return res


def day_matched(p, arr, w=30):
    lo = np.searchsorted(T, T[p] - np.timedelta64(w, "m"), side="left")
    hi = np.searchsorted(T, T[p] + np.timedelta64(w, "m"), side="right")
    idx = np.arange(lo, hi)
    idx = idx[REG[idx] & (DATE[idx] == DATE[p])]
    return np.nanmean(arr[idx]) if len(idx) else np.nan


def score(pos, rng, cost=0.0, events=False):
    p = pos[in_rng(DATE[pos], rng) & REG[pos]] if len(pos) else pos
    b = baseline(rng)
    out = {"n": int(len(p)), "days": int(len(np.unique(DATE[p]))), "per_day": round(len(p) / b["ndays"], 2)}
    if not len(p):
        return out
    # 매수 체결가를 cost만큼 불리하게: (1+r)/(1+cost)-1
    r60 = (1 + R60S[p]) / (1 + cost) - 1
    rcl = (1 + RCLS[p]) / (1 + cost) - 1
    out["r60_mean%"] = round(100 * np.nanmean(r60), 3)
    out["r60_med%"] = round(100 * np.nanmedian(r60), 3)
    out["primary_h"] = round(100 * (np.nanmean(r60) - b["r60_h"]), 3)
    out["primary_s"] = round(100 * (np.nanmean(r60) - b["r60_s"]), 3)
    out["rclose_x_ctx_h"] = round(100 * (np.nanmean(rcl) - b["rcl_h"]), 3)
    out["rclose_x_ctx_s"] = round(100 * (np.nanmean(rcl) - b["rcl_s"]), 3)
    dm = np.array([day_matched(i, R60) for i in p])
    xd = r60 - dm
    out["r60_x_dm%"] = round(100 * np.nanmean(xd), 3)
    g = pd.Series(xd, index=DATE[p]).groupby(level=0).mean().dropna()
    out["r60_x_dm_t"] = round(float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))), 2) if len(g) > 2 and g.std(ddof=1) > 0 else None
    mm = np.array([mae60(i) for i in p])
    out["mae60_med%"] = round(100 * np.nanmedian(mm[:, 0]), 3)
    out["mfe60_med%"] = round(100 * np.nanmedian(mm[:, 1]), 3)
    fp = np.array([fpass(i) for i in p])
    out["fp_good%"] = round(100 * np.nanmean(fp), 1)
    out["fp_base%"] = round(b["fp_base"], 1)
    out["in_decline%"] = round(100 * DECL[p].mean(), 1)
    if events:
        out["events"] = [f"{pd.Timestamp(D.index[i]).strftime('%m-%d %H:%M')} r60={100 * R60S[i]:+.2f}" for i in p]
    return out


def run(sig, bars, gate=True, reg_first=True, dd=30, gate_mask=None):
    gm = DECL if gate_mask is None else gate_mask
    p = fill_index(sig, bars)
    if gate:
        p = p[gm[p]]
    if reg_first:
        p = p[REG[p]]
    return my_dedup(p, dd)


def full(pos, cost=0.0, events=False):
    return {k: score(pos, rng, cost, events and k != "PRE") for k, rng in RANGES.items()}


def brief(res):
    """이웃 표용 요약."""
    return {k: {"n": v.get("n"), "primary": v.get("primary_h"), "dm_t": v.get("r60_x_dm_t")} for k, v in res.items()}


def harness_rclose_bug(pos, rng):
    """harness.event_metrics는 rows['date'].max()(사건 중 최신 날짜)의 rclose를 비운다. 사건이 11-19에 없으면
    진짜 마감된 날(예: 11-18)의 rclose가 빠진다. 버그 재현값과 올바른 값을 나란히 적는다."""
    p = pos[in_rng(DATE[pos], rng) & REG[pos]]
    b = baseline(rng)
    allpos = pos[REG[pos]]
    last_ev_day = DATE[allpos].max()
    bug = RCLS[p].copy()
    if SESSION_OPEN:
        bug[DATE[p] == last_ev_day] = np.nan
    return {"last_event_day": str(pd.Timestamp(last_ev_day).date()),
            "rclose_x_ctx_bug_emulated": round(100 * (np.nanmean(bug) - b["rcl_h"]), 3),
            "rclose_x_ctx_correct": round(100 * (np.nanmean(RCLS[p]) - b["rcl_h"]), 3)}


def lodo(pos, rng):
    """날짜 하나씩 빼고 primary_h."""
    p = pos[in_rng(DATE[pos], rng) & REG[pos]]
    b = baseline(rng)
    out = {}
    for dt in np.unique(DATE[p]):
        q = p[DATE[p] != dt]
        out[str(pd.Timestamp(dt).date())] = round(100 * (np.nanmean(R60S[q]) - b["r60_h"]), 3) if len(q) else None
    return out


# ───────────── 후보 신호 생성기 ─────────────
def entry_cross(zone: pd.Series) -> pd.Series:
    z = zone.fillna(False).astype(bool)
    return z & ~z.shift(1, fill_value=False)


def sig_dayhigh(bars, x):
    dh = bars["High"].groupby(bars["date"]).cummax()
    return entry_cross(bars["Close"] <= dh * (1 - x / 100))


def sig_ema(bars, x, n=60):
    e = H.ema(bars["Close"], n)
    return entry_cross(bars["Close"] <= e * (1 - x / 100))


RSI1 = H.rsi(B1["Close"], 14)
RSI3 = H.rsi(B3["Close"], 14)


def crossover(a, lvl):
    return (a > lvl) & (a.shift(1) <= lvl)


def sig_d1(xl=30, th=25, n1=14, n3=14):
    r1 = RSI1 if n1 == 14 else H.rsi(B1["Close"], n1)
    r3 = RSI3 if n3 == 14 else H.rsi(B3["Close"], n3)
    r3c = pd.Series(htf_state_on_1m(r3.to_numpy(float), B3), index=B1.index)
    return crossover(r1, xl) & (r3c < th)


def sig_d1_swapped(xl=30, th=25):
    """1분<->3분 바꿈: 3분 RSI가 xl 상향 교차하는 3분봉 완성 시점, 그 3분봉 칸 안 1분 RSI 최저 < th.
    (칸 마지막 1분 RSI < th 조건은 3분 RSI 상향 교차와 동시에 성립한 적이 0건이라 칸 안 최저로 바꿨다.)"""
    r1min = RSI1.groupby(B1.index.floor("3min")).min()
    r1min = r1min.reindex(B3.index)
    return crossover(RSI3, xl) & (r1min < th)


def sig_rsi3_only(xl=30):
    """3분 RSI 30 상향 교차만(1분 조건 없음)."""
    return crossover(RSI3, xl)


def sig_e(bars, L=25, J=10, W=20, n=14):
    r = H.rsi(bars["Close"], n)
    m = r.rolling(W).min()
    c = (m < L) & (r - m >= J)
    return c & ~c.shift(1, fill_value=False)


# ───────────── 실행 ─────────────
def main():
    out = {"verifier": "verify_bottom", "question": "bottom", "data_last_bar": str(D.index[-1]),
           "baseline_decline": {k: baseline(rng) for k, rng in RANGES.items()},
           "conventions": "primary_h = harness 관례(진행 중 11-19 대조군 잘린 창 포함), primary_s = 엄격(대조군도 60분 미완 창 제외). "
                          "비용은 매수 체결가 x(1+0.001). 이웃 표의 primary는 primary_h."}
    CC = {}

    # 1) 3m 장중고점 -4% 구역 진입
    s1 = sig_dayhigh(B3, 4)
    c1 = {"recompute": full(run(s1, B3), events=True),
          "order_note": "갈래도 정규장·DECLINE 거른 뒤 dedup(정의와 같음)",
          "cost_0.1%": brief(full(run(s1, B3), cost=0.001)), "neighbors": {}}
    for x in (3, 5, 6):
        c1["neighbors"][f"x={x}%"] = brief(full(run(sig_dayhigh(B3, x), B3)))
    for look, pct in ((30, 0.03), (50, 0.03), (40, 0.02), (40, 0.04)):
        c1["neighbors"][f"gate {look}봉 -{int(pct * 100)}%"] = brief(full(run(s1, B3, gate_mask=decline_mask(look, pct))))
    c1["neighbors"]["gate 없음"] = brief(full(run(s1, B3, gate=False)))
    for dd in (15, 45, 60):
        c1["neighbors"][f"dedup {dd}분"] = brief(full(run(s1, B3, dd=dd)))
    c1["neighbors"]["1m로 바꿈(x=4%)"] = brief(full(run(sig_dayhigh(B1, 4), B1)))
    c1["test_leave_one_day_out"] = lodo(run(s1, B3), TEST)
    CC["3m 장중고점 -4% 구역 진입"] = c1

    # 2) 1m EMA60 -3% 구역 진입
    s2 = sig_ema(B1, 3)
    c2 = {"recompute": full(run(s2, B1), events=True),
          "cost_0.1%": brief(full(run(s2, B1), cost=0.001)), "neighbors": {}}
    for x in (2.5, 3.5, 1.5, 4.5):
        c2["neighbors"][f"x={x}%"] = brief(full(run(sig_ema(B1, x), B1)))
    for n in (50, 70):
        c2["neighbors"][f"EMA{n}"] = brief(full(run(sig_ema(B1, 3, n), B1)))
    for look, pct in ((30, 0.03), (50, 0.03), (40, 0.02), (40, 0.04)):
        c2["neighbors"][f"gate {look}봉 -{int(pct * 100)}%"] = brief(full(run(s2, B1, gate_mask=decline_mask(look, pct))))
    c2["neighbors"]["gate 없음"] = brief(full(run(s2, B1, gate=False)))
    for dd in (15, 45):
        c2["neighbors"][f"dedup {dd}분"] = brief(full(run(s2, B1, dd=dd)))
    c2["neighbors"]["3m로 바꿈(EMA60 -3%)"] = brief(full(run(sig_ema(B3, 3), B3)))
    CC["1m EMA60 -3% 구역 진입"] = c2

    # 3) D1 rsi1 X30 & rsi3<25 (갈래: 조건 없음, 연장시간 포함 dedup 뒤 정규장만 평가)
    s3 = sig_d1()
    p3 = run(s3, B1, gate=False, reg_first=False)
    c3 = {"recompute": full(p3, events=True),
          "decline_gate_branch_order": brief(full(run(s3, B1, gate=True, reg_first=False))),
          "reg_first_order": brief(full(run(s3, B1, gate=False, reg_first=True))),
          "cost_0.1%": brief(full(p3, cost=0.001)), "neighbors": {}}
    for xl in (25, 35):
        c3["neighbors"][f"1m 교차선 {xl}"] = brief(full(run(sig_d1(xl=xl), B1, gate=False, reg_first=False)))
    for th in (20, 30):
        c3["neighbors"][f"3m RSI<{th}"] = brief(full(run(sig_d1(th=th), B1, gate=False, reg_first=False)))
    for n in (12, 16):
        c3["neighbors"][f"RSI 길이 {n}(둘 다)"] = brief(full(run(sig_d1(n1=n, n3=n), B1, gate=False, reg_first=False)))
    for dd in (15, 45):
        c3["neighbors"][f"dedup {dd}분"] = brief(full(run(s3, B1, gate=False, reg_first=False, dd=dd)))
    c3["neighbors"]["1m<->3m 바꿈(3m RSI X30 & 칸 안 1m RSI 최저<25)"] = brief(full(run(sig_d1_swapped(), B3, gate=False, reg_first=False)))
    c3["neighbors"]["3m RSI X30만"] = brief(full(run(sig_rsi3_only(), B3, gate=False, reg_first=False)))
    # harness event_metrics 버그 재현: rclose를 '사건 중 가장 늦은 날짜'에서 비움 -> 11-18 사건 rclose가 빠짐
    c3["rclose_harness_bug_check"] = harness_rclose_bug(p3, TEST)
    CC["D1_rsi1X30_rsi3lt25"] = c3

    # 4) E rsiMin25 reb10 3m | DECLINE (갈래: DECLINE 거른 뒤 연장시간 포함 dedup)
    s4 = sig_e(B3)
    p4 = run(s4, B3, gate=True, reg_first=False)
    c4 = {"recompute": full(p4, events=True),
          "reg_first_order": brief(full(run(s4, B3, gate=True, reg_first=True))),
          "cost_0.1%": brief(full(p4, cost=0.001)), "neighbors": {},
          "rclose_harness_bug_check": harness_rclose_bug(p4, TEST)}
    for L in (20, 30):
        c4["neighbors"][f"L={L}"] = brief(full(run(sig_e(B3, L=L), B3, reg_first=False)))
    for J in (5, 15):
        c4["neighbors"][f"J={J}"] = brief(full(run(sig_e(B3, J=J), B3, reg_first=False)))
    for W in (15, 25):
        c4["neighbors"][f"W={W}"] = brief(full(run(sig_e(B3, W=W), B3, reg_first=False)))
    for look, pct in ((30, 0.03), (50, 0.03), (40, 0.02), (40, 0.04)):
        c4["neighbors"][f"gate {look}봉 -{int(pct * 100)}%"] = brief(full(run(s4, B3, reg_first=False, gate_mask=decline_mask(look, pct))))
    c4["neighbors"]["gate 없음"] = brief(full(run(s4, B3, gate=False, reg_first=False)))
    c4["neighbors"]["1m로 바꿈"] = brief(full(run(sig_e(B1), B1, reg_first=False)))
    CC["E_rsiMin25_reb10_3m|DECLINE"] = c4

    out["candidates"] = CC
    out["harness_issues"] = [
        "event_metrics: rows['date'].max()(사건 중 최신 날짜)의 rclose를 비운다. 사건이 진행 중인 11-19에 없으면 마감된 11-18 사건 rclose가 빠진다. "
        "D1 TEST rclose_x_ctx -4.611 -> 바르게는 -3.540, E TEST -8.432 -> -6.545. primary(r60)에는 영향 없음.",
        "_fwd_tables: 진행 중인 11-19 대조군 봉의 r60·rclose를 자료 끝(10:36)에서 잘린 창 그대로 쓴다(사건은 비움). "
        "TEST 대조군 r60 -0.391% vs 엄격 -0.344%, primary 차이 약 0.05%p(primary_h vs primary_s).",
        "dedup 순서가 갈래마다 다르다(bot_dev는 정규장·DECLINE 거른 뒤, bot_osc는 연장시간 포함 거른 뒤). 이번 네 후보는 TRAIN·TEST 사건이 두 순서에서 같다.",
    ]
    out["lookahead_audit"] = {
        "bot_dev_scan.py": "ddDay = 3분봉 High를 날짜별 cummax(완성봉까지, 04:00 재시작) -> 실시간. EMA60·ATR·VWAP σ는 과거만. ddPC는 전일 정규장 종가. "
                           "DECLINE은 체결 1분봉 시작 시각까지 완성된 3분봉. 피벗·시가 범위·center 창·하루 전체 값 없음. "
                           "explain()의 지그재그 바닥과 '칸 안 1분봉에 그 3분봉 완성값'을 붙이는 부분은 사후값이지만 설명용이고 규칙 선택에 안 쓰인다. 위반 없음.",
        "bot_osc_run.py": "htf_on_1m은 1분봉 시작 시각에 end<=t인 3분봉 값 = Pine request.security(rsi[1], lookahead_on)와 같아 미래 없음. "
                          "rsi_rebound는 뒤쪽 rolling. 다이버전스 피벗은 확정봉 q에서만 신호(후보 아님). location_desc는 cmh[p-1](직전 봉까지 누적 고가). 위반 없음.",
    }
    out["verdicts"] = {
        "3m 장중고점 -4% 구역 진입": {"verdict": "fragile", "recompute_match": True,
            "reason": "재계산 일치(TRAIN 2.601, TEST 0.374), 미래정보 없음. TEST는 대조보다 +0.37%p(dm t 2.07, 비용 0.1% 뒤 +0.27)로 나으나 "
                      "r60 평균 -0.02%·중앙 -0.89%·fp_good 46.2<48.9이고 11-12 하루를 빼면 -1.41로 뒤집힌다. x=5% 이웃 TEST -0.47, x=6% -1.98, 1분봉 바꿈 -0.37, PRE -1.29. 게이트·dedup 이웃은 안정."},
        "1m EMA60 -3% 구역 진입": {"verdict": "reject", "recompute_match": True,
            "reason": "재계산 일치, 미래정보 없음. TEST -1.59(이웃 2.5% -1.93, 3.5% -0.75, EMA50/70 -0.74/-1.96 모두 음수), TRAIN n=10으로 얇음, 3m 바꿈 TRAIN부터 -0.26."},
        "D1_rsi1X30_rsi3lt25": {"verdict": "reject", "recompute_match": True,
            "reason": "재계산 일치, 미래정보 없음. TEST -3.66(11-05 09:33 -15.3%, 11-13 09:47 -12.3%), 모든 이웃 TEST 음수(-1.54~-4.31), PRE -0.33. 장 초반 급락일 칼날 잡기."},
        "E_rsiMin25_reb10_3m|DECLINE": {"verdict": "reject", "recompute_match": True,
            "reason": "재계산 일치, 미래정보 없음. TRAIN부터 dm t 0.47로 약하고 TEST -3.85, fp_good 16.7. 모든 이웃 TEST 음수, PRE -0.16."},
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("BASE", out["baseline_decline"])
    for nm, c in CC.items():
        print("==", nm)
        for k in ("TRAIN", "TEST", "PRE"):
            r = c["recompute"][k]
            print(" ", k, {kk: r.get(kk) for kk in ("n", "per_day", "primary_h", "primary_s", "r60_x_dm%", "r60_x_dm_t",
                                                    "rclose_x_ctx_h", "mae60_med%", "mfe60_med%", "fp_good%", "fp_base%", "in_decline%")})
            if "events" in r and k == "TEST":
                print("   ev", r["events"])
        for k, v in c.items():
            if k not in ("recompute", "neighbors"):
                print("  ", k, v)
        for k, v in c["neighbors"].items():
            print("   nb", k, v)


if __name__ == "__main__":
    main()
