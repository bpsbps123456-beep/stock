"""독립 검증자: SOXL 긴급탈출(EXIT) 후보 4종 재계산·미래정보 감사·견고성 (2026-11-20).

실행: py code/verify_exit.py  (작업 폴더 soxl_escape_20261120)
결과: results/verify_exit.json

harness.py에서는 지표 함수(ema, rsi, resample)만 쓴다. 체결 위치·앞으로 수익·보유 모의·사건 성적·
같은 날 대조·빠른 판정(first passage)은 여기서 다시 짠다.

primary = 보유 모의: 정규장 5분마다 가상 진입 -> 진입 다음 봉부터 첫 탈출 체결(그 1분봉 시가) 또는 16:00 정규장 마지막 봉 종가.
대조 = 같은 날 안에서 보유시간(봉 수)을 섞은 무작위 탈출 20회 평균(seed 7, 과제 정의 그대로) -> 규칙 - 대조 평균(%).
보조 대조 2종(검증자 추가):
  exact = 섞기의 정확한 기댓값(같은 날 모든 보유시간의 평균) - 난수 잡음 없음.
  null  = 같은 날 같은 개수의 무작위 정규장 1분봉 사건을 탈출로 넣은 보유 모의(30회)의 primary 평균을 뺀 값.
          (보유시간 섞기 대조는 16:00에서 자르므로 드문 규칙일수록 음으로 치우친다 - 갈래 exit_struct가 찾은 편향을 독립 확인.)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402  (지표·상위봉만 사용)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "results", "verify_exit.json")
TRAIN, TEST, PRE = ("2026-10-15", "2026-11-04"), ("2026-11-05", "2026-11-19"), ("2026-08-04", "2026-10-14")
RANGES = (("train", TRAIN), ("test", TEST), ("pre", PRE))
OPEN_M, CLOSE_M = 570, 960

# ───────────── 자료(직접 적재) ─────────────
raw = pd.read_parquet(os.path.join(ROOT, "cache", "SOXL_1m.parquet"))[["Open", "High", "Low", "Close", "Volume"]]
raw = raw[~raw.index.duplicated()].sort_index()
D = raw.copy()
D["date"] = pd.to_datetime(D.index.date)
D["mod"] = D.index.hour * 60 + D.index.minute
D["reg"] = (D["mod"] >= OPEN_M) & (D["mod"] < CLOSE_M)
N = len(D)
T = D.index.tz_localize(None).values.astype("datetime64[m]")   # 벽시계(ET) 분
O, Hh, Lo, C = (D[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
DATES = D["date"].values
MOD = D["mod"].to_numpy()
REG = D["reg"].to_numpy()
SESSION_OPEN = (MOD[-1] < CLOSE_M)          # 마지막 날(11-19) 진행 중
LAST_DATE = DATES[-1]

_lr = pd.Series(np.arange(N))[REG].groupby(DATES[REG]).max()
LAST_REG = pd.Series(DATES).map(pd.Series(_lr.values, index=pd.to_datetime(_lr.index))).fillna(-1).astype(int).to_numpy()


def in_rng(dates, rng):
    return (dates >= np.datetime64(rng[0])) & (dates <= np.datetime64(rng[1]))


# ───────────── 체결: 봉 완성 시각 이후 첫 1분봉(같은 날) ─────────────
def fill_pos(sig: pd.Series, bars: pd.DataFrame, minutes: int) -> np.ndarray:
    s = sig.fillna(False).astype(bool).to_numpy()
    start = bars.index.tz_localize(None).values.astype("datetime64[m]")[s]
    end = start + np.timedelta64(minutes, "m")
    p = np.searchsorted(T, end, side="left")
    sd = pd.to_datetime(bars.index[s].date).values
    ok = p < N
    p2 = np.where(ok, p, 0)
    ok &= DATES[p2] == sd
    return np.unique(p[ok])


def dedup30(pos):
    keep, lt, ld = [], None, None
    for p in np.sort(pos):
        if lt is not None and DATES[p] == ld and (T[p] - lt) < np.timedelta64(30, "m"):
            continue
        keep.append(p)
        lt, ld = T[p], DATES[p]
    return np.array(keep, dtype=int)


# ───────────── 앞으로 수익 ─────────────
def fwd_ret(h):
    j = np.searchsorted(T, T + np.timedelta64(h, "m"), side="left") - 1
    j = np.minimum(j, np.where(LAST_REG >= 0, LAST_REG, j))
    return np.where(LAST_REG >= 0, C[np.clip(j, 0, N - 1)] / O - 1, np.nan)


R60 = fwd_ret(60)
RCLOSE = np.where(LAST_REG >= 0, C[np.clip(LAST_REG, 0, N - 1)] / O - 1, np.nan)


def fpass(i, up=0.015, dn=0.015):
    e = LAST_REG[i]
    if e < i:
        return np.nan
    P = O[i]
    a = np.nonzero(Hh[i:e + 1] >= P * (1 + up))[0]
    b = np.nonzero(Lo[i:e + 1] <= P * (1 - dn))[0]
    a = a[0] if len(a) else 10**9
    b = b[0] if len(b) else 10**9
    if a == b == 10**9:
        return np.nan
    return 1.0 if a < b else (0.0 if b < a else 0.5)


_FPB = {}


def fp_base(rng):
    if rng not in _FPB:
        base = np.nonzero(REG & in_rng(DATES, rng))[0][::7]
        _FPB[rng] = round(100 * np.nanmean([1 - fpass(i) for i in base]), 1)
    return _FPB[rng]


def event_stats(pos_raw, rng):
    """사건(30분 겹침 제거) 성적: n, 하루 횟수, r60·rclose 같은 날 ±30분 대조 차(탈출 이익 = -r), fp_good."""
    pos = dedup30(pos_raw)
    pos = pos[REG[pos] & in_rng(DATES[pos], rng)]
    ndays = len(np.unique(DATES[REG & in_rng(DATES, rng)]))
    out = {"n": int(len(pos)), "per_day": round(len(pos) / max(ndays, 1), 2)}
    if not len(pos):
        return out
    avail = T[-1]
    for nm, arr in (("r60_dm", R60), ("rclose_dm", RCLOSE)):
        v = -arr[pos].copy()
        if SESSION_OPEN:
            if nm == "r60_dm":
                v[(T[pos] + np.timedelta64(60, "m")) > avail] = np.nan
            else:
                v[DATES[pos] == LAST_DATE] = np.nan
        dm = np.full(len(pos), np.nan)
        for k, p in enumerate(pos):
            lo = np.searchsorted(T, T[p] - np.timedelta64(30, "m"))
            hi = np.searchsorted(T, T[p] + np.timedelta64(30, "m"), side="right")
            idx = np.arange(lo, hi)
            idx = idx[REG[idx] & (DATES[idx] == DATES[p])]
            dm[k] = np.nanmean(-arr[idx])
        out[nm] = round(100 * float(np.nanmean(v - dm)), 3)
    fp = np.array([fpass(p) for p in pos])
    out["fp_good"] = round(100 * float(np.nanmean(1 - fp)), 1)
    out["fp_base"] = fp_base(rng)
    return out


# ───────────── 보유 모의(독립 구현) ─────────────
_ENT = {}


def entries(rng, mask_key=None, mask=None, step=5):
    key = (rng, mask_key)
    if key in _ENT:
        return _ENT[key]
    sel = REG & in_rng(DATES, rng) & ((MOD - OPEN_M) % step == 0)
    if mask is not None:
        sel &= mask
    e = np.nonzero(sel)[0]
    e = e[e < LAST_REG[e]]
    if SESSION_OPEN:
        e = e[DATES[e] != LAST_DATE]
    _ENT[key] = e
    return e


def cvar5(x):
    c = np.nanquantile(x, 0.05)
    return float(np.nanmean(x[x <= c]))


def tday(x, dates):
    g = pd.Series(x).groupby(dates).mean()
    return round(float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))), 2) if len(g) > 2 and g.std(ddof=1) > 0 else None


def holdsim(exit_pos, rng, mask_key=None, mask=None, cost=0.0, n_perm=20, seed=7, detail=True):
    ent = entries(rng, mask_key, mask)
    xs = np.sort(np.asarray(exit_pos))
    k = np.searchsorted(xs, ent, side="right")          # ent보다 큰 첫 탈출 = 진입 다음 봉부터
    xs_pad = np.r_[xs, 10**12].astype(np.int64)
    ex = xs_pad[k]
    lr = LAST_REG[ent]
    use = ex <= lr
    px = np.where(use, O[np.minimum(ex, N - 1)] * (1 - cost), C[lr])
    pr_rule = px / O[ent] - 1
    dur = np.where(use, ex - ent, lr - ent + 1)
    dts = DATES[ent]
    g = np.random.default_rng(seed)
    acc = []
    for _ in range(n_perm):
        dd = dur.copy()
        for dt in np.unique(dts):
            m = np.nonzero(dts == dt)[0]
            dd[m] = g.permutation(dd[m])
        e2 = ent + dd
        acc.append(np.where(e2 <= lr, O[np.minimum(e2, N - 1)], C[lr]) / O[ent] - 1)
    pr_rand = np.mean(acc, axis=0)
    pr_exact = np.empty(len(ent))
    for dt in np.unique(dts):
        m = np.nonzero(dts == dt)[0]
        e2 = ent[m][:, None] + dur[m][None, :]
        L2 = np.broadcast_to(lr[m][:, None], e2.shape)
        v = np.where(e2 <= L2, O[np.minimum(e2, N - 1)], C[L2]) / O[ent[m]][:, None] - 1
        pr_exact[m] = v.mean(axis=1)
    x = pr_rule - pr_rand
    out = {"primary": round(100 * float(np.mean(x)), 3), "primary_t": tday(x, dts),
           "primary_exact": round(100 * float(np.mean(pr_rule - pr_exact)), 3)}
    if detail:
        out.update({"entries": int(len(ent)), "exit_rate": round(100 * use.mean(), 1),
                    "hold_min": round(float(dur.mean()), 1), "rule_mean": round(100 * pr_rule.mean(), 3),
                    "rand_mean": round(100 * pr_rand.mean(), 3),
                    "rule_cvar5": round(100 * cvar5(pr_rule), 3), "rand_cvar5": round(100 * cvar5(pr_rand), 3)})
    return out


REGPOS = np.nonzero(REG)[0]


def null_adj(pos_raw, rng, actual, ndraw=30, seed=11):
    p = pos_raw[REG[pos_raw] & in_rng(DATES[pos_raw], rng)]
    cnt = pd.Series(DATES[p]).value_counts()
    g = np.random.default_rng(seed)
    vals = []
    for _ in range(ndraw):
        dr = [g.choice(REGPOS[DATES[REGPOS] == dt], size=int(c), replace=False) for dt, c in cnt.items()]
        dp = np.sort(np.concatenate(dr)) if dr else np.array([], dtype=int)
        vals.append(holdsim(dp, rng, detail=False)["primary"])
    vals = np.array(vals)
    return {"null_mean": round(float(vals.mean()), 3), "null_adj": round(float(actual - vals.mean()), 3),
            "null_pct": round(float(100 * (vals < actual).mean()), 1)}


# ───────────── 지표(직접) ─────────────
B1 = H.resample(1)
B3 = H.resample(3)
assert len(B1) == N


def pivlow_line_higher(low: np.ndarray, L: int, R: int, strict_right=False):
    """ta.pivotlow(low,L,R): 피벗 j는 왼쪽 L봉보다 엄격히 낮고 오른쪽 R봉 이하(strict_right면 엄격히 낮음).
    j+R봉에서 확정. 확정 때 새 피벗 > 직전 피벗이면 hl=새 피벗, 아니면 na. 반환은 hl[1](확정 다음 봉부터)."""
    n = len(low)
    s = pd.Series(low)
    lmin = s.shift(1).rolling(L, min_periods=L).min().to_numpy()                       # j-L..j-1
    rmin = s[::-1].shift(1).rolling(R, min_periods=R).min()[::-1].to_numpy()           # j+1..j+R
    with np.errstate(invalid="ignore"):
        okr = (low < rmin) if strict_right else (low <= rmin)
        isp = (low < lmin) & okr
    isp[:L] = False
    isp[n - R:] = False
    conf = np.full(n, np.nan)
    pj = np.nonzero(isp)[0]
    conf[pj + R] = low[pj]
    prev, cur = np.nan, np.nan
    # 순차 갱신(확정 봉에서만 값이 바뀜)
    idx = np.nonzero(~np.isnan(conf))[0]
    vals = np.full(len(idx), np.nan)
    for q, i in enumerate(idx):
        p = conf[i]
        cur = p if (not np.isnan(prev) and p > prev) else np.nan
        prev = p
        vals[q] = cur
    seg = np.searchsorted(idx, np.arange(n), side="right") - 1
    hl = np.where(seg >= 0, vals[np.clip(seg, 0, max(len(vals) - 1, 0))], np.nan)
    return np.r_[np.nan, hl[:-1]]


def map3_prev_complete(s3: pd.Series) -> np.ndarray:
    """1분봉 시작 시각 t에 이미 완성된(끝 <= t) 최신 3분봉 값 = request.security('3', x[1], lookahead_on)."""
    ends = B3.index.tz_localize(None).values.astype("datetime64[m]") + np.timedelta64(3, "m")
    k = np.searchsorted(ends, T, side="right") - 1
    v = s3.to_numpy(float)
    return np.where(k >= 0, v[np.clip(k, 0, len(v) - 1)], np.nan)


def first_touch(cond: np.ndarray) -> np.ndarray:
    c = np.asarray(cond, dtype=bool)
    return c & ~np.r_[False, c[:-1]]


_HLC = {}


def hl_line(low, L, tf, strict_right=False):
    k = (tf, L, strict_right)
    if k not in _HLC:
        _HLC[k] = pivlow_line_higher(low, L, L, strict_right)
    return _HLC[k]


def sig_struct(kind, tf=1, L=9, fast=25, slow=60, rthr=30, win=30, drop=0.02, r3thr=40, conf="touch", strict_right=False):
    bars = B1 if tf == 1 else B3
    low, close, high = (bars[c].to_numpy(float) for c in ("Low", "Close", "High"))
    line = hl_line(low, L, tf, strict_right)
    with np.errstate(invalid="ignore"):
        cond = (low < line) if conf == "touch" else (close < line)
    ev = first_touch(cond)
    with np.errstate(invalid="ignore"):
        if kind == "ema_rsi":
            e_f, e_s = H.ema(B3["Close"], fast), H.ema(B3["Close"], slow)
            f3 = ((e_f < e_s) & e_f.notna() & e_s.notna()).astype(float)
            f1 = (map3_prev_complete(f3) == 1) if tf == 1 else (f3.to_numpy() == 1)
            r = H.rsi(bars["Close"], 14).to_numpy(float)
            filt = f1 & (r > rthr)
        elif kind == "slow":
            w = max(1, win // tf)
            hi = pd.Series(high).rolling(w, min_periods=1).max().to_numpy()
            filt = close >= (1 - drop) * hi
        elif kind == "rsi3":
            r3 = H.rsi(B3["Close"], 14)
            v = map3_prev_complete(r3) if tf == 1 else r3.to_numpy(float)
            filt = v > r3thr
        else:
            raise ValueError(kind)
    return pd.Series(ev & filt, index=bars.index), bars, tf


def sig_volfade(tf=1, ema_n=10, lb=60, pct=0.015, thr=0.7, vs=10, vl=40):
    bars = B1 if tf == 1 else B3
    c = bars["Close"]
    e = H.ema(c, ema_n)
    cross = (c < e) & (c.shift(1) >= e.shift(1))
    lbn = max(1, lb // tf)
    rally_prev = (c >= bars["Low"].rolling(lbn, min_periods=1).min() * (1 + pct)).shift(1, fill_value=False)
    vr_prev = (bars["Volume"].rolling(vs).mean() / bars["Volume"].rolling(vl).mean()).shift(1)
    s = cross & rally_prev.astype(bool) & (vr_prev < thr)
    return s.fillna(False), bars, tf


# 랠리 진입 마스크(갈래와 같은 상황 정의: 3분 최근 20봉 저점 +1.5%, 1분봉 시각에 이미 완성된 3분봉)
_r3 = (B3["Close"] >= B3["Low"].rolling(20, min_periods=1).min() * 1.015).astype(float)
RALLY = map3_prev_complete(_r3) == 1


# ───────────── 평가 ─────────────
def evaluate(sigfn, kw, full=True, extras=False):
    sig, bars, tf = sigfn(**kw)
    pos = fill_pos(sig, bars, tf)
    out = {}
    for nm, rng in RANGES:
        hs = holdsim(pos, rng, detail=full)
        es = event_stats(pos, rng)
        r = {**hs, **es} if full else {**hs, "n": es["n"], "per_day": es["per_day"]}
        if extras:
            r["primary_cost10bp"] = holdsim(pos, rng, cost=0.001, detail=False)["primary"]
            hr = holdsim(pos, rng, mask_key="rally", mask=RALLY, detail=False)
            r["rally_primary"], r["rally_t"] = hr["primary"], hr["primary_t"]
            r.update(null_adj(pos, rng, hs["primary"]))
        out[nm] = r
    return out, pos


CANDS = [
    {"name": "HL9 저가터치 + 3분 EMA25<EMA60 + 1분 RSI>30", "branch": "exit_struct", "fn": sig_struct,
     "base": dict(kind="ema_rsi", L=9, fast=25, slow=60, rthr=30),
     "neighbors": [("L8", dict(L=8)), ("L10", dict(L=10)), ("fast20", dict(fast=20)), ("fast30", dict(fast=30)),
                   ("slow50", dict(slow=50)), ("slow70", dict(slow=70)), ("rsi25", dict(rthr=25)), ("rsi35", dict(rthr=35)),
                   ("close_below", dict(conf="c0")), ("pivot_right_strict", dict(strict_right=True))],
     "swap": dict(tf=3, L=3), "gt": {"train": 0.228, "test": -0.658}},
    {"name": "HL9 저가터치 + 완만한 하락(30분 고점 대비 2% 미만)", "branch": "exit_struct", "fn": sig_struct,
     "base": dict(kind="slow", L=9, win=30, drop=0.02),
     "neighbors": [("L8", dict(L=8)), ("L10", dict(L=10)), ("win20", dict(win=20)), ("win40", dict(win=40)),
                   ("drop1.5", dict(drop=0.015)), ("drop2.5", dict(drop=0.025)), ("close_below", dict(conf="c0")),
                   ("pivot_right_strict", dict(strict_right=True))],
     "swap": dict(tf=3, L=3), "gt": {"train": 0.204, "test": -0.504}},
    {"name": "HL9 저가터치 + 3분 RSI>40", "branch": "exit_struct", "fn": sig_struct,
     "base": dict(kind="rsi3", L=9, r3thr=40),
     "neighbors": [("L8", dict(L=8)), ("L10", dict(L=10)), ("rsi35", dict(r3thr=35)), ("rsi45", dict(r3thr=45)),
                   ("close_below", dict(conf="c0")), ("pivot_right_strict", dict(strict_right=True))],
     "swap": dict(tf=3, L=3), "gt": {"train": 0.193, "test": -0.223}},
    {"name": "E_volfade_1m_EMA10_thr0.7", "branch": "exit_fail", "fn": sig_volfade,
     "base": dict(ema_n=10, lb=60, pct=0.015, thr=0.7, vs=10, vl=40),
     "neighbors": [("ema9", dict(ema_n=9)), ("ema11", dict(ema_n=11)), ("lb50", dict(lb=50)), ("lb70", dict(lb=70)),
                   ("pct1.0", dict(pct=0.01)), ("pct2.0", dict(pct=0.02)), ("thr0.6", dict(thr=0.6)), ("thr0.8", dict(thr=0.8)),
                   ("vs8", dict(vs=8)), ("vs12", dict(vs=12)), ("vl30", dict(vl=30)), ("vl50", dict(vl=50))],
     "swap": dict(tf=3), "gt": {"train": 0.171, "test": -0.028}},
]


AUDIT_COMMON = [
    "체결: 1분봉 신호는 봉 끝(시작+1분) 이후 첫 1분봉 시가, 3분봉은 칸 끝 이후 첫 1분봉 시가(같은 날만). 갈래(H.exec_index)와 위치가 같다.",
    "피벗: exit_struct pivot_low_confirmed는 piv[j+R]에 값을 두고 higher_low_line(...).shift(1)로 j+R+1봉부터 사용(Pine hl[1]과 같음, 확정 봉보다 1봉 더 보수적). 미래정보 없음.",
    "피벗 동률: 왼쪽 엄격·오른쪽 이하(갈래)와 오른쪽도 엄격(검증자 대안)의 차이는 neighbors.pivot_right_strict에 있다(결론 불변).",
    "3분 맥락: to_1m_mask는 1분봉 시작 시각에 끝난 3분봉만 쓴다 = request.security('3', x[1], lookahead_on). 상위봉 미완성 사용 없음.",
    "장중 고저 누적·시가 범위·전일값·하루 전체값·중앙 창: 4개 후보 모두 사용하지 않음. 30분 고점(ta.highest(high,30))과 60봉 저점은 뒤쪽 창(현재 봉까지 또는 직전 봉까지).",
    "hl 선·EMA·RSI는 연장시간을 포함해 연속 계산(사용자 연장시간 차트와 같음). 첫 터치가 프리장에서 나면 정규장 신호는 생략된다(Pine도 같음).",
]
HARNESS_NOTES = [
    "harness.event_metrics 버그: last_day = rows['date'] == rows['date'].max() 가 '자료 마지막 날(진행 중 11-19)'이 아니라 '사건이 있는 마지막 날'을 고른다. "
    "11-19 사건이 없는 신호는 11-18 사건의 rclose가 지워진다. 1번 후보 TEST rclose_x_dm: 갈래 -0.945(11-18 2건 제외, 13건) vs 올바른 값 -0.916(15건). "
    "수정안: last_day = rows['date'] == d['date'].values[-1]. 결론 불변.",
    "harness.hold_sim 대조 편향(버그 아님, 설계 특성): 보유시간을 섞으면 16:00에서 잘려 대조의 실효 보유가 짧아지고, 이 폭락 장세에선 짧을수록 덜 잃어 드문 규칙의 x_rand가 음으로 치우친다. "
    "검증자 null(같은 날 같은 개수 무작위 사건) 평균은 후보별 -0.31~+0.07로 갈래 null과 같은 크기. primary와 null_adj를 함께 봐야 한다.",
    "primary(20회 섞기)와 정확한 기댓값(primary_exact)의 차는 0.01~0.03%p로 난수 잡음은 작다.",
]
VERDICTS = {
    "HL9 저가터치 + 3분 EMA25<EMA60 + 1분 RSI>30": ("reject",
        "재계산 일치(TRAIN 0.228, TEST -0.658). TEST에서 무작위보다 나쁨(t -2.31, null 백분위 0, CVaR -11.7 대 -9.1), 이웃 10개 TEST 전부 음(최고 -0.112), "
        "PRE -0.146·null 백분위 0, 3분 차트로 바꾸면 TRAIN도 0.036으로 사라짐. 403개 중 TRAIN 1위 선택편향."),
    "HL9 저가터치 + 완만한 하락(30분 고점 대비 2% 미만)": ("reject",
        "재계산 일치(0.204/-0.504). TEST 음(null 백분위 27, CVaR -16.0 대 -10.5), 이웃 8개 TEST 전부 음, 창 20분이면 TRAIN도 -0.002. "
        "PRE -0.092(t -2.03). 3분 바꿈 TRAIN 0.069/TEST -0.265."),
    "HL9 저가터치 + 3분 RSI>40": ("reject",
        "재계산 일치(0.193/-0.223). TEST 음(null 백분위 30), 이웃 6개 TEST 전부 음, RSI 45면 TRAIN 0.012로 무너짐, "
        "PRE -0.13(t -3.01), 3분 바꿈 TRAIN -0.163/TEST -0.816."),
    "E_volfade_1m_EMA10_thr0.7": ("fragile",
        "재계산 일치(0.171/-0.028), 미래정보 없음. TEST는 무작위와 같음(null_adj +0.011, 백분위 50)이고 꼬리는 더 나쁨(CVaR -11.3 대 -7.9), "
        "비용 0.1%면 TEST -0.099. 이웃 12개 중 TEST 양 3개(ema9 +0.129, thr0.8 +0.187, vs8 +0.084), thr0.6은 -0.457로 흔들림. "
        "PRE +0.125(t 1.47, null 백분위 90)는 지지하나 TEST 확인 실패라 채택 불가 — 종이 기록 후보로만."),
}


def main():
    t0 = time.time()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "data_last_bar": str(D.index[-1]),
           "method": __doc__, "audit_common": AUDIT_COMMON, "harness_notes": HARNESS_NOTES,
           "search_size": {"exit_struct_configs": 476, "exit_struct_eligible": 403, "exit_struct_train_test_rank_corr": 0.233,
                           "exit_fail_configs": 84, "exit_fail_eligible": 68,
                           "verifier_configs": "기준 4 + 이웃 36 + 시간봉 바꿈 4 = 44 설정, 각 TRAIN·TEST·PRE"},
           "candidates": []}
    for c in CANDS:
        base, pos = evaluate(c["fn"], c["base"], full=True, extras=True)
        case = [f"{pd.Timestamp(D.index[p]):%H:%M} @{O[p]:.2f}" for p in pos
                if DATES[p] == np.datetime64("2026-11-19") and REG[p]]
        nb = {}
        for nm, kw in c["neighbors"]:
            r, _ = evaluate(c["fn"], {**c["base"], **kw}, full=False)
            nb[nm] = {k: {kk: r[k][kk] for kk in ("primary", "primary_t", "primary_exact", "n", "per_day")} for k in r}
        sw, _ = evaluate(c["fn"], {**c["base"], **c["swap"]}, full=True)
        diff = {k: round(base[k]["primary"] - c["gt"][k], 3) for k in ("train", "test")}
        tr_nb = np.array([v["train"]["primary"] for v in nb.values()])
        te_nb = np.array([v["test"]["primary"] for v in nb.values()])
        pr_nb = np.array([v["pre"]["primary"] for v in nb.values()])
        row = {"name": c["name"], "branch": c["branch"], "base_params": c["base"], "recomputed": base,
               "branch_primary": c["gt"], "diff_vs_branch": diff,
               "recompute_match": bool(all(abs(v) <= 0.05 for v in diff.values())),
               "case_2026_11_19": case, "neighbors": nb,
               "neighbor_summary": {"n": int(len(nb)), "train_mean": round(float(tr_nb.mean()), 3),
                                    "train_min": round(float(tr_nb.min()), 3), "train_max": round(float(tr_nb.max()), 3),
                                    "train_share_pos": round(float((tr_nb > 0).mean()), 2),
                                    "test_mean": round(float(te_nb.mean()), 3), "test_max": round(float(te_nb.max()), 3),
                                    "test_share_pos": round(float((te_nb > 0).mean()), 2),
                                    "pre_mean": round(float(pr_nb.mean()), 3)},
               "swap_tf": {"params": {**c["base"], **c["swap"]}, **sw},
               "lookahead_issues": [], "verdict": VERDICTS[c["name"]][0], "reason": VERDICTS[c["name"]][1]}
        res["candidates"].append(row)
        print(c["name"], json.dumps({k: base[k]["primary"] for k in base}), diff, row["neighbor_summary"],
              "swap", {k: sw[k]["primary"] for k in sw}, round(time.time() - t0, 1), flush=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    print("saved", OUT, round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
