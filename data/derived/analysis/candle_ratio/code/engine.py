"""양봉 비율 매매 연구 엔진.

- 분봉 묶음: 트레이딩뷰처럼 세션 시작(연장 04:00, 정규장 09:30 ET)에 맞춰 묶고 세션 끝에서 자른다.
- 비율 3종(count·body·vol)은 지금 봉을 포함한 최근 N봉으로 봉 마감 때 계산한다.
- 상태 기계는 봉 마감에 판단하고 다음 봉 시가에 체결한다. 수익은 봉별 로그수익(시가→다음 시가)으로 쌓는다.
"""
from __future__ import annotations

import pathlib
from datetime import date

import numpy as np
import pandas as pd
from numba import njit

HERE = pathlib.Path(__file__).resolve().parent.parent
CACHE = HERE / "cache"
ET = "America/New_York"
SESS = {"eth": (240, 1200), "rth": (570, 960)}  # 분 (ET)

TF_5M = list(range(5, 361, 5))
TF_1H = [60, 120, 180, 240, 300, 360]
NS = [3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
BS = [0.5, 0.6, 0.7, 0.8]
SS = [0.5, 0.4, 0.3, 0.2]
RATIOS = ["count", "body", "vol"]
# 판정 구간 (BRIEF.md)
WIN = {
    "5m": dict(warm=date(2026, 7, 17), sel=(date(2026, 7, 17), date(2026, 9, 4)), conf=(date(2026, 9, 8), date(2026, 10, 22))),
    "1h": dict(warm=date(2026, 5, 1), sel=(date(2026, 6, 1), date(2026, 8, 14)), conf=(date(2026, 8, 17), date(2026, 10, 22))),
}
LAST_DAY = date(2026, 10, 22)


def load(sym: str, base: str) -> pd.DataFrame:
    d = pd.read_parquet(CACHE / f"{sym}_{base}.parquet")
    d.index = d.index.tz_convert(ET)
    return d[d.index.date <= LAST_DAY]


def tf_bars(base: pd.DataFrame, minutes: int, sess: str) -> pd.DataFrame:
    """기본봉(5분·1시간)을 세션 시작 기준 minutes분 봉으로 묶는다. 빈 묶음은 만들지 않는다."""
    start, end = SESS[sess]
    t = base.index
    mins = np.asarray(t.hour * 60 + t.minute)
    keep = (mins >= start) & (mins < end)
    b = base[keep]
    mins = mins[keep]
    dates = np.asarray(b.index.date)
    k = (mins - start) // minutes
    g = pd.DataFrame({"O": b.Open.values, "H": b.High.values, "L": b.Low.values, "C": b.Close.values,
                      "V": b.Volume.values, "d": dates, "k": k, "t": b.index}).groupby(["d", "k"], sort=True)
    out = pd.DataFrame({"O": g.O.first(), "H": g.H.max(), "L": g.L.min(), "C": g.C.last(), "V": g.V.sum(),
                        "t": g.t.first()})
    idx = out.index.to_frame(index=False)
    out = out.reset_index(drop=True)
    out["day"] = idx.d.values
    out["start"] = pd.DatetimeIndex(pd.to_datetime(idx.d.values)).tz_localize(ET) + pd.to_timedelta(start + idx.k.values * minutes, unit="m")
    out["lod"] = np.r_[out.day.values[1:] != out.day.values[:-1], True]  # 세션 마지막 봉
    return out


def segments(days: np.ndarray, win: dict) -> np.ndarray:
    """봉별 구간 번호: 0 고르기, 1 확인, -1 예열."""
    seg = np.full(len(days), -1, np.int64)
    s0, s1 = win["sel"]
    c0, c1 = win["conf"]
    seg[(days >= s0) & (days <= s1)] = 0
    seg[(days >= c0) & (days <= c1)] = 1
    return seg


def awake_mask(start: pd.Series) -> np.ndarray:
    """봉 시가 시각이 한국시간 08~24시(사용자 주문 가능 시간)면 True."""
    h = pd.DatetimeIndex(start).tz_convert("Asia/Seoul").hour
    return np.asarray(h >= 8)


# ---------------------------------------------------------------------------
# 비율
# ---------------------------------------------------------------------------

@njit(cache=True)
def _rolling_ratio(num, den, n, empty):
    m = len(num)
    out = np.full(m, np.nan)
    a = 0.0
    d = 0.0
    for i in range(m):
        a += num[i]
        d += den[i]
        if i >= n:
            a -= num[i - n]
            d -= den[i - n]
        if i >= n - 1:
            out[i] = a / d if d > 1e-12 else empty
    return out


def ratio(O, C, V, n: int, kind: str) -> np.ndarray:
    up = (C > O).astype(np.float64)
    dn = (C < O).astype(np.float64)
    if kind == "count":
        return _rolling_ratio(up, up + dn, n, 0.5)
    if kind == "body":
        d = C - O
        r = _rolling_ratio(d, np.abs(d), n, 0.0)  # Σ(C−O)/Σ|C−O| ∈ [−1, 1], 모두 도지면 0 → p 0.5 (10/24 수정: 전엔 0.75)
        return (1.0 + r) / 2.0
    if kind == "vol":
        # 야후 프리·애프터 봉 거래량은 0이라 연장 세션에서는 사실상 정규장 거래량 비율이다(audit 갈래)
        return _rolling_ratio(V * up, V * (up + dn), n, 0.5)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# 모의매매
# ---------------------------------------------------------------------------

@njit(cache=True)
def sim(o, c, p, lod, seg, awake, b, s, follow, dayflat, cost, out):
    """out[0:2] 구간 로그수익, [2:4] 진입 수, [4:6] 보유 봉 수, [6:8] 봉 수, [8:10] 최대 낙폭(로그), [10] 전체 최대 낙폭.
    awake가 False인 봉의 시가에는 포지션을 바꾸지 않는다(모두 True면 이상적 체결)."""
    n = len(o)
    lc = np.log(1.0 - cost)
    for k in range(out.shape[0]):
        out[k] = 0.0
    eq0 = 0.0; pk0 = 0.0; eq1 = 0.0; pk1 = 0.0; eqa = 0.0; pka = 0.0
    carry = 0
    target = 0
    state = 0
    for j in range(n):
        pos = target if awake[j] else carry
        sg = seg[j]
        r = 0.0
        if pos != carry:
            r += lc
            if pos == 1 and sg >= 0:
                out[2 + sg] += 1
        if pos == 1:
            if dayflat and lod[j]:
                r += np.log(c[j] / o[j]) + lc
                carry = 0
            elif j + 1 < n:
                r += np.log(o[j + 1] / o[j])
                carry = 1
            else:
                r += np.log(c[j] / o[j])
                carry = 1
        else:
            carry = 0
        if sg >= 0:
            out[sg] += r
            out[4 + sg] += pos
            out[6 + sg] += 1
            if sg == 0:
                eq0 += r
                pk0 = max(pk0, eq0)
                out[8] = min(out[8], eq0 - pk0)
            else:
                eq1 += r
                pk1 = max(pk1, eq1)
                out[9] = min(out[9], eq1 - pk1)
            eqa += r
            pka = max(pka, eqa)
            out[10] = min(out[10], eqa - pka)
        pj = p[j]
        if not np.isnan(pj):
            if state == 0:
                if (follow and pj > b) or ((not follow) and pj < 1.0 - b):
                    state = 1
            else:
                if (follow and pj < s) or ((not follow) and pj > 1.0 - s):
                    state = 0
        target = state


@njit(cache=True)
def positions(o, c, p, lod, awake, b, s, follow, dayflat):
    """봉별 보유(시가 기준)와 체결 표시: 1 매수, -1 매도(시가), -2 dayflat 종가 매도."""
    n = len(o)
    pos_arr = np.zeros(n, np.int64)
    ev = np.zeros(n, np.int64)
    carry = 0; target = 0; state = 0
    for j in range(n):
        pos = target if awake[j] else carry
        if pos != carry:
            ev[j] = 1 if pos == 1 else -1
        pos_arr[j] = pos
        if pos == 1 and dayflat and lod[j]:
            ev[j] = -2 if ev[j] == 0 else ev[j]
            carry = 0
        else:
            carry = pos
        pj = p[j]
        if not np.isnan(pj):
            if state == 0:
                if (follow and pj > b) or ((not follow) and pj < 1.0 - b):
                    state = 1
            else:
                if (follow and pj < s) or ((not follow) and pj > 1.0 - s):
                    state = 0
        target = state
    return pos_arr, ev


@njit(cache=True)
def grid_kernel(o, c, p, lod, seg, awake, bs, ss, costs, res):
    """문턱 16 × 방향 2 × 밤 보유 2 × 비용 k를 한 번에. res[칸, 11]."""
    out = np.zeros(11)
    row = 0
    for ci in range(len(costs)):
        for fi in range(2):
            for di in range(2):
                for bi in range(len(bs)):
                    for si in range(len(ss)):
                        sim(o, c, p, lod, seg, awake, bs[bi], ss[si], fi == 0, di == 1, costs[ci], out)
                        for k in range(11):
                            res[row, k] = out[k]
                        row += 1


def config_rows(costs):
    rows = []
    for cost in costs:
        for follow in (True, False):
            for dayflat in (False, True):
                for b in BS:
                    for s in SS:
                        rows.append((cost, "follow" if follow else "fade", dayflat, b, s))
    return rows


COLS = ["ret_sel", "ret_conf", "ent_sel", "ent_conf", "hold_sel", "hold_conf", "bars_sel", "bars_conf",
        "mdd_sel", "mdd_conf", "mdd_all"]


def bh(o, c, seg):
    """보유 기준: 늘 들고 있을 때 구간 로그수익(같은 시가→다음 시가 회계)."""
    r = np.r_[np.log(o[1:] / o[:-1]), np.log(c[-1] / o[-1])]
    return [float(r[seg == k].sum()) for k in (0, 1)]
