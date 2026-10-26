"""Kalman ATR 5단계 연구 엔진: TradingView식 분봉, 지표(Pine 동일 계산), 사용자 체결 조건 모의매매."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit

SYMS = ["AMD", "ARM", "DELL", "INTC", "SNDK", "STX", "WDC"]
RTH_START, RTH_END, ETH_START, ETH_END = 570, 960, 240, 1200  # 분 (ET)


def load():
    out = {}
    for s in SYMS:
        for iv in ("5m", "1h"):
            d = pd.read_parquet(f"data/{s}_{iv}.parquet")
            d.index = d.index.tz_convert("America/New_York")
            out[(s, iv)] = d
    return out


def half_days(d1h: pd.DataFrame) -> set:
    """1시간봉에 13:00 시작 봉이 있는 날 = 13시 조기 폐장일."""
    t = d1h.index
    return set(t[(t.hour == 13) & (t.minute == 0)].date)


# ---------------------------------------------------------------------------
# 분봉 구성 (TradingView: 세션 시작에 맞추고 마지막 봉은 세션 끝에서 자름)
# ---------------------------------------------------------------------------

def tf_bars(base: pd.DataFrame, minutes: int, eth: bool, halfs: set) -> pd.DataFrame:
    t = base.index
    mins = np.asarray(t.hour * 60 + t.minute)
    dates = np.asarray(t.date)
    start = ETH_START if eth else RTH_START
    half_end = 17 * 60 if eth else 13 * 60
    full_end = ETH_END if eth else RTH_END
    endv = np.array([half_end if d in halfs else full_end for d in dates])
    keep = (mins >= start) & (mins < endv)
    mins, dates, endv = mins[keep], dates[keep], endv[keep]
    b = base[keep]
    k = (mins - start) // minutes
    g = pd.DataFrame({"O": b.Open.values, "H": b.High.values, "L": b.Low.values, "C": b.Close.values,
                      "d": dates, "k": k, "e": endv}).groupby(["d", "k"], sort=True)
    out = pd.DataFrame({"O": g.O.first(), "H": g.H.max(), "L": g.L.min(), "C": g.C.last(), "e": g.e.first()})
    idx = out.index.to_frame(index=False)
    end_min = np.minimum(start + (idx.k.values + 1) * minutes, out.e.values)
    day0 = pd.DatetimeIndex(pd.to_datetime(idx.d.values)).tz_localize("America/New_York")
    out = out.reset_index(drop=True)
    out["end_ns"] = (day0 + pd.to_timedelta(end_min, unit="m")).as_unit("ns").asi8
    out["day"] = idx.d.values
    return out


def base_arrays(base: pd.DataFrame, halfs: set) -> dict:
    """체결용 기본봉(연장시간 포함) 배열과 정규장·깨어 있음(08~24시 KST)·일말 표시."""
    t = base.index
    mins = np.asarray(t.hour * 60 + t.minute)
    dates = np.asarray(t.date)
    reg_end = np.array([13 * 60 if d in halfs else RTH_END for d in dates])
    is_reg = (mins >= RTH_START) & (mins < reg_end)
    kst = t.tz_convert("Asia/Seoul")
    is_awake = np.asarray(kst.hour >= 8)
    day_codes, day_idx = np.unique(dates, return_inverse=True)
    last = {}
    for j in np.flatnonzero(is_reg):
        last[day_idx[j]] = j
    is_end = np.zeros(len(base), bool)
    is_end[list(last.values())] = True
    return dict(o=base.Open.values.astype(np.float64), h=base.High.values.astype(np.float64),
                l=base.Low.values.astype(np.float64), c=base.Close.values.astype(np.float64),
                start_ns=np.asarray(t.as_unit("ns").asi8), is_reg=is_reg, is_awake=is_awake, is_end=is_end,
                day_idx=day_idx, days=day_codes, dates=dates)


def map_completed(start_ns: np.ndarray, end_ns: np.ndarray) -> np.ndarray:
    """기본봉 시작 시각에 이미 끝난 마지막 분봉 번호(-1이면 없음)."""
    return (np.searchsorted(end_ns, start_ns, side="right") - 1).astype(np.int64)


# ---------------------------------------------------------------------------
# 지표 (Pine과 같은 계산)
# ---------------------------------------------------------------------------

def length_to_gain(L: float) -> float:
    return 2.0 / (L + 1.0)


def gain_to_q(g: float, r: float = 3.0) -> float:
    """정상상태 칼만 이득 g를 만드는 Process Noise(Measurement Noise r 고정)."""
    s = g / (1 - g)
    return r * s * s / (1 + s)


def band_scale(g: float) -> float:
    """무작위 보행이면 (가격-칼만선) 표준편차 = 봉당 변동 x (1-g)/sqrt(g(2-g))."""
    return (1 - g) / math.sqrt(g * (2 - g))


@njit(cache=True)
def kalman(c, q, r):
    n = len(c)
    out = np.empty(n)
    x = np.nan
    P = 1.0
    for i in range(n):
        z = c[i]
        if np.isnan(x):
            x = z
            P = 1.0
        Pp = P + q
        K = Pp / (Pp + r)
        x = x + K * (z - x)
        P = (1 - K) * Pp
        out[i] = x
    return out


@njit(cache=True)
def rma(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    s = np.nan
    for i in range(m):
        if i == n - 1:
            acc = 0.0
            for k in range(n):
                acc += x[k]
            s = acc / n
        elif i >= n:
            s = (x[i] + (n - 1) * s) / n
        out[i] = s
    return out


@njit(cache=True)
def atr(h, l, c, n):
    m = len(c)
    tr = np.empty(m)
    for i in range(m):
        if i == 0:
            tr[i] = h[i] - l[i]
        else:
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return rma(tr, n)


@njit(cache=True)
def rms(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    acc = 0.0
    for i in range(m):
        acc += x[i] * x[i]
        if i >= n:
            acc -= x[i - n] * x[i - n]
        if i >= n - 1:
            out[i] = math.sqrt(max(acc, 0.0) / n)
    return out


@njit(cache=True)
def ratchet(K, a, factor):
    """원본 Kalman ATR 선과 색(1 초록, -1 빨강, 0 시작 흰색)."""
    n = len(K)
    line = np.empty(n)
    trend = np.zeros(n, np.int8)
    prev = np.nan
    cur = 0
    for i in range(n):
        v = K[i] if np.isnan(prev) else prev
        if not np.isnan(a[i]):
            lo = K[i] - factor * a[i]
            hi = K[i] + factor * a[i]
            if lo > v:
                v = lo
            if hi < v:
                v = hi
        line[i] = v
        prev = v
        if i >= 2:
            if line[i] > line[i - 1] and line[i - 1] <= line[i - 2]:
                cur = 1
            if line[i] < line[i - 1] and line[i - 1] >= line[i - 2]:
                cur = -1
        trend[i] = cur
    return line, trend


@njit(cache=True)
def zones(c, K, u, a, b):
    """2 과매수 / 1 매수 / 0 평범 / -1 매도 / -2 패닉셀. 단위가 없으면 -9."""
    n = len(c)
    z = np.empty(n, np.int8)
    for i in range(n):
        if np.isnan(u[i]) or u[i] <= 0:
            z[i] = -9
            continue
        d = (c[i] - K[i]) / u[i]
        if d >= b:
            z[i] = 2
        elif d >= a:
            z[i] = 1
        elif d > -a:
            z[i] = 0
        elif d > -b:
            z[i] = -1
        else:
            z[i] = -2
    return z


# ---------------------------------------------------------------------------
# 모의매매 (사용자 조건)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _trade(cash, sh, state, new, px, cost):
    """state 0 없음 / 1 절반 / 2 전량 → new 로 맞춘다."""
    if new == state:
        return cash, sh
    if new > state:  # 남은 현금 전부 매수
        sh += cash * (1 - cost) / px
        cash = 0.0
    elif new == 1:   # 전량 → 절반
        cash += 0.5 * sh * px * (1 - cost)
        sh *= 0.5
    else:            # 전부 매도
        cash += sh * px * (1 - cost)
        sh = 0.0
    return cash, sh


@njit(cache=True)
def simulate(o, h, l, c, is_reg, is_awake, is_end, tf_idx, zone, line_tp, line_pb,
             j0, tp_half, panic_hold, use_limits, cost, rec):
    """반환: 일말 평가금, 체결 수, 체결 기록, 보유 비중 합(노출 계산용).

    판단: 깨어 있을 때(08~24시 KST) 새로 끝난 분봉이 있으면 그 봉의 구간으로 목표 보유를 정한다.
    체결: 정규장 봉의 시가로 목표에 맞춘다(프리·애프터·휴장 중 판단은 다음 정규장 시가).
    지정가: 마지막 판단 때 라인 가격으로 걸어 둔 패닉 매수·과매수 익절이 정규장에서 닿으면 체결.
    """
    n = len(o)
    cash, sh = 1.0, 0.0
    state, desired, etype = 0, 0, 0
    last_tf, last_zone = -1, -99
    buy_lim, sell_lim = np.nan, np.nan
    n_end = 0
    for j in range(j0, n):
        if is_end[j]:
            n_end += 1
    eq = np.empty(n_end)
    expo = np.empty(n_end)
    cap = 4096 if rec else 1
    log = np.zeros((cap, 6))  # j, side, price, new_state, reason(0 판단 시가 / 1 패닉 지정가 / 2 익절 지정가), zone
    nlog = 0
    trades = 0
    k = 0
    for j in range(j0, n):
        i = tf_idx[j]
        if is_awake[j] and i >= 0 and i != last_tf:
            if zone[i] != -9:
                last_tf = i
                z = int(zone[i])
                if panic_hold and etype == 2 and z >= 0:
                    etype = 1
                if z == 2:
                    if last_zone != 2 and desired > 0:
                        desired = (desired - 1) if tp_half else 0
                elif z == 1:
                    if desired < 2:
                        if desired == 0:
                            etype = 1
                        desired = 2
                elif z == -1:
                    if desired > 0 and not (panic_hold and etype == 2):
                        desired = 0
                elif z == -2:
                    if desired < 2:
                        if desired == 0:
                            etype = 2
                        desired = 2
                last_zone = z
                if use_limits:
                    buy_lim = line_pb[i] if desired < 2 else np.nan
                    sell_lim = line_tp[i] if desired > 0 else np.nan
        if is_reg[j]:
            if desired != state:
                cash, sh = _trade(cash, sh, state, desired, o[j], cost)
                if rec and nlog < cap:
                    log[nlog, 0] = j
                    log[nlog, 1] = 1.0 if desired > state else -1.0
                    log[nlog, 2] = o[j]
                    log[nlog, 3] = desired
                    log[nlog, 4] = 0
                    log[nlog, 5] = last_zone
                    nlog += 1
                state = desired
                trades += 1
            if use_limits:
                if (not np.isnan(buy_lim)) and state < 2 and l[j] <= buy_lim:
                    px = min(o[j], buy_lim)
                    if state == 0:
                        etype = 2
                    cash, sh = _trade(cash, sh, state, 2, px, 0.0)
                    if rec and nlog < cap:
                        log[nlog, 0] = j
                        log[nlog, 1] = 1.0
                        log[nlog, 2] = px
                        log[nlog, 3] = 2
                        log[nlog, 4] = 1
                        log[nlog, 5] = -2
                        nlog += 1
                    state = 2
                    desired = 2
                    buy_lim = np.nan
                    trades += 1
                elif (not np.isnan(sell_lim)) and state > 0 and h[j] >= sell_lim:
                    px = max(o[j], sell_lim)
                    new = (state - 1) if tp_half else 0
                    cash, sh = _trade(cash, sh, state, new, px, 0.0)
                    if rec and nlog < cap:
                        log[nlog, 0] = j
                        log[nlog, 1] = -1.0
                        log[nlog, 2] = px
                        log[nlog, 3] = new
                        log[nlog, 4] = 2
                        log[nlog, 5] = 2
                        nlog += 1
                    state = new
                    desired = new
                    sell_lim = np.nan
                    trades += 1
        if is_end[j]:
            eq[k] = cash + sh * c[j]
            expo[k] = (sh * c[j]) / eq[k] if eq[k] > 0 else 0.0
            k += 1
    return eq, trades, log[:nlog], expo


@njit(cache=True)
def buy_hold(o, c, is_reg, is_end, j0):
    n_end = 0
    for j in range(j0, len(o)):
        if is_end[j]:
            n_end += 1
    eq = np.empty(n_end)
    sh = 0.0
    k = 0
    for j in range(j0, len(o)):
        if sh == 0.0 and is_reg[j]:
            sh = 1.0 / o[j]
        if is_end[j]:
            eq[k] = sh * c[j] if sh > 0 else 1.0
            k += 1
    return eq


def metrics(eq_port: np.ndarray, base: float = 1.0) -> dict:
    """일말 포트폴리오 평가금 → 수익률·샤프(연율)·최대낙폭."""
    e = np.r_[base, eq_port]
    r = e[1:] / e[:-1] - 1
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    peak = np.maximum.accumulate(e)
    return {"ret": e[-1] / e[0] - 1,
            "sharpe": float(r.mean() / sd * math.sqrt(252)) if sd and sd > 0 else 0.0,
            "mdd": float((e / peak - 1).min())}


@njit(cache=True)
def simulate2(o, h, l, c, is_reg, is_awake, is_end, tf_idx, zone, trend, line_tp, line_pb,
              j0, tp_half, panic_hold, cost, stop_mode, buy_filter, dip_buy, rec):
    """simulate + 추세 조건. trend: 분봉별 칼만선 방향(1 상승, -1 하락).
    stop_mode 0 매도 구간이면 손절 / 1 손절 없음 / 2 칼만선 하락일 때만 손절.
    buy_filter 1 이면 매수 구간 진입은 칼만선 상승일 때만. dip_buy 1 이면 상승 중 매도 구간을 눌림 매수.
    """
    n = len(o)
    cash, sh = 1.0, 0.0
    state, desired, etype = 0, 0, 0
    last_tf, last_zone = -1, -99
    buy_lim, sell_lim = np.nan, np.nan
    n_end = 0
    for j in range(j0, n):
        if is_end[j]:
            n_end += 1
    eq = np.empty(n_end)
    expo = np.empty(n_end)
    cap = 4096 if rec else 1
    log = np.zeros((cap, 6))
    nlog = 0
    trades = 0
    k = 0
    for j in range(j0, n):
        i = tf_idx[j]
        if is_awake[j] and i >= 0 and i != last_tf:
            if zone[i] != -9:
                last_tf = i
                z = int(zone[i])
                tr = int(trend[i])
                if panic_hold and etype == 2 and z >= 0:
                    etype = 1
                if z == 2:
                    if last_zone != 2 and desired > 0:
                        desired = (desired - 1) if tp_half else 0
                elif z == 1:
                    if desired < 2 and (buy_filter == 0 or tr == 1):
                        if desired == 0:
                            etype = 1
                        desired = 2
                elif z == -1:
                    if dip_buy == 1 and tr == 1 and desired < 2:
                        if desired == 0:
                            etype = 1
                        desired = 2
                    elif desired > 0 and not (panic_hold and etype == 2):
                        if stop_mode == 0 or (stop_mode == 2 and tr == -1):
                            desired = 0
                elif z == -2:
                    if desired < 2:
                        if desired == 0:
                            etype = 2
                        desired = 2
                last_zone = z
                buy_lim = line_pb[i] if desired < 2 else np.nan
                sell_lim = line_tp[i] if desired > 0 else np.nan
        if is_reg[j]:
            if desired != state:
                cash, sh = _trade(cash, sh, state, desired, o[j], cost)
                if rec and nlog < cap:
                    log[nlog, 0] = j
                    log[nlog, 1] = 1.0 if desired > state else -1.0
                    log[nlog, 2] = o[j]
                    log[nlog, 3] = desired
                    log[nlog, 4] = 0
                    log[nlog, 5] = last_zone
                    nlog += 1
                state = desired
                trades += 1
            if (not np.isnan(buy_lim)) and state < 2 and l[j] <= buy_lim:
                px = min(o[j], buy_lim)
                if state == 0:
                    etype = 2
                cash, sh = _trade(cash, sh, state, 2, px, 0.0)
                if rec and nlog < cap:
                    log[nlog, 0] = j
                    log[nlog, 1] = 1.0
                    log[nlog, 2] = px
                    log[nlog, 3] = 2
                    log[nlog, 4] = 1
                    log[nlog, 5] = -2
                    nlog += 1
                state = 2
                desired = 2
                buy_lim = np.nan
                trades += 1
            elif (not np.isnan(sell_lim)) and state > 0 and h[j] >= sell_lim:
                px = max(o[j], sell_lim)
                new = (state - 1) if tp_half else 0
                cash, sh = _trade(cash, sh, state, new, px, 0.0)
                if rec and nlog < cap:
                    log[nlog, 0] = j
                    log[nlog, 1] = -1.0
                    log[nlog, 2] = px
                    log[nlog, 3] = new
                    log[nlog, 4] = 2
                    log[nlog, 5] = 2
                    nlog += 1
                state = new
                desired = new
                sell_lim = np.nan
                trades += 1
        if is_end[j]:
            eq[k] = cash + sh * c[j]
            expo[k] = (sh * c[j]) / eq[k] if eq[k] > 0 else 0.0
            k += 1
    return eq, trades, log[:nlog], expo


def kalman_trend(K: np.ndarray) -> np.ndarray:
    t = np.ones(len(K), np.int8)
    t[1:] = np.where(K[1:] > K[:-1], 1, -1)
    return t
