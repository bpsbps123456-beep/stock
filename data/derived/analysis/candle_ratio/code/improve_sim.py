"""improve 갈래: 양봉 비율 신호의 개선안 4개(추세 거름·일봉 국면·눌림 매수·손절)용 상태 기계.

engine.tf_bars·ratio·segments·bh를 그대로 쓰고, engine.sim은 건드리지 않는다.
- 판단은 봉 마감, 체결은 다음 봉 시가(engine과 같은 회계: 봉별 시가→다음 시가 로그수익).
- 손절은 보유 중 그 시간봉 안의 5분봉을 시각 순서대로 보며 저가가 손절가에 닿은 첫 5분봉에서 판다.
  체결가는 min(그 5분봉 시가, 손절가)(갭이면 시가). 목표가가 없어 '한 봉 안 손절·목표 동시 도달' 문제는 없다.
- 손절 뒤에는 원래 매도 조건(follow면 p < s)이 한 번 나올 때까지 다시 사지 않는다(새 신호를 기다림).
- 추세 거름 EMA60은 그 시간봉 종가로 계산하고, 5분봉 시작(7/17) 전 구간은 1시간봉(5/1~)을 같은 시간봉 칸 끝 시각에
  이미 끝난 1시간봉 종가로 채워 예열한다(미래 값 없음, 칸 안 위치만 근사).
- 일봉 국면: 1시간봉 정규장 마지막 봉 종가 = 일봉 종가, 20일 단순 이동평균. 봉 날짜 T에는 T보다 앞선 마지막 거래일의 값만 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from engine import SESS, WIN, bh, load, ratio, segments, tf_bars

TFS = [20, 30, 40, 60, 90, 120, 240]
NS_I = [5, 10, 15, 20]
BS_I = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]
SESSES = ["eth", "rth"]
COST_SENS = {"rth": 0.0005, "eth": 0.001}
# 이름: (follow, 진입 거름, 강제 청산, 손절)
VARIANTS = {
    "P": (True, "all", "none", 0.0),     # 플레인 원신호(follow)
    "PF": (False, "all", "none", 0.0),   # 플레인 fade(눌림 매수 비교용)
    "T1": (True, "ema", "none", 0.0),    # 1a 추세 거름: 종가>EMA60일 때만 매수, 매도는 원래 규칙
    "T1x": (True, "ema", "ema", 0.0),    # 1b 추세 거름: 매수 같음, 매도는 원래 규칙 또는 종가<EMA60
    "D2": (True, "reg", "none", 0.0),    # 2 일봉 국면: 전일 종가>일봉 SMA20일 때만 매수
    "V3": (False, "ema", "none", 0.0),   # 3 오름 추세 속 눌림 매수: 종가>EMA60이고 p<1−b 매수, p>1−s 매도
    "S2": (True, "all", "none", 0.02),   # 4 원신호 + −2% 손절
    "S3": (True, "all", "none", 0.03),   # 4 원신호 + −3% 손절
}
# 분해용 기준(비율 신호 없이 거름만, p ≡ 1로 돌린다): 시간봉 EMA60 위면 보유 / 일봉 국면이 켜져 있으면 보유
BASELINES = {
    "E": (True, "ema", "ema", 0.0),
    "R": (True, "reg", "reg", 0.0),
}
NOUT = 13
OUT_COLS = ["ret_sel", "ret_conf", "ent_sel", "ent_conf", "hold_sel", "hold_conf", "bars_sel", "bars_conf",
            "mdd_sel", "mdd_conf", "mdd_all", "stop_sel", "stop_conf"]


# ---------------------------------------------------------------------------
# 자료 준비
# ---------------------------------------------------------------------------

def _sess_filter(base: pd.DataFrame, sess: str):
    start, end = SESS[sess]
    t = base.index
    mins = np.asarray(t.hour * 60 + t.minute)
    keep = (mins >= start) & (mins < end)
    return base[keep], mins[keep]


def sub_index(base: pd.DataFrame, tf: int, sess: str, tb: pd.DataFrame):
    """tf_bars와 같은 묶음으로 기본봉의 [st, en) 범위를 돌려준다(손절 판정용)."""
    start, _ = SESS[sess]
    b, mins = _sess_filter(base, sess)
    d = np.asarray(pd.DatetimeIndex(b.index).tz_localize(None).normalize().asi8 // 86_400_000_000_000)
    key = d * 10_000 + (mins - start) // tf
    chg = np.r_[True, key[1:] != key[:-1]]
    st = np.flatnonzero(chg).astype(np.int64)
    en = np.r_[st[1:], len(key)].astype(np.int64)
    if len(st) != len(tb) or not np.allclose(b.Open.values[st], tb.O.values):
        raise RuntimeError("sub_index 묶음이 tf_bars와 다르다")
    return st, en, b.Open.values.astype(np.float64), b.Low.values.astype(np.float64)


def warm_closes(h1: pd.DataFrame, tf: int, sess: str, first_day) -> np.ndarray:
    """5분봉 시작 전 날들의 tf분 칸 종가 근사: 칸 끝 시각까지 끝난 마지막 1시간봉 종가(미래 값 없음)."""
    start, end = SESS[sess]
    b, mins = _sess_filter(h1, sess)
    dates = np.asarray(b.index.date)
    keep = dates < first_day
    mins, dates, cl = mins[keep], dates[keep], b.Close.values[keep].astype(np.float64)
    nslot = -(-(end - start) // tf)
    slot_end = np.minimum(start + (np.arange(nslot) + 1) * tf, end)
    out = []
    last = np.nan
    for dd in sorted(set(dates)):
        m = mins[dates == dd]
        cc = cl[dates == dd]
        nxt = np.r_[m[1:], end]
        be = np.minimum(np.minimum(m + 60, nxt), end)
        for se in slot_end:
            i = np.searchsorted(be, se, side="right") - 1
            if i >= 0:
                last = cc[i]
            if not np.isnan(last):
                out.append(last)
    return np.asarray(out)


def daily_regime(h1: pd.DataFrame, days: np.ndarray) -> np.ndarray:
    """봉 날짜 T마다: T 이전 마지막 거래일 정규장 종가 > 그날까지 20일 SMA."""
    b, mins = _sess_filter(h1, "rth")
    dc = pd.Series(b.Close.values.astype(np.float64), index=pd.to_datetime(np.asarray(b.index.date))).groupby(level=0).last()
    sma = dc.rolling(20).mean()
    ok = (dc > sma).values  # SMA 없으면 False
    dts = dc.index.values.astype("datetime64[D]")
    q = pd.to_datetime(days).values.astype("datetime64[D]")
    i = np.searchsorted(dts, q, side="left") - 1
    flag = np.where(i >= 0, ok[np.clip(i, 0, None)], False)
    return flag.astype(np.bool_)


def prep(sym: str, tf: int, sess: str, base_name: str = "5m", d5=None, h1=None) -> dict:
    d5 = load(sym, base_name) if d5 is None else d5
    h1 = load(sym, "1h") if h1 is None else h1
    win = WIN[base_name]
    tb = tf_bars(d5, tf, sess)
    tb = tb[tb.day >= win["warm"]].reset_index(drop=True)
    st, en, o5, l5 = sub_index(d5, tf, sess, tb) if base_name == "5m" else _sub_after_warm(d5, tf, sess, win)
    O = tb.O.values.astype(np.float64)
    C = tb.C.values.astype(np.float64)
    V = tb.V.values.astype(np.float64)
    seg = segments(tb.day.values, win)
    first_day = tb.day.values[0]
    warm = warm_closes(h1, tf, sess, first_day)
    ema = pd.Series(np.r_[warm, C]).ewm(span=60, adjust=False).mean().values[len(warm):]
    ema_ok = C > ema
    reg_ok = daily_regime(h1, tb.day.values)
    dayv = tb.day.values
    day_start = np.flatnonzero(np.r_[True, dayv[1:] != dayv[:-1]])
    bh_sel, bh_conf = bh(O, C, seg)
    return dict(tb=tb, O=O, C=C, V=V, seg=seg, ema=ema, ema_ok=ema_ok, reg_ok=reg_ok, st=st, en=en, o5=o5, l5=l5,
                day_start=day_start, bh_sel=bh_sel, bh_conf=bh_conf, nwarm=len(warm))


def _sub_after_warm(base, tf, sess, win):
    tb = tf_bars(base, tf, sess)
    keep = (tb.day >= win["warm"]).values
    st, en, o5, l5 = sub_index(base, tf, sess, tb)
    return st[keep], en[keep], o5, l5


def masks(P: dict, which: str) -> np.ndarray:
    n = len(P["O"])
    if which == "all":
        return np.ones(n, np.bool_)
    if which == "none":
        return np.zeros(n, np.bool_)
    if which == "ema":
        return P["ema_ok"].copy()
    if which == "reg":
        return P["reg_ok"].copy()
    raise ValueError(which)


def variant_arrays(P: dict, name: str):
    follow, ent, ex, stop = VARIANTS[name] if name in VARIANTS else BASELINES[name]
    ent_ok = masks(P, ent)
    if ex == "ema":
        ex_force = ~P["ema_ok"]
    elif ex == "reg":
        ex_force = ~P["reg_ok"]
    else:
        ex_force = np.zeros(len(P["O"]), np.bool_)
    return follow, ent_ok, ex_force, stop


# ---------------------------------------------------------------------------
# 상태 기계
# ---------------------------------------------------------------------------

@njit(cache=True)
def _sell(pj, exf, follow, s):
    if exf:
        return True
    if np.isnan(pj):
        return False
    return (follow and pj < s) or ((not follow) and pj > 1.0 - s)


@njit(cache=True)
def _buy(pj, ok, follow, b):
    if (not ok) or np.isnan(pj):
        return False
    return (follow and pj > b) or ((not follow) and pj < 1.0 - b)


@njit(cache=True)
def sim2(o, c, p, ent_ok, ex_force, seg, b, s, follow, stop, st, en, o5, l5, cost, forced, use_forced, out):
    """out: OUT_COLS 순서. use_forced면 진입은 forced[j](무작위) 봉 마감에만, 청산 규칙은 같다(손절 뒤 잠금 없음)."""
    n = len(o)
    lc = np.log(1.0 - cost)
    for k in range(out.shape[0]):
        out[k] = 0.0
    eq0 = 0.0; pk0 = 0.0; eq1 = 0.0; pk1 = 0.0; eqa = 0.0; pka = 0.0
    carry = 0
    target = 0
    state = 0  # 0 현금, 1 보유, 2 손절 뒤 잠금
    entry_px = 0.0
    for j in range(n):
        pos = target
        sg = seg[j]
        r = 0.0
        if pos != carry:
            r += lc
            if pos == 1:
                entry_px = o[j]
                if sg >= 0:
                    out[2 + sg] += 1
        stopped = False
        if pos == 1:
            if stop > 0.0:
                spx = entry_px * (1.0 - stop)
                for i in range(st[j], en[j]):
                    if l5[i] <= spx:
                        fill = o5[i] if o5[i] < spx else spx
                        r += np.log(fill / o[j]) + lc
                        stopped = True
                        if sg >= 0:
                            out[11 + sg] += 1
                        break
            if stopped:
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
        # 봉 마감 판단
        if stopped:
            state = 0 if use_forced else 2
        if state == 2:
            if _sell(p[j], ex_force[j], follow, s):
                state = 0
        elif state == 0:
            if use_forced:
                if forced[j]:
                    state = 1
            elif _buy(p[j], ent_ok[j], follow, b):
                state = 1
        else:
            if _sell(p[j], ex_force[j], follow, s):
                state = 0
        target = 1 if state == 1 else 0


@njit(cache=True)
def exit_bars(o, p, ex_force, s, follow, stop, st, en, l5):
    """판단 봉 d(마감에 매수 결정, d+1 시가 진입)마다 같은 청산 규칙으로 끝나는 봉 x[d]."""
    n = len(o)
    x = np.full(n, n - 1, np.int64)
    if stop <= 0.0:
        nxt = n - 1
        for j in range(n - 1, -1, -1):
            if j < n - 1:
                x[j] = nxt
            if _sell(p[j], ex_force[j], follow, s):
                nxt = j
        return x
    for d in range(n - 1):
        e = d + 1
        spx = o[e] * (1.0 - stop)
        xd = n - 1
        for j in range(e, n):
            hit = False
            for i in range(st[j], en[j]):
                if l5[i] <= spx:
                    hit = True
                    break
            if hit or _sell(p[j], ex_force[j], follow, s):
                xd = j
                break
        x[d] = xd
    return x


@njit(cache=True)
def rand_flags(seg, x, k0, k1, seed, flags):
    """구간별로 진입 수 k0·k1개의 겹치지 않는 무작위 매매를 고른다. 실제로 고른 수를 돌려준다."""
    np.random.seed(seed)
    n = len(seg)
    occ = np.zeros(n, np.bool_)
    for j in range(n):
        flags[j] = False
    got = np.zeros(2, np.int64)
    for sg in range(2):
        k = k0 if sg == 0 else k1
        cand = np.empty(n, np.int64)
        m = 0
        for d in range(n - 1):
            if seg[d + 1] == sg:
                cand[m] = d
                m += 1
        if m == 0 or k == 0:
            continue
        tries = 0
        while got[sg] < k and tries < 3000 * k:
            tries += 1
            d = cand[np.random.randint(0, m)]
            ok = True
            for t in range(d, x[d] + 1):
                if occ[t]:
                    ok = False
                    break
            if ok:
                for t in range(d, x[d] + 1):
                    occ[t] = True
                flags[d] = True
                got[sg] += 1
    return got


def run_cell(P, p, name, b, s, cost, shift=0):
    follow, ent_ok, ex_force, stop = variant_arrays(P, name)
    if shift:
        p = np.roll(p, shift); ent_ok = np.roll(ent_ok, shift); ex_force = np.roll(ex_force, shift)
    out = np.zeros(NOUT)
    dummy = np.zeros(len(P["O"]), np.bool_)
    sim2(P["O"], P["C"], p, ent_ok, ex_force, P["seg"], b, s, follow, stop, P["st"], P["en"], P["o5"], P["l5"],
         cost, dummy, False, out)
    return out


def random_control(P, p, name, b, s, cost, k0, k1, seeds=50):
    """같은 청산 규칙(원래 매도 조건 + 같은 손절)의 무작위 진입. 반환 [seed, ret_sel, ret_conf, ent_sel, ent_conf]."""
    follow, ent_ok, ex_force, stop = variant_arrays(P, name)
    x = exit_bars(P["O"], p, ex_force, s, follow, stop, P["st"], P["en"], P["l5"])
    flags = np.zeros(len(P["O"]), np.bool_)
    out = np.zeros(NOUT)
    res = np.zeros((seeds, 5))
    for sd in range(seeds):
        rand_flags(P["seg"], x, int(k0), int(k1), sd + 1, flags)
        sim2(P["O"], P["C"], p, ent_ok, ex_force, P["seg"], b, s, follow, stop, P["st"], P["en"], P["o5"], P["l5"],
             cost, flags, True, out)
        res[sd] = (sd + 1, out[0], out[1], out[2], out[3])
    return res
