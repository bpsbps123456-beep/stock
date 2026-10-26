"""양봉 비율 3단계 보조지표 공통 계산(BRIEF2.md).

- 원안 신호 s(40분 연장 N10 개수 따라가기 0.5/0.5)와 추세 판정 후보 28개(ER·FLIP·EXT·ADX·HTF·Z5)를 봉 마감 기준으로 만든다.
- 구간 Z: 추세이고 s=1 → +1, 추세이고 s=0 → −1, 추세 아님 → 0.
- 목표 보유: +1 → 1, −1 → 0, 0 → A0 현금 / A1 절반 / A2 동결, 기준 A3 = s.
- 모의매매는 봉 마감 판단·다음 봉 시가 체결, 보유 비율(0·0.5·1) 변화량 × 편도 비용, awake=False 봉 시가에서는 바꾸지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from engine import CACHE, ET, WIN, awake_mask, load, ratio, segments, tf_bars

TF, SESS, N, B, S = 40, "eth", 10, 0.5, 0.5
ACTIONS = ["A0", "A1", "A2"]
CLASSIFIERS = (
    [("ER", dict(K=k, th=t)) for k in (10, 20, 40) for t in (0.2, 0.3, 0.4)]
    + [("FLIP", dict(K=k, m=m)) for k in (20, 40, 80) for m in (2, 3, 4)]
    + [("EXT", dict(n=10, d=0.2)), ("EXT", dict(n=10, d=0.3)), ("EXT", dict(n=20, d=0.1)), ("EXT", dict(n=20, d=0.2))]
    + [("ADX", dict(th=20)), ("ADX", dict(th=25))]
    + [("HTF", dict(tf=160)), ("HTF", dict(tf=320)), ("HTF", dict(tf="D"))]
    + [("Z5", {})]
)
DEFAULT = ("ER", dict(K=20, th=0.3), "A2")


def cname(kind, prm):
    return kind + "".join(f"_{k}{v}" for k, v in prm.items())


@njit(cache=True)
def signal_state(p, b, s):
    """봉 마감 뒤 원안 상태(1 보유 신호, 0 현금 신호). 처음 na 구간은 0."""
    n = len(p)
    out = np.zeros(n, np.int64)
    st = 0
    for j in range(n):
        pj = p[j]
        if not np.isnan(pj):
            if st == 0 and pj > b:
                st = 1
            elif st == 1 and pj < s:
                st = 0
        out[j] = st
    return out


@njit(cache=True)
def efficiency(c, k):
    n = len(c)
    out = np.full(n, np.nan)
    for j in range(k, n):
        path = 0.0
        for i in range(j - k + 1, j + 1):
            path += abs(c[i] - c[i - 1])
        out[j] = abs(c[j] - c[j - k]) / path if path > 0 else 0.0
    return out


@njit(cache=True)
def flips(st, k):
    n = len(st)
    out = np.full(n, -1, np.int64)
    for j in range(k, n):
        f = 0
        for i in range(j - k + 1, j + 1):
            if st[i] != st[i - 1]:
                f += 1
        out[j] = f
    return out


@njit(cache=True)
def adx(h, l, c, n):
    m = len(c)
    out = np.full(m, np.nan)
    tr = np.zeros(m); pdm = np.zeros(m); ndm = np.zeros(m)
    for i in range(1, m):
        up = h[i] - h[i - 1]; dn = l[i - 1] - l[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        ndm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = 0.0; sp = 0.0; sn = 0.0; dx_s = 0.0; cnt = 0
    for i in range(1, m):
        if i <= n:
            atr += tr[i]; sp += pdm[i]; sn += ndm[i]
            if i < n:
                continue
        else:
            atr = atr - atr / n + tr[i]; sp = sp - sp / n + pdm[i]; sn = sn - sn / n + ndm[i]
        pdi = 100 * sp / atr if atr > 0 else 0.0
        ndi = 100 * sn / atr if atr > 0 else 0.0
        dx = 100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) > 0 else 0.0
        cnt += 1
        if cnt <= n:
            dx_s += dx
            if cnt == n:
                dx_s /= n
                out[i] = dx_s
        else:
            dx_s = (dx_s * (n - 1) + dx) / n
            out[i] = dx_s
    return out


def bar_end(tb: pd.DataFrame, minutes: int) -> np.ndarray:
    """봉 끝 시각(ns). 세션 끝 20:00 ET에서 자른다."""
    start = pd.DatetimeIndex(tb.start)
    sess_end = pd.DatetimeIndex(pd.to_datetime(tb.day.values)).tz_localize(ET) + pd.Timedelta(minutes=1200)
    end = np.minimum((start + pd.Timedelta(minutes=minutes)).asi8, sess_end.asi8)
    return end


def zone5_daily(d1: pd.DataFrame, n=20, th=0.5) -> pd.Series:
    """추세 5단계(trend_5zone.pine과 같은 계산), 날짜 → 단계."""
    c = d1.Close.astype(float)
    lc = np.log(c)
    lr = lc.diff()
    sd = lr.rolling(n).std(ddof=1)
    k = np.sqrt((n - 1.0) * (2.0 * n - 1.0) / (6.0 * n))
    z = (lc - lc.rolling(n).mean()) / (sd * k)
    dlt = c.diff()
    up = dlt.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-dlt.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    zn = np.where(rsi >= 70, 2, np.where(rsi <= 30, -2, np.where(z >= th, 1, np.where(z <= -th, -1, 0))))
    zn = pd.Series(zn, index=pd.DatetimeIndex(d1.index).date).astype(float)
    zn[(z.isna() | rsi.isna()).values] = np.nan
    zn.iloc[:129] = np.nan
    return zn


def daily_state(d1: pd.DataFrame) -> pd.Series:
    O, C = d1.Open.values.astype(float), d1.Close.values.astype(float)
    st = signal_state(ratio(O, C, np.ones(len(C)), N, "count"), B, S)
    return pd.Series(st, index=pd.DatetimeIndex(d1.index).date)


def last_daily(values: pd.Series, tb: pd.DataFrame, end_ns: np.ndarray) -> np.ndarray:
    """40분봉 마감 시각에 이미 끝난(16:00 ET 정규장 마감) 마지막 일봉 값."""
    days = np.array(sorted(values.index))
    close_ns = (pd.DatetimeIndex(pd.to_datetime(days)).tz_localize(ET) + pd.Timedelta(minutes=960)).asi8
    idx = np.searchsorted(close_ns, end_ns, side="right") - 1
    v = values.reindex(days).values.astype(float)
    out = np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan)
    return out


def htf_state(base: pd.DataFrame, minutes: int, end40: np.ndarray, win_warm) -> np.ndarray:
    tb = tf_bars(base, minutes, SESS)
    tb = tb[tb.day >= win_warm].reset_index(drop=True)
    st = signal_state(ratio(tb.O.values.astype(float), tb.C.values.astype(float), tb.V.values.astype(float), N, "count"), B, S)
    e = bar_end(tb, minutes)
    idx = np.searchsorted(e, end40, side="right") - 1
    return np.where(idx >= 0, st[np.clip(idx, 0, None)], -1)


def prepare(sym: str, base: str = "5m", awake_kind: str = "dst") -> dict:
    win = WIN[base]
    d = load(sym, base)
    tb = tf_bars(d, TF if base == "5m" else 60, SESS)
    tb = tb[tb.day >= win["warm"]].reset_index(drop=True)
    tf_min = TF if base == "5m" else 60
    O, H, L, C, V = (tb[k].values.astype(np.float64) for k in ("O", "H", "L", "C", "V"))
    p10 = ratio(O, C, V, N, "count")
    p20 = ratio(O, C, V, 20, "count")
    s = signal_state(p10, B, S)
    end = bar_end(tb, tf_min)
    flags = {}
    er = {k: efficiency(C, k) for k in (10, 20, 40)}
    fl = {k: flips(s, k) for k in (20, 40, 80)}
    ax = adx(H, L, C, 14)
    d1 = pd.read_parquet(CACHE / f"{sym}_1d.parquet")
    d1 = d1[pd.DatetimeIndex(d1.index).date <= pd.Timestamp("2026-10-22").date()]
    for kind, prm in CLASSIFIERS:
        nm = cname(kind, prm)
        if kind == "ER":
            flags[nm] = er[prm["K"]] >= prm["th"]
        elif kind == "FLIP":
            f = fl[prm["K"]]
            flags[nm] = (f >= 0) & (f < prm["m"])
        elif kind == "EXT":
            pp = p10 if prm["n"] == 10 else p20
            flags[nm] = np.abs(pp - 0.5) >= prm["d"] - 1e-12
        elif kind == "ADX":
            flags[nm] = ax >= prm["th"]
        elif kind == "HTF":
            if prm["tf"] == "D":
                hs = last_daily(daily_state(d1), tb, end)
            else:
                hs = htf_state(d, prm["tf"], end, win["warm"]) if base == "5m" else np.full(len(tb), np.nan)
            flags[nm] = hs == s
        elif kind == "Z5":
            z5 = last_daily(zone5_daily(d1), tb, end)
            flags[nm] = (~np.isnan(z5)) & (z5 != 0)
        flags[nm] = np.nan_to_num(flags[nm].astype(float), nan=0.0).astype(bool)
    z5v = last_daily(zone5_daily(d1), tb, end)
    aw = awake_mask(tb.start) if awake_kind == "dst" else awake_std(tb.start)
    return dict(sym=sym, tb=tb, O=O, H=H, L=L, C=C, V=V, p10=p10, s=s, flags=flags, end=end, z5=z5v,
                seg=segments(tb.day.values, win), awake=aw, lod=tb.lod.values)


def awake_std(start) -> np.ndarray:
    """서머타임 기간 자료에 표준시(11/1 이후) 시간표를 씌운다: 한국 = ET 시계 + 14시간, 08~24시.
    서머타임 자료의 UTC는 ET+4시간이므로 UTC+10시간 = ET+14시간이다(표준시 기간 자료에는 쓰지 않는다)."""
    h = (pd.DatetimeIndex(start).tz_convert("UTC") + pd.Timedelta(hours=10)).hour
    return np.asarray(h >= 8)


@njit(cache=True)
def targets(s, flag, action):
    """봉 마감 뒤 목표 보유. action 0 A0 현금, 1 A1 절반, 2 A2 동결, 3 A3 원안."""
    n = len(s)
    out = np.zeros(n)
    prev = 0.0
    for j in range(n):
        if action == 3:
            t = float(s[j])
        elif flag[j]:
            t = 1.0 if s[j] == 1 else 0.0
        elif action == 0:
            t = 0.0
        elif action == 1:
            t = 0.5
        else:
            t = prev
        out[j] = t
        prev = t
    return out


@njit(cache=True)
def sim_frac(o, c, tgt, seg, awake, cost, pos_out):
    """out: [0:2] 구간 로그수익, [2:4] 평균 보유 비율 분자(합), [4:6] 봉 수, [6:8] 구간 최대 낙폭, [8] 전체 낙폭, [9:11] 바꾼 양."""
    n = len(o)
    out = np.zeros(11)
    carry = 0.0
    target = 0.0
    eq = np.zeros(2); pk = np.zeros(2); eqa = 0.0; pka = 0.0
    for j in range(n):
        pos = target if awake[j] else carry
        sg = seg[j]
        r = 0.0
        dpos = abs(pos - carry)
        if dpos > 0:
            r += np.log(1.0 - cost * dpos)
            if sg >= 0:
                out[9 + sg] += dpos
        nxt = o[j + 1] if j + 1 < n else c[j]
        r += np.log(1.0 + pos * (nxt / o[j] - 1.0))
        carry = pos
        pos_out[j] = pos
        if sg >= 0:
            out[sg] += r
            out[2 + sg] += pos
            out[4 + sg] += 1
            eq[sg] += r
            pk[sg] = max(pk[sg], eq[sg])
            out[6 + sg] = min(out[6 + sg], eq[sg] - pk[sg])
            eqa += r
            pka = max(pka, eqa)
            out[8] = min(out[8], eqa - pka)
        target = tgt[j]
    return out


def static_path(o, c, seg, x, which):
    """같은 비중 그냥 보유: 구간 시작 시가에 x만큼 사서 계속 보유. which = 0·1·'all'."""
    m = seg >= 0 if which == "all" else seg == which
    j = np.flatnonzero(m)
    px = np.r_[o[j], o[j[-1] + 1] if j[-1] + 1 < len(o) else c[j[-1]]]
    eq = 1 + x * (px / px[0] - 1)
    return eq[-1] - 1, float((eq / np.maximum.accumulate(eq) - 1).min()), px[-1] / px[0] - 1, float((px / np.maximum.accumulate(px) - 1).min())


def run_symbol(P: dict, cost: float, ideal: bool) -> pd.DataFrame:
    """전략 85개(분류 28 × 행동 3 + 원안) 성적."""
    O, C, seg = P["O"], P["C"], P["seg"]
    aw = np.ones(len(O), np.bool_) if ideal else P["awake"]
    pos = np.zeros(len(O))
    rows = []
    strategies = [("BASE", None, 3)] + [(cname(k, p), cname(k, p), ai) for k, p in CLASSIFIERS for ai in range(3)]
    for nm, fl, ai in strategies:
        flag = P["flags"][fl] if fl else np.zeros(len(O), np.bool_)
        out = sim_frac(O, C, targets(P["s"], flag, ai), seg, aw, cost, pos)
        r = dict(sym=P["sym"], clf=nm, action="A3" if ai == 3 else ACTIONS[ai], cost=cost, ideal=ideal)
        for w, k in (("sel", 0), ("conf", 1)):
            x = out[2 + k] / out[4 + k]
            st, st_mdd, bh, bh_mdd = static_path(O, C, seg, x, k)
            r.update({f"ret_{w}": float(np.expm1(out[k])), f"mdd_{w}": float(np.expm1(out[6 + k])), f"x_{w}": x,
                      f"static_{w}": st, f"static_mdd_{w}": st_mdd, f"bh_{w}": bh, f"bh_mdd_{w}": bh_mdd, f"turn_{w}": out[9 + k]})
        x = (out[2] + out[3]) / (out[4] + out[5])
        st, st_mdd, bh, bh_mdd = static_path(O, C, seg, x, "all")
        r.update(ret_all=float(np.expm1(out[0] + out[1])), mdd_all=float(np.expm1(out[8])), x_all=x,
                 static_all=st, static_mdd_all=st_mdd, bh_all=bh, bh_mdd_all=bh_mdd)
        rows.append(r)
    return pd.DataFrame(rows)
