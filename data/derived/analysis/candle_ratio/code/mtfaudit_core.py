"""mtfaudit 공통: zones_mtf·zones3·engine을 쓰지 않는 독립 재계산(BRIEF3 정의를 글로 읽고 다시 짬).

- 묶음: 5분봉·1시간봉을 (날짜, (분 − 240) // tf)로 묶는다. 끝 시각은 구성 봉 끝의 최댓값이다.
  1시간봉 끝은 '정시 + 60분, 단 정규장 경계 09:30·16:00·세션 끝 20:00에서 자름'으로 따로 유도한다(코드의 09:00·15:30 특례와 독립).
- 비율·상태·EXT는 정수 개수(양봉 U, 양·음봉 T)로 정확히 판정한다: p > 0.5 ⟺ 2U > T, |p − 0.5| ≥ 0.1 ⟺ T > 0 그리고 5|2U − T| ≥ T.
- ER은 pandas rolling, ADX는 교과서 Wilder(합 평활, 첫 ADX = DX 14개 평균)를 파이썬 반복으로.
- z봉 → b봉 대응은 pandas merge_asof(뒤쪽, 같은 시각 허용) = 'b봉 끝 시각에 이미 끝난 마지막 z봉'.
- 모의매매: 봉 마감 판단 → 다음 봉 시가 체결, awake가 아니면 직전 보유 유지, 비용 = 편도 × |보유 변화|.
"""
from __future__ import annotations

import pathlib
from datetime import date

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent.parent
CACHE = HERE / "cache"
ET = "America/New_York"
LAST = date(2026, 10, 22)
START5 = date(2026, 7, 17)
WARM = date(2026, 6, 1)
SEL = (date(2026, 7, 17), date(2026, 9, 4))
CONF = (date(2026, 9, 8), date(2026, 10, 22))
MIN = 60_000_000_000
BASE_TFS = [10, 20, 30, 40, 60, 90, 120, 180, 240]


def read_raw(sym, cache=CACHE):
    out = {}
    for base in ("5m", "1h"):
        d = pd.read_parquet(cache / f"{sym}_{base}.parquet")
        d.index = pd.DatetimeIndex(d.index).tz_convert(ET)
        out[base] = d[d.index.date <= LAST]
    d1 = pd.read_parquet(cache / f"{sym}_1d.parquet")
    out["1d"] = d1[pd.DatetimeIndex(d1.index).date <= LAST]
    return out


def zone_list(b):
    return [b] + [m * b for m in (2, 4, 8) if m * b <= 960] + ["D"]


# ---------------------------------------------------------------------------
# 묶음
# ---------------------------------------------------------------------------

def group(df, tf, kind):
    idx = pd.DatetimeIndex(df.index)
    mins = np.asarray(idx.hour * 60 + idx.minute)
    keep = (mins >= 240) & (mins < 1200)
    df, idx, mins = df[keep], idx[keep], mins[keep]
    st = idx.as_unit("ns").asi8
    if kind == "5m":
        en = st + 5 * MIN
    else:
        cap = np.where(mins < 570, 570, np.where(mins < 960, 960, 1200))
        en = st + (np.minimum(mins + 60, cap) - mins) * MIN
    f = pd.DataFrame(dict(day=np.asarray(idx.date), k=(mins - 240) // tf, O=df.Open.values.astype(float),
                          H=df.High.values.astype(float), L=df.Low.values.astype(float), C=df.Close.values.astype(float),
                          V=df.Volume.values.astype(float), end=en, first=st))
    out = f.groupby(["day", "k"], sort=True).agg(O=("O", "first"), H=("H", "max"), L=("L", "min"), C=("C", "last"),
                                                  V=("V", "sum"), end=("end", "max"), first=("first", "min"),
                                                  n=("O", "size")).reset_index()
    d0 = pd.DatetimeIndex(pd.to_datetime(out.day.values)).tz_localize(ET)
    out["start"] = (d0 + pd.to_timedelta(240 + out.k.values * tf, unit="m")).as_unit("ns").asi8
    out["warm"] = kind == "1h"
    return out


def bars(raw, tf, warm=True):
    parts = []
    if warm:
        h1 = raw["1h"]
        h1 = h1[(h1.index.date >= WARM) & (h1.index.date < START5)]
        if len(h1):
            parts.append(group(h1, tf, "1h"))
    parts.append(group(raw["5m"], tf, "5m"))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def counts(O, C, n):
    up = pd.Series((C > O).astype(float))
    dn = pd.Series((C < O).astype(float))
    U = up.rolling(n, min_periods=n).sum().values
    T = U + dn.rolling(n, min_periods=n).sum().values
    return U, T


def state(U, T):
    out = np.zeros(len(U), np.int64)
    st = 0
    for j in range(len(U)):
        if not np.isnan(U[j]):
            if st == 0 and 2 * U[j] > T[j]:
                st = 1
            elif st == 1 and 2 * U[j] < T[j]:
                st = 0
        out[j] = st
    return out


def ext(U, T):
    with np.errstate(invalid="ignore"):
        return (~np.isnan(U)) & (T > 0) & (5 * np.abs(2 * U - T) >= T)


def er(C, k=20):
    c = pd.Series(C)
    path = c.diff().abs().rolling(k, min_periods=k).sum()
    num = (c - c.shift(k)).abs()
    v = np.where(path > 0, num / path.replace(0, np.nan), 0.0)
    v = np.where(path.isna(), np.nan, v)
    return v


def adx(H, L, C, n=14):
    m = len(C)
    out = np.full(m, np.nan)
    if m <= 2 * n - 1:
        return out
    tr = [0.0] * m; pdm = [0.0] * m; ndm = [0.0] * m
    for i in range(1, m):
        up, dn = H[i] - H[i - 1], L[i - 1] - L[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        ndm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1]))
    a = sum(tr[1:n + 1]); p = sum(pdm[1:n + 1]); q = sum(ndm[1:n + 1])
    dxs = []
    for i in range(n, m):
        if i > n:
            a = a - a / n + tr[i]; p = p - p / n + pdm[i]; q = q - q / n + ndm[i]
        di_p = 100 * p / a if a > 0 else 0.0
        di_n = 100 * q / a if a > 0 else 0.0
        dx = 100 * abs(di_p - di_n) / (di_p + di_n) if di_p + di_n > 0 else 0.0
        dxs.append(dx)
        if len(dxs) == n:
            cur = sum(dxs) / n
            out[i] = cur
        elif len(dxs) > n:
            cur = (cur * (n - 1) + dx) / n
            out[i] = cur
    return out


def indicators(O, H, L, C):
    U10, T10 = counts(O, C, 10)
    U20, T20 = counts(O, C, 20)
    return dict(state=state(U10, T10).astype(float), er=er(C), adx=adx(list(H), list(L), list(C)), ext=ext(U20, T20))


def daily(raw):
    d1 = raw["1d"]
    O, H, L, C = (d1[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    ind = indicators(O, H, L, C)
    d0 = pd.DatetimeIndex(pd.to_datetime(pd.DatetimeIndex(d1.index).date)).tz_localize(ET)
    ind["end"] = (d0 + pd.Timedelta(hours=16)).as_unit("ns").asi8
    return ind


def asof_index(base_end, z_end):
    left = pd.DataFrame({"e": np.asarray(base_end, np.int64), "i": np.arange(len(base_end))})
    right = pd.DataFrame({"e": np.asarray(z_end, np.int64), "zi": np.arange(len(z_end), dtype=float)})
    m = pd.merge_asof(left.sort_values("e"), right.sort_values("e"), on="e", direction="backward", allow_exact_matches=True)
    return m.sort_values("i").zi.values  # NaN = 아직 끝난 z봉 없음


def pick(v, zi):
    v = np.asarray(v, float)
    ok = ~np.isnan(zi)
    out = np.full(len(zi), np.nan)
    out[ok] = v[zi[ok].astype(int)]
    return out


def build(raw, b, warm=True):
    tb = bars(raw, b, warm)
    O, H, L, C = (tb[k].values for k in ("O", "H", "L", "C"))
    base = indicators(O, H, L, C)
    s = base["state"].astype(np.int64)
    flags, zinfo = {}, {}
    for z in zone_list(b):
        if z == b:
            flags[f"z{b}_ER"] = np.nan_to_num(base["er"], nan=-1) >= 0.3
            flags[f"z{b}_ADX"] = np.nan_to_num(base["adx"], nan=-1) >= 20
            flags[f"z{b}_EXT"] = base["ext"].copy()
            continue
        if z == "D":
            zi_ = daily(raw)
        else:
            zb = bars(raw, z, warm)
            zi_ = indicators(*(zb[k].values for k in ("O", "H", "L", "C")))
            zi_["end"] = zb.end.values
        ix = asof_index(tb.end.values, zi_["end"])
        zinfo[z] = ix
        st = pick(zi_["state"], ix)
        flags[f"z{z}_AGREE"] = (~np.isnan(st)) & (st == s)
        flags[f"z{z}_ER"] = np.nan_to_num(pick(zi_["er"], ix), nan=-1) >= 0.3
        flags[f"z{z}_ADX"] = np.nan_to_num(pick(zi_["adx"], ix), nan=-1) >= 20
        flags[f"z{z}_EXT"] = np.nan_to_num(pick(zi_["ext"].astype(float), ix), nan=0) > 0.5
    day = tb.day.values
    seg = np.full(len(tb), -1)
    seg[(day >= SEL[0]) & (day <= SEL[1])] = 0
    seg[(day >= CONF[0]) & (day <= CONF[1])] = 1
    seg[tb.warm.values] = -1
    kst_h = pd.DatetimeIndex(pd.to_datetime(tb.start.values, utc=True)).tz_convert("Asia/Seoul").hour
    awake = np.asarray(kst_h >= 8)
    return dict(b=b, tb=tb, O=O, C=C, s=s, flags=flags, seg=seg, awake=awake, zinfo=zinfo)


# ---------------------------------------------------------------------------
# 모의매매
# ---------------------------------------------------------------------------

def target(s, flag, action):
    out = np.zeros(len(s))
    prev = 0.0
    for j in range(len(s)):
        if action == "A3":
            t = float(s[j])
        elif flag[j]:
            t = float(s[j] == 1)
        elif action == "A0":
            t = 0.0
        else:  # A2 동결
            t = prev
        out[j] = t
        prev = t
    return out


def positions(tgt, awake):
    pos = np.zeros(len(tgt))
    carry, want = 0.0, 0.0
    for j in range(len(tgt)):
        p = want if awake[j] else carry
        pos[j] = p
        carry = p
        want = tgt[j]
    return pos


def metrics(P, tgt, cost, ideal):
    O, C, seg = P["O"], P["C"], P["seg"]
    aw = np.ones(len(O), bool) if ideal else P["awake"]
    pos = positions(tgt, aw)
    dpos = np.abs(np.diff(np.r_[0.0, pos]))
    nxt = np.r_[O[1:], C[-1]]
    lr = np.log1p(-cost * dpos) + np.log1p(pos * (nxt / O - 1))
    r = {}
    for w, k in (("sel", 0), ("conf", 1)):
        m = seg == k
        e = np.cumsum(lr[m])
        pk = np.maximum.accumulate(np.maximum(e, 0.0))
        x = pos[m].mean()
        j = np.flatnonzero(m)
        px = np.r_[O[j], O[j[-1] + 1] if j[-1] + 1 < len(O) else C[j[-1]]]
        eq = 1 + x * (px / px[0] - 1)
        r.update({f"ret_{w}": float(np.expm1(lr[m].sum())), f"mdd_{w}": float(np.expm1(min(0.0, (e - pk).min()))),
                  f"x_{w}": float(x), f"static_{w}": float(eq[-1] - 1),
                  f"static_mdd_{w}": float((eq / np.maximum.accumulate(eq) - 1).min()),
                  f"bh_{w}": float(px[-1] / px[0] - 1), f"turn_{w}": float(dpos[m].sum())})
    return r, pos


def diag(P, flag, pos):
    O, C, seg = P["O"], P["C"], P["seg"]
    r = np.r_[O[1:] / O[:-1] - 1, C[-1] / O[-1] - 1]
    zf = np.r_[False, flag[:-1]]
    out = {}
    for w, k in (("sel", 0), ("conf", 1)):
        for nm, m in (("trend", zf), ("amb", ~zf)):
            mm = m & (seg == k)
            n = int(mm.sum())
            out[f"{nm}_{w}_n"] = n
            out[f"{nm}_{w}"] = float((pos[mm] * r[mm]).sum() - pos[mm].mean() * r[mm].sum()) if n else np.nan
    return out
