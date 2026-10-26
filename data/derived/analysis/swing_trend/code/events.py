"""BRIEF2 사건표: 실시간 지그재그 확정 사건 × 단서 18개 × 결과(y1·y10·y20).

py code/events.py → results/events_ZZA3.parquet, results/events_ZZP10.parquet
모든 단서는 확정일 종가까지의 정보다. 결과는 확정 다음 날 시가 체결 기준이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing import HERE, PERIODS, atr, load, zigzag

ETF = {"SPY", "QQQ", "SOXX", "SMH", "IGV", "IWM", "XBI", "ARKG"}
DETS = {"ZZA3": dict(k=3.0), "ZZP10": dict(th=0.10)}
FEATURES = ["M1", "M2", "M3a", "M3b", "M4a", "M4b", "M4c", "M5",
            "S1", "S2", "S3a", "S3b", "S4", "S5", "S6", "S7", "S8", "S9"]


def period_of(d):
    for k, (a, b) in PERIODS.items():
        if a <= d <= b:
            return k
    return None


def det_state(g, det):
    O, H, L, C = (g[k].values.astype(np.float64) for k in ("O", "H", "L", "C"))
    a20 = atr(H, L, C, 20)
    if "k" in DETS[det]:
        return zigzag(C, 0.0, DETS[det]["k"] * a20), a20
    return zigzag(C, DETS[det]["th"], np.zeros(0)), a20


def reaction_days(earn: pd.DataFrame, dates: np.ndarray) -> set:
    """실적 반응 거래일: 장 마감 뒤(amc) 발표는 다음 거래일, 그 밖은 그날(그날이 휴장이면 다음 거래일)."""
    out = set()
    dd = np.array(dates)
    for _, r in earn.iterrows():
        i = np.searchsorted(dd, r.date_et, side="left")
        if r.session == "amc" and i < len(dd) and dd[i] == r.date_et:
            i += 1
        if i < len(dd):
            out.add(dd[i])
    return out


def build(det: str) -> pd.DataFrame:
    D = load()
    vix = pd.read_parquet(HERE / "cache" / "vix.parquet")
    earn = pd.read_parquet(HERE / "cache" / "earnings.parquet")
    stocks = sorted(set(D) - ETF)
    # 공통 달력과 날짜 × 종목 표
    cal = sorted(set().union(*[set(D[s].date) for s in stocks]))
    ix = {d: i for i, d in enumerate(cal)}
    T, Nn = len(cal), len(stocks)
    conf = np.zeros((T, Nn), np.int8)           # 그날 확정 방향
    above20 = np.full((T, Nn), np.nan)
    tmr = np.full((T, Nn), np.nan)               # 추세 − 뒤집기 일 손익(종가→종가, pos = 전날 상태)
    info = {}
    for j, s in enumerate(stocks):
        g = D[s]
        (st, pi, pc, pt), a20 = det_state(g, det)
        C = g.C.values.astype(float)
        rows = np.array([ix[d] for d in g.date])
        for c_, t_ in zip(pc, pt):
            conf[rows[c_], j] = t_
        sma = pd.Series(C).rolling(20).mean().values
        above20[rows, j] = np.where(np.isnan(sma), np.nan, (C > sma).astype(float))
        r = np.r_[np.nan, C[1:] / C[:-1] - 1]
        pos = np.r_[np.nan, st[:-1]].astype(float)
        tmr[rows, j] = (2 * pos - 1) * r
        info[s] = (g, st, pi, pc, pt, a20, rows)
    # 시장 단서
    have = ~np.isnan(above20)
    M2 = np.nanmean(above20, axis=1)
    up3 = pd.DataFrame((conf == 1).astype(float)).rolling(3, min_periods=1).sum().values > 0
    dn3 = pd.DataFrame((conf == -1).astype(float)).rolling(3, min_periods=1).sum().values > 0
    nhave = have.sum(axis=1).clip(min=1)
    M1_up = (up3 & have).sum(axis=1) / nhave
    M1_dn = (dn3 & have).sum(axis=1) / nhave
    M5 = pd.Series(np.nanmean(tmr, axis=1)).rolling(40, min_periods=30).sum().values
    sx = D["SOXX"].set_index("date").C.astype(float).reindex(cal).ffill()
    M3a = (sx > sx.rolling(50).mean()).astype(float).values
    M3b = np.log(sx / sx.shift(20)).values
    v = vix.reindex(cal).ffill()
    M4a = v.VIX.values
    M4b = np.log(v.VIX / v.VIX.shift(5)).values
    M4c = (v.VIX / v.VIX3M).values
    out = []
    for s in stocks:
        g, st, pi, pc, pt, a20, rows = info[s]
        O, H, L, C, V = (g[k].values.astype(np.float64) for k in ("O", "H", "L", "C", "V"))
        dates = g.date.values
        n = len(C)
        rd = reaction_days(earn[earn.sym == s], dates)
        is_earn = np.array([d in rd for d in dates])
        lr = np.r_[np.nan, np.log(C[1:] / C[:-1])]
        vavg = pd.Series(V).rolling(20).mean().shift(1).values
        vrat = V / vavg
        hi252 = pd.Series(H).rolling(252, min_periods=120).max().values
        atrp = a20 / C
        atr_pct = pd.Series(atrp).rolling(252, min_periods=120).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True).values
        sxr = M3b[rows]
        for e, (p, c, t) in enumerate(zip(pi, pc, pt)):
            d = dates[c]
            per = period_of(d)
            if per is None or c < 260 or c + 1 >= n:
                continue
            ri = rows[c]
            # 결과
            nxt = [k for k in range(e + 1, len(pc)) if pt[k] == -t]
            c2 = pc[nxt[0]] if nxt else None
            exit_px = O[c2 + 1] if (c2 is not None and c2 + 1 < n) else C[-1]
            y1 = t * np.log(exit_px / O[c + 1])
            y10 = t * np.log(O[c + 11] / O[c + 1]) if c + 11 < n else np.nan
            y20 = t * np.log(O[c + 21] / O[c + 1]) if c + 21 < n else np.nan
            p_prev = pi[e - 1] if e > 0 else None
            # 종목 단서
            win = lr[max(c - 59, 1):c + 1]
            ac = sum(pd.Series(win).autocorr(lag=k) for k in range(1, 6)) if len(win) > 30 else np.nan
            feats = dict(
                M1=(M1_up if t == 1 else M1_dn)[ri], M2=M2[ri], M3a=M3a[ri], M3b=M3b[ri], M4a=M4a[ri], M4b=M4b[ri], M4c=M4c[ri], M5=M5[ri],
                S1=c - p,
                S2=abs(C[p] - C[p_prev]) / a20[p] if p_prev is not None and a20[p] > 0 else np.nan,
                S3a=vrat[c], S3b=np.nanmax(vrat[p:c + 1]) if c >= p else np.nan,
                S4=t * (O[c] - C[c - 1]) / a20[c - 1] if a20[c - 1] > 0 else np.nan,
                S5=lr[c - 19:c + 1].sum() - sxr[c] if c >= 20 else np.nan,
                S6=float(is_earn[p:c + 1].any()),
                S7=ac, S8=C[c] / hi252[c] - 1, S9=atr_pct[c],
            )
            out.append(dict(sym=s, det=det, type=int(t), date=d, period=per, pivot_date=dates[p], conf_idx=int(c),
                            exit_date=dates[c2] if c2 is not None else None, censored=c2 is None or c2 + 1 >= n,
                            y1=y1, y10=y10, y20=y20, **feats))
    return pd.DataFrame(out)


if __name__ == "__main__":
    for det in DETS:
        ev = build(det)
        ev.to_parquet(HERE / "results" / f"events_{det}.parquet")
        print(det, len(ev), "사건", ev.groupby(["period", "type"]).size().unstack().to_dict(), "날짜 수", ev.groupby("period").date.nunique().to_dict())
        print(ev[FEATURES].describe().T[["count", "mean", "50%"]].round(3).to_string())
