"""tf_extremes.py·tf_detail.py 독립 재계산(검증용, 2026-10-11).

자체 코드: 종목별 1시간 정규장 봉을 '시계 기준'(09:30부터 분 // TF)으로 묶고, 지표·순위·초과 수익을 새로 짠다.
청산은 진입 h거래일 뒤 '같은 시계 시각' 봉 시가(없으면 제외 — 원본은 그날 마지막 봉으로 대체).
앞 구간은 청산일이 5/29 이하인 신호만(h마다 정확히).
정의 묶음: RSI 64 + 볼린저 15 + 칼만식 90 + N봉 수익 z 25 = 194개(StochRSI·52주·연속 18개 제외).
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/verify_tf_recompute.py
결과: results/verify_tf_recompute.txt
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5))
A = os.path.join(ROOT, "data/derived/analysis")
OUT = os.path.join(A, "trend_zones/results/verify_tf_recompute.txt")
NY = "America/New_York"
REG = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
PER = {"앞": ("2026-01-02", "2026-05-29"), "뒤": ("2026-06-01", "2026-10-09"),
       "2025": ("2025-01-02", "2025-12-31"), "2024": ("2024-01-02", "2024-12-31")}
WARM = 130
L = []


def p(s=""):
    print(s)
    L.append(s)


def load_hourly():
    syms = sorted(set(pd.read_parquet(f"{A}/rsi_breakout/cache/1d_long.parquet").sym) - {"QQQ"})
    rb = pd.read_parquet(f"{A}/rsi_breakout/cache/1h.parquet")
    rb["ts"] = pd.to_datetime(rb.ts).dt.tz_convert(NY)
    out = {}
    for s in syms:
        r = rb[rb.sym == s].set_index("ts")[["Open", "High", "Low", "Close"]]
        f = f"{A}/premarket_shock/bars/1h_{s}.parquet"
        if os.path.exists(f):
            a = pd.read_parquet(f)[["Open", "High", "Low", "Close"]]
            a.index = a.index.tz_convert(NY)
            a = a[a.index.strftime("%H:%M").isin(REG)]
            a = pd.concat([a, r[r.index > a.index.max()]])
        else:
            a = r
        out[s] = a[~a.index.duplicated()].sort_index()
    return out


def tf_panel(bars, tf):
    """시계 기준 묶음: key = (분-570)//tf. 반환 O,H,L,C(wide, 행 = (날짜,key)), 각 행의 마지막 기본 봉 시각."""
    parts, lasts = {}, []
    for s, d in bars.items():
        m = d.index.hour * 60 + d.index.minute - 570
        k = pd.MultiIndex.from_arrays([d.index.strftime("%Y-%m-%d"), np.asarray(m // tf)], names=["day", "key"])
        g = d.set_axis(k).assign(ts=d.index).groupby(level=["day", "key"])
        parts[s] = pd.DataFrame({"O": g.Open.first(), "H": g.High.max(), "L": g.Low.min(), "C": g.Close.last()})
        lasts.append(g.ts.max().rename(s))
    big = pd.concat(parts, axis=1).sort_index()
    last = pd.concat(lasts, axis=1).reindex(big.index).max(axis=1)
    W = {f: big.xs(f, axis=1, level=1) for f in "OHLC"}
    return W, last


def rsi(C, n):
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    r = 100 * up / (up + dn)
    r = r.mask((dn == 0) & (up > 0), 100.0).mask((up == 0) & (dn > 0), 0.0)
    return r


def defs(W):
    O, H, Lw, C = W["O"], W["H"], W["L"], W["C"]
    out = []
    for n in (2, 3, 5, 7, 9, 14, 21, 28):
        r = rsi(C, n).to_numpy()
        for hi in (60, 65, 70, 75, 80, 85, 90, 95):
            out.append((f"RSI{n} {hi}/{100 - hi}", r >= hi, r <= 100 - hi))
    for n in (10, 20, 50):
        z = ((C - C.rolling(n).mean()) / C.rolling(n).std(ddof=0)).to_numpy()
        for k in (1.0, 1.5, 2.0, 2.5, 3.0):
            out.append((f"볼린저{n} ±{k}", z >= k, z <= -k))
    pc = C.shift(1)
    tr = np.fmax(np.fmax((H - Lw).to_numpy(), (H - pc).abs().to_numpy()), (Lw - pc).abs().to_numpy())
    atr = pd.DataFrame(tr, index=C.index, columns=C.columns).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    for Ln in (5, 10, 20, 35, 50, 100):
        dev = C - C.ewm(span=Ln, adjust=False, min_periods=Ln).mean()
        wd = {"rms20": np.sqrt((dev ** 2).rolling(20).mean()), "rms60": np.sqrt((dev ** 2).rolling(60).mean()), "atr14": atr}
        for w in ("rms20", "rms60", "atr14"):
            z = (dev / wd[w]).to_numpy()
            for b in (1.0, 1.5, 2.0, 2.5, 3.0):
                out.append((f"칼만식 EMA{Ln}·{w} ±{b}", z >= b, z <= -b))
    lr = np.log(C).diff()
    sd = lr.rolling(60).std()
    for N in (3, 5, 10, 20, 60):
        z = (np.log(C / C.shift(N)) / (sd * np.sqrt(N))).to_numpy()
        for k in (1.0, 1.5, 2.0, 2.5, 3.0):
            out.append((f"{N}일 수익 z ±{k}", z >= k, z <= -k))
    V = ((C.notna().cumsum() >= WARM) & C.notna() & O.notna()).to_numpy()
    return [(nm, t & V, b & V) for nm, t, b in out]


def forward(bars, last, h):
    """진입 = 시간봉 마지막 기본 봉 다음 기본 봉 시가, 청산 = h거래일 뒤 같은 시계 시각 시가(없으면 NaN)."""
    BO = pd.concat({s: d.Open for s, d in bars.items()}, axis=1).sort_index()
    U = pd.DatetimeIndex(BO.index).as_unit("ns")
    pos = np.searchsorted(U.asi8, pd.DatetimeIndex(last).as_unit("ns").asi8, side="right")
    ok = pos < len(U)
    pos = np.minimum(pos, len(U) - 1)
    udays = np.array(U.strftime("%Y-%m-%d"))
    days = sorted(set(udays))
    di = {d: i for i, d in enumerate(days)}
    lookup = {(d, t): i for i, (d, t) in enumerate(zip(udays, U.strftime("%H:%M")))}
    hm = U.strftime("%H:%M")
    ex = np.full(len(pos), -1)
    exday = np.array(["9999"] * len(pos), dtype=object)
    for r, q in enumerate(pos):
        if not ok[r]:
            continue
        j = di[udays[q]] + h
        if j < len(days):
            exday[r] = days[j]
            ex[r] = lookup.get((days[j], hm[q]), -1)
    Bn = BO.to_numpy()
    f = np.where((ex >= 0)[:, None], Bn[np.maximum(ex, 0)] / Bn[pos] - 1, np.nan)
    return f, exday, U[pos]


def crank(f):
    df = pd.DataFrame(f)
    n = df.notna().sum(axis=1)
    return ((df.rank(axis=1) - 0.5).div(n, axis=0) - 0.5).to_numpy()


def daymean(vals, day):
    return pd.Series(vals).groupby(day).mean().dropna()


def score(M, rk, day, sel, sign):
    rs = rk[sel]
    fin = np.isfinite(rs)
    Mr = M[sel] & fin
    n = Mr.sum(axis=1)
    if fin.sum() == 0 or n.sum() == 0:
        return 0.0, np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, np.where(Mr, rs, 0).sum(axis=1) / n, np.nan)
    return Mr.sum() / fin.sum(), sign * daymean(m, day[sel]).mean()


def main():
    bars = load_hourly()
    # T1 점검: NVDA 2026-09-15 40분봉(5분봉에서) 원본 resample vs 시계 기준
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tf_extremes as T
    B5 = T.load_five()
    O, Hh, Lo, C, end, dd = T.resample(B5, 8)
    j = list(B5.O.columns).index("NVDA")
    x = pd.read_parquet(T.FIVE)
    x = x[x.sym == "NVDA"].set_index("ts").loc["2026-09-15"]
    mm = (x.index.hour * 60 + x.index.minute - 570) // 40
    g = x.groupby(np.asarray(mm)).agg(O=("Open", "first"), H=("High", "max"), L=("Low", "min"), C=("Close", "last"))
    sel = dd == "2026-09-15"
    mine = g.to_numpy()
    theirs = np.c_[O.to_numpy()[sel, j], Hh.to_numpy()[sel, j], Lo.to_numpy()[sel, j], C.to_numpy()[sel, j]]
    p(f"T1 5분→40분 NVDA 2026-09-15: 봉 {len(g)}개(원본 {sel.sum()}), 최대 차이 {np.nanmax(np.abs(mine - theirs)):.6f}, "
      f"마지막 봉 시작 {x.index[mm == mm.max()][0].strftime('%H:%M')}~16:00, 5분봉 수 {len(x)}")
    p(f"   원본 40분봉 진입 시각(첫 3개): {[str(B5.O.index[e + 1])[11:16] for e in end[sel][:3]]}")

    for tf in (60, 120, 390):
        W, last = tf_panel(bars, tf)
        day = np.array(W["C"].index.get_level_values("day"))
        D = defs(W)
        R, EXD, F = {}, {}, {}
        for h in (1, 5, 10):
            f, exday, _ = forward(bars, last, h)
            F[h], R[h], EXD[h] = f, crank(f), exday
        res = {}
        for nm, t, b in D:
            r = {}
            for h in (1, 5, 10):
                for p_, (a, z) in PER.items():
                    s = (day >= a) & (day <= z)
                    if p_ == "앞":
                        s &= np.array([e <= z for e in EXD[h]])
                    for side, M, sg in (("위", t, -1), ("아래", b, 1)):
                        sh, v = score(M, R[h], day, s, sg)
                        r[(side, h, p_)] = v if 0.01 <= sh <= 0.20 else None
            res[nm] = r
        p(f"\n== {tf}분 (행 {len(day)}, 날 {len(set(day))}) — 자격 정의 중 타당도>0 비율 · 중앙(순위 %p) · n")
        for side in ("위", "아래"):
            for h in (1, 5, 10):
                cells = []
                for p_ in PER:
                    v = np.array([r[(side, h, p_)] for r in res.values() if r[(side, h, p_)] is not None])
                    cells.append(f"{p_} {np.mean(v > 0) * 100:3.0f}% {np.median(v) * 100:+5.2f} n{len(v)}" if len(v) else f"{p_} —")
                allp = sum(1 for r in res.values() if all(r[(side, h, q)] is not None and r[(side, h, q)] > 0 for q in PER))
                p(f"  {side} h{h:<2}| " + " | ".join(cells) + f" | 네 구간 모두 + {allp}")
        for nm in ("RSI14 70/30", "RSI2 95/5", "RSI3 90/10", "볼린저20 ±2.0", "볼린저50 ±1.5", "칼만식 EMA20·atr14 ±2.0", "칼만식 EMA50·rms60 ±1.5"):
            r = res[nm]
            f = lambda v: "  —  " if v is None else f"{v * 100:+5.2f}"
            p(f"  {nm:<24} 위 h5 " + " ".join(f(r[('위', 5, q)]) for q in PER) + " | 아래 h5 " + " ".join(f(r[('아래', 5, q)]) for q in PER)
              + " | 위 h10 " + " ".join(f(r[('위', 10, q)]) for q in PER))
        # T4 수익 환산(5일)
        if tf in (60, 120):
            f = F[5]
            fdf = pd.DataFrame(f)
            lo, hi = fdf.quantile(0.01, axis=1), fdf.quantile(0.99, axis=1)
            exc = fdf.clip(lower=lo, upper=hi, axis=0)
            exc = exc.sub(exc.mean(axis=1), axis=0).to_numpy()
            excu = fdf.sub(fdf.mean(axis=1), axis=0).to_numpy()
            p(f"  -- {tf}분 5일 수익: 초과(1·99% 자름) / 초과(안 자름) / 상승확률 / 동료 대비 뒤짐 비율 / 초과 중앙(풀) / 초과>+10% 비율 / 신호 수")
            Dd = {nm: (t, b) for nm, t, b in D}
            for nm in ("RSI14 70/30", "볼린저20 ±2.0", "칼만식 EMA20·atr14 ±2.0"):
                for side, M in (("극상방", Dd[nm][0]), ("극하방", Dd[nm][1])):
                    cells = []
                    for p_, (a, z) in PER.items():
                        s = (day >= a) & (day <= z)
                        Mr = M[s] & np.isfinite(exc[s])
                        n = Mr.sum(axis=1)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            e1 = np.where(n > 0, np.where(Mr, exc[s], 0).sum(1) / n, np.nan)
                            e2 = np.where(n > 0, np.where(Mr, excu[s], 0).sum(1) / n, np.nan)
                            up = np.where(n > 0, (Mr & (f[s] > 0)).sum(1) / n, np.nan)
                            lag = np.where(n > 0, (Mr & (excu[s] < 0)).sum(1) / n, np.nan)
                        pool = excu[s][Mr]
                        cells.append(f"{p_} {daymean(e1, day[s]).mean() * 100:+.2f}/{daymean(e2, day[s]).mean() * 100:+.2f}/"
                                     f"{daymean(up, day[s]).mean() * 100:.0f}/{daymean(lag, day[s]).mean() * 100:.0f}/"
                                     f"{np.median(pool) * 100:+.2f}/{np.mean(pool > 0.10) * 100:.1f}/{Mr.sum()}")
                    p(f"  {nm:<22}{side} " + " | ".join(cells))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
