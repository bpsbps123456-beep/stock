"""dip_by_zone.py 독립 검증 (2026-10-11, 검증용). 결과는 results/verify_dz_check.txt.
- 자체 3분 재표본·EMA·Pine RSI·상태 기계·전일 +7% 판정·KST 8시 계산으로 V0~V6를 다시 돌린다.
- 단계는 rsi_breakout 1d_long(긴 이력, auto_adjust=False)에 pine_port.pine_zone을 적용, 매수 시각 이전에 끝난 정규장 일봉으로 붙인다.
- SOXX 경고는 dip_raw_1010/1d_SOXX로 따로 계산한다.
- dip_trades.csv와 거래 단위로 맞춰 본다. 거래 목록은 results/verify_dz_my.parquet.
"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa
from pine_port import pine_zone  # noqa

RAW = os.path.join(HERE, "../cache/dip_raw_1010")
RES = os.path.join(HERE, "../results")
ETF = {"ARKG", "IGV", "IWM", "QQQ", "SMH", "SOXX", "SPY", "XBI"}
WIN = {"5분 근사 7/20~9/8": (5, date(2026, 7, 20), date(2026, 9, 8)),
       "3분 정확 9/9~10/9": (3, date(2026, 9, 9), date(2026, 10, 9))}
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    L.append(s)


def my_rsi(c, n):
    d = np.diff(c, prepend=np.nan)
    up, dn = np.maximum(d, 0), np.maximum(-d, 0)
    au, ad = np.full(len(c), np.nan), np.full(len(c), np.nan)
    if len(c) > n:
        au[n], ad[n] = up[1:n + 1].mean(), dn[1:n + 1].mean()
        for i in range(n + 1, len(c)):
            au[i] = (au[i - 1] * (n - 1) + up[i]) / n
            ad[i] = (ad[i - 1] * (n - 1) + dn[i]) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(ad == 0, 100.0, 100 - 100 / (1 + au / ad))
    return np.where(np.isfinite(au), r, np.nan)


def my_ema(c, n):
    a = 2 / (n + 1)
    e = np.empty(len(c))
    e[0] = c[0]
    for i in range(1, len(c)):
        e[i] = a * c[i] + (1 - a) * e[i - 1]
    return e


def load_bars(s, mins):
    if mins == 5:
        b = pd.read_parquet(f"{RAW}/5m_{s}.parquet")
    else:
        m = pd.read_parquet(f"{RAW}/1m_{s}.parquet")
        g = m.groupby(m.index.floor("3min"))
        b = pd.DataFrame({"Open": g.Open.first(), "High": g.High.max(), "Low": g.Low.min(), "Close": g.Close.last()})
        b = b.dropna(subset=["Close"])
    return b[b.High > 0]


def my_sim(o, l, r, ema, utc, gap, hot):
    off = 9 * 3600 * 10**9
    loc = utc + off                                      # KST 벽시계 ns
    day, h8 = 86400 * 10**9, 8 * 3600 * 10**9
    res, pos = [], None
    for i in range(61, len(o)):
        if pos is not None:
            ei, ent, dead = pos
            if r[i - 1] > 70 or utc[i] >= dead:
                why = "RSI" if r[i - 1] > 70 else "8시"
                res.append((ei, i, ent, o[i], why, l[ei:i].min() / ent - 1))
                pos = None
            continue
        if hot[i] or not np.isfinite(gap[i]):
            continue
        lvl = ema[i - 1] * (1 - gap[i])
        if l[i] <= lvl:
            t8 = (loc[i] // day) * day + h8
            dl = t8 if loc[i] < t8 else t8 + day
            pos = (i, min(o[i], lvl), dl - off)
    return res


def zone_at(zs, s, key, aft):
    if s not in zs:
        return -9
    z = zs[s]
    idx = z.index[z.index <= key] if aft else z.index[z.index < key]
    if len(idx) == 0 or not np.isfinite(z.loc[idx[-1]]):
        return -9
    return int(z.loc[idx[-1]])


def main():
    S = sorted(f[3:-8] for f in os.listdir(RAW) if f.startswith("1m_"))
    S = [x for x in S if x not in ETF]
    lg = pd.read_parquet(os.path.join(Zm.ROOT, "data/derived/analysis/rsi_breakout/cache/1d_long.parquet"))
    lg["d"] = pd.to_datetime(lg["ts"]).dt.strftime("%Y-%m-%d")
    LC = lg.pivot_table(index="d", columns="sym", values="Close").sort_index()
    zlong = {}
    for s in S:
        if s in LC:
            c = LC[s].dropna()
            zp, _, _ = pine_zone(c.to_numpy(float))
            zlong[s] = pd.Series(zp, index=c.index)
    d1, zshort = {}, {}
    for s in S:
        d = pd.read_parquet(f"{RAW}/1d_{s}.parquet")
        d.index = pd.to_datetime(d.index).strftime("%Y-%m-%d")
        d1[s] = d
        zp, _, _ = pine_zone(d.Close.to_numpy(float))
        zshort[s] = pd.Series(zp, index=d.index)
    nolong = [s for s in S if s not in zlong]
    for s in nolong:
        zlong[s] = zshort[s]          # 긴 이력에 없는 종목은 dip_raw 1d(1년)로 파인 단계
    sx = pd.read_parquet(f"{RAW}/1d_SOXX.parquet").Close
    sx.index = pd.to_datetime(sx.index).date
    below = (sx < sx.rolling(20, min_periods=20).mean())
    warnwk, prev = {}, None
    for dd in sx.index:
        wk = dd.isocalendar()[:2]
        if wk not in warnwk:
            warnwk[wk] = bool(below.loc[prev]) if prev is not None else False
        prev = dd
    P("dip_raw 1d 마지막 날짜 분포:", pd.Series([d1[s].index[-1] for s in S]).value_counts().head(3).to_dict(),
      "| SOXX 1d 끝:", sx.index[-1], "| 1d_long 끝:", LC.index[-1], "| 1d_long에 없는 종목 수:", len(nolong))
    P("SOXX 경고 주(dip_raw 1d):", {f"{k[0]}-W{k[1]}": v for k, v in warnwk.items() if (2026, 29) <= k <= (2026, 41)})

    rows = []
    for wname, (mins, lo, hi) in WIN.items():
        el, rl = (60, 14) if mins == 3 else (36, 9)
        for s in S:
            try:
                b = load_bars(s, mins)
            except FileNotFoundError:
                continue
            o, l, c = (b[k].to_numpy(float) for k in ("Open", "Low", "Close"))
            et = b.index.tz_convert("America/New_York")
            utc = et.as_unit("ns").asi8
            ema, r = my_ema(c, el), my_rsi(c, rl)
            mod = et.hour * 60 + et.minute
            ses = np.where((mod >= 570) & (mod < 960), "정규", np.where(mod < 570, "프리", "애프터"))
            dc = d1[s].Close
            dates = np.array(dc.index)
            r1 = (dc / dc.shift(1) - 1).to_numpy()
            etd = np.array([x.strftime("%Y-%m-%d") for x in et.date])
            hotd, zpre, zaft = {}, {}, {}
            for x in sorted(set(etd)):
                k = np.searchsorted(dates, x) - 1                 # x보다 앞 마지막 일봉
                hotd[x] = bool(k >= 1 and r1[k] >= 0.07)
                zpre[x], zaft[x] = zone_at(zlong, s, x, False), zone_at(zlong, s, x, True)
            hot = np.array([hotd[x] for x in etd]) & (ses == "정규")
            zone = np.where(ses == "애프터", [zaft[x] for x in etd], [zpre[x] for x in etd])
            warn = np.array([warnwk.get(x.isocalendar()[:2], False) for x in et.date])
            dn, up = np.isin(zone, (-1, -2)), np.isin(zone, (1, 2))
            g = np.full(len(c), 0.02)
            V = {"V0": g, "V1": np.where(dn, np.inf, g), "V2": np.where(dn, 0.03, g), "V3": np.where(dn, 0.04, g),
                 "V4": np.where(warn, np.inf, g), "V5": np.where(warn, 0.03, g), "V6": np.where(up, g, np.inf),
                 "V0_nohot": g}
            for vn, gp in V.items():
                hh = np.zeros(len(c), bool) if vn == "V0_nohot" else hot
                for ei, xi, ent, px, why, mae in my_sim(o, l, r, ema, utc, gp, hh):
                    if lo <= et[ei].date() <= hi:
                        rows.append(dict(win=wname, var=vn, sym=s, t=et[ei], day=et[ei].date(), ses=ses[ei], ent=ent,
                                         ret=px / ent - 1, why=why, mae=mae, zone=int(zone[ei]), warn=bool(warn[ei]),
                                         hot=bool(hot[ei])))
    M = pd.DataFrame(rows)
    M.assign(t=M.t.astype(str), day=M.day.astype(str)).to_parquet(os.path.join(RES, "verify_dz_my.parquet"))

    T = pd.read_csv(os.path.join(RES, "dip_trades.csv"), encoding="utf-8-sig")
    T["t"] = pd.to_datetime(T["t"], utc=True).dt.tz_convert("America/New_York")
    M0 = M[M["var"] == "V0"]
    P("\n== D2 거래 대조(V0): 내 재현 vs dip_trades.csv")
    for w in WIN:
        a, b = M0[M0.win == w], T[T.win == w]
        ka, kb = set(zip(a.sym, a.t)), set(zip(b.sym, b.t))
        mm = a.merge(b, on=["sym", "t"], suffixes=("_m", "_c"))
        P(f" {w}: 내 {len(a)}건 / csv {len(b)}건 / 공통 {len(ka & kb)} / 나만 {len(ka - kb)} / csv만 {len(kb - ka)}")
        P(f"   공통 거래 수익 최대차 {np.abs(mm.ret_m - mm.ret_c).max():.2e}, MAE 최대차 {np.abs(mm.mae_m - mm.mae_c).max():.2e},"
          f" 매도이유 불일치 {(mm.why_m != mm.why_c).sum()}, 단계 불일치(긴 이력 파인 vs csv) {(mm.zone_m != mm.zone_c).sum()}"
          f" (그중 csv '계산 전' {int(((mm.zone_c == -9) & (mm.zone_m != -9)).sum())}), SOXX 경고 불일치 {(mm.warn_m != mm.warn_c).sum()}")
        dz = mm[(mm.zone_m != mm.zone_c) & (mm.zone_c != -9)]
        if len(dz):
            P("   단계 불일치 짝(csv→내):", dz.groupby(["zone_c", "zone_m"]).size().to_dict())
            P("   " + dz[["sym", "t", "ses_m", "zone_m", "zone_c"]].head(10).to_string().replace("\n", "\n   "))
        om = a[[k in (ka - kb) for k in zip(a.sym, a.t)]]
        oc = b[[k in (kb - ka) for k in zip(b.sym, b.t)]]
        if len(om):
            P("   나만:\n   " + om[["sym", "t", "ses", "hot", "ret"]].head(12).to_string().replace("\n", "\n   "))
        if len(oc):
            P("   csv만:\n   " + oc[["sym", "t", "ses", "ret"]].head(12).to_string().replace("\n", "\n   "))
    nh = M[(M["var"] == "V0_nohot") & (M.sym == "TEM") & (M.day == date(2026, 10, 6))]
    P("\nTEM 10/6 hot 끄면:\n" + nh[["t", "ses", "ent", "ret", "why", "mae", "zone"]].to_string())
    P("TEM 10/6 실제 V0:\n" + M0[(M0.sym == "TEM") & (M0.day == date(2026, 10, 6))][["t", "ses", "ent", "ret", "zone", "hot"]].to_string())
    P("TEM 긴 이력 파인 단계 10/1~10/9:", zlong["TEM"].loc["2026-10-01":].to_dict())

    P("\n== 변형 표(내 재현, 24시간): 합 / 건수 / 평균 / 최악 / -5%↓")
    for w in WIN:
        P(" --", w)
        for vn in ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V0_nohot"]:
            x = M[(M.win == w) & (M["var"] == vn)]
            P(f"   {vn:<9} 합 {x.ret.sum() * 100:+8.1f}%  {len(x):5d}건  평균 {x.ret.mean() * 100:+6.2f}%  최악 {x.ret.min() * 100:+6.1f}%"
              f"  -5%↓ {int((x.mae <= -0.05).sum())}")
    with open(os.path.join(RES, "verify_dz_check.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
