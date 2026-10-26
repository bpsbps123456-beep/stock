"""추세 5단계 가운데 판정 대안 13개 비교 (impl_B, 독립 재현용). 저장소 파일은 읽기·import만 한다.

실행: PYTHONIOENCODING=utf-8 py compare.py
결과: 같은 폴더 results.json
"""
import json
import os
import sys

sys.dont_write_bytecode = True  # 저장소 code/ 폴더에 .pyc를 만들지 않는다

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.abspath(os.path.join(HERE, "..", "panel_1d_20261012.parquet"))
ZCODE = "C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/trend_zones/code"
sys.path.insert(0, ZCODE)
import zones as ZM  # noqa: E402  (Ind, Fwd, period_rows, evaluate)

MIN_BARS = 130
P_ALL = ("2024-01-02", "2026-10-12")
P_CAL = ("2024-01-02", "2025-12-31")
P_26 = ("2026-01-02", "2026-10-12")
P_LY = ("2025-10-13", "2026-10-12")
K3MUL = np.sqrt(19 * 39 / 120)


# ------------------------------------------------------------------ 자료
def load():
    x = pd.read_parquet(PANEL)
    x["d"] = pd.to_datetime(x["ts"]).dt.strftime("%Y-%m-%d")
    etf = sorted(x.loc[x.is_etf, "sym"].unique())
    f = {k: x.pivot_table(index="d", columns="sym", values=k).sort_index() for k in ("Open", "High", "Low", "Close")}
    return f, etf


def rsi_pine(c, n=14):
    """파인 ta.rsi와 같게: RMA, 첫 14개 변화의 단순평균으로 시작."""
    out = np.full(len(c), np.nan)
    idx = np.where(np.isfinite(c))[0]
    if len(idx) < n + 1:
        return out
    s = idx[0]
    d = np.diff(c[s:])
    u, dn = np.maximum(d, 0), np.maximum(-d, 0)
    au, ad = u[:n].mean(), dn[:n].mean()

    def val(a, b):
        if b == 0:
            return 100.0 if a > 0 else np.nan
        return 100 - 100 / (1 + a / b)

    out[s + n] = val(au, ad)
    for k in range(n, len(d)):
        au = (au * (n - 1) + u[k]) / n
        ad = (ad * (n - 1) + dn[k]) / n
        out[s + k + 1] = val(au, ad)
    return out


# ------------------------------------------------------------------ 지표 값
def features(C, ddof=1):
    lC = np.log(C)
    lr = lC.diff()
    sd20 = lr.rolling(20, min_periods=20).std(ddof=ddof)
    sd10 = lr.rolling(10, min_periods=10).std(ddof=ddof)
    rsi = pd.DataFrame({s: rsi_pine(C[s].to_numpy()) for s in C.columns}, index=C.index)
    T0 = (lC - lC.shift(20)) / (sd20 * np.sqrt(20))
    T2 = sum((lC - lC.shift(N)) / (sd20 * np.sqrt(N)) for N in (5, 10, 15, 20)) / 4
    T3 = (lC - lC.rolling(20, min_periods=20).mean()) / (sd20 * K3MUL)
    # OLS 기울기: 창 안 위치 x=0(19일 전)..19(오늘), b = Σ(x-9.5)·y / 665
    w = np.arange(20) - 9.5
    b = sum(w[k] * lC.shift(19 - k) for k in range(20)) / float((w ** 2).sum())
    T4 = b * 19 / sd20
    ema10 = C.ewm(span=10, adjust=False).mean()
    T5 = (lC - np.log(ema10)) / sd20
    T7 = (lC - lC.shift(10)) / (sd10 * np.sqrt(10))
    mx10, mn10 = C.rolling(10, min_periods=10).max(), C.rolling(10, min_periods=10).min()
    dn = np.log(C / mx10)
    up = np.log(C / mn10)
    u10 = sd20 * np.sqrt(10)
    nb = C.notna().cumsum()
    V = (nb >= MIN_BARS) & C.notna()
    for a in (rsi, sd20, sd10, T0, T2, T3, T4, T5, T7, dn, up):
        V &= np.isfinite(a)
    V &= (sd20 > 0) & (sd10 > 0)
    return dict(lC=lC, lr=lr, sd20=sd20, sd10=sd10, rsi=rsi, T0=T0, T2=T2, T3=T3, T4=T4, T5=T5, T7=T7,
                dn=dn, up=up, u10=u10, V=V, mx10=mx10, mn10=mn10,
                mx20=C.rolling(20, min_periods=20).max(), mn20=C.rolling(20, min_periods=20).min())


# ------------------------------------------------------------------ 단계
def five(F, T, tau):
    r, t = F["rsi"].to_numpy(), T.to_numpy()
    with np.errstate(invalid="ignore"):
        z = np.select([r >= 70, r <= 30, t >= tau, t <= -tau], [2, -2, 1, -1], 0)
    return np.where(F["V"].to_numpy(), z, -9).astype(np.int8)


def seven(F, T, a, b):
    r, t = F["rsi"].to_numpy(), T.to_numpy()
    with np.errstate(invalid="ignore"):
        z = np.select([r >= 70, r <= 30, t >= b, t <= -b, t >= a, t <= -a], [3, -3, 2, -2, 1, -1], 0)
    return np.where(F["V"].to_numpy(), z, -9).astype(np.int8)


def leg_override(F, Z0, m):
    """K6: K0 단계(극단 아님)에 최근 10일 다리를 덮어쓴다."""
    dn, up, u = F["dn"].to_numpy(), F["up"].to_numpy(), F["u10"].to_numpy()
    with np.errstate(invalid="ignore"):
        cdn = dn <= -m * u
        cup = up >= m * u
    both = cdn & cup
    z = Z0.copy().astype(np.int16)
    nonext = (Z0 != -9) & (np.abs(Z0) != 2)
    z = np.where(nonext & cdn & ~cup, -1, z)
    z = np.where(nonext & cup & ~cdn, 1, z)
    with np.errstate(invalid="ignore"):
        z = np.where(nonext & both, np.where(-dn > up, -1, 1), z)  # 같으면 상방(사실상 없음)
    return z.astype(np.int8)


def calibrate(F, T, rows_mask, target):
    t = np.abs(T.to_numpy()[rows_mask])
    lo, hi = 0.0, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.mean(t < mid) < target:
            lo = mid
        else:
            hi = mid
    tau = hi
    return float(tau), float(np.mean(t < tau))


# ------------------------------------------------------------------ 지표(평가)
def pmask(dates, p):
    a, b = p
    return np.array([a <= d <= b for d in dates])


def m0(Z, colmask, dates, p):
    rm = pmask(dates, p)
    zz = Z[rm][:, colmask]
    v = zz != -9
    return float((zz[v] == 0).mean())


def m1(Z, F, colmask, dates, p):
    rm = pmask(dates, p)
    lr = F["lr"].to_numpy()[:, colmask]
    sd = F["sd20"].to_numpy()[:, colmask]
    z = Z[:, colmask].astype(np.int16)
    zt, zp = z[1:], z[:-1]
    ok = (zt != -9) & (zp != -9) & rm[1:, None]
    ch = ok & (zt != zp)
    d = np.sign(zt - zp)
    r = lr[1:]
    with np.errstate(invalid="ignore"):
        rev = ch & (d * np.sign(r) < 0)
        strong = rev & (np.abs(r) >= sd[1:])
    vdays = ((z != -9) & rm[:, None]).sum()
    sy = vdays / 252
    n_ch, n_rev, n_st = int(ch.sum()), int(rev.sum()), int(strong.sum())
    return {"changes": n_ch, "reversals": n_rev, "strong_reversals": n_st, "valid_days": int(vdays),
            "sym_years": float(sy), "rev_share": n_rev / n_ch if n_ch else float("nan"),
            "changes_per_sy": n_ch / sy, "rev_per_sy": n_rev / sy, "strong_rev_per_sy": n_st / sy}


def events(F, C):
    Cn = C.to_numpy()
    sd = F["sd20"].to_numpy()
    mx10, mn10, mx20, mn20 = (F[k].to_numpy() for k in ("mx10", "mn10", "mx20", "mn20"))
    k = 1.5 * sd * np.sqrt(10)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "dn_abs": Cn / mx10 - 1 <= -0.10,
            "dn_vol": np.log(Cn / mx10) <= -k,
            "dn20": (Cn / mx20 - 1 <= -0.15) & (mx10 == mx20),
            "up_abs": Cn / mn10 - 1 >= 0.10,
            "up_vol": np.log(Cn / mn10) >= k,
            "up20": (Cn / mn20 - 1 >= 0.15) & (mn10 == mn20),
        }


def m2(Z, E, colmask, dates, p):
    rm = pmask(dates, p)
    out = {}
    for name, e in E.items():
        z = Z[:, colmask]
        sel = e[:, colmask] & (z != -9) & rm[:, None]
        zz = z[sel]
        n = len(zz)
        out[name] = {"n": int(n),
                     "down": float((zz < 0).mean()) if n else float("nan"),
                     "gray": float((zz == 0).mean()) if n else float("nan"),
                     "up": float((zz > 0).mean()) if n else float("nan")}
    return out


def m3(Z, colmask, dates, p):
    rm = pmask(dates, p)
    z = Z[:, colmask].astype(np.int16)
    zt, zp = z[1:], z[:-1]
    ok = (zt != -9) & (zp != -9) & rm[1:, None]
    ch = (ok & (zt != zp))
    vd = ((z != -9) & rm[:, None])
    rate = ch.sum() / (vd.sum() / 252)
    per_sym = []
    runs = []
    for j in range(z.shape[1]):
        nv = vd[:, j].sum()
        if nv >= 126:
            per_sym.append(ch[:, j].sum() / (nv / 252))
        cur, ln = None, 0
        for i in np.where(rm)[0]:
            v = z[i, j]
            if v == -9:
                if ln:
                    runs.append(ln)
                cur, ln = None, 0
                continue
            if v == cur:
                ln += 1
            else:
                if ln:
                    runs.append(ln)
                cur, ln = v, 1
        if ln:
            runs.append(ln)
    return {"flips_per_sy": float(rate), "flips_per_sym_median": float(np.median(per_sym)) if per_sym else float("nan"),
            "median_run": float(np.median(runs)) if runs else float("nan"), "n_runs": len(runs),
            "mean_run": float(np.mean(runs)) if runs else float("nan")}


def m4(Z, C, colmask, dates, p, crash=True):
    rm = pmask(dates, p)
    Cn = C.to_numpy()[:, colmask]
    z = Z[:, colmask]
    nD = Cn.shape[0]
    prev = C.shift(1)
    if crash:
        ref = prev.rolling(15, min_periods=15).max().to_numpy()[:, colmask]
        with np.errstate(invalid="ignore"):
            cond = Cn <= 0.85 * ref
    else:
        ref = prev.rolling(15, min_periods=15).min().to_numpy()[:, colmask]
        with np.errstate(invalid="ignore"):
            cond = Cn >= 1.15 * ref
    lags, early, reached = [], [], []
    for j in range(Cn.shape[1]):
        for q in range(17, nD):
            if not (cond[q, j] and not cond[q - 1, j] and rm[q]):
                continue
            if q + 20 > nD - 1:
                continue  # 지연 창(q+20)이 자료 끝을 넘으면 제외
            w = Cn[q - 15:q, j]
            ext = w.max() if crash else w.min()
            p_ = q - 15 + int(np.where(w == ext)[0][-1])  # 같은 값이면 가장 최근
            if z[p_, j] == -9:
                continue
            seg = z[p_ + 1:q + 21, j]
            hit = np.where((seg < 0) if crash else ((seg > 0) & (seg != -9)))[0]
            hit = hit[seg[hit] != -9]
            if len(hit):
                d = p_ + 1 + int(hit[0])
                lags.append(d - p_)
                early.append(d <= q)
                reached.append(True)
            else:
                lags.append(q + 20 - p_ + 1)  # 미도달: 창 끝 다음 값으로 넣는다
                early.append(False)
                reached.append(False)
    n = len(lags)
    return {"n": n, "median_lag": float(np.median(lags)) if n else float("nan"),
            "early": float(np.mean(early)) if n else float("nan"),
            "reached": float(np.mean(reached)) if n else float("nan")}


def fold(Z, seven_level):
    if not seven_level:
        return Z
    z = Z.astype(np.int16)
    out = np.where(z == -9, -9, np.sign(z) * np.where(np.abs(z) == 3, 2, np.where(z == 0, 0, 1)))
    return out.astype(np.int8)


# ------------------------------------------------------------------ 실행
def run(f, etf, ddof=1, verbose=True):
    C = f["Close"]
    dates = list(C.index)
    syms = list(C.columns)
    stock = np.array([s not in etf for s in syms])
    etfm = ~stock
    F = features(C, ddof)

    # --- 재현 확인
    j = syms.index("EWY")
    chk = {}
    for d in ("2026-10-09", "2026-10-12"):
        i = dates.index(d)
        chk[d] = {"T0": float(F["T0"].iat[i, j]), "RSI": float(F["rsi"].iat[i, j]), "C": float(C.iat[i, j])}
    print("재현 EWY", chk)

    # --- 보정
    rm_cal = pmask(dates, P_CAL)
    Vn = F["V"].to_numpy()
    r = F["rsi"].to_numpy()
    nonext = Vn & (r > 30) & (r < 70) & rm_cal[:, None] & stock[None, :]
    target = float(np.mean(np.abs(F["T0"].to_numpy()[nonext]) < 0.5))
    cal = {}
    for k in ("T2", "T3", "T4", "T5"):
        tau, ratio = calibrate(F, F[k], nonext, target)
        cal[k] = {"tau": tau, "gray_nonext": ratio}
    print("보정 목표", target, cal)

    Z0 = five(F, F["T0"], 0.5)
    C_ = {
        "K0": (Z0, "T0", 0.5, False),
        "K1a": (seven(F, F["T0"], 0.25, 0.75), "T0", 0.25, True),
        "K1b": (seven(F, F["T0"], 0.5, 1.0), "T0", 0.5, True),
        "K2n": (five(F, F["T2"], 0.5), "T2", 0.5, False),
        "K2c": (five(F, F["T2"], cal["T2"]["tau"]), "T2", cal["T2"]["tau"], False),
        "K3n": (five(F, F["T3"], 0.5), "T3", 0.5, False),
        "K3c": (five(F, F["T3"], cal["T3"]["tau"]), "T3", cal["T3"]["tau"], False),
        "K4c": (five(F, F["T4"], cal["T4"]["tau"]), "T4", cal["T4"]["tau"], False),
        "K5c": (five(F, F["T5"], cal["T5"]["tau"]), "T5", cal["T5"]["tau"], False),
        "K6a": (leg_override(F, Z0, 1.0), "T0", 0.5, False),
        "K6b": (leg_override(F, Z0, 1.5), "T0", 0.5, False),
        "K7": (five(F, F["T7"], 0.5), "T7", 0.5, False),
        "K8": (seven(F, F["T3"], 0.25, 0.75), "T3", 0.25, True),
    }

    # M5 준비 (zones.py 엔진 그대로, 새 패널 150종목)
    Os, Hs, Ls, Cs = (f[k].loc[:, stock] for k in ("Open", "High", "Low", "Close"))
    I = ZM.Ind(Os, Hs, Ls, Cs)
    assert list(I.dates) == dates and list(I.syms) == [s for s in syms if s not in etf]
    Fw = ZM.Fwd(I, 10)
    prow = {p: ZM.period_rows(I, p, 10) for p in ("앞", "뒤", "2025")}

    E = events(F, C)
    out = {}
    for name, (Z, tk, tau, sev) in C_.items():
        res = {"T": tk, "tau": tau, "seven": sev}
        res["gray_2024_25"] = m0(Z, stock, dates, P_CAL)
        res["gray_2026"] = m0(Z, stock, dates, P_26)
        res["M1"] = m1(Z, F, stock, dates, P_ALL)
        res["M2"] = m2(Z, E, stock, dates, P_ALL)
        res["M3"] = m3(Z, stock, dates, P_LY)
        res["M4_crash"] = m4(Z, C, stock, dates, P_ALL, crash=True)
        res["M4_surge"] = m4(Z, C, stock, dates, P_ALL, crash=False)
        Zf = fold(Z, sev)[:, stock]
        m5 = {}
        for p in ("앞", "뒤", "2025"):
            e = ZM.evaluate(Zf, Fw, prow[p], full=True)
            m5[p] = {"S": e["S"] * 100, "terms_pp": {k: v * 100 for k, v in e["terms"].items()},
                     "t": e["t"], "share": {str(k): v for k, v in e["share"].items()}}
        res["M5"] = m5
        res["ETF"] = {"M1": m1(Z, F, etfm, dates, P_ALL), "M2": m2(Z, E, etfm, dates, P_ALL),
                      "M3": m3(Z, etfm, dates, P_LY)}
        out[name] = res
        if verbose:
            print(name, f"gray24-25 {res['gray_2024_25']:.3f} rev {res['M1']['rev_share']:.3f} "
              f"chg/sy {res['M1']['changes_per_sy']:.1f} dn_abs↓ {res['M2']['dn_abs']['down']:.3f} "
                  f"S앞 {m5['앞']['S']:+.2f} S뒤 {m5['뒤']['S']:+.2f}")

    # --- M6 사례
    def case(sym, a, b):
        j = syms.index(sym)
        rows = [i for i, d in enumerate(dates) if a <= d <= b]
        tab = []
        for i in rows:
            row = {"date": dates[i], "close": round(float(C.iat[i, j]), 4), "rsi": round(float(F["rsi"].iat[i, j]), 2)}
            for k in ("T0", "T2", "T3", "T4", "T5", "T7"):
                row[k] = round(float(F[k].iat[i, j]), 4)
            u = float(F["u10"].iat[i, j])
            row["dn_over_u"] = round(float(F["dn"].iat[i, j]) / u, 4)
            row["up_over_u"] = round(float(F["up"].iat[i, j]) / u, 4)
            row["stage"] = {n: int(C_[n][0][i, j]) for n in C_}
            tab.append(row)
        return tab

    ewy = case("EWY", "2026-09-08", "2026-10-12")
    mu = case("MU", "2026-06-22", "2026-08-14")
    summ = {}
    for n, (Z, tk, tau, sev) in C_.items():
        e12 = [r for r in ewy if r["date"] == "2026-10-12"][0]
        first = next((r["date"] for r in ewy if r["date"] > "2026-10-02" and r["stage"][n] < 0 and r["stage"][n] != -9), "none")
        mj = [r for r in mu if "2026-06-26" <= r["date"] <= "2026-07-28"]
        summ[n] = {"ewy_1012_level": e12["stage"][n], "ewy_1012_T": e12[tk], "ewy_first_down_after_1002": first,
                   "mu_jul_down_days": sum(1 for r in mj if r["stage"][n] < 0),
                   "mu_jul_up_days": sum(1 for r in mj if r["stage"][n] > 0),
                   "mu_jul_gray_days": sum(1 for r in mj if r["stage"][n] == 0), "mu_jul_days": len(mj)}
        out[n]["M6"] = summ[n]
        if verbose:
            print(n, summ[n])

    return {"ddof": ddof, "repro_EWY": chk, "calibration": {"target_gray_nonext_K0_2024_25": target, **cal},
           "n_stocks": int(stock.sum()), "etfs": etf, "candidates": out,
           "M6_tables": {"EWY": ewy, "MU": mu}}


def compact(r):
    out = {}
    for n, v in r["candidates"].items():
        out[n] = {"tau": v["tau"], "gray_2024_25": v["gray_2024_25"], "gray_2026": v["gray_2026"],
                  "rev_share": v["M1"]["rev_share"], "changes_per_sy": v["M1"]["changes_per_sy"],
                  "dn_abs_down": v["M2"]["dn_abs"]["down"], "up_abs_up": v["M2"]["up_abs"]["up"],
                  "flips_last_year": v["M3"]["flips_per_sy"], "peak_median_lag": v["M4_crash"]["median_lag"],
                  "peak_early": v["M4_crash"]["early"],
                  "S": {p: v["M5"][p]["S"] for p in v["M5"]},
                  "T1": {p: v["M5"][p]["terms_pp"]["T1 상방-하방"] for p in v["M5"]}, **v["M6"]}
    return out


def main():
    f, etf = load()
    res = run(f, etf, 1)
    s0 = run(f, etf, 0, verbose=False)
    res["sensitivity_ddof0"] = {"repro_EWY": s0["repro_EWY"], "calibration": s0["calibration"],
                                "EWY_K0_stage": {r["date"]: r["stage"]["K0"] for r in s0["M6_tables"]["EWY"]},
                                "candidates": compact(s0)}
    print("ddof0 재현", s0["repro_EWY"])
    for n, v in res["sensitivity_ddof0"]["candidates"].items():
        print("ddof0", n, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k not in ("S", "T1")})
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("saved", os.path.join(HERE, "results.json"))


if __name__ == "__main__":
    main()
