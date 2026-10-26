# -*- coding: utf-8 -*-
"""부록 12 판정 4칸(H03·H08·H70·H66) 독립 재계산.

judge.py·judge_run.py를 읽지 않고 동결 문서(HYPOTHESES.md §2.0~2.7, §3 F1·F2, 부록 2·4·11·12·15)의
정의만으로 다시 계산한다. 입력은 final/run2026_10_26/results 의 특징·가격 파일뿐이다.
특징값(A_rz8, fight, P_share_rz8, P_bar 등)은 results 파일 값을 그대로 쓰고,
가격 변수(M0·HAR-K 통제·목표), 일정 더미, 회귀·HAC·Hodrick·원형 이동·강건·위약은 여기서 새로 만든다.

사용: py recheck.py [--kospi0629 close_ok|harness_like|naver_fill] [--tag 이름] [옵션]
"""
import argparse
import hashlib
import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/ant_index/final"
RES = ROOT + "/run2026_10_26/results"
OUT = ROOT + "/run2026_10_26/recheck_judge_indep"
JUDGE_SHA = "ddb1dec40a91364e148c4eb411bd184fd6b7da7a67efcb3affa8519a99575c8c"

ap = argparse.ArgumentParser()
ap.add_argument("--kospi0629", default="close_ok", choices=["close_ok", "harness_like", "naver_fill"])
ap.add_argument("--tag", default="main")
ap.add_argument("--no_cal", action="store_true", help="일정 더미를 모두 뺀다(민감도)")
ap.add_argument("--usopex_same_day", action="store_true", help="미국 월물 만기를 같은 날짜 KRX 세션에 붙인다(민감도)")
ap.add_argument("--cal_map", default="after", choices=["after", "same_day"], help="미국 일정(FOMC·CPI·고용·미국 실적·미국 만기): 발표 뒤 첫 KRX 세션 / 같은 날짜 KRX 세션")
ap.add_argument("--fomc_from_0917", action="store_true", help="FOMC를 2025-09-17부터만 넣는다(민감도)")
ap.add_argument("--drop_rule", default="D", choices=["D", "touch"], help="급락 4일 제외: D만 / D 또는 P")
ap.add_argument("--opened_rule", default="touch", choices=["D", "touch"], help="열어 본 8일 제외: D만 / 창이 그 날짜에 걸치면")
ap.add_argument("--hac_ss", action="store_true", help="HAC에 n/(n-k) 소표본 보정")
ap.add_argument("--ic_mode", default="partial", choices=["partial", "plain"], help="H66 IC: 통제 부분 스피어만 / 단순 스피어만")
ap.add_argument("--placebo_mode", default="same_test", choices=["same_test", "plain"], help="H66 위약 IC 방식")
ap.add_argument("--rv", default="rms", choices=["rms", "std"])
ap.add_argument("--harness_sample", action="store_true", help="뒤 구간에서 judgment.json price_gap_windows의 창을 뺀다(같은 표본 대조용)")
ap.add_argument("--m0_extra_cal", action="store_true", help="M0에 창 종류(주말·연휴)·휴장 엇갈림·방학 달 표시를 더한다(민감도)")
ap.add_argument("--m0_all24", action="store_true", help="M0에 §2.4 표시 전부(창 종류·휴장 엇갈림·서머타임 주·방학·감사 시즌·연말 양도세)를 더한다(민감도)")
args = ap.parse_args()

# ---------------------------------------------------------------- 0. 지문
sha = hashlib.sha256(open(ROOT + "/code/judge.py", "rb").read()).hexdigest()
if sha != JUDGE_SHA:
    raise SystemExit(f"judge.py SHA-256 불일치: {sha} — 판정하지 않는다")

# ---------------------------------------------------------------- 1. 가격
px = pd.read_parquet(RES + "/prices.parquet")
pmeta = json.load(open(RES + "/prices_meta.json", encoding="utf-8"))


def bars(tk, adj=False):
    d = px[px.ticker == tk].sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.set_index("date")
    c = d["adj_close"] if adj else d["close"]
    return pd.DataFrame({"O": d["open"], "H": d["high"], "L": d["low"], "C": c, "V": d["volume"]})


ks = bars("^KS11")
d0629 = pd.Timestamp("2026-06-29")
if args.kospi0629 == "harness_like":
    ks.loc[d0629, ["O", "H", "L", "C", "V"]] = np.nan
elif args.kospi0629 == "close_ok":
    # 종가는 네이버와 일치(prices_meta 검증, 불일치 0). 시·고·저가와 거래량은 평평한 막대(0 거래량)라 결측
    ks.loc[d0629, ["O", "H", "L", "V"]] = np.nan
else:  # naver_fill
    diag = pmeta["naver"]["^KS11"]["kospi_ohl_vs_naver_diagnostic"]
    for col, key in (("O", "open"), ("H", "high"), ("L", "low")):
        row = [r for r in diag[key]["rows"] if r["date"] == "2026-06-29"][0]
        ks.loc[d0629, col] = row["naver"]
    ks.loc[d0629, "V"] = np.nan

sox = bars("^SOX")
ewy = bars("EWY", adj=True)
krx_days = list(ks.index)
kpos = {d: i for i, d in enumerate(krx_days)}
C = ks["C"].values.astype(float)
O = ks["O"].values.astype(float)
H = ks["H"].values.astype(float)
L = ks["L"].values.astype(float)
V = ks["V"].values.astype(float)
with np.errstate(divide="ignore", invalid="ignore"):
    r_s = np.r_[np.nan, C[1:] / C[:-1] - 1]
    lr_s = np.r_[np.nan, np.log(C[1:] / C[:-1])]
    oc_s = np.log(C / O)  # 그날 시가→종가
    gap_s = np.r_[np.nan, np.log(O[1:] / C[:-1])]  # 직전 종가→그날 시가

us_days = list(sox.index)
sox_r = (sox["C"] / sox["C"].shift(1) - 1)
ewy_r = (ewy["C"] / ewy["C"].shift(1) - 1)


def last_us_before(D):
    i = np.searchsorted(np.array(us_days, dtype="datetime64[ns]"), np.datetime64(D), side="left") - 1
    return us_days[i] if i >= 0 else None


def rv(lrs):
    if np.any(~np.isfinite(lrs)):
        return np.nan
    return float(np.sqrt(np.mean(lrs ** 2))) if args.rv == "rms" else float(np.std(lrs, ddof=1))


# ---------------------------------------------------------------- 2. 일정 더미
FOMC = ["2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29",
        "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16"]
sched = json.load(open(RES + "/schedule_dates.json", encoding="utf-8"))
earn = json.load(open(RES + "/earnings_dates.json", encoding="utf-8"))
krx_set = set(krx_days)
nyse_cal = pd.read_parquet(RES + "/loader_calendar.parquet")
nyse_set = set(pd.to_datetime(nyse_cal.loc[nyse_cal.nyse, "date"]).dt.normalize())


def first_krx_after(d, inclusive):
    d = pd.Timestamp(d)
    arr = np.array(krx_days, dtype="datetime64[ns]")
    i = np.searchsorted(arr, np.datetime64(d), side="left" if inclusive else "right")
    return krx_days[i] if i < len(krx_days) else None


def nth_weekday(y, m, wd, n):
    d = date(y, m, 1)
    while d.weekday() != wd:
        d += timedelta(days=1)
    return d + timedelta(days=7 * (n - 1))


cal = {k: set() for k in ["fomc", "cpi", "jobs", "earn", "kr_opex", "us_opex", "msci"]}
US_INCL = args.cal_map == "same_day"
for f in FOMC:  # 14:00 ET → 다음 KST 새벽, 첫 KRX 세션은 F 다음 거래일
    if args.fomc_from_0917 and f < "2025-09-17":
        continue
    cal["fomc"].add(first_krx_after(f, inclusive=US_INCL))
for f in sched["cpi"]:  # 08:30 ET = 21:30/22:30 KST → 다음 KRX 세션
    cal["cpi"].add(first_krx_after(f, inclusive=US_INCL))
for f in sched["jobs"]:
    cal["jobs"].add(first_krx_after(f, inclusive=US_INCL))
for f in sched["msci"]:  # 그날 종가에 리밸런싱 → 그날 KRX 세션
    cal["msci"].add(first_krx_after(f, inclusive=True))
for tk, ds in earn.items():
    for e in ds:
        if not ("2024-10-01" <= e <= "2026-10-23"):
            continue
        if tk.endswith(".KS"):  # 국내 실적: 장 전 공시 → 그날 세션
            cal["earn"].add(first_krx_after(e, inclusive=True))
        else:  # 미국 장 마감 뒤 → 다음 KRX 세션
            cal["earn"].add(first_krx_after(e, inclusive=US_INCL))
for y in (2024, 2025, 2026):
    for m in range(1, 13):
        k2 = pd.Timestamp(nth_weekday(y, m, 3, 2))  # 둘째 목요일
        while k2 not in krx_set and k2 > pd.Timestamp("2024-10-01"):
            k2 -= pd.Timedelta(days=1)
            if k2 < pd.Timestamp("2024-10-01"):
                break
        if k2 in krx_set:
            cal["kr_opex"].add(k2)
        u3 = pd.Timestamp(nth_weekday(y, m, 4, 3))  # 셋째 금요일
        while u3 not in nyse_set and u3 >= pd.Timestamp("2024-11-01"):
            u3 -= pd.Timedelta(days=1)
        if args.usopex_same_day or US_INCL:
            cal["us_opex"].add(first_krx_after(u3, inclusive=True))
        else:
            cal["us_opex"].add(first_krx_after(u3, inclusive=False))
for k in cal:
    cal[k].discard(None)

# ---------------------------------------------------------------- 3. 창 표
act = pd.read_parquet(RES + "/feat_activity.parquet")
kr = act[act.market == "kr"].copy()
kr["D"] = pd.to_datetime(kr["D"]).dt.normalize()
kr["P"] = pd.to_datetime(kr["P"]).dt.normalize()
pw = pd.read_parquet(RES + "/pred_windows.parquet")[["win_id", "P_bar", "P_share_rz8"]]
kr = kr.merge(pw, on="win_id", how="left")
kr = kr.sort_values("D").reset_index(drop=True)

rows = []
for _, w in kr.iterrows():
    D, P = w.D, w.P
    iD = kpos.get(D)
    iP = kpos.get(P)
    rec = {"win_id": w.win_id}
    if iD is None or iP is None or iP != iD - 1:
        rec["bad_map"] = True
        rows.append(rec)
        continue
    r1 = r_s[iP]
    rec["r"] = r1
    rec["rpos"] = max(r1, 0) if np.isfinite(r1) else np.nan
    rec["rneg"] = min(r1, 0) if np.isfinite(r1) else np.nan
    rec["absr"] = abs(r1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec["ln_absr"] = np.log(abs(r1)) if np.isfinite(r1) and r1 != 0 else np.nan
    for h in (5, 20, 60):
        rec[f"r{h}"] = C[iP] / C[iP - h] - 1 if iP - h >= 0 else np.nan
    rec["lnRV5"] = np.log(rv(lr_s[iP - 4:iP + 1])) if iP - 4 >= 1 else np.nan
    rec["lnRV20"] = np.log(rv(lr_s[iP - 19:iP + 1])) if iP - 19 >= 1 else np.nan
    rec["lnRV22"] = np.log(rv(lr_s[iP - 21:iP + 1])) if iP - 21 >= 1 else np.nan
    win = C[iP - 19:iP + 1]
    rec["DD20"] = C[iP] / np.max(win) - 1 if iP - 19 >= 0 and np.all(np.isfinite(win)) else np.nan
    vb = V[iP - 20:iP]
    vb = vb[np.isfinite(vb) & (vb > 0)]
    rec["vol_anom"] = np.log(V[iP] / vb.mean()) if (len(vb) >= 15 and np.isfinite(V[iP]) and V[iP] > 0) else np.nan
    rec["g"] = O[iD] / C[iP] - 1
    rec["absg"] = abs(rec["g"])
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(H[iD] / L[iD])
        rec["lnHL"] = np.log(hl) if np.isfinite(hl) and hl > 0 else np.nan
    rec["R5"] = C[iD + 4] / O[iD] - 1 if iD + 4 < len(C) else np.nan
    rec["lnR5"] = np.log(C[iD + 4] / O[iD]) if iD + 4 < len(C) else np.nan
    u = last_us_before(D)
    rec["sox_r"] = sox_r.get(u, np.nan)
    rec["ewy_r"] = ewy_r.get(u, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec["ln_abs_sox"] = np.log(abs(rec["sox_r"]))
        rec["ln_abs_ewy"] = np.log(abs(rec["ewy_r"]))
    for k in cal:
        rec[f"cal_{k}"] = 1.0 if D in cal[k] else 0.0
    wd = D.weekday()
    for j in range(1, 5):
        rec[f"wd_{j}"] = 1.0 if wd == j else 0.0
    rec["iD"] = iD
    rows.append(rec)

pxw = pd.DataFrame(rows)
df = kr.merge(pxw, on="win_id", how="left")
assert not df.get("bad_map", pd.Series(dtype=bool)).fillna(False).any(), "창-세션 대응 실패"
df["judge_ok"] = df["judge_ok"].astype(bool)
df["D_str"] = df["D"].dt.strftime("%Y-%m-%d")
for c in list(df.columns):
    try:
        if isinstance(df[c].dtype, np.dtype) and np.issubdtype(df[c].dtype, np.floating):
            df.loc[~np.isfinite(df[c].astype(float)), c] = np.nan
    except TypeError:
        pass

CAL_COLS = [] if args.no_cal else [f"cal_{k}" for k in ["fomc", "earn", "kr_opex", "us_opex", "cpi", "jobs", "msci"]]
WD_COLS = ["wd_1", "wd_2", "wd_3", "wd_4"]
# HAR-K(§3 F1): ln|r(D−1)| ln RV5 ln RV22 [전일 VKOSPI: 자료 없음] min(r,0) ln|SOX| ln|EWY| ln 거래대금 이상치(거래량으로 대신)
#               |g(D)| 요일 일정 더미 결번율 [결번: 결번율로 대신]
HARK = ["ln_absr", "lnRV5", "lnRV22", "rneg", "ln_abs_sox", "ln_abs_ewy", "vol_anom", "absg", "missing_rate"] + WD_COLS + CAL_COLS
# 한국 M0(§2.3): max(r,0) min(r,0) |r| r5 r20 r60 DD20 lnRV20 거래량 이상치 SOX·EWY 직전 [VKOSPI 1일 변화: 없음] 일정 더미 결번율
M0 = ["rpos", "rneg", "absr", "r5", "r20", "r60", "DD20", "lnRV20", "vol_anom", "sox_r", "ewy_r", "missing_rate"] + CAL_COLS
if args.m0_extra_cal:
    df["cal_weekend"] = (df["kind"] == "weekend").astype(float)
    df["cal_holiday"] = (df["kind"] == "holiday").astype(float)
    df["cal_xholiday"] = df["xholiday_mismatch"].astype(float)
    df["cal_vacation"] = df["D"].dt.month.isin([1, 2, 7, 8]).astype(float)
    M0 = M0 + ["cal_weekend", "cal_holiday", "cal_xholiday", "cal_vacation"]
if args.m0_all24:
    df["cal_weekend"] = (df["kind"] == "weekend").astype(float)
    df["cal_holiday"] = (df["kind"] == "holiday").astype(float)
    df["cal_xholiday"] = df["xholiday_mismatch"].astype(float)
    df["cal_dst"] = df["dst_week"].astype(float)
    df["cal_vacation"] = df["D"].dt.month.isin([1, 2, 7, 8]).astype(float)
    md = df["D"].dt.month * 100 + df["D"].dt.day
    df["cal_audit"] = ((md >= 310) & (md <= 430)).astype(float)
    df["cal_tax"] = ((md >= 1201) | (md <= 131)).astype(float)
    M0 = M0 + ["cal_weekend", "cal_holiday", "cal_xholiday", "cal_dst", "cal_vacation", "cal_audit", "cal_tax"]
# H66 판정 회귀 통제(§2.3): 대상 r(D−1) r5 r20 RV20 g(D) + 전야 SOX
H66_CTRL = ["r", "r5", "r20", "lnRV20", "g", "sox_r"]

BACK = ("2026-06-01", "2026-10-23")
BACK_LABEL = ("2026-07-01", "2026-10-23")
FRONT = ("2026-01-02", "2026-05-29")
CRASH4 = ["2026-07-29", "2026-07-30", "2026-07-31", "2026-10-13"]
JUL3 = ["2026-07-29", "2026-07-30", "2026-07-31"]
OPENED8 = ["2026-08-20", "2026-09-02", "2026-09-03", "2026-09-08", "2026-09-16", "2026-10-12", "2026-10-14", "2026-10-15"]


def in_rng(s, rng):
    return (df["D_str"] >= rng[0]) & (df["D_str"] <= rng[1])


def drop_mask(days, rule):
    days = pd.to_datetime(days)
    if rule == "D":
        return df["D"].isin(days)
    if rule == "touch_DP":
        return df["D"].isin(days) | df["P"].isin(days)
    # touch: 창 구간 (open_kst, close_kst]가 그 날짜(0~24시)와 겹치면
    o = pd.to_datetime(df["open_kst"])
    c = pd.to_datetime(df["close_kst"])
    m = pd.Series(False, index=df.index)
    for d in days:
        m |= (o < d + pd.Timedelta(days=1)) & (c > d)
    return m


CRASH_RULE = "D" if args.drop_rule == "D" else "touch_DP"
crash_m = drop_mask(CRASH4, CRASH_RULE)
jul3_m = drop_mask(JUL3, CRASH_RULE)
opened_m = drop_mask(OPENED8, "D" if args.opened_rule == "D" else "touch")


# ---------------------------------------------------------------- 4. 확장 OLS M0 (월 1회 재적합, 최소 120창)
def m0_expanding(xcol, regs=M0, min_n=120):
    xhat = pd.Series(np.nan, index=df.index)
    xperp = pd.Series(np.nan, index=df.index)
    r2s = []
    first_month = None
    months = sorted(df["D"].dt.to_period("M").unique())
    for mth in months:
        mstart = mth.to_timestamp()
        fit = df[(df["D"] < mstart) & df["judge_ok"]].dropna(subset=[xcol] + regs)
        if len(fit) < min_n:
            continue
        Xf = np.column_stack([np.ones(len(fit))] + [fit[c].values for c in regs])
        yf = fit[xcol].values
        b, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
        res = yf - Xf @ b
        rank = np.linalg.matrix_rank(Xf)
        sd = np.sqrt(res @ res / (len(fit) - rank))
        r2s.append(1 - res.var() / yf.var())
        if first_month is None:
            first_month = str(mth)
        tgt = df[(df["D"].dt.to_period("M") == mth)].dropna(subset=regs)
        Xt = np.column_stack([np.ones(len(tgt))] + [tgt[c].values for c in regs])
        pred = Xt @ b
        xhat.loc[tgt.index] = pred
        xperp.loc[tgt.index] = (tgt[xcol].values - pred) / sd
    return xhat, xperp, {"first_month": first_month, "R2_mean": float(np.mean(r2s)) if r2s else None,
                          "R2_last": float(r2s[-1]) if r2s else None}


# ---------------------------------------------------------------- 5. 회귀 도구
def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ b
    return b, u


def nw_cov(X, u, lags=5, ss=False):
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    if ss:
        V *= n / (n - k)
    return V


def prep(sub_cols, mask, x_override=None):
    d = df[mask].copy()
    if x_override is not None:
        d["__x"] = x_override[mask]
    d = d.dropna(subset=sub_cols)
    return d.sort_values("D")


def hark(xs, ycol, ctrls, mask, sign=+1, return_contrib=False):
    """xs: Series(전체 df 색인)의 특징. 반환: 부분 상관·HAC(5) t·단측 p 등."""
    d = df[mask].copy()
    d["__x"] = xs[mask]
    d = d.dropna(subset=["__x", ycol] + ctrls).sort_values("D")
    n = len(d)
    if n < 20:
        return {"n": n}
    Z = np.column_stack([np.ones(n)] + [d[c].values.astype(float) for c in ctrls])
    keep = [0] + [j for j in range(1, Z.shape[1]) if np.std(Z[:, j]) > 0]
    dropped = [ctrls[j - 1] for j in range(1, Z.shape[1]) if j not in keep]
    Z = Z[:, keep]
    x = d["__x"].values.astype(float)
    y = d[ycol].values.astype(float)
    X = np.column_stack([Z, x])
    b, u = ols(y, X)
    k = np.linalg.matrix_rank(X)
    V = nw_cov(X, u, 5, args.hac_ss)
    se = np.sqrt(V[-1, -1])
    t = b[-1] / se
    dfree = n - k
    p = stats.t.sf(sign * t, dfree)
    _, ux = ols(x, Z)
    _, uy = ols(y, Z)
    pc = float(np.corrcoef(ux, uy)[0, 1])
    r2_full = 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))
    r2_res = 1 - (uy @ uy) / ((y - y.mean()) @ (y - y.mean()))
    out = {"n": n, "k": int(k), "est": pc, "coef": float(b[-1]), "t": float(t), "p_an": float(p),
           "dR2": float(r2_full - r2_res), "dropped": dropped, "first": d["D_str"].iloc[0], "last": d["D_str"].iloc[-1],
           "se_hac_pc": float(abs(pc / t)) if t != 0 else None}
    if return_contrib:
        contrib = ux * uy * sign
        out["_ids"] = d["win_id"].values
        out["_contrib"] = contrib
    return out


def hodrick_t(d, xr, h=5):
    """Hodrick 1B: 목표 R_h = 시가→종가(첫날) + 종가→종가(뒤 4일). 귀무에서 시가→종가와 갭 조각을 따로 둔다."""
    n = len(d)
    ctrls = [c for c in H66_CTRL]
    W = np.column_stack([np.ones(n)] + [d[c].values.astype(float) for c in ctrls] + [xr])
    y = d["lnR5"].values.astype(float)
    b, _ = ols(y, W)
    iDs = d["iD"].values.astype(int)
    lo, hi = iDs.min(), iDs.max() + h - 1
    k = W.shape[1]
    S = np.zeros((k, k))
    oc_all = oc_s.copy()
    gap_all = gap_s.copy()
    # 표본 평균을 뺀 조각(귀무: 예측 불가)
    span = np.arange(lo, hi + 1)
    oc_m = np.nanmean(oc_all[span])
    gp_m = np.nanmean(gap_all[span])
    for s in span:
        w0 = np.zeros(k)
        w1 = np.zeros(k)
        for j, t0 in enumerate(iDs):
            if t0 <= s <= t0 + h - 1:
                w0 += W[j]
                if s > t0:
                    w1 += W[j]
        e0 = oc_all[s] - oc_m
        e1 = gap_all[s] - gp_m
        if np.isfinite(e0):
            S += e0 ** 2 * np.outer(w0, w0)
        if np.isfinite(e1):
            S += e1 ** 2 * np.outer(w1, w1)
    WtWi = np.linalg.pinv(W.T @ W)
    Vb = WtWi @ S @ WtWi
    return float(b[-1] / np.sqrt(Vb[-1, -1])), int(np.linalg.matrix_rank(W))


def rankz(v):
    r = stats.rankdata(v)
    return (r - r.mean()) / r.std()


def ic_stat(d, xcol, mode=None):
    mode = mode or args.ic_mode
    x = d[xcol].values.astype(float)
    y = d["R5"].values.astype(float)
    if mode == "plain":
        return float(stats.spearmanr(x, y)[0]), rankz(x), rankz(y)
    Z = np.column_stack([np.ones(len(d))] + [d[c].values.astype(float) for c in H66_CTRL])
    _, ux = ols(rankz(x), Z)
    _, uy = ols(rankz(y), Z)
    return float(np.corrcoef(ux, uy)[0, 1]), ux, uy


def ic_test(xs, mask, sign=-1, with_t=True, mode=None):
    d = df[mask].copy()
    d["__x"] = xs[mask]
    d = d.dropna(subset=["__x", "R5", "lnR5"] + H66_CTRL).sort_values("D")
    n = len(d)
    if n < 15:
        return {"n": n}
    est, ux, uy = ic_stat(d, "__x", mode)
    out = {"n": n, "est": est, "first": d["D_str"].iloc[0], "last": d["D_str"].iloc[-1]}
    if with_t:
        t, k = hodrick_t(d, rankz(d["__x"].values.astype(float)))
        out["t"] = t
        out["p_an"] = float(stats.t.sf(sign * t, n - k))
        out["se_hod"] = float(abs(est / t)) if t != 0 else None
    out["_ids"] = d["win_id"].values
    out["_contrib"] = ux * uy * sign
    return out


# ---------------------------------------------------------------- 6. 원형 이동 p
rot = df[(df["D"] >= "2026-01-01") & df["judge_ok"]].sort_values("D").index.values
N_ROT = len(rot)
SHIFTS = list(range(21, N_ROT - 21 + 1))


def shift_series(xs, k):
    v = xs.loc[rot].values
    s = xs.copy()
    s.loc[rot] = np.roll(v, k)
    return s


def shift_p_hark(xs, ycol, ctrls, mask, sign, obs):
    cnt_t = cnt_e = 0
    for k in SHIFTS:
        r = hark(shift_series(xs, k), ycol, ctrls, mask, sign)
        if sign * r["t"] >= sign * obs["t"]:
            cnt_t += 1
        if sign * r["est"] >= sign * obs["est"]:
            cnt_e += 1
    K = len(SHIFTS)
    return (1 + cnt_t) / (1 + K), (1 + cnt_e) / (1 + K), cnt_t, cnt_e


def shift_p_ic(xs, mask, sign, obs):
    cnt = 0
    for k in SHIFTS:
        r = ic_test(shift_series(xs, k), mask, sign, with_t=False)
        if sign * r["est"] >= sign * obs["est"]:
            cnt += 1
    K = len(SHIFTS)
    return (1 + cnt) / (1 + K), cnt


# ---------------------------------------------------------------- 7. 칸 계산
base_ok = df["judge_ok"]


HGAP = json.load(open(RES + "/judgment.json", encoding="utf-8"))["price_gap_windows"]


def gap_excl(cid):
    if not args.harness_sample:
        return pd.Series(False, index=df.index)
    ids = [w["win_id"] for w in HGAP[cid]["windows"]]
    return df["win_id"].isin(ids)


def run_t1(cid, xcol, extra_ctrls, label_layer, sign=+1):
    ctrls = HARK[:9] + extra_ctrls + HARK[9:]
    back_m = base_ok & in_rng(df, BACK_LABEL if label_layer else BACK) & ~gap_excl(cid)
    front_m = base_ok & in_rng(df, FRONT)
    y25_m = base_ok & (df["D"].dt.year == 2025)
    xs = df[xcol].astype(float)
    obs = hark(xs, "lnHL", ctrls, back_m, sign, return_contrib=True)
    fr = hark(xs, "lnHL", ctrls, front_m, sign)
    y25 = hark(xs, "lnHL", ctrls, y25_m, sign)
    p_t, p_e, ct, ce = shift_p_hark(xs, "lnHL", ctrls, back_m, sign, obs)
    rob = {}
    rob["crash4"] = hark(xs, "lnHL", ctrls, back_m & ~crash_m, sign)
    order = np.argsort(-obs["_contrib"])
    top2 = list(obs["_ids"][order[:2]])
    rob["top2"] = hark(xs, "lnHL", ctrls, back_m & ~df["win_id"].isin(top2), sign)
    rob["opened8"] = hark(xs, "lnHL", ctrls, back_m & ~opened_m, sign)
    if cid == "H03":
        rob["jul3"] = hark(xs, "lnHL", ctrls, back_m & ~jul3_m, sign)
    ratios = {k: v["est"] / obs["est"] for k, v in rob.items()}
    # 위약: X̂ (가격만으로 만든 가짜 특징)과 X⊥
    xhat, xperp, m0i = m0_expanding(xcol)
    pl_m = back_m & xhat.notna()
    pl = hark(xhat, "lnHL", ctrls, pl_m, sign)
    plx = hark(xs, "lnHL", ctrls, pl_m & xs.notna(), sign)
    xp = hark(xperp, "lnHL", ctrls, back_m, sign, return_contrib=True)
    xp_fr = hark(xperp, "lnHL", ctrls, front_m, sign)
    xp_pt, xp_pe, _, _ = shift_p_hark(xperp, "lnHL", ctrls, back_m, sign, xp)
    xp_rob = {}
    xp_rob["crash4"] = hark(xperp, "lnHL", ctrls, back_m & ~crash_m, sign)
    o2 = np.argsort(-xp["_contrib"])
    xp_rob["top2"] = hark(xperp, "lnHL", ctrls, back_m & ~df["win_id"].isin(list(xp["_ids"][o2[:2]])), sign)
    xp_rob["opened8"] = hark(xperp, "lnHL", ctrls, back_m & ~opened_m, sign)
    comp = None
    if cid == "H03":
        comp = hark(df["A_saved_rz8"].astype(float), "lnHL", ctrls, back_m, sign)
    res = {
        "id": cid, "x": xcol, "layer": "label" if label_layer else "forum", "sign": sign,
        "n": obs["n"], "first": obs["first"], "last": obs["last"], "est": obs["est"], "coef": obs["coef"],
        "t": obs["t"], "p_an": obs["p_an"], "dR2": obs["dR2"], "dropped": obs["dropped"],
        "p_shift_t": p_t, "p_shift_est": p_e, "shift_exceed_t": ct, "shift_exceed_est": ce,
        "p_final": max(obs["p_an"], p_t), "se_hac_pc": obs["se_hac_pc"],
        "est_front": fr.get("est"), "n_front": fr.get("n"), "est_2025": y25.get("est"), "n_2025": y25.get("n"),
        "robust": {k: {"n": v["n"], "est": v["est"], "ratio": ratios[k]} for k, v in rob.items()},
        "top2_dropped": top2,
        "placebo": {"n": pl.get("n"), "est_xhat": pl.get("est"), "est_x_same_sample": plx.get("est"),
                    "ratio": (pl["est"] / plx["est"]) if pl.get("est") is not None and plx.get("est") else None},
        "m0": m0i,
        "xperp": {"n": xp.get("n"), "est": xp.get("est"), "t": xp.get("t"), "p_an": xp.get("p_an"),
                  "p_shift": xp_pt, "p_final": max(xp["p_an"], xp_pt), "est_front": xp_fr.get("est"),
                  "robust_ratio": {k: v["est"] / xp["est"] for k, v in xp_rob.items()}},
        "comp_saved": (None if comp is None else {"n": comp["n"], "est": comp["est"], "t": comp["t"]}),
    }
    return res


def run_h66():
    sign = -1
    back_m = base_ok & in_rng(df, BACK_LABEL) & ~gap_excl("H66")
    front_m = base_ok & in_rng(df, FRONT)
    y25_m = base_ok & (df["D"].dt.year == 2025)
    xraw = df["P_bar"].astype(float)
    xhat, xperp, m0i = m0_expanding("P_bar")
    obs = ic_test(xperp, back_m, sign)
    fr = ic_test(xperp, front_m, sign)
    y25 = ic_test(xperp, y25_m, sign)
    p_sh, cnt = shift_p_ic(xperp, back_m, sign, obs)
    rob = {}
    rob["crash4"] = ic_test(xperp, back_m & ~crash_m, sign)
    order = np.argsort(-obs["_contrib"])
    top2 = list(obs["_ids"][order[:2]])
    rob["top2"] = ic_test(xperp, back_m & ~df["win_id"].isin(top2), sign)
    rob["opened8"] = ic_test(xperp, back_m & ~opened_m, sign)
    ratios = {k: v["est"] / obs["est"] for k, v in rob.items()}
    pl_m = back_m & xhat.notna() & xperp.notna()
    pmode = None if args.placebo_mode == "same_test" else "plain"
    pl = ic_test(xhat, pl_m, sign, with_t=False, mode=pmode)
    plx = ic_test(xraw, pl_m, sign, with_t=False, mode=pmode)
    raw = ic_test(xraw, back_m, sign)
    return {
        "id": "H66", "x": "P_bar⊥(M0 확장 OLS 잔차 ÷ 잔차 SD)", "layer": "label", "sign": sign,
        "n": obs["n"], "first": obs["first"], "last": obs["last"], "est": obs["est"], "t": obs["t"], "p_an": obs["p_an"],
        "p_shift": p_sh, "shift_exceed": cnt, "p_final": max(obs["p_an"], p_sh), "se_hod": obs["se_hod"],
        "est_front": fr.get("est"), "n_front": fr.get("n"), "est_2025": y25.get("est"), "n_2025": y25.get("n"),
        "robust": {k: {"n": v["n"], "est": v["est"], "ratio": ratios[k]} for k, v in rob.items()},
        "top2_dropped": top2,
        "placebo": {"n": pl.get("n"), "ic_xhat": pl.get("est"), "ic_x_same_sample": plx.get("est"),
                    "ratio": pl["est"] / plx["est"] if plx.get("est") else None},
        "raw_P_bar": {"n": raw.get("n"), "est": raw.get("est"), "t": raw.get("t")},
        "m0": m0i,
    }


def verdict(c, thr_ok, expected, front_required):
    s = c["sign"]
    est_s = s * c["est"]
    se = c.get("se_hac_pc") or c.get("se_hod")
    ci_up = est_s + 1.645 * se
    c["ci90_upper_signed"] = ci_up
    rob_ok = all((v["ratio"] is not None and v["ratio"] >= 0.5) for v in c["robust"].values() if True)
    c["robust_ok"] = rob_ok
    front_ok = (c["est_front"] is not None) and (s * c["est_front"] > 0)
    c["front_ok"] = front_ok
    pl = c["placebo"]
    pl_ratio = pl.get("ratio")
    xperp_pass = None
    if c["id"] == "H66":
        main_pass = thr_ok and (front_ok or not front_required) and rob_ok
        xperp_pass = main_pass
    else:
        xp = c["xperp"]
        xperp_pass = (xp["p_final"] <= c["thr_local"]) and ((s * (xp["est_front"] or 0) > 0) or not front_required) \
            and all(r >= 0.5 for r in xp["robust_ratio"].values())
    pl_flag = (pl_ratio is not None) and (pl_ratio >= 0.7) and (not xperp_pass)
    c["placebo_flag"] = bool(pl_flag)
    c["xperp_pass"] = bool(xperp_pass)
    if est_s < 0:
        v, why = "기각", "뒤 구간 점추정 부호 반대"
    elif ci_up < expected / 2:
        v, why = "기각", "90% 상한 < 기대 효과 절반"
    elif pl_flag:
        v, why = "기각", "메아리 위약(X̂가 같은 부호로 X 효과의 70% 이상, X⊥ 미통과)"
    elif thr_ok and (front_ok or not front_required) and rob_ok:
        v, why = "통과", ""
    elif thr_ok and (front_ok or not front_required):
        v, why = "취약 후보", "강건 3개 미달"
    else:
        v, why = "판정 불가", "보정 p 미달 또는 앞 구간 부호 다름"
    c["verdict"], c["why"] = v, why
    return c


t1 = [run_t1("H03", "A_rz8", [], False), run_t1("H08", "fight", ["A_rz8", "ln_n_comments_onposts", "trunc_share"], False),
      run_t1("H70", "P_share_rz8", [], True)]
# Holm (단측 족오류 0.05, 3칸)
order = sorted(range(3), key=lambda i: t1[i]["p_final"])
stop = False
for rank, i in enumerate(order):
    thr = 0.05 / (3 - rank)
    t1[i]["thr_local"] = thr
    ok = (not stop) and t1[i]["p_final"] <= thr
    if not ok:
        stop = True
    t1[i]["holm_ok"] = ok
for c in t1:
    verdict(c, c["holm_ok"], 0.15, front_required=(c["layer"] == "forum"))
h66 = run_h66()
h66["thr_local"] = 0.10
verdict(h66, h66["p_final"] <= 0.10, 0.10, front_required=False)

for c in t1 + [h66]:
    for k in ("_ids", "_contrib"):
        c.pop(k, None)

harn = json.load(open(RES + "/judgment.json", encoding="utf-8"))["judgment_cells"]
cmp = {}
for c in t1 + [h66]:
    hj = harn[c["id"]]
    cmp[c["id"]] = {
        "verdict_recheck": c["verdict"], "verdict_harness": hj["verdict"], "same_category": c["verdict"] == hj["verdict"],
        "n": [c["n"], hj["n_back"]], "est": [c["est"], hj["est"]], "t": [c["t"], hj["t"]],
        "p_final": [c["p_final"], hj["p_final"]], "est_front": [c["est_front"], hj["est_front"]],
        "placebo_ratio": [c["placebo"]["ratio"], (hj["placebo_est"] / hj["placebo_x_est"]) if hj.get("placebo_x_est") else None],
        "robust_ratio": [{k: v["ratio"] for k, v in c["robust"].items()}, hj["robust_ratio"]],
    }

out = {"tag": args.tag, "args": vars(args), "judge_sha256": sha, "n_rot": N_ROT, "n_shifts": len(SHIFTS),
       "cal_counts": {k: len(v) for k, v in cal.items()}, "HARK": HARK, "M0": M0, "H66_CTRL": H66_CTRL,
       "cells": {c["id"]: c for c in t1 + [h66]}, "compare": cmp}
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/recheck_{args.tag}.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
for cid, v in cmp.items():
    print(cid, json.dumps(v, ensure_ascii=False, default=lambda o: round(float(o), 4)))
