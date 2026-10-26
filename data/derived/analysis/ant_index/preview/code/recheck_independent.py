"""독립 재계산: H03, H46, H66 (부록 9 미리보기 검산).

시험 코드(preview/code/의 다른 파일)를 읽지 않고, 동결 정의와
preview/results/의 패널·특징 파일, prices.parquet, preview/labels/만으로 다시 계산한다.
결과는 preview/results/recheck_independent.json 에 쓴다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "recheck_independent.json"


# ---------------------------------------------------------------- 통계 도구
def ols(y, X):
    """y (n,), X (n,k) 상수 포함. 계수, 잔차, (X'X)^-1."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    return b, e, XtX_inv


def nw_se(X, e, lags, small=True):
    """Newey-West(Bartlett) HAC, 관측 순서 기준."""
    n, k = X.shape
    u = X * e[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    XtX_inv = np.linalg.pinv(X.T @ X)
    V = XtX_inv @ S @ XtX_inv
    if small:
        V *= n / (n - k)
    return np.sqrt(np.diag(V))


def white_se(X, e, small=True):
    return nw_se(X, e, 0, small)


def nw_se_cal(X, e, pos, lags, small=True):
    """Newey-West(Bartlett), 시차를 관측 순서가 아니라 거래일 위치 차이로 잰다(빈 창 반영)."""
    n, k = X.shape
    u = X * e[:, None]
    S = u.T @ u
    for i in range(n):
        for j in range(i):
            L = pos[i] - pos[j]
            if 0 < L <= lags:
                G = np.outer(u[i], u[j])
                S += (1 - L / (lags + 1)) * (G + G.T)
    XtX_inv = np.linalg.pinv(X.T @ X)
    V = XtX_inv @ S @ XtX_inv
    if small:
        V *= n / (n - k)
    return np.sqrt(np.diag(V))


def hodrick_full_t(yy, XX, pos, h, daily_lr, small=True):
    """Hodrick 1B(전체 행렬): S = Σ_s ε_s² z_s z_s', z_s = Σ_{t: s-h+1<=t<=s} X_t,
    ε_s = 하루 로그 종가 대 종가 수익 − 평균. 마지막 열 계수의 t."""
    n, k = XX.shape
    b, _, XtX_inv = ols(yy, XX)
    s_all = sorted({p + j for p in pos for j in range(h)})
    idx = {s: i for i, s in enumerate(s_all)}
    Z = np.zeros((len(s_all), k))
    for p, row in zip(pos, XX):
        for j in range(h):
            Z[idx[p + j]] += row
    rs = np.array([daily_lr.get(s, np.nan) for s in s_all])
    ok = ~np.isnan(rs)
    eps = rs - np.nanmean(rs)
    S = (Z[ok] * eps[ok, None] ** 2).T @ Z[ok]
    V = XtX_inv @ S @ XtX_inv
    if small:
        V *= n / (n - k)
    return float(b[-1] / np.sqrt(V[-1, -1])), float(b[-1])


def classic_se(X, e):
    n, k = X.shape
    s2 = e @ e / (n - k)
    return np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * s2)


def hodrick_1b_t(R, xcol, C, day_pos, h, daily_r):
    """Hodrick(1992) 1B: 겹치는 h일 수익 R(t)를 x에 회귀할 때 귀무(예측력 없음)
    아래 SE. FWL로 통제 C를 뺀 x̃를 쓰고, 분자 Σ x̃_t R_t = Σ_s r_s Σ_{t: s-h+1<=t<=s} x̃_t
    로 바꿔 r_s(하루 수익, 평균 제거)로 분산을 잰다.
    day_pos: 관측 t의 진입일 위치(시장 달력 정수), daily_r: 위치→하루 수익 dict."""
    Cc = np.column_stack([np.ones(len(R)), C]) if C is not None and C.size else np.ones((len(R), 1))
    bx, _, _ = ols(xcol, Cc)
    xt = xcol - Cc @ bx
    beta = (xt @ R) / (xt @ xt)
    s_all = sorted({p + j for p in day_pos for j in range(h)})
    rs = np.array([daily_r.get(s, np.nan) for s in s_all])
    if np.isnan(rs).any():
        return np.nan, np.nan, beta
    rs = rs - rs.mean()
    z = np.zeros(len(s_all))
    idx = {s: i for i, s in enumerate(s_all)}
    for p, v in zip(day_pos, xt):
        for j in range(h):
            z[idx[p + j]] += v
    var_num = np.sum(rs ** 2 * z ** 2)
    se = np.sqrt(var_num) / (xt @ xt)
    return beta / se, se, beta


def rankz(v):
    r = stats.rankdata(v)
    return (r - r.mean()) / r.std(ddof=0)


def partial_spearman(x, y, C):
    """rank(x), rank(y)를 rank(C)에 회귀한 잔차의 상관."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if C is None or C.shape[1] == 0:
        return stats.pearsonr(rx, ry)[0]
    RC = np.column_stack([np.ones(len(x))] + [stats.rankdata(C[:, j]) for j in range(C.shape[1])])
    _, ex, _ = ols(rx, RC)
    _, ey, _ = ols(ry, RC)
    return float(np.corrcoef(ex, ey)[0, 1])


def partial_spearman_raw(x, y, C):
    """x, y를 C(원값)에 회귀한 잔차의 스피어만."""
    CC = np.column_stack([np.ones(len(x)), C])
    _, ex, _ = ols(x, CC)
    _, ey, _ = ols(y, CC)
    return float(stats.spearmanr(ex, ey)[0])


def resid_std(x, M):
    """X⊥ = X를 M0에 OLS(표본 안) 회귀한 잔차 ÷ 잔차 SD. R²도 낸다."""
    MM = np.column_stack([np.ones(len(x)), M])
    b, e, _ = ols(x, MM)
    r2 = 1 - (e @ e) / np.sum((x - x.mean()) ** 2)
    return e / e.std(ddof=1), r2, MM @ b


# ---------------------------------------------------------------- 가격 재구성
prices = pd.read_parquet(RES / "prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])


def px(tk):
    d = prices[prices.ticker == tk].set_index("date").sort_index()
    bad = d["flag_flat_bar"] | d["flag_zero_volume"] if tk in ("^KS11", "^KQ11") else pd.Series(False, d.index)
    d.loc[bad, ["open", "high", "low", "close", "volume"]] = np.nan
    return d


def rebuild_daily(tk):
    d = px(tk)
    c = d["close"]
    out = pd.DataFrame(index=d.index)
    out["open"], out["high"], out["low"], out["close"], out["vol"] = d["open"], d["high"], d["low"], c, d["volume"]
    out["r"] = c / c.shift(1) - 1
    out["r5"] = c / c.shift(5) - 1
    out["r20"] = c / c.shift(20) - 1
    out["r60"] = c / c.shift(60) - 1
    out["lnRV5"] = np.log(out["r"].rolling(5).std())
    out["lnRV20"] = np.log(out["r"].rolling(20).std())
    out["lnRV22"] = np.log(out["r"].rolling(22).std())
    out["DD20"] = c / c.rolling(20).max() - 1
    out["ln_vol_anom"] = np.log(d["volume"] / d["volume"].shift(1).rolling(20).mean())
    out["lnHL"] = np.log(np.log(d["high"] / d["low"]))
    out["g"] = d["open"] / c.shift(1) - 1
    out["pos"] = np.arange(len(out))
    return out


kospi = rebuild_daily("^KS11")
soxx = rebuild_daily("SOXX")
sox = rebuild_daily("^SOX")
ewy = rebuild_daily("EWY")


def fwd(d, D, h):
    """진입일 D 시가 → D+h-1 종가 단순 수익."""
    if D not in d.index:
        return np.nan
    p = d.loc[D, "pos"]
    if p + h - 1 >= len(d):
        return np.nan
    return d["close"].iloc[int(p) + h - 1] / d.loc[D, "open"] - 1


kr = pd.read_parquet(RES / "panel_kr.parquet")
us = pd.read_parquet(RES / "panel_us.parquet")
for df in (kr, us):
    df["D"] = pd.to_datetime(df["D"])
    df["P"] = pd.to_datetime(df["P"])

# 검산 1: 패널의 가격 열을 내 재구성과 비교
chk = {}


def cmp(name, mine, theirs):
    a = pd.Series(mine, dtype=float).values
    b = pd.Series(theirs, dtype=float).values
    m = ~np.isnan(a) & ~np.isnan(b)
    chk[name] = {
        "n_both": int(m.sum()),
        "n_only_panel": int((np.isnan(a) & ~np.isnan(b)).sum()),
        "n_only_mine": int((~np.isnan(a) & np.isnan(b)).sum()),
        "max_abs_diff": float(np.max(np.abs(a[m] - b[m]))) if m.any() else None,
        "corr": float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else None,
    }


# KR: P = 직전 KRX 세션(창 안 마지막 세션). 미국 직전 세션 = m0_us_session.
kr["us_sess"] = pd.to_datetime(kr["m0_us_session"])
my = {}
my["m0_kospi_r"] = kr.P.map(kospi["r"])
my["m0_kospi_r5"] = kr.P.map(kospi["r5"])
my["m0_kospi_r20"] = kr.P.map(kospi["r20"])
my["m0_kospi_r60"] = kr.P.map(kospi["r60"])
my["m0_kospi_lnRV20"] = kr.P.map(kospi["lnRV20"])
my["m0_kospi_lnRV5"] = kr.P.map(kospi["lnRV5"])
my["m0_kospi_lnRV22"] = kr.P.map(kospi["lnRV22"])
my["m0_kospi_DD20"] = kr.P.map(kospi["DD20"])
my["m0_kospi_ln_vol_anom"] = kr.P.map(kospi["ln_vol_anom"])
my["m0_kospi_g"] = kr.D.map(kospi["g"])
my["m0_sox_r_u"] = kr.us_sess.map(sox["r"])
my["m0_ewy_r_u"] = kr.us_sess.map(ewy["r"])
my["fut_kospi_lnHL"] = kr.D.map(kospi["lnHL"])
my["fut_kospi_R5"] = [fwd(kospi, D, 5) for D in kr.D]
for c, v in my.items():
    cmp("kr." + c, v, kr[c])
kr_my = pd.DataFrame(my)
kr_my["m0_kospi_ln_absr"] = np.log(np.abs(kr_my["m0_kospi_r"]))
kr_my["m0_kospi_rneg"] = np.minimum(kr_my["m0_kospi_r"], 0)
kr_my["m0_kospi_absg"] = np.abs(kr_my["m0_kospi_g"])
kr_my["m0_sox_ln_absr_u"] = np.log(np.abs(kr_my["m0_sox_r_u"]))
kr_my["m0_ewy_ln_absr_u"] = np.log(np.abs(kr_my["m0_ewy_r_u"]))
for c in ["m0_kospi_ln_absr", "m0_kospi_rneg", "m0_kospi_absg", "m0_sox_ln_absr_u", "m0_ewy_ln_absr_u"]:
    cmp("kr." + c, kr_my[c], kr[c])

# US: P = 직전 NYSE 세션, D = 진입 세션
myu = {}
myu["m0_soxx_r"] = us.P.map(soxx["r"])
myu["m0_soxx_r5"] = us.P.map(soxx["r5"])
myu["m0_soxx_r20"] = us.P.map(soxx["r20"])
myu["m0_soxx_lnRV20"] = us.P.map(soxx["lnRV20"])
myu["m0_soxx_g"] = us.D.map(soxx["g"])
myu["m0_soxx_minus_ewy_r20"] = us.P.map(soxx["r20"]) - us.P.map(ewy["r20"])
myu["fut_soxx_R10"] = [fwd(soxx, D, 10) for D in us.D]
myu["m0_kospi_r20_k"] = pd.to_datetime(us["m0_kr_session"]).map(kospi["r20"])
for c, v in myu.items():
    cmp("us." + c, v, us[c])

# ---------------------------------------------------------------- rz8 재계산(H03 특징)
act = pd.read_parquet(RES / "feat_activity.parquet")
ak = act[act.market == "kr"].copy()
ak["D"] = pd.to_datetime(ak["D"])
ak = ak.sort_values("D").reset_index(drop=True)


def rz8(df, col):
    out = np.full(len(df), np.nan)
    for i in range(len(df)):
        if pd.isna(df.loc[i, col]) or df.loc[i, "outage"] is True:
            continue
        wd = df.loc[i, "weekday"]
        prev = df.iloc[:i]
        base = prev[(prev.weekday == wd) & (prev.kind != "holiday") & prev[col].notna() & (prev.outage != True)]
        base = base.tail(8)
        if len(base) < 8:
            continue
        med = base[col].median()
        mad = 1.4826 * np.median(np.abs(base[col] - med))
        if mad <= 0:
            continue
        out[i] = np.clip((df.loc[i, col] - med) / mad, -6, 6)
    return out


ak["A_rz8_mine"] = rz8(ak, "d_no_per_h")
cmp("kr.A_rz8", ak["A_rz8_mine"], ak["A_rz8"])

# ---------------------------------------------------------------- H03
res = {"checks_panel_vs_prices": chk}
K = kr.merge(ak[["win_id", "A_rz8", "A_rz8_mine", "A_saved_rz8"]], on="win_id", how="left")
for wd in range(1, 5):
    K[f"wd_{wd}"] = (K["weekday"] == wd).astype(float)
har = ["m0_kospi_ln_absr", "m0_kospi_lnRV5", "m0_kospi_lnRV22", "m0_kospi_rneg", "m0_sox_ln_absr_u",
       "m0_ewy_ln_absr_u", "m0_kospi_ln_vol_anom", "m0_kospi_absg", "m0_missing_rate",
       "wd_1", "wd_2", "wd_3", "wd_4", "m0_cal_earn", "m0_cal_kr_opex", "m0_cal_us_opex", "m0_cal_fomc"]


def h03_fit(df, xcol, ctrls, ycol="fut_kospi_lnHL", drop_const=True, lags=5):
    d = df[df.judge_ok & (df.D >= "2026-07-01")].dropna(subset=[ycol, xcol] + ctrls).sort_values("D")
    use = [c for c in ctrls if not (drop_const and d[c].nunique() <= 1)]
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in use] + [d[xcol].values])
    y = d[ycol].values
    b, e, _ = ols(y, X)
    se_nw = nw_se(X, e, lags, small=True)
    se_nw_ns = nw_se(X, e, lags, small=False)
    se_cl = classic_se(X, e)
    pos = [int(kospi.loc[D, "pos"]) for D in d.D]
    se_cal = nw_se_cal(X, e, pos, lags, small=True)
    # 부분 R² 증가
    Xr = X[:, :-1]
    _, er, _ = ols(y, Xr)
    tss = np.sum((y - y.mean()) ** 2)
    dR2 = (er @ er - e @ e) / tss
    # 부분 상관(FWL)
    _, ex, _ = ols(d[xcol].values, Xr)
    pr = float(np.corrcoef(ex, er)[0, 1])
    return {
        "n": len(d), "k": X.shape[1], "df": len(d) - X.shape[1],
        "first_D": str(d.D.min().date()), "last_D": str(d.D.max().date()),
        "c": float(b[-1]), "t_nw5_cal_small": float(b[-1] / se_cal[-1]), "se_nw5_cal_small": float(se_cal[-1]),
        "t_nw5_small": float(b[-1] / se_nw[-1]), "t_nw5_nosmall": float(b[-1] / se_nw_ns[-1]),
        "t_classic": float(b[-1] / se_cl[-1]), "partial_r": pr, "delta_R2": float(dR2),
        "R2_full": float(1 - e @ e / tss), "controls_used": use,
        "dropped_const": [c for c in ctrls if c not in use],
    }


h03 = {}
h03["main_panel_feature"] = h03_fit(K, "A_rz8", har)
h03["main_no_fomc"] = h03_fit(K, "A_rz8", [c for c in har if c != "m0_cal_fomc"])
h03["mine_rz8"] = h03_fit(K, "A_rz8_mine", [c for c in har if c != "m0_cal_fomc"])
h03["saved_cap5"] = h03_fit(K, "A_saved_rz8", [c for c in har if c != "m0_cal_fomc"])
# 내 가격 재구성으로 바꾼 판(대상·HAR 통제)
K2 = K.copy()
for c in ["fut_kospi_lnHL", "m0_kospi_ln_absr", "m0_kospi_lnRV5", "m0_kospi_lnRV22", "m0_kospi_rneg",
          "m0_sox_ln_absr_u", "m0_ewy_ln_absr_u", "m0_kospi_ln_vol_anom", "m0_kospi_absg"]:
    K2[c] = kr_my[c].values
h03["my_prices"] = h03_fit(K2, "A_rz8", [c for c in har if c != "m0_cal_fomc"])
# 일정 더미 시점 민감도: KR 옵션 만기 더미를 만기일 그 D로 옮긴 판
kr_opex_days = pd.to_datetime(["2026-06-11", "2026-07-09", "2026-08-13", "2026-09-10", "2026-10-08"])
K3 = K.copy()
K3["m0_cal_kr_opex"] = K3.D.isin(kr_opex_days).astype(float)
h03["kr_opex_on_expiry_day"] = h03_fit(K3, "A_rz8", [c for c in har if c != "m0_cal_fomc"])
# 강건: 급락 4일, 열어 본 8일
crash4 = pd.to_datetime(["2026-07-29", "2026-07-30", "2026-07-31", "2026-10-13"])
opened8 = pd.to_datetime(["2026-08-20", "2026-09-02", "2026-09-03", "2026-09-08", "2026-09-16", "2026-10-12", "2026-10-14", "2026-10-15"])
h03["robust_crash4"] = h03_fit(K[~K.D.isin(crash4)], "A_rz8", [c for c in har if c != "m0_cal_fomc"])
h03["robust_opened8"] = h03_fit(K[~K.D.isin(opened8)], "A_rz8", [c for c in har if c != "m0_cal_fomc"])
res["H03"] = h03


# ---------------------------------------------------------------- 방향 칸 공통
def direction_cell(df, xcol, ycol, m0, ctrls, h, daily_both, pos_map, extra_mask=None):
    daily, daily_lr = daily_both["simple"], daily_both["log"]
    d = df.copy()
    if extra_mask is not None:
        d = d[extra_mask(d)]
    d = d.dropna(subset=[xcol, ycol] + m0 + ctrls).sort_values("D").reset_index(drop=True)
    m0u = [c for c in m0 if d[c].nunique() > 1]
    x = d[xcol].values.astype(float)
    xp, r2, xhat = resid_std(x, d[m0u].values.astype(float))
    y = d[ycol].values.astype(float)
    C = d[ctrls].values.astype(float)
    pos = [int(pos_map[D]) for D in d.D]
    out = {"n": len(d), "first_D": str(d.D.min().date()), "last_D": str(d.D.max().date()),
           "M0_k": len(m0u), "M0_R2": float(r2), "M0_dropped_const": [c for c in m0 if c not in m0u]}
    out["ic_raw_X"] = float(stats.spearmanr(x, y)[0])
    out["ic_raw_Xperp"] = float(stats.spearmanr(xp, y)[0])
    out["ic_partial_rank_Xperp"] = partial_spearman(xp, y, C)
    out["ic_partial_rawresid_Xperp"] = partial_spearman_raw(xp, y, C)
    out["ic_partial_rank_X"] = partial_spearman(x, y, C)
    # t: R ~ rankz(X⊥) + 통제, Hodrick 1B(진입일 기준 하루 수익 = 종가 대 종가)
    xr = rankz(xp)
    t_h, se_h, beta = hodrick_1b_t(y, xr, C, pos, h, daily)
    XX = np.column_stack([np.ones(len(d)), C, xr])
    b, e, _ = ols(y, XX)
    out["beta_rankz"] = float(b[-1])
    out["t_hodrick1b"] = float(t_h)
    out["t_nw_lag_h"] = float(b[-1] / nw_se(XX, e, h)[-1])
    out["t_nw_lag_hm1"] = float(b[-1] / nw_se(XX, e, h - 1)[-1])
    out["t_classic"] = float(b[-1] / classic_se(XX, e)[-1])
    out["df"] = len(d) - XX.shape[1]
    # 원값 X⊥판
    t_h2, _, beta2 = hodrick_1b_t(y, xp, C, pos, h, daily)
    out["t_hodrick1b_rawXperp"] = float(t_h2)
    # 메아리 위약: X̂
    out["placebo_ic_partial_rank"] = partial_spearman(xhat, y, C)
    t_hp, _, _ = hodrick_1b_t(y, rankz(xhat), C, pos, h, daily)
    out["placebo_t_hodrick1b"] = float(t_hp)
    out["rho1_xperp_obsorder"] = float(np.corrcoef(xp[1:], xp[:-1])[0, 1])
    # 보고값 방식 재현판: ln(1+R) ~ 통제 + rankz(X⊥, SD ddof=1), Hodrick 1B(하루 로그 종가 대 종가, 평균 제거, n/(n−k))
    rk = stats.rankdata(xp)
    xr1 = (rk - rk.mean()) / rk.std(ddof=1)
    XXl = np.column_stack([np.ones(len(d)), C, xr1])
    t_rep, b_rep = hodrick_full_t(np.log1p(y), XXl, pos, h, daily_lr)
    out["repspec_t_hodrick1b"] = t_rep
    out["repspec_beta_rankz"] = b_rep
    rkh = stats.rankdata(xhat)
    t_rep_p, _ = hodrick_full_t(np.log1p(y), np.column_stack([np.ones(len(d)), C, (rkh - rkh.mean()) / rkh.std(ddof=1)]),
                                pos, h, daily_lr)
    out["repspec_placebo_t"] = t_rep_p
    out["_frame"] = d
    return out


# 하루 수익(종가 대 종가) 위치 사전
def daily_map(dd):
    lr = np.log1p(dd["r"])
    return {"simple": {int(p): float(r) for p, r in zip(dd["pos"], dd["r"]) if not np.isnan(r)},
            "log": {int(p): float(r) for p, r in zip(dd["pos"], lr) if not np.isnan(r)}}


def overlap_mask(df, dates):
    """창 시간 범위 (open_kst, close_kst]가 그 KST 달력일과 겹치면 True."""
    m = pd.Series(False, index=df.index)
    for d0 in pd.to_datetime(dates):
        d1 = d0 + pd.Timedelta(days=1)
        m |= (pd.to_datetime(df["open_kst"]) < d1) & (pd.to_datetime(df["close_kst"]) > d0)
    return m


# ---------------------------------------------------------------- H46
top = pd.read_parquet(RES / "feat_topic.parquet")
tu = top[top.table == "US_D"].copy()
tu["D"] = pd.to_datetime(tu["D"])
tu = tu.sort_values("D").reset_index(drop=True)
# Tilt 재계산 = ln(n_us / n_kr); 유효 창 = 한쪽 휴장 아님·장애 아님
tu["tilt_mine"] = np.log(tu["h46_n_us"] / tu["h46_n_kr"])
cmp("us.h46_tilt", tu["tilt_mine"], tu["h46_tilt"])
valid = (~tu["xholiday_mismatch"].astype(bool)) & (tu["outage"] != True) & tu["tilt_mine"].notna()
tv = tu[valid].reset_index(drop=True)
tv["tilt5"] = tv["tilt_mine"].rolling(5).mean()
tv["dtilt_A"] = tv["tilt5"] - tv["tilt5"].shift(20)  # 5창 평균 − 20창 전 5창 평균
tv["dtilt_B"] = tv["tilt5"] - tv["tilt_mine"].shift(20)  # 5창 평균 − 20창 전 Tilt 값
tu = tu.merge(tv[["win_id", "tilt5", "dtilt_A", "dtilt_B"]], on="win_id", how="left")
cmp("us.h46_tilt5(valid)", tu.loc[valid, "tilt5"], tu.loc[valid, "h46_tilt5"])
cmp("us.h46_dtilt(valid) vs A", tu.loc[valid, "dtilt_A"], tu.loc[valid, "h46_dtilt"])
cmp("us.h46_dtilt_alt(valid) vs B", tu.loc[valid, "dtilt_B"], tu.loc[valid, "h46_dtilt_alt"])

U = us.merge(tu[["win_id", "h46_dtilt", "h46_dtilt_alt", "dtilt_A", "dtilt_B", "h46_valid_win"]], on="win_id", how="left")
U["valid_h46"] = U["judge_ok"] & (~U["xholiday_mismatch"].astype(bool)) & (U.D >= "2026-07-01")
m0_us = ["m0_kospi_r_k", "m0_kospi_oc_k", "m0_skh_r_k", "m0_skh_oc_k", "m0_sox_r", "m0_sox_r5", "m0_sox_r20",
         "m0_sox_DD20", "m0_sox_lnRV20", "m0_vix_chg", "m0_nq_close_to_winclose", "m0_missing_rate"]
c_h46 = ["m0_soxx_r", "m0_soxx_absr", "m0_soxx_r5", "m0_soxx_r20", "m0_soxx_lnRV20", "m0_soxx_minus_ewy_r20",
         "m0_kospi_r20_k", "m0_nq_close_to_winclose", "m0_soxx_g"]
soxx_daily = daily_map(soxx)
soxx_pos = soxx["pos"].to_dict()
h46 = {}
for nm, xc in [("main_panel_dtilt", "h46_dtilt"), ("mine_dtilt_A", "dtilt_A"), ("alt_dtilt_B", "dtilt_B")]:
    o = direction_cell(U, xc, "fut_soxx_R10", m0_us, c_h46, 10, soxx_daily, soxx_pos,
                       extra_mask=lambda d: d["valid_h46"])
    o.pop("_frame")
    h46[nm] = o
# 비교: 7/1 제한 없이(특징 첫 값부터)
o = direction_cell(U, "h46_dtilt", "fut_soxx_R10", m0_us, c_h46, 10, soxx_daily, soxx_pos,
                   extra_mask=lambda d: d["judge_ok"] & ~d["xholiday_mismatch"].astype(bool))
o.pop("_frame")
h46["main_no_0701_filter"] = o
# M0에서 결번율 뺀 판(11변수)
o = direction_cell(U, "h46_dtilt", "fut_soxx_R10", [c for c in m0_us if c != "m0_missing_rate"], c_h46, 10,
                   soxx_daily, soxx_pos, extra_mask=lambda d: d["valid_h46"])
o.pop("_frame")
h46["m0_without_missing_rate"] = o
res["H46"] = h46

# ---------------------------------------------------------------- H66: P̄ 재계산
key = pd.read_parquet(ROOT / "label_key.parquet")
labs = []
for f in sorted((ROOT / "labels").glob("pred_*.json")):
    labs += json.loads(f.read_text(encoding="utf-8"))
lab = pd.DataFrame(labs)
L = key.merge(lab[["id", "pred"]], on="id", how="inner")
L["dt_kst"] = pd.to_datetime(L["dt_kst"])
L = L[L["dt_kst"] >= "2026-07-01"]  # LLM 지식 마감
L = L[L["pred"].between(1, 5)]
pw = {}
# (i) (창, 작성자, pred) 쌍 1회
pairs = L.drop_duplicates(["win_id", "a", "pred"])
pw["P_bar_pairs"] = pairs.groupby("win_id")["pred"].mean() - 3
# (ii) 작성자 평균 후 창 평균(작성자 1표)
pw["P_bar_authmean"] = L.groupby(["win_id", "a"])["pred"].mean().groupby("win_id").mean() - 3
# (iii) 단위 그대로
pw["P_bar_units"] = L.groupby("win_id")["pred"].mean() - 3
pw["n_pairs"] = pairs.groupby("win_id").size()
PW = pd.DataFrame(pw).reset_index().rename(columns={"index": "win_id"})
pwin = pd.read_parquet(RES / "pred_windows.parquet")
chkP = PW.merge(pwin[["win_id", "P_bar", "P_bar_unit", "n_pred"]], on="win_id", how="outer")
cmp("kr.P_bar(pairs) vs pred_windows.P_bar", chkP["P_bar_pairs"], chkP["P_bar"])
cmp("kr.P_bar(units) vs pred_windows.P_bar_unit", chkP["P_bar_units"], chkP["P_bar_unit"])
res["P_bar_windows"] = {"mine_windows": int(PW["P_bar_pairs"].notna().sum()),
                        "panel_windows": int(pwin["P_bar"].notna().sum()),
                        "windows_only_panel": sorted(set(pwin.loc[pwin.P_bar.notna(), "win_id"]) - set(PW.win_id)),
                        "windows_only_mine": sorted(set(PW.win_id) - set(pwin.loc[pwin.P_bar.notna(), "win_id"]))}

KK = kr.merge(PW, on="win_id", how="left").merge(pwin[["win_id", "P_bar"]], on="win_id", how="left")
m0_kr = ["m0_kospi_rpos", "m0_kospi_rneg", "m0_kospi_absr", "m0_kospi_r5", "m0_kospi_r20", "m0_kospi_r60",
         "m0_kospi_DD20", "m0_kospi_lnRV20", "m0_kospi_ln_vol_anom", "m0_sox_r_u", "m0_ewy_r_u",
         "m0_cal_weekend", "m0_cal_holiday", "m0_cal_xholiday", "m0_cal_us_opex", "m0_cal_fomc", "m0_cal_earn",
         "m0_cal_vacation", "m0_cal_kr_opex", "m0_missing_rate"]
c_h66 = ["m0_kospi_r", "m0_kospi_r5", "m0_kospi_r20", "m0_kospi_lnRV20", "m0_kospi_g", "m0_sox_r_u"]
kospi_daily = daily_map(kospi)
kospi_pos = kospi["pos"].to_dict()
h66 = {}
mask66 = lambda d: d["judge_ok"] & (d.D >= "2026-07-02")
for nm, xc in [("main_panel_Pbar", "P_bar"), ("mine_pairs", "P_bar_pairs"), ("mine_author_mean", "P_bar_authmean"),
               ("mine_units", "P_bar_units")]:
    o = direction_cell(KK, xc, "fut_kospi_R5", m0_kr, c_h66, 5, kospi_daily, kospi_pos, extra_mask=mask66)
    fr = o.pop("_frame")
    h66[nm] = o
# M0 결측(6/29 불량 봉 파급) 대신 M0 변수 중 결측 많은 r60을 뺀 판
o = direction_cell(KK, "P_bar", "fut_kospi_R5", [c for c in m0_kr if c != "m0_kospi_r60"], c_h66, 5,
                   kospi_daily, kospi_pos, extra_mask=mask66)
o.pop("_frame")
h66["m0_without_r60"] = o
# 강건: 급락 4일·열어 본 8일 제외
o = direction_cell(KK, "P_bar", "fut_kospi_R5", m0_kr, c_h66, 5, kospi_daily, kospi_pos,
                   extra_mask=lambda d: mask66(d) & ~d.D.isin(crash4))
o.pop("_frame")
h66["robust_crash4"] = o
o = direction_cell(KK, "P_bar", "fut_kospi_R5", m0_kr, c_h66, 5, kospi_daily, kospi_pos,
                   extra_mask=lambda d: mask66(d) & ~d.D.isin(opened8))
o.pop("_frame")
h66["robust_opened8"] = o
res["H66"] = h66

# ---------------------------------------------------------------- 강건판(창 시간 범위가 그 달력일과 겹치면 뺌)
ctr03 = [c for c in har if c != "m0_cal_fomc"]
h03["robust_crash4_overlap"] = h03_fit(K[~overlap_mask(K, crash4)], "A_rz8", ctr03)
h03["robust_opened8_overlap"] = h03_fit(K[~overlap_mask(K, opened8)], "A_rz8", ctr03)
for nm, dates in [("robust_crash4_overlap", crash4), ("robust_opened8_overlap", opened8)]:
    o = direction_cell(U, "h46_dtilt", "fut_soxx_R10", [c for c in m0_us if c != "m0_missing_rate"], c_h46, 10,
                       soxx_daily, soxx_pos, extra_mask=lambda d, dd=dates: d["valid_h46"] & ~overlap_mask(d, dd))
    o.pop("_frame")
    h46[nm] = o
    o = direction_cell(KK, "P_bar", "fut_kospi_R5", m0_kr, c_h66, 5, kospi_daily, kospi_pos,
                       extra_mask=lambda d, dd=dates: mask66(d) & ~overlap_mask(d, dd))
    o.pop("_frame")
    h66[nm] = o


# 기여 상위 2창: 빼면 통계가 예상 부호에서 가장 멀어지는 두 관측(한 개씩 뺀 재계산, LOO)
def top2_h03(df, xcol, ctrls, exp_sign=+1):
    d = df[df.judge_ok & (df.D >= "2026-07-01")].dropna(subset=["fut_kospi_lnHL", xcol] + ctrls).sort_values("D")
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in ctrls] + [d[xcol].values])
    y = d["fut_kospi_lnHL"].values
    loo = np.array([ols(np.delete(y, i), np.delete(X, i, 0))[0][-1] for i in range(len(d))])
    w = d.win_id.values[np.argsort(exp_sign * loo)[:2]].tolist()
    return w, h03_fit(df[~df.win_id.isin(w)], xcol, ctrls)


def top2_dir(fr, xcol, ycol, m0, ctrls, exp_sign, h, daily_both, pos_map):
    xp, _, _ = resid_std(fr[xcol].values.astype(float), fr[m0].values.astype(float))
    y = fr[ycol].values.astype(float)
    C = fr[ctrls].values.astype(float)
    loo = np.array([partial_spearman(np.delete(xp, i), np.delete(y, i), np.delete(C, i, 0)) for i in range(len(fr))])
    w = fr.win_id.values[np.argsort(exp_sign * loo)[:2]].tolist()
    o = direction_cell(fr, xcol, ycol, m0, ctrls, h, daily_both, pos_map, extra_mask=lambda d: ~d.win_id.isin(w))
    o.pop("_frame")
    return w, o


m0_us11 = [c for c in m0_us if c != "m0_missing_rate"]
fr46 = direction_cell(U, "h46_dtilt", "fut_soxx_R10", m0_us11, c_h46, 10, soxx_daily, soxx_pos,
                      extra_mask=lambda d: d["valid_h46"])["_frame"]
fr66 = direction_cell(KK, "P_bar", "fut_kospi_R5", m0_kr, c_h66, 5, kospi_daily, kospi_pos, extra_mask=mask66)["_frame"]
w03, o03 = top2_h03(K, "A_rz8", ctr03, +1)
w46, o46 = top2_dir(fr46, "h46_dtilt", "fut_soxx_R10", m0_us11, c_h46, -1, 10, soxx_daily, soxx_pos)
w66, o66 = top2_dir(fr66, "P_bar", "fut_kospi_R5", m0_kr, c_h66, -1, 5, kospi_daily, kospi_pos)
h03["robust_top2_loo"] = dict(o03, dropped=w03)
h46["robust_top2_loo"] = dict(o46, dropped=w46)
h66["robust_top2_loo"] = dict(o66, dropped=w66)

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
