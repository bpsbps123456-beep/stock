"""미리보기(부록 9) 4단계: 사전 없이 되는 칸 시험 — H03·H08·H35·H46·H31 + 설명용 표 2개.

지위: 미리보기다. 동결 판정(2026 앞 1~5월 같은 부호 + 뒤 단측)을 대신하지 않는다. 앞 구간 포럼 자료가 없다.
정의: HYPOTHESES.md §2·§3 그대로. 결과를 보기 전에 정한 해석(코드 상수·아래 주석)은 test_free.md '해석' 절에 적는다.

입력(읽기만): preview/results/panel_kr·panel_us(m0_* 지난 값, fut_* 앞날 값), feat_activity·feat_topic·feat_text,
             prices.parquet(가설 SE의 1기간 수익과 설명용 표), loader_missing_by_day.csv(하루 번호 증가).
방화벽: 특징(X)은 feat_* 파일 값 그대로다. M0 적합·잔차화는 m0_* 열만 쓴다(진입 순간 값 m0_*_g는 판정 통제로만).
        fut_* 열은 결과(y)로만 쓴다. Hodrick SE의 1기간 수익(종가 대 종가)은 분산 계산에만 쓰고 점추정에는 쓰지 않는다.
산출: preview/results/test_free.json (test_free.md는 test_free_report.py가 쓴다)
실행: py test_free.py
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PREVIEW = HERE.parent
ANT = PREVIEW.parent
RES = PREVIEW / "results"

SHIFT_MIN = 21          # §2.7 p ①: 21~(N−21) 원형 이동 전부
HAC_L = 5               # §3 F1 HAR-K HAC(5)
EXPECTED = {"IC": 0.10, "T1": 0.15}   # §2.7 기대 효과(사전 지정값)
CRASH_DAYS = [date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31), date(2026, 10, 13)]
OPENED_DAYS = [date(2026, 8, 20), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 8), date(2026, 9, 16),
               date(2026, 10, 12), date(2026, 10, 14), date(2026, 10, 15)]
DUMMY_MIN = 2           # 표본 안에서 1(또는 0)이 2개 미만인 더미는 뺀다(h01 규칙)

# HAR-K(§3 F1). 전일 VKOSPI는 원천 없음. 거래대금 이상치는 거래량 이상치로 대신(panel_meta).
HAR_CONT = ["m0_kospi_ln_absr", "m0_kospi_lnRV5", "m0_kospi_lnRV22", "m0_kospi_rneg", "m0_sox_ln_absr_u",
            "m0_ewy_ln_absr_u", "m0_kospi_ln_vol_anom", "m0_kospi_absg", "m0_missing_rate"]
HAR_DUM = ["wd_1", "wd_2", "wd_3", "wd_4", "m0_cal_fomc", "m0_cal_earn", "m0_cal_kr_opex", "m0_cal_us_opex"]
# M0(§2.3). 한국: |r|은 max·min과 완전 공선이라 뺀다(h01 notes). VKOSPI 없음.
M0_KR_CONT = ["m0_kospi_rpos", "m0_kospi_rneg", "m0_kospi_r5", "m0_kospi_r20", "m0_kospi_r60", "m0_kospi_DD20",
              "m0_kospi_lnRV20", "m0_kospi_ln_vol_anom", "m0_sox_r_u", "m0_ewy_r_u", "m0_missing_rate"]
M0_KR_DUM = ["m0_cal_weekend", "m0_cal_holiday", "m0_cal_xholiday", "m0_cal_vacation", "m0_cal_kr_opex",
             "m0_cal_us_opex", "m0_cal_fomc", "m0_cal_earn"]
M0_US_CONT = ["m0_kospi_r_k", "m0_kospi_oc_k", "m0_skh_r_k", "m0_skh_oc_k", "m0_sox_r", "m0_sox_r5", "m0_sox_r20",
              "m0_sox_DD20", "m0_sox_lnRV20", "m0_vix_chg", "m0_nq_close_to_winclose"]
H46_CTRL = ["m0_soxx_r", "m0_soxx_absr", "m0_soxx_r5", "m0_soxx_r20", "m0_soxx_lnRV20", "m0_soxx_minus_ewy_r20",
            "m0_kospi_r20_k", "m0_nq_close_to_winclose", "m0_soxx_g"]
H31_CTRL = ["m0_soxx_r", "m0_soxx_r5", "m0_soxx_r20", "m0_soxx_lnRV20", "m0_soxx_g"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 자료
def load() -> dict:
    pk = pd.read_parquet(RES / "panel_kr.parquet").reset_index(drop=True)
    pu = pd.read_parquet(RES / "panel_us.parquet").reset_index(drop=True)
    for p in (pk, pu):
        p["pos"] = np.arange(len(p))
        p["Dd"] = pd.to_datetime(p["D"]).dt.date
        for k in (1, 2, 3, 4):
            p[f"wd_{k}"] = (p["m0_weekday"] == k).astype(float)
    fa = pd.read_parquet(RES / "feat_activity.parquet")
    ft = pd.read_parquet(RES / "feat_topic.parquet")
    fx = pd.read_parquet(RES / "feat_text.parquet")
    ka = fa[fa["market"] == "kr"][["win_id", "A_rz8", "A_saved_rz8", "fight", "fight_pc", "ln_n_comments_onposts",
                                   "trunc_share", "trunc_share_pc", "NW90_rz8", "NW90", "NW30_rz8"]]
    kr = pk.merge(ka, on="win_id", how="left", validate="1:1")
    tu = ft[ft["table"] == "US_D"][["win_id", "h46_dtilt", "h46_valid_win"]]
    xu = fx[fx["market"] == "us"][["win_id", "LL_US_d20", "LL_US_noNVDL_d20"]]
    us = pu.merge(tu, on="win_id", how="left", validate="1:1").merge(xu, on="win_id", how="left", validate="1:1")
    assert (kr["win_id"].to_numpy() == pk["win_id"].to_numpy()).all()
    assert (us["win_id"].to_numpy() == pu["win_id"].to_numpy()).all()
    px = pd.read_parquet(RES / "prices.parquet")
    px["date"] = pd.to_datetime(px["date"]).dt.date
    bad = px["flag_flat_bar"] & px["flag_zero_volume"]
    px.loc[bad, ["open", "high", "low", "close"]] = np.nan
    P = {t: g.set_index("date").sort_index() for t, g in px.groupby("ticker")}
    return {"kr": kr, "us": us, "px": P}


def cc_by_window(df: pd.DataFrame, pxd: pd.DataFrame) -> np.ndarray:
    """창 위치 s의 1기간 로그수익 = ln(C_{D_s} ÷ 직전 세션 종가). Hodrick SE 분산 계산 전용."""
    c = pxd["close"].astype(float)
    lr = np.log(c / c.shift(1))
    return np.array([float(lr.get(d, np.nan)) for d in df["Dd"]])


def overlap_mask(df: pd.DataFrame, days: list[date]) -> np.ndarray:
    """창 (open, close]이 달력일 [d 00:00, d+1 00:00)과 시간이 겹치면 True(h01 주 읽기: 가장 많이 뺀다)."""
    o = pd.to_datetime(df["open_kst"])
    c = pd.to_datetime(df["close_kst"])
    m = np.zeros(len(df), bool)
    for d in days:
        a = pd.Timestamp(d)
        b = a + pd.Timedelta(days=1)
        m |= ((o < b) & (c >= a)).to_numpy()
    return m


# ---------------------------------------------------------------- 통계
def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def build_design(df: pd.DataFrame, protect: list[np.ndarray], cont: list[str], dums: list[str]):
    """[상수, 보호 열(특징 등)] 뒤에 연속 통제, 더미를 하나씩 더한다. 랭크를 늘리지 않는 열과 1(또는 0)이 2개 미만인
    더미는 뺀다. 빼는 결정은 통제 열과 표본에만 기대고 결과(y)와 무관하다."""
    n = len(df)
    cols = [np.ones(n)] + list(protect)
    names, dropped = [], []
    for c in cont + dums:
        v = df[c].to_numpy(float)
        if c in dums:
            ones = int(np.nansum(v))
            if ones < DUMMY_MIN or n - ones < DUMMY_MIN:
                dropped.append(c)
                continue
        trial = np.column_stack(cols + [v])
        if np.linalg.matrix_rank(trial) > np.linalg.matrix_rank(np.column_stack(cols)):
            cols.append(v)
            names.append(c)
        else:
            dropped.append(c)
    return np.column_stack(cols), names, dropped


def hac_cov(X, e, pos, L):
    """Newey–West(Bartlett). 지연 l의 짝은 창 위치 차가 정확히 l인 행끼리만. 작은 표본 보정 n/(n−k)."""
    n, k = X.shape
    inv = np.linalg.pinv(X.T @ X)
    u = X * e[:, None]
    S = u.T @ u
    where = {int(p): i for i, p in enumerate(pos)}
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        G = np.zeros((k, k))
        for i, p in enumerate(pos):
            j = where.get(int(p) - l)
            if j is not None:
                G += np.outer(u[i], u[j])
        S += w * (G + G.T)
    return inv @ S @ inv * (n / (n - k))


def hodrick_cov(X, pos, eps_all: np.ndarray, h: int):
    """Hodrick(1992) 1B. h기간 수익 = 1기간 수익 h개의 합이라는 귀무 아래 점수 Σ_t X_t e_t = Σ_s ε_s w_s,
    w_s = Σ_{j=0}^{h−1} X_{s−j}(표본 행만). ε_s = 쓰는 세션 구간의 1기간 로그수익 − 평균(결측 세션은 건너뜀)."""
    n, k = X.shape
    inv = np.linalg.pinv(X.T @ X)
    where = {int(p): i for i, p in enumerate(pos)}
    s_lo, s_hi = int(min(pos)), min(int(max(pos)) + h - 1, len(eps_all) - 1)
    mu = np.nanmean(eps_all[s_lo:s_hi + 1])
    S = np.zeros((k, k))
    for s in range(s_lo, s_hi + 1):
        e = eps_all[s]
        if not np.isfinite(e):
            continue
        w = np.zeros(k)
        for j in range(h):
            i = where.get(s - j)
            if i is not None:
                w += X[i]
        S += (e - mu) ** 2 * np.outer(w, w)
    return inv @ S @ inv * (n / (n - k))


def tstat(y, X, j, cov: str, pos, L=HAC_L, eps=None, h=None) -> dict:
    beta, e = ols(y, X)
    V = hac_cov(X, e, pos, L) if cov == "hac" else hodrick_cov(X, pos, eps, h)
    se = math.sqrt(max(V[j, j], 0.0))
    n, k = X.shape
    return {"beta": float(beta[j]), "se": se, "t": float(beta[j] / se) if se > 0 else float("nan"), "df": int(n - k)}


def r2(y, X) -> float:
    _, e = ols(y, X)
    d = y - y.mean()
    return float(1 - (e @ e) / (d @ d))


def partial_corr(y, x, C):
    _, ey = ols(y, C)
    _, ex = ols(x, C)
    return float(np.corrcoef(ex, ey)[0, 1]), ex, ey


def rank(v):
    return stats.rankdata(v)


def zs(v):
    return (v - v.mean()) / v.std(ddof=1)


def spearman_partial(x, y, Ctrl):
    """순위 x·y를 순위 통제에 잔차화한 Pearson(= 순위 부분 상관). 통제가 없으면 Spearman."""
    rx, ry = rank(x), rank(y)
    if Ctrl is None or Ctrl.shape[1] == 0:
        C = np.ones((len(x), 1))
    else:
        C = np.column_stack([np.ones(len(x))] + [rank(Ctrl[:, i]) for i in range(Ctrl.shape[1])])
    return partial_corr(ry, rx, C)


def ac1_pos(v, pos) -> float:
    where = {int(p): i for i, p in enumerate(pos)}
    a, b = [], []
    for i, p in enumerate(pos):
        j = where.get(int(p) - 1)
        if j is not None:
            a.append(v[j])
            b.append(v[i])
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 3 else float("nan")


def n_eff(n: int, rho_x: float, h: int) -> float:
    """Bartlett 근사: 특징은 AR(1)(ρ_x), h일 목표는 겹침 삼각형(1 − k/h). 1일 목표면 n."""
    r = max(rho_x, 0.0) if np.isfinite(rho_x) else 0.0
    den = 1 + 2 * sum((r ** k) * (1 - k / h) for k in range(1, h))
    return float(min(n, n / den))


def shifts(nf: int) -> list[int]:
    return list(range(SHIFT_MIN, nf - SHIFT_MIN + 1))


def shift_p(obs: float, null: np.ndarray, sign: int):
    if not len(null):
        return None
    return float((1 + np.sum(sign * null >= sign * obs)) / (1 + len(null)))


def rolled(x_full: np.ndarray, F: np.ndarray, k: int) -> np.ndarray:
    """F(특징 유효 창, 시간순) 안에서 특징열을 k칸 원형 이동. 결과·통제는 제자리."""
    out = x_full.copy()
    idx = np.where(F)[0]
    out[idx] = np.roll(x_full[idx], k)
    return out


def verdict_shape(signed_est: float, se: float, p_adj: float, alpha: float, robust_ok: bool, placebo_flag: bool,
                  half: float) -> str:
    """§2.7을 뒤 구간 조건만으로 본 모양. 앞 구간(1~5월) 같은 부호 조건은 자료가 없어 늘 '확인 없음'."""
    up = signed_est + 1.645 * se
    if signed_est < 0 or up < half:
        return "기각 모양"
    if placebo_flag and p_adj > alpha:
        return "기각 모양(메아리 위약: 가격만으로 만든 X̂가 같은 부호로 70% 이상, X는 미통과)"
    if p_adj <= alpha and not placebo_flag:
        return "통과 모양(앞 구간 확인 없음)" if robust_ok else "취약 모양"
    return "판정 불가 모양"


# ---------------------------------------------------------------- T1: HAR-K 회귀 칸
def har_fit(df: pd.DataFrame, S: np.ndarray, x: np.ndarray, extra: list[str]) -> dict:
    d = df[S]
    y = d["fut_kospi_lnHL"].to_numpy(float)
    xs = x[S]
    prot = [xs] + [d[c].to_numpy(float) for c in extra]
    X, names, dropped = build_design(d, prot, HAR_CONT, HAR_DUM)
    t = tstat(y, X, 1, "hac", d["pos"].to_numpy())
    C = np.delete(X, 1, axis=1)
    pr, ex, ey = partial_corr(y, xs, C)
    dr2 = r2(y, X) - r2(y, C)
    return {"t": t, "pr": pr, "dr2": dr2, "ex": ex, "ey": ey, "names": names, "dropped": dropped, "n": int(S.sum()),
            "r2_full": r2(y, X)}


def m0_hat(df: pd.DataFrame, S: np.ndarray, x: np.ndarray, cont: list[str], dums: list[str]) -> dict:
    """X를 M0에 표본 안 OLS로 적합(동결 확장 OLS 최소 120창을 못 채워 미리보기에서 대신). X̂ = 적합값, X⊥ = 잔차."""
    d = df[S]
    X, names, dropped = build_design(d, [], cont, dums)
    xs = x[S]
    _, e = ols(xs, X)
    n, k = X.shape
    R2 = r2(xs, X)
    return {"xhat": xs - e, "xperp": e, "R2": R2, "adjR2": 1 - (1 - R2) * (n - 1) / (n - k), "k": int(k),
            "names": names, "dropped": dropped}


def har_cell(kr: pd.DataFrame, cid: str, xcol: str, extra: list[str], aux: dict) -> dict:
    sign = +1
    base = kr["judge_ok"].to_numpy(bool)
    F = base & kr[xcol].notna().to_numpy()
    need = ["fut_kospi_lnHL", xcol] + HAR_CONT + extra + M0_KR_CONT + M0_KR_DUM
    S = base & kr[need].notna().all(axis=1).to_numpy()   # 위약과 같은 표본을 쓰려고 M0 결측 창도 뺀다
    x = kr[xcol].to_numpy(float)
    fit = har_fit(kr, S, x, extra)
    t = fit["t"]
    p_an = float(stats.t.sf(sign * t["t"], t["df"]))
    ks = shifts(int(F.sum()))
    null_t = np.array([har_fit(kr, S, rolled(x, F, k), extra)["t"]["t"] for k in ks])
    p_sh = shift_p(t["t"], null_t, sign)
    p_sh_opp = shift_p(t["t"], null_t, -sign)
    p_final = max(p_an, p_sh) if p_sh is not None else p_an
    # 메아리 위약: X̂ = M0 표본 안 적합값을 X 자리에 넣은 같은 회귀
    mh = m0_hat(kr, S, x, M0_KR_CONT, M0_KR_DUM)
    xh = np.full(len(kr), np.nan)
    xh[S] = mh["xhat"]
    fph = har_fit(kr, S, xh, extra)
    null_r2 = [m0_hat(kr, S, rolled(x, F, k), M0_KR_CONT, M0_KR_DUM)["R2"] for k in ks]
    placebo_ratio = fph["pr"] / fit["pr"] if fit["pr"] != 0 else float("nan")
    placebo_flag = bool(np.sign(fph["pr"]) == np.sign(fit["pr"]) and abs(fph["pr"]) >= 0.7 * abs(fit["pr"]))
    # 강건 3개(§2.7)
    rob = {}
    for key, m in (("crash4", S & ~overlap_mask(kr, CRASH_DAYS)), ("opened8", S & ~overlap_mask(kr, OPENED_DAYS))):
        f2 = har_fit(kr, m, x, extra)
        rob[key] = {"n": f2["n"], "c": f2["t"]["beta"], "pr": f2["pr"], "ratio": f2["t"]["beta"] / t["beta"]}
    contrib = sign * fit["ex"] * fit["ey"]
    top2 = np.argsort(contrib)[-2:]
    sidx = np.where(S)[0]
    m = S.copy()
    m[sidx[top2]] = False
    f2 = har_fit(kr, m, x, extra)
    rob["top2"] = {"n": f2["n"], "c": f2["t"]["beta"], "pr": f2["pr"], "ratio": f2["t"]["beta"] / t["beta"],
                   "dropped_win": [str(kr.loc[i, "win_id"]) for i in sidx[top2]]}
    robust_ok = bool(all(r["ratio"] >= 0.5 for r in rob.values()))
    se_pr = abs(fit["pr"]) / abs(t["t"]) if abs(t["t"]) > 1e-9 else 1 / math.sqrt(t["df"])
    out = {"id": cid, "family": "T1", "target": "KOSPI D일 ln HL (HAR-K 계수 c)", "feature": xcol,
           "extra_controls": extra, "n_obs": fit["n"], "n_feature_windows": int(F.sum()),
           "first_D": str(kr.loc[S, "Dd"].min()), "last_D": str(kr.loc[S, "Dd"].max()),
           "estimate_c": t["beta"], "se_hac5": t["se"], "t_hac5": t["t"], "df": t["df"], "partial_r": fit["pr"],
           "delta_R2": fit["dr2"], "p_analytic_one": p_an, "shift_K": len(ks),
           "shift_min_p": (1 / (1 + len(ks))) if ks else None, "p_shift_one": p_sh, "p_final_one": p_final,
           "p_shift_opposite_info": p_sh_opp, "R2_har_full": fit["r2_full"],
           "expected_sign": "+", "sign_ok": bool(t["beta"] > 0),
           "placebo": {"c": fph["t"]["beta"], "t": fph["t"]["t"], "partial_r": fph["pr"],
                       "ratio_partial_r": placebo_ratio, "flag_70pct_same_sign": placebo_flag,
                       "M0_R2": mh["R2"], "M0_adjR2": mh["adjR2"], "M0_k": mh["k"],
                       "M0_R2_chance_mean": float(np.mean(null_r2)) if null_r2 else None,
                       "echo_dominant_R2_ge_0.5": bool(mh["R2"] >= 0.5), "M0_dropped": mh["dropped"]},
           "robust": rob, "robust_ok": robust_ok, "controls_used": fit["names"], "controls_dropped": fit["dropped"],
           "mde_partial_r": 2.49 / math.sqrt(t["df"]),
           "ci90_signed_upper": sign * fit["pr"] + 1.645 * se_pr, "se_partial_r": se_pr, "aux": {}}
    for name, spec in aux.items():
        col, ex2 = spec[0], spec[1]
        F2 = base & kr[col].notna().to_numpy()
        S2 = base & kr[["fut_kospi_lnHL", col] + HAR_CONT + ex2].notna().all(axis=1).to_numpy()
        if len(spec) > 2:   # 표본 제한(보조): 주판과 같은 창
            S2 = S2 & spec[2]
        x2 = kr[col].to_numpy(float)
        f3 = har_fit(kr, S2, x2, ex2)
        ks2 = shifts(int(F2.sum()))
        nt2 = np.array([har_fit(kr, S2, rolled(x2, F2, k), ex2)["t"]["t"] for k in ks2])
        out["aux"][name] = {"feature": col, "extra_controls": ex2, "n": f3["n"], "c": f3["t"]["beta"],
                            "t": f3["t"]["t"], "df": f3["t"]["df"], "partial_r": f3["pr"],
                            "p_analytic_one": float(stats.t.sf(f3["t"]["t"], f3["t"]["df"])),
                            "shift_K": len(ks2), "p_shift_one": shift_p(f3["t"]["t"], nt2, +1)}
    return out


# ---------------------------------------------------------------- T2: IC 칸
def ic_calc(df: pd.DataFrame, S: np.ndarray, x: np.ndarray, ycol: str, kind: str, h: int, m0c: list[str],
            m0d: list[str], ctrl: list[str], eps, full: bool = True) -> dict:
    """주 IC = X⊥(M0 표본 안 잔차)와 목표의 순위 부분 상관(판정 통제: 대상 r·r5·r20·RV20·g 등)."""
    d = df[S]
    y = d[ycol].to_numpy(float)
    if kind == "R":
        y = np.log1p(y)
    mh = m0_hat(df, S, x, m0c, m0d)
    xp = mh["xperp"]
    C = d[ctrl].to_numpy(float) if ctrl else None
    ic, ex, ey = spearman_partial(xp, y, C)
    out = {"ic": ic, "ex": ex, "ey": ey, "n": int(S.sum())}
    if not full:
        return out
    pos = d["pos"].to_numpy()
    Xr = np.column_stack([np.ones(len(y)), zs(rank(xp))] + ([C] if C is not None else []))
    if kind == "R":
        t = tstat(y, Xr, 1, "hodrick", pos, eps=eps, h=h)
        se_kind = f"Hodrick 1B (h={h})"
    else:
        t = tstat(y, Xr, 1, "hac", pos, L=h)
        se_kind = f"NW HAC({h}) — MDD는 1기간 수익의 합으로 나뉘지 않아 Hodrick 1B 대신"
    xs = x[S]
    ic_raw = float(stats.spearmanr(xs, y)[0])
    ic_xc, _, _ = spearman_partial(xs, y, C)
    ic_ph, _, _ = spearman_partial(mh["xhat"], y, C)
    Xh = np.column_stack([np.ones(len(y)), zs(rank(mh["xhat"]))] + ([C] if C is not None else []))
    th = tstat(y, Xh, 1, "hodrick", pos, eps=eps, h=h) if kind == "R" else tstat(y, Xh, 1, "hac", pos, L=h)
    out.update({"t": t, "se_kind": se_kind, "ic_raw": ic_raw, "ic_x_ctrl": ic_xc, "ic_placebo": ic_ph,
                "t_placebo": th["t"], "m0": mh, "rho_x": ac1_pos(xp, pos)})
    return out


def ic_cell(df: pd.DataFrame, cid: str, xcol: str, ycol: str, kind: str, h: int, m0c, m0d, ctrl, eps,
            sign: int, extra_mask, target_label: str, aux_cols: dict | None = None) -> dict:
    base = df["judge_ok"].to_numpy(bool)
    if extra_mask is not None:
        base = base & extra_mask
    F = base & df[xcol].notna().to_numpy()
    S = base & df[[xcol, ycol] + m0c + m0d + ctrl].notna().all(axis=1).to_numpy()
    x = df[xcol].to_numpy(float)
    r = ic_calc(df, S, x, ycol, kind, h, m0c, m0d, ctrl, eps)
    t = r["t"]
    p_an = float(stats.t.sf(sign * t["t"], t["df"]))
    ks = shifts(int(F.sum()))
    null = np.array([ic_calc(df, S, rolled(x, F, k), ycol, kind, h, m0c, m0d, ctrl, eps, full=False)["ic"]
                     for k in ks])
    null_r2 = [m0_hat(df, S, rolled(x, F, k), m0c, m0d)["R2"] for k in ks]
    p_sh = shift_p(r["ic"], null, sign)
    p_sh_opp = shift_p(r["ic"], null, -sign)
    p_final = max(p_an, p_sh) if p_sh is not None else p_an
    ne = n_eff(r["n"], r["rho_x"], h)
    rob = {}
    for key, m in (("crash4", S & ~overlap_mask(df, CRASH_DAYS)), ("opened8", S & ~overlap_mask(df, OPENED_DAYS))):
        rr = ic_calc(df, m, x, ycol, kind, h, m0c, m0d, ctrl, eps, full=False)
        rob[key] = {"n": rr["n"], "ic": rr["ic"], "ratio": rr["ic"] / r["ic"]}
    contrib = sign * r["ex"] * r["ey"]
    top2 = np.argsort(contrib)[-2:]
    sidx = np.where(S)[0]
    m = S.copy()
    m[sidx[top2]] = False
    rr = ic_calc(df, m, x, ycol, kind, h, m0c, m0d, ctrl, eps, full=False)
    rob["top2"] = {"n": rr["n"], "ic": rr["ic"], "ratio": rr["ic"] / r["ic"],
                   "dropped_win": [str(df.loc[i, "win_id"]) for i in sidx[top2]]}
    robust_ok = bool(all(v["ratio"] >= 0.5 for v in rob.values()))
    ratio_ph = r["ic_placebo"] / r["ic_x_ctrl"] if r["ic_x_ctrl"] != 0 else float("nan")
    placebo_flag = bool(np.sign(r["ic_placebo"]) == np.sign(r["ic_x_ctrl"])
                        and abs(r["ic_placebo"]) >= 0.7 * abs(r["ic_x_ctrl"]))
    out = {"id": cid, "family": "T2", "target": target_label, "feature": xcol, "controls": ctrl,
           "n_obs": r["n"], "n_feature_windows": int(F.sum()), "first_D": str(df.loc[S, "Dd"].min()),
           "last_D": str(df.loc[S, "Dd"].max()),
           "ic_main_xperp_partial": r["ic"], "ic_raw": r["ic_raw"], "ic_x_ctrl_partial": r["ic_x_ctrl"],
           "t": t["t"], "se_kind": r["se_kind"], "beta_rankz": t["beta"], "df": t["df"], "p_analytic_one": p_an,
           "shift_K": len(ks), "shift_min_p": (1 / (1 + len(ks))) if ks else None, "p_shift_one": p_sh,
           "p_final_one": p_final, "p_shift_opposite_info": p_sh_opp, "expected_sign": "+" if sign > 0 else "−",
           "sign_ok": bool(np.sign(r["ic"]) == sign),
           "placebo": {"ic": r["ic_placebo"], "t": r["t_placebo"], "ratio_vs_x_ctrl": ratio_ph,
                       "flag_70pct_same_sign": placebo_flag, "M0_R2": r["m0"]["R2"], "M0_adjR2": r["m0"]["adjR2"],
                       "M0_k": r["m0"]["k"], "M0_R2_chance_mean": float(np.mean(null_r2)) if null_r2 else None,
                       "echo_dominant_R2_ge_0.5": bool(r["m0"]["R2"] >= 0.5), "M0_dropped": r["m0"]["dropped"]},
           "rho1_xperp": r["rho_x"], "n_eff": ne, "mde_ic": 2.49 / math.sqrt(ne),
           "ci90_signed_upper": sign * r["ic"] + 1.645 / math.sqrt(ne),
           "robust": rob, "robust_ok": robust_ok, "aux": {}}
    for name, col in (aux_cols or {}).items():
        F2 = base & df[col].notna().to_numpy()
        S2 = base & df[[col, ycol] + m0c + m0d + ctrl].notna().all(axis=1).to_numpy()
        x2 = df[col].to_numpy(float)
        a = ic_calc(df, S2, x2, ycol, kind, h, m0c, m0d, ctrl, eps)
        ks2 = shifts(int(F2.sum()))
        n2 = np.array([ic_calc(df, S2, rolled(x2, F2, k), ycol, kind, h, m0c, m0d, ctrl, eps, full=False)["ic"]
                       for k in ks2])
        out["aux"][name] = {"feature": col, "n": a["n"], "ic_main": a["ic"], "ic_raw": a["ic_raw"],
                            "t": a["t"]["t"], "p_analytic_one": float(stats.t.sf(sign * a["t"]["t"], a["t"]["df"])),
                            "shift_K": len(ks2), "p_shift_one": shift_p(a["ic"], n2, sign)}
    return out


# ---------------------------------------------------------------- 다중 비교(미리보기)
def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def bh(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    q = [min(1.0, p * m / (i + 1)) for i, (_, p) in enumerate(items)]
    for i in range(m - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    return {k: q[i] for i, (k, _) in enumerate(items)}


# ---------------------------------------------------------------- 설명용
def daily_tables(P: dict) -> dict:
    md = pd.read_csv(RES / "loader_missing_by_day.csv", encoding="utf-8-sig")
    md["date"] = pd.to_datetime(md["date"]).dt.date
    posts = md.set_index("date")["d_no"].astype(float)
    out = {"source": "loader_missing_by_day.csv d_no(KST 달력일 0시 경계 번호 증가, 지운 글 포함)",
           "days": int(len(posts)), "first": str(min(posts.index)), "last": str(max(posts.index)),
           "lag_rule": "lag k: corr(그날 글 수(t), 수익(t+k)), t·k는 그 시장 거래일. k>0 = 글이 먼저, k<0 = 가격이 먼저"}
    lag, noise = {}, {}
    for tk, lab in (("^KS11", "kospi"), ("SOXX", "soxx")):
        c = P[tk]["close"].astype(float)
        r = c / c.shift(1) - 1
        sess = [d for d in r.index if min(posts.index) <= d <= max(posts.index)]
        pv = np.array([posts.get(d, np.nan) for d in sess])
        rv = r.reindex(sess).to_numpy(float)
        for ylab, yv in (("absr", np.abs(rv)), ("r", rv)):
            rows = []
            for k in range(-10, 11):
                if k >= 0:
                    a, b = pv[:len(pv) - k], yv[k:]
                else:
                    a, b = pv[-k:], yv[:len(yv) + k]
                ok = np.isfinite(a) & np.isfinite(b)
                rows.append({"lag": k, "n": int(ok.sum()), "spearman": float(stats.spearmanr(a[ok], b[ok])[0]),
                             "pearson_lnposts": float(np.corrcoef(np.log(a[ok]), b[ok])[0, 1])})
            lag[f"{lab}_{ylab}"] = rows
            ok = np.isfinite(pv) & np.isfinite(yv)
            a, b = pv[ok], yv[ok]
            ks = shifts(len(a))
            nl = [abs(stats.spearmanr(np.roll(a, k), b)[0]) for k in ks]
            noise[f"{lab}_{ylab}"] = {"K": len(ks), "p95_abs_spearman_lag0": float(np.percentile(nl, 95))}
    out["lead_lag"] = lag
    out["noise_lag0_circular"] = noise
    # 시점 감사(audit.md ⑤): 종가 대 종가 수익의 구간은 앞 세션 종가에서 시작한다. 그 시작이 그날 24:00 전이면
    # 그날 글과 수익 구간이 겹친다(KOSPI +1일: 그날 15:30 시작, SOXX 0일: 그날 05:00 KST 시작). 겹치지 않는 판:
    # KOSPI +1일 = 다음 세션 시가(09:00) → 종가. KOSPI +2일부터·SOXX +1일부터는 종가 대 종가 그대로 겹치지 않는다.
    kc0 = P["^KS11"]["close"].astype(float)
    ko0 = P["^KS11"]["open"].astype(float)
    ksess0 = [d for d in kc0.index if min(posts.index) <= d <= max(posts.index)]
    pv0 = np.array([posts.get(d, np.nan) for d in ksess0])
    oc0 = (kc0 / ko0 - 1).reindex(ksess0).to_numpy(float)
    clean1 = {}
    for ylab, yv in (("absr", np.abs(oc0)), ("r", oc0)):
        a, b = pv0[:-1], yv[1:]
        ok = np.isfinite(a) & np.isfinite(b)
        clean1[ylab] = {"lag": 1, "n": int(ok.sum()), "spearman": float(stats.spearmanr(a[ok], b[ok])[0]),
                        "ret": "다음 세션 시가→종가"}
    out["lead_lag_clean"] = {"kospi_lag1": clean1, "overlap_lags": {"kospi": [0, 1], "soxx": [0]},
                             "rule": "종가 대 종가 수익 구간이 그날 24:00 전에 시작하는 시차는 겹친다. KOSPI +1일만 "
                                     "다음 세션 시가→종가로 바꾼 판을 낸다. SOXX 0일은 겹침 표시만 한다"}
    kc = P["^KS11"]["close"].astype(float)
    kr_ = kc / kc.shift(1) - 1
    ksess = sorted(kc.index)
    rows = []
    for d, v in posts.sort_values(ascending=False).head(10).items():
        same = float(kr_[d]) if d in kr_.index and np.isfinite(kr_[d]) else None
        nxt = next((s for s in ksess if s > d), None)
        ko_ = P["^KS11"]["open"].astype(float)
        oc_nxt = float(kc[nxt] / ko_[nxt] - 1) if nxt and np.isfinite(ko_[nxt]) and np.isfinite(kc[nxt]) else None
        rows.append({"date": str(d), "weekday": "월화수목금토일"[d.weekday()], "d_no": int(v),
                     "krx_open_that_day": bool(d in kr_.index), "kospi_r_that_day": same,
                     "next_session": str(nxt) if nxt else None,
                     "kospi_r_next_session": (float(kr_[nxt]) if nxt and np.isfinite(kr_[nxt]) else None),
                     "kospi_oc_next_session": oc_nxt})
    out["top10"] = rows
    out["median_d_no"] = float(posts.median())
    a, b = date(2026, 6, 1), date(2026, 10, 16)
    kk = kr_[[d for d in kr_.index if a <= d <= b]].abs()
    out["kospi_mean_abs_r_0601_1016"] = float(kk.mean())
    out["kospi_n_sessions_0601_1016"] = int(kk.notna().sum())
    return out


# ---------------------------------------------------------------- 실행
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k not in ("ex", "ey")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return None
    return o


def main() -> int:
    t0 = datetime.now().replace(microsecond=0)
    Dt = load()
    kr, us, P = Dt["kr"], Dt["us"], Dt["px"]
    eps_soxx = cc_by_window(us, P["SOXX"])
    cells = {}
    cells["H03"] = har_cell(kr, "H03", "A_rz8", [], aux={"저장 글 수판(부호 확인)": ("A_saved_rz8", [])})
    a_ok = kr["A_rz8"].notna().to_numpy()
    cells["H08"] = har_cell(kr, "H08", "fight", ["A_rz8", "ln_n_comments_onposts", "trunc_share"],
                            aux={"A 뺀 판(A 결측 창 포함, 보조)": ("fight", ["ln_n_comments_onposts", "trunc_share"]),
                                 "A 뺀 판, A 있는 창만(보조)": ("fight", ["ln_n_comments_onposts", "trunc_share"], a_ok),
                                 "마감 전 잘림판 fight_pc(보조)": ("fight_pc", ["A_rz8", "ln_n_comments_onposts",
                                                                                "trunc_share_pc"])})
    valid46 = us["h46_valid_win"].fillna(False).astype(bool).to_numpy()
    cells["H46"] = ic_cell(us, "H46", "h46_dtilt", "fut_soxx_R10", "R", 10, M0_US_CONT, [], H46_CTRL, eps_soxx,
                           -1, valid46, "SOXX R10 IC")
    cells["H31"] = ic_cell(us, "H31", "LL_US_d20", "fut_soxx_MDD10", "MDD", 10, M0_US_CONT, [], H31_CTRL, eps_soxx,
                           -1, None, "SOXX MDD10 IC", aux_cols={"NVDL 뺀 판(강건)": "LL_US_noNVDL_d20"})
    j = kr["judge_ok"].to_numpy(bool)
    nw90_valid = int((j & kr["NW90_rz8"].notna()).sum())
    f30 = j & kr["NW30_rz8"].notna().to_numpy()
    m30 = f30 & kr["fut_kospi_R10"].notna().to_numpy()
    ic30 = float(stats.spearmanr(kr.loc[m30, "NW30_rz8"], kr.loc[m30, "fut_kospi_R10"])[0]) if m30.sum() > 3 else None
    cells["H35"] = {"id": "H35", "family": "T2", "target": "KOSPI R10 IC", "feature": "NW90_rz8",
                    "n_obs": nw90_valid, "testable": False, "expected_sign": "−",
                    "note": "NW90(90일 이력)은 8/28부터 31창이고 rz8은 같은 요일 직전 8창이 필요해 등록 특징 값이 0창이다",
                    "explanatory_NW30_rz8": {"n": int(m30.sum()), "n_feature_windows": int(f30.sum()),
                                             "ic_raw": ic30, "shift_K": len(shifts(int(f30.sum()))),
                                             "label": "비등록 30일 이력판, 통제 없음, 설명용"}}
    t1 = {k: cells[k]["p_final_one"] for k in ("H03", "H08")}
    t2 = {k: cells[k]["p_final_one"] for k in ("H46", "H31")}
    corr = {"T1_holm_adj": holm(t1), "T2_bh_q_H46_H31": bh(t2), "T2_bh_q_H46_only": {"H46": t2["H46"]},
            "note": "미리보기 표시다. 최종 T1 Holm은 H64·H65·H67·H70, T2 BH는 H60~H63·H66·H68·H69·H71~H74를 더해 "
                    "실제 시험 칸 수로 다시 센다. H31은 8월 규제 시행일 미확보로 동결 규칙상 잠정 설명용이다"}
    for k in ("H03", "H08"):
        c = cells[k]
        c["verdict_shape"] = verdict_shape(c["partial_r"], c["se_partial_r"], corr["T1_holm_adj"][k], 0.05,
                                           c["robust_ok"], c["placebo"]["flag_70pct_same_sign"], EXPECTED["T1"] / 2)
    for k in ("H46", "H31"):
        c = cells[k]
        c["verdict_shape"] = verdict_shape(-1 * c["ic_main_xperp_partial"], 1 / math.sqrt(c["n_eff"]),
                                           corr["T2_bh_q_H46_H31"][k], 0.10, c["robust_ok"],
                                           c["placebo"]["flag_70pct_same_sign"], EXPECTED["IC"] / 2)
    desc = daily_tables(P)
    inputs = ["panel_kr.parquet", "panel_us.parquet", "feat_activity.parquet", "feat_topic.parquet",
              "feat_text.parquet", "prices.parquet", "loader_missing_by_day.csv"]
    out = {"made": t0.isoformat(), "status": "미리보기(부록 9). 동결 판정 아님. 앞 구간(1~5월) 포럼 자료 없음",
           "cells": cells, "corrections": corr, "descriptive": desc,
           "inputs_sha256": {f: sha(RES / f) for f in inputs},
           "frozen_sha256": {f: sha(ANT / f) for f in ("HYPOTHESES.md", "HYPOTHESES_addendum4.md",
                                                       "HYPOTHESES_addendum9.md")}}
    (RES / "test_free.json").write_text(json.dumps(clean(out), ensure_ascii=False, indent=1, default=str), "utf-8")
    for k, c in cells.items():
        print(k, {kk: c.get(kk) for kk in ("n_obs", "estimate_c", "t_hac5", "partial_r", "ic_main_xperp_partial",
                                            "t", "p_analytic_one", "p_shift_one", "shift_K", "verdict_shape")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
