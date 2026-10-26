"""V2 반박 시도 (2026-10-12): 최종 사양 FINAL_E14s10_HYB의 주장 6개를 깨려고 공격한다.

주장
 1 front(1~5월) S 등급 h20 초과 +7%p 안팎, S>A>B>C>D 순서
 2 back(6~10월) 거꾸로(S −3%p, D +6%p)
 3 7월 충격(6/22 종가 → 7/29 종가)에서 S가 가장 크게 빠졌다(−42% 대 전체 −17%)
 4 9월 이후 다시 맞는 중(9월 신호 IC20 +0.24)
 5 2025년 효과 0, 음수 달은 급락·급반등 무렵(2·4·5월)
 6 S 등급의 이후 20일 최대 낙폭이 전체보다 깊다

공격: 신호 1일 지연, 종가→종가(t+1 종가 매수·t 종가 매수 두 가지), 보유 시점 대신 패널 150종목 전체,
기여 상위 종목 빼기(빈도 상위 3, 기여 상위 3·5), 날짜 블록 부트스트랩(10·20일, 원형, 5000회),
5분위 경계, 평활 0·5·20일, 관문 없음, 사건 날짜 흔들기, 9월 독립 표본 수·종목 부트스트랩·순열 검정.

실행: cd data/derived/analysis/trend_grade/code && PYTHONIOENCODING=utf-8 py verify_v2_rebut.py
결과: ../results/verify_v2_rebut.json. 저장소의 다른 파일은 고치지 않는다(harness.membership은 실행 중에만 잠시 바꾼다).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import fam_composite as FC  # noqa: E402
import fam_ladder as FL  # noqa: E402
import harness as H  # noqa: E402

GR = H.GRADES
P = H.load_panel()
C, O, HI, LO = P["C"], P["O"], P["H"], P["L"]
CAL, COLS = C.index, C.columns
QC, QO = P["bench"]["C"], P["bench"]["O"]
S50, S200 = FL.sma(C, 50), FL.sma(C, 200)
BARS_OK = C.notna().cumsum() >= H.MIN_BARS
NXT = O.shift(-1).notna()
MEM0 = H.membership().copy()
RNG = np.random.default_rng(20261012)
REPS = 5000
T0 = time.time()


def r_(x, k=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, k) if np.isfinite(x) else None


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ 구성 요소
def eligible_of(M: pd.DataFrame) -> pd.DataFrame:
    return M & BARS_OK & C.notna() & NXT


def score_with(M: pd.DataFrame, smooth: int) -> pd.DataFrame:
    """composite 안의 단면 백분위가 쓰는 보유 종목군을 M으로 바꿔 E14 점수를 다시 낸다."""
    orig = H.membership
    H.membership = lambda: M
    try:
        sc = FC.make(FC.GRID["E14"], smooth)(P)
    finally:
        H.membership = orig
    return sc.reindex(index=CAL, columns=COLS)


def grades(sc: pd.DataFrame, E: pd.DataFrame, cuts=H.PYRAMID, gate=True) -> pd.DataFrame:
    pr = H.pct_rank(sc, E & sc.notna())
    g = H.grade_of(pr, cuts).to_numpy(object).copy()
    if gate:
        gt = ((C > S50) & (S50 > S200)).to_numpy()
        dc = ((C < S200) & (S50 < S200)).to_numpy()
        graded = pd.notna(g)
        top = graded & ((g == "S") | (g == "A"))
        g[top & ~gt] = "B"
        g[graded & dc] = "D"
    return pd.DataFrame(g, index=CAL, columns=COLS)


_FWD = {}


def fwd(kind: str, h: int) -> pd.DataFrame:
    key = (kind, h)
    if key not in _FWD:
        if kind == "oo":  # t+1 시가 → t+1+h 시가 (기준)
            _FWD[key] = H.forward(h)
        elif kind == "cc":  # t+1 종가 → t+1+h 종가
            _FWD[key] = C.shift(-(1 + h)) / C.shift(-1) - 1
        elif kind == "cc0":  # t 종가 → t+h 종가 (신호 종가에 바로 산다고 가정, 낙관적)
            _FWD[key] = C.shift(-h) / C - 1
        else:
            raise ValueError(kind)
    return _FWD[key]


def path_mdd20() -> pd.DataFrame:
    """t+1 시가 매수 뒤 20거래일 종가 경로의 고점 대비 최대 낙폭(진짜 최대 낙폭)."""
    entry = O.shift(-1)
    runmax = entry.copy()
    worst = pd.DataFrame(0.0, index=CAL, columns=COLS)
    for k in range(1, 21):
        px = C.shift(-k)
        runmax = np.fmax(runmax, px)
        worst = np.fmin(worst, px / runmax - 1)
    return worst.where(entry.notna() & C.shift(-20).notna())


MDD_LOW = H.forward_mdd20()  # harness 방식: 진입가 대비 20일 최저 저가
MDD_PATH = path_mdd20()


def build(name, M=None, smooth=10, cuts=H.PYRAMID, gate=True, lag=0, fkind="oo", drop=()):
    M = (MEM0 if M is None else M).copy()
    if drop:
        M.loc[:, [d for d in drop if d in M.columns]] = False
    E = eligible_of(M)
    sc = score_with(M, smooth)
    G = grades(sc, E, cuts, gate)
    if lag:
        sc, G = sc.shift(lag), G.shift(lag)
    G = G.where(E)
    sc = sc.where(G.notna())
    return {"name": name, "sc": sc, "G": G, "E": E, "fkind": fkind}


def window_frames(V, win, h):
    dates = H.window_dates(win, h)
    G = V["G"].loc[dates]
    fr = fwd(V["fkind"], h).loc[dates].where(G.notna())
    m = fr.mean(axis=1)
    ex = pd.DataFrame({k: fr.where(G == k).mean(axis=1) - m for k in GR})
    sc = V["sc"].loc[dates]
    pr = sc.rank(axis=1, pct=True)
    ic = H._row_spearman(pr, fr)
    return dates, G, fr, m, ex, ic


def ladder(V, win, h=20) -> dict:
    dates, G, fr, m, ex, ic = window_frames(V, win, h)
    if len(dates) < 5:
        return {"n_dates": int(len(dates))}
    mean = {k: r_(ex[k].mean() * 100) for k in GR}
    vals = [mean[k] for k in GR]
    mono = all(v is not None for v in vals) and all(vals[i] > vals[i + 1] for i in range(4))
    out = {"n_dates": int(len(dates)), "ex": mean, "mono": mono,
           "ic": r_(ic.mean(), 4), "ic_t": r_(H.newey_west_t(ic.dropna(), h), 2),
           "S_count": r_((G == "S").sum(axis=1).mean(), 1)}
    if h == 20:
        for tag, MD in (("mdd_low", MDD_LOW), ("mdd_path", MDD_PATH)):
            md = MD.loc[dates].where(G.notna())
            out[tag] = {k: r_(md.where(G == k).mean(axis=1).mean() * 100, 2) for k in GR}
            out[tag]["all"] = r_(md.mean(axis=1).mean() * 100, 2)
            out[tag]["S_minus_all"] = r_(out[tag]["S"] - out[tag]["all"], 2)
    return out


# ------------------------------------------------------------------ 부트스트랩
def mbb_idx(n: int, L: int, reps: int = REPS) -> np.ndarray:
    nb = int(np.ceil(n / L))
    starts = RNG.integers(0, n, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % n  # 원형 블록
    return idx.reshape(reps, -1)[:, :n]


def ci(x, scale=100.0):
    x = np.asarray(x, float) * scale
    x = x[np.isfinite(x)]
    return [r_(np.percentile(x, 2.5), 2), r_(np.percentile(x, 97.5), 2)]


def boot_window(V, win, h=20, Ls=(10, 20)) -> dict:
    dates, G, fr, m, ex, ic = window_frames(V, win, h)
    X = ex[list(GR)].to_numpy(float)
    y = ic.to_numpy(float)
    md = MDD_LOW.loc[dates].where(G.notna())
    mdS = (md.where(G == "S").mean(axis=1) - md.mean(axis=1)).to_numpy(float)
    mp = MDD_PATH.loc[dates].where(G.notna())
    mpS = (mp.where(G == "S").mean(axis=1) - mp.mean(axis=1)).to_numpy(float)
    out = {"n_dates": int(len(dates)), "nonoverlap_windows": int(np.ceil(len(dates) / h))}
    for L in Ls:
        idx = mbb_idx(len(X), L)
        with np.errstate(all="ignore"):
            mu = np.nanmean(X[idx], axis=1)
            icm = np.nanmean(y[idx], axis=1)
            dS = np.nanmean(mdS[idx], axis=1)
            dP = np.nanmean(mpS[idx], axis=1)
        mono = np.all(mu[:, :-1] > mu[:, 1:], axis=1)
        out[f"L{L}"] = {
            "S": ci(mu[:, 0]), "A": ci(mu[:, 1]), "C": ci(mu[:, 3]), "D": ci(mu[:, 4]),
            "S_minus_D": ci(mu[:, 0] - mu[:, 4]),
            "S_gt_A_prob": r_(np.mean(mu[:, 0] > mu[:, 1]), 3),
            "mono_prob": r_(mono.mean(), 3),
            "P_S_le_0": r_(np.mean(mu[:, 0] <= 0), 4), "P_S_ge_0": r_(np.mean(mu[:, 0] >= 0), 4),
            "P_D_le_0": r_(np.mean(mu[:, 4] <= 0), 4),
            "ic": ci(icm, 1.0), "P_ic_le_0": r_(np.mean(icm <= 0), 4), "P_ic_ge_0": r_(np.mean(icm >= 0), 4),
            "mddS_minus_all_low": ci(dS), "P_mddS_shallower_low": r_(np.mean(dS >= 0), 4),
            "mddS_minus_all_path": ci(dP), "P_mddS_shallower_path": r_(np.mean(dP >= 0), 4),
        }
    # 겹치지 않는 표본: 시작 오프셋 0..h-1마다 h일 간격 날짜만 쓴 S 초과·순서
    offs = []
    for o in range(h):
        sub = ex.iloc[o::h]
        mu = sub.mean()
        offs.append((mu["S"] * 100, bool(all(mu.iloc[i] > mu.iloc[i + 1] for i in range(4))), len(sub)))
    s = np.array([a for a, _, _ in offs])
    out["nonoverlap_offsets"] = {"S_min": r_(s.min(), 2), "S_median": r_(np.median(s), 2), "S_max": r_(s.max(), 2),
                                 "S_pos_share": r_(np.mean(s > 0), 3), "mono_share": r_(np.mean([b for _, b, _ in offs]), 3),
                                 "n_per_offset": [int(min(n for *_, n in offs)), int(max(n for *_, n in offs))]}
    return out


def monthly(V, win_a, win_b, h=20) -> dict:
    """신호월별 S·D 초과(h20)와 IC (구간 안 날짜, 매도일이 패널 안)."""
    ex_d = H.exit_date(h)
    dates = CAL[(CAL >= pd.Timestamp(win_a)) & (CAL <= pd.Timestamp(win_b)) & ex_d.notna().to_numpy()]
    G = V["G"].loc[dates]
    fr = fwd(V["fkind"], h).loc[dates].where(G.notna())
    m = fr.mean(axis=1)
    exS = fr.where(G == "S").mean(axis=1) - m
    exD = fr.where(G == "D").mean(axis=1) - m
    ic = H._row_spearman(V["sc"].loc[dates].rank(axis=1, pct=True), fr)
    df = pd.DataFrame({"ic": ic, "S": exS * 100, "D": exD * 100})
    g = df.groupby(df.index.to_period("M"))
    return {str(k): {"ic": r_(v["ic"].mean(), 3), "S": r_(v["S"].mean(), 2), "D": r_(v["D"].mean(), 2), "days": int(len(v))}
            for k, v in g}


# ------------------------------------------------------------------ 기여 종목
def contributors(V, win, grade, h=20, top=8) -> list:
    dates, G, fr, m, ex, ic = window_frames(V, win, h)
    isk = G == grade
    nk = isk.sum(axis=1).replace(0, np.nan)
    part = fr.sub(m, axis=0).where(isk).div(nk, axis=0)
    nd = int(nk.notna().sum())
    contrib = part.sum(axis=0) / nd * 100
    freq = isk.sum(axis=0)
    df = pd.DataFrame({"contrib_pp": contrib, "days": freq}).query("days > 0")
    return df


# ------------------------------------------------------------------ 사건(7월 충격)
def event(V, d0, d1, entry="close") -> dict:
    d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
    g = V["G"].loc[d0]
    if entry == "close":
        px0 = C.loc[d0]
    else:
        nxt = CAL[CAL.get_loc(d0) + 1]
        px0 = O.loc[nxt]
    r = C.loc[d1] / px0 - 1
    ok = g.notna() & r.notna()
    base = float(r[ok].mean())
    out = {"all": r_(base * 100, 2), "n_all": int(ok.sum()),
           "QQQ": r_((QC.loc[d1] / (QC.loc[d0] if entry == "close" else QO.loc[CAL[CAL.get_loc(d0) + 1]]) - 1) * 100, 2)}
    rets = {}
    for k in GR:
        sel = ok & (g == k)
        out[f"n_{k}"] = int(sel.sum())
        rets[k] = r_(r[sel].mean() * 100, 2) if sel.any() else None
    out["ret"] = rets
    valid = {k: v for k, v in rets.items() if v is not None}
    out["worst"] = min(valid, key=valid.get) if valid else None
    out["S_is_worst"] = out["worst"] == "S"
    out["S_median"] = r_(r[ok & (g == "S")].median() * 100, 2) if (ok & (g == "S")).any() else None
    return out


def event_detail(V, d0, d1) -> dict:
    d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
    g = V["G"].loc[d0]
    r = (C.loc[d1] / C.loc[d0] - 1) * 100
    S = g[g == "S"].index
    sr = r[S].sort_values()
    loo = [r_(sr.drop(s).mean(), 2) for s in sr.index]
    ok = g.notna() & r.notna()
    allm = r[ok].mean()
    # S를 빼고도 A가 S보다 덜 빠졌는가, S에서 가장 크게 빠진 1·2종목 빼면?
    return {"S_names": {k: r_(v, 1) for k, v in sr.items()},
            "S_leave_one_out_range": [min(loo), max(loo)],
            "S_drop_worst2": r_(sr.iloc[2:].mean(), 2),
            "A_mean": r_(r[ok & (g == "A")].mean(), 2), "all_mean": r_(allm, 2),
            "all_vol_matched": vol_matched(g, r, ok, d0)}


def vol_matched(g, r, ok, d0) -> dict:
    """S 종목과 이전 60일 변동성이 비슷한 비S 종목(가까운 3개씩)의 같은 기간 수익 — '그냥 변동성 큰 종목이라 빠졌나'."""
    lr = np.log(C / C.shift(1))
    vol = lr.rolling(60, min_periods=40).std().loc[d0]
    S = [s for s in g[g == "S"].index if np.isfinite(vol.get(s, np.nan))]
    pool = [s for s in g[ok & (g != "S")].index if np.isfinite(vol.get(s, np.nan))]
    picks = []
    for s in S:
        d = (vol[pool] - vol[s]).abs().sort_values()
        picks += list(d.index[:3])
    picks = sorted(set(picks))
    return {"S_vol60_mean": r_(vol[S].mean() * np.sqrt(252) * 100, 1),
            "matched_vol60_mean": r_(vol[picks].mean() * np.sqrt(252) * 100, 1),
            "all_vol60_mean": r_(vol[pool].mean() * np.sqrt(252) * 100, 1),
            "matched_n": len(picks), "matched_ret": r_(r[picks].mean(), 2),
            "matched_grades": {k: int((g[picks] == k).sum()) for k in GR}}


# ------------------------------------------------------------------ 9월(주장4)
def september(V) -> dict:
    out = {}
    for h in (5, 10, 20):
        ex_d = H.exit_date(h)
        dates = CAL[(CAL >= pd.Timestamp("2026-09-01")) & (CAL <= pd.Timestamp("2026-09-30")) & ex_d.notna().to_numpy()]
        G = V["G"].loc[dates]
        fr = fwd(V["fkind"], h).loc[dates].where(G.notna())
        sc = V["sc"].loc[dates]
        ic = H._row_spearman(sc.rank(axis=1, pct=True), fr)
        m = fr.mean(axis=1)
        exS = (fr.where(G == "S").mean(axis=1) - m) * 100
        exD = (fr.where(G == "D").mean(axis=1) - m) * 100
        d = {"days": int(len(dates)), "first": str(dates[0].date()) if len(dates) else None,
             "last": str(dates[-1].date()) if len(dates) else None,
             "last_exit": str(ex_d.loc[dates[-1]].date()) if len(dates) else None,
             "nonoverlap_windows": int(np.ceil(len(dates) / h)),
             "ic_mean": r_(ic.mean(), 3), "ic_daily": [r_(v, 3) for v in ic.to_numpy()],
             "S_ex": r_(exS.mean(), 2), "D_ex": r_(exD.mean(), 2)}
        if h == 20 and len(dates):
            d.update(_stock_boot_perm(sc, fr))
        out[f"h{h}"] = d
    return out


def _stock_boot_perm(sc, fr, reps=REPS) -> dict:
    """종목 재표집(날짜 공통) 부트스트랩 CI와 종목 라벨 순열(날짜 공통) p값 — 7일이 사실상 한 표본이라 단면만 남는다."""
    cols = [c for c in sc.columns if sc[c].notna().all() and fr[c].notna().all()]
    X = sc[cols].rank(axis=1).to_numpy(float)  # 날짜 × 종목 순위
    Y = fr[cols].rank(axis=1).to_numpy(float)
    n = len(cols)

    def mean_ic(Xs, Ys):
        xc = Xs - Xs.mean(axis=-1, keepdims=True)
        yc = Ys - Ys.mean(axis=-1, keepdims=True)
        return ((xc * yc).sum(-1) / np.sqrt((xc ** 2).sum(-1) * (yc ** 2).sum(-1))).mean(-1)

    obs = float(mean_ic(X, Y))
    bi = RNG.integers(0, n, size=(reps, n))
    boot = np.array([mean_ic(X[:, b], Y[:, b]) for b in bi])
    perm = np.array([mean_ic(X[:, RNG.permutation(n)], Y) for _ in range(reps)])
    # 하루 단면 Spearman의 대략 표준오차 1/sqrt(n-1)
    return {"n_stocks_common": n, "ic_obs_common": r_(obs, 3), "stock_boot_ci95": ci(boot, 1.0),
            "perm_p_one_sided": r_(np.mean(perm >= obs), 4), "single_day_se": r_(1 / np.sqrt(n - 1), 3)}


# ------------------------------------------------------------------ 2025 장세(주장5)
def qqq_context(win_a, win_b, h=20) -> pd.DataFrame:
    """신호월별 QQQ 사정: 이후 h일 수익(t+1 시가→t+1+h 시가), 이후 h일 실현 변동성, 신호월~이후 h일의 고점 대비 낙폭·저점 대비 반등."""
    qret = QO.shift(-(1 + h)) / QO.shift(-1) - 1
    lr = np.log(QC / QC.shift(1))
    fvol = lr[::-1].rolling(h, min_periods=h).std()[::-1].shift(-1) * np.sqrt(252)
    rows = {}
    ex_d = H.exit_date(h)
    dates = CAL[(CAL >= pd.Timestamp(win_a)) & (CAL <= pd.Timestamp(win_b)) & ex_d.notna().to_numpy()]
    per = pd.Series(dates.to_period("M"), index=dates)
    for mth, ds in per.groupby(per):
        d0, d1 = ds.index[0], ex_d.loc[ds.index[-1]]
        seg = QC.loc[d0:d1]
        prior_hi = QC.loc[:d0].iloc[-60:].max()
        runmax = np.maximum.accumulate(np.r_[prior_hi, seg.to_numpy()])[1:]
        dd = float((seg.to_numpy() / runmax - 1).min())
        runmin = np.minimum.accumulate(seg.to_numpy())
        reb = float((seg.to_numpy() / runmin - 1).max())
        rows[str(mth)] = {"qqq_fwd20": float(qret.loc[ds.index].mean() * 100), "qqq_fwd_vol": float(fvol.loc[ds.index].mean() * 100),
                          "qqq_dd": dd * 100, "qqq_rebound": reb * 100}
    return pd.DataFrame(rows).T


def spearman(a, b) -> float:
    a, b = pd.Series(a), pd.Series(b)
    ok = a.notna() & b.notna()
    return float(a[ok].rank().corr(b[ok].rank()))


def regime_check(V) -> dict:
    mo = monthly(V, "2025-01-01", "2026-09-30")
    ctx = qqq_context("2025-01-01", "2026-09-30")
    df = ctx.copy()
    df["ic"] = [mo.get(k, {}).get("ic") for k in df.index]
    df = df.astype(float)
    df["turbulent"] = (df["qqq_dd"] <= -10) | (df["qqq_rebound"] >= 15)
    y25 = df[df.index.str.startswith("2025")]
    out = {"table": {k: {c: r_(v, 3 if c == "ic" else 1) for c, v in row.items() if c != "turbulent"} | {"turbulent": bool(row["turbulent"])}
                     for k, row in df.iterrows()},
           "rule": "turbulent = 신호월 첫날~마지막 신호일의 매도일 사이 QQQ 종가가 (직전 60일 고점 포함) 고점 대비 −10% 이하로 빠졌거나 저점 대비 +15% 이상 반등",
           "spearman_ic_vs_fwd_vol_all21": r_(spearman(df["ic"], df["qqq_fwd_vol"]), 3),
           "spearman_ic_vs_dd_all21": r_(spearman(df["ic"], df["qqq_dd"]), 3),
           "spearman_ic_vs_fwd20_all21": r_(spearman(df["ic"], df["qqq_fwd20"]), 3),
           "spearman_ic_vs_fwd_vol_2025": r_(spearman(y25["ic"], y25["qqq_fwd_vol"]), 3),
           "spearman_ic_vs_dd_2025": r_(spearman(y25["ic"], y25["qqq_dd"]), 3),
           "ic_turbulent_mean": r_(df.loc[df["turbulent"], "ic"].mean(), 3),
           "ic_calm_mean": r_(df.loc[~df["turbulent"], "ic"].mean(), 3),
           "turbulent_months": [k for k in df.index[df["turbulent"]]],
           "neg_months_2025": [k for k in y25.index if y25.loc[k, "ic"] < 0],
           "neg_lt_m0p1_2025": [k for k in y25.index if y25.loc[k, "ic"] < -0.1]}
    # 날짜 단위: IC20(t)와 QQQ 이후 20일 실현 변동성·수익의 상관(2025, 2026 각각)
    ex_d = H.exit_date(20)
    qret = QO.shift(-21) / QO.shift(-1) - 1
    lr = np.log(QC / QC.shift(1))
    fvol = lr[::-1].rolling(20, min_periods=20).std()[::-1].shift(-1)
    for yr in ("2025", "2026"):
        dates = CAL[(CAL >= pd.Timestamp(f"{yr}-01-01")) & (CAL <= pd.Timestamp(f"{yr}-12-31")) & ex_d.notna().to_numpy()]
        G = V["G"].loc[dates]
        fr = fwd(V["fkind"], 20).loc[dates].where(G.notna())
        ic = H._row_spearman(V["sc"].loc[dates].rank(axis=1, pct=True), fr)
        out[f"daily_corr_{yr}"] = {"ic_vs_qqq_fwd_vol": r_(spearman(ic, fvol.loc[dates]), 3),
                                   "ic_vs_qqq_fwd20": r_(spearman(ic, qret.loc[dates]), 3),
                                   "ic_vs_abs_qqq_fwd20": r_(spearman(ic, qret.loc[dates].abs()), 3), "days": int(len(dates))}
    return out


# ------------------------------------------------------------------ 실행
def summarize(V) -> dict:
    o = {w: ladder(V, w) for w in ("front", "back", "y2025")}
    o["late_h10"] = ladder(V, "late", 10)
    o["late_h5"] = ladder(V, "late", 5)
    o["july"] = event(V, "2026-06-22", "2026-07-29")
    o["sept"] = {f"h{h}": {k: v for k, v in d.items() if k in ("days", "ic_mean", "S_ex", "D_ex", "nonoverlap_windows")}
                 for h, d in ((int(k[1:]), d) for k, d in september(V).items())}
    return o


def line(name, o):
    f, b, y = o["front"], o["back"], o["y2025"]
    j = o["july"]
    return (f"{name:16s} F S {f['ex']['S']:+.2f} A {f['ex']['A']:+.2f} B {f['ex']['B']:+.2f} C {f['ex']['C']:+.2f} D {f['ex']['D']:+.2f} "
            f"mono {f['mono']} IC {f['ic']:+.3f} | B S {b['ex']['S']:+.2f} D {b['ex']['D']:+.2f} IC {b['ic']:+.3f} | "
            f"25 IC {y['ic']:+.3f} | jul S {j['ret']['S']} all {j['all']} worst {j['worst']} | sepIC20 {o['sept']['h20']['ic_mean']} "
            f"| mddS-all F {f['mdd_low']['S_minus_all']} B {b['mdd_low']['S_minus_all']} 25 {y['mdd_low']['S_minus_all']}")


def main():
    res = {"spec": "FINAL_E14s10_HYB (final.py) 재현 + 공격", "reps": REPS}
    base = build("base")
    # 재현 점검: fam_ladder.hyb_grades와 같은가
    ref = FL.hyb_grades(FC.make(FC.GRID["E14"], 10)(P), P, H.eligible()).where(H.eligible())
    same = (ref.fillna("_") == base["G"].fillna("_")).to_numpy().mean()
    res["reproduce_grade_match_share"] = r_(same, 6)
    log("grade match", same)

    variants = {"base": base,
                "lag1": build("lag1", lag=1),
                "cc_t1close": build("cc", fkind="cc"),
                "cc_tclose": build("cc0", fkind="cc0"),
                "univ150": build("univ150", M=pd.DataFrame(True, index=CAL, columns=COLS)),
                "quintile": build("quintile", cuts=H.QUINTILE),
                "smooth0": build("smooth0", smooth=0),
                "smooth5": build("smooth5", smooth=5),
                "smooth20": build("smooth20", smooth=20),
                "nogate": build("nogate", gate=False)}
    log("variants built")

    # 기여 종목
    cf = contributors(base, "front", "S")
    freq3 = list(cf.sort_values("days", ascending=False).index[:3])
    con3 = list(cf.sort_values("contrib_pp", ascending=False).index[:3])
    con5 = list(cf.sort_values("contrib_pp", ascending=False).index[:5])
    cbD = contributors(base, "back", "D")
    cbS = contributors(base, "back", "S")
    backD3 = list(cbD.sort_values("contrib_pp", ascending=False).index[:3])
    backS3 = list(cbS.sort_values("contrib_pp", ascending=True).index[:3])
    res["contributors"] = {
        "front_S_top_by_contrib": {k: [r_(v.contrib_pp, 2), int(v.days)] for k, v in cf.sort_values("contrib_pp", ascending=False).head(10).iterrows()},
        "front_S_top_by_days": {k: [r_(v.contrib_pp, 2), int(v.days)] for k, v in cf.sort_values("days", ascending=False).head(10).iterrows()},
        "back_D_top_by_contrib": {k: [r_(v.contrib_pp, 2), int(v.days)] for k, v in cbD.sort_values("contrib_pp", ascending=False).head(8).iterrows()},
        "back_S_most_negative": {k: [r_(v.contrib_pp, 2), int(v.days)] for k, v in cbS.sort_values("contrib_pp").head(8).iterrows()},
        "front_S_n_distinct": int(len(cf)),
        "note": "[기여 %p, S(또는 D)였던 날 수]. 기여 합 = 그 등급 평균 초과."}
    variants["drop_freq3"] = build("drop_freq3", drop=freq3)
    variants["drop_contrib3"] = build("drop_contrib3", drop=con3)
    variants["drop_contrib5"] = build("drop_contrib5", drop=con5)
    variants["drop_backD3"] = build("drop_backD3", drop=backD3)
    variants["drop_backS3"] = build("drop_backS3", drop=backS3)
    res["dropped"] = {"freq3": freq3, "contrib3": con3, "contrib5": con5, "backD3": backD3, "backS3": backS3}
    log("contrib variants built", res["dropped"])

    res["variants"] = {}
    for k, V in variants.items():
        o = summarize(V)
        res["variants"][k] = o
        log(line(k, o))

    # 부트스트랩(기준 + 주요 공격)
    res["bootstrap"] = {}
    for k in ("base", "lag1", "cc_t1close", "univ150", "drop_contrib3"):
        res["bootstrap"][k] = {w: boot_window(variants[k], w) for w in ("front", "back", "y2025")}
        b = res["bootstrap"][k]
        log(f"boot {k}: front S L20 {b['front']['L20']['S']} mono {b['front']['L20']['mono_prob']} IC {b['front']['L20']['ic']} | "
            f"back S {b['back']['L20']['S']} D {b['back']['L20']['D']} | 2025 IC {b['y2025']['L20']['ic']} | "
            f"mdd F {b['front']['L20']['mddS_minus_all_low']} B {b['back']['L20']['mddS_minus_all_low']} 25 {b['y2025']['L20']['mddS_minus_all_low']}")

    # 주장3 세부: 날짜 흔들기, 다음 날 시가 진입, 구성 종목
    jul = {"base_detail": event_detail(base, "2026-06-22", "2026-07-29")}
    starts = [d for d in CAL[(CAL >= "2026-06-08") & (CAL <= "2026-07-08")]]
    jul["start_sweep_to_0729"] = {str(d.date()): (lambda e: {"S": e["ret"]["S"], "all": e["all"], "worst": e["worst"], "nS": e["n_S"],
                                                             "A": e["ret"]["A"], "D": e["ret"]["D"]})(event(base, d, "2026-07-29"))
                                  for d in starts}
    ends = [d for d in CAL[(CAL >= "2026-07-15") & (CAL <= "2026-08-14")]]
    jul["end_sweep_from_0622"] = {str(d.date()): (lambda e: {"S": e["ret"]["S"], "all": e["all"], "worst": e["worst"]})(event(base, "2026-06-22", d))
                                  for d in ends}
    jul["open_entry_0623"] = event(base, "2026-06-22", "2026-07-29", entry="open")
    qseg = QC.loc["2026-06-01":"2026-08-31"]
    jul["qqq_peak_trough"] = {"peak": str(qseg.loc[:"2026-07-15"].idxmax().date()), "trough": str(qseg.loc["2026-07-01":].idxmin().date())}
    sw = jul["start_sweep_to_0729"]
    jul["start_sweep_summary"] = {"S_is_worst_share": r_(np.mean([v["worst"] == "S" for v in sw.values()]), 3),
                                  "S_range": [min(v["S"] for v in sw.values()), max(v["S"] for v in sw.values())],
                                  "n_starts": len(sw)}
    ew = jul["end_sweep_from_0622"]
    jul["end_sweep_summary"] = {"S_is_worst_share": r_(np.mean([v["worst"] == "S" for v in ew.values()]), 3), "n_ends": len(ew)}
    res["july"] = jul
    log("july", jul["start_sweep_summary"], jul["end_sweep_summary"], jul["qqq_peak_trough"])

    # 주장4
    res["september_base"] = september(base)
    late_boot = {}
    for h, L in ((5, 5), (10, 10)):
        dates, G, fr, m, ex, ic = window_frames(base, "late", h)
        y = ic.to_numpy(float)
        idx = mbb_idx(len(y), L)
        icm = np.nanmean(y[idx], axis=1)
        late_boot[f"h{h}_L{L}"] = {"n_dates": int(len(y)), "nonoverlap": int(np.ceil(len(y) / h)), "ic": r_(np.nanmean(y), 3),
                                   "ci95": ci(icm, 1.0), "P_le_0": r_(np.mean(icm <= 0), 4)}
    res["late_boot"] = late_boot
    log("sept", json.dumps(res["september_base"]["h20"], ensure_ascii=False))
    log("late boot", late_boot)

    # 주장5
    res["monthly_base"] = monthly(base, "2025-01-01", "2026-09-30")
    res["regime"] = regime_check(base)
    res["monthly_ic_2025_by_variant"] = {k: {m: v["ic"] for m, v in monthly(variants[k], "2025-01-01", "2025-12-31").items()}
                                         for k in ("base", "lag1", "cc_t1close", "smooth0", "smooth20")}
    log("regime", {k: v for k, v in res["regime"].items() if k != "table"})

    # 보강: 월별 IC 시계열의 흔들림·지속성(주장4), 월별 S 낙폭 부호(주장6), late 낙폭, 7월 2차원 날짜 격자(주장3)
    mb = res["monthly_base"]
    ks = sorted(mb)
    ics = np.array([mb[k]["ic"] for k in ks], float)
    full = np.array([mb[k]["days"] >= 15 for k in ks])
    icf = ics[full]
    pairs = [(ics[i], ics[i + 1]) for i in range(len(ks) - 1) if full[i] and full[i + 1]]
    a_, b_ = np.array(pairs).T
    res["monthly_ic_stats"] = {
        "months_full": int(full.sum()), "sd": r_(icf.std(ddof=1), 3), "mean": r_(icf.mean(), 3),
        "lag1_corr": r_(np.corrcoef(a_, b_)[0, 1], 3), "n_pairs": len(pairs),
        "next_pos_after_pos": r_(np.mean(b_[a_ > 0.2] > 0), 3), "n_after_pos": int((a_ > 0.2).sum()),
        "one_window_ci95_from_sd": [r_(0.243 - 1.96 * icf.std(ddof=1), 2), r_(0.243 + 1.96 * icf.std(ddof=1), 2)]}
    ex_d = H.exit_date(20)
    dates = CAL[(CAL >= pd.Timestamp("2025-01-01")) & (CAL <= pd.Timestamp("2026-09-30")) & ex_d.notna().to_numpy()]
    G = base["G"].loc[dates]
    md = MDD_LOW.loc[dates].where(G.notna())
    dS = (md.where(G == "S").mean(axis=1) - md.mean(axis=1)) * 100
    mm = dS.groupby(dS.index.to_period("M")).mean()
    res["mdd_S_minus_all_monthly"] = {str(k): r_(v, 2) for k, v in mm.items()}
    res["mdd_S_deeper_month_share"] = r_((mm < 0).mean(), 3)
    res["late_h20_base"] = ladder(base, "late", 20)
    grid = []
    for d0 in CAL[(CAL >= "2026-06-08") & (CAL <= "2026-07-08")]:
        for d1 in CAL[(CAL >= "2026-07-15") & (CAL <= "2026-08-14")]:
            e = event(base, d0, d1)
            grid.append((e["ret"]["S"], e["all"], e["worst"] == "S"))
    g = np.array([(s, a, w) for s, a, w in grid], float)
    res["july"]["grid2d"] = {"n_pairs": len(grid), "S_worst_share": r_(g[:, 2].mean(), 3),
                             "S_median": r_(np.median(g[:, 0]), 2), "S_min": r_(g[:, 0].min(), 2), "S_max": r_(g[:, 0].max(), 2),
                             "all_median": r_(np.median(g[:, 1]), 2), "S_minus_all_median": r_(np.median(g[:, 0] - g[:, 1]), 2),
                             "base_pair_rank_of_S": int((g[:, 0] < -42.33).sum())}
    log("extra", res["monthly_ic_stats"], res["mdd_S_deeper_month_share"], res["july"]["grid2d"],
        res["late_h20_base"].get("mdd_low"))

    path = os.path.join(H.RESULTS, "verify_v2_rebut.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
    log("saved", os.path.abspath(path))


if __name__ == "__main__":
    main()
