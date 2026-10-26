"""completeness_critic: 6개 갈래가 남긴 빈칸을 싸게 메우는 확인(2026-10-14, 모두 결과를 본 뒤 추가 = 사후).

harness·grid·triple_combos·leadlag은 import만 한다(수정 없음). 결과: results/completeness_critic.json.

절(실행: py completeness_critic.py [s1 s2 ...], 인자 없으면 전부)
  s1 변동성 기준선: '최근 변동성이 크면 21일 낙폭도 깊다'(변동성 군집)만으로 고점 쪽 확인 칸의 M21 차이가 나오는가.
     신호: QQQ 20일 실현변동성(종가 d−1까지) 250일 백분위 ≥ 0.8, ^VIX 종가(d−1) 250일 백분위 ≥ 0.8. 진입 d 시가.
  s2 셋 조합 '확인 11칸, p 0.005'를 2026 전체 원형 이동 p(verify_grid 대안)로 다시 세고, 같은 기준으로 귀무 200번(triple의 이동 거리 그대로).
  s3 SOXX 결과로 둘 조합 격자 고르기·확인(grid 규칙·귀무 seed 11 그대로). 사용자 보유가 AI 종목이라 QQQ만으로 고른 빈칸.
  s4 CFTC 값의 '측정일 기준' 나이(events가 짚은 셧다운 묵은 값): 2026 행 영향, 2025 보조 수치 재계산.
  s5 종목 횡단면(어느 갈래도 안 함): AI 종목별 OCC 고객 풋/콜·FINRA 공매도 거래량 비율·공매도 잔고 변화 → 종목별 다음 5·21일 수익 순위.
     사전 방향(학계: Pan·Poteshman 2006, Diether·Lee·Werner 2009, 공매도 잔고 증가): 셋 모두 '높으면 덜 오름'(IC < 0).
     통과 = front IC < 0, back IC < 0, back Newey-West t ≤ −2. 둘 조합 모서리 12개 × h 2는 탐색(판정 없음).
  s6 leadlag 사후 후보(HF매도·개인매수 → 다음 1주 QQQ 낮음)를 손대지 않은 2024에서 확인.
  s7 개인 환호 × 큰손 방어 합성(BRIEF T1·T2 정의)을 250개 순위 삼분위 3 × 3 상태로 나눠 앞으로의 R21·M21(T1·T2 모서리 밖 7칸).
  s8 s3에서 남은 lev_soxx_bull5+(SOXL 거래 몫 높음)가 지난 SOXX 상승의 대리인지, 문턱·하루 늦춤·절대 평균·지금 값.
  s9 고르기 규칙 민감도: front 최소 신호일 5 → 20(짧은 한 묶음 배제), QQQ·SOXX, 같은 귀무.
  s10 문턱 없이 24신호 × 6결과 순위상관의 front·back 부호 일치 수 vs 같은 원형 이동 귀무.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402
import grid as G  # noqa: E402

OUT = H.RES / "completeness_critic.json"
TOP_CONF = ["eq_pcr5+&shvr_ai5-", "skew+&shvr_ai5-", "skew+&nq_am_chg4-"]


def f(x, nd=3):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return None if np.isnan(x) else round(x, nd)


def p_shift_2026(sig26, y26, back, side, min_shift=21):
    """verify_grid 대안 p: 2026 전체 신호열을 원형 이동하고 back 날 차이만 잰다(신호일·나머지 3일 미만 이동은 뺌)."""
    defined = ~np.isnan(sig26)
    on = np.nan_to_num(sig26) > 0
    yok = back & ~np.isnan(y26)

    def st(o_, d_):
        s_, r_ = yok & o_ & d_, yok & ~o_ & d_
        if s_.sum() < 3 or r_.sum() < 3:
            return np.nan
        return y26[s_].mean() - y26[r_].mean()

    obs = st(on, defined)
    N = len(on)
    null = np.array([st(np.roll(on, k), np.roll(defined, k)) for k in range(min_shift, N - min_shift + 1)])
    null = null[~np.isnan(null)]
    if np.isnan(obs) or not len(null):
        return np.nan, 0
    hit = (null <= obs) if side == "less" else (null >= obs)
    return float((hit.sum() + 1) / (len(null) + 1)), int(len(null))


def cell_col(P: pd.DataFrame, cell: str, th1=0.8, th2=0.7) -> np.ndarray:
    parts = cell.split("&")
    th = th1 if len(parts) == 1 else th2
    on, na = np.ones(len(P), bool), np.zeros(len(P), bool)
    for p in parts:
        x = P[p[:-1]].to_numpy()
        na |= np.isnan(x)
        on &= (x >= th) if p[-1] == "+" else (x <= 1 - th + 1e-12)
    return np.where(na, np.nan, on.astype(float))


def period_stats(sig, y, idx, side):
    out = {}
    for per in ("front", "back", "y2025"):
        m = H.period_mask(idx, per)
        s, v = np.where(m, sig, np.nan), np.where(m, y, np.nan)
        d, t, n = H.welch(s, v)
        out[per] = {"d": f(d, 2), "t": f(t, 2), "n": n, "p": f(H.shift_p(s, v, side), 3) if n >= 3 else None}
    y26 = np.asarray(idx >= "2026-01-02")
    p26, ns = p_shift_2026(sig[y26], y[y26], H.period_mask(idx[y26], "back"), side)
    out["back"]["p_2026shift"], out["back"]["n_shifts_2026"] = f(p26, 3), ns
    return out


# ---------------- s1 변동성 기준선 ----------------
def vol_signals(dd: pd.Index) -> pd.DataFrame:
    px = H.prices()
    lr = np.log(px["QQQ_c"]).diff()
    rv = lr.rolling(20).std().dropna() * np.sqrt(252)
    rv_pct = H._place(H.rolling_pct(rv, 250), pd.Series(rv.index, index=rv.index), dd, H.DAILY_AGE)
    y = pd.read_parquet(H.OS / "cache/yahoo.parquet").dropna(subset=["QQQ"])
    vix_pct = H._place(H.rolling_pct(y["^VIX"], 250), pd.Series(y.index, index=y.index), dd, H.DAILY_AGE)
    return pd.DataFrame({"rv20": rv_pct, "vix": vix_pct}, index=dd)


def s1() -> dict:
    P = H.panel()
    idx = P.index
    Y, Ys = H.outcomes("QQQ").reindex(idx), H.outcomes("SOXX").reindex(idx)
    V = vol_signals(idx)
    res = {"baseline": {}, "confirmed_cells_vol_state": {}, "partial_ols": {}}
    for nm in ("rv20", "vix"):
        for th in (0.8, 0.7):
            x = V[nm].to_numpy()
            sig = np.where(np.isnan(x), np.nan, (x >= th).astype(float))
            rec = {}
            for lab, YY, col in (("QQQ_M21", Y, "M21"), ("QQQ_M10", Y, "M10"), ("QQQ_R21", Y, "R21"), ("SOXX_M21", Ys, "M21")):
                rec[lab] = period_stats(sig, YY[col].to_numpy(), idx, "less")
            res["baseline"][f"{nm}>={th}"] = rec
    # 확인 칸 신호일의 변동성 상태와 변동성 통제 뒤 차이
    for cell in TOP_CONF:
        sig = cell_col(P, cell)
        st = {}
        for per in ("front", "back", "y2025"):
            m = H.period_mask(idx, per) & (np.nan_to_num(sig) > 0)
            rest = H.period_mask(idx, per) & ~np.isnan(sig) & (np.nan_to_num(sig) == 0)
            st[per] = {"n_sig": int(m.sum()), "rv20_sig_mean": f(V["rv20"].to_numpy()[m].mean() if m.any() else np.nan),
                       "rv20_rest_mean": f(np.nanmean(V["rv20"].to_numpy()[rest])),
                       "share_sig_days_rv20>=0.8": f((V["rv20"].to_numpy()[m] >= 0.8).mean() if m.any() else np.nan),
                       "vix_sig_mean": f(V["vix"].to_numpy()[m].mean() if m.any() else np.nan)}
        res["confirmed_cells_vol_state"][cell] = st
        ols = {}
        y = Y["M21"].to_numpy()
        for per in ("front", "back", "y2025"):
            m = H.period_mask(idx, per) & ~np.isnan(sig) & ~np.isnan(y) & ~np.isnan(V["rv20"].to_numpy())
            if m.sum() < 10 or np.nan_to_num(sig[m]).sum() < 3:
                ols[per] = None
                continue
            X1 = np.column_stack([np.ones(m.sum()), sig[m]])
            X2 = np.column_stack([X1, V["rv20"].to_numpy()[m]])
            b1 = np.linalg.lstsq(X1, y[m], rcond=None)[0]
            b2 = np.linalg.lstsq(X2, y[m], rcond=None)[0]
            ols[per] = {"sig_coef_raw": f(b1[1], 2), "sig_coef_ctrl_rv20": f(b2[1], 2), "rv20_coef": f(b2[2], 2), "n": int(m.sum())}
        res["partial_ols"][cell] = ols
    # 기본 비율 맥락: 2026 front·back에서 rv20 ≥ 0.8인 날 비율
    for per in ("front", "back", "y2025"):
        m = H.period_mask(idx, per)
        res.setdefault("rv20_share_days_ge0.8", {})[per] = f((V["rv20"].to_numpy()[m] >= 0.8).mean())
    return res


# ---------------- s2·s3 고르기·확인(두 p 기준) ----------------
def select_confirm2(S26, Y26, fr, bk, topk=20, min_front=5):
    recs = []
    for h in H.HORIZONS:
        for y in ("R", "M"):
            yy = Y26[f"{y}{h}"].to_numpy()
            d, t, n = H.grid_diffs(S26, yy, fr)
            recs.append(pd.DataFrame({"k": np.arange(S26.shape[1]), "h": h, "y": y, "t": t, "n": n}))
    g = pd.concat(recs, ignore_index=True)
    g = g[(g["n"] >= min_front) & g["t"].notna()]
    out = {}
    for side in ("top", "bottom"):
        sub = g if side == "top" else g[g["y"] == "R"]
        pick = sub.nsmallest(topk, "t") if side == "top" else sub.nlargest(topk, "t")
        rows, sets_b, sets_26, cb, c26 = [], set(), set(), 0, 0
        for r in pick.itertuples():
            k, h, y = int(r.k), int(r.h), r.y
            yy = Y26[f"{y}{h}"].to_numpy()
            sig = S26[:, k]
            bs, by = np.where(bk, sig, np.nan), np.where(bk, yy, np.nan)
            d, _, nb = H.welch(bs, by)
            want = bool((d < 0) if side == "top" else (d > 0))
            sd = "less" if side == "top" else "greater"
            pb = H.shift_p(bs, by, sd) if (nb >= 3 and want) else np.nan
            p26, ns = p_shift_2026(sig, yy, bk, sd) if (nb >= 3 and want) else (np.nan, 0)
            okb, ok26 = bool(want and pb < 0.05), bool(want and p26 < 0.05)
            key = tuple(np.flatnonzero(fr & (np.nan_to_num(sig) > 0)))
            cb += okb
            c26 += ok26
            if okb:
                sets_b.add(key)
            if ok26:
                sets_26.add(key)
            rows.append({"k": k, "h": h, "y": y, "front_t": float(r.t), "front_n": int(r.n), "back_d": d, "back_n": nb,
                         "p_backonly": pb, "p_2026shift": p26, "n_shifts_2026": ns, "conf_backonly": okb, "conf_2026shift": ok26})
        out[side] = {"conf_backonly": cb, "conf_2026shift": c26, "sets_backonly": len(sets_b), "sets_2026shift": len(sets_26),
                     "n_front_sets": len({tuple(np.flatnonzero(fr & (np.nan_to_num(S26[:, rr["k"]]) > 0))) for rr in rows}), "rows": rows}
    return out


def run_null(S26, Y26, fr, bk, shifts, min_front=5):
    keys = ("conf_backonly", "conf_2026shift", "sets_backonly", "sets_2026shift")
    obs = select_confirm2(S26, Y26, fr, bk, min_front=min_front)
    null = {s: {k: [] for k in keys} for s in ("top", "bottom")}
    for i, k in enumerate(shifts):
        r = select_confirm2(np.roll(S26, int(k), axis=0), Y26, fr, bk, min_front=min_front)
        for s in ("top", "bottom"):
            for kk in keys:
                null[s][kk].append(r[s][kk])
    summ = {}
    for s in ("top", "bottom"):
        summ[s] = {}
        for kk in keys:
            nn = np.array(null[s][kk])
            summ[s][kk] = {"obs": obs[s][kk], "null_mean": f(nn.mean(), 2), "null_p95": f(np.percentile(nn, 95), 1),
                           "p": f(((nn >= obs[s][kk]).sum() + 1) / (len(nn) + 1), 3), "null_share_ge1": f((nn >= 1).mean(), 3)}
    return obs, summ


def s2() -> dict:
    import triple_combos as T
    P = H.panel()
    idx = P.index
    Y = H.outcomes("QQQ").reindex(idx)
    S, names, _ = T.cells3(P)
    y26 = np.asarray(idx >= "2026-01-02")
    S26, Y26, i26 = S[y26], Y[y26].reset_index(drop=True), idx[y26]
    fr, bk = H.period_mask(i26, "front"), H.period_mask(i26, "back")
    tj = json.loads((H.RES / "triple_combos.json").read_text(encoding="utf-8"))
    shifts = tj["null_shifts"]
    t0 = time.time()
    obs, summ = run_null(S26, Y26, fr, bk, shifts)
    picks = [{"cell": names[r["k"]], "h": r["h"], "y": r["y"], "front_t": f(r["front_t"], 1), "front_n": r["front_n"], "back_d": f(r["back_d"], 2),
              "back_n": r["back_n"], "p_backonly": f(r["p_backonly"]), "p_2026shift": f(r["p_2026shift"]), "n_shifts_2026": r["n_shifts_2026"]}
             for r in obs["top"]["rows"] if r["conf_backonly"] or r["conf_2026shift"]]
    return {"null_shifts_from": "triple_combos.json null_shifts (seed 13, 200)", "summary": summ, "confirmed_rows_top": picks,
            "runtime_s": round(time.time() - t0, 1)}


def s3() -> dict:
    P = H.panel()
    idx = P.index
    S, names = G.cells(P)
    y26 = np.asarray(idx >= "2026-01-02")
    i26 = idx[y26]
    fr, bk = H.period_mask(i26, "front"), H.period_mask(i26, "back")
    rng = np.random.default_rng(G.SEED)
    shifts = [int(rng.integers(H.MIN_SHIFT, len(i26) - H.MIN_SHIFT)) for _ in range(G.N_NULL)]
    out = {}
    for asset in ("QQQ", "SOXX"):
        Y = H.outcomes(asset).reindex(idx)
        Y26 = Y[y26].reset_index(drop=True)
        obs, summ = run_null(S[y26], Y26, fr, bk, shifts)
        rows = []
        for side in ("top", "bottom"):
            for r in obs[side]["rows"]:
                if r["conf_backonly"] or r["conf_2026shift"] or len(rows) < 3:
                    rows.append({"side": side, "cell": names[r["k"]], "h": r["h"], "y": r["y"], "front_t": f(r["front_t"], 1), "front_n": r["front_n"],
                                 "back_d": f(r["back_d"], 2), "back_n": r["back_n"], "p_backonly": f(r["p_backonly"]), "p_2026shift": f(r["p_2026shift"])})
        out[asset] = {"summary": summ, "rows": rows,
                      "front_sets": {s: obs[s]["n_front_sets"] for s in ("top", "bottom")}}
    out["note"] = "QQQ는 grid.json 재현 확인용(같은 seed 11 이동). 칸은 grid.cells 그대로(− 단독 문턱 부동소수 버그 포함)."
    return out


# ---------------- s4 CFTC 측정일 기준 나이 ----------------
def s4() -> dict:
    P = H.panel()
    idx = P.index
    dd = idx
    cf = pd.read_parquet(H.CACHE / "cftc_tff.parquet")
    asof_pos = pd.Series(np.searchsorted(dd.to_numpy(), cf["asof"].to_numpy(), side="left").astype(float), index=cf.index)
    # _place와 같은 규칙: known 다음 거래일부터 15거래일 이어 씀. 값 = asof의 거래일 위치
    placed = H._place(asof_pos, pd.Series(cf.index, index=cf.index), dd, H.WEEKLY_AGE).to_numpy()
    age = np.arange(len(dd)) - placed  # 진입일 − 측정일(거래일)
    res = {"age_td_by_period": {}}
    for per in ("front", "back", "y2025"):
        m = H.period_mask(idx, per) & ~np.isnan(age)
        a = age[m]
        res["age_td_by_period"][per] = {"n": int(m.sum()), "max": f(a.max(), 0), "p95": f(np.percentile(a, 95), 0),
                                        "n_gt10": int((a > 10).sum()), "first_gt10": str(idx[m][a > 10][0]) if (a > 10).any() else None,
                                        "last_gt10": str(idx[m][a > 10][-1]) if (a > 10).any() else None}
    # 2025 보조 수치 재계산: 측정일 기준 10거래일 넘게 묵은 CFTC 값을 비우면
    stale = age > 10
    cftc_cols = [c for c in P.columns if c.startswith(("nq_", "es_", "vx_"))]
    P2 = P.copy()
    P2.loc[stale, cftc_cols] = np.nan
    Y = H.outcomes("QQQ").reindex(idx)
    for cell, col, side in (("es_lev_net_oi+", "R21", "greater"), ("skew+&nq_am_chg4-", "M21", "less"), ("nq_lev_chg4-", "M10", "less")):
        res.setdefault("recheck", {})[cell] = {"original": period_stats(cell_col(P, cell), Y[col].to_numpy(), idx, side),
                                               "stale_masked": period_stats(cell_col(P2, cell), Y[col].to_numpy(), idx, side), "outcome": col}
    return res


# ---------------- s5 종목 횡단면 ----------------
ETF = {"QQQ", "SPY", "SOXX", "SMH", "TQQQ", "SQQQ", "SOXL", "SOXS"}


def place_matrix(M: pd.DataFrame, dd: pd.Index, age: int, lag_days: int = 1) -> np.ndarray:
    """M: 측정(또는 known)일 문자열 색인 × 종목. 그날 다음 lag_days번째 거래일부터 age거래일까지 쓴다."""
    pos = np.searchsorted(dd.to_numpy(), M.index.to_numpy(), side="right") + (lag_days - 1)
    out = np.full((len(dd), M.shape[1]), np.nan)
    ok = pos < len(dd)
    out[pos[ok]] = M.to_numpy()[ok]
    return pd.DataFrame(out).ffill(limit=age).to_numpy()


def cs_rank(X: np.ndarray) -> np.ndarray:
    R = np.full_like(X, np.nan)
    for i in range(X.shape[0]):
        m = ~np.isnan(X[i])
        if m.sum() >= 20:
            R[i, m] = (rankdata(X[i, m]) - 0.5) / m.sum()
    return R


def nw_t(x: np.ndarray, lag: int) -> float:
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 15:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def s5() -> dict:
    dd = H.days()
    sv = pd.read_parquet(H.CACHE / "finra_shvol_by_symbol.parquet")
    univ_f = sorted(set(sv["symbol"]) - ETF)
    px = pd.read_parquet(H.ROOT / "data/derived/analysis/rsi_breakout/cache/1d_long.parquet")
    px["d"] = pd.to_datetime(px["ts"]).dt.strftime("%Y-%m-%d")
    univ = sorted(set(univ_f) & set(px["sym"]))
    px = px[px["sym"].isin(univ)]
    O = px.pivot_table(index="d", columns="sym", values="Open").reindex(index=dd, columns=univ)
    C = px.pivot_table(index="d", columns="sym", values="Close").reindex(index=dd, columns=univ)
    o, c = O.to_numpy(), C.to_numpy()
    n = len(dd)
    R = {}
    for h in (5, 21):
        r = np.full_like(o, np.nan)
        r[: n - h + 1] = (c[h - 1:] / o[: n - h + 1] - 1) * 100
        R[h] = r
    # 신호 원값
    sv = sv[sv["symbol"].isin(univ)].copy()
    sv["r"] = sv["short"] / sv["total"].where(sv["total"] > 0)
    shv = sv.pivot_table(index="asof", columns="symbol", values="r").reindex(columns=univ).sort_index().rolling(5, min_periods=4).mean()
    occ = []
    for fp in sorted(glob.glob(str(H.OS / "cache/occ/*.parquet"))):
        x = pd.read_parquet(fp, columns=["underlying", "C_C", "C_P"])
        x = x[x["underlying"].isin(univ)]
        x["d"] = Path(fp).stem
        occ.append(x)
    occ = pd.concat(occ)
    cc = occ.pivot_table(index="d", columns="underlying", values="C_C").reindex(columns=univ).sort_index().fillna(0)
    cp = occ.pivot_table(index="d", columns="underlying", values="C_P").reindex(columns=univ).sort_index().fillna(0)
    c5, p5 = cc.rolling(5).sum(), cp.rolling(5).sum()
    cpr = np.log((p5 + 1) / (c5 + 1)).where((c5 + p5) >= 500)  # 5일 고객 거래 500계약 미만은 비움
    si = pd.read_parquet(H.CACHE / "finra_si_by_symbol.parquet")
    si = si[si["symbol"].isin(univ)]
    sic = si.pivot_table(index="known", columns="symbol", values="chg").reindex(columns=univ).sort_index()
    meta = {"universe_n": len(univ), "occ_days": int(len(cc)), "occ_first": str(cc.index[0]), "shvol_first": str(shv.index[0]),
            "si_first_known": str(sic.index[0]), "price_last": str(px["d"].max())}

    def build(lag_daily: int):
        return {"cpr5": cs_rank(place_matrix(cpr, dd, H.DAILY_AGE, lag_daily)),
                "shvr5": cs_rank(place_matrix(shv, dd, H.DAILY_AGE, lag_daily)),
                "si_chg": cs_rank(place_matrix(sic, dd, 25, 1))}

    periods = {p: H.period_mask(dd, p) for p in ("front", "back", "y2025")}

    def ic_series(X, Rh):
        out = np.full(n, np.nan)
        for i in range(n):
            m = ~np.isnan(X[i]) & ~np.isnan(Rh[i])
            if m.sum() >= 20:
                a, b = rankdata(X[i, m]), rankdata(Rh[i, m])
                out[i] = np.corrcoef(a, b)[0, 1]
        return out

    def corner_series(X1, X2, s1_, s2_, Rh):
        out = np.full(n, np.nan)
        for i in range(n):
            m = ~np.isnan(X1[i]) & ~np.isnan(X2[i]) & ~np.isnan(Rh[i])
            if m.sum() < 20:
                continue
            a = (X1[i] >= 2 / 3) if s1_ == "+" else (X1[i] <= 1 / 3)
            b = (X2[i] >= 2 / 3) if s2_ == "+" else (X2[i] <= 1 / 3)
            cn = m & a & b
            if cn.sum() >= 3 and (m & ~cn).sum() >= 3:
                out[i] = Rh[i, cn].mean() - Rh[i, m & ~cn].mean()
        return out

    def summarize(series, h):
        rec = {}
        for p, mk in periods.items():
            x = np.where(mk, series, np.nan)
            rec[p] = {"mean": f(np.nanmean(x), 4), "nw_t": f(nw_t(x, h), 2), "days": int((~np.isnan(x)).sum())}
        return rec

    res = {"meta": meta, "singles": {}, "singles_lag2": {}, "corners": {}}
    S1 = build(1)
    for nm, X in S1.items():
        for h in (5, 21):
            rec = summarize(ic_series(X, R[h]), h)
            rec["pass"] = bool((rec["front"]["mean"] or 0) < 0 and (rec["back"]["mean"] or 0) < 0 and (rec["back"]["nw_t"] or 0) <= -2)
            res["singles"][f"{nm}_h{h}"] = rec
    S2 = build(2)
    for nm in ("cpr5", "shvr5"):
        for h in (5, 21):
            res["singles_lag2"][f"{nm}_h{h}"] = summarize(ic_series(S2[nm], R[h]), h)
    pairs = [("cpr5", "shvr5"), ("cpr5", "si_chg"), ("shvr5", "si_chg")]
    hit = 0
    for a, b in pairs:
        for sa in "+-":
            for sb in "+-":
                for h in (5, 21):
                    rec = summarize(corner_series(S1[a], S1[b], sa, sb, R[h]), h)
                    fm, bm, bt = rec["front"]["mean"], rec["back"]["mean"], rec["back"]["nw_t"]
                    rec["same_sign_back_t2"] = bool(fm is not None and bm is not None and bt is not None and np.sign(fm) == np.sign(bm) and abs(bt) >= 2)
                    hit += rec["same_sign_back_t2"]
                    res["corners"][f"{a}{sa}&{b}{sb}_h{h}"] = rec
    res["corners_hits"] = hit
    res["corners_n"] = len(res["corners"])
    return res


# ---------------- s6 leadlag 사후 후보의 2024 ----------------
def s6() -> dict:
    import leadlag as L
    E = L.aligned_variant()
    idx = E.index
    ok = E["nq_lev_1w"].notna() & E["flow1w"].notna()
    out = {}
    for nm, cond, hf_sign in (("HF매도_개인매수", (E["nq_lev_1w"] < 0) & (E["flow1w"] > 0), -1),
                              ("HF매수_개인매도", (E["nq_lev_1w"] > 0) & (E["flow1w"] < 0), +1)):
        for k in (1, 2, 4):
            for lab, lo, hi in (("y2024", "2024-01-01", "2024-12-31"), ("y2025", "2025-01-01", "2025-12-31"),
                                ("front", "2026-01-02", "2026-05-29"), ("back", "2026-06-01", "2026-12-31")):
                m = ok.to_numpy() & np.asarray((idx >= lo) & (idx <= hi)) & E[f"fwd{k}"].notna().to_numpy()
                s, y = cond.to_numpy()[m].astype(float), E[f"fwd{k}"].to_numpy()[m]
                rec = {"n_weeks": int(m.sum()), "n_sig": int(s.sum())}
                if 2 <= s.sum() < len(s):
                    pg, pl, ns = L.circ_p(s, y)
                    rec.update({"diff": f(y[s > 0].mean() - y.mean(), 2), "p_hf_right": f(pl if hf_sign < 0 else pg, 3), "p_floor": f(1 / (ns + 1), 3)})
                out[f"{nm}_k{k}_{lab}"] = rec
    return out


# ---------------- s7 합성 3 × 3 상태 ----------------
def s7() -> dict:
    P = H.panel()
    idx = P.index
    Y, Ys = H.outcomes("QQQ").reindex(idx), H.outcomes("SOXX").reindex(idx)
    re, sd = G.composite(P, G.RETAIL_EUPH), G.composite(P, G.SMART_DEF)
    rr = H.rolling_pct(re, 250).reindex(idx)
    sr = H.rolling_pct(sd, 250).reindex(idx)
    ter = lambda x: np.where(np.isnan(x), np.nan, np.where(x <= 1 / 3, 0, np.where(x >= 2 / 3, 2, 1)))
    a, b = ter(rr.to_numpy()), ter(sr.to_numpy())
    out, agree = {}, {"R21": 0, "M21": 0}
    for i in range(3):
        for j in range(3):
            sig = np.where(np.isnan(a) | np.isnan(b), np.nan, ((a == i) & (b == j)).astype(float))
            rec = {}
            for col in ("R21", "M21"):
                rec[col] = period_stats(sig, Y[col].to_numpy(), idx, "greater" if col == "R21" else "less")
                fd, bd = rec[col]["front"]["d"], rec[col]["back"]["d"]
                rec[col]["front_back_same_sign"] = bool(fd is not None and bd is not None and np.sign(fd) == np.sign(bd))
                agree[col] += rec[col]["front_back_same_sign"]
            rec["SOXX_R21"] = period_stats(sig, Ys["R21"].to_numpy(), idx, "greater")
            out[f"retail{i}_smartdef{j}"] = rec
    return {"states": out, "front_back_same_sign_of_9": agree,
            "note": "retail = 개인 환호 합성(높음 = 환호), smartdef = 큰손 방어 합성(높음 = 방어). 0/1/2 = 지난 250개 합성값 안 순위 삼분위(자기 포함, 최소 125)."}


# ---------------- s8 SOXX 격자에서 남은 lev_soxx_bull5+ 묶음: 지난 상승의 대리인가 ----------------
def episodes(days: list[str], dd: pd.Index, gap: int = 5) -> list[list[str]]:
    pos = np.searchsorted(dd.to_numpy(), np.array(days))
    out, cur = [], []
    for p, d in zip(pos, days):
        if cur and p - cur[-1][0] > gap:
            out.append(cur)
            cur = []
        cur.append((p, d))
    if cur:
        out.append(cur)
    return [[e[0][1], e[-1][1], len(e)] for e in out]


def s8() -> dict:
    P = H.panel()
    idx = P.index
    Ys, Y = H.outcomes("SOXX").reindex(idx), H.outcomes("QQQ").reindex(idx)
    px = H.prices()
    past = {}
    for k in (5, 20):
        r = (px["SOXX_c"] / px["SOXX_c"].shift(k) - 1).dropna()
        past[f"soxx_past{k}"] = H._place(H.rolling_pct(r, 250), pd.Series(r.index, index=r.index), idx, H.DAILY_AGE)
    X = pd.DataFrame(past, index=idx)
    X["lev_soxx_bull5"] = P["lev_soxx_bull5"]
    res = {"corr_2025_26": {}, "cells": {}, "ols": {}, "episodes": {}}
    m25 = np.asarray(idx >= "2025-01-02")
    for k in ("soxx_past5", "soxx_past20"):
        res["corr_2025_26"][k] = f(X.loc[m25, ["lev_soxx_bull5", k]].corr(method="spearman").iloc[0, 1])
    for nm in ("lev_soxx_bull5", "soxx_past5", "soxx_past20"):
        x = X[nm].to_numpy()
        sig = np.where(np.isnan(x), np.nan, (x >= 0.8).astype(float))
        rec = {}
        for lab, YY, col, side in (("SOXX_R21", Ys, "R21", "less"), ("SOXX_R10", Ys, "R10", "less"), ("SOXX_R5", Ys, "R5", "less"),
                                   ("SOXX_M21", Ys, "M21", "less"), ("QQQ_R21", Y, "R21", "less")):
            rec[lab] = period_stats(sig, YY[col].to_numpy(), idx, side)
        res["cells"][f"{nm}>=0.8"] = rec
        on = [str(d) for d, v in zip(idx, sig) if v == 1 and d >= "2026-01-02"]
        res["episodes"][f"{nm}>=0.8_2026"] = episodes(on, idx)
    sig = np.where(np.isnan(X["lev_soxx_bull5"].to_numpy()), np.nan, (X["lev_soxx_bull5"].to_numpy() >= 0.8).astype(float))
    y = Ys["R21"].to_numpy()
    for per in ("front", "back", "y2025"):
        m = H.period_mask(idx, per) & ~np.isnan(sig) & ~np.isnan(y) & X[["soxx_past5", "soxx_past20"]].notna().all(axis=1).to_numpy()
        Z1 = np.column_stack([np.ones(m.sum()), sig[m]])
        Z2 = np.column_stack([Z1, X.loc[m, ["soxx_past5", "soxx_past20"]].to_numpy()])
        b1, b2 = np.linalg.lstsq(Z1, y[m], rcond=None)[0], np.linalg.lstsq(Z2, y[m], rcond=None)[0]
        res["ols"][per] = {"sig_coef_raw": f(b1[1], 2), "sig_coef_ctrl_past": f(b2[1], 2), "past5_coef": f(b2[2], 2), "past20_coef": f(b2[3], 2), "n": int(m.sum())}
    # 민감도: 문턱 0.75·0.85, 하루 늦춘 진입, TQQQ 쪽 같은 신호, 신호일·나머지 평균(절대값)
    x = X["lev_soxx_bull5"].to_numpy()
    for th in (0.75, 0.85):
        sg = np.where(np.isnan(x), np.nan, (x >= th).astype(float))
        res.setdefault("threshold", {})[str(th)] = {"SOXX_R21": period_stats(sg, Ys["R21"].to_numpy(), idx, "less"),
                                                   "QQQ_R21": period_stats(sg, Y["R21"].to_numpy(), idx, "less")}
    xl = np.concatenate([[np.nan], x[:-1]])  # 하루 늦춤(d−1의 패널 값을 d에)
    sg = np.where(np.isnan(xl), np.nan, (xl >= 0.8).astype(float))
    res["lag1"] = {"SOXX_R21": period_stats(sg, Ys["R21"].to_numpy(), idx, "less"), "QQQ_R21": period_stats(sg, Y["R21"].to_numpy(), idx, "less")}
    xq = P["lev_qqq_bull5"].to_numpy()
    sg = np.where(np.isnan(xq), np.nan, (xq >= 0.8).astype(float))
    res["lev_qqq_bull5>=0.8"] = {"QQQ_R21": period_stats(sg, Y["R21"].to_numpy(), idx, "less"), "SOXX_R21": period_stats(sg, Ys["R21"].to_numpy(), idx, "less")}
    means = {}
    for per in ("front", "back", "y2025"):
        m = H.period_mask(idx, per) & ~np.isnan(sig) & ~np.isnan(y)
        on, off = m & (sig == 1), m & (sig == 0)
        means[per] = {"SOXX_R21_sig": f(y[on].mean(), 2), "SOXX_R21_rest": f(y[off].mean(), 2), "n_sig": int(on.sum()), "n_rest": int(off.sum()),
                      "share_sig_R21_neg": f((y[on] < 0).mean(), 2), "share_rest_R21_neg": f((y[off] < 0).mean(), 2)}
    res["abs_means"] = means
    res["now"] = {str(d): f(v, 2) for d, v in P["lev_soxx_bull5"].tail(5).items()}
    return res


# ---------------- s9 고르기 규칙 민감도: front 신호일 최소 20일(짧은 한 묶음 배제) ----------------
def s9() -> dict:
    """격자의 Welch t 고르기는 겹친 21일 창을 가진 5~9일 한 묶음을 맨 위로 올린다(상위 20칸 전부 한 묶음).
    같은 격자·같은 귀무(seed 11)에서 front 신호일 최소 5 → 20만 바꿔 본다. 사후 변형이라 판정에 쓰지 않는다."""
    P = H.panel()
    idx = P.index
    S, names = G.cells(P)
    y26 = np.asarray(idx >= "2026-01-02")
    i26 = idx[y26]
    fr, bk = H.period_mask(i26, "front"), H.period_mask(i26, "back")
    rng = np.random.default_rng(G.SEED)
    shifts = [int(rng.integers(H.MIN_SHIFT, len(i26) - H.MIN_SHIFT)) for _ in range(G.N_NULL)]
    out = {}
    for asset in ("QQQ", "SOXX"):
        Y26 = H.outcomes(asset).reindex(idx)[y26].reset_index(drop=True)
        obs, summ = run_null(S[y26], Y26, fr, bk, shifts, min_front=20)
        rows = [{"side": side, "cell": names[r["k"]], "h": r["h"], "y": r["y"], "front_t": f(r["front_t"], 1), "front_n": r["front_n"],
                 "back_d": f(r["back_d"], 2), "back_n": r["back_n"], "p_backonly": f(r["p_backonly"]), "p_2026shift": f(r["p_2026shift"])}
                for side in ("top", "bottom") for r in obs[side]["rows"] if r["conf_backonly"] or r["conf_2026shift"]]
        out[asset] = {"summary": summ, "confirmed_rows": rows, "front_sets": {s_: obs[s_]["n_front_sets"] for s_ in ("top", "bottom")}}
    return out


# ---------------- s10 문턱 없이: 신호 백분위와 앞날 결과의 순위상관(24 × 6 × QQQ·SOXX) ----------------
def s10() -> dict:
    """격자는 문턱(0.8·0.7·0.65)만 봤다. 연속값으로 front·back 순위상관 부호가 같은 칸 수를 세고,
    같은 원형 이동(seed 11, 200번, 2026 안에서 모든 신호를 같이 이동) 귀무와 비교한다. 겹친 창이라 상관 크기 자체의 p는 쓰지 않는다."""
    P = H.panel()
    idx = P.index
    y26 = np.asarray(idx >= "2026-01-02")
    i26 = idx[y26]
    fr, bk = H.period_mask(i26, "front"), H.period_mask(i26, "back")
    X26 = P.to_numpy()[y26]
    rng = np.random.default_rng(G.SEED)
    shifts = [int(rng.integers(H.MIN_SHIFT, len(i26) - H.MIN_SHIFT)) for _ in range(G.N_NULL)]

    def rho(x, y, m):
        ok = m & ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < 20 or np.nanstd(x[ok]) == 0:
            return np.nan
        return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])

    out = {}
    for asset in ("QQQ", "SOXX"):
        Y26 = H.outcomes(asset).reindex(idx)[y26]
        cols = [f"{a}{h}" for h in H.HORIZONS for a in ("R", "M")]

        def count(X):
            same, strong, rows = 0, 0, []
            for j, sname in enumerate(P.columns):
                for c in cols:
                    rf, rb = rho(X[:, j], Y26[c].to_numpy(), fr), rho(X[:, j], Y26[c].to_numpy(), bk)
                    if np.isnan(rf) or np.isnan(rb):
                        continue
                    sm = np.sign(rf) == np.sign(rb)
                    same += sm
                    strong += bool(sm and abs(rf) >= 0.3 and abs(rb) >= 0.3)
                    rows.append((sname, c, rf, rb))
            return same, strong, rows

        def best(rows_):
            v = [min(abs(a_), abs(b_)) for _, _, a_, b_ in rows_ if np.sign(a_) == np.sign(b_)]
            return max(v) if v else 0.0

        s_obs, st_obs, rows = count(X26)
        b_obs = best(rows)
        nul = [count(np.roll(X26, k, axis=0)) for k in shifts]
        ns, nst = np.array([a for a, _, _ in nul]), np.array([b for _, b, _ in nul])
        nb = np.array([best(r_) for _, _, r_ in nul])
        rows = sorted(rows, key=lambda r: -min(abs(r[2]), abs(r[3])) if np.sign(r[2]) == np.sign(r[3]) else 0)
        out[asset] = {"n_pairs": len(rows), "same_sign": int(s_obs), "null_same_mean": f(ns.mean(), 1), "p_same": f(((ns >= s_obs).sum() + 1) / (len(ns) + 1)),
                      "same_sign_both_abs_ge0.3": int(st_obs), "null_strong_mean": f(nst.mean(), 2), "p_strong": f(((nst >= st_obs).sum() + 1) / (len(nst) + 1)),
                      "best_min_abs_rho": f(b_obs, 2), "null_best_mean": f(nb.mean(), 2), "p_best": f(((nb >= b_obs).sum() + 1) / (len(nb) + 1)),
                      "top_same_sign": [{"signal": a, "y": b, "rho_front": f(c_, 2), "rho_back": f(d_, 2)} for a, b, c_, d_ in rows[:8]]}
    return out


SECTIONS = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "s8": s8, "s9": s9, "s10": s10}


def main() -> None:
    want = [a for a in sys.argv[1:] if a in SECTIONS] or list(SECTIONS)
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for k in want:
        t0 = time.time()
        res[k] = SECTIONS[k]()
        res[k + "_runtime_s"] = round(time.time() - t0, 1)
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(k, "done", res[k + "_runtime_s"], "s")


if __name__ == "__main__":
    main()
