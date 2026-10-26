"""통과 후보 칸의 관문·강건 검사(BRIEF v2 §8). 주 가설 8칸과 탐색 상위 10칸에만 쓴다.

- 가격 메아리: logit(신호) ~ 대상 지난 5·20·65세션 로그 수익 + min(지난 5세션, 0) + SOXL÷QQQ 65세션 상대 수익
  (+ S2·S4·C2·C3: SOXL 250세션 가격 백분위). 설명 변수는 e − 1 종가까지, 확장 창(최소 250행), 그때까지의 행만으로 적합.
  잔차판·위약(적합값)을 백분위로 바꾸고, 뒤 구간 켜짐 수가 원래와 같게 문턱을 맞춘다.
- 강건 ① 막차 창 빼기 ② 가장 큰 켜짐 덩어리 ± h 빼기 ③ 2026-07 진입 빼기 ④ ^SOX ⑤ 분모 = 지난 65세션 B 중앙값 ⑥ 고정 품목판
- 뒤 구간 소구간 몫, 상태 나눔(추격·물타기, 설명)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import harness as hz

BREAK_DAYS = ["2025-12-15", "2026-05-22", "2026-07-31"]
SUBS = [("2026-06-01", "2026-07-10"), ("2026-07-13", "2026-07-31"), ("2026-08-03", "2026-10-23")]
PRICE_PCT_SIGS = {"S2", "S4", "C2", "C3"}


def transform(x: pd.Series) -> pd.Series:
    v = x.astype(float)
    if v.min() > 0 and v.max() < 1:
        v = v.clip(1e-4, 1 - 1e-4)
        return np.log(v / (1 - v))
    if v.min() > 0:
        return np.log(v)
    return v


def regressors(px: pd.DataFrame, entries: pd.DatetimeIndex, target: str, with_pct: bool) -> pd.DataFrame:
    def cl(t):
        s = px[px["ticker"] == t].set_index("date")["Close"]
        return s[~s.index.duplicated()].reindex(entries).shift(1)  # e − 1 종가
    c, soxl, qqq = cl(target), cl("SOXL"), cl("QQQ")
    X = pd.DataFrame(index=entries)
    for h in (5, 20, 65):
        X[f"r{h}"] = np.log(c / c.shift(h))
    X["neg5"] = X["r5"].clip(upper=0)
    X["rel"] = np.log(soxl / soxl.shift(65)) - np.log(qqq / qqq.shift(65))
    if with_pct:
        X["soxl_pct"] = hz.rolling_pct(soxl, 250, 120)
    return X


def echo_split(raw_sig: pd.Series, X: pd.DataFrame, start="2024-01-01") -> tuple[pd.Series, pd.Series]:
    y = transform(raw_sig)
    D = pd.concat([y.rename("y"), X], axis=1)
    fit, res = pd.Series(np.nan, index=D.index), pd.Series(np.nan, index=D.index)
    for i, e in enumerate(D.index):
        if e < pd.Timestamp(start):
            continue
        past = D.iloc[:i].dropna()
        if len(past) < 250 or D.iloc[i].isna().any():
            continue
        A = np.column_stack([np.ones(len(past)), past.drop(columns="y").values])
        beta, *_ = np.linalg.lstsq(A, past["y"].values, rcond=None)
        f = float(np.r_[1.0, D.iloc[i].drop("y").values] @ beta)
        fit.iloc[i], res.iloc[i] = f, D["y"].iloc[i] - f
    return res, fit


def matched_on(pct: pd.Series, period: hz.Period, n_on: int, direction: str) -> pd.Series:
    """뒤 구간 켜짐 수가 n_on이 되게 문턱을 고른다."""
    v = pct.reindex(period.entries)
    vals = v.dropna().sort_values(ascending=(direction == "low"))
    if len(vals) == 0 or n_on == 0:
        return pd.Series(np.nan, index=pct.index)
    cut = vals.iloc[min(n_on, len(vals)) - 1]
    on = (pct <= cut) if direction == "low" else (pct >= cut)
    return on.astype(float).where(pct.notna())


def cell_effect(on: pd.Series, period: hz.Period, cell: hz.Cell, target=None) -> dict:
    y = period.y[(target or cell.target, cell.outcome)]
    eff = hz.shift_effects(on.reindex(period.entries).values[None, :], y, cell.outcome in hz.EVENT)[0]
    e = eff * hz.sign_of(cell)
    null = e[1:][~np.isnan(e[1:])]
    p = (1 + (null >= e[0]).sum()) / (1 + len(null)) if not np.isnan(e[0]) and len(null) else np.nan
    o = on.reindex(period.entries).values
    return {"e_h": float(e[0]), "p": float(p), "n_on": int(np.nansum((o == 1) & ~np.isnan(y)))}


def masked_period(period: hz.Period, drop: np.ndarray) -> hz.Period:
    ys = {k: np.where(drop, np.nan, v) for k, v in period.y.items()}
    return hz.Period(period.name, period.entries, ys)


def contains_break_trades(raw: pd.DataFrame, sessions: pd.DatetimeIndex, signal: str) -> pd.Series:
    """신호 창이 단절 전 10세션 ~ 후 5세션의 거래를 담은 진입."""
    bad = pd.Series(False, index=raw.index)
    for b in BREAK_DAYS:
        i = sessions.searchsorted(pd.Timestamp(b))
        lo, hi = sessions[i - 10], sessions[min(i + 5, len(sessions) - 1)]
        if signal.startswith("C"):
            cd = pd.to_datetime(raw["cdate"])
            span = 20 if signal in ("C1", "C4") else 0
            bad |= (cd >= lo) & (cd - pd.Timedelta(days=int(span * 1.45)) <= hi)
        else:
            ts, te = pd.to_datetime(raw["t_start"]), pd.to_datetime(raw["t_end"])
            bad |= (te >= lo) & (ts <= hi)
    return bad


def run_gates(cell: hz.Cell, base_on: pd.Series, raw: pd.DataFrame, pct: pd.DataFrame, px, front, back,
              sd_front, rebuild) -> dict:
    out = {}
    base = cell_effect(base_on, back, cell)
    out["base"] = base
    # 가격 메아리(신호 하나짜리 칸만; 조합·K1은 구성 신호별로 보고만)
    sigs = {"P1": ("S1", "C1"), "P2": ("S3", "S7"), "P4": ("L2", "C3")}.get(cell.signal, (cell.signal,))
    echo = {}
    if cell.signal != "K1":
        for s in sigs:
            X = regressors(px, raw.index, cell.target, s in PRICE_PCT_SIGS)
            res, fit = echo_split(raw[s], X)
            r_pct, f_pct = hz.rolling_pct(res), hz.rolling_pct(fit)
            d = "high" if cell.direction == "high" else "low"
            r_on = matched_on(r_pct, back, base["n_on"], d)
            f_on = matched_on(f_pct, back, base["n_on"], d)
            echo[s] = {"resid": cell_effect(r_on, back, cell), "placebo": cell_effect(f_on, back, cell)}
        ok = all(v["placebo"]["e_h"] < 0.5 * base["e_h"] and v["resid"]["e_h"] >= 0.5 * base["e_h"]
                 and v["resid"]["p"] < 0.10 for v in echo.values())
        if cell.signal == "C2":
            c2s = rebuild("C2star")
            if c2s is not None:
                on = matched_on(hz.rolling_pct(c2s), back, base["n_on"], cell.direction)
                echo["C2star"] = {"placebo": cell_effect(on, back, cell)}
                ok = ok and echo["C2star"]["placebo"]["e_h"] < 0.5 * base["e_h"]
    else:
        ok = None
    out["echo"], out["echo_ok"] = echo, ok
    # 강건 ① 막차 창(진입과 기준 분포 모두에서 뺌)
    sessions = raw.index
    sig_for_break = sigs[0]
    bad = contains_break_trades(raw, sessions, sig_for_break)
    raw1 = raw.copy()
    for s in sigs:
        raw1.loc[bad, s] = np.nan
    on1 = rebuild("on", raw1, cell)
    out["r1_breakwin"] = cell_effect(on1, masked_period(back, bad.reindex(back.entries).values), cell)
    # ② 가장 큰 켜짐 덩어리 ± h
    y = back.y[(cell.target, cell.outcome)]
    o = base_on.reindex(back.entries).values
    use = ~np.isnan(y) & ~np.isnan(o)
    sgn = hz.sign_of(cell)
    off_mean = np.nanmean(np.where(use & (o == 0), y, np.nan))
    idx = np.flatnonzero(use & (o == 1))
    h = hz.H[cell.outcome]
    if len(idx):
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > h) + 1)
        contrib = [sgn * np.nansum(y[g] - off_mean) for g in groups]
        g = groups[int(np.argmax(contrib))]
        drop = np.zeros(len(y), bool)
        drop[max(0, g[0] - h):g[-1] + h + 1] = True
        out["r2_dropcluster"] = cell_effect(base_on, masked_period(back, drop), cell)
        out["r2_cluster"] = [str(back.entries[g[0]].date()), str(back.entries[g[-1]].date())]
    # ③ 2026-07 진입 빼기
    jul = (back.entries >= "2026-07-01") & (back.entries <= "2026-07-31")
    out["r3_nojuly"] = cell_effect(base_on, masked_period(back, np.asarray(jul)), cell)
    # ④ ^SOX
    if cell.target == "SOXX":
        out["r4_sox"] = cell_effect(base_on, back, cell, target="^SOX")
    # ⑤ 분모 바꾸기, ⑥ 고정 품목판
    for tag in ("denom", "fixed"):
        on_alt = rebuild(tag, raw, cell)
        if on_alt is not None:
            out[f"r_{tag}"] = cell_effect(on_alt, back, cell)
    # 소구간 몫
    shares = []
    if len(idx):
        tot = np.nansum(sgn * (y[idx] - off_mean))
        for lo, hi in SUBS:
            msk = (back.entries[idx] >= lo) & (back.entries[idx] <= hi)
            shares.append({"sub": f"{lo}~{hi}", "n_on": int(msk.sum()),
                           "share": float(np.nansum(sgn * (y[idx][msk] - off_mean)) / tot) if tot else np.nan})
    out["subperiods"] = shares
    out["regime_dependent"] = bool(any(s["share"] > 2 / 3 for s in shares if not np.isnan(s["share"])))
    # 상태 나눔(설명): 대상 지난 20세션 수익의 250세션 백분위
    c = px[px["ticker"] == cell.target].set_index("date")["Close"]
    c = c[~c.index.duplicated()].reindex(raw.index).shift(1)
    st = hz.rolling_pct(np.log(c / c.shift(20)))
    split = {}
    for name, m in [("chase", st >= 50), ("dip", st < 50)]:
        mm = m.reindex(back.entries).values & use & (o == 1)
        split[name] = {"n_on": int(mm.sum()), "mean_on": float(np.nanmean(y[mm])) if mm.any() else np.nan}
    split["off_mean"] = float(off_mean)
    out["state_split"] = split
    out["robust_12_ok"] = bool(out["r1_breakwin"]["e_h"] > 0 and out.get("r2_dropcluster", {"e_h": -1})["e_h"] > 0)
    return out
