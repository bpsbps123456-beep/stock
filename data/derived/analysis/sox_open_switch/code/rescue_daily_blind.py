"""BRIEF7 K1·K2 일봉판(합성 SOXS 1995~2010) 눈가림 구현 B.

원래 연구 코드(lib.py, fam_F1.py, stage5.py, redfirst.py, ev4_build.py, c3_hyp.py)와
rescue_daily.py를 import하지 않는다. 정의는 BRIEF7·BRIEF6·BRIEF5와 trend_5zone.pine에서 직접 옮겼다.

- 단계: trend_5zone.pine zoneCalc (RSI14 Wilder, z20 표본 표준편차, bar_index < 129 은 na)
- K1: 단계(D-1) = -2, 단계(D-2) != -2 → D 시가 합성 SOXS 매수 → D+1 시가 매도
- K2: 단계(D-1) ∈ {-1,-2}, 단계(D-2) ∉ {-1,-2} → D 시가 → D 종가
- 합성: g_D = O_D/C_{D-1} - 1, r_D = C_D/C_{D-1} - 1
    D 시가 → D+1 시가: (1-3r_D)/(1-3g_D) × (1-3g_{D+1}) - 1
    D 시가 → D 종가 : (1-3r_D)/(1-3g_D) - 1
- 1995~2001 사건: ^SOX 단계·^SOX 합성(시가 = 전날 종가 비율 > 5%인 해는 제외)
  2002~2010 사건: SOXX 단계·SOXX 합성
- 비용 왕복 0.2%(감도 0.4%), 손익 %(1.0 = 1%)
- 무작위 날 대조: 같은 해 아무 거래일에서 같은 수(비복원)를 10,000번, seed 20261104, 평균 ≥ 관측 비율
- 대조: 2011~2025 실제 SOXS(yfinance) 같은 사건의 합성값과 상관
- 덧붙임(설명용): 2002~2010 ^SOX 단계판, K2 1분봉 O2 보조(2016~2022, Alpaca, cost_table_2016.json)
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "rescue_daily_blind.json")
CACHE_1M = os.path.join(ROOT, "cache", "alp1m_2016")
COST_2016 = os.path.join(ROOT, "cost_table_2016.json")

SEED = 20261104
NSIM = 10000
COST = 0.2      # 왕복 %, 합성 일봉
COST2 = 0.4     # 감도(비용 2배)
OPEN_BAD = 0.05  # ^SOX 시가 = 전날 종가 비율 문턱


# ───────────────────────── 자료 ─────────────────────────
def yf_daily(tic, start, end):
    df = yf.download(tic, start=start, end=end, auto_adjust=False, progress=False, actions=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].astype(float).dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ───────────────────────── 추세 5단계 (pine 이식) ─────────────────────────
def pine_rma(x, n):
    out = np.full(len(x), np.nan)
    s = np.nan
    for i in range(len(x)):
        if np.isnan(s):
            if i >= n - 1:
                w = x[i - n + 1:i + 1]
                if not np.isnan(w).any():
                    s = w.mean()
        else:
            s = (x[i] + (n - 1) * s) / n
        out[i] = s
    return out


def pine_rsi(c, n=14):
    c = np.asarray(c, float)
    ch = np.full(len(c), np.nan)
    ch[1:] = c[1:] - c[:-1]
    up = np.where(np.isnan(ch), np.nan, np.maximum(ch, 0.0))
    dn = np.where(np.isnan(ch), np.nan, np.maximum(-ch, 0.0))
    ru, rd = pine_rma(up, n), pine_rma(dn, n)
    rsi = np.full(len(c), np.nan)
    for i in range(len(c)):
        u, d = ru[i], rd[i]
        if np.isnan(u) or np.isnan(d):
            continue
        rsi[i] = 100.0 if d == 0 else (0.0 if u == 0 else 100.0 - 100.0 / (1.0 + u / d))
    return rsi


def stage_series(close, N=20, th=0.5, rlen=14, hi=70.0, lo=30.0):
    c = pd.Series(np.asarray(close, float), index=close.index)
    lc = np.log(c)
    lr = lc.diff()
    sd = lr.rolling(N).std(ddof=1)
    k = math.sqrt((N - 1.0) * (2.0 * N - 1.0) / (6.0 * N))
    z = (lc - lc.rolling(N).mean()) / (sd * k)
    r = pd.Series(pine_rsi(c.values, rlen), index=c.index)
    st = pd.Series(np.nan, index=c.index)
    bi = np.arange(len(c))
    ok = (~z.isna()) & (~r.isna()) & (bi >= 129)
    v = np.where(r >= hi, 2, np.where(r <= lo, -2, np.where(z >= th, 1, np.where(z <= -th, -1, 0))))
    st[ok] = v[ok.values]
    return pd.DataFrame({"close": c, "rsi": r, "z": z, "stage": st})


# ───────────────────────── 합성 SOXS ─────────────────────────
def synth_frame(px):
    """px: Open/Close 일봉. 날 D 기준 K1(D 시가→D+1 시가)·K2(D 시가→D 종가) 합성 SOXS %."""
    O, C = px["Open"], px["Close"]
    pc = C.shift(1)
    g = O / pc - 1.0
    r = C / pc - 1.0
    k2 = (1 - 3 * r) / (1 - 3 * g) - 1.0
    k1 = (1 - 3 * r) / (1 - 3 * g) * (1 - 3 * g.shift(-1)) - 1.0
    return pd.DataFrame({"g": g, "r": r, "k1": 100 * k1, "k2": 100 * k2,
                         "next_date": pd.Series(px.index, index=px.index).shift(-1)})


def open_quality(px):
    pc = px["Close"].shift(1)
    d = pd.DataFrame({"eq": (px["Open"] == pc), "eqc": (px["Open"] == px["Close"]), "y": px.index.year})
    d = d.iloc[1:]
    t = d.groupby("y").agg(n=("eq", "size"), open_eq_prevclose=("eq", "mean"), open_eq_close=("eqc", "mean"))
    return t


def events_from_stage(st):
    s1 = st["stage"].shift(1)  # 단계(D-1)
    s2 = st["stage"].shift(2)  # 단계(D-2)
    k1 = (s1 == -2) & s2.notna() & (s2 != -2)
    red1 = s1.isin([-1, -2])
    red2 = s2.isin([-1, -2])
    k2 = red1 & s2.notna() & (~red2)
    # D-2가 na(예열 끝 첫 값)인 경우 따로 센다
    k1_na2 = (s1 == -2) & s2.isna() & s1.notna()
    k2_na2 = red1 & s2.isna()
    return pd.DataFrame({"K1": k1, "K2": k2, "K1_d2na": k1_na2, "K2_d2na": k2_na2,
                         "stage_d1": s1, "stage_d2": s2})


# ───────────────────────── 통계 ─────────────────────────
def rand_p(obs_vals, obs_years, pool, nsim=NSIM, seed=SEED):
    """같은 해 아무 거래일에서 같은 수를 비복원으로 뽑은 평균 ≥ 관측 평균 비율."""
    obs_vals = np.asarray(obs_vals, float)
    if len(obs_vals) == 0:
        return None, None
    obs = obs_vals.mean()
    rng = np.random.default_rng(seed)
    yrs, cnt = np.unique(np.asarray(obs_years), return_counts=True)
    tot = np.zeros(nsim)
    for y, n in zip(yrs, cnt):
        arr = np.asarray(pool[int(y)], float)
        if len(arr) < n:
            raise RuntimeError(f"pool too small {y}: {len(arr)} < {n}")
        # 비복원 추출: 무작위 키 정렬의 앞 n개
        keys = rng.random((nsim, len(arr)))
        idx = np.argpartition(keys, n - 1, axis=1)[:, :n]
        tot += arr[idx].sum(axis=1)
    sims = tot / len(obs_vals)
    p = float((sims >= obs - 1e-12).mean())
    return p, float(sims.mean())


def summarize(df, val_gross, cost=COST, cost2=COST2, pool=None, years_col="year"):
    """df: 사건 표(year, gross %). 판정 5칸과 부가 통계."""
    g = df[val_gross].astype(float).values
    n = len(g)
    if n == 0:
        return {"n": 0}
    net = g - cost
    srt = np.sort(net)[::-1]
    drop2 = float(srt[2:].mean()) if n > 2 else None
    by = {}
    for y, sub in df.groupby(years_col):
        v = sub[val_gross].astype(float).values - cost
        by[str(int(y))] = {"n": int(len(v)), "mean": round(float(v.mean()), 4)}
    ypos = sum(1 for v in by.values() if v["mean"] > 0)
    out = {
        "n": int(n),
        "mean": round(float(net.mean()), 4),
        "mean_gross": round(float(g.mean()), 4),
        "median": round(float(np.median(net)), 4),
        "hit": round(float((net > 0).mean()), 4),
        "worst": round(float(net.min()), 4),
        "best": round(float(net.max()), 4),
        "drop_top2": None if drop2 is None else round(drop2, 4),
        "cost2x": round(float((g - cost2).mean()), 4),
        "by_year": by,
        "years_positive": f"{ypos}/{len(by)}",
        "years_half_positive": bool(ypos * 2 >= len(by)),
        "t": round(float(net.mean() / (net.std(ddof=1) / math.sqrt(n))), 3) if n > 2 else None,
    }
    if pool is not None:
        p, simmean = rand_p(net, df[years_col].values, {y: np.asarray(v) - cost for y, v in pool.items()})
        out["p_one_sided"] = p
        out["rand_mean"] = round(simmean, 4)
    out["pass"] = {
        "1_mean_pos": out["mean"] > 0,
        "2_p_raw_le_0.05": (out.get("p_one_sided") is not None and out["p_one_sided"] <= 0.05),
        "3_drop_top2_pos": (drop2 is not None and drop2 > 0),
        "4_cost2x_pos": out["cost2x"] > 0,
        "5_years_half_pos": out["years_half_positive"],
    }
    return out


def ev_list(df, cols):
    rows = []
    for d, r in df.iterrows():
        row = {"date": d.strftime("%Y-%m-%d")}
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                row[c] = None if np.isnan(v) else round(float(v), 4)
            elif isinstance(v, pd.Timestamp):
                row[c] = v.strftime("%Y-%m-%d")
            else:
                row[c] = bool(v) if isinstance(v, (bool, np.bool_)) else (int(v) if isinstance(v, np.integer) else v)
        rows.append(row)
    return rows


# ───────────────────────── 본 계산 ─────────────────────────
def build_main(sox, soxx, st_sox, st_soxx, oq_sox, soxx_bad_years=(), stage_src_late="SOXX", synth_src_late="SOXX",
               include_d2na=False):
    """1995~2010 사건 표. 앞 구간(≤2001)은 ^SOX, 뒤 구간(≥2002)은 지정 출처."""
    syn_sox = synth_frame(sox)
    syn_soxx = synth_frame(soxx)
    ev_sox = events_from_stage(st_sox)
    ev_soxx = events_from_stage(st_soxx)
    if include_d2na:  # 감도: 예열 끝 첫 값(D-2 na)도 첫날로 센다
        for ev in (ev_sox, ev_soxx):
            ev["K1"] = ev["K1"] | ev["K1_d2na"]
            ev["K2"] = ev["K2"] | ev["K2_d2na"]
    bad_sox = set(int(y) for y, v in oq_sox["open_eq_prevclose"].items() if v > OPEN_BAD)

    def year_ok(y, src):
        if src == "^SOX":
            return y not in bad_sox
        return y not in set(soxx_bad_years)

    res = {}
    pools = {"K1": {}, "K2": {}}
    excluded = {"K1": {}, "K2": {}}
    rows = {"K1": [], "K2": []}
    for y in range(1995, 2011):
        if y <= 2001:
            st_src, sy_src = "^SOX", "^SOX"
        else:
            st_src, sy_src = stage_src_late, synth_src_late
        ev = ev_sox if st_src == "^SOX" else ev_soxx
        syn = syn_sox if sy_src == "^SOX" else syn_soxx
        evy = ev[ev.index.year == y]
        syy = syn[syn.index.year == y]
        for K, col in (("K1", "k1"), ("K2", "k2")):
            days = evy.index[evy[K].values]
            # 합성 출처 달력에 그날이 있어야 한다
            days = [d for d in days if d in syn.index]
            ok_y = year_ok(y, sy_src)
            if K == "K1":
                # D+1 시가가 다른 해면 그 해 시가 품질도 본다
                def nxt_ok(d):
                    nd = syn.at[d, "next_date"]
                    return (not pd.isna(nd)) and year_ok(pd.Timestamp(nd).year, sy_src)
            else:
                def nxt_ok(d):
                    return True
            if not ok_y:
                excluded[K][str(y)] = len(days)
                continue
            for d in days:
                v = syn.at[d, col]
                if not nxt_ok(d) or pd.isna(v):
                    excluded[K].setdefault(f"{y}_edge", 0)
                    excluded[K][f"{y}_edge"] += 1
                    continue
                rows[K].append({"date": d, "year": y, "gross": float(v), "stage_d1": ev.at[d, "stage_d1"],
                                "stage_d2": ev.at[d, "stage_d2"], "g": float(syn.at[d, "g"]) * 100,
                                "r": float(syn.at[d, "r"]) * 100, "src": f"{st_src}/{sy_src}"})
            pv = syy[col].dropna()
            if K == "K1":
                pv = pv[[nxt_ok(d) for d in pv.index]]
            pools[K][y] = pv.values
    for K in ("K1", "K2"):
        df = pd.DataFrame(rows[K])
        if len(df):
            df = df.set_index("date")
        df_pool = {y: v for y, v in pools[K].items() if len(df) and y in set(df["year"])}
        res[K] = {"summary": summarize(df, "gross", pool=df_pool) if len(df) else {"n": 0},
                  "excluded_by_open_quality": excluded[K],
                  "events": ev_list(df, ["year", "gross", "stage_d1", "stage_d2", "g", "r", "src"]) if len(df) else []}
    return res


def compare_real(soxx, soxs, st_soxx, sox=None, y0=2011, y1=2025):
    syn = synth_frame(soxx)
    ev = events_from_stage(st_soxx)
    O, C = soxs["Open"], soxs["Close"]
    real_k2 = 100 * (C / O - 1.0)
    real_k1 = 100 * (O.shift(-1) / O - 1.0)
    real = pd.DataFrame({"k1": real_k1, "k2": real_k2})
    syn_sox = synth_frame(sox) if sox is not None else None
    out = {}
    for K, col in (("K1", "k1"), ("K2", "k2")):
        days = ev.index[ev[K].values & (ev.index.year >= y0) & (ev.index.year <= y1)]
        days = [d for d in days if d in real.index and d in syn.index]
        a = syn.loc[days, col].values.astype(float)
        b = real.loc[days, col].values.astype(float)
        m = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[m], b[m]
        dd = [d for d, mm in zip(days, m) if mm]
        o = {"n": int(len(a)), "corr": round(float(np.corrcoef(a, b)[0, 1]), 4),
             "synth_mean_gross": round(float(a.mean()), 4), "real_mean_gross": round(float(b.mean()), 4),
             "synth_mean_net": round(float(a.mean() - COST), 4), "real_mean_net": round(float(b.mean() - COST), 4),
             "mean_abs_diff": round(float(np.abs(a - b).mean()), 4),
             "slope_real_on_synth": round(float(np.polyfit(a, b, 1)[0]), 4),
             "sign_agree": round(float((np.sign(a) == np.sign(b)).mean()), 4)}
        # 모든 날 상관(사건 무관)
        allidx = syn.index.intersection(real.index)
        allidx = allidx[(allidx.year >= y0) & (allidx.year <= y1)]
        aa, bb = syn.loc[allidx, col].values, real.loc[allidx, col].values
        mm = ~np.isnan(aa) & ~np.isnan(bb)
        o["all_days_corr"] = round(float(np.corrcoef(aa[mm], bb[mm])[0, 1]), 4)
        o["all_days_n"] = int(mm.sum())
        if syn_sox is not None:
            dd2 = [d for d in dd if d in syn_sox.index]
            a2 = syn_sox.loc[dd2, col].values.astype(float)
            b2 = real.loc[dd2, col].values.astype(float)
            m2 = ~np.isnan(a2) & ~np.isnan(b2)
            o["sox_synth_corr"] = round(float(np.corrcoef(a2[m2], b2[m2])[0, 1]), 4)
            o["sox_synth_mean_gross"] = round(float(a2[m2].mean()), 4)
        # 해마다 실제 SOXS 평균(순)
        by = {}
        yrs = np.array([d.year for d in dd])
        for y in sorted(set(yrs)):
            v = b[yrs == y] - COST
            by[str(y)] = {"n": int(len(v)), "real_mean": round(float(v.mean()), 4),
                          "synth_mean": round(float(a[yrs == y].mean() - COST), 4)}
        o["by_year"] = by
        o["years_positive_real"] = f"{sum(1 for v in by.values() if v['real_mean'] > 0)}/{len(by)}"
        out[K] = o
    return out


# ───────────────────────── K2 1분봉 보조 (2016~2022) ─────────────────────────
def k2_minute_aux(st_soxx, costs, soxs_daily=None):
    px = pd.read_parquet(os.path.join(CACHE_1M, "SOXS.parquet"))
    idx = px.index
    day = idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()
    px = px.assign(day=day, hm=idx.hour * 100 + idx.minute)
    hs_open5 = costs["SOXS"]["open5"]
    hs_late = costs["SOXS"]["late"]

    def bar_close(sub, hm):
        """그 분 봉 종가. 없으면 그 앞 마지막 종가(같은 날)로 채우고 채움 표시."""
        hit = sub[sub["hm"] == hm]
        if len(hit):
            return float(hit["Close"].iloc[-1]), False
        prev = sub[sub["hm"] < hm]
        if len(prev):
            return float(prev["Close"].iloc[-1]), True
        return np.nan, True

    days_all = st_soxx.index[(st_soxx.index >= "2016-01-04") & (st_soxx.index <= "2022-12-30")]
    grp = {d: s for d, s in px.groupby("day")}
    rows = []
    for d in days_all:
        sub = grp.get(d)
        if sub is None:
            continue
        reg = sub[(sub["hm"] >= 930) & (sub["hm"] < 1600)]
        if len(reg) == 0:
            continue
        e, fe = bar_close(sub, 931)
        x, fx = bar_close(sub, 1059)
        if np.isnan(e) or np.isnan(x):
            continue
        r1 = 100 * (x * (1 - hs_late) / (e * (1 + hs_open5)) - 1)
        r2 = 100 * (x * (1 - 2 * hs_late) / (e * (1 + 2 * hs_open5)) - 1)
        r0 = 100 * (x / e - 1)
        rows.append({"date": d, "year": d.year, "net": r1, "net2x": r2, "gross": r0, "fill_e": fe, "fill_x": fx})
    allday = pd.DataFrame(rows).set_index("date")
    ev = events_from_stage(st_soxx)
    evd = ev.index[ev["K2"].values]
    evd = [d for d in evd if d in allday.index]
    E = allday.loc[evd].copy()
    E["year"] = E["year"].astype(int)
    pool = {int(y): allday[allday["year"] == y]["net"].values for y in set(E["year"])}

    def summ(E, pool):
        net = E["net"].values
        n = len(net)
        srt = np.sort(net)[::-1]
        by = {}
        for y, sub in E.groupby("year"):
            by[str(int(y))] = {"n": int(len(sub)), "mean": round(float(sub["net"].mean()), 4)}
        ypos = sum(1 for v in by.values() if v["mean"] > 0)
        p, sm = rand_p(net, E["year"].values, pool)
        return {"n": int(n), "mean": round(float(net.mean()), 4), "median": round(float(np.median(net)), 4),
                "hit": round(float((net > 0).mean()), 4), "worst": round(float(net.min()), 4),
                "drop_top2": round(float(srt[2:].mean()), 4) if n > 2 else None,
                "cost2x": round(float(E["net2x"].mean()), 4), "p_one_sided": p, "rand_mean": round(sm, 4),
                "by_year": by, "years_positive": f"{ypos}/{len(by)}"}

    s_all = summ(E, pool)
    filled = E["fill_e"] | E["fill_x"]
    s_all["filled_trades"] = int(filled.sum())
    s_all["filled_entry"] = int(E["fill_e"].sum())
    s_all["filled_exit"] = int(E["fill_x"].sum())
    s_all["filled_by_year"] = {str(int(y)): int(v) for y, v in E[filled].groupby("year").size().items()}
    Ef = E[~filled]
    pool_f = {int(y): allday[(allday["year"] == y) & ~(allday["fill_e"] | allday["fill_x"])]["net"].values
              for y in set(Ef["year"])}
    s_nofill = summ(Ef, pool_f) if len(Ef) else {"n": 0}
    s_all["events"] = ev_list(E, ["year", "net", "gross", "fill_e", "fill_x"])
    s_all["pool_days"] = int(len(allday))
    s_all["pool_filled_days"] = int((allday["fill_e"] | allday["fill_x"]).sum())
    if soxs_daily is not None:  # 같은 사건의 일봉 근사(실제 SOXS D 시가 → D 종가, 비용 0.2%)
        dd = [d for d in E.index if d in soxs_daily.index]
        v = 100 * (soxs_daily.loc[dd, "Close"] / soxs_daily.loc[dd, "Open"] - 1) - COST
        s_all["daily_open_close_same_events"] = {"n": int(len(v)), "mean_net": round(float(v.mean()), 4),
                                                 "corr_with_1m": round(float(np.corrcoef(v.values, E.loc[dd, "net"].values)[0, 1]), 4)}
    return {"all": s_all, "excl_filled": s_nofill,
            "half_spread": {"entry_open5": hs_open5, "exit_late": hs_late}}


# ───────────────────────── main ─────────────────────────
def main():
    sox = yf_daily("^SOX", "1994-01-01", "2011-01-10")
    soxx = yf_daily("SOXX", "2001-01-01", "2011-01-10")
    soxx_long = yf_daily("SOXX", "2001-01-01", "2026-01-10")
    soxs = yf_daily("SOXS", "2010-01-01", "2026-01-10")
    sox_long = yf_daily("^SOX", "1994-01-01", "2026-01-10")

    st_sox = stage_series(sox["Close"])
    st_soxx = stage_series(soxx["Close"])
    st_soxx_long = stage_series(soxx_long["Close"])
    st_sox_long = stage_series(sox_long["Close"])

    oq_sox = open_quality(sox)
    oq_soxx = open_quality(soxx)
    bad_sox = sorted(int(y) for y, v in oq_sox["open_eq_prevclose"].items() if v > OPEN_BAD and 1995 <= y <= 2010)
    bad_soxx = sorted(int(y) for y, v in oq_soxx["open_eq_prevclose"].items() if v > OPEN_BAD and 2002 <= y <= 2010)

    checks = {}
    # 단계 함수 검증 1: 앞선 연구의 SOXX 단계 CSV(2021~ 예열)와 2022-01~ 비교 (같은 계산이면 거의 100%)
    try:
        ref = pd.read_csv(os.path.join(ROOT, "results", "stage5_stages.csv"), encoding="utf-8-sig", parse_dates=["date"])
        ref = ref.set_index("date")["stage"].dropna()
        ref = ref[ref.index >= "2022-01-01"]
        mine = st_soxx_long["stage"].reindex(ref.index)
        m = mine.notna()
        checks["stage_vs_stage5_stages_csv"] = {"n": int(m.sum()), "agree": round(float((mine[m] == ref[m]).mean()), 5),
                                                 "note": "ref는 2021 예열, 내 값은 2001 예열 — 2022 이후 비교"}
    except Exception as e:  # noqa
        checks["stage_vs_stage5_stages_csv"] = f"skip: {e}"
    # 단계 검증 2: 짧은 SOXX(2011-01-10까지)와 긴 SOXX(2026까지)가 2002~2010에서 같은 단계인가
    a = st_soxx["stage"]
    b = st_soxx_long["stage"].reindex(a.index)
    m = a.notna() & b.notna()
    checks["stage_short_vs_long_soxx_2002_2010"] = round(float((a[m] == b[m]).mean()), 5)
    a = st_sox["stage"]
    b = st_sox_long["stage"].reindex(a.index)
    m = a.notna() & b.notna()
    checks["stage_short_vs_long_sox"] = round(float((a[m] == b[m]).mean()), 5)
    checks["first_stage_date"] = {"^SOX": str(st_sox["stage"].first_valid_index().date()),
                                  "SOXX": str(st_soxx["stage"].first_valid_index().date())}
    checks["data_start"] = {"^SOX": str(sox.index[0].date()), "SOXX": str(soxx.index[0].date()),
                            "SOXS": str(soxs.index[0].date())}
    # 2002년 SOXX 단계 na 구간에 놓인 빨강·극하방 첫날(^SOX 기준) 수 — 설명
    ev_sox_all = events_from_stage(st_sox)
    soxx_first = st_soxx["stage"].first_valid_index()
    early2002 = ev_sox_all[(ev_sox_all.index.year == 2002) & (ev_sox_all.index <= soxx_first + pd.Timedelta(days=3))]
    checks["early2002_sox_events_lost"] = {"K1": int(early2002["K1"].sum()), "K2": int(early2002["K2"].sum())}
    ev_soxx_all = events_from_stage(st_soxx)
    checks["d2_na_events"] = {"SOXX_K1": int(ev_soxx_all["K1_d2na"].sum()), "SOXX_K2": int(ev_soxx_all["K2_d2na"].sum()),
                              "SOX_K1": int(ev_sox_all["K1_d2na"].sum()), "SOX_K2": int(ev_sox_all["K2_d2na"].sum())}

    # 본 판정
    main_res = build_main(sox, soxx, st_sox, st_soxx, oq_sox)
    # 감도: SOXX에도 같은 5% 시가 규칙
    sens_soxx_rule = build_main(sox, soxx, st_sox, st_soxx, oq_sox, soxx_bad_years=bad_soxx)
    sens_d2na = build_main(sox, soxx, st_sox, st_soxx, oq_sox, include_d2na=True)
    # 설명용: 2002~2010 ^SOX 단계 + SOXX 합성 / ^SOX 단계 + ^SOX 합성(통과한 해만)
    sox_stage_soxx_syn = build_main(sox, soxx, st_sox, st_soxx, oq_sox, stage_src_late="^SOX", synth_src_late="SOXX")
    sox_stage_sox_syn = build_main(sox, soxx, st_sox, st_soxx, oq_sox, stage_src_late="^SOX", synth_src_late="^SOX")

    def only_late(r):
        out = {}
        for K in ("K1", "K2"):
            evs = [e for e in r[K]["events"] if e["year"] >= 2002]
            df = pd.DataFrame(evs)
            if len(df) == 0:
                out[K] = {"n": 0}
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            out[K] = summarize(df, "gross")
            out[K]["src"] = evs[0]["src"]
        return out

    real_cmp = compare_real(soxx_long, soxs, st_soxx_long, sox=sox_long)
    oq_long = open_quality(sox_long)

    # 1분봉 보조
    costs = json.load(open(COST_2016, encoding="utf-8"))
    try:
        aux = k2_minute_aux(st_soxx_long, costs, soxs_daily=soxs)
    except Exception as e:  # noqa
        aux = {"error": repr(e)}

    res = {
        "brief": "BRIEF7 K1·K2 일봉판(합성 SOXS 1995~2010), 눈가림 구현 B",
        "unit": "% (1.0 = 1%), 사건당, 비용 왕복 0.2% 뺀 값(cost2x는 0.4%)",
        "open_quality": {
            "^SOX": {str(int(y)): {"n": int(r.n), "open_eq_prevclose": round(float(r.open_eq_prevclose), 4),
                                   "open_eq_close": round(float(r.open_eq_close), 4)} for y, r in oq_sox.iterrows()},
            "SOXX": {str(int(y)): {"n": int(r.n), "open_eq_prevclose": round(float(r.open_eq_prevclose), 4)}
                     for y, r in oq_soxx.iterrows()},
            "^SOX_2011_2025": {str(int(y)): round(float(r.open_eq_prevclose), 4) for y, r in oq_long.iterrows() if y >= 2011},
            "rule": "^SOX 시가 = 전날 종가 비율 > 5%인 해는 시가 규칙에서 뺀다(1995~2001에 적용)",
            "sox_bad_years_1995_2010": bad_sox,
            "soxx_bad_years_2002_2010_(감도)": bad_soxx,
        },
        "main": main_res,
        "sens_soxx_open_rule": {K: sens_soxx_rule[K]["summary"] for K in ("K1", "K2")},
        "sens_include_d2na_first_stage": {K: {k: sens_d2na[K]["summary"].get(k) for k in
                                              ("n", "mean", "drop_top2", "cost2x", "p_one_sided", "years_positive")}
                                          for K in ("K1", "K2")},
        "explain_2002_2010_sox_stage": {"sox_stage+soxx_synth": only_late(sox_stage_soxx_syn),
                                         "sox_stage+sox_synth(통과 해만)": only_late(sox_stage_sox_syn),
                                         "soxx_stage+soxx_synth(본판의 2002~2010)": only_late(main_res)},
        "real_vs_synth_2011_2025": real_cmp,
        "k2_minute_aux_2016_2022": aux,
        "checks": checks,
        "settings": {"seed": SEED, "nsim": NSIM, "cost": COST, "cost2x": COST2, "open_bad": OPEN_BAD,
                     "p": "같은 해 아무 거래일(합성값 있는 날)에서 같은 수 비복원 추출, 평균 ≥ 관측 비율",
                     "holm": "Holm m=6은 K1~K6 묶음을 합친 뒤 적용(여기 p는 원 p)"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    # 요약 출력
    for K in ("K1", "K2"):
        s = main_res[K]["summary"]
        print(K, {k: s.get(k) for k in ("n", "mean", "median", "hit", "worst", "drop_top2", "cost2x", "p_one_sided", "years_positive")})
        print("  by_year", s.get("by_year"))
        print("  excluded", main_res[K]["excluded_by_open_quality"])
        print("  sens soxx rule", {k: sens_soxx_rule[K]["summary"].get(k) for k in ("n", "mean", "p_one_sided", "years_positive")})
        print("  real", {k: v for k, v in real_cmp[K].items() if k != "by_year"})
    print("explain", json.dumps(res["explain_2002_2010_sox_stage"], ensure_ascii=False, default=str)[:3000])
    if "all" in aux:
        print("aux", {k: aux["all"].get(k) for k in ("n", "mean", "median", "drop_top2", "cost2x", "p_one_sided", "years_positive", "filled_trades")})
        print("aux daily same events", aux["all"].get("daily_open_close_same_events"))
        print("aux excl filled", {k: aux["excl_filled"].get(k) for k in ("n", "mean", "drop_top2", "cost2x", "p_one_sided", "years_positive")})
    else:
        print("aux", aux)
    print("checks", checks)
    print("d2na sens", res["sens_include_d2na_first_stage"])
    print("open bad", bad_sox, bad_soxx)


if __name__ == "__main__":
    main()
