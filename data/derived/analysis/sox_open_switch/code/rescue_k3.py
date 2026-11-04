"""BRIEF7 K3 — 극상방 날 SOXS 09:31→09:59, 안 본 자료 2016~2022 한 번 (구현 A, 2026-11-04). K2 보조(판정 밖) 함께.

정본: ../BRIEF7.md(결과 보기 전 고정·커밋). K3 출처: BRIEF5 사후 단서(표 A 극상방 × 늘 SOXS × std).
단계 함수: stage5.zone_calc, 범주: redfirst.flags, 무작위 뽑기: redfirst.null_draws, 전략: stage5.Always(-1). 엔진: lib.py.
위 코드의 계산은 고치지 않고 불러 쓴다(자료 경로·기간·비용표만 바꿈).

결과 보기 전에 이 파일에서 정한 해석(첫 실행 전 고정):
- 일봉: yfinance SOXX 2001-01-01~2023-01-10(auto_adjust=False Close = 분할 반영·배당 미반영, 실제 첫 봉 2001-07-13).
  내려받기에 실패하면 기존 캐시(soxx_1d_yf_ohlc_20010101_20261029.csv)를 2023-01-10에서 자른다(어느 쪽인지 기록).
  내려받으면 기존 캐시와 단계가 같은지 겹치는 날 전부를 대조한다(단계는 앞 봉만 쓰므로 같아야 한다).
- 단계: stage5.zone_calc(첫 130봉 값 없음). D(Data 날)의 D−1 = yfinance 날짜열에서 D 바로 앞 거래일.
- 1분봉: Data('alp1m_2016') 2016-01-04~2022-12-30. 비용: cost_table_2016.json을 Engine(D, costs=...)에 넘긴다.
- K3 사건: 단계(D−1) = +2(극상방, RSI14 ≥ 70). 매매: stage5.Always(-1) = 09:30 봉 마감 결정 → 09:31 봉 종가 × (1 + 반 호가 open5)
  → Engine(scheme='std') 마감 1분 전 09:59 봉 종가 × (1 − 반 호가 early). 날 값 = 그 매매 수익(%). 사건 날 평균이 통계량.
- K2 보조(판정 밖): redfirst.flags(zn)['R1'](단계(D−1) ≤ −1, 단계(D−2) > −1, 둘 다 값 있음) 날 D, 같은 Always(-1)를
  Engine(scheme='w90') 10:59 봉 종가 청산. 같은 날 청산이라 겹침 없음.
- 무작위 날 대조(K3·K2 같은 방법): 해 층화. rng = np.random.default_rng(20261104) 하나를 2016→2022 순서로 쓰고,
  해마다 그 해 1분봉 날 중 값이 있는 날 전부(사건 날 포함)에서 그 해 사건 수만큼 비복원 10,000회(redfirst.null_draws).
  귀무 평균 = 해마다 뽑은 합의 합 ÷ 전체 사건 수. p(단측) = #{귀무 평균 ≥ 관측 − 1e-12} ÷ 10,000, 참고로 (1+#)/(1+10,000).
  해마다 p는 같은 뽑기의 그 해 평균으로 낸다. 검정(K3 본, K3 감도, K2 본, K2 감도)마다 rng를 새로 만든다.
- 판정(K3, BRIEF7 다섯 항목): (1) 평균 > 0 (2) 단측 p ≤ 0.05(Holm m = 6 — 다른 후보 p가 필요해 여기서는
  6p ≤ 0.05면 확정 통과, p > 0.05면 확정 불통과, 그 사이는 '다른 후보에 달림'으로 적는다) (3) 날 값 큰 2일을 뺀 평균 > 0
  (4) 비용 2배(Engine cost_mult=2) 평균 > 0 (5) 사건 있는 해 중 해 평균 > 0인 해 수 × 2 ≥ 사건 있는 해 수.
- 채운 봉: Data는 빈 분을 앞 종가로 채운다. 체결 봉(09:31)이나 청산 봉(K3 09:59, K2 10:59)의 D.has['SOXS']가 False인
  매매 수를 해마다 센다. 감도: 그런 날을 사건과 무작위 풀에서 모두 빼고 같은 다섯 항목을 다시 잰다(판정은 본 값으로).
- 대조: 엔진 값과 직접 계산(C[09:31]·C[09:59 또는 10:59], 같은 반 호가 칸) 최대 차이, 엔진 체결·청산 시각,
  yfinance 날과 Data 날 대조, yfinance SOXX 종가 대 Alpaca SOXX 정규장 종가(분할 반영 확인).
- 이어진 구간: 해 안 Data 날 순서로 극상방(D−1)이 연속한 묶음 수(실질 독립 표본 참고).
실행: cd code && py rescue_k3.py
결과: ../results/rescue_k3.json, ../results/rescue_k3_events.csv
"""
import hashlib, json, os, sys, time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import Data, Engine, bucket, idx  # noqa: E402
import stage5 as S5  # noqa: E402
import redfirst as RF  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cache")
SEED, NR = 20261104, 10000
YEARS = tuple(range(2016, 2023))
I_ENT, I_X_STD, I_X_W90 = idx("09:31"), idx("09:59"), idx("10:59")
M_HOLM, ALPHA = 6, 0.05
T0 = time.time()
fnum = S5.fnum


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- 일봉
def soxx_daily():
    p = os.path.join(CACHE, "soxx_1d_yf_ohlc_20010101_20230110.csv")
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0, parse_dates=True), "yfinance(캐시 " + os.path.basename(p) + ")"
    try:
        import yfinance as yf
        df = yf.download("SOXX", start="2001-01-01", end="2023-01-11", auto_adjust=False, progress=False, actions=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].astype(float)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[df["Close"].notna()]
        assert len(df) > 5000 and df.index[-1] == pd.Timestamp("2023-01-10")
        df.to_csv(p)
        return df, "yfinance 새로 내려받음 2001-01-01~2023-01-10"
    except Exception as e:  # noqa: BLE001
        q = os.path.join(CACHE, "soxx_1d_yf_ohlc_20010101_20261029.csv")
        df = pd.read_csv(q, index_col=0, parse_dates=True)
        return df[df.index <= "2023-01-10"], f"내려받기 실패({type(e).__name__}) → 기존 캐시를 2023-01-10에서 자름"


# ---------------------------------------------------------------- 통계
def strat_test(vals, years, pools, seed=SEED):
    """vals: 사건 날 값, years: 사건 날 해, pools: {해: 그 해 풀 값}. 해 층화 무작위 날 대조."""
    rng = np.random.default_rng(seed)
    obs = float(np.mean(vals))
    null_sum, n_tot, per = np.zeros(NR), 0, {}
    for y in YEARS:
        v = vals[years == y]
        n = len(v)
        cell = {"n": int(n), "pool_days": int(len(pools[y])), "pool_mean": fnum(np.mean(pools[y]))}
        if n:
            nm, ns = RF.null_draws(pools[y], n, rng)
            k = int((nm >= v.mean() - 1e-12).sum())
            cell.update({"p_random": k / NR, "null_mean": fnum(nm.mean())})
            null_sum += ns
            n_tot += n
        per[y] = cell
    null = null_sum / n_tot
    k = int((null >= obs - 1e-12).sum())
    return {"p_random": k / NR, "p_random_plus1": (1 + k) / (NR + 1), "null_mean": fnum(null.mean()),
            "null_q95": fnum(np.percentile(null, 95)), "null_sd": fnum(null.std(ddof=1))}, per


def block(vals, c2, years, dates, pools, label):
    """한 후보(또는 감도)의 다섯 항목과 해마다 표."""
    vals, c2, years = np.asarray(vals, float), np.asarray(c2, float), np.asarray(years)
    n = len(vals)
    out = {"label": label, "n": int(n)}
    if n == 0:
        return out
    srt = np.argsort(-vals, kind="stable")
    top2 = srt[:2]
    keep = np.ones(n, bool)
    keep[top2] = False
    out.update({"mean": fnum(vals.mean()), "median": fnum(np.median(vals)), "sd": fnum(vals.std(ddof=1)) if n > 1 else None,
                "hit": fnum((vals > 0).mean()), "worst": fnum(vals.min()), "best": fnum(vals.max()), "sum": fnum(vals.sum()),
                "ci95_boot": S5.boot_mean(vals, seed=SEED) if n > 1 else None,
                "cost2x_mean": fnum(c2.mean()),
                "drop_top2": {"dates": [str(dates[i]) for i in top2], "values": [fnum(vals[i]) for i in top2],
                              "mean": fnum(vals[keep].mean()) if keep.any() else None, "n": int(keep.sum())}})
    tst, per = strat_test(vals, years, pools)
    out.update(tst)
    by = {}
    for y in YEARS:
        m = years == y
        v = vals[m]
        by[str(y)] = {"n": int(m.sum()), "mean": fnum(v.mean()) if m.any() else None,
                      "hit": fnum((v > 0).mean()) if m.any() else None,
                      "cost2x_mean": fnum(c2[m].mean()) if m.any() else None,
                      "p_random": per[y].get("p_random"), "null_mean": per[y].get("null_mean"),
                      "pool_mean": per[y]["pool_mean"], "pool_days": per[y]["pool_days"]}
    out["by_year"] = by
    ye = [c for c in by.values() if c["n"]]
    out["years_with_events"] = len(ye)
    out["years_positive"] = int(sum(1 for c in ye if c["mean"] > 0))
    return out


def judge(b):
    p = b["p_random"]
    if M_HOLM * p <= ALPHA:
        p_state = "확정 통과(6p ≤ 0.05)"
    elif p > ALPHA:
        p_state = "확정 불통과(p > 0.05)"
    else:
        p_state = "다른 후보 p에 달림(0.05/6 < p ≤ 0.05)"
    crit = {"(1) mean>0": bool(b["mean"] > 0),
            "(2) p<=0.05_holm_m6": p_state,
            "(3) drop_top2_mean>0": bool(b["drop_top2"]["mean"] is not None and b["drop_top2"]["mean"] > 0),
            "(4) cost2x_mean>0": bool(b["cost2x_mean"] > 0),
            "(5) years_positive>=half": bool(2 * b["years_positive"] >= b["years_with_events"])}
    others = [crit[k] for k in crit if k != "(2) p<=0.05_holm_m6"]
    if not all(others) or p_state.startswith("확정 불통과"):
        verdict = "불통과"
    elif p_state.startswith("확정 통과"):
        verdict = "통과"
    else:
        verdict = "(2) 외 통과, Holm은 다른 후보에 달림"
    return {"p_raw": p, "p_bonferroni_m6": fnum(min(1.0, M_HOLM * p)), "criteria": crit, "verdict": verdict}


def runs_of(mask):
    m = np.asarray(mask, bool)
    return int(m[0] + ((m[1:]) & (~m[:-1])).sum()) if len(m) else 0


# ---------------------------------------------------------------- 본체
def main():
    os.makedirs(RES, exist_ok=True)
    checks = {}
    costs = json.load(open(os.path.join(ROOT, "cost_table_2016.json"), encoding="utf-8"))
    checks["costs_2016"] = costs

    # 1. 일봉·단계
    X, src = soxx_daily()
    datesX = pd.DatetimeIndex(X.index)
    cv = X["Close"].to_numpy(float)
    zn, r, z = S5.zone_calc(cv)
    checks["soxx_daily"] = {"source": src, "first": str(datesX[0].date()), "last": str(datesX[-1].date()), "bars": int(len(cv)),
                            "first_valid_stage": str(datesX[np.argmax(np.isfinite(zn))].date())}
    log("SOXX", checks["soxx_daily"])
    q = os.path.join(CACHE, "soxx_1d_yf_ohlc_20010101_20261029.csv")
    if os.path.exists(q):
        old = pd.read_csv(q, index_col=0, parse_dates=True)
        old = old[old.index <= datesX[-1]]
        zo, _, _ = S5.zone_calc(old["Close"].to_numpy(float))
        so = pd.Series(zo, index=old.index).reindex(datesX)
        okm = np.isfinite(so.values) & np.isfinite(zn)
        checks["stage_vs_cached_2001_2026_file"] = {
            "same_dates": bool(len(old) == len(datesX) and (pd.DatetimeIndex(old.index) == datesX).all()),
            "n": int(okm.sum()), "agree": fnum((so.values[okm] == zn[okm]).mean()),
            "close_rel_diff_max_pct": fnum((np.abs(old["Close"].reindex(datesX).values / cv - 1) * 100).max())}
        log("캐시 대조", checks["stage_vs_cached_2001_2026_file"])
    s5p = os.path.join(RES, "stage5_stages.csv")
    if os.path.exists(s5p):
        s5 = pd.read_csv(s5p, parse_dates=["date"]).set_index("date")
        s5 = s5[(s5.index >= "2022-12-01") & (s5.index <= datesX[-1])]
        mine = pd.Series(zn, index=datesX).reindex(s5.index)
        okm = s5["stage"].notna() & mine.notna()
        checks["stage_vs_stage5_2022_12_to_2023_01_10"] = {"n": int(okm.sum()), "agree": fnum((s5["stage"][okm] == mine[okm]).mean())}

    # 2. 1분봉
    D = Data("alp1m_2016")
    days = [pd.Timestamp(d) for d in D.days]
    nd = len(days)
    tX = {d: t for t, d in enumerate(datesX)}
    tD = np.array([tX.get(d, -1) for d in days])
    in_rng = [d for d in datesX if days[0] <= d <= days[-1]]
    checks["data_days"] = {"first": str(days[0].date()), "last": str(days[-1].date()), "n": nd,
                           "missing_in_yf_soxx": [str(days[k].date()) for k in np.flatnonzero(tD < 0)],
                           "yf_days_not_in_data": [str(d.date()) for d in in_rng if d not in set(days)],
                           "consecutive_in_yf": bool((np.diff(tD[tD >= 0]) == 1).all())}
    log("Data", {k: (v if not isinstance(v, list) else len(v)) for k, v in checks["data_days"].items()})
    pos = np.searchsorted(datesX.values, np.array(days, dtype="datetime64[ns]"), side="left") - 1
    assert (pos >= 0).all()
    stage_prev = zn[pos]
    rsi_prev = r[pos]
    assert np.isfinite(stage_prev).all()
    checks["prev_date_equals_data_prev_day"] = {"n_checked": nd - 1,
                                                "equal": int(sum(datesX[pos[k]] == days[k - 1] for k in range(1, nd)))}
    al = D.daily.loc["SOXX"]["close"].values
    yc = X["Close"].reindex(days).values
    rel = np.abs(yc / al - 1) * 100
    checks["yf_vs_alpaca_soxx_close_abs_pct"] = {"median": fnum(np.nanmedian(rel)), "p99": fnum(np.nanpercentile(rel, 99)),
                                                 "max": fnum(np.nanmax(rel))}
    yrD = np.array([d.year for d in days])
    dstr = np.array([str(d.date()) for d in days])

    # 3. 엔진
    arr, eng_chk = {}, {}
    for key, sch in (("std", "std"), ("w90", "w90")):
        for cm in (1.0, 2.0):
            E = Engine(D, costs=costs, scheme=sch, cost_mult=cm)
            daily, tr = E.run(lambda: S5.Always(-1), days=list(range(nd)))
            assert len(daily) == nd and (daily["n"] == 1).all()
            arr[f"{key}_c{int(cm)}"] = daily["ret"].values * 100
            eng_chk[f"{key}_c{int(cm)}"] = {"entry": sorted(tr["entry"].unique().tolist()), "exit": sorted(tr["exit"].unique().tolist()),
                                           "why": {k: int(x) for k, x in tr["why"].value_counts().items()},
                                           "sym": sorted(tr["sym"].unique().tolist()),
                                           "nonfinite": int((~np.isfinite(daily["ret"].values)).sum()),
                                           "nonfinite_days": [str(days[k].date()) for k in np.flatnonzero(~np.isfinite(daily["ret"].values))]}
    checks["engine"] = eng_chk
    log("엔진", eng_chk)
    C = D.A["SOXS"]["Close"]
    for key, ix in (("std", I_X_STD), ("w90", I_X_W90)):
        for cm in (1.0, 2.0):
            E = Engine(D, costs=costs, cost_mult=cm)
            dd = (C[:, ix] * (1 - E.hs("SOXS", ix)) / (C[:, I_ENT] * (1 + E.hs("SOXS", I_ENT))) - 1) * 100
            checks[f"direct_vs_engine_{key}_c{int(cm)}_max_abs_diff"] = fnum(np.nanmax(np.abs(dd - arr[f"{key}_c{int(cm)}"])))
            checks[f"direct_cost_buckets_{key}"] = [bucket(I_ENT), bucket(ix)]
    assert checks["direct_vs_engine_std_c1_max_abs_diff"] < 1e-9 and checks["direct_vs_engine_w90_c1_max_abs_diff"] < 1e-9
    log("직접 대조", {k: v for k, v in checks.items() if k.startswith("direct_vs")})

    # 4. 채운 봉
    has = D.has["SOXS"]
    f_ent = ~has[:, I_ENT]
    f_x = {"std": ~has[:, I_X_STD], "w90": ~has[:, I_X_W90]}
    filled = {k: f_ent | f_x[k] for k in f_x}
    checks["filled_bars_all_days_by_year"] = {str(y): {"days": int((yrD == y).sum()), "entry_0931": int(f_ent[yrD == y].sum()),
                                                       "exit_0959": int(f_x["std"][yrD == y].sum()),
                                                       "exit_1059": int(f_x["w90"][yrD == y].sum())} for y in YEARS}

    # 5. 사건
    ev_k3 = stage_prev == 2
    FX = RF.flags(zn)
    ev_k2 = np.array([bool(FX["R1"][t]) if t >= 0 else False for t in tD])
    st_runs = {str(y): {"k3_days": int(ev_k3[yrD == y].sum()), "k3_runs": runs_of(ev_k3[yrD == y]),
                        "k2_days": int(ev_k2[yrD == y].sum())} for y in YEARS}

    def pools(key, excl=None):
        out = {}
        for y in YEARS:
            m = (yrD == y) & np.isfinite(arr[f"{key}_c1"])
            if excl is not None:
                m &= ~excl
            out[y] = arr[f"{key}_c1"][m]
        return out

    def run_block(ev, key, label, excl=None):
        m = ev & np.isfinite(arr[f"{key}_c1"])
        if excl is not None:
            m &= ~excl
        k = np.flatnonzero(m)
        b = block(arr[f"{key}_c1"][k], arr[f"{key}_c2"][k], yrD[k], dstr[k], pools(key, excl), label)
        b["dropped_nonfinite"] = int((ev & ~np.isfinite(arr[f"{key}_c1"])).sum())
        return b

    k3 = run_block(ev_k3, "std", "K3 본(모든 사건 날)")
    k3["judgement"] = judge(k3)
    k3["filled_trades"] = {"total": int((ev_k3 & filled["std"]).sum()), "entry_0931": int((ev_k3 & f_ent).sum()),
                           "exit_0959": int((ev_k3 & f_x["std"]).sum()),
                           "by_year": {str(y): int((ev_k3 & filled["std"] & (yrD == y)).sum()) for y in YEARS}}
    k3s = run_block(ev_k3, "std", "K3 감도(체결·청산 봉이 채운 봉인 날을 사건·풀에서 뺌)", excl=filled["std"])
    k3s["judgement_same_rules"] = judge(k3s)
    log("K3", {kk: k3.get(kk) for kk in ("n", "mean", "p_random", "cost2x_mean")}, k3["judgement"])
    log("K3 감도", {kk: k3s.get(kk) for kk in ("n", "mean", "p_random", "cost2x_mean")})

    k2 = run_block(ev_k2, "w90", "K2 보조(빨강 첫날 R1, 09:31→10:59, 판정 밖)")
    k2["filled_trades"] = {"total": int((ev_k2 & filled["w90"]).sum()), "entry_0931": int((ev_k2 & f_ent).sum()),
                           "exit_1059": int((ev_k2 & f_x["w90"]).sum()),
                           "by_year": {str(y): int((ev_k2 & filled["w90"] & (yrD == y)).sum()) for y in YEARS}}
    k2s = run_block(ev_k2, "w90", "K2 보조 감도(채운 봉 날 뺌)", excl=filled["w90"])
    log("K2", {kk: k2.get(kk) for kk in ("n", "mean", "p_random", "cost2x_mean")})

    # 조건 없는 늘 SOXS(설명용)
    base = {key: {str(y): fnum(np.nanmean(arr[f"{key}_c1"][yrD == y])) for y in YEARS} for key in ("std", "w90")}
    for key in ("std", "w90"):
        base[key]["all"] = fnum(np.nanmean(arr[f"{key}_c1"]))

    # 앞 연구(2023~2026, 기본 비용표) 참고값 — 다시 계산하지 않고 읽기만
    prior = None
    s5j = os.path.join(RES, "stage5.json")
    if os.path.exists(s5j):
        J = json.load(open(s5j, encoding="utf-8"))
        prior = {y: next(({"days": c["days"], "mean": c["mean"]} for c in J["tableA"][y]["cells"]
                          if c["stage"] == 2 and c["col"] == "SOXS_std"), None) for y in J["tableA"]}

    # 사건 표
    rows = []
    XO, XC = D.A["SOXX"]["Open"], D.A["SOXX"]["Close"]
    for k in np.flatnonzero(ev_k3 | ev_k2):
        rows.append({"date": dstr[k], "year": int(yrD[k]), "K3": bool(ev_k3[k]), "K2_R1": bool(ev_k2[k]),
                     "stage_d1": int(stage_prev[k]), "rsi14_d1": fnum(rsi_prev[k]),
                     "soxs_0931_0959": fnum(arr["std_c1"][k]), "soxs_0931_0959_c2": fnum(arr["std_c2"][k]),
                     "soxs_0931_1059": fnum(arr["w90_c1"][k]), "soxs_0931_1059_c2": fnum(arr["w90_c2"][k]),
                     "filled_0931": bool(f_ent[k]), "filled_0959": bool(f_x["std"][k]), "filled_1059": bool(f_x["w90"][k]),
                     "soxx_open_to_1000_pct": fnum((XC[k, I_X_STD] / XO[k, idx("09:30")] - 1) * 100)})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "rescue_k3_events.csv"), index=False, encoding="utf-8-sig")

    code_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
    out = {"brief": "BRIEF7.md", "candidate": "K3 (+ K2 보조)", "impl": "A (rescue_k3.py)", "code_sha256_16": code_sha,
           "units": "% (1.0 = 1%)",
           "settings": {"seed": SEED, "random_draws": NR, "years": list(YEARS),
                        "k3": "단계(D−1) = +2 → SOXS 09:31 봉 종가 + 반 호가(open5) → 09:59 봉 종가 − 반 호가(early), Engine std",
                        "k2_aux": "R1(단계(D−1) ≤ −1, 단계(D−2) > −1) → SOXS 09:31 → 10:59 봉 종가 − 반 호가(late), Engine w90",
                        "data": f"Data('alp1m_2016') {days[0].date()}~{days[-1].date()} {nd}일",
                        "costs": "cost_table_2016.json", "holm_m": M_HOLM},
           "checks": checks, "stage_days_by_year": st_runs,
           "K3": k3, "K3_sensitivity_no_filled": k3s,
           "K2_aux": k2, "K2_aux_sensitivity_no_filled": k2s,
           "always_soxs_all_days_mean": base,
           "prior_2023_2026_stage5_tableA_SOXS_std_extreme_up": prior,
           "files": ["rescue_k3.json", "rescue_k3_events.csv"]}
    json.dump(out, open(os.path.join(RES, "rescue_k3.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    log("끝")


if __name__ == "__main__":
    main()
