"""BRIEF6 — 추세 5단계 빨강이 뜬 첫날 SOXS (구현 A, 2026-11-03).

정본: ../BRIEF6.md(결과 보기 전 고정). 단계 함수: stage5.zone_calc(BRIEF5에서 지표 이식과 100% 대조됨). 엔진: lib.py(고치지 않음).

결과 보기 전에 이 파일에서 정한 해석(첫 실행 전 고정):
- 단계: yfinance 일봉(auto_adjust=False Close = 분할 반영·배당 미반영)
  SOXX 2001-01-01~2026-10-29(실제 첫 봉 2001-07-13), SOXL 2010-01-01~2026-10-29(첫 봉 2010-03-11)에 stage5.zone_calc.
  내려받은 첫 130봉은 값 없음. 빨강 = 단계 ≤ −1(하방 −1·극하방 −2). D의 'D−k'는 yfinance 거래일 순서로 k칸 앞.
- 사건(D = yfinance 거래일, 쓰는 단계는 모두 값이 있어야 한다)
  R1: 단계(D−1) 빨강, 단계(D−2) 빨강 아님. R2: 단계(D−1) = −2, 단계(D−2) ≠ −2.
  R3: R1이면서 단계(D−6..D−2) 다섯 날 모두 빨강 아님. R1-L: SOXL 단계로 R1(매매는 똑같이 SOXS).
  나머지 빨강 날 = 단계(D−1)·(D−2) 모두 빨강. 빨강 아닌 날 = 단계(D−1) 빨강 아님.
- 1분봉(Data('alp1m_long') 2023-01-03~2026-10-28): 09:30 봉 마감 결정 → 09:31 봉 종가 × (1 + 반 호가 open5).
  O1 = Engine(scheme='std') 09:59 봉 종가 청산, O2 = Engine(scheme='w90') 10:59 봉 종가 청산(엔진 그대로, 늘 SOXS 전략).
  O3·O4 = Data 날 색인 d+1·d+5의 10:59 봉 종가 × (1 − 반 호가 late) ÷ 진입가 − 1(직접 계산, 같은 진입가).
  직접 계산을 h = 0(같은 날 10:59)으로 돌려 엔진 O2와 같은지 대조한다. 비용 2배 = 반 호가 × 2(엔진 cost_mult=2와 같은 규칙).
  청산 날이 자료 밖이면(2026-10 끝) 그 사건은 그 청산에서 뺀다(건수에 적음).
- 겹침: 해(年)를 넘어 시간순으로, 앞 사건 보유 중(새 사건 진입 날 d' ≤ 앞 청산 날 d+h)인 새 사건은 건너뛴다(O3 h=1, O4 h=5).
  O1·O2는 같은 날 청산이라 겹침이 없다.
- 무작위 날 대조: 같은 해 1분봉 날 중 그 청산 값이 있는 날 전부에서 사건 수 n만큼 비복원으로 뽑기 10,000회.
  검정마다 np.random.default_rng(20261103)을 새로 만든다. p = #{뽑은 평균 ≥ 관측} ÷ 10,000(BRIEF6 '비율'), 참고로 (1+#)/(1+10,000).
  Holm m = 4: 2026 R1 × O1~O4의 p. 판정(O2 주, 넷 모두 표시): (1) 평균 > 0 (2) Holm p ≤ 0.05 (3) 비용 2배 평균 > 0.
- 95% 구간: 사건 값 복원 추출 10,000회(seed 20261103)의 2.5·97.5 분위(stage5.boot_mean).
- 사건·달 빼기: 가장 좋은 사건 1건 뺀 평균, 사건 값 합이 가장 큰 달의 사건을 뺀 평균.
- 첫날 대 나머지(판정 밖): 모든 날 SOXS를 같은 청산으로 샀을 때(겹침 규칙 없음)의 날 값 평균을 범주별로.
- 긴 재현(판정 밖): yfinance SOXS 일봉(2010-03-11~, auto_adjust=False Open·Close, 분할 반영). 사건 D의 SOXS 시가로 산다.
  oc = D 종가 ÷ D 시가 − 1, o1 = D+1 시가 ÷ D 시가 − 1, o5 = D+5 시가 ÷ D 시가 − 1(비용 없음, 근사).
  겹침: 시가에 팔고 같은 시가에 다시 사는 것은 겹침이 아니다(새 사건 j < 앞 청산 j+h면 건너뜀).
  해마다 2011~2025와 합계. 무작위 날 p는 해마다 같은 해 SOXS 거래일에서 같은 수(비복원 10,000회, 변형·청산마다
  rng 하나를 2011→2025 순서로 씀), 합계 p는 해마다 뽑은 합을 더한 귀무(해 층화). 2023~2026도 같은 근사(1분봉 값과 나란히).
  일봉 근사 대조: 1분봉에서 같은 정의(09:30 봉 시가, 정규장 마지막 봉 종가, D+1·D+5 09:30 봉 시가)로 다시 재서 사건마다 차이.
실행: cd code && py redfirst.py
결과: ../results/redfirst.json, ../results/redfirst_events.csv
"""
import hashlib, json, os, sys, time

os.environ["SOS_UNLOCK"] = "frozen"          # BRIEF6 허용: 2023~2026 전체를 쓴다
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import Data, Engine, RTH0, idx, bucket  # noqa: E402
import stage5 as S5  # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
CACHE = os.path.normpath(os.path.join(HERE, "..", "cache"))
SEED, NR, NB = 20261103, 10000, 10000
YEARS_1M = (2023, 2024, 2025, 2026)
YEARS_LONG = tuple(range(2011, 2026))
VARS = ("R1", "R2", "R3", "R1-L")
OS_ = ("O1", "O2", "O3", "O4")
HOLD = {"O1": 0, "O2": 0, "O3": 1, "O4": 5}
DS_ = ("oc", "o1", "o5")
DHOLD = {"oc": 0, "o1": 1, "o5": 5}
I_ENT, I_X = idx("09:31"), idx("10:59")
NAMES = S5.NAMES
T0 = time.time()
fnum = S5.fnum


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- 자료
def yf_daily(sym, start):
    p = os.path.join(CACHE, f"{sym.lower()}_1d_yf_ohlc_{start.replace('-', '')}_20261029.csv")
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0, parse_dates=True)
    import yfinance as yf
    df = yf.download(sym, start=start, end="2026-10-30", auto_adjust=False, progress=False, actions=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[df["Close"].notna()]
    os.makedirs(CACHE, exist_ok=True)
    df.to_csv(p)
    return df


def flags(zn):
    """yfinance 날 색인 t마다 범주 불(bool) 배열."""
    n = len(zn)
    fin = np.isfinite(zn)
    red = fin & (zn <= -1)
    z = np.where(fin, zn, 99)
    f = {k: np.zeros(n, bool) for k in ("R1", "R2", "R3", "rest_red", "nonred")}
    for t in range(2, n):
        if not (fin[t - 1] and fin[t - 2]):
            continue
        f["R1"][t] = red[t - 1] and not red[t - 2]
        f["R2"][t] = z[t - 1] == -2 and z[t - 2] != -2
        f["rest_red"][t] = red[t - 1] and red[t - 2]
        if t >= 6 and f["R1"][t]:
            w = slice(t - 6, t - 1)          # D−6..D−2
            f["R3"][t] = bool(fin[w].all() and not red[w].any())
    f["nonred"] = np.r_[False, fin[:-1] & ~red[:-1]]
    return f


def overlap_keep(pos, h, nmax, strict):
    """pos: 시간순 사건 색인. h: 보유 날 수. strict=True면 새 진입 날 ≤ 앞 청산 날을 건너뜀(1분봉 10:59 청산),
    False면 새 진입 날 < 앞 청산 날만 건너뜀(시가 청산 → 같은 시가 재진입 허용). 반환: 상태 배열."""
    st, last = [], -10 ** 9
    for p in pos:
        if h > 0 and ((p <= last) if strict else (p < last)):
            st.append("overlap_skip")
            continue
        if p + h >= nmax:
            st.append("no_exit")
            continue
        st.append("kept")
        if h > 0:
            last = p + h
    return np.array(st, dtype=object)


# ---------------------------------------------------------------- 통계
def null_draws(pool, n, rng):
    """pool에서 n개 비복원 NR회 → (NR,) 평균, (NR,) 합."""
    pool = np.asarray(pool, float)
    if n == 0 or len(pool) < n:
        return None, None
    keys = rng.random((NR, len(pool)))
    pick = np.argpartition(keys, n - 1, axis=1)[:, :n] if n < len(pool) else np.tile(np.arange(len(pool)), (NR, 1))
    s = pool[pick].sum(1)
    return s / n, s


def ev_stats(v, months, pool, c2=None, rng_seed=SEED):
    v = np.asarray(v, float)
    n = len(v)
    out = {"n": int(n)}
    if n == 0:
        return out
    obs = v.mean()
    out.update({"mean": fnum(obs), "median": fnum(np.median(v)), "sum": fnum(v.sum()),
                "hit": fnum((v > 0).mean()), "min": fnum(v.min()), "max": fnum(v.max()),
                "sd": fnum(v.std(ddof=1)) if n > 1 else None,
                "ci95": S5.boot_mean(v) if n > 1 else None})
    nm, _ = null_draws(pool, n, np.random.default_rng(rng_seed))
    if nm is not None:
        k = int((nm >= obs - 1e-12).sum())
        out.update({"p_random": k / NR, "p_random_plus1": (1 + k) / (NR + 1), "null_mean": fnum(nm.mean()),
                    "null_q95": fnum(np.percentile(nm, 95)), "pool_days": int(len(pool)),
                    "pool_mean": fnum(np.mean(pool))})
    if c2 is not None:
        out["cost2x_mean"] = fnum(np.mean(c2))
    if n > 1:
        b = int(np.argmax(v))
        out["drop_best_event"] = {"dropped_value": fnum(v[b]), "mean": fnum(np.delete(v, b).mean()), "n": int(n - 1)}
        ms = pd.Series(v, index=months).groupby(level=0).sum()
        bm = ms.idxmax()
        keep = np.array([m != bm for m in months])
        out["drop_best_month"] = {"month": bm, "month_sum": fnum(ms[bm]), "n": int(keep.sum()),
                                  "mean": fnum(v[keep].mean()) if keep.any() else None}
        out["by_month_sum"] = {k: fnum(x) for k, x in ms.items()}
    return out


def holm(ps):
    return S5.holm(ps)


# ---------------------------------------------------------------- 본체
def main():
    os.makedirs(RES, exist_ok=True)
    checks = {}
    # 1. 일봉·단계
    X = yf_daily("SOXX", "2001-01-01")
    L = yf_daily("SOXL", "2010-01-01")
    S = yf_daily("SOXS", "2010-01-01")
    log(f"SOXX {X.index[0].date()}~{X.index[-1].date()} {len(X)}, SOXL {L.index[0].date()}~ {len(L)}, SOXS {S.index[0].date()}~ {len(S)}")
    znX, rX, zX = S5.zone_calc(X["Close"].to_numpy(float))
    znL, _, _ = S5.zone_calc(L["Close"].to_numpy(float))
    datesX = pd.DatetimeIndex(X.index)
    tX = {d: t for t, d in enumerate(datesX)}
    # 대조: BRIEF5 단계표(2021 시작)와 2022-12 이후 비교
    s5 = pd.read_csv(os.path.join(RES, "stage5_stages.csv"), parse_dates=["date"]).set_index("date")
    s5 = s5[s5.index >= "2022-12-01"]
    mine = pd.Series(znX, index=datesX).reindex(s5.index)
    ok = s5["stage"].notna() & mine.notna()
    checks["stage_vs_stage5_from2022_12"] = {
        "n": int(ok.sum()), "agree": fnum((s5["stage"][ok] == mine[ok]).mean()),
        "mismatch_dates": [str(d.date()) for d in s5.index[ok & (s5["stage"] != mine)]],
        "close_rel_diff_max_pct": fnum((np.abs(X["Close"].reindex(s5.index) / s5["close"] - 1) * 100).max())}
    log("단계 대조", checks["stage_vs_stage5_from2022_12"])
    # SOXL 날짜 = SOXX 날짜(2010-03-11~)
    dL = pd.DatetimeIndex(L.index)
    checks["soxl_dates_equal_soxx_dates_since_2010_03_11"] = bool((datesX[datesX >= dL[0]] == dL).all()) if len(datesX[datesX >= dL[0]]) == len(dL) else False
    znL_on_X = pd.Series(znL, index=dL).reindex(datesX).to_numpy(float)
    FX = flags(znX)
    FL = flags(znL_on_X)
    EV = {"R1": FX["R1"], "R2": FX["R2"], "R3": FX["R3"], "R1-L": FL["R1"]}
    CAT = {"first": FX["R1"], "rest_red": FX["rest_red"], "nonred": FX["nonred"]}
    stage_prev = np.r_[np.nan, znX[:-1]]
    stage_prev2 = np.r_[np.nan, np.nan, znX[:-2]]
    soxl_prev = np.r_[np.nan, znL_on_X[:-1]]
    soxl_prev2 = np.r_[np.nan, np.nan, znL_on_X[:-2]]

    # SOXS 일봉 분할 점검(밤사이 로그 틈 최대) — 분할 반영이면 큰 틈은 진짜 갭뿐
    so, sc = S["Open"].to_numpy(float), S["Close"].to_numpy(float)
    gap = np.abs(np.log(so[1:] / sc[:-1]))
    top = np.argsort(-gap)[:5]
    checks["soxs_yf_overnight_abs_log_gap_top5"] = {str(S.index[i + 1].date()): fnum(gap[i]) for i in top}
    checks["soxs_yf_open_nonpositive_or_nan"] = int((~(so > 0)).sum())
    datesS = pd.DatetimeIndex(S.index)
    jS = {d: j for j, d in enumerate(datesS)}
    nS = len(datesS)
    dly = {"oc": (sc / so - 1) * 100,
           "o1": np.r_[(so[1:] / so[:-1] - 1) * 100, np.nan],
           "o5": np.r_[(so[5:] / so[:-5] - 1) * 100, [np.nan] * 5]}

    # 2. 1분봉
    D = Data("alp1m_long")
    days = [pd.Timestamp(d) for d in D.days]
    nd = len(days)
    tD = np.array([tX.get(d, -1) for d in days])
    checks["data_days"] = {"first": str(days[0].date()), "last": str(days[-1].date()), "n": nd,
                           "missing_in_yf_soxx": int((tD < 0).sum()),
                           "consecutive_in_yf": bool((np.diff(tD) == 1).all()),
                           "yf_days_in_range_not_in_data": int(sum(1 for d in datesX if days[0] <= d <= days[-1] and d not in set(days)))}
    assert (tD >= 0).all() and (np.diff(tD) == 1).all()
    log("Data", checks["data_days"])
    C = D.A["SOXS"]["Close"]
    O = D.A["SOXS"]["Open"]
    allk = list(range(nd))
    arr, arr2, eng_chk = {}, {}, {}
    for o, sch in (("O1", "std"), ("O2", "w90")):
        for cm in (1.0, 2.0):
            E = Engine(D, scheme=sch, cost_mult=cm)
            daily, tr = E.run(lambda: S5.Always(-1), days=allk)
            assert len(daily) == nd and (daily["n"] == 1).all()
            (arr if cm == 1.0 else arr2)[o] = daily["ret"].values * 100
            eng_chk[f"{o}_c{int(cm)}"] = {"entry": sorted(tr["entry"].unique().tolist()), "exit": sorted(tr["exit"].unique().tolist()),
                                         "why": {k: int(x) for k, x in tr["why"].value_counts().items()}, "sym": sorted(tr["sym"].unique().tolist())}
    checks["engine"] = eng_chk

    def direct(h, cm):
        E = Engine(D, cost_mult=cm)
        hi, hx = E.hs("SOXS", I_ENT), E.hs("SOXS", I_X)
        ent = C[:, I_ENT] * (1 + hi)
        out = np.full(nd, np.nan)
        out[:nd - h] = (C[h:, I_X] * (1 - hx) / ent[:nd - h] - 1) * 100
        return out, (bucket(I_ENT), hi, bucket(I_X), hx)

    for cm in (1.0, 2.0):
        d0, info = direct(0, cm)
        ref = arr["O2"] if cm == 1.0 else arr2["O2"]
        checks[f"direct_h0_vs_engine_O2_c{int(cm)}_max_abs_diff"] = fnum(np.abs(d0 - ref).max())
        checks[f"direct_cost_buckets_c{int(cm)}"] = info
        for o in ("O3", "O4"):
            (arr if cm == 1.0 else arr2)[o] = direct(HOLD[o], cm)[0]
    assert checks["direct_h0_vs_engine_O2_c1_max_abs_diff"] < 1e-9
    log("엔진·직접 대조", {k: v for k, v in checks.items() if k.startswith("direct_h0")})
    # 1분봉 SOXX 시가→11:00, 1분봉으로 다시 잰 일봉 근사(대조용)
    XC, XO = D.A["SOXX"]["Close"], D.A["SOXX"]["Open"]
    soxx_o_1100 = (XC[:, I_X] / XO[:, RTH0] - 1) * 100
    rth_close = np.array([C[k, e - 1] for k, e in enumerate(D.rth_end)])
    o930 = O[:, RTH0]
    m1 = {"oc": (rth_close / o930 - 1) * 100,
          "o1": np.r_[(o930[1:] / o930[:-1] - 1) * 100, np.nan],
          "o5": np.r_[(o930[5:] / o930[:-5] - 1) * 100, [np.nan] * 5]}
    yrD = np.array([d.year for d in days])
    monD = np.array([f"{d.year}-{d.month:02d}" for d in days])

    # 3. 사건 표(모든 해·변형)
    rows = []
    status_1m, status_d = {}, {}
    for v in VARS:
        tpos = np.flatnonzero(EV[v])
        # 1분봉 겹침(Data 색인)
        kD = np.array([k for k in range(nd) if EV[v][tD[k]]], int)
        st1 = {o: overlap_keep(kD, HOLD[o], nd, strict=True) for o in OS_}
        status_1m[v] = (kD, st1)
        # 일봉 겹침(SOXS 색인)
        jv = np.array([jS.get(datesX[t], -1) for t in tpos])
        has = jv >= 0
        std_ = {ds: np.full(len(tpos), "no_soxs", dtype=object) for ds in DS_}
        for ds in DS_:
            std_[ds][has] = overlap_keep(jv[has], DHOLD[ds], nS, strict=False)
            if ds == "oc":
                std_[ds][has] = np.where(np.isfinite(dly["oc"][jv[has]]), "kept", "no_exit")
        status_d[v] = (tpos, jv, std_)
        kmap = {k: q for q, k in enumerate(kD)}
        for q, t in enumerate(tpos):
            d = datesX[t]
            r = {"variant": v, "date": str(d.date()), "year": d.year, "month": f"{d.year}-{d.month:02d}",
                 "soxx_stage_d1": stage_prev[t], "soxx_stage_d2": stage_prev2[t],
                 "soxl_stage_d1": soxl_prev[t], "soxl_stage_d2": soxl_prev2[t],
                 "soxx_yf_oc_pct": fnum((X["Close"].iloc[t] / X["Open"].iloc[t] - 1) * 100)}
            k = int(np.searchsorted(tD, t)) if (tD[0] <= t <= tD[-1]) else -1
            in1 = k >= 0 and tD[k] == t
            r["in_1m"] = bool(in1)
            for o in OS_:
                if in1:
                    sst = st1[o][kmap[k]]
                    r[f"{o}_status"] = sst
                    r[o] = fnum(arr[o][k]) if sst != "no_exit" else None
                    r[f"{o}_c2"] = fnum(arr2[o][k]) if sst != "no_exit" else None
                else:
                    r[f"{o}_status"], r[o], r[f"{o}_c2"] = "", None, None
            r["soxx_1m_open_1100_pct"] = fnum(soxx_o_1100[k]) if in1 else None
            j = jv[q]
            for ds in DS_:
                r[f"d_{ds}_status"] = std_[ds][q]
                r[f"d_{ds}"] = fnum(dly[ds][j]) if j >= 0 and np.isfinite(dly[ds][j]) else None
                r[f"m1_{ds}"] = fnum(m1[ds][k]) if in1 and np.isfinite(m1[ds][k]) else None
            rows.append(r)
    evdf = pd.DataFrame(rows)
    evdf.to_csv(os.path.join(RES, "redfirst_events.csv"), index=False, encoding="utf-8-sig")
    # 일봉 근사 대조(1분봉 같은 정의)
    cmp = {}
    e1 = evdf[evdf["in_1m"]]
    for ds in DS_:
        a, b = e1[f"d_{ds}"].astype(float), e1[f"m1_{ds}"].astype(float)
        okk = a.notna() & b.notna()
        cmp[ds] = {"n": int(okk.sum()), "abs_diff_median_pctp": fnum((a - b).abs()[okk].median()),
                   "abs_diff_max_pctp": fnum((a - b).abs()[okk].max()), "corr": fnum(np.corrcoef(a[okk], b[okk])[0, 1]) if okk.sum() > 2 else None}
    checks["daily_approx_yf_vs_1m_same_def"] = cmp
    log("일봉 근사 대조", cmp)

    # 4. 사건 수
    yrX = np.array([d.year for d in datesX])
    counts = {v: {str(y): int(EV[v][yrX == y].sum()) for y in range(2002, 2027)} for v in VARS}
    counts_1m = {v: {str(y): int(sum(1 for k in status_1m[v][0] if yrD[k] == y)) for y in YEARS_1M} for v in VARS}

    # 5. 1분봉 통계(변형 × 해 × 청산)
    def stats_1m(v, y, o):
        kD, st1 = status_1m[v]
        sel = np.array([(yrD[k] == y) and (s == "kept") for k, s in zip(kD, st1[o])], bool)
        ks = kD[sel]
        pool = arr[o][(yrD == y) & np.isfinite(arr[o])]
        res = ev_stats(arr[o][ks], list(monD[ks]), pool, c2=arr2[o][ks])
        res["skipped_overlap"] = int(sum(1 for k, s in zip(kD, st1[o]) if yrD[k] == y and s == "overlap_skip"))
        res["no_exit"] = int(sum(1 for k, s in zip(kD, st1[o]) if yrD[k] == y and s == "no_exit"))
        res["dates"] = [str(days[k].date()) for k in ks]
        return res

    S1 = {v: {str(y): {o: stats_1m(v, y, o) for o in OS_} for y in YEARS_1M} for v in VARS}
    log("1분봉 통계 끝")

    # 2026 R1 판정
    p26 = {o: S1["R1"]["2026"][o].get("p_random") for o in OS_}
    ph = holm({o: p for o, p in p26.items() if p is not None})
    judge = {}
    for o in OS_:
        s = S1["R1"]["2026"][o]
        crit = {"(1) mean>0": bool(s.get("mean") is not None and s["mean"] > 0),
                "(2) holm_p<=0.05": bool(ph.get(o, 1.0) <= 0.05),
                "(3) cost2x>0": bool(s.get("cost2x_mean") is not None and s["cost2x_mean"] > 0)}
        judge[o] = {"p_raw": p26[o], "p_holm_m4": fnum(ph.get(o)), "criteria": crit, "pass": all(crit.values())}
    # 겹침 없이 모든 R1 날로 잰 O4(민감도)
    kD, _ = status_1m["R1"]
    k26 = np.array([k for k in kD if yrD[k] == 2026], int)
    sens = {o: {"n": int(np.isfinite(arr[o][k26]).sum()), "mean": fnum(np.nanmean(arr[o][k26]))} for o in ("O3", "O4")}
    # 설명용(사후, 판정 밖): 1~6월·7~10월, R1 중 R3 아닌 사건(D−6..D−3에 빨강이 있던 '다시 빨강')
    kD1, st1R1 = status_1m["R1"]
    r3set = set(int(k) for k in status_1m["R3"][0])
    posthoc = {}
    for o in OS_:
        kk = [int(k) for k, s in zip(kD1, st1R1[o]) if yrD[k] == 2026 and s == "kept"]
        h1 = [arr[o][k] for k in kk if days[k].month <= 6]
        h2 = [arr[o][k] for k in kk if days[k].month >= 7]
        nr3 = [arr[o][k] for k in kk if k not in r3set]
        posthoc[o] = {"jan_jun": {"n": len(h1), "mean": fnum(np.mean(h1)) if h1 else None},
                      "jul_oct": {"n": len(h2), "mean": fnum(np.mean(h2)) if h2 else None},
                      "r1_not_r3": {"n": len(nr3), "mean": fnum(np.mean(nr3)) if nr3 else None}}
    primary = {"stats": S1["R1"]["2026"], "judgement": judge, "primary_exit": "O2",
               "pass_primary": judge["O2"]["pass"], "sensitivity_no_overlap_rule": sens,
               "posthoc_description_only": posthoc}
    log("2026 R1", {o: (S1["R1"]["2026"][o].get("n"), S1["R1"]["2026"][o].get("mean"), judge[o]) for o in OS_})

    # 2026 R1 사건 목록
    lst = []
    for k in k26:
        t = tD[k]
        lst.append({"date": str(days[k].date()), "stage_d1": NAMES[int(stage_prev[t])], "stage_d2": NAMES[int(stage_prev2[t])],
                    "soxx_open_to_1100_pct": fnum(soxx_o_1100[k]), "soxx_open_to_close_pct": fnum((X["Close"].iloc[t] / X["Open"].iloc[t] - 1) * 100),
                    **{f"SOXS_{o}": fnum(arr[o][k]) for o in OS_},
                    **{f"{o}_status": str(status_1m["R1"][1][o][list(kD).index(k)]) for o in ("O3", "O4")}})

    # 6. 첫날 대 나머지(1분봉, 모든 날, 겹침 없음)
    fvr = {}
    for y in list(YEARS_1M) + ["all"]:
        ym = np.ones(nd, bool) if y == "all" else (yrD == y)
        res = {}
        for o in OS_:
            a = arr[o]
            cell = {}
            for cn, f in CAT.items():
                m = ym & f[tD] & np.isfinite(a)
                cell[cn] = {"days": int(m.sum()), "mean": fnum(a[m].mean()) if m.any() else None}
            m_all = ym & np.isfinite(a)
            cell["all_days"] = {"days": int(m_all.sum()), "mean": fnum(a[m_all].mean())}
            f_, r_, n_ = (cell[c]["mean"] for c in ("first", "rest_red", "nonred"))
            cell["first_minus_rest_red"] = fnum(f_ - r_) if f_ is not None and r_ is not None else None
            cell["first_minus_nonred"] = fnum(f_ - n_) if f_ is not None and n_ is not None else None
            res[o] = cell
        fvr[str(y)] = res

    # 7. 긴 재현(일봉 근사)
    yrS = np.array([d.year for d in datesS])
    monS = np.array([f"{d.year}-{d.month:02d}" for d in datesS])

    def long_block(v, ds, years):
        tpos, jv, std_ = status_d[v]
        rng = np.random.default_rng(SEED)
        per, tot_v, null_sum, null_n = {}, [], np.zeros(NR), 0
        a = dly[ds]
        for y in years:
            sel = np.array([(j >= 0) and (yrS[j] == y) and (s == "kept") for j, s in zip(jv, std_[ds])], bool)
            js = jv[sel]
            vals = a[js]
            pool = a[(yrS == y) & np.isfinite(a)]
            n = len(vals)
            cell = {"n": int(n), "mean": fnum(vals.mean()) if n else None, "sum": fnum(vals.sum()) if n else 0.0,
                    "sign": (int(np.sign(vals.mean())) if n else None),
                    "skipped_overlap": int(sum(1 for j, s in zip(jv, std_[ds]) if j >= 0 and yrS[j] == y and s == "overlap_skip"))}
            if n:
                nm, ns = null_draws(pool, n, rng)
                k_ = int((nm >= vals.mean() - 1e-12).sum())
                cell.update({"p_random": k_ / NR, "null_mean": fnum(nm.mean())})
                null_sum += ns
                null_n += n
            per[str(y)] = cell
            tot_v += list(vals)
        tv = np.array(tot_v)
        tot = {"n": int(len(tv)), "mean": fnum(tv.mean()) if len(tv) else None, "sum": fnum(tv.sum()) if len(tv) else None,
               "years_positive": int(sum(1 for c in per.values() if c["n"] and c["mean"] > 0)),
               "years_with_events": int(sum(1 for c in per.values() if c["n"]))}
        if null_n:
            nmt = null_sum / null_n
            k_ = int((nmt >= tv.mean() - 1e-12).sum())
            tot.update({"p_random_year_stratified": k_ / NR, "null_mean": fnum(nmt.mean()), "ci95": S5.boot_mean(tv)})
        return {"by_year": per, "total": tot}

    LONG = {v: {ds: long_block(v, ds, YEARS_LONG) for ds in DS_} for v in VARS}
    LONG_1M_YEARS = {v: {ds: long_block(v, ds, YEARS_1M) for ds in DS_} for v in VARS}
    # 긴 재현의 첫날 대 나머지(2011~2025 모은 값, 겹침 없음, 일봉 근사)
    fvr_long = {}
    tS_of_X = np.array([jS.get(d, -1) for d in datesX])
    for ds in DS_:
        a = dly[ds]
        cell = {}
        for cn, f in CAT.items():
            ts = np.flatnonzero(f & (tS_of_X >= 0))
            js = tS_of_X[ts]
            js = js[(yrS[js] >= 2011) & (yrS[js] <= 2025)]
            vv = a[js][np.isfinite(a[js])]
            cell[cn] = {"days": int(len(vv)), "mean": fnum(vv.mean()) if len(vv) else None}
        cell["first_minus_rest_red"] = fnum(cell["first"]["mean"] - cell["rest_red"]["mean"])
        cell["first_minus_nonred"] = fnum(cell["first"]["mean"] - cell["nonred"]["mean"])
        fvr_long[ds] = cell
    log("긴 재현 끝", {ds: LONG["R1"][ds]["total"] for ds in DS_})

    # 8. 1분봉 대 일봉 근사(2023~2026, R1 나란히)
    side = {}
    for v in VARS:
        side[v] = {}
        for y in YEARS_1M:
            side[v][str(y)] = {**{o: {"n": S1[v][str(y)][o]["n"], "mean": S1[v][str(y)][o].get("mean")} for o in OS_},
                               **{f"daily_{ds}": {"n": LONG_1M_YEARS[v][ds]["by_year"][str(y)]["n"],
                                                  "mean": LONG_1M_YEARS[v][ds]["by_year"][str(y)]["mean"]} for ds in DS_}}

    years_1m = {str(y): {o: {kk: S1["R1"][str(y)][o].get(kk) for kk in
                             ("n", "mean", "ci95", "hit", "p_random", "cost2x_mean", "skipped_overlap", "no_exit")}
                         for o in OS_} for y in YEARS_1M}
    variants = {v: {"years_1m": {str(y): {o: {kk: S1[v][str(y)][o].get(kk) for kk in
                                               ("n", "mean", "ci95", "hit", "p_random", "cost2x_mean", "skipped_overlap", "no_exit",
                                                "drop_best_event", "drop_best_month")}
                                           for o in OS_} for y in YEARS_1M},
                    "long_daily": LONG[v], "daily_2023_2026": LONG_1M_YEARS[v]} for v in ("R2", "R3", "R1-L")}

    code_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
    out = {"brief": "BRIEF6.md", "impl": "A (redfirst.py)", "code_sha256_16": code_sha, "units": "% (1.0 = 1%)",
           "settings": {"seed": SEED, "random_draws": NR, "boot": NB, "entry": "D 09:31 봉 종가 + 반 호가(open5)",
                        "exits": {"O1": "D 09:59 봉 종가 − 반 호가(Engine std)", "O2": "D 10:59(Engine w90)",
                                  "O3": "D+1 10:59 − 반 호가(late)", "O4": "D+5 10:59 − 반 호가(late)"},
                        "soxx": f"yfinance SOXX {X.index[0].date()}~{X.index[-1].date()} auto_adjust=False Close",
                        "soxl": f"yfinance SOXL {L.index[0].date()}~{L.index[-1].date()}",
                        "soxs": f"yfinance SOXS {S.index[0].date()}~{S.index[-1].date()} Open·Close(분할 반영)",
                        "data": f"Data('alp1m_long') {days[0].date()}~{days[-1].date()} {nd}일"},
           "checks": checks,
           "events": {"counts_by_year_daily_calendar": counts, "counts_by_year_1m": counts_1m, "r1_2026": lst},
           "primary_2026": primary, "first_vs_rest": {"1m_all_days": fvr, "daily_2011_2025": fvr_long},
           "years_1m": years_1m, "stats_1m_all": S1,
           "long_daily": {"R1": LONG["R1"], "R1_daily_2023_2026": LONG_1M_YEARS["R1"]},
           "side_by_side_1m_vs_daily": side, "variants": variants,
           "files": ["redfirst.json", "redfirst_events.csv"]}
    json.dump(out, open(os.path.join(RES, "redfirst.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    log("끝")


if __name__ == "__main__":
    main()
