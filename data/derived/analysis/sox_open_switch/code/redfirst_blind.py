"""BRIEF6 눈가림 재구현(구현 B): 추세 5단계 빨강 첫날 SOXS.

redfirst.py·lib.py·stage5.py를 쓰지 않는다. 단계는 ../trend_zones/trend_5zone.pine을 보고 직접 짠다.
- 단계: SOXX 정규장 일봉 종가(yfinance, auto_adjust=False = 분할만 반영), Wilder RSI14(ta.rsi), 20일 표본 sd(ta.stdev biased=false),
  k = sqrt((N-1)(2N-1)/(6N)), z = (ln C - sma(ln C, 20)) / (sd*k), bar_index < 129 → 없음.
  RSI ≥ 70 → 2, RSI ≤ 30 → -2, z ≥ 0.5 → 1, z ≤ -0.5 → -1, 그 밖 0.
- 사건 R1: 단계(D-1) ∈ {-1,-2}, 단계(D-2) ∉ {-1,-2}. R2: 단계(D-1) = -2, 단계(D-2) ≠ -2. R3: R1 중 D-6~D-2 모두 빨강 아님. R1-L: SOXL 일봉 단계로 R1.
- 1분봉(Alpaca alp1m_long SOXS, 봉 시각 = 시작 시각): 매수 09:31 봉 종가 × (1 + 반호가 open5),
  O1 D 09:59 봉, O2 D 10:59 봉, O3 D+1 10:59 봉, O4 D+5 10:59 봉 종가 × (1 - 반호가). 청산 반호가 시간대는 호가 시각(봉+1분) 기준:
  09:59 봉 → 10:00 → early, 10:59 봉 → 11:00 → late (build_costs.py 구간 정의). 봉이 없으면 그날 09:30 이후 마지막 봉.
  O3·O4는 보유 중(매수 시각 < 앞 청산 시각) 새 사건을 건너뛴다.
- 무작위 날 대조: 같은 해 아무 거래일에서 같은 수의 날을 비복원 10,000번(seed 20261103), 평균 ≥ 관측 비율. Holm m=4(O1~O4).
- 긴 재현: yfinance 일봉 SOXS 시가 → D 종가·D+1 시가·D+5 시가(비용 없음 = 총수익, 비용 반영 칸 따로).
"""
import json
import os

import numpy as np
import pandas as pd
import yfinance as yf

os.environ.setdefault("SOS_UNLOCK", "frozen")  # lib 잠금(이 스크립트는 lib를 쓰지 않음)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SEED = 20261103
NDRAW = 10000
RED = (-1, -2)


# ── 단계 계산(pine 직역) ──────────────────────────────────
def wilder_rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    c = np.asarray(close, float)
    out = np.full(len(c), np.nan)
    ch = np.diff(c)
    up = np.maximum(ch, 0.0)
    dn = np.maximum(-ch, 0.0)
    if len(ch) < n:
        return out
    a = 1.0 / n
    ru = up[:n].mean()
    rd = dn[:n].mean()
    for i in range(n, len(ch) + 1):  # ch 인덱스 i-1 → close 인덱스 i
        if i > n:
            ru = a * up[i - 1] + (1 - a) * ru
            rd = a * dn[i - 1] + (1 - a) * rd
        if rd == 0:
            out[i] = 100.0
        elif ru == 0:
            out[i] = 0.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + ru / rd)
    return out


def stage_series(close: pd.Series, N: int = 20, th: float = 0.5) -> pd.Series:
    c = close.astype(float)
    lc = np.log(c)
    lr = lc.diff()
    sd = lr.rolling(N).std(ddof=1)
    k = np.sqrt((N - 1.0) * (2.0 * N - 1.0) / (6.0 * N))
    z = (lc - lc.rolling(N).mean()) / (sd * k)
    r = pd.Series(wilder_rsi(c.values, 14), index=c.index)
    zn = pd.Series(np.nan, index=c.index)
    ok = z.notna() & r.notna() & (np.arange(len(c)) >= 129)
    v = np.where(r >= 70, 2, np.where(r <= 30, -2, np.where(z >= th, 1, np.where(z <= -th, -1, 0))))
    zn[ok] = v[ok.values]
    return zn


def yf_daily(sym, start):
    d = yf.download(sym, start=start, auto_adjust=False, progress=False, multi_level_index=False)
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d[["Open", "Close"]].dropna()


soxx_d = yf_daily("SOXX", "2001-01-01")
soxs_d = yf_daily("SOXS", "2010-01-01")
soxl_d = yf_daily("SOXL", "2010-01-01")

stg = stage_series(soxx_d.Close)
stg_l = stage_series(soxl_d.Close)

cal = soxx_d.index  # SOXX 거래일 달력


def events_from(stage: pd.Series, calendar: pd.DatetimeIndex):
    s = stage.reindex(calendar)
    s1 = s.shift(1)  # 단계(D-1)
    s2 = s.shift(2)
    red1 = s1.isin(RED)
    red2 = s2.isin(RED)
    R1 = red1 & ~red2 & s2.notna()
    R2 = (s1 == -2) & (s2 != -2) & s2.notna()
    prior_ok = pd.Series(True, index=calendar)
    for j in range(2, 7):
        sj = s.shift(j)
        prior_ok &= sj.notna() & ~sj.isin(RED)
    R3 = R1 & prior_ok
    second = red1 & red2           # 빨강 둘째 날 이후
    nonred = s1.notna() & ~red1    # 빨강 아닌 날
    return dict(R1=R1, R2=R2, R3=R3, second=second, nonred=nonred, s1=s1, s2=s2)


ev = events_from(stg, cal)
# R1-L: SOXL 차트 단계, 달력은 SOXL 거래일(SOXX와 같음)을 SOXX 달력으로 맞춘다
ev_l = events_from(stg_l.reindex(cal), cal)
ev["R1L"] = ev_l["R1"]

# ── 1분봉 SOXS ───────────────────────────────────────────
cost = json.load(open(os.path.join(ROOT, "cost_table.json")))["SOXS"]
m1 = pd.read_parquet(os.path.join(ROOT, "cache/alp1m_long/SOXS.parquet"))
idx = m1.index
mins = idx.hour * 60 + idx.minute
m1 = m1.assign(date=idx.tz_localize(None).normalize(), m=mins)
m1 = m1[(m1.m >= 570) & (m1.m < 960)]
P = m1.pivot_table(index="date", columns="m", values="Close", aggfunc="last").sort_index()
P = P.reindex(columns=range(570, 960)).ffill(axis=1)  # 봉이 없으면 그날 09:30 이후 마지막 봉
days1 = P.index
mday = pd.Index(days1)

# SOXX 1분봉(사건 목록의 그날 시가→11시)
x1 = pd.read_parquet(os.path.join(ROOT, "cache/alp1m_long/SOXX.parquet"))
xi = x1.index
x1 = x1.assign(date=xi.tz_localize(None).normalize(), m=xi.hour * 60 + xi.minute)
x1 = x1[(x1.m >= 570) & (x1.m < 960)]
XO = x1[x1.m == 570].groupby("date").Open.first()
XC = x1.pivot_table(index="date", columns="m", values="Close", aggfunc="last").reindex(columns=range(570, 960)).ffill(axis=1)

ENTRY_M = 571
EXIT = {"O1": (0, 599, "early"), "O2": (0, 659, "late"), "O3": (1, 659, "late"), "O4": (5, 659, "late")}
hs_in = cost["open5"]


def ret_1m(d, o, cmult=1.0):
    """D일 사건의 SOXS 수익(%)과 청산 날짜. 없으면 None."""
    if d not in mday:
        return None, None
    i = mday.get_loc(d)
    off, mx, b = EXIT[o]
    if i + off >= len(mday):
        return None, None
    e = P.iat[i, ENTRY_M - 570]
    x = P.iat[i + off, mx - 570]
    if not (np.isfinite(e) and np.isfinite(x)):
        return None, None
    r = x * (1 - cmult * cost[b]) / (e * (1 + cmult * hs_in)) - 1
    return 100 * r, mday[i + off]


# 모든 날의 결과(무작위 대조·첫날 대 나머지용)
ALL = {}
for o in EXIT:
    vals = {}
    for d in mday:
        r, _ = ret_1m(d, o)
        if r is not None:
            vals[d] = r
    ALL[o] = pd.Series(vals)
ALL2 = {}
for o in EXIT:
    vals = {}
    for d in mday:
        r, _ = ret_1m(d, o, 2.0)
        if r is not None:
            vals[d] = r
    ALL2[o] = pd.Series(vals)


def take_events(mask: pd.Series, o: str, cmult=1.0, lo=None, hi=None):
    """사건 날짜 목록 → (날짜, 수익). O3·O4는 보유 중 새 사건 건너뜀."""
    dates = [d for d in mask.index[mask.values] if d in mday]
    if lo is not None:
        dates = [d for d in dates if d >= lo]
    if hi is not None:
        dates = [d for d in dates if d <= hi]
    out = []
    held_until = None  # (청산 날짜) — 청산 10:59, 새 매수 09:31: 같은 날이면 보유 중
    for d in dates:
        if o in ("O3", "O4") and held_until is not None and d <= held_until:
            continue
        r, xd = ret_1m(d, o, cmult)
        if r is None:
            continue
        out.append((d, r))
        held_until = xd
    return out


def holm(ps):
    keys = list(ps)
    order = sorted(keys, key=lambda k: ps[k])
    m = len(keys)
    adj = {}
    run = 0.0
    for j, k in enumerate(order):
        v = min(1.0, (m - j) * ps[k])
        run = max(run, v)
        adj[k] = run
    return adj


def rand_p(pool: pd.Series, n: int, obs: float, seed=SEED):
    rng = np.random.default_rng(seed)
    a = pool.values
    if n == 0 or len(a) < n:
        return None
    means = np.empty(NDRAW)
    for t in range(NDRAW):
        means[t] = a[rng.choice(len(a), n, replace=False)].mean()
    return float((means >= obs).mean())


def boot_ci(x, seed=SEED):
    x = np.asarray(x)
    if len(x) < 2:
        return None
    rng = np.random.default_rng(seed)
    bm = x[rng.integers(0, len(x), (NDRAW, len(x)))].mean(1)
    return [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def t_ci(x):
    from scipy import stats
    x = np.asarray(x)
    if len(x) < 2:
        return None
    se = x.std(ddof=1) / np.sqrt(len(x))
    q = stats.t.ppf(0.975, len(x) - 1)
    return [float(x.mean() - q * se), float(x.mean() + q * se)]


def year_bounds(y):
    return pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")


def summ(mask, y, with_p=True):
    lo, hi = year_bounds(y)
    res = {}
    ps = {}
    for o in EXIT:
        evs = take_events(mask, o, 1.0, lo, hi)
        evs2 = take_events(mask, o, 2.0, lo, hi)
        r = np.array([v for _, v in evs])
        r2 = np.array([v for _, v in evs2])
        row = {"n": int(len(r)), "mean": float(r.mean()) if len(r) else None,
               "median": float(np.median(r)) if len(r) else None,
               "win": float((r > 0).mean()) if len(r) else None,
               "mean_cost2x": float(r2.mean()) if len(r2) else None}
        if with_p and len(r):
            pool = ALL[o][(ALL[o].index >= lo) & (ALL[o].index <= hi)]
            row["p_random"] = rand_p(pool, len(r), r.mean())
            row["pool_n"] = int(len(pool))
            row["pool_mean"] = float(pool.mean())
            ps[o] = row["p_random"]
            row["ci95_boot"] = boot_ci(r)
            row["ci95_t"] = t_ci(r)
            # 가장 좋은 사건·가장 좋은 달 빼기
            if len(r) >= 2:
                s = pd.Series(r, index=[d for d, _ in evs])
                row["ex_best_event"] = float(s.drop(s.idxmax()).mean())
                mon = s.groupby(s.index.to_period("M")).sum()
                bm = mon.idxmax()
                rest = s[s.index.to_period("M") != bm]
                row["best_month"] = str(bm)
                row["ex_best_month"] = float(rest.mean()) if len(rest) else None
                row["ex_best_month_n"] = int(len(rest))
        res[o] = row
    if with_p and ps:
        h = holm(ps)
        for o in ps:
            res[o]["p_holm"] = h[o]
    return res


# ── 사건 수 ─────────────────────────────────────────────
counts = {}
for y in range(2011, 2027):
    lo, hi = year_bounds(y)
    if y == 2026:
        hi = pd.Timestamp("2026-10-28")
    c = {}
    for nm in ["R1", "R2", "R3", "R1L"]:
        msk = ev[nm]
        c[nm] = int(msk[(msk.index >= lo) & (msk.index <= hi)].sum())
    counts[str(y)] = c

# 2026 R1 사건 목록
lst = []
lo26, hi26 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-28")
R1d = [d for d in ev["R1"].index[ev["R1"].values] if lo26 <= d <= hi26]
o3_taken = {d for d, _ in take_events(ev["R1"], "O3", 1.0, lo26, hi26)}
o4_taken = {d for d, _ in take_events(ev["R1"], "O4", 1.0, lo26, hi26)}
for d in R1d:
    row = {"date": str(d.date()), "stage_D-1": int(ev["s1"][d]), "stage_D-2": int(ev["s2"][d])}
    for o in EXIT:
        r, _ = ret_1m(d, o)
        row[o] = None if r is None else round(r, 3)
    row["O3_counted"] = d in o3_taken
    row["O4_counted"] = d in o4_taken
    if d in XO.index and d in XC.index:
        row["SOXX_open_to_1100"] = round(100 * (XC.loc[d, 659] / XO[d] - 1), 3)
    if d in soxx_d.index:
        row["SOXX_open_to_close_daily"] = round(100 * (soxx_d.Close[d] / soxx_d.Open[d] - 1), 3)
    lst.append(row)

# ── 주 판정 2026 ─────────────────────────────────────────
prim = summ(ev["R1"], 2026)
verdict_rows = {}
for o in EXIT:
    r = prim[o]
    verdict_rows[o] = {
        "mean_gt0": bool(r["mean"] is not None and r["mean"] > 0),
        "p_holm_le_0.05": bool(r.get("p_holm") is not None and r["p_holm"] <= 0.05),
        "cost2x_gt0": bool(r["mean_cost2x"] is not None and r["mean_cost2x"] > 0),
    }
O2_pass = all(verdict_rows["O2"].values())

# ── 해마다 1분봉 ─────────────────────────────────────────
years_1m = {str(y): summ(ev["R1"], y) for y in (2023, 2024, 2025, 2026)}
# 2023~2025 O2 합계 평균
o2_2325 = []
for y in (2023, 2024, 2025):
    lo, hi = year_bounds(y)
    o2_2325 += [v for _, v in take_events(ev["R1"], "O2", 1.0, lo, hi)]
years_1m["2023-2025_O2_pooled"] = {"n": len(o2_2325), "mean": float(np.mean(o2_2325)) if o2_2325 else None}

# ── 변형 ────────────────────────────────────────────────
variants = {nm: {"2026": summ(ev[nm], 2026),
                 "2023-2025_O2": None} for nm in ["R2", "R3", "R1L"]}
for nm in ["R2", "R3", "R1L"]:
    v = []
    for y in (2023, 2024, 2025):
        lo, hi = year_bounds(y)
        v += [x for _, x in take_events(ev[nm], "O2", 1.0, lo, hi)]
    variants[nm]["2023-2025_O2"] = {"n": len(v), "mean": float(np.mean(v)) if v else None}


# ── 첫날 대 나머지 ───────────────────────────────────────
def mean_on(mask, o, lo, hi):
    s = ALL[o]
    s = s[(s.index >= lo) & (s.index <= hi)]
    m = mask.reindex(s.index).fillna(False).astype(bool)
    x = s[m]
    return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None}


fvr = {}
for tag, (lo, hi) in {"2026": (lo26, hi26),
                      "2023-2025": (pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31"))}.items():
    t = {}
    for o in EXIT:
        first = [v for _, v in take_events(ev["R1"], o, 1.0, lo, hi)]
        first_all = mean_on(ev["R1"], o, lo, hi)  # 건너뜀 없이
        sec = mean_on(ev["second"], o, lo, hi)
        nr = mean_on(ev["nonred"], o, lo, hi)
        fm = float(np.mean(first)) if first else None
        t[o] = {"first_n": len(first), "first_mean": fm,
                "first_all_days_mean": first_all["mean"],
                "red_second_plus": sec, "non_red": nr,
                "first_minus_red_rest": (fm - sec["mean"]) if (fm is not None and sec["mean"] is not None) else None}
    fvr[tag] = t

# ── 긴 재현: 일봉 시가 근사 ────────────────────────────────
dc = soxs_d.reindex(cal)
dO, dC = dc.Open, dc.Close
H = {"oc": 0, "o1": 1, "o5": 5}
calidx = pd.Index(cal)


def daily_ret(d, h, net=False):
    i = calidx.get_loc(d)
    off = H[h]
    if i + off >= len(calidx):
        return None, None
    e = dO.iat[i]
    x = dC.iat[i] if h == "oc" else dO.iat[i + off]
    if not (np.isfinite(e) and np.isfinite(x)) or e <= 0:
        return None, None
    if net:
        r = x * (1 - (cost["late"] if h == "oc" else cost["open5"])) / (e * (1 + cost["open5"])) - 1
    else:
        r = x / e - 1
    return 100 * r, i + off


DALL = {}
for h in H:
    vals = {}
    for d in cal:
        if d < pd.Timestamp("2011-01-01") or d > hi26:
            continue
        r, _ = daily_ret(d, h)
        if r is not None:
            vals[d] = r
    DALL[h] = pd.Series(vals)


def daily_events(mask, h, lo, hi, net=False, skip=True):
    dates = [d for d in mask.index[mask.values] if lo <= d <= hi]
    out = []
    held = None
    for d in dates:
        i = calidx.get_loc(d)
        if skip and h == "o5" and held is not None and i < held:  # 시가 청산과 같은 시가의 새 매수는 겹치지 않음
            continue
        r, xi_ = daily_ret(d, h, net)
        if r is None:
            continue
        out.append((d, r))
        held = xi_
    return out


def daily_block(mask, years):
    per = {}
    pooled = {h: [] for h in H}
    pooled_net = {h: [] for h in H}
    for y in years:
        lo, hi = year_bounds(y)
        if y == 2026:
            hi = hi26
        row = {}
        for h in H:
            e = daily_events(mask, h, lo, hi)
            en = daily_events(mask, h, lo, hi, net=True)
            r = [v for _, v in e]
            pooled[h] += e
            pooled_net[h] += en
            row[h] = {"n": len(r), "mean": float(np.mean(r)) if r else None,
                      "sign": (None if not r else ("+" if np.mean(r) > 0 else "-"))}
        per[str(y)] = row
    return per, pooled, pooled_net


def strat_p(pooled_events, h, seed=SEED):
    """해마다 같은 수를 그 해에서 비복원으로 뽑아 합친 평균 ≥ 관측 비율."""
    if not pooled_events:
        return None
    obs = np.mean([v for _, v in pooled_events])
    by_y = pd.Series([d.year for d, _ in pooled_events]).value_counts().to_dict()
    pools = {y: DALL[h][DALL[h].index.year == y].values for y in by_y}
    rng = np.random.default_rng(seed)
    tot = sum(by_y.values())
    means = np.empty(NDRAW)
    for t in range(NDRAW):
        s = 0.0
        for y, n in by_y.items():
            a = pools[y]
            s += a[rng.choice(len(a), n, replace=False)].sum()
        means[t] = s / tot
    return float((means >= obs).mean())


per_long, pooled_long, pooled_long_net = daily_block(ev["R1"], range(2011, 2026))
long_daily = {"per_year": per_long, "pooled_2011_2025": {}}
for h in H:
    r = [v for _, v in pooled_long[h]]
    rn = [v for _, v in pooled_long_net[h]]
    yrs = [per_long[str(y)][h]["mean"] for y in range(2011, 2026) if per_long[str(y)][h]["mean"] is not None]
    long_daily["pooled_2011_2025"][h] = {
        "n": len(r), "mean": float(np.mean(r)), "mean_net_half_spread": float(np.mean(rn)),
        "median": float(np.median(r)), "win": float((np.array(r) > 0).mean()),
        "years_positive": int(sum(1 for v in yrs if v > 0)), "years_n": len(yrs),
        "p_random_same_year": strat_p(pooled_long[h], h),
        "ci95_boot": boot_ci(r),
    }
# 2011~2025 첫날 대 나머지(일봉)
lo_l, hi_l = pd.Timestamp("2011-01-01"), pd.Timestamp("2025-12-31")
fvr_long = {}
for h in H:
    s = DALL[h][(DALL[h].index >= lo_l) & (DALL[h].index <= hi_l)]
    first = [v for _, v in pooled_long[h]]
    sec = s[ev["second"].reindex(s.index).fillna(False).astype(bool)]
    nr = s[ev["nonred"].reindex(s.index).fillna(False).astype(bool)]
    fvr_long[h] = {"first_n": len(first), "first_mean": float(np.mean(first)),
                   "red_second_plus": {"n": int(len(sec)), "mean": float(sec.mean())},
                   "non_red": {"n": int(len(nr)), "mean": float(nr.mean())},
                   "first_minus_red_rest": float(np.mean(first) - sec.mean())}
fvr["daily_2011-2025"] = fvr_long

# 같은 근사를 2023~2026에 1분봉 값과 나란히
per_recent, pooled_recent, _ = daily_block(ev["R1"], (2023, 2024, 2025, 2026))
side = {}
for y in (2023, 2024, 2025, 2026):
    side[str(y)] = {"daily_open_to_close": per_recent[str(y)]["oc"],
                    "daily_open_to_D+1_open": per_recent[str(y)]["o1"],
                    "daily_open_to_D+5_open": per_recent[str(y)]["o5"],
                    "1m_O2": {"n": years_1m[str(y)]["O2"]["n"], "mean": years_1m[str(y)]["O2"]["mean"]},
                    "1m_O3": {"n": years_1m[str(y)]["O3"]["n"], "mean": years_1m[str(y)]["O3"]["mean"]},
                    "1m_O4": {"n": years_1m[str(y)]["O4"]["n"], "mean": years_1m[str(y)]["O4"]["mean"]}}
long_daily["side_by_side_2023_2026"] = side

# ── 검산 메모 ────────────────────────────────────────────
cal_23 = cal[(cal >= pd.Timestamp("2023-01-03")) & (cal <= hi26)]
missing_1m = [str(d.date()) for d in cal_23 if d not in mday]
extra_1m = [str(d.date()) for d in mday if d not in set(cal_23)]
stage_now = stg.dropna()
out = {
    "impl": "B (blind): pine 직역, redfirst.py·lib.py·stage5.py 미사용",
    "data": {"soxx_daily": [str(soxx_d.index[0].date()), str(soxx_d.index[-1].date())],
             "soxs_daily": [str(soxs_d.index[0].date()), str(soxs_d.index[-1].date())],
             "soxl_daily": [str(soxl_d.index[0].date()), str(soxl_d.index[-1].date())],
             "soxs_1m_days": [str(mday[0].date()), str(mday[-1].date()), int(len(mday))],
             "calendar_days_missing_in_1m": missing_1m, "1m_days_not_in_calendar": extra_1m,
             "cost_half_spread_SOXS": cost},
    "events": {"counts_by_year": counts, "R1_2026": lst},
    "primary_2026": {"R1": prim, "criteria": verdict_rows, "O2_pass_all3": O2_pass},
    "first_vs_rest": fvr,
    "years_1m": years_1m,
    "variants": variants,
    "long_daily": long_daily,
    "stage_distribution_2026": {str(k): int(v) for k, v in stg[(stg.index >= lo26)].value_counts().sort_index().items()},
}
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results/redfirst_blind.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)

# 요약 출력
print("counts", json.dumps(counts))
for r in lst:
    print(r)
for o in EXIT:
    print(o, {k: prim[o].get(k) for k in ["n", "mean", "p_random", "p_holm", "mean_cost2x", "ci95_boot", "ex_best_event", "ex_best_month", "best_month"]})
print("O2 pass", O2_pass)
for y in ("2023", "2024", "2025"):
    print(y, {o: (years_1m[y][o]["n"], round(years_1m[y][o]["mean"], 3) if years_1m[y][o]["mean"] is not None else None) for o in EXIT})
print("2023-2025 O2", years_1m["2023-2025_O2_pooled"])
print("long", json.dumps(long_daily["pooled_2011_2025"], indent=0))
print("missing", missing_1m[:10], "extra", extra_1m[:10])
