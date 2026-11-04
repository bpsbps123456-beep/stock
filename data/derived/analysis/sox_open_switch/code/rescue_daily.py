"""BRIEF7 — 묶음 daily: K1·K2 일봉판(합성 SOXS 1995~2010) (구현 A, 2026-11-04).

정본: ../BRIEF7.md(결과 보기 전 고정, 커밋 b241633). 후보 정의: BRIEF6 R2(K1)·R1(K2) = redfirst.flags 그대로,
단계 함수 = stage5.zone_calc 그대로(둘 다 고치지 않고 import만 한다). 무작위 뽑기 = redfirst.null_draws 그대로.

결과 보기 전에 이 파일에서 정한 해석(첫 실행 전 고정):
- 자료(yfinance, auto_adjust=False, Open·High·Low·Close·Volume, 날짜 정규화, 종가 없는 행 버림):
  ^SOX 1994-01-01~2011-01-10(end 미포함), SOXX 2001-01-01~2011-01-10.
  대조용: SOXX·SOXS = BRIEF6 캐시(redfirst.yf_daily: SOXX 2001-01-01~, SOXS 2010-01-01~, 2026-10-29까지),
  ^SOX 2010-01-01~2026-01-01.
- 단계: stage5.zone_calc(그 지수 종가). 지수마다 자기 거래일 순서로 D−1·D−2를 센다. 내려받은 첫 130봉은 값 없음.
  1995~2001 사건(D의 해)은 ^SOX 단계·^SOX 합성, 2002~2010 사건은 SOXX 단계·SOXX 합성.
  2002년 초 SOXX 단계가 아직 없는 날(상장 2001-07-13 뒤 첫 130봉)은 사건이 될 수 없다(^SOX로 메우지 않고 날 수를 적는다).
- 사건: K1 = redfirst.flags의 R2(단계(D−1) = −2, 단계(D−2) ≠ −2), K2 = R1(단계(D−1) ≤ −1, 단계(D−2) ≥ 0). 두 단계 모두 값이 있어야 함.
  K1·K2는 정의상 연속한 날에 생길 수 없어 보유가 겹치지 않는다(확인용으로 센다).
- 합성 SOXS(%, 비용 전): g_t = O_t/C_{t−1} − 1, r_t = C_t/C_{t−1} − 1,
  K1 값 = ((1 − 3r_D)/(1 − 3g_D) × (1 − 3g_{D+1}) − 1) × 100, K2 값 = ((1 − 3r_D)/(1 − 3g_D) − 1) × 100.
  D+1은 같은 지수의 다음 거래일(2010-12-31 사건은 2011-01-03 시가). 1 − 3g ≤ 0이거나 값이 유한하지 않으면 사건에서 뺀다(수를 적음).
- 비용: 순수익 = 값 − 0.2(%p, 왕복). 비용 2배 = 값 − 0.4. 운용 보수 무시.
- 시가 품질: 해마다 '시가 = 전날 종가'(|O_t − C_{t−1}| ≤ 1e-6·C_{t−1}) 비율, 분모 = 그 해 O_t > 0·C_{t−1}이 있는 날.
  ^SOX가 5%를 넘는 해의 ^SOX 사건(1995~2001)은 뺀다(BRIEF7 시가 규칙). SOXX도 같은 표를 내고, 5%를 넘는 해가 있으면
  그 해를 뺀 감도를 따로 낸다(주 판정은 BRIEF7 문구대로 ^SOX에만 규칙 적용).
  참고 열: |g| < 0.01% 비율, 시가 = 같은 날 종가 비율, O=H=L=C 비율, 시가가 [저가, 고가] 밖인 날, 시가 결측·0.
- 판정(후보마다, 1995~2010 합성 주 표본, 순수익):
  (1) 평균 > 0
  (2) 무작위 날 p(단측): rng = default_rng(20261104)(후보마다 새로). 해를 1995→2010 순서로 돌며 그 해 사건 수 n_y만큼
      그 해(사건과 같은 지수의 그 해 모든 거래일 중 합성값이 있는 날, 시가 규칙에서 뺀 해는 사건·풀 모두 없음)에서
      비복원 10,000회(redfirst.null_draws). 해마다 뽑은 합을 더해 전체 사건 수로 나눈 귀무 평균 ≥ 관측 평균(−1e-12)인 비율 = p.
      참고로 (1+#)/(1+10,000). Holm m = 6은 K1~K6 전체 p가 있어야 하므로 여기서는 p와 Holm 범위 [p, min(1, 6p)]만 낸다
      (6p ≤ 0.05면 다른 후보와 무관하게 통과, p > 0.05면 무관하게 불통과).
  (3) 순값 상위 2건을 뺀 평균 > 0 (4) 비용 2배 평균 > 0
  (5) 사건이 있는 해 중 해 순 평균 > 0인 해 수 × 2 ≥ 사건이 있는 해 수.
- 함께(판정 밖): 95% 구간(사건 복원 추출 10,000회, seed 20261104), 해마다 n·평균·p, 원천별(1995~2001 ^SOX, 2002~2010 SOXX),
  2002~2010 ^SOX 단계판(^SOX 합성·SOXX 합성), ^SOX 단계·합성 1995~2010 전 구간판, 규칙상 뺀 해를 넣은 참고판(뺀 해가 있을 때).
  1분봉 '채운 봉' 문제는 일봉이라 해당 없다.
- 2011~2025 대조: SOXX 단계(BRIEF6 캐시)로 같은 K1·K2 사건을 찾고 실제 SOXS(yfinance 분할 반영 Open·Close:
  K1 = SO_{D+1}/SO_D − 1, K2 = SC_D/SO_D − 1)와 SOXX 합성·^SOX 합성(같은 날짜)을 비교한다. 피어슨 상관, 평균, 평균 차,
  중앙 절대 차, 기울기(실제 = a + b·합성). 모든 날 대조도 같이. 실제 SOXS 사건 평균은 redfirst.json(R2 o1·R1 oc 2011~2025)과 맞춘다.
- 2002~2010 ^SOX·SOXX 대조: 단계 일치율, 사건 날짜 겹침, g·r 상관과 표준편차 비(^SOX 시가가 묵은 값인지).
- SOXX 새 내려받기(2001~2011-01-10)와 BRIEF6 캐시를 겹치는 날로 대조(가격·단계).
실행: cd code && py rescue_daily.py
결과: ../results/rescue_daily.json, ../results/rescue_daily_events.csv
"""
import hashlib, json, os, sys, time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage5 as S5  # noqa: E402
import redfirst as RF  # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
CACHE = os.path.normpath(os.path.join(HERE, "..", "cache"))
SEED, NR, NB = 20261104, 10000, 10000
assert RF.NR == NR
COST, COST2, QTH = 0.2, 0.4, 0.05
YEARS = tuple(range(1995, 2011))
Y_SOX, Y_SOXX = tuple(range(1995, 2002)), tuple(range(2002, 2011))
CMP_YEARS = tuple(range(2011, 2026))
KS = ("K1", "K2")
FLAG = {"K1": "R2", "K2": "R1"}
NAMES = S5.NAMES
fnum = S5.fnum
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- 자료
def yf_get(sym, start, end):
    tag = sym.replace("^", "").lower()
    p = os.path.join(CACHE, f"{tag}_1d_yf_ohlcv_{start.replace('-', '')}_{end.replace('-', '')}.csv")
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0, parse_dates=True)
    import yfinance as yf
    df = yf.download(sym, start=start, end=end, auto_adjust=False, progress=False, actions=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[df["Close"].notna()]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    os.makedirs(CACHE, exist_ok=True)
    df.to_csv(p)
    return df


def synth(df):
    """합성 SOXS(%, 비용 전) — 날 색인 t마다 K1(D 시가 → D+1 시가)·K2(D 시가 → D 종가), g, r."""
    O = df["Open"].to_numpy(float)
    C = df["Close"].to_numpy(float)
    Cp = np.r_[np.nan, C[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(O > 0, O / Cp - 1, np.nan)
        r = C / Cp - 1
        den = 1 - 3 * g
        a = np.where(den > 0, (1 - 3 * r) / den, np.nan)
        g1 = np.r_[g[1:], np.nan]
        k2 = (a - 1) * 100
        k1 = (a * (1 - 3 * g1) - 1) * 100
    return {"K1": np.where(np.isfinite(k1), k1, np.nan), "K2": np.where(np.isfinite(k2), k2, np.nan), "g": g, "r": r}


def open_quality(df):
    O, H, L, C = (df[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
    Cp = np.r_[np.nan, C[:-1]]
    yr = df.index.year.to_numpy()
    out = {}
    for y in sorted(set(yr.tolist())):
        ym = yr == y
        m = ym & np.isfinite(O) & (O > 0) & np.isfinite(Cp)
        n = int(m.sum())
        eq = np.abs(O - Cp) <= 1e-6 * Cp
        g = O / Cp - 1
        out[str(y)] = {"days": int(ym.sum()), "days_with_open_and_prev_close": n,
                       "open_eq_prev_close_n": int((m & eq).sum()),
                       "open_eq_prev_close_frac": fnum((m & eq).sum() / n) if n else None,
                       "abs_gap_lt_0.01pct_frac": fnum((m & (np.abs(g) < 1e-4)).sum() / n) if n else None,
                       "open_eq_same_close_frac": fnum((m & (np.abs(O - C) <= 1e-6 * C)).sum() / n) if n else None,
                       "ohlc_flat_frac": fnum((m & (H == L) & (O == C) & (H == O)).sum() / n) if n else None,
                       "open_outside_hl_n": int((m & ((O > H * (1 + 1e-9)) | (O < L * (1 - 1e-9)))).sum()),
                       "open_missing_or_nonpos_n": int((ym & ~(np.isfinite(O) & (O > 0))).sum()),
                       "pass_5pct": bool(n and (m & eq).sum() / n <= QTH)}
    return out


def cal_gaps(idx, min_days=5):
    d = pd.DatetimeIndex(idx)
    gaps = (d[1:] - d[:-1]).days
    return [f"{d[i].date()}→{d[i + 1].date()}" for i in np.flatnonzero(gaps >= min_days)]


# ---------------------------------------------------------------- 통계
def stats_block(rows, pools, seed=SEED):
    """rows: 사건 dict(year, gross). pools: {해: 그 해 순수익 풀 배열}. 판정 항목까지."""
    rows = sorted(rows, key=lambda r: r["date"])
    v = np.array([r["gross"] - COST for r in rows], float)
    c2 = np.array([r["gross"] - COST2 for r in rows], float)
    yrs = np.array([r["year"] for r in rows], int)
    n = len(v)
    out = {"n": int(n)}
    if n == 0:
        return out
    obs = v.mean()
    srt = np.sort(v)[::-1]
    out.update({"mean": fnum(obs), "median": fnum(np.median(v)), "hit": fnum((v > 0).mean()),
                "worst": fnum(v.min()), "best": fnum(v.max()), "sd": fnum(v.std(ddof=1)) if n > 1 else None,
                "sum": fnum(v.sum()), "gross_mean": fnum(v.mean() + COST),
                "ci95": S5.boot_mean(v, seed=seed) if n > 1 else None,
                "drop_top2": fnum(srt[2:].mean()) if n > 2 else None, "top2_values": [fnum(x) for x in srt[:2]],
                "cost2x": fnum(c2.mean())})
    rng = np.random.default_rng(seed)
    per, null_sum, null_n = {}, np.zeros(NR), 0
    for y in sorted(set(yrs.tolist()) | set(pools)):
        vy = v[yrs == y]
        ny = len(vy)
        pool = pools.get(y)
        cell = {"n": int(ny), "mean": fnum(vy.mean()) if ny else None,
                "pool_days": int(len(pool)) if pool is not None else 0,
                "pool_mean": fnum(np.mean(pool)) if pool is not None and len(pool) else None}
        if ny:
            assert pool is not None and len(pool) >= ny, (y, ny)
            nm, ns = RF.null_draws(pool, ny, rng)
            k_ = int((nm >= vy.mean() - 1e-12).sum())
            cell.update({"p_random": k_ / NR, "null_mean": fnum(nm.mean()), "cost2x_mean": fnum(c2[yrs == y].mean())})
            null_sum += ns
            null_n += ny
        per[str(y)] = cell
    nmt = null_sum / null_n
    k_ = int((nmt >= obs - 1e-12).sum())
    p = k_ / NR
    yw = [c for c in per.values() if c["n"]]
    ypos = sum(1 for c in yw if c["mean"] > 0)
    out.update({"p_one_sided": p, "p_plus1": (1 + k_) / (NR + 1), "null_mean": fnum(nmt.mean()),
                "null_q95": fnum(np.percentile(nmt, 95)),
                "holm_m6_range": [fnum(p), fnum(min(1.0, 6 * p))],
                "years_with_events": len(yw), "years_positive": int(ypos),
                "years_positive_str": f"{ypos}/{len(yw)}", "by_year": per})
    crit = {"(1) mean>0": bool(obs > 0),
            "(2) holm_m6_p<=0.05": ("pass_any" if 6 * p <= 0.05 else ("fail_any" if p > 0.05 else "depends_on_K3_K6")),
            "(3) drop_top2>0": bool(n > 2 and srt[2:].mean() > 0),
            "(4) cost2x>0": bool(c2.mean() > 0),
            "(5) half_years_positive": bool(2 * ypos >= len(yw))}
    out["criteria"] = crit
    others = [crit[k] for k in crit if k != "(2) holm_m6_p<=0.05"]
    out["pass_1_3_4_5"] = bool(all(others))
    out["pass_all"] = ("yes" if all(others) and crit["(2) holm_m6_p<=0.05"] == "pass_any" else
                       ("no" if (not all(others)) or crit["(2) holm_m6_p<=0.05"] == "fail_any" else "depends_on_holm"))
    return out


class Src:
    """한 지수 일봉과 그 단계·합성·사건 불 배열."""

    def __init__(self, name, df):
        self.name, self.df = name, df
        self.dates = pd.DatetimeIndex(df.index)
        self.pos = {d: t for t, d in enumerate(self.dates)}
        self.yr = self.dates.year.to_numpy()
        self.zn, self.rsi, self.z = S5.zone_calc(df["Close"].to_numpy(float))
        self.F = RF.flags(self.zn)
        self.syn = synth(df)


def build(stage_src, val_src, years, excl, kind, tag):
    """stage_src 단계로 사건, val_src 합성으로 값(날짜로 짝). excl: val_src 시가 품질로 뺀 해."""
    flag = stage_src.F[FLAG[kind]]
    rows, cnt = [], {"flag_days": 0, "excluded_quality_year": 0, "no_value_date": 0, "no_value_nan": 0, "kept": 0}
    zp = np.r_[np.nan, stage_src.zn[:-1]]
    zp2 = np.r_[np.nan, np.nan, stage_src.zn[:-2]]
    rp = np.r_[np.nan, stage_src.rsi[:-1]]
    for t in np.flatnonzero(flag & np.isin(stage_src.yr, years)):
        d = stage_src.dates[t]
        cnt["flag_days"] += 1
        if d.year in excl:
            cnt["excluded_quality_year"] += 1
            continue
        j = val_src.pos.get(d, -1)
        if j < 0:
            cnt["no_value_date"] += 1
            continue
        val = val_src.syn[kind][j]
        if not np.isfinite(val):
            cnt["no_value_nan"] += 1
            continue
        cnt["kept"] += 1
        rows.append({"variant": tag, "candidate": kind, "date": str(d.date()), "year": int(d.year),
                     "stage_src": stage_src.name, "val_src": val_src.name,
                     "stage_d1": int(zp[t]), "stage_d2": int(zp2[t]), "rsi_d1": fnum(rp[t]),
                     "g_D_pct": fnum(val_src.syn["g"][j] * 100), "r_D_pct": fnum(val_src.syn["r"][j] * 100),
                     "g_D1_pct": fnum(val_src.syn["g"][j + 1] * 100) if j + 1 < len(val_src.dates) else None,
                     "next_date": str(val_src.dates[j + 1].date()) if j + 1 < len(val_src.dates) else None,
                     "gross": float(val), "net": float(val - COST)})
    pools = {}
    for y in years:
        if y in excl:
            continue
        a = val_src.syn[kind][(val_src.yr == y)]
        a = a[np.isfinite(a)]
        if len(a):
            pools[y] = a - COST
    return rows, pools, cnt


def run_variant(parts, kind, tag):
    rows, pools, cnts = [], {}, {}
    for (ss, vs, years, excl) in parts:
        r_, p_, c_ = build(ss, vs, years, excl, kind, tag)
        rows += r_
        pools.update(p_)
        cnts[f"{ss.name}->{vs.name} {years[0]}-{years[-1]}"] = c_
    st = stats_block(rows, pools)
    st["counts"] = cnts
    # 겹침 확인(K1: D+1 시가 청산 → 다음 사건 날짜가 앞 사건 next_date보다 앞이면 겹침)
    rs = sorted(rows, key=lambda r: r["date"])
    if kind == "K1":
        st["overlaps"] = int(sum(1 for a, b in zip(rs[:-1], rs[1:]) if b["date"] < a["next_date"]))
    else:
        st["overlaps"] = int(sum(1 for a, b in zip(rs[:-1], rs[1:]) if b["date"] == a["date"]))
    return st, rows


def by_source(rows):
    out = {}
    for nm, ys in (("1995_2001", Y_SOX), ("2002_2010", Y_SOXX)):
        v = np.array([r["net"] for r in rows if r["year"] in ys])
        out[nm] = {"n": int(len(v)), "mean": fnum(v.mean()) if len(v) else None,
                   "hit": fnum((v > 0).mean()) if len(v) else None}
    return out


def cmp_stats(a, b):
    """a: 합성, b: 실제(같은 길이, 유한)."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3:
        return {"n": int(len(a))}
    sl, ic = np.polyfit(a, b, 1)
    return {"n": int(len(a)), "corr": fnum(np.corrcoef(a, b)[0, 1]), "synth_mean": fnum(a.mean()), "actual_mean": fnum(b.mean()),
            "mean_diff_actual_minus_synth": fnum((b - a).mean()), "median_abs_diff": fnum(np.median(np.abs(b - a))),
            "slope_actual_on_synth": fnum(sl), "intercept": fnum(ic),
            "sign_agree": fnum((np.sign(a) == np.sign(b)).mean())}


# ---------------------------------------------------------------- 본체
def main():
    os.makedirs(RES, exist_ok=True)
    checks = {}
    sox_df = yf_get("^SOX", "1994-01-01", "2011-01-10")
    soxx_df = yf_get("SOXX", "2001-01-01", "2011-01-10")
    SOX, SOXX = Src("SOX", sox_df), Src("SOXX", soxx_df)
    log(f"^SOX {SOX.dates[0].date()}~{SOX.dates[-1].date()} {len(SOX.dates)}봉, SOXX {SOXX.dates[0].date()}~{SOXX.dates[-1].date()} {len(SOXX.dates)}봉")
    checks["sox"] = {"first": str(SOX.dates[0].date()), "last": str(SOX.dates[-1].date()), "bars": len(SOX.dates),
                     "first_valid_stage": str(SOX.dates[np.argmax(np.isfinite(SOX.zn))].date()),
                     "calendar_gaps_ge5d": cal_gaps(SOX.dates),
                     "close_nan_or_nonpos": int((~(sox_df["Close"] > 0)).sum())}
    checks["soxx"] = {"first": str(SOXX.dates[0].date()), "last": str(SOXX.dates[-1].date()), "bars": len(SOXX.dates),
                      "first_valid_stage": str(SOXX.dates[np.argmax(np.isfinite(SOXX.zn))].date()),
                      "calendar_gaps_ge5d": cal_gaps(SOXX.dates)}
    # 2002~2010 SOXX 단계 없는 날
    m02 = (SOXX.yr >= 2002) & (SOXX.yr <= 2010)
    zp1, zp2 = np.r_[np.nan, SOXX.zn[:-1]], np.r_[np.nan, np.nan, SOXX.zn[:-2]]
    checks["soxx_days_2002_2010_without_stage_d1_d2"] = int((m02 & ~(np.isfinite(zp1) & np.isfinite(zp2))).sum())
    log("자료", checks["sox"], checks["soxx"], "SOXX 단계 없는 2002~ 날", checks["soxx_days_2002_2010_without_stage_d1_d2"])

    # 시가 품질
    q_sox, q_soxx = open_quality(sox_df), open_quality(soxx_df)
    sox_fail = sorted(int(y) for y, c in q_sox.items() if 1995 <= int(y) <= 2010 and not c["pass_5pct"])
    soxx_fail = sorted(int(y) for y, c in q_soxx.items() if 2002 <= int(y) <= 2010 and not c["pass_5pct"])
    sox_fail_primary = [y for y in sox_fail if y in Y_SOX]
    log("^SOX 시가 품질", {y: (c["open_eq_prev_close_frac"], c["open_eq_same_close_frac"]) for y, c in q_sox.items()})
    log("SOXX 시가 품질", {y: c["open_eq_prev_close_frac"] for y, c in q_soxx.items()})
    log("뺀 해 ^SOX", sox_fail, "SOXX 5% 넘는 해", soxx_fail)

    # SOXX 새 내려받기 대 BRIEF6 캐시
    Xc = RF.yf_daily("SOXX", "2001-01-01")
    Sc = RF.yf_daily("SOXS", "2010-01-01")
    com = soxx_df.index.intersection(Xc.index)
    rel = {c: fnum((np.abs(soxx_df.loc[com, c] / Xc.loc[com, c] - 1)).max() * 100) for c in ("Open", "Close")}
    znc, rsic, _ = S5.zone_calc(Xc["Close"].to_numpy(float))
    zc = pd.Series(znc, index=Xc.index).reindex(com)
    zn_new = pd.Series(SOXX.zn, index=SOXX.dates).reindex(com)
    okz = zc.notna() & zn_new.notna()
    checks["soxx_new_vs_brief6_cache"] = {"common_days": int(len(com)), "max_rel_diff_pct": rel,
                                          "stage_agree": fnum((zc[okz] == zn_new[okz]).mean()), "stage_n": int(okz.sum()),
                                          "new_dates_not_in_cache": int(len(soxx_df.index.difference(Xc.index))),
                                          "cache_dates_in_range_not_in_new": int(len(Xc.index[(Xc.index >= soxx_df.index[0]) & (Xc.index <= soxx_df.index[-1])].difference(soxx_df.index)))}
    log("SOXX 캐시 대조", checks["soxx_new_vs_brief6_cache"])

    # ^SOX 대 SOXX 2002~2010
    comm = SOX.dates.intersection(SOXX.dates)
    comm = comm[(comm.year >= 2002) & (comm.year <= 2010)]
    iS = np.array([SOX.pos[d] for d in comm])
    iX = np.array([SOXX.pos[d] for d in comm])
    a_s, a_x = SOX.zn[iS], SOXX.zn[iX]
    okb = np.isfinite(a_s) & np.isfinite(a_x)
    sv = {}
    for k in ("g", "r"):
        s_, x_ = SOX.syn[k][iS], SOXX.syn[k][iX]
        m = np.isfinite(s_) & np.isfinite(x_)
        sl = np.polyfit(x_[m], s_[m], 1)[0]
        sv[k] = {"corr": fnum(np.corrcoef(s_[m], x_[m])[0, 1]), "sd_ratio_sox_over_soxx": fnum(s_[m].std() / x_[m].std()),
                 "slope_sox_on_soxx": fnum(sl), "n": int(m.sum())}
    sv_year = {}
    for y in Y_SOXX:
        my = comm.year == y
        s_, x_ = SOX.syn["g"][iS][my], SOXX.syn["g"][iX][my]
        m = np.isfinite(s_) & np.isfinite(x_)
        sv_year[str(y)] = {"g_corr": fnum(np.corrcoef(s_[m], x_[m])[0, 1]), "g_sd_ratio": fnum(s_[m].std() / x_[m].std())}
    ev_ov = {}
    for kind in KS:
        fs = set(comm[SOX.F[FLAG[kind]][iS]])
        fx = set(comm[SOXX.F[FLAG[kind]][iX]])
        ev_ov[kind] = {"sox_events": len(fs), "soxx_events": len(fx), "both": len(fs & fx),
                       "jaccard": fnum(len(fs & fx) / len(fs | fx)) if fs | fx else None}
    checks["sox_vs_soxx_2002_2010"] = {"common_days": int(len(comm)),
                                       "sox_days_not_in_soxx": int(sum(1 for d in SOX.dates if 2002 <= d.year <= 2010 and d not in SOXX.pos)),
                                       "soxx_days_not_in_sox": int(sum(1 for d in SOXX.dates if 2002 <= d.year <= 2010 and d not in SOX.pos)),
                                       "stage_agree": fnum((a_s[okb] == a_x[okb]).mean()), "stage_n": int(okb.sum()),
                                       "g_r": sv, "g_by_year": sv_year, "event_overlap": ev_ov}
    log("^SOX 대 SOXX", {k: v for k, v in checks["sox_vs_soxx_2002_2010"].items() if k != "g_by_year"})

    # ---------------------------------------------------------------- 주 판정
    cand, expl, all_rows = {}, {}, []
    for kind in KS:
        parts = [(SOX, SOX, Y_SOX, set(sox_fail_primary)), (SOXX, SOXX, Y_SOXX, set())]
        st, rows = run_variant(parts, kind, "primary")
        st["by_source"] = by_source(rows)
        st["excluded_years_sox_open_rule"] = sox_fail_primary
        st["filled_bar_issue"] = "해당 없음(일봉)"
        cand[kind] = st
        all_rows += rows
        ex = {}
        if soxx_fail:
            s2, r2 = run_variant([(SOX, SOX, Y_SOX, set(sox_fail_primary)), (SOXX, SOXX, Y_SOXX, set(soxx_fail))], kind, "sens_soxx_quality")
            ex["sens_drop_soxx_years_over_5pct"] = s2
            all_rows += r2
        if sox_fail_primary:
            s3, r3 = run_variant([(SOX, SOX, Y_SOX, set()), (SOXX, SOXX, Y_SOXX, set())], kind, "ref_incl_excluded_years")
            ex["ref_including_excluded_sox_years"] = s3
            all_rows += r3
        s4, r4 = run_variant([(SOX, SOX, Y_SOXX, set(y for y in sox_fail if y in Y_SOXX))], kind, "sox_stage_sox_synth_2002_2010")
        ex["sox_stage_sox_synth_2002_2010"] = s4
        all_rows += r4
        s5, r5 = run_variant([(SOX, SOXX, Y_SOXX, set())], kind, "sox_stage_soxx_synth_2002_2010")
        ex["sox_stage_soxx_synth_2002_2010"] = s5
        all_rows += r5
        s6, r6 = run_variant([(SOX, SOX, YEARS, set(sox_fail))], kind, "sox_only_1995_2010")
        ex["sox_only_1995_2010"] = s6
        all_rows += r6
        expl[kind] = ex
        log(kind, {k: st.get(k) for k in ("n", "mean", "median", "hit", "p_one_sided", "drop_top2", "cost2x", "years_positive_str", "pass_all")},
            st["criteria"], st["counts"])
        log(kind, "원천별", st["by_source"])
        log(kind, "설명용", {k: (v.get("n"), v.get("mean"), v.get("p_one_sided")) for k, v in ex.items()})

    # ---------------------------------------------------------------- 2011~2025 대조
    sox2_df = yf_get("^SOX", "2010-01-01", "2026-01-01")
    SOX2 = Src("SOX2010", sox2_df)
    X2 = Src("SOXX_cache", Xc[["Open", "High", "Low", "Close"]].assign(Volume=np.nan))
    sd_ = pd.DatetimeIndex(Sc.index)
    jS = {d: j for j, d in enumerate(sd_)}
    so, sc = Sc["Open"].to_numpy(float), Sc["Close"].to_numpy(float)
    act = {"K1": np.r_[(so[1:] / so[:-1] - 1) * 100, np.nan], "K2": (sc / so - 1) * 100}
    rf = json.load(open(os.path.join(RES, "redfirst.json"), encoding="utf-8"))
    rf_ref = {"K1": rf["variants"]["R2"]["long_daily"]["o1"]["total"], "K2": rf["long_daily"]["R1"]["oc"]["total"]}
    cmpres = {}
    for kind in KS:
        tt = np.flatnonzero(X2.F[FLAG[kind]] & np.isin(X2.yr, CMP_YEARS))
        ds = [X2.dates[t] for t in tt]
        a_act = np.array([act[kind][jS[d]] if d in jS else np.nan for d in ds])
        a_sx = np.array([X2.syn[kind][t] for t in tt])
        a_ss = np.array([SOX2.syn[kind][SOX2.pos[d]] if d in SOX2.pos else np.nan for d in ds])
        okm = np.isfinite(a_act)
        res = {"events": int(len(tt)), "actual_event_mean_gross": fnum(np.nanmean(a_act)), "actual_n": int(okm.sum()),
               "brief6_reference": {"n": rf_ref[kind]["n"], "mean": rf_ref[kind]["mean"]},
               "actual_matches_brief6": bool(int(okm.sum()) == rf_ref[kind]["n"] and abs(np.nanmean(a_act) - rf_ref[kind]["mean"]) < 1e-9),
               "events_soxx_synth_vs_actual": cmp_stats(a_sx, a_act),
               "events_sox_synth_vs_actual": cmp_stats(a_ss, a_act),
               "events_soxx_synth_net_mean": fnum(np.nanmean(a_sx[okm]) - COST),
               "events_actual_net_mean": fnum(np.nanmean(a_act[okm]) - COST)}
        # 모든 날
        m_all = np.isin(X2.yr, CMP_YEARS)
        dall = X2.dates[m_all]
        b_act = np.array([act[kind][jS[d]] if d in jS else np.nan for d in dall])
        b_sx = X2.syn[kind][m_all]
        b_ss = np.array([SOX2.syn[kind][SOX2.pos[d]] if d in SOX2.pos else np.nan for d in dall])
        res["all_days_soxx_synth_vs_actual"] = cmp_stats(b_sx, b_act)
        res["all_days_sox_synth_vs_actual"] = cmp_stats(b_ss, b_act)
        # 2011~2020(SOXS 기초 = PHLX SOX)·2021~2025 나눠서
        for nm, ys in (("2011_2020", range(2011, 2021)), ("2021_2025", range(2021, 2026))):
            mm = np.isin(dall.year, list(ys))
            res[f"all_days_sox_synth_vs_actual_{nm}"] = cmp_stats(b_ss[mm], b_act[mm])
            res[f"all_days_soxx_synth_vs_actual_{nm}"] = cmp_stats(b_sx[mm], b_act[mm])
        cmpres[kind] = res
        log("대조", kind, {k: v for k, v in res.items() if not k.startswith("all_days_") or k in ("all_days_soxx_synth_vs_actual", "all_days_sox_synth_vs_actual")})
    checks["sox2010_vs_sox1994_overlap"] = {
        "days": int(len(SOX2.dates.intersection(SOX.dates))),
        "max_rel_close_diff_pct": fnum((np.abs(sox2_df["Close"].reindex(SOX.dates.intersection(SOX2.dates)) /
                                               sox_df["Close"].reindex(SOX.dates.intersection(SOX2.dates)) - 1)).max() * 100)}

    ev = pd.DataFrame(all_rows)
    ev.to_csv(os.path.join(RES, "rescue_daily_events.csv"), index=False, encoding="utf-8-sig")
    code_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
    out = {"brief": "BRIEF7.md", "bundle": "daily (K1, K2 주)", "impl": "A (rescue_daily.py)", "code_sha256_16": code_sha,
           "units": "% (1.0 = 1%)",
           "settings": {"seed": SEED, "random_draws": NR, "boot": NB, "cost_roundtrip": COST, "cost2x": COST2,
                        "open_rule": "^SOX 해마다 시가 = 전날 종가(상대 1e-6) 비율 > 5%면 그 해 ^SOX 사건 제외",
                        "stage": "stage5.zone_calc(종가), D−1·D−2는 지수 자기 거래일 순",
                        "events": {"K1": "redfirst.flags R2(단계(D−1) = −2, 단계(D−2) ≠ −2)", "K2": "redfirst.flags R1(빨강 첫날)"},
                        "synth": {"K1": "(1−3r_D)/(1−3g_D)·(1−3g_{D+1}) − 1", "K2": "(1−3r_D)/(1−3g_D) − 1"},
                        "sources": {"1995_2001": "^SOX 단계·합성", "2002_2010": "SOXX 단계·합성"},
                        "sox": f"yfinance ^SOX {SOX.dates[0].date()}~{SOX.dates[-1].date()} auto_adjust=False",
                        "soxx": f"yfinance SOXX {SOXX.dates[0].date()}~{SOXX.dates[-1].date()} auto_adjust=False",
                        "compare": "SOXX·SOXS BRIEF6 캐시(~2026-10-29), ^SOX 2010-01-01~2026-01-01",
                        "p": "해 층화 무작위 날(같은 해·같은 수, 비복원 10,000회, 1995→2010 한 rng), 귀무 평균 ≥ 관측 비율"},
           "checks": checks,
           "open_quality": {"SOX": q_sox, "SOXX": q_soxx, "sox_years_over_5pct_1995_2010": sox_fail,
                            "sox_excluded_from_primary": sox_fail_primary, "soxx_years_over_5pct_2002_2010": soxx_fail},
           "notes": [
               "^SOX 시가 규칙(시가 = 전날 종가 > 5%)에 1995·1997~2007이 걸려, 1995~2001 몫은 1996 한 해(K1 6건·K2 6건)뿐이다.",
               "규칙을 통과한 해도 ^SOX 시가가 묵은 값일 수 있다: 2008 ^SOX 갭 g의 표준편차는 SOXX의 0.35배·상관 0.44"
               "(2009·2010은 0.85배). 1996 ^SOX 합성 시가는 대조할 ETF가 없어 확인하지 못한다(설명용 주의).",
               "K1·K2는 정의상 연속한 날에 생기지 않아 보유 겹침 0건. 일봉이라 1분봉 '채운 봉' 문제는 해당 없음.",
               "Holm m = 6은 K3~K6 p가 있어야 확정된다. 여기의 holm_m6_range = [p, min(1, 6p)]."],
           "candidates": cand, "explanatory": expl, "compare_2011_2025_actual_soxs": cmpres,
           "files": ["rescue_daily.json", "rescue_daily_events.csv"]}
    json.dump(out, open(os.path.join(RES, "rescue_daily.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    log("끝")


if __name__ == "__main__":
    main()
