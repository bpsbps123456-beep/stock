"""H01 관문 미리보기 — 순심리 S(W_KR)와 '지난' 가격의 상관, M0 블록별 부분 R². HYPOTHESES.md §2.3·§2.7·§3 H01.

미리보기다. 지금 자료(국내 본판 창 2026-07-15~10-15)만 쓴다. 관문의 최종 판정은 수집 완료 뒤
2026 전체 창(2026-01-02~10-23)으로 한다(HYPOTHESES.md §1 순서 5). 이 파일의 결과로 관문을 통과·실패로 적지 않는다.

입력(읽기만):
  results/feat_text.parquet       국내 본판 창의 S·E·F 작성자 비율·결번율·창 정보(사전 v0.2, 10/15 스냅샷)
  results/loader_calendar.parquet KRX·NYSE 거래일(날짜만)
  가격: yfinance ^KS11·000660.KS·^SOX·EWY 일봉 종가(KOSPI는 거래량도), 2026-10-14(마지막 창 D=10/15의 P일)까지.
  일정: FOMC(stop_research/code/fam_H_events_funds.py에 적힌 연준 성명일), 실적일(yfinance get_earnings_dates의 날짜만).
방화벽:
  - 창 D에는 KRX는 P일(창 안에서 끝난 마지막 세션) 종가까지, 미국은 D−1일까지의 세션(KST D 05:00/06:00에 끝남)까지만 붙인다.
  - 시가·고가·저가는 받자마자 버린다. 진입 이후 값(R_h·MDD_h·D일 HL·시가→종가·갭)은 만들지 않는다.
  - 받은 가격의 마지막 날짜가 10/14를 넘으면 멈춘다.
산출: results/h01_preview.md, results/h01_preview.json, results/h01_m0_matrix.parquet(창별 S와 M0 열 — 모두 지난 값)
해석 기록: interim/notes_h01_preview.md
실행: py h01_preview.py   (이 폴더에서)
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

INTERIM = Path(__file__).resolve().parents[1]
RESULTS = INTERIM / "results"
ANT = INTERIM.parent

# ---------------------------------------------------------------- 고정값(문서 값 그대로, 문서에 없는 값은 notes에 근거)
LAST_PRICE_DAY = date(2026, 10, 14)   # 마지막 창 K2026-10-15의 P일. 이 뒤 가격은 받지 않는다
PRICE_START = date(2026, 1, 1)        # r60·RV20·거래량 기준(직전 20일)의 예열
HAC_L = 5                             # §2.7 HAC(5)
GATE_R, GATE_P = 0.15, 0.05           # §2.7 H01 관문
SHIFT_MIN = 21                        # §2.7 p 계산 ①: 21~(N−21) 원형 이동
LAGS = 20                             # H01 '지연 1~20일은 설명용'
DUMMY_MIN_ONES = 2                    # 표본 안에 1이 2개 미만인 일정 더미는 뺀다(한 점만 맞추는 열, notes)
CRASH_DAYS = [date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31), date(2026, 10, 13)]   # §2.7 급락 4일
OPENED_DAYS = [date(2026, 8, 20), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 8), date(2026, 9, 16),
               date(2026, 10, 12), date(2026, 10, 14), date(2026, 10, 15)]                  # §2.7 강건 ③(보조로만)
TICKERS = {"kospi": "^KS11", "hynix": "000660.KS", "sox": "^SOX", "ewy": "EWY"}
# 연준 성명일(14:00 ET = KST 다음 날 03:00). 출처: data/derived/analysis/stop_research/code/fam_H_events_funds.py FOMC
FOMC_2026 = [date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17), date(2026, 7, 29),
             date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9)]
# docs/desk_system.md §12.5 동결 지문(실행 때마다 대조한다)
FROZEN_EXPECTED = {"HYPOTHESES.md": "b23d1eacc95d0ae4ce1361c2449d9efd0f8feecc9792c4515ee2aff80ac308e5",
                   "HYPOTHESES_addendum1.md": "430dd4ffe856420fa9779dff91dc6e3cb3d3b7548551e02ed4dfab2596044eb4",
                   "lexicon_draft.md": "a4e28b08ddc3e6cef8b4f49c746c98845cc0f97bb8f03f534ba181a42795f8e1"}
EARN_TICKERS = ["NVDA", "MU", "AVGO", "000660.KS", "005930.KS"]   # §2.4 하닉·NVDA·MU·AVGO 실적 + 삼전(잠정 대신 확정일, notes)

KR_PRICE = ["kospi_rpos", "kospi_rneg", "kospi_r5", "kospi_r20", "kospi_r60", "kospi_DD20"]
US_NIGHT = ["sox_r_u", "ewy_r_u"]
VOLTRADE = ["kospi_lnRV20", "kospi_ln_vol_anom"]
CAL_ALL = ["cal_weekend", "cal_holiday", "cal_xholiday", "cal_vacation", "cal_kr_opex", "cal_us_opex",
           "cal_fomc", "cal_earn", "missing_rate"]
HYNIX_EXTRA = ["hynix_rpos", "hynix_rneg", "hynix_r5", "hynix_r20"]
BLOCK_NAMES = {"kr_price": "한국 가격", "us_night": "미국 밤", "voltrade": "변동·거래", "cal": "달력·품질"}


# ---------------------------------------------------------------- 가격(지난 값만)
def guard_past(days, last: date = LAST_PRICE_DAY) -> None:
    """방화벽: last 뒤 날짜가 하나라도 있으면 멈춘다."""
    bad = sorted(d for d in days if d > last)
    if bad:
        raise RuntimeError(f"방화벽: {last} 뒤 가격이 들어왔다({bad[0]} 등 {len(bad)}개)")


def fetch_daily(ticker: str, want_volume: bool = False) -> pd.DataFrame:
    import yfinance as yf
    h = yf.Ticker(ticker).history(start=PRICE_START.isoformat(),
                                  end=(LAST_PRICE_DAY + timedelta(days=1)).isoformat(), auto_adjust=False)
    keep = ["Close"] + (["Volume"] if want_volume else [])
    h = h[keep].copy()                                    # 시가·고가·저가는 여기서 버린다
    h.index = [pd.Timestamp(t).date() for t in h.index]
    h = h[~h.index.duplicated(keep="last")]
    guard_past(h.index)
    return h.dropna(subset=["Close"])


def session_features(close: pd.Series, volume: pd.Series | None = None, lags: int = LAGS) -> pd.DataFrame:
    """세션 d의 값 = d 종가까지로만 정해지는 값(뒤로만 보는 창). 앞 세션 값은 쓰지 않는다."""
    c = close.astype(float)
    r = c / c.shift(1) - 1
    lr = np.log(c / c.shift(1))
    out = pd.DataFrame(index=c.index)
    out["r"] = r
    out["rpos"] = r.clip(lower=0)
    out["rneg"] = r.clip(upper=0)
    out["r5"] = c / c.shift(5) - 1
    out["r20"] = c / c.shift(20) - 1
    out["r60"] = c / c.shift(60) - 1
    out["DD20"] = c / c.rolling(20, min_periods=20).max() - 1
    out["lnRV20"] = np.log(lr.rolling(20, min_periods=20).std(ddof=1))
    if volume is not None:
        v = volume.astype(float).where(volume > 0)        # 거래량 0(원천 결측)은 결측으로
        base = v.shift(1).rolling(20, min_periods=15).mean()
        out["ln_vol_anom"] = np.log(v / base)
    for k in range(1, lags + 1):
        out[f"r_lag{k}"] = r.shift(k)
    return out


def last_us_session(D: date, us_sessions) -> date:
    """W_KR(D) 안에서 끝난 마지막 미국 세션: 세션일 u는 KST u+1 05:00/06:00에 끝나므로 u ≤ D−1 중 가장 늦은 날."""
    c = [s for s in us_sessions if s <= D - timedelta(days=1)]
    if not c:
        raise RuntimeError(f"{D}: 앞선 미국 세션 없음")
    return max(c)


# ---------------------------------------------------------------- 달력
def nth_weekday(y: int, m: int, weekday: int, n: int) -> date:
    d = date(y, m, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def to_prev_session(d: date, sessions: set, cal_last: date) -> date:
    """만기일이 휴장이면 직전 거래일로. 달력 끝(cal_last) 뒤 날짜는 옮기지 않는다(어느 창에도 들지 않는다)."""
    if d > cal_last:
        return d
    while d not in sessions:
        d -= timedelta(days=1)
    return d


def in_window(e: date, P: date, D: date) -> bool:
    """일정일 e가 W_KR(D) = [P 08:00, D 08:00) 안에 든다: 미국 장전·장후·FOMC(e 21:30~e+1 06:00 KST)와
    국내 장전·장중 발표(e 08:00 뒤)를 같은 규칙으로 읽어 e ∈ [P, D−1](notes)."""
    return P <= e <= D - timedelta(days=1)


def fetch_earnings() -> dict:
    import yfinance as yf
    out = {}
    for t in EARN_TICKERS:
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=12)
            out[t] = sorted({pd.Timestamp(x).date() for x in e.index})   # 날짜만 쓴다(EPS 열은 버림)
        except Exception as ex:                                          # noqa: BLE001
            out[t] = f"못 함: {type(ex).__name__}"
    return out


# ---------------------------------------------------------------- 통계
def ols(y: np.ndarray, X: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def hac_cov(X: np.ndarray, e: np.ndarray, pos: np.ndarray, L: int = HAC_L, small: bool = True) -> np.ndarray:
    """Newey–West(Bartlett) 공분산. 지연 l의 짝은 원래 창 위치 차가 정확히 l인 행끼리만(빠진 창을 잇지 않는다).
    small=True면 n/(n−k)를 곱한다(보수)."""
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
    V = inv @ S @ inv
    return V * (n / (n - k)) if small else V


def hac_t(y: np.ndarray, X: np.ndarray, pos: np.ndarray, j: int, L: int = HAC_L) -> dict:
    beta, e = ols(y, X)
    V = hac_cov(X, e, pos, L)
    se = math.sqrt(max(V[j, j], 0.0))
    n, k = X.shape
    t = beta[j] / se if se > 0 else float("nan")
    df = n - k
    return {"beta": float(beta[j]), "se": se, "t": float(t), "df": int(df),
            "p_two": float(2 * stats.t.sf(abs(t), df)), "p_one_pos": float(stats.t.sf(t, df))}


def zs(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std(ddof=1)


def corr_test(s: np.ndarray, r: np.ndarray, pos: np.ndarray, method: str = "pearson",
              controls: np.ndarray | None = None) -> dict:
    """corr(S, r)와 HAC(5) p. 표준화한 S를 표준화한 r에 회귀하면 기울기 = 상관(Pearson). Spearman은 순위로 같은 셈.
    controls가 있으면 부분 상관(둘 다 controls에 잔차화)과, S ~ r + controls 회귀의 r 계수 HAC t."""
    s = np.asarray(s, float)
    r = np.asarray(r, float)
    if method == "spearman":
        s, r = stats.rankdata(s), stats.rankdata(r)
        if controls is not None:
            controls = np.column_stack([stats.rankdata(c) for c in np.atleast_2d(controls.T)])
    n = len(s)
    one = np.ones(n)
    if controls is None:
        rho = float(np.corrcoef(s, r)[0, 1])
        h = hac_t(zs(s), np.column_stack([one, zs(r)]), pos, 1)
    else:
        C = np.column_stack([one, controls])
        _, es = ols(s, C)
        _, er = ols(r, C)
        rho = float(np.corrcoef(es, er)[0, 1])
        h = hac_t(zs(s), np.column_stack([one, zs(r), controls]), pos, 1)
    return {"rho": rho, "n": n, "p_two": h["p_two"], "p_one_pos": h["p_one_pos"], "t": h["t"], "df": h["df"]}


def circ_shifts(n: int, kmin: int = SHIFT_MIN) -> list[int]:
    return list(range(kmin, n - kmin + 1))


def circ_p_corr(s: np.ndarray, r: np.ndarray, method: str = "pearson") -> dict:
    f = (lambda a, b: stats.spearmanr(a, b)[0]) if method == "spearman" else (lambda a, b: np.corrcoef(a, b)[0, 1])
    obs = f(s, r)
    ks = circ_shifts(len(s))
    if not ks:
        return {"K": 0, "p_two": None, "p_one_pos": None}
    null = np.array([f(np.roll(s, k), r) for k in ks])
    return {"K": len(ks), "p_two": float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(ks))),
            "p_one_pos": float((1 + np.sum(null >= obs)) / (1 + len(ks))), "min_p": 1 / (1 + len(ks))}


def r2(y: np.ndarray, X: np.ndarray) -> float:
    _, e = ols(y, X)
    d = y - y.mean()
    return float(1 - (e @ e) / (d @ d))


def design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])


def shapley(r2_of, blocks: list[str]) -> dict:
    """블록 Shapley(LMG) 몫. 합 = 전체 R²."""
    m = len(blocks)
    out = {}
    for b in blocks:
        rest = [x for x in blocks if x != b]
        v = 0.0
        for k in range(m):
            for S in combinations(rest, k):
                w = math.factorial(k) * math.factorial(m - k - 1) / math.factorial(m)
                v += w * (r2_of(set(S) | {b}) - r2_of(set(S)))
        out[b] = v
    return out


def block_table(y: np.ndarray, df: pd.DataFrame, blocks: dict[str, list[str]]) -> dict:
    """전체 R²·수정 R², 블록마다 단독 R²·뺄 때 ΔR²·부분 R²((SSE_빼고 − SSE_전체) ÷ SSE_빼고)·Shapley 몫."""
    names = list(blocks)
    cache: dict[frozenset, float] = {}

    def r2_of(S) -> float:
        key = frozenset(S)
        if key not in cache:
            cols = [c for b in names if b in key for c in blocks[b]]
            cache[key] = r2(y, design(df, cols)) if cols else 0.0
        return cache[key]

    full = r2_of(set(names))
    n = len(y)
    k = sum(len(v) for v in blocks.values())
    adj = 1 - (1 - full) * (n - 1) / (n - k - 1)
    sh = shapley(r2_of, names)
    rows = {}
    for b in names:
        without = r2_of(set(names) - {b})
        rows[b] = {"cols": blocks[b], "k": len(blocks[b]), "r2_alone": r2_of({b}), "delta_r2": full - without,
                   "partial_r2": (full - without) / (1 - without) if without < 1 else float("nan"),
                   "shapley": sh[b]}
    return {"n": n, "k": k, "r2": full, "adj_r2": adj, "iid_null_mean_r2": k / (n - 1), "blocks": rows}


def block_null(y: np.ndarray, df: pd.DataFrame, blocks: dict[str, list[str]]) -> dict:
    """S를 21~(N−21) 원형 이동한 귀무에서 전체 R²·블록 부분 R²의 분포. p = (1 + 귀무 ≥ 관측) ÷ (1 + K)."""
    obs = block_table(y, df, blocks)
    ks = circ_shifts(len(y))
    nulls = [block_table(np.roll(y, k), df, blocks) for k in ks]
    out = {"K": len(ks), "min_p": 1 / (1 + len(ks)),
           "r2": {"null_median": float(np.median([t["r2"] for t in nulls])),
                  "null_p90": float(np.quantile([t["r2"] for t in nulls], 0.9)),
                  "p": float((1 + sum(t["r2"] >= obs["r2"] for t in nulls)) / (1 + len(ks)))}}
    for b in blocks:
        v = [t["blocks"][b]["partial_r2"] for t in nulls]
        out[b] = {"null_median": float(np.median(v)), "null_p90": float(np.quantile(v, 0.9)),
                  "p": float((1 + sum(x >= obs["blocks"][b]["partial_r2"] for x in v)) / (1 + len(ks)))}
    return out


def r_min_for_p(n: int, alpha: float = GATE_P, k: int = 2) -> float:
    """iid일 때 양측 p < alpha가 되는 최소 |r|(검출 한계 대신 보이는 값)."""
    t = stats.t.ppf(1 - alpha / 2, n - k)
    return float(t / math.sqrt(n - k + t * t))


def ac1(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


# ---------------------------------------------------------------- 창 고르기
def overlaps_days(open_kst: pd.Series, close_kst: pd.Series, days: list[date]) -> np.ndarray:
    """창 [open, close)가 달력일 d의 [d 00:00, d+1 00:00) 하나라도와 겹치면 True."""
    m = np.zeros(len(open_kst), bool)
    for d in days:
        a = pd.Timestamp(d)
        b = a + pd.Timedelta(days=1)
        m |= (open_kst.to_numpy() < b.to_datetime64()) & (close_kst.to_numpy() > a.to_datetime64())
    return m


# ---------------------------------------------------------------- 조립
def build_matrix(win: pd.DataFrame, px: dict[str, pd.DataFrame], us_sessions: list[date],
                 kr_cal: set, us_cal: set, cal_last: date, earn: dict) -> tuple[pd.DataFrame, dict]:
    """us_sessions = 가격 봉이 있는 미국 세션(r(u)용), kr_cal·us_cal = 로더 거래일 달력(만기일 옮기기용)."""
    kf = session_features(px["kospi"]["Close"], px["kospi"]["Volume"])
    hf = session_features(px["hynix"]["Close"])
    sf = session_features(px["sox"]["Close"], lags=0)
    ef = session_features(px["ewy"]["Close"], lags=0)
    m = win.copy()
    Ps = [pd.Timestamp(p).date() for p in m["P"]]
    Ds = [pd.Timestamp(d).date() for d in m["D"]]
    guard_past(Ps)
    for p in Ps:
        if p not in kf.index or p not in hf.index:
            raise RuntimeError(f"P {p}: KRX 세션 가격 없음")
    for c in ["r", "rpos", "rneg", "r5", "r20", "r60", "DD20", "lnRV20", "ln_vol_anom"] + [f"r_lag{k}" for k in range(1, LAGS + 1)]:
        m[f"kospi_{c}"] = [kf.at[p, c] for p in Ps]
    for c in ["r", "rpos", "rneg", "r5", "r20", "r60", "DD20", "lnRV20"] + [f"r_lag{k}" for k in range(1, LAGS + 1)]:
        m[f"hynix_{c}"] = [hf.at[p, c] for p in Ps]
    us = [last_us_session(D, us_sessions) for D in Ds]
    guard_past(us)
    m["us_session"] = [u.isoformat() for u in us]
    m["sox_r_u"] = [sf.at[u, "r"] for u in us]
    m["ewy_r_u"] = [ef.at[u, "r"] for u in us]

    # 달력·품질 더미(§2.4): 창 종류, 휴장 엇갈림, 방학 달, 예정 일정
    months = sorted({(d.year, d.month) for d in Ps + Ds})
    kr_opex = [to_prev_session(nth_weekday(y, mo, 3, 2), kr_cal, cal_last) for y, mo in months]   # 둘째 목요일
    us_opex = [to_prev_session(nth_weekday(y, mo, 4, 3), us_cal, cal_last) for y, mo in months]   # 셋째 금요일
    earn_days = sorted({d for v in earn.values() if isinstance(v, list) for d in v})
    m["cal_weekend"] = (m["kind"] == "weekend").astype(float)
    m["cal_holiday"] = (m["kind"] == "holiday").astype(float)
    m["cal_xholiday"] = m["xholiday_mismatch"].astype(float)
    m["cal_vacation"] = [float(d.month in (1, 2, 7, 8)) for d in Ds]
    m["cal_kr_opex"] = [float(any(in_window(e, P, D) for e in kr_opex)) for P, D in zip(Ps, Ds)]
    m["cal_us_opex"] = [float(any(in_window(e, P, D) for e in us_opex)) for P, D in zip(Ps, Ds)]
    m["cal_fomc"] = [float(any(in_window(e, P, D) for e in FOMC_2026)) for P, D in zip(Ps, Ds)]
    m["cal_earn"] = [float(any(in_window(e, P, D) for e in earn_days)) for P, D in zip(Ps, Ds)]
    info = {"kr_opex": [str(d) for d in kr_opex], "us_opex": [str(d) for d in us_opex],
            "fomc_in_range": [str(d) for d in FOMC_2026 if min(Ps) <= d <= max(Ds)],
            "earn_in_range": {t: ([str(d) for d in v if min(Ps) <= d <= max(Ds)] if isinstance(v, list) else v)
                              for t, v in earn.items()}}
    return m, info


def gate_cells(df: pd.DataFrame, samples: dict[str, np.ndarray]) -> dict:
    """KOSPI·하닉 칸마다 표본별 Pearson·Spearman과 HAC(5) p, 원형 이동 p."""
    out = {}
    for asset in ("kospi", "hynix"):
        out[asset] = {}
        for nm, mask in samples.items():
            d = df[mask]
            s, r, pos = d["S"].to_numpy(float), d[f"{asset}_r"].to_numpy(float), d["pos"].to_numpy(int)
            row = {}
            for meth in ("pearson", "spearman"):
                c = corr_test(s, r, pos, meth)
                c["circ"] = circ_p_corr(s, r, meth)
                row[meth] = c
            out[asset][nm] = row
    return out


def judge(cells: dict, main: str = "main", excl: str = "excl_crash_union") -> dict:
    """§2.7 관문 조건별 충족 여부(미리보기). 칸 = KOSPI 또는 하닉."""
    res = {}
    for asset, v in cells.items():
        a, x = v[main], v[excl]
        cond = {
            "pearson_ge_0.15": a["pearson"]["rho"] >= GATE_R,
            "spearman_ge_0.15": a["spearman"]["rho"] >= GATE_R,
            "pearson_hac_p_two_lt_0.05": a["pearson"]["p_two"] < GATE_P,
            "spearman_hac_p_two_lt_0.05": a["spearman"]["p_two"] < GATE_P,
            "excl_crash_pearson_ge_0.15": x["pearson"]["rho"] >= GATE_R,
            "excl_crash_spearman_ge_0.15": x["spearman"]["rho"] >= GATE_R,
        }
        res[asset] = {"conditions": cond, "all_met": all(cond.values()),
                      "p_one_side_would_change": (a["pearson"]["p_one_pos"] < GATE_P) != cond["pearson_hac_p_two_lt_0.05"]
                      or (a["spearman"]["p_one_pos"] < GATE_P) != cond["spearman_hac_p_two_lt_0.05"]}
    res["gate_position"] = "충족" if any(v["all_met"] for k, v in res.items() if k in cells) else "미달"
    return res


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    frozen = {n: sha256(ANT / n) for n in ("HYPOTHESES.md", "HYPOTHESES_addendum1.md", "lexicon_draft.md")}
    ft_path = RESULTS / "feat_text.parquet"
    ft = pd.read_parquet(ft_path)
    win = ft[(ft["market"] == "kr")].sort_values("D").reset_index(drop=True)
    win = win[["win_id", "D", "P", "open_kst", "close_kst", "length_h", "kind", "over96", "xholiday_mismatch",
               "valid", "outage_flag", "outage_unknown", "missing_rate", "S", "E_A_P", "F_A_P", "E_A_C", "F_A_C",
               "S_P", "S_C"]].copy()
    win["pos"] = np.arange(len(win))                       # 원래 창 순서(HAC 지연 짝, 원형 이동용)
    if win["D"].max() > pd.Timestamp(LAST_PRICE_DAY + timedelta(days=1)):
        raise RuntimeError("창이 10/15 뒤까지 있다")

    cal = pd.read_parquet(RESULTS / "loader_calendar.parquet")
    kr_sessions = {pd.Timestamp(d).date() for d in cal.loc[cal["krx"], "date"]}
    px = {"kospi": fetch_daily(TICKERS["kospi"], want_volume=True), "hynix": fetch_daily(TICKERS["hynix"]),
          "sox": fetch_daily(TICKERS["sox"]), "ewy": fetch_daily(TICKERS["ewy"])}
    us_sessions = sorted(set(px["sox"].index) & set(px["ewy"].index))
    nyse_cal = {pd.Timestamp(d).date() for d in cal.loc[cal["nyse"], "date"]}
    us_cal_check = sorted(set(d for d in nyse_cal if PRICE_START <= d <= LAST_PRICE_DAY) ^ set(us_sessions))
    kr_cal_check = sorted(set(d for d in kr_sessions if PRICE_START <= d <= LAST_PRICE_DAY) ^ set(px["kospi"].index))
    earn = fetch_earnings()
    cal_last = pd.Timestamp(cal["date"].max()).date()
    m, cal_info = build_matrix(win, px, us_sessions, kr_sessions, nyse_cal, cal_last, earn)

    # 표본: 판정 자격 = 유효 창이면서 96시간 초과가 아닌 창(§2.1), 국내 장애 창은 없음
    elig = (m["valid"].astype(bool) & ~m["over96"].astype(bool) & ~m["outage_flag"].fillna(False).astype(bool)).to_numpy()
    crash_u = overlaps_days(m["open_kst"], m["close_kst"], CRASH_DAYS)
    crash_P = np.array([pd.Timestamp(p).date() in CRASH_DAYS for p in m["P"]])
    crash_D = np.array([pd.Timestamp(d).date() in CRASH_DAYS for d in m["D"]])
    opened = overlaps_days(m["open_kst"], m["close_kst"], OPENED_DAYS)
    need = ["S", "kospi_r", "hynix_r"]
    ok = elig & m[need].notna().all(axis=1).to_numpy()
    samples = {"main": ok, "excl_crash_union": ok & ~crash_u, "excl_crash_P": ok & ~crash_P,
               "excl_crash_D": ok & ~crash_D, "excl_opened8": ok & ~opened}
    # 기여가 가장 큰 2창(보조): Pearson 기여 z_S·z_r가 추정 부호 쪽으로 가장 큰 2창
    top2 = {}
    for asset in ("kospi", "hynix"):
        d = m[ok]
        c = zs(d["S"].to_numpy(float)) * zs(d[f"{asset}_r"].to_numpy(float))
        rho = np.corrcoef(d["S"], d[f"{asset}_r"])[0, 1]
        idx = np.argsort(-np.sign(rho) * c)[:2]
        drop = set(d["win_id"].iloc[idx])
        top2[asset] = sorted(drop)
        samples[f"excl_top2_{asset}"] = ok & ~m["win_id"].isin(drop).to_numpy()
    # 창별 Pearson 기여(z_S·z_r) 위·아래 3창(설명용)
    contrib = {}
    for asset in ("kospi", "hynix"):
        d = m[ok]
        c = zs(d["S"].to_numpy(float)) * zs(d[f"{asset}_r"].to_numpy(float)) / (len(d) - 1)
        o = np.argsort(c)
        contrib[asset] = {"most_negative": [[d["win_id"].iloc[i], float(c[i])] for i in o[:3]],
                          "most_positive": [[d["win_id"].iloc[i], float(c[i])] for i in o[::-1][:3]]}
    cells = gate_cells(m, samples)
    # 하닉 칸은 자기 top2만, KOSPI 칸은 자기 top2만 남긴다
    for asset in ("kospi", "hynix"):
        other = "hynix" if asset == "kospi" else "kospi"
        cells[asset].pop(f"excl_top2_{other}")
    verdict = judge(cells)

    # 결번율 통제판(H01 '결번율을 통제한다'): 부분 상관
    d0 = m[ok]
    partial_mr = {}
    for asset in ("kospi", "hynix"):
        partial_mr[asset] = {meth: corr_test(d0["S"].to_numpy(float), d0[f"{asset}_r"].to_numpy(float),
                                             d0["pos"].to_numpy(int), meth,
                                             controls=d0[["missing_rate"]].to_numpy(float))
                             for meth in ("pearson", "spearman")}
        dx = m[samples["excl_crash_union"]]
        partial_mr[asset]["excl_crash_union"] = {meth: corr_test(dx["S"].to_numpy(float), dx[f"{asset}_r"].to_numpy(float),
                                                                 dx["pos"].to_numpy(int), meth,
                                                                 controls=dx[["missing_rate"]].to_numpy(float))
                                                 for meth in ("pearson", "spearman")}

    # 보조 상관: 미국 밤, 가격 수준, E·F 성분, 지연 1~20
    def ctab(col_y: str, col_x: str, mask=ok) -> dict:
        d = m[mask & m[[col_y, col_x]].notna().all(axis=1).to_numpy()]
        return {meth: corr_test(d[col_y].to_numpy(float), d[col_x].to_numpy(float), d["pos"].to_numpy(int), meth)
                for meth in ("pearson", "spearman")}

    aux = {x: ctab("S", x) for x in ["sox_r_u", "ewy_r_u", "kospi_r5", "kospi_r20", "kospi_r60", "kospi_DD20",
                                     "kospi_lnRV20", "kospi_ln_vol_anom", "hynix_r5", "hynix_r20", "missing_rate"]}
    comp = {y: {a: ctab(y, f"{a}_r") for a in ("kospi", "hynix")} for y in ["E_A_P", "F_A_P", "E_A_C", "F_A_C", "S_P", "S_C"]}
    lags = {a: {k: ctab("S", f"{a}_r_lag{k}") for k in range(1, LAGS + 1)} for a in ("kospi", "hynix")}

    # M0 블록별 부분 R²(§2.3 한국 M0; |r(D−1)|은 max·min의 차라 뺀다, VKOSPI는 못 받음)
    def m0_blocks(mask) -> tuple[dict, list[str]]:
        d = m[mask]
        cal_cols = []
        dropped = []
        for c in CAL_ALL:
            v = d[c].to_numpy(float)
            if np.nanstd(v) == 0:
                dropped.append(f"{c}(표본 안 상수)")
            elif c.startswith("cal_") and np.nansum(v) < DUMMY_MIN_ONES:
                dropped.append(f"{c}(1이 {int(np.nansum(v))}개)")
            else:
                cal_cols.append(c)
        return {"kr_price": KR_PRICE, "us_night": US_NIGHT, "voltrade": VOLTRADE, "cal": cal_cols}, dropped

    m0 = {}
    for nm in ("main", "excl_crash_union"):
        mask = samples[nm] & m[KR_PRICE + US_NIGHT + VOLTRADE + CAL_ALL].notna().all(axis=1).to_numpy()
        blocks, dropped = m0_blocks(mask)
        y = m.loc[mask, "S"].to_numpy(float)
        d = m[mask]
        tab = block_table(y, d, blocks)
        tab["dropped"] = dropped
        tab["null"] = block_null(y, d, blocks)
        # 보조: 하닉 가격 4열을 M0에 더할 때
        b2 = dict(blocks, hynix=HYNIX_EXTRA)
        tab2 = block_table(y, d, b2)
        tab["hynix_extra"] = tab2["blocks"]["hynix"] | {"r2_with": tab2["r2"], "adj_r2_with": tab2["adj_r2"]}
        m0[nm] = tab

    n_main = int(ok.sum())
    res = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "status": "미리보기(판정 아님). 최종 판정은 수집 완료 뒤 2026 전체 창",
        "frozen_sha256": frozen,
        "frozen_match": all(frozen[k_] == FROZEN_EXPECTED[k_] for k_ in FROZEN_EXPECTED),
        "kospi_zero_volume_days": [str(d) for d, v_ in px["kospi"]["Volume"].items() if v_ == 0],
        "inputs": {"feat_text_sha256": sha256(ft_path), "rows_kr": int(len(m))},
        "prices": {k: {"ticker": TICKERS[k], "first": str(min(v.index)), "last": str(max(v.index)), "n": int(len(v))}
                   for k, v in px.items()},
        "calendar_check": {"kr_sessions_vs_kospi_bars_diff": [str(d) for d in kr_cal_check],
                           "nyse_vs_us_bars_diff": [str(d) for d in us_cal_check]},
        "calendar_info": cal_info,
        "samples": {k: {"n": int(v.sum()), "dropped": sorted(m.loc[ok & ~v, "win_id"].tolist())} for k, v in samples.items()},
        "excluded_from_main": sorted(m.loc[~ok, "win_id"].tolist()),
        "top2": top2,
        "contrib": contrib,
        "cells": cells,
        "verdict_preview": verdict,
        "partial_missing_rate": partial_mr,
        "aux": aux, "components": comp, "lags": lags,
        "m0": m0,
        "ac1": {"S": ac1(m.loc[ok, "S"].to_numpy(float)), "kospi_r": ac1(m.loc[ok, "kospi_r"].to_numpy(float)),
                "hynix_r": ac1(m.loc[ok, "hynix_r"].to_numpy(float))},
        "r_min_iid_p05": {"main": r_min_for_p(n_main), "excl_crash_union": r_min_for_p(int(samples["excl_crash_union"].sum())),
                          "full_2026_approx_195": r_min_for_p(195)},
        "outage_unknown_in_main": int(m.loc[ok, "outage_unknown"].fillna(False).astype(bool).sum()),
    }
    (RESULTS / "h01_preview.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    cols = (["win_id", "D", "P", "us_session", "kind", "over96", "valid", "S", "E_A_P", "F_A_P", "E_A_C", "F_A_C",
             "kospi_r", "hynix_r", "hynix_rpos", "hynix_rneg", "hynix_r5", "hynix_r20"]
            + KR_PRICE + US_NIGHT + VOLTRADE + CAL_ALL)
    mm = m[cols].copy()
    mm["in_main"], mm["in_excl_crash_union"] = samples["main"], samples["excl_crash_union"]
    mm.to_parquet(RESULTS / "h01_m0_matrix.parquet", index=False)
    import h01_preview_report as R
    R.write_md(json.loads((RESULTS / "h01_preview.json").read_text(encoding="utf-8")), RESULTS / "h01_preview.md")
    print(json.dumps({"n": n_main, "verdict": verdict, "kospi_main": {k: round(v["rho"], 3) for k, v in cells["kospi"]["main"].items()},
                      "hynix_main": {k: round(v["rho"], 3) for k, v in cells["hynix"]["main"].items()},
                      "m0_r2": round(m0["main"]["r2"], 3)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
