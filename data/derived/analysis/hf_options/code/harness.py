"""헤지펀드 × 옵션 종합 연구(2026-10-14, 3단계) 공용 자료·채점. 모든 갈래가 이 파일로 같은 값을 읽는다.

시점 규칙: 패널의 색인은 '진입일' d다. d의 값은 d 시가 전에 공개된 관측만 쓴다.
  - 장 마감 뒤(그날 저녁·다음 날 새벽) 공개되는 일별 자료(CBOE·OCC·FINRA 공매도 거래량·DIX/GEX·Yahoo 종가): 측정일 다음 거래일부터.
  - 장중·장 마감 전 공개(CFTC 금요일 15:30 ET 등): known(그날 종가까지 공개된 날) 다음 거래일부터.
결과: R_h(d) = 종가(d+h−1) ÷ 시가(d) − 1, M_h(d) = [시가(d), 종가(d) … 종가(d+h−1)] 경로의 고점 대비 최저(%).
백분위: 각 원자료 자기 빈도로 지난 N개(일별 250, 주간 52, 월 2회 24) 안 순위(자기 포함, 최소 절반)를 구한 뒤 진입일로 옮긴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
CACHE = STUDY / "cache"
RES = STUDY / "results"
ROOT = HERE.parents[4]
OS = ROOT / "data/derived/analysis/options_sentiment"
sys.path.insert(0, str(OS / "code"))

PERIODS = {"front": ("2026-01-02", "2026-05-29"), "back": ("2026-06-01", "2026-12-31"), "y2025": ("2025-01-02", "2025-12-31")}
HORIZONS = (5, 10, 21)
MIN_SHIFT = 21


# ---------- 가격·거래일 ----------
def prices() -> pd.DataFrame:
    """QQQ·SOXX 시가·종가(분할 반영). 마지막 완료 거래일까지."""
    px = pd.read_parquet(OS / "cache/yahoo_ohlc.parquet")
    last = pd.read_parquet(OS / "cache/yahoo.parquet").dropna(subset=["^SKEW"]).index.max()  # 지수 종가가 있는 마지막 완료일
    return px[px.index <= last].dropna()


def days() -> pd.Index:
    return prices().index


def outcomes(asset: str = "QQQ") -> pd.DataFrame:
    px = prices()
    o, c = px[f"{asset}_o"].to_numpy(), px[f"{asset}_c"].to_numpy()
    n = len(c)
    out = {}
    for h in HORIZONS:
        r, m = np.full(n, np.nan), np.full(n, np.nan)
        for i in range(n - h + 1):
            path = np.concatenate([[o[i]], c[i:i + h]])
            r[i] = (path[-1] / path[0] - 1) * 100
            m[i] = (path / np.maximum.accumulate(path) - 1).min() * 100
        out[f"R{h}"], out[f"M{h}"] = r, m
    return pd.DataFrame(out, index=px.index)


def rolling_pct(v: pd.Series, win: int) -> pd.Series:
    v = v.dropna()
    return v.rolling(win, min_periods=max(win // 2, 3)).apply(lambda w: (w <= w[-1]).mean(), raw=True)


# ---------- 기간 ----------
def period_mask(idx: pd.Index, name: str) -> np.ndarray:
    lo, hi = PERIODS[name]
    return np.asarray((idx >= lo) & (idx <= hi))


# ---------- 통계 ----------
def welch(sig: np.ndarray, y: np.ndarray, min_sig: int = 3) -> tuple[float, float, int]:
    m = ~np.isnan(sig) & ~np.isnan(y)
    s, v = sig[m] > 0, y[m]
    if s.sum() < min_sig or (~s).sum() < min_sig:
        return np.nan, np.nan, int(s.sum())
    a, b = v[s], v[~s]
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(d), float(d / se) if se > 0 else np.nan, int(s.sum())


def shift_p(sig: np.ndarray, y: np.ndarray, side: str = "less") -> float:
    """신호열 원형 이동(21거래일 이상, 모든 이동) 귀무분포에서 관측 차이 이상으로 치우친 비율. side='less'면 신호 뒤가 더 나쁨."""
    m = ~np.isnan(sig) & ~np.isnan(y)
    s, v = sig[m] > 0, y[m]
    n = len(s)
    if s.sum() < 3 or n < 3 * MIN_SHIFT:
        return np.nan
    obs = v[s].mean() - v[~s].mean()
    null = []
    for k in range(MIN_SHIFT, n - MIN_SHIFT + 1):
        r = np.roll(s, k)
        if 0 < r.sum() < n:
            null.append(v[r].mean() - v[~r].mean())
    null = np.array(null)
    hit = (null <= obs) if side == "less" else (null >= obs)
    return float((hit.sum() + 1) / (len(null) + 1))


def grid_diffs(S: np.ndarray, Y: np.ndarray, mask: np.ndarray, min_sig: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """여러 신호(S: 날×칸, 0/1/NaN)와 결과 하나(Y: 날)의 차이·Welch t·신호일 수를 한 번에. mask 밖은 무시."""
    S = np.where(mask[:, None], S, np.nan)
    ok = ~np.isnan(S) & ~np.isnan(Y)[:, None]
    sig = (np.nan_to_num(S) > 0) & ok
    rest = ok & ~sig
    y = np.nan_to_num(Y)[:, None]
    n1, n0 = sig.sum(0), rest.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1, m0 = (y * sig).sum(0) / n1, (y * rest).sum(0) / n0
        v1 = ((y - m1) ** 2 * sig).sum(0) / (n1 - 1)
        v0 = ((y - m0) ** 2 * rest).sum(0) / (n0 - 1)
        d = m1 - m0
        t = d / np.sqrt(v1 / n1 + v0 / n0)
    bad = (n1 < min_sig) | (n0 < min_sig)
    d[bad], t[bad] = np.nan, np.nan
    return d, t, n1


def save(name: str, obj) -> None:
    RES.mkdir(exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


# ---------- 신호 (BRIEF 24개) ----------
DAILY_AGE, WEEKLY_AGE, BIMONTH_AGE = 3, 15, 25
SIGNALS = ["eq_pcr5", "ai_cpcr5", "ai_breadth5", "skew", "vix_ts", "gex", "lev_qqq_bull5", "lev_soxx_bull5", "aaii_spread",
           "nq_lev_net_oi", "nq_lev_chg4", "nq_am_net_oi", "nq_am_chg4", "nq_dealer_net_oi", "es_lev_net_oi", "vx_lev_net_oi",
           "nq_nonrept_net_oi", "dix", "gvip_rel20", "qai_beta60", "shvr_ai5", "si_ai_chg", "naaim", "lev_qqq_flow20p"]
GROUP = {s: ("헤지펀드·큰손" if s.startswith(("nq_lev", "nq_am", "nq_dealer", "es_", "vx_", "dix", "gvip", "qai", "shvr", "si_", "naaim"))
             else "옵션·개인") for s in SIGNALS}


def _place(pct: pd.Series, when: pd.Series, dd: pd.Index, age: int) -> pd.Series:
    """백분위(pct, 원자료 순서)를 when 다음 거래일부터 age거래일까지 쓴다."""
    pct = pct.dropna()
    when = when.reindex(pct.index)
    pos = np.searchsorted(dd.to_numpy(), when.to_numpy(), side="right")
    s = pd.Series(pct.to_numpy(), index=pos)
    s = s[s.index < len(dd)]
    s = s[~s.index.duplicated(keep="last")]
    full = pd.Series(np.nan, index=range(len(dd)))
    full.loc[s.index] = s.to_numpy()
    return pd.Series(full.ffill(limit=age).to_numpy(), index=dd)


def build_panel() -> pd.DataFrame:
    """진입일 × 신호 백분위(0~1, 값이 클수록 원값이 큼). 결과는 cache/panel.parquet."""
    import fetch as osf  # options_sentiment 수집기
    import lag as osl
    dd = days()
    out = {}
    # 옵션(장 마감 뒤 공개 → 측정일 다음 거래일)
    cb = osf.load_cboe()
    w = pd.Series(cb.index, index=cb.index)
    eq = cb["equity_pv"].rolling(5).sum() / cb["equity_cv"].rolling(5).sum()
    out["eq_pcr5"] = _place(rolling_pct(eq, 250), w, dd, DAILY_AGE)
    ind = osl.indicators(cb.index)  # AI_5·AIB_5는 이미 250일 백분위(AI는 환호 방향 = 풋/콜 낮음)
    out["ai_cpcr5"] = _place(1 - ind["AI_5"], w, dd, DAILY_AGE)
    out["ai_breadth5"] = _place(ind["AIB_5"], w, dd, DAILY_AGE)
    y = pd.read_parquet(OS / "cache/yahoo.parquet").dropna(subset=["QQQ"])
    wy = pd.Series(y.index, index=y.index)
    out["skew"] = _place(rolling_pct(y["^SKEW"], 250), wy, dd, DAILY_AGE)
    out["vix_ts"] = _place(rolling_pct(y["^VIX"] / y["^VIX3M"], 250), wy, dd, DAILY_AGE)
    sq = pd.read_parquet(CACHE / "squeeze.parquet")
    sq = sq.set_index("asof")
    ws = pd.Series(sq.index, index=sq.index)
    out["gex"] = _place(rolling_pct(sq["gex_bn"], 250), ws, dd, DAILY_AGE)
    out["dix"] = _place(rolling_pct(sq["dix"], 250), ws, dd, DAILY_AGE)
    # Yahoo 가격 지표(known 그날 종가 → 다음 거래일)
    hp = pd.read_parquet(CACHE / "hf_proxies.parquet")
    wk = pd.Series(hp.index, index=hp.index)
    for s, c in (("lev_qqq_bull5", "lev_qqq_bull5"), ("lev_soxx_bull5", "lev_soxx_bull5"), ("gvip_rel20", "gvip_rel20"), ("qai_beta60", "qai_beta60")):
        out[s] = _place(rolling_pct(hp[c], 250), wk, dd, DAILY_AGE)
    # ProShares 흐름(측정일 asof 저녁 공개 → asof 다음 거래일)
    fl = hp.dropna(subset=["lev_qqq_flow20"]).set_index("asof")
    flow = fl["lev_qqq_flow20"] / (fl["aum_tqqq"] + fl["aum_sqqq"])
    out["lev_qqq_flow20p"] = _place(rolling_pct(flow, 250), pd.Series(flow.index, index=flow.index), dd, DAILY_AGE)
    # FINRA 공매도 거래량(장 마감 뒤 → asof 다음 거래일, 개장 전 게시가 아니면 known 다음 거래일)
    sv = pd.read_parquet(CACHE / "finra_shvol.parquet")
    sv5 = sv.set_index("asof")["shvr_ai_eq"].rolling(5).mean()
    wsv = pd.Series(np.where(sv["open_ok"].to_numpy() == 1, sv["asof"].to_numpy(), sv.index.to_numpy()), index=sv["asof"].to_numpy())
    out["shvr_ai5"] = _place(rolling_pct(sv5, 250), wsv.reindex(sv5.index), dd, DAILY_AGE)
    # CFTC(주간, known 다음 거래일)
    cf = pd.read_parquet(CACHE / "cftc_tff.parquet")
    wc = pd.Series(cf.index, index=cf.index)
    nq_lev = cf["nq_lev_net"] / cf["nq_oi"]
    nq_am = cf["nq_am_net"] / cf["nq_oi"]
    series = {"nq_lev_net_oi": nq_lev, "nq_lev_chg4": nq_lev.diff(4), "nq_am_net_oi": nq_am, "nq_am_chg4": nq_am.diff(4),
              "nq_dealer_net_oi": cf["nq_dealer_net"] / cf["nq_oi"], "es_lev_net_oi": cf["es_lev_net"] / cf["es_oi"],
              "vx_lev_net_oi": cf["vx_lev_net"] / cf["vx_oi"], "nq_nonrept_net_oi": cf["nq_nonrept_net"] / cf["nq_oi"]}
    for s, v in series.items():
        out[s] = _place(rolling_pct(v, 52), wc, dd, WEEKLY_AGE)
    # 공매도 잔고(월 2회, known 다음 거래일)
    si = pd.read_parquet(CACHE / "finra_si.parquet")
    out["si_ai_chg"] = _place(rolling_pct(si["si_ai_chg"], 24), pd.Series(si.index, index=si.index), dd, BIMONTH_AGE)
    # 설문(주간)
    sv_ = pd.read_parquet(CACHE / "surveys.parquet")
    a = sv_["aaii_spread"].dropna()
    out["aaii_spread"] = _place(rolling_pct(a, 52), pd.Series(a.index, index=a.index), dd, WEEKLY_AGE)
    n = sv_["naaim"].dropna()
    out["naaim"] = _place(rolling_pct(n, 52), pd.Series(n.index, index=n.index), dd, WEEKLY_AGE)
    P = pd.DataFrame(out, index=dd)[SIGNALS]
    P.to_parquet(CACHE / "panel.parquet")
    return P


def panel() -> pd.DataFrame:
    f = CACHE / "panel.parquet"
    return pd.read_parquet(f) if f.exists() else build_panel()
