"""8단계 갈래 FGDIRECT(2026-10-11): CNN 공포탐욕지수(F&G)를 그대로 매매 참고로 쓰면 '아무 날 사기'·'덜 담기'보다 나은가.

사용자: "그럼 CNN의 공포탐욕지수를 바탕으로 매매에 참고하면? 아니면 이런 것들에서 아이디어 얻어서 통하는 방법을 개발해도 좋고요."

자료
  F&G 실제: fg_data.fg() 2011-01-03~2026-10-09(2021-01까지 GitHub 보관본 정수, 그 뒤 CNN 공식). 그날 미국 장 마감 뒤 확정.
  F&G 재구성(이 갈래 안에서만 쓰는 대리값, 1991-03~): CNN 7요소 중 공개 가격으로 1990년대부터 만들 수 있는 5개(사전 고정, 맞춤 없음).
    ① 모멘텀 = S&P500/125일선 − 1   ② 신고가 강도 대리 = S&P500의 252일 고저 범위 안 위치
    ③ 변동성 = −(VIX/50일선 − 1)    ④ 정크 수요 대리 = 하이일드 펀드 VWEHX 20일 수익 − 5년 듀레이션 국채(^TNX로 근사) 20일 수익
    ⑤ 안전자산 수요 = S&P500 20일 수익 − 장기국채 펀드 VUSTX 20일 수익
    (풋콜 비율·맥클렐런 폭·NYSE 신고가 수는 1990년대 공개 자료가 없어 뺐다. FRED는 이 환경에서 응답이 없어 펀드 기준가로 대신했다.)
    요소마다 지난 504거래일(최소 252) 안 백분위 → 5개 평균 → 2011~2018 실제 F&G 분포에 분위 맞춤(값 분포만 맞추고 시점·수익은 안 씀).
    펀드 기준가(VWEHX·VUSTX)는 하루 늦춰 쓴다(공개가 장 마감 뒤라 보수적으로).
  가격: cache/p7_idx.parquet(SPY·QQQ·EWY·^GSPC·^NDX·^SOX·^VIX), lev5(SOXX·TQQQ·SOXL·KORU 실제, 3배 합성), harness.base(7종목),
        cache/p4_stocks.parquet(일반 주식 2014~, 생존편향). 새 자료는 cache/p8_FGDIRECT_yf.parquet(VWEHX·VUSTX·^TNX, yfinance).

시점(사전 고정)
  F&G는 그날 장 마감 뒤 확정 → 다음 거래일 시가 체결(harness.at_open과 같은 '엄격히 앞선 날짜' 규칙).
  실제 F&G 트랙(2011~)은 ETF·7종목 모두 시가 체결·시가 청산. 재구성 트랙(1991~)은 옛 지수 시가가 없어 '다음 날 종가 체결'.
  하루 늦춘 판(신호 다음다음 거래일 체결)을 대표 규칙마다 한 줄로 보인다.

사건형(그날 종가 F&G → 다음 날 시가 매수 → H거래일 보유 → H거래일 뒤 시가 매도)
  E1_T   레벨 진입: F&G ≤ T 처음(직전에 T+10 위였던 뒤에만 새 사건), T ∈ {5,10,15,20,25,30}
  E2_T_δ 공포 탈출: T 이하에 있다가 T+δ 위로 올라선 첫날, T ∈ {10,15,20,25}, δ ∈ {0,5,10}
  E3_G   탐욕 경고: F&G ≥ G 처음(직전 G−10 아래였던 뒤), G ∈ {70,75,80,85,90}. 덜기 근거가 되려면 아무 날보다 나빠야 한다.
  E4_T   매일 판: F&G ≤ T인 모든 날, T ∈ {10,20,25}
  기준: B200 = 그 대상 종가가 200일선 아래인 모든 날(단순 추세 기준), ANY = 아무 날.
  H ∈ {5,21,63,126,252}거래일. 통계: 평균 수익, 아무 날 대비 초과, 승률, 추가 −20% 확률(보유 중 종가·청산가가 진입가 × 0.8 이하), 사건 수,
  겹치지 않는 사건 수. 고르기 창은 청산까지 창 안에서 끝난 거래만 쓴다(뒤 가격 누출 없음).
  대상: SPY·QQQ·SOXX·EWY·TQQQ·SOXL·KORU(2013-04 전은 EWY 3배 합성, 추가 비용 2.6%), 7종목(2026·hold),
        일반 주식(p4 전 종목: 사건일에 모든 종목을 산 평균 − 같은 종목 아무 날 평균; AI 유니버스 미국 상장 부분집합 따로).
  재구성 트랙 대상: ^GSPC(SPY 대리)·^NDX(QQQ)·^SOX→SOXX(SOXX)·EWY·^NDX 3배(TQQQ, 추가 1.3%)·SOX 3배(SOXL, 1.5%)·EWY 3배(KORU, 2.6%).
전략형(매일 F&G를 보고 다음 날 시가에 비중 맞춤, 나머지 현금 0%)
  S1 공포 추가: 평소 w0 ∈ {0.6,0.7,0.8}, F&G ≤ T → 1.0, F&G ≥ T_exit → w0. T ∈ {10,20,25,30}, T_exit ∈ {40,50,60}  (36)
  S2 탐욕 덜기: 평소 1.0, F&G ≥ G → w1 ∈ {0.5,0.75}, F&G ≤ G_back → 1.0. G ∈ {75,80,85}, G_back ∈ {50,60}         (12)
  S3 결합 6개(아래 S3_LIST, 사전 선언). S4 연속형 6개: w = clip(a − b·(F&G−50)/50, lo, 1), 목표와 0.05 이상 다를 때만 맞춤.
  대상: 7종목(2026·hold), QQQ·TQQQ·SOXX·SOXL(실제 2011~), 재구성 트랙 ^NDX·^NDX 3배·SOX·SOX 3배(1991~).
  지표: 효율선초과 = 규칙 − 같은 최대 낙폭보다 깊지 않은 고정 비중(w 0~1, 0.01 간격, 매일 맞춤) 최고 수익(긴 창은 연율 %p),
        같은노출 = 같은 평균 노출 고정 비중 대비. 미끄러짐 0.5%(비중 변화 1단위당), 현금 이자 4% 판(고정 비중도 같은 이자).
고르기(사전 고정)
  긴 역사(실제): 대상마다 2011~2018에서 가장 좋은 규칙(사건형: 사건 15건 이상·초과 최대, 매수 사건 E1·E2·E4만;
    전략형: 비중 변경 2번 이상·효율선초과 최대) → 2019~2025 확인(> 0이면 통과).
  긴 역사(재구성): 같은 방법, 1991-04~2010 고르기 → 2011~2025 확인.
  2026: 전략형 대상마다 2026 앞(1/2~5/29) 효율선초과 최대(앞에 비중 변경 1번 이상) → 뒤(6/1~10/9)·7월 쇼크·2025 확인.
  대표 후보: 긴 역사(실제) 확인 통과 + 2026 뒤 플러스인 고른 규칙 중 확인 구간 지표 최대(전략형·사건형 따로).
위약: F&G 시계열을 63거래일 이상 원형으로 민 20판(간격 고르게)에서 같은 규칙의 확인 구간 지표 순위,
      그리고 '고르기 → 확인' 절차 전체를 위약 판에서 다시 했을 때의 확인 값 분포.
출력: results/p8_FGDIRECT.md, results/p8_FGDIRECT.json. 실행: PYTHONIOENCODING=utf-8 py p8_FGDIRECT.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import fg_data
import harness as HN
import lev5

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ROOT = HERE.parents[4]
ID = "p8_FGDIRECT"

# ============================================================================ 사전 고정 상수(확인 구간을 본 뒤 바꾸지 않는다)
E1_T = (5, 10, 15, 20, 25, 30)
E1_REARM = 10
E2_T = (10, 15, 20, 25)
E2_D = (0, 5, 10)
E3_G = (70, 75, 80, 85, 90)
E3_REARM = 10
E4_T = (10, 20, 25)
H_LIST = (5, 21, 63, 126, 252)
S1_W0 = (0.6, 0.7, 0.8)
S1_T = (10, 20, 25, 30)
S1_TX = (40, 50, 60)
S2_W1 = (0.5, 0.75)
S2_G = (75, 80, 85)
S2_GB = (50, 60)
S3_LIST = (  # (w0, T, T_exit, G, w1, G_back)
    (0.8, 20, 50, 80, 0.5, 60), (0.8, 25, 50, 75, 0.5, 60), (0.7, 20, 50, 80, 0.4, 60),
    (0.8, 10, 40, 85, 0.6, 50), (0.7, 25, 60, 75, 0.5, 50), (0.9, 20, 50, 80, 0.7, 60))
S4_LIST = ((0.8, 0.2, 0.5), (0.8, 0.4, 0.4), (0.7, 0.3, 0.4), (0.9, 0.2, 0.6), (0.75, 0.5, 0.25), (0.85, 0.3, 0.5))
S4_BAND = 0.05
EXTRA_DD = 0.20
MIN_EV = 15
MIN_TR_LONG = 2
MIN_TR_26 = 1
SLIP = 0.005
CASH4 = 0.04
N_PLA = 20
MIN_SHIFT = 63
W_GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
RECON_WIN, RECON_MIN = 504, 252
FIT = ("2011-01-01", "2018-12-31")
DUR_MID = 5.0
FEE3 = 0.0095
END = "2026-10-09"
FG_START = "2011-01-03"
KORU_START = "2013-04-10"

# 사건형 창: (사건일 시작, 사건일 끝, 청산 한도). 2026앞x = 앞 구간 사건을 끝까지 잰 표시용(고르기에 안 씀)
EVW_REAL = {"sel": ("2011-01-01", "2018-12-31", "2018-12-31"), "chk": ("2019-01-01", "2025-12-31", END),
            "2025": ("2025-01-01", "2025-12-31", END), "2026앞": ("2026-01-02", "2026-05-29", "2026-05-29"),
            "2026앞x": ("2026-01-02", "2026-05-29", END),
            "2026뒤": ("2026-06-01", END, END), "AI쇼크": ("2026-06-22", "2026-07-29", END), "전체": ("2011-01-01", END, END)}
EVW_26 = {"2026전체": ("2026-01-02", END, END), "2026앞": EVW_REAL["2026앞"], "2026앞x": EVW_REAL["2026앞x"], "2026뒤": EVW_REAL["2026뒤"], "AI쇼크": EVW_REAL["AI쇼크"]}
EVW_HOLD = {"hold전체": ("2023-05-22", "2025-12-31", "2025-12-31"), "2025": ("2025-01-01", "2025-12-31", "2025-12-31"),
            "2024": ("2024-01-01", "2024-12-31", "2025-12-31")}
EVW_RECON = {"rsel": ("1991-04-01", "2010-12-31", "2010-12-31"), "rchk": ("2011-01-01", "2025-12-31", END),
             "2000~02": ("2000-03-01", "2002-10-31", END), "2008": ("2007-10-01", "2009-03-31", END)}
# 전략형 창(기간 수익, LONG은 연율)
STW_REAL = {"sel": ("2011-01-03", "2018-12-31"), "chk": ("2019-01-01", "2025-12-31"), "전체": ("2011-01-03", END),
            "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", END), "AI쇼크": ("2026-06-22", "2026-07-29"),
            "2025": ("2025-01-01", "2025-12-31"), "2022": ("2022-01-01", "2022-12-31"), "2020": ("2020-01-01", "2020-12-31")}
STW_26 = {"2026전체": ("2026-01-02", END), "2026앞": STW_REAL["2026앞"], "2026뒤": STW_REAL["2026뒤"], "AI쇼크": STW_REAL["AI쇼크"]}
STW_HOLD = {"hold전체": ("2023-05-22", "2025-12-31"), "2025": ("2025-01-01", "2025-12-31"), "2024": ("2024-01-01", "2024-12-31")}
STW_RECON = {"rsel": ("1991-04-01", "2010-12-31"), "rchk": ("2011-01-01", "2025-12-31"), "2000~02": ("2000-03-01", "2002-10-31"),
             "2008": ("2007-10-01", "2009-03-31"), "2020": ("2020-01-01", "2020-12-31"), "2022": ("2022-01-01", "2022-12-31")}
LONG = {"sel", "chk", "전체", "rsel", "rchk"}

ET_REAL = ("SPY", "QQQ", "SOXX", "EWY", "TQQQ", "SOXL", "KORU")
ET_RECON = {"GSPC": "SPY", "NDX": "QQQ", "SOX": "SOXX", "EWY_r": "EWY", "NDX3": "TQQQ", "SOX3": "SOXL", "EWY3": "KORU"}
ST_REAL = ("QQQ", "TQQQ", "SOXX", "SOXL")
ST_RECON = {"NDX": "QQQ", "NDX3": "TQQQ", "SOX": "SOXX", "SOX3": "SOXL"}
# 표시용(사후, 판정에 쓰지 않음)
DISP_EV = ("E1_10", "E1_20", "E1_25", "E2_20_0", "E4_10", "E4_20", "E4_25", "E3_80")
DISP_H = (21, 63, 126)
DISP_T = ("SPY", "QQQ", "SOXX", "TQQQ", "SOXL")
DISP_TC = ("GSPC", "NDX", "SOX", "NDX3", "SOX3")


# ============================================================================ 규칙 이름
def ev_defs():
    out = {}
    for t in E1_T:
        out[f"E1_{t}"] = f"F&G ≤{t} 첫날"
    for t in E2_T:
        for d in E2_D:
            out[f"E2_{t}_{d}"] = f"≤{t} 뒤 >{t + d} 첫날"
    for g in E3_G:
        out[f"E3_{g}"] = f"F&G ≥{g} 첫날(탐욕)"
    for t in E4_T:
        out[f"E4_{t}"] = f"F&G ≤{t} 매일"
    return out


EV_LABEL = ev_defs()
BUY_EV = [k for k in EV_LABEL if not k.startswith("E3")]


def st_rules():
    out = []
    for w0 in S1_W0:
        for t in S1_T:
            for tx in S1_TX:
                out.append(("S1", w0, t, tx))
    for w1 in S2_W1:
        for g in S2_G:
            for gb in S2_GB:
                out.append(("S2", w1, g, gb))
    for p in S3_LIST:
        out.append(("S3",) + p)
    for p in S4_LIST:
        out.append(("S4",) + p)
    return out


RULES = st_rules()


def rule_ko(r) -> str:
    k = r[0]
    if k == "S1":
        return f"평소 {r[1]:.0%}, F&G≤{r[2]}→100%, ≥{r[3]}→{r[1]:.0%}"
    if k == "S2":
        return f"평소 100%, F&G≥{r[2]}→{r[1]:.0%}, ≤{r[3]}→100%"
    if k == "S3":
        return f"평소 {r[1]:.0%}, ≤{r[2]}→100%(≥{r[3]} 복귀), ≥{r[4]}→{r[5]:.0%}(≤{r[6]} 복귀)"
    return f"비중 = {r[1]:g} − {r[2]:g}×(F&G−50)/50, 하한 {r[3]:.0%}"


def rkey(r) -> str:
    return "_".join(str(x) for x in r)


# ============================================================================ 자료
P7 = pd.read_parquet(CACHE / "p7_idx.parquet")


def p7_ser(sym):
    g = P7[P7.sym == sym].sort_values("date").drop_duplicates("date").dropna(subset=["Close"])
    g = g[g.Close > 0]
    f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy(float)
    c = g.Close.to_numpy(float) * f
    o = g.Open.to_numpy(float)
    o = np.where(np.isfinite(o) & (o > 0), o * f, c)
    return g.date.astype(str).to_numpy(), o, c


def yf_extra() -> pd.DataFrame:
    p = CACHE / f"{ID}_yf.parquet"
    if not p.exists():
        import yfinance as yf
        d = yf.download(["VWEHX", "VUSTX", "^TNX"], start="1985-01-01", end="2026-10-10", auto_adjust=False,
                        progress=False, threads=False)
        out = d["Adj Close"].copy()
        out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
        out.to_parquet(p)
    d = pd.read_parquet(p)
    d.index = d.index.astype(str)
    return d


def pct_rank(s: pd.Series) -> pd.Series:
    """지난 RECON_WIN거래일(최소 RECON_MIN) 안에서 오늘 값의 백분위(0~100)."""
    a = s.to_numpy(float)
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        if not np.isfinite(a[i]):
            continue
        lo = max(0, i - RECON_WIN + 1)
        w = a[lo:i + 1]
        w = w[np.isfinite(w)]
        if len(w) < RECON_MIN:
            continue
        out[i] = 100.0 * ((w < a[i]).sum() + 0.5 * ((w == a[i]).sum() - 1)) / (len(w) - 1)
    return pd.Series(out, index=s.index)


def build_recon(real: pd.Series):
    p = CACHE / f"{ID}_recon.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        d.index = d.index.astype(str)
        return d["recon"], d.drop(columns=["recon"])
    d, _, c = p7_ser("^GSPC")
    spx = pd.Series(c, index=d)
    d, _, c = p7_ser("^VIX")
    vix = pd.Series(c, index=d)
    yx = yf_extra()
    idx = spx.index[spx.index >= "1986-06-01"]
    spx = spx.reindex(idx)
    vix = vix.reindex(idx)
    hy = yx["VWEHX"].reindex(sorted(set(yx.index) | set(idx))).ffill().reindex(idx).shift(1)
    lt = yx["VUSTX"].reindex(sorted(set(yx.index) | set(idx))).ffill().reindex(idx).shift(1)
    tnx = yx["^TNX"].reindex(sorted(set(yx.index) | set(idx))).ffill().reindex(idx) / 100.0
    comp = pd.DataFrame(index=idx)
    comp["momentum"] = spx / spx.rolling(125, min_periods=125).mean() - 1
    hi, lo = spx.rolling(252, min_periods=252).max(), spx.rolling(252, min_periods=252).min()
    comp["strength"] = (spx - lo) / (hi - lo)
    comp["vix"] = -(vix / vix.rolling(50, min_periods=50).mean() - 1)
    bond_mid = -DUR_MID * (tnx - tnx.shift(20)) + tnx.shift(20) * 20 / 252
    comp["junk"] = (hy / hy.shift(20) - 1) - bond_mid
    comp["safe_haven"] = (spx / spx.shift(20) - 1) - (lt / lt.shift(20) - 1)
    ranks = pd.DataFrame({k: pct_rank(comp[k]) for k in comp.columns})
    raw = ranks.mean(axis=1, skipna=False).dropna()
    # 2011~2018 실제 F&G 분포에 분위 맞춤
    ov = sorted(set(raw.index) & set(real.index))
    ov = [x for x in ov if FIT[0] <= x <= FIT[1]]
    rq = np.sort(raw.reindex(ov).to_numpy())
    fq = np.sort(real.reindex(ov).to_numpy())
    qq = np.searchsorted(rq, raw.to_numpy(), side="right") / len(rq)
    rec = pd.Series(np.quantile(fq, np.clip(qq, 0, 1)), index=raw.index, name="recon")
    out = ranks.reindex(rec.index)
    out["recon"] = rec
    out.to_parquet(p)
    return rec, out.drop(columns=["recon"])


# ============================================================================ 사건
def make_events(dates: np.ndarray, v: np.ndarray) -> dict:
    out = {}
    n = len(v)
    for t in E1_T:
        ev, armed = [], v[0] > t + E1_REARM
        for i in range(n):
            if armed and v[i] <= t:
                ev.append(i)
                armed = False
            elif not armed and v[i] > t + E1_REARM:
                armed = True
        out[f"E1_{t}"] = dates[ev]
    for t in E2_T:
        for dlt in E2_D:
            ev, armed = [], False
            for i in range(n):
                if v[i] <= t:
                    armed = True
                elif armed and v[i] > t + dlt:
                    ev.append(i)
                    armed = False
            out[f"E2_{t}_{dlt}"] = dates[ev]
    for g in E3_G:
        ev, armed = [], v[0] < g - E3_REARM
        for i in range(n):
            if armed and v[i] >= g:
                ev.append(i)
                armed = False
            elif not armed and v[i] < g - E3_REARM:
                armed = True
        out[f"E3_{g}"] = dates[ev]
    for t in E4_T:
        out[f"E4_{t}"] = dates[v <= t]
    return out


def shifts(n):
    return np.linspace(MIN_SHIFT, n - MIN_SHIFT, N_PLA).astype(int)


# ============================================================================ 사건형 대상(1개 자산)
class ET:
    def __init__(self, name, days, pb, c, start, mode="open", note=""):
        self.name, self.mode, self.note = name, mode, note
        self.days = np.asarray(days).astype(str)
        self.pb, self.c = np.asarray(pb, float), np.asarray(c, float)
        self.n = len(self.c)
        self.j0 = int(np.searchsorted(self.days, start))
        self.sma = pd.Series(self.c).rolling(200, min_periods=200).mean().to_numpy()
        self.R, self.HIT = {}, {}
        for H in H_LIST:
            R = np.full(self.n, np.nan)
            HIT = np.full(self.n, np.nan)
            if self.n > H + 1:
                m = self.n - H
                R[:m] = self.pb[H:] / self.pb[:m] - 1
                mn = np.minimum(sliding_window_view(self.c, H).min(1)[:m], self.pb[H:])
                HIT[:m] = (mn <= (1 - EXTRA_DD) * self.pb[:m]).astype(float)
            self.R[H], self.HIT[H] = R, HIT
        self._b = {}

    def capi(self, cap):
        return int(np.searchsorted(self.days, cap, side="right")) - 1

    def base(self, H, win):
        key = (H, win)
        if key not in self._b:
            a, z, cap = win
            j0 = max(int(np.searchsorted(self.days, a)), self.j0)
            j1 = int(np.searchsorted(self.days, z, side="right")) - 1
            e = np.arange(j0 + 1, j1 + 2)
            e = e[e + H <= self.capi(cap)]
            r, h = self.R[H][e], self.HIT[H][e]
            self._b[key] = (float(np.mean(r)) if len(e) else np.nan, float(np.mean(r > 0)) if len(e) else np.nan,
                            float(np.mean(h)) if len(e) else np.nan, int(len(e)))
        return self._b[key]

    def b200(self):
        j = np.flatnonzero(np.isfinite(self.sma) & (self.c < self.sma))
        j = j[j >= self.j0]
        return self.days[j]

    def cell(self, dec, H, win, delay=0):
        a, z, cap = win
        dec = np.asarray(dec).astype(str)
        d = dec[(dec >= a) & (dec <= z)]
        e = np.searchsorted(self.days, d, side="right") + delay
        ok = (e + H <= self.capi(cap)) & (e < self.n)
        pend = int((~ok).sum())
        e = e[ok]
        b = self.base(H, win)
        if not len(e):
            return {"n": 0, "pend": pend, "base": b[0]}
        r, h = self.R[H][e], self.HIT[H][e]
        n_ind, endp = 0, -1
        for x in np.sort(e):
            if x > endp:
                n_ind += 1
                endp = x + H
        return {"n": int(len(e)), "n_ind": n_ind, "pend": pend, "mean": float(r.mean()), "win": float((r > 0).mean()),
                "hit": float(h.mean()), "ex": float(r.mean() - b[0]), "base": b[0], "base_win": b[1], "base_hit": b[2],
                "ex_win": float((r > 0).mean() - b[1]), "ex_hit": float(h.mean() - b[2])}


# ============================================================================ 사건형 대상(일반 주식 전체)
class STK:
    def __init__(self, name, O: pd.DataFrame, C: pd.DataFrame, cols=None):
        self.name, self.note = name, "생존편향(지금 상장 종목만)"
        self.days = O.index.to_numpy().astype(str)
        self.n = len(self.days)
        self.j0 = 0
        self.cols = np.arange(O.shape[1]) if cols is None else np.asarray(cols)
        O = O.iloc[:, self.cols]
        C = C.iloc[:, self.cols]
        self.R, self.HIT = {}, {}
        Ov = O.to_numpy(np.float64)
        for H in H_LIST:
            R = np.full(Ov.shape, np.nan, np.float32)
            R[:-H] = (Ov[H:] / Ov[:-H] - 1).astype(np.float32)
            mnc = C.rolling(H, min_periods=1).min().shift(-(H - 1)).to_numpy(np.float64)
            mn = np.full(Ov.shape, np.nan)
            mn[:-H] = np.fmin(mnc[:-H], Ov[H:])
            HIT = np.where(np.isfinite(R), (mn <= (1 - EXTRA_DD) * Ov).astype(np.float32), np.nan).astype(np.float32)
            self.R[H], self.HIT[H] = R, HIT
        sma = C.rolling(200, min_periods=200).mean().to_numpy()
        self.below = np.isfinite(sma) & (C.to_numpy() < sma)
        self._b = {}

    def capi(self, cap):
        return int(np.searchsorted(self.days, cap, side="right")) - 1

    def base(self, H, win):
        key = (H, win)
        if key not in self._b:
            a, z, cap = win
            j0 = int(np.searchsorted(self.days, a))
            j1 = int(np.searchsorted(self.days, z, side="right")) - 1
            e = np.arange(j0 + 1, j1 + 2)
            e = e[e + H <= self.capi(cap)]
            R, Hh = self.R[H][e], self.HIT[H][e]
            fin = np.isfinite(R)
            bs = np.nanmean(R, 0)
            self._b[key] = (bs, float(np.nanmean(R)), float((R[fin] > 0).mean()), float(np.nanmean(Hh)), e)
        return self._b[key]

    def _stats(self, e, H, win, pairs_mask=None):
        bs, bm, bw, bh, _ = self.base(H, win)
        R, Hh = self.R[H][e], self.HIT[H][e]
        if pairs_mask is not None:
            R = np.where(pairs_mask, R, np.nan)
            Hh = np.where(pairs_mask, Hh, np.nan)
        X = R - bs[None, :]
        okc = np.isfinite(X)
        rows = okc.any(1)
        if not rows.any():
            return None
        R, Hh, X, okc = R[rows], Hh[rows], X[rows], okc[rows]
        ex_ev = np.nanmean(X, 1)
        mean_ev = np.nanmean(np.where(okc, R, np.nan), 1)
        win_ev = np.nansum(np.where(okc, R > 0, 0), 1) / okc.sum(1)
        hit_ev = np.nansum(np.where(okc, Hh, 0), 1) / okc.sum(1)
        return {"n": int(rows.sum()), "pairs": int(okc.sum()), "mean": float(mean_ev.mean()), "win": float(win_ev.mean()),
                "hit": float(hit_ev.mean()), "ex": float(ex_ev.mean()), "base": bm, "base_win": bw, "base_hit": bh,
                "ex_win": float(win_ev.mean() - bw), "ex_hit": float(hit_ev.mean() - bh), "e": np.flatnonzero(rows)}

    def cell(self, dec, H, win, delay=0):
        a, z, cap = win
        dec = np.asarray(dec).astype(str)
        d = dec[(dec >= a) & (dec <= z)]
        e = np.searchsorted(self.days, d, side="right") + delay
        ok = (e + H <= self.capi(cap)) & (e < self.n)
        pend = int((~ok).sum())
        e = e[ok]
        if not len(e):
            return {"n": 0, "pend": pend, "base": self.base(H, win)[1]}
        s = self._stats(e, H, win)
        if s is None:
            return {"n": 0, "pend": pend, "base": self.base(H, win)[1]}
        sel = s.pop("e")
        e2 = e[sel]
        n_ind, endp = 0, -1
        for x in np.sort(e2):
            if x > endp:
                n_ind += 1
                endp = x + H
        s["n_ind"], s["pend"] = n_ind, pend
        return s

    def cell_b200(self, H, win):
        a, z, cap = win
        j0 = int(np.searchsorted(self.days, a))
        j1 = int(np.searchsorted(self.days, z, side="right")) - 1
        j = np.arange(max(j0, 0), j1 + 1)
        j = j[j + 1 + H <= self.capi(cap)]
        s = self._stats(j + 1, H, win, pairs_mask=self.below[j])
        if s is None:
            return {"n": 0}
        s.pop("e")
        s["n_ind"] = None
        return s


# ============================================================================ 전략형 대상
class ST:
    def __init__(self, name, days, px, wins, note=""):
        self.name, self.note = name, note
        self.days = np.asarray(days).astype(str)
        self.px = np.asarray(px, float)
        self.ret = self.px[1:] / self.px[:-1] - 1
        self.rd = self.days[:-1]
        self.win = {}
        for w, (a, z) in wins.items():
            ia, iz = int(np.searchsorted(self.rd, a)), int(np.searchsorted(self.rd, z, side="right")) - 1
            if iz > ia:
                self.win[w] = (ia, iz)
        self._fr = {}

    def frontier(self, w, cash=0.0):
        key = (w, cash)
        if key not in self._fr:
            ia, iz = self.win[w]
            rr = self.ret[ia:iz + 1]
            lg = np.log1p(W_GRID[:, None] * rr[None, :] + (1 - W_GRID[:, None]) * cash / 252)
            v = np.exp(np.concatenate([np.zeros((len(W_GRID), 1)), np.cumsum(lg, 1)], 1))
            ret = v[:, -1] - 1
            mdd = (v / np.maximum.accumulate(v, 1) - 1).min(1)
            yrs = (iz - ia + 1) / 252
            cagr = (1 + ret) ** (1 / yrs) - 1
            self._fr[key] = (ret, mdd, cagr)
        return self._fr[key]

    def expo(self, sd, sv, delay=0):
        """sd·sv = 신호 날짜·그날 정한 비중. 각 거래일 k 시가(재구성 트랙은 종가)에 쓸 값 = k보다 엄격히 앞선 날짜의 값."""
        idx = np.searchsorted(sd, self.days, side="left") - 1 - delay
        e = np.where(idx >= 0, sv[np.clip(idx, 0, None)], sv[0])
        return e[:-1]

    def run(self, e, slip=0.0, cash=0.0, wins=None):
        de = np.abs(np.diff(e, prepend=e[0]))
        r = e * self.ret + (1 - e) * cash / 252 - slip * de
        out = {}
        for w in (wins or self.win):
            if w not in self.win:
                continue
            ia, iz = self.win[w]
            x = r[ia:iz + 1]
            v = np.concatenate([[1.0], np.cumprod(1 + x)])
            ret = v[-1] - 1
            mdd = float((v / np.maximum.accumulate(v) - 1).min())
            yrs = (iz - ia + 1) / 252
            cagr = (1 + ret) ** (1 / yrs) - 1
            fret, fmdd, fcagr = self.frontier(w, cash)
            met, fm = (cagr, fcagr) if w in LONG else (ret, fret)
            ok = fmdd >= mdd - 1e-12
            eff = met - fm[ok].max()
            ee = e[ia:iz + 1]
            ex = float(ee.mean())
            same = met - np.interp(ex, W_GRID, fm)
            ntr = int((np.abs(np.diff(ee)) > 1e-9).sum())
            out[w] = {"ret": float(ret), "mdd": mdd, "cagr": float(cagr), "eff": float(eff), "same": float(same),
                      "expo": ex, "ntr": ntr, "tpy": ntr / yrs, "base": float(fret[-1]) if w not in LONG else float(fcagr[-1]),
                      "base_mdd": float(fmdd[-1])}
        return out


def rule_weights(v: np.ndarray, r) -> np.ndarray:
    n = len(v)
    out = [0.0] * n
    va = np.asarray(v, float)
    v = va.tolist()
    k = r[0]
    if k == "S1":
        _, w0, t, tx = r
        w = w0
        for i in range(n):
            if v[i] <= t:
                w = 1.0
            elif v[i] >= tx:
                w = w0
            out[i] = w
    elif k == "S2":
        _, w1, g, gb = r
        w = 1.0
        for i in range(n):
            if v[i] >= g:
                w = w1
            elif v[i] <= gb:
                w = 1.0
            out[i] = w
    elif k == "S3":
        _, w0, t, tx, g, w1, gb = r
        st = 0  # 0 평소, 1 공포 추가, 2 탐욕 덜기
        for i in range(n):
            x = v[i]
            if x <= t:
                st = 1
            elif x >= g:
                st = 2
            elif st == 1 and x >= tx:
                st = 0
            elif st == 2 and x <= gb:
                st = 0
            out[i] = (w0, 1.0, w1)[st]
    else:
        _, a, b, lo = r
        tgt = np.clip(a - b * (va - 50) / 50, lo, 1.0).tolist()
        w = tgt[0]
        for i in range(n):
            if abs(tgt[i] - w) >= S4_BAND - 1e-12:
                w = tgt[i]
            out[i] = w
    return np.asarray(out)


# ============================================================================ 대상 만들기
def bars_arr(b):
    return np.asarray(b.days).astype(str), np.asarray(b.o, float), np.asarray(b.c, float)


def koru_spliced():
    syn = lev5.synth(lev5.bars("EWY"), 3, FEE3, 0.026)
    k = lev5.bars("KORU")
    sd, so, sc = bars_arr(syn)
    kd, ko, kc = bars_arr(k)
    i0 = int(np.searchsorted(sd, kd[0]))
    scale = kc[0] / sc[i0]
    days = np.concatenate([sd[:i0], kd])
    return days, np.concatenate([so[:i0] * scale, ko]), np.concatenate([sc[:i0] * scale, kc])


def real_assets():
    out = {}
    for s in ("SPY", "QQQ", "EWY"):
        out[s] = p7_ser(s) + ("",)
    for s in ("SOXX", "TQQQ", "SOXL"):
        out[s] = bars_arr(lev5.bars(s)) + ("",)
    out["KORU"] = koru_spliced() + (f"{KORU_START} 전은 EWY 3배 합성(추가 2.6%)",)
    return out


def recon_assets():
    out = {}
    d, _, c = p7_ser("^GSPC")
    out["GSPC"] = (d, c, c, "S&P500 가격지수(배당 제외)")
    d, _, c = p7_ser("^NDX")
    out["NDX"] = (d, c, c, "나스닥100 가격지수")
    sx = lev5.splice("^SOX", "SOXX", "2001-07-16")
    d, _, c = bars_arr(sx)
    out["SOX"] = (d, c, c, "^SOX(~2001-07)→SOXX")
    d, _, c = p7_ser("EWY")
    out["EWY_r"] = (d, c, c, "EWY(2000~)")
    for nm, u, ex, note in (("NDX3", lev5.bars("^NDX"), 0.013, "^NDX 3배 합성(추가 1.3%)"),
                            ("SOX3", sx, 0.015, "SOX 3배 합성(추가 1.5%)"),
                            ("EWY3", lev5.bars("EWY"), 0.026, "EWY 3배 합성(추가 2.6%)")):
        d, _, c = bars_arr(lev5.synth(u, 3, FEE3, ex))
        out[nm] = (d, c, c, note)
    return out


def load_stocks_wide():
    p = CACHE / "p4_stocks.parquet"
    df = pd.read_parquet(p, columns=["date", "sym", "Open", "High", "Low", "Close", "Adj Close"])
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df.Close > 0) & (df.Open > 0) & (df.Low > 0) & (df.High > 0)]
    f = (df["Adj Close"] / df["Close"]).fillna(1.0)
    df = df.assign(O=df.Open * f, C=df.Close * f)
    cnt = df.groupby("sym").size()
    keep = cnt[cnt >= 260].index
    df = df[df.sym.isin(keep)]
    O = df.pivot_table(index="date", columns="sym", values="O").sort_index()
    C = df.pivot_table(index="date", columns="sym", values="C").sort_index()
    O.index = O.index.astype(str)
    C.index = C.index.astype(str)
    # 자료 오류 의심 종목(하루 +150% 이상 또는 −70% 이하) 제외 = trade_lab.bad_bars와 같은 기준
    rr = C / C.shift(1) - 1
    bad = ((rr > 1.5) | (rr < -0.7)).any()
    good = bad[~bad].index
    O, C = O[good], C[good]
    sys.path.insert(0, str(ROOT))
    import watchlist
    ai = sorted({s for g in watchlist.AI_UNIVERSE.values() for subs in g.values() for s in subs if "." not in s})
    ai_cols = [i for i, s in enumerate(O.columns) if s in set(ai)]
    return O, C, ai_cols, int(bad.sum())


# ============================================================================ 사건형 전체 계산
def ev_grid(T, evs: dict, wins: dict, with_b200=True):
    out = {}
    for name, dec in evs.items():
        for H in H_LIST:
            for w, win in wins.items():
                out[(name, H, w)] = T.cell(dec, H, win)
    if with_b200:
        for H in H_LIST:
            for w, win in wins.items():
                out[("B200", H, w)] = T.cell_b200(H, win) if isinstance(T, STK) else T.cell(T.b200(), H, win)
    return out


def pick_event(G: dict, sel: str, chk: str):
    """고르기 창 사건 MIN_EV건 이상 중 아무 날 대비 초과 최대(매수 사건만)."""
    best = None
    for (name, H, w), c in G.items():
        if w != sel or name not in BUY_EV or c.get("n", 0) < MIN_EV:
            continue
        if best is None or c["ex"] > best[2]:
            best = (name, H, c["ex"])
    if best is None:
        return None
    name, H, _ = best
    return {"ev": name, "H": H, "sel": G[(name, H, sel)], "chk": G[(name, H, chk)]}


def ev_placebo(T, pla_evs, pick, wins_keys, wins):
    """같은 사건·H를 원형으로 민 F&G에서 다시 잰 값과, 고르기→확인 절차 전체의 위약 결과."""
    same, proc = [], []
    for evs in pla_evs:
        res = {}
        for w in wins_keys:
            res[w] = T.cell(evs[pick["ev"]], pick["H"], wins[w]).get("ex", np.nan)
        same.append(res)
        if proc is not None:
            G = {}
            for name in BUY_EV:
                for H in H_LIST:
                    for w in wins_keys[:2]:
                        G[(name, H, w)] = T.cell(evs[name], H, wins[w])
            p = pick_event(G, wins_keys[0], wins_keys[1])
            proc.append(p["chk"].get("ex", np.nan) if p else np.nan)
    return same, proc


def rank_of(x, arr):
    a = np.asarray([y for y in arr if np.isfinite(y)])
    if not np.isfinite(x):
        return None
    return int((a > x).sum()) + 1, len(a) + 1


# ============================================================================ 전략형 전체 계산
def st_grid(T: ST, sd, sv_by_rule, wins=None, slip=0.0, cash=0.0, delay=0):
    out = {}
    for r in RULES:
        e = T.expo(sd, sv_by_rule[rkey(r)], delay)
        out[rkey(r)] = T.run(e, slip, cash, wins)
    return out


def pick_st(G: dict, sel: str, min_tr: int):
    best = None
    for k, res in G.items():
        if sel not in res or res[sel]["ntr"] < min_tr:
            continue
        if best is None or res[sel]["eff"] > G[best][sel]["eff"]:
            best = k
    return best


def st_placebo(T: ST, sd, pla_w, rk, sel, chk, extra_w, min_tr):
    same, proc = [], []
    for Wv in pla_w:
        e = T.expo(sd, Wv[rk])
        res = T.run(e, wins=[sel, chk] + list(extra_w))
        same.append({w: res[w]["eff"] for w in res})
        G = {}
        for r in RULES:
            G[rkey(r)] = T.run(T.expo(sd, Wv[rkey(r)]), wins=[sel, chk])
        p = pick_st(G, sel, min_tr)
        proc.append(G[p][chk]["eff"] if p else np.nan)
    return same, proc


# ============================================================================ 출력 도우미
def pct(x, d=1, sign=True):
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def pp(x, d=1):
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x * 100:+.{d}f}%p"


def evname(k):
    return EV_LABEL.get(k, {"B200": "200일선 아래 매일", "ANY": "아무 날"}.get(k, k))


def jclean(o):
    if isinstance(o, dict):
        return {str(k) if not isinstance(k, tuple) else "|".join(map(str, k)): jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(x) for x in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return jclean(o.tolist())
    return o


# ============================================================================ 실행
def main():
    t0 = time.time()
    real = fg_data.fg()
    rd, rv = real.index.to_numpy().astype(str), real.to_numpy(float)
    recon, comps = build_recon(real)
    cd, cv = recon.index.to_numpy().astype(str), recon.to_numpy(float)
    ov = sorted(set(recon.index) & set(real.index))
    corr = {w: float(np.corrcoef(recon.reindex([d for d in ov if a <= d <= b]), real.reindex([d for d in ov if a <= d <= b]))[0, 1])
            for w, (a, b) in {"2011~2018(분포 맞춤 구간)": ("2011", "2018-12-31"), "2019~2026": ("2019", END)}.items()}
    rinfo = {"start": cd[0], "corr": corr, "le10_by_year": {k: int(v) for k, v in recon[recon <= 10].groupby(recon[recon <= 10].index.str[:4]).size().items()},
             "le20_days_1991_2010": int(((recon <= 20) & (recon.index <= "2010-12-31")).sum())}
    print("재구성 F&G", rinfo, f"{time.time() - t0:.0f}s")

    EVR = make_events(rd, rv)
    EVC = make_events(cd, cv)
    PEVR = [make_events(rd, np.roll(rv, k)) for k in shifts(len(rv))]
    PEVC = [make_events(cd, np.roll(cv, k)) for k in shifts(len(cv))]
    PWR = [{rkey(r): rule_weights(np.roll(rv, k), r) for r in RULES} for k in shifts(len(rv))]
    PWC = [{rkey(r): rule_weights(np.roll(cv, k), r) for r in RULES} for k in shifts(len(cv))]
    print("위약 준비", f"{time.time() - t0:.0f}s")
    J = {"id": ID, "recon": rinfo, "events_2026": {k: [d for d in v if d >= "2026"] for k, v in EVR.items()}}
    J["event_counts"] = {k: {"2011~2018": int(((v >= "2011") & (v <= "2018-12-31")).sum()),
                             "2019~2025": int(((v >= "2019") & (v <= "2025-12-31")).sum()),
                             "2026": int((v >= "2026").sum())} for k, v in EVR.items()}
    J["event_counts_recon"] = {k: {"1991~2010": int((v <= "2010-12-31").sum()), "2011~2025": int(((v >= "2011") & (v <= "2025-12-31")).sum())}
                               for k, v in EVC.items()}

    # ---------------------------------------------------------------- 사건형: 실제 F&G
    RA = real_assets()
    ETs = {s: ET(s, d, o, c, FG_START, "open", note) for s, (d, o, c, note) in RA.items()}
    b26, bh = HN.base("2026"), HN.base("hold")
    ETs["7종목26"] = ET("7종목26", b26.days, b26.open, b26.close, "2026-01-02", "open", "데스크 7종목 경로(2026)")
    ETs["7종목hold"] = ET("7종목hold", bh.days, bh.open, bh.close, "2023-05-22", "open", "데스크 7종목 경로(주간 PDF 2023-05~2025)")
    EVG = {}
    for s in ET_REAL:
        EVG[s] = ev_grid(ETs[s], EVR, EVW_REAL)
    EVG["7종목26"] = ev_grid(ETs["7종목26"], EVR, EVW_26, with_b200=False)
    EVG["7종목hold"] = ev_grid(ETs["7종목hold"], EVR, EVW_HOLD, with_b200=False)
    print("사건형 ETF·7종목", f"{time.time() - t0:.0f}s")
    O, C, ai_cols, nbad = load_stocks_wide()
    stk_all = STK("주식전체", O, C)
    EVG["주식전체"] = ev_grid(stk_all, EVR, EVW_REAL)
    del stk_all
    stk_ai = STK("AI종목", O, C, ai_cols)
    EVG["AI종목"] = ev_grid(stk_ai, EVR, EVW_REAL)
    J["stocks"] = {"n_syms": int(O.shape[1]), "n_ai": len(ai_cols), "n_bad_excluded": nbad, "start": O.index[0]}
    print("사건형 주식", O.shape, len(ai_cols), f"{time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- 사건형: 재구성 F&G
    RC = recon_assets()
    ETc = {s: ET(s, d, o, c, cd[0], "close", note) for s, (d, o, c, note) in RC.items()}
    EVGc = {s: ev_grid(ETc[s], EVC, EVW_RECON) for s in ET_RECON}
    print("사건형 재구성", f"{time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- 사건형 고르기
    ev_picks = {}
    for s in list(ET_REAL) + ["주식전체", "AI종목"]:
        p = pick_event(EVG[s], "sel", "chk")
        if p:
            for w in ("2026앞", "2026뒤", "AI쇼크", "2025", "전체"):
                p[w] = EVG[s][(p["ev"], p["H"], w)]
            p["pass"] = bool(p["chk"].get("n", 0) > 0 and p["chk"]["ex"] > 0)
            p["pos26b"] = bool(p["2026뒤"].get("n", 0) > 0 and p["2026뒤"]["ex"] > 0)
        ev_picks[s] = p
    ev_picks_c = {}
    for s in ET_RECON:
        p = pick_event(EVGc[s], "rsel", "rchk")
        if p:
            p["pass"] = bool(p["chk"].get("n", 0) > 0 and p["chk"]["ex"] > 0)
            for w in ("2000~02", "2008"):
                p[w] = EVGc[s][(p["ev"], p["H"], w)]
            # 같은 규칙을 실제 F&G로 2019~2025·2026에 잰 값
            tr = ET_RECON[s]
            for w in ("chk", "2026앞", "2026뒤", "AI쇼크", "2025"):
                p["real_" + w] = EVG[tr][(p["ev"], p["H"], w)]
        ev_picks_c[s] = p
    # 위약(사건형 고른 규칙)
    for s, p in ev_picks.items():
        if not p:
            continue
        if s in ("주식전체", "AI종목"):
            continue
        same, proc = ev_placebo(ETs[s], PEVR, p, ["sel", "chk", "2026뒤"], EVW_REAL)
        p["placebo_rank_chk"] = rank_of(p["chk"].get("ex", np.nan), [x["chk"] for x in same])
        p["placebo_rank_26b"] = rank_of(p["2026뒤"].get("ex", np.nan), [x["2026뒤"] for x in same])
        p["placebo_proc_chk"] = [float(x) for x in proc]
        p["delay1_chk"] = ETs[s].cell(EVR[p["ev"]], p["H"], EVW_REAL["chk"], delay=1).get("ex")
    for s, p in ev_picks_c.items():
        if not p:
            continue
        same, proc = ev_placebo(ETc[s], PEVC, p, ["rsel", "rchk"], EVW_RECON)
        p["placebo_rank_chk"] = rank_of(p["chk"].get("ex", np.nan), [x["rchk"] for x in same])
        p["placebo_proc_chk"] = [float(x) for x in proc]
        p["delay1_chk"] = ETc[s].cell(EVC[p["ev"]], p["H"], EVW_RECON["rchk"], delay=1).get("ex")
    print("사건형 고르기·위약", f"{time.time() - t0:.0f}s")

    # 사건형 위약: 고르기 창에서 '아무 날'을 이긴 매수 규칙 중 확인 창에서도 이긴 비율(실제 vs 위약 평균), ETF 7개 합산
    def share_pass(G, sel, chk):
        """(고르기 창 사건 15건 이상·초과>0 칸 수, 그중 확인 창도 >0, 확인 창 사건 5건 이상 칸 중 >0 비율, 그 칸 수)"""
        a = b = 0
        for (name, H, w), c in G.items():
            if w != sel or name not in BUY_EV or c.get("n", 0) < MIN_EV or c["ex"] <= 0:
                continue
            a += 1
            cc = G[(name, H, chk)]
            b += int(cc.get("n", 0) > 0 and cc["ex"] > 0)
        cc = [c["ex"] for (name, H, w), c in G.items() if w == chk and name in BUY_EV and c.get("n", 0) >= 5]
        return a, b, float(np.mean(np.asarray(cc) > 0)) if cc else np.nan, len(cc)
    sp_real = {s: share_pass(EVG[s], "sel", "chk") for s in ET_REAL}
    sp_pla = {s: [] for s in ET_REAL}
    for evs in PEVR:
        for s in ET_REAL:
            G = {(n, H, w): ETs[s].cell(evs[n], H, EVW_REAL[w]) for n in BUY_EV for H in H_LIST for w in ("sel", "chk")}
            sp_pla[s].append(share_pass(G, "sel", "chk"))
    J["ev_share_pass"] = {"real": sp_real, "placebo": sp_pla}
    print("사건형 격자 위약", f"{time.time() - t0:.0f}s")

    # 표시용 규칙의 위약 순위(사후 관찰: 격자를 본 뒤 고른 표시 목록이라 판정 근거가 아님)
    disp = {}
    for s in DISP_T:
        for ev in DISP_EV:
            for H in DISP_H:
                d = {"sel": EVG[s][(ev, H, "sel")], "chk": EVG[s][(ev, H, "chk")]}
                for w in ("sel", "chk"):
                    pl = [ETs[s].cell(evs[ev], H, EVW_REAL[w]).get("ex", np.nan) for evs in PEVR]
                    d["rank_" + w] = rank_of(d[w].get("ex", np.nan), pl)
                    d["pla_" + w] = float(np.nanmean(pl))
                disp[f"{s}|{ev}|{H}"] = d
    for s in DISP_TC:
        for ev in DISP_EV:
            for H in DISP_H:
                d = {"sel": EVGc[s][(ev, H, "rsel")], "chk": EVGc[s][(ev, H, "rchk")]}
                for w, ww in (("sel", "rsel"), ("chk", "rchk")):
                    pl = [ETc[s].cell(evs[ev], H, EVW_RECON[ww]).get("ex", np.nan) for evs in PEVC]
                    d["rank_" + w] = rank_of(d[w].get("ex", np.nan), pl)
                    d["pla_" + w] = float(np.nanmean(pl))
                disp[f"{s}|{ev}|{H}"] = d
    J["disp_rank"] = disp

    # 과제 정의 대표 후보(격자 필터): 고르기 창 사건 15건 이상·초과>0, 확인 초과>0, 2026 뒤 초과>0 인 매수 칸 중 확인 초과 최대(대상마다)
    cand = {}
    for s in list(ET_REAL) + ["주식전체", "AI종목"]:
        G = EVG[s]
        rows = []
        for ev in BUY_EV:
            for H in H_LIST:
                a, b, c = G[(ev, H, "sel")], G[(ev, H, "chk")], G[(ev, H, "2026뒤")]
                if a.get("n", 0) >= MIN_EV and a["ex"] > 0 and b.get("n", 0) > 0 and b["ex"] > 0 and c.get("n", 0) > 0 and c["ex"] > 0:
                    rows.append({"ev": ev, "H": H, "sel": a, "chk": b, "2026뒤": c, "2026앞x": G[(ev, H, "2026앞x")],
                                 "AI쇼크": G[(ev, H, "AI쇼크")], "2025": G[(ev, H, "2025")]})
        rows.sort(key=lambda r: -r["chk"]["ex"])
        cand[s] = {"n_pass": len(rows), "n_cells": len(BUY_EV) * len(H_LIST), "top": rows[:5]}
    J["grid_candidates"] = cand
    # 격자 필터 절차 자체의 위약: 민 F&G로 같은 필터를 돌렸을 때 통과 칸 수와 그중 확인 초과 최대
    gfp = {}
    for s in DISP_T:
        out = []
        for evs in PEVR:
            best, npass = np.nan, 0
            for ev in BUY_EV:
                for H in H_LIST:
                    a = ETs[s].cell(evs[ev], H, EVW_REAL["sel"])
                    if a.get("n", 0) < MIN_EV or a["ex"] <= 0:
                        continue
                    b = ETs[s].cell(evs[ev], H, EVW_REAL["chk"])
                    c = ETs[s].cell(evs[ev], H, EVW_REAL["2026뒤"])
                    if b.get("n", 0) > 0 and b["ex"] > 0 and c.get("n", 0) > 0 and c["ex"] > 0:
                        npass += 1
                        best = b["ex"] if not np.isfinite(best) else max(best, b["ex"])
            out.append((npass, best))
        gfp[s] = {"real": (cand[s]["n_pass"], cand[s]["top"][0]["chk"]["ex"] if cand[s]["top"] else np.nan), "placebo": out}
    J["grid_filter_placebo"] = gfp
    # 대표(가장 자주 1위인 칸)를 모든 대상에서 같은 정의로 다시 잰다: 하루 늦춘 판 포함
    from collections import Counter
    top = Counter((c["top"][0]["ev"], c["top"][0]["H"]) for c in cand.values() if c["top"])
    rev, rH = top.most_common(1)[0][0]
    rep = {"ev": rev, "H": rH, "top_count": dict((f"{k[0]}|{k[1]}", v) for k, v in top.items()), "by_target": {}}
    for s in list(ET_REAL) + ["7종목26", "7종목hold"]:
        T = ETs[s]
        wins = EVW_REAL if s in ET_REAL else (EVW_26 if s == "7종목26" else EVW_HOLD)
        d = {w: T.cell(EVR[rev], rH, win) for w, win in wins.items()}
        d.update({w + "_늦춤": T.cell(EVR[rev], rH, win, delay=1) for w, win in wins.items() if w in ("sel", "chk", "2026앞x", "2026뒤", "2026전체", "2025")})
        d.update({f"H21_{w}": T.cell(EVR[rev], 21, win) for w, win in wins.items()})
        rep["by_target"][s] = d
    for s in ET_RECON:
        T = ETc[s]
        rep["by_target"]["r_" + s] = {w: T.cell(EVC[rev], rH, win) for w, win in EVW_RECON.items()}
        rep["by_target"]["r_" + s].update({w + "_늦춤": T.cell(EVC[rev], rH, win, delay=1) for w, win in EVW_RECON.items()})
    for s in ("주식전체", "AI종목"):
        rep["by_target"][s] = {w: EVG[s][(rev, rH, w)] for w in EVW_REAL}
    J["rep"] = rep
    print("표시 규칙 위약", f"{time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- 전략형
    SW_r = {rkey(r): rule_weights(rv, r) for r in RULES}
    SW_c = {rkey(r): rule_weights(cv, r) for r in RULES}
    STs = {}
    for s in ST_REAL:
        d, o, c, note = RA[s]
        STs[s] = ST(s, d, o, STW_REAL, note)
    STs["7종목26"] = ST("7종목26", b26.days, b26.open, STW_26, "데스크 7종목(2026)")
    STs["7종목hold"] = ST("7종목hold", bh.days, bh.open, STW_HOLD, "데스크 7종목(hold)")
    STc = {}
    for s in ST_RECON:
        d, o, c, note = RC[s]
        STc[s] = ST(s, d, c, STW_RECON, note + ", 종가 체결")
    SG = {s: st_grid(T, rd, SW_r) for s, T in STs.items()}
    SG4 = {s: st_grid(T, rd, SW_r, cash=CASH4) for s, T in STs.items()}
    SGs = {s: st_grid(T, rd, SW_r, slip=SLIP) for s, T in STs.items()}
    SGc = {s: st_grid(T, cd, SW_c) for s, T in STc.items()}
    print("전략형 격자", f"{time.time() - t0:.0f}s")

    st_long = {}
    for s in ST_REAL:
        k = pick_st(SG[s], "sel", MIN_TR_LONG)
        p = {"rule": k, "res": SG[s][k], "res_cash4": SG4[s][k], "res_slip": SGs[s][k]}
        p["pass"] = bool(SG[s][k]["chk"]["eff"] > 0)
        p["pos26b"] = bool(SG[s][k]["2026뒤"]["eff"] > 0)
        same, proc = st_placebo(STs[s], rd, PWR, k, "sel", "chk", ["2026뒤"], MIN_TR_LONG)
        p["placebo_rank_chk"] = rank_of(SG[s][k]["chk"]["eff"], [x["chk"] for x in same])
        p["placebo_rank_26b"] = rank_of(SG[s][k]["2026뒤"]["eff"], [x["2026뒤"] for x in same])
        p["placebo_proc_chk"] = [float(x) for x in proc]
        e1 = STs[s].expo(rd, SW_r[k], delay=1)
        p["delay1"] = STs[s].run(e1, wins=["chk", "2026뒤"])
        p["7종목26"] = SG["7종목26"][k]
        p["7종목hold"] = SG["7종목hold"][k]
        st_long[s] = p
    st_long_c = {}
    for s in ST_RECON:
        k = pick_st(SGc[s], "rsel", MIN_TR_LONG)
        p = {"rule": k, "res": SGc[s][k], "pass": bool(SGc[s][k]["rchk"]["eff"] > 0)}
        same, proc = st_placebo(STc[s], cd, PWC, k, "rsel", "rchk", [], MIN_TR_LONG)
        p["placebo_rank_chk"] = rank_of(SGc[s][k]["rchk"]["eff"], [x["rchk"] for x in same])
        p["placebo_proc_chk"] = [float(x) for x in proc]
        e1 = STc[s].expo(cd, SW_c[k], delay=1)
        p["delay1"] = STc[s].run(e1, wins=["rchk"])
        tr = ST_RECON[s]
        p["real"] = SG[tr][k]
        st_long_c[s] = p
    st_26 = {}
    for s in ("7종목26",) + ST_REAL:
        k = pick_st(SG[s], "2026앞", MIN_TR_26)
        if k is None:
            st_26[s] = None
            continue
        p = {"rule": k, "res": SG[s][k]}
        if s == "7종목26":
            p["hold"] = SG["7종목hold"][k]
        same, proc = st_placebo(STs[s], rd, PWR, k, "2026앞", "2026뒤", ["AI쇼크"], MIN_TR_26)
        p["placebo_rank_26b"] = rank_of(SG[s][k]["2026뒤"]["eff"], [x["2026뒤"] for x in same])
        p["placebo_proc_26b"] = [float(x) for x in proc]
        e1 = STs[s].expo(rd, SW_r[k], delay=1)
        p["delay1"] = STs[s].run(e1, wins=["2026뒤", "AI쇼크"])
        st_26[s] = p
    print("전략형 고르기·위약", f"{time.time() - t0:.0f}s")

    # 전략형 격자 위약: 고르기 창 효율선초과 > 0 인 규칙 중 확인에서도 > 0 비율
    def st_share(G, sel, chk):
        """(고르기 효율선초과>0 규칙 수, 그중 확인도 >0, 확인 >0 규칙 수, 확인 효율선초과 중앙값, 고르기 중앙값)"""
        a = [k for k, r in G.items() if r[sel]["eff"] > 0 and r[sel]["ntr"] >= MIN_TR_LONG]
        ch = np.array([r[chk]["eff"] for r in G.values()])
        se = np.array([r[sel]["eff"] for r in G.values()])
        return len(a), sum(1 for k in a if G[k][chk]["eff"] > 0), int((ch > 0).sum()), float(np.median(ch)), float(np.median(se))
    stsp = {s: {"real": st_share(SG[s], "sel", "chk"), "pla": []} for s in ST_REAL}
    for Wv in PWR:
        for s in ST_REAL:
            G = {kk: STs[s].run(STs[s].expo(rd, Wv[kk]), wins=["sel", "chk"]) for kk in Wv}
            stsp[s]["pla"].append(st_share(G, "sel", "chk"))
    J["st_share_pass"] = stsp
    stspc = {s: {"real": st_share(SGc[s], "rsel", "rchk"), "pla": []} for s in ST_RECON}
    for Wv in PWC:
        for s in ST_RECON:
            G = {kk: STc[s].run(STc[s].expo(cd, Wv[kk]), wins=["rsel", "rchk"]) for kk in Wv}
            stspc[s]["pla"].append(st_share(G, "rsel", "rchk"))
    J["st_share_pass_recon"] = stspc
    print("전략형 격자 위약", f"{time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- 2026 경로(날짜)
    SHOW = [("S1", 0.8, 20, 50), ("S1", 0.8, 25, 50), ("S1", 0.7, 30, 60), ("S2", 0.75, 80, 60), ("S4", 0.8, 0.2, 0.5)]
    picked = {p["rule"] for p in list(st_long.values()) + [x for x in st_26.values() if x]} | {p["rule"] for p in st_long_c.values()}
    rmap = {rkey(r): r for r in RULES}
    show_keys = [rkey(r) for r in SHOW] + sorted(k for k in picked if k not in {rkey(r) for r in SHOW})
    paths = {}
    T26 = STs["7종목26"]
    for k in show_keys:
        e = T26.expo(rd, SW_r[k])
        ch = [(T26.rd[0], float(e[0]))] + [(T26.rd[i], float(e[i])) for i in range(1, len(e)) if abs(e[i] - e[i - 1]) > 1e-9]
        paths[k] = {"rule": rule_ko(rmap[k]), "changes": ch, "7종목26": SG["7종목26"][k], "QQQ": SG["QQQ"][k],
                    "SOXL": SG["SOXL"][k], "TQQQ": SG["TQQQ"][k]}
    J["paths_2026"] = paths

    # ---------------------------------------------------------------- 저장
    J["ev_grid"] = {s: {f"{n}|{H}|{w}": c for (n, H, w), c in G.items()} for s, G in EVG.items()}
    J["ev_grid_recon"] = {s: {f"{n}|{H}|{w}": c for (n, H, w), c in G.items()} for s, G in EVGc.items()}
    J["ev_picks"], J["ev_picks_recon"] = ev_picks, ev_picks_c
    J["st_grid"] = SG
    J["st_grid_cash4"] = {s: {k: {w: v["eff"] for w, v in r.items()} for k, r in G.items()} for s, G in SG4.items()}
    J["st_grid_slip"] = {s: {k: {w: v["eff"] for w, v in r.items()} for k, r in G.items()} for s, G in SGs.items()}
    J["st_grid_recon"] = SGc
    J["st_long"], J["st_long_recon"], J["st_26"] = st_long, st_long_c, st_26
    J["grid_size"] = {"event_defs": len(EV_LABEL), "H": len(H_LIST), "event_cells_per_target": len(EV_LABEL) * len(H_LIST),
                      "event_targets_real": len(ET_REAL) + 4, "event_targets_recon": len(ET_RECON),
                      "st_rules": len(RULES), "st_targets_real": len(STs), "st_targets_recon": len(STc),
                      "total": len(EV_LABEL) * len(H_LIST) + len(RULES)}
    J["fg_2026"] = {d: float(x) for d, x in zip(rd, rv) if d >= "2026"}
    J["fg_last"] = (rd[-1], float(rv[-1]))
    (RESULTS / f"{ID}.json").write_text(json.dumps(jclean(J), ensure_ascii=False, indent=0), encoding="utf-8")
    print("저장 json", f"{time.time() - t0:.0f}s")
    return J


# ============================================================================ 보고서
def _c(c, unit="일"):
    if not c or not c.get("n"):
        return "-"
    return f"{c['ex'] * 100:+.1f}%p({c['n']}{unit})"


def _e(r, w, key="eff"):
    if not r or w not in r:
        return "-"
    return f"{r[w][key] * 100:+.1f}%p"


def _shock(r):
    if not r or "AI쇼크" not in r:
        return "-"
    return f"{r['AI쇼크']['ret'] * 100:+.1f}%(기본 {r['AI쇼크']['base'] * 100:+.1f}%)"


def _rk(x):
    return "-" if not x else f"{x[0]}/{x[1]}"


def write_md(J):
    EG, EGc, SG, SGc = J["ev_grid"], J["ev_grid_recon"], J["st_grid"], J["st_grid_recon"]
    g = lambda t, ev, H, w: EG[t].get(f"{ev}|{H}|{w}")
    gc = lambda t, ev, H, w: EGc[t].get(f"{ev}|{H}|{w}")
    rep = J["rep"]
    rev, rH = rep["ev"], rep["H"]
    rmap = {rkey(r): r for r in RULES}
    L = []
    A = L.append
    A("# 8단계 FGDIRECT: CNN 공포탐욕지수(F&G)를 그대로 매매에 쓰면 (2026-10-11)")
    A("")
    A("> 질문: F&G를 사기·덜기·다시 사기 시점에 쓰면 '아무 날 사기'·'덜 담기'보다 나은가. 코드 `code/p8_FGDIRECT.py`, 숫자 전부 `results/p8_FGDIRECT.json`.")
    A(f"> F&G 실제 2011-01-03~2026-10-09(3,943일). 긴 역사용 재구성 F&G는 {J['recon']['start']}~(공개 가격 5요소 대리값, 실제와 상관 "
      f"{J['recon']['corr']['2011~2018(분포 맞춤 구간)']:.2f}·{J['recon']['corr']['2019~2026']:.2f}). 시점: 그날 종가 F&G → 다음 거래일 시가(재구성 트랙은 다음 날 종가).")
    A("")
    q = rep["by_target"]["QQQ"]
    qr = rep["by_target"]["r_NDX"]
    k7 = rep["by_target"]["7종목26"]
    sl = SG["7종목26"]
    nS1 = sum(1 for k in sl if k.startswith("S1") and sl[k]["2026앞"]["eff"] > 0)
    sh = J["st_share_pass"]
    pla_pos = np.mean([np.mean([x[0] for x in v["pla"]]) for v in sh.values()])
    A("## 결론")
    A("")
    A("1. **비중 규칙(평소 덜 들고 공포에 채우기, 탐욕에 덜기)은 '같은 낙폭의 덜 담기'를 못 이겼다.** 60개 규칙 × QQQ·TQQQ·SOXX·SOXL에서 "
      f"2011~2018 효율선초과가 플러스인 규칙은 0개였다(F&G를 시간축으로 민 위약 판은 평균 {pla_pos:.0f}개). 공포에 더 사면 최대 낙폭이 깊어지기 때문이다. "
      + (f"2026 앞 구간 7종목에서는 공포 추가 36개가 모두 마이너스였고" if nS1 == 0 else f"2026 앞 구간 7종목에서는 공포 추가 36개 중 {nS1}개만 플러스였고")
      + ", 탐욕 덜기는 2026에 한 번도 켜지지 않았다(최고 71).")
    A(f"2. **남은 것 하나는 새 돈을 넣는 시점이다: 'F&G ≤25인 날마다 사서 63거래일 들기'.** QQQ 아무 날 대비 2011~2018 {q['sel']['ex'] * 100:+.1f}%p → "
      f"2019~2025 {q['chk']['ex'] * 100:+.1f}%p, 재구성 1991~2010 {qr['rsel']['ex'] * 100:+.1f}%p → 2011~2025 {qr['rchk']['ex'] * 100:+.1f}%p. "
      f"2026 3월 공포에 산 돈은 {q['2026앞x']['ex'] * 100:+.1f}%p였지만, 7월 쇼크 중(6/26) 산 돈은 21일 뒤 {q['H21_2026뒤']['ex'] * 100:+.1f}%p"
      f"(7종목 {k7['H21_2026뒤']['ex'] * 100:+.1f}%p)였다. 독립 사건은 구간마다 16~17번뿐이고, 공포가 오래 간 2008년(재구성)에는 손해였다.")
    A("3. 실무(규칙 변경 아님): F&G는 손절·재진입·비중 신호로 쓰지 않는다. 'F&G ≤25 구간에 새 돈 나눠 넣기, 3개월 기준'만 관찰 후보로 기록한다. "
      f"'처음 ≤T로 떨어진 날'·'공포 탈출 첫날'은 확인 구간에서 무너졌다. 10/9 F&G {J['fg_last'][1]:.0f}로 신호 없음(지수 탐욕·폭 공포의 '좁은 장'은 RECON 갈래).")
    A("")

    # ---------------------------------------------------------------- 사용자 표
    A("## 사용자 표")
    A("")
    A("사건형 칸 = 그날 산 돈의 '아무 날 산 돈' 대비 초과(사건 일수). 전략형 칸 = 효율선초과(같은 최대 낙폭이 나오는 고정 비중 대비, 긴 구간은 연 %p). "
      "7월 쇼크(6/22~7/29) 칸은 사건형은 그 사이 산 돈, 전략형은 그 구간 수익(기본 = 100% 보유). 2026 앞 사건형은 앞 구간에 산 돈을 끝까지 잰 값.")
    A("")
    A("| 규칙 | 대상 | 2026 앞 | 2026 뒤 | 7월 쇼크 | 2025 | 실제 11~18 → 19~25 | 재구성 91~10 → 11~25 | 사건 수·연 매매 |")
    A("|---|---|---|---|---|---|---|---|---|")

    def ev_row(label, t, ev, H, tr=None, t26=None, thold=None):
        if t26:
            a, b, c = g(t26, ev, H, "2026앞x"), g(t26, ev, H, "2026뒤"), g(t26, ev, H, "AI쇼크")
            d = g(thold, ev, H, "2025")
            lo = "-"
        else:
            a, b, c, d = g(t, ev, H, "2026앞x"), g(t, ev, H, "2026뒤"), g(t, ev, H, "AI쇼크"), g(t, ev, H, "2025")
            lo = f"{_c(g(t, ev, H, 'sel'))} → {_c(g(t, ev, H, 'chk'))}"
        rc = f"{_c(gc(tr, ev, H, 'rsel'))} → {_c(gc(tr, ev, H, 'rchk'))}" if tr else "-"
        cnt = J["event_counts"].get(ev)
        cn = f"{cnt['2011~2018']}/{cnt['2019~2025']}/{cnt['2026']}" if cnt else "-"
        A(f"| {label} | {t26 or t} | {_c(a)} | {_c(b)} | {_c(c)} | {_c(d)} | {lo} | {rc} | {cn} |")

    ev_row(f"F&G ≤25인 날마다 사서 {rH}일 (대표)", "QQQ", rev, rH, "NDX")
    ev_row("같은 규칙", "SPY", rev, rH, "GSPC")
    ev_row("같은 규칙", "SOXL", rev, rH, "SOX3")
    ev_row("같은 규칙", None, rev, rH, None, "7종목26", "7종목hold")
    ev_row("같은 규칙, 21일 보유", "QQQ", rev, 21, "NDX")
    ev_row("F&G ≤10 첫날 사서 63일", "QQQ", "E1_10", 63, "NDX")
    ev_row("≤20 뒤 20 위로 올라선 첫날, 63일(긴 역사 고르기)", "QQQ", "E2_20_0", 63, "NDX")
    ev_row("F&G ≥80 첫날(탐욕) 뒤 63일", "QQQ", "E3_80", 63, "NDX")
    ev_row("200일선 아래인 날마다 63일(단순 기준)", "QQQ", "B200", 63, "NDX")

    def st_row(label, t, k, tr="NDX"):
        r = SG[t][k]
        if t == "7종목26":
            h = SG["7종목hold"][k]
            lo, rc, y25 = "-", "-", _e(h, "2025")
            tp = f"{r['2026전체']['ntr']}번(2026)"
        else:
            lo = f"{_e(r, 'sel')} → {_e(r, 'chk')}"
            rr = SGc[tr][k]
            rc = f"{_e(rr, 'rsel')} → {_e(rr, 'rchk')}"
            y25 = _e(r, "2025")
            tp = f"연 {r['전체']['tpy']:.1f}번"
        A(f"| {label} | {t} | {_e(r, '2026앞')} | {_e(r, '2026뒤')} | {_shock(r)} | {y25} | {lo} | {rc} | {tp} |")

    kq = J["st_long"]["QQQ"]["rule"]
    st_row(f"{rule_ko(rmap[kq])} (QQQ 긴 역사 고르기)", "QQQ", kq)
    st_row("같은 규칙", "7종목26", kq)
    st_row(rule_ko(rmap["S1_0.8_25_50"]), "7종목26", "S1_0.8_25_50")
    st_row(rule_ko(rmap["S1_0.8_25_50"]), "TQQQ", "S1_0.8_25_50", "NDX3")
    st_row(rule_ko(rmap["S2_0.5_75_60"]) + " (탐욕 덜기)", "QQQ", "S2_0.5_75_60")
    st_row(rule_ko(rmap["S4_0.8_0.2_0.5"]), "7종목26", "S4_0.8_0.2_0.5")
    A("")
    A("- 사건 수 = F&G 사건일 수(2011~18/2019~25/2026). '≤25 매일'은 날짜가 몰려 겹치지 않는 사건은 2011~18 17번, 2019~25 16번, 2026 2번(3월, 6/26)이다.")
    A(f"- 대표 규칙 하루 늦춘 판(다다음 날 시가): QQQ 2019~25 {q['chk_늦춤']['ex'] * 100:+.1f}%p, 2026 뒤 {q['2026뒤_늦춤']['ex'] * 100:+.1f}%p. "
      "미끄러짐 0.5%는 '아무 날'도 같이 내므로 초과는 그대로다.")
    A(f"- 대표 규칙의 추가 −20% 확률(63일 안): QQQ 2019~25 {q['chk']['hit'] * 100:.0f}% 대 아무 날 {q['chk']['base_hit'] * 100:.0f}%, "
      f"TQQQ {rep['by_target']['TQQQ']['chk']['hit'] * 100:.0f}% 대 {rep['by_target']['TQQQ']['chk']['base_hit'] * 100:.0f}%, "
      f"SOXL {rep['by_target']['SOXL']['chk']['hit'] * 100:.0f}% 대 {rep['by_target']['SOXL']['chk']['base_hit'] * 100:.0f}%. 3배는 공포에 사도 꼬리가 줄지 않았다.")
    A("")

    # ---------------------------------------------------------------- 2026 경로
    E26 = J["events_2026"]
    P = J["paths_2026"]
    A("## 2026 실제 경로")
    A("")
    A("- F&G 2026: 최저 5.8(3/30), 최고 71(5/1), 6월 말~7월 25~40, 9월 중순 27~31, 10/9 45.")
    A(f"- 공포 첫날: ≤30 {', '.join(d[5:] for d in E26['E1_30'])} / ≤25 {', '.join(d[5:] for d in E26['E1_25'])} / ≤20 {', '.join(d[5:] for d in E26['E1_20'])} / "
      f"≤10 {', '.join(d[5:] for d in E26['E1_10'])}. ≤25인 날은 3/9~3/31 20일과 6/26 하루. 탐욕 ≥75는 한 번도 없었다(≥70 첫날 5/1).")
    A(f"- 3월: 공포에 산 돈은 크게 이겼다(63일, 아무 날 대비 QQQ {q['2026앞x']['ex'] * 100:+.1f}%p, 7종목 {k7['2026앞x']['ex'] * 100:+.1f}%p, "
      f"SOXL {rep['by_target']['SOXL']['2026앞x']['ex'] * 100:+.1f}%p). V자 바닥이었다.")
    A(f"- 7월 쇼크: F&G는 6/26에 25까지 내려갔지만 7종목은 7/29까지 더 빠졌다. 6/26 산 돈은 21일 뒤 아무 날 대비 7종목 {k7['H21_2026뒤']['ex'] * 100:+.1f}%p, "
      f"SOXL {rep['by_target']['SOXL']['H21_2026뒤']['ex'] * 100:+.1f}%p, 63일 뒤에는 7종목 {k7['2026뒤']['ex'] * 100:+.1f}%p. ≤30 첫날 7/29가 바닥날이었다.")
    for k in ("S1_0.8_25_50", kq, "S4_0.8_0.2_0.5"):
        ch = [f"{d[5:]} {w:.0%}" for d, w in P[k]["changes"] if d >= "2026-01-02"]
        r7 = P[k]["7종목26"]
        A(f"- {P[k]['rule']}: {' → '.join(ch[:9])}{' …' if len(ch) > 9 else ''}. 7종목 7월 쇼크 {r7['AI쇼크']['ret'] * 100:+.1f}%(기본 −18.7%), "
          f"2026 전체 {r7['2026전체']['ret'] * 100:+.1f}%(기본 +223.7%).")
    A("- 공포 추가 규칙은 7월 쇼크를 오히려 100%로 맞았다(6/11 또는 6/29에 채움). 평소 덜어 둔 몫만큼만 덜 잃었다.")
    A("")

    # ---------------------------------------------------------------- 격자
    A("## 조합 격자 (사건형, QQQ, 실제 F&G)")
    A("")
    A("칸 = 2011~18 초과 → 2019~25 초과 %p (사건 수 앞/뒤). 고르기 창은 매도까지 2018 안에 끝난 거래만.")
    A("")
    A("| 사건 | " + " | ".join(f"{h}일" for h in H_LIST) + " |")
    A("|---|" + "---|" * len(H_LIST))
    for ev in list(EV_LABEL) + ["B200"]:
        cells = []
        for H in H_LIST:
            a, b = g("QQQ", ev, H, "sel"), g("QQQ", ev, H, "chk")
            bt = f"{b['ex'] * 100:+.1f}" if b and b.get("n") else "-"
            if not a or not a.get("n"):
                cells.append(f"- → {bt}")
                continue
            cells.append(f"{a['ex'] * 100:+.1f} → {bt} ({a['n']}/{b.get('n', 0) if b else 0})")
        A(f"| {evname(ev)} | " + " | ".join(cells) + " |")
    A("")
    es = J["ev_share_pass"]
    ra = sum(v[1] for v in es["real"].values()) / max(1, sum(v[0] for v in es["real"].values()))
    pla = np.nanmean([np.nanmean([x[1] / x[0] if x[0] else np.nan for x in v]) for v in es["placebo"].values()])
    rc = np.mean([v[2] for v in es["real"].values()])
    pc = np.mean([np.nanmean([x[2] for x in v]) for v in es["placebo"].values()])
    A(f"- ETF 7개 합산: 고르기에서 아무 날을 이긴 매수 칸 중 확인에서도 이긴 비율 실제 {ra:.0%} 대 위약 {pla:.0%}. "
      f"확인 구간에서 이긴 칸 비율 실제 {rc:.0%} 대 위약 {pc:.0%}. 차이는 대부분 '≤T 매일' 줄에서 나왔다.")
    dr = J["disp_rank"]
    A(f"- '≤10 매일, 63일' 위약 순위(21판 중, 1이 최고): 2019~25 QQQ {_rk(dr['QQQ|E4_10|63']['rank_chk'])}, SPY {_rk(dr['SPY|E4_10|63']['rank_chk'])}, "
      f"SOXL {_rk(dr['SOXL|E4_10|63']['rank_chk'])}; 2011~18 QQQ {_rk(dr['QQQ|E4_10|63']['rank_sel'])}; 재구성 1991~2010 NDX {_rk(dr['NDX|E4_10|63']['rank_sel'])}, "
      f"3배 {_rk(dr['NDX3|E4_10|63']['rank_sel'])}. 2019~25에만 뚜렷했다(독립 사건 6번, 주로 2020-03·2022-05·2025-04·2025-11 급락 뒤 반등). "
      "2018-10~12 극단 공포는 고르기 창(매도까지 2018 안)과 확인 창(2019년 사건) 어디에도 들어가지 않았다.")
    A(f"- '≤25 매일, 63일' 위약 순위: QQQ 2011~18 {_rk(dr['QQQ|E4_25|63']['rank_sel'])}, 2019~25 {_rk(dr['QQQ|E4_25|63']['rank_chk'])}; "
      f"재구성 NDX 1991~2010 {_rk(dr['NDX|E4_25|63']['rank_sel'])}, 2011~25 {_rk(dr['NDX|E4_25|63']['rank_chk'])}.")
    A("")
    A("### 전략형 격자 요약 (효율선초과 중앙값, 괄호는 플러스 규칙 수)")
    A("")
    A("| 대상 | 갈래 | 2011~18(연) | 2019~25(연) | 2026 앞 | 2026 뒤 | 2025 |")
    A("|---|---|---|---|---|---|---|")
    for t in ("QQQ", "TQQQ", "SOXX", "SOXL", "7종목26"):
        G = SG[t]
        for fam in ("S1", "S2", "S3", "S4"):
            ks = [k for k in G if k.startswith(fam)]

            def md(w, GG=G, ks=ks):
                if w not in GG[ks[0]]:
                    return "-"
                v = np.array([GG[k][w]["eff"] for k in ks])
                return f"{np.median(v) * 100:+.1f}%p ({int((v > 0).sum())}/{len(v)})"
            y25 = md("2025") if t != "7종목26" else md("2025", SG["7종목hold"])
            A(f"| {t} | {fam} | {md('sel')} | {md('chk')} | {md('2026앞')} | {md('2026뒤')} | {y25} |")
    A("")
    A("- S1 공포 추가(36), S2 탐욕 덜기(12), S3 결합(6), S4 연속형(6). 7종목 2025는 hold 경로.")
    A("")

    # ---------------------------------------------------------------- 고르기 표
    A("## 고르기 표")
    A("")
    A("### 2026 판정: 앞(1/2~5/29)에서 효율선초과 최대 → 뒤(6/1~10/9) 확인 (전략형)")
    A("")
    A("| 대상 | 고른 규칙 | 앞 | 뒤 | 7월 쇼크 수익 | 2025 | 뒤 위약 순위 | 하루 늦춤(뒤) |")
    A("|---|---|---|---|---|---|---|---|")
    for t, p in J["st_26"].items():
        if not p:
            A(f"| {t} | 앞에 비중 변경이 있는 규칙 없음 | | | | | | |")
            continue
        r = p["res"]
        y25 = _e(p.get("hold") or r, "2025")
        A(f"| {t} | {rule_ko(rmap[p['rule']])} | {_e(r, '2026앞')} | {_e(r, '2026뒤')} | {_shock(r)} | {y25} | {_rk(p['placebo_rank_26b'])} | "
          f"{_e(p['delay1'], '2026뒤')} |")
    A("")
    A("- 사건형은 2026 앞에서 끝난 거래가 대부분 1건이라 2026 고르기를 하지 않았다(사용자 표의 2026 칸 참고).")
    A("")
    A("### 긴 역사 판정 1: 실제 F&G 2011~2018 고르기 → 2019~2025 확인")
    A("")
    A("| 대상 | 종류 | 고른 규칙 | 고르기 | 확인 | 통과 | 확인 위약 순위 | 위약 절차 평균 | 2026 뒤 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for t, p in J["ev_picks"].items():
        if not p:
            A(f"| {t} | 사건형 | 사건 15건 이상 칸 없음 | | | | | | |")
            continue
        pm = np.nanmean(p["placebo_proc_chk"]) if p.get("placebo_proc_chk") else np.nan
        A(f"| {t} | 사건형 | {evname(p['ev'])}, {p['H']}일 | {_c(p['sel'], '번')} | {_c(p['chk'], '번')} | {'예' if p['pass'] else '아니오'} | "
          f"{_rk(p.get('placebo_rank_chk'))} | {pp(pm)} | {_c(p['2026뒤'], '번')} |")
    for t, p in J["st_long"].items():
        r = p["res"]
        pm = np.nanmean(p["placebo_proc_chk"])
        A(f"| {t} | 전략형 | {rule_ko(rmap[p['rule']])} | {_e(r, 'sel')} | {_e(r, 'chk')} | {'예' if p['pass'] else '아니오'} | "
          f"{_rk(p['placebo_rank_chk'])} | {pp(pm)} | {_e(r, '2026뒤')} |")
    A("")
    A("- 전략형은 2011~2018에 플러스 규칙이 없어 '덜 진 것'을 골랐다. 사건형 2026 뒤 '-'는 그 구간에 사건이 없었다는 뜻(F&G가 20 아래로 안 감). "
      "일반 주식 두 줄은 위약을 돌리지 않았다. SOXL·KORU 사건형 통과는 위약 절차 평균(+31%p·+28%p)도 커서 특별하지 않다.")
    A("")
    A("### 긴 역사 판정 2: 재구성 F&G 1991~2010 고르기 → 2011~2025 확인")
    A("")
    A("| 대상(대리) | 종류 | 고른 규칙 | 고르기 | 확인 | 통과 | 확인 위약 순위 | 2008 | 실제 F&G 2019~25 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for t, p in J["ev_picks_recon"].items():
        if not p:
            continue
        A(f"| {t}({ET_RECON[t]}) | 사건형 | {evname(p['ev'])}, {p['H']}일 | {_c(p['sel'], '번')} | {_c(p['chk'], '번')} | {'예' if p['pass'] else '아니오'} | "
          f"{_rk(p.get('placebo_rank_chk'))} | {_c(p['2008'], '번')} | {_c(p['real_chk'], '번')} |")
    for t, p in J["st_long_recon"].items():
        r = p["res"]
        A(f"| {t}({ST_RECON[t]}) | 전략형 | {rule_ko(rmap[p['rule']])} | {_e(r, 'rsel')} | {_e(r, 'rchk')} | {'예' if p['pass'] else '아니오'} | "
          f"{_rk(p['placebo_rank_chk'])} | 수익 {r['2008']['ret'] * 100:+.1f}% | {_e(p['real'], 'chk')} |")
    A("")
    A("### 과제 정의 대표 후보 (격자 필터: 2011~18 초과>0·사건 15건 이상, 2019~25 초과>0, 2026 뒤 초과>0)")
    A("")
    gf = J["grid_filter_placebo"]
    A("| 대상 | 통과 칸(105칸 중) | 1위 칸 | 2019~25 | 위약 판 통과 칸 평균 | 위약 1위 ≥ 실제 1위 |")
    A("|---|---|---|---|---|---|")
    for t, c in J["grid_candidates"].items():
        top = c["top"][0] if c["top"] else None
        pl = gf.get(t)
        pn = f"{np.mean([x[0] for x in pl['placebo']]):.1f}" if pl else "-"
        pb = f"{sum(1 for x in pl['placebo'] if x[1] is not None and x[1] >= pl['real'][1])}/20" if pl else "-"
        A(f"| {t} | {c['n_pass']} | {evname(top['ev']) + ', ' + str(top['H']) + '일' if top else '-'} | {_c(top['chk']) if top else '-'} | {pn} | {pb} |")
    A("")
    A(f"- 9개 대상 중 {rep['top_count'].get(rev + '|' + str(rH), 0)}개에서 1위가 같은 칸('≤25 매일, 63일')이었다. 위약 판에서는 같은 필터의 1위가 한 번도 실제 1위에 못 미쳤다. "
      "다만 2026 뒤 사건은 6/26 하루뿐이다.")
    A(f"- 재구성 트랙 같은 규칙: 1991~2010 NDX {_c(qr['rsel'])}, 3배 {_c(rep['by_target']['r_NDX3']['rsel'])}; 2000~02 NDX {_c(qr['2000~02'])}; "
      f"2008 NDX {_c(qr['2008'])}, 3배 {_c(rep['by_target']['r_NDX3']['2008'])}.")
    A("")

    # ---------------------------------------------------------------- 탐욕
    A("## 탐욕(≥G 첫날)은 덜기 근거가 되나")
    A("")
    A(f"- 아니다. QQQ 63일 기준 '≥80 첫날' 뒤 수익은 아무 날 대비 2011~18 {_c(g('QQQ', 'E3_80', 63, 'sel'), '번')}, 2019~25 {_c(g('QQQ', 'E3_80', 63, 'chk'), '번')}, "
      f"재구성 NDX 1991~2010 {_c(gc('NDX', 'E3_80', 63, 'rsel'), '번')}, 2011~25 {_c(gc('NDX', 'E3_80', 63, 'rchk'), '번')}. 작고, 방향이 구간마다 바뀐다.")
    A("- 탐욕 덜기(S2)는 QQQ·TQQQ·SOXX·SOXL 모두 2011~18·2019~25 효율선초과가 마이너스였다(전략형 격자 요약). 2026에는 F&G가 75를 넘은 날이 없어 켜지지 않았다.")
    A("")

    # ---------------------------------------------------------------- 한계
    gs = J["grid_size"]
    A("## 시험 범위·한계")
    A("")
    A(f"- 격자(사전 선언): 사건형 {len(EV_LABEL)}개 사건 × 보유 {len(H_LIST)}개 = {len(EV_LABEL) * len(H_LIST)}칸(+200일선 기준 5칸), "
      f"대상 실제 {gs['event_targets_real']}개·재구성 {gs['event_targets_recon']}개. 전략형 {len(RULES)}개(S1 36·S2 12·S3 6·S4 6) × "
      f"실제 {gs['st_targets_real']}개·재구성 {gs['st_targets_recon']}개. 규칙 조합 {gs['total']}개. 확인 구간을 본 뒤 격자를 바꾸지 않았다. "
      "대량 탐색이라 '1위 칸'은 선택편향이 있다 → F&G를 63거래일 이상 원형으로 민 20판의 같은 규칙 순위와 절차 위약을 같이 적었다.")
    A("- 표시용 규칙(사용자 표의 ≤10 첫날·≥80 첫날·≤25 21일, 위약 순위 줄)과 '격자 필터 대표'는 격자를 본 뒤의 정리다. 사전 판정은 고르기 표 두 개다.")
    A("- F&G 2011~2020 값은 CNN이 2020년 개편 때 거꾸로 채운 값일 수 있다. 재구성은 7요소 중 5개 대리(풋콜·맥클렐런 폭 없음)이고 2011~18 실제 분포에 값만 맞췄다. "
      "FRED가 이 환경에서 응답하지 않아 정크 수요·안전자산은 펀드 기준가(VWEHX·VUSTX)와 ^TNX로 만들었다.")
    A("- 사건이 몰린다: '≤25 매일' 225일이 독립 16번, '≤10 매일' 34일이 독립 6번. 2026 뒤 사건은 6/26 하루.")
    A("- 재구성 트랙은 다음 날 종가 체결, 지수는 배당 제외(아무 날 기준도 같아 초과에는 영향 작음). KORU 2013-04 전은 EWY 3배 합성.")
    A("- 일반 주식(p4)은 지금 살아 있는 종목만이라 생존편향이 있고 2014년부터다. 7종목 경로는 2023-05~라 긴 역사 판정이 없다.")
    A("- 효율선초과는 내리기만 한 구간(7월 쇼크·2022)에서는 현금 0%가 기준이 되어 뜻이 없다. 그 구간은 수익과 기본(100%)을 같이 보였다.")
    sl_ = J["st_long"]["QQQ"]
    A(f"- 수수료 0. 미끄러짐 0.5%(QQQ 고른 전략 2019~25 효율선초과 {sl_['res_slip']['chk']['eff'] * 100:+.1f}%p), 현금 이자 4% 판 {sl_['res_cash4']['chk']['eff'] * 100:+.1f}%p"
      f"(이자 0 판 {sl_['res']['chk']['eff'] * 100:+.1f}%p). 매수 추천·주문 수량이 아니다.")
    A("- 다른 해(장세 전환 위험): 재구성 2008(공포가 반년 이어짐)에서 '≤25 매일'은 손해였고, 1991~2010 고르기 구간에서는 1배 지수 기준 0 근처였다.")
    (RESULTS / f"{ID}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


if __name__ == "__main__":
    if sys.argv[1:] == ["md"]:
        print(write_md(json.loads((RESULTS / f"{ID}.json").read_text(encoding="utf-8"))))
    else:
        main()
        write_md(json.loads((RESULTS / f"{ID}.json").read_text(encoding="utf-8")))
