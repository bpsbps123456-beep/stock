"""8단계 갈래 FGREENTRY(2026-10-11): 손절 뒤 '언제 다시 사나'와 '언제 손절을 미루나'에 CNN 공포탐욕지수(F&G)를 쓰면
앞 단계의 시간 기준(손절 뒤 20거래일 시가 재매수)보다 나은가.

사용자(2026-10-11): "그럼 CNN의 공포탐욕지수를 바탕으로 매매에 참고하면? 아니면 이런 것들에서 아이디어 얻어서 통하는 방법을 개발해도 좋고요."

대상·손절(앞 단계 대표를 고정)
  G1 일반 주식 고정 −25%(진입가 대비, 장중): p4_stocks(2014~2026, 생존편향), 자료 오류 의심 종목(trade_lab.bad_bars) 제외.
     진입 = 무작위(10거래일마다 위상 고정, 같은 종목 20거래일 간격 → 사실상 20거래일마다) | dip7(그날 −7% 이하 → 다음 날 시가).
     거래 1건 = 진입일 시가에 사서 최대 126거래일, 평가 구간 끝에서 자른다(앞·뒤 구간이 서로 섞이지 않게). 손절 뒤 재진입하면 그 값에서 다시 −25%.
  G2 단일 종목 2배 합성(lev_lab.synth, 보수 1.05% + 실제 2배 상품 대조 연 로그 비용) 추적 −15%(산 뒤 최고 종가 대비, 장중).
     기초 = p4_meta 2배 기초 − QQQ·SPY·SOXX − 자료 오류(하루 +150%/−70%) − SPAC 전 구간(IONQ·RGTI·OKLO, 6단계와 같음).
  G3 3배 지수 TQQQ·SOXL·KORU 추적 −25%·−30%. 실제 상품 시작 전은 lev5.synth 합성(추가 비용 TQQQ 1.3%·SOXL 1.5%·KORU 2.6%)을 이어 붙임.
  G4 7종목 경로(harness.base('2026')·base('hold')) 그림자 종가가 산 뒤(또는 경로 시작 뒤) 최고 대비 −10%·−15%면 다음 시가에 전량 현금.

재진입 규칙(사전 고정, 모두 최대 대기 126거래일 뒤 무조건 시가 매수)
  R0 바로(손절 다음 거래일 시가) | R1 손절일 + 20·40거래일 시가
  R2 손절일 종가부터 본 F&G 최저 + Δ 이상 회복(Δ 10·15·20·25) | R3 F&G ≥ X(25·35·45·55) | R4 F&G ≤ T인 날(15·20·25, 공포일에 산다)
  R5 먼저 = R1(20) 또는 R3(45) 중 먼저 | R5 나중 = 20거래일이 지났고 그동안 F&G ≥ 45가 한 번이라도 있었음(둘 중 나중)
손절 보류 규칙
  H1 그날 쓸 수 있는 F&G ≤ T(10·15·20·25)이면 손절 주문을 걸지 않는다(F&G > T가 된 날 다시 같은 선으로 판단).
  H2 F&G ≥ G(70·75·80)인 날만 손절 폭을 좁힌다: 폭 ≥ 20%면 −10%p(일반 −25 → −15, 3배 −25 → −15·−30 → −20), 그 밖은 −5%p(2배 −15 → −10, 7종목 −15 → −10·−10 → −5).
격자 = 재진입 16 × 보류 8 = 128조합 × 설정 11(G1 무작위·dip7, G2, G3 3상품 × 2폭, G4 2폭) = 1,408. 기준 B = R1 20일 + H0. 참고: 보유만, 손절만(재진입 없음).

시점
  F&G 값은 그날 미국 장 마감 뒤 확정 → 다음 거래일 시가 판단에 쓴다(FGP[t] = t보다 앞선 날의 최신 값). 2021-01까지 GitHub 보관본, 그 뒤 CNN 공식(fg_data.fg()).
  장중 손절(G1~G3): t일 시가 ≤ 선이면 시가, 저가 ≤ 선이면 선 가격. 선은 t−1일까지 정보(산 뒤 최고 종가·진입가). 같은 날 순서: 시가 재매수 → 손절.
  G4: t일 종가 신호 → harness.at_open으로 다음 시가 노출.
  옛 지수(시가가 전일 종가로 채워진 2001년 이전)를 쓰는 재구성 구간(G3 1991~2010 고르기, 2011~2025 확인)은 '종가 모드':
    t일 종가 판단 → t+1일 종가 체결(손절도 종가가 선 아래인 날 다음 날 종가에 판다). 같은 비교를 위해 2011~2025 확인도 종가 모드로 돈다.
  하루 늦춘 판 = F&G를 하루 더 늦게 안다고 보고(F&G 값 자체를 1거래일 밀어서) 같은 규칙을 다시 돌린다.

F&G 재구성(G3 1991~2010 전용, cache/p8_FGREENTRY_recon.parquet)
  구성요소 5개(모두 t일 종가까지): S&P/125일선 − 1, S&P/252일 최고 − 1(신고가 강도 대용), −(VIX/50일선 − 1),
  안전자산 = S&P 20일 수익 − 10년물 20일 수익(^TNX로 듀레이션 8 근사), 정크 = VWEHX(하이일드 펀드) 20일 수익 − 10년물 20일 수익.
  각 요소를 지난 252일 평균·표준편차로 z → 정규분포 백분위 → 평균. 2011~2018 실제 F&G 분포에 분위수로 맞춘다(2019~2025는 표본 밖 점검).

평가(사전 고정)
  G1: 거래당 수익 평균(규칙 − B, %p, 진입 월 묶음 t), 승률, 거래당 −30% 이하 확률. 앞 = 2026-01-02~05-29 진입(5/29에서 자름),
      뒤 = 2026-06-01~10-09 진입(10/9에서 자름), 2025 = 2025 진입(12/31에서 자름), 긴 역사 2014~2018 진입 → 2019~2025 진입.
      7월 AI 쇼크 = 6/22 시가에 전 종목을 산 묶음의 7/29 종가 평균(그리고 10/9).
  G2: 기초별 2026 경로를 잘라 앞·뒤·AI 쇼크 수익, 기초 중앙값(규칙 − B). 2025·긴 역사는 상품-해(그해 첫날 들고 시작) 중앙값, 한 해 −80% 이하 확률.
  G3: 상품별 연속 경로. 2026 앞·뒤·AI 쇼크(잘라서), 2025, 2011~2018 → 2019~2025(연율), 재구성 1991~2010 → 2011~2025(연율, 종가 모드).
  G4: harness 기간(2026 앞·뒤·AI쇼크, 보유 경로 2025).
  모든 설정: 효율선초과(같은 최대 낙폭이 나오는 고정 비중 대비), 같은 평균 노출 고정 대비, 손절 주기(판 값/다시 산 값 − 1)의 승률·손익비·기대값, 휩소 비율.
고르기(사전 고정)
  설정마다 128조합 중 2026 앞 'B 대비'가 가장 큰 조합 → 뒤에서도 B를 이기면 통과(2025는 보조). 대표 후보 = 통과한 설정 고르기 가운데 앞 우위가 가장 큰 것.
  긴 역사: 고르기 구간 'B 대비' 최대 조합 → 확인 구간에서 B를 이기면 통과(F&G 실제, G3는 재구성도 따로).
  위약: F&G를 63일 이상 원형으로 민 20판(이동량 63 + i × (N − 126)/19). 고른 조합의 순위와, 설정 안 F&G 조합 전체의 '뒤에서 B를 이긴 비율'을 위약 평균과 비교.
비용: 수수료 0, 미끄러짐 0.5% 민감도(체결마다 불리), 현금 이자 0(4% 판 한 줄).

실행: cd code && PYTHONIOENCODING=utf-8 py p8_FGREENTRY.py        (재구성 F&G가 없으면 yfinance로 ^TNX·VWEHX를 받아 만든다)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fg_data  # noqa: E402
import harness as HN  # noqa: E402
import lev5 as L5  # noqa: E402
import lev_lab as LL  # noqa: E402
import stock_lab as S  # noqa: E402
import trade_lab as TL  # noqa: E402

ID = "p8_FGREENTRY"
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
TMP = os.environ.get("P8_TMP")            # 중간 결과(JSON) 저장 폴더(선택, 저장소 밖)

# ============================================================================ 사전 고정 격자·상수(확인 구간을 본 뒤 바꾸지 않는다)
MAXWAIT = 126
RE_RULES = [("R0 바로", 0, 0, 0), ("R1 20일", 1, 20, 0), ("R1 40일", 1, 40, 0),
            ("R2 저점+10", 2, 10, 0), ("R2 저점+15", 2, 15, 0), ("R2 저점+20", 2, 20, 0), ("R2 저점+25", 2, 25, 0),
            ("R3 ≥25", 3, 25, 0), ("R3 ≥35", 3, 35, 0), ("R3 ≥45", 3, 45, 0), ("R3 ≥55", 3, 55, 0),
            ("R4 ≤15", 4, 15, 0), ("R4 ≤20", 4, 20, 0), ("R4 ≤25", 4, 25, 0),
            ("R5 먼저", 5, 20, 45), ("R5 나중", 6, 20, 45)]
HOLD_RULES = [("H0", 0, 0), ("H1 ≤10", 10, 0), ("H1 ≤15", 15, 0), ("H1 ≤20", 20, 0), ("H1 ≤25", 25, 0),
              ("H2 ≥70", 0, 70), ("H2 ≥75", 0, 75), ("H2 ≥80", 0, 80)]
COMBOS = [(r, h) for r in RE_RULES for h in HOLD_RULES]          # 128
REFS = {"보유만": ("hold",), "손절만": ("stoponly",)}
B_KEY = "R1 20일|H0"
NO_FG = {"R0 바로|H0", "R1 20일|H0", "R1 40일|H0"}                 # F&G를 쓰지 않는 조합(위약 제외)
SETTINGS = [  # (이름, 갈래, 대상, 손절 폭, 손절 종류)
    ("G1 무작위 −25%", "G1", "random", 0.25, 0), ("G1 급락(−7%) −25%", "G1", "dip7", 0.25, 0),
    ("G2 2배 추적 −15%", "G2", None, 0.15, 1),
    ("G3 TQQQ −25%", "G3", "TQQQ", 0.25, 1), ("G3 TQQQ −30%", "G3", "TQQQ", 0.30, 1),
    ("G3 SOXL −25%", "G3", "SOXL", 0.25, 1), ("G3 SOXL −30%", "G3", "SOXL", 0.30, 1),
    ("G3 KORU −25%", "G3", "KORU", 0.25, 1), ("G3 KORU −30%", "G3", "KORU", 0.30, 1),
    ("G4 7종목 −10%", "G4", None, 0.10, 1), ("G4 7종목 −15%", "G4", None, 0.15, 1)]
G1_H, G1_GAP, G1_EVERY = 126, 20, 10
SLIP, CASH_RATE = 0.005, 0.04
PLACEBO_N, PLACEBO_MIN = 20, 63
FEE2, EXTRA_FALLBACK = 0.0105, 0.10
SPAC_START = {"IONQ": "2021-10-01", "RGTI": "2022-03-02", "OKLO": "2024-05-10"}
G3_SPEC = {"TQQQ": ("^NDX", None, 0.013), "SOXL": ("^SOX", "SOXX", 0.015), "KORU": ("EWY", None, 0.026)}
D26 = ("2026-01-02", "2026-10-09")
FRONT, BACK, SHOCK = ("2026-01-02", "2026-05-29"), ("2026-06-01", "2026-10-09"), ("2026-06-22", "2026-07-29")
G1_RUNS = {"LP": ("2014-01-02", "2018-12-31"), "LC": ("2019-01-02", "2025-12-31"), "앞": FRONT, "2026": D26}
G3_RUNS = {"LP": ("2011-01-03", "2018-12-31"), "LC": ("2019-01-02", "2025-12-31"), "2025": ("2025-01-02", "2025-12-31"),
           "2026": D26, "RP": ("1991-01-02", "2010-12-31"), "RC": ("2011-01-03", "2025-12-31")}
G2_YEARS = list(range(2014, 2026))
WIN_ORDER = ["앞", "뒤", "AI", "2026", "2025", "LP", "LC", "RP", "RC"]


def narrow_of(x: float) -> float:
    return round(x - 0.10, 4) if x >= 0.20 - 1e-9 else round(x - 0.05, 4)


def ckey(r, h) -> str:
    return f"{r[0]}|{h[0]}"


# ============================================================================ F&G 자료
def fg_actual() -> pd.Series:
    s = fg_data.fg()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _yf_raw() -> pd.DataFrame:
    p = CACHE / f"{ID}_yf.parquet"
    if p.exists():
        return pd.read_parquet(p)
    import yfinance as yf
    d = yf.download(["^TNX", "VWEHX"], start="1988-01-01", end="2026-10-10", auto_adjust=False, progress=False)
    out = pd.DataFrame({"TNX": d["Close"]["^TNX"], "VWEHX": d["Adj Close"]["VWEHX"]})
    out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    out.to_parquet(p)
    return out


def build_recon() -> pd.DataFrame:
    p = CACHE / f"{ID}_recon.parquet"
    if p.exists():
        return pd.read_parquet(p)
    from scipy.stats import norm
    idx = pd.read_parquet(CACHE / "p7_idx.parquet")
    px = idx.pivot_table(index="date", columns="sym", values="Close").sort_index()
    px.index = pd.to_datetime(px.index).strftime("%Y-%m-%d")
    yf_ = _yf_raw()
    days = px.index[px.index >= "1989-01-01"]
    spx = px["^GSPC"].reindex(days).ffill()
    vix = px["^VIX"].reindex(days).ffill()
    y = yf_["TNX"].reindex(days).ffill() / 100.0
    hy = yf_["VWEHX"].reindex(days).ffill()
    rb = y.shift(1) / 252 - 8.0 * (y - y.shift(1))
    bond20 = (1 + rb.fillna(0)).rolling(20).apply(np.prod, raw=True) - 1
    comp = pd.DataFrame({
        "momentum": spx / spx.rolling(125).mean() - 1,
        "strength": spx / spx.rolling(252, min_periods=60).max() - 1,
        "vix": -(vix / vix.rolling(50).mean() - 1),
        "safe_haven": spx.pct_change(20) - bond20,
        "junk": hy.pct_change(20) - bond20})
    z = (comp - comp.rolling(252, min_periods=126).mean()) / comp.rolling(252, min_periods=126).std()
    sc = pd.DataFrame(norm.cdf(z.to_numpy()) * 100, index=z.index, columns=z.columns).where(z.notna())
    avg = sc.mean(axis=1, skipna=False)
    act = fg_actual()
    fit = avg[(avg.index >= "2011-01-03") & (avg.index <= "2018-12-31")].dropna()
    fa = act[(act.index >= "2011-01-03") & (act.index <= "2018-12-31")]
    qs_avg = np.sort(fit.to_numpy())
    qs_act = np.sort(fa.to_numpy())
    pct_ = np.searchsorted(qs_avg, avg.fillna(-1).to_numpy(), side="right") / len(qs_avg)
    recon = pd.Series(np.where(avg.notna(), np.quantile(qs_act, np.clip(pct_, 0, 1)), np.nan), index=avg.index)
    out = sc.copy()
    out["avg"] = avg
    out["recon"] = recon
    out = out[out.index >= "1990-01-01"]
    out.to_parquet(p)
    return out


def recon_check(rc: pd.DataFrame) -> dict:
    act = fg_actual()
    out = {}
    for name, (a, b) in {"맞춤 2011~2018": ("2011-01-03", "2018-12-31"), "표본 밖 2019~2025": ("2019-01-02", "2025-12-31"),
                         "2026": D26}.items():
        x = rc["recon"][(rc.index >= a) & (rc.index <= b)]
        yv = act.reindex(x.index)
        m = x.notna() & yv.notna()
        out[name] = {"상관": float(np.corrcoef(x[m], yv[m])[0, 1]), "평균절대차": float((x[m] - yv[m]).abs().mean()),
                     "실제≤20일": int((yv[m] <= 20).sum()), "재구성≤20일": int((x[m] <= 20).sum()), "일수": int(m.sum())}
    r = rc["recon"]
    out["재구성 1991~2010 ≤20일(해마다)"] = {k: int(v) for k, v in
                                     r[(r.index >= "1991-01-01") & (r.index <= "2010-12-31") & (r <= 20)]
                                     .groupby(lambda d: d[:4]).size().items()}
    return out


def fg_variant(base: pd.Series, kind: str, k: int = 0) -> pd.Series:
    """base(날짜 → 값)의 변형. lag1 = 값을 1거래일 뒤로 민다(하루 늦게 안다). shift = 값만 k칸 원형 이동(날짜 그대로)."""
    if kind == "real":
        return base
    if kind == "lag1":
        return base.shift(1).dropna()
    if kind == "shift":
        return pd.Series(np.roll(base.to_numpy(), k), index=base.index)
    raise ValueError(kind)


def fg_map(days: np.ndarray, s: pd.Series):
    """(FGC, FGP): FGC[t] = t일 종가 기준 최신 값(날짜 ≤ t), FGP[t] = t일 시가에 쓸 값(날짜 < t). 없으면 NaN."""
    d = s.index.to_numpy().astype("U10")
    v = s.to_numpy(float)
    i = np.searchsorted(d, days, side="right") - 1
    j = np.searchsorted(d, days, side="left") - 1
    fgc = np.where(i >= 0, v[np.clip(i, 0, None)], np.nan)
    fgp = np.where(j >= 0, v[np.clip(j, 0, None)], np.nan)
    return fgc, fgp


def placebo_shifts(n: int) -> list[int]:
    return [int(PLACEBO_MIN + round(i * (n - 2 * PLACEBO_MIN) / (PLACEBO_N - 1))) for i in range(PLACEBO_N)]


# ============================================================================ 엔진(numba)
@njit(cache=True)
def _run(O, Hh, Lw, C, FGP, FGC, a, z, start_open, mode, stop_kind, x, narrow, h1T, h2G,
         rkind, rp1, rp2, maxwait, slip, cash_d, vals, cyc):
    """a..z. start_open=True면 a일 시가에 산다, 아니면 a−1 종가에 들고 시작(가치 1).
    mode 0 = 장중 손절·시가 재매수, mode 1 = 종가 모드(t 종가 판단 → t+1 종가 체결).
    stop_kind −1 = 손절 없음, 0 = 진입가 대비 고정, 1 = 산 뒤 최고 종가 대비 추적.
    rkind 0 바로, 1 고정 rp1일, 2 F&G 최저+rp1, 3 F&G≥rp1, 4 F&G≤rp1, 5 (rp1일 | F&G≥rp2), 6 (rp1일 & 그동안 F&G≥rp2), 9 재진입 없음.
    cyc 행 = (판 t, 다시 산 t 또는 −1, 판 가격, 다시 산 가격 또는 끝 종가, 열린 주기 1)."""
    if start_open:
        entry = O[a] * (1.0 + slip)
    else:
        entry = C[a - 1]
    sh = 1.0 / entry
    cash = 0.0
    held = True
    peak = entry
    exit_t = -1
    exit_px = 0.0
    fgmin = 1e9
    seen = False
    pend = False
    ncyc = 0
    vals[0] = 1.0
    for t in range(a, z + 1):
        fillp = O[t] if mode == 0 else C[t]
        if mode == 1 and held and pend:                      # 종가 모드: 전날 종가 신호 → 오늘 종가 매도
            exit_px = C[t] * (1.0 - slip)
            cash = sh * exit_px
            sh = 0.0
            held = False
            pend = False
            exit_t = t
            fgmin = 1e9
            seen = False
        if (not held) and exit_t >= 0 and t > exit_t:
            d = t - exit_t
            go = False
            if rkind != 9 and d >= maxwait:
                go = True
            elif rkind == 0:
                go = True
            elif rkind == 1:
                go = d >= rp1
            elif rkind == 6:
                go = (d >= rp1) and seen
            elif rkind != 9:
                f = FGP[t]
                ok = f == f
                if rkind == 2:
                    go = ok and (f >= fgmin + rp1)
                elif rkind == 3:
                    go = ok and (f >= rp1)
                elif rkind == 4:
                    go = ok and (f <= rp1)
                elif rkind == 5:
                    go = (d >= rp1) or (ok and f >= rp2)
            if go:
                fill = fillp * (1.0 + slip)
                sh = cash / fill
                cash = 0.0
                held = True
                entry = fill
                peak = fill
                cyc[ncyc, 0] = exit_t
                cyc[ncyc, 1] = t
                cyc[ncyc, 2] = exit_px
                cyc[ncyc, 3] = fill
                cyc[ncyc, 4] = 0.0
                ncyc += 1
        if mode == 0 and held and stop_kind >= 0:
            f = FGP[t]
            active = True
            if h1T > 0 and f == f and f <= h1T:
                active = False
            if active:
                xe = x
                if h2G > 0 and f == f and f >= h2G:
                    xe = narrow
                ref = entry if stop_kind == 0 else peak
                line = ref * (1.0 - xe)
                px = -1.0
                if O[t] <= line:
                    px = O[t]
                elif Lw[t] <= line:
                    px = line
                if px > 0:
                    exit_px = px * (1.0 - slip)
                    cash = sh * exit_px
                    sh = 0.0
                    held = False
                    exit_t = t
                    fgmin = 1e9
                    seen = False
        if held:
            if C[t] > peak:
                peak = C[t]
            if mode == 1 and stop_kind >= 0:
                g = FGC[t]
                active = True
                if h1T > 0 and g == g and g <= h1T:
                    active = False
                if active:
                    xe = x
                    if h2G > 0 and g == g and g >= h2G:
                        xe = narrow
                    ref = entry if stop_kind == 0 else peak
                    if C[t] <= ref * (1.0 - xe):
                        pend = True
        else:
            cash = cash * (1.0 + cash_d)
            g = FGC[t]
            if g == g:
                if g < fgmin:
                    fgmin = g
                if rp2 > 0 and g >= rp2:
                    seen = True
        vals[t - a + 1] = cash + sh * C[t]
    if (not held) and exit_t >= 0:
        cyc[ncyc, 0] = exit_t
        cyc[ncyc, 1] = -1
        cyc[ncyc, 2] = exit_px
        cyc[ncyc, 3] = C[z]
        cyc[ncyc, 4] = 1.0
        ncyc += 1
    return ncyc


@njit(cache=True)
def _run_trades(O, Hh, Lw, C, FGP, FGC, A, Z, stop_kind, x, narrow, h1T, h2G, rkind, rp1, rp2, maxwait, slip, cash_d,
                maxlen, out):
    """거래마다 _run(시가 매수). out 열: 0 수익, 1 보유만, 2 주기 수, 3 이긴 주기, 4 주기 수익 합, 5 이긴 합, 6 진 합,
    7 휩소(다시 산 값 > 판 값), 8 현금 일수, 9 거래 일수."""
    vals = np.empty(maxlen + 2)
    cyc = np.empty((maxlen + 2, 5))
    for j in range(len(A)):
        a = A[j]
        z = Z[j]
        n = _run(O, Hh, Lw, C, FGP, FGC, a, z, True, 0, stop_kind, x, narrow, h1T, h2G, rkind, rp1, rp2, maxwait,
                 slip, cash_d, vals, cyc)
        out[j, 0] = vals[z - a + 1] - 1.0
        out[j, 1] = C[z] / (O[a] * (1.0 + slip)) - 1.0
        nw = 0
        sg = 0.0
        sw = 0.0
        sl = 0.0
        wh = 0
        dout = 0.0
        for k in range(n):
            g = cyc[k, 2] / cyc[k, 3] - 1.0
            sg += g
            if g > 0:
                nw += 1
                sw += g
            else:
                sl += g
            if g < 0:
                wh += 1
            if cyc[k, 4] > 0.5:
                dout += z - cyc[k, 0] + 1
            else:
                dout += cyc[k, 1] - cyc[k, 0]
        out[j, 2] = n
        out[j, 3] = nw
        out[j, 4] = sg
        out[j, 5] = sw
        out[j, 6] = sl
        out[j, 7] = wh
        out[j, 8] = dout
        out[j, 9] = z - a + 1


def rule_args(key: str, x: float):
    """조합 키 → (stop_kind 덮어쓰기, x, narrow, h1T, h2G, rkind, rp1, rp2, maxwait)."""
    if key == "보유만":
        return (-1, x, x, 0.0, 0.0, 9, 0.0, 0.0, 10 ** 9)
    if key == "손절만":
        return (None, x, x, 0.0, 0.0, 9, 0.0, 0.0, 10 ** 9)
    rname, hname = key.split("|")
    r = next(v for v in RE_RULES if v[0] == rname)
    h = next(v for v in HOLD_RULES if v[0] == hname)
    return (None, x, narrow_of(x), float(h[1]), float(h[2]), r[1], float(r[2]), float(r[3]), MAXWAIT)


ALL_KEYS = [ckey(r, h) for r, h in COMBOS] + list(REFS)


# ============================================================================ 공통 계산
def mdd(v) -> float:
    v = np.asarray(v, float)
    return float((v / np.maximum.accumulate(v) - 1).min())


def cagr(ret: float, years: float) -> float:
    return float((1 + ret) ** (1 / years) - 1) if ret > -1 else -1.0


def frontier(r_daily: np.ndarray, rule_ret: float, rule_mdd: float, years: float | None = None) -> tuple[float, float]:
    """같은 최대 낙폭 고정 비중 효율선 대비 초과. r_daily = 대상 1배 보유의 일수익(그 구간). years가 있으면 연율 차."""
    best, be = -1.0, 0.0
    for w in np.round(np.arange(0, 1.0001, 0.01), 2):
        cur = np.concatenate([[1.0], np.cumprod(1 + w * r_daily)])
        if mdd(cur) >= rule_mdd - 1e-12 and cur[-1] - 1 > best:
            best, be = cur[-1] - 1, w
    if years:
        return float(cagr(rule_ret, years) - cagr(best, years)), float(be)
    return float(rule_ret - best), float(be)


def fixed_expo(r_daily: np.ndarray, w: float, years: float | None = None) -> float:
    v = float(np.prod(1 + w * r_daily) - 1)
    return cagr(v, years) if years else v


def cyc_stats(gains: np.ndarray) -> dict:
    g = np.asarray(gains, float)
    if not len(g):
        return {"n": 0}
    win, loss = g[g > 0], g[g <= 0]
    return {"n": int(len(g)), "승률": float(len(win) / len(g)), "평균이익": float(win.mean()) if len(win) else float("nan"),
            "평균손실": float(loss.mean()) if len(loss) else float("nan"),
            "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
            "기대값": float(g.mean()), "휩소": float((g < 0).mean())}


# ============================================================================ G1 일반 주식
class G1Data:
    def __init__(self):
        t0 = time.time()
        data = TL.load_stocks()
        syms = [s for s in sorted(data) if not TL.bad_bars(data[s])]
        self.excluded = len(data) - len(syms)
        self.syms = syms
        segs, off = [], 0
        self.off = {}
        for s in syms:
            b = data[s]
            self.off[s] = off
            segs.append(b)
            off += len(b.c)
        self.O = np.concatenate([b.o for b in segs]).astype(float)
        self.H = np.concatenate([b.h for b in segs]).astype(float)
        self.L = np.concatenate([b.l for b in segs]).astype(float)
        self.C = np.concatenate([b.c for b in segs]).astype(float)
        self.days = np.concatenate([np.asarray(b.days).astype("U10") for b in segs])
        self.bars = {s: data[s] for s in syms}
        self._ent = {}
        print(f"G1 자료 {len(syms)}종목(오류 의심 제외 {self.excluded}) {time.time() - t0:.0f}s", flush=True)

    def fg(self, s: pd.Series):
        fgc, fgp = fg_map(self.days, s)
        return fgp, fgc

    def entries(self, kind: str, d0: str, d1: str):
        """(전역 a, 전역 z, 진입일, 종목) — 진입일 ≤ d1, 거래는 min(e+125, d1 마지막 거래일)에서 끝."""
        key = (kind, d0, d1)
        if key in self._ent:
            return self._ent[key]
        A, Z, D, SY = [], [], [], []
        for s in self.syms:
            b = self.bars[s]
            c = b.c
            n = len(c)
            last = b.idx(d1, "right") - 1
            s_lo, s_hi = max(b.idx(d0) - 1, 1), min(last - 1, n - 2)
            if s_hi < s_lo or last < 1:
                continue
            ss = np.arange(s_lo, s_hi + 1)
            if kind == "random":
                phase = sum(map(ord, s)) % G1_EVERY
                sig = (ss % G1_EVERY) == phase
            else:
                sig = c[ss] / c[ss - 1] - 1 <= -0.07
            prev = -10 ** 9
            o = self.off[s]
            for si in ss[sig]:
                if si - prev >= G1_GAP:
                    e = si + 1
                    z = min(e + G1_H - 1, last)
                    if z >= e:
                        A.append(o + e)
                        Z.append(o + z)
                        D.append(b.days[e])
                        SY.append(s)
                    prev = si
        out = (np.array(A, np.int64), np.array(Z, np.int64), np.array(D).astype("U10"), np.array(SY))
        self._ent[key] = out
        return out

    def cohort(self, d0: str, d1: str):
        A, Z = [], []
        for s in self.syms:
            b = self.bars[s]
            a, z = b.idx(d0), b.idx(d1, "right") - 1
            if a < 1 or z <= a or b.days[a] != d0:
                continue
            A.append(self.off[s] + a)
            Z.append(self.off[s] + z)
        return np.array(A, np.int64), np.array(Z, np.int64)


def g1_run(G: G1Data, A, Z, key: str, x: float, fgp, fgc, slip=0.0, cash_rate=0.0) -> np.ndarray:
    sk, x_, nar, h1, h2, rk, p1, p2, mw = rule_args(key, x)
    sk = 0 if sk is None else sk
    out = np.empty((len(A), 10))
    if len(A):
        _run_trades(G.O, G.H, G.L, G.C, fgp, fgc, A, Z, sk, x_, nar, h1, h2, rk, p1, p2, mw, slip, cash_rate / 252,
                    int((Z - A).max()) + 1, out)
    return out


def g1_summary(out: np.ndarray, outB: np.ndarray, dates: np.ndarray) -> dict:
    if not len(out):
        return {"n": 0}
    r, bh = out[:, 0], out[:, 1]
    d = r - outB[:, 0]
    m = pd.Series(d).groupby(pd.Series(dates).str[:7].to_numpy()).mean()
    sd = m.std(ddof=1) if len(m) > 2 else 0.0
    t = float(m.mean() / (sd / np.sqrt(len(m)))) if sd and sd > 0 else float("nan")
    nc = out[:, 2].sum()
    nw = out[:, 3].sum()
    cy = {"n": int(nc)}
    if nc > 0:
        aw = out[:, 5].sum() / nw if nw else float("nan")
        al = out[:, 6].sum() / (nc - nw) if nc - nw else float("nan")
        cy = {"n": int(nc), "승률": float(nw / nc), "평균이익": float(aw), "평균손실": float(al),
              "손익비": float(aw / -al) if nw and nc - nw and al < 0 else float("nan"), "기대값": float(out[:, 4].sum() / nc),
              "휩소": float(out[:, 7].sum() / nc)}
    return {"n": int(len(r)), "평균": float(r.mean()), "보유만": float(bh.mean()), "B대비": float(d.mean()), "B대비_t": t,
            "보유대비": float((r - bh).mean()), "승률": float((r > 0).mean()), "P(-30%)": float((r <= -0.30).mean()),
            "P(-50%)": float((r <= -0.50).mean()), "하위5%": float(np.quantile(r, 0.05)),
            "노출": float(1 - out[:, 8].sum() / out[:, 9].sum()), "손절/거래": float(nc / len(r)), "주기": cy}


G1_SUBS = {"LP": [("LP", None)], "LC": [("LC", None), ("2025", "2025-01-01")], "앞": [("앞", None)],
           "2026": [("2026", None), ("뒤", "2026-06-01")]}


def g1_setting(G: G1Data, kind: str, x: float, fgs, keys=None, windows=None, slip=0.0, cash_rate=0.0) -> dict:
    fgp, fgc = fgs
    keys = keys or ALL_KEYS
    res = {k: {} for k in keys}
    for rn, (d0, d1) in G1_RUNS.items():
        subs = [(wn, lo) for wn, lo in G1_SUBS[rn] if not windows or wn in windows]
        if not subs:
            continue
        A, Z, D, SY = G.entries(kind, d0, d1)
        outB = g1_run(G, A, Z, B_KEY, x, fgp, fgc, slip, cash_rate)
        for k in keys:
            out = outB if k == B_KEY else g1_run(G, A, Z, k, x, fgp, fgc, slip, cash_rate)
            for wn, lo in subs:
                m = D >= lo if lo else np.ones(len(D), bool)
                res[k][wn] = g1_summary(out[m], outB[m], D[m])
    return res


def g1_shock(G: G1Data, x: float, keys, fgp, fgc, slip=0.0, cash_rate=0.0) -> dict:
    """6/22 시가에 전 종목을 산 묶음: 7/29 종가와 10/9 종가 평균 수익."""
    out = {}
    for lab, d1 in (("AI", "2026-07-29"), ("AI~10/9", "2026-10-09")):
        A, Z = G.cohort("2026-06-22", d1)
        oB = g1_run(G, A, Z, B_KEY, x, fgp, fgc, slip, cash_rate)
        D = np.full(len(A), "2026-06-22")
        for k in keys:
            o = oB if k == B_KEY else g1_run(G, A, Z, k, x, fgp, fgc, slip, cash_rate)
            out.setdefault(k, {})[lab] = g1_summary(o, oB, D)
    return out


# ============================================================================ G2·G3 경로
class Inst:
    def __init__(self, name, b: S.Bars, i0: int = 0):
        self.name = name
        self.days = np.asarray(b.days).astype("U10")[i0:]
        self.O, self.H, self.L, self.C = (np.asarray(v, float)[i0:].copy() for v in (b.o, b.h, b.l, b.c))
        self.cache = {}
        self.actual_start = None

    def idx(self, d, side="left"):
        return int(np.searchsorted(self.days, d, side=side))

    def fg(self, tag: str, s: pd.Series):
        if tag not in self.cache:
            fgc, fgp = fg_map(self.days, s)
            self.cache[tag] = (fgp, fgc)
        return self.cache[tag]


def path_run(inst: Inst, d0: str, d1: str, key: str, x: float, stop_kind: int, fgp, fgc, mode=0, slip=0.0, cash_rate=0.0):
    a, z = inst.idx(d0), inst.idx(d1, "right") - 1
    if a < 1 or z <= a:
        return None
    sk, x_, nar, h1, h2, rk, p1, p2, mw = rule_args(key, x)
    sk = stop_kind if sk is None else sk
    vals = np.empty(z - a + 2)
    cyc = np.empty((z - a + 3, 5))
    n = _run(inst.O, inst.H, inst.L, inst.C, fgp, fgc, a, z, False, mode, sk, x_, nar, h1, h2, rk, p1, p2, mw, slip,
             cash_rate / 252, vals, cyc)
    return {"a": a, "z": z, "vals": vals.copy(), "cyc": cyc[:n].copy()}


def slice_stats(inst: Inst, pr: dict, d0: str, d1: str, annual: bool = False) -> dict | None:
    a = pr["a"]
    i0 = inst.idx(d0) - a
    i1 = inst.idx(d1, "right") - 1 - a + 1
    if i0 < 0 or i1 <= i0:
        return None
    v = pr["vals"][i0:i1 + 1]
    v = v / v[0]
    ret = float(v[-1] - 1)
    md = mdd(v)
    c = inst.C[a - 1 + i0:a + i1]
    r_u = c[1:] / c[:-1] - 1
    yrs = (i1 - i0) / 252 if annual else None
    fx, fw = frontier(r_u, ret, md, yrs)
    cyc = pr["cyc"]
    lo_t, hi_t = a + i0, a + i1 - 1
    sel = (cyc[:, 0] >= lo_t) & (cyc[:, 0] <= hi_t) if len(cyc) else np.zeros(0, bool)
    g = (cyc[sel, 2] / cyc[sel, 3] - 1) if len(cyc) else np.zeros(0)
    out_days = 0
    for row in cyc:
        e, rr = int(row[0]), int(row[1])
        end = pr["z"] if rr < 0 else rr - 1
        lo, hi = max(e, lo_t), min(end, hi_t)
        if hi >= lo:
            out_days += hi - lo + 1
    expo = 1 - out_days / max(i1 - i0, 1)
    same = fixed_expo(r_u, expo, yrs)
    rr_ = cagr(ret, yrs) if yrs else ret
    return {"수익": rr_, "누적": ret, "낙폭": md, "효율선초과": fx, "효율선_w": fw,
            "노출": float(expo), "같은노출초과": rr_ - same,
            "보유": cagr(float(c[-1] / c[0] - 1), yrs) if yrs else float(c[-1] / c[0] - 1),
            "손절": int(sel.sum()), "주기": cyc_stats(g), "연수": (i1 - i0) / 252}


def build_g2() -> tuple[dict, list, dict]:
    lv = LL.load()
    meta = LL.products()
    unders = sorted({v["under"] for v in meta.values() if v["L"] == 2} - {"QQQ", "SPY", "SOXX"})
    excl = [u for u in unders if u in lv and TL.bad_bars(lv[u])]
    insts, extra = {}, {}
    for u in unders:
        if u in excl or u not in lv:
            continue
        drags = []
        for p, info in meta.items():
            if info["under"] == u and info["L"] == 2:
                v = LL.validate(p)
                if "연비용_로그" in v and v.get("일수", 0) >= 60:
                    drags.append(v["연비용_로그"])
        ex = max(0.0, float(np.median(drags))) if drags else EXTRA_FALLBACK
        extra[u] = ex
        b = LL.synth(u, 2, fee=FEE2 + ex)
        i0 = int(np.searchsorted(np.asarray(b.days).astype("U10"), SPAC_START[u])) if u in SPAC_START else 0
        insts[u] = Inst(u, b, i0)
    return insts, excl, extra


def build_g3() -> dict:
    out = {}
    for prod, (u1, u2, ex) in G3_SPEC.items():
        und = L5.splice(u1, u2, "2001-07-16") if u2 else L5.bars(u1)
        syn = L5.synth(und, 3.0, 0.0095, ex)
        act = L5.bars(prod)
        a0 = act.days[0]
        i = int(np.searchsorted(syn.days, a0))
        scale = act.c[0] / syn.c[i]
        cat = lambda s_, a_: np.concatenate([s_[:i] * scale, a_])  # noqa: E731
        b = S.Bars(prod, np.concatenate([syn.days[:i], act.days]), cat(syn.o, act.o), cat(syn.h, act.h), cat(syn.l, act.l),
                   cat(syn.c, act.c), np.concatenate([syn.v[:i], act.v]))
        out[prod] = Inst(prod, b)
        out[prod].actual_start = str(a0)
    return out


G2_WINS_2026 = {"앞": FRONT, "뒤": BACK, "AI": SHOCK, "2026": D26}


def g2_measure(insts: dict, key: str, x: float, fgtag: str, fgser: pd.Series, windows=None, slip=0.0, cash_rate=0.0) -> dict:
    """기초별 결과: {창: {기초: stats}}. 2026 창은 2026 경로를 자름, 해 창은 그해 첫날 들고 시작."""
    res = {}
    need26 = windows is None or bool(set(windows) & set(G2_WINS_2026))
    years = [y for y in G2_YEARS if windows is None or ("2025" in windows and y == 2025)
             or ("LP" in windows and y <= 2018) or ("LC" in windows and y >= 2019)]
    for u, inst in insts.items():
        fgp, fgc = inst.fg(fgtag, fgser)
        if need26 and inst.days[0] <= "2025-06-30":                          # 2026 시작 전 자료 126일 이상
            pr = path_run(inst, D26[0], D26[1], key, x, 1, fgp, fgc, 0, slip, cash_rate)
            if pr is not None:
                for wn, (d0, d1) in G2_WINS_2026.items():
                    if windows and wn not in windows:
                        continue
                    st = slice_stats(inst, pr, d0, d1)
                    if st:
                        res.setdefault(wn, {})[u] = st
        for y in years:
            d0, d1 = (f"{y}-01-01", f"{y}-12-31")
            a = inst.idx(d0)
            if a < (1 if y == 2014 else 126):
                continue
            pr = path_run(inst, d0, d1, key, x, 1, fgp, fgc, 0, slip, cash_rate)
            if pr is None:
                continue
            st = slice_stats(inst, pr, d0, d1)
            if st:
                res.setdefault(f"Y{y}", {})[u] = st
    return res


def _pool_cyc(lst) -> dict:
    n = sum(c.get("n", 0) for c in lst)
    if not n:
        return {"n": 0}
    w = [c for c in lst if c.get("n", 0)]
    return {"n": int(n), "승률": float(sum(c["승률"] * c["n"] for c in w) / n),
            "기대값": float(sum(c["기대값"] * c["n"] for c in w) / n), "휩소": float(sum(c["휩소"] * c["n"] for c in w) / n)}


def g2_agg(per: dict, perB: dict) -> dict:
    """창별: 기초(또는 상품-해) 중앙값과 B 대비 차의 중앙값·이긴 비율."""
    out = {}
    pools = {"앞": ["앞"], "뒤": ["뒤"], "AI": ["AI"], "2026": ["2026"], "2025": ["Y2025"],
             "LP": [f"Y{y}" for y in range(2014, 2019)], "LC": [f"Y{y}" for y in range(2019, 2026)]}
    for wn, srcs in pools.items():
        cells = [(s, u) for s in srcs for u in per.get(s, {}) if u in perB.get(s, {})]
        if not cells:
            continue
        r = np.array([per[s][u]["수익"] for s, u in cells])
        rb = np.array([perB[s][u]["수익"] for s, u in cells])
        d = r - rb
        out[wn] = {"n": len(cells), "중앙": float(np.median(r)), "평균": float(r.mean()), "B대비": float(np.median(d)),
                   "B대비_평균": float(d.mean()), "B이긴비율": float((d > 1e-12).mean()),
                   "보유": float(np.median([per[s][u]["보유"] for s, u in cells])),
                   "보유대비": float(np.median([per[s][u]["수익"] - per[s][u]["보유"] for s, u in cells])),
                   "효율선초과": float(np.median([per[s][u]["효율선초과"] for s, u in cells])),
                   "같은노출초과": float(np.median([per[s][u]["같은노출초과"] for s, u in cells])),
                   "낙폭": float(np.median([per[s][u]["낙폭"] for s, u in cells])),
                   "노출": float(np.mean([per[s][u]["노출"] for s, u in cells])),
                   "P(-80%)": float(np.mean([per[s][u]["수익"] <= -0.80 for s, u in cells])),
                   "손절": int(sum(per[s][u]["손절"] for s, u in cells)),
                   "주기": _pool_cyc([per[s][u]["주기"] for s, u in cells])}
    return out


def g3_measure(inst: Inst, key: str, x: float, fg_full: pd.Series, tag: str, windows=None, slip=0.0, cash_rate=0.0) -> dict:
    res = {}
    fgp, fgc = inst.fg(tag, fg_full)
    for rn, (d0, d1) in G3_RUNS.items():
        if windows and not ((rn == "2026" and set(windows) & set(G2_WINS_2026)) or rn in windows):
            continue
        d0 = max(d0, str(inst.days[1]))
        mode = 1 if rn in ("RP", "RC") else 0
        pr = path_run(inst, d0, d1, key, x, 1, fgp, fgc, mode, slip, cash_rate)
        if pr is None:
            continue
        if rn == "2026":
            for wn, (a_, b_) in G2_WINS_2026.items():
                if windows and wn not in windows:
                    continue
                res[wn] = slice_stats(inst, pr, a_, b_)
            res["_trace26"] = [(str(inst.days[int(r[0])]), None if r[1] < 0 else str(inst.days[int(r[1])]),
                                float(r[2] / r[3] - 1)) for r in pr["cyc"]]
        else:
            res[rn] = slice_stats(inst, pr, d0, d1, annual=rn != "2025")
            res[rn]["시작"] = d0
    return res


def with_diff(res: dict, resB: dict) -> dict:
    out = {}
    for wn in WIN_ORDER:
        if wn in res and res[wn] and wn in resB and resB[wn]:
            out[wn] = dict(res[wn])
            out[wn]["B대비"] = res[wn]["수익"] - resB[wn]["수익"]
            out[wn]["보유대비"] = res[wn]["수익"] - res[wn]["보유"]
    return out


# ============================================================================ G4 7종목 경로
def g4_state(bn: str, key: str, x: float, fg: pd.Series) -> pd.Series:
    """t일 종가 뒤 상태(1 보유 / 0 현금). at_open 전 값."""
    b = HN.base(bn)
    sh = b.close_s.to_numpy(float)
    days = np.asarray(b.days).astype("U10")
    fgc, _ = fg_map(days, fg)
    sk, x_, nar, h1, h2, rk, p1, p2, mw = rule_args(key, x)
    n = len(sh)
    st = np.ones(n)
    held, peak, exit_i, fgmin, seen = True, sh[0], -1, 1e9, False
    for i in range(n):
        f = fgc[i]
        if held:
            peak = max(peak, sh[i])
            if sk != -1:
                active = not (h1 > 0 and f == f and f <= h1)
                xe = nar if (h2 > 0 and f == f and f >= h2) else x_
                if active and sh[i] / peak - 1 <= -xe:
                    held, exit_i, fgmin, seen = False, i + 1, 1e9, False
        elif i >= exit_i:
            if f == f:
                fgmin = min(fgmin, f)
                if p2 > 0 and f >= p2:
                    seen = True
            d = i + 1 - exit_i
            ok = f == f
            if rk == 9:
                go = False
            elif d >= mw or rk == 0:
                go = True
            elif rk == 1:
                go = d >= p1
            elif rk == 6:
                go = d >= p1 and seen
            elif rk == 2:
                go = ok and f >= fgmin + p1
            elif rk == 3:
                go = ok and f >= p1
            elif rk == 4:
                go = ok and f <= p1
            else:
                go = d >= p1 or (ok and f >= p2)
            if go:
                held, peak = True, sh[i]
        st[i] = 1.0 if held else 0.0
    return pd.Series(st, index=b.days)


G4_WINS = {"앞": ("2026", "앞"), "뒤": ("2026", "뒤"), "AI": ("2026", "AI쇼크"), "2026": ("2026", "전체"),
           "2025": ("hold", "2025")}


def g4_measure(key: str, x: float, fg: pd.Series, slip=0.0, cash_rate=0.0) -> dict:
    res = {}
    for bn in ("2026", "hold"):
        b = HN.base(bn)
        e = HN.at_open(g4_state(bn, key, x, fg), b, fill=1.0)
        er = e.reindex(b.ret.index)
        r = HN.overlay(b, e, cost=slip) + (1 - er) * cash_rate / 252
        ov = pd.Series(b.open, index=b.days)
        sw_all = (er.shift(1).fillna(1.0) >= 0.999) & (er < 0.999)          # 그날 시가에 판 날
        for wn, (bb, pn) in G4_WINS.items():
            if bb != bn:
                continue
            lo, hi = b.periods[pn]
            m = (r.index >= lo) & (r.index <= hi)
            x_r, br = r[m].to_numpy(), b.ret[m].to_numpy()
            v = np.concatenate([[1.0], np.cumprod(1 + x_r)])
            ret, md = float(v[-1] - 1), mdd(v)
            fx, fw = frontier(br, ret, md)
            ex = er[m]
            sw = sw_all[m]
            gains, k_out = [], None
            for d_, val in ex.items():
                if val < 0.5 and k_out is None:
                    k_out = d_
                elif val > 0.5 and k_out is not None:
                    gains.append(ov[k_out] / ov[d_] - 1)
                    k_out = None
            if k_out is not None:
                gains.append(ov[k_out] / ov[ov.index <= hi].iloc[-1] - 1)
            res[wn] = {"수익": ret, "누적": ret, "낙폭": md, "효율선초과": fx, "효율선_w": fw, "노출": float(ex.mean()),
                       "같은노출초과": ret - float(np.prod(1 + ex.mean() * br) - 1), "보유": float(np.prod(1 + br) - 1),
                       "손절": int(sw.sum()), "주기": cyc_stats(np.array(gains))}
        if bn == "2026":
            res["_trace26"] = HN.trace(e, "2026-06-01", "2026-10-09")
    return res


# ============================================================================ 실행
def margins(R: dict, wn: str, key: str | None) -> float | None:
    if key is None:
        return None
    v = R.get(key, {}).get(wn)
    return None if not v or v.get("B대비") is None else float(v["B대비"])


def run_setting(name, grp, tgt, x, kind, ctx, fgtag="real", keys=None, windows=None, slip=0.0, cash_rate=0.0,
                fg_override: pd.Series | None = None) -> dict:
    """설정 하나의 {조합: {창: stats(B대비 포함)}}."""
    keys = list(keys or ALL_KEYS)
    fga = fg_override if fg_override is not None else ctx["fg"][fgtag]
    if grp == "G1":
        G = ctx["G1"]
        fgp, fgc = ctx["G1fg"] if (fgtag == "real" and fg_override is None) else G.fg(fga)
        R = g1_setting(G, tgt, x, (fgp, fgc), keys, windows, slip, cash_rate)
        if windows is None or "AI" in windows:
            sh = g1_shock(G, x, keys, fgp, fgc, slip, cash_rate)
            for k in keys:
                R[k].update(sh[k])
        return R
    if grp == "G2":
        insts = ctx["G2"]
        tag = fgtag if fg_override is None else f"ovr{id(fg_override)}"
        per = {k: g2_measure(insts, k, x, tag, fga, windows, slip, cash_rate) for k in set(keys) | {B_KEY}}
        if tag.startswith("ovr"):
            for inst in insts.values():
                inst.cache.pop(tag, None)
        return {k: g2_agg(per[k], per[B_KEY]) for k in keys}
    if grp == "G3":
        inst = ctx["G3"][tgt]
        full = ctx["fg_full"][fgtag] if fg_override is None else fg_override
        tag = fgtag if fg_override is None else f"ovr{id(fg_override)}"
        resB = g3_measure(inst, B_KEY, x, full, tag, windows, slip, cash_rate)
        out = {}
        for k in keys:
            r = resB if k == B_KEY else g3_measure(inst, k, x, full, tag, windows, slip, cash_rate)
            out[k] = with_diff(r, resB)
            out[k]["_trace26"] = r.get("_trace26")
        if tag.startswith("ovr"):
            inst.cache.pop(tag, None)
        return out
    if grp == "G4":
        resB = g4_measure(B_KEY, x, fga, slip, cash_rate)
        out = {}
        for k in keys:
            r = resB if k == B_KEY else g4_measure(k, x, fga, slip, cash_rate)
            out[k] = with_diff(r, resB)
            out[k]["_trace26"] = r.get("_trace26")
        return out
    raise ValueError(grp)


POOL = [ckey(r, h) for r, h in COMBOS if ckey(r, h) != B_KEY]


def pick(R: dict, wn: str) -> tuple[str | None, float | None]:
    vals = [(margins(R, wn, k), k) for k in POOL]
    vals = [(v, k) for v, k in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None, None
    v, k = max(vals, key=lambda t: t[0])
    return k, v


def setting_selection(R: dict) -> dict:
    out = {}
    k, v = pick(R, "앞")
    out["2026"] = {"pick": k, "앞": v, "뒤": margins(R, "뒤", k), "2025": margins(R, "2025", k), "AI": margins(R, "AI", k),
                   "통과": bool(v is not None and v > 1e-12 and (margins(R, "뒤", k) or -1) > 1e-12)}
    cnt = {}
    for a_, b_ in (("앞", "뒤"), ("앞", "2025"), ("LP", "LC"), ("RP", "RC")):
        A = [margins(R, a_, k_) for k_ in POOL]
        Bv = [margins(R, b_, k_) for k_ in POOL]
        ok = [(x_, y_) for x_, y_ in zip(A, Bv) if x_ is not None and y_ is not None]
        if not ok:
            continue
        fa = np.array([x_ > 1e-12 for x_, _ in ok])
        fb = np.array([y_ > 1e-12 for _, y_ in ok])
        cnt[f"{a_}→{b_}"] = {"조합": len(ok), "고르기승": int(fa.sum()), "확인승": int(fb.sum()), "둘다": int((fa & fb).sum()),
                            "P(확인|고르기)": float((fa & fb).sum() / fa.sum()) if fa.sum() else float("nan"),
                            "P(확인)": float(fb.mean())}
    out["counts"] = cnt
    for a_, b_, lab in (("LP", "LC", "long"), ("RP", "RC", "recon")):
        k2, v2 = pick(R, a_)
        if k2 is not None:
            out[lab] = {"pick": k2, a_: v2, b_: margins(R, b_, k2),
                        "통과": bool(v2 > 1e-12 and (margins(R, b_, k2) or -1) > 1e-12)}
    return out


def _js(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _clean(v):
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return None if not math.isfinite(v) else round(v, 6)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _roundtrip(obj):
    return json.loads(json.dumps(_clean(obj), ensure_ascii=False, default=_js))


def main():
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    fga = fg_actual()
    rc = build_recon()
    rchk = recon_check(rc)
    print("재구성 점검", json.dumps(_clean(rchk), ensure_ascii=False), flush=True)
    recon = rc["recon"].dropna()
    pre = recon[recon.index < "2011-01-03"]
    fg_lag = fg_variant(fga, "lag1")
    ctx = {"fg": {"real": fga, "lag1": fg_lag},
           "fg_full": {"real": pd.concat([pre, fga]), "lag1": pd.concat([pre.shift(1).dropna(), fg_lag])}}
    G = G1Data()
    ctx["G1"] = G
    ctx["G1fg"] = G.fg(fga)
    insts2, excl2, extra2 = build_g2()
    ctx["G2"] = insts2
    ctx["G3"] = build_g3()
    print(f"G2 기초 {len(insts2)}(제외 {excl2}) · G3 {list(ctx['G3'])} 준비 {time.time() - t0:.0f}s", flush=True)
    spec = {n: (g, t, x, k) for n, g, t, x, k in SETTINGS}

    gpath = Path(TMP, f"{ID}_grid.json") if TMP else None
    raw = json.loads(gpath.read_text(encoding="utf-8")) if gpath and gpath.exists() else {}
    for (name, grp, tgt, x, kind) in SETTINGS:
        if name in raw:
            continue
        t1 = time.time()
        raw[name] = _roundtrip(run_setting(name, grp, tgt, x, kind, ctx))
        print(f"  {name}: {len(raw[name])}조합 {time.time() - t1:.0f}s", flush=True)
        if gpath:
            gpath.parent.mkdir(parents=True, exist_ok=True)
            gpath.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    sel = {name: setting_selection(raw[name]) for (name, *_r) in SETTINGS}

    # ---- 대표 후보: 2026 통과 고르기 중 앞 우위 최대(없으면 뒤 손해가 가장 작은 고르기)
    passed = [(sel[n]["2026"]["앞"], n) for n, *_ in SETTINGS if sel[n]["2026"]["통과"]]
    if passed:
        rep_setting, rep_status = max(passed)[1], "통과"
    else:
        cands = [(sel[n]["2026"]["뒤"], n) for n, *_ in SETTINGS if sel[n]["2026"]["뒤"] is not None]
        rep_setting, rep_status = max(cands)[1], "통과 없음(뒤 손해가 가장 작은 고르기)"
    rep_key = sel[rep_setting]["2026"]["pick"]
    print("대표", rep_setting, rep_key, rep_status, flush=True)

    xpath = Path(TMP, f"{ID}_extra.json") if TMP else None
    ext = json.loads(xpath.read_text(encoding="utf-8")) if xpath and xpath.exists() else {"sens": {}, "plac": {}, "grp": {}}
    sens, plac, grp_plac = ext["sens"], ext["plac"], ext["grp"]
    shifts = placebo_shifts(len(fga))
    for n, *_ in SETTINGS:
        if n in sens:
            continue
        g, t, x, k = spec[n]
        keys = sorted({kk for kk in (sel[n]["2026"]["pick"], sel[n].get("long", {}).get("pick"),
                                     sel[n].get("recon", {}).get("pick"), B_KEY) if kk})
        wins = ["앞", "뒤", "AI", "2026", "2025", "LP", "LC"] + (["RP", "RC"] if g == "G3" else [])
        s_ = {"하루늦춤": run_setting(n, g, t, x, k, ctx, "lag1", keys),
              "미끄러짐0.5%": run_setting(n, g, t, x, k, ctx, "real", keys, slip=SLIP),
              "이자4%": run_setting(n, g, t, x, k, ctx, "real", keys, cash_rate=CASH_RATE)}
        sens[n] = _roundtrip({lab: {kk: {wn: margins(rr, wn, kk) for wn in wins} for kk in keys} for lab, rr in s_.items()})
        pk = [kk for kk in keys if kk not in NO_FG]
        pw = ["앞", "뒤", "2025", "LC"]
        pl = {kk: {wn: [] for wn in pw} for kk in pk}
        if pk:
            for sh in shifts:
                fs = fg_variant(fga, "shift", sh)
                ov = fs if g != "G3" else pd.concat([pre, fs])
                rr = run_setting(n, g, t, x, k, ctx, "real", pk + [B_KEY], windows=pw, fg_override=ov)
                for kk in pk:
                    for wn in pw:
                        pl[kk][wn].append(margins(rr, wn, kk))
        plac[n] = {}
        for kk in pk:
            plac[n][kk] = {}
            for wn in pw:
                real = margins(raw[n], wn, kk)
                arr = [v for v in pl[kk][wn] if v is not None]
                if real is None or not arr:
                    continue
                plac[n][kk][wn] = {"실제": real, "위약중앙": float(np.median(arr)), "위약최대": float(np.max(arr)),
                                   "순위": int(1 + sum(v > real + 1e-12 for v in arr)), "판수": len(arr) + 1}
        allk = [kk for kk in POOL if kk not in NO_FG]
        real_b = float(np.mean([(margins(raw[n], "뒤", kk) or -1) > 1e-12 for kk in allk]))
        real_f = float(np.mean([(margins(raw[n], "앞", kk) or -1) > 1e-12 for kk in allk]))
        fr_b, fr_f = [], []
        for sh in shifts[::2]:                                   # 10판(사전 고정)
            fs = fg_variant(fga, "shift", sh)
            ov = fs if g != "G3" else pd.concat([pre, fs])
            rr = run_setting(n, g, t, x, k, ctx, "real", allk + [B_KEY], windows=["앞", "뒤"], fg_override=ov)
            fr_b.append(float(np.mean([(margins(rr, "뒤", kk) or -1) > 1e-12 for kk in allk])))
            fr_f.append(float(np.mean([(margins(rr, "앞", kk) or -1) > 1e-12 for kk in allk])))
        grp_plac[n] = {"뒤_실제": real_b, "뒤_위약평균": float(np.mean(fr_b)), "뒤_위약최대": float(np.max(fr_b)),
                       "앞_실제": real_f, "앞_위약평균": float(np.mean(fr_f)), "판수": len(fr_b)}
        plac[n] = _roundtrip(plac[n])
        print(f"  민감도·위약 {n} {time.time() - t0:.0f}s", flush=True)
        if xpath:
            xpath.write_text(json.dumps({"sens": sens, "plac": plac, "grp": grp_plac}, ensure_ascii=False), encoding="utf-8")

    # ---- 다른 갈래의 재구성(p8_RECON R252, 읽기만)으로 G3 재구성 구간을 다시 잰다(재구성 방식 민감도)
    alt_p = CACHE / "p8_RECON_index.parquet"
    if alt_p.exists():
        alt = pd.read_parquet(alt_p)["R252"].dropna()
        alt.index = pd.Index(alt.index.astype(str).str[:10])
        alt_full = pd.concat([alt[alt.index < "2011-01-03"], fga])
        for n, g, t, x, k in SETTINGS:
            if g != "G3" or "다른재구성" in sens[n]:
                continue
            keys = sorted({kk for kk in (sel[n]["2026"]["pick"], sel[n].get("long", {}).get("pick"),
                                         sel[n].get("recon", {}).get("pick"), B_KEY) if kk})
            rr = run_setting(n, g, t, x, k, ctx, "real", keys, windows=["RP", "RC"], fg_override=alt_full)
            sens[n]["다른재구성"] = _roundtrip({kk: {wn: margins(rr, wn, kk) for wn in ("RP", "RC")} for kk in keys})
        if xpath:
            xpath.write_text(json.dumps({"sens": sens, "plac": plac, "grp": grp_plac}, ensure_ascii=False), encoding="utf-8")

    payload = build_payload(raw, sel, sens, plac, grp_plac, rep_setting, rep_key, rep_status, rchk, excl2, extra2, ctx, G)
    (RESULTS / f"{ID}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=_js), encoding="utf-8")
    md_path = RESULTS / f"{ID}.md"
    head = ""
    if md_path.exists():                                       # 사람이 쓴 결론·한계(자동 표 앞부분)는 남긴다
        head = md_path.read_text(encoding="utf-8").split("<!-- 자동 생성 표")[0]
    md_path.write_text(head + write_md(payload), encoding="utf-8")
    print("저장", RESULTS / f"{ID}.json", md_path, f"{time.time() - t0:.0f}s")


def build_payload(raw, sel, sens, plac, grp_plac, rep_setting, rep_key, rep_status, rchk, excl2, extra2, ctx, G) -> dict:
    keep = {}
    for n, R in raw.items():
        keep[n] = {}
        tr_keys = {B_KEY, "보유만", "손절만", "R0 바로|H0", sel[n]["2026"]["pick"], sel[n].get("long", {}).get("pick"),
                   sel[n].get("recon", {}).get("pick")}
        for k, v in R.items():
            keep[n][k] = {wn: v[wn] for wn in WIN_ORDER + ["AI~10/9"] if wn in v and v[wn]}
            if k in tr_keys and v.get("_trace26") is not None:
                keep[n][k]["_trace26"] = v["_trace26"]
    return _clean({
        "id": ID, "date": "2026-10-11",
        "grid": {"재진입": [r[0] for r in RE_RULES], "보류": [h[0] for h in HOLD_RULES], "조합/설정": len(COMBOS),
                 "설정": [s[0] for s in SETTINGS], "총조합": len(COMBOS) * len(SETTINGS), "최대대기": MAXWAIT, "기준": B_KEY},
        "data": {"G1_종목": len(G.syms), "G1_제외": G.excluded, "G2_기초": sorted(ctx["G2"]), "G2_제외": excl2,
                 "G2_추가비용": extra2, "G3_실제시작": {k: v.actual_start for k, v in ctx["G3"].items()},
                 "재구성점검": rchk},
        "selection": sel, "rep": {"setting": rep_setting, "key": rep_key, "status": rep_status},
        "sensitivity": sens, "placebo": plac, "group_placebo": grp_plac, "results": keep})


# ============================================================================ 보고서(표 자동 생성)
def pct(v, d=1, sign=True):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%"


def pp(v, d=1):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return f"{v * 100:+.{d}f}"


def short(k: str | None) -> str:
    if not k:
        return "–"
    return k.replace("|H0", "").replace("|", " + ")


def write_md(P: dict) -> str:
    R, sel = P["results"], P["selection"]
    names = P["grid"]["설정"]
    L = ["<!-- 자동 생성 표: py p8_FGREENTRY.py (이 줄 위는 사람이 쓴 결론, 아래는 다시 돌리면 바뀜) -->", ""]
    L.append("## 사용자 표 (값, 괄호 = B 'R1 20일' 대비 %p)")
    L.append("")
    L.append("G1 = 거래당 평균 수익, G2 = 기초(상품-해) 중앙값, G3·G4 = 경로 수익(긴 역사는 연율). 사건 = 2026 뒤 손절 수(G1은 거래당).")
    L.append("")
    L.append("꼬리 = G1 거래당 −30% 이하 확률(2019~2025 진입), G2 한 해 −80% 이하 상품-해 비율(2019~2025), G3 2019~2025 최대 낙폭, G4 2025 최대 낙폭.")
    L.append("")
    L.append("| 설정 | 규칙 | 2026 앞 | 2026 뒤 | 7월 AI 쇼크 | 2025 | 긴 역사 고르기 | 긴 역사 확인 | 꼬리 | 사건(뒤) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        lk = sel[n].get("long", {}).get("pick")
        rows = list(dict.fromkeys(k for k in ["보유만", "손절만", B_KEY, "R0 바로|H0", sel[n]["2026"]["pick"], lk] if k))
        for k in rows:
            v = R[n].get(k, {})
            cells = []
            for wn in ("앞", "뒤", "AI", "2025", "LP", "LC"):
                s_ = v.get(wn)
                if not s_:
                    cells.append("–")
                    continue
                val = s_.get("중앙", s_.get("평균", s_.get("수익")))      # G2 중앙값, G1 거래당 평균, G3·G4 경로 수익
                cells.append(pct(val) + ("" if k == B_KEY else f" ({pp(s_.get('B대비'))})"))
            ev = v.get("뒤", {})
            evs = ev.get("손절/거래", ev.get("손절"))
            lab = short(k)
            if k == sel[n]["2026"]["pick"]:
                lab += " ★"
            if k == lk:
                lab += " ☆"
            evs_t = "–" if evs is None else (f"{evs:.2f}" if isinstance(evs, float) else str(evs))
            lc = v.get("LC") or {}
            tail = lc.get("P(-30%)", lc.get("P(-80%)"))
            if tail is not None:
                tail_t = pct(tail, 1, False)
            else:
                md_ = (lc or v.get("2025") or {}).get("낙폭")
                tail_t = pct(md_, 0) if md_ is not None else "–"
            L.append(f"| {n} | {lab} | " + " | ".join(cells) + f" | {tail_t} | {evs_t} |")
    L.append("")
    L.append("★ = 2026 앞 고르기, ☆ = 긴 역사 고르기.")
    L.append("")
    L.append("## 고르기 표 (B 대비 %p)")
    L.append("")
    L.append("| 설정 | 2026 앞 고르기 | 앞 | 뒤 | 2025 | 판정 | 긴 역사 고르기 | 고르기 | 확인 | 판정 | 재구성 고르기 | 1991~2010 | 2011~2025 | 판정 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        s = sel[n]
        a = s["2026"]
        lo = s.get("long", {})
        rc = s.get("recon", {})
        L.append(f"| {n} | {short(a['pick'])} | {pp(a['앞'])} | {pp(a['뒤'])} | {pp(a['2025'])} | {'통과' if a['통과'] else '탈락'} | "
                 f"{short(lo.get('pick'))} | {pp(lo.get('LP'))} | {pp(lo.get('LC'))} | {('통과' if lo.get('통과') else '탈락') if lo else '–'} | "
                 f"{short(rc.get('pick'))} | {pp(rc.get('RP'))} | {pp(rc.get('RC'))} | {('통과' if rc.get('통과') else '탈락') if rc else '–'} |")
    L.append("")
    L.append("## 조합 격자 표 (B 대비 %p, 2026 앞/뒤)")
    L.append("")
    hdr = [n.replace("G3 ", "").replace("7종목 ", "").replace("2배 추적 ", "2배 ") for n in names]
    L.append("| 조합 | " + " | ".join(hdr) + " |")
    L.append("|---|" + "---|" * len(names))
    for r in P["grid"]["재진입"]:
        k = f"{r}|H0"
        L.append(f"| {r} | " + " | ".join(
            f"{pp(R[n].get(k, {}).get('앞', {}).get('B대비'))}/{pp(R[n].get(k, {}).get('뒤', {}).get('B대비'))}" for n in names) + " |")
    for h in P["grid"]["보류"][1:]:
        k = f"R1 20일|{h}"
        L.append(f"| R1 20일 + {h} | " + " | ".join(
            f"{pp(R[n].get(k, {}).get('앞', {}).get('B대비'))}/{pp(R[n].get(k, {}).get('뒤', {}).get('B대비'))}" for n in names) + " |")
    L.append("")
    L.append("127조합(B 제외) 가운데 B를 이긴 수와 위약(F&G 원형 이동 10판) 비교")
    L.append("")
    L.append("| 설정 | 앞 승 | 뒤 승 | 둘 다 | P(뒤 승 / 앞 승) | P(뒤 승) | 긴 역사 고르기 승 → 둘 다 | F&G 조합 뒤 승 비율: 실제 / 위약 평균 / 위약 최대 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for n in names:
        c = sel[n]["counts"].get("앞→뒤", {})
        cl = sel[n]["counts"].get("LP→LC", {})
        gp = P["group_placebo"].get(n, {})
        L.append(f"| {n} | {c.get('고르기승')} | {c.get('확인승')} | {c.get('둘다')} | {pct(c.get('P(확인|고르기)'), 0, False)} | "
                 f"{pct(c.get('P(확인)'), 0, False)} | {cl.get('고르기승', '–')} → {cl.get('둘다', '–')} | "
                 f"{pct(gp.get('뒤_실제'), 0, False)} / {pct(gp.get('뒤_위약평균'), 0, False)} / {pct(gp.get('뒤_위약최대'), 0, False)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
