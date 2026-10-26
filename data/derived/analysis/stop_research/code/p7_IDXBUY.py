"""7단계 갈래 IDXBUY(2026-10-11): 지수가 폭락한 뒤 N거래일 지나 기계적으로 사면 어떻게 되나 (사건형 통계 + 전략형).

사용자: "마삼룰이라고 있거든? 이거랑 컨셉이 비슷한데, 폭락 이후에는 어느 정도 기간 뒤에 기계적으로 매수하는 방법을
연구해볼래요. 다양하게 조합해서." (마삼룰: 나스닥 종합 −3% → 전량 매도, 마지막 −3% 날부터 한 달 + 1일 뒤 재매수.)

자료: cache/p7_idx.parquet(^IXIC 1971~, ^NDX 1985~, ^GSPC 1971~, ^SOX 1994~, ^VIX 1990~, ^IRX 1971~, EWY 2000~),
      SOXX는 lev5.bars('SOXX')(p5_lev_long, 배당 수정). 지수는 가격 지수(배당 없음), EWY·SOXX는 배당 수정.

사건 정의(그날 종가로 판단, 결과를 보기 전에 고정)
  E1_X  하루 하락 ≤ −X%, X ∈ {2,3,4,5}. 그런 날마다 사건.
  E2_Y  252거래일 최고 종가 대비 낙폭이 −Y%를 처음 넘은 날, Y ∈ {10,15,20,25,30,40}.
        한 번 나면 낙폭이 −Y×0.5보다 얕아질 때까지 다시 세지 않는다(같은 하락을 한 번만 셈). 252일 자료가 쌓인 뒤부터.
  E3_K  VIX 종가 ≥ K 처음 닿은 날, K ∈ {30,40,50}(1990~). 다시 세려면 VIX가 0.8K 아래로 내려가야 한다.
  E4_k  21거래일 안 −3% 이하 날 수가 k회 이상이 된 날(k−1 → k로 넘을 때), k ∈ {2,3,4}(마삼 '공황'은 k=4).
대기 N ∈ {0,1,3,5,10,21,42,63}거래일 + 달력 {1개월, 2개월}(마삼 '한 달 + 1일'의 두 해석).
  결정일 d = 사건일 s + N거래일(달력이면 s 날짜 + k개월 이후 첫 거래일). 체결 b = d + 1거래일. N=0은 '다음 날 바로 매수'.
조건
  C0 무조건.  C1 대기 중 같은 사건이 또 나면 그 사건부터 다시 셈(마삼식, 앞 사건의 매수는 취소).
  C2 결정일부터 종가 ≤ 사건일 종가인 첫날(최대 63거래일 더 기다림, 없으면 안 삼).
  C3 결정일부터 종가 ≥ 사건 뒤 최저 종가 × 1.05인 첫날(최대 63거래일 더, 없으면 안 삼).
  C4 N·2N·3N(달력이면 k·2k·3k개월)에 1/3씩. 각 1/3은 자기 체결일부터 H만큼 보유.
  N=0에서 C1·C2·C4는 C0과 같아 따로 세지 않는다 → 사건 16 × (조건 5 × 대기 10 − 3) = 752조합(지수마다).
보유 H ∈ {21,63,126,252}거래일 + REC(사건일 직전 252일 최고 종가 회복: 종가 ≥ 그 값인 날 다음 날 팜, 최대 504거래일).
  REC에서 회복 뒤 체결될 C4 조각은 취소(그 몫은 현금).

체결가(사전 고정)
  지수(^IXIC·^NDX·^GSPC·^SOX와 그 이음 상품)는 시가가 없거나(^IXIC 1971~85 시가=종가) 전날 종가와 같은 해가 길어
  (^NDX ~1999, ^GSPC ~2014, ^SOX ~2004) 모든 해를 '다음 날 종가 체결'로 통일했다. EWY와 그 3배는 '다음 날 시가'.
  매도도 같은 규칙(H일째 체결가). 수수료 0, 미끄러짐 0.5%는 민감도, 현금 이자 0(연 4% 판은 한 줄).

사건형 통계(지수 1배: ^IXIC·^NDX·^GSPC·^SOX·EWY, 신호와 수익 모두 그 지수 자신, E3만 VIX)
  사건마다 1건(겹침 허용). 평균 수익, 플러스 비율, 추가 −20% 확률(보유 중 종가가 평균 단가의 0.8 이하),
  '아무 날 매수' 대비 초과 = 같은 창·같은 보유 기간에 아무 날 산 평균 수익과의 차(REC는 실제 보유 일수에 맞춘 아무 날 평균).
  '바로 매수' 대비 = 같은 사건 C0·N=0과의 차. 겹치지 않는 사건 수(유효 표본)와 대략 t = 초과 ÷ (표준편차/√유효 표본).
  창은 사건일로 나눈다. 고르기 창(시작~2005)은 매도까지 2005-12-30 안에 끝난 거래만 쓴다(뒤 구간 가격 누출 없음).
  아무 날 기준은 그 사건 종류를 잴 수 있는 날부터(E2 252일 뒤, E3 VIX 시작) 같은 창 안.
전략형(평소 현금 → 사건 뒤 매수 → H 보유 → 현금, 보유 중 새 사건은 무시)
  나스닥: 신호 ^IXIC(마삼 기준 지수), 상품 = ^IXIC(~1985-09) + ^NDX(1985-10~) 1배와 3배 합성(TQQQ 대리, 추가 비용 1.3%).
  반도체: 신호 ^SOX, 상품 = ^SOX(~2001-07-13) + SOXX(2001-07-16~) 1배와 3배(SOXL 대리, 추가 1.5%).
  한국: 신호 EWY, 상품 EWY 1배와 3배(KORU 대리, 추가 2.6%). 3배 = 일일 3배 − 2×^IRX − 보수 0.95% − 추가 비용.
  긴 실행 한 번(신호 252일 준비 뒤 시작)을 창으로 자른다. 효율선초과 = 규칙 − 같은 최대 낙폭보다 깊지 않은 '덜 담기'
  (상품 w배 매일 맞춤, w 0~1, 0.01 간격) 최고 수익(긴 창은 연율 %p). 같은노출 = 같은 평균 노출 w의 덜 담기 대비.
고르기(사전 고정)
  긴 역사 전방: 사건형 대표 = ^IXIC 고르기 창(1971~2005)에서 H=126 '아무 날' 대비 초과 최대, 사건 30건 이상 → 2006~2025 확인.
    전략형 = 갈래(나스닥·반도체·한국 × 1·3배)마다 고르기 창 효율선초과(연) 최대, 거래 10번 이상 → 2006~2025 확인.
  2026: 전략형 갈래마다 2026 앞(1/2~5/29) 효율선초과 최대, 앞에 거래 1번 이상 → 2026 뒤(6/1~10/9)·2025 확인.
출력: results/p7_IDXBUY.md, results/p7_IDXBUY.json. 실행: PYTHONIOENCODING=utf-8 py p7_IDXBUY.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lev5

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ID = "p7_IDXBUY"

# ============================================================================ 사전 고정 상수(확인 구간을 본 뒤 바꾸지 않는다)
E1_X = (2, 3, 4, 5)
E2_Y = (10, 15, 20, 25, 30, 40)
E3_K = (30, 40, 50)
E4_K = (2, 3, 4)
N_LIST = (0, 1, 3, 5, 10, 21, 42, 63, "1M", "2M")
CONDS = ("C0", "C1", "C2", "C3", "C4")
H_LIST = (21, 63, 126, 252, "REC")
WAIT_MAX = 63
BOUNCE = 0.05
REC_MAX = 504
DD_WIN = 252
E2_REARM = 0.5
E3_REARM = 0.8
E4_WIN, E4_DROP = 21, -0.03
EXTRA_DD = 0.20
MIN_EVENTS = 30
MIN_TRADES = 10
SLIP, CASH_ALT = 0.005, 0.04
FEE = 0.0095
W_GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
SEL_END, CHK0, CHK1, END = "2005-12-31", "2006-01-01", "2025-12-31", "2026-10-09"
NQ_CUT, SOX_CUT = "1985-10-01", "2001-07-16"

INDICES = {"^IXIC": "close", "^NDX": "close", "^GSPC": "close", "^SOX": "close", "EWY": "open"}
LANES = {  # 이름: (신호 지수, 상품 설명, 추가 비용)
    "나스닥": ("^IXIC", "^IXIC→^NDX(1985-10)", 0.013),
    "반도체": ("^SOX", "^SOX→SOXX(2001-07)", 0.015),
    "한국": ("EWY", "EWY", 0.026),
}
ERAS = {  # 사건일 기준(맥락, 고르기에 쓰지 않음)
    "1971~86": ("1971-01-01", "1986-12-31"), "1987": ("1987-01-01", "1987-12-31"),
    "1988~99": ("1988-01-01", "1999-12-31"), "2000~02": ("2000-01-01", "2002-12-31"),
    "2008": ("2007-10-01", "2009-03-31"), "2010년대": ("2010-01-01", "2019-12-31"),
    "2020": ("2020-01-01", "2020-12-31"), "2022": ("2022-01-01", "2022-12-31"),
    "2025": ("2025-01-01", "2025-12-31"), "2026": ("2026-01-01", END),
}
EV_WIN = {"sel": ("1900-01-01", SEL_END), "chk": (CHK0, CHK1), "all": ("1900-01-01", END), **ERAS}
ST_WIN = {  # 전략형 창(기간 수익). sel·chk·long은 연율
    "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", END), "2025": ("2025-01-02", "2025-12-31"),
    "2022": ("2022-01-03", "2022-12-30"), "2020": ("2020-01-02", "2020-12-31"), "2008": ("2007-10-01", "2009-03-31"),
    "2000~02": ("2000-03-01", "2002-10-31"), "1987": ("1987-01-02", "1987-12-31"),
    "sel": (None, SEL_END), "chk": (CHK0, CHK1), "long": (None, END),
}
LONGW = ("sel", "chk", "long")
EV_LABEL = {**{f"E1_{x}": f"하루 −{x}% 이하" for x in E1_X}, **{f"E2_{y}": f"1년 고점 −{y}% 첫 돌파" for y in E2_Y},
            **{f"E3_{k}": f"VIX ≥ {k}" for k in E3_K}, **{f"E4_{k}": f"21일 안 −3% 날 {k}회" for k in E4_K}}
C_LABEL = {"C0": "무조건", "C1": "또 나면 다시 셈", "C2": "사건일 종가 이하일 때", "C3": "저점 +5% 반등 뒤", "C4": "1/3씩 N·2N·3N"}
N_LABEL = {**{n: f"{n}일" for n in (0, 1, 3, 5, 10, 21, 42, 63)}, "1M": "1개월", "2M": "2개월"}
H_LABEL = {21: "21일", 63: "63일", 126: "126일", 252: "252일", "REC": "고점 회복"}

RAW = pd.read_parquet(CACHE / "p7_idx.parquet")


# ============================================================================ 자료
def raw_series(sym):
    g = RAW[RAW.sym == sym].sort_values("date").drop_duplicates("date").dropna(subset=["Close"])
    g = g[g.Close > 0]
    f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy(float)
    c = g.Close.to_numpy(float)
    o = g.Open.to_numpy(float)
    o = np.where(np.isfinite(o) & (o > 0), o, c)
    return g.date.to_numpy().astype(str), o * f, c * f


class RMQ:
    """구간 최소(희소표)."""

    def __init__(self, a):
        self.t = [np.asarray(a, float)]
        k = 1
        while (1 << k) <= len(a):
            p, h = self.t[-1], 1 << (k - 1)
            self.t.append(np.minimum(p[:-h], p[h:]))
            k += 1

    def q(self, lo, hi):
        lo, hi = np.asarray(lo, int), np.asarray(hi, int)
        out = np.empty(len(lo))
        if len(lo) == 0:
            return out
        k = np.floor(np.log2(hi - lo + 1)).astype(int)
        for kk in np.unique(k):
            m = k == kk
            tab = self.t[kk]
            out[m] = np.minimum(tab[lo[m]], tab[hi[m] - (1 << kk) + 1])
        return out


class Ser:
    def __init__(self, name, days, o, c, mode):
        self.name, self.days, self.o, self.c, self.mode = name, np.asarray(days).astype(str), np.asarray(o, float), np.asarray(c, float), mode
        self.n = len(self.c)
        self.pb = self.c if mode == "close" else self.o
        self.r = np.r_[np.nan, self.c[1:] / self.c[:-1] - 1]
        cs = pd.Series(self.c)
        self.dd = (cs / cs.rolling(DD_WIN, min_periods=DD_WIN).max() - 1).to_numpy()
        self.pk = cs.rolling(DD_WIN, min_periods=1).max().to_numpy()
        self.rmq = RMQ(self.c)
        self._mon = {}
        self._base = {}

    def idx(self, day, side="left"):
        return int(np.searchsorted(self.days, day, side=side))

    def mon(self, k):
        if k not in self._mon:
            dt = pd.to_datetime(pd.Series(self.days))
            tgt = (dt + pd.DateOffset(months=k)).dt.strftime("%Y-%m-%d").to_numpy().astype(str)
            self._mon[k] = np.searchsorted(self.days, tgt, side="left")
        return self._mon[k]

    # ---- 아무 날 매수 기준(결정일 d ∈ [a, z], 체결 b = d+1, 매도 b+H ≤ endcap)
    def base_fixed(self, a, z, endcap, H):
        key = ("F", a, z, endcap, H)
        if key not in self._base:
            b = np.arange(a + 1, min(z + 1, endcap - H) + 1)
            if len(b) == 0:
                self._base[key] = (np.nan, np.nan, np.nan, 0)
            else:
                ret = self.pb[b + H] / self.pb[b] - 1
                lo = self.rmq.q(b, b + H)
                self._base[key] = (float(ret.mean()), float((ret > 0).mean()), float((lo <= (1 - EXTRA_DD) * self.pb[b]).mean()), len(b))
        return self._base[key]

    def base_L(self, a, z, endcap):
        key = ("L", a, z, endcap)
        if key not in self._base:
            out = np.full(REC_MAX + 2, np.nan)
            for L in range(1, REC_MAX + 2):
                b = np.arange(a + 1, min(z + 1, endcap - L) + 1)
                if len(b):
                    out[L] = float((self.pb[b + L] / self.pb[b]).mean() - 1)
            self._base[key] = out
        return self._base[key]


def make_index(sym):
    d, o, c = raw_series(sym)
    return Ser(sym, d, o, c, INDICES[sym])


def splice_close(days, first_c, second_days, second_c, cut):
    """days 격자 위에서 cut 전은 first 수익률, cut 다음 날부터 second 수익률로 이은 가격(첫날 100)."""
    s2 = pd.Series(second_c, index=second_days).reindex(days).ffill().to_numpy()
    r1 = np.r_[0.0, first_c[1:] / first_c[:-1] - 1]
    r2 = np.r_[0.0, s2[1:] / s2[:-1] - 1]
    ic = int(np.searchsorted(days, cut))
    r = np.where(np.arange(len(days)) <= ic, r1, np.nan_to_num(r2))
    return 100 * np.cumprod(1 + r)


def rf_on(days):
    d, _, c = raw_series("^IRX")
    s = pd.Series(c / 100.0, index=d)
    s = s[~s.index.duplicated()]
    return s.reindex(sorted(set(s.index) | set(days))).ffill().reindex(days).fillna(0.0).to_numpy()


def synth3(P: Ser, extra: float) -> Ser:
    """lev5.synth와 같은 식(일일 3배 − 2×^IRX − 보수 − 추가 비용), ^IRX는 p7(1971~)."""
    rf = rf_on(P.days)
    n, L = P.n, 3.0
    c = np.empty(n); o = np.empty(n)
    c[0] = o[0] = 100.0
    for t in range(1, n):
        pc = P.c[t - 1]
        r = L * (P.c[t] / pc - 1) - (L - 1) * rf[t] / 252 - (FEE + extra) / 252
        c[t] = c[t - 1] * max(1 + r, 0.001)
        o[t] = c[t - 1] * max(1 + L * (P.o[t] / pc - 1), 0.001)
    return Ser(P.name + "x3", P.days, o, c, P.mode)


def lane_products(name, IDX):
    sig_sym, _, extra = LANES[name]
    S = IDX[sig_sym]
    if name == "나스닥":
        nd = IDX["^NDX"]
        c = splice_close(S.days, S.c, nd.days, nd.c, NQ_CUT)
        P1 = Ser("NQ1", S.days, c, c, "close")
    elif name == "반도체":
        sx = lev5.bars("SOXX")
        c = splice_close(S.days, S.c, sx.days.astype(str), sx.c, SOX_CUT)
        P1 = Ser("SOX1", S.days, c, c, "close")
    else:
        P1 = Ser("EWY1", S.days, S.o, S.c, "open")
    return S, P1, synth3(P1, extra)


# ============================================================================ 사건
def events(S: Ser, vix: np.ndarray) -> dict:
    out = {}
    r = S.r
    for x in E1_X:
        out[f"E1_{x}"] = (np.flatnonzero(r <= -x / 100), 1)
    first_dd = int(np.flatnonzero(np.isfinite(S.dd))[0])
    for y in E2_Y:
        ev, armed, th = [], False, -y / 100
        for t in range(first_dd, S.n):
            x = S.dd[t]
            if armed and x <= th:
                ev.append(t); armed = False
            elif not armed and x > th * E2_REARM:
                armed = True
        out[f"E2_{y}"] = (np.array(ev, int), first_dd)
    okv = np.flatnonzero(np.isfinite(vix))
    first_v = int(okv[0]) if len(okv) else S.n
    for k in E3_K:
        ev, armed = [], False
        for t in okv:
            v = vix[t]
            if armed and v >= k:
                ev.append(t); armed = False
            elif not armed and v < k * E3_REARM:
                armed = True
        out[f"E3_{k}"] = (np.array(ev, int), first_v)
    hit = pd.Series(np.nan_to_num(r, nan=0.0) <= E4_DROP).astype(float)
    cnt = hit.rolling(E4_WIN, min_periods=E4_WIN).sum().to_numpy()
    prev = np.r_[np.nan, cnt[:-1]]
    for k in E4_K:
        out[f"E4_{k}"] = (np.flatnonzero((cnt >= k) & np.isfinite(prev) & (prev < k)), E4_WIN)
    return out


def dec_day(S, s, N, mult=1):
    if isinstance(N, str):
        return S.mon(int(N[0]) * mult)[s]
    return s + N * mult


def build(S: Ser, ev: np.ndarray, cond: str, N):
    """사건 → (사건일 Sv, 체결일 B (m,3)). B = −1: 아직 체결 전(자료 끝 뒤)."""
    n, c = S.n, S.c
    if len(ev) == 0:
        return np.zeros(0, int), np.zeros((0, 3), int)
    if cond == "C4":
        B = np.stack([dec_day(S, ev, N, k) + 1 for k in (1, 2, 3)], 1)
        Sv = ev
    else:
        d0 = dec_day(S, ev, N)
        if cond == "C0":
            d, keep = d0, np.ones(len(ev), bool)
        elif cond == "C1":
            nxt = np.r_[ev[1:], np.iinfo(np.int64).max]
            d, keep = d0, nxt > d0
        else:
            d = np.empty(len(ev), dtype=np.int64)
            for i, (s, a) in enumerate(zip(ev, d0)):
                if a >= n:
                    d[i] = n + 10
                    continue
                z = min(a + WAIT_MAX, n - 1)
                seg = c[a:z + 1]
                if cond == "C2":
                    ok = seg <= c[s]
                else:
                    mn = np.minimum.accumulate(c[s:z + 1])[a - s:]
                    ok = seg >= (1 + BOUNCE) * mn
                h = np.flatnonzero(ok)
                if len(h):
                    d[i] = a + h[0]
                elif a + WAIT_MAX <= n - 1:
                    d[i] = -10
                else:
                    d[i] = n + 10
            keep = d != -10
        B = np.repeat((np.asarray(d) + 1)[:, None], 3, 1)
        Sv, B = ev[keep], B[keep]
    B = np.where(B < n, B, -1)
    return Sv, B


def sells(S: Ser, Sv, B, H):
    """매도 체결일 Q. −1 = 아직 안 팔림(진행 중), −3 = 취소된 조각(REC에서 회복 뒤 체결 예정이던 1/3)."""
    n = S.n
    if H != "REC":
        Q = np.where(B >= 0, B + H, -1)
        return np.where(Q < n, Q, -1)
    Q = np.full(B.shape, -1, dtype=np.int64)
    for i in range(len(Sv)):
        b1 = B[i, 0]
        if b1 < 0:
            continue
        act = B[i][B[i] >= 0]
        last = int(act.max()) + REC_MAX
        seg = S.c[b1:min(last, n - 1) + 1]
        h = np.flatnonzero(seg >= S.pk[Sv[i]])
        e = b1 + int(h[0]) if len(h) else None
        for k in range(3):
            b = B[i, k]
            if b < 0:
                Q[i, k] = -3 if (e is not None and e + 1 < n) else -1
                continue
            if e is not None and b >= e + 1:
                Q[i, k] = -3
                continue
            cap = b + REC_MAX
            q = cap if e is None else min(e + 1, cap)
            Q[i, k] = q if q < n else -1
    return Q


# ============================================================================ 사건형 통계
def ev_stats(S: Ser, Sv, B, Q, H, a, z, endcap, base_avail):
    """사건일 ∈ [a, z], 마지막 매도 ≤ endcap인 끝난 거래만."""
    act = (B >= 0) & (Q >= 0)
    canc = Q == -3
    done = np.all(act | canc, 1) & act.any(1)
    lastq = np.where(act, Q, -1).max(1)
    m = done & (Sv >= a) & (Sv <= z) & (lastq <= endcap)
    pend = int(((Sv >= a) & (Sv <= z) & ~done).sum())
    a_b = max(a, base_avail)
    if H == "REC":
        bL = S.base_L(a_b, z, endcap)
        b_ret = np.nan
    else:
        b_ret, b_pos, b_hit, b_n = S.base_fixed(a_b, z, endcap, H)
    k = int(m.sum())
    if k == 0:
        return {"n": 0, "n_ind": 0, "pend": pend, "base": b_ret}
    Bm, Qm, am = B[m], Q[m], act[m]
    pb = S.pb
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = np.where(am, pb[np.where(am, Qm, 0)] / pb[np.where(am, Bm, 0)], np.nan)
    ret = np.nanmean(rr, 1) - 1
    inv = np.where(am, 1 / pb[np.where(am, Bm, 0)], 0).sum(1)
    avgc = am.sum(1) / inv
    first = np.where(am, Bm, 1 << 40).min(1)
    last = np.where(am, Qm, -1).max(1)
    lo = S.rmq.q(first, last)
    hit = lo <= (1 - EXTRA_DD) * avgc
    if H == "REC":
        Ls = np.where(am, Qm - Bm, 0)
        bb = np.where(am, bL[np.clip(Ls, 0, REC_MAX + 1)], np.nan)
        btr = np.nanmean(bb, 1)
        ex = ret - btr
        hold = np.nanmean(np.where(am, Ls, np.nan), 1)
    else:
        ex = ret - b_ret
        hold = np.full(k, float(H))
    # 겹치지 않는 사건 수
    o = np.argsort(first, kind="stable")
    n_ind, endp = 0, -1
    for i in o:
        if first[i] > endp:
            n_ind += 1
            endp = last[i]
    exm = float(np.nanmean(ex))
    sd = float(np.nanstd(ex, ddof=1)) if k > 1 else np.nan
    t = exm / (sd / np.sqrt(n_ind)) if (k > 1 and sd > 0) else np.nan
    return {"n": k, "n_ind": n_ind, "pend": pend, "mean": float(ret.mean()), "med": float(np.median(ret)),
            "pos": float((ret > 0).mean()), "hit20": float(hit.mean()), "ex": exm, "t": float(t),
            "hold": float(np.nanmean(hold)), "base": float(np.nanmean(ret - ex)) if H == "REC" else b_ret,
            "base_pos": None if H == "REC" else b_pos, "base_hit": None if H == "REC" else b_hit}


def ev_windows(S: Ser):
    out = {}
    sel_cap = S.idx(SEL_END, "right") - 1
    for w, (w0, w1) in EV_WIN.items():
        a, z = S.idx(w0), S.idx(w1, "right") - 1
        if z < a or z < 1:
            continue
        out[w] = (a, z, sel_cap if w == "sel" else S.n - 1)
    return out


# ============================================================================ 전략형
def strat_trades(B, Q, n, rs, c4):
    """겹치지 않게 앞에서부터 받는다. 진행 중(Q=−1)은 자료 끝까지 보유."""
    cols = 3 if c4 else 1
    first = np.where(B[:, :cols] >= 0, B[:, :cols], 1 << 40).min(1)
    out, endp = [], -1
    for i in np.argsort(first, kind="stable"):
        if first[i] >= (1 << 40) or first[i] < rs or first[i] <= endp:
            continue
        bs, qs = B[i, :cols], Q[i, :cols]
        ok = (bs >= 0) & (qs != -3)
        if not ok.any():
            continue
        bs, qs = bs[ok], np.where(qs[ok] >= 0, qs[ok], n - 1)
        out.append((bs, qs, cols))
        endp = int(qs.max())
    return out


def equity(P: Ser, trades, rs, slip=0.0, cash=0.0):
    n, c, pb = P.n, P.c, P.pb
    E = np.ones(n); X = np.zeros(n)
    g = 1 + cash / 252
    val, cur = 1.0, rs
    closem = P.mode == "close"
    f1s = []
    for bs, qs, cols in trades:
        f1, end = int(bs.min()), int(qs.max())
        if f1 > cur:
            E[cur:f1] = val * g ** np.arange(1, f1 - cur + 1) if cash else val
            val = E[f1 - 1]
        T = np.arange(f1, end + 1)
        gT = g ** (T - f1 + 1) if cash else 1.0
        A = val / cols
        tot = (val - A * len(bs)) * (gT if cash else np.ones(len(T)))
        inv = np.zeros(len(T))
        for b, q in zip(bs, qs):
            gb = (g ** (b - f1 + (1 if closem else 0))) if cash else 1.0
            u = A * gb * (1 - slip) / pb[b]
            v = np.empty(len(T))
            pre = T < b
            v[pre] = A * (gT[pre] if cash else 1.0)
            hold = (T >= b) & (T < q)
            v[hold] = u * c[T[hold]]
            exitv = u * (c[q] if closem else pb[q]) * (1 - slip)
            post = T >= q
            v[post] = exitv * (g ** (T[post] - q) if cash else 1.0)
            inv[hold] += v[hold]
            tot = tot + v
        E[f1:end + 1] = tot
        X[f1:end + 1] = inv / tot
        val, cur = float(E[end]), end + 1
        f1s.append(f1)
    if cur < n:
        E[cur:] = val * g ** np.arange(1, n - cur + 1) if cash else val
    E[:rs] = 1.0
    return E, X, np.array(f1s, int)


def st_windows(P: Ser, rs):
    out = {}
    for w, (w0, w1) in ST_WIN.items():
        a = rs if w0 is None else P.idx(w0)
        z = P.idx(w1, "right") - 1
        if a < rs or z <= a:
            continue
        out[w] = (a, z)
    return out


def frontier(P: Ser, wins, cash=0.0):
    r = np.r_[0.0, P.c[1:] / P.c[:-1] - 1]
    out = {}
    for w, (a, z) in wins.items():
        rr = r[a:z + 1]
        lg = np.log1p(W_GRID[:, None] * rr[None, :] + (1 - W_GRID[:, None]) * cash / 252)
        cum = np.concatenate([np.zeros((len(W_GRID), 1)), np.cumsum(lg, 1)], 1)
        v = np.exp(cum)
        ret = v[:, -1] - 1
        mdd = (v / np.maximum.accumulate(v, 1) - 1).min(1)
        yrs = (z - a + 1) / 252
        cagr = (1 + ret) ** (1 / yrs) - 1
        out[w] = (ret, mdd, cagr)
    return out


def st_stats(E, X, f1s, wins, fr):
    res = {}
    for w, (a, z) in wins.items():
        seg = E[a - 1:z + 1]
        ret = seg[-1] / seg[0] - 1
        mdd = float((seg / np.maximum.accumulate(seg) - 1).min())
        yrs = (z - a + 1) / 252
        cagr = (1 + ret) ** (1 / yrs) - 1
        fret, fmdd, fcagr = fr[w]
        met, fm = (cagr, fcagr) if w in LONGW else (ret, fret)
        ok = fmdd >= mdd - 1e-12
        eff = met - fm[ok].max()
        expo = float(X[a:z + 1].mean())
        same = met - np.interp(expo, W_GRID, fm)
        ntr = int(((f1s >= a) & (f1s <= z)).sum())
        res[w] = {"ret": float(ret), "mdd": mdd, "cagr": float(cagr), "eff": float(eff), "same": float(same),
                  "ntr": ntr, "tpy": ntr / yrs, "expo": expo}
    return res


# ============================================================================ 실행
def combos():
    for eid in EV_LABEL:
        for cond in CONDS:
            for N in N_LIST:
                if N == 0 and cond in ("C1", "C2", "C4"):
                    continue
                yield eid, cond, N


def main():
    t0 = time.time()
    IDX = {s: make_index(s) for s in INDICES}
    vd, _, vc = raw_series("^VIX")
    vix_s = pd.Series(vc, index=vd)
    EVS = {s: events(S, vix_s.reindex(S.days).to_numpy()) for s, S in IDX.items()}
    LP = {ln: lane_products(ln, IDX) for ln in LANES}
    lane_of = {LANES[ln][0]: ln for ln in LANES}
    combo_list = list(combos())
    print(f"조합 {len(combo_list)}개/지수, 보유 {len(H_LIST)}, 지수 {len(IDX)}, 전략 갈래 {len(LANES)}×2배", flush=True)

    ev_rows, st_rows, keep_tr = [], [], {}
    LW, FR, RS = {}, {}, {}
    for ln, (S, P1, P3) in LP.items():
        rs = DD_WIN
        RS[ln] = rs
        for lev, P in ((1, P1), (3, P3)):
            LW[(ln, lev)] = st_windows(P, rs)
            FR[(ln, lev)] = frontier(P, LW[(ln, lev)])

    for sym, S in IDX.items():
        wins = ev_windows(S)
        ln = lane_of.get(sym)
        t1 = time.time()
        for eid, cond, N in combo_list:
            ev, avail = EVS[sym][eid]
            Sv, B = build(S, ev, cond, N)
            for H in H_LIST:
                Q = sells(S, Sv, B, H)
                for w, (a, z, cap) in wins.items():
                    st = ev_stats(S, Sv, B, Q, H, a, z, cap, avail)
                    ev_rows.append({"idx": sym, "eid": eid, "cond": cond, "N": N, "H": H, "win": w, **st})
                if ln is not None:
                    _, P1, P3 = LP[ln]
                    trs = strat_trades(B, Q, S.n, RS[ln], cond == "C4")
                    for lev, P in ((1, P1), (3, P3)):
                        E, X, f1s = equity(P, trs, RS[ln])
                        res = st_stats(E, X, f1s, LW[(ln, lev)], FR[(ln, lev)])
                        for w, r in res.items():
                            st_rows.append({"lane": ln, "lev": lev, "eid": eid, "cond": cond, "N": N, "H": H, "win": w, **r})
                    keep_tr[(ln, eid, cond, N, H)] = trs
        print(f"  {sym} 끝 {time.time() - t1:.0f}s", flush=True)
    EVD = pd.DataFrame(ev_rows)
    STD = pd.DataFrame(st_rows)
    EVD["Nk"] = EVD.N.astype(str)
    STD["Nk"] = STD.N.astype(str)
    EVD["Hk"] = EVD.H.astype(str)
    STD["Hk"] = STD.H.astype(str)
    print(f"계산 끝 {time.time() - t0:.0f}s, 사건형 칸 {len(EVD)}, 전략형 칸 {len(STD)}", flush=True)
    report(IDX, LP, EVS, EVD, STD, keep_tr, LW, FR, RS, len(combo_list))
    print(f"전체 {time.time() - t0:.0f}s")


# ============================================================================ 보고
def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def pp(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%p"


def pr(x):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.0f}%"


def key(eid, cond, N, H=None):
    s = f"{EV_LABEL[eid]} · {C_LABEL[cond]} · {N_LABEL[N]}"
    return s + (f" · {H_LABEL[H]} 보유" if H is not None else "")


def report(IDX, LP, EVS, EVD, STD, keep_tr, LW, FR, RS, ncombo):
    def ev(sym, eid, cond, N, H, w):
        q = EVD[(EVD.idx == sym) & (EVD.eid == eid) & (EVD.cond == cond) & (EVD.Nk == str(N)) & (EVD.Hk == str(H)) & (EVD.win == w)]
        return q.iloc[0].to_dict() if len(q) else None

    def st(ln, lev, eid, cond, N, H, w):
        q = STD[(STD.lane == ln) & (STD.lev == lev) & (STD.eid == eid) & (STD.cond == cond) & (STD.Nk == str(N)) & (STD.Hk == str(H)) & (STD.win == w)]
        return q.iloc[0].to_dict() if len(q) else None

    J = {"id": ID, "grid": {"사건": len(EV_LABEL), "조건": len(CONDS), "대기": len(N_LIST), "보유": len(H_LIST),
                            "조합_지수당": ncombo, "지수": list(INDICES), "사건형_칸": int(len(EVD)),
                            "전략형_규칙": int(len(STD[STD.win == "long"])), "전략형_칸": int(len(STD))}}

    # ---------------- 사건형 대표(긴 역사 전방, ^IXIC, H=126)
    def pick_ev(sym, H=126):
        q = EVD[(EVD.idx == sym) & (EVD.Hk == str(H)) & (EVD.win == "sel") & (EVD.n >= MIN_EVENTS)]
        q = q[np.isfinite(q.ex)]
        if not len(q):
            return None
        return q.sort_values("ex", ascending=False).iloc[0].to_dict()

    rep = pick_ev("^IXIC")
    RK = (rep["eid"], rep["cond"], rep["N"])
    picks_ev = {s: pick_ev(s) for s in INDICES}
    # 고르기 정보: sel>0 → chk>0 비율
    info = {}
    for sym in INDICES:
        a = EVD[(EVD.idx == sym) & (EVD.Hk == "126") & (EVD.win == "sel") & (EVD.n >= MIN_EVENTS)].set_index(["eid", "cond", "Nk"])
        b = EVD[(EVD.idx == sym) & (EVD.Hk == "126") & (EVD.win == "chk") & (EVD.n >= 5)].set_index(["eid", "cond", "Nk"])
        jn = a[["ex"]].join(b[["ex"]], rsuffix="_c", how="inner").dropna()
        if len(jn) < 5:
            info[sym] = None
            continue
        pos = jn[jn.ex > 0]
        info[sym] = {"조합": int(len(jn)), "고르기>0": int(len(pos)),
                     "그중_확인>0": float((pos.ex_c > 0).mean()) if len(pos) else np.nan,
                     "무조건_확인>0": float((jn.ex_c > 0).mean()),
                     "순위상관": float(jn.ex.rank().corr(jn.ex_c.rank())),
                     "상위10_확인평균": float(jn.sort_values("ex", ascending=False).head(10).ex_c.mean()),
                     "전체_확인평균": float(jn.ex_c.mean())}
    J["사건형_대표"] = {"정의": key(*RK, 126), "고르기": rep}
    J["사건형_지수별_고르기"] = {s: (None if p is None else {"정의": key(p["eid"], p["cond"], p["N"], 126), "sel": p,
                                                       "chk": ev(s, p["eid"], p["cond"], p["N"], 126, "chk")}) for s, p in picks_ev.items()}
    J["고르기_정보"] = info

    # ---------------- 전략형 고르기
    def pick_st(ln, lev, win, metric, min_tr):
        q = STD[(STD.lane == ln) & (STD.lev == lev) & (STD.win == win) & (STD.ntr >= min_tr)]
        if not len(q):
            return None, 0
        return q.sort_values(metric, ascending=False).iloc[0].to_dict(), int(len(q))

    st_picks = {}
    for ln in LANES:
        for lev in (1, 3):
            pL, nL = pick_st(ln, lev, "sel", "eff", MIN_TRADES)
            p26, n26 = pick_st(ln, lev, "2026앞", "eff", 1)
            st_picks[(ln, lev)] = {"long": pL, "n_long": nL, "y26": p26, "n_26": n26}
    # 전략형 고르기 정보
    st_info = {}
    for ln in LANES:
        for lev in (1, 3):
            a = STD[(STD.lane == ln) & (STD.lev == lev) & (STD.win == "sel") & (STD.ntr >= MIN_TRADES)].set_index(["eid", "cond", "Nk", "Hk"])
            b = STD[(STD.lane == ln) & (STD.lev == lev) & (STD.win == "chk")].set_index(["eid", "cond", "Nk", "Hk"])
            jn = a[["eff"]].join(b[["eff"]], rsuffix="_c", how="inner").dropna()
            a2 = STD[(STD.lane == ln) & (STD.lev == lev) & (STD.win == "2026앞") & (STD.ntr >= 1)].set_index(["eid", "cond", "Nk", "Hk"])
            b2 = STD[(STD.lane == ln) & (STD.lev == lev) & (STD.win == "2026뒤")].set_index(["eid", "cond", "Nk", "Hk"])
            j2 = a2[["eff"]].join(b2[["eff"]], rsuffix="_c", how="inner").dropna()
            st_info[f"{ln}{lev}배"] = {
                "긴_후보": int(len(jn)), "긴_고르기>0": int((jn.eff > 0).sum()),
                "긴_그중확인>0": float((jn[jn.eff > 0].eff_c > 0).mean()) if (jn.eff > 0).any() else np.nan,
                "긴_무조건확인>0": float((jn.eff_c > 0).mean()) if len(jn) else np.nan,
                "긴_순위상관": float(jn.eff.rank().corr(jn.eff_c.rank())) if len(jn) > 4 else np.nan,
                "26_후보": int(len(j2)), "26_앞>0": int((j2.eff > 0).sum()),
                "26_그중뒤>0": float((j2[j2.eff > 0].eff_c > 0).mean()) if (j2.eff > 0).any() else np.nan,
                "26_무조건뒤>0": float((j2.eff_c > 0).mean()) if len(j2) else np.nan}
    J["전략형_고르기"] = {f"{ln}{lev}배": {k: v for k, v in d.items()} for (ln, lev), d in st_picks.items()}
    J["전략형_고르기_정보"] = st_info

    # ---------------- 민감도(미끄러짐 0.5%, 현금 4%) — 대표·고른 규칙
    def sens(ln, lev, eid, cond, N, H):
        S, P1, P3 = LP[ln]
        P = P1 if lev == 1 else P3
        trs = keep_tr[(ln, eid, cond, N, H)]
        out = {}
        for nm, kw in (("기본", {}), ("미끄러짐0.5%", {"slip": SLIP}), ("현금4%", {"cash": CASH_ALT})):
            E, X, f1s = equity(P, trs, RS[ln], **kw)
            fr = FR[(ln, lev)] if nm != "현금4%" else frontier(P, LW[(ln, lev)], cash=CASH_ALT)
            out[nm] = st_stats(E, X, f1s, LW[(ln, lev)], fr)
        return out

    # 대표를 다음 날 시가로 체결하면(^IXIC 1986~, QQQ 시가) — 사건형 민감도 한 줄
    def open_fill_check():
        S = IDX["^IXIC"]
        So = Ser("^IXIC_open", S.days, S.o, S.c, "open")
        ev_, av = EVS["^IXIC"][RK[0]]
        Sv, B = build(So, ev_, RK[1], RK[2])
        Q = sells(So, Sv, B, 126)
        a, z = So.idx(CHK0), So.idx(CHK1, "right") - 1
        return ev_stats(So, Sv, B, Q, 126, a, z, So.n - 1, av)
    J["대표_시가체결_2006_2025"] = open_fill_check()

    # ============================================================ md
    L = []
    rep_all = ev("^IXIC", *RK, 126, "all")
    rep_sel = ev("^IXIC", *RK, 126, "sel")
    rep_chk = ev("^IXIC", *RK, 126, "chk")
    n0_chk = ev("^IXIC", RK[0], "C0", 0, 126, "chk")
    n0_sel = ev("^IXIC", RK[0], "C0", 0, 126, "sel")

    # ---- 전략형 대표 숫자
    nq_hold1 = None
    L.append("# 7단계 IDXBUY: 지수 폭락 뒤 N일 지나 기계적으로 사기 (2026-10-11)\n")
    L.append("> 사용자: \"마삼룰이랑 컨셉이 비슷한데, 폭락 이후에는 어느 정도 기간 뒤에 기계적으로 매수하는 방법을 연구해볼래요. 다양하게 조합해서.\"")
    L.append(f"> 사건 16개(하루 −2~−5%, 1년 고점 −10~−40% 첫 돌파, VIX 30·40·50, 21일 안 −3% 날 2~4회) × 조건 5 × 대기 10(0~63거래일, 1·2개월) = 지수마다 {ncombo}조합, 보유 5가지(21·63·126·252일, 고점 회복). "
             f"사건형은 ^IXIC·^NDX·^GSPC·^SOX·EWY 1배, 전략형(평소 현금)은 나스닥·반도체·한국 1배·3배 합성. 사건형 {len(EVD):,}칸, 전략형 규칙 {J['grid']['전략형_규칙']:,}개.")
    L.append("> 체결: 신호는 그날 종가, 지수와 그 이음 상품은 다음 날 **종가** 체결(옛 지수 시가가 없거나 전날 종가와 같아 모든 해를 통일), EWY는 다음 날 시가. 수수료 0, 현금 이자 0. 스크립트 `code/p7_IDXBUY.py`, 숫자 `results/p7_IDXBUY.json`.\n")
    J["_md_placeholder"] = True

    # 결론은 아래 표들을 만든 뒤 맨 위에 끼운다
    body = []

    # ---- 사용자 표(전략형)
    body.append("## 사용자 표: 전략형(평소 현금 → 사건 뒤 매수 → 보유 → 현금), 수익/최대 낙폭\n")
    body.append("| 규칙 | 상품 | 2026 앞 | 2026 뒤 | 2025 | 2022 | 2020 | 2007.10~09.03 | 2000.03~02.10 | 1987 | 긴 역사 연율/최대 낙폭/연 매매 |")
    body.append("|---|---|---|---|---|---|---|---|---|---|---|")

    def hold_row(ln, lev, label):
        S, P1, P3 = LP[ln]
        P = P1 if lev == 1 else P3
        E = np.ones(P.n)
        E[RS[ln] - 1:] = P.c[RS[ln] - 1:] / P.c[RS[ln] - 1]
        X = np.zeros(P.n); X[RS[ln]:] = 1
        r = st_stats(E, X, np.array([RS[ln]]), LW[(ln, lev)], FR[(ln, lev)])
        return r

    def const_row(ln, lev, w):
        S, P1, P3 = LP[ln]
        P = P1 if lev == 1 else P3
        rr = np.r_[0.0, P.c[1:] / P.c[:-1] - 1]
        E = np.ones(P.n)
        E[RS[ln]:] = np.cumprod(1 + w * rr[RS[ln]:])
        X = np.zeros(P.n); X[RS[ln]:] = w
        return st_stats(E, X, np.array([RS[ln]]), LW[(ln, lev)], FR[(ln, lev)])

    def cells(r):
        out = []
        for w in ("2026앞", "2026뒤", "2025", "2022", "2020", "2008", "2000~02", "1987"):
            x = r.get(w)
            out.append("—" if x is None else f"{pct(x['ret'])}/{x['mdd'] * 100:.0f}%")
        lg = r.get("long")
        out.append("—" if lg is None else f"연 {pct(lg['cagr'])}/{lg['mdd'] * 100:.0f}%/{lg['tpy']:.1f}번")
        return out

    def strow(ln, lev, eid, cond, N, H):
        return {w: st(ln, lev, eid, cond, N, H, w) for w in ST_WIN}

    prod_name = {("나스닥", 1): "나스닥 1배", ("나스닥", 3): "나스닥 3배(TQQQ 대리)", ("반도체", 1): "반도체 1배",
                 ("반도체", 3): "반도체 3배(SOXL 대리)", ("한국", 1): "EWY 1배", ("한국", 3): "EWY 3배(KORU 대리)"}
    user_rows = []
    for ln, lev in (("나스닥", 1), ("나스닥", 3)):
        user_rows.append(("그냥 보유", prod_name[(ln, lev)], hold_row(ln, lev, "")))
    user_rows.append(("덜 담기 1/3(매일 맞춤, 나머지 현금)", prod_name[("나스닥", 3)], const_row("나스닥", 3, 0.33)))
    masam = RK == ("E1_3", "C1", 21)
    for lev in (1, 3):
        user_rows.append((f"사건형 대표{' (= 마삼식 대기)' if masam else ''}: {key(*RK, 126)}", prod_name[("나스닥", lev)], strow("나스닥", lev, *RK, 126)))
    if not masam:
        user_rows.append(("마삼식 참고: 하루 −3% · 또 나면 다시 셈 · 21일 · 126일 보유", prod_name[("나스닥", 3)], strow("나스닥", 3, "E1_3", "C1", 21, 126)))
    user_rows.append(("비교: 하루 −3% 다음 날 바로 사서 126일", prod_name[("나스닥", 3)], strow("나스닥", 3, "E1_3", "C0", 0, 126)))
    user_rows.append(("마삼 공황 참고: 21일 안 −3% 날 4회 · 또 나면 다시 셈 · 42일 · 126일 보유", prod_name[("나스닥", 3)], strow("나스닥", 3, "E4_4", "C1", 42, 126)))
    for (ln, lev), d in st_picks.items():
        p = d["long"]
        if p is not None and lev == 3:
            user_rows.append((f"긴 역사로 고른 것({ln} 3배): {key(p['eid'], p['cond'], p['N'] if p['Nk'] not in ('1M', '2M') else p['Nk'], p['H'] if p['Hk'] != 'REC' else 'REC')}",
                              prod_name[(ln, lev)], strow(ln, lev, p["eid"], p["cond"], p["N"], p["H"])))
    for lev in (1, 3):
        user_rows.append(("사후 관찰(고르기 밖): 1년 고점 −10% 첫 돌파 · 무조건 · 21일 · 126일 보유", prod_name[("나스닥", lev)], strow("나스닥", lev, "E2_10", "C0", 21, 126)))
    for ln in ("반도체", "한국"):
        user_rows.append(("그냥 보유", prod_name[(ln, 3)], hold_row(ln, 3, "")))
        user_rows.append(("덜 담기 1/3", prod_name[(ln, 3)], const_row(ln, 3, 0.33)))
        user_rows.append((f"사건형 대표 규칙(신호는 {LANES[ln][0]} 자신)", prod_name[(ln, 3)], strow(ln, 3, *RK, 126)))
    for lab, prod, r in user_rows:
        body.append(f"| {lab} | {prod} | " + " | ".join(cells(r)) + " |")
    body.append("")
    body.append("- 긴 역사: 나스닥 1972-02~, 반도체 1995-05~, 한국 2001-05~(신호 252일 준비 뒤). 연 매매 = 한 해 매수 횟수(1/3씩 사기는 한 묶음을 1번).")
    J["사용자표"] = [{"규칙": a, "상품": b, "창": r} for a, b, r in user_rows]

    # ---- 사건형 조합 격자(^IXIC, H=126, C0·C1)
    body.append("\n## 조합 격자: 나스닥 종합, 126일 보유, '아무 날 매수' 대비 초과 — 1971~2005(고르기) / 2006~2025(확인) (사건 수)\n")
    hdr = "| 사건 · 조건 | " + " | ".join(N_LABEL[n] for n in N_LIST) + " |"
    body.append(hdr)
    body.append("|---|" + "---|" * len(N_LIST))
    sub = EVD[(EVD.idx == "^IXIC") & (EVD.Hk == "126")]
    idxd = {(r.eid, r.cond, r.Nk, r.win): r for r in sub.itertuples()}

    def cell2(eid, cond, N):
        Nn = N if cond == "C0" or N != 0 else N
        cc = "C0" if (N == 0 and cond in ("C1", "C2", "C4")) else cond
        a = idxd.get((eid, cc, str(N), "sel"))
        b = idxd.get((eid, cc, str(N), "chk"))
        fa = "—" if a is None or a.n == 0 else f"{a.ex * 100:+.1f}"
        fb = "—" if b is None or b.n == 0 else f"{b.ex * 100:+.1f}"
        na = 0 if a is None else a.n
        nb = 0 if b is None else b.n
        mark = "**" if (eid, cc, str(N)) == (RK[0], RK[1], str(RK[2])) else ""
        return f"{mark}{fa} / {fb}{mark} ({na}·{nb})"

    for eid in EV_LABEL:
        for cond in ("C0", "C1"):
            body.append(f"| {EV_LABEL[eid]} · {C_LABEL[cond]} | " + " | ".join(cell2(eid, cond, n) for n in N_LIST) + " |")
    b126_sel = IDX["^IXIC"].base_fixed(1, IDX["^IXIC"].idx(SEL_END, "right") - 1, IDX["^IXIC"].idx(SEL_END, "right") - 1, 126)
    a_chk, z_chk = IDX["^IXIC"].idx(CHK0), IDX["^IXIC"].idx(CHK1, "right") - 1
    b126_chk = IDX["^IXIC"].base_fixed(a_chk, z_chk, IDX["^IXIC"].n - 1, 126)
    body.append("")
    body.append(f"- 단위 %p. 아무 날 126일 평균: 1971~2005 {pct(b126_sel[0])}(플러스 {pr(b126_sel[1])}, 추가 −20% {pr(b126_sel[2])}), 2006~2025 {pct(b126_chk[0])}(플러스 {pr(b126_chk[1])}, 추가 −20% {pr(b126_chk[2])}). "
                "E2는 1972-02~, E3은 1990~ 같은 기간 아무 날과 비교. 굵게 = 사건형 대표. 0일 열은 조건 없이 같다. 나머지 조건(C2~C4)·보유·지수는 json.")

    # ---- 조건 효과 요약(전체 기간, ^IXIC, H=126)
    body.append("\n## 조건이 바꾼 것: 나스닥 종합 1971~2026, 126일 보유, 무조건(C0) 대비 초과의 변화 (16개 사건 중앙값, 나아진 사건 수)\n")
    body.append("| 조건 | " + " | ".join(N_LABEL[n] for n in N_LIST if n != 0) + " |")
    body.append("|---|" + "---|" * (len(N_LIST) - 1))
    alld = EVD[(EVD.idx == "^IXIC") & (EVD.Hk == "126") & (EVD.win == "all")].set_index(["eid", "cond", "Nk"])
    cond_eff = {}
    for cond in ("C1", "C2", "C3", "C4"):
        cs = []
        for n in N_LIST:
            if n == 0:
                continue
            ds = []
            for eid in EV_LABEL:
                try:
                    x, y = alld.loc[(eid, cond, str(n))], alld.loc[(eid, "C0", str(n))]
                except KeyError:
                    continue
                if x.n >= 5 and y.n >= 5:
                    ds.append(x.ex - y.ex)
            ds = np.array(ds)
            cond_eff[(cond, str(n))] = ds.tolist()
            cs.append("—" if not len(ds) else f"{np.median(ds) * 100:+.1f}%p ({int((ds > 0).sum())}/{len(ds)})")
        body.append(f"| {C_LABEL[cond]} | " + " | ".join(cs) + " |")
    body.append("\n- 사건 수 5건 이상인 사건만 셈. C1은 같은 사건이 몰린 하락장에서 마지막 사건 뒤로 매수를 미룬다(마삼식).")

    # ---- 핵심 행 전체 지표(전체 기간 ^IXIC)
    body.append("\n## 핵심 조합 전체 지표: 나스닥 종합 1971~2026(끝난 거래), 63일 / 126일 보유\n")
    body.append("| 조합 | 사건 수(겹침 뺀 수) | 평균 수익 63·126 | 플러스 63·126 | 추가 −20% 63·126 | 아무 날 대비 63·126 | 바로 매수 대비 126 | 대략 t(126) |")
    body.append("|---|---|---|---|---|---|---|---|")
    key_rows = [RK, ("E1_3", "C0", 0), ("E1_3", "C1", 21), ("E1_3", "C1", "1M"), ("E4_4", "C1", 42), ("E4_4", "C1", "2M"),
                ("E2_20", "C0", 0), ("E2_20", "C0", 63), ("E2_40", "C0", 0), ("E3_40", "C0", 0), ("E3_40", "C0", 21), ("E1_5", "C0", 0)]
    seen = set()
    krows_json = []
    for kr in key_rows:
        if kr in seen:
            continue
        seen.add(kr)
        a63, a126 = ev("^IXIC", *kr, 63, "all"), ev("^IXIC", *kr, 126, "all")
        n0 = ev("^IXIC", kr[0], "C0", 0, 126, "all")
        if a126 is None or a126["n"] == 0:
            continue
        vs0 = a126["mean"] - n0["mean"] if n0 and n0["n"] else np.nan
        body.append(f"| {key(*kr)} | {a126['n']}({a126['n_ind']}) | {pct(a63['mean'])} · {pct(a126['mean'])} | {pr(a63['pos'])} · {pr(a126['pos'])} | "
                    f"{pr(a63['hit20'])} · {pr(a126['hit20'])} | {pp(a63['ex'])} · {pp(a126['ex'])} | {pp(vs0)} | {a126['t']:.1f} |")
        krows_json.append({"조합": key(*kr), "63": a63, "126": a126})
    b63_all = IDX["^IXIC"].base_fixed(1, IDX["^IXIC"].n - 1, IDX["^IXIC"].n - 1, 63)
    b126_all = IDX["^IXIC"].base_fixed(1, IDX["^IXIC"].n - 1, IDX["^IXIC"].n - 1, 126)
    body.append(f"| 아무 날 매수(기준) | {b126_all[3]:,}일 | {pct(b63_all[0])} · {pct(b126_all[0])} | {pr(b63_all[1])} · {pr(b126_all[1])} | {pr(b63_all[2])} · {pr(b126_all[2])} | 0 | — | — |")
    body.append("\n- 겹침 뺀 수 = 보유 기간이 겹치지 않게 앞에서부터 센 사건 수(실제 독립 표본에 가까움). 대략 t는 그 수로 나눈 값이라 2 아래면 우연과 구별이 어렵다.")
    J["핵심행"] = krows_json

    # ---- 대표의 지수별·시대별
    body.append(f"\n## 사건형 대표({key(*RK, 126)}) — 시대별, 지수별\n")
    body.append("| 시대(사건일) | 나스닥 종합: 사건 수 · 평균 · 플러스 · 추가 −20% · 아무 날 대비 · 바로 매수 대비 | 나스닥100 | S&P500 | 반도체(^SOX) | EWY |")
    body.append("|---|---|---|---|---|---|")
    era_json = {}
    for w in ["sel", "chk"] + list(ERAS):
        cells_ = []
        for sym in INDICES:
            r = ev(sym, *RK, 126, w)
            r0 = ev(sym, RK[0], "C0", 0, 126, w)
            era_json.setdefault(w, {})[sym] = r
            if r is None or r["n"] == 0:
                cells_.append("—" if r is None or r.get("pend", 0) == 0 else f"진행 중 {r['pend']}건")
                continue
            vs0 = r["mean"] - r0["mean"] if (r0 and r0["n"]) else np.nan
            if sym == "^IXIC":
                cells_.append(f"{r['n']} · {pct(r['mean'])} · {pr(r['pos'])} · {pr(r['hit20'])} · {pp(r['ex'])} · {pp(vs0)}")
            else:
                cells_.append(f"{r['n']}건 {pct(r['mean'])} ({pp(r['ex'])})")
        lab = {"sel": "고르기 1971~2005", "chk": "확인 2006~2025"}.get(w, w)
        body.append(f"| {lab} | " + " | ".join(cells_) + " |")
    body.append("\n- 다른 지수 칸 = 사건 수, 126일 평균 수익(아무 날 대비). 신호도 그 지수 자신의 것. 시대 칸의 매도는 시대 끝을 넘어가도 된다.")
    J["대표_시대별"] = era_json

    # ---- 대기 보호 표(2000~02·2008, 나스닥 종합, C1, 126일)
    body.append("\n## 폭락이 이어진 시기에 기다림이 지켜 줬나: 나스닥 종합, 또 나면 다시 셈(C1), 126일 보유 — 평균 수익 / 추가 −20% 확률 (사건 수)\n")
    NS = (0, 5, 21, 42, 63, "1M", "2M")
    body.append("| 사건 · 시기 | " + " | ".join(N_LABEL[n] for n in NS) + " |")
    body.append("|---|" + "---|" * len(NS))
    prot = {}
    for eid in ("E1_3", "E2_10", "E4_4"):
        for w in ("2000~02", "2008", "2020", "2022"):
            cs = []
            for n in NS:
                r = ev("^IXIC", eid, "C0" if n == 0 else "C1", n, 126, w)
                prot[f"{eid}|{w}|{n}"] = r
                cs.append("—" if r is None or r["n"] == 0 else f"{pct(r['mean'], 0)} / {pr(r['hit20'])} ({r['n']})")
            body.append(f"| {EV_LABEL[eid]} · {w} | " + " | ".join(cs) + " |")
    body.append("\n- 같은 시기 아무 날 126일: " + ", ".join(
        f"{w} {pct(ev('^IXIC', 'E1_3', 'C0', 0, 126, w)['base'])}" for w in ("2000~02", "2008", "2020", "2022")))
    J["대기보호"] = prot

    # ---- 고르기 표
    body.append("\n## 고르기 표\n")
    body.append("### 긴 역사 전방 (앞 구간으로 고르고 2006~2025로 확인)\n")
    body.append("| 방법 | 고른 조합 | 고르기 구간 | 확인 2006~2025 | 2026 | 유효 표본·대략 t(고르기) |")
    body.append("|---|---|---|---|---|---|")
    for sym, p in picks_ev.items():
        if p is None:
            body.append(f"| 사건형 {sym} 126일 | 사건 30건 이상 조합 없음 | — | — | — | — |")
            continue
        c_ = ev(sym, p["eid"], p["cond"], p["N"] if p["Nk"] not in ("1M", "2M") else p["Nk"], 126, "chk")
        y26 = ev(sym, p["eid"], p["cond"], p["N"] if p["Nk"] not in ("1M", "2M") else p["Nk"], 126, "2026")
        s0 = IDX[sym].days[0][:4]
        body.append(f"| 사건형 {sym} 126일, 아무 날 대비 최대 | {key(p['eid'], p['cond'], p['N'] if p['Nk'] not in ('1M', '2M') else p['Nk'])} | "
                    f"{s0}~2005: {pp(p['ex'])}, {p['n']}건 | {('—' if c_ is None or c_['n'] == 0 else f'{pp(c_['ex'])}, {c_['n']}건, 평균 {pct(c_['mean'])}')} | "
                    f"{('—' if y26 is None or y26['n'] == 0 else f'{y26['n']}건 {pct(y26['mean'])}') + ('' if y26 is None or not y26.get('pend') else f', 진행 중 {y26['pend']}건')} | {p['n_ind']} · {p['t']:.1f} |")
    for (ln, lev), d in st_picks.items():
        p = d["long"]
        if p is None:
            body.append(f"| 전략형 {prod_name[(ln, lev)]} | 거래 10번 이상 규칙 없음 | — | — | — | — |")
            continue
        Nn = p["N"] if p["Nk"] not in ("1M", "2M") else p["Nk"]
        Hh = p["H"] if p["Hk"] != "REC" else "REC"
        c_ = st(ln, lev, p["eid"], p["cond"], Nn, Hh, "chk")
        a26 = st(ln, lev, p["eid"], p["cond"], Nn, Hh, "2026앞")
        b26 = st(ln, lev, p["eid"], p["cond"], Nn, Hh, "2026뒤")
        body.append(f"| 전략형 {prod_name[(ln, lev)]}, 효율선초과(연) 최대 | {key(p['eid'], p['cond'], Nn, Hh)} | 연 {pct(p['cagr'])}/{p['mdd'] * 100:.0f}%, 효율선 {pp(p['eff'])}, {p['ntr']}번 | "
                    f"연 {pct(c_['cagr'])}/{c_['mdd'] * 100:.0f}%, 효율선 {pp(c_['eff'])} | 앞 {pct(a26['ret'])}·뒤 {pct(b26['ret'])} | {d['n_long']}개 중 1등 |")
    body.append("")
    body.append("고르기 정보(사건형, 126일): '고르기 > 0'인 조합이 확인에서도 > 0인 비율 vs 무조건, 순위상관, 고르기 상위 10개의 확인 평균 vs 전체 평균\n")
    body.append("| 지수 | 조합(30건+) | 고르기>0 | 그중 확인>0 | 무조건 확인>0 | 순위상관 | 상위10 확인 평균 | 전체 확인 평균 |")
    body.append("|---|---|---|---|---|---|---|---|")
    for sym, d in info.items():
        if d is None:
            body.append(f"| {sym} | 5개 미만 | — | — | — | — | — | — |")
            continue
        body.append(f"| {sym} | {d['조합']} | {d['고르기>0']} | {pr(d['그중_확인>0'])} | {pr(d['무조건_확인>0'])} | {d['순위상관']:+.2f} | {pp(d['상위10_확인평균'])} | {pp(d['전체_확인평균'])} |")
    body.append("")
    body.append("전략형(효율선초과): 긴 역사 '고르기 > 0' → 확인 > 0 비율, 2026 앞 > 0 → 뒤 > 0 비율\n")
    body.append("| 갈래 | 긴 역사 후보(10번+) | 고르기>0 | 그중 확인>0 | 무조건 | 순위상관 | 2026 앞 거래 있는 규칙 | 앞>0 | 그중 뒤>0 | 무조건 |")
    body.append("|---|---|---|---|---|---|---|---|---|---|")
    for k_, d in st_info.items():
        body.append(f"| {k_} | {d['긴_후보']} | {d['긴_고르기>0']} | {pr(d['긴_그중확인>0'])} | {pr(d['긴_무조건확인>0'])} | {d['긴_순위상관']:+.2f} | {d['26_후보']} | {d['26_앞>0']} | {pr(d['26_그중뒤>0'])} | {pr(d['26_무조건뒤>0'])} |")

    body.append("\n### 2026 판정 (2026 앞 1/2~5/29로 고르고 뒤 6/1~10/9·2025로 확인, 전략형)\n")
    body.append("| 상품 | 앞에 거래가 있던 규칙 | 고른 규칙 | 앞 수익/효율선 | 뒤 수익/효율선 | 2025 수익/효율선 | 그냥 보유 앞·뒤 |")
    body.append("|---|---|---|---|---|---|---|")
    y26_json = {}
    for (ln, lev), d in st_picks.items():
        p = d["y26"]
        hr = hold_row(ln, lev, "")
        hh = f"{pct(hr['2026앞']['ret'])} · {pct(hr['2026뒤']['ret'])}"
        if p is None:
            body.append(f"| {prod_name[(ln, lev)]} | 0 | 앞 구간에 신호 없음 | — | — | — | {hh} |")
            continue
        Nn = p["N"] if p["Nk"] not in ("1M", "2M") else p["Nk"]
        Hh = p["H"] if p["Hk"] != "REC" else "REC"
        b = st(ln, lev, p["eid"], p["cond"], Nn, Hh, "2026뒤")
        c25 = st(ln, lev, p["eid"], p["cond"], Nn, Hh, "2025")
        y26_json[f"{ln}{lev}"] = {"규칙": key(p["eid"], p["cond"], Nn, Hh), "앞": p, "뒤": b, "2025": c25}
        body.append(f"| {prod_name[(ln, lev)]} | {d['n_26']} | {key(p['eid'], p['cond'], Nn, Hh)} | {pct(p['ret'])}/{pp(p['eff'])} | {pct(b['ret'])}/{pp(b['eff'])} | "
                    f"{pct(c25['ret'])}/{pp(c25['eff'])} | {hh} |")
    J["2026판정"] = y26_json

    # ---- 2026 사건 목록(대표 사건 종류, 나스닥 종합·반도체·EWY)
    body.append("\n### 2026년 사건(대표 사건 종류)\n")
    lines26 = []
    for sym in INDICES:
        S = IDX[sym]
        ev_, av = EVS[sym][RK[0]]
        e26 = [S.days[i] for i in ev_ if S.days[i] >= "2026-01-01"]
        lines26.append(f"- {sym}: {len(e26)}번" + (f" ({', '.join(d_[5:] for d_ in e26[:12])}{' …' if len(e26) > 12 else ''})" if e26 else ""))
    ixe = EVS["^IXIC"]
    for eid in EV_LABEL:
        S = IDX["^IXIC"]
        e26 = [S.days[i][5:] for i in ixe[eid][0] if S.days[i] >= "2026-01-01"]
        if e26:
            lines26.append(f"- 나스닥 종합 {EV_LABEL[eid]}: {', '.join(e26)}")
    trades26 = []
    for sym in INDICES:
        S = IDX[sym]
        ev_, av = EVS[sym][RK[0]]
        Sv, B = build(S, ev_, RK[1], RK[2])
        Q = sells(S, Sv, B, 126)
        for s, b, q in zip(Sv, B[:, 0], Q[:, 0]):
            if S.days[s] < "2026-01-01":
                continue
            if b < 0:
                trades26.append(f"{sym} {S.days[s][5:]} 사건 → 아직 매수 전")
            elif q < 0:
                trades26.append(f"{sym} {S.days[s][5:]} 사건 → {S.days[b][5:]} 매수, 10/9까지 {pct(S.pb[-1] / S.pb[b] - 1)}(보유 중)")
            else:
                trades26.append(f"{sym} {S.days[s][5:]} 사건 → {S.days[b][5:]} 매수 → {S.days[q][5:]} 매도 {pct(S.pb[q] / S.pb[b] - 1)}")
    lines26.append(f"- 사건형 대표의 2026 거래: " + ("; ".join(trades26) if trades26 else "없음"))
    body.extend(lines26)
    J["2026사건"] = lines26

    # ---- 민감도
    body.append("\n## 민감도(사건형 대표·전략형으로 고른 것)\n")
    sens_json = {}
    srows = [("나스닥", 1, *RK, 126), ("나스닥", 3, *RK, 126)]
    for (ln, lev), d in st_picks.items():
        p = d["long"]
        if p is not None:
            srows.append((ln, lev, p["eid"], p["cond"], p["N"] if p["Nk"] not in ("1M", "2M") else p["Nk"], p["H"] if p["Hk"] != "REC" else "REC"))
    body.append("| 규칙 | 상품 | 긴 역사 연율/효율선: 기본 · 미끄러짐 0.5% · 현금 4% | 2026 앞·뒤 효율선(기본) |")
    body.append("|---|---|---|---|")
    for ln, lev, eid, cond, N, H in srows:
        s_ = sens(ln, lev, eid, cond, N, H)
        sens_json[f"{ln}{lev}|{key(eid, cond, N, H)}"] = s_
        lg = [s_[k]["long"] for k in ("기본", "미끄러짐0.5%", "현금4%")]
        a_, b_ = s_["기본"].get("2026앞"), s_["기본"].get("2026뒤")
        body.append(f"| {key(eid, cond, N, H)} | {prod_name[(ln, lev)]} | " + " · ".join(f"{pct(x['cagr'])}/{pp(x['eff'])}" for x in lg) +
                    f" | {pp(a_['eff']) if a_ else '—'} · {pp(b_['eff']) if b_ else '—'} |")
    of = J["대표_시가체결_2006_2025"]
    body.append(f"\n- 사건형 대표를 2006~2025에 다음 날 시가(나스닥 종합 시가)로 사고팔면: {of.get('n', 0)}건, 평균 {pct(of.get('mean'))}, 아무 날 대비 {pp(of.get('ex'))} "
                f"(종가 체결 {rep_chk['n']}건, {pct(rep_chk['mean'])}, {pp(rep_chk['ex'])}).")
    body.append("- 현금 4%는 규칙과 '덜 담기' 모두 남는 현금에 연 4%를 붙였다. 미끄러짐은 규칙에만 붙였다(덜 담기는 거의 안 사고팔아서).")
    J["민감도"] = sens_json

    # ---- 부록: 전체 격자(^IXIC, 1971~2026, 126일, 아무 날 대비 초과 %p (사건 수))
    body.append("\n## 부록: 전체 격자 — 나스닥 종합 1971~2026, 126일 보유, 아무 날 대비 초과 %p (사건 수)\n")
    body.append("| 사건 · 조건 | " + " | ".join(N_LABEL[n] for n in N_LIST) + " |")
    body.append("|---|" + "---|" * len(N_LIST))
    for eid in EV_LABEL:
        for cond in CONDS:
            cs = []
            for n in N_LIST:
                cc = "C0" if (n == 0 and cond in ("C1", "C2", "C4")) else cond
                try:
                    x = alld.loc[(eid, cc, str(n))]
                    cs.append("—" if x.n == 0 else f"{x.ex * 100:+.1f} ({x.n})")
                except KeyError:
                    cs.append("—")
            body.append(f"| {EV_LABEL[eid]} · {C_LABEL[cond]} | " + " | ".join(cs) + " |")

    # ---- 결론·한계에 쓸 사실(모두 위 표의 숫자)
    F = {}
    vs0 = {}
    for w in ERAS:
        r, r0 = ev("^IXIC", *RK, 126, w), ev("^IXIC", RK[0], "C0", 0, 126, w)
        if r and r0 and r["n"] and r0["n"]:
            vs0[w] = r["mean"] - r0["mean"]
    F["era_vs0"] = vs0
    e1s, e1c = [], []
    for x in E1_X:
        for n in N_LIST:
            a_, b_ = ev("^IXIC", f"E1_{x}", "C0", n, 126, "sel"), ev("^IXIC", f"E1_{x}", "C0", n, 126, "chk")
            e1s.append(a_["ex"]); e1c.append(b_["ex"])
    F["E1_C0_sel"], F["E1_C0_chk"] = (min(e1s), max(e1s)), (min(e1c), max(e1c))
    e2s, e2c = [], []
    for n in (0, 1, 3, 5, 10, 21, "1M"):
        e2s.append(ev("^IXIC", "E2_10", "C0", n, 126, "sel")["ex"]); e2c.append(ev("^IXIC", "E2_10", "C0", n, 126, "chk")["ex"])
    F["E2_10_sel"], F["E2_10_chk"] = (min(e2s), max(e2s)), (min(e2c), max(e2c))
    F["E2_10_n"] = (ev("^IXIC", "E2_10", "C0", 21, 126, "sel")["n"], ev("^IXIC", "E2_10", "C0", 21, 126, "chk")["n"])
    F["E2_10_st"] = {f"{ln}{lev}": {w: st(ln, lev, "E2_10", "C0", 21, 126, w) for w in ("sel", "chk", "2026앞", "2026뒤", "2025")}
                     for ln in LANES for lev in (1, 3)}
    F["p2000"] = (prot.get("E1_3|2000~02|0"), prot.get("E1_3|2000~02|21"))
    F["p2008"] = (prot.get("E1_3|2008|0"), prot.get("E1_3|2008|42"), prot.get("E1_3|2008|63"))
    F["rep_st"] = {lev: strow("나스닥", lev, *RK, 126) for lev in (1, 3)}
    F["c0_st"] = {lev: strow("나스닥", lev, "E1_3", "C0", 0, 126) for lev in (1, 3)}
    F["picks_chk"] = {f"{ln}{lev}": (st(ln, lev, d["long"]["eid"], d["long"]["cond"], d["long"]["N"], d["long"]["H"], "chk")["eff"]
                                     if d["long"] is not None else None) for (ln, lev), d in st_picks.items()}
    F["hold_nq"] = {lev: hold_row("나스닥", lev, "") for lev in (1, 3)}
    F["rep_lane3"] = {ln: strow(ln, 3, *RK, 126) for ln in ("반도체", "한국")}
    J["사실"] = F

    J["_body"] = body
    J["_head"] = L
    # json 격자(넓은 표). 사건형: 모든 지수 63·126일 + 나스닥 종합 모든 보유, 창 sel·chk·all
    ew = EVD[EVD.win.isin(["sel", "chk", "all"]) & ((EVD.idx == "^IXIC") | EVD.Hk.isin(["63", "126"]))]
    ew = ew.pivot_table(index=["idx", "eid", "cond", "Nk", "Hk"], columns="win", values=["n", "ex", "mean", "hit20"], aggfunc="first")
    ew.columns = [f"{w}_{v}" for v, w in ew.columns]
    ew = ew.reset_index()
    J["사건형_격자"] = {"columns": list(ew.columns), "rows": [[_r(v) for v in row] for row in ew.itertuples(index=False)]}
    sw = STD[STD.win.isin(["sel", "chk", "long", "2026앞", "2026뒤", "2025", "2000~02", "2008"])]
    sw = sw.pivot_table(index=["lane", "lev", "eid", "cond", "Nk", "Hk"], columns="win", values=["eff", "ret", "cagr", "mdd", "ntr"], aggfunc="first")
    keepc = [("cagr", "long"), ("mdd", "long"), ("ntr", "long"), ("eff", "sel"), ("eff", "chk"), ("eff", "long"), ("ntr", "sel"),
             ("ret", "2026앞"), ("eff", "2026앞"), ("ntr", "2026앞"), ("ret", "2026뒤"), ("eff", "2026뒤"), ("eff", "2025"),
             ("ret", "2000~02"), ("ret", "2008")]
    sw = sw[[c for c in keepc if c in sw.columns]]
    sw.columns = [f"{w}_{v}" for v, w in sw.columns]
    sw = sw.reset_index()
    J["전략형_격자"] = {"columns": list(sw.columns), "rows": [[_r(v) for v in row] for row in sw.itertuples(index=False)]}
    J["조건효과"] = {f"{c}|{n}": v for (c, n), v in cond_eff.items()}
    finish(J, rep, rep_sel, rep_chk, n0_chk, st_picks, info, st_info, F)


def _r(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v


def finish(J, rep, rep_sel, rep_chk, n0_chk, st_picks, info, st_info, F):
    """결론 3줄·한계를 숫자로 쓰고 저장."""
    RK = (rep["eid"], rep["cond"], rep["N"] if rep["Nk"] not in ("1M", "2M") else rep["Nk"])
    head, body = J.pop("_head"), J.pop("_body")
    J.pop("_md_placeholder", None)
    masam = RK == ("E1_3", "C1", 21)
    vs0 = F["era_vs0"]
    neg = [w for w, v in vs0.items() if v < 0]
    p0, p21 = F["p2000"]
    q0, q42, q63 = F["p2008"]
    r1, r3 = F["rep_st"][1], F["rep_st"][3]
    c3 = F["c0_st"][3]
    pk = {k: v for k, v in F["picks_chk"].items() if v is not None}
    corr = {s: d["순위상관"] for s, d in info.items() if d}
    e2 = F["E2_10_st"]
    s26 = {k: (d["26_그중뒤>0"], d["26_무조건뒤>0"]) for k, d in st_info.items()}
    concl = ["## 결론 3줄\n",
             f"1. **미리 정한 방법으로 1971~2005에서 고른 대표{'는 마삼의 대기 규칙과 같은 모양이었다' if masam else ''}**: {key(*RK, 126)}. "
             f"126일 수익이 '아무 날 매수'보다 고르기 구간에서 {pp(rep_sel['ex'])}({rep_sel['n']}건, 겹치지 않는 표본 {rep_sel['n_ind']}건), 확인 구간 2006~2025에서 {pp(rep_chk['ex'])}"
             f"({rep_chk['n']}건, 평균 {pct(rep_chk['mean'])}, 대략 t {rep_chk['t']:.1f})로 줄었다. 확인 구간에서는 −3% 다음 날 바로 사기가 {pp(n0_chk['ex'])}로 더 나았다. "
             f"기다림이 1971~2005에 이긴 이유는 같은 하락 안에서 더 싸게 산 것이 아니었다. 2000~02처럼 −3% 날이 몰린 때 덜 샀기 때문이다"
             f"(2000~02 매수 {p0['n']}번 → {p21['n']}번). 시대별로 보면 기다린 쪽이 바로 매수보다 {len(neg)}/{len(vs0)}개 시대에서 낮았다"
             f"({min(vs0.values()) * 100:+.1f}~{max(vs0.values()) * 100:+.1f}%p).",
             f"2. **폭락이 이어진 시기에 기다림이 지켜 주지 못했고, 조합 순위는 시대마다 뒤집혔다.** 2000~02에는 −3% 뒤 21일을 기다려도 126일 평균 {pct(p21['mean'], 0)}, "
             f"추가 −20% 확률이 {pr(p21['hit20'])}였다(바로 매수 {pct(p0['mean'], 0)}·{pr(p0['hit20'])}). 2008에는 42·63일 기다린 매수가 리먼 전(2008년 4~5월, 8월 말)에 들어가 "
             f"{pct(q42['mean'], 0)}·{pct(q63['mean'], 0)}가 됐다. 하루 급락일로 센 사건(무조건)은 1971~2005에 아무 날보다 {F['E1_C0_sel'][0] * 100:+.0f}~{F['E1_C0_sel'][1] * 100:+.0f}%p, "
             f"2006~2025에 {F['E1_C0_chk'][0] * 100:+.0f}~{F['E1_C0_chk'][1] * 100:+.0f}%p였다. 고르기·확인 순위상관은 나스닥 종합 {corr.get('^IXIC', np.nan):+.2f}, "
             f"나스닥100 {corr.get('^NDX', np.nan):+.2f}, 반도체 {corr.get('^SOX', np.nan):+.2f}, EWY {corr.get('EWY', np.nan):+.2f}였다(S&P500만 {corr.get('^GSPC', np.nan):+.2f}).",
             f"3. **전략으로(평소 현금, 사건 뒤 사서 126일) 돌리면 '덜 담기'를 넘지 못했다.** 대표의 나스닥 1배는 긴 역사 연 {pct(r1['long']['cagr'])}/최대 낙폭 {r1['long']['mdd'] * 100:.0f}%였다. "
             f"같은 낙폭의 덜 담기보다 연 {pp(r1['long']['eff'])}였고, 3배는 {pp(r3['long']['eff'])}, 2000~02에 3배 {pct(r3['2000~02']['ret'])}였다. "
             f"같은 규칙을 반도체 3배·EWY 3배에 쓰면 덜 담기보다 연 {pp(F['rep_lane3']['반도체']['long']['eff'])}·{pp(F['rep_lane3']['한국']['long']['eff'])}였다. "
             f"긴 역사로 고른 갈래별 1등 6개는 확인 구간에서 {sum(v < 0 for v in pk.values())}/{len(pk)}개가 덜 담기보다 낮았다({min(pk.values()) * 100:+.1f}~{max(pk.values()) * 100:+.1f}%p/연). "
             f"2026은 나스닥 종합 −3% 날이 6/5 한 번뿐이다(대표는 7/9 매수, 10/9까지 보유 중). 반도체·EWY는 −3% 날이 28·29번 이어져 이 규칙으로는 2026에 새로 사지 못했다. "
             f"2026 앞에서 효율선을 이긴 규칙이 뒤에서도 이긴 비율은 6개 상품에서 {min(a for a, b in s26.values()) * 100:.0f}~{max(a for a, b in s26.values()) * 100:.0f}%로, "
             f"고르지 않았을 때({min(b for a, b in s26.values()) * 100:.0f}~{max(b for a, b in s26.values()) * 100:.0f}%)와 비슷하거나 낮았다. 그래서 2026으로 고를 정보가 없다.",
             "",
             f"- 관찰(사후, 미리 정한 고르기 밖): '1년 고점 대비 −10%에 처음 닿은 날'(하락이 반쯤 회복되면 다시 셈)부터 0~1개월 안에 사서 126일 들고 있기. "
             f"나스닥 종합 사건형은 두 시대 모두 아무 날보다 나았다(1971~2005 {F['E2_10_sel'][0] * 100:+.1f}~{F['E2_10_sel'][1] * 100:+.1f}%p, "
             f"2006~2025 {F['E2_10_chk'][0] * 100:+.1f}~{F['E2_10_chk'][1] * 100:+.1f}%p, 21일 대기 사건 {F['E2_10_n'][0]}·{F['E2_10_n'][1]}건). "
             f"나스닥 3배 전략(21일 대기)은 효율선을 고르기 {pp(e2['나스닥3']['sel']['eff'])}, 확인 {pp(e2['나스닥3']['chk']['eff'])} 앞섰다. "
             f"하지만 사건 {MIN_EVENTS}건 문턱에 못 미쳐({F['E2_10_n'][0]}건) 대표에서 빠졌고, 확인 구간을 본 뒤 찾았다. 반도체·한국 3배에서는 덜 담기보다 낮았다(확인 {pp(e2['반도체3']['chk']['eff'])}, {pp(e2['한국3']['chk']['eff'])}). "
             f"2026 뒤에서도 나스닥 3배 {pp(e2['나스닥3']['2026뒤']['eff'])}였다. 그래서 근거로 쓰지 않고 앞으로 기록만 한다.\n"]
    limits = ["\n## 한계\n",
              f"- 탐색 범위: 사건 16 × 조건 5 × 대기 10 − 중복 3×16 = 지수마다 752조합, 보유 5가지, 지수 5개 → 사건형 규칙 {752 * 5 * 5:,}개(창별 칸 {J['grid']['사건형_칸']:,}). "
              f"전략형은 752 × 보유 5 × 갈래 3 × 1·3배 = {J['grid']['전략형_규칙']:,}개다. 고르기 1등은 이 중 최대값이라 선택편향이 크다(나스닥 종합 고르기 상위 10개의 확인 평균 {pp(info['^IXIC']['상위10_확인평균'])}, 전체 평균 {pp(info['^IXIC']['전체_확인평균'])}).",
              "- 사건이 몰린다. 하루 −2~−3% 날은 하락장에 줄지어 나와, 겹치지 않는 표본이 사건 수의 1/2~1/6이다. 대략 t는 대부분 2 아래다.",
              "- '바로 매수 대비'는 같은 사건의 다음 날 매수(C0·0일) 평균과 비교했다. '또 나면 다시 셈'(C1)은 사는 횟수가 달라, 이 차에는 '몰린 때 덜 사기' 효과가 섞인다(그래서 시대별로도 봤다).",
              "- 지수는 가격 지수다(배당 없음). 아무 날 기준도 같은 지수라 초과에는 영향이 작지만, 전략 연율은 배당만큼 낮다. EWY·SOXX는 배당 수정.",
              f"- 체결은 지수·이음 상품 모두 다음 날 종가로 통일했다(옛 시가 자료가 없거나 전날 종가와 같음). 대표를 2006~2025에 다음 날 시가로 바꿔도 {J['대표_시가체결_2006_2025']['n']}건, 평균 {pct(J['대표_시가체결_2006_2025']['mean'])}(종가 체결 {pct(rep_chk['mean'])})로 거의 같았다.",
              "- 3배는 합성이다(일일 3배 − 2×^IRX − 보수 0.95% − 추가 비용). 1970~80년대는 단기 금리가 최고 15%대라 금융비용이 컸고, 그때 실제 상품은 없었다. 나스닥 3배 상품은 1985-09까지 나스닥 종합, 그 뒤 나스닥100이다.",
              "- 사건 정의의 재무장 값(1년 고점 낙폭은 −Y/2까지 회복, VIX는 0.8K 아래)과 C2·C3 최대 63일 대기는 미리 하나로 정했고 다른 값은 재지 않았다.",
              "- 표본은 결국 회복한 미국·한국 시장이다. 회복하지 못한 시장(1990년 뒤 일본 같은)은 없다. 고점 회복 보유는 504일에서 자른다.",
              "- 2026 사건은 126일이 아직 안 지나 대부분 '진행 중'이다. 전략형은 보유 중 새 사건을 무시하고, 현금 이자 0이다(연 4% 판은 민감도 표).",
              "- 매수 추천이나 주문 수량이 아니다. 데스크 규칙을 바꾸지 않는다."]
    md = head + concl + body + limits
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{ID}.json").write_text(json.dumps(J, ensure_ascii=False, separators=(",", ":"), default=_j), encoding="utf-8")
    (RESULTS / f"{ID}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("저장:", RESULTS / f"{ID}.md")




def _j(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


if __name__ == "__main__":
    main()
