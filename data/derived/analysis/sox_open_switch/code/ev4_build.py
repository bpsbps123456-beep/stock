"""BRIEF4 사건 표 — 면 2 × 문턱 4 단방향 H0를 lib.Engine(w90) 위에서 돌리고, 결정 봉 조건 28개와 결과 변형을 붙인다.

사용: cd code && py ev4_build.py → results/ev4_events.parquet, results/ev4_events_meta.json

사건(BRIEF4 '사건과 결과', h0.H0 단방향판과 같은 규칙)
- 면 user = SOXL 오름 → SOXS 매수, mirror = SOXS 오름 → SOXL 매수. 문턱 x ∈ {2.5, 3, 3.5, 4}%. 면·문턱마다 따로 엔진 실행.
- 오른 종목 R의 봉 i 종가 ≥ L × (1 + x). L = s0(09:30 봉, 또는 앞 매매 청산 다음 봉)부터 봉 i까지 R 저가의 최저.
- 결정 봉 i ≤ cut − 16(w90: 10:44 봉, 마감 10:45). 들고 있는 동안(주문 봉 ~ 청산 봉) 신호 무시, 청산 봉에서는 새로 안 들어감.
- 주 결과 r_w90 = 엔진 매매(다음 봉 종가 + 반 호가 매수, 반대 종목 +1% 지정가, 못 닿으면 10:59 봉 종가 − 반 호가, 손절 없음).
결과 변형(같은 결정 봉, h0.sim_trade 단일 매매 계산): r_w90_s2(−2% 손절), r_w90_c2(비용 2배), r_follow(오른 종목 R을 같은 규칙으로),
  r_std(결정 봉 ≤ 09:44 봉(마감 09:45)만, 창 끝 09:59 봉 종가; 아니면 NaN). MAE·MFE = 체결 다음 봉 ~ 청산 봉(포함) 저가 최저·고가 최고
  ÷ 체결가 − 1을 각각 0 이하·0 이상으로 자름(h0 정의). 손익 %(1.0 = 1%).

조건 28개(BRIEF4 표): on_bar 안에서 ctx(오늘 배열은 봉 i까지만, 지난 날은 prev_bars·prev_day)로만 계산한다. 지난 날 값은 절대 날 색인으로
기억(Past)하지만 모두 ctx.prev_* 로 읽은 과거 값이다. 예외 하나: RSI 사슬의 '거래 있는 봉' 표시는 Ctx가 주지 않아 has_bars()가
ctx._D.has를 ctx.bars·prev_bars와 같은 창(오늘은 봉 i까지, 지난 날은 k ≥ 1)으로 잘라 읽는다(미래 참조 시험이 이 배열도 지운다). 수치 열 <이름>과 수준 열 <이름>_lv(순서 있는 범주)를 둔다. 정의는 COND 표와 meta JSON.
수준 경계(2026-11-02 ev4_audit 반영): 값이 경계와 EPS(1e-9) 안이면 경계값으로 맞춘 뒤 BRIEF4 글자대로 배정, c03 '2~4%'는 [2, 4] 닫힘.
모호한 정의의 선택은 meta 'interpretation_decisions'(조건 17~19 창은 사건 L과 같은 s0~i, 09:30 판은 보조 열 alt_*).
점검: (1) 엔진 r_w90 = sim_trade 재계산, (2) h0.H0(thr=x, sides=(면,)) 엔진 실행과 매매 비트 일치(8조합), (3) 사용자 방향 x=3 w90
2026 하루 평균 = h0.json 단독 실행 값, (4) 미래 참조 시험: 오늘 봉 i 뒤와 다음 날들을 지운 자료로 조건을 다시 계산해 같은지.
"""
import hashlib
import json
import os
import sys
import time

os.environ["SOS_UNLOCK"] = "frozen"          # BRIEF4: 2023-01-03~2026-10-28 전체를 쓴다(사전 등록이 허용)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
from lib import Data, Ctx, Engine, Order, INST, RTH0, RTH1, ROOT, PERIODS, idx, hm  # noqa: E402
from h0 import H0, sim_trade  # noqa: E402

RES = os.path.join(ROOT, "results")
SIDES = {"user": ("SOXL", -1), "mirror": ("SOXS", 1)}        # (오른 종목 R, 사는 pos)
XS = [0.025, 0.03, 0.035, 0.04]
TGT = 0.01
CUT_W90, CUT_STD = idx("11:00"), idx("10:00")
STD_LAST_DEC = CUT_STD - 16                                 # 09:44 봉(마감 09:45)
I0800, I1300, I0934 = idx("08:00"), idx("13:00"), idx("09:34")
E90_END = idx("11:00")                                      # 장초 90분 = 09:30~10:59 봉
RSI_N = 14
T0 = time.time()


# ------------------------------------------------------------------ 수준 나누기(경계는 BRIEF4 표, 결과 보기 전 고정)
# 경계 부동소수 보정(2026-11-02, ev4_audit): 정확한 값이 경계와 같은데 float 계산이 경계 바로 옆(0.29999999999999466,
# 4.0000000000000036 등)에 떨어지면 BRIEF4 글자와 반대 칸에 들어간다(5,194건 중 13건). 그래서 수준을 정하기 전에 값이 경계와
# EPS 안이면 경계값으로 맞춘다. 수치 열(c##_*)은 그대로 두고 수준 열(_lv)에만 쓴다. 값은 % 또는 1 근처 비율이라 1e-9가 float
# 오차(~1e-13)보다 충분히 크고 실제로 다른 값 사이 간격보다 충분히 작다.
EPS = 1e-9


def snap(v, edges):
    """v가 경계 e와 EPS 안이면 e를 돌려준다."""
    for e in edges:
        if abs(v - e) <= EPS:
            return e
    return v


def lv_right(v, edges, labels):
    """[e_k, e_{k+1}) 구간(왼쪽 닫힘). NaN → None."""
    if v is None or not np.isfinite(v):
        return None
    return labels[int(np.searchsorted(edges, snap(v, edges), side="right"))]


def lv_left(v, edges, labels):
    """(e_k, e_{k+1}] 구간(오른쪽 닫힘: '≤ a' 꼴). NaN → None."""
    if v is None or not np.isfinite(v):
        return None
    return labels[int(np.searchsorted(edges, snap(v, edges), side="left"))]


def lv_sym(v, a, labels):
    """≤ −a / (−a, +a) / ≥ +a."""
    if v is None or not np.isfinite(v):
        return None
    v = snap(v, (-a, a))
    return labels[0] if v <= -a else labels[2] if v >= a else labels[1]


def lv_band(v, a, labels):
    """< −a / [−a, +a] / > +a."""
    if v is None or not np.isfinite(v):
        return None
    v = snap(v, (-a, a))
    return labels[0] if v < -a else labels[2] if v > a else labels[1]


C03_LV = ["<0", "0~2%", "2~4%", ">4%"]


def lv_c03(v, x=None):
    """조건 3 '<0 / 0~2% / 2~4% / >4%'. '>4%'는 4를 넣지 않으므로 '2~4%'는 [2, 4] 닫힘(정확히 4% → '2~4%').
    2%는 '0~2%'·'2~4%' 두 글자에 다 걸리는 겹친 경계라 위 칸 '2~4%'(앞 판과 같은 배정)."""
    if v is None or not np.isfinite(v):
        return None
    v = snap(v, (0.0, 2.0, 4.0))
    return C03_LV[0] if v < 0 else C03_LV[1] if v < 2 else C03_LV[2] if v <= 4 else C03_LV[3]


def lv_bool(v, labels=("예", "아니오")):
    if v is None or not np.isfinite(v):
        return None
    return labels[0] if v >= 0.5 else labels[1]


# (번호, 열 이름, 수준 라벨, 수준 함수(v, x), 정의) — 수치 단위: '%'가 붙은 값은 % (1.0 = 1%)
COND = [
    (1, "c01_time", ["09:31~09:44", "09:45~09:59", "10:00~10:14", "10:15~10:45"],
     lambda v, x: lv_left(v, [14, 29, 44], ["09:31~09:44", "09:45~09:59", "10:00~10:14", "10:15~10:45"]),
     "신호(결정) 봉 마감 시각, 수치 = 09:30부터 분(1 = 09:31 마감 = 09:30 봉)"),
    (2, "c02_speed", ["≤5", "6~15", "16~30", ">30"],
     lambda v, x: lv_left(v, [5, 15, 30], ["≤5", "6~15", "16~30", ">30"]),
     "L 봉(s0~i 저가 최저를 만든 마지막 봉) → 신호 봉 분"),
    (3, "c03_vs_open", C03_LV, lv_c03,
     "R 신호 봉 종가 ÷ R 09:30 봉 시가 − 1 (%), 구간 (−∞,0)·[0,2)·[2,4]·(4,∞) — BRIEF4 '>4%'가 4를 넣지 않으므로 4는 '2~4%'"),
    (4, "c04_vs_prevclose", ["아래", "위"], lambda v, x: lv_right(v, [0], ["아래", "위"]),
     "R 신호 봉 종가 ÷ R 전날 정규장 종가 − 1 (%), 0 이상 = 위"),
    (5, "c05_gap", ["≤−1%", "−1~+1%", "≥+1%"], lambda v, x: lv_sym(v, 1.0, ["≤−1%", "−1~+1%", "≥+1%"]),
     "SOXL 09:30 시가 ÷ SOXL 전날 정규장 종가 − 1 (%) × 부호(user +1, mirror −1)"),
    (6, "c06_first5", ["하락", "상승"], lambda v, x: lv_right(v, [0], ["하락", "상승"]),
     "R 09:34 봉 종가 ÷ R 09:30 봉 시가 − 1 (%), 0 이상 = 상승. 결정 봉 < 09:34 봉(마감 09:35 전)이면 아직 모르는 값이라 NaN"),
    (7, "c07_vwap", ["<1%", "1~2%", "2~3%", "≥3%"], lambda v, x: lv_right(v, [1, 2, 3], ["<1%", "1~2%", "2~3%", "≥3%"]),
     "R 신호 봉 종가 ÷ R 세션 VWAP(09:30 봉~신호 봉, 봉 VWAP × 거래량 누적) − 1 (%)"),
    (8, "c08_rsi", ["<70", "70~80", "≥80"], lambda v, x: lv_right(v, [70, 80], ["<70", "70~80", "≥80"]),
     "R 1분 RSI14(Wilder, 첫 14개 단순 평균 씨앗). 정규장 봉 중 거래 있는 봉(Data.has)만 자료 첫날(2023-01-03)부터 날 넘어 이음, "
     "빈 분은 앞 값(events_sel.py와 같은 정의)"),
    (9, "c09_rsidiv", ["예", "아니오"], lambda v, x: lv_bool(v),
     "1 = 신호 봉 고가가 장중(09:30~) 신고가(앞 봉들 고가 최고 초과)이고 RSI(i) < 앞 신고가 봉 RSI. "
     "앞 신고가 봉 = 신호 봉 전 마지막 신고가 봉(없으면 09:30 봉). 신호 봉이 신고가가 아니면 0"),
    (10, "c10_wick", ["<0.3", "≥0.3"], lambda v, x: lv_right(v, [0.3], ["<0.3", "≥0.3"]),
     "R 신호 봉 (고가 − max(시가, 종가)) ÷ (고가 − 저가), 고가 = 저가면 0"),
    (11, "c11_vol", ["<1", "1~2", "≥2"], lambda v, x: lv_right(v, [1, 2], ["<1", "1~2", "≥2"]),
     "R 신호 봉 거래량 ÷ 직전 20봉 평균(정규장 분만 날 넘어 이음, 빈 분 0 포함)"),
    (12, "c12_risevol", ["<1", "≥1"], lambda v, x: lv_right(v, [1], ["<1", "≥1"]),
     "R 오름 구간(L 봉~신호 봉) 평균 거래량 ÷ L 봉 직전 20봉 평균(정규장 분만 날 넘어 이음)"),
    (13, "c13_newhighs", ["0~1", "2~3", "≥4"], lambda v, x: lv_left(v, [1, 3], ["0~1", "2~3", "≥4"]),
     "봉 i−14~i 중 R 고가가 장중(09:30~) 신고가를 갱신한 봉 수(09:30 봉 자체는 갱신으로 안 셈)"),
    (14, "c14_topclose", ["예", "아니오"], lambda v, x: lv_bool(v),
     "1 = R 신호 봉 종가 ≥ 09:30 봉~신호 봉 종가 최고"),
    (15, "c15_vs_prevhigh", ["0.5% 넘게 아래", "±0.5% 안", "0.5% 넘게 위"],
     lambda v, x: lv_band(v, 0.5, ["0.5% 넘게 아래", "±0.5% 안", "0.5% 넘게 위"]),
     "R 신호 봉 종가 ÷ R 전날 정규장 고가 − 1 (%)"),
    (16, "c16_vs_prehigh", ["아래", "위"], lambda v, x: lv_right(v, [0], ["아래", "위"]),
     "R 신호 봉 종가 ÷ R 프리장(04:00~09:29 봉) 고가 최고 − 1 (%), 0 이상 = 위"),
    (17, "c17_qqq", ["<0.5%", "≥0.5%"], lambda v, x: lv_right(v, [0.5], ["<0.5%", "≥0.5%"]),
     "QQQ 같은 기준 움직임 (%): user = 종가 ÷ s0~i 저가 최저 − 1, mirror = 1 − 종가 ÷ s0~i 고가 최고(R 쪽 방향으로 맞춤). "
     "'같은 기준' = 사건 L과 같은 창 s0~i(순번 1은 s0 = 09:30 봉이라 장중 저점과 같다). 09:30부터 판은 보조 열 alt_c17_qqq_0930"),
    (18, "c18_nvda", ["<x/3", "≥x/3"], lambda v, x: lv_right(v, [x * 100 / 3], ["<x/3", "≥x/3"]),
     "NVDA 같은 기준 움직임 (%), 경계 x/3 (x = 3%면 1%). 창·부호는 조건 17과 같음, 09:30부터 판은 alt_c18_nvda_0930"),
    (19, "c19_soxx", ["<0.8", "≥0.8"], lambda v, x: lv_right(v, [0.8], ["<0.8", "≥0.8"]),
     "SOXX 같은 기준 움직임 ÷ (R 오름(종가 ÷ L − 1) ÷ 3). 창·부호는 조건 17과 같음, 09:30부터 판(분모도 09:30~i R 저가 최저)은 "
     "alt_c19_soxx_0930"),
    (20, "c20_prevret", ["≤−3%", "−3~+3%", "≥+3%"], lambda v, x: lv_sym(v, 3.0, ["≤−3%", "−3~+3%", "≥+3%"]),
     "SOXL 전날 정규장 종가 ÷ 그 전날 정규장 종가 − 1 (%)"),
    (21, "c21_prevrange", ["<1", "≥1"], lambda v, x: lv_right(v, [1], ["<1", "≥1"]),
     "SOXL 전날 정규장 (고가 − 저가) ÷ 시가 ÷ 지난 60거래일(d−1~d−60) 같은 값 중앙값. 60일 미만이면 NaN"),
    (22, "c22_preveff", ["<0.6", "≥0.6"], lambda v, x: lv_right(v, [0.6], ["<0.6", "≥0.6"]),
     "SOXL 전날 장초 90분 |방향 효율| = |10:59 봉 종가 − 09:30 시가| ÷ (09:30~10:59 고가 최고 − 저가 최저) (structure.py g 정의)"),
    (23, "c23_ret5", ["≤−5%", "−5~+5%", "≥+5%"], lambda v, x: lv_sym(v, 5.0, ["≤−5%", "−5~+5%", "≥+5%"]),
     "SOXL 전날 정규장 종가 ÷ 6거래일 전 정규장 종가 − 1 (%) (지난 5거래일 수익)"),
    (24, "c24_pre", ["≤−1%", "−1~+1%", "≥+1%"], lambda v, x: lv_sym(v, 1.0, ["≤−1%", "−1~+1%", "≥+1%"]),
     "SOXL 09:29 봉 종가 ÷ 08:00 가격(07:59 봉 종가, 없으면 08:00~09:29 첫 거래 봉 시가) − 1 (%) (structure.py pre_ret)"),
    (25, "c25_weekday", ["월", "화", "수", "목", "금"], lambda v, x: None if not np.isfinite(v) else "월화수목금"[int(v)],
     "요일, 수치 0 = 월 ~ 4 = 금"),
    (26, "c26_ma20", ["<1", "≥1"], lambda v, x: lv_right(v, [1], ["<1", "≥1"]),
     "SOXL 전날 정규장 종가 ÷ 지난 20거래일(d−1~d−20) 정규장 종가 평균"),
    (27, "c27_e90range", ["<1", "≥1"], lambda v, x: lv_right(v, [1], ["<1", "≥1"]),
     "지난 5거래일 SOXL 장초 90분 고저폭((09:30~10:59 고가 최고 − 저가 최저) ÷ 09:30 시가) 평균 ÷ 지난 60거래일 같은 값 중앙값"),
    (28, "c28_seq", ["1", "≥2"], lambda v, x: lv_left(v, [1], ["1", "≥2"]),
     "같은 날 같은 면·문턱 몇 번째 사건(1부터)"),
]
CNAME = [c[1] for c in COND]
ALT = ["alt_c17_qqq_0930", "alt_c18_nvda_0930", "alt_c19_soxx_0930"]   # 보조(해석 민감도), C2 특징·C1 칸에 넣지 않는다
CKIND = {"c09_rsidiv": "범주(0/1)", "c14_topclose": "범주(0/1)", "c25_weekday": "범주(요일 0~4)", "c28_seq": "범주(순번)"}


# ------------------------------------------------------------------ RSI 사슬

def rsi_run(closes, st):
    """Wilder RSI14를 closes(거래 있는 정규장 봉)에 이어서 계산. st = (앞 종가, 차분 수, 씨앗 합 g, l, ag, al, 마지막 RSI)."""
    prev, n, gs, ls, ag, al, last = st
    out = np.full(len(closes), np.nan)
    for k, c in enumerate(closes):
        if prev is None:
            prev = c
            continue
        dl = c - prev
        prev = c
        g, l = (dl, 0.0) if dl > 0 else (0.0, -dl)
        n += 1
        if n <= RSI_N:
            gs += g
            ls += l
            if n < RSI_N:
                continue
            ag, al = gs / RSI_N, ls / RSI_N
        else:
            ag = ag + (g - ag) / RSI_N
            al = al + (l - al) / RSI_N
        last = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
        out[k] = last
    return out, (prev, n, gs, ls, ag, al, last)


RSI0 = (None, 0, 0.0, 0.0, None, None, np.nan)


class Past:
    """지난 날 값. 모두 ctx.prev_day / ctx.prev_bars(k ≥ 1)로 읽고, 같은 과거 날이면 값이 같으므로 절대 날 색인으로 기억한다."""
    def __init__(self):
        self.dly, self.rend, self.e90, self.rsi, self.dayf = {}, {}, {}, {}, {}

    def day(self, ctx, sym, k):
        key = (sym, ctx.d - k)
        if key not in self.dly:
            self.dly[key] = ctx.prev_day(sym, k)
        return self.dly[key]

    def rth_end(self, ctx, k):
        """k일 전 정규장 끝 색인. lib.Data와 같은 반일장 규칙(13:00~15:59 SOXL 거래량 < 09:30~12:59의 15%)을 그날 거래량으로 다시 잰다."""
        dabs = ctx.d - k
        if dabs not in self.rend:
            v = ctx.prev_bars("SOXL", "Volume", k)
            self.rend[dabs] = I1300 if v[I1300:RTH1].sum() < 0.15 * v[RTH0:I1300].sum() else RTH1
        return self.rend[dabs]

    def early90(self, ctx, k):
        """k일 전 SOXL 장초 90분 (방향 효율, 고저폭 ÷ 시가)."""
        dabs = ctx.d - k
        if dabs not in self.e90:
            o = ctx.prev_bars("SOXL", "Open", k)[RTH0]
            h = ctx.prev_bars("SOXL", "High", k)[RTH0:E90_END].max()
            lo = ctx.prev_bars("SOXL", "Low", k)[RTH0:E90_END].min()
            c = ctx.prev_bars("SOXL", "Close", k)[E90_END - 1]
            self.e90[dabs] = ((c - o) / (h - lo) if h > lo else np.nan, (h - lo) / o)
        return self.e90[dabs]

    def rsi_end(self, ctx, sym, k):
        """k일 전 정규장 끝의 RSI 사슬 상태(사슬은 자료 첫날부터)."""
        if ctx.d - k < 0:
            return RSI0
        if (sym, ctx.d - k) in self.rsi:
            return self.rsi[(sym, ctx.d - k)]
        kk = k
        while ctx.d - (kk + 1) >= 0 and (sym, ctx.d - (kk + 1)) not in self.rsi:
            kk += 1
        st = self.rsi.get((sym, ctx.d - (kk + 1)), RSI0)
        for j in range(kk, k - 1, -1):
            c = ctx.prev_bars(sym, "Close", j)
            e = self.rth_end(ctx, j)
            m = has_bars(ctx, sym, j)[RTH0:e]
            _, st = rsi_run(c[RTH0:e][m], st)
            self.rsi[(sym, ctx.d - j)] = st
        return st


def has_bars(ctx, sym, k=0):
    """거래가 있었던 봉 표시(lib.Data.has). ctx.bars·prev_bars와 같은 창: k = 0이면 오늘 봉 i까지만, k ≥ 1이면 k일 전 하루.
    거래량 > 0으로는 대신 못 한다: SOXS 2023~24는 역분할 조정으로 실제 거래 봉의 거래량이 0으로 반올림된 봉이 있다."""
    H = ctx._D.has[sym]
    return H[ctx.d, :ctx.i + 1] if k == 0 else H[ctx.d - k]


def rsi_today(ctx, past, sym, i):
    """오늘 09:30 봉~봉 i의 RSI(빈 분은 앞 값, 첫 값 앞은 전날 끝 RSI)."""
    st = past.rsi_end(ctx, sym, 1)
    c = ctx.bars(sym, "Close")[RTH0:i + 1]
    m = has_bars(ctx, sym)[RTH0:i + 1]
    vals, _ = rsi_run(c[m], st)
    out = np.empty(len(c))
    last, k = st[6], 0
    for j in range(len(c)):
        if m[j]:
            if np.isfinite(vals[k]):
                last = vals[k]
            k += 1
        out[j] = last
    return out


def prev20_vol(ctx, past, sym, j):
    """봉 j 직전 20봉 거래량 평균(정규장 분만, 날 넘어 이음, 빈 분 0 포함)."""
    today = ctx.bars(sym, "Volume")[RTH0:j]
    need = 20 - len(today)
    if need <= 0:
        w = today[-20:]
    else:
        if ctx.d < 1:
            return np.nan
        e = past.rth_end(ctx, 1)
        w = np.r_[ctx.prev_bars(sym, "Volume", 1)[RTH0:e][-need:], today]
    m = w.mean()
    return m if m > 0 else np.nan


def move(ctx, sym, s0, i, sigma):
    """s0~i 같은 기준 움직임 (%). sigma +1 = 저가 최저 대비 오름, −1 = 고가 최고 대비 내림."""
    c = ctx.bars(sym, "Close")[i]
    if sigma > 0:
        return (c / ctx.bars(sym, "Low")[s0:i + 1].min() - 1) * 100
    return (1 - c / ctx.bars(sym, "High")[s0:i + 1].max()) * 100


def day_features(ctx, past):
    """오늘 09:30 봉 시가와 지난 날만 쓰는 SOXL 날 값(결정 봉 ≥ 09:30 봉이라 모두 이미 앎)."""
    if ctx.d in past.dayf:
        return past.dayf[ctx.d]
    f = {k: np.nan for k in ["gap", "prevret", "prevrange", "preveff", "preveff_signed", "ret5", "pre", "ma20", "e90range"]}
    d = ctx.d
    so = ctx.bars("SOXL", "Open")[RTH0]
    if d >= 1:
        p1 = past.day(ctx, "SOXL", 1)
        f["gap"] = (so / p1["close"] - 1) * 100
        e, _ = past.early90(ctx, 1)
        f["preveff_signed"], f["preveff"] = e, abs(e)
    if d >= 2:
        f["prevret"] = (past.day(ctx, "SOXL", 1)["close"] / past.day(ctx, "SOXL", 2)["close"] - 1) * 100
    if d >= 6:
        f["ret5"] = (past.day(ctx, "SOXL", 1)["close"] / past.day(ctx, "SOXL", 6)["close"] - 1) * 100
    if d >= 20:
        cl = [past.day(ctx, "SOXL", k)["close"] for k in range(1, 21)]
        f["ma20"] = cl[0] / np.mean(cl)
    if d >= 60:
        rg = [(lambda p: (p["high"] - p["low"]) / p["open"])(past.day(ctx, "SOXL", k)) for k in range(1, 61)]
        f["prevrange"] = rg[0] / np.median(rg)
        er = [past.early90(ctx, k)[1] for k in range(1, 61)]
        f["e90range"] = np.mean(er[:5]) / np.median(er)
    # 프리장 08:00 → 09:29 (structure.py pre_ret과 같은 정의)
    c = ctx.bars("SOXL", "Close")
    p0 = c[I0800 - 1]
    if not np.isfinite(p0):
        hs = np.nonzero(has_bars(ctx, "SOXL")[I0800:RTH0])[0]
        p0 = ctx.bars("SOXL", "Open")[I0800 + hs[0]] if len(hs) else np.nan
    f["pre"] = (c[RTH0 - 1] / p0 - 1) * 100
    f["weekday"] = float(ctx.day.weekday())
    past.dayf[d] = f
    return f


def features(ctx, past, i, R, sigma, x, s0, seq):
    """결정 봉 i 마감 때 아는 값만으로 조건 28개(수치)를 계산한다."""
    o_ = ctx.bars(R, "Open")
    h_ = ctx.bars(R, "High")
    l_ = ctx.bars(R, "Low")
    c_ = ctx.bars(R, "Close")
    v_ = ctx.bars(R, "Volume")
    c = c_[i]
    lo = l_[s0:i + 1]
    L = lo.min()
    jL = s0 + len(lo) - 1 - int(np.argmin(lo[::-1]))      # 최저를 만든 마지막 봉
    df = day_features(ctx, past)
    f = {"L": L, "jL": jL, "ratio": c / L, "trig_close": c}
    f["c01_time"] = float(i + 1 - RTH0)
    f["c02_speed"] = float(i - jL)
    f["c03_vs_open"] = (c / o_[RTH0] - 1) * 100
    p1 = past.day(ctx, R, 1) if ctx.d >= 1 else None
    f["c04_vs_prevclose"] = (c / p1["close"] - 1) * 100 if p1 else np.nan
    f["c05_gap"] = df["gap"] * sigma
    f["c06_first5"] = (c_[I0934] / o_[RTH0] - 1) * 100 if i >= I0934 else np.nan
    vw = ctx.bars(R, "VWAP")[RTH0:i + 1]
    vv = v_[RTH0:i + 1]
    f["c07_vwap"] = (c / ((vw * vv).sum() / vv.sum()) - 1) * 100 if vv.sum() > 0 else np.nan
    rsi = rsi_today(ctx, past, R, i)
    f["c08_rsi"] = rsi[-1]
    hh = h_[RTH0:i + 1]
    newhigh = np.r_[False, hh[1:] > np.maximum.accumulate(hh)[:-1]]
    prevk = np.nonzero(newhigh[:-1])[0]
    k = int(prevk[-1]) if len(prevk) else 0
    if newhigh[-1]:
        f["c09_rsidiv"] = float(rsi[-1] < rsi[k]) if np.isfinite(rsi[-1]) and np.isfinite(rsi[k]) else np.nan
        f["rsidiv_gap"] = rsi[-1] - rsi[k]
    else:
        f["c09_rsidiv"], f["rsidiv_gap"] = 0.0, np.nan
    rng = h_[i] - l_[i]
    f["c10_wick"] = (h_[i] - max(o_[i], c)) / rng if rng > 0 else 0.0
    f["c11_vol"] = v_[i] / prev20_vol(ctx, past, R, i)
    f["c12_risevol"] = v_[jL:i + 1].mean() / prev20_vol(ctx, past, R, jL)
    f["c13_newhighs"] = float(newhigh[max(0, i - 14 - RTH0):].sum())
    f["c14_topclose"] = float(c >= c_[RTH0:i + 1].max())
    f["c15_vs_prevhigh"] = (c / p1["high"] - 1) * 100 if p1 else np.nan
    pre = h_[:RTH0]
    pre = pre[np.isfinite(pre)]
    f["c16_vs_prehigh"] = (c / pre.max() - 1) * 100 if len(pre) else np.nan
    f["c17_qqq"] = move(ctx, "QQQ", s0, i, sigma)
    f["c18_nvda"] = move(ctx, "NVDA", s0, i, sigma)
    f["c19_soxx"] = move(ctx, "SOXX", s0, i, sigma) / ((c / L - 1) * 100 / 3)
    # 보조: 조건 17~19를 '장중 저점 = 09:30 봉부터'로 읽은 판(BRIEF4 해석 민감도 (a), 순번 1이면 위와 같음)
    f["alt_c17_qqq_0930"] = move(ctx, "QQQ", RTH0, i, sigma)
    f["alt_c18_nvda_0930"] = move(ctx, "NVDA", RTH0, i, sigma)
    f["alt_c19_soxx_0930"] = move(ctx, "SOXX", RTH0, i, sigma) / ((c / l_[RTH0:i + 1].min() - 1) * 100 / 3)
    f["c20_prevret"] = df["prevret"]
    f["c21_prevrange"] = df["prevrange"]
    f["c22_preveff"] = df["preveff"]
    f["preveff_signed"] = df["preveff_signed"]
    f["c23_ret5"] = df["ret5"]
    f["c24_pre"] = df["pre"]
    f["c25_weekday"] = df["weekday"]
    f["c26_ma20"] = df["ma20"]
    f["c27_e90range"] = df["e90range"]
    f["c28_seq"] = float(seq)
    f["soxl_gap"] = df["gap"]
    return f


class Ev4:
    """한 면·한 문턱 단방향 H0(h0.H0(sides=(면,))와 같은 규칙). 결정 봉에서 조건을 계산해 log에 남긴다."""
    start = "09:30"

    def __init__(self, side, x, log, past):
        self.R, self.buy = SIDES[side]
        self.sigma = 1 if self.R == "SOXL" else -1
        self.x, self.log, self.past = x, log, past
        self.state, self.seq = "flat", 0

    def on_day(self, ctx):
        self.s0 = ctx.i + 1                                   # 09:30 봉

    def on_bar(self, ctx, i):
        just = False
        if self.state == "ordered":
            if i >= self.fill_bar and ctx.pos == 0:           # 청산이 이 봉에서 일어났다
                self.state, self.s0, just = "flat", i + 1, True
            else:
                return None
        if just or i < self.s0 or i > ctx.cut - 16:
            return None
        L = ctx.bars(self.R, "Low")[self.s0:i + 1].min()
        c = ctx.c(self.R)[i]
        if not (np.isfinite(c) and np.isfinite(L) and L > 0 and c >= L * (1 + self.x)):
            return None
        self.seq += 1
        f = features(ctx, self.past, i, self.R, self.sigma, self.x, self.s0, self.seq)
        self.state, self.fill_bar = "ordered", i + 1
        self.log.append(dict(d=ctx.d, dec=i, fill=i + 1, s0=self.s0, pos=self.buy, **f))
        return Order(self.buy, target=TGT)


# ------------------------------------------------------------------ 실행

def run_combo(D, side, x, past, days):
    log = []
    E = Engine(D, scheme="w90")
    daily, tr = E.run(lambda: Ev4(side, x, log, past), days=days)
    assert len(log) == len(tr), (side, x, len(log), len(tr))
    lg = pd.DataFrame(log)
    tr = tr.reset_index(drop=True)
    assert (tr["entry"].values == np.array([hm(f) for f in lg["fill"]])).all()
    assert (tr["day"].values == np.array([D.days[d] for d in lg["d"]])).all()
    rows = []
    maxdiff = 0.0
    for k in range(len(lg)):
        g = lg.iloc[k]
        d, dec, pos = int(g["d"]), int(g["dec"]), int(g["pos"])
        sym = INST[pos]
        r, f, j, why, mae = sim_trade(D, d, dec, pos, CUT_W90)
        assert why == tr.at[k, "why"] and hm(j) == tr.at[k, "exit"], (side, x, k)
        maxdiff = max(maxdiff, abs(r - tr.at[k, "ret"]))
        A = D.A[sym]
        mfe = max(A["High"][d, f + 1:j + 1].max() / tr.at[k, "entry_px"] - 1, 0.0)
        r2, _, _, why2, _ = sim_trade(D, d, dec, pos, CUT_W90, stop=0.02)
        rc, _, _, _, _ = sim_trade(D, d, dec, pos, CUT_W90, cm=2.0)
        rf, _, _, whyf, _ = sim_trade(D, d, dec, -pos, CUT_W90)
        if dec <= STD_LAST_DEC:
            rs, _, js, whys, _ = sim_trade(D, d, dec, pos, CUT_STD)
        else:
            rs, whys = np.nan, None
        day = D.days[d]
        rows.append({"side": side, "rise": "SOXL" if pos == -1 else "SOXS", "buy": sym, "x": x,
                     "day": pd.Timestamp(day), "year": day.year, "month": f"{day.year}-{day.month:02d}", "d": d,
                     "dec_i": dec, "dec_close": hm(dec + 1), "fill_i": f, "exit_i": j,
                     "entry": tr.at[k, "entry"], "exit": tr.at[k, "exit"],
                     "entry_px": tr.at[k, "entry_px"], "exit_px": tr.at[k, "exit_px"],
                     "s0_i": int(g["s0"]), "L_i": int(g["jL"]), "L": g["L"], "trig_close": g["trig_close"], "ratio": g["ratio"],
                     "r_w90": tr.at[k, "ret"] * 100, "why": why, "hit": int(why == "target"),
                     "min_to_target": float(j - f) if why == "target" else np.nan,
                     "mae": mae * 100, "mfe": mfe * 100,
                     "r_w90_s2": r2 * 100, "why_s2": why2, "r_w90_c2": rc * 100,
                     "r_follow": rf * 100, "why_follow": whyf,
                     "r_std": rs * 100 if np.isfinite(rs) else np.nan, "why_std": whys,
                     **{c: g[c] for c in CNAME}, "rsidiv_gap": g["rsidiv_gap"], "preveff_signed": g["preveff_signed"],
                     "soxl_gap": g["soxl_gap"], **{a: g[a] for a in ALT}})
    return pd.DataFrame(rows), daily, maxdiff


def h0_bitcheck(D, side, x, ev, days):
    """h0.H0(thr=x, sides=(면,)) 엔진 실행과 매매(날·체결·청산·수익)가 같은지."""
    _, buy = SIDES[side]
    E = Engine(D, scheme="w90")
    _, tr = E.run(lambda: H0(ref="low", thr=x, sides=(buy,)), days=days)
    a = tr[["day", "entry", "exit", "ret"]].reset_index(drop=True)
    b = ev[["day", "entry", "exit", "r_w90"]].reset_index(drop=True)
    if len(a) != len(b):
        return {"n_h0": int(len(a)), "n_ev4": int(len(b)), "match": False}
    same = ((pd.to_datetime(a["day"]).values == b["day"].values) & (a["entry"].values == b["entry"].values)
            & (a["exit"].values == b["exit"].values))
    diff = float(np.max(np.abs(a["ret"].values * 100 - b["r_w90"].values))) if len(a) else 0.0
    return {"n_h0": int(len(a)), "n_ev4": int(len(b)), "match": bool(same.all() and diff < 1e-10), "max_abs_diff_pct": diff}


def leak_test(D, ev, n=40, seed=20261102):
    """미래 참조 시험: 사건 n개를 골라, 오늘 봉 i 뒤를 쓰레기 값으로 바꾸고 다음 날들을 잘라 낸 자료에서 조건을 새로(기억 없이) 다시 계산한다."""
    rng = np.random.default_rng(seed)
    pick = ev.iloc[rng.choice(len(ev), size=min(n, len(ev)), replace=False)]
    bad = 0
    detail = []
    for _, e in pick.iterrows():
        d, i = int(e["d"]), int(e["dec_i"])
        D2 = Data.__new__(Data)
        D2.days = D.days[:d + 1]
        D2.pos = {dd: k for k, dd in enumerate(D2.days)}
        D2.A, D2.has = {}, {}
        for s in D.A:
            D2.A[s] = {}
            for fld, arr in D.A[s].items():
                row = arr[d].copy()
                row[i + 1:] = rng.uniform(1, 1e6, NMIN_ - i - 1)
                D2.A[s][fld] = np.vstack([arr[:d], row[None, :]])
            hrow = D.has[s][d].copy()
            hrow[i + 1:] = rng.random(NMIN_ - i - 1) < 0.5
            D2.has[s] = np.vstack([D.has[s][:d], hrow[None, :]])
        dd = D.daily.reset_index()
        D2.daily = dd[dd["day"] < D.days[d]].set_index(["sym", "day"]).sort_index()
        ctx = Ctx(D2, d)
        ctx.i = i
        R = e["rise"]
        sigma = 1 if R == "SOXL" else -1
        f = features(ctx, Past(), i, R, sigma, e["x"], int(e["s0_i"]), int(e["c28_seq"]))
        diffs = [c for c in CNAME + ALT
                 if not (np.isclose(f[c], e[c], rtol=0, atol=1e-9) or (np.isnan(f[c]) and np.isnan(e[c])))]
        if diffs:
            bad += 1
            detail.append({"day": str(e["day"].date()), "side": e["side"], "x": e["x"], "dec": hm(i + 1), "diff": diffs})
    return {"n": int(len(pick)), "mismatch_events": bad, "detail": detail[:10]}


NMIN_ = 960


def main():
    D = Data("alp1m_long")
    days = list(range(len(D.days)))
    print(f"자료 {D.days[0]}~{D.days[-1]} {len(D.days)}일 ({time.time() - T0:.0f}s)", flush=True)
    past = Past()
    parts, checks = [], {"engine_vs_sim_max_abs_diff": {}, "h0_bitmatch": {}}
    for side in SIDES:
        for x in XS:
            ev, daily, md = run_combo(D, side, x, past, days)
            parts.append(ev)
            checks["engine_vs_sim_max_abs_diff"][f"{side}_{x}"] = md
            checks["h0_bitmatch"][f"{side}_{x}"] = h0_bitcheck(D, side, x, ev, days)
            print(f"{side} x={x}: 사건 {len(ev)}  sim차 {md:.2e}  H0 {checks['h0_bitmatch'][f'{side}_{x}']['match']} "
                  f"({time.time() - T0:.0f}s)", flush=True)
    ev = pd.concat(parts, ignore_index=True)
    # 수준 열
    for num, col, labels, fn, _ in COND:
        lv = [fn(v, x) for v, x in zip(ev[col].values, ev["x"].values)]
        ev[col + "_lv"] = pd.Categorical(lv, categories=labels, ordered=True)
    for c in ["side", "rise", "buy", "why", "why_s2", "why_follow", "why_std"]:
        ev[c] = ev[c].astype("category")
    for c in ["year", "d", "dec_i", "fill_i", "exit_i", "s0_i", "L_i"]:
        ev[c] = ev[c].astype("int32")
    ev["hit"] = ev["hit"].astype("int8")
    # 점검 3: 사용자 방향 x=3 w90 2026 하루 평균 vs h0.json 단독 실행
    h0j = json.load(open(os.path.join(RES, "h0.json"), encoding="utf-8"))
    ref = h0j["by_side"]["w90"]
    chk3 = {}
    for side, lab in (("user", "SOXL_up→buy_SOXS"), ("mirror", "SOXS_up→buy_SOXL")):
        e3 = ev[(ev["side"] == side) & (ev["x"] == 0.03)]
        for pl, ps in (("2026", ["sel", "conf"]), ("2026H1", ["sel"]), ("2026H2", ["conf"]), ("2025", ["aux"])):
            dd = sorted(sum((D.days_in(p) for p in ps), []))
            dset = {pd.Timestamp(D.days[k]) for k in dd}
            sub = e3[e3["day"].isin(dset)]
            per = sub.groupby("day")["r_w90"].apply(lambda r: (np.prod(1 + r.values / 100) - 1) * 100)
            mean = float(per.reindex(sorted(dset), fill_value=0.0).mean())
            o = ref[lab]["side_only_run"][pl]
            chk3[f"{side}_{pl}"] = {"ev4_mean_daily": mean, "h0json_mean_daily": o["mean_daily"],
                                    "ev4_trades": int(len(sub)), "h0json_trades": o["trades"],
                                    "ev4_active_days": int(per.size), "h0json_active_days": o["active_days"],
                                    "days": len(dset), "abs_diff": abs(mean - o["mean_daily"])}
    checks["h0json_side_only"] = chk3
    print(json.dumps(chk3, ensure_ascii=False, indent=0), flush=True)
    # 점검 4: 미래 참조
    checks["leak_test"] = leak_test(D, ev)
    print("leak", checks["leak_test"]["mismatch_events"], "/", checks["leak_test"]["n"], f"({time.time() - T0:.0f}s)", flush=True)
    # 저장
    out = os.path.join(RES, "ev4_events.parquet")
    ev.to_parquet(out, index=False, compression="zstd")
    n_events = {s: {f"{x:g}": {str(y): int(((ev["side"] == s) & (ev["x"] == x) & (ev["year"] == y)).sum())
                               for y in sorted(ev["year"].unique())} for x in XS} for s in SIDES}
    nulls = {c: int(ev[c + "_lv"].isna().sum()) for c in CNAME}
    code_hash = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
    meta = {"brief": "BRIEF4.md '사건과 결과'·'조건 목록'", "data": f"Data('alp1m_long') {D.days[0]}~{D.days[-1]} {len(D.days)}일",
            "rows": int(len(ev)), "file_mb": round(os.path.getsize(out) / 1e6, 3), "code_sha256_16": code_hash,
            "units": "r_*, mae, mfe, '%' 붙은 조건 수치는 % (1.0 = 1%)",
            "event_rule": "면(user: SOXL 오름 → SOXS 매수, mirror: SOXS 오름 → SOXL 매수) × x. 오른 종목 1분 종가 ≥ L × (1 + x), "
                          "L = s0(09:30 봉 또는 앞 매매 청산 다음 봉)~봉 i 저가 최저, 결정 봉 i ≤ 10:44 봉(마감 10:45), 한 번에 하나, "
                          "청산 봉에서는 새로 안 들어감 = h0.H0(sides=(면,), thr=x) 단방향판(비트 일치 점검)",
            "columns": {
                "side/rise/buy/x": "면, 오른 종목, 산 종목, 문턱(소수, 0.03 = 3%)",
                "day/year/month/d": "날(자정 Timestamp), 연, 'YYYY-MM', Data 날 색인",
                "dec_i/dec_close": "결정(신호) 봉 색인(04:00부터 분)과 그 봉 마감 시각",
                "fill_i/entry/exit_i/exit": "체결 봉(= 결정 봉 + 1)·청산 봉 색인과 시작 시각(엔진 매매 기록과 같은 표기)",
                "entry_px/exit_px": "엔진 체결가(반 호가 포함)·청산가",
                "s0_i/L_i/L/trig_close/ratio": "L을 재기 시작한 봉, L을 만든 마지막 봉, L, 신호 봉 종가, 종가 ÷ L",
                "r_w90/why/hit/min_to_target": "엔진 매매 손익 %, 청산 사유(target/cutoff), +1% 도달(1/0), 도달까지 분(청산 봉 − 체결 봉)",
                "mae/mfe": "체결 다음 봉~청산 봉(포함) 저가 최저·고가 최고 ÷ 체결가 − 1 (%), 각각 ≤ 0·≥ 0으로 자름",
                "r_w90_s2/why_s2": "같은 진입, −2% 손절(손절 우선, 갭이면 시가) 추가",
                "r_w90_c2": "같은 진입, 반 호가 2배",
                "r_follow/why_follow": "같은 결정 봉에 오른 종목(R)을 같은 규칙(+1%·10:59)으로 산 결과",
                "r_std/why_std": "결정 봉 ≤ 09:44 봉(마감 09:45)만, 같은 +1% 목표, 못 닿으면 09:59 봉 종가 − 반 호가. 아니면 NaN",
                "c##_* / c##_*_lv": "조건 수치 / 수준(순서 있는 범주, 계산 불가면 결측)",
                "rsidiv_gap": "(보조) 신호 봉이 신고가일 때 RSI(i) − 앞 신고가 봉 RSI",
                "preveff_signed": "(보조) 조건 22의 부호 있는 값",
                "soxl_gap": "(보조) SOXL 갭 % 부호 그대로(조건 5는 면 부호를 곱한 값)",
                "alt_c17_qqq_0930/alt_c18_nvda_0930/alt_c19_soxx_0930": "(보조, 해석 민감도) 조건 17~19를 09:30 봉부터 창으로 "
                    "잰 값. 순번 1이면 c17~c19와 같다. C1 칸·C2 특징에는 넣지 않는다"},
            "level_rule": f"수준은 BRIEF4 표 글자대로 배정한다. 수준을 정하기 전에 값이 경계와 {EPS:g} 안이면 경계값으로 맞춘다"
                          "(정확한 값이 경계인데 float 오차로 반대 칸에 들어가던 13건 수정, ev4_audit 2026-11-02). 수치 열은 그대로다. "
                          "수치 열을 고정 문턱과 직접 비교할 때(C3 등)는 같은 여유를 두거나 _lv 열을 쓴다. "
                          "'1~2%'·'2~3%'처럼 글자가 겹치는 경계는 '<a'·'≥a'·'>a'로 적힌 칸이 있으면 그 글자를 따르고, "
                          "없으면 위 칸(왼쪽 닫힘)에 넣는다: c03 = 2 → '2~4%', c03 = 4 → '2~4%'('>4%'가 4를 뺌), c07 = 2 → '2~3%', "
                          "c04·c06·c16 = 0 → '위'/'상승'",
            "interpretation_decisions": {
                "c17~c19 창": "'같은 기준' = 사건 L과 같은 창(s0~i: 09:30 봉 또는 앞 사건 청산 다음 봉부터). 괄호 '장중 저점 대비'는 "
                              "기준이 저점이라는 설명으로 읽음. 순번 1(s0 = 09:30)은 두 읽기가 같다. 09:30 판은 alt_* 열. "
                              "C3 H2·H3은 c17·c18(주)을 쓰고 alt_* 결과를 함께 적는다",
                "c17~c19 거울 면 부호": "거울 면(R = SOXS)은 '같은 기준'을 R 쪽 방향으로 맞춰 고가 최고 대비 내림으로 잰다. "
                                       "조건 19 '레버리지 대비 기초 지수 확인'이 이 방향에서만 뜻이 있다",
                "c23 창": "'5거래일 SOXL 수익' = 끝난 5거래일(전날 종가 ÷ 6거래일 전 종가 − 1). 조건 20~27의 날 값 묶음과 같은 기준, "
                          "결정 봉의 오늘 움직임은 조건 3·4가 따로 잰다",
                "c02 L 봉 동률": "L과 같은 저가 봉이 여럿이면 마지막 봉(오름이 시작된 봉). 조건 12 '오름 구간(L 봉~신호 봉)'도 같은 봉",
                "c08 빈 분": "거래 있는 봉만 이음(빈 분 포함과 수준이 바뀌는 사건 0, ev4_audit)"},
            "conditions": [{"no": no, "col": col, "levels": labels, "kind": CKIND.get(col, "수치"), "def": desc,
                            "null_events": nulls[col]} for no, col, labels, _, desc in COND],
            "n_events": n_events, "checks": checks}
    with open(os.path.join(RES, "ev4_events_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    print(json.dumps(n_events, ensure_ascii=False))
    print(json.dumps(nulls, ensure_ascii=False))
    print(f"저장 {out} {meta['file_mb']}MB, {len(ev)}행 × {ev.shape[1]}열 ({time.time() - T0:.0f}s)")


if __name__ == "__main__":
    main()
