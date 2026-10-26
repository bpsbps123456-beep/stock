"""3단계 갈래 PORT3: 사용자 7종목 포트폴리오에 '선 이탈 때 일부만 판다'를 얹는다 (2026-10-10).

사용자 요청(원문 요지): "기계적 매수처럼 손절도 최대 하락 %를 다양하게 조합", "한 번에 올 매도가 아니라 특정 선을 이탈하면
일부(예: 25%)를 매도". 2단계에서 "승률 30%라도 손익비로 이기면 된다".

경로: harness.base('2026')(데스크 현행 7종목, 2026-01-02~10-09), 보조 base('hold')(2023-05-22~2025-12-31, 주간 PDF, 옛 분류).
그림자 = base.close_s(그날 종가 평가, t 종가에 앎). 체결은 다음 시가(base.open 경로).

P1 포트폴리오 낙폭 분할(조각 2개: 움직이는 q + 늘 보유 1−q)
   고점 = 그림자 추적 최고 종가(경로 시작부터, t 종가 포함). 들고 있고 무장된 조각이 그림자 ≤ 고점×(1−X)면 다음 시가에 판다.
   판 조각: 그림자 ≤ 고점×(1−D)이고 그 선 < 판 값이면 다음 시가에 되산다(깊은 재매수).
           또는 그림자 종가 ≥ 판 값×1.03이면 다음 시가에 되산다(되돌림). D 없음 = 되돌림만.
   무장: 되산 조각은 그림자 종가 > 고점×(1−X)가 될 때 다시 무장(선 아래에서 되사자마자 또 파는 것을 막음).
         [tranche_lab은 되돌림 조각을 바로 무장하지만 거기선 판 값 ≈ 선이라 1.03배면 이미 선 위다. 종가→시가 체결에서는
          판 값이 선보다 낮을 수 있어 '선 위로 올라와야 무장'으로 통일했다. 결과를 보기 전에 정함.]
   이미 D선 아래(그림자 ≤ 고점×(1−D))에서 처음 X선 이탈이 잡히면 팔지 않고 무장만 푼다(팔고 같은 시가에 되사는 셈이라).
   돈: 조각마다 자기 주식(그림자 단위)·현금 통. 되살 때 판 돈 전부로 산다 → 주기 손익 = 판 값/되산 값 − 1.
P2 포트폴리오 사다리: 25%씩 4조각, 조각마다 X_i, 재매수 D_i = X_i+5 / X_i+10 / 없음(되돌림만). 규칙은 P1과 같다.
P3 종목별 일부 손절(자체 루프, StockEngine 자료·재조정 그대로)
   종목마다 고점 = 산 뒤 최고 종가(k−1까지, 계속 들고 있으면 이어짐). k일 장중 저가 ≤ 고점×(1−X)면 그 종목 보유의 q만 판다
   (시가 ≤ 선이면 시가). 판 몫은 그 주 재조정까지 현금, 재조정 때 기본 목표 비중으로 되돌린다. 종목당 한 주에 한 번.
   q = 100%면 전량이 빠지므로 다음 재조정에서 새로 사고 고점도 새로 잰다(StockEngine 전량 손절과 같아야 함 → 점검).

평가(사전 고정, 2단계와 같음)
  창: 2026 전체·앞(1/2~5/29, 고르기)·뒤(6/1~10/9, 확인)·AI쇼크(6/22~7/29)·7~9월. 보조 2025(2023-05부터 이어 돌린 경로를 자름), 맥락 2024·관세·24여름·25가을.
  효율선초과 = 규칙 수익 − (같은 창 기본 경로 × 고정 노출 e, 0.005 간격, 낙폭이 규칙보다 깊지 않은 것 중 최대 수익).
  타이밍 값 = 규칙 수익 − (같은 창 평균 노출로 고정해 들고 있었을 때 수익). 둘 다 '선 이탈 때 판 것'의 값을 잰다.
  손절 주기: 조각(또는 종목 몫) 하나의 판 → 되산. 승률·손익비·기대값을 조각 수 기준과 비중 가중으로.
  고르기(사전 고정): G1 앞 효율선초과 최대, G2 앞 주기 기대값 최대(앞 주기 ≥ 30), G3 앞 칼마 최대. 동점은 격자 순서 앞.
실행: cd code && PYTHONIOENCODING=utf-8 py p3_PORT3.py
"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402
import stock_lab as SL  # noqa: E402

ID = "p3_PORT3"
RESULTS = HERE.parent / "results"

# ============================================================================ 격자(사전 고정 — 뒤 구간·2025를 본 뒤 바꾸지 않는다)
P1_Q = [0.25, 0.50]
P1_X = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
P1_DGAP = [0.05, 0.10, None]                     # D = X + 5%p / X + 10%p / 없음(되돌림만)
P2_LADDERS = [(0.05, 0.10, 0.15, 0.20), (0.08, 0.12, 0.16, 0.20), (0.10, 0.15, 0.20, 0.25), (0.05, 0.08, 0.12, 0.15)]
P2_DGAP = [0.05, 0.10, None]
P3_Q = [0.25, 0.50]
P3_X = [0.10, 0.15, 0.20]
RECOVER_B = 0.03                                 # 되돌림: 그림자 종가 ≥ 판 값 × 1.03
SLIP = 0.005                                     # 대표 후보 미끄러짐 민감도(팔 때 −0.5%, 살 때 +0.5%)
MIN_CYCLES_G2 = 30
SHADOW_MDD_2026 = -0.2287                        # 2026 그림자(종가) 최대 낙폭 → D ≥ 25%는 2026에 발동하지 않음
# 비교용(격자 수·고르기에 넣지 않음, 사전 선언): 같은 선에서 전량(100%) 팔기
CMP_P1_FULL = True                               # q = 100% × P1_X × P1_DGAP = 18
CMP_P3_FULL = True                               # q = 100% × P3_X = 3
# 고른 변형 이웃(사전 선언): P1·P2 X(모든 X_i) ±2%p(D 간격 유지), 되돌림 여유 2%·5%, 체결 하루 늦춤. P3 X ±2.5%p.
NB_X, NB_X_P3, NB_B = 0.02, 0.025, (0.02, 0.05)

W26 = {"전체": ("2026-01-02", "2026-10-09"), "앞": ("2026-01-02", "2026-05-29"), "뒤": ("2026-06-01", "2026-10-09"),
       "AI쇼크": ("2026-06-22", "2026-07-29"), "7~9월": ("2026-07-01", "2026-09-30")}
WH = {"2025": ("2025-01-01", "2025-12-31"), "2024": ("2024-01-01", "2024-12-31"), "관세": ("2025-02-18", "2025-04-04"),
      "24여름": ("2024-06-18", "2024-08-05"), "25가을": ("2025-10-13", "2025-11-21")}
WINS = {"2026": W26, "hold": WH}

# 첫 실행 뒤 채우는 결론·맥락·한계 문장(격자·계산과 무관, md 전용)
CONCLUSION: list[str] = [
    "**기각(규칙으로 더하지 않음).** 54개 중 앞 구간에서 같은 낙폭 고정 노출을 이긴 30개 가운데 뒤 구간과 2025를 둘 다 이긴 것은 "
    "0개였다(무조건 0/54). 앞으로 고른 대표 G1(=G3, 그림자 −8%에서 50% 매도 → −18% 재매수 또는 판 값 +3% 복귀)은 "
    "2026 +239.0%/−14.6%(기본 +223.7%/−20.3%), 7월 쇼크 −12.9%(−18.7%)였다. 하지만 뒤 구간은 +1.9%(기본 +4.6%, 효율선초과 −2.4%)였다. "
    "체결을 하루 늦추면 2026 +156.1%, 손절선을 ±2%p 옮기면 +189.9~+194.4%로 무너졌다.",
    "**일부 매도는 같은 선 전량 매도의 '작은 용량'이다.** 그림자 선 18칸에서 25%·50%·100% 팔기의 타이밍 값(같은 평균 노출 고정 대비)은 "
    "18칸 중 17칸에서 부호가 같고 크기가 대체로 비례했다(−8%→−18%: +31.8%/+61.0%/+113.2%, −15%→되돌림: −21.8%/−43.1%/−83.1%). 나쁜 선을 좋은 선으로 "
    "바꾸지 못하고 이익과 손해를 함께 줄인다. 종목별 손절(P3)만 일부가 전량보다 덜 나빴다(−10%선 2026 타이밍 값 −8.8% 대 −85.6%). "
    "그래도 2026은 6/6이 마이너스였다.",
    "**'넓게'(−15~−20%)는 2026 7종목에서 가장 나빴다.** 2026 하락의 바닥이 −15~−23%라 그 선에서 팔면 바닥 근처에서 판다"
    "(50% 매도 2026 타이밍 값 −23~−43%). 좁은 선(−5~−12%)의 2026 플러스는 대부분 '+5~10%p 더 깊은 재매수'를 붙인 칸에서 나왔다"
    "(50% 매도 8칸 중 7칸 플러스). 되돌림만이면 4칸 중 3칸이 마이너스였다. 2025(−47%까지 이어진 하락)는 반대로 −20% + 깊은 재매수가 가장 좋았다"
    "(50%: 효율선초과 +5.4~+9.2%, 좁은 선은 −5~−10% + 10%p 재매수만 +0.7~+3.5%). "
    "'승률 30%, 손익비 3.3' 모양은 G1 2026 주기 10개에서 나왔지만(기대값 +1.3%) 뒤 구간은 승률 20%, 기대값 −0.6%였다.",
]
CONTEXT: list[str] = [
    "7월 보호의 정체: G1의 7월 쇼크 −12.9%는 같은 구간 평균 노출(0.71)로 고정해 들고 있었을 때(−13.3%)와 거의 같다(타이밍 값 +0.4%). "
    "7월에 덜 잃은 것은 '덜 들고 있었기' 때문이지 잘 판 덕이 아니다. 54개 중 42개가 7월 손실을 줄였지만, 2026 수익이 기본보다 높은 것은 10개(모두 P1)였다.",
    "사다리 P2 12개: 2026 수익이 기본보다 높은 것 0/12, 7월 손실을 줄인 것 12/12, 뒤 효율선초과 0/12, 2025 1/12. "
    "사다리 노출은 같은 선 전량 매도 4개의 평균과 거의 같았다(날별 최대 차이 4~10%p). 그래서 나쁜 선(−15·−20%)의 비용도 함께 진다.",
    "종목별 일부 손절 P3 6개: 2026 타이밍 값 0/6이 플러스(−8.0~−27.6%), 2025 효율선초과는 6/6이 플러스(50%·−10%: 2025 +41.1%/−29.9%, 기본 +42.2%/−47.1%). "
    "1·2단계와 같은 장세 의존이다(이어지는 하락에선 통하고 V자에선 진다). 합격 기준인 2026에서 실패했다.",
    "2024(맥락): G1 2024 +23.9%(기본 +33.4%, 효율선초과 −8.9%). 관세 −45.7%(−47.1%), 24여름 −27.3%(−28.2%), 25가을 −27.4%(−35.3%). "
    "긴 하락에서는 −18% 재매수가 하락 중간에 사서 보호가 작았다. 2018~2022는 7종목 경로가 없다(TIME PDF 2023-05부터).",
    "밤사이 갭: 2026 7종목 경로의 평균 밤사이 수익(종가→다음 시가)은 +0.33%, 고점 대비 −15% 아래 종가 다음 날은 +1.46%였다. "
    "G1의 매도 10번 중 5번은 다음 시가가 전날 종가보다 2.6~4.7% 높았다(2/11·3/4·3/23·6/8·7/14). 체결을 하루 늦추면 이 갭을 놓친다(2026 +156.1%).",
]
LIMITS: list[str] = [
    "시험 범위: 사전 격자 54(P1 36·P2 12·P3 6)에서만 골랐다. 비교용 전량 매도 21(P1 18·P3 3), 고른 변형 이웃 14, 미끄러짐 0.5% 4개는 고르기에 쓰지 않았다. "
    "사후 점검 12개(뒤 구간 승자 6개의 하루 늦춤·미끄러짐)는 뒤·2025를 본 뒤 추가했다.",
    "고르기 정보: 앞 효율선초과 > 0인 30개 중 뒤 성공 20%(무조건 11%), 2025 성공 30%(무조건 31%), 둘 다 0%(무조건 0%)였다. "
    "뒤 성공 6개는 모두 P1 '−8~−12% 매도 → 5%p 더 깊은 재매수'다. 1단계 G′(−10% 손절·−15% 재매수, 취약 판정)와 같은 계열이고, 6개 모두 2025 효율선초과가 마이너스(−1.3~−8.9%)였다. "
    "하루 늦추면 6개 모두 뒤가 마이너스(−0.2~−9.2%)가 됐다.",
    "G1은 54개 중 앞 효율선초과 최대(+96.5%)라 최대값 선택 편향이 있다. 미끄러짐 0.5%: 2026 +206.8%/−17.3%, 앞 효율선초과 +22.6%, 뒤 −0.3%(효율선초과 −4.7%), 2025 +38.4%(효율선초과 −3.4%). "
    "되돌림 여유 2%·5%: 2026 +225.5%·+212.3%.",
    "설계 선택(결과를 보기 전에 정함): 되산 조각은 그림자가 손절선 위로 올라와야 다시 무장(tranche_lab은 되돌림 조각을 바로 무장). "
    "처음 X선 이탈이 이미 D선 아래면 팔지 않는다. 고점은 경로 시작(2026-01-02, 보조 2023-05-22)부터의 그림자 최고 종가라 그 전 고점은 보지 않는다.",
    "P3: 종목 고점은 계속 들고 있는 한 이어진다. 일부 판 종목이 선 아래에 머무르면 매주 재조정 때 되사고, 시가가 선 아래면 바로 다시 판다"
    "(같은 시가의 되사기·다시 팔기는 상쇄, 미끄러짐 없음). 판 몫은 그 주 현금이고 주중 재진입은 없다.",
    "보조 경로(2023-05~2025)는 주간 PDF와 옛 분류(미분류 56종목이 반도체 예산)라 실제 7종목과 구성이 다르다. 2025 창은 이어 돌린 경로를 자른 값이다(조각 상태가 이어짐).",
    "생존편향: 경로는 그때의 TIME 보유(PDF)로 만들어 종목 선택 편향은 작다. 다만 2023~26은 AI 강세장이라 깊은 재매수가 좋게 나오는 쪽으로 기운다. "
    "2000~02·2008·2022형 장기 약세장은 7종목 경로에 없다. QUBT 장외 자료 문제는 이 갈래와 무관하다(2023-05 이후 경로).",
    "USD 기준, 수수료 0, 현금 이자 0, 환율 무시. 그림자 판단은 미국 종가(KST 05:00 무렵)에 하고 다음 정규장 시가에 체결한다고 가정했다.",
]


@dataclass(frozen=True)
class Tr:
    w: float
    x: float | None
    d: float | None


# ============================================================================ 격자
def dval(x, gap):
    return None if gap is None else round(x + gap, 4)


def grid():
    out = []
    for q in P1_Q:
        for x in P1_X:
            for g in P1_DGAP:
                out.append({"fam": "P1", "q": q, "x": x, "gap": g, "b": RECOVER_B, "lag": 1})
    for lad in P2_LADDERS:
        for g in P2_DGAP:
            out.append({"fam": "P2", "xs": lad, "gap": g, "b": RECOVER_B, "lag": 1})
    for q in P3_Q:
        for x in P3_X:
            out.append({"fam": "P3", "q": q, "x": x})
    return out


def compare_specs():
    out = []
    if CMP_P1_FULL:
        for x in P1_X:
            for g in P1_DGAP:
                out.append({"fam": "P1", "q": 1.0, "x": x, "gap": g, "b": RECOVER_B, "lag": 1, "cmp": True})
    if CMP_P3_FULL:
        for x in P3_X:
            out.append({"fam": "P3", "q": 1.0, "x": x, "cmp": True})
    return out


def tranches(sp):
    if sp["fam"] == "P1":
        tr = [Tr(sp["q"], sp["x"], dval(sp["x"], sp["gap"]))]
        if sp["q"] < 1:
            tr.append(Tr(1 - sp["q"], None, None))
        return tuple(tr)
    return tuple(Tr(0.25, x, dval(x, sp["gap"])) for x in sp["xs"])


def name(sp):
    tail = ""
    if sp.get("b", RECOVER_B) != RECOVER_B:
        tail += f" [되돌림 {sp['b']:.0%}]"
    if sp.get("lag", 1) != 1:
        tail += f" [체결 {sp['lag'] - 1}일 늦춤]"
    if sp["fam"] == "P1":
        d = dval(sp["x"], sp["gap"])
        re = f"−{d:.0%} 재매수 또는 되돌림" if d is not None else "되돌림만"
        return f"P1 그림자 −{sp['x']:.0%}에서 {sp['q']:.0%} 매도 → {re}{tail}"
    if sp["fam"] == "P2":
        xs = "/".join(f"{x * 100:.0f}" for x in sp["xs"])
        re = f"재매수 +{sp['gap'] * 100:.0f}%p 또는 되돌림" if sp["gap"] is not None else "되돌림만"
        return f"P2 사다리 −{xs}% 각 25% → {re}{tail}"
    return f"P3 종목별 고점 −{sp['x']:.1%}에서 {sp['q']:.0%} 장중 매도 → 재조정까지 현금{tail}"


def never_2026(sp):
    """2026 그림자 최대 낙폭(−22.9%)보다 깊은 재매수선(D ≥ 25%) 또는 손절선이 있는지 표시."""
    if sp["fam"] == "P3":
        return ""
    tr = tranches(sp)
    flag = []
    if any(t.d is not None and -t.d < SHADOW_MDD_2026 for t in tr):
        flag.append("D>22.9%")
    if any(t.x is not None and -t.x < SHADOW_MDD_2026 for t in tr):
        flag.append("X>22.9%")
    return ",".join(flag)


# ============================================================================ P1·P2: 그림자 조각 엔진
def sim_port(b: H.Base, tr: tuple, b_rec: float = RECOVER_B, slip: float = 0.0, lag: int = 1,
             close: np.ndarray | None = None):
    """조각별 통(주식=그림자 시가 단위, 현금). t 종가 판단 → t+lag 시가 체결.
    반환: r(index = ret.index, 시가 k → k+1, 미끄러짐 포함), e(시가 k 거래 뒤 노출), cycles, events."""
    o = np.asarray(b.open, float)
    c = np.asarray(b.close if close is None else close, float)
    n, K = len(o), len(tr)
    sh = np.array([t.w / o[0] for t in tr])
    cash = np.zeros(K)
    held = np.ones(K, bool)
    armed = np.array([t.x is not None for t in tr])
    sold_px = np.full(K, np.nan)
    sold_k = np.full(K, -1)
    pending = np.full(K, -1)                 # 체결 예정 시가 인덱스
    pend_kind = [None] * K
    peak = -np.inf
    V, E = np.zeros(n), np.zeros(n)
    cycles, events = [], []
    days = b.days
    for k in range(n):
        # 1) 시가 체결
        for i in range(K):
            if pending[i] != k:
                continue
            kind = pend_kind[i]
            pending[i], pend_kind[i] = -1, None
            if kind == "sell" and held[i]:
                px = o[k] * (1 - slip)
                cash[i], sh[i], held[i] = sh[i] * px, 0.0, False
                sold_px[i], sold_k[i] = px, k
                events.append({"day": days[k], "i": i, "type": "sell", "px": float(px)})
            elif kind in ("깊은 재매수", "되돌림") and not held[i]:
                px = o[k] * (1 + slip)
                cycles.append({"exit_day": days[sold_k[i]], "exit_px": float(sold_px[i]), "re_day": days[k], "re_px": float(px),
                               "gain": float(sold_px[i] / px - 1), "w": tr[i].w, "days_out": int(k - sold_k[i]), "kind": kind,
                               "open": False, "x": tr[i].x, "d": tr[i].d})
                sh[i], cash[i], held[i], armed[i] = cash[i] / px, 0.0, True, False
                events.append({"day": days[k], "i": i, "type": kind, "px": float(px)})
        hv = float(np.sum(sh * o[k]))
        V[k] = hv + float(np.sum(cash))
        E[k] = hv / V[k] if V[k] > 0 else 0.0
        # 2) 종가 판단(t = k)
        peak = max(peak, c[k])
        for i in range(K):
            t = tr[i]
            if t.x is None or pending[i] >= 0:
                continue
            lx = peak * (1 - t.x)
            ld = peak * (1 - t.d) if t.d is not None else -np.inf
            if held[i]:
                if not armed[i]:
                    if c[k] > lx:
                        armed[i] = True
                    continue
                if c[k] <= lx:
                    if c[k] <= ld:
                        armed[i] = False
                        events.append({"day": days[k], "i": i, "type": "skip_below_D", "px": float(c[k])})
                    elif k + lag < n:
                        pending[i], pend_kind[i] = k + lag, "sell"
            elif sold_k[i] <= k:
                if c[k] >= sold_px[i] * (1 + b_rec):
                    if k + lag < n:
                        pending[i], pend_kind[i] = k + lag, "되돌림"
                elif t.d is not None and ld < sold_px[i] and c[k] <= ld:
                    if k + lag < n:
                        pending[i], pend_kind[i] = k + lag, "깊은 재매수"
    for i in range(K):
        if not held[i]:
            cycles.append({"exit_day": days[sold_k[i]], "exit_px": float(sold_px[i]), "re_day": None, "re_px": float(c[-1]),
                           "gain": float(sold_px[i] / c[-1] - 1), "w": tr[i].w, "days_out": int(n - 1 - sold_k[i]),
                           "kind": "열림", "open": True, "x": tr[i].x, "d": tr[i].d})
    v = pd.Series(V, index=days)
    r = (v.shift(-1) / v - 1).dropna()
    return r, pd.Series(E, index=days), cycles, events


# ============================================================================ P3: 종목별 일부 손절
ENG: dict = {}


def engine(bn):
    if bn not in ENG:
        ENG[bn] = H.StockEngine(bn)
    return ENG[bn]


def run_partial(eng, q: float, x: float, slip: float = 0.0):
    """StockEngine과 같은 자료·재조정. 정책: 종목 고점(산 뒤 최고 종가, k−1까지) × (1−x)에 장중 q 매도, 주당 1회.
    반환 r(시가 k → k+1), e(시가 k 노출 / 기본 비중 합), events, cycles."""
    b, days = eng.b, eng.days
    w: dict = {}
    st: dict = {}
    done: set = set()                 # 이번 주 이미 일부 판 종목
    sold_last: dict = {}              # 이번 주 판 종목 → 판 비중(재조정 때 되사기 비용용)
    value = [1.0]
    expo = []
    events = []
    intraday: dict = {}
    for i, day in enumerate(days):
        k = eng.k_of[day]
        if i > 0:
            kp = eng.k_of[days[i - 1]]
            gross, new = 0.0, {}
            for key, wt in w.items():
                o0, o1 = eng.O[key][kp], eng.O[key][k]
                r = (o1 / o0 - 1) if np.isfinite(o0) and np.isfinite(o1) and o0 > 0 else 0.0
                if key in intraday:
                    f, fill = intraday[key]
                    rs = fill / o0 - 1
                    gross += wt * (f * rs + (1 - f) * r)
                    if f < 1 - 1e-12:
                        new[key] = wt * (1 - f) * (1 + r)
                else:
                    gross += wt * r
                    new[key] = wt * (1 + r)
            tot = 1 + gross
            for key in intraday:
                if intraday[key][0] >= 1 - 1e-12:
                    st.pop(key, None)
            w = {key: v / tot for key, v in new.items()}
            value.append(value[-1] * (1 + gross))
            for key in w:
                c, h = eng.C[key][kp], eng.H[key][kp]
                if np.isfinite(c):
                    st[key]["peak"] = max(st[key]["peak"], c)
        intraday = {}
        cost = 0.0
        if day in eng.reb:
            target = dict(b.weights[i])
            if slip:
                for key, sw in sold_last.items():
                    if key in target:
                        cost += slip * sw          # 판 몫을 재조정 때 되사는 비용(아래 같은 시가 재매도면 상쇄)
            topped = {key: sw for key, sw in sold_last.items() if key in target}
            sold_last = {}
            done = set()
            for key in target:
                if key not in w:
                    o = eng.O[key][k]
                    st[key] = {"entry": o, "peak": o, "entry_k": k}
            for key in list(st):
                if key not in target:
                    st.pop(key)
            w = target
        else:
            topped = {}
        bs = sum(b.weights[i].values())
        expo.append(1.0 if bs < 1e-9 else min(1.0, sum(w.values()) / bs))
        for key in list(w):
            if key in done:
                continue
            L = st[key]["peak"] * (1 - x)
            o, lo = eng.O[key][k], eng.L[key][k]
            gap = np.isfinite(o) and o <= L
            if gap:
                fill = o
            elif np.isfinite(lo) and lo <= L:
                fill = L
            else:
                continue
            wash = gap and key in topped           # 재조정 시가에 되산 몫을 같은 시가에 다시 판다 = 거래 없음
            if slip:
                if wash:
                    cost -= slip * min(topped[key], q * w[key])
                else:
                    fill = fill * (1 - slip)
            intraday[key] = (q, fill)
            done.add(key)
            sold_last[key] = q * w[key]
            events.append({"day": day, "key": key, "tk": eng.tk[key], "k": k, "q": q, "w": float(q * w[key]),
                           "px": float(fill), "level": float(L), "gap": bool(gap), "wash": bool(wash)})
        if cost:
            value[-1] *= (1 - cost)
    v = pd.Series(value, index=days)
    r = (v.shift(-1) / v - 1).dropna()
    return r, pd.Series(expo, index=days), events, p3_cycles(eng, events, slip)


def p3_cycles(eng, events, slip=0.0):
    """일부 매도 → 다음 재조정 시가(목표에 있으면 되삼, 없으면 기본도 그날 정리) / 자료 끝(마지막 종가)."""
    b = eng.b
    reb = sorted(d for d in eng.reb if d in set(b.days))
    idx = {d: i for i, d in enumerate(b.days)}
    out = []
    for e in events:
        d, key = e["day"], e["key"]
        nr = next((x for x in reb if x > d), None)
        if nr is not None:
            px = float(eng.O[key][eng.k_of[nr]])
            if key in b.weights[idx[nr]]:
                re_px, how = px * (1 + slip), "재조정 재매수"
            else:
                re_px, how = px, "목록 탈락(기본도 이날 정리)"
            re_day = nr
        else:
            re_day, re_px, how = None, float(eng.C[key][eng.k_of[b.days[-1]]]), "자료 끝"
        out.append({"exit_day": d, "tk": e["tk"], "exit_px": e["px"], "re_day": re_day, "re_px": float(re_px), "how": how,
                    "gain": float(e["px"] / re_px - 1), "w": e["w"], "kind": how, "open": re_day is None})
    return out


# ============================================================================ 지표
def wstats(r: pd.Series, e: pd.Series, b, cycles: list, lo: str, hi: str) -> dict:
    m = (r.index >= lo) & (r.index <= hi)
    x, br = r[m].to_numpy(), b.ret[m].to_numpy()
    cur = np.concatenate([[1.0], np.cumprod(1 + x)])
    ret, mdd = float(cur[-1] - 1), SL._mdd(cur)
    exc, e_eq = SL.same_mdd_excess(cur, br)
    bcur = np.concatenate([[1.0], np.cumprod(1 + br)])
    ea = float(e.reindex(r.index)[m].mean())
    fcur = np.concatenate([[1.0], np.cumprod(1 + ea * br)])
    cyc = [cy for cy in cycles if lo <= cy["exit_day"] <= hi]
    g = np.array([cy["gain"] for cy in cyc]) if cyc else np.array([])
    wt = np.array([cy["w"] for cy in cyc]) if cyc else np.array([])
    win, loss = g[g > 0], g[g <= 0]
    return {"수익": ret, "낙폭": mdd, "기본_수익": float(bcur[-1] - 1), "기본_낙폭": SL._mdd(bcur),
            "효율선초과": exc, "효율선_e": e_eq, "칼마": ret / abs(mdd) if mdd < -1e-9 else float("inf"),
            "평균노출": ea, "같은노출_수익": float(fcur[-1] - 1), "같은노출_낙폭": SL._mdd(fcur),
            "타이밍값": ret - float(fcur[-1] - 1),
            "주기": int(len(g)), "승률": float(len(win) / len(g)) if len(g) else float("nan"),
            "평균이익": float(win.mean()) if len(win) else float("nan"),
            "평균손실": float(loss.mean()) if len(loss) else float("nan"),
            "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
            "기대값": float(g.mean()) if len(g) else float("nan"),
            "기대값_가중": float((g * wt).sum() / wt.sum()) if len(g) and wt.sum() > 0 else float("nan"),
            "깊은재매수비율": float(np.mean([cy["kind"] == "깊은 재매수" for cy in cyc])) if cyc else float("nan")}


def run_one(sp, bn, slip=0.0):
    b = H.base(bn)
    if sp["fam"] == "P3":
        r, e, ev, cyc = run_partial(engine(bn), sp["q"], sp["x"], slip)
    else:
        r, e, cyc, ev = sim_port(b, tranches(sp), sp.get("b", RECOVER_B), slip, sp.get("lag", 1))
    return r, e, cyc, ev


def run_variant(sp, slip=0.0, keep=False):
    res = {"name": name(sp) + (f" (미끄러짐 {slip:.1%})" if slip else ""), "spec": sp, "2026에_안_닿는_선": never_2026(sp)}
    det = {}
    for bn in ("2026", "hold"):
        b = H.base(bn)
        r, e, cyc, ev = run_one(sp, bn, slip)
        res[bn] = {w: wstats(r, e, b, cyc, lo, hi) for w, (lo, hi) in WINS[bn].items()}
        res[bn]["빠짐횟수"] = int(((e.shift(1).fillna(1.0) >= 0.999) & (e < 0.999)).sum())
        res[bn]["매도사건"] = int(sum(1 for x in ev if x.get("type", "sell") == "sell"))
        if keep:
            det[bn] = {"r": r, "e": e, "cycles": cyc, "events": ev}
    return (res, det) if keep else res


def base_row():
    res = {"name": "기본(7종목 현행)", "spec": {"fam": "base"}, "2026에_안_닿는_선": ""}
    for bn in ("2026", "hold"):
        b = H.base(bn)
        e = pd.Series(1.0, index=b.days)
        res[bn] = {w: wstats(b.ret, e, b, [], lo, hi) for w, (lo, hi) in WINS[bn].items()}
        res[bn]["빠짐횟수"] = 0
        res[bn]["매도사건"] = 0
    return res


# ============================================================================ 점검
class _WholeStop(H.NoPolicy):
    def __init__(self, x): self.x = x
    def stop(self, key, k, st, eng): return st["peak"] * (1 - self.x)


def checks():
    out = {}
    for bn in ("2026", "hold"):
        b = H.base(bn)
        # 1) 손절 없는 조각 = 기본
        r, e, _, _ = sim_port(b, (Tr(1.0, None, None),))
        out[f"P1_손절없음_vs_기본_{bn}"] = float(np.max(np.abs(r.to_numpy() - b.ret.to_numpy())))
        # 2) 미끄러짐 0이면 통 계산 = overlay(노출)
        sp = {"fam": "P2", "xs": (0.05, 0.10, 0.15, 0.20), "gap": 0.05}
        r, e, _, _ = sim_port(b, tranches(sp))
        out[f"P2_통계산_vs_overlay_{bn}"] = float(np.max(np.abs(r.to_numpy() - H.overlay(b, e).to_numpy())))
        # 3) 1조각 = 같은 선 4조각
        r1, _, _, _ = sim_port(b, (Tr(1.0, 0.10, 0.20),))
        r4, _, _, _ = sim_port(b, tuple(Tr(0.25, 0.10, 0.20) for _ in range(4)))
        out[f"P1_1조각_vs_4조각_{bn}"] = float(np.max(np.abs(r1.to_numpy() - r4.to_numpy())))
        # 4) P3: 손절이 안 걸리면 기본 재현, q=100%는 StockEngine 전량 손절과 같음
        eng = engine(bn)
        r, _, ev, _ = run_partial(eng, 0.25, 0.999)
        out[f"P3_손절없음_vs_기본_{bn}"] = float(np.max(np.abs(r.to_numpy() - b.ret.to_numpy())))
        for x in (0.10, 0.20):
            rp, _, evp, _ = run_partial(eng, 1.0, x)
            rs, evs = eng.run(_WholeStop(x))
            out[f"P3_q100_vs_StockEngine_x{x:.2f}_{bn}"] = float(np.max(np.abs(rp.to_numpy() - rs.to_numpy())))
            out[f"P3_q100_사건수_{bn}_x{x:.2f}"] = [len(evp), sum(1 for z in evs if z["type"] == "stop")]
    # 5) 미래 교란(그림자): T 종가 이후 그림자·시가를 흔들어도 T+1 시가까지 노출·가치가 같아야
    b = H.base("2026")
    T = 100
    rng = np.random.default_rng(3)
    b2 = copy.copy(b)
    o2, c2 = np.asarray(b.open, float).copy(), np.asarray(b.close, float).copy()
    o2[T + 2:] *= 1 + rng.normal(0, 0.05, len(o2) - T - 2)        # T+1 시가는 T 종가 판단의 체결가라 그대로 둔다
    c2[T + 1:] *= 1 + rng.normal(0, 0.05, len(c2) - T - 1)
    b2 = H.Base("2026", b.days, o2, c2, b.weights, b.ticker, b.sector)
    worst_e, worst_v = 0.0, 0.0
    for sp in grid():
        if sp["fam"] == "P3":
            continue
        r1, e1, _, _ = sim_port(b, tranches(sp))
        r2, e2, _, _ = sim_port(b2, tranches(sp))
        worst_e = max(worst_e, float(np.max(np.abs(e1.to_numpy()[:T + 2] - e2.to_numpy()[:T + 2]))))
        v1, v2 = np.cumprod(1 + r1.to_numpy()), np.cumprod(1 + r2.to_numpy())
        worst_v = max(worst_v, float(np.max(np.abs(v1[:T + 1] - v2[:T + 1]))))   # v[j] = j+1 시가 가치, T+1 시가까지
    out["P1P2_미래교란_노출차(T+1시가까지)"] = worst_e
    out["P1P2_미래교란_가치차(T시가까지)"] = worst_v
    # 6) 미래 교란(P3): K 이후 종목 가격을 흔들어도 K 시가까지 가치 같음
    eng = engine("2026")
    e2 = copy.copy(eng)
    K = eng.k_of[b.days[100]]
    rng = np.random.default_rng(4)
    for attr in ("O", "C", "H", "L"):
        d = {}
        for key, arr in getattr(eng, attr).items():
            a = arr.copy()
            a[K + 1:] = a[K + 1:] * (1 + rng.normal(0, 0.05, len(a) - K - 1))
            d[key] = a
        setattr(e2, attr, d)
    for key in eng.keys:
        e2.H[key][K + 1:] = np.fmax.reduce([e2.H[key][K + 1:], e2.O[key][K + 1:], e2.C[key][K + 1:]])
        e2.L[key][K + 1:] = np.fmin.reduce([e2.L[key][K + 1:], e2.O[key][K + 1:], e2.C[key][K + 1:]])
    worst = 0.0
    for q in (0.25, 1.0):
        for x in P3_X:
            r1 = run_partial(eng, q, x)[0]
            r2 = run_partial(e2, q, x)[0]
            v1, v2 = np.cumprod(1 + r1.to_numpy()), np.cumprod(1 + r2.to_numpy())
            worst = max(worst, float(np.max(np.abs(v1[:99] - v2[:99]))))
    out["P3_미래교란_가치차"] = worst
    return out


# ============================================================================ 고르기·보고
def pick(rows, key, need=None):
    ok = [r for r in rows if (need is None or need(r)) and np.isfinite(key(r))]
    if not ok:
        return None
    best = max(key(r) for r in ok)
    return next(r for r in ok if key(r) == best)


def neighbors(sp):
    out = []
    if sp["fam"] == "P1":
        out += [dict(sp, x=round(sp["x"] + d, 4)) for d in (-NB_X, NB_X) if sp["x"] + d > 0.005]
        out += [dict(sp, b=bb) for bb in NB_B]
        out.append(dict(sp, lag=2))
    elif sp["fam"] == "P2":
        out += [dict(sp, xs=tuple(round(x + d, 4) for x in sp["xs"])) for d in (-NB_X, NB_X)
                if min(sp["xs"]) + d > 0.005]
        out += [dict(sp, b=bb) for bb in NB_B]
        out.append(dict(sp, lag=2))
    else:
        out += [dict(sp, x=round(sp["x"] + d, 4)) for d in (-NB_X_P3, NB_X_P3)]
    return out


def brief(r):
    a, h = r["2026"], r["hold"]
    return {"name": r["name"], "2026": {k: a[k]["수익"] for k in W26}, "2026_낙폭": a["전체"]["낙폭"],
            "앞_효율선초과": a["앞"]["효율선초과"], "뒤_효율선초과": a["뒤"]["효율선초과"], "2026_효율선초과": a["전체"]["효율선초과"],
            "2026_타이밍값": a["전체"]["타이밍값"], "2026_평균노출": a["전체"]["평균노출"],
            "2025": h["2025"]["수익"], "2025_낙폭": h["2025"]["낙폭"], "2025_효율선초과": h["2025"]["효율선초과"],
            "2024": h["2024"]["수익"], "2024_효율선초과": h["2024"]["효율선초과"]}


def pct(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def share(sub, f):
    return (sum(1 for r in sub if f(r)), len(sub))


def main():
    chk = checks()
    print("점검", json.dumps(chk, ensure_ascii=False))
    for k, v in chk.items():
        if isinstance(v, float):
            lim = 1e-6 if ("기본" in k or "StockEngine" in k) else 1e-12
            assert v < lim, (k, v)

    br = base_row()
    specs = grid()
    counts = {f: sum(s["fam"] == f for s in specs) for f in ("P1", "P2", "P3")}
    print("격자", len(specs), counts)
    rows = []
    for sp in specs:
        r = run_variant(sp)
        rows.append(r)
        a, h = r["2026"], r["hold"]
        print(f"{r['name'][:62]:<62} 26 {a['전체']['수익']:+.0%}/{a['전체']['낙폭']:.1%} 앞초과 {a['앞']['효율선초과']:+.1%} "
              f"뒤 {a['뒤']['수익']:+.1%} 초과 {a['뒤']['효율선초과']:+.1%} AI {a['AI쇼크']['수익']:+.1%} | "
              f"25 {h['2025']['수익']:+.1%}/{h['2025']['낙폭']:.1%} 초과 {h['2025']['효율선초과']:+.1%} | 노출 {a['전체']['평균노출']:.2f} "
              f"타이밍 {a['전체']['타이밍값']:+.1%} | 앞주기 {a['앞']['주기']} 승 {a['앞']['승률']:.0%} 기대 {a['앞']['기대값']:+.1%}")
    cmp_rows = []
    for sp in compare_specs():
        r = run_variant(sp)
        cmp_rows.append(r)
        a, h = r["2026"], r["hold"]
        print(f"[비교] {r['name'][:56]:<56} 26 {a['전체']['수익']:+.0%}/{a['전체']['낙폭']:.1%} 앞초과 {a['앞']['효율선초과']:+.1%} "
              f"뒤 {a['뒤']['수익']:+.1%} AI {a['AI쇼크']['수익']:+.1%} | 25 {h['2025']['수익']:+.1%} 초과 {h['2025']['효율선초과']:+.1%} "
              f"| 타이밍 {a['전체']['타이밍값']:+.1%}")

    F = lambda r: r["2026"]["앞"]
    picks = {"G1": pick(rows, lambda r: F(r)["효율선초과"]),
             "G2": pick(rows, lambda r: F(r)["기대값"], lambda r: F(r)["주기"] >= MIN_CYCLES_G2),
             "G3": pick(rows, lambda r: F(r)["칼마"])}
    fam_g1 = {f: pick([r for r in rows if r["spec"]["fam"] == f], lambda r: F(r)["효율선초과"]) for f in ("P1", "P2", "P3")}
    n_g2 = sum(F(r)["주기"] >= MIN_CYCLES_G2 for r in rows)

    posF = [r for r in rows if F(r)["효율선초과"] > 0]
    ok_back = lambda r: r["2026"]["뒤"]["효율선초과"] > 0
    ok_25 = lambda r: r["hold"]["2025"]["효율선초과"] > 0
    info = {"앞>0 개수": len(posF), "전체 개수": len(rows),
            "뒤>0 | 앞>0": share(posF, ok_back), "뒤>0 무조건": share(rows, ok_back),
            "2025>0 | 앞>0": share(posF, ok_25), "2025>0 무조건": share(rows, ok_25),
            "둘다>0 | 앞>0": share(posF, lambda r: ok_back(r) and ok_25(r)),
            "둘다>0 무조건": share(rows, lambda r: ok_back(r) and ok_25(r)),
            "2026전체 기본보다 수익↑": share(rows, lambda r: r["2026"]["전체"]["수익"] > br["2026"]["전체"]["수익"]),
            "AI쇼크 기본보다 덜 잃음": share(rows, lambda r: r["2026"]["AI쇼크"]["수익"] > br["2026"]["AI쇼크"]["수익"]),
            "2026 타이밍값>0": share(rows, lambda r: r["2026"]["전체"]["타이밍값"] > 0),
            "2026 효율선초과>0": share(rows, lambda r: r["2026"]["전체"]["효율선초과"] > 0)}
    info_fam = {}
    for f in ("P1", "P2", "P3"):
        sub = [r for r in rows if r["spec"]["fam"] == f]
        pf = [r for r in sub if F(r)["효율선초과"] > 0]
        info_fam[f] = {"개수": len(sub), "앞>0": len(pf),
                       "뒤>0 | 앞>0": share(pf, ok_back), "뒤>0 무조건": share(sub, ok_back),
                       "2025>0 | 앞>0": share(pf, ok_25), "2025>0 무조건": share(sub, ok_25),
                       "2026전체 기본보다 수익↑": share(sub, lambda r: r["2026"]["전체"]["수익"] > br["2026"]["전체"]["수익"]),
                       "AI쇼크 기본보다 덜 잃음": share(sub, lambda r: r["2026"]["AI쇼크"]["수익"] > br["2026"]["AI쇼크"]["수익"]),
                       "2026 타이밍값>0": share(sub, lambda r: r["2026"]["전체"]["타이밍값"] > 0),
                       "뒤 타이밍값>0": share(sub, lambda r: r["2026"]["뒤"]["타이밍값"] > 0),
                       "2025 타이밍값>0": share(sub, lambda r: r["hold"]["2025"]["타이밍값"] > 0)}

    # 고른 변형: 미끄러짐·이웃·행동
    pick_detail = {}
    done = {}
    for g, r in list(picks.items()) + [(f"{f}최선(G1)", v) for f, v in fam_g1.items()]:
        if r is None:
            pick_detail[g] = None
            continue
        if r["name"] in done:
            pick_detail[g] = {"same_as": done[r["name"]]}
            continue
        done[r["name"]] = g
        sp = r["spec"]
        rs = run_variant(sp, slip=SLIP)
        nb = [run_variant(x) for x in neighbors(sp)]
        _, det = run_variant(sp, keep=True)
        e26 = det["2026"]["e"]
        cyc26 = det["2026"]["cycles"]
        pick_detail[g] = {"row": r, "slip": rs, "neighbors": [brief(x) for x in nb],
                          "trace_2026_6_15_9_30": H.trace(e26, "2026-06-15", "2026-09-30"),
                          "trace_2026_front": H.trace(e26, "2026-01-02", "2026-05-29"),
                          "cycles_2026": cyc26, "cycles_2025": [c for c in det["hold"]["cycles"]
                                                               if "2025-01-01" <= c["exit_day"] <= "2025-12-31"]}

    # 같은 선 25·50·100% 비교(P1 전 칸, P3)
    allr = rows + cmp_rows
    def find(fam, **kw):
        for r in allr:
            s = r["spec"]
            if s["fam"] == fam and all(s.get(k) == v for k, v in kw.items()):
                return r
        return None
    pvf = []
    for x in P1_X:
        for gp in P1_DGAP:
            row = {"fam": "P1", "x": x, "gap": gp}
            for q in (0.25, 0.50, 1.0):
                rr = find("P1", q=q, x=x, gap=gp)
                row[f"{q:.2f}"] = brief(rr)
                row[f"{q:.2f}_타이밍"] = {w: rr["2026"][w]["타이밍값"] for w in ("전체", "앞", "뒤", "AI쇼크")}
                row[f"{q:.2f}_타이밍"]["2025"] = rr["hold"]["2025"]["타이밍값"]
                row[f"{q:.2f}_노출"] = rr["2026"]["전체"]["평균노출"]
            pvf.append(row)
    for x in P3_X:
        row = {"fam": "P3", "x": x}
        for q in (0.25, 0.50, 1.0):
            rr = find("P3", q=q, x=x)
            row[f"{q:.2f}"] = brief(rr)
            row[f"{q:.2f}_타이밍"] = {w: rr["2026"][w]["타이밍값"] for w in ("전체", "앞", "뒤", "AI쇼크")}
            row[f"{q:.2f}_타이밍"]["2025"] = rr["hold"]["2025"]["타이밍값"]
            row[f"{q:.2f}_노출"] = rr["2026"]["전체"]["평균노출"]
        pvf.append(row)
    # 사후(뒤·2025를 본 뒤 추가, 후보·고르기 아님): 뒤 효율선초과 > 0인 변형의 하루 늦춤·미끄러짐·7월 보호의 정체
    back_win = []
    for r in rows:
        if r["2026"]["뒤"]["효율선초과"] > 0:
            sp = r["spec"]
            lagr = run_variant(dict(sp, lag=2)) if sp["fam"] != "P3" else None
            slr = run_variant(sp, slip=SLIP)
            back_win.append({"row": brief(r), "lag": brief(lagr) if lagr else None, "slip": brief(slr),
                             "AI쇼크_같은노출": r["2026"]["AI쇼크"]["같은노출_수익"], "AI쇼크_타이밍": r["2026"]["AI쇼크"]["타이밍값"]})
            print("[사후 뒤 승자]", r["name"], "하루 늦춤", brief(lagr)["2026"] if lagr else None, "미끄러짐", brief(slr)["2026"])
    # 사다리 = 단일선 전량 4개의 평균인가(사후 설명용): 사다리 노출 vs 같은 선 1조각 전량 노출 평균
    ladder_blend = []
    for lad in P2_LADDERS:
        for gp in P2_DGAP:
            sp = {"fam": "P2", "xs": lad, "gap": gp}
            b = H.base("2026")
            _, e_l, _, _ = sim_port(b, tranches(sp))
            es = [sim_port(b, (Tr(1.0, x, dval(x, gp)),))[1] for x in lad]
            avg = sum(es) / 4
            ladder_blend.append({"xs": lad, "gap": gp, "노출차_최대": float((e_l - avg).abs().max())})

    # 출력
    print("\n기본", brief(br))
    for g, r in picks.items():
        print(f"[{g}]", brief(r) if r else "없음")
    for f, r in fam_g1.items():
        print(f"[{f} G1]", brief(r) if r else "없음")
    print("G2 자격(앞 주기≥30)", n_g2)
    print("정보", info)
    for f, v in info_fam.items():
        print(" ", f, v)
    for g, d in pick_detail.items():
        if d and "row" in d:
            print(f"\n== {g} {d['row']['name']}")
            print("  미끄러짐", brief(d["slip"]))
            for x in d["neighbors"]:
                print("  이웃", x)
            print("  노출 변화 6/15~9/30", d["trace_2026_6_15_9_30"][:30])
    print("사다리≈단일선 평균 노출 최대 차", [round(x["노출차_최대"], 4) for x in ladder_blend])

    payload = {"id": ID, "grid_size": len(specs), "grid_counts": counts, "compare_size": len(cmp_rows), "checks": chk,
               "base": br, "picks": {g: (r["name"] if r else None) for g, r in picks.items()},
               "fam_g1": {f: (r["name"] if r else None) for f, r in fam_g1.items()}, "g2_eligible": n_g2,
               "info": info, "info_fam": info_fam, "pick_detail": pick_detail, "rows": rows, "compare_rows": cmp_rows,
               "partial_vs_full": pvf, "ladder_blend_posthoc": ladder_blend, "back_winners_posthoc": back_win,
               "params": {"P1_Q": P1_Q, "P1_X": P1_X, "P1_DGAP": P1_DGAP, "P2_LADDERS": P2_LADDERS, "P2_DGAP": P2_DGAP,
                          "P3_Q": P3_Q, "P3_X": P3_X, "RECOVER_B": RECOVER_B, "SLIP": SLIP, "MIN_CYCLES_G2": MIN_CYCLES_G2,
                          "NB_X": NB_X, "NB_X_P3": NB_X_P3, "NB_B": NB_B}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{ID}.json").write_text(json.dumps(clean(payload), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_md(payload)
    print("저장", RESULTS / f"{ID}.json", RESULTS / f"{ID}.md")


# ============================================================================ md
def write_md(P):
    br, rows, cmpr = P["base"], P["rows"], P["compare_rows"]
    L = []
    A = L.append
    b26, b25 = br["2026"], br["hold"]
    A("# 3단계 PORT3: 7종목 포트폴리오에 '선 이탈 때 일부만 팔기' 얹기 (2026-10-10)")
    A("")
    A("> 코드 `code/p3_PORT3.py`, 전 변형 `results/p3_PORT3.json`. USD, 시가→시가, 수수료 0, 현금 이자 0. "
      "그림자 = 7종목 모델의 그날 종가 평가(t 종가에 앎 → t+1 시가 체결).")
    A("")
    A("## 결론")
    A("")
    for i, s in enumerate(CONCLUSION, 1):
        A(f"{i}. {s}")
    if not CONCLUSION:
        A("(작성 전)")
    A("")
    gc = P["grid_counts"]
    A(f"## 격자 (사전 고정, 합계 {P['grid_size']}) + 비교용 전량 매도 {P['compare_size']}")
    A("")
    A(f"- P1 포트폴리오 낙폭 분할 {gc['P1']} = 파는 몫 q 2(25%, 50%) × 손절선 X 6(그림자 고점 대비 −5·−8·−10·−12·−15·−20%) × "
      f"재매수 3(X+5%p, X+10%p, 없음 = 되돌림만). 되돌림 = 그림자 종가 ≥ 판 값 × 1.03 → 다음 시가.")
    A(f"- P2 사다리 {gc['P2']} = 25%씩 4조각, 손절선 4묶음(−5/10/15/20, −8/12/16/20, −10/15/20/25, −5/8/12/15) × 재매수 3(각 X+5%p, X+10%p, 되돌림만).")
    A(f"- P3 종목별 일부 손절 {gc['P3']} = q 2(25%, 50%) × X 3(종목 고점 대비 −10·−15·−20%), 장중, 그 주 재조정까지 현금.")
    A(f"- 비교용(사전 선언, 고르기 제외) {P['compare_size']} = 같은 선 전량(100%) 매도: P1 18칸, P3 3칸.")
    A("- 고른 변형마다 사전 선언 이웃(X ±2%p[P3 ±2.5%p], 되돌림 여유 2%·5%, 체결 하루 늦춤)과 미끄러짐 0.5%를 더 쟀다(격자 수 제외).")
    A(f"- **2026 그림자 최대 낙폭은 −22.9%** 다. 그래서 재매수선 D ≥ 25%(−15%+10%p, −20%+5·10%p, 사다리 −25% 조각과 그 재매수)는 "
      "2026에 한 번도 발동하지 않는다. 표에서 †로 표시했다.")
    A("")
    A("## 변수 시점")
    A("")
    A("| 변수 | 언제 아는가 | 체결 |")
    A("|---|---|---|")
    A("| 그림자 종가(7종목 모델 종가 평가)·추적 고점·낙폭 | t일 미국 종가 | t+1 시가(P1·P2 매도·재매수 모두) |")
    A("| 판 값(되돌림 기준) | 판 시가에 확정 | 그 뒤 종가가 1.03배 이상이면 다음 시가 |")
    A("| 종목 고점(산 뒤 최고 종가) | k−1일 종가까지 | k일 장중 스탑(시가가 선 아래면 시가) |")
    A("| 주간 목표(7종목 규칙) | 재조정일 시가(기본 경로와 같음) | 그날 시가 |")
    A("")

    def cell(r, f):
        return f(r) + ("†" if r["2026에_안_닿는_선"] else "")

    def find(fam, **kw):
        for r in rows + cmpr:
            s = r["spec"]
            if s["fam"] == fam and all(s.get(k) == v for k, v in kw.items()):
                return r
        return None

    A("## 핵심 격자: 손절 % × 재매수 % (P1, 한 선에서 일부 매도)")
    A("")
    A(f"기본(그대로 보유): 2026 {pct(b26['전체']['수익'])}/{pct(b26['전체']['낙폭'])}, 뒤 {pct(b26['뒤']['수익'])}, "
      f"7월 쇼크 {pct(b26['AI쇼크']['수익'])}, 2025 {pct(b25['2025']['수익'])}/{pct(b25['2025']['낙폭'])}. "
      "칸의 재매수 '+5%p'는 손절선보다 5%p 더 깊은 곳(예: −10% 손절 → −15% 재매수). †는 2026에 그 재매수선이 한 번도 안 닿은 칸.")
    A("")
    for title, f in (("2026 전체 수익", lambda r: pct(r["2026"]["전체"]["수익"])),
                     ("2026 효율선초과(같은 낙폭 고정 노출 대비)", lambda r: pct(r["2026"]["전체"]["효율선초과"])),
                     ("7월 쇼크(6/22~7/29) 수익", lambda r: pct(r["2026"]["AI쇼크"]["수익"])),
                     ("뒤(6/1~10/9) 수익", lambda r: pct(r["2026"]["뒤"]["수익"])),
                     ("2025 수익(보조)", lambda r: pct(r["hold"]["2025"]["수익"]))):
        A(f"**{title}**")
        A("")
        A("| 손절선 \\ 재매수 | 25%·+5%p | 25%·+10%p | 25%·되돌림만 | 50%·+5%p | 50%·+10%p | 50%·되돌림만 | 100%·되돌림만(비교) |")
        A("|---|---|---|---|---|---|---|---|")
        for x in P1_X:
            cells = []
            for q in (0.25, 0.50):
                for gp in P1_DGAP:
                    cells.append(cell(find("P1", q=q, x=x, gap=gp), f))
            cells.append(cell(find("P1", q=1.0, x=x, gap=None), f))
            A(f"| −{x:.0%} | " + " | ".join(cells) + " |")
        A("")
    A("## 사다리 (P2, 25%씩 4선)")
    A("")
    A("| 사다리 | 재매수 | 2026 수익/낙폭 | 앞 효율선초과 | 뒤 수익 · 효율선초과 | 7월 쇼크 | 2025 수익 · 효율선초과 | 평균 노출 | 타이밍 값(2026) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        s = r["spec"]
        if s["fam"] != "P2":
            continue
        a, h = r["2026"], r["hold"]
        lad = "/".join(f"−{x * 100:.0f}" for x in s["xs"])
        re = f"+{s['gap'] * 100:.0f}%p" if s["gap"] is not None else "되돌림만"
        A(f"| {lad}{'†' if r['2026에_안_닿는_선'] else ''} | {re} | {pct(a['전체']['수익'])}/{pct(a['전체']['낙폭'])} | "
          f"{pct(a['앞']['효율선초과'])} | {pct(a['뒤']['수익'])} · {pct(a['뒤']['효율선초과'])} | {pct(a['AI쇼크']['수익'])} | "
          f"{pct(h['2025']['수익'])} · {pct(h['2025']['효율선초과'])} | {a['전체']['평균노출']:.2f} | {pct(a['전체']['타이밍값'])} |")
    A("")
    A("## 종목별 일부 손절 (P3, 종목 고점 대비, 장중)")
    A("")
    A("| 파는 몫 \\ 손절선 | 2026 수익/낙폭 | 앞 효율선초과 | 뒤 수익 · 효율선초과 | 7월 쇼크 | 2025 수익 · 효율선초과 | 평균 노출 | 타이밍 값(2026) |")
    A("|---|---|---|---|---|---|---|---|")
    for q in (0.25, 0.50, 1.0):
        for x in P3_X:
            r = find("P3", q=q, x=x)
            a, h = r["2026"], r["hold"]
            A(f"| {q:.0%}{'(비교)' if q == 1.0 else ''} · −{x:.0%} | {pct(a['전체']['수익'])}/{pct(a['전체']['낙폭'])} | "
              f"{pct(a['앞']['효율선초과'])} | {pct(a['뒤']['수익'])} · {pct(a['뒤']['효율선초과'])} | {pct(a['AI쇼크']['수익'])} | "
              f"{pct(h['2025']['수익'])} · {pct(h['2025']['효율선초과'])} | {a['전체']['평균노출']:.2f} | {pct(a['전체']['타이밍값'])} |")
    A("")
    A("## 같은 선에서 25%·50%·100% 팔기: '판 것'의 값(타이밍 값 = 수익 − 같은 평균 노출 고정 보유 수익)")
    A("")
    A("| 선 | 재매수 | 평균 노출 25/50/100% | 2026 타이밍 값 25/50/100% | 뒤 25/50/100% | 7월 쇼크 25/50/100% | 2025 25/50/100% |")
    A("|---|---|---|---|---|---|---|")
    for row in P["partial_vs_full"]:
        if row["fam"] == "P1":
            lab = f"P1 그림자 −{row['x']:.0%}"
            re = f"+{row['gap'] * 100:.0f}%p" if row["gap"] is not None else "되돌림만"
        else:
            lab, re = f"P3 종목 −{row['x']:.0%}", "재조정"
        qs = ("0.25", "0.50", "1.00")
        A(f"| {lab} | {re} | " + "/".join(f"{row[q + '_노출']:.2f}" for q in qs) + " | "
          + " | ".join("/".join(pct(row[q + "_타이밍"][w]) for q in qs) for w in ("전체", "뒤", "AI쇼크", "2025")) + " |")
    A("")
    A("## 고르기 (앞 1/2~5/29로 고르고 뒤 6/1~10/9·2025로 확인)")
    A("")
    A("| 고르기 | 변형 | 2026 수익/낙폭 | 앞 수익 · 효율선초과 | 뒤 수익/낙폭 · 효율선초과 | 7월 쇼크 | 2025 수익/낙폭 · 효율선초과 | 평균 노출 · 타이밍 값 |")
    A("|---|---|---|---|---|---|---|---|")

    def line(lab, r):
        a, h = r["2026"], r["hold"]
        return (f"| {lab} | {r['name']} | {pct(a['전체']['수익'])}/{pct(a['전체']['낙폭'])} | "
                f"{pct(a['앞']['수익'])} · {pct(a['앞']['효율선초과'])} | {pct(a['뒤']['수익'])}/{pct(a['뒤']['낙폭'])} · "
                f"{pct(a['뒤']['효율선초과'])} | {pct(a['AI쇼크']['수익'])} | {pct(h['2025']['수익'])}/{pct(h['2025']['낙폭'])} · "
                f"{pct(h['2025']['효율선초과'])} | {a['전체']['평균노출']:.2f} · {pct(a['전체']['타이밍값'])} |")
    A(line("기본", br))
    byname = {r["name"]: r for r in rows}
    for g in ("G1", "G2", "G3"):
        n = P["picks"][g]
        A(line(g, byname[n]) if n else f"| {g} | 해당 없음(앞 주기 ≥ {MIN_CYCLES_G2}인 변형 {P['g2_eligible']}개) | | | | | | |")
    for f in ("P1", "P2", "P3"):
        n = P["fam_g1"][f]
        if n:
            A(line(f"{f} 안 G1", byname[n]))
    A("")
    A("고르기에 정보가 있나(앞 효율선초과 > 0 집합 vs 무조건):")
    A("")
    A("| 묶음 | 앞>0 개수 | 뒤>0 (앞>0 중) | 뒤>0 (무조건) | 2025>0 (앞>0 중) | 2025>0 (무조건) | 2026 타이밍 값>0 |")
    A("|---|---|---|---|---|---|---|")
    fr = lambda t: f"{t[0]}/{t[1]}" + (f" ({t[0] / t[1]:.0%})" if t[1] else "")
    i = P["info"]
    A(f"| 전체 {i['전체 개수']} | {i['앞>0 개수']} | {fr(i['뒤>0 | 앞>0'])} | {fr(i['뒤>0 무조건'])} | {fr(i['2025>0 | 앞>0'])} | "
      f"{fr(i['2025>0 무조건'])} | {fr(i['2026 타이밍값>0'])} |")
    for f, v in P["info_fam"].items():
        A(f"| {f} {v['개수']} | {v['앞>0']} | {fr(v['뒤>0 | 앞>0'])} | {fr(v['뒤>0 무조건'])} | {fr(v['2025>0 | 앞>0'])} | "
          f"{fr(v['2025>0 무조건'])} | {fr(v['2026 타이밍값>0'])} |")
    A("")
    A("## 손절 주기 승률·손익비 (이익 = 판 값/되산 값 − 1, 판 날 기준 창)")
    A("")
    A("| 변형 | 창 | 주기 | 승률 | 평균이익 | 평균손실 | 손익비 | 기대값(조각 수) | 기대값(비중 가중) | 깊은 재매수 비율 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    shown = []
    for g in ("G1", "G2", "G3"):
        if P["picks"][g] and P["picks"][g] not in shown:
            shown.append(P["picks"][g])
    for f in ("P1", "P2", "P3"):
        if P["fam_g1"][f] and P["fam_g1"][f] not in shown:
            shown.append(P["fam_g1"][f])
    for n in shown:
        r = byname[n]
        for bn, w in (("2026", "앞"), ("2026", "뒤"), ("hold", "2025")):
            s = r[bn][w]
            A(f"| {n} | {w if bn == '2026' else '2025'} | {s['주기']} | {pct(s['승률'], 0, False)} | {pct(s['평균이익'])} | "
              f"{pct(s['평균손실'])} | {num(s['손익비'])} | {pct(s['기대값'], 2)} | {pct(s['기대값_가중'], 2)} | {pct(s['깊은재매수비율'], 0, False)} |")
    A("")
    A("묶음별 전 변형 주기 합산(변형 간 중복 사건 포함, 분포 감 잡기용):")
    A("")
    A("| 묶음 | 창 | 주기 | 승률 | 손익비(중앙) | 기대값 | 깊은 재매수 비율 |")
    A("|---|---|---|---|---|---|---|")
    for f in ("P1", "P2", "P3"):
        sub = [r for r in rows if r["spec"]["fam"] == f]
        for bn, w in (("2026", "앞"), ("2026", "뒤"), ("hold", "2025")):
            n = sum(r[bn][w]["주기"] for r in sub)
            if not n:
                A(f"| {f} | {w if bn == '2026' else '2025'} | 0 | - | - | - | - |")
                continue
            wins = sum(r[bn][w]["승률"] * r[bn][w]["주기"] for r in sub if r[bn][w]["주기"])
            ev = sum(r[bn][w]["기대값"] * r[bn][w]["주기"] for r in sub if r[bn][w]["주기"]) / n
            dp = sum((r[bn][w]["깊은재매수비율"] or 0) * r[bn][w]["주기"] for r in sub if r[bn][w]["주기"]) / n
            pw = [r[bn][w]["손익비"] for r in sub if r[bn][w]["주기"] and r[bn][w]["손익비"] is not None
                  and np.isfinite(r[bn][w]["손익비"])]
            A(f"| {f} | {w if bn == '2026' else '2025'} | {n} | {wins / n:.0%} | {num(float(np.median(pw))) if pw else '-'} | "
              f"{pct(ev, 2)} | {dp:.0%} |")
    A("")
    A("## 대표 후보(G1) 점검")
    A("")
    g1 = P["pick_detail"].get("G1")
    if g1 and "row" in g1:
        s = g1["slip"]
        A(f"- 대표: {g1['row']['name']}")
        A(f"- 미끄러짐 0.5%: 2026 {pct(s['2026']['전체']['수익'])}/{pct(s['2026']['전체']['낙폭'])}, 앞 효율선초과 {pct(s['2026']['앞']['효율선초과'])}, "
          f"뒤 {pct(s['2026']['뒤']['수익'])} (효율선초과 {pct(s['2026']['뒤']['효율선초과'])}), 2025 {pct(s['hold']['2025']['수익'])} "
          f"(효율선초과 {pct(s['hold']['2025']['효율선초과'])}).")
        A("- 이웃:")
        for x in g1["neighbors"]:
            A(f"  - {x['name']}: 2026 {pct(x['2026']['전체'])}, 앞 효율선초과 {pct(x['앞_효율선초과'])}, 뒤 {pct(x['2026']['뒤'])} "
              f"(효율선초과 {pct(x['뒤_효율선초과'])}), 7월 쇼크 {pct(x['2026']['AI쇼크'])}, 2025 {pct(x['2025'])} (효율선초과 {pct(x['2025_효율선초과'])})")
        if g1["trace_2026_6_15_9_30"]:
            A(f"- 6/15~9/30 노출 변화: {', '.join(f'{d} {v:g}' for d, v in g1['trace_2026_6_15_9_30'][:24])}")
        cy = [c for c in g1["cycles_2026"]]
        if cy:
            A("- 2026 손절 주기: " + "; ".join(
                f"{c['exit_day']}{(' ' + c['tk']) if 'tk' in c else ''} → {c['re_day'] or '열림'} ({pct(c['gain'])}, {c['kind']})"
                for c in cy[:20]) + (f" … 외 {len(cy) - 20}개" if len(cy) > 20 else ""))
    A("")
    A("## 사후 점검: 뒤 구간에서 효율선을 이긴 변형 (뒤·2025를 본 뒤 추가, 후보 아님)")
    A("")
    A("| 변형 | 2026 · 뒤 · 2025 | 뒤·2025 효율선초과 | 7월 쇼크 (같은 평균 노출 고정) | 체결 하루 늦춤: 2026 · 뒤 · 2025 | 미끄러짐 0.5%: 2026 · 뒤 · 2025 |")
    A("|---|---|---|---|---|---|")
    for x in P.get("back_winners_posthoc", []):
        r, lg, sl = x["row"], x["lag"], x["slip"]
        lgs = f"{pct(lg['2026']['전체'])} · {pct(lg['2026']['뒤'])} · {pct(lg['2025'])}" if lg else "-"
        A(f"| {r['name']} | {pct(r['2026']['전체'])} · {pct(r['2026']['뒤'])} · {pct(r['2025'])} | "
          f"{pct(r['뒤_효율선초과'])} · {pct(r['2025_효율선초과'])} | {pct(r['2026']['AI쇼크'])} ({pct(x['AI쇼크_같은노출'])}) | {lgs} | "
          f"{pct(sl['2026']['전체'])} · {pct(sl['2026']['뒤'])} · {pct(sl['2025'])} |")
    A("")
    A("## 보조·맥락")
    A("")
    for s in CONTEXT:
        A(f"- {s}")
    if not CONTEXT:
        A("(작성 전)")
    A("")
    A("## 선택편향·한계")
    A("")
    for s in LIMITS:
        A(f"- {s}")
    if not LIMITS:
        A("(작성 전)")
    A("")
    A("## 점검")
    A("")
    for k, v in P["checks"].items():
        A(f"- {k}: {v}")
    (RESULTS / f"{ID}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
