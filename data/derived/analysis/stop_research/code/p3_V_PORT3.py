"""p3_V_PORT3: 3단계 갈래 PORT3 대표 후보 반박 검증 (2026-10-11, claude).

후보(G1=G3): 7종목 그림자(모델 종가 평가)가 2026-01-02 이후 최고 종가 대비 −8% 이하로 마감 → 다음 시가에 보유의 50% 매도
  (나머지 50%는 늘 보유). 판 몫은 (1) 그림자 종가 ≤ 고점×0.82이고 그 선 < 판 값이면 다음 시가 깊은 재매수,
  (2) 그림자 종가 ≥ 판 값×1.03이면 다음 시가 되돌림 재매수. 되산 몫은 그림자 종가가 −8% 선 위로 올라와야 다시 무장.
  처음 −8% 이탈이 이미 −18% 아래면 팔지 않는다. 신호 t 종가, 체결 t+1 시가.

독립 재계산: harness·stock_lab·tranche_lab·p3_PORT3을 import하지 않는다.
  base_*.json의 일별 비중(weights)·종목(ticker)과 ohlc.parquet 시가·종가로 7종목 시가 경로 vo와 종가 평가 vc를 다시 만든다.
  규칙은 '반나절 경로'(시가·종가를 번갈아 놓은 값 H[2k]=vo[k], H[2k+1]=vc[k]) 위의 새 상태 기계로 짠다.
  체결 지연 L은 반나절 단위다: L=1 다음 시가(후보), L=2 다음 종가, L=3 이틀째 시가(= PORT3 '하루 늦춤'), L=0 같은 종가(불가능, 진단).
장세 대리(7종목 경로가 없는 2018·2022): p2_long.parquet(배당 수정)로 만든 SMH·QQQ·AI 동일가중·AI 모멘텀 7종목. QUBT는 뺀다.
실행: cd code && PYTHONIOENCODING=utf-8 py p3_V_PORT3.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ID = "p3_V_PORT3"

# ============================================================================ 사전 고정(결과를 본 뒤 바꾸지 않는다)
CAND = {"q": 0.50, "x": 0.08, "d": 0.18, "b": 0.03, "L": 1, "rearm": "line", "skip": True}
SLIP = 0.005                          # 팔 때 −0.5%, 살 때 +0.5%
# 이웃(각 모수 앞뒤 2칸): 손절선 X × 재매수선 D 5×5(가운데 = 후보) + 파는 몫 q 4 + 되돌림 여유 b 4 = 32
NB_X = [0.04, 0.06, 0.08, 0.10, 0.12]
NB_D = [0.14, 0.16, 0.18, 0.20, 0.22]
NB_Q = [0.30, 0.40, 0.50, 0.60, 0.70]
NB_B = [0.01, 0.02, 0.03, 0.04, 0.05]
LAGS = [0, 1, 2, 3, 4, 5]             # 반나절 단위. 0은 불가능(진단), 1 후보, 3 = 하루 늦춤, 5 = 이틀 늦춤
# 설계 선택 대안(이웃 수에 넣지 않음): 되돌림 조각 바로 무장(tranche_lab 규약), D 아래 첫 이탈도 판다, 깊은 재매수 없음(되돌림만)
DESIGN = [("되돌림 조각 바로 무장", {"rearm": "immediate"}), ("D 아래 첫 이탈도 판다", {"skip": False}),
          ("깊은 재매수 없음(되돌림만)", {"d": None})]
PLACEBO_SEEDS = 20                    # 주 판정. 보조로 가능한 원형 이동 전부(n−1개)
# 선택 보정 위약(첫 실행 뒤 추가한 보조 점검, 판정 기준 아님): PORT3 P1 격자 36개를 같은 이동으로 밀었을 때 '최대값'의 분포
P1_GRID = [(q, x, (round(x + g, 4) if g is not None else None)) for q in (0.25, 0.50)
           for x in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20) for g in (0.05, 0.10, None)]
PROXIES = ["SMH", "QQQ", "EW_AI", "MOM7_AI"]
PROXY_YEARS = list(range(2018, 2027))
EXCLUDE = {"QUBT"}                    # 2020년 전 장외 자료 → 대리 포트폴리오에서 뺀다
ETFS = {"QQQ", "SPY", "SOXX", "SMH", "IWM", "RSP", "^VIX", "^VIX3M"}
MOM_LB, MOM_N, MIN_HIST = 120, 7, 252
RATIO_CLIP = (0.2, 5.0)
TOL = 0.005                           # 0.5%p

W26 = {"전체": ("2026-01-02", "2026-10-09"), "앞": ("2026-01-02", "2026-05-29"), "뒤": ("2026-06-01", "2026-10-09"),
       "AI쇼크": ("2026-06-22", "2026-07-29"), "7~9월": ("2026-07-01", "2026-09-30")}
WH = {"2025": ("2025-01-01", "2025-12-31"), "2024": ("2024-01-01", "2024-12-31"), "2023(5/22~)": ("2023-05-22", "2023-12-31"),
      "관세": ("2025-02-18", "2025-04-04"), "24여름": ("2024-06-18", "2024-08-05"), "25가을": ("2025-10-13", "2025-11-21")}

# PORT3가 주장한 숫자(p3_PORT3.md·json과 과제 문구, 소수)
CLAIMS = {
    ("2026", "전체", "수익"): 2.390, ("2026", "전체", "낙폭"): -0.146, ("2026", "전체", "기본_수익"): 2.237,
    ("2026", "전체", "기본_낙폭"): -0.203, ("2026", "전체", "효율선초과"): 1.006, ("2026", "전체", "평균노출"): 0.852,
    ("2026", "전체", "타이밍값"): 0.610, ("2026", "전체", "주기"): 10, ("2026", "전체", "승률"): 0.30,
    ("2026", "전체", "손익비"): 3.27, ("2026", "전체", "기대값"): 0.0132,
    ("2026", "앞", "수익"): 2.327, ("2026", "앞", "효율선초과"): 0.965,
    ("2026", "뒤", "수익"): 0.019, ("2026", "뒤", "기본_수익"): 0.046, ("2026", "뒤", "효율선초과"): -0.024,
    ("2026", "뒤", "승률"): 0.20, ("2026", "뒤", "기대값"): -0.0064,
    ("2026", "AI쇼크", "수익"): -0.129, ("2026", "AI쇼크", "기본_수익"): -0.187, ("2026", "AI쇼크", "같은노출_수익"): -0.133,
    ("hold", "2025", "수익"): 0.448, ("hold", "2025", "낙폭"): -0.457, ("hold", "2025", "효율선초과"): 0.031,
    ("hold", "2025", "기본_수익"): 0.422, ("hold", "2025", "기본_낙폭"): -0.471, ("hold", "2025", "주기"): 10,
    ("hold", "2025", "승률"): 0.40, ("hold", "2025", "손익비"): 1.86, ("hold", "2025", "기대값"): 0.0068,
    ("hold", "2024", "수익"): 0.239, ("hold", "2024", "효율선초과"): -0.089,
}
CLAIMS_SLIP = {("2026", "전체", "수익"): 2.068, ("2026", "전체", "낙폭"): -0.173, ("2026", "앞", "효율선초과"): 0.226,
               ("2026", "뒤", "수익"): -0.003, ("2026", "뒤", "효율선초과"): -0.047, ("hold", "2025", "수익"): 0.384,
               ("hold", "2025", "효율선초과"): -0.034}
CLAIMS_LAG = {("2026", "전체", "수익"): 1.561, ("2026", "앞", "효율선초과"): -0.272, ("2026", "뒤", "수익"): -0.092,
              ("hold", "2025", "수익"): 0.145}
CLAIMS_NB = {"X6_D16": (("2026", "전체", "수익"), 1.899), "X10_D20": (("2026", "전체", "수익"), 1.944),
             "b2": (("2026", "전체", "수익"), 2.255), "b5": (("2026", "전체", "수익"), 2.123)}
CLAIM_CYCLES_2026 = [("2026-02-05", "2026-02-09", -0.0452), ("2026-02-11", "2026-02-19", -0.0209),
                     ("2026-03-04", "2026-03-09", 0.1440), ("2026-03-23", "2026-03-31", 0.1539),
                     ("2026-05-19", "2026-05-20", -0.0674), ("2026-06-08", "2026-06-15", -0.0786),
                     ("2026-07-06", "2026-07-10", -0.0208), ("2026-07-14", "2026-07-29", 0.1627),
                     ("2026-08-19", "2026-09-09", -0.0462), ("2026-09-15", "2026-09-18", -0.0492)]

# 첫 실행 뒤 채우는 결론·해석 문장(계산과 무관, md 전용)
CONCLUSION: list[str] = [
    "**숫자는 재현됐지만 우위 주장은 반박된다(refuted).** 주장 숫자 47개가 모두 허용 오차 안에서 맞았다(수익·낙폭 최대 차 0.05%p). "
    "2026 손절 주기 10개의 날짜·손익과 PORT3 P1 36개 전 변형도 같았고, 미래 정보 누수는 없다. "
    "그러나 고른 뒤 확인하는 뒤 구간(6/1~10/9)은 효율선초과 −2.4%, 타이밍 값 −2.5%였다. 노출 경로를 날짜만 민 위약보다 나을 게 없었고(p 0.58), 2025도 p 0.40이다.",
    "**앞 구간 +96.5%는 −18% 재매수선이 2026 바닥에 걸린 결과다.** 깊은 재매수 3번이 모두 바닥 근처에서 일어났다. "
    "3/6 바닥은 −18.18%로 선을 0.18%p만 넘었고, 나머지 둘은 3/30 −20.5%, 7/29 −22.9%였다. 재매수선을 −20%로 옮기면 2026 타이밍 값이 +61%에서 +31%로, −22%면 −5%로 준다. "
    "이웃 32개 중 앞·뒤·2025가 모두 +인 것은 3개다. 그 3개도 앞이 +2~+10%로 작고 2026 수익은 기본보다 낮다.",
    "**체결이 '다음 시가'일 때만 좋다.** 매도 10번 중 6번이 밤사이 +1.8~+4.7% 반등한 시가에 팔렸다. 같은 날 종가 체결(실제로는 불가능)은 2026 +185.4%였다. "
    "다음 날 종가는 +186.9%(앞 효율선초과 −7.2%), 하루 늦춤은 +156.1%로 모두 기본 +223.7%보다 낮았다. 미끄러짐 0.5%만 넣어도 +206.8%다. "
    "대리 포트폴리오 36개 연도 중 타이밍 값이 +인 것은 16개였다. 2022는 4/4가 +(+3.5~+7.0%)였지만 효율선초과는 음수라 현금이 더 나았다.",
]
NOTES: list[str] = [
    "전방 검증 실패의 모양: 앞 구간 위약 p는 단독 0.02, P1 36개 최대값으로 보정해 0.05다. 같은 앞 구간에서 P1 36개 중 9개가 단독 p ≤ 0.05였다. "
    "그래서 앞의 좋은 결과는 후보 하나가 아니라 '3월 두 번 하락이 −18~−21%에서 멈추고 회복한' 구간 모양 덕이다. 뒤와 2025에서 단독 p ≤ 0.05인 변형은 둘 다 0/36이었다.",
    "'승률 30%, 손익비 3.27'은 깊은 재매수 3번(+14.4%·+15.4%·+16.3%)과 되돌림 7번(−2.1~−7.9%)으로 만든 모양이다. "
    "2026 주기 기대값 +1.32%의 t는 0.42였다. 뒤 −0.15, 2025 0.26, 2024 −1.25로 모두 0과 구별되지 않는다. 깊은 재매수 3번 중 2번은 고르기 구간인 3월이었다.",
    "밤사이 갭: 2026 7종목의 평균 밤사이 수익은 +0.33%다. 고점 대비 −8% 아래 종가 다음 날은 +0.50%(63일)였다. "
    "앞 구간 깊은 재매수 두 주기의 판 값은 신호 종가보다 고점 대비 2.4~2.8%p 높았다(3/4 +2.6%, 3/23 +3.0% 갭). "
    "사용자는 한국시간 08~24시에 주문할 수 있어 정규장 시가(22:30, 겨울 23:30) 체결 자체는 가능하다. 다만 7종목을 모두 공식 시가에 체결한다는 가정은 낙관적이다. "
    "미끄러짐 0.5%를 넣으면 앞 효율선초과가 +96.5%에서 +22.6%로 준다.",
    "7월 쇼크 보호(−12.9%, 기본 −18.7%)는 같은 구간 평균 노출(0.71)로 고정해 들고 있었을 때(−13.3%)와 거의 같다(타이밍 값 +0.4%). 잘 판 덕이 아니라 덜 들고 있어서 덜 잃었다. PORT3 설명과 같다.",
    "설계 선택: 되돌림으로 되산 조각을 바로 무장하면(tranche_lab 규약) 2025 효율선초과가 +3.1%에서 −7.7%로 바뀐다. 깊은 재매수를 빼면 2026 타이밍 값이 −2.6%가 된다. "
    "그래서 값은 전부 '−18% 재매수'에서 나온다. 'D 아래 첫 이탈은 팔지 않음' 규칙은 2026·2025에 한 번도 발동하지 않았다.",
    "장세: 2022(이어진 하락)에는 대리 4개 모두 −18%에서 되산 뒤 다시 무장되지 않아 그 아래 하락을 다 들고 갔다. 주기 1~2번이 모두 이익(+11~+14%)이었고 타이밍 값은 +3.5~+7.0%였다. "
    "2018(QUBT 제외)은 타이밍 값 −3.2~+0.9%, 2020 V자는 SMH·QQQ·동일가중이 −2.6~−2.9%(모멘텀 7종목만 +18.4%)였다. "
    "2026 대리는 SMH +3.0%, QQQ −3.8%, 동일가중 +6.9%, 모멘텀 7종목 −21.8%로 갈렸다. 7종목 2024는 효율선초과 −8.9%, 주기 기대값 −2.32%였다.",
    "시험 범위(모두 첫 실행 전 상수로 고정): 이웃 32개(X×D 5×5에서 후보를 뺀 24, q 4, b 4), 체결 지연 6판(반나절 0~5), 미끄러짐 2판, 설계 대안 3, "
    "위약 시드 20 + 가능한 이동 전부(2026 전체 192, 앞 101, 뒤 90, 2025 248), 대리 4개 × 2018~2026. "
    "선택 보정 위약(P1 36개 최대값)과 PORT3 P1 36개 대조는 첫 실행 뒤 보조로 더했고 판정을 바꾸지 않았다.",
    "생존편향: 7종목 경로는 그때의 TIME 보유로 만들어 종목 선택 편향이 작다. 대리 AI 동일가중·모멘텀은 2026-10 목록이라 매수·보유 수익이 부풀려진다(규칙과 기본 모두). "
    "QUBT는 2020년 전 장외 자료 때문에 대리 전 연도에서 뺐다. 7종목 경로는 2023-05 이전이 없어 2018·2022는 대리로만 봤다.",
    "한계: 보조 경로(2023-05~2025)는 주간 PDF와 옛 분류라 현행 7종목과 구성이 다르다. 2025 창은 2023-05부터 이어 돌린 경로를 잘랐다(고점·조각 상태가 이어짐). "
    "USD, 수수료 0, 현금 이자 0, 환율은 무시했다.",
]


# ============================================================================ 자료: 7종목 경로 재구성
def ohlc_wide():
    o = pd.read_parquet(CACHE / "ohlc.parquet")
    o["date"] = pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d")
    o = o.drop_duplicates(["date", "sym"], keep="last")
    return o.pivot(index="date", columns="sym", values="Open"), o.pivot(index="date", columns="sym", values="Close")


def rebuild(name, O, C, bump_after=None, seed=0):
    """일별 비중 w[k](k일 시가 거래 뒤)로 vo[k](k일 시가 가치)와 vc[k](k일 종가 평가 = 그림자)를 만든다.
    bump_after=T면 T 다음 날부터 시가·종가를 ±5% 흔든다(미래 교란 점검)."""
    b = json.loads((CACHE / f"{name}.json").read_text(encoding="utf-8"))
    days, W, tk = b["days"], b["weights"], b["ticker"]
    syms = sorted(set(tk.values()))
    Od = O.reindex(days)[syms].to_numpy(float).copy()
    Cd = C.reindex(days)[syms].to_numpy(float).copy()
    if bump_after is not None:
        rng = np.random.default_rng(seed)
        n = len(days) - bump_after - 1
        Od[bump_after + 1:] *= 1 + rng.normal(0, 0.05, (n, len(syms)))
        Cd[bump_after + 1:] *= 1 + rng.normal(0, 0.05, (n, len(syms)))
    col = {s: i for i, s in enumerate(syms)}
    N = len(days)
    vo, vc = np.ones(N), np.ones(N)
    for k in range(N):
        cash = 1.0 - sum(W[k].values())
        sc = so = cash
        for key, wt in W[k].items():
            j = col[tk[key]]
            o0, c0 = Od[k, j], Cd[k, j]
            ok0 = np.isfinite(o0) and o0 > 0
            sc += wt * (c0 / o0 if ok0 and np.isfinite(c0) else 1.0)
            if k + 1 < N:
                o1 = Od[k + 1, j]
                so += wt * (o1 / o0 if ok0 and np.isfinite(o1) else 1.0)
        vc[k] = vo[k] * sc
        if k + 1 < N:
            vo[k + 1] = vo[k] * so
    return {"days": days, "vo": vo, "vc": vc, "json_open": np.array(b["open"]), "json_close": np.array(b["close"])}


def half(P):
    H = np.empty(2 * len(P["vo"]))
    H[0::2], H[1::2] = P["vo"], P["vc"]
    return H


# ============================================================================ 규칙: 반나절 경로 위 상태 기계(새로 짬)
def simulate(H, days, p, slip=0.0):
    """H 반나절 가치 경로. 홀수 j(종가)에서 판단, j+L에서 체결. 반환 V(반나절 가치), E(반나절 노출, 그 시점 거래 뒤), cycles, events."""
    q, x, d, b, L = p["q"], p["x"], p["d"], p["b"], p["L"]
    rearm_now, skip = p.get("rearm", "line") == "immediate", p.get("skip", True)
    J = len(H)
    st = {"fix": (1 - q) / H[0], "mov": q / H[0], "cash": 0.0, "held": True, "armed": True,
          "sold_px": np.nan, "sold_j": -1}
    V, E = np.empty(J), np.empty(J)
    cycles, events = [], []

    def execute(j, kind):
        if kind == "sell":
            px = H[j] * (1 - slip)
            st.update(cash=st["mov"] * px, mov=0.0, held=False, sold_px=px, sold_j=j)
        else:
            px = H[j] * (1 + slip)
            cycles.append({"exit_day": days[st["sold_j"] // 2], "exit_j": st["sold_j"], "re_day": days[j // 2], "re_j": j,
                           "exit_px": float(st["sold_px"]), "re_px": float(px), "gain": float(st["sold_px"] / px - 1),
                           "kind": kind, "w": q, "open": False})
            st.update(mov=st["cash"] / px, cash=0.0, held=True, armed=(rearm_now and kind == "되돌림"))
        events.append({"j": j, "day": days[j // 2], "kind": kind, "px": float(px)})

    def mark(j):
        hv = (st["fix"] + st["mov"]) * H[j]
        V[j] = hv + st["cash"]
        E[j] = hv / V[j] if V[j] > 0 else 0.0

    peak = -np.inf
    pend = None                                   # (체결 j, 종류)
    for j in range(J):
        if pend is not None and pend[0] == j:
            execute(j, pend[1])
            pend = None
        mark(j)
        if j % 2 == 0:
            continue
        c = H[j]
        peak = max(peak, c)
        if pend is not None or x is None:
            continue
        lx = peak * (1 - x)
        ld = peak * (1 - d) if d is not None else -np.inf
        order = None
        if st["held"]:
            if not st["armed"]:
                if c > lx:
                    st["armed"] = True
            elif c <= lx:
                if skip and c <= ld:
                    st["armed"] = False
                    events.append({"j": j, "day": days[j // 2], "kind": "skip", "px": float(c)})
                else:
                    order = "sell"
        else:
            if c >= st["sold_px"] * (1 + b):
                order = "되돌림"
            elif d is not None and ld < st["sold_px"] and c <= ld:
                order = "깊은 재매수"
        if order is not None:
            if L == 0:
                execute(j, order)
                mark(j)
            elif j + L < J:
                pend = (j + L, order)
    if not st["held"]:
        cycles.append({"exit_day": days[st["sold_j"] // 2], "exit_j": st["sold_j"], "re_day": None, "re_j": None,
                       "exit_px": float(st["sold_px"]), "re_px": float(H[-1]), "gain": float(st["sold_px"] / H[-1] - 1),
                       "kind": "열림", "w": q, "open": True})
    return V, E, cycles, events


def daily(V, E, H, L):
    """반나절 → 시가→시가 일수익 r[k], 기본 rb[k], 노출 e[k](홀수 L은 시가 거래 뒤 노출, 짝수 L은 시가·종가 노출 평균)."""
    vo = H[0::2]
    r = V[2::2] / V[:-2:2] - 1
    rb = vo[1:] / vo[:-1] - 1
    eo, ec = E[0::2][:-1], E[1::2][:-1]
    e = eo if L % 2 == 1 else (eo + ec) / 2
    return r, rb, e


# ============================================================================ 지표
EG = np.round(np.arange(0.0, 1.0001, 0.005), 3)


def mdd(cur):
    return float((cur / np.maximum.accumulate(cur) - 1).min())


def frontier(rb):
    cur = np.concatenate([np.ones((len(EG), 1)), np.cumprod(1 + np.outer(EG, rb), axis=1)], axis=1)
    return cur[:, -1] - 1, (cur / np.maximum.accumulate(cur, axis=1) - 1).min(axis=1)


def eff_excess(ret, dd, fr):
    fret, fdd = fr
    ok = fdd >= dd - 1e-12
    i = int(np.argmax(np.where(ok, fret, -np.inf)))
    return float(ret - fret[i]), float(EG[i])


def cyc_stats(cyc):
    g = np.array([c["gain"] for c in cyc], float)
    w = np.array([c["w"] for c in cyc], float)
    win, loss = g[g > 0], g[g <= 0]
    out = {"주기": int(len(g)), "승률": float(len(win) / len(g)) if len(g) else float("nan"),
           "평균이익": float(win.mean()) if len(win) else float("nan"),
           "평균손실": float(loss.mean()) if len(loss) else float("nan"),
           "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
           "기대값": float(g.mean()) if len(g) else float("nan"),
           "기대값_가중": float((g * w).sum() / w.sum()) if len(g) and w.sum() > 0 else float("nan"),
           "깊은재매수비율": float(np.mean([c["kind"] == "깊은 재매수" for c in cyc])) if cyc else float("nan"),
           "기대값_t": float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))) if len(g) > 2 and g.std(ddof=1) > 0 else float("nan")}
    return out


def wstats(days, r, rb, e, cyc, lo, hi):
    d = np.asarray(days[:-1])
    m = (d >= lo) & (d <= hi)
    x, b, ew = r[m], rb[m], e[m]
    cur, bcur = np.concatenate([[1.0], np.cumprod(1 + x)]), np.concatenate([[1.0], np.cumprod(1 + b)])
    ret, dd = float(cur[-1] - 1), mdd(cur)
    exc, e_eq = eff_excess(ret, dd, frontier(b))
    ea = float(ew.mean()) if m.any() else float("nan")
    fcur = np.concatenate([[1.0], np.cumprod(1 + ea * b)])
    out = {"수익": ret, "낙폭": dd, "기본_수익": float(bcur[-1] - 1), "기본_낙폭": mdd(bcur), "효율선초과": exc, "효율선_e": e_eq,
           "평균노출": ea, "같은노출_수익": float(fcur[-1] - 1), "타이밍값": ret - float(fcur[-1] - 1),
           "칼마": ret / abs(dd) if dd < -1e-9 else float("inf")}
    out.update(cyc_stats([c for c in cyc if lo <= c["exit_day"] <= hi]))
    return out


def run(P, p, slip=0.0, wins=None):
    H = half(P)
    V, E, cyc, ev = simulate(H, P["days"], p, slip)
    r, rb, e = daily(V, E, H, p["L"])
    res = {w: wstats(P["days"], r, rb, e, cyc, lo, hi) for w, (lo, hi) in (wins or {}).items()}
    return res, {"r": r, "rb": rb, "e": e, "cycles": cyc, "events": ev, "V": V, "E": E}


def measure7(P26, PH, p, slip=0.0):
    a, da = run(P26, p, slip, W26)
    h, dh = run(PH, p, slip, WH)
    return {"2026": a, "hold": h}, da, dh


# ============================================================================ 위약: 노출 경로 원형 이동
def wmet(ew, rb, fr):
    cur = np.concatenate([[1.0], np.cumprod(1 + ew * rb)])
    ret, dd = float(cur[-1] - 1), mdd(cur)
    return {"수익": ret, "낙폭": dd, "효율선초과": eff_excess(ret, dd, fr)[0], "칼마": ret / abs(dd) if dd < -1e-9 else np.inf}


def placebo(days, e, rb, lo, hi):
    d = np.asarray(days[:-1])
    m = (d >= lo) & (d <= hi)
    ew, b = e[m], rb[m]
    n = len(ew)
    fr = frontier(b)
    act = wmet(ew, b, fr)
    fixed = float(np.prod(1 + ew.mean() * b) - 1)
    seeds = [int(np.random.default_rng(s).integers(1, n)) for s in range(PLACEBO_SEEDS)]
    S = [wmet(np.roll(ew, s), b, fr) for s in seeds]
    A = [wmet(np.roll(ew, s), b, fr) for s in range(1, n)]

    def rank(L, k):
        v = np.array([z[k] for z in L], float)
        a = act[k]
        hi_ = int((v >= a - 1e-12).sum())
        return {"실제": a, "위약_중앙": float(np.median(v)), "위약_상위10%": float(np.quantile(v, 0.9)),
                "위약_최대": float(v.max()), "이상개수": hi_, "개수": len(v), "p": float((1 + hi_) / (1 + len(v)))}
    return {"창일수": n, "평균노출": float(ew.mean()), "같은노출_고정_수익": fixed, "시드": seeds,
            "시드20": {k: rank(S, k) for k in ("수익", "효율선초과", "낙폭", "칼마")},
            "전수": {k: rank(A, k) for k in ("수익", "효율선초과", "낙폭", "칼마")}}


def grid_maxstat(P, lo, hi):
    """P1 격자 36개 노출 경로를 같은 이동 s로 함께 밀어 '격자 최대 효율선초과'의 위약 분포를 만든다(선택 보정).
    변형별 p도 함께(각자 실제 이상인 이동 수)."""
    d = np.asarray(P["days"][:-1])
    m = (d >= lo) & (d <= hi)
    es, rb = [], None
    for q, x, dd in P1_GRID:
        _, det = run(P, dict(CAND, q=q, x=x, d=dd), 0.0, {})
        es.append(det["e"][m])
        rb = det["rb"][m]
    fr = frontier(rb)
    act = np.array([wmet(e, rb, fr)["효율선초과"] for e in es])
    n = len(rb)
    sh = np.array([[wmet(np.roll(e, s), rb, fr)["효율선초과"] for e in es] for s in range(1, n)])   # (이동, 변형)
    mx = sh.max(axis=1)
    pv = [(1 + int((sh[:, i] >= act[i] - 1e-12).sum())) / n for i in range(len(es))]
    i_c = P1_GRID.index((CAND["q"], CAND["x"], CAND["d"]))
    return {"이동수": n - 1, "격자수": len(es), "실제_최대": float(act.max()), "실제_최대_변형": list(P1_GRID[int(act.argmax())]),
            "후보_실제": float(act[i_c]), "위약최대_중앙": float(np.median(mx)), "위약최대_상위10%": float(np.quantile(mx, 0.9)),
            "p_최대값": float((1 + int((mx >= act.max() - 1e-12).sum())) / n), "후보_p_단독": pv[i_c],
            "p≤0.05_변형수": int(sum(p <= 0.05 for p in pv)), "실제>0_변형수": int((act > 0).sum())}


# ============================================================================ 하락 구간(그림자 −8% 이탈 뒤 새 고점 전까지의 최저)
def episodes(days, vc, x=0.08):
    out, peak, pk_i, low, low_i, inn = [], -np.inf, 0, None, None, False
    for i, c in enumerate(vc):
        if c > peak:
            if inn:
                out.append({"고점일": days[pk_i], "이탈일": days[x_i], "바닥일": days[low_i], "깊이": float(low / peak - 1),
                            "회복": True})
                inn = False
            peak, pk_i = c, i
            continue
        if not inn and c <= peak * (1 - x):
            inn, x_i, low, low_i = True, i, c, i
        if inn and c < low:
            low, low_i = c, i
    if inn:
        out.append({"고점일": days[pk_i], "이탈일": days[x_i], "바닥일": days[low_i], "깊이": float(low / peak - 1), "회복": False})
    return out


# ============================================================================ 장세 대리(p2_long)
def proxies():
    df = pd.read_parquet(CACHE / "p2_long.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["Open", "Close"])
    df = df[(df.Close > 0) & (df.Open > 0)].drop_duplicates(["date", "sym"], keep="last")
    f = (df["Adj Close"] / df["Close"]).fillna(1.0)
    df = df.assign(O=df.Open * f, C=df.Close * f)
    cal = sorted(df[df.sym == "SPY"].date.unique())
    O = df.pivot(index="date", columns="sym", values="O").reindex(cal)
    C = df.pivot(index="date", columns="sym", values="C").reindex(cal)
    out = {}
    for s in ("SMH", "QQQ"):
        o, c = O[s].to_numpy(float), C[s].to_numpy(float)
        out[s] = {"days": cal, "vo": o / o[0], "vc": c / o[0]}
    stocks = [s for s in O.columns if s not in ETFS and s not in EXCLUDE]
    Oa, Ca = O[stocks].to_numpy(float), C[stocks].to_numpy(float)
    hist = np.cumsum(np.isfinite(Ca), axis=0) - np.isfinite(Ca)
    N = len(cal)
    clip = lambda v: np.clip(v, *RATIO_CLIP)
    vo, vc = np.ones(N), np.ones(N)
    for k in range(N):
        el = (hist[k] >= MIN_HIST) & np.isfinite(Oa[k])
        if not el.any():
            vc[k] = vo[k]
            if k + 1 < N:
                vo[k + 1] = vo[k]
            continue
        rc = clip(np.where(np.isfinite(Ca[k][el]), Ca[k][el] / Oa[k][el], 1.0))
        vc[k] = vo[k] * rc.mean()
        if k + 1 < N:
            ro = clip(np.where(np.isfinite(Oa[k + 1][el]), Oa[k + 1][el] / Oa[k][el], rc))
            vo[k + 1] = vo[k] * ro.mean()
    out["EW_AI"] = {"days": cal, "vo": vo, "vc": vc}
    wk = [date.fromisoformat(d).isocalendar()[:2] for d in cal]
    vo, vc = np.ones(N), np.ones(N)
    held = {}
    for k in range(N):
        if k > MOM_LB + 1 and wk[k] != wk[k - 1]:
            j = k - 1
            el = (hist[k] >= MIN_HIST) & np.isfinite(Ca[j]) & np.isfinite(Ca[j - MOM_LB]) & np.isfinite(Oa[k])
            sc = np.where(el, Ca[j] / np.where(np.isfinite(Ca[j - MOM_LB]), Ca[j - MOM_LB], 1.0) - 1, -np.inf)
            top = [int(i) for i in np.argsort(-sc)[:MOM_N] if np.isfinite(sc[i])]
            held = {i: 1.0 / len(top) for i in top} if top else {}
        cash = 1.0 - sum(held.values())
        sc_ = so_ = cash
        nw = {}
        for i, wt in held.items():
            o0, c0 = Oa[k, i], Ca[k, i]
            rc = float(clip(c0 / o0)) if np.isfinite(o0) and np.isfinite(c0) and o0 > 0 else 1.0
            sc_ += wt * rc
            if k + 1 < N:
                o1 = Oa[k + 1, i]
                ro = float(clip(o1 / o0)) if np.isfinite(o0) and np.isfinite(o1) and o0 > 0 else rc
                so_ += wt * ro
                nw[i] = wt * ro
        vc[k] = vo[k] * sc_
        if k + 1 < N:
            vo[k + 1] = vo[k] * so_
            held = {i: v / so_ for i, v in nw.items()}
    out["MOM7_AI"] = {"days": cal, "vo": vo, "vc": vc}
    return out


def proxy_year(P, Y, p):
    days = P["days"]
    a = next(i for i, d in enumerate(days) if d >= f"{Y}-01-01")
    z = max(i for i, d in enumerate(days) if d <= f"{Y}-12-31")
    end = min(z + 1, len(days) - 1)
    seg = {"days": days[a:end + 1], "vo": P["vo"][a:end + 1] / P["vo"][a], "vc": P["vc"][a:end + 1] / P["vo"][a]}
    wins = {str(Y): (f"{Y}-01-01", f"{Y}-12-31")}
    if Y == 2026:
        wins = {"2026": W26["전체"], "2026앞": W26["앞"], "2026뒤": W26["뒤"]}
    res, det = run(seg, p, 0.0, wins)
    res["_episodes"] = episodes(seg["days"], seg["vc"], p["x"])
    return res


# ============================================================================ 출력 도우미
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def pct(v, d=1, sign=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%"


def num(v, d=2):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def tol_of(k):
    return 0.5 if k == "주기" else (0.01 if k == "손익비" else (0.006 if k in ("승률", "평균노출") else TOL))


# ============================================================================ 본 실행
def main():
    O, C = ohlc_wide()
    P26, PH = rebuild("base_2026", O, C), rebuild("base_hold", O, C)
    chk = {}
    for nm, P in (("2026", P26), ("hold", PH)):
        chk[f"재구성_시가경로_최대상대차_{nm}"] = float(np.max(np.abs(P["vo"] / P["json_open"] - 1)))
        chk[f"재구성_종가평가_최대상대차_{nm}"] = float(np.max(np.abs(P["vc"] / P["json_close"] - 1)))
    print("재구성", chk)

    # ---- 1) 재현
    res, d26, dh = measure7(P26, PH, CAND)
    chk["노출×기본=통계산_최대차_2026"] = float(np.max(np.abs(d26["e"] * d26["rb"] - d26["r"])))
    rep = []

    def add(label, mine, cl, k):
        rep.append({"항목": label, "주장": cl, "재계산": mine, "차": mine - cl, "일치": bool(abs(mine - cl) <= tol_of(k))})
    for (bn, w, k), cl in CLAIMS.items():
        add(f"{bn}/{w}/{k}", res[bn][w][k], cl, k)
    res_slip, ds26, _ = measure7(P26, PH, CAND, SLIP)
    for (bn, w, k), cl in CLAIMS_SLIP.items():
        add(f"미끄러짐0.5%/{bn}/{w}/{k}", res_slip[bn][w][k], cl, k)
    lag_res = {}
    for L in LAGS:
        lag_res[L] = measure7(P26, PH, dict(CAND, L=L))[0]
    for (bn, w, k), cl in CLAIMS_LAG.items():
        add(f"하루늦춤(L=3)/{bn}/{w}/{k}", lag_res[3][bn][w][k], cl, k)
    nb_claim = {"X6_D16": dict(CAND, x=0.06, d=0.16), "X10_D20": dict(CAND, x=0.10, d=0.20), "b2": dict(CAND, b=0.02),
                "b5": dict(CAND, b=0.05)}
    for key, ((bn, w, k), cl) in CLAIMS_NB.items():
        add(f"이웃 {key}/{bn}/{w}/{k}", measure7(P26, PH, nb_claim[key])[0][bn][w][k], cl, k)
    # PORT3 json의 P1 36개 전 변형과 대조(결과 파일만 읽음, 코드 import 없음)
    pj = json.loads((RESULTS / "p3_PORT3.json").read_text(encoding="utf-8"))
    worst = {"2026전체수익": 0.0, "앞효율선초과": 0.0, "뒤효율선초과": 0.0, "2025효율선초과": 0.0}
    for row in pj["rows"]:
        s = row["spec"]
        if s["fam"] != "P1":
            continue
        dd = round(s["x"] + s["gap"], 4) if s["gap"] is not None else None
        mr = measure7(P26, PH, dict(CAND, q=s["q"], x=s["x"], d=dd))[0]
        for key, (bn, w, k) in {"2026전체수익": ("2026", "전체", "수익"), "앞효율선초과": ("2026", "앞", "효율선초과"),
                                "뒤효율선초과": ("2026", "뒤", "효율선초과"), "2025효율선초과": ("hold", "2025", "효율선초과")}.items():
            worst[key] = max(worst[key], abs(mr[bn][w][k] - row[bn][w][k]))
    chk.update({f"PORT3_P1_36개_최대차_{k}": v for k, v in worst.items()})
    print("PORT3 P1 36개 대조", worst)
    mine_cyc = [(c["exit_day"], c["re_day"], round(c["gain"], 4)) for c in d26["cycles"]]
    cyc_match = len(mine_cyc) == len(CLAIM_CYCLES_2026) and all(
        a[0] == b[0] and a[1] == b[1] and abs(a[2] - b[2]) < 0.0006 for a, b in zip(mine_cyc, CLAIM_CYCLES_2026))
    print("재현")
    for x in rep:
        print(f"  {x['항목']:<40} 주장 {x['주장']:+.4f} 재계산 {x['재계산']:+.4f} 차 {x['차']:+.4f} {'OK' if x['일치'] else '불일치'}")
    print("2026 주기 일치", cyc_match, mine_cyc)

    # ---- 2) 시점 감사
    audit = {}
    worst_ev, worst_v, worst_c = 0, 0.0, 0.0
    base_ev = [(e["j"], e["kind"]) for e in d26["events"]]
    H0 = half(P26)
    Ts = [20, 45, 60, 80, 100, 120, 140, 160, 180]
    for T in Ts:
        Pb = rebuild("base_2026", O, C, bump_after=T, seed=T)
        Hb = half(Pb)
        Vb, Eb, cb, evb = simulate(Hb, Pb["days"], CAND)
        V0 = d26["V"]
        a1 = [z for z in base_ev if z[0] <= 2 * T + 2]
        a2 = [(e["j"], e["kind"]) for e in evb if e["j"] <= 2 * T + 2]
        worst_ev += int(a1 != a2)
        worst_v = max(worst_v, float(np.max(np.abs(Vb[:2 * T + 2] - V0[:2 * T + 2]))))
        worst_c = max(worst_c, float(np.max(np.abs(Hb[:2 * T + 2] - H0[:2 * T + 2]))))
    audit["미래교란_T개수"] = len(Ts)
    audit["미래교란_사건불일치(T+1 시가까지)"] = worst_ev
    audit["미래교란_가치차(T 종가까지)"] = worst_v
    audit["미래교란_그림자차(T 종가까지)"] = worst_c
    # 체결 시점: 모든 매매가 종가 판단(홀수 j) 다음 반나절(짝수 j = 시가)에 일어났나
    sells = [e for e in d26["events"] if e["kind"] in ("sell", "되돌림", "깊은 재매수")]
    audit["체결이_모두_시가"] = all(e["j"] % 2 == 0 for e in sells)
    # 판 날 시가와 전날 종가(밤사이 갭)
    gaps = []
    for e in sells:
        j = e["j"]
        gaps.append({"day": e["day"], "kind": e["kind"], "밤사이": float(H0[j] / H0[j - 1] - 1)})
    audit["체결일_밤사이"] = gaps
    hd = np.asarray(P26["days"])
    on = P26["vo"][1:] / P26["vc"][:-1] - 1
    audit["2026_평균_밤사이"] = float(on.mean())
    print("시점 감사", {k: v for k, v in audit.items() if k != "체결일_밤사이"})

    # ---- 3) 늦춤·미끄러짐
    slip_lag3 = measure7(P26, PH, dict(CAND, L=3), SLIP)[0]

    # ---- 4) 이웃
    nbs = [("XD", x, dd, dict(CAND, x=x, d=dd)) for x in NB_X for dd in NB_D]
    nbs += [("q", q, None, dict(CAND, q=q)) for q in NB_Q]
    nbs += [("b", bb, None, dict(CAND, b=bb)) for bb in NB_B]
    nb_rows = []
    for kind, v1, v2, sp in nbs:
        is_c = sp == CAND
        r = measure7(P26, PH, sp)[0]
        a, h = r["2026"], r["hold"]
        nb_rows.append({"kind": kind, "v1": v1, "v2": v2, "center": is_c,
                        "앞": a["앞"]["효율선초과"], "뒤": a["뒤"]["효율선초과"], "2025": h["2025"]["효율선초과"],
                        "2026수익": a["전체"]["수익"], "2026낙폭": a["전체"]["낙폭"], "2026초과": a["전체"]["효율선초과"],
                        "2026타이밍": a["전체"]["타이밍값"], "뒤수익": a["뒤"]["수익"], "2025수익": h["2025"]["수익"],
                        "AI쇼크": a["AI쇼크"]["수익"], "주기2026": a["전체"]["주기"], "기대값2026": a["전체"]["기대값"]})
    uniq = []
    seen = set()
    for z in nb_rows:
        key = (z["kind"], z["v1"], z["v2"])
        if z["center"] or key in seen:
            continue
        seen.add(key)
        uniq.append(z)
    nb_sum = {"개수": len(uniq), "앞>0": sum(z["앞"] > 0 for z in uniq), "뒤>0": sum(z["뒤"] > 0 for z in uniq),
              "2025>0": sum(z["2025"] > 0 for z in uniq), "셋다>0": sum(z["앞"] > 0 and z["뒤"] > 0 and z["2025"] > 0 for z in uniq),
              "2026초과>0": sum(z["2026초과"] > 0 for z in uniq), "2026타이밍>0": sum(z["2026타이밍"] > 0 for z in uniq),
              "2026수익>기본": sum(z["2026수익"] > res["2026"]["전체"]["기본_수익"] for z in uniq)}
    print("이웃", nb_sum)
    design_rows = []
    for lab, ch in DESIGN:
        r = measure7(P26, PH, dict(CAND, **ch))[0]
        design_rows.append({"name": lab, "res": r})

    # ---- 5) 위약
    pl = {"2026전체": placebo(P26["days"], d26["e"], d26["rb"], *W26["전체"]),
          "2026앞": placebo(P26["days"], d26["e"], d26["rb"], *W26["앞"]),
          "2026뒤": placebo(P26["days"], d26["e"], d26["rb"], *W26["뒤"]),
          "2025": placebo(PH["days"], dh["e"], dh["rb"], *WH["2025"])}
    gm = {w: grid_maxstat(P26, *W26[w]) for w in ("앞", "뒤", "전체")}
    gm["2025"] = grid_maxstat(PH, *WH["2025"])
    for k, v in gm.items():
        print("선택 보정 위약", k, v)
    for k, v in pl.items():
        print("위약", k, {m: (round(v["시드20"][m]["실제"], 4), round(v["시드20"][m]["위약_중앙"], 4), round(v["시드20"][m]["p"], 3),
                            round(v["전수"][m]["p"], 3)) for m in ("수익", "효율선초과", "낙폭")})

    # ---- 6) 장세
    eps = {"2026": episodes(P26["days"], P26["vc"]), "hold": episodes(PH["days"], PH["vc"])}
    PX = proxies()
    preg = {}
    for nm in PROXIES:
        for Y in PROXY_YEARS:
            preg[f"{nm}_{Y}"] = proxy_year(PX[nm], Y, CAND)
    for Y in (2018, 2022):
        for nm in PROXIES:
            s = preg[f"{nm}_{Y}"][str(Y)]
            print(f"대리 {nm} {Y}: 수익 {s['수익']:+.1%}/{s['낙폭']:.1%} 기본 {s['기본_수익']:+.1%}/{s['기본_낙폭']:.1%} "
                  f"초과 {s['효율선초과']:+.1%} 타이밍 {s['타이밍값']:+.1%} 주기 {s['주기']} 승 {s['승률']} 손익비 {s['손익비']} 기대 {s['기대값']}")

    payload = {"id": ID, "cand": CAND, "checks": chk, "audit": audit, "claims": rep, "cycles_match": cyc_match,
               "cycles_2026": d26["cycles"], "cycles_2025": [c for c in dh["cycles"] if "2025-01-01" <= c["exit_day"] <= "2025-12-31"],
               "res": res, "slip": res_slip, "slip_lag3": slip_lag3, "lags": lag_res, "neighbors": nb_rows, "nb_sum": nb_sum,
               "design": design_rows, "placebo": pl, "grid_maxstat": gm, "episodes": eps, "proxy": preg,
               "params": {"NB_X": NB_X, "NB_D": NB_D, "NB_Q": NB_Q, "NB_B": NB_B, "LAGS": LAGS, "SLIP": SLIP,
                          "PLACEBO_SEEDS": PLACEBO_SEEDS, "PROXIES": PROXIES, "PROXY_YEARS": PROXY_YEARS,
                          "EXCLUDE": sorted(EXCLUDE)}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{ID}.json").write_text(json.dumps(clean(payload), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_md(payload)
    print("저장", RESULTS / f"{ID}.json", RESULTS / f"{ID}.md")


# ============================================================================ md
def write_md(P):
    L = []
    A = L.append
    r26, rh = P["res"]["2026"], P["res"]["hold"]
    A("# 3단계 반박 검증 V_PORT3: 그림자 −8%에서 50% 매도 → −18% 재매수 또는 되돌림 (2026-10-11)")
    A("")
    A("> 코드 `code/p3_V_PORT3.py`, 전체 숫자 `results/p3_V_PORT3.json`. harness·stock_lab·tranche_lab·p3_PORT3을 import하지 않았다. "
      "`base_*.json`의 일별 비중과 `ohlc.parquet`의 시가·종가로 7종목 경로와 그림자를 다시 만들었다"
      f"(json 경로와 상대차 최대 {max(v for k, v in P['checks'].items() if k.startswith('재구성')):.1e}). "
      "규칙은 시가·종가를 번갈아 놓은 반나절 경로 위의 새 상태 기계로 짰다. USD, 수수료 0, 현금 이자 0.")
    A("")
    A("## 결론 3줄")
    A("")
    for i, s in enumerate(CONCLUSION, 1):
        A(f"{i}. {s}")
    if not CONCLUSION:
        A("(작성 전)")
    A("")
    # 재현
    A("## 1. 재현 (주장 → 재계산)")
    A("")
    ok = sum(x["일치"] for x in P["claims"])
    A(f"주장 숫자 {len(P['claims'])}개 중 {ok}개가 허용 오차(수익·낙폭 0.5%p, 승률 0.6%p, 손익비 0.01, 주기 수 정확) 안에서 맞았다. "
      f"2026 손절 주기 10개의 날짜·손익: {'모두 일치' if P['cycles_match'] else '불일치'}.")
    A("")
    A("| 항목 | 주장 | 재계산 | 차(%p) | 일치 |")
    A("|---|---|---|---|---|")
    for x in P["claims"]:
        k = x["항목"].split("/")[-1]
        if k in ("주기",):
            f = lambda v: f"{v:.0f}"
            dtxt = f"{x['차']:+.0f}"
        elif k in ("손익비",):
            f = lambda v: f"{v:.2f}"
            dtxt = f"{x['차']:+.3f}"
        elif k == "평균노출":
            f = lambda v: f"{v:.3f}"
            dtxt = f"{x['차']:+.3f}"
        else:
            f = lambda v: pct(v, 1 if abs(v) > 0.02 else 2)
            dtxt = f"{x['차'] * 100:+.2f}"
        A(f"| {x['항목']} | {f(x['주장'])} | {f(x['재계산'])} | {dtxt} | {'예' if x['일치'] else '**아니오**'} |")
    A("")
    # 시점
    a = P["audit"]
    A("## 2. 시점 감사")
    A("")
    A("| 점검 | 결과 |")
    A("|---|---|")
    A("| 그림자 | t일 종가 평가 = 그날 시가 거래 뒤 비중 × 종가/시가. 고점은 t 종가를 포함한 과거 최고 종가 |")
    A(f"| 체결 | 종가 판단(반나절 홀수) → 다음 시가(짝수) 체결. 매매 {len(a['체결일_밤사이'])}건 모두 시가 체결: {'예' if a['체결이_모두_시가'] else '아니오'} |")
    A(f"| 미래 교란 | {a['미래교란_T개수']}개 날짜 T마다 T 다음 날부터 가격을 ±5% 흔들었다. T+1 시가까지 매매 사건 불일치 {a['미래교란_사건불일치(T+1 시가까지)']}건, "
      f"T 종가까지 가치 차 {a['미래교란_가치차(T 종가까지)']:.1e}, 그림자 차 {a['미래교란_그림자차(T 종가까지)']:.1e} |")
    l0 = P["lags"]["0"] if "0" in P["lags"] else P["lags"][0]
    A(f"| 같은 종가 체결(불가능한 판, 진단) | 2026 {pct(l0['2026']['전체']['수익'])}, 앞 효율선초과 {pct(l0['2026']['앞']['효율선초과'])}, "
      f"2025 {pct(l0['hold']['2025']['수익'])}. 주장 숫자는 이 판이 아니라 다음 시가 체결 판과 맞는다 → 누수 없음 |")
    g = a["체결일_밤사이"]
    sg = [x for x in g if x["kind"] == "sell"]
    A(f"| 밤사이 갭 | 2026 7종목 평균 밤사이(종가→다음 시가) {pct(a['2026_평균_밤사이'], 2)}. 매도 체결 {len(sg)}번의 밤사이: "
      + ", ".join(f"{x['day'][5:]} {pct(x['밤사이'])}" for x in sg) + " |")
    A("")
    # 늦춤·미끄러짐
    A("## 3. 체결 늦춤·미끄러짐")
    A("")
    A("| 판 | 2026 수익/낙폭 | 앞 수익 · 효율선초과 | 뒤 수익 · 효율선초과 | 7월 쇼크 | 2025 수익/낙폭 · 효율선초과 | 2026 타이밍 값 | 2026 주기 · 기대값 |")
    A("|---|---|---|---|---|---|---|---|")

    def lrow(lab, r):
        a_, h_ = r["2026"], r["hold"]
        return (f"| {lab} | {pct(a_['전체']['수익'])}/{pct(a_['전체']['낙폭'])} | {pct(a_['앞']['수익'])} · {pct(a_['앞']['효율선초과'])} | "
                f"{pct(a_['뒤']['수익'])} · {pct(a_['뒤']['효율선초과'])} | {pct(a_['AI쇼크']['수익'])} | "
                f"{pct(h_['2025']['수익'])}/{pct(h_['2025']['낙폭'])} · {pct(h_['2025']['효율선초과'])} | {pct(a_['전체']['타이밍값'])} | "
                f"{a_['전체']['주기']} · {pct(a_['전체']['기대값'], 2)} |")
    base = {"2026": {w: {"수익": r26[w]["기본_수익"], "낙폭": r26[w]["기본_낙폭"], "효율선초과": 0.0, "타이밍값": 0.0, "주기": 0,
                         "기대값": float("nan")} for w in r26},
            "hold": {w: {"수익": rh[w]["기본_수익"], "낙폭": rh[w]["기본_낙폭"], "효율선초과": 0.0} for w in rh}}
    A(lrow("기본(그대로 보유)", base))
    labs = {0: "같은 종가 체결(불가능, 진단)", 1: "**후보: 다음 시가**", 2: "다음 날 종가", 3: "이틀째 시가(하루 늦춤)", 4: "이틀째 종가",
            5: "사흘째 시가(이틀 늦춤)"}
    for Lg in LAGS:
        A(lrow(labs[Lg], P["lags"][Lg] if Lg in P["lags"] else P["lags"][str(Lg)]))
    A(lrow("후보 + 미끄러짐 0.5%", P["slip"]))
    A(lrow("하루 늦춤 + 미끄러짐 0.5%", P["slip_lag3"]))
    A("")
    # 이웃
    A(f"## 4. 이웃 (사전 고정 {P['nb_sum']['개수']}개, 각 모수 앞뒤 2칸)")
    A("")
    s = P["nb_sum"]
    A(f"앞·뒤·2025 효율선초과가 셋 다 + 인 이웃: **{s['셋다>0']}/{s['개수']}**. 앞 + {s['앞>0']}, 뒤 + {s['뒤>0']}, 2025 + {s['2025>0']}, "
      f"2026 전체 효율선초과 + {s['2026초과>0']}, 2026 타이밍 값 + {s['2026타이밍>0']}, 2026 수익이 기본보다 높음 {s['2026수익>기본']}.")
    A("")
    A("**손절선 X × 재매수선 D: 칸 = 앞 · 뒤 · 2025 효율선초과(%)** (가운데 굵게 = 후보, 파는 몫 50%, 되돌림 3%)")
    A("")
    A("| 손절 \\ 재매수 | " + " | ".join(f"−{d * 100:.0f}%" for d in NB_D) + " |")
    A("|---|" + "---|" * len(NB_D))
    nbr = P["neighbors"]
    for x in NB_X:
        cells = []
        for dd in NB_D:
            z = next(z for z in nbr if z["kind"] == "XD" and abs(z["v1"] - x) < 1e-9 and abs(z["v2"] - dd) < 1e-9)
            t = f"{z['앞'] * 100:+.0f} · {z['뒤'] * 100:+.0f} · {z['2025'] * 100:+.0f}"
            cells.append(f"**{t}**" if z["center"] else t)
        A(f"| −{x * 100:.0f}% | " + " | ".join(cells) + " |")
    A("")
    A("**같은 표, 칸 = 2026 수익 · 2026 타이밍 값(같은 평균 노출 고정 대비)**")
    A("")
    A("| 손절 \\ 재매수 | " + " | ".join(f"−{d * 100:.0f}%" for d in NB_D) + " |")
    A("|---|" + "---|" * len(NB_D))
    for x in NB_X:
        cells = []
        for dd in NB_D:
            z = next(z for z in nbr if z["kind"] == "XD" and abs(z["v1"] - x) < 1e-9 and abs(z["v2"] - dd) < 1e-9)
            t = f"{z['2026수익'] * 100:+.0f} · {z['2026타이밍'] * 100:+.0f}"
            cells.append(f"**{t}**" if z["center"] else t)
        A(f"| −{x * 100:.0f}% | " + " | ".join(cells) + " |")
    A("")
    A("| 파는 몫 / 되돌림 여유 | 앞 · 뒤 · 2025 효율선초과 | 2026 수익/낙폭 | 2026 타이밍 값 | 뒤 수익 | 2025 수익 |")
    A("|---|---|---|---|---|---|")
    for kind, lab in (("q", "파는 몫"), ("b", "되돌림 여유")):
        for z in [z for z in nbr if z["kind"] == kind]:
            nm = f"{lab} {z['v1'] * 100:.0f}%" + (" (후보)" if z["center"] else "")
            A(f"| {nm} | {pct(z['앞'])} · {pct(z['뒤'])} · {pct(z['2025'])} | {pct(z['2026수익'])}/{pct(z['2026낙폭'])} | "
              f"{pct(z['2026타이밍'])} | {pct(z['뒤수익'])} | {pct(z['2025수익'])} |")
    A("")
    A("설계 선택 대안(이웃 수에 넣지 않음):")
    A("")
    A("| 대안 | 2026 수익/낙폭 | 앞 · 뒤 · 2025 효율선초과 | 2026 타이밍 값 |")
    A("|---|---|---|---|")
    for dr in P["design"]:
        a_, h_ = dr["res"]["2026"], dr["res"]["hold"]
        A(f"| {dr['name']} | {pct(a_['전체']['수익'])}/{pct(a_['전체']['낙폭'])} | {pct(a_['앞']['효율선초과'])} · {pct(a_['뒤']['효율선초과'])} · "
          f"{pct(h_['2025']['효율선초과'])} | {pct(a_['전체']['타이밍값'])} |")
    A("")
    # 위약
    A("## 5. 위약: 같은 노출 경로를 날짜만 원형 이동")
    A("")
    A("후보의 노출 경로(판 날 0.45~0.5, 나머지 1)를 창 안에서 원형으로 밀었다. 노출 분포·빠짐 길이는 같고 '언제 빠졌나'만 달라진다. "
      "p = (1 + 실제 이상인 위약 수)/(1 + 위약 수). 시드 20개가 주 판정, 가능한 이동 전부(창 일수 − 1개)가 보조다. "
      "같은 평균 노출 고정 보유 수익도 함께 적었다.")
    A("")
    A("| 창 | 평균 노출 | 같은 노출 고정 수익 | 지표 | 실제 | 위약 중앙(시드 20) | 위약 상위 10% | p 시드 20 | p 전수 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for wn, v in P["placebo"].items():
        for m in ("수익", "효율선초과", "낙폭"):
            s20, al = v["시드20"][m], v["전수"][m]
            A(f"| {wn} | {v['평균노출']:.2f} | {pct(v['같은노출_고정_수익'])} | {m} | {pct(s20['실제'])} | {pct(s20['위약_중앙'])} | "
              f"{pct(al['위약_상위10%'])} | {s20['p']:.2f} | {al['p']:.2f} ({al['이상개수']}/{al['개수']}) |")
    A("")
    A("선택 보정(첫 실행 뒤 추가한 보조 점검): PORT3 P1 격자 36개의 노출 경로를 같은 이동으로 함께 밀어 '36개 중 최대 효율선초과'의 위약 분포와 비교했다. "
      "후보는 54개 중 앞 효율선초과 최대로 뽑혔으므로 단독 p보다 이 p가 맞는 비교다.")
    A("")
    A("| 창 | 이동 수 | 실제 최대(변형) | 후보 실제 | 위약 최대 중앙 · 상위 10% | p(최대값) | 후보 단독 p | 단독 p ≤ 0.05인 변형 | 실제 > 0인 변형 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for wn, v in P["grid_maxstat"].items():
        q_, x_, d_ = v["실제_최대_변형"]
        vn = f"{q_:.0%}·−{x_:.0%}·" + (f"−{d_:.0%}" if d_ is not None else "되돌림만")
        A(f"| {wn} | {v['이동수']} | {pct(v['실제_최대'])} ({vn}) | {pct(v['후보_실제'])} | {pct(v['위약최대_중앙'])} · {pct(v['위약최대_상위10%'])} | "
          f"{v['p_최대값']:.2f} | {v['후보_p_단독']:.2f} | {v['p≤0.05_변형수']}/{v['격자수']} | {v['실제>0_변형수']}/{v['격자수']} |")
    A("")
    # 장세
    A("## 6. 장세")
    A("")
    A("7종목 경로(2026: 현행 규칙, 2023-05~2025: 주간 PDF 보조 경로, 이어 돌림):")
    A("")
    A("| 창 | 수익/낙폭 (기본) | 효율선초과 | 타이밍 값 | 주기 | 승률 | 손익비 | 기대값 | 기대값 t | 깊은 재매수 비율 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for bn, ws in (("2026", ("전체", "앞", "뒤", "AI쇼크")), ("hold", ("2025", "2024", "2023(5/22~)"))):
        for w in ws:
            s_ = P["res"][bn][w]
            A(f"| {w if bn == 'hold' else '2026 ' + w} | {pct(s_['수익'])}/{pct(s_['낙폭'])} ({pct(s_['기본_수익'])}/{pct(s_['기본_낙폭'])}) | "
              f"{pct(s_['효율선초과'])} | {pct(s_['타이밍값'])} | {s_['주기']} | {pct(s_['승률'], 0, False)} | {num(s_['손익비'])} | "
              f"{pct(s_['기대값'], 2)} | {num(s_['기대값_t'])} | {pct(s_['깊은재매수비율'], 0, False)} |")
    A("")
    A("대리 포트폴리오(같은 규칙, 해마다 1월 첫 시가에 보유로 시작·고점도 그해 첫 종가부터, 배당 수정 가격, QUBT 제외):")
    A("")
    A("| 대리 | 해 | 수익/낙폭 (기본) | 효율선초과 | 타이밍 값 | 주기 | 승률 | 손익비 | 기대값 | 깊은 재매수 비율 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for Y in (2018, 2022, 2020, 2025, 2026):
        for nm in PROXIES:
            pr = P["proxy"][f"{nm}_{Y}"]
            keys = ["2026", "2026앞", "2026뒤"] if Y == 2026 else [str(Y)]
            for kk in keys:
                s_ = pr[kk]
                A(f"| {nm} | {kk} | {pct(s_['수익'])}/{pct(s_['낙폭'])} ({pct(s_['기본_수익'])}/{pct(s_['기본_낙폭'])}) | "
                  f"{pct(s_['효율선초과'])} | {pct(s_['타이밍값'])} | {s_['주기']} | {pct(s_['승률'], 0, False)} | {num(s_['손익비'])} | "
                  f"{pct(s_['기대값'], 2)} | {pct(s_['깊은재매수비율'], 0, False)} |")
    A("")
    pos = {}
    for nm in PROXIES:
        for Y in PROXY_YEARS:
            for kk, s_ in P["proxy"][f"{nm}_{Y}"].items():
                if kk.startswith("_") or kk in ("2026앞", "2026뒤"):
                    continue
                pos.setdefault("n", 0)
                pos["n"] = pos["n"] + 1
                pos["exc"] = pos.get("exc", 0) + int(s_["효율선초과"] > 0)
                pos["tim"] = pos.get("tim", 0) + int(s_["타이밍값"] > 0)
                pos["ev"] = pos.get("ev", 0) + int(np.isfinite(s_["기대값"]) and s_["기대값"] > 0)
    A(f"대리 4개 × 2018~2026 {pos['n']}개 연도: 효율선초과 + {pos['exc']}개, 타이밍 값 + {pos['tim']}개, 주기 기대값 + {pos['ev']}개.")
    A("")
    A("그림자 −8% 이탈 구간의 깊이(이탈 뒤 새 고점 전까지 최저 종가, 고점 대비):")
    A("")
    for bn, lab in (("2026", "2026 7종목"), ("hold", "2023-05~2025 7종목")):
        e = P["episodes"][bn]
        A(f"- {lab}: {len(e)}번 — " + ", ".join(f"{z['이탈일'][2:]} {pct(z['깊이'], 0)}" for z in e))
    A("")
    # 주기
    A("## 7. 손절 주기 승률·손익비 (후보, 이익 = 판 값/되산 값 − 1)")
    A("")
    A("| 창 | 주기 | 승률 | 평균이익 | 평균손실 | 손익비 | 기대값(조각 수) | 기대값(비중 가중) | 기대값 t | 깊은 재매수 비율 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for bn, w in (("2026", "전체"), ("2026", "앞"), ("2026", "뒤"), ("hold", "2025"), ("hold", "2024")):
        s_ = P["res"][bn][w]
        A(f"| {('2026 ' + w) if bn == '2026' else w} | {s_['주기']} | {pct(s_['승률'], 0, False)} | {pct(s_['평균이익'])} | {pct(s_['평균손실'])} | "
          f"{num(s_['손익비'])} | {pct(s_['기대값'], 2)} | {pct(s_['기대값_가중'], 2)} | {num(s_['기대값_t'])} | {pct(s_['깊은재매수비율'], 0, False)} |")
    A("")
    A("2026 주기: " + "; ".join(f"{c['exit_day'][5:]}→{(c['re_day'] or '열림')[5:]} {pct(c['gain'])} ({c['kind']})" for c in P["cycles_2026"]))
    A("")
    A("2025 주기: " + "; ".join(f"{c['exit_day'][5:]}→{(c['re_day'] or '열림')[5:]} {pct(c['gain'])} ({c['kind']})" for c in P["cycles_2025"]))
    A("")
    A("## 8. 해석·선택편향·한계")
    A("")
    for s_ in NOTES:
        A(f"- {s_}")
    if not NOTES:
        A("(작성 전)")
    A("")
    A("## 점검")
    A("")
    for k, v in P["checks"].items():
        A(f"- {k}: {v:.2e}")
    (RESULTS / f"{ID}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
