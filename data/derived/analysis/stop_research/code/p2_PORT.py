"""2단계 갈래 PORT: 사용자 7종목 포트폴리오 위에 '가격선 손절 + 진정·깊은 하락 재매수'를 얹는다 (2026-10-10).

사용자 아이디어(원문 요지): 특정 가격 라인 이탈 시 기계적 매도, 다시 사는 건 변동성(위아래 진폭)이 진정됐을 때,
또는 고점 대비 −40% 같은 깊은 자리에서 기계적으로 다시 산다. 승률이 30%라도 손익비가 크면 된다.

P1 종목별 가격선 손절(장중 스탑) + 진폭 진정 재진입 (H.StockEngine)
   손절선 L(k일 장중, k-1까지 정보): 직전 n일 최저 저가 / 샹들리에 = 산 뒤 최고 고가 − m×ATR22(k-1까지).
   주중 재진입(k일 시가, k-1 종가까지 정보):
     calm  = 10일 평균 진폭 ≤ 1.2 × 252일 중앙 진폭 그리고 종가 > 직전 종가 (진폭 = (고가−저가)/전일 종가, p2_long 일봉)
     brk   = 종가 > 그 앞 20일 최고 고가(20일 고가 돌파)
     cash  = 주중 재진입 없음(다음 재조정까지 현금)
   재조정일(주 첫 거래일)에는 기본 규칙대로 다시 맞춘다(목표에 남아 있으면 되산다).
P2 포트폴리오 가격선(그림자 = 기본 경로 종가 평가 base.close_s, t 종가 → t+1 시가)
   이탈: 그림자 < 직전 20일 최저 종가 / 그림자 < SMA20 / < SMA50(이평은 아래로 '건너갈 때'만, 위로 돌아오면 다시 장전)
   이탈 시 노출 0 또는 0.5. 재진입: rvol10/rvol60 < 0.8 또는 < 1.0(진폭 진정) / 선 회복(그림자 > SMA_n, 20일 최저선은 SMA20).
P3 포트폴리오 깊은 하락 기계 재매수
   그림자가 (재)진입 뒤 최고 종가 대비 −X(10·15%)면 현금. 그림자 252일 고점 대비 −D(25·30·35·40%) 이하면
   다음 시가 기계 매수(들고 있으면 계속 보유)하고 252일 신고가까지 손절 끔. 그 밖의 복귀: 신고가 회복 / SMA20 / SMA50 회복.
   주의: 2026 그림자 최대 낙폭 −22.9%라 D ≥ 25%는 2026에 한 번도 발동하지 않는다(2024 −32.3%·2025 −42.9%에서만).

평가(사전 고정): 2026 전체·앞(1/2~5/29 고르기)·뒤(6/1~10/9 확인)·AI쇼크(6/22~7/29), 2025(보조, 2023-05부터 이어 돌린 경로를 자름),
  2024(맥락). 효율선초과 = 규칙 수익 − (같은 창에서 기본 경로 × 고정 노출 e(0~1, 0.005 간격) 중 낙폭이 규칙보다 깊지 않은 것의 최대 수익).
  손절 주기 = 판 가격 → 다시 산 가격(P1 종목 가격, P2·P3 그림자 시가 경로), 이익 = 판/되산 − 1.
  고르기: G1 앞 효율선초과 최대, G2 앞 주기 기대값 최대(앞 주기 ≥ 30), G3 앞 칼마(수익/|낙폭|) 최대. 동점은 격자 순서 앞.
실행: cd code && PYTHONIOENCODING=utf-8 py p2_PORT.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402
import stock_lab as SL  # noqa: E402

ID = "p2_PORT"
RESULTS = HERE.parent / "results"

# ============================================================================ 격자(사전 고정 — 뒤 구간·2025를 본 뒤 바꾸지 않는다)
P1_STOPS = [("low", 10), ("low", 20), ("chand", 3.0), ("chand", 4.0)]
P1_REENTRY = ["cash", "calm", "brk", "calm_or_brk"]
P1_CALM_MULT = 1.2
P1_AMP_SHORT, P1_AMP_LONG, P1_AMP_MINP = 10, 252, 60
P1_BRK_N = 20
P1_ATR_N = 22

P2_LINES = [("low", 20), ("sma", 20), ("sma", 50)]
P2_OUT_E = [0.0, 0.5]
P2_REENTRY = [("calm", 0.8), ("calm", 1.0), ("line", None)]
P2_VOL = (10, 60)

P3_X = [0.10, 0.15]
P3_D = [0.25, 0.30, 0.35, 0.40]
P3_NORMAL = [("high", None), ("sma", 20), ("sma", 50)]
P3_ROLL = 252

WARM_N = 120              # 2026 그림자 지표 예열: 보유 기간 경로 끝 120일을 1/2 종가에 맞춰 붙임(SMA50·rvol60용)
SLIP = 0.005              # 대표 후보 미끄러짐 민감도(손절 체결 −0.5%, 되사기 +0.5%)
MIN_CYCLES_G2 = 30
# 고른 변형의 이웃 점검(사전 선언): P1 손절 모수 ±1칸(저가 n±5, 샹들리에 m±0.5)·진정 배수 1.0/1.5,
#   P2 선 n±5(SMA50은 ±10)·진정 문턱 ±0.1·하루 늦은 체결, P3 X±2.5%p·D±5%p·하루 늦은 체결.
NB_P1_N, NB_P1_M, NB_P1_CALM = 5, 0.5, (1.0, 1.5)
NB_P2_N, NB_P2_N50, NB_P2_TH = 5, 10, 0.1
NB_P3_X, NB_P3_D = 0.025, 0.05

W26 = {"전체": ("2026-01-02", "2026-10-09"), "앞": ("2026-01-02", "2026-05-29"), "뒤": ("2026-06-01", "2026-10-09"),
       "AI쇼크": ("2026-06-22", "2026-07-29"), "7~9월": ("2026-07-01", "2026-09-30")}
WH = {"2025": ("2025-01-01", "2025-12-31"), "2024": ("2024-01-01", "2024-12-31"),
      "관세": ("2025-02-18", "2025-04-04"), "24여름": ("2024-06-18", "2024-08-05")}
WINS = {"2026": W26, "hold": WH}

STOP_KO = {"low": "직전 {n}일 최저 저가", "chand": "샹들리에 {n}×ATR22"}
RE_KO = {"cash": "재조정까지 현금", "calm": "진폭 진정(10일≤1.2×252일 중앙)+양봉 종가", "brk": "20일 고가 돌파",
         "calm_or_brk": "진폭 진정 또는 20일 고가 돌파"}

# 첫 실행 뒤 채우는 결론·맥락·한계 문장(격자·계산과 무관, md 전용)
CONCLUSION: list[str] = [
    "**기각.** 7종목 위에 얹은 58개 중, 앞 구간에서 같은 낙폭의 고정 노출을 이긴 6개는 뒤 구간과 2025에서 하나도 이기지 못했다"
    "(0/6; 무조건 비율은 뒤 7%, 2025 38%). 앞 구간으로 고른 대표 G1(그림자 SMA20 하향 → 전량 현금, SMA20 회복 시 복귀)은 "
    "뒤 −12.9%(기본 +4.6%), 2026 +138.0%(기본 +223.7%), 2025 +0.4%(기본 +42.2%)였다.",
    "**2026 손절은 승률도 손익비도 낮았다.** 고른 변형 4개의 2026 손절 주기는 승률 10~50%, 손익비 0.03~0.8이었다. 기대값은 모두 마이너스였다"
    "(묶음 합산 앞 −3.4~−9.0%, 뒤 −1.8~−7.2%). '승률 30%라도 손익비가 크다'는 모양(승률 29~38%, 손익비 2.5~5.2)은 "
    "2025 P3에서만 나왔다. 그 이익도 2025-03~04 폭락 바닥 근처의 매수 한두 번에서 나왔다.",
    "**아이디어별 결과.** ① 종목 가격선 손절: 판 종목의 89~98%를 같은 주나 다음 주 재조정에서 다시 샀다. 그래서 16개 모두 7월 손실이 "
    "기본(−18.7%)보다 컸다(−21.0~−24.0%). ② 진폭 진정 재진입: 주중에 켜진 것은 주기의 0~22%뿐이었다. 나머지는 다음 주 재조정이 되샀다. "
    "③ 고점 대비 −25~−40% 기계 재매수: 2026 7종목 그림자 최대 낙폭이 −22.9%라 한 번도 발동하지 않았다. 2024~25에는 바닥 근처를 잡았다. "
    "하지만 그 앞에 붙는 −10·−15% 손절이 2026 V자 하락에서 2026 수익을 +27~+128%로 깎았다(복귀 규칙에 따라).",
]
CONTEXT: list[str] = [
    "1단계 비교 기준: 기본 2026 +223.7%/−20.3%, 뒤 +4.6%, 7월 쇼크 −18.7%, 2025 +42.2%/−47.1%. "
    "이번 58개 가운데 2026 전체 수익이 기본보다 높은 것은 2개(P2)뿐이었다. 7월 손실이 기본보다 작은 것은 P2 16/18, P3 4/24, P1 0/16이었다.",
    "2025(보조): 종목 손절 P1은 8/16이 효율선을 이겼다(G2 2025 +64.7%/−40.8%). P3 깊은 재매수는 14/24가 이겼다. "
    "2025는 몇 주씩 이어진 하락이라 손절이 통했다. 1단계와 같은 장세 의존이다.",
    "2024(맥락): P3 대표(−10% 손절, −25% 기계 매수, SMA20 복귀)는 2024 +83.4%로 기본 +33.4%보다 높았다. "
    "2024-08-02(−28.2%)에 산 것이 60거래일 뒤 +40.2%였다.",
    "긴 역사(맥락, 1단계 결과 인용): ^SOX가 고점 대비 −40%에 닿은 5번은 60일 뒤 평균 −11.3%였다. 80%는 −10% 더 빠졌다(2001·2008형 약세장). "
    "−40% 기계 재매수가 2025에 좋았던 것은 바닥이 −42.9%에서 멈췄기 때문이다. 바닥이 −38%였으면 발동하지 않았고, −60%였으면 일찍 산 것이 된다.",
    "−40% 아이디어를 손절 없이 쓰려면(예: 현금 일부를 남겨 두었다가 −40%에 투입) 별도 시험이 필요하다. 이번 격자에는 없다.",
]
LIMITS: list[str] = [
    "시험 범위: 사전 격자 58(P1 16·P2 18·P3 24)이다. 고른 변형의 사전 선언 이웃 12개와 미끄러짐 0.5% 재측정 4개를 더 쟀다. "
    "사후 설명용으로 '기계 매수 없음' 분해 6개와 깊은 매수 사건 표를 2025 결과를 본 뒤 추가했다(후보·고르기에 쓰지 않음).",
    "고르기: G1과 G3은 같은 변형이다. G2(앞 주기 ≥ 30)는 P1 직전 10일 저가 4개만 자격이 있었다. "
    "포트폴리오 단위 P2·P3는 5개월에 주기가 2~4번뿐이다. G2가 고른 변형도 앞 기대값이 −3.3%였다(자격 4개 모두 마이너스).",
    "G1의 앞 우위(효율선초과 +37.1%)는 2~3월 이탈 4번에서 나왔고 날카롭다. SMA25로 바꾸면 같고, SMA15면 −9.0%, 하루 늦게 체결하면 −25.0%다. "
    "미끄러짐 0.5%를 넣으면 2026 +109.0%, 뒤 −20.4%다.",
    "생존편향: 7종목 경로는 그때의 TIME 보유(PDF)로 만들어 종목 선택 편향은 작다. 하지만 2024~25는 AI 강세장 안의 조정이었다. "
    "그래서 깊은 하락 매수가 좋게 나오는 쪽으로 기운다. 2000~02·2008·2022형 장기 약세장은 7종목 경로에 없다.",
    "2025 창은 2023-05-22부터 이어 돌린 보유 기간 경로를 자른 값이다(상태가 이어짐). 이 기간은 미분류 56종목이 반도체 예산에 들어가 "
    "실제 7종목과 구성이 다르다(1단계 한계와 같음).",
    "2026 그림자 SMA50·rvol60 예열에 보유 기간 경로 끝 120일(주간 PDF 경로)을 1/2 종가에 맞춰 붙였다. 1~3월 지표값에 영향이 있다.",
    "P1 주기 종료 규칙: 주중 재진입가, 다음 재조정 재매수가, 목록에서 빠지면 그 재조정 시가(기본도 그날 정리)로 닫는다. "
    "P2·P3 주기는 그림자 시가 경로로 잰다. 노출 0.5 이탈은 가격 비만 적었고 실제 손익 영향은 절반이다.",
    "점검: 정책 없는 엔진이 기본을 재현했다(최대 차 1.7e-9, 비중 소수 8자리 반올림 때문에 허용치를 1e-9→1e-6으로 고침, 결과를 보기 전). "
    "미래 가격을 흔들어도 그 전 노출·가치는 같았다(0). 환율은 무시(USD)했고 수수료·현금 이자는 0이다.",
]


def grid():
    out = []
    for kind, n in P1_STOPS:
        for re in P1_REENTRY:
            out.append({"fam": "P1", "stop": kind, "n": n, "re": re, "calm": P1_CALM_MULT})
    for kind, n in P2_LINES:
        for oe in P2_OUT_E:
            for rk, rv in P2_REENTRY:
                out.append({"fam": "P2", "line": kind, "n": n, "out_e": oe, "re": rk, "th": rv, "lag": 1})
    for X in P3_X:
        for D in P3_D:
            for nk, nn in P3_NORMAL:
                out.append({"fam": "P3", "X": X, "D": D, "normal": nk, "nn": nn, "lag": 1})
    return out


def name(sp):
    if sp["fam"] == "P1":
        s = STOP_KO[sp["stop"]].format(n=sp["n"] if sp["stop"] == "low" else f"{sp['n']:g}")
        re = RE_KO[sp["re"]]
        if sp["re"] in ("calm", "calm_or_brk") and abs(sp["calm"] - P1_CALM_MULT) > 1e-9:
            re = re.replace("1.2×", f"{sp['calm']:g}×")
        return f"P1 장중 {s} / {re}"
    if sp["fam"] == "P2":
        ln = f"20일 최저 종가 이탈" if sp["line"] == "low" else f"SMA{sp['n']} 하향"
        if sp["line"] == "low" and sp["n"] != 20:
            ln = f"{sp['n']}일 최저 종가 이탈"
        re = (f"rvol10/60<{sp['th']:.1f}" if sp["re"] == "calm"
              else f"선 회복(>SMA{sp['n'] if sp['line'] == 'sma' else 20})")
        lag = "" if sp.get("lag", 1) == 1 else f" [체결 {sp['lag']}일 뒤]"
        return f"P2 그림자 {ln} → 노출 {sp['out_e']:.1f}, 재진입 {re}{lag}"
    nk = {"high": "신고가 회복", "sma": f"SMA{sp['nn']} 회복"}[sp["normal"]]
    lag = "" if sp.get("lag", 1) == 1 else f" [체결 {sp['lag']}일 뒤]"
    deep = "기계 매수 없음" if sp["D"] is None else f"−{sp['D']:.0%}에 기계 매수"
    return f"P3 그림자 −{sp['X']:.1%} 손절, {deep}, 그 밖 복귀 {nk}{lag}"


# ============================================================================ P1: 종목별 장중 손절
ENG: dict = {}
AMP: dict = {}


def engine(bn):
    if bn not in ENG:
        ENG[bn] = H.StockEngine(bn)
    return ENG[bn]


def amp_arrays(bn):
    """종목별 (10일 평균 진폭, 252일 중앙 진폭)을 장부 날짜에 맞춘 배열. [j] = j일 종가까지의 값(정책은 k-1을 읽음)."""
    if bn not in AMP:
        eng, data = engine(bn), SL.load()
        out = {}
        for key, tk in eng.tk.items():
            bars = data[tk]
            rp = pd.Series(bars.range_pct(), index=bars.days)
            m10 = rp.rolling(P1_AMP_SHORT, min_periods=P1_AMP_SHORT).mean()
            med = rp.rolling(P1_AMP_LONG, min_periods=P1_AMP_MINP).median()
            out[key] = (m10.reindex(eng.book.days).to_numpy(float), med.reindex(eng.book.days).to_numpy(float))
        AMP[bn] = out
    return AMP[bn]


def _run_locals():
    f = sys._getframe(1)
    while f is not None and not (f.f_code.co_name == "run" and "w" in f.f_locals and "target" in f.f_locals):
        f = f.f_back
    return f.f_locals


class P1Policy(H.NoPolicy):
    def __init__(self, eng, amp, sp):
        self.eng, self.amp, self.sp = eng, amp, sp
        self.exp: dict = {}
        self.buyw: dict = {}            # k → 그날 손절 종목을 되산 비중 합(미끄러짐 비용용)
        self.stopped_keys: set = set()
        self._reb_k = None
        self._atr: dict = {}

    def atr(self, key, k):
        if (key, k) not in self._atr:
            self._atr[(key, k)] = self.eng.atr(key, k, P1_ATR_N)
        return self._atr[(key, k)]

    def level(self, key, k, st):
        if self.sp["stop"] == "low":
            n = int(self.sp["n"])
            a = self.eng.L[key][max(0, k - n):k]
            a = a[np.isfinite(a)]
            return float(a.min()) if len(a) >= n - 2 else float("nan")
        a = self.atr(key, k)
        return st["high_peak"] - self.sp["n"] * a if np.isfinite(a) else float("nan")

    def allow(self, key, k, st, eng):                 # 재조정일: 기본 규칙대로 다시 맞춤(손절 종목도 목표면 되산다)
        loc = _run_locals()
        target = loc["target"]
        if self._reb_k != k:
            self._reb_k = k
            self.stopped_keys &= set(target)
        if key in self.stopped_keys and key not in loc["w"]:
            self.buyw[k] = self.buyw.get(k, 0.0) + float(target[key])
            self.stopped_keys.discard(key)
        return True

    def reenter(self, key, k, st, eng):
        re = self.sp["re"]
        if re == "cash" or k <= st.get("stopped_k", k):
            return False
        j = k - 1
        C = eng.C[key]
        ok = False
        if re in ("calm", "calm_or_brk"):
            m10, med = self.amp[key]
            if (np.isfinite(m10[j]) and np.isfinite(med[j]) and np.isfinite(C[j]) and np.isfinite(C[j - 1])
                    and m10[j] <= self.sp["calm"] * med[j] and C[j] > C[j - 1]):
                ok = True
        if not ok and re in ("brk", "calm_or_brk"):
            hh = eng.H[key][max(0, j - P1_BRK_N):j]
            hh = hh[np.isfinite(hh)]
            if len(hh) >= P1_BRK_N - 2 and np.isfinite(C[j]) and C[j] > hh.max():
                ok = True
        if ok:
            loc = _run_locals()
            add = min(loc["target"][key], 1 - sum(loc["w"].values()))
            if add > 1e-9 and np.isfinite(eng.O[key][k]):
                self.buyw[k] = self.buyw.get(k, 0.0) + float(add)
                self.stopped_keys.discard(key)
        return ok

    def stop(self, key, k, st, eng):
        loc = _run_locals()
        self.exp[k] = float(sum(loc["w"].values()))
        L = self.level(key, k, st)
        if not np.isfinite(L):
            return None
        o, lo = eng.O[key][k], eng.L[key][k]
        if k == st["entry_k"] and np.isfinite(o) and o <= L:   # 산 시가가 이미 선 아래면 그날은 주문하지 않음
            return None
        if (np.isfinite(o) and o <= L) or (np.isfinite(lo) and lo <= L):
            self.stopped_keys.add(key)
        return L


def p1_exposure(eng, pol):
    b = eng.b
    vals = []
    for d, wt in zip(b.days, b.weights):
        bs = sum(wt.values())
        vals.append(1.0 if bs < 1e-9 else min(1.0, pol.exp.get(eng.k_of[d], 0.0) / bs))
    return pd.Series(vals, index=b.days)


def p1_cycles(eng, events, slip=0.0):
    """손절 → (주중 재진입 | 다음 재조정 재매수 | 목록 탈락: 그 재조정 시가 | 자료 끝: 마지막 종가)."""
    b = eng.b
    inv = {v: k for k, v in eng.tk.items()}
    dayset = set(b.days)
    reb = sorted(d for d in eng.reb if d in dayset)
    idx = {d: i for i, d in enumerate(b.days)}
    re_ev: dict = {}
    for e in events:
        if e["type"] == "reenter":
            re_ev.setdefault(e["key"], []).append((e["day"], e["px"]))
    out = []
    for e in events:
        if e["type"] != "stop":
            continue
        d, tk = e["day"], e["key"]
        key = inv[tk]
        nr = next((x for x in reb if x > d), None)
        rr = next((x for x in re_ev.get(tk, []) if x[0] > d), None)
        if rr is not None and (nr is None or rr[0] < nr):
            re_day, re_px, how = rr[0], rr[1] * (1 + slip), "주중 재진입"
        elif nr is not None:
            px = float(eng.O[key][eng.k_of[nr]])
            if key in b.weights[idx[nr]]:
                re_day, re_px, how = nr, px * (1 + slip), "재조정 재매수"
            else:
                re_day, re_px, how = nr, px, "목록 탈락(기본도 이날 정리)"
        else:
            re_day, re_px, how = None, float(eng.C[key][eng.k_of[b.days[-1]]]), "자료 끝"
        out.append({"exit_day": d, "tk": tk, "exit_px": float(e["px"]), "re_day": re_day, "re_px": float(re_px),
                    "how": how, "gain": float(e["px"]) / float(re_px) - 1})
    return out


def run_p1(sp, bn, slip=0.0, eng=None):
    eng = eng or engine(bn)
    pol = P1Policy(eng, amp_arrays(bn) if eng is engine(bn) else eng._amp, sp)
    r, ev = eng.run(pol, slip=slip)
    if slip:
        inv_k = {k: d for d, k in eng.k_of.items()}
        adj = pd.Series(0.0, index=r.index)
        for k, wv in pol.buyw.items():
            d = inv_k[k]
            if d in adj.index:
                adj[d] += slip * wv
        r = (1 + r) * (1 - adj) - 1
    return r, p1_exposure(eng, pol), p1_cycles(eng, ev, slip), ev


# ============================================================================ P2·P3: 그림자 상태 기계
def shadow_signals(bn, close_override: pd.Series | None = None):
    """t일 종가에 아는 값만(index = t)."""
    b = H.base(bn)
    c = (close_override if close_override is not None else b.close_s).copy()
    if bn == "2026":
        hc = H.base("hold").close_s
        pre = hc[hc.index < b.days[0]].iloc[-WARM_N:]
        ext = pd.concat([pre * (c.iloc[0] / pre.iloc[-1]), c])
    else:
        ext = c
    s = pd.DataFrame(index=b.days)
    s["c"] = c
    for n in sorted({15, 20, 25, 40, 50, 60}):
        s[f"sma{n}"] = ext.rolling(n, min_periods=n).mean().reindex(b.days)
        s[f"low{n}"] = ext.shift(1).rolling(n, min_periods=n).min().reindex(b.days)     # 직전 n일 최저 종가(t 제외)
    lr = np.log(ext).diff()
    s["rv_s"] = lr.rolling(P2_VOL[0], min_periods=P2_VOL[0]).std().reindex(b.days)
    s["rv_l"] = lr.rolling(P2_VOL[1], min_periods=P2_VOL[1]).std().reindex(b.days)
    s["dd"] = c / c.rolling(P3_ROLL, min_periods=1).max() - 1
    return s


def sim_p2(sig, sp):
    c = sig.c.to_numpy()
    line = sig[f"{sp['line']}{sp['n']}"].to_numpy()
    rec = sig[f"sma{sp['n'] if sp['line'] == 'sma' else 20}"].to_numpy()
    rs, rl = sig.rv_s.to_numpy(), sig.rv_l.to_numpy()
    mode, armed = "in", True
    expo = np.ones(len(c))
    for i in range(len(c)):
        if mode == "in":
            if not np.isfinite(line[i]):
                trig = False
            elif sp["line"] == "sma":
                if c[i] >= line[i]:
                    armed = True
                trig = armed and c[i] < line[i]
            else:
                trig = c[i] < line[i]
            if trig:
                mode, armed = "out", False
        else:
            if sp["re"] == "calm":
                back = np.isfinite(rs[i]) and np.isfinite(rl[i]) and rl[i] > 0 and rs[i] / rl[i] < sp["th"]
            else:
                back = np.isfinite(rec[i]) and c[i] > rec[i]
            if back:
                mode = "in"
                armed = bool(np.isfinite(line[i]) and c[i] >= line[i])
        expo[i] = 1.0 if mode == "in" else sp["out_e"]
    return pd.Series(expo, index=sig.index), []


def sim_p3(sig, sp):
    c, dd = sig.c.to_numpy(), sig.dd.to_numpy()
    rec = sig[f"sma{sp['nn']}"].to_numpy() if sp["normal"] == "sma" else None
    X, D = sp["X"], (sp["D"] if sp["D"] is not None else np.inf)     # D=None: 기계 매수 없음(사후 분해용)
    mode, peak = "in", c[0]
    expo = np.ones(len(c))
    deep = []
    for i in range(len(c)):
        if mode == "in":
            peak = max(peak, c[i])
            if dd[i] <= -D:
                mode = "keep"
                deep.append({"day": sig.index[i], "kind": "보유 중 깊은 구간 → 손절 끔", "dd": float(dd[i])})
            elif c[i] / peak - 1 <= -X:
                mode = "out"
        elif mode == "out":
            if dd[i] <= -D:
                mode = "keep"
                deep.append({"day": sig.index[i], "kind": "현금 → 기계 매수", "dd": float(dd[i])})
            else:
                back = dd[i] >= -1e-9 if sp["normal"] == "high" else (np.isfinite(rec[i]) and c[i] > rec[i])
                if back:
                    mode, peak = "in", c[i]
        else:
            if dd[i] >= -1e-9:
                mode, peak = "in", c[i]
        expo[i] = 0.0 if mode == "out" else 1.0
    return pd.Series(expo, index=sig.index), deep


def port_cycles(b, e, slip=0.0):
    """노출(시가 기준)이 1→낮게 바뀐 날 판 것, 다시 1이 된 날 되산 것. 가격 = 그림자 시가 경로 b.open."""
    o = np.asarray(b.open)
    ev = e.to_numpy()
    out, xi = [], None
    for k in range(1, len(ev)):
        if xi is None and ev[k - 1] >= 0.999 and ev[k] < 0.999:
            xi = k
        elif xi is not None and ev[k] >= 0.999 and ev[k - 1] < 0.999:
            ex, re_ = o[xi] * (1 - slip), o[k] * (1 + slip)
            out.append({"exit_day": b.days[xi], "re_day": b.days[k], "exit_px": float(ex), "re_px": float(re_),
                        "out_e": float(ev[xi]), "days_out": k - xi, "gain": float(ex / re_ - 1)})
            xi = None
    if xi is not None:
        ex, last = o[xi] * (1 - slip), float(np.asarray(b.close)[-1])
        out.append({"exit_day": b.days[xi], "re_day": None, "exit_px": float(ex), "re_px": last,
                    "out_e": float(ev[xi]), "days_out": len(ev) - 1 - xi, "gain": float(ex / last - 1)})
    return out


SIG: dict = {}


def run_port(sp, bn, slip=0.0):
    if bn not in SIG:
        SIG[bn] = shadow_signals(bn)
    b = H.base(bn)
    ec, deep = (sim_p2 if sp["fam"] == "P2" else sim_p3)(SIG[bn], sp)
    e = H.at_open(ec, b)
    if sp.get("lag", 1) == 2:
        e = e.shift(1).fillna(1.0)
    r = H.overlay(b, e, cost=slip)
    return r, e, port_cycles(b, e, slip), deep


# ============================================================================ 지표
def wstats(r: pd.Series, e: pd.Series, b, cycles: list, lo: str, hi: str) -> dict:
    m = (r.index >= lo) & (r.index <= hi)
    x, br = r[m].to_numpy(), b.ret[m].to_numpy()
    cur = np.concatenate([[1.0], np.cumprod(1 + x)])
    ret, mdd = float(cur[-1] - 1), SL._mdd(cur)
    exc, e_eq = SL.same_mdd_excess(cur, br)
    bcur = np.concatenate([[1.0], np.cumprod(1 + br)])
    cyc = [cy for cy in cycles if lo <= cy["exit_day"] <= hi]
    g = np.array([cy["gain"] for cy in cyc]) if cyc else np.array([])
    win, loss = g[g > 0], g[g <= 0]
    return {"수익": ret, "낙폭": mdd, "기본_수익": float(bcur[-1] - 1), "기본_낙폭": SL._mdd(bcur),
            "효율선초과": exc, "효율선_e": e_eq, "칼마": ret / abs(mdd) if mdd < -1e-9 else float("inf"),
            "평균노출": float(e.reindex(r.index)[m].mean()),
            "주기": int(len(g)), "승률": float(len(win) / len(g)) if len(g) else float("nan"),
            "평균이익": float(win.mean()) if len(win) else float("nan"),
            "평균손실": float(loss.mean()) if len(loss) else float("nan"),
            "손익비": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
            "기대값": float(g.mean()) if len(g) else float("nan")}


def run_variant(sp, slip=0.0, keep=False):
    res = {"name": name(sp) + (f" (미끄러짐 {slip:.1%})" if slip else ""), "spec": sp}
    det = {}
    for bn in ("2026", "hold"):
        b = H.base(bn)
        if sp["fam"] == "P1":
            r, e, cyc, ev = run_p1(sp, bn, slip)
            extra = ev
        else:
            r, e, cyc, extra = run_port(sp, bn, slip)
        res[bn] = {w: wstats(r, e, b, cyc, lo, hi) for w, (lo, hi) in WINS[bn].items()}
        res[bn]["빠짐횟수"] = int(((e.shift(1).fillna(1.0) >= 0.999) & (e < 0.999)).sum())
        if sp["fam"] == "P1":
            res[bn]["주기처리"] = {w: pd.Series([c["how"] for c in cyc if lo <= c["exit_day"] <= hi]).value_counts().to_dict()
                               for w, (lo, hi) in WINS[bn].items() if w in ("앞", "뒤", "2025")}
        if keep:
            det[bn] = {"r": r, "e": e, "cycles": cyc, "extra": extra}
    return (res, det) if keep else res


def base_row():
    res = {"name": "기본(7종목 현행)", "spec": {"fam": "base"}}
    for bn in ("2026", "hold"):
        b = H.base(bn)
        e = pd.Series(1.0, index=b.days)
        res[bn] = {w: wstats(b.ret, e, b, [], lo, hi) for w, (lo, hi) in WINS[bn].items()}
        res[bn]["빠짐횟수"] = 0
    return res


# ============================================================================ 점검
def checks():
    out = {}
    for bn in ("2026", "hold"):
        eng = engine(bn)
        r, _ = eng.run(H.NoPolicy())
        out[f"엔진재현_{bn}"] = float(np.max(np.abs(r.to_numpy() - H.base(bn).ret.to_numpy())))
    # P2·P3 시점: T 이후 그림자를 흔들어도 T+1 시가까지 노출이 같아야 한다
    b = H.base("2026")
    T = 100
    rng = np.random.default_rng(1)
    c2 = b.close_s.copy()
    c2.iloc[T + 1:] = c2.iloc[T + 1:].to_numpy() * (1 + rng.normal(0, 0.05, len(c2) - T - 1))
    s1, s2 = shadow_signals("2026"), shadow_signals("2026", c2)
    worst = 0.0
    for sp in grid():
        if sp["fam"] == "P1":
            continue
        f = sim_p2 if sp["fam"] == "P2" else sim_p3
        e1, e2 = H.at_open(f(s1, sp)[0], b), H.at_open(f(s2, sp)[0], b)
        worst = max(worst, float(np.max(np.abs(e1.to_numpy()[:T + 2] - e2.to_numpy()[:T + 2]))))
    out["그림자_미래교란_노출차"] = worst
    # P1 시점: K일 시가 이후 가격·진폭을 흔들어도 K일 시가까지 가치가 같아야 한다
    eng = engine("2026")
    e2 = copy.copy(eng)
    K = eng.k_of[b.days[100]]
    rng = np.random.default_rng(2)
    for attr in ("O", "C", "H", "L"):
        d = {}
        for key, arr in getattr(eng, attr).items():
            a = arr.copy()
            a[K + 1:] = a[K + 1:] * (1 + rng.normal(0, 0.05, len(a) - K - 1))
            d[key] = a
        setattr(e2, attr, d)
    for key in eng.keys:          # 고가 ≥ 시가·종가, 저가 ≤ 시가·종가
        e2.H[key][K + 1:] = np.fmax.reduce([e2.H[key][K + 1:], e2.O[key][K + 1:], e2.C[key][K + 1:]])
        e2.L[key][K + 1:] = np.fmin.reduce([e2.L[key][K + 1:], e2.O[key][K + 1:], e2.C[key][K + 1:]])
    amp = amp_arrays("2026")
    e2._amp = {k: (np.concatenate([v[0][:K + 1], v[0][K + 1:] * 1.7]), np.concatenate([v[1][:K + 1], v[1][K + 1:] * 0.6]))
               for k, v in amp.items()}
    worst = 0.0
    for sp in grid():
        if sp["fam"] != "P1":
            continue
        r1 = run_p1(sp, "2026")[0]
        pol = P1Policy(e2, e2._amp, sp)
        r2, _ = e2.run(pol)
        v1, v2 = np.cumprod(1 + r1.to_numpy()), np.cumprod(1 + r2.to_numpy())
        worst = max(worst, float(np.max(np.abs(v1[:99] - v2[:99]))))      # r[i]은 시가 i→i+1, 100일 시가까지
    out["종목_미래교란_가치차"] = worst
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
        if sp["stop"] == "low":
            out += [dict(sp, n=sp["n"] + d) for d in (-NB_P1_N, NB_P1_N)]
        else:
            out += [dict(sp, n=sp["n"] + d) for d in (-NB_P1_M, NB_P1_M)]
        if sp["re"] in ("calm", "calm_or_brk"):
            out += [dict(sp, calm=c) for c in NB_P1_CALM]
    elif sp["fam"] == "P2":
        step = NB_P2_N50 if sp["n"] == 50 else NB_P2_N
        out += [dict(sp, n=sp["n"] + d) for d in (-step, step)]
        if sp["re"] == "calm":
            out += [dict(sp, th=round(sp["th"] + d, 2)) for d in (-NB_P2_TH, NB_P2_TH)]
        out.append(dict(sp, lag=2))
    else:
        out += [dict(sp, X=round(sp["X"] + d, 4)) for d in (-NB_P3_X, NB_P3_X)]
        out += [dict(sp, D=round(sp["D"] + d, 4)) for d in (-NB_P3_D, NB_P3_D)]
        out.append(dict(sp, lag=2))
    return out


def brief(r):
    a, h = r["2026"], r["hold"]
    return {"name": r["name"],
            "2026": {k: a[k]["수익"] for k in W26}, "2026_낙폭": a["전체"]["낙폭"], "뒤_낙폭": a["뒤"]["낙폭"],
            "앞_효율선초과": a["앞"]["효율선초과"], "뒤_효율선초과": a["뒤"]["효율선초과"], "2026_효율선초과": a["전체"]["효율선초과"],
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
        return {k: clean(v) for k, v in o.items()}
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


def main():
    chk = checks()
    print("점검", chk)
    assert chk["엔진재현_2026"] < 1e-6 and chk["엔진재현_hold"] < 1e-6      # 기본 비중 소수 8자리 저장 반올림(갈래 B와 같은 허용)
    assert chk["그림자_미래교란_노출차"] == 0.0 and chk["종목_미래교란_가치차"] < 1e-12

    br = base_row()
    specs = grid()
    print("격자", len(specs), {f: sum(s["fam"] == f for s in specs) for f in ("P1", "P2", "P3")})
    rows = []
    for sp in specs:
        r = run_variant(sp)
        rows.append(r)
        a, h = r["2026"], r["hold"]
        print(f"{r['name'][:70]:<70} 앞 {a['앞']['수익']:+.0%}/{a['앞']['낙폭']:.1%} 초과 {a['앞']['효율선초과']:+.1%} | "
              f"뒤 {a['뒤']['수익']:+.1%}/{a['뒤']['낙폭']:.1%} 초과 {a['뒤']['효율선초과']:+.1%} | AI {a['AI쇼크']['수익']:+.1%} | "
              f"25 {h['2025']['수익']:+.1%}/{h['2025']['낙폭']:.1%} 초과 {h['2025']['효율선초과']:+.1%} | "
              f"앞주기 {a['앞']['주기']} 승 {a['앞']['승률']:.0%} 기대 {a['앞']['기대값']:+.1%}")

    F = lambda r: r["2026"]["앞"]
    picks = {"G1": pick(rows, lambda r: F(r)["효율선초과"]),
             "G2": pick(rows, lambda r: F(r)["기대값"], lambda r: F(r)["주기"] >= MIN_CYCLES_G2),
             "G3": pick(rows, lambda r: F(r)["칼마"])}
    fam_g1 = {f: pick([r for r in rows if r["spec"]["fam"] == f], lambda r: F(r)["효율선초과"]) for f in ("P1", "P2", "P3")}
    n_g2 = sum(F(r)["주기"] >= MIN_CYCLES_G2 for r in rows)

    # 고르기에 정보가 있나: 앞 효율선초과 > 0 집합의 뒤·2025 효율선초과 > 0 비율 vs 무조건
    def share(sub, f):
        return (sum(f(r) for r in sub), len(sub))
    posF = [r for r in rows if F(r)["효율선초과"] > 0]
    info = {"앞>0 개수": len(posF), "전체 개수": len(rows),
            "뒤>0 | 앞>0": share(posF, lambda r: r["2026"]["뒤"]["효율선초과"] > 0),
            "뒤>0 무조건": share(rows, lambda r: r["2026"]["뒤"]["효율선초과"] > 0),
            "2025>0 | 앞>0": share(posF, lambda r: r["hold"]["2025"]["효율선초과"] > 0),
            "2025>0 무조건": share(rows, lambda r: r["hold"]["2025"]["효율선초과"] > 0),
            "둘다>0 | 앞>0": share(posF, lambda r: r["2026"]["뒤"]["효율선초과"] > 0 and r["hold"]["2025"]["효율선초과"] > 0),
            "둘다>0 무조건": share(rows, lambda r: r["2026"]["뒤"]["효율선초과"] > 0 and r["hold"]["2025"]["효율선초과"] > 0)}
    info_fam = {}
    for f in ("P1", "P2", "P3"):
        sub = [r for r in rows if r["spec"]["fam"] == f]
        pf = [r for r in sub if F(r)["효율선초과"] > 0]
        info_fam[f] = {"개수": len(sub), "앞>0": len(pf),
                       "뒤>0 | 앞>0": share(pf, lambda r: r["2026"]["뒤"]["효율선초과"] > 0),
                       "뒤>0 무조건": share(sub, lambda r: r["2026"]["뒤"]["효율선초과"] > 0),
                       "2025>0 | 앞>0": share(pf, lambda r: r["hold"]["2025"]["효율선초과"] > 0),
                       "2025>0 무조건": share(sub, lambda r: r["hold"]["2025"]["효율선초과"] > 0),
                       "2026전체 기본보다 수익↑": share(sub, lambda r: r["2026"]["전체"]["수익"] > br["2026"]["전체"]["수익"]),
                       "AI쇼크 기본보다 덜 잃음": share(sub, lambda r: r["2026"]["AI쇼크"]["수익"] > br["2026"]["AI쇼크"]["수익"])}

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
        trace = H.trace(e26, "2026-06-15", "2026-09-30") if sp["fam"] != "P1" else None
        july = [c for c in cyc26 if "2026-06-15" <= c["exit_day"] <= "2026-09-30"]
        deep = {bn: det[bn]["extra"] for bn in ("2026", "hold")} if sp["fam"] == "P3" else None
        hold_cyc = [c for c in det["hold"]["cycles"] if "2024-01-01" <= c["exit_day"] <= "2025-12-31"]
        pick_detail[g] = {"row": r, "slip": rs, "neighbors": [brief(x) for x in nb], "trace_2026_6_15_9_30": trace,
                          "cycles_2026": cyc26, "cycles_july_sep": july, "cycles_2024_2025": hold_cyc, "deep": deep,
                          "how_counts_2026": pd.Series([c.get("how", "그림자") for c in cyc26]).value_counts().to_dict()
                          if cyc26 else {}}

    # 사후 분해(후보 아님 — 뒤·2025를 본 뒤 설명용으로 추가): P3에서 기계 매수만 뺀 경우(D 없음)와 나란히
    decomp = []
    for X in P3_X:
        for nk, nn in P3_NORMAL:
            r0 = run_variant({"fam": "P3", "X": X, "D": None, "normal": nk, "nn": nn, "lag": 1})
            row = {"X": X, "normal": nk, "nn": nn, "기계매수없음": brief(r0)}
            for D in P3_D:
                rr = next(r for r in rows if r["spec"]["fam"] == "P3" and r["spec"]["X"] == X and r["spec"]["D"] == D
                          and r["spec"]["normal"] == nk and r["spec"]["nn"] == nn)
                row[f"{D:.2f}"] = brief(rr)
            decomp.append(row)
    # 깊은 구간 매수 사건(보유 기간 경로): 신호일 낙폭, 체결(다음 시가) 뒤 20·60거래일 기본 경로 수익, 60일 안 추가 하락
    hb = H.base("hold")
    ho, hc = np.asarray(hb.open), np.asarray(hb.close)
    hidx = {d: i for i, d in enumerate(hb.days)}
    deep_ev: dict = {}
    for sp in specs:
        if sp["fam"] != "P3":
            continue
        for d in run_port(sp, "hold")[3]:
            key = (d["day"], d["kind"])
            if key not in deep_ev:
                i = hidx[d["day"]] + 1
                j20, j60 = min(i + 20, len(ho) - 1), min(i + 60, len(ho) - 1)
                deep_ev[key] = {"신호일": d["day"], "종류": d["kind"], "낙폭": d["dd"], "체결일": hb.days[i],
                                "20일뒤": float(ho[j20] / ho[i] - 1), "60일뒤": float(ho[j60] / ho[i] - 1),
                                "60일안_추가하락": float(hc[i:j60 + 1].min() / ho[i] - 1), "60일_실제거래일": int(j60 - i),
                                "D": set(), "변형수": 0}
            deep_ev[key]["D"].add(sp["D"])
            deep_ev[key]["변형수"] += 1
    deep_tab = sorted(({**v, "D": sorted(v["D"])} for v in deep_ev.values()), key=lambda x: (x["신호일"], x["종류"]))

    # 출력
    print("\n기본", brief(br))
    for row in decomp:
        print("분해", row["X"], row["normal"], row["nn"], "없음:", {k: round(v, 3) for k, v in
              (("26", row["기계매수없음"]["2026"]["전체"]), ("뒤", row["기계매수없음"]["2026"]["뒤"]),
               ("25", row["기계매수없음"]["2025"]), ("24", row["기계매수없음"]["2024"]))},
              {D: round(row[f"{D:.2f}"]["2025"], 3) for D in P3_D})
    for x in deep_tab:
        print("깊은매수", x)
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
            print("  주기 처리", d["how_counts_2026"])
            if d["trace_2026_6_15_9_30"]:
                print("  노출 변화", d["trace_2026_6_15_9_30"])
            for c in d["cycles_july_sep"]:
                print("   ", c)
            if d["deep"]:
                print("  깊은 구간", d["deep"])

    payload = {"id": ID, "grid_size": len(specs), "grid_counts": {f: sum(s["fam"] == f for s in specs) for f in ("P1", "P2", "P3")},
               "checks": chk, "base": br, "picks": {g: (r["name"] if r else None) for g, r in picks.items()},
               "fam_g1": {f: (r["name"] if r else None) for f, r in fam_g1.items()}, "g2_eligible": n_g2,
               "info": info, "info_fam": info_fam, "pick_detail": pick_detail, "rows": rows,
               "p3_decomposition_posthoc": decomp, "p3_deep_events_hold": deep_tab,
               "params": {"P1_CALM_MULT": P1_CALM_MULT, "P1_AMP": [P1_AMP_SHORT, P1_AMP_LONG, P1_AMP_MINP], "P1_BRK_N": P1_BRK_N,
                          "P1_ATR_N": P1_ATR_N, "P2_VOL": P2_VOL, "P3_ROLL": P3_ROLL, "WARM_N": WARM_N, "SLIP": SLIP}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{ID}.json").write_text(json.dumps(clean(payload), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_md(payload)
    print("저장", RESULTS / f"{ID}.json", RESULTS / f"{ID}.md")


# ============================================================================ md
def write_md(P):
    br, rows = P["base"], P["rows"]
    L = []
    A = L.append
    A("# 2단계 PORT: 7종목 포트폴리오에 가격선 손절 + 진정·깊은 하락 재매수 얹기 (2026-10-10)")
    A("")
    A("> 코드 `code/p2_PORT.py`, 전 변형 `results/p2_PORT.json`. USD, 시가→시가, 수수료 0, 현금 이자 0.")
    A("")
    A("## 결론")
    A("")
    for i, s in enumerate(CONCLUSION, 1):
        A(f"{i}. {s}")
    if not CONCLUSION:
        A("(작성 전)")
    A("")
    A("## 격자 (사전 고정, 합계 %d)" % P["grid_size"])
    A("")
    gc = P["grid_counts"]
    A(f"- P1 종목별 장중 손절 {gc['P1']} = 손절선 4(직전 10·20일 최저 저가, 샹들리에 3·4×ATR22) × 주중 재진입 4"
      f"(재조정까지 현금, 진폭 진정[10일 평균 진폭 ≤ 1.2×252일 중앙 + 종가>직전 종가], 20일 고가 돌파, 진정 또는 돌파).")
    A(f"- P2 포트폴리오 가격선 {gc['P2']} = 이탈선 3(그림자 20일 최저 종가 이탈, SMA20 하향, SMA50 하향) × 이탈 노출 2(0, 0.5) × "
      f"재진입 3(rvol10/rvol60 < 0.8, < 1.0, 선 회복).")
    A(f"- P3 깊은 하락 기계 재매수 {gc['P3']} = 손절 X 2(그림자 −10·−15%) × 기계 매수 D 4(252일 고점 대비 −25·−30·−35·−40%) × "
      f"그 밖 복귀 3(신고가 회복, SMA20, SMA50).")
    A("- 고른 변형마다 사전 선언한 이웃(모수 ±1칸, 하루 늦은 체결)과 미끄러짐 0.5%를 추가로 쟀다(격자 수에 넣지 않음).")
    A("")
    A("## 변수 시점")
    A("")
    A("| 변수 | 언제 아는가 | 체결 |")
    A("|---|---|---|")
    A("| 종목 손절선(직전 n일 최저 저가, 산 뒤 최고 고가 − m×ATR22) | k−1일 종가까지 | k일 장중 스탑(시가가 선 아래면 시가) |")
    A("| 종목 진폭 진정(10일 평균·252일 중앙 (고가−저가)/전일 종가), 20일 고가 돌파 | k−1일 종가 | k일 시가 |")
    A("| 7종목 그림자(모델 종가 평가)·SMA·20일 최저 종가·rvol10/60·252일 고점 낙폭 | t일 종가 | t+1일 시가 |")
    A("| 주간 목표(7종목 규칙) | 재조정일 시가(기본 경로와 같음) | 그날 시가 |")
    A("")
    A("## 고르기 결과 (앞 1/2~5/29로 고르고 뒤 6/1~10/9·2025로 확인)")
    A("")
    A("| 고르기 | 변형 | 2026 수익/낙폭 | 앞 수익 · 효율선초과 | 뒤 수익/낙폭 · 효율선초과 | AI쇼크 | 2025 수익/낙폭 · 효율선초과 |")
    A("|---|---|---|---|---|---|---|")

    def line(lab, r):
        a, h = r["2026"], r["hold"]
        return (f"| {lab} | {r['name']} | {pct(a['전체']['수익'])}/{pct(a['전체']['낙폭'])} | "
                f"{pct(a['앞']['수익'])} · {pct(a['앞']['효율선초과'])} | {pct(a['뒤']['수익'])}/{pct(a['뒤']['낙폭'])} · "
                f"{pct(a['뒤']['효율선초과'])} | {pct(a['AI쇼크']['수익'])} | {pct(h['2025']['수익'])}/{pct(h['2025']['낙폭'])} · "
                f"{pct(h['2025']['효율선초과'])} |")
    A(line("기본", br))
    byname = {r["name"]: r for r in rows}
    for g in ("G1", "G2", "G3"):
        n = P["picks"][g]
        A(line(g, byname[n]) if n else f"| {g} | 해당 없음(앞 주기 ≥ {MIN_CYCLES_G2}인 변형 {P['g2_eligible']}개) | | | | | |")
    for f in ("P1", "P2", "P3"):
        n = P["fam_g1"][f]
        if n:
            A(line(f"{f} 안 G1", byname[n]))
    A("")
    A("고르기에 정보가 있나(앞 효율선초과 > 0 집합 vs 무조건):")
    A("")
    A("| 묶음 | 앞>0 개수 | 뒤>0 (앞>0 중) | 뒤>0 (무조건) | 2025>0 (앞>0 중) | 2025>0 (무조건) |")
    A("|---|---|---|---|---|---|")
    fr = lambda t: f"{t[0]}/{t[1]}" + (f" ({t[0] / t[1]:.0%})" if t[1] else "")
    i = P["info"]
    A(f"| 전체 {i['전체 개수']} | {i['앞>0 개수']} | {fr(i['뒤>0 | 앞>0'])} | {fr(i['뒤>0 무조건'])} | {fr(i['2025>0 | 앞>0'])} | {fr(i['2025>0 무조건'])} |")
    for f, v in P["info_fam"].items():
        A(f"| {f} {v['개수']} | {v['앞>0']} | {fr(v['뒤>0 | 앞>0'])} | {fr(v['뒤>0 무조건'])} | {fr(v['2025>0 | 앞>0'])} | {fr(v['2025>0 무조건'])} |")
    A("")
    A("## 손절 주기 승률·손익비 (이익 = 판 가격/되산 가격 − 1, 판 날 기준 창)")
    A("")
    A("| 변형 | 창 | 주기 | 승률 | 평균이익 | 평균손실 | 손익비 | 기대값 |")
    A("|---|---|---|---|---|---|---|---|")
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
              f"{pct(s['평균손실'])} | {num(s['손익비'])} | {pct(s['기대값'], 2)} |")
    # 묶음 전체 주기(같은 묶음 모든 변형의 주기를 합침)
    A("")
    A("묶음별 전 변형 주기 합산(변형 간 중복 사건 포함, 분포 감 잡기용):")
    A("")
    A("| 묶음 | 창 | 주기 | 승률 | 손익비 | 기대값 |")
    A("|---|---|---|---|---|---|")
    for f in ("P1", "P2", "P3"):
        sub = [r for r in rows if r["spec"]["fam"] == f]
        for bn, w in (("2026", "앞"), ("2026", "뒤"), ("hold", "2025")):
            n = sum(r[bn][w]["주기"] for r in sub)
            if not n:
                A(f"| {f} | {w if bn == '2026' else '2025'} | 0 | - | - | - |")
                continue
            wins = sum(r[bn][w]["승률"] * r[bn][w]["주기"] for r in sub if r[bn][w]["주기"])
            ev = sum(r[bn][w]["기대값"] * r[bn][w]["주기"] for r in sub if r[bn][w]["주기"]) / n
            pw = [r[bn][w]["손익비"] for r in sub if r[bn][w]["주기"] and np.isfinite(r[bn][w]["손익비"])]
            A(f"| {f} | {w if bn == '2026' else '2025'} | {n} | {wins / n:.0%} | {num(float(np.median(pw))) if pw else '-'} (중앙) | {pct(ev, 2)} |")
    A("")
    A("## 대표 후보(G1) 점검")
    A("")
    g1 = P["pick_detail"].get("G1")
    if g1 and "row" in g1:
        s = g1["slip"]
        A(f"- 미끄러짐 0.5%: 2026 {pct(s['2026']['전체']['수익'])}/{pct(s['2026']['전체']['낙폭'])}, 뒤 {pct(s['2026']['뒤']['수익'])} "
          f"(효율선초과 {pct(s['2026']['뒤']['효율선초과'])}), 2025 {pct(s['hold']['2025']['수익'])} (효율선초과 {pct(s['hold']['2025']['효율선초과'])}).")
        A("- 이웃:")
        for x in g1["neighbors"]:
            A(f"  - {x['name']}: 2026 {pct(x['2026']['전체'])}, 앞 효율선초과 {pct(x['앞_효율선초과'])}, 뒤 {pct(x['2026']['뒤'])} "
              f"(효율선초과 {pct(x['뒤_효율선초과'])}), AI쇼크 {pct(x['2026']['AI쇼크'])}, 2025 {pct(x['2025'])} (효율선초과 {pct(x['2025_효율선초과'])})")
        if g1["trace_2026_6_15_9_30"]:
            A(f"- 6/15~9/30 노출 변화: {', '.join(f'{d} {v:g}' for d, v in g1['trace_2026_6_15_9_30'])}")
        if g1["cycles_july_sep"]:
            A("- 6/15~9/30 손절 주기: " + "; ".join(
                f"{c['exit_day']} {c.get('tk', '그림자')} → {c['re_day']} ({pct(c['gain'])})" for c in g1["cycles_july_sep"][:14]))
    A("")
    A("## P1 손절 뒤 실제로 어떻게 되샀나 (2026 앞+뒤 주기 수)")
    A("")
    A("| 변형 | 주중 재진입 | 재조정 재매수 | 목록 탈락 | 자료 끝 |")
    A("|---|---|---|---|---|")
    for r in rows:
        if r["spec"]["fam"] != "P1":
            continue
        hc = {}
        for w in ("앞", "뒤"):
            for k, v in r["2026"]["주기처리"].get(w, {}).items():
                hc[k] = hc.get(k, 0) + v
        A(f"| {r['name']} | {hc.get('주중 재진입', 0)} | {hc.get('재조정 재매수', 0)} | {hc.get('목록 탈락(기본도 이날 정리)', 0)} | {hc.get('자료 끝', 0)} |")
    A("")
    A("## 사용자 아이디어 '−40%에 기계 재매수' 분해 (사후 설명용, 후보 아님)")
    A("")
    A("같은 손절·복귀에서 기계 매수만 뺀 경우(없음)와 D별 2025 수익. 2026은 그림자 최대 낙폭 −22.9%라 D와 무관하게 '없음'과 같다.")
    A("")
    A("| 손절 / 그 밖 복귀 | 2026 (없음) | 뒤 (없음) | 2025 없음 | 2025 −25% | −30% | −35% | −40% | 2024 없음 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for row in P["p3_decomposition_posthoc"]:
        nk = "신고가" if row["normal"] == "high" else f"SMA{row['nn']}"
        z = row["기계매수없음"]
        A(f"| −{row['X']:.0%} / {nk} | {pct(z['2026']['전체'])} | {pct(z['2026']['뒤'])} | {pct(z['2025'])} | "
          + " | ".join(pct(row[f"{D:.2f}"]["2025"]) for D in P3_D) + f" | {pct(z['2024'])} |")
    A("")
    A("깊은 구간 매수 사건(보유 기간 경로, 신호 종가 → 다음 시가 체결):")
    A("")
    A("| 신호일 | 종류 | 그때 낙폭 | 발동한 D | 20거래일 뒤 | 60거래일 뒤 | 60일 안 추가 하락 |")
    A("|---|---|---|---|---|---|---|")
    for x in P["p3_deep_events_hold"]:
        nd = x["60일_실제거래일"]
        cut = "" if nd == 60 else f" ({nd}일, 12/31까지)"
        A(f"| {x['신호일']} | {x['종류']} | {pct(x['낙폭'])} | {', '.join(f'{d:.0%}' for d in x['D'])} | {pct(x['20일뒤'])} | "
          f"{pct(x['60일뒤'])}{cut} | {pct(x['60일안_추가하락'])} |")
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
    (RESULTS / f"{ID}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
