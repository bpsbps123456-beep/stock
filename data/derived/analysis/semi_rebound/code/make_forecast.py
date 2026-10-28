"""2026-10-28 SOXX 반등 탑픽 10 예측 원장 입력을 만든다 (desk_system.md §14).

입력: ../results/decision_table_2026-10-28.csv(점수표), live_2026-10-22.csv(7월 반등), snapshot·fund_flows.
출력: ../results/forecast_2026-10-28.json → `py -m desk forecast record <파일>`로 기록 시각·해시를 찍는다.
점수 규칙(사전 고정): 0.45 전망 상향(30·90일 백분위 평균) + 0.30 베타 + 0.15 반등 초반(10/22→10/27)
+ 0.10 두 펀드 최근 매수. 공매도 15% 초과·시총 50억 달러 미만·90일 전망 하향 제외, 세부 분야당 2개.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
d = pd.read_csv(RES / "decision_table_2026-10-28.csv", index_col=0)
live = pd.read_csv(RES / "live_2026-10-22.csv", index_col=0)

pct = lambda s: s.rank(pct=True)
rev = (pct(d["rev30"].fillna(d["rev30"].median())) + pct(d["rev90"].fillna(d["rev90"].median()))) / 2
# 7월 반등에서 베타로 설명되는 몫을 뺀 순위 잔차(7월 '주도력' 자체)
jr = pd.concat([live["july_rebound"].rank(), live["beta"].rank()], axis=1, keys=["j", "b"]).dropna()
fit = pd.Series(jr["j"]).cov(jr["b"]) / jr["b"].var()
july_net = jr["j"] - (jr["j"].mean() + fit * (jr["b"] - jr["b"].mean()))

r4 = lambda v: None if pd.isna(v) else round(float(v), 4)
universe = {t: {"group": live.loc[t, "group"], "sub": d.loc[t, "sub"], "beta": r4(d.loc[t, "beta"]),
                "rev": r4(rev[t]), "rev30": r4(d.loc[t, "rev30"]), "rev90": r4(d.loc[t, "rev90"]),
                "early": r4(d.loc[t, "early"]), "october_drop": r4(d.loc[t, "drop"]),
                "july": r4(live.loc[t, "july_rebound"]), "july_net": r4(july_net.get(t)),
                "si": r4(d.loc[t, "si"]), "mcap_busd": r4(d.loc[t, "mcap"]), "fpe": r4(d.loc[t, "fpe"]),
                "score": r4(d.loc[t, "score"]), "eligible": bool(d.loc[t, "eligible"])}
            for t in d.index}

PICKS = ["MU", "WDC", "MRVL", "FORM", "ALAB", "LRCX", "STX", "AMD", "MTSI", "ASX"]
RULE = ["FORM", "MU", "MRVL", "ALAB", "WDC", "SIMO", "LRCX", "STX", "MTSI", "ASX"]
JULY = list(live["july_rebound"].sort_values(ascending=False).index[:10])
BETA = list(live["beta"].sort_values(ascending=False).index[:10])
EXCLUDED = ["NVDA", "AVGO", "SNDK", "AXTI", "TSM"]
beta_of = lambda ts: round(sum(universe[t]["beta"] for t in ts) / len(ts), 4)
baskets = {
    "picks": {"label": "탑픽 10", "tickers": PICKS, "beta": beta_of(PICKS)},
    "rule10": {"label": "점수 규칙 그대로 10", "tickers": RULE, "beta": beta_of(RULE),
               "note": "내 판단 교체(SIMO→AMD) 전"},
    "july10": {"label": "7월 반등 상위 10", "tickers": JULY, "beta": beta_of(JULY),
               "note": "7/29→8/17 반등 상위. '7월 주도주를 다시 산다'의 대조군"},
    "beta10": {"label": "베타 상위 10", "tickers": BETA, "beta": beta_of(BETA),
               "note": "직전 답변의 반등 민감도 순위"},
    "excluded5": {"label": "주요 제외 5", "tickers": EXCLUDED, "beta": beta_of(EXCLUDED),
                  "note": "일부러 뺀 종목(아래 제외 이유)"},
    "all43": {"label": "반도체 43종목", "tickers": list(d.index), "beta": beta_of(list(d.index))},
}

S = lambda t, k: universe[t][k]
pc = lambda v: f"{v * 100:+.1f}%".replace("-", "−")
picks = [
    {"ticker": "MU", "theme": "D램·HBM 가격 사이클",
     "thesis": "10월 급락 중에도 이익 전망이 가장 꾸준히 올라간 대형 메모리. 한 달 안에 실적 발표가 없어 '좋은 실적에도 판다' 위험을 비켜 간다.",
     "reasons": [f"다음 회계연도 EPS 컨센서스 30일 {pc(S('MU','rev30'))}(상향 19·하향 1), 90일 {pc(S('MU','rev90'))} — 급락은 이익이 아니라 PER이 줄어든 것",
                 "TrendForce 4분기 D램 계약가 +10~15%(HBM 포함 +15~20%) 전망",
                 f"선행 PER {S('MU','fpe'):.1f}배, 베타 {S('MU','beta'):.2f}, 52주 고점 대비 −23%"],
     "risks": ["메모리 경쟁 심화·사이클 정점 우려(10월 하락 이유 중 하나)", "KoAct가 최근 20거래일 수량 −32% 매도"]},
    {"ticker": "WDC", "theme": "HDD 가격 결정력 · 두 펀드 저점 매수",
     "thesis": "기대 이상 실적·전망 상향 뒤에도 주가가 밀렸고, 두 펀드가 바닥에서 함께 샀다.",
     "reasons": ["10/22 실적 EPS 4.64달러(예상 4.11), 다음 분기 EPS 5.60~6.00달러 전망, 배당 17% 인상, TB당 가격 +28%",
                 f"EPS 컨센서스 30일 {pc(S('WDC','rev30'))}(43종목 1위), 선행 PER {S('WDC','fpe'):.1f}배",
                 "TIME 수량 +60%(D-5), KoAct 10/23 신규 편입"],
     "risks": ["실적 당일 −6.4%처럼 좋은 소식에도 파는 장", "52주 고점 대비 −49% — 정점 우려가 이미 큼"]},
    {"ticker": "MRVL", "theme": "맞춤형 AI 칩(ASIC)",
     "thesis": "애널리스트 상향이 26:1로 한쪽으로 쏠렸고, 이번 주 하이퍼스케일러 실적이 직접 재료다. 한 달 안 실적 발표는 없다(12/1).",
     "reasons": [f"EPS 컨센서스 30일 {pc(S('MRVL','rev30'))}(상향 26·하향 1), 90일 {pc(S('MRVL','rev90'))}",
                 "MSFT·GOOGL(10/28 장 마감 뒤)·AMZN(10/29) 설비투자 발표가 맞춤형 칩 수요의 첫 확인",
                 f"베타 {S('MRVL','beta'):.2f}, 52주 고점 대비 −23%"],
     "risks": ["KoAct 20거래일 수량 −20%", "선행 PER 33배 — 12월 금리 인상 기대에 약함"]},
    {"ticker": "FORM", "theme": "HBM·첨단 프로브카드",
     "thesis": f"43종목 점수 1위(90일 전망 {pc(S('FORM','rev90'))}, 베타 {S('FORM','beta'):.2f}). 오늘 밤 실적이 한 달 성적을 크게 좌우한다.",
     "reasons": [f"EPS 컨센서스 90일 {pc(S('FORM','rev90'))}, 30일 {pc(S('FORM','rev30'))}",
                 f"반등 초반(10/22→10/27) {pc(S('FORM','early'))}로 강함",
                 "3분기 가이던스 EPS 0.77~0.95달러(예상 0.86) — HBM·첨단 패키지 테스트 수요"],
     "risks": ["10/28 장 마감 뒤 실적 — 기준가 바로 뒤의 사건", "시총 83억 달러 소형주, 10월 −36%처럼 양방향 증폭"]},
    {"ticker": "ALAB", "theme": "AI 서버 내부 연결 칩",
     "thesis": f"90일 전망 상향 {pc(S('ALAB','rev90'))}(상향 20·하향 0)로 성장이 확인됐고 반등 민감도가 크다.",
     "reasons": [f"EPS 컨센서스 90일 {pc(S('ALAB','rev90'))}, 30일 상향 20건·하향 0건",
                 f"베타 {S('ALAB','beta'):.2f}, 반등 초반 {pc(S('ALAB','early'))}",
                 "11/3 실적 — 랙 단위 연결(스케일업) 수요 확인"],
     "risks": ["KoAct 20거래일 수량 −69%(사실상 정리)", "선행 PER 45배 — 금리에 가장 민감"]},
    {"ticker": "LRCX", "theme": "메모리 설비투자 수혜 장비",
     "thesis": "메모리 가격이 오르면 메모리 회사가 장비를 더 산다. 전공정 장비 4사 중 메모리 비중이 가장 크다.",
     "reasons": [f"EPS 컨센서스 90일 {pc(S('LRCX','rev90'))}, 30일 {pc(S('LRCX','rev30'))}",
                 f"10월 {pc(S('LRCX','october_drop'))}로 깊게 빠졌고 52주 고점 대비 −44%",
                 f"베타 {S('LRCX','beta'):.2f}, 10/21 실적이 끝나 한 달 안 실적 사건 없음"],
     "risks": ["KoAct 20거래일 수량 −49%", "중국 수출 규제 뉴스"]},
    {"ticker": "STX", "theme": "HDD · TIME의 저점 집중 매수",
     "thesis": "TIME이 실적 직전 바닥에서 수량을 두 배로 늘렸다. 기대 이상 실적에도 시간외 −7%라 기준가가 그 하락 뒤다.",
     "reasons": ["10/27 실적 EPS 7.68달러(예상 7.39), 다음 분기 EPS 9.00~9.40달러 전망(예상 8.61)",
                 f"TIME 수량 +100%(10/23 이후), 선행 PER {S('STX','fpe'):.1f}배",
                 f"EPS 컨센서스 90일 {pc(S('STX','rev90'))} — 가이던스 상향으로 추가 상향이 나올 자리"],
     "risks": ["1년 +192% 뒤 차익 실현", f"베타 {S('STX','beta'):.2f}로 반등 민감도는 중간"]},
    {"ticker": "AMD", "theme": "이번 반등의 직접 촉매(판단으로 교체)",
     "thesis": "점수 규칙은 SIMO를 골랐지만, Anthropic 상장 서류가 AMD 칩 200억 달러 이상 구매를 밝혀 이번 반등을 이끈 종목을 넣었다.",
     "reasons": ["Anthropic S-1(10/26): AMD 칩 200억 달러 이상, MI450 Helios 2027 공급 — 10/27 +5.5%",
                 "11/3 실적 — 4분기 가이던스로 MI450 증산 확인",
                 f"베타 {S('AMD','beta'):.2f}, EPS 컨센서스 90일 {pc(S('AMD','rev90'))}(상향 7·하향 1)"],
     "risks": ["두 펀드가 모두 판다(TIME D-5 −12%, KoAct D-20 −9%)", "내 7종목 규칙은 11/2 재조정에서 AMD를 뺀다(모멘텀 15위)"]},
    {"ticker": "MTSI", "theme": "광·RF 아날로그(데이터센터)",
     "thesis": f"90일 전망 {pc(S('MTSI','rev90'))}, 반등 초반 {pc(S('MTSI','early'))}로 고르게 강하다.",
     "reasons": [f"EPS 컨센서스 90일 {pc(S('MTSI','rev90'))}", f"베타 {S('MTSI','beta'):.2f}, 반등 초반 {pc(S('MTSI','early'))}",
                 "11/5 실적 — 데이터센터 광 연결 수요"],
     "risks": ["시총 195억 달러 중형주", f"30일 상향 {pc(S('MTSI','rev30'))}로 최근 상향은 약함"]},
    {"ticker": "ASX", "theme": "첨단 패키징 후공정(ASE)",
     "thesis": f"급락 때 덜 빠졌고(52주 고점 대비 −12%) 30일 전망 {pc(S('ASX','rev30'))}. 첨단 패키징 병목의 수혜.",
     "reasons": [f"EPS 컨센서스 30일 {pc(S('ASX','rev30'))}", f"반등 초반 {pc(S('ASX','early'))}, 베타 {S('ASX','beta'):.2f}",
                 "10/29 실적 — 첨단 패키징 매출 확인"],
     "risks": ["대만 ADR — 환율·대만 휴장 영향", f"90일 상향 {pc(S('ASX','rev90'))}로 중간"]},
]

claims = [
    # 시장
    {"id": "m1", "group": "시장", "kind": "benchmark_return", "op": ">", "value": 0.0, "probability": 0.60,
     "text": "한 달 뒤(11/27 종가) SOXX가 기준가(10/28 종가)보다 높다"},
    {"id": "m2", "group": "시장", "kind": "benchmark_min_close", "op": ">", "value": 475.69, "probability": 0.75,
     "text": "한 달 동안 SOXX 종가가 10/22 저점 475.69 아래로 내려가지 않는다"},
    {"id": "m3", "group": "시장", "kind": "benchmark_return", "op": ">", "value": 0.10, "probability": 0.30,
     "text": "SOXX가 한 달에 +10% 넘게 오른다(본격 반등)"},
    {"id": "m4", "group": "시장", "kind": "manual", "probability": 0.85,
     "text": "MSFT·GOOGL(10/28)·AMZN(10/29)이 AI 설비투자 계획을 낮추지 않는다",
     "how": "세 회사 실적 발표의 설비투자 가이던스를 뉴스로 확인"},
    # 바스켓
    {"id": "b1", "group": "탑픽 성적", "kind": "basket_excess", "basket": "picks", "op": ">", "value": 0.0, "probability": 0.60,
     "text": "탑픽 10(같은 금액)이 SOXX보다 더 오른다"},
    {"id": "b2", "group": "탑픽 성적", "kind": "basket_excess", "basket": "picks", "adjust": "beta", "op": ">", "value": 0.0,
     "probability": 0.55, "text": f"베타({baskets['picks']['beta']:.2f}배)를 감안해도 SOXX를 이긴다 — 레버리지가 아니라 종목 선택 효과"},
    {"id": "b3", "group": "탑픽 성적", "kind": "basket_excess", "basket": "picks", "versus": "all43", "op": ">", "value": 0.0,
     "probability": 0.55, "text": "반도체 43종목 같은 금액 바스켓보다 더 오른다"},
    {"id": "b4", "group": "탑픽 성적", "kind": "basket_excess", "basket": "picks", "versus": "july10", "adjust": "beta", "op": ">",
     "value": 0.0, "probability": 0.60, "text": "베타를 감안하면 '7월 반등 상위 10'보다 낫다 — 7월 주도주 재매수는 베타 이상을 주지 못한다"},
    {"id": "b5", "group": "탑픽 성적", "kind": "basket_excess", "basket": "picks", "versus": "rule10", "op": ">", "value": 0.0,
     "probability": 0.55, "text": "점수 규칙 그대로(SIMO)보다 내 교체(AMD)가 낫다"},
    {"id": "b6", "group": "탑픽 성적", "kind": "basket_count_beat", "basket": "picks", "op": ">=", "value": 6, "probability": 0.50,
     "text": "10개 중 6개 이상이 SOXX를 이긴다"},
    {"id": "b7", "group": "탑픽 성적", "kind": "basket_excess", "basket": "picks", "versus": "excluded5", "op": ">", "value": 0.0,
     "probability": 0.55, "text": "일부러 뺀 5종목(NVDA·AVGO·SNDK·AXTI·TSM)보다 더 오른다"},
    # 요인(43종목 순위 상관)
    {"id": "f1", "group": "요인", "kind": "factor_rho", "factor": "rev", "op": ">", "value": 0.0, "probability": 0.60,
     "text": "실적 전망 상향 폭(30·90일)이 클수록 한 달 동안 더 오른다 — 핵심 가정"},
    {"id": "f2", "group": "요인", "kind": "factor_rho", "factor": "beta", "op": ">", "value": 0.0, "probability": 0.60,
     "text": "베타가 클수록 더 오른다(반등이 이어지면)"},
    {"id": "f3", "group": "요인", "kind": "factor_rho", "factor": "july_net", "op": "<=", "value": 0.10, "probability": 0.65,
     "text": "7월 반등에서 베타 몫을 뺀 '7월 주도력'은 이번 한 달 수익을 설명하지 못한다(ρ ≤ 0.10)"},
    {"id": "f4", "group": "요인", "kind": "factor_rho", "factor": "si", "op": "<", "value": 0.0, "probability": 0.55,
     "text": "공매도 비율이 높을수록 덜 오른다"},
    {"id": "f5", "group": "요인", "kind": "factor_rho", "factor": "early", "op": ">", "value": 0.0, "probability": 0.55,
     "text": "반등 초반(10/22→10/27)에 많이 오른 종목이 계속 더 오른다"},
]
event = {"MRVL": ("2026-10-28", "2026-10-30", "MSFT·GOOGL·AMZN 실적 뒤 이틀 SOXX보다 더 오른다", 0.55),
         "FORM": ("2026-10-28", "2026-10-29", "10/28 실적 다음 날 SOXX보다 더 오른다", 0.50),
         "ALAB": ("2026-11-03", "2026-11-04", "11/3 실적 다음 날 SOXX보다 더 오른다", 0.50),
         "AMD": ("2026-11-03", "2026-11-04", "11/3 실적 다음 날 SOXX보다 더 오른다", 0.50),
         "MTSI": ("2026-11-04", "2026-11-06", "11/5 실적 앞뒤 이틀 SOXX보다 더 오른다", 0.50),
         "ASX": ("2026-10-28", "2026-10-29", "10/29 실적 날 SOXX보다 더 오른다", 0.50)}
snap = json.loads((RES / "snapshot_2026-10-28.json").read_text(encoding="utf-8"))["tickers"]
eps = {t: snap[t]["eps_trend"]["+1y"]["current"] for t in PICKS}
eps_claim = {"MU": (0.0, 0.75, "다음 회계연도 EPS 컨센서스가 11/27까지 더 오른다"),
             "WDC": (0.0, 0.60, "다음 회계연도 EPS 컨센서스가 11/27까지 더 오른다"),
             "LRCX": (0.0, 0.65, "다음 회계연도 EPS 컨센서스가 11/27까지 더 오른다"),
             "STX": (0.02, 0.70, "가이던스 상향으로 다음 회계연도 EPS 컨센서스가 11/27까지 2% 넘게 오른다")}
xp = {"MU": 0.60, "WDC": 0.60, "MRVL": 0.55, "FORM": 0.50, "ALAB": 0.55, "LRCX": 0.55, "STX": 0.55,
      "AMD": 0.50, "MTSI": 0.50, "ASX": 0.50}
for p in picks:
    t = p["ticker"]
    claims.append({"id": f"x_{t}", "group": "종목", "kind": "ticker_excess", "ticker": t, "op": ">", "value": 0.0,
                   "probability": xp[t], "text": f"{t}가 한 달 동안 SOXX보다 더 오른다"})
    if t in event:
        a, z, text, prob = event[t]
        claims.append({"id": f"e_{t}", "group": "종목", "kind": "event_excess", "ticker": t, "start": a, "end": z,
                       "op": ">", "value": 0.0, "probability": prob, "text": f"{t}: {text}"})
    if t in eps_claim:
        v, prob, text = eps_claim[t]
        claims.append({"id": f"eps_{t}", "group": "종목", "kind": "eps_change", "ticker": t, "base": eps[t],
                       "op": ">", "value": v, "probability": prob, "text": f"{t}: {text}(기록 {eps[t]:.2f})"})

forecast = {
    "id": "2026-10-28-soxx-rebound-top10",
    "title": "SOXX 10월 반등장 탑픽 10",
    "author": "claude",
    "request": "이번 10월 27일부터 시작된 soxx반등장에서 탑픽10개를 고르고 마지막 탭에 예측하는 탭을 만들어줘. 너의 예측 이유를 적고 내가 한달뒤에 실제로 확인해볼게.",
    "benchmark": "SOXX",
    "horizon_sessions": 21,
    "summary": ("10/22 바닥 뒤 세 거래일 +7.4%로 바닥 확인 연구의 문턱(+7~8%)에 막 닿았다. 한 달 뒤 SOXX가 기준가보다 높을 확률을 60%로 본다"
                "(+10% 넘는 본격 반등은 30%). 고른 기준은 '10월 급락 중에도 실적 전망이 올라간 종목 + 반등을 탈 만한 베타'다. "
                "근거는 메모리 계약가 상승(4분기 D램 +10~20%, 낸드 +15~20%), 하이퍼스케일러 설비투자, Anthropic 상장 서류(컴퓨트 약정 5,670억 달러)다. "
                "위험은 12월 금리 인상(시장 약 80% 반영)과 '좋은 실적에도 판다'는 정점 우려다(WDC·STX·TER가 기대 이상 실적에도 6~8% 하락)."),
    "context": [
        "10/16까지 필라델피아 반도체 지수 8거래일 연속 하락(2008년 이후 최장): AI 투자 재평가, 뜨거운 물가·금리, 메모리 경쟁, 높은 밸류에이션",
        "10/26 Anthropic S-1 공개(컴퓨트 약정 최대 5,670억 달러, 브로드컴 장비 해지 불가 약정 1,612억 달러, AMD 칩 200억 달러 이상) → 10/27 SOXX +3.9%",
        "10/28 FOMC 동결 예상(약 90~95%), 12월 인상 약 80% 반영 · MSFT·GOOGL 10/28, AMZN 10/29 실적 · NVDA 11/17 실적",
        "두 펀드: TIME은 최근 5거래일(10/21→10/28) STX(+99%)·WDC(+60%)·SNDK 수량을 늘리고 NVDA·ASML·인텔을 줄임, KoAct는 10/23 WDC·AXTI 신규 편입",
        "2026 네 국면에서 SOXX +7% 확인 뒤 가격만으로 고른 규칙 7개는 모두 3/4 이하(우연 수준) — 종목 선택 우위는 입증되지 않았다",
    ],
    "method_notes": [
        "기준가: 기록 뒤 첫 미국 정규장 종가. 마감: 그 뒤 21거래일째 종가(11/27, 추수감사절 다음 날 조기 폐장, 한국 11/28 03:00)",
        "수익률은 수정 종가(배당 반영), SOXX 저점 비교는 종가. 바스켓은 같은 금액으로 사서 들고 있는 수익(재조정 없음)",
        "베타 조정 초과 = 바스켓 수익 − 바스켓 베타(기록 시점, 고점 전 60거래일 SOXX 대비) × SOXX 수익",
        "선정: 반도체 43종목(watchlist.AI_UNIVERSE 반도체 네 묶음의 미국 상장)에 점수 — 전망 상향 45%·베타 30%·반등 초반 15%·두 펀드 최근 매수 10%. "
        "공매도 15% 초과·시총 50억 달러 미만·90일 전망 하향 제외, 세부 분야당 2개 → 규칙 10에서 SIMO를 AMD로 바꿨다(판단)",
        "근거 자료: data/derived/analysis/semi_rebound/ (report.md, results/entry_rules.json, snapshot_2026-10-28.json)",
        "점검: 11/28(토) 한국 아침 이후 '예측 점검'을 요청하면 탭의 적중·빗나감, 요인 상관, 확률 점수(Brier)를 보고 "
        "크게 빗나간 종목은 뉴스로 이유(실적·돌발 재료)를 확인한다. 중간 점검은 언제든 탭에서 본다",
        "예측 기록이다. 목표 비중·주문 수량이 없고 데스크 7종목 규칙과 무관하다",
    ],
    "exclusions": [
        {"ticker": "NVDA", "reason": "베타 0.52로 반등 민감도가 가장 낮은 축, TIME D-5 수량 −29%. 11/17 실적은 큰 사건이지만 SOXX를 이기기는 어렵다고 봄"},
        {"ticker": "AVGO", "reason": "Anthropic의 최대 공급사지만 30일 전망 상향 13·하향 26으로 하향이 더 많고(고객 대출 420억 달러), 베타 0.60"},
        {"ticker": "SNDK", "reason": "베타 1.98·낸드 가격 상승·TIME 매수에도 30·90일 전망 상향이 +1%·+4%로 작고, 10/29 실적이 '좋은 실적에도 판다' 장에 놓임"},
        {"ticker": "AXTI", "reason": "KoAct 신규 편입·7월 +160%지만 공매도 19%·시총 38억 달러로 규칙 제외"},
        {"ticker": "TSM", "reason": "TIME D-20 수량 +81%로 좋은 신호지만 베타 0.66 — 반등장에서 SOXX를 이기기 어렵다고 봄"},
    ],
    "wrong_if": [
        "SOXX가 10/22 저점 475.69를 종가로 깨면 시장 판단(m2)이 틀린 것",
        "탑픽이 SOXX를 이겨도 베타 조정 초과(b2)가 마이너스면 고른 게 아니라 레버리지였던 것",
        "전망 상향 요인(f1)이 마이너스면 핵심 가정이 틀린 것",
        "7월 상위 10이 베타를 감안하고도 이기면(b4 빗나감) '7월 주도주 재매수'가 이번엔 통한 것",
    ],
    "universe": universe,
    "baskets": baskets,
    "primary_basket": "picks",
    "picks": [{"rank": i + 1, **p} for i, p in enumerate(picks)],
    "claims": claims,
}
assert [p["ticker"] for p in forecast["picks"]] == PICKS
(RES / "forecast_2026-10-28.json").write_text(json.dumps(forecast, ensure_ascii=False, indent=1), encoding="utf-8")
print(len(claims), "claims;", {k: (len(v["tickers"]), v["beta"]) for k, v in baskets.items()})
print("july10", JULY, "\nbeta10", BETA)
