"""2026-10-28 두 번째 예측: 첫 기록(탑픽 10) 뒤 사용자가 물어본 4종목(AXTI·BE·VRT·NBIS)의 당시 판단.

같은 기준가(10/28 종가)·마감(11/27)으로 비교하려고 10/28 미국 장 마감 전에 기록한다.
출력: ../results/forecast_asked_2026-10-28.json → `py -m desk forecast record <파일>`.
전망 수정률은 (지금 − 과거) ÷ |과거|(적자 기업도 개선이면 +). 점수 순위는 첫 기록과 같은 규칙을 45종목에 다시 매긴 값.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[5]
RES = Path(__file__).resolve().parents[1] / "results"
ASKED = ["AXTI", "BE", "VRT", "NBIS"]
first = json.loads((ROOT / "data" / "forecast" / "ledger.json").read_text(encoding="utf-8"))["forecasts"][0]
PICKS = first["baskets"]["picks"]["tickers"]

px = yf.download(ASKED + ["SOXX"], start="2025-06-01", auto_adjust=True, progress=False)["Close"].dropna(how="all")
px = px.loc[:"2026-10-27"]
ret = px.pct_change().loc[:"2026-10-05"].iloc[-60:]
r4 = lambda v: None if v is None or pd.isna(v) else round(float(v), 4)
change = lambda now, then: None if not then else (now - then) / abs(then)
universe, eps = {}, {}
for t in ASKED:
    p = px[t]
    tk = yf.Ticker(t)
    info = tk.info
    y = tk.eps_trend.loc["+1y"]
    eps[t] = float(y["current"])
    universe[t] = {"beta": r4(ret[t].cov(ret["SOXX"]) / ret["SOXX"].var()),
                   "rev30": r4(change(y["current"], y["30daysAgo"])), "rev90": r4(change(y["current"], y["90daysAgo"])),
                   "early": r4(p["2026-10-27"] / p["2026-10-22"] - 1), "october_drop": r4(p["2026-10-22"] / p["2026-10-05"] - 1),
                   "july": r4(p["2026-08-17"] / p["2026-07-29"] - 1), "si": r4(info.get("shortPercentOfFloat")),
                   "mcap_busd": r4((info.get("marketCap") or 0) / 1e9), "fpe": r4(info.get("forwardPE"))}
for t in PICKS:
    universe[t] = {k: first["universe"][t].get(k) for k in ("beta", "rev30", "rev90", "early", "october_drop", "july", "si", "mcap_busd", "fpe")}
U = lambda t, k: universe[t][k]
pc = lambda v: f"{v * 100:+.1f}%".replace("-", "−")
beta_of = lambda ts: round(sum(U(t, "beta") for t in ts) / len(ts), 4)

picks = [
    {"ticker": "AXTI", "theme": "인듐인(InP) 기판 · 광통신 소재",
     "thesis": "첫 기록의 점수로는 2위였지만 공매도·시총 규칙으로 뺐다. 판단: SOXX를 이길 확률은 반반이고, 위아래 폭은 가장 크다.",
     "reasons": [f"다음 회계연도 EPS 전망 90일 {pc(U('AXTI','rev90'))} — 7/30 실적 직후 한 번 뛴 값이고 30일 상향은 {pc(U('AXTI','rev30'))}, 애널리스트 5명",
                 "KoAct 10/23 신규 편입(1.46%), 2분기 InP 매출 사상 최대·InP 생산 능력 2배 계획",
                 f"베타 {U('AXTI','beta'):.2f}(최고), 7월 반등 {pc(U('AXTI','july'))}"],
     "risks": ["공장이 중국: InP 기판은 2025-02부터 중국 수출 통제, 허가 약 60영업일, 미국 고객 허가 미발급(2026 증권신고서)",
               "10/29 장 마감 뒤 실적 — '좋은 실적에도 판다' 장", f"공매도 {pc(U('AXTI','si')).lstrip('+')}, 2026년 주식 발행 신고(424B5)"]},
    {"ticker": "BE", "theme": "연료전지 발전 · AI 데이터센터 전력",
     "thesis": "반도체 밖이라 첫 기록의 후보(43종목)에 없었다. 같은 점수로는 45종목 중 7위라, 넣었다면 탑픽 안이었다. 판단: SOXX보다 조금 나을 쪽.",
     "reasons": ["두 펀드가 최근 5거래일 함께 매수(TIME 수량 +115%, KoAct +147%)",
                 f"베타 {U('BE','beta'):.2f}, EPS 전망 30일 {pc(U('BE','rev30'))}·90일 {pc(U('BE','rev90'))}",
                 "한 달 안 실적 발표 없음(다음 2/4)"],
     "risks": [f"반등 초반(10/22→10/27) {pc(U('BE','early'))}로 거의 못 따라옴", f"선행 PER {U('BE','fpe'):.0f}배",
               "KoAct는 그 전 20거래일에 수량 −55%를 팔았다가 다시 산 것"]},
    {"ticker": "VRT", "theme": "전력·냉각 장비",
     "thesis": "같은 점수로 45종목 중 43위. 반도체와 같이 움직이는 정도가 낮고 전망 상향이 약하다. 판단: SOXX보다 덜 오를 쪽.",
     "reasons": [f"SOXX 대비 베타 {U('VRT','beta'):.2f}", f"EPS 전망 30일 {pc(U('VRT','rev30'))}·90일 {pc(U('VRT','rev90'))}",
                 "두 펀드 모두 미보유"],
     "risks": ["52주 고점 대비 −44.5%로 이미 많이 빠져 있다", f"반등 초반 {pc(U('VRT','early'))}로 SOXX만큼은 탔다",
               "AI 전력 수요는 반도체와 다른 축으로 움직일 수 있다"]},
    {"ticker": "NBIS", "theme": "네오클라우드(AI 데이터센터 임대)",
     "thesis": "같은 점수로 44종목 중 30위, 공매도 20%·90일 전망 하락으로 규칙상 제외. 이익 전망이 나빠지는 쪽이라 '급락 중에도 전망이 오른 종목' 기준과 반대다. 판단: SOXX보다 덜 오를 쪽.",
     "reasons": [f"다음 회계연도 EPS 전망 30일 {pc(U('NBIS','rev30'))}·90일 {pc(U('NBIS','rev90'))} — 적자 전망 확대({pc(eps['NBIS'] / 100)[:-1]}달러)",
                 f"공매도 {pc(U('NBIS','si')).lstrip('+')}, KoAct 최근 20거래일 안 편출",
                 "10월 급락 이유 중 하나가 네오클라우드 차입 비용"],
     "risks": [f"베타 {U('NBIS','beta'):.2f} — 반등이 커지면 가장 많이 오를 수 있다", "매출 전년 대비 +454%",
               f"7월 반등 {pc(U('NBIS','july'))}"]},
]
claims = [
    {"id": "x_AXTI", "group": "종목", "kind": "ticker_excess", "ticker": "AXTI", "op": ">", "value": 0.0, "probability": 0.50,
     "text": "AXTI가 한 달 동안 SOXX보다 더 오른다"},
    {"id": "e_AXTI", "group": "종목", "kind": "event_excess", "ticker": "AXTI", "start": "2026-10-29", "end": "2026-10-30",
     "op": ">", "value": 0.0, "probability": 0.45, "text": "AXTI: 10/29 실적 다음 날 SOXX보다 더 오른다"},
    {"id": "eps_AXTI", "group": "종목", "kind": "eps_change", "ticker": "AXTI", "base": eps["AXTI"], "op": ">", "value": 0.0,
     "probability": 0.60, "text": f"AXTI: 다음 회계연도 EPS 전망이 11/27까지 오른다(기록 {eps['AXTI']:.2f})"},
    {"id": "x_BE", "group": "종목", "kind": "ticker_excess", "ticker": "BE", "op": ">", "value": 0.0, "probability": 0.55,
     "text": "BE가 한 달 동안 SOXX보다 더 오른다"},
    {"id": "eps_BE", "group": "종목", "kind": "eps_change", "ticker": "BE", "base": eps["BE"], "op": ">", "value": 0.0,
     "probability": 0.60, "text": f"BE: 다음 회계연도 EPS 전망이 11/27까지 오른다(기록 {eps['BE']:.2f})"},
    {"id": "x_VRT", "group": "종목", "kind": "ticker_excess", "ticker": "VRT", "op": ">", "value": 0.0, "probability": 0.40,
     "text": "VRT가 한 달 동안 SOXX보다 더 오른다"},
    {"id": "x_NBIS", "group": "종목", "kind": "ticker_excess", "ticker": "NBIS", "op": ">", "value": 0.0, "probability": 0.40,
     "text": "NBIS가 한 달 동안 SOXX보다 더 오른다"},
    {"id": "eps_NBIS", "group": "종목", "kind": "eps_change", "ticker": "NBIS", "base": eps["NBIS"], "op": "<", "value": 0.0,
     "probability": 0.55, "text": f"NBIS: 적자 전망이 11/27까지 더 커진다(기록 {pc(eps['NBIS'] / 100)[:-1]})"},
    {"id": "b1", "group": "비교", "kind": "basket_excess", "basket": "asked4", "op": ">", "value": 0.0, "probability": 0.50,
     "text": "물어본 4종목(같은 금액)이 SOXX보다 더 오른다"},
    {"id": "b2", "group": "비교", "kind": "basket_excess", "basket": "picks_ref", "versus": "asked4", "op": ">", "value": 0.0,
     "probability": 0.50, "text": "탑픽 10이 물어본 4종목보다 더 오른다"},
    {"id": "b3", "group": "비교", "kind": "basket_excess", "basket": "picks_ref", "versus": "asked4", "adjust": "beta", "op": ">",
     "value": 0.0, "probability": 0.60, "text": "베타를 감안하면 탑픽 10이 물어본 4종목보다 낫다"},
]
forecast = {
    "id": "2026-10-28-asked-axti-be-vrt-nbis",
    "title": "물어본 4종목 판단 (AXTI·BE·VRT·NBIS)",
    "author": "claude",
    "request": "axti는? / be는 왜없을까? vrt도 / nbis도 궁금해 / 응 지금당시의 판단을 적고 나중에비교해보자",
    "benchmark": "SOXX",
    "horizon_sessions": 21,
    "summary": ("첫 기록(탑픽 10)을 잠근 뒤 질문받은 4종목의 당시 판단이다. AXTI는 점수 2위였지만 공매도·시총 규칙으로 뺐고, SOXX를 이길 확률은 반반으로 본다. "
                "BE는 반도체 밖이라 후보에 없었지만 같은 점수로 45종목 중 7위라 넣었다면 탑픽 안이었다. SOXX보다 조금 나을 쪽(55%)으로 본다. "
                "VRT(43위)와 NBIS(30위, 적자 전망 확대)는 SOXX보다 덜 오를 쪽(각 40%)으로 본다. 탑픽 10과 기준가(10/28 종가)·마감(11/27)이 같아 나란히 비교한다."),
    "context": ["첫 기록 2026-10-28-soxx-rebound-top10(커밋 369d042)의 유니버스는 반도체 43종목이었다. BE·VRT·NBIS는 AI 전력·데이터센터 종목이라 후보가 아니었다",
                "점수 규칙은 첫 기록과 같다(전망 상향 45·베타 30·반등 초반 15·두 펀드 최근 매수 10). 이 기록의 전망 수정률은 적자 기업도 부호가 맞는 (지금 − 과거) ÷ |과거|"],
    "method_notes": ["기준가·마감·수익률·베타 조정 계산은 첫 기록과 같다(docs/desk_system.md §14)",
                     "생성: data/derived/analysis/semi_rebound/code/make_forecast_asked.py",
                     "예측 기록이다. 목표 비중·주문 수량이 없고 데스크 7종목 규칙과 무관하다"],
    "wrong_if": ["BE가 SOXX에 지면, 반도체 밖 종목을 같은 점수로 매겨 '넣었으면 탑픽'이라 본 판단이 틀린 것",
                 "VRT·NBIS가 SOXX를 크게 이기면, 낮은 베타·전망 악화로 낮춰 본 판단이 틀린 것",
                 "AXTI가 SOXX를 크게 이기면, 공매도·시총 규칙으로 뺀 판단이 비싼 대가를 치른 것"],
    "universe": universe,
    "baskets": {"asked4": {"label": "물어본 4종목", "tickers": ASKED, "beta": beta_of(ASKED)},
                "picks_ref": {"label": "탑픽 10(첫 기록)", "tickers": PICKS, "beta": first["baskets"]["picks"]["beta"],
                              "note": "첫 기록과 같은 종목·기준가"}},
    "primary_basket": "asked4",
    "picks": [{"rank": i + 1, **p} for i, p in enumerate(picks)],
    "claims": claims,
}
(RES / "forecast_asked_2026-10-28.json").write_text(json.dumps(forecast, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({t: universe[t] for t in ASKED}, ensure_ascii=False), {t: round(v, 3) for t, v in eps.items()},
      forecast["baskets"]["asked4"]["beta"])
