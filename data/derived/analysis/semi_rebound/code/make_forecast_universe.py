"""세 번째 예측: AI 유니버스 미국 상장 115종목 전체 순위와 종목별 'SOXX를 이길 확률' (2026-10-29 KST).

사용자 요청(10/29): 아예 다양한 종목을 미리 예측해 두자. 탑픽은 맨 위에 올려 두는 요소다.
점수는 첫 기록과 같은 규칙(전망 상향 45 · 베타 30 · 반등 초반 15 · 두 펀드 최근 매수 10)을 115종목에 매긴다.
확률 = 0.40 + 0.20 × 점수 백분위, 공매도 15% 초과면 −0.05(최저 0.35). 2026 네 국면에서 종목 선택 우위가
입증되지 않아 폭을 좁게 잡았다. 입력 ../results/universe_snapshot_2026-10-28.json, 출력 forecast_universe_2026-10-28.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
RES = Path(__file__).resolve().parents[1] / "results"
snap = json.loads((RES / "universe_snapshot_2026-10-28.json").read_text(encoding="utf-8"))
first = json.loads((ROOT / "data" / "forecast" / "ledger.json").read_text(encoding="utf-8"))["forecasts"][0]
d = pd.DataFrame(snap["tickers"]).T


def fund_score(f) -> int:
    if not isinstance(f, dict):
        return 0
    s = 0
    for k, v in f.items():
        if k.endswith(("_D5", "_3d")):
            s += 1 if v["kind"] in ("매수", "신규") else -1 if v["kind"] in ("매도", "편출") else 0
    return s


def fund_text(f) -> str | None:
    if not isinstance(f, dict):
        return None
    name = {"TIME_D5": "TIME 5일", "TIME_3d": "TIME 10/23 이후", "TIME_D20": "TIME 20일", "KoAct_D5": "KoAct 5일",
            "KoAct_3d": "KoAct 10/23 이후", "KoAct_D20": "KoAct 20일"}
    parts = []
    for k, v in f.items():
        if v["kind"] == "유지" or k not in name:
            continue
        size = "" if v["pct"] is None or v["kind"] in ("신규", "편출") else f" {v['pct']:+.0f}%".replace("-", "−")
        parts.append(f"{name[k]} {v['kind']}{size}")
    return " · ".join(parts) or None


for c in ("beta", "drop", "early", "july", "hi", "si", "mcap", "fpe", "rev30", "rev90", "eps"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["fund"] = [fund_score(f) for f in d.get("funds", pd.Series(index=d.index))]
pct = lambda s: s.rank(pct=True)
rev = (pct(d["rev30"].fillna(d["rev30"].median())) + pct(d["rev90"].fillna(d["rev90"].median()))) / 2
d["rev_pct"] = pct(rev)
d["score"] = 0.45 * d["rev_pct"] + 0.30 * pct(d["beta"].fillna(d["beta"].median())) \
    + 0.15 * pct(d["early"].fillna(d["early"].median())) + 0.10 * pct(d["fund"])
d["prob"] = (0.40 + 0.20 * pct(d["score"]) - 0.05 * (d["si"].fillna(0) > 0.15)).clip(lower=0.35).round(2)
d = d.sort_values(["prob", "score"], ascending=False)  # 탑픽 = 확률 순(공매도 감점 반영)
d["rank"] = range(1, len(d) + 1)
RANKED = list(d.index)
N = len(RANKED)
pc = lambda v: "—" if v is None or pd.isna(v) else f"{v * 100:+.1f}%".replace("-", "−")
r4 = lambda v: None if v is None or pd.isna(v) else round(float(v), 4)

universe = {t: {"group": d.loc[t, "group"], "sub": d.loc[t, "sub"], "beta": r4(d.loc[t, "beta"]), "rev": r4(d.loc[t, "rev_pct"]),
                "rev30": r4(d.loc[t, "rev30"]), "rev90": r4(d.loc[t, "rev90"]), "early": r4(d.loc[t, "early"]),
                "october_drop": r4(d.loc[t, "drop"]), "july": r4(d.loc[t, "july"]), "si": r4(d.loc[t, "si"]),
                "mcap_busd": r4(d.loc[t, "mcap"] / 1e9 if pd.notna(d.loc[t, "mcap"]) else None), "fpe": r4(d.loc[t, "fpe"]),
                "fund": int(d.loc[t, "fund"]), "score": r4(d.loc[t, "score"]), "prob": float(d.loc[t, "prob"]),
                "rank": int(d.loc[t, "rank"])} for t in RANKED}

picks = []
for t in RANKED:
    u, row = universe[t], d.loc[t]
    risks = []
    if u["si"] and u["si"] > 0.15:
        risks.append(f"공매도 {pc(u['si']).lstrip('+')} — 7월 연구에서 공매도가 높을수록 회복이 더뎠다")
    if u["mcap_busd"] is not None and u["mcap_busd"] < 5:
        risks.append(f"시총 {u['mcap_busd'] * 10:.0f}억 달러 소형주 — 위아래로 크게 움직인다")
    if u["rev90"] is not None and u["rev90"] < 0:
        risks.append(f"90일 전망 {pc(u['rev90'])} — 이익 전망이 내려가는 중")
    if u["beta"] is not None and u["beta"] < 0.8:
        risks.append(f"SOXX 대비 베타 {u['beta']:.2f} — 반도체가 크게 오르면 뒤처지기 쉽다")
    if not risks:
        risks.append("가격·전망 요인으로 고른 순위의 우위는 2026년 네 국면에서 입증되지 않았다")
    beta = "—" if u["beta"] is None else f"{u['beta']:.2f}"
    reasons = [f"전망 수정 30일 {pc(u['rev30'])} · 90일 {pc(u['rev90'])}",
               f"베타 {beta} · 반등 초반(10/22→10/27) {pc(u['early'])} · 10월 하락 {pc(u['october_drop'])}"]
    ft = fund_text(row.get("funds"))
    if ft:
        reasons.append(f"두 펀드: {ft}")
    picks.append({"ticker": t, "theme": f"{u['group']} · {u['sub']}",
                  "thesis": f"{u['rank']}위/{N} · SOXX를 이길 확률 {u['prob']:.0%}", "reasons": reasons, "risks": risks})

TOP, BOTTOM = RANKED[:10], RANKED[-10:]
SEMIS = [t for t in RANKED if universe[t]["group"] in ("칩 설계", "파운드리·후공정", "메모리·스토리지", "장비·소재·테스트")]
beta_of = lambda ts: round(sum(universe[t]["beta"] for t in ts if universe[t]["beta"] is not None)
                           / max(1, sum(universe[t]["beta"] is not None for t in ts)), 4)
baskets = {"ranked": {"label": f"AI 유니버스 {N}종목", "tickers": RANKED, "beta": beta_of(RANKED), "chart": True},
           "top10": {"label": "새 순위 상위 10(탑픽)", "tickers": TOP, "beta": beta_of(TOP), "chart": True},
           "bottom10": {"label": "새 순위 하위 10", "tickers": BOTTOM, "beta": beta_of(BOTTOM), "chart": True},
           "prior_top10": {"label": "첫 기록 탑픽 10", "tickers": first["baskets"]["picks"]["tickers"],
                           "beta": first["baskets"]["picks"]["beta"], "chart": True, "note": "반도체 43종목에서 고른 첫 기록"},
           "semis": {"label": "반도체 4묶음", "tickers": SEMIS, "beta": beta_of(SEMIS)}}
for g in dict.fromkeys(universe[t]["group"] for t in RANKED):
    ts = [t for t in RANKED if universe[t]["group"] == g]
    baskets["g_" + str(len(baskets))] = {"label": g, "tickers": ts, "beta": beta_of(ts), "note": "그룹"}
sw = next(k for k, b in baskets.items() if b["label"] == "소프트웨어·보안")

claims = [
    {"id": "f_score", "group": "순위 전체", "kind": "factor_rho", "factor": "score", "op": ">", "value": 0.0, "probability": 0.60,
     "text": f"내 점수(예측 순위)가 높을수록 한 달 동안 더 오른다 — {N}종목 순위 상관"},
    {"id": "f_rev", "group": "순위 전체", "kind": "factor_rho", "factor": "rev", "op": ">", "value": 0.0, "probability": 0.58,
     "text": "실적 전망 상향 폭이 클수록 더 오른다"},
    {"id": "f_beta", "group": "순위 전체", "kind": "factor_rho", "factor": "beta", "op": ">", "value": 0.0, "probability": 0.60,
     "text": "SOXX 대비 베타가 클수록 더 오른다(반도체 반등이 이어지면)"},
    {"id": "f_si", "group": "순위 전체", "kind": "factor_rho", "factor": "si", "op": "<", "value": 0.0, "probability": 0.55,
     "text": "공매도 비율이 높을수록 덜 오른다"},
    {"id": "b_top_soxx", "group": "바스켓", "kind": "basket_excess", "basket": "top10", "op": ">", "value": 0.0, "probability": 0.58,
     "text": "새 순위 상위 10이 SOXX보다 더 오른다"},
    {"id": "b_top_alpha", "group": "바스켓", "kind": "basket_excess", "basket": "top10", "adjust": "beta", "op": ">", "value": 0.0,
     "probability": 0.53, "text": "베타를 감안해도 상위 10이 SOXX를 이긴다"},
    {"id": "b_top_bottom", "group": "바스켓", "kind": "basket_excess", "basket": "top10", "versus": "bottom10", "op": ">", "value": 0.0,
     "probability": 0.62, "text": "상위 10이 하위 10보다 더 오른다"},
    {"id": "b_top_all", "group": "바스켓", "kind": "basket_excess", "basket": "top10", "versus": "ranked", "op": ">", "value": 0.0,
     "probability": 0.58, "text": f"상위 10이 {N}종목 같은 금액 바스켓보다 더 오른다"},
    {"id": "b_top_prior", "group": "바스켓", "kind": "basket_excess", "basket": "top10", "versus": "prior_top10", "op": ">", "value": 0.0,
     "probability": 0.50, "text": "후보를 넓힌 상위 10이 첫 기록 탑픽 10보다 낫다"},
    {"id": "b_top_count", "group": "바스켓", "kind": "basket_count_beat", "basket": "top10", "op": ">=", "value": 6, "probability": 0.50,
     "text": "상위 10 중 6개 이상이 SOXX를 이긴다"},
    {"id": "g_semis_sw", "group": "바스켓", "kind": "basket_excess", "basket": "semis", "versus": sw, "op": ">", "value": 0.0,
     "probability": 0.55, "text": "반도체 4묶음이 소프트웨어·보안보다 더 오른다"},
]
for t in RANKED:
    claims.append({"id": f"x_{t}", "group": "종목", "kind": "ticker_excess", "ticker": t, "op": ">", "value": 0.0,
                   "probability": universe[t]["prob"], "text": f"{t}가 한 달 동안 SOXX보다 더 오른다"})

top_line = " · ".join(TOP)
forecast = {
    "id": "2026-10-29-ai-universe-ranked",
    "title": f"AI 유니버스 {N}종목 순위 예측",
    "author": "claude",
    "request": "아예 다양한 종목을 미리 예측을 해둘래요? 탑픽은 맨위에올려두는 요소인거고",
    "benchmark": "SOXX",
    "horizon_sessions": 20,
    "headline_basket": "top10",
    "summary": (f"AI 유니버스 미국 상장 {N}종목 전부에 순위와 'SOXX를 이길 확률'을 붙였다. 맨 위 10개({top_line})가 이 기록의 탑픽이다. "
                "점수는 첫 기록과 같은 규칙(전망 상향 45·베타 30·반등 초반 15·두 펀드 매수 10)이고, 확률은 점수에 따라 0.35~0.60으로 좁게 잡았다. "
                "반도체 반등이 이어질 확률을 60%로 보니 베타가 낮은 소프트웨어·빅테크는 SOXX를 이기기 어렵다고 본다. "
                "기준가는 10/29 종가, 마감은 11/27 종가다. 탑픽 몇 개가 아니라 순위 전체가 실제 수익과 맞았는지(순위 상관, 상위·하위 10 차이, 확률 점수)로 채점한다."),
    "context": ["10/28 미국 장 마감 뒤에 기록해 기준가는 기록 뒤 첫 종가인 10/29 종가다. 앞의 두 기록보다 하루 늦게 시작하고, 마감을 11/27로 맞추려고 20거래일로 잡았다",
                "첫 기록 탑픽 10을 대조군으로 넣었다(이 기록 안에서는 10/29 종가부터 잰다)",
                "가격 요인은 첫 기록과 같은 정의로 10/27 종가까지, 전망치·공매도·펀드 매매는 10/29 07:59 KST에 받은 값이다(results/universe_snapshot_2026-10-28.json)"],
    "method_notes": ["확률 = 0.40 + 0.20 × 점수 백분위, 공매도 15% 초과면 −0.05(최저 0.35). 종목끼리 같은 시장을 타므로 확률 판정은 서로 독립이 아니다",
                     "전망 수정률은 (지금 − 과거) ÷ |과거|(적자 기업도 개선이면 +)",
                     "생성: data/derived/analysis/semi_rebound/code/universe_snapshot.py, make_forecast_universe.py",
                     "예측 기록이다. 목표 비중·주문 수량이 없고 데스크 7종목 규칙과 무관하다"],
    "wrong_if": ["점수 순위 상관(f_score)이 0 이하면 순위 매기는 방식 자체가 틀린 것",
                 "상위 10이 하위 10에 지면(b_top_bottom) 맨 위에 올린 이유가 없던 것",
                 "베타 상관(f_beta)이 마이너스면 반도체 반등이 이어진다는 시장 판단이 틀린 것"],
    "universe": universe,
    "baskets": baskets,
    "primary_basket": "ranked",
    "picks": [{"rank": i + 1, **p} for i, p in enumerate(picks)],
    "claims": claims,
}
(RES / "forecast_universe_2026-10-28.json").write_text(json.dumps(forecast, ensure_ascii=False, indent=1), encoding="utf-8")
show = d[["group", "score", "prob", "beta", "rev30", "rev90", "early", "si", "fund"]].copy()
print(show.head(15).round(3).to_string())
print("...")
print(show.tail(5).round(3).to_string())
print({k: (b["label"], len(b["tickers"]), b["beta"]) for k, b in baskets.items()})
print("claims", len(claims), "prob range", d["prob"].min(), d["prob"].max())
