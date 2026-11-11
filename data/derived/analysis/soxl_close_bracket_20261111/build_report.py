"""Render saved SOXL bracket study results; does not run the trading engine."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SCRATCH_NAME = "soxl-close-bracket-20261111-5c9e72.html"


def pct(value, digits=2):
    return "—" if value is None else f"{value:+.{digits}f}%"


def num(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def model(summary, window, start="2026-01-02", gap="open", policy="stop_first"):
    return next(m for m in summary["models"] if m["window"] == window and m["run_start"] == start
                and m["gap_tp"] == gap and m["policy"] == policy)


def main_trades(trades, window):
    return trades[(trades.window == window) & (trades.policy == "stop_first")
                  & (trades.gap_tp == "open") & (trades.run_start == "2026-01-02")
                  & trades.exit_day.notna()].copy()


def described_trades(trades, window):
    t = main_trades(trades, window)
    wins, losses = t.loc[t.net_10bp_pct > 0], t.loc[t.net_10bp_pct < 0]
    gaps = t.loc[t.gap_stop]
    worst = t.sort_values("net_10bp_pct").head(5)
    return {
        "closed_trades": len(t), "average_win_net10bp_pct": float(wins.net_10bp_pct.mean()) if len(wins) else None,
        "average_loss_net10bp_pct": float(losses.net_10bp_pct.mean()) if len(losses) else None,
        "average_holding_calendar_hours": float(t.holding_calendar_hours.mean()) if len(t) else None,
        "maximum_holding_calendar_hours": float(t.holding_calendar_hours.max()) if len(t) else None,
        "gap_stop_count": len(gaps), "gap_stop_worst_gross_pct": float(gaps.gross_pct.min()) if len(gaps) else None,
        "worst_five": [{"entry_day": r.entry_day, "exit_day": r.exit_day, "entry_price": float(r.entry_price),
                        "exit_price": float(r.exit_price), "reason": r.reason, "gross_pct": float(r.gross_pct),
                        "net10bp_pct": float(r.net_10bp_pct)} for r in worst.itertuples()],
    }


def table_row(label, cell):
    st, nav = cell["costs"]["10"], cell["portfolio"]["10"]
    lo, hi = st["mean_ci95_pct"]
    return (f"| {label} | {cell['closed']} | {cell['target']}/{cell['stop']} | {num(st['win_rate_pct'],1)}% "
            f"| {pct(st['mean_pct'])} | {pct(lo)}~{pct(hi)} | {num(st['profit_factor'])} "
            f"| {pct(nav['return_pct'])} | {pct(nav['daily_close_mdd_pct'])} |")


def report_text(summary, desc):
    rth, ext = model(summary, "RTH"), model(summary, "EXT")
    bh = summary["buy_hold"]
    ra, ea = rth["all"], ext["all"]
    lines = [
        "# SOXL 종가 매수 → −3% 손절·+2% 익절",
        "",
        "2026-11-11 · Path-Q 연구 · 고정 평가: 2026-01-02~11-10, 216거래일",
        "",
        "**계산 결과: 정규장 청산 모형은 연초 이후 소폭 수익이지만, 하반기 손실과 큰 낙폭 때문에 수익 우위가 확인됐다고 보기 어렵다.**",
        f"편도10bp 가정에서 정규장 모형은 {pct(ra['portfolio']['10']['return_pct'])}, "
        f"연속 원장의 하반기 수익은 {pct(rth['H2']['portfolio']['10']['return_pct'])}다. "
        f"04~20시 조건청산을 가정한 별도 감도는 전체 {pct(ea['portfolio']['10']['return_pct'])}다. "
        "새 조건의 좋은 숫자를 고르지 않고 두 모형을 함께 공개한다.",
        "",
        f"**수수료0·추가 거래비용0이면 정규장 모형 전체 {pct(ra['portfolio']['0']['return_pct'])}, "
        f"연속 하반기 {pct(rth['H2']['portfolio']['0']['return_pct'])}다.** "
        "편도10bp는 실제 호가에서 측정한 비용이 아니라 민감도 가정이다. 214회 반복 거래에서는 비용 차이가 누적되어 "
        "연간 결과가 크게 달라지지만, 하반기 손실은 비용0에서도 남는다. 거래당 평균의95% 구간도0을 포함한다.",
        "",
        "핵심 위험은 **손절선을 −3%로 정해도 −3%에 팔린다는 보장이 없다는 것**이다. "
        f"정규장 모형에는 하락 갭 손절 {ra['gap_stops']}건이 있고, "
        f"최악 거래는 비용 전 {pct(ra['gross']['worst_pct'])}, 비용 후 {pct(ra['costs']['10']['worst_pct'])}다. "
        "상승 갭도 첫 분봉 시가의 가격 개선을 같은 원칙으로 반영했다.",
        "",
        "## 계산 조건",
        "",
        "| 항목 | 고정 기준 |", "|---|---|",
        "| 사용자 요청 | SOXL 정규장 종가 매수, 거래가격 대비 −3% 손절 또는 +2% 익절 중 먼저 도달 |",
        "| 매수 신호 | RSI·추세 필터 없이 현금인 날 종가 매수 |",
        "| 중복·보유 | 한 포지션만 보유, 물타기·추가 입금 없음. 어느 선에도 닿지 않으면 계속 보유 |",
        "| 종가 주문 시점 | 마감10분 전까지 완성된 분봉으로 현금 여부를 확정했다고 가정. 그 뒤 청산되면 다음 날 종가부터 재진입 |",
        "| 주 청산 모형 | 정규장(RTH) 관측 가격으로 익절·손절. 시가가 경계 밖이면 그 시가, 경계 안이면 닿은 경계가격 |",
        "| 연장시간 감도 | 04:00~20:00 ET의 분봉으로 조건청산. 신규 매수일에는 마감 시각 분봉을 제외하고 1분 뒤부터. 실제 애프터 스탑 가능 여부 미확인 |",
        "| 동일 분봉 양쪽 도달 | 시가가 두 선 안에 있고 고저가 양쪽에 닿으면 손절 우선. 익절 우선도 별도 확인 |",
        "| 비용 | 수수료0. 한쪽0·5·10·20bp 가상 거래비용, 주 비교10bp. 경계가격은 비용별로 옮기지 않음 |",
        "| 평가 | 초기 가상 자산100, 청산 대금 재투자. 매일 정규장 종가 직후 평가, 그날 애프터 이전. 최종 미청산은 승패에서 제외하고 시가평가 |",
        "",
        "주문시점·보유·재투자·비용은 사용자 개인 사실이 아니라 연구 구현 가정이다. "
        "Alpaca SIP 분할수정1Day Close를 정규장 종가 기준값으로 썼으며 주거래소 단일 경매에서 그 가격으로 체결된다는 보장은 없다. "
        "정규장 종가는 사용자 직접 주문 시간(KST08~24) 밖이므로 예약 종가 주문과 매수 직후 익절·손절 연동 가능 여부도 별도 확인되지 않았다.",
        "",
        "## 주 결과 — 편도10bp 모형",
        "",
        "승률·평균·PF는 종료된 거래, 누적 수익과 낙폭은 미청산 평가를 포함한 일별 자산 원장이다. "
        "낙폭은 **일별 종가 기준**이며 장중 최대 낙폭이 아니다.",
        "",
        "| 청산 관측 | 종료 거래 | 익절/손절 | 승률 | 거래당 평균 | 평균95% 구간 | PF | 자산 수익 | 종가 MDD |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
        table_row("정규장", ra), table_row("04~20시 가정", ea),
        "",
        f"같은 첫날 종가에 매수해 마지막 종가까지 보유한 비교 기준은 {pct(bh['all']['10']['return_pct'])}, "
        f"종가 MDD {pct(bh['all']['10']['daily_close_mdd_pct'])}다. "
        "두 기준 모두 마지막 가격으로 평가하므로 미청산 매도비용은 차감하지 않았다. 단순 보유는 매수비용 한 번만 반영한다.",
        "",
        "## 상반기·하반기와 7월 현금 출발",
        "",
        "익절·손절 한 쌍은 사용자가 고정했으므로 상반기로 수치를 고르는 단계는 없다. "
        "기존2026자료를 재사용한 후속 연구이며 새 독립 표본이 생긴 것은 아니다. "
        "아래 연속 원장은6/30 자산을 하반기로 넘기며, 7/1 현금 출발은 그날 종가에 처음 진입하는 별도 경로다.",
        "",
        "| 구간·시작 | 청산 관측 | 종료 거래 | 거래당 평균 | 자산 수익 | 종가 MDD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for period, label in (("H1", "1/2~6/30 연속"), ("H2", "7/1~11/10 연속")):
        for window, wlabel in (("RTH", "정규장"), ("EXT", "04~20시 가정")):
            c = model(summary, window)[period]
            lines.append(f"| {label} | {wlabel} | {c['closed']} | {pct(c['costs']['10']['mean_pct'])} "
                         f"| {pct(c['portfolio']['10']['return_pct'])} | {pct(c['portfolio']['10']['daily_close_mdd_pct'])} |")
    for window, label in (("RTH", "정규장"), ("EXT", "04~20시 가정")):
        c = model(summary, window, start="2026-07-01")["all"]
        lines.append(f"| 7/1 현금 출발 | {label} | {c['closed']} | {pct(c['costs']['10']['mean_pct'])} "
                     f"| {pct(c['portfolio']['10']['return_pct'])} | {pct(c['portfolio']['10']['daily_close_mdd_pct'])} |")
    lines.extend(["", f"단순 보유의 상반기 수익 {pct(bh['H1']['10']['return_pct'])}, "
                  f"연속 하반기 수익 {pct(bh['H2_continuous']['return_pct'])}, "
                  f"7/1 현금 출발 수익 {pct(bh['H2']['10']['return_pct'])}다.", "",
                  "## 손익 크기와 보유 시간", "",
                  "| 청산 관측 | 평균 이익 | 평균 손실 | 최악/최고 | 평균 보유 세션 | 평균/최대 보유 시간 | 최종 미청산 |",
                  "|---|---:|---:|---|---:|---|---:|"])
    for window, label in (("RTH", "정규장"), ("EXT", "04~20시 가정")):
        c, d = model(summary, window)["all"], desc[window]
        lines.append(f"| {label} | {pct(d['average_win_net10bp_pct'])} | {pct(d['average_loss_net10bp_pct'])} "
                     f"| {pct(c['costs']['10']['worst_pct'])}/{pct(c['costs']['10']['best_pct'])} "
                     f"| {num(c['average_hold_sessions'])} | {num(d['average_holding_calendar_hours'],1)}/{num(d['maximum_holding_calendar_hours'],1)}시간 "
                     f"| {c['open']} |")
    for window, label in (("RTH", "정규장"), ("EXT", "04~20시 가정")):
        lines.extend(["", f"### {label} 최악5건", "",
                      "| 매수일 | 청산일 | 사유 | 매수가→청산가 | 비용 전 | 비용 후 |",
                      "|---|---|---|---|---:|---:|"])
        for row in desc[window]["worst_five"]:
            lines.append(f"| {row['entry_day']} | {row['exit_day']} | {row['reason']} "
                         f"| {num(row['entry_price'],4)}→{num(row['exit_price'],4)} "
                         f"| {pct(row['gross_pct'])} | {pct(row['net10bp_pct'])} |")
    lines.extend(["", "## 실행·비용 감도", "",
                  "첫 계산은 상승 갭도 목표가격까지만 받는 보수 모형이었다. 이는 하락 갭 손실은 전부 반영하면서 "
                  "상승 갭 가격 개선은 버리는 비대칭이 있어 **최초 결과를 본 뒤 주 모형을 경계 밖 첫 시가로 대칭 수정**했다. "
                  "기존 목표가격 제한 모형도 아래에 남겼다. 익절·손절 수치나 신호를 수익에 맞춰 바꾸지는 않았다. "
                  "시가 가격 개선도 실제 지정가 체결 보장은 아니므로 실행 가정의 민감도로 읽어야 한다.", "",
                  "| 청산 관측 | 상승 갭 청산 | 비용0 | 편도5bp | 편도10bp | 편도20bp |", "|---|---|---:|---:|---:|---:|"])
    for window, label in (("RTH", "정규장"), ("EXT", "04~20시 가정")):
        for gap, glabel in (("open", "시가 개선 반영·주 모형"), ("target", "+2% 목표가격 제한")):
            c = model(summary, window, gap=gap)["all"]
            lines.append(f"| {label} | {glabel} | " + " | ".join(pct(c["portfolio"][str(cost)]["return_pct"]) for cost in (0,5,10,20)) + " |")
    ambiguities = {w: model(summary, w)["all"]["ambiguous"] for w in ("RTH", "EXT")}
    lines.extend(["", f"동일 분봉 양쪽 도달은 정규장 {ambiguities['RTH']}건·04~20시 {ambiguities['EXT']}건이다. "
                  "우선순위 감도를 별도 저장했고 현재 자료에서는 두 우선순위 결과가 같다.", "",
                  "## 자료·범위·제한", "",
                  "- 사용자 손절/익절 한 쌍만 시험했다. 2관측시간×2동일봉 우선순위×2상승갭 체결=8실행 경로, "
                  "비용4개·시작일2개를 비교했다. 경계 수치 최적화는0회다. 이전 RSI 연구732조합을 이번 최적 조건 선택의 근거로 합치지 않는다.",
                  "- Alpaca SIP split-adjusted1분봉과1Day Close,216거래일을 썼다. 매수 당일의 장중 고저를 청산 근거로 쓰지 않는다. "
                  "경계 확인 시점은 관측 분봉이고 분 안의 정확한 체결 초는 알 수 없다.",
                  "- 정규장 모든 예상 분봉과 양수 거래량을 확인했다. 연장시간에 빠진 분은 보간하지 않았고20~04시·한국 낮 ATS 거래는 포함하지 않았다.",
                  "- 분봉 고저 터치는 실제 bid/ask·물량·주문 대기열을 보장하지 않는다. 비용은 실측 호가가 아닌 별도 모형이다. "
                  "익절 지정가·손절 주문의 동시 연동과 연장시간 스탑 기능은 미확인이다.",
                  "- 현금배당·환율·세금·이자·추가 시장충격은 제외했다. 승패·95% 구간은 종료 거래 기준이고, "
                  "평균95% 구간은 기존 통계 함수의 거래단위10,000회 부트스트랩이다. 반복 거래와 장세 의존성이 있으므로 미래 수익 확률로 바꾸지 않는다.",
                  "- 연구 결과만으로 기존 운용 규칙·주문·비중·수량을 바꾸지 않는다.", "",
                  "자료와 가격 구성: [Alpaca market data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq). "
                  "주문 시간과 연장시간 제약: [Alpaca orders](https://docs.alpaca.markets/us/docs/orders-at-alpaca). "
                  "스탑가격과 실제 체결가격의 차이: [SEC Investor.gov bulletin](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-15).", "",
                  "독립 검산은16실행 원장·종료 거래2,440행·일별 자산2,472행과14개 합성 사례를 확인했다. "
                  "미래 가격을 바꿔도 상반기 결과는 바뀌지 않았다. 검산 통과 원장은 [audit.json](results/audit.json)이다.", "",
                  "[가격·달력 검증](results/data_audit.json) · [전체 모형](results/summary.json) · "
                  "[거래 원장](results/trades.csv) · [자산 원장](results/equity.csv) · [대화형 차트](performance.html)", ""])
    return "\n".join(lines)


def chart_payload(summary, equity, daily):
    result = {"periods": {}, "source": "Alpaca SIP split-adjusted daily close + minute OHLC", "start_nav": 100}
    for start, key in (("2026-01-02", "all"), ("2026-07-01", "freshH2")):
        dates = daily.loc[daily.day >= start, "day"].tolist()
        prices = daily.loc[daily.day >= start, "Close"].tolist()
        item = {"days": dates, "series": {}, "metrics": {}, "period_metrics": {}}
        for cost in (0, 5, 10, 20):
            ck = str(cost)
            item["series"][ck] = {"BH": [100 * price / prices[0] / (1 + cost / 10000) for price in prices]}
            item["metrics"][ck] = {"BH": summary["buy_hold"]["all" if key == "all" else "H2"][ck]}
            item["period_metrics"][ck] = {}
            for window in ("RTH", "EXT"):
                q = equity[(equity.window == window) & (equity.policy == "stop_first")
                           & (equity.gap_tp == "open") & (equity.run_start == start)].sort_values("day")
                assert q.day.tolist() == dates
                item["series"][ck][window] = [float(v) for v in q[f"nav_{cost}bp"]]
                c = model(summary, window, start=start)
                item["metrics"][ck][window] = {**c["all"]["portfolio"][ck],
                                               "closed": c["all"]["closed"],
                                               "mean_pct": c["all"]["costs"][ck]["mean_pct"]}
                item["period_metrics"][ck][window] = {
                    p: {**c[p]["portfolio"][ck], "closed": c[p]["closed"],
                        "mean_pct": c[p]["costs"][ck]["mean_pct"]} for p in ("H1", "H2") if p in c}
        result["periods"][key] = item
    return result


HTML = r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SOXL 종가 매수 −3% / +2%</title><style>
*{box-sizing:border-box}body{margin:0;padding:24px;background:var(--background,#fff);color:var(--foreground,#1b1b20);font:14px/1.55 var(--font-sans,system-ui,sans-serif)}h1{font-size:22px;letter-spacing:-.8px;margin:0 0 6px}p{margin:5px 0 14px}.muted{color:var(--muted-foreground,#61616d)}.toolbar{display:flex;flex-wrap:wrap;gap:14px;margin:18px 0}.toolbar label{display:flex;gap:7px;align-items:center}select{border:1px solid var(--input,#c9c9d0);border-radius:7px;background:var(--card,#fff);color:var(--card-foreground,#1b1b20);padding:7px 10px;font:inherit}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.card{border:1px solid var(--border,#dedee5);background:var(--card,#fff);color:var(--card-foreground,#1b1b20);border-radius:10px;padding:13px}.value{font-size:23px;letter-spacing:-.8px;font-weight:650}.detail{font-size:12px;color:var(--muted-foreground,#61616d)}.plot{margin-top:15px;border:1px solid var(--border,#dedee5);border-radius:10px;padding:10px}svg{display:block;width:100%;height:auto;touch-action:pan-y}.legend{display:flex;flex-wrap:wrap;gap:18px;font-size:12px}.dot{display:inline-block;width:8px;height:8px;border-radius:100%;margin-right:5px}.hover{min-height:24px;font-size:12px;color:var(--muted-foreground,#61616d);font-family:var(--font-mono,monospace)}table{border-collapse:collapse;width:100%;margin-top:13px;font-size:13px}th,td{padding:9px 6px;border-bottom:1px solid var(--border,#dedee5);text-align:right}th:first-child,td:first-child{text-align:left}th{font-weight:550;color:var(--muted-foreground,#61616d)}.notes{font-size:12px;border-top:1px solid var(--border,#dedee5);padding-top:13px;margin-top:17px}button{font:inherit}a{color:var(--primary,#2554b0)}@media(max-width:550px){body{padding:14px}.cards{gap:6px}.card{padding:9px}.value{font-size:19px}.detail{font-size:11px}h1{font-size:20px}.tablewrap{overflow:auto}}
</style><h1>SOXL · 종가에 사고 −3% / +2%에 청산</h1>
<p class="muted">RSI 조건 없음 · 한 포지션 · 두 가격 중 먼저 닿을 때까지 보유 · 2026.01.02~11.10</p>
<div class="toolbar"><label>시작 <select id="period"><option value="all">1월 2일 · 자산100</option><option value="freshH2">7월 1일 · 현금100 재출발</option></select></label><label>청산 관측 <select id="window"><option value="both">두 모형 함께</option><option value="RTH">정규장</option><option value="EXT">04~20시 가정</option></select></label><label>편도 비용 <select id="cost"><option value="0">0bp</option><option value="5">5bp</option><option value="10" selected>10bp · 주 비교</option><option value="20">20bp</option></select></label></div>
<div class="cards" id="cards"></div><div class="plot"><div class="legend" id="legend"></div><svg id="chart" viewBox="0 0 900 340" role="img" aria-label="일별 종가 자산 지수"></svg><div class="hover" id="hover">차트 위에서 날짜별 자산을 확인하세요. 초기 자본100, 첫 매수비용 차감.</div></div>
<p class="muted" style="font-size:12px;margin-top:8px">사용자 수수료는0원이며, 10bp는 별도 가상 비용입니다. 위에서 비용0 결과도 확인할 수 있습니다. 상승·하락 갭은 모두 첫 관측 분봉 시가 반영. 단순 보유도 첫날 종가 매수비용을 반영하고 마지막 종가로 평가합니다.</p>
<div class="tablewrap"><table><thead><tr><th>원장 구간</th><th>청산 모형</th><th>종료 거래</th><th>거래당 평균</th><th>자산 수익</th><th>종가 MDD</th></tr></thead><tbody id="period-table"></tbody></table></div>
<div class="notes"><p><strong>−3%는 손실 한도가 아닙니다.</strong> 밤사이 가격이 경계를 넘으면 다음 관측가격에 청산합니다. 주 모형은 정규장 관측이며, 04~20시 모형은 연장시간 스탑이 가능하다는 연구 가정입니다.</p><p class="muted">마감10분 전 현금 여부로 종가 주문을 정했다고 가정합니다. 분봉 터치·종가 근사·비용모형이며 실제 체결 검증은 아닙니다. 낙폭은 일별 종가 기준입니다. 마지막 미청산은 평가손익에 포함하고 종료 거래 승패에서는 제외합니다. 사용자 고정 −3/+2 한 쌍만 시험했고 실행8경로·비용4개·시작일2개를 확인했습니다.</p></div>
<script>const DATA=__DATA__;
const $=id=>document.getElementById(id),names={RTH:'정규장',EXT:'04~20시 가정',BH:'단순 보유'};
const fmt=x=>(x>=0?'+':'')+x.toFixed(2)+'%',val=x=>x.toFixed(2);
function colors(){let s=getComputedStyle(document.documentElement);return {RTH:s.getPropertyValue('--chart-1').trim()||'#2563eb',EXT:s.getPropertyValue('--chart-2').trim()||'#e07b27',BH:s.getPropertyValue('--chart-3').trim()||'#7a62b5',grid:s.getPropertyValue('--border').trim()||'#dedee5',text:s.getPropertyValue('--muted-foreground').trim()||'#61616d'};}
let current=null;
function render(){let pk=$('period').value,ck=$('cost').value,w=$('window').value,d=DATA.periods[pk],c=colors(),keys=w==='both'?['RTH','EXT','BH']:[w,'BH'],metrics=d.metrics[ck],ser=d.series[ck];
$('cards').innerHTML=['RTH','EXT','BH'].map(k=>{let m=metrics[k];return `<div class="card"><span class="dot" style="background:${c[k]}"></span>${names[k]}<div class="value">${fmt(m.return_pct)}</div><div class="detail">종가 MDD ${fmt(m.daily_close_mdd_pct)}${k==='BH'?'':`<br>${m.closed}건 · 평균 ${fmt(m.mean_pct)}`}</div></div>`;}).join('');
$('legend').innerHTML=keys.map(k=>`<span><i class="dot" style="background:${c[k]}"></i>${names[k]}</span>`).join('');
let all=keys.flatMap(k=>ser[k]),hi=Math.ceil(Math.max(100,...all)/50)*50,lo=Math.max(0,Math.floor(Math.min(100,...all)/50)*50),left=55,right=875,top=18,bottom=300,x=i=>left+i/(d.days.length-1)*(right-left),y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top),out='';
for(let j=0;j<=5;j++){let z=lo+(hi-lo)*j/5;out+=`<line x1="${left}" x2="${right}" y1="${y(z)}" y2="${y(z)}" stroke="${c.grid}"/><text x="${left-9}" y="${y(z)+4}" text-anchor="end" fill="${c.text}" font-size="11">${z.toFixed(0)}</text>`;}
if(lo<100&&hi>100)out+=`<line x1="${left}" x2="${right}" y1="${y(100)}" y2="${y(100)}" stroke="${c.text}" opacity=".45" stroke-dasharray="3 4"/>`;
for(let j=0;j<=4;j++){let i=Math.round((d.days.length-1)*j/4);out+=`<text x="${x(i)}" y="326" text-anchor="middle" fill="${c.text}" font-size="11">${d.days[i].slice(5)}</text>`;}
for(let k of keys){let path=ser[k].map((v,i)=>(i?'L':'M')+x(i).toFixed(2)+','+y(v).toFixed(2)).join(' ');out+=`<path d="${path}" fill="none" stroke="${c[k]}" stroke-width="${k==='BH'?1.7:2.2}" ${k==='BH'?'stroke-dasharray="5 3" opacity=".85"':''}/>`;}
$('chart').innerHTML=out;current={d,ser,keys,left,right,x,y};
let periods=pk==='all'?[['H1','1/2~6/30'],['H2','7/1~11/10']]:[['H2','7/1 현금 출발']];
$('period-table').innerHTML=periods.flatMap(([p,label])=>['RTH','EXT'].map(k=>{let m=d.period_metrics[ck][k][p];return `<tr><td>${label}</td><td>${names[k]}</td><td>${m.closed}</td><td>${fmt(m.mean_pct)}</td><td>${fmt(m.return_pct)}</td><td>${fmt(m.daily_close_mdd_pct)}</td></tr>`;})).join('');
$('hover').textContent=`${d.days[d.days.length-1]} 평가 · `+keys.map(k=>names[k]+' '+val(ser[k].at(-1))).join(' / ');}
for(let id of ['period','window','cost'])$(id).addEventListener('change',render);
$('chart').addEventListener('pointermove',e=>{if(!current)return;let {d,ser,keys,left,right}=current,b=$('chart').getBoundingClientRect(),sx=(e.clientX-b.left)/b.width*900,i=Math.max(0,Math.min(d.days.length-1,Math.round((sx-left)/(right-left)*(d.days.length-1))));$('hover').textContent=d.days[i]+' · '+keys.map(k=>names[k]+' '+val(ser[k][i])).join(' / ');});
render();new MutationObserver(()=>render()).observe(document.documentElement,{attributes:true,attributeFilter:['style','class','data-theme']});
</script></html>'''


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["models"]) == 16
    trades = pd.read_csv(OUT / "trades.csv")
    equity = pd.read_csv(OUT / "equity.csv")
    daily = pd.read_csv(OUT / "daily_prices.csv")
    daily["day"] = daily.timestamp.str[:10]
    descriptions = {w: described_trades(trades, w) for w in ("RTH", "EXT")}
    metadata = {"kind": "saved-result presentation; no trading engine recalculation", "protocol": summary["protocol"],
                "source_inputs": ["results/summary.json", "results/trades.csv", "results/equity.csv", "results/daily_prices.csv"],
                "descriptive_trade_aggregates": descriptions, "html_scratch_filename": SCRATCH_NAME,
                "official_close_note": "SIP1Day Close is a daily regular-close benchmark; single-primary-auction fill is not guaranteed.",
                "execution_revision_disclosed": True, "price_threshold_pairs_tested": 1,
                "execution_paths": 8, "cost_sensitivities": 4, "start_dates": 2}
    audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    assert audit["pass"] is True
    metadata["independent_audit"] = {"pass": True, "path": "results/audit.json",
                                     "models": len(audit["models"]), "synthetic_checks": len(audit["synthetic_checks"]),
                                     "trade_rows_checked": audit["trade_rows_checked"],
                                     "daily_equity_rows_checked": audit["daily_equity_rows_checked"]}
    (ROOT / "report.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (ROOT / "report.md").write_text(report_text(summary, descriptions), encoding="utf-8")
    payload = chart_payload(summary, equity, daily)
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    assert len(html.encode("utf-8")) < 512 * 1024
    (ROOT / "performance.html").write_text(html, encoding="utf-8")
    scratch = os.environ.get("ORCA_CHAT_SCRATCH_DIR")
    if scratch:
        dest = Path(scratch) / SCRATCH_NAME
        dest.write_text(html, encoding="utf-8")
    print(json.dumps({"report": str(ROOT / "report.md"), "html_bytes": len(html.encode("utf-8")),
                      "scratch": SCRATCH_NAME if scratch else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
