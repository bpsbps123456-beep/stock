"""Render the frozen study's audit-ready Korean report from its saved outputs."""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def pct(x):
    return "—" if x is None else f"{x:+.2f}%"


def three_minute_section(model, quote, signals, trades):
    """Describe the explicitly requested extension without reselecting the RSI."""
    selected = next(x for x in model["walk_forward"] if x["tf"] == "3m")
    actual = next(x for x in quote["walk_forward"] if x["tf"] == "3m")
    ex = selected["selected_exit"]
    first, later = selected["H1"]["costs"]["10"], selected["H2"]["costs"]["10"]
    qlater = actual["H2"]["quote"]
    ci = qlater.get("mean_ci95_pct", [None, None])
    days = {tf: set(signals[signals.tf.eq(tf) & signals.signal].day) for tf in ("1m", "3m", "1m42")}
    h2same = {d for d in days["1m"] if d >= "2026-07-01"} == {d for d in days["3m"] if d >= "2026-07-01"}
    lines = ["## 3분봉 RSI14 추가 테스트", "",
             "2026-11-05 사용자 추가 요청. 정규장09:30 기준으로3분봉을 만들고,15:27~15:30 마지막 완성 봉의 RSI≤30을 판단했다.15:31 매수·청산4시각·비용은 기존과 같다.", "",
             f'신호는3분봉 {len(days["3m"])}일,1분봉 {len(days["1m"])}일이며 {len(days["3m"] & days["1m"])}일이 겹친다. 2026 전체 가격을 재사용한 후속 연구다.', "",
             "| 매도 ET |3분봉 유효 건수 |3분봉 평균(호가) |승률 |최악 최종 손익 |1분봉 평균(호가) |",
             "|---|---:|---:|---:|---:|---:|"]
    for hm in ("16:30", "17:00", "18:00", "19:00"):
        a, b = [next(c for c in quote["cells"] if c["tf"] == tf and c["exit_hm"] == hm and c["period"] == "all")["quote"] for tf in ("3m", "1m")]
        lines.append(f'| {hm} | {a["n"]} | {pct(a.get("mean_pct"))} | {a.get("win_rate_pct", 0):.1f}% | {pct(a.get("worst_pct"))} | {pct(b.get("mean_pct"))} |')
    lines += ["", "1분봉·3분봉19:00의 유효 표본은각11건이며 다른 청산은각12건이다. 서로 다른 신호일의 평균을 나란히 놓은 표이므로 수익 차이를 봉 주기의 인과 효과로 해석하지 않는다.", "",
              f'상반기3분봉 {first["n"]}건의 한쪽0.10% 모델 비용 후 평균으로 고른 청산은 **{ex} ET**다. 하반기는 {later["n"]}건, 모델 평균 {pct(later.get("mean_pct"))}, 실제 호가 평균 **{pct(qlater.get("mean_pct"))}**, 평균95% 구간 {pct(ci[0])}~{pct(ci[1])}다.',
              f'하반기 신호일이 기존1분봉과 완전히 같은가: **{"예" if h2same else "아니오"}**. 같은 날짜·같은 체결이면 결과도 같으므로 독립 확인 횟수로 더하지 않는다.', "",
              "|3분봉·선택 청산의 감도 |비용0 |한쪽0.05% |한쪽0.10% |한쪽0.20% |", "|---|---:|---:|---:|---:|"]
    c = next(c for c in model["cells"] if c["tf"] == "3m" and c["exit_hm"] == ex and c["period"] == "all")
    lines.append(f'| {ex} | ' + " | ".join(pct(c["costs"][bp].get("mean_pct")) for bp in ("0", "5", "10", "20")) + " |")
    timed = next(w for w in model["walk_forward"] if w["tf"] == "1m42")
    lines += ["", f'3분RSI14는 명목42분,1분RSI14는14분이다. 시간 길이 대조용1분RSI42는 {len(days["1m42"])}일 신호였고, 하반기 {timed.get("H2", {}).get("costs", {}).get("10", {}).get("n", 0)}건이라 비교 근거가 부족하다. 명목 길이를 맞춰도 서로 다른 봉 종가와 RMA 감쇠를 쓰므로 동일 필터가 아니다.',
              f'3분봉 선택 청산까지의 최악 중간 하락(분봉 저가 기준)은 {pct(c["mae_worst_pct"])}였다. 손절·익절은 추가하지 않았다.',
              f'사전 판정 통과: **{"예" if selected["passes_frozen_gate"] else "아니오"}**. 주기 추가는4개 청산의 후속 탐색이며 기존 자료의 재사용·작은 표본 한계가 남는다.', ""]
    return lines


def main():
    j = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    q = json.loads((OUT / "quotes_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    trades = pd.read_csv(OUT / "trades.csv")
    signals = pd.read_csv(OUT / "signals.csv")
    quotes = pd.read_csv(OUT / "quotes_trades.csv")
    lines = [
        "# SOXL 마감 30분 전 RSI ≤30 매수 → 당일 애프터장 매도",
        "", "2026-11-05 · Path-Q 연구 · 사전 설계: [BRIEF.md](BRIEF.md)", "",
        "**주 판정(1분봉 RSI14): 평균 수익은 양수지만 하반기 확인 표본이 3건뿐이라 채택 근거 부족.**",
        "사용자 확정 조건의 정본은 docs/desk_system.md §7이다. 3분봉은 사용자 추가 테스트,5분봉·일봉은 최초 참고 결과다. 시각을 고른 상반기를 포함한 전체 성과를 새 구간 검증 성과로 읽으면 안 된다.", "",
        "## 조건과 자료", "",
        "- 2026-01-02~11-03 미국 211거래일. 상반기 123일로 청산 시각을 고르고 하반기 88일에 고정 적용했다.",
        "- 사용자 확정에 따라 1분봉 RSI14만 주 판정으로 읽는다. 최초 계산한 5분·진행 중 일봉 결과는 참고로 보존했으며 주기 간 최적화는 하지 않았다.",
        "- 정확히 15:30 ET의 RSI≤30을 판정한다. 마지막 30분 중 언제든 처음 하회하는 규칙과는 다르다.",
        "- RSI는 정규장 봉만 연결한 Wilder14. 1분봉15:29 종가, 5분봉15:25~15:30 종가, 일봉은 어제까지 상태+15:29 가격을 사용했다.",
        "- 판단 후 1분 지연한15:31 매수,16:30·17:00·18:00·19:00 ET 고정 매도. 손절·익절을 넣지 않았다.",
        "- 가격은 Alpaca SIP 분할 반영 1분봉. 기존 두 캐시 겹침 OHLC 차이0. 2026 정규장 누락0, 반일장0. 2025는 지표 초기화만 사용했다.",
        "- 일부 무신호일 청산 분은 없지만, 실제 RSI 신호일 진입·청산 분은 모두 있다. 누락 분을 가짜 가격으로 채우지 않았다.",
        "- 수수료0. 모델의 한쪽 비용0.10%는 가정이며 실측 비용이 아니다. 실제 호가 비교는 매수ask/매도bid, 추가 슬리피지·환율·세금·규모 제약은 제외했다.", "",
        "## 전체 구간의 호가 반영 결과", "",
        "시각별 매수·매도 직전 30초 이내 마지막 유효 SIP 호가로 계산한 체결 근사다. 체결 확정 기록이 아니다.", "",
        "| 매도 ET | 1분봉 건수(주) | 1분봉 평균(주) | 승률 | 5분봉 건수(참고) | 5분봉 평균(참고) | 승률 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ex in ("16:30", "17:00", "18:00", "19:00"):
        a, b = [next(c for c in q["cells"] if c["tf"] == tf and c["exit_hm"] == ex and c["period"] == "all") for tf in ("1m", "5m")]
        aa, bb = a["quote"], b["quote"]
        lines.append(f'| {ex} | {aa["n"]} | {pct(aa["mean_pct"])} | {aa["win_rate_pct"]:.1f}% | {bb["n"]} | {pct(bb["mean_pct"])} | {bb["win_rate_pct"]:.1f}% |')
    lines += ["", "1분봉 신호는12일,5분봉은10일이며3일 겹쳐 총19일이다. 1분봉19:00은 1/13의 호가가68.51초 전 값이라 1건을 제외했다. 유리한 호가를 찾아 앞 시점으로 바꾸지 않았다.",
              "일봉 RSI 최저는30.254로 신호0건이었다. 결과를 보고 기준을31로 완화하지 않았다.", "",
              "## 2026년 내 전방 검증", "",
              "RSI별로 상반기 한쪽0.10% 비용 후 평균이 가장 높은 시각을 골랐고, 모두19:00이었다. 실제 호가를 보고 재선택하지 않았다.", "",
              "| RSI | 상반기 모델 건수/평균 | 하반기 건수 | 하반기 모델 평균 | 하반기 실제 호가 평균 | 호가 평균 95% 구간 |",
              "|---|---:|---:|---:|---:|---|" ]
    for tf in ("1m", "5m"):
        model = next(w for w in j["walk_forward"] if w["tf"] == tf)
        quote = next(w for w in q["walk_forward"] if w["tf"] == tf)
        first, second = model["H1"]["costs"]["10"], model["H2"]["costs"]["10"]
        actual = quote["H2"]["quote"]
        ci = actual["mean_ci95_pct"]
        lines.append(f'| {tf} | {first["n"]} / {pct(first["mean_pct"])} | {second["n"]} | {pct(second["mean_pct"])} | {pct(actual["mean_pct"])} | {pct(ci[0])}~{pct(ci[1])} |')
    lines += ["", "사전 최소20건·평균 양수·95% 구간 하한 양수·매일 같은 시각 매수 대조보다 우수라는 조건을 두었다. 두 가설 모두 하반기 건수와 구간 조건을 통과하지 못했다.20건은 연구 설계 기준이지 통계적 안전을 보장하는 숫자가 아니다.",
              "부트스트랩은 각 가설의 하루1거래 수익을10,000회 복원 추출(seed20261105)했다. 거래 수가2~3개인 하반기 구간은 관측한 손실 밖의 꼬리를 충분히 표현할 수 없다.", "",
              "## 비용·진입 지연 감도와 대조군", "",
              "아래는 호가가 없는 행을 뺀 결과가 아닌, 모든 신호일의 분봉 모델 결과다. 한쪽 비용은 매수 가격 가산·매도 가격 차감에 각각 적용했다.", "",
              "| RSI /19:00 매도 | 비용0 | 한쪽0.05% | 한쪽0.10% | 한쪽0.20% |",
              "|---|---:|---:|---:|---:|" ]
    for tf in ("1m", "5m"):
        c = next(c for c in j["cells"] if c["tf"] == tf and c["exit_hm"] == "19:00" and c["period"] == "all")
        lines.append(f'| {tf} | ' + " | ".join(pct(c["costs"][bp]["mean_pct"]) for bp in ("0", "5", "10", "20")) + " |")
    lines += ["", "| RSI /19:00 매도·한쪽0.10% |15:30 즉시 매수 |15:31 주 조건 |15:32 매수 |하반기 매일 매수 대조 평균 |", "|---|---:|---:|---:|---:|"]
    for tf in ("1m", "5m"):
        a = trades[trades.tf.eq(tf) & trades.signal & trades.exit_hm.eq("19:00")]
        c = next(c for c in j["cells"] if c["tf"] == tf and c["exit_hm"] == "19:00" and c["period"] == "H2")
        lines.append(f'| {tf} | ' + " | ".join(pct(a[a.delay.eq(d)].net_10bp_pct.mean()) for d in (0, 1, 2)) + f' | {pct(c["all_day_net10bp"]["mean_pct"])} |')
    lines += ["", "매일 매수 대조는 같은 종목·시각·비용의 무조건 매수다. 신호가 뜬 날만의 장세와 다르므로 평균 차이가 RSI의 인과 효과를 입증하지 않는다. 호가 대조는 신호일만 수집했으므로 무신호일 호가 성과를 가정해 만들지 않았다.", "",
              "## 손실·애프터 보유의 의미", "",
              "19:00까지 보유한 신호일의 분봉 저가 기준 중간 최대 하락은 두 가설 모두−2.32%(2026-03-06)였다. 최종 손익이 작은 손실이어도 중간 손실이 작다는 뜻은 아니다.", "",
              "| RSI |16:00 직후 호가 평균 |19:00 동일 유효 표본의 평균 |최고2건 제외19:00 평균 |",
              "|---|---:|---:|---:|" ]
    for tf in ("1m", "5m"):
        end = quotes[quotes.tf.eq(tf) & quotes.exit_hm.eq("19:00") & quotes.quote_valid]
        early = quotes[quotes.tf.eq(tf) & quotes.exit_hm.eq("16:00") & quotes.quote_valid & quotes.day.isin(end.day)]
        trimmed = end.quote_return_pct.sort_values().iloc[:-2]
        lines.append(f'| {tf} | {pct(early.quote_return_pct.mean())} | {pct(end.quote_return_pct.mean())} | {pct(trimmed.mean())} |')
    lines += ["", "위 차이는 같은 날의 청산 시각 비교다. 애프터 보유가 이 표본에서 평균을 높였지만,19:00은 탐색한4개 시각 중 상반기 최고값이었다. 전체 구간의 구간 추정은 시각 선택 편향을 보정하지 않았다.", "",
              "## 탐색 범위·검증·실행 가능 시간", "",
              "- 원래1분봉4칸·최초참고 포함12칸에 이번3분봉4칸·시간길이 대조1분RSI42의4칸을 더해 누적20칸이다. 문턱30은 고정. RSI42는 시간 길이 대조 하나이며 길이 탐색이 아니다. 비용4개·지연3개는 감도만 확인했다.16:00 비교 청산은 선택 대상 밖이다.",
              f'- 독립 재계산: RSI{audit["signal_rows"]}행·모델 거래{audit["trade_rows"]:,}행 일치. 미래 가격7배 변조와 미래 자료 삭제에도 과거 신호 불변. [audit.json](results/audit.json).',
              "- 전체 py -m unittest 407개 통과. 기존 호가 검증 모듈 테스트9개도 별도 통과했다.",
              f'- 호가 요청{q["n_unique_requests"]}개 중{q["n_unique_valid"]}개 유효. 제외 사유: {q["invalid_quote_reasons"]}。양수 bid/ask·양수 수량·비교차·시각≤체결 시각·나이≤30초를 확인했다. 원호가/분할가격 비율 변동 경고는 quotes_trades.csv에 보존했다.',
              "- SIP 호가 잔량은 해당 크기 전체의 체결을 보장하지 않는다. 애프터장에서는 지정가·예약 주문 지원 조건도 증권사마다 확인해야 한다.",
              "- 매수 결정15:30 ET는 다음 날 KST04:30(서머타임)/05:30(표준시),19:00 ET 매도는 KST08:00/09:00이다. 주 조건 매수는 여기서1분 늦다.",
              "- 기존 docs/desk_system.md §1 직접 주문 가능 시간08~24 밖의 진입이다. 사용자 요청대로 연구했지만 예약·조건 주문으로 RSI 신호 연동과 애프터 체결이 가능한지는 확인되지 않았다. 기존 운용 규칙은 변경하지 않았다.",
              "- 현재211일은 모두 열어 보았다. 새 확인 구간은 앞으로 쌓이는2026년 기록이어야 하며, 결과를 살리기 위해2024년 이전으로 넓히지 않았다.", "",
              "## 재현", "", "저장소 루트에서:", "", "```powershell",
              "py data/derived/analysis/soxl_close_rsi/code/study.py",
              "py data/derived/analysis/soxl_close_rsi/code/audit.py",
              "py data/derived/analysis/soxl_close_rsi/code/quotes.py --offline",
              "py data/derived/analysis/soxl_close_rsi/code/report.py",
              "py -m unittest", "```", "",
              "`--offline`은 이 PC의 Git 제외 호가 캐시를 사용한다. 다른 환경은 기존 fetch_alpaca.one으로 SOXL2025-01-01~2026-11-03(끝2026-11-04) 봉을 두 입력 폴더에 준비하고, quotes.py를--offline 없이 실행해야 한다. 키는 Git 제외.env에 둔다. 원시 봉과 호가는 올리지 않으며 출처 파일 해시는summary.json에 기록했다.", "",
              "출처: [Alpaca 분봉 생성·시작 시각](https://alpaca.markets/learn/stock-minute-bars), [Historical bars API](https://docs.alpaca.markets/us/reference/stockbars), [Wilder RSI 설명](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi).", ""]
    lines[7:7] = three_minute_section(j, q, signals, trades)
    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(ROOT / "report.md")


if __name__ == "__main__":
    main()
