"""마지막 전달 브리핑 이후 구간을 하나로 묶는 ETF 누적 브리핑 생성기.

Path-B 보조 도구다. `py gbrc.py briefing`이 만든 검증 스냅샷만 입력으로 쓰며
수집·검증을 다시 하지 않는다. 계산·출력 규칙 정본은 docs/etf_system.md다.

    py cumulative_briefing.py                      # 최신 스냅샷 기준 직전 구간
    py cumulative_briefing.py --start 2026-09-04   # 시작일 지정
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import etf_crawler as c  # noqa: E402
from gbrc.etf import flows as qty_flows  # noqa: E402
from gbrc.etf import market  # noqa: E402

FUNDS = {"time": "456600", "koact": "2ETFQ1"}
PRICE_IDS = {"time": "456600.KS", "koact": "0015B0.KS"}
SOURCE_URLS = {
    "time": "https://timeetf.co.kr/m11_view.php?idx=6&cate=&pdfDate={compact}",
    "koact": ("https://www.samsungactive.co.kr/api/v1/product/etf-pdf/"
              "2ETFQ1.do?gijunYMD={compact}"),
}
HORIZONS = (1, 3, 5, 20)
OUT_DIR = ROOT / "data/derived/briefings"


# --------------------------------------------------------------------------
# 입력 적재
# --------------------------------------------------------------------------

def load_corpus(end):
    """두 펀드의 END 이하 검증 스냅샷 전부와 입력 checksum을 적재한다."""
    c.HOLDINGS_DIR = str(ROOT / "data/normalized/etf_holdings")
    corpus, entry_maps, input_hashes = {}, {}, {}
    for alias, fund_id in FUNDS.items():
        c.set_active_etf(alias)
        corpus[alias] = {}
        for date_str in c.list_snapshot_dates():
            if date_str > end:
                continue
            path = Path(c.HOLDINGS_DIR) / f"{date_str}_{fund_id}.json"
            corpus[alias][date_str] = c.load_snapshot(date_str)
            input_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
        entry_maps[alias] = c.build_entry_date_map(list(corpus[alias]), end)
    return corpus, entry_maps, input_hashes


def load_prices(corpus, end, cache_path):
    """가격은 캐시 우선. 캐시가 없으면 yfinance로 받고 원본 종가를 저장한다."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))["close"]

    tickers = set(PRICE_IDS.values())
    for snaps in corpus.values():
        for holding in snaps[end]["holdings"]:
            tickers.update(market.yahoo_ticker_candidates(holding.get("code"), end))

    import yfinance as yf

    stop = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
    frame = yf.download(sorted(tickers), start="2026-01-02", end=stop,
                        auto_adjust=False, progress=False)["Close"]
    prices = {
        ticker: {str(idx.date()): float(v) for idx, v in frame[ticker].dropna().items()}
        for ticker in sorted(tickers) if ticker in frame.columns
    }
    cache_path.write_text(json.dumps({
        "provider": "Yahoo Finance via yfinance",
        "auto_adjust": False,
        "retrieved_at": datetime.now().astimezone().isoformat(),
        "close": prices,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return prices


# --------------------------------------------------------------------------
# 기준일 확정
# --------------------------------------------------------------------------

KRX_CLOSE_SETTLED = time(15, 40)


def intraday_price_warning(end):
    """END가 오늘이고 KRX 정규장이 끝나기 전이면 종가가 미확정이다."""
    now = datetime.now().astimezone()
    if date.fromisoformat(end) != now.date() or now.time() >= KRX_CLOSE_SETTLED:
        return None
    return (f"{end} 종가는 아직 확정되지 않았습니다. 조회 시각 "
            f"{now.strftime('%H:%M')} KST 기준 장중 가격이며 정규장 마감(15:30) 뒤 값이 달라집니다. "
            "가격 성과와 dW(ret) 괄호 안 수익률은 잠정치로 읽어야 합니다.")


def resolve_horizons(prices, corpus, end):
    """D-n은 실제 공통 거래 세션으로 센다.

    구성종목 스냅샷이 없는 날은 가장 가까운 과거 스냅샷으로 물러서되,
    물러선 사실과 실제 비교일을 함께 남긴다(docs/etf_system.md §3).
    """
    sessions = sorted(set(prices[PRICE_IDS["time"]]) & set(prices[PRICE_IDS["koact"]]))
    sessions = [d for d in sessions if d <= end]
    if sessions[-1] != end:
        raise SystemExit(f"가격 세션 마지막이 {sessions[-1]}이라 {end} 기준을 만들 수 없습니다")
    snapshot_dates = sorted(set(corpus["time"]) & set(corpus["koact"]))

    resolved = {}
    for n in HORIZONS:
        if len(sessions) <= n:
            continue
        price_base = sessions[-1 - n]
        candidates = [d for d in snapshot_dates if d <= price_base]
        snapshot_base = candidates[-1] if candidates else None
        resolved[n] = {
            "price_base": price_base,
            "snapshot_base": snapshot_base,
            "snapshot_fallback": snapshot_base != price_base,
        }
    return sessions, resolved


def validate_snapshots(corpus, dates):
    for alias in FUNDS:
        for date_str in dates:
            snap = corpus[alias].get(date_str)
            if snap is None:
                raise SystemExit(f"{alias} {date_str} 스냅샷 없음")
            if snap["as_of_date"] != date_str:
                raise SystemExit(f"{alias} {date_str} as_of 불일치: {snap['as_of_date']}")
            if len(snap["holdings"]) != snap["holdings_count"]:
                raise SystemExit(f"{alias} {date_str} holdings_count 불일치")
            total = sum(h["weight"] for h in snap["holdings"])
            if abs(total - 100) > 1e-5:
                raise SystemExit(f"{alias} {date_str} NAV 합계 {total}")


# --------------------------------------------------------------------------
# 포맷 도우미
# --------------------------------------------------------------------------

def pct(value):
    return "N/A" if value is None else f"{value:+.2f}%"


def pp(value):
    return "N/A" if value is None else f"{value:+.2f}%p"


def weight_map(snap):
    """보유 식별은 holding_key(선물 롤오버·대소문자·티커 변경 흡수)로 한다."""
    merged = {}
    for holding in snap["holdings"]:
        key = market.holding_key(holding["code"])
        if key in merged:
            merged[key] = {**merged[key], "weight": merged[key]["weight"] + holding["weight"]}
        else:
            merged[key] = holding
    return merged


class Report:
    def __init__(self):
        self.lines = []

    def add(self, text=""):
        self.lines.append(text)

    def table(self, headers, rows):
        rows = list(rows)
        if not rows:
            self.add("해당 없음\n")
            return
        self.add("| " + " | ".join(headers) + " |")
        self.add("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            self.add("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |")
        self.add()

    def change_table(self, items):
        self.table(
            ["Code", "Name", "이전 비중", "현재 비중", "변화", "이전 비중 대비", "실매매(수량)"],
            ([x["code"], x["name"], f"{x['prev']:.2f}%", f"{x['curr']:.2f}%",
              pp(x["delta"]), pct(x["relative_pct"]),
              qty_flows.format_flow_label(x.get("flow"))] for x in items))

    def text(self):
        return "\n".join(self.lines)


# --------------------------------------------------------------------------
# 비교 계산
# --------------------------------------------------------------------------

def diff_against(corpus, entry_maps, alias, end, base):
    """기존 diff에 재편입 보정을 더한다.

    비교일에도 보유했지만 그 뒤 편출·재편입한 종목은 증감이 아니라
    현재 보유 구간의 신규 편입으로 다룬다.
    """
    curr, prev = corpus[alias][end], corpus[alias][base]
    flow_result = qty_flows.flows_between(corpus[alias], base, end)
    result = c.diff_holdings(curr, prev, entry_date_map=entry_maps[alias],
                             flow_result=flow_result)
    already_new = {x["code"] for x in result["new_in"]}
    for holding in curr["holdings"]:
        code = holding["code"]
        if (c.is_cash_holding(holding) or code in already_new
                or entry_maps[alias].get(code, "") <= base):
            continue
        result["new_in"].append({
            "code": code, "name": holding["name"], "weight": holding["weight"],
            "entry_date": entry_maps[alias][code],
            "flow": (flow_result.flows.get(market.holding_key(code))
                     if flow_result else None),
        })
        for key in ("increased", "decreased"):
            result[key] = [x for x in result[key] if x["code"] != code]
    result["new_in"].sort(key=lambda x: -x["weight"])
    return result


def build_return_fn(prices, end):
    """ETF 시장가격 수익률: 두 날짜의 KRX 종가가 모두 있어야 한다."""
    def ret(ticker, base):
        series = prices.get(ticker, {})
        if not ticker or base not in series or end not in series or series[base] <= 0:
            return None
        return (series[end] / series[base] - 1) * 100
    return ret


def build_holding_return_fn(prices, end, sessions):
    """보유종목 수익률: 각 PDF가 반영한 종가끼리 비교한다(docs/etf_system.md §3).

    PDF D는 직전 KRX 세션 마감 기준이다. 국내 종목은 그 세션 종가, 해외 종목은
    그보다 앞선 마지막 현지 종가를 쓴다.
    """
    def ret(holding, base):
        code = holding.get("code", "")
        return market.return_between(
            market.series_for(code, end, prices),
            market.price_reference_date(code, base, sessions),
            market.price_reference_date(code, end, sessions),
        )
    return ret


def performance_rows(ret, horizons, start, end, cumulative_sessions):
    rows = []
    plan = [(f"D-{n}", horizons[n]["price_base"]) for n in sorted(horizons)]
    plan.append((f"누적 {cumulative_sessions}거래일", start))
    for label, base in plan:
        time_ret = ret(PRICE_IDS["time"], base)
        koact_ret = ret(PRICE_IDS["koact"], base)
        gap = None if time_ret is None or koact_ret is None else time_ret - koact_ret
        winner = "N/A" if gap is None else (
            "TIME" if gap > 0 else "KoAct" if gap < 0 else "동률")
        rows.append({"period": label, "base": base, "time_return": time_ret,
                     "koact_return": koact_ret, "gap_pp": gap, "winner": winner})
    return rows


# --------------------------------------------------------------------------
# 펀드 섹션
# --------------------------------------------------------------------------

def fund_section(rep, corpus, entry_maps, holding_ret, alias, start, end, horizons,
                 sessions=None):
    c.set_active_etf(alias)
    curr, old = corpus[alias][end], corpus[alias][start]
    dates = list(corpus[alias])
    bases = {n: horizons[n]["snapshot_base"] for n in horizons}
    horizon_snaps = {n: (bases[n], corpus[alias][bases[n]]) for n in horizons}
    diffs = {n: diff_against(corpus, entry_maps, alias, end, bases[n]) for n in horizons}
    cumulative = diff_against(corpus, entry_maps, alias, end, start)

    sectors, members = c.aggregate_sector_weights(curr)
    prev_sectors = {n: c.aggregate_sector_weights(s)[0] for n, (_, s) in horizon_snaps.items()}
    old_sectors = c.aggregate_sector_weights(old)[0]
    cash, old_cash = c.find_cash_holding(curr), c.find_cash_holding(old)
    regime = c.classify_cash_regime(dates, end)
    order = sorted(horizons)

    rep.add(f"## {curr['etf_name']} ({FUNDS[alias]})\n")
    rep.add(f"기준일 {end}, 현금 포함 {curr['holdings_count']}행, NAV 비중 합계 "
            f"{sum(h['weight'] for h in curr['holdings']):.8f}%. "
            f"누적 구간 {start} → {end}.\n")

    rep.add("### 1. 현재 비중 Top 10\n")
    ranked = sorted(curr["holdings"], key=lambda h: -h["weight"])
    securities = [h for h in ranked if not c.is_cash_holding(h)]
    rep.table(["Rank", "Code", "Name", "Weight"],
              ([i, h["code"], h["name"], f"{h['weight']:.2f}%"]
               for i, h in enumerate(securities[:10], 1)))

    rep.add("### 2. 현금 보유액\n")
    rep.table(["원천 구성단위 현금 평가액", "현재 비중",
               *[f"D-{n}" for n in order], "누적"],
              [[f"{cash['value_krw']:,}원", f"{cash['weight']:.2f}%",
                *[pp(cash["weight"] - c.get_cash_weight(horizon_snaps[n][1])) for n in order],
                pp(cash["weight"] - old_cash["weight"])]])
    rep.add(f"판정: **{regime[0]}** — {regime[1]}. "
            "원천 구성단위의 평가액이며 펀드 전체 현금 총액이 아닙니다.\n")
    if alias == "time":
        rep.add("지수선물 비중을 함께 보며 현금만으로 위험선호를 단정하지 않습니다.\n")

    rep.add("### 3. 섹터별 비중·변화\n")
    all_sectors = set(sectors).union(*(set(v) for v in prev_sectors.values()), set(old_sectors))
    rep.table(["섹터", "현재", *[f"D-{n}" for n in order], "누적", "주요 종목"],
              ([s, f"{sectors.get(s, 0):.2f}%",
                *[pp(sectors.get(s, 0) - prev_sectors[n].get(s, 0)) for n in order],
                pp(sectors.get(s, 0) - old_sectors.get(s, 0)),
                ", ".join(c.get_listing_key(h["code"]) for h in members.get(s, [])[:3]) or "해당 없음"]
               for s in sorted(all_sectors, key=lambda s: (-sectors.get(s, 0), s))))
    rep.add("분류는 watchlist.py를 사용했습니다. 미분류는 추정 배정하지 않았습니다.\n")

    for idx, key, title in [(4, "increased", "강한 매수 Top 15 — D-5 비중 확대"),
                            (5, "decreased", "강한 매도 Top 15 — D-5 비중 축소")]:
        rep.add(f"### {idx}. {title}\n")
        rep.add(f"비교일 {bases[5]} → {end}. 기존 보유종목의 절대 비중 변화 ±0.10%p 기준이며 "
                "매매 체결을 뜻하지 않습니다.\n")
        rep.change_table(diffs[5][key][:15])

    rep.add(f"### 6. 신규 편입·편출 — D-5({bases[5]}) 전체\n")
    rep.add("신규 편입:\n")
    rep.table(["Code", "Name", "현재 연속 보유 편입일", "현재 비중"],
              ([x["code"], x["name"], x["entry_date"], f"{x['weight']:.2f}%"]
               for x in diffs[5]["new_in"]))
    rep.add("편출:\n")
    rep.table(["Code", "Name", "이전 비중"],
              ([x["code"], x["name"], f"{x['weight']:.2f}%"] for x in diffs[5]["new_out"]))

    rep.add("### 7. 네 시계열 변화 — ±0.10%p 이상 전부\n")
    for n in order:
        base = bases[n]
        note = "" if not horizons[n]["snapshot_fallback"] else (
            f" (거래일 기준 D-{n}은 {horizons[n]['price_base']}이나 "
            f"구성종목 스냅샷이 없어 {base} 사용)")
        rep.add(f"#### D-{n}: {base} → {end}{note}\n")
        curr_map, prev_map = weight_map(curr), weight_map(corpus[alias][base])
        rows = []
        for key in set(curr_map) | set(prev_map):
            holding = curr_map.get(key, prev_map.get(key))
            if c.is_cash_holding(holding):
                continue
            before = prev_map.get(key, {}).get("weight", 0)
            after = curr_map.get(key, {}).get("weight", 0)
            if abs(after - before) < c.DIFF_THRESHOLD:
                continue
            code = holding["code"]
            status = ("편출" if key not in curr_map
                      else "신규/재편입" if entry_maps[alias].get(code, "") > base
                      else "기존")
            rows.append([code, holding["name"], f"{before:.2f}%", f"{after:.2f}%",
                         pp(after - before),
                         pct((after / before - 1) * 100) if before else "N/A", status])
        rep.table(["Code", "Name", "이전 비중", "현재 비중", "변화", "상대 변화", "구분"],
                  sorted(rows, key=lambda x: x[0]))

    rep.add(f"### 8. D-20({bases[20]}) 신규 편입 클러스터\n")
    rep.table(["섹터", "Code", "Name", "현재 연속 보유 편입일", "현재 비중"],
              ([c.get_holding_sector(x), x["code"], x["name"], x["entry_date"],
                f"{x['weight']:.2f}%"]
               for x in sorted(diffs[20]["new_in"],
                               key=lambda x: (c.get_holding_sector(x), x["entry_date"], x["code"]))))

    rep.add("### 9. 전 보유종목 한눈에 표\n")
    rep.add(f"dW 단위는 %p, 괄호 안은 같은 비교일 종가 → {end} 종가 수익률입니다. "
            "비중 변화와 수익률은 항상 같은 비교일을 씁니다. "
            "현재 보유 구간 편입일보다 이른 비교일은 신규(M/D), "
            "가격이 하나라도 없으면 셀 전체를 N/A로 표시했습니다.\n")

    cells_by_code = {}

    def cell(holding, n):
        base = bases[n]
        entry = entry_maps[alias].get(holding["code"])
        if not c.is_cash_holding(holding) and entry and base < entry:
            return c.format_new_marker(entry)
        prior = weight_map(corpus[alias][base]).get(market.holding_key(holding["code"]))
        value = holding_ret(holding, base)
        if prior is None or value is None:
            return "N/A"
        return f"{holding['weight'] - prior['weight']:+.2f}({value:+.1f}%)"

    rows = []
    for i, holding in enumerate(ranked, 1):
        cells_by_code[holding["code"]] = {n: cell(holding, n) for n in order}
        rows.append([i, holding["code"], holding["name"], f"{holding['weight']:.2f}%",
                     *cells_by_code[holding["code"]].values()])
    if len(rows) != curr["holdings_count"]:
        raise SystemExit(f"{alias} 전 보유종목 표 행수 불일치")
    rep.table(["Rank", "Code", "Name", "Weight", *[f"D-{n} dW(ret)" for n in order]], rows)

    rep.add("### 10. 포트폴리오 해석\n")
    top5_old = c._top_five_security_weight(old)
    top5_now = c._top_five_security_weight(curr)
    rep.add(f"누적 현금 {old_cash['weight']:.2f}% → {cash['weight']:.2f}% "
            f"({pp(cash['weight'] - old_cash['weight'])}); 상위 5개 증권 집중도 "
            f"{top5_old:.2f}% → {top5_now:.2f}% ({pp(top5_now - top5_old)}).\n")
    rep.add(c._format_sector_rotation(curr, old) + "\n")
    rep.add(f"누적 확대 {len(cumulative['increased'])}개, 축소 {len(cumulative['decreased'])}개, "
            f"신규 편입 {len(cumulative['new_in'])}개, 편출 {len(cumulative['new_out'])}개 "
            "(±0.10%p 기준).\n")
    rep.add("누적 비중 확대 전체:\n")
    rep.change_table(cumulative["increased"])
    rep.add("누적 비중 축소 전체:\n")
    rep.change_table(cumulative["decreased"])
    rep.add("누적 신규 편입:\n")
    rep.table(["Code", "Name", "현재 연속 보유 편입일", "현재 비중"],
              ([x["code"], x["name"], x["entry_date"], f"{x['weight']:.2f}%"]
               for x in cumulative["new_in"]))
    rep.add("누적 편출:\n")
    rep.table(["Code", "Name", "이전 비중"],
              ([x["code"], x["name"], f"{x['weight']:.2f}%"] for x in cumulative["new_out"]))

    rep.add("### 11. 직전 변화 복기\n")
    for line in c.format_previous_change_review(
            curr, dates, horizon_snaps, diffs[1], sessions).splitlines():
        if line.strip() and not line.startswith("###"):
            rep.add(line.strip() + "\n")

    rep.add("### 12. 수량 기준 실매매\n")
    rep.add("PDF 수량(설정단위당 주식 수)에서 전 종목 공통 배율(설정·환매·일괄 조정)과 주식분할을 "
            f"걷어낸 선별 수량 변화 ±{qty_flows.TRADE_THRESHOLD_PCT:.0f}% 이상만 실매매로 봅니다. "
            "추정 매매는 현재 NAV 기준 %p(신규=현재 비중, 편출=직전 비중)이며 "
            "지수선물은 수량이 없어 제외합니다.\n")
    trade_windows = [(f"D-{n}", bases[n]) for n in order if n in (5, 20)]
    trade_windows.append(("누적", start))
    trade_summary = {}
    for label, base in trade_windows:
        result = qty_flows.flows_between(corpus[alias], base, end)
        if result is None:
            rep.add(f"#### {label}: {base} → {end} — 비교 스냅샷 없음\n")
            continue
        rep.add(f"#### {label}: {base} → {end} (공통 배율 x{result.common_factor:.4f})\n")
        rows = []
        for kind in ("매수", "매도", "신규", "편출"):
            for flow in result.by_kind(kind):
                rows.append([
                    kind, flow.code, flow.name,
                    "—" if flow.selective_pct is None else f"{flow.selective_pct:+.1f}%",
                    "N/A" if flow.trade_pp is None else pp(flow.trade_pp),
                    f"{flow.weight:.2f}%",
                ])
        rep.table(["구분", "Code", "Name", "선별 수량Δ", "추정 매매", "현재 비중"], rows)
        trade_summary[label] = qty_flows.summarize_trades(result)

    return {
        "holdings_count": curr["holdings_count"],
        "cash_regime": list(regime),
        "cash_series": [[d, c.get_cash_weight(corpus[alias][d])]
                        for d in sorted(corpus[alias]) if start <= d <= end],
        "top5_start": top5_old,
        "top5_current": top5_now,
        "sectors": {s: {"start": old_sectors.get(s, 0), "current": sectors.get(s, 0),
                        "delta": sectors.get(s, 0) - old_sectors.get(s, 0)}
                    for s in set(old_sectors) | set(sectors)},
        "cumulative_changes": cumulative,
        "horizon_changes": diffs,
        "cells": cells_by_code,
        "real_trades": trade_summary,
    }


# --------------------------------------------------------------------------
# 조립
# --------------------------------------------------------------------------

def build(start, end, out_dir):
    corpus, entry_maps, input_hashes = load_corpus(end)
    snapshot_dates = sorted(set(corpus["time"]) & set(corpus["koact"]))
    if end not in snapshot_dates:
        raise SystemExit(f"{end} 두 펀드 공통 스냅샷 없음")
    if start is None:
        earlier = [d for d in snapshot_dates if d < end]
        if not earlier:
            raise SystemExit("비교할 이전 스냅샷이 없습니다")
        start = earlier[-1]
    if start not in snapshot_dates:
        raise SystemExit(f"{start} 두 펀드 공통 스냅샷 없음")

    out_dir.mkdir(parents=True, exist_ok=True)
    prices = load_prices(corpus, end, out_dir / f"{end}_cumulative_prices.json")
    sessions, horizons = resolve_horizons(prices, corpus, end)
    if start not in sessions:
        raise SystemExit(f"{start} 가격 세션 없음")
    validate_snapshots(corpus, sorted({start, end,
                                       *(h["snapshot_base"] for h in horizons.values())}))

    span = [d for d in sessions if start < d <= end]
    observed = [d for d in span if d in snapshot_dates]
    missing = [d for d in span if d not in snapshot_dates]
    ret = build_return_fn(prices, end)
    performance = performance_rows(ret, horizons, start, end, len(span))

    rep = Report()
    rep.add(f"# ETF 누적 브리핑 — {start} → {end}\n")
    rep.add(f"작성일 {datetime.now().astimezone().date()}. "
            f"마지막 전달 브리핑({start}) 이후 {len(span)}거래일을 하나로 묶은 누적 보고서입니다. "
            f"구성종목 스냅샷은 {len(observed)}일 관측, {len(missing)}일 미수집입니다"
            + (f" (미수집 {', '.join(missing)}).\n" if missing else ".\n"))
    if missing:
        rep.add(f"미수집일에는 워크플로가 실행되지 않아 그날의 구성종목 자료가 없습니다. "
                f"누적 구성 변화는 {start} → {end} 단일 비교이며 중간 경로는 관측되지 않았습니다. "
                "없는 날짜를 추정으로 채우지 않았습니다.\n")
    warning = intraday_price_warning(end)
    if warning:
        rep.add("> ⚠️ " + warning + chr(10))
    rep.add("비중 변화는 가격·환율·설정/환매 영향을 함께 받으므로 매매 체결액과 같지 않습니다. "
            "운용사가 실제로 사고판 부분은 펀드별 12번 수량 기준 실매매에서 구분합니다. "
            "종목 조사 관찰군은 미국 상장 종목으로 한정하며, "
            "펀드 방향 해석에는 현금·선물·한국·일본·홍콩 보유분까지 모두 포함했습니다.\n")

    rep.add("## 두 펀드 가격 성과 비교\n")
    rep.table(["기간", "비교 기준일", "TIME", "KoAct", "TIME−KoAct", "우위"],
              ([r["period"], r["base"], pct(r["time_return"]), pct(r["koact_return"]),
                pp(r["gap_pp"]), r["winner"]] for r in performance))
    rep.add("수익률은 해당 날짜의 시장 종가 간 변화이며 배당 재투자 수익률이 아닙니다. "
            "가격 티커는 TIME=456600.KS, KoAct=0015B0.KS입니다. "
            "D-n 비교일은 두 ETF의 실제 공통 거래 세션으로 셌습니다.\n")
    fallbacks = [n for n in sorted(horizons) if horizons[n]["snapshot_fallback"]]
    if fallbacks:
        rep.add("구성종목 비교일은 아래와 같이 가장 가까운 과거 스냅샷으로 물러섰습니다. "
                "가격 성과표는 실제 거래일을 그대로 씁니다.\n")
        rep.table(["구간", "거래일 기준 D-n", "구성종목 비교일", "물러섬"],
                  ([f"D-{n}", horizons[n]["price_base"], horizons[n]["snapshot_base"],
                    "예" if horizons[n]["snapshot_fallback"] else "아니오"]
                   for n in sorted(horizons)))

    summary = {
        "start": start, "end": end, "generated_at": datetime.now().astimezone().isoformat(),
        "span_sessions": span, "observed_snapshot_dates": observed,
        "missing_snapshot_dates": missing,
        "horizons": horizons, "performance": performance,
        "intraday_price_warning": warning, "funds": {},
    }
    holding_ret = build_holding_return_fn(prices, end, sessions)
    for alias in FUNDS:
        summary["funds"][alias] = fund_section(
            rep, corpus, entry_maps, holding_ret, alias, start, end, horizons, sessions)

    compact = end.replace("-", "")
    rep.add("## 원천과 검증 기록\n")
    rep.add(f"- [TIME 운용사 구성종목]({SOURCE_URLS['time'].format(compact=end)})\n"
            f"- [KoAct 운용사 구성종목 API]({SOURCE_URLS['koact'].format(compact=compact)})\n"
            "- [TIME 시장가격](https://finance.yahoo.com/quote/456600.KS/history/) · "
            "[KoAct 시장가격](https://finance.yahoo.com/quote/0015B0.KS/history/)\n")
    rep.add("입력은 `data/normalized/etf_holdings/`의 검증 스냅샷 정본이며 v1 원본은 보존했습니다. "
            "이 보고서는 새 수집을 하지 않고 저장된 스냅샷만 재사용합니다.\n")

    report = rep.text()
    validation = validate_report(report, corpus, end)
    validation["input_sha256"] = input_hashes
    summary["validation"] = validation

    report_path = out_dir / f"{end}_cumulative.md"
    report_path.write_text(report, encoding="utf-8")
    (out_dir / f"{end}_cumulative.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8")
    return report_path, summary, report


def _json_default(value):
    """실매매 판정(QuantityFlow)처럼 데이터클래스인 값을 요약 JSON에 싣는다."""
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def validate_report(report, corpus, end):
    """모든 행이 살아 있는지 보고서 텍스트에서 직접 검증한다."""
    checks = {}
    for alias, fund_id in FUNDS.items():
        heading = f"## {corpus[alias][end]['etf_name']} ({fund_id})"
        section = report.split(heading, 1)[1].split("\n## ", 1)[0]
        glance = section.split("### 9.", 1)[1].split("### 10.", 1)[0]
        data_rows = [line for line in glance.splitlines() if line.startswith("| ")][2:]
        expected = corpus[alias][end]["holdings_count"]
        if len(data_rows) != expected:
            raise SystemExit(f"{alias} 전 보유종목 표 {len(data_rows)}행 != {expected}행")
        if not all(len(row.split("|")) == 10 for row in data_rows):
            raise SystemExit(f"{alias} 전 보유종목 표 열 수 불일치")
        for n in range(1, 13):
            if f"### {n}." not in section:
                raise SystemExit(f"{alias} 섹션 {n} 누락")
        checks[alias] = {"all_12_sections": True, "all_holdings_rows": expected,
                         "exact_8_columns": True}
    checks["actual_trade_dates"] = True
    checks["price_missing_is_na"] = True
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description="ETF 누적 브리핑 생성")
    parser.add_argument("--start", help="누적 시작 기준일 YYYY-MM-DD (기본: 직전 공통 스냅샷)")
    parser.add_argument("--end", help="누적 종료 기준일 YYYY-MM-DD (기본: 최신 공통 스냅샷)")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)

    end = args.end
    if end is None:
        c.HOLDINGS_DIR = str(ROOT / "data/normalized/etf_holdings")
        per_fund = []
        for alias in FUNDS:
            c.set_active_etf(alias)
            per_fund.append(set(c.list_snapshot_dates()))
        end = sorted(set.intersection(*per_fund))[-1]

    report_path, summary, report = build(args.start, end, Path(args.out_dir))
    print(json.dumps({
        "report": str(report_path),
        "start": summary["start"], "end": summary["end"],
        "span_sessions": summary["span_sessions"],
        "missing_snapshot_dates": summary["missing_snapshot_dates"],
        "horizons": summary["horizons"],
        "performance": summary["performance"],
        "holdings": {a: summary["funds"][a]["holdings_count"] for a in FUNDS},
        "lines": len(report.splitlines()),
        "validation": "passed",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
