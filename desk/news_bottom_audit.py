"""뉴스 근거의 2026년 복기. 가격 구간 추정이나 주문 계산에는 쓰지 않는다."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import config, news

PB = config.ROOT / "data/derived/analysis/pb_flow_census_20261026/census.csv"
CALLS = config.ROOT / "data/derived/analysis/bottom_zone_20261111/news/v2/calls_census.csv"
H1, H2, END = map(pd.Timestamp, ("2026-01-01", "2026-07-01", "2026-12-31"))
LABELS = {
    "price": ("저점에서 7% 반등", "반등 뒤 확인"),
    "D": ("PB 기록적 매도", "매도 압력 관찰"),
    "R": ("PB 매수 복귀·숏 청산", "매물 소화 관찰"),
    "S": ("PB 극단적 숏", "반등 연료 관찰"),
    "T": ("PB 롱·레버리지 쏠림", "포지션 위험 관찰"),
    "upgrade": ("전략가 비중 상향", "의견 변화 관찰"),
    "downgrade": ("전략가 비중 하향", "의견 변화 관찰"),
    "target_raise": ("전략가 지수 목표 상향", "의견 변화 관찰"),
    "target_cut": ("전략가 지수 목표 하향", "의견 변화 관찰"),
    "ledger": ("원장에 제때 기록한 매수 복귀", "전방 기록"),
}


def _after_date(value: str) -> pd.Timestamp:
    """공개일의 시각을 모르면 다음 달력 날짜 이후 첫 정규장 종가에서 평가."""
    return config.market_calendar().date_to_session(pd.Timestamp(value) + pd.Timedelta(days=1), direction="next")


def _archive_dates(path: Path, category: str, day: str, precision: str) -> dict:
    groups = defaultdict(set)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            value, key = row[day], row[category]
            if key not in LABELS or not value.startswith("2026-") or row[precision] == "unknown":
                continue
            anchor = (news.anchor_session({"time_precision": "exact", "available_at": value})
                      if row[precision] == "exact" and "T" in value else _after_date(value[:10]))
            groups[key].add(anchor)
    return groups


def _ledger_dates(ledger: dict, cutoff: pd.Timestamp) -> set:
    """원장 전방 R은 실제 기록 후 맞는 첫 장에서 평가. 알려진 정정과 재보도는 한 번만."""
    dates = set()
    seen = set()
    calendar = config.market_calendar()
    episodes = {ep["id"] for ep in ledger["episodes"] if ep["benchmark"] == "SOXX"}
    for row in sorted(ledger["events"], key=lambda r: news.stamp(r["recorded_at"])):
        if row["episode"] not in episodes:
            continue
        anchor = news.forward_r_anchor(row, cutoff)
        if anchor is None or anchor.year != 2026:
            continue
        if calendar.session_close(anchor) > cutoff or row["source_group"] in seen:
            continue
        visible_ids = {r["id"] for r in news.visible_events(ledger, calendar.session_close(anchor), "recorded")}
        if row["id"] in visible_ids:
            seen.add(row["source_group"])
            dates.add(anchor)
    return dates


def _counts(fr, dates, start, end, cutoff, cooldown: bool) -> dict:
    from .bottom import HOLD

    # 같은 달력 세션에 맞춰 누락 종가를 보존한다. 앞선 실패도 20일이 끝난 뒤에만 분모에 넣는다.
    dates = sorted(day for day in dates if start <= day <= min(end, cutoff))
    hits = n = pending = missing = 0
    early_breaks = []
    next_date = start
    calendar = config.market_calendar()
    for day in dates:
        if day not in fr.index or not fr.at[day, "in_dd"] or (cooldown and day < next_date):
            continue
        last = calendar.session_offset(day, HOLD)
        if cooldown:
            next_date = last
        if last > cutoff:
            pending += 1
            watched = fr.loc[calendar.next_session(day):cutoff, "close"]
            below = watched[watched < fr.at[day, "low"] - 1e-12]
            if not below.empty:
                early_breaks.append({"signal": day.date().isoformat(), "break": below.index[0].date().isoformat()})
            continue
        future = fr.loc[calendar.next_session(day):last, "close"]
        if len(future) != HOLD or future.isna().any() or pd.isna(fr.at[day, "low"]):
            missing += 1
            continue
        hits += int(future.min() >= fr.at[day, "low"] - 1e-12)
        n += 1
    return dict(hits=hits, n=n, pending=pending, missing=missing, early_breaks=early_breaks)


def _ledger_review(ledger: dict, fr: pd.DataFrame, latest: pd.Timestamp) -> str:
    """화면 원장의 실제 SOXX 기사별 당시 저점을, 기사 공개일 기준으로 사후 복기."""
    labels = {"pressure": "매도 압력", "relief": "매물 소화", "squeeze": "숏 쏠림", "fundamental": "사업·신용 위험"}
    episodes = {ep["id"] for ep in ledger["episodes"] if ep["benchmark"] == "SOXX"}
    cutoff = config.market_calendar().session_close(latest)
    observed = set()
    dates = defaultdict(set)
    unknown = 0
    for row in news.visible_events(ledger, cutoff, "recorded"):
        if row["episode"] not in episodes or not row["published_date"].startswith("2026-") or row["source_group"] in observed:
            continue
        observed.add(row["source_group"])
        anchor = news.anchor_session(row)
        if anchor is None:
            unknown += 1
        else:
            dates[row["kind"]].add(anchor)
    parts = []
    for kind, days in dates.items():
        stats = _counts(fr, days, H1, END, latest, False)
        resolved = stats["n"] + len(stats["early_breaks"])
        parts.append(f"{labels[kind]} {stats['hits']}/{resolved}")
    summary = " · ".join(parts) if parts else "평가할 날짜 없음"
    return (f"2026년 SOXX 기사 원장 {len(observed)}개 원자료 중 공개 시각 미확인 {unknown}개. "
            f"공개 뒤 첫 종가 기준, 하락 국면의 저점 유지/결과가 난 날짜는 {summary}입니다. 사후 수집 기사와 조기 이탈을 포함합니다.")


def evidence_audit(fr: pd.DataFrame) -> dict:
    """완료 종가로 만든 bottom.frame. 상반기 컷 뒤 재조정 없이 같은 정의로 하반기를 평가한다."""
    from .bottom import signals

    calendar = config.market_calendar()
    latest = fr.index[-1]
    last_2026 = min(END, latest)
    dates = _archive_dates(PB, "family", "published_at", "time_precision")
    dates.update(_archive_dates(CALLS, "action", "published_date", "time_precision"))
    dates["price"] = {pd.Timestamp(r["date"]) for r in signals(fr)}
    ledger = news.read_ledger()
    dates["ledger"] = _ledger_dates(ledger, calendar.session_close(latest))
    coverage = {key: _after_date("2026-10-23") for key in ("D", "R", "S", "T")}
    coverage.update({key: _after_date("2026-11-10") for key in ("upgrade", "downgrade", "target_raise", "target_cut")})
    in_dd = fr.index[fr.in_dd & (fr.index >= H1)]
    baseline = {}
    rows = []
    for key, (label, role) in LABELS.items():
        start = pd.Timestamp(news.FIELDS_FROM[:10]) if key == "ledger" else H1
        end = min(coverage.get(key, END), last_2026)
        anchors = dates.get(key, ())
        stats = _counts(fr, anchors, max(start, H2), end, latest, key != "price")
        prior = _counts(fr, anchors, start, min(H2 - pd.Timedelta(days=1), end),
                        min(latest, H2 - pd.Timedelta(days=1)), key != "price")
        key_range = max(start, H2), end
        if key_range not in baseline:
            baseline[key_range] = _counts(fr, in_dd, *key_range, latest, False)
        base = baseline[key_range]
        detail = (f"상반기 {prior['hits']}/{prior['n']}회 유지 · 하반기 발생일은 "
                  f"{end:%m/%d}까지" if key != "ledger" else "10/28부터 공개 48시간 안에 기록한 기사만 평가")
        if stats["early_breaks"]:
            failed = stats["early_breaks"][-1]
            detail += (f" · 20일 미도래 중 이미 이탈 {len(stats['early_breaks'])}건"
                       f" ({failed['signal'][5:]} 신호 → {failed['break'][5:]})")
        rows.append(dict(id=key, label=label, role=role, detail=detail, **stats,
                         baseline_hits=base["hits"], baseline_n=base["n"],
                         first_half=prior))
    return {
        "as_of": latest.date().isoformat(), "period": "2026년 하반기 · SOXX",
        "method": "하락 국면에서 신호가 나온 종가 이후 20거래일간 당시 저점을 지킨 횟수입니다. "
                  "같은 종류 뉴스는 20거래일 간격으로 셉니다. 비교값은 같은 기간 하락 국면의 모든 완료일입니다.",
        "rows": rows,
        "ledger_review": _ledger_review(ledger, fr, latest),
        "limitations": "7% 반등은 운영 규칙의 발생일 기준입니다. 같은 하락에서 관찰일이 겹쳐 비교 일수가 독립 사례 수보다 많습니다. "
                       "기사 유형은 사후에 분류했고 반등 폭도 다릅니다. 표본이 작은 비율은 가격 예측 가중치에 쓰지 않습니다.",
    }
