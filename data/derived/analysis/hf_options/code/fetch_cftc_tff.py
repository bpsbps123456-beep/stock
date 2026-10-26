"""헤지펀드 선물 포지션(CFTC TFF, 선물만) 수집 — 옵션·헤지펀드 종합 연구(2026-10-14).

출처: CFTC Public Reporting Environment(Socrata) 데이터셋 gpe5-46if = Traders in Financial Futures, Futures Only.
  계약(코드는 자료의 cftc_contract_market_code·contract_market_name으로 확인):
    nq  NASDAQ MINI 209742 + MICRO E-MINI NASDAQ-100 INDEX 209747 ×0.1  (E-mini 환산)
    es  E-MINI S&P 500 13874A + MICRO E-MINI S&P 500 INDEX 13874U ×0.1
    vx  VIX FUTURES(CFE) 1170E1
    rty RUSSELL E-MINI 239742 (마이크로 239747은 2022~23년 대부분 주가 공개 기준 미달로 빠져 넣지 않는다. 비중 약 1%)
  대조용: NASDAQ-100 Consolidated 20974+, S&P 500 Consolidated 13874+ (CFTC 통합 보고서. 2023-05부터 E-mini 단위,
    그 전은 큰 계약 단위라 E-mini 단위 주만 비교한다).
  마이크로가 공개되지 않은 주(마이크로 S&P 2022-03-22)는 그 묶음 값을 비워 둔다(E-mini만으로 채우지 않는다).
  투자자 구분: dealer(딜러) · am(자산운용) · lev(레버리지 펀드 = 헤지펀드·CTA) · other(기타 보고) · nonrept(보고 기준 미만 소액).
  net = 매수 − 매도(스프레드 포지션 제외). 다섯 구분의 net 합은 계약마다 0이다(매수 합 = 매도 합 = 미결제약정).

알게 된 날(known): 포지션은 화요일 기준(asof), 보통 그 주 금요일 오후 3:30(ET, 장 마감 전) 공개 → known = 공개일.
  - 수~금에 연방 공휴일·연방 휴무가 끼면 다음 영업일(보통 월요일) 공개. 화요일이 공휴일이면 월요일 자료로 금요일 공개.
    이 규칙은 2025 공식 일정(2024-12~2025-09)과 같아야 한다(실행 때 확인, 다르면 멈춘다).
  - 2025-10-01~11-12 연방정부 셧다운으로 발표가 멈췄다. 밀린 14주는 CFTC 2025-12-09 보도자료(9147-25)의 앞당긴 일정과
    2025-12-19 공지(12/23분 → 12/31)대로 늦게 공개됐다. 그 주들은 실제 공개일을 쓴다(오후 3:30 공개라고 공지됨).
  - 2023-02 ION 사이버 사고로 2/3·2/10·2/17·3/17 정기 발표가 없었고 밀린 7주가 2/24~3/21에 공개됐다(CFTC 공지, Mondo Visione 전재).
    공개 시각이 공지되지 않아 그 주들은 공개일 다음 거래일을 known으로 한다(보수적).
  - 공개일이 미국 휴장일(성금요일)이면 다음 거래일을 known으로 한다. 끝나지 않은 장의 날짜는 쓰지 않는다.
  - 실제 공개 시각 확인(검증 2026-10-14 추가): PRE 행의 시스템 칸 :created_at(행이 처음 올라간 시각)을 받아
    뉴욕 시각 오후 4시 이전이면 그 거래일, 아니면 다음 거래일을 시각 기준 known으로 하고, 규칙 known과 비교해 늦은 쪽을 쓴다.
    2022-09-13 첫 일괄 적재처럼 여러 기준일이 같은 시각을 가진 값은 공개 시각이 아니라서 쓰지 않는다(그 주는 규칙만).
    2025-07-18(17:45)·2026-07-17(16:54) 공개분이 장 마감 뒤라 다음 월요일로 밀린다.

실행: py fetch_cftc_tff.py [--since 2022] [--gap 1.1] [--verify]  (--verify: Mondo Visione 게시일로 공개일 대조, 쪽당 약 1분)
이어 받기: 연도별 원본 JSON을 cache/raw_cftc_tff/에 둔다. 지난해 파일은 다음 해 2월 이후 받은 것이면 다시 받지 않는다.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RAW = CACHE / "raw_cftc_tff"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
      "Accept": "application/json"}
API = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
GROUPS = {"nq": [("209742", 1.0), ("209747", 0.1)],
          "es": [("13874A", 1.0), ("13874U", 0.1)],
          "vx": [("1170E1", 1.0)],
          "rty": [("239742", 1.0)]}
CHECK = {"nq": "20974+", "es": "13874+"}
NAMES = {"209742": "NASDAQ MINI", "209747": "MICRO E-MINI NASDAQ-100 INDEX", "13874A": "E-MINI S&P 500",
         "13874U": "MICRO E-MINI S&P 500 INDEX", "1170E1": "VIX FUTURES", "239742": "RUSSELL E-MINI",
         "20974+": "NASDAQ-100 Consolidated", "13874+": "S&P 500 Consolidated"}
# 구분 → (매수 칸, 매도 칸)
CLASSES = {"dealer": ("dealer_positions_long_all", "dealer_positions_short_all"),
           "am": ("asset_mgr_positions_long", "asset_mgr_positions_short"),
           "lev": ("lev_money_positions_long", "lev_money_positions_short"),
           "other": ("other_rept_positions_long", "other_rept_positions_short"),
           "nonrept": ("nonrept_positions_long_all", "nonrept_positions_short_all")}

# 셧다운 밀린 보고서: 기준일 → 실제 공개일(CFTC 2025-12-09 9147-25 앞당긴 일정, 12/23분은 2025-12-19 공지로 12/31,
# 12/30분은 정상 일정 복귀 + 1/1 휴일로 2026-01-05). 11/10은 11/11(화) 휴일이라 월요일 기준.
SHUTDOWN = {"2025-09-30": "2025-11-19", "2025-10-07": "2025-11-21", "2025-10-14": "2025-11-25",
            "2025-10-21": "2025-12-02", "2025-10-28": "2025-12-05", "2025-11-04": "2025-12-09",
            "2025-11-10": "2025-12-10", "2025-11-18": "2025-12-12", "2025-11-25": "2025-12-15",
            "2025-12-02": "2025-12-17", "2025-12-09": "2025-12-19", "2025-12-16": "2025-12-23",
            "2025-12-23": "2025-12-31", "2025-12-30": "2026-01-05"}
# ION 사고 밀린 보고서: 기준일 → 실제 공개일(CFTC 공지 "week ending <원래 공개일> is now available" 게시일). 시각 미상.
ION = {"2023-01-31": "2023-02-24", "2023-02-07": "2023-03-03", "2023-02-14": "2023-03-08", "2023-02-21": "2023-03-10",
       "2023-02-28": "2023-03-14", "2023-03-07": "2023-03-16", "2023-03-14": "2023-03-21"}
# 공휴일 외 연방 휴무: 카터 전 대통령 국가 애도일 2025-01-09(CFTC 2025-01-07 공지로 그 주 발표 1/13),
# 2025-12-24·26 행정 휴무(CFTC 2025-12-19 공지. 셧다운 표가 먼저 적용된다)
EXTRA_CLOSED = ["2025-01-09", "2025-12-24", "2025-12-26"]
# 2025 공식 발표 일정(cftc.gov Release Schedule, 2025-08 사본): 규칙 확인용. 1/10은 애도일 공지로 1/13.
OFFICIAL = ["2024-12-02", "2024-12-06", "2024-12-13", "2024-12-20", "2024-12-30", "2025-01-06", "2025-01-13",
            "2025-01-17", "2025-01-24", "2025-01-31", "2025-02-07", "2025-02-14", "2025-02-21", "2025-02-28",
            "2025-03-07", "2025-03-14", "2025-03-21", "2025-03-28", "2025-04-04", "2025-04-11", "2025-04-18",
            "2025-04-25", "2025-05-02", "2025-05-09", "2025-05-16", "2025-05-23", "2025-05-30", "2025-06-06",
            "2025-06-13", "2025-06-23", "2025-06-27", "2025-07-07", "2025-07-11", "2025-07-18", "2025-07-25",
            "2025-08-01", "2025-08-08", "2025-08-15", "2025-08-22", "2025-08-29", "2025-09-05", "2025-09-12",
            "2025-09-19", "2025-09-26"]


def _get(url: str, timeout: int = 60) -> bytes:
    """429는 바로 멈춘다. 그 밖의 오류는 두 번 더 시도한 뒤 올린다."""
    for k in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code == 429 or k == 2:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if k == 2:
                raise
        time.sleep(30 * (k + 1))
    raise RuntimeError("도달하지 않음")


def fetch_year(code: str, year: int, gap: float) -> bool:
    """한 계약·한 해. 이미 확정된 파일이면 건너뛴다(받았으면 True)."""
    out = RAW / f"{code.replace('+', 'p')}_{year}.json"
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old["fetched"] >= f"{year + 1}-02-01":
            return False
    where = (f"cftc_contract_market_code='{code}' AND report_date_as_yyyy_mm_dd between "
             f"'{year}-01-01T00:00:00' and '{year}-12-31T23:59:59'")
    q = urllib.parse.urlencode({"$where": where, "$order": "report_date_as_yyyy_mm_dd", "$limit": 50000})
    rows = json.loads(_get(f"{API}?{q}"))
    for r in rows:  # 코드와 이름이 기대와 다르면 멈춘다
        if r["cftc_contract_market_code"] != code or r["contract_market_name"].strip() != NAMES[code]:
            raise ValueError(f"{code} 자리에 다른 계약: {r['contract_market_name']}")
    out.write_text(json.dumps({"fetched": time.strftime("%Y-%m-%d"), "code": code, "rows": rows}), encoding="utf-8")
    time.sleep(gap)
    return True


def load_code(code: str, units: bool = False) -> pd.DataFrame:
    """기준일(asof) × 원본 숫자 칸(units=True면 계약 단위 글자 칸도)."""
    rows = []
    for f in sorted(RAW.glob(f"{code.replace('+', 'p')}_*.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))["rows"]
    d = pd.DataFrame(rows)
    d["asof"] = d["report_date_as_yyyy_mm_dd"].str[:10]
    if d["asof"].duplicated().any():
        raise ValueError(f"{code} 같은 기준일이 두 번")
    cols = ["open_interest_all", "lev_money_positions_spread"] + [c for p in CLASSES.values() for c in p]
    out = d.set_index("asof")[cols].astype(float).sort_index()
    if units:
        out["units"] = d.set_index("asof")["contract_units"].reindex(out.index)
    return out


def group_frame(prefix: str, parts: list[tuple[str, float]]) -> pd.DataFrame:
    """E-mini + 마이크로×0.1 → 구분별 net, lev 매수·매도·스프레드, am 매수·매도, 미결제약정."""
    frames = [load_code(c) * m for c, m in parts]
    idx = frames[0].index
    for (c, _), f in zip(parts[1:], frames[1:]):
        miss = idx.difference(f.index)
        if len(miss):
            print(f"{prefix}: {c} 미공개 주 {list(miss)} → 그 주 {prefix} 값 비움")
    tot = sum(f.reindex(idx) for f in frames)  # 하나라도 빠진 주는 NaN
    out = pd.DataFrame(index=idx)
    for k, (lo, sh) in CLASSES.items():
        out[f"{prefix}_{k}_net"] = tot[lo] - tot[sh]
    out[f"{prefix}_lev_long"] = tot["lev_money_positions_long"]
    out[f"{prefix}_lev_short"] = tot["lev_money_positions_short"]
    out[f"{prefix}_lev_spread"] = tot["lev_money_positions_spread"]
    out[f"{prefix}_am_long"] = tot["asset_mgr_positions_long"]
    out[f"{prefix}_am_short"] = tot["asset_mgr_positions_short"]
    out[f"{prefix}_oi"] = tot["open_interest_all"]
    bad = out[[f"{prefix}_{k}_net" for k in CLASSES]].sum(axis=1, min_count=5).abs()
    if (bad > 1.0).any():  # 다섯 구분 net 합 = 0 확인(마이크로 ×0.1 반올림 여유 1계약)
        raise ValueError(f"{prefix}: net 합이 0이 아닌 주 {bad[bad > 1.0].head().to_dict()}")
    return out


def fetch_created(codes: list[str], need: set[str], gap: float) -> pd.DataFrame:
    """계약별 행 생성 시각(:created_at). 저장본이 필요한 기준일을 다 덮으면 다시 받지 않는다
    (마이크로가 빠진 주가 있어 계약까지는 따지지 않는다)."""
    out = RAW / "created_at.json"
    if out.exists():
        rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
        if need <= {r["report_date_as_yyyy_mm_dd"][:10] for r in rows}:
            return pd.DataFrame(rows)
    inn = ",".join(f"'{c}'" for c in codes)
    q = urllib.parse.urlencode({"$where": f"cftc_contract_market_code in({inn}) AND report_date_as_yyyy_mm_dd >= '{min(need)}T00:00:00'",
                                "$select": "report_date_as_yyyy_mm_dd,cftc_contract_market_code,:created_at",
                                "$order": "report_date_as_yyyy_mm_dd", "$limit": 50000})
    rows = json.loads(_get(f"{API}?{q}"))
    out.write_text(json.dumps({"fetched": time.strftime("%Y-%m-%d"), "rows": rows}), encoding="utf-8")
    time.sleep(gap)
    return pd.DataFrame(rows)


def published_at(cr: pd.DataFrame) -> pd.Series:
    """기준일 → 쓰는 계약이 모두 올라간 시각(뉴욕). 여러 기준일이 한 시각을 나눠 가지면 일괄 적재라 비운다."""
    cr = cr.assign(asof=cr["report_date_as_yyyy_mm_dd"].str[:10],
                   ts=pd.to_datetime(cr[":created_at"], utc=True).dt.tz_convert("America/New_York"))
    bulk = cr.groupby(["cftc_contract_market_code", "ts"])["asof"].transform("nunique") > 1
    cr.loc[bulk, "ts"] = pd.NaT
    g = cr.groupby("asof")["ts"]
    return g.max().where(g.count() == g.size())  # 한 계약이라도 일괄 적재 시각이면 그 주는 규칙만


def closed_days() -> set[str]:
    h = USFederalHolidayCalendar().holidays("2021-12-01", "2027-12-31")
    return {d.strftime("%Y-%m-%d") for d in h} | set(EXTRA_CLOSED)


def release_date(asof: str, closed: set[str]) -> str:
    """기준일 → CFTC 공개일(셧다운·ION 표 우선, 아니면 휴일 규칙)."""
    if asof in SHUTDOWN:
        return SHUTDOWN[asof]
    if asof in ION:
        return ION[asof]
    a = pd.Timestamp(asof)
    if a.weekday() not in (0, 1):
        raise ValueError(f"화·월요일이 아닌 기준일 {asof}")
    tue = a + pd.Timedelta(days=1 - a.weekday())
    fri = tue + pd.Timedelta(days=3)
    rel = fri
    if any((tue + pd.Timedelta(days=k)).strftime("%Y-%m-%d") in closed for k in (1, 2, 3)):
        rel = fri + pd.Timedelta(days=1)
        while rel.weekday() >= 5 or rel.strftime("%Y-%m-%d") in closed:
            rel += pd.Timedelta(days=1)
    return rel.strftime("%Y-%m-%d")


def trading_days() -> list[str]:
    """QQQ 일봉 날짜. 뉴욕 시각으로 아직 끝나지 않은 오늘 장은 뺀다."""
    import yfinance as yf
    h = yf.Ticker("QQQ").history(start="2021-12-01", auto_adjust=False)
    days = [d.strftime("%Y-%m-%d") for d in h.index]
    now = pd.Timestamp.now(tz="America/New_York")
    if days and days[-1] == now.strftime("%Y-%m-%d") and now.hour * 60 + now.minute < 16 * 60 + 15:
        days = days[:-1]
    return days


def _mv_dates(slug: str) -> list[pd.Timestamp]:
    """Mondo Visione 주소 끝 날짜(YYYY + 월 + 일, 0 채움 없음) → 가능한 날짜들(예: 2024122 = 1/22 또는 12/2)."""
    import re
    m = re.search(r"-(\d{6,8})$", slug)
    if not m:
        return []
    y, r = int(m.group(1)[:4]), m.group(1)[4:]
    out = []
    for a, b in {2: [(1, 1)], 3: [(1, 2), (2, 1)], 4: [(2, 2)]}.get(len(r), []):
        try:
            out.append(pd.Timestamp(y, int(r[:a]), int(r[a:a + b])))
        except ValueError:
            pass
    return out


def verify_release(rel: pd.Series, gap: float) -> pd.DataFrame:
    """독립 확인: CFTC가 공개 때 보내는 'Commitments of Traders Reports Update' 메일을 Mondo Visione가 그날(가끔 하루 뒤) 싣는다.
    연도별 검색 목록(cache/raw_cftc_tff/mv_<연도>.tsv)을 받아, 계산한 공개일 r에 대해 [r-3, r+3] 안의 게시일을 찾는다."""
    import html
    import re
    this_year = pd.Timestamp.now(tz="America/New_York").year
    posts = []
    for y in range(int(rel.index.min()[:4]), this_year + 1):
        f = RAW / f"mv_{y}.tsv"
        if not f.exists() or y == this_year:
            lines = []
            for page in range(1, 20):
                q = urllib.parse.urlencode({"Module": "News", "q": "Commitments of Traders", "DateFrom": f"01/01/{y} 00:00:00",
                                            "DateTo": f"31/12/{y} 23:59:59", "page": page})
                t = _get(f"https://mondovisione.com/search/moreresults/?{q}", timeout=180).decode("utf-8", "replace")
                items = re.findall(r'<h2>\s*<a href="(/media-and-resources/news/[^"]+)">(.*?)</a>', t, flags=re.S)
                lines += [f"{u.rstrip('/').split('/')[-1]}\t{html.unescape(s).strip()}" for u, s in items]
                time.sleep(gap)
                if len(items) < 10:
                    break
            f.write_text("\n".join(lines), encoding="utf-8")
        for ln in f.read_text(encoding="utf-8").splitlines():
            slug, title = ln.split("\t", 1)
            if "cftc" in slug and "commitments-of-traders" in slug or "traders-report" in slug:
                posts.append((slug, title, _mv_dates(slug)))
    have = {}
    for slug, title, ds in posts:
        for d in ds:
            have.setdefault(d, []).append(title)
    rows = []
    for asof, r in rel.items():
        r = pd.Timestamp(r)
        lag = next((k for k in range(0, 4) if r + pd.Timedelta(days=k) in have), None)
        early = next((k for k in range(1, 4) if r - pd.Timedelta(days=k) in have), None)
        rows.append({"asof": asof, "release": r.strftime("%Y-%m-%d"), "post_lag_days": lag, "post_earlier_days": early})
    return pd.DataFrame(rows).set_index("asof")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2022)
    ap.add_argument("--gap", type=float, default=1.1)
    ap.add_argument("--verify", action="store_true", help="Mondo Visione 게시일로 공개일 대조(느림, 쪽당 약 1분)")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    this_year = pd.Timestamp.now(tz="America/New_York").year
    codes = [c for parts in GROUPS.values() for c, _ in parts] + list(CHECK.values())
    n = 0
    for code in codes:
        for y in range(a.since, this_year + 1):
            n += fetch_year(code, y, a.gap)
    print(f"받은 파일 {n}개")

    closed = closed_days()
    wide = pd.concat([group_frame(p, parts) for p, parts in GROUPS.items()], axis=1).rename_axis(None)
    rel = pd.Series({d: release_date(d, closed) for d in wide.index})
    # 규칙 확인: 2025 공식 일정 기간의 정상 주는 공식 날짜와 같아야 한다
    span = rel[(rel.index >= "2024-11-26") & (rel.index <= "2025-09-23")]
    if sorted(span.tolist()) != OFFICIAL:
        raise ValueError(f"공개일 규칙이 공식 일정과 다르다: {sorted(set(span) ^ set(OFFICIAL))}")
    days = trading_days()
    pos = pd.Index(days)

    def nth(r: str, after: bool) -> str | None:  # r 당일(거래일이면) 또는 그 뒤 첫 거래일
        i = pos.searchsorted(r, side="right" if after else "left")
        return days[i] if i < len(days) else None

    # 규칙 known: 공개일 당일 장 마감 전 공개 → 그날, 시각 미상(ION)은 다음 거래일
    rule_known = {d: nth(r, d in ION) for d, r in rel.items()}
    # 시각 known: PRE 행 생성 시각이 오후 4시 이전이면 그날, 아니면 다음 거래일. 늦은 쪽을 쓴다
    pub = published_at(fetch_created([c for parts in GROUPS.values() for c, _ in parts], set(wide.index), a.gap)).reindex(wide.index)
    ts_known = {d: (nth(t.strftime("%Y-%m-%d"), t.hour * 60 + t.minute >= 16 * 60) if pd.notna(t) else None)
                for d, t in pub.items()}
    odd = pub.notna() & (pub.dt.strftime("%Y-%m-%d") != rel)
    if odd.any():
        print("PRE 생성일 ≠ 규칙 공개일:", {d: (rel[d], pub[d].strftime("%Y-%m-%d %H:%M")) for d in pub.index[odd]})
    def pick(d: str) -> str | None:  # 시각이 없으면 규칙만, 있으면 둘 다 끝난 장이 있어야 하고 늦은 쪽
        k, t = rule_known[d], ts_known[d]
        if pd.isna(pub[d]):
            return k
        return max(k, t) if (k and t) else None

    known = [pick(d) for d in wide.index]
    later = [d for d in wide.index if rule_known[d] and ts_known[d] and ts_known[d] > rule_known[d]]
    print("오후 4시 뒤 공개로 known을 미룬 주:", {d: (rule_known[d], ts_known[d], pub[d].strftime("%H:%M")) for d in later})
    wide.insert(0, "asof", wide.index)
    wide["known"] = known
    late = wide["known"].isna()
    if late.any():
        print("아직 끝난 장이 없는 공개분 제외:", list(wide.index[late]))
    wide = wide[~late].sort_values(["known", "asof"]).drop_duplicates("known", keep="last")
    wide = wide.set_index("known").sort_index()
    wide.to_parquet(CACHE / "cftc_tff.parquet")

    # 대조: CFTC 통합 보고서(E-mini 단위인 주만)와 우리 합산
    chk = {}
    for p, code in CHECK.items():
        c = load_code(code, units=True)
        c = c[c["units"].str.contains(r"\$(?:20|50)(?:\.00)?\)", regex=True)]
        ours = wide.set_index("asof")
        for k, (lo, sh) in CLASSES.items():
            cn = (c[lo] - c[sh]).reindex(ours.index)
            diff = (ours[f"{p}_{k}_net"] - cn).abs()
            chk[f"{p}_{k}"] = {"주수": int(cn.notna().sum()), "첫주": cn.dropna().index.min(),
                               "상관": round(float(ours[f"{p}_{k}_net"].corr(cn)), 4),
                               "최대차": round(float(diff.max()), 1),
                               "차/OI 중앙": round(float((diff / ours[f"{p}_oi"]).median()), 5)}
    pd.DataFrame(chk).T.to_csv(CACHE / "raw_cftc_tff" / "consolidated_check.csv", encoding="utf-8-sig")
    pd.DataFrame({"release": rel, "published_et": pub.dt.strftime("%Y-%m-%d %H:%M"), "rule_known": pd.Series(rule_known),
                  "ts_known": pd.Series(ts_known), "known": pd.Series(known, index=rel.index)}
                 ).rename_axis("asof").to_csv(CACHE / "raw_cftc_tff" / "release_dates.csv")
    if a.verify:
        v = verify_release(rel, max(a.gap, 1.5))
        v.to_csv(CACHE / "raw_cftc_tff" / "release_check.csv")
        print("게시일 대조(계산 공개일 대비 늦은 날 수):", v["post_lag_days"].value_counts(dropna=False).to_dict(),
              "더 이른 게시:", int(v["post_earlier_days"].notna().sum()))
    print(f"저장 {len(wide)}주 {wide.index.min()} ~ {wide.index.max()} (기준일 {wide['asof'].iloc[-1]})")


if __name__ == "__main__":
    main()
