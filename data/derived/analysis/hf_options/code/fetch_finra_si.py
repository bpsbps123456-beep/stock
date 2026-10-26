"""헤지펀드·옵션 연구(2026-10-14) 공매도 잔고(FINRA 보름 단위) 수집. 분석 코드는 이 파일이 만든 cache/를 읽는다.

출처: FINRA Query API 공개 데이터셋 otcMarket/consolidatedShortInterest(인증 없음, POST JSON 필터).
  기준일(settlement)은 매달 15일(휴일이면 앞 영업일)과 월말 영업일. 칸: 현재·직전 공매도 주식 수, 평균 일거래량, 커버 일수.
  FINRA 커버 일수는 1.00 아래를 1로 올리고 거래량 0이면 999.99로 적으므로 묶음 중앙값에는 직접 나눈 값(dtc_raw)을 쓴다.
  FINRA 직전 수량은 분할 전 주식 단위 그대로라서 yfinance 분할 이력으로 분모를 맞춘다(raw_finra_si/_splits.json).
공개 시점: FINRA는 기준일 뒤 7번째 영업일에 공개한다(finra.org equity-short-interest 설명, 2026-10-14 확인).
  공개 시각이 적혀 있지 않아 그날 장 마감 뒤로 보고 known = 공개일 다음 거래일로 둔다(보수적).
  공식 표(finra.org short-interest 일정, 2025-11 기준일~2026-12)가 있는 기준일은 그 공개일을 쓰고, 나머지는
  '7영업일' 규칙으로 계산한다. 이때 영업일에서 NYSE 휴장일과 함께 콜럼버스·재향군인의 날도 빼서 하루 늦게 잡힐 수 있게 했다.
API 값은 나중 정정(revisionFlag)이 반영된 최신 값이다. 당시 처음 공개된 값과 조금 다를 수 있다.

실행: py fetch_finra_si.py [--since 2023-01] [--gap 1.5]  (이어 받기: 기준일 2개가 다 찬 달 파일은 다시 받지 않는다)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RAW = CACHE / "raw_finra_si"
REPO = HERE.parents[4]
URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
      "Accept": "application/json", "Content-Type": "application/json"}
FIELDS = ["settlementDate", "symbolCode", "issueName", "marketClassCode", "currentShortPositionQuantity",
          "previousShortPositionQuantity", "averageDailyVolumeQuantity", "daysToCoverQuantity",
          "stockSplitFlag", "revisionFlag"]
ETFS = ["QQQ", "SOXX", "SMH", "TQQQ", "SQQQ", "SOXL", "SOXS"]
# FINRA 공식 공개일(기준일 → 공개일). finra.org/filing-reporting/regulatory-filing-systems/short-interest, 2026-10-14 열람.
OFFICIAL_PUB = {
    "2025-11-14": "2025-11-25", "2025-11-28": "2025-12-09", "2025-12-15": "2025-12-24", "2025-12-31": "2026-01-12",
    "2026-01-15": "2026-01-27", "2026-01-30": "2026-02-10", "2026-02-13": "2026-02-25", "2026-02-27": "2026-03-10",
    "2026-03-13": "2026-03-24", "2026-03-31": "2026-04-10", "2026-04-15": "2026-04-24", "2026-04-30": "2026-05-11",
    "2026-05-15": "2026-05-27", "2026-05-29": "2026-06-09", "2026-06-15": "2026-06-25", "2026-06-30": "2026-07-10",
    "2026-07-15": "2026-07-24", "2026-07-31": "2026-08-11", "2026-08-14": "2026-08-25", "2026-08-31": "2026-09-10",
    "2026-09-15": "2026-09-24", "2026-09-30": "2026-10-09", "2026-10-15": "2026-10-26", "2026-10-30": "2026-11-10",
    "2026-11-13": "2026-11-24", "2026-11-30": "2026-12-09", "2026-12-15": "2026-12-24", "2026-12-31": "2027-01-12",
}


def ai_symbols() -> list[str]:
    sys.path.insert(0, str(REPO))
    import watchlist
    return sorted({s for subs in watchlist.AI_UNIVERSE.values() for lst in subs.values() for s in lst if "." not in s})


def _post(body: dict) -> tuple[list[dict], int]:
    """429는 바로 멈춘다. 그 밖의 오류는 두 번 더 시도한 뒤 올린다. (행 목록, 전체 행 수)."""
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=UA, method="POST")
    for k in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                b = r.read()
                total = int(r.headers.get("record-total", "-1"))
            return (json.loads(b) if b.strip() else []), total
        except urllib.error.HTTPError as e:
            if e.code == 429 or k == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if k == 2:
                raise
        time.sleep(30 * (k + 1))
    return [], 0


def fetch_month(ym: str, symbols: list[str], gap: float) -> str:
    """한 달(기준일 2개) × 종목. 'cached' | 'fetched' | 'partial'(기준일이 아직 2개가 안 됨)."""
    out = RAW / f"{ym}.json"
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if set(symbols) <= set(old["symbols"]) and len({r["settlementDate"] for r in old["rows"]}) >= 2:
            return "cached"
    start = pd.Period(ym, "M").start_time.strftime("%Y-%m-%d")
    end = pd.Period(ym, "M").end_time.strftime("%Y-%m-%d")
    rows, offset = [], 0
    while True:
        body = {"limit": 5000, "offset": offset, "fields": FIELDS,
                "domainFilters": [{"fieldName": "symbolCode", "values": symbols}],
                "dateRangeFilters": [{"fieldName": "settlementDate", "startDate": start, "endDate": end}]}
        got, total = _post(body)
        rows += got
        time.sleep(gap)
        offset += len(got)
        if not got or offset >= total:
            break
    out.write_text(json.dumps({"symbols": symbols, "fetched": time.strftime("%Y-%m-%d %H:%M"), "rows": rows}),
                   encoding="utf-8")
    return "fetched" if len({r["settlementDate"] for r in rows}) >= 2 else "partial"


def trading_days() -> list[str]:
    """QQQ 일봉 날짜. 오늘 장이 아직 안 끝났으면(뉴욕 16:15 전) 마지막 줄을 뺀다."""
    import yfinance as yf
    h = yf.Ticker("QQQ").history(start="2022-12-01", auto_adjust=False)
    days = [d.strftime("%Y-%m-%d") for d in h.index]
    now = pd.Timestamp.now(tz="America/New_York")
    if days and days[-1] == now.strftime("%Y-%m-%d") and (now.hour, now.minute) < (16, 15):
        days = days[:-1]
    return days


def _bond_only_holidays(years: range) -> set[str]:
    """NYSE는 열지만 연방 휴일인 날(콜럼버스: 10월 둘째 월요일, 재향군인: 11/11, 주말이면 금·월로)."""
    out = set()
    for y in years:
        oct1 = pd.Timestamp(y, 10, 1)
        out.add((oct1 + pd.Timedelta(days=(7 - oct1.weekday()) % 7 + 7)).strftime("%Y-%m-%d"))
        v = pd.Timestamp(y, 11, 11)
        v = v - pd.Timedelta(days=1) if v.weekday() == 5 else v + pd.Timedelta(days=1) if v.weekday() == 6 else v
        out.add(v.strftime("%Y-%m-%d"))
    return out


def rule_pub(settle: str, tdays: list[str], extra_off: set[str]) -> str | None:
    """기준일 뒤 7번째 영업일(NYSE 거래일에서 콜럼버스·재향군인의 날을 뺀 날). 달력이 모자라면 None."""
    biz = [d for d in tdays if d > settle and d not in extra_off]
    return biz[6] if len(biz) >= 7 else None


def publication_date(settle: str, tdays: list[str], extra_off: set[str]) -> str | None:
    """공식 표에 있으면 그 날짜, 없으면 7영업일 규칙."""
    return OFFICIAL_PUB.get(settle) or rule_pub(settle, tdays, extra_off)


def fetch_splits(symbols: list[str], flagged: dict[str, str], gap: float) -> dict[str, dict[str, float]]:
    """yfinance 분할 이력(배수, 1:5 병합은 0.2). FINRA 직전 수량은 분할을 반영하지 않아 변화율 분모를 맞추는 데 쓴다.
    이미 받은 종목은 건너뛰고, 받은 날 뒤 기준일에 FINRA 분할 표시(S)가 새로 생긴 종목만 다시 받는다."""
    import yfinance as yf
    path = RAW / "_splits.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    today = time.strftime("%Y-%m-%d")
    for t in symbols:
        c = cache.get(t)
        if c and flagged.get(t, "") <= c["fetched"]:
            continue
        sp = yf.Ticker(t).splits
        if isinstance(sp, pd.DataFrame):  # yfinance 1.2는 'Stock Splits' 한 칸짜리 표를 준다
            sp = sp.iloc[:, 0] if sp.shape[1] else pd.Series(dtype=float)
        cache[t] = {"fetched": today, "splits": {d.strftime("%Y-%m-%d"): float(v) for d, v in sp.items()}}
        path.write_text(json.dumps(cache), encoding="utf-8")
        time.sleep(gap)
    return {t: v["splits"] for t, v in cache.items()}


def load_raw() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW.glob("20??-??.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))["rows"]
    return pd.DataFrame(rows).drop_duplicates(["symbolCode", "settlementDate"], keep="last")


def build(raw: pd.DataFrame, tdays: list[str], symbols: list[str], splits: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    extra = _bond_only_holidays(range(2022, 2028))
    settles = sorted(raw["settlementDate"].unique())
    prev_of = dict(zip(settles[1:], settles[:-1]))
    pub = {s: publication_date(s, tdays, extra) for s in settles}
    check = {s: (OFFICIAL_PUB[s], rule_pub(s, tdays, extra)) for s in settles if s in OFFICIAL_PUB}
    known = {}
    for s, p in pub.items():
        nxt = [d for d in tdays if d > p] if p else []
        known[s] = nxt[0] if nxt else None  # 공개일 다음 거래일 종가에 안다고 본다
    long = pd.DataFrame({
        "known": raw["settlementDate"].map(known), "asof": raw["settlementDate"],
        "published": raw["settlementDate"].map(pub), "symbol": raw["symbolCode"],
        "issue_name": raw["issueName"], "market": raw["marketClassCode"],
        "short_shares": raw["currentShortPositionQuantity"].astype(float),
        "prev_short_shares": raw["previousShortPositionQuantity"].astype(float),
        "avg_daily_volume": raw["averageDailyVolumeQuantity"].astype(float),
        "days_to_cover": raw["daysToCoverQuantity"].astype(float),
        "split_flag": raw["stockSplitFlag"].fillna(""), "revision_flag": raw["revisionFlag"].fillna("")})
    # 직전 기준일 뒤 ~ 이번 기준일(포함) 사이 분할 배수를 직전 수량에 곱해 이번 주식 단위로 맞춘다.
    # yfinance는 분사(FLEX 1.327, WDC 1.323)도 가격 조정용 '분할'로 적으므로 깔끔한 비율(1/4 단위 또는 1/정수)만 쓴다.
    def clean(v: float) -> bool:
        return abs(v * 4 - round(v * 4)) < 1e-6 if v >= 1 else abs(1 / v - round(1 / v)) < 1e-6

    def factor(sym: str, asof: str) -> float:
        p = prev_of.get(asof) or (pd.Timestamp(asof) - pd.Timedelta(days=12)).strftime("%Y-%m-%d")  # 첫 기준일
        return float(np.prod([v for d, v in splits.get(sym, {}).items() if p < d <= asof and v > 0 and clean(v)] or [1.0]))
    long["split_factor"] = [factor(s, a) for s, a in zip(long["symbol"], long["asof"])]
    long["prev_short_adj"] = long["prev_short_shares"] * long["split_factor"]
    long["chg"] = long["short_shares"] / long["prev_short_adj"].where(long["prev_short_adj"] > 0) - 1
    long["dtc_raw"] = long["short_shares"] / long["avg_daily_volume"].where(long["avg_daily_volume"] > 0)
    mism = long[(long["split_flag"] == "S") != (long["split_factor"] != 1.0)][["asof", "symbol", "split_flag", "split_factor"]]
    pending = sorted(long.loc[long["known"].isna(), "asof"].unique())
    long = long.dropna(subset=["known"]).sort_values(["known", "symbol"]).reset_index(drop=True)
    agg = aggregate(long, symbols, prev_of)
    return long, agg, {"check": check, "pending": pending, "split_mismatch": mism}


def aggregate(long: pd.DataFrame, symbols: list[str], prev_of: dict[str, str]) -> pd.DataFrame:
    """known별 AI 묶음(직전 기준일에도 있던 종목만)과 ETF 공매도 주식 수.
    si_ai_shares·si_<etf>는 FINRA 원래 주식 수라 분할 날에 계단이 생긴다. 수준은 분할을 맞춘 변화율을 이어 붙인 si_ai_idx를 쓴다."""
    ai = long[long["symbol"].isin(symbols)].copy()
    # 상장 직후 두 기준일은 뺀다(ARM 2023-09-29는 직전 400주 → 1,383만 주라 합계 변화율을 0.9%p 부풀린다)
    ai["nth"] = ai.sort_values("asof").groupby("symbol").cumcount().reindex(ai.index)
    first = ai.groupby("symbol")["asof"].transform("min")
    ai.loc[first == ai["asof"].min(), "nth"] += 2  # 받은 첫 기준일부터 있던 종목은 이미 상장돼 있던 종목
    out = []
    for asof, g in ai.groupby("asof"):
        r = {"known": g["known"].iloc[0], "asof": asof}
        p = prev_of.get(asof)
        both = g[g["symbol"].isin(ai.loc[ai["asof"] == p, "symbol"])] if p else g.iloc[0:0]
        both = both[(both["prev_short_adj"] > 0) & (both["nth"] >= 2)]
        r["si_ai_n"] = float(len(both))
        r["si_ai_shares"] = both["short_shares"].sum() if len(both) else np.nan
        r["si_ai_chg"] = both["short_shares"].sum() / both["prev_short_adj"].sum() - 1 if len(both) else np.nan
        r["si_ai_chg_median"] = both["chg"].median() if len(both) else np.nan
        r["si_ai_up_share"] = (both["chg"] > 0).mean() if len(both) else np.nan
        r["si_ai_dtc_median"] = g["dtc_raw"].median()
        r["si_ai_dtc_agg"] = g["short_shares"].sum() / g["avg_daily_volume"].sum()
        out.append(r)
    agg = pd.DataFrame(out).set_index("known")
    agg["si_ai_idx"] = 100 * (1 + agg["si_ai_chg"].fillna(0)).cumprod()
    for t in ETFS:
        e = long[long["symbol"] == t].set_index("known")
        agg[f"si_{t.lower()}"] = e["short_shares"]
        agg[f"si_{t.lower()}_chg"] = e["chg"]
        agg[f"si_{t.lower()}_dtc"] = e["dtc_raw"]
    agg = agg.sort_index()
    agg = agg[~agg.index.duplicated(keep="last")]
    agg.index.name = "known"
    cols = ["asof"] + [c for c in agg.columns if c != "asof"]
    return agg[cols].astype({c: float for c in cols if c != "asof"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2023-01", help="첫 달(YYYY-MM). 2023은 2024 백분위 창을 채우는 용도")
    ap.add_argument("--gap", type=float, default=1.5)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    symbols = ai_symbols()
    want = symbols + [t for t in ETFS if t not in symbols]
    tdays = trading_days()
    months = [str(p) for p in pd.period_range(a.since, pd.Timestamp(tdays[-1]).to_period("M"), freq="M")]
    for ym in months:
        st = fetch_month(ym, want, a.gap)
        print(ym, st, flush=True)
    raw = load_raw()
    flagged = raw[raw["stockSplitFlag"].fillna("") == "S"].groupby("symbolCode")["settlementDate"].max().to_dict()
    splits = fetch_splits(want, flagged, a.gap)
    long, agg, info = build(raw, tdays, symbols, splits)
    long.to_parquet(CACHE / "finra_si_by_symbol.parquet", index=False)
    agg.to_parquet(CACHE / "finra_si.parquet")
    bad = {s: v for s, v in info["check"].items() if v[1] is not None and v[0] != v[1]}
    print("공식 공개일 대비 규칙 계산 불일치:", bad or "없음")
    print("아직 공개 전(known 없음) 기준일:", info["pending"] or "없음")
    print("FINRA 분할 표시와 yfinance 분할 배수 불일치:", info["split_mismatch"].to_dict("records") or "없음")
    print(f"종목 {long['symbol'].nunique()} · 행 {len(long)} · 기준일 {long['asof'].nunique()} · "
          f"known {agg.index[0]} ~ {agg.index[-1]}")


if __name__ == "__main__":
    main()
