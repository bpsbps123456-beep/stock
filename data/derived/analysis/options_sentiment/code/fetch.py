"""옵션 심리 연구(2026-10-14) 자료 수집. 분석 코드는 모두 이 파일이 만든 cache/를 읽는다.

출처(전부 무료 공개 자료, 실행 하나가 연결 하나로 간격을 두고 받는다):
  - CBOE 일별 시장 통계 JSON(cdn.cboe.com/.../daily/<날짜>_daily_options): 풋/콜 비율(전체·지수·ETP·주식·VIX·SPX)과
    부문별 콜·풋 거래량·미결제약정. 거래량은 CBOE 거래소 몫, 미결제약정은 상품 전체 값이다.
  - OCC 거래량 조회(marketdata.theocc.com/volume-query, 전 종목 하루 1건): 기초자산 × 계좌(C 고객·F 증권사·M 시장조성자) × 콜/풋.
    20개 거래소 합이다. 한 거래의 양쪽을 따로 센다(고객이 시장조성자에게 산 1계약 = C 1 + M 1).
    매수·매도와 신규·청산은 나누지 않는다. C(고객)는 개인과 기관 고객을 함께 담는다.
  - Yahoo: VIX 계열·SKEW·VXN·VVIX·DSPX 지수, 가격(QQQ SPY SOXX SMH), 13주 국채(^IRX).
값은 그날 미국 장 마감 뒤 나온다. 분석은 다음 거래일 종가부터 쓴다.

실행: py fetch.py [--since 2024-01-02] [--until YYYY-MM-DD] [--gap 1.0]  (이어 받기: 이미 받은 날은 건너뛴다)
2024는 2025 백분위 창(250거래일)을 채우는 용도다. 2026-10-14에 2025~와 2024를 두 연결로 나눠 받았다.
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
      "Accept": "application/json, text/csv, */*"}
CBOE_URL = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{d}_daily_options"
OCC_URL = ("https://marketdata.theocc.com/volume-query?reportDate={d}&format=csv&volumeQueryType=O&symbolType=ALL&symbol="
           "&reportType=D&accountType=ALL&productKind=ALL&porc=BOTH")
CBOE_SEG = {"SUM OF ALL PRODUCTS": "all", "INDEX OPTIONS": "index", "EXCHANGE TRADED PRODUCTS": "etp",
            "EQUITY OPTIONS": "equity", "CBOE VOLATILITY INDEX (VIX)": "vix", "SPX + SPXW": "spx"}
YAHOO = ["^VIX", "^VIX9D", "^VIX3M", "^VIX6M", "^VIX1D", "^VVIX", "^SKEW", "^VXN", "^DSPX", "^IRX",
         "QQQ", "SPY", "SOXX", "SMH"]


def _get(url: str, timeout: int = 60) -> bytes | None:
    """404·403(아직 안 나온 날, 휴장)은 None. 429는 바로 멈춘다. 그 밖의 오류는 두 번 더 시도한 뒤 올린다."""
    for k in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if e.code == 429 or k == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if k == 2:
                raise
        time.sleep(30 * (k + 1))
    return None


def trading_days(since: str) -> list[str]:
    import yfinance as yf
    h = yf.Ticker("QQQ").history(start=since, auto_adjust=False)
    return [d.strftime("%Y-%m-%d") for d in h.index]


def fetch_cboe(day: str) -> bool:
    out = CACHE / "cboe" / f"{day}.json"
    if out.exists():
        return True
    b = _get(CBOE_URL.format(d=day), timeout=30)
    if b is None:
        return False
    json.loads(b)  # 깨진 응답이면 여기서 멈춘다
    out.write_bytes(b)
    return True


def fetch_occ(day: str) -> bool:
    """전 종목 CSV(약 4MB)를 받아 기초자산별로 합친 6칸만 남긴다(C_C = 고객 콜, M_P = 시장조성자 풋 …)."""
    out = CACHE / "occ" / f"{day}.parquet"
    if out.exists():
        return True
    b = _get(OCC_URL.format(d=day.replace("-", "")), timeout=120)
    if not b or len(b) < 1000:
        return False
    df = pd.read_csv(io.BytesIO(b), index_col=False, dtype={"underlying": str, "actype": str, "porc": str})
    got = pd.to_datetime(df["actdate"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d").unique()
    if list(got) != [day]:
        raise ValueError(f"OCC {day} 요청에 다른 날짜가 왔다: {got[:3]}")
    df["k"] = df["actype"] + "_" + df["porc"]
    w = df.pivot_table(index="underlying", columns="k", values="quantity", aggfunc="sum", fill_value=0)
    w = w.reindex(columns=["C_C", "C_P", "F_C", "F_P", "M_C", "M_P"], fill_value=0).astype("int64")
    w.reset_index().to_parquet(out, index=False)
    return True


def fetch_yahoo(since: str) -> None:
    import yfinance as yf
    rows = {}
    for t in YAHOO:  # 가격은 분할 반영 종가(배당 미반영) — 데스크 연구와 같은 기준
        c = yf.Ticker(t).history(start=since, auto_adjust=False)["Close"]
        c.index = c.index.strftime("%Y-%m-%d")  # 지수(시카고)·ETF(뉴욕) 시간대가 달라 날짜 문자열로 먼저 맞춘다
        rows[t] = c[~c.index.duplicated(keep="last")]
    pd.DataFrame(rows).sort_index().to_parquet(CACHE / "yahoo.parquet")


def load_cboe() -> pd.DataFrame:
    """날짜 × (부문_칸). 칸: cv·pv(콜·풋 거래량), coi·poi(콜·풋 미결제약정), pcr(공식 풋/콜 비율)."""
    recs = []
    for f in sorted((CACHE / "cboe").glob("*.json")):
        j = json.loads(f.read_text())
        r = {"date": f.stem}
        for name, seg in CBOE_SEG.items():
            for it in j.get(name, []):
                if it["name"] == "VOLUME":
                    r[f"{seg}_cv"], r[f"{seg}_pv"] = it["call"], it["put"]
                elif it["name"] == "OPEN INTEREST":
                    r[f"{seg}_coi"], r[f"{seg}_poi"] = it["call"], it["put"]
        for it in j.get("ratios", []):
            key = {"TOTAL PUT/CALL RATIO": "all", "INDEX PUT/CALL RATIO": "index",
                   "EXCHANGE TRADED PRODUCTS PUT/CALL RATIO": "etp", "EQUITY PUT/CALL RATIO": "equity",
                   "CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO": "vix", "SPX + SPXW PUT/CALL RATIO": "spx"}.get(it["name"])
            if key:
                r[f"{key}_pcr"] = float(it["value"])
        recs.append(r)
    return pd.DataFrame(recs).set_index("date").sort_index()


def load_occ(symbols: list[str] | None = None) -> pd.DataFrame:
    """긴 표: date, underlying, C_C … M_P. symbols를 주면 그 기초자산만."""
    parts = []
    for f in sorted((CACHE / "occ").glob("*.parquet")):
        d = pd.read_parquet(f)
        if symbols is not None:
            d = d[d["underlying"].isin(symbols)]
        d.insert(0, "date", f.stem)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2024-01-02")
    ap.add_argument("--until", default=None, help="이 날짜까지만(앞 구간을 따로 받을 때)")
    ap.add_argument("--gap", type=float, default=1.0)
    a = ap.parse_args()
    for sub in ("cboe", "occ"):
        (CACHE / sub).mkdir(parents=True, exist_ok=True)
    if a.until is None:
        fetch_yahoo(a.since)
    days = [d for d in trading_days(a.since) if a.until is None or d <= a.until]
    log = open(CACHE / "fetch.log", "a", encoding="utf-8")
    miss = {"cboe": [], "occ": []}
    for i, d in enumerate(days):
        for name, fn in (("cboe", fetch_cboe), ("occ", fetch_occ)):
            fresh = not (CACHE / name / f"{d}.{'json' if name == 'cboe' else 'parquet'}").exists()
            if not fn(d):
                miss[name].append(d)
            if fresh:
                time.sleep(a.gap)
        if i % 20 == 0 or i == len(days) - 1:
            print(f"{time.strftime('%H:%M:%S')} {i + 1}/{len(days)} {d} 빠짐 cboe {len(miss['cboe'])} occ {len(miss['occ'])}",
                  file=log, flush=True)
    print("빠진 날", json.dumps(miss), file=log, flush=True)


if __name__ == "__main__":
    main()
