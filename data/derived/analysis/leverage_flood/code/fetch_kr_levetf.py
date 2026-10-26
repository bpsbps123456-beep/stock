"""한국·홍콩·미국 레버리지 ETF 수급 지표 수집기 (leverage_flood 연구용).

받는 것
  1. 네이버 ETF 전체 목록(finance.naver.com/api/sise/etfItemList.nhn)에서 삼성전자·SK하이닉스 단일종목
     레버리지·인버스 16종 + KODEX 레버리지(122630)·코스닥150레버리지(233740)·200선물인버스2X(252670) 코드.
  2. 네이버 모바일 API(m.stock.naver.com/api/stock/<코드>/integration)에서 "현재" 순자산(totalNav)·시가총액.
     (일별 순자산 이력은 네이버에 없다. KRX 정보데이터시스템은 2026-10 현재 로그인 필수라 스크립트로 못 받는다.)
  3. 네이버 모바일 API price(일별 OHLCV)·trend(투자자별 순매수 수량)로 상장일(2026-05-27)부터 일별 이력.
  4. yfinance로 기초자산(005930·000660) 일별 종가·거래량, 홍콩 CSOP 7709·7747, 미국 KORU·EWY, 코스피 지수.
  5. 파생 지표: 레버리지 ETF 거래대금(종가×거래량 근사) ÷ 기초자산 거래대금, 개인 순매수 금액(순매수 수량×종가).

출력(results/):
  kr_levetf_list.csv            16+3종 코드·이름·현재 순자산·시가총액(네이버 당일 값)
  kr_levetf_prices.csv          종목×일자 OHLCV + 투자자별 순매수 수량 (long)
  kr_levetf_daily_ratio.csv     일별 집계: 레버리지 거래대금, 기초자산 거래대금, 비율, 개인 순매수 금액
  kr_levetf_hk_us.csv           7709.HK·7747.HK·KORU·EWY 일별 종가·거래량 + 현재 totalAssets
  kr_levetf_fetch_meta.json     수집 시각·출처·단위

실행: py fetch_kr_levetf.py            (기본 시작일 2026-05-01)
      py fetch_kr_levetf.py 20260101
"""
import json
import os
import re
import sys
import time
from datetime import datetime

import pandas as pd
import requests

RESULTS = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis\leverage_flood\results"
MOBILE_H = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1",
    "Referer": "https://m.stock.naver.com/",
    "Accept": "application/json",
}
INDEX_CODES = {"122630": "KODEX 레버리지", "233740": "KODEX 코스닥150레버리지", "252670": "KODEX 200선물인버스2X"}
UNDERLYING = {"005930": "삼성전자", "000660": "SK하이닉스"}
SINGLE_RE = re.compile(r"삼성전자|SK하이닉스")
LEV_RE = re.compile(r"레버리지|인버스2X|2X")


def to_int(s):
    if s is None:
        return None
    s = str(s).replace(",", "").replace("+", "").strip()
    if s in ("", "-", "N/A"):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def korean_won(s):
    """'1조 4,286억' -> 원 단위 정수."""
    if not s:
        return None
    s = str(s).replace(",", "")
    total = 0
    m = re.search(r"(\d+)조", s)
    if m:
        total += int(m.group(1)) * 10**12
    m = re.search(r"(\d+)억", s)
    if m:
        total += int(m.group(1)) * 10**8
    return total or None


SESSION = requests.Session()


def get_json(url, tries=5):
    """네이버는 연속 호출이 많으면 JSON 대신 HTML을 돌려준다. 지수 백오프로 재시도."""
    last = None
    for i in range(tries):
        try:
            r = SESSION.get(url, headers=MOBILE_H, timeout=60)
            if r.status_code == 200 and r.text[:1] in "[{":
                return r.json()
            last = f"status={r.status_code} body={r.text[:80]!r}"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)[:120]
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"naver non-json after {tries} tries: {url} {last}")


def naver_etf_list():
    url = "https://finance.naver.com/api/sise/etfItemList.nhn?etfType=0&targetColumn=market_sum&sortOrder=desc"
    js = get_json(url)
    return js["result"]["etfItemList"]


def naver_integration(code):
    js = get_json(f"https://m.stock.naver.com/api/stock/{code}/integration")
    key = js.get("etfKeyIndicator") or {}
    infos = {t.get("code"): t.get("value") for t in js.get("totalInfos", [])}
    return {
        "nav": to_int(key.get("nav") or infos.get("nav")),
        "total_nav_won": korean_won(key.get("totalNav")),
        "market_value_won": korean_won(key.get("marketValue")),
        "issuer": key.get("issuerName"),
        "fee_pct": key.get("totalFee"),
        "base_index": infos.get("etfBaseIdx"),
        "acc_trading_value_text": infos.get("accumulatedTradingValue"),
    }


def naver_price_history(code, start):
    rows = []
    page = 1
    while True:
        url = f"https://m.stock.naver.com/api/stock/{code}/price?pageSize=60&page={page}"
        js = get_json(url)
        if not isinstance(js, list) or not js:
            break
        for r in js:
            d = r["localTradedAt"].replace("-", "")
            rows.append({"code": code, "date": d, "open": to_int(r.get("openPrice")), "high": to_int(r.get("highPrice")),
                         "low": to_int(r.get("lowPrice")), "close": to_int(r.get("closePrice")),
                         "volume": to_int(r.get("accumulatedTradingVolume"))})
        if rows[-1]["date"] < start or len(js) < 60:
            break
        page += 1
        time.sleep(0.2)
    return [r for r in rows if r["date"] >= start]


def naver_trend_history(code, start):
    """trend API는 page를 무시하고 bizdate(그 날짜 '이전' 60건)를 커서로 쓴다. pageSize 최대 60."""
    rows = []
    cursor = None
    seen = set()
    for _ in range(40):
        url = f"https://m.stock.naver.com/api/stock/{code}/trend?pageSize=60" + (f"&bizdate={cursor}" if cursor else "")
        js = get_json(url)
        if not isinstance(js, list) or not js:
            break
        new = 0
        for r in js:
            d = r["bizdate"]
            if d in seen:
                continue
            seen.add(d)
            new += 1
            rows.append({"code": code, "date": d, "indiv_net_qty": to_int(r.get("individualPureBuyQuant")),
                         "foreign_net_qty": to_int(r.get("foreignerPureBuyQuant")), "inst_net_qty": to_int(r.get("organPureBuyQuant")),
                         "foreign_hold_pct": (r.get("foreignerHoldRatio") or "").replace("%", "")})
        last = js[-1]["bizdate"]
        if new == 0 or last < start or len(js) < 60:
            break
        cursor = last
        time.sleep(0.2)
    return [r for r in rows if r["date"] >= start]


def yf_history(ticker, start):
    import yfinance as yf
    tk = yf.Ticker(ticker)
    h = tk.history(start=f"{start[:4]}-{start[4:6]}-{start[6:]}", auto_adjust=False)
    info = {}
    try:
        info = tk.get_info() or {}
    except Exception:  # noqa: BLE001
        info = {}
    out = []
    for idx, r in h.iterrows():
        out.append({"ticker": ticker, "date": idx.strftime("%Y%m%d"), "close": float(r["Close"]), "volume": float(r["Volume"]),
                    "trading_value": float(r["Close"]) * float(r["Volume"])})
    return out, {"ticker": ticker, "total_assets_now": info.get("totalAssets"), "nav_now": info.get("navPrice"),
                 "shares_out_now": info.get("sharesOutstanding"), "name": info.get("longName") or info.get("shortName")}


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "20260501"
    os.makedirs(RESULTS, exist_ok=True)
    meta = {"fetched_at": datetime.now().isoformat(timespec="seconds"), "start": start, "sources": {}, "notes": []}

    # 1. 목록
    items = naver_etf_list()
    single = [it for it in items if SINGLE_RE.search(it["itemname"]) and LEV_RE.search(it["itemname"])]
    picked = single + [it for it in items if it["itemcode"] in INDEX_CODES]
    rows = []
    for it in picked:
        name = it["itemname"]
        kind = "single_inverse" if "인버스" in name else ("single_lev" if it in single else "index_lev")
        under = "005930" if "삼성전자" in name else ("000660" if "하이닉스" in name else "")
        try:
            integ = naver_integration(it["itemcode"])
        except Exception as exc:  # noqa: BLE001
            integ = {"error": str(exc)[:80]}
        rows.append({"code": it["itemcode"], "name": name, "kind": kind, "underlying": under,
                     "naver_market_sum_eok": it.get("marketSum"), "naver_nav": it.get("nav"),
                     "naver_today_volume": it.get("quant"), "naver_today_value_mil": it.get("amonut"), **integ})
        time.sleep(0.2)
    lst = pd.DataFrame(rows)
    lst.to_csv(os.path.join(RESULTS, "kr_levetf_list.csv"), index=False, encoding="utf-8-sig")
    print("list:", len(single), "single-stock +", len(rows) - len(single), "index lev; total_nav(single) won =",
          int(lst.loc[lst.kind != "index_lev", "total_nav_won"].fillna(0).sum()))
    meta["sources"]["list"] = "https://finance.naver.com/api/sise/etfItemList.nhn (시가총액 억원, 당일)"
    meta["sources"]["integration"] = "https://m.stock.naver.com/api/stock/<code>/integration etfKeyIndicator.totalNav (현재값만, 이력 없음)"

    # 2. 일별 이력
    price_rows, trend_rows = [], []
    for code in list(lst.code) + list(UNDERLYING):
        try:
            price_rows += naver_price_history(code, start)
            trend_rows += naver_trend_history(code, start)
            print("  hist", code, len(price_rows), len(trend_rows))
        except Exception as exc:  # noqa: BLE001
            print("  hist error", code, exc)
            meta["notes"].append(f"{code} history error: {exc}")
    px = pd.DataFrame(price_rows, columns=["code", "date", "open", "high", "low", "close", "volume"])
    tr = pd.DataFrame(trend_rows, columns=["code", "date", "indiv_net_qty", "foreign_net_qty", "inst_net_qty", "foreign_hold_pct"])
    df = px.merge(tr, on=["code", "date"], how="left")
    df["approx_trading_value_won"] = df["close"] * df["volume"]
    df["indiv_net_value_won"] = df["indiv_net_qty"] * df["close"]
    df = df.sort_values(["code", "date"])
    df.to_csv(os.path.join(RESULTS, "kr_levetf_prices.csv"), index=False, encoding="utf-8-sig")
    meta["sources"]["price"] = "https://m.stock.naver.com/api/stock/<code>/price?pageSize=60&page=N (일별 OHLCV, 당일 장중 포함; pageSize 최대 60)"
    meta["sources"]["trend"] = "https://m.stock.naver.com/api/stock/<code>/trend?pageSize=60&bizdate=YYYYMMDD (투자자별 순매수 수량, 전일까지; page 무시, bizdate 이전 60건 커서)"
    meta["notes"].append("거래대금은 종가×거래량 근사(네이버 price API에 거래대금 없음). 네이버 integration의 당일 거래대금 텍스트로 대조 가능.")

    # 3. 일별 집계
    kinds = lst.set_index("code")["kind"]
    unders = lst.set_index("code")["underlying"]
    df["kind"] = df["code"].map(kinds).fillna("underlying")
    df["underlying"] = df["code"].map(unders)
    g = df[df.kind == "single_lev"].groupby("date")
    daily = pd.DataFrame({
        "single_lev_trading_value_won": g["approx_trading_value_won"].sum(),
        "single_lev_indiv_net_value_won": g["indiv_net_value_won"].sum(),
        "single_lev_foreign_net_qty": g["foreign_net_qty"].sum(),
        "n_single_lev": g["code"].count(),
    })
    gi = df[df.kind == "single_inverse"].groupby("date")
    daily["single_inv_trading_value_won"] = gi["approx_trading_value_won"].sum()
    daily["single_inv_indiv_net_value_won"] = gi["indiv_net_value_won"].sum()
    for u, nm in UNDERLYING.items():
        sub = df[df.code == u].set_index("date")
        daily[f"{nm}_close"] = sub["close"]
        daily[f"{nm}_trading_value_won"] = sub["approx_trading_value_won"]
        daily[f"{nm}_indiv_net_value_won"] = sub["indiv_net_value_won"]
        lev_u = df[(df.kind == "single_lev") & (df.underlying == u)].groupby("date")["approx_trading_value_won"].sum()
        daily[f"{nm}_lev_etf_trading_value_won"] = lev_u
        daily[f"{nm}_lev_etf_to_underlying_value_ratio"] = lev_u / sub["approx_trading_value_won"]
    daily["underlying_trading_value_won"] = daily["삼성전자_trading_value_won"] + daily["SK하이닉스_trading_value_won"]
    daily["single_lev_to_underlying_value_ratio"] = daily["single_lev_trading_value_won"] / daily["underlying_trading_value_won"]
    for code, nm in INDEX_CODES.items():
        sub = df[df.code == code].set_index("date")
        daily[f"{code}_close"] = sub["close"]
        daily[f"{code}_trading_value_won"] = sub["approx_trading_value_won"]
        daily[f"{code}_indiv_net_value_won"] = sub["indiv_net_value_won"]
    daily = daily.sort_index()
    daily.to_csv(os.path.join(RESULTS, "kr_levetf_daily_ratio.csv"), encoding="utf-8-sig")
    print("daily ratio rows", len(daily))

    # 4. 홍콩·미국
    hk_rows, hk_meta = [], []
    for t in ["7709.HK", "7747.HK", "KORU", "EWY", "^KS11", "000660.KS", "005930.KS"]:
        try:
            r, m = yf_history(t, start)
            hk_rows += r
            hk_meta.append(m)
            print("  yf", t, len(r), m)
        except Exception as exc:  # noqa: BLE001
            print("  yf error", t, exc)
            meta["notes"].append(f"yfinance {t} error: {exc}")
    pd.DataFrame(hk_rows).to_csv(os.path.join(RESULTS, "kr_levetf_hk_us.csv"), index=False, encoding="utf-8-sig")
    meta["yfinance_now"] = hk_meta
    meta["sources"]["hk_us"] = "yfinance history + get_info().totalAssets (현재값, 이력 없음; CSOP 사이트·HKEX 위젯은 403)"
    meta["unit"] = "원, 수량 주, 비율 소수"
    with open(os.path.join(RESULTS, "kr_levetf_fetch_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print("done")


if __name__ == "__main__":
    main()
