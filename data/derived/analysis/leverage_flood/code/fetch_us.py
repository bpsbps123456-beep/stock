"""미국 레버리지·포지션·심리 지표 수집기 (leverage_flood / 미국 갈래).

실행: py fetch_us.py [--skip cnms,cboe_pc]   (느린 소스는 --skip으로 건너뜀)
출력: results/us_*.csv + results/us_fetch_meta.json

소스(2026-10-14 확인):
  yfinance            가격·거래량 (최근 종가 NaN이면 메우지 않고 meta에 표시)
  ProShares           https://accounts.profunds.com/etfdata/ByFund/<T>-historical_nav.csv  (NAV·발행좌수·순자산 일별)
  GraniteShares       NVDL 상품 페이지 JS의 ProductNavHistory 엔드포인트 (NAV·AUM 일별, 좌수 = AUM/NAV)
  Direxion            https://www.direxion.com/holdings/<T>.csv  (당일 발행좌수만; 이력 페이지는 Cloudflare 차단)
  CBOE 지수           https://cdn.cboe.com/api/global/us_indices/daily_prices/<IDX>_History.csv
  CBOE 풋/콜          https://cdn.cboe.com/data/us/options/market_statistics/daily/<YYYY-MM-DD>_daily_options  (JSON, 거래일별)
  CFTC TFF            https://publicreporting.cftc.gov/resource/gpe5-46if.json (선물만) / yw9f-hn96.json (선물+옵션)
  FINRA 마진부채      https://www.thetrading.tools/margin-debt 페이지 내장 JSON(FINRA 월간 표 미러) + convextrade 표로 교차 확인
  NAAIM               https://index.naaim.org/embeddable/table  (2026-08-01부터 공개분은 3개월 지연)
  AAII                https://www.aaii.com/files/surveys/sentiment.xls
  FINRA 공매도잔고    https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest (POST, 월 2회)
  FINRA 일별 숏볼륨   https://cdn.finra.org/equity/regsho/daily/CNMSshvol<YYYYMMDD>.txt
"""
from __future__ import annotations

import io
import json
import re
import sys
import html as html_mod
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import pandas as pd
import requests

RESULTS = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis\leverage_flood\results"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
TODAY = date.today()
START = "2023-01-01"

PRICE_TICKERS = ["SOXL", "SOXS", "NVDL", "MUU", "TQQQ", "SQQQ", "SMH", "SOXX", "SPY", "QQQ", "GVIP",
                 "NVDA", "MU", "AMD", "^SOX", "^VIX", "^VXN", "^SKEW", "^VIX3M", "^GSPC", "^NDX"]
PROSHARES = ["TQQQ", "SQQQ", "QLD", "UPRO", "SPXU", "SSO", "SDS"]
DIREXION = ["SOXL", "SOXS", "MUU", "NVDU", "NVDD", "TSLL", "SPXL", "SPXS", "TECL", "TECS"]
CBOE_IDX = ["VIX", "VXN", "SKEW", "VIX3M", "VVIX"]
CFTC_MARKETS = ["NASDAQ-100 Consolidated", "S&P 500 Consolidated", "E-MINI S&P 500", "NASDAQ MINI", "VIX FUTURES"]
SI_SYMBOLS = ["NVDA", "AMD", "MU", "AVGO", "SOXL", "SOXS", "SMH", "SOXX", "TQQQ", "SQQQ", "SPY", "QQQ", "NVDL", "MUU", "GVIP"]
CNMS_SYMBOLS = set(SI_SYMBOLS)
CNMS_START = date(2025, 1, 2)
CBOE_PC_START = date(2024, 6, 3)

META: dict = {"fetched_at": datetime.now().isoformat(timespec="seconds"), "files": {}, "errors": {}, "notes": {}}


def save(df: pd.DataFrame, name: str, note: str = "") -> str:
    path = f"{RESULTS}\\{name}"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    META["files"][name] = {"rows": int(len(df)), "cols": list(map(str, df.columns))[:30], "note": note}
    print(f"  saved {name} rows={len(df)}")
    return path


def step(name):
    def deco(fn):
        def wrapper(*a, **k):
            print(f"== {name}")
            try:
                return fn(*a, **k)
            except Exception as exc:  # noqa: BLE001
                META["errors"][name] = f"{type(exc).__name__}: {exc}"
                print(f"  ERROR {name}: {exc}")
                return None
        return wrapper
    return deco


def bdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


# ---------------------------------------------------------------- yfinance
@step("yfinance prices")
def fetch_prices():
    import yfinance as yf

    raw = yf.download(PRICE_TICKERS, start=START, auto_adjust=False, progress=False, threads=True, group_by="column")
    if raw is None or raw.empty:
        raise RuntimeError("yfinance empty")
    close = raw["Close"].copy()
    vol = raw["Volume"].copy()
    close.index.name = "Date"
    vol.index.name = "Date"
    last = close.index.max()
    nan_last = [t for t in close.columns if pd.isna(close.loc[last, t])]
    META["notes"]["yf_last_date"] = str(last.date())
    META["notes"]["yf_nan_on_last_date"] = nan_last
    # 마지막 날 NaN은 앞 값으로 메우지 않고 그대로 둔다
    save(close.reset_index(), "us_prices_close.csv", f"yfinance Close {START}~; 마지막 날 NaN 티커={nan_last}")
    save(vol.reset_index(), "us_prices_volume.csv", "yfinance Volume")
    return close


# ---------------------------------------------------------------- ProShares
@step("ProShares NAV history")
def fetch_proshares():
    frames = []
    for t in PROSHARES:
        r = requests.get(f"https://accounts.profunds.com/etfdata/ByFund/{t}-historical_nav.csv", headers=H, timeout=60)
        if r.status_code != 200:
            META["errors"][f"proshares_{t}"] = f"HTTP {r.status_code}"
            continue
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"Date": "date", "Ticker": "ticker", "NAV": "nav", "Prior NAV": "prior_nav",
                                "NAV Change (%)": "nav_chg_pct", "Shares Outstanding (000)": "shares_out_k",
                                "Assets Under Management": "aum"})
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
        df = df[["date", "ticker", "nav", "prior_nav", "nav_chg_pct", "shares_out_k", "aum"]]
        frames.append(df)
    out = pd.concat(frames).sort_values(["ticker", "date"])
    out = out[out["date"] >= "2022-01-01"]
    out["shares_out"] = out["shares_out_k"] * 1000
    out["d_shares_1d"] = out.groupby("ticker")["shares_out"].diff()
    out["flow_usd_1d"] = out["d_shares_1d"] * out["nav"]  # 창출·환매 좌수 × NAV = 일별 자금흐름 근사
    return save(out, "us_proshares_nav.csv", "ProShares 공식 일별 NAV·발행좌수·AUM; flow_usd_1d = Δ좌수×NAV")


# ---------------------------------------------------------------- GraniteShares NVDL
@step("GraniteShares NVDL NAV history")
def fetch_nvdl():
    page = requests.get("https://graniteshares.com/institutional/us/en-us/etfs/nvdl/", headers=H, timeout=60).text
    m = re.search(r'var NAV_HISTORY_URL\s*=\s*"([^"]+)"\s*\+\s*PRODUCT_ID\s*\+\s*"([^"]+)"', page)
    pid = re.search(r'var PRODUCT_ID\s*=\s*(\d+)', page).group(1)
    url = m.group(1) + pid + m.group(2)
    j = requests.get(url, headers=H, timeout=60).json()
    df = pd.DataFrame(j)[["PriceDate", "Price", "AUM", "ClosingPrice", "BidAsk", "PremiumDiscount"]]
    df = df.rename(columns={"PriceDate": "date", "Price": "nav", "AUM": "aum", "ClosingPrice": "close",
                            "BidAsk": "bid_ask", "PremiumDiscount": "prem_disc"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last")  # 원본에 같은 날 중복 행이 있음(2026-10-12 확인)
    df["shares_implied"] = (df["aum"] / df["nav"]).where(df["nav"] > 0)
    df["d_shares_1d"] = df["shares_implied"].diff()
    df["flow_usd_1d"] = df["d_shares_1d"] * df["nav"]
    df["prem_pct"] = (df["close"] / df["nav"] - 1) * 100
    return save(df, "us_nvdl_nav.csv", "GraniteShares 공식 일별 NAV·AUM; shares_implied=AUM/NAV")


# ---------------------------------------------------------------- Direxion snapshot
@step("Direxion holdings snapshot (today only)")
def fetch_direxion_snapshot():
    rows = []
    for t in DIREXION:
        r = requests.get(f"https://www.direxion.com/holdings/{t}.csv", headers=H, timeout=60)
        if r.status_code != 200 or not r.text.startswith("Direxion"):
            META["errors"][f"direxion_{t}"] = f"HTTP {r.status_code}"
            continue
        lines = r.text.splitlines()
        name = lines[0].strip()
        so = re.search(r"Shares Outstanding:(\d+)", r.text)
        body = "\n".join(lines[5:])
        h = pd.read_csv(io.StringIO(body))
        h.columns = [c.strip() for c in h.columns]
        trade_date = pd.to_datetime(h["TradeDate"].iloc[0]).date()
        mv = pd.to_numeric(h["MarketValue"], errors="coerce").sum()
        swaps = h[h["SecurityDescription"].str.contains("SWAP", case=False, na=False)]
        swap_mv = pd.to_numeric(swaps["MarketValue"], errors="coerce").sum()
        rows.append({"date": trade_date, "ticker": t, "name": name, "shares_out": int(so.group(1)) if so else None,
                     "holdings_mv_sum": mv, "swap_mv_sum": swap_mv, "n_holdings": len(h)})
    return save(pd.DataFrame(rows), "us_direxion_snapshot.csv",
                "Direxion 당일 보유 CSV 머리글의 발행좌수 스냅샷. 이력 페이지는 Cloudflare 차단으로 못 받음")


# ---------------------------------------------------------------- CBOE indices
@step("CBOE volatility indices history")
def fetch_cboe_idx():
    merged = None
    for idx in CBOE_IDX:
        r = requests.get(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{idx}_History.csv", headers=H, timeout=60)
        if r.status_code != 200:
            META["errors"][f"cboe_{idx}"] = f"HTTP {r.status_code}"
            continue
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().upper() for c in df.columns]
        df["date"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
        val_col = "CLOSE" if "CLOSE" in df.columns else idx
        s = df[["date", val_col]].rename(columns={val_col: idx})
        merged = s if merged is None else merged.merge(s, on="date", how="outer")
    merged = merged.sort_values("date")
    merged = merged[merged["date"] >= "2015-01-01"]
    merged["VIX_VIX3M"] = merged["VIX"] / merged["VIX3M"]
    return save(merged, "us_cboe_vol.csv", "CBOE 공식 지수 종가; VIX_VIX3M>1이면 역전(단기 공포)")


# ---------------------------------------------------------------- CBOE put/call daily
def _cboe_pc_one(d: date):
    url = f"https://cdn.cboe.com/data/us/options/market_statistics/daily/{d.isoformat()}_daily_options"
    try:
        r = requests.get(url, headers=H, timeout=30)
        if r.status_code != 200:
            return None
        j = r.json()
    except Exception:  # noqa: BLE001
        return None
    row = {"date": d}
    for item in j.get("ratios", []):
        n = item["name"]
        key = {"TOTAL PUT/CALL RATIO": "pc_total", "INDEX PUT/CALL RATIO": "pc_index",
               "EXCHANGE TRADED PRODUCTS PUT/CALL RATIO": "pc_etp", "EQUITY PUT/CALL RATIO": "pc_equity",
               "CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO": "pc_vix", "SPX + SPXW PUT/CALL RATIO": "pc_spx"}.get(n)
        if key:
            try:
                row[key] = float(item["value"])
            except Exception:  # noqa: BLE001
                row[key] = None
    for sec, pre in [("SUM OF ALL PRODUCTS", "all"), ("INDEX OPTIONS", "index"), ("EQUITY OPTIONS", "equity"),
                     ("EXCHANGE TRADED PRODUCTS", "etp"), ("CBOE VOLATILITY INDEX (VIX)", "vix"), ("SPX + SPXW", "spx")]:
        for item in j.get(sec, []):
            kind = "vol" if item["name"] == "VOLUME" else "oi"
            row[f"{pre}_{kind}_call"] = item.get("call")
            row[f"{pre}_{kind}_put"] = item.get("put")
    return row


@step("CBOE put/call daily (JSON per trading day)")
def fetch_cboe_pc():
    days = list(bdays(CBOE_PC_START, TODAY))
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = [r for r in ex.map(_cboe_pc_one, days) if r]
    df = pd.DataFrame(rows).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    return save(df, "us_cboe_putcall.csv", f"CBOE 일별 풋/콜 비율·거래량·OI {CBOE_PC_START}~ (휴장일은 없음)")


# ---------------------------------------------------------------- CFTC TFF
@step("CFTC TFF (Socrata)")
def fetch_cftc():
    frames = []
    names = ",".join(f"'{n}'" for n in CFTC_MARKETS)
    for ds, label in [("gpe5-46if", "futures_only"), ("yw9f-hn96", "futures_and_options")]:
        r = requests.get(f"https://publicreporting.cftc.gov/resource/{ds}.json",
                         params={"$where": f"report_date_as_yyyy_mm_dd >= '2022-01-01' AND contract_market_name in ({names})",
                                 "$limit": 10000, "$order": "report_date_as_yyyy_mm_dd"}, headers=H, timeout=120)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        keep = ["report_date_as_yyyy_mm_dd", "contract_market_name", "open_interest_all",
                "dealer_positions_long_all", "dealer_positions_short_all",
                "asset_mgr_positions_long", "asset_mgr_positions_short",
                "lev_money_positions_long", "lev_money_positions_short", "lev_money_positions_spread",
                "other_rept_positions_long", "other_rept_positions_short",
                "nonrept_positions_long_all", "nonrept_positions_short_all",
                "change_in_lev_money_long", "change_in_lev_money_short"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df = df.rename(columns={"report_date_as_yyyy_mm_dd": "report_date", "contract_market_name": "market"})
        df["report_date"] = pd.to_datetime(df["report_date"]).dt.date
        for c in df.columns[2:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["lev_net"] = df["lev_money_positions_long"] - df["lev_money_positions_short"]
        df["asset_mgr_net"] = df["asset_mgr_positions_long"] - df["asset_mgr_positions_short"]
        df["dealer_net"] = df["dealer_positions_long_all"] - df["dealer_positions_short_all"]
        df["lev_net_pct_oi"] = df["lev_net"] / df["open_interest_all"] * 100
        df["report_type"] = label
        # 화요일 기준 → 금요일 15:30 ET 발표. 공개 시점을 release_date로 둔다
        df["release_date"] = pd.to_datetime(df["report_date"]) + pd.Timedelta(days=3)
        frames.append(df)
    out = pd.concat(frames).sort_values(["report_type", "market", "report_date"])
    return save(out, "us_cftc_tff.csv", "CFTC TFF 레버리지펀드·자산운용·딜러 포지션; release_date=report_date+3일(금)")


# ---------------------------------------------------------------- FINRA margin
@step("FINRA margin debt (thetrading.tools mirror + convextrade cross-check)")
def fetch_margin():
    s = requests.get("https://www.thetrading.tools/margin-debt", headers=H, timeout=60).text.replace('\\"', '"')
    pairs = re.findall(r'"date":"(\d{4}-\d{2}-\d{2})","primary":(\d+),"secondary":([\d.\-]+|null),"values":\{([^}]*)\}', s)
    rows = []
    for d, prim, sec, vals in pairs:
        row = {"month": d[:7], "debit_balances_musd": int(prim), "yoy_pct": None if sec == "null" else float(sec)}
        for k, v in re.findall(r'"([^"]+)":(-?[\d.]+|null)', vals):
            row[k.replace(":", "_")] = None if v == "null" else float(v)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("month")
    # 교차 확인: convextrade 표
    x = requests.get("https://convextrade.com/metrics/finra-margin-debt", headers=H, timeout=60).text
    tb = re.findall(r'<table.*?</table>', x, flags=re.S)
    check = {}
    if tb:
        for w in re.findall(r'<tr.*?</tr>', tb[0], flags=re.S):
            cells = [re.sub(r'<[^>]+>', '', c).strip() for c in re.findall(r'<td[^>]*>(.*?)</td>', w, flags=re.S)]
            if len(cells) >= 2:
                try:
                    dt = datetime.strptime(cells[0], "%b %d, %Y")
                    check[dt.strftime("%Y-%m")] = int(cells[1].replace(",", ""))
                except Exception:  # noqa: BLE001
                    pass
    df["convextrade_value"] = df["month"].map(check)
    df["cross_check_ok"] = (df["convextrade_value"] == df["debit_balances_musd"]) | df["convextrade_value"].isna()
    # 발표 시차: 다음 달 셋째 주 → 공개 가능일을 월말+20일로 둔다
    df["release_date_est"] = (pd.to_datetime(df["month"] + "-01") + pd.offsets.MonthEnd(0) + pd.Timedelta(days=20)).dt.date
    META["notes"]["margin_cross_check_mismatch"] = df.loc[~df["cross_check_ok"], "month"].tolist()
    return save(df, "us_finra_margin.csv", "FINRA 월간 마진부채(백만$) 미러; convextrade 표와 교차; release_date_est=월말+20일")


# ---------------------------------------------------------------- NAAIM
@step("NAAIM exposure index (public, 3-month delayed since 2026-08-01)")
def fetch_naaim():
    t = requests.get("https://index.naaim.org/embeddable/table", headers=H, timeout=60).text
    rows = re.findall(r'<tr.*?</tr>', t, flags=re.S)
    parsed = [[re.sub(r'<[^>]+>', '', c).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, flags=re.S)] for row in rows]
    hdr = ["date", "naaim_number", "most_bearish", "q1", "q2", "q3", "most_bullish", "std_dev"]
    df = pd.DataFrame(parsed[1:], columns=hdr[:len(parsed[1])])
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    for c in hdr[1:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("date")
    df["release_date"] = df["date"] + pd.Timedelta(days=1)  # 수요일 설문 → 목요일 발표(2026-08 이전 공개 기준)
    return save(df, "us_naaim.csv", "NAAIM 공개 표(2024-01~). 2026-08-01부터 비회원 공개는 3개월 지연·최신값 유료")


# ---------------------------------------------------------------- AAII
@step("AAII sentiment (xls)")
def fetch_aaii():
    r = requests.get("https://www.aaii.com/files/surveys/sentiment.xls", headers=H, timeout=120)
    x = pd.read_excel(io.BytesIO(r.content), sheet_name="SENTIMENT", header=None)
    body = x.iloc[5:].copy()
    body = body[pd.to_datetime(body[0], errors="coerce").notna()]
    df = pd.DataFrame({
        "date": pd.to_datetime(body[0]),
        "bullish": pd.to_numeric(body[1], errors="coerce"),
        "neutral": pd.to_numeric(body[2], errors="coerce"),
        "bearish": pd.to_numeric(body[3], errors="coerce"),
        "bull_8wk_ma": pd.to_numeric(body[5], errors="coerce"),
        "bull_bear_spread": pd.to_numeric(body[6], errors="coerce"),
        "spx_weekly_close": pd.to_numeric(body[12], errors="coerce"),
    }).sort_values("date")
    df = df[df["date"] >= "2015-01-01"]
    df["release_date"] = df["date"] + pd.Timedelta(days=1)  # 수요일 마감 설문 → 목요일 발표
    return save(df, "us_aaii.csv", "AAII 주간 설문(비율 0~1); release_date=설문일+1")


# ---------------------------------------------------------------- FINRA short interest
@step("FINRA consolidated short interest (API)")
def fetch_short_interest():
    frames = []
    for sym in SI_SYMBOLS:
        body = {"compareFilters": [{"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": sym}], "limit": 2000}
        r = requests.post("https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest", json=body,
                          headers={**H, "Accept": "application/json", "Content-Type": "application/json"}, timeout=60)
        if r.status_code != 200:
            META["errors"][f"finra_si_{sym}"] = f"HTTP {r.status_code}"
            continue
        j = r.json()
        if not j:
            META["errors"][f"finra_si_{sym}"] = "no rows"
            continue
        df = pd.DataFrame(j)[["settlementDate", "symbolCode", "currentShortPositionQuantity", "previousShortPositionQuantity",
                              "averageDailyVolumeQuantity", "daysToCoverQuantity", "changePercent", "marketClassCode"]]
        frames.append(df)
    out = pd.concat(frames).rename(columns={"settlementDate": "settlement_date", "symbolCode": "symbol",
                                            "currentShortPositionQuantity": "short_interest",
                                            "previousShortPositionQuantity": "short_interest_prev",
                                            "averageDailyVolumeQuantity": "adv", "daysToCoverQuantity": "days_to_cover",
                                            "changePercent": "chg_pct", "marketClassCode": "market"})
    out["settlement_date"] = pd.to_datetime(out["settlement_date"])
    out = out.sort_values(["symbol", "settlement_date"])
    out["release_date_est"] = out["settlement_date"] + pd.offsets.BDay(9)  # 결제일 기준 약 9영업일 뒤 공표
    return save(out, "us_finra_short_interest.csv", "FINRA 공매도잔고(월 2회, 결제일 기준); release_date_est=결제일+9영업일")


# ---------------------------------------------------------------- FINRA daily short volume
def _cnms_one(d: date):
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d.strftime('%Y%m%d')}.txt"
    try:
        r = requests.get(url, headers=H, timeout=60)
        if r.status_code != 200:
            return []
    except Exception:  # noqa: BLE001
        return []
    out = []
    for line in r.text.splitlines()[1:]:
        p = line.split("|")
        if len(p) >= 5 and p[1] in CNMS_SYMBOLS:
            out.append({"date": d, "symbol": p[1], "short_volume": float(p[2]), "short_exempt_volume": float(p[3]),
                        "total_volume": float(p[4])})
    return out


@step("FINRA daily short volume (CNMS files)")
def fetch_short_volume():
    days = list(bdays(CNMS_START, TODAY))
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for part in ex.map(_cnms_one, days):
            rows.extend(part)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["short_ratio"] = df["short_volume"] / df["total_volume"]
    df = df.sort_values(["symbol", "date"])
    return save(df, "us_finra_short_volume.csv", f"FINRA CNMS 일별 숏볼륨(TRF 보고분, 거래소 전체 아님) {CNMS_START}~; 당일 저녁 공개")


def main():
    skip = set()
    for i, a in enumerate(sys.argv):
        if a == "--skip" and i + 1 < len(sys.argv):
            skip = set(sys.argv[i + 1].split(","))
    fetch_prices()
    fetch_proshares()
    fetch_nvdl()
    fetch_direxion_snapshot()
    fetch_cboe_idx()
    if "cboe_pc" not in skip:
        fetch_cboe_pc()
    fetch_cftc()
    fetch_margin()
    fetch_naaim()
    fetch_aaii()
    fetch_short_interest()
    if "cnms" not in skip:
        fetch_short_volume()
    with open(f"{RESULTS}\\us_fetch_meta.json", "w", encoding="utf-8") as f:
        json.dump(META, f, ensure_ascii=False, indent=2, default=str)
    print("meta saved; errors:", META["errors"])


if __name__ == "__main__":
    main()
