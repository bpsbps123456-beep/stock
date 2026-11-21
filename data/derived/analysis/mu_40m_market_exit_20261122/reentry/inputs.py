"""클로드 원 지표의 정규장 체결과 SOXX 공시 비교군의 가격을 같은 거래일에 맞춘다."""
from pathlib import Path
import sys
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from check_indicator import grid40, study

FIRST, LAST = "2025-10-01", "2026-11-20"
ORIGINAL = grid40.SYMS
FILING = "https://www.sec.gov/Archives/edgar/data/1100663/000207169125007653/primary_doc.xml"
# 이 공시의 보통주 표기에는 거래소 티커가 없다. 상위권 발행사명과 미국 상장 티커를 대응한다.
TICKERS = {
    "Broadcom, Inc.": "AVGO", "NVIDIA Corp.": "NVDA", "Advanced Micro Devices, Inc.": "AMD",
    "QUALCOMM, Inc.": "QCOM", "Micron Technology, Inc.": "MU", "Intel Corp.": "INTC",
    "Texas Instruments, Inc.": "TXN", "Marvell Technology, Inc.": "MRVL",
    "Lam Research Corp.": "LRCX", "Applied Materials, Inc.": "AMAT",
    "ASML Holding NV": "ASML", "KLA Corp.": "KLAC",
    "Taiwan Semiconductor Manufacturing Co. Ltd.": "TSM",
}


def filing_universe():
    xml = study.HERE / "cache" / "soxx_20250930.xml"
    if xml.exists():
        doc = ElementTree.parse(xml).getroot()
    else:
        response = requests.get(FILING, headers={"User-Agent": "GBRC research bpsbps123456-beep@users.noreply.github.com"}, timeout=25)
        response.raise_for_status()
        doc = ElementTree.fromstring(response.content)
    for node in doc.iter():
        node.tag = node.tag.split("}")[-1]
    if (doc.findtext(".//seriesName"), doc.findtext(".//repPdDate")) != ("iShares Semiconductor ETF", "2025-09-30"):
        raise ValueError("SOXX 공시 식별 불일치")
    equities = [node for node in doc.iter("invstOrSec") if node.findtext("assetCat") == "EC"
                and node.findtext("curCd") == "USD" and node.findtext("payoffProfile") == "Long"]
    rows = []
    for node in sorted(equities, key=lambda n: (-float(n.findtext("pctVal")), n.findtext("name"))):
        name = node.findtext("name")
        symbol = TICKERS[name]
        if symbol not in ORIGINAL:
            rows.append(dict(symbol=symbol, issuer=name, weight_pct=float(node.findtext("pctVal"))))
        if len(rows) == 10:
            break
    if len({r["symbol"] for r in rows}) != 10:
        raise ValueError("공시 비교군 10종목 불일치")
    return pd.DataFrame(rows)


def sip_bars(symbols, timeframe):
    params = dict(symbols=",".join(symbols), timeframe=timeframe, adjustment="split", feed="sip",
                  start=FIRST + "T00:00:00Z", end="2026-11-21T00:00:00Z",
                  asof=LAST, limit=10000, sort="asc")
    parts = []
    with requests.Session() as session:
        session.headers.update(study.keys())
        while True:
            response = session.get(study.URL, params=params, timeout=45)
            response.raise_for_status()
            page = response.json()
            for symbol, bars in page["bars"].items():
                parts.append(pd.DataFrame(bars).assign(symbol=symbol))
            if not page.get("next_page_token"):
                break
            params["page_token"] = page["next_page_token"]
    bars = pd.concat(parts, ignore_index=True)
    bars["t"] = pd.to_datetime(bars.t, utc=True)
    bars = bars.set_index(["symbol", "t"]).sort_index()
    if not bars.index.is_unique:
        raise ValueError("시세 중복")
    return bars[["o", "c"]]


def extra_daily(symbols, calendar):
    bars = sip_bars(symbols, "30Min")
    rows = []
    for symbol in symbols:
        prices = bars.loc[symbol]
        # 정규장 첫 30분봉의 시가, 마지막 30분봉의 종가. 단축장도 거래소 달력을 따른다.
        day = pd.DataFrame({"open": prices.o.reindex(calendar.open).to_numpy(),
                            "close": prices.c.reindex(calendar.close - pd.Timedelta(minutes=30)).to_numpy()},
                           index=calendar.index)
        if day.isna().any().any() or (day <= 0).any().any():
            raise ValueError(f"{symbol} 정규장 시가 또는 종가 누락")
        fast, slow = (grid40.E.ema(day.close, n) for n in (10, 20))
        side = np.select([day.close > np.maximum(fast, slow), day.close < np.minimum(fast, slow)],
                         [1., 0.], default=np.nan)
        day["original_at_open"] = pd.Series(side, index=day.index).ffill().shift().fillna(1).astype(int)
        rows.append(day.assign(symbol=symbol))
    return rows


def load_panel():
    cache = study.HERE / "cache" / "reentry_20261120.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frozen = HERE / "universe.csv"
    if frozen.exists():
        universe = pd.read_csv(frozen)
    else:
        universe = filing_universe()
        universe.to_csv(frozen, index=False, float_format="%.8f")
    cal = study.market_calendar().schedule.loc[FIRST:LAST]
    hours = pd.DatetimeIndex(cal.open).tz_convert("Asia/Seoul").hour
    if not hours.isin([22, 23]).all():
        raise ValueError("주문 가능 시간 밖의 개장")
    rows = extra_daily(universe.symbol.tolist(), cal)
    for symbol in ORIGINAL:
        grid40.E.use(symbol)
        quotes = grid40.E.load(symbol)
        daily = grid40.E.bars(symbol, "D", "reg").set_index("date")
        raw = grid40.refs(symbol)["느린 선(일봉 EMA10·20 종가)"][0]
        # 원 엔진의 바로 그 다음 개장 보유 상태를 쓴다.
        regular_state = pd.Series(raw, index=quotes.index)[quotes.reg]
        at_open = regular_state.groupby(quotes.date).first().reindex(cal.index)
        day = daily.reindex(cal.index)[["Open", "Close"]].set_axis(["open", "close"], axis=1)
        day["original_at_open"] = at_open.to_numpy()
        if day.isna().any().any() or not day.original_at_open.isin([0, 1]).all():
            raise ValueError(f"{symbol} 원 지표 또는 시세 누락")
        rows.append(day.assign(symbol=symbol))
        grid40.E._C.clear()
    panel = pd.concat(rows).rename_axis("date").sort_index()
    cache.parent.mkdir(exist_ok=True)
    panel.to_parquet(cache)
    return panel


def replay_panel():
    """분봉 캐시·API 키 없이 재생하도록 종목별 순수 가격 변화와 개장 신호만 보존한다."""
    path = HERE / "daily_factors.csv"
    if not path.exists():
        panel = load_panel()
        prev = panel.groupby("symbol").close.shift().fillna(panel.open)
        factors = panel[["symbol", "original_at_open"]].copy()
        factors["gap_log"] = np.log(panel.open / prev)
        factors["session_log"] = np.log(panel.close / panel.open)
        factors.to_csv(path, float_format="%.15g")
    factors = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    daily = factors.gap_log + factors.session_log
    close = np.exp(daily.groupby(factors.symbol).cumsum())
    return factors[["symbol", "original_at_open"]].assign(open=close*np.exp(-factors.session_log), close=close)


if __name__ == "__main__":
    print(load_panel().groupby("symbol").size().to_string())
