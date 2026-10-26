"""이격 매수 '이유 있는 하락' 검증용 데이터: yfinance 1분봉(최근 29일, 프리·애프터 포함)과 5분봉(60일), 일봉(1년).
대상: watchlist.AI_UNIVERSE 미국 상장 115종목 + 바이오·유전체 12종목 + 비교 ETF 7개, 그리고 사용자 종목(user_universe).
TradingView MCP는 스크립트에 연결하지 않는다(AGENTS.md). 저장: <out>/{1m,5m,1d}_<티커>.parquet
사용: py fetch.py <out> [티커 ...]   (티커를 주면 그 종목만 받는다)"""
import json, os, re, sys
import pandas as pd, yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))
import watchlist as w

BIO = ["TWST", "TXG", "RXRX", "GH", "NTRA", "ILMN", "CRSP", "NTLA", "BEAM", "SDGR", "VCYT", "GRAL"]
ETF = ["QQQ", "SMH", "IGV", "ARKG", "XBI", "IWM", "SPY"]


def universe():
    """{티커: 그룹}. TEM은 바이오·유전체 그룹으로 본다(같이 움직이는 쪽)."""
    g = {}
    for big, subs in w.AI_UNIVERSE.items():
        for ts in subs.values():
            for t in ts:
                if re.fullmatch(r"[A-Z]{1,5}", t):
                    g[t] = big
    for t in BIO + ["TEM"]:
        g[t] = "바이오·유전체"
    return g


# 2026-10-07 사용자가 알려 준 트레이딩뷰 'AI' 관심목록(종목 이름만 옮김, 트레이딩뷰 데이터는 쓰지 않음)
TV_AI = ["APLD", "IREN", "NBIS", "NVTS", "AMAT", "KLAC", "TER", "ENTG", "TXN", "NXPI", "MCHP", "ON", "MPWR",
         "NVDA", "AMD", "AVGO", "MRVL", "MU", "ALAB", "CRDO", "ARM", "RMBS", "QCOM", "SWKS", "TSM", "UMC",
         "INTC", "SNDK", "CRWV", "ORCL", "BE", "STX", "WDC", "DELL", "NOW", "GLW", "CRCL", "SOXX"]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../..")


def etf_us(fund, day="2026-10-06"):
    """TIME(456600)·KoAct(2ETFQ1) 검증 스냅샷의 미국 상장 보유 종목."""
    d = json.load(open(os.path.join(ROOT, f"data/normalized/etf_holdings/{day}_{fund}.json"), encoding="utf-8"))
    return [m.group(1) for h in d["holdings"] if h["asset_type"] == "security"
            and (m := re.fullmatch(r"([A-Z.]+) US EQUITY", h["code"]))]


def user_universe():
    """{이름: 티커 목록} — 트레이딩뷰 AI 목록, TIME·KoAct 미국 보유, 그 합집합."""
    t, k = etf_us("456600"), etf_us("2ETFQ1")
    return {"트레이딩뷰 AI": TV_AI, "TIME·KoAct": sorted(set(t) | set(k)), "합집합": sorted(set(TV_AI) | set(t) | set(k))}


def save(d, out, iv):
    for t in d.columns.get_level_values(0).unique():
        x = d[t].dropna(subset=["Open", "High", "Low", "Close"])
        if len(x):
            x.index = x.index.tz_convert("America/New_York")
            x[["Open", "High", "Low", "Close", "Volume"]].to_parquet(os.path.join(out, f"{iv}_{t}.parquet"))


def download(syms, out):
    end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    parts = []
    a = end - pd.Timedelta(days=29)
    while a < end:                                    # 1분봉은 한 번에 7일까지
        b = min(a + pd.Timedelta(days=7), end)
        parts.append(yf.download(syms, interval="1m", start=a.strftime("%Y-%m-%d"), end=b.strftime("%Y-%m-%d"),
                                 prepost=True, progress=False, auto_adjust=False, group_by="ticker", threads=True))
        a = b
    m1 = pd.concat(parts).sort_index()
    m1 = m1[~m1.index.duplicated()]
    save(m1, out, "1m")
    save(yf.download(syms, interval="5m", period="60d", prepost=True, progress=False, auto_adjust=False,
                     group_by="ticker", threads=True), out, "5m")
    dd = yf.download(syms, interval="1d", period="1y", progress=False, auto_adjust=False, group_by="ticker", threads=True)
    for t in dd.columns.get_level_values(0).unique():
        x = dd[t].dropna(subset=["Close"])
        if len(x):
            x[["Open", "High", "Low", "Close", "Volume"]].to_parquet(os.path.join(out, f"1d_{t}.parquet"))
    return m1


if __name__ == "__main__":
    out = sys.argv[1]
    syms = sys.argv[2:] or list(universe()) + ETF
    m1 = download(syms, out)
    n1 = len([f for f in os.listdir(out) if f.startswith("1m_")])
    print("종목", len(syms), "1분봉 파일", n1, "범위", m1.index[0], "->", m1.index[-1])
