"""시장·매크로 일봉(yfinance)과 FRED 시계열을 한 번 받아 cache/에 고정한다(2026-10-10).

- ohlc.parquet: 기본 경로가 들고 있던 모든 종목 + 참고 지수의 일봉 OHLCV(분할 수정, 배당 미수정)와 Adj Close. long 형식(sym, date, ...).
- breadth.parquet: watchlist.AI_UNIVERSE 미국 상장 종목 종가(Adj Close) — 시장 폭 계산용. wide 형식.
- fred.parquet: FRED 일·주간 시계열(HY·IG 스프레드, 금융스트레스 등). 공개 날짜 기준이 아니라 관측일 기준이라
  신호로 쓸 때는 하루 더 늦춘다(harness가 안내).
TradingView MCP는 스크립트에 연결하지 않는다(AGENTS.md). 가격은 yfinance만 쓴다.
"""
import io, json, sys, urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
CACHE = HERE.parent / "cache"
START, END = "2022-06-01", "2026-10-10"

REFS = ["QQQ", "SPY", "RSP", "IWM", "SOXX", "SMH", "^SOX", "^NDX", "^GSPC", "^VIX", "^VIX3M", "^VIX9D", "^VVIX",
        "^MOVE", "^SKEW", "HYG", "LQD", "JNK", "IEF", "TLT", "SHY", "^TNX", "^IRX", "^FVX", "DX-Y.NYB", "JPY=X",
        "KRW=X", "GLD", "USO", "CL=F", "BTC-USD", "SOXS", "SOXL", "SQQQ", "PSQ", "SH", "QID", "NVDA", "MSFT", "META",
        "GOOGL", "AMZN", "AVGO", "MU", "TSM", "XLK", "IGV", "XLU", "XLP", "456600.KS", "^KS11"]
FRED = ["BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLH0A3HYC", "T10Y2Y", "T10Y3M", "STLFSI4", "NFCI", "DGS10", "DGS2",
        "VIXCLS", "DTWEXBGS"]


def base_tickers():
    out = set()
    for name in ("base_2026", "base_2026_774", "base_hold"):
        b = json.loads((CACHE / f"{name}.json").read_text(encoding="utf-8"))
        out |= set(b["ticker"].values())
    return sorted(out)


def ai_universe_us():
    import watchlist
    out = []
    for sub in watchlist.AI_UNIVERSE.values():
        for syms in sub.values():
            out += [s for s in syms if "." not in s and s.isascii()]
    return sorted(set(out))


def download(tickers):
    import yfinance as yf
    f = yf.download(tickers, start=START, end=END, auto_adjust=False, progress=False, threads=True, group_by="column")
    rows = []
    for field in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        x = f[field].stack(future_stack=True).rename(field)
        rows.append(x)
    df = pd.concat(rows, axis=1).reset_index()
    df.columns = ["date", "sym", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return df.dropna(subset=["Close"])


def fred(series):
    out = {}
    for s in series:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}&cosd={START}&coed={END}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = pd.read_csv(io.BytesIO(r.read()))
            d.columns = ["date", s]
            d["date"] = pd.to_datetime(d["date"])
            out[s] = pd.to_numeric(d.set_index("date")[s], errors="coerce")
        except Exception as e:
            print("FRED 실패", s, type(e).__name__, e)
    return pd.DataFrame(out)


def main():
    tick = sorted(set(base_tickers()) | set(REFS))
    ohlc = download(tick)
    ohlc.to_parquet(CACHE / "ohlc.parquet")
    miss = sorted(set(tick) - set(ohlc.sym))
    print("ohlc", ohlc.shape, "종목", ohlc.sym.nunique(), "빠짐", miss)
    uni = ai_universe_us()
    br = download(uni)
    wide = br.pivot(index="date", columns="sym", values="Adj Close")
    wide.to_parquet(CACHE / "breadth.parquet")
    print("breadth", wide.shape, "AI 미국 종목", len(uni))
    fr = fred(FRED)
    fr.to_parquet(CACHE / "fred.parquet")
    print("fred", fr.shape, {c: (str(fr[c].dropna().index.min().date()), str(fr[c].dropna().index.max().date()))
                             for c in fr.columns})


if __name__ == "__main__":
    main()
