"""단서 자료: ^VIX·^VIX3M 일봉(2022-06~), 종목 실적 발표일(yfinance get_earnings_dates, 최대 28분기).
→ cache/vix.parquet, cache/earnings.parquet(sym, ts_et, date_et, session: bmo/amc/day)"""
import pathlib, time
import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).resolve().parent.parent
d = yf.download(["^VIX", "^VIX3M"], start="2022-06-01", end="2026-10-23", interval="1d", auto_adjust=False, progress=False)["Close"]
d.index = pd.DatetimeIndex(d.index).date
d.columns = ["VIX", "VIX3M"] if list(d.columns) == ["^VIX", "^VIX3M"] else [c.replace("^", "") for c in d.columns]
d.to_parquet(HERE / "cache" / "vix.parquet")
print("VIX", len(d), d.index.min(), d.index.max(), d.isna().sum().to_dict(), flush=True)

daily = pd.read_parquet(HERE / "cache" / "daily.parquet")
ETF = {"SPY", "QQQ", "SOXX", "SMH", "IGV", "IWM", "XBI", "ARKG"}
syms = sorted(set(daily.sym) - ETF)
rows, fail = [], []
for s in syms:
    for k in range(3):
        try:
            e = yf.Ticker(s).get_earnings_dates(limit=28)
            break
        except Exception as ex:
            e = None
            time.sleep(3 * (k + 1))
    if e is None or len(e) == 0:
        fail.append(s); continue
    t = pd.DatetimeIndex(e.index).tz_convert("America/New_York")
    for ts in t:
        m = ts.hour * 60 + ts.minute
        sess = "bmo" if m < 570 else ("amc" if m >= 960 else "day")
        rows.append((s, ts, ts.date(), sess))
out = pd.DataFrame(rows, columns=["sym", "ts_et", "date_et", "session"]).drop_duplicates(["sym", "date_et"])
out.to_parquet(HERE / "cache" / "earnings.parquet")
print("실적", out.sym.nunique(), "종목", len(out), "건", "실패", fail, flush=True)
