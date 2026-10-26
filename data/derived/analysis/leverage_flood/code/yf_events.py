"""Pull yfinance daily windows around the historical leverage crashes and compute simple timing facts.
Usage: py yf_events.py
"""
import os

import pandas as pd
import yfinance as yf

RESULTS = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis\leverage_flood\results"

WINDOWS = {
    "2015_china": (["000001.SS"], "2015-04-01", "2015-09-30"),
    "2018_xiv": (["^VIX", "^VIX3M", "^GSPC"], "2017-12-01", "2018-03-15"),
    "2020_covid": (["^VIX", "^VIX3M", "^GSPC"], "2020-01-15", "2020-04-15"),
    "2021_archegos": (["^GSPC", "VIAC", "DISCA"], "2021-02-15", "2021-04-15"),
    "2024_yen": (["^N225", "JPY=X", "^VIX", "^VIX3M", "^GSPC"], "2024-06-01", "2024-08-31"),
    "2026_korea": (["^KS11", "005930.KS", "000660.KS", "^VIX", "^VIX3M", "^SOX"], "2026-01-01", "2026-10-14"),
}


def pull(tickers, start, end):
    frames = []
    for t in tickers:
        try:
            d = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        except Exception as exc:  # noqa: BLE001
            print("  error", t, exc)
            continue
        if d is None or d.empty:
            print("  empty", t)
            continue
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        frames.append(d["Close"].rename(t))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def main():
    for name, (tickers, start, end) in WINDOWS.items():
        print(f"== {name}")
        df = pull(tickers, start, end)
        if df.empty:
            print("  nothing")
            continue
        path = os.path.join(RESULTS, f"yf_{name}.csv")
        df.to_csv(path, encoding="utf-8-sig")
        print("  saved", path, df.shape)
        if "^VIX" in df.columns and "^VIX3M" in df.columns:
            ratio = (df["^VIX"] / df["^VIX3M"]).dropna()
            inv = ratio[ratio > 1.0]
            print("  VIX/VIX3M>1 dates:", [d.strftime("%Y-%m-%d") for d in inv.index[:12]], "... total", len(inv))
        for t in df.columns:
            s = df[t].dropna()
            if s.empty:
                continue
            pk = s.idxmax()
            tr = s.idxmin()
            ret = s.pct_change()
            worst = ret.idxmin()
            print(f"  {t}: max {pk.date()} {s[pk]:.2f} | min {tr.date()} {s[tr]:.2f} | worst day {worst.date()} {ret[worst]*100:.2f}%")


if __name__ == "__main__":
    main()
