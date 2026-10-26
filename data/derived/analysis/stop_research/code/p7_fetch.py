"""7단계(마삼룰·폭락 후 기계적 매수) 자료(2026-10-11): 나스닥 종합(^IXIC) 등 긴 지수 일봉.
마삼룰 기준 지수는 나스닥 종합지수다(sondoctor.co.kr/664). p5_lev_long.parquet에는 ^NDX·^GSPC·^SOX·EWY·^VIX·^IRX가 있다.
출력 cache/p7_idx.parquet (long: date, sym, Open, High, Low, Close, Adj Close, Volume)."""
import time
from pathlib import Path
import pandas as pd
CACHE = Path(__file__).resolve().parent.parent / "cache"
SYMS = ["^IXIC", "^NDX", "^GSPC", "^RUT", "^SOX", "^VIX", "^IRX", "QQQ", "SPY", "EWY"]

def main():
    import yfinance as yf
    for _ in range(3):
        try:
            f = yf.download(SYMS, start="1971-01-01", end="2026-10-10", auto_adjust=False, progress=False,
                            threads=True, group_by="column"); break
        except Exception as e:
            print("재시도", e); time.sleep(5)
    cols = [f[c].stack(future_stack=True).rename(c) for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume")]
    df = pd.concat(cols, axis=1).reset_index()
    df.columns = ["date", "sym", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df.dropna(subset=["Close"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_parquet(CACHE / "p7_idx.parquet")
    g = df.groupby("sym")
    print(pd.DataFrame({"시작": g.date.min(), "끝": g.date.max(), "일수": g.size()}).to_string())
    ix = df[df.sym == "^IXIC"].set_index("date").Close
    r = ix.pct_change()
    print("나스닥 종합 −3% 이하 날:", int((r <= -0.03).sum()), "| 2026:", int((r[r.index >= "2026-01-01"] <= -0.03).sum()),
          list(r[(r.index >= "2026-01-01") & (r <= -0.03)].index))

if __name__ == "__main__":
    main()
