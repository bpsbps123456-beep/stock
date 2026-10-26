"""5단계(SOXL·KORU 손절 규칙) 자료(2026-10-11): 3배 상품과 기초 지수의 긴 역사.

- 기초: ^SOX(필라델피아 반도체 지수, 1994~; SOXL은 2021년부터 ICE 반도체 지수지만 같은 종목군), SOXX(2001~),
  EWY(iShares MSCI 한국, 2000~; KORU는 MSCI Korea 25/50 3배, 미국 장 시간에 거래), ^NDX·QQQ, ^GSPC·SPY, IWM, XLK,
  FXI(중국), EEM(신흥), XLF(금융), XBI(바이오), GDX(금광) — 다른 3배 상품으로 규칙이 일반화되는지 보는 데 쓴다.
- 실제 3배: SOXL(2010~), KORU(2013~), TQQQ, UPRO, TNA, TECL, YINN, EDC, FAS, LABU, NUGT.
- ^IRX(13주 국채, 1990~): 합성 금융비용.
주의: 지수(^SOX, ^NDX, ^GSPC)의 초기 일봉은 시가·고가·저가가 종가와 같을 수 있다(종가만 있음). 그 구간에서 장중 손절은 잴 수 없어
종가 판단 규칙만 쓴다. bars_quality()가 시가=고가=저가=종가 비율을 해마다 보인다.
출력 cache/p5_lev_long.parquet (long: date, sym, Open, High, Low, Close, Adj Close, Volume).
"""
import sys, time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
START, END = "1990-01-01", "2026-10-10"
SYMS = ["^SOX", "SOXX", "EWY", "^NDX", "QQQ", "^GSPC", "SPY", "IWM", "XLK", "FXI", "EEM", "XLF", "XBI", "GDX",
        "SOXL", "KORU", "TQQQ", "UPRO", "TNA", "TECL", "YINN", "EDC", "FAS", "LABU", "NUGT", "^IRX", "^VIX", "KRW=X"]


def main():
    import yfinance as yf
    for attempt in range(3):
        try:
            f = yf.download(SYMS, start=START, end=END, auto_adjust=False, progress=False, threads=True, group_by="column")
            break
        except Exception as e:
            print("재시도", type(e).__name__, e)
            time.sleep(5)
    cols = [f[c].stack(future_stack=True).rename(c) for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume")]
    df = pd.concat(cols, axis=1).reset_index()
    df.columns = ["date", "sym", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df.dropna(subset=["Close"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_parquet(CACHE / "p5_lev_long.parquet")
    g = df.groupby("sym")
    flat = df.assign(flat=(df.Open == df.Close) & (df.High == df.Close) & (df.Low == df.Close))
    summ = pd.DataFrame({"시작": g.date.min(), "끝": g.date.max(), "일수": g.size(),
                         "OHLC같음비율": flat.groupby("sym").flat.mean().round(3)})
    print(summ.to_string())
    print("빠짐", sorted(set(SYMS) - set(df.sym)))


if __name__ == "__main__":
    main()
