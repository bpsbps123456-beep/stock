"""2단계(종목 단위 손절·재진입·깊은 하락 매수) 공통 자료: 2014-01-02~2026-10-09 일봉을 한 번 받아 고정한다(2026-10-10).

대상 = watchlist.AI_UNIVERSE 미국 상장 + 7종목 기본 경로(2023-05~2026-10)가 들고 있던 미국 종목 + 참고 ETF.
생존편향: AI_UNIVERSE는 2026-10 목록이라 이긴 종목 쪽으로 기운다. TIME 보유 이력(QUBT 등 진 종목 포함)을 더해 조금 줄였지만
상장폐지 종목은 yfinance에 없어 빠진다. 깊은 하락 매수 결론은 이 편향 때문에 좋게 나올 수 있다.
출력 cache/p2_long.parquet (long: date, sym, Open, High, Low, Close, Adj Close, Volume; Open/High/Low/Close는 분할 수정·배당 미수정)
      cache/p2_universe.json (sym → 출처 목록)
"""
import json, sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
CACHE = HERE.parent / "cache"
START, END = "2014-01-01", "2026-10-10"
ETFS = ["QQQ", "SPY", "SOXX", "SMH", "IWM", "RSP", "^VIX", "^VIX3M"]


def main():
    import watchlist
    import yfinance as yf
    src: dict = {}
    for sub in watchlist.AI_UNIVERSE.values():
        for syms in sub.values():
            for s in syms:
                if "." not in s and s.isascii():
                    src.setdefault(s, []).append("AI_UNIVERSE")
    for n in ("base_2026", "base_hold"):
        b = json.loads((CACHE / f"{n}.json").read_text(encoding="utf-8"))
        for t in b["ticker"].values():
            src.setdefault(t, []).append(n)
    for e in ETFS:
        src.setdefault(e, []).append("ETF")
    tick = sorted(src)
    f = yf.download(tick, start=START, end=END, auto_adjust=False, progress=False, threads=True, group_by="column")
    cols = [f[c].stack(future_stack=True).rename(c) for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume")]
    df = pd.concat(cols, axis=1).reset_index()
    df.columns = ["date", "sym", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df.dropna(subset=["Close"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_parquet(CACHE / "p2_long.parquet")
    (CACHE / "p2_universe.json").write_text(json.dumps(src, ensure_ascii=False, indent=0), encoding="utf-8")
    first = df.groupby("sym").date.min()
    print("종목", df.sym.nunique(), "/", len(tick), "행", len(df), "빠짐", sorted(set(tick) - set(df.sym)))
    print("2018 이전 시작 종목", int((first < "2018-01-01").sum()), "· 2022 이전", int((first < "2022-01-01").sum()))


if __name__ == "__main__":
    main()
