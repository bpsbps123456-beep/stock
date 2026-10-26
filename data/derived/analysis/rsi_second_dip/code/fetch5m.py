"""40분봉 RSI 정확 시험용 5분봉(yfinance, 최근 60일, 프리·애프터 포함). 40분봉은 5분봉을 미 동부 04:00 기준으로 묶어 만든다.
대상: 8/11 이후 정규장 종가 기준 하루 -7% 이하가 있던 종목(premarket_shock 1시간봉으로 고름) + SPY·QQQ.
TradingView MCP는 스크립트에 연결하지 않는다(AGENTS.md). 저장: <out>/5m_<티커>.parquet
사용: py fetch5m.py <종목 json> <out> [k n]   (k n = n갈래 중 k번째만 받기)"""
import json, os, sys, time
import pandas as pd, yfinance as yf


def main():
    syms = json.load(open(sys.argv[1])) + ["SPY", "QQQ"]
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    if len(sys.argv) > 4:
        k, n = int(sys.argv[3]), int(sys.argv[4])
        syms = sorted(set(syms))[k::n]
    syms = [s for s in syms if not os.path.exists(os.path.join(out, f"5m_{s}.parquet"))]
    print("받을 종목", len(syms), flush=True)
    for a in range(0, len(syms), 40):
        part = syms[a:a + 40]
        for _ in range(3):
            try:
                d = yf.download(part, interval="5m", period="60d", prepost=True, progress=False, auto_adjust=False,
                                group_by="ticker", threads=True)
                break
            except Exception as e:                       # 일시 오류는 잠깐 쉬고 다시
                print("재시도", e, flush=True)
                time.sleep(10)
        else:
            continue
        for t in part:
            if t not in d.columns.get_level_values(0):
                continue
            x = d[t].dropna(subset=["Open", "High", "Low", "Close"])
            if len(x):
                x.index = x.index.tz_convert("America/New_York")
                x[["Open", "High", "Low", "Close", "Volume"]].to_parquet(os.path.join(out, f"5m_{t}.parquet"))
        print(a + len(part), "/", len(syms), flush=True)


if __name__ == "__main__":
    main()
