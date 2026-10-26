"""프리장 급락 사례 연구용 데이터: yfinance 1시간봉(최근 730일, 프리·애프터 포함).
대상: 나스닥 공개 종목표(api.nasdaq.com)의 미국 상장 보통주 중 현재 시총 3억 달러 이상.
현재 시총으로 고르면 급락 뒤 계속 빠진 종목이 빠지므로 문턱을 낮게 잡고, 사건 시점 시총은 analyze.py에서 다시 거른다.
TradingView MCP는 스크립트에 연결하지 않는다(AGENTS.md). 저장: <out>/1h_<티커>.parquet, <out>/universe.json
사용: py fetch.py <out> [티커 ...]"""
import json, os, re, sys, time
import pandas as pd, requests, yfinance as yf


def nasdaq_universe(min_cap=3e8):
    h = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    rows = requests.get("https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true",
                        headers=h, timeout=60).json()["data"]["rows"]
    out = {}
    for r in rows:
        s = r["symbol"].strip()
        try:
            cap, px = float(r["marketCap"] or 0), float(r["lastsale"].replace("$", "") or 0)
        except ValueError:
            continue
        if re.fullmatch(r"[A-Z]{1,5}", s) and cap >= min_cap and px > 0:
            out[s] = {"cap": cap, "price": px, "name": r["name"], "sector": r["sector"], "industry": r["industry"]}
    return out


def save(d, out):
    n = 0
    for t in d.columns.get_level_values(0).unique():
        x = d[t].dropna(subset=["Open", "High", "Low", "Close"])
        if len(x):
            x.index = x.index.tz_convert("America/New_York")
            x[["Open", "High", "Low", "Close", "Volume"]].to_parquet(os.path.join(out, f"1h_{t}.parquet"))
            n += 1
    return n


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[2] == "--part":       # 병렬로 나눠 받기: --part k n
        k, n = int(sys.argv[3]), int(sys.argv[4])
        syms = sorted(json.load(open(os.path.join(out, "universe.json"), encoding="utf-8")))[k::n]
    elif len(sys.argv) > 2:
        syms = sys.argv[2:]
    else:
        uni = nasdaq_universe()
        json.dump(uni, open(os.path.join(out, "universe.json"), "w", encoding="utf-8"), ensure_ascii=False)
        syms = sorted(uni)
    syms = [s for s in syms if not os.path.exists(os.path.join(out, f"1h_{s}.parquet"))]
    print("받을 종목", len(syms), flush=True)
    for i in range(0, len(syms), 40):
        chunk = syms[i:i + 40]
        for attempt in range(3):
            try:
                d = yf.download(chunk, interval="1h", period="730d", prepost=True, progress=False,
                                auto_adjust=False, group_by="ticker", threads=True)
                print(i + len(chunk), "저장", save(d, out), flush=True)
                break
            except Exception as e:
                print("재시도", attempt, e, flush=True)
                time.sleep(20)
        time.sleep(2)
