"""4단계(일반 주식·레버리지 상품 손절) 공통 자료(2026-10-11).

- p4_stocks.parquet: 데스크 '급락 후 담기' 종목표(data/derived/desk/dip_universe.json, 시총 10억 달러 이상 미국 보통주 2,535개,
  2026-10 기준) 일봉 2014-01-01~2026-10-09. 생존편향: 지금 살아 있는 종목만 있다(상장폐지·파산 종목 없음 → 손절에 불리).
- p4_lev.parquet: 실제 레버리지 ETF와 그 기초자산·참고 지수, ^IRX(단기 금리, 합성 레버리지 금융비용).
- p4_meta.json: 종목별 시총·섹터(종목표 기준), 레버리지 상품표(배수·기초자산).
long 형식: date, sym, Open, High, Low, Close, Adj Close, Volume (분할 수정, 배당 미수정; Adj Close는 배당 수정).
"""
import json, sys, time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CACHE = HERE.parent / "cache"
START, END = "2014-01-01", "2026-10-10"

# 레버리지 상품: 티커 → (배수, 기초자산). 기초자산이 지수면 대표 ETF로 대리(SOXL·SOXX는 같은 ICE 반도체 지수).
LEV = {
    "TQQQ": (3, "QQQ"), "QLD": (2, "QQQ"), "SOXL": (3, "SOXX"), "USD": (2, "SOXX"), "UPRO": (3, "SPY"),
    "SPXL": (3, "SPY"), "SSO": (2, "SPY"), "TECL": (3, "XLK"), "TNA": (3, "IWM"), "LABU": (3, "XBI"),
    "NVDL": (2, "NVDA"), "NVDX": (2, "NVDA"), "NVDU": (2, "NVDA"), "TSLL": (2, "TSLA"), "TSLT": (2, "TSLA"),
    "MSTU": (2, "MSTR"), "MSTX": (2, "MSTR"), "CONL": (2, "COIN"), "AMDL": (2, "AMD"), "PLTU": (2, "PLTR"),
    "MUU": (2, "MU"), "AVL": (2, "AVGO"), "SMCX": (2, "SMCI"), "TSMX": (2, "TSM"), "METU": (2, "META"),
    "AAPU": (2, "AAPL"), "GGLL": (2, "GOOGL"), "AMZU": (2, "AMZN"), "MSFU": (2, "MSFT"), "NFXL": (2, "NFLX"),
    "HOOX": (2, "HOOD"), "IONX": (2, "IONQ"), "RGTX": (2, "RGTI"), "QUBX": (2, "QUBT"), "SNDU": (2, "SNDK"),
    "ARMG": (2, "ARM"), "CRWG": (2, "CRWV"), "OKLL": (2, "OKLO"), "AVGX": (2, "AVGO"),
}
REFS = ["QQQ", "SPY", "SOXX", "XLK", "IWM", "XBI", "^IRX", "^VIX"]


def download(tickers, batch=150):
    import yfinance as yf
    parts = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        for attempt in range(3):
            try:
                f = yf.download(chunk, start=START, end=END, auto_adjust=False, progress=False, threads=True,
                                group_by="column")
                break
            except Exception as e:                      # 네트워크 흔들림
                print("재시도", i, type(e).__name__, e)
                time.sleep(5)
        else:
            continue
        cols = [f[c].stack(future_stack=True).rename(c) for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume")]
        df = pd.concat(cols, axis=1).reset_index()
        df.columns = ["date", "sym", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        parts.append(df.dropna(subset=["Close"]))
        print(f"{i + len(chunk)}/{len(tickers)}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out


def main():
    uni = json.loads((ROOT / "data/derived/desk/dip_universe.json").read_text(encoding="utf-8"))["symbols"]
    syms = sorted(s for s in uni if s.replace("-", "").replace(".", "").isalnum())
    lev_syms = sorted(set(LEV) | {u for _, u in LEV.values()} | set(REFS))
    lev = download(lev_syms)
    lev.to_parquet(CACHE / "p4_lev.parquet")
    print("레버리지 자료", lev.sym.nunique(), "/", len(lev_syms), "빠짐", sorted(set(lev_syms) - set(lev.sym)))
    st = download([s.replace(".", "-") for s in syms])
    st.to_parquet(CACHE / "p4_stocks.parquet")
    first = st.groupby("sym").date.min()
    print("주식", st.sym.nunique(), "/", len(syms), "· 2016 이전 시작", int((first < "2016-01-01").sum()),
          "· 2022 이전", int((first < "2022-01-01").sum()))
    meta = {"stocks": {s.replace(".", "-"): uni[s] for s in syms}, "lev": {k: {"L": v[0], "under": v[1]} for k, v in LEV.items()}}
    (CACHE / "p4_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
