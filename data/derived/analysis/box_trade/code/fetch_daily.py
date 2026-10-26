"""박스권 매매 연구용 일봉: S&P 500 구성 종목(위키백과 목록) + 이격매수 연구 종목(사용자78·AI 127) + 3배 ETF, 2년 일봉(yfinance).
저장: <out>/1d_<티커>.parquet (Open High Low Close Volume, 배당·분할 조정 안 함). 사용: py fetch_daily.py <out>"""
import io, os, sys
import pandas as pd, requests, yfinance as yf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../dip_reason/code"))
from fetch import universe, user_universe

LEV = ["SOXL", "SOXS", "TQQQ", "SQQQ", "TECL", "LABU", "NAIL", "DPST", "KORU", "SPXL", "TNA"]


def sp500():
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    t = pd.read_html(io.StringIO(html), flavor="lxml")[0]
    return [s.replace(".", "-") for s in t["Symbol"].astype(str)]


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sp = sp500()
    syms = sorted(set(sp) | set(universe()) | set(user_universe()["합집합"]) | set(LEV) | {"QQQ", "SPY", "IWM", "SMH"})
    pd.Series(sp).to_csv(os.path.join(out, "sp500_list.csv"), index=False)
    for k in range(0, len(syms), 100):
        part = syms[k:k + 100]
        d = yf.download(part, period="2y", interval="1d", progress=False, auto_adjust=False, group_by="ticker", threads=True)
        for s in part:
            try:
                x = d[s][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            except KeyError:
                continue
            if len(x) > 100:
                x.to_parquet(os.path.join(out, f"1d_{s}.parquet"))
    n = len([f for f in os.listdir(out) if f.startswith("1d_")])
    print("S&P500", len(sp), "| 전체 요청", len(syms), "| 저장", n)
