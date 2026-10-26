"""급등 후 눌림목 연구용 정규장 일봉(2026-10-10, 키 surge_pullback).

premarket_shock/bars의 1시간봉(현재 상장·시총 3억 달러 이상 3,570종목 + SPY·QQQ, 2023-11~2026-10-07 09:30)에서
정규장(09:30~16:00 ET) 봉만 묶어 일봉을 만든다. 프리·애프터 봉은 쓰지 않는다(yfinance 장외 봉의 엉터리 고가·저가 때문,
rsi_second_dip 연구에서 확인). 마지막 날(10/7)은 09:30 봉 하나뿐이라 뺀다.
열: sym day o h l c v(정규장 거래량 합) nbar(정규장 봉 수, 반일장·자료 빠짐 확인용)
메모리: 종목 하나씩 읽고 작은 배열만 모은다.
사용: py build_daily.py → <scratchpad>/sp_daily.parquet
"""
import glob, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.normpath(os.path.join(HERE, "..", ".."))
B1H = os.path.join(AN, "premarket_shock", "bars")
SP = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/6a57b230-3ac1-4ae6-9d34-b1d8a1c2ce1b/scratchpad"
LAST = pd.Timestamp("2026-10-07").date()


def main():
    parts = []
    files = sorted(glob.glob(os.path.join(B1H, "1h_*.parquet")))
    for k, f in enumerate(files):
        s = os.path.basename(f)[3:-8]
        d = pd.read_parquet(f)
        d = d[d.High > 0]
        mod = d.index.hour * 60 + d.index.minute
        d = d[(mod >= 570) & (mod < 960)]
        if not len(d):
            continue
        day = d.index.date
        g = d.groupby(day)
        x = pd.DataFrame({"o": g.Open.first(), "h": g.High.max(), "l": g.Low.min(), "c": g.Close.last(),
                          "v": g.Volume.sum(), "nbar": g.Close.size()})
        x = x[x.index < LAST]
        x["sym"] = s
        x.index.name = "day"
        parts.append(x.reset_index().astype({"o": "float32", "h": "float32", "l": "float32", "c": "float32",
                                             "v": "float64", "nbar": "int8"}))
        if k % 500 == 0:
            print(k, "/", len(files), flush=True)
    out = pd.concat(parts, ignore_index=True)
    out["sym"] = out.sym.astype("category")
    out.to_parquet(os.path.join(SP, "sp_daily.parquet"))
    print("행", len(out), "종목", out.sym.nunique(), out.day.min(), "->", out.day.max())


if __name__ == "__main__":
    main()
