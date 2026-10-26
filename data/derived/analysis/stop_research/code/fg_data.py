"""8단계 CNN 공포탐욕지수(F&G) 자료(2026-10-11). 모든 갈래가 이 파일로 같은 값을 읽는다.

출처 두 개를 잇는다:
  - CNN 공식 graphdata(production.dataviz.cnn.io/index/fearandgreed/graphdata/2020-09-01): 2020-09-01~, 지수와 구성요소 7개 원값.
    2020-09~2021-01은 50.0 자리표시가 섞여 있어(2020년 75일·2021년 13일) 2021-02-01부터만 쓴다.
  - GitHub whit3rabbit/fear-greed-data fear-greed.csv(정수, 2011-01-03~): 2021-02 이후 공식값과 평균 차 0.14점, 상관 0.997.
    2011~2020 값은 CNN이 2020년 개편 때 거꾸로 채운 값일 수 있다(정규화 방식이 그때 정해졌을 위험) → 한계로 적는다.
fg(): 날짜 문자열('YYYY-MM-DD') 색인 Series, 2021-01-31까지 보관본·그 뒤 공식값. 값은 그날 미국 장 마감 뒤 확정 → 다음 거래일 시가부터 쓴다.
components(): 공식 구성요소 원값(2020-09~, 같은 이유로 2021-02 이후 권장). 열: momentum_sp500(S&P 종가), momentum_sp125(125일 평균),
  strength(52주 신고가−신저가 순비율), breadth(맥클렐런 거래량 합산), put_call(5일 평균), vix, vix_50, junk(정크−투자등급 금리차), safe_haven(20일 주식−채권 수익 차).
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
CUT = "2021-02-01"
_COMP = {"market_momentum_sp500": "momentum_sp500", "market_momentum_sp125": "momentum_sp125", "stock_price_strength": "strength",
         "stock_price_breadth": "breadth", "put_call_options": "put_call", "market_volatility_vix": "vix",
         "market_volatility_vix_50": "vix_50", "junk_bond_demand": "junk", "safe_haven_demand": "safe_haven"}


def fetch():
    h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
         "Accept": "application/json, text/plain, */*", "Origin": "https://edition.cnn.com", "Referer": "https://edition.cnn.com/"}
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/2020-09-01"
    (CACHE / "fg_cnn.json").write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30).read())
    url = "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/fear-greed.csv"
    (CACHE / "fg_archive.csv").write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read())


def _series(points) -> pd.Series:
    s = pd.Series([p["y"] for p in points], index=pd.to_datetime([p["x"] for p in points], unit="ms").strftime("%Y-%m-%d"), dtype=float)
    return s[~s.index.duplicated(keep="last")].sort_index()


def fg() -> pd.Series:
    j = json.loads((CACHE / "fg_cnn.json").read_text())
    off = _series(j["fear_and_greed_historical"]["data"])
    arc = pd.read_csv(CACHE / "fg_archive.csv")
    arc = pd.Series(arc["Fear Greed"].astype(float).to_numpy(), index=arc["Date"].astype(str)).sort_index()
    s = pd.concat([arc[arc.index < CUT], off[off.index >= CUT]])
    return s[~s.index.duplicated(keep="last")].rename("fg")


def components() -> pd.DataFrame:
    j = json.loads((CACHE / "fg_cnn.json").read_text())
    return pd.DataFrame({v: _series(j[k]["data"]) for k, v in _COMP.items()}).sort_index()


if __name__ == "__main__":
    if not (CACHE / "fg_cnn.json").exists():
        fetch()
    s = fg()
    print("F&G", s.index[0], "~", s.index[-1], len(s), "일 | 2026 최저", s[s.index >= "2026"].min())
    c = components()
    print("구성요소", c.index[0], "~", c.index[-1], c.shape)
    for t in (10, 20, 25):
        d = s[s <= t]
        print(f"≤{t}: {len(d)}일, 해마다", d.groupby(d.index.str[:4]).size().to_dict())
