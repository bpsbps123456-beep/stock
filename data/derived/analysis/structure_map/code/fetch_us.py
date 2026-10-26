"""미국 ETF·주식 정규장 봉 받기 (yfinance). AGENTS.md 규칙상 트레이딩뷰 MCP는 대화 중 표본 대조에만 쓴다.

- 1시간봉: 최근 약 3년. 그날 일봉 고·저가 밖으로 튄 체결은 일봉 범위로 자른다. 일봉이 비어 있는 날(최근일)은 뺀다.
- 2·4시간봉: 트레이딩뷰처럼 09:30 기준으로 묶는다(2시간 09:30·11:30·13:30·15:30, 4시간 09:30·13:30).
- 일봉: 2022년부터(200일 이동평균 준비).
- 나쁜 날: 1시간봉이 모자라거나(반일장은 3봉 이상이면 정상) 1시간봉으로 묶은 고·저·종가가 일봉과 그날 변동폭의 25% 넘게 다른 날.
  `cache/hist/<종목>_bad.json`에 저장하고, 백테스트에서 그날을 지나는 거래를 뺀다.
KORU는 1시간봉과 일봉의 분할 보정이 20배 어긋나(2026-07-15 분할) 쓰지 않는다 → EWY.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
import yfinance as yf

OUT = pathlib.Path(__file__).resolve().parent.parent / "cache" / "hist"
SYMS = ["SOXL", "SOXX", "EWY"]
HALF = {"2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24"}


def ms(ix) -> int:
    return int(ix.timestamp() * 1000)


def rows_of(df) -> list:
    return [[ms(ix), float(r.o), float(r.h), float(r.l), float(r.c), float(r.v)] for ix, r in df.iterrows()]


def main(syms):
    OUT.mkdir(parents=True, exist_ok=True)
    for sym in syms:
        try:
            one(sym)
        except Exception as e:   # 한 종목이 실패해도 나머지는 받는다
            print(sym, "실패", type(e).__name__, str(e)[:160], flush=True)


def one(sym):
    h = yf.Ticker(sym).history(period="730d", interval="60m", prepost=False, auto_adjust=False, actions=False)
    if h.empty:   # 상장 2년 안팎(ALAB 2024-03 상장)은 period가 상장일부터 잡혀 730일 한도에 걸린다
        h = yf.Ticker(sym).history(start=(pd.Timestamp.today() - pd.Timedelta(days=720)).strftime("%Y-%m-%d"), interval="60m", prepost=False, auto_adjust=False, actions=False)
    d = yf.Ticker(sym).history(start="2022-01-01", interval="1d", auto_adjust=False, actions=False)
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    h = h.rename(columns={"Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"})[["o", "h", "l", "c", "v"]]
    d = d.rename(columns={"Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"})[["o", "h", "l", "c", "v"]]
    day = pd.Series(h.index.date, index=h.index)
    dmap = {ix.date(): r for ix, r in d.iterrows()}
    keep = day.map(lambda x: x in dmap)
    h = h[keep.values].copy()
    day = day[keep.values]
    dh = day.map(lambda x: dmap[x].h).values
    dl = day.map(lambda x: dmap[x].l).values
    clipped = int(((h["h"] > dh) | (h["l"] < dl)).sum())
    h["h"] = h["h"].clip(upper=dh)
    h["l"] = h["l"].clip(lower=dl)
    h["h"] = h[["h", "o", "c"]].max(axis=1)
    h["l"] = h[["l", "o", "c"]].min(axis=1)
    # 나쁜 날
    g = h.groupby(day.values).agg(hi=("h", "max"), lo=("l", "min"), c=("c", "last"), n=("c", "size"))
    bad = []
    for dt, r in g.iterrows():
        dr = dmap[dt]
        rng = max(dr.h - dr.l, 1e-9)
        mis = max(abs(r.hi - dr.h), abs(r.lo - dr.l), abs(r.c - dr.c)) / rng
        need = 3 if str(dt) in HALF else 7
        if r.n < need or mis > 0.25:
            bad.append(str(dt))
    # 2·4시간봉: 09:30 기준 묶음
    mins = h.index.hour * 60 + h.index.minute - 570
    out = {"1h": rows_of(h), "1d": rows_of(d)}
    for size, iv in ((120, "2h"), (240, "4h")):
        key = [f"{dt}_{m // size}" for dt, m in zip(day.values, mins)]
        gg = h.assign(k=key, t=h.index).groupby("k", sort=False).agg(t=("t", "first"), o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"))
        out[iv] = [[ms(r.t), float(r.o), float(r.h), float(r.l), float(r.c), float(r.v)] for _, r in gg.iterrows()]
    for iv, rows in out.items():
        (OUT / f"{sym}_{iv}.json").write_text(json.dumps(rows))
    (OUT / f"{sym}_bad.json").write_text(json.dumps(bad))
    print(sym, {iv: len(r) for iv, r in out.items()}, "잘라 낸 1시간봉", clipped, "나쁜 날", bad,
          "1h", h.index[0].date(), "~", h.index[-1].date())


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else SYMS)
