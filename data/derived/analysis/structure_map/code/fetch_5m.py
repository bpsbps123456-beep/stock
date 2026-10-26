"""최근 60일 5분봉으로 40분·160분봉 만들기 (2026-10-23 사용자 "최근 60일로 검증해 보면 되지").

미국: yfinance 5분봉 정규장은 최근 60일까지만 받을 수 있다. 그날 일봉(fetch_us.py가 받은 `<종목>_1d.json`) 고·저가 밖으로 튄 체결은
  일봉 범위로 자르고, 일봉이 없는 날은 뺀다. 묶음은 트레이딩뷰처럼 09:30 기준이다.
  - 40분 09:30·10:10·…·15:30, 160분 09:30·12:10·14:50
  - 점검용 `1hc`·`4hc`: 같은 60일로만 만든 1·4시간봉. 백테스트에서 이력이 짧은 탓(초기 구간)을 재는 데 쓴다.
  나쁜 날: 5분봉이 70개 미만이거나, 5분봉으로 묶은 고·저·종가가 일봉과 그날 변동폭의 25% 넘게 다른 날 → `<종목>_bad5m.json`.
코인: 바이낸스 5분봉(fetch_hist.py)을 UTC 00:00 기준으로 묶는다(40분 하루 36봉, 160분 9봉). 덜 찬 마지막 묶음은 뺀다.
사용: py code/fetch_5m.py us SOXL,SOXX,...  /  py code/fetch_5m.py crypto
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import pandas as pd
import yfinance as yf

HIST = pathlib.Path(__file__).resolve().parent.parent / "cache" / "hist"
ET = "America/New_York"
US_SIZES = {"40m": 40, "1hc": 60, "160m": 160, "4hc": 240}
CR_MS = {"40m": 2_400_000, "160m": 9_600_000}


def us(sym):
    d = json.loads((HIST / f"{sym}_1d.json").read_text())
    dmap = {pd.Timestamp(r[0], unit="ms", tz="UTC").tz_convert(ET).date(): r for r in d}
    h = None
    for k in range(3):
        try:
            h = yf.Ticker(sym).history(period="60d", interval="5m", prepost=False, auto_adjust=False, actions=False)
            break
        except Exception as e:  # 잠깐 막히면 쉬었다 다시
            print(sym, "다시 시도", type(e).__name__, flush=True)
            time.sleep(10 * (k + 1))
    if h is None or h.empty:
        print(sym, "5분봉 없음", flush=True)
        return
    h = h.rename(columns={"Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"})[["o", "h", "l", "c", "v"]].dropna()
    day = pd.Series(h.index.date, index=h.index)
    keep = day.map(lambda x: x in dmap).values
    h, day = h[keep].copy(), day[keep]
    dh = day.map(lambda x: dmap[x][2]).values
    dl = day.map(lambda x: dmap[x][3]).values
    clipped = int(((h["h"] > dh) | (h["l"] < dl)).sum())
    h["h"] = h["h"].clip(upper=dh)
    h["l"] = h["l"].clip(lower=dl)
    h["h"] = h[["h", "o", "c"]].max(axis=1)
    h["l"] = h[["l", "o", "c"]].min(axis=1)
    g = h.groupby(day.values).agg(hi=("h", "max"), lo=("l", "min"), c=("c", "last"), n=("c", "size"))
    bad = []
    for dt_, r in g.iterrows():
        dr = dmap[dt_]
        rng = max(dr[2] - dr[3], 1e-9)
        mis = max(abs(r.hi - dr[2]), abs(r.lo - dr[3]), abs(r.c - dr[4])) / rng
        if r.n < 70 or mis > 0.25:
            bad.append(str(dt_))
    out = {"5m": [[int(ix.timestamp() * 1000), float(r.o), float(r.h), float(r.l), float(r.c), float(r.v)] for ix, r in h.iterrows()]}
    mins = (h.index.hour * 60 + h.index.minute - 570).values
    base = h.index.normalize()
    for iv, size in US_SIZES.items():
        bk = mins // size
        start = base + pd.to_timedelta(570 + bk * size, unit="m")
        gg = h.assign(s=start).groupby("s", sort=True).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"))
        out[iv] = [[int(s.timestamp() * 1000), float(r.o), float(r.h), float(r.l), float(r.c), float(r.v)] for s, r in gg.iterrows()]
    for iv, rows in out.items():
        (HIST / f"{sym}_{iv}.json").write_text(json.dumps(rows))
    (HIST / f"{sym}_bad5m.json").write_text(json.dumps(bad))
    print(sym, {iv: len(r) for iv, r in out.items()}, "잘라 낸 5분봉", clipped, "나쁜 날", bad, h.index[0].date(), "~", h.index[-1].date(), flush=True)


def resample_us(sym, sizes):
    """이미 받은 미국 5분봉(`<종목>_5m.json`)을 09:30 기준 n분봉으로 다시 묶는다(이웃 시간봉 점검용, 내려받기 없음)."""
    rows = json.loads((HIST / f"{sym}_5m.json").read_text())
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    ix = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET)
    mins = (ix.dt.hour * 60 + ix.dt.minute - 570).values
    base = ix.dt.normalize()
    for iv, size in sizes.items():
        df["s"] = base + pd.to_timedelta(570 + (mins // size) * size, unit="m")
        gg = df.groupby("s", sort=True).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"))
        (HIST / f"{sym}_{iv}.json").write_text(json.dumps([[int(s.timestamp() * 1000), float(r.o), float(r.h), float(r.l), float(r.c), float(r.v)] for s, r in gg.iterrows()]))


def crypto(sym):
    rows = json.loads((HIST / f"{sym}_5m.json").read_text())
    for iv, ms in CR_MS.items():
        need = ms // 300_000
        agg, cnt = {}, {}
        for t, o, hi, lo, c, v in rows:
            s = t - t % ms
            if s not in agg:
                agg[s] = [s, o, hi, lo, c, v]
                cnt[s] = 1
            else:
                a = agg[s]
                a[2], a[3], a[4], a[5] = max(a[2], hi), min(a[3], lo), c, a[5] + v
                cnt[s] += 1
        keys = sorted(agg)
        if keys and cnt[keys[-1]] < need:
            keys = keys[:-1]
        (HIST / f"{sym}_{iv}.json").write_text(json.dumps([agg[k] for k in keys]))
        print(sym, iv, len(keys), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "crypto":
        for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            crypto(s)
    elif sys.argv[1] == "us-resample":     # py code/fetch_5m.py us-resample SYM,... 20,30,45,50
        sizes = {f"{m}m": int(m) for m in sys.argv[3].split(",")}
        for s in sys.argv[2].split(","):
            resample_us(s, sizes)
        print("다시 묶음", list(sizes), len(sys.argv[2].split(",")), "종목")
    else:
        for s in sys.argv[2].split(","):
            try:
                us(s)
            except Exception as e:
                print(s, "실패", type(e).__name__, str(e)[:160], flush=True)
