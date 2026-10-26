"""미리보기(부록 9) 3단계 ①: 가격 받기.

yfinance 일봉 2026-01-01 ~ 2026-10-17(auto_adjust=False, 배당·분할 열 포함):
  KRX  ^KS11, 000660.KS, 005930.KS, ^KQ11
  NYSE SOXX, ^SOX, QQQ, EWY, ^VIX
  CME  NQ=F
정리: 시가·고가·저가·종가가 모두 빈 줄을 뺀다. KRX·NYSE 종목은 로더 달력(preview/results/loader_calendar.parquet,
yfinance ^KS11·^GSPC 날짜)에 없는 날짜 줄(휴장 빈 줄)을 뺀다. 달력에 있는데 봉이 없는 날은 결측으로 적는다.
NQ=F는 CME 세션 날짜 그대로 둔다(달력 정리 없음).

보충(동결 §2.3 미국 M0 'NQ=F 직전 미국 종가 → 창 마감'은 일봉으로 잴 수 없다): NQ=F 60분봉을 따로 받아
prices_nq_1h.parquet에 둔다. 이 값은 M0 한 열(m0_nq_close_to_winclose)에만 쓴다.

산출: preview/results/prices.parquet(긴 표), prices_nq_1h.parquet, prices_meta.json
실행: py prices.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIEW = HERE.parent
RES = PREVIEW / "results"

START, END = date(2026, 1, 1), date(2026, 10, 17)
TICKERS = {
    "^KS11": "krx", "000660.KS": "krx", "005930.KS": "krx", "^KQ11": "krx",
    "SOXX": "nyse", "^SOX": "nyse", "QQQ": "nyse", "EWY": "nyse", "^VIX": "nyse",
    "NQ=F": "cme",
}
VOLUME_MEANINGFUL = {"000660.KS", "005930.KS", "SOXX", "QQQ", "EWY", "^KS11", "^KQ11"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch_daily(t: str) -> pd.DataFrame:
    import yfinance as yf
    h = yf.Ticker(t).history(start=START.isoformat(), end=(pd.Timestamp(END) + pd.Timedelta(days=1)).date().isoformat(),
                             interval="1d", auto_adjust=False, actions=True)
    h = h.copy()
    h.index = [pd.Timestamp(x).date() for x in h.index]
    h = h[~pd.Index(h.index).duplicated(keep="last")]
    h = h[[d <= END for d in h.index]]
    cols = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close",
            "Volume": "volume", "Dividends": "dividends", "Stock Splits": "splits"}
    h = h.rename(columns=cols)
    for c in cols.values():
        if c not in h:
            h[c] = np.nan
    h = h[list(cols.values())]
    h.index.name = "date"
    return h.reset_index()


def fetch_nq_1h() -> pd.DataFrame:
    import yfinance as yf
    h = yf.Ticker("NQ=F").history(start=START.isoformat(),
                                  end=(pd.Timestamp(END) + pd.Timedelta(days=1)).date().isoformat(),
                                  interval="60m", auto_adjust=False)
    idx = pd.DatetimeIndex(h.index)
    if idx.tz is None:
        raise RuntimeError("NQ=F 60분봉에 시간대가 없다")
    et = idx.tz_convert("America/New_York")
    out = pd.DataFrame({"bar_start_et": et.tz_localize(None), "bar_start_utc": idx.tz_convert("UTC").tz_localize(None),
                        "open": h["Open"].to_numpy(float), "high": h["High"].to_numpy(float),
                        "low": h["Low"].to_numpy(float), "close": h["Close"].to_numpy(float),
                        "volume": h["Volume"].to_numpy(float)})
    out["bar_end_et"] = out["bar_start_et"] + pd.Timedelta(hours=1)
    out = out.dropna(subset=["close"]).drop_duplicates("bar_start_utc", keep="last").sort_values("bar_start_utc")
    return out.reset_index(drop=True)


def main() -> int:
    cal = pd.read_parquet(RES / "loader_calendar.parquet")
    days = {"krx": {d.date() for d in cal.loc[cal["krx"], "date"] if START <= d.date() <= END},
            "nyse": {d.date() for d in cal.loc[cal["nyse"], "date"] if START <= d.date() <= END}}
    parts, meta = [], {"made": datetime.now().replace(microsecond=0).isoformat(), "start": str(START), "end": str(END),
                       "source": "yfinance Ticker.history(interval='1d', auto_adjust=False, actions=True)",
                       "calendar": "preview/results/loader_calendar.parquet (yfinance ^KS11·^GSPC 날짜, 로더와 같음)",
                       "tickers": {}}
    for t, mk in TICKERS.items():
        h = fetch_daily(t)
        n_raw = len(h)
        allnan = h[["open", "high", "low", "close"]].isna().all(axis=1)
        h = h[~allnan]
        info = {"market": mk, "rows_raw": int(n_raw), "rows_all_nan_dropped": int(allnan.sum())}
        if mk in days:
            off = ~h["date"].isin(days[mk])
            info["rows_off_calendar_dropped"] = [str(d) for d in h.loc[off, "date"]]
            h = h[~off]
            miss = sorted(days[mk] - set(h["date"]))
            info["calendar_days"] = len(days[mk])
            info["calendar_days_without_bar"] = [str(d) for d in miss]
        h = h.copy()
        h["ticker"], h["market"] = t, mk
        bad_open = h["open"].isna() | (h["open"] <= 0)
        flat = (h["open"] == h["high"]) & (h["high"] == h["low"]) & (h["low"] == h["close"])
        hl_bad = h["high"] < h["low"]
        h["flag_open_bad"] = bad_open
        h["flag_flat_bar"] = flat
        h["flag_high_lt_low"] = hl_bad
        h["flag_zero_volume"] = (h["volume"].fillna(0) == 0) if t in VOLUME_MEANINGFUL else False
        info.update({"rows": int(len(h)), "first": str(h["date"].min()), "last": str(h["date"].max()),
                     "open_bad": [str(d) for d in h.loc[bad_open, "date"]],
                     "flat_bar": [str(d) for d in h.loc[flat, "date"]],
                     "high_lt_low": [str(d) for d in h.loc[hl_bad, "date"]],
                     "zero_volume": [str(d) for d in h.loc[h["flag_zero_volume"], "date"]],
                     "dividend_days": {str(d): float(v) for d, v in zip(h["date"], h["dividends"]) if v and v > 0},
                     "split_days": {str(d): float(v) for d, v in zip(h["date"], h["splits"]) if v and v > 0}})
        meta["tickers"][t] = info
        parts.append(h)
    px = pd.concat(parts, ignore_index=True)
    px["date"] = pd.to_datetime(px["date"])
    px = px[["ticker", "market", "date", "open", "high", "low", "close", "adj_close", "volume", "dividends", "splits",
             "flag_open_bad", "flag_flat_bar", "flag_high_lt_low", "flag_zero_volume"]]
    px.to_parquet(RES / "prices.parquet", index=False)
    nq = fetch_nq_1h()
    nq.to_parquet(RES / "prices_nq_1h.parquet", index=False)
    meta["nq_1h"] = {"rows": int(len(nq)), "first_start_et": str(nq["bar_start_et"].min()),
                     "last_start_et": str(nq["bar_start_et"].max()),
                     "use": "m0_nq_close_to_winclose 한 열만. 바 끝(시작 + 1시간)이 목표 시각 이하인 마지막 바의 종가"}
    meta["files_sha256"] = {"prices.parquet": sha(RES / "prices.parquet"),
                            "prices_nq_1h.parquet": sha(RES / "prices_nq_1h.parquet")}
    (RES / "prices_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({t: {k: v for k, v in i.items() if k in ("rows", "first", "last", "calendar_days_without_bar",
                                                               "rows_off_calendar_dropped", "open_bad", "flat_bar",
                                                               "zero_volume", "high_lt_low")}
                      for t, i in meta["tickers"].items()}, ensure_ascii=False, indent=1))
    print(json.dumps(meta["nq_1h"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
