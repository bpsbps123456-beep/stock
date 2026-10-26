"""FINRA 일별 공매도 거래량(Reg SHO) 수집 — 헤지펀드·옵션 연구(2026-10-14)의 장외 거래 흐름 지표.

출처: cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt (TRF·ADF 합본, 2018-08-01부터 있다).
  칸: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market, 마지막 줄은 행 수.
  - FINRA 장외 시설(TRF·ADF)에 보고된 정규장 거래만 센다. 거래소 체결은 빠진다 → 전체 거래량의 일부다.
  - ShortVolume은 면제분(ShortExempt)을 포함하고, 고객 매수를 받아 주는 시장조성자 공매도도 들어간다
    → 비율이 높다고 곧 약세 베팅은 아니다.
  - FINRA는 거래일 D의 파일을 D 오후 6시(ET)까지 올린다. 드물게 다음 날 고친다(Last-Modified를 남긴다).
날짜 규칙: asof = D(측정일), known = D 다음 거래일(그날 4시 종가 때는 공개돼 있다) → 분석은 known 다음 날 시가.
  D+1 시가 진입 변형은 asof로 따로 시험한다(open_ok = 1인 날만: 파일이 D+1 9:30 전에 올라왔다).
  지금 받은 파일이 늦게 고쳐 올린 것이면(Last-Modified가 D+1 16시 뒤) 고친 날 종가로 known을 늦춘다.
  그러면 known이 다음 측정과 겹쳐 그 asof는 빠진다(2023-02-13, 2025-08-20, 2026-02-05).

실행: py fetch_finra_shvol.py [--since 2024-01-02] [--until YYYY-MM-DD] [--gap 1.0] [--build-only]
  이어 받기: cache/raw_finra_shvol/<날짜>.parquet가 있으면 건너뛴다. 받기가 끝나면 늘 결과 표를 다시 만든다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RAW = CACHE / "raw_finra_shvol"
REPO = HERE.parents[4]
ET = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
      "Accept": "text/plain, */*"}
URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"
ETFS = ["QQQ", "SPY", "SOXX", "SMH", "TQQQ", "SQQQ", "SOXL", "SOXS"]
# 옛 종목코드 → 지금 코드. 에버퓨어(P)는 2026-04-16까지 PSTG(퓨어스토리지)였다(파일에서 겹치는 날 없이 바뀜을 확인).
ALIAS = {"PSTG": "P"}
CAL_START = "2018-08-01"  # 거래일 달력은 FINRA 합본 시작일부터 한 번에 받는다
# 달력 끝(아직 없는 다음 거래일)을 정할 때만 쓰는 NYSE 전일 휴장일
NYSE_HOLIDAYS = {"2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
                 "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"}


def ai_symbols() -> list[str]:
    sys.path.insert(0, str(REPO))
    import watchlist
    return sorted({s for subs in watchlist.AI_UNIVERSE.values() for lst in subs.values() for s in lst if "." not in s})


def _get(url: str, timeout: int = 60) -> tuple[bytes, str] | None:
    """(본문, Last-Modified). 403·404(휴장, 아직 안 올라온 날)는 None. 429는 바로 올려 멈춘다. 그 밖은 두 번 더 시도."""
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read(), r.headers.get("Last-Modified", "")
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if e.code == 429 or k == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if k == 2:
                raise
        time.sleep(30 * (k + 1))
    return None


def trading_days() -> list[str]:
    """QQQ 일봉 날짜. 오늘 장이 아직 안 끝났으면(ET 16시 전) 마지막 줄을 뺀다."""
    import yfinance as yf
    h = yf.Ticker("QQQ").history(start=CAL_START, auto_adjust=False)
    days = [d.strftime("%Y-%m-%d") for d in h.index]
    now = dt.datetime.now(ET)
    if days and days[-1] == now.strftime("%Y-%m-%d") and now.hour < 16:
        days = days[:-1]
    return days


def extend_days(days: list[str], n: int = 30) -> list[str]:
    """달력 뒤에 아직 안 온 거래일 n개(주말·NYSE 휴장일을 건너뛴 평일)를 붙인다."""
    ext, d = list(days), dt.date.fromisoformat(days[-1])
    while len(ext) < len(days) + n:
        d += dt.timedelta(days=1)
        if d.weekday() < 5 and d.isoformat() not in NYSE_HOLIDAYS:
            ext.append(d.isoformat())
    return ext


def _at(day: str, h: int, m: int = 0) -> dt.datetime:
    return dt.datetime.combine(dt.date.fromisoformat(day), dt.time(h, m), tzinfo=ET)


def known_and_open(day: str, last_mod: str, ext: list[str], pos: dict[str, int]) -> tuple[str, float]:
    """known = D 다음 거래일부터, 지금 가진 파일의 최종 수정 시각이 그날 16시(ET) 전인 첫 거래일.
    늦게 고쳐 올린 파일은 고친 값이 그때야 공개된 것으로 본다(보수적). open_ok = D+1 9:30 전에 올라왔으면 1."""
    i = pos[day] + 1
    if not last_mod:
        return ext[i], float("nan")
    lm = parsedate_to_datetime(last_mod).astimezone(ET)
    open_ok = 1.0 if lm <= _at(ext[i], 9, 30) else 0.0
    while i < len(ext) - 1 and _at(ext[i], 16) < lm:
        i += 1
    return ext[i], open_ok


def fetch_day(day: str) -> bool:
    """하루 파일을 받아 전 종목 symbol·short·exempt·total만 남긴다. 게시 시각(Last-Modified)은 파일 메타에 둔다."""
    out = RAW / f"{day}.parquet"
    if out.exists():
        return True
    got = _get(URL.format(d=day.replace("-", "")))
    if got is None:
        return False
    body, last_mod = got
    df = pd.read_csv(io.BytesIO(body), sep="|", dtype=str, keep_default_na=False)
    tail = df[df["Date"] != day.replace("-", "")]
    df = df[df["Date"] == day.replace("-", "")]
    # 마지막 줄(행 수)과 맞춰 본다. 다른 날짜가 섞이거나 행 수가 다르면 멈춘다.
    if len(tail) != 1 or not tail["Date"].iloc[0].isdigit() or int(tail["Date"].iloc[0]) != len(df):
        raise ValueError(f"FINRA {day}: 끝줄·행 수 불일치 ({tail['Date'].tolist()[:3]}, 행 {len(df)})")
    if df["Symbol"].duplicated().any():
        raise ValueError(f"FINRA {day}: 같은 종목이 두 번 나온다")
    # 소수점 주식(분할 주문) 몫은 반올림해 정수로 둔다 — 비율에는 영향이 없고 하루 파일이 260KB → 170KB로 준다.
    t = pd.DataFrame({"symbol": df["Symbol"].to_numpy(),
                      **{k: pd.to_numeric(df[c]).round().astype("int64").to_numpy()
                         for k, c in (("short", "ShortVolume"), ("exempt", "ShortExemptVolume"), ("total", "TotalVolume"))}})
    tb = pa.Table.from_pandas(t, preserve_index=False)
    tb = tb.replace_schema_metadata({**(tb.schema.metadata or {}), b"last_modified": last_mod.encode(),
                                     b"fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").encode()})
    tmp = out.with_suffix(".tmp")
    pq.write_table(tb, tmp, compression="zstd", compression_level=9)
    tmp.replace(out)
    return True


def load_raw(symbols: list[str] | None = None) -> pd.DataFrame:
    """긴 표: asof, symbol, short, exempt, total. symbols를 주면 그 종목만(다른 묶음을 뒤에 만들 때 쓴다)."""
    parts = []
    for f in sorted(RAW.glob("*.parquet")):
        d = pq.read_table(f).to_pandas()
        if symbols is not None:
            d = d[d["symbol"].isin(symbols)]
        d.insert(0, "asof", f.stem)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def post_lag_hours(day: str, last_mod: str) -> float:
    """D 16:00 ET부터 Last-Modified까지 시간. 6시 마감이면 2 이하. 나중에 고친 파일은 크게 나온다."""
    if not last_mod:
        return float("nan")
    lm = parsedate_to_datetime(last_mod).astimezone(ET)
    return (lm - _at(day, 16)).total_seconds() / 3600


def build(days: list[str]) -> None:
    ai = ai_symbols()
    keep = set(ai) | set(ETFS)
    ext = extend_days(days)
    pos = {d: i for i, d in enumerate(ext)}
    rows, by_sym = [], []
    for f in sorted(RAW.glob("*.parquet")):
        day = f.stem
        if day not in pos:
            print(f"달력에 없는 날 건너뜀: {day}")
            continue
        t = pq.read_table(f)
        d = t.to_pandas()
        for old, new in ALIAS.items():
            if not (d["symbol"] == new).any():
                d.loc[d["symbol"] == old, "symbol"] = new
        lm = (t.schema.metadata or {}).get(b"last_modified", b"").decode()
        known, open_ok = known_and_open(day, lm, ext, pos)
        r = {"known": known, "asof": day, "post_lag_h": post_lag_hours(day, lm), "open_ok": open_ok,
             "tot_all": d["total"].sum(), "short_all": d["short"].sum(), "n_sym": float(len(d))}
        r["shvr_all"] = r["short_all"] / r["tot_all"]
        a = d[d["symbol"].isin(ai) & (d["total"] > 0)]
        r["tot_ai"], r["n_ai"] = a["total"].sum(), float(len(a))
        r["shvr_ai"] = a["short"].sum() / r["tot_ai"]
        r["shvr_ai_eq"] = (a["short"] / a["total"]).mean()  # 종목 같은 무게
        e = d.set_index("symbol")
        for s in ETFS:
            ok = s in e.index and e.at[s, "total"] > 0
            r[f"shvr_{s.lower()}"] = e.at[s, "short"] / e.at[s, "total"] if ok else float("nan")
        rows.append(r)
        k = d[d["symbol"].isin(keep)].copy()
        k.insert(0, "asof", day)
        k.insert(0, "known", known)
        by_sym.append(k)
    out = pd.DataFrame(rows).sort_values(["known", "asof"])
    late = out[out.duplicated("known", keep="last")]
    out = out.drop_duplicates("known", keep="last").set_index("known")  # 같은 known이면 나중 측정만
    cols = ["asof", "shvr_all", "shvr_ai", "shvr_ai_eq"] + [f"shvr_{s.lower()}" for s in ETFS] + \
           ["tot_all", "short_all", "tot_ai", "n_ai", "n_sym", "post_lag_h", "open_ok"]
    out = out[cols]
    out[cols[1:]] = out[cols[1:]].astype("float64")
    out.to_parquet(CACHE / "finra_shvol.parquet")
    bs = pd.concat(by_sym, ignore_index=True)[["known", "asof", "symbol", "short", "exempt", "total"]]
    bs[["short", "exempt", "total"]] = bs[["short", "exempt", "total"]].astype("float64")
    bs = bs.sort_values(["known", "asof"]).drop_duplicates(["known", "symbol"], keep="last")
    bs.sort_values(["known", "symbol"]).reset_index(drop=True).to_parquet(CACHE / "finra_shvol_by_symbol.parquet", index=False)
    print(f"표 {len(out)}일 {out.index[0]}~{out.index[-1]}, 종목별 {len(bs)}행, "
          f"늦게 고쳐져 뒤 측정에 밀린 asof {late['asof'].tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2024-01-02")
    ap.add_argument("--until", default=None, help="이 날짜까지만(앞 구간을 따로 받을 때)")
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    days = trading_days()
    if not a.build_only:
        todo = [d for d in days if d >= a.since and (a.until is None or d <= a.until)]
        log = open(RAW / "_fetch.log", "a", encoding="utf-8")
        miss = []
        for i, d in enumerate(todo):
            fresh = not (RAW / f"{d}.parquet").exists()
            try:
                ok = fetch_day(d)
            except urllib.error.HTTPError as e:
                print(f"{time.strftime('%H:%M:%S')} {d} HTTP {e.code} — 멈춤", file=log, flush=True)
                raise
            if not ok:
                miss.append(d)
            if fresh:
                time.sleep(a.gap)
            if i % 25 == 0 or i == len(todo) - 1:
                print(f"{time.strftime('%H:%M:%S')} {i + 1}/{len(todo)} {d} 빠짐 {len(miss)}", file=log, flush=True)
        print(f"빠진 날 {miss}", file=log, flush=True)
    build(days)


if __name__ == "__main__":
    main()
