"""헤지펀드·옵션 연구(2026-10-14) 스퀴즈메트릭스 DIX·GEX 수집. 분석 코드는 cache/squeeze.parquet를 읽는다.

출처(무료 공개): https://squeezemetrics.com/monitor/static/DIX.csv (칸 date, price, dix, gex. 2011-05-02~)
  - dix: S&P 500 구성 종목의 다크풀(FINRA 장외 보고) 공매도 거래량 비율을 금액 가중한 값. 회사 주장은 높을수록 기관 매수.
    재료는 FINRA 일별 공매도 거래량 파일(거래일 저녁 6시 ET 전후 공개)이다(사이트 /monitor/docs의 finra_short.csv 예시).
  - gex: S&P 500 옵션의 딜러 감마 노출 추정(지수 1% 움직임당 달러). 양수 = 딜러 롱감마(변동 완화), 음수 = 증폭.
  - price: 그날 S&P 500 종가.
공개 시점: 2026-10-13 행이 든 파일의 Last-Modified가 2026-10-13 21:56 UTC(=17:56 ET)였다. 웨이백 사본 24개(19:17~19:55 ET)의
  digest가 지금 파일을 그 날짜까지 자른 바이트와 모두 같았다 = 그날 저녁 공개·과거 행 소급 수정 없음(2025-04-29 이후).
날짜 규칙(검증 뒤 고침): 값은 측정일 D 장 마감(16시) 뒤에 나오므로 D 종가 때는 공개 전이다.
  asof = D, known = D 다음 거래일(그날 16시에는 공개돼 있다) → 분석은 known 다음 날 시가. D+1 시가 진입 변형은 asof로 따로 시험한다
  (FINRA 공매도 거래량 갈래 fetch_finra_shvol.py와 같은 규칙. DIX 재료가 그 FINRA 파일이다).

실행: py fetch_squeeze.py [--wayback]  (이어 받기: 오늘(KST) 날짜 원본 CSV·웨이백 목록이 있으면 다시 받지 않는다)
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RAW = CACHE / "raw_squeeze"
URL = "https://squeezemetrics.com/monitor/static/DIX.csv"
CDX = ("https://web.archive.org/cdx/search/cdx?url=squeezemetrics.com/monitor/static/DIX.csv"
       "&output=json&fl=timestamp,statuscode,digest,length&filter=statuscode:200&from=2023")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
      "Accept": "text/csv, application/json, */*"}
ET = ZoneInfo("America/New_York")
LAG = 1  # 측정일 → known 거래일 차이. 그날 저녁(16시 뒤) 공개라 D 종가에는 모른다 → 1
# 달력 끝(아직 없는 다음 거래일)을 정할 때만 쓰는 NYSE 전일 휴장일(fetch_finra_shvol.py와 같은 목록)
NYSE_HOLIDAYS = {"2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
                 "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"}


def _get(url: str, timeout: int = 60) -> tuple[bytes, dict]:
    """429는 바로 멈춘다. 그 밖의 연결 오류는 두 번 더 시도한 뒤 올린다."""
    for k in range(3):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)
            return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code == 429 or k == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if k == 2:
                raise
        time.sleep(30 * (k + 1))
    raise RuntimeError("도달하지 않음")


def fetch_csv(today: str) -> Path:
    """오늘 날짜 이름으로 원본과 응답 머리(Last-Modified·ETag)를 남긴다."""
    out = RAW / f"DIX_{today}.csv"
    if out.exists():
        return out
    b, h = _get(URL)
    pd.read_csv(io.BytesIO(b), nrows=5)  # 깨진 응답이면 여기서 멈춘다
    out.write_bytes(b)
    meta = {"url": URL, "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "last_modified": h.get("Last-Modified"), "etag": h.get("ETag"), "bytes": len(b)}
    (RAW / f"DIX_{today}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out


def last_session(days: list[str]) -> str:
    """마지막 '끝난' 미국 거래일. 오늘(ET) 장이 아직 안 끝났으면 그 전날."""
    now = dt.datetime.now(ET)
    today = now.strftime("%Y-%m-%d")
    done = [d for d in days if d < today or (d == today and now.hour >= 16)]
    return done[-1]


def calendar() -> tuple[list[str], pd.Series]:
    """거래일 = yfinance QQQ 일봉 날짜. 가격 칸 검증용 ^GSPC 종가도 함께."""
    import yfinance as yf
    q = yf.Ticker("QQQ").history(start="2011-01-01", auto_adjust=False)
    time.sleep(1)
    g = yf.Ticker("^GSPC").history(start="2011-01-01", auto_adjust=False)["Close"]
    g.index = g.index.strftime("%Y-%m-%d")
    return sorted(set(q.index.strftime("%Y-%m-%d"))), g[~g.index.duplicated(keep="last")]


def load_raw(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, float_precision="round_trip")  # 원본 소수 문자열과 끝자리까지 같게
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    return d.drop_duplicates("date", keep="last").set_index("date").sort_index().astype(float)


def known_days(days: list[str], last: str) -> list[str]:
    """끝난 거래일 달력 뒤에 '다음 거래일' LAG개를 붙인다(주말·NYSE 휴장일 건너뜀). 마지막 측정일의 known을 정하려고."""
    cal = [d for d in days if d <= last]
    d = dt.date.fromisoformat(cal[-1])
    for _ in range(LAG):
        d += dt.timedelta(days=1)
        while d.weekday() >= 5 or d.isoformat() in NYSE_HOLIDAYS:
            d += dt.timedelta(days=1)
        cal.append(d.isoformat())
    return cal


def build(raw: pd.DataFrame, days: list[str], last: str) -> tuple[pd.DataFrame, dict]:
    """측정일(asof) → LAG 거래일 뒤 known. 거래일 밖 날짜·마지막 끝난 거래일 이후 측정 행은 버리고 개수를 남긴다.
    마지막 측정일의 known은 아직 오지 않은 다음 거래일일 수 있다(그날 종가 때 공개돼 있음, 분석 달력 밖이면 자연히 안 쓰인다)."""
    cal = known_days(days, last)
    pos = {d: i for i, d in enumerate(cal)}
    off = [d for d in raw.index if d <= last and d not in pos]
    r = raw[raw.index.isin(pos) & (raw.index <= last)]
    known = [cal[pos[d] + LAG] for d in r.index]
    out = pd.DataFrame({"asof": r.index, "dix": r["dix"].values, "gex": r["gex"].values,
                        "spx": r["price"].values, "gex_bn": r["gex"].values / 1e9}, index=pd.Index(known, name="known"))
    out = out[~out.index.duplicated(keep="last")].sort_index()
    gaps = [d for d in days if raw.index[0] <= d <= raw.index[-1] and d not in raw.index]
    return out, {"off_calendar": off, "missing_trading_days": gaps, "dropped_after_last": int((raw.index > last).sum())}


def wayback(raw_path: Path) -> dict:
    """웨이백 요청은 목록(CDX) 1회뿐. 사본을 받지 않고, 각 사본의 SHA-1 digest를
    '지금 받은 CSV를 그 사본 날짜(ET)까지 자른 바이트'의 SHA-1과 맞춘다. 같으면
    ① 그날 저녁 사본에 그날 행이 이미 있었고 ② 그 뒤로 과거 행이 한 바이트도 안 바뀌었다는 뜻이다.
    (2026-10-14 사본 직접 받기는 웨이백 HTTP 429로 멈췄다. 이 방식은 추가 요청이 없다.)"""
    cdx_f = RAW / "wayback_cdx.json"
    if not cdx_f.exists():
        b, _ = _get(CDX, timeout=90)
        cdx_f.write_bytes(b)
    b = raw_path.read_bytes()
    lines = b.split(b"\n")  # 줄 끝은 CRLF. \r은 각 줄에 남겨 원본 바이트를 그대로 되살린다
    head, body = lines[0], [x for x in lines[1:] if x]
    caps = []
    for ts, _st, digest, _len in json.loads(cdx_f.read_text(encoding="utf-8"))[1:]:
        t = dt.datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc).astimezone(ET)
        day = t.strftime("%Y-%m-%d")
        keep = [x for x in body if x.split(b",")[0].decode() <= day]
        tries = {"same_day": keep, "prev_day_only": keep[:-1]}  # 그날 행 포함 / 전날까지만
        hit = [k for k, rows in tries.items()
               if base64.b32encode(hashlib.sha1(b"\n".join([head] + rows) + b"\n").digest()).decode() == digest]
        caps.append({"ts_utc": ts, "et": t.strftime("%a %Y-%m-%d %H:%M"), "match": hit[0] if hit else "none",
                     "last_row_in_capture": keep[-1][:10].decode() if hit == ["same_day"] else None})
    n = {k: sum(c["match"] == k for c in caps) for k in ("same_day", "prev_day_only", "none")}
    return {"n_captures_since_2023": len(caps), "match_counts": n,
            "et_time_range": [min(c["et"][-5:] for c in caps), max(c["et"][-5:] for c in caps)], "captures": caps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wayback", action="store_true", help="웨이백 사본 digest와 비교(목록 요청 1회, 캐시)")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    raw_path = fetch_csv(today)
    raw = load_raw(raw_path)
    days, gspc = calendar()
    last = last_session(days)
    out, gaps = build(raw, days, last)
    out.to_parquet(CACHE / "squeeze.parquet")
    both = out.set_index("asof").index.intersection(gspc.index)
    px = (out.set_index("asof").loc[both, "spx"] / gspc.loc[both] - 1).abs()
    qa = {"raw_rows": len(raw), "raw_first": raw.index[0], "raw_last": raw.index[-1], "last_session": last,
          "rows": len(out), "first_known": out.index[0], "last_known": out.index[-1], "last_asof": out["asof"].iloc[-1],
          "lag_trading_days": LAG, "known_after_last_session": int((out.index > last).sum()),
          "nan": out[["dix", "gex", "spx"]].isna().sum().to_dict(), "spx_vs_gspc_max_rel": float(px.max()),
          "spx_vs_gspc_n_over_0.1pct": int((px > 1e-3).sum()), "spx_mismatch_dates": px[px > 1e-3].index[:10].tolist(),
          **{k: (v if k == "dropped_after_last" else {"n": len(v), "dates": v[:20]}) for k, v in gaps.items()}}
    if a.wayback:
        qa["wayback"] = wayback(raw_path)
    (RAW / f"qa_{today}.json").write_text(json.dumps(qa, indent=1, default=str), encoding="utf-8")
    print(json.dumps(qa, indent=1, default=str))


if __name__ == "__main__":
    main()
