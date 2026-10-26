"""가격 전 점검표 #11: 외부 자료를 받을 수 있는지만 확인한다(HYPOTHESES.md §5.1 #11).

- 가격 이력은 날짜·시각(색인)만 보고 값은 저장하지 않는다. 값 비교(시가 원천 대조)는 2025-01 첫 주처럼
  모든 포럼 창(2026-07-14~)보다 앞선 구간에서만 한다. 그래서 어떤 창의 마감 뒤 값도 읽지 않는다.
- 웹 검색은 쓰지 않는다. 저장소에 이미 있는 원천 주소와 공개 첫 화면에 HTTP 요청만 보낸다(개인 정보 없음).

산출: results/prechecks_external.json
실행: py prechecks_external.py
"""
from __future__ import annotations

import json
import re
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[6]
RESULTS = Path(__file__).resolve().parents[1] / "results"
UA = {"User-Agent": "Mozilla/5.0"}
OLD = ("2025-01-02", "2025-01-10")   # 모든 포럼 창보다 앞선 구간


def yf_index_only(ticker: str, start: str, end: str, interval: str = "1d") -> dict:
    import yfinance as yf
    try:
        h = yf.Ticker(ticker).history(start=start, end=end, interval=interval)
    except Exception as ex:  # noqa: BLE001
        return {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
    idx = h.index
    del h                                   # 값은 버린다
    if len(idx) == 0:
        return {"ok": False, "rows": 0}
    return {"ok": True, "rows": int(len(idx)), "first": str(idx.min()), "last": str(idx.max()),
            "hours_et": sorted({int(t.tz_convert("America/New_York").hour) for t in idx}) if interval != "1d" else None}


def http_status(url: str, method: str = "GET", **kw) -> dict:
    try:
        r = requests.request(method, url, headers=UA, timeout=30, **kw)
        return {"ok": r.status_code == 200, "status": r.status_code, "bytes": len(r.content),
                "content_type": r.headers.get("Content-Type", "")[:60]}
    except Exception as ex:  # noqa: BLE001
        return {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}


def naver_kosdaq_open_check() -> dict:
    """KOSDAQ 지수 일봉 시가: 네이버 fchart(2025-01 첫 주만) ↔ yfinance ^KQ11 시가 일치 비율. 값은 저장하지 않는다."""
    import yfinance as yf
    url = ("https://fchart.stock.naver.com/siseJson.nhn?symbol=KOSDAQ&requestType=1"
           "&startTime=20250102&endTime=20250110&timeframe=day")
    out = {"url": url}
    try:
        r = requests.get(url, headers=UA, timeout=30)
        out["status"] = r.status_code
        rows = re.findall(r"\[\"(\d{8})\",\s*([\d.]+),", r.text)
        nav = {datetime.strptime(d, "%Y%m%d").date(): float(o) for d, o in rows}
        out["naver_rows"] = len(nav)
    except Exception as ex:  # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {str(ex)[:160]}"
        return out
    try:
        h = yf.Ticker("^KQ11").history(start=OLD[0], end=OLD[1], auto_adjust=False)
        yo = {pd.Timestamp(t).date(): float(v) for t, v in h["Open"].items()}
        yc = {pd.Timestamp(t).date(): float(v) for t, v in h["Close"].shift(1).items()}
        del h
    except Exception as ex:  # noqa: BLE001
        out["yf_error"] = f"{type(ex).__name__}: {str(ex)[:160]}"
        return out
    common = sorted(set(nav) & set(yo))
    out["common_days"] = len(common)
    if common:
        out["open_match_0.1pct"] = int(sum(abs(yo[d] / nav[d] - 1) < 0.001 for d in common))
        out["yf_open_equals_prev_close"] = int(sum(yc.get(d) is not None and yc[d] == yc[d]
                                                   and abs(yo[d] / yc[d] - 1) < 1e-6 for d in common))
    return out


def kofia_probe() -> dict:
    url = "https://freesis.kofia.or.kr/meta/getMetaDataList.do"
    payload = {"dmSearch": {"tmpV40": "10000000", "tmpV41": "1", "tmpV1": "D", "tmpV45": "20250102",
                            "tmpV46": "20250103", "OBJ_NM": "STATSCU0100000060BO"}}
    try:
        r = requests.post(url, data=json.dumps(payload), timeout=60,
                          headers={**UA, "Content-Type": "application/json; charset=UTF-8",
                                   "Referer": "https://freesis.kofia.or.kr/"})
        j = r.json()
        ds = j.get("ds1") or j.get("dsResult") or next((v for v in j.values() if isinstance(v, list)), [])
        keys = sorted(ds[0].keys()) if ds else []
        return {"ok": r.status_code == 200 and bool(ds), "status": r.status_code, "rows": len(ds),
                "has_publish_time_field": any("TIME" in k.upper() or "DTTM" in k.upper() for k in keys),
                "fields_n": len(keys)}
    except Exception as ex:  # noqa: BLE001
        return {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:160]}"}


def local_kofia_files() -> dict:
    d = ROOT / "data" / "derived" / "analysis" / "leverage_flood" / "results"
    fs = sorted(p.name for p in d.glob("kofia_STATSCU*.csv"))
    return {"dir": str(d.relative_to(ROOT)), "files": fs,
            "range_in_name": sorted({re.search(r"_(\d{8})_(\d{8})", f).groups() for f in fs})}


def earnings_dates() -> dict:
    import yfinance as yf
    out = {}
    for t in ("NVDA", "MU", "AVGO", "000660.KS", "005930.KS"):
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=12)
            idx = [] if e is None else [pd.Timestamp(x).date() for x in e.index]
            del e
            out[t] = {"ok": bool(idx), "n_2025_2026": sum(date(2025, 1, 1) <= d <= date(2026, 10, 23) for d in idx),
                      "first": str(min(idx)) if idx else None}
        except Exception as ex:  # noqa: BLE001
            out[t] = {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:120]}"}
    return out


def repo_search(patterns: list[str], dirs: list[Path], exts=(".md", ".json", ".py")) -> dict:
    """저장소 문서·코드에 그 자료(또는 날짜)가 이미 있는지. 이 점검 폴더(ant_index)는 자기 글이라 뺀다."""
    hits = {p: [] for p in patterns}
    for d in dirs:
        for f in d.rglob("*"):
            if f.suffix not in exts or not f.is_file() or ".git" in f.parts or "ant_index" in f.parts:
                continue
            try:
                txt = f.read_text("utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            for p in patterns:
                if re.search(p, txt):
                    hits[p].append(str(f.relative_to(ROOT)))
    return {p: v[:10] for p, v in hits.items()}


def main() -> int:
    today = date.today()
    res: dict = {"checked_at": datetime.now().isoformat(timespec="seconds")}
    res["NQ=F_1h"] = {"old_week": yf_index_only("NQ=F", OLD[0], OLD[1], "60m"),
                      "beyond_730d": yf_index_only("NQ=F", "2024-10-01", "2024-10-10", "60m"),
                      "earliest_available_approx": str(today - timedelta(days=730)),
                      "deadline_to_freeze_2025-01": str(date(2025, 1, 2) + timedelta(days=730))}
    res["ES=F_1h"] = {"old_week": yf_index_only("ES=F", OLD[0], OLD[1], "60m")}
    res["KOSDAQ_open"] = {"yf_^KQ11_daily_old_week": yf_index_only("^KQ11", OLD[0], OLD[1]),
                          "naver_vs_yf": naver_kosdaq_open_check()}
    res["VKOSPI"] = {"yf_^VKOSPI": yf_index_only("^VKOSPI", OLD[0], OLD[1]),
                     "repo_manual_csv": str((ROOT / "data/derived/analysis/leverage_flood/results/"
                                             "kr_levetf_vkospi_manual.csv").exists())}
    res["KOFIA_freesis"] = {"api_old_days": kofia_probe(), "local": local_kofia_files()}
    res["KIND"] = {"home": http_status("https://kind.krx.co.kr/"),
                   "corp_list_download": http_status("https://kind.krx.co.kr/corpgeneral/corpList.do?method=download"),
                   "ipo_schedule": http_status("https://kind.krx.co.kr/listinvstg/pubofrschdl.do?method=searchPubofrScholMain")}
    res["SEIBro"] = {"home": http_status("https://seibro.or.kr/")}
    res["FOMC_calendar"] = http_status("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    res["BLS_schedule"] = http_status("https://www.bls.gov/schedule/news_release/cpi.htm")
    res["earnings_dates_yf"] = earnings_dates()
    res["repo_search"] = repo_search(
        [r"야간선물|야간 선물|KOSPI200 야간|night futures", r"규제 시행|시행일|기본예탁금",
         r"보관 상위|보관금액|seibro", r"FOMC", r"수요예측 일정|청약 일정|공모 일정"],
        [ROOT / "docs", ROOT / "data" / "derived" / "analysis"])
    out = RESULTS / "prechecks_external.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
