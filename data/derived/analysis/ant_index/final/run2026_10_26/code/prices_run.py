"""부록 15 등록 판정(부록 12, 2026-10-26) 가격 단계: 실제 가격을 받아 judge.py 입력 형식으로 저장한다.

동결 코드는 한 줄도 고치지 않고 불러 쓴다. 바꾸는 것은 기간·종목 목록·결과 폴더뿐이다.
  preview/code/prices.py   fetch_daily(yfinance Ticker.history, auto_adjust=False, actions=True)·fetch_nq_1h·플래그 규칙
                           (START·END만 2024-10-01 ~ 2026-10-23으로 바꿈)
  events/code/events.py    naver_history·reconcile(KOSPI = 네이버 일별 종가, SK하이닉스 = 네이버 다음 거래일 기준가)
종목: judge.py KR_T·US_T·VIX_T 전부(^KS11, 000660.KS, 005930.KS, ^KQ11, SOXX, ^SOX, QQQ, EWY, MU, SNDK, WDC, ^VIX).
      MU·SNDK·WDC는 judge.build_panel이 메모리 3종(H50 순차 칸) 열을 만들 때 읽으므로 넣는다(없으면 judge가 멈춤).
달력: run2026_10_26/results/loader_calendar.parquet(2024-11-01 ~ 2026-10-23, yfinance ^KS11·^GSPC 날짜). 그 앞
      2024-10-01 ~ 10-31은 같은 원천(yfinance ^KS11·^GSPC 일봉 날짜)으로 채우고 exchange_calendars XKRX·XNYS와 대조만 한다.
      달력에 없는 날짜 줄(휴장 빈 줄)은 뺀다(preview 규칙).
NQ=F 60분봉: yfinance 60분봉은 최근 730일만 주므로 시작을 그 한계 안(오늘 − 729일)으로 당긴다. judge가 쓰는 첫 시각은
      첫 창(2025-01-03)의 직전 미국 종가(2025-01-02 16:00 ET)라 범위 안이다.
네이버 대조(events.py 방식 그대로): 날마다 yfinance 종가와 네이버 대조값을 비교해(상대 오차 1e-6) 어긋나거나 yfinance가 비면
      네이버 값으로 종가만 바꾼다. yfinance 막대가 아예 없는 날은 종가만 있는 줄(시가·고가·저가 결측)을 넣는다.
      시가·고가·저가는 고치지 않고, KOSPI는 네이버 시가·고가·저가와 다른 날을 진단으로만 적는다.
산출: run2026_10_26/results/prices.parquet(긴 표, preview 형식), prices_nq_1h.parquet, prices_meta.json
실행: py -B prices_run.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from datetime import date, datetime, timedelta  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_paths as RP  # noqa: E402

KST = ZoneInfo("Asia/Seoul")
START, END = date(2024, 10, 1), date(2026, 10, 23)
TICKERS = {
    "^KS11": "krx", "000660.KS": "krx", "005930.KS": "krx", "^KQ11": "krx",
    "SOXX": "nyse", "^SOX": "nyse", "QQQ": "nyse", "EWY": "nyse", "^VIX": "nyse",
    "MU": "nyse", "SNDK": "nyse", "WDC": "nyse",
}
NAVER_ASSETS = {"^KS11": "KOSPI", "000660.KS": "SKHYNIX"}       # events.ASSETS 키
CAL_SRC = {"krx": "^KS11", "nyse": "^GSPC"}                     # 로더 달력과 같은 원천
EVCODE = RP.EV / "code"
PREVIEW_PRICES = RP.ANT / "preview" / "code" / "prices.py"


def _load(path, name: str):
    """RP.load와 같고, dataclass가 모듈을 찾도록 sys.modules에 먼저 올린다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_modules():
    # events/code 모듈은 `import snap_paths`를 쓴다 → 이 실행의 snap_paths(events/run2026_10_26/code)를 먼저 올린다
    _load(RP.EV / "run2026_10_26" / "code" / "snap_paths.py", "snap_paths")
    ev = _load(EVCODE / "events.py", "events_frozen")
    pr = _load(PREVIEW_PRICES, "preview_prices")
    pr.START, pr.END = START, END                               # 기간만 바꾼다
    return ev, pr


def calendar_days(pr) -> tuple[dict[str, set[date]], dict]:
    cal = pd.read_parquet(RP.RES / "loader_calendar.parquet")
    c0 = cal["date"].min().date()
    days = {"krx": {d.date() for d in cal.loc[cal["krx"], "date"] if START <= d.date() <= END},
            "nyse": {d.date() for d in cal.loc[cal["nyse"], "date"] if START <= d.date() <= END}}
    info = {"loader_calendar": str(RP.RES / "loader_calendar.parquet"), "loader_start": str(c0),
            "loader_sha256": RP.sha(RP.RES / "loader_calendar.parquet"), "pre_loader": {}}
    import exchange_calendars as xc
    for mk, t in CAL_SRC.items():
        h = pr.fetch_daily(t)
        h = h[h[["open", "high", "low", "close"]].notna().any(axis=1)]
        pre = {d for d in h["date"] if START <= d < c0}
        xs = {x.date() for x in xc.get_calendar("XKRX" if mk == "krx" else "XNYS").sessions_in_range(
            pd.Timestamp(START), pd.Timestamp(c0 - timedelta(days=1)))}
        days[mk] |= pre
        info["pre_loader"][mk] = {"source": f"yfinance {t} 일봉 날짜", "n": len(pre),
                                  "first": str(min(pre)) if pre else None, "last": str(max(pre)) if pre else None,
                                  "only_in_source": sorted(str(d) for d in pre - xs),
                                  "only_in_xcal": sorted(str(d) for d in xs - pre)}
    info["n_days"] = {k: len(v) for k, v in days.items()}
    return days, info


def naver_fix(ev, t: str, h: pd.DataFrame, days: list[date], session) -> tuple[pd.DataFrame, dict]:
    """events.reconcile로 날마다 종가를 정하고, 바뀐 날만 막대의 종가를 고친다."""
    asset = NAVER_ASSETS[t]
    cfg = ev.ASSETS[asset]
    yfd = pd.DataFrame({"date": [d.isoformat() for d in h["date"]], "yf_close": h["close"].astype(float).to_numpy(),
                        "yf_adj": h["adj_close"].astype(float).to_numpy()})
    nv = ev.naver_history(cfg["naver_url"], START.isoformat(), session)
    dstr = [d.isoformat() for d in sorted(days)]
    rec, repairs = ev.reconcile(asset, cfg, dstr, yfd, nv)
    h = h.copy()
    h["close_src"] = "yfinance"
    added, changed = [], []
    for r in repairs:
        if r.get("used") is None:
            continue
        d = date.fromisoformat(r["date"])
        m = h["date"] == d
        if m.any():
            old = float(h.loc[m, "close"].iloc[0]) if pd.notna(h.loc[m, "close"].iloc[0]) else None
            h.loc[m, "close"] = float(r["used"])
            h.loc[m, "close_src"] = f"naver_{cfg['verify']}"
            hi, lo = float(h.loc[m, "high"].iloc[0]), float(h.loc[m, "low"].iloc[0])
            changed.append({**r, "yf_close_before": old,
                            "close_outside_yf_high_low": bool(np.isfinite(hi) and np.isfinite(lo)
                                                              and not (lo <= r["used"] <= hi))})
        else:
            row = {c: np.nan for c in h.columns}
            row.update({"date": d, "close": float(r["used"]), "close_src": f"naver_{cfg['verify']}_bar_missing",
                        "dividends": 0.0, "splits": 0.0})
            h = pd.concat([h, pd.DataFrame([row])], ignore_index=True)
            added.append(r)
    h = h.sort_values("date").reset_index(drop=True)
    info = {"asset": asset, "naver_url": cfg["naver_url"], "verify": cfg["verify"], "match_rel": ev.MATCH_REL,
            "days_checked": len(dstr), "source_counts": rec["source"].value_counts().to_dict(),
            "repairs_close_changed": changed, "repairs_bar_missing_close_only": added,
            "unresolved": [r for r in repairs if r.get("used") is None],
            "naver_rows": int(len(nv)), "naver_first": str(nv["date"].min()), "naver_last": str(nv["date"].max())}
    if cfg["verify"] == "ref_next":
        both = rec.dropna(subset=["naver_close"])
        diff = both.loc[(both["naver_close"] / both["close"] - 1).abs() > ev.MATCH_REL, "date"].tolist()
        info["naver_close_col_differs_from_official"] = {"n": len(diff), "dates": diff}
    if asset == "KOSPI":          # 진단만: 시가·고가·저가는 고치지 않는다
        nvo = pd.DataFrame(nv_ohl(ev, cfg["naver_url"], session))
        mm = h.assign(ds=[d.isoformat() for d in h["date"]]).merge(nvo, left_on="ds", right_on="date",
                                                                     how="inner", suffixes=("", "_nv"))
        diag = {}
        for c in ("open", "high", "low"):
            rel = (mm[c] / mm[f"{c}_nv"] - 1).abs()
            bad = mm.loc[rel > 1e-6, ["ds", c, f"{c}_nv"]]
            diag[c] = {"n_compared": int(rel.notna().sum()), "n_differ": int(len(bad)),
                       "rows": [{"date": a, "yf": float(b), "naver": float(x)} for a, b, x in bad.itertuples(index=False)]}
        lnhl_y = np.log(np.log(mm["high"] / mm["low"]))
        lnhl_n = np.log(np.log(mm["high_nv"] / mm["low_nv"]))
        dd = (lnhl_y - lnhl_n).abs()
        diag["lnHL_abs_diff"] = {"n": int(dd.notna().sum()), "max": float(dd.max()), "n_gt_0.01": int((dd > 0.01).sum())}
        info["kospi_ohl_vs_naver_diagnostic"] = diag
    return h, info


def nv_ohl(ev, url: str, session) -> list[dict]:
    """네이버 일별 시가·고가·저가(진단용). events.naver_history와 같은 요청 꼴."""
    import time
    headers = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1",
               "Referer": "https://m.stock.naver.com/", "Accept": "application/json"}
    rows, page = [], 1
    while page <= 40:
        r = session.get(f"{url}?pageSize=60&page={page}", headers=headers, timeout=60)
        js = r.json()
        if not isinstance(js, list) or not js:
            break
        for x in js:
            rows.append({"date": str(x["localTradedAt"])[:10], "open_nv": ev._num(x.get("openPrice")),
                         "high_nv": ev._num(x.get("highPrice")), "low_nv": ev._num(x.get("lowPrice"))})
        if rows[-1]["date"] < START.isoformat() or len(js) < 60:
            break
        page += 1
        time.sleep(0.3)
    return list({r["date"]: r for r in rows}.values())


def main() -> int:
    j = RP.sha(RP.FINAL / "code" / "judge.py")
    if j != RP.JUDGE_SHA256:
        raise SystemExit(f"멈춤: judge.py SHA-256 {j} ≠ 등록값")
    now = datetime.now(KST)
    if now < datetime(2026, 10, 24, 6, 0, tzinfo=KST):
        raise SystemExit("멈춤: 10/23 종가 전")
    import requests
    ev, pr = load_modules()
    days, cal_info = calendar_days(pr)
    session = requests.Session()
    parts = []
    meta = {"made": now.replace(microsecond=0).isoformat(), "start": str(START), "end": str(END),
            "judge_sha256": j, "source": "yfinance Ticker.history(interval='1d', auto_adjust=False, actions=True) — "
            "preview/code/prices.fetch_daily 그대로(기간만 바꿈)", "calendar": cal_info,
            "volume_meaningful": sorted(set(pr.VOLUME_MEANINGFUL) | {"MU", "SNDK", "WDC"}),
            "naver": {}, "tickers": {},
            "code_sha256": {"preview/code/prices.py": RP.sha(PREVIEW_PRICES), "events/code/events.py": RP.sha(EVCODE / "events.py"),
                            "final/run2026_10_26/code/prices_run.py": RP.sha(RP.HERE / "prices_run.py")}}
    vol_ok = set(meta["volume_meaningful"])
    for t, mk in TICKERS.items():
        h = pr.fetch_daily(t)
        n_raw = len(h)
        allnan = h[["open", "high", "low", "close"]].isna().all(axis=1)
        h = h[~allnan]
        off = ~h["date"].isin(days[mk])
        info = {"market": mk, "rows_raw": int(n_raw), "rows_all_nan_dropped": int(allnan.sum()),
                "rows_off_calendar_dropped": [str(d) for d in h.loc[off, "date"]]}
        h = h[~off].copy()
        if t in NAVER_ASSETS:
            h, ninfo = naver_fix(ev, t, h, days[mk], session)
            meta["naver"][t] = ninfo
        else:
            h["close_src"] = "yfinance"
        first = h["date"].min()
        miss = sorted(d for d in days[mk] - set(h.loc[h["close"].notna(), "date"]) if d >= first)
        info["calendar_days"] = len(days[mk])
        info["calendar_days_without_close_after_first"] = [str(d) for d in miss]
        h["ticker"], h["market"] = t, mk
        bad_open = h["open"].isna() | (h["open"] <= 0)
        flat = (h["open"] == h["high"]) & (h["high"] == h["low"]) & (h["low"] == h["close"])
        hl_bad = h["high"] < h["low"]
        h["flag_open_bad"] = bad_open
        h["flag_flat_bar"] = flat
        h["flag_high_lt_low"] = hl_bad
        h["flag_zero_volume"] = (h["volume"].fillna(0) == 0) if t in vol_ok else False
        info.update({"rows": int(len(h)), "first": str(h["date"].min()), "last": str(h["date"].max()),
                     "last_close": float(h.loc[h["date"] == h["date"].max(), "close"].iloc[0]),
                     "has_2026_10_23_close": bool(((h["date"] == END) & h["close"].notna()).any()),
                     "open_bad": [str(d) for d in h.loc[bad_open, "date"]],
                     "flat_bar": [str(d) for d in h.loc[flat, "date"]],
                     "high_lt_low": [str(d) for d in h.loc[hl_bad, "date"]],
                     "zero_volume": [str(d) for d in h.loc[h["flag_zero_volume"].astype(bool), "date"]],
                     "flat_and_zero_volume_judge_sets_nan": [str(d) for d in h.loc[flat & h["flag_zero_volume"].astype(bool), "date"]],
                     "close_src_counts": h["close_src"].value_counts().to_dict(),
                     "dividend_days": {str(d): float(v) for d, v in zip(h["date"], h["dividends"]) if pd.notna(v) and v > 0},
                     "split_days": {str(d): float(v) for d, v in zip(h["date"], h["splits"]) if pd.notna(v) and v > 0}})
        meta["tickers"][t] = info
        parts.append(h)
    px = pd.concat(parts, ignore_index=True)
    px["date"] = pd.to_datetime(px["date"])
    px = px[["ticker", "market", "date", "open", "high", "low", "close", "adj_close", "volume", "dividends", "splits",
             "close_src", "flag_open_bad", "flag_flat_bar", "flag_high_lt_low", "flag_zero_volume"]]
    for c in ("flag_open_bad", "flag_flat_bar", "flag_high_lt_low", "flag_zero_volume"):
        px[c] = px[c].astype(bool)
    px.to_parquet(RP.RES / "prices.parquet", index=False)
    # NQ=F 60분봉(730일 한계 안)
    pr.START = max(START, (now.date() - timedelta(days=729)))
    nq = pr.fetch_nq_1h()
    pr.START = START
    nq = nq[nq["bar_start_et"] < pd.Timestamp(END + timedelta(days=1))].reset_index(drop=True)
    nq.to_parquet(RP.RES / "prices_nq_1h.parquet", index=False)
    meta["nq_1h"] = {"rows": int(len(nq)), "first_start_et": str(nq["bar_start_et"].min()),
                     "last_start_et": str(nq["bar_start_et"].max()),
                     "requested_start": str(max(START, (now.date() - timedelta(days=729)))),
                     "why_start": "yfinance 60분봉은 최근 730일만 준다. judge 첫 사용 시각은 2025-01-02 16:00 ET",
                     "use": "m0_nq_close_to_winclose·m0_nq_kr·m0_nq_1530. 바 끝(시작 + 1시간)이 목표 시각 이하인 마지막 바의 종가"}
    meta["files_sha256"] = {"prices.parquet": RP.sha(RP.RES / "prices.parquet"),
                            "prices_nq_1h.parquet": RP.sha(RP.RES / "prices_nq_1h.parquet")}
    (RP.RES / "prices_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), "utf-8")
    show = {t: {k: v for k, v in i.items() if k in ("rows", "first", "last", "last_close", "has_2026_10_23_close",
                                                     "calendar_days_without_close_after_first", "rows_off_calendar_dropped",
                                                     "open_bad", "flat_bar", "zero_volume", "high_lt_low",
                                                     "flat_and_zero_volume_judge_sets_nan", "close_src_counts")}
            for t, i in meta["tickers"].items()}
    print(json.dumps(show, ensure_ascii=False, indent=1))
    for t, n in meta["naver"].items():
        print(t, json.dumps({k: v for k, v in n.items() if k not in ("kospi_ohl_vs_naver_diagnostic",)},
                            ensure_ascii=False, default=str)[:3000])
        if "kospi_ohl_vs_naver_diagnostic" in n:
            d = n["kospi_ohl_vs_naver_diagnostic"]
            print({k: (v if k == "lnHL_abs_diff" else {"n_compared": v["n_compared"], "n_differ": v["n_differ"],
                                                        "rows": v["rows"][:10]}) for k, v in d.items()})
    print(json.dumps(meta["nq_1h"], ensure_ascii=False))
    print(json.dumps(cal_info["pre_loader"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
