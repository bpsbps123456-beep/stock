"""헤지펀드 쏠림·개인 레버리지 대리 지표(hf_proxies) 수집 — 2026-10-14 옵션×헤지펀드 종합 연구.

출처(전부 무료 공개 자료, 호스트마다 연결 하나로 간격을 두고 받는다):
  - Yahoo(yfinance, auto_adjust=False): GVIP(헤지펀드 13F 상위 10종목에 자주 드는 50종목) QAI(멀티전략 추종) HDG(헤지펀드 복제)
    QQQ SPY SOXX, 레버리지 짝 TQQQ/SQQQ SOXL/SOXS NVDL/NVDD(+NVDU/NVDS).
    수익률·비율은 Adj Close, 거래대금은 Close×Volume(둘 다 분할 반영이라 곱이 실제 거래대금). 값은 그날 종가 → known = 그날.
  - ProShares 펀드별 일별 NAV CSV(accounts.profunds.com/etfdata/ByFund/<T>-historical_nav.csv, 2010년~):
    NAV·발행주식수(천 주)·순자산. 분할은 과거 줄까지 소급 반영돼 있다. 날짜 T 줄은 T 장 마감 뒤 늦게 올라온다
    (2026-10-13 19:56 ET에 받은 파일의 마지막 줄이 10-12) → known = T 다음 거래일. 자금 흐름 = 발행주식수 변화 × 그날 NAV.
  - SOXL/SOXS(Direxion)·QQQ(Invesco)는 무료 일별 발행주식수 이력을 찾지 못해 넣지 않았다(results/qa_hf_proxies.md).

실행: py fetch_hf_proxies.py [--since 2022-01-01] [--max-age-hours 12] [--gap 1.5]
  이어 받기: cache/raw_hf_proxies/의 원본이 max-age-hours보다 새로우면 다시 받지 않는다.
출력: cache/hf_proxies.parquet(지표, index known), cache/hf_proxies_by_symbol.parquet(종목별 가격·거래대금 긴 표).
"""
from __future__ import annotations

import argparse
import io
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RAW = CACHE / "raw_hf_proxies"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
      "Accept": "text/csv, */*"}
YAHOO = ["QQQ", "SPY", "SOXX", "GVIP", "QAI", "HDG", "TQQQ", "SQQQ", "SOXL", "SOXS", "NVDL", "NVDD", "NVDU", "NVDS"]
PROSHARES_URL = "https://accounts.profunds.com/etfdata/ByFund/{t}-historical_nav.csv"
# QQQ 레버리지(3배·2배·-1배), 반도체 2배(USD)·-2배(SSG), S&P 3배 짝
PROSHARES = ["TQQQ", "SQQQ", "QLD", "QID", "PSQ", "USD", "SSG", "UPRO", "SPXU"]


def _fresh(path: Path, max_age_h: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_h * 3600


def _get(url: str, timeout: int = 60) -> bytes | None:
    """404·403은 None. 429는 바로 멈춘다. 그 밖의 오류는 두 번 더 시도한 뒤 올린다."""
    for k in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if e.code == 429:
                raise SystemExit(f"HTTP 429(요청 과다) — 즉시 멈춤: {url}")
            if k == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if k == 2:
                raise
        time.sleep(30 * (k + 1))
    return None


def fetch_yahoo(since: str, max_age_h: float, gap: float) -> None:
    """종목마다 Close·Adj Close·Volume을 날짜 문자열 index로 저장한다."""
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError
    for t in YAHOO:
        out = RAW / f"yahoo_{t}.parquet"
        if _fresh(out, max_age_h):
            continue
        try:
            h = yf.Ticker(t).history(start=since, auto_adjust=False)
        except YFRateLimitError:
            raise SystemExit(f"Yahoo 429(요청 과다) — 즉시 멈춤: {t}")
        if h.empty:
            print(f"[yahoo] {t}: 자료 없음")
            continue
        h = h[["Close", "Adj Close", "Volume"]].astype("float64")
        h.index = h.index.strftime("%Y-%m-%d")
        h = h[~h.index.duplicated(keep="last")]
        h.index.name = "date"
        h.to_parquet(out)
        time.sleep(gap)


def fetch_proshares(max_age_h: float, gap: float) -> None:
    """펀드별 전체 이력 CSV 한 건씩. 받은 시각(UTC)을 파일 수정 시각으로 남긴다."""
    for t in PROSHARES:
        out = RAW / f"proshares_{t}.csv"
        if _fresh(out, max_age_h):
            continue
        b = _get(PROSHARES_URL.format(t=t))
        if not b or b[:4] != b"Date":
            print(f"[proshares] {t}: 자료 없음")
        else:
            out.write_bytes(b)
        time.sleep(gap)


def trading_days() -> list[str]:
    """QQQ 일봉 날짜. 아직 끝나지 않은 오늘(미국 동부 16:15 전) 줄은 버린다."""
    d = pd.read_parquet(RAW / "yahoo_QQQ.parquet").index.tolist()
    now = pd.Timestamp.now(tz="America/New_York")
    if d and d[-1] >= now.strftime("%Y-%m-%d") and (now.hour, now.minute) < (16, 15):
        d = [x for x in d if x < now.strftime("%Y-%m-%d")]
    return d


def load_yahoo(td: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for t in YAHOO:
        f = RAW / f"yahoo_{t}.parquet"
        if f.exists():
            out[t] = pd.read_parquet(f).reindex(td)
    return out


def _beta(y: pd.Series, x: pd.Series, n: int = 60) -> pd.Series:
    """y 일간 수익률을 x 일간 수익률에 회귀한 n일 이동 OLS 기울기 = cov/var."""
    ok = y.notna() & x.notna()
    y, x = y.where(ok), x.where(ok)
    return y.rolling(n, min_periods=n).cov(x) / x.rolling(n, min_periods=n).var()


def _bull(dv: dict[str, pd.Series], bulls: list[str], bears: list[str], n: int = 1) -> pd.Series:
    """상승 쪽 거래대금 몫 = 상승 / (상승 + 하락). n>1이면 n일 합으로 잰다. 한쪽이라도 없는 날은 NaN."""
    if any(t not in dv for t in bulls + bears):
        return pd.Series(np.nan, index=next(iter(dv.values())).index)
    b = sum(dv[t] for t in bulls)
    s = sum(dv[t] for t in bears)
    if n > 1:
        b, s = b.rolling(n, min_periods=n).sum(), s.rolling(n, min_periods=n).sum()
    return b / (b + s)


def price_series(px: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """그날 종가로 아는 지표(known = 그날)."""
    adj = {t: d["Adj Close"] for t, d in px.items()}
    ret = {t: s.pct_change(fill_method=None) for t, s in adj.items()}
    dv = {t: d["Close"] * d["Volume"] for t, d in px.items()}
    o = pd.DataFrame(index=px["QQQ"].index)
    o["gvip_rel"] = adj["GVIP"] / adj["QQQ"]
    o["gvip_rel20"] = o["gvip_rel"].pct_change(20, fill_method=None)
    o["gvip_ret20"] = adj["GVIP"].pct_change(20, fill_method=None)
    o["gvip_rel_spy"] = adj["GVIP"] / adj["SPY"]
    o["gvip_rel_spy20"] = o["gvip_rel_spy"].pct_change(20, fill_method=None)
    o["qai_beta60"] = _beta(ret["QAI"], ret["QQQ"])
    o["qai_beta60_spy"] = _beta(ret["QAI"], ret["SPY"])
    o["hdg_beta60"] = _beta(ret["HDG"], ret["SPY"])
    o["hdg_beta60_qqq"] = _beta(ret["HDG"], ret["QQQ"])
    o["qai_ret20"] = adj["QAI"].pct_change(20, fill_method=None)
    o["hdg_ret20"] = adj["HDG"].pct_change(20, fill_method=None)
    for name, bulls, bears in [("qqq", ["TQQQ"], ["SQQQ"]), ("soxx", ["SOXL"], ["SOXS"]),
                               ("nvda", ["NVDL"], ["NVDD"]), ("nvda_all", ["NVDL", "NVDU"], ["NVDD", "NVDS"])]:
        o[f"lev_{name}_bull"] = _bull(dv, bulls, bears)
        o[f"lev_{name}_bull5"] = _bull(dv, bulls, bears, 5)
    # 레버리지 짝 거래대금 / 기초 ETF 거래대금(개인 레버리지 활동 강도)
    o["lev_qqq_share"] = (dv["TQQQ"] + dv["SQQQ"]) / dv["QQQ"]
    o["lev_soxx_share"] = (dv["SOXL"] + dv["SOXS"]) / dv["SOXX"]
    o["lev_qqq_share5"] = o["lev_qqq_share"].rolling(5, min_periods=5).mean()
    o["lev_soxx_share5"] = o["lev_soxx_share"].rolling(5, min_periods=5).mean()
    return o


def load_proshares(t: str) -> pd.DataFrame | None:
    f = RAW / f"proshares_{t}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, index_col=False)
    d["asof"] = pd.to_datetime(d["Date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    d = d.drop_duplicates("asof", keep="first").set_index("asof").sort_index()
    out = pd.DataFrame(index=d.index)
    out["nav"] = d["NAV"].astype(float)
    out["so"] = d["Shares Outstanding (000)"].astype(float) * 1000
    out["aum"] = d["Assets Under Management"].astype(float)
    out["flow"] = out["so"].diff() * out["nav"]  # 그날 바뀐 주식 수 × 그날 NAV(설정·환매 금액)
    return out


def flow_series(td: list[str], since: str) -> pd.DataFrame:
    """ProShares 발행주식수·자금 흐름. 측정일(asof) 다음 거래일을 known으로 둔다. 다음 거래일이 아직 없는 줄은 버린다."""
    tdi = pd.Index(td)
    cols = {}
    for t in PROSHARES:
        d = load_proshares(t)
        if d is None:
            continue
        d = d[d.index >= since]
        for c in ("so", "flow", "aum"):
            cols[f"{c}_{t.lower()}"] = d[c]
    m = pd.DataFrame(cols).sort_index()
    pos = tdi.searchsorted(m.index, side="right")
    keep = pos < len(tdi)
    m = m[keep]
    m.insert(0, "known", tdi[pos[keep]])
    m.index.name = "asof"
    m = m.reset_index().drop_duplicates("known", keep="last").set_index("known")
    m = m.reindex(td)

    def g(c: str) -> pd.Series:
        return m[c] if c in m else pd.Series(np.nan, index=m.index)

    m["lev_qqq_flow"] = g("flow_tqqq") - g("flow_sqqq")
    m["lev_qqq_flow5"] = m["lev_qqq_flow"].rolling(5, min_periods=5).sum()
    m["lev_qqq_flow20"] = m["lev_qqq_flow"].rolling(20, min_periods=20).sum()
    lev_aum = g("aum_tqqq") + g("aum_sqqq")
    m["lev_qqq_flow5_pct"] = m["lev_qqq_flow5"] / lev_aum
    m["lev_qqq_aum_bull"] = g("aum_tqqq") / lev_aum
    # 3배·2배 상승 − 3배·2배·1배 하락(QQQ 레버리지 전체)
    m["lev_qqq_flow_all"] = g("flow_tqqq") + g("flow_qld") - g("flow_sqqq") - g("flow_qid") - g("flow_psq")
    m["lev_qqq_flow_all5"] = m["lev_qqq_flow_all"].rolling(5, min_periods=5).sum()
    m["lev_semi_flow"] = g("flow_usd") - g("flow_ssg")
    m["lev_semi_flow5"] = m["lev_semi_flow"].rolling(5, min_periods=5).sum()
    m["lev_spx_flow"] = g("flow_upro") - g("flow_spxu")
    m["lev_spx_flow5"] = m["lev_spx_flow"].rolling(5, min_periods=5).sum()
    return m


def build(since: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    td = [d for d in trading_days() if d >= since]
    px = load_yahoo(td)
    p = price_series(px)
    f = flow_series(td, since)
    asof = f.pop("asof")
    out = pd.concat([p, f], axis=1).astype("float64")
    out.insert(0, "asof", asof)  # 자금 흐름(so_ flow_ aum_ lev_*_flow*) 칸의 측정일. 가격 지표는 known 그날 값.
    out.index.name = "known"
    long = []
    for t, d in px.items():
        x = d.rename(columns={"Close": "close", "Adj Close": "adj_close", "Volume": "volume"})
        x["dollar_volume"] = x["close"] * x["volume"]
        x.insert(0, "symbol", t)
        long.append(x.dropna(subset=["close"]))
    by = pd.concat(long).rename_axis("known").reset_index().sort_values(["known", "symbol"]).reset_index(drop=True)
    return out, by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2022-01-01", help="2023 값의 250일 백분위 창을 채우려고 2022부터 받는다")
    ap.add_argument("--max-age-hours", type=float, default=12.0)
    ap.add_argument("--gap", type=float, default=1.5)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    fetch_yahoo(a.since, a.max_age_hours, a.gap)
    fetch_proshares(a.max_age_hours, a.gap)
    out, by = build(a.since)
    out.to_parquet(CACHE / "hf_proxies.parquet")
    by.to_parquet(CACHE / "hf_proxies_by_symbol.parquet", index=False)
    print(f"hf_proxies: {len(out)}줄 {out.index[0]}~{out.index[-1]}, 칸 {out.shape[1]}개 / by_symbol {len(by)}줄")


if __name__ == "__main__":
    main()
