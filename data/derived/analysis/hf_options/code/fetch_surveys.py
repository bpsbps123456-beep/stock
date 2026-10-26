"""설문 계열(개인투자자 vs 액티브 매니저 심리) 수집 — hf_options 연구(2026-10-14). 결과: cache/surveys.parquet

출처(전부 무료 공개 자료, 호스트마다 연결 하나, 요청 사이 간격을 둔다):
  - AAII 개인투자자 심리 조사: www.aaii.com/files/surveys/sentiment.xls (주간, 1987~).
    투표는 목 00:01 ~ 수 23:59(ET), 결과는 목요일 아침 AAII.com 게시(설문 페이지 문구). 표의 Reported Date = 게시일.
  - NAAIM 노출 지수: naaim.org 워드프레스 미디어 목록(wp-json/wp/v2/media?search=USE_Data)에 매주 올라온 엑셀
    'USE_Data-since-Inception_<측정일>.xlsx'(2006~). 회원 운용사가 매주 수요일 종가 기준 미국 주식 노출(-200~+200%)을 낸다.
    2026-08-01부터 비회원은 유료 구독, 무료 화면(index.naaim.org/embeddable/table)은 3개월 늦게 보여준다.
known(그날 미국 종가 전에 공개돼 있던 거래일, 분석은 다음 거래일 시가 진입) 규칙:
  - AAII: Reported Date가 목요일이고 거래일이면 그날(아침 게시). 그 밖(추수감사절 주·휴장 목요일)은 다음 거래일로 늦춘다.
  - NAAIM: 그 주를 처음 담은 엑셀의 업로드 시각(ET)이 거래일 마감(16:00, 반일장 13:00) 전이면 그날, 아니면 다음 거래일.
    업로드 기록이 없는 2024-05 이전 주는 측정 수요일 뒤 둘째 거래일(보통 금요일)로 늦춰 잡는다(기록된 주의 98%는 목 정오 전후).
    기록 구간 안에서 그 주 파일이 빠진 주는 그 주를 처음 담은 다음 파일의 업로드 시각을 쓴다(제때 공개 증거 없음).
    2026-08-01 이후 측정 주는 무료 지연 표에 처음 보인 시각(이 스크립트가 naaim_table_seen.csv에 남긴다)을 쓰고, 안 보이면 뺀다.
거래일은 yfinance QQQ 일봉 날짜, 아직 끝나지 않은 오늘 장은 뺀다.

실행: py fetch_surveys.py [--since 2010-01-01] [--gap 1.5] [--refresh]  (이어 받기: 받아 둔 엑셀은 다시 받지 않는다)
"""
from __future__ import annotations

import argparse
import bisect
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RAW = CACHE / "raw_surveys"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
      "Accept": "*/*"}
AAII_URL = "https://www.aaii.com/files/surveys/sentiment.xls"
NAAIM_MEDIA = ("https://naaim.org/wp-json/wp/v2/media?search=USE_Data&per_page=100&page={p}&orderby=date&order=desc"
               "&_fields=id,date,date_gmt,source_url")
NAAIM_TABLE = "https://index.naaim.org/embeddable/table"
PAYWALL = "2026-08-01"  # 이날부터 비회원은 3개월 지연
ET = "America/New_York"
NAAIM_COLS = {"NAAIM Number": "naaim", "Most Bearish": "naaim_most_bearish", "Quart 1": "naaim_q1",
              "Quart 2": "naaim_median", "Quart 3": "naaim_q3", "Most Bullish": "naaim_most_bullish",
              "Standard Deviation": "naaim_std"}
TABLE_COLS = {"NAAIM Number": "naaim", "Bearish": "naaim_most_bearish", "Quarter 1": "naaim_q1",
              "Quarter 2": "naaim_median", "Quarter 3": "naaim_q3", "Bullish": "naaim_most_bullish",
              "Deviation": "naaim_std"}
_LAST: dict[str, float] = {}


def _get(url: str, gap: float, timeout: int = 60) -> bytes | None:
    """호스트마다 gap초 간격. 404·403은 None, 429는 바로 멈춘다. 그 밖의 오류는 두 번 더 시도한 뒤 올린다."""
    host = urllib.parse.urlparse(url).netloc
    for k in range(3):
        wait = _LAST.get(host, 0) + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise SystemExit(f"429 Too Many Requests: {url} — 수집을 멈춘다")
            if e.code in (403, 404):
                return None
            if k == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if k == 2:
                raise
        finally:
            _LAST[host] = time.time()
        time.sleep(30 * (k + 1))
    return None


def _cached(path: Path, url: str, max_age_h: float, gap: float, refresh: bool) -> bytes:
    """원본을 raw에 둔다. max_age_h보다 새 파일이면 다시 받지 않는다(inf = 한 번 받으면 끝). 받기에 실패하면 남은 파일을 쓴다."""
    if path.exists():
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if max_age_h == float("inf") or (age_h < max_age_h and not refresh):
            return path.read_bytes()
    b = _get(url, gap)
    if b is None:
        if path.exists():
            print(f"받기 실패, 남은 파일 사용: {path.name}")
            return path.read_bytes()
        raise RuntimeError(f"받지 못함: {url}")
    path.write_bytes(b)
    return b


def calendar(since: str) -> tuple[list[str], set[str]]:
    """끝난 미국 거래일 목록과 반일장(13:00 마감) 날짜 집합."""
    import yfinance as yf
    h = yf.Ticker("QQQ").history(start=since, auto_adjust=False)
    days = sorted({d.strftime("%Y-%m-%d") for d in h.index})
    now = pd.Timestamp.now(tz=ET)
    if days and days[-1] == now.strftime("%Y-%m-%d") and now.hour < 16:
        days = days[:-1]  # 아직 끝나지 않은 오늘 장
    s = set(days)
    half = set()
    for y in range(int(since[:4]), now.year + 1):
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
        thanks = [d for d in nov if d.weekday() == 3][3]
        for d in (thanks + pd.Timedelta(days=1), pd.Timestamp(f"{y}-12-24"), pd.Timestamp(f"{y}-07-03")):
            if d.strftime("%Y-%m-%d") in s:
                half.add(d.strftime("%Y-%m-%d"))
    return days, half


def next_td(d: str, days: list[str], k: int = 1) -> str | None:
    """d 다음(d 제외) k번째 거래일. 달력 밖이면 None."""
    i = bisect.bisect_right(days, d) + k - 1
    return days[i] if i < len(days) else None


def known_from_ts(ts: pd.Timestamp, days: list[str], half: set[str]) -> str | None:
    """ET 시각 ts에 공개된 값의 known: 그날이 거래일이고 마감 전이면 그날, 아니면 다음 거래일."""
    d = ts.strftime("%Y-%m-%d")
    i = bisect.bisect_left(days, d)
    if i < len(days) and days[i] == d and ts.hour < (13 if d in half else 16):
        return d
    return next_td(d, days)


def _keep_latest(df: pd.DataFrame, asof: str, start: str) -> pd.DataFrame:
    """달력 시작 전 측정은 버리고, known이 겹치면 측정일이 늦은 쪽만 남긴다."""
    df = df[df[asof] >= start].dropna(subset=["known"]).sort_values(["known", asof])
    return df.drop_duplicates("known", keep="last").set_index("known")


def aaii(days: list[str], gap: float, refresh: bool) -> pd.DataFrame:
    b = _cached(RAW / "aaii_sentiment.xls", AAII_URL, 20, gap, refresh)
    raw = pd.read_excel(io.BytesIO(b), sheet_name="SENTIMENT", header=None)
    head = [str(x) for x in raw.iloc[3, :7]]
    if head[:4] != ["Date", "Bullish", "Neutral", "Bearish"] or "8-week" not in str(raw.iloc[2, 5]):
        raise ValueError(f"AAII 표 머리글이 바뀌었다: {head}")
    d = raw.iloc[5:, [0, 1, 2, 3]].copy()
    d.columns = ["rep", "bull", "neutral", "bear"]
    d["rep"] = pd.to_datetime(d["rep"], errors="coerce")
    for c in ("bull", "neutral", "bear"):
        d[c] = pd.to_numeric(d[c], errors="coerce") * 100  # 퍼센트로
    d = d.dropna(subset=["rep", "bull", "bear"]).sort_values("rep").reset_index(drop=True)
    out = pd.DataFrame({"aaii_bull": d["bull"], "aaii_neutral": d["neutral"], "aaii_bear": d["bear"],
                        "aaii_spread": d["bull"] - d["bear"], "aaii_bull_8w": d["bull"].rolling(8).mean()})
    rep = d["rep"].dt.strftime("%Y-%m-%d")
    out["aaii_asof"] = (d["rep"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")  # 투표 마지막 날(보통 수)
    ds = set(days)
    out["known"] = [r if (r in ds and pd.Timestamp(r).weekday() == 3) else next_td(r, days) for r in rep]
    return _keep_latest(out, "aaii_asof", days[0])


def naaim_media(gap: float, refresh: bool) -> pd.DataFrame:
    """주간 엑셀 업로드 목록: 측정일(파일 이름), 업로드 시각(ET), 주소."""
    rows, p = [], 1
    while True:
        b = _cached(RAW / f"naaim_media_p{p}.json", NAAIM_MEDIA.format(p=p), 12, gap, refresh)
        page = json.loads(b)
        rows += page
        if len(page) < 100:
            break
        p += 1
    m = pd.DataFrame(rows)
    m["file"] = m["source_url"].str.rsplit("/", n=1).str[-1]
    m["name_date"] = pd.to_datetime(m["file"].str.extract(r"Inception_(\d{4}-\d{1,2}-\d{1,2})")[0], errors="coerce")
    m["up_et"] = pd.to_datetime(m["date_gmt"]).dt.tz_localize("UTC").dt.tz_convert(ET)
    return m.dropna(subset=["name_date"]).sort_values(["name_date", "up_et"]).reset_index(drop=True)


def _naaim_xlsx(b: bytes) -> pd.DataFrame:
    x = pd.read_excel(io.BytesIO(b))
    ren = {"Date": "asof"}
    for c in x.columns:
        for key, name in NAAIM_COLS.items():
            if str(c).startswith(key):
                ren[c] = name
    if set(NAAIM_COLS.values()) - set(ren.values()) or "Date" not in x.columns:
        raise ValueError(f"NAAIM 엑셀 머리글이 바뀌었다: {list(x.columns)}")
    x = x.rename(columns=ren)[list(ren.values())]
    x["asof"] = pd.to_datetime(x["asof"], errors="coerce")
    return x.dropna(subset=["asof", "naaim"]).drop_duplicates("asof", keep="first")


def naaim(days: list[str], half: set[str], gap: float, refresh: bool) -> pd.DataFrame:
    m = naaim_media(gap, refresh)
    last = m.iloc[-1]  # 가장 늦은 측정일, 같은 날이면 늦게 올라온 정정본
    b = _cached(RAW / f"naaim_{last['file']}", last["source_url"], float("inf"), gap, refresh)
    x = _naaim_xlsx(b)
    x = x[x["asof"] < PAYWALL].copy()  # 유료 전환 뒤 주는 무료 지연 표에서만 받는다
    # 업로드 기록: 그 주(측정일 ~ 6일 뒤) 이름을 단 파일의 첫 업로드 시각.
    # 기록 구간(2024-05~) 안에서 그 주 파일이 없으면(2024-05-08·2026-04-22, 둘 다 창립자상 발표 무렵) 그 주를 처음 담은
    # 다음 파일(누적 파일)의 업로드 시각을 쓴다. 기록 구간 전 주만 기본 규칙(측정 뒤 둘째 거래일).
    nd, up = m["name_date"].tolist(), m["up_et"].tolist()
    first = min(nd)

    def pub(a: pd.Timestamp):
        c = [u for n, u in zip(nd, up) if a <= n < a + pd.Timedelta(days=7)]
        if c:
            return min(c), 1.0
        later = [u for n, u in zip(nd, up) if n >= a]
        if a >= first and later:
            return min(later), 3.0
        return None, 0.0
    p = [pub(a) for a in x["asof"]]
    x["pub"] = [t for t, _ in p]
    x["naaim_known_src"] = [k for _, k in p]  # 1 그 주 파일 업로드, 3 다음 파일 업로드, 0 기본 규칙
    x["known"] = [known_from_ts(p, days, half) if pd.notna(p) else next_td(a.strftime("%Y-%m-%d"), days, 2)
                  for a, p in zip(x["asof"], x["pub"])]
    # 무료 지연 표: 처음 보인 시각을 남기고, 유료 전환 뒤 주는 그 시각으로 known을 잡는다
    t = pd.read_html(io.StringIO(_cached(RAW / "naaim_emb_table.html", NAAIM_TABLE, 12, gap, refresh)
                                 .decode("utf-8")))[0]
    if set(TABLE_COLS) - set(t.columns) or "Date" not in t.columns:
        raise ValueError(f"NAAIM 지연 표 머리글이 바뀌었다: {list(t.columns)}")
    t = t.rename(columns={"Date": "asof", **TABLE_COLS})
    t["asof"] = pd.to_datetime(t["asof"], format="%m/%d/%Y")
    seen_f = RAW / "naaim_table_seen.csv"
    seen = pd.read_csv(seen_f) if seen_f.exists() else pd.DataFrame(columns=["asof", "first_seen_et"])
    now = pd.Timestamp.now(tz=ET).strftime("%Y-%m-%d %H:%M:%S")
    new = sorted(set(t["asof"].dt.strftime("%Y-%m-%d")) - set(seen["asof"].astype(str)))
    if new:
        seen = pd.concat([seen, pd.DataFrame({"asof": new, "first_seen_et": now})], ignore_index=True)
        seen.sort_values("asof").to_csv(seen_f, index=False)
    late = t[t["asof"] >= PAYWALL].copy()
    if len(late):
        fs = dict(zip(seen["asof"].astype(str), seen["first_seen_et"]))
        late["known"] = [known_from_ts(pd.Timestamp(fs[a.strftime("%Y-%m-%d")]).tz_localize(ET), days, half)
                         for a in late["asof"]]
        late["naaim_known_src"] = 2.0  # 무료 지연 표에 처음 보인 시각
        x = pd.concat([x, late[x.columns.drop("pub").tolist()]], ignore_index=True)
    x["naaim_asof"] = x["asof"].dt.strftime("%Y-%m-%d")
    cols = list(NAAIM_COLS.values()) + ["naaim_known_src", "naaim_asof", "known"]
    return _keep_latest(x[cols], "naaim_asof", days[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2010-01-01")
    ap.add_argument("--gap", type=float, default=1.5)
    ap.add_argument("--refresh", action="store_true", help="목록·AAII 표를 나이와 무관하게 다시 받는다")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    days, half = calendar((pd.Timestamp(a.since) - pd.Timedelta(days=15)).strftime("%Y-%m-%d"))
    s = aaii(days, a.gap, a.refresh).join(naaim(days, half, a.gap, a.refresh), how="outer")
    s = s[(s.index >= a.since) & (s.index <= days[-1])].sort_index()
    s.index.name = "known"
    num = [c for c in s.columns if not c.endswith("_asof")]
    s[num] = s[num].astype(float)
    s.to_parquet(CACHE / "surveys.parquet")
    print(f"surveys.parquet {s.shape} {s.index[0]}~{s.index[-1]} | 마지막 거래일 {days[-1]}")
    for c in ("aaii_bull", "naaim"):
        v = s[c].dropna()
        print(f"  {c}: {len(v)}행, 마지막 known {v.index[-1]} (측정 {s.loc[v.index[-1], c.split('_')[0] + '_asof']})")


if __name__ == "__main__":
    main()
