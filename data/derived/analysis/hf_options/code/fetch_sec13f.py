"""옵션·헤지펀드 종합 연구(2026-10-14) sec13f 묶음: 헤지펀드 13F 분기 보유 중 AI 미국 종목.

출처: SEC 'Form 13F Data Sets'(https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets).
  zip 하나에 제출일 구간(약 3개월)의 13F 전부가 tsv로 들어 있다. SUBMISSION(제출일·CIK·보고 분기),
  COVERPAGE(운용사 이름·수정 종류), SUMMARYPAGE(표 합계), INFOTABLE(종목별 보유: CUSIP·평가액 달러·주식 수·풋/콜).
  옵션 행(PUTCALL Put/Call)의 평가액은 기초자산 주식 가치(명목)이고 주식 수는 기초자산 주식 수다.
  SEC 요구대로 용도를 밝힌 User-Agent 하나(개인 이메일 없음)로, 요청 사이 1초 이상 띄워 받는다. 403·429면 바로 멈춘다.
시점(known): 13F 마감은 분기 말 + 45일(2/14·5/15·8/14·11/14). 주말·연방 휴일이면 다음 영업일이다.
  마감일까지 낸 원본·수정본만 쓰고 늦게 낸 것은 버린다. 마감일 날짜로 찍힌 제출은 그날 오후 5시 30분(ET,
  Regulation S-T 13(a))까지 접수되므로 4시 종가 뒤에 공개될 수 있다. 그래서 known = 마감일 '다음' 뉴욕 거래일
  (그날 4시 종가 전에는 전부 공개됨). 검증(2026-10-14)에서 마감일 당일 → 다음 거래일로 한 칸 늦췄다.
헤지펀드: HF 표의 이름 정규식으로 COVERPAGE FILINGMANAGER_NAME을 찾는다. 시장조성자·은행·'SECURITIES' 법인은 뺀다.
  같은 운용사의 여러 CIK(예: Two Sigma Investments·Advisers)는 한 펀드(key)로 합친다.
AI 종목: watchlist.AI_UNIVERSE 미국 상장(심볼에 '.' 없음). CUSIP→티커는 발행사 이름 정규식으로 붙이고,
  13F 평가액/주식 수로 잰 가격이 Yahoo 분기 말 종가(분할 전 원래 가격)와 맞는 CUSIP만 쓴다.
  GOOG(클래스 C)는 저장소 별칭 규칙(AI_UNIVERSE_ALIASES)대로 GOOGL에 합친다.
분기 순매수(shares_chg)는 두 분기 모두 낸 펀드만, (이번 주식 수 - 지난 주식 수 x 분할 비율) x 이번 분기 말 종가로 잰다.

실행: py fetch_sec13f.py [--gap 1.0] [--since 2023-06-30] [--keep-zip] [--offline]
  이어 받기: 처리를 마친 zip(raw_sec13f/<이름>/info.parquet)은 다시 받지 않는다. --offline은 받은 것만으로 다시 계산한다.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]  # 저장소 루트(watchlist.py)
CACHE = HERE.parent / "cache"
RAW = CACHE / "raw_sec13f"
RESULTS = HERE.parent / "results"
LOG = RAW / "fetch.log"
UA = {"User-Agent": "GBRC-research personal-noncommercial", "Accept-Encoding": "gzip, deflate"}
LANDING = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
SEC = "https://www.sec.gov"

# (펀드 key, 분류, FILINGMANAGER_NAME 정규식 — 대문자·구두점을 공백으로 바꾼 이름에 적용)
# 분류: multi 멀티매니저·포드형, quant 계량·시스템 매크로, growth 기술·성장 롱숏(타이거 계열 포함), macro_value 가치·행동주의·이벤트·매크로
HF = [
    ("citadel", "multi", r"^CITADEL ADVISORS"), ("millennium", "multi", r"^MILLENNIUM MANAGEMENT"),
    ("point72", "multi", r"^POINT72 ASSET MANAGEMENT"), ("balyasny", "multi", r"^BALYASNY ASSET MANAGEMENT"),
    ("schonfeld", "multi", r"^SCHONFELD STRATEGIC ADVISORS"), ("exoduspoint", "multi", r"^EXODUSPOINT CAPITAL"),
    ("hudson_bay", "multi", r"^HUDSON BAY CAPITAL"), ("sculptor", "multi", r"^SCULPTOR CAPITAL"),
    ("marshall_wace", "multi", r"^MARSHALL WACE"), ("verition", "multi", r"^VERITION FUND MANAGEMENT"),
    ("eisler", "multi", r"^EISLER CAPITAL"), ("woodline", "multi", r"^WOODLINE PARTNERS"),
    ("two_sigma", "quant", r"^TWO SIGMA (?:INVESTMENTS|ADVISERS)"), ("renaissance", "quant", r"^RENAISSANCE TECHNOLOGIES"),
    ("de_shaw", "quant", r"^D E SHAW"), ("aqr", "quant", r"^AQR CAPITAL MANAGEMENT"),
    ("arrowstreet", "quant", r"^ARROWSTREET CAPITAL"), ("man_group", "quant", r"^MAN GROUP"),
    ("bridgewater", "quant", r"^BRIDGEWATER ASSOCIATES"), ("gotham", "quant", r"^GOTHAM ASSET MANAGEMENT"),
    ("qube", "quant", r"^QUBE RESEARCH"), ("pdt", "quant", r"^PDT PARTNERS"),
    ("voleon", "quant", r"^VOLEON CAPITAL"), ("graham", "quant", r"^GRAHAM CAPITAL MANAGEMENT"),
    ("tiger_global", "growth", r"^TIGER GLOBAL MANAGEMENT"), ("coatue", "growth", r"^COATUE MANAGEMENT"),
    ("viking", "growth", r"^VIKING GLOBAL INVESTORS"), ("lone_pine", "growth", r"^LONE PINE CAPITAL"),
    ("maverick", "growth", r"^MAVERICK CAPITAL"), ("light_street", "growth", r"^LIGHT STREET CAPITAL"),
    ("whale_rock", "growth", r"^WHALE ROCK CAPITAL"), ("altimeter", "growth", r"^ALTIMETER CAPITAL"),
    ("d1", "growth", r"^D1 CAPITAL PARTNERS"), ("dragoneer", "growth", r"^DRAGONEER INVESTMENT"),
    ("durable", "growth", r"^DURABLE CAPITAL PARTNERS"), ("holocene", "growth", r"^HOLOCENE ADVISORS"),
    ("alkeon", "growth", r"^ALKEON CAPITAL"), ("atreides", "growth", r"^ATREIDES MANAGEMENT"),
    ("suvretta", "growth", r"^SUVRETTA CAPITAL"), ("steadfast", "growth", r"^STEADFAST CAPITAL"),
    ("darsana", "growth", r"^DARSANA CAPITAL"), ("discovery", "growth", r"^DISCOVERY CAPITAL MANAGEMENT"),
    ("egerton", "growth", r"^EGERTON CAPITAL"), ("lansdowne", "growth", r"^LANSDOWNE PARTNERS"),
    ("kora", "growth", r"^KORA MANAGEMENT"), ("jericho", "growth", r"^JERICHO CAPITAL"),
    ("contour", "growth", r"^CONTOUR ASSET MANAGEMENT"), ("xn", "growth", r"^XN (?:LP|EXPONENT)"),
    ("hhlr", "growth", r"^HHLR ADVISORS"), ("cadian", "growth", r"^CADIAN CAPITAL"),
    ("palestra", "growth", r"^PALESTRA CAPITAL"), ("senvest", "growth", r"^SENVEST MANAGEMENT"),
    ("srs", "growth", r"^SRS INVESTMENT MANAGEMENT"), ("eminence", "growth", r"^EMINENCE CAPITAL"),
    ("glenview", "growth", r"^GLENVIEW CAPITAL"),
    ("pershing", "macro_value", r"^PERSHING SQUARE CAPITAL"), ("third_point", "macro_value", r"^THIRD POINT LLC"),
    ("elliott", "macro_value", r"^ELLIOTT INVESTMENT MANAGEMENT"), ("appaloosa", "macro_value", r"^APPALOOSA"),
    ("duquesne", "macro_value", r"^DUQUESNE FAMILY OFFICE"), ("soros", "macro_value", r"^SOROS FUND MANAGEMENT"),
    ("greenlight", "macro_value", r"^GREENLIGHT CAPITAL"), ("jana", "macro_value", r"^JANA PARTNERS"),
    ("valueact", "macro_value", r"^VALUEACT"), ("farallon", "macro_value", r"^FARALLON CAPITAL"),
    ("baupost", "macro_value", r"^BAUPOST GROUP"), ("tudor", "macro_value", r"^TUDOR INVESTMENT"),
    ("brevan_howard", "macro_value", r"^BREVAN HOWARD"), ("moore", "macro_value", r"^MOORE CAPITAL MANAGEMENT"),
    ("starboard", "macro_value", r"^STARBOARD VALUE"), ("trian", "macro_value", r"^TRIAN FUND MANAGEMENT"),
    ("corvex", "macro_value", r"^CORVEX MANAGEMENT"), ("sachem_head", "macro_value", r"^SACHEM HEAD"),
    ("soroban", "macro_value", r"^SOROBAN CAPITAL"), ("tci", "macro_value", r"^TCI FUND MANAGEMENT"),
    ("caxton", "macro_value", r"^CAXTON ASSOCIATES"),
]
CATS = ["multi", "quant", "growth", "macro_value"]
HF_EXCLUDE = re.compile(r"\bSECURITIES\b|\bBANK\b|\bBROKER")

# 티커 -> 발행사 이름 정규식(정규화한 NAMEOFISSUER에 적용). 가격 검사가 오매칭을 한 번 더 거른다.
ISSUER = {
    "AAOI": r"^APPLIED OPTOELECTRONICS", "AAPL": r"^APPLE INC", "ACN": r"^ACCENTURE", "ADI": r"^ANALOG DEVICES",
    "AEHR": r"^AEHR TEST", "AKAM": r"^AKAMAI", "ALAB": r"^ASTERA LABS", "AMAT": r"^APPLIED MAT",
    "AMD": r"^ADVANCED MICRO DEVICES", "AMKR": r"^AMKOR", "AMZN": r"^AMAZON", "ANET": r"^ARISTA NETWORKS",
    "AOSL": r"^ALPHA (?:AND )?OMEGA SEMI", "APH": r"^AMPHENOL", "APLD": r"^APPLIED DIGITAL", "ARM": r"^ARM HOLDINGS",
    "ASML": r"^ASML HOLDING", "ASX": r"^ASE TECH", "AVGO": r"^BROADCOM", "AXTI": r"^AXT INC", "BABA": r"^ALIBABA",
    "BE": r"^BLOOM ENERGY", "CAT": r"^CATERPILLAR", "CBRS": r"^CEREBRAS", "CEG": r"^CONSTELLATION ENERGY",
    "CIEN": r"^CIENA", "CIFR": r"^CIPHER (?:MINING|DIGITAL)", "CLS": r"^CELESTICA", "CMI": r"^CUMMINS",
    "COHR": r"^COHERENT", "CRDO": r"^CREDO TECH", "CRM": r"^SALESFORCE", "CRWD": r"^CROWDSTRIKE",
    "CRWV": r"^COREWEAVE", "CSCO": r"^CISCO", "DDOG": r"^DATADOG", "DELL": r"^DELL TECHNOLOGIES",
    "DT": r"^DYNATRACE", "EME": r"^EMCOR", "ENTG": r"^ENTEGRIS", "ETN": r"^EATON CORP", "FIX": r"^COMFORT SYS",
    "FLEX": r"^FLEX LTD", "FN": r"^FABRINET", "FORM": r"^FORMFACTOR", "FSLY": r"^FASTLY", "FTNT": r"^FORTINET",
    "GEV": r"^GE VERNOVA", "GFS": r"^GLOBALFOUNDRIES", "GLW": r"^CORNING INC", "GLXY": r"^GALAXY DIGITAL",
    "GOOGL": r"^ALPHABET", "GTLB": r"^GITLAB", "HPE": r"^HEWLETT PACKARD ENTERPRISE", "HUT": r"^HUT 8",
    "IBM": r"^(?:INTERNATIONAL|INTL) BUSINESS MACH|^IBM\b", "INTC": r"^INTEL CORP", "IPGP": r"^IPG PHOTONICS",
    "IREN": r"^IREN\b|^IRIS ENERGY", "KEYS": r"^KEYSIGHT", "KLAC": r"^KLA (?:CORP|TENCOR|INC)", "LITE": r"^LUMENTUM",
    "LRCX": r"^LAM RESEARCH", "MCHP": r"^MICROCHIP TECH", "MDB": r"^MONGODB", "META": r"^META PLATFORMS",
    "MOD": r"^MODINE", "MPWR": r"^MONOLITHIC P", "MRVL": r"^MARVELL", "MSFT": r"^MICROSOFT", "MTSI": r"^MACOM",
    "MU": r"^MICRON TECH", "NBIS": r"^NEBIUS|^YANDEX", "NET": r"^CLOUDFLARE", "NOW": r"^SERVICENOW",
    "NTAP": r"^NETAPP", "NVDA": r"^NVIDIA", "NVT": r"^NVENT", "NVTS": r"^NAVITAS SEMI", "NXPI": r"^NXP SEMI",
    "OKLO": r"^OKLO", "OKTA": r"^OKTA", "ON": r"^ON SEMI|^ONSEMI", "ORCL": r"^ORACLE CORP",
    "P": r"^PURE STORAGE|^EVERPURE", "PANW": r"^PALO ALTO NETWORKS", "PLTR": r"^PALANTIR", "QCOM": r"^QUALCOMM",
    "RMBS": r"^RAMBUS", "ROP": r"^ROPER TECH", "SAP": r"^SAP (?:SE|AG)\b", "SIMO": r"^SILICON MOTION",
    "SMCI": r"^SUPER MICRO COMPUTER", "SMR": r"^NUSCALE", "SNDK": r"^SANDISK", "SNOW": r"^SNOWFLAKE",
    "STM": r"^STMICROELECTRONICS", "STX": r"^SEAGATE", "SWKS": r"^SKYWORKS", "TEM": r"^TEMPUS AI",
    "TER": r"^TERADYNE", "TLN": r"^TALEN ENERGY", "TSEM": r"^TOWER SEMI", "TSLA": r"^TESLA",
    "TSM": r"^TAIWAN SEMI", "TTMI": r"^TTM TECH", "TXN": r"^TEXAS INSTR", "UMC": r"^UNITED MICROELECTR",
    "VIAV": r"^VIAVI", "VICR": r"^VICOR", "VRT": r"^VERTIV", "VST": r"^VISTRA", "WDC": r"^WESTERN DIGITAL",
    "WULF": r"^TERAWULF", "XE": r"^X ENERGY",
}
PRICE_OK = (0.8, 1.25)   # CUSIP 채택: 13F 내재 가격 / Yahoo 분기 말 종가의 분기 중앙값
ROW_OK = (0.2, 5.0)      # 행 단위 큰 오류(천 달러 단위 기재 등) 제외


def _log(msg: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _norm(s: pd.Series) -> pd.Series:
    """대문자, 영숫자 밖은 공백, 공백 하나로."""
    return s.fillna("").astype(str).str.upper().str.replace(r"[^A-Z0-9]+", " ", regex=True).str.strip()


def ai_symbols() -> list[str]:
    sys.path.insert(0, str(ROOT))
    import watchlist
    return sorted({s for subs in watchlist.AI_UNIVERSE.values() for lst in subs.values() for s in lst if "." not in s})


# ---------------------------------------------------------------- 받기(SEC 한 연결, 간격 유지)
class Sec:
    def __init__(self, gap: float):
        self.gap, self.last = max(gap, 1.0), 0.0

    def open(self, url: str, timeout: int = 120):
        wait = self.gap - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)
        except urllib.error.HTTPError as e:
            body = e.read()[:4000].decode("utf-8", "replace")
            title = re.search(r"<title>(.*?)</title>", body, re.S | re.I)
            _log(f"HTTP {e.code} {url} {title.group(1).strip() if title else ''}")
            if e.code in (403, 429):
                raise SystemExit(f"SEC가 HTTP {e.code}로 거절했다. 멈춘다(재시도 안 함): {url}")
            raise
        enc = (r.headers.get("Content-Encoding") or "").lower()
        if enc == "gzip":
            return gzip.GzipFile(fileobj=r)
        if enc == "deflate":
            return io.BytesIO(zlib.decompress(r.read()))
        return r

    def text(self, url: str) -> str:
        return self.open(url, timeout=60).read().decode("utf-8", "replace")

    def download(self, url: str, out: Path) -> None:
        part = out.with_suffix(".part")
        r = self.open(url, timeout=300)
        with open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        part.replace(out)


def zip_links(html: str) -> list[dict]:
    """랜딩 페이지의 zip 링크와 제출일 구간. 이름: 01jun2024-31aug2024_form13f.zip 또는 옛 2023q3_form13f.zip."""
    out = {}
    for href in re.findall(r'href="([^"]*?form13f[^"]*?\.zip)"', html, re.I):
        name = href.rsplit("/", 1)[-1].lower()
        m = re.match(r"(\d{1,2}[a-z]{3}\d{4})-(\d{1,2}[a-z]{3}\d{4})_form13f\.zip$", name)
        if m:
            lo, hi = (datetime.strptime(x, "%d%b%Y") for x in m.groups())
        else:
            m = re.match(r"(\d{4})q([1-4])_form13f\.zip$", name)
            if not m:
                continue
            p = pd.Period(f"{m.group(1)}Q{m.group(2)}", "Q")
            lo, hi = p.start_time.to_pydatetime(), p.end_time.normalize().to_pydatetime()
        url = href if href.startswith("http") else SEC + ("" if href.startswith("/") else "/") + href
        out[name] = {"name": name, "url": url, "lo": lo.strftime("%Y-%m-%d"), "hi": hi.strftime("%Y-%m-%d")}
    return sorted(out.values(), key=lambda z: z["lo"])


def _member(z: zipfile.ZipFile, base: str) -> str:
    for n in z.namelist():
        if n.rsplit("/", 1)[-1].upper() == base:
            return n
    raise KeyError(f"zip에 {base}가 없다: {z.namelist()}")


def _tsv(z: zipfile.ZipFile, base: str, cols: list[str], chunksize: int | None = None):
    want = set(cols)
    f = z.open(_member(z, base))
    return pd.read_csv(f, sep="\t", dtype=str, usecols=lambda c: c in want, quoting=csv.QUOTE_NONE,
                       keep_default_na=False, encoding="utf-8", encoding_errors="replace",
                       on_bad_lines="skip", chunksize=chunksize)


def _date(s: pd.Series) -> pd.Series:
    """'30-JUN-2024'(13F 자료 형식)과 ISO 둘 다 받는다."""
    d = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    rest = d.isna() & s.astype(str).str.len().gt(0)
    if rest.any():
        d[rest] = pd.to_datetime(s[rest], errors="coerce", format="mixed")
    return d.dt.strftime("%Y-%m-%d")


def hf_regex() -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for _, _, p in HF))


def process_zip(path: Path, outdir: Path) -> None:
    """zip 하나 -> sub.parquet(전체 제출 목록) + info.parquet(헤지펀드 후보 제출의 INFOTABLE 전 행)."""
    with zipfile.ZipFile(path) as z:
        sub = _tsv(z, "SUBMISSION.TSV", ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
        cov = _tsv(z, "COVERPAGE.TSV", ["ACCESSION_NUMBER", "REPORTCALENDARORQUARTER", "ISAMENDMENT", "AMENDMENTTYPE",
                                        "FILINGMANAGER_NAME", "REPORTTYPE"])
        summ = _tsv(z, "SUMMARYPAGE.TSV", ["ACCESSION_NUMBER", "TABLEENTRYTOTAL", "TABLEVALUETOTAL"])
        need = {"ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"}
        if not need <= set(sub.columns) or "FILINGMANAGER_NAME" not in cov.columns:
            raise ValueError(f"{path.name}: 열 이름이 예상과 다르다 {list(sub.columns)} {list(cov.columns)}")
        s = sub.merge(cov, on="ACCESSION_NUMBER", how="left").merge(summ, on="ACCESSION_NUMBER", how="left")
        col = lambda c: s[c].fillna("").astype(str) if c in s.columns else pd.Series("", index=s.index)  # noqa: E731
        s = pd.DataFrame({
            "accession": s["ACCESSION_NUMBER"], "cik": s["CIK"].str.lstrip("0"),
            "filing_date": _date(s["FILING_DATE"]), "subtype": s["SUBMISSIONTYPE"].str.upper(),
            "period": _date(col("PERIODOFREPORT").where(col("PERIODOFREPORT").str.len() > 0,
                                                        col("REPORTCALENDARORQUARTER"))),
            "manager": col("FILINGMANAGER_NAME"), "isamend": col("ISAMENDMENT").str.upper(),
            "amendtype": col("AMENDMENTTYPE").str.upper(), "reporttype": col("REPORTTYPE").str.upper(),
            "n_entries": pd.to_numeric(col("TABLEENTRYTOTAL"), errors="coerce"),
            "value_total": pd.to_numeric(col("TABLEVALUETOTAL"), errors="coerce"),
        })
        nm = _norm(s["manager"])
        cand = set(s.loc[nm.str.contains(hf_regex()) & ~nm.str.contains(HF_EXCLUDE), "accession"])
        parts = []
        cols = ["ACCESSION_NUMBER", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"]
        for ch in _tsv(z, "INFOTABLE.TSV", cols, chunksize=500_000):
            parts.append(ch[ch["ACCESSION_NUMBER"].isin(cand)])
        info = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    info.columns = [c.lower() for c in info.columns]
    info = info.rename(columns={"accession_number": "accession"})
    info["value"] = pd.to_numeric(info["value"], errors="coerce")
    info["sshprnamt"] = pd.to_numeric(info["sshprnamt"], errors="coerce")
    outdir.mkdir(parents=True, exist_ok=True)
    s.to_parquet(outdir / "sub.parquet", index=False)
    info.to_parquet(outdir / "info.parquet", index=False)  # 마지막에 써서 완료 표시를 겸한다
    _log(f"{path.name}: 제출 {len(s)}건, 헤지펀드 후보 제출 {len(cand)}건, 보유 행 {len(info)}")


def fetch_sec(gap: float, since: str, keep_zip: bool) -> list[str]:
    """필요한 zip을 받아 처리한다. 반환: 처리된 zip 이름."""
    sec = Sec(gap)
    links = zip_links(sec.text(LANDING))
    if not links:
        raise SystemExit("랜딩 페이지에서 zip 링크를 못 찾았다(페이지 구조 변경?). 멈춘다.")
    _log(f"zip 링크 {len(links)}개: {links[0]['name']} ~ {links[-1]['name']}")
    since_dt = pd.Timestamp(since)
    done = []
    for z in links:
        if pd.Timestamp(z["hi"]) <= since_dt:  # since 분기 보고서(마감 since+45일)는 since 뒤에 제출된다
            continue
        outdir = RAW / z["name"].removesuffix(".zip")
        if (outdir / "info.parquet").exists():
            done.append(z["name"])
            continue
        zp = RAW / z["name"]
        if not zp.exists():
            _log(f"받기 {z['url']}")
            sec.download(z["url"], zp)
        process_zip(zp, outdir)
        if not keep_zip:
            zp.unlink()
        done.append(z["name"])
    return done


# ---------------------------------------------------------------- Yahoo 가격(분기 말 원래 종가·분할)
def fetch_prices(tickers: list[str], since: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """일별 분할 반영 종가(넓은 표)와 분할 목록(ticker, date, ratio). 받은 것은 raw_sec13f/에 두고 다시 쓴다."""
    import yfinance as yf
    fc, fs = RAW / "yahoo_close.parquet", RAW / "yahoo_splits.parquet"
    close = pd.read_parquet(fc) if fc.exists() else pd.DataFrame()
    splits = pd.read_parquet(fs) if fs.exists() else pd.DataFrame(columns=["ticker", "date", "ratio"])
    old = (pd.Timestamp.now() - pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    if not close.empty and close.index.max() < old:  # 오래된 캐시: 새 거래일·새 분할이 없으니 전부 다시 받는다
        close, splits = pd.DataFrame(), pd.DataFrame(columns=["ticker", "date", "ratio"])
    need = [t for t in ["QQQ"] + tickers if t not in close.columns]
    if need:
        cols = {}
        for t in need:
            h = yf.Ticker(t).history(start=since, auto_adjust=False, actions=True)
            time.sleep(1.0)
            if h.empty:
                cols[t] = pd.Series(dtype=float)
                continue
            idx = h.index.strftime("%Y-%m-%d")
            cols[t] = pd.Series(h["Close"].to_numpy(), index=idx)
            sp = h.loc[h["Stock Splits"] > 0, "Stock Splits"]
            if len(sp):
                splits = pd.concat([splits, pd.DataFrame({"ticker": t, "date": sp.index.strftime("%Y-%m-%d"),
                                                          "ratio": sp.to_numpy()})], ignore_index=True)
        new = pd.DataFrame({k: v[~v.index.duplicated(keep="last")] for k, v in cols.items()})
        close = pd.concat([close, new], axis=1).sort_index()
        close.to_parquet(fc)
        splits.drop_duplicates().to_parquet(fs, index=False)
    return close, splits.drop_duplicates()


def raw_close_at(close: pd.DataFrame, splits: pd.DataFrame, qend: str) -> pd.Series:
    """분기 말(그날 이전 마지막 거래일) 종가를 그 시점 주식 수 기준(뒤 분할을 되돌린 값)으로."""
    upto = close.loc[close.index <= qend]
    if upto.empty:
        return pd.Series(np.nan, index=close.columns)
    c = upto.ffill(limit=5).iloc[-1]  # 상장 전(값 없음)은 NaN으로 남는다
    after = splits[splits["date"] > qend].groupby("ticker")["ratio"].prod()
    return c * after.reindex(c.index).fillna(1.0)


def split_between(splits: pd.DataFrame, a: str, b: str) -> pd.Series:
    """a < 분할일 <= b 분할 비율의 곱(티커별, 없으면 1)."""
    m = (splits["date"] > a) & (splits["date"] <= b)
    return splits[m].groupby("ticker")["ratio"].prod()


# ---------------------------------------------------------------- 시점 규칙
def edgar_deadline(qend: str) -> str:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    d = pd.Timestamp(qend) + pd.Timedelta(days=45)
    hol = set(USFederalHolidayCalendar().holidays(d - pd.Timedelta(days=5), d + pd.Timedelta(days=10)))
    while d.weekday() >= 5 or d in hol:
        d += pd.Timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def to_trading_day(d: str, days: list[str]) -> str | None:
    """마감일 d '다음' 거래일(d 오후 4시~5시 30분 접수분까지 공개된 첫 종가일). 아직 없으면 None."""
    later = [x for x in days if x > d]
    return later[0] if later else None


# ---------------------------------------------------------------- 계산
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    subs, infos = [], []
    for d in sorted(p for p in RAW.iterdir() if p.is_dir() and (p / "info.parquet").exists()):
        s = pd.read_parquet(d / "sub.parquet")
        s["zip"] = d.name
        subs.append(s)
        i = pd.read_parquet(d / "info.parquet")
        i["zip"] = d.name
        infos.append(i)
    if not subs:
        raise SystemExit("처리된 13F zip이 없다. 먼저 받아야 한다.")
    sub = pd.concat(subs, ignore_index=True)
    first_zip = sub.drop_duplicates("accession")[["accession", "zip"]]
    sub = sub.drop_duplicates("accession").drop(columns="zip")
    info = pd.concat(infos, ignore_index=True).merge(first_zip, on=["accession", "zip"])  # 겹치는 zip 중복 제거
    return sub, info.drop(columns="zip")


def resolve_funds(sub: pd.DataFrame) -> pd.DataFrame:
    """HF 표 -> (key, category, cik, 정확한 운용사 이름들)."""
    s = sub[sub["subtype"].isin(["13F-HR", "13F-HR/A"])].copy()
    s["nm"] = _norm(s["manager"])
    s = s[~s["nm"].str.contains(HF_EXCLUDE)]
    rows = []
    for key, cat, pat in HF:
        m = s[s["nm"].str.contains(pat, regex=True)]
        for cik, g in m.groupby("cik"):
            rows.append({"key": key, "category": cat, "cik": cik,
                         "names": " | ".join(sorted(set(g["manager"]))), "n_filings": len(g),
                         "first_period": g["period"].min(), "last_period": g["period"].max()})
        if m.empty:
            rows.append({"key": key, "category": cat, "cik": "", "names": "", "n_filings": 0,
                         "first_period": "", "last_period": ""})
    return pd.DataFrame(rows)


def select_filings(sub: pd.DataFrame, funds: pd.DataFrame, periods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(key, period)별로 마감일까지 낸 원본 + 수정본(RESTATEMENT는 대체, NEW HOLDINGS는 추가). 늦은 제출 기록도 반환."""
    f = funds[funds["cik"] != ""]
    s = sub.merge(f[["key", "category", "cik"]], on="cik")
    s["qend"] = pd.to_datetime(s["period"]).dt.to_period("Q").dt.end_time.dt.strftime("%Y-%m-%d")
    s = s[s["qend"].isin(periods) & s["subtype"].isin(["13F-HR", "13F-HR/A"])]
    s = s[s["reporttype"] != "13F NOTICE"]
    s["deadline"] = s["qend"].map({p: edgar_deadline(p) for p in periods})
    late = s[s["filing_date"] > s["deadline"]]
    s = s[s["filing_date"] <= s["deadline"]]
    use = []
    for (key, cik, q), g in s.groupby(["key", "cik", "qend"]):
        g = g.sort_values(["filing_date", "accession"])
        base = g[(g["subtype"] == "13F-HR") | (g["amendtype"] == "RESTATEMENT")]
        if base.empty:
            continue
        b = base.iloc[-1]
        adds = g[(g["subtype"] == "13F-HR/A") & (g["amendtype"] == "NEW HOLDINGS") & (g["filing_date"] >= b["filing_date"])]
        for _, r in pd.concat([b.to_frame().T, adds]).iterrows():
            use.append({"key": key, "category": r["category"], "cik": cik, "qend": q, "accession": r["accession"],
                        "filing_date": r["filing_date"], "deadline": r["deadline"], "original": r["accession"] == b["accession"]})
    return pd.DataFrame(use), late


def cusip_map(rows: pd.DataFrame, close: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """주식 행(풋/콜 아님, SH)의 CUSIP을 이름 정규식으로 티커에 붙이고 가격 검사로 확정한다."""
    sh = rows[(rows["kind"] == "sh") & (rows["sshprnamt"] > 0)].copy()
    sh["nm"] = _norm(sh["nameofissuer"])
    un = pd.Series(sh["nm"].unique())
    name2t: dict[str, str] = {}
    for t, p in ISSUER.items():  # 이름 하나가 두 티커에 걸리면 표 앞쪽 티커(가격 검사가 다시 거른다)
        for n in un[un.str.contains(p, regex=True)]:
            name2t.setdefault(n, t)
    sh["t"] = sh["nm"].map(name2t)
    hit = sh.groupby("cusip")["t"].agg(lambda x: x.value_counts().idxmax() if x.notna().any() else None)
    hit = hit[hit.notna()]
    out = []
    for cusip, g in sh[sh["cusip"].isin(hit.index)].groupby("cusip"):
        best = hit[cusip]
        share = float((g["t"] == best).mean())
        names = g["nm"].value_counts()
        if share < 0.5:
            continue
        ratios = []
        for q, gq in g.groupby("qend"):
            px = raw_close_at(close[[best]], splits, q).get(best, np.nan)
            if pd.notna(px) and px > 0:
                ratios.append(float(np.median(gq["value"] / gq["sshprnamt"])) / px)
        r = float(np.median(ratios)) if ratios else np.nan
        ok = bool(ratios) and PRICE_OK[0] <= r <= PRICE_OK[1]
        out.append({"cusip": cusip, "ticker": best, "issuer": names.index[0], "name_share": round(share, 3),
                    "n_rows": len(g), "n_quarters": g["qend"].nunique(), "price_ratio": r, "ok": ok})
    return pd.DataFrame(out)


def build(since: str) -> None:
    sub, info = load_raw()
    syms = ai_symbols()
    missing_pat = sorted(set(syms) - set(ISSUER))
    if missing_pat:
        raise SystemExit(f"ISSUER 표에 없는 AI 티커: {missing_pat}")
    funds = resolve_funds(sub)
    funds.to_csv(CACHE / "sec13f_funds.csv", index=False, encoding="utf-8-sig")

    close, splits = fetch_prices(syms, "2023-01-01")
    tdays = list(close["QQQ"].dropna().index)
    qs = pd.period_range(pd.Timestamp(since).to_period("Q"), pd.Timestamp(tdays[-1]).to_period("Q"), freq="Q")
    periods = [p.end_time.strftime("%Y-%m-%d") for p in qs]
    known = {p: to_trading_day(edgar_deadline(p), tdays) for p in periods}
    periods = [p for p in periods if known[p] is not None]  # 마감이 아직 안 온 분기는 뺀다

    used, late = select_filings(sub, funds, periods)
    used.to_csv(CACHE / "sec13f_filings_used.csv", index=False, encoding="utf-8-sig")
    late.to_csv(CACHE / "sec13f_filings_late.csv", index=False, encoding="utf-8-sig")

    rows = info.merge(used[["accession", "key", "category", "qend"]], on="accession")
    pc = rows["putcall"].fillna("").str.upper()
    rows["kind"] = np.where(pc.isin(["PUT", "CALL"]), pc.str.lower(),
                            np.where(rows["sshprnamttype"].fillna("").str.upper() == "SH", "sh", "other"))
    total_long = rows[rows["kind"] == "sh"].groupby(["qend", "key"])["value"].sum()

    cmap = cusip_map(rows, close, splits)
    cmap.to_csv(CACHE / "sec13f_cusip_map.csv", index=False, encoding="utf-8-sig")
    good = cmap[cmap["ok"]].set_index("cusip")["ticker"]
    ai = rows[rows["cusip"].isin(good.index) & rows["kind"].isin(["sh", "call", "put"])].copy()
    ai["symbol"] = ai["cusip"].map(good)
    px = pd.DataFrame({q: raw_close_at(close[syms], splits, q) for q in periods})  # 티커 x 분기
    ai["px"] = [px.at[s, q] if s in px.index else np.nan for s, q in zip(ai["symbol"], ai["qend"])]
    n_nopx = int(ai["px"].isna().sum())
    ai = ai[ai["px"].notna()]  # 분기 말에 미국에서 거래되지 않던 종목(상장 전·거래 정지)은 뺀다
    ratio = ai["value"] / ai["sshprnamt"] / ai["px"]
    bad = ratio.notna() & ~ratio.between(*ROW_OK)
    n_bad, v_bad = int(bad.sum()), float(ai.loc[bad, "value"].sum())
    ai = ai[~bad]

    pos = ai.pivot_table(index=["qend", "key", "category", "symbol"], columns="kind",
                         values=["value", "sshprnamt"], aggfunc="sum", fill_value=0.0)
    pos.columns = [f"{a}_{b}" for a, b in pos.columns]
    pos = pos.reindex(columns=[f"{a}_{b}" for a in ("value", "sshprnamt") for b in ("sh", "call", "put")], fill_value=0.0)
    pos = pos.reset_index()

    filed = used.groupby("qend")["key"].apply(set).to_dict()
    cat_of = dict(zip(funds["key"], funds["category"]))
    agg, bysym, fstats = [], [], []
    for i, q in enumerate(periods):
        cur = pos[pos["qend"] == q]
        r = {"known": known[q], "asof": q, "hf_n_funds_filed": float(len(filed.get(q, ())))}
        r["hf_total_long_value"] = float(total_long.loc[q].sum()) if q in total_long.index.get_level_values(0) else np.nan
        r["hf_ai_value"] = float(cur["value_sh"].sum())
        r["hf_ai_weight"] = r["hf_ai_value"] / r["hf_total_long_value"] if r["hf_total_long_value"] else np.nan
        r["hf_ai_n_pos"] = float((cur["sshprnamt_sh"] > 0).sum())
        r["hf_ai_n_funds"] = float(cur.loc[cur["sshprnamt_sh"] > 0, "key"].nunique())
        r["hf_ai_call_value"] = float(cur["value_call"].sum())
        r["hf_ai_put_value"] = float(cur["value_put"].sum())
        r["hf_ai_putcall"] = r["hf_ai_put_value"] / r["hf_ai_call_value"] if r["hf_ai_call_value"] > 0 else np.nan
        chg = None
        p0 = periods[i - 1] if i > 0 else None
        both = filed.get(q, set()) & filed.get(p0, set()) if p0 else set()
        if both:  # 직전 분기 자료가 없으면 순매수는 0이 아니라 빈칸
            prev =pos[(pos["qend"] == p0) & pos["key"].isin(both)][["key", "symbol", "sshprnamt_sh"]]
            prev = prev.assign(sh0=prev["sshprnamt_sh"] * prev["symbol"].map(split_between(splits, p0, q)).fillna(1.0))
            c = cur[cur["key"].isin(both)][["key", "symbol", "sshprnamt_sh"]].rename(columns={"sshprnamt_sh": "sh1"})
            chg = c.merge(prev[["key", "symbol", "sh0"]], on=["key", "symbol"], how="outer").fillna({"sh1": 0.0, "sh0": 0.0})
            chg["px"] = chg["symbol"].map(px[q])
            chg = chg[chg["px"].notna()]
            chg["d"] = (chg["sh1"] - chg["sh0"]) * chg["px"]
            chg["category"] = chg["key"].map(cat_of)
            per_fund = chg.groupby("key")["d"].sum()
            base = float((chg["sh0"] * chg["px"]).sum())
            r["hf_n_funds_both"] = float(len(both))
            r["hf_ai_shares_chg"] = float(chg["d"].sum())
            r["hf_ai_shares_chg_pct"] = r["hf_ai_shares_chg"] / base if base > 0 else np.nan
            r["hf_ai_n_funds_increasing"] = float((per_fund > 0).sum())
            r["hf_ai_n_funds_decreasing"] = float((per_fund < 0).sum())
        else:
            for k in ("hf_n_funds_both", "hf_ai_shares_chg", "hf_ai_shares_chg_pct",
                      "hf_ai_n_funds_increasing", "hf_ai_n_funds_decreasing"):
                r[k] = np.nan
        for cat in CATS:
            cc = cur[cur["category"] == cat]
            r[f"hf_ai_value_{cat}"] = float(cc["value_sh"].sum())
            r[f"hf_ai_call_value_{cat}"] = float(cc["value_call"].sum())
            r[f"hf_ai_put_value_{cat}"] = float(cc["value_put"].sum())
            r[f"hf_ai_shares_chg_{cat}"] = float(chg.loc[chg["category"] == cat, "d"].sum()) if chg is not None else np.nan
        agg.append(r)

        g = cur.groupby("symbol")
        b = pd.DataFrame({"hf_value": g["value_sh"].sum(), "hf_shares": g["sshprnamt_sh"].sum(),
                          "n_funds": cur[cur["sshprnamt_sh"] > 0].groupby("symbol")["key"].nunique(),
                          "hf_call_value": g["value_call"].sum(), "hf_put_value": g["value_put"].sum()})
        if chg is not None:
            gs = chg.groupby("symbol")["d"]
            b = b.join(pd.DataFrame({"hf_shares_chg": gs.sum(), "n_funds_increasing": gs.apply(lambda x: float((x > 0).sum())),
                                     "n_funds_decreasing": gs.apply(lambda x: float((x < 0).sum()))}), how="outer")
        fill = ["hf_value", "hf_shares", "n_funds", "hf_call_value", "hf_put_value"]
        b[fill] = b[fill].fillna(0.0)  # 다 판 종목(변화만 있는 행)은 보유 0
        b["price"] = px[q].reindex(b.index)
        b = b.reset_index().rename(columns={"index": "symbol"})
        b.insert(0, "asof", q)
        b.insert(0, "known", known[q])
        bysym.append(b)

        u = used[(used["qend"] == q) & used["original"].astype(bool)]
        if len(u):
            fd = pd.to_datetime(u["filing_date"])
            fstats.append({"asof": q, "deadline": edgar_deadline(q), "known": known[q], "n_filings": len(u),
                           "median_filing_date": fd.median().strftime("%Y-%m-%d"),
                           "share_on_deadline": float((u["filing_date"] == u["deadline"]).mean()),
                           "n_late_original": int(((late["qend"] == q) & (late["subtype"] == "13F-HR")).sum()),
                           "n_late_amendment": int(((late["qend"] == q) & (late["subtype"] == "13F-HR/A")).sum())})

    out = pd.DataFrame(agg).set_index("known").sort_index()
    nf = out["hf_n_funds_filed"]
    full = (nf > 0) & (nf >= 0.8 * nf[nf > 0].median())  # 제출 0건 분기(자료 집합 미공개)는 0이 아니라 빈칸
    if (~full).any():  # 자료 집합이 아직 그 분기 제출을 다 담지 않은 경우
        _log(f"제출 펀드 수가 모자라 뺀 분기: {list(out.loc[~full, 'asof'])}")
    out = out[full]
    out = out[~out.index.duplicated(keep="last")]
    out.to_parquet(CACHE / "sec13f.parquet")
    bs = pd.concat(bysym, ignore_index=True)
    bs = bs[bs["asof"].isin(out["asof"])]
    bs = bs.reindex(columns=["known", "asof", "symbol", "hf_value", "hf_shares", "n_funds", "hf_shares_chg",
                             "hf_call_value", "hf_put_value", "n_funds_increasing", "n_funds_decreasing", "price"])
    bs.sort_values(["known", "symbol"]).to_parquet(CACHE / "sec13f_by_symbol.parquet", index=False)
    pd.DataFrame(fstats).to_csv(CACHE / "sec13f_filing_dates.csv", index=False, encoding="utf-8-sig")
    unm = sorted(set(syms) - set(good))
    _log(f"완료: 분기 {len(out)}개({out.index[0]}~{out.index[-1]}), 펀드 key {funds.loc[funds['cik'] != '', 'key'].nunique()}개, "
         f"CUSIP 채택 {len(good)}개, 못 붙인 티커 {unm}, 가격 없는 행 제외 {n_nopx}개, 큰 오류 행 제외 {n_bad}개(${v_bad:,.0f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=1.0, help="SEC 요청 사이 초(최소 1)")
    ap.add_argument("--since", default="2023-06-30", help="첫 보고 분기 말(이 분기는 순매수 기준점)")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--offline", action="store_true", help="받지 않고 이미 처리한 zip만으로 계산")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if not a.offline:
        fetch_sec(a.gap, a.since, a.keep_zip)
    build(a.since)


if __name__ == "__main__":
    main()
