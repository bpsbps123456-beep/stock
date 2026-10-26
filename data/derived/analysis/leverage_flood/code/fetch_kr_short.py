"""한국 공매도·대차·외국인 파생 지표 수집 (leverage_flood 연구용).

사용:  py fetch_kr_short.py [stock|market|program|lending|lending_stock|adr|krx|all]

결과는 ../results/kr_short_*.csv 로 저장한다. 받지 못한 항목은 CSV를 만들지 않고 stderr에 이유를 남긴다.
모든 숫자는 받은 값 그대로 쓰고, 환산 열(조원 등)은 열 이름에 단위를 적는다.

소스 (2026-10-14 확인)
  네이버 PC 증권(stock.naver.com)  — 비공식 JSON, 로그인 불필요, 당일 장 마감 후(약 18시) 확정
    /api/domestic/detail/<코드>/trend?tradeType=KRX&startIdx=<페이지>&pageSize=200
        종목별 투자자 순매수(주)·외국인 보유주수·보유비율. startIdx는 페이지 번호(0,1,2…), 1페이지 200거래일.
    /api/domestic/market/trend/daily?tradeType=KRX&marketType=<KOSPI|KOSDAQ|FUT>&bizdate=<오늘>&startIdx=<페이지>&pageSize=200
        시장 전체 투자자별 매매. KOSPI/KOSDAQ는 금액(원), FUT(코스피200 선물)는 계약수(quant)·백만원(price).
        investorGubun: 8000 개인, 9000 외국인, 9999 기관계, 1000 금융투자, 2000 보험, 3000 투신(사모), 4000 은행,
        5000 기타금융, 6000 연기금등, 7100 기타법인, 9001 기타외국인(주식만). 옵션 투자자별은 제공하지 않는다.
    /api/domestic/market/trendProgram?tradeType=KRX&krxMarketType=KOSPI&bizdate=<오늘>&startIdx=<페이지>&pageSize=200&periodType=DAY
        프로그램 매매 차익(diff*)·비차익(biDiff*)·합계(total*) 매수·매도·순매수(원).
    (네이버 모바일 m.stock.naver.com/api/stock/<코드>/trend 는 pageSize 60 초과·page 지정이 모두 400이라 60일만 준다.)
  금융투자협회 freesis  POST https://freesis.kofia.or.kr/meta/getMetaDataList.do   — 대차거래(예탁결제원 집계 재게시)
    STATSCU0100000170BO 대차거래 전체 일별: TMPV1 날짜, TMPV2 체결주수, TMPV3 상환주수, TMPV4 잔고주수, TMPV5 잔고금액(천만원).
    STATSCU0100000180BO 종목별 대차거래(하루 스냅샷): TMPV1 종목명, TMPV2 코드, TMPV3 체결, TMPV4 상환, TMPV5 잔고주수, TMPV6 잔고금액(천만원).
        단위 검증: 10/13 삼성전자 잔고 79,729,671주 × 종가 232,000 = 18.497조 = TMPV6 1,849,728 × 1e7. 마지막 두 행은 '합계'·'전체대비'.
  yfinance  SMSN.IL(삼성전자 GDR, 런던, USD, 1GDR=25주) / 005930.KS / KRW=X,  HY9H.F(SK하이닉스 GDR 프랑크푸르트, EUR) / 000660.KS / EURKRW=X
KRX data.krx.co.kr(getJsonData.cmd·generate.cmd)과 short.krx.co.kr(공매도 통합포털)은 2026-10 현재 모든 조회가 로그인 필수
('LOGOUT' / '로그인 또는 회원가입이 필요합니다')라 종목별 공매도 잔고·공매도 거래대금 비중·옵션 투자자별은 이 스크립트로 못 받는다.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
START = "20250601"
END = date.today().strftime("%Y%m%d")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
NAVER = "https://stock.naver.com"
CODES = {"005930": "삼성전자", "000660": "SK하이닉스"}
GUBUN = {"8000": "indiv", "9000": "foreign", "9999": "inst", "1000": "fin_invest", "2000": "insurance", "3000": "trust_private",
         "4000": "bank", "5000": "other_fin", "6000": "pension", "7000": "gov", "7100": "other_corp", "9001": "other_foreign"}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    log(f"saved {path} rows={len(rows)}")


def num(s):
    if s is None or s == "":
        return ""
    if isinstance(s, (int, float)):
        return s
    s = str(s).replace(",", "").replace("+", "").replace("%", "").strip()
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return s


def naver_get(path: str, referer: str, **params):
    for attempt in range(4):
        try:
            r = requests.get(NAVER + path, params=params, headers={**UA, "Referer": referer, "Accept": "application/json"}, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as ex:  # noqa: BLE001
            log(f"naver {path} retry {attempt}: {ex}")
            time.sleep(1 + attempt)
    raise RuntimeError(f"naver {path} failed")


# ---------------------------------------------------------------- ① 종목별 외국인 보유·순매수 (네이버 PC)
def fetch_stock_trend(code: str) -> list[dict]:
    out = []
    for page in range(0, 6):
        j = naver_get(f"/api/domestic/detail/{code}/trend", f"{NAVER}/domestic/stock/{code}/investmentinfo", tradeType="KRX", startIdx=page, pageSize=200)
        rows = j if isinstance(j, list) else j.get("content", [])
        if not rows:
            break
        for x in rows:
            out.append({"date": x["bizdate"], "code": code, "name": CODES.get(code, ""), "close": num(x.get("closePrice")), "chg": num(x.get("prevChangePrice")),
                        "volume": num(x.get("tradeVolume")), "foreign_net_shares": num(x.get("foreignerPureBuyQuant")),
                        "foreign_hold_shares": num(x.get("frgnStock")), "foreign_hold_ratio_pct": num(x.get("frgnHoldRatio")),
                        "inst_net_shares": num(x.get("organPureBuyQuant")), "indiv_net_shares": num(x.get("individualPureBuyQuant"))})
        if rows[-1]["bizdate"] < START:
            break
        time.sleep(0.4)
    return sorted([r for r in out if r["date"] >= START], key=lambda r: r["date"])


def do_stock():
    rows = []
    for code in CODES:
        rows += fetch_stock_trend(code)
    write_csv(RESULTS / "kr_short_foreign_stock.csv", rows,
              ["date", "code", "name", "close", "chg", "volume", "foreign_net_shares", "foreign_hold_shares", "foreign_hold_ratio_pct", "inst_net_shares", "indiv_net_shares"])


# ---------------------------------------------------------------- ② 시장·선물 투자자별 (네이버 PC)
def fetch_market_trend(market: str) -> list[dict]:
    """market: KOSPI/KOSDAQ(금액 원, quant 주) 또는 FUT(코스피200 선물, quant 계약, price 백만원)."""
    out = []
    for page in range(0, 6):
        j = naver_get("/api/domestic/market/trend/daily", f"{NAVER}/market/stock/kr/trend/trader", tradeType="KRX", marketType=market, bizdate=END, startIdx=page, pageSize=200)
        rows = j.get("content", [])
        if not rows:
            break
        for x in rows:
            rec = {"date": x["bizdate"], "market": market}
            for g in x.get("netAmounts", []):
                k = GUBUN.get(g["investorGubun"], g["investorGubun"])
                rec[f"{k}_net"] = num(g.get("diffValue"))
                rec[f"{k}_buy_q"] = num(g.get("buyQuant"))
                rec[f"{k}_sell_q"] = num(g.get("sellQuant"))
                rec[f"{k}_buy_amt"] = num(g.get("buyPrice"))
                rec[f"{k}_sell_amt"] = num(g.get("sellPrice"))
            out.append(rec)
        if rows[-1]["bizdate"] < START:
            break
        time.sleep(0.4)
    return sorted([r for r in out if START <= r["date"] <= END], key=lambda r: r["date"])


def do_market():
    # 코스피 현물 투자자별 순매수(원)
    rows = fetch_market_trend("KOSPI")
    keys = ["indiv", "foreign", "other_foreign", "inst", "fin_invest", "insurance", "trust_private", "bank", "other_fin", "pension", "other_corp"]
    fields = ["date", "market"] + [f"{k}_net" for k in keys]
    for r in rows:
        for k in keys:
            v = r.get(f"{k}_net", "")
            r[f"{k}_net"] = round(v / 1e8, 1) if isinstance(v, (int, float)) else v  # 억원
    write_csv(RESULTS / "kr_short_investor_kospi_100m_krw.csv", rows, fields)
    # 코스피200 선물 투자자별 (계약수). diffValue = 순매수 계약수, buy/sell quant = 계약수, amt = 백만원
    fut = fetch_market_trend("FUT")
    fkeys = ["indiv", "foreign", "inst", "fin_invest", "insurance", "trust_private", "bank", "other_fin", "pension", "other_corp"]
    cum = 0
    for r in fut:
        v = r.get("foreign_net", 0)
        cum += v if isinstance(v, (int, float)) else 0
        r["foreign_net_cum_from_start"] = cum
        tot_q = r.get("inst_buy_q", 0)  # 9999 기관계 행은 전체 거래량(매수=매도)이 들어 있다
        r["total_contracts"] = tot_q
    ffields = ["date", "market"] + [f"{k}_net" for k in fkeys] + ["foreign_buy_q", "foreign_sell_q", "foreign_buy_amt", "foreign_sell_amt", "total_contracts", "foreign_net_cum_from_start"]
    write_csv(RESULTS / "kr_short_foreign_futures_k200.csv", fut, ffields)


# ---------------------------------------------------------------- ③ 프로그램 매매 (네이버 PC)
def do_program():
    out = []
    for page in range(0, 6):
        j = naver_get("/api/domestic/market/trendProgram", f"{NAVER}/market/stock/kr/trend/program", tradeType="KRX", krxMarketType="KOSPI", bizdate=END, startIdx=page, pageSize=200, periodType="DAY")
        rows = j.get("content", [])
        if not rows:
            break
        for x in rows:
            out.append({"date": x["bizdate"],
                        "arb_buy_100m": round(num(x["diffBuyAmt"]) / 1e8, 1), "arb_sell_100m": round(num(x["diffSellAmt"]) / 1e8, 1), "arb_net_100m": round(num(x["diffPureBuyAmt"]) / 1e8, 1),
                        "nonarb_buy_100m": round(num(x["biDiffBuyAmt"]) / 1e8, 1), "nonarb_sell_100m": round(num(x["biDiffSellAmt"]) / 1e8, 1), "nonarb_net_100m": round(num(x["biDiffPureBuyAmt"]) / 1e8, 1),
                        "total_buy_100m": round(num(x["totalDiffBuyAmt"]) / 1e8, 1), "total_sell_100m": round(num(x["totalDiffSellAmt"]) / 1e8, 1), "total_net_100m": round(num(x["totalDiffPureBuyAmt"]) / 1e8, 1)})
        if rows[-1]["bizdate"] < START:
            break
        time.sleep(0.4)
    out = sorted([r for r in out if START <= r["date"] <= END], key=lambda r: r["date"])
    write_csv(RESULTS / "kr_short_program_kospi_100m_krw.csv", out,
              ["date", "arb_buy_100m", "arb_sell_100m", "arb_net_100m", "nonarb_buy_100m", "nonarb_sell_100m", "nonarb_net_100m", "total_buy_100m", "total_sell_100m", "total_net_100m"])


# ---------------------------------------------------------------- ④ 금투협 대차
KOFIA_URL = "https://freesis.kofia.or.kr/meta/getMetaDataList.do"


def kofia(obj: str, s: str, e: str, freq: str = "D") -> list[dict]:
    body = {"dmSearch": {"tmpV40": "10000000", "tmpV41": "1", "tmpV1": freq, "tmpV45": s, "tmpV46": e, "OBJ_NM": obj}}
    for attempt in range(4):
        try:
            r = requests.post(KOFIA_URL, json=body, headers={**UA, "Referer": "https://freesis.kofia.or.kr/"}, timeout=90)
            j = r.json()
            return j.get("ds1", [])
        except Exception as ex:  # noqa: BLE001
            log(f"kofia {obj} {s}-{e} retry {attempt}: {ex}")
            time.sleep(2 + attempt * 2)
    return []


def do_lending_total():
    ds = kofia("STATSCU0100000170BO", START, END)
    rows = []
    for x in ds:
        d = str(x.get("TMPV1", ""))
        if not d.isdigit():
            continue  # 합계/평균 행 제외
        rows.append({"date": d, "lend_shares": x.get("TMPV2"), "return_shares": x.get("TMPV3"),
                     "balance_shares": x.get("TMPV4"), "balance_amount_raw_10m_krw": x.get("TMPV5")})
    rows.sort(key=lambda r: r["date"])
    for r in rows:
        try:
            r["balance_amount_trillion_krw"] = round(float(r["balance_amount_raw_10m_krw"]) * 1e7 / 1e12, 3)
        except (TypeError, ValueError):
            r["balance_amount_trillion_krw"] = ""
    write_csv(RESULTS / "kr_short_lending_total.csv", rows,
              ["date", "lend_shares", "return_shares", "balance_shares", "balance_amount_raw_10m_krw", "balance_amount_trillion_krw"])


def trading_days(start: str, end: str) -> list[str]:
    """네이버 삼성전자 종목 trend의 bizdate 목록을 거래일 달력으로 쓴다."""
    return sorted({r["date"] for r in fetch_stock_trend("005930") if start <= r["date"] <= end})


def do_lending_stock(codes=tuple(CODES)):
    days = trading_days(START, END)
    cache = HERE / "_cache_lending_stock.json"
    done: dict = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    for i, d in enumerate(days):
        if d in done:
            continue
        ds = kofia("STATSCU0100000180BO", d, d)
        rec = {"n_rows": len(ds)}
        for x in ds:
            if x.get("TMPV1") == "합계":
                rec["total"] = {"lend": x.get("TMPV3"), "ret": x.get("TMPV4"), "bal_shares": x.get("TMPV5"), "bal_amt_raw": x.get("TMPV6")}
            if x.get("TMPV2") in codes:
                rec[x["TMPV2"]] = {"name": x.get("TMPV1"), "lend": x.get("TMPV3"), "ret": x.get("TMPV4"),
                                   "bal_shares": x.get("TMPV5"), "bal_amt_raw": x.get("TMPV6")}
        done[d] = rec
        if i % 10 == 0:
            cache.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
            log(f"lending_stock {d} ({i+1}/{len(days)}) n={len(ds)}")
        time.sleep(0.3)
    cache.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    rows = []
    for d in sorted(done):
        rec = done[d]
        tot = rec.get("total", {})
        for c in codes:
            x = rec.get(c)
            if not x:
                continue
            try:
                bal_t = round(float(x["bal_amt_raw"]) * 1e7 / 1e12, 3)
            except (TypeError, ValueError):
                bal_t = ""
            rows.append({"date": d, "code": c, "name": x["name"], "lend_shares": x["lend"], "return_shares": x["ret"],
                         "balance_shares": x["bal_shares"], "balance_amount_raw_10m_krw": x["bal_amt_raw"], "balance_amount_trillion_krw": bal_t,
                         "all_stocks_balance_amount_raw_10m_krw": tot.get("bal_amt_raw", ""), "all_stocks_balance_shares": tot.get("bal_shares", ""), "n_rows": rec["n_rows"]})
    write_csv(RESULTS / "kr_short_lending_stock.csv", rows,
              ["date", "code", "name", "lend_shares", "return_shares", "balance_shares", "balance_amount_raw_10m_krw", "balance_amount_trillion_krw",
               "all_stocks_balance_amount_raw_10m_krw", "all_stocks_balance_shares", "n_rows"])


# ---------------------------------------------------------------- ⑤ GDR 프리미엄
def do_adr():
    import pandas as pd
    import yfinance as yf

    def hist(t):
        h = yf.Ticker(t).history(start="2025-05-20", end=(date.today() + timedelta(days=1)).isoformat(), auto_adjust=False)
        if h.empty:
            log(f"yfinance {t}: empty")
            return pd.Series(dtype=float), pd.Series(dtype=float)
        h.index = pd.to_datetime(h.index.date)
        return h["Close"], h["Volume"]

    smsn, smsn_vol = hist("SMSN.IL")      # 삼성전자 GDR, LSE, USD. 1 GDR = 25주 (2018년 50:1 액면분할 뒤 비율)
    sec, _ = hist("005930.KS")
    fx, _ = hist("KRW=X")
    hy_f, hy_f_vol = hist("HY9H.F")        # SK하이닉스 GDR 프랑크푸르트(EUR, 유동성 낮음)
    hyn, _ = hist("000660.KS")
    eurkrw, _ = hist("EURKRW=X")
    df = pd.DataFrame({"smsn_usd": smsn, "smsn_vol": smsn_vol, "sec_krw": sec, "usdkrw": fx,
                       "hy9h_eur": hy_f, "hy9h_vol": hy_f_vol, "hynix_krw": hyn, "eurkrw": eurkrw})
    df = df[df.index >= "2025-06-01"]
    # 런던 종가(현지 16:30 = 한국 다음날 00:30/01:30)와 같은 달력일의 한국 종가를 비교: GDR이 한국 장 마감 뒤에 거래되므로
    # 같은 날짜의 GDR 종가는 "그날 한국 종가 대비 해외가 더 비싸게/싸게 거래했나"를 뜻한다. 한국 휴장일은 직전 종가를 쓴다(ffill).
    df["sec_krw_ff"] = df["sec_krw"].ffill()
    df["usdkrw_ff"] = df["usdkrw"].ffill()
    df["sec_fair_usd"] = df["sec_krw_ff"] * 25 / df["usdkrw_ff"]
    df["samsung_gdr_premium_pct"] = (df["smsn_usd"] / df["sec_fair_usd"] - 1) * 100
    df["hynix_krw_ff"] = df["hynix_krw"].ffill()
    df["eurkrw_ff"] = df["eurkrw"].ffill()
    df["hynix_fair_eur"] = df["hynix_krw_ff"] / df["eurkrw_ff"]
    df["hynix_gdr_premium_pct"] = (df["hy9h_eur"] / df["hynix_fair_eur"] - 1) * 100
    df = df[df["smsn_usd"].notna()]
    out = df.round(3).reset_index().rename(columns={"index": "date"})
    out["date"] = out["date"].dt.strftime("%Y%m%d")
    out.to_csv(RESULTS / "kr_short_adr_premium.csv", index=False, encoding="utf-8-sig")
    log(f"saved adr rows={len(out)}")


# ---------------------------------------------------------------- ⑥ KRX 공매도 잔고 (로그인 필수 확인용)
def do_krx():
    url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    params = {"bld": "dbms/MDC/STAT/srt/MDCSTAT30502", "locale": "ko_KR", "isuCd": "KR7005930003", "strtDd": START, "endDd": END, "share": "1", "money": "1", "csvxls_isNo": "false"}
    try:
        r = requests.post(url, data=params, headers={**UA, "Referer": "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020301"}, timeout=30)
        body = r.text.strip()
    except Exception as ex:  # noqa: BLE001
        log(f"krx error: {ex}")
        return
    if body == "LOGOUT" or r.status_code != 200:
        log(f"krx 공매도 잔고: 못 받음 — status={r.status_code} body={body[:40]!r} (로그인 필수)")
        return
    try:
        j = r.json()
    except ValueError:
        log(f"krx: JSON 아님 {body[:80]!r}")
        return
    key = next((k for k in j if isinstance(j[k], list)), None)
    rows = j.get(key, []) if key else []
    if rows:
        write_csv(RESULTS / "kr_short_krx_short_balance_005930.csv", list(reversed(rows)), list(rows[0].keys()))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"stock": do_stock, "market": do_market, "program": do_program, "lending": do_lending_total, "adr": do_adr, "krx": do_krx, "lending_stock": do_lending_stock}
    if what == "all":
        for name, fn in steps.items():
            try:
                fn()
            except Exception as ex:  # noqa: BLE001
                log(f"{name}: 못 받음 — {ex}")
    else:
        steps[what]()
