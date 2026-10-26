"""Probe KRX data.krx.co.kr for the ETF full-list quote table (순자산총액) on given dates and sum
single-stock leveraged ETFs on Samsung/SK Hynix.
Usage: py krx_etf_probe.py 20260622 20260728 20261002 20261013
"""
import json
import os
import sys

import pandas as pd
import requests

RESULTS = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis\leverage_flood\results"
URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201030101",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
}


SESSION = requests.Session()


def fetch_etf_all(date):
    if not SESSION.cookies:
        SESSION.get("https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201030101",
                    headers={"User-Agent": HEADERS["User-Agent"]}, timeout=60)
    payload = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT04301",
        "locale": "ko_KR",
        "trdDd": date,
        "share": "1",
        "money": "1",
        "csvxls_isNo": "false",
    }
    r = SESSION.post(URL, data=payload, headers=HEADERS, timeout=60)
    if r.status_code != 200:
        print("   status", r.status_code, r.text[:300])
    r.raise_for_status()
    js = r.json()
    key = next((k for k in js if isinstance(js[k], list)), None)
    return pd.DataFrame(js[key]) if key else pd.DataFrame()


def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").replace({"-": None}), errors="coerce")


def main():
    dates = sys.argv[1:] or ["20260622"]
    rows = []
    for d in dates:
        try:
            df = fetch_etf_all(d)
        except Exception as exc:  # noqa: BLE001
            print(d, "error", exc)
            continue
        if df.empty:
            print(d, "empty")
            continue
        print(d, "columns:", list(df.columns)[:20], "rows", len(df))
        name_col = next((c for c in df.columns if "NM" in c and "ISU" in c), None)
        nav_col = next((c for c in df.columns if "NETASST" in c or "NAV_TOT" in c or "NETAST" in c), None)
        val_col = next((c for c in df.columns if "ACC_TRDVAL" in c), None)
        print("   name_col", name_col, "nav_col", nav_col, "val_col", val_col)
        if not name_col or not nav_col:
            df.head(3).to_csv(os.path.join(RESULTS, f"krx_etf_sample_{d}.csv"), index=False, encoding="utf-8-sig")
            continue
        df["nav"] = to_num(df[nav_col])
        df["trdval"] = to_num(df[val_col]) if val_col else None
        mask = df[name_col].str.contains("삼성전자|SK하이닉스", na=False) & df[name_col].str.contains("2X|레버리지|인버스", na=False)
        sub = df[mask].copy()
        sub.to_csv(os.path.join(RESULTS, f"krx_single_stock_lev_etf_{d}.csv"), index=False, encoding="utf-8-sig")
        lev = sub[~sub[name_col].str.contains("인버스", na=False)]
        inv = sub[sub[name_col].str.contains("인버스", na=False)]
        rows.append({"date": d, "n_lev": len(lev), "lev_nav_won": lev["nav"].sum(), "lev_trdval_won": lev["trdval"].sum() if val_col else None,
                     "n_inv": len(inv), "inv_nav_won": inv["nav"].sum()})
        print("   ", rows[-1])
        print(sub[[name_col, "nav", "trdval"]].to_string(index=False)[:3000])
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "krx_single_stock_lev_etf_summary.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
