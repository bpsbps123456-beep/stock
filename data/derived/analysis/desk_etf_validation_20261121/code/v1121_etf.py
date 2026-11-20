"""TIME·KoAct issuer market closes through KRX 2026-11-20, using desk_etf_validation_20261120's own
download and pairing rules (US session t -> the first KRX close after calendar day t).

py v1121_etf.py --refresh   re-downloads ../results/issuer_closes.csv from the two issuers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OLD = HERE.parents[1] / "desk_etf_validation_20261120"
AS_OF_KRX = pd.Timestamp("2026-11-20")


def _old_module():
    spec = importlib.util.spec_from_file_location("etf_validation_20261120", OLD / "compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NAVER = "https://api.finance.naver.com/siseJson.naver"
CODES = {"time": "456600", "koact": "0015B0"}


def naver(code):
    r = requests.get(NAVER, params=dict(symbol=code, requestType=1, startTime="20251229",
                                       endTime=AS_OF_KRX.strftime("%Y%m%d"), timeframe="day"),
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    rows = json.loads(r.text.strip().replace("'", '"'))[1:]
    return pd.Series({pd.Timestamp(row[0]): float(row[4]) for row in rows})


def refresh():
    """Issuer closes through 11/19 exactly as before; the 11/20 close from Naver's KRX daily chart,
    accepted only if Naver equals the issuer close on every earlier session of the study (KoAct's
    issuer API had not published 11/20 yet; TIME's issuer table is checked against it as well)."""
    mod = _old_module()
    RESULTS.mkdir(exist_ok=True)
    mod.HERE, mod.AS_OF_KRX = RESULTS, pd.Timestamp("2026-11-19")
    mod.download_quotes()
    issuer = pd.read_csv(RESULTS / "issuer_closes.csv", index_col="krx_date", parse_dates=True).astype(float)
    old = pd.read_csv(OLD / "issuer_closes.csv", index_col="krx_date", parse_dates=True).astype(float)
    pd.testing.assert_frame_equal(issuer, old)  # the re-download agrees with the 11/20 freeze
    sources, row = {}, {}
    for key, code in CODES.items():
        series = naver(code)
        common = issuer.index.intersection(series.index)
        if len(common) != len(issuer) or not np.array_equal(series.loc[common], issuer.loc[common, key]):
            raise ValueError(f"Naver and issuer closes disagree for {key}")
        row[key] = series.at[AS_OF_KRX]
        sources[key] = f"Naver KRX daily chart, identical to the issuer on all {len(common)} earlier sessions"
    r = requests.get(mod.TIME_URL, params={"idx": "6", "navStartDate": "2026-11-20", "navEndDate": "2026-11-20"},
                     timeout=30)
    doc = BeautifulSoup(r.content, "html.parser")
    for table in doc.select("table"):
        if "시장가격(원)" in table.get_text(strip=True)[:250]:
            for tr in table.select("tbody tr"):
                cells = [c.get_text(strip=True) for c in tr.select("td")]
                if cells and cells[0] == "2026.11.20":
                    if float(cells[3].replace(",", "")) != row["time"]:
                        raise ValueError("TIME issuer and Naver 11/20 closes disagree")
                    sources["time"] += "; TIME issuer table shows the same 11/20 close"
    full = pd.concat([issuer, pd.DataFrame([row], index=pd.DatetimeIndex([AS_OF_KRX], name="krx_date"))])
    full.index.name = "krx_date"
    full.astype(int).to_csv(RESULTS / "issuer_closes.csv", lineterminator="\n")
    (RESULTS / "issuer_closes_sources.json").write_text(
        json.dumps({"through_2026-11-19": "issuer pages (desk_etf_validation_20261120 download_quotes)",
                    "2026-11-20": sources}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def closes():
    kr = pd.read_csv(RESULTS / "issuer_closes.csv", index_col="krx_date", parse_dates=True)
    if kr.index[-1] != AS_OF_KRX:
        raise ValueError("Run with --refresh to fetch the 11/20 KRX close")
    return kr


def pair(kr, us_dates):
    """Index positions of the first KRX close after each US date; drop US dates not yet priced."""
    us = pd.DatetimeIndex(us_dates)
    found = kr.index.searchsorted(us, side="right")
    arrived = found < len(kr)
    if not (us[arrived] < kr.index[found[arrived]]).all():
        raise ValueError("Same-day or earlier Korean close matched to a US session")
    return us[arrived], kr.index[found[arrived]]


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--refresh", action="store_true")
    if arg.parse_args().refresh:
        refresh()
    kr = closes()
    print(kr.tail(4))
    us, krx = pair(kr, ["2025-12-31", "2026-06-30", "2026-11-18", "2026-11-19"])
    print(list(zip(us.date, krx.date)))
    assert np.isfinite(kr.to_numpy()).all()
