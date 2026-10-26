"""분류 후보표: ETF·ETN·레버리지 단어가 든 ISIN 전부 + 첫·끝 등장, 금액, OpenFIGI 정보, 사슬 접미사. 가격 미사용."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "derived"

CHAIN_RE = re.compile(r"\s(SPLR|MRGR|EXOF|CHAN|SOFF)\s\d+\s([A-Z]{2}[A-Z0-9]{9}\d)\s*$")
LEV_WORDS = re.compile(
    r"\d(?:\.\d+)?\s?X\b|\bX\d|ULTRA|\bBULL|\bBEAR|\bBU\b|\bBE\b|INVERSE|\bSHORT|\bSHT\b|\bSRT\b|\bLONG\b|\bLNG\b|LEVERAG|\bLEV\b|"
    r"DIREX|DIRXN|\bDXN\b|PROSHARES|PROETF|GRANITE|GRNTSHR|GNTSH|T-REX|\bTREX\b|\bREX\b|DEFIANCE|TRADR|LEVERAGE SHARES|"
    r"ROUNDHILL|RUNDH|VOLATILITY SHARES|MICROSECTORS|\bAXS\b|TUTTLE|VS TRUST|\bETF\b|\bETN\b|\bETP\b|FUND|\bTRUST\b|\bSHS\b|"
    r"YIELDMAX|\bYL\b|ISHARES|SPDR|VANGUARD|INVESCO|SCHWAB|GLOBAL X|ARK |KRANE|WISDOMTREE|FIRST TRUST|JP MORGAN|JPMORGAN|PIMCO",
    re.I)


def main() -> None:
    figi = json.loads((DER / "figi.json").read_text(encoding="utf-8"))
    rows = []
    for kind, f, amt in [("weekly", "settle_weekly.parquet", "tot"), ("daily", "settle_daily.parquet", "tot"),
                         ("custody", "custody.parquet", "amt")]:
        p = pd.read_parquet(DER / f)
        dcol = "start" if "start" in p.columns else "date"
        g = p.groupby("isin").agg(first=(dcol, "min"), last=(dcol, "max"), n=(dcol, "nunique"), amount=(amt, "sum"),
                                  name=("name", "last"))
        g["src"] = kind
        rows.append(g.reset_index())
    a = pd.concat(rows)
    agg = a.groupby("isin").agg(first_seen=("first", "min"), last_seen=("last", "max"), seibro_name=("name", "last"))
    for kind in ["weekly", "daily", "custody"]:
        sub = a[a["src"] == kind].set_index("isin")
        agg[f"n_{kind}"] = sub["n"]
        agg[f"amt_{kind}"] = sub["amount"]
    agg = agg.reset_index()
    agg["figi_ticker"] = agg["isin"].map(lambda i: (figi.get(i) or {}).get("ticker"))
    agg["figi_name"] = agg["isin"].map(lambda i: (figi.get(i) or {}).get("name"))
    agg["figi_type"] = agg["isin"].map(lambda i: (figi.get(i) or {}).get("sectype"))
    m = agg["seibro_name"].str.extract(CHAIN_RE)
    agg["suffix"], agg["suffix_target"] = m[0], m[1]
    text = agg["seibro_name"].fillna("") + " | " + agg["figi_name"].fillna("")
    etfish = agg["figi_type"].isin(["ETP", "Closed-End Fund"]) | (agg["figi_type"].isna() & text.str.contains(LEV_WORDS))
    levish = text.str.contains(LEV_WORDS) & ~agg["figi_type"].isin(["Common Stock", "ADR", "REIT"])
    agg["candidate"] = etfish | levish
    agg = agg.sort_values("amt_weekly", ascending=False)
    agg.to_csv(DER / "isin_master.csv", index=False, encoding="utf-8-sig")
    cand = agg[agg["candidate"]]
    cand.to_csv(DER / "class_candidates.csv", index=False, encoding="utf-8-sig")
    print("전체", len(agg), "후보", len(cand), "티커 없음 후보", int(cand["figi_ticker"].isna().sum()),
          "접미사", agg["suffix"].value_counts().to_dict())


if __name__ == "__main__":
    main()
