"""Download CFTC legacy COT (futures only) 2024 and extract JPY non-commercial net position by week.
Usage: py cftc_jpy_2024.py
"""
import io
import os
import zipfile

import pandas as pd
import requests

RESULTS = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis\leverage_flood\results"
URL = "https://www.cftc.gov/files/dea/history/deacot2024.zip"


def main():
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    print("status", r.status_code, len(r.content))
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    name = zf.namelist()[0]
    df = pd.read_csv(zf.open(name), low_memory=False)
    cols = {c.strip(): c for c in df.columns}
    mcol = cols.get("Market and Exchange Names")
    jpy = df[df[mcol].str.contains("JAPANESE YEN", na=False)].copy()
    dcol = cols.get("As of Date in Form YYYY-MM-DD")
    jpy["date"] = pd.to_datetime(jpy[dcol])
    lc = cols.get("Noncommercial Positions-Long (All)")
    sc = cols.get("Noncommercial Positions-Short (All)")
    oi = cols.get("Open Interest (All)")
    jpy["nc_long"] = jpy[lc]
    jpy["nc_short"] = jpy[sc]
    jpy["nc_net"] = jpy[lc] - jpy[sc]
    jpy["open_interest"] = jpy[oi]
    out = jpy[["date", mcol, "nc_long", "nc_short", "nc_net", "open_interest"]].sort_values("date")
    path = os.path.join(RESULTS, "cftc_jpy_noncommercial_2024.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print("saved", path)
    print(out[(out["date"] >= "2024-05-01") & (out["date"] <= "2024-09-01")].to_string(index=False))
    mn = out.loc[out["nc_net"].idxmin()]
    print("MOST NET SHORT:", mn["date"].date(), int(mn["nc_net"]))


if __name__ == "__main__":
    main()
