"""Build a key-date table from the saved KOFIA freesis series (2025-01-01 .. 2026-10-13).

0070BO columns (unit 1천만원, verified against 6/19 news: 38조4786억 / 29조3977억 / 9조809억):
  TMPV2 신용융자 전체, TMPV3 유가증권, TMPV4 코스닥, TMPV9 예탁증권담보융자
0060BO columns (unit 1천만원, verified against 6/19 news: 예탁금 129조3534억 / 미수금 1조2057억 / 반대매매 234억 / 1.9%):
  TMPV2 투자자예탁금, TMPV5 위탁매매 미수금, TMPV6 반대매매 금액, TMPV7 반대매매 비중(%)
Usage: py kofia_keydates.py
"""
import os

import pandas as pd

RESULTS = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis\leverage_flood\results"
KEY = ["2026-03-03", "2026-03-04", "2026-06-05", "2026-06-08", "2026-06-19", "2026-06-22", "2026-06-23", "2026-06-24",
       "2026-07-07", "2026-07-13", "2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31", "2026-08-03",
       "2026-09-23", "2026-10-02", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-12", "2026-10-13"]


def load(obj):
    df = pd.read_csv(os.path.join(RESULTS, f"kofia_{obj}_20250101_20261013.csv"))
    df["date"] = pd.to_datetime(df["TMPV1"].astype(str))
    return df.set_index("date").sort_index()


def main():
    c = load("STATSCU0100000070BO")
    f = load("STATSCU0100000060BO")
    out = pd.DataFrame(index=c.index)
    out["credit_total_tril"] = c["TMPV2"] / 1e5
    out["credit_kospi_tril"] = c["TMPV3"] / 1e5
    out["credit_kosdaq_tril"] = c["TMPV4"] / 1e5
    out["collateral_loan_tril"] = c["TMPV9"] / 1e5
    out["credit_chg_20d_pct"] = out["credit_total_tril"].pct_change(20) * 100
    out["credit_chg_60d_pct"] = out["credit_total_tril"].pct_change(60) * 100
    out["credit_yoy_pct"] = out["credit_total_tril"].pct_change(245) * 100
    out["credit_vs_record_pct"] = (out["credit_total_tril"] / out["credit_total_tril"].cummax() - 1) * 100
    out["is_record"] = out["credit_total_tril"] >= out["credit_total_tril"].cummax()
    ff = f.reindex(out.index)
    out["deposit_tril"] = ff["TMPV2"] / 1e5
    out["misu_tril"] = ff["TMPV5"] / 1e5
    out["forced_sale_bil"] = ff["TMPV6"] / 1e2
    out["forced_sale_ratio_pct"] = ff["TMPV7"]
    out["credit_to_deposit"] = out["credit_total_tril"] / out["deposit_tril"]
    path = os.path.join(RESULTS, "kofia_daily_derived_2025_2026.csv")
    out.round(3).to_csv(path, encoding="utf-8-sig")
    print("saved", path)
    key = out.loc[[d for d in KEY if pd.Timestamp(d) in out.index]]
    pd.set_option("display.width", 250)
    print(key.round(2).to_string())
    rec = out[out["is_record"]].tail(15)
    print("\nlast record-high dates of credit_total:")
    print(rec[["credit_total_tril"]].round(3).to_string())
    print("\nmax forced_sale days:")
    print(out.sort_values("forced_sale_bil", ascending=False).head(12)[["forced_sale_bil", "forced_sale_ratio_pct", "misu_tril"]].round(1).to_string())
    print("\n2025 reference: credit_total on 2025-06-23, 2025-07-29, 2025-10-13 (nearest):")
    for d in ["2025-06-23", "2025-07-29", "2025-10-13"]:
        idx = out.index.get_indexer([pd.Timestamp(d)], method="nearest")[0]
        print(out.index[idx].date(), round(out.iloc[idx]["credit_total_tril"], 3))


if __name__ == "__main__":
    main()
