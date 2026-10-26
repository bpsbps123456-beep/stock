"""보관금액의 평가 종가 정하기(자료 정렬 단계, BRIEF v2 §6). 앞날 수익을 쓰지 않는다.

보관 날짜 D의 아는 시각 = D 다음 둘째 KSD 영업일 09:00 KST. 그 전에 끝난 미국 종가(미국 날짜 < 그 KST 날짜) 중
마지막 4개를 늦은 것부터 0, −1, −2, −3으로 부른다. Δlog(보관)을 Δlog(후보 종가)에 회귀해 R²가 가장 큰 후보를 고른다.
적합은 2023-01 ~ 2025-12 자료만 쓰고, 연도마다 같은 후보가 뽑히는지 본다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "derived"
ISINS = {"SOXL": "US25459W4583", "TSLA": "US88160R1014", "NVDA": "US67066G1040", "SPY": "US78462F1030",
         "QQQ": "US46090E1038"}


def ksd_days():
    import exchange_calendars as xc

    s = [d.date() for d in xc.get_calendar("XKRX", start="2022-12-01", end="2027-03-01").sessions]
    return pd.DatetimeIndex([pd.Timestamp(d) for d in s if d.isoformat() != "2026-06-03"])


def known_date(d: pd.Timestamp, ksd: pd.DatetimeIndex, n: int) -> pd.Timestamp:
    i = ksd.searchsorted(d, side="right")
    return ksd[i + n - 1]


def main() -> None:
    import exchange_calendars as xc

    nyse = xc.get_calendar("XNYS", start="2022-12-01", end="2027-03-01").sessions
    ksd = ksd_days()
    cust = pd.read_parquet(DER / "custody.parquet")
    cust["date"] = pd.to_datetime(cust["date"])
    px = pd.read_csv(ROOT / "raw" / "prices.csv", parse_dates=["date"])
    out = {}
    for tk, isin in ISINS.items():
        amt = cust[cust["isin"] == isin].set_index("date")["amt"].sort_index()
        close = px[px["ticker"] == tk].set_index("date")["Close"]
        rec = {}
        for lag in [0, -1, -2, -3]:
            vd = []
            for d in amt.index:
                k = known_date(d, ksd, 2)
                j = nyse.searchsorted(k) - 1 + lag  # k 전 마지막 세션 = 0
                vd.append(nyse[j])
            df = pd.DataFrame({"amt": amt.values, "c": close.reindex(vd).values}, index=amt.index)
            y, x = np.log(df["amt"]).diff(), np.log(df["c"]).diff()
            ok = (x.abs() > 0) & y.notna() & x.notna()
            per = {}
            for yr, lo, hi in [("2023", "2023", "2024"), ("2024", "2024", "2025"), ("2025", "2025", "2026"),
                               ("2023-2025", "2023", "2026")]:
                m = ok & (df.index >= lo) & (df.index < hi)
                r2 = float(np.corrcoef(x[m], y[m])[0, 1] ** 2)
                b = float(np.polyfit(x[m], y[m], 1)[0])
                per[yr] = {"r2": round(r2, 4), "slope": round(b, 3), "n": int(m.sum())}
            rec[lag] = per
        best = {yr: max(rec, key=lambda l: rec[l][yr]["r2"]) for yr in ["2023", "2024", "2025", "2023-2025"]}
        out[tk] = {"by_lag": rec, "best": best}
        print(tk, best, {l: rec[l]["2023-2025"]["r2"] for l in rec})
    votes = [v["best"]["2023-2025"] for v in out.values()]
    chosen = max(set(votes), key=votes.count)
    res = {"tickers": out, "chosen_lag": chosen,
           "rule": "보관 날짜 D의 평가 종가 = 아는 시각(D 다음 둘째 KSD 영업일) 전 마지막 NYSE 세션에서 chosen_lag만큼 앞선 세션의 종가",
           "fit_period": "2023-01-02 ~ 2025-12-31"}
    (DER / "custody_alignment.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("chosen", chosen)


if __name__ == "__main__":
    main()
