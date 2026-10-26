"""c2 audit 3: VIX 종가 시각. 일봉 VIX 종가(16:15 ET)와 16:00 ET 값(15:55 봉 종가)의 차이를 최근 60일 5분봉으로 잰다.
M4a·M4b·M4c가 주식 종가(16:00) 뒤 15분 정보를 얼마나 담는지, P1 3분위 경계 대비 얼마나 칸을 바꾸는지 본다.

5분봉은 2026-10-25에 yfinance(^VIX·^VIX3M, 5분봉 60일 한도, 2026-08-03~10-23)로 받아 cache/c2audit_VIX*_5m.parquet에 두었다.
py code/c2audit_vixclock.py → results/c2audit_vixclock.csv
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent


def at_1600(x: pd.DataFrame) -> pd.DataFrame:
    c = x["Close"].iloc[:, 0] if isinstance(x.columns, pd.MultiIndex) else x["Close"]
    c.index = c.index.tz_convert("America/New_York")
    df = c.to_frame("c")
    df["d"] = df.index.date
    df["hm"] = df.index.strftime("%H:%M")
    out = df[df.hm == "15:55"].set_index("d").c.rename("c1600")
    last = df.groupby("d").agg(last_hm=("hm", "last"), last_c=("c", "last"))
    return last.join(out)


def main(folder: pathlib.Path):
    daily = pd.read_parquet(ROOT / "cache" / "vix.parquet")
    ev = pd.read_parquet(ROOT / "results" / "events_ZZA3.parquet")
    p1 = ev[ev.period == "P1"]
    rows = []
    a = at_1600(pd.read_parquet(folder / "c2audit_VIX_5m.parquet")).add_prefix("vix_")
    b = at_1600(pd.read_parquet(folder / "c2audit_VIX3M_5m.parquet")).add_prefix("v3m_")
    j = a.join(b, how="inner").join(daily, how="inner")
    j["lr_vix"] = np.log(j.VIX / j.vix_c1600)
    j["lr_v3m"] = np.log(j.VIX3M / j.v3m_c1600)
    j["M4c_close"] = j.VIX / j.VIX3M
    j["M4c_1600"] = j.vix_c1600 / j.v3m_c1600
    # P1 3분위 경계로 칸이 바뀌는지(M4a·M4c; M4b는 5일 전 값도 같은 시각 차이를 갖는다)
    for f, now, alt in (("M4a", "VIX", "vix_c1600"), ("M4c", "M4c_close", "M4c_1600")):
        q = p1[f].quantile([1 / 3, 2 / 3]).values
        cell = lambda x: np.searchsorted(q, x)  # noqa: E731
        j[f + "_cell_change"] = cell(j[now].values) != cell(j[alt].values)
    j.to_csv(ROOT / "results" / "c2audit_vixclock.csv")
    print(j[["vix_last_hm", "v3m_last_hm"]].value_counts().head())
    print("일 수", len(j))
    for c in ("lr_vix", "lr_v3m"):
        x = j[c].dropna()
        print(c, "중앙 |차이| %.3f%%, 90분위 %.3f%%, 최대 %.3f%%" % (100 * x.abs().median(), 100 * x.abs().quantile(0.9), 100 * x.abs().max()))
    d4c = (j.M4c_close / j.M4c_1600 - 1).abs()
    print("M4c 중앙 |차이| %.3f%%, 최대 %.3f%%" % (100 * d4c.median(), 100 * d4c.max()))
    print("P1 3분위 칸이 바뀐 날: M4a", int(j.M4a_cell_change.sum()), "M4c", int(j.M4c_cell_change.sum()), "/", len(j))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cache")
