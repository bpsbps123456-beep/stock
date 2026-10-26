"""N2 자료: n1이 받은 yfinance 일봉(^SOX·SOXX·QQQ·SPY·^TNX, 1994~2026-10-23)을 n2 전용으로 고정하고,
^TNX 날짜 정렬을 IEF(7~10년 국채 ETF, 16:00 ET 종가)로 점검한다.

py code/n2_data.py → cache/n2_prices_raw.parquet · cache/n2_prices_adj.parquet · cache/n2_ief.parquet
                    → results/n2_data_meta.json · results/n2_tnx_align.csv

- 원본: cache/n1_raw.parquet(auto_adjust=False), cache/n1_adj.parquet(auto_adjust=True). 다른 갈래가 다시 받아도 n2 결과가 흔들리지 않게 사본을 둔다.
- 정렬 점검: corr(ΔTNX[d], IEF 수익[d+L]), L = −2..2. 날짜가 맞으면 L = 0이 강한 음의 상관이고 나머지는 0 근처다.
- ^TNX 종가 시각: Cboe 지수 종가는 미국 주식 종가(16:00 ET) 이전·같은 시각이라는 가정은 확인하지 못했다. 그래서 n2_test에서 하루 늦춘 금리(d−1)로 민감도를 함께 낸다.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil

import numpy as np
import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).resolve().parent.parent
CACHE, RES = HERE / "cache", HERE / "results"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    meta = {}
    for src, dst in (("n1_raw", "n2_prices_raw"), ("n1_adj", "n2_prices_adj")):
        s, d = CACHE / f"{src}.parquet", CACHE / f"{dst}.parquet"
        if not d.exists():
            shutil.copyfile(s, d)
        meta[dst] = dict(source=f"cache/{src}.parquet (n1_fetch.py, yfinance)", sha256=sha(d))
    ief_p = CACHE / "n2_ief.parquet"
    if not ief_p.exists():
        x = yf.download("IEF", start="2002-07-01", end="2026-10-24", interval="1d", auto_adjust=True, progress=False, actions=False)
        x.columns = [c[0] if isinstance(c, tuple) else c for c in x.columns]
        x = x[["Open", "High", "Low", "Close"]].rename(columns=dict(Open="O", High="H", Low="L", Close="C"))
        x.index = pd.DatetimeIndex(x.index).tz_localize(None).normalize()
        x.index.name = "date"
        x.reset_index().to_parquet(ief_p)
    meta["n2_ief"] = dict(source="yfinance IEF auto_adjust=True", sha256=sha(ief_p))

    raw = pd.read_parquet(CACHE / "n2_prices_raw.parquet")
    tnx = raw[raw.sym == "^TNX"].set_index("date").C
    spx = raw[raw.sym == "SPY"].set_index("date").C
    ief = pd.read_parquet(ief_p).set_index("date").C
    # 달력 차이: 주식 거래일인데 ^TNX가 없는 날(채권 휴장 등), ^TNX만 있는 날
    eq_days, t_days = set(spx.index), set(tnx.index)
    miss = sorted(eq_days - t_days)
    extra = sorted(t_days - eq_days)
    meta["tnx_calendar"] = dict(equity_days=len(eq_days), tnx_days=len(t_days),
                                equity_days_without_tnx=len(miss), tnx_days_without_equity=len(extra),
                                examples_missing=[str(d.date()) for d in miss[:8]], examples_extra=[str(d.date()) for d in extra[:8]])
    # 정렬 점검(IEF 거래일 기준)
    rows = []
    cal = ief.index
    t = tnx.reindex(cal).ffill()
    dt = t.diff()
    ri = np.log(ief).diff()
    rs = np.log(spx.reindex(cal)).diff()
    for era, (a, b) in {"2002-2013": ("2002-08-01", "2013-12-31"), "2014-2023": ("2014-01-01", "2023-12-31"),
                        "2024-2026": ("2024-01-01", "2026-10-23")}.items():
        m = (cal >= a) & (cal <= b)
        for L in range(-2, 3):
            rows.append(dict(era=era, lag=L, corr_dTNX_IEF=float(dt[m].corr(ri.shift(-L)[m])),
                             corr_dTNX_SPY=float(dt[m].corr(rs.shift(-L)[m])), n=int(m.sum())))
    al = pd.DataFrame(rows)
    al.to_csv(RES / "n2_tnx_align.csv", index=False, encoding="utf-8-sig")
    (RES / "n2_data_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(al.round(3).to_string())


if __name__ == "__main__":
    main()
