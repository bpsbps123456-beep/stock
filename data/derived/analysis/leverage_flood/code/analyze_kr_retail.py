"""kr_retail_daily.csv 로 급락 전 레버리지 상태를 잰다.

입력: ../results/kr_retail_daily.csv (fetch_kr_retail.py 출력, 금액 억원)
출력: ../results/kr_retail_indicators.csv  (일별 파생 지표)
      ../results/kr_retail_event_snapshot.csv (급락 전후 기준일 값)
      ../results/kr_retail_signal_test.csv  (단순 신호등 적중/오경보)

파생 지표(모두 그날 장 마감 후 T+1 저녁에 알 수 있는 값, KOFIA 공표 시차 기준):
  pct250_*   : 직전 250거래일(최소 120일) 안에서의 백분위(0~100, 그날 포함)
  chg20_*    : 20거래일 변화율 %
  *_to_mcap  : (KOSPI+KOSDAQ 시가총액) 대비 비율 %
  credit_to_deposit : 신용융자 / 투자자예탁금
  credit_dd250 : 250일 고점 대비 신용융자 낙폭 %
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

# 급락 기준일(사건일)과 "그 앞에서" 볼 마지막 거래일
EVENTS = [
    ("3/4 미·이란", "2026-03-04", "2026-03-03"),
    ("6/23 삼성·하이닉스 -12% 서킷", "2026-06-23", "2026-06-22"),
    ("7/29 코스피 -10.84%", "2026-07-29", "2026-07-28"),
    ("10/6~13 5일 -13.8%", "2026-10-06", "2026-10-02"),
]

LEVEL_COLS = [
    "credit_loan_total", "credit_loan_kospi", "credit_loan_kosdaq",
    "securities_collateral_loan", "investor_deposit", "misu_balance",
    "forced_sale_amount", "forced_sale_ratio_pct", "cfd_gross_total",
    "credit_loan_shares_total", "kospi_index",
]
RATIO_COLS = ["credit_loan_total", "securities_collateral_loan", "investor_deposit",
              "misu_balance", "cfd_gross_total"]


def rolling_pct(s: pd.Series, window: int = 250, min_periods: int = 120) -> pd.Series:
    def _p(x):
        v = x[-1]
        if np.isnan(v) or len(x) <= 1:
            return np.nan  # 값이 없는 날(아직 공표 전)은 백분위도 없음
        past = x[:-1][~np.isnan(x[:-1])]
        return 100.0 * np.sum(past <= v) / len(past) if len(past) else np.nan
    return s.rolling(window, min_periods=min_periods).apply(_p, raw=True)


def build(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values("date").reset_index(drop=True).copy()
    # KOFIA 는 12/31(휴장) 행에도 잔고를 넣는다. 지수가 없는 날은 지수·시총을 앞값으로 채운다.
    d["mcap_total"] = (d["kospi_mktcap"] + d["kosdaq_mktcap"]).ffill()
    d["kospi_index"] = d["kospi_index"].ffill()
    out = d[["date"]].copy()
    for c in LEVEL_COLS:
        if c not in d:
            continue
        out[c] = d[c]
        out[f"pct250_{c}"] = rolling_pct(d[c])
        out[f"chg20_{c}"] = d[c].pct_change(20) * 100
    for c in RATIO_COLS:
        out[f"{c}_to_mcap"] = d[c] / d["mcap_total"] * 100
        out[f"pct250_{c}_to_mcap"] = rolling_pct(out[f"{c}_to_mcap"])
    out["credit_to_deposit"] = d["credit_loan_total"] / d["investor_deposit"]
    out["pct250_credit_to_deposit"] = rolling_pct(out["credit_to_deposit"])
    out["credit_dd250"] = (d["credit_loan_total"] / d["credit_loan_total"].rolling(250, min_periods=60).max() - 1) * 100
    out["misu_to_credit_pct"] = d["misu_balance"] / d["credit_loan_total"] * 100
    out["mcap_total"] = d["mcap_total"]
    # 앞으로 5·10거래일 KOSPI 수익률(사후값, 신호 평가용)
    out["fwd5_kospi"] = d["kospi_index"].shift(-5) / d["kospi_index"] - 1
    out["fwd10_kospi"] = d["kospi_index"].shift(-10) / d["kospi_index"] - 1
    return out


def snapshot(ind: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = [
        "credit_loan_total", "pct250_credit_loan_total", "chg20_credit_loan_total", "credit_loan_total_to_mcap",
        "pct250_credit_loan_total_to_mcap", "credit_dd250",
        "securities_collateral_loan", "pct250_securities_collateral_loan", "chg20_securities_collateral_loan",
        "securities_collateral_loan_to_mcap",
        "investor_deposit", "pct250_investor_deposit", "chg20_investor_deposit", "investor_deposit_to_mcap",
        "credit_to_deposit", "pct250_credit_to_deposit",
        "misu_balance", "pct250_misu_balance", "chg20_misu_balance", "misu_to_credit_pct",
        "forced_sale_amount", "pct250_forced_sale_amount", "forced_sale_ratio_pct", "pct250_forced_sale_ratio_pct",
        "cfd_gross_total", "pct250_cfd_gross_total", "chg20_cfd_gross_total", "cfd_gross_total_to_mcap",
        "credit_loan_shares_total", "pct250_credit_loan_shares_total",
        "kospi_index", "pct250_kospi_index", "chg20_kospi_index", "mcap_total",
    ]
    ind = ind.set_index("date")
    for label, ev, pre in EVENTS:
        pre_ts = pd.Timestamp(pre)
        i = ind.index.get_loc(pre_ts)
        for off, tag in [(-20, "T-20"), (-5, "T-5"), (0, "T-1(마지막 사전값)"), (1, "T(사건일)"), (2, "T+1"), (5, "T+4")]:
            j = i + off
            if 0 <= j < len(ind):
                r = ind.iloc[j]
                rows.append({"event": label, "tag": tag, "date": ind.index[j].date(), **{k: r.get(k) for k in keys}})
    return pd.DataFrame(rows)


def signal_test(ind: pd.DataFrame) -> pd.DataFrame:
    """단순 신호등: 조건이 켜진 날 수, 그중 5거래일 안에 KOSPI -5% 이상이 온 비율, 각 사건 T-1 에 켜져 있었는지."""
    tests = {
        "신용융자 250일 백분위>=90": ind["pct250_credit_loan_total"] >= 90,
        "신용융자 250일 백분위>=95": ind["pct250_credit_loan_total"] >= 95,
        "신용융자/시총 백분위>=90": ind["pct250_credit_loan_total_to_mcap"] >= 90,
        "신용융자 20일 변화>=+10%": ind["chg20_credit_loan_total"] >= 10,
        "신용융자/예탁금 백분위>=90": ind["pct250_credit_to_deposit"] >= 90,
        "담보융자 250일 백분위>=90": ind["pct250_securities_collateral_loan"] >= 90,
        "미수금 250일 백분위>=90": ind["pct250_misu_balance"] >= 90,
        "반대매매비중>=3%": ind["forced_sale_ratio_pct"] >= 3,
        "CFD 250일 백분위>=90": ind["pct250_cfd_gross_total"] >= 90,
        "KOSPI 250일 백분위>=95": ind["pct250_kospi_index"] >= 95,
    }
    pre_dates = {pd.Timestamp(pre): label for label, _, pre in EVENTS}
    ind = ind.set_index("date")
    rows = []
    valid = ind["pct250_credit_loan_total"].notna()
    for name, cond in tests.items():
        cond = cond.set_axis(ind.index) & valid
        on_days = int(cond.sum())
        hit = ind.loc[cond, "fwd5_kospi"]
        hit5 = float((hit <= -0.05).mean() * 100) if len(hit) else np.nan
        base = float((ind.loc[valid, "fwd5_kospi"] <= -0.05).mean() * 100)
        flags = {label: (bool(cond.get(ts, False)) if ts in cond.index else None) for ts, label in pre_dates.items()}
        rows.append({"signal": name, "on_days": on_days, "total_days": int(valid.sum()),
                     "p_fwd5_drop5pct_when_on": round(hit5, 1), "p_fwd5_drop5pct_base": round(base, 1),
                     **{f"on_at_T-1[{k}]": v for k, v in flags.items()}})
    return pd.DataFrame(rows)


KEYDATES = ["2026-03-03", "2026-06-22", "2026-07-28", "2026-10-02", "2026-10-12", "2026-10-13"]
KEYCOLS = [
    "kospi_index", "chg20_kospi_index", "mcap_total",
    "credit_loan_total", "pct250_credit_loan_total", "chg20_credit_loan_total", "credit_dd250",
    "credit_loan_total_to_mcap", "pct250_credit_loan_total_to_mcap",
    "credit_loan_kospi", "credit_loan_kosdaq",
    "securities_collateral_loan", "pct250_securities_collateral_loan", "chg20_securities_collateral_loan",
    "securities_collateral_loan_to_mcap",
    "investor_deposit", "pct250_investor_deposit", "chg20_investor_deposit", "investor_deposit_to_mcap",
    "credit_to_deposit", "pct250_credit_to_deposit",
    "misu_balance", "pct250_misu_balance", "chg20_misu_balance", "misu_to_credit_pct",
    "forced_sale_amount", "pct250_forced_sale_amount", "forced_sale_ratio_pct", "pct250_forced_sale_ratio_pct",
    "cfd_gross_total", "pct250_cfd_gross_total", "chg20_cfd_gross_total", "cfd_gross_total_to_mcap",
    "credit_loan_shares_total", "pct250_credit_loan_shares_total",
]


def keydates(ind: pd.DataFrame) -> pd.DataFrame:
    """급락 직전 마지막 사전값(T-1)과 최근값을 지표별 행으로 뒤집어 저장(읽기용)."""
    ind = ind.set_index("date")
    cols = [c for c in KEYCOLS if c in ind.columns]
    dates = [pd.Timestamp(x) for x in KEYDATES if pd.Timestamp(x) in ind.index]
    t = ind.loc[dates, cols].T
    t.columns = [c.strftime("%Y-%m-%d") for c in t.columns]
    t.index.name = "indicator"
    return t.round(3)


def main() -> int:
    d = pd.read_csv(RESULTS / "kr_retail_daily.csv", parse_dates=["date"])
    ind = build(d)
    ind.to_csv(RESULTS / "kr_retail_indicators.csv", index=False, encoding="utf-8-sig")
    snap = snapshot(ind)
    snap.to_csv(RESULTS / "kr_retail_event_snapshot.csv", index=False, encoding="utf-8-sig")
    st = signal_test(ind)
    st.to_csv(RESULTS / "kr_retail_signal_test.csv", index=False, encoding="utf-8-sig")
    kd = keydates(ind)
    kd.to_csv(RESULTS / "kr_retail_keydates.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    show = ["event", "tag", "date", "credit_loan_total", "pct250_credit_loan_total", "chg20_credit_loan_total",
            "credit_loan_total_to_mcap", "pct250_credit_loan_total_to_mcap", "credit_dd250",
            "investor_deposit", "pct250_investor_deposit", "credit_to_deposit", "pct250_credit_to_deposit",
            "misu_balance", "pct250_misu_balance", "forced_sale_amount", "forced_sale_ratio_pct",
            "securities_collateral_loan", "pct250_securities_collateral_loan", "cfd_gross_total", "pct250_cfd_gross_total",
            "kospi_index", "pct250_kospi_index", "chg20_kospi_index"]
    print(snap[show].round(2).to_string())
    print()
    print(st.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
