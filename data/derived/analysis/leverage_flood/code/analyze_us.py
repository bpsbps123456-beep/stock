"""미국 레버리지·심리 지표의 급락 직전 상태(백분위·z값)와 단순 신호 검정.

실행: py analyze_us.py   (fetch_us.py 결과 us_*.csv 를 읽는다)
출력:
  us_daily_panel.csv     일별 지표 패널(공개 시점 기준으로 정렬된 값)
  us_event_snapshot.csv  3/3·6/22·7/28·10/5·10/13 시점 값·250일 백분위·z값 (주간·월간은 그 날 알 수 있던 최신값)
  us_signal_test.csv     지표 극단(백분위) 신호 ON일 때 ^SOX 5일 뒤 -5% 확률 vs 기본율, 세 급락 T-1 ON 여부
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

RESULTS = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis\leverage_flood\results"
EVENTS = {  # 라벨: (급락일, T-1 = 직전 거래일)
    "3/4 미·이란": ("2026-03-04", "2026-03-03"),
    "6/23 삼성·하이닉스 -12% 서킷": ("2026-06-23", "2026-06-22"),
    "7/29 코스피 -10.84%": ("2026-07-29", "2026-07-28"),
    "10/6~13 5일 -13.8%": ("2026-10-06", "2026-10-05"),
    "현재": (None, "2026-10-13"),
}
WIN = 250


def rd(name, **kw):
    try:
        return pd.read_csv(f"{RESULTS}\\{name}", **kw)
    except FileNotFoundError:
        print("  missing", name)
        return None


def pct_rank(s: pd.Series, win=WIN) -> pd.Series:
    return s.rolling(win, min_periods=int(win * 0.6)).apply(lambda x: (x[:-1] <= x[-1]).mean() * 100 if len(x) > 1 else np.nan, raw=True)


def zscore(s: pd.Series, win=WIN) -> pd.Series:
    m = s.rolling(win, min_periods=int(win * 0.6)).mean()
    sd = s.rolling(win, min_periods=int(win * 0.6)).std()
    return (s - m) / sd


def build_daily_panel() -> pd.DataFrame:
    close = rd("us_prices_close.csv", parse_dates=["Date"]).set_index("Date")
    vol = rd("us_prices_volume.csv", parse_dates=["Date"]).set_index("Date")
    P = pd.DataFrame(index=close.index)
    P["SOX"] = close["^SOX"]
    P["NDX"] = close["^NDX"]
    P["SPY"] = close["SPY"]
    P["GVIP_SPY"] = close["GVIP"] / close["SPY"]
    P["GVIP_SPY_chg20"] = P["GVIP_SPY"].pct_change(20) * 100
    for t in ["SOXL", "SOXS", "NVDL", "MUU", "TQQQ", "SQQQ", "SMH", "SOXX"]:
        P[f"{t}_dollar_vol"] = close[t] * vol[t]
        P[f"{t}_dollar_vol_20d"] = P[f"{t}_dollar_vol"].rolling(20).mean()
    P["SOXL_SOXS_dollar_vol_ratio_5d"] = P["SOXL_dollar_vol"].rolling(5).sum() / P["SOXS_dollar_vol"].rolling(5).sum()
    P["TQQQ_SQQQ_dollar_vol_ratio_5d"] = P["TQQQ_dollar_vol"].rolling(5).sum() / P["SQQQ_dollar_vol"].rolling(5).sum()
    P["lev_bull_dollar_vol_share"] = (P["SOXL_dollar_vol"] + P["NVDL_dollar_vol"] + P["TQQQ_dollar_vol"] + P["MUU_dollar_vol"].fillna(0)) / (
        P["SMH_dollar_vol"] + P["SOXX_dollar_vol"] + close["SPY"] * vol["SPY"] + close["QQQ"] * vol["QQQ"])

    cv = rd("us_cboe_vol.csv", parse_dates=["date"]).set_index("date")
    for c in ["VIX", "VXN", "SKEW", "VVIX", "VIX_VIX3M"]:
        P[c] = cv[c]
    P["VXN_VIX"] = cv["VXN"] / cv["VIX"]

    pc = rd("us_cboe_putcall.csv", parse_dates=["date"]).set_index("date")
    for c in ["pc_total", "pc_equity", "pc_index", "pc_etp"]:
        P[c] = pc[c]
        P[f"{c}_5d"] = pc[c].rolling(5).mean()
    P["opt_vol_total"] = pc["all_vol_call"] + pc["all_vol_put"]
    P["equity_call_vol_20d"] = pc["equity_vol_call"].rolling(20).mean()

    ps = rd("us_proshares_nav.csv", parse_dates=["date"])
    for t in ["TQQQ", "SQQQ", "QLD", "UPRO", "SPXU"]:
        s = ps[ps["ticker"] == t].set_index("date")
        P[f"{t}_shares"] = s["shares_out"]
        P[f"{t}_aum"] = s["aum"]
        P[f"{t}_flow_5d"] = s["flow_usd_1d"].rolling(5).sum()
        P[f"{t}_flow_20d"] = s["flow_usd_1d"].rolling(20).sum()
    P["TQQQ_SQQQ_aum_ratio"] = P["TQQQ_aum"] / P["SQQQ_aum"]
    P["UPRO_SPXU_aum_ratio"] = P["UPRO_aum"] / P["SPXU_aum"]

    nv = rd("us_nvdl_nav.csv", parse_dates=["date"]).drop_duplicates("date", keep="last").set_index("date")
    P["NVDL_shares"] = nv["shares_implied"]
    P["NVDL_aum"] = nv["aum"]
    P["NVDL_flow_5d"] = nv["flow_usd_1d"].rolling(5).sum()
    P["NVDL_flow_20d"] = nv["flow_usd_1d"].rolling(20).sum()

    sv = rd("us_finra_short_volume.csv", parse_dates=["date"])
    if sv is not None and len(sv):
        for t in ["NVDA", "MU", "SOXL", "SMH", "SOXX", "TQQQ", "SPY"]:
            s = sv[sv["symbol"] == t].set_index("date")["short_ratio"]
            P[f"{t}_short_ratio_5d"] = s.rolling(5).mean()

    P = P.sort_index()
    P = P[P.index >= "2023-06-01"]
    return P


# 주간·월간: 공개일(release_date) 기준으로 "그 날 알 수 있던 최신값"
def weekly_monthly_series() -> dict[str, pd.Series]:
    out = {}
    cf = rd("us_cftc_tff.csv", parse_dates=["release_date"])
    fo = cf[cf["report_type"] == "futures_only"]
    for mk, lab in [("NASDAQ-100 Consolidated", "NQ"), ("S&P 500 Consolidated", "ES"), ("VIX FUTURES", "VX")]:
        s = fo[fo["market"] == mk].set_index("release_date").sort_index()
        out[f"CFTC_{lab}_lev_net"] = s["lev_net"]
        out[f"CFTC_{lab}_lev_net_pct_oi"] = s["lev_net_pct_oi"]
        out[f"CFTC_{lab}_assetmgr_net"] = s["asset_mgr_net"]
    aa = rd("us_aaii.csv", parse_dates=["release_date"]).set_index("release_date").sort_index()
    out["AAII_bull_bear_spread"] = aa["bull_bear_spread"] * 100
    out["AAII_bullish"] = aa["bullish"] * 100
    out["AAII_bull_8wk_ma"] = aa["bull_8wk_ma"] * 100
    na = rd("us_naaim.csv", parse_dates=["release_date"]).set_index("release_date").sort_index()
    out["NAAIM_number"] = na["naaim_number"]
    out["NAAIM_q3"] = na["q3"]
    mg = rd("us_finra_margin.csv", parse_dates=["release_date_est"]).set_index("release_date_est").sort_index()
    out["MARGIN_debit_musd"] = mg["debit_balances_musd"]
    out["MARGIN_yoy_pct"] = mg["yoy_pct"]
    out["MARGIN_net_free_credit"] = mg["net_free_credit"]
    si = rd("us_finra_short_interest.csv", parse_dates=["release_date_est"])
    for t in ["NVDA", "MU", "SOXL", "SOXS", "SMH", "SOXX", "TQQQ", "SQQQ", "SPY"]:
        s = si[si["symbol"] == t].set_index("release_date_est").sort_index()
        out[f"SI_{t}_short_interest"] = s["short_interest"]
        out[f"SI_{t}_days_to_cover"] = s["days_to_cover"]
    return out


def snapshot(P: pd.DataFrame, WM: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for col in P.columns:
        s = P[col].dropna()
        if s.empty:
            continue
        pr = pct_rank(s)
        z = zscore(s)
        for ev, (_, t1) in EVENTS.items():
            t1 = pd.Timestamp(t1)
            sub = s[s.index <= t1]
            if sub.empty:
                rows.append({"indicator": col, "freq": "daily", "event": ev, "as_of": None, "value": None, "pct250": None, "z250": None})
                continue
            d = sub.index[-1]
            rows.append({"indicator": col, "freq": "daily", "event": ev, "as_of": d.date(), "value": s.loc[d],
                         "pct250": pr.loc[d], "z250": z.loc[d]})
    for name, s in WM.items():
        s = s.dropna()
        if s.empty:
            continue
        n = 52 if not name.startswith("MARGIN") else 36
        pr = s.rolling(n, min_periods=int(n * 0.6)).apply(lambda x: (x[:-1] <= x[-1]).mean() * 100, raw=True)
        m = s.rolling(n, min_periods=int(n * 0.6)).mean()
        sd = s.rolling(n, min_periods=int(n * 0.6)).std()
        z = (s - m) / sd
        for ev, (_, t1) in EVENTS.items():
            t1 = pd.Timestamp(t1)
            sub = s[s.index <= t1]
            if sub.empty:
                rows.append({"indicator": name, "freq": "weekly/monthly", "event": ev, "as_of": None, "value": None, "pct250": None, "z250": None})
                continue
            d = sub.index[-1]
            rows.append({"indicator": name, "freq": f"weekly/monthly(win={n})", "event": ev, "as_of": d.date(),
                         "value": s.loc[d], "pct250": pr.loc[d], "z250": z.loc[d]})
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["indicator", "freq"], columns="event", values=["value", "pct250", "z250"], aggfunc="first")
    wide.columns = [f"{b}|{a}" for a, b in wide.columns]
    order = [f"{ev}|{m}" for ev in EVENTS for m in ["value", "pct250", "z250"]]
    wide = wide[[c for c in order if c in wide.columns]].reset_index()
    return wide


def signal_test(P: pd.DataFrame) -> pd.DataFrame:
    sox = P["SOX"]
    fwd5 = sox.shift(-5) / sox - 1
    target = (fwd5 <= -0.05)
    tests = []
    hi = ["TQQQ_aum", "TQQQ_shares", "TQQQ_SQQQ_aum_ratio", "UPRO_SPXU_aum_ratio", "NVDL_shares", "NVDL_aum", "TQQQ_flow_20d", "NVDL_flow_20d",
          "SOXL_dollar_vol_20d", "NVDL_dollar_vol_20d", "MUU_dollar_vol_20d", "SOXL_SOXS_dollar_vol_ratio_5d", "lev_bull_dollar_vol_share",
          "GVIP_SPY", "GVIP_SPY_chg20", "SKEW", "equity_call_vol_20d", "opt_vol_total",
          "NVDA_short_ratio_5d", "SOXL_short_ratio_5d", "SMH_short_ratio_5d"]
    lo = ["VIX", "VXN", "VIX_VIX3M", "VVIX", "pc_total_5d", "pc_equity_5d", "pc_index_5d", "pc_etp_5d"]
    for col, side in [(c, "hi") for c in hi] + [(c, "lo") for c in lo]:
        if col not in P.columns:
            continue
        s = P[col].dropna()
        pr = pct_rank(s).reindex(P.index)
        for th in ([90, 95] if side == "hi" else [10, 5]):
            on = (pr >= th) if side == "hi" else (pr <= th)
            valid = target.notna() & pr.notna()
            n_on = int((on & valid).sum())
            p_on = target[on & valid].mean() * 100 if n_on else np.nan
            p_base = target[valid].mean() * 100
            row = {"signal": f"{col} 250일 백분위{'>=' if side == 'hi' else '<='}{th}", "on_days": n_on, "total_days": int(valid.sum()),
                   "p_fwd5_sox_drop5pct_when_on": round(p_on, 1) if n_on else None, "p_fwd5_sox_drop5pct_base": round(p_base, 1)}
            for ev, (_, t1) in EVENTS.items():
                if ev == "현재":
                    continue
                t1 = pd.Timestamp(t1)
                sub = pr[pr.index <= t1].dropna()
                row[f"on_at_T-1[{ev}]"] = (bool(sub.iloc[-1] >= th) if side == "hi" else bool(sub.iloc[-1] <= th)) if len(sub) else None
            tests.append(row)
    return pd.DataFrame(tests)


def main():
    P = build_daily_panel()
    P.reset_index().rename(columns={"index": "date", "Date": "date"}).to_csv(f"{RESULTS}\\us_daily_panel.csv", index=False, encoding="utf-8-sig")
    print("panel", P.shape, P.index.min().date(), P.index.max().date())
    WM = weekly_monthly_series()
    snap = snapshot(P, WM)
    snap.to_csv(f"{RESULTS}\\us_event_snapshot.csv", index=False, encoding="utf-8-sig")
    print("snapshot rows", len(snap))
    st = signal_test(P)
    st.to_csv(f"{RESULTS}\\us_signal_test.csv", index=False, encoding="utf-8-sig")
    print("signal tests", len(st))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 300)
    print(snap.to_string())
    print(st.to_string())


if __name__ == "__main__":
    main()
