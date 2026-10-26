"""kr_short_*.csv 를 합쳐 급락 기준일 앞 상태와 단순 경고 규칙을 점검한다.

사용: py analyze_kr_short.py
출력(../results):
  kr_short_daily_merged.csv   날짜별 지표 합본 + 코스피 종가(yfinance ^KS11) + 5일 뒤 수익률
  kr_short_event_snapshot.csv 급락 기준일 전후 각 지표의 값·직전 120거래일 백분위(그날 알 수 있던 값만 사용)
  kr_short_signal_test.csv    규칙별: 급락 3일 전 창(T-3~T-1)에 켜졌는지, 전체 발동 횟수, 발동 뒤 5일 평균 수익률
모든 백분위는 그날까지의 값만 쓴다(사후 정보 없음). 대차잔고 금액은 주가에 따라 움직이므로 잔고주수도 같이 본다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
EVENTS = {"20260623": "6/23 서킷브레이커(전날 6/22 고점)", "20260728": "7/28 코스피 -10.84%(금투협·yfinance 날짜 기준; 과제문의 7/29는 -5.98%)", "20260729": "7/29 -5.98%", "20261006": "10/6~13 5일 -13.8% 시작", "20261013": "10/13 -5.46%(10/6~13 끝)"}
SNAP_DATES = ["20260615", "20260618", "20260619", "20260622", "20260623", "20260624", "20260724", "20260727", "20260728", "20260729", "20260730",
              "20261001", "20261002", "20261006", "20261007", "20261008", "20261012", "20261013"]


def rd(name, **kw):
    p = R / name
    if not p.exists():
        print(f"못 받음: {name} 없음")
        return None
    return pd.read_csv(p, dtype={"date": str, "code": str}, **kw)


def main():
    fs = rd("kr_short_foreign_stock.csv")
    fut = rd("kr_short_foreign_futures_k200.csv")
    pg = rd("kr_short_program_kospi_100m_krw.csv")
    lt = rd("kr_short_lending_total.csv")
    ki = rd("kr_short_investor_kospi_100m_krw.csv")
    ls = rd("kr_short_lending_stock.csv")
    adr = rd("kr_short_adr_premium.csv")

    df = pd.DataFrame({"date": sorted(set(lt.date) | set(fut.date))})
    df = df.merge(lt[["date", "lend_shares", "return_shares", "balance_shares", "balance_amount_trillion_krw"]].rename(columns={
        "lend_shares": "lend_fill_shares", "return_shares": "lend_return_shares", "balance_shares": "lend_bal_shares", "balance_amount_trillion_krw": "lend_bal_tril"}), on="date", how="left")
    df = df.merge(fut[["date", "foreign_net", "fin_invest_net", "indiv_net", "total_contracts"]].rename(columns={
        "foreign_net": "fut_foreign_net", "fin_invest_net": "fut_fininv_net", "indiv_net": "fut_indiv_net", "total_contracts": "fut_total"}), on="date", how="left")
    df = df.merge(pg[["date", "arb_net_100m", "nonarb_net_100m", "total_net_100m"]].rename(columns={"arb_net_100m": "prog_arb_net", "nonarb_net_100m": "prog_nonarb_net", "total_net_100m": "prog_total_net"}), on="date", how="left")
    df = df.merge(ki[["date", "foreign_net", "indiv_net", "fin_invest_net", "pension_net"]].rename(columns={"foreign_net": "kospi_foreign_net", "indiv_net": "kospi_indiv_net", "fin_invest_net": "kospi_fininv_net", "pension_net": "kospi_pension_net"}), on="date", how="left")
    for code, tag in [("005930", "sec"), ("000660", "hynix")]:
        s = fs[fs.code == code][["date", "close", "foreign_net_shares", "foreign_hold_ratio_pct"]].rename(columns={"close": f"{tag}_close", "foreign_net_shares": f"{tag}_frgn_net_sh", "foreign_hold_ratio_pct": f"{tag}_frgn_ratio"})
        df = df.merge(s, on="date", how="left")
        if ls is not None:
            l = ls[ls.code == code][["date", "balance_shares", "balance_amount_trillion_krw", "lend_shares"]].rename(columns={"balance_shares": f"{tag}_lend_bal_sh", "balance_amount_trillion_krw": f"{tag}_lend_bal_tril", "lend_shares": f"{tag}_lend_fill_sh"})
            df = df.merge(l, on="date", how="left")
    if adr is not None:
        df = df.merge(adr[["date", "samsung_gdr_premium_pct", "hynix_gdr_premium_pct"]], on="date", how="left")

    # 코스피 종가·앞 5일 수익률 (yfinance)
    try:
        import yfinance as yf
        k = yf.Ticker("^KS11").history(start="2025-05-01", end="2026-10-15", auto_adjust=False)["Close"]
        k.index = pd.to_datetime(k.index.date).strftime("%Y%m%d")
        df["kospi_close"] = df.date.map(k)
    except Exception as ex:  # noqa: BLE001
        print("yfinance ^KS11 못 받음:", ex)
        df["kospi_close"] = np.nan
    df = df.sort_values("date").reset_index(drop=True)
    df["kospi_ret_1d_pct"] = df.kospi_close.pct_change() * 100
    df["kospi_fwd5_pct"] = (df.kospi_close.shift(-5) / df.kospi_close - 1) * 100
    df["kospi_fwd5_min_pct"] = (df.kospi_close.rolling(5).min().shift(-5) / df.kospi_close - 1) * 100
    # 파생 열
    df["fut_foreign_net_5d"] = df.fut_foreign_net.rolling(5).sum()
    df["fut_foreign_net_20d"] = df.fut_foreign_net.rolling(20).sum()
    df["kospi_foreign_net_5d"] = df.kospi_foreign_net.rolling(5).sum()
    df["prog_nonarb_net_5d"] = df.prog_nonarb_net.rolling(5).sum()
    df["lend_bal_sh_chg20_pct"] = df.lend_bal_shares.pct_change(20) * 100
    df["lend_bal_tril_vs_60d_high_pct"] = (df.lend_bal_tril / df.lend_bal_tril.rolling(60).max() - 1) * 100
    df["sec_frgn_ratio_chg20"] = df.sec_frgn_ratio.diff(20)
    df["hynix_frgn_ratio_chg20"] = df.hynix_frgn_ratio.diff(20)
    df.round(3).to_csv(R / "kr_short_daily_merged.csv", index=False, encoding="utf-8-sig")

    # 직전 120거래일 백분위(그날 값 포함, 사후 정보 없음)
    def pct_rank_trailing(s: pd.Series, win: int = 120) -> pd.Series:
        out = pd.Series(np.nan, index=s.index)
        vals = s.values
        for i in range(len(s)):
            if np.isnan(vals[i]):
                continue
            w = vals[max(0, i - win + 1): i + 1]
            w = w[~np.isnan(w)]
            if len(w) < 40:
                continue
            out.iloc[i] = (w <= vals[i]).mean() * 100
        return out

    cols = ["lend_fill_shares", "lend_bal_shares", "lend_bal_tril", "lend_bal_sh_chg20_pct", "fut_foreign_net", "fut_foreign_net_5d", "fut_foreign_net_20d",
            "prog_arb_net", "prog_nonarb_net", "prog_nonarb_net_5d", "kospi_foreign_net", "kospi_foreign_net_5d", "sec_frgn_ratio", "sec_frgn_ratio_chg20", "hynix_frgn_ratio", "hynix_frgn_ratio_chg20"]
    for c in ["sec_lend_bal_sh", "hynix_lend_bal_sh", "sec_lend_fill_sh", "hynix_lend_fill_sh", "samsung_gdr_premium_pct", "hynix_gdr_premium_pct"]:
        if c in df:
            cols.append(c)
    pr = {c: pct_rank_trailing(df[c]) for c in cols}
    snap = []
    for d in SNAP_DATES:
        if d not in set(df.date):
            continue
        i = df.index[df.date == d][0]
        for c in cols:
            snap.append({"date": d, "event": EVENTS.get(d, ""), "indicator": c, "value": df.at[i, c], "pct_rank_120d": round(pr[c].iloc[i], 1) if not np.isnan(pr[c].iloc[i]) else "",
                         "kospi_close": df.at[i, "kospi_close"], "kospi_ret_1d_pct": round(df.at[i, "kospi_ret_1d_pct"], 2) if not np.isnan(df.at[i, "kospi_ret_1d_pct"]) else ""})
    pd.DataFrame(snap).to_csv(R / "kr_short_event_snapshot.csv", index=False, encoding="utf-8-sig")

    # 단순 규칙 검정: 규칙이 켜진 날 = flag. 급락 기준일 T의 T-3~T-1 창에 flag가 있었나, 전체 flag 수, flag 뒤 5일 수익률 평균.
    rules = {
        "대차체결주수 상위10%(120일)": pr["lend_fill_shares"] >= 90,
        "대차잔고주수 20일 증가율 상위10%": pr["lend_bal_sh_chg20_pct"] >= 90,
        "대차잔고금액 60일 고점(=최고)": df["lend_bal_tril_vs_60d_high_pct"] >= -0.5,
        "외국인 K200선물 당일 순매도 하위10%": pr["fut_foreign_net"] <= 10,
        "외국인 K200선물 5일 누적 하위10%": pr["fut_foreign_net_5d"] <= 10,
        "외국인 K200선물 20일 누적 하위10%": pr["fut_foreign_net_20d"] <= 10,
        "비차익 프로그램 당일 순매도 하위10%": pr["prog_nonarb_net"] <= 10,
        "비차익 프로그램 5일 누적 하위10%": pr["prog_nonarb_net_5d"] <= 10,
        "코스피 외국인 당일 순매도 하위10%": pr["kospi_foreign_net"] <= 10,
        "코스피 외국인 5일 누적 하위10%": pr["kospi_foreign_net_5d"] <= 10,
        "삼성전자 외국인 보유율 20일 변화 하위10%": pr["sec_frgn_ratio_chg20"] <= 10,
        "하이닉스 외국인 보유율 20일 변화 하위10%": pr["hynix_frgn_ratio_chg20"] <= 10,
    }
    if "hynix_lend_bal_sh" in pr:
        rules["하이닉스 대차잔고주수 상위10%(120일)"] = pr["hynix_lend_bal_sh"] >= 90
        rules["삼성전자 대차잔고주수 상위10%(120일)"] = pr["sec_lend_bal_sh"] >= 90
    if "samsung_gdr_premium_pct" in pr:
        rules["삼성 GDR 프리미엄 하위10%(120일)"] = pr["samsung_gdr_premium_pct"] <= 10
    ev_idx = {d: df.index[df.date == d][0] for d in ["20260623", "20260728", "20261006"] if d in set(df.date)}
    base_fwd = df.kospi_fwd5_pct.mean()
    rows = []
    for name, flag in rules.items():
        flag = flag.fillna(False).astype(bool)
        n = int(flag.sum())
        hit = {}
        for d, i in ev_idx.items():
            win = flag.iloc[max(0, i - 3): i]  # T-3..T-1
            hit[d] = "켜짐" if win.any() else "없음"
        fwd = df.loc[flag, "kospi_fwd5_pct"]
        fwdmin = df.loc[flag, "kospi_fwd5_min_pct"]
        rows.append({"rule": name, "n_flags": n, "n_days": int(flag.notna().sum()),
                     "before_0623(T-3~T-1)": hit.get("20260623", ""), "before_0728(-10.84%)": hit.get("20260728", ""), "before_1006": hit.get("20261006", ""),
                     "avg_fwd5_pct_when_flag": round(fwd.mean(), 2) if n else "", "avg_fwd5_pct_all_days": round(base_fwd, 2),
                     "share_fwd5_below_-5pct_when_flag": round((fwd < -5).mean() * 100, 1) if n else "", "share_fwd5_below_-5pct_all": round((df.kospi_fwd5_pct < -5).mean() * 100, 1),
                     "avg_fwd5_min_pct_when_flag": round(fwdmin.mean(), 2) if n else ""})
    out = pd.DataFrame(rows)
    out.to_csv(R / "kr_short_signal_test.csv", index=False, encoding="utf-8-sig")

    # GDR 야간 검정: 런던 달력일 D 종가(한국 D+1 01:30) 프리미엄 → 다음 한국 거래일 삼성전자 수익률
    if adr is not None:
        sec = fs[fs.code == "005930"].set_index("date")["close"].astype(float)
        sec_ret = sec.pct_change() * 100
        dates = list(sec.index)
        t = []
        for _, r in adr.iterrows():
            nxt = [x for x in dates if x > r["date"]]
            if nxt and not np.isnan(r["samsung_gdr_premium_pct"]):
                t.append({"london_date": r["date"], "next_kr_date": nxt[0], "gdr_premium_pct": r["samsung_gdr_premium_pct"], "next_day_samsung_ret_pct": sec_ret.get(nxt[0], np.nan)})
        t = pd.DataFrame(t).dropna()
        g = [{"rule": "전체", "n": len(t), "mean_next_ret_pct": round(t.next_day_samsung_ret_pct.mean(), 2), "share_next_ret_below_-3pct": round((t.next_day_samsung_ret_pct < -3).mean() * 100, 1),
              "corr_prem_vs_next_ret": round(t.gdr_premium_pct.corr(t.next_day_samsung_ret_pct), 3)}]
        for th in [-2, -3, -5]:
            sub = t[t.gdr_premium_pct <= th]
            g.append({"rule": f"GDR 프리미엄 <= {th}%", "n": len(sub), "mean_next_ret_pct": round(sub.next_day_samsung_ret_pct.mean(), 2) if len(sub) else "",
                      "share_next_ret_below_-3pct": round((sub.next_day_samsung_ret_pct < -3).mean() * 100, 1) if len(sub) else "", "corr_prem_vs_next_ret": ""})
        for th in [2, 3, 5]:
            sub = t[t.gdr_premium_pct >= th]
            g.append({"rule": f"GDR 프리미엄 >= {th}%", "n": len(sub), "mean_next_ret_pct": round(sub.next_day_samsung_ret_pct.mean(), 2) if len(sub) else "",
                      "share_next_ret_below_-3pct": round((sub.next_day_samsung_ret_pct < -3).mean() * 100, 1) if len(sub) else "", "corr_prem_vs_next_ret": ""})
        pd.DataFrame(g).to_csv(R / "kr_short_gdr_overnight_test.csv", index=False, encoding="utf-8-sig")
        t.round(3).to_csv(R / "kr_short_gdr_overnight_pairs.csv", index=False, encoding="utf-8-sig")
        print(pd.DataFrame(g).to_string(index=False))
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
