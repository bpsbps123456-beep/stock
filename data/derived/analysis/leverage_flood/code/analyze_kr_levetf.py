"""fetch_kr_levetf.py 결과로 급락 기준일 앞뒤의 레버리지 수급 지표를 표로 뽑는다.

- 기준일: 6/23(전날 6/22 고점), 7/28(-10.84%)·7/29, 10/6~13.
- 지표: 단일종목 레버리지 ETF 거래대금(종가×거래량 근사) ÷ 기초자산 거래대금, 개인 순매수 금액, 종가.
- "경고" 1차 판단: 급락 전 5거래일 평균이 상장 후 전체 분포의 상위 20%(80분위) 이상이면 '경고 있었음'으로 표시.
  사후 값(급락 당일·이후)은 설명용.
출력: results/kr_levetf_event_table.csv, results/kr_levetf_signal_test.csv, 화면 표.
"""
import os

import pandas as pd

RESULTS = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis\leverage_flood\results"
EVENTS = {"6/23 급락(6/22 고점)": "20260623", "7/28 급락(-10.84%)": "20260728", "7/29 2차": "20260729",
          "10/6 시작": "20261006", "10/13 -5.5%": "20261013"}
KEYS = ["20260619", "20260622", "20260623", "20260624", "20260727", "20260728", "20260729", "20260730",
        "20261002", "20261006", "20261007", "20261008", "20261012", "20261013", "20261014"]


def main():
    daily = pd.read_csv(os.path.join(RESULTS, "kr_levetf_daily_ratio.csv"), dtype={"date": str}).set_index("date").sort_index()
    daily = daily[daily.index >= "20260527"]
    hk = pd.read_csv(os.path.join(RESULTS, "kr_levetf_hk_us.csv"), dtype={"date": str})
    ks = hk[hk.ticker == "^KS11"].set_index("date")["close"].rename("kospi")
    daily = daily.join(ks, how="left")
    daily["kospi_ret_pct"] = daily["kospi"].pct_change() * 100
    cols = {
        "kospi": "코스피",
        "kospi_ret_pct": "코스피 등락%",
        "single_lev_trading_value_won": "단일레버 거래대금(원)",
        "single_inv_trading_value_won": "단일인버스 거래대금(원)",
        "underlying_trading_value_won": "삼성+하이닉스 거래대금(원)",
        "single_lev_to_underlying_value_ratio": "레버/기초 거래대금 비율",
        "SK하이닉스_lev_etf_to_underlying_value_ratio": "하이닉스 레버/기초 비율",
        "삼성전자_lev_etf_to_underlying_value_ratio": "삼성 레버/기초 비율",
        "single_lev_indiv_net_value_won": "단일레버 개인 순매수(원)",
        "single_inv_indiv_net_value_won": "단일인버스 개인 순매수(원)",
        "122630_indiv_net_value_won": "KODEX레버 개인 순매수(원)",
        "252670_indiv_net_value_won": "인버스2X 개인 순매수(원)",
    }
    have = [c for c in cols if c in daily.columns]
    ev = daily.loc[[k for k in KEYS if k in daily.index], have].rename(columns=cols)
    ev.to_csv(os.path.join(RESULTS, "kr_levetf_event_table.csv"), encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}")
    print(ev.to_string())

    # 사전 신호 검정: 급락 전 5거래일 평균 vs 전체 분포
    tests = []
    sig_cols = [c for c in ["single_lev_to_underlying_value_ratio", "single_lev_trading_value_won", "single_lev_indiv_net_value_won",
                            "single_inv_indiv_net_value_won", "122630_indiv_net_value_won"] if c in daily.columns]
    idx = list(daily.index)
    for label, d in EVENTS.items():
        if d not in idx:
            continue
        i = idx.index(d)
        pre = daily.iloc[max(0, i - 5):i]
        for c in sig_cols:
            series = daily[c].dropna()
            q80 = series.quantile(0.8)
            q20 = series.quantile(0.2)
            pre_mean = pre[c].mean()
            pct_rank = (series < pre_mean).mean()
            tests.append({"event": label, "date": d, "indicator": cols.get(c, c), "pre5_mean": pre_mean, "all_q20": q20, "all_q80": q80,
                          "pre5_percentile": round(pct_rank, 2), "flag_top20pct": bool(pre_mean >= q80), "n_days": len(series)})
    tdf = pd.DataFrame(tests)
    tdf.to_csv(os.path.join(RESULTS, "kr_levetf_signal_test.csv"), index=False, encoding="utf-8-sig")
    print("\n[사전 5거래일 평균의 전체 분포 내 백분위]")
    print(tdf.pivot(index="indicator", columns="event", values="pre5_percentile").to_string())
    # 분포 요약
    print("\n[분포 요약]")
    print(daily[sig_cols].describe().T.to_string())


if __name__ == "__main__":
    main()
