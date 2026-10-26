"""단일종목 레버리지 ETF 거래대금 ÷ 기초자산(삼성+하이닉스) 거래대금 비율 — 사전 고정 경고 규칙 전방검증.

입력: results/kr_levetf_daily_ratio.csv (fetch_kr_levetf.py, 네이버 모바일 price API 종가×거래량 근사, 2026-05-27 상장일부터)
      results/kr_retail_kospi.csv (네이버 코스피 지수), results/us_prices_close.csv (yfinance ^SOX)

발표 시차: 네이버 price API는 당일 장중·종가를 당일 포함해 돌려준다 → 15:30 종가 확정 후 당일 저녁에 값 확정 = 영업일 시차 0.
          D일 값은 D일 저녁에 알고, 행동은 D+1 시가부터 가능.

경고 규칙(결과를 보기 전에 고정):
  R_D = 단일레버 거래대금 ÷ (삼성+하이닉스 거래대금), 시차 0.
  빨강 = R_D 가 상장 이후 D 이전 모든 거래일 값 중 95% 이상보다 높다(순위 백분위 ≥ 0.95), 단 이전 관측 ≥ 10일.
         ("지난 250일 백분위 ≥ 95"의 상장 94일짜리 판: 250일이 안 되므로 상장 이후 전체 = 확장창.)
  에피소드 = 빨강 날들을 5거래일 이내 간격으로 묶은 것.
  급락일 = 코스피 또는 SOX 종가가 직전 5거래일 내 어느 종가 대비 −8% 이하. SOX 급락일(미국 날짜)은 한국 다음 거래일에 알게 되므로 그 날로 옮긴다.
  적중 = 에피소드 [첫 빨강, 마지막 빨강 + 10거래일] 안에 급락일이 있음. 아니면 오경보.
  선행일수 = 사건일 E(6/23·7/28·10/6) 직전 15거래일 안에서 가장 이른 빨강 날과 E 사이의 거래일 수(E−1 빨강이면 1). 없으면 null.

출력(results/): kr_levetf_warning_daily.csv, kr_levetf_warning_episodes.csv, kr_levetf_warning_summary.json
실행: py analyze_kr_levetf_warning.py
"""
import json
import os

import pandas as pd

RESULTS = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis\leverage_flood\results"
COL = "single_lev_to_underlying_value_ratio"
MIN_PRIOR = 10
PCT_THRESH = 0.95
EP_GAP = 5
HIT_HORIZON = 10
LEAD_LOOKBACK = 15
CRASH_PCT = -8.0
PUBLISH_LAG_BDAYS = 0
EVENTS = {"6/23": "20260623", "7/28(7/29 2차)": "20260728", "10/6": "20261006"}
KEY_DATES = ["20260622", "20260728", "20261002", "20261013"]
LAST_COMPLETE = "20261013"  # 10/14 행은 수집 시각(13:58) 장중 값이라 제외


def load():
    d = pd.read_csv(os.path.join(RESULTS, "kr_levetf_daily_ratio.csv"), dtype={"date": str}).set_index("date").sort_index()
    d = d[(d.index >= "20260527") & (d.index <= LAST_COMPLETE)]
    ks = pd.read_csv(os.path.join(RESULTS, "kr_retail_kospi.csv"))
    ks["date"] = ks["date"].str.replace("-", "")
    ks = ks.set_index("date")["kospi_index"].sort_index()
    us = pd.read_csv(os.path.join(RESULTS, "us_prices_close.csv"))
    us["date"] = us["Date"].str.replace("-", "")
    sox = us.set_index("date")["^SOX"].dropna().sort_index()
    return d, ks, sox


def crash_days(close, pct=CRASH_PCT, lookback=5):
    """종가가 직전 lookback 거래일 내 어느 종가 대비 pct% 이하로 떨어진 날."""
    out = {}
    vals = close.values
    idx = list(close.index)
    for i in range(1, len(vals)):
        lo = max(0, i - lookback)
        worst = min(vals[i] / vals[j] - 1 for j in range(lo, i)) * 100
        out[idx[i]] = worst
    return pd.Series(out)


def main():
    d, ks, sox = load()
    kr_days = list(d.index)
    r = d[COL]

    # 1. 빨강: 확장창 순위 백분위
    rows = []
    for i, dt in enumerate(kr_days):
        prior = r.iloc[:i].dropna()
        v = r.iloc[i]
        if len(prior) >= MIN_PRIOR and pd.notna(v):
            pct = float((prior < v).mean())
            red = pct >= PCT_THRESH
        else:
            pct, red = None, False
        rows.append({"date": dt, "ratio": v, "n_prior": len(prior), "pct_rank_prior": pct, "red": red})
    w = pd.DataFrame(rows).set_index("date")

    # 2. 급락일 (코스피: 한국 날짜 그대로, SOX: 다음 한국 거래일로 이동)
    ks_in = ks[(ks.index >= "20260501") & (ks.index <= LAST_COMPLETE)]
    sox_in = sox[(sox.index >= "20260501") & (sox.index <= LAST_COMPLETE)]
    ks_worst = crash_days(ks_in)
    sox_worst = crash_days(sox_in)
    w["kospi"] = ks_in.reindex(w.index)
    w["kospi_ret1_pct"] = ks_in.pct_change().reindex(w.index) * 100
    w["kospi_worst5_pct"] = ks_worst.reindex(w.index)
    w["kospi_crash"] = w["kospi_worst5_pct"] <= CRASH_PCT
    # SOX 날짜 -> 그 다음 한국 거래일
    sox_known = {}
    for us_dt, val in sox_worst.items():
        nxt = next((k for k in kr_days if k > us_dt), None)
        if nxt:
            sox_known[nxt] = min(val, sox_known.get(nxt, 0))
    w["sox_worst5_pct_known"] = pd.Series(sox_known).reindex(w.index)
    w["sox_crash_known"] = w["sox_worst5_pct_known"] <= CRASH_PCT
    w["crash"] = w["kospi_crash"] | w["sox_crash_known"]
    w["sox_close_prev_us_day"] = pd.Series({next((k for k in kr_days if k > us_dt), None): v for us_dt, v in sox_in.items() if next((k for k in kr_days if k > us_dt), None)}).reindex(w.index)

    # 3. 에피소드
    red_days = [dt for dt in kr_days if w.loc[dt, "red"]]
    episodes = []
    for dt in red_days:
        if episodes and kr_days.index(dt) - kr_days.index(episodes[-1]["last"]) <= EP_GAP:
            episodes[-1]["last"] = dt
            episodes[-1]["n_red"] += 1
        else:
            episodes.append({"first": dt, "last": dt, "n_red": 1})
    crash_list = [dt for dt in kr_days if w.loc[dt, "crash"]]
    for ep in episodes:
        i0 = kr_days.index(ep["first"])
        i1 = min(len(kr_days) - 1, kr_days.index(ep["last"]) + HIT_HORIZON)
        window = kr_days[i0:i1 + 1]
        hits = [c for c in crash_list if c in window]
        ep["window_end"] = kr_days[i1]
        ep["crash_in_window"] = hits[0] if hits else ""
        ep["hit"] = bool(hits)
        ep["max_ratio"] = float(w.loc[ep["first"]:ep["last"], "ratio"].max())
    epdf = pd.DataFrame(episodes)

    # 4. 사건별 선행일수
    leads = {}
    for label, e in EVENTS.items():
        ie = kr_days.index(e)
        look = kr_days[max(0, ie - LEAD_LOOKBACK):ie]
        reds = [dt for dt in look if w.loc[dt, "red"]]
        leads[label] = {
            "event": e,
            "lead_days_first_red": (ie - kr_days.index(reds[0])) if reds else None,
            "lead_days_last_red": (ie - kr_days.index(reds[-1])) if reds else None,
            "red_dates": reds,
            "ratio_on_event_minus1": float(w.loc[kr_days[ie - 1], "ratio"]),
            "pct_rank_on_event_minus1": w.loc[kr_days[ie - 1], "pct_rank_prior"],
        }

    # 5. 핵심 날짜 값·백분위 (확장창 사전 백분위 + 상장 이후 전체 표본 백분위)
    full = r.dropna()
    keys = {}
    for k in KEY_DATES:
        if k in w.index:
            keys[k] = {"ratio": float(w.loc[k, "ratio"]), "pct_rank_prior": w.loc[k, "pct_rank_prior"],
                       "pct_rank_full_sample": float((full < w.loc[k, "ratio"]).mean()), "red": bool(w.loc[k, "red"])}

    false_alarms = int((~epdf["hit"]).sum()) if len(epdf) else 0

    # 6. 기저율: 평가 가능한 날 아무 날이나 '빨강'이라 치고 [D, D+10] 창에 급락일이 있을 확률 (적중이 얼마나 쉬운 장세였나)
    evaluable = [dt for dt in kr_days if w.loc[dt, "n_prior"] >= MIN_PRIOR]
    base = []
    for dt in evaluable:
        i0 = kr_days.index(dt)
        window = kr_days[i0:min(len(kr_days) - 1, i0 + HIT_HORIZON) + 1]
        base.append(any(c in window for c in crash_list))
    base_rate_all = float(sum(base) / len(base))
    base_jun_jul = [b for dt, b in zip(evaluable, base) if dt <= "20260731"]
    base_aug_oct = [b for dt, b in zip(evaluable, base) if dt > "20260731"]
    summary_base = {"base_rate_any_day_hits": base_rate_all,
                    "base_rate_jun_jul": float(sum(base_jun_jul) / len(base_jun_jul)) if base_jun_jul else None,
                    "base_rate_aug_oct": float(sum(base_aug_oct) / len(base_aug_oct)) if base_aug_oct else None,
                    "n_crash_days": len(crash_list), "n_evaluable_days": len(evaluable)}
    summary = {
        "base_rate": summary_base,
        "indicator": COL, "publish_lag_bdays": PUBLISH_LAG_BDAYS,
        "rule": f"R_D 확장창(상장 이후 D 이전 전체) 순위 백분위 >= {PCT_THRESH}, 이전 관측 >= {MIN_PRIOR}일, 시차 {PUBLISH_LAG_BDAYS}; "
                f"에피소드 간격 <= {EP_GAP}거래일; 적중창 마지막 빨강 + {HIT_HORIZON}거래일; 급락 = 코스피/SOX 5거래일 내 {CRASH_PCT}% 이하",
        "rows": int(len(w)), "date_range": f"{w.index[0]}~{w.index[-1]}",
        "n_red_days": int(w["red"].sum()), "red_days": red_days,
        "n_episodes": int(len(epdf)), "n_hits": int(epdf["hit"].sum()) if len(epdf) else 0, "false_alarms": false_alarms,
        "crash_days_known": crash_list, "leads": leads, "key_dates": keys,
        "first_evaluable_day": next((dt for dt in kr_days if w.loc[dt, "n_prior"] >= MIN_PRIOR), None),
    }
    w.to_csv(os.path.join(RESULTS, "kr_levetf_warning_daily.csv"), encoding="utf-8-sig")
    epdf.to_csv(os.path.join(RESULTS, "kr_levetf_warning_episodes.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(RESULTS, "kr_levetf_warning_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(w[["ratio", "n_prior", "pct_rank_prior", "red", "kospi_ret1_pct", "kospi_worst5_pct", "sox_worst5_pct_known", "crash"]].round(3).to_string())
    print("\n[episodes]\n", epdf.to_string())
    print("\n[summary]\n", json.dumps({k: v for k, v in summary.items() if k not in ("red_days",)}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
