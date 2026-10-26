"""반대매매 비중(미수금 대비)을 '사전에 정한 규칙 하나'로 경고 신호등 전방검증.

입력: ../results/kr_forced_sale.csv (fetch_kr_forced_sale.py, 2024-06~2026-10, 억원·%)
      ../results/us_prices_close.csv 의 ^SOX (fetch_us.py 출력)
출력: ../results/kr_forced_sale_signal_daily.csv  일별 지표·점등·급락 표
      ../results/kr_forced_sale_episodes.csv      점등 에피소드별 분류(선행/동행·후행/오경보)
      ../results/kr_forced_sale_keydates.csv      6/22·7/28·10/2·10/12·10/13 값과 백분위
      ../results/kr_forced_sale_summary.json      요약(시차·규칙·리드·오경보 수)

규칙(결과를 보기 전에 고정, 2026-10-14):
  지표 = 반대매매비중(%) = 당일 반대매매 / 전일 미수금 ×100 (freesis TMPV7)
  점등 = 직전 250거래일(당일 제외, 최소 120일) 안에서 백분위 >= 95  (백분위 = 과거값 <= 당일값 비율×100)
  시차 = T일 값은 T+1 영업일 장 마감 후(16시 이후) 공표 → '아는 날' = T+1 거래일. 매매는 T+2 시가부터.
  에피소드 = 아는 날 기준 점등일을 5거래일 이하 간격으로 묶은 것
  급락 = KOSPI 또는 SOX 의 5거래일 수익률 <= -8% (종가 기준, 각자 달력). 급락일을 5거래일 이하 간격으로 묶고
         시작일 = 첫 급락일 앞 5거래일 안 최고 종가 다음 거래일. SOX 날짜는 그 다음 한국 거래일로 옮긴다.
  분류 = 에피소드 창 [첫 점등-5, 끝 점등+5] 과 급락 구간 [시작일, 마지막 급락일] 이 겹치면
         리드 = 시작일 - 첫 점등(거래일). 리드>=1 선행, <=0 동행·후행. 겹치는 급락이 없으면 오경보.
  사건 리드 = 사건일 앞 20거래일 안 점등(아는 날)이 있으면 '사건일 - 마지막 점등일'(거래일). 없으면 null.
  평가 구간 = 첫 점등일이 2025-10-01~2026-10-13 인 에피소드.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

PCT_WINDOW, PCT_MIN, PCT_THRESH = 250, 120, 95.0
GAP_DAYS = 5
CRASH_RET, CRASH_LOOKBACK = -0.08, 5
EVENT_LOOKBACK = 20
EVAL_START, EVAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2026-10-13")
EVENTS = [("6/23", "2026-06-23"), ("7/29", "2026-07-29"), ("10/6", "2026-10-06")]
KEYDATES = ["2026-03-03", "2026-06-22", "2026-07-28", "2026-10-02", "2026-10-12", "2026-10-13"]


def rolling_pct_excl(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """당일을 제외한 직전 window 일 안에서 당일값의 백분위(과거값 <= 당일값 비율)."""
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        v = vals[i]
        if np.isnan(v):
            continue
        past = vals[max(0, i - window):i]
        past = past[~np.isnan(past)]
        if len(past) >= min_periods:
            out[i] = 100.0 * np.sum(past <= v) / len(past)
    return pd.Series(out, index=s.index)


def crash_episodes(close: pd.Series, dates: pd.DatetimeIndex) -> list[dict]:
    """5거래일 수익률 <= CRASH_RET 인 날을 묶어 급락 구간(시작일·첫 급락일·마지막 급락일·저점)을 만든다."""
    ret5 = close / close.shift(CRASH_LOOKBACK) - 1
    flags = np.where(ret5.to_numpy() <= CRASH_RET)[0]
    eps = []
    for i in flags:
        if eps and i - eps[-1]["last_i"] <= GAP_DAYS:
            eps[-1]["last_i"] = i
        else:
            eps.append({"first_i": i, "last_i": i})
    out = []
    for e in eps:
        c = e["first_i"]
        lo = max(0, c - CRASH_LOOKBACK)
        peak_i = lo + int(np.argmax(close.to_numpy()[lo:c]))  # c 앞 5일 안 최고 종가
        start_i = peak_i + 1
        seg = close.to_numpy()[start_i:e["last_i"] + 1]
        trough_i = start_i + int(np.argmin(seg))
        out.append({
            "start": dates[start_i], "first_crash_day": dates[c], "last_crash_day": dates[e["last_i"]],
            "peak_date": dates[peak_i], "trough_date": dates[trough_i],
            "peak_to_trough_pct": round(float(close.iloc[trough_i] / close.iloc[peak_i] - 1) * 100, 2),
            "worst_ret5_pct": round(float(ret5.iloc[e["first_i"]:e["last_i"] + 1].min()) * 100, 2),
        })
    return out


def main() -> int:
    fs = pd.read_csv(RESULTS / "kr_forced_sale.csv", parse_dates=["date"])
    us = pd.read_csv(RESULTS / "us_prices_close.csv", parse_dates=["Date"])[["Date", "^SOX"]].rename(
        columns={"Date": "date", "^SOX": "sox"}).dropna()

    # 한국 거래일 달력 = KOSPI 지수가 있는 날. 지수 없는 날(12/31 등)에 찍힌 반대매매 값은 다음 거래일로 옮기지 않고 버린다(확인용 출력).
    kr = fs[fs["kospi_index"].notna()].copy().reset_index(drop=True)
    dropped = fs[fs["kospi_index"].isna() & fs["forced_sale_ratio_pct"].notna()]
    kr["pct250_ratio"] = rolling_pct_excl(kr["forced_sale_ratio_pct"], PCT_WINDOW, PCT_MIN)
    kr["pct250_amount"] = rolling_pct_excl(kr["forced_sale_amount"], PCT_WINDOW, PCT_MIN)  # 참고용(규칙 아님)
    kr["on_T"] = kr["pct250_ratio"] >= PCT_THRESH
    # 아는 날 = T+1 거래일. 마지막 행(아직 다음 거래일이 없으면)은 다음 영업일 추정 대신 NaT.
    kr["known_date"] = kr["date"].shift(-1)
    kr["on_known"] = kr["on_T"].shift(1, fill_value=False)  # 아는 날 기준으로 민 점등
    kr["ret5_kospi"] = kr["kospi_index"] / kr["kospi_index"].shift(CRASH_LOOKBACK) - 1
    kr["fwd5min_kospi"] = pd.concat([kr["kospi_index"].shift(-k) for k in range(1, 6)], axis=1).min(axis=1) / kr["kospi_index"] - 1

    # SOX: 미국 달력에서 5일 수익률 → 그 다음 한국 거래일에 붙인다(미국 종가는 다음날 아침 한국에서 본다)
    us = us.sort_values("date").reset_index(drop=True)
    us["ret5_sox"] = us["sox"] / us["sox"].shift(CRASH_LOOKBACK) - 1
    us["kr_date"] = us["date"] + pd.Timedelta(days=1)
    kr_dates = pd.DatetimeIndex(kr["date"])
    us["kr_date"] = [kr_dates[kr_dates.searchsorted(d)] if kr_dates.searchsorted(d) < len(kr_dates) else pd.NaT for d in us["kr_date"]]
    us_last = us.dropna(subset=["kr_date"]).groupby("kr_date").last()[["sox", "ret5_sox"]]
    kr = kr.merge(us_last, left_on="date", right_index=True, how="left")
    kr[["sox", "ret5_sox"]] = kr[["sox", "ret5_sox"]].ffill()
    kr["crash_day_kospi"] = kr["ret5_kospi"] <= CRASH_RET
    kr["crash_day_sox"] = kr["ret5_sox"] <= CRASH_RET
    kr["crash_day"] = kr["crash_day_kospi"] | kr["crash_day_sox"]

    # 급락 구간(KOSPI 는 한국 달력, SOX 는 미국 달력 → 한국 거래일로 옮김)
    ce_k = crash_episodes(kr["kospi_index"], kr_dates)
    for e in ce_k:
        e["index"] = "KOSPI"
    us_dates = pd.DatetimeIndex(us["date"])
    ce_s = crash_episodes(us["sox"].reset_index(drop=True), us_dates)
    for e in ce_s:
        e["index"] = "SOX"
        for k in ["start", "first_crash_day", "last_crash_day", "peak_date", "trough_date"]:
            j = kr_dates.searchsorted(e[k] + pd.Timedelta(days=1))
            e[k + "_us"] = e[k]
            e[k] = kr_dates[j] if j < len(kr_dates) else pd.NaT
    crashes = sorted(ce_k + ce_s, key=lambda e: e["start"])
    for e in crashes:
        e["start_i"] = int(kr_dates.get_loc(e["start"])) if pd.notna(e["start"]) else None
        e["last_i"] = int(kr_dates.get_loc(e["last_crash_day"])) if pd.notna(e["last_crash_day"]) else None

    # 점등 에피소드(아는 날 기준)
    on_idx = np.where(kr["on_known"].to_numpy())[0]
    eps = []
    for i in on_idx:
        if eps and i - eps[-1]["last_i"] <= GAP_DAYS:
            eps[-1]["last_i"] = i
            eps[-1]["n_on"] += 1
        else:
            eps.append({"first_i": int(i), "last_i": int(i), "n_on": 1})
    ep_rows = []
    for e in eps:
        f, l = e["first_i"], e["last_i"]
        seg = kr.iloc[max(0, f - 1):l]  # T 기준 행(아는 날-1)
        wlo, whi = f - GAP_DAYS, l + GAP_DAYS
        hits = [c for c in crashes if c["start_i"] is not None and c["last_i"] is not None
                and not (c["last_i"] < wlo or c["start_i"] > whi)]
        if hits:
            c = min(hits, key=lambda x: x["start_i"])
            lead = c["start_i"] - f
            kind = "선행" if lead >= 1 else "동행·후행"
        else:
            c, lead, kind = None, None, "오경보"
        in_eval = EVAL_START <= kr_dates[f] <= EVAL_END
        ep_rows.append({
            "first_on_known": kr_dates[f].date(), "last_on_known": kr_dates[l].date(), "n_on_days": e["n_on"],
            "span_days": l - f + 1,
            "peak_ratio_pct": float(seg["forced_sale_ratio_pct"].max()), "peak_amount_eok": float(seg["forced_sale_amount"].max()),
            "peak_pct250": round(float(seg["pct250_ratio"].max()), 1),
            "kospi_at_first_on": float(kr["kospi_index"].iloc[f]),
            "kospi_fwd5_from_first_on_pct": round(float(kr["fwd5min_kospi"].iloc[f]) * 100, 2) if pd.notna(kr["fwd5min_kospi"].iloc[f]) else None,
            "kospi_chg_prev5_pct": round(float(kr["ret5_kospi"].iloc[f]) * 100, 2),
            "crash_index": c["index"] if c else None, "crash_start": c["start"].date() if c else None,
            "crash_peak_to_trough_pct": c["peak_to_trough_pct"] if c else None,
            "lead_days": lead, "type": kind, "in_eval_period": in_eval,
        })
    ep_df = pd.DataFrame(ep_rows)

    # 세 사건 리드
    event_rows, leads = [], {}
    for label, d in EVENTS:
        ci = int(kr_dates.get_loc(pd.Timestamp(d)))
        prior = [i for i in on_idx if ci - EVENT_LOOKBACK <= i <= ci - 1]
        if prior:
            last_on, first_on = max(prior), min(prior)
            lead = ci - last_on
            leads[label] = lead
            event_rows.append({"event": label, "crash_date": d, "lit_within_20d": True,
                               "first_lit_known": kr_dates[first_on].date(), "last_lit_known": kr_dates[last_on].date(),
                               "lead_days_from_last_lit": lead, "lead_days_from_first_lit": ci - first_on,
                               "n_lit_days": len(prior),
                               "ratio_at_T-1": float(kr["forced_sale_ratio_pct"].iloc[ci - 1]),
                               "pct250_at_T-1": round(float(kr["pct250_ratio"].iloc[ci - 1]), 1)})
        else:
            leads[label] = None
            event_rows.append({"event": label, "crash_date": d, "lit_within_20d": False,
                               "first_lit_known": None, "last_lit_known": None, "lead_days_from_last_lit": None,
                               "lead_days_from_first_lit": None, "n_lit_days": 0,
                               "ratio_at_T-1": float(kr["forced_sale_ratio_pct"].iloc[ci - 1]),
                               "pct250_at_T-1": round(float(kr["pct250_ratio"].iloc[ci - 1]), 1)})
    ev_df = pd.DataFrame(event_rows)

    # 기준일 값
    kd = kr.set_index("date")
    key_rows = []
    for d in KEYDATES:
        ts = pd.Timestamp(d)
        if ts in kd.index:
            r = kd.loc[ts]
            key_rows.append({"date": d, "forced_sale_amount_eok": r["forced_sale_amount"], "forced_sale_ratio_pct": r["forced_sale_ratio_pct"],
                             "misu_balance_eok": r["misu_balance"], "pct250_ratio": round(r["pct250_ratio"], 1) if pd.notna(r["pct250_ratio"]) else None,
                             "pct250_amount": round(r["pct250_amount"], 1) if pd.notna(r["pct250_amount"]) else None,
                             "on_T": bool(r["on_T"]) if pd.notna(r["pct250_ratio"]) else None,
                             "known_date": r["known_date"].date() if pd.notna(r["known_date"]) else None, "kospi": r["kospi_index"]})
    key_df = pd.DataFrame(key_rows)

    # 요약 통계(평가 구간, 아는 날 기준)
    ev = kr[(kr["date"] >= EVAL_START) & (kr["date"] <= EVAL_END) & kr["pct250_ratio"].shift(1).notna()]
    on_days = int(ev["on_known"].sum())
    p_on = float((ev.loc[ev["on_known"], "fwd5min_kospi"] <= CRASH_RET).mean() * 100) if on_days else None
    p_base = float((ev["fwd5min_kospi"] <= CRASH_RET).mean() * 100)
    ep_eval = ep_df[ep_df["in_eval_period"]] if len(ep_df) else ep_df
    summary = {
        "rule": f"반대매매비중(TMPV7) 직전 {PCT_WINDOW}거래일 백분위(당일 제외, 최소 {PCT_MIN}일) >= {PCT_THRESH:.0f} → 점등. T값은 T+1 거래일에 아는 것으로 밀어 평가. 에피소드=점등일 {GAP_DAYS}거래일 이하 간격 묶음. 급락=KOSPI 또는 SOX 5거래일 <= {CRASH_RET*100:.0f}%.",
        "data_range": f"{kr['date'].min().date()}~{kr['date'].max().date()}",
        "rows_forced_sale": int(kr["forced_sale_ratio_pct"].notna().sum()),
        "dropped_non_trading_rows": [d.date().isoformat() for d in dropped["date"]],
        "eval_period": f"{EVAL_START.date()}~{EVAL_END.date()}",
        "eval_days": int(len(ev)), "on_days_known": on_days,
        "p_kospi_fwd5_le_-8_when_on_pct": round(p_on, 1) if p_on is not None else None,
        "p_kospi_fwd5_le_-8_base_pct": round(p_base, 1),
        "episodes_total_all_data": int(len(ep_df)),
        "episodes_eval": int(len(ep_eval)),
        "episodes_eval_by_type": {k: int(v) for k, v in ep_eval["type"].value_counts().items()} if len(ep_eval) else {},
        "false_alarms_eval": int((ep_eval["type"] == "오경보").sum()) if len(ep_eval) else 0,
        "event_leads": leads,
        "crash_episodes": [{k: (v.date().isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in c.items()
                            if k in ("index", "start", "first_crash_day", "last_crash_day", "peak_date", "trough_date", "peak_to_trough_pct", "worst_ret5_pct")}
                           for c in crashes if pd.notna(c["start"]) and c["start"] >= pd.Timestamp("2025-06-01")],
    }

    daily_cols = ["date", "known_date", "forced_sale_amount", "forced_sale_ratio_pct", "misu_balance", "pct250_ratio", "pct250_amount",
                  "on_T", "on_known", "kospi_index", "ret5_kospi", "fwd5min_kospi", "sox", "ret5_sox", "crash_day_kospi", "crash_day_sox", "crash_day"]
    kr[daily_cols].to_csv(RESULTS / "kr_forced_sale_signal_daily.csv", index=False, encoding="utf-8-sig")
    ep_df.to_csv(RESULTS / "kr_forced_sale_episodes.csv", index=False, encoding="utf-8-sig")
    ev_df.to_csv(RESULTS / "kr_forced_sale_event_leads.csv", index=False, encoding="utf-8-sig")
    key_df.to_csv(RESULTS / "kr_forced_sale_keydates.csv", index=False, encoding="utf-8-sig")
    (RESULTS / "kr_forced_sale_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print("\n[급락 구간]")
    print(pd.DataFrame(summary["crash_episodes"]).to_string())
    print("\n[점등 에피소드]")
    print(ep_df.to_string())
    print("\n[사건 리드]")
    print(ev_df.to_string())
    print("\n[기준일]")
    print(key_df.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
