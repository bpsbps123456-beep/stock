"""지표 #2 투자자예탁금·신용/예탁금 비율 — 사전 규칙 하나로 전방검증 (갈래 kr_retail).

입력: ../results/kr_retail_deposit_ratio.csv  (fetch_kr_retail_deposit.py, 2024-01-02~, 억원)
      ../results/kr_retail_deposit_prices.csv (yfinance ^KS11·^SOX 종가)
출력: ../results/kr_retail_deposit_signal_daily.csv  (거래일별 '알 수 있는 날' 기준 신호)
      ../results/kr_retail_deposit_episodes.csv      (켜진 에피소드와 5일 내 급락 여부)
      ../results/kr_retail_deposit_events.csv        (세 급락 앞 리드 일수)
      ../results/kr_retail_deposit_summary.json

사전 규칙(결과 보기 전에 고정, 2026-10-14):
  RULE: 신용융자잔고/투자자예탁금(credit_to_deposit) 의 직전 250거래일(그날 포함, 250개가 다 있어야 계산)
        백분위 >= 95  → 빨강.  백분위 = 과거 249개 중 현재값 이하인 개수 / 249 × 100.
  시차: T일 값은 T+1 영업일 오후·저녁 공표 → '알 수 있는 날' = T 다음 코스피 거래일. 신호는 그날 저녁에 켜진다.
  급락: 알 수 있는 날 t 종가 대비 다음 5거래일 종가 최저가 -8% 이하(코스피 ^KS11),
        또는 SOX: t 전날까지의 마지막 종가 대비 다음 5거래일 종가 최저가 -8% 이하.
  에피소드: 빨강이 연속한 구간. 꺼진 날이 5거래일 이하면 같은 에피소드로 묶는다.
  적중: 에피소드 안의 어느 날이든 그날 기준 5일 내 급락이 있으면 적중, 아니면 오경보.
  리드: 사건일 E 앞 20거래일(E-20~E-1) 안에서 처음 켜진 날부터 E 까지의 거래일 수. 없으면 null.
  평가 구간: 알 수 있는 날 2025-10-01 ~ 2026-10-14 (마지막 5거래일은 5일 뒤가 없어 미확정으로 따로 센다).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

RULE_NAME = "credit_to_deposit pct250 >= 95"
PCT_WINDOW = 250
PCT_THRESHOLD = 95.0
DROP_THRESHOLD = -0.08
FWD_DAYS = 5
EPISODE_GAP = 5
PRE_WINDOW = 20
EVAL_START = pd.Timestamp("2025-10-01")
EVAL_END = pd.Timestamp("2026-10-14")
EVENTS = [
    ("3/4 미·이란 -12.06% (참고)", "2026-03-04"),
    ("6/23 삼성·하이닉스 -12% 서킷", "2026-06-23"),
    ("7/29 코스피 -10.84%", "2026-07-29"),
    ("10/6~13 5일 -13.8%", "2026-10-06"),
]
KEYDATES = ["2026-06-22", "2026-07-28", "2026-10-02", "2026-10-12", "2026-10-13"]


def rolling_pct(s: pd.Series, window: int) -> pd.Series:
    def _p(x):
        v = x[-1]
        past = x[:-1]
        if np.isnan(v) or np.isnan(past).any():
            return np.nan
        return 100.0 * np.sum(past <= v) / len(past)
    return s.rolling(window, min_periods=window).apply(_p, raw=True)


def main() -> int:
    d = pd.read_csv(RESULTS / "kr_retail_deposit_ratio.csv", parse_dates=["date"]).sort_values("date")
    px = pd.read_csv(RESULTS / "kr_retail_deposit_prices.csv", parse_dates=["date"]).sort_values("date")
    kospi = px.dropna(subset=["kospi_close"])[["date", "kospi_close"]].reset_index(drop=True)
    sox = px.dropna(subset=["sox_close"])[["date", "sox_close"]].reset_index(drop=True)

    # 1) 데이터 날짜 기준 지표
    d = d.dropna(subset=["credit_to_deposit"]).reset_index(drop=True)
    d["pct250_credit_to_deposit"] = rolling_pct(d["credit_to_deposit"], PCT_WINDOW)
    d["pct250_investor_deposit"] = rolling_pct(d["investor_deposit"], PCT_WINDOW)  # 설명용(규칙 아님)
    d["signal_raw"] = d["pct250_credit_to_deposit"] >= PCT_THRESHOLD

    # 2) 알 수 있는 날 = 데이터 날짜 다음 코스피 거래일
    kdates = kospi["date"].to_numpy()
    idx = np.searchsorted(kdates, d["date"].to_numpy(), side="right")  # 첫 거래일 > T
    d["known_date"] = pd.Series([kdates[i] if i < len(kdates) else pd.NaT for i in idx])
    known = d.dropna(subset=["known_date"]).groupby("known_date").tail(1)  # 같은 날 둘이면 최신 데이터
    known = known.rename(columns={"date": "data_date"})

    # 3) 거래일 패널 + 사후 급락
    panel = kospi.copy()
    panel["kospi_fwd5_min"] = (
        panel["kospi_close"][::-1].rolling(FWD_DAYS, min_periods=FWD_DAYS).min()[::-1].shift(-1) / panel["kospi_close"] - 1
    )
    sox_vals = sox["sox_close"].to_numpy()
    sox_dates = sox["date"].to_numpy()
    sox_fwd = []
    for t in panel["date"]:
        j = np.searchsorted(sox_dates, np.datetime64(t), side="left") - 1  # 마지막 SOX 종가(날짜 < t)
        if j < 0 or j + FWD_DAYS >= len(sox_vals):
            sox_fwd.append(np.nan)
        else:
            sox_fwd.append(sox_vals[j + 1:j + 1 + FWD_DAYS].min() / sox_vals[j] - 1)
    panel["sox_fwd5_min"] = sox_fwd
    panel["drop8_fwd5"] = (panel["kospi_fwd5_min"] <= DROP_THRESHOLD) | (panel["sox_fwd5_min"] <= DROP_THRESHOLD)
    panel.loc[panel["kospi_fwd5_min"].isna() & panel["sox_fwd5_min"].isna(), "drop8_fwd5"] = np.nan

    cols = ["known_date", "data_date", "investor_deposit", "credit_loan_total", "credit_to_deposit",
            "pct250_credit_to_deposit", "pct250_investor_deposit", "signal_raw"]
    panel = panel.merge(known[cols], left_on="date", right_on="known_date", how="left").drop(columns=["known_date"])
    # 공표가 없는 거래일(휴장 뒤 등)은 마지막 공표값을 그대로 든다
    for c in cols[1:]:
        panel[c] = panel[c].ffill()
    panel["signal"] = panel["signal_raw"].astype("boolean").fillna(False).astype(bool)
    panel = panel.rename(columns={"date": "date"})
    panel.to_csv(RESULTS / "kr_retail_deposit_signal_daily.csv", index=False, encoding="utf-8-sig")

    ev = panel[(panel["date"] >= EVAL_START) & (panel["date"] <= EVAL_END)].reset_index(drop=True)

    # 4) 에피소드
    episodes = []
    on_idx = ev.index[ev["signal"]].tolist()
    if on_idx:
        start = prev = on_idx[0]
        for i in on_idx[1:] + [None]:
            if i is None or i - prev > EPISODE_GAP + 1:
                seg = ev.loc[start:prev]
                on_days = int(seg["signal"].sum())
                drops = seg.loc[seg["signal"], "drop8_fwd5"]
                hit = bool(drops.fillna(False).any())
                undecided = bool(drops.isna().any()) and not hit
                worst_k = seg.loc[seg["signal"], "kospi_fwd5_min"].min()
                worst_s = seg.loc[seg["signal"], "sox_fwd5_min"].min()
                episodes.append({
                    "start": seg["date"].iloc[0].date(), "end": seg["date"].iloc[-1].date(),
                    "on_days": on_days, "max_pct250": round(float(seg["pct250_credit_to_deposit"].max()), 1),
                    "min_kospi_fwd5": None if pd.isna(worst_k) else round(float(worst_k) * 100, 2),
                    "min_sox_fwd5": None if pd.isna(worst_s) else round(float(worst_s) * 100, 2),
                    "hit_drop8": hit, "undecided": undecided,
                })
                if i is not None:
                    start = i
            if i is not None:
                prev = i
    ep = pd.DataFrame(episodes)
    ep.to_csv(RESULTS / "kr_retail_deposit_episodes.csv", index=False, encoding="utf-8-sig")

    # 5) 사건별 리드
    ev_idx = panel.set_index("date")
    rows = []
    for label, e in EVENTS:
        e_ts = pd.Timestamp(e)
        if e_ts not in ev_idx.index:
            rows.append({"event": label, "event_date": e, "lead_days": None, "note": "거래일 아님"})
            continue
        pos = panel.index[panel["date"] == e_ts][0]
        pre = panel.iloc[max(0, pos - PRE_WINDOW):pos]
        on_pre = pre[pre["signal"]]
        first_on = on_pre["date"].iloc[0] if len(on_pre) else None
        lead = int(pos - panel.index[panel["date"] == first_on][0]) if first_on is not None else None
        tm1 = panel.iloc[pos - 1]
        rows.append({
            "event": label, "event_date": e,
            "on_at_E-1": bool(tm1["signal"]),
            "E-1_known_date": tm1["date"].date(), "E-1_data_date": pd.Timestamp(tm1["data_date"]).date(),
            "E-1_credit_to_deposit": round(float(tm1["credit_to_deposit"]), 3),
            "E-1_pct250": round(float(tm1["pct250_credit_to_deposit"]), 1),
            "on_days_in_pre20": int(len(on_pre)),
            "first_on_in_pre20": first_on.date() if first_on is not None else None,
            "lead_days": lead,
            "max_pct250_pre20": round(float(pre["pct250_credit_to_deposit"].max()), 1),
        })
    evdf = pd.DataFrame(rows)
    evdf.to_csv(RESULTS / "kr_retail_deposit_events.csv", index=False, encoding="utf-8-sig")

    # 6) 기준일 값(데이터 날짜 기준) + 요약
    di = d.set_index("date")
    key = {}
    for k in KEYDATES:
        ts = pd.Timestamp(k)
        if ts in di.index:
            r = di.loc[ts]
            key[k] = {
                "investor_deposit_억원": round(float(r["investor_deposit"]), 0),
                "credit_loan_total_억원": round(float(r["credit_loan_total"]), 0),
                "credit_to_deposit": round(float(r["credit_to_deposit"]), 3),
                "pct250_credit_to_deposit": round(float(r["pct250_credit_to_deposit"]), 1),
                "pct250_investor_deposit": round(float(r["pct250_investor_deposit"]), 1),
                "known_date": str(pd.Timestamp(r["known_date"]).date()),
            }
        else:
            key[k] = "못 받음(freesis 미공표, 2026-10-14 14:04 조회)"
    valid = ev.dropna(subset=["drop8_fwd5"])
    base_rate = float(valid["drop8_fwd5"].astype(bool).mean() * 100)
    on_valid = valid[valid["signal"]]
    on_rate = float(on_valid["drop8_fwd5"].astype(bool).mean() * 100) if len(on_valid) else None
    summary = {
        "rule": RULE_NAME,
        "rule_detail": "신용융자잔고/투자자예탁금 250거래일(그날 포함, 250개 필수) 백분위 >= 95, T+1 거래일 저녁에 확인",
        "eval_window_known_dates": [str(EVAL_START.date()), str(ev["date"].max().date())],
        "eval_days": int(len(ev)), "on_days": int(ev["signal"].sum()),
        "p_drop8_fwd5_when_on_pct": None if on_rate is None else round(on_rate, 1),
        "p_drop8_fwd5_base_pct": round(base_rate, 1),
        "episodes": len(ep), "episodes_hit": int(ep["hit_drop8"].sum()) if len(ep) else 0,
        "episodes_false_alarm": int((~ep["hit_drop8"] & ~ep["undecided"]).sum()) if len(ep) else 0,
        "episodes_undecided": int(ep["undecided"].sum()) if len(ep) else 0,
        "events": rows,
        "key_values": key,
        "data_range": [str(d["date"].min().date()), str(d["date"].max().date())],
        "pct_valid_from": str(d.dropna(subset=["pct250_credit_to_deposit"])["date"].min().date()),
    }
    (RESULTS / "kr_retail_deposit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("events", "key_values")}, ensure_ascii=False))
    print(evdf.to_string())
    print(ep.to_string())
    print(json.dumps(key, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
