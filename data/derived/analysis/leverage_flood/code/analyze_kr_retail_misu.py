"""위탁매매 미수금 → 급락 경고 신호등 전방검증.

입력: ../results/kr_retail_misu_history.csv (freesis 0060, 2024-06-03~, 억원)
      ../results/kr_retail_misu_prices.csv  (yfinance ^KS11, ^SOX 종가)
출력: ../results/kr_retail_misu_signal_daily.csv   (알 수 있는 날 기준 일별 신호·앞으로 5일 수익률)
      ../results/kr_retail_misu_episodes.csv       (켜진 에피소드별 적중/오경보)
      ../results/kr_retail_misu_events.csv         (6/23·7/29·10/6 앞 신호 상태)
      ../results/kr_retail_misu_summary.json

시차: freesis 는 T 일 값을 T+1 영업일 오후에 올린다(10/14 14:04 조회 때 10/13 행 없음, 10/12 행은 10/13 보도).
      → 값 날짜 T 의 신호를 "알 수 있는 날" k = T 다음 코스피 거래일로 밀어서 평가한다.
규칙(사건 보기 전에 정함, 하나만): misu_balance 의 직전 250거래일 백분위(그날 제외, 최소 120일) ≥ 95.
급락 정의: k 종가 대비 앞으로 5거래일 안 최저 종가가 코스피 또는 SOX −8% 이하.
에피소드: 켜진 날이 5거래일 이내로 이어지면 한 에피소드. 에피소드 안 어느 하루라도 5일 내 급락이 오면 적중, 아니면 오경보.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

PCT_WINDOW = 250
PCT_MIN = 120
THRESH = 95.0
CRASH = -0.08
FWD = 5
GAP = 5  # 에피소드 병합 간격(거래일)
EVAL_START = pd.Timestamp("2025-10-01")
EVAL_END = pd.Timestamp("2026-10-13")
EVENTS = [("6/23", "2026-06-23"), ("7/29", "2026-07-29"), ("10/6", "2026-10-06")]
KEYDATES = ["2026-06-22", "2026-07-28", "2026-10-02", "2026-10-12", "2026-10-13"]


def rolling_pct_excl(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """그날을 뺀 직전 window 개 값 중 현재값 이하 비율(%)."""
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        past = vals[max(0, i - window):i]
        past = past[~np.isnan(past)]
        if len(past) >= min_periods and not np.isnan(vals[i]):
            out[i] = 100.0 * np.sum(past <= vals[i]) / len(past)
    return pd.Series(out, index=s.index)


def main() -> int:
    misu = pd.read_csv(RESULTS / "kr_retail_misu_history.csv", parse_dates=["date"]).sort_values("date")
    px = pd.read_csv(RESULTS / "kr_retail_misu_prices.csv", parse_dates=["date"]).sort_values("date")
    kospi = px.dropna(subset=["kospi"]).set_index("date")["kospi"]
    sox = px.dropna(subset=["sox"]).set_index("date")["sox"]
    cal = kospi.index  # 코스피 거래일 달력

    # 1) 값 날짜 기준 백분위
    misu = misu.set_index("date")
    misu["pct250"] = rolling_pct_excl(misu["misu_balance"], PCT_WINDOW, PCT_MIN)
    misu["on_raw"] = misu["pct250"] >= THRESH

    # 2) 알 수 있는 날로 밀기: T → T 다음 코스피 거래일
    pos = np.searchsorted(cal.values, misu.index.values, side="right")
    known = np.where(pos < len(cal), cal.values[np.minimum(pos, len(cal) - 1)], np.datetime64("NaT"))
    misu["known_date"] = pd.to_datetime(known)
    misu["value_date"] = misu.index
    # 같은 known_date 에 둘 이상(12/31 휴장 행) → 마지막 값
    daily = misu.dropna(subset=["known_date"]).groupby("known_date").last()

    sig = pd.DataFrame(index=cal)
    sig.index.name = "known_date"
    sig = sig.join(daily[["value_date", "misu_balance", "pct250", "on_raw"]])
    sig["on"] = sig["on_raw"].fillna(False).astype(bool)
    sig["kospi"] = kospi
    # 앞으로 5거래일 최저 종가 수익률(코스피)
    arr = kospi.to_numpy()
    fwd_min = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        w = arr[i + 1:i + 1 + FWD]
        if len(w) == FWD:
            fwd_min[i] = w.min() / arr[i] - 1
    sig["kospi_fwd5_min"] = fwd_min
    # SOX: k 전날까지의 마지막 SOX 종가를 기준, 그 다음 5개 SOX 종가 최저
    sarr = sox.to_numpy()
    sidx = sox.index.values
    sox_fwd = np.full(len(cal), np.nan)
    sox_base = np.full(len(cal), np.nan)
    for i, k in enumerate(cal.values):
        b = np.searchsorted(sidx, k, side="left") - 1  # 날짜 < k 인 마지막
        if b < 0:
            continue
        w = sarr[b + 1:b + 1 + FWD]
        sox_base[i] = sarr[b]
        if len(w) == FWD:
            sox_fwd[i] = w.min() / sarr[b] - 1
    sig["sox_base"] = sox_base
    sig["sox_fwd5_min"] = sox_fwd
    sig["crash5"] = (sig["kospi_fwd5_min"] <= CRASH) | (sig["sox_fwd5_min"] <= CRASH)
    sig.loc[sig["kospi_fwd5_min"].isna() & sig["sox_fwd5_min"].isna(), "crash5"] = np.nan
    # 10거래일 고점 대비 현재 낙폭(에피소드가 급락 뒤에 켜졌는지 보려고)
    sig["kospi_dd10"] = kospi / kospi.rolling(10, min_periods=1).max() - 1
    sig.reset_index().to_csv(RESULTS / "kr_retail_misu_signal_daily.csv", index=False, encoding="utf-8-sig")

    # 3) 평가 구간
    ev = sig[(sig.index >= EVAL_START) & (sig.index <= EVAL_END)].copy()
    valid = ev["pct250"].notna()
    on = ev["on"] & valid
    base_rate = float(ev.loc[valid & ev["crash5"].notna(), "crash5"].mean())
    on_rate = float(ev.loc[on & ev["crash5"].notna(), "crash5"].mean()) if on.any() else float("nan")

    # 에피소드
    on_dates = list(ev.index[on])
    pos_map = {d: i for i, d in enumerate(cal)}
    episodes = []
    cur = []
    for d in on_dates:
        if cur and pos_map[d] - pos_map[cur[-1]] > GAP:
            episodes.append(cur)
            cur = []
        cur.append(d)
    if cur:
        episodes.append(cur)
    ep_rows = []
    for ep in episodes:
        sub = ev.loc[ep]
        c = sub["crash5"]
        if c.notna().any():
            hit = bool(c.dropna().any())
            status = "적중" if hit else "오경보"
            if hit is False and c.isna().any():
                status = "미정(창 미완)"
        else:
            status = "미정(창 미완)"
        ep_rows.append({
            "start_known": ep[0].date(), "end_known": ep[-1].date(), "n_on_days": len(ep),
            "start_value_date": pd.Timestamp(sub["value_date"].iloc[0]).date(),
            "misu_start": round(float(sub["misu_balance"].iloc[0]), 1),
            "misu_max": round(float(sub["misu_balance"].max()), 1),
            "pct250_max": round(float(sub["pct250"].max()), 1),
            "kospi_dd10_at_start_pct": round(float(sub["kospi_dd10"].iloc[0]) * 100, 2),
            "kospi_fwd5_min_pct": round(float(sub["kospi_fwd5_min"].min()) * 100, 2) if sub["kospi_fwd5_min"].notna().any() else None,
            "sox_fwd5_min_pct": round(float(sub["sox_fwd5_min"].min()) * 100, 2) if sub["sox_fwd5_min"].notna().any() else None,
            "status": status,
        })
    ep_df = pd.DataFrame(ep_rows)
    ep_df.to_csv(RESULTS / "kr_retail_misu_episodes.csv", index=False, encoding="utf-8-sig")

    # 4) 사건별 선행
    ev_rows = []
    leads = {}
    for label, e in EVENTS:
        E = pd.Timestamp(e)
        iE = pos_map[E]
        win = cal[max(0, iE - 20):iE]  # E-20 .. E-1 (알 수 있는 날)
        on_win = [d for d in win if bool(sig.loc[d, "on"])]
        lead = (iE - pos_map[on_win[0]]) if on_win else None
        last_on = on_win[-1] if on_win else None
        after = [d for d in cal[iE:iE + 10] if bool(sig.loc[d, "on"])]
        ev_rows.append({
            "event": label, "event_date": E.date(),
            "on_at_E-1": bool(sig.loc[cal[iE - 1], "on"]),
            "pct250_at_E-1": round(float(sig.loc[cal[iE - 1], "pct250"]), 1) if pd.notna(sig.loc[cal[iE - 1], "pct250"]) else None,
            "value_date_at_E-1": pd.Timestamp(sig.loc[cal[iE - 1], "value_date"]).date() if pd.notna(sig.loc[cal[iE - 1], "value_date"]) else None,
            "n_on_in_E-20..E-1": len(on_win),
            "first_on_in_window": on_win[0].date() if on_win else None,
            "last_on_in_window": last_on.date() if last_on is not None else None,
            "lead_trading_days": lead,
            "first_on_at_or_after_E": after[0].date() if after else None,
            "pct250_first_on_after": round(float(sig.loc[after[0], "pct250"]), 1) if after else None,
        })
        leads[label] = lead
    ev_df = pd.DataFrame(ev_rows)
    ev_df.to_csv(RESULTS / "kr_retail_misu_events.csv", index=False, encoding="utf-8-sig")

    # 5) 주요 날짜 값(값 날짜 기준)
    key = {}
    for d in KEYDATES:
        ts = pd.Timestamp(d)
        if ts in misu.index:
            r = misu.loc[ts]
            key[d] = {"misu_balance_eok": round(float(r["misu_balance"]), 1),
                      "pct250": round(float(r["pct250"]), 1),
                      "known_date": str(pd.Timestamp(r["known_date"]).date())}
        else:
            key[d] = {"misu_balance_eok": None, "pct250": None, "note": "freesis 미공표(2026-10-14 14:04 조회)"}

    summary = {
        "rule": f"misu_balance 직전 {PCT_WINDOW}거래일 백분위(그날 제외, 최소 {PCT_MIN}일) >= {THRESH}, T 값은 T+1 거래일에 알 수 있다고 보고 평가",
        "eval_range": f"{EVAL_START.date()}..{EVAL_END.date()} (알 수 있는 날 기준)",
        "n_eval_days": int(valid.sum()),
        "n_on_days": int(on.sum()),
        "p_crash5_base": round(base_rate * 100, 1),
        "p_crash5_when_on": round(on_rate * 100, 1) if not np.isnan(on_rate) else None,
        "n_episodes": len(ep_df),
        "n_hit": int((ep_df["status"] == "적중").sum()) if len(ep_df) else 0,
        "n_false": int((ep_df["status"] == "오경보").sum()) if len(ep_df) else 0,
        "n_pending": int((ep_df["status"] == "미정(창 미완)").sum()) if len(ep_df) else 0,
        "leads": leads,
        "key_values": key,
        "data_range_value_date": f"{misu.index.min().date()}..{misu.index.max().date()}",
        "rows": int(len(misu)),
    }
    (RESULTS / "kr_retail_misu_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    print("\n[episodes]")
    print(ep_df.to_string())
    print("\n[events]")
    print(ev_df.to_string())
    print("\n[on days near events]")
    for label, e in EVENTS:
        E = pd.Timestamp(e)
        iE = pos_map[E]
        w = sig.iloc[iE - 8:iE + 6][["value_date", "misu_balance", "pct250", "on", "kospi", "kospi_fwd5_min", "sox_fwd5_min", "crash5"]]
        print(f"-- {label}")
        print(w.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
