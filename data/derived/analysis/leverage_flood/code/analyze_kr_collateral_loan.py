"""예탁증권담보융자 잔고를 경고 신호등으로 쓸 수 있는지 전방검증.

입력: ../results/kr_collateral_loan.csv         (fetch_kr_collateral_loan.py, 억원, 데이터 기준일 D)
      ../results/kr_collateral_loan_prices.csv  (yfinance ^KS11·^SOX 종가)
출력: ../results/kr_collateral_loan_signal_daily.csv   (알 수 있는 날 K 기준 일별 지표·신호·앞 수익률)
      ../results/kr_collateral_loan_episodes.csv       (경고 에피소드별 적중/오경보)
      ../results/kr_collateral_loan_events.csv         (6/23·7/29·10/6(·3/4) 사전 상태와 시차)
      ../results/kr_collateral_loan_summary.json

시차: freesis 신용공여 잔고는 T+1 오후 공표(10/14 14:04 조회 시 10/13 행 없음, 10/12 행은 10/13 중 보도).
      → 기준일 D 의 값은 다음 코스피 거래일 K 에 "알 수 있는 날"로 민다. 신호는 K 에 켜진 것으로 본다.

규칙(결과를 보기 전에 정함, 하나만):
      z60 = (x_D − 직전 60거래일 평균(당일 포함)) / 같은 창 표준편차(ddof=1)  ≥ 2  → 빨강.
      레벨 백분위는 1년 추세 상승 구간에서 오래 켜져 있어(1차 판단) 느리고, z60 은 "최근 3개월 대비 급팽창"만 잡는다.
      에피소드 = 켜진 K 가 3거래일 이내 간격으로 이어진 묶음. 적중 = [에피소드 시작, 끝+10거래일] 안에 급락 시작일이 있음.
      급락 시작일 = 그날 종가 대비 이후 5거래일 안 최저 종가가 −8% 이하(코스피 또는 SOX, 각자 달력).
      사건 시차 = 사건일 E 앞 20거래일 안에 켜진 에피소드 중 가장 최근 것의 첫 K 부터 E 까지의 거래일 수.

실행: py analyze_kr_collateral_loan.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
COL = "collateral_loan_100m_krw"
Z_WIN, Z_THR = 60, 2.0
PCT_WIN = 250
GAP_MERGE = 3          # 에피소드 병합 허용 간격(거래일)
HIT_AFTER = 10         # 에피소드 끝 뒤 급락 허용 거래일
CRASH_H, CRASH_THR = 5, -0.08
PRE_WIN = 20           # 사건 앞 창(거래일)
EVAL_START, EVAL_END = "2025-10-01", "2026-10-13"
EVENTS = [("3/4 미·이란(참고)", "2026-03-04"), ("6/23 삼성·하이닉스 서킷", "2026-06-23"),
          ("7/29 코스피 -10.84%", "2026-07-29"), ("10/6~13 5일 -13.8%", "2026-10-06")]
KEY_DATES = ["2026-06-22", "2026-07-28", "2026-10-02", "2026-10-12", "2026-10-13"]


def pct_rank(s: pd.Series, win: int) -> pd.Series:
    def _p(x):
        v = x[-1]
        if np.isnan(v):
            return np.nan
        past = x[:-1][~np.isnan(x[:-1])]
        return 100.0 * np.mean(past <= v) if len(past) else np.nan
    return s.rolling(win, min_periods=win).apply(_p, raw=True)


def crash_starts(close: pd.Series) -> pd.Series:
    """각 날짜에서 이후 1..H 거래일 최저 종가 수익률(자기 달력)."""
    c = close.dropna()
    fwd_min = pd.concat([c.shift(-k) / c - 1 for k in range(1, CRASH_H + 1)], axis=1).min(axis=1)
    return fwd_min


def main() -> int:
    v = pd.read_csv(RESULTS / "kr_collateral_loan.csv", parse_dates=["date"]).sort_values("date")
    px = pd.read_csv(RESULTS / "kr_collateral_loan_prices.csv", parse_dates=["date"]).set_index("date").sort_index()
    kospi = px["kospi_close"].dropna()
    sox = px["sox_close"].dropna()
    cal = kospi.index  # 코스피 거래일 달력

    # 데이터 기준일 D 의 지표(창은 데이터 행 기준) -> 알 수 있는 날 K = D 다음 코스피 거래일
    v = v.set_index("date")
    x = v[COL]
    roll = x.rolling(Z_WIN, min_periods=Z_WIN)
    v["z60"] = (x - roll.mean()) / roll.std(ddof=1)
    v["pct250"] = pct_rank(x, PCT_WIN)
    v["chg20_pct"] = x.pct_change(20) * 100
    v["known_date"] = [cal[cal > d][0] if (cal > d).any() else pd.NaT for d in v.index]
    v = v.reset_index().rename(columns={"date": "data_date"})
    v = v.dropna(subset=["known_date"]).drop_duplicates("known_date", keep="last")

    # 알 수 있는 날 기준 패널(코스피 달력). 공표 안 된 날은 직전 알려진 값을 그대로 든다(그날 아는 최신값).
    d = pd.DataFrame(index=cal)
    d.index.name = "known_date"
    d = d.join(v.set_index("known_date")[["data_date", COL, "z60", "pct250", "chg20_pct"]], how="left")
    d[["data_date", COL, "z60", "pct250", "chg20_pct"]] = d[["data_date", COL, "z60", "pct250", "chg20_pct"]].ffill()
    d["on"] = (d["z60"] >= Z_THR) & d["z60"].notna()
    d["kospi_close"] = kospi
    d["kospi_fwd5_min"] = crash_starts(kospi)
    sox_fwd = crash_starts(sox)
    # SOX 급락 시작일(미국 달력)을 그 날짜 이후 첫 코스피 거래일에 붙인다
    sox_crash_dates = sox_fwd[sox_fwd <= CRASH_THR].index
    d["sox_crash_start"] = False
    for sd in sox_crash_dates:
        nxt = cal[cal >= sd]
        if len(nxt):
            d.loc[nxt[0], "sox_crash_start"] = True
    d["kospi_crash_start"] = d["kospi_fwd5_min"] <= CRASH_THR
    d["crash_start"] = d["kospi_crash_start"] | d["sox_crash_start"]
    d = d.reset_index()
    d.to_csv(RESULTS / "kr_collateral_loan_signal_daily.csv", index=False, encoding="utf-8-sig")

    # 에피소드
    idx = {ts: i for i, ts in enumerate(cal)}
    on_days = d.loc[d["on"], "known_date"].tolist()
    episodes: list[dict] = []
    for k in on_days:
        if episodes and idx[k] - idx[episodes[-1]["end"]] <= GAP_MERGE:
            episodes[-1]["end"] = k
            episodes[-1]["n_on"] += 1
        else:
            episodes.append({"start": k, "end": k, "n_on": 1})
    crash_days = d.loc[d["crash_start"], "known_date"].tolist()
    ep_rows = []
    for ep in episodes:
        i0, i1 = idx[ep["start"]], min(idx[ep["end"]] + HIT_AFTER, len(cal) - 1)
        win_end = cal[i1]
        hits = [c for c in crash_days if ep["start"] <= c <= win_end]
        row = d.set_index("known_date")
        ep_rows.append({
            "start_known": ep["start"].date(), "end_known": ep["end"].date(), "n_on_days": ep["n_on"],
            "span_days": idx[ep["end"]] - i0 + 1, "window_end": win_end.date(),
            "z60_max": round(float(row.loc[ep["start"]:ep["end"], "z60"].max()), 2),
            "value_at_start": row.loc[ep["start"], COL],
            "first_crash_start": hits[0].date() if hits else None,
            "crash_lead_days": (idx[hits[0]] - i0) if hits else None,
            "hit": bool(hits),
            "in_eval_window": pd.Timestamp(EVAL_START) <= ep["start"] <= pd.Timestamp(EVAL_END),
            "window_complete": i1 < len(cal) - 1 or bool(hits),
        })
    eps = pd.DataFrame(ep_rows)
    eps.to_csv(RESULTS / "kr_collateral_loan_episodes.csv", index=False, encoding="utf-8-sig")

    # 사건별 시차
    dd = d.set_index("known_date")
    ev_rows = []
    for label, e in EVENTS:
        E = pd.Timestamp(e)
        iE = idx[E]
        pre = cal[max(0, iE - PRE_WIN):iE]  # E 앞 20거래일(E 제외)
        on_pre = dd.loc[pre, "on"]
        lead = None
        ep_start = None
        if on_pre.any():
            last_on = on_pre[on_pre].index[-1]
            ep = [p for p in episodes if p["start"] <= last_on <= p["end"]][0]
            ep_start = ep["start"]
            lead = iE - idx[ep_start]
        tm1 = cal[iE - 1]
        ev_rows.append({
            "event": label, "event_date": E.date(), "last_pre_known_date": tm1.date(),
            "last_pre_data_date": dd.loc[tm1, "data_date"].date(),
            "value": dd.loc[tm1, COL], "z60": round(float(dd.loc[tm1, "z60"]), 2),
            "pct250": round(float(dd.loc[tm1, "pct250"]), 1), "chg20_pct": round(float(dd.loc[tm1, "chg20_pct"]), 2),
            "on_at_T-1": bool(dd.loc[tm1, "on"]), "on_days_in_pre20": int(on_pre.sum()),
            "episode_start_known": ep_start.date() if ep_start is not None else None,
            "lead_days": lead,
        })
    ev = pd.DataFrame(ev_rows)
    ev.to_csv(RESULTS / "kr_collateral_loan_events.csv", index=False, encoding="utf-8-sig")

    # 요약
    ev_eps = eps[eps["in_eval_window"]] if len(eps) else eps
    valid = dd["z60"].notna() & (dd.index >= EVAL_START) & (dd.index <= EVAL_END)
    # 켜진 날 기준 10거래일 안 급락 시작 확률 vs 기준
    crash_within = pd.Series(False, index=cal)
    ci = [idx[c] for c in crash_days]
    for i, ts in enumerate(cal):
        crash_within[ts] = any(i <= j <= i + HIT_AFTER for j in ci)
    p_on = float(crash_within[dd.index[valid & dd["on"]]].mean() * 100) if (valid & dd["on"]).any() else None
    p_base = float(crash_within[dd.index[valid]].mean() * 100)
    keyvals = {}
    vv = v.set_index("data_date")
    for kd in KEY_DATES:
        ts = pd.Timestamp(kd)
        if ts in vv.index:
            r = vv.loc[ts]
            keyvals[kd] = {"value_100m_krw": float(r[COL]), "pct250": round(float(r["pct250"]), 1),
                           "z60": round(float(r["z60"]), 2), "known_date": str(r["known_date"].date())}
        else:
            keyvals[kd] = "못 받음(freesis 미공표)"
    summary = {
        "rule": f"z60 = (x - mean60)/std60 >= {Z_THR}, 값은 T+1 거래일에 아는 것으로 밀어 평가",
        "data_range": f"{v['data_date'].min().date()}~{v['data_date'].max().date()}", "rows": int(len(v)),
        "eval_window": f"{EVAL_START}~{EVAL_END}",
        "eval_days": int(valid.sum()), "on_days_eval": int((valid & dd["on"]).sum()),
        "episodes_eval": int(len(ev_eps)), "episodes_hit": int(ev_eps["hit"].sum()) if len(ev_eps) else 0,
        "episodes_false_alarm": int((~ev_eps["hit"]).sum()) if len(ev_eps) else 0,
        "p_crash_within10_when_on_pct": None if p_on is None else round(p_on, 1),
        "p_crash_within10_base_pct": round(p_base, 1),
        "crash_start_days_eval": [str(c.date()) for c in crash_days if pd.Timestamp(EVAL_START) <= c <= pd.Timestamp(EVAL_END)],
        "events": ev_rows and [{k: (str(x) if hasattr(x, "isoformat") else x) for k, x in r.items()} for r in ev_rows],
        "key_values": keyvals,
    }
    (RESULTS / "kr_collateral_loan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(ev.to_string())
    print()
    print(eps.to_string())
    print()
    print(json.dumps({k: summary[k] for k in summary if k not in ("events",)}, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
