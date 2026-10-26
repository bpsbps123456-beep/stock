"""신용융자 잔고(합계·유가·코스닥) — 사전 확정 경고 규칙 전방검증(공표 시차 반영).

입력: ../results/kr_retail_margin_balance.csv (fetch_kr_retail.py 출력, 억원)
      yfinance ^KS11(KOSPI), ^SOX(필라델피아 반도체) 종가
출력: ../results/kr_retail_margin_rule_daily.csv     신호일(알 수 있는 날) 기준 일별 패널
      ../results/kr_retail_margin_rule_episodes.csv  빨강 에피소드별 적중/오경보
      ../results/kr_retail_margin_rule_keyvalues.csv 6/22·7/28·10/2·10/13 값
      ../results/kr_retail_margin_rule_summary.json  요약
      ../results/kr_retail_margin_rule_prices.csv    사용한 KOSPI·SOX 종가(재현용)

규칙(결과를 보기 전에 고정, 2026-10-14):
  z60 = (D일 잔고 − 직전 60거래일 평균(D 포함)) / 60거래일 표준편차.  z60 >= 2 이면 빨강.
  참고선(1차 분석 규칙): 250일 백분위 >= 95 (같은 틀로 다시 잼, 판정에는 쓰지 않음)
공표 시차: D일 값은 D+1 영업일 오후(14:04 조회 때 전일 행 없음)~저녁에 freesis 에 올라온다.
  → 신호일 K = D 다음 KOSPI 거래일.  K 장중에는 못 쓰므로 K 종가를 기준가로 두고 K+1 부터 잰다.
급락: KOSPI 는 K 종가 대비 K+1..K+5 최저 종가 <= -8%,
      SOX 는 K 직전 미국 세션 종가 대비 K..K+4 세션 최저 종가 <= -8% (K 저녁 22:30 미국 개장 전에 신호를 보므로).
에피소드: 빨강이 꺼진 날이 5거래일 이하이면 같은 에피소드. 에피소드 안의 어느 신호일에서든 급락이 시작되면 적중, 아니면 오경보.
시차 반영 선행일: 사건일 E 앞 10거래일 안에 켜져 있던 에피소드의 첫 빨강 신호일부터 E 까지 거래일 수.

실행: py forward_test_kr_retail_margin.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

RULE_WINDOW = 60
RULE_Z = 2.0
REF_PCT = 95.0
CRASH_H = 5
CRASH_PCT = -0.08
EPISODE_GAP = 5
LEAD_LOOKBACK = 10
EVAL_START, EVAL_END = "2025-10-01", "2026-10-13"
PRICE_END = "2026-10-13"  # 10/14 는 장중 값이라 제외
EVENTS = {"6/23": "2026-06-23", "7/29": "2026-07-29", "10/6": "2026-10-06"}
KEYDATES = ["2026-06-22", "2026-07-28", "2026-10-02", "2026-10-13"]
SERIES = ["credit_loan_total", "credit_loan_kospi", "credit_loan_kosdaq"]


def rolling_pct(s: pd.Series, window: int = 250, min_periods: int = 120) -> pd.Series:
    """1차 분석과 같은 정의: 창 안의 과거값(오늘 제외) 중 오늘 이하인 비율."""
    def _p(x):
        v = x[-1]
        if np.isnan(v) or len(x) <= 1:
            return np.nan
        past = x[:-1][~np.isnan(x[:-1])]
        return 100.0 * np.sum(past <= v) / len(past) if len(past) else np.nan
    return s.rolling(window, min_periods=min_periods).apply(_p, raw=True)


def zscore(s: pd.Series, window: int = RULE_WINDOW) -> pd.Series:
    m = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=1)
    return (s - m) / sd


def load_prices() -> tuple[pd.Series, pd.Series]:
    out = {}
    for tk in ["^KS11", "^SOX"]:
        df = yf.download(tk, start="2025-05-01", end="2026-10-15", auto_adjust=False, progress=False)
        c = df["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c.index = pd.to_datetime(c.index).tz_localize(None)
        c = c[c.index <= pd.Timestamp(PRICE_END)].dropna()
        out[tk] = c
    return out["^KS11"], out["^SOX"]


def fwd_min_ret(close: pd.Series, h: int) -> pd.Series:
    """각 날 종가 대비 이후 1..h 세션 최저 종가 수익률(앞 h 일이 없으면 NaN)."""
    arr = close.to_numpy()
    n = len(arr)
    res = np.full(n, np.nan)
    for i in range(n):
        if i + h < n:
            res[i] = arr[i + 1:i + h + 1].min() / arr[i] - 1
    return pd.Series(res, index=close.index)


def build_panel(margin: pd.DataFrame, kospi: pd.Series, sox: pd.Series) -> pd.DataFrame:
    tdays = kospi.index  # KOSPI 거래일 달력
    m = margin.sort_values("date").reset_index(drop=True).copy()
    for c in SERIES:
        m[f"z60_{c}"] = zscore(m[c])
        m[f"pct250_{c}"] = rolling_pct(m[c])
    # 알 수 있는 날 K = D 다음 거래일
    pos = tdays.searchsorted(m["date"].to_numpy(), side="right")
    m["known_day"] = [tdays[p] if p < len(tdays) else pd.NaT for p in pos]
    m = m.dropna(subset=["known_day"])
    # 같은 K 에 두 개 D 가 오면(연말 휴장일 행) 마지막 D 만 쓴다
    m = m.groupby("known_day").tail(1).set_index("known_day")

    panel = pd.DataFrame(index=tdays)
    panel.index.name = "signal_day"
    panel["data_date"] = m["date"]
    for c in SERIES:
        panel[c] = m[c]
        panel[f"z60_{c}"] = m[f"z60_{c}"]
        panel[f"pct250_{c}"] = m[f"pct250_{c}"]
    # 새 값이 없는 거래일은 직전 상태를 그대로 쓴다(공표 누락·휴일 어긋남)
    panel = panel.ffill()
    panel["on_primary"] = panel["z60_credit_loan_total"] >= RULE_Z
    panel["on_ref_pct95"] = panel["pct250_credit_loan_total"] >= REF_PCT
    panel["on_primary_kospi"] = panel["z60_credit_loan_kospi"] >= RULE_Z
    panel["on_primary_kosdaq"] = panel["z60_credit_loan_kosdaq"] >= RULE_Z

    panel["kospi_close"] = kospi
    panel["kospi_fwd5_min"] = fwd_min_ret(kospi, CRASH_H)
    # SOX: K 이상 첫 세션 idx, 기준가는 idx-1 세션, 창은 idx..idx+4
    sarr = sox.to_numpy()
    sidx = sox.index
    sox_fwd = []
    sox_base_date = []
    for k in tdays:
        i = sidx.searchsorted(k, side="left")
        if i - 1 < 0 or i + CRASH_H - 1 >= len(sarr):
            sox_fwd.append(np.nan)
            sox_base_date.append(pd.NaT)
            continue
        sox_fwd.append(sarr[i:i + CRASH_H].min() / sarr[i - 1] - 1)
        sox_base_date.append(sidx[i - 1])
    panel["sox_base_date"] = sox_base_date
    panel["sox_fwd5_min"] = sox_fwd
    panel["crash_kospi"] = panel["kospi_fwd5_min"] <= CRASH_PCT
    panel["crash_sox"] = panel["sox_fwd5_min"] <= CRASH_PCT
    panel["crash_any"] = panel["crash_kospi"] | panel["crash_sox"]
    panel["crash_defined"] = panel["kospi_fwd5_min"].notna() & panel["sox_fwd5_min"].notna()
    return panel.reset_index()


def episodes(panel: pd.DataFrame, on_col: str) -> pd.DataFrame:
    p = panel[(panel["signal_day"] >= EVAL_START) & (panel["signal_day"] <= EVAL_END)].reset_index(drop=True)
    on_idx = p.index[p[on_col].fillna(False).astype(bool)].to_list()
    groups: list[list[int]] = []
    for i in on_idx:
        if groups and i - groups[-1][-1] - 1 <= EPISODE_GAP:
            groups[-1].append(i)
        else:
            groups.append([i])
    rows = []
    for g in groups:
        sub = p.loc[g]
        crash_days = sub[sub["crash_any"]]
        first_crash = crash_days.iloc[0] if len(crash_days) else None
        rows.append({
            "rule": on_col,
            "start": p.loc[g[0], "signal_day"].date(),
            "end": p.loc[g[-1], "signal_day"].date(),
            "on_days": len(g),
            "span_tdays": g[-1] - g[0] + 1,
            "hit": bool(len(crash_days)),
            "first_crash_signal_day": first_crash["signal_day"].date() if first_crash is not None else None,
            "first_crash_kospi_fwd5": round(float(first_crash["kospi_fwd5_min"]) * 100, 2) if first_crash is not None else None,
            "first_crash_sox_fwd5": round(float(first_crash["sox_fwd5_min"]) * 100, 2) if first_crash is not None else None,
            "max_z60_total": round(float(sub["z60_credit_loan_total"].max()), 2),
        })
    return pd.DataFrame(rows)


def lead_days(panel: pd.DataFrame, eps: pd.DataFrame, event: str) -> dict:
    tdays = pd.DatetimeIndex(panel["signal_day"])
    e = pd.Timestamp(event)
    iE = tdays.get_loc(e)
    out = {"event": event, "on_at_T-1": bool(panel.loc[iE - 1, "on_primary"]), "lead_tdays": None,
           "episode_start": None, "episode_end": None,
           "z60_at_T-1": round(float(panel.loc[iE - 1, "z60_credit_loan_total"]), 2),
           "data_date_at_T-1": panel.loc[iE - 1, "data_date"].date()}
    if eps.empty:
        return out
    cands = []
    for _, r in eps.iterrows():
        i_s = tdays.get_loc(pd.Timestamp(r["start"]))
        i_e = tdays.get_loc(pd.Timestamp(r["end"]))
        if i_s <= iE - 1 and i_e >= iE - LEAD_LOOKBACK:
            cands.append((i_s, i_e, r))
    if cands:
        i_s, i_e, r = max(cands, key=lambda t: t[0])
        out.update({"lead_tdays": int(iE - i_s), "episode_start": r["start"], "episode_end": r["end"]})
    return out


def main() -> int:
    margin = pd.read_csv(RESULTS / "kr_retail_margin_balance.csv", parse_dates=["date"])
    kospi, sox = load_prices()
    prices = pd.DataFrame({"kospi_close": kospi, "sox_close": sox}).sort_index()
    prices.index.name = "date"
    prices.to_csv(RESULTS / "kr_retail_margin_rule_prices.csv", encoding="utf-8-sig")

    panel = build_panel(margin, kospi, sox)
    panel.to_csv(RESULTS / "kr_retail_margin_rule_daily.csv", index=False, encoding="utf-8-sig")

    eps_all = []
    for col in ["on_primary", "on_ref_pct95", "on_primary_kospi", "on_primary_kosdaq"]:
        eps_all.append(episodes(panel, col))
    eps = pd.concat(eps_all, ignore_index=True)
    eps.to_csv(RESULTS / "kr_retail_margin_rule_episodes.csv", index=False, encoding="utf-8-sig")
    eps_primary = eps[eps["rule"] == "on_primary"]
    eps_ref = eps[eps["rule"] == "on_ref_pct95"]

    ev = panel[(panel["signal_day"] >= EVAL_START) & (panel["signal_day"] <= EVAL_END)]
    evd = ev[ev["crash_defined"]]
    base = float(evd["crash_any"].mean())
    on = evd[evd["on_primary"]]
    on_ref = evd[evd["on_ref_pct95"]]

    # 급락 시작일 목록(설명용, 사후값)
    crash_starts = []
    prev = False
    for _, r in ev.iterrows():
        cur = bool(r["crash_any"])
        if cur and not prev:
            crash_starts.append({"signal_day": r["signal_day"].date(),
                                 "kospi_fwd5_min_pct": round(float(r["kospi_fwd5_min"]) * 100, 2),
                                 "sox_fwd5_min_pct": round(float(r["sox_fwd5_min"]) * 100, 2) if pd.notna(r["sox_fwd5_min"]) else None})
        prev = cur

    leads = {k: lead_days(panel, eps_primary, v) for k, v in EVENTS.items()}
    leads_ref = {k: lead_days(panel.assign(on_primary=panel["on_ref_pct95"]), eps_ref, v) for k, v in EVENTS.items()}

    # 핵심 날짜 값(데이터 날짜 D 기준)
    m = margin.set_index("date").sort_index()
    for c in SERIES:
        m[f"z60_{c}"] = zscore(m[c])
        m[f"pct250_{c}"] = rolling_pct(m[c])
    rows = []
    for d in KEYDATES:
        ts = pd.Timestamp(d)
        if ts in m.index:
            r = m.loc[ts]
            rows.append({"data_date": d, "status": "받음",
                         **{c: round(float(r[c]), 1) for c in SERIES},
                         "z60_total": round(float(r["z60_credit_loan_total"]), 2),
                         "pct250_total": round(float(r["pct250_credit_loan_total"]), 1),
                         "pct250_kospi": round(float(r["pct250_credit_loan_kospi"]), 1),
                         "pct250_kosdaq": round(float(r["pct250_credit_loan_kosdaq"]), 1)})
        else:
            rows.append({"data_date": d, "status": "못 받음(freesis 미공표, 10/14 14:04 조회)"})
    kv = pd.DataFrame(rows)
    kv.to_csv(RESULTS / "kr_retail_margin_rule_keyvalues.csv", index=False, encoding="utf-8-sig")

    summary = {
        "rule": f"z60(credit_loan_total) >= {RULE_Z} (60거래일 평균·표준편차, 오늘 포함), 신호일 = 데이터일 다음 거래일",
        "crash_def": f"신호일 종가 대비 이후 {CRASH_H}거래일 최저 종가 <= {CRASH_PCT*100:.0f}% (KOSPI) 또는 SOX 동일(기준=직전 세션)",
        "eval_window": [EVAL_START, EVAL_END],
        "eval_days": int(len(evd)),
        "on_days_primary": int(len(on)),
        "p_crash_given_on_primary": round(float(on["crash_any"].mean()) * 100, 1) if len(on) else None,
        "p_crash_base": round(base * 100, 1),
        "on_days_ref_pct95": int(len(on_ref)),
        "p_crash_given_on_ref": round(float(on_ref["crash_any"].mean()) * 100, 1) if len(on_ref) else None,
        "episodes_primary": int(len(eps_primary)),
        "false_alarms_primary": int((~eps_primary["hit"]).sum()) if len(eps_primary) else 0,
        "hits_primary": int(eps_primary["hit"].sum()) if len(eps_primary) else 0,
        "episodes_ref": int(len(eps_ref)),
        "false_alarms_ref": int((~eps_ref["hit"]).sum()) if len(eps_ref) else 0,
        "leads_primary": {k: {kk: (str(vv) if vv is not None else None) for kk, vv in v.items()} for k, v in leads.items()},
        "leads_ref_pct95": {k: {kk: (str(vv) if vv is not None else None) for kk, vv in v.items()} for k, v in leads_ref.items()},
        "crash_starts": [{k: str(v) for k, v in c.items()} for c in crash_starts],
        "margin_rows": int(len(margin)),
        "margin_range": [str(margin["date"].min().date()), str(margin["date"].max().date())],
        "price_range": [str(kospi.index.min().date()), str(kospi.index.max().date())],
    }
    (RESULTS / "kr_retail_margin_rule_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print("\n[episodes]")
    print(eps.to_string())
    print("\n[key values]")
    print(kv.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
