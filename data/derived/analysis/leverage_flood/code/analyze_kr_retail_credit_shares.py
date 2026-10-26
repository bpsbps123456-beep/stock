"""신용거래 잔고 주수(STATSCU0100000080BO) 를 급락 경고 신호등으로 쓸 수 있는지 전방검증.

입력: ../results/kr_retail_credit_shares_long.csv (fetch_kr_retail_credit_shares.py, 단위 주, 2024-06~)
      ../results/kr_retail_margin_balance.csv     (신용융자 금액 억원, 설명용 평균 신용단가 계산)
      yfinance ^KS11(코스피), ^SOX(필라델피아 반도체) 종가
출력: ../results/kr_retail_credit_shares_signal.csv    알 수 있는 날 기준 일별 값·백분위·신호·사후 5일 최저
      ../results/kr_retail_credit_shares_episodes.csv  신호 ON 에피소드와 적중/오경보
      ../results/kr_retail_credit_shares_summary.json  핵심 날짜 값·시차·오경보 수

발표 시차: freesis 는 D 일 잔고를 D+1 오후 이후에 올린다(10/14 14:04 조회 시 10/13 행 없음, 10/12 행은 10/13 보도).
  → D 일 값은 D+1 거래일(K)에 "알 수 있는 날"로 밀어서 평가한다. 장중 매매에 쓰려면 사실상 D+2 시가다.

사전에 정한 규칙(세 사건을 보고 고르지 않음):
  RED = 신용거래융자 잔고 주수 합계의 직전 250거래일 백분위 >= 95 (그날 제외 과거 250개 중 오늘 이하 비율).
  레버리지 홍수 가설 그대로 "빌려 산 주식 수가 1년 최상단" 을 켜짐으로 본다.

급락 정의(사후 판정용): 신호 K 일 종가 대비 뒤 5거래일 중 최저 종가가 코스피 또는 SOX 에서 -8% 이하.
에피소드: 연속 ON(5거래일 이내 틈은 하나로 합침). 에피소드 안 어느 날이든 급락이 따라오면 적중, 아니면 오경보.
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
WIN = 250
THRESH = 95.0
DROP = -0.08
EVAL_START, EVAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2026-10-13")
EVENTS = [  # (라벨, 급락 기준일, 직전 거래일 = 마지막 사전값의 D)
    ("6/23 삼성·하이닉스 -12% 서킷", "2026-06-23", "2026-06-22"),
    ("7/29 코스피 -10.84%", "2026-07-29", "2026-07-28"),
    ("10/6~13 5일 -13.8%", "2026-10-06", "2026-10-02"),
]
KEYDATES = ["2026-06-22", "2026-07-28", "2026-10-02", "2026-10-13"]


def pct_rank_past(s: pd.Series, win: int = WIN) -> pd.Series:
    def _p(x):
        past = x[:-1]
        past = past[~np.isnan(past)]
        return 100.0 * np.sum(past <= x[-1]) / len(past) if len(past) else np.nan
    return s.rolling(win + 1, min_periods=win + 1).apply(_p, raw=True)


def load_prices() -> pd.DataFrame:
    ks = yf.download("^KS11", start="2024-05-01", end="2026-10-16", progress=False, auto_adjust=False)["Close"]
    sox = yf.download("^SOX", start="2024-05-01", end="2026-10-16", progress=False, auto_adjust=False)["Close"]
    ks = ks.iloc[:, 0] if isinstance(ks, pd.DataFrame) else ks
    sox = sox.iloc[:, 0] if isinstance(sox, pd.DataFrame) else sox
    px = pd.DataFrame({"kospi": ks})
    px.index = pd.to_datetime(px.index).tz_localize(None)
    sox.index = pd.to_datetime(sox.index).tz_localize(None)
    px["sox"] = sox.reindex(px.index).ffill()  # 미국 K 일 세션 종가를 한국 K 일 칸에 둔다(한국 K+1 아침에 앎)
    px = px.dropna(subset=["kospi"])
    for c in ["kospi", "sox"]:
        fwd = pd.concat([px[c].shift(-h) / px[c] - 1 for h in range(1, 6)], axis=1)
        px[f"{c}_fwd5_min"] = fwd.min(axis=1)
        px[f"{c}_chg5"] = px[c] / px[c].shift(5) - 1
    return px


def main() -> int:
    d = pd.read_csv(RESULTS / "kr_retail_credit_shares_long.csv", parse_dates=["date"]).sort_values("date")
    d = d.reset_index(drop=True)
    d["pct250_total"] = pct_rank_past(d["credit_loan_shares_total"])
    d["pct250_kospi"] = pct_rank_past(d["credit_loan_shares_kospi"])
    d["pct250_kosdaq"] = pct_rank_past(d["credit_loan_shares_kosdaq"])
    d["chg20_total_pct"] = d["credit_loan_shares_total"].pct_change(20) * 100
    d["rule_on"] = d["pct250_total"] >= THRESH

    px = load_prices()
    cal = px.index  # 코스피 거래일 달력
    # 알 수 있는 날 K = D 뒤 첫 코스피 거래일 (T+1)
    pos = np.searchsorted(cal.values, d["date"].values, side="right")
    d["known_date"] = [cal[p] if p < len(cal) else pd.NaT for p in pos]
    d = d.dropna(subset=["known_date"])
    d = d.drop_duplicates("known_date", keep="last")  # 12/31 같은 휴장일 행은 다음 거래일 값에 덮인다

    sig = d.set_index("known_date")[["date", "credit_loan_shares_total", "credit_loan_shares_kospi",
                                     "credit_loan_shares_kosdaq", "pct250_total", "pct250_kospi",
                                     "pct250_kosdaq", "chg20_total_pct", "rule_on"]]
    sig = sig.rename(columns={"date": "data_date"})
    sig = sig.join(px[["kospi", "sox", "kospi_fwd5_min", "sox_fwd5_min", "kospi_chg5", "sox_chg5"]])
    sig["drop_follows"] = (sig["kospi_fwd5_min"] <= DROP) | (sig["sox_fwd5_min"] <= DROP)
    sig.index.name = "known_date"
    sig.to_csv(RESULTS / "kr_retail_credit_shares_signal.csv", encoding="utf-8-sig")

    # 에피소드(평가 구간: 알 수 있는 날 기준 2025-10-01~2026-10-13, 백분위 유효한 날만)
    ev = sig[(sig.index >= EVAL_START) & (sig.index <= EVAL_END) & sig["pct250_total"].notna()].copy()
    idx = {ts: i for i, ts in enumerate(cal)}
    on_days = [ts for ts, v in ev["rule_on"].items() if v]
    episodes = []
    for ts in on_days:
        if episodes and idx[ts] - idx[episodes[-1]["end"]] <= 5:
            episodes[-1]["end"] = ts
        else:
            episodes.append({"start": ts, "end": ts})
    rows = []
    for e in episodes:
        seg = ev.loc[e["start"]:e["end"]]
        hit = bool(seg["drop_follows"].any())
        rows.append({
            "start_known": e["start"].date(), "end_known": e["end"].date(),
            "on_days": int(seg["rule_on"].sum()),
            "trading_days_span": idx[e["end"]] - idx[e["start"]] + 1,
            "max_pct250": round(float(seg["pct250_total"].max()), 1),
            "peak_shares": int(seg["credit_loan_shares_total"].max()),
            "min_kospi_fwd5_pct": round(float(seg["kospi_fwd5_min"].min() * 100), 2),
            "min_sox_fwd5_pct": round(float(seg["sox_fwd5_min"].min() * 100), 2),
            "hit": hit,
        })
    ep = pd.DataFrame(rows)
    ep.to_csv(RESULTS / "kr_retail_credit_shares_episodes.csv", index=False, encoding="utf-8-sig")
    false_alarms = int((~ep["hit"]).sum()) if len(ep) else 0

    # 세 사건: 사건일 앞 20거래일 안에 ON 이 있었나, 있었으면 에피소드 시작(알 수 있는 날)부터 사건일까지 거래일 수
    leads = {}
    for label, ev_day, pre in EVENTS:
        E = pd.Timestamp(ev_day)
        ei = idx[E]
        win = sig[(sig.index < E) & (sig.index >= cal[max(0, ei - 20)])]
        on_win = win[win["rule_on"].fillna(False)]
        lead = None
        if len(on_win):
            first = on_win.index[0]
            # 그 ON 이 속한 에피소드의 시작까지 거슬러 간다
            for e in episodes:
                if e["start"] <= first <= e["end"]:
                    first = e["start"]
                    break
            lead = ei - idx[first]
        pre_known = sig[sig["data_date"] == pd.Timestamp(pre)]
        leads[label] = {
            "event_day": ev_day,
            "last_pre_data_date": pre,
            "known_on": pre_known.index[0].strftime("%Y-%m-%d") if len(pre_known) else None,
            "pct250_at_pre": round(float(pre_known["pct250_total"].iloc[0]), 1) if len(pre_known) else None,
            "rule_on_in_prev_20d": bool(len(on_win)),
            "lead_trading_days": lead,
        }

    # 핵심 날짜 값(자료일 기준)과 설명용 평균 신용단가(금액/주수, 사후 아님 — 같은 날 둘 다 공표)
    dd = d.set_index("date")
    try:
        mb = pd.read_csv(RESULTS / "kr_retail_margin_balance.csv", parse_dates=["date"]).set_index("date")
    except FileNotFoundError:
        mb = None
    key = {}
    for k in KEYDATES:
        ts = pd.Timestamp(k)
        if ts in dd.index:
            r = dd.loc[ts]
            item = {
                "total_shares": int(r["credit_loan_shares_total"]),
                "kospi_shares": int(r["credit_loan_shares_kospi"]),
                "kosdaq_shares": int(r["credit_loan_shares_kosdaq"]),
                "pct250_total": round(float(r["pct250_total"]), 1),
                "pct250_kospi": round(float(r["pct250_kospi"]), 1),
                "pct250_kosdaq": round(float(r["pct250_kosdaq"]), 1),
                "chg20_total_pct": round(float(r["chg20_total_pct"]), 1),
            }
            if mb is not None and ts in mb.index:
                amt = float(mb.loc[ts, "credit_loan_total"])  # 억원
                item["credit_amount_tril_krw"] = round(amt / 10000, 2)
                item["avg_krw_per_credit_share"] = int(round(amt * 1e8 / r["credit_loan_shares_total"]))
            key[k] = item
        else:
            key[k] = {"status": "못 받음 — freesis 에 해당 일자 행 없음(10/14 14:04 조회 시 10/13 미공표)"}

    summary = {
        "indicator": "신용거래융자 잔고 주수 합계 (freesis STATSCU0100000080BO TMPV2)",
        "rows": int(len(d)),
        "date_range": f"{d['date'].min().date()}~{d['date'].max().date()}",
        "rule": f"직전 {WIN}거래일 백분위 >= {THRESH:.0f} (알 수 있는 날 K=D+1 기준)",
        "eval_window_known": f"{EVAL_START.date()}~{EVAL_END.date()}",
        "eval_days": int(len(ev)),
        "on_days_in_eval": int(ev["rule_on"].sum()),
        "episodes": rows,
        "false_alarms": false_alarms,
        "hits": int(ep["hit"].sum()) if len(ep) else 0,
        "events": leads,
        "key_values": key,
        "series_extremes_eval": {
            "max_total_shares": int(ev["credit_loan_shares_total"].max()),
            "max_date": ev["data_date"][ev["credit_loan_shares_total"].idxmax()].strftime("%Y-%m-%d"),
            "min_total_shares": int(ev["credit_loan_shares_total"].min()),
            "min_date": ev["data_date"][ev["credit_loan_shares_total"].idxmin()].strftime("%Y-%m-%d"),
        },
    }
    (RESULTS / "kr_retail_credit_shares_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
