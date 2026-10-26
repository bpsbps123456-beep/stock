"""CFD 명목잔고(증거금포함·증거금차감) 경고 신호등 전방검증 — 홍수 지표 #6.

입력: ../results/kr_retail_cfd_gross_long.csv, ../results/kr_retail_cfd_net_of_margin_long.csv
      (fetch_kr_retail_cfd.py, 2024-06-03~2026-10-13, 억원)
가격: yfinance ^KS11(코스피), ^SOX(필라델피아 반도체) 2024-05-01~ — ../results/kr_retail_cfd_prices.csv 로 저장.
      ^KS11 은 금투협 kr_retail_kospi.csv 의 kospi_index 와 대조해 최대 오차를 출력한다.

[사전 고정 규칙 — 결과를 보기 전에 적음]
  R: 증거금포함 CFD 명목잔고 합계(cfd_total) 의 직전 250거래일(그날 제외, 250개가 꽉 찬 날만) 백분위 >= 95 → 빨강.
     백분위 = 100 × (과거 250개 중 오늘 값 이하인 개수) / 250.
  시차: 잔고일 T 의 값은 T+1 낮에 볼 수 있다(2026-10-14 직접 확인) → 신호일(알 수 있는 날) = T 다음 코스피 거래일.
        12/31 처럼 휴장일 잔고가 있으면 다음 거래일로 밀고, 같은 날로 밀린 값이 여럿이면 가장 늦은 잔고일을 쓴다.
  에피소드: 신호일 기준 연속 빨강. 꺼진 날이 5거래일 미만이면 같은 에피소드로 묶는다.
  급락 시작일: 코스피(거래일) 또는 SOX(미국 거래일) 종가가 그날부터 5거래일 안에 -8% 이하가 되는 첫 날(달력일로 비교).
  적중: 에피소드 [시작일, 끝일+5 코스피 거래일] 안에 급락 시작일이 하나라도 있으면 적중, 없으면 오경보.
  사건 선행일수: 사건일 E(6/23, 7/29, 10/6)에 대해 E-1 신호일에 빨강이면 그 에피소드 시작일부터 E 까지 거래일 수.
        E-1 에 꺼져 있으면 null 로 두고, 직전 20거래일 안의 마지막 빨강일을 따로 적는다.
  평가 구간: 신호일 2025-10-01 ~ 2026-10-13. 같은 규칙을 증거금차감 합계에도 적용하지만 판정은 증거금포함 기준이다.

출력(../results/):
  kr_retail_cfd_prices.csv        yfinance 종가(코스피·SOX)
  kr_retail_cfd_signal_daily.csv  신호일별 값·백분위·빨강·앞 5일 최저수익률
  kr_retail_cfd_episodes.csv      에피소드별 시작·끝·길이·적중 여부·급락일
  kr_retail_cfd_event_lead.csv    사건별 선행일수
  kr_retail_cfd_keydates.csv      6/22·7/28·10/2·10/13 값·백분위·구성비
  kr_retail_cfd_signal_summary.json 요약
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

PCT_THRESHOLD = 95.0
WINDOW = 250
MERGE_GAP = 5          # 꺼진 날이 이보다 짧으면 같은 에피소드
HIT_TAIL = 5           # 에피소드 끝 뒤 몇 거래일까지 급락을 인정하나
CRASH_PCT = -0.08
EVAL_START = pd.Timestamp("2025-10-01")
EVAL_END = pd.Timestamp("2026-10-13")
EVENTS = [("6/23", "2026-06-23"), ("7/29", "2026-07-29"), ("10/6", "2026-10-06")]
KEYDATES = ["2026-06-22", "2026-07-28", "2026-10-02", "2026-10-13"]


def load_prices() -> pd.DataFrame:
    path = RESULTS / "kr_retail_cfd_prices.csv"
    import yfinance as yf
    px = yf.download(["^KS11", "^SOX"], start="2024-05-01", end="2026-10-15", auto_adjust=False, progress=False)
    close = px["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.rename(columns={"^KS11": "kospi", "^SOX": "sox"})
    close.index.name = "date"
    close.to_csv(path, encoding="utf-8-sig")
    return close


def rolling_pct(s: pd.Series) -> pd.Series:
    def _p(x):
        v = x[-1]
        past = x[:-1]
        if np.isnan(v) or np.isnan(past).any():
            return np.nan
        return 100.0 * np.sum(past <= v) / len(past)
    return s.rolling(WINDOW + 1, min_periods=WINDOW + 1).apply(_p, raw=True)


def crash_starts(close: pd.Series) -> pd.Series:
    """그날부터 5거래일 안 최저 종가 수익률. -8% 이하면 급락 시작일."""
    c = close.dropna()
    fwd_min = pd.concat([c.shift(-k) / c - 1 for k in range(1, 6)], axis=1).min(axis=1)
    return fwd_min


def main() -> int:
    close = load_prices()
    kospi = close["kospi"].dropna()
    sox = close["sox"].dropna()
    kr_days = kospi.index

    # 금투협 코스피 지수와 대조
    kofia = pd.read_csv(RESULTS / "kr_retail_kospi.csv", parse_dates=["date"]).set_index("date")["kospi_index"].dropna()
    cmp = pd.concat([kospi.rename("yf"), kofia.rename("kofia")], axis=1).dropna()
    max_dev = float(((cmp["yf"] / cmp["kofia"] - 1).abs()).max() * 100)
    print(f"^KS11 vs KOFIA kospi_index: {len(cmp)}일 대조, 최대 오차 {max_dev:.3f}%")

    kospi_fwd = crash_starts(kospi)
    sox_fwd = crash_starts(sox)
    kospi_crash_days = kospi_fwd[kospi_fwd <= CRASH_PCT].index
    sox_crash_days = sox_fwd[sox_fwd <= CRASH_PCT].index
    all_crash = sorted(set(kospi_crash_days.date) | set(sox_crash_days.date))

    out_tables = {}
    summary = {"rule": f"증거금포함 CFD명목잔고 합계의 직전 {WINDOW}거래일 백분위 >= {PCT_THRESHOLD:g} (T+1 신호일), "
                       f"에피소드는 {MERGE_GAP}거래일 미만 끊김 병합, 적중=에피소드 시작~끝+{HIT_TAIL}거래일 안에 "
                       f"코스피 또는 SOX 5일 {CRASH_PCT*100:.0f}% 시작일",
               "publish_lag_days": 1, "eval_start": str(EVAL_START.date()), "eval_end": str(EVAL_END.date()),
               "kospi_crash_start_days": [str(d.date()) for d in kospi_crash_days],
               "sox_crash_start_days": [str(d.date()) for d in sox_crash_days],
               "ks11_vs_kofia_max_dev_pct": round(max_dev, 3), "series": {}}

    for tag, fname in [("gross", "kr_retail_cfd_gross_long.csv"), ("net_of_margin", "kr_retail_cfd_net_of_margin_long.csv")]:
        d = pd.read_csv(RESULTS / fname, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        d["pct250"] = rolling_pct(d["cfd_total"])
        d["chg20_pct"] = d["cfd_total"].pct_change(20) * 100
        d["foreign_share_pct"] = (d["cfd_buy_foreign_stock"] + d["cfd_buy_foreign_other"]) / d["cfd_total"] * 100
        # 신호일 = 잔고일 다음 코스피 거래일
        pos = kr_days.searchsorted(d["date"].values, side="right")
        d = d[pos < len(kr_days)].copy()
        d["signal_date"] = kr_days[pos[pos < len(kr_days)]]
        d = d.sort_values(["signal_date", "date"]).drop_duplicates("signal_date", keep="last")
        d["red"] = d["pct250"] >= PCT_THRESHOLD
        # 신호일 기준 앞으로의 급락(사후값, 평가용)
        d = d.set_index("signal_date")
        d["kospi_close"] = kospi.reindex(d.index)
        d["kospi_fwd5_min_pct"] = (kospi_fwd.reindex(d.index) * 100).round(2)
        # SOX: 신호일(한국 날짜) 이후 첫 미국 거래일부터의 5일 최저
        sox_pos = sox.index.searchsorted(d.index.values, side="left")
        sox_pos = np.minimum(sox_pos, len(sox) - 1)
        d["sox_fwd5_min_pct"] = (sox_fwd.values[sox_pos] * 100).round(2)
        daily = d.reset_index()[["signal_date", "date", "cfd_total", "pct250", "chg20_pct", "foreign_share_pct",
                                 "red", "kospi_close", "kospi_fwd5_min_pct", "sox_fwd5_min_pct"]]
        daily = daily.rename(columns={"date": "value_date"})
        daily.to_csv(RESULTS / f"kr_retail_cfd_signal_daily_{tag}.csv", index=False, encoding="utf-8-sig")

        # 에피소드(평가 구간 안의 신호일만)
        ev = daily[(daily["signal_date"] >= EVAL_START) & (daily["signal_date"] <= EVAL_END) & daily["pct250"].notna()]
        ev = ev.reset_index(drop=True)
        episodes = []
        cur = None
        on_idx = ev.index[ev["red"]].tolist()
        for i in on_idx:
            if cur is None:
                cur = [i, i]
            elif i - cur[1] < MERGE_GAP:
                cur[1] = i
            else:
                episodes.append(cur)
                cur = [i, i]
        if cur is not None:
            episodes.append(cur)
        ep_rows = []
        for s, e in episodes:
            start = ev.loc[s, "signal_date"]
            end = ev.loc[e, "signal_date"]
            end_pos = kr_days.get_loc(end)
            tail_end = kr_days[min(end_pos + HIT_TAIL, len(kr_days) - 1)]
            crashes = [c for c in all_crash if start.date() <= c <= tail_end.date()]
            n_on = int(ev.loc[s:e, "red"].sum())
            ep_rows.append({"start": start.date(), "end": end.date(), "window_end": tail_end.date(),
                            "on_days": n_on, "span_days": int(e - s + 1), "hit": bool(crashes),
                            "crash_days": ";".join(str(c) for c in crashes),
                            "max_pct250": round(float(ev.loc[s:e, "pct250"].max()), 1),
                            "last_data_ok": bool(tail_end < kr_days[-1])})
        ep_df = pd.DataFrame(ep_rows)
        ep_df.to_csv(RESULTS / f"kr_retail_cfd_episodes_{tag}.csv", index=False, encoding="utf-8-sig")

        # 사건 선행일수
        sig = daily.set_index("signal_date")["red"]
        lead_rows = []
        for label, es in EVENTS:
            E = pd.Timestamp(es)
            e_pos = kr_days.get_loc(E)
            pre = kr_days[e_pos - 1]
            on_pre = bool(sig.get(pre, False))
            lead = None
            ep_start = None
            if on_pre:
                for s, e in episodes:
                    if ev.loc[s, "signal_date"] <= pre <= ev.loc[e, "signal_date"]:
                        ep_start = ev.loc[s, "signal_date"]
                        break
                if ep_start is None:  # 평가 구간 밖 에피소드
                    ep_start = pre
                    k = kr_days.get_loc(pre)
                    while k - 1 >= 0 and bool(sig.get(kr_days[k - 1], False)):
                        k -= 1
                    ep_start = kr_days[k]
                lead = int(e_pos - kr_days.get_loc(ep_start))
            win = kr_days[max(0, e_pos - 20):e_pos]
            last_on = [dd for dd in win if bool(sig.get(dd, False))]
            lead_rows.append({"event": label, "event_date": es, "last_pre_signal_date": pre.date(),
                              "on_at_E-1": on_pre, "episode_start": None if ep_start is None else ep_start.date(),
                              "lead_days": lead, "pct250_at_E-1": round(float(daily.set_index("signal_date")["pct250"].get(pre, np.nan)), 1),
                              "last_red_in_20d": None if not last_on else last_on[-1].date()})
        lead_df = pd.DataFrame(lead_rows)
        lead_df.to_csv(RESULTS / f"kr_retail_cfd_event_lead_{tag}.csv", index=False, encoding="utf-8-sig")

        # 주요 날짜(잔고일 기준) 값
        kd = d.reset_index().set_index("date")
        key_rows = []
        for ks in KEYDATES:
            t = pd.Timestamp(ks)
            if t in kd.index:
                r = kd.loc[t]
                key_rows.append({"value_date": ks, "signal_date": r["signal_date"].date(), "cfd_total_eok": round(float(r["cfd_total"]), 1),
                                 "pct250": round(float(r["pct250"]), 1), "chg20_pct": round(float(r["chg20_pct"]), 1),
                                 "foreign_share_pct": round(float(r["foreign_share_pct"]), 1),
                                 "dom_kospi_eok": round(float(r["cfd_buy_dom_kospi"]), 1),
                                 "dom_kosdaq_eok": round(float(r["cfd_buy_dom_kosdaq"]), 1),
                                 "foreign_stock_eok": round(float(r["cfd_buy_foreign_stock"]), 1), "red": bool(r["red"])})
        key_df = pd.DataFrame(key_rows)
        key_df.to_csv(RESULTS / f"kr_retail_cfd_keydates_{tag}.csv", index=False, encoding="utf-8-sig")

        on_days = int(ev["red"].sum())
        base = float((ev["kospi_fwd5_min_pct"] <= CRASH_PCT * 100).mean() * 100)
        when_on = ev.loc[ev["red"], "kospi_fwd5_min_pct"]
        p_on = float((when_on <= CRASH_PCT * 100).mean() * 100) if len(when_on) else None
        summary["series"][tag] = {
            "rows": int(len(d)), "first_value_date": str(d["date"].iloc[0].date()), "last_value_date": str(d["date"].iloc[-1].date()),
            "eval_days": int(len(ev)), "on_days": on_days,
            "p_kospi_fwd5_le_-8_when_on": None if p_on is None else round(p_on, 1), "p_kospi_fwd5_le_-8_base": round(base, 1),
            "episodes": int(len(ep_df)), "episodes_hit": int(ep_df["hit"].sum()) if len(ep_df) else 0,
            "false_alarms": int((~ep_df["hit"]).sum()) if len(ep_df) else 0,
            "episode_list": ep_rows, "event_lead": lead_rows, "keydates": key_rows,
        }
        print(f"\n=== {tag} ===")
        print(f"평가 {len(ev)}일, 빨강 {on_days}일, 에피소드 {len(ep_df)}개(적중 {summary['series'][tag]['episodes_hit']}, 오경보 {summary['series'][tag]['false_alarms']})")
        print(ep_df.to_string() if len(ep_df) else "(에피소드 없음)")
        print(lead_df.to_string())
        print(key_df.to_string())
        out_tables[tag] = (ep_df, lead_df, key_df)

    def _default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    (RESULTS / "kr_retail_cfd_signal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")
    print("\n급락 시작일(코스피):", summary["kospi_crash_start_days"])
    print("급락 시작일(SOX):", summary["sox_crash_start_days"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
