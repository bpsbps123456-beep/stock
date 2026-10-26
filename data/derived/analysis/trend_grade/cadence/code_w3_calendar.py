"""W3 달력·이벤트 점검: 2026년 6~7월 분기 경계 전후 S등급 종목 가격 흐름.

웹 조사(W3)에서 찾은 달력 이벤트(분기말 연기금 리밸런싱, 나스닥100 분기 심사
6/22 시행, MU 실적 6/24, 분기 초 7/1)와 저장소 가격 캐시를 맞춰 본다.
인과 검증이 아니라 사례 1건의 날짜 정렬이다.

입력: data/derived/analysis/rsi_breakout/cache/1d_long.parquet (일봉 종가)
출력: results_w3_calendar.json 의 "local_check" 항목에 넣을 숫자를 표준 출력
실행: py data/derived/analysis/trend_grade/cadence/code_w3_calendar.py
"""
from __future__ import annotations

import datetime as dt
import json
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 5))
PX = os.path.join(ROOT, "data", "derived", "analysis", "rsi_breakout", "cache", "1d_long.parquet")

# report.md 7월 급락 표의 S 7종목(6/22 종가 등급)과 비교 기준
S_NAMES = ["ALAB", "ARM", "MRVL", "MU", "SNDK", "STX", "WDC"]
BENCH = ["QQQ", "NVDA"]
CHECK_DATES = ["2026-06-18", "2026-06-19", "2026-06-22", "2026-06-30", "2026-07-01"]


def load_close() -> pd.DataFrame:
    df = pd.read_parquet(PX, columns=["ts", "Close", "sym"])
    ts = pd.to_datetime(df["ts"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    df["date"] = ts.dt.normalize()
    return df.pivot_table(index="date", columns="sym", values="Close")


def ret(p: pd.DataFrame, sym: str, a: str, b: str) -> float | None:
    if sym not in p.columns:
        return None
    s = p[sym].dropna()
    try:
        return round(float(s.loc[b] / s.loc[a] - 1) * 100, 1)
    except KeyError:
        return None


def main() -> None:
    p = load_close()
    days = set(p.index.date)
    out = {
        "trading_day_present": {d: dt.date.fromisoformat(d) in days for d in CHECK_DATES},
        "windows_pct": {},
    }
    windows = {
        "6/22->6/30 (분기말 주간)": ("2026-06-22", "2026-06-30"),
        "6/30->7/01 (분기 첫날)": ("2026-06-30", "2026-07-01"),
        "6/30->7/29 (분기 초~급락 저점)": ("2026-06-30", "2026-07-29"),
        "6/24->6/25 (MU 실적 다음 날)": ("2026-06-24", "2026-06-25"),
    }
    for name, (a, b) in windows.items():
        row = {s: ret(p, s, a, b) for s in S_NAMES + BENCH}
        vals = [row[s] for s in S_NAMES if row[s] is not None]
        row["S7_equal_avg"] = round(sum(vals) / len(vals), 1) if vals else None
        out["windows_pct"][name] = row
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
