"""SEIBRO 결제 날짜 → 담긴 미국 거래일 날짜표(가격 미사용).

규약: 미국 거래일 t는 결제일 s(t)(T+1, 2024-05-28 전 거래는 T+2)에 결제되고, SEIBRO에는 s(t) + 1일(토·일이면 월)에 찍힌다.
결제 달력 = 평일 − NYSE 휴장 − 연준 휴일(컬럼버스 데이, 재향군인의 날).
일별 Top50 합으로 빈 날·겹친 날을 실측해 예측과 대조하고, 어긋난 날을 표시한다.
출력: derived/seibro_calendar.csv, derived/calendar_check.json
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "derived"
T1 = dt.date(2024, 5, 28)


def fed_holidays(years) -> set[dt.date]:
    out = set()
    for y in years:
        d = dt.date(y, 10, 1)
        mondays = [d + dt.timedelta(days=i) for i in range(31) if (d + dt.timedelta(days=i)).weekday() == 0]
        out.add(mondays[1])  # 10월 둘째 월요일
        v = dt.date(y, 11, 11)
        if v.weekday() == 6:
            v += dt.timedelta(days=1)
        if v.weekday() < 5:
            out.add(v)
    return out


def calendars():
    import exchange_calendars as xc

    nyse = [s.date() for s in xc.get_calendar("XNYS", start="2022-12-01", end="2027-03-01").sessions]
    fed = fed_holidays(range(2022, 2028))
    # 2025-01-09(카터 추모일)은 NYSE만 닫고 결제는 열렸다(SEIBRO 2025-01-10이 정상, 01-13이 빔으로 확인)
    nyse_set = set(nyse) | {dt.date(2025, 1, 9)}
    settle = []
    d = dt.date(2022, 12, 1)
    while d <= dt.date(2027, 2, 28):
        if d.weekday() < 5 and d in nyse_set and d not in fed:
            settle.append(d)
        d += dt.timedelta(days=1)
    return nyse, settle


def seibro_of(t: dt.date, settle: list[dt.date]) -> dt.date:
    import bisect

    i = bisect.bisect_right(settle, t)  # t 다음 결제일
    s = settle[i + (0 if t >= T1 else 1)]
    d = s + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def main() -> None:
    nyse, settle = calendars()
    tot = pd.read_parquet(DER / "totals_daily.parquet")
    tot["date"] = pd.to_datetime(tot["start"]).dt.date
    tot = tot.set_index("date").sort_index()
    total = tot["sum_tot_d3"].fillna(0.0)
    first, last = total.index.min(), total.index.max()
    trades = [t for t in nyse if dt.date(2022, 12, 26) <= t <= last]
    m = {}
    for t in trades:
        m.setdefault(seibro_of(t, settle), []).append(t)
    rows = []
    for d in total.index:
        ts = [t for t in m.get(d, [])]
        rows.append({"seibro_date": d, "trades": ";".join(x.isoformat() for x in ts), "n_pred": len(ts),
                     "total": float(total[d])})
    cal = pd.DataFrame(rows)
    # 실측: 한 거래일 몫 날짜들의 20날짜 이동 중앙값 대비 비율
    single = cal["total"].where(cal["n_pred"] == 1)
    med = single.rolling(41, center=True, min_periods=10).median()
    cal["ratio"] = cal["total"] / med
    cal["obs"] = pd.cut(cal["ratio"], [-1, 0.3, 1.6, 99], labels=["empty", "normal", "double"]).astype(str)
    cal["pred"] = cal["n_pred"].map(lambda n: "empty" if n == 0 else "double" if n >= 2 else "normal")
    cal["match"] = cal["obs"] == cal["pred"]
    mism = cal[~cal["match"]]
    check = {
        "dates": int(len(cal)), "match_rate": round(float(cal["match"].mean()), 4),
        "mismatch": mism[["seibro_date", "trades", "n_pred", "ratio", "obs", "pred"]].astype(str).to_dict("records"),
        "trades_unmapped": [t.isoformat() for t in trades if first <= seibro_of(t, settle) <= last
                            and seibro_of(t, settle) not in set(cal["seibro_date"])],
    }
    cal.to_csv(DER / "seibro_calendar.csv", index=False)
    (DER / "calendar_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=1), encoding="utf-8")
    print("dates", check["dates"], "match", check["match_rate"], "mismatch", len(mism))
    print(mism[["seibro_date", "trades", "n_pred", "ratio", "obs", "pred"]].to_string())
    print("sha256", hashlib.sha256((DER / "seibro_calendar.csv").read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
