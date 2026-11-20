"""2026 실적 사건표: 반응일, 결과 수익률(QQQ 대비), 발표 당시 두 펀드 보유 여부.

반응일 = 장 마감 뒤 발표면 다음 거래일, 장 전·장중 발표면 그날.
발표 시각: SEC 8-K 2.02 접수 시각(±3일, 2026-10-28까지 캐시) → yfinance 시각 → 둘 다 없으면 거래량 큰 날.

결과(모두 QQQ 같은 구간 대비 초과, %):
  r0_x   전날 종가 → 반응일 종가 (반응일 하루)
  d20_x  반응일 다음 날 시가 → 반응일+20 종가 (주 결과: 실적을 본 뒤 살 수 있는 첫 자리부터)
  d40_x  같은 출발 → 반응일+40 종가
  t20_x  전날 종가 → 반응일+20 종가 (반응일 포함 전체)
  nx_x   다음 날 시가 → 다음 실적 반응일 전날 종가(최대 +63)
결과: results/events.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
CACHE = BASE / "cache"
SEC = ROOT / "data/derived/analysis/earnings_trend/cache/wide/sec_202.csv"
FIRST, LAST = "2026-01-02", "2026-11-19"


def load_px():
    px = {k: pd.read_parquet(CACHE / f"px_{k}.parquet") for k in ("open", "close", "adjclose", "volume", "high", "low")}
    idx = px["close"]["QQQ"].dropna().index
    px = {k: v.reindex(idx) for k, v in px.items()}
    f = px["adjclose"] / px["close"]  # 배당·분할 수정 계수를 시가에도 적용
    px["aopen"] = px["open"] * f
    return px, idx


def timing(row, sec_by_tk):
    """(발표 날짜, 'amc'|'bmo'|'unk', 출처)"""
    ts = pd.Timestamp(row.ts)
    d = ts.tz_convert("America/New_York") if ts.tzinfo else ts
    day = pd.Timestamp(d.date())
    s = sec_by_tk.get(row.ticker)
    if s is not None:
        near = s[(s.acc.dt.normalize() - day).abs() <= pd.Timedelta(days=3)]
        if len(near):
            a = near.iloc[(near.acc.dt.normalize() - day).abs().argmin()].acc
            hm = a.hour * 60 + a.minute
            return pd.Timestamp(a.date()), "amc" if hm >= 16 * 60 else ("bmo" if hm < 9 * 60 + 30 else "mid"), "sec"
    # yfinance가 시각 없이 날짜만 줄 때는 UTC 00:00으로 온다(뉴욕 시각으로는 전날 19~20시로 보인다).
    # 그 날짜를 발표일로 쓰고, 해외 상장사는 공시 관행으로 시각을 정한다(Astra가 ARM 공시로 확인, 2026-11-20).
    if ts.tzinfo and ts.tz_convert("UTC").hour == 0 and ts.tz_convert("UTC").minute == 0:
        day = pd.Timestamp(ts.tz_convert("UTC").date())
        conv = {"ARM": "amc", "SAP": "amc", "ASML": "bmo", "TSM": "bmo", "SKHY": "bmo"}
        return day, conv.get(row.ticker, "unk"), "yf_dateonly"
    hm = d.hour * 60 + d.minute
    if hm == 0:
        return day, "unk", "yf"
    return day, "amc" if hm >= 16 * 60 else ("bmo" if hm < 9 * 60 + 30 else "mid"), "yf"


def main() -> None:
    px, idx = load_px()
    ed = pd.read_csv(CACHE / "earn_dates.csv")
    ed = ed[ed.eps_rep.notna() | (pd.to_datetime(ed.ts, utc=True) < pd.Timestamp("2026-11-20", tz="UTC"))]
    sec = pd.read_csv(SEC)
    sec["acc"] = pd.to_datetime(sec.acceptance_et)
    sec_by_tk = {k: g for k, g in sec.groupby("ticker")}
    held = pd.read_csv(BASE / "results/held_days.csv")
    held_dates = sorted(held.date.unique())
    held_set = {(r.date, r.fund, r.ticker): (r.qty, r.weight) for r in held.itertuples()}

    rows = []
    for r in ed.itertuples():
        if r.ticker not in px["close"].columns:
            continue
        day, tm, src = timing(r, sec_by_tk)
        if day < pd.Timestamp("2025-12-20") or day > pd.Timestamp(LAST):
            continue
        pos = idx.searchsorted(day)  # 발표일 이후 첫 거래일(발표일이 거래일이면 그날)
        if pos >= len(idx):
            continue
        is_session = pos < len(idx) and idx[pos] == day
        if tm == "amc":
            k = pos + 1 if is_session else pos
        elif tm in ("bmo", "mid"):
            k = pos
        else:  # 시각 모름: 발표일과 다음 거래일 중 거래량 큰 날
            v = px["volume"][r.ticker]
            k = pos if (pos + 1 >= len(idx) or v.iloc[pos] >= v.iloc[pos + 1]) else pos + 1
        if k >= len(idx) or k < 70:
            continue
        rd = idx[k]
        if rd < pd.Timestamp(FIRST) or rd > pd.Timestamp(LAST):
            continue
        rows.append({"ticker": r.ticker, "ann_date": day.date(), "timing": tm, "time_src": src, "k": k,
                     "reaction_date": rd.date(), "eps_est": r.eps_est, "eps_rep": r.eps_rep,
                     "eps_surprise": r.surprise_pct})
    ev = pd.DataFrame(rows).drop_duplicates(["ticker", "reaction_date"]).sort_values(["ticker", "k"]).reset_index(drop=True)
    # 같은 종목 30거래일 안 중복(yfinance 날짜 이중 기재)은 앞의 것만
    ev = ev[~(ev.groupby("ticker").k.diff() < 30)].reset_index(drop=True)
    # 다음 실적(2026-11-20 이후 예정 포함)의 반응일 위치
    fut = pd.read_csv(CACHE / "earn_dates.csv")
    fut_day = pd.to_datetime(fut.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()

    ac, ao, n = px["adjclose"], px["aopen"], len(idx)
    q_c, q_o = ac["QQQ"], ao["QQQ"]

    def ret(tk, a_series, i0, c_i1):
        if c_i1 >= n or i0 < 0:
            return np.nan
        return a_series[tk].iloc[i0], ac[tk].iloc[c_i1]

    out = []
    for e in ev.itertuples():
        k, tk = e.k, e.ticker
        c, o = ac[tk], ao[tk]
        rec = e._asdict()
        rec.pop("Index")

        def x(p0, p1, q0, q1):
            if np.isnan(p0) or np.isnan(p1) or np.isnan(q0) or np.isnan(q1):
                return np.nan, np.nan
            r_, q_ = p1 / p0 - 1, q1 / q0 - 1
            return 100 * r_, 100 * (r_ - q_)

        rec["r0"], rec["r0_x"] = x(c.iloc[k - 1], c.iloc[k], q_c.iloc[k - 1], q_c.iloc[k])
        rec["gap"] = 100 * (px["open"][tk].iloc[k] / px["close"][tk].iloc[k - 1] - 1)
        for h in (5, 20, 40):
            if k + h < n:
                rec[f"d{h}"], rec[f"d{h}_x"] = x(o.iloc[k + 1], c.iloc[k + h], q_o.iloc[k + 1], q_c.iloc[k + h])
                rec[f"t{h}"], rec[f"t{h}_x"] = x(c.iloc[k - 1], c.iloc[k + h], q_c.iloc[k - 1], q_c.iloc[k + h])
            else:
                for nm in (f"d{h}", f"d{h}_x", f"t{h}", f"t{h}_x"):
                    rec[nm] = np.nan
        # 다음 실적 반응일 전날까지(최대 63거래일)
        later = fut_day[(fut.ticker == tk) & (fut_day > pd.Timestamp(e.ann_date) + pd.Timedelta(days=40))]
        if len(later):
            nd = later.min()
            npos = idx.searchsorted(nd)
            end = min(npos - 1, k + 63)  # 다음 발표일 전날 종가(장 후 발표여도 보수적으로 발표일 전날)
        else:
            end = k + 63
        rec["nx_end_k"] = end
        rec["nx_complete"] = end < n
        if end < n and end > k + 1:
            rec["nx"], rec["nx_x"] = x(o.iloc[k + 1], c.iloc[end], q_o.iloc[k + 1], q_c.iloc[end])
        else:
            rec["nx"], rec["nx_x"] = np.nan, np.nan
        rec["last_k"] = n - 1
        # 발표 전날 기준 보유(그날 이전 마지막 스냅샷)
        pre = str(idx[k - 1].date())
        snap = [d for d in held_dates if d <= pre]
        snap = snap[-1] if snap else held_dates[0]
        rec["snap"] = snap
        for f in ("TIME", "KoAct"):
            qv = held_set.get((snap, f, tk))
            rec[f"held_{f}"] = qv is not None
            rec[f"w_{f}"] = qv[1] if qv else 0.0
        rec["held_any"] = rec["held_TIME"] or rec["held_KoAct"]
        rd = pd.Timestamp(e.reaction_date)
        rec["season"] = f"{rd.year}Q{(rd.month - 1) // 3 + 1}"
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_csv(BASE / "results/events.csv", index=False)
    print(len(df), "events;", df.held_any.sum(), "held at time")
    print(df.groupby("season").agg(n=("ticker", "size"), held=("held_any", "sum"), d20=("d20_x", "count")))
    print(df.time_src.value_counts().to_dict(), df.timing.value_counts().to_dict())


if __name__ == "__main__":
    main()
