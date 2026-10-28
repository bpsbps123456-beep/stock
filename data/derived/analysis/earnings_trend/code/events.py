"""실적 추세 연구 — 발표 이벤트 표(반응일·수익률·사전 변수).

실행: py data/derived/analysis/earnings_trend/code/events.py  → results/events.csv

시점 규칙(모든 사전 변수는 발표 직전 종가까지만 쓴다):
- 반응일: 8-K 접수가 미 동부 16:00 이전이면 그날, 이후면 다음 거래일. 기준가 = 반응일 전날 종가.
- 진입가 = 반응일 종가(반응을 보고 들어가는 경우). 보유 끝 = 다음 실적 반응일 전날 종가
  (다음 실적이 아직이면 자료 마지막 날까지 — 'partial' 표시).
- 초과수익은 QQQ 대비 단순 차이와, 발표 전 250거래일로 잰 베타로 뺀 비정상 수익 두 가지.
- 같은 실적 시즌(달력 분기 기준 발표 묶음) 안의 순위 비교가 매크로를 지우는 주 방법이다(analyze.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse  # noqa: E402

from fetch_sec import RESULTS, suffix  # noqa: E402

BENCH = "QQQ"


def load_prices(sfx: str = "", kind: str = "close") -> pd.DataFrame:
    p = pd.read_csv(RESULTS / f"prices_adj_{kind}{sfx}.csv", index_col=0)
    p.index = pd.to_datetime(p.index)
    return p


def ret2(a_px: float, b_px: float) -> float:
    return float((b_px / a_px - 1) * 100) if pd.notna(a_px) and pd.notna(b_px) and a_px > 0 else np.nan


def reaction_day(accept_et: pd.Timestamp, days: pd.DatetimeIndex) -> int:
    """반응일의 거래일 위치."""
    d = accept_et.normalize()
    if accept_et.hour * 60 + accept_et.minute >= 16 * 60:
        return int(days.searchsorted(d, side="right"))  # 다음 거래일
    return int(days.searchsorted(d, side="left"))  # 그날(휴일이면 다음 거래일)


def beta(stock: pd.Series, bench: pd.Series, end: int, n: int = 250) -> float:
    lo = max(1, end - n)
    rs = stock.iloc[lo - 1:end].pct_change().dropna()
    rb = bench.iloc[lo - 1:end].pct_change().dropna()
    j = rs.index.intersection(rb.index)
    if len(j) < 60:
        return np.nan
    c = np.cov(rs[j], rb[j])
    return float(c[0, 1] / c[1, 1])


def ret(p: pd.Series, a: int, b: int) -> float:
    if a < 0 or b >= len(p) or b < a:
        return np.nan
    x, y = p.iloc[a], p.iloc[b]
    return float((y / x - 1) * 100) if pd.notna(x) and pd.notna(y) and x > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="core", choices=["core", "broad"])
    sfx = suffix(ap.parse_args().set)
    ev = pd.read_csv(RESULTS / f"sec_events{sfx}.csv", dtype=str)
    q = pd.read_csv(RESULTS / f"quarters{sfx}.csv")
    px = load_prices(sfx)
    po = load_prices(sfx, "open").reindex(px.index)
    days = px.index
    bench = px[BENCH]
    ev["accept"] = pd.to_datetime(ev["acceptance_et"])
    # 분기 실적이 아닌 2.02(예비 실적 뒤 본 발표 중복, 다른 공시에 붙은 실적 언급)를 걸러 보유 구간이 잘리지 않게 한다.
    # 핵심 표본은 정독 결과의 is_quarterly_results로 먼저 거르고, 두 표본 모두 앞 접수에서 45일 안의 접수는 뺀다
    # (분기 발표 간격은 약 90일이다. XBRL 분기말로 묶으면 10-Q가 아직 없는 최신 발표가 빠져 보유 구간이 다음 실적을 넘는다).
    isq_path = RESULTS / "is_quarterly.csv"
    if not sfx and isq_path.exists():
        isq = pd.read_csv(isq_path, dtype={"accession": str})
        ev = ev[ev["accession"].isin(isq.loc[isq["is_q"], "accession"])]
    if sfx:
        # 확대 표본은 정독이 없으니 yfinance 실적 발표일 ±2일 안의 2.02만 남긴다
        # (TSLA 인도량·채굴사 월간 생산량처럼 실적이 아닌 2.02를 뺀다)
        ed = pd.read_csv(RESULTS / f"eps_consensus{sfx}.csv")
        ed["d"] = pd.to_datetime(ed["date"])
        fd_ = pd.to_datetime(ev["filing_date"])
        near = [((ed.loc[ed["ticker"] == t, "d"] - d).abs().dt.days.min() <= 2) for t, d in zip(ev["ticker"], fd_)]
        ev = ev[pd.Series(near, index=ev.index).fillna(False).astype(bool)]
    rows = []
    for t, g in ev.sort_values("accept").groupby("ticker"):
        if t not in px:
            continue
        p = px[t]
        o = po[t] if t in po else pd.Series(np.nan, index=px.index)
        ob = po[BENCH]
        qt = q[q["ticker"] == t].copy()
        qt["pe"] = pd.to_datetime(qt["period_end"])
        keep, last = [], None
        for d in pd.to_datetime(g["filing_date"]):
            ok = last is None or (d - last).days >= 45
            keep.append(ok)
            last = d if ok else last
        g = g[keep].reset_index(drop=True)
        for i, r in g.iterrows():
            k = reaction_day(r["accept"], days)
            if k >= len(days) or k < 1:
                continue
            nxt = reaction_day(g.loc[i + 1, "accept"], days) if i + 1 < len(g) else None
            end = (nxt - 1) if nxt is not None and nxt - 1 < len(days) else len(days) - 1
            partial = nxt is None or nxt - 1 >= len(days)
            b = beta(p, bench, k - 1)
            # 이 발표가 다루는 분기: 접수일 전 100일 안의 가장 늦은 분기말
            fd = pd.Timestamp(r["filing_date"])
            cand = qt[(qt["pe"] < fd) & ((fd - qt["pe"]).dt.days <= 100)]
            qrow = cand.iloc[-1] if len(cand) else None
            prev_q = qt[qt["pe"] < (qrow["pe"] if qrow is not None else fd)].iloc[-1:] if qrow is not None else None
            shares = None
            if prev_q is not None and len(prev_q):
                shares = prev_q["dil_shares"].iloc[0]  # 직전 분기 희석 주식수(발표 전에 알려진 값)
            price0 = p.iloc[k - 1]
            mcap = float(price0 * shares) if shares and pd.notna(shares) and pd.notna(price0) else np.nan
            row = {
                "ticker": t, "accession": r["accession"], "filing_date": r["filing_date"],
                "acceptance_et": r["acceptance_et"], "reaction_date": days[k].strftime("%Y-%m-%d"),
                "hold_end_date": days[end].strftime("%Y-%m-%d"), "partial": partial,
                "hold_days": end - k, "period_end": qrow["period_end"] if qrow is not None else None,
                "price_pre": round(float(price0), 4) if pd.notna(price0) else np.nan,
                "mcap_pre_b": round(mcap / 1e9, 2) if pd.notna(mcap) else np.nan,
                "beta_250": round(b, 3) if pd.notna(b) else np.nan,
                # 사전 변수(발표 전날 종가까지)
                "pre20": ret(p, k - 21, k - 1), "pre60": ret(p, k - 61, k - 1),
                "pre20_x": ret(p, k - 21, k - 1) - ret(bench, k - 21, k - 1),
                "pre60_x": ret(p, k - 61, k - 1) - ret(bench, k - 61, k - 1),
                # 결과 변수
                "r0": ret(p, k - 1, k), "r0_b": ret(bench, k - 1, k),
                "drift": ret(p, k, end), "drift_b": ret(bench, k, end),
                "total": ret(p, k - 1, end), "total_b": ret(bench, k - 1, end),
                "d5": ret(p, k, k + 5), "d5_b": ret(bench, k, k + 5),
                "d20": ret(p, k, k + 20), "d20_b": ret(bench, k, k + 20),
                "d60": ret(p, k, k + 60), "d60_b": ret(bench, k, k + 60),
                # 반응 5일 뒤 진입(직후 소음을 피하는 경우) → 보유 끝
                "drift5": ret(p, k + 5, end) if end > k + 5 else np.nan,
                "drift5_b": ret(bench, k + 5, end) if end > k + 5 else np.nan,
                # 발표 뒤 첫 정규장 시가에 들어가는 경우(장후 발표면 다음 날 시가, 장전 발표면 그날 시가)
                "gap": ret2(p.iloc[k - 1], o.iloc[k]), "gap_b": ret2(bench.iloc[k - 1], ob.iloc[k]),
                "o2c": ret2(o.iloc[k], p.iloc[k]), "o2c_b": ret2(ob.iloc[k], bench.iloc[k]),
                "odrift": ret2(o.iloc[k], p.iloc[end]), "odrift_b": ret2(ob.iloc[k], bench.iloc[end]),
            }
            # 회복: 반응일에 떨어졌다면 발표 전날 종가를 몇 거래일 만에 되찾나(다음 실적 전까지만 본다)
            row["recover_days"] = np.nan
            if pd.notna(row["r0"]) and row["r0"] < 0:
                path = p.iloc[k + 1:end + 1]
                hit = np.where(path.values >= price0)[0]
                row["recover_days"] = float(hit[0] + 1) if len(hit) else np.inf
            for c in ["r0", "drift", "total", "d5", "d20", "d60", "drift5", "gap", "o2c", "odrift"]:
                row[f"{c}_x"] = row[c] - row[f"{c}_b"] if pd.notna(row[c]) and pd.notna(row[f"{c}_b"]) else np.nan
                row[f"{c}_ab"] = (row[c] - b * row[f"{c}_b"]) if pd.notna(row[c]) and pd.notna(b) and pd.notna(row[f"{c}_b"]) else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(3)
    out.to_csv(RESULTS / f"events{sfx}.csv", index=False)
    print(f"이벤트 {len(out)}건, 종목 {out['ticker'].nunique()}개, 반응일 {out['reaction_date'].min()}~{out['reaction_date'].max()}")


if __name__ == "__main__":
    main()
