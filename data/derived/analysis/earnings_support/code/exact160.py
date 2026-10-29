"""160분봉 근사 점검: 1시간봉 근사(12:30·14:30 경계)와 5분봉 진짜 160분봉(12:10·14:50 경계)의 지지 구역 비교.

실행: py data/derived/analysis/earnings_support/code/exact160.py
대상: 반응일 2026-10-01~10-28 사건(스윙 60개 이상). 5분봉은 최근 60일만 있으므로 두 방식 모두 5분봉 첫날부터 자른
같은 기간으로 구역을 만든다(봉 경계 차이만 본다). 참고로 전체 이력 근사 구역과 자른 근사 구역도 비교한다(이력 길이 효과).
출력: results/exact160_check.csv, 화면 요약
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lib import CACHE, RES, bars160, load_h1, pivots, zones  # noqa: E402

M5 = CACHE / "m5"


def fetch5(tickers: list[str]) -> None:
    M5.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if not (M5 / f"{t}.parquet").exists()]
    for i in range(0, len(todo), 40):
        chunk = todo[i:i + 40]
        d = yf.download(chunk, period="60d", interval="5m", prepost=False, auto_adjust=False, progress=False,
                        threads=True, group_by="ticker")
        for t in chunk:
            try:
                x = (d[t] if len(chunk) > 1 else d.droplevel(1, axis=1) if isinstance(d.columns, pd.MultiIndex) else d)
                x = x[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "Close"])
            except KeyError:
                continue
            if len(x):
                x = x.copy()
                x.index = x.index.tz_convert("America/New_York")
                x.to_parquet(M5 / f"{t}.parquet")
        print(f"5m {i + len(chunk)}/{len(todo)}", flush=True)
        time.sleep(1)


def match(a: list, b: list, k: int) -> tuple[float | None, float | None]:
    """a의 k번째 지지 하단이 b의 k번째와 0.5% 안인지, b의 어느 구역과 0.5% 안인지."""
    if len(a) < k:
        return None, None
    x = a[k - 1][0]
    same = abs(x / b[k - 1][0] - 1) <= 0.005 if len(b) >= k else False
    anyz = any(abs(x / z[0] - 1) <= 0.005 or (z[0] * 0.995 <= x <= z[1] * 1.005) for z in b) if b else False
    return float(same), float(anyz)


def main() -> None:
    meta = pd.read_parquet(CACHE / "ev_meta.parquet")
    ev = meta[(meta.reaction_date >= "2026-10-01") & (meta.swings >= 60)]
    fetch5(sorted(set(ev.ticker)))
    rows = []
    for r in ev.itertuples():
        p5 = M5 / f"{r.ticker}.parquet"
        h = load_h1(r.ticker)
        if not p5.exists() or h is None:
            continue
        m5 = pd.read_parquet(p5)
        m5.index = m5.index.tz_convert("America/New_York")
        first = m5.index.tz_localize(None).normalize().min()
        bx = bars160(m5, exact=True)
        ba_full = bars160(h)
        ba = ba_full[ba_full["day"] >= first].reset_index(drop=True)
        bx = bx[bx["day"] >= first].reset_index(drop=True)
        base = pd.Timestamp(r.base_date)
        if base not in set(bx["day"]) or base not in set(ba["day"]):
            continue
        Dx = int(np.flatnonzero(bx["day"] == base)[-1])
        Da = int(np.flatnonzero(ba["day"] == base)[-1])
        Df = int(np.flatnonzero(ba_full["day"] == base)[-1])
        sx, _, nx = zones(pivots(bx), Dx, float(bx["close"].iloc[Dx]))
        sa, _, na = zones(pivots(ba), Da, float(ba["close"].iloc[Da]))
        sf, _, nf = zones(pivots(ba_full), Df, float(ba_full["close"].iloc[Df]))
        for k in range(1, 6):
            same, anyz = match(sa, sx, k)
            hs, ha = match(sf, sa, k)
            rows.append({"ticker": r.ticker, "reaction_date": r.reaction_date, "k": k, "swings_exact": nx,
                         "swings_approx": na, "approx_vs_exact_same_k": same, "approx_vs_exact_any": anyz,
                         "full_vs_trunc_same_k": hs, "full_vs_trunc_any": ha,
                         "dist_exact": (sx[k - 1][0] / r.base_close - 1) if len(sx) >= k else np.nan,
                         "dist_approx": (sa[k - 1][0] / r.base_close - 1) if len(sa) >= k else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "exact160_check.csv", index=False)
    print("사건", out[["ticker", "reaction_date"]].drop_duplicates().shape[0])
    print(out.groupby("k")[["approx_vs_exact_same_k", "approx_vs_exact_any", "full_vs_trunc_same_k",
                            "full_vs_trunc_any"]].mean().round(3))
    print(out[["swings_exact", "swings_approx"]].describe().round(1))


if __name__ == "__main__":
    main()
