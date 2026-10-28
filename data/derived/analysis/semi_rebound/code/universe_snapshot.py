"""AI 유니버스 미국 상장 전 종목의 예측 입력 스냅샷 (2026-10-29 02시 KST, 10/28 미국 장중).

가격 요인(첫 기록과 같은 정의): 베타 = 10/5 고점 전 60거래일 SOXX 대비, 10월 하락 = 10/5→10/22,
반등 초반 = 10/22→10/27, 7월 반등 = 7/29→8/17, 52주 고점 대비. 전망 수정률 = (지금 − 과거) ÷ |과거|.
두 펀드 = TIME·KoAct PDF 수량 기준 실매매(D-5 10/21, 10/23 이후, D-20 9/28).
결과: ../results/universe_snapshot_2026-10-28.json
"""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import watchlist  # noqa: E402
from gbrc.etf import flows, storage  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
GROUP = {s: g for g, subs in watchlist.AI_UNIVERSE.items() for sub, syms in subs.items() for s in syms if "." not in s}
SUB = {s: sub for g, subs in watchlist.AI_UNIVERSE.items() for sub, syms in subs.items() for s in syms if "." not in s}
TICKERS = sorted(GROUP)


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def fundamentals(t: str) -> dict:
    out = {}
    try:
        tk = yf.Ticker(t)
        info = tk.info or {}
        out.update(si=num(info.get("shortPercentOfFloat")), mcap=num(info.get("marketCap")), fpe=num(info.get("forwardPE")),
                   name=info.get("shortName") or info.get("longName"))
        y = tk.eps_trend.loc["+1y"]
        cur, d30, d90 = num(y["current"]), num(y["30daysAgo"]), num(y["90daysAgo"])
        change = lambda a, b: None if a is None or not b else (a - b) / abs(b)
        out.update(eps=cur, rev30=change(cur, d30), rev90=change(cur, d90))
    except Exception as exc:  # one ticker must not stop the snapshot
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def main() -> None:
    px = yf.download(TICKERS + ["SOXX"], start="2025-06-01", auto_adjust=True, progress=False, threads=True)["Close"]
    px = px.dropna(how="all").loc[:"2026-10-27"]
    ret = px.pct_change().loc[:"2026-10-05"].iloc[-60:]
    rows = {}
    for t in TICKERS:
        p = px[t] if t in px else pd.Series(dtype=float)
        at = lambda d: num(p.get(pd.Timestamp(d)))
        r = lambda a, b: None if at(a) is None or at(b) is None else at(b) / at(a) - 1
        b = ret[t].dropna() if t in ret else pd.Series(dtype=float)
        beta = None if len(b) < 40 else num(b.cov(ret["SOXX"].loc[b.index]) / ret["SOXX"].loc[b.index].var())
        last = p.dropna()
        rows[t] = {"group": GROUP[t], "sub": SUB[t], "beta": beta, "drop": r("2026-10-05", "2026-10-22"),
                   "early": r("2026-10-22", "2026-10-27"), "july": r("2026-07-29", "2026-08-17"),
                   "hi": None if last.empty else num(last.iloc[-1] / last.iloc[-252:].max() - 1)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for t, f in zip(TICKERS, pool.map(fundamentals, TICKERS)):
            rows[t].update(f)
    alias = {k: v for k, v in watchlist.AI_UNIVERSE_ALIASES.items() if "." not in v}
    for fid, fund in (("456600", "TIME"), ("2ETFQ1", "KoAct")):
        snaps = storage.load_fund_snapshots(fid, "2026-10-28")
        for label, base in (("D5", "2026-10-21"), ("3d", "2026-10-23"), ("D20", "2026-09-28")):
            res = flows.flows_between(snaps, base, "2026-10-28")
            for f in res.flows.values():
                code = f.code.split()[0]
                code = alias.get(code, code)
                if code in rows:
                    rows[code].setdefault("funds", {})[f"{fund}_{label}"] = {
                        "kind": f.kind, "pct": None if f.selective_pct is None else round(f.selective_pct, 1), "weight": round(f.weight, 2)}
    snap = {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "last_close": px.index[-1].date().isoformat(),
            "tickers": rows}
    (RES / "universe_snapshot_2026-10-28.json").write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [t for t, v in rows.items() if v.get("error") or v.get("beta") is None]
    print(f"{len(rows)} tickers, last close {snap['last_close']}, missing beta/fundamentals: {bad}")


if __name__ == "__main__":
    main()
