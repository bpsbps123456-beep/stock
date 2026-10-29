"""실적 지지 구역 지정가 연구 — 원자료 받기(yfinance만, 트레이딩뷰 금지).

실행: py data/derived/analysis/earnings_support/code/fetch.py --set live|priority|rest|all --kind h1|d1|both

- h1: 1시간봉 프리·애프터 포함(period=730d). 종목마다 cache/h1/<T>.parquet(뉴욕 시각). 이미 받은 종목은 건너뛴다.
- d1: 일봉(auto_adjust=False → 분할 반영·배당 미반영, 1시간봉과 같은 기준) 시가·고가·저가·종가 → cache/d1_<field>.parquet.
- 묶음 40종목씩, 빈 결과는 쉬었다가 한 종목씩 최대 2번 다시 묻는다.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
CACHE = ROOT / "cache"
H1 = CACHE / "h1"
EVENTS = REPO / "data/derived/analysis/earnings_trend/results/events_wide.csv"
LIVE = ["SNDK", "INTC", "RDDT", "AMZN", "AXTI", "TWLO", "MSTR", "ILMN", "COIN", "FSLR"]
WIN = ("2025-03-01", "2026-10-28")
D1_START = "2023-06-01"


def ai_us() -> set[str]:
    sys.path.insert(0, str(REPO))
    import watchlist  # noqa: E402
    out = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for t in x:
                if isinstance(t, str) and "." not in t:
                    out.add(t)
    walk(watchlist.AI_UNIVERSE)
    return out


def universe(which: str) -> list[str]:
    ev = pd.read_csv(EVENTS)
    ev = ev[(ev.reaction_date >= WIN[0]) & (ev.reaction_date <= WIN[1])]
    ai = ai_us()
    pri = sorted(set(ev.loc[(ev.mcap_b >= 10) | ev.ticker.isin(ai), "ticker"]))
    rest = sorted(set(ev.ticker) - set(pri))
    if which == "live":
        return LIVE
    if which == "priority":
        return LIVE + [t for t in pri if t not in LIVE]
    if which == "rest":
        return rest
    return LIVE + [t for t in pri if t not in LIVE] + rest


def _frame(d: pd.DataFrame, t: str, multi: bool) -> pd.DataFrame | None:
    try:
        x = d[t] if multi else d
    except KeyError:
        return None
    x = x[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "Close"])
    return x if len(x) else None


def fetch_h1(tickers: list[str], batch: int = 40) -> None:
    H1.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if not (H1 / f"{t}.parquet").exists()]
    print(f"h1: {len(tickers)}종목 중 받을 것 {len(todo)}", flush=True)
    miss = []
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        t0 = time.time()
        try:
            d = yf.download(chunk, period="730d", interval="1h", prepost=True, auto_adjust=False,
                            progress=False, threads=True, group_by="ticker")
        except Exception as exc:  # 묶음 전체 실패면 한 종목씩 다시
            print("  묶음 실패", type(exc).__name__, exc, flush=True)
            miss += chunk
            time.sleep(20)
            continue
        got = 0
        for t in chunk:
            x = _frame(d, t, len(chunk) > 1) if not d.empty else None
            if x is None:
                miss.append(t)
                continue
            x = x.copy()
            x.index = x.index.tz_convert("America/New_York")
            x.index.name = "ts"
            x.to_parquet(H1 / f"{t}.parquet")
            got += 1
        print(f"  {i + len(chunk)}/{len(todo)} 받음 {got}/{len(chunk)} ({time.time() - t0:.1f}s)", flush=True)
        if got == 0:
            time.sleep(30)
    for attempt in range(2):
        if not miss:
            break
        time.sleep(15)
        again = []
        for t in miss:
            try:
                start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=728)).strftime("%Y-%m-%d")
                d = yf.download(t, start=start, interval="1h", prepost=True, auto_adjust=False,
                                progress=False, threads=False)  # period=730d가 상장일로 시작을 잡아 거부되는 종목(RDDT)
                x = _frame(d.droplevel(1, axis=1) if isinstance(d.columns, pd.MultiIndex) else d, t, False)
            except Exception:
                x = None
            if x is None:
                again.append(t)
                time.sleep(2)
                continue
            x = x.copy()
            x.index = x.index.tz_convert("America/New_York")
            x.index.name = "ts"
            x.to_parquet(H1 / f"{t}.parquet")
        print(f"  다시 묻기 {attempt + 1}: 남은 실패 {len(again)}", flush=True)
        miss = again
    if miss:
        (CACHE / "h1_missing.txt").write_text("\n".join(miss), encoding="utf-8")
        print("못 받음:", len(miss), miss[:20], flush=True)


def fetch_d1(tickers: list[str], batch: int = 200) -> None:
    fields = {f: [] for f in ("Open", "High", "Low", "Close")}
    have = {}
    for f in fields:
        p = CACHE / f"d1_{f.lower()}.parquet"
        if p.exists():
            have[f] = pd.read_parquet(p)
    done = set(have["Close"].columns) if "Close" in have else set()
    todo = [t for t in tickers if t not in done]
    print(f"d1: 받을 것 {len(todo)}", flush=True)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        d = yf.download(chunk, start=D1_START, end="2026-10-30", interval="1d", auto_adjust=False,
                        progress=False, threads=True, group_by="column")
        for f in fields:
            x = d[f] if isinstance(d.columns, pd.MultiIndex) else d[[f]].rename(columns={f: chunk[0]})
            fields[f].append(x)
        print(f"  {i + len(chunk)}/{len(todo)}", flush=True)
    for f, parts in fields.items():
        if not parts:
            continue
        new = pd.concat(parts, axis=1)
        if f in have:
            new = pd.concat([have[f], new[[c for c in new.columns if c not in have[f].columns]]], axis=1)
        new.index = pd.to_datetime(new.index).tz_localize(None)
        new.to_parquet(CACHE / f"d1_{f.lower()}.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="priority", choices=("live", "priority", "rest", "all"))
    ap.add_argument("--kind", default="both", choices=("h1", "d1", "both"))
    a = ap.parse_args()
    tick = universe(a.set)
    CACHE.mkdir(parents=True, exist_ok=True)
    if a.kind in ("d1", "both"):
        fetch_d1(tick + (["SPY", "QQQ"] if a.set != "rest" else []))
    if a.kind in ("h1", "both"):
        fetch_h1(tick)


if __name__ == "__main__":
    main()
