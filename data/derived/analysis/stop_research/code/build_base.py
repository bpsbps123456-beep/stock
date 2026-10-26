"""손절·재진입 연구(2026-10-10)의 기본 경로: 데스크 7종목 규칙을 그대로 돌린 모델 경로를 저장한다.

- base_2026: 데스크 현행 규칙(반도체 50%, desk/config.py RULE), 2026-01-02~10-09, 일별 TIME PDF, 데스크 run_model 그대로.
- base_2026_774: #774(반도체 60%) 같은 구간 — 이전 연구와 맞추기용.
- base_hold: 같은 규칙(반도체 50%)을 2023-05-22~2025-12-31 주간 TIME PDF(strategy_lab/holdout)로 돌린 경로. 보조 확인용.
  주의: 그 기간 보유 118종목 중 56개가 watchlist 미분류라 반도체 한도가 과하게 걸린다(desk §5).
가격: 2026은 rsi_breakout/cache/deskx_prices(2026-10-10 고정본), 보유 기간은 strategy_lab/holdout/prices_*.csv.
출력: cache/base_<name>.json — days, open(시가 경로 USD, 시작 1.0), close(그날 종가 평가), weights(그날 시가 거래 뒤), ticker, sector.
desk/ 코드는 고치지 않는다.
"""
import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from desk import config, marketdata, strategies as S                     # noqa: E402
from gbrc.etf.market import is_cash_holding, listing_market, yahoo_ticker  # noqa: E402
from gbrc.etf.research_data import exclusion_reason                      # noqa: E402

CACHE = HERE.parent / "cache"
FROZEN = ROOT / "data/derived/analysis/rsi_breakout/cache/deskx_prices"
HOLD = ROOT / "data/derived/analysis/strategy_lab/holdout"
END_2026 = "2026-10-09"


def frozen_history(folder):
    files = {f: folder / f"prices_{f.replace(' ', '_').lower()}.csv" for f in marketdata.FIELDS}
    return {f: pd.read_csv(p, index_col=0, parse_dates=True) for f, p in files.items()}


def book_2026():
    prices = frozen_history(FROZEN)
    marketdata.history = lambda tickers, force=False: prices
    return S.load_book()


def book_hold():
    snaps = {}
    for p in sorted((HOLD / "time_weekly").glob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        snaps[s["actual_date"]] = s
    keys, names = set(), {}
    for snap in snaps.values():
        for row in snap["holdings"]:
            if is_cash_holding(row) or exclusion_reason(row):
                continue
            key = S.research_key(row.get("code"))
            if listing_market(key) == "US":
                keys.add(key)
                names.setdefault(key, row.get("name", ""))
    ticker = {k: yahoo_ticker(k) for k in keys}
    prices = {f: pd.read_csv(HOLD / f"prices_{f.replace(' ', '_').lower()}.csv", index_col=0, parse_dates=True)
              for f in marketdata.FIELDS}
    adj = prices["Adj Close"]
    days = [d.strftime("%Y-%m-%d") for d in adj["QQQ"].dropna().index]

    def on_days(df):
        out = df.copy()
        out.index = out.index.strftime("%Y-%m-%d")
        return out.reindex(days)

    return S.Book(days=days, adj=on_days(adj), opn=on_days(prices["Open"]), cls=on_days(prices["Close"]),
                  fx=marketdata.ffill_on(adj["KRW=X"], days), snaps=snaps, pdfs=sorted(snaps),
                  keys=sorted(keys, key=lambda k: ticker[k]), ticker=ticker, names=names,
                  sector={k: S.sector_of(k, ticker[k]) for k in keys}, etf=adj["456600.KS"].dropna())


def run(book, start, end, rule):
    saved = dict(config.RULE)
    try:
        config.RULE.update(rule)
        path = S.run_model(book, start=start)
    finally:
        config.RULE.clear()
        config.RULE.update(saved)
    n = sum(d <= end for d in path.days)
    used = {k for w in path.weights[:n] for k in w}
    return {"days": path.days[:n], "open": [float(x) for x in path.value[:n]],
            "close": [float(x) for x in path.close[:n]],
            "weights": [{k: round(float(v), 8) for k, v in w.items()} for w in path.weights[:n]],
            "ticker": {k: book.ticker[k] for k in used}, "sector": {k: book.sector[k] for k in used},
            "rule": dict(rule)}


def summary(name, b, lo=None, hi=None):
    days = b["days"]
    v = pd.Series(b["open"], index=days)
    r = (v.shift(-1) / v - 1).dropna()          # 시가 k → 시가 k+1 수익을 k일에 붙인다(deskx와 같음)
    if lo:
        r = r[(r.index >= lo) & (r.index <= hi)]
    c = (1 + r).cumprod()
    return f"{name} {lo or days[0]}~{hi or days[-1]}: 수익 {c.iloc[-1] - 1:+.1%}, 낙폭 {(c / c.cummax() - 1).min():.1%}"


def main():
    CACHE.mkdir(exist_ok=True)
    b26 = book_2026()
    cur = dict(config.RULE)
    out = {"base_2026": run(b26, "2026-01-02", END_2026, cur),
           "base_2026_774": run(b26, "2026-01-02", END_2026, dict(config.CODEX_774_RULE))}
    bh = book_hold()
    out["base_hold"] = run(bh, "2023-05-22", "2025-12-31", cur)
    for name, b in out.items():
        (CACHE / f"{name}.json").write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
        print(summary(name, b))
    print(summary("base_2026", out["base_2026"], "2026-01-02", "2026-05-29"))
    print(summary("base_2026", out["base_2026"], "2026-06-01", "2026-10-09"))
    print(summary("base_hold", out["base_hold"], "2025-01-01", "2025-12-31"))
    print(summary("base_hold", out["base_hold"], "2024-01-01", "2024-12-31"))
    unk = sorted({b["ticker"][k] for k, s in out["base_hold"]["sector"].items() if s == S.UNKNOWN})
    print("보유 기간 미분류 보유 종목:", len(unk), unk)


if __name__ == "__main__":
    main()
