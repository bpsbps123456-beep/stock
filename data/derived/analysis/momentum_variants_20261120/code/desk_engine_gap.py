"""연구 엔진(engine.py)과 데스크 엔진(desk.strategies.run_model)의 2026 숫자 차이 확인 (2026-11-21).

1) 두 엔진의 주간 바구니(종목·비중)가 같은지, 2) 데스크 재생을 배당 반영 시가로 바꾸면 연구 숫자와 같아지는지 본다.
데스크 가격은 로컬 데스크 캐시(data/derived/desk/prices_*.csv)를 그대로 읽고 새로 받지 않는다.
사용: py data/derived/analysis/momentum_variants_20261120/code/desk_engine_gap.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
sys.path[:0] = [str(ROOT), str(BASE / "code"), str(BASE / "branches/composite")]
import composite as C  # noqa: E402
import engine as E  # noqa: E402
from desk import config, marketdata  # noqa: E402
from desk import strategies as S  # noqa: E402

marketdata.history = lambda tickers, force=False: {
    f: pd.read_csv(marketdata._path(f), index_col=0, parse_dates=True) for f in marketdata.FIELDS}
RULES = {"res60": {**config.RULE, "score": "res", "lookback": 60},
         "ret120": {k: v for k, v in config.RULE.items() if k != "score"} | {"lookback": 120}}


def stats(p):
    d, o, c = p.days, p.value, p.close

    def span(a, b):
        i = next(j for j, x in enumerate(d) if x >= a)
        k = max(j for j, x in enumerate(d) if x <= b)
        return c[k] / o[i] - 1
    curve = np.concatenate([[1.0], c])
    return {"Y2026": span("2026-01-02", E.LAST), "H1": span("2026-01-02", "2026-06-30"),
            "H2": span("2026-07-01", E.LAST), "RECENT": span("2026-09-01", E.LAST),
            "MDD": float((curve / np.maximum.accumulate(curve) - 1).min())}


def replay(book, rule):
    saved = dict(config.RULE)
    config.RULE.clear(); config.RULE.update(rule)
    try:
        return S.run_model(book)
    finally:
        config.RULE.clear(); config.RULE.update(saved)


def main():
    e = E.Engine()
    research = e.run(C.residual(e, 60), universe="TIME", n=7, buffer=5, weight="time", caps=True)
    book = S.load_book(through=E.LAST)
    desk = replay(book, RULES["res60"])
    differ = 0
    for day, target in desk.rebalances.items():
        a = {book.ticker[k]: v for k, v in target.items()}
        b = {k: v for k, v in research.holdings.get(day, {}).items() if v > 1e-9}
        if set(a) != set(b) or max(abs(a[k] - b[k]) for k in a) > 0.005:
            differ += 1
    print(f"주간 교체 {len(desk.rebalances)}회 중 바구니가 다른 회차 {differ}")
    print("연구 엔진 RES60:", {k: round(v, 4) for k, v in e.metrics(research).items() if k in ("Y2026", "H1", "H2", "RECENT", "MDD2026")})
    for name, rule in RULES.items():
        print("데스크 원시 시가", name, {k: round(v, 4) for k, v in stats(replay(book, rule)).items()})
    book.opn = book.opn * book.adj / book.cls
    book.cls = book.adj
    for name, rule in RULES.items():
        print("데스크 배당 반영 시가", name, {k: round(v, 4) for k, v in stats(replay(book, rule)).items()})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
