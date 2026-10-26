"""Read-only reproduction of the frozen Claude H6/H6e outputs.

Run from the repository: py data/derived/analysis/2026-09-23_strategy_research/strategy_comparison/claude_reproduce.py
All new artifacts, including Numba caches, stay beside this script.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[4]
SRC = REPO / "data/derived/analysis/strategy_lab"
sys.dont_write_bytecode = True
os.environ["NUMBA_CACHE_DIR"] = str(OUT / "claude_numba_cache")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
sys.path.insert(0, str(SRC / "code"))
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

FILES = [SRC / name for name in (
    "lab_panels.json", "prices_open.csv", "prices_close.csv", "prices_adj_close.csv",
    "logs/final_2026.txt", "2026-09-23_strategy_lab.md", "code/data.py",
    "code/engine.py", "code/common.py", "code/grid.py", "code/finalists.py",
    "code/compare_codex.py",
)]


def hashes():
    return {str(p.relative_to(REPO)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES}


def main():
    before = hashes()
    import data as D
    import engine as E
    import common as C

    # Read caches directly: never invoke a loader that might refresh/write source data.
    d = json.loads((SRC / "lab_panels.json").read_text(encoding="utf-8"))
    d["prices"] = {f: pd.read_csv(SRC / f"prices_{f.replace(' ', '_').lower()}.csv", index_col=0, parse_dates=True)
                   for f in ("Open", "Close", "Adj Close")}
    specs = {"H6": {"source": "T", "score": "MOM120", "max_stocks": 5, "rebalance_id": 0,
                    "rebalance": "new_PDF_decision_days", "buffer": 5, "weighting_id": 1, "weighting": "selected_TIME_weights_normalized"},
             "H6e": {"source": "T", "score": "MOM120", "max_stocks": 5, "rebalance_id": 1,
                     "rebalance": "first_new_PDF_decision_day_of_ISO_week", "buffer": 5, "weighting_id": 0, "weighting": "equal"}}
    result = {"scope": "frozen Claude engine reproduction; no selection or source modification", "configs": specs,
              "source_hashes": before, "paths": {}, "cache": {}, "validation": {}}
    result["methodology"] = {
        "momentum": "previous USD Adj Close / USD Adj Close 120 QQQ sessions earlier - 1; includes pre-2026 warmup",
        "eligibility": "positive TIME NAV holding on signal PDF, finite MOM120, same-date Adj Close available",
        "buffer": "retain incumbents ranked <= 10, then fill remaining slots to at most 5 by momentum rank",
        "execution": "Open * (Adj Close / raw Close); hold from one adjusted open to the next; final mark is September 22 open",
        "FX": "KRW=X daily-close ratios applied to entire USD NAV, including cash; not contemporaneous open FX",
        "delay": "shift scores, weights, held masks, inverse volatility by new-PDF decision rows, keeping execution masks fixed",
        "disclosure": "same-day PDF <= US date; ~21:45 KST publication assumed by original source, unverified",
        "segment_returns": "continuing portfolio slices, not fresh cash starts",
        "excluded": "explicit commissions 0; no taxes, bid-ask spread, FX spread or order-book/partial-fill model",
    }
    result["selection_audit"] = {
        "finding": "Jan-May-only selection is demonstrated at source/prefilter/score/rebalance family level; exact H6e N5/equal/buffer5 training-only freeze is not evidenced by saved code",
        "evidence": ["code/analyze9.py:48-72 groups N>=3 neighbours and ranks IS p25; no exact configuration selection",
                     "code/analyze8.py:87-97 and code/finalists.py:16-22 hard-code exact finalists",
                     "code/analyze3.py:56-82 initially ranks families by full-period p25",
                     "2026-09-23_strategy_lab.md sections 6 and 10 disclose post-full-sample robustness design"],
        "interpretation": "H6e later-period outperformance reproduces arithmetically; do not label exact H6e a prospectively frozen untouched test",
        "original_family_rank_from_log": {"T/all/MOM120/D": {"base": 2, "delay1": 1, "delay1_30bp": 2},
                                          "T/all/MOM120/W": {"base": "not in printed top12", "delay1": 7, "delay1_30bp": 6}},
    }
    for field, frame in d["prices"].items():
        result["cache"][field] = {"first": str(frame.index.min().date()), "last": str(frame.index.max().date()),
                                   "rows": len(frame), "columns": len(frame.columns),
                                   "QQQ_first": str(frame["QQQ"].dropna().index.min().date()),
                                   "QQQ_last": str(frame["QQQ"].dropna().index.max().date()),
                                   "all_missing_tickers": [d["ticker"][c] for c in d["companies"]
                                                           if d["ticker"][c] not in frame or frame[d["ticker"][c]].isna().all()]}
    result["cache"].update({"companies": len(d["companies"]), "company_order_sorted": d["companies"] == sorted(d["companies"]),
                            "snapshot_count": len(d["pdf_dates"]), "snapshot_first": d["pdf_dates"][0], "snapshot_last": d["pdf_dates"][-1],
                            "GOOG_present": "GOOG" in d["prices"]["Adj Close"], "GOOGL_present": "GOOGL" in d["prices"]["Adj Close"],
                            "company_aliases": D.COMPANY_ALIASES,
                            "units": {"US_stock_Open_Close_AdjClose": "USD/share", "KRW=X": "KRW/USD", "456600.KS_0015B0.KS": "KRW/share", "VIX": "index"}})
    caught = []
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        for execution in ("open", "close"):
            base = E.build_market(d, exec_mode=execution)
            names, original_stack = E.build_scores(base)
            if execution == "open":
                days_all = base.feat["days_all"]
                first_idx = int(base.feat["dec_t"][0])
                result["calendar"] = {"first": base.days[0], "last": base.days[-1], "US_days": len(base.days),
                                      "decision_rows": len(base.dec_days), "first_MOM120_start": days_all[first_idx - 121],
                                      "first_MOM120_end": days_all[first_idx - 1], "days": base.days,
                                      "decision_days": base.dec_days, "decision_PDF_dates": base.dec_pdf}
            for delay in (0, 1, 2):
                m = copy.deepcopy(base)
                stack = E.apply_delay(m, original_stack, delay)
                for label, spec in specs.items():
                    param = [names.index(("T", "all", "MOM120")), 5, spec["weighting_id"], 5, spec["rebalance_id"], 0, 0]
                    r, nh, turn = C.simulate(m, stack, param, cost=0.0)
                    stat = C.stats(r)
                    dates = np.asarray(m.days)
                    segments = {}
                    for tag, start, end in (("Jan_May", "2026-01-02", "2026-05-31"),
                                             ("Jun_Jul", "2026-06-01", "2026-07-31"),
                                             ("Aug_Sep", "2026-08-01", "2026-09-22")):
                        mask = (dates >= start) & (dates <= end)
                        segments[tag] = float(100 * (np.prod(1 + r[mask]) - 1))
                    result["paths"][f"{label}_{execution}_delay{delay}"] = {
                        "rule": label, "execution": execution, "delay_decision_rows": delay, "cost_bps": 0,
                        "params": param, "total_return_pct": float(100 * stat["total"]),
                        "max_drawdown_pct": float(100 * stat["mdd"]), "segments_return_pct": segments,
                        "max_holdings": int(nh.max()), "average_holdings": float(nh.mean()),
                        "cash_days": int((nh == 0).sum()), "turnover_sum": float(turn.sum()),
                        "monthly_return_pct": {mo: float(100 * (np.prod(1 + r[np.char.startswith(dates, mo)]) - 1))
                                               for mo in sorted({x[:7] for x in m.days})},
                        "returns": r.tolist(), "nav": np.cumprod(1 + r).tolist()}
        caught = sorted({str(w.message) for w in ws})
    result["runtime_warnings"] = caught
    adj = d["prices"]["Adj Close"]
    result["original_benchmarks"] = {}
    for name, ticker in D.ETF_TICKERS.items():
        p = adj[ticker].dropna().loc["2026-01-05":"2026-09-23"]
        result["original_benchmarks"][name] = {"ticker": ticker, "first": str(p.index[0].date()), "last": str(p.index[-1].date()),
                                                  "total_return_pct": float(100 * (p.iloc[-1] / p.iloc[0] - 1)),
                                                  "basis": "KRX Adj Close; different calendar bounds from US strategy"}
    expected = {"H6": [261, 237, 242], "H6e": [210, 204, 167]}
    checks = []
    for rule, values in expected.items():
        for delay, value in enumerate(values):
            actual = result["paths"][f"{rule}_open_delay{delay}"]["total_return_pct"]
            checks.append({"key": f"{rule}_open_delay{delay}", "expected_rounded_pct": value,
                           "actual_pct": actual, "passed": round(actual) == value})
    for rule, seg, expected_pct in (("H6e", "Jun_Jul", -12.8), ("H6e", "Aug_Sep", 8.1),
                                     ("H6", "Jun_Jul", -14.4), ("H6", "Aug_Sep", 7.0)):
        actual = result["paths"][f"{rule}_open_delay1"]["segments_return_pct"][seg]
        checks.append({"key": f"{rule}_open_delay1_{seg}", "expected_rounded_pct": expected_pct,
                       "actual_pct": actual, "passed": round(actual, 1) == expected_pct})
    result["validation"] = {"source_hashes_unchanged": hashes() == before, "logged_values": checks,
                            "all_paths_max5": all(p["max_holdings"] <= 5 for p in result["paths"].values()),
                            "all_logged_values_reproduced": all(x["passed"] for x in checks),
                            "new_selection": False}
    (OUT / "native_reproduction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"validation": result["validation"], "cache": result["cache"],
                      "calendar": {k: v for k, v in result["calendar"].items() if not isinstance(v, list)},
                      "benchmarks": result["original_benchmarks"]}, ensure_ascii=False, indent=2))
    assert result["validation"]["source_hashes_unchanged"]
    assert result["validation"]["all_logged_values_reproduced"]
    assert result["validation"]["all_paths_max5"]


if __name__ == "__main__":
    main()
