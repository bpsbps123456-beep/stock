"""Isolated identity-filter diagnostic; preserves every original lab artifact.

Only changes the three US ISIN rows' canonical listing codes. Reuses the original
historical H6e engine, dates, open/+1-session availability, FX and zero-cost model.
This is a retrospective sensitivity audit, not an independently validated result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["NUMBA_DISABLE_JIT"] = "1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LAB = REPO / "data/derived/analysis/strategy_lab"
sys.path.insert(0, str(LAB / "code"))
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import common as C
import data as D
import engine as E
import holdout as H
from gbrc.etf import market

ISIN_TO_CODE = {
    "US86800U1043": "SMCI US EQUITY",
    "US12430A1025": "BZFD US EQUITY",
    "US0404131064": "ANET US EQUITY",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs = sorted((LAB / "holdout").rglob("*.json")) + sorted((LAB / "holdout").glob("*.csv"))
    inputs += [LAB / "code" / f"{name}.py" for name in ("holdout", "engine", "common", "data")]
    inputs += [LAB / "lab_panels.json", *sorted(LAB.glob("prices_*.csv"))]
    before = {str(p.relative_to(REPO)): digest(p) for p in inputs}
    snaps = H.load_pdfs()
    fixed = copy.deepcopy(snaps)
    omitted = []
    for day, snap in snaps.items():
        for i, row in enumerate(snap["holdings"]):
            if row["code"] in ISIN_TO_CODE:
                assert not market.is_us_listed(row["code"])
                omitted.append({"date": day, "source": str((H.PDF_DIR / f"{day}.json").relative_to(REPO)),
                                "code": row["code"], "name": row["name"], "weight_pct": float(row["weight"]),
                                "mapped_code": ISIN_TO_CODE[row["code"]]})
                fixed[day]["holdings"][i]["code"] = ISIN_TO_CODE[row["code"]]
    assert len(omitted) == 98
    prices = {field: pd.read_csv(LAB / "holdout" / f"prices_{field.replace(' ', '_').lower()}.csv",
                                index_col=0, parse_dates=True) for field in D.PRICE_FIELDS}
    missing = [symbol for symbol in ("SMCI", "BZFD", "ANET")
               if any(symbol not in prices[field].columns for field in D.PRICE_FIELDS)]
    price_notes = []
    if missing:
        cache_paths = {field: HERE / f"added_prices_{field.replace(' ', '_').lower()}.csv"
                       for field in D.PRICE_FIELDS}
        cached = all(path.exists() for path in cache_paths.values())
        if cached:
            print(f"Reusing isolated added-price cache: {missing}", flush=True)
            extra_fields = {field: pd.read_csv(path, index_col=0, parse_dates=True)
                            for field, path in cache_paths.items()}
        else:
            import yfinance as yf
            print(f"Downloading isolated missing prices only: {missing}", flush=True)
            frame = yf.download(missing, start=H.PRICE_START, end=H.PRICE_END,
                                auto_adjust=False, progress=False, threads=False)
            extra_fields = {field: frame[field] for field in D.PRICE_FIELDS}
        for field in D.PRICE_FIELDS:
            extra = extra_fields[field]
            if isinstance(extra, pd.Series):
                extra = extra.to_frame(name=missing[0])
            assert all(symbol in extra.columns and extra[symbol].notna().any() for symbol in missing)
            path = cache_paths[field]
            if not cached:
                extra.to_csv(path)
            price_notes.append({"file": path.name, "sha256": digest(path), "source": "Yahoo Finance via yfinance",
                                "field": field, "symbols": missing, "start": H.PRICE_START, "end_exclusive": H.PRICE_END})
            prices[field] = pd.concat([prices[field], extra[missing].reindex(prices[field].index)], axis=1)

    def run(snapshots):
        d = H.build_d(snapshots, prices, shift=1)
        E.BT_START, E.BT_END = d["pdf_dates"][0], H.BT_END
        m = E.build_market(d, exec_mode="open")
        names, stack = E.build_scores(m)
        param = [names.index(("T", "all", "MOM120")), 5, 0, 5, 1, 0, E.SOURCES.index("T")]
        returns, counts, turnover = C.simulate(m, stack, param, cost=0.0)
        stats = C.stats(returns)
        return {"total": float(stats["total"]), "mdd": float(stats["mdd"]),
                "sharpe": float(stats["sharpe"]), "start": m.days[0], "end": m.days[-1],
                "sessions": len(m.days), "universe_count": len(m.tickers),
                "omitted_price_companies": [company for company, ticker in d["ticker"].items()
                                            if ticker not in prices["Adj Close"].columns],
                "average_holdings": float(np.mean(counts[1:]))}

    original = run(snaps)
    corrected = run(fixed)
    benchmark = H.etf_return(prices, "456600.KS", H.BT_START, H.ETF_END)
    # Separate exact reproduction of the original 2026 "17 entries" statistic.
    # This is close/+0, not the common engine's open/+1 case.
    E.BT_START, E.BT_END = "2026-01-02", "2026-09-22"
    m2026 = E.build_market(D.load(), exec_mode="close")
    names2026, stack2026 = E.build_scores(m2026)
    p2026 = [names2026.index(("T", "all", "MOM120")), 5, 0, 5, 1, 0, E.SOURCES.index("T")]
    weights2026, _, turn2026 = C.weights_path(m2026, stack2026, p2026)
    held2026 = weights2026 > 0
    initial_entries = int(held2026[0].sum())
    later_entries = int((held2026[1:] & ~held2026[:-1]).sum())
    tmax, jmax = np.unravel_index(np.argmax(weights2026), weights2026.shape)
    h6e2026 = {
        "scope": "Original Claude 2026 engine, close execution, no information delay; distinct from common open/+1 comparison",
        "start": m2026.days[0], "end": m2026.days[-1],
        "initial_position_entries": initial_entries, "later_zero_to_positive_entries": later_entries,
        "reported_total_entries": initial_entries + later_entries,
        "days_with_any_turnover_including_initial_entry": int(np.count_nonzero(turn2026 > 1e-10)),
        "max_end_of_day_weight_pct": float(weights2026[tmax, jmax] * 100),
        "max_weight_date": m2026.days[tmax], "max_weight_symbol": m2026.tickers[jmax],
        "interpretation": "17 includes initial five plus twelve later position entries, not seventeen trading days or complete portfolio replacements. Weekly targets reset all selected names to equal weights, including retained names. Twenty percent is the five-name rebalance target, not a continuously enforced maximum.",
        "fee_note": "Original report baseline fee is 0.10% per side. This holdings/turnover trace precedes the engine's separate fee adjustment and is unchanged by that scalar fee assumption.",
    }
    day_weights = {}
    for row in omitted:
        day_weights[row["date"]] = day_weights.get(row["date"], 0.0) + row["weight_pct"]
    after = {str(p.relative_to(REPO)): digest(p) for p in inputs}
    assert before == after, "Original artifact modified"
    result = {
        "purpose": "Historical identity-filter sensitivity, not a new optimization or independent validation",
        "change": "Resolve three US ISIN rows before original US-listed filter; all other simulation assumptions unchanged",
        "mapping_basis": "Original PDF snapshot company names paired with their ISIN identifiers; see every omitted source row",
        "rule": "H6e: TIME MOM120, five names, weekly equal targets, rank-10 retention buffer, open/+1 US-session PDF shift, zero cost",
        "snapshot_count": len(snaps), "affected_snapshots": len(day_weights), "omitted_rows": omitted,
        "max_omitted_nav_weight_pct": max(day_weights.values()), "added_prices": price_notes,
        "original": original, "isin_corrected": corrected,
        "same_original_etf_benchmark_total": float(benchmark),
        "original_return_multiple": original["total"] / benchmark,
        "corrected_return_multiple": corrected["total"] / benchmark,
        "total_change_percentage_points": (corrected["total"] - original["total"]) * 100,
        "original_2026_h6e_trade_and_weight_audit": h6e2026,
        "original_inputs_unchanged": before == after, "input_sha256": before,
        "remaining_limitations": [
            "Preserves original synthetic adjusted-open, whole-NAV FX, weekly PDF and close-availability assumptions.",
            "Historical PDF +1-session shift recalculates momentum on delayed date; differs from 2026 feature-row delay.",
            "ETF investment interval follows original comparison and is not exactly the same open-to-open interval.",
            "This repairs only the three verified ISIN exclusions and does not certify all historical identifiers or delistings.",
            "Retrospective diagnostic after both periods were viewed; not proof of prospective performance.",
        ],
    }
    path = HERE / "holdout_isin_audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("original", "isin_corrected", "same_original_etf_benchmark_total",
                                           "corrected_return_multiple", "total_change_percentage_points", "original_inputs_unchanged")},
                     ensure_ascii=False, indent=2))
    print(path)


if __name__ == "__main__":
    main()
