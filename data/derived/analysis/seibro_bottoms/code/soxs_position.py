"""Describe SOXX price locations at the existing SOXS gross-buy L5 P80 signal.

Ranks and drawdowns come from the canonical ledgers/price helper. No return,
threshold, or strategy search is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
DEFAULT = STUDY / "results/soxs_position_20261106"
spec = importlib.util.spec_from_file_location("position_ledger", HERE / "soxs_ledger.py")
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)
KEY = "gross_usd_L5"
THRESHOLD = 80


def price_locations(pf):
    """Current-inclusive 60-session range; no future extrema or gap filling."""
    result = pf.copy()
    close = result.SOXX_Close
    minimum = close.rolling(60, min_periods=60).min()
    maximum = close.rolling(60, min_periods=60).max()
    span = maximum - minimum
    result["range60"] = 100 * (close - minimum) / span.where(span > 0)
    return result


def preceding_location(entry, sessions, pf):
    """Require the exact preceding exchange session, including missing prices."""
    entry = pd.Timestamp(entry)
    pos = sessions.get_indexer([entry])[0]
    if pos <= 0:
        return {"previous_price_date": None, "close": None, "dd60": None, "range60": None}
    previous = sessions[pos - 1]
    if previous not in pf.index:
        return {"previous_price_date": None, "close": None, "dd60": None, "range60": None}
    row = pf.loc[previous]
    return {"previous_price_date": previous.strftime("%Y-%m-%d"),
            "close": row.SOXX_Close, "dd60": row.dd60 * 100, "range60": row.range60}


def summarize(rows, state):
    valid = [r for r in rows if r["status"] == state and pd.notna(r["dd60"])]
    dd = pd.Series([r["dd60"] for r in valid], dtype=float)
    ranges = pd.Series([r["range60"] for r in valid], dtype=float)
    # Exact boundaries belong to the deeper bin, as in the independent audit.
    masks = [dd > -5, (dd <= -5) & (dd > -10),
             (dd <= -10) & (dd > -20), dd <= -20]
    bins = [{"label": label, "n": int(mask.sum()),
             "pct": float(mask.mean() * 100) if len(dd) else None}
            for label, mask in zip(("낙폭 0~5% 미만", "낙폭 5~10% 미만", "낙폭 10~20% 미만", "낙폭 20% 이상"), masks)]
    return {"n": len(dd), "missing_price_n": sum(r["status"] == state for r in rows) - len(dd),
            "median_dd60": dd.median(), "q25_dd60": dd.quantile(.25),
            "q75_dd60": dd.quantile(.75), "min_dd60": dd.min(), "max_dd60": dd.max(),
            "median_range60": ranges.median(), "near_high_n": bins[0]["n"],
            "below10_n": int((dd <= -10).sum()), "below20_n": int((dd <= -20).sum()),
            "bins": bins}


def check_same(actual, expected, name):
    if not np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                       atol=1e-8, rtol=1e-10, equal_nan=True):
        raise ValueError(f"Canonical price mismatch: {name}")


def assemble(source, entry, pf, sessions):
    pf = price_locations(pf)
    source_rows, entry_rows = [], []
    for _, r in source.iterrows():
        date = pd.Timestamp(r.source_date)
        price = pf.loc[date]
        check_same([r.SOXX_Close, r.dd60_pct], [price.SOXX_Close, price.dd60 * 100], str(date))
        previous = preceding_location(r.entry_settle, sessions, pf)
        check_same([r.entry_previous_close, r.entry_previous_dd60_pct],
                   [previous["close"], previous["dd60"]], str(date) + " entry")
        source_rows.append({"date": date.strftime("%Y-%m-%d"), "close": price.SOXX_Close,
            "dd60": price.dd60 * 100, "range60": price.range60,
            "p5": r[KEY + "_P"], "usd5": r[KEY], "status": r[KEY + "_status"],
            "on": r[KEY + "_status"] == "ON", "observed": bool(r.source_observed),
            "entry_date": str(r.entry_settle)[:10], "entry_dd60": previous["dd60"],
            "entry_range60": previous["range60"],
            "selected_for_entry": bool(r.selected_for_entry), "shared": bool(r.unit_shared_source)})
    for _, r in entry.iterrows():
        previous = preceding_location(r.entry, sessions, pf)
        check_same([r.previous_close, r.previous_dd60_pct],
                   [previous["close"], previous["dd60"]], str(r.entry))
        entry_rows.append({"date": str(r.entry)[:10],
            "source_date": None if pd.isna(r.source_date) else str(r.source_date)[:10],
            **previous, "p5": r[KEY + "_P"], "usd5": r[KEY],
            "status": r[KEY + "_status"], "on": r[KEY + "_status"] == "ON"})
    prices = [{"date": d.strftime("%Y-%m-%d"), "close": r.SOXX_Close,
               "dd60": r.dd60 * 100, "range60": r.range60}
              for d, r in pf.loc["2024-01-01":].iterrows()]
    summary = {}
    for year in (2024, 2025, 2026, "all"):
        summary[str(year)] = {}
        for name, rows in (("source", source_rows), ("entry", entry_rows)):
            selected = rows if year == "all" else [r for r in rows if r["date"].startswith(str(year))]
            summary[str(year)][name] = {state: summarize(selected, state) for state in ("ON", "OFF")}
    latest_source = next((r for r in reversed(source_rows) if r["observed"]), None)
    latest_entry = next((r for r in reversed(entry_rows)
                         if latest_source and r["source_date"] == latest_source["date"]), None)
    return ledger.clean({"as_of": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "latest_source_date": latest_source["date"] if latest_source else None,
        "latest_price_date": prices[-1]["date"], "years": [2024, 2025, 2026],
        "price_basis_note": "2026-11-05 SOXX 3:1 분할 후 기준. 새 전체 가격 응답에서 혼재한 11/2~4 OHLC를 공식 비율로 교정했습니다. 11/5 종가 $131.90은 Yahoo 제공값이며 발행사 페이지는 수집 당시 11/4까지였습니다.",
        "threshold": THRESHOLD, "new_settings_searched": 0,
        "rank_basis": "current-inclusive 250 NYSE sessions, minimum 120 valid, midpoint ties",
        "range60_definition": "100*(close - past/current 60-session minimum close)/(maximum-minimum); no future prices",
        "source_rows": source_rows, "entry_rows": entry_rows, "prices": prices,
        "summary": summary, "latest": {"source": latest_source, "entry": latest_entry, "price": prices[-1]}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT / "fresh/ledger/source_daily.csv")
    parser.add_argument("--entry", type=Path, default=DEFAULT / "fresh/ledger/entry_daily.csv")
    parser.add_argument("--prices", type=Path, default=DEFAULT / "fresh/market.prices.normalized.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args(argv)
    sessions = ledger.build.nyse_sessions()
    pf, _ = ledger.load_price_panel(args.prices, sessions)
    data = assemble(pd.read_csv(args.source), pd.read_csv(args.entry), pf, sessions)
    data["inputs"] = {str(p.relative_to(STUDY)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (args.source, args.entry, args.prices)}
    args.output.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    (args.output / "position.json").write_text(serialized + "\n", encoding="utf-8")
    for key, filename in (("source_rows", "source_position.csv"), ("entry_rows", "entry_position.csv")):
        pd.DataFrame(data[key]).to_csv(args.output / filename, index=False, encoding="utf-8-sig")
    template = (HERE / "soxs_position_template.html").read_text(encoding="utf-8")
    html = template.replace("__SOXS_POSITION_JSON__", serialized.replace("<", "\\u003c"))
    (args.output / "position.html").write_text(html, encoding="utf-8")
    print(json.dumps({"latest": data["latest"], "bytes": len(html.encode("utf-8"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
