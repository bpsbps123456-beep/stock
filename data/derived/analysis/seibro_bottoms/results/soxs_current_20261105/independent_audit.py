"""Read-only-input audit artifact, not an execution authority; no current.py imports."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    out = Path(__file__).resolve().parent
    base = out.parents[2]
    raw = out / "raw_snapshot.json"
    frozen = base / "seibro_bottoms/derived/measures_daily.parquet"
    snapshot = json.loads(raw.read_text(encoding="utf-8"))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (raw, frozen)}
    md = pd.read_parquet(frozen).set_index("date").sort_index()
    classes = pd.read_csv(base / "seibro_flows/classes.csv")
    isins = set(classes.loc[classes["product"].eq("SOXS"), "isin"])
    spec = importlib.util.spec_from_file_location("independent_calendar_only", base / "seibro_flows/code/calendar_map.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    nyse, settle = cal.calendars()
    reverse = {}
    for d in nyse:
        if dt.date(2026, 10, 1) <= d <= dt.date(2026, 11, 5):
            reverse.setdefault(cal.seibro_of(d, settle).isoformat(), []).append(d)
    requests = [q for q in snapshot["requests"] if q["kind"] == "daily"]
    new, overlap, unpublished = [], [], []
    for day in sorted({q["start"] for q in requests}):
        qs = [q for q in requests if q["start"] == day]
        assert len(qs) == 4 and sorted(str(q["dtype"]) for q in qs) == ["1", "2", "3", "4"]
        union, rows = {}, []
        for q in qs:
            assert len(q["rows"]) == q["n"]
            for r in (r for r in q["rows"] if r["isin"] in isins):
                v = tuple(float(r[k]) for k in ("buy", "sell", "net", "tot"))
                assert r["isin"] not in union or union[r["isin"]] == v
                union[r["isin"]] = v
                rows.append({"dtype": str(q["dtype"]), **{k: r[k] for k in ("rank", "isin", "buy", "sell", "net", "tot")}})
        if not any(q["n"] for q in qs):
            unpublished.append({"seibro_date": day, "source_if_published": [d.isoformat() for d in reverse.get(day, [])], "action": "not inserted; empty is not zero"})
            continue
        assert len(reverse[day]) == 1
        source = reverse[day][0]
        totals = {k: float(sum(v[i] for v in union.values())) for i, k in enumerate(("buy", "sell", "net", "tot"))}
        entry = {"seibro_date": day, "source_date": source.isoformat(), "request_row_counts": [q["n"] for q in qs], "four_list_rows": rows, "totals_unique_isin": totals, "four_list_amounts_consistent": True}
        if pd.Timestamp(source) in md.index:
            differences = {k: totals[k] - float(md.at[pd.Timestamp(source), "SOXS_" + k]) for k in ("buy", "sell", "net")}
            assert not any(differences.values())
            overlap.append({"seibro_date": day, "source_date": source.isoformat(), "differences": differences})
        else:
            new.append(entry)
    assert len(new) == 1 and new[0]["source_date"] == "2026-11-02" and new[0]["seibro_date"] == "2026-11-04"
    latest = new[0]
    assert len(latest["four_list_rows"]) == 4
    x = md.loc["2023-01-03":, ["SOXS_buy", "SOXS_sell", "SOXS_net", "cens_SOXS", "part_SOXS"]].copy()
    day = pd.Timestamp(latest["source_date"])
    assert day > md.index.max()
    x.loc[day] = [latest["totals_unique_isin"][k] for k in ("buy", "sell", "net")] + [False, False]
    calendar = pd.DatetimeIndex([d for d in nyse if dt.date(2023, 1, 3) <= d <= day.date()])
    x = x.reindex(calendar)
    ok = ~x.cens_SOXS.fillna(True).astype(bool) & ~x.part_SOXS.fillna(True).astype(bool)
    ok &= np.isfinite(x.iloc[:, :3]).all(axis=1) & x.SOXS_buy.ge(0) & x.SOXS_sell.ge(0)
    values = x.iloc[:, :3].where(ok, axis=0).to_numpy(dtype=float)
    signals = {}
    for h in (1, 5):
        sums = np.full_like(values, np.nan)
        for i in range(h - 1, len(values)):
            w = values[i - h + 1:i + 1]
            if np.isfinite(w).all():
                sums[i] = w.sum(axis=0)
        share = np.divide(sums[:, 0], sums[:, 0] + sums[:, 1], out=np.full(len(values), np.nan), where=(sums[:, 0] + sums[:, 1]) > 0)
        for measure, series in (("gross_usd", sums[:, 0]), ("net_usd", sums[:, 2]), ("buy_share", share)):
            v = series[-1]
            clean = series[-250:][np.isfinite(series[-250:])]
            n, less, ties = len(clean), int((clean < v).sum()), int((clean == v).sum())
            p = float(100 * (less + .5 * ties) / n) if n >= 120 and np.isfinite(v) else None
            signals[f"{measure}_L{h}"] = {"value": float(v), "percentile": p, "n_valid": n, "n_less": less, "n_ties": ties, "P80_on": p is not None and p >= 80, "rank_window_start": calendar[-250].date().isoformat(), "rank_window_end": day.date().isoformat()}
    recent = [{"source_date": d.date().isoformat(), "buy": float(r.SOXS_buy), "sell": float(r.SOXS_sell), "net": float(r.SOXS_net)} for d, r in x.tail(5).iterrows()]
    audit = {"scope": "independent audit artifact; no current.py/build/features/pct_rank imports", "method": "2023-01-03 onward NYSE sessions; censored/partial missing, strict five-session sums; share=sum buy/(sum buy+sum sell); current-inclusive last250, min120 valid; 100*(less+0.5*ties)/n", "input_sha256": hashes, "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "latest": latest, "source_map_check": "seibro_of(2026-11-02)=2026-11-04", "known_settle_convention": "2026-11-05 09:00 KST, not a historical publication log", "US_entry_convention": "2026-11-05 regular open; extra-delay case 2026-11-06", "unpublished_not_inserted": unpublished, "overlap_frozen_exact_matches": overlap, "recent_five_source_sessions": recent, "signals": signals, "amount_note": "Reported net retained: five-day buy-sell differs from summed net by USD1; reported total differs from buy+sell by USD1. No split scaling."}
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (raw, frozen)}
    (out / "independent_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"signals": signals, "overlap_exact_matches": len(overlap)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
