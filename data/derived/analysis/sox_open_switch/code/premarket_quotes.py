"""BRIEF8: 신호일 SOXS 실제 직전 NBBO로 프리장 체결 근사를 감사한다.

py data/derived/analysis/sox_open_switch/code/premarket_quotes.py [--trades CSV] [--offline]
원호가는 분할 보정하지 않는다. 같은 날 진입 ask와 청산 bid의 비율로 손익을 재며,
분할 반영 분봉과 원호가 mid의 비율이 진입/청산 사이 1% 이상 변하면 표시하고,
유효 원호가 전체와 표시한 행을 뺀 민감도를 함께 낸다.
호가대로 전량 체결된다는 보장은 없고, 추가 시장가 슬리피지는 포함하지 않는다.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ET = "America/New_York"
URL = "https://data.alpaca.markets/v2/stocks/{sym}/quotes"
MAX_AGE_SECONDS = 30.0
MAX_RATIO_CHANGE = 0.01


def aware_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("호가 시각에는 시간대가 필요하다")
    return stamp.tz_convert("UTC")


def quote_record(sym: str, target: str | pd.Timestamp, quote: dict | None) -> dict:
    """마지막 호가 하나를 검증한다. 앞선 유효 호가로 바꾸어 고르지 않는다."""
    t = aware_timestamp(target)
    row = {"sym": sym, "target": t.isoformat(), "quote_timestamp": None,
           "bid": np.nan, "ask": np.nan, "bid_size": np.nan, "ask_size": np.nan,
           "age_seconds": np.nan, "valid": False, "reason": "missing_quote"}
    if not quote:
        return row
    try:
        qt = aware_timestamp(quote["t"])
        row.update(quote_timestamp=qt.isoformat(), bid=float(quote["bp"]), ask=float(quote["ap"]),
                   bid_size=float(quote["bs"]), ask_size=float(quote["as"]),
                   age_seconds=float((t - qt).total_seconds()))
    except (KeyError, TypeError, ValueError):
        row["reason"] = "malformed_quote"
        return row
    numeric = [row[x] for x in ("bid", "ask", "bid_size", "ask_size", "age_seconds")]
    if not all(np.isfinite(numeric)):
        row["reason"] = "nonfinite_quote"
    elif row["age_seconds"] < 0:
        row["reason"] = "future_quote"
    elif row["age_seconds"] > MAX_AGE_SECONDS:
        row["reason"] = "stale_quote"
    elif row["bid"] <= 0 or row["ask"] <= 0:
        row["reason"] = "nonpositive_price"
    elif row["bid"] > row["ask"]:
        row["reason"] = "crossed_quote"
    elif row["bid_size"] <= 0 or row["ask_size"] <= 0:
        row["reason"] = "empty_size"
    else:
        row.update(valid=True, reason="ok")
    return row


def last_quote(sym: str, target: str | pd.Timestamp, headers: dict, pace) -> dict:
    t = aware_timestamp(target)
    params = {"start": (t - pd.Timedelta(seconds=120)).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "end": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "sort": "desc", "limit": 1, "feed": "sip"}
    for attempt in range(6):
        pace.wait()
        try:
            response = requests.get(URL.format(sym=sym), params=params, headers=headers, timeout=30)
        except requests.RequestException:
            if attempt < 5:
                time.sleep(min(2 + attempt, 7))
            continue
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < 5:
                time.sleep(min(2 + attempt, 7))
            continue
        response.raise_for_status()
        quotes = response.json().get("quotes") or []
        return quote_record(sym, t, quotes[0] if quotes else None)
    row = quote_record(sym, t, None)
    row["reason"] = "request_failed"
    return row


def audit_trade(entry: dict, exit_: dict, adjusted_entry: float, adjusted_exit: float) -> dict:
    """원호가 비율 손익과 같은 날 분할 단위 일관성을 함께 검사한다."""
    out = {"quote_valid": False, "quote_reason": None, "quote_return_pct": np.nan,
           "quote_return_cost2_pct": np.nan, "entry_mid_raw_to_adjusted": np.nan,
           "exit_mid_raw_to_adjusted": np.nan, "ratio_change_pct": np.nan,
           "ratio_flag": False, "ratio_reason": "ok"}
    for name, quote in (("entry", entry), ("exit", exit_)):
        if not quote.get("valid", False):
            out["quote_reason"] = f"{name}:{quote.get('reason', 'missing_quote')}"
            return out
    et = aware_timestamp(entry["target"]).tz_convert(ET)
    xt = aware_timestamp(exit_["target"]).tz_convert(ET)
    if et.date() != xt.date() or xt <= et:
        out["quote_reason"] = "not_same_day_ordered_trade"
        return out
    em = (entry["bid"] + entry["ask"]) / 2
    xm = (exit_["bid"] + exit_["ask"]) / 2
    eh = (entry["ask"] - entry["bid"]) / 2
    xh = (exit_["ask"] - exit_["bid"]) / 2
    if not all(np.isfinite([adjusted_entry, adjusted_exit])) or min(adjusted_entry, adjusted_exit) <= 0:
        out.update(ratio_flag=True, ratio_reason="invalid_adjusted_bar")
    else:
        er, xr = em / adjusted_entry, xm / adjusted_exit
        change = xr / er - 1
        out.update(entry_mid_raw_to_adjusted=er, exit_mid_raw_to_adjusted=xr,
                   ratio_change_pct=change * 100)
        if abs(change) >= MAX_RATIO_CHANGE:
            out.update(ratio_flag=True, ratio_reason="raw_adjusted_ratio_changed")
    out.update(quote_valid=True, quote_reason="ok",
               quote_return_pct=(exit_["bid"] / entry["ask"] - 1) * 100,
               quote_return_cost2_pct=(max(0.0, xm - 2 * xh) / (em + 2 * eh) - 1) * 100,
               cost2_exit_clipped=(xm - 2 * xh < 0))
    return out


def segment_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    days = frame["day"].astype(str)
    return {"2026_all": days.between("2026-01-01", "2026-11-03"),
            "2026_H1": days.between("2026-01-01", "2026-06-30"),
            "2026_H2": days.between("2026-07-01", "2026-11-03"),
            "2026_JulAug": days.between("2026-07-01", "2026-08-31"),
            "2026_SepNov": days.between("2026-09-01", "2026-11-03"),
            "2025_aux": days.between("2025-01-01", "2025-12-31")}


def summarize(frame: pd.DataFrame) -> dict:
    rows = []
    for strategy, group in frame.groupby("strategy_id", sort=True):
        for segment, mask in segment_masks(group).items():
            all_ = group[mask]
            good = all_[all_["quote_valid"].eq(True)]  # noqa: E712
            if all_.empty:
                continue
            ret = good["quote_return_pct"]
            ratio_clean = good[~good["ratio_flag"].eq(True)]  # noqa: E712
            item = {"strategy_id": strategy, "segment": segment,
                    "n_input_signal_trades": len(all_), "n_valid": len(good),
                    "n_excluded": len(all_) - len(good),
                    "n_ratio_flagged_valid": int(good["ratio_flag"].sum()),
                    "n_without_ratio_flags": len(ratio_clean),
                    "mean_trade_return_without_ratio_flags_pct": float(ratio_clean["quote_return_pct"].mean()) if len(ratio_clean) else None,
                    "excluded_reasons": all_.loc[~all_["quote_valid"].eq(True), "quote_reason"].value_counts().to_dict(),  # noqa: E712
                    "mean_trade_return_pct": float(ret.mean()) if len(good) else None,
                    "median_trade_return_pct": float(ret.median()) if len(good) else None,
                    "win_rate": float((ret > 0).mean()) if len(good) else None,
                    "mean_trade_return_cost2_pct": float(good["quote_return_cost2_pct"].mean()) if len(good) else None,
                    "mean_model_return_on_same_trades_pct": float(good["return_pct"].mean()) if len(good) else None,
                    "drop_best_2_trade_mean_pct": float(ret.sort_values().iloc[:-2].mean()) if len(good) > 2 else None,
                    "worst_trade_return_pct": float(ret.min()) if len(good) else None}
            rows.append(item)
    return {"scope": "actual last NBBO, entry ask -> exit bid; signal trades only, not daily/account returns",
            "max_quote_age_seconds": MAX_AGE_SECONDS,
            "raw_to_adjusted_ratio_change_flag": MAX_RATIO_CHANGE,
            "slippage": "additional market impact/slippage excluded; displayed size is not a guaranteed fill",
            "cost2": "entry mid + two half-spreads, exit mid - two half-spreads (exit floored at zero)",
            "partial_coverage": "missing/invalid quotes are excluded, not filled with zero; compare matched model rows",
            "rows": rows}


def daily_quote_statistics(daily: pd.DataFrame, quote_trades: pd.DataFrame,
                           expected_counts: dict | None = None) -> dict:
    """Join complete sessions to valid quote trades; absent quote rows must fail.

    Non-signal sessions are zero. A missing/invalid quote on a signal session is
    an error, rather than an implicit zero or a changed account denominator.
    Bootstrap, drawdown and Holm calculation reuse the frozen experiment module.
    """
    spec = importlib.util.spec_from_file_location("premarket_fade_quote_statistics", HERE / "premarket_fade.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    key = ["strategy_id", "day"]
    daily_required = set(key + ["signal"])
    quote_required = set(key + ["quote_valid", "quote_return_pct", "quote_return_cost2_pct", "ratio_flag"])
    if not daily_required.issubset(daily) or not quote_required.issubset(quote_trades):
        raise ValueError("daily/quote trade fields missing")
    d, q = daily.copy(), quote_trades.copy()
    for frame in (d, q):
        frame["day"] = pd.to_datetime(frame["day"]).dt.strftime("%Y-%m-%d")
        if frame.duplicated(key).any():
            raise ValueError("duplicate strategy_id + day")
    for frame, name in ((d, "signal"), (q, "quote_valid"), (q, "ratio_flag")):
        if frame[name].isna().any() or not frame[name].isin([True, False]).all():
            raise ValueError(f"{name} must be complete boolean values")
    signal_keys = set(d.loc[d.signal.eq(True), key].itertuples(index=False, name=None))  # noqa: E712
    quote_keys = set(q[key].itertuples(index=False, name=None))
    if signal_keys != quote_keys:
        raise ValueError(f"quote coverage mismatch: missing={len(signal_keys - quote_keys)}, extra={len(quote_keys - signal_keys)}")
    if not q.quote_valid.eq(True).all():  # noqa: E712
        raise ValueError("invalid quote on signal session; daily account series unavailable")
    return_columns = ["quote_return_pct", "quote_return_cost2_pct"]
    if not np.isfinite(q[return_columns].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite quote return on signal session")
    joined = d.merge(q[key + ["quote_valid", "ratio_flag"] + return_columns], on=key,
                     how="left", validate="one_to_one", indicator=True)
    if (joined.loc[joined.signal.eq(True), "_merge"] != "both").any():  # noqa: E712
        raise ValueError("signal quote missing after join")
    joined[return_columns] = joined[return_columns].fillna(0.0)
    periods = {"2026_all": ("2026-01-02", "2026-11-03"), "2026_front": ("2026-01-02", "2026-06-30"),
               "2026_back": ("2026-07-01", "2026-11-03"), "2026_jul_aug": ("2026-07-01", "2026-08-31"),
               "2026_sep_nov": ("2026-09-01", "2026-11-03"), "2025_auxiliary": ("2025-01-02", "2025-12-31")}
    rows = []
    for strategy, group in joined.groupby("strategy_id", sort=True):
        group = group.sort_values("day")
        row = {"strategy_id": strategy, "periods": {}}
        for period, (start, end) in periods.items():
            frame = group[group.day.between(start, end)]
            if expected_counts is not None and len(frame) != expected_counts[period]:
                raise ValueError(f"{strategy} {period}: expected {expected_counts[period]} sessions, got {len(frame)}")
            if frame.empty:
                continue
            r = frame.quote_return_pct.to_numpy(dtype=float)
            c2 = frame.quote_return_cost2_pct.to_numpy(dtype=float)
            sig = frame.signal.to_numpy(dtype=bool)
            tr = r[sig]
            ci, tail = common.bootstrap(r)
            top = np.argsort(-r, kind="stable")[:min(2, len(r))]
            keep = np.ones(len(r), dtype=bool)
            keep[top] = False
            row["periods"][period] = {
                "days": len(r), "signals": int(sig.sum()), "n_valid_quote_trades": len(tr),
                "n_ratio_flagged_valid": int(frame.loc[sig, "ratio_flag"].eq(True).sum()),  # noqa: E712
                "per_day_pct": float(r.mean()), "per_trade_pct": float(tr.mean()) if len(tr) else None,
                "median_trade_pct": float(np.median(tr)) if len(tr) else None,
                "win_rate": float((tr > 0).mean()) if len(tr) else None,
                "worst_trade_pct": float(tr.min()) if len(tr) else None,
                "ci95_per_day_pct": ci, "bootstrap_tail_le_zero_reference": tail,
                "cost2_per_day_pct": float(c2.mean()),
                "cost2_per_trade_pct": float(c2[sig].mean()) if len(tr) else None,
                "drop_top2_per_day_pct": float(r[keep].mean()) if keep.any() else None,
                "removed_top2": [{"day": str(frame.iloc[i].day), "return_pct": float(r[i])} for i in top],
                "compound_pct": float((np.prod(1 + r / 100) - 1) * 100),
                "max_dd_pct": common.max_dd(r)}
        rows.append(row)
    back = [r for r in rows if "2026_back" in r["periods"]]
    adj = common.holm([r["periods"]["2026_back"]["bootstrap_tail_le_zero_reference"] for r in back])
    for row, value in zip(back, adj):
        row["periods"]["2026_back"]["holm6_bootstrap_tail_reference"] = value
    return {"scope": "complete regular sessions, non-signal zero; signal entry raw ask -> exit raw bid",
            "quote_coverage": "every signal key has a valid finite actual quote return; otherwise calculation fails",
            "n_joined_signal_trades": len(signal_keys), "n_strategies": len(rows),
            "statistics_reuse": "premarket_fade.bootstrap / max_dd / holm", "seed": common.SEED,
            "bootstrap_draws": common.NBOOT, "units": "percentage points (1.0 = 1%)",
            "reference_only": "2026 reused history, bootstrap tails/Holm are not selection-corrected confirmation",
            "slippage": "extra market impact excluded; NBBO displayed size does not guarantee full execution",
            "rows": rows}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, default=ROOT / "results/premarket_fade_trades.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/premarket_fade_quotes.parquet")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "results/premarket_quote")
    parser.add_argument("--daily", type=Path, default=ROOT / "results/premarket_fade_daily.csv",
                        help="complete daily sessions matching every input trade, for account statistics")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    trades = pd.read_csv(args.trades)
    required = {"strategy_id", "day", "sym", "entry_timestamp", "exit_timestamp", "entry_px", "exit_px", "return_pct"}
    if not required.issubset(trades):
        raise ValueError(f"거래 CSV 필수 칼럼 누락: {sorted(required - set(trades))}")
    if trades.duplicated(["strategy_id", "day"]).any():
        raise ValueError("strategy_id + day 거래 키 중복")
    want = sorted({(r.sym, aware_timestamp(getattr(r, field)).isoformat())
                   for r in trades.itertuples() for field in ("entry_timestamp", "exit_timestamp")})
    old = pd.read_parquet(args.cache) if args.cache.exists() else pd.DataFrame()
    records = old.to_dict("records")
    have = {(r["sym"], aware_timestamp(r["target"]).isoformat()) for r in records}
    missing = [x for x in want if x not in have]
    if missing and not args.offline:
        sys.path.insert(0, str(ROOT.parent / "hatta_bar/code"))
        from fetch_alpaca import keys, Pace  # noqa: E402
        headers, pace = keys(), Pace(3.0)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"SOXS actual quotes: {len(missing)} requests", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for i, record in enumerate(executor.map(lambda x: last_quote(x[0], x[1], headers, pace), missing), 1):
                records.append(record)
                if i % 50 == 0:
                    pd.DataFrame(records).to_parquet(args.cache, index=False)
                    print(f"quotes {i}/{len(missing)}", flush=True)
        pd.DataFrame(records).to_parquet(args.cache, index=False)
    lookup = {(r["sym"], aware_timestamp(r["target"]).isoformat()): r for r in records}
    out = []
    for trade in trades.to_dict("records"):
        er = lookup.get((trade["sym"], aware_timestamp(trade["entry_timestamp"]).isoformat()))
        xr = lookup.get((trade["sym"], aware_timestamp(trade["exit_timestamp"]).isoformat()))
        er = er or quote_record(trade["sym"], trade["entry_timestamp"], None)
        xr = xr or quote_record(trade["sym"], trade["exit_timestamp"], None)
        adjusted_entry = float(trade.get("entry_bar_close", trade["entry_px"]))
        adjusted_exit = float(trade.get("exit_bar_close", trade["exit_px"]))
        audit = audit_trade(er, xr, adjusted_entry, adjusted_exit)
        row = dict(trade, **audit)
        for prefix, record in (("entry", er), ("exit", xr)):
            for field in ("quote_timestamp", "bid", "ask", "bid_size", "ask_size", "age_seconds", "reason"):
                row[f"{prefix}_{field}"] = record.get(field)
        out.append(row)
    result = pd.DataFrame(out)
    if result.empty:
        result = trades.assign(quote_valid=pd.Series(dtype=bool), quote_reason=pd.Series(dtype=str),
                               quote_return_pct=pd.Series(dtype=float), quote_return_cost2_pct=pd.Series(dtype=float),
                               ratio_flag=pd.Series(dtype=bool))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(str(args.output_prefix) + "_trades.csv", index=False)
    summary = summarize(result)
    summary.update(source_trades=str(args.trades), quote_cache=str(args.cache), n_requested_unique=len(want),
                   n_cached=len(set(lookup) & set(want)), n_missing_requests=len(missing) if args.offline else 0)
    Path(str(args.output_prefix) + "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    daily_stats = daily_quote_statistics(pd.read_csv(args.daily), result,
                                        expected_counts={"2026_all": 211, "2026_front": 123, "2026_back": 88,
                                                         "2026_jul_aug": 43, "2026_sep_nov": 45, "2025_auxiliary": 250})
    daily_stats.update(source_daily=str(args.daily), source_quote_trades=str(args.output_prefix) + "_trades.csv")
    Path(str(args.output_prefix) + "_daily_stats.json").write_text(
        json.dumps(daily_stats, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"n_signal_trades": len(result), "n_valid": int(result.quote_valid.sum()),
                      "summary": str(args.output_prefix) + "_summary.json"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
