"""Independently reconstruct EMA holdings, then compare with the engine's printed report."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PERIODS = {"SEL": ("2026-01-02", "2026-10-14"), "CONF": ("2026-10-15", "2026-11-19")}


def reconstruct(raw, daily):
    session = raw.between_time("09:30", "15:59") if daily else raw.between_time("04:00", "19:59")
    groups = session.groupby(session.index.normalize() if daily else session.index.floor("h"))
    closing = groups.Close.last()
    first = groups.head(1).index
    events = pd.Series(np.nan, index=raw.index)
    ema = closing.iloc[0]
    for price, fill_time in zip(closing.iloc[:-1], first[1:]):
        ema += 2 / 21 * (price - ema)
        if price != ema:
            events.loc[fill_time] = float(price > ema)
    position = events.ffill().fillna(1.0)
    switched = position.diff().fillna(0).abs()
    out = {}
    for name, (start, end) in PERIODS.items():
        prices = raw.Open.loc[start:end]
        change = np.diff(np.log(prices.to_numpy()))
        p = position.loc[start:end].to_numpy()[:-1]
        gain = p * change
        equity = np.exp(gain.cumsum())
        out[name] = {
            "ret%": round(100 * np.expm1(gain.sum()), 2),
            "bh%": round(100 * np.expm1(change.sum()), 2),
            "tv%p": round(100 * (gain.sum() - p.mean() * change.sum()), 2),
            "mdd%": round(100 * (equity / np.maximum.accumulate(equity) - 1).min(), 2),
            "expo%": round(100 * p.mean(), 1),
            "switches": int(switched.loc[start:end].iloc[:-1].sum()),
        }
    return out


def main():
    data = pd.read_parquet(ROOT / "cache/SOXL_1m.parquet").sort_index()
    data = data.loc[~data.index.duplicated()]
    independent = {r: reconstruct(data, r == "R1") for r in ("R1", "R2")}
    report = subprocess.check_output([sys.executable, str(ROOT / "code/engine.py")], text=True)
    result = {}
    for rule, prefix in (("R1", "reg D "), ("R2", "ext 60 ")):
        line = next(line for line in report.splitlines() if line.startswith(prefix))
        reference = ast.literal_eval(line[line.index("{"):].replace("np.float64(", "(").replace("nan", "None"))
        for win, values in independent[rule].items():
            engine = {k: reference[win][k] for k in values}
            delta = {k: round(v - engine[k], 6) for k, v in values.items()}
            result[f"{rule}_{win}"] = dict(independent=values, engine=engine, delta=delta)
    result["match"] = all(all(abs(x) <= 0.05 for x in v["delta"].values()) for v in result.values())
    path = ROOT / "results/codex_engine_check.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(path, result["match"])


if __name__ == "__main__":
    main()
