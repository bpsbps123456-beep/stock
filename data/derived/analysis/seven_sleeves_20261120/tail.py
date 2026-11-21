"""Freeze the next completed US session after the prior study's input window."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
PREV = HERE.parent/"finviz_rotation_20261119"
sys.path.insert(0, str(PREV))
from add_midcaps import ALIASES


def rebuild():
    source = HERE.parent/"alpaca_sip/fetch_alpaca.py"
    spec = importlib.util.spec_from_file_location("sip_client", source)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    previous = [pd.read_csv(d/"technical_inputs/close.csv", index_col=0).iloc[-1]
                for d in (PREV, PREV/"midcaps")]
    known = pd.concat(previous)
    known.index = [ALIASES.get(s,s) for s in known.index]
    known = known[~known.index.duplicated()]
    names = sorted(set(known.dropna().index) | {"QQQ","SOXX","SPY"})
    rows = {}
    for i in range(0, len(names), 90):
        batch = {s.replace("-","."):s for s in names[i:i+90]}
        params = dict(symbols=",".join(batch), start="2026-11-17T05:00:00Z", end="2026-11-19T04:59:59Z",
                      timeframe="1Day", adjustment="split", asof="-", feed="sip", limit=10000)
        response = requests.get(api.URL, params=params, headers=api.keys(), timeout=35)
        response.raise_for_status()
        data = response.json()
        assert not data.get("next_page_token")
        for ticker, history in data["bars"].items():
            bars = {r["t"][:10]: r for r in history}
            if "2026-11-18" in bars:
                b = bars["2026-11-18"]
                rows[batch[ticker]] = [b["o"],b["c"],bars.get("2026-11-17",{}).get("c")]
    for sym in sorted(set(names)-set(rows) | {"KRW=X"}):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        start = pd.Timestamp("2026-11-16", tz="UTC")
        end = pd.Timestamp("2026-11-19", tz="UTC")
        res = requests.get(url, params={"period1":int(start.timestamp()), "period2":int(end.timestamp()),
                                        "interval":"1d"}, headers={"User-Agent":"Mozilla/5.0"}, timeout=20)
        res.raise_for_status()
        js = res.json()["chart"]["result"][0]
        idx = pd.to_datetime(js["timestamp"], unit="s", utc=True).tz_convert(js["meta"]["exchangeTimezoneName"])
        bars = pd.DataFrame(js["indicators"]["quote"][0], index=idx.strftime("%Y-%m-%d"))
        b = bars.loc["2026-11-18"]
        rows[sym] = [b.open,b.close,bars.loc["2026-11-17","close"]]
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=["open","close","nov17_source_close"])
    frame["nov17_frozen_close"] = known
    gap = (frame.nov17_source_close/frame.nov17_frozen_close-1).abs().dropna()
    if (gap > .03).any():
        raise ValueError(f"Check overlap and share units: {gap[gap > .03].to_dict()}")
    assert set(names)<=set(frame.index) and np.isfinite(frame[["open","close"]]).all().all()
    frame.sort_index().to_csv(HERE/"tail_prices.csv",index_label="symbol")
    print(f"{len(frame)} prices; maximum Nov17 quote-source discrepancy {gap.max():.4%}")


if __name__ == "__main__":
    rebuild()
