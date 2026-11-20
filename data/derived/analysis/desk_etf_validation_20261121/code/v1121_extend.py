"""Mark every 11/18 closing book through the last completed US session, 2026-11-19 (Thu).

When this study ran (2026-11-20 12:20 New York) the 11/20 session had not closed, so 11/19
is the latest completed session. No decision falls on 11/19: it is not a new ISO week or
month, and the SOXX 100-session state for its auction (11/18 close, or 11/17 for the
one-day-late runs) is asserted unchanged. Each book therefore just holds. The quotes are
the frozen finviz_recipes_20261120/extension_quotes.csv (11/19 close in 11/18 share units,
Yahoo KRW=X), read through that study's own get_forward().
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v1121_ledgers as L  # noqa: E402
from extension import get_forward  # noqa: E402  (finviz_recipes_20261120)
from gates import soxx_closes  # noqa: E402  (finviz_trends_20261120)

MARK = "2026-11-19"


def gate_for_mark(src, lag):
    """SOXX >= its 100-session mean at the close the MARK auction may use."""
    tail = pd.read_csv(L.ANALYSIS / "seven_sleeves_20261120/tail_prices.csv", index_col="symbol")
    close = soxx_closes(src)
    close = pd.concat([close, pd.Series([tail.at["SOXX", "close"]], index=[src.days[-1]])])
    line = close.rolling(100).mean()
    k = len(close) - 1 - lag
    return bool(close.iat[k] >= line.iat[k])


def extend(built):
    src = built.src
    quotes, fx_mark, info = get_forward(src)
    assert pd.Timestamp(MARK).dayofweek == 3 and src.days[-1][:7] == MARK[:7]
    fx_ratio = fx_mark / src.fx[-1]
    for execution, frames in built.ledgers.items():
        lag = 0 if execution == "timely" else 1
        gate = built.gates[lag]
        on = gate_for_mark(src, lag)
        if on != bool(gate[-1]):
            raise ValueError("SOXX gate crossing at the mark session: a decision would be needed")
        for name, frame in frames.items():
            w = frame.attrs["final_weights_10bp"]
            usd = w["CASH"] + sum(part * quotes[src.loc[s]] / src.close[-1, src.loc[s]]
                                  for s, part in w.items() if s != "CASH")
            last = frame.iloc[-1]
            row = {c: last[c] for c in frame.columns}
            for col in ("krw", "krw10", "krw25"):
                if col in frame:
                    row[col] = last[col] * usd * fx_ratio
            row.update(usd=last.usd * usd, usd10=last.usd10 * usd, turn=0.0)
            row["gate_on"] = True if name == "base" else on
            stock = sum(p * quotes[src.loc[s]] / src.close[-1, src.loc[s]] for s, p in w.items() if s != "CASH")
            row["exposure"] = stock / usd
            extended = pd.concat([frame, pd.DataFrame([row], index=pd.Index([MARK], name=frame.index.name))])
            extended.attrs = dict(frame.attrs, mark_return_krw=float(usd * fx_ratio - 1))
            frames[name] = extended
    built.days = [*src.days, MARK]
    built.gate_days = {lag: np.r_[g, gate_for_mark(src, lag)] for lag, g in built.gates.items()}
    built.forward_info = info
    return built


def reproduce_forward_mark(built):
    """Old-rule 11/19 marks must equal finviz_recipes_20261120/forward_mark.csv."""
    assert built.rule == "old120"
    ref = pd.read_csv(L.RECIPES / "forward_mark.csv", index_col="key")
    out = {}
    for mine, theirs in [("base", "base"), ("exit", "exit"), ("switch", "balance/monthly1"),
                         ("original", "original/monthly1")]:
        f = built.ledgers["timely"][mine]
        np.testing.assert_allclose(f.attrs["mark_return_krw"], ref.at[theirs, "mark_return"], atol=1e-10)
        np.testing.assert_allclose(f.krw10.iat[-1] - 1, ref.at[theirs, "marked_ytd"], atol=1e-9)
        out[f"{mine}=forward_mark:{theirs}"] = float(abs(f.krw10.iat[-1] - 1 - ref.at[theirs, "marked_ytd"]))
    return out


if __name__ == "__main__":
    b = extend(L.build("old120"))
    print(reproduce_forward_mark(b))
