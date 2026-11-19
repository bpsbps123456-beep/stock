"""Codex's remaining candidate: the weekly 7-stock rule plus a SOXX 100-day exit (weekly_SOXX100r1).

Trade-by-trade ledger of every cash spell (avoided loss / missed gain, base USD between the exit open and
the re-entry open), fill-delay variants, the exposure-matched control, a placebo that places the same
spell pattern at every other circular offset, the current state of the signal, and a 2025 auxiliary run
on the 2025 desk-like book (season_regime_20261119/aux2025.py) and on SOXX itself.
Run: py data/derived/analysis/desk_defense_check_20261119/codex_soxx100.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import common
import study

SEASON = common.HERE.parent / "season_regime_20261119"
WINDOWS = {**common.WINDOWS, "hell1": ("2026-06-23", "2026-07-29"), "rebound": ("2026-07-30", "2026-09-22"),
           "hell2": ("2026-10-13", "2026-11-17")}


def lagged(gate, n):
    return np.r_[np.repeat(gate[0], n), gate[:-n]] if n else gate


def stats(curve):
    out = {}
    for w, (a, b) in WINDOWS.items():
        s = common.window_stats(curve, a, b)
        out.update({f"{w}_usd": s["usd"], f"{w}_usd_mdd": s["usd_mdd"], f"{w}_krw": s["krw"],
                    f"{w}_krw_mdd": s["krw_mdd"], f"{w}_expo": s["expo"]})
    return out


def spells(days, gate, baseline):
    """Cash spells: base USD from the exit open to the re-entry open (or the last close if still out)."""
    rows, start = [], None
    for j, invested in enumerate(gate):
        if not invested and start is None:
            start = j
        if start is not None and (invested or j == len(days) - 1):
            end_value = baseline.value[j] if invested else baseline.close[j]
            rows.append({"exit_open": days[start], "reentry_open": days[j] if invested else "still out",
                         "sessions_out": j - start + int(not invested),
                         "base_usd_while_out": float(end_value / baseline.value[start] - 1)})
            start = None
    return pd.DataFrame(rows)


def main():
    ctx = common.context()
    days, base_gate = ctx.days, ctx.gates["SOXX100r1"]
    args = (days, ctx.opn, ctx.close, ctx.fx)
    base = study.replay(*args, ctx.weekly, ctx.gates["none"])
    curves = {"base": base, "SOXX100": study.replay(*args, ctx.weekly, base_gate)}
    for n in (1, 2):
        curves[f"SOXX100_late{n}"] = study.replay(*args, ctx.weekly, lagged(base_gate, n))
    table = pd.DataFrame({k: stats(v) for k, v in curves.items()}).T
    ledger = spells(days, base_gate, ctx.baseline)
    ledger["effect"] = np.where(ledger.base_usd_while_out < 0, "avoided", "missed")

    me = table.loc["SOXX100"]
    control = {w: common.matched_control(base, me[f"{w}_expo"], w) for w in ("full", "h2", "sep_nov")}
    T = len(days)
    shifts = range(10, T - 9)
    placebo = np.array([[common.window_stats(study.replay(*args, ctx.weekly, np.roll(base_gate, s)), a, b)[m]
                         for (a, b), m in ((WINDOWS["full"], "usd"), (WINDOWS["h2"], "usd"),
                                           (WINDOWS["sep_nov"], "usd"), (WINDOWS["full"], "usd_mdd"))]
                        for s in shifts])
    pct = {name: float((placebo[:, i] < me[col]).mean())
           for i, (name, col) in enumerate((("full", "full_usd"), ("h2", "h2_usd"), ("sep_nov", "sep_nov_usd"),
                                             ("full_mdd", "full_usd_mdd")))}

    soxx = ctx.book.cls["SOXX"]
    sma100 = soxx.rolling(100).mean()
    last = days[-1]
    state = {"last_close_date": last, "soxx_close": float(soxx[last]), "soxx_sma100": float(sma100[last]),
             "gap_to_sma100": float(soxx[last] / sma100[last] - 1),
             "gate_for_next_session": "invested" if soxx[last] >= sma100[last] else "cash",
             "base_usd_since_last_exit_open": float(ledger.base_usd_while_out.iloc[-1])}

    # 2025 auxiliary: the same gate on the 2025 desk-like book, and on SOXX itself.
    sys.path.insert(0, str(SEASON))
    import aux2025  # noqa: E402
    aux = aux2025.build()
    mom25 = aux2025.stats(aux.ledger(aux.plans["MOM"]))
    gate25 = aux2025.stats(aux.ledger(aux.plans["MOM"], aux.gates["SOXX100r1"]))
    etf = pd.read_csv(common.HERE / "frozen_etf" / "SOXX.csv", index_col=0)
    o, c = etf.Open.loc[aux.days].to_numpy(float), etf.Close.loc[aux.days].to_numpy(float)
    hold = study.replay(aux.days, o[:, None], c[:, None], np.ones(len(aux.days)), {0: np.ones(1)},
                        np.ones(len(aux.days), dtype=bool))
    gated = study.replay(aux.days, o[:, None], c[:, None], np.ones(len(aux.days)), {0: np.ones(1)},
                         aux.gates["SOXX100r1"])
    aux_rows = {"desk_like_MOM_2025": mom25, "desk_like_MOM_SOXX100_2025": gate25,
                "SOXX_hold_2025": aux2025.stats(hold), "SOXX_SOXX100_2025": aux2025.stats(gated)}
    # 2025 spells, approximated close-to-close on the ungated 2025 book (close before exit -> close before re-entry).
    usd25 = aux.ledger(aux.plans["MOM"]).usd.to_numpy()
    g25, rows25, start = aux.gates["SOXX100r1"], [], None
    for j, invested in enumerate(g25):
        if not invested and start is None:
            start = j
        if start is not None and (invested or j == len(g25) - 1):
            a = usd25[start - 1] if start else 1.0
            b = usd25[j - 1] if invested else usd25[j]
            rows25.append({"exit_open": aux.days[start], "reentry_open": aux.days[j] if invested else "still out",
                           "base_usd_while_out": float(b / a - 1)})
            start = None
    aux_rows["spells_2025"] = rows25

    out = {"table": table.round(6).to_dict(orient="index"), "spells": ledger.to_dict(orient="records"),
           "matched_control": {w: {"usd": v[0], "usd_mdd": v[1]} for w, v in control.items()},
           "placebo": {"draws": len(placebo), "percentile": pct,
                       "median_full_usd": float(np.median(placebo[:, 0])), "median_h2_usd": float(np.median(placebo[:, 1]))},
           "state": state, "aux2025": aux_rows}
    (common.OUT / "codex_soxx100.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float),
                                                   encoding="utf-8")
    pd.set_option("display.width", 250, "display.float_format", lambda v: f"{v:+.4f}")
    cols = ["full_usd", "full_usd_mdd", "full_krw", "h1_usd", "h2_usd", "h2_krw", "hell1_usd", "rebound_usd",
            "hell2_usd", "sep_nov_usd", "h2_expo"]
    print(table[cols].to_string())
    print("\nspells\n", ledger.to_string(index=False))
    print("\nmatched control (same mean exposure of base, every day):",
          {w: tuple(round(x, 4) for x in v) for w, v in control.items()})
    print("placebo percentiles (circular offsets of the same gate):", {k: round(v, 3) for k, v in pct.items()},
          "n =", len(placebo))
    print("state:", state)
    for k, d in aux_rows.items():
        print("2025", k, d if isinstance(d, list) else {m: round(v, 4) for m, v in d.items()})


if __name__ == "__main__":
    main()
