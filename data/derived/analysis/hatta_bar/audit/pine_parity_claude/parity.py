"""Run the literal Pine translation over every symbol and compare with the engine dump.

No outcome / performance numbers are computed here (blinded study): only signal sets.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pine_literal as P

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
OUT = Path(__file__).resolve().parent

CELLS = {
    "orig_A": dict(),
    "orig_B": dict(),
    "alt_A1": dict(nBars=60, tolPct=0.0, pSw=2, kModeA=P.M_TOUCH, kMinA=3, trendMd=P.T_NOW, dropPct=0.0,
                   recMd=P.R_CPOS, recMin=0.65, rngMin=0.0, needAb=True, volMin=2.0),
    "alt_A2": dict(nBars=390, tolPct=0.1, pSw=5, kModeA=P.M_LEVEL, kMinA=3, trendMd=P.T_OFF, dropPct=3.0,
                   recMd=P.R_WICK, recMin=0.65, rngMin=2.0, needAb=False, volMin=0.0),
    "alt_B1": dict(nBars=240, tolPct=0.1, pSw=2, kModeB=P.M_LEVEL, kMinB=2, trendMd=P.T_NOW, dropPct=1.0,
                   volMin=2.0, needBl=False, engMd=P.E_FULL),
    "alt_B2": dict(nBars=60, tolPct=0.0, pSw=5, kModeB=P.M_TOUCH, kMinB=2, trendMd=P.T_OFF, dropPct=0.0,
                   volMin=0.0, needBl=False, engMd=P.E_RECL),
}
PAT = {"orig_A": "A", "orig_B": "B", "alt_A1": "A", "alt_A2": "A", "alt_B1": "B", "alt_B2": "B"}
ORDER = ["nBars", "tolPct", "pSw", "kModeA", "kMinA", "kModeB", "kMinB", "trendMd", "dropPct", "showA",
         "recMd", "recMin", "rngMin", "needAb", "showB", "needBl", "engMd", "volMin", "minNot", "minRng",
         "minDens", "noNight"]


def load(sym):
    d = pd.read_parquet(CACHE / "yf1m" / f"{sym}.parquet")
    idx = d.index
    t = (idx.tz_convert("UTC").asi8 // 10**6 if idx.asi8.dtype == np.int64 else None)
    # asi8 unit depends on resolution; normalise to epoch ms
    t = idx.tz_convert("UTC").as_unit("ms").asi8.astype(np.int64)
    ny = idx.tz_convert("America/New_York")
    return (d["Open"].to_numpy(float), d["High"].to_numpy(float), d["Low"].to_numpy(float),
            d["Close"].to_numpy(float), d["Volume"].to_numpy(float), t,
            ny.day.to_numpy(np.int64), ny.hour.to_numpy(np.int64), ny.strftime("%Y-%m-%d %H:%M").to_numpy())


def run_all(div0_inf=False):
    syms = sorted(p.stem for p in (CACHE / "yf1m").glob("*.parquet") if p.stem not in ("SPY", "QQQ"))
    rows = []
    for sym in syms:
        o, h, l, c, v, t, day, hr, ts = load(sym)
        for cell, ov in CELLS.items():
            prm = dict(P.DEFAULTS); prm.update(ov)
            args = [prm[k] for k in ORDER]
            sA, sB, S, nT, nL, brk = P.run(o, h, l, c, v, t, day, hr, *args, div0_inf)
            sig = sA if PAT[cell] == "A" else sB
            for i in np.flatnonzero(sig):
                k = i if PAT[cell] == "A" else i - 1
                rows.append((cell, sym, ts[i], PAT[cell], int(k), S[k], nT[k], nL[k]))
    return pd.DataFrame(rows, columns=["cell", "sym", "ts_et", "pattern", "k", "S", "touch", "level"]), syms


def main():
    div0_inf = "--inf" in sys.argv
    mine, syms = run_all(div0_inf)
    tag = "_inf" if div0_inf else ""
    mine.to_csv(OUT / f"pine_signals{tag}.csv", index=False)
    eng = pd.read_csv(CACHE / "engine_signals_yf1m.csv")
    print("engine syms not in universe:", sorted(set(eng.sym) - set(syms)))
    m = mine.merge(eng, on=["cell", "sym", "ts_et"], how="outer", suffixes=("_pine", "_eng"), indicator=True)
    m.to_csv(OUT / f"diff{tag}.csv", index=False)
    for cell in CELLS:
        mm = m[m.cell == cell]
        print(f"{cell}: pine={int((mm._merge != 'right_only').sum())} engine={int((mm._merge != 'left_only').sum())} "
              f"matched={int((mm._merge == 'both').sum())} pine_only={int((mm._merge == 'left_only').sum())} "
              f"engine_only={int((mm._merge == 'right_only').sum())}")


if __name__ == "__main__":
    main()
