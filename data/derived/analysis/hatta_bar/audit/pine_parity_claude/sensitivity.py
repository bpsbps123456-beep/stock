"""Non-vacuity check: perturb one Pine rule at a time and count how many signals change.

If a perturbation changes the signal set while the literal translation matches the engine 100%,
the engine provably encodes that rule the same way the Pine does (on this data).
Counts only - no outcome numbers.
"""
import importlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
import parity as PR

HERE = Path(__file__).resolve().parent
SRC = (HERE / "pine_literal.py").read_text(encoding="utf-8")
VAR = {
    "no_open_ge_thr(A)": ("and o[bi] >= thr[bi] and rec", "and rec"),
    "S_window_N_bars": ("L = nBars - 1", "L = nBars"),
    "trend_same_bar": ("tn1 = trendNow[bi - 1] if bi >= 1 else False", "tn1 = trendNow[bi]"),
    "trend_same_bar(B)": ("tn2 = trendNow[bi - 2] if bi >= 2 else False", "tn2 = trendNow[bi - 1]"),
    "atr_same_bar": ("atr1 = atr14[bi - 1] if bi >= 1 else NAN", "atr1 = atr14[bi]"),
    "B_density_at_break": ("and sane[bi] and densOk[bi]\n                        and c[bi] > o[bi]",
                           "and sane[bi] and densOk[bi - 1]\n                        and c[bi] > o[bi]"),
    "swing_left_non_strict": ("isSw[bi] = (lowp < lo_before)", "isSw[bi] = (lowp <= lo_before)"),
    "level_chain_anchor": ("                        lc += 1\n                    prev = x",
                           "                        lc += 1\n                        prev = x"),
    "vol_avg_incl_current": ("den = smaV[bi - 1] if bi >= 1 else NAN", "den = smaV[bi]"),
    "drop_uses_close": ("dd[bi] = (hh120[bi] - l[bi]) / hh120[bi]", "dd[bi] = (hh120[bi] - c[bi]) / hh120[bi]"),
    "touch_0.5pct": ("x <= S[bi] * (1 + 0.003)", "x <= S[bi] * (1 + 0.005)"),
}


def main():
    base = pd.read_csv(HERE / "pine_signals_inf.csv")
    key = lambda d: set(zip(d.cell, d.sym, d.ts_et))
    b = key(base)
    vdir = HERE / "variants"; vdir.mkdir(exist_ok=True)
    sys.path.insert(0, str(vdir))
    out = []
    for name, (old, new) in VAR.items():
        assert SRC.count(old) == 1, name
        mod = "v_" + "".join(ch if ch.isalnum() else "_" for ch in name)
        (vdir / f"{mod}.py").write_text(SRC.replace(old, new), encoding="utf-8")
        PR.P = importlib.import_module(mod)
        d, _ = PR.run_all(div0_inf=True)
        s = key(d)
        for cell in PR.CELLS:
            bc = {x for x in b if x[0] == cell}; sc = {x for x in s if x[0] == cell}
            out.append((name, cell, len(bc - sc), len(sc - bc)))
        print(name, "done", flush=True)
    r = pd.DataFrame(out, columns=["variant", "cell", "lost", "gained"])
    r["changed"] = r.lost + r.gained
    print(r.pivot(index="variant", columns="cell", values="changed").to_string())
    r.to_csv(HERE / "sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
