"""Post-hoc mechanism diagnostics for run.py outputs (not used by any verdict)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent


def main():
    dec = pd.read_csv(OUT / "decisions.csv")
    daily = pd.read_csv(OUT / "daily.csv")
    result = {}
    # 1. how many names each decision carried over from the previous decision of the same path
    for path in ("R0", "R1", "R2", "RF"):
        rows = dec[dec.path == path].reset_index(drop=True)
        prev, carried = set(), []
        for chosen in rows.chosen.fillna(""):
            now = set(filter(None, chosen.split("|")))
            carried.append(len(now & prev))
            prev = now
        result[f"{path}_mean_carried_over_per_decision"] = float(np.mean(carried[1:]))
        result[f"{path}_decisions"] = int(len(rows))
        result[f"{path}_mean_pass_count"] = float(rows.pass_count.mean())
    # 2. R2 vs R1 on the mid-month dates: how often the buffer changed the basket at all
    r1 = dec[dec.path == "R1"].set_index("decision_session").chosen.fillna("")
    r2 = dec[dec.path == "R2"].set_index("decision_session").chosen.fillna("")
    differs = [d for d in r1.index if set(r1[d].split("|")) != set(r2[d].split("|"))]
    result["R2_baskets_differing_from_R1"] = f"{len(differs)}/{len(r1)}"
    # 3. holdings bought at each spell entry (first off session), per path
    entries = {}
    off = daily[(daily.path == "R0")].reset_index(drop=True)
    flags = off.desk_regime.to_numpy()
    starts = [off.date[j] for j in range(1, len(off)) if flags[j - 1] and not flags[j]]
    for path in ("R0", "R1", "R2", "RF"):
        frame = daily[daily.path == path].set_index("date")
        entries[path] = {d: frame.at[d, "holdings"] for d in starts}
    result["spell_entry_holdings"] = entries
    # 4. how much of in-spell turnover is the switch itself (entry + exit days) vs. intra-spell refreshes
    for path in ("R0", "R1", "R2", "RF"):
        frame = daily[daily.path == path].reset_index(drop=True)
        g = frame.desk_regime.to_numpy()
        switch_days = [j for j in range(1, len(frame)) if g[j] != g[j - 1]]
        inside = [j for j in range(len(frame)) if not g[j] and j not in switch_days]
        result[f"{path}_turn_on_switch_days"] = float(frame.turn.iloc[switch_days].sum())
        result[f"{path}_turn_inside_spells"] = float(frame.turn.iloc[inside].sum())
    (OUT / "diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "spell_entry_holdings"}, indent=1))
    for path, e in entries.items():
        print(path, e)


if __name__ == "__main__":
    main()
