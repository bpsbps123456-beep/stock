"""grid.py·finalize.py 뒤에 돌린다. 선택 편향 진단과 엔진 이상 점검을 results.json에 더한다."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))
import engine as E  # noqa: E402


def rank(x):
    return np.asarray(x).argsort().argsort()


def main():
    path = HERE / "results.json"
    out = json.loads(path.read_text(encoding="utf-8"))
    rows = out["rows"]
    base = next(r for r in rows if r["id"] == out["baseline"]["id"])
    el = sorted([r for r in rows if r["eligible"]], key=lambda r: -r["calmar_H1"])
    d = {
        "spearman_calmarH1_vs_H2_eligible": float(np.corrcoef(rank([r["calmar_H1"] for r in el]), rank([r["H2"] for r in el]))[0, 1]),
        "spearman_H1_vs_H2_eligible": float(np.corrcoef(rank([r["H1"] for r in el]), rank([r["H2"] for r in el]))[0, 1]),
        "median_H2_top50_calmar": float(np.median([r["H2"] for r in el[:50]])),
        "median_H2_all_eligible": float(np.median([r["H2"] for r in el])),
        "share_n3_in_top50": float(np.mean([r["n"] == 3 for r in el[:50]])),
        "baseline_Mar_Nov": base["Mar_Nov"],
        "median_Mar_Nov_all": float(np.median([r["Mar_Nov"] for r in rows])),
        "share_Mar_Nov_gt_baseline": float(np.mean([r["Mar_Nov"] > base["Mar_Nov"] for r in rows])),
    }
    e = E.Engine()
    s = e.ret(120)
    chk = {"snapshot_before_2026-01-02": e.snapshot_before("2026-01-02")}
    for fq in ("W", "M"):
        r = e.run(s, universe="TIME", n=7, buffer=5, weight="time", caps=True, freq=fq)
        chk[f"TIME_n7_time_cap_{fq}_Jan"] = E.Engine.span(r, "2026-01-02", "2026-01-30")
        chk[f"TIME_n7_time_cap_{fq}_first_holding_day"] = next(iter(r.holdings))
    inv = {}
    for u in ("TIME", "FUNDS", "ALL50", "WIDETECH50"):
        for n in (3, 7, 20):
            r = e.run(s, universe=u, n=n, buffer=5, weight="equal", caps=True, freq="W")
            v = [sum(x.values()) for x in r.holdings.values()]
            inv[f"{u}_n{n}_equal_cap_W"] = {"mean_invested_at_rebalance": float(np.mean(v)), "min": float(np.min(v))}
    chk["caps_invested"] = inv
    out["selection_diagnostics"] = d
    out["engine_checks"] = chk
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"diag": d, "chk": chk}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
