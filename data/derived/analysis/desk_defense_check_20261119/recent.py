"""Most recent quarter and rolling walk-forward for the consolidated research record (AGENTS.md, 2026-11-20).

Pool: every ledger saved by this folder (Codex's 303 and the 12 extension families) plus the season
study's 138 dials and references (season_regime_20261119/cache/meta_daily.parquet).
  1) Fixed headline rules in the most recent quarter (2026-09-01..11-17) and the full year, USD.
  2) Rolling walk-forward: at each month end hold, for the next month, the ledger with the best
     USD return over the previous 1, 3 or 6 months. Month hand-overs are chained close to close
     (zero commission, no fill at the switch) - an approximation that favours the walk-forward.
Run: py data/derived/analysis/desk_defense_check_20261119/recent.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common
import merge

SEASON = common.HERE.parent / "season_regime_20261119" / "cache" / "meta_daily.parquet"
RECENT = ("2026-09-01", "2026-11-17")
HEADLINE = {
    "base": "현행 주간 7종목",
    "half": "현행 절반 고정",
    "weekly_SOXX100r1": "SOXX 100일선 대피(Codex 남은 후보)",
    "stock100_cash_SOXX100r1": "SOXX·개별 100일선(Codex 상반기 선택)",
    "oos:sr_soxx50_cash20": "SOXX 50일선 → 20거래일 현금(10/10 동결)",
    "reentry:E1_R1_wait5_orgate": "SOXX 10일선 → 5거래일 대피",
    "crowd:C4r_SOXX10_cash": "SOXX 10일선 + BB-1 복귀",
    "season:M2_h20_K20_REV": "성격 다이얼",
    "season:M5_SOXX20r1_h10_K10_CASH": "두 다이얼",
}


def pool():
    curves, family = merge.load()
    season = pd.read_parquet(SEASON)
    for key, part in season.groupby("strategy", sort=False):
        if key.startswith("ref:"):
            continue
        curves[f"season:{key}"] = part.set_index("date")[common.LEDGER].astype(float)
        family[f"season:{key}"] = "season"
    base = curves["base"]
    for key, curve in curves.items():
        if not curve.index.equals(base.index):
            raise ValueError(f"Misaligned ledger {key}")
    return curves, family


def walk_forward(nav, lookback):
    """Hold next month the best ledger by USD return over the previous `lookback` months."""
    month = pd.Index(nav.index).str[:7]
    months = sorted(set(month))
    ends = {m: np.flatnonzero(month == m)[-1] for m in months}
    rows, value = [], 1.0
    for i, m in enumerate(months):
        if i < lookback:
            continue
        cut = ends[months[i - 1]]
        lo = ends[months[i - 1 - lookback]] if i - 1 - lookback >= 0 else None
        start = nav.iloc[lo] if lo is not None else pd.Series(1.0, index=nav.columns)
        trailing = nav.iloc[cut] / start - 1
        pick = trailing.idxmax()
        seg = nav[pick].iloc[ends[m]] / nav[pick].iloc[cut]
        base_seg = nav["base"].iloc[ends[m]] / nav["base"].iloc[cut]
        value *= seg
        rows.append({"lookback_months": lookback, "month": m, "pick": pick, "pick_ret": seg - 1,
                     "base_ret": base_seg - 1})
    return pd.DataFrame(rows)


def chained(part, months):
    sub = part[part.month.isin(months)]
    return float(np.prod(1 + sub.pick_ret) - 1), float(np.prod(1 + sub.base_ret) - 1)


def main():
    curves, family = pool()
    fixed = {}
    for key, label in HEADLINE.items():
        c = curves[key]
        r = common.window_stats(c, *RECENT)
        f = common.window_stats(c, "2026-01-01", "2026-12-31")
        fixed[key] = {"label": label, "recent_usd": r["usd"], "recent_usd_mdd": r["usd_mdd"],
                      "recent_krw": r["krw"], "full_usd": f["usd"], "full_usd_mdd": f["usd_mdd"]}
    nav = pd.DataFrame({k: v.usd for k, v in curves.items()})
    wf = pd.concat([walk_forward(nav, L) for L in (1, 3, 6)])
    wf.to_csv(common.OUT / "recent_walkforward.csv", index=False)
    summary = {}
    for L, part in wf.groupby("lookback_months"):
        jul_nov = chained(part, ["2026-07", "2026-08", "2026-09", "2026-10", "2026-11"])
        recent = chained(part, ["2026-09", "2026-10", "2026-11"])
        summary[int(L)] = {"jul_nov_pick": jul_nov[0], "jul_nov_base": jul_nov[1],
                           "recent_pick": recent[0], "recent_base": recent[1],
                           "months_pick_beats_base": int((part.pick_ret > part.base_ret).sum()),
                           "months": int(len(part)), "distinct_picks": int(part.pick.nunique())}
    out = {"pool": len(curves), "recent_window": RECENT, "fixed": fixed, "walk_forward": summary,
           "note": "month hand-overs chained close to close, zero commission"}
    (common.OUT / "recent_quarter.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:+.3f}")
    print("pool", len(curves))
    print(pd.DataFrame(fixed).T.to_string())
    print(pd.DataFrame(summary).T.to_string())
    print(wf[wf.month >= "2026-07"].to_string(index=False))


if __name__ == "__main__":
    main()
