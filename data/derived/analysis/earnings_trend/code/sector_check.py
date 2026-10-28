"""실적 추세 연구 — 업종 쏠림 제거 확인: 드리프트를 업종 ETF 대비로 바꿔 다시 잰다.

실행: py data/derived/analysis/earnings_trend/code/sector_check.py  → results/sector_check.json

실적 점수 상위가 한 업종(반도체)에 몰리면, 시즌 안 순위 비교만으로는 업종 회전(2026-09~10 반도체 급락 같은)을
지우지 못한다. 업종 ETF: 반도체·장비·메모리·인터커넥트 = SOXX, 소프트웨어·보안 = IGV, 헬스케어 = XLV,
발전 = XLU, 나머지 = QQQ. 분류는 watchlist.AI_SUBSECTOR(정본)의 대분류를 쓰고, 비AI 5종은 아래에 둔다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import season_z, summarize, window  # noqa: E402
from fetch_sec import ROOT, RESULTS  # noqa: E402

sys.path.insert(0, str(ROOT.parents[3]))
import watchlist  # noqa: E402

TOP_TO_ETF = {"칩 설계": "SOXX", "파운드리·후공정": "SOXX", "메모리·스토리지": "SOXX", "장비·소재·테스트": "SOXX",
              "소프트웨어·보안": "IGV", "전력·에너지": "XLU"}
NON_AI = {"LLY": "XLV", "UNH": "XLV", "NFLX": "QQQ", "UBER": "QQQ", "SOFI": "QQQ"}


def etf_of(t: str) -> str:
    if t in NON_AI:
        return NON_AI[t]
    for top, subs in watchlist.AI_UNIVERSE.items():
        if any(t in syms for syms in subs.values()):
            return TOP_TO_ETF.get(top, "QQQ")
    return "QQQ"


def sector_drift(df: pd.DataFrame, px: pd.DataFrame) -> pd.Series:
    out = []
    for _, r in df.iterrows():
        e = etf_of(r["ticker"])
        a, b = r["reaction_date"], r["hold_end_date"]
        if e not in px or a not in px.index or b not in px.index:
            out.append(np.nan)
            continue
        out.append(r["drift"] - (px.at[b, e] / px.at[a, e] - 1) * 100)
    return pd.Series(out, index=df.index)


def main() -> None:
    px = pd.read_csv(RESULTS / "prices_adj_close.csv", index_col=0)
    res = {}
    core = pd.read_csv(RESULTS / "features.csv")
    core = core[core["is_q"]].copy()
    core["window"] = core["reaction_date"].apply(window)
    core = core[core["window"].notna()].copy()
    core["sn136"] = core[["sn1", "sn3", "sn6"]].sum(axis=1, min_count=3)
    core["q_mech"] = (season_z(core, "eps_surprise") + season_z(core, "beat_guide") + season_z(core, "guide_accel")) / 3
    core["drift_sec"] = sector_drift(core, px)
    core["etf"] = core["ticker"].map(etf_of)
    rows = []
    for f in ["sn_total", "sn136", "q_mech", "beat_guide", "eps_surprise", "guide_accel", "yoy", "narrative", "fcf_margin", "pre60_x"]:
        for w in ["2025", "2026A", "2026B"]:
            d = core[core["window"] == w]
            rows.append({"set": "core", "factor": f, "window": w, "ic_drift_x": summarize(d, f, "drift_x")["ic"],
                         "ic_drift_sec": summarize(d, f, "drift_sec")["ic"]})
    mech = pd.read_csv(RESULTS / "features_mech.csv")
    mech["window"] = mech["reaction_date"].apply(window)
    mech = mech[mech["window"].notna()].copy()
    mech["mech3"] = (season_z(mech, "yoy") + season_z(mech, "accel") + season_z(mech, "eps_surprise")) / 3
    pxb = pd.read_csv(RESULTS / "prices_adj_close_broad.csv", index_col=0)
    mech["drift_sec"] = sector_drift(mech, pd.concat([px, pxb.drop(columns=[c for c in pxb if c in px])], axis=1))
    for f in ["mech3", "eps_surprise", "yoy", "accel", "fcf_margin", "pre60_x", "op_margin_chg"]:
        for w in ["2025", "2026A", "2026B"]:
            d = mech[mech["window"] == w]
            rows.append({"set": "108", "factor": f, "window": w, "ic_drift_x": summarize(d, f, "drift_x")["ic"],
                         "ic_drift_sec": summarize(d, f, "drift_sec")["ic"]})
    out = pd.DataFrame(rows)
    res["ic"] = out.to_dict("records")
    res["core_etf_share_top_third_2026B"] = (
        core[core["window"] == "2026B"].assign(r=lambda x: x["sn_total"].rank(pct=True))
        .query("r > 0.6667")["etf"].value_counts().to_dict())
    (RESULTS / "sector_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(out.pivot_table(index=["set", "factor"], columns="window", values=["ic_drift_x", "ic_drift_sec"]).round(2).to_string())
    print(res["core_etf_share_top_third_2026B"])


if __name__ == "__main__":
    main()
