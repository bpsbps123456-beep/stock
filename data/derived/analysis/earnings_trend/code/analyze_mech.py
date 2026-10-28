"""실적 추세 연구 — 기계 계산 요인의 예측력(핵심 32 + 확대 76 = 108종목).

실행: py data/derived/analysis/earnings_trend/code/analyze_mech.py  → results/mech_ic.csv, results/mech.json
구간·IC 계산은 analyze.py와 같다(시즌 안 스피어만 IC 평균, 상·하위 1/3 차이).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import season_z, summarize, window  # noqa: E402
from fetch_sec import RESULTS  # noqa: E402

FACTORS = {"yoy": 1, "accel": 1, "eps_surprise": 1, "rule40": 1, "fcf_margin": 1, "sbc_gp": -1,
           "dilution_yoy": -1, "op_leverage": 1, "op_margin_chg": 1, "pre60_x": -1, "ps_ttm": -1,
           "log_mcap": -1, "r0_x": 1, "mech3": 1}
OUTCOMES = ["r0_x", "gap_x", "o2c_x", "odrift_x", "drift_x", "total_x", "d20_x", "d60_x", "drift5_x", "drift_ab"]


def main() -> None:
    df = pd.read_csv(RESULTS / "features_mech.csv")
    df["window"] = df["reaction_date"].apply(window)
    df = df[df["window"].notna()].copy()
    df["log_mcap"] = np.log(df["mcap_pre_b"].where(df["mcap_pre_b"] > 0))
    # 사전 지정 기계 합성: 성장 수준 + 가속 + EPS 서프라이즈(시즌 안 순위, 같은 무게)
    df["mech3"] = (season_z(df, "yoy") + season_z(df, "accel") + season_z(df, "eps_surprise")) / 3
    rows = []
    for sset in ["all", "core", "broad"]:
        d0 = df if sset == "all" else df[df["set"] == sset]
        for w in ["2025", "2026A", "2026B", "2026"]:
            d = d0[d0["window"].str.startswith("2026")] if w == "2026" else d0[d0["window"] == w]
            for f, sign in FACTORS.items():
                for y in OUTCOMES:
                    if f == "r0_x" and y in ("r0_x", "total_x"):
                        continue
                    s = summarize(d, f, y)
                    rows.append({"set": sset, "window": w, "factor": f, "sign": sign, "outcome": y,
                                 **{k: v for k, v in s.items() if k != "by_season"}})
    ic = pd.DataFrame(rows)
    ic.to_csv(RESULTS / "mech_ic.csv", index=False)
    # 규모별(전체 표본): 대형(≥$50B) vs 중소형
    df["big"] = df["mcap_pre_b"] >= 50
    sz = []
    for big, g in df.groupby("big"):
        for w in ["2025", "2026"]:
            gg = g[g["window"].str.startswith("2026")] if w == "2026" else g[g["window"] == w]
            for f in ["yoy", "accel", "eps_surprise", "mech3", "r0_x", "pre60_x"]:
                for y in ["drift_x", "total_x", "r0_x"]:
                    if f == "r0_x" and y in ("r0_x", "total_x"):
                        continue
                    s = summarize(gg, f, y)
                    sz.append({"big": bool(big), "window": w, "factor": f, "outcome": y, "ic": s["ic"], "t": s["t"], "n": s["n"]})
    size = df.groupby("size_tier").agg(n=("r0_x", "size"), abs_r0=("r0_x", lambda s: s.abs().mean()),
                                       abs_drift=("drift_x", lambda s: s.abs().mean())).round(2)
    # 가설 H1(기계판): 시즌 안 기계 합성(mech3) 상위 1/3 = '실적 좋음' × 반응일 방향 → 이후 드리프트
    d = df[df["mech3"].notna() & df["drift_x"].notna()].copy()
    rk = d.groupby("season")["mech3"].rank(pct=True)
    d["grade"] = np.where(rk > 2 / 3, "good", np.where(rk <= 1 / 3, "bad", "mid"))
    d["reaction"] = np.where(d["r0_x"] > 0, "up", "down")
    d["period"] = np.where(d["window"] == "2025", "2025", "2026")
    h1 = d.groupby(["period", "grade", "reaction"]).agg(
        n=("drift_x", "size"), drift_mean=("drift_x", "mean"), drift_median=("drift_x", "median"),
        win_pct=("drift_x", lambda s: (s > 0).mean() * 100)).round(2)
    h1.to_csv(RESULTS / "mech_h1.csv")
    # 사용자 가설 9(회복): 반응일 하락 발표만, 실적 등급별 발표 전 가격 회복 비율
    dd = d[d["r0"] < 0]
    rec = dd.groupby(["period", "grade"]).agg(
        n=("recover_days", "size"), rec20_pct=("recover_days", lambda s: (s <= 20).mean() * 100),
        rec40_pct=("recover_days", lambda s: (s <= 40).mean() * 100),
        never_pct=("recover_days", lambda s: np.isinf(s).mean() * 100), r0_mean=("r0_x", "mean")).round(1)
    rec.to_csv(RESULTS / "mech_h9_recovery.csv")
    out = {"n": len(df), "tickers": int(df["ticker"].nunique()), "by_window": df["window"].value_counts().to_dict(),
           "size_ic": sz, "size_move": size.reset_index().to_dict("records"), "n_tests": int(ic["ic"].notna().sum()),
           "h1": h1.reset_index().to_dict("records")}
    (RESULTS / "mech.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    show = ic[(ic["set"] == "all") & ic["outcome"].isin(["drift_x", "total_x", "r0_x"])]
    piv = show.pivot_table(index=["factor"], columns=["outcome", "window"], values="ic")
    print(piv.round(2).to_string())
    print(json.dumps({k: out[k] for k in ["n", "tickers", "by_window", "n_tests"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
