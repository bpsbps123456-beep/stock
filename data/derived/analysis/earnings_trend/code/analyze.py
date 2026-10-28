"""실적 추세 연구 — 요인별 예측력·가설 검정·합성 점수 전방 검증.

실행: py data/derived/analysis/earnings_trend/code/analyze.py  → results/analysis.json, results/*.csv

구간(연구 보고 규칙: 2026이 합격 기준, 연도 내 전방 검증이 주 근거, 2025는 보조):
  2025   반응일 2025-01~12 (4시즌)          보조 확인
  2026A  반응일 2026-01~06 (2시즌)          고르는 구간
  2026B  반응일 2026-07~09 (1시즌)          확인 구간 — 모델 지식 시점(2026-06) 뒤라 LLM 정성 판단도 깨끗하다
  2026-10 이후 반응은 보유 구간이 며칠뿐이라 추세 검정에서 뺀다.
매크로 제거: 같은 시즌(반응일 달력 분기) 안에서만 순위를 비교한다(시즌별 스피어만 IC → 평균).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sec import RESULTS  # noqa: E402

OUTCOMES = ["r0_x", "gap_x", "o2c_x", "odrift_x", "drift_x", "total_x", "d20_x", "d60_x", "drift5_x", "drift_ab"]
# 부호: +면 클수록 좋다고 본 요인. 음(−)은 작을수록 좋다고 본 요인(희석·SBC·비싼 값)
FACTORS = {
    "yoy": 1, "yoy_peer_pct": 1, "accel": 1, "guide_yoy": 1, "guide_accel": 1, "beat_guide": 1,
    "fy_change": 1, "profit_change": 1, "eps_surprise": 1, "seg_yoy": 1, "backlog_yoy": 1,
    "rule40": 1, "fcf_margin": 1, "sbc_gp": -1, "dilution_yoy": -1, "op_leverage": 1, "op_margin_chg": 1,
    "narrative": 1, "tone": 1, "catalyst": 1, "showstopper": 1,
    "sn1": 1, "sn2": 1, "sn3": 1, "sn4": 1, "sn5": 1, "sn6": 1, "sn136": 1, "sn_total": 1,
    "pre60_x": -1, "ps_ttm": -1, "log_mcap": -1, "r0_x": 1,
}


def window(d: str) -> str | None:
    if "2025-01-01" <= d < "2026-01-01":
        return "2025"
    if "2026-01-01" <= d < "2026-07-01":
        return "2026A"
    if "2026-07-01" <= d < "2026-10-01":
        return "2026B"
    return None


def season_ic(df: pd.DataFrame, f: str, y: str, min_n: int = 8) -> list[tuple[str, float, int]]:
    out = []
    for s, g in df.groupby("season"):
        g = g[[f, y]].dropna()
        if len(g) >= min_n and g[f].nunique() > 2:
            out.append((s, stats.spearmanr(g[f], g[y]).statistic, len(g)))
    return out


def summarize(df: pd.DataFrame, f: str, y: str) -> dict:
    ics = season_ic(df, f, y)
    if not ics:
        return {"ic": np.nan, "t": np.nan, "seasons": 0, "n": 0, "spread": np.nan}
    v = np.array([x[1] for x in ics])
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 and v.std(ddof=1) > 0 else np.nan
    # 시즌 안 상위 1/3 − 하위 1/3 평균(%p)
    sp = []
    for _, g in df.groupby("season"):
        g = g[[f, y]].dropna()
        if len(g) < 8 or g[f].nunique() <= 2:
            continue
        lo, hi = g[f].quantile([1 / 3, 2 / 3])
        top, bot = g[g[f] >= hi][y], g[g[f] <= lo][y]
        if len(top) and len(bot):
            sp.append(top.mean() - bot.mean())
    return {"ic": round(float(v.mean()), 3), "t": round(float(t), 2) if pd.notna(t) else np.nan,
            "seasons": len(v), "n": int(sum(x[2] for x in ics)),
            "spread": round(float(np.mean(sp)), 2) if sp else np.nan,
            "by_season": {s: round(float(i), 3) for s, i, _ in ics}}


def season_z(df: pd.DataFrame, f: str) -> pd.Series:
    """시즌 안 순위를 −1..+1로(결측은 0 = 중간)."""
    r = df.groupby("season")[f].rank(pct=True)
    return ((r - 0.5) * 2).fillna(0)


def main() -> None:
    df = pd.read_csv(RESULTS / "features.csv")
    df = df[df["is_q"]].copy()
    df["window"] = df["reaction_date"].apply(window)
    df["log_mcap"] = np.log(df["mcap_pre_b"])
    df["sn136"] = df[["sn1", "sn3", "sn6"]].sum(axis=1, min_count=3)
    df = df[df["window"].notna()].copy()
    res: dict = {"n_events": len(df), "by_window": df["window"].value_counts().to_dict(),
                 "tickers": int(df["ticker"].nunique())}

    # 1) 요인 × 결과 × 구간
    rows = []
    for w in ["2025", "2026A", "2026B", "2026"]:
        d = df[df["window"].str.startswith("2026")] if w == "2026" else df[df["window"] == w]
        for f, sign in FACTORS.items():
            if f not in d:
                continue
            for y in OUTCOMES:
                if f == "r0_x" and y in ("r0_x", "total_x"):
                    continue
                s = summarize(d, f, y)
                rows.append({"window": w, "factor": f, "sign": sign, "outcome": y,
                             **{k: v for k, v in s.items() if k != "by_season"}})
    ic = pd.DataFrame(rows)
    ic.to_csv(RESULTS / "factor_ic.csv", index=False)
    res["n_tests"] = int(ic["ic"].notna().sum())

    # 2) 가설 H1: 실적 좋음(시즌 안 S&N 상위 절반) × 반응일 방향 → 이후 드리프트
    d = df[df["sn_total"].notna()].copy()
    d["good"] = d.groupby("season")["sn_total"].rank(pct=True) > 0.5
    d["up"] = d["r0_x"] > 0
    h1 = d.groupby(["window", "good", "up"]).agg(n=("drift_x", "size"), drift_x=("drift_x", "mean"),
                                                 drift_med=("drift_x", "median"),
                                                 win=("drift_x", lambda s: (s > 0).mean() * 100)).round(2)
    h1.to_csv(RESULTS / "h1_good_drop.csv")
    res["h1"] = h1.reset_index().to_dict("records")

    # 2b) H1 기계판: LLM 판단 없이 반응일을 가장 잘 맞힌 세 숫자(EPS 서프라이즈·가이던스 비트·가이던스 궤도)로 '실적 좋음'
    df["q_mech"] = (season_z(df, "eps_surprise") + season_z(df, "beat_guide") + season_z(df, "guide_accel")) / 3
    d = df[df["drift_x"].notna()].copy()
    rk = d.groupby("season")["q_mech"].rank(pct=True)
    d["grade"] = np.where(rk > 2 / 3, "good", np.where(rk <= 1 / 3, "bad", "mid"))
    d["reaction"] = np.where(d["r0_x"] > 0, "up", "down")
    d["period"] = np.where(d["window"] == "2025", "2025", "2026")
    h1m = d.groupby(["period", "grade", "reaction"]).agg(
        n=("drift_x", "size"), drift_mean=("drift_x", "mean"), drift_median=("drift_x", "median"),
        win_pct=("drift_x", lambda s: (s > 0).mean() * 100)).round(2)
    h1m.to_csv(RESULTS / "h1_mech_core.csv")
    res["h1_mech"] = h1m.reset_index().to_dict("records")

    # 2b') 사용자 가설 9: 반응일에 떨어진 발표 중 실적 좋음/나쁨별로 발표 전 가격을 20·40거래일 안에 되찾은 비율
    dd = d[d["r0_x"].notna() & (df.loc[d.index, "r0"] < 0)].copy()
    dd["rd"] = df.loc[dd.index, "recover_days"]
    rec = dd.groupby(["period", "grade"]).agg(
        n=("rd", "size"), rec20_pct=("rd", lambda s: (s <= 20).mean() * 100),
        rec40_pct=("rd", lambda s: (s <= 40).mean() * 100), never_pct=("rd", lambda s: np.isinf(s).mean() * 100),
        r0_mean=("r0_x", "mean")).round(1)
    rec.to_csv(RESULTS / "h9_recovery.csv")
    res["h9_recovery"] = rec.reset_index().to_dict("records")

    # 2c) LLM 사후지식 오염 점검: 모델 지식 시점(2026-06) 앞뒤로 정성 판단의 반응일 IC가 무너지면 오염 신호
    cont = []
    for f in ["narrative", "tone", "catalyst", "showstopper", "eps_surprise", "beat_guide", "guide_accel"]:
        for w in ["2025", "2026A", "2026B"]:
            s = summarize(df[df["window"] == w], f, "r0_x")
            cont.append({"factor": f, "window": w, "ic_r0": s["ic"], "n": s["n"]})
    res["contamination"] = cont

    # 3) 규모: 반응·드리프트 크기와 요인 IC(대형 vs 중소형)
    df["big"] = df["mcap_pre_b"] >= 50
    size = df.groupby("size_tier").agg(n=("r0", "size"), abs_r0=("r0_x", lambda s: s.abs().mean()),
                                       abs_drift=("drift_x", lambda s: s.abs().mean()),
                                       sd_drift=("drift_x", "std")).round(2)
    res["size"] = size.reset_index().to_dict("records")
    sz = []
    for big, g in df.groupby("big"):
        for f in ["yoy", "accel", "guide_accel", "beat_guide", "sn_total", "narrative", "r0_x"]:
            for y in ["drift_x", "total_x"]:
                s = summarize(g, f, y)
                sz.append({"big": bool(big), "factor": f, "outcome": y, "ic": s["ic"], "t": s["t"], "n": s["n"]})
    res["size_ic"] = sz

    # 4) 추세 유지: 반응 방향이 다음 실적까지 이어지나 / 발표 전 추세가 뒤집히나
    p = df.dropna(subset=["r0_x", "drift_x", "pre60_x"])
    res["persistence"] = {
        "same_sign_r0_drift_pct": round(float((np.sign(p["r0_x"]) == np.sign(p["drift_x"])).mean() * 100), 1),
        "same_sign_pre60_drift_pct": round(float((np.sign(p["pre60_x"]) == np.sign(p["drift_x"])).mean() * 100), 1),
        "n": len(p),
    }
    # 같은 종목의 연속 분기: 이번 드리프트와 다음 드리프트 부호
    q = df.sort_values(["ticker", "reaction_date"])
    q["next_drift"] = q.groupby("ticker")["drift_x"].shift(-1)
    qq = q.dropna(subset=["drift_x", "next_drift"])
    res["persistence"]["same_sign_drift_next_pct"] = round(float((np.sign(qq["drift_x"]) == np.sign(qq["next_drift"])).mean() * 100), 1)

    # 5) 합성 점수: 2026A에서 drift_x IC가 기대 부호로 0.10 이상인 요인만 같은 무게 → 2026B·2025에서 확인
    a = ic[(ic["window"] == "2026A") & (ic["outcome"] == "drift_x")].dropna(subset=["ic"])
    a = a[~a["factor"].str.startswith("sn") & (a["factor"] != "r0_x")]
    pick = a[(a["ic"] * a["sign"] >= 0.10)]
    chosen = {r["factor"]: int(r["sign"]) for _, r in pick.iterrows()}
    res["v2_selection"] = {"rule": "2026A drift_x 시즌 IC × 기대부호 ≥ 0.10, sn·r0 제외, 같은 무게",
                           "candidates": len(a), "chosen": chosen}
    if chosen:
        df["v2"] = sum(season_z(df, f) * s for f, s in chosen.items()) / len(chosen)
    else:
        df["v2"] = np.nan
    # 사전 지정 합성(고르기 없음): 사용자 틀의 핵심 — 성장 수준·가속·전망 궤도·비트·연간 전망·내러티브
    pre = {"yoy": 1, "accel": 1, "guide_accel": 1, "beat_guide": 1, "fy_change": 1, "narrative": 1}
    df["v_pre"] = sum(season_z(df, f) * s for f, s in pre.items()) / len(pre)
    comp = []
    for w in ["2025", "2026A", "2026B"]:
        d = df[df["window"] == w]
        for f in ["sn_total", "sn136", "v_pre", "v2", "q_mech"]:
            for y in ["drift_x", "total_x", "d60_x", "drift5_x", "r0_x"]:
                s = summarize(d, f, y)
                comp.append({"window": w, "score": f, "outcome": y, "ic": s["ic"], "t": s["t"], "spread": s["spread"], "n": s["n"]})
    res["composite"] = comp
    df.to_csv(RESULTS / "scored_events.csv", index=False)
    (RESULTS / "analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ["n_events", "by_window", "n_tests", "persistence", "v2_selection"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
