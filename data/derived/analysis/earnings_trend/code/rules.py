"""실적 후 흐름 조건 규칙 찾기 — 2026 연도 내 전방 검증 + 우연 기준선(순열).

실행: py data/derived/analysis/earnings_trend/code/rules.py [--perm 60]  → results/rules_*.csv, results/rules.json

절차(결과를 보기 전에 정한 기준):
  조건 = 아래 PRED의 1~3개 AND(같은 묶음끼리는 섞지 않는다). 진입 E0(반응일 종가)·E5(5일 뒤 종가).
  결과 = SPY 대비 초과수익이 0보다 큰가(승). 기간 h20(20거래일), nx(다음 실적 반응일 전날까지).
  고르기(2026A = 반응일 2026-01~06): 건수 ≥ 50, 승률 − 그 구간 기준 승률 ≥ +12%p(매수 규칙) 또는 ≤ −12%p(회피 규칙),
           두 시즌(2026Q1·Q2) 각각 같은 쪽으로 ≥ 5%p(시즌별 건수 ≥ 15).
  확인(2026B = 반응일 2026-07~09): 건수 ≥ 20, 같은 쪽으로 ≥ 8%p, 평균 초과수익도 같은 부호.
  2025는 보조로 적기만 한다(연구 보고 규칙). 우연 기준선: 결과를 시즌 안에서 섞어 같은 절차를 --perm번 반복.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import window  # noqa: E402
from fetch_sec import RESULTS  # noqa: E402

N_A, LIFT_A, SEASON_LIFT, N_SEASON = 50, 0.12, 0.05, 15
N_B, LIFT_B = 20, 0.08


def predicates(df: pd.DataFrame, e5: bool) -> dict[str, tuple[str, str, np.ndarray]]:
    """이름 → (묶음, 한국어 설명, 마스크). 문턱은 숫자 감각으로 미리 정했다(데이터로 고르지 않음)."""
    g = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
    r0, sig, gap, o2c, clv, vr = g("r0_x"), g("r0_sig"), g("gap"), g("o2c"), g("clv"), g("vol_ratio")
    p60, p20, dh, eps, ry, ra, cap, v60 = (g("pre60_x"), g("pre20_x"), g("dist_hi"), g("eps_surprise"),
                                          g("rev_yoy"), g("rev_accel"), g("mcap_b"), g("vol60"))
    P = {
        "r0_big_up": ("r0", "반응일 SPY 대비 +8% 이상", r0 >= 8), "r0_up": ("r0", "반응일 +2~8%", (r0 >= 2) & (r0 < 8)),
        "r0_flat": ("r0", "반응일 ±2% 안", r0.abs() < 2), "r0_down": ("r0", "반응일 −2~−8%", (r0 <= -2) & (r0 > -8)),
        "r0_big_down": ("r0", "반응일 −8% 이하", r0 <= -8),
        "sig_up": ("sig", "반응일이 평소 하루 변동의 3배 이상 상승", sig >= 3),
        "sig_down": ("sig", "반응일이 평소 하루 변동의 3배 이상 하락", sig <= -3),
        "gap_go": ("go", "+3% 이상 갭 상승 뒤 장중 +1% 더", (gap >= 3) & (o2c >= 1)),
        "gap_fade": ("go", "+3% 이상 갭 상승 뒤 장중 −1% 밀림", (gap >= 3) & (o2c <= -1)),
        "gapdn_rev": ("go", "−3% 이하 갭 하락 뒤 장중 +1% 반등", (gap <= -3) & (o2c >= 1)),
        "gapdn_cont": ("go", "−3% 이하 갭 하락 뒤 장중 −1% 더", (gap <= -3) & (o2c <= -1)),
        "clv_hi": ("clv", "반응일 종가가 그날 범위 위 25%", clv >= 0.75),
        "clv_lo": ("clv", "반응일 종가가 그날 범위 아래 25%", clv <= 0.25),
        "vol_hi": ("vol", "반응일 거래량 50일 평균의 3배 이상", vr >= 3),
        "vol_mid": ("vol", "거래량 1.5~3배", (vr >= 1.5) & (vr < 3)), "vol_lo": ("vol", "거래량 1.5배 미만", vr < 1.5),
        "pre60_up": ("p60", "발표 전 60일 SPY 대비 +15% 이상", p60 >= 15),
        "pre60_dn": ("p60", "발표 전 60일 −15% 이하", p60 <= -15), "pre60_mid": ("p60", "발표 전 60일 ±5% 안", p60.abs() < 5),
        "pre20_up": ("p20", "발표 전 20일 +8% 이상", p20 >= 8), "pre20_dn": ("p20", "발표 전 20일 −8% 이하", p20 <= -8),
        "above_ma": ("ma", "발표 전 50·200일선 위", g("above_ma") == 1),
        "below_ma": ("ma", "발표 전 50·200일선 아래", g("below_ma") == 1),
        "near_hi": ("hi", "발표 전 52주 고점 −5% 안", dh >= -5), "mid_dd": ("hi", "52주 고점 −10~−30%", (dh < -10) & (dh > -30)),
        "deep_dd": ("hi", "52주 고점 −30% 이하", dh <= -30),
        "eps_big": ("eps", "EPS 서프라이즈 +15% 이상", eps >= 15), "eps_ok": ("eps", "EPS 서프라이즈 +3~15%", (eps >= 3) & (eps < 15)),
        "eps_small": ("eps", "EPS 서프라이즈 0~3%", (eps >= 0) & (eps < 3)), "eps_miss": ("eps", "EPS 미스", eps < 0),
        "rev_fast": ("rev", "매출 YoY +20% 이상", ry >= 20), "rev_mid": ("rev", "매출 YoY +5~20%", (ry >= 5) & (ry < 20)),
        "rev_slow": ("rev", "매출 YoY 0~5%", (ry >= 0) & (ry < 5)), "rev_decl": ("rev", "매출 역성장", ry < 0),
        "acc_up": ("acc", "매출 성장 가속 +3%p 이상", ra >= 3), "acc_dn": ("acc", "매출 성장 감속 −3%p 이하", ra <= -3),
        "mega": ("size", "시총 2,000억 달러 이상", cap >= 200), "large": ("size", "시총 300~2,000억", (cap >= 30) & (cap < 200)),
        "midcap": ("size", "시총 70~300억", (cap >= 7) & (cap < 30)), "smallcap": ("size", "시총 20~70억", cap < 7),
        "ai": ("seg", "AI 유니버스 종목", df["ai_sub"].fillna("") != ""),
        "tech": ("sec", "기술 업종", df["sector"] == "Technology"), "health": ("sec", "헬스케어", df["sector"] == "Health Care"),
        "fin": ("sec", "금융", df["sector"] == "Finance"), "consd": ("sec", "경기소비재", df["sector"] == "Consumer Discretionary"),
        "indus": ("sec", "산업재", df["sector"] == "Industrials"),
        "hivol": ("vlt", "평소 하루 변동 3.5% 이상(고변동)", v60 >= 3.5), "lowvol": ("vlt", "평소 하루 변동 1.8% 미만", v60 < 1.8),
    }
    if e5:
        d5 = g("d5_x")
        P.update({
            "d5_up": ("d5", "반응 뒤 5일 SPY 대비 +3% 이상", d5 >= 3), "d5_dn": ("d5", "반응 뒤 5일 −3% 이하", d5 <= -3),
            "d5_flat": ("d5", "반응 뒤 5일 ±1.5% 안", d5.abs() < 1.5),
            "held": ("hold", "5일 뒤 종가 ≥ 반응일 종가", g("held_reaction") == 1),
            "notheld": ("hold", "5일 뒤 종가 < 반응일 종가", g("held_reaction") == 0),
            "undone": ("undo", "5일 안에 반응을 모두 되돌림", g("back_to_pre") == 1),
        })
    return {k: (f, lab, np.asarray(m.fillna(False), dtype=bool)) for k, (f, lab, m) in P.items()}


def enumerate_rules(names: list[str], fam: dict[str, str], max_k: int = 3) -> list[tuple]:
    out = []
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(range(len(names)), k):
            fams = [fam[names[i]] for i in combo]
            if len(set(fams)) == len(fams):
                out.append(combo)
    return out


def stats(M: np.ndarray, y: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """M: (사건, 규칙) bool. 반환 (건수, 승수) — 결과가 있는 사건만."""
    Mv = M[valid]
    return Mv.sum(0), (y[valid].astype(np.float32) @ Mv.astype(np.float32))


def evaluate(df: pd.DataFrame, P: dict, rules: list[tuple], names: list[str], ycol: str, yvals: np.ndarray) -> pd.DataFrame:
    win = yvals > 0
    valid = ~np.isnan(yvals)
    masks = np.stack([P[n][2] for n in names], axis=1)
    out = {}
    groups = {"A": df["window"].eq("2026A").values, "A1": df["season"].eq("2026Q1").values,
              "A2": df["season"].eq("2026Q2").values, "B": df["window"].eq("2026B").values, "P": df["window"].eq("2025").values}
    # 규칙 마스크를 덩어리로 만들어 센다(메모리 절약)
    for g, sel in groups.items():
        out[f"n_{g}"], out[f"w_{g}"] = [], []
        out[f"base_{g}"] = float(win[sel & valid].mean()) if (sel & valid).any() else np.nan
    sums_B = []
    for i in range(0, len(rules), 4000):
        chunk = rules[i:i + 4000]
        M = np.stack([np.logical_and.reduce([masks[:, j] for j in c]) for c in chunk], axis=1)
        for g, sel in groups.items():
            n, w = stats(M, win, sel & valid)
            out[f"n_{g}"].append(n)
            out[f"w_{g}"].append(w)
        sv = sel_B = groups["B"] & valid
        sums_B.append(np.nan_to_num(yvals[sv]).astype(np.float64) @ M[sv].astype(np.float64))
    res = pd.DataFrame({"rule": [" & ".join(names[j] for j in c) for c in rules]})
    for g in groups:
        n = np.concatenate(out[f"n_{g}"])
        w = np.concatenate(out[f"w_{g}"])
        res[f"n_{g}"] = n
        res[f"win_{g}"] = np.where(n > 0, w / np.maximum(n, 1), np.nan)
        res[f"lift_{g}"] = res[f"win_{g}"] - out[f"base_{g}"]
    res["mean_B"] = np.concatenate(sums_B) / np.maximum(res["n_B"], 1)
    res["target"] = ycol
    return res


def select(res: pd.DataFrame) -> pd.DataFrame:
    side = np.sign(res["lift_A"])
    ok_a = (res["n_A"] >= N_A) & (res["lift_A"].abs() >= LIFT_A)
    ok_s = ((res["n_A1"] >= N_SEASON) & (res["n_A2"] >= N_SEASON)
            & (res["lift_A1"] * side >= SEASON_LIFT) & (res["lift_A2"] * side >= SEASON_LIFT))
    ok_b = (res["n_B"] >= N_B) & (res["lift_B"] * side >= LIFT_B) & (np.sign(res["mean_B"]) == side)
    res = res.assign(side=np.where(side > 0, "매수", "회피"), pass_A=ok_a & ok_s, pass_B=ok_a & ok_s & ok_b)
    return res


ALT = {"h20_x": ["h20_e", "h20_s"], "nx_x": ["nx_e", "nx_s"], "e5_h20_x": ["e5_h20_e", "e5_h20_s"], "e5_nx_x": ["e5_nx_e", "e5_nx_s"]}


def robustness(df: pd.DataFrame, passed: pd.DataFrame) -> pd.DataFrame:
    """통과 규칙을 다른 기준(동일가중 평균 종목·업종 ETF)과 AI 종목만으로 다시 잰다(2026B 확인 구간)."""
    out = []
    for _, r in passed.iterrows():
        P = predicates(df, e5=(r["entry"] == "E5"))
        m = np.logical_and.reduce([P[n][2] for n in r["rule"].split(" & ")])
        b = df["window"].eq("2026B").values
        row = r.to_dict()
        for alt in ALT[r["target"]]:
            y = pd.to_numeric(df[alt], errors="coerce").values
            ok = b & ~np.isnan(y)
            base = (y[ok] > 0).mean()
            sel = ok & m
            row[f"B_{alt}_win"] = (y[sel] > 0).mean() if sel.any() else np.nan
            row[f"B_{alt}_base"] = base
        y = pd.to_numeric(df[r["target"]], errors="coerce").values
        for g, name in [(df["window"].str.startswith("2026").values, "2026"), (b, "B")]:
            sel = g & m & ~np.isnan(y) & (df["ai_sub"].fillna("") != "").values
            row[f"ai_{name}_n"] = int(sel.sum())
            row[f"ai_{name}_win"] = (y[sel] > 0).mean() if sel.any() else np.nan
        row["median_B"] = float(np.nanmedian(y[b & m])) if (b & m).any() else np.nan
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=60)
    a = ap.parse_args()
    df = pd.read_csv(RESULTS / "events_wide.csv")
    df["window"] = df["reaction_date"].apply(window).fillna("other")
    targets = [("E0", "h20_x"), ("E0", "nx_x"), ("E5", "e5_h20_x"), ("E5", "e5_nx_x")]
    summary, keep = {"n_events": len(df), "by_window": df["window"].value_counts().to_dict(), "targets": {}}, []
    rng = np.random.default_rng(20261029)
    for entry, ycol in targets:
        P = predicates(df, e5=(entry == "E5"))
        names = list(P)
        fam = {n: P[n][0] for n in names}
        rules = enumerate_rules(names, fam)
        y = pd.to_numeric(df[ycol], errors="coerce").values
        res = select(evaluate(df, P, rules, names, ycol, y))
        res["entry"] = entry
        passed = res[res["pass_B"]].copy()
        keep.append(passed)
        # 우연 기준선: 시즌 안에서 결과를 섞는다
        null_a, null_b = [], []
        seasons = df["season"].values
        for _ in range(a.perm):
            yp = y.copy()
            for s in np.unique(seasons):
                idx = np.where(seasons == s)[0]
                yp[idx] = yp[rng.permutation(idx)]
            rp = select(evaluate(df, P, rules, names, ycol, yp))
            null_a.append(int(rp["pass_A"].sum()))
            null_b.append(int(rp["pass_B"].sum()))
        summary["targets"][f"{entry}:{ycol}"] = {
            "rules_tested": len(rules), "pass_A": int(res["pass_A"].sum()), "pass_B": int(res["pass_B"].sum()),
            "null_pass_A_mean": float(np.mean(null_a)), "null_pass_A_p95": float(np.percentile(null_a, 95)),
            "null_pass_B_mean": float(np.mean(null_b)), "null_pass_B_p95": float(np.percentile(null_b, 95)),
            "null_pass_B_max": int(np.max(null_b)),
            "base": {g: float(res[f"win_{g}"].iloc[0] - res[f"lift_{g}"].iloc[0]) for g in ["A", "B", "P"]},
        }
        print(f"{entry}:{ycol} 규칙 {len(rules)}개 → 고르기 통과 {int(res['pass_A'].sum())} (우연 평균 {np.mean(null_a):.1f}) "
              f"→ 확인 통과 {int(res['pass_B'].sum())} (우연 평균 {np.mean(null_b):.1f}, 95% {np.percentile(null_b, 95):.0f})", flush=True)
    allp = pd.concat(keep, ignore_index=True)
    allp = robustness(df, allp)
    allp.to_csv(RESULTS / "rules_passed.csv", index=False)
    (RESULTS / "rules.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
