"""PREREG3 넓은 격자: 고르기 S(넓은 표본 2026 상반기) → 관문 A(펀드 밖 하반기) → 관문 B(펀드 하반기).

조건: S 33·67분위 문턱, 참/거짓, 업종 같음. 1~3조건 AND. 진입 2 × 기간 3 × 방향 2.
우연 기준: S와 A의 결과(QQQ 대비·업종 대비 한 쌍)를 시즌 안에서 섞어 100번.
결과: results/rules_wide_summary.json, results/rules_wide_pass.csv(S·A 통과 규칙 전부와 B 성적)
--datedm: 결과에서 같은 표본 묶음(S/A/B)·같은 반응일 사건 평균을 뺀다(같은 기간 시장·업종 공통 움직임 제거). 사전 등록 뒤
추가한 갈래(2026-11-20): 첫 실행의 통과 규칙이 A에서 2주에 몰려 날짜 군집이 우연 기준을 느슨하게 만든 것을 보고 더했다.
결과 파일 접미사 _datedm, 가장 가까운 후보 results/rules_wide_top{_datedm}.csv
"""
import sys
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PERM = 100
RNG = np.random.default_rng(3)
FEATS = ["mcap_b", "eps_surprise", "rev_yoy", "rev_accel", "pre20_x", "pre60_x", "dist_hi", "vol60", "vsurge5", "dvol20",
         "intra20", "over20", "eff20", "r0_x", "gap", "o2c", "clv", "vol0", "r0_sig", "qqq_pre20", "vix", "soxx_rel20",
         "same_day_n", "occ_tot_ab", "occ_call_ab", "occ_putshare", "dtc", "si_ch", "kr_days_pre", "kr_buy_share"]
BOOLS = ["qqq_above50", "react_mon"]
E6_ONLY = ["d5_x", "str5"]


def main() -> None:
    p = pd.read_csv(BASE / "results/panel_wide.csv")
    rd = p.reaction_date.astype(str)
    S = ((p.src == "wide") & (rd < "2026-07-01")).to_numpy()
    q1 = S & (rd < "2026-04-01").to_numpy()
    q2 = S & ~q1
    A = ((p.src == "wide") & ~p.fund_ticker & (rd >= "2026-07-01") & (rd <= "2026-10-21")).to_numpy()
    Bm = ((p.src == "fund") & (rd >= "2026-07-01") & (rd <= "2026-10-21")).to_numpy()
    season = np.where(rd < "2026-04-01", "Q1", np.where(rd < "2026-07-01", "Q2", np.where(rd < "2026-10-01", "Q3", "Q4")))
    dm = "--datedm" in sys.argv
    tag = "_datedm" if dm else ""
    grp = np.where(S, "S", np.where(A, "A", np.where(Bm, "B", "x")))
    summary, passed, tops = {}, [], []
    for entry in ("e1", "e6"):
        feats = FEATS + (E6_ONLY if entry == "e6" else [])
        names, masks = [], []
        for c in feats:
            x = p[c].to_numpy(float)
            lo, hi = np.nanpercentile(x[S], 33.333), np.nanpercentile(x[S], 66.667)
            names += [(c, f"{c}≤{lo:.4g}"), (c, f"{c}≥{hi:.4g}")]
            masks += [x <= lo, x >= hi]
        for c in BOOLS:
            b = p[c].astype(bool).to_numpy()
            names += [(c, f"{c}=참"), (c, f"{c}=거짓")]
            masks += [b, ~b]
        for s in p.sector.dropna().unique():
            names.append(("sector", f"sector={s}"))
            masks.append((p.sector == s).to_numpy())
        masks = np.array(masks)
        m = len(names)
        cmb = [(i,) for i in range(m)]
        cmb += [(i, j) for i, j in itertools.combinations(range(m), 2) if names[i][0] != names[j][0]]
        cmb += [(i, j, k) for i, j, k in itertools.combinations(range(m), 3) if len({names[i][0], names[j][0], names[k][0]}) == 3]
        for h in (5, 10, 20):
            y, ys = p[f"{entry}_h{h}"].to_numpy(), p[f"{entry}_h{h}_s"].to_numpy()
            if dm:
                y = y - pd.Series(y).groupby([grp, rd.to_numpy()]).transform("mean").to_numpy()
                ys = ys - pd.Series(ys).groupby([grp, rd.to_numpy()]).transform("mean").to_numpy()
            valid = ~np.isnan(y)
            Y, YS = np.tile(np.nan_to_num(y), (PERM + 1, 1)).T, np.tile(np.nan_to_num(ys), (PERM + 1, 1)).T
            for pp in range(1, PERM + 1):  # S·A 안에서 시즌별로 (QQQ 대비, 업종 대비)를 한 쌍으로 섞는다
                for g_ in (S, A):
                    for s in ("Q1", "Q2", "Q3", "Q4"):
                        rows = np.where(g_ & valid & (season == s))[0]
                        if len(rows) > 1:
                            perm = RNG.permutation(rows)
                            Y[rows, pp], YS[rows, pp] = y[perm], ys[perm]
            sets = {"q1": q1 & valid, "q2": q2 & valid, "A": A & valid, "B": Bm & valid}
            for direction in ("up", "down"):
                W = (Y > 0) if direction == "up" else (Y < 0)
                WS = (YS > 0) if direction == "up" else (YS < 0)
                key = f"{entry}_h{h}_{direction}"
                cnt = {"S": np.zeros(PERM + 1, int), "SA": np.zeros(PERM + 1, int), "SAB": np.zeros(PERM + 1, int)}
                for st in range(0, len(cmb), 4000):
                    ch = cmb[st:st + 4000]
                    M = np.ones((len(ch), len(p)), dtype=bool)
                    for r, c in enumerate(ch):
                        for i in c:
                            M[r] &= masks[i]
                    Mf = M.astype(np.float32)
                    n = {k_: Mf[:, np.where(v)[0]].sum(1) for k_, v in sets.items()}
                    w = {k_: Mf[:, np.where(v)[0]] @ W[np.where(v)[0]].astype(np.float32) for k_, v in sets.items()}
                    wsA = Mf[:, np.where(sets["A"])[0]] @ WS[np.where(sets["A"])[0]].astype(np.float32)
                    nS, wS = n["q1"] + n["q2"], w["q1"] + w["q2"]
                    okS = ((nS >= 60)[:, None] & (wS >= 0.70 * nS[:, None]) & (n["q1"] >= 20)[:, None]
                           & (w["q1"] >= 0.65 * n["q1"][:, None]) & (n["q2"] >= 20)[:, None] & (w["q2"] >= 0.65 * n["q2"][:, None]))
                    okA = okS & (n["A"] >= 40)[:, None] & (w["A"] >= 0.70 * n["A"][:, None]) & (wsA >= 0.60 * n["A"][:, None])
                    nB = n["B"][:, None]
                    okB = okA & ((nB < 10) | (w["B"] >= 0.65 * nB))  # B 10건 미만은 '펀드 미확인'으로 통과 표시
                    cnt["S"] += okS.sum(0)
                    near = np.where((nS >= 60) & (wS[:, 0] >= 0.62 * nS))[0]  # 가장 가까운 후보 기록(진짜 자료만)
                    for r in near:
                        tops.append({"family": key, "rule": " & ".join(names[i][1] for i in ch[r]), "n_S": int(nS[r]),
                                     "hit_S": round(float(wS[r, 0] / nS[r]), 3), "hit_q1": round(float(w["q1"][r, 0] / max(n["q1"][r], 1)), 3),
                                     "hit_q2": round(float(w["q2"][r, 0] / max(n["q2"][r], 1)), 3), "n_A": int(n["A"][r]),
                                     "hit_A": round(float(w["A"][r, 0] / n["A"][r]), 3) if n["A"][r] else np.nan,
                                     "n_B": int(n["B"][r]), "hit_B": round(float(w["B"][r, 0] / n["B"][r]), 3) if n["B"][r] else np.nan})
                    cnt["SA"] += okA.sum(0)
                    cnt["SAB"] += (okA & (nB >= 10) & (w["B"] >= 0.65 * nB)).sum(0)
                    for r in np.where(okA[:, 0])[0]:
                        sel = M[r]
                        yb = y[sel & sets["B"]]
                        passed.append({"family": key, "rule": " & ".join(names[i][1] for i in ch[r]), "k": len(ch[r]),
                                       "n_S": int(nS[r]), "hit_S": round(float(wS[r, 0] / nS[r]), 3),
                                       "hit_q1": round(float(w["q1"][r, 0] / n["q1"][r]), 3), "hit_q2": round(float(w["q2"][r, 0] / n["q2"][r]), 3),
                                       "n_A": int(n["A"][r]), "hit_A": round(float(w["A"][r, 0] / n["A"][r]), 3),
                                       "hit_A_sector": round(float(wsA[r, 0] / n["A"][r]), 3),
                                       "n_B": int(n["B"][r]), "hit_B": round(float(w["B"][r, 0] / n["B"][r]), 3) if n["B"][r] else np.nan,
                                       "med_A": round(float(np.median(y[sel & sets["A"]])), 2),
                                       "med_B": round(float(np.median(yb)), 2) if len(yb) else np.nan,
                                       "tickers_A": int(p.ticker[sel & sets["A"]].nunique()),
                                       "members_A": ",".join(map(str, np.where(sel & sets["A"])[0]))})
                base = {k_: round(float(W[np.where(v)[0], 0].mean()), 3) for k_, v in (("S", S & valid), ("A", sets["A"]), ("B", sets["B"]))}
                summary[key] = {"tested": len(cmb), "conditions": m, "base_hit": base,
                                **{f"real_{k_}": int(v[0]) for k_, v in cnt.items()},
                                **{f"perm_{k_}_mean": round(float(v[1:].mean()), 2) for k_, v in cnt.items()},
                                **{f"perm_{k_}_p95": float(np.percentile(v[1:], 95)) for k_, v in cnt.items()},
                                "perm_SA_share_ge_real": round(float((cnt["SA"][1:] >= cnt["SA"][0]).mean()), 3)}
                print(key, json.dumps(summary[key], ensure_ascii=False), flush=True)
    (BASE / f"results/rules_wide_summary{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(passed).to_csv(BASE / f"results/rules_wide_pass{tag}.csv", index=False)
    t = pd.DataFrame(tops)
    if len(t):
        t = t.sort_values(["family", "hit_S"], ascending=[True, False]).groupby("family").head(15)
    t.to_csv(BASE / f"results/rules_wide_top{tag}.csv", index=False)
    print("passed rows", len(passed))


if __name__ == "__main__":
    main()
