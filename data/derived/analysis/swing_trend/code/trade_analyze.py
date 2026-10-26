"""갈래 trade 집계: trade_run.py 결과 → 판정 지도·고르기·귀무·채택 기준·장세 의존·MU/BE/NBIS.
→ results/trade_map.csv, trade_select.csv, trade_criteria.csv, trade_count.csv, trade_regime.csv, trade_focus.csv, trade_verdict.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import swing

OUT = swing.HERE / "results"
NULL_PERS = ("P1", "P2", "Y25")
FOCUS = ("MU", "BE", "NBIS")


def q(a, p):
    return float(np.nanquantile(a, p))


def main():
    df = pd.read_csv(OUT / "trade_actual.csv")
    z = np.load(OUT / "trade_null.npz")
    names = list(z["names"]); costs = list(z["costs"])
    nul = {p: z[f"{p}_null"] for p in NULL_PERS}            # (종목, 비용, 판정, k, [ex, dd_ex])
    med_null = {p: np.nanmedian(nul[p], axis=0) for p in NULL_PERS}  # (비용, 판정, k, 2)
    syms = {p: list(z[f"{p}_syms"]) for p in NULL_PERS}

    # 1. 판정 지도
    g = df.groupby(["per", "cost", "det"])
    mp = g.agg(kind=("kind", "first"), n=("sym", "size"), ex=("ex", "median"), dd_ex=("dd_ex", "median"),
               ent=("ent", "median"), x=("x", "median"), ret=("ret", "median"), sw_ret=("sw_ret", "median"),
               bh_ret=("bh_ret", "median"), mdd=("mdd", "median"), sw_mdd=("sw_mdd", "median"), bh_mdd=("bh_mdd", "median"),
               win=("ex", lambda s: float((s > 0).mean())), flat=("x", lambda s: float(((s == 0) | (s == 1)).mean())),
               ex_bh=("ret", "median")).reset_index()
    mp["ex_bh"] = g.apply(lambda t: float((t.ret - t.bh_ret).median())).values
    # 판정별 귀무 분위(고르기 없음)
    for col in ("null_q95_ex", "p_ex", "null_q50_ex", "null_q50_dd", "dd_pct"):
        mp[col] = np.nan
    for i, r in mp.iterrows():
        if r.per not in NULL_PERS:
            continue
        ci, di = costs.index(r.cost), names.index(r.det)
        ne = med_null[r.per][ci, di, :, 0]; nd = med_null[r.per][ci, di, :, 1]
        mp.loc[i, ["null_q95_ex", "p_ex", "null_q50_ex", "null_q50_dd", "dd_pct"]] = [
            q(ne, 0.95), float((ne >= r.ex).mean()), q(ne, 0.5), q(nd, 0.5), float((nd <= r.dd_ex).mean())]
    mp.to_csv(OUT / "trade_map.csv", index=False)
    # 판정 종류별 지도: 종류 안 판정 중앙값들의 중앙(범위)
    kd = mp[mp.cost.isin([0.0, 0.0005])].groupby(["per", "cost", "kind"]).agg(
        n_det=("det", "size"), ex=("ex", "median"), ex_min=("ex", "min"), ex_max=("ex", "max"),
        dd_ex=("dd_ex", "median"), ent=("ent", "median"), x=("x", "median"), ex_bh=("ex_bh", "median"),
        n_below_null05=("p_ex", lambda s: int((s >= 0.95).sum())), n_above_null95=("p_ex", lambda s: int((s <= 0.05).sum()))).reset_index()
    kd.to_csv(OUT / "trade_kind.csv", index=False)

    def act(per, cost, det, col="ex"):
        r = mp[(mp.per == per) & (mp.cost == cost) & (mp.det == det)]
        return float(r[col].iloc[0])

    # 2. 고르기(P1 중앙 ex 1위)와 귀무 고르기
    sel_rows, crit_rows, verdict = [], [], {}
    for ci, cost in enumerate(costs):
        p1 = mp[(mp.per == "P1") & (mp.cost == cost)].set_index("det").ex.reindex(names)
        best = p1.idxmax()
        top5 = p1.sort_values(ascending=False).head(5)
        for rank, (d, v) in enumerate(top5.items(), 1):
            sel_rows.append(dict(cost=cost, rank=rank, det=d, P1_ex=v, P2_ex=act("P2", cost, d), Y25_ex=act("Y25", cost, d)))
        dstar = np.argmax(med_null["P1"][ci, :, :, 0], axis=0)  # k1마다 귀무 1위
        verdict[f"sel_{cost}"] = best
        verdict[f"null_sel_dist_{cost}"] = pd.Series([names[i] for i in dstar]).value_counts().head(5).to_dict()
        for lab, det in (("selected", best), ("default", swing.DEFAULT), ("default2", swing.DEFAULT2)):
            di = names.index(det)
            row = dict(cost=cost, role=lab, det=det)
            for per in ("P2", "Y25"):
                if lab == "selected":
                    ne = med_null[per][ci, dstar, :, 0].ravel()   # (k1, k2) 모든 쌍
                    nd = med_null[per][ci, dstar, :, 1].ravel()
                else:
                    ne = med_null[per][ci, di, :, 0]; nd = med_null[per][ci, di, :, 1]
                ex, dd = act(per, cost, det), act(per, cost, det, "dd_ex")
                row.update({f"{per}_ex": ex, f"{per}_null_q50": q(ne, .5), f"{per}_null_q95": q(ne, .95),
                            f"{per}_p": float((ne >= ex).mean()), f"{per}_dd_ex": dd,
                            f"{per}_null_dd_q50": q(nd, .5), f"{per}_null_dd_q05": q(nd, .05),
                            f"{per}_dd_pct": float((nd <= dd).mean()),
                            f"{per}_ret": act(per, cost, det, "ret"), f"{per}_sw": act(per, cost, det, "sw_ret"),
                            f"{per}_bh": act(per, cost, det, "bh_ret"), f"{per}_x": act(per, cost, det, "x"),
                            f"{per}_win": act(per, cost, det, "win")})
            row["P1_ex"] = act("P1", cost, det)
            row["c1"] = bool(row["P2_ex"] > 0 and row["P2_ex"] > row["P2_null_q95"] and row["P2_dd_ex"] >= -0.02)
            row["c2"] = bool(row["Y25_ex"] >= 0)
            crit_rows.append(row)
    sel = pd.DataFrame(sel_rows); crit = pd.DataFrame(crit_rows)
    # 기준 3: 비용 0.05%에서 1 유지(그 비용으로 고르기를 다시 한 것). 비용 0 고른 판정을 0.05%로 평가한 것도 함께.
    c3 = {r.role: bool(r.c1) for r in crit[crit.cost == 0.0005].itertuples()}
    crit["c3"] = crit.role.map(c3)
    sel0 = verdict["sel_0.0"]
    fixed = crit[(crit.cost == 0.0005) & (crit.det == sel0)]
    verdict["sel0_at_0.05_c1"] = bool(fixed.c1.iloc[0]) if len(fixed) else None
    sel.to_csv(OUT / "trade_select.csv", index=False)
    crit.to_csv(OUT / "trade_criteria.csv", index=False)

    # 3. 두 구간(P2·Y25) 모두 같은 비중을 이긴 판정 수: 실제 vs 귀무(k_P2 × k_Y25 모든 쌍, 모든 판정 같은 k)
    cnt_rows = []
    for ci, cost in enumerate(costs):
        a2 = mp[(mp.per == "P2") & (mp.cost == cost)].set_index("det").ex.reindex(names).values > 0
        a5 = mp[(mp.per == "Y25") & (mp.cost == cost)].set_index("det").ex.reindex(names).values > 0
        actual = int((a2 & a5).sum())
        b2 = (med_null["P2"][ci, :, :, 0] > 0).astype(int)    # (판정, k2)
        b5 = (med_null["Y25"][ci, :, :, 0] > 0).astype(int)   # (판정, k5)
        nc = (b2.T @ b5).ravel()
        cnt_rows.append(dict(cost=cost, actual_both=actual, actual_P2=int(a2.sum()), actual_Y25=int(a5.sum()),
                             null_both_mean=float(nc.mean()), null_both_q95=q(nc, .95), p_both=float((nc >= actual).mean()),
                             null_P2_mean=float(b2.sum(0).mean()), null_P2_q95=q(b2.sum(0), .95),
                             null_Y25_mean=float(b5.sum(0).mean()), null_Y25_q95=q(b5.sum(0), .95),
                             p_P2_le_actual=float((b2.sum(0) <= a2.sum()).mean()), p_Y25_ge_actual=float((b5.sum(0) >= a5.sum()).mean()),
                             both_dets=",".join(np.array(names)[a2 & a5])))
    cnt = pd.DataFrame(cnt_rows); cnt.to_csv(OUT / "trade_count.csv", index=False)

    # 4. 장세 의존: 종목 횡단 ex vs 보유 수익
    reg_rows = []
    for (per, cost, det), t in df[df.cost.isin([0.0, 0.0005])].groupby(["per", "cost", "det"]):
        rho = spearmanr(t.ex, t.bh_ret)[0]
        up, dn = t[t.bh_ret > 0], t[t.bh_ret <= 0]
        ter = pd.qcut(t.bh_ret.rank(method="first"), 3, labels=["lo", "mid", "hi"])
        r = dict(per=per, cost=cost, det=det, kind=det.split("_")[0], rho=rho, n_up=len(up), n_dn=len(dn),
                 ex_up=up.ex.median(), ex_dn=dn.ex.median(), win_up=(up.ex > 0).mean(), win_dn=(dn.ex > 0).mean(),
                 ex_bh_up=(up.ret - up.bh_ret).median(), ex_bh_dn=(dn.ret - dn.bh_ret).median(),
                 **{f"ex_{lab}": t.ex[ter == lab].median() for lab in ("lo", "mid", "hi")})
        if per in NULL_PERS:   # 귀무(시점만 섞음)에서도 같은 관계가 나오는지
            ci, di = costs.index(cost), names.index(det)
            bh = t.set_index("sym").bh_ret.reindex(syms[per]).values
            ne = nul[per][:, ci, di, :, 0]
            rhos = [spearmanr(ne[:, k], bh)[0] for k in range(ne.shape[1])]
            r["null_rho_mean"] = float(np.nanmean(rhos))
            r["null_ex_up"] = float(np.nanmedian([np.nanmedian(ne[bh > 0, k]) for k in range(ne.shape[1])]))
            r["null_ex_dn"] = float(np.nanmedian([np.nanmedian(ne[bh <= 0, k]) for k in range(ne.shape[1])]))
        reg_rows.append(r)
    reg = pd.DataFrame(reg_rows); reg.to_csv(OUT / "trade_regime.csv", index=False)

    # 5. MU·BE·NBIS
    keep = {sel0, swing.DEFAULT, swing.DEFAULT2, verdict["sel_0.0005"]}
    foc = df[df.sym.isin(FOCUS)].copy()
    foc.to_csv(OUT / "trade_focus.csv", index=False)

    verdict.update(dict(
        criteria=crit[["cost", "role", "det", "P1_ex", "P2_ex", "P2_null_q95", "P2_p", "P2_dd_ex", "P2_dd_pct", "Y25_ex", "Y25_null_q95", "c1", "c2", "c3"]].to_dict("records"),
        count=cnt.to_dict("records")))
    (OUT / "trade_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
    pct = lambda v: f"{v * 100:+.1f}"
    print("== 고르기 상위 5 ==")
    print(sel.to_string(float_format=pct))
    print("== 채택 기준 ==")
    print(crit[["cost", "role", "det", "P1_ex", "P2_ex", "P2_null_q50", "P2_null_q95", "P2_p", "P2_dd_ex", "P2_null_dd_q50", "P2_dd_pct",
                "Y25_ex", "Y25_null_q50", "Y25_null_q95", "Y25_p", "Y25_dd_ex", "c1", "c2", "c3"]].to_string(float_format=lambda v: f"{v:.3f}"))
    print("sel0 at 0.05:", verdict["sel0_at_0.05_c1"], verdict["null_sel_dist_0.0"])
    print("== 두 구간 모두 이긴 수 ==")
    print(cnt.to_string())
    print("== 지도 (비용 0) ==")
    for per in ("P1", "P2", "Y25", "Y24", "Y23"):
        t = mp[(mp.per == per) & (mp.cost == 0.0)].set_index("det").reindex(names)
        print(per, "n", int(t.n.iloc[0]), "보유 중앙", pct(t.bh_ret.iloc[0]))
        print(t[["ex", "dd_ex", "ent", "x", "ret", "sw_ret", "win", "ex_bh", "p_ex", "dd_pct"]].to_string(float_format=lambda v: f"{v:.3f}"))
    print("== 종류별 ==")
    print(kd[kd.cost == 0.0].to_string(float_format=lambda v: f"{v:.3f}"))
    print("P1 flat share", mp[(mp.per == "P1") & (mp.cost == 0)].set_index("det").flat.round(2).to_dict())
    print("== 장세 의존(비용 0, 기본·고른) ==")
    print(reg[(reg.cost == 0.0) & reg.det.isin(keep)].to_string(float_format=lambda v: f"{v:.3f}"))
    print("== 장세 의존: 28개 판정 rho 요약 ==")
    print(reg[reg.cost == 0.0].groupby("per")[["rho", "null_rho_mean", "ex_up", "ex_dn", "null_ex_up", "null_ex_dn", "win_up", "win_dn"]].median().to_string(float_format=lambda v: f"{v:.3f}"))
    print("== MU·BE·NBIS ==")
    print(foc[foc.det.isin(keep) & foc.cost.isin([0.0, 0.0005])][["sym", "per", "cost", "det", "ret", "mdd", "x", "ent", "sw_ret", "sw_mdd", "bh_ret", "bh_mdd", "ex"]]
          .sort_values(["sym", "cost", "per", "det"]).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
