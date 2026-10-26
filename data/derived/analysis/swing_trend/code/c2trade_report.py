"""갈래 c2trade 집계: c2trade_run.py 결과 → 전략 표·귀무 분위·채택 기준 2~4·선택 통계·M5 시점 진단·MU/BE/NBIS.
→ results/c2trade_summary.csv, c2trade_criteria.csv, c2trade_focus.csv, c2trade_m5diag.csv, c2trade_verdict.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import swing
from events import ETF

OUT = swing.HERE / "results"
FOCUS = ("MU", "BE", "NBIS")
STRATS = ("hold", "zz", "flip", "meta40", "meta20", "meta60", "lg_a", "lg_b", "and_a", "and_b")
TESTED = ("meta40", "meta20", "meta60", "lg_a", "lg_b", "and_a", "and_b")


def q(a, p):
    return float(np.nanquantile(a, p))


def main():
    act = pd.read_csv(OUT / "c2trade_actual.csv")
    z = np.load(OUT / "c2trade_null.npz")
    costs = list(z["costs"]); wins = list(z["wins"]); mstr = list(z["model_strats"])
    sel = pd.read_csv(OUT / "c2trade_select.csv")

    # 1. 전략 표(주식 묶음 기본, 전체 141은 메타 민감도)
    hold = act[act.strat == "zz"].copy()
    hold["strat"] = "hold"; hold["ret"] = hold.bh_ret; hold["mdd"] = hold.bh_mdd; hold["x"] = 1.0; hold["ent"] = 0
    hold["sw_ret"] = hold.bh_ret; hold["sw_mdd"] = hold.bh_mdd
    hold["ex_sw"] = 0.0; hold["ex_zz"] = hold.bh_ret - hold.zz_ret; hold["ex_bh"] = 0.0; hold["dd_sw"] = 0.0
    full = pd.concat([act, hold])
    rows = []
    for uni, t in (("stock", full[full.stock]), ("all", full)):
        for (per, cost, strat), g in t.groupby(["per", "cost", "strat"]):
            rows.append(dict(uni=uni, per=per, cost=cost, strat=strat, n=len(g), ret=g.ret.median(), sw=g.sw_ret.median(),
                             zz=g.zz_ret.median(), bh=g.bh_ret.median(), ex_sw=g.ex_sw.median(), ex_zz=g.ex_zz.median(),
                             ex_bh=g.ex_bh.median(), mean_ex_sw=g.ex_sw.mean(), mean_ex_zz=g.ex_zz.mean(),
                             win_sw=float((g.ex_sw > 0).mean()), tie_sw=float((g.ex_sw.abs() < 1e-12).mean()),
                             win_zz=float((g.ex_zz > 0).mean()), mdd=g.mdd.median(), zz_mdd=g.zz_mdd.median(), bh_mdd=g.bh_mdd.median(),
                             dd_sw=g.dd_sw.median(), x=g.x.median(), x_mean=g.x.mean(), x0=float((g.x == 0).mean()), ent=g.ent.median()))
    sm = pd.DataFrame(rows)

    # 2. 귀무 분위
    def nq(arr, actual, lo_good=False):
        arr = np.asarray(arr, float).ravel()
        return dict(q50=q(arr, .5), q95=q(arr, .95), q05=q(arr, .05),
                    p=float((arr <= actual).mean()) if lo_good else float((arr >= actual).mean()), nn=int(np.isfinite(arr).sum()))

    nul_rows = []
    for r in sm.itertuples():
        if r.strat not in TESTED or (r.uni == "all" and not r.strat.startswith("meta")):
            continue
        ci = costs.index(r.cost)
        srcs = []
        if r.strat.startswith("meta"):
            wi = wins.index(int(r.strat[4:])); ui = 0 if r.uni == "stock" else 1
            for kind in ("in", "full"):
                srcs.append((kind, {m: z[f"meta_{kind}_{r.per}_{m}"][ui, ci, wi] for m in ("ex_sw", "ex_zz", "dd_sw")}))
        else:
            mi = mstr.index(r.strat)
            for kind in ("perm", "circ"):
                key = f"{kind}_{r.per}_ex_sw"
                if key in z.files:
                    srcs.append((kind, {m: z[f"{kind}_{r.per}_{m}"][ci, mi] for m in ("ex_sw", "ex_zz", "dd_sw")}))
        for kind, d in srcs:
            a, b, c = nq(d["ex_sw"], r.ex_sw), nq(d["ex_zz"], r.ex_zz), nq(d["dd_sw"], r.dd_sw, lo_good=True)
            nul_rows.append(dict(uni=r.uni, per=r.per, cost=r.cost, strat=r.strat, null=kind, nn=a["nn"],
                                 sw_q50=a["q50"], sw_q95=a["q95"], sw_p=a["p"], zz_q50=b["q50"], zz_q95=b["q95"], zz_p=b["p"],
                                 dd_q50=c["q50"], dd_q05=c["q05"], dd_pct=c["p"]))
    nl = pd.DataFrame(nul_rows)
    sm.to_csv(OUT / "c2trade_summary.csv", index=False)
    nl.to_csv(OUT / "c2trade_nullq.csv", index=False)

    # 3. 채택 기준 2~4 (주식 묶음). 주 귀무: 메타는 구간 안 원형 이동, 거름·전환은 날짜 블록 순열.
    def get(per, cost, strat, col, uni="stock"):
        t = sm[(sm.uni == uni) & (sm.per == per) & (sm.cost == cost) & (sm.strat == strat)]
        return float(t[col].iloc[0])

    def getn(per, cost, strat, kind, col):
        t = nl[(nl.uni == "stock") & (nl.per == per) & (nl.cost == cost) & (nl.strat == strat) & (nl.null == kind)]
        return float(t[col].iloc[0]) if len(t) else np.nan

    crit = []
    for strat in TESTED:
        prim, alt = ("in", "full") if strat.startswith("meta") else ("perm", "circ")
        row = dict(strat=strat, null=prim, alt_null=alt)
        for cost in costs:
            tag = "c0" if cost == 0 else "c5"
            ex_sw, ex_zz = get("P2", cost, strat, "ex_sw"), get("P2", cost, strat, "ex_zz")
            q_sw, q_zz = getn("P2", cost, strat, prim, "sw_q95"), getn("P2", cost, strat, prim, "zz_q95")
            aq_sw, aq_zz = getn("P2", cost, strat, alt, "sw_q95"), getn("P2", cost, strat, alt, "zz_q95")
            row.update({f"P2_ex_sw_{tag}": ex_sw, f"P2_ex_zz_{tag}": ex_zz, f"P2_null_sw_q95_{tag}": q_sw, f"P2_null_zz_q95_{tag}": q_zz,
                        f"P2_p_sw_{tag}": getn("P2", cost, strat, prim, "sw_p"), f"P2_p_zz_{tag}": getn("P2", cost, strat, prim, "zz_p"),
                        f"P2_alt_sw_q95_{tag}": aq_sw, f"P2_alt_zz_q95_{tag}": aq_zz,
                        f"P2_alt_p_sw_{tag}": getn("P2", cost, strat, alt, "sw_p"), f"P2_alt_p_zz_{tag}": getn("P2", cost, strat, alt, "zz_p"),
                        f"pass2_{tag}": bool(ex_sw > 0 and ex_zz > 0 and ex_sw > q_sw and ex_zz > q_zz),
                        f"pass2_alt_{tag}": bool(ex_sw > 0 and ex_zz > 0 and ex_sw > aq_sw and ex_zz > aq_zz)})
        row["P2_dd_sw"] = get("P2", 0.0, strat, "dd_sw")
        row["P1_ex_sw"] = get("P1", 0.0, strat, "ex_sw"); row["P1_ex_zz"] = get("P1", 0.0, strat, "ex_zz")
        row["Y25_ex_sw"] = get("Y25", 0.0, strat, "ex_sw"); row["Y25_ex_zz"] = get("Y25", 0.0, strat, "ex_zz")
        row["Y25_p_sw"] = getn("Y25", 0.0, strat, prim, "sw_p")
        row["pass3"] = bool(row["Y25_ex_sw"] > 0 and row["Y25_ex_zz"] > 0)
        row["pass4"] = row["pass2_c5"]
        row["Y24_ex_sw"] = get("Y24", 0.0, strat, "ex_sw"); row["Y23_ex_sw"] = get("Y23", 0.0, strat, "ex_sw")
        row["all_pass_2_3_4"] = bool(row["pass2_c0"] and row["pass3"] and row["pass4"])
        crit.append(row)
    cr = pd.DataFrame(crit)
    cr.to_csv(OUT / "c2trade_criteria.csv", index=False)

    # 4. 선택 통계(기준 1 참고): P2 이어짐 대 아님 y1 차이, 귀무 분위
    ss_rows = []
    real = sel[sel.src == "real"]
    for r in real.itertuples():
        row = r._asdict(); row.pop("Index")
        for src in ("perm", "circ"):
            t = sel[(sel.src == src) & (sel.per == r.per) & (sel.type == r.type) & (sel.model == r.model)]
            if len(t):
                row[f"{src}_q95"] = q(t["diff"], .95); row[f"{src}_p"] = float((t["diff"] >= r.diff).mean())
                row[f"{src}_nsel_q50"] = q(t.n_sel, .5)
        ss_rows.append(row)
    ss = pd.DataFrame(ss_rows)
    ss.to_csv(OUT / "c2trade_select_stats.csv", index=False)

    # 5. M5 시점 진단: 메타 부호(d−1 종가)별 다음 날 '추세 − 뒤집기' 시가→시가 손익(포함 주식 평균)
    D = swing.load()
    stocks = sorted(set(D) - ETF)
    import c2trade_run as R
    ref = D["SPY"]
    cal = sorted(set().union(*[set(D[s].date) for s in stocks]))
    ix = {d: i for i, d in enumerate(cal)}
    info = {}
    for s in stocks:
        g = D[s]
        O, H, L, C = (g[k].values.astype(np.float64) for k in ("O", "H", "L", "C"))
        st = swing.zigzag(C, 0.0, 3.0 * swing.atr(H, L, C, 20))[0]
        info[s] = dict(st=st, c=C, o=O, rows=np.array([ix[d] for d in g.date]))
    M5 = {w: R.m5_series(stocks, D, info, cal, w, int(round(w * 0.75))) for w in wins}
    diag = []
    for per in ("P1", "P2", "Y25", "Y24", "Y23"):
        Dp = len(swing.period_index(ref, per))
        inc = [s for s in stocks if (lambda idx: len(idx) == Dp and idx[0] >= 60 and D[s].date.iloc[-1] >= swing.PERIODS[per][1])(swing.period_index(D[s], per))]
        tf = np.full((Dp, len(inc)), np.nan)
        for k, s in enumerate(inc):
            I = info[s]; idx = swing.period_index(D[s], per); o, c, st = I["o"], I["c"], I["st"]
            for t, j in enumerate(idx):
                r = (o[j + 1] / o[j] - 1) if t < Dp - 1 else (c[j] / o[j] - 1)
                tf[t, k] = (2 * st[j - 1] - 1) * r
        daily = np.nanmean(tf, axis=1)
        crow = np.array([ix[d] for d in ref.date.values[swing.period_index(ref, per)]])
        for w in wins:
            sg = M5[w][crow - 1] > 0
            diag.append(dict(per=per, w=w, days=Dp, trend_days=int(sg.sum()),
                             tf_trend_mode=float(daily[sg].mean()) if sg.any() else np.nan,
                             tf_flip_mode=float(daily[~sg].mean()) if (~sg).any() else np.nan,
                             tf_all=float(daily.mean()), hit=float(((daily > 0) == sg).mean()),
                             sum_meta=float(np.sum(np.where(sg, daily, -daily))), sum_trend=float(daily.sum())))
    dg = pd.DataFrame(diag)
    dg.to_csv(OUT / "c2trade_m5diag.csv", index=False)

    # 6. MU·BE·NBIS
    foc = full[full.sym.isin(FOCUS) & full.strat.isin(STRATS)][["sym", "per", "cost", "strat", "ret", "mdd", "x", "ent", "sw_ret", "bh_ret", "zz_ret", "ex_sw", "ex_zz"]]
    foc = foc.sort_values(["sym", "cost", "per", "strat"])
    foc.to_csv(OUT / "c2trade_focus.csv", index=False)

    models = json.loads((OUT / "c2trade_models.json").read_text(encoding="utf-8"))
    verdict = dict(
        criteria=cr.to_dict("records"),
        any_pass_2_3_4=bool(cr.all_pass_2_3_4.any()),
        n_tested=len(TESTED), n_cost=len(costs),
        select_P2=ss[ss.per == "P2"].to_dict("records"),
        m5diag=dg.to_dict("records"),
        models=models["models"], trend_mode_share=models["trend_mode_share"],
        missing=models["missing_pred_confirms_in_period"], p1_leak=models["p1_events_exit_after_p1"])
    (OUT / "c2trade_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)
    f3 = lambda v: f"{v:.3f}"
    print("== 전략 표(주식, 비용 0) ==")
    for per in ("P1", "P2", "Y25", "Y24", "Y23"):
        t = sm[(sm.uni == "stock") & (sm.per == per) & (sm.cost == 0)].set_index("strat").reindex(STRATS)
        print(per, "n", int(t.n.iloc[0]))
        print(t[["ret", "sw", "zz", "bh", "ex_sw", "ex_zz", "ex_bh", "mean_ex_sw", "win_sw", "tie_sw", "win_zz", "mdd", "dd_sw", "x", "x0", "ent"]].to_string(float_format=f3))
    print("== 전체 141(메타 민감도), 비용 0 ==")
    print(sm[(sm.uni == "all") & (sm.cost == 0) & sm.strat.isin(["zz", "flip", "meta40", "meta20", "meta60"])][["per", "strat", "n", "ex_sw", "ex_zz", "x"]].to_string(float_format=f3))
    print("== 귀무 ==")
    print(nl[nl.uni == "stock"].to_string(float_format=f3))
    print(nl[nl.uni == "all"][["per", "cost", "strat", "null", "sw_q95", "sw_p", "zz_p"]].to_string(float_format=f3))
    print("== 채택 기준 ==")
    print(cr.T.to_string())
    print("== 선택 통계 ==")
    print(ss.to_string(float_format=f3))
    print("== M5 시점 진단 ==")
    print(dg.to_string(float_format=lambda v: f"{v:.4f}"))
    print("== MU·BE·NBIS(비용 0) ==")
    print(foc[foc.cost == 0].to_string(float_format=f3))


if __name__ == "__main__":
    main()
