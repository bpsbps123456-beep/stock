"""loss 갈래 분석: results/loss_rows_{5m,1h}.parquet → 종목 횡단 통계, 귀무 분위·p, 종목별 표.

py code/loss_analyze.py → results/loss_stats.csv(이동 k별 횡단 통계), results/loss_tests.csv(실제 vs 귀무),
                          results/loss_focus.csv(MU·BE·NBIS), results/loss_symbols_<base>.csv(실제 종목별), results/loss_summary.json
통계는 모두 '클수록 좋다'로 맞췄다(낙폭은 음수라 클수록 얕다). p = (귀무 ≥ 실제 개수 + 1) / (귀무 수 + 1), 한쪽.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from engine import HERE

OUT = HERE / "results"
FOCUS = ["MU", "BE", "NBIS"]


def simple(df):
    d = df.copy()
    for c in ("ret", "mdd", "st_ret", "st_mdd", "bh_ret", "bh_mdd"):
        d[c] = np.expm1(d[c].astype(np.float64)) * 100.0
    d["ratio"] = d.ret / np.maximum(-d.mdd, 1e-9)
    d["st_ratio"] = d.st_ret / np.maximum(-d.st_mdd, 1e-9)
    d["bh_ratio"] = d.bh_ret / np.maximum(-d.bh_mdd, 1e-9)
    return d


def xsec(g):
    """한 (base, sess, ver, method, k)의 종목 횡단 통계. g는 seg별 행."""
    a = g[g.seg == "all"].set_index("sym")
    c = g[g.seg == "conf"].set_index("sym")
    s = g[g.seg == "sel"].set_index("sym")
    dom = (a.ret >= a.st_ret) & (a.mdd >= a.st_mdd)
    return {
        # ① ② 전체 구간
        "all_ret_med": a.ret.median(), "all_mdd_med": a.mdd.median(), "all_hold_med": a.hold.median(),
        "all_st_ret_med": a.st_ret.median(), "all_st_mdd_med": a.st_mdd.median(),
        "all_bh_ret_med": a.bh_ret.median(), "all_bh_mdd_med": a.bh_mdd.median(),
        # ③ ④ static 대비
        "dom_st_frac": dom.mean(), "dom_st_n": int(dom.sum()),
        "ratio_gt_st_frac": (a.ratio > a.st_ratio).mean(),
        "ret_minus_st_med": (a.ret - a.st_ret).median(),
        "mdd_minus_st_med": (a.mdd - a.st_mdd).median(),  # + = static보다 얕다(타이밍 몫)
        "ret_ge_st_frac": (a.ret >= a.st_ret).mean(), "mdd_ge_st_frac": (a.mdd >= a.st_mdd).mean(),
        "dom_bh_frac": ((a.ret >= a.bh_ret) & (a.mdd >= a.bh_mdd)).mean(),
        "top5_ret_minus_st_mean": (a.ret - a.st_ret).nlargest(5).mean(),
        # ⑤ 확인 구간
        "conf_ret_med": c.ret.median(), "conf_bh_med": c.bh_ret.median(),
        "conf_minus_bh_med": (c.ret - c.bh_ret).median(), "conf_minus_st_med": (c.ret - c.st_ret).median(),
        "conf_dom_st_frac": ((c.ret >= c.st_ret) & (c.mdd >= c.st_mdd)).mean(),
        "sel_ret_med": s.ret.median(), "sel_bh_med": s.bh_ret.median(),
        "sel_minus_bh_med": (s.ret - s.bh_ret).median(), "sel_minus_st_med": (s.ret - s.st_ret).median(),
        "n_syms": int(len(a)),
    }


TEST_STATS = ["all_ret_med", "all_mdd_med", "dom_st_frac", "dom_st_n", "ratio_gt_st_frac", "ret_minus_st_med",
              "mdd_minus_st_med", "conf_ret_med", "conf_minus_bh_med", "conf_minus_st_med", "sel_minus_st_med",
              "top5_ret_minus_st_mean", "all_hold_med"]


def pval(act, nul):
    nul = np.asarray(nul, np.float64)
    return dict(actual=float(act), null_med=float(np.median(nul)), null_q05=float(np.quantile(nul, 0.05)),
                null_q95=float(np.quantile(nul, 0.95)), null_mean=float(nul.mean()),
                q=float(((nul < act).sum() + 0.5 * (nul == act).sum()) / len(nul)),
                p=float(((nul >= act).sum() + 1) / (len(nul) + 1)), K=int(len(nul)))


def self_null_hits(g, sess, ver):
    """종목마다 자기 귀무 안 분위를 교환 가능하게 잰다: 행 집합 U = 실제 ∪ 귀무 이동(day). 각 행 i의 종목별 분위 =
    U의 다른 행 중 i보다 작은 비율. '분위 ≥ 0.95'인 종목 수를 실제 행과 귀무 행들 각각에서 센다(공통 k라 종목 간 상관이 남는다)."""
    a = g[(g.sess == sess) & (g.ver == ver) & (g.seg == "all") & ((g.k == 0) | (g.method == "day"))]
    out = {}
    for nm, f in (("ret_minus_st", lambda d: d.ret - d.st_ret), ("mdd_minus_st", lambda d: d.mdd - d.st_mdd)):
        m = a.assign(v=f(a)).pivot_table(index="k", columns="sym", values="v")  # (K+1, 종목)
        A = m.values
        U = A.shape[0]
        below = np.stack([(A < A[i]).sum(0) for i in range(U)]) / (U - 1)
        hits = (below >= 0.95).sum(1)
        out[nm] = dict(actual=int(hits[0]), null_mean=float(hits[1:].mean()), null_q95=float(np.quantile(hits[1:], 0.95)),
                       p=float(((hits[1:] >= hits[0]).sum() + 1) / U), n_syms=int(A.shape[1]), K=int(U - 1),
                       hit_syms=sorted(m.columns[below[0] >= 0.95].tolist()))
    both = a.assign(dom=(a.ret >= a.st_ret) & (a.mdd >= a.st_mdd)).pivot_table(index="k", columns="sym", values="dom")
    out["dom_syms_actual"] = sorted(both.columns[both.loc[0].astype(bool)].tolist())
    return out


def main():
    stats_rows, test_rows, focus_rows, summary = [], [], [], {}
    for base in ("5m", "1h"):
        g = simple(pd.read_parquet(OUT / f"loss_rows_{base}.parquet"))
        for col in ("sym", "sess", "method", "ver", "seg"):
            g[col] = g[col].astype(str)
        # 실제 종목별 표
        act = g[g.k == 0]
        wide = act.pivot_table(index=["sym", "sess", "ver"], columns="seg",
                               values=["ret", "mdd", "hold", "ratio", "st_ret", "st_mdd", "st_ratio", "bh_ret", "bh_mdd", "ent"])
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        wide.reset_index().to_csv(OUT / f"loss_symbols_{base}.csv", index=False, float_format="%.4f")
        for (sess, ver, method, k), gg in g.groupby(["sess", "ver", "method", "k"], sort=True):
            if k == 0 and method == "roll":
                continue
            r = xsec(gg)
            r.update(base=base, sess=sess, ver=ver, method=method if k else "actual", k=int(k))
            stats_rows.append(r)
        st = pd.DataFrame([r for r in stats_rows if r["base"] == base])
        for sess in ("eth", "rth"):
            for ver in ("ideal", "real"):
                a = st[(st.sess == sess) & (st.ver == ver) & (st.k == 0)].iloc[0]
                for method in ("day", "roll"):
                    nl = st[(st.sess == sess) & (st.ver == ver) & (st.method == method)]
                    for c in TEST_STATS:
                        t = pval(a[c], nl[c].values)
                        t.update(base=base, sess=sess, ver=ver, null=method, stat=c)
                        test_rows.append(t)
                # 종목별 자기 귀무(MU·BE·NBIS)
                for sym in FOCUS:
                    one = g[(g.sym == sym) & (g.sess == sess) & (g.ver == ver)]
                    for seg in ("all", "sel", "conf"):
                        o = one[one.seg == seg]
                        a0 = o[o.k == 0].iloc[0]
                        nl = o[(o.method == "day") & (o.k > 0)]
                        r = dict(base=base, sess=sess, ver=ver, sym=sym, seg=seg, ret=a0.ret, mdd=a0.mdd, hold=a0.hold,
                                 ent=a0.ent, ratio=a0.ratio, st_ret=a0.st_ret, st_mdd=a0.st_mdd, st_ratio=a0.st_ratio,
                                 bh_ret=a0.bh_ret, bh_mdd=a0.bh_mdd, bh_ratio=a0.bh_ratio, K=len(nl),
                                 null_ret_med=nl.ret.median(), null_mdd_med=nl.mdd.median(), null_hold_med=nl.hold.median())
                        for nm, av, nv in (("ret", a0.ret, nl.ret), ("mdd", a0.mdd, nl.mdd),
                                           ("ret_minus_st", a0.ret - a0.st_ret, nl.ret - nl.st_ret),
                                           ("mdd_minus_st", a0.mdd - a0.st_mdd, nl.mdd - nl.st_mdd),
                                           ("ratio", a0.ratio, nl.ratio)):
                            pv = pval(av, nv.values)
                            r[f"q_{nm}"], r[f"p_{nm}"] = pv["q"], pv["p"]
                        focus_rows.append(r)
                summary[f"{base}_{sess}_{ver}_self_null_hits"] = self_null_hits(g, sess, ver)
                # 종목 순위(실제 판 ret − static)
                aa = act[(act.sess == sess) & (act.ver == ver) & (act.seg == "all")].set_index("sym")
                ex = (aa.ret - aa.st_ret).sort_values(ascending=False)
                summary[f"{base}_{sess}_{ver}_rank_ret_minus_st"] = {s: int(ex.index.get_loc(s)) + 1 for s in FOCUS if s in ex.index}
                summary[f"{base}_{sess}_{ver}_top5_ret_minus_st"] = ex.head(5).round(1).to_dict()
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(OUT / "loss_stats.csv", index=False, float_format="%.5f")
    tests = pd.DataFrame(test_rows)
    tests.to_csv(OUT / "loss_tests.csv", index=False, float_format="%.5f")
    focus = pd.DataFrame(focus_rows)
    focus.to_csv(OUT / "loss_focus.csv", index=False, float_format="%.4f")
    meta = json.loads((OUT / "loss_run_meta.json").read_text(encoding="utf-8"))
    summary["scope"] = {b: {k: v for k, v in m.items() if k != "per_sym"} for b, m in meta.items()}
    summary["scope"]["cells"] = "원안 N10·count·follow·b0.5·s0.5·밤 보유 × 세션 eth·rth (5분봉 창 40분, 1시간봉 창 60분)"
    for _, t in tests[tests.null == "day"].iterrows():
        summary.setdefault(f"{t.base}_{t.sess}_{t.ver}", {})[t.stat] = {k: round(float(t[k]), 4) for k in ("actual", "null_med", "null_q05", "null_q95", "q", "p")}
    (OUT / "loss_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["all_ret_med", "all_mdd_med", "all_hold_med", "all_st_ret_med", "all_st_mdd_med", "all_bh_ret_med", "all_bh_mdd_med",
            "dom_st_frac", "dom_st_n", "ratio_gt_st_frac", "ret_minus_st_med", "mdd_minus_st_med", "conf_ret_med", "conf_bh_med",
            "conf_minus_bh_med", "conf_minus_st_med", "sel_minus_st_med"]
    a = stats[stats.k == 0].set_index(["base", "sess", "ver"])[show]
    print("== 실제(k=0) 횡단 통계 ==")
    print(a.round(3).T.to_string())
    print("\n== 귀무(day) 대비 p ==")
    tv = tests[tests.null == "day"].copy()
    tv["cell"] = tv.base + "_" + tv.sess + "_" + tv.ver
    print(tv.pivot_table(index="stat", columns="cell", values="p").round(3).to_string())
    print("\n== 귀무(day) 중앙 ==")
    print(tv.pivot_table(index="stat", columns="cell", values="null_med").round(2).to_string())
    print("\n== 귀무(roll 점검) p ==")
    tr = tests[tests.null == "roll"].copy()
    tr["cell"] = tr.base + "_" + tr.sess + "_" + tr.ver
    print(tr.pivot_table(index="stat", columns="cell", values="p").round(3).to_string())
    print("\n== MU·BE·NBIS (전체 구간) ==")
    f = focus[focus.seg == "all"]
    print(f[["base", "sess", "ver", "sym", "ret", "mdd", "hold", "st_ret", "st_mdd", "bh_ret", "bh_mdd", "null_ret_med", "null_mdd_med",
             "q_ret", "q_mdd", "q_ret_minus_st", "q_mdd_minus_st", "p_ret_minus_st", "p_mdd_minus_st"]].round(2).to_string(index=False))
    print({k: v for k, v in summary.items() if "rank" in k or "top5" in k})
    print("\n== static을 수익·낙폭 모두 이긴 종목 수: 실제 vs 귀무(day) ==")
    d = tv[tv.stat == "dom_st_n"][["cell", "actual", "null_mean", "null_med", "null_q05", "null_q95", "q", "p", "K"]]
    print(d.round(2).to_string(index=False))
    print("\n== 자기 귀무 95분위 이상 종목 수(교환 가능 검정) ==")
    for k, v in summary.items():
        if k.endswith("self_null_hits"):
            print(k, {m: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v[m].items() if kk != "hit_syms"}
                      for m in ("ret_minus_st", "mdd_minus_st")}, "| 둘 다 이긴 종목:", v["dom_syms_actual"])


if __name__ == "__main__":
    main()
