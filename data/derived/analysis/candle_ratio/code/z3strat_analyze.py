"""z3strat 갈래 집계: BRIEF2 채택 기준 2(전략)·3(1시간봉 방향).

입력 results/z3strat_rows_{5m,1h}.parquet(z3strat_run.py). 모든 값은 단순 수익률(소수), 표시는 %.
- 고르기 1위: 고르기 구간 real 판에서 144종목 중앙 (ret_sel − static_sel)이 가장 큰 전략(원안 제외 84개, 1시간봉 78개).
- 보조 1위: 중앙 (mdd_sel − static_mdd_sel)이 가장 큰(얕은) 전략.
- 기본값: ER K20 θ0.3 A2. 원안: BASE.
- 확인 구간 지표(144 중앙): 수익·낙폭, 같은 비중(static, 전략 자기 평균 보유 비율 x), 원안과의 짝 차이, 종목 비율.
- 귀무: k마다 같은 고르기 절차(real 판 고르기 구간)를 다시 돌려 1위·보조 1위를 뽑고 확인 지표를 잰다. 기본값은 자기 귀무.
  분위 = 귀무 값이 실제보다 작은 비율(같으면 절반).
- 판정 지표(귀무 분위의 주 지표): 확인 구간 중앙 (ret − static). 낙폭 판(mdd − static_mdd), 원안과의 짝 차이도 같이 낸다.

py code/z3strat_analyze.py → results/z3strat_summary.json, z3strat_select_{base}.csv, z3strat_actions_{base}.csv,
                             z3strat_null_{base}.csv, z3strat_syms.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from engine import HERE

OUT = HERE / "results"
DEFAULT = "ER_K20_th0.3|A2"
BASE = "BASE|A3"
SHOW = ["BE", "NBIS", "MU"]


def load(base):
    g = pd.read_parquet(OUT / f"z3strat_rows_{base}.parquet")
    g["sid"] = g.clf.astype(str) + "|" + g.action.astype(str)
    g["ex_sel"] = g.ret_sel - g.static_sel
    g["mex_sel"] = g.mdd_sel - g.static_mdd_sel
    g["ex_conf"] = g.ret_conf - g.static_conf
    g["mex_conf"] = g.mdd_conf - g.static_mdd_conf
    g["ex_all"] = g.ret_all - g.static_all
    g["mex_all"] = g.mdd_all - g.static_mdd_all
    meta = json.loads((OUT / f"z3strat_meta_{base}.json").read_text(encoding="utf-8"))
    return g, meta


def add_base(df):
    """원안(같은 판·같은 k·같은 종목)과의 짝 비교 열."""
    b = df[df.sid == BASE].set_index("sym")
    for w in ("conf", "all", "sel"):
        br = df.sym.map(b[f"ret_{w}"]).astype(float)
        bm = df.sym.map(b[f"mdd_{w}"]).astype(float)
        df[f"dr_{w}"] = df[f"ret_{w}"] - br
        df[f"dm_{w}"] = df[f"mdd_{w}"] - bm
        df[f"bb_{w}"] = ((df[f"ret_{w}"] > br) & (df[f"mdd_{w}"] > bm)).astype(float)  # 원안보다 수익·낙폭 모두 나음
        df[f"bs_{w}"] = ((df[f"ret_{w}"] > df[f"static_{w}"]) & (df[f"mdd_{w}"] > df[f"static_mdd_{w}"])).astype(float)
    return df


MED = ["ret_sel", "mdd_sel", "x_sel", "static_sel", "static_mdd_sel", "ex_sel", "mex_sel", "turn_sel",
       "ret_conf", "mdd_conf", "x_conf", "static_conf", "static_mdd_conf", "ex_conf", "mex_conf", "turn_conf",
       "dr_conf", "dm_conf", "bb_conf", "bs_conf",
       "ret_all", "mdd_all", "x_all", "static_all", "static_mdd_all", "ex_all", "mex_all", "dr_all", "dm_all", "bb_all", "bs_all"]


def per_strategy(df):
    """종목 중앙(비율 열은 평균 = 종목 비율)."""
    frac = [c for c in MED if c.startswith(("bb_", "bs_"))]
    med = [c for c in MED if c not in frac]
    a = df.groupby("sid", observed=True)[med].median()
    b = df.groupby("sid", observed=True)[frac].mean()
    t = a.join(b)
    t["action"] = [s.split("|")[1] for s in t.index]
    return t


def crit2(r, base_r):
    """채택 기준 2의 앞 두 조건(144 중앙 수준 비교, 짝 차이 중앙도 같이)."""
    return dict(
        better_base_level=bool(r.ret_conf > base_r.ret_conf and r.mdd_conf > base_r.mdd_conf),
        better_base_paired=bool(r.dr_conf > 0 and r.dm_conf > 0),
        ge_static_level=bool(r.ret_conf >= r.static_conf and r.mdd_conf >= r.static_mdd_conf),
        ge_static_paired=bool(r.ex_conf >= 0 and r.mex_conf >= 0),
    )


def pick(t):
    c = t.drop(index=BASE)
    return c.ex_sel.idxmax(), c.mex_sel.idxmax()


STAT = ["ex_conf", "mex_conf", "dr_conf", "dm_conf", "ret_conf", "mdd_conf", "bb_conf", "bs_conf", "x_conf"]


def quant(actual, null):
    null = np.asarray(null, float)
    return float(((null < actual).sum() + 0.5 * (null == actual).sum()) / len(null))


def analyze(base, sel_from=None):
    g, meta = load(base)
    boards = list(g.board.cat.categories)
    tabs = {}
    for (bd, k), df in g.groupby(["board", "k"], observed=True):
        tabs[(bd, int(k))] = per_strategy(add_base(df.copy()))
    t0 = tabs[("real", 0)]
    top1, top1m = pick(t0)
    picks = {"top1": top1, "top1_mdd": top1m, "default": DEFAULT, "base": BASE}
    if sel_from:  # 5분봉에서 고른 전략을 그대로(1시간봉 방향 확인)
        for nm, sid in sel_from.items():
            if nm not in ("default", "base") and sid in t0.index:
                picks[f"5m_{nm}"] = sid
    ks = sorted({k for (_, k) in tabs if k != 0})
    res = dict(base=base, D=meta["D"], n_shifts=len(ks), n_syms=meta["n_syms"], n_strat=meta["n_strat"], picks=picks, boards={})
    nulls = []
    for bd in boards:
        t = tabs[(bd, 0)]
        out = {}
        for nm, sid in picks.items():
            r = t.loc[sid]
            d = {c: float(r[c]) for c in MED}
            d["crit2"] = crit2(r, t.loc[BASE]) if sid != BASE else None
            out[nm] = d
        # 귀무: 같은 절차를 k마다(고르기는 real 판)
        for k in ks:
            tk = tabs[(bd, k)]
            p1, p1m = pick(tabs[("real", k)])
            fixed = [(nm, sid) for nm, sid in picks.items() if nm not in ("top1", "top1_mdd", "base")]
            for nm, sid in [("top1", p1), ("top1_mdd", p1m)] + fixed:
                r = tk.loc[sid]
                nulls.append(dict(board=bd, k=k, who=nm, sid=sid, **{c: float(r[c]) for c in STAT},
                                  **{f"c_{kk}": v for kk, v in crit2(r, tk.loc[BASE]).items()}))
        nd = pd.DataFrame([x for x in nulls if x["board"] == bd])
        for nm in [n for n in picks if n != "base"]:
            z = nd[nd.who == nm]
            out[nm]["null"] = {c: dict(q=quant(out[nm][c], z[c]), p50=float(z[c].median()), p95=float(z[c].quantile(0.95)))
                               for c in STAT}
            cc = out[nm]["crit2"]
            out[nm]["null"]["pass_rate"] = {kk: float(z[f"c_{kk}"].mean()) for kk in cc}
            # 기준 2 전체(수준 비교 + 주 지표 귀무 95분위 초과)
            cc["null95_ex"] = bool(out[nm]["ex_conf"] > out[nm]["null"]["ex_conf"]["p95"])
            cc["null95_mex"] = bool(out[nm]["mex_conf"] > out[nm]["null"]["mex_conf"]["p95"])
            cc["pass"] = bool(cc["better_base_level"] and cc["ge_static_level"] and cc["null95_ex"])
        # 84개 중 확인에서 기준 2 앞 두 조건(수준)을 모두 만족한 전략 수 — 실제 대 귀무
        def n_ok(tt):
            b = tt.loc[BASE]
            c = tt.drop(index=BASE)
            return int(((c.ret_conf > b.ret_conf) & (c.mdd_conf > b.mdd_conf) & (c.ret_conf >= c.static_conf) & (c.mdd_conf >= c.static_mdd_conf)).sum())
        cnt = [n_ok(tabs[(bd, k)]) for k in ks]
        out["n_strat_crit2ab"] = dict(actual=n_ok(t), null_p50=float(np.median(cnt)), null_p95=float(np.quantile(cnt, 0.95)), q=quant(n_ok(t), cnt))
        out["bh"] = None
        res["boards"][bd] = out
    # 고르기 표·행동 요약(실제)
    sel_tab = t0.sort_values("ex_sel", ascending=False)
    sel_tab.to_csv(OUT / f"z3strat_select_{base}.csv", float_format="%.5f")
    res["rank_default_sel"] = int(list(sel_tab.drop(index=BASE).index).index(DEFAULT) + 1) if DEFAULT in sel_tab.index else None
    act_rows = []
    for bd in boards:
        t = tabs[(bd, 0)]
        b = t.loc[BASE]
        for a, z in t.drop(index=BASE).groupby("action"):
            row = dict(board=bd, action=a, n=len(z))
            row.update({c: float(z[c].median()) for c in ["x_sel", "ex_sel", "mex_sel", "x_conf", "ret_conf", "mdd_conf", "static_conf",
                                                           "static_mdd_conf", "ex_conf", "mex_conf", "dr_conf", "dm_conf", "bb_conf", "bs_conf", "turn_conf"]})
            row["n_ex_conf_pos"] = int((z.ex_conf > 0).sum())
            row["n_mex_conf_pos"] = int((z.mex_conf > 0).sum())
            row["n_better_base"] = int(((z.ret_conf > b.ret_conf) & (z.mdd_conf > b.mdd_conf)).sum())
            act_rows.append(row)
        act_rows.append(dict(board=bd, action="BASE", n=1, **{c: float(b[c]) for c in ["x_sel", "ex_sel", "mex_sel", "x_conf", "ret_conf", "mdd_conf",
                                                                                         "static_conf", "static_mdd_conf", "ex_conf", "mex_conf", "turn_conf"]}))
    acts = pd.DataFrame(act_rows)
    acts.to_csv(OUT / f"z3strat_actions_{base}.csv", index=False, float_format="%.5f")
    # 행동별 귀무(real 판): 같은 행동 28개 중앙의 ex_conf·mex_conf
    an = []
    for k in ks:
        t = tabs[("real", k)].drop(index=BASE)
        for a, z in t.groupby("action"):
            an.append(dict(k=k, action=a, ex_conf=float(z.ex_conf.median()), mex_conf=float(z.mex_conf.median()),
                           dr_conf=float(z.dr_conf.median()), dm_conf=float(z.dm_conf.median())))
    an = pd.DataFrame(an)
    res["actions_null_real"] = {}
    for a, z in an.groupby("action"):
        r = acts[(acts.board == "real") & (acts.action == a)].iloc[0]
        res["actions_null_real"][a] = {c: dict(actual=float(r[c]), q=quant(r[c], z[c]), p50=float(z[c].median()), p95=float(z[c].quantile(0.95)))
                                       for c in ("ex_conf", "mex_conf", "dr_conf", "dm_conf")}
    pd.DataFrame(nulls).to_csv(OUT / f"z3strat_null_{base}.csv", index=False, float_format="%.5f")
    res["null_top1_ids"] = pd.DataFrame(nulls).query("board=='real' and who=='top1'").sid.value_counts().head(10).to_dict()
    # 보유·원안 전체 구간(2차 대조값)
    gb = add_base(g[(g.board == "real") & (g.k == 0)].copy())
    bb = gb[gb.sid == BASE]
    bh = pd.DataFrame(meta["per_sym"]).T
    res["check_base_real_all"] = dict(ret=float(bb.ret_all.median()), mdd=float(bb.mdd_all.median()), x=float(bb.x_all.median()),
                                      static=float(bb.static_all.median()), static_mdd=float(bb.static_mdd_all.median()),
                                      bh=float(bh.bh_all.astype(float).median()), bh_mdd=float(bh.bh_mdd_all.astype(float).median()),
                                      bh_conf=float(bh.bh_conf.astype(float).median()), bh_mdd_conf=float(bh.bh_mdd_conf.astype(float).median()))
    return res, g, tabs, picks, meta


def sym_table(g, meta, picks, base):
    rows = []
    g0 = g[(g.k == 0) & (g.board == "real") & (g.sym.isin(SHOW))]
    gk = g[(g.board == "real") & (g.sym.isin(SHOW)) & (g.k != 0)]
    for sym in SHOW:
        bh = meta["per_sym"].get(sym)
        if bh is None:
            continue
        for nm, sid in picks.items():
            r = g0[(g0.sym == sym) & (g0.sid == sid)]
            if r.empty:
                continue
            r = r.iloc[0]
            z = gk[(gk.sym == sym) & (gk.sid == sid)]
            row = dict(base=base, sym=sym, who=nm, sid=sid)
            for w in ("all", "conf"):
                row.update({f"ret_{w}": r[f"ret_{w}"], f"mdd_{w}": r[f"mdd_{w}"], f"x_{w}": r[f"x_{w}"],
                            f"static_{w}": r[f"static_{w}"], f"static_mdd_{w}": r[f"static_mdd_{w}"]})
            row["q_ex_all_own_null"] = quant(r.ex_all, z.ex_all) if sid != BASE else None
            rows.append(row)
        rows.append(dict(base=base, sym=sym, who="bh", sid="보유", ret_all=bh["bh_all"], mdd_all=bh["bh_mdd_all"], x_all=1.0,
                         ret_conf=bh["bh_conf"], mdd_conf=bh["bh_mdd_conf"], x_conf=1.0))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    summ = {}
    r5, g5, tabs5, picks5, meta5 = analyze("5m")
    summ["5m"] = r5
    syms = [sym_table(g5, meta5, picks5, "5m")]
    try:
        r1, g1, tabs1, picks1, meta1 = analyze("1h", sel_from=picks5)
        summ["1h"] = r1
        syms.append(sym_table(g1, meta1, picks1, "1h"))
    except FileNotFoundError:
        pass
    pd.concat(syms).to_csv(OUT / "z3strat_syms.csv", index=False, float_format="%.5f")
    (OUT / "z3strat_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({b: dict(picks=s["picks"], check=s["check_base_real_all"]) for b, s in summ.items()}, ensure_ascii=False, indent=1))
