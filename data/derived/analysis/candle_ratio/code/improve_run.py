"""improve 갈래 실행.

py code/improve_run.py grid   → results/improve_grid.parquet (MU + 20종목, 개선안 8개 × 168칸 × 비용 2)
py code/improve_run.py null   → results/improve_null_MU.parquet (MU 일 단위 원형 이동, 가능한 모든 이동)
py code/improve_run.py rand   → results/improve_rand.parquet (무작위 진입 대조, 시드 50)
py code/improve_run.py report → results/improve_summary_*.csv와 화면 요약
py code/improve_run.py base   → results/improve_baselines.parquet (거름만: EMA60 위 보유 E, 일봉 국면 보유 R)
py code/improve_run.py grid1h → results/improve_grid_1h.parquet, improve_null_1h_*MU.parquet
                                (보조: 1시간봉 창 6/1~8/14 → 8/17~10/22, 정규장 60·120·240분)
py code/improve_run.py extra  → 분해·노출·1시간봉 보조 요약
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from engine import HERE, load, ratio
from improve_sim import (BASELINES, BS_I, COST_SENS, NOUT, NS_I, OUT_COLS, SESSES, TFS, VARIANTS, prep,
                         random_control, run_cell)

RES = HERE / "results"
SYMS20 = ["SNDK", "WDC", "STX", "NVDA", "AMD", "AVGO", "TSM", "ARM", "INTC", "LRCX", "AMAT", "KLAC", "MRVL", "DELL",
          "SMCI", "ANET", "ORCL", "PLTR", "TSLA", "QQQ"]
ALL = ["MU"] + SYMS20
ORIG = dict(tf=40, sess="eth", N=10, b=0.5, s=0.5)
KEY = ["tf", "sess", "N", "b", "s"]
WORKERS = 3  # 다른 에이전트와 CPU 공유


def costs_for(sess):
    return (0.0, COST_SENS[sess])


def preps(sym):
    d5 = load(sym, "5m")
    h1 = load(sym, "1h")
    return {(tf, sess): prep(sym, tf, sess, d5=d5, h1=h1) for sess in SESSES for tf in TFS}


def grid_rows(sym, PP, shift_day=0, variants=None):
    variants = variants or list(VARIANTS)
    rows = []
    for (tf, sess), P in PP.items():
        sh = int(P["day_start"][shift_day]) if shift_day else 0
        for n in NS_I:
            p = ratio(P["O"], P["C"], P["V"], n, "count")
            for b, s in BS_I:
                for cost in costs_for(sess):
                    for v in variants:
                        out = run_cell(P, p, v, b, s, cost, shift=sh)
                        rows.append((sym, v, cost, tf, sess, n, b, s, *out, P["bh_sel"], P["bh_conf"]))
    return pd.DataFrame(rows, columns=["sym", "var", "cost", *KEY, *OUT_COLS, "bh_sel", "bh_conf"])


def _grid_job(sym):
    t0 = time.time()
    g = grid_rows(sym, preps(sym))
    print(sym, len(g), round(time.time() - t0, 1), "초", flush=True)
    return g


def cmd_grid():
    with ProcessPoolExecutor(WORKERS) as ex:
        frames = list(ex.map(_grid_job, ALL))
    g = pd.concat(frames, ignore_index=True)
    g.to_parquet(RES / "improve_grid.parquet")
    print("저장", len(g), "행")


def cost_tag(df):
    return np.where(df.cost == 0, "c0", "sens")


def select(g):
    """종목·개선안·비용별 고르기 1위 칸과 상위 20칸 확인 중앙값."""
    g = g.assign(ctag=cost_tag(g))
    best = g.loc[g.groupby(["sym", "var", "ctag"]).ret_sel.idxmax()].copy()
    top = (g.sort_values("ret_sel", ascending=False).groupby(["sym", "var", "ctag"]).head(20)
           .groupby(["sym", "var", "ctag"]).ret_conf.median().rename("top20_conf"))
    return best.join(top, on=["sym", "var", "ctag"])


def _null_job(args):
    d0, d1 = args
    PP = preps("MU")
    out = []
    for d in range(d0, d1):
        g = grid_rows("MU", PP, shift_day=d)
        sb = select(g)
        sb["shift"] = d
        out.append(sb)
    return pd.concat(out, ignore_index=True)


def cmd_null():
    """일 단위 원형 이동: (가) 고른 칸 고정, (나) 이동한 격자에서 다시 고르기(같은 절차)."""
    PP = preps("MU")
    nd = len(PP[(TFS[0], "eth")]["day_start"])
    g0 = pd.read_parquet(RES / "improve_grid.parquet")
    g0 = g0[g0.sym == "MU"]
    chosen = select(g0)
    # (가) 고정 칸
    fixed = []
    for _, r in chosen.iterrows():
        P = PP[(r.tf, r.sess)]
        p = ratio(P["O"], P["C"], P["V"], int(r.N), "count")
        for d in range(1, nd):
            out = run_cell(P, p, r["var"], r.b, r.s, r.cost, shift=int(P["day_start"][d]))
            fixed.append((r["var"], r.ctag, d, out[0], out[1]))
    fixed = pd.DataFrame(fixed, columns=["var", "ctag", "shift", "ret_sel", "ret_conf"])
    fixed.to_parquet(RES / "improve_null_fixed_MU.parquet")
    # (나) 같은 절차
    chunks = [(a, min(a + 23, nd)) for a in range(1, nd, 23)]
    with ProcessPoolExecutor(WORKERS) as ex:
        parts = list(ex.map(_null_job, chunks))
    proc = pd.concat(parts, ignore_index=True)
    proc.to_parquet(RES / "improve_null_MU.parquet")
    print("이동 수", nd - 1, "고정", len(fixed), "절차", len(proc))


def _rand_job(args):
    sym, cells = args
    PP = preps(sym)
    rows = []
    for r in cells:
        P = PP[(r["tf"], r["sess"])]
        p = ratio(P["O"], P["C"], P["V"], int(r["N"]), "count")
        rc = random_control(P, p, r["var"], r["b"], r["s"], r["cost"], r["ent_sel"], r["ent_conf"], seeds=50)
        for row in rc:
            rows.append((sym, r["var"], r["cost"], r["tf"], r["sess"], r["N"], r["b"], r["s"], *row))
    return pd.DataFrame(rows, columns=["sym", "var", "cost", *KEY, "seed", "ret_sel", "ret_conf", "ent_sel", "ent_conf"])


def cmd_rand():
    g = pd.read_parquet(RES / "improve_grid.parquet")
    rv = ["P", "S2", "S3"]
    mu = g[(g.sym == "MU") & g["var"].isin(rv)]
    jobs = []
    # MU는 칸 전체(168 × 3안 × 비용 2)를 세 묶음으로
    recs = mu.to_dict("records")
    for k in range(WORKERS):
        jobs.append(("MU", recs[k::WORKERS]))
    # 20종목은 고르기 1위 칸만
    sb = select(g[g["var"].isin(rv) & (g.sym != "MU")])
    for sym, part in sb.groupby("sym"):
        jobs.append((sym, part.to_dict("records")))
    with ProcessPoolExecutor(WORKERS) as ex:
        parts = list(ex.map(_rand_job, jobs))
    rd = pd.concat(parts, ignore_index=True)
    rd.to_parquet(RES / "improve_rand.parquet")
    print("무작위 대조", len(rd), "행")


# ---------------------------------------------------------------------------
# 보고
# ---------------------------------------------------------------------------

def pct(x):
    return (np.exp(x) - 1.0) * 100.0


def expo_excess(df, seg="conf"):
    """노출 맞춘 초과(단순 %p): exp(ret) − exp(보유비율 × 보유로그수익)."""
    h = df[f"hold_{seg}"] / df[f"bars_{seg}"]
    return pct(df[f"ret_{seg}"]) - pct(h * df[f"bh_{seg}"])


def cmd_report():
    g = pd.read_parquet(RES / "improve_grid.parquet")
    g["ctag"] = cost_tag(g)
    sb = select(g)
    sb["ex_conf"] = expo_excess(sb)
    sb["ex_sel"] = expo_excess(sb, "sel")
    plain = sb[sb["var"] == "P"].set_index(["sym", "ctag"]).ret_conf.rename("plain_conf")
    sb = sb.join(plain, on=["sym", "ctag"])
    pf = sb[sb["var"] == "PF"].set_index(["sym", "ctag"]).ret_conf.rename("pfade_conf")
    sb = sb.join(pf, on=["sym", "ctag"])
    # 같은 칸의 플레인
    gp = g[g["var"] == "P"].set_index(["sym", "ctag", *KEY]).ret_conf.rename("samecell_plain_conf")
    sb = sb.join(gp, on=["sym", "ctag", *KEY])

    # MU 고른 칸 표
    mu = sb[sb.sym == "MU"].copy()
    nf = pd.read_parquet(RES / "improve_null_fixed_MU.parquet")
    npr = pd.read_parquet(RES / "improve_null_MU.parquet")
    rows = []
    for _, r in mu.iterrows():
        f = nf[(nf["var"] == r["var"]) & (nf.ctag == r.ctag)]
        q = npr[(npr["var"] == r["var"]) & (npr.ctag == r.ctag)]
        rows.append(dict(var=r["var"], ctag=r.ctag, cell=f"{r.tf}m {r.sess} N{r.N} {r.b}/{r.s}",
                         sel=pct(r.ret_sel), conf=pct(r.ret_conf), bh_sel=pct(r.bh_sel), bh_conf=pct(r.bh_conf),
                         plain_proc_conf=pct(r.plain_conf), samecell_plain_conf=pct(r.samecell_plain_conf),
                         top20_conf=pct(r.top20_conf), ent=f"{int(r.ent_sel)}/{int(r.ent_conf)}",
                         expo=f"{r.hold_sel / r.bars_sel:.2f}/{r.hold_conf / r.bars_conf:.2f}",
                         ex_conf=r.ex_conf, mdd_conf=pct(r.mdd_conf), stops=f"{int(r.stop_sel)}/{int(r.stop_conf)}",
                         p_fixed=(1 + (f.ret_conf >= r.ret_conf).sum()) / (1 + len(f)),
                         p_proc=(1 + (q.ret_conf >= r.ret_conf).sum()) / (1 + len(q)),
                         proc_q95=pct(q.ret_conf.quantile(0.95)), proc_med=pct(q.ret_conf.median()), nshift=len(f)))
    mut = pd.DataFrame(rows)
    mut.to_csv(RES / "improve_summary_MU.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.1f}".format):
        print("== MU 고르기 1위 → 확인 (단순 %) ==")
        print(mut.to_string(index=False))

    # 원안 칸
    o = g[(g.sym == "MU") & (g.tf == ORIG["tf"]) & (g.sess == ORIG["sess"]) & (g.N == ORIG["N"]) &
          (g.b == ORIG["b"]) & (g.s == ORIG["s"])]
    o = o.assign(sel=pct(o.ret_sel), conf=pct(o.ret_conf), mdd=pct(o.mdd_all))
    print("== MU 원안 칸(40m eth N10 0.5/0.5) ==")
    print(o[["var", "ctag", "sel", "conf", "ent_sel", "ent_conf", "stop_sel", "stop_conf", "mdd"]].round(1).to_string(index=False))

    # 20종목
    oth = sb[sb.sym != "MU"].copy()
    oth["beat_bh"] = oth.ret_conf > oth.bh_conf
    oth["beat_plain"] = oth.ret_conf > oth.plain_conf
    oth["beat_samecell"] = oth.ret_conf > oth.samecell_plain_conf
    oth["beat_bh_both"] = (oth.ret_sel > oth.bh_sel) & (oth.ret_conf > oth.bh_conf)
    agg = oth.groupby(["var", "ctag"]).agg(n=("sym", "size"), beat_bh=("beat_bh", "mean"),
                                           beat_plain=("beat_plain", "mean"), beat_samecell=("beat_samecell", "mean"),
                                           conf_med=("ret_conf", lambda x: pct(x.median())),
                                           bh_med=("bh_conf", lambda x: pct(x.median())),
                                           ex_med=("ex_conf", "median"), top20_med=("top20_conf", lambda x: pct(x.median())))
    # MU 1위 칸을 그대로 쓴 결과
    mucell = sb[sb.sym == "MU"].set_index(["var", "ctag"])[KEY]
    ap = []
    for (v, ct), cell in mucell.iterrows():
        m = g[(g.sym != "MU") & (g["var"] == v) & (g.ctag == ct)]
        for k in KEY:
            m = m[m[k] == cell[k]]
        ap.append(dict(var=v, ctag=ct, mucell_beat_bh=(m.ret_conf > m.bh_conf).mean(),
                       mucell_conf_med=pct(m.ret_conf.median())))
    agg = agg.join(pd.DataFrame(ap).set_index(["var", "ctag"]))
    agg.to_csv(RES / "improve_summary_20.csv", encoding="utf-8-sig")
    oth.to_csv(RES / "improve_selected_20.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.2f}".format):
        print("== 20종목 고르기→확인 ==")
        print(agg.to_string())

    # 칸 전체: 개선안 − 같은 칸 플레인 (21종목)
    gi = g.set_index(["sym", "ctag", *KEY])
    base = gi[gi["var"] == "P"][["ret_sel", "ret_conf"]]
    basef = gi[gi["var"] == "PF"][["ret_sel", "ret_conf"]]
    cw = []
    for v in VARIANTS:
        if v == "P":
            continue
        x = gi[gi["var"] == v][["ret_sel", "ret_conf", "bh_sel", "bh_conf"]]
        ref = basef if v == "V3" else base
        d = x[["ret_sel", "ret_conf"]] - ref
        for sym_grp, dd in (("MU", d.xs("MU", level="sym")), ("20", d.drop("MU", level="sym"))):
            xx = x.xs("MU", level="sym") if sym_grp == "MU" else x.drop("MU", level="sym")
            for ct in ("c0", "sens"):
                ddd = dd.xs(ct, level="ctag")
                xxx = xx.xs(ct, level="ctag")
                cw.append(dict(var=v, ref="PF" if v == "V3" else "P", grp=sym_grp, ctag=ct, cells=len(ddd),
                               better_sel=(ddd.ret_sel > 0).mean(), better_conf=(ddd.ret_conf > 0).mean(),
                               dconf_med=(ddd.ret_conf.median()) * 100,
                               bh_both=((xxx.ret_sel > xxx.bh_sel) & (xxx.ret_conf > xxx.bh_conf)).mean(),
                               bh_conf=(xxx.ret_conf > xxx.bh_conf).mean()))
    for ct in ("c0", "sens"):
        for sym_grp in ("MU", "20"):
            xx = gi[gi["var"] == "P"]
            xx = xx.xs("MU", level="sym") if sym_grp == "MU" else xx.drop("MU", level="sym")
            xxx = xx.xs(ct, level="ctag")
            cw.append(dict(var="P", ref="-", grp=sym_grp, ctag=ct, cells=len(xxx), bh_both=((xxx.ret_sel > xxx.bh_sel) & (xxx.ret_conf > xxx.bh_conf)).mean(),
                           bh_conf=(xxx.ret_conf > xxx.bh_conf).mean()))
    cw = pd.DataFrame(cw)
    cw.to_csv(RES / "improve_summary_cells.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print("== 칸 전체: 같은 칸 플레인 대비(로그%p 중앙값), 두 구간 보유 이김 비율 ==")
        print(cw.to_string(index=False))

    # 무작위 대조
    rd = pd.read_parquet(RES / "improve_rand.parquet")
    rd["ctag"] = cost_tag(rd)
    rm = rd.groupby(["sym", "var", "ctag", *KEY]).agg(r_sel=("ret_sel", "mean"), r_conf=("ret_conf", "mean"),
                                                      r_conf_med=("ret_conf", "median"))
    gg = g[g["var"].isin(["P", "S2", "S3"])].set_index(["sym", "var", "ctag", *KEY])
    j = rm.join(gg[["ret_sel", "ret_conf"]], how="inner")
    # 시드별 백분위
    rdi = rd.set_index(["sym", "var", "ctag", *KEY])
    jj = rdi.join(gg[["ret_conf"]].rename(columns={"ret_conf": "strat_conf"}), how="inner")
    jj["ge"] = jj.ret_conf >= jj.strat_conf
    pv = jj.groupby(level=list(range(8))).ge.mean().rename("p_rand")
    j = j.join(pv)
    j.to_parquet(RES / "improve_rand_cells.parquet")
    mu_j = j.xs("MU", level="sym")
    rows = []
    for v in ["P", "S2", "S3"]:
        for ct in ("c0", "sens"):
            x = mu_j.xs((v, ct), level=("var", "ctag"))
            rows.append(dict(var=v, ctag=ct, cells=len(x), strat_conf_med=pct(x.ret_conf.median()),
                             rand_conf_med=pct(x.r_conf.median()), strat_beats_rand_conf=(x.ret_conf > x.r_conf).mean(),
                             strat_sel_med=pct(x.ret_sel.median()), rand_sel_med=pct(x.r_sel.median()),
                             strat_beats_rand_sel=(x.ret_sel > x.r_sel).mean()))
    rt = pd.DataFrame(rows)
    # 손절이 무작위 진입에도 도움이 되는지: 같은 칸 무작위 평균(S2/S3 − P)
    rp = mu_j.xs("P", level="var")
    for v in ["S2", "S3"]:
        rs = mu_j.xs(v, level="var")
        d = (rs.r_conf - rp.r_conf).dropna()
        ds = (rs.r_sel - rp.r_sel).dropna()
        print(f"MU 무작위 진입에 {v} 손절을 붙인 효과(같은 칸 평균, 로그%p 중앙값): 고르기 {ds.median() * 100:.2f}, 확인 {d.median() * 100:.2f}, 확인 개선 비율 {(d > 0).mean():.2f}")
    rt.to_csv(RES / "improve_summary_rand.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250, "display.float_format", "{:.2f}".format):
        print("== MU 칸 전체 무작위 진입 대조(같은 청산 규칙·같은 진입 수, 시드 50) ==")
        print(rt.to_string(index=False))
        sel_rand = j.reset_index()
        sel_rand = sel_rand.merge(sb[["sym", "var", "ctag", *KEY]], on=["sym", "var", "ctag", *KEY])
        sel_rand.to_csv(RES / "improve_selected_rand.csv", index=False, encoding="utf-8-sig")
        sr = sel_rand.groupby(["var", "ctag"]).agg(n=("sym", "size"), beat_rand=("p_rand", lambda x: (x < 0.5).mean()),
                                                   p05=("p_rand", lambda x: (x <= 0.05).mean()))
        print("== 고른 칸(21종목) 무작위 대조: 확인 구간에서 시드 절반 이상 이김 비율, p≤0.05 비율 ==")
        print(sr.to_string())


def _base_job(sym):
    PP = preps(sym)
    rows = []
    for (tf, sess), P in PP.items():
        ones = np.ones(len(P["O"]))
        for cost in costs_for(sess):
            for v in BASELINES:
                out = run_cell(P, ones, v, 0.5, 0.5, cost)
                rows.append((sym, v, cost, tf, sess, *out, P["bh_sel"], P["bh_conf"]))
    return pd.DataFrame(rows, columns=["sym", "var", "cost", "tf", "sess", *OUT_COLS, "bh_sel", "bh_conf"])


def cmd_base():
    with ProcessPoolExecutor(WORKERS) as ex:
        g = pd.concat(list(ex.map(_base_job, ALL)), ignore_index=True)
    g.to_parquet(RES / "improve_baselines.parquet")
    print("기준", len(g))


TF1H = [60, 120, 240]


def preps1h(sym):
    h1 = load(sym, "1h")
    return {(tf, "rth"): prep(sym, tf, "rth", base_name="1h", d5=h1, h1=h1) for tf in TF1H}


def _grid1h_job(sym):
    return grid_rows(sym, preps1h(sym))


def cmd_grid1h():
    with ProcessPoolExecutor(WORKERS) as ex:
        g = pd.concat(list(ex.map(_grid1h_job, ALL)), ignore_index=True)
    g.to_parquet(RES / "improve_grid_1h.parquet")
    PP = preps1h("MU")
    nd = len(PP[(60, "rth")]["day_start"])
    chosen = select(g[g.sym == "MU"])
    fixed = []
    for _, r in chosen.iterrows():
        P = PP[(r.tf, r.sess)]
        p = ratio(P["O"], P["C"], P["V"], int(r.N), "count")
        for d in range(1, nd):
            out = run_cell(P, p, r["var"], r.b, r.s, r.cost, shift=int(P["day_start"][d]))
            fixed.append((r["var"], r.ctag, d, out[0], out[1]))
    fixed = pd.DataFrame(fixed, columns=["var", "ctag", "shift", "ret_sel", "ret_conf"])
    proc = []
    for d in range(1, nd):
        sb = select(grid_rows("MU", PP, shift_day=d))
        sb["shift"] = d
        proc.append(sb)
    proc = pd.concat(proc, ignore_index=True)
    fixed.to_parquet(RES / "improve_null_1h_fixed_MU.parquet")
    proc.to_parquet(RES / "improve_null_1h_MU.parquet")
    print("1h", len(g), "이동", nd - 1)


def selected_table(g, nf=None, npr=None):
    g = g.assign(ctag=cost_tag(g))
    sb = select(g)
    sb["ex_conf"] = expo_excess(sb)
    plain = sb[sb["var"] == "P"].set_index(["sym", "ctag"]).ret_conf.rename("plain_conf")
    sb = sb.join(plain, on=["sym", "ctag"])
    rows = []
    for _, r in sb[sb.sym == "MU"].iterrows():
        d = dict(var=r["var"], ctag=r.ctag, cell=f"{r.tf}m {r.sess} N{r.N} {r.b}/{r.s}", sel=pct(r.ret_sel),
                 conf=pct(r.ret_conf), bh_sel=pct(r.bh_sel), bh_conf=pct(r.bh_conf), plain=pct(r.plain_conf),
                 ent=f"{int(r.ent_sel)}/{int(r.ent_conf)}", ex_conf=r.ex_conf)
        if nf is not None:
            f = nf[(nf["var"] == r["var"]) & (nf.ctag == r.ctag)]
            q = npr[(npr["var"] == r["var"]) & (npr.ctag == r.ctag)]
            d.update(p_fixed=(1 + (f.ret_conf >= r.ret_conf).sum()) / (1 + len(f)),
                     p_proc=(1 + (q.ret_conf >= r.ret_conf).sum()) / (1 + len(q)),
                     proc_q95=pct(q.ret_conf.quantile(0.95)))
        rows.append(d)
    oth = sb[sb.sym != "MU"].copy()
    oth["b_bh"] = oth.ret_conf > oth.bh_conf
    oth["b_plain"] = oth.ret_conf > oth.plain_conf
    oth["b_both"] = (oth.ret_conf > oth.bh_conf) & (oth.ret_sel > oth.bh_sel)
    grp = oth.groupby(["var", "ctag"])
    agg = pd.DataFrame(dict(n=grp.size(), beat_bh=grp.b_bh.mean(), beat_plain=grp.b_plain.mean(),
                            beat_both_bh=grp.b_both.mean(), conf_med=grp.ret_conf.median().map(pct),
                            bh_med=grp.bh_conf.median().map(pct), ex_med=grp.ex_conf.median()))
    return pd.DataFrame(rows), agg, sb


def cmd_extra():
    g = pd.read_parquet(RES / "improve_grid.parquet")
    g["ctag"] = cost_tag(g)
    bs = pd.read_parquet(RES / "improve_baselines.parquet")
    bs["ctag"] = cost_tag(bs)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    # (1) 거름만 기준 E·R: 14칸(시간봉 7 × 세션 2)에서 고르기 1위 → 확인
    bsel = bs.loc[bs.groupby(["sym", "var", "ctag"]).ret_sel.idxmax()].copy()
    bsel["ex_conf"] = expo_excess(bsel)
    print("== 거름만 기준(E: 시간봉 종가>EMA60이면 보유, R: 전일 종가>일봉 SMA20이면 보유), 14칸에서 고르기 1위 ==")
    m = bsel[bsel.sym == "MU"].copy()
    for a, b in (("sel", "ret_sel"), ("conf", "ret_conf"), ("bhs", "bh_sel"), ("bhc", "bh_conf")):
        m[a] = pct(m[b])
    print(m[["var", "ctag", "tf", "sess", "sel", "conf", "bhs", "bhc", "ent_sel", "ent_conf", "ex_conf"]].round(1).to_string(index=False))
    o = bsel[bsel.sym != "MU"].copy()
    o["b_bh"] = o.ret_conf > o.bh_conf
    og = o.groupby(["var", "ctag"])
    print(pd.DataFrame(dict(beat_bh=og.b_bh.mean(), conf_med=og.ret_conf.median().map(pct),
                            ex_med=og.ex_conf.median())).round(2).to_string())
    # (2) 칸 단위 분해: 비율 신호를 얹은 개선안 vs 거름만(같은 시간봉·세션)
    e = bs.set_index(["sym", "ctag", "tf", "sess", "var"])[["ret_sel", "ret_conf"]]
    rows = []
    for v, ref in (("T1x", "E"), ("T1", "E"), ("V3", "E"), ("D2", "R")):
        x = g[g["var"] == v].join(e.xs(ref, level="var"), on=["sym", "ctag", "tf", "sess"], rsuffix="_ref")
        for grp, xx in (("MU", x[x.sym == "MU"]), ("20", x[x.sym != "MU"])):
            for ct in ("c0", "sens"):
                y = xx[xx.ctag == ct]
                rows.append(dict(var=v, ref=ref, grp=grp, ctag=ct, cells=len(y),
                                 better_sel=(y.ret_sel > y.ret_sel_ref).mean(),
                                 better_conf=(y.ret_conf > y.ret_conf_ref).mean(),
                                 dsel_med=(y.ret_sel - y.ret_sel_ref).median() * 100,
                                 dconf_med=(y.ret_conf - y.ret_conf_ref).median() * 100))
    print("== 칸 단위: 비율 신호를 얹은 개선안 − 거름만 기준(로그%p 중앙값) ==")
    pd.DataFrame(rows).to_csv(RES / "improve_summary_vs_filteronly.csv", index=False, encoding="utf-8-sig")
    bsel.to_csv(RES / "improve_selected_filteronly.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # (3) 칸 단위(비용 0): 노출 맞춘 초과와 전체 구간(고르기+확인) 로그수익의 같은 칸 기준 대비
    g["exc"] = expo_excess(g)
    g["exs"] = expo_excess(g, "sel")
    g["tot"] = g.ret_sel + g.ret_conf
    g["bh_tot"] = g.bh_sel + g.bh_conf
    gi = g[g.ctag == "c0"].set_index(["sym", *KEY])
    rows = []
    for v in VARIANTS:
        ref = "PF" if v == "V3" else "P"
        x = gi[gi["var"] == v]
        r0 = gi[gi["var"] == ref]
        for grp in ("MU", "20"):
            a = x.loc[["MU"]] if grp == "MU" else x.drop("MU", level="sym")
            bb = r0.loc[["MU"]] if grp == "MU" else r0.drop("MU", level="sym")
            rows.append(dict(var=v, ref=ref, grp=grp, dexs_med=(a.exs - bb.exs).median(), dexc_med=(a.exc - bb.exc).median(),
                             better_exc=((a.exc - bb.exc) > 0).mean(), dtot_med=((a.tot - bb.tot) * 100).median(),
                             better_tot=((a.tot - bb.tot) > 0).mean(), beat_bh_tot=(a.tot > a.bh_tot).mean(),
                             expo_sel=(a.hold_sel / a.bars_sel).median(), expo_conf=(a.hold_conf / a.bars_conf).median()))
    print("== 칸 단위(비용 0): 노출 맞춘 초과(단순 %p)·전체 구간 로그수익(로그%p)의 같은 칸 기준 대비 ==")
    pd.DataFrame(rows).to_csv(RES / "improve_summary_exposure.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # (4) 1시간봉 보조
    g1 = pd.read_parquet(RES / "improve_grid_1h.parquet")
    nf = pd.read_parquet(RES / "improve_null_1h_fixed_MU.parquet")
    npr = pd.read_parquet(RES / "improve_null_1h_MU.parquet")
    mt, agg, _ = selected_table(g1, nf, npr)
    mt.to_csv(RES / "improve_summary_1h_MU.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(RES / "improve_summary_1h_20.csv", encoding="utf-8-sig")
    print("== 보조: 1시간봉 창(정규장 60·120·240분, 안마다 36칸) MU 고르기 6/1~8/14 → 확인 8/17~10/22 ==")
    print(mt.round(2).to_string(index=False))
    print(agg.round(2).to_string())
    # (5) 두 구간 모두 보유를 이긴 칸이 종목 사이에 얼마나 겹치나(5분봉 창, 비용 0)
    g0 = g[g.ctag == "c0"].copy()
    g0["both"] = (g0.ret_sel > g0.bh_sel) & (g0.ret_conf > g0.bh_conf)
    cnt = g0.groupby(["var", *KEY]).both.sum().rename("n_sym").reset_index()
    print("== 두 구간 모두 보유를 이긴 종목 수(21종목 중) 상위 칸 ==")
    print(cnt.sort_values("n_sym", ascending=False).head(12).to_string(index=False))
    print(cnt.groupby("var").n_sym.describe()[["mean", "50%", "max"]].round(2).to_string())


if __name__ == "__main__":
    {"grid": cmd_grid, "null": cmd_null, "rand": cmd_rand, "report": cmd_report, "base": cmd_base,
     "grid1h": cmd_grid1h, "extra": cmd_extra}[sys.argv[1]]()
