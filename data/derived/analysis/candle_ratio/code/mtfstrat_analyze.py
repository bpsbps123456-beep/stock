"""BRIEF3 채택 기준 1~3 판정과 시간봉 조합 지도 (갈래 strat).

py code/mtfstrat_analyze.py
→ results/mtfstrat_cells.csv      (b, z배수, 판정, 행동) 칸별 고르기·확인 중앙과 칸별 귀무 분위 (실제·비용 0 판)
→ results/mtfstrat_select.csv     판·k마다 고르기 1위와 그 확인 값(귀무 고르기 절차 포함)
→ results/mtfstrat_family.csv     '두 구간 모두 같은 비중 이김' 칸 수 등 계열 수, 실제 대 귀무
→ results/mtfstrat_base_tf.csv    원안(A3) 시간봉별 고르기·확인 (판 3개)
→ results/mtfstrat_summary.json   기준 1~3 판정 숫자
모든 값은 단순 수익(%가 아닌 비율). 144종목 중앙은 짝지은 차이의 중앙이다.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parents[1]
RES = HERE / "results"
DEFAULT = (40, "z160_AGREE", "A0")
BASE_TFS = [10, 20, 30, 40, 60, 90, 120, 180, 240]
MULTS = ["1b", "2b", "4b", "8b", "D"]
KEY = ["board", "k", "b", "clf", "action"]
MET = ["ds", "dc", "dsb", "dcb", "mms", "mmc", "mbs", "mbc"]
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def zmult(b, clf):
    if clf == "BASE":
        return "-"
    z = clf.split("_")[0][1:]
    return "D" if z == "D" else f"{int(z) // b}b"


def load():
    g = pd.read_parquet(RES / "mtf_rows.parquet")
    for c in ("clf", "action", "board", "sym"):
        g[c] = g[c].astype(str)
    base = g[g.action == "A3"][["sym", "b", "board", "k", "ret_sel", "ret_conf", "mdd_sel", "mdd_conf"]]
    base = base.rename(columns={c: c + "_B" for c in ("ret_sel", "ret_conf", "mdd_sel", "mdd_conf")})
    g = g.merge(base, on=["sym", "b", "board", "k"], how="left", validate="many_to_one")
    f = lambda c: g[c].astype(np.float64)
    g["ds"] = f("ret_sel") - f("static_sel")
    g["dc"] = f("ret_conf") - f("static_conf")
    g["dsb"] = f("ret_sel") - f("ret_sel_B")
    g["dcb"] = f("ret_conf") - f("ret_conf_B")
    g["mms"] = f("mdd_sel") - f("static_mdd_sel")
    g["mmc"] = f("mdd_conf") - f("static_mdd_conf")
    g["mbs"] = f("mdd_sel") - f("mdd_sel_B")
    g["mbc"] = f("mdd_conf") - f("mdd_conf_B")
    return g


def medians(g):
    agg = {m: "median" for m in MET}
    agg.update({c: "median" for c in ("ret_sel", "ret_conf", "mdd_sel", "mdd_conf", "static_sel", "static_conf",
                                      "static_mdd_sel", "static_mdd_conf", "x_sel", "x_conf", "turn_sel", "turn_conf",
                                      "bh_sel", "bh_conf", "ret_conf_B", "mdd_conf_B")})
    M = g.groupby(KEY, sort=False).agg(agg)
    pos = g[KEY + ["ds", "dc", "dcb", "mmc", "mbc"]].copy()
    for m in ("ds", "dc", "dcb", "mmc", "mbc"):
        pos[m] = (pos[m] > 0).astype(float) if m in ("dcb", "mbc") else (pos[m] >= 0).astype(float)
    P = pos.groupby(KEY, sort=False).mean().add_prefix("f_")
    M = M.join(P).reset_index()
    M["m"] = [zmult(b, c) for b, c in zip(M.b, M.clf)]
    M["kind"] = [c.split("_")[1] if c != "BASE" else "BASE" for c in M.clf]
    return M


def trend_share():
    d = pd.read_parquet(RES / "mtf_diag.parquet")
    d = d[(d.k == 0) & (d.board == "real0")].copy()
    for w in ("sel", "conf"):
        d[f"ts_{w}"] = d[f"trend_{w}_n"] / (d[f"trend_{w}_n"] + d[f"amb_{w}_n"])
    d["clf"] = d.clf.astype(str)
    t = d.groupby(["b", "clf"])[["ts_sel", "ts_conf", "trend_sel", "amb_sel", "trend_conf", "amb_conf"]].median().reset_index()
    return t


def qpos(val, null):
    null = np.asarray(null, float)
    return float(np.mean(null < val) + 0.5 * np.mean(null == val))


def pct(x, d=1):
    return f"{100 * x:+.{d}f}"



def alt_null(g, M, path):
    """AGREE 칸의 귀무(k>0)를 z 상태 이동 귀무로 바꾼 중앙 표."""
    a = pd.read_parquet(path)
    for c in ("clf", "action", "board", "sym"):
        a[c] = a[c].astype(str)
    base = g[(g.action == "A3")][["sym", "b", "board", "k", "ret_sel_B", "ret_conf_B", "mdd_sel_B", "mdd_conf_B"]]
    base = base[base.k == 0].drop(columns="k")
    a = a.merge(base, on=["sym", "b", "board"], how="left", validate="many_to_one")
    f = lambda c: a[c].astype(np.float64)
    a["ds"] = f("ret_sel") - f("static_sel"); a["dc"] = f("ret_conf") - f("static_conf")
    a["dsb"] = f("ret_sel") - f("ret_sel_B"); a["dcb"] = f("ret_conf") - f("ret_conf_B")
    a["mms"] = f("mdd_sel") - f("static_mdd_sel"); a["mmc"] = f("mdd_conf") - f("static_mdd_conf")
    a["mbs"] = f("mdd_sel") - f("mdd_sel_B"); a["mbc"] = f("mdd_conf") - f("mdd_conf_B")
    A = medians(a)
    chk = A[A.k == 0].merge(M[M.k == 0], on=KEY, suffixes=("", "_o"))
    say(f"z 이동 귀무 k=0 재현: 칸 {len(chk)}, 최대 차이 dc {np.abs(chk.dc - chk.dc_o).max():.1e}")
    extra = [c for c in M.columns if c not in A.columns]
    A = A[A.k > 0].merge(M[KEY + extra], on=KEY, how="left")
    keep = ~((M.k > 0) & (M.kind == "AGREE") & M.board.isin(["real", "real0"]))
    M2 = pd.concat([M[keep], A[M.columns]], ignore_index=True)
    t = A.groupby(["board", "b", "clf", "action"]).turn_conf.median().rename("turn_alt")
    t0 = M[(M.k > 0) & (M.kind == "AGREE")].groupby(["board", "b", "clf", "action"]).turn_conf.median().rename("turn_null")
    t1 = M[(M.k == 0) & (M.kind == "AGREE")].set_index(["board", "b", "clf", "action"]).turn_conf.rename("turn_actual")
    tt = pd.concat([t1, t0, t], axis=1).reset_index()
    tt = tt[tt.board == "real"]
    say("AGREE 칸 확인 회전(종목 중앙의 칸 중앙): 실제 %.0f, 원래 귀무 %.0f, z 이동 귀무 %.0f" % (tt.turn_actual.median(), tt.turn_null.median(), tt.turn_alt.median()))
    say("기본값 b40 z160_AGREE A0 회전: " + str(tt[(tt.b == 40) & (tt.clf == "z160_AGREE") & (tt.action == "A0")][["turn_actual", "turn_null", "turn_alt"]].values.tolist()))
    return M2


def evaluate(M, ks, tag="", maps=True):
    say(f"\n######## 귀무 판: {tag or '원래(mtf_run flag 이동)'} ########")
    # ---------- 고르기 절차 ----------
    Z = M[M.action != "A3"]
    sel_rows = []
    for bd in ("real", "real0"):
        for k in [0] + ks:
            x = Z[(Z.board == bd) & (Z.k == k)]
            r = x.loc[x.ds.idxmax()].copy()
            r["pick"] = "top1"
            sel_rows.append(r)
            r2 = x[(x.b == DEFAULT[0]) & (x.clf == DEFAULT[1]) & (x.action == DEFAULT[2])].iloc[0].copy()
            r2["pick"] = "default"
            sel_rows.append(r2)
            xa = M[(M.board == bd) & (M.k == k)]
            r3 = xa.loc[xa.ds.idxmax()].copy()
            r3["pick"] = "top1_all335"
            sel_rows.append(r3)
    S = pd.DataFrame(sel_rows)
    S.to_csv(RES / f"mtfstrat_select{tag}.csv", index=False)

    summ = {}
    say("\n[기준 1·2: 고르기 1위·기본값의 확인 구간, 144 중앙]")
    for bd in ("real", "real0"):
        for pick in ("top1", "default", "top1_all335"):
            s = S[(S.board == bd) & (S.pick == pick)]
            a = s[s.k == 0].iloc[0]
            nl = s[s.k > 0]
            q = {m: qpos(a[m], nl[m]) for m in ("dc", "dcb", "mmc", "mbc")}
            n95 = float(np.quantile(nl.dc, 0.95))
            key = f"{bd}_{pick}"
            summ[key] = dict(
                strategy=f"b{a.b} {a.clf} {a.action}", m=a.m, degen=bool(a.degen),
                sel_ret_minus_static=a.ds, conf_ret_minus_static=a.dc, conf_ret_minus_base=a.dcb,
                conf_mdd_minus_static=a.mmc, conf_mdd_minus_base=a.mbc,
                conf_ret=a.ret_conf, conf_mdd=a.mdd_conf, conf_static=a.static_conf, conf_static_mdd=a.static_mdd_conf,
                conf_base_ret=a.ret_conf_B, conf_base_mdd=a.mdd_conf_B, x_conf=a.x_conf, turn_conf=a.turn_conf,
                ts_sel=a.ts_sel, ts_conf=a.ts_conf,
                frac_ret_ge_static=a.f_dc, frac_ret_gt_base=a.f_dcb, frac_mdd_ge_static=a.f_mmc, frac_mdd_gt_base=a.f_mbc,
                null_q=q, null_dc_median=float(nl.dc.median()), null_dc_p95=n95,
                null_picks=nl.apply(lambda r: f"b{r.b} {r.clf} {r.action}", axis=1).value_counts().head(5).to_dict(),
                pass_vs_base=bool(a.dcb > 0 and a.mbc > 0), pass_vs_static=bool(a.dc >= 0 and a.mmc >= 0),
                pass_null95=bool(a.dc > n95))
            summ[key]["pass_all"] = summ[key]["pass_vs_base"] and summ[key]["pass_vs_static"] and summ[key]["pass_null95"]
            d = summ[key]
            say(f" {bd:5s} {pick:11s} {d['strategy']}{' [퇴화]' if d['degen'] else ''}: 고르기 {pct(a.ds)} | 확인 수익−같은비중 {pct(a.dc)}"
                f" (종목 {a.f_dc:.0%}≥, 귀무 분위 {q['dc']:.2f}, 귀무 95분위 {pct(n95)}) | 수익−원안 {pct(a.dcb)} ({a.f_dcb:.0%}>, q {q['dcb']:.2f})"
                f" | 낙폭−같은비중 {pct(a.mmc)} ({a.f_mmc:.0%}≥, q {q['mmc']:.2f}) | 낙폭−원안 {pct(a.mbc)} ({a.f_mbc:.0%}>, q {q['mbc']:.2f})"
                f" | x {a.x_conf:.2f} 추세봉 {a.ts_conf:.0%} | 통과 원안 {d['pass_vs_base']} 같은비중 {d['pass_vs_static']} 귀무 {d['pass_null95']}")
            say(f"      확인 원값: 수익 {pct(a.ret_conf)} 낙폭 {pct(a.mdd_conf)} | 같은 비중 {pct(a.static_conf)} {pct(a.static_mdd_conf)}"
                f" | 원안 {pct(a.ret_conf_B)} {pct(a.mdd_conf_B)} | 귀무 고르기 1위 상위: {d['null_picks']}")

    # 같은 칸(실제 판 1위·기본값)을 비용 0 판에서
    say("\n[기준 2: 같은 칸을 비용 0 판에서]")
    for pick in ("top1", "default"):
        st = summ[f"real_{pick}"]["strategy"].split()
        b, clf, act = int(st[0][1:]), st[1], st[2]
        x = M[(M.board == "real0") & (M.b == b) & (M.clf == clf) & (M.action == act)]
        a = x[x.k == 0].iloc[0]
        nl = x[x.k > 0]
        q = {m: qpos(a[m], nl[m]) for m in ("dc", "dcb", "mmc", "mbc")}
        summ[f"real0_samecell_{pick}"] = dict(strategy=f"b{b} {clf} {act}", sel_ret_minus_static=a.ds, conf_ret_minus_static=a.dc,
                                              conf_ret_minus_base=a.dcb, conf_mdd_minus_static=a.mmc, conf_mdd_minus_base=a.mbc,
                                              null_q=q, null_dc_p95=float(np.quantile(nl.dc, 0.95)))
        say(f" real0 같은 칸 {pick}: b{b} {clf} {act} 고르기 {pct(a.ds)} 확인 수익−같은비중 {pct(a.dc)} (q {q['dc']:.2f}) "
            f"수익−원안 {pct(a.dcb)} (q {q['dcb']:.2f}) 낙폭−같은비중 {pct(a.mmc)} 낙폭−원안 {pct(a.mbc)}")

    # ---------- 기준 3: 이웃 ----------
    say("\n[기준 3: 이웃 조합 (z는 b의 배수로 맞춤), 실제 판 확인 중앙]")
    order = MULTS
    nb = {}
    for pick in ("top1", "default"):
        st = summ[f"real_{pick}"]["strategy"].split()
        b, clf, act = int(st[0][1:]), st[1], st[2]
        kind = clf.split("_")[1]
        m = zmult(b, clf)
        bi = BASE_TFS.index(b)
        mi = order.index(m)
        cands = []
        for b2 in (BASE_TFS[bi - 1] if bi > 0 else None, BASE_TFS[bi + 1] if bi + 1 < len(BASE_TFS) else None):
            if b2 is not None:
                cands.append(("b", b2, m))
        for m2 in (order[mi - 1] if mi > 0 else None, order[mi + 1] if mi + 1 < len(order) else None):
            if m2 is not None:
                cands.append(("z", b, m2))
        rows = []
        a0 = M[(M.board == "real") & (M.k == 0) & (M.b == b) & (M.clf == clf) & (M.action == act)].iloc[0]
        for axis, b2, m2 in cands:
            x = M[(M.board == "real") & (M.k == 0) & (M.b == b2) & (M.m == m2) & (M.kind == kind) & (M.action == act)]
            if not len(x):
                say(f"  {pick} 이웃 {axis} b{b2} {m2} {kind} {act}: 없음")
                continue
            r = x.iloc[0]
            same = {mm: bool(np.sign(r[mm]) == np.sign(a0[mm])) for mm in ("dc", "dcb", "mmc", "mbc")}
            rows.append(dict(axis=axis, strategy=f"b{b2} {r.clf} {act}", conf_ret_minus_static=r.dc, conf_ret_minus_base=r.dcb,
                             conf_mdd_minus_static=r.mmc, conf_mdd_minus_base=r.mbc, sel_ret_minus_static=r.ds, same_sign=same,
                             degen=bool(r.degen)))
            say(f"  {pick} 이웃 {axis} b{b2} {r.clf} {act}: 고르기 {pct(r.ds)} 확인 수익−같은비중 {pct(r.dc)} 수익−원안 {pct(r.dcb)}"
                f" 낙폭−같은비중 {pct(r.mmc)} 낙폭−원안 {pct(r.mbc)} | 같은 부호 {same}{' [퇴화]' if r.degen else ''}")
        nb[pick] = rows
        # 기준 3 '부호가 같다'는 확인 구간 수익−같은 비중과 수익−원안의 부호로 본다
        summ[f"neighbors_{pick}"] = rows
    # ---------- 시간봉 조합 지도 ----------
    say("\n[시간봉 조합 지도: 확인 구간 실제 판 144 중앙 (수익 − 같은 비중) %p, 괄호는 칸별 귀무 분위]")
    cells = []
    for bd in ("real", "real0"):
        x0 = M[(M.board == bd) & (M.k == 0) & (M.action != "A3")]
        xn = M[(M.board == bd) & (M.k > 0) & (M.action != "A3")]
        grp = {key: v for key, v in xn.groupby(["b", "clf", "action"])}
        for _, r in x0.iterrows():
            nl = grp[(r.b, r.clf, r.action)]
            row = dict(board=bd, b=r.b, m=r.m, kind=r.kind, clf=r.clf, action=r.action, degen=bool(r.degen),
                       x_sel=r.x_sel, x_conf=r.x_conf, ts_sel=r.ts_sel, ts_conf=r.ts_conf, turn_conf=r.turn_conf)
            for mm in MET:
                row[mm] = r[mm]
            for mm in ("ds", "dc", "dcb", "mmc", "mbc"):
                row["q_" + mm] = qpos(r[mm], nl[mm])
            for mm in ("f_dc", "f_dcb", "f_mmc", "f_mbc"):
                row[mm] = r[mm]
            cells.append(row)
    C = pd.DataFrame(cells)
    C.to_csv(RES / f"mtfstrat_cells{tag}.csv", index=False)
    for bd in (("real",) if maps else ()):
        for met, nm in (("dc", "수익−같은 비중"), ("dcb", "수익−같은 b 원안"), ("mmc", "낙폭−같은 비중 낙폭")):
            say(f"\n -- {bd} {nm} (확인) --")
            for kind in ("ER", "ADX", "EXT", "AGREE"):
                for act in ("A0", "A2"):
                    x = C[(C.board == bd) & (C.kind == kind) & (C.action == act)]
                    say(f"  {kind} {act}: " + "  ".join(MULTS))
                    for b in BASE_TFS:
                        cellv = []
                        for m in MULTS:
                            y = x[(x.b == b) & (x.m == m)]
                            if not len(y):
                                cellv.append("      .      ")
                            else:
                                y = y.iloc[0]
                                cellv.append(f"{100 * y[met]:+6.1f}({y['q_' + met]:.2f}){'*' if y.degen else ' '}")
                        say(f"   b{b:<4d}" + " ".join(cellv))

    # 판정·행동별 요약
    say("\n[칸 요약: 판정·행동별 확인 실제 판 칸 수와 중앙]")
    for (kind, act), x in C[C.board == "real"].groupby(["kind", "action"]):
        say(f"  {kind:5s} {act}: 칸 {len(x)}, 확인 수익−같은비중 중앙 {pct(x.dc.median())} (>0 칸 {int((x.dc > 0).sum())}), "
            f"수익−원안 중앙 {pct(x.dcb.median())} (>0 {int((x.dcb > 0).sum())}), 낙폭−같은비중 중앙 {pct(x.mmc.median())} (≥0 {int((x.mmc >= 0).sum())}), "
            f"귀무 95분위 넘은 칸(수익−같은비중) {int((x.q_dc > 0.95).sum())}")

    # ---------- 계열 수: 실제 대 귀무 ----------
    say("\n[계열: 326칸 중 조건을 만족한 칸 수, 실제 대 귀무 40개]")
    fam = []
    conds = {
        "both_ret_beat_static": lambda x: (x.ds > 0) & (x.dc > 0),
        "both_ret_and_mdd_beat_static": lambda x: (x.ds > 0) & (x.dc > 0) & (x.mms > 0) & (x.mmc > 0),
        "conf_ret_beat_static": lambda x: x.dc > 0,
        "conf_beat_static_and_base(ret,mdd)": lambda x: (x.dc >= 0) & (x.mmc >= 0) & (x.dcb > 0) & (x.mbc > 0),
        "both_beat_base_ret": lambda x: (x.dsb > 0) & (x.dcb > 0),
        "both_ret_beat_static_nondegen": lambda x: (x.ds > 0) & (x.dc > 0) & ~x.degen,
    }
    for bd in ("real", "real0"):
        for nm, fn in conds.items():
            cnt = {k: int(fn(Z[(Z.board == bd) & (Z.k == k)]).sum()) for k in [0] + ks}
            a = cnt[0]
            nl = np.array([cnt[k] for k in ks])
            p = (1 + int((nl >= a).sum())) / (1 + len(nl))
            fam.append(dict(board=bd, cond=nm, actual=a, null_median=float(np.median(nl)), null_mean=float(nl.mean()),
                            null_p95=float(np.quantile(nl, 0.95)), null_max=int(nl.max()), p=p))
            say(f"  {bd:5s} {nm:36s} 실제 {a:3d} | 귀무 중앙 {np.median(nl):.1f} 평균 {nl.mean():.1f} 95분위 {np.quantile(nl, 0.95):.1f} 최대 {nl.max()} | p {p:.3f}")
    # 순위 상관(고르기 → 확인), 실제 대 귀무
    from scipy.stats import spearmanr
    for bd in ("real", "real0"):
        rho = {k: spearmanr(*Z[(Z.board == bd) & (Z.k == k)][["ds", "dc"]].values.T).statistic for k in [0] + ks}
        nl = np.array([rho[k] for k in ks])
        fam.append(dict(board=bd, cond="spearman_sel_conf", actual=rho[0], null_median=float(np.median(nl)), null_mean=float(nl.mean()),
                        null_p95=float(np.quantile(nl, 0.95)), null_max=float(nl.max()), p=(1 + int((nl >= rho[0]).sum())) / (1 + len(nl))))
        say(f"  {bd:5s} 고르기→확인 순위상관 실제 {rho[0]:+.2f} | 귀무 중앙 {np.median(nl):+.2f} 95분위 {np.quantile(nl, 0.95):+.2f}")
    pd.DataFrame(fam).to_csv(RES / f"mtfstrat_family{tag}.csv", index=False)
    summ["family"] = fam
    # 퇴화 칸 수
    dg = Z[(Z.board == "real") & (Z.k == 0)]
    summ["n_degen_real"] = int(dg.degen.sum())
    say(f"\n퇴화 칸(평균 보유<0.1 또는 추세 봉<5%, 고르기·확인 중 하나): {int(dg.degen.sum())} / {len(dg)}")
    say("고르기 상위 10 (실제 판):")
    for _, r in dg.sort_values("ds", ascending=False).head(10).iterrows():
        say(f"  b{r.b} {r.clf} {r.action}{' [퇴화]' if r.degen else ''}: 고르기 {pct(r.ds)} 확인 {pct(r.dc)} 수익−원안 {pct(r.dcb)} 낙폭−같은비중 {pct(r.mmc)} x {r.x_conf:.2f} 추세봉 {r.ts_conf:.0%}")

    return summ



def main():
    g = load()
    say("행", len(g), "종목", g.sym.nunique())
    M = medians(g)
    ts = trend_share()
    M = M.merge(ts, on=["b", "clf"], how="left")
    M["degen"] = (M.x_sel < 0.1) | (M.x_conf < 0.1) | (M.ts_sel < 0.05) | (M.ts_conf < 0.05)
    M.loc[M.action == "A3", "degen"] = False
    ks = sorted(k for k in M.k.unique() if k > 0)
    say("귀무 이동", ks[0], "~", ks[-1], len(ks), "개")

    # ---------- 원안 시간봉 ----------
    B = M[(M.action == "A3") & (M.k == 0)].sort_values(["board", "b"])
    say("\n[원안 A3 시간봉별, 144 중앙] (수익−같은 비중 sel/conf, 수익 sel/conf, 보유 비율, 회전 conf)")
    for bd in ("real", "real0", "ideal"):
        x = B[B.board == bd]
        for _, r in x.iterrows():
            say(f" {bd:5s} b{r.b:<3d} sel {pct(r.ds)} conf {pct(r.dc)} | ret {pct(r.ret_sel)} {pct(r.ret_conf)} | "
                f"x {r.x_sel:.2f} {r.x_conf:.2f} | 회전 {r.turn_sel:.0f} {r.turn_conf:.0f} | f(conf≥static) {r.f_dc:.2f}")
        from scipy.stats import spearmanr
        rs = spearmanr(x.b, x.ds).statistic
        rc = spearmanr(x.b, x.dc).statistic
        say(f" {bd} 순위상관(b, 중앙 수익−같은 비중): 고르기 {rs:+.2f}, 확인 {rc:+.2f}")
    B.to_csv(RES / "mtfstrat_base_tf.csv", index=False)

    summ = evaluate(M, ks)
    alt = RES / "mtfstrat_agree_null.parquet"
    if alt.exists():
        M2 = alt_null(g, M, alt)
        summ["altnull"] = evaluate(M2, ks, "_altnull", maps=False)
        say("\n[AGREE 칸 확인 실제 판 (수익−같은 비중) 귀무 분위: 원래 귀무 → z 상태 이동 귀무]")
        c1 = pd.read_csv(RES / "mtfstrat_cells.csv"); c2 = pd.read_csv(RES / "mtfstrat_cells_altnull.csv")
        c = c1.merge(c2[["board", "b", "clf", "action", "q_dc", "q_dcb", "q_mmc"]], on=["board", "b", "clf", "action"], suffixes=("", "_alt"))
        c = c[c.kind == "AGREE"]
        for (bd, act), x in c.groupby(["board", "action"]):
            say(f"  {bd:5s} {act}: 칸 {len(x)}, q_dc>0.95 칸 {int((x.q_dc > 0.95).sum())} → {int((x.q_dc_alt > 0.95).sum())}, q_dc 중앙 {x.q_dc.median():.2f} → {x.q_dc_alt.median():.2f}")
        for act in ("A0", "A2"):
            x = c[(c.board == "real") & (c.action == act)]
            say(f"  AGREE {act} 실제 판 (수익−같은 비중)(원래 q / z 이동 q):")
            for b in BASE_TFS:
                cellv = []
                for m in MULTS:
                    y = x[(x.b == b) & (x.m == m)]
                    cellv.append("        .         " if not len(y) else f"{100 * y.iloc[0].dc:+6.1f}({y.iloc[0].q_dc:.2f}/{y.iloc[0].q_dc_alt:.2f})")
                say(f"   b{b:<4d}" + " ".join(cellv))
    json.dump(summ, open(RES / "mtfstrat_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    (RES / "mtfstrat_analyze.log").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
