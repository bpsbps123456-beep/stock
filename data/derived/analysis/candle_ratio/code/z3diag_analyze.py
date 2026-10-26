"""z3 diag 갈래 분석: z3diag_run.py 원자료 → 판정 통계·귀무·사후 표지 혼동표.

판정 통계(봉당, 단순 수익 %): 종목별 d = 추세 봉당 초과 − 애매 봉당 초과, 144종목 중앙.
- 보조: '애매 < 0 < 추세' 종목 비율, 중앙 추세·애매 봉당 초과(채택 기준 1의 부호 조건).
- 귀무 = flag 공통 k 달력 이동(5분봉 40개, 1시간봉 92개). 95분위는 np.quantile(…, 0.95), p = (귀무 ≥ 관측 수 + 1)/(귀무 수 + 1).
사후 표지(5분봉, real01 주·ideal 보조): ① 보유(평균 pos ≥ 0.5)·날 수익 상위 20%, ② 현금(< 0.5)·하위 20%,
③ 나머지 중 그날 노출 맞춘 초과 < 0, ④ 그 밖. 분위는 종목별 7/20~10/22 전체 날(고르기+확인).
py code/z3diag_analyze.py → results/z3diag_bars_5m.csv, z3diag_bars_1h.csv, z3diag_posthoc_5m.csv, z3diag_confusion_5m.csv, z3diag_summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from engine import HERE

OUT = HERE / "results"
DEFAULT = "ER_K20_th0.3"
MAIN = "real01"
WN = ("sel", "conf")


def load(base):
    d = np.load(OUT / f"z3diag_raw_{base}.npz")
    return {k: d[k] for k in d.files}


def perbar(R):
    """(.., 2[추세, 애매], 2[초과, 봉 수]) → 봉당 % (추세, 애매), 합계 %p (추세, 애매)."""
    ex, n = R[..., 0], R[..., 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        pb = np.where(n > 0, ex / n, np.nan) * 100
    return pb[..., 0], pb[..., 1], ex[..., 0] * 100, ex[..., 1] * 100


def stock_stats(R):
    """R: (종목, k, 후보, 2, 2) → k·후보별 통계 사전."""
    t, a, tt, ta = perbar(R)
    d = t - a
    both = ~np.isnan(d)
    with np.errstate(invalid="ignore"):
        frac = np.where(both, (a < 0) & (t > 0), np.nan)
    return dict(diff=np.nanmedian(d, axis=0), trend=np.nanmedian(t, axis=0), amb=np.nanmedian(a, axis=0),
                frac=np.nanmean(frac, axis=0), tot_trend=np.nanmedian(tt, axis=0), tot_amb=np.nanmedian(ta, axis=0),
                tot_diff=np.nanmedian(tt - ta, axis=0), n_stocks=both.sum(axis=0),
                share=np.nanmedian(R[..., 0, 1] / (R[..., 0, 1] + R[..., 1, 1]), axis=0))


def null_cmp(obs, null):
    q95 = np.quantile(null, 0.95, axis=0)
    p = ((null >= obs[None]).sum(axis=0) + 1) / (null.shape[0] + 1)
    return q95, p


def bar_table(base, D):
    RES, cands, boards = D["RES"], list(D["cands"]), list(D["boards"])
    rows = []
    stats = {}
    for bi, bn in enumerate(boards):
        for wi, wn in enumerate(WN):
            S = stock_stats(RES[:, bi, wi])
            stats[(bn, wn)] = S
            q95, p = null_cmp(S["diff"][0], S["diff"][1:])
            fq95, fp = null_cmp(S["frac"][0], S["frac"][1:])
            for ci, c in enumerate(cands):
                rows.append(dict(base=base, board=bn, win=wn, cand=c, trend_share=S["share"][0, ci],
                                 trend_pb=S["trend"][0, ci], amb_pb=S["amb"][0, ci], diff_pb=S["diff"][0, ci],
                                 diff_null_med=np.median(S["diff"][1:, ci]), diff_null_q95=q95[ci], diff_p=p[ci],
                                 frac=S["frac"][0, ci], frac_null_med=np.median(S["frac"][1:, ci]), frac_null_q95=fq95[ci], frac_p=fp[ci],
                                 tot_trend=S["tot_trend"][0, ci], tot_amb=S["tot_amb"][0, ci], tot_diff=S["tot_diff"][0, ci],
                                 n_stocks=int(S["n_stocks"][0, ci])))
    df = pd.DataFrame(rows)
    df["diff_pass"] = df.diff_pb > df.diff_null_q95
    df["sign_ok"] = (df.amb_pb < 0) & (df.trend_pb > 0)
    df["crit1"] = df.diff_pass & df.sign_ok
    df["frac_pass"] = df.frac > df.frac_null_q95
    return df, stats


def sel_to_conf(stats, cands, board):
    s, c = stats[(board, "sel")]["diff"], stats[(board, "conf")]["diff"]
    top = int(np.argmax(s[0]))
    rho = [spearmanr(s[k], c[k]).statistic for k in range(s.shape[0])]
    q95c, pc = null_cmp(c[0], c[1:])
    return dict(sel_top=cands[top], sel_top_sel_diff=round(float(s[0, top]), 5), sel_top_conf_diff=round(float(c[0, top]), 5),
                sel_top_conf_null_q95=round(float(q95c[top]), 5), sel_top_conf_p=round(float(pc[top]), 3),
                sel_top_conf_rank=int((c[0] > c[0, top]).sum()) + 1,
                rho_sel_conf=round(float(rho[0]), 3), rho_null_med=round(float(np.median(rho[1:])), 3),
                rho_null_q95=round(float(np.quantile(rho[1:], 0.95)), 3), rho_p=round(float((np.sum(np.array(rho[1:]) >= rho[0]) + 1) / len(rho)), 3))


def roll_check(D):
    """real01 달력 이동 대 np.roll 귀무: 확인 구간 diff가 귀무 95분위를 넘는 후보 수."""
    S = stock_stats(D["ROLL"][:, 1])
    q95, p = null_cmp(S["diff"][0], S["diff"][1:])
    return dict(n_pass_roll=int((S["diff"][0] > q95).sum()), p_roll={c: round(float(x), 3) for c, x in zip(D["cands"], p)})


def labels(days, board):
    g = days.groupby("sym").ret
    q80 = g.transform(lambda x: x.quantile(0.8))
    q20 = g.transform(lambda x: x.quantile(0.2))
    pos, ex = days[f"pos_{board}"], days[f"ex_{board}"]
    L = np.full(len(days), 4)
    l1 = (pos >= 0.5) & (days.ret >= q80)
    l2 = (pos < 0.5) & (days.ret <= q20)
    l3 = ~l1 & ~l2 & (ex < -1e-12)
    L[l3.values] = 3
    L[l2.values] = 2
    L[l1.values] = 1
    return L


def posthoc(days, Z, cands, board, win):
    """사후 표지 ①②③과 장 시작 전 구간의 맞음: 각 표지 안에서 해당 구간 비율 − 전체 날의 그 비율(%p), 종목 풀링."""
    L = labels(days, board)
    m = np.ones(len(days), bool) if win == "all" else (days.seg.values == (0 if win == "sel" else 1))
    Lm, Zm = L[m], Z[:, :, m]
    out = {}
    for lab, z in ((1, 1), (2, -1), (3, 0)):
        sel = Lm == lab
        base_rate = (Zm == z).mean(axis=2) * 100
        in_rate = (Zm[:, :, sel] == z).mean(axis=2) * 100
        lift = in_rate - base_rate
        q95, p = null_cmp(lift[0], lift[1:])
        out[lab] = dict(n=int(sel.sum()), base=base_rate[0], rate=in_rate[0], lift=lift[0], null_med=np.median(lift[1:], axis=0), q95=q95, p=p)
    # ③ 날의 손해 중 사전에 애매로 표시된 몫
    exm = days[f"ex_{board}"].values[m]
    sel3 = Lm == 3
    loss_amb = np.array([exm[sel3 & (Zm[0, ci] == 0)].sum() for ci in range(len(cands))])
    loss_all = exm[sel3].sum()
    out["loss_amb_share"] = loss_amb / loss_all * 100
    out["n_days"] = int(m.sum())
    out["label_counts"] = {int(k): int(v) for k, v in zip(*np.unique(Lm, return_counts=True))}
    return out, L


def confusion(days, Z, L, cands, pick, win):
    m = np.ones(len(days), bool) if win == "all" else (days.seg.values == (0 if win == "sel" else 1))
    rows = []
    for c in pick:
        ci = cands.index(c)
        z = Z[0, ci, m]
        for lab, nm in ((1, "① 보유·많이 번 날"), (2, "② 현금·덜 잃은 날"), (3, "③ 애매에서 손해"), (4, "④ 그 밖")):
            sel = L[m] == lab
            r = dict(win=win, cand=c, label=nm, n=int(sel.sum()))
            for zz, zn in ((1, "상승추세"), (-1, "하락추세"), (0, "애매")):
                r[zn] = int((z[sel] == zz).sum())
                r[zn + "_%"] = round((z[sel] == zz).mean() * 100, 1) if sel.any() else np.nan
            rows.append(r)
        r = dict(win=win, cand=c, label="전체 날", n=int(m.sum()))
        for zz, zn in ((1, "상승추세"), (-1, "하락추세"), (0, "애매")):
            r[zn] = int((z == zz).sum())
            r[zn + "_%"] = round((z == zz).mean() * 100, 1)
        rows.append(r)
    return pd.DataFrame(rows)


def rnd(x, k=5):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


if __name__ == "__main__":
    summ = {}
    D5, D1 = load("5m"), load("1h")
    c5, c1 = list(D5["cands"]), list(D1["cands"])
    b5, st5 = bar_table("5m", D5)
    b1, st1 = bar_table("1h", D1)
    b5.to_csv(OUT / "z3diag_bars_5m.csv", index=False, encoding="utf-8-sig")
    b1.to_csv(OUT / "z3diag_bars_1h.csv", index=False, encoding="utf-8-sig")
    for base, b, cands, st in (("5m", b5, c5, st5), ("1h", b1, c1, st1)):
        nc = len(cands)
        cnt = b.groupby(["board", "win"]).agg(diff_pass=("diff_pass", "sum"), crit1=("crit1", "sum"), sign_ok=("sign_ok", "sum"),
                                              frac_pass=("frac_pass", "sum")).reset_index()
        summ[base] = dict(n_cands=nc, n_shifts=int(len(D5["ks"] if base == "5m" else D1["ks"]) - 1), chance=round(nc * 0.05, 2),
                          counts=cnt.to_dict("records"),
                          sel_to_conf={bd: sel_to_conf(st, cands, bd) for bd in ("real01", "ideal", "std01")})
        mb = b[(b.board == MAIN) & (b.win == "conf")].sort_values("diff_pb", ascending=False)
        summ[base]["conf_real01_top5"] = mb.head(5)[["cand", "trend_pb", "amb_pb", "diff_pb", "diff_null_q95", "diff_p", "frac", "frac_null_q95"]].round(5).to_dict("records")
        summ[base]["conf_real01_default"] = mb[mb.cand == DEFAULT][["trend_pb", "amb_pb", "diff_pb", "diff_null_med", "diff_null_q95", "diff_p", "frac", "frac_null_med", "frac_null_q95", "trend_share", "tot_trend", "tot_amb"]].round(5).to_dict("records")[0]
    summ["5m"]["roll_check_real01_conf"] = roll_check(D5)
    # 1시간봉 방향 확인: 같은 후보의 확인 구간 diff 부호
    m5 = b5[(b5.board == MAIN) & (b5.win == "conf")].set_index("cand")
    m1 = b1[(b1.board == MAIN) & (b1.win == "conf")].set_index("cand")
    j = m5.join(m1, lsuffix="_5m", rsuffix="_1h", how="inner")
    summ["1h_direction"] = dict(n=len(j), same_sign=int((np.sign(j.diff_pb_5m) == np.sign(j.diff_pb_1h)).sum()),
                                pos_5m=int((j.diff_pb_5m > 0).sum()), pos_1h=int((j.diff_pb_1h > 0).sum()),
                                pass_1h=int(j.diff_pass_1h.sum()), crit1_1h=int(j.crit1_1h.sum()),
                                rho=round(float(spearmanr(j.diff_pb_5m, j.diff_pb_1h).statistic), 3))
    # 사후 표지
    days = pd.read_parquet(OUT / "z3diag_days_5m.parquet")
    dz = np.load(OUT / "z3diag_days_5m.npz")
    Z, cz = dz["Z"], list(dz["cands"])
    assert cz == c5 and Z.shape[2] == len(days)
    top3 = list(m5.sort_values("diff_pb", ascending=False).index[:3])
    pick = [DEFAULT] + [c for c in top3 if c != DEFAULT]
    ph_rows, conf_tabs = [], []
    for board in ("real01", "ideal"):
        for win in ("conf", "sel", "all"):
            ph, L = posthoc(days, Z, c5, board, win)
            for lab in (1, 2, 3):
                o = ph[lab]
                for ci, c in enumerate(c5):
                    ph_rows.append(dict(board=board, win=win, label=lab, cand=c, n=o["n"], base=o["base"][ci], rate=o["rate"][ci],
                                        lift=o["lift"][ci], null_med=o["null_med"][ci], q95=o["q95"][ci], p=o["p"][ci],
                                        loss_amb_share=ph["loss_amb_share"][ci] if lab == 3 else np.nan))
            summ.setdefault("posthoc", {})[f"{board}_{win}"] = dict(n_days=ph["n_days"], label_counts=ph["label_counts"])
            if win in ("conf", "all"):
                ct = confusion(days, Z, L, c5, pick, win)
                ct.insert(0, "board", board)
                conf_tabs.append(ct)
    ph = pd.DataFrame(ph_rows)
    ph["pass"] = ph.lift > ph.q95
    ph.to_csv(OUT / "z3diag_posthoc_5m.csv", index=False, encoding="utf-8-sig")
    pd.concat(conf_tabs).to_csv(OUT / "z3diag_confusion_5m.csv", index=False, encoding="utf-8-sig")
    summ["posthoc_pick"] = pick
    summ["posthoc_pass_counts"] = ph.groupby(["board", "win", "label"])["pass"].sum().astype(int).reset_index().to_dict("records")
    sub = ph[(ph.board == MAIN) & (ph.win == "conf") & ph.cand.isin(pick)]
    summ["posthoc_pick_real01_conf"] = sub[["label", "cand", "n", "base", "rate", "lift", "null_med", "q95", "p", "loss_amb_share"]].round(3).to_dict("records")
    (OUT / "z3diag_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    cols = ["cand", "trend_share", "trend_pb", "amb_pb", "diff_pb", "diff_null_med", "diff_null_q95", "diff_p", "frac", "frac_null_q95", "crit1"]
    for base, b in (("5m", b5), ("1h", b1)):
        for bd in ("real01", "ideal"):
            for wn in WN:
                print(f"\n== {base} {bd} {wn} ==")
                print(b[(b.board == bd) & (b.win == wn)][cols].round(5).to_string(index=False))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("posthoc_pick_real01_conf",)}, ensure_ascii=False, indent=1, default=str))
    print(pd.concat(conf_tabs).to_string(index=False))
    print(sub.round(3).to_string(index=False))
