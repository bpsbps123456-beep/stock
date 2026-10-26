"""mtf diag 갈래: 시간봉 조합 판정이 '애매'를 미리 알아보는지(BRIEF3 진단). 입력 results/mtf_diag.parquet(mtf_run.py).

판정 = (신호 시간봉 b, 판정 clf) 163개. 종목별 통계(원안 비용 0 보유, 노출 맞춘 초과):
- 봉당 차이 d = trend/trend_n − amb/amb_n (단순 %), 합계 차이 = trend − amb (%p). 144종목 중앙.
- 하루치 환산 = 봉당 × (960/b) (정규·프리·애프터 16시간 봉 수). b가 다른 판정끼리 크기를 볼 때만 쓴다.
- 부호 조건 = 중앙 애매 봉당 < 0 < 중앙 추세 봉당. 통과 = 부호 조건 & d 중앙 > 귀무 95분위(k 15~54, 40개).
- 계열 귀무: 귀무 k마다 같은 통계(그 k의 부호 조건 & 나머지 39개 95분위 초과)로 센 통과 수의 분포(z3diag_family와 같은 방식).
- 표준화 점수 = (관측 − 귀무 평균) ÷ 귀무 표준편차(귀무 k는 나머지 39개로). b가 섞인 비교(순위 상관·묶음)에 쓴다.
구조 표시: z = b의 EXT(신호와 같은 비율 재료)와 AGREE(정의에 s가 들어감)는 판정만 옮기는 귀무가 구조상 맞지 않는다 → 'struct'.
결론 집합 'clean' = 나머지 120개. 민감도: EXT 전부를 뺀 'noext'(86), z > b만 남긴 'zgtb'(102).
사후 관찰(사전 등록 밖): 반대 방향(차이 < 귀무 5분위 & 추세 < 0 < 애매) 통과 수와 계열 평균 표준화 점수의 귀무 비교.
py code/mtfdiag_analyze.py → results/mtfdiag_judgments.csv, mtfdiag_groups.csv, mtfdiag_zmap.csv, mtfdiag_summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from engine import HERE
from zones_mtf import BASE_TFS, zone_tfs

OUT = HERE / "results"
BOARDS = ("real0", "ideal")
WN = ("sel", "conf")
DEFAULT = (40, "z160_AGREE")
SEL_TOP_STRAT = (40, "z320_ER")  # 전략 고르기 1위(실제 판, A0)의 판정


def judgments():
    rows = []
    for b in BASE_TFS:
        for z in zone_tfs(b):
            types = ("ER", "ADX", "EXT") if z == b else ("AGREE", "ER", "ADX", "EXT")
            for t in types:
                m = "D" if z == "D" else str(z // b)
                zmin = 1440 if z == "D" else z
                rows.append(dict(b=b, clf=f"z{z}_{t}", z=str(z), m=m, zmin=zmin, type=t,
                                 struct=(t == "AGREE") or (t == "EXT" and z == b)))
    J = pd.DataFrame(rows)
    J["clean"] = ~J.struct
    J["noext"] = J.clean & (J.type != "EXT")
    J["zgtb"] = J.clean & (J.z != J.b.astype(str))  # z = b의 ER·ADX(뒤집힘과 중간 정도로 묶임, mtfdiag_struct)도 뺀 민감도
    return J


def load_arrays(J):
    d = pd.read_parquet(OUT / "mtf_diag.parquet")
    d["clf"] = d.clf.astype(str); d["sym"] = d.sym.astype(str); d["board"] = d.board.astype(str)
    ks = [0] + sorted(int(k) for k in d.k.unique() if k > 0)
    syms = sorted(d.sym.unique())
    jkey = {(b, c): i for i, (b, c) in enumerate(zip(J.b, J.clf))}
    d["ji"] = [jkey[(b, c)] for b, c in zip(d.b, d.clf)]
    d["ki"] = d.k.map({k: i for i, k in enumerate(ks)})
    d["si"] = d.sym.map({s: i for i, s in enumerate(syms)})
    assert d.groupby(["board", "ki", "ji", "si"]).size().max() == 1
    A = {}
    for bd in BOARDS:
        x = d[d.board == bd]
        for w in WN:
            arr = {}
            for nm in ("trend", "amb"):
                v = np.full((len(ks), len(J), len(syms)), np.nan)
                n = np.zeros_like(v)
                v[x.ki, x.ji, x.si] = x[f"{nm}_{w}"].values
                n[x.ki, x.ji, x.si] = x[f"{nm}_{w}_n"].values
                arr[nm], arr[nm + "_n"] = v, n
            A[(bd, w)] = arr
    return A, ks, syms


def stats(arr, J):
    """(k, 판정, 종목) → k·판정별 중앙 통계."""
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(arr["trend_n"] > 0, arr["trend"] / arr["trend_n"], np.nan) * 100
        a = np.where(arr["amb_n"] > 0, arr["amb"] / arr["amb_n"], np.nan) * 100
        share = arr["trend_n"] / (arr["trend_n"] + arr["amb_n"])
    d = t - a
    tot = (arr["trend"] - arr["amb"]) * 100  # 한쪽이 비면 NaN
    per_day = (960 / J.b.values)[None, :]
    S = dict(trend=np.nanmedian(t, axis=2), amb=np.nanmedian(a, axis=2), diff=np.nanmedian(d, axis=2),
             tot=np.nanmedian(tot, axis=2), share=np.nanmedian(share, axis=2), n=(~np.isnan(d)).sum(axis=2))
    with np.errstate(invalid="ignore"):
        S["negfrac"] = np.nanmean(np.where(np.isnan(d), np.nan, d < 0), axis=2)  # 추세 봉당 < 애매 봉당인 종목 비율
    S["diff_day"] = S["diff"] * per_day
    S["trend_day"] = S["trend"] * per_day
    S["amb_day"] = S["amb"] * per_day
    S["sign"] = (S["amb"] < 0) & (S["trend"] > 0)
    S["zs"] = zscore(S["diff"])
    S["zs_tot"] = zscore(S["tot"])
    return S


def zscore(x):
    """행 0 = 관측(귀무 40개로 표준화), 행 i ≥ 1 = 귀무 i(나머지 39개로 표준화)."""
    out = np.empty_like(x)
    null = x[1:]
    out[0] = (x[0] - null.mean(0)) / null.std(0, ddof=1)
    for i in range(null.shape[0]):
        rest = np.delete(null, i, axis=0)
        out[1 + i] = (null[i] - rest.mean(0)) / rest.std(0, ddof=1)
    return out


def pass_mat(x, sign):
    """(k, 판정) 통과 행렬: k=0은 귀무 40개 95분위, 귀무 k는 나머지 39개 95분위(한 개 빼기)."""
    null = x[1:]
    P = np.zeros(x.shape, bool)
    P[0] = x[0] > np.quantile(null, 0.95, axis=0)
    for i in range(null.shape[0]):
        P[1 + i] = null[i] > np.quantile(np.delete(null, i, axis=0), 0.95, axis=0)
    return P & sign


def pval_ge(obs, null):
    return float(((np.asarray(null) >= obs).sum() + 1) / (len(null) + 1))


def pval_le(obs, null):
    return float(((np.asarray(null) <= obs).sum() + 1) / (len(null) + 1))


def fam(Pk, mask):
    c = Pk[:, mask].sum(axis=1)
    return dict(n=int(mask.sum()), obs=int(c[0]), null_mean=round(float(c[1:].mean()), 2),
                null_q95=float(np.quantile(c[1:], 0.95)), null_max=int(c[1:].max()), p=round(pval_ge(c[0], c[1:]), 3))


def pass_low(x, sign):
    """반대 방향: 귀무 5분위 아래(k=0은 40개, 귀무 k는 나머지 39개) & 반대 부호."""
    null = x[1:]
    P = np.zeros(x.shape, bool)
    P[0] = x[0] < np.quantile(null, 0.05, axis=0)
    for i in range(null.shape[0]):
        P[1 + i] = null[i] < np.quantile(np.delete(null, i, axis=0), 0.05, axis=0)
    return P & sign


def mean_z(S, mask):
    m = S["zs"][:, mask].mean(axis=1)
    lo = float(((m[1:] <= m[0]).sum() + 1) / len(m))
    hi = float(((m[1:] >= m[0]).sum() + 1) / len(m))
    return dict(obs=round(float(m[0]), 3), null_med=round(float(np.median(m[1:])), 3), null_q05=round(float(np.quantile(m[1:], 0.05)), 3),
                null_q95=round(float(np.quantile(m[1:], 0.95)), 3), p_low=round(lo, 3), p_high=round(hi, 3),
                neg_share=round(float((S["zs"][0, mask] < 0).mean()), 3))


def judgment_table(S, J, bd, w):
    null = S["diff"][1:]
    q95 = np.quantile(null, 0.95, axis=0)
    tq95 = np.quantile(S["tot"][1:], 0.95, axis=0)
    df = J.copy()
    df.insert(0, "win", w); df.insert(0, "board", bd)
    df["n_stocks"] = S["n"][0]
    df["trend_share"] = S["share"][0]
    for c in ("trend", "amb", "diff"):
        df[f"{c}_pb"] = S[c][0]
        df[f"{c}_day"] = S[f"{c}_day"][0]
    df["neg_frac"] = S["negfrac"][0]
    df["neg_frac_null_med"] = np.median(S["negfrac"][1:], axis=0)
    df["diff_null_med"] = np.median(null, axis=0)
    df["diff_null_q95"] = q95
    df["diff_p"] = [pval_ge(S["diff"][0, j], null[:, j]) for j in range(len(J))]
    df["diff_null_pct"] = (null < S["diff"][0][None]).mean(axis=0)  # 귀무 분위(관측보다 작은 귀무 비율)
    df["diff_z"] = S["zs"][0]
    df["tot_diff"] = S["tot"][0]
    df["tot_null_q95"] = tq95
    df["tot_p"] = [pval_ge(S["tot"][0, j], S["tot"][1:, j]) for j in range(len(J))]
    df["sign_ok"] = S["sign"][0]
    df["diff_pass"] = S["diff"][0] > q95
    df["pass"] = df.sign_ok & df.diff_pass
    df["tot_pass"] = df.sign_ok & (S["tot"][0] > tq95)
    return df


def persistence(Ss, Sc, J, mask):
    """고르기 → 확인: 통과 겹침, 고르기 1위의 확인 순위, 순위 상관(각각 귀무 k로 같은 절차).
    1위·순위 기준 두 가지: 하루치 환산 차이(diff_day, 크기) · 표준화 점수(zs, 자기 귀무 대비). 순위 1 = 최고."""
    idx = np.flatnonzero(mask)
    Ps, Pc = pass_mat(Ss["diff"], Ss["sign"]), pass_mat(Sc["diff"], Sc["sign"])
    ov = (Ps[:, idx] & Pc[:, idx]).sum(axis=1)
    ns = Ps[:, idx].sum(axis=1)
    out = dict(n=len(idx), sel_pass=int(ns[0]), sel_pass_also_conf=int(ov[0]), overlap_null_mean=round(float(ov[1:].mean()), 2),
               overlap_null_q95=float(np.quantile(ov[1:], 0.95)), overlap_p=round(pval_ge(ov[0], ov[1:]), 3))
    Rs = pass_low(Ss["diff"], (Ss["amb"] > 0) & (Ss["trend"] < 0))
    Rc = pass_low(Sc["diff"], (Sc["amb"] > 0) & (Sc["trend"] < 0))
    rv = (Rs[:, idx] & Rc[:, idx]).sum(axis=1)
    zneg = ((Ss["zs"][:, idx] < 0) & (Sc["zs"][:, idx] < 0)).sum(axis=1)
    out.update(rev_sel=int(Rs[0, idx].sum()), rev_conf=int(Rc[0, idx].sum()), rev_both=int(rv[0]), rev_both_null_mean=round(float(rv[1:].mean()), 2),
               rev_both_null_max=int(rv[1:].max()), rev_both_p=round(pval_ge(rv[0], rv[1:]), 3),
               zneg_both=int(zneg[0]), zneg_both_null_med=float(np.median(zneg[1:])), zneg_both_null_q95=float(np.quantile(zneg[1:], 0.95)),
               zneg_both_p=round(pval_ge(zneg[0], zneg[1:]), 3),
               rev_both_list=[f"b{J.b[i]}·{J.clf[i]}" for i in idx[(Rs[0, idx] & Rc[0, idx])]])
    K = Ss["diff"].shape[0]
    for key in ("diff_day", "zs"):
        vs, vc = Ss[key][:, idx], Sc[key][:, idx]
        rho = np.array([spearmanr(vs[k], vc[k]).statistic for k in range(K)])
        top = np.argmax(vs, axis=1)
        rank = np.array([int((vc[k] > vc[k, top[k]]).sum()) + 1 for k in range(K)])
        t0 = idx[top[0]]
        out[key] = dict(sel_top=f"b{J.b[t0]}·{J.clf[t0]}", sel_top_sel_diff_day=round(float(Ss["diff_day"][0, t0]), 4),
                        sel_top_sel_p=round(pval_ge(Ss["diff"][0, t0], Ss["diff"][1:, t0]), 3),
                        sel_top_conf_diff_day=round(float(Sc["diff_day"][0, t0]), 4),
                        sel_top_conf_trend_day=round(float(Sc["trend_day"][0, t0]), 4), sel_top_conf_amb_day=round(float(Sc["amb_day"][0, t0]), 4),
                        sel_top_conf_p=round(pval_ge(Sc["diff"][0, t0], Sc["diff"][1:, t0]), 3), sel_top_conf_pass=bool(Pc[0, t0]),
                        sel_top_conf_rank=int(rank[0]), rank_null_med=float(np.median(rank[1:])), rank_p=round(pval_le(rank[0], rank[1:]), 3),
                        rho=round(float(rho[0]), 3), rho_null_med=round(float(np.median(rho[1:])), 3),
                        rho_null_q95=round(float(np.quantile(rho[1:], 0.95)), 3), rho_p=round(pval_ge(rho[0], rho[1:]), 3))
    return out


def zscale(S, J, mask):
    """'큰 봉일수록 애매를 잘 알아보나': (b, 종류) 칸 안에서 판정 시간봉 순서와 표준화 점수의 순위 상관, 칸 평균. 귀무 k로."""
    cells = []
    for (b, t), g in J[mask].groupby(["b", "type"]):
        if len(g) >= 3:
            order = g.zmin.values  # 일봉 = 1440(가장 큼)
            cells.append((g.index.values, order))
    K = S["zs"].shape[0]
    stat = np.zeros(K)
    for k in range(K):
        stat[k] = np.mean([spearmanr(o, S["zs"][k, ix]).statistic for ix, o in cells])
    # 전체 판정 순위 상관(절대 판정 시간봉, b 섞음)
    ix = np.flatnonzero(mask)
    allr = np.array([spearmanr(J.zmin.values[ix], S["zs"][k, ix]).statistic for k in range(K)])
    return dict(cells=len(cells), within_rho=round(float(stat[0]), 3), within_null_med=round(float(np.median(stat[1:])), 3),
                within_null_q95=round(float(np.quantile(stat[1:], 0.95)), 3), within_p=round(pval_ge(stat[0], stat[1:]), 3),
                all_rho=round(float(allr[0]), 3), all_null_med=round(float(np.median(allr[1:])), 3),
                all_null_q95=round(float(np.quantile(allr[1:], 0.95)), 3), all_p=round(pval_ge(allr[0], allr[1:]), 3))


def group_table(S, J, P, bd, w, keys):
    rows = []
    for key, g in J.groupby(keys, sort=False):
        ix = g.index.values
        gz = np.median(S["zs"][:, ix], axis=1)
        gd = np.median(S["diff_day"][:, ix], axis=1)
        cnt = P[:, ix].sum(axis=1)
        key = key if isinstance(key, tuple) else (key,)
        r = dict(board=bd, win=w, **dict(zip(keys, key)), n=len(ix), struct=bool(g.struct.all()),
                 trend_day=float(np.median(S["trend_day"][0, ix])), amb_day=float(np.median(S["amb_day"][0, ix])),
                 diff_day=float(gd[0]), diff_day_null_med=float(np.median(gd[1:])),
                 neg_frac=float(np.median(S["negfrac"][0, ix])), neg_frac_null=float(np.median(S["negfrac"][1:, ix])),
                 z_med=float(gz[0]), z_null_q95=float(np.quantile(gz[1:], 0.95)), z_p=pval_ge(gz[0], gz[1:]),
                 sign_ok=int(S["sign"][0, ix].sum()), passed=int(cnt[0]), pass_null_mean=float(cnt[1:].mean()),
                 pass_p=pval_ge(cnt[0], cnt[1:]))
        rows.append(r)
    return rows


def main():
    J = judgments()
    assert len(J) == 163 and J.clean.sum() == 120, (len(J), J.clean.sum())
    A, ks, syms = load_arrays(J)
    assert len(ks) == 41 and len(syms) == 144
    tabs, groups, summ = [], [], dict(n_judgments=len(J), n_clean=int(J.clean.sum()), n_noext=int(J.noext.sum()),
                                      n_struct=int(J.struct.sum()), n_shifts=len(ks) - 1, n_stocks=len(syms), shifts=[ks[1], ks[-1]])
    SS = {}
    for bd in BOARDS:
        for w in WN:
            S = stats(A[(bd, w)], J)
            SS[(bd, w)] = S
            tabs.append(judgment_table(S, J, bd, w))
            P = pass_mat(S["diff"], S["sign"])
            Pt = pass_mat(S["tot"], S["sign"])
            Pd = pass_mat(S["diff"], np.ones_like(S["sign"]))
            rsign = (S["amb"] > 0) & (S["trend"] < 0)
            Pr = pass_low(S["diff"], rsign)
            Prd = pass_low(S["diff"], np.ones_like(rsign))
            fk = f"{bd}_{w}"
            summ.setdefault("reverse", {})[fk] = {nm: dict(pass_=fam(Pr, J[nm].values), diff_only=fam(Prd, J[nm].values),
                                                            rsign=int(rsign[0, J[nm].values].sum()), mean_z=mean_z(S, J[nm].values))
                                                  for nm in ("clean", "noext", "zgtb", "struct")}
            summ.setdefault("family", {})[fk] = {nm: dict(pass_=fam(P, J[nm].values), diff_only=fam(Pd, J[nm].values),
                                                           tot=fam(Pt, J[nm].values), sign_ok=int(S["sign"][0, J[nm].values].sum()))
                                                 for nm in ("clean", "noext", "zgtb", "struct")}
            summ.setdefault("zscale", {})[fk] = {nm: zscale(S, J, J[nm].values) for nm in ("clean", "noext", "zgtb")}
            for keys in (["type", "m"], ["type"], ["m"], ["b"]):
                whole = keys[0] == "type"  # 종류 묶음은 전부(struct 표시), m·b 묶음은 clean만
                gr = group_table(S, J if whole else J[J.clean], P, bd, w, keys)
                for r in gr:
                    r["keys"] = "+".join(keys)
                    r["subset"] = "all" if whole else "clean"
                groups += gr
    for bd in BOARDS:
        summ.setdefault("persistence", {})[bd] = {nm: persistence(SS[(bd, "sel")], SS[(bd, "conf")], J, J[nm].values)
                                                  for nm in ("clean", "noext", "zgtb", "struct")}
    T = pd.concat(tabs, ignore_index=True)
    G = pd.DataFrame(groups)
    # 시간봉 지도: 종류(EXT는 z = b 구조와 z > b를 나눔) × 배수 m, 판·구간별 판정 중앙 표준화 점수와 통과 수
    kind = np.where(J.type == "EXT", np.where(J.struct, "EXT(z=b, 구조)", "EXT"), np.where(J.type == "AGREE", "AGREE(구조)", J.type))
    zm = []
    for bd in BOARDS:
        for w in WN:
            S = SS[(bd, w)]
            P = pass_mat(S["diff"], S["sign"])
            Pr = pass_low(S["diff"], (S["amb"] > 0) & (S["trend"] < 0))
            for (kd, m), g in J.assign(kind=kind).groupby(["kind", "m"], sort=False):
                ix = g.index.values
                zm.append(dict(board=bd, win=w, kind=kd, m=m, n=len(ix), z_med=float(np.median(S["zs"][0, ix])),
                               diff_day=float(np.median(S["diff_day"][0, ix])), trend_day=float(np.median(S["trend_day"][0, ix])),
                               amb_day=float(np.median(S["amb_day"][0, ix])), neg_frac=float(np.median(S["negfrac"][0, ix])),
                               passed=int(P[0, ix].sum()), rev_passed=int(Pr[0, ix].sum())))
    ZM = pd.DataFrame(zm)
    ZM.to_csv(OUT / "mtfdiag_zmap.csv", index=False, encoding="utf-8-sig")
    T.to_csv(OUT / "mtfdiag_judgments.csv", index=False, encoding="utf-8-sig")
    G.to_csv(OUT / "mtfdiag_groups.csv", index=False, encoding="utf-8-sig")
    # 눈여겨볼 판정: 기본값·전략 고르기 1위의 판정
    pick = {}
    for nm, (b, c) in (("default", DEFAULT), ("sel_top_strategy", SEL_TOP_STRAT)):
        r = T[(T.b == b) & (T.clf == c)]
        pick[nm] = r[["board", "win", "trend_day", "amb_day", "diff_day", "diff_p", "tot_diff", "sign_ok", "pass", "trend_share", "struct"]].round(4).to_dict("records")
    summ["picks"] = pick
    cr = T[(T.board == "real0") & (T.win == "conf")]
    summ["conf_real0_pass_list"] = cr[cr["pass"]][["b", "clf", "struct", "trend_day", "amb_day", "diff_day", "diff_p"]].round(4).to_dict("records")
    summ["conf_real0_top5_clean_z"] = cr[cr.clean].sort_values("diff_z", ascending=False).head(5)[
        ["b", "clf", "trend_day", "amb_day", "diff_day", "diff_z", "diff_p", "sign_ok", "pass"]].round(4).to_dict("records")
    sr = T[(T.board == "real0") & (T.win == "sel")]
    summ["sel_real0_pass_list"] = sr[sr["pass"]][["b", "clf", "struct", "trend_day", "amb_day", "diff_day", "diff_p"]].round(4).to_dict("records")
    summ["sign_frac_conf_real0"] = dict(trend_pos=int((cr.trend_pb > 0).sum()), amb_neg=int((cr.amb_pb < 0).sum()), n=len(cr),
                                        clean_trend_pos=int((cr[cr.clean].trend_pb > 0).sum()), clean_amb_neg=int((cr[cr.clean].amb_pb < 0).sum()))
    (OUT / "mtfdiag_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1,
                                                         default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
    for k, v in summ["family"].items():
        print(k, json.dumps(v, ensure_ascii=False))
    for k, v in summ["reverse"].items():
        print("reverse", k, json.dumps(v, ensure_ascii=False))
    for k, v in summ["persistence"].items():
        print("persist", k, json.dumps(v, ensure_ascii=False))
    for k, v in summ["zscale"].items():
        print("zscale", k, json.dumps(v, ensure_ascii=False))
    print(json.dumps({k: summ[k] for k in ("picks", "conf_real0_pass_list", "sel_real0_pass_list", "conf_real0_top5_clean_z", "sign_frac_conf_real0")},
                     ensure_ascii=False, indent=1, default=str))
    print(ZM.round(3).to_string(index=False))
    gcols = ["board", "win", "keys", "subset", "type", "m", "b", "n", "struct", "trend_day", "amb_day", "diff_day", "diff_day_null_med",
             "z_med", "z_null_q95", "z_p", "sign_ok", "passed", "pass_null_mean", "pass_p"]
    print(G[[c for c in gcols if c in G]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
