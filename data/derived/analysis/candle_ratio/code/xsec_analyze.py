"""xsec 갈래 분석: results/xsec/<base>/*.npy → 종목별 요약·칸별 종목 집계·핵심 숫자.

py code/xsec_analyze.py 5m
→ results/xsec_summary_<base>.parquet  종목별 요약
→ results/xsec_cells_<base>.parquet    칸별 종목 집계(노출 맞춘 초과 중앙값·+비율)
→ results/xsec_key_<base>.json         핵심 숫자

노출 맞춘 초과(로그) = ret − 보유비율 × 같은 세션·시간봉 보유 로그수익. 보고는 exp(x)−1 (%).
프리·애프터가 듬성한 종목: 5분봉 기준 연장 하루 봉 수 중앙값 < 150 (메타는 5m에서 가져온다).
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent.parent
XS = HERE / "results" / "xsec"
SPARSE_BPD = 150

ORIG = dict(tf=40, N=10, ratio="count", dir="follow", dayflat=False, b=0.5, s=0.5)  # 원안(세션 eth·rth 둘 다)
MU1 = dict(sess="eth", tf=70, N=15, ratio="vol", dir="fade", dayflat=False, b=0.8, s=0.2)  # MU 고르기 1위


def pct(x):
    return float(np.expm1(x) * 100)


def cell_idx(cells, **kw):
    m = np.ones(len(cells), bool)
    for k, v in kw.items():
        m &= (cells[k] == v).to_numpy()
    return np.flatnonzero(m)


def load_all(base, sub):
    cells = pd.read_parquet(XS / f"cells_{base}.parquet")
    syms = sorted(p.stem for p in (XS / sub).glob("*.npy") if not p.stem.endswith("_ent"))
    key = cells.sess.astype(str) + "_" + cells.tf.astype(str)
    R = {}
    for k in ("rs", "rc", "hs", "hc", "bs", "bc", "es", "ec"):
        R[k] = np.empty((len(syms), len(cells)), np.float32)
    metas = []
    for i, s in enumerate(syms):
        a = np.load(XS / sub / f"{s}.npy")
        e = np.load(XS / sub / f"{s}_ent.npy")
        m = json.loads((XS / sub / f"{s}_meta.json").read_text(encoding="utf-8"))
        bh = pd.DataFrame(m["bh"])
        bk = bh.sess.astype(str) + "_" + bh.tf.astype(str)
        R["rs"][i], R["rc"][i], R["hs"][i], R["hc"][i] = a
        R["bs"][i] = key.map(dict(zip(bk, bh.bh_sel))).to_numpy(np.float32)
        R["bc"][i] = key.map(dict(zip(bk, bh.bh_conf))).to_numpy(np.float32)
        R["es"][i], R["ec"][i] = e
        metas.append({k: v for k, v in m.items() if k != "bh"} | {
            f"bh_{ss}_{p}": float(bh[(bh.sess == ss)].sort_values("tf")[f"bh_{p}"].iloc[0])
            for ss in ("eth", "rth") for p in ("sel", "conf") if (bh.sess == ss).any()})
    R["xs"] = R["rs"] - R["hs"] * R["bs"]
    R["xc"] = R["rc"] - R["hc"] * R["bc"]
    meta = pd.DataFrame(metas).set_index("sym")
    return cells, syms, R, meta


def per_stock(cells, syms, R, meta, base):
    sess = cells.sess.to_numpy()
    dirs = cells.dir.to_numpy()
    rows = []
    orig = ORIG if (cells.tf == ORIG["tf"]).any() else ORIG | {"tf": 60}  # 1시간봉 창에는 40분이 없어 60분 판으로 본다
    o_eth = cell_idx(cells, sess="eth", **orig)[0]
    o_rth = cell_idx(cells, sess="rth", **orig)[0]
    mu1 = cell_idx(cells, **MU1)
    mu1 = mu1[0] if len(mu1) else None
    for i, s in enumerate(syms):
        rs, rc, xs, xc, bc, hc = R["rs"][i], R["rc"][i], R["xs"][i], R["xc"][i], R["bc"][i], R["hc"][i]
        ok = ~np.isnan(rs) & ~np.isnan(rc)
        row = dict(sym=s)
        for scope, msk in (("all", ok), ("rth", ok & (sess == "rth")), ("eth", ok & (sess == "eth"))):
            idx = np.flatnonzero(msk)
            j = idx[np.argmax(rs[idx])]
            top = idx[np.argsort(-rs[idx], kind="stable")[:20]]
            row |= {
                f"{scope}_top1": f"{cells.sess[j]} {cells.tf[j]}m N{cells.N[j]} {cells.ratio[j]} {cells.dir[j]} "
                                 f"{cells.b[j]}/{cells.s[j]}{' dayflat' if cells.dayflat[j] else ''}",
                f"{scope}_top1_sel": rs[j], f"{scope}_top1_conf": rc[j], f"{scope}_top1_bhconf": bc[j],
                f"{scope}_top1_xconf": xc[j], f"{scope}_top1_hconf": hc[j],
                f"{scope}_top1_ent": f"{int(R['es'][i][j])}/{int(R['ec'][i][j])}",
                f"{scope}_top20_conf_med": float(np.median(rc[top])),
                f"{scope}_top20_xconf_med": float(np.median(xc[top])),
                f"{scope}_top20_beat": float(np.mean(rc[top] > bc[top])),
                f"{scope}_rho": float(spearmanr(rs[idx], rc[idx]).statistic),
                f"{scope}_cells_beat_conf": float(np.mean(rc[idx] > bc[idx])),
                f"{scope}_cells_xpos_conf": float(np.mean(xc[idx] > 0)),
                f"{scope}_cells_xconf_med": float(np.median(xc[idx])),
            }
        for d in ("follow", "fade"):
            m = ok & (dirs == d)
            row[f"x_{d}_sel_med"] = float(np.median(xs[m]))
            row[f"x_{d}_conf_med"] = float(np.median(xc[m]))
        for nm, j in (("orig_eth", o_eth), ("orig_rth", o_rth), ("mu1", mu1)):
            if j is None:
                continue
            row |= {f"{nm}_sel": rs[j], f"{nm}_conf": rc[j], f"{nm}_bhsel": R["bs"][i][j], f"{nm}_bhconf": bc[j],
                    f"{nm}_xsel": xs[j], f"{nm}_xconf": xc[j], f"{nm}_hsel": R["hs"][i][j], f"{nm}_hconf": hc[j],
                    f"{nm}_ent": f"{int(R['es'][i][j])}/{int(R['ec'][i][j])}"}
        rows.append(row)
    df = pd.DataFrame(rows).set_index("sym")
    return meta.join(df)


def per_cell(cells, R, sparse):
    out = cells.copy()
    dense = ~sparse
    eth = (cells.sess == "eth").to_numpy()
    for nm, rows in (("all", np.ones(len(sparse), bool)), ("dense", dense)):
        xs, xc = R["xs"][rows], R["xc"][rows]
        out[f"{nm}_xs_med"] = np.nanmedian(xs, axis=0)
        out[f"{nm}_xc_med"] = np.nanmedian(xc, axis=0)
        out[f"{nm}_xs_pos"] = np.mean(xs > 0, axis=0)
        out[f"{nm}_xc_pos"] = np.mean(xc > 0, axis=0)
        out[f"{nm}_rc_med"] = np.nanmedian(R["rc"][rows], axis=0)
        out[f"{nm}_beat_conf"] = np.mean(R["rc"][rows] > R["bc"][rows], axis=0)
        out[f"{nm}_hc_med"] = np.nanmedian(R["hc"][rows], axis=0)
        out[f"{nm}_ent_med"] = np.median(R["es"][rows] + R["ec"][rows], axis=0)
    # 판정용: 연장 칸은 촘촘한 종목만, 정규장 칸은 전 종목
    for k in ("xs_med", "xc_med", "xs_pos", "xc_pos", "rc_med", "beat_conf", "hc_med", "ent_med"):
        out[f"use_{k}"] = np.where(eth, out[f"dense_{k}"], out[f"all_{k}"])
    out["region"] = ((out.use_xs_med > 0) & (out.use_xc_med > 0) & (out.use_xs_pos >= 0.6) & (out.use_xc_pos >= 0.6))
    return out


def shape(df, cols=("sess", "dir", "ratio", "dayflat", "N", "b", "s")):
    d = {}
    for c in cols:
        d[c] = df[c].value_counts().sort_index().to_dict()
    tfb = pd.cut(df.tf, [0, 30, 60, 120, 240, 360], labels=["5-30", "35-60", "65-120", "125-240", "245-360"])
    d["tf_band"] = tfb.value_counts().sort_index().to_dict()
    return {k: {str(a): int(b) for a, b in v.items()} for k, v in d.items()}


def key_numbers(S, C, base):
    sparse = S["eth_sparse"].to_numpy()
    K = {"n_stocks": int(len(S)), "n_sparse": int(sparse.sum()), "n_cells": int(len(C)),
         "sparse_syms": sorted(S.index[sparse].tolist())}
    for grp, m in (("all", np.ones(len(S), bool)), ("dense", ~sparse), ("sparse", sparse)):
        s = S[m]
        g = {"n": int(m.sum())}
        for sc in ("all", "rth", "eth"):
            g[sc] = {
                "top1_beat_bh": float(np.mean(s[f"{sc}_top1_conf"] > s[f"{sc}_top1_bhconf"])),
                "top1_xpos": float(np.mean(s[f"{sc}_top1_xconf"] > 0)),
                "top1_sel_med%": pct(s[f"{sc}_top1_sel"].median()),
                "top1_conf_med%": pct(s[f"{sc}_top1_conf"].median()),
                "top1_bhconf_med%": pct(s[f"{sc}_top1_bhconf"].median()),
                "top1_xconf_med%": pct(s[f"{sc}_top1_xconf"].median()),
                "top1_hconf_med": float(s[f"{sc}_top1_hconf"].median()),
                "top20_conf_med%": pct(s[f"{sc}_top20_conf_med"].median()),
                "top20_xconf_med%": pct(s[f"{sc}_top20_xconf_med"].median()),
                "top20_beat_mean": float(s[f"{sc}_top20_beat"].mean()),
                "rho_med": float(s[f"{sc}_rho"].median()),
                "rho_pos_frac": float(np.mean(s[f"{sc}_rho"] > 0)),
                "cells_beat_conf_med": float(s[f"{sc}_cells_beat_conf"].median()),
                "cells_xpos_conf_med": float(s[f"{sc}_cells_xpos_conf"].median()),
                "cells_xconf_med%": pct(s[f"{sc}_cells_xconf_med"].median()),
            }
        for d in ("follow", "fade"):
            g[f"x_{d}_sel_med%"] = pct(s[f"x_{d}_sel_med"].median())
            g[f"x_{d}_conf_med%"] = pct(s[f"x_{d}_conf_med"].median())
            g[f"x_{d}_conf_pos"] = float(np.mean(s[f"x_{d}_conf_med"] > 0))
            g[f"x_{d}_sel_pos"] = float(np.mean(s[f"x_{d}_sel_med"] > 0))
        g["bh_rth_sel_med%"] = pct(s.bh_rth_sel.median())
        g["bh_rth_conf_med%"] = pct(s.bh_rth_conf.median())
        g["bh_rth_conf_neg_frac"] = float(np.mean(s.bh_rth_conf < 0))
        g["bh_eth_conf_med%"] = pct(s.bh_eth_conf.median())
        for nm in ("orig_eth", "orig_rth", "mu1"):
            if f"{nm}_sel" not in s:
                continue
            g[nm] = {
                "sel_med%": pct(s[f"{nm}_sel"].median()), "conf_med%": pct(s[f"{nm}_conf"].median()),
                "bhsel_med%": pct(s[f"{nm}_bhsel"].median()), "bhconf_med%": pct(s[f"{nm}_bhconf"].median()),
                "beat_sel": float(np.mean(s[f"{nm}_sel"] > s[f"{nm}_bhsel"])),
                "beat_conf": float(np.mean(s[f"{nm}_conf"] > s[f"{nm}_bhconf"])),
                "xpos_sel": float(np.mean(s[f"{nm}_xsel"] > 0)), "xpos_conf": float(np.mean(s[f"{nm}_xconf"] > 0)),
                "xsel_med%": pct(s[f"{nm}_xsel"].median()), "xconf_med%": pct(s[f"{nm}_xconf"].median()),
                "hsel_med": float(s[f"{nm}_hsel"].median()), "hconf_med": float(s[f"{nm}_hconf"].median()),
            }
        K[grp] = g
    # 칸별 영역
    reg = C[C.region]
    K["region_n"] = int(len(reg))
    K["region_shape"] = shape(reg) if len(reg) else {}
    K["cells_xs_med_pos"] = int((C.use_xs_med > 0).sum())
    K["cells_xc_med_pos"] = int((C.use_xc_med > 0).sum())
    K["cells_both_med_pos"] = int(((C.use_xs_med > 0) & (C.use_xc_med > 0)).sum())
    K["cells_both_pos60"] = int(((C.use_xs_pos >= 0.6) & (C.use_xc_pos >= 0.6)).sum())
    K["cells_conf_pos60"] = int((C.use_xc_pos >= 0.6).sum())
    K["cells_sel_pos60"] = int((C.use_xs_pos >= 0.6).sum())
    K["cell_rho_xs_xc_med"] = float(spearmanr(C.use_xs_med, C.use_xc_med).statistic)
    if len(reg):
        top = reg.assign(score=np.minimum(reg.use_xs_pos, reg.use_xc_pos)).sort_values("score", ascending=False).head(15)
        K["region_top"] = [
            {"cell": f"{r.sess} {r.tf}m N{r.N} {r.ratio} {r.dir} {r.b}/{r.s}{' dayflat' if r.dayflat else ''}",
             "xs_med%": pct(r.use_xs_med), "xc_med%": pct(r.use_xc_med), "xs_pos": float(r.use_xs_pos),
             "xc_pos": float(r.use_xc_pos), "rc_med%": pct(r.use_rc_med), "beat_conf": float(r.use_beat_conf),
             "hc_med": float(r.use_hc_med), "ent_med": float(r.use_ent_med)} for r in top.itertuples()]
    return K


if __name__ == "__main__":
    base = sys.argv[1]
    sub = sys.argv[2] if len(sys.argv) > 2 else base  # 귀무 폴더(예: 5m_shift20)
    cells, syms, R, meta = load_all(base, sub)
    m5 = {p.stem[:-5]: json.loads(p.read_text(encoding="utf-8"))["bpd_eth"] for p in (XS / "5m").glob("*_meta.json")}
    bpd = pd.Series(m5).reindex(meta.index)
    meta["bpd_eth_5m"] = bpd
    meta["eth_sparse"] = (bpd < SPARSE_BPD).fillna(False)
    S = per_stock(cells, syms, R, meta, base)
    S.to_parquet(HERE / "results" / f"xsec_summary_{sub}.parquet")
    C = per_cell(cells, R, S["eth_sparse"].to_numpy())
    C.to_parquet(HERE / "results" / f"xsec_cells_{sub}.parquet")
    K = key_numbers(S, C, base)
    (HERE / "results" / f"xsec_key_{sub}.json").write_text(json.dumps(K, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(K, ensure_ascii=False, indent=1, default=float))
