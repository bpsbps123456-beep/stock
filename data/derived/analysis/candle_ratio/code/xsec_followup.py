"""xsec 갈래 후속: 칸별 영역의 모양(비율), 비용 반영, 종목 전반 고르기→확인(칸 단위 전방 검증).

py code/xsec_followup.py 5m [sub]
→ results/xsec_followup_<sub>.json

비용: 수수료 0 기본 + 민감도 편도 정규장 0.05%, 연장 0.1%(BRIEF). 저장된 진입 수로 근사:
net = ret + 2 × 진입 수 × log(1−c). MU 격자 실제 비용 판과 차이는 최대 비용 1회분(구간 끝에 열린 포지션).
"""
import json
import sys

import numpy as np
import pandas as pd

from xsec_analyze import HERE, SPARSE_BPD, XS, load_all, pct

COST = {"eth": 0.001, "rth": 0.0005}


def region_mask(xs_med, xc_med, xs_pos, xc_pos):
    return (xs_med > 0) & (xc_med > 0) & (xs_pos >= 0.6) & (xc_pos >= 0.6)


def cell_stats(R, rows_eth, rows_rth, eth_cells, net=False, cost_lc=None):
    """칸별 종목 집계(연장 칸은 촘촘한 종목, 정규장 칸은 전 종목)."""
    out = {}
    for nm, rows in (("eth", rows_eth), ("rth", rows_rth)):
        rs, rc = R["rs"][rows], R["rc"][rows]
        if net:
            rs = rs + 2 * R["es"][rows] * cost_lc
            rc = rc + 2 * R["ec"][rows] * cost_lc
        xs = rs - R["hs"][rows] * R["bs"][rows]
        xc = rc - R["hc"][rows] * R["bc"][rows]
        out[nm] = dict(xs_med=np.nanmedian(xs, 0), xc_med=np.nanmedian(xc, 0), xs_pos=np.mean(xs > 0, 0),
                       xc_pos=np.mean(xc > 0, 0), rc_med=np.nanmedian(rc, 0), rs_med=np.nanmedian(rs, 0),
                       beat_conf=np.mean(rc > R["bc"][rows], 0), beat_sel=np.mean(rs > R["bs"][rows], 0))
    return {k: np.where(eth_cells, out["eth"][k], out["rth"][k]) for k in out["eth"]}


def rates(cells, mask):
    d = {}
    for c in ("sess", "dir", "ratio", "dayflat", "N", "b", "s"):
        d[c] = {str(k): round(float(v), 4) for k, v in mask.groupby(cells[c]).mean().items()}
    tfb = pd.cut(cells.tf, [0, 15, 30, 60, 120, 240, 360])
    d["tf_band"] = {str(k): round(float(v), 4) for k, v in mask.groupby(tfb, observed=True).mean().items()}
    d["sess_dir"] = {f"{a}_{b}": round(float(v), 4) for (a, b), v in mask.groupby([cells.sess, cells.dir]).mean().items()}
    return d


def describe(cells, st, idx):
    return [{"cell": f"{cells.sess[j]} {cells.tf[j]}m N{cells.N[j]} {cells.ratio[j]} {cells.dir[j]} {cells.b[j]}/{cells.s[j]}"
                     f"{' dayflat' if cells.dayflat[j] else ''}",
             "xs_med%": round(pct(st["xs_med"][j]), 2), "xc_med%": round(pct(st["xc_med"][j]), 2),
             "xs_pos": round(float(st["xs_pos"][j]), 3), "xc_pos": round(float(st["xc_pos"][j]), 3),
             "rs_med%": round(pct(st["rs_med"][j]), 2), "rc_med%": round(pct(st["rc_med"][j]), 2),
             "beat_conf": round(float(st["beat_conf"][j]), 3)} for j in idx]


def forward(st, ks=(1, 20, 100, 1000)):
    """칸 단위 전방 검증: 고르기 구간 종목 중앙 초과로 칸을 골라 확인 구간 종목 중앙 초과를 본다."""
    order = np.argsort(-np.nan_to_num(st["xs_med"], nan=-9), kind="stable")
    res = {}
    for k in ks:
        top = order[:k]
        res[f"top{k}"] = {"xs_med%": round(pct(np.median(st["xs_med"][top])), 2),
                          "xc_med%": round(pct(np.median(st["xc_med"][top])), 2),
                          "xc_pos_med": round(float(np.median(st["xc_pos"][top])), 3),
                          "frac_xc_med_pos": round(float(np.mean(st["xc_med"][top] > 0)), 3),
                          "beat_conf_med": round(float(np.median(st["beat_conf"][top])), 3)}
    # 고르기 구간 60% 이상 종목에서 + 인 칸 → 확인
    sel60 = (st["xs_pos"] >= 0.6) & (st["xs_med"] > 0)
    res["sel60"] = {"n": int(sel60.sum()), "xc_med%": round(pct(np.median(st["xc_med"][sel60])), 2) if sel60.any() else None,
                    "frac_xc_med_pos": round(float(np.mean(st["xc_med"][sel60] > 0)), 3) if sel60.any() else None,
                    "frac_xc_pos60": round(float(np.mean(st["xc_pos"][sel60] >= 0.6)), 3) if sel60.any() else None}
    return res


def per_stock_net(R, cells, cost_lc, eth_cells, sparse):
    """종목별 고르기 1위(비용 반영 수익)를 확인 구간에 쓰기."""
    rs = R["rs"] + 2 * R["es"] * cost_lc
    rc = R["rc"] + 2 * R["ec"] * cost_lc
    out = {}
    for scope, cm in (("all", np.ones(len(cells), bool)), ("rth", ~eth_cells)):
        r = np.where(cm[None, :], np.nan_to_num(rs, nan=-9), -9)
        j = np.argmax(r, axis=1)
        i = np.arange(len(j))
        bc = R["bc"][i, j]
        x = rc[i, j] - R["hc"][i, j] * bc
        out[scope] = {"beat_bh": round(float(np.mean(rc[i, j] > bc)), 3), "xpos": round(float(np.mean(x > 0)), 3),
                      "conf_med%": round(pct(np.median(rc[i, j])), 2), "bhconf_med%": round(pct(np.median(bc)), 2),
                      "xconf_med%": round(pct(np.median(x)), 2),
                      "dense_beat_bh": round(float(np.mean((rc[i, j] > bc)[~sparse])), 3),
                      "dense_xpos": round(float(np.mean((x > 0)[~sparse])), 3)}
    return out


if __name__ == "__main__":
    base = sys.argv[1]
    sub = sys.argv[2] if len(sys.argv) > 2 else base
    cells, syms, R, meta = load_all(base, sub)
    m5 = {p.stem[:-5]: json.loads(p.read_text(encoding="utf-8"))["bpd_eth"] for p in (XS / "5m").glob("*_meta.json")}
    sparse = (pd.Series(m5).reindex(syms) < SPARSE_BPD).fillna(False).to_numpy()
    eth_cells = (cells.sess == "eth").to_numpy()
    cost_lc = np.where(eth_cells, np.log(1 - COST["eth"]), np.log(1 - COST["rth"])).astype(np.float32)[None, :]
    rows_all = np.ones(len(syms), bool)
    K = {"n_stocks": len(syms), "n_cells": len(cells)}
    for tag, net in (("gross", False), ("net", True)):
        st = cell_stats(R, ~sparse, rows_all, eth_cells, net=net, cost_lc=cost_lc)
        reg = pd.Series(region_mask(st["xs_med"], st["xc_med"], st["xs_pos"], st["xc_pos"]))
        score = np.where(reg, np.minimum(st["xs_pos"], st["xc_pos"]) + 1e-3 * (st["xs_med"] + st["xc_med"]), -9)
        top = np.argsort(-score, kind="stable")[: min(15, int(reg.sum()))]
        K[tag] = {"region_n": int(reg.sum()), "region_rate": rates(cells, reg) if reg.any() else {},
                  "region_top": describe(cells, st, top), "forward": forward(st),
                  "cells_xc_med_pos": int((st["xc_med"] > 0).sum()), "cells_xs_med_pos": int((st["xs_med"] > 0).sum()),
                  "beat_conf_med_all_cells": round(float(np.median(st["beat_conf"])), 3)}
        if net:
            K[tag]["per_stock_top1"] = per_stock_net(R, cells, cost_lc, eth_cells, sparse)
        # 영역 칸을 rth·fade·tf≤15·N≤15로 묶은 모양 확인용 수치
        fam = (cells.sess == "rth") & (cells.dir == "fade") & (cells.tf <= 15)
        K[tag]["family_rth_fade_tf<=15"] = {"n": int(fam.sum()), "region_rate": round(float(reg[fam].mean()), 4),
                                            "xs_med%": round(pct(np.median(st["xs_med"][fam])), 2),
                                            "xc_med%": round(pct(np.median(st["xc_med"][fam])), 2)}
    (HERE / "results" / f"xsec_followup_{sub}.json").write_text(json.dumps(K, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(K, ensure_ascii=False, indent=1))
