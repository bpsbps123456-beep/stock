"""awake 갈래 분석: awake_grid.py 결과(실제 k=0 + 공통 k 원형 이동 귀무)를 읽어 절차별 확인 성적과 귀무 분위를 낸다.

py code/awake_analyze.py 5m   (AWAKE_TMP는 awake_grid.py와 같게)
→ results/awake_summary_<base>.json  절차별 실제·귀무 중앙·95분위·분위·p, 남는 칸 수, BE·NBIS·MU
→ results/awake_null_<base>.csv      k별 통계(행 0 = 실제)
→ results/awake_stocks_<base>.csv    종목별 원안·자기 1위·상위 20칸·단서 칸
→ results/awake_top20_<base>.csv     칸별 종목 중앙 '고르기 상위 20칸'(두 기준)
→ results/awake_pass_<base>.csv      두 구간 모두 종목 60% 이상에서 +인 칸(실제)

지표(확인 구간, 종목별)
- dbh: 단순 수익 − 보유 단순 수익(%p)
- ex: 노출 맞춘 초과 = 로그 수익 − x·보유 로그 수익(xsec와 같은 정의)
- dd: static 대비 낙폭 축소 = 전략 최대 낙폭(단순) − static 최대 낙폭(단순). +면 전략이 덜 빠졌다.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from awake_grid import COLS8, COLS12, OUT, TMP, ORIG_TF, specials  # noqa: E402
from engine import HERE, WIN, load, segments, tf_bars  # noqa: E402

RES = HERE / "results"
FOCUS = ["MU", "BE", "NBIS"]
PASS = 0.6
C8 = {c: i for i, c in enumerate(COLS8)}
C12 = {c: i for i, c in enumerate(COLS12)}


def pct(v):
    return None if v is None or not np.isfinite(v) else round(float(np.expm1(v)) * 100, 1)


def pp(v):
    return None if v is None or not np.isfinite(v) else round(float(v) * 100, 1)


def med(v):
    v = np.asarray(v, np.float64)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan


def pos(v):
    v = np.asarray(v, np.float64)
    m = np.isfinite(v)
    return float((v[m] > 0).mean()) if m.any() else np.nan


def metrics(A, BHS, BHC):
    """A: (S, 8, M) → 종목×칸 지표 사전."""
    rs, rc = A[:, C8["ret_sel"]].astype(np.float64), A[:, C8["ret_conf"]].astype(np.float64)
    xs, xc = A[:, C8["x_sel"]].astype(np.float64), A[:, C8["x_conf"]].astype(np.float64)
    return dict(rs=rs, rc=rc, xs=xs, xc=xc,
                dbh_s=np.expm1(rs) - np.expm1(BHS), dbh_c=np.expm1(rc) - np.expm1(BHC),
                ex_s=rs - xs * BHS, ex_c=rc - xc * BHC,
                dd_s=np.expm1(A[:, C8["mdd_sel"]].astype(np.float64)) - A[:, C8["smdd_sel"]],
                dd_c=np.expm1(A[:, C8["mdd_conf"]].astype(np.float64)) - A[:, C8["smdd_conf"]])


CONF = ["rc", "dbh_c", "ex_c", "dd_c"]


def per_stock_top1(m, key):
    v = np.where(np.isfinite(m[key]), m[key], -np.inf)
    j = np.argmax(v, axis=1)
    ok = np.isfinite(m[key]).any(1)
    S = np.arange(len(j))
    out = {c: np.where(ok, m[c][S, j], np.nan) for c in CONF + ["xc", key]}
    out["j"] = np.where(ok, j, -1)
    return out


SELM = ["rs", "dbh_s", "ex_s", "dd_s"]


def stats_vec(prefix, d, row, sel=False):
    for c in (CONF + SELM) if sel else CONF:
        row[f"{prefix}_{c}_med"] = med(d[c])
        if c not in ("rc", "rs"):
            row[f"{prefix}_{c}_pos"] = pos(d[c])
    row[f"{prefix}_xc_med"] = med(d["xc"])


def top_cells(m, key, n=20):
    cm = np.nanmedian(m[key], axis=0)
    cm = np.where(np.isfinite(cm), cm, -np.inf)
    return np.argsort(-cm, kind="stable")[:n], cm


def cells_stats(m, idx):
    """칸마다 종목 중앙·+비율 → 칸들 중앙."""
    r = {}
    for c in CONF:
        r[c + "_med"] = float(np.median([med(m[c][:, j]) for j in idx]))
        if c != "rc":
            r[c + "_pos"] = float(np.median([pos(m[c][:, j]) for j in idx]))
    return r


def spec_metrics(sp, bh_s, bh_c):
    """sp: (S, 12) 특수칸 한 k → 종목별 지표."""
    rs, rc, xs, xc = sp[:, 0], sp[:, 1], sp[:, 2], sp[:, 3]
    return dict(rs=rs, rc=rc, xs=xs, xc=xc, dbh_s=np.expm1(rs) - np.expm1(bh_s), dbh_c=np.expm1(rc) - np.expm1(bh_c),
                ex_s=rs - xs * bh_s, ex_c=rc - xc * bh_c,
                dd_s=np.expm1(sp[:, 4]) - sp[:, 7], dd_c=np.expm1(sp[:, 5]) - sp[:, 8])


def seg_counts(sym, base, tf, sess):
    tb = tf_bars(load(sym, base), tf, sess)
    tb = tb[tb.day >= WIN[base]["warm"]]
    seg = segments(tb.day.values, WIN[base])
    return int((seg == 0).sum()), int((seg == 1).sum())


def nullq(v0, vn):
    vn = np.asarray(vn, np.float64)
    vn = vn[np.isfinite(vn)]
    if not np.isfinite(v0) or len(vn) == 0:
        return dict(q=None, p=None)
    q = ((vn < v0).sum() + 0.5 * (vn == v0).sum()) / len(vn)
    return dict(q=round(float(q), 3), p=round(float(((vn >= v0).sum() + 1) / (len(vn) + 1)), 3))


def main(base):
    sh = json.loads((RES / f"awake_shifts_{base}.json").read_text(encoding="utf-8"))
    ks = sh["ks"]
    K1 = len(ks)
    cells = pd.read_parquet(RES / f"awake_cells_{base}.parquet")
    M = len(cells)
    syms = sorted(p.stem for p in (OUT / base).glob("*.npy") if not p.stem.endswith("_spec"))
    S = len(syms)
    metas = {s: json.loads((OUT / base / f"{s}_meta.json").read_text(encoding="utf-8")) for s in syms}
    tf_arr = cells.tf.to_numpy()
    BHS = np.full((S, M), np.nan)
    BHC = np.full((S, M), np.nan)
    for i, s in enumerate(syms):
        for tf, v in metas[s]["tf"].items():
            mk = tf_arr == int(tf)
            BHS[i, mk] = v["bh_sel"]
            BHC[i, mk] = v["bh_conf"]
    mm = [np.load(TMP / base / f"{s}.npy", mmap_mode="r") for s in syms]
    spec = np.stack([np.load(OUT / base / f"{s}_spec.npy") for s in syms])  # (S, K1, nspec, 12)
    spn = [x[0] for x in specials(base)]
    rth = {s: metas[s].get("rth60", {}) for s in syms}
    LBS = np.array([rth[s].get("bh_sel", np.nan) for s in syms])
    LBC = np.array([rth[s].get("bh_conf", np.nan) for s in syms])
    tfo = ORIG_TF[base]
    i0 = int(cells.index[(cells.tf == tfo) & (cells.N == 10) & (cells.ratio == "count") & (cells.dir == "follow")
                         & (cells.b == 0.5) & (cells.s == 0.5)][0])

    rows = []
    top1_k = {key: {c: np.full((K1, S), np.nan) for c in CONF + ["xc", "j"]} for key in ("rs", "ex_s")}
    orig_k = {c: np.full((K1, S), np.nan) for c in CONF}
    top20_idx = {}
    pass_cells = None
    sym_top20 = {}
    for kk, k in enumerate(ks):
        A = np.stack([a[kk] for a in mm])  # (S, 8, M)
        m = metrics(A, BHS, BHC)
        row = dict(k=k)
        for key, tag in (("rs", "t1"), ("ex_s", "t1x")):
            d = per_stock_top1(m, key)
            stats_vec(tag, d, row)
            for c in CONF + ["xc", "j"]:
                top1_k[key][c][kk] = d[c]
        for key, tag in (("rs", "t20"), ("ex_s", "t20x")):
            idx, cm = top_cells(m, key)
            for c, v in cells_stats(m, idx).items():
                row[f"{tag}_{c}"] = v
            if kk == 0:
                top20_idx[key] = (idx, cm[idx])
                sym_top20[key] = {c: np.nanmedian(m[c][:, idx], axis=1) for c in CONF}
        d0 = {c: m[c][:, i0] for c in CONF + SELM + ["xc"]}
        stats_vec("orig", d0, row, sel=True)
        oi = spec_metrics(spec[:, kk, spn.index("orig_ideal0")].astype(np.float64), BHS[:, i0], BHC[:, i0])
        stats_vec("origI", oi, row, sel=True)
        for c in CONF:
            orig_k[c][kk] = m[c][:, i0]
        # 단서 칸(정규장 60분 N5 body fade, 깨어 있음 + 0.05%)
        ls = spec_metrics(spec[:, kk, spn.index("lead_awake_c")].astype(np.float64), LBS, LBC)
        stats_vec("lead", ls, row, sel=True)
        stats_vec("leadI", spec_metrics(spec[:, kk, spn.index("lead_ideal0")].astype(np.float64), LBS, LBC), row, sel=True)
        # 남는 칸: 두 구간 모두 종목 PASS 이상에서 +
        shp = {}
        for c in ("ex_s", "ex_c", "dd_s", "dd_c"):
            v = m[c]
            fin = np.isfinite(v)
            shp[c] = np.where(fin.sum(0) > 0, (np.where(fin, v, 0) > 0).sum(0) / np.maximum(fin.sum(0), 1), np.nan)
        pex = (shp["ex_s"] >= PASS) & (shp["ex_c"] >= PASS)
        pdd = (shp["dd_s"] >= PASS) & (shp["dd_c"] >= PASS)
        row["pass_ex"] = int(pex.sum()); row["pass_dd"] = int(pdd.sum()); row["pass_both"] = int((pex & pdd).sum())
        row["cells_exc_med_med"] = float(np.nanmedian(np.nanmedian(m["ex_c"], axis=0)))
        row["cells_ddc_med_med"] = float(np.nanmedian(np.nanmedian(m["dd_c"], axis=0)))
        if kk == 0:
            pass_cells = cells.copy()
            for c in shp:
                pass_cells["pos_" + c] = shp[c]
            for c in ("ex_s", "ex_c", "dd_s", "dd_c", "rc", "rs"):
                pass_cells["med_" + c] = np.nanmedian(m[c], axis=0)
            pass_cells["pass_ex"] = pex; pass_cells["pass_dd"] = pdd
            m0 = m
        for c in ("dd_s", "dd_c", "ex_s", "ex_c"):
            row["maxpos_" + c] = float(np.nanmax(shp[c]))  # 칸 중 가장 높은 '종목 + 비율'
        rows.append(row)
        print(base, "k", k, "t1 ex+", round(row["t1_ex_c_pos"], 2), "orig ex+", round(row["orig_ex_c_pos"], 2),
              "pass_ex", row["pass_ex"], "pass_dd", row["pass_dd"], flush=True)
    N = pd.DataFrame(rows)
    N.to_csv(RES / f"awake_null_{base}.csv", index=False)

    act, nul = N.iloc[0], N.iloc[1:]
    summ = {}
    for c in N.columns:
        if c == "k":
            continue
        summ[c] = dict(actual=float(act[c]), null_med=float(nul[c].median()), null_q95=float(nul[c].quantile(0.95)),
                       **nullq(float(act[c]), nul[c].values))

    # 상위 20칸 표
    tops = []
    for key, (idx, cmv) in top20_idx.items():
        t = cells.iloc[idx].copy()
        t["by"] = key
        t["sel_med"] = cmv
        for c in ("rs", "ex_s", "dd_s") + tuple(CONF):
            t["med_" + c] = [med(m0[c][:, j]) for j in idx]
        for c in ("ex_s", "ex_c", "dd_s", "dd_c", "dbh_c"):
            t["pos_" + c] = [pos(m0[c][:, j]) for j in idx]
        tops.append(t)
    pd.concat(tops).to_csv(RES / f"awake_top20_{base}.csv", index=False)
    pass_cells[pass_cells.pass_ex | pass_cells.pass_dd].sort_values("pos_ex_c", ascending=False).to_csv(
        RES / f"awake_pass_{base}.csv", index=False)

    # 종목별 표 (전체 구간 = 고르기 + 확인, 9/5 제외 — 1차 보고와 같은 회계)
    io = spn.index("orig_awake_c")
    iil = spn.index("orig_ideal0")
    il = spn.index("lead_awake_c")
    srows = []
    for i, s in enumerate(syms):
        mt = metas[s]["tf"].get(str(tfo), {})
        r = dict(sym=s, bh_sel=pct(mt.get("bh_sel")), bh_conf=pct(mt.get("bh_conf")), bh_all=pct(mt.get("bh_all")),
                 bh_mdd_all=pp(mt.get("bh_mdd_all")), bh_mdd_conf=pp(mt.get("bh_mdd_conf")))
        try:
            ns, nc = seg_counts(s, base, tfo, "eth")
        except Exception:
            ns = nc = 0
        for tag, si_ in (("orig", io), ("orig_ideal", iil)):
            v = spec[i, 0, si_].astype(np.float64)
            xa = (v[2] * ns + v[3] * nc) / max(ns + nc, 1)
            bha = mt.get("bh_all", np.nan)
            r.update({f"{tag}_sel": pct(v[0]), f"{tag}_conf": pct(v[1]), f"{tag}_all": pct(v[0] + v[1]),
                      f"{tag}_x_all": round(float(xa), 3), f"{tag}_mdd_all": pp(np.expm1(v[6])),
                      f"{tag}_static_all": pp(xa * np.expm1(bha)), f"{tag}_smdd_all": pp(v[9]),
                      f"{tag}_conf_mdd": pp(np.expm1(v[5])), f"{tag}_conf_smdd": pp(v[8]),
                      f"{tag}_conf_ex": pct(v[1] - v[3] * mt.get("bh_conf", np.nan))})
        # 원안 종목별 귀무 분위(같은 종목, 같은 칸, 이동만 다름)
        allr = spec[i, :, io, 0].astype(np.float64) + spec[i, :, io, 1]
        r["orig_all_q"] = nullq(allr[0], allr[1:])["q"]
        r["orig_conf_ex_q"] = nullq(orig_k["ex_c"][0, i], orig_k["ex_c"][1:, i])["q"]
        r["orig_conf_dd_q"] = nullq(orig_k["dd_c"][0, i], orig_k["dd_c"][1:, i])["q"]
        dda = np.expm1(spec[i, :, io, 6].astype(np.float64)) - spec[i, :, io, 9]
        r["orig_all_dd"] = pp(dda[0]); r["orig_all_dd_q"] = nullq(dda[0], dda[1:])["q"]
        j = int(top1_k["rs"]["j"][0, i])
        if j >= 0:
            c = cells.iloc[j]
            r["top1_cell"] = f"{c.tf}m N{c.N} {c.ratio} {c.dir} {c.b}/{c.s}"
            r["top1_sel"] = pct(m0["rs"][i, j]); r["top1_conf"] = pct(m0["rc"][i, j])
            r["top1_conf_dbh"] = pp(m0["dbh_c"][i, j]); r["top1_conf_ex"] = pct(m0["ex_c"][i, j])
            r["top1_conf_dd"] = pp(m0["dd_c"][i, j]); r["top1_xc"] = round(float(m0["xc"][i, j]), 3)
            r["top1_conf_ex_q"] = nullq(top1_k["rs"]["ex_c"][0, i], top1_k["rs"]["ex_c"][1:, i])["q"]
        for c in CONF:
            r[f"top20_{c}"] = pct(sym_top20["rs"][c][i]) if c in ("rc", "ex_c") else pp(sym_top20["rs"][c][i])
        lv = spec[i, 0, il].astype(np.float64)
        r.update(lead_sel=pct(lv[0]), lead_conf=pct(lv[1]), lead_all=pct(lv[0] + lv[1]),
                 lead_conf_ex=pct(lv[1] - lv[3] * LBC[i]), lead_conf_dd=pp(np.expm1(lv[5]) - lv[8]),
                 lead_mdd_all=pp(np.expm1(lv[6])), lead_smdd_all=pp(lv[9]), lead_bh_conf=pct(LBC[i]))
        srows.append(r)
    SD = pd.DataFrame(srows)
    SD.to_csv(RES / f"awake_stocks_{base}.csv", index=False)

    # 전체 구간(원안) 종목 중앙
    full = dict(orig_all_med=med(SD.orig_all), static_all_med=med(SD.orig_static_all), bh_all_med=med(SD.bh_all),
                orig_mdd_all_med=med(SD.orig_mdd_all), orig_smdd_all_med=med(SD.orig_smdd_all),
                bh_mdd_all_med=med(SD.bh_mdd_all),
                orig_beats_static_all=pos(SD.orig_all - SD.orig_static_all),
                orig_dd_all_pos=pos(SD.orig_mdd_all - SD.orig_smdd_all),
                orig_all_q_med=med(SD.orig_all_q), orig_all_q_ge95=float((SD.orig_all_q >= 0.95).mean()))
    # 원안 전체 구간 낙폭 축소의 공통 k 귀무
    ddk = np.expm1(spec[:, :, io, 6].astype(np.float64)) - spec[:, :, io, 9]  # (S, K1)
    ddpos = np.array([pos(ddk[:, kk]) for kk in range(K1)])
    full["orig_dd_all_pos_null"] = dict(actual=float(ddpos[0]), null_med=float(np.median(ddpos[1:])), **nullq(ddpos[0], ddpos[1:]))
    exa = np.array([med(spec[:, kk, io, 0].astype(np.float64) + spec[:, kk, io, 1]) for kk in range(K1)])
    full["orig_all_med_null"] = dict(actual=float(exa[0]), null_med=float(np.median(exa[1:])), **nullq(exa[0], exa[1:]))

    out = dict(base=base, n_syms=S, n_cells=M, D=sh["D"], n_null=K1 - 1, null_ks=f"{ks[1]}~{ks[-1]}",
               grid=dict(sess="eth", tf=sorted(cells.tf.unique().tolist()), N=sorted(cells.N.unique().tolist()),
                         ratio=sorted(cells.ratio.unique().tolist()), dir=["follow", "fade"], thresholds=16, overnight="hold"),
               stats=summ, full=full,
               focus=SD[SD.sym.isin(FOCUS)].set_index("sym").to_dict(orient="index"),
               top20=[f"{r.tf}m N{r.N} {r.ratio} {r.dir} {r.b}/{r.s}" for r in cells.iloc[top20_idx["rs"][0]].itertuples()],
               top20x=[f"{r.tf}m N{r.N} {r.ratio} {r.dir} {r.b}/{r.s}" for r in cells.iloc[top20_idx["ex_s"][0]].itertuples()])
    (RES / f"awake_summary_{base}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = main(sys.argv[1])
    print(json.dumps({k: o[k] for k in ("base", "n_syms", "n_cells", "D", "n_null", "null_ks", "full")}, ensure_ascii=False, indent=1, default=float))
