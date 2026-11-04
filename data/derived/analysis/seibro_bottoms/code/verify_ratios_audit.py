"""ratios.py 반박 감사(BRIEF2.md 대조). ratios.py는 고치지 않고 함수만 가져와 다시 돈다.

실행(연구 폴더에서): py -B code/verify_ratios_audit.py
출력: results/ratios/verify_audit.md, results/ratios/verify_audit.json
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import ratios as R  # noqa: E402

OUT = R.OUT
WINS = R.WINS
RES: dict = {}


def log(k, v):
    RES[k] = v
    print(k, v if not isinstance(v, (dict, list)) or len(str(v)) < 300 else str(v)[:300] + "...", flush=True)


# ---------- 자료 ----------
dates, prods, pg, Bm, Sm, G = R.load()
pf = pd.read_parquet(R.DER / "price_feats.parquet")
pf["date"] = pd.to_datetime(pf["date"])
pf = pf.set_index("date").reindex(dates).reset_index().rename(columns={"index": "date"})
ev = json.loads((R.DER / "events.json").read_text(encoding="utf-8"))
S = R.event_sets(ev, dates, pf)
fit_mask = dates >= R.START
N = len(dates)

feat, Pm, Em, Cm = {}, {}, {}, {}
for L in (1, 5, 15):
    F = R.features(Bm, Sm, G, prods, L)
    feat[L] = F
    P, C = R.rolling_pct(F.values)
    Pm[L], Cm[L] = P, C
    Em[L] = R.echo_resid(P, C, R.design(pf, L), fit_mask)
names = list(feat[1].columns)
BANK = {"P": Pm, "E": Em}


def vals(bank, norm, w, idx):
    L, off = WINS[w]
    return R.placed(bank[norm][L], idx, off)


# ---------- 1. 창 위치·사건 거르기 ----------
pos = {d: i for i, d in enumerate(dates)}
b = pos[pd.Timestamp("2024-08-07")]
k = prods.index("SOXL")
man = {}
for w, (lo, hi) in {"W1": (-20, -6), "W2": (-5, -1), "W3": (0, 0), "W4": (1, 5)}.items():
    rows = range(b + lo, b + hi + 1)
    bs, ss = Bm[list(rows), k].sum(), Sm[list(rows), k].sum()
    L, off = WINS[w]
    got = feat[L]["A:imb:SOXL"].values[b + off]
    man[w] = round(abs((bs - ss) / (bs + ss) - got), 12)
log("window_manual_absdiff_SOXLimb_2024-08-07", man)
f_all = [x["date"] for x in ev["F"]]
log("events_counts", {k2: int(len(S[k2])) for k2 in ("B10", "B15", "F", "D", "B10_2425", "B10_26", "F_2425", "F_26", "D_2425", "D_26")})
log("F_dropped_pre2024_or_noflow", int(len(f_all) - len(S["F"])))
log("F_undetermined", ev.get("F_undetermined"))
# F 날이 바닥 ±5세션 안
nearF = {dates[i].date().isoformat(): int(min(abs(i - S["B10"]))) for i in S["F"] if min(abs(i - S["B10"])) <= 5}
log("F_within5_of_B10", nearF)
# 창 끝이 자료 밖(W4)인 사건
tail = {k2: int(((S[k2] + 5) >= N).sum()) for k2 in ("B10", "F", "D")}
log("W4_out_of_range_counts", tail)
log("last_flow_date", dates[-1].date().isoformat())

# ---------- 2. 결측 ----------
nan_rows = []
for norm in ("P", "E"):
    for w in WINS:
        for key in ("B10", "F", "D", "F_26", "F_2425"):
            v = vals(BANK, norm, w, S[key])
            nan_rows.append({"norm": norm, "w": w, "set": key, "mean_ok": float(np.isfinite(v).mean()),
                             "cells_all_nan": int((~np.isfinite(v)).all(0).sum())})
nan_df = pd.DataFrame(nan_rows)
log("cells_all_nan_by_set", nan_df.groupby("set")["cells_all_nan"].sum().to_dict())
# 원 특징 자체가 전 구간 결측인 열
log("features_all_nan_2024plus", {L: int((~np.isfinite(feat[L].values[fit_mask])).all(0).sum()) for L in feat})
log("E_unfitted_features", {L: int((~np.isfinite(Em[L][fit_mask])).all(0).sum()) for L in Em})

# ---------- 3. Eq 잔차 ----------
eq = {}
for L in (1, 5, 15):
    Z = Em[L][fit_mask]
    X = R.design(pf, L)[fit_mask]
    okc = np.isfinite(Z).sum(0) > 120
    mu = np.nanmean(Z[:, okc], 0)
    sd = np.nanstd(Z[:, okc], 0)
    cors = []
    for j in np.flatnonzero(okc)[::7]:
        m = np.isfinite(Z[:, j]) & np.isfinite(X).all(1)
        for c in (1, 2, 4):
            cors.append(abs(np.corrcoef(Z[m, j], X[m, c])[0, 1]))
    rk = np.linalg.matrix_rank(X[np.isfinite(X).all(1)])
    eq[L] = {"n_fit": int(np.isfinite(X).all(1).sum()), "rank": int(rk), "ncol": int(X.shape[1]),
             "mean_med": round(float(np.median(mu)), 4), "sd_med": round(float(np.median(sd)), 3),
             "max_abs_corr_design": round(float(np.nanmax(cors)), 4)}
log("Eq_checks", eq)

# ---------- 4~5. 칸 통계(변형 가능) ----------


def common_pass2(vB, norm, k, zmode="signed"):
    hi, lo, ok = R.sides(vB, norm)
    nh, nl = hi.sum(0), lo.sum(0)
    side = np.where(nh >= nl, 1, -1)
    cnt = np.where(side == 1, nh, nl)
    passed = cnt >= k
    if norm == "E" and zmode != "none":
        if zmode == "signed":
            med = np.nanmedian(vB, axis=0)
            passed &= (np.abs(med) >= 0.5) & (np.sign(med) == side)
        elif zmode == "abs":
            passed &= np.nanmedian(np.abs(vB), axis=0) >= 0.5
    return passed, side, cnt, ok.sum(0)


def rate_n(vX, side, norm):
    hi, lo, ok = R.sides(vX, norm)
    on = np.where(side == 1, hi.sum(0), lo.sum(0))
    n = ok.sum(0)
    return on / np.maximum(n, 1), n


def excl_near(base, pick, gap=5):
    if len(base) == 0:
        return base
    d = np.abs(base[:, None] - np.asarray(pick)[None, :]).min(1)
    return base[d > gap]


def real_cells(bank, nm, zmode="signed", sel_zmode=None, fexcl=False):
    sel_zmode = zmode if sel_zmode is None else sel_zmode
    Fs, Ds = S["F"], S["D"]
    F24, D24, F26 = S["F_2425"], S["D_2425"], S["F_26"]
    if fexcl:
        Fs, F24, F26 = excl_near(Fs, S["B10"]), excl_near(F24, S["B10"]), excl_near(F26, S["B10"])
    rows = []
    for norm in ("P", "E"):
        for w in WINS:
            vB = vals(bank, norm, w, S["B10"])
            p1, side, cnt, nok = common_pass2(vB, norm, 8, zmode)
            fr, fn = rate_n(vals(bank, norm, w, Fs), side, norm)
            dr, dn = rate_n(vals(bank, norm, w, Ds), side, norm)
            p3, s3, c3, _ = common_pass2(vals(bank, norm, w, S["B10_2425"]), norm, 5, sel_zmode)
            f24, f24n = rate_n(vals(bank, norm, w, F24), s3, norm)
            d24, _ = rate_n(vals(bank, norm, w, D24), s3, norm)
            hi, lo, _ = R.sides(vals(bank, norm, w, S["B10_26"]), norm)
            c26 = np.where(s3 == 1, hi.sum(0), lo.sum(0))
            f26, f26n = rate_n(vals(bank, norm, w, F26), s3, norm)
            rows.append(pd.DataFrame({"feature": nm, "window": w, "norm": norm, "common": p1, "side": side, "B10_on": cnt,
                                      "F_rate": fr, "F_n": fn, "D_rate": dr, "D_n": dn, "sel_side": s3, "B2425_on": c3,
                                      "F2425_rate": f24, "F2425_n": f24n, "D2425_rate": d24,
                                      "selected": p3 & (f24 <= 1 / 3) & (d24 <= 1 / 3), "B26_on": c26, "F26_rate": f26, "F26_n": f26n}))
    c = pd.concat(rows, ignore_index=True)
    c["distinct"] = c["common"] & (c["F_rate"] <= 1 / 3) & (c["D_rate"] <= 1 / 3)
    c["checked"] = c["selected"] & (c["B26_on"] >= 3) & (c["F26_rate"] <= 1 / 3)
    return c


def summ(c):
    out = {}
    for n in ("P", "E"):
        cc = c[c.norm == n]
        out[n] = {"common": int(cc.common.sum()), "distinct": int(cc.distinct.sum()),
                  "selected": int(cc.selected.sum()), "checked": int(cc.checked.sum())}
    return out


base = real_cells(BANK, names)
orig = pd.read_csv(OUT / "cells.csv")
mrg = orig.merge(base, on=["feature", "window", "norm"], suffixes=("_o", "_n"))
log("reproduce_match", {"common": bool((mrg.common_o == mrg.common_n).all()), "selected": bool((mrg.selected_o == mrg.selected_n).all()),
                        "F_rate_maxdiff": float(np.nanmax(np.abs(mrg.F_rate_o - mrg.F_rate_n)))})
log("base_summary", summ(base))
# 분모 0 → 비율 0 문제
log("zero_denominator_cells", {"F_n==0": int((base.F_n == 0).sum()), "F2425_n==0": int((base.F2425_n == 0).sum()),
                               "F26_n==0": int((base.F26_n == 0).sum()),
                               "selected_with_F26_n<5": int((base.selected & (base.F26_n < 5)).sum()),
                               "common_with_F_n<12": int((base.common & (base.F_n < 12)).sum())})
# W4 F26 분모
log("F26_n_by_window_P", base[base.norm == "P"].groupby("window")["F26_n"].median().to_dict())

# 변형: 중앙 |z|, z 문턱 없음, 고르기 문턱 없음, F ±5 제외
var = {"abs_median_z": real_cells(BANK, names, "abs"), "no_zmin": real_cells(BANK, names, "none"),
       "sel_without_zmin": real_cells(BANK, names, "signed", "none"), "F_excl_pm5_of_B10": real_cells(BANK, names, fexcl=True)}
log("variant_summaries", {k2: summ(v) for k2, v in var.items()})
for k2, v in var.items():
    ch = v[v.checked]
    if len(ch):
        log(f"variant_checked_cells_{k2}", ch[["feature", "window", "norm", "B2425_on", "F2425_rate", "D2425_rate", "B26_on", "F26_rate", "F26_n"]].to_dict("records"))
    dd = v[v.distinct]
    if len(dd):
        log(f"variant_distinct_cells_{k2}", dd[["feature", "window", "norm", "B10_on", "F_rate", "D_rate"]].head(10).to_dict("records"))

# ---------- 6. 우연 기대(공통) 다시 ----------


def null_common(bank, pool, reps, seed, zmode="signed", excl=False, rest_from=("F", "D")):
    R.RNG = np.random.default_rng(seed)
    recs = []
    for rep in range(reps):
        pick = R.spaced_draw(pool, len(S["B10"]))
        if pick is None:
            continue
        if excl:
            rF, rD = excl_near(S["F"], pick), excl_near(S["D"], pick)
        else:
            rF, rD = np.setdiff1d(S["F"], pick), np.setdiff1d(S["D"], pick)
        rec = {}
        for norm in ("P", "E"):
            tot = dist = 0
            for w in WINS:
                p1, side, _, _ = common_pass2(vals(bank, norm, w, pick), norm, 8, zmode)
                fr, _ = rate_n(vals(bank, norm, w, rF), side, norm)
                dr, _ = rate_n(vals(bank, norm, w, rD), side, norm)
                tot += int(p1.sum())
                dist += int((p1 & (fr <= 1 / 3) & (dr <= 1 / 3)).sum())
            rec[f"{norm}_common"], rec[f"{norm}_distinct"] = tot, dist
        recs.append(rec)
    return pd.DataFrame(recs)


def nsum(nl, real):
    o = {"reps": int(len(nl))}
    for n in ("P", "E"):
        o[n] = {"real": real[n]["common"], "null_mean": round(float(nl[f"{n}_common"].mean()), 1),
                "null_p95": round(float(nl[f"{n}_common"].quantile(0.95)), 1),
                "p_ge": round(float((nl[f"{n}_common"] >= real[n]["common"]).mean()), 3),
                "real_distinct": real[n]["distinct"], "null_distinct_mean": round(float(nl[f"{n}_distinct"].mean()), 3),
                "null_distinct_p_ge": round(float((nl[f"{n}_distinct"] >= real[n]["distinct"]).mean()), 3)}
    return o


pool = np.unique(np.concatenate([S["B10"], S["F"], S["D"]]))
sb = summ(base)
nulls = {}
for seed in (20261104, 7, 99991):
    nl = null_common(BANK, pool, 1000, seed)
    nulls[f"seed{seed}_1000"] = nsum(nl, sb)
nulls["abs_median_z_seed7_1000"] = nsum(null_common(BANK, pool, 1000, 7, "abs"), summ(var["abs_median_z"]))
nulls["excl_pm5_seed7_1000"] = nsum(null_common(BANK, pool, 1000, 7, excl=True), summ(var["F_excl_pm5_of_B10"]))
# 참고: 사건 아닌 평범한 날 무작위(2024 이후, 창 안쪽)
allpool = np.flatnonzero(fit_mask)
allpool = allpool[(allpool >= 20) & (allpool < N - 5)]
nulls["uniform_days_seed7_300"] = nsum(null_common(BANK, allpool, 300, 7), sb)
log("nulls", nulls)

# ---------- 7. 확인 우연 기대·조합 ----------


def check_null(sel, bank, reps, seed):
    R.RNG = np.random.default_rng(seed)
    pool26 = np.unique(np.concatenate([S["B10_26"], S["F_26"], S["D_26"]]))
    hits = np.zeros(len(sel))
    nrep = 0
    for rep in range(reps):
        pick = R.spaced_draw(pool26, len(S["B10_26"]))
        if pick is None:
            continue
        nrep += 1
        restF = np.setdiff1d(S["F_26"], pick)
        for i, r in enumerate(sel.itertuples()):
            j = names.index(r.feature) if r.feature in names else None
            v = vals(bank, r.norm, r.window, pick)[:, [j]]
            hi, lo, _ = R.sides(v, r.norm)
            on = (hi if r.sel_side == 1 else lo).sum()
            fr, _ = rate_n(vals(bank, r.norm, r.window, restF)[:, [j]], np.array([r.sel_side]), r.norm)
            hits[i] += float(on >= 3 and fr[0] <= 1 / 3)
    return hits / max(nrep, 1), nrep


sel = base[base.selected].reset_index(drop=True)
pn, nrep = check_null(sel, BANK, 1000, 7)
log("check_null_1000_seed7", {"selected": int(len(sel)), "expected_pass": round(float(pn.sum()), 4), "actual": int(sel.checked.sum()), "reps": nrep})
sel2 = var["sel_without_zmin"]
sel2 = sel2[sel2.selected].reset_index(drop=True)
pn2, _ = check_null(sel2, BANK, 1000, 7)
log("check_null_sel_without_zmin", {"selected": int(len(sel2)), "expected_pass": round(float(pn2.sum()), 4), "actual": int(sel2.checked.sum())})


def onmask(bank, r, idx):
    L, off = WINS[r["window"]]
    v = R.placed(bank[r["norm"]][L], idx, off)[:, names.index(r["feature"])]
    hi, lo, ok = R.sides(v[:, None], r["norm"])
    return np.where(r["sel_side"] == 1, hi[:, 0], lo[:, 0]), ok[:, 0]


def combos_of(top, bank, sets=None):
    sets = sets or {k2: S[k2] for k2 in ("B10_2425", "F_2425", "D_2425", "B10_26", "F_26")}
    out = []
    for (_, r1), (_, r2) in itertools.combinations(top.iterrows(), 2):
        rec = {"c1": f"{r1['feature']}·{r1['window']}·{r1['norm']}", "c2": f"{r2['feature']}·{r2['window']}·{r2['norm']}"}
        for key, idx in sets.items():
            m1, o1 = onmask(bank, r1, idx)
            m2, o2 = onmask(bank, r2, idx)
            both = m1 & m2
            den = (o1 & o2).sum()
            rec[key] = float(both.sum()) if key.startswith("B10") else (float(both.sum() / den) if den else np.nan)
            if not key.startswith("B10"):
                rec[key + "_n"] = int(den)
        rec["selected"] = rec["B10_2425"] >= 5 and rec["F_2425"] <= 1 / 3 and rec["D_2425"] <= 1 / 3
        rec["checked"] = bool(rec["selected"] and rec["B10_26"] >= 3 and rec["F_26"] <= 1 / 3)
        out.append(rec)
    return pd.DataFrame(out)


# 넓힌 조합: 고르기의 F·D 조건 없이 2024~25 바닥 5/6 이상인 칸 상위 20개(바닥 몫 − F 동반율)
cand = base[base.B2425_on >= 5].copy()
cand["score"] = cand["B2425_on"] / 6 - cand["F2425_rate"]
cand = cand.sort_values(["score", "D2425_rate"], ascending=[False, True])
wide = {}
for topn in (20, 40):
    t = cand.head(topn)
    cb = combos_of(t, BANK)
    wide[topn] = {"cands_total": int(len(cand)), "combos": int(len(cb)), "selected": int(cb.selected.sum()), "checked": int(cb.checked.sum()),
                  "checked_rows": cb[cb.checked][["c1", "c2", "B10_2425", "F_2425", "D_2425", "B10_26", "F_26", "F_26_n"]].head(8).to_dict("records")}
    if topn == 40:
        wide_cb, wide_top = cb, t
log("wide_combos", wide)

# 넓힌 조합의 확인 우연 기대: 고른 조합마다 2026 가짜 바닥 4개
wsel = wide_cb[wide_cb.selected].reset_index(drop=True)
if len(wsel):
    R.RNG = np.random.default_rng(7)
    pool26 = np.unique(np.concatenate([S["B10_26"], S["F_26"], S["D_26"]]))
    tmap = {f"{r['feature']}·{r['window']}·{r['norm']}": r for _, r in wide_top.iterrows()}
    hits = np.zeros(len(wsel))
    nr = 0
    for rep in range(500):
        pick = R.spaced_draw(pool26, 4)
        if pick is None:
            continue
        nr += 1
        restF = np.setdiff1d(S["F_26"], pick)
        for i, rr in wsel.iterrows():
            m1, _ = onmask(BANK, tmap[rr.c1], pick)
            m2, _ = onmask(BANK, tmap[rr.c2], pick)
            f1, o1 = onmask(BANK, tmap[rr.c1], restF)
            f2, o2 = onmask(BANK, tmap[rr.c2], restF)
            den = (o1 & o2).sum()
            fr = (f1 & f2).sum() / den if den else np.nan
            hits[i] += float((m1 & m2).sum() >= 3 and fr <= 1 / 3)
    log("wide_combos40_check_null", {"selected": int(len(wsel)), "expected_pass": round(float(hits.sum() / nr), 3), "actual": int(wsel.checked.sum())})

# ---------- 8. 직관 표: 해 섞임 ----------
yrs = dates.year.values
allidx = np.flatnonzero(fit_mask)


def raw(name, w, idx):
    L, off = WINS[w]
    return R.placed(feat[L][[name]].values, idx, off)[:, 0]


intu_rows = []
for name in ("A:imb:SOXL", "D:fund", "D:breadth", "D:bull", "B:nshare:SOXL", "C:lbr:SOXL|TQQQ", "A:imb:STK1", "B:sshare:SOXL"):
    for w in ("W2", "W3"):
        allv = raw(name, w, allidx)
        ymed = {y: np.nanmedian(allv[yrs[allidx] == y]) for y in (2024, 2025, 2026)}
        rec = {"feature": name, "w": w, **{f"평소{y}": round(float(ymed[y]), 3) for y in ymed}}
        for key in ("B10", "F", "D"):
            v = raw(name, w, S[key])
            adj = v - np.array([ymed[y] for y in yrs[S[key]]])
            rec[f"{key}_raw"] = round(float(np.nanmedian(v)), 3)
            rec[f"{key}_yadj"] = round(float(np.nanmedian(adj)), 3)
        intu_rows.append(rec)
intu = pd.DataFrame(intu_rows)
log("event_year_mix", {k2: pd.Series(yrs[S[k2]]).value_counts().sort_index().to_dict() for k2 in ("B10", "F", "D")})
# 짝수 표본: 변환 뒤 중앙 vs 중앙 뒤 변환
v = raw("A:imb:SOXL", "W3", S["B10"])
v = v[np.isfinite(v)]
log("median_transform_check_SOXLimb_W3_B10", {"f(median)": round(float((1 + np.median(v)) / (1 - np.median(v))), 4),
                                               "median(f)": round(float(np.median((1 + v) / (1 - v))), 4)})
ref = pd.read_csv(OUT / "intuitive.csv", encoding="utf-8-sig")

# ---------- 9. 빠진 족: 매도 상대 비율·순매수 ÷ 전체 거래·창 간 변화 ----------


def extra_feats(L):
    Bp, Sp = R.winsum(Bm, L), R.winsum(Sm, L)
    Bg, Sg = Bp @ G, Sp @ G
    Ng, Tg = Bg - Sg, Bg + Sg
    Tall = Bp.sum(1) + Sp.sum(1)
    cols = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for (a, g), (b2, h) in itertools.combinations(enumerate(R.GROUPS), 2):
            cols[f"C:lsr:{g}|{h}"] = np.where((Sg[:, a] > 0) & (Sg[:, b2] > 0), np.log(Sg[:, a] / Sg[:, b2]), np.nan)
        for j, g in enumerate(R.GROUPS):
            cols[f"B:ntall:{g}"] = Ng[:, j] / Tall
    return pd.DataFrame(cols)


xfeat, XP, XE = {}, {}, {}
for L in (1, 5, 15):
    xf = extra_feats(L)
    xfeat[L] = xf
    P, C = R.rolling_pct(xf.values)
    XP[L] = P
    XE[L] = R.echo_resid(P, C, R.design(pf, L), fit_mask)
xnames = list(xfeat[1].columns)
XB = {"P": XP, "E": XE}
xc = real_cells(XB, xnames)
xn = null_common(XB, pool, 300, 7)
log("extra_family_lsr_ntall", {"features": len(xnames), "cells": int(len(xc)), "summary": summ(xc), "null": nsum(xn, summ(xc))})

# 창 간 변화(W2 − W1, W4 − W2): imb 16개 + D:imball·D:bull·D:levnet
dcols = [f"A:imb:{g}" for g in R.GROUPS] + ["D:imball", "D:bull", "D:levnet"]
f5, f15 = feat[5][dcols].values, feat[15][dcols].values
d_pre = np.full_like(f5, np.nan)
d_pre[5:] = f5[5:] - f15[:-5]          # 행 t: [t−4, t] 5일 − [t−19, t−5] 15일 → 바닥 −1에 두면 W2 − W1
d_turn = np.full_like(f5, np.nan)
d_turn[5:] = f5[5:] - f5[:-5]          # 행 t: [t−4, t] − [t−9, t−5] → 바닥 +5에 두면 W4 − W2
dP, dE = {}, {}
for key, M, off in (("pre", d_pre, -1), ("turn", d_turn, 5)):
    P, C = R.rolling_pct(M)
    dP[key] = (P, off)
    dE[key] = (R.echo_resid(P, C, R.design(pf, 5), fit_mask), off)
drows = []
for norm, src in (("P", dP), ("E", dE)):
    for key, (M, off) in src.items():
        vB = R.placed(M, S["B10"], off)
        p1, side, cnt, _ = common_pass2(vB, norm, 8)
        fr, _ = rate_n(R.placed(M, S["F"], off), side, norm)
        dr, _ = rate_n(R.placed(M, S["D"], off), side, norm)
        for j, n in enumerate(dcols):
            drows.append({"feature": f"Δ{key}:{n}", "norm": norm, "B10_on": int(cnt[j]), "side": int(side[j]), "common": bool(p1[j]),
                          "F_rate": round(float(fr[j]), 3), "D_rate": round(float(dr[j]), 3)})
dd = pd.DataFrame(drows)
dd["distinct"] = dd.common & (dd.F_rate <= 1 / 3) & (dd.D_rate <= 1 / 3)
log("extra_family_delta", {"cells": int(len(dd)), "common": int(dd.common.sum()), "distinct": int(dd.distinct.sum()),
                           "common_rows": dd[dd.common].to_dict("records")[:8]})

# ---------- 상위 일관 칸 재확인 ----------
top5 = [("C:lbr:SOXS|TQQQ", "W3", "P"), ("C:lbr:SOXL|SOXS", "W3", "P"), ("B:nshare:SOXL", "W3", "P"),
        ("B:sshare:SOXL", "W4", "P"), ("C:nrel:SOXL|STK1", "W2", "P")]
chk = []
for f_, w, n in top5:
    r = base[(base.feature == f_) & (base.window == w) & (base.norm == n)].iloc[0]
    e = base[(base.feature == f_) & (base.window == w) & (base.norm == "E")].iloc[0]
    chk.append({"cell": f"{f_}·{w}", "B10_on": int(r.B10_on), "F": round(r.F_rate, 2), "D": round(r.D_rate, 2), "F_n": int(r.F_n),
                "E_B10_on": int(e.B10_on), "E_common": bool(e.common), "E_F": round(e.F_rate, 2), "E_D": round(e.D_rate, 2)})
log("top5_P_vs_E", chk)

# E 공통 칸의 F·D 동반율 분포
ec = base[(base.norm == "E") & base.common]
log("E_common_FD_rates", {"n": int(len(ec)), "F_med": round(float(ec.F_rate.median()), 3), "D_med": round(float(ec.D_rate.median()), 3),
                          "min_max_FD": round(float(np.minimum.reduce([ec.F_rate.values, ec.D_rate.values]).min()) if len(ec) else np.nan, 3),
                          "F<=1/3": int((ec.F_rate <= 1 / 3).sum()), "D<=1/3": int((ec.D_rate <= 1 / 3).sum()),
                          "closest": ec.assign(mx=np.maximum(ec.F_rate, ec.D_rate)).nsmallest(5, "mx")[["feature", "window", "B10_on", "F_rate", "D_rate"]].round(3).to_dict("records")})
pc = base[(base.norm == "P") & base.common]
log("P_common_closest", pc.assign(mx=np.maximum(pc.F_rate, pc.D_rate)).nsmallest(5, "mx")[["feature", "window", "B10_on", "F_rate", "D_rate"]].round(3).to_dict("records"))

# ---------- 10. 보탠 점검: Δ족 우연 기대·2026 확인, 넓힌 조합 20개 우연 기대, 체제 더미 뺀 Eq ----------
DB = {"P": dP, "E": dE}


def delta_count(pick, rF, rD, k=8):
    dist = com = 0
    for norm in ("P", "E"):
        for key, (M, off) in DB[norm].items():
            p1, side, _, _ = common_pass2(R.placed(M, pick, off), norm, k)
            fr, _ = rate_n(R.placed(M, rF, off), side, norm)
            dr, _ = rate_n(R.placed(M, rD, off), side, norm)
            com += int(p1.sum())
            dist += int((p1 & (fr <= 1 / 3) & (dr <= 1 / 3)).sum())
    return com, dist


R.RNG = np.random.default_rng(7)
dn = []
for rep in range(1000):
    pick = R.spaced_draw(pool, 10)
    if pick is None:
        continue
    dn.append(delta_count(pick, np.setdiff1d(S["F"], pick), np.setdiff1d(S["D"], pick)))
dn = np.array(dn)
dsel = []
for norm in ("P", "E"):
    for key, (M, off) in DB[norm].items():
        p3, s3, c3, _ = common_pass2(R.placed(M, S["B10_2425"], off), norm, 5)
        f24, _ = rate_n(R.placed(M, S["F_2425"], off), s3, norm)
        d24, _ = rate_n(R.placed(M, S["D_2425"], off), s3, norm)
        hi, lo, _ = R.sides(R.placed(M, S["B10_26"], off), norm)
        c26 = np.where(s3 == 1, hi.sum(0), lo.sum(0))
        f26, _ = rate_n(R.placed(M, S["F_26"], off), s3, norm)
        sel_ = p3 & (f24 <= 1 / 3) & (d24 <= 1 / 3)
        for j in np.flatnonzero(sel_):
            dsel.append({"feature": f"Δ{key}:{dcols[j]}", "norm": norm, "B2425_on": int(c3[j]), "F2425": round(float(f24[j]), 3),
                         "D2425": round(float(d24[j]), 3), "B26_on": int(c26[j]), "F26": round(float(f26[j]), 3),
                         "checked": bool(c26[j] >= 3 and f26[j] <= 1 / 3)})
log("delta_null_1000", {"real_common": int(dd.common.sum()), "null_common_mean": round(float(dn[:, 0].mean()), 2),
                        "p_common_ge": round(float((dn[:, 0] >= dd.common.sum()).mean()), 3),
                        "real_distinct": int(dd.distinct.sum()), "null_distinct_mean": round(float(dn[:, 1].mean()), 3),
                        "p_distinct_ge": round(float((dn[:, 1] >= dd.distinct.sum()).mean()), 3)})
log("delta_select_check", dsel)
for nm_ in ("D:bull", "D:levnet"):
    j = dcols.index(nm_)
    M, off = dP["turn"]
    vb = R.placed(M, S["B10"], off)[:, j]
    vf = R.placed(M, S["F_26"], off)[:, j]
    log(f"delta_turn_{nm_}_P_by_event", {"B10": [None if not np.isfinite(x) else round(float(x), 1) for x in vb],
                                         "F26_low20": int((vf <= 20).sum()), "F26_n": int(np.isfinite(vf).sum())})

cb20 = combos_of(cand.head(20), BANK)
w20 = cb20[cb20.selected].reset_index(drop=True)
tmap20 = {f"{r['feature']}·{r['window']}·{r['norm']}": r for _, r in cand.head(20).iterrows()}
R.RNG = np.random.default_rng(11)
pool26b = np.unique(np.concatenate([S["B10_26"], S["F_26"], S["D_26"]]))
h20 = np.zeros(len(w20))
nr = 0
for rep in range(500):
    pick = R.spaced_draw(pool26b, 4)
    if pick is None:
        continue
    nr += 1
    restF = np.setdiff1d(S["F_26"], pick)
    for i, rr in w20.iterrows():
        m1, _ = onmask(BANK, tmap20[rr.c1], pick)
        m2, _ = onmask(BANK, tmap20[rr.c2], pick)
        f1, o1 = onmask(BANK, tmap20[rr.c1], restF)
        f2, o2 = onmask(BANK, tmap20[rr.c2], restF)
        den = (o1 & o2).sum()
        h20[i] += float((m1 & m2).sum() >= 3 and den > 0 and (f1 & f2).sum() / den <= 1 / 3)
log("wide_combos20_check_null", {"selected": int(len(w20)), "expected_pass": round(float(h20.sum() / nr), 3), "actual": int(w20.checked.sum())})

Em2 = {L: R.echo_resid(Pm[L], Cm[L], R.design(pf, L)[:, :7], fit_mask) for L in (1, 5, 15)}
B2 = {"P": Pm, "E": Em2}
c2 = real_cells(B2, names)
n2 = null_common(B2, pool, 300, 7)
log("Eq_no_regime", {"summary": summ(c2)["E"], "null": nsum(n2, summ(c2))["E"]})

intu.to_csv(OUT / "verify_audit_intuitive_year.csv", index=False, encoding="utf-8-sig")
(OUT / "verify_audit.json").write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print("INTU")
print(intu.to_string())

# ---------- 11. Δturn 반등 메아리 점검 ----------
ret5 = pf["ret_5"].values
echo = {}
for j_name in ("D:bull", "D:levnet"):
    j = dcols.index(j_name)
    Mp, offp = dP["turn"]
    Me, _ = dE["turn"]
    m = fit_mask & np.isfinite(Mp[:, j]) & np.isfinite(ret5)
    echo[j_name] = {"corr_P_vs_W4ret5": round(float(np.corrcoef(Mp[m, j], ret5[m])[0, 1]), 3),
                    "E_rate_le0": {k2: round(float(np.nanmean(R.placed(Me, S[k2], offp)[:, j] <= 0)), 2) for k2 in ("B10", "F", "D")}}
echo["W4_SOXX_ret5_median"] = {k2: round(float(np.nanmedian(R.placed(ret5[:, None], S[k2], 5)[:, 0])), 3) for k2 in ("B10", "F", "D")}
log("delta_turn_rebound_echo", echo)
(OUT / "verify_audit.json").write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
