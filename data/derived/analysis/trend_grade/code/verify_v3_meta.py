"""V3 선택편향·메타 분석 (2026-10-12, 검증 전용 — 연구 파일은 고치지 않는다).

(a) 7계열 결과 JSON의 변형별 front IC20 ↔ back IC20 순위상관, 구간 묶음 요약, back IC20>0 개수,
    날짜 블록 부트스트랩 신뢰구간, 변형 간 일별 IC 상관으로 본 '유효 독립 변형 수'.
(b) front 1위 clenow90과 FINAL(E14_s10)의 같은 날짜 IC20 차이 시계열 NW t, 블록 부트스트랩으로 본
    1위 자리 안정성, 1위와 통계적으로 구별 안 되는 변형 수.
(c) '9월 이후 긴 창(≥120일·SMA200) +, 짧은 창(≤63일) −' 관찰: late 구간 날짜 블록 부트스트랩,
    창 이름표 무작위 순열, 2024~2026 같은 길이 이동 구간의 기저 빈도, QQQ 조정 뒤 같은 위치의 반등 구간
    (2024·2025), 2025-05~06, '최근 실현 차이를 따라가기' 지속성.
(d) 장세 지표 24개(L·E2 × 12)를 다시 계산해 back on IC>0이 있는지, FINAL 점수에도 같은 12개를 적용.

IC 계산은 harness.evaluate와 같다(t 종가 점수 → t+1 시가 매수 → t+1+h 시가 매도, eligible 안 Spearman,
y2024 날짜는 DRUG 제외). 재계산한 구간 평균이 결과 JSON과 같은지 먼저 확인한다.
실행: PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 py verify_v3_meta.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H  # noqa: E402
import fam_composite as FC  # noqa: E402
import fam_ladder as FL  # noqa: E402
import fam_ma_distance as FD  # noqa: E402
import fam_ma_stack as FS  # noqa: E402
import fam_momentum as FM  # noqa: E402
import fam_regime as FR  # noqa: E402
import fam_trend_quality as FT  # noqa: E402

RES = os.path.join(HERE, "..", "results")
FAMS = ("composite", "ma_stack", "ma_distance", "momentum", "trend_quality", "ladder", "regime")
RNG = np.random.default_rng(20261012)
NBOOT = 2000


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


# ------------------------------------------------------------------ 결과 JSON 모으기

def g(R, w, h, k):
    d = (R or {}).get(w, {}).get(h, {})
    return d.get(k, d.get("all_ic")) if k == "ic" else d.get(k)


def collect_rows():
    rows = []
    for fam in FAMS:
        d = json.load(open(os.path.join(RES, f"{fam}.json"), encoding="utf-8"))
        for r in d["rows"]:
            R = r.get("result") or {}
            rows.append({
                "family": fam, "name": r["name"],
                "kind": ("regime_gate" if fam == "regime" and "×" in r["name"] else
                         "grade_map_only" if g(R, "front", "h20", "ic") is None else "score"),
                "f20": g(R, "front", "h20", "ic"), "f20_t": g(R, "front", "h20", "ic_t"),
                "b20": g(R, "back", "h20", "ic"), "b20_t": g(R, "back", "h20", "ic_t"),
                "y25": g(R, "y2025", "h20", "ic"), "y24": g(R, "y2024", "h20", "ic"),
                "l10": g(R, "late", "h10", "ic"), "l5": g(R, "late", "h5", "ic"),
                "extra": r["name"].startswith("extra_"),
            })
    return rows


def dedupe(rows):
    seen, uniq, dup = {}, [], []
    for r in rows:
        if r["kind"] != "score":
            continue
        sig = tuple(None if r[k] is None else round(r[k], 4) for k in ("f20", "b20", "y25", "y24", "l10"))
        if sig in seen:
            dup.append((r["family"] + ":" + r["name"], seen[sig]))
            continue
        seen[sig] = r["family"] + ":" + r["name"]
        uniq.append(r)
    return uniq, dup


# ------------------------------------------------------------------ 점수 함수 등록부

def registry():
    reg = {}
    for n, _, fn in FC.variants():
        reg[f"composite:{n}"] = fn
    for n, _, fn in FS.build_variants():
        reg[f"ma_stack:{n}"] = fn
    for n, _, fn in FD.build_variants():
        reg[f"ma_distance:{n}"] = fn
    for n, _, fn in FM.variants():
        reg[f"momentum:{n}"] = fn
    for n, _, fn in FT.REGISTRY:
        reg[f"trend_quality:{n}"] = fn
    for s, (fn, _) in FL.SCORES.items():
        for m in ("REL_P", "REL_Q"):
            reg[f"ladder:{s}_{m}"] = fn
    reg["regime:base_L"] = FR.score_L
    reg["regime:base_E2"] = FR.score_E2
    return reg


def ic_series(score: pd.DataFrame, h: int) -> pd.Series:
    """harness.evaluate와 같은 날짜별 Spearman IC. 2025 이전 날짜는 DRUG 제외(y2024 규칙)."""
    e = H.eligible()
    sc = score.reindex(index=e.index, columns=e.columns)
    mask = e & sc.notna()
    pr = H.pct_rank(sc, mask)
    fr = H.forward(h).where(mask)
    ic_all = H._row_spearman(pr, fr)
    cols = [c for c in pr.columns if c != "DRUG"]
    ic_ex = H._row_spearman(pr[cols], fr[cols])
    pre = ic_all.index < pd.Timestamp("2025-01-01")
    out = ic_all.copy()
    out[pre] = ic_ex[pre]
    return out


def build_ic_matrices(uniq, P):
    reg = registry()
    ic = {5: {}, 10: {}, 20: {}}
    t0 = time.time()
    for r in uniq:
        key = r["family"] + ":" + r["name"]
        sc = reg[key](P)
        for h in ic:
            ic[h][key] = ic_series(sc, h)
    print(f"IC series for {len(uniq)} variants: {time.time() - t0:.0f}s", flush=True)
    return {h: pd.DataFrame(v) for h, v in ic.items()}


def wmean(icm: pd.DataFrame, w: str, h: int) -> pd.Series:
    d = H.window_dates(w, h)
    return icm.loc[d].mean(skipna=True)


# ------------------------------------------------------------------ 통계 도구

def spearman(a, b):
    a, b = pd.Series(a), pd.Series(b)
    ok = a.notna() & b.notna()
    return float(a[ok].rank().corr(b[ok].rank()))


def block_idx(n, L, rng):
    nb = int(np.ceil(n / L))
    st = rng.integers(0, n, nb)
    return ((st[:, None] + np.arange(L)[None, :]).ravel()[:n]) % n


def ci(x, q=(2.5, 97.5)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [r4(np.percentile(x, q[0])), r4(np.percentile(x, q[1]))]


def eff_number(icm: pd.DataFrame, dates) -> dict:
    X = icm.loc[dates].dropna(axis=1, how="any")
    X = X.loc[:, X.std() > 0]
    C = np.corrcoef(X.to_numpy(float).T)
    lam = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
    return {"variants": int(X.shape[1]), "days": int(X.shape[0]),
            "pc1_share": r4(lam[0] / lam.sum()), "pc3_share": r4(lam[:3].sum() / lam.sum()),
            "participation_ratio": round(float(lam.sum() ** 2 / (lam ** 2).sum()), 2),
            "median_pair_corr": r4(np.median(C[np.triu_indices_from(C, 1)]))}


# ------------------------------------------------------------------ (a)

def part_a(uniq, IC):
    df = pd.DataFrame(uniq)
    df["key"] = df["family"] + ":" + df["name"]
    n = len(df)
    rho_fb = spearman(df.f20, df.b20)
    pear_fb = float(df.f20.corr(df.b20))
    rho_f25 = spearman(df.f20, df.y25)
    rho_f24 = spearman(df.f20, df.y24)
    rho_fl = spearman(df.f20, df.l10)
    rho_bl = spearman(df.b20, df.l10)
    df["fq"] = pd.qcut(df.f20.rank(method="first"), 5, labels=["Q1(하위)", "Q2", "Q3", "Q4", "Q5(상위)"])
    bins = df.groupby("fq", observed=True).agg(n=("f20", "size"), front_ic20=("f20", "mean"),
                                               back_ic20=("b20", "mean"), back_min=("b20", "min"),
                                               back_max=("b20", "max"), y2025_ic20=("y25", "mean"),
                                               late_ic10=("l10", "mean"))
    bins = {str(k): {c: (int(v) if c == "n" else r4(v)) for c, v in row.items()} for k, row in bins.iterrows()}
    top10 = df.sort_values("f20", ascending=False).head(10)[["key", "f20", "b20", "y25", "l10"]]
    backpos = df[df.b20 > 0][["key", "f20", "b20", "y25", "l10"]]
    # 날짜 블록 부트스트랩: front·back 날짜를 각각 다시 뽑아 변형별 평균 → Spearman
    keys = list(df.key)
    F = IC[20].loc[H.window_dates("front", 20), keys].to_numpy(float)
    B = IC[20].loc[H.window_dates("back", 20), keys].to_numpy(float)
    boots = []
    for _ in range(NBOOT):
        fm = np.nanmean(F[block_idx(len(F), 20, RNG)], axis=0)
        bm = np.nanmean(B[block_idx(len(B), 20, RNG)], axis=0)
        boots.append(spearman(fm, bm))
    # 재계산 일치 확인
    re_f = IC[20].loc[H.window_dates("front", 20), keys].mean()
    re_b = IC[20].loc[H.window_dates("back", 20), keys].mean()
    re_l = IC[10].loc[H.window_dates("late", 10), keys].mean()
    re_25 = IC[20].loc[H.window_dates("y2025", 20), keys].mean()
    re_24 = IC[20].loc[H.window_dates("y2024", 20), keys].mean()
    diff = max(float((re_f.values - df.f20.values).__abs__().max()), float((re_b.values - df.b20.values).__abs__().max()),
               float(np.nanmax(np.abs(re_l.values - df.l10.astype(float).values))),
               float(np.nanmax(np.abs(re_25.values - df.y25.astype(float).values))),
               float(np.nanmax(np.abs(re_24.values - df.y24.astype(float).values))))
    fb_dates = H.window_dates("front", 20).union(H.window_dates("back", 20))
    all_dates = IC[20].index[(IC[20].index >= pd.Timestamp("2024-02-01")) & H.exit_date(20).reindex(IC[20].index).notna()]
    return {
        "unique_variants": n,
        "recompute_max_abs_diff_vs_json": r4(diff),
        "spearman_front_vs_back_ic20": r4(rho_fb), "pearson_front_vs_back_ic20": r4(pear_fb),
        "spearman_front_vs_back_boot95": ci(boots),
        "spearman_front_vs_back_boot_share_below0": r4(np.mean(np.array(boots) < 0)),
        "spearman_front_vs_y2025_ic20": r4(rho_f25), "spearman_front_vs_y2024_ic20": r4(rho_f24),
        "spearman_front_ic20_vs_late_ic10": r4(rho_fl), "spearman_back_ic20_vs_late_ic10": r4(rho_bl),
        "front_ic20_range": [r4(df.f20.min()), r4(df.f20.median()), r4(df.f20.max())],
        "back_ic20_range": [r4(df.b20.min()), r4(df.b20.median()), r4(df.b20.max())],
        "front_ic20_pos": int((df.f20 > 0).sum()), "back_ic20_pos": int((df.b20 > 0).sum()),
        "back_ic20_pos_list": backpos.round(4).to_dict("records"),
        "both_pos": int(((df.f20 > 0) & (df.b20 > 0)).sum()),
        "y2025_ic20_pos": int((df.y25 > 0).sum()), "late_ic10_pos": int((df.l10 > 0).sum()),
        "front_quintile_bins": bins,
        "front_top10": top10.round(4).to_dict("records"),
        "effective_independent_variants_front_back": eff_number(IC[20], fb_dates),
        "effective_independent_variants_2024_2026": eff_number(IC[20], all_dates),
    }


# ------------------------------------------------------------------ (b)

def part_b(uniq, IC, P):
    keys = [r["family"] + ":" + r["name"] for r in uniq]
    fin_fn = FC.make(FC.GRID["E14"], 10)
    ic_fin = {h: ic_series(fin_fn(P), h) for h in (10, 20)}
    ic_c90 = {h: IC[h]["trend_quality:D1_clenow90"] for h in (10, 20)}
    out = {"final_key_in_grid": "composite:E14_s10"}
    for w in ("front", "back", "y2025", "y2024"):
        d = H.window_dates(w, 20)
        a, b = ic_c90[20].loc[d], ic_fin[20].loc[d]
        diff = (a - b).dropna()
        out[w] = {"days": int(len(diff)), "clenow90_ic20": r4(a.mean()), "final_ic20": r4(b.mean()),
                  "diff": r4(diff.mean()), "nw_t_lag20": H.newey_west_t(diff, 20),
                  "nw_t_lag10": H.newey_west_t(diff, 10), "nw_t_lag40": H.newey_west_t(diff, 40),
                  "share_days_clenow_better": r4((diff > 0).mean()),
                  "daily_ic_corr": r4(a.corr(b))}
    # front 1위 자리 안정성: 고유 변형 + FINAL
    d = H.window_dates("front", 20)
    M = IC[20].loc[d, keys].copy()
    out["final_equals_grid_E14_s10_maxdiff"] = r4(float((M["composite:E14_s10"] - ic_fin[20].loc[d]).abs().max()))
    M = M.rename(columns={"composite:E14_s10": "FINAL(E14_s10)"})
    cols = list(M.columns)
    X = M.to_numpy(float)
    obs = np.nanmean(X, axis=0)
    i_c90 = cols.index("trend_quality:D1_clenow90")
    i_fin = cols.index("FINAL(E14_s10)")
    win_counts, rank_fin, rank_c90, dsum = {}, [], [], []
    for _ in range(NBOOT):
        m = np.nanmean(X[block_idx(len(X), 20, RNG)], axis=0)
        order = (-m).argsort()
        rk = np.empty(len(m), int)
        rk[order] = np.arange(1, len(m) + 1)
        win = cols[order[0]]
        win_counts[win] = win_counts.get(win, 0) + 1
        rank_fin.append(rk[i_fin])
        rank_c90.append(rk[i_c90])
        dsum.append(m[i_c90] - m[i_fin])
    top_w = sorted(win_counts.items(), key=lambda kv: -kv[1])[:8]
    # 1위와 구별 안 되는 변형: 각 변형 v에 대해 (clenow90 − v) 일별 차이의 NW t(lag20) < 1.96
    tvals = {}
    for j, c in enumerate(cols):
        if j == i_c90:
            continue
        tvals[c] = H.newey_west_t(pd.Series(X[:, i_c90] - X[:, j]), 20)
    t_arr = np.array([v for v in tvals.values() if np.isfinite(v)])
    obs_rank = int((obs > obs[i_fin]).sum()) + 1
    out["front_selection"] = {
        "candidates": len(cols), "obs_rank_final": obs_rank, "obs_rank_clenow90": int((obs > obs[i_c90]).sum()) + 1,
        "boot_share_clenow90_first": r4(win_counts.get("trend_quality:D1_clenow90", 0) / NBOOT),
        "boot_share_final_first": r4(win_counts.get("FINAL(E14_s10)", 0) / NBOOT),
        "boot_distinct_winners": len(win_counts),
        "boot_top_winners": [[k, r4(v / NBOOT)] for k, v in top_w],
        "boot_rank_final_median_iqr": [int(np.median(rank_fin)), int(np.percentile(rank_fin, 25)), int(np.percentile(rank_fin, 75))],
        "boot_rank_clenow90_median_iqr": [int(np.median(rank_c90)), int(np.percentile(rank_c90, 25)), int(np.percentile(rank_c90, 75))],
        "boot_diff_c90_minus_final_95": ci(dsum), "boot_share_final_ge_c90": r4(np.mean(np.array(dsum) <= 0)),
        "not_sig_worse_than_top_t_lt_1p96": int((t_arr < 1.96).sum()),
        "not_sig_worse_than_top_t_lt_1p645": int((t_arr < 1.645).sum()),
        "sig_worse_than_top": int((t_arr >= 1.96).sum()),
    }
    # 앞 구간 안 분할(1~3월로 고르고 3~5월로 확인): 1위 고르기의 승자의 저주 크기
    a_dates = d[d <= pd.Timestamp("2026-03-13")]
    b_dates = d[d > pd.Timestamp("2026-03-13")]
    ma, mb = M.loc[a_dates].mean(), M.loc[b_dates].mean()
    top_a = ma.idxmax()
    out["front_internal_split"] = {
        "first_half_days": len(a_dates), "second_half_days": len(b_dates),
        "first_half_winner": top_a, "winner_ic_first": r4(ma[top_a]), "winner_ic_second": r4(mb[top_a]),
        "winner_rank_second": int((mb > mb[top_a]).sum()) + 1,
        "spearman_first_vs_second": r4(spearman(ma, mb)),
        "final_ic_first": r4(ma["FINAL(E14_s10)"]), "final_ic_second": r4(mb["FINAL(E14_s10)"]),
        "clenow90_ic_first": r4(ma["trend_quality:D1_clenow90"]), "clenow90_ic_second": r4(mb["trend_quality:D1_clenow90"]),
    }
    return out, ic_fin


# ------------------------------------------------------------------ (c)

def horizon_class(key: str) -> tuple[str, int | None]:
    """추세를 재는 주 창(정규화용 σ60·ATR20·베타 창 W는 제외)으로 long(≥120)·short(≤63)·mid·mixed."""
    fam, n = key.split(":", 1)
    W = None
    mixed = {"C3_ibd_rs_s0", "C3_ibd_rs_s5", "C5_athdist_H", "C5_newhigh_cnt20", "C5_newhigh_cnt60",
             "extra_rs_high", "extra_accel63", "extra_stack_align", "extra_min_z", "extra_ema_ribbon_pct",
             "extra_ribbon_z_long"}
    if fam in ("composite", "ma_stack", "ladder", "regime") or n in mixed or n.startswith("B3_"):
        return "mixed", None
    m = (re.match(r"B1_C_[SE]MA(\d+)$", n) or re.match(r"B2_(?:z|atr)_SMA(\d+)$", n)
         or re.match(r"B5_slope_SMA(\d+)_k\d+$", n) or re.match(r"C1_ret(\d+)_s\d+$", n)
         or re.match(r"C2_voladj(\d+)$", n) or re.match(r"C4_resid_W\d+_L(\d+)$", n)
         or re.match(r"D1_\w+?(\d+)$", n) or re.match(r"D2_er(\d+)_", n)
         or re.match(r"D4_fip(\d+)_", n) or re.match(r"D5_negMDD(\d+)$", n) or re.match(r"D6_updays(\d+)$", n))
    if m:
        W = int(m.group(1))
    elif (m := re.match(r"B4_SMA(\d+)_SMA(\d+)$", n)):
        W = max(int(m.group(1)), int(m.group(2)))
    elif n.startswith("B4_macd"):
        W = 26
    elif (m := re.match(r"D3_aboveSMA(\d+)_(\d+)d$", n)):
        W = max(int(m.group(1)), int(m.group(2)))
    elif n in ("C5_hidist_H252", "C5_hidist_C252", "C5_days_since_H252", "extra_highdist_voladj"):
        W = 252
    elif n == "D5_pct_ret126_plus_negMDD126" or n in ("extra_fip126", "extra_fip126_noskip"):
        W = 126
    elif n in ("extra_clenow252_mp120",):
        W = 252
    elif n in ("extra_clenow90_filt",):
        W = 90
    elif n in ("extra_z50_x_slope50", "extra_slope50_sig"):
        W = 50
    if W is None:
        return "mixed", None
    return ("long" if W >= 120 else "short" if W <= 63 else "mid"), W


def group_daily(icm: pd.DataFrame, keys_long, keys_short):
    L = icm[keys_long].mean(axis=1, skipna=True)
    S = icm[keys_short].mean(axis=1, skipna=True)
    return L, S


def part_c(uniq, IC, P):
    keys = [r["family"] + ":" + r["name"] for r in uniq]
    cls = {k: horizon_class(k) for k in keys}
    kl = [k for k in keys if cls[k][0] == "long"]
    ks = [k for k in keys if cls[k][0] == "short"]
    km = [k for k in keys if cls[k][0] == "mid"]
    jl = {r["family"] + ":" + r["name"]: r for r in uniq}
    out = {"n_long": len(kl), "n_short": len(ks), "n_mid": len(km),
           "n_mixed": sum(1 for k in keys if cls[k][0] == "mixed"),
           "long_members": {k.split(":")[1]: r4(jl[k]["l10"]) for k in kl},
           "short_members": {k.split(":")[1]: r4(jl[k]["l10"]) for k in ks},
           "mid_members": {k.split(":")[1]: r4(jl[k]["l10"]) for k in km}}
    # 관찰 그대로(JSON late IC10)
    l10 = pd.Series({k: jl[k]["l10"] for k in keys}, dtype=float)
    out["late_ic10_json"] = {
        "long_pos": f"{int((l10[kl] > 0).sum())}/{len(kl)}", "short_neg": f"{int((l10[ks] < 0).sum())}/{len(ks)}",
        "long_mean": r4(l10[kl].mean()), "short_mean": r4(l10[ks].mean()), "gap": r4(l10[kl].mean() - l10[ks].mean()),
        "mid_mean": r4(l10[km].mean()),
        "spearman_window_vs_late_ic10": r4(spearman([cls[k][1] for k in kl + ks + km], l10[kl + ks + km].to_numpy(float))),
    }
    res = {}
    for h in (5, 10):
        dts = H.window_dates("late", h)
        Ld, Sd = group_daily(IC[h].loc[dts], kl, ks)
        gap = (Ld - Sd).dropna()
        obs = {"days": int(len(gap)), "first": str(dts[0].date()), "last": str(dts[-1].date()),
               "independent_periods": round(len(gap) / h, 1),
               "long_mean": r4(Ld.mean()), "short_mean": r4(Sd.mean()), "gap": r4(gap.mean()),
               "gap_nw_t": H.newey_west_t(gap, h), "long_nw_t": H.newey_west_t(Ld, h), "short_nw_t": H.newey_west_t(Sd, h),
               "share_days_gap_pos": r4((gap > 0).mean())}
        # 날짜 블록 부트스트랩(블록 = h, 원형)
        X = IC[h].loc[dts, kl + ks].to_numpy(float)
        nL = len(kl)
        bl, bs, split = [], [], 0
        for _ in range(NBOOT):
            idx = block_idx(len(X), h, RNG)
            m = np.nanmean(X[idx], axis=0)
            a, b = np.nanmean(m[:nL]), np.nanmean(m[nL:])
            bl.append(a)
            bs.append(b)
            split += int(a > 0 and b < 0)
        bl, bs = np.array(bl), np.array(bs)
        obs["boot_long_95"] = ci(bl)
        obs["boot_short_95"] = ci(bs)
        obs["boot_gap_95"] = ci(bl - bs)
        obs["boot_share_gap_le0"] = r4(np.mean(bl - bs <= 0))
        obs["boot_share_split_holds"] = r4(split / NBOOT)
        # 창 이름표 순열: long·short 이름표를 섞어도 같은 크기 차이가 나오는가(창 길이가 차이를 설명하는가)
        m_all = IC[h].loc[dts, kl + ks].mean()
        vals = m_all.to_numpy(float)
        perm = []
        for _ in range(NBOOT):
            p = RNG.permutation(len(vals))
            perm.append(np.nanmean(vals[p[:nL]]) - np.nanmean(vals[p[nL:]]))
        obs["label_perm_p_gap"] = r4(np.mean(np.array(perm) >= obs["gap"]))
        res[f"h{h}"] = obs
    out["late"] = res
    # 기저 빈도: 2024-02~2026-08의 같은 길이(late h10 날짜 수) 이동 구간
    h = 10
    nlate = res["h10"]["days"]
    obs_gap = res["h10"]["gap"]
    cal = IC[h].index
    ex = H.exit_date(h).reindex(cal)
    hist = cal[(cal >= pd.Timestamp("2024-02-01")) & (cal < pd.Timestamp("2026-09-01")) & ex.notna()]
    Ld, Sd = group_daily(IC[h].loc[hist], kl, ks)
    rl = Ld.rolling(nlate).mean().dropna()
    rs = Sd.rolling(nlate).mean().reindex(rl.index)
    split = (rl > 0) & (rs < 0)
    rev = (rl < 0) & (rs > 0)
    gapr = rl - rs
    nonover = gapr.iloc[::nlate]
    out["base_rate_rolling"] = {
        "window_len": nlate, "windows": int(len(rl)), "nonoverlapping_windows": int(len(nonover)),
        "share_split_long_pos_short_neg": r4(split.mean()), "share_reverse_long_neg_short_pos": r4(rev.mean()),
        "share_gap_ge_observed": r4((gapr >= obs_gap).mean()),
        "nonoverlap_share_gap_ge_observed": r4((nonover >= obs_gap).mean()),
        "nonoverlap_share_split": r4(((rl > 0) & (rs < 0)).iloc[::nlate].mean()),
        "gap_quantiles_5_50_95": [r4(gapr.quantile(q)) for q in (0.05, 0.5, 0.95)],
        "gap_max_and_end_date": [r4(gapr.max()), str(gapr.idxmax().date())],
        "gap_min_and_end_date": [r4(gapr.min()), str(gapr.idxmin().date())],
        "long_mean_max_hist": r4(rl.max()), "short_mean_min_hist": r4(rs.min()),
        "full_period_gap_mean": r4((Ld - Sd).mean()), "full_period_gap_nw_t": H.newey_west_t(Ld - Sd, 10),
        "by_year_gap_mean": {str(y): r4((Ld - Sd)[(Ld - Sd).index.year == y].mean()) for y in (2024, 2025, 2026)},
    }
    # 지속성: 최근 실현 차이(매도 끝난 신호 20개, shift 11)가 +면 다음 차이도 +인가
    full = cal[(cal >= pd.Timestamp("2024-02-01")) & ex.notna()]
    Lf, Sf = group_daily(IC[h].loc[full], kl, ks)
    gf = (Lf - Sf)
    sig = gf.shift(h + 1).rolling(20, min_periods=10).mean()
    pers = {}
    for w in ("y2024", "y2025", "front", "back"):
        d = H.window_dates(w, h)
        gg, ss = gf.reindex(d), sig.reindex(d)
        on, off = gg[ss > 0].dropna(), gg[ss <= 0].dropna()
        pers[w] = {"on_days": int(len(on)), "on_gap": r4(on.mean()), "off_days": int(len(off)), "off_gap": r4(off.mean()),
                   "corr_signal_next_gap": r4(ss.corr(gg))}
    out["persistence_follow_recent_gap"] = pers
    # 비슷한 반등 국면: QQQ 조정(고점 대비 −7% 이하) 저점 뒤 같은 위치
    q = P["bench"]["C"].dropna()
    peak = q.cummax()
    dd = q / peak - 1
    eps, cur = [], None
    for t, v in dd.items():
        if v < 0 and cur is None:
            cur = {"start": t}
        if v == 0 and cur is not None:
            seg = dd.loc[cur["start"]:t]
            cur.update(end=t, trough=seg.idxmin(), depth=float(seg.min()))
            eps.append(cur)
            cur = None
    if cur is not None:
        seg = dd.loc[cur["start"]:]
        cur.update(end=None, trough=seg.idxmin(), depth=float(seg.min()))
        eps.append(cur)
    eps = [e for e in eps if e["depth"] <= -0.07]
    t26 = [e for e in eps if pd.Timestamp("2026-06-01") <= e["trough"] <= pd.Timestamp("2026-08-31")]
    calq = q.index
    an = {"episodes_qqq_dd_le_7pct": [{"peak_start": str(e["start"].date()), "trough": str(e["trough"].date()),
                                       "depth_pct": round(e["depth"] * 100, 1),
                                       "recovered": None if e["end"] is None else str(e["end"].date())} for e in eps]}
    if t26:
        T26 = t26[0]["trough"]
        k = int(calq.get_loc(pd.Timestamp("2026-09-01")) - calq.get_loc(T26))
        an["ref_2026_trough"] = str(T26.date())
        an["offset_trading_days_trough_to_late"] = k
        rows = []
        for e in eps:
            T = e["trough"]
            if T == T26 or T < pd.Timestamp("2023-07-01"):
                continue  # 2023 저점은 252일 창 변형 일부가 아직 NaN(그룹 평균은 있는 변형만)
            i0 = calq.get_loc(T) + k
            for hh, length in ((10, nlate), (10, 40), (20, 40)):
                d = calq[i0: i0 + length]
                d = d[H.exit_date(hh).reindex(d).notna()]
                Lw, Sw = group_daily(IC[hh].loc[d], kl, ks)
                rows.append({"trough": str(T.date()), "depth_pct": round(e["depth"] * 100, 1), "h": hh,
                             "n_long_avail": int(IC[hh].loc[d, kl].notna().all().sum()),
                             "n_short_avail": int(IC[hh].loc[d, ks].notna().all().sum()),
                             "from": str(d[0].date()), "to": str(d[-1].date()), "days": int(len(d)),
                             "long": r4(Lw.mean()), "short": r4(Sw.mean()), "gap": r4((Lw - Sw).mean()),
                             "split_holds": bool(Lw.mean() > 0 and Sw.mean() < 0)})
        an["analog_windows"] = rows
    named = []
    for a, b in (("2025-05-01", "2025-06-30"), ("2025-04-09", "2025-05-30"), ("2024-08-08", "2024-09-30")):
        for hh in (10, 20):
            d = cal[(cal >= pd.Timestamp(a)) & (cal <= pd.Timestamp(b)) & H.exit_date(hh).reindex(cal).notna()]
            Lw, Sw = group_daily(IC[hh].loc[d], kl, ks)
            named.append({"window": f"{a}~{b}", "h": hh, "days": int(len(d)), "long": r4(Lw.mean()),
                          "short": r4(Sw.mean()), "gap": r4((Lw - Sw).mean()),
                          "split_holds": bool(Lw.mean() > 0 and Sw.mean() < 0)})
    an["named_windows"] = named
    out["analogs"] = an
    # 구간별 같은 분할(참고): front/back/2025/2024 h20
    per = {}
    for w in ("front", "back", "y2025", "y2024"):
        d = H.window_dates(w, 20)
        Lw, Sw = group_daily(IC[20].loc[d], kl, ks)
        per[w] = {"long": r4(Lw.mean()), "short": r4(Sw.mean()), "gap": r4((Lw - Sw).mean()),
                  "gap_nw_t": H.newey_west_t(Lw - Sw, 20)}
    out["window_h20_long_short"] = per
    # 월별 h10 차이(2024-02~)
    mg = gf.groupby(gf.index.to_period("M")).mean()
    out["monthly_gap_h10"] = {str(k): r4(v) for k, v in mg.items()}
    return out


# ------------------------------------------------------------------ (d)

def part_d(P):
    tab = json.load(open(os.path.join(RES, "regime.json"), encoding="utf-8"))["table"]
    js = {
        "indicators": len(tab),
        "back_on_ic20_pos": [t["name"] for t in tab if (t["back_on_ic20"] or -9) > 0],
        "back_on_ic10_pos": {t["name"]: t["back_on_ic10"] for t in tab if (t["back_on_ic10"] or -9) > 0},
        "back_on_gt_off_ic20": [t["name"] for t in tab if t["back_on_ic20"] is not None and t["back_off_ic20"] is not None
                                and t["back_on_ic20"] > t["back_off_ic20"]],
        "front_on_gt_off_ic20": [t["name"] for t in tab if t["front_on_ic20"] is not None and t["front_off_ic20"] is not None
                                 and t["front_on_ic20"] > t["front_off_ic20"]],
        "back_on_ic20_max": max((t["back_on_ic20"], t["name"]) for t in tab if t["back_on_ic20"] is not None),
        "no_off_days_in_back": [t["name"] for t in tab if t["back_off_ic20"] is None],
    }
    js["consistent_front_and_back_on_gt_off"] = sorted(set(js["back_on_gt_off_ic20"]) & set(js["front_on_gt_off_ic20"]))
    # 다시 계산(L·E2는 결과 표와 대조, FINAL은 새로 적용)
    FR.BASES["FINAL"] = (FC.make(FC.GRID["E14"], 10), "E14_s10")
    rec, mism = {}, 0
    tabd = {t["name"]: t for t in tab}
    for base in ("L", "E2", "FINAL"):
        score = FR.BASES[base][0](P)
        ind = FR.indicators(P, base)
        for gname in FR.GRID + FR.EXTRA:
            on = ind[gname][1]
            r = {w: FR.split_ic(score, on, w) for w in ("front", "back", "y2025", "y2024", "late")}
            name = f"{base}×{gname}"
            row = {"front_on20": r["front"]["h20"]["on_ic"], "front_off20": r["front"]["h20"]["off_ic"],
                   "back_on20": r["back"]["h20"]["on_ic"], "back_off20": r["back"]["h20"]["off_ic"],
                   "back_on_days20": r["back"]["h20"]["on_days"], "back_on_t20": r["back"]["h20"]["on_t"],
                   "back_on10": r["back"]["h10"]["on_ic"], "back_on5": r["back"]["h5"]["on_ic"],
                   "y2025_on20": r["y2025"]["h20"]["on_ic"], "y2025_off20": r["y2025"]["h20"]["off_ic"],
                   "y2024_on20": r["y2024"]["h20"]["on_ic"], "y2024_off20": r["y2024"]["h20"]["off_ic"],
                   "late_on10": r["late"]["h10"]["on_ic"], "late_off10": r["late"]["h10"]["off_ic"]}
            rec[name] = row
            if name in tabd:
                t = tabd[name]
                for a, b in (("back_on20", "back_on_ic20"), ("back_off20", "back_off_ic20"),
                             ("front_on20", "front_on_ic20"), ("back_on10", "back_on_ic10")):
                    x, y = row[a], t[b]
                    if (x is None) != (y is None) or (x is not None and abs(x - y) > 1e-4):
                        mism += 1
    pos = lambda v: v is not None and v > 0
    summ = {}
    for base in ("L", "E2", "FINAL"):
        names = [n for n in rec if n.startswith(base + "×")]
        summ[base] = {
            "indicators": len(names),
            "back_on_ic20_pos": [n for n in names if pos(rec[n]["back_on20"])],
            "back_on_ic10_pos": {n: rec[n]["back_on10"] for n in names if pos(rec[n]["back_on10"])},
            "back_on_ic5_pos": {n: rec[n]["back_on5"] for n in names if pos(rec[n]["back_on5"])},
            "back_on_gt_off_ic20": [n for n in names if rec[n]["back_on20"] is not None and rec[n]["back_off20"] is not None
                                    and rec[n]["back_on20"] > rec[n]["back_off20"]],
            "back_on_ic20_best": max((rec[n]["back_on20"], n) for n in names if rec[n]["back_on20"] is not None),
            "y2025_on_gt_off_ic20": [n for n in names if rec[n]["y2025_on20"] is not None and rec[n]["y2025_off20"] is not None
                                     and rec[n]["y2025_on20"] > rec[n]["y2025_off20"]],
        }
    return {"from_json_table": js, "recomputed_mismatch_vs_table": mism, "recomputed_summary": summ,
            "recomputed_rows": rec}


# ------------------------------------------------------------------ 실행

def main():
    t0 = time.time()
    P = H.load_panel()
    rows = collect_rows()
    uniq, dup = dedupe(rows)
    kinds = pd.Series([r["kind"] for r in rows]).value_counts().to_dict()
    print("rows", len(rows), kinds, "unique scores", len(uniq), "dup", len(dup), flush=True)
    IC = build_ic_matrices(uniq, P)
    A = part_a(uniq, IC)
    print("(a) done", json.dumps({k: A[k] for k in ("unique_variants", "recompute_max_abs_diff_vs_json",
                                                     "spearman_front_vs_back_ic20", "spearman_front_vs_back_boot95",
                                                     "back_ic20_pos")}, ensure_ascii=False), flush=True)
    B, _ = part_b(uniq, IC, P)
    print("(b) done", json.dumps(B["front"], ensure_ascii=False), flush=True)
    C = part_c(uniq, IC, P)
    print("(c) done", json.dumps(C["late_ic10_json"], ensure_ascii=False), flush=True)
    D = part_d(P)
    print("(d) done mismatch", D["recomputed_mismatch_vs_table"], flush=True)
    out = {
        "generated": "2026-10-12", "panel_last": str(P["C"].index[-1].date()), "nboot": NBOOT,
        "inventory": {"rows_total": len(rows), "by_kind": kinds, "unique_scores": len(uniq),
                      "duplicates_dropped": dup,
                      "per_family_rows": pd.Series([r["family"] for r in rows]).value_counts().to_dict(),
                      "extra_in_unique": int(sum(r["extra"] for r in uniq))},
        "a_front_vs_back": A, "b_clenow90_vs_final": B, "c_late_long_vs_short": C, "d_regime": D,
        "seconds": round(time.time() - t0, 1),
    }
    path = os.path.join(RES, "verify_v3_meta.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    print("saved", os.path.abspath(path), out["seconds"], "s")


if __name__ == "__main__":
    main()
