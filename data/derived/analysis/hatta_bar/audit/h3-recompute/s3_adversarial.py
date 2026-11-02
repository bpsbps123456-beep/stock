"""Pass 3: adversarial checks on my independently rebuilt H2 (2026-01-02..06-23).

Every slice here is EXPLORATORY (not pre-registered).  The script counts the slices it evaluates.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/geunt/OneDrive/Desktop/주식/data/derived/analysis/hatta_bar/audit/h3-recompute")
import h3r_core as C  # noqa: E402
from s2_compare import D0, D1, day_boot, dedup, summ  # noqa: E402

N_SLICES = 0


def S(df, label, out, col="r1n", boot=True):
    global N_SLICES
    N_SLICES += 1
    r = summ(df, col)
    if boot and len(df) >= 10 and df["day"].nunique() >= 3:
        b = day_boot(df, col)
        r["se"] = b["se"]
        r["p_one_sided"] = b["p"]
    r["sum_r1n"] = float(df[col].sum())
    out[label] = r
    return r


def tod_bucket(m):
    if m < 300:
        return "1_pre_0400-0459"
    if m < 420:
        return "2_pre_0500-0659"
    if m < 480:
        return "3_pre_0700-0759"
    if m < 540:
        return "4_pre_0800-0859"
    if m < 570:
        return "5_pre_0900-0929"
    if m < 585:
        return "6_open15_0930-0944"
    if m < 600:
        return "7_am_0945-0959"
    if m < 660:
        return "8_am_1000-1059"
    return "9_post_1900-1929"


def main():
    sig = pd.read_parquet(C.OUT + "/sig1.parquet")
    sig["inwin"] = C.in_window(sig["mins"].to_numpy())
    per = sig[(sig.day >= D0) & (sig.day <= D1)]
    base = per[per.inwin]
    h2 = dedup(base[base.vgap < -0.03]).copy()
    out = {}

    # ---------- table cross-checks against h3_tables.json (pre-registered rows) ----------
    tab = {}
    tab["base_(1)"] = summ(dedup(base))
    tab["opposite_vgap>=-3%"] = summ(dedup(base[base.vgap >= -0.03]))
    tab["thr_-2%"] = summ(dedup(base[base.vgap < -0.02]))
    tab["thr_-4%"] = summ(dedup(base[base.vgap < -0.04]))
    for mth, g in h2.groupby(h2.day // 100):
        tab[f"month_{mth}"] = summ(g)
    capname = {"<5B": "<50억$", "5-20B": "50억~200억$", "20-100B": "200억~1000억$", "100B+": "1000억$+"}
    for tr, g in h2.groupby("tier"):
        tab[f"cap_{capname.get(tr, tr)}"] = summ(g)
    big = h2.groupby("day")["r1n"].sum().idxmax()
    tab["drop_biggest_day"] = dict(day=int(big), **summ(h2[h2.day != big]))
    # KST 08-24 start
    kst = pd.to_datetime(h2["ts"] * 60, unit="s", utc=True).dt.tz_convert("Asia/Seoul")
    kh = kst.dt.hour.to_numpy()
    tab["kst_awake_0800-2400"] = summ(h2[(kh >= 8)])
    out["table_crosscheck"] = tab

    # ---------- (a) time of day ----------
    a = {}
    h2["bucket"] = h2["mins"].map(tod_bucket)
    for bk, g in h2.groupby("bucket"):
        S(g, bk, a)
    h2["tod4"] = h2["mins"].map(C.tod_in)
    for bk, g in h2.groupby("tod4"):
        S(g, "tod4_" + bk, a)
    out["a_time_of_day"] = a

    # ---------- (b) first minutes and bad-print indicators ----------
    b = {}
    f15 = h2[(h2.mins >= 570) & (h2.mins < 585)]
    f30pre = h2[(h2.mins >= 240) & (h2.mins < 270)]
    b["n_first15_regular"] = int(len(f15))
    b["n_first30_premarket"] = int(len(f30pre))
    S(f15, "first15_regular", b)
    S(f30pre, "first30_premarket", b)
    S(h2[(h2.mins >= 570) & (h2.mins < 575)], "first5_regular_0930-0934", b)
    rest = h2[~(((h2.mins >= 570) & (h2.mins < 585)) | ((h2.mins >= 240) & (h2.mins < 270)))]
    S(rest, "excluding_first15reg_and_first30pre", b)
    # wick isolation: how far the signal-bar low sits below the lowest neighbour low (t-1, t+1)
    h2["wick_pct"] = (np.minimum(h2.o, h2.c) - h2.l) / h2.c
    h2["iso_pct"] = (np.minimum(h2.l_m1, h2.l_p1) - h2.l) / h2.c
    h2["iso_share"] = h2["iso_pct"] / h2["wick_pct"]
    h2["vw_pos"] = (h2.bvw - h2.l) / (h2.h - h2.l)
    q = h2[["trades", "wick_pct", "iso_pct", "iso_share", "vw_pos"]].describe(percentiles=[.1, .25, .5, .75, .9]).T
    b["feature_quantiles"] = q.round(5).to_dict(orient="index")
    b["share_trades_lt20"] = float((h2.trades < 20).mean())
    b["share_trades_lt50"] = float((h2.trades < 50).mean())
    b["share_iso_share_ge_0.8"] = float((h2.iso_share >= 0.8).mean())
    b["share_iso_pct_ge_1pct"] = float((h2.iso_pct >= 0.01).mean())
    S(h2[h2.trades < 20], "sig_trades_lt20", b)
    S(h2[h2.trades >= 20], "sig_trades_ge20", b)
    S(h2[h2.trades >= 50], "sig_trades_ge50", b)
    S(h2[h2.iso_share >= 0.8], "iso_share_ge0.8(low far below neighbours)", b)
    S(h2[h2.iso_share < 0.8], "iso_share_lt0.8", b)
    S(h2[h2.iso_pct >= 0.01], "iso_pct_ge1%", b)
    S(h2[h2.iso_pct < 0.01], "iso_pct_lt1%", b)
    # hit side: is the +1% touch confirmed?
    hits = h2[h2.hit]
    b["hits"] = int(len(hits))
    b["hits_single_bar_touch"] = int((hits.n_touch == 1).sum())
    b["hits_touchbar_close_lt_target"] = int((hits.tb_c < hits.entry * 1.01).sum())
    b["hits_touchbar_vwap_lt_target"] = int((hits.tb_vw < hits.entry * 1.01).sum())
    b["hits_touchbar_trades_lt5"] = int((hits.tb_n < 5).sum())
    weak = (hits.n_touch == 1) & (hits.tb_c < hits.entry * 1.01) & (hits.tb_n < 10)
    b["hits_weak(single bar, close<tgt, trades<10)"] = int(weak.sum())
    out["b_badprint"] = b
    # strict touch: +1% counted only if high >= entry * 1.011 (trade-through; BRIEF.md '엄격 × 1.011')
    h2["r1n_strict"] = h2.r1_strict - h2.c_in - np.where(h2.hit_strict, 0.0, h2.c_out)
    h2s = h2.copy()
    h2s["hit"] = h2s.hit_strict
    h2s["r1"] = h2s.r1_strict
    S(h2s, "strict_target_1.1pct_net", b, col="r1n_strict")

    # ---------- (c) concentration ----------
    c = {}
    bysym = h2.groupby("sym").agg(n=("r1n", "size"), pnl=("r1n", "sum")).sort_values("n", ascending=False)
    tot_n, tot_p = len(h2), h2.r1n.sum()
    top10n = bysym.head(10)
    c["top10_by_count"] = top10n.round(5).to_dict(orient="index")
    c["top10_by_count_share_n"] = float(top10n.n.sum() / tot_n)
    c["top10_by_count_share_pnl"] = float(top10n.pnl.sum() / tot_p)
    top10p = bysym.sort_values("pnl", ascending=False).head(10)
    c["top10_by_pnl"] = top10p.round(5).to_dict(orient="index")
    c["top10_by_pnl_share_n"] = float(top10p.n.sum() / tot_n)
    c["top10_by_pnl_share_pnl"] = float(top10p.pnl.sum() / tot_p)
    c["n_syms"] = int(len(bysym))
    c["syms_with_positive_pnl"] = int((bysym.pnl > 0).sum())
    byday = h2.groupby("day").agg(n=("r1n", "size"), pnl=("r1n", "sum")).sort_values("pnl", ascending=False)
    c["top10_days_by_pnl"] = byday.head(10).round(5).to_dict(orient="index")
    c["top10_days_share_pnl"] = float(byday.head(10).pnl.sum() / tot_p)
    c["top5_days_share_pnl"] = float(byday.head(5).pnl.sum() / tot_p)
    c["days_with_positive_pnl"] = int((byday.pnl > 0).sum())
    c["n_days"] = int(len(byday))
    byday_n = byday.sort_values("n", ascending=False)
    c["top10_days_by_count_share_n"] = float(byday_n.head(10).n.sum() / tot_n)
    c["max_day_share_n"] = float(byday_n.n.iloc[0] / tot_n)
    # same-minute clustering: signals sharing the minute with >=5 other H2 signals
    mc = h2.groupby("ts")["sym"].transform("size")
    c["share_signals_in_minutes_with_ge5_signals"] = float((mc >= 5).mean())
    out["c_concentration"] = c

    # ---------- (d) excluding top-5 pnl symbols / top-5 days ----------
    d = {}
    top5s = list(bysym.sort_values("pnl", ascending=False).head(5).index)
    d["top5_syms"] = top5s
    S(h2[~h2.sym.isin(top5s)], "excl_top5_pnl_syms", d)
    top5d = list(byday.head(5).index)
    d["top5_days"] = [int(x) for x in top5d]
    S(h2[~h2.day.isin(top5d)], "excl_top5_pnl_days", d)
    top10s = list(bysym.sort_values("pnl", ascending=False).head(10).index)
    S(h2[~h2.sym.isin(top10s)], "excl_top10_pnl_syms", d)
    # per-symbol cap: at most 3 signals per symbol (first 3 in time)
    cap3 = h2.sort_values("ts").groupby("sym").head(3)
    S(cap3, "first3_per_symbol", d)
    # symbol-equal-weight mean
    d["symbol_equal_weight_mean_r1n"] = float(bysym.pnl.div(bysym.n).mean())
    out["d_exclusions"] = d

    # ---------- (e) VWAP definition ----------
    e = {}
    reg = base[(base.mins >= 570) & (base.mins < 660)]
    e["regular_signals_all"] = int(len(dedup(reg)))
    h2_reg04 = dedup(reg[reg.vgap < -0.03])
    h2_reg0930 = dedup(reg[reg.rvgap < -0.03])
    S(h2_reg04, "regular_0930-1059_vwap_from_0400", e)
    S(h2_reg0930, "regular_0930-1059_vwap_from_0930", e)
    k1, k2 = set(zip(h2_reg04.sym, h2_reg04.ts)), set(zip(h2_reg0930.sym, h2_reg0930.ts))
    e["overlap_0400_vs_0930"] = dict(both=len(k1 & k2), only_0400=len(k1 - k2), only_0930=len(k2 - k1))
    only04 = h2_reg04[[(s, t) not in k2 for s, t in zip(h2_reg04.sym, h2_reg04.ts)]]
    only0930 = h2_reg0930[[(s, t) not in k1 for s, t in zip(h2_reg0930.sym, h2_reg0930.ts)]]
    S(only04, "only_in_0400_def", e)
    S(only0930, "only_in_0930_def", e)
    # first 15 minutes: regular VWAP from 09:30 has 1-15 bars only
    S(h2_reg0930[h2_reg0930.mins >= 585], "regular_0945-1059_vwap_from_0930", e)
    S(h2_reg04[h2_reg04.mins >= 585], "regular_0945-1059_vwap_from_0400", e)
    # combined alternative H2: pre/post use 04:00 VWAP, regular uses 09:30 VWAP
    alt = base[((base.mins < 570) | (base.mins >= 1140)) & (base.vgap < -0.03)]
    alt = pd.concat([alt, reg[reg.rvgap < -0.03]])
    S(dedup(alt), "H2_alt_regular_uses_0930_vwap", e)
    # is VWAP gap just 'down a lot today'? compare with the day-drop variable on the same base
    e["corr_vgap_vs_dayret_close_base"] = float(base[["vgap", "dayret_close"]].corr().iloc[0, 1])
    # matched-count alternative: lowest dayret_close with the same count as H2 (exploratory)
    bd = dedup(base)
    nn = len(h2)
    thr = bd["dayret_close"].nsmallest(nn).max()
    alt2 = dedup(base[base.dayret_close <= thr])
    e["dayret_threshold_same_count"] = float(thr)
    S(alt2, "alt_dayret_close_matched_count", e)
    kk = set(zip(alt2.sym, alt2.ts)) & set(zip(h2.sym, h2.ts))
    e["overlap_dayret_alt_with_H2"] = len(kk)
    # residual: H2 signals whose day return is NOT extreme (dayret_close > -3%)
    S(h2[h2.dayret_close > -0.03], "H2_with_dayret_close_gt_-3%", e)
    S(h2[h2.dayret_close <= -0.03], "H2_with_dayret_close_le_-3%", e)
    out["e_vwap_definition"] = e
    out["n_exploratory_slices_evaluated"] = N_SLICES
    with open(C.OUT + "/s3_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    h2.to_parquet(C.OUT + "/h2_mine_feat.parquet")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
