"""exec 갈래 보고: (1) 비용 민감도 (2) 깨어 있음 판 격자 비교 (3) 원안 체결 시각·미뤄진 신호.

입력: results/grid_MU_{5m,1h}.parquet(이상적 체결), results/exec_grid_MU_{5m,1h}_awake.parquet(exec_grid.py)
출력: results/exec_summary.json, results/exec_top20_5m.csv, results/exec_original_fills.csv, results/exec_original_sleep.csv
수익은 모두 단순 수익률 %(exp(로그)−1).
사용: py code/exec_report.py
"""
import json

import numpy as np
import pandas as pd

from engine import *

RES = HERE / "results"
KEYS = ["dir", "dayflat", "b", "s", "tf", "sess", "N", "ratio"]
ORIG = dict(tf=40, sess="eth", N=10, ratio="count", dir="follow", b=0.5, s=0.5, dayflat=False)
HI_ENT = 50  # 거래가 많은 칸: 진입(고르기+확인) ≥ 50회


def pct(x):
    return (np.exp(np.asarray(x, dtype=float)) - 1.0) * 100.0


def r1(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 1)


def med(x):
    x = pct(x)
    return r1(np.median(x)) if len(x) else None


def is_orig(df):
    m = np.ones(len(df), bool)
    for k, v in ORIG.items():
        m &= (df[k] == v).values
    return m


def split_cost(g):
    """비용 0 행과 민감도 행을 칸 키로 맞붙인다."""
    z = g[g.cost == 0].reset_index(drop=True)
    c = g[g.cost > 0][KEYS + ["cost", "ret_sel", "ret_conf"]].rename(
        columns={"cost": "cost_s", "ret_sel": "ret_sel_c", "ret_conf": "ret_conf_c"})
    m = z.merge(c, on=KEYS, how="left", validate="one_to_one")
    assert m.ret_sel_c.notna().all()
    return m


def top(df, col, k):
    return df.sort_values(col, ascending=False, kind="stable").head(k)


def beat_bh(df, col="ret_conf"):
    return r1(100.0 * (df[col] > df.bh_conf).mean())


# ---------------------------------------------------------------------------
# (1) 비용 민감도
# ---------------------------------------------------------------------------

def cost_part(base):
    g = split_cost(pd.read_parquet(RES / f"grid_MU_{base}.parquet"))
    out = {"cells": len(g), "bh_sel": r1(pct(g.bh_sel.iloc[0])), "bh_conf": r1(pct(g.bh_conf.iloc[0]))}
    for s in ("eth", "rth"):
        x = g[g.sess == s]
        out[f"bh_{s}"] = [r1(pct(x.bh_sel.iloc[0])), r1(pct(x.bh_conf.iloc[0]))]
    if base == "5m":
        o = g[is_orig(g)].iloc[0]
        out["original"] = dict(sel0=r1(pct(o.ret_sel)), selc=r1(pct(o.ret_sel_c)), conf0=r1(pct(o.ret_conf)),
                               confc=r1(pct(o.ret_conf_c)), ent=[int(o.ent_sel), int(o.ent_conf)])
    t = top(g, "ret_sel", 20)
    t1 = t.iloc[0]
    out["top1_cost0"] = dict(cell={k: (t1[k].item() if hasattr(t1[k], "item") else t1[k]) for k in KEYS},
                             sel0=r1(pct(t1.ret_sel)), selc=r1(pct(t1.ret_sel_c)),
                             conf0=r1(pct(t1.ret_conf)), confc=r1(pct(t1.ret_conf_c)),
                             ent=[int(t1.ent_sel), int(t1.ent_conf)])
    out["top20_cost0"] = dict(sel0=med(t.ret_sel), selc=med(t.ret_sel_c), conf0=med(t.ret_conf), confc=med(t.ret_conf_c),
                              ent_med=float(np.median(t.ent_sel + t.ent_conf)))
    tc = top(g, "ret_sel_c", 20)
    out["top_repicked_costc"] = dict(top1_selc=r1(pct(tc.ret_sel_c.iloc[0])), top1_confc=r1(pct(tc.ret_conf_c.iloc[0])),
                                     top1_ent=[int(tc.ent_sel.iloc[0]), int(tc.ent_conf.iloc[0])],
                                     top20_selc=med(tc.ret_sel_c), top20_confc=med(tc.ret_conf_c),
                                     top20_ent_med=float(np.median(tc.ent_sel + tc.ent_conf)))
    out["all_median"] = dict(sel0=med(g.ret_sel), selc=med(g.ret_sel_c), conf0=med(g.ret_conf), confc=med(g.ret_conf_c),
                             beat_bh_conf0=beat_bh(g), beat_bh_confc=beat_bh(g, "ret_conf_c"))
    hi = g[(g.ent_sel + g.ent_conf) >= HI_ENT]
    out["hi_ent"] = dict(n=len(hi), sel0=med(hi.ret_sel), selc=med(hi.ret_sel_c), conf0=med(hi.ret_conf),
                         confc=med(hi.ret_conf_c), ent_med=float(np.median(hi.ent_sel + hi.ent_conf)),
                         beat_bh_conf0=beat_bh(hi), beat_bh_confc=beat_bh(hi, "ret_conf_c"))
    for s in ("eth", "rth"):
        x = g[g.sess == s]
        out[f"all_median_{s}"] = dict(sel0=med(x.ret_sel), selc=med(x.ret_sel_c), conf0=med(x.ret_conf), confc=med(x.ret_conf_c))
    return out


# ---------------------------------------------------------------------------
# (2) 깨어 있음 판
# ---------------------------------------------------------------------------

def awake_part(base):
    ideal = pd.read_parquet(RES / f"grid_MU_{base}.parquet")
    aw = pd.read_parquet(RES / f"exec_grid_MU_{base}_awake.parquet")
    a = aw.merge(ideal[KEYS + ["cost", "ret_sel", "ret_conf", "ent_sel", "ent_conf"]], on=KEYS + ["cost"],
                 suffixes=("", "_i"), how="left", validate="one_to_one")
    assert a.ret_sel_i.notna().all()
    out = {"rows": len(a), "feasible_rows": int(a.feasible.sum())}
    tables = []
    for cost_tag, cm in (("cost0", a.cost == 0), ("costc", a.cost > 0)):
        x = a[cm]
        f = x[x.feasible]
        o = {}
        if base == "5m":
            r = x[is_orig(x)].iloc[0]
            o["original"] = dict(sel_i=r1(pct(r.ret_sel_i)), sel_a=r1(pct(r.ret_sel)), conf_i=r1(pct(r.ret_conf_i)),
                                 conf_a=r1(pct(r.ret_conf)), ent_i=[int(r.ent_sel_i), int(r.ent_conf_i)],
                                 ent_a=[int(r.ent_sel), int(r.ent_conf)], mdd_all_a=r1(pct(r.mdd_all)))
        # 이상적 체결로 고른 상위 (전체 격자, 이 비용에서 고름)
        ti = x.sort_values("ret_sel_i", ascending=False, kind="stable").head(20)
        t1 = ti.iloc[0]
        tf_ = ti[ti.feasible]
        o["ideal_top1"] = dict(cell={k: (t1[k].item() if hasattr(t1[k], "item") else t1[k]) for k in KEYS},
                               feasible=bool(t1.feasible), sel_i=r1(pct(t1.ret_sel_i)), sel_a=r1(pct(t1.ret_sel)),
                               conf_i=r1(pct(t1.ret_conf_i)), conf_a=r1(pct(t1.ret_conf)),
                               retain_sel=r1(100 * pct(t1.ret_sel) / pct(t1.ret_sel_i)))
        o["ideal_top20"] = dict(n_feasible=len(tf_), sel_i=med(ti.ret_sel_i), sel_a_feas=med(tf_.ret_sel),
                                sel_i_feas=med(tf_.ret_sel_i), conf_i=med(ti.ret_conf_i), conf_a_feas=med(tf_.ret_conf),
                                conf_i_feas=med(tf_.ret_conf_i),
                                retain_sel_feas=r1(100 * np.median(pct(tf_.ret_sel)) / np.median(pct(tf_.ret_sel_i))))
        # 깨어 있음 판 안에서 새로 고른 상위 (실행 가능 칸만)
        ta = f.sort_values("ret_sel", ascending=False, kind="stable").head(20)
        a1 = ta.iloc[0]
        o["awake_top1"] = dict(cell={k: (a1[k].item() if hasattr(a1[k], "item") else a1[k]) for k in KEYS},
                               sel_a=r1(pct(a1.ret_sel)), conf_a=r1(pct(a1.ret_conf)), sel_i=r1(pct(a1.ret_sel_i)),
                               conf_i=r1(pct(a1.ret_conf_i)), ent_a=[int(a1.ent_sel), int(a1.ent_conf)])
        o["awake_top20"] = dict(sel_a=med(ta.ret_sel), conf_a=med(ta.ret_conf), sel_i=med(ta.ret_sel_i),
                                conf_i=med(ta.ret_conf_i), ent_med=float(np.median(ta.ent_sel + ta.ent_conf)),
                                beat_bh_conf=beat_bh(ta))
        o["feasible_median"] = dict(sel_i=med(f.ret_sel_i), sel_a=med(f.ret_sel), conf_i=med(f.ret_conf_i),
                                    conf_a=med(f.ret_conf), beat_bh_conf_i=beat_bh(f, "ret_conf_i"),
                                    beat_bh_conf_a=beat_bh(f))
        pos = f[pct(f.ret_sel_i) > 0]
        o["retain_sel_median_pos_cells"] = dict(n=len(pos), median_ratio=r1(100 * np.median(pct(pos.ret_sel) / pct(pos.ret_sel_i))),
                                                sum_ratio=r1(100 * pct(pos.ret_sel).sum() / pct(pos.ret_sel_i).sum()))
        # 이상적 체결 고르기 순위 상위 k칸이 깨어 있음 판에서 남기는 몫(중앙값 기준, 실행 가능 칸)
        fi = f.sort_values("ret_sel_i", ascending=False, kind="stable")
        o["retain_by_ideal_rank"] = {
            str(k): dict(sel_i=med(fi.ret_sel_i.head(k)), sel_a=med(fi.ret_sel.head(k)),
                         conf_i=med(fi.ret_conf_i.head(k)), conf_a=med(fi.ret_conf.head(k)))
            for k in (1, 20, 100, 1000, 10000)}
        d = pct(f.ret_sel) - pct(f.ret_sel_i)
        o["sel_diff_pctpt_quantiles"] = {q: r1(np.quantile(d, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
        o["identical_share"] = r1(100 * (np.isclose(f.ret_sel, f.ret_sel_i) & np.isclose(f.ret_conf, f.ret_conf_i)).mean())
        o["rank_corr_sel_conf_awake"] = round(float(f.ret_sel.rank().corr(f.ret_conf.rank())), 3)
        o["rank_corr_ideal_vs_awake_sel"] = round(float(f.ret_sel.rank().corr(f.ret_sel_i.rank())), 3)
        hi = f[(f.ent_sel + f.ent_conf) >= HI_ENT]
        o["hi_ent_awake"] = dict(n=len(hi), sel_a=med(hi.ret_sel), conf_a=med(hi.ret_conf))
        out[cost_tag] = o
        if base == "5m":
            for lab, t in (("ideal_top20", ti), ("awake_top20", ta)):
                t = t[KEYS + ["cost", "feasible", "ret_sel_i", "ret_conf_i", "ret_sel", "ret_conf", "ent_sel_i", "ent_conf_i",
                              "ent_sel", "ent_conf"]].copy()
                for c in ("ret_sel_i", "ret_conf_i", "ret_sel", "ret_conf"):
                    t[c] = pct(t[c]).round(1)
                t.insert(0, "pick", f"{lab}_{cost_tag}")
                tables.append(t)
    return out, tables


# ---------------------------------------------------------------------------
# (3) 원안 체결 시각과 미뤄진 신호
# ---------------------------------------------------------------------------

def original_detail():
    d = load("MU", "5m")
    tb = tf_bars(d, ORIG["tf"], ORIG["sess"])
    tb = tb[tb.day >= WIN["5m"]["warm"]].reset_index(drop=True)
    O, C, V = tb.O.values.astype(np.float64), tb.C.values.astype(np.float64), tb.V.values.astype(np.float64)
    lod = tb.lod.values
    seg = segments(tb.day.values, WIN["5m"])
    p = ratio(O, C, V, ORIG["N"], ORIG["ratio"])
    aw = awake_mask(tb.start)
    ones = np.ones(len(tb), np.bool_)
    pi, ei = positions(O, C, p, lod, ones, ORIG["b"], ORIG["s"], True, False)
    pa, ea = positions(O, C, p, lod, aw, ORIG["b"], ORIG["s"], True, False)
    kst = pd.DatetimeIndex(tb.start).tz_convert("Asia/Seoul").hour.values
    et = pd.DatetimeIndex(tb.start).hour.values
    inwin = seg >= 0
    fills = []
    for lab, ev in (("ideal", ei), ("awake", ea)):
        j = np.where((ev != 0) & inwin)[0]
        for jj in j:
            fills.append(dict(mode=lab, start_et=tb.start[jj].strftime("%Y-%m-%d %H:%M"), kst_hour=int(kst[jj]),
                              side="buy" if ev[jj] == 1 else "sell", seg=int(seg[jj])))
    fills = pd.DataFrame(fills)
    hist = fills.pivot_table(index="kst_hour", columns="mode", values="side", aggfunc="count", fill_value=0)
    # 자는 구간(연속된 자는 봉) 단위로 지연 측정: 그 안의 이상적 체결은 깨어날 때 순 효과만 남는다
    r = np.r_[np.log(O[1:] / O[:-1]), np.log(C[-1] / O[-1])]
    ji = np.where((ei != 0) & inwin)[0]
    asleep = ~aw
    delayed = ji[asleep[ji]]
    rows = []
    j = 0
    n = len(tb)
    while j < n:
        if not asleep[j]:
            j += 1
            continue
        j0 = j
        while j < n and asleep[j]:
            j += 1
        k = j  # 깨어나는 봉(없으면 n)
        f_in = [x for x in range(j0, k) if ei[x] != 0 and seg[x] >= 0]
        if not f_in or k >= n or j0 == 0:
            continue
        before = pi[j0 - 1]
        executed = pi[k] != before  # 깨어났을 때 목표가 잠들기 전 보유와 다르면 그 봉 시가에 1건 체결
        hrs = (tb.start[k] - tb.start[f_in[0]]).total_seconds() / 3600
        rows.append(dict(sleep_start=tb.start[j0].strftime("%Y-%m-%d %H:%M"), wake=tb.start[k].strftime("%Y-%m-%d %H:%M"),
                         n_fills=len(f_in), executed=bool(executed), side=("buy" if pi[k] == 1 else "sell") if executed else "",
                         delay_h=round(hrs, 2), seg=int(seg[f_in[0]])))
    dl = pd.DataFrame(rows)
    dl.to_csv(RES / "exec_original_sleep.csv", index=False, encoding="utf-8-sig")
    # 손익 차 분해(비용 0): Σ (pa − pi) × r
    diff = (pa - pi) * r
    out = {
        "fills_ideal": int(((ei != 0) & inwin).sum()), "fills_awake": int(((ea != 0) & inwin).sum()),
        "fills_ideal_asleep": int(len(delayed)),
        "fills_ideal_asleep_by_side": {s: int(((ei == (1 if s == "buy" else -1)) & inwin & asleep).sum()) for s in ("buy", "sell")},
        "sleep_episodes_with_signal": len(dl),
        "episodes_executed_at_wake": int(dl.executed.sum()) if len(dl) else 0,
        "executed_side": dl[dl.executed].side.value_counts().to_dict() if len(dl) else {},
        "fills_cancelled_in_sleep": int(dl.n_fills.sum() - dl.executed.sum()) if len(dl) else 0,
        "delay_hours_median_executed": r1(dl[dl.executed].delay_h.median()) if len(dl) else None,
        "delay_hours_max_executed": r1(dl[dl.executed].delay_h.max()) if len(dl) else None,
        "kst_hist": {int(h): {m: int(hist.loc[h, m]) for m in hist.columns} for h in hist.index},
        "et_awake_hours": "19:00~20:00, 04:00~10:59 ET (KST 08~24시)",
        "pnl_diff_logpct": {
            "sel": round(100 * float(diff[seg == 0].sum()), 2), "conf": round(100 * float(diff[seg == 1].sum()), 2),
            "late_sell_extra_hold_sel": round(100 * float(diff[(seg == 0) & (pa > pi)].sum()), 2),
            "late_buy_missed_sel": round(100 * float(diff[(seg == 0) & (pa < pi)].sum()), 2),
            "late_sell_extra_hold_conf": round(100 * float(diff[(seg == 1) & (pa > pi)].sum()), 2),
            "late_buy_missed_conf": round(100 * float(diff[(seg == 1) & (pa < pi)].sum()), 2),
        },
        "ret_ideal": [r1(pct((pi * r)[seg == 0].sum())), r1(pct((pi * r)[seg == 1].sum()))],
        "ret_awake": [r1(pct((pa * r)[seg == 0].sum())), r1(pct((pa * r)[seg == 1].sum()))],
        "hold_share_ideal": [round(float(pi[seg == 0].mean()), 3), round(float(pi[seg == 1].mean()), 3)],
        "hold_share_awake": [round(float(pa[seg == 0].mean()), 3), round(float(pa[seg == 1].mean()), 3)],
        "bars_total": int(inwin.sum()), "bars_awake": int((inwin & aw).sum()),
    }
    # 깨어 있는 봉에서 나온 신호도 시가 체결 봉 기준으로 ET 분포 확인
    out["awake_bar_et_starts"] = sorted({tb.start[j].strftime("%H:%M") for j in range(len(tb)) if aw[j]})
    return out, fills


if __name__ == "__main__":
    summary = {"cost": {b: cost_part(b) for b in ("5m", "1h")}}
    tabs = []
    summary["awake"] = {}
    for b in ("5m", "1h"):
        summary["awake"][b], t = awake_part(b)
        tabs += t
    summary["original_detail"], fills = original_detail()
    pd.concat(tabs, ignore_index=True).to_csv(RES / "exec_top20_5m.csv", index=False, encoding="utf-8-sig")
    fills.to_csv(RES / "exec_original_fills.csv", index=False, encoding="utf-8-sig")
    (RES / "exec_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
