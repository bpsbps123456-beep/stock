"""반박 검증(시점 렌즈): 갈래 4 통과 칸 `N20|gvip_rel20_p≥0.5382(상위3분위)`.

사용: py verify_b4_flow_timing.py   → 표준 출력 + results/verify_b4_flow_timing.json

질문: 이 칸의 특징·문턱·진입이 그 날짜에 정말 알 수 있었나(결과를 보기 전에 정한 점검 목록).
  0) 재현: b4_flow.attach + harness.tercile_cells + harness.evaluate_cells(null 끔)로 앞·뒤 수치가 후보와 같은지.
  1) 원자료 when 감사: 사건마다 특징 값의 원자료 날짜 w ≤ t(이탈일 종가)인지, w = t / w < t(묵은 값) 몇 건인지.
  2) 날짜 정렬: 패널 GVIP_SPY(D)를 독립 원천(hf_options yahoo_GVIP 종가 ÷ 하네스 SPY 종가)의 D−1·D·D+1과 비교.
     D에서 가장 잘 맞아야 한다(D+1에 맞으면 하루 미래 값).
  3) 독립 재계산: b4 함수 없이 직접 짠 비율→20행 변화율→250개 백분위→'t 이하 최근 값(3거래일 안)'으로 특징을
     다시 만들고 b4 값과 비교, 독립 원천 값으로 칸 표지·뒤 차이를 다시 잰다.
  4) 미래 값 교란(b4 함수 경로 그대로):
     A) 기준일 c 뒤(date > c) GVIP_SPY 행만 ×U(0.6, 1.4) → t ≤ c 사건 특징이 하나라도 바뀌면 누수.
     B) date == c 행만 교란 → t = c 사건은 바뀌어야 한다(검출력: t 종가 값을 실제로 쓴다는 확인).
     C) date == c 다음 거래일(진입일 t+1의 종가) 행만 교란 → t = c 사건은 그대로여야 한다.
     D) 2026-05-29 뒤 전부 교란 → 3분위 문턱(앞 구간 N20 사건)과 앞 구간 표지가 그대로인지.
     E) 전체 attach 경로(종목 특징 포함)로 기준일 3개 반복.
  5) 고르기 결과 창 겹침: 앞 구간 사건의 h10 청산이 6월로 넘어가므로 '이 칸을 고른 정보'가 다 모인 날 K를 구하고,
     (a) 청산일 > 2026-05-29인 앞 구간 사건을 빼고 다시 골라도 이 칸이 상위 10에 드는지,
     (b) 뒤 구간을 K 이후 이탈(진입 t+1 > K)로만 잡아도 통과하는지.
  6) 진입 가격 감사: 뒤 구간 켜진 사건의 공매도 손익을 일봉에서 직접 다시 계산(t+1 시가 진입, 그 뒤 종가만 보고 청산).
  7) 공개 시차 민감도(누수 아님, 참고): 특징을 하루 묵힌 값(w ≤ t−1)·하루 미래 값(w ≤ t+1, 일부러 누수)으로 바꿔 같은
     칸(문턱은 각 값의 앞 구간 3분위)을 재면 얼마나 달라지나. 그리고 b4 가설 표의 `*_lag1`이 실제로는 하루 미래 값인지.
판정: 1)·4A·4C·4D 위반이 하나라도 있거나 2)에서 D+1이 가장 잘 맞거나 5)에서 K 이후로 통과가 사라지면 refuted.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import harness as H  # noqa: E402
import b4_flow as B  # noqa: E402

CELL = "N20|gvip_rel20_p≥0.5382(상위3분위)"
FEAT = "gvip_rel20_p"
OUT = H.RESULTS / "verify_b4_flow_timing.json"
YAHOO_GVIP = H.ANALYSIS / "hf_options/cache/raw_hf_proxies/yahoo_GVIP.parquet"
FRONT_END = pd.Timestamp(H.FRONT[1])
RNG = np.random.default_rng(20261014)


def gvip_feature(pan, cal, lag=0):
    """b4_flow.date_features의 gvip_rel20_p 줄(169~170행)을 그 함수 그대로 부른다. 반환 (진입일 색인 값, when)."""
    g = B._series(pan, "GVIP_SPY")
    p = B.pct_rank(g / g.shift(20) - 1, 250, 60)
    v, w = B.place(p.to_numpy(), p.index, cal, B.DAILY_AGE)
    return v, w


def event_values(ev, v, w, lag=0):
    e = ev["i"].to_numpy() + 1 + lag
    ok = (e >= 0) & (e < len(v))
    ec = np.where(ok, e, 0)
    vals = np.where(ok, v[ec], np.nan)
    whens = np.where(ok, w[ec], np.datetime64("NaT"))
    return vals, pd.to_datetime(whens)


def n20_label(ev, col=FEAT):
    """tercile_cells로 N20 상위 3분위 칸 표지와 문턱(정확값)을 만든다."""
    cells = dict(H.tercile_cells(ev, [col], by=("N",)))
    name = [k for k in cells if k.startswith(f"N20|{col}≥")][0]
    xf = ev.loc[(ev["N"] == 20) & H.in_period(ev, H.FRONT) & ev[col].notna(), col].astype(float)
    q2 = float(np.quantile(xf, 2 / 3))
    return name, cells[name], q2


def cell_stats(ev, lab, period):
    m = (ev["N"] == 20) & H.in_period(ev, period) & ev["short_ret_h10"].notna() & lab.reindex(ev.index).notna()
    g = ev[m]
    M = lab.reindex(g.index).to_numpy(float)[:, None]
    y = g["short_ret_h10"].to_numpy(float)
    gc = H._codes(g["t"].to_numpy())
    n1, n0, m1, m0, d, t = H.cluster_diff(M, y, gc)
    on = M[:, 0] == 1
    return {"n_on": int(n1[0]), "dates_on": int(g.loc[on, "t"].nunique()), "n_off": int(n0[0]),
            "mean_on": float(m1[0]), "mean_off": float(m0[0]), "diff": float(d[0]), "t": float(t[0])}


def main():
    t0 = time.time()
    R = {"cell": CELL}
    cal = H.load_panel().cal
    P = H.load_panel()
    ev0 = H.load_events_all()
    assert (pd.DatetimeIndex(cal[ev0["i"].to_numpy()]) == pd.DatetimeIndex(ev0["t"])).all(), "i ↔ t 불일치"
    src = B.load_sources()
    pan = src["pan"]
    print("자료 적재", round(time.time() - t0, 1), "s")

    # ── 0) 재현 ──
    ev = B.attach(ev0, src=src, cal=cal)
    saved = pd.read_parquet(H.RESULTS / "b4_flow_events.parquet")
    same_saved = bool(np.allclose(ev[FEAT].to_numpy(float), saved[FEAT].to_numpy(float), equal_nan=True, atol=0, rtol=0))
    cells_all = H.tercile_cells(ev, B.FEAT_NAMES, by=("N",))
    tab, summ = H.evaluate_cells(ev, cells_all, null=False)
    row = tab[tab["cell"] == CELL].iloc[0]
    R["reproduce"] = {"saved_events_identical": same_saved, "n_cells": len(cells_all), "rank": float(row["rank"]),
                      "f_diff": float(row["f_diff"]), "b_diff": float(row["b_diff"]), "b_t": float(row["b_t"]),
                      "b_p": float(row["b_p"]), "b_mean_on": float(row["b_mean_on"]), "b_n_on": int(row["b_n_on"]),
                      "pass": bool(row["pass"]), "passed_cells": summ["passed"]}
    print("0) 재현", R["reproduce"])
    name, lab, q2 = n20_label(ev)
    assert name == CELL, name
    R["threshold_exact"] = q2

    # ── 1) 원자료 when 감사 ──
    v0, w0 = gvip_feature(pan, cal)
    vals, whens = event_values(ev, v0, w0)
    same_path = bool(np.allclose(vals, ev[FEAT].to_numpy(float), equal_nan=True, rtol=0, atol=0))
    t = pd.DatetimeIndex(ev["t"])
    entry = pd.DatetimeIndex(cal[np.minimum(ev["i"].to_numpy() + 1, len(cal) - 1)])
    has = ~np.isnan(vals)
    viol = has & ((whens > t) | (whens >= entry))
    eq = has & (whens == t)
    stale = has & (whens < t)
    n20b_on = (ev["N"] == 20) & H.in_period(ev, H.BACK) & (lab == 1) & ev["short_ret_h10"].notna()
    lagd = [int(np.searchsorted(cal, t_) - np.searchsorted(cal, w_)) for t_, w_ in zip(t[stale], whens[stale])]
    R["when_audit"] = {
        "gvip_only_path_equals_attach": same_path, "events_with_value": int(has.sum()),
        "violations_w_after_t": int(viol.sum()), "w_equals_t": int(eq.sum()), "w_before_t": int(stale.sum()),
        "stale_lag_tradingdays_counts": pd.Series(lagd).value_counts().sort_index().to_dict() if lagd else {},
        "back_on_N20": (pd.DataFrame({"t": t[n20b_on.to_numpy()], "w": whens[n20b_on.to_numpy()]})
                        .groupby("t")["w"].agg(lambda s: ",".join(sorted({str(x.date()) for x in s})))
                        .rename(lambda d: str(d.date())).to_dict()),
    }
    print("1) when", {k: v for k, v in R["when_audit"].items() if k != "back_on_N20"})
    print("   뒤 켜짐 날짜 → 원자료 날짜", R["when_audit"]["back_on_N20"])

    # ── 2) 날짜 정렬(독립 원천) ──
    yg = pd.read_parquet(YAHOO_GVIP)["Close"]
    yg.index = pd.to_datetime(yg.index)
    spy = pd.Series(P.col("C", "SPY"), index=cal)
    indep = (yg.reindex(cal) / spy).dropna()
    pg = pan.set_index("date")["GVIP_SPY"].dropna()
    pg = pg[pg.index.isin(cal)]
    align = {}
    pos = pd.Series(np.arange(len(cal)), index=cal)
    for k in (-1, 0, 1):
        ip = pos.reindex(pg.index).to_numpy() + k
        ok = (ip >= 0) & (ip < len(cal))
        other = indep.reindex(cal[ip[ok]]).to_numpy()
        rel = np.abs(pg.to_numpy()[ok] / other - 1)
        rel = rel[np.isfinite(rel)]
        align[f"panel(D) vs indep(D{k:+d})"] = {"n": int(len(rel)), "median_abs_rel": float(np.median(rel)),
                                                "share_lt_1e-4": float((rel < 1e-4).mean())}
    R["alignment"] = align
    print("2) 정렬", align)

    # ── 3) 독립 재계산(b4 함수 없이) ──
    def my_feature(ratio):
        r = ratio.dropna()
        x = (r / r.shift(20) - 1).dropna()
        a = x.to_numpy()
        p = np.full(len(a), np.nan)
        for j in range(len(a)):
            w = a[max(0, j - 249):j + 1]
            if len(w) < 60:
                continue
            prev = w[:-1]
            p[j] = (prev < w[-1]).mean() + 0.5 * (prev == w[-1]).mean()
        s = pd.Series(p, index=x.index).dropna()
        out = np.full(len(ev), np.nan)
        src_pos = pos.reindex(s.index)
        sp = src_pos.dropna().astype(int)
        sv = s[sp.index].to_numpy()
        spp = sp.to_numpy()
        for k, i in enumerate(ev["i"].to_numpy()):
            j = np.searchsorted(spp, i, side="right") - 1       # t 이하 가장 최근 원자료
            if j >= 0 and spp[j] >= i - B.DAILY_AGE:            # place: 놓인 날(D+1)부터 age일 → D ≥ t−3
                out[k] = sv[j]
        return out

    mine_panel = my_feature(pan.set_index("date")["GVIP_SPY"])
    d_panel = np.nanmax(np.abs(mine_panel - ev[FEAT].to_numpy(float)))
    nan_mismatch = int((np.isnan(mine_panel) != np.isnan(ev[FEAT].to_numpy(float))).sum())
    mine_indep = my_feature(indep)
    ev_i = ev[["sym", "N", "t", "i", "short_ret_h10"]].copy()
    ev_i[FEAT] = mine_indep
    _, lab_i, q2_i = n20_label(ev_i)
    m20 = (ev["N"] == 20) & ev["short_ret_h10"].notna()
    flips = {}
    for pn, per in (("front", H.FRONT), ("back", H.BACK)):
        mm = m20 & H.in_period(ev, per)
        a, b = lab[mm], lab_i[mm]
        flips[pn] = int(((a != b) & a.notna() & b.notna()).sum())
    R["independent"] = {"own_code_vs_b4_max_abs_diff": float(d_panel), "own_code_nan_mismatch": nan_mismatch,
                        "indep_source_threshold": q2_i, "indep_label_flips": flips,
                        "indep_back": cell_stats(ev_i, lab_i, H.BACK), "indep_front": cell_stats(ev_i, lab_i, H.FRONT),
                        "b4_back": cell_stats(ev, lab, H.BACK), "b4_front": cell_stats(ev, lab, H.FRONT)}
    print("3) 독립 재계산", R["independent"])

    # ── 4) 미래 값 교란(b4 함수 경로) ──
    n20 = ev["N"] == 20
    dates = sorted(pd.DatetimeIndex(ev.loc[H.in_period(ev, ("2025-01-02", "2026-09-29")), "t"].unique()))
    base = vals
    A_bad = A_detect = B_bad = B_n = C_bad = C_n = 0
    A_past = 0
    for c in dates:
        c = pd.Timestamp(c)
        ci = int(np.searchsorted(cal, c))
        nxt = cal[ci + 1] if ci + 1 < len(cal) else None
        # A) date > c 전부 교란
        pa = pan.copy()
        fut = pa["date"] > c
        pa.loc[fut, "GVIP_SPY"] = pa.loc[fut, "GVIP_SPY"].to_numpy() * RNG.uniform(0.6, 1.4, int(fut.sum()))
        va, _ = event_values(ev, *gvip_feature(pa, cal))
        past = (t <= c)
        neq = ~((va == base) | (np.isnan(va) & np.isnan(base)))
        A_bad += int(neq[past].sum())
        A_past += int(past.sum())
        A_detect += int(neq[~past].sum())
        # B) date == c 만 교란 → t == c 사건 값이 바뀌어야
        pb = pan.copy()
        pb.loc[pb["date"] == c, "GVIP_SPY"] *= 1.3
        vb, _ = event_values(ev, *gvip_feature(pb, cal))
        at = (t == c) & ~np.isnan(base)
        neqb = ~((vb == base) | (np.isnan(vb) & np.isnan(base)))
        B_n += int(at.sum())
        B_bad += int((at & ~neqb).sum())
        # C) date == 진입일(t+1) 만 교란 → t == c 사건 그대로
        if nxt is not None:
            pc = pan.copy()
            pc.loc[pc["date"] == nxt, "GVIP_SPY"] *= 1.3
            vc, _ = event_values(ev, *gvip_feature(pc, cal))
            neqc = ~((vc == base) | (np.isnan(vc) & np.isnan(base)))
            C_n += int(((t <= c)).sum())
            C_bad += int((neqc & (t <= c)).sum())
    R["perturb"] = {"cutoffs": len(dates), "A_past_event_checks": A_past, "A_past_changed": A_bad,
                    "A_future_changed(detection)": A_detect,
                    "B_events_t_eq_c": B_n, "B_unchanged(should be 0 unless tie)": B_bad,
                    "C_event_checks_t_le_c": C_n, "C_changed(should be 0)": C_bad}
    print("4A~C 교란", R["perturb"])
    # D) 5/29 뒤 전부 교란 → 문턱·앞 표지 그대로
    Dres = []
    for rep in range(20):
        pd_ = pan.copy()
        fut = pd_["date"] > FRONT_END
        pd_.loc[fut, "GVIP_SPY"] = pd_.loc[fut, "GVIP_SPY"].to_numpy() * RNG.uniform(0.6, 1.4, int(fut.sum()))
        vd, _ = event_values(ev, *gvip_feature(pd_, cal))
        evd = ev[["sym", "N", "t", "i", "short_ret_h10"]].copy()
        evd[FEAT] = vd
        nm, labd, q2d = n20_label(evd)
        fr = n20 & H.in_period(ev, H.FRONT)
        Dres.append({"thr_same": q2d == q2, "name_same": nm == CELL,
                     "front_labels_changed": int(((labd != lab) & fr & (labd.notna() | lab.notna())).sum()),
                     "back_labels_changed": int(((labd != lab) & n20 & H.in_period(ev, H.BACK)
                                                 & (labd.notna() | lab.notna())).sum())})
    R["perturb_D_after_front"] = {"reps": len(Dres), "thr_same_all": all(d["thr_same"] for d in Dres),
                                  "name_same_all": all(d["name_same"] for d in Dres),
                                  "front_labels_changed_max": max(d["front_labels_changed"] for d in Dres),
                                  "back_labels_changed_mean(detection)": float(np.mean([d["back_labels_changed"] for d in Dres]))}
    print("4D 문턱", R["perturb_D_after_front"])
    # E) 전체 attach 경로
    Eres = []
    for c in ("2026-05-29", "2026-06-24", "2026-07-02"):
        c = pd.Timestamp(c)
        s2 = dict(src)
        pa = pan.copy()
        fut = pa["date"] > c
        pa.loc[fut, "GVIP_SPY"] = pa.loc[fut, "GVIP_SPY"].to_numpy() * RNG.uniform(0.6, 1.4, int(fut.sum()))
        s2["pan"] = pa
        eva = B.attach(ev0, src=s2, cal=cal)
        a, b = eva[FEAT].to_numpy(float), ev[FEAT].to_numpy(float)
        neq = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        past = (t <= c)
        bad_t = B.check_timing(eva, cal)
        Eres.append({"cutoff": str(c.date()), "past_changed": int(neq[past].sum()), "past_events": int(past.sum()),
                     "future_changed": int(neq[~past].sum()), "timing_violations_all_feats": int(sum(bad_t.values()))})
    R["perturb_E_full_attach"] = Eres
    print("4E 전체 경로", Eres)

    # ── 5) 고르기 결과 창 겹침 ──
    fr_all = H.in_period(ev, H.FRONT) & ev["short_ret_h10"].notna()
    exits = pd.to_datetime(ev.loc[fr_all, "exit_date"])
    K = exits.max()
    late = fr_all & (pd.to_datetime(ev["exit_date"]) > FRONT_END)
    late20 = late & n20
    lab_late = lab[late20]
    R["selection_overlap"] = {"K_last_front_exit": str(K.date()), "front_events": int(fr_all.sum()),
                              "front_events_exit_after_0529": int(late.sum()),
                              "N20_front_exit_after_0529": int(late20.sum()),
                              "N20_late_on": int((lab_late == 1).sum()),
                              "N20_late_on_mean": float(ev.loc[late20 & (lab == 1), "short_ret_h10"].mean()),
                              "N20_late_off_mean": float(ev.loc[late20 & (lab == 0), "short_ret_h10"].mean())}
    # (a) 늦게 청산된 앞 구간 사건을 빼고 고르기(문턱은 그대로: 특징 값만 쓰므로 5/29에 이미 정해짐)
    ev_p = ev[~late]
    tab_p, summ_p = H.evaluate_cells(ev_p, cells_all, null=False)
    rp = tab_p[tab_p["cell"] == CELL].iloc[0]
    R["selection_overlap"]["purged"] = {"f_diff": float(rp["f_diff"]), "f_n_on": int(rp["f_n_on"]),
                                        "f_dates_on": int(rp["f_dates_on"]), "eligible": bool(rp["eligible"]),
                                        "rank": float(rp["rank"]) if np.isfinite(rp["rank"]) else None,
                                        "pass": bool(rp["pass"]), "b_p": float(rp["b_p"]),
                                        "selected": summ_p["selected"], "passed": summ_p["passed"]}
    # (b) 뒤 구간을 K 이후 이탈로만(진입 t+1 > K)
    bstart = cal[np.searchsorted(cal, K)]
    tab_k, summ_k = H.evaluate_cells(ev, cells_all, back=(str(bstart.date()), H.BACK[1]), null=False)
    rk = tab_k[tab_k["cell"] == CELL].iloc[0]
    R["selection_overlap"]["back_from_K"] = {"back_start": str(bstart.date()), "b_diff": float(rk["b_diff"]),
                                             "b_t": float(rk["b_t"]), "b_p": float(rk["b_p"]),
                                             "b_mean_on": float(rk["b_mean_on"]), "b_n_on": int(rk["b_n_on"]),
                                             "b_dates_on": int(rk["b_dates_on"]), "pass": bool(rk["pass"])}
    # (a)+(b) 함께
    tab_pk, _ = H.evaluate_cells(ev_p, cells_all, back=(str(bstart.date()), H.BACK[1]), null=False)
    rpk = tab_pk[tab_pk["cell"] == CELL].iloc[0]
    R["selection_overlap"]["purged_and_back_from_K"] = {"rank": float(rpk["rank"]) if np.isfinite(rpk["rank"]) else None,
                                                        "b_diff": float(rpk["b_diff"]), "b_t": float(rpk["b_t"]),
                                                        "b_p": float(rpk["b_p"]), "b_mean_on": float(rpk["b_mean_on"]),
                                                        "pass": bool(rpk["pass"])}
    print("5) 겹침", R["selection_overlap"])

    # ── 6) 진입 가격 감사(뒤 구간 켜진 사건) ──
    sub = ev[n20b_on]
    worst = 0.0
    for r in sub.itertuples():
        o, c = P.col("O", r.sym), P.col("C", r.sym)
        i = int(r.i)
        entry_px = o[i + 1]
        res = None
        for k in range(i + 1, i + 10):
            if c[k] > r.L:
                res = (entry_px - o[k + 1]) / entry_px - H.COST
                break
        if res is None:
            res = (entry_px - c[i + 10]) / entry_px - H.COST
        worst = max(worst, abs(res - r.short_ret_h10))
        assert abs(entry_px - r.entry_open) < 1e-9
        assert c[i] < r.L                                       # t 종가로 이탈 확인
    R["entry_audit"] = {"back_on_events": int(len(sub)), "max_abs_diff_recomputed_short_ret": worst,
                        "entry_is_t_plus_1_open": True}
    print("6) 진입", R["entry_audit"])

    # ── 7) 공개 시차 민감도(참고) ──
    sens = {}
    for nm, lg in (("stale_1d(w<=t-1)", -1), ("as_coded(w<=t)", 0), ("leak_1d(w<=t+1)", 1)):
        vv, ww = event_values(ev, v0, w0, lag=lg)
        evs = ev[["sym", "N", "t", "i", "short_ret_h10", "exit_date"]].copy()
        evs[FEAT] = vv
        _, labs, q2s = n20_label(evs)
        cs = dict(H.tercile_cells(evs, [FEAT], by=("N",)))
        ks = [k for k in cs if k.startswith(f"N20|{FEAT}≥")][0]
        evs["episode"] = ev["episode"]
        evs["excess_h10"] = ev["excess_h10"]
        evs["kind"] = ev["kind"]
        tb, _ = H.evaluate_cells(evs, [(ks, cs[ks])], null=False)
        rr = tb.iloc[0]
        sens[nm] = {"threshold": q2s, "f_diff": float(rr["f_diff"]), "b_diff": float(rr["b_diff"]),
                    "b_t": float(rr["b_t"]), "b_p(single-cell perm)": float(rr["b_p"]),
                    "b_mean_on": float(rr["b_mean_on"]), "b_n_on": int(rr["b_n_on"]),
                    "w_after_t": int((~np.isnan(vv) & (ww > t)).sum())}
    R["lag_sensitivity"] = sens
    print("7) 시차 민감도", sens)
    # b4 가설 표 *_lag1 시점(b4는 attach(lag=1)을 '하루 늦춤'이라 적었다)
    dv, dw = B.date_features(src, cal)
    e1 = np.minimum(ev["i"].to_numpy() + 2, len(cal) - 1)
    w1 = pd.to_datetime(dw["nvdl_top10"].to_numpy()[e1])
    has1 = ~np.isnan(dv["nvdl_top10"].to_numpy()[e1])
    R["b4_lag1_flag_timing"] = {"events_with_value": int(has1.sum()),
                                "source_date_equals_t_plus_1": int((has1 & (w1 == entry)).sum()),
                                "source_date_after_t": int((has1 & (w1 > t)).sum())}
    print("7b) b4 *_lag1 시점", R["b4_lag1_flag_timing"])
    R["elapsed_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("저장", OUT)


if __name__ == "__main__":
    main()
