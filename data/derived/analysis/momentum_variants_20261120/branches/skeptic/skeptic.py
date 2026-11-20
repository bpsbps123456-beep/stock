"""회의적 감사(2026-11-20): 미래 정보 교란 시험, 생존 편향 크기, 선택 편향(기여·구간 나눔).

엔진 파일은 고치지 않는다. 교란은 메모리 안 Engine 복사본의 AC/AO/CL/DV(+기준 ETF)를 바꾼다.
실행: py data/derived/analysis/momentum_variants_20261120/branches/skeptic/skeptic.py
결과: 같은 폴더 results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code"))
sys.path.insert(0, str(HERE.parent / "adaptive"))
sys.path.insert(0, str(HERE.parent / "composite"))
import engine as E  # noqa: E402
import adaptive as AD  # noqa: E402
import composite as CP  # noqa: E402

CUT = "2026-08-10"


def cands(e, B=None, univ_over=None):
    """후보 8개. univ_over: 종목군 이름 바꿔치기(생존 편향 변형용)."""
    B = B or AD.Builder(e)
    U = (lambda u: univ_over.get(u, u)) if univ_over else (lambda u: u)
    return {
        "WT50_lb60_M_n3_ivol": lambda: e.run(e.ret(60), universe=U("WIDETECH50"), n=3, buffer=5, weight="ivol", caps=False, freq="M"),
        "TIME_lb120_W_n3": lambda: e.run(e.ret(120), universe="TIME", n=3, buffer=5, weight="time", caps=False, freq="W"),
        "WT50_lb60_W_n5b0": lambda: e.run(e.ret(60, 0), universe=U("WIDETECH50"), n=5, buffer=0, weight="equal", caps=False, freq="W"),
        "F1_AI_dialALL50": lambda: B.icdial("AI", 7, 60, 20, 10, "rev5", U("ALL50")),
        "RES60_AI_n5": lambda: e.run(CP.residual(e, 60), universe="AI", n=5, buffer=5, weight="equal", caps=True),
        "FUNDS_lb60_n3_b10": lambda: e.run(e.ret(60), universe="FUNDS", n=3, buffer=10, weight="equal", caps=False, freq="W"),
        "TIME_lb120s5_n5": lambda: e.run(e.ret(120, 5), universe="TIME", n=5, buffer=5, weight="time", caps=False),
        "WT50_r60_n7": lambda: e.run(e.ret(60), universe=U("WIDETECH50"), n=7, buffer=5, weight="equal", caps=False),
    }


def met(e, r):
    m = e.metrics(r)
    m["H1_MDD"] = e.mdd(r, "2026-01-02", "2026-06-30")
    return {k: round(float(v), 4) for k, v in m.items()}


# ------------------------------------------------------------------ 1) 교란 시험
def perturb(e, cut, seed=7):
    rng = np.random.default_rng(seed)
    kc = e.pos[cut]
    for nm in ("AC", "AO", "CL", "DV"):
        A = getattr(e, nm).copy()
        A[kc:] *= rng.uniform(0.5, 1.5, size=A[kc:].shape)
        setattr(e, nm, A)
    for nm in ("bench_ac", "bench_ao"):
        df = getattr(e, nm).copy()
        df.iloc[kc:] = df.iloc[kc:].to_numpy() * rng.uniform(0.5, 1.5, size=df.iloc[kc:].shape)
        setattr(e, nm, df)
    e.adv20 = pd.DataFrame(e.DV).rolling(20, min_periods=15).mean().shift(1).to_numpy()
    e._u_cache = {}


def compare(r0, r1, cut):
    hk = [d for d in r0.holdings if d <= cut]
    same_h = all(r0.holdings[d] == r1.holdings.get(d) for d in hk)
    jc = max(j for j, d in enumerate(r0.days) if d < cut)
    dv = float(np.max(np.abs(r0.v_close[: jc + 1] - r1.v_close[: jc + 1])))
    after = [d for d in r0.holdings if d > cut]
    changed_after = sum(1 for d in after if r0.holdings[d] != r1.holdings.get(d))
    return {"rebal_le_cut": len(hk), "holdings_same_le_cut": same_h, "max_abs_vclose_diff_before_cut": dv,
            "rebal_after_cut": len(after), "rebal_after_cut_changed": changed_after}


# ------------------------------------------------------------------ 3) 기여·구간
def weights_path(e, r):
    """세션별 시가 보유 비중 행렬(엔진과 같은 표류). 검산: w·oo 가 r_oo 와 같아야 한다."""
    k0 = e.pos[r.days[0]]
    T = len(r.days)
    W = np.zeros((T, len(e.tickers)))
    w = np.zeros(len(e.tickers))
    reb = {d: h for d, h in r.holdings.items()}
    for j in range(T):
        k = k0 + j
        d = r.days[j]
        if d in reb:
            w = np.zeros(len(e.tickers))
            for t, v in reb[d].items():
                w[e.col[t]] = v
        W[j] = w
        if j + 1 < T:
            oc = e.AC[k] / e.AO[k] - 1
            oc = np.where(np.isfinite(oc), oc, 0.0)
            oo = e.AO[k + 1] / e.AO[k] - 1
            bad = (w > 0) & ~np.isfinite(oo)
            oo = np.where(np.isfinite(oo), oo, np.where(bad, oc, 0.0))
            g = w * (1 + oo)
            cash = 1 - w.sum()
            w = g / (g.sum() + cash) if g.sum() + cash > 0 else w
    return W


def contrib(e, r, W, a, b):
    """[a 시가, b 종가] 구간 종목별 금액 기여(합 = 구간 수익)."""
    k0 = e.pos[r.days[0]]
    js = [j for j, d in enumerate(r.days) if a <= d <= b]
    i0, i1 = js[0], js[-1]
    V0 = r.v_open[i0]
    c = np.zeros(len(e.tickers))
    for j in js:
        k = k0 + j
        if j < i1:
            rr = e.AO[k + 1] / e.AO[k] - 1
            oc = e.AC[k] / e.AO[k] - 1
            rr = np.where(np.isfinite(rr), rr, np.where(np.isfinite(oc), oc, 0.0))
            c += r.v_open[j] / V0 * W[j] * rr
        else:
            oc = e.AC[k] / e.AO[k] - 1
            c += r.v_open[j] / V0 * W[j] * np.where(np.isfinite(oc), oc, 0.0)
    tot = float(r.v_close[i1] / V0 - 1)
    order = np.argsort(-np.abs(c))
    top = [(e.tickers[i], round(float(c[i]), 4)) for i in order[:6] if c[i] != 0]
    return {"total": round(tot, 4), "sum_contrib": round(float(c.sum()), 4), "top": top,
            "ex_top1": round(tot - float(c[order[0]]), 4), "best_pos": e.tickers[int(np.argmax(c))],
            "best_pos_val": round(float(c.max()), 4)}


SEGS = {"H1": ("2026-01-02", "2026-06-30"), "JulAug": ("2026-07-01", "2026-08-31"), "Sep": ("2026-09-01", "2026-09-30"),
        "OctNov": ("2026-10-01", E.LAST), "H2": ("2026-07-01", E.LAST)}


def segs(e, r):
    return {s: round(E.Engine.span(r, a, b), 4) for s, (a, b) in SEGS.items()}


def main():
    out = {}
    e = E.Engine()
    B = AD.Builder(e)
    C = cands(e, B)
    R0 = {k: f() for k, f in C.items()}
    out["reproduce"] = {k: met(e, r) for k, r in R0.items()}
    out["base_segments"] = segs(e, e.run(e.ret(120), universe="TIME", n=7, buffer=5, weight="time", caps=True))
    out["qqq_segments"] = segs(e, e.bench("QQQ"))
    out["soxx_segments"] = segs(e, e.bench("SOXX"))

    # 3) 기여·구간
    att = {}
    for k, r in R0.items():
        W = weights_path(e, r)
        # 검산
        k0 = e.pos[r.days[0]]
        chk = []
        for j in range(len(r.days) - 1):
            kk = k0 + j
            oc = e.AC[kk] / e.AO[kk] - 1
            oc = np.where(np.isfinite(oc), oc, 0.0)
            oo = e.AO[kk + 1] / e.AO[kk] - 1
            bad = (W[j] > 0) & ~np.isfinite(oo)
            oo = np.where(np.isfinite(oo), oo, np.where(bad, oc, 0.0))
            chk.append(abs(float(W[j] @ oo) - r.r_oo[j]))
        # 보유 중 가격 결측(수익 0 처리) 세션 수
        miss = 0
        for j in range(len(r.days)):
            kk = k0 + j
            miss += int(((W[j] > 0) & ~np.isfinite(e.AO[kk])).sum())
        att[k] = {"weights_check_max_err": float(max(chk)), "held_missing_open_cells": miss,
                  "segments": segs(e, r),
                  **{f"contrib_{s}": contrib(e, r, W, a, b) for s, (a, b) in SEGS.items() if s in ("H1", "H2", "JulAug", "OctNov")},
                  "last_holdings": list(r.holdings.items())[-1]}
    out["attribution"] = att

    # 2) 생존 편향: PIT 마스크(2025-06-30·2025-12-31 기준 거래대금 문턱 통과 종목만)
    pit = {}
    for tag, day in (("P0630", "2025-07-01"), ("P1231", "2026-01-02")):
        adv = np.nan_to_num(e.adv20[e.pos[day]], nan=0.0)
        pit[tag] = adv
    orig_u = e.universe

    def uni(name, k):
        for tag in pit:
            if name.endswith("_" + tag):
                base = name[: -len(tag) - 1]
                thr = AD_thr(base)
                return orig_u(base, k) & (pit[tag] >= thr)
        return orig_u(name, k)

    def AD_thr(u):
        for p in ("WIDETECH", "WIDE", "ALL"):
            if u.startswith(p):
                return float(u[len(p):]) * 1e6
        raise ValueError(u)
    e.universe = uni
    surv = {}
    for tag in pit:
        B2 = AD.Builder(e)
        C2 = cands(e, B2, univ_over={"WIDETECH50": f"WIDETECH50_{tag}", "ALL50": f"ALL50_{tag}"})
        for k in ("WT50_lb60_M_n3_ivol", "WT50_lb60_W_n5b0", "WT50_r60_n7", "F1_AI_dialALL50"):
            surv[f"{k}|{tag}"] = {**met(e, C2[k]()), "segments": None}
    # 종목군 크기
    sizes = {}
    for d in ("2026-01-02", "2026-06-01", "2026-09-01", "2026-11-02"):
        k = e.pos[d]
        sizes[d] = {u: int(e.universe(u, k).sum()) for u in ("WIDETECH50", "WIDETECH50_P0630", "WIDETECH50_P1231", "ALL50", "ALL50_P0630", "AI", "FUNDS", "TIME")}
    # 같은 규칙 다른 종목군
    same_rule = {
        "WT50_lb60_M_n3_ivol@FUNDS": e.run(e.ret(60), universe="FUNDS", n=3, buffer=5, weight="ivol", caps=False, freq="M"),
        "WT50_lb60_M_n3_ivol@AI": e.run(e.ret(60), universe="AI", n=3, buffer=5, weight="ivol", caps=False, freq="M"),
        "WT50_lb60_W_n5b0@FUNDS": e.run(e.ret(60), universe="FUNDS", n=5, buffer=0, weight="equal", caps=False, freq="W"),
        "WT50_lb60_W_n5b0@AI": e.run(e.ret(60), universe="AI", n=5, buffer=0, weight="equal", caps=False, freq="W"),
        "WT50_r60_n7@FUNDS": e.run(e.ret(60), universe="FUNDS", n=7, buffer=5, weight="equal", caps=False),
        "WT50_r60_n7@AI": e.run(e.ret(60), universe="AI", n=7, buffer=5, weight="equal", caps=False),
        "F1@u=FUNDS,du=FUNDS": B.icdial("FUNDS", 7, 60, 20, 10, "rev5", "FUNDS"),
        "F1@u=AI,du=AI": B.icdial("AI", 7, 60, 20, 10, "rev5", "AI"),
        "F1@u=FUNDS,du=ALL50": B.icdial("FUNDS", 7, 60, 20, 10, "rev5", "ALL50"),
        "F1_static_ret60_AI": e.run(e.ret(60), universe="AI", n=7, buffer=5, weight="equal", caps=True),
        "RES60@FUNDS": e.run(CP.residual(e, 60), universe="FUNDS", n=5, buffer=5, weight="equal", caps=True),
        "RES60@TIME_equal": e.run(CP.residual(e, 60), universe="TIME", n=5, buffer=5, weight="equal", caps=True),
        "RES60@AI_nocap": e.run(CP.residual(e, 60), universe="AI", n=5, buffer=5, weight="equal", caps=False),
        "RES60@AI_n7": e.run(CP.residual(e, 60), universe="AI", n=7, buffer=5, weight="equal", caps=True),
        "ret60@AI_n5_cap": e.run(e.ret(60), universe="AI", n=5, buffer=5, weight="equal", caps=True),
    }
    e.universe = orig_u
    out["survivorship_pit"] = surv
    out["universe_sizes"] = sizes
    out["same_rule_other_universe"] = {k: {**met(e, r), "segments": segs(e, r)} for k, r in same_rule.items()}
    # AI 종목군 사후 선택 점검: 후보 상위 기여 종목이 2026-01-02 전 펀드 보유였나/2025-06-30 거래대금
    first_snap = e.snap_dates[0]
    held0 = set(e.held.get((first_snap, "TIME"), {})) | set(e.held.get((first_snap, "KoAct"), {}))
    k0630 = e.pos["2025-07-01"]
    chk = {}
    for k in ("RES60_AI_n5", "F1_AI_dialALL50"):
        tops = [t for t, _ in att[k]["contrib_H1"]["top"]] + [t for t, _ in att[k]["contrib_H2"]["top"]]
        chk[k] = {t: {"in_fund_first_snap": t in held0, "adv20_2025_06_30_musd": round(float(np.nan_to_num(e.adv20[k0630][e.col[t]])) / 1e6, 1),
                      "in_wide": bool(e.meta.in_wide.get(t)), "first_px": str(pd.Series(e.AC[:, e.col[t]], index=e.days).first_valid_index())}
                  for t in dict.fromkeys(tops)}
    out["ai_top_names_check"] = chk
    out["first_snapshot"] = first_snap

    # 1) 교란 시험
    ep = E.Engine()
    perturb(ep, CUT)
    Bp = AD.Builder(ep)
    Cp = cands(ep, Bp)
    out["perturb"] = {"cut": CUT, "results": {k: compare(R0[k], Cp[k](), CUT) for k in C}}

    (HERE / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
