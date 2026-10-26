"""갈래 F_hedge: 팔지 않고 덮기(인버스 ETF 헤지) vs 같은 순노출만큼 현금으로 줄이기 (2026-10-10).

7종목 기본 경로는 그대로 두고, 계기(T1~T5)가 켜지면 주식 q만큼을 팔아 인버스 ETF를 산다(주식 1-q + 인버스 q ≤ 1).
같은 계기·같은 q마다 '현금' 짝을 만든다: 그림자의 기초지수 60일 베타 β로 환산해 순노출이 같아지는 주식 비중
  헤지 순베타 = (1-q)·β - q·L   →   현금 짝 노출 e = clip((1-q) - q·L/β, 0, 1)
(L = 인버스 배수, 기초지수: SOXS→SOXX, SQQQ·QID·PSQ→QQQ, SH→SPY).

시점: 모든 계기·β는 t일 종가까지의 값 → H.at_open으로 t 다음 거래일 시가에 체결. 인버스 ETF 수익은 실제 시가→시가(일일 재설정 손실 포함).
격자(사전 고정, 튜닝 금지): 계기 5 × (3배 상품 2개×q 4 + 1·2배 상품 3개×q 3 = 17) = 85 헤지 + 85 현금 짝 = 170.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

# ----------------------------------------------------------------------------- 격자(사전 선언)
TRIGGERS = ("T1_SOXX<SMA50", "T2_VIX/VIX3M>1", "T3_그림자낙폭<=-10%", "T4_그림자변동성20>60%", "T5_상시")
PRODUCTS = {"SOXS": (3, "SOXX"), "SQQQ": (3, "QQQ"), "QID": (2, "QQQ"), "PSQ": (1, "QQQ"), "SH": (1, "SPY")}
Q_3X = (0.05, 0.10, 0.15, 0.20)
Q_LOW = (0.1, 0.2, 0.3)
MODES = ("헤지", "현금")
SMA_N, VIX_TH, DD_TH, VOL_N, VOL_TH = 50, 1.0, -0.10, 20, 0.60
BETA_N, BETA_MINP, BETA_FLOOR, BETA_FALLBACK = 60, 20, 0.2, 1.0
BASES = ("2026", "hold")


def qs(prod):
    return Q_3X if PRODUCTS[prod][0] == 3 else Q_LOW


GRID = [(m, t, p, q) for m in MODES for t in TRIGGERS for p in PRODUCTS for q in qs(p)]


# ----------------------------------------------------------------------------- 신호(t일 종가 기준 값, index = 아는 날)
def shadow_chain() -> pd.Series:
    """그림자(기본 경로) 종가 수익을 보유 기간(주간 PDF)→2026(일별 PDF)으로 이은 것. 창 길이 데우기용(변동성·β)."""
    rs = []
    for bn in ("hold", "2026"):
        cs = H.base(bn).close_s
        rs.append((cs / cs.shift(1) - 1).dropna())
    r = pd.concat(rs)
    return r[~r.index.duplicated(keep="last")].sort_index()


def raw_signals(bn: str) -> dict:
    b = H.base(bn)
    soxx = H.px("SOXX")
    t1 = (soxx < soxx.rolling(SMA_N).mean()).astype(float)
    vr = (H.px("^VIX") / H.px("^VIX3M")).dropna()
    t2 = (vr > VIX_TH).astype(float)
    cs = b.close_s
    t3 = ((cs / cs.cummax() - 1) <= DD_TH).astype(float)
    rch = shadow_chain()
    vol = rch.rolling(VOL_N).std() * np.sqrt(252)
    t4 = (vol > VOL_TH).astype(float)[vol.notna()]
    t5 = pd.Series(1.0, index=b.days)
    return {"T1_SOXX<SMA50": t1, "T2_VIX/VIX3M>1": t2, "T3_그림자낙폭<=-10%": t3, "T4_그림자변동성20>60%": t4,
            "T5_상시": t5}


def betas() -> dict:
    """그림자 60일 이동 β(최소 20일) 대 SOXX·QQQ·SPY 종가 수익. index = 그 β를 아는 날(t 종가)."""
    rs = shadow_chain()
    out = {}
    for u in ("SOXX", "QQQ", "SPY"):
        ru = H.px(u).pct_change()
        d = pd.concat([rs, ru], axis=1, join="inner").dropna()
        d.columns = ["s", "u"]
        cov = d["s"].rolling(BETA_N, min_periods=BETA_MINP).cov(d["u"])
        var = d["u"].rolling(BETA_N, min_periods=BETA_MINP).var()
        out[u] = (cov / var).dropna()
    return out


# ----------------------------------------------------------------------------- 보고용 보조
def episodes(on: pd.Series) -> list:
    """on(거래일 k 시가 기준 0/1) → [(시작일, 마지막 보유일)]"""
    eps, start, prev = [], None, None
    for d, v in on.items():
        if v > 0.5 and start is None:
            start = d
        if v <= 0.5 and start is not None:
            eps.append((start, prev))
            start = None
        prev = d
    if start is not None:
        eps.append((start, prev))
    return eps


def seg_ret(r: pd.Series, lo: str, hi: str) -> float:
    x = r[(r.index >= lo) & (r.index <= hi)]
    return float((1 + x).prod() - 1)


def episode_table(b, on, r_var, lo, hi):
    rows = []
    for s, e in episodes(on):
        if e < lo or s > hi:
            continue
        rb, rv = seg_ret(b.ret, s, e), seg_ret(r_var, s, e)
        rows.append({"시작": s, "끝": e, "일수": int(((on.index >= s) & (on.index <= e)).sum()),
                     "기본": round(rb, 4), "변형": round(rv, 4), "차(변형-기본)": round(rv - rb, 4)})
    return rows


def trough_split(b, on, r_var, trough="2026-07-29"):
    """저점(7/29 종가 → 7/30 시가)을 품은 보호 구간을 '진입~저점'(아낀 손실)과 '저점 뒤~해제'(놓친 반등)로 나눈다."""
    out = []
    nxt = b.ret.index[b.ret.index > trough][0]
    for s, e in episodes(on):
        if s <= trough and e >= nxt:
            for lab, a, z in (("진입~저점", s, trough), ("저점 뒤~해제", nxt, e)):
                rb, rv = seg_ret(b.ret, a, z), seg_ret(r_var, a, z)
                out.append({"구간": lab, "시작": a, "끝": z, "기본": round(rb, 4), "변형": round(rv, 4),
                            "차": round(rv - rb, 4)})
    return out


def episode_hold_returns(bn, on, prod, q, weekly=False):
    """민감도: 계기가 켜질 때 주식 q를 팔아 인버스를 한 번 사고 꺼질 때까지 비중을 다시 맞추지 않는 경우(일별 비중 유지 대신).
    weekly=True면 7종목 주간 교체일마다 (1-q, q)로 다시 맞춘다."""
    b = H.base(bn)
    reb = set(b.rebalance_days()) if weekly else set()
    ir = H.open_ret(prod, b)
    o = on.reindex(b.ret.index).fillna(0.0)
    out, sv, iv, prev = [], 1.0, 0.0, 0.0
    for k in b.ret.index:
        tot = sv + iv
        if o[k] > 0.5 and (prev <= 0.5 or k in reb):
            sv, iv = (1 - q) * tot, q * tot
        elif o[k] <= 0.5 and prev > 0.5:
            sv, iv = tot, 0.0
        sv2, iv2 = sv * (1 + b.ret[k]), iv * (1 + ir[k])
        out.append((sv2 + iv2) / tot - 1)
        sv, iv, prev = sv2, iv2, o[k]
    return pd.Series(out, index=b.ret.index)


def soxs_check() -> dict:
    """2026 6/22 고점 → 7/29 저점(SOXX 종가 −29%) 구간 SOXS 실제 수익과 −3× 단순 배수 비교."""
    out = {}
    for lo, hi, lab in (("2026-06-22", "2026-07-29", "종가 6/22→7/29"),):
        sx, ss = H.px("SOXX"), H.px("SOXS")
        r_sx = sx[hi] / sx[lo] - 1
        r_ss = ss[hi] / ss[lo] - 1
        daily = ss.pct_change().loc[lo:hi].iloc[1:]
        dsx = sx.pct_change().loc[lo:hi].iloc[1:]
        out[lab] = {"SOXX": round(float(r_sx), 4), "SOXS": round(float(r_ss), 4), "-3×SOXX": round(float(-3 * r_sx), 4),
                    "일일재설정 손실(SOXS - (-3×))": round(float(r_ss + 3 * r_sx), 4),
                    "SOXS 일별 추적오차 최대(|SOXS+3·SOXX|)": round(float((daily + 3 * dsx).abs().max()), 4),
                    "거래일": int(len(daily))}
    # 시가→시가(헤지 체결 기준): 6/23 시가(6/22 종가 신호) → 7/30 시가
    so, xo = H.px("SOXS", "Open"), H.px("SOXX", "Open")
    out["시가 6/23→7/30"] = {"SOXX": round(float(xo["2026-07-30"] / xo["2026-06-23"] - 1), 4),
                            "SOXS": round(float(so["2026-07-30"] / so["2026-06-23"] - 1), 4)}
    # 7월 전체(6/30 종가→7/31 종가)
    sx, ss = H.px("SOXX"), H.px("SOXS")
    out["7월 6/30→7/31"] = {"SOXX": round(float(sx["2026-07-31"] / sx["2026-06-30"] - 1), 4),
                           "SOXS": round(float(ss["2026-07-31"] / ss["2026-06-30"] - 1), 4)}
    # 2026 연간(1/2→10/9 종가): 상시 보유 비용
    out["2026 1/2→10/9"] = {"SOXX": round(float(sx["2026-10-09"] / sx["2026-01-02"] - 1), 4),
                           "SOXS": round(float(ss["2026-10-09"] / ss["2026-01-02"] - 1), 4),
                           "SQQQ": round(float(H.px("SQQQ")["2026-10-09"] / H.px("SQQQ")["2026-01-02"] - 1), 4),
                           "QQQ": round(float(H.px("QQQ")["2026-10-09"] / H.px("QQQ")["2026-01-02"] - 1), 4)}
    return out


# ----------------------------------------------------------------------------- 실행
def main():
    sig = {bn: raw_signals(bn) for bn in BASES}
    bet = betas()
    on = {bn: {t: H.at_open(s, H.base(bn), fill=0.0) for t, s in sig[bn].items()} for bn in BASES}
    beta_open = {bn: {u: H.at_open(s, H.base(bn), fill=np.nan).fillna(BETA_FALLBACK).clip(lower=BETA_FLOOR)
                      for u, s in bet.items()} for bn in BASES}

    rows, paths = [], {}
    for mode, trig, prod, q in GRID:
        L, u = PRODUCTS[prod]
        ex, hd = {}, {}
        for bn in BASES:
            o = on[bn][trig]
            if mode == "헤지":
                ex[bn] = 1.0 - q * o
                hd[bn] = {prod: q * o}
            else:
                e_eq = ((1 - q) - q * L / beta_open[bn][u]).clip(0.0, 1.0)
                ex[bn] = (1.0 - o) + o * e_eq
        name = f"{mode}|{trig}|{prod}|q={q}"
        params = {"mode": mode, "trigger": trig, "product": prod, "q": q, "lev": -L, "underlying": u}
        res = H.evaluate(name, ex, hd if mode == "헤지" else None, params)
        rows.append(res)
        paths[name] = (ex, hd if mode == "헤지" else None)
    assert len(rows) == 170, len(rows)

    br = H.base_result()
    by = {r["name"]: r for r in rows}

    # 헤지 vs 현금 짝 비교
    pairs = []
    for trig in TRIGGERS:
        for prod in PRODUCTS:
            for q in qs(prod):
                h, c = by[f"헤지|{trig}|{prod}|q={q}"], by[f"현금|{trig}|{prod}|q={q}"]
                d = {"trigger": trig, "product": prod, "q": q}
                for bn, k in (("2026", "전체"), ("2026", "앞"), ("2026", "뒤"), ("2026", "AI쇼크"), ("hold", "2025"),
                              ("hold", "2024")):
                    d[f"{bn}_{k}_수익_헤지"] = round(h[bn][k]["수익"], 4)
                    d[f"{bn}_{k}_수익_현금"] = round(c[bn][k]["수익"], 4)
                    d[f"{bn}_{k}_낙폭_헤지"] = round(h[bn][k]["낙폭"], 4)
                    d[f"{bn}_{k}_낙폭_현금"] = round(c[bn][k]["낙폭"], 4)
                d["2026_평균노출_현금"] = round(c["2026"]["평균노출"], 3)
                pairs.append(d)
    pdf = pd.DataFrame(pairs)
    pair_sum = {}
    for bn, k in (("2026", "전체"), ("2026", "앞"), ("2026", "뒤"), ("2026", "AI쇼크"), ("hold", "2025"), ("hold", "2024")):
        dr = pdf[f"{bn}_{k}_수익_헤지"] - pdf[f"{bn}_{k}_수익_현금"]
        dd = pdf[f"{bn}_{k}_낙폭_헤지"] - pdf[f"{bn}_{k}_낙폭_현금"]
        pair_sum[f"{bn}_{k}"] = {"헤지 수익 우위 짝 수": int((dr > 0).sum()), "헤지 낙폭 얕은 짝 수": int((dd > 0).sum()),
                                 "수익차 중앙값": round(float(dr.median()), 4), "낙폭차 중앙값": round(float(dd.median()), 4),
                                 "짝 수": int(len(pdf))}
    by_prod = {}
    for prod in PRODUCTS:
        x = pdf[pdf["product"] == prod]
        by_prod[prod] = {f"{bn}_{k}": round(float((x[f"{bn}_{k}_수익_헤지"] - x[f"{bn}_{k}_수익_현금"]).median()), 4)
                         for bn, k in (("2026", "전체"), ("2026", "앞"), ("2026", "AI쇼크"), ("hold", "2025"))}

    # 헤지만·현금만 따로 앞 고르기
    hedge_rows = [r for r in rows if r["params"]["mode"] == "헤지"]
    cash_rows = [r for r in rows if r["params"]["mode"] == "현금"]
    sep = {}
    for lab, rs in (("헤지만", hedge_rows), ("현금만", cash_rows)):
        for kp in (0.9, 0.8, 0.7):
            s = H.select_front(rs, br, kp)
            sep[f"{lab}_keep{int(kp * 100)}"] = H.pick_row(f"{lab} keep{int(kp * 100)}", s)

    # 계기별 켜진 날(2026·보유)
    trig_days = {}
    for bn in BASES:
        b = H.base(bn)
        for t in TRIGGERS:
            o = on[bn][t].reindex(b.ret.index)
            trig_days[f"{bn}|{t}"] = {"켜진 날 비율": round(float(o.mean()), 3), "진입 횟수": len(episodes(o))}
    b26 = H.base("2026")
    for t in TRIGGERS:
        o = on["2026"][t].reindex(b26.ret.index)
        trig_days[f"2026|{t}"]["앞 켜진 비율"] = round(float(o[o.index <= "2026-05-29"].mean()), 3)
        trig_days[f"2026|{t}"]["AI쇼크 켜진 비율"] = round(float(o[(o.index >= "2026-06-22") & (o.index <= "2026-07-29")].mean()), 3)

    # 같은 앞 낙폭 한도에서 헤지·현금 중 앞 수익 최대(효율선 비교, 기술 통계)
    frontier = {}
    for thr in (-0.18, -0.16, -0.14, -0.12, -0.10):
        f = {}
        for lab, rs in (("헤지", hedge_rows), ("현금", cash_rows)):
            ok = [r for r in rs if r["2026"]["앞"]["낙폭"] >= thr]
            best = max(ok, key=lambda r: r["2026"]["앞"]["수익"]) if ok else None
            f[lab] = ({"name": best["name"], "앞수익": round(best["2026"]["앞"]["수익"], 4),
                       "앞낙폭": round(best["2026"]["앞"]["낙폭"], 4), "뒤수익": round(best["2026"]["뒤"]["수익"], 4),
                       "뒤낙폭": round(best["2026"]["뒤"]["낙폭"], 4), "2025수익": round(best["hold"]["2025"]["수익"], 4)}
                      if best else None)
        frontier[f"앞낙폭≥{thr}"] = f

    # 헤지 고른 것: '주식만 q 판 경우' 대비 인버스 다리의 몫 + 구간 보유(재조정 없음) 민감도
    decomp = {}
    hedge_picks = {s["name"] for s in sep.values() if s and s["name"].startswith("헤지")}
    for nm in sorted(hedge_picks):
        _, trig, prod, qq = nm.split("|")
        q = float(qq.split("=")[1])
        dd = {}
        for bn in BASES:
            b = H.base(bn)
            o = on[bn][trig]
            r_h = H.overlay(b, 1.0 - q * o, {prod: q * o})
            r_s = H.overlay(b, 1.0 - q * o)
            r_e = episode_hold_returns(bn, o, prod, q)
            r_w = episode_hold_returns(bn, o, prod, q, weekly=True)
            for k, (a, z) in b.periods.items():
                if k not in ("전체", "앞", "뒤", "AI쇼크", "7~9월", "2025", "관세", "24여름", "25가을"):
                    continue
                sh, ss, se, sw = H.stats(r_h, a, z), H.stats(r_s, a, z), H.stats(r_e, a, z), H.stats(r_w, a, z)
                dd[f"{bn}_{k}"] = {"헤지": round(sh["수익"], 4), "주식만q매도": round(ss["수익"], 4),
                                   "인버스다리몫": round(sh["수익"] - ss["수익"], 4), "구간보유": round(se["수익"], 4),
                                   "주간재조정": round(sw["수익"], 4), "주간재조정낙폭": round(sw["낙폭"], 4),
                                   "헤지낙폭": round(sh["낙폭"], 4), "매도낙폭": round(ss["낙폭"], 4), "구간보유낙폭": round(se["낙폭"], 4)}
            dd[f"{bn}_전체_스프레드0.1%"] = round(H.stats(H.overlay(b, 1.0 - q * o, {prod: q * o}, cost=0.001),
                                                        *b.periods["전체"])["수익"], 4)
        decomp[nm] = dd

    beta_info = {u: {"2026 평균": round(float(beta_open["2026"][u].mean()), 3),
                     "2026-06-22 값": round(float(beta_open["2026"][u]["2026-06-22"]), 3),
                     "보유 평균": round(float(beta_open["hold"][u].mean()), 3)} for u in ("SOXX", "QQQ", "SPY")}

    extra = {"soxs_check": soxs_check(), "pair_summary": pair_sum, "pair_median_by_product": by_prod,
             "separate_front_picks": sep, "trigger_days": trig_days, "beta": beta_info,
             "frontier_front": frontier, "hedge_decomp": decomp, "pairs": pairs}

    payload = H.summarize_family("F_hedge", rows, GRID_DESC, extra)

    # 7월·2~3월 행동: 앞 고르기 → keep80 → 사후 최선
    fp = payload["front_picks"]
    chosen_lab, chosen = None, None
    for kk in ("keep90", "keep80", "keep70"):
        if fp[kk]:
            chosen_lab, chosen = kk, fp[kk]["name"]
            break
    if chosen is None and payload["pareto_2026_expost"]:
        chosen_lab, chosen = "사후 최선", payload["pareto_2026_expost"][0]["name"]
    behav = {}
    targets = [(chosen_lab, chosen)] if chosen else []
    # 짝(현금/헤지)도 같이
    if chosen:
        m, t, p, qq = chosen.split("|")
        other = ("현금" if m == "헤지" else "헤지") + f"|{t}|{p}|{qq}"
        targets.append(("짝", other))
    for s in sep.values():
        if s and s["name"] not in [x[1] for x in targets]:
            targets.append((s["label"], s["name"]))
    for lab, nm in targets:
        ex, hd = paths[nm]
        e26 = ex["2026"]
        r26 = H.overlay(b26, e26, hd["2026"] if hd else None)
        o_on = (e26 < 0.999).astype(float).reindex(b26.ret.index)
        behav[f"{lab}:{nm}"] = {
            "trace_jun_sep": H.trace(e26),
            "trace_feb_apr": H.trace(e26, "2026-01-26", "2026-04-15"),
            "eps_jun_sep": episode_table(b26, o_on, r26, "2026-06-15", "2026-09-30"),
            "eps_feb_mar": episode_table(b26, o_on, r26, "2026-01-26", "2026-04-15"),
            "trough_split": trough_split(b26, o_on, r26),
            "seg": {k: {"기본": round(seg_ret(b26.ret, a, z), 4), "변형": round(seg_ret(r26, a, z), 4)}
                    for k, (a, z) in {"2/1~4/15": ("2026-02-01", "2026-04-15"),
                                      "6/22~7/29": ("2026-06-22", "2026-07-29"),
                                      "7/30~9/30": ("2026-07-30", "2026-09-30"),
                                      "6/15~9/30": ("2026-06-15", "2026-09-30")}.items()}}
    payload["extra"]["behavior"] = behav
    payload["extra"]["chosen"] = {"label": chosen_lab, "name": chosen}
    H.save("F_hedge", payload)

    print("\n[헤지 vs 현금 짝 요약]")
    for k, v in pair_sum.items():
        print(" ", k, v)
    print("[상품별 수익차 중앙값(헤지-현금)]")
    for k, v in by_prod.items():
        print(" ", k, v)
    print("[헤지만/현금만 앞 고르기]")
    for k, v in sep.items():
        print(" ", k, (v["name"], v.get("2026_전체_수익"), v.get("2026_전체_낙폭"), v.get("2026_앞_수익"),
                       v.get("2026_앞_낙폭"), v.get("2026_AI쇼크_수익"), v.get("hold_2025_수익")) if v else None)
    print("[SOXS 확인]", json.dumps(extra["soxs_check"], ensure_ascii=False))
    print("[β]", beta_info)
    print("[계기]", json.dumps(trig_days, ensure_ascii=False))
    print("[선택]", chosen_lab, chosen)
    for k, v in behav.items():
        print("\n##", k)
        print(" trace 6~9월", v["trace_jun_sep"])
        print(" trace 2~4월", v["trace_feb_apr"])
        print(" seg", v["seg"])
        for e in v["eps_jun_sep"]:
            print("   6~9월", e)
        for e in v["eps_feb_mar"]:
            print("   2~3월", e)
        for e in v["trough_split"]:
            print("   저점분할", e)
    print("\n[효율선: 앞 낙폭 한도별 앞 수익 최대]")
    for k, v in frontier.items():
        print(" ", k, v)
    print("[헤지 분해: 인버스 다리 몫·구간 보유 민감도]")
    for nm, dd in decomp.items():
        print(" ", nm)
        for k, v in dd.items():
            print("    ", k, v)


GRID_DESC = ("모드 2(헤지: 주식 q 팔고 인버스 q 매수 / 현금: 그림자 60일 β로 환산한 같은 순노출만큼 주식만 축소) × "
             "계기 5(T1 SOXX<SMA50, T2 VIX/VIX3M>1, T3 그림자 낙폭≤−10%, T4 그림자 20일 변동성>60%, T5 상시) × "
             "상품·크기 17(SOXS·SQQQ q∈{0.05,0.10,0.15,0.20}, QID·PSQ·SH q∈{0.1,0.2,0.3}) = 170")

if __name__ == "__main__":
    main()
