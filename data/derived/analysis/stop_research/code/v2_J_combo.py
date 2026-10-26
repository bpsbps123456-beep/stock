"""v2 갈래 J(2026-10-10): 고정 노출 기준선 대비 '타이밍 가치' 전수 정리 + 사전 선언 조합 8개 + 7월 쇼크 피해 금액.

1부  results/{A..H}.json의 모든 변형(rows)을 같은 평균 노출의 '고정 노출'과 비교한다.
     고정 노출 표 = 노출 0.50~1.00, 0.01 간격 51개를 harness.overlay로 한 번 계산(매일 같은 비율 유지).
     각 변형의 2026 평균 노출(주식 비중 평균, 헤지 다리는 빼고 잼)에 가장 가까운 칸(보간 없음)과 비교한다.
     비교 지표: 2026 전체·뒤·AI쇼크, 보유 2025·2024 수익(소수 4자리에서 엄격히 클 때만 '나음').
2부  사전 선언 조합 C1~C8(아래 정의 외에는 만들지 않음) → harness.evaluate(2026·보유) → summarize_family('J_combo').
3부  2026-06-22 종가 평가액 1억 원일 때 6/22→7/29 손실액, 7/29→9/30 회복액(종가→종가, USD 비율 그대로·환율 무시).

시점: 모든 신호는 t일 종가(그림자·SOXX·HYG/IEF)로 정하고 H.at_open으로 t+1 시가에 적용(앞 갈래와 같은 함수 재사용).
조합 정의(실행 전 고정)
  공통 '깊으면 keep': 그림자 252일 고점 대비 낙폭(G 갈래 signals().dd) ≤ −15%인 종가 → 다음 시가부터 노출 1.0·헤지 0,
     그림자 252일 신고가(낙폭 0) 종가 → 다음 시가부터 원래 규칙 재개. C1·C4·C7은 상태와 무관하게 켠다(G (iii) 방식).
     C6만 G 후보와 같게 '나가 있을 때만' 되산다(G (ii) 방식).
  C1  F T1(SOXX 종가<SMA50, 매일 판정) → 주식 0.8 + SOXS 0.2, 깊으면 keep 동안 헤지 0.
  C2  E HYG/IEF(Adj Close)<SMA50 → 노출 0.5(최소 5거래일 유지 뒤 신호 꺼지면 복귀) + F T1 → SOXS 0.15.
      둘 다 켜지면 주식 0.5 + SOXS 0.15, T1만이면 주식 0.85 + SOXS 0.15.
  C3  G P3: SOXX<SMA50 하향 돌파(다시 위로 올라가야 재무장) → 노출 0.5, 20거래일 뒤 복귀.
  C4  C3 + 깊으면 keep(G simulate no_sell_D=deep_buy_D=0.15).
  C5  주식 상시 0.9 + F T1이면 남는 0.1로 SOXS 0.1.
  C6  G P2 X10% (ii)D15%를 반만: 그림자 트레일링(진입 뒤 최고 종가) −10% → 0.5, 그림자>SMA20 복귀,
      나가 있는 동안 252일 낙폭 ≤ −15% → 1.0, 252일 신고가까지 keep.
  C7  E HYG/IEF<SMA50 → 0.5(최소 5일) + 깊으면 keep.
  C8  SOXX<SMA50(F T1)인 연속 구간의 처음 20거래일(종가 기준 0~19일째)만 SOXS 0.2(주식 0.8). 21일째부터 해제,
      SOXX가 SMA50 위로 올라갔다가 다시 내려가면 새 구간.
실행: cd code && PYTHONIOENCODING=utf-8 py v2_J_combo.py
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
import fam_E_macro as E  # noqa: E402
import fam_G_hold_reentry as G  # noqa: E402

FAMILY = "J_combo"
BASES = ("2026", "hold")
FAMS = ["A_portfolio_dd", "B_stock_stops", "C_vol_target", "D_trend_breadth", "E_macro", "F_hedge",
        "G_hold_reentry", "H_events_funds"]
FIX_GRID = [round(0.50 + 0.01 * i, 2) for i in range(51)]
CMP = [("2026", "전체"), ("2026", "뒤"), ("2026", "AI쇼크"), ("hold", "2025"), ("hold", "2024")]
FOUR = [("2026", "뒤"), ("2026", "AI쇼크"), ("hold", "2025"), ("hold", "2024")]
DEEP = 0.15
KRW = 100_000_000

GRID_DESC = ("사전 선언 조합 8개(C1~C8, 정의는 스크립트 머리말·md). 앞 갈래 A~H 결과를 본 뒤 만든 조합이라 선택편향이 크다. "
             "1부 비교 범위: A~H 변형 1,299개 × 고정 노출 51칸(0.50~1.00, 0.01 간격).")


# ============================================================================ 1부: 고정 노출 표와 전수 비교
def fixed_table() -> dict:
    out = {}
    for e in FIX_GRID:
        out[e] = H.evaluate(f"고정 {e:.2f}", {bn: pd.Series(float(e), index=H.base(bn).days) for bn in BASES})
    return out


def nearest(e: float) -> float:
    return min(max(round(round(e * 100) / 100, 2), 0.50), 1.00)


def part1(fix: dict):
    recs = []
    for fam in FAMS:
        d = json.loads((H.RESULTS / f"{fam}.json").read_text(encoding="utf-8"))
        for r in d["rows"]:
            e = float(r["2026_평균노출"])
            fe = nearest(e)
            fx = fix[fe]
            rec = {"family": fam, "name": r["name"], "e": round(e, 3), "fixed_e": fe, "out_of_range": e < 0.495}
            for bn, k in CMP:
                v = r[f"{bn}_{k}_수익"]
                f = round(fx[bn][k]["수익"], 4)
                rec[f"{bn}_{k}"] = v
                rec[f"{bn}_{k}_fixed"] = f
                rec[f"{bn}_{k}_diff"] = round(v - f, 4)
                rec[f"{bn}_{k}_win"] = v > f
            rec["2026_전체_낙폭"] = r["2026_전체_낙폭"]
            rec["2026_전체_낙폭_fixed"] = round(fx["2026"]["전체"]["낙폭"], 4)
            rec["dd_win"] = r["2026_전체_낙폭"] > rec["2026_전체_낙폭_fixed"]
            rec["four_win"] = all(rec[f"{bn}_{k}_win"] for bn, k in FOUR)
            rec["five_win"] = rec["four_win"] and rec["2026_전체_win"]
            recs.append(rec)
    return recs


def fam_table(recs: list, sub=None) -> list:
    rows = []
    groups = [(fam, [x for x in recs if x["family"] == fam]) for fam in FAMS] + [("합계", recs)]
    for fam, xs in groups:
        if sub:
            xs = [x for x in xs if sub(x)]
        n = len(xs)
        if not n:
            rows.append({"갈래": fam, "n": 0})
            continue
        row = {"갈래": fam, "n": n, "범위밖(e<0.495)": sum(x["out_of_range"] for x in xs)}
        for bn, k in CMP:
            w = sum(x[f"{bn}_{k}_win"] for x in xs)
            row[f"{bn}_{k}"] = [w, round(w / n, 3), round(float(np.median([x[f"{bn}_{k}_diff"] for x in xs])), 4)]
        w = sum(x["dd_win"] for x in xs)
        row["2026_전체_낙폭얕음"] = [w, round(w / n, 3)]
        row["네개모두"] = sum(x["four_win"] for x in xs)
        row["다섯모두"] = sum(x["five_win"] for x in xs)
        rows.append(row)
    return rows


# ============================================================================ 2부: 조합 신호
def t1_raw() -> pd.Series:
    """F 갈래 T1과 같은 식: SOXX 종가 < SMA50(종가일 index)."""
    soxx = H.px("SOXX")
    return (soxx < soxx.rolling(50).mean()).astype(float)


def t1_first20_raw(n: int = 20) -> pd.Series:
    """C8: SOXX<SMA50 연속 구간의 0~(n-1)번째 종가일만 1."""
    soxx = H.px("SOXX")
    below = (soxx < soxx.rolling(50).mean()).to_numpy()
    out, run = np.zeros(len(below)), 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        out[i] = 1.0 if (b and run <= n) else 0.0
    return pd.Series(out, index=soxx.index)


def hyg_raw() -> pd.Series:
    """E 갈래와 같은 식: HYG/IEF(Adj Close) < SMA50, SMA 정의 전은 NaN."""
    ratio = (H.px("HYG", "Adj Close") / H.px("IEF", "Adj Close")).dropna()
    sma = ratio.rolling(50).mean()
    return E._flag(ratio < sma, sma.notna())


def on_open(sig: pd.Series, b: H.Base) -> np.ndarray:
    return (H.at_open(sig, b, fill=0.0) > 0.5).to_numpy()


def deep_flags(bn: str):
    """그림자 252일 낙폭(G signals) → (깊음 ≤ −15%, 신고가) 시가 적용 배열."""
    b, sig = H.base(bn), G.signals(bn)
    deep = on_open((sig.dd <= -DEEP).astype(float), b)
    nh = on_open((sig.dd >= -1e-9).astype(float), b)
    return deep, nh


def keep_state(deep: np.ndarray, nh: np.ndarray) -> np.ndarray:
    """상태 무관 '깊으면 keep': 깊음이 보이면 켜고 252일 신고가가 보이면 끈다(시가 기준)."""
    out, keep = np.zeros(len(deep), bool), False
    for k in range(len(deep)):
        if keep and nh[k]:
            keep = False
        if not keep and deep[k]:
            keep = True
        out[k] = keep
    return out


def hyg_exposure(on: np.ndarray, low: float, keep: np.ndarray | None = None) -> np.ndarray:
    """E '최소5일' 상태 기계 + 선택적 keep(켜지면 노출 1, 줄임 상태 해제)."""
    e, reduced, held = np.ones(len(on)), False, 0
    for k in range(len(on)):
        if keep is not None and keep[k]:
            reduced, held = False, 0
            continue
        if not reduced:
            if on[k]:
                reduced, held = True, 0
        else:
            held += 1
            if held >= E.MIN_DAYS and not on[k]:
                reduced = False
        e[k] = low if reduced else 1.0
    return e


def g_half(bn: str, exit_kind, X, reentry, no_sell_D, deep_buy_D, low: float) -> pd.Series:
    """G 갈래 simulate 그대로(나가면 0) → 나간 몫을 low로 바꿔 at_open."""
    b = H.base(bn)
    e_close, _ = G.simulate(G.signals(bn), exit_kind, X, reentry, no_sell_D, deep_buy_D)
    return H.at_open(1.0 - (1.0 - low) * (1.0 - e_close), b, fill=1.0)


def build_paths() -> dict:
    """대표 후보(재현 확인용) + 조합 8개의 {이름: (노출 by base, 헤지 by base 또는 None, params)}."""
    raw_t1, raw_t1_20, raw_hyg = t1_raw(), t1_first20_raw(), hyg_raw()
    P = {}

    def add(name, ex, hd, params):
        P[name] = (ex, hd, params)

    ex, hd = {}, {}
    S = {}
    for bn in BASES:
        b = H.base(bn)
        idx = b.days
        t1 = on_open(raw_t1, b).astype(float)
        t20 = on_open(raw_t1_20, b).astype(float)
        hy = on_open(raw_hyg, b)
        deep, nh = deep_flags(bn)
        kp = keep_state(deep, nh)
        e_hyg_E = E.exposure(hy, 0.5, "최소5일", idx).to_numpy()
        e_hyg_mine = hyg_exposure(hy, 0.5)
        assert np.allclose(e_hyg_E, e_hyg_mine), "E 최소5일 재구현 불일치"
        S[bn] = dict(idx=idx, t1=t1, t20=t20, hy=hy, kp=kp)

    def ser(arr, bn):
        return pd.Series(np.asarray(arr, float), index=S[bn]["idx"])

    # --- 대표 후보(앞 갈래 재현)
    add("[F] 헤지|T1_SOXX<SMA50|SOXS|q=0.2",
        {bn: ser(1 - 0.2 * S[bn]["t1"], bn) for bn in BASES},
        {bn: {"SOXS": ser(0.2 * S[bn]["t1"], bn)} for bn in BASES}, {"ref": "F_hedge"})
    add("[E] HYG/IEF<SMA50 | 노출0.5 | 최소5일",
        {bn: ser(hyg_exposure(S[bn]["hy"], 0.5), bn) for bn in BASES}, None, {"ref": "E_macro"})
    add("[E] HYG/IEF<SMA50 | 노출0 | 최소5일",
        {bn: ser(hyg_exposure(S[bn]["hy"], 0.0), bn) for bn in BASES}, None, {"ref": "E_macro"})
    add("[G] P3 SOXX<SMA50 → time20",
        {bn: g_half(bn, "soxx_sma50", None, ("time", 20), None, None, 0.0) for bn in BASES}, None,
        {"ref": "G_hold_reentry"})
    add("[G] P2 X10% (ii)산다 D15%",
        {bn: g_half(bn, "trail", 0.10, ("sma", 20), None, 0.15, 0.0) for bn in BASES}, None,
        {"ref": "G_hold_reentry"})

    # --- 조합 8개(사전 선언)
    C = {}
    for bn in BASES:
        s = S[bn]
        h1 = s["t1"] * (~s["kp"])
        C.setdefault("C1", ({}, {}))
        C["C1"][0][bn] = ser(1 - 0.2 * h1, bn)
        C["C1"][1][bn] = {"SOXS": ser(0.2 * h1, bn)}
        e_h = hyg_exposure(s["hy"], 0.5)
        red = e_h < 0.999
        st = np.where(red, 0.5, np.where(s["t1"] > 0.5, 0.85, 1.0))
        C.setdefault("C2", ({}, {}))
        C["C2"][0][bn] = ser(st, bn)
        C["C2"][1][bn] = {"SOXS": ser(0.15 * s["t1"], bn)}
        C.setdefault("C3", ({}, None))
        C["C3"][0][bn] = g_half(bn, "soxx_sma50", None, ("time", 20), None, None, 0.5)
        C.setdefault("C4", ({}, None))
        C["C4"][0][bn] = g_half(bn, "soxx_sma50", None, ("time", 20), DEEP, DEEP, 0.5)
        C.setdefault("C5", ({}, {}))
        C["C5"][0][bn] = ser(np.full(len(s["idx"]), 0.9), bn)
        C["C5"][1][bn] = {"SOXS": ser(0.1 * s["t1"], bn)}
        C.setdefault("C6", ({}, None))
        C["C6"][0][bn] = g_half(bn, "trail", 0.10, ("sma", 20), None, DEEP, 0.5)
        C.setdefault("C7", ({}, None))
        C["C7"][0][bn] = ser(hyg_exposure(s["hy"], 0.5, s["kp"]), bn)
        C.setdefault("C8", ({}, {}))
        C["C8"][0][bn] = ser(1 - 0.2 * s["t20"], bn)
        C["C8"][1][bn] = {"SOXS": ser(0.2 * s["t20"], bn)}
    names = {
        "C1": "C1 SOXX<SMA50→SOXS20% + 깊으면(−15%) 헤지 해제",
        "C2": "C2 HYG/IEF→0.5(최소5일) + SOXX<SMA50→SOXS15%",
        "C3": "C3 SOXX<SMA50 하향 돌파→0.5, 20일 뒤 복귀",
        "C4": "C4 C3 + 깊으면(−15%) 1.0 keep",
        "C5": "C5 상시 0.9 + SOXX<SMA50→SOXS10%",
        "C6": "C6 그림자 −10%→0.5, −15%에 1.0 되사기 keep",
        "C7": "C7 HYG/IEF→0.5(최소5일) + 깊으면(−15%) 1.0 keep",
        "C8": "C8 SOXX<SMA50→SOXS20%, 20거래일 뒤 해제",
    }
    params = {
        "C1": {"hedge": "SOXS", "q": 0.2, "trigger": "SOXX<SMA50(매일)", "deep_keep": DEEP, "deep_mode": "상태 무관"},
        "C2": {"macro": "HYG/IEF<SMA50", "low": 0.5, "min_days": 5, "hedge": "SOXS", "q": 0.15,
               "trigger": "SOXX<SMA50(매일)"},
        "C3": {"exit": "SOXX<SMA50 하향 돌파(재무장)", "low": 0.5, "reentry": "time20"},
        "C4": {"exit": "SOXX<SMA50 하향 돌파(재무장)", "low": 0.5, "reentry": "time20", "deep_keep": DEEP,
               "deep_mode": "상태 무관(G iii)"},
        "C5": {"base_exposure": 0.9, "hedge": "SOXS", "q": 0.1, "trigger": "SOXX<SMA50(매일)"},
        "C6": {"exit": "그림자 트레일링 −10%", "low": 0.5, "reentry": "그림자>SMA20", "deep_buy": DEEP,
               "deep_mode": "나가 있을 때만(G ii)"},
        "C7": {"macro": "HYG/IEF<SMA50", "low": 0.5, "min_days": 5, "deep_keep": DEEP, "deep_mode": "상태 무관"},
        "C8": {"hedge": "SOXS", "q": 0.2, "trigger": "SOXX<SMA50 연속 구간 처음 20거래일"},
    }
    for c in sorted(C):
        exb, hdb = C[c]
        add(names[c], exb, hdb if hdb else None, params[c])
    return P


# ============================================================================ 3부: 종가 평가액
def close_values(bn: str, e: pd.Series, hedge: dict | None):
    """시가 경로(overlay) + 그날 장중(시가→종가) 비율로 종가 평가액. 기본 경로 장중 = close/open."""
    b = H.base(bn)
    r = H.overlay(b, e, hedge)
    vo = pd.Series(np.concatenate([[1.0], np.cumprod(1 + r.to_numpy())]), index=b.days)
    intr = pd.Series(np.asarray(b.close) / np.asarray(b.open), index=b.days) - 1
    g = e.reindex(b.days).astype(float) * intr
    for sym, h in (hedge or {}).items():
        hh = h.reindex(b.days).astype(float).fillna(0.0)
        o = H.px(sym, "Open").reindex(b.days).ffill()
        c = H.px(sym, "Close").reindex(b.days).ffill()
        g = g + hh * (c / o - 1)
    return vo, vo * (1 + g)


def damage(name, e, hd, a="2026-06-22", t="2026-07-29", z="2026-09-30"):
    _, vc = close_values("2026", e, hd)
    loss = float(vc[t] / vc[a] - 1)
    rec = float(vc[z] / vc[t] - 1)
    return {"name": name, "손실률": round(loss, 4), "손실액": int(round(KRW * loss)), "회복률": round(rec, 4),
            "회복액": int(round(KRW * (1 + loss) * rec)), "9/30평가액": int(round(KRW * (1 + loss) * (1 + rec))),
            "순손익": int(round(KRW * ((1 + loss) * (1 + rec) - 1)))}


# ============================================================================ 보고 보조
def trace2(e: pd.Series, hd: dict | None, lo="2026-06-15", hi="2026-09-30") -> list:
    df = pd.DataFrame({"e": e})
    if hd:
        for sym, h in hd.items():
            df[sym] = h
    df = df[(df.index >= lo) & (df.index <= hi)].round(3)
    ch = df[df.ne(df.shift()).any(axis=1)]
    return [(d, *[float(x) for x in row]) for d, row in ch.iterrows()]


def seg(r: pd.Series, lo, hi):
    x = r[(r.index >= lo) & (r.index <= hi)]
    return float((1 + x).prod() - 1)


def main():
    fix = fixed_table()
    print("고정 0.9 확인:", {k: round(v["수익"], 4) for k, v in fix[0.9]["2026"].items() if k in ("전체", "AI쇼크")},
          round(fix[0.9]["hold"]["2025"]["수익"], 4))
    recs = part1(fix)
    print("1부 변형 수", len(recs))
    t_all = fam_table(recs)
    t_mid = fam_table(recs, lambda x: 0.495 <= x["e"] < 0.98)
    four = [x for x in recs if x["four_win"]]
    for row in t_all:
        print(" ", row)
    print("네 개 모두 나은 변형", len(four))
    for x in four:
        print("   ", x["family"], x["name"], x["e"], [x[f"{bn}_{k}_diff"] for bn, k in CMP])

    # 2부
    P = build_paths()
    rows, ref_rows = [], {}
    for name, (ex, hd, params) in P.items():
        res = H.evaluate(name, ex, hd, params)
        if name.startswith("C"):
            rows.append(res)
        else:
            ref_rows[name] = res
    # 재현 확인
    stored = {}
    for fam in ("F_hedge", "E_macro", "G_hold_reentry"):
        d = json.loads((H.RESULTS / f"{fam}.json").read_text(encoding="utf-8"))
        stored.update({r["name"]: r for r in d["rows"]})
    repro = {}
    for name, r in ref_rows.items():
        key = name.split("] ", 1)[1]
        s = stored[key]
        repro[name] = {"2026_전체(재계산)": round(r["2026"]["전체"]["수익"], 4), "2026_전체(저장)": s["2026_전체_수익"],
                       "2025(재계산)": round(r["hold"]["2025"]["수익"], 4), "2025(저장)": s["hold_2025_수익"]}
        assert abs(round(r["2026"]["전체"]["수익"], 4) - s["2026_전체_수익"]) < 2e-4, (name, repro[name])
        assert abs(round(r["hold"]["2025"]["수익"], 4) - s["hold_2025_수익"]) < 2e-4, (name, repro[name])
    print("재현 확인", json.dumps(repro, ensure_ascii=False))

    # 조합별 고정 노출 비교·헤지 평균·7월 행동
    b26 = H.base("2026")
    combo_vs_fixed, behavior = {}, {}
    for r in rows + list(ref_rows.values()):
        nm = r["name"]
        ex, hd, _ = P[nm]
        e = r["2026"]["평균노출"]
        fe = nearest(e)
        fx = fix[fe]
        d = {"평균노출": round(e, 3), "고정칸": fe}
        if hd:
            d["SOXS평균"] = round(float(hd["2026"]["SOXS"].reindex(b26.ret.index).mean()), 3)
            d["SOXS켜진날비율"] = round(float((hd["2026"]["SOXS"].reindex(b26.ret.index) > 0).mean()), 3)
        for bn, k in CMP:
            d[f"{bn}_{k}"] = [round(r[bn][k]["수익"], 4), round(fx[bn][k]["수익"], 4),
                              round(r[bn][k]["수익"] - fx[bn][k]["수익"], 4)]
        d["2026_전체_낙폭"] = [round(r["2026"]["전체"]["낙폭"], 4), round(fx["2026"]["전체"]["낙폭"], 4)]
        d["2026_뒤_낙폭"] = [round(r["2026"]["뒤"]["낙폭"], 4), round(fx["2026"]["뒤"]["낙폭"], 4)]
        combo_vs_fixed[nm] = d
        r26 = H.overlay(b26, ex["2026"], hd["2026"] if hd else None)
        behavior[nm] = {"trace_6_15_9_30(날짜,주식,SOXS)": trace2(ex["2026"], hd["2026"] if hd else None),
                        "trace_2_1_4_15": trace2(ex["2026"], hd["2026"] if hd else None, "2026-02-01", "2026-04-15"),
                        "seg": {k: {"기본": round(seg(b26.ret, a, z), 4), "변형": round(seg(r26, a, z), 4)}
                                for k, (a, z) in {"2/1~4/15": ("2026-02-01", "2026-04-15"),
                                                  "6/22~7/29": ("2026-06-22", "2026-07-29"),
                                                  "7/30~9/30": ("2026-07-30", "2026-09-30")}.items()}}

    # 3부 피해 금액
    dmg = [damage("기본(7종목)", pd.Series(1.0, index=b26.days), None),
           damage("고정 0.9", pd.Series(0.9, index=b26.days), None)]
    for nm, (ex, hd, _) in P.items():
        dmg.append(damage(nm, ex["2026"], hd["2026"] if hd else None))
    # 기본 종가 재현(장중 비율 확인)
    _, vc_b = close_values("2026", pd.Series(1.0, index=b26.days), None)
    assert np.allclose(vc_b.to_numpy(), np.asarray(b26.close) / b26.open[0]), "종가 평가 재현 실패"
    for x in dmg:
        print(f"  {x['name']:<50} 손실 {x['손실률']:+.1%} {x['손실액']:>13,} 회복 {x['회복률']:+.1%} {x['회복액']:>13,} "
              f"9/30 {x['9/30평가액']:>13,}")

    # 깊으면 keep 구간(상태 무관 판정, 시가 적용일)
    keep_spans = {}
    for bn in BASES:
        b = H.base(bn)
        deep, nh = deep_flags(bn)
        kp = keep_state(deep, nh)
        sp, s0 = [], None
        for i, v in enumerate(kp):
            if v and s0 is None:
                s0 = i
            if not v and s0 is not None:
                sp.append((b.days[s0], b.days[i - 1]))
                s0 = None
        if s0 is not None:
            sp.append((b.days[s0], b.days[-1]))
        keep_spans[bn] = sp
    full = {}
    for r in [H.base_result()] + rows + list(ref_rows.values()):
        full[r["name"]] = {bn: {k: v for k, v in r[bn].items()} for bn in BASES}

    extra = {"keep_spans": keep_spans, "full_precision": full,
             "part1_fixed_table": {f"{e:.2f}": {f"{bn}_{k}": [round(fix[e][bn][k]["수익"], 4),
                                                                round(fix[e][bn][k]["낙폭"], 4)] for bn, k in CMP}
                                   for e in FIX_GRID},
             "part1_family_table": t_all, "part1_family_table_mid(0.495<=e<0.98)": t_mid,
             "part1_four_win": four, "part1_records": recs,
             "combo_vs_fixed": combo_vs_fixed, "reference_repro": repro,
             "reference_rows": {k: H.pick_row(k, v) for k, v in ref_rows.items()},
             "behavior": behavior, "july_damage_krw": dmg}
    payload = H.summarize_family(FAMILY, rows, GRID_DESC, extra)
    print("\n[조합 vs 같은 노출 고정]")
    for nm, d in combo_vs_fixed.items():
        print(" ", nm, d)
    return payload


if __name__ == "__main__":
    main()
