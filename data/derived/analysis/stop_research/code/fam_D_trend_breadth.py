"""갈래 D: 시장·섹터 추세와 AI 시장폭으로 7종목 노출 줄이기(2026-10-10).

7종목 규칙(기본 경로)은 그대로 두고, 전체 노출 비율 e만 바꾼다(빠진 몫은 현금, 이자 0).
신호는 모두 t일 종가(미국 장 마감, KST 05:00)까지의 값 → t 다음 거래일 시가 체결(H.at_open).
문턱은 고정 상수 또는 이동 창(SMA·N일 고점·N일 최소)만 쓴다. 전체 기간 통계로 문턱을 정하지 않는다.

격자(실행 전 고정, 총 156개)
  A  지수 추세(단순):  {QQQ,SOXX,SMH,RSP} 종가 < SMA N, N∈{10,20,50,100,200}, 빠질 때 노출∈{0,0.5}         40
  A' 지수 추세(버퍼):  같은 4×5×2, 나감 = 종가 < SMA×0.99 이틀 연속, 복귀 = 종가 > SMA 이틀 연속           40
  B  둘 다/하나라도:  QQQ·SOXX 모두 아래(both) / 하나라도 아래(any) × N 5개 × 노출 2개                   20
  C  급락 감지:       SOXX 5일 수익 ≤ {-8,-10,-12}% 또는 20일 고점 대비 ≤ {-10,-15}% (발생 시점) → 노출 {0,0.5},
                      복귀 = SOXX 종가 > SMA {10,20}                                                       20
  D  AI 시장폭:       50일선 위 비율 < {30,40,50}% (하향 돌파) × 노출 2 × 복귀 {비율 회복, 폭 급등 또는 비율 회복} 12
                      20일 신저가 비율 > {20,30}% (발생) × 노출 2 × 복귀 {20일선 위 > 50%, 폭 급등 또는 그것}       8
  E  상대 강도:       SOXX/QQQ 비율 < 그 SMA N, N∈{20,50,100} × 노출 {0,0.5}                               6
  F  조합(사전 고정): 추세(QQQ/SOXX × SMA20/50, 0.5) ∧ 급락(5일 ≤-10% 또는 20일 고점 대비 ≤-10%, 0, 복귀 SMA20)
                      의 최소 8 + 상대강도50(0.5)×SOXX<SMA50(0.5) 곱 1 + 폭 50일선<40%(0.5) ∧ 급락 5일 ≤-10%(0) 1   10
  폭 급등(thrust) = 20일선 위 비율이 최근 10거래일(오늘 포함) 안에 20% 미만이었고 오늘 60% 초과.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import harness as H

FAMILY = "D_trend_breadth"

# ----------------------------------------------------------------------------- 격자(사전 고정)
TREND_SYMS = ["QQQ", "SOXX", "SMH", "RSP"]
TREND_N = [10, 20, 50, 100, 200]
LOWS = [0.0, 0.5]
BUF_BELOW, BUF_DAYS = 0.01, 2
PAIR = ("QQQ", "SOXX")
PAIR_MODES = ["both", "any"]
CRASH_TRIG = [("ret5", -0.08), ("ret5", -0.10), ("ret5", -0.12), ("dd20", -0.10), ("dd20", -0.15)]
CRASH_SYM = "SOXX"
CRASH_REENTRY_N = [10, 20]
BR_P50 = [0.30, 0.40, 0.50]
BR_P50_RE = ["level", "thrust_or_level"]
BR_NL = [0.20, 0.30]
BR_NL_RE = ["p20>50", "thrust_or_p20>50"]
THRUST_LO, THRUST_HI, THRUST_WIN = 0.20, 0.60, 10
RS_N = [20, 50, 100]
COMBO_TREND = [("QQQ", 20), ("QQQ", 50), ("SOXX", 20), ("SOXX", 50)]
COMBO_CRASH = [("ret5", -0.10), ("dd20", -0.10)]

# 사후 참고(1차 실행 결과를 본 뒤 행동 분석만 추가한 변형. 고르기에 쓰지 않는다 — 선택편향 있음)
REF_NAMES = ["C SOXX ret5≤-12% →0.0 복귀>SMA20", "C SOXX ret5≤-10% →0.0 복귀>SMA20",
             "B QQQ·SOXX both<SMA10 →0.0", "A SMH<SMA10 →0.5", "E SOXX/QQQ<SMA20 →0.5"]

# 2~3월 하락 구간(지시문 기준 날짜)과 7월 쇼크
FEBMAR = [("2026-02-03", "2026-02-05"), ("2026-02-25", "2026-03-06"), ("2026-03-19", "2026-03-30")]
AISHOCK = ("2026-06-22", "2026-07-29")


# ----------------------------------------------------------------------------- 보조
def close(sym):
    return H.px(sym, "Close")


def machine(trig: pd.Series, reenter: pd.Series, low: float) -> pd.Series:
    """trig(나감)·reenter(복귀) 불리언(같은 index, t일 종가 기준). 나간 날은 복귀 판단을 하지 않고,
    나가 있는 동안 trig가 다시 켜지면 그대로 머문다. 값 = 그날 종가 뒤 정한 노출."""
    t = trig.fillna(False).to_numpy(bool)
    r = reenter.fillna(False).to_numpy(bool)
    out, vals = False, np.empty(len(t))
    for i in range(len(t)):
        if not out:
            out = bool(t[i])
        elif not t[i] and r[i]:
            out = False
        vals[i] = low if out else 1.0
    return pd.Series(vals, index=trig.index)


def onset(cond: pd.Series) -> pd.Series:
    c = cond.fillna(False).astype(bool)
    return c & ~c.shift(1, fill_value=False)


def valid(s: pd.Series, *deps) -> pd.Series:
    """창이 덜 찬 앞부분(NaN)은 신호 없음으로 잘라 at_open의 fill=1.0에 맡긴다."""
    m = pd.concat(deps, axis=1).notna().all(axis=1)
    return s[m]


def trend_sig(sym, n, low, buffer=False):
    c = close(sym)
    sma = c.rolling(n, min_periods=n).mean()
    if not buffer:
        below = c < sma
        return valid(machine(below, ~below, low), sma)
    b1 = c < sma * (1 - BUF_BELOW)
    a1 = c > sma
    trig = b1.copy()
    rein = a1.copy()
    for j in range(1, BUF_DAYS):
        trig &= b1.shift(j, fill_value=False)
        rein &= a1.shift(j, fill_value=False)
    return valid(machine(trig, rein, low), sma)


def pair_sig(mode, n, low):
    cs = [close(s) for s in PAIR]
    df = pd.concat(cs, axis=1).dropna()
    sm = df.rolling(n, min_periods=n).mean()
    below = df < sm
    cond = below.all(axis=1) if mode == "both" else below.any(axis=1)
    return valid(machine(cond, ~cond, low), sm.iloc[:, 0], sm.iloc[:, 1])


def crash_cond(kind, th):
    c = close(CRASH_SYM)
    if kind == "ret5":
        x = c / c.shift(5) - 1
    else:
        x = c / c.rolling(20, min_periods=20).max() - 1
    return (x <= th), x


def crash_sig(kind, th, low, re_n):
    c = close(CRASH_SYM)
    cond, x = crash_cond(kind, th)
    sma = c.rolling(re_n, min_periods=re_n).mean()
    return valid(machine(onset(cond[x.notna()]).reindex(c.index, fill_value=False), c > sma, low), sma, x)


_BR = {}


def breadth_series():
    if not _BR:
        a = H.breadth()
        a = a[a.index >= "2022-06-01"]
        sma50 = a.rolling(50, min_periods=50).mean()
        sma20 = a.rolling(20, min_periods=20).mean()
        min20 = a.rolling(20, min_periods=20).min()
        ok50, ok20 = sma50.notna() & a.notna(), sma20.notna() & a.notna()
        p50 = ((a > sma50) & ok50).sum(axis=1) / ok50.sum(axis=1).replace(0, np.nan)
        p20 = ((a > sma20) & ok20).sum(axis=1) / ok20.sum(axis=1).replace(0, np.nan)
        okm = min20.notna() & a.notna()
        nl = ((a <= min20) & okm).sum(axis=1) / okm.sum(axis=1).replace(0, np.nan)
        p20n = p20.where(ok20.sum(axis=1) >= 50)
        thrust = (p20n > THRUST_HI) & (p20n.rolling(THRUST_WIN, min_periods=THRUST_WIN).min() < THRUST_LO)
        _BR.update(p50=p50.where(ok50.sum(axis=1) >= 50), p20=p20n, nl=nl.where(okm.sum(axis=1) >= 50),
                   thrust=thrust)
    return _BR


def breadth_p50_sig(th, low, re):
    B = breadth_series()
    p50 = B["p50"].dropna()
    trig = onset(p50 < th)
    rein = p50 >= th
    if re == "thrust_or_level":
        rein = rein | B["thrust"].reindex(p50.index, fill_value=False)
    return machine(trig, rein, low)


def breadth_nl_sig(th, low, re):
    B = breadth_series()
    nl = B["nl"].dropna()
    p20 = B["p20"].reindex(nl.index)
    trig = onset(nl > th)
    rein = p20 > 0.50
    if re == "thrust_or_p20>50":
        rein = rein | B["thrust"].reindex(nl.index, fill_value=False)
    return machine(trig, rein, low)


def rs_sig(n, low):
    df = pd.concat([close("SOXX"), close("QQQ")], axis=1).dropna()
    ratio = df.iloc[:, 0] / df.iloc[:, 1]
    sma = ratio.rolling(n, min_periods=n).mean()
    below = ratio < sma
    return valid(machine(below, ~below, low), sma)


def combine(sigs, how="min"):
    df = pd.concat(sigs, axis=1).dropna()
    return df.min(axis=1) if how == "min" else df.prod(axis=1)


# ----------------------------------------------------------------------------- 격자 만들기
def build_grid():
    g = []   # (name, params, signal)
    for sym in TREND_SYMS:
        for n in TREND_N:
            for low in LOWS:
                g.append((f"A {sym}<SMA{n} →{low}", {"sub": "A", "sym": sym, "n": n, "low": low},
                          lambda sym=sym, n=n, low=low: trend_sig(sym, n, low)))
    for sym in TREND_SYMS:
        for n in TREND_N:
            for low in LOWS:
                g.append((f"A' {sym}<SMA{n}×0.99 2일 →{low}", {"sub": "A'", "sym": sym, "n": n, "low": low,
                                                             "buffer": [BUF_BELOW, BUF_DAYS]},
                          lambda sym=sym, n=n, low=low: trend_sig(sym, n, low, True)))
    for mode in PAIR_MODES:
        for n in TREND_N:
            for low in LOWS:
                g.append((f"B QQQ·SOXX {mode}<SMA{n} →{low}", {"sub": "B", "mode": mode, "n": n, "low": low},
                          lambda mode=mode, n=n, low=low: pair_sig(mode, n, low)))
    for kind, th in CRASH_TRIG:
        for low in LOWS:
            for rn in CRASH_REENTRY_N:
                g.append((f"C SOXX {kind}≤{th:.0%} →{low} 복귀>SMA{rn}",
                          {"sub": "C", "kind": kind, "th": th, "low": low, "re_sma": rn},
                          lambda kind=kind, th=th, low=low, rn=rn: crash_sig(kind, th, low, rn)))
    for th in BR_P50:
        for low in LOWS:
            for re in BR_P50_RE:
                g.append((f"D 폭50<{th:.0%} →{low} {re}", {"sub": "D-p50", "th": th, "low": low, "re": re},
                          lambda th=th, low=low, re=re: breadth_p50_sig(th, low, re)))
    for th in BR_NL:
        for low in LOWS:
            for re in BR_NL_RE:
                g.append((f"D 신저가20>{th:.0%} →{low} {re}", {"sub": "D-nl", "th": th, "low": low, "re": re},
                          lambda th=th, low=low, re=re: breadth_nl_sig(th, low, re)))
    for n in RS_N:
        for low in LOWS:
            g.append((f"E SOXX/QQQ<SMA{n} →{low}", {"sub": "E", "n": n, "low": low},
                      lambda n=n, low=low: rs_sig(n, low)))
    for sym, n in COMBO_TREND:
        for kind, th in COMBO_CRASH:
            g.append((f"F {sym}<SMA{n}(0.5)∧{kind}≤{th:.0%}(0)",
                      {"sub": "F", "trend": [sym, n, 0.5], "crash": [kind, th, 0.0, 20]},
                      lambda sym=sym, n=n, kind=kind, th=th: combine([trend_sig(sym, n, 0.5),
                                                                       crash_sig(kind, th, 0.0, 20)])))
    g.append(("F RS50(0.5)×SOXX<SMA50(0.5) 곱", {"sub": "F", "rs": [50, 0.5], "trend": ["SOXX", 50, 0.5], "how": "prod"},
              lambda: combine([rs_sig(50, 0.5), trend_sig("SOXX", 50, 0.5)], "prod")))
    g.append(("F 폭50<40%(0.5)∧ret5≤-10%(0)", {"sub": "F", "p50": [0.4, 0.5, "level"], "crash": ["ret5", -0.1, 0.0, 20]},
              lambda: combine([breadth_p50_sig(0.4, 0.5, "level"), crash_sig("ret5", -0.10, 0.0, 20)])))
    return g


# ----------------------------------------------------------------------------- 행동 분석
def spells(e: pd.Series, bn: str, lo: str, hi: str):
    """노출 < 1 연속 구간: (판 시가일, 돌아온 시가일, 최저 노출, 그동안 기본 수익, 변형 수익)."""
    b = H.base(bn)
    ee = e.reindex(b.ret.index)
    red = (ee < 0.999).to_numpy()
    idx = list(b.ret.index)
    out, i = [], 0
    while i < len(idx):
        if red[i]:
            j = i
            while j + 1 < len(idx) and red[j + 1]:
                j += 1
            if idx[j] >= lo and idx[i] <= hi:
                r = b.ret.iloc[i:j + 1]
                x = ee.iloc[i:j + 1]
                back = b.days[j + 1] if j + 1 < len(b.days) else "끝"
                out.append({"판날": idx[i], "돌아온날": back, "최저노출": float(x.min()),
                            "기본": float((1 + r).prod() - 1), "변형": float((1 + x * r).prod() - 1)})
            i = j + 1
        else:
            i += 1
    return out


def window_cmp(e: pd.Series, bn: str, lo: str, hi: str):
    b = H.base(bn)
    rb = b.ret
    rv = H.overlay(b, e)
    sb, sv = H.stats(rb, lo, hi), H.stats(rv, lo, hi)
    ee = e.reindex(rb.index)
    m = (ee.index >= lo) & (ee.index <= hi)
    x = ee[m]
    exits = int(((x.shift(1).fillna(ee[ee.index < lo].iloc[-1] if (ee.index < lo).any() else 1.0) >= 0.999)
                 & (x < 0.999)).sum())
    return {"구간": f"{lo}~{hi}", "기본": round(sb["수익"], 4), "변형": round(sv["수익"], 4),
            "기본낙폭": round(sb["낙폭"], 4), "변형낙폭": round(sv["낙폭"], 4), "빠짐": exits,
            "평균노출": round(float(x.mean()), 3)}


def plus_days(bn, d, n):
    days = H.base(bn).days
    i = next(k for k, x in enumerate(days) if x >= d)
    return days[min(i + n, len(days) - 2)]


def behavior(e26: pd.Series):
    out = {"trace": H.trace(e26), "spells_jun_sep": spells(e26, "2026", "2026-06-15", "2026-09-30"),
           "ai_shock": window_cmp(e26, "2026", *AISHOCK),
           "rebound_7/29~9/30": window_cmp(e26, "2026", "2026-07-29", "2026-09-30"),
           "spells_feb_apr": spells(e26, "2026", "2026-01-26", "2026-04-30"),
           "feb_apr_total": window_cmp(e26, "2026", "2026-01-26", "2026-04-30"),
           "febmar_drops": []}
    for lo, hi in FEBMAR:
        out["febmar_drops"].append({"하락": window_cmp(e26, "2026", lo, hi),
                                     "반등10일": window_cmp(e26, "2026", hi, plus_days("2026", hi, 10))})
    return out


def main():
    grid = build_grid()
    print("격자 수", len(grid))
    b26, bh = H.base("2026"), H.base("hold")
    rows, expo = [], {}
    for name, params, fn in grid:
        s = fn()
        e26, eh = H.at_open(s, b26), H.at_open(s, bh)
        rows.append(H.evaluate(name, {"2026": e26, "hold": eh}, None, params))
        expo[name] = (e26, eh)
    br = H.base_result()
    picks = {k: H.select_front(rows, br, kv) for k, kv in (("keep90", 0.9), ("keep80", 0.8), ("keep70", 0.7))}
    par = sorted(H.pareto_2026(rows, br, 0.90), key=lambda r: -r["2026"]["전체"]["수익"])
    focus = picks["keep90"] or picks["keep80"] or (par[0] if par else None)
    extra = {"focus": focus["name"] if focus else None, "behavior": {}}
    names = []
    for k in ("keep90", "keep80", "keep70"):
        if picks[k] and picks[k]["name"] not in names:
            names.append(picks[k]["name"])
    for r in par[:3]:
        if r["name"] not in names:
            names.append(r["name"])
    names += [n for n in REF_NAMES if n not in names]
    byname = {r["name"]: r for r in rows}
    for nm in names:
        extra["behavior"][nm] = behavior(expo[nm][0])
        extra["behavior"][nm]["hold_spells_2025"] = spells(expo[nm][1], "hold", "2025-01-01", "2025-12-31")
        extra["behavior"][nm]["hold_spells_2024"] = spells(expo[nm][1], "hold", "2024-06-01", "2024-08-31")
        extra["behavior"][nm]["spread0.1%_2026전체"] = byname[nm]["2026"]["전체_스프레드0.1%"]
        extra["behavior"][nm]["spread0.1%_hold전체"] = byname[nm]["hold"]["전체_스프레드0.1%"]
        extra["behavior"][nm]["exits_2026"] = byname[nm]["2026"]["빠짐횟수"]
        extra["behavior"][nm]["avg_exposure_2026"] = byname[nm]["2026"]["평균노출"]
    # 하위 갈래별 앞 고르기(참고): 각 하위 갈래 안에서 같은 규칙
    extra["sub_front_keep80"] = {}
    for sub in ("A", "A'", "B", "C", "D-p50", "D-nl", "E", "F"):
        rr = [r for r in rows if r["params"].get("sub") == sub]
        p = H.select_front(rr, br, 0.8)
        extra["sub_front_keep80"][sub] = H.pick_row(sub, p) if p else None
    payload = H.summarize_family(FAMILY, rows, GRID_DESC, extra)
    # 전 변형 짧은 표(정렬: 2026 앞 낙폭)
    print("\n하위 갈래별 앞 고르기(keep80)")
    for sub, v in extra["sub_front_keep80"].items():
        print(f"  {sub:6}", v["name"] if v else "없음")
    for nm in names:
        bh_ = extra["behavior"][nm]
        print("\n==", nm)
        print(" trace", bh_["trace"])
        print(" AI쇼크", bh_["ai_shock"], "\n 반등", bh_["rebound_7/29~9/30"])
        for s in bh_["spells_jun_sep"]:
            print("   spell", s)
        print(" 2~4월 합계", bh_["feb_apr_total"])
        for d in bh_["febmar_drops"]:
            print("   ", d["하락"], "|", d["반등10일"])
        print(" 스프레드0.1% 2026전체", round(bh_["spread0.1%_2026전체"], 4), "hold전체", round(bh_["spread0.1%_hold전체"], 4))
        print(" 2025 spells", len(bh_["hold_spells_2025"]))
        for s in bh_["hold_spells_2025"][:12]:
            print("   h25", s)
        for s in bh_["hold_spells_2024"][:8]:
            print("   h24", s)
    return payload


GRID_DESC = ("156개 = A 추세 단순 {QQQ,SOXX,SMH,RSP}×SMA{10,20,50,100,200}×노출{0,0.5} 40 + A' 같은 격자에 버퍼"
             "(SMA×0.99 이틀 연속 나감·SMA 위 이틀 연속 복귀) 40 + B QQQ·SOXX both/any×SMA 5×노출 2 20 + "
             "C SOXX 급락(5일 ≤-8/-10/-12%, 20일 고점 대비 ≤-10/-15%)×노출 2×복귀 SMA{10,20} 20 + "
             "D 시장폭 50일선 위<{30,40,50}%×노출 2×복귀{비율 회복, 폭 급등 또는 회복} 12 + 20일 신저가>{20,30}%×노출 2×"
             "복귀{20일선 위>50%, 폭 급등 또는 그것} 8 + E SOXX/QQQ<SMA{20,50,100}×노출 2 6 + F 사전 고정 조합 10")

if __name__ == "__main__":
    main()
